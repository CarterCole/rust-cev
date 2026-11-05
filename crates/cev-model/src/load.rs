//! From a model name to a running backend: resolve the files, recognise the
//! decision-model family, merge adapters, pick a device and dtype, and build
//! the engine that speaks the family's prompt.
//!
//! | what the repository holds | family |
//! |---|---|
//! | `config.json` with `model_type: qwen3` | cev's headless code readout on the packed Qwen3 engine |
//! | `config.json` with `model_type: qwen3_5` / `qwen3_5_text` | the same readout on the hybrid Qwen3.5 decoder |
//! | `decision_config.json` + `readout.safetensors` | d3 |
//! | `adapter_config.json` + `schema_config.json` | Nimble (LoRA on Qwen3.5, merged at load) |
//! | `adapter_config.json` + `head.pt` | Kev (LoRA on Qwen3.5-Base, merged at load, plus the pointer head) |

// The browser build has only the in-memory half of this module.
#![cfg_attr(target_arch = "wasm32", allow(unused_imports, dead_code))]

use crate::decoder::Decoder;
use crate::engine::{Engine, EngineOptions};
#[cfg(not(target_arch = "wasm32"))]
use crate::hub::{self, RepoFiles, WEIGHTS};
use crate::hybrid::{HybridEngine, Kind, PointerHead};
use crate::lora::Adapter;
use crate::lfm2::{self, Lfm2};
use crate::media::{LfmTileParams, QwenImageParams};
use crate::qwen35::{Config as Qwen35Config, Qwen35};
use crate::qwen35_vision::{Qwen35Vision, VisionConfig};
use crate::siglip2::{Projector, Siglip2, VisionConfig as SiglipConfig};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cev_core::prompt::cev as cev_prompt;
use cev_core::{Backend, Family, PromptFormat};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;

pub fn pick_device(s: &str) -> Result<Device> {
    Ok(match s {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        "cuda" => Device::new_cuda(0)?,
        "auto" => {
            if candle_core::utils::metal_is_available() {
                Device::new_metal(0)?
            } else if candle_core::utils::cuda_is_available() {
                Device::new_cuda(0)?
            } else {
                Device::Cpu
            }
        }
        other => bail!("unknown device `{other}`"),
    })
}

/// f32 is exact (matches transformers) and on Metal costs little for small
/// models; bigger checkpoints default to bf16 to halve memory.
pub fn pick_dtype(s: &str, device: &Device, weight_bytes: u64) -> Result<DType> {
    Ok(match s {
        "auto" if device.is_cuda() => DType::BF16,
        "auto" if device.is_metal() && weight_bytes > 4 << 30 => DType::BF16,
        "auto" | "f32" => DType::F32,
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        other => bail!("unknown dtype `{other}`"),
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn weight_bytes(paths: &[std::path::PathBuf]) -> u64 {
    paths.iter().filter_map(|w| std::fs::metadata(w).ok()).map(|m| m.len()).sum()
}

#[cfg(not(target_arch = "wasm32"))]
/// `cev-<repo name>` unless the caller named the model.
fn backend_name(opts: &EngineOptions, files: &RepoFiles) -> String {
    opts.name.clone().unwrap_or_else(|| format!("cev-{}", short_name(&files.repo)))
}

fn short_name(model: &str) -> String {
    model.trim_end_matches('/').rsplit('/').next().unwrap_or("model").split('@').next().unwrap_or("model").to_lowercase()
}

#[cfg(not(target_arch = "wasm32"))]
/// What a repository turned out to be.
#[derive(Debug)]
pub enum Detected {
    /// A plain backbone (`model_type` as in its config).
    Backbone { files: RepoFiles, model_type: String },
    D3 { files: RepoFiles },
    Nimble { adapter: RepoFiles, base: RepoFiles },
    Kev { adapter: RepoFiles, base: RepoFiles },
}

#[derive(Deserialize)]
struct SchemaConfig {
    model: String,
    #[serde(default)]
    revision: Option<String>,
    #[serde(default)]
    candidate_codes: Vec<String>,
    #[serde(default)]
    candidate_token_ids: Vec<u32>,
}

#[derive(Deserialize)]
pub struct DecisionConfig {
    pub codes: Vec<String>,
    pub token_ids: Vec<u32>,
    #[serde(default = "one")]
    pub temperature: f32,
    #[serde(default)]
    pub attention_mode: Option<String>,
    #[serde(default)]
    pub pooling: Option<String>,
    #[serde(default)]
    pub format_id: Option<String>,
    #[serde(default)]
    pub max_length: Option<usize>,
}

fn one() -> f32 {
    1.0
}

#[cfg(not(target_arch = "wasm32"))]
/// Resolve `model` (alias, repo id or directory) and work out its family.
pub fn detect(model: &str) -> Result<Detected> {
    let model = hub::alias(model);
    let probe = hub::repo(&model, &[], &["config.json", "adapter_config.json", "schema_config.json", "decision_config.json"])?;
    if probe.has("decision_config.json") {
        let files = hub::repo(&model, &["config.json", "tokenizer.json", "decision_config.json", "readout.safetensors", WEIGHTS], &["chat_template.jinja", "tokenizer_config.json", "preprocessor_config.json"])?;
        return Ok(Detected::D3 { files });
    }
    if probe.has("adapter_config.json") {
        let adapter = hub::repo(&model, &["adapter_config.json", "adapter_model.safetensors"], &["schema_config.json", "head.pt", "tokenizer.json", "tokenizer_config.json", "provenance.json"])?;
        if adapter.has("schema_config.json") {
            let sc: SchemaConfig = serde_json::from_slice(&adapter.read("schema_config.json")?)?;
            let base_id = match &sc.revision {
                Some(r) => format!("{}@{r}", sc.model),
                None => sc.model.clone(),
            };
            let base = hub::backbone(&base_id)?;
            return Ok(Detected::Nimble { adapter, base });
        }
        if adapter.has("head.pt") {
            let cfg = crate::lora::AdapterConfig::load(&adapter.dir)?;
            let base_name = cfg.base_model_name_or_path.clone().context("adapter_config.json has no base_model_name_or_path")?;
            let meta = kev_meta(&adapter.path("head.pt"))?;
            let base_id = match &meta.base_revision {
                Some(r) => format!("{base_name}@{r}"),
                None => base_name,
            };
            let base = hub::backbone(&base_id)?;
            return Ok(Detected::Kev { adapter, base });
        }
        bail!("{model}: a LoRA adapter without schema_config.json (Nimble) or head.pt (Kev); cev cannot tell how to read its answers");
    }
    let files = hub::backbone(&model)?;
    let cfg: serde_json::Value = serde_json::from_slice(&files.read("config.json")?)?;
    let model_type = cfg["model_type"].as_str().unwrap_or_default().to_string();
    Ok(Detected::Backbone { files, model_type })
}

/// Contents of Kev's `head.pt` besides the head weights.
#[derive(Debug, Default)]
pub struct KevMeta {
    pub base_revision: Option<String>,
    pub temperature: f32,
    pub head_dim: usize,
}

#[cfg(not(target_arch = "wasm32"))]
/// Read the scalar fields of `head.pt` (a torch pickle inside a zip).
pub fn kev_meta(path: &std::path::Path) -> Result<KevMeta> {
    let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
    Ok(kev_head(&bytes)?.0)
}

/// Kev's `head.pt` from its bytes: the scalar fields and the pointer head's
/// tensors by name (`q.weight`, ...), on the CPU.
pub fn kev_head(bytes: &[u8]) -> Result<(KevMeta, HashMap<String, Tensor>)> {
    use candle_core::pickle::{Object, Stack};
    use std::io::Read;
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).context("head.pt is not a torch zip")?;
    let names: Vec<String> = zip.file_names().map(String::from).collect();
    let pkl = names.iter().find(|n| n.ends_with("data.pkl")).context("head.pt: no data.pkl")?;
    let dir = std::path::PathBuf::from(pkl.strip_suffix(".pkl").expect("ends with .pkl"));
    let obj = {
        let mut reader = std::io::BufReader::new(zip.by_name(pkl)?);
        let mut stack = Stack::empty();
        stack.read_loop(&mut reader)?;
        stack.finalize()?
    };
    let mut meta = KevMeta { base_revision: None, temperature: 1.0, head_dim: 256 };
    let mut infos = Vec::new();
    if let Object::Dict(kv) = obj {
        for (k, v) in kv {
            let Object::Unicode(k) = k else { continue };
            match (k.as_str(), v) {
                ("temperature", Object::Float(t)) => meta.temperature = t as f32,
                ("temperature", Object::Int(t)) => meta.temperature = t as f32,
                ("head_dim", Object::Int(d)) => meta.head_dim = d as usize,
                ("base_revision", Object::Unicode(r)) => meta.base_revision = Some(r),
                ("head", Object::Dict(tensors)) => {
                    for (name, t) in tensors {
                        if let Some(info) = t.into_tensor_info(name, &dir)? {
                            infos.push(info);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut head = HashMap::new();
    for info in infos {
        if !info.layout.is_contiguous() {
            bail!("head.pt: `{}` is not stored contiguously", info.name);
        }
        let mut reader = zip.by_name(&info.path).with_context(|| format!("head.pt: no {}", info.path))?;
        std::io::copy(&mut reader.by_ref().take(info.layout.start_offset() as u64), &mut std::io::sink())?;
        let mut raw = vec![0u8; info.layout.shape().elem_count() * info.dtype.size_in_bytes()];
        reader.read_exact(&mut raw).with_context(|| format!("head.pt: {} is truncated", info.path))?;
        let t = Tensor::from_raw_buffer(&raw, info.dtype, info.layout.dims(), &Device::Cpu)?;
        head.insert(info.name, t);
    }
    Ok((meta, head))
}

/// Where a Qwen3.5 checkpoint keeps its text model.
fn text_root(names: &[String]) -> Result<(&'static str, Option<&'static str>)> {
    let has = |p: &str| names.iter().any(|n| n.starts_with(p));
    let root = if has("model.language_model.layers.") {
        "model.language_model"
    } else if has("language_model.layers.") {
        "language_model"
    } else if has("model.layers.") {
        "model"
    } else {
        bail!("no Qwen3.5 text layers in the checkpoint");
    };
    Ok((root, has("lm_head.").then_some("lm_head")))
}

#[cfg(not(target_arch = "wasm32"))]
fn load_tokenizer(files: &RepoFiles) -> Result<Tokenizer> {
    Tokenizer::from_file(files.path("tokenizer.json")).map_err(anyhow::Error::msg)
}

#[cfg(not(target_arch = "wasm32"))]
/// Build a Qwen3.5 decoder from `base` (optionally with a merged adapter),
/// and its vision tower when the checkpoint has one.
pub fn qwen35(base: &RepoFiles, adapter: Option<&Adapter>, opts: &EngineOptions, device: &Device) -> Result<(Qwen35, Qwen35Config, Option<Qwen35Vision>)> {
    let weights = base.weights()?;
    let (cfg, text) = Qwen35Config::parse(&base.read("config.json")?)?;
    let dtype = pick_dtype(&opts.dtype, device, weight_bytes(&weights))?;
    tracing::info!(dir = %base.dir.display(), ?device, ?dtype, layers = text.num_hidden_layers, "loading Qwen3.5 weights");
    let names: Vec<String> = {
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&weights)? };
        st.tensors().into_iter().map(|(n, _)| n).collect()
    };
    let (root, _) = text_root(&names)?;
    let vb = match adapter {
        Some(a) => VarBuilder::from_tensors(crate::lora::merged_tensors(&weights, dtype, device, a, root)?, dtype, device),
        None => unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device)? },
    };
    qwen35_from(cfg, text, &names, vb, opts.max_context, dtype)
}

/// The decoder (and vision tower, if any) from a `VarBuilder` over a
/// Qwen3.5 checkpoint whose tensor names are `names`.
fn qwen35_from(cfg: Qwen35Config, text: crate::qwen35::TextConfig, names: &[String], vb: VarBuilder, max_context: usize, embed_dtype: DType) -> Result<(Qwen35, Qwen35Config, Option<Qwen35Vision>)> {
    let (root, head) = text_root(names)?;
    let model = Qwen35::with_embed_dtype(text, vb.pp(root), head.map(|h| vb.pp(h)), max_context, embed_dtype)?;
    let visual_root = ["model.visual", "visual"].into_iter().find(|r| names.iter().any(|n| n.starts_with(&format!("{r}.blocks."))));
    let vision = match (visual_root, &cfg.vision_config) {
        (Some(r), Some(vc)) => {
            let vc: VisionConfig = serde_json::from_value(vc.clone()).context("vision_config")?;
            Some(Qwen35Vision::new(vc, vb.pp(r))?)
        }
        _ => None,
    };
    Ok((model, cfg, vision))
}

/// Codes that are one token on their own and stay that token right after the
/// prompt's answer slot (`probe` ends where the answer starts).
pub fn one_token_codes<'a>(tok: &Tokenizer, probe: &str, candidates: impl Iterator<Item = &'a str>, max: usize) -> Result<(Vec<String>, Vec<u32>)> {
    let enc = |s: &str| -> Result<Vec<u32>> { Ok(tok.encode(s, false).map_err(anyhow::Error::msg)?.get_ids().to_vec()) };
    let base = enc(probe)?;
    let (mut codes, mut ids) = (Vec::new(), Vec::new());
    for code in candidates {
        if codes.len() == max {
            break;
        }
        let one = enc(code)?;
        if one.len() != 1 || ids.contains(&one[0]) {
            continue;
        }
        let joined = enc(&format!("{probe}{code}"))?;
        if joined.len() == base.len() + 1 && joined[..base.len()] == base[..] && joined[base.len()] == one[0] {
            codes.push(code.to_string());
            ids.push(one[0]);
        }
    }
    Ok((codes, ids))
}

/// Nimble's 255 candidate codes: A..Z, then the tokenizer's upper-case ASCII
/// vocabulary entries of two or three letters that are one token, by length
/// then alphabetically (`extended_schema.extended_codes`).
pub fn nimble_codes(tok: &Tokenizer) -> Vec<String> {
    let mut extra: Vec<String> = tok
        .get_vocab(false)
        .into_keys()
        .filter(|s| s.is_ascii() && (2..=3).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_uppercase()))
        .filter(|s| tok.encode(s.as_str(), false).map(|e| e.len() == 1).unwrap_or(false))
        .collect();
    extra.sort_by(|a, b| (a.len(), a).cmp(&(b.len(), b)));
    let mut codes: Vec<String> = (b'A'..=b'Z').map(|c| (c as char).to_string()).collect();
    codes.extend(extra);
    codes.truncate(255);
    codes
}

#[cfg(not(target_arch = "wasm32"))]
/// Load any supported model as a backend.
pub fn load(opts: EngineOptions) -> Result<Arc<dyn Backend>> {
    let detected = detect(&opts.model)?;
    let device = pick_device(&opts.device)?;
    let name = |_family: &str, files: &RepoFiles| backend_name(&opts, files);
    match detected {
        Detected::Backbone { files, model_type } if model_type == "qwen3" => {
            let id = name("cev", &files);
            let engine = Engine::load(EngineOptions { model: files.dir.display().to_string(), name: Some(id), ..opts })?;
            Ok(Arc::new(engine))
        }
        Detected::Backbone { files, model_type } if model_type == "qwen3_5" || model_type == "qwen3_5_text" => {
            let tokenizer = load_tokenizer(&files)?;
            let (model, _cfg, vision) = qwen35(&files, None, &opts, &device)?;
            let format = PromptFormat { think_stub: tokenizer.token_to_id("<think>").is_some() };
            let (codes, ids) = one_token_codes(&tokenizer, &format.probe(), cev_prompt::candidate_codes().collect::<Vec<_>>().iter().map(String::as_str), cev_core::compile::MAX_OPTIONS)?;
            if codes.len() < 26 {
                bail!("tokenizer only supports {} single-token answer codes", codes.len());
            }
            let head = model.head_rows(&ids)?;
            let kind = Kind::Codes { family: Family::Cev, format: Some(format), codes, head };
            let id = name("cev", &files);
            let mut engine = HybridEngine::new(Box::new(model), tokenizer, kind, id, files.repo.clone(), opts)?;
            if let Some(v) = vision {
                engine = engine.with_vision(v, QwenImageParams::default())?;
            }
            engine.warmup()?;
            Ok(Arc::new(engine))
        }
        Detected::Backbone { files, model_type } if model_type == "lfm2_vl" || model_type == "lfm2" => {
            let tokenizer = load_tokenizer(&files)?;
            let raw: serde_json::Value = serde_json::from_slice(&files.read("config.json")?)?;
            let text_cfg: lfm2::Config = serde_json::from_value(raw.get("text_config").cloned().unwrap_or_else(|| raw.clone())).context("LFM2 text config")?;
            let weights = files.weights()?;
            let dtype = pick_dtype(&opts.dtype, &device, weight_bytes(&weights))?;
            let (names, w1_rows) = {
                let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&weights)? };
                let names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
                let w1 = names.iter().find(|n| n.ends_with("layers.0.feed_forward.w1.weight")).and_then(|n| st.get(n).ok()).map(|t| t.shape()[0]);
                (names, w1)
            };
            let root = ["model.language_model", "language_model", "model"].into_iter().find(|r| names.iter().any(|n| n.starts_with(&format!("{r}.layers.0.")))).context("no LFM2 layers in the checkpoint")?;
            tracing::info!(dir = %files.dir.display(), ?device, ?dtype, layers = text_cfg.num_hidden_layers, "loading LFM2 weights");
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, &device)? };
            let adjust = w1_rows.is_some_and(|r| lfm2::ffn_adjust_from_shape(&text_cfg, r)) || text_cfg.block_auto_adjust_ff_dim;
            let model = Lfm2::new(text_cfg, vb.pp(root), opts.max_context, adjust)?;
            let bos = tokenizer.id_to_token(tokenizer.token_to_id("<|startoftext|>").context("tokenizer has no <|startoftext|>")?).unwrap_or_else(|| "<|startoftext|>".into());
            let kind = Kind::Groups { lm_head: model.lm_head().clone(), bos };
            let id = name("d1", &files);
            let mut engine = HybridEngine::new(Box::new(model), tokenizer, kind, id, files.repo.clone(), opts)?;
            if model_type == "lfm2_vl" {
                let vc: SiglipConfig = serde_json::from_value(raw["vision_config"].clone()).context("vision_config")?;
                let vroot = ["model.vision_tower.vision_model", "vision_tower.vision_model"].into_iter().find(|r| names.iter().any(|n| n.starts_with(&format!("{r}.embeddings.")))).context("no SigLIP2 tower in the checkpoint")?;
                let proot = ["model.multi_modal_projector", "multi_modal_projector"].into_iter().find(|r| names.iter().any(|n| n.starts_with(&format!("{r}.linear_1")))).context("no projector in the checkpoint")?;
                let g = |k: &str, d: u64| raw[k].as_u64().unwrap_or(d);
                let factor = g("downsample_factor", 2) as usize;
                let hidden = g("projector_hidden_size", 2048) as usize;
                let out_dim = raw["text_config"]["hidden_size"].as_u64().unwrap_or(2048) as usize;
                let tower = Siglip2::new(vc.clone(), vb.pp(vroot))?;
                let projector = Projector::new(vc.hidden_size, hidden, out_dim, factor, vb.pp(proot))?;
                let params = LfmTileParams {
                    tile: g("tile_size", 512) as u32,
                    patch: g("encoder_patch_size", 16) as u32,
                    downsample: factor as u32,
                    min_image_tokens: g("min_image_tokens", 64),
                    max_image_tokens: g("max_image_tokens", 256),
                    max_pixels_tolerance: raw["max_pixels_tolerance"].as_f64().unwrap_or(2.0),
                    min_tiles: g("min_tiles", 2) as u32,
                    max_tiles: g("max_tiles", 10) as u32,
                    use_thumbnail: raw["use_thumbnail"].as_bool().unwrap_or(true),
                    bicubic: true,
                };
                engine = engine.with_lfm_vision(tower, projector, params)?;
            }
            engine.warmup()?;
            Ok(Arc::new(engine))
        }
        Detected::Backbone { files, model_type } if model_type == "d1_omni" => {
            let engine = crate::omni::OmniEngine::load(&files, opts, &device)?;
            Ok(Arc::new(engine))
        }
        Detected::Backbone { files, model_type } => bail!("{}: unsupported model_type `{model_type}` (cev runs qwen3, qwen3_5, lfm2_vl and d1_omni checkpoints, Nimble and Kev adapters, and d3 packages)", files.dir.display()),
        Detected::D3 { files } => {
            let tokenizer = load_tokenizer(&files)?;
            let dc: DecisionConfig = serde_json::from_slice(&files.read("decision_config.json")?).context("decision_config.json")?;
            if dc.format_id.as_deref().is_some_and(|f| f != "d3-code-readout-v1") {
                bail!("unsupported d3 format `{}`", dc.format_id.unwrap_or_default());
            }
            if dc.pooling.as_deref().is_some_and(|p| p != "last") {
                bail!("unsupported d3 pooling `{}`", dc.pooling.unwrap_or_default());
            }
            let causal = match dc.attention_mode.as_deref() {
                None | Some("causal") => true,
                Some("noncausal_full_attention") => false,
                Some(other) => bail!("unsupported d3 attention_mode `{other}`"),
            };
            let (model, _cfg, vision) = qwen35(&files, None, &opts, &device)?;
            let readout = candle_core::safetensors::load(files.path("readout.safetensors"), &device)?;
            let weight = readout.get("weight").context("readout.safetensors has no `weight`")?.to_dtype(DType::F32)?;
            if weight.dims() != [dc.codes.len(), model.cfg.hidden_size] {
                bail!("readout weight is {:?}, expected [{}, {}]", weight.dims(), dc.codes.len(), model.cfg.hidden_size);
            }
            // The codes are read by row of the readout, so only their count matters here;
            // check they still tokenize as the package says.
            for (c, &id) in dc.codes.iter().zip(&dc.token_ids).take(26) {
                if tokenizer.token_to_id(c) != Some(id) && tokenizer.encode(c.as_str(), false).map(|e| e.get_ids().to_vec()).unwrap_or_default() != [id] {
                    bail!("code `{c}` is not token {id} in this tokenizer");
                }
            }
            let kind = Kind::Matrix { codes: dc.codes, weight, temperature: dc.temperature, causal };
            let id = name("d3", &files);
            let mut opts = opts;
            if let Some(m) = dc.max_length {
                opts.max_context = opts.max_context.min(m);
            }
            let mut engine = HybridEngine::new(Box::new(model), tokenizer, kind, id, files.repo.clone(), opts)?;
            if let Some(v) = vision {
                engine = engine.with_vision(v, QwenImageParams::default())?;
            }
            engine.warmup()?;
            Ok(Arc::new(engine))
        }
        Detected::Nimble { adapter, base } => {
            let tokenizer = load_tokenizer(&base)?;
            let sc: SchemaConfig = serde_json::from_slice(&adapter.read("schema_config.json")?)?;
            let lora = Adapter::load(&adapter.dir, &Device::Cpu)?;
            let (model, _cfg, _vision) = qwen35(&base, Some(&lora), &opts, &device)?;
            let codes = if sc.candidate_codes.is_empty() { nimble_codes(&tokenizer) } else { sc.candidate_codes.clone() };
            let ids: Vec<u32> = if sc.candidate_token_ids.len() == codes.len() {
                sc.candidate_token_ids.clone()
            } else {
                codes.iter().map(|c| tokenizer.encode(c.as_str(), false).ok().filter(|e| e.len() == 1).map(|e| e.get_ids()[0]).with_context(|| format!("code `{c}` is not one token"))).collect::<Result<_>>()?
            };
            // The adapter's contract: our code list must be what it was trained with.
            let ours = nimble_codes(&tokenizer);
            if !sc.candidate_codes.is_empty() && ours[..ours.len().min(codes.len())] != codes[..ours.len().min(codes.len())] {
                bail!("the tokenizer's candidate codes differ from the adapter's schema_config.json");
            }
            let head = model.head_rows(&ids)?;
            let kind = Kind::Codes { family: Family::Nimble, format: None, codes, head };
            let id = name("nimble", &adapter);
            let engine = HybridEngine::new(Box::new(model), tokenizer, kind, id, adapter.repo.clone(), opts)?;
            engine.warmup()?;
            Ok(Arc::new(engine))
        }
        Detected::Kev { adapter, base } => {
            let tokenizer = load_tokenizer(&base)?;
            let lora = Adapter::load(&adapter.dir, &Device::Cpu)?;
            let (meta, head) = kev_head(&adapter.read("head.pt")?)?;
            let (model, _cfg, _vision) = qwen35(&base, Some(&lora), &opts, &device)?;
            let id = name("kev", &adapter);
            let engine = kev_engine(model, tokenizer, meta, head, id, adapter.repo.clone(), opts)?;
            engine.warmup()?;
            Ok(Arc::new(engine))
        }
    }
}

/// Kev: the merged decoder plus the pointer head from `head.pt`.
fn kev_engine(model: Qwen35, tokenizer: Tokenizer, meta: KevMeta, head: HashMap<String, Tensor>, id: String, repo: String, opts: EngineOptions) -> Result<HybridEngine> {
    let device = model.device().clone();
    let get = |n: &str| head.get(n).with_context(|| format!("head.pt has no `head.{n}`")).and_then(|t| Ok(t.to_device(&device)?.to_dtype(DType::F32)?));
    let head = PointerHead { q_w: get("q.weight")?, q_b: get("q.bias")?, k_w: get("k.weight")?, k_b: get("k.bias")?, temperature: meta.temperature };
    if head.q_w.dims() != [meta.head_dim, model.cfg.hidden_size] {
        bail!("pointer head is {:?}, expected [{}, {}]", head.q_w.dims(), meta.head_dim, model.cfg.hidden_size);
    }
    let tid = |t: &str| tokenizer.token_to_id(t).with_context(|| format!("tokenizer has no {t}"));
    let kind = Kind::Pointer { head, decide_id: tid(cev_core::prompt::kev::DECIDE)?, option_end_id: tid(cev_core::prompt::kev::OPTION_END)? };
    tracing::info!(temperature = meta.temperature, head_dim = meta.head_dim, "Kev pointer head");
    HybridEngine::new(Box::new(model), tokenizer, kind, id, repo, opts)
}

/// A model whose files are already in memory, as in the browser: small files
/// by name, and the weights streamed in as CPU tensors.
///
/// | files | backend |
/// |---|---|
/// | `adapter_config.json`, `adapter_model.safetensors`, `head.pt` + the base's `config.json`, `tokenizer.json` | Kev (the LoRA is merged here) |
/// | `config.json` with `model_type: d1_omni`, `tokenizer.json` | d1-omni |
/// | `config.json` with `model_type: qwen3`, `tokenizer.json`, optional `tokenizer_config.json` | cev's readout on Qwen3 |
///
/// Runs on the CPU in f32 (tensors may hold other dtypes where the backend
/// reads them that way: Qwen3.5's embedding table). No warmup: there are no
/// kernels to compile.
pub struct MemoryModel {
    /// Repository the files came from (the adapter's, for Kev).
    pub repo: String,
    pub files: HashMap<String, Vec<u8>>,
    pub tensors: HashMap<String, Tensor>,
}

/// The weight streamer for `kind` (`kev`, `d1-omni`, `qwen3`) in the
/// browser: f32, minus what the backend never reads. Kev's base checkpoint
/// carries a vision tower and a multi-token-prediction block it does not use,
/// and its 248k-row embedding table stays bf16 (1 GB saved of a 4 GB address
/// space).
pub fn stream_loader(kind: &str) -> crate::weights::StreamLoader {
    let l = crate::weights::StreamLoader::new(DType::F32);
    match kind {
        "kev" => l.skip("model.visual.").skip("mtp.").keep_dtype("embed_tokens.weight"),
        _ => l,
    }
}

pub fn from_memory(model: MemoryModel, opts: EngineOptions) -> Result<Arc<dyn Backend>> {
    let MemoryModel { repo, files, mut tensors } = model;
    let file = |n: &str| files.get(n).map(Vec::as_slice).with_context(|| format!("{repo}: {n} was not provided"));
    let tokenizer = || Tokenizer::from_bytes(file("tokenizer.json")?).map_err(anyhow::Error::msg);
    let id = opts.name.clone().unwrap_or_else(|| format!("cev-{}", short_name(&repo)));
    if files.contains_key("head.pt") {
        let lora = Adapter::from_bytes(file("adapter_config.json")?, file("adapter_model.safetensors")?)?;
        let (meta, head) = kev_head(file("head.pt")?)?;
        let (cfg, text) = Qwen35Config::parse(file("config.json")?)?;
        let names: Vec<String> = tensors.keys().cloned().collect();
        let (root, _) = text_root(&names)?;
        let embed_dtype = tensors.get(&format!("{root}.embed_tokens.weight")).map_or(DType::F32, Tensor::dtype);
        crate::lora::merge_in_place(&mut tensors, &lora, root)?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let (model, _, _) = qwen35_from(cfg, text, &names, vb, opts.max_context, embed_dtype)?;
        return Ok(Arc::new(kev_engine(model, tokenizer()?, meta, head, id, repo, opts)?));
    }
    let raw: serde_json::Value = serde_json::from_slice(file("config.json")?).context("config.json")?;
    match raw["model_type"].as_str().unwrap_or_default() {
        "d1_omni" => {
            let w1_rows = tensors.get("encoder.layers.0.feed_forward.w1.weight").map_or(0, |t| t.dims()[0]);
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
            Ok(Arc::new(crate::omni::OmniEngine::from_parts(file("config.json")?, tokenizer()?, vb, w1_rows, repo, opts)?))
        }
        "qwen3" => {
            let tc = files.get("tokenizer_config.json").map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default();
            Ok(Arc::new(Engine::from_parts(file("config.json")?, file("tokenizer.json")?, &tc, tensors, EngineOptions { name: Some(id), ..opts })?))
        }
        other => bail!("{repo}: model_type `{other}` cannot be loaded from memory (Kev, d1-omni and Qwen3 can)"),
    }
}
