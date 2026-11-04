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

use crate::engine::{Engine, EngineOptions};
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

fn weight_bytes(paths: &[std::path::PathBuf]) -> u64 {
    paths.iter().filter_map(|w| std::fs::metadata(w).ok()).map(|m| m.len()).sum()
}

/// `cev-<repo name>` unless the caller named the model.
fn backend_name(opts: &EngineOptions, files: &RepoFiles) -> String {
    opts.name.clone().unwrap_or_else(|| format!("cev-{}", short_name(&files.repo)))
}

fn short_name(model: &str) -> String {
    model.trim_end_matches('/').rsplit('/').next().unwrap_or("model").split('@').next().unwrap_or("model").to_lowercase()
}

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

/// Read the scalar fields of `head.pt` (a torch pickle inside a zip).
pub fn kev_meta(path: &std::path::Path) -> Result<KevMeta> {
    use candle_core::pickle::{Object, Stack};
    let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))?;
    let names: Vec<String> = zip.file_names().map(String::from).collect();
    let pkl = names.iter().find(|n| n.ends_with("data.pkl")).context("head.pt: no data.pkl")?;
    let mut reader = std::io::BufReader::new(zip.by_name(pkl)?);
    let mut stack = Stack::empty();
    stack.read_loop(&mut reader)?;
    let obj = stack.finalize()?;
    let mut meta = KevMeta { base_revision: None, temperature: 1.0, head_dim: 256 };
    if let Object::Dict(kv) = obj {
        for (k, v) in kv {
            let Object::Unicode(k) = k else { continue };
            match (k.as_str(), v) {
                ("temperature", Object::Float(t)) => meta.temperature = t as f32,
                ("temperature", Object::Int(t)) => meta.temperature = t as f32,
                ("head_dim", Object::Int(d)) => meta.head_dim = d as usize,
                ("base_revision", Object::Unicode(r)) => meta.base_revision = Some(r),
                _ => {}
            }
        }
    }
    Ok(meta)
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

fn load_tokenizer(files: &RepoFiles) -> Result<Tokenizer> {
    Tokenizer::from_file(files.path("tokenizer.json")).map_err(anyhow::Error::msg)
}

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
    let (root, head) = text_root(&names)?;
    let vb = match adapter {
        Some(a) => VarBuilder::from_tensors(crate::lora::merged_tensors(&weights, dtype, device, a, root)?, dtype, device),
        None => unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device)? },
    };
    let model = Qwen35::new(text, vb.pp(root), head.map(|h| vb.pp(h)), opts.max_context)?;
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
            let meta = kev_meta(&adapter.path("head.pt"))?;
            let (model, _cfg, _vision) = qwen35(&base, Some(&lora), &opts, &device)?;
            let head: HashMap<String, Tensor> = candle_core::pickle::read_all_with_key(adapter.path("head.pt"), Some("head"))?.into_iter().collect();
            let get = |n: &str| head.get(n).with_context(|| format!("head.pt has no `head.{n}`")).and_then(|t| Ok(t.to_device(&device)?.to_dtype(DType::F32)?));
            let head = PointerHead { q_w: get("q.weight")?, q_b: get("q.bias")?, k_w: get("k.weight")?, k_b: get("k.bias")?, temperature: meta.temperature };
            if head.q_w.dims() != [meta.head_dim, model.cfg.hidden_size] {
                bail!("pointer head is {:?}, expected [{}, {}]", head.q_w.dims(), meta.head_dim, model.cfg.hidden_size);
            }
            let id = |t: &str| tokenizer.token_to_id(t).with_context(|| format!("tokenizer has no {t}"));
            let kind = Kind::Pointer { head, decide_id: id(cev_core::prompt::kev::DECIDE)?, option_end_id: id(cev_core::prompt::kev::OPTION_END)? };
            tracing::info!(temperature = meta.temperature, head_dim = meta.head_dim, "Kev pointer head");
            let id = name("kev", &adapter);
            let engine = HybridEngine::new(Box::new(model), tokenizer, kind, id, adapter.repo.clone(), opts)?;
            engine.warmup()?;
            Ok(Arc::new(engine))
        }
    }
}
