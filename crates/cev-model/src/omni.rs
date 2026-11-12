//! Liquid's d1-omni: a bidirectional LFM2.5 encoder trunk shared by text,
//! image and speech, a decision head that scores the hidden state at each
//! option's `<|mask|>` marker, a SigLIP2 image path and a FastConformer
//! audio path whose outputs enter as a prefix of embeddings before the text.
//!
//! The record (see [`cev_core::prompt::omni`]) is budgeted in tokens, so the
//! sequence is assembled here from ids rather than tokenized from text.

use crate::audio::{AudioConfig, AudioEncoder};
use crate::engine::EngineOptions;
#[cfg(not(target_arch = "wasm32"))]
use crate::hub::RepoFiles;
use crate::decoder::Decoder;
use crate::lfm2::{self, Lfm2};
use crate::media::{self, LfmTileParams};
use crate::siglip2::{Projector, Siglip2, VisionConfig as SiglipConfig};
use anyhow::{Context, Result, bail};
#[cfg(not(target_arch = "wasm32"))]
use candle_core::Device;
use candle_core::{D, DType, Module, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder, layer_norm, linear};
use cev_core::prompt::omni::{self as prompt, Mode, Parts};
use cev_core::{Backend, BackendOutput, Capabilities, CompiledQuestion, CompiledRequest, Debias, Family, QuestionType, Readout};
use serde::Deserialize;
use std::collections::HashMap;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Deserialize)]
pub struct OmniConfig {
    pub text_config: lfm2::Config,
    pub vision_config: SiglipConfig,
    pub audio_config: AudioConfig,
    #[serde(default = "d2048")]
    pub projector_hidden_size: usize,
    #[serde(default = "two")]
    pub head_layers: usize,
    #[serde(default = "d16k")]
    pub max_length: usize,
    #[serde(default = "d896")]
    pub image_text_length: usize,
    #[serde(default = "d15360")]
    pub audio_text_length: usize,
    #[serde(default)]
    pub temperatures: HashMap<String, f32>,
    #[serde(default = "one")]
    pub bos_token_id: u32,
}

fn d2048() -> usize {
    2048
}
fn two() -> usize {
    2
}
fn d16k() -> usize {
    16_384
}
fn d896() -> usize {
    896
}
fn d15360() -> usize {
    15_360
}
fn one() -> u32 {
    1
}

impl OmniConfig {
    /// `temperatures["choice:3-5"]`, else `temperatures["choice"]`, else 1.
    fn temperature(&self, kind: QuestionType, k: usize) -> f32 {
        let bucket = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
        let key = format!("{}:{bucket}", kind.as_str());
        self.temperatures.get(&key).or_else(|| self.temperatures.get(kind.as_str())).copied().unwrap_or(1.0)
    }
}

/// One `nn.TransformerEncoderLayer(norm_first=True)` with ReLU.
struct HeadLayer {
    in_w: Tensor,
    in_b: Tensor,
    out: Linear,
    l1: Linear,
    l2: Linear,
    n1: LayerNorm,
    n2: LayerNorm,
}

struct DecisionHead {
    type_emb: Tensor,
    layers: Vec<HeadLayer>,
    s_ln: LayerNorm,
    s_1: Linear,
    s_2: Linear,
    heads: usize,
}

impl DecisionHead {
    fn new(d: usize, layers: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let heads = d / 64;
        let layers = (0..layers)
            .map(|i| {
                let l = vb.pp("head.layers").pp(i);
                Ok(HeadLayer {
                    in_w: l.pp("self_attn").get((3 * d, d), "in_proj_weight")?,
                    in_b: l.pp("self_attn").get(3 * d, "in_proj_bias")?,
                    out: linear(d, d, l.pp("self_attn.out_proj"))?,
                    l1: linear(d, 4 * d, l.pp("linear1"))?,
                    l2: linear(4 * d, d, l.pp("linear2"))?,
                    n1: layer_norm(d, 1e-5, l.pp("norm1"))?,
                    n2: layer_norm(d, 1e-5, l.pp("norm2"))?,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            type_emb: vb.pp("type_emb").get((3, d), "weight")?,
            layers,
            s_ln: layer_norm(d, 1e-5, vb.pp("scorer").pp(0))?,
            s_1: linear(d, d, vb.pp("scorer").pp(1))?,
            s_2: linear(d, 1, vb.pp("scorer").pp(3))?,
            heads,
        })
    }

    /// `text`: (B, L, d) text positions (right-padded to L; `lengths` real);
    /// `markers[b]`: option marker positions; `qtype[b]`: 0 choice, 1 score, 2 noul.
    fn forward(&self, text: &Tensor, lengths: &[usize], markers: &[Vec<usize>], qtype: &[u32]) -> candle_core::Result<Vec<Vec<f32>>> {
        let (b, l, d) = text.dims3()?;
        let types = self.type_emb.index_select(&Tensor::new(qtype, text.device())?, 0)?.unsqueeze(1)?; // (B,1,d)
        let mut x = text.broadcast_add(&types)?;
        // Key padding mask (B, 1, 1, L).
        let mut m = vec![0f32; b * l];
        for (r, &len) in lengths.iter().enumerate() {
            for j in len..l {
                m[r * l + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(m, (b, 1, 1, l), text.device())?.to_dtype(x.dtype())?;
        let (h, dk) = (self.heads, d / self.heads);
        for layer in &self.layers {
            let n = layer.n1.forward(&x)?;
            let qkv = n.broadcast_matmul(&layer.in_w.t()?)?.broadcast_add(&layer.in_b)?; // (B,L,3d)
            let split = |i: usize| -> candle_core::Result<Tensor> { qkv.narrow(2, i * d, d)?.reshape((b, l, h, dk))?.transpose(1, 2)?.contiguous() };
            let (q, k, v) = (split(0)?, split(1)?, split(2)?);
            let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (dk as f64).sqrt())?.broadcast_add(&mask)?;
            let attn = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, l, d))?;
            x = (x + layer.out.forward(&ctx)?)?;
            let f = layer.l2.forward(&layer.l1.forward(&layer.n2.forward(&x)?)?.relu()?)?;
            x = (x + f)?;
        }
        let mut out = Vec::with_capacity(b);
        let flat = x.reshape((b * l, d))?;
        for (r, ms) in markers.iter().enumerate() {
            let idx: Vec<u32> = ms.iter().map(|&p| (r * l + p) as u32).collect();
            let g = flat.index_select(&Tensor::new(idx.as_slice(), flat.device())?, 0)?;
            let z = self.s_2.forward(&self.s_1.forward(&self.s_ln.forward(&g)?)?.gelu_erf()?)?.squeeze(D::Minus1)?.to_dtype(DType::F32)?;
            out.push(z.to_vec1()?);
        }
        Ok(out)
    }
}

struct Ids {
    bos: u32,
    state: u32,
    question: u32,
    option: u32,
    option_end: u32,
    decide: u32,
    marker: u32,
}

pub struct OmniEngine {
    trunk: Lfm2,
    head: DecisionHead,
    vision: Siglip2,
    projector: Projector,
    audio: AudioEncoder,
    tokenizer: Tokenizer,
    ids: Ids,
    cfg: OmniConfig,
    name: String,
    backbone: String,
    opts: EngineOptions,
    tiles: LfmTileParams,
}

/// One question's token row.
struct Row {
    ids: Vec<u32>,
    markers: Vec<usize>,
    text: String,
    kind: QuestionType,
    k: usize,
}

impl OmniEngine {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(files: &RepoFiles, opts: EngineOptions, device: &Device) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(files.path("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let weights = files.weights()?;
        let dtype = crate::load::pick_dtype(&opts.dtype, device, 0)?;
        tracing::info!(dir = %files.dir.display(), ?device, ?dtype, "loading d1-omni weights");
        let w1_rows = {
            let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&weights)? };
            st.get("encoder.layers.0.feed_forward.w1.weight").map(|t| t.shape()[0]).unwrap_or(0)
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, dtype, device)? };
        Self::from_parts(&files.read("config.json")?, tokenizer, vb, w1_rows, files.repo.clone(), opts)
    }

    /// From `config.json`, the tokenizer and a `VarBuilder` over the weights
    /// (`w1_rows`: rows of the first feed-forward `w1`, to detect the
    /// adjusted FFN width).
    pub fn from_parts(config: &[u8], tokenizer: Tokenizer, vb: VarBuilder, w1_rows: usize, repo: String, opts: EngineOptions) -> Result<Self> {
        let cfg: OmniConfig = serde_json::from_slice(config).context("d1-omni config.json")?;
        let adjust = lfm2::ffn_adjust_from_shape(&cfg.text_config, w1_rows) || cfg.text_config.block_auto_adjust_ff_dim;
        let d = cfg.text_config.hidden_size;
        let trunk = Lfm2::new(cfg.text_config.clone(), vb.pp("encoder"), cfg.max_length, adjust)?;
        let head = DecisionHead::new(d, cfg.head_layers, vb.pp("head"))?;
        let vision = Siglip2::new(cfg.vision_config.clone(), vb.pp("vision.tower.vision_model"))?;
        let projector = Projector::new(cfg.vision_config.hidden_size, cfg.projector_hidden_size, d, 2, vb.pp("vision.projector"))?;
        let audio = AudioEncoder::new(cfg.audio_config.clone(), d, vb.pp("audio"))?;
        let id = |t: &str| tokenizer.token_to_id(t).with_context(|| format!("tokenizer has no {t}"));
        let ids = Ids { bos: cfg.bos_token_id, state: id(prompt::STATE)?, question: id(prompt::QUESTION)?, option: id(prompt::OPTION)?, option_end: id(prompt::OPTION_END)?, decide: id(prompt::DECIDE)?, marker: id(prompt::MARKER)? };
        let name = opts.name.clone().unwrap_or_else(|| format!("cev-{}", repo.rsplit('/').next().unwrap_or("d1-omni").to_lowercase()));
        Ok(Self {
            trunk,
            head,
            vision,
            projector,
            audio,
            tokenizer,
            ids,
            cfg,
            name,
            backbone: repo,
            opts,
            tiles: LfmTileParams { bicubic: false, ..LfmTileParams::default() },
        })
    }

    fn encode(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(s, false).map_err(anyhow::Error::msg)?.get_ids().to_vec())
    }

    /// The media prefix embeddings (p, d), if any.
    fn prefix(&self, req: &CompiledRequest) -> Result<Option<Tensor>> {
        if let Some(wav) = &req.media.audio {
            let samples = crate::audio::wav_to_16k_mono(wav)?;
            return Ok(Some(self.audio.forward(&samples)?));
        }
        if req.media.images.is_empty() {
            return Ok(None);
        }
        let mut parts = Vec::new();
        for (i, bytes) in req.media.images.iter().enumerate() {
            let img = media::decode_image(bytes).with_context(|| format!("images[{i}]"))?;
            let li = media::lfm_image(&img, &self.tiles);
            for c in &li.crops {
                let px = Tensor::from_vec(c.pixels.clone(), (c.h * c.w, self.vision.cfg.patch_dim()), self.trunk.device())?;
                let f = self.vision.forward(&px, c.h, c.w)?;
                parts.push(self.projector.forward(&f, c.h, c.w)?);
            }
        }
        Ok(Some(Tensor::cat(&parts, 0)?))
    }

    /// `prompt.encode`: the record's ids and marker positions within a token budget.
    ///
    /// `state_memo` memoizes the state's tokens: every question shares the
    /// state, and re-encoding a large one per row was the expensive part.
    fn row(&self, parts: &Parts, q: &CompiledQuestion, max_len: usize, state_memo: &mut Option<(String, Vec<u32>)>) -> Result<Row> {
        let k = parts.options.len();
        let (block, per) = prompt::budget(k, max_len, 24);
        let mut question = vec![self.ids.question];
        question.extend(self.encode(&parts.instructions)?);
        question.truncate(16.max(block));
        let mut markers = Vec::with_capacity(k);
        for text in &parts.options {
            markers.push(question.len() + 1);
            question.push(self.ids.option);
            question.push(self.ids.marker);
            let mut t = self.encode(&format!(" {text}"))?;
            t.truncate(per);
            question.extend(t);
            question.push(self.ids.option_end);
        }
        question.push(self.ids.decide);
        let room = max_len.saturating_sub(question.len() + 2);
        let mut state_ids = vec![self.ids.state];
        if state_memo.as_ref().is_none_or(|(text, _)| *text != parts.state) {
            *state_memo = Some((parts.state.clone(), self.encode(&parts.state)?));
        }
        let st = state_memo.as_ref().map(|(_, ids)| ids[..ids.len().min(room)].to_vec()).unwrap_or_default();
        state_ids.extend(st);
        let mut ids = vec![self.ids.bos];
        ids.extend(&state_ids);
        ids.extend(&question);
        ids.truncate(max_len);
        let markers: Vec<usize> = markers.iter().map(|m| m + 1 + state_ids.len()).collect();
        if markers.last().is_some_and(|&m| m >= max_len) {
            return Err(cev_core::InputError(format!("question `{}`: the options do not fit in the context", q.id)).into());
        }
        Ok(Row { ids, markers, text: prompt::suffix(parts), kind: q.kind, k })
    }
}

impl Backend for OmniEngine {
    fn id(&self) -> &str {
        &self.name
    }
    fn backbone(&self) -> &str {
        &self.backbone
    }
    fn hidden_size(&self) -> usize {
        self.cfg.text_config.hidden_size
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities { family: Family::D1Omni, images: true, audio: true, max_options: cev_core::compile::MAX_OPTIONS, debias: Debias::None }
    }

    fn run(&self, req: &CompiledRequest) -> Result<BackendOutput> {
        if req.media.audio.is_some() && !req.media.images.is_empty() {
            bail!("a request carries images or audio, not both");
        }
        let mode = if req.media.audio.is_some() { Mode::Audio } else if !req.media.images.is_empty() { Mode::Image } else { Mode::Text };
        let t0 = web_time::Instant::now();
        let prefix = self.prefix(req)?;
        let p = prefix.as_ref().map_or(0, |t| t.dim(0).unwrap_or(0));
        let budget = match mode {
            Mode::Text => self.cfg.max_length,
            Mode::Image => self.cfg.image_text_length,
            Mode::Audio => self.cfg.audio_text_length,
        };
        let max_len = budget.min(self.cfg.max_length.saturating_sub(p)).min(self.opts.max_context);
        if max_len < 64 {
            return Err(cev_core::InputError(format!("the media take {p} of the {} positions; send fewer images", self.cfg.max_length)).into());
        }
        let mut state_memo = None;
        let rows: Vec<Row> = req.questions.iter().map(|q| self.row(&prompt::parts(req, q, mode), q, max_len, &mut state_memo)).collect::<Result<_>>()?;
        let d = self.hidden_size();
        let dev = self.trunk.device();
        let text_len = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let l = p + text_len;
        // Batches under the token budget: rows × (prefix + longest row).
        let per_batch = (self.opts.pass_tokens / l.max(1)).max(1);
        let mut readouts = Vec::with_capacity(rows.len());
        let qtype = |k: QuestionType| match k {
            QuestionType::Choice => 0u32,
            QuestionType::Score => 1,
            QuestionType::Noul => 2,
        };
        for chunk in rows.chunks(per_batch) {
            let b = chunk.len();
            let tl = chunk.iter().map(|r| r.ids.len()).max().unwrap_or(0);
            let l = p + tl;
            let mut ids = Vec::with_capacity(b * tl);
            for r in chunk {
                ids.extend_from_slice(&r.ids);
                ids.extend(std::iter::repeat_n(0u32, tl - r.ids.len()));
            }
            let text_emb = self.trunk.embed(&ids)?.reshape((b, tl, d))?;
            let embeds = match &prefix {
                Some(pf) => Tensor::cat(&[pf.unsqueeze(0)?.broadcast_as((b, p, d))?.contiguous()?.to_dtype(text_emb.dtype())?, text_emb], 1)?,
                None => text_emb,
            };
            let lengths: Vec<usize> = chunk.iter().map(|r| p + r.ids.len()).collect();
            let h = self.trunk.encode(&embeds, &lengths, &vec![p; b])?; // (B, L, d) f32
            let text = h.narrow(1, p, l - p)?.contiguous()?;
            let text_lengths: Vec<usize> = chunk.iter().map(|r| r.ids.len()).collect();
            let markers: Vec<Vec<usize>> = chunk.iter().map(|r| r.markers.clone()).collect();
            let types: Vec<u32> = chunk.iter().map(|r| qtype(r.kind)).collect();
            let logits = self.head.forward(&text.to_dtype(self.trunk.dtype())?, &text_lengths, &markers, &types)?;
            let flat = text.reshape((b * tl, d))?;
            for (i, (r, mut z)) in chunk.iter().zip(logits).enumerate() {
                z.truncate(r.k);
                if mode == Mode::Text {
                    let t = self.cfg.temperature(r.kind, r.k);
                    for v in &mut z {
                        *v /= t;
                    }
                }
                let hidden: Vec<f32> = flat.get(i * tl + r.ids.len() - 1)?.to_vec1()?;
                readouts.push(Readout { logits: z, hidden, suffix: r.text.clone(), codes: Vec::new() });
            }
        }
        tracing::debug!(ms = t0.elapsed().as_secs_f64() * 1e3, ?mode, prefix = p, questions = rows.len(), "forward");
        let _ = dev;
        Ok(BackendOutput { readouts, input_tokens: rows.iter().map(|r| p + r.ids.len()).sum(), prefix_cached: false, prefix: prompt::prefix(req, mode) })
    }
}
