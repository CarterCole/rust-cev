//! `Backend` over any [`Decoder`]: the state prefix runs once (chunked) into a
//! cached [`PrefixState`], and every question runs as one row of a batch that
//! continues from it, right-padded to the longest row. Exact by construction:
//! rows are independent, padding sits after every real token and every layer
//! is causal, so no row sees another or its own padding.
//!
//! This is the execution shape the recurrent (Gated DeltaNet, short-conv)
//! layers need; attention-only Qwen3 keeps the packed block-mask engine in
//! [`crate::engine`]. On top of the decoder sits one of the family readouts:
//!
//! | kind | prompt | answer |
//! |---|---|---|
//! | [`Kind::Codes`] | cev's own or Nimble's | LM-head logits of one-token option codes at the last position |
//! | [`Kind::Matrix`] | d3 | a separate (255, hidden) readout at the last position |
//! | [`Kind::Pointer`] | Kev | two projections score each `<|box_end|>` against `<|fim_suffix|>` |

use crate::decoder::{Attn, Decoder, Positions, PrefixState};
use crate::engine::EngineOptions;
use crate::media::{self, LfmTileParams, QwenImageParams};
use crate::qwen35_vision::{Grid, Qwen35Vision};
use crate::siglip2::{Projector, Siglip2};
use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use cev_core::prompt;
use cev_core::{Backend, BackendOutput, Capabilities, CompiledQuestion, CompiledRequest, Debias, Family, PromptFormat, Readout};
use parking_lot::Mutex;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Kev's pointer head: `z_k = (K h_opt_k · Q h_decide) / √dp / T`.
pub struct PointerHead {
    pub q_w: Tensor,
    pub q_b: Tensor,
    pub k_w: Tensor,
    pub k_b: Tensor,
    pub temperature: f32,
}

impl PointerHead {
    /// `decide`: (n, d); `opts`: (m, d); `owner[i]`: which decide row option i
    /// belongs to. Returns the m logits.
    fn logits(&self, decide: &Tensor, opts: &Tensor, owner: &[u32]) -> Result<Vec<f32>> {
        let dp = self.q_w.dim(0)? as f64;
        let q = decide.matmul(&self.q_w.t()?)?.broadcast_add(&self.q_b)?; // (n, dp)
        let k = opts.matmul(&self.k_w.t()?)?.broadcast_add(&self.k_b)?; // (m, dp)
        let q = q.index_select(&Tensor::new(owner, q.device())?, 0)?;
        let z = (k.mul(&q)?.sum(1)? * (1.0 / dp.sqrt() / self.temperature as f64))?;
        Ok(z.to_vec1()?)
    }
}

pub enum Kind {
    /// One-token codes read from the LM head. `head`: (n_codes, hidden) f32 in code order.
    Codes { family: Family, format: Option<PromptFormat>, codes: Vec<String>, head: Tensor },
    /// d1: each option is a group of token ids (a code and ` code`, the
    /// spellings of yes/no, a digit), scored as the best of the group's LM
    /// logits at the last position. `lm_head`: (vocab, hidden).
    Groups { lm_head: Tensor, bos: String },
    /// d3: a (255, hidden) f32 readout over the codes. `causal` false runs
    /// the attention layers bidirectionally over each whole prompt
    /// (`attention_mode: noncausal_full_attention`), which rules out a prefix
    /// state: every question is one full row.
    Matrix { codes: Vec<String>, weight: Tensor, temperature: f32, causal: bool },
    /// Kev's pointer head over the record's special tokens.
    Pointer { head: PointerHead, decide_id: u32, option_end_id: u32 },
}

impl Kind {
    fn family(&self) -> Family {
        match self {
            Kind::Codes { family, .. } => *family,
            Kind::Groups { .. } => Family::D1,
            Kind::Matrix { .. } => Family::D3,
            Kind::Pointer { .. } => Family::Kev,
        }
    }

    fn max_options(&self) -> usize {
        match self {
            Kind::Codes { codes, .. } | Kind::Matrix { codes, .. } => codes.len(),
            Kind::Groups { .. } | Kind::Pointer { .. } => cev_core::compile::MAX_OPTIONS,
        }
    }

    fn debias(&self) -> Debias {
        match self {
            Kind::Codes { family: Family::Cev, .. } => Debias::Full,
            _ => Debias::None,
        }
    }
}

/// A vision tower and how its output enters the prompt.
pub enum Vision {
    /// Qwen3.5: the prompt's single `<|image_pad|>` per image expands to the
    /// merged patch count; image tokens take multi-axis RoPE positions.
    Qwen { tower: Qwen35Vision, params: QwenImageParams, image_pad_id: u32 },
    /// LFM2-VL: SigLIP2 over 512 px tiles and a thumbnail, each `<image>`
    /// placeholder replaced by the processor's tile markup.
    Lfm { tower: Siglip2, projector: Projector, params: LfmTileParams, image_id: u32 },
}

/// One image through the tower: its tokens' features and, for LFM2-VL, the
/// text that stands for it in the prompt.
struct Encoded {
    features: Tensor,
    tokens: usize,
    grid: Option<Grid>,
    markup: Option<String>,
}

impl Vision {
    fn image_token_id(&self) -> u32 {
        match self {
            Vision::Qwen { image_pad_id, .. } => *image_pad_id,
            Vision::Lfm { image_id, .. } => *image_id,
        }
    }

    fn mrope(&self) -> bool {
        matches!(self, Vision::Qwen { .. })
    }

    fn merge(&self) -> usize {
        match self {
            Vision::Qwen { tower, .. } => tower.cfg.spatial_merge_size,
            Vision::Lfm { .. } => 1,
        }
    }

    fn encode(&self, bytes: &[u8], dev: &candle_core::Device, dtype: candle_core::DType) -> Result<Encoded> {
        let img = media::decode_image(bytes)?;
        match self {
            Vision::Qwen { tower, params, .. } => {
                let p = media::qwen_patches(&img, params)?;
                let pixels = Tensor::from_vec(p.pixels, (p.grid.patches(), p.patch_dim), dev)?;
                let features = tower.forward(&pixels, p.grid)?.to_dtype(dtype)?;
                Ok(Encoded { tokens: features.dim(0)?, features, grid: Some(p.grid), markup: None })
            }
            Vision::Lfm { tower, projector, params, .. } => {
                let li = media::lfm_image(&img, params);
                let mut parts = Vec::with_capacity(li.crops.len());
                for c in &li.crops {
                    let pixels = Tensor::from_vec(c.pixels.clone(), (c.h * c.w, tower.cfg.patch_dim()), dev)?;
                    let f = tower.forward(&pixels, c.h, c.w)?;
                    parts.push(projector.forward(&f, c.h, c.w)?.to_dtype(dtype)?);
                }
                let features = Tensor::cat(&parts, 0)?;
                Ok(Encoded { tokens: features.dim(0)?, features, grid: None, markup: Some(li.markup(params.downsample as usize)) })
            }
        }
    }
}

pub struct HybridEngine {
    decoder: Box<dyn Decoder>,
    tokenizer: Tokenizer,
    kind: Kind,
    vision: Option<Vision>,
    name: String,
    backbone: String,
    opts: EngineOptions,
    pad_id: u32,
    cache: Mutex<PrefixCache>,
}

/// A contiguous run of image tokens inside the prefix and the features
/// (len, hidden) that replace their embeddings.
struct ImageRun {
    start: usize,
    len: usize,
    features: Tensor,
}

/// How to read one row's answer.
enum Read {
    /// Option `k` is scored at the row's last token by `ids[k]` (a code's row
    /// of the readout).
    Last { ids: Vec<u32> },
    /// Option `k` is the best LM logit over `groups[k]` at the row's last token.
    Groups { groups: Vec<Vec<u32>> },
    /// Pointer: the decide position and each option's position in the row.
    Pointer { decide: usize, opts: Vec<usize> },
}

struct Branch {
    ids: Vec<u32>,
    text: String,
    codes: Vec<String>,
    read: Read,
}

struct Plan {
    /// Prefix token ids, image pads expanded to one per image token.
    prefix: Vec<u32>,
    prefix_text: String,
    prefix_pos: Positions,
    images: Vec<ImageRun>,
    /// Hash of the request's images (part of the cache key).
    media_hash: u64,
    branches: Vec<Branch>,
    cached: bool,
    state: Option<Arc<PrefixState>>,
}

/// Positions of a sequence that starts with `prefix` and goes on with `extra`
/// text tokens.
fn extend_positions(prefix: &Positions, extra: usize) -> Positions {
    let next = prefix.next();
    let mut text = prefix.text.clone();
    text.extend(next..next + extra as u32);
    let thw = prefix.thw.as_ref().map(|[t, h, w]| {
        let ext = |a: &Vec<u32>| {
            let mut v = a.clone();
            v.extend(next..next + extra as u32);
            v
        };
        [ext(t), ext(h), ext(w)]
    });
    Positions { text, thw }
}

/// Multimodal RoPE positions of a prefix (HF `get_rope_index`): text tokens
/// count up on all three axes; an image's tokens share its temporal position
/// and spread over height and width, and the text after it continues from
/// the larger of the two.
fn image_positions(len: usize, runs: &[(usize, usize, Grid)], merge: usize) -> Positions {
    let (mut t, mut h, mut w) = (Vec::with_capacity(len), Vec::with_capacity(len), Vec::with_capacity(len));
    let mut cur = 0u32;
    let mut i = 0;
    let mut runs = runs.iter().peekable();
    while i < len {
        if let Some(&&(start, n, grid)) = runs.peek().filter(|r| r.0 == i) {
            let (gh, gw) = (grid.h / merge, grid.w / merge);
            for frame in 0..grid.t {
                for r in 0..gh {
                    for c in 0..gw {
                        t.push(cur + frame as u32);
                        h.push(cur + r as u32);
                        w.push(cur + c as u32);
                    }
                }
            }
            cur += gh.max(gw) as u32;
            i = start + n;
            runs.next();
        } else {
            t.push(cur);
            h.push(cur);
            w.push(cur);
            cur += 1;
            i += 1;
        }
    }
    Positions { text: t.clone(), thw: Some([t, h, w]) }
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl HybridEngine {
    pub fn new(decoder: Box<dyn Decoder>, tokenizer: Tokenizer, kind: Kind, name: String, backbone: String, opts: EngineOptions) -> Result<Self> {
        let pad_id = ["<|endoftext|>", "<|pad|>", "<pad>"].iter().find_map(|t| tokenizer.token_to_id(t)).unwrap_or(0);
        let engine = Self { decoder, tokenizer, kind, vision: None, name, backbone, pad_id, cache: Mutex::new(PrefixCache::new(opts.prefix_cache_tokens)), opts };
        Ok(engine)
    }

    /// Take images through a Qwen3.5 vision tower whose output replaces the
    /// `<|image_pad|>` tokens of the prompt.
    pub fn with_vision(mut self, tower: Qwen35Vision, params: QwenImageParams) -> Result<Self> {
        let image_pad_id = self.tokenizer.token_to_id("<|image_pad|>").context("tokenizer has no <|image_pad|>")?;
        self.vision = Some(Vision::Qwen { tower, params, image_pad_id });
        Ok(self)
    }

    /// Take images through LFM2-VL's SigLIP2 tower and projector.
    pub fn with_lfm_vision(mut self, tower: Siglip2, projector: Projector, params: LfmTileParams) -> Result<Self> {
        let image_id = self.tokenizer.token_to_id("<image>").context("tokenizer has no <image>")?;
        self.vision = Some(Vision::Lfm { tower, projector, params, image_id });
        Ok(self)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn decoder(&self) -> &dyn Decoder {
        self.decoder.as_ref()
    }

    /// Compile GPU kernels up front so the first real request is not slow.
    pub fn warmup(&self) -> Result<()> {
        let t = web_time::Instant::now();
        let req = serde_json::from_value(serde_json::json!({
            "state": "warmup",
            "questions": {"a": {"type": "noul", "instructions": "ok?"}, "b": {"type": "score", "instructions": "level?", "criteria": ["a", "b", "c"]}}
        }))?;
        self.run(&cev_core::compile(&req, self.kind.max_options())?)?;
        self.cache.lock().entries.clear();
        tracing::info!(ms = t.elapsed().as_millis() as u64, "warmed up");
        Ok(())
    }

    fn encode(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(s, false).map_err(anyhow::Error::msg)?.get_ids().to_vec())
    }

    /// Prompt text of a request: the shared prefix and one suffix per question.
    fn render(&self, req: &CompiledRequest) -> Result<(String, Vec<(String, Vec<String>)>)> {
        Ok(match &self.kind {
            Kind::Codes { family: Family::Cev, format, codes, .. } => {
                let fmt = format.context("cev prompt needs a format")?;
                (fmt.request_prefix(req), req.questions.iter().map(|q| (fmt.question_suffix(q, codes), codes[..q.options.len()].to_vec())).collect())
            }
            Kind::Codes { family: Family::Nimble, codes, .. } => {
                (prompt::nimble::prefix(req, codes), req.questions.iter().map(|q| (prompt::nimble::suffix(q), codes[..q.options.len()].to_vec())).collect())
            }
            Kind::Codes { family, .. } => bail!("no code prompt for family {}", family.as_str()),
            Kind::Groups { bos, .. } => (
                prompt::d1::prefix(req, bos),
                req.questions
                    .iter()
                    .map(|q| {
                        let codes: Vec<String> = self.d1_aliases(q).into_iter().map(|(c, _)| c).collect();
                        (prompt::d1::suffix(q, &codes), codes)
                    })
                    .collect(),
            ),
            Kind::Matrix { codes, .. } => (prompt::d3::prefix(req), req.questions.iter().map(|q| (prompt::d3::suffix(q, codes), codes[..q.options.len()].to_vec())).collect()),
            Kind::Pointer { .. } => (prompt::kev::prefix(req), req.questions.iter().map(|q| (prompt::kev::suffix(q), Vec::new())).collect()),
        })
    }

    /// d1's option codes for a choice question: the labels when they are
    /// single letters, else `A`.., each replaced by the first fallback that is
    /// one token. Returns (code, token id) per option in display order.
    fn d1_aliases(&self, q: &CompiledQuestion) -> Vec<(String, u32)> {
        let labels: Vec<&str> = q.options.iter().map(|o| o.name.as_str()).collect();
        let one = |s: &str| -> Option<u32> {
            let e = self.tokenizer.encode(s, false).ok()?;
            (e.len() == 1).then(|| e.get_ids()[0])
        };
        let mut used = std::collections::HashSet::new();
        let mut out = Vec::with_capacity(labels.len());
        let pool = prompt::d1::fallback_pool();
        for code in prompt::d1::option_codes(&labels) {
            let mut take = |raw: &str| -> bool {
                match one(raw) {
                    Some(id) if !used.contains(&id) => {
                        used.insert(id);
                        out.push((raw.to_string(), id));
                        true
                    }
                    _ => false,
                }
            };
            if !take(&code) && !pool.iter().any(|raw| take(raw)) {
                out.push((code, u32::MAX));
            }
        }
        out
    }

    /// Token ids d1 scores for each option of a question (display order).
    fn d1_groups(&self, q: &CompiledQuestion) -> Result<Vec<Vec<u32>>> {
        let ids = |texts: &[&str]| -> Vec<u32> {
            let mut v = Vec::new();
            for t in texts {
                if let Ok(e) = self.tokenizer.encode(*t, false) {
                    if e.len() == 1 && !v.contains(&e.get_ids()[0]) {
                        v.push(e.get_ids()[0]);
                    }
                }
            }
            v
        };
        Ok(match q.kind {
            cev_core::QuestionType::Noul => {
                let (yes, no) = (ids(&prompt::d1::YES_FORMS), ids(&prompt::d1::NO_FORMS));
                if yes.is_empty() || no.is_empty() {
                    bail!("tokenizer has no single-token yes/no");
                }
                // cev's noul order is [false, true].
                q.options.iter().map(|o| if o.name == "true" { yes.clone() } else { no.clone() }).collect()
            }
            cev_core::QuestionType::Score => {
                if q.options.len() > prompt::d1::MAX_SCORE_LEVELS {
                    bail!("question `{}`: d1 scores 2 to {} levels", q.id, prompt::d1::MAX_SCORE_LEVELS);
                }
                q.options
                    .iter()
                    .map(|o| {
                        let g = ids(&[o.name.as_str()]);
                        if g.is_empty() { bail!("digit `{}` is not one token", o.name) } else { Ok(g) }
                    })
                    .collect::<Result<_>>()?
            }
            cev_core::QuestionType::Choice => self
                .d1_aliases(q)
                .into_iter()
                .map(|(code, id)| {
                    if id == u32::MAX {
                        bail!("question `{}`: no single-token code left for {} options", q.id, q.options.len());
                    }
                    let mut g = vec![id];
                    g.extend(ids(&[&format!(" {code}")]).into_iter().filter(|i| *i != id));
                    Ok(g)
                })
                .collect::<Result<_>>()?,
        })
    }

    /// Run the request's images through the tower.
    fn encode_images(&self, req: &CompiledRequest) -> Result<Vec<Encoded>> {
        let Some(v) = &self.vision else { return Ok(Vec::new()) };
        req.media.images.iter().enumerate().map(|(i, bytes)| v.encode(bytes, self.decoder.device(), self.decoder.dtype()).with_context(|| format!("images[{i}]"))).collect()
    }

    /// The prompt text with each image placeholder expanded (LFM2-VL markup).
    fn expand_text(&self, text: &str, encoded: &[Encoded]) -> String {
        let mut out = text.to_string();
        for e in encoded {
            if let Some(m) = &e.markup {
                if let Some(i) = out.find(prompt::d1::IMAGE_PLACEHOLDER) {
                    out.replace_range(i..i + prompt::d1::IMAGE_PLACEHOLDER.len(), m);
                }
            }
        }
        out
    }

    /// Replace each single image pad in `ids` with the image's token count
    /// (Qwen); LFM2-VL's markup already has one token per feature.
    fn expand_pads(&self, ids: &[u32], encoded: &[Encoded]) -> Vec<u32> {
        let Some(v) = &self.vision else { return ids.to_vec() };
        if !v.mrope() {
            return ids.to_vec();
        }
        let pad = v.image_token_id();
        let mut out = Vec::with_capacity(ids.len() + encoded.iter().map(|e| e.tokens).sum::<usize>());
        let mut k = 0;
        for &t in ids {
            if t == pad && k < encoded.len() {
                out.extend(std::iter::repeat_n(t, encoded[k].tokens));
                k += 1;
            } else {
                out.push(t);
            }
        }
        out
    }

    fn plan(&self, req: &CompiledRequest) -> Result<Plan> {
        if req.media.audio.is_some() {
            bail!("the {} backend `{}` does not take audio", self.kind.family().as_str(), self.name);
        }
        if !req.media.images.is_empty() && self.vision.is_none() {
            bail!("the {} backend `{}` takes text only", self.kind.family().as_str(), self.name);
        }
        let encoded = self.encode_images(req)?;
        let merge = self.vision.as_ref().map_or(1, Vision::merge);
        let media_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            req.media.images.hash(&mut h);
            h.finish()
        };
        if matches!(self.kind, Kind::Codes { family: Family::Nimble, .. }) && req.questions.iter().any(|q| q.order.iter().enumerate().any(|(i, &o)| i != o)) {
            bail!("Nimble's prompt lists every question's options in one schema, so option rotation (debias permute/full) is not supported; use `debias: none` or `calibrate`");
        }
        let (prefix_text, suffixes) = self.render(req)?;
        let prefix_text = self.expand_text(&prefix_text, &encoded);
        // Tokenize every full prompt, then split at the longest token prefix
        // shared with the prefix text, so BPE merges across the boundary can
        // never change what the model sees.
        let prefix_ids = self.expand_pads(&self.encode(&prefix_text)?, &encoded);
        crate::engine::check_fits(prefix_ids.len(), suffixes.iter().map(|(s, _)| self.encode(s).map(|t| t.len())), self.decoder.max_len())?;
        let fulls: Vec<Vec<u32>> = suffixes.iter().map(|(s, _)| Ok(self.expand_pads(&self.encode(&format!("{prefix_text}{s}"))?, &encoded))).collect::<Result<_>>()?;
        let split = fulls.iter().map(|f| lcp(&prefix_ids, f)).min().unwrap_or(0).min(prefix_ids.len());
        let split = split.min(fulls.iter().map(|f| f.len().saturating_sub(1)).min().unwrap_or(0));
        if split == 0 {
            bail!("empty shared prefix");
        }
        let longest = fulls.iter().map(Vec::len).max().unwrap_or(0);
        if longest >= self.decoder.max_len() {
            return Err(cev_core::InputError(format!("prompt is {longest} tokens; the context limit is {}", self.decoder.max_len())).into());
        }
        // Image token runs inside the prefix take the features in order; the
        // whole prefix (images included) must be shared by every question.
        let mut images = Vec::new();
        let mut grids = Vec::new();
        if let Some(v) = &self.vision {
            let tok = v.image_token_id();
            let all = Tensor::cat(&encoded.iter().map(|e| &e.features).collect::<Vec<_>>(), 0).ok();
            let total: usize = encoded.iter().map(|e| e.tokens).sum();
            let positions: Vec<usize> = prefix_ids.iter().enumerate().filter(|(_, t)| **t == tok).map(|(i, _)| i).collect();
            if positions.len() != total {
                bail!("{} image tokens in the prompt for {total} image features", positions.len());
            }
            let mut used = 0;
            let mut i = 0;
            while i < positions.len() {
                let start = positions[i];
                let mut len = 1;
                while i + len < positions.len() && positions[i + len] == start + len {
                    len += 1;
                }
                let features = all.as_ref().context("no image features")?.narrow(0, used, len)?;
                images.push(ImageRun { start, len, features });
                used += len;
                i += len;
            }
            grids = encoded.iter().filter_map(|e| e.grid).collect();
            if images.last().is_some_and(|r| r.start + r.len > split) {
                bail!("image tokens must lie within the shared prefix");
            }
        }
        let text_prefix_pos = Positions::text(0, split);
        let prefix_pos_full = if grids.is_empty() { Positions::text(0, prefix_ids.len()) } else { image_positions(prefix_ids.len(), &images.iter().zip(&grids).map(|(r, g)| (r.start, r.len, *g)).collect::<Vec<_>>(), merge) };
        // Whole prompts as rows when the model attends bidirectionally.
        let (split, prefix_pos) = if self.uses_prefix() { (split, if grids.is_empty() { text_prefix_pos } else { prefix_pos_full.slice(0..split) }) } else { (0, prefix_pos_full) };
        let prefix = fulls[0][..split].to_vec();
        let mut branches = Vec::with_capacity(fulls.len());
        for ((full, (text, codes)), q) in fulls.iter().zip(suffixes).zip(&req.questions) {
            let ids = full[split..].to_vec();
            let read = self.read_spec(q, &ids, &codes)?;
            branches.push(Branch { ids, text, codes, read });
        }
        let state = if split > 0 { self.cache.lock().get(&prefix, media_hash) } else { None };
        Ok(Plan { cached: state.is_some(), state, prefix, prefix_text, prefix_pos, images, media_hash, branches })
    }

    fn read_spec(&self, q: &CompiledQuestion, ids: &[u32], codes: &[String]) -> Result<Read> {
        let k = q.options.len();
        Ok(match &self.kind {
            Kind::Codes { .. } | Kind::Matrix { .. } => {
                if codes.len() < k {
                    bail!("question `{}`: {k} options but only {} codes", q.id, codes.len());
                }
                Read::Last { ids: (0..k as u32).collect() }
            }
            Kind::Groups { .. } => Read::Groups { groups: self.d1_groups(q)? },
            Kind::Pointer { decide_id, option_end_id, .. } => {
                let opts: Vec<usize> = ids.iter().enumerate().filter(|(_, t)| **t == *option_end_id).map(|(i, _)| i).collect();
                if opts.len() != k || ids.last() != Some(decide_id) {
                    bail!("question `{}`: record layout mismatch ({} option ends for {k} options)", q.id, opts.len());
                }
                Read::Pointer { decide: ids.len() - 1, opts }
            }
        })
    }

    /// Run the prefix in chunks into a state (and cache it).
    fn prefill(&self, plan: &mut Plan) -> Result<Arc<PrefixState>> {
        if let Some(s) = &plan.state {
            return Ok(s.clone());
        }
        let chunk = self.opts.prefill_chunk.max(1);
        let all = self.embeds(&plan.prefix, &plan.images)?; // (P, d)
        let mut state: Option<PrefixState> = None;
        let mut done = 0;
        while done < plan.prefix.len() {
            let n = chunk.min(plan.prefix.len() - done);
            let embeds = all.narrow(0, done, n)?.unsqueeze(0)?;
            let pos = plan.prefix_pos.slice(done..done + n);
            let (_, next) = self.decoder.forward(&embeds, &pos, state.as_ref(), true, &Attn::Causal)?;
            state = next;
            done += n;
        }
        let state = Arc::new(state.context("empty prefix")?);
        self.cache.lock().put(plan.prefix.clone(), plan.media_hash, state.clone());
        plan.state = Some(state.clone());
        Ok(state)
    }

    /// Token embeddings (n, hidden) with image features in place of the pads.
    fn embeds(&self, ids: &[u32], images: &[ImageRun]) -> Result<Tensor> {
        let mut e = self.decoder.embed(ids)?;
        for run in images {
            e = e.slice_assign(&[run.start..run.start + run.len, 0..e.dim(1)?], &run.features)?;
        }
        Ok(e)
    }

    /// Whether questions continue a cached prefix state (causal models) or
    /// run as whole rows (bidirectional attention).
    fn uses_prefix(&self) -> bool {
        !matches!(self.kind, Kind::Matrix { causal: false, .. })
    }

    /// Run a batch of rows; returns (B, T, hidden) f32. With a `state` the
    /// rows are branches continuing it; without, each row is a whole prompt
    /// (starting with the plan's prefix, images included).
    fn rows(&self, plan: &Plan, state: Option<&PrefixState>, rows: &[&Branch]) -> Result<Tensor> {
        let t = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let mut ids = Vec::with_capacity(rows.len() * t);
        for r in rows {
            ids.extend_from_slice(&r.ids);
            ids.extend(std::iter::repeat_n(self.pad_id, t - r.ids.len()));
        }
        let d = self.decoder.hidden_size();
        let mut embeds = self.decoder.embed(&ids)?.reshape((rows.len(), t, d))?;
        let pos = match state {
            Some(s) => Positions::text(s.next_pos, t),
            None => {
                for run in &plan.images {
                    let f = run.features.unsqueeze(0)?.broadcast_as((rows.len(), run.len, d))?.contiguous()?;
                    embeds = embeds.slice_assign(&[0..rows.len(), run.start..run.start + run.len, 0..d], &f)?;
                }
                extend_positions(&plan.prefix_pos, t - plan.prefix_pos.len())
            }
        };
        let attn = if self.uses_prefix() { Attn::Causal } else { Attn::Bidirectional { lengths: rows.iter().map(|r| r.ids.len()).collect() } };
        let (hidden, _) = self.decoder.forward(&embeds, &pos, state, false, &attn)?;
        Ok(hidden)
    }

    fn readouts(&self, hidden: &Tensor, rows: &[&Branch]) -> Result<Vec<Readout>> {
        let (b, t, d) = hidden.dims3()?;
        let flat = hidden.reshape((b * t, d))?;
        // Gather every position any row reads: its answer position first.
        let mut answer_at = Vec::with_capacity(b);
        let mut opt_at = Vec::new();
        let mut owner = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            match &r.read {
                Read::Last { .. } | Read::Groups { .. } => answer_at.push((i * t + r.ids.len() - 1) as u32),
                Read::Pointer { decide, opts } => {
                    answer_at.push((i * t + decide) as u32);
                    for &o in opts {
                        opt_at.push((i * t + o) as u32);
                        owner.push(i as u32);
                    }
                }
            }
        }
        let answers = flat.index_select(&Tensor::new(answer_at.as_slice(), flat.device())?, 0)?; // (B, d) f32
        let hiddens: Vec<Vec<f32>> = answers.to_vec2()?;
        let logits: Vec<Vec<f32>> = match &self.kind {
            Kind::Codes { head, .. } => answers.matmul(&head.t()?)?.to_vec2()?,
            Kind::Groups { lm_head, .. } => {
                // Every token any row scores, once; rows read their groups by column.
                let mut ids: Vec<u32> = rows.iter().flat_map(|r| match &r.read { Read::Groups { groups } => groups.iter().flatten().copied().collect::<Vec<_>>(), _ => Vec::new() }).collect();
                ids.sort_unstable();
                ids.dedup();
                let head = lm_head.index_select(&Tensor::new(ids.as_slice(), lm_head.device())?, 0)?.to_dtype(candle_core::DType::F32)?;
                let all: Vec<Vec<f32>> = answers.matmul(&head.t()?)?.to_vec2()?;
                rows.iter()
                    .zip(all)
                    .map(|(r, z)| match &r.read {
                        Read::Groups { groups } => groups.iter().map(|g| g.iter().map(|id| z[ids.binary_search(id).expect("scored")]).fold(f32::NEG_INFINITY, f32::max)).collect(),
                        _ => Vec::new(),
                    })
                    .collect()
            }
            Kind::Matrix { weight, temperature, .. } => (answers.matmul(&weight.t()?)? / *temperature as f64)?.to_vec2()?,
            Kind::Pointer { head, .. } => {
                let opts = flat.index_select(&Tensor::new(opt_at.as_slice(), flat.device())?, 0)?;
                let z = head.logits(&answers, &opts, &owner)?;
                let mut per_row = vec![Vec::new(); b];
                for (o, v) in owner.iter().zip(z) {
                    per_row[*o as usize].push(v);
                }
                per_row
            }
        };
        Ok(rows
            .iter()
            .zip(hiddens)
            .zip(logits)
            .map(|((r, h), z)| {
                let logits = match &r.read {
                    Read::Last { ids } => ids.iter().map(|&i| z[i as usize]).collect(),
                    Read::Groups { .. } | Read::Pointer { .. } => z,
                };
                Readout { logits, hidden: h, suffix: r.text.clone(), codes: r.codes.clone() }
            })
            .collect())
    }
}

impl Backend for HybridEngine {
    fn id(&self) -> &str {
        &self.name
    }
    fn backbone(&self) -> &str {
        &self.backbone
    }
    fn hidden_size(&self) -> usize {
        self.decoder.hidden_size()
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities { family: self.kind.family(), images: self.vision.is_some(), audio: false, max_options: self.kind.max_options(), debias: self.kind.debias() }
    }

    fn run(&self, req: &CompiledRequest) -> Result<BackendOutput> {
        let t0 = web_time::Instant::now();
        let mut plan = self.plan(req)?;
        let t_tok = t0.elapsed();
        let state = if plan.prefix.is_empty() { None } else { Some(self.prefill(&mut plan)?) };
        let t_prefill = t0.elapsed();
        // Rows in batches under the token budget (rows × longest row).
        let mut readouts = Vec::with_capacity(plan.branches.len());
        let mut start = 0;
        let mut passes = 0;
        while start < plan.branches.len() {
            let mut end = start + 1;
            let mut longest = plan.branches[start].ids.len();
            while end < plan.branches.len() {
                let l = longest.max(plan.branches[end].ids.len());
                if l * (end + 1 - start) > self.opts.pass_tokens {
                    break;
                }
                longest = l;
                end += 1;
            }
            let rows: Vec<&Branch> = plan.branches[start..end].iter().collect();
            let hidden = self.rows(&plan, state.as_deref(), &rows)?;
            readouts.extend(self.readouts(&hidden, &rows)?);
            passes += 1;
            start = end;
        }
        tracing::debug!(
            tokenize_ms = t_tok.as_secs_f64() * 1e3,
            prefill_ms = (t_prefill - t_tok).as_secs_f64() * 1e3,
            rows_ms = (t0.elapsed() - t_prefill).as_secs_f64() * 1e3,
            passes,
            prefix = plan.prefix.len(),
            cached = plan.cached,
            "forward"
        );
        Ok(BackendOutput {
            readouts,
            input_tokens: plan.prefix.len() + plan.branches.iter().map(|b| b.ids.len()).sum::<usize>(),
            prefix_cached: plan.cached,
            prefix: plan.prefix_text,
        })
    }
}

/// LRU of prefix states keyed by token ids and image hash, bounded by total
/// cached tokens.
struct PrefixCache {
    budget: usize,
    entries: Vec<((Vec<u32>, u64), Arc<PrefixState>)>,
}

impl PrefixCache {
    fn new(budget: usize) -> Self {
        Self { budget, entries: Vec::new() }
    }

    fn get(&mut self, key: &[u32], media: u64) -> Option<Arc<PrefixState>> {
        let i = self.entries.iter().position(|((k, m), _)| k == key && *m == media)?;
        let e = self.entries.remove(i);
        let s = e.1.clone();
        self.entries.push(e);
        Some(s)
    }

    fn put(&mut self, key: Vec<u32>, media: u64, s: Arc<PrefixState>) {
        if key.len() > self.budget {
            return;
        }
        self.entries.retain(|((k, m), _)| !(*k == key && *m == media));
        self.entries.push(((key, media), s));
        while self.entries.iter().map(|((k, _), _)| k.len()).sum::<usize>() > self.budget {
            self.entries.remove(0);
        }
    }
}

