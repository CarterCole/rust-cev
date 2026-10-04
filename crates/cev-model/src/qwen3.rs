//! Qwen3 decoder, prefill-only, with an explicit shared-prefix KV cache.
//!
//! Unlike `candle_transformers::models::qwen3`, weights are immutable (`&self`)
//! and caches are values, so one model serves concurrent requests and a state
//! prefix is computed once and read by every question of a request:
//!
//! ```text
//!   prefix tokens ──prefill (chunked)──▶ PrefixKv  (1 × kv_heads × P × d per layer)
//!                                            │ shared, never copied
//!   question suffixes (packed, block mask) ──┴─▶ attend [prefix ‖ own suffix]
//! ```
//!
//! A request is one packed pass: the (uncached tail of the) state prefix and
//! every question suffix go in one sequence, and an additive mask decides what
//! each token sees: prefix tokens are causal, each question sees the prefix
//! plus earlier tokens of its own question only. Several prefixes (e.g. the
//! state and the content-free calibration prompt) can share one pass. On Metal
//! attention runs in candle's fused SDPA kernel; on the CPU each layer splits
//! the new tokens across the cores (see [`Layer::forward`]).

use crate::quant::{QLinear, SCALE_SUFFIX};
use candle_core::{DType, Device, Module, Result, Tensor, bail};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, linear_b, rms_norm};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

/// Raw tensors of an int8 checkpoint (see [`crate::quant`]), by name.
pub type Tensors = HashMap<String, Tensor>;

/// A weight matrix: dense in the model dtype, or int8 on the CPU.
enum Proj {
    Dense(Linear),
    Quant(Arc<QLinear>),
}

impl Module for Proj {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Dense(l) => l.forward(x),
            Proj::Quant(q) => q.forward(x),
        }
    }
}

fn qlinear(name: &str, quant: &Tensors) -> Result<Arc<QLinear>> {
    match (quant.get(name), quant.get(&format!("{name}{SCALE_SUFFIX}"))) {
        (Some(q), Some(scale)) => Ok(Arc::new(QLinear::new(q, scale)?)),
        _ => bail!("quantized checkpoint is missing `{name}`"),
    }
}

fn proj(inp: usize, out: usize, bias: bool, vb: VarBuilder, quant: Option<&Tensors>) -> Result<Proj> {
    Ok(match quant {
        None => Proj::Dense(linear_b(inp, out, bias, vb)?),
        Some(_) if bias => bail!("quantized checkpoints with attention bias are not supported"),
        Some(t) => Proj::Quant(qlinear(&format!("{}.weight", vb.prefix()), t)?),
    })
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_bias: bool,
    pub max_position_embeddings: usize,
    /// Top-level in older configs; under `rope_parameters` in transformers >= 5.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
}

impl Config {
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().and_then(|r| r.rope_theta))
            .unwrap_or(1_000_000.0)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

struct Rotary {
    sin: Tensor,
    cos: Tensor,
}

impl Rotary {
    fn new(cfg: &Config, max_len: usize, dtype: DType, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / dim as f64) as f32)
            .collect();
        let n = inv.len();
        let inv = Tensor::from_vec(inv, (1, n), dev)?;
        let t = Tensor::arange(0u32, max_len as u32, dev)?.to_dtype(DType::F32)?.reshape((max_len, 1))?;
        let f = t.matmul(&inv)?;
        Ok(Self { sin: f.sin()?.to_dtype(dtype)?, cos: f.cos()?.to_dtype(dtype)? })
    }

    /// x: (1, H, T, D); `pos` holds the absolute position of each of the T tokens.
    fn apply(&self, x: &Tensor, pos: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, &self.cos.index_select(pos, 0)?, &self.sin.index_select(pos, 0)?)
    }
}

struct Attention {
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

/// K/V of one layer: (1, kv_heads, S, D).
#[derive(Clone)]
pub struct LayerKv {
    pub k: Tensor,
    pub v: Tensor,
}

/// Additive attention mask for `n` new tokens after `past` cached ones.
/// `segments[i]` is the segment of new token i; a new token sees every cached
/// token plus earlier new tokens of its own segment. One segment = causal
/// prefill; one segment per question = isolated question branches.
fn segment_mask(segments: &[u32], past: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let n = segments.len();
    let mut m = vec![0f32; n * (past + n)];
    for i in 0..n {
        let row = &mut m[i * (past + n) + past..(i + 1) * (past + n)];
        for (j, v) in row.iter_mut().enumerate() {
            if j > i || segments[j] != segments[i] {
                *v = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(m, (n, past + n), dev)?.to_dtype(dtype)
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder, quant: Option<&Tensors>) -> Result<Self> {
        let (h, kvh, d, hs) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim(), cfg.hidden_size);
        Ok(Self {
            q: proj(hs, h * d, cfg.attention_bias, vb.pp("q_proj"), quant)?,
            k: proj(hs, kvh * d, cfg.attention_bias, vb.pp("k_proj"), quant)?,
            v: proj(hs, kvh * d, cfg.attention_bias, vb.pp("v_proj"), quant)?,
            o: proj(h * d, hs, cfg.attention_bias, vb.pp("o_proj"), quant)?,
            q_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            heads: h,
            kv_heads: kvh,
            head_dim: d,
        })
    }

    /// Q, K and V of new tokens x (1, T, hidden), each (1, heads, T, D), with
    /// Q and K normed and rotated to the tokens' positions.
    fn qkv(&self, x: &Tensor, rot: &Rotary, pos: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, _) = x.dims3()?;
        let shape = |y: Tensor, h: usize| y.reshape((b, t, h, self.head_dim))?.transpose(1, 2);
        let q = self.q_norm.forward(&shape(self.q.forward(x)?, self.heads)?.contiguous()?)?;
        let k = self.k_norm.forward(&shape(self.k.forward(x)?, self.kv_heads)?.contiguous()?)?;
        let v = shape(self.v.forward(x)?, self.kv_heads)?.contiguous()?;
        Ok((rot.apply(&q, pos)?, rot.apply(&k, pos)?, v))
    }

    /// What queries q (1, heads, T, D) read from K/V (1, kv_heads, S, D)
    /// through `mask` (T, S): (1, T, hidden).
    fn read(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, _, t, _) = q.dims4()?;
        let ctx = self.attend(q, k, v, mask)?;
        self.o.forward(&ctx.transpose(1, 2)?.reshape((b, t, self.heads * self.head_dim))?)
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let (_, h, t, _) = q.dims4()?;
        let s = k.dim(2)?;
        if q.device().is_metal() && matches!(self.head_dim, 64 | 128 | 256) {
            // Fused kernel with native GQA: K/V are never repeated or copied.
            let m = mask.reshape((1, 1, t, s))?.broadcast_as((1, h, t, s))?;
            return candle_nn::ops::sdpa(q, k, v, Some(&m), false, scale as f32, 1.0);
        }
        // Grouped queries without repeating K/V: the queries of the heads
        // that share a K/V head are stacked along the token axis instead.
        let (b, d) = (q.dim(0)?, self.head_dim);
        let rows = (b, self.kv_heads, h / self.kv_heads * t, d);
        let scores = (q.reshape(rows)?.matmul(&k.t()?)?.to_dtype(DType::F32)? * scale)?;
        let scores = scores.reshape((b, h, t, s))?.broadcast_add(&mask.to_dtype(DType::F32)?)?;
        let p = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
        p.reshape((b, self.kv_heads, rows.2, s))?.matmul(v)?.reshape((b, h, t, d))
    }
}

struct Mlp {
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        self.down.forward(&(g * self.up.forward(x)?)?)
    }
}

struct Layer {
    attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Layer {
    fn new(cfg: &Config, vb: VarBuilder, quant: Option<&Tensors>) -> Result<Self> {
        let (hs, is) = (cfg.hidden_size, cfg.intermediate_size);
        let m = vb.pp("mlp");
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("self_attn"), quant)?,
            mlp: Mlp {
                gate: proj(hs, is, false, m.pp("gate_proj"), quant)?,
                up: proj(hs, is, false, m.pp("up_proj"), quant)?,
                down: proj(is, hs, false, m.pp("down_proj"), quant)?,
            },
            ln1: rms_norm(hs, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: rms_norm(hs, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn mlp_block(&self, x: Tensor) -> Result<Tensor> {
        let h = self.mlp.forward(&self.ln2.forward(&x)?)?;
        x + h
    }

    /// New tokens x (1, T, hidden) attending to `past` K/V (if any) and to
    /// each other through `mask` (T, past+T). Returns the layer's output and
    /// the concatenated K/V.
    ///
    /// Given K/V, a token's way through the layer does not depend on the other
    /// new tokens, so on the CPU the rows are split across the rayon pool:
    /// each chunk computes its Q/K/V, then each reads from all of K/V and runs
    /// the MLP. candle's CPU ops are single-threaded or fork once per row,
    /// which leaves most cores idle; whole chunks of rows keep them busy.
    fn forward(&self, x: &Tensor, rot: &Rotary, pos: &Tensor, mask: &Tensor, past: Option<&LayerKv>) -> Result<(Tensor, LayerKv)> {
        let chunks = row_chunks(x.dim(1)?, x.device());
        let qkv = each(&chunks, |_, start, len| {
            self.attn.qkv(&self.ln1.forward(&x.narrow(1, start, len)?)?, rot, &pos.narrow(0, start, len)?)
        })?;
        let join = |past: Option<&Tensor>, new: Vec<&Tensor>| match (past, new.as_slice()) {
            (None, [one]) => Ok((*one).clone()),
            _ => Tensor::cat(&past.into_iter().chain(new).collect::<Vec<_>>(), 2),
        };
        let k = join(past.map(|p| &p.k), qkv.iter().map(|c| &c.1).collect())?;
        let v = join(past.map(|p| &p.v), qkv.iter().map(|c| &c.2).collect())?;
        let mut out = each(&chunks, |i, start, len| {
            let h = self.attn.read(&qkv[i].0, &k, &v, &mask.narrow(0, start, len)?)?;
            self.mlp_block((x.narrow(1, start, len)? + h)?)
        })?;
        let out = if out.len() == 1 { out.remove(0) } else { Tensor::cat(&out, 1)? };
        Ok((out, LayerKv { k, v }))
    }
}

/// Fewest tokens worth a thread of their own.
const MIN_ROWS: usize = 8;

/// `(start, len)` ranges of `t` new tokens to run side by side: one per CPU
/// thread while each keeps `MIN_ROWS` tokens, a single one on a GPU.
fn row_chunks(t: usize, dev: &Device) -> Vec<(usize, usize)> {
    let n = if dev.is_cpu() { rayon::current_num_threads().min(t / MIN_ROWS).max(1) } else { 1 };
    (0..n).map(|i| (i * t / n, (i + 1) * t / n - i * t / n)).collect()
}

/// `f(index, start, len)` for every chunk, in parallel when there are several.
fn each<T: Send>(chunks: &[(usize, usize)], f: impl Fn(usize, usize, usize) -> Result<T> + Sync) -> Result<Vec<T>> {
    match *chunks {
        [(start, len)] => Ok(vec![f(0, start, len)?]),
        _ => chunks.par_iter().enumerate().map(|(i, &(start, len))| f(i, start, len)).collect(),
    }
}

/// K/V for every layer of a prefilled state prefix.
pub struct PrefixKv {
    pub layers: Vec<LayerKv>,
    pub len: usize,
}

impl PrefixKv {
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| 2 * l.k.elem_count() * l.k.dtype().size_in_bytes())
            .sum()
    }
}

/// A (vocab_size, hidden) table we only ever gather rows from.
enum Table {
    Dense(Embedding),
    Quant(Arc<QLinear>),
}

impl Table {
    /// Rows for `ids`: (n, hidden).
    fn rows(&self, ids: &[u32], dev: &Device) -> Result<Tensor> {
        match self {
            Table::Dense(e) => e.embeddings().index_select(&Tensor::new(ids, dev)?, 0),
            Table::Quant(q) => q.select(ids),
        }
    }
}

pub struct Qwen3 {
    embed: Table,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Table,
    rot: Rotary,
    pub cfg: Config,
    pub device: Device,
    pub dtype: DType,
    pub max_len: usize,
}

impl Qwen3 {
    /// `quant` holds the raw tensors when the checkpoint is int8 (CPU only);
    /// norms always come from `vb`.
    pub fn new(cfg: Config, vb: VarBuilder, quant: Option<&Tensors>, max_len: usize) -> Result<Self> {
        let max_len = max_len.min(cfg.max_position_embeddings);
        let table = |name: &str| -> Result<Table> {
            Ok(match quant {
                Some(t) => Table::Quant(qlinear(&format!("{name}.weight"), t)?),
                None => Table::Dense(Embedding::new(vb.pp(name).get((cfg.vocab_size, cfg.hidden_size), "weight")?, cfg.hidden_size)),
            })
        };
        let embed = table("model.embed_tokens")?;
        let lm_head = match &embed {
            _ if !cfg.tie_word_embeddings => table("lm_head")?,
            Table::Dense(e) => Table::Dense(e.clone()),
            Table::Quant(q) => Table::Quant(q.clone()),
        };
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::new(&cfg, vb.pp("model.layers").pp(i), quant))
            .collect::<Result<_>>()?;
        Ok(Self {
            rot: Rotary::new(&cfg, max_len, vb.dtype(), vb.device())?,
            norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            embed,
            layers,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            max_len,
            cfg,
        })
    }

    /// LM-head rows for the given token ids: (n, hidden).
    pub fn head_rows(&self, ids: &[u32]) -> Result<Tensor> {
        self.lm_head.rows(ids, &self.device)
    }

    /// Prefill the shared prefix in chunks (bounds attention memory to chunk × P).
    pub fn prefill(&self, tokens: &[u32], chunk: usize) -> Result<PrefixKv> {
        let mut caches: Vec<Option<LayerKv>> = (0..self.layers.len()).map(|_| None).collect();
        let mut past = 0;
        for part in tokens.chunks(chunk.max(1)) {
            let mask = segment_mask(&vec![0; part.len()], past, self.dtype, &self.device)?;
            let pos = Tensor::arange(past as u32, (past + part.len()) as u32, &self.device)?;
            past += part.len();
            let mut x = self.embed.rows(part, &self.device)?.unsqueeze(0)?;
            for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
                let (h, kv) = layer.forward(&x, &self.rot, &pos, &mask, cache.as_ref())?;
                *cache = Some(kv);
                x = h;
            }
        }
        Ok(PrefixKv { layers: caches.into_iter().map(|c| c.expect("non-empty prefix")).collect(), len: tokens.len() })
    }

    /// One forward pass over packed new tokens.
    ///
    /// * `past`: per-layer K/V of already computed tokens this pass may read
    ///   (several prefixes concatenated), or `None`.
    /// * `mask`: additive (T, past_len + T); it alone decides which past
    ///   prefix and which new tokens each new token sees.
    /// * `pos`: absolute position of each new token.
    ///
    /// Returns the final normed hidden state at `gather` (f32) and, for each
    /// `keep` range of new tokens, its per-layer K/V (for the prefix cache).
    pub fn forward_packed(
        &self,
        past: Option<&[LayerKv]>,
        tokens: &[u32],
        pos: &[u32],
        mask: Vec<f32>,
        gather: &[u32],
        keep: &[(usize, usize)],
    ) -> Result<(Tensor, Vec<Vec<LayerKv>>)> {
        let t = tokens.len();
        let past_len = past.and_then(|p| p.first()).map(|l| l.k.dim(2)).transpose()?.unwrap_or(0);
        let mask = Tensor::from_vec(mask, (t, past_len + t), &self.device)?.to_dtype(self.dtype)?;
        let pos = Tensor::new(pos, &self.device)?;
        let mut kept: Vec<Vec<LayerKv>> = keep.iter().map(|_| Vec::with_capacity(self.layers.len())).collect();
        let mut x = self.embed.rows(tokens, &self.device)?.unsqueeze(0)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let (h, kv) = layer.forward(&x, &self.rot, &pos, &mask, past.map(|p| &p[i]))?;
            for (slot, &(start, len)) in kept.iter_mut().zip(keep) {
                slot.push(LayerKv {
                    k: kv.k.narrow(2, past_len + start, len)?.contiguous()?,
                    v: kv.v.narrow(2, past_len + start, len)?.contiguous()?,
                });
            }
            x = h;
        }
        let x = x.squeeze(0)?.index_select(&Tensor::new(gather, &self.device)?, 0)?;
        Ok((self.norm.forward(&x)?.to_dtype(DType::F32)?, kept))
    }
}
