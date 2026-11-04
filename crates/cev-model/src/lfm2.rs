//! Liquid's LFM2 / LFM2.5 language model: a hybrid of gated short-convolution
//! blocks and GQA attention blocks with a SwiGLU MLP, prefill-only, with an
//! explicit prefix state. Two shapes:
//!
//! * [`Lfm2`] implements [`Decoder`] causally (d1 / LFM2.5-VL): conv blocks
//!   keep their last two inputs, attention blocks their keys and values;
//! * [`Lfm2::encode`] runs the same weights as d1-omni's bidirectional trunk:
//!   a centred 3-tap convolution, no causal mask, and media prefix positions
//!   that never read the text after them.
//!
//! Module names follow HF transformers' `Lfm2Model`.

use crate::decoder::{Attn, Decoder, LayerState, Positions, PrefixState};
use candle_core::{D, DType, Device, Module, Result, Tensor, bail};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, linear_b, rms_norm};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub layer_types: Vec<String>,
    #[serde(default = "three", rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default = "eps5")]
    pub norm_eps: f64,
    #[serde(default = "max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub block_auto_adjust_ff_dim: bool,
    #[serde(default)]
    pub block_ffn_dim_multiplier: Option<f64>,
    #[serde(default = "m256")]
    pub block_multiple_of: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
}

fn three() -> usize {
    3
}
fn eps5() -> f64 {
    1e-5
}
fn max_pos() -> usize {
    128_000
}
fn m256() -> usize {
    256
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_theta.or_else(|| self.rope_parameters.as_ref().and_then(|r| r["rope_theta"].as_f64())).unwrap_or(1_000_000.0)
    }

    /// The MLP width after HF's sizing rule. d1-omni's trunk config has no
    /// `block_auto_adjust_ff_dim` key yet sizes its MLP as if it were set,
    /// so `adjust` is decided by the caller from the weights' shapes.
    pub fn ffn_dim(&self, adjust: bool) -> usize {
        if !adjust {
            return self.intermediate_size;
        }
        let mut h = (2 * self.intermediate_size) / 3;
        if let Some(m) = self.block_ffn_dim_multiplier {
            h = (m * h as f64) as usize;
        }
        self.block_multiple_of * h.div_ceil(self.block_multiple_of)
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        self.layer_types.get(layer).is_some_and(|t| t == "full_attention")
    }
}

struct Rotary {
    inv: Tensor,
}

impl Rotary {
    fn new(cfg: &Config, dev: &Device) -> Result<Self> {
        let d = cfg.head_dim();
        let inv: Vec<f32> = (0..d).step_by(2).map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / d as f64) as f32).collect();
        Ok(Self { inv: Tensor::from_vec(inv, (1, d / 2), dev)? })
    }

    fn cos_sin(&self, pos: &[u32], dtype: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
        let p = Tensor::from_vec(pos.iter().map(|&v| v as f32).collect::<Vec<_>>(), (pos.len(), 1), dev)?;
        let f = p.broadcast_mul(&self.inv)?;
        Ok((f.cos()?.to_dtype(dtype)?, f.sin()?.to_dtype(dtype)?))
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, h, s, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, h, groups, s, d))?.reshape((b, h * groups, s, d))
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, kvh, d, hs) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim(), cfg.hidden_size);
        Ok(Self {
            q: linear_b(hs, h * d, false, vb.pp("q_proj"))?,
            k: linear_b(hs, kvh * d, false, vb.pp("k_proj"))?,
            v: linear_b(hs, kvh * d, false, vb.pp("v_proj"))?,
            o: linear_b(h * d, hs, false, vb.pp("out_proj"))?,
            q_norm: rms_norm(d, cfg.norm_eps, vb.pp("q_layernorm"))?,
            k_norm: rms_norm(d, cfg.norm_eps, vb.pp("k_layernorm"))?,
            heads: h,
            kv_heads: kvh,
            head_dim: d,
        })
    }

    /// `mask`: additive (1 or B, 1, T, past + T).
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor, past: Option<&LayerState>) -> Result<(Tensor, LayerState)> {
        let (b, t, _) = x.dims3()?;
        let (h, d) = (self.heads, self.head_dim);
        let q = self.q_norm.forward(&self.q.forward(x)?.reshape((b, t, h, d))?)?.transpose(1, 2)?.contiguous()?;
        let k = self.k_norm.forward(&self.k.forward(x)?.reshape((b, t, self.kv_heads, d))?)?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(x)?.reshape((b, t, self.kv_heads, d))?.transpose(1, 2)?.contiguous()?;
        let q = candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k, cos, sin)?;
        let (k, v) = match past {
            Some(LayerState::Attn { k: pk, v: pv }) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            Some(_) => bail!("wrong state kind for an attention layer"),
            None => (k, v),
        };
        let scale = 1.0 / (d as f64).sqrt();
        let s = k.dim(2)?;
        let ctx = if q.device().is_metal() && matches!(d, 64 | 128) {
            let m = mask.broadcast_as((b, h, t, s))?;
            candle_nn::ops::sdpa(&q, &k, &v, Some(&m), false, scale as f32, 1.0)?
        } else {
            let g = h / self.kv_heads;
            let (kk, vv) = (repeat_kv(&k, g)?, repeat_kv(&v, g)?);
            let scores = (q.matmul(&kk.t()?.contiguous()?)?.to_dtype(DType::F32)? * scale)?.broadcast_add(&mask.to_dtype(DType::F32)?)?;
            candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(vv.dtype())?.matmul(&vv.contiguous()?)?
        };
        let out = self.o.forward(&ctx.transpose(1, 2)?.reshape((b, t, h * d))?)?;
        Ok((out, LayerState::Attn { k, v }))
    }
}

struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    /// (hidden, 1, kernel)
    weight: Tensor,
    bias: Option<Tensor>,
    kernel: usize,
    dim: usize,
}

impl ShortConv {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            in_proj: linear_b(d, 3 * d, cfg.conv_bias, vb.pp("in_proj"))?,
            out_proj: linear_b(d, d, cfg.conv_bias, vb.pp("out_proj"))?,
            weight: vb.pp("conv").get((d, 1, cfg.conv_l_cache), "weight")?,
            bias: if cfg.conv_bias { Some(vb.pp("conv").get(d, "bias")?) } else { None },
            kernel: cfg.conv_l_cache,
            dim: d,
        })
    }

    /// Causal: `out(c · conv(b · u))`, the conv continuing from the last
    /// `kernel - 1` inputs in `past`.
    fn forward(&self, x: &Tensor, past: Option<&LayerState>) -> Result<(Tensor, LayerState)> {
        let (bsz, t, _) = x.dims3()?;
        let bcu = self.in_proj.forward(x)?.transpose(1, 2)?.contiguous()?; // (B, 3d, T)
        let b = bcu.narrow(1, 0, self.dim)?;
        let c = bcu.narrow(1, self.dim, self.dim)?;
        let u = bcu.narrow(1, 2 * self.dim, self.dim)?;
        let bu = b.mul(&u)?;
        let last = match past {
            Some(LayerState::Conv { last }) => last.clone(),
            Some(_) => bail!("wrong state kind for a conv layer"),
            None => Tensor::zeros((bsz, self.dim, self.kernel - 1), bu.dtype(), bu.device())?,
        };
        let full = Tensor::cat(&[&last, &bu], 2)?; // (B, d, T + K - 1)
        let new_last = full.narrow(2, t, self.kernel - 1)?.contiguous()?;
        let mut z = crate::decoder::depthwise_conv1d(&full, &self.weight)?; // (B, d, T)
        if let Some(bias) = &self.bias {
            z = z.broadcast_add(&bias.reshape((1, self.dim, 1))?)?;
        }
        let y = c.mul(&z)?.transpose(1, 2)?.contiguous()?;
        Ok((self.out_proj.forward(&y)?, LayerState::Conv { last: new_last }))
    }

    /// Bidirectional, centred 3-tap (d1-omni's trunk). `pad`: (B, T, 1) ones
    /// on real tokens; `keep_right`: (B, 1, T) zero where the right neighbour
    /// must not be read (the last media position).
    fn centred(&self, x: &Tensor, pad: &Tensor, keep_right: &Tensor) -> Result<Tensor> {
        if self.kernel != 3 {
            bail!("the centred convolution needs a 3-tap kernel");
        }
        let (_, t, _) = x.dims3()?;
        let bcu = self.in_proj.forward(&x.broadcast_mul(pad)?)?.transpose(1, 2)?.contiguous()?;
        let b = bcu.narrow(1, 0, self.dim)?;
        let c = bcu.narrow(1, self.dim, self.dim)?;
        let u = bcu.narrow(1, 2 * self.dim, self.dim)?;
        let bx = b.mul(&u)?; // (B, d, T)
        let xp = bx.pad_with_zeros(2, 1, 1)?; // (B, d, T + 2)
        let w = self.weight.reshape((1, self.dim, self.kernel))?;
        let tap = |i: usize| w.narrow(2, i, 1)?.contiguous();
        let left = xp.narrow(2, 0, t)?;
        let mid = xp.narrow(2, 1, t)?;
        let right = xp.narrow(2, 2, t)?.broadcast_mul(keep_right)?;
        let y = left.broadcast_mul(&tap(0)?)?.add(&mid.broadcast_mul(&tap(1)?)?)?.add(&right.broadcast_mul(&tap(2)?)?)?;
        let y = c.mul(&y)?.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&y)
    }
}

struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w2.forward(&(candle_nn::ops::silu(&self.w1.forward(x)?)? * self.w3.forward(x)?)?)
    }
}

enum Operator {
    Conv(ShortConv),
    Attn(Attention),
}

struct Layer {
    op: Operator,
    mlp: Mlp,
    op_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

pub struct Lfm2 {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rot: Rotary,
    pub cfg: Config,
    device: Device,
    dtype: DType,
    max_len: usize,
}

/// Additive causal mask (1, 1, T, past + T).
fn causal_mask(t: usize, past: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; t * (past + t)];
    for i in 0..t {
        for j in i + 1..t {
            m[i * (past + t) + past + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(m, (1, 1, t, past + t), dev)?.to_dtype(dtype)
}

impl Lfm2 {
    /// `vb` rooted at the language model (`model.language_model`, `model` or
    /// `encoder`). `ffn_adjust`: whether HF's `block_auto_adjust_ff_dim` rule
    /// sizes the MLP (read from the weights when unsure).
    pub fn new(cfg: Config, vb: VarBuilder, max_len: usize, ffn_adjust: bool) -> Result<Self> {
        let embed = Embedding::new(vb.pp("embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?, cfg.hidden_size);
        let ffn = cfg.ffn_dim(ffn_adjust);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let l = vb.pp("layers").pp(i);
                let op = if cfg.is_attention(i) { Operator::Attn(Attention::new(&cfg, l.pp("self_attn"))?) } else { Operator::Conv(ShortConv::new(&cfg, l.pp("conv"))?) };
                let f = l.pp("feed_forward");
                Ok(Layer {
                    op,
                    mlp: Mlp { w1: linear_b(cfg.hidden_size, ffn, false, f.pp("w1"))?, w2: linear_b(ffn, cfg.hidden_size, false, f.pp("w2"))?, w3: linear_b(cfg.hidden_size, ffn, false, f.pp("w3"))? },
                    op_norm: rms_norm(cfg.hidden_size, cfg.norm_eps, l.pp("operator_norm"))?,
                    ffn_norm: rms_norm(cfg.hidden_size, cfg.norm_eps, l.pp("ffn_norm"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rot: Rotary::new(&cfg, vb.device())?,
            norm: rms_norm(cfg.hidden_size, cfg.norm_eps, vb.pp("embedding_norm"))?,
            embed,
            layers,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            max_len: max_len.min(cfg.max_position_embeddings),
            cfg,
        })
    }

    /// The (tied) output table.
    pub fn lm_head(&self) -> &Tensor {
        self.embed.embeddings()
    }

    /// Output-table rows for `ids`: (n, hidden) f32.
    pub fn head_rows(&self, ids: &[u32]) -> Result<Tensor> {
        self.embed.embeddings().index_select(&Tensor::new(ids, &self.device)?, 0)?.to_dtype(DType::F32)
    }

    /// d1-omni's trunk: bidirectional over `embeds` (B, T, hidden), where
    /// row `b` has `lengths[b]` real tokens of which the first `prefix[b]` are
    /// media. Media positions read only media; text reads everything.
    /// Returns the normed hidden states (B, T, hidden) f32.
    pub fn encode(&self, embeds: &Tensor, lengths: &[usize], prefix: &[usize]) -> Result<Tensor> {
        let (b, t, _) = embeds.dims3()?;
        let neg = -1e9f32;
        let mut mask = vec![0f32; b * t * t];
        let mut pad = vec![0f32; b * t];
        let mut keep_right = vec![1f32; b * t];
        for r in 0..b {
            for i in 0..t {
                if i < lengths[r] {
                    pad[r * t + i] = 1.0;
                }
                for j in 0..t {
                    let hidden = j >= lengths[r] || (i < prefix[r] && j >= prefix[r]);
                    if hidden {
                        mask[(r * t + i) * t + j] = neg;
                    }
                }
            }
            if prefix[r] > 0 {
                keep_right[r * t + prefix[r] - 1] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask, (b, 1, t, t), &self.device)?.to_dtype(self.dtype)?;
        let pad = Tensor::from_vec(pad, (b, t, 1), &self.device)?.to_dtype(self.dtype)?;
        let keep_right = Tensor::from_vec(keep_right, (b, 1, t), &self.device)?.to_dtype(self.dtype)?;
        let pos: Vec<u32> = (0..t as u32).collect();
        let (cos, sin) = self.rot.cos_sin(&pos, self.dtype, &self.device)?;
        let mut x = embeds.to_dtype(self.dtype)?;
        for layer in &self.layers {
            let h = layer.op_norm.forward(&x)?;
            let h = match &layer.op {
                Operator::Conv(c) => c.centred(&h, &pad, &keep_right)?,
                Operator::Attn(a) => a.forward(&h, &cos, &sin, &mask, None)?.0,
            };
            x = (x + h)?;
            let h = layer.mlp.forward(&layer.ffn_norm.forward(&x)?)?;
            x = (x + h)?;
        }
        self.norm.forward(&x)?.to_dtype(DType::F32)
    }
}

impl Decoder for Lfm2 {
    fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn max_len(&self) -> usize {
        self.max_len
    }
    fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        self.embed.embeddings().index_select(&Tensor::new(ids, &self.device)?, 0)
    }

    fn forward(&self, embeds: &Tensor, pos: &Positions, state: Option<&PrefixState>, keep: bool, attn: &Attn) -> Result<(Tensor, Option<PrefixState>)> {
        let (b, t, _) = embeds.dims3()?;
        if !matches!(attn, Attn::Causal) {
            bail!("LFM2 runs causally; use `encode` for the bidirectional trunk");
        }
        if keep && b != 1 {
            bail!("a prefix state is kept for one sequence at a time");
        }
        let past = state.map_or(0, |s| s.len);
        let (cos, sin) = self.rot.cos_sin(&pos.text, self.dtype, &self.device)?;
        let mask = causal_mask(t, past, self.dtype, &self.device)?;
        let mut x = embeds.to_dtype(self.dtype)?;
        let mut new_states = Vec::with_capacity(self.layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            let prev = match state {
                Some(s) => Some(s.layers[i].broadcast(b)?),
                None => None,
            };
            let h = layer.op_norm.forward(&x)?;
            let (h, st) = match &layer.op {
                Operator::Conv(c) => c.forward(&h, prev.as_ref())?,
                Operator::Attn(a) => a.forward(&h, &cos, &sin, &mask, prev.as_ref())?,
            };
            if keep {
                new_states.push(st);
            }
            x = (x + h)?;
            let h = layer.mlp.forward(&layer.ffn_norm.forward(&x)?)?;
            x = (x + h)?;
        }
        let hidden = self.norm.forward(&x)?.to_dtype(DType::F32)?;
        let state = keep.then(|| PrefixState { layers: new_states, len: past + t, next_pos: pos.next() });
        Ok((hidden, state))
    }
}

/// Whether the MLP weights were sized by HF's adjust rule (read from `w1`).
pub fn ffn_adjust_from_shape(cfg: &Config, w1_rows: usize) -> bool {
    w1_rows != cfg.intermediate_size && w1_rows == cfg.ffn_dim(true)
}

#[allow(dead_code)]
fn _unused(_: D) {}
