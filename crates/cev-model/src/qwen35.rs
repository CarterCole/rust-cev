//! Qwen3.5 text decoder: a hybrid of Gated DeltaNet (linear attention) layers
//! and gated softmax-attention layers, prefill-only, with an explicit prefix
//! state (see [`crate::decoder`]).
//!
//! What differs from Qwen3 ([`crate::qwen3`]):
//!
//! * three of every four layers are Gated DeltaNet: a short causal
//!   convolution over q/k/v, then the chunked gated delta rule (in f32), a
//!   gated RMSNorm and an output projection. Their state is a (heads × dk ×
//!   dv) matrix plus the last three conv inputs, not keys and values;
//! * attention layers project a sigmoid gate next to the queries, apply RoPE
//!   to the first quarter of each head (multi-axis, interleaved, for images)
//!   and gate the output before `o_proj`;
//! * RMSNorm weights are zero-centred: the scale is `1 + w`.
//!
//! Because the recurrent layers cannot honour a packed block-causal mask,
//! questions run as rows of a batch continuing from the prefix state, each
//! right-padded to the longest. Matches HF transformers `Qwen3_5TextModel`.

use crate::decoder::{Attn, Decoder, LayerState, Positions, PrefixState};
use candle_core::{D, DType, Device, Module, Result, Tensor, bail};
use candle_nn::{Embedding, Linear, VarBuilder, linear_b, linear_no_bias};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_theta() -> f64 {
    10_000_000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub layer_types: Vec<String>,
    #[serde(default = "four")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "d128")]
    pub linear_key_head_dim: usize,
    #[serde(default = "d128")]
    pub linear_value_head_dim: usize,
    #[serde(default = "d16")]
    pub linear_num_key_heads: usize,
    #[serde(default = "d16")]
    pub linear_num_value_heads: usize,
    #[serde(default = "eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "yes")]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    /// Top-level `rope_theta` of older configs.
    #[serde(default)]
    pub rope_theta: Option<f64>,
}

fn four() -> usize {
    4
}
fn d128() -> usize {
    128
}
fn d16() -> usize {
    16
}
fn eps() -> f64 {
    1e-6
}
fn max_pos() -> usize {
    262_144
}
fn yes() -> bool {
    true
}

impl TextConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref().map(|r| r.rope_theta).or(self.rope_theta).unwrap_or(10_000_000.0)
    }

    /// Dimensions of each head that RoPE rotates.
    pub fn rotary_dim(&self) -> usize {
        let f = self.rope_parameters.as_ref().and_then(|r| r.partial_rotary_factor).unwrap_or(1.0);
        ((self.head_dim() as f64 * f) as usize) & !1
    }

    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_parameters.as_ref().and_then(|r| r.mrope_section.clone()).unwrap_or_else(|| vec![11, 11, 10])
    }

    pub fn is_linear(&self, layer: usize) -> bool {
        self.layer_types.get(layer).is_some_and(|t| t == "linear_attention")
    }
}

/// `config.json` of a Qwen3.5 checkpoint: multimodal (`qwen3_5`, with
/// `text_config`) or text-only (`qwen3_5_text`, fields at the top level).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_type: String,
    #[serde(default)]
    pub text_config: Option<TextConfig>,
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub vision_start_token_id: Option<u32>,
    #[serde(default)]
    pub vision_end_token_id: Option<u32>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

impl Config {
    pub fn parse(config: &[u8]) -> anyhow::Result<(Config, TextConfig)> {
        let cfg: Config = serde_json::from_slice(config)?;
        let mut text = match &cfg.text_config {
            Some(t) => t.clone(),
            None => serde_json::from_slice(config)?,
        };
        if let Some(t) = cfg.tie_word_embeddings {
            text.tie_word_embeddings |= t;
        }
        if text.layer_types.len() != text.num_hidden_layers {
            anyhow::bail!("layer_types has {} entries for {} layers", text.layer_types.len(), text.num_hidden_layers);
        }
        Ok((cfg, text))
    }
}

/// RMSNorm with a zero-centred weight: `x · rsqrt(mean x² + ε) · (1 + w)`.
struct RmsNorm {
    scale: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let w = vb.get(dim, "weight")?;
        let scale = (w.to_dtype(DType::F32)? + 1.0)?.to_dtype(w.dtype())?;
        Ok(Self { scale, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(&x.contiguous()?, &self.scale, self.eps as f32)
    }
}

/// Multi-axis RoPE over the first `rotary_dim` dims of a head. Frequency `i`
/// takes its position from the temporal, height or width axis in an
/// interleaved pattern (text tokens have all three equal).
struct Rotary {
    inv: Tensor,
    /// (3, 1, half) one-hot: which axis each frequency reads.
    axis: Tensor,
}

impl Rotary {
    fn new(cfg: &TextConfig, dev: &Device) -> Result<Self> {
        let dim = cfg.rotary_dim();
        let half = dim / 2;
        let inv: Vec<f32> = (0..dim).step_by(2).map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / dim as f64) as f32).collect();
        let section = cfg.mrope_section();
        let interleaved = cfg.rope_parameters.as_ref().and_then(|r| r.mrope_interleaved).unwrap_or(true);
        let mut axis = vec![0f32; 3 * half];
        for i in 0..half {
            let a = if interleaved {
                if i % 3 == 1 && i < section.get(1).copied().unwrap_or(0) * 3 {
                    1
                } else if i % 3 == 2 && i < section.get(2).copied().unwrap_or(0) * 3 {
                    2
                } else {
                    0
                }
            } else {
                // Contiguous sections: t, then h, then w.
                let (t, h) = (section.first().copied().unwrap_or(half), section.get(1).copied().unwrap_or(0));
                if i < t {
                    0
                } else if i < t + h {
                    1
                } else {
                    2
                }
            };
            axis[a * half + i] = 1.0;
        }
        Ok(Self { inv: Tensor::from_vec(inv, (1, 1, half), dev)?, axis: Tensor::from_vec(axis, (3, 1, half), dev)? })
    }

    /// cos/sin (T, half) for the given positions.
    fn cos_sin(&self, pos: &Positions, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
        let t = pos.len();
        let axes = pos.axes();
        let p: Vec<f32> = axes.iter().flat_map(|a| a.iter().map(|&v| v as f32)).collect();
        let p = Tensor::from_vec(p, (3, t, 1), dev)?;
        let freqs = p.broadcast_mul(&self.inv)?.broadcast_mul(&self.axis)?.sum(0)?; // (T, half)
        Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
    }

    /// Rotate the first `2 * half` dims of x (B, H, T, D).
    fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = x.dim(D::Minus1)?;
        let rot = 2 * cos.dim(D::Minus1)?;
        let x = x.contiguous()?;
        if rot == d {
            return candle_nn::rotary_emb::rope(&x, cos, sin);
        }
        let head = candle_nn::rotary_emb::rope(&x.narrow(D::Minus1, 0, rot)?.contiguous()?, cos, sin)?;
        Tensor::cat(&[head, x.narrow(D::Minus1, rot, d - rot)?], D::Minus1)
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    // max(x, 0) + ln(1 + e^{-|x|}): stable for large |x|.
    let pos = x.relu()?;
    let neg = x.abs()?.neg()?.exp()?;
    pos + (neg + 1.0)?.log()?
}

/// L2 normalisation over the last dim, FLA style: x · rsqrt(Σx² + ε).
fn l2norm(x: &Tensor) -> Result<Tensor> {
    let n = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    x.broadcast_div(&n)
}

/// Inverse of `I + L` where `L` is the strictly lower triangle of the last two
/// dims of `m` (…, n, n), by recursive block inversion:
///
/// ```text
///   [A 0]^-1   [ A^-1              0    ]
///   [C B]    = [ -B^-1 C A^-1    B^-1 ]
/// ```
///
/// The two diagonal blocks of a level are inverted together, so a 64×64
/// inverse is six levels of a few batched matmuls instead of a 63-step scan.
fn inv_unit_lower(m: &Tensor) -> Result<Tensor> {
    let dims = m.dims().to_vec();
    let n = dims[dims.len() - 1];
    let lead: usize = dims[..dims.len() - 2].iter().product();
    let m = m.reshape((lead, n, n))?;
    let strict = m.broadcast_mul(&Tensor::tril2(n, m.dtype(), m.device())?.sub(&Tensor::eye(n, m.dtype(), m.device())?)?)?;
    let inv = inv_unit_lower_strict(&strict)?;
    inv.reshape(dims)
}

/// `m`: (M, n, n) strictly lower triangular. Returns (I + m)^-1.
fn inv_unit_lower_strict(m: &Tensor) -> Result<Tensor> {
    let (b, n, _) = m.dims3()?;
    if n == 1 {
        return Tensor::ones((b, 1, 1), m.dtype(), m.device());
    }
    let h = n / 2;
    let a = m.narrow(1, 0, h)?.narrow(2, 0, h)?;
    let c = m.narrow(1, h, n - h)?.narrow(2, 0, h)?;
    let d = m.narrow(1, h, n - h)?.narrow(2, h, n - h)?;
    let (ai, di) = if n - h == h {
        let both = inv_unit_lower_strict(&Tensor::cat(&[&a, &d], 0)?.contiguous()?)?;
        (both.narrow(0, 0, b)?, both.narrow(0, b, b)?)
    } else {
        (inv_unit_lower_strict(&a.contiguous()?)?, inv_unit_lower_strict(&d.contiguous()?)?)
    };
    let ll = di.matmul(&c.contiguous()?)?.matmul(&ai)?.neg()?;
    let top = Tensor::cat(&[&ai, &Tensor::zeros((b, h, n - h), m.dtype(), m.device())?], 2)?;
    let bottom = Tensor::cat(&[&ll, &di], 2)?;
    Tensor::cat(&[top, bottom], 1)
}

/// The chunked gated delta rule (FLA / HF `torch_chunk_gated_delta_rule`),
/// all in f32. `q`, `k`: (B, H, T, dk); `v`: (B, H, T, dv); `g`, `beta`:
/// (B, H, T); `s0`: (B, H, dk, dv). Returns (B, H, T, dv) and the new state.
fn chunk_gated_delta_rule(q: &Tensor, k: &Tensor, v: &Tensor, g: &Tensor, beta: &Tensor, s0: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
    const CHUNK: usize = 64;
    let (b, h, t, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let dev = q.device();
    let q = (l2norm(q)? * (dk as f64).powf(-0.5))?;
    let k = l2norm(k)?;
    let pad = (CHUNK - t % CHUNK) % CHUNK;
    let tp = t + pad;
    let n = tp / CHUNK;
    let padt = |x: &Tensor| -> Result<Tensor> { if pad == 0 { Ok(x.clone()) } else { x.pad_with_zeros(2, 0, pad) } };
    let (q, k, v, g, beta) = (padt(&q)?, padt(&k)?, padt(v)?, padt(g)?, padt(beta)?);
    let beta4 = beta.unsqueeze(3)?; // (B,H,T,1)
    let v_beta = v.broadcast_mul(&beta4)?.reshape((b, h, n, CHUNK, dv))?.contiguous()?;
    let k_beta = k.broadcast_mul(&beta4)?.reshape((b, h, n, CHUNK, dk))?.contiguous()?;
    let q = q.contiguous()?.reshape((b, h, n, CHUNK, dk))?;
    let k = k.contiguous()?.reshape((b, h, n, CHUNK, dk))?;
    let g = g.contiguous()?.reshape((b, h, n, CHUNK))?;
    let cum = g.cumsum(3)?; // (B,H,N,C)
    // pair[i, j] = exp(cum_i - cum_j) for j <= i, else 0. The upper triangle
    // is pushed to -inf *before* the exp: those differences are positive and
    // can overflow, and inf · 0 would be NaN.
    let pair = cum.unsqueeze(4)?.broadcast_sub(&cum.unsqueeze(3)?)?;
    let lower = Tensor::tril2(CHUNK, DType::F32, dev)?.reshape((1, 1, 1, CHUNK, CHUNK))?;
    let upper_penalty = ((lower.ones_like()? - &lower)? * -1e30)?;
    let pair = pair.broadcast_add(&upper_penalty)?.exp()?;
    // candle's batched matmul wants a flat batch dim: (B·H·N, C, d).
    let flat = |x: &Tensor| -> Result<Tensor> {
        let d = x.dim(4)?;
        x.contiguous()?.reshape((b * h * n, CHUNK, d))
    };
    let (qf, kf, kbf, vbf) = (flat(&q)?, flat(&k)?, flat(&k_beta)?, flat(&v_beta)?);
    let pair = pair.reshape((b * h * n, CHUNK, CHUNK))?;
    let kt = kf.transpose(1, 2)?.contiguous()?;
    let ut = kbf.matmul(&kt)?.mul(&pair)?;
    let intra = qf.matmul(&kt)?.mul(&pair)?.reshape((b, h, n, CHUNK, CHUNK))?;
    let ecum = cum.exp()?.unsqueeze(4)?; // (B,H,N,C,1)
    let decayed_k_beta = flat(&k_beta.broadcast_mul(&ecum)?)?;
    let inv = inv_unit_lower(&ut)?.contiguous()?; // (B·H·N,C,C)
    let new_values = inv.matmul(&vbf)?.reshape((b, h, n, CHUNK, dv))?;
    let k_cumdecay = inv.matmul(&decayed_k_beta)?.reshape((b, h, n, CHUNK, dk))?;
    let q = q.broadcast_mul(&ecum)?.contiguous()?;
    let last = cum.narrow(3, CHUNK - 1, 1)?; // (B,H,N,1)
    let k = k.broadcast_mul(&last.broadcast_sub(&cum)?.exp()?.unsqueeze(4)?)?.contiguous()?;
    let chunk_decay = last.exp()?.unsqueeze(4)?; // (B,H,N,1,1)
    let s = match s0 {
        Some(s) => s.to_dtype(DType::F32)?.contiguous()?,
        None => Tensor::zeros((b, h, dk, dv), DType::F32, dev)?,
    };
    // Sequential scan over chunks, each step on (B·H, C, ·) tensors.
    let bh = b * h;
    let chunk3 = |x: &Tensor, i: usize| -> Result<Tensor> {
        let d = x.dim(4)?;
        x.narrow(2, i, 1)?.squeeze(2)?.contiguous()?.reshape((bh, CHUNK, d))
    };
    let mut s = s.reshape((bh, dk, dv))?;
    let mut outs = Vec::with_capacity(n);
    for i in 0..n {
        let v_new = chunk3(&new_values, i)?.sub(&chunk3(&k_cumdecay, i)?.matmul(&s)?)?;
        let inter = chunk3(&q, i)?.matmul(&s)?;
        let out = inter.add(&chunk3(&intra, i)?.matmul(&v_new)?)?;
        outs.push(out.reshape((b, h, 1, CHUNK, dv))?);
        let decay = chunk_decay.narrow(2, i, 1)?.reshape((bh, 1, 1))?;
        s = s.broadcast_mul(&decay)?.add(&chunk3(&k, i)?.transpose(1, 2)?.contiguous()?.matmul(&v_new)?)?;
    }
    let out = Tensor::cat(&outs, 2)?.reshape((b, h, tp, dv))?;
    Ok((out.narrow(2, 0, t)?, s.reshape((b, h, dk, dv))?))
}

struct GatedDeltaNet {
    in_qkv: Linear,
    in_z: Linear,
    in_b: Linear,
    in_a: Linear,
    conv: Tensor,
    /// -exp(A_log), f32 (Hv).
    neg_a: Tensor,
    dt_bias: Tensor,
    norm_w: Tensor,
    out: Linear,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    kernel: usize,
    eps: f64,
}

impl GatedDeltaNet {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let (nk, nv, dk, dv) = (cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim);
        let (key_dim, value_dim) = (nk * dk, nv * dv);
        let conv_dim = 2 * key_dim + value_dim;
        let conv = vb.get((conv_dim, 1, cfg.linear_conv_kernel_dim), "conv1d.weight")?;
        Ok(Self {
            in_qkv: linear_no_bias(cfg.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?,
            in_z: linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?,
            in_b: linear_no_bias(cfg.hidden_size, nv, vb.pp("in_proj_b"))?,
            in_a: linear_no_bias(cfg.hidden_size, nv, vb.pp("in_proj_a"))?,
            neg_a: vb.get(nv, "A_log")?.to_dtype(DType::F32)?.exp()?.neg()?,
            dt_bias: vb.get(nv, "dt_bias")?.to_dtype(DType::F32)?,
            norm_w: vb.get(dv, "norm.weight")?.to_dtype(DType::F32)?,
            out: linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?,
            conv,
            nk,
            nv,
            dk,
            dv,
            kernel: cfg.linear_conv_kernel_dim,
            eps: cfg.rms_norm_eps,
        })
    }

    /// x: (B, T, hidden). Returns the output and the new (conv, s) state.
    fn forward(&self, x: &Tensor, state: Option<&LayerState>) -> Result<(Tensor, LayerState)> {
        let (b, t, _) = x.dims3()?;
        let (key_dim, value_dim) = (self.nk * self.dk, self.nv * self.dv);
        let conv_dim = 2 * key_dim + value_dim;
        let mixed = self.in_qkv.forward(x)?.transpose(1, 2)?.contiguous()?; // (B, C, T)
        let z = self.in_z.forward(x)?;
        let bb = self.in_b.forward(x)?.to_dtype(DType::F32)?; // (B,T,Hv)
        let aa = self.in_a.forward(x)?.to_dtype(DType::F32)?;
        let (conv_state, s0) = match state {
            Some(LayerState::Gdn { conv, s }) => (conv.clone(), Some(s)),
            Some(_) => bail!("wrong state kind for a Gated DeltaNet layer"),
            None => (Tensor::zeros((b, conv_dim, self.kernel - 1), mixed.dtype(), mixed.device())?, None),
        };
        let full = Tensor::cat(&[&conv_state, &mixed], 2)?; // (B, C, T + K - 1)
        let new_conv = full.narrow(2, t, self.kernel - 1)?.contiguous()?;
        let conved = candle_nn::ops::silu(&crate::decoder::depthwise_conv1d(&full, &self.conv)?)?; // (B, C, T)
        let conved = conved.transpose(1, 2)?.contiguous()?; // (B, T, C)
        let heads = |x: Tensor, h: usize, d: usize| -> Result<Tensor> { x.reshape((b, t, h, d))?.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32) };
        let q = heads(conved.narrow(2, 0, key_dim)?, self.nk, self.dk)?;
        let k = heads(conved.narrow(2, key_dim, key_dim)?, self.nk, self.dk)?;
        let v = heads(conved.narrow(2, 2 * key_dim, value_dim)?, self.nv, self.dv)?;
        let (q, k) = if self.nv > self.nk {
            let g = self.nv / self.nk;
            let rep = |x: &Tensor| -> Result<Tensor> { x.unsqueeze(2)?.expand((b, self.nk, g, t, self.dk))?.reshape((b, self.nv, t, self.dk)) };
            (rep(&q)?, rep(&k)?)
        } else {
            (q, k)
        };
        let beta = candle_nn::ops::sigmoid(&bb)?.transpose(1, 2)?.contiguous()?; // (B,Hv,T)
        let g = softplus(&aa.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.neg_a)?.transpose(1, 2)?.contiguous()?;
        let (core, s) = chunk_gated_delta_rule(&q, &k, &v, &g, &beta, s0)?;
        // Gated RMSNorm per head: norm(o) * w * silu(z).
        let core = core.transpose(1, 2)?.contiguous()?; // (B,T,Hv,dv) f32
        let var = core.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = core.broadcast_div(&(var + self.eps)?.sqrt()?)?.broadcast_mul(&self.norm_w)?;
        let z = candle_nn::ops::silu(&z.to_dtype(DType::F32)?.reshape((b, t, self.nv, self.dv))?)?;
        let y = normed.mul(&z)?.reshape((b, t, value_dim))?.to_dtype(x.dtype())?;
        Ok((self.out.forward(&y)?, LayerState::Gdn { conv: new_conv, s }))
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
    gate: bool,
}

/// Additive mask (1, 1, T, past + T): branch tokens see every past token and
/// earlier tokens of their own row.
fn branch_mask(t: usize, past: usize, dtype: DType, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; t * (past + t)];
    for i in 0..t {
        for j in i + 1..t {
            m[i * (past + t) + past + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(m, (1, 1, t, past + t), dev)?.to_dtype(dtype)
}

/// Additive mask (B, 1, T, T) for bidirectional rows: every real token of a
/// row sees every other real token of it; padding columns are hidden.
fn bidirectional_mask(t: usize, lengths: &[usize], dtype: DType, dev: &Device) -> Result<Tensor> {
    let b = lengths.len();
    let mut m = vec![0f32; b * t * t];
    for (r, &len) in lengths.iter().enumerate() {
        for i in 0..t {
            for j in len..t {
                m[(r * t + i) * t + j] = f32::NEG_INFINITY;
            }
            // A fully padded query row keeps its diagonal so softmax stays finite.
            if i >= len {
                m[(r * t + i) * t + i] = 0.0;
            }
        }
    }
    Tensor::from_vec(m, (b, 1, t, t), dev)?.to_dtype(dtype)
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, h, s, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, h, groups, s, d))?.reshape((b, h * groups, s, d))
}

impl Attention {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let (h, kvh, d, hs) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim(), cfg.hidden_size);
        let qo = if cfg.attn_output_gate { 2 * h * d } else { h * d };
        Ok(Self {
            q: linear_b(hs, qo, cfg.attention_bias, vb.pp("q_proj"))?,
            k: linear_b(hs, kvh * d, cfg.attention_bias, vb.pp("k_proj"))?,
            v: linear_b(hs, kvh * d, cfg.attention_bias, vb.pp("v_proj"))?,
            o: linear_b(h * d, hs, cfg.attention_bias, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            heads: h,
            kv_heads: kvh,
            head_dim: d,
            gate: cfg.attn_output_gate,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, state: Option<&LayerState>, attn: &Attn) -> Result<(Tensor, LayerState)> {
        let (b, t, _) = x.dims3()?;
        let (h, d) = (self.heads, self.head_dim);
        let qg = self.q.forward(x)?;
        let (q, gate) = if self.gate {
            let qg = qg.reshape((b, t, h, 2 * d))?;
            (qg.narrow(3, 0, d)?, Some(qg.narrow(3, d, d)?.reshape((b, t, h * d))?))
        } else {
            (qg.reshape((b, t, h, d))?, None)
        };
        let q = self.q_norm.forward(&q.transpose(1, 2)?.contiguous()?)?;
        let k = self.k_norm.forward(&self.k.forward(x)?.reshape((b, t, self.kv_heads, d))?.transpose(1, 2)?.contiguous()?)?;
        let v = self.v.forward(x)?.reshape((b, t, self.kv_heads, d))?.transpose(1, 2)?.contiguous()?;
        let (q, k) = (Rotary::apply(&q, cos, sin)?, Rotary::apply(&k, cos, sin)?);
        let (k, v, past) = match state {
            Some(LayerState::Attn { k: pk, v: pv }) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?, pk.dim(2)?),
            Some(_) => bail!("wrong state kind for an attention layer"),
            None => (k, v, 0),
        };
        let mask = match attn {
            Attn::Causal => branch_mask(t, past, q.dtype(), q.device())?,
            Attn::Bidirectional { lengths } => {
                if past != 0 {
                    bail!("bidirectional attention cannot continue a prefix state");
                }
                bidirectional_mask(t, lengths, q.dtype(), q.device())?
            }
        };
        let ctx = self.attend(&q, &k, &v, &mask)?; // (B,H,T,D)
        let mut out = ctx.transpose(1, 2)?.reshape((b, t, h * d))?;
        if let Some(g) = gate {
            out = out.mul(&candle_nn::ops::sigmoid(&g)?)?;
        }
        Ok((self.o.forward(&out)?, LayerState::Attn { k, v }))
    }

    /// `mask`: additive, (1 or B, 1, T, S).
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let (b, h, t, _) = q.dims4()?;
        let s = k.dim(2)?;
        // candle's fused Metal kernel has no f32 variant for 256-wide heads
        // (Qwen3.5's), so f32 runs take the plain path there.
        let fused = q.device().is_metal() && (matches!(self.head_dim, 64 | 128) || (self.head_dim == 256 && q.dtype() != DType::F32));
        if fused {
            let m = mask.broadcast_as((b, h, t, s))?;
            return candle_nn::ops::sdpa(q, k, v, Some(&m), false, scale as f32, 1.0);
        }
        let g = self.heads / self.kv_heads;
        let (k, v) = (repeat_kv(k, g)?, repeat_kv(v, g)?);
        let scores = (q.matmul(&k.t()?.contiguous()?)?.to_dtype(DType::F32)? * scale)?.broadcast_add(&mask.to_dtype(DType::F32)?)?;
        let p = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
        p.matmul(&v.contiguous()?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        self.down.forward(&(g * self.up.forward(x)?)?)
    }
}

enum Mixer {
    Linear(GatedDeltaNet),
    Attn(Attention),
}

struct Layer {
    mixer: Mixer,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Layer {
    fn new(cfg: &TextConfig, i: usize, vb: VarBuilder) -> Result<Self> {
        let (hs, is) = (cfg.hidden_size, cfg.intermediate_size);
        let mixer = if cfg.is_linear(i) { Mixer::Linear(GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?) } else { Mixer::Attn(Attention::new(cfg, vb.pp("self_attn"))?) };
        let m = vb.pp("mlp");
        Ok(Self {
            mixer,
            mlp: Mlp {
                gate: linear_no_bias(hs, is, m.pp("gate_proj"))?,
                up: linear_no_bias(hs, is, m.pp("up_proj"))?,
                down: linear_no_bias(is, hs, m.pp("down_proj"))?,
            },
            ln1: RmsNorm::new(hs, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: RmsNorm::new(hs, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }
}

pub struct Qwen35 {
    embed: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Tensor,
    rot: Rotary,
    pub cfg: TextConfig,
    device: Device,
    dtype: DType,
    max_len: usize,
}

impl Qwen35 {
    /// `vb` is rooted at the text model (`model.language_model`,
    /// `language_model` or `model`); `lm_head` is the untied output table
    /// when the checkpoint has one.
    pub fn new(cfg: TextConfig, vb: VarBuilder, lm_head: Option<VarBuilder>, max_len: usize) -> Result<Self> {
        let embed = Embedding::new(vb.pp("embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?, cfg.hidden_size);
        let lm_head = match lm_head {
            Some(h) if !cfg.tie_word_embeddings => h.get((cfg.vocab_size, cfg.hidden_size), "weight")?,
            _ => embed.embeddings().clone(),
        };
        let layers = (0..cfg.num_hidden_layers).map(|i| Layer::new(&cfg, i, vb.pp("layers").pp(i))).collect::<Result<_>>()?;
        Ok(Self {
            rot: Rotary::new(&cfg, vb.device())?,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            embed,
            layers,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            max_len: max_len.min(cfg.max_position_embeddings),
            cfg,
        })
    }

    /// Output-table rows for `ids`: (n, hidden) f32.
    pub fn head_rows(&self, ids: &[u32]) -> Result<Tensor> {
        self.lm_head.index_select(&Tensor::new(ids, &self.device)?, 0)?.to_dtype(DType::F32)
    }

    /// The whole output table (vocab, hidden) in the model dtype.
    pub fn lm_head(&self) -> &Tensor {
        &self.lm_head
    }
}

impl Decoder for Qwen35 {
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
        if pos.len() != t {
            bail!("{t} tokens but {} positions", pos.len());
        }
        if keep && b != 1 {
            bail!("a prefix state is kept for one sequence at a time");
        }
        if let Attn::Bidirectional { lengths } = attn {
            if lengths.len() != b || keep || state.is_some() {
                bail!("bidirectional rows take one length per row and no prefix state");
            }
        }
        let (cos, sin) = self.rot.cos_sin(pos, self.dtype, &self.device)?;
        let mut x = embeds.to_dtype(self.dtype)?;
        let mut new_states = Vec::with_capacity(self.layers.len());
        let profile = std::env::var_os("CEV_PROFILE").is_some();
        let (mut t_gdn, mut t_attn, mut t_mlp) = (0f64, 0f64, 0f64);
        let mut clock = web_time::Instant::now();
        for (i, layer) in self.layers.iter().enumerate() {
            let past = match state {
                Some(s) => Some(s.layers[i].broadcast(b)?),
                None => None,
            };
            let h = layer.ln1.forward(&x)?;
            let (h, st) = match &layer.mixer {
                Mixer::Linear(g) => g.forward(&h, past.as_ref())?,
                Mixer::Attn(a) => a.forward(&h, &cos, &sin, past.as_ref(), attn)?,
            };
            if keep {
                new_states.push(st);
            }
            x = (x + h)?;
            if profile {
                self.device.synchronize()?;
                let dt = clock.elapsed().as_secs_f64() * 1e3;
                if self.cfg.is_linear(i) { t_gdn += dt } else { t_attn += dt }
                clock = web_time::Instant::now();
            }
            let h = layer.mlp.forward(&layer.ln2.forward(&x)?)?;
            x = (x + h)?;
            if profile {
                self.device.synchronize()?;
                t_mlp += clock.elapsed().as_secs_f64() * 1e3;
                clock = web_time::Instant::now();
            }
        }
        if profile {
            eprintln!("qwen35 forward b={b} t={t} past={}: gdn {t_gdn:.1} ms, attn {t_attn:.1} ms, mlp {t_mlp:.1} ms", state.map_or(0, |s| s.len));
        }
        let hidden = self.norm.forward(&x)?.to_dtype(DType::F32)?;
        let state = keep.then(|| PrefixState { layers: new_states, len: state.map_or(0, |s| s.len) + t, next_pos: pos.next() });
        Ok((hidden, state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_lower_inverse_matches_definition() {
        let dev = Device::Cpu;
        let n = 8;
        let vals: Vec<f32> = (0..n * n).map(|i| ((i * 37 % 11) as f32 - 5.0) / 7.0).collect();
        let m = Tensor::from_vec(vals, (1, 1, n, n), &dev).unwrap();
        let strict = m.broadcast_mul(&Tensor::tril2(n, DType::F32, &dev).unwrap().sub(&Tensor::eye(n, DType::F32, &dev).unwrap()).unwrap()).unwrap();
        let a = strict.add(&Tensor::eye(n, DType::F32, &dev).unwrap().reshape((1, 1, n, n)).unwrap()).unwrap();
        let inv = inv_unit_lower(&m).unwrap();
        let id = a.matmul(&inv).unwrap().reshape((n, n)).unwrap().to_vec2::<f32>().unwrap();
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((id[i][j] - want).abs() < 1e-4, "({i},{j}) = {}", id[i][j]);
            }
        }
    }

    /// The chunked rule must equal the token-by-token recurrence.
    #[test]
    fn chunked_rule_matches_recurrence() {
        let dev = Device::Cpu;
        let (b, h, t, dk, dv) = (1, 2, 70, 4, 3);
        let f = |n: usize, seed: u32| -> Vec<f32> { (0..n).map(|i| (((i as u32 + 1) * (seed + 7919)) % 1000) as f32 / 500.0 - 1.0).collect() };
        let q = Tensor::from_vec(f(b * h * t * dk, 1), (b, h, t, dk), &dev).unwrap();
        let k = Tensor::from_vec(f(b * h * t * dk, 2), (b, h, t, dk), &dev).unwrap();
        let v = Tensor::from_vec(f(b * h * t * dv, 3), (b, h, t, dv), &dev).unwrap();
        let g = (Tensor::from_vec(f(b * h * t, 4), (b, h, t), &dev).unwrap().abs().unwrap() * -0.5).unwrap();
        let beta = candle_nn::ops::sigmoid(&Tensor::from_vec(f(b * h * t, 5), (b, h, t), &dev).unwrap()).unwrap();
        let (out, s) = chunk_gated_delta_rule(&q, &k, &v, &g, &beta, None).unwrap();
        // Reference recurrence.
        let qn = (l2norm(&q).unwrap() * (dk as f64).powf(-0.5)).unwrap().to_vec3_flat(b, h, t, dk);
        let kn = l2norm(&k).unwrap().to_vec3_flat(b, h, t, dk);
        let vv = v.to_vec3_flat(b, h, t, dv);
        let gg: Vec<f32> = g.flatten_all().unwrap().to_vec1().unwrap();
        let bb: Vec<f32> = beta.flatten_all().unwrap().to_vec1().unwrap();
        let got: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let sgot: Vec<f32> = s.flatten_all().unwrap().to_vec1().unwrap();
        for hh in 0..h {
            let mut s = vec![0f32; dk * dv];
            for ti in 0..t {
                let at = |x: &Vec<f32>, d: usize, i: usize| x[(hh * t + ti) * d + i];
                let gt = gg[hh * t + ti].exp();
                for x in s.iter_mut() {
                    *x *= gt;
                }
                let bt = bb[hh * t + ti];
                let mem: Vec<f32> = (0..dv).map(|j| (0..dk).map(|i| s[i * dv + j] * at(&kn, dk, i)).sum()).collect();
                let delta: Vec<f32> = (0..dv).map(|j| (at(&vv, dv, j) - mem[j]) * bt).collect();
                for i in 0..dk {
                    for j in 0..dv {
                        s[i * dv + j] += at(&kn, dk, i) * delta[j];
                    }
                }
                for j in 0..dv {
                    let o: f32 = (0..dk).map(|i| s[i * dv + j] * at(&qn, dk, i)).sum();
                    let g = got[(hh * t + ti) * dv + j];
                    assert!((o - g).abs() < 2e-4, "head {hh} t {ti} dim {j}: {o} vs {g}");
                }
            }
            for (i, x) in s.iter().enumerate() {
                assert!((x - sgot[hh * dk * dv + i]).abs() < 2e-4);
            }
        }
    }

    trait Flat {
        fn to_vec3_flat(&self, b: usize, h: usize, t: usize, d: usize) -> Vec<f32>;
    }
    impl Flat for Tensor {
        fn to_vec3_flat(&self, b: usize, h: usize, t: usize, d: usize) -> Vec<f32> {
            assert_eq!(b, 1);
            self.reshape((h * t * d,)).unwrap().to_vec1().unwrap()
        }
    }
}
