//! What every prefill-only decoder in cev-model exposes: embed tokens, run a
//! batch of sequences that continue a cached prefix, and return the final
//! normed hidden states. The hybrid engine ([`crate::hybrid`]) drives any
//! implementation the same way: the state prefix once, then every question
//! as one row of a batch continuing from it.

use candle_core::{DType, Device, Result, Tensor};

/// Position ids of a sequence: the text position of every token, and for
/// multimodal RoPE the temporal / height / width positions. A text-only
/// sequence has all four equal.
#[derive(Debug, Clone)]
pub struct Positions {
    pub text: Vec<u32>,
    pub thw: Option<[Vec<u32>; 3]>,
}

impl Positions {
    pub fn text(start: u32, len: usize) -> Self {
        Self { text: (start..start + len as u32).collect(), thw: None }
    }

    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The three rotary axes (equal to the text positions when not multimodal).
    pub fn axes(&self) -> [&[u32]; 3] {
        match &self.thw {
            Some([t, h, w]) => [t, h, w],
            None => [&self.text, &self.text, &self.text],
        }
    }

    /// The position the next token after this sequence takes.
    pub fn next(&self) -> u32 {
        self.axes().iter().flat_map(|a| a.iter().copied()).max().map_or(0, |m| m + 1)
    }

    /// Positions `..n` of this sequence.
    pub fn slice(&self, range: std::ops::Range<usize>) -> Positions {
        Positions { text: self.text[range.clone()].to_vec(), thw: self.thw.as_ref().map(|[t, h, w]| [t[range.clone()].to_vec(), h[range.clone()].to_vec(), w[range].to_vec()]) }
    }
}

/// Per-layer state left behind by a prefix: keys and values for softmax
/// attention, the recurrent matrix and the last conv inputs for Gated
/// DeltaNet, the last conv inputs for a short convolution.
#[derive(Clone)]
pub enum LayerState {
    /// (B, kv_heads, S, head_dim) each.
    Attn { k: Tensor, v: Tensor },
    /// `conv`: (B, channels, kernel - 1) last inputs; `s`: (B, heads, dk, dv) f32.
    Gdn { conv: Tensor, s: Tensor },
    /// (B, channels, kernel - 1) last inputs.
    Conv { last: Tensor },
}

impl LayerState {
    pub fn bytes(&self) -> usize {
        let n = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        match self {
            LayerState::Attn { k, v } => n(k) + n(v),
            LayerState::Gdn { conv, s } => n(conv) + n(s),
            LayerState::Conv { last } => n(last),
        }
    }

    /// The same state for every row of a batch of `b`.
    pub fn broadcast(&self, b: usize) -> Result<LayerState> {
        let bc = |t: &Tensor| -> Result<Tensor> {
            if t.dim(0)? == b {
                return Ok(t.clone());
            }
            let mut shape = t.dims().to_vec();
            shape[0] = b;
            t.broadcast_as(shape)?.contiguous()
        };
        Ok(match self {
            LayerState::Attn { k, v } => LayerState::Attn { k: bc(k)?, v: bc(v)? },
            LayerState::Gdn { conv, s } => LayerState::Gdn { conv: bc(conv)?, s: bc(s)? },
            LayerState::Conv { last } => LayerState::Conv { last: bc(last)? },
        })
    }
}

/// Everything a decoder keeps about a computed prefix.
#[derive(Clone)]
pub struct PrefixState {
    pub layers: Vec<LayerState>,
    /// Tokens in the prefix.
    pub len: usize,
    /// Text position the first branch token takes.
    pub next_pos: u32,
}

impl PrefixState {
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(LayerState::bytes).sum()
    }
}

/// How softmax-attention layers see the batch.
#[derive(Debug, Clone)]
pub enum Attn {
    /// Each token sees the prefix and earlier tokens of its own row.
    Causal,
    /// Each token sees every real token of its own row (d3's
    /// `noncausal_full_attention`); `lengths[b]` real tokens per row, the
    /// rest right padding. No prefix state can be used or kept.
    Bidirectional { lengths: Vec<usize> },
}

pub trait Decoder: Send + Sync {
    fn hidden_size(&self) -> usize;
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;
    /// Longest sequence (prefix + branch) the model takes.
    fn max_len(&self) -> usize;
    /// Token embeddings: (n, hidden) in the model dtype.
    fn embed(&self, ids: &[u32]) -> Result<Tensor>;
    /// Run `embeds` (B, T, hidden) at `pos` (T positions shared by the rows,
    /// right-padded rows allowed) after `state` (B = 1 states are broadcast).
    /// Returns the final normed hidden states (B, T, hidden) in f32 and, when
    /// `keep`, the state after these tokens (B must be 1).
    fn forward(&self, embeds: &Tensor, pos: &Positions, state: Option<&PrefixState>, keep: bool, attn: &Attn) -> Result<(Tensor, Option<PrefixState>)>;
}

/// Depthwise 1-D convolution over `x` (B, C, L) with `kernel` (C, 1, K), no
/// padding: (B, C, L - K + 1). Written as K shifted multiply-adds rather
/// than candle's grouped convolution, which launches one kernel per channel
/// (thousands per layer) and is minutes slow on Metal.
pub fn depthwise_conv1d(x: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    let (_, c, l) = x.dims3()?;
    let k = kernel.dim(2)?;
    let out = l + 1 - k;
    let w = kernel.reshape((1, c, k))?.to_dtype(x.dtype())?;
    let mut acc: Option<Tensor> = None;
    for i in 0..k {
        let term = x.narrow(2, i, out)?.broadcast_mul(&w.narrow(2, i, 1)?)?;
        acc = Some(match acc {
            Some(a) => (a + term)?,
            None => term,
        });
    }
    acc.ok_or_else(|| candle_core::Error::Msg("empty kernel".into()))
}
