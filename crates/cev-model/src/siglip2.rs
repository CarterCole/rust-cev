//! SigLIP2 NaFlex vision tower (LFM2-VL, d1, d1-omni): 16×16 patches through
//! a linear embedding plus a 16×16 learned position grid resampled to the
//! image's patch grid, pre-norm attention blocks with GELU-tanh MLPs, and a
//! final LayerNorm. Then LFM2-VL's projector: a 2×2 pixel unshuffle and a
//! two-layer GELU MLP into the language model's width. Matches HF
//! `Siglip2VisionModel` and `Lfm2VlMultiModalProjector`.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder, layer_norm, linear};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "three")]
    pub num_channels: usize,
    #[serde(default = "sixteen")]
    pub patch_size: usize,
    #[serde(default = "n256")]
    pub num_patches: usize,
    #[serde(default = "eps6")]
    pub layer_norm_eps: f64,
}

fn three() -> usize {
    3
}
fn sixteen() -> usize {
    16
}
fn n256() -> usize {
    256
}
fn eps6() -> f64 {
    1e-6
}

impl VisionConfig {
    pub fn patch_dim(&self) -> usize {
        self.num_channels * self.patch_size * self.patch_size
    }
}

/// Weights of `F.interpolate(mode="bilinear", align_corners=False, antialias=True)`.
pub fn resample_weights(inp: usize, out: usize) -> Vec<Vec<(usize, f32)>> {
    crate::media::antialias_weights(inp, out, false)
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, _) = x.dims3()?;
        let shape = |t: Tensor| -> Result<Tensor> { t.reshape((b, n, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous() };
        let q = shape(self.q.forward(x)?)?;
        let k = shape(self.k.forward(x)?)?;
        let v = shape(self.v.forward(x)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let ctx = if x.device().is_metal() && matches!(self.head_dim, 64 | 72 | 96 | 128) {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?
        } else {
            let scores = (q.matmul(&k.t()?.contiguous()?)?.to_dtype(DType::F32)? * scale)?;
            candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?.matmul(&v)?
        };
        self.o.forward(&ctx.transpose(1, 2)?.reshape((b, n, self.heads * self.head_dim))?)
    }
}

struct Layer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    attn: Attention,
    fc1: Linear,
    fc2: Linear,
}

pub struct Siglip2 {
    patch: Linear,
    pos: Tensor,
    layers: Vec<Layer>,
    post: LayerNorm,
    pub cfg: VisionConfig,
    device: Device,
    dtype: DType,
}

impl Siglip2 {
    /// `vb` rooted at `vision_model` (e.g. `model.vision_tower.vision_model`).
    pub fn new(cfg: VisionConfig, vb: VarBuilder) -> Result<Self> {
        let e = vb.pp("embeddings");
        let hd = cfg.hidden_size / cfg.num_attention_heads;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let l = vb.pp("encoder.layers").pp(i);
                Ok(Layer {
                    ln1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, l.pp("layer_norm1"))?,
                    ln2: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, l.pp("layer_norm2"))?,
                    attn: Attention {
                        q: linear(cfg.hidden_size, cfg.hidden_size, l.pp("self_attn.q_proj"))?,
                        k: linear(cfg.hidden_size, cfg.hidden_size, l.pp("self_attn.k_proj"))?,
                        v: linear(cfg.hidden_size, cfg.hidden_size, l.pp("self_attn.v_proj"))?,
                        o: linear(cfg.hidden_size, cfg.hidden_size, l.pp("self_attn.out_proj"))?,
                        heads: cfg.num_attention_heads,
                        head_dim: hd,
                    },
                    fc1: linear(cfg.hidden_size, cfg.intermediate_size, l.pp("mlp.fc1"))?,
                    fc2: linear(cfg.intermediate_size, cfg.hidden_size, l.pp("mlp.fc2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch: linear(cfg.patch_dim(), cfg.hidden_size, e.pp("patch_embedding"))?,
            pos: e.pp("position_embedding").get((cfg.num_patches, cfg.hidden_size), "weight")?,
            layers,
            post: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cfg,
        })
    }

    /// The learned grid resampled to (h, w), raster order: (h·w, hidden).
    fn positions(&self, h: usize, w: usize) -> Result<Tensor> {
        let side = (self.cfg.num_patches as f64).sqrt() as usize;
        let rows = resample_weights(side, h);
        let cols = resample_weights(side, w);
        // Separable: first along width (side×w), then along height (h×w).
        let table = self.pos.to_dtype(DType::F32)?; // (side*side, hidden)
        let mut wmat = vec![0f32; side * w];
        for (o, taps) in cols.iter().enumerate() {
            for &(i, wt) in taps {
                wmat[i * w + o] = wt;
            }
        }
        let mut hmat = vec![0f32; h * side];
        for (o, taps) in rows.iter().enumerate() {
            for &(i, wt) in taps {
                hmat[o * side + i] = wt;
            }
        }
        let wmat = Tensor::from_vec(wmat, (side, w), &self.device)?;
        let hmat = Tensor::from_vec(hmat, (h, side), &self.device)?;
        // (hidden, side, side) → width resample → (hidden, side, w) → height → (hidden, h, w)
        let t = table.t()?.contiguous()?.reshape((self.cfg.hidden_size, side, side))?;
        let tw = t.broadcast_matmul(&wmat)?; // (hidden, side, w)
        let thw = hmat.broadcast_matmul(&tw)?; // (hidden, h, w)
        thw.reshape((self.cfg.hidden_size, h * w))?.t()?.contiguous()?.to_dtype(self.dtype)
    }

    /// `pixels`: (h·w, patch_dim) flattened patches of one crop in raster
    /// order, channel-last within a patch, in [-1, 1]. Returns the patch
    /// features (h·w, hidden).
    pub fn forward(&self, pixels: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let n = pixels.dim(0)?;
        if n != h * w {
            candle_core::bail!("{n} patches for a {h}×{w} grid");
        }
        let mut x = self.patch.forward(&pixels.to_dtype(self.dtype)?)?.add(&self.positions(h, w)?)?.unsqueeze(0)?;
        for l in &self.layers {
            let a = l.attn.forward(&l.ln1.forward(&x)?)?;
            x = (x + a)?;
            let m = l.fc2.forward(&l.fc1.forward(&l.ln2.forward(&x)?)?.gelu()?)?;
            x = (x + m)?;
        }
        self.post.forward(&x)?.squeeze(0)
    }
}

/// LFM2-VL's projector: 2×2 pixel unshuffle, then `linear_2(gelu(linear_1(x)))`.
pub struct Projector {
    l1: Linear,
    l2: Linear,
    factor: usize,
    vision_dim: usize,
}

impl Projector {
    pub fn new(vision_dim: usize, hidden: usize, out: usize, factor: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { l1: linear(vision_dim * factor * factor, hidden, vb.pp("linear_1"))?, l2: linear(hidden, out, vb.pp("linear_2"))?, factor, vision_dim })
    }

    /// `x`: (h·w, vision_dim) raster-order features of an (h, w) grid; both
    /// even. Returns (h·w / 4, out) in raster order of the merged grid.
    pub fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let f = self.factor;
        let c = self.vision_dim;
        // HF: view (1, w, h, c) -> ... the reshape pairs rows then columns; the
        // result is, for merged cell (i, j), the concatenation over the 2×2
        // block in order (row 2i, col 2j), (row 2i, col 2j+1), (row 2i+1, col 2j), (row 2i+1, col 2j+1)
        // laid out as HF's two reshape/permute steps produce it.
        let x = x.reshape((h, w, c))?;
        // Step 1: (h, w, c) -> (h, w/f, c*f): adjacent columns concatenated.
        let x = x.reshape((h, w / f, c * f))?;
        // Step 2: transpose to (w/f, h, c*f) -> reshape (w/f, h/f, c*f*f): adjacent rows concatenated -> transpose back.
        let x = x.transpose(0, 1)?.contiguous()?.reshape((w / f, h / f, c * f * f))?.transpose(0, 1)?.contiguous()?;
        let x = x.reshape((h / f * (w / f), c * f * f))?;
        self.l2.forward(&self.l1.forward(&x)?.gelu_erf()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_is_identity_and_normalised() {
        let id = resample_weights(16, 16);
        assert!(id.iter().enumerate().all(|(o, taps)| taps.len() == 1 && taps[0] == (o, 1.0)));
        for taps in resample_weights(16, 7) {
            let s: f32 = taps.iter().map(|t| t.1).sum();
            assert!((s - 1.0).abs() < 1e-5);
        }
        let up = resample_weights(16, 32);
        // Upsampling: two taps, half-pixel centres.
        assert_eq!(up[0].len(), 1); // clamped at the border
        assert_eq!(up[1].len(), 2);
    }
}
