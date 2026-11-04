//! Qwen3.5's vision tower: 16×16 patches (two identical temporal frames for a
//! still image) through a linear patch embedding, a learned 48×48 position
//! grid resampled to the image's grid, 2D rotary attention blocks, and a
//! 2×2 patch merger into the text model's width. Matches HF transformers
//! `Qwen3_5VisionModel` (no DeepStack).
//!
//! Patches arrive in spatial-merge-block order (see [`crate::media`]), so the
//! merger's `view(-1, 4·hidden)` groups each 2×2 block.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder, layer_norm, linear};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    #[serde(default = "three")]
    pub in_channels: usize,
    #[serde(default = "sixteen")]
    pub patch_size: usize,
    #[serde(default = "two")]
    pub temporal_patch_size: usize,
    #[serde(default = "two")]
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    #[serde(default = "pos")]
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
}

fn three() -> usize {
    3
}
fn sixteen() -> usize {
    16
}
fn two() -> usize {
    2
}
fn pos() -> usize {
    2304
}

impl VisionConfig {
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.as_ref().and_then(|r| r["rope_theta"].as_f64()).unwrap_or(10_000.0)
    }
}

/// Patch grid of one image after the patch embedding: `t` frames of `h × w`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl Grid {
    pub fn patches(&self) -> usize {
        self.t * self.h * self.w
    }

    /// Tokens the text model sees after merging.
    pub fn tokens(&self, merge: usize) -> usize {
        self.patches() / (merge * merge)
    }
}

/// (row, col) of every patch in spatial-merge-block order.
pub fn block_order(h: usize, w: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(h * w);
    for br in 0..h / merge {
        for bc in 0..w / merge {
            for ir in 0..merge {
                for ic in 0..merge {
                    out.push((br * merge + ir, bc * merge + ic));
                }
            }
        }
    }
    out
}

/// Bilinear (align_corners) taps into a `side`-point axis for `index` of `size`.
fn axis_taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    let src = index as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
    let src = if size == 1 { 0.0 } else { src };
    let floor = src.floor();
    let mut taps = [(0usize, 0f32); 2];
    for (k, off) in [0f32, 1.0].iter().enumerate() {
        let raw = floor + off;
        let tap = raw.clamp(0.0, (side - 1) as f32) as usize;
        let w = (1.0 - (src - raw).abs()).max(0.0);
        taps[k] = (tap, w);
    }
    taps
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    /// x: (N, hidden); cos/sin: (N, head_dim/2) f32.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let qkv = self.qkv.forward(x)?.reshape((n, 3, self.heads, self.head_dim))?.permute((1, 2, 0, 3))?.contiguous()?; // (3, H, N, D)
        let rot = |t: Tensor| -> Result<Tensor> {
            let t32 = t.unsqueeze(0)?.to_dtype(DType::F32)?.contiguous()?;
            candle_nn::rotary_emb::rope(&t32, cos, sin)?.to_dtype(x.dtype())
        };
        let q = rot(qkv.get(0)?)?;
        let k = rot(qkv.get(1)?)?;
        let v = qkv.get(2)?.unsqueeze(0)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let ctx = if x.device().is_metal() && matches!(self.head_dim, 64 | 128) {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?
        } else {
            let scores = (q.matmul(&k.t()?.contiguous()?)?.to_dtype(DType::F32)? * scale)?;
            candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?.matmul(&v)?
        };
        let ctx = ctx.squeeze(0)?.transpose(0, 1)?.reshape((n, self.heads * self.head_dim))?;
        self.proj.forward(&ctx)
    }
}

struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: Attention,
    fc1: Linear,
    fc2: Linear,
}

pub struct Qwen35Vision {
    patch: Linear,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    merger_norm: LayerNorm,
    merger_fc1: Linear,
    merger_fc2: Linear,
    inv_freq: Tensor,
    pub cfg: VisionConfig,
    device: Device,
    dtype: DType,
}

impl Qwen35Vision {
    /// `vb` rooted at the vision tower (`model.visual` or `visual`).
    pub fn new(cfg: VisionConfig, vb: VarBuilder) -> Result<Self> {
        let hd = cfg.hidden_size / cfg.num_heads;
        let w = vb.pp("patch_embed.proj").get((cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size), "weight")?;
        let b = vb.pp("patch_embed.proj").get(cfg.hidden_size, "bias")?;
        let patch = Linear::new(w.reshape((cfg.hidden_size, cfg.patch_dim()))?, Some(b));
        let pos_embed = vb.pp("pos_embed").get((cfg.num_position_embeddings, cfg.hidden_size), "weight")?;
        let blocks = (0..cfg.depth)
            .map(|i| {
                let vb = vb.pp("blocks").pp(i);
                Ok(Block {
                    norm1: layer_norm(cfg.hidden_size, 1e-6, vb.pp("norm1"))?,
                    norm2: layer_norm(cfg.hidden_size, 1e-6, vb.pp("norm2"))?,
                    attn: Attention { qkv: linear(cfg.hidden_size, 3 * cfg.hidden_size, vb.pp("attn.qkv"))?, proj: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attn.proj"))?, heads: cfg.num_heads, head_dim: hd },
                    fc1: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.linear_fc1"))?,
                    fc2: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("mlp.linear_fc2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let merged = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let spatial = hd / 2;
        let inv: Vec<f32> = (0..spatial).step_by(2).map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / spatial as f64) as f32).collect();
        Ok(Self {
            patch,
            pos_embed,
            blocks,
            merger_norm: layer_norm(cfg.hidden_size, 1e-6, vb.pp("merger.norm"))?,
            merger_fc1: linear(merged, merged, vb.pp("merger.linear_fc1"))?,
            merger_fc2: linear(merged, cfg.out_hidden_size, vb.pp("merger.linear_fc2"))?,
            inv_freq: Tensor::from_vec(inv, (1, spatial / 2), vb.device())?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cfg,
        })
    }

    /// Learned position embeddings resampled to the grid, in block order: (N, hidden).
    fn positions(&self, grid: Grid) -> Result<Tensor> {
        let side = (self.cfg.num_position_embeddings as f64).sqrt() as usize;
        let order = block_order(grid.h, grid.w, self.cfg.spatial_merge_size);
        let mut idx = Vec::with_capacity(order.len() * 4);
        let mut wts = Vec::with_capacity(order.len() * 4);
        for &(r, c) in &order {
            for (hi, hw) in axis_taps(r, grid.h, side) {
                for (wi, ww) in axis_taps(c, grid.w, side) {
                    idx.push((hi * side + wi) as u32);
                    wts.push(hw * ww);
                }
            }
        }
        let n = order.len();
        let rows = self.pos_embed.index_select(&Tensor::new(idx.as_slice(), &self.device)?, 0)?.to_dtype(DType::F32)?; // (N*4, hidden)
        let w = Tensor::from_vec(wts, (n * 4, 1), &self.device)?;
        let pe = rows.broadcast_mul(&w)?.reshape((n, 4, self.cfg.hidden_size))?.sum(1)?;
        // Repeated for every temporal frame.
        let pe = if grid.t > 1 { Tensor::cat(&vec![&pe; grid.t], 0)? } else { pe };
        pe.to_dtype(self.dtype)
    }

    /// 2D rotary cos/sin (N, head_dim/2) in block order.
    fn rotary(&self, grid: Grid) -> Result<(Tensor, Tensor)> {
        let order = block_order(grid.h, grid.w, self.cfg.spatial_merge_size);
        let n = order.len();
        let hs: Vec<f32> = order.iter().map(|&(r, _)| r as f32).collect();
        let ws: Vec<f32> = order.iter().map(|&(_, c)| c as f32).collect();
        let fh = Tensor::from_vec(hs, (n, 1), &self.device)?.broadcast_mul(&self.inv_freq)?;
        let fw = Tensor::from_vec(ws, (n, 1), &self.device)?.broadcast_mul(&self.inv_freq)?;
        let f = Tensor::cat(&[fh, fw], 1)?; // (N, head_dim/2)
        let f = if grid.t > 1 { Tensor::cat(&vec![&f; grid.t], 0)? } else { f };
        Ok((f.cos()?, f.sin()?))
    }

    /// `pixels`: (N, patch_dim) flattened patches of one image in block order
    /// (any float dtype). Returns the merged image tokens (N / merge², out_hidden).
    pub fn forward(&self, pixels: &Tensor, grid: Grid) -> Result<Tensor> {
        let n = pixels.dim(0)?;
        if n != grid.patches() {
            candle_core::bail!("{n} patches for a {:?} grid", grid);
        }
        let mut x = self.patch.forward(&pixels.to_dtype(self.dtype)?)?;
        x = x.add(&self.positions(grid)?)?;
        let (cos, sin) = self.rotary(grid)?;
        for b in &self.blocks {
            let h = b.attn.forward(&b.norm1.forward(&x)?, &cos, &sin)?;
            x = (x + h)?;
            let h = b.fc2.forward(&b.fc1.forward(&b.norm2.forward(&x)?)?.gelu()?)?;
            x = (x + h)?;
        }
        let m = self.cfg.spatial_merge_size * self.cfg.spatial_merge_size;
        let x = self.merger_norm.forward(&x)?.reshape((n / m, m * self.cfg.hidden_size))?;
        self.merger_fc2.forward(&self.merger_fc1.forward(&x)?.gelu_erf()?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_order_and_taps() {
        assert_eq!(block_order(4, 4, 2)[..6], [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)]);
        // Endpoints map to the table's edges with align_corners.
        assert_eq!(axis_taps(0, 10, 48)[0], (0, 1.0));
        let last = axis_taps(9, 10, 48);
        assert_eq!(last[0].0, 47);
        assert!((last[0].1 - 1.0).abs() < 1e-6);
        let mid = axis_taps(3, 10, 48); // src = 3 * 47 / 9 = 15.666
        assert_eq!((mid[0].0, mid[1].0), (15, 16));
        assert!((mid[0].1 - 0.3333).abs() < 1e-3 && (mid[1].1 - 0.6667).abs() < 1e-3);
    }
}
