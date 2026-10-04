//! int8 weights for CPU serving, mainly the browser: a quarter of the memory
//! and download of f32, and an integer inner loop.
//!
//! A weight matrix is stored row by row as `W[o][k] ≈ scale[o] · q[o][k]` with
//! `q` in -127..=127. At run time each input row is quantized too (to as many
//! bits as cannot overflow an i32 accumulator), so a matmul is integer dot
//! products followed by one multiply per output.

use candle_core::{DType, Device, Result, Tensor, bail};
use rayon::prelude::*;

/// Suffix of the f32 per-row scale stored next to a quantized `….weight`.
pub const SCALE_SUFFIX: &str = ".scale";

/// Bits per quantized input value. With 8, a third of the zero-shot answers
/// moved by more than 0.1 in probability; with 13 they track f32 closely.
const ACT_BITS: u32 = 13;

pub struct QLinear {
    q: Vec<i8>,
    scale: Vec<f32>,
    rows: usize,
    cols: usize,
    /// Largest quantized input magnitude: |x_q| · 127 · cols fits in an i32.
    act_max: f32,
}

/// Quantize a (rows, cols) f32 matrix into the (u8, f32) pair stored on disk:
/// the int8 values as raw bytes, and the per-row scale.
pub fn quantize(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = w.dims2()?;
    let w: Vec<f32> = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut q = Vec::with_capacity(w.len());
    let mut scale = Vec::with_capacity(rows);
    for row in w.chunks(cols) {
        let s = row.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        q.extend(row.iter().map(|v| (v * inv).round() as i8 as u8));
        scale.push(s);
    }
    Ok((Tensor::from_vec(q, (rows, cols), &Device::Cpu)?, Tensor::from_vec(scale, rows, &Device::Cpu)?))
}

impl QLinear {
    /// From the stored pair; see [`quantize`].
    pub fn new(q: &Tensor, scale: &Tensor) -> Result<Self> {
        let (rows, cols) = q.dims2()?;
        if q.dtype() != DType::U8 || scale.dims() != [rows] {
            bail!("bad quantized weight: {:?} {:?} with scale {:?}", q.dtype(), q.shape(), scale.shape());
        }
        let q = q.flatten_all()?.to_vec1::<u8>()?.into_iter().map(|b| b as i8).collect();
        let act_max = (((1u32 << (ACT_BITS - 1)) - 1) as f32).min((i32::MAX as usize / (127 * cols)) as f32);
        Ok(Self { q, scale: scale.to_dtype(DType::F32)?.to_vec1()?, rows, cols, act_max })
    }

    /// Dequantized rows (n, cols): an embedding lookup.
    pub fn select(&self, ids: &[u32]) -> Result<Tensor> {
        let mut out = Vec::with_capacity(ids.len() * self.cols);
        for &i in ids {
            let i = i as usize;
            if i >= self.rows {
                bail!("row {i} out of range ({} rows)", self.rows);
            }
            let s = self.scale[i];
            out.extend(self.q[i * self.cols..(i + 1) * self.cols].iter().map(|&v| v as f32 * s));
        }
        Tensor::from_vec(out, (ids.len(), self.cols), &Device::Cpu)
    }

    /// x (.., cols) → (.., rows), f32 in and out.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut shape = x.dims().to_vec();
        match shape.last_mut() {
            Some(k) if *k == self.cols => *k = self.rows,
            _ => bail!("expected (.., {}), got {:?}", self.cols, x.shape()),
        }
        let xs: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut out = vec![0f32; xs.len() / self.cols * self.rows];
        // Four input rows at a time: the wasm kernel shares each weight load
        // between them.
        out.par_chunks_mut(4 * self.rows).zip(xs.par_chunks(4 * self.cols)).for_each(|(ys, xs)| {
            let mut xq = vec![0i16; xs.len()];
            let mut sx = [0f32; 4];
            for ((x, xq), sx) in xs.chunks(self.cols).zip(xq.chunks_mut(self.cols)).zip(&mut sx) {
                let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
                if amax > 0.0 && amax.is_finite() {
                    let s = self.act_max / amax;
                    xq.iter_mut().zip(x).for_each(|(q, v)| *q = (v * s).round() as i16);
                    *sx = 1.0 / s;
                }
            }
            #[cfg(target_arch = "wasm32")]
            if xq.len() == 4 * self.cols {
                return dots4(&self.q, &self.scale, &xq, sx, ys);
            }
            for ((xq, sx), y) in xq.chunks(self.cols).zip(sx).zip(ys.chunks_mut(self.rows)) {
                dots(&self.q, &self.scale, xq, sx, y)
            }
        });
        Tensor::from_vec(out, shape, x.device())
    }
}

/// y[o] = scale[o] · sx · Σ_k q[o][k] · x[k].
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
fn dots(q: &[i8], scale: &[f32], x: &[i16], sx: f32, y: &mut [f32]) {
    for ((w, s), y) in q.chunks_exact(x.len()).zip(scale).zip(y) {
        *y = dot(x, w) as f32 * s * sx;
    }
}

/// Natively the compiler vectorizes this well (NEON/AVX).
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn dot(x: &[i16], w: &[i8]) -> i32 {
    x.iter().zip(w).map(|(a, b)| *a as i32 * *b as i32).sum()
}

/// For wasm it does not, so spell it out: 16 products per step.
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline]
fn dot(x: &[i16], w: &[i8]) -> i32 {
    use core::arch::wasm32::*;
    let n = x.len().min(w.len());
    let (mut a0, mut a1) = (i32x4_splat(0), i32x4_splat(0));
    let mut i = 0;
    while i + 16 <= n {
        // SAFETY: i + 16 <= n bounds every load.
        unsafe {
            let wv = v128_load(w.as_ptr().add(i) as *const v128);
            let x0 = v128_load(x.as_ptr().add(i) as *const v128);
            let x1 = v128_load(x.as_ptr().add(i + 8) as *const v128);
            a0 = i32x4_add(a0, i32x4_dot_i16x8(x0, i16x8_extend_low_i8x16(wv)));
            a1 = i32x4_add(a1, i32x4_dot_i16x8(x1, i16x8_extend_high_i8x16(wv)));
        }
        i += 16;
    }
    let a = i32x4_add(a0, a1);
    let mut acc = i32x4_extract_lane::<0>(a) + i32x4_extract_lane::<1>(a) + i32x4_extract_lane::<2>(a) + i32x4_extract_lane::<3>(a);
    while i < n {
        acc += x[i] as i32 * w[i] as i32;
        i += 1;
    }
    acc
}

/// [`dots`] for four input rows at once (`x` and `y` hold them back to back).
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
fn dots4(q: &[i8], scale: &[f32], x: &[i16], sx: [f32; 4], y: &mut [f32]) {
    use core::arch::wasm32::*;
    let (cols, rows) = (x.len() / 4, scale.len());
    let sum = |a: v128| i32x4_extract_lane::<0>(a) + i32x4_extract_lane::<1>(a) + i32x4_extract_lane::<2>(a) + i32x4_extract_lane::<3>(a);
    for (o, (w, s)) in q.chunks_exact(cols).zip(scale).enumerate() {
        let mut acc = [i32x4_splat(0); 8];
        let mut i = 0;
        while i + 16 <= cols {
            // SAFETY: i + 16 <= cols bounds every load.
            unsafe {
                let wv = v128_load(w.as_ptr().add(i) as *const v128);
                let (lo, hi) = (i16x8_extend_low_i8x16(wv), i16x8_extend_high_i8x16(wv));
                for t in 0..4 {
                    let p = x.as_ptr().add(t * cols + i);
                    acc[2 * t] = i32x4_add(acc[2 * t], i32x4_dot_i16x8(v128_load(p as *const v128), lo));
                    acc[2 * t + 1] = i32x4_add(acc[2 * t + 1], i32x4_dot_i16x8(v128_load(p.add(8) as *const v128), hi));
                }
            }
            i += 16;
        }
        for t in 0..4 {
            let mut a = sum(i32x4_add(acc[2 * t], acc[2 * t + 1]));
            for j in i..cols {
                a += x[t * cols + j] as i32 * w[j] as i32;
            }
            y[t * rows + o] = a as f32 * s * sx[t];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_f32_matmul() {
        let (rows, cols, n) = (7, 96, 5);
        let mut seed = 1u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let w: Vec<f32> = (0..rows * cols).map(|_| rand() * 0.1).collect();
        let x: Vec<f32> = (0..n * cols).map(|_| rand() * 3.0).collect();
        let w = Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap();
        let x = Tensor::from_vec(x, (1, n, cols), &Device::Cpu).unwrap();
        let (q, scale) = quantize(&w).unwrap();
        let lin = QLinear::new(&q, &scale).unwrap();

        let want: Vec<f32> = x.broadcast_matmul(&w.t().unwrap()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let got = lin.forward(&x).unwrap();
        assert_eq!(got.dims(), [1, n, rows]); // n = 5: one block of four and a leftover row
        let got: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
        let norm = want.iter().map(|v| v * v).sum::<f32>().sqrt();
        let err = want.iter().zip(&got).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        assert!(err / norm < 0.01, "relative error {}", err / norm);

        let row: Vec<f32> = lin.select(&[3]).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let exact: Vec<f32> = w.get(3).unwrap().to_vec1().unwrap();
        assert!(row.iter().zip(&exact).all(|(a, b)| (a - b).abs() < 0.1 / 127.0));
    }
}
