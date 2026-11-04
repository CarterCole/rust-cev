//! Speech for d1-omni: WAV decoding and resampling to 16 kHz mono, NeMo's
//! log-mel front end, and the FastConformer encoder with its adapter and
//! residual (LFM2-Audio lineage). One prefix embedding per 80 ms of audio.

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Module, Result as CResult, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder, layer_norm, linear, linear_no_bias};
use rustfft::{FftPlanner, num_complex::Complex};
use serde::Deserialize;

pub const SAMPLE_RATE: usize = 16_000;
pub const MIN_SAMPLES: usize = 8_000;
pub const MAX_SECONDS: usize = 30;

/// Decode a WAV file to 16 kHz mono f32 in [-1, 1], cut to 30 s and padded
/// to half a second.
pub fn wav_to_16k_mono(bytes: &[u8]) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).context("cannot decode WAV")?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, _) => reader.samples::<f32>().collect::<std::result::Result<_, _>>()?,
        (hound::SampleFormat::Int, bits) => {
            let scale = 1.0 / (1u64 << (bits - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<std::result::Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = samples.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect();
    let mut x = if spec.sample_rate as usize == SAMPLE_RATE { mono } else { resample(&mono, spec.sample_rate as usize, SAMPLE_RATE) };
    x.truncate(MAX_SECONDS * SAMPLE_RATE);
    if x.len() < MIN_SAMPLES {
        x.resize(MIN_SAMPLES, 0.0);
    }
    Ok(x)
}

/// Windowed-sinc resampling (Kaiser-free Hann window, 32 taps per side).
pub fn resample(x: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = (x.len() as f64 / ratio).floor() as usize;
    let cutoff = 0.5 * ratio.min(1.0) / ratio.max(1.0) * ratio.max(1.0); // in input-sample units
    let taps = 32.0 * ratio.max(1.0);
    (0..out_len)
        .map(|i| {
            let center = i as f64 * ratio;
            let lo = (center - taps).floor().max(0.0) as usize;
            let hi = ((center + taps).ceil() as usize).min(x.len() - 1);
            let (mut acc, mut wsum) = (0f64, 0f64);
            for j in lo..=hi {
                let d = j as f64 - center;
                let win = 0.5 * (1.0 + (std::f64::consts::PI * d / taps).cos());
                let s = if d.abs() < 1e-9 { 1.0 } else { (2.0 * std::f64::consts::PI * cutoff * d).sin() / (2.0 * std::f64::consts::PI * cutoff * d) };
                let w = s * win * 2.0 * cutoff;
                acc += w * x[j] as f64;
                wsum += w;
            }
            (acc / wsum.max(1e-9) * (2.0 * cutoff).min(1.0) / (2.0 * cutoff)) as f32
        })
        .collect()
}

/// librosa's Slaney mel filterbank, (n_mels, 1 + n_fft/2).
pub fn slaney_filterbank(sr: usize, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let (f_sp, min_log_hz, logstep) = (200.0 / 3.0, 1000.0f64, (6.4f64).ln() / 27.0);
    let min_log_mel = min_log_hz / f_sp;
    let hz_to_mel = |f: f64| if f >= min_log_hz { min_log_mel + (f / min_log_hz).ln() / logstep } else { f / f_sp };
    let mel_to_hz = |m: f64| if m >= min_log_mel { min_log_hz * (logstep * (m - min_log_mel)).exp() } else { f_sp * m };
    let (m0, m1) = (hz_to_mel(0.0), hz_to_mel(sr as f64 / 2.0));
    let mel_f: Vec<f64> = (0..n_mels + 2).map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (n_mels + 1) as f64)).collect();
    let fft_f: Vec<f64> = (0..=n_fft / 2).map(|k| k as f64 * sr as f64 / n_fft as f64).collect();
    (0..n_mels)
        .map(|i| {
            let (lo, mid, hi) = (mel_f[i], mel_f[i + 1], mel_f[i + 2]);
            let enorm = 2.0 / (hi - lo);
            fft_f
                .iter()
                .map(|&f| {
                    let lower = (f - lo) / (mid - lo);
                    let upper = (hi - f) / (hi - mid);
                    (lower.min(upper).max(0.0) * enorm) as f32
                })
                .collect()
        })
        .collect()
}

/// NeMo's `FilterbankFeatures` in eval mode: preemphasis, |STFT|², Slaney
/// mel, log, per-feature normalisation. Returns (n_mels, frames) with the
/// number of valid frames (the last STFT frame is dropped, as NeMo does).
pub fn log_mel(x: &[f32], n_mels: usize) -> (Vec<Vec<f32>>, usize) {
    const N_FFT: usize = 512;
    const WIN: usize = 400;
    const HOP: usize = 160;
    let n = x.len();
    let valid = n / HOP;
    let frames = n / HOP + 1;
    // Preemphasis.
    let mut y = Vec::with_capacity(n);
    y.push(x[0]);
    for i in 1..n {
        y.push(x[i] - 0.97 * x[i - 1]);
    }
    // Hann window (periodic=False) of 400 centred in 512.
    let mut window = vec![0f32; N_FFT];
    let off = (N_FFT - WIN) / 2;
    for i in 0..WIN {
        window[off + i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (WIN - 1) as f32).cos());
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let fb = slaney_filterbank(SAMPLE_RATE, N_FFT, n_mels);
    let mut mel = vec![vec![0f32; frames]; n_mels];
    let mut buf = vec![Complex::new(0f32, 0f32); N_FFT];
    for t in 0..frames {
        let start = t as i64 * HOP as i64 - (N_FFT / 2) as i64;
        for (k, b) in buf.iter_mut().enumerate() {
            let idx = start + k as i64;
            let v = if idx >= 0 && (idx as usize) < n { y[idx as usize] } else { 0.0 };
            *b = Complex::new(v * window[k], 0.0);
        }
        fft.process(&mut buf);
        let power: Vec<f32> = buf[..=N_FFT / 2].iter().map(|c| c.norm_sqr()).collect();
        for (m, row) in fb.iter().enumerate() {
            let e: f32 = row.iter().zip(&power).map(|(w, p)| w * p).sum();
            mel[m][t] = (e + 2f32.powi(-24)).ln();
        }
    }
    // Per-feature mean/std over the valid frames; the rest zeroed.
    for row in mel.iter_mut() {
        let cnt = valid as f32;
        let mean = row[..valid].iter().sum::<f32>() / cnt;
        let var = row[..valid].iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (cnt - 1.0).max(0.0);
        let std = if var.is_finite() && cnt > 1.0 { var.sqrt() } else { 0.0 };
        for (t, v) in row.iter_mut().enumerate() {
            *v = if t < valid { (*v - mean) / (std + 1e-5) } else { 0.0 };
        }
    }
    (mel, valid)
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub feat_in: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub subsampling_conv_channels: usize,
    pub ff_expansion_factor: usize,
    pub n_heads: usize,
    pub conv_kernel_size: usize,
    pub residual_width: usize,
}

struct Conv2d {
    w: Tensor,
    b: Tensor,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl Conv2d {
    fn new(vb: VarBuilder, cin: usize, cout: usize, k: usize, stride: usize, padding: usize, groups: usize) -> CResult<Self> {
        Ok(Self { w: vb.get((cout, cin / groups, k, k), "weight")?, b: vb.get(cout, "bias")?, stride, padding, groups })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        x.conv2d(&self.w, self.padding, self.stride, 1, self.groups)?.broadcast_add(&self.b.reshape((1, self.b.dim(0)?, 1, 1))?)
    }
}

/// 8× time subsampling: a 3×3 stride-2 conv, then two depthwise-separable
/// stride-2 convs, then a linear.
struct Subsampling {
    convs: Vec<(Conv2d, bool)>,
    out: Linear,
}

impl Subsampling {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> CResult<Self> {
        let c = cfg.subsampling_conv_channels;
        let conv = vb.pp("conv");
        // Sequential indices: 0 conv, 1 relu, 2 dw, 3 pw, 4 relu, 5 dw, 6 pw, 7 relu.
        let convs = vec![
            (Conv2d::new(conv.pp(0), 1, c, 3, 2, 1, 1)?, true),
            (Conv2d::new(conv.pp(2), c, c, 3, 2, 1, c)?, false),
            (Conv2d::new(conv.pp(3), c, c, 1, 1, 0, 1)?, true),
            (Conv2d::new(conv.pp(5), c, c, 3, 2, 1, c)?, false),
            (Conv2d::new(conv.pp(6), c, c, 1, 1, 0, 1)?, true),
        ];
        let mut freq = cfg.feat_in;
        for _ in 0..3 {
            freq = (freq - 1) / 2 + 1;
        }
        Ok(Self { convs, out: linear(c * freq, cfg.d_model, vb.pp("out"))? })
    }

    /// x: (1, T, feat) → (1, T', d_model) and T'. (Single clip: no time mask needed.)
    fn forward(&self, x: &Tensor, len: usize) -> CResult<(Tensor, usize)> {
        let mut h = x.unsqueeze(1)?; // (1, 1, T, F)
        let mut l = len;
        for (c, relu) in &self.convs {
            h = c.forward(&h)?;
            if c.stride == 2 {
                l = (l + 2 - 3) / 2 + 1;
            }
            if *relu {
                h = h.relu()?;
            }
        }
        let (b, ch, t, f) = h.dims4()?;
        let h = h.transpose(1, 2)?.contiguous()?.reshape((b, t, ch * f))?;
        Ok((self.out.forward(&h)?, l.min(t)))
    }
}

struct RelAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    pos: Linear,
    u: Tensor,
    vb: Tensor,
    heads: usize,
    dk: usize,
}

impl RelAttention {
    /// x: (1, T, d); pos_emb: (2T-1, d). No padding (one clip at a time).
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> CResult<Tensor> {
        let (b, t, _) = x.dims3()?;
        let (h, dk) = (self.heads, self.dk);
        let q = self.q.forward(x)?.reshape((b, t, h, dk))?; // (B,T,H,D)
        let k = self.k.forward(x)?.reshape((b, t, h, dk))?.transpose(1, 2)?.contiguous()?; // (B,H,T,D)
        let v = self.v.forward(x)?.reshape((b, t, h, dk))?.transpose(1, 2)?.contiguous()?;
        let p = self.pos.forward(pos_emb)?.reshape((1, 2 * t - 1, h, dk))?.transpose(1, 2)?.contiguous()?; // (1,H,2T-1,D)
        let qu = q.broadcast_add(&self.u)?.transpose(1, 2)?.contiguous()?; // (B,H,T,D)
        let qv = q.broadcast_add(&self.vb)?.transpose(1, 2)?.contiguous()?;
        let ac = qu.matmul(&k.transpose(2, 3)?.contiguous()?)?; // (B,H,T,T)
        let bd = qv.matmul(&p.transpose(2, 3)?.contiguous()?.broadcast_as((b, h, dk, 2 * t - 1))?.contiguous()?)?; // (B,H,T,2T-1)
        // Relative shift: shifted[i, j] = bd[i, t - 1 - i + j].
        let idx: Vec<u32> = (0..t).flat_map(|i| (0..t).map(move |j| (t - 1 - i + j) as u32)).collect();
        let idx = Tensor::from_vec(idx, (1, 1, t, t), x.device())?.broadcast_as((b, h, t, t))?.contiguous()?;
        let bd = bd.gather(&idx, 3)?;
        let scores = ((ac + bd)? / (dk as f64).sqrt())?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, t, h * dk))?;
        self.o.forward(&ctx)
    }
}

struct ConvModule {
    pw1: Linear,
    dw: Tensor,
    dw_b: Tensor,
    bn_scale: Tensor,
    bn_shift: Tensor,
    pw2: Linear,
    kernel: usize,
}

impl ConvModule {
    fn new(d: usize, kernel: usize, vb: VarBuilder) -> CResult<Self> {
        let pw = |name: &str, i: usize, o: usize| -> CResult<Linear> {
            let w = vb.pp(name).get((o, i, 1), "weight")?.reshape((o, i))?;
            Ok(Linear::new(w, Some(vb.pp(name).get(o, "bias")?)))
        };
        let bn = vb.pp("batch_norm");
        let (w, b, mean, var) = (bn.get(d, "weight")?, bn.get(d, "bias")?, bn.get(d, "running_mean")?, bn.get(d, "running_var")?);
        let scale = w.to_dtype(DType::F32)?.div(&(var.to_dtype(DType::F32)? + 1e-5)?.sqrt()?)?;
        let shift = b.to_dtype(DType::F32)?.sub(&mean.to_dtype(DType::F32)?.mul(&scale)?)?;
        Ok(Self {
            pw1: pw("pointwise_conv1", d, 2 * d)?,
            dw: vb.pp("depthwise_conv").get((d, 1, kernel), "weight")?,
            dw_b: vb.pp("depthwise_conv").get(d, "bias")?,
            bn_scale: scale.to_dtype(w.dtype())?.reshape((1, d, 1))?,
            bn_shift: shift.to_dtype(w.dtype())?.reshape((1, d, 1))?,
            pw2: pw("pointwise_conv2", d, d)?,
            kernel,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, t, d) = x.dims3()?;
        let y = self.pw1.forward(x)?; // (B,T,2d)
        let (a, g) = (y.narrow(2, 0, d)?, y.narrow(2, d, d)?);
        let y = a.mul(&candle_nn::ops::sigmoid(&g)?)?.transpose(1, 2)?.contiguous()?; // GLU → (B,d,T)
        let pad = (self.kernel - 1) / 2;
        let y = y.conv1d(&self.dw, pad, 1, 1, d)?.broadcast_add(&self.dw_b.reshape((1, d, 1))?)?;
        let y = y.broadcast_mul(&self.bn_scale)?.broadcast_add(&self.bn_shift)?;
        let y = candle_nn::ops::silu(&y)?.transpose(1, 2)?.contiguous()?; // (B,T,d)
        let _ = (b, t);
        self.pw2.forward(&y)
    }
}

struct ConformerLayer {
    n_ff1: LayerNorm,
    ff1_1: Linear,
    ff1_2: Linear,
    n_att: LayerNorm,
    attn: RelAttention,
    n_conv: LayerNorm,
    conv: ConvModule,
    n_ff2: LayerNorm,
    ff2_1: Linear,
    ff2_2: Linear,
    n_out: LayerNorm,
}

impl ConformerLayer {
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> CResult<Tensor> {
        let ff = |a: &Linear, b: &Linear, h: Tensor| -> CResult<Tensor> { b.forward(&candle_nn::ops::silu(&a.forward(&h)?)?) };
        let mut x = (x + (ff(&self.ff1_1, &self.ff1_2, self.n_ff1.forward(x)?)? * 0.5)?)?;
        x = (&x + self.attn.forward(&self.n_att.forward(&x)?, pos_emb)?)?;
        x = (&x + self.conv.forward(&self.n_conv.forward(&x)?)?)?;
        x = (&x + (ff(&self.ff2_1, &self.ff2_2, self.n_ff2.forward(&x)?)? * 0.5)?)?;
        self.n_out.forward(&x)
    }
}

/// The whole speech path: mel → subsampling → conformer → adapter → residual.
pub struct AudioEncoder {
    sub: Subsampling,
    layers: Vec<ConformerLayer>,
    adapter_norm: LayerNorm,
    adapter_1: Linear,
    adapter_2: Linear,
    res_ln: LayerNorm,
    res_down: Linear,
    res_up: Linear,
    pub cfg: AudioConfig,
    device: Device,
    dtype: DType,
}

impl AudioEncoder {
    /// `vb` rooted at `audio`; `out` is the trunk's width.
    pub fn new(cfg: AudioConfig, out: usize, vb: VarBuilder) -> CResult<Self> {
        let d = cfg.d_model;
        let enc = vb.pp("encoder");
        let layers = (0..cfg.n_layers)
            .map(|i| {
                let l = enc.pp("layers").pp(i);
                let a = l.pp("self_attn");
                Ok(ConformerLayer {
                    n_ff1: layer_norm(d, 1e-5, l.pp("norm_feed_forward1"))?,
                    ff1_1: linear(d, d * cfg.ff_expansion_factor, l.pp("feed_forward1.linear1"))?,
                    ff1_2: linear(d * cfg.ff_expansion_factor, d, l.pp("feed_forward1.linear2"))?,
                    n_att: layer_norm(d, 1e-5, l.pp("norm_self_att"))?,
                    attn: RelAttention {
                        q: linear(d, d, a.pp("linear_q"))?,
                        k: linear(d, d, a.pp("linear_k"))?,
                        v: linear(d, d, a.pp("linear_v"))?,
                        o: linear(d, d, a.pp("linear_out"))?,
                        pos: linear_no_bias(d, d, a.pp("linear_pos"))?,
                        u: a.get((cfg.n_heads, d / cfg.n_heads), "pos_bias_u")?,
                        vb: a.get((cfg.n_heads, d / cfg.n_heads), "pos_bias_v")?,
                        heads: cfg.n_heads,
                        dk: d / cfg.n_heads,
                    },
                    n_conv: layer_norm(d, 1e-5, l.pp("norm_conv"))?,
                    conv: ConvModule::new(d, cfg.conv_kernel_size, l.pp("conv"))?,
                    n_ff2: layer_norm(d, 1e-5, l.pp("norm_feed_forward2"))?,
                    ff2_1: linear(d, d * cfg.ff_expansion_factor, l.pp("feed_forward2.linear1"))?,
                    ff2_2: linear(d * cfg.ff_expansion_factor, d, l.pp("feed_forward2.linear2"))?,
                    n_out: layer_norm(d, 1e-5, l.pp("norm_out"))?,
                })
            })
            .collect::<CResult<Vec<_>>>()?;
        Ok(Self {
            sub: Subsampling::new(&cfg, enc.pp("pre_encode"))?,
            layers,
            adapter_norm: layer_norm(d, 1e-5, vb.pp("adapter.norm"))?,
            adapter_1: linear(d, out, vb.pp("adapter.linear_1"))?,
            adapter_2: linear(out, out, vb.pp("adapter.linear_2"))?,
            res_ln: layer_norm(out, 1e-5, vb.pp("residual.ln"))?,
            res_down: linear(out, cfg.residual_width, vb.pp("residual.down"))?,
            res_up: linear(cfg.residual_width, out, vb.pp("residual.up"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cfg,
        })
    }

    /// Sinusoids for relative positions t-1 .. -(t-1): (2t-1, d).
    fn pos_emb(&self, t: usize) -> CResult<Tensor> {
        let d = self.cfg.d_model;
        let mut pe = vec![0f32; (2 * t - 1) * d];
        for (r, pos) in (0..2 * t - 1).map(|r| (r, (t as i64 - 1 - r as i64) as f32)) {
            for i in (0..d).step_by(2) {
                let div = (-(i as f32) * (10_000f32).ln() / d as f32).exp();
                pe[r * d + i] = (pos * div).sin();
                if i + 1 < d {
                    pe[r * d + i + 1] = (pos * div).cos();
                }
            }
        }
        Tensor::from_vec(pe, (2 * t - 1, d), &self.device)?.to_dtype(self.dtype)
    }

    /// 16 kHz mono samples → prefix embeddings (n, out) f32-convertible.
    pub fn forward(&self, samples: &[f32]) -> Result<Tensor> {
        if samples.len() < MIN_SAMPLES {
            bail!("audio clip too short");
        }
        let (mel, valid) = log_mel(samples, self.cfg.feat_in);
        let frames = mel[0].len();
        // (1, T, feat): time-major for the conv input.
        let mut flat = vec![0f32; frames * self.cfg.feat_in];
        for (m, row) in mel.iter().enumerate() {
            for (t, v) in row.iter().enumerate() {
                flat[t * self.cfg.feat_in + m] = *v;
            }
        }
        let x = Tensor::from_vec(flat, (1, frames, self.cfg.feat_in), &self.device)?.to_dtype(self.dtype)?;
        let (mut h, len) = self.sub.forward(&x, valid)?;
        // The subsampling ran over every frame; keep only the valid ones.
        h = h.narrow(1, 0, len)?.contiguous()?;
        let t = h.dim(1)?;
        let pos = self.pos_emb(t)?;
        for l in &self.layers {
            h = l.forward(&h, &pos)?;
        }
        let a = self.adapter_2.forward(&self.adapter_1.forward(&self.adapter_norm.forward(&h)?)?.gelu_erf()?)?;
        let r = self.res_up.forward(&self.res_down.forward(&self.res_ln.forward(&a)?)?.gelu_erf()?)?;
        Ok((a + r)?.squeeze(0)?)
    }
}

#[allow(dead_code)]
fn _d(_: D) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_shapes_and_normalisation() {
        let x: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let (mel, valid) = log_mel(&x, 128);
        assert_eq!(mel.len(), 128);
        assert_eq!(mel[0].len(), 101);
        assert_eq!(valid, 100);
        let mean: f32 = mel[10][..valid].iter().sum::<f32>() / valid as f32;
        assert!(mean.abs() < 1e-3);
        assert_eq!(mel[10][100], 0.0);
        let fb = slaney_filterbank(16_000, 512, 128);
        assert_eq!(fb.len(), 128);
        assert!(fb.iter().all(|r| r.len() == 257 && r.iter().all(|v| *v >= 0.0)));
    }

    #[test]
    fn resampling_keeps_length_ratio() {
        let x: Vec<f32> = (0..44_100).map(|i| (i as f32 * 0.01).sin()).collect();
        let y = resample(&x, 44_100, 16_000);
        assert!((y.len() as i64 - 16_000).abs() <= 1);
        // A smooth tone keeps its amplitude.
        assert!(y[8000..].iter().fold(0f32, |m, v| m.max(v.abs())) > 0.9);
    }
}
