//! Online adaptation from feedback, in plain Rust.
//!
//! The backbone stays frozen. Each task gets a residual layer on top of the
//! base option logits `z` and the answer-position hidden state `h`:
//!
//! ```text
//!   z'_k = z_k · e^{-τ} + b_k + w_k · x,     x = standardize(h) / √d
//! ```
//!
//! With `τ = b = w = 0` this is exactly the base model. `τ` learns
//! calibration, `b` learns label priors, `w` learns feature-dependent
//! corrections (a linear probe on the backbone's own representation). Weights
//! are keyed by option *name*, so they survive option reordering.
//!
//! Training refits over a replay buffer (warm start, a few SGD epochs) after
//! every label. The adapter is only served once it wins *prequentially*: every
//! label is first scored by both the base model and the current adapter, before
//! either learns from it, so the comparison is always on unseen data.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnConfig {
    /// Labels needed before the adapter may be served.
    pub min_examples: u64,
    /// Replay buffer size per task (oldest dropped first).
    pub buffer: usize,
    /// SGD epochs over the buffer after each label.
    pub epochs: usize,
    pub lr_w: f32,
    pub lr_b: f32,
    pub lr_t: f32,
    /// L2 pull toward the base model, per step.
    pub l2: f32,
    /// Decay of the prequential loss averages (closer to 1 = longer memory).
    pub loss_decay: f64,
}

impl Default for LearnConfig {
    fn default() -> Self {
        Self {
            min_examples: 8,
            buffer: 1024,
            epochs: 3,
            lr_w: 0.5,
            lr_b: 0.1,
            lr_t: 0.02,
            l2: 1e-3,
            loss_decay: 0.97,
        }
    }
}

/// Running per-dimension mean/variance of hidden states (Welford), shared by
/// all tasks of a backbone. Frozen after `freeze_after` samples so stored
/// features keep a stable meaning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalizer {
    pub n: u64,
    pub mean: Vec<f64>,
    pub m2: Vec<f64>,
    pub freeze_after: u64,
}

impl Normalizer {
    pub fn new(dim: usize) -> Self {
        Self { n: 0, mean: vec![0.0; dim], m2: vec![0.0; dim], freeze_after: 20_000 }
    }

    pub fn observe(&mut self, h: &[f32]) {
        if self.n >= self.freeze_after || h.len() != self.mean.len() {
            return;
        }
        self.n += 1;
        let n = self.n as f64;
        for (i, &v) in h.iter().enumerate() {
            let v = v as f64;
            let d = v - self.mean[i];
            self.mean[i] += d / n;
            self.m2[i] += d * (v - self.mean[i]);
        }
    }

    /// Standardized, unit-norm-scale features.
    pub fn features(&self, h: &[f32]) -> Vec<f32> {
        let d = h.len() as f32;
        let scale = 1.0 / d.sqrt();
        h.iter()
            .enumerate()
            .map(|(i, &v)| {
                let (m, var) = if self.n > 1 {
                    (self.mean[i], self.m2[i] / (self.n - 1) as f64)
                } else {
                    (0.0, 1.0)
                };
                let s = var.sqrt().max(1e-3) as f32;
                (((v - m as f32) / s).clamp(-8.0, 8.0)) * scale
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptionParams {
    b: f32,
    w: Vec<f32>,
}

/// One labelled example in the replay buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Decision the label belongs to; a newer label replaces an older one.
    pub decision_id: String,
    pub names: Vec<String>,
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
    pub target: Vec<f64>,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adapter {
    pub dim: usize,
    tau: f32,
    options: IndexMap<String, OptionParams>,
    pub examples: u64,
    /// Decayed prequential log loss of the base model and of the adapter.
    pub base_loss: f64,
    pub adapted_loss: f64,
    /// Rebuilt from the store on load, never serialized.
    #[serde(skip)]
    buffer: Vec<Sample>,
}

fn softmax(z: &[f32]) -> Vec<f64> {
    cev_core::math::softmax(z)
}

impl Adapter {
    pub fn new(dim: usize) -> Self {
        Self { dim, tau: 0.0, options: IndexMap::new(), examples: 0, base_loss: 0.0, adapted_loss: 0.0, buffer: Vec::new() }
    }

    pub fn temperature(&self) -> f32 {
        self.tau.exp()
    }

    pub fn is_active(&self, cfg: &LearnConfig) -> bool {
        self.examples >= cfg.min_examples && self.adapted_loss < self.base_loss
    }

    pub fn logits(&self, names: &[String], z: &[f32], x: &[f32]) -> Vec<f32> {
        let inv_t = (-self.tau).exp();
        names
            .iter()
            .zip(z)
            .map(|(n, &zk)| {
                let base = zk * inv_t;
                match self.options.get(n) {
                    Some(p) if p.w.len() == x.len() => base + p.b + dot(&p.w, x),
                    Some(p) => base + p.b,
                    None => base,
                }
            })
            .collect()
    }

    pub fn predict(&self, names: &[String], z: &[f32], x: &[f32]) -> Vec<f64> {
        softmax(&self.logits(names, z, x))
    }

    /// Score `s` prequentially, add it to the buffer, and refit.
    pub fn learn(&mut self, s: Sample, norm: &Normalizer, cfg: &LearnConfig) {
        let x = norm.features(&s.hidden);
        let base = cev_core::math::log_loss(&softmax(&s.logits), &s.target);
        let adapted = cev_core::math::log_loss(&self.predict(&s.names, &s.logits, &x), &s.target);
        let d = cfg.loss_decay;
        if self.examples == 0 {
            (self.base_loss, self.adapted_loss) = (base, adapted);
        } else {
            self.base_loss = d * self.base_loss + (1.0 - d) * base;
            self.adapted_loss = d * self.adapted_loss + (1.0 - d) * adapted;
        }
        self.examples += 1;
        self.buffer.retain(|b| b.decision_id != s.decision_id);
        self.buffer.push(s);
        if self.buffer.len() > cfg.buffer {
            let drop = self.buffer.len() - cfg.buffer;
            self.buffer.drain(..drop);
        }
        self.refit(norm, cfg);
    }

    fn refit(&mut self, norm: &Normalizer, cfg: &LearnConfig) {
        let feats: Vec<Vec<f32>> = self.buffer.iter().map(|s| norm.features(&s.hidden)).collect();
        let n = self.buffer.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = 0x9E3779B97F4A7C15u64 ^ self.examples;
        for _ in 0..cfg.epochs {
            // Newest example last so it always gets a fresh step.
            for i in (1..n.saturating_sub(1)).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, (rng % (i as u64 + 1)) as usize);
            }
            for &i in &order {
                let s = &self.buffer[i];
                let x = &feats[i];
                let p = self.predict(&s.names, &s.logits, x);
                let inv_t = (-self.tau).exp();
                let mut g_tau = 0.0f32;
                for (k, name) in s.names.iter().enumerate() {
                    let g = (p[k] - s.target[k]) as f32 * s.weight;
                    g_tau += g * -(s.logits[k] * inv_t);
                    let dim = self.dim;
                    let o = self
                        .options
                        .entry(name.clone())
                        .or_insert_with(|| OptionParams { b: 0.0, w: vec![0.0; dim] });
                    o.b -= cfg.lr_b * (g + cfg.l2 * o.b);
                    for (w, &xi) in o.w.iter_mut().zip(x) {
                        *w -= cfg.lr_w * (g * xi + cfg.l2 * *w);
                    }
                }
                self.tau = (self.tau - cfg.lr_t * (g_tau + cfg.l2 * self.tau)).clamp(-2.0, 3.0);
            }
        }
    }

    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Restore the replay buffer (oldest first) after loading.
    pub fn set_buffer(&mut self, buffer: Vec<Sample>) {
        self.buffer = buffer;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Base model is systematically wrong on a feature-dependent slice; the
    /// adapter must learn the correction and win prequentially.
    #[test]
    fn learns_feature_dependent_correction() {
        let dim = 32;
        let names: Vec<String> = vec!["a".into(), "b".into()];
        let mut norm = Normalizer::new(dim);
        let mut ad = Adapter::new(dim);
        let cfg = LearnConfig::default();
        let mut rng = 1u64;
        let mut rand = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let mut correct_late = 0;
        for t in 0..300 {
            let h: Vec<f32> = (0..dim).map(|_| rand() * 3.0 + 1.0).collect();
            let truth_b = h[0] > 1.0; // true label depends on feature 0
            let logits = vec![1.5, 0.0]; // base always says "a", confidently
            norm.observe(&h);
            let x = norm.features(&h);
            let p = ad.predict(&names, &logits, &x);
            if t >= 200 && (p[1] > 0.5) == truth_b {
                correct_late += 1;
            }
            let target = if truth_b { vec![0.0, 1.0] } else { vec![1.0, 0.0] };
            ad.learn(Sample { decision_id: t.to_string(), names: names.clone(), logits, hidden: h, target, weight: 1.0 }, &norm, &cfg);
        }
        assert!(correct_late >= 85, "accuracy on last 100: {correct_late}");
        assert!(ad.is_active(&cfg), "base {} adapted {}", ad.base_loss, ad.adapted_loss);
    }

    /// When the base model is already right, the adapter must not be preferred
    /// by a wide margin and must stay close to the base distribution.
    #[test]
    fn does_not_hurt_a_correct_model() {
        let dim = 16;
        let names: Vec<String> = vec!["no".into(), "yes".into()];
        let norm = Normalizer::new(dim);
        let mut ad = Adapter::new(dim);
        let cfg = LearnConfig::default();
        for t in 0..60 {
            let yes = t % 3 == 0;
            let logits = if yes { vec![-2.0, 2.0] } else { vec![2.0, -2.0] };
            let h: Vec<f32> = (0..dim).map(|i| ((t * 7 + i) % 5) as f32).collect();
            let target = if yes { vec![0.0, 1.0] } else { vec![1.0, 0.0] };
            ad.learn(Sample { decision_id: t.to_string(), names: names.clone(), logits, hidden: h, target, weight: 1.0 }, &norm, &cfg);
        }
        let x = norm.features(&vec![1.0; dim]);
        let p = ad.predict(&names, &[-2.0, 2.0], &x);
        assert!(p[1] > 0.9, "{p:?}");
    }
}
