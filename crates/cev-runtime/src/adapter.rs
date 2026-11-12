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
//!
//! "Wins" means the mean per-label loss gap over a recent window is negative by
//! more than its standard error, so a handful of lucky labels cannot switch an
//! overfit probe on. The probe itself can be count-sketched down to
//! [`LearnConfig::probe_dim`] features, which keeps it from memorizing a few
//! labels in a 2048-dimensional space.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
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
    /// L2 pull toward the base model, per step, on the bias and temperature.
    pub l2: f32,
    /// Same for the probe weights. Raising it much above `l2` stops the probe
    /// from learning dense signals at all, because features are scaled by
    /// `1/√d`; the activation test, not this, is what guards against overfit.
    pub l2_w: f32,
    /// Width of the probe, by count-sketching the hidden state down to this
    /// many features. `0`, or anything at least the hidden size, keeps it full
    /// (the default: a sketch loses signal spread across many dimensions).
    pub probe_dim: usize,
    /// Per-label loss gaps kept for the activation test.
    pub gate_window: usize,
    /// The adapter is served once its mean loss gap to the base model is below
    /// minus `gate_z` standard errors.
    pub gate_z: f64,
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
            l2_w: 1e-3,
            probe_dim: 0,
            gate_window: 64,
            gate_z: 0.8,
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
    /// The same for the Brier score.
    #[serde(default)]
    pub base_brier: f64,
    #[serde(default)]
    pub adapted_brier: f64,
    /// Width the hidden state is sketched to before the probe; `0` is full.
    #[serde(default)]
    probe: usize,
    /// Recent per-label `adapted - base` log loss, oldest first.
    #[serde(default)]
    gaps: Vec<f64>,
    /// Rebuilt from the store on load, never serialized.
    #[serde(skip)]
    buffer: Vec<Sample>,
}

fn softmax(z: &[f32]) -> Vec<f64> {
    cev_core::math::softmax(z)
}

impl Adapter {
    pub fn new(dim: usize) -> Self {
        Self::with_probe(dim, 0)
    }

    /// An adapter whose probe sees the hidden state sketched down to `probe`
    /// features (`0` for all of them).
    pub fn with_probe(dim: usize, probe: usize) -> Self {
        Self {
            dim,
            tau: 0.0,
            options: IndexMap::new(),
            examples: 0,
            base_loss: 0.0,
            adapted_loss: 0.0,
            base_brier: 0.0,
            adapted_brier: 0.0,
            probe,
            gaps: Vec::new(),
            buffer: Vec::new(),
        }
    }

    /// A copy for a possibly different backbone, to be earned again there.
    ///
    /// The temperature and per-option biases depend only on the option logits
    /// and option names, so they carry over to any backbone. The probe weights
    /// only mean something on the same hidden size and sketch width; otherwise
    /// they restart at zero. Evidence counters restart either way, so the
    /// adapter must beat the base model on new labels before it is served.
    pub fn transplant(&self, dim: usize, probe: usize) -> Adapter {
        let mut a = Adapter::with_probe(dim, probe);
        a.tau = self.tau;
        let width = a.width();
        let keep = self.dim == dim && self.width() == width;
        for (name, o) in &self.options {
            let w = if keep && o.w.len() == width { o.w.clone() } else { vec![0.0; width] };
            a.options.insert(name.clone(), OptionParams { b: o.b, w });
        }
        a
    }

    /// Number of weights per option.
    fn width(&self) -> usize {
        if self.probe > 0 && self.probe < self.dim { self.probe } else { self.dim }
    }

    /// Count-sketch of the standardized features: every input dimension is
    /// added, with a fixed pseudo-random sign, to one of `width()` buckets.
    /// That keeps squared norms in expectation, needs no stored matrix, and
    /// costs one pass over the input.
    fn project<'a>(&self, x: &'a [f32]) -> Cow<'a, [f32]> {
        let k = self.width();
        if k == self.dim || x.len() != self.dim {
            return Cow::Borrowed(x);
        }
        let mut out = vec![0.0f32; k];
        for (i, &v) in x.iter().enumerate() {
            let mut h = (i as u64).wrapping_add(0x9E3779B97F4A7C15);
            h = (h ^ (h >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            h = (h ^ (h >> 27)).wrapping_mul(0x94D049BB133111EB);
            h ^= h >> 31;
            let sign = if (h >> 40) & 1 == 0 { 1.0 } else { -1.0 };
            out[(h as u32 as u64 % k as u64) as usize] += sign * v;
        }
        Cow::Owned(out)
    }

    /// Mean and standard error of the recent `adapted - base` log loss gaps.
    pub fn gap(&self) -> Option<(f64, f64)> {
        let n = self.gaps.len();
        if n < 2 {
            return None;
        }
        let mean = self.gaps.iter().sum::<f64>() / n as f64;
        let var = self.gaps.iter().map(|g| (g - mean) * (g - mean)).sum::<f64>() / (n - 1) as f64;
        Some((mean, (var / n as f64).sqrt()))
    }

    pub fn temperature(&self) -> f32 {
        self.tau.exp()
    }

    /// Served only once enough labels have been scored *and* the adapter is
    /// ahead of the base model by more than the noise in that comparison.
    pub fn is_active(&self, cfg: &LearnConfig) -> bool {
        if self.examples < cfg.min_examples || (self.gaps.len() as u64) < cfg.min_examples {
            return false;
        }
        self.gap().is_some_and(|(mean, se)| mean + cfg.gate_z * se < 0.0)
    }

    pub fn logits(&self, names: &[String], z: &[f32], x: &[f32]) -> Vec<f32> {
        self.logits_projected(names, z, &self.project(x))
    }

    fn logits_projected(&self, names: &[String], z: &[f32], x: &[f32]) -> Vec<f32> {
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

    fn predict_projected(&self, names: &[String], z: &[f32], x: &[f32]) -> Vec<f64> {
        softmax(&self.logits_projected(names, z, x))
    }

    /// Score `s` prequentially, add it to the buffer, and refit.
    pub fn learn(&mut self, s: Sample, norm: &Normalizer, cfg: &LearnConfig) {
        let x = norm.features(&s.hidden);
        let base_p = softmax(&s.logits);
        let adapted_p = self.predict(&s.names, &s.logits, &x);
        let base = cev_core::math::log_loss(&base_p, &s.target);
        let adapted = cev_core::math::log_loss(&adapted_p, &s.target);
        let base_b = cev_core::math::brier(&base_p, &s.target);
        let adapted_b = cev_core::math::brier(&adapted_p, &s.target);
        let d = cfg.loss_decay;
        if self.examples == 0 {
            (self.base_loss, self.adapted_loss) = (base, adapted);
            (self.base_brier, self.adapted_brier) = (base_b, adapted_b);
        } else {
            self.base_loss = d * self.base_loss + (1.0 - d) * base;
            self.adapted_loss = d * self.adapted_loss + (1.0 - d) * adapted;
            self.base_brier = d * self.base_brier + (1.0 - d) * base_b;
            self.adapted_brier = d * self.adapted_brier + (1.0 - d) * adapted_b;
        }
        self.gaps.push(adapted - base);
        let keep = cfg.gate_window.max(2);
        if self.gaps.len() > keep {
            let drop = self.gaps.len() - keep;
            self.gaps.drain(..drop);
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
        let feats: Vec<Vec<f32>> = self.buffer.iter().map(|s| self.project(&norm.features(&s.hidden)).into_owned()).collect();
        let width = self.width();
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
                let p = self.predict_projected(&s.names, &s.logits, x);
                let inv_t = (-self.tau).exp();
                let mut g_tau = 0.0f32;
                for (k, name) in s.names.iter().enumerate() {
                    let g = (p[k] - s.target[k]) as f32 * s.weight;
                    g_tau += g * -(s.logits[k] * inv_t);
                    let o = self
                        .options
                        .entry(name.clone())
                        .or_insert_with(|| OptionParams { b: 0.0, w: vec![0.0; width] });
                    o.b -= cfg.lr_b * (g + cfg.l2 * o.b);
                    for (w, &xi) in o.w.iter_mut().zip(x) {
                        *w -= cfg.lr_w * (g * xi + cfg.l2_w * *w);
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

    #[test]
    fn transplant_keeps_priors_and_restarts_evidence() {
        let names: Vec<String> = vec!["a".into(), "b".into()];
        let cfg = LearnConfig::default();
        let mut norm = Normalizer::new(16);
        let mut ad = Adapter::new(16);
        for t in 0..40 {
            let h: Vec<f32> = (0..16).map(|i| ((t * 5 + i) % 7) as f32).collect();
            norm.observe(&h);
            // The base always says "a"; the truth is always "b".
            ad.learn(Sample { decision_id: t.to_string(), names: names.clone(), logits: vec![1.0, 0.0], hidden: h, target: vec![0.0, 1.0], weight: 1.0 }, &norm, &cfg);
        }
        assert!(ad.is_active(&cfg));

        // Same hidden size: everything learned carries over, evidence does not.
        let same = ad.transplant(16, 0);
        assert!(!same.is_active(&cfg) && same.examples == 0 && same.buffer_len() == 0);
        assert_eq!(same.temperature(), ad.temperature());
        let x = vec![0.1; 16];
        assert_eq!(same.predict(&names, &[1.0, 0.0], &x), ad.predict(&names, &[1.0, 0.0], &x));

        // Different hidden size: bias and temperature survive, the probe restarts.
        let other = ad.transplant(32, 0);
        assert_eq!(other.dim, 32);
        let p = other.predict(&names, &[1.0, 0.0], &[0.5; 32]);
        let q = ad.predict(&names, &[1.0, 0.0], &[0.0; 16]);
        assert!(p[1] > 0.5 && q[1] > 0.5, "{p:?} {q:?}");
        // And it can still learn on the new width.
        assert!(other.options.values().all(|o| o.w.len() == 32));
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

    /// Labels that carry no signal must not switch the adapter on. Over many
    /// seeds, only the occasional lucky run may pass the evidence test.
    #[test]
    fn random_labels_do_not_activate() {
        let dim = 256;
        let names: Vec<String> = vec!["a".into(), "b".into()];
        let cfg = LearnConfig::default();
        let trials = 100;
        let mut active = 0;
        for seed in 0..trials {
            let mut rng = 0x2545F4914F6CDD1Du64 ^ (seed as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
            let mut rand = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 11) as f32 / (1u64 << 53) as f32
            };
            let norm = Normalizer::new(dim);
            let mut ad = Adapter::new(dim);
            for t in 0..16 {
                let h: Vec<f32> = (0..dim).map(|_| rand() * 2.0 - 1.0).collect();
                let target = if rand() < 0.5 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
                let s = Sample { decision_id: t.to_string(), names: names.clone(), logits: vec![0.3, -0.3], hidden: h, target, weight: 1.0 };
                ad.learn(s, &norm, &cfg);
            }
            active += usize::from(ad.is_active(&cfg));
        }
        // Measured about 4% here; the old decayed-average rule gave about 20%.
        assert!(active <= trials / 10, "{active}/{trials} adapters activated on random labels");
    }

    /// A sketched probe keeps far fewer weights than the hidden size and still
    /// learns a strong dense signal.
    #[test]
    fn sketched_probe_learns() {
        let dim = 512;
        let names: Vec<String> = vec!["a".into(), "b".into()];
        let mut norm = Normalizer::new(dim);
        let cfg = LearnConfig { probe_dim: 32, ..LearnConfig::default() };
        let mut ad = Adapter::with_probe(dim, cfg.probe_dim);
        let mut rng = 7u64;
        let mut rand = move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        let shift: Vec<f32> = (0..dim).map(|i| if i % 3 == 0 { 0.5 } else { -0.5 }).collect();
        let mut right = 0;
        for t in 0..400 {
            let truth_b = rand() > 0.0;
            let sign = if truth_b { 1.0 } else { -1.0 };
            let h: Vec<f32> = (0..dim).map(|i| rand() + sign * shift[i]).collect();
            let logits = vec![0.3, -0.3];
            norm.observe(&h);
            if t >= 300 && (ad.predict(&names, &logits, &norm.features(&h))[1] > 0.5) == truth_b {
                right += 1;
            }
            let target = if truth_b { vec![0.0, 1.0] } else { vec![1.0, 0.0] };
            ad.learn(Sample { decision_id: t.to_string(), names: names.clone(), logits, hidden: h, target, weight: 1.0 }, &norm, &cfg);
        }
        assert!(right >= 80, "accuracy on last 100: {right}");
        assert!(ad.options.values().all(|o| o.w.len() == 32));
    }
}
