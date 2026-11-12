//! Answer math shared by serving, learning and evaluation.

use crate::api::{Answer, AnswerMeta, QuestionType};
use crate::compile::CompiledQuestion;
use indexmap::IndexMap;

pub fn softmax(z: &[f32]) -> Vec<f64> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = z.iter().map(|&v| (v as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

pub fn log_softmax(z: &[f32]) -> Vec<f64> {
    let m = z.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + z.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
    z.iter().map(|&v| v as f64 - lse).collect()
}

/// Cross-entropy of `p` against target distribution `y`.
pub fn log_loss(p: &[f64], y: &[f64]) -> f64 {
    p.iter()
        .zip(y)
        .filter(|(_, y)| **y > 0.0)
        .map(|(p, y)| -y * p.max(1e-12).ln())
        .sum()
}

/// Brier score: squared distance between `p` and the target distribution `y`.
pub fn brier(p: &[f64], y: &[f64]) -> f64 {
    p.iter().zip(y).map(|(p, y)| (p - y) * (p - y)).sum()
}

/// One bin of a reliability diagram.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ReliabilityBin {
    pub count: usize,
    /// Mean top probability of the samples in the bin.
    pub confidence: f64,
    /// Fraction of them whose top option was right.
    pub accuracy: f64,
}

/// Equal-width reliability bins over `(top probability, top option was right)`.
pub fn reliability(samples: &[(f64, bool)], bins: usize) -> Vec<ReliabilityBin> {
    let bins = bins.max(1);
    let mut out = vec![ReliabilityBin::default(); bins];
    for &(c, right) in samples {
        let b = &mut out[((c.clamp(0.0, 1.0) * bins as f64) as usize).min(bins - 1)];
        b.count += 1;
        b.confidence += c;
        b.accuracy += f64::from(u8::from(right));
    }
    for b in out.iter_mut().filter(|b| b.count > 0) {
        b.confidence /= b.count as f64;
        b.accuracy /= b.count as f64;
    }
    out
}

/// Expected calibration error: bin-size-weighted gap between confidence and accuracy.
pub fn ece(samples: &[(f64, bool)], bins: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let n = samples.len() as f64;
    reliability(samples, bins).iter().map(|b| b.count as f64 / n * (b.confidence - b.accuracy).abs()).sum()
}

/// TypeSafe's choice confidence: distance of the top probability above uniform.
pub fn choice_confidence(p: &[f64]) -> f64 {
    let k = p.len() as f64;
    let top = p.iter().copied().fold(0.0, f64::max);
    ((top - 1.0 / k) / (1.0 - 1.0 / k)).clamp(0.0, 1.0)
}

/// Score confidence: 1 - E|level - mode| / (K - 1).
pub fn score_confidence(p: &[f64]) -> f64 {
    let d = (p.len() - 1) as f64;
    let mode = argmax(p) as f64;
    let spread: f64 = p.iter().enumerate().map(|(i, p)| p * (i as f64 - mode).abs()).sum();
    (1.0 - spread / d).clamp(0.0, 1.0)
}

pub fn argmax(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
        .0
}

fn round(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

/// Build the typed answer for a question from its option distribution.
pub fn answer(q: &CompiledQuestion, p: &[f64], meta: Option<AnswerMeta>) -> Answer {
    let probs = || -> IndexMap<String, f64> {
        q.options.iter().zip(p).map(|(o, &v)| (o.name.clone(), round(v))).collect()
    };
    match q.kind {
        QuestionType::Noul => Answer::Noul { noul: round(p[1]), x_cev: meta },
        QuestionType::Choice => Answer::Choice {
            choice: q.options[argmax(p)].name.clone(),
            confidence: round(choice_confidence(p)),
            probabilities: probs(),
            x_cev: meta,
        },
        QuestionType::Score => Answer::Score {
            score: round(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum()),
            confidence: round(score_confidence(p)),
            legend: q.legend.clone().unwrap_or_default(),
            probabilities: probs(),
            x_cev: meta,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidences() {
        assert!((choice_confidence(&[0.5, 0.5]) - 0.0).abs() < 1e-9);
        assert!((choice_confidence(&[0.85, 0.15]) - 0.7).abs() < 1e-9);
        assert!((score_confidence(&[0.0, 1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!((score_confidence(&[0.0, 0.57, 0.43]) - (1.0 - 0.43 / 2.0)).abs() < 1e-9);
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9 && argmax(&p) == 2);
    }

    #[test]
    fn calibration_metrics() {
        // A perfect, certain prediction has zero Brier; a confident miss scores 2.
        assert!(brier(&[1.0, 0.0], &[1.0, 0.0]).abs() < 1e-12);
        assert!((brier(&[1.0, 0.0], &[0.0, 1.0]) - 2.0).abs() < 1e-12);
        // Ten samples at 0.9 confidence, nine right: perfectly calibrated.
        let ok: Vec<(f64, bool)> = (0..10).map(|i| (0.9, i != 0)).collect();
        assert!(ece(&ok, 15).abs() < 1e-9);
        // Same confidence, only half right: gap of 0.4.
        let bad: Vec<(f64, bool)> = (0..10).map(|i| (0.9, i % 2 == 0)).collect();
        assert!((ece(&bad, 15) - 0.4).abs() < 1e-9);
        assert_eq!(reliability(&bad, 15).iter().map(|b| b.count).sum::<usize>(), 10);
        // Confidence of exactly 1.0 lands in the last bin instead of overflowing.
        assert_eq!(reliability(&[(1.0, true)], 10)[9].count, 1);
        assert_eq!(ece(&[], 15), 0.0);
    }
}
