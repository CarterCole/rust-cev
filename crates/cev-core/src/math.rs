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
}
