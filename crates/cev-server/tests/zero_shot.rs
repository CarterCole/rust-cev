//! Zero-shot accuracy and calibration of a real backbone, per debias mode.
//!
//! It needs the weights, so it is ignored by default:
//!
//! ```text
//! cargo test --release -p cev-server --test zero_shot -- --ignored --nocapture
//! ```
//!
//! `CEV_EVAL_MODEL` picks the model (default `Qwen/Qwen3-0.6B`). The items are
//! the same `eval/items.jsonl` that `scripts/eval_zero_shot.py` reads, so the
//! two report comparable numbers; this one needs no server.

use cev_core::{Answer, Debias};
use cev_model::{Engine, EngineOptions};
use cev_runtime::{Cev, RuntimeConfig, Store};
use serde_json::{Value, json};
use std::sync::Arc;

const LEVELS: [&str; 3] = ["Cosmetic; no impact", "Degraded, workaround exists", "Blocking; no workaround"];
const TEAMS: [(&str, &str); 3] = [("billing", "Payments, invoices, refunds"), ("tech", "Bugs, crashes, errors"), ("sales", "Pricing, new purchases, upgrades")];

fn probs(a: &Answer) -> Vec<f64> {
    match a {
        Answer::Noul { noul, .. } => vec![1.0 - noul, *noul],
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => probabilities.values().copied().collect(),
    }
}

#[derive(Default)]
struct Tally {
    hits: usize,
    loss: f64,
    brier: f64,
    /// (top probability, top option was right)
    top: Vec<(f64, bool)>,
}

impl Tally {
    fn add(&mut self, p: &[f64], target: usize) {
        let top = cev_core::math::argmax(p);
        let mut y = vec![0.0; p.len()];
        y[target] = 1.0;
        self.hits += usize::from(top == target);
        self.loss += cev_core::math::log_loss(p, &y);
        self.brier += cev_core::math::brier(p, &y);
        self.top.push((p[top], top == target));
    }

    fn n(&self) -> f64 {
        self.top.len() as f64
    }
}

#[test]
#[ignore = "needs model weights; run with --ignored"]
fn zero_shot_calibration_by_debias_mode() {
    let model = std::env::var("CEV_EVAL_MODEL").unwrap_or_else(|_| "Qwen/Qwen3-0.6B".into());
    let engine = Engine::load(EngineOptions { model: model.clone(), ..Default::default() }).unwrap();
    let cev = Cev::new(Arc::new(engine), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    let items: Vec<Value> = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../eval/items.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(items.len() >= 40, "{} items", items.len());

    println!("{model}: {} items, ECE over 15 bins pooled over all answers", items.len());
    let mut results = Vec::new();
    for (name, mode) in [("none", Debias::None), ("calibrate", Debias::Calibrate), ("permute", Debias::Permute), ("full", Debias::Full)] {
        let mut all = Tally::default();
        for it in &items {
            let questions = if it["kind"] == "severity" {
                json!({
                    "severity": {"type": "score", "instructions": "How severe is the reported issue?", "criteria": LEVELS},
                    "blocked": {"type": "noul", "instructions": "Is the user completely blocked from using the product?"},
                })
            } else {
                json!({"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                                "criteria": TEAMS.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>()}})
            };
            let req = serde_json::from_value(json!({"state": it["text"], "questions": questions, "no_store": true, "debias": mode})).unwrap();
            let r = cev.decide(&req).unwrap();
            if it["kind"] == "severity" {
                let level = it["level"].as_u64().unwrap() as usize;
                all.add(&probs(&r.answers["severity"]), level);
                all.add(&probs(&r.answers["blocked"]), usize::from(level == 2));
            } else {
                let team = TEAMS.iter().position(|(k, _)| *k == it["team"].as_str().unwrap()).unwrap();
                all.add(&probs(&r.answers["team"]), team);
            }
        }
        let (acc, loss, brier, ece) = (all.hits as f64 / all.n(), all.loss / all.n(), all.brier / all.n(), cev_core::math::ece(&all.top, 15));
        println!("{name:>9}: accuracy {acc:.2}  logloss {loss:.3}  brier {brier:.3}  ece {ece:.3}");
        assert!([acc, loss, brier, ece].iter().all(|v| v.is_finite()));
        results.push((name, acc, ece));
    }
    // A sanity floor, not a benchmark: the zero-shot model must beat guessing
    // (the questions have 2 to 3 options).
    let full = results.iter().find(|r| r.0 == "full").unwrap();
    assert!(full.1 > 0.55, "full-debias accuracy {:.2}", full.1);
}
