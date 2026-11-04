//! Whole decision models against reference answers computed from their
//! published parts in PyTorch (`training/parity/dump_*.py`):
//!
//!     CEV_PARITY_KEV=/path/kev_0.8b.json cargo test --release -p cev-model --test parity_models -- --nocapture
//!     CEV_PARITY_D3=/path/d3_nano.json  cargo test --release -p cev-model --test parity_models -- --nocapture
//!
//! Each file names the Hugging Face repo it was computed from; cev downloads
//! (or finds in the HF cache) the same files and must give the same
//! probabilities in f32 on the CPU.

use cev_core::{SystemOneRequest, compile, math};
use cev_model::engine::EngineOptions;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Answer {
    probs: Vec<f64>,
    #[serde(default)]
    tokens: usize,
}


#[derive(Deserialize)]
struct Row {
    request: SystemOneRequest,
    answers: BTreeMap<String, Answer>,
}

#[derive(Deserialize)]
struct Dump {
    repo: String,
    rows: Vec<Row>,
}

fn check(var: &str, tol: f64) {
    let Ok(path) = std::env::var(var) else {
        eprintln!("skipped: set {var} to the dump file");
        return;
    };
    // bf16 on an accelerator is a different computation: allow the usual drift.
    let tol = if std::env::var("CEV_PARITY_DTYPE").is_ok_and(|d| d != "f32") { tol.max(3e-2) } else { tol };
    let dump: Dump = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let backend = cev_model::load(EngineOptions { model: dump.repo.clone(), device: std::env::var("CEV_PARITY_DEVICE").unwrap_or_else(|_| "cpu".into()), dtype: std::env::var("CEV_PARITY_DTYPE").unwrap_or_else(|_| "f32".into()), ..Default::default() }).unwrap();
    let caps = backend.capabilities();
    eprintln!("{} ({}): {:?}", backend.id(), caps.family.as_str(), caps.modalities());
    let mut worst = 0f64;
    for (i, row) in dump.rows.iter().enumerate() {
        let compiled = compile(&row.request, caps.max_options).unwrap();
        let out = backend.run(&compiled).unwrap();
        for (q, r) in compiled.questions.iter().zip(&out.readouts) {
            let p = math::softmax(&r.logits);
            let want = &dump.rows[i].answers[&q.id].probs;
            let d = p.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            worst = worst.max(d);
            eprintln!("  {:<12} cev {:?} ref {:?} (max |Δp| {d:.2e}, ref {} tokens)", q.id, p.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(), want.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(), row.answers[&q.id].tokens);
            assert!(d < tol, "{}: probabilities differ by {d}", q.id);
            assert!(r.hidden.len() == backend.hidden_size());
        }
        // A second pass hits the prefix cache and must agree with the first.
        let again = backend.run(&compiled).unwrap();
        for (a, b) in out.readouts.iter().zip(&again.readouts) {
            assert!(a.logits.iter().zip(&b.logits).all(|(x, y)| (x - y).abs() < 1e-4), "cached pass differs");
        }
        if !matches!(caps.family, cev_core::Family::D3 | cev_core::Family::D1Omni) {
            assert!(again.prefix_cached);
        }
        if !row.request.images.is_empty() && caps.family != cev_core::Family::D1Omni {
            assert!(out.prefix.contains("<|image_start|>") || out.prefix.contains("<|vision_start|>"), "stored prefix should show the image markup");
        }
    }
    eprintln!("worst |Δp| = {worst:.2e}");
}

#[test]
fn kev_matches_reference() {
    check("CEV_PARITY_KEV", 2e-3);
}

#[test]
fn d3_matches_reference() {
    check("CEV_PARITY_D3", 2e-3);
}

/// d1's image question goes through our own resize, so it gets a looser bound.
#[test]
fn d1_matches_reference() {
    check("CEV_PARITY_D1", 1.5e-2);
}

/// d1-omni: text, image (own resize) and audio (own mel front end).
#[test]
fn omni_matches_reference() {
    check("CEV_PARITY_OMNI", 1.5e-2);
}

/// Nimble: the 9B base with the LoRA merged, the schema prompt and the code readout.
#[test]
fn nimble_matches_reference() {
    check("CEV_PARITY_NIMBLE", 2e-3);
}
