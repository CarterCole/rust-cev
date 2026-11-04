//! Qwen3.5 decoder against HF transformers (f32, CPU). Needs the reference
//! file written by `training/parity/dump_qwen35.py`:
//!
//!     CEV_PARITY_QWEN35=/path/to/qwen35_0.8b.json cargo test -p cev-model --test parity_qwen35 -- --nocapture

use candle_core::{Device, Tensor};
use cev_model::decoder::{Attn, Decoder, Positions};
use cev_model::engine::EngineOptions;
use serde::Deserialize;

#[derive(Deserialize)]
struct Row {
    ids: Vec<u32>,
    hidden_last: Vec<f32>,
    hidden_mid: Vec<f32>,
    logits_last: Vec<f32>,
    probe_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Dump {
    model: String,
    rows: Vec<Row>,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
    max_abs_diff(a, b) / scale
}

#[test]
fn matches_transformers() {
    let Ok(path) = std::env::var("CEV_PARITY_QWEN35") else {
        eprintln!("skipped: set CEV_PARITY_QWEN35 to the dump file");
        return;
    };
    let dump: Dump = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let files = cev_model::hub::backbone(&dump.model).unwrap();
    let opts = EngineOptions { model: dump.model.clone(), device: "cpu".into(), dtype: "f32".into(), ..Default::default() };
    let (model, _cfg, _) = cev_model::load::qwen35(&files, None, &opts, &Device::Cpu).unwrap();
    for (i, row) in dump.rows.iter().enumerate() {
        let t = row.ids.len();
        // Whole row at once.
        let emb = model.embed(&row.ids).unwrap().unsqueeze(0).unwrap();
        let (h, _) = model.forward(&emb, &Positions::text(0, t), None, false, &Attn::Causal).unwrap();
        let last: Vec<f32> = h.get(0).unwrap().get(t - 1).unwrap().to_vec1().unwrap();
        let mid: Vec<f32> = h.get(0).unwrap().get(t / 2).unwrap().to_vec1().unwrap();
        let r_last = rel(&last, &row.hidden_last);
        let r_mid = rel(&mid, &row.hidden_mid);
        let head = model.head_rows(&row.probe_ids).unwrap();
        let logits: Vec<f32> = Tensor::new(last.as_slice(), &Device::Cpu).unwrap().unsqueeze(0).unwrap().matmul(&head.t().unwrap()).unwrap().squeeze(0).unwrap().to_vec1().unwrap();
        let d_logit = max_abs_diff(&logits, &row.logits_last);
        eprintln!("row {i} ({t} tokens): rel err last {r_last:.2e} mid {r_mid:.2e}; logits max |Δ| {d_logit:.2e}");
        assert!(r_last < 2e-3 && r_mid < 2e-3, "row {i}: hidden state mismatch");
        assert!(d_logit < 2e-2, "row {i}: logits mismatch {logits:?} vs {:?}", row.logits_last);

        // Prefix state then the rest as a branch, in two prefill chunks.
        let split = t / 2;
        let c1 = split / 2;
        let e1 = model.embed(&row.ids[..c1]).unwrap().unsqueeze(0).unwrap();
        let (_, s1) = model.forward(&e1, &Positions::text(0, c1), None, true, &Attn::Causal).unwrap();
        let e2 = model.embed(&row.ids[c1..split]).unwrap().unsqueeze(0).unwrap();
        let (_, s2) = model.forward(&e2, &Positions::text(c1 as u32, split - c1), s1.as_ref(), true, &Attn::Causal).unwrap();
        let state = s2.unwrap();
        assert_eq!(state.len, split);
        // Two rows: the real branch, and a shorter row (right-padded) that is a prefix of it.
        let branch = &row.ids[split..];
        let short = &branch[..branch.len() / 2];
        let d = model.hidden_size();
        let mut ids = branch.to_vec();
        ids.extend_from_slice(short);
        ids.extend(std::iter::repeat_n(0u32, branch.len() - short.len()));
        let emb = model.embed(&ids).unwrap().reshape((2, branch.len(), d)).unwrap();
        let (hb, _) = model.forward(&emb, &Positions::text(split as u32, branch.len()), Some(&state), false, &Attn::Causal).unwrap();
        let last_b: Vec<f32> = hb.get(0).unwrap().get(branch.len() - 1).unwrap().to_vec1().unwrap();
        let r_branch = rel(&last_b, &row.hidden_last);
        // The short row's last real token equals the full row at that position.
        let pos = split + short.len() - 1;
        let want: Vec<f32> = h.get(0).unwrap().get(pos).unwrap().to_vec1().unwrap();
        let got: Vec<f32> = hb.get(1).unwrap().get(short.len() - 1).unwrap().to_vec1().unwrap();
        let r_short = rel(&got, &want);
        eprintln!("row {i}: prefix+branch rel err {r_branch:.2e}; padded short row rel err {r_short:.2e}");
        assert!(r_branch < 2e-3, "row {i}: prefix/branch mismatch");
        assert!(r_short < 2e-3, "row {i}: padded row mismatch");
    }
}
