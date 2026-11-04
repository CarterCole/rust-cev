//! Qwen3.5 vision tower and image prompts against HF transformers, from the
//! reference written by `training/parity/dump_qwen35_vision.py`:
//!
//!     CEV_PARITY_VISION=/path/d3_vision.json cargo test --release -p cev-model --test parity_vision -- --nocapture

use base64::Engine as _;
use candle_core::{Device, Tensor};
use cev_core::{SystemOneRequest, compile, math};
#[allow(unused_imports)]
use cev_model::media::Pixels;
use cev_model::engine::EngineOptions;
use cev_model::media::{QwenImageParams, decode_image, qwen_patches};
use cev_model::qwen35_vision::Grid;
use serde::Deserialize;

#[derive(Deserialize)]
struct Dump {
    repo: String,
    grid_thw: [usize; 3],
    pixel_values: Vec<Vec<f32>>,
    pooled: Vec<Vec<f32>>,
    question: serde_json::Value,
    probs: Vec<f64>,
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1e-6);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / scale
}

#[test]
fn vision_matches_transformers() {
    let Ok(path) = std::env::var("CEV_PARITY_VISION") else {
        eprintln!("skipped: set CEV_PARITY_VISION to the dump file");
        return;
    };
    let dump: Dump = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let png = std::fs::read(path.replace(".json", ".png")).unwrap();
    let files = cev_model::hub::backbone(&dump.repo).unwrap();
    let opts = EngineOptions { model: dump.repo.clone(), device: std::env::var("CEV_PARITY_DEVICE").unwrap_or_else(|_| "cpu".into()), dtype: std::env::var("CEV_PARITY_DTYPE").unwrap_or_else(|_| "f32".into()), ..Default::default() };
    let device = cev_model::load::pick_device(&opts.device).unwrap();
    let (_model, _cfg, vision) = cev_model::load::qwen35(&files, None, &opts, &device).unwrap();
    let tower = vision.expect("checkpoint has a vision tower");
    let grid = Grid { t: dump.grid_thw[0], h: dump.grid_thw[1], w: dump.grid_thw[2] };

    // 1. Our preprocessing of the same PNG against the HF processor's patches.
    let img = decode_image(&png).unwrap();
    let ours = qwen_patches(&img, &QwenImageParams::default()).unwrap();
    assert_eq!((ours.grid.h, ours.grid.w), (grid.h, grid.w));
    let hf_pixels: Vec<f32> = dump.pixel_values.iter().flatten().copied().collect();
    let d_pix = rel(&ours.pixels, &hf_pixels);
    eprintln!("patches: {} × {}, preprocessing rel err {d_pix:.2e}", ours.grid.patches(), ours.patch_dim);
    assert!(d_pix < 1e-2, "patch layout or normalisation differs");

    // 2. The tower on HF's exact patches.
    let n = dump.pixel_values.len();
    let px = Tensor::from_vec(hf_pixels, (n, dump.pixel_values[0].len()), &device).unwrap();
    let pooled = tower.forward(&px, grid).unwrap().to_dtype(candle_core::DType::F32).unwrap().to_device(&Device::Cpu).unwrap();
    let got: Vec<Vec<f32>> = pooled.to_vec2().unwrap();
    assert_eq!(got.len(), dump.pooled.len());
    let worst = got.iter().zip(&dump.pooled).map(|(a, b)| rel(a, b)).fold(0.0, f32::max);
    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        dot / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt()).max(1e-12)
    };
    let min_cos = got.iter().zip(&dump.pooled).map(|(a, b)| cos(a, b)).fold(1.0, f32::min);
    let mean_cos = got.iter().zip(&dump.pooled).map(|(a, b)| cos(a, b)).sum::<f32>() / got.len() as f32;
    eprintln!("tower output {} × {}: worst rel err {worst:.2e}, cosine min {min_cos:.4} mean {mean_cos:.4}", got.len(), got[0].len());
    let f32_run = std::env::var("CEV_PARITY_DTYPE").map_or(true, |d| d == "f32");
    // f32 must match elementwise. In bf16 a few channels grow past 5,000 by the
    // last block, where bf16 resolves steps of 32, so tokens with small outputs
    // drift (min cosine ~0.85); the tower is held to its mean direction and the
    // answer below to the bf16 tolerance.
    if f32_run {
        assert!(worst < 2e-3, "vision tower mismatch");
    } else {
        assert!(mean_cos > 0.97, "vision tower mismatch (mean cosine {mean_cos})");
    }

    // 3. The whole image question through the d3 backend.
    let backend = cev_model::load(opts).unwrap();
    assert!(backend.capabilities().images);
    let req: SystemOneRequest = serde_json::from_value(serde_json::json!({
        "state": "A photo.",
        "images": [base64::engine::general_purpose::STANDARD.encode(&png)],
        "questions": {"q": dump.question}
    }))
    .unwrap();
    let compiled = compile(&req, 255).unwrap();
    let out = backend.run(&compiled).unwrap();
    let p = math::softmax(&out.readouts[0].logits);
    let d = p.iter().zip(&dump.probs).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("image question: cev {:?} ref {:?} (max |Δp| {d:.2e}, {} tokens)", p.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(), dump.probs.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(), out.input_tokens);
    // The resize is reproduced up to torchvision's fixed-point 8-bit rounding.
    assert!(d < if f32_run { 1.5e-2 } else { 3e-2 }, "image question probabilities differ by {d}");
}
