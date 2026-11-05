//! The browser's load path (files in memory, weights streamed in chunks)
//! must give the same answers as loading from disk. Needs the checkpoints in
//! the Hugging Face cache; run with `CEV_MEMORY_LOAD=1`:
//!
//!     CEV_MEMORY_LOAD=1 cargo test --release -p cev-model --test memory_load -- --nocapture

use cev_core::{Backend, SystemOneRequest, compile};
use cev_model::hub::RepoFiles;
use cev_model::load::{Detected, detect, stream_loader};
use cev_model::{EngineOptions, MemoryModel, from_memory, load};
use std::collections::HashMap;
use std::io::Read;

fn opts(model: &str) -> EngineOptions {
    EngineOptions { model: model.into(), device: "cpu".into(), dtype: "f32".into(), max_context: 8192, ..Default::default() }
}

fn stream(kind: &str, files: &RepoFiles) -> HashMap<String, candle_core::Tensor> {
    let mut l = stream_loader(kind);
    for w in files.weights().unwrap() {
        let mut f = std::fs::File::open(w).unwrap();
        let mut buf = vec![0u8; 3 << 20];
        loop {
            let n = f.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            l.push(&buf[..n]).unwrap();
        }
    }
    l.finish().unwrap()
}

fn read(files: &RepoFiles, names: &[&str]) -> HashMap<String, Vec<u8>> {
    names.iter().filter(|n| files.has(n)).map(|n| (n.to_string(), files.read(n).unwrap())).collect()
}

fn png() -> Vec<u8> {
    let img = image::RgbImage::from_fn(96, 64, |x, y| image::Rgb([(x * 2) as u8, (y * 3) as u8, 128]));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn wav() -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec { channels: 1, sample_rate: 16000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::new(&mut out, spec).unwrap();
    for i in 0..16000 {
        w.write_sample(((i as f32 * 0.05).sin() * 8000.0) as i16).unwrap();
    }
    w.finalize().unwrap();
    out.into_inner()
}

fn request(media: serde_json::Value) -> SystemOneRequest {
    use base64::Engine as _;
    let mut v = serde_json::json!({
        "state": {"ticket": "The app crashes when I tap Log in. I was charged twice and want one payment back."},
        "questions": {
            "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Payments, refunds", "tech": "Bugs, crashes", "sales": "Pricing"}},
            "refund": {"type": "noul", "instructions": "Is the user asking for a refund?"},
            "severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["Cosmetic", "Degraded", "Blocking"]}
        }
    });
    let b64 = |b: Vec<u8>| base64::engine::general_purpose::STANDARD.encode(b);
    if media == "image" {
        v["images"] = serde_json::json!([b64(png())]);
    } else if media == "audio" {
        v["audio"] = serde_json::json!(b64(wav()));
    }
    serde_json::from_value(v).unwrap()
}

fn compare(name: &str, disk: &dyn Backend, mem: &dyn Backend, media: &[&str]) {
    assert_eq!(disk.capabilities().modalities(), mem.capabilities().modalities());
    for m in media {
        let req = compile(&request(serde_json::json!(m)), disk.capabilities().max_options).unwrap();
        let (a, b) = (disk.run(&req).unwrap(), mem.run(&req).unwrap());
        assert_eq!(a.input_tokens, b.input_tokens);
        let mut worst = 0f32;
        for (ra, rb) in a.readouts.iter().zip(&b.readouts) {
            for (x, y) in ra.logits.iter().zip(&rb.logits) {
                worst = worst.max((x - y).abs());
            }
        }
        println!("{name} {m}: {} tokens, worst |Δlogit| {worst:.2e}", a.input_tokens);
        assert!(worst < 1e-3, "{name} {m}: in-memory load differs by {worst}");
    }
}

#[test]
fn memory_load_matches_disk() {
    if std::env::var_os("CEV_MEMORY_LOAD").is_none() {
        eprintln!("skipped: set CEV_MEMORY_LOAD=1");
        return;
    }
    // Kev: adapter + base, merged in memory, bf16 embedding table.
    let Detected::Kev { adapter, base } = detect("kev:0.8b").unwrap() else { panic!("kev:0.8b is not Kev") };
    let mut files = read(&adapter, &["adapter_config.json", "adapter_model.safetensors", "head.pt"]);
    files.extend(read(&base, &["config.json", "tokenizer.json"]));
    let mem = from_memory(MemoryModel { repo: adapter.repo.clone(), files, tensors: stream("kev", &base) }, opts("kev:0.8b")).unwrap();
    let disk = load(opts("kev:0.8b")).unwrap();
    assert_eq!(disk.id(), mem.id());
    compare("kev", disk.as_ref(), mem.as_ref(), &["text"]);
    drop((disk, mem));

    let Detected::Backbone { files: omni, .. } = detect("d1-omni").unwrap() else { panic!("d1-omni is not a backbone") };
    let mem = from_memory(MemoryModel { repo: omni.repo.clone(), files: read(&omni, &["config.json", "tokenizer.json"]), tensors: stream("d1-omni", &omni) }, opts("d1-omni")).unwrap();
    let disk = load(opts("d1-omni")).unwrap();
    assert_eq!(disk.id(), mem.id());
    compare("d1-omni", disk.as_ref(), mem.as_ref(), &["text", "image", "audio"]);
}
