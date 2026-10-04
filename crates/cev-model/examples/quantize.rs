//! Write an int8 copy of a Qwen3 checkpoint (see `cev_model::quant`):
//!
//!     cargo run --release -p cev-model --example quantize -- Qwen/Qwen3-0.6B web/models/Qwen3-0.6B-q8
//!
//! The output directory loads like any other model (`cev --model <dir>`, CPU).
use anyhow::Context;
use candle_core::{DType, Device};
use cev_model::quant::{SCALE_SUFFIX, quantize};
use std::collections::HashMap;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (model, out) = match &args[1..] {
        [model, out] => (model, Path::new(out)),
        _ => anyhow::bail!("usage: quantize <model dir or repo id> <output dir>"),
    };
    let files = cev_model::hub::resolve(model)?;
    std::fs::create_dir_all(out)?;
    let mut tensors = HashMap::new();
    for w in &files.weights {
        for (name, t) in candle_core::safetensors::load(w, &Device::Cpu)? {
            if name.ends_with(".weight") && t.rank() == 2 {
                let (q, scale) = quantize(&t)?;
                tensors.insert(format!("{name}{SCALE_SUFFIX}"), scale);
                tensors.insert(name, q);
            } else {
                tensors.insert(name, t.to_dtype(DType::F32)?);
            }
        }
    }
    let weights = out.join("model.safetensors");
    candle_core::safetensors::save(&tensors, &weights)?;
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
        let src = files.dir.join(f);
        if src.exists() {
            std::fs::copy(&src, out.join(f)).with_context(|| format!("copy {f}"))?;
        }
    }
    println!("{} tensors, {:.0} MB -> {}", tensors.len(), std::fs::metadata(&weights)?.len() as f64 / 1e6, weights.display());
    Ok(())
}
