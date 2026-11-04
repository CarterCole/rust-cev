//! Load any supported model and answer one request, with per-phase timings.
//!
//!     cargo run --release -p cev-model --features metal --example decide -- kev:0.8b [device] [dtype]
use cev_core::{compile, math};
use cev_model::engine::EngineOptions;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,cev_model=debug".into())).init();
    let args: Vec<String> = std::env::args().collect();
    let t = std::time::Instant::now();
    let backend = cev_model::load(EngineOptions {
        model: args.get(1).cloned().unwrap_or("kev:0.8b".into()),
        device: args.get(2).cloned().unwrap_or("auto".into()),
        dtype: args.get(3).cloned().unwrap_or("auto".into()),
        ..Default::default()
    })?;
    let caps = backend.capabilities();
    println!("loaded {} ({}, {:?}) in {:.1}s", backend.id(), caps.family.as_str(), caps.modalities(), t.elapsed().as_secs_f64());
    let req = serde_json::from_value(serde_json::json!({
        "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}},
            "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
            "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}
        }
    }))?;
    let c = compile(&req, caps.max_options)?;
    for round in 0..3 {
        let t = std::time::Instant::now();
        let out = backend.run(&c)?;
        println!("round {round}: {:.1} ms, {} tokens, prefix cached={}", t.elapsed().as_secs_f64() * 1e3, out.input_tokens, out.prefix_cached);
        if round == 0 {
            for (q, r) in c.questions.iter().zip(&out.readouts) {
                println!("  {:12} {}", q.id, serde_json::to_string(&math::answer(q, &math::softmax(&r.logits), None))?);
            }
        }
    }
    Ok(())
}
