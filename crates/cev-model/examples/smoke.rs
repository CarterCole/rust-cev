//! cargo run --release -p cev-model --features metal --example smoke -- [model] [device]
use cev_core::{Backend, compile, math};
use cev_model::{Engine, EngineOptions};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let engine = Engine::load(EngineOptions {
        model: args.get(1).cloned().unwrap_or("Qwen/Qwen3-0.6B".into()),
        device: args.get(2).cloned().unwrap_or("auto".into()),
        ..Default::default()
    })?;
    println!("codes: {} (first {:?}, last {:?}) think_stub={}", engine.codes().len(), &engine.codes()[..3], engine.codes().last(), engine.format().think_stub);
    let req = serde_json::from_value(serde_json::json!({
        "state": "Hi, I was charged twice for my subscription this month. Please refund the duplicate charge. The app works fine otherwise.",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "bug": {"type": "noul", "instructions": "Is the customer reporting a software bug?"},
            "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                     "criteria": {"billing": "Payments, invoices, refunds", "tech": "Bugs, crashes, errors", "sales": "New purchases and upgrades"}},
            "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]}
        }
    }))?;
    let c = compile(&req, engine.codes(), engine.format())?;
    for round in 0..3 {
        let t = std::time::Instant::now();
        let out = engine.run(&c)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("round {round}: {ms:.1} ms, {} tokens, cached={}", out.input_tokens, out.prefix_cached);
        if round == 0 {
            for (q, r) in c.questions.iter().zip(&out.readouts) {
                let p = math::softmax(&r.logits);
                println!("  {:8} {}", q.id, serde_json::to_string(&math::answer(q, &p, None))?);
            }
        }
    }
    Ok(())
}
