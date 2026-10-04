//! A support-ticket router where the branches are Rust enums the model picks.
//!
//!     cargo run --release -p cev-rs --features local,metal --example router -- Qwen/Qwen3-1.7B
//!
//! Runs the model in-process (no server). Decisions and feedback are logged
//! to `router.db`, so corrections accumulate across runs.

use cev::{Cev, Choice};
use cev_model::{Engine, EngineOptions};
use cev_runtime::{RuntimeConfig, Store};
use serde_json::json;
use std::sync::Arc;

#[derive(Choice, Debug, Clone, Copy, PartialEq)]
#[cev(instructions = "Which team should handle this support ticket?", task = "router.team")]
enum Team {
    /// Payments, invoices, refunds and subscription charges
    Billing,
    /// Bugs, crashes, error messages and outages
    Engineering,
    /// Pricing questions, new purchases and plan upgrades
    Sales,
    /// Account access, passwords and profile changes
    Accounts,
}

#[derive(Choice, Debug, Clone, Copy, PartialEq)]
#[cev(task = "router.severity")]
enum Severity {
    /// Cosmetic; no impact on use
    Low,
    /// Something is degraded but a workaround exists
    Medium,
    /// Blocking; the customer cannot proceed
    High,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = std::env::args().nth(1).unwrap_or_else(|| "Qwen/Qwen3-0.6B".into());
    let engine = Engine::load(EngineOptions { model, ..Default::default() })?;
    let runtime = cev_runtime::Cev::new(Arc::new(engine), Store::open("router.db")?, RuntimeConfig::default())?;
    let cev = Cev::local(runtime);

    let tickets = [
        json!({"from": "dana@example.com", "subject": "Charged twice", "body": "My card was billed two times for March. Please refund one."}),
        json!({"from": "li@example.com", "subject": "App won't open", "body": "Since today's update the app crashes on launch. I can't work."}),
        json!({"from": "sam@example.com", "subject": "Team plan", "body": "We have 40 people. What would the annual plan cost?"}),
        json!({"from": "ari@example.com", "subject": "Locked out", "body": "I forgot my password and the reset email never arrives."}),
    ];

    for t in &tickets {
        // Three typed questions, one forward pass over the ticket.
        let mut q = cev.ask(t);
        let team = q.choose::<Team>(Team::INSTRUCTIONS.unwrap());
        let severity = q.score::<Severity>("How severe is the customer's problem?");
        let refund = q.check("Is the customer asking for money back?");
        let a = q.send().await?;
        let (team, severity, refund) = (a.get(team)?, a.get(severity)?, a.get(refund)?);

        let queue = match *team {
            Team::Billing if *refund => "billing/refunds",
            Team::Billing => "billing",
            Team::Engineering if *severity == Severity::High => "engineering/pager",
            Team::Engineering => "engineering",
            Team::Sales => "sales",
            Team::Accounts => "accounts",
        };
        let low_confidence = team.confident(0.6).is_none();
        println!(
            "{:<16} -> {:<18} team={:?} ({:.2}) severity={:?} (score {:.2}) refund={} ({:.2}){} [{:.0} ms]",
            t["subject"].as_str().unwrap(),
            queue,
            *team,
            team.confidence,
            *severity,
            severity.score.unwrap(),
            *refund,
            refund.probability(),
            if low_confidence { "  <- needs a human" } else { "" },
            a.raw().latency_ms,
        );
    }

    // When an agent reroutes a ticket, the correction becomes a label (by decision id):
    let d = cev.pick::<Team>(&tickets[3]).await?;
    if *d != Team::Accounts {
        let fb = d.correct(Team::Accounts).await?;
        println!("\ncorrected {} -> accounts; task {} now has {} labels", d.decision_id, fb.task, fb.task_examples);
    } else {
        d.confirm().await?;
        println!("\nconfirmed {} (labels are kept for training)", d.decision_id);
    }
    Ok(())
}
