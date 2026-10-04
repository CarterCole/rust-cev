use cev::{Cev, Choice};
use cev_core::MockBackend;
use cev_runtime::{RuntimeConfig, Store};
use serde_json::json;
use std::sync::Arc;

#[derive(Choice, Debug, Clone, Copy, PartialEq)]
#[cev(instructions = "Which team should handle this ticket?", task = "router")]
enum Team {
    /// Payments, invoices and refunds
    Billing,
    /// Crashes and error messages
    TechSupport,
    #[cev(rename = "sales", description = "New purchases and upgrades")]
    Sales,
}

#[derive(Choice, Debug, Clone, Copy, PartialEq)]
enum Severity {
    /// Cosmetic
    Low,
    /// Workaround exists
    Medium,
    /// Blocking crashes
    High,
}

fn runtime() -> cev_runtime::Cev {
    cev_runtime::Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap()
}

#[test]
fn derive_metadata() {
    assert_eq!(Team::OPTIONS.iter().map(|o| o.name).collect::<Vec<_>>(), ["billing", "tech_support", "sales"]);
    assert_eq!(Team::OPTIONS[0].description, Some("Payments, invoices and refunds"));
    assert_eq!(Team::OPTIONS[2].description, Some("New purchases and upgrades"));
    assert_eq!(Team::TASK, Some("router"));
    assert_eq!(Team::from_name("tech_support"), Some(Team::TechSupport));
    assert_eq!(Severity::High.index(), 2);
    assert_eq!(Severity::score_criteria(), json!(["Cosmetic", "Workaround exists", "Blocking crashes"]));
}

async fn exercise(cev: Cev) {
    let ticket = json!({"body": "the invoices page shows error messages"});
    let team = cev.pick::<Team>(&json!({"body": "refunds for invoices"})).await.unwrap();
    let routed = match *team {
        Team::Billing => "billing",
        Team::TechSupport => "tech",
        Team::Sales => "sales",
    };
    assert_eq!(routed, "billing");
    assert!(!team.decision_id.is_empty());
    let fb = team.correct(Team::TechSupport).await.unwrap();
    assert_eq!(fb.task, "router");
    assert!(fb.learned);

    let mut q = cev.ask(&ticket);
    let t = q.choose::<Team>("Which team?");
    let s = q.score::<Severity>("How severe?");
    let r = q.check("Is it about invoices?");
    let a = q.send().await.unwrap();
    let (t, s, r) = (a.get(t).unwrap(), a.get(s).unwrap(), a.get(r).unwrap());
    assert_eq!(a.raw().answers.len(), 3);
    assert!(s.score.is_some());
    assert!((0.0..=1.0).contains(&r.probability()));
    let _: bool = *r;
    s.correct(Severity::High).await.unwrap();
    t.confirm().await.unwrap();
    t.comment("looked fine").await.unwrap();

    let unstored = cev.clone().without_storage().pick::<Team>(&ticket).await.unwrap();
    assert!(unstored.correct(Team::Sales).await.is_err());
}

#[tokio::test]
async fn local_transport() {
    exercise(Cev::local(runtime())).await;
}

#[tokio::test]
async fn http_transport() {
    let app = cev_server::app(cev_server::AppState::new(runtime(), 1, Some("k".into())));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let base = format!("http://{addr}");
    let err = Cev::http(&base).pick::<Team>(&"x").await.unwrap_err();
    assert!(matches!(err, cev::Error::Api { status: 401, .. }), "{err}");
    exercise(Cev::http(&base).with_api_key("k")).await;
}
