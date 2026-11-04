use cev_core::{Answer, FeedbackRequest, Label, MockBackend, SystemOneRequest};
use cev_runtime::{Cev, ExportFilter, RuntimeConfig, Store};
use serde_json::json;
use std::sync::Arc;

fn request(text: &str) -> SystemOneRequest {
    serde_json::from_value(json!({
        "state": text,
        "questions": {
            "team": {"type": "choice", "instructions": "Which team handles this?", "task": "router",
                     "criteria": {"billing": "invoices", "tech": "errors"}},
            "urgent": {"type": "noul", "instructions": "Is it urgent?"}
        }
    }))
    .unwrap()
}

fn decision_id(a: &Answer) -> String {
    match a {
        Answer::Noul { x_cev, .. } | Answer::Choice { x_cev, .. } | Answer::Score { x_cev, .. } => {
            x_cev.as_ref().unwrap().decision_id.clone()
        }
    }
}

fn label(id: &str, l: Label) -> FeedbackRequest {
    FeedbackRequest { decision_id: Some(id.into()), request_id: None, question_id: None, label: Some(l), weight: None, comment: None, metadata: None }
}

#[test]
fn feedback_is_stored_with_full_context_and_exported() {
    let cev = Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    let r = cev.decide(&request("the invoices are wrong")).unwrap();
    let Answer::Choice { choice, .. } = &r.answers["team"] else { panic!() };
    assert_eq!(choice, "billing");

    let id = decision_id(&r.answers["team"]);
    let fb = cev.feedback(&FeedbackRequest { comment: Some("actually a bug".into()), metadata: Some(json!({"by": "agent-7"})), ..label(&id, Label::Text("tech".into())) }).unwrap();
    assert!(fb.learned && fb.served_loss.unwrap() > 0.1);
    // Also addressable by request id + question id.
    cev.feedback(&FeedbackRequest { decision_id: None, request_id: Some(r.request_id.clone()), question_id: Some("urgent".into()), ..label("", Label::Bool(false)) }).unwrap();
    // Comment-only feedback must not hide the label.
    cev.feedback(&FeedbackRequest { label: None, comment: Some("second look".into()), ..label(&id, Label::Bool(true)) }).unwrap();

    let mut rows = Vec::new();
    cev.export(&ExportFilter::default(), |row| {
        rows.push(row);
        Ok(())
    }).unwrap();
    assert_eq!(rows.len(), 2);
    let t = rows.iter().find(|r| r.question_id == "team").unwrap();
    assert_eq!(t.target.as_deref(), Some(&[0.0, 1.0][..]));
    assert!(t.prompt.contains("the invoices are wrong") && t.prompt.contains("B. tech: errors"));
    assert_eq!(t.state, json!("the invoices are wrong"));
    assert_eq!(t.comment.as_deref(), Some("actually a bug"));

    let mut all = 0;
    cev.export(&ExportFilter { labeled: Some(false), ..Default::default() }, |_| {
        all += 1;
        Ok(())
    }).unwrap();
    assert_eq!(all, 2);
    assert!(cev.feedback(&label("dec_missing", Label::Bool(true))).is_err());
    assert!(cev.feedback(&label(&id, Label::Text("sales".into()))).is_err());
}

#[test]
fn adapter_takes_over_after_consistent_corrections_and_persists() {
    let dir = std::env::temp_dir().join(format!("cev-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("cev.db");
    let _ = std::fs::remove_file(&db);
    let backend = Arc::new(MockBackend::default());
    let texts = ["invoices broken", "invoices missing", "invoices late", "invoices doubled", "invoices wrong total"];
    {
        let cev = Cev::new(backend.clone(), Store::open(&db).unwrap(), RuntimeConfig::default()).unwrap();
        // The mock says "billing" for these; the business says they are "tech".
        for i in 0..30 {
            let r = cev.decide(&request(texts[i % texts.len()])).unwrap();
            cev.feedback(&label(&decision_id(&r.answers["team"]), Label::Text("tech".into()))).unwrap();
        }
        let t = cev.task("", "router").unwrap();
        assert!(t.active, "{t:?}");
        let r = cev.decide(&request("invoices broken")).unwrap();
        let Answer::Choice { choice, x_cev, .. } = &r.answers["team"] else { panic!() };
        assert_eq!(choice, "tech");
        assert!(x_cev.as_ref().unwrap().adapted);
    }
    // Restart: adapter and replay buffer come back from the store.
    let cev = Cev::new(backend, Store::open(&db).unwrap(), RuntimeConfig::default()).unwrap();
    let t = cev.task("", "router").unwrap();
    assert!(t.active && t.buffered == 30, "{t:?}");
    let r = cev.decide(&request("invoices late")).unwrap();
    let Answer::Choice { choice, .. } = &r.answers["team"] else { panic!() };
    assert_eq!(choice, "tech");
    assert!(cev.reset_task("", "router").unwrap());
    let _ = std::fs::remove_dir_all(dir);
}

fn in_namespace(text: &str, ns: &str) -> SystemOneRequest {
    SystemOneRequest { namespace: Some(ns.into()), ..request(text) }
}

#[test]
fn feedback_only_adapts_its_own_namespace() {
    let cev = Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    for i in 0..30 {
        let r = cev.decide(&in_namespace(&format!("invoices broken {i}"), "acme")).unwrap();
        let fb = cev.feedback(&label(&decision_id(&r.answers["team"]), Label::Text("tech".into()))).unwrap();
        assert_eq!(fb.namespace.as_deref(), Some("acme"));
    }
    assert!(cev.task("acme", "router").unwrap().active);
    assert!(cev.task("", "router").is_none() && cev.task("globex", "router").is_none());
    assert_eq!(cev.tasks(None).len(), 1);
    assert!(cev.tasks(Some("")).is_empty());

    let served = |ns: Option<&str>| {
        let req = SystemOneRequest { namespace: ns.map(String::from), ..request("invoices broken") };
        let r = cev.decide(&req).unwrap();
        let Answer::Choice { choice, x_cev, .. } = &r.answers["team"] else { panic!() };
        let meta = x_cev.clone().unwrap();
        (choice.clone(), meta.adapted, meta.namespace)
    };
    assert_eq!(served(Some("acme")), ("tech".into(), true, Some("acme".into())));
    assert_eq!(served(Some("globex")), ("billing".into(), false, Some("globex".into())));
    assert_eq!(served(None), ("billing".into(), false, None));

    let mut rows = 0;
    cev.export(&ExportFilter { namespace: Some("acme".into()), ..Default::default() }, |row| {
        assert_eq!(row.namespace, "acme");
        rows += 1;
        Ok(())
    }).unwrap();
    assert_eq!(rows, 30);
    assert!(!cev.reset_task("", "router").unwrap());
    assert!(cev.reset_task("acme", "router").unwrap());
}

#[test]
fn databases_from_before_namespaces_are_migrated() {
    let dir = std::env::temp_dir().join(format!("cev-migrate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("cev.db");
    let _ = std::fs::remove_file(&db);
    {
        let c = rusqlite::Connection::open(&db).unwrap();
        c.execute_batch(
            "CREATE TABLE decisions (decision_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, question_id TEXT NOT NULL,
                 model TEXT NOT NULL, task TEXT NOT NULL, kind TEXT NOT NULL, question TEXT NOT NULL, options TEXT NOT NULL,
                 codes TEXT NOT NULL, suffix TEXT NOT NULL, base_logits BLOB NOT NULL, served TEXT NOT NULL,
                 adapted INTEGER NOT NULL, hidden BLOB, created_at INTEGER NOT NULL, UNIQUE (request_id, question_id));
             CREATE TABLE adapters (model TEXT NOT NULL, task TEXT NOT NULL, data TEXT NOT NULL, updated_at INTEGER NOT NULL,
                 PRIMARY KEY (model, task));
             INSERT INTO adapters VALUES ('m', 'router', '{}', 1);",
        )
        .unwrap();
    }
    let backend = Arc::new(MockBackend::default());
    let cev = Cev::new(backend, Store::open(&db).unwrap(), RuntimeConfig::default()).unwrap();
    let r = cev.decide(&in_namespace("invoices broken", "acme")).unwrap();
    cev.feedback(&label(&decision_id(&r.answers["team"]), Label::Text("tech".into()))).unwrap();
    assert_eq!(cev.task("acme", "router").unwrap().examples, 1);
    let c = rusqlite::Connection::open(&db).unwrap();
    let old: String = c.query_row("SELECT namespace FROM adapters WHERE model = 'm'", [], |r| r.get(0)).unwrap();
    assert_eq!(old, "");
    drop(c);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn storage_can_be_skipped_and_learning_disabled() {
    let cfg = RuntimeConfig { online_learning: false, ..Default::default() };
    let cev = Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), cfg).unwrap();
    let mut req = request("x");
    req.no_store = true;
    let r = cev.decide(&req).unwrap();
    assert!(decision_id(&r.answers["team"]).is_empty());
    let r = cev.decide(&request("x")).unwrap();
    let fb = cev.feedback(&label(&decision_id(&r.answers["team"]), Label::Text("tech".into()))).unwrap();
    assert!(!fb.learned);
    assert_eq!(cev.stats().unwrap().labeled_decisions, 1);
}

#[test]
fn debias_rotations_map_back_to_canonical_options() {
    // The mock is position-invariant, so averaging rotations must reproduce
    // the undebiased distribution exactly; any index mix-up would change it.
    let cev = Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    let mut req: SystemOneRequest = serde_json::from_value(json!({
        "state": "refund for the invoice, the export crashes",
        "questions": {"t": {"type": "choice", "instructions": "?", "criteria": {
            "billing": "refund invoice", "tech": "export crashes", "sales": "pricing", "hr": "hiring", "ops": "export invoice refund"}}}
    }))
    .unwrap();
    let probs = |r: &cev_core::SystemOneResponse| match &r.answers["t"] {
        Answer::Choice { probabilities, .. } => probabilities.values().copied().collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    req.debias = Some(cev_core::Debias::None);
    let plain = probs(&cev.decide(&req).unwrap());
    req.debias = Some(cev_core::Debias::Permute);
    let permuted = probs(&cev.decide(&req).unwrap());
    for (a, b) in plain.iter().zip(&permuted) {
        assert!((a - b).abs() < 1e-5, "{plain:?} vs {permuted:?}");
    }
    assert!(plain[4] > plain[0] && plain[0] > plain[2]);
}

#[test]
fn media_is_decoded_and_checked_against_the_backend() {
    use base64::Engine as _;
    let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\nnot really a picture");
    let with_image = |text: &str| SystemOneRequest { images: vec![format!("data:image/png;base64,{png}")], ..request(text) };
    // A text-only backend refuses pictures with a client error.
    let cev = Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    let err = cev.decide(&with_image("invoices")).unwrap_err().to_string();
    assert!(err.contains("`images`") && err.contains("text only"), "{err}");
    // A backend that takes them gets the decoded bytes and reports its modalities.
    let cev = Cev::new(Arc::new(MockBackend::default().with_media()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    let info = cev.model_info();
    assert_eq!(info.modalities, ["text", "image", "audio"]);
    assert_eq!(info.family, "cev");
    let r = cev.decide(&with_image("the invoices are wrong")).unwrap();
    assert!(r.usage.input_tokens > 256, "image tokens counted: {}", r.usage.input_tokens);
    let bad = SystemOneRequest { images: vec!["%%% not base64".into()], ..request("x") };
    assert!(cev.decide(&bad).unwrap_err().to_string().contains("base64"));
    // Audio needs an audio-capable backend too.
    let cev = Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    let clip = SystemOneRequest { audio: Some(base64::engine::general_purpose::STANDARD.encode(b"RIFF....WAVE")), ..request("x") };
    assert!(cev.decide(&clip).unwrap_err().to_string().contains("`audio`"));
}
