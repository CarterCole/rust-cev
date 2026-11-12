use axum::body::Body;
use axum::http::{Request, StatusCode};
use cev_core::MockBackend;
use cev_runtime::{Cev, RuntimeConfig, Store};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

fn app(key: Option<&str>) -> axum::Router {
    app_with(MockBackend::default(), key)
}

fn app_with(backend: MockBackend, key: Option<&str>) -> axum::Router {
    let cev = Cev::new(Arc::new(backend), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    cev_server::app(cev_server::AppState::new(cev, 1, key.map(String::from)))
}

async fn call(app: &axum::Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value, String) {
    let mut req = Request::builder().method(method).uri(uri).header("content-type", "application/json");
    if uri.contains("secret") {
        req = req.header("authorization", "Bearer secret");
    }
    let req = req.body(body.map(|b| Body::from(b.to_string())).unwrap_or_default()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).to_string();
    (status, serde_json::from_str(&text).unwrap_or(Value::Null), text)
}

#[tokio::test]
async fn rest_decide_feedback_export() {
    let app = app(None);
    let (s, r, _) = call(&app, "POST", "/v1/systemone", Some(json!({
        "state": {"ticket": "My invoice total is wrong"},
        "model": "jev-latest",
        "questions": {
            "team": {"type": "choice", "instructions": "Route", "criteria": {"billing": "invoice problems", "tech": "crashes"}},
            "sev": {"type": "score", "instructions": "Severity", "criteria": ["low", "high"]},
            "refund": {"type": "noul", "instructions": "Refund requested?"}
        }
    }))).await;
    assert_eq!(s, StatusCode::OK, "{r}");
    assert_eq!(r["answers"]["team"]["choice"], "billing");
    assert_eq!(r["answers"]["sev"]["legend"]["1"], "high");
    assert!(r["answers"]["refund"]["noul"].is_number());
    assert_eq!(r["usage"]["output_tokens"], 0);
    let id = r["answers"]["team"]["x_cev"]["decision_id"].as_str().unwrap().to_string();

    let (s, fb, _) = call(&app, "POST", "/v1/feedback", Some(json!({"decision_id": id, "label": "tech", "comment": "crash in invoice screen"}))).await;
    assert_eq!(s, StatusCode::OK, "{fb}");
    assert_eq!(fb["learned"], true);

    let (s, d, _) = call(&app, "GET", &format!("/v1/decisions/{id}"), None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(d["state"]["ticket"], "My invoice total is wrong");
    assert!(d["prompt"].as_str().unwrap().contains("Question: Route"));
    assert_eq!(d["feedback"][0]["comment"], "crash in invoice screen");

    let (s, _, text) = call(&app, "GET", "/v1/export", None).await;
    assert_eq!(s, StatusCode::OK);
    let rows: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["target"], json!([0.0, 1.0]));
    assert_eq!(rows[0]["codes"], json!(["A", "B"]));

    let (s, e, _) = call(&app, "POST", "/v1/feedback", Some(json!({"decision_id": "nope", "label": true}))).await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{e}");
    let (s, _, _) = call(&app, "POST", "/v1/systemone", Some(json!({"state": "x", "questions": {}}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let (_, t, _) = call(&app, "GET", "/v1/tasks", None).await;
    assert_eq!(t["data"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn graphql_decide_and_feedback() {
    let app = app(None);
    let q = r#"mutation($s: JSON!) { decide(state: $s, questions: [
        {id: "team", type: CHOICE, instructions: "Route", criteria: {billing: "invoice problems", tech: "crashes"}, task: "router"},
        {id: "urgent", type: NOUL, instructions: "Urgent?"}
      ]) { requestId answers { questionId type decisionId answer confidence probabilities { option probability } } } }"#;
    let (s, r, _) = call(&app, "POST", "/graphql", Some(json!({"query": q, "variables": {"s": {"text": "invoice wrong"}}}))).await;
    assert_eq!(s, StatusCode::OK);
    assert!(r["errors"].is_null(), "{r}");
    let a = &r["data"]["decide"]["answers"][0];
    assert_eq!(a["answer"], "billing");
    let id = a["decisionId"].as_str().unwrap();
    let m = format!(r#"mutation {{ feedback(decisionId: "{id}", label: "tech", metadata: {{by: "qa"}}) {{ learned task taskExamples }} }}"#);
    let (_, r, _) = call(&app, "POST", "/graphql", Some(json!({"query": m}))).await;
    assert!(r["errors"].is_null(), "{r}");
    assert_eq!(r["data"]["feedback"]["task"], "router");
    let (_, r, _) = call(&app, "POST", "/graphql", Some(json!({"query": format!(r#"{{ tasks {{ task examples }} decision(id: "{id}") {{ state prompt feedback }} stats {{ decisions feedback }} }}"#)}))).await;
    assert!(r["errors"].is_null(), "{r}");
    assert_eq!(r["data"]["tasks"][0]["examples"], 1);
    assert_eq!(r["data"]["decision"]["state"]["text"], "invoice wrong");
    assert_eq!(r["data"]["stats"]["decisions"], 2);
}

#[tokio::test]
async fn bearer_auth() {
    let app = app(Some("secret"));
    let (s, _, _) = call(&app, "GET", "/v1/models", None).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let (s, _, _) = call(&app, "GET", "/v1/models?k=secret", None).await;
    assert_eq!(s, StatusCode::OK);
    let (s, _, _) = call(&app, "GET", "/health", None).await;
    assert_eq!(s, StatusCode::OK);
}

#[tokio::test]
async fn images_and_audio_on_the_wire() {
    use base64::Engine as _;
    let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\nstub");
    let body = json!({
        "state": "a receipt",
        "images": [format!("data:image/png;base64,{png}"), png],
        "questions": {"total": {"type": "noul", "instructions": "Is the total above 100?"}}
    });
    // Text-only backbone: a 400 that names the field.
    let (s, _, text) = call(&app(None), "POST", "/v1/systemone", Some(body.clone())).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(text.contains("images"), "{text}");
    // Image-capable backbone: answered, and the model advertises its modalities.
    let app = app_with(MockBackend::default().with_media(), None);
    let (s, r, _) = call(&app, "POST", "/v1/systemone", Some(body)).await;
    assert_eq!(s, StatusCode::OK, "{r}");
    assert!(r["answers"]["total"]["noul"].is_number());
    let (s, m, _) = call(&app, "GET", "/v1/models", None).await;
    assert_eq!(s, StatusCode::OK);
    let model = if m["data"].is_array() { m["data"][0].clone() } else { m.clone() };
    assert_eq!(model["modalities"], json!(["text", "image", "audio"]));
    assert_eq!(model["family"], "cev");
}

#[tokio::test]
async fn serves_the_page_and_takes_big_media() {
    let dir = std::env::temp_dir().join(format!("cev-web-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("index.html"), "<!doctype html><title>cev</title>").unwrap();
    let cev = Cev::new(Arc::new(MockBackend::default().with_media()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    let app = cev_server::app_serving(cev_server::AppState::new(cev, 1, Some("secret".into())), Some(&dir));
    // The page needs no token; the API still does.
    let (s, _, text) = call(&app, "GET", "/", None).await;
    assert_eq!(s, StatusCode::OK);
    assert!(text.contains("<title>cev</title>"));
    let (s, _, _) = call(&app, "GET", "/v1/models", None).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    // A 5 MB image (a camera frame, a long audio window) is past axum's 2 MB default.
    use base64::Engine as _;
    let big = base64::engine::general_purpose::STANDARD.encode(vec![7u8; 5 << 20]);
    let body = json!({"state": "a frame", "images": [big], "questions": {"q": {"type": "noul", "instructions": "ok?"}}});
    let (s, r, _) = call(&app, "POST", "/v1/systemone?key=secret", Some(body)).await;
    assert_eq!(s, StatusCode::OK, "{r}");
    std::fs::remove_dir_all(&dir).ok();
}

async fn raw(app: &axum::Router, req: Request<Body>) -> StatusCode {
    app.clone().oneshot(req).await.unwrap().status()
}

#[tokio::test]
async fn graphql_refuses_cross_site_simple_posts() {
    let app = app(None);
    let body = json!({"query": "mutation { resetTask(name: \"t\") }"}).to_string();
    // What a web page can send without a CORS preflight.
    for ct in ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"] {
        let req = Request::post("/graphql").header("content-type", ct).body(Body::from(body.clone())).unwrap();
        assert_eq!(raw(&app, req).await, StatusCode::UNSUPPORTED_MEDIA_TYPE, "{ct}");
    }
    let req = Request::post("/graphql").body(Body::from(body.clone())).unwrap();
    assert_eq!(raw(&app, req).await, StatusCode::UNSUPPORTED_MEDIA_TYPE, "no content type");
    let req = Request::post("/graphql").header("content-type", "application/json; charset=utf-8").body(Body::from(body)).unwrap();
    assert_eq!(raw(&app, req).await, StatusCode::OK);
}

#[tokio::test]
async fn host_pinning_blocks_dns_rebinding() {
    let cev = Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
    let state = cev_server::AppState::new(cev, 1, None).with_allowed_hosts(cev_server::LOOPBACK_HOSTS.map(String::from));
    let app = cev_server::app(state);
    for (host, want) in [
        ("localhost:8080", StatusCode::OK),
        ("LOCALHOST", StatusCode::OK),
        ("127.0.0.1:8080", StatusCode::OK),
        ("[::1]:8080", StatusCode::OK),
        ("[::1]", StatusCode::OK),
        ("rebind.example.com:8080", StatusCode::FORBIDDEN),
        ("127.0.0.1.example.com", StatusCode::FORBIDDEN),
    ] {
        let req = Request::get("/v1/decisions").header("host", host).body(Body::empty()).unwrap();
        assert_eq!(raw(&app, req).await, want, "{host}");
    }
}

#[tokio::test]
async fn graphql_body_is_capped() {
    let app = app(None);
    let pad = "a".repeat(cev_server::MAX_BODY_BYTES);
    let body = format!(r#"{{"query":"{{ model {{ id }} }}","variables":{{"pad":"{pad}"}}}}"#);
    // Declared up front: refused before the body is read.
    let req = Request::post("/graphql").header("content-type", "application/json").header("content-length", body.len()).body(Body::from(body.clone())).unwrap();
    assert_eq!(raw(&app, req).await, StatusCode::PAYLOAD_TOO_LARGE);
    // Streamed without a length: the read stops at the cap and fails.
    let req = Request::post("/graphql").header("content-type", "application/json").body(Body::from(body)).unwrap();
    assert!(raw(&app, req).await.is_client_error());
}
