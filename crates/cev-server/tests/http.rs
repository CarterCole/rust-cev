use axum::body::Body;
use axum::http::{Request, StatusCode};
use cev_core::MockBackend;
use cev_runtime::{Cev, RuntimeConfig, Store};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

fn app(key: Option<&str>) -> axum::Router {
    let cev = Cev::new(Arc::new(MockBackend::default()), Store::in_memory().unwrap(), RuntimeConfig::default()).unwrap();
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
