//! REST API. `/v1/systemone` and `/v1/models` are wire-compatible with
//! TypeSafe's System One API; the rest is cev's feedback/learning surface.

use crate::AppState;
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use cev_core::{ExamplesRequest, FeedbackRequest, SystemOneRequest};
use cev_runtime::{CevError, ExportFilter};
use serde::Deserialize;
use serde_json::{Value, json};

pub struct ApiError(pub CevError);

impl From<CevError> for ApiError {
    fn from(e: CevError) -> Self {
        Self(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self.0 {
            CevError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
            CevError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            CevError::Internal(e) => {
                tracing::error!("{e:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
            }
        };
        (status, Json(json!({"error": {"type": kind, "message": self.0.to_string()}}))).into_response()
    }
}

type ApiResult<T> = Result<Json<T>, ApiError>;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route("/v1/feedback", post(feedback))
        .route("/v1/examples", post(examples))
        .route("/v1/decisions", get(decisions))
        .route("/v1/decisions/{id}", get(decision))
        .route("/v1/tasks", get(tasks))
        .route("/v1/tasks/{task}", get(task).delete(reset_task))
        .route("/v1/export", get(export))
        .route("/v1/stats", get(stats))
}

async fn models(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"object": "list", "data": [s.cev.model_info()]}))
}

async fn systemone(State(s): State<AppState>, Json(req): Json<SystemOneRequest>) -> ApiResult<cev_core::SystemOneResponse> {
    Ok(Json(s.decide(req).await?))
}

async fn feedback(State(s): State<AppState>, Json(req): Json<FeedbackRequest>) -> ApiResult<cev_core::FeedbackResponse> {
    Ok(Json(s.blocking(move |c| c.feedback(&req)).await?))
}

async fn examples(State(s): State<AppState>, Json(req): Json<ExamplesRequest>) -> ApiResult<Value> {
    let _permit = s.permits.acquire().await.expect("semaphore closed");
    let out = s.blocking(move |c| c.examples(&req)).await?;
    Ok(Json(json!({
        "results": out.into_iter().map(|(resp, fb)| json!({"response": resp, "feedback": fb})).collect::<Vec<_>>()
    })))
}

#[derive(Deserialize)]
struct Page {
    task: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

async fn decisions(State(s): State<AppState>, Query(p): Query<Page>) -> ApiResult<Value> {
    let rows = s
        .blocking(move |c| Ok(c.store().recent_decisions(p.task.as_deref(), p.limit.unwrap_or(50).min(1000), p.offset.unwrap_or(0))?))
        .await?;
    Ok(Json(json!({"data": rows})))
}

async fn decision(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Value> {
    let found = s
        .blocking(move |c| {
            let Some((d, fb)) = c.decision(&id)? else { return Ok(None) };
            let req = c.store().request(&d.request_id)?;
            Ok(Some((d, fb, req)))
        })
        .await?;
    let (d, fb, req) = found.ok_or_else(|| CevError::NotFound("decision not found".into()))?;
    let prompt = req.as_ref().map(|r| format!("{}{}", r.prefix, d.suffix));
    Ok(Json(json!({
        "decision": d,
        "state": req.map(|r| r.state),
        "prompt": prompt,
        "feedback": fb,
    })))
}

async fn tasks(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"data": s.cev.tasks()}))
}

async fn task(State(s): State<AppState>, Path(task): Path<String>) -> ApiResult<cev_runtime::TaskInfo> {
    Ok(Json(s.cev.task(&task).ok_or_else(|| CevError::NotFound(format!("no adapter for task `{task}`")))?))
}

async fn reset_task(State(s): State<AppState>, Path(task): Path<String>) -> ApiResult<Value> {
    let deleted = s.blocking(move |c| c.reset_task(&task)).await?;
    Ok(Json(json!({"deleted": deleted})))
}

/// Newline-delimited JSON training rows (see `cev_runtime::ExportRow`).
async fn export(State(s): State<AppState>, Query(f): Query<ExportFilter>) -> Result<Response, ApiError> {
    let body = s
        .blocking(move |c| {
            let mut buf = Vec::new();
            c.export(&f, |row| {
                serde_json::to_writer(&mut buf, &row)?;
                buf.push(b'\n');
                Ok(())
            })?;
            Ok(buf)
        })
        .await?;
    Ok(([(header::CONTENT_TYPE, "application/x-ndjson")], Body::from(body)).into_response())
}

async fn stats(State(s): State<AppState>) -> ApiResult<Value> {
    let st = s.blocking(|c| c.stats()).await?;
    Ok(Json(json!({"store": st, "tasks": s.cev.tasks().len(), "model": s.cev.model_info()})))
}
