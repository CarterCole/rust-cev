//! HTTP front end for cev: REST (TypeSafe-compatible + feedback) and GraphQL.

pub mod graphql;
pub mod rest;

use async_graphql::http::GraphiQLSource;
use async_graphql_axum::GraphQL;
use axum::{
    Router,
    extract::{DefaultBodyLimit, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use cev_runtime::{Cev, CevError, CevResult};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Largest request body: up to 64 MB of media (`cev_core::compile`'s
/// limit), base64-encoded, plus the JSON around it.
pub const MAX_BODY_BYTES: usize = 96 << 20;

#[derive(Clone)]
pub struct AppState {
    pub cev: Arc<Cev>,
    /// Bounds concurrent forward passes (1 is right for a single GPU).
    pub permits: Arc<Semaphore>,
    pub api_key: Option<Arc<str>>,
    /// Host names (no port) a request's `Host` header may carry; `None`
    /// accepts any. Pinning these defeats DNS rebinding, where a web page
    /// re-points its own domain at 127.0.0.1 to read this API.
    pub allowed_hosts: Option<Arc<[String]>>,
}

/// The names a loopback-bound server answers to by default.
pub const LOOPBACK_HOSTS: [&str; 3] = ["localhost", "127.0.0.1", "[::1]"];

impl AppState {
    pub fn new(cev: Cev, concurrency: usize, api_key: Option<String>) -> Self {
        Self {
            cev: Arc::new(cev),
            permits: Arc::new(Semaphore::new(concurrency.max(1))),
            api_key: api_key.map(Into::into),
            allowed_hosts: None,
        }
    }

    /// Only answer requests whose `Host` is one of `hosts` (case-insensitive,
    /// port ignored).
    pub fn with_allowed_hosts(mut self, hosts: impl IntoIterator<Item = String>) -> Self {
        self.allowed_hosts = Some(hosts.into_iter().map(|h| h.to_ascii_lowercase()).collect());
        self
    }

    /// Run blocking runtime work off the async executor.
    pub async fn blocking<T: Send + 'static>(&self, f: impl FnOnce(&Cev) -> CevResult<T> + Send + 'static) -> CevResult<T> {
        let cev = self.cev.clone();
        tokio::task::spawn_blocking(move || f(&cev))
            .await
            .map_err(|e| CevError::Internal(anyhow::anyhow!("worker panicked: {e}")))?
    }

    pub async fn decide(&self, req: cev_core::SystemOneRequest) -> CevResult<cev_core::SystemOneResponse> {
        let _permit = self.permits.acquire().await.expect("semaphore closed");
        self.blocking(move |c| c.decide(&req)).await
    }
}

/// `Host` without its port: `localhost:8080` → `localhost`, `[::1]:80` → `[::1]`.
fn host_name(host: &str) -> &str {
    match host.rfind(':') {
        Some(i) if !host[i..].contains(']') => &host[..i],
        _ => host,
    }
}

/// Requests a browser lets any web page send without a CORS preflight
/// ("simple" requests: `text/plain`, form posts) must not reach GraphQL,
/// which would otherwise parse them as JSON and run their mutations. A
/// cross-origin page cannot send `application/json` without a preflight,
/// and this server answers no preflight.
fn graphql_post_is_json(req: &Request) -> bool {
    req.headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|m| m.trim().eq_ignore_ascii_case("application/json"))
}

/// Host pinning (all paths), JSON-only GraphQL posts, and the bearer token:
/// the API (`/v1/...`, `/graphql`) needs it when one is set; `/health` and
/// the static page do not.
async fn auth(State(s): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(allowed) = &s.allowed_hosts {
        // HTTP/1.1 clients always send Host and browsers cannot omit it;
        // HTTP/2 carries it as the URI authority.
        let host = req.headers().get(header::HOST).and_then(|v| v.to_str().ok()).or_else(|| req.uri().host());
        if let Some(host) = host {
            let name = host_name(host).to_ascii_lowercase();
            if !allowed.contains(&name) {
                return (StatusCode::FORBIDDEN, format!("host `{host}` is not allowed (see --allowed-host)")).into_response();
            }
        }
    }
    let path = req.uri().path();
    if req.method() == axum::http::Method::POST && path.starts_with("/graphql") && !graphql_post_is_json(&req) {
        return (StatusCode::UNSUPPORTED_MEDIA_TYPE, "GraphQL POSTs must be `Content-Type: application/json`").into_response();
    }
    let protected = path.starts_with("/v1/") || path.starts_with("/graphql");
    if let Some(key) = s.api_key.as_ref().filter(|_| protected) {
        let ok = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|t| t == &**key);
        if !ok {
            return (StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response();
        }
    }
    next.run(req).await
}

pub fn app(state: AppState) -> Router {
    app_serving(state, None)
}

/// The API, plus the files of `web` (the browser page in `web/`) at `/`
/// when given: the page then runs its decisions on this server.
pub fn app_serving(state: AppState, web: Option<&Path>) -> Router {
    let schema = graphql::schema(state.clone());
    let mut router = Router::new()
        .merge(rest::router())
        .route(
            "/graphql",
            get(|| async { Html(GraphiQLSource::build().endpoint("/graphql").finish()) }).post_service(GraphQL::new(schema)),
        );
    if let Some(dir) = web {
        router = router.fallback_service(tower_http::services::ServeDir::new(dir));
    }
    router
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        // DefaultBodyLimit only binds axum's own extractors (`Json`,
        // `Bytes`); the GraphQL service reads the raw body, so cap the
        // stream itself too.
        .layer(tower_http::limit::RequestBodyLimitLayer::new(MAX_BODY_BYTES))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state)
}
