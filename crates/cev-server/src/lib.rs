//! HTTP front end for cev: REST (TypeSafe-compatible + feedback) and GraphQL.

pub mod graphql;
pub mod rest;

use async_graphql::http::GraphiQLSource;
use async_graphql_axum::GraphQL;
use axum::{
    Router,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use cev_runtime::{Cev, CevError, CevResult};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Clone)]
pub struct AppState {
    pub cev: Arc<Cev>,
    /// Bounds concurrent forward passes (1 is right for a single GPU).
    pub permits: Arc<Semaphore>,
    pub api_key: Option<Arc<str>>,
}

impl AppState {
    pub fn new(cev: Cev, concurrency: usize, api_key: Option<String>) -> Self {
        Self { cev: Arc::new(cev), permits: Arc::new(Semaphore::new(concurrency.max(1))), api_key: api_key.map(Into::into) }
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

async fn auth(State(s): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(key) = &s.api_key {
        let ok = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|t| t == &**key);
        if !ok && req.uri().path() != "/health" {
            return (StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response();
        }
    }
    next.run(req).await
}

pub fn app(state: AppState) -> Router {
    let schema = graphql::schema(state.clone());
    Router::new()
        .merge(rest::router())
        .route(
            "/graphql",
            get(|| async { Html(GraphiQLSource::build().endpoint("/graphql").finish()) }).post_service(GraphQL::new(schema)),
        )
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .layer(tower_http::trace::TraceLayer::new_for_http())
        .with_state(state)
}
