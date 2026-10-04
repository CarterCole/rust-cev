//! cev in the browser. The same compile → Qwen3 → adapter pipeline as the
//! server, with JSON strings in and out (the REST request/response bodies).
//! The decision log lives in memory; weights are streamed in by the page.

use cev_core::{Backend, FeedbackRequest, MockBackend, SystemOneRequest};
use cev_model::weights::{DType, StreamLoader};
use cev_model::{Engine, EngineOptions};
use cev_runtime::{ExportFilter, RuntimeConfig, Store};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&format!("{e:#}"))
}

fn json(v: &impl serde::Serialize) -> Result<String, JsError> {
    serde_json::to_string(v).map_err(err)
}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

/// Receives `model.safetensors` chunk by chunk, converting to f32 as it goes.
#[wasm_bindgen]
pub struct WeightLoader(StreamLoader);

#[wasm_bindgen]
impl WeightLoader {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(StreamLoader::new(DType::F32))
    }

    pub fn push(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        self.0.push(chunk).map_err(err)
    }
}

#[wasm_bindgen]
pub struct Cev(cev_runtime::Cev);

fn runtime(backend: Arc<dyn Backend>) -> Result<Cev, JsError> {
    let store = Store::in_memory().map_err(err)?;
    Ok(Cev(cev_runtime::Cev::new(backend, store, RuntimeConfig::default()).map_err(err)?))
}

#[wasm_bindgen]
impl Cev {
    /// Keyword-overlap stub; no weights needed.
    pub fn mock() -> Result<Cev, JsError> {
        runtime(Arc::new(MockBackend::default()))
    }

    /// A Qwen3 checkpoint from its files: `model` is the repo id it came from.
    pub fn load(model: &str, config: &[u8], tokenizer: &[u8], tokenizer_config: &str, weights: WeightLoader) -> Result<Cev, JsError> {
        let opts = EngineOptions {
            model: model.into(),
            device: "cpu".into(),
            dtype: "f32".into(),
            max_context: 8192,
            // Cached prefixes cost ~0.25 MB per token on Qwen3-0.6B and wasm
            // memory tops out at 4 GB, most of it weights.
            prefix_cache_tokens: 1024,
            ..Default::default()
        };
        let engine = Engine::from_parts(config, tokenizer, tokenizer_config, weights.0.finish().map_err(err)?, opts).map_err(err)?;
        runtime(Arc::new(engine))
    }

    /// `GET /v1/models`
    pub fn model(&self) -> Result<String, JsError> {
        json(&self.0.model_info())
    }

    /// `POST /v1/systemone`
    pub fn decide(&self, request: &str) -> Result<String, JsError> {
        let req: SystemOneRequest = serde_json::from_str(request).map_err(err)?;
        json(&self.0.decide(&req).map_err(err)?)
    }

    /// `POST /v1/feedback`
    pub fn feedback(&self, request: &str) -> Result<String, JsError> {
        let req: FeedbackRequest = serde_json::from_str(request).map_err(err)?;
        json(&self.0.feedback(&req).map_err(err)?)
    }

    /// `GET /v1/tasks`
    pub fn tasks(&self) -> Result<String, JsError> {
        json(&self.0.tasks())
    }

    /// `DELETE /v1/tasks/{task}`
    pub fn reset_task(&self, task: &str) -> Result<bool, JsError> {
        self.0.reset_task(task).map_err(err)
    }

    /// `GET /v1/export`: NDJSON training rows.
    pub fn export(&self, labeled: bool) -> Result<String, JsError> {
        let mut out = String::new();
        let filter = ExportFilter { labeled: Some(labeled), ..Default::default() };
        self.0
            .export(&filter, |row| {
                out.push_str(&serde_json::to_string(&row)?);
                out.push('\n');
                Ok(())
            })
            .map_err(err)?;
        Ok(out)
    }

    /// `GET /v1/stats`
    pub fn stats(&self) -> Result<String, JsError> {
        json(&self.0.stats().map_err(err)?)
    }
}
