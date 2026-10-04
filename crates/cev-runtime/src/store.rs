//! Durable log of every decision and every piece of feedback (SQLite).
//!
//! Each decision is stored with everything needed to train on it later: the
//! original state and question, the exact prompt text the model saw, the
//! answer codes, the base logits, what was served, and (optionally) the hidden
//! features. Feedback rows reference a decision id. `export` joins them into
//! training rows, so labels are useful even when online learning is off.
//!
//! In the browser (`wasm32`) the same API is backed by memory instead.

use cev_core::{Label, OptionSpec, Question, QuestionType};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(not(target_arch = "wasm32"))]
mod sqlite;
#[cfg(not(target_arch = "wasm32"))]
pub use sqlite::Store;

#[cfg(target_arch = "wasm32")]
mod memory;
#[cfg(target_arch = "wasm32")]
pub use memory::Store;

pub fn now_ms() -> i64 {
    web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestRecord {
    pub request_id: String,
    pub model: String,
    pub state: Value,
    pub prefix: String,
    pub input_tokens: usize,
    pub latency_ms: f64,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub decision_id: String,
    pub request_id: String,
    pub question_id: String,
    pub model: String,
    pub task: String,
    pub kind: QuestionType,
    pub question: Question,
    pub options: Vec<OptionSpec>,
    pub codes: Vec<String>,
    /// Prompt text after the shared prefix; full prompt = request prefix + suffix.
    pub suffix: String,
    pub base_logits: Vec<f32>,
    /// Distribution that was returned to the caller.
    pub served: Vec<f64>,
    pub adapted: bool,
    #[serde(skip)]
    pub hidden: Option<Vec<f32>>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRecord {
    pub feedback_id: String,
    pub decision_id: String,
    pub label: Option<Label>,
    pub target: Option<Vec<f64>>,
    pub weight: f64,
    pub comment: Option<String>,
    pub metadata: Option<Value>,
    pub created_at: i64,
}

/// One training row: exactly what the model saw plus what it should say.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRow {
    pub decision_id: String,
    pub request_id: String,
    pub question_id: String,
    pub model: String,
    pub task: String,
    #[serde(rename = "type")]
    pub kind: QuestionType,
    /// Full prompt text; the answer is the next token.
    pub prompt: String,
    pub codes: Vec<String>,
    pub options: Vec<String>,
    /// Target distribution over `options` (absent for unlabelled rows).
    pub target: Option<Vec<f64>>,
    pub label: Option<Label>,
    pub weight: f64,
    pub served: Vec<f64>,
    pub base_probabilities: Vec<f64>,
    pub adapted: bool,
    pub comment: Option<String>,
    pub metadata: Option<Value>,
    pub state: Value,
    pub question: Question,
    pub feedback_id: Option<String>,
    pub decided_at: i64,
    pub labeled_at: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ExportFilter {
    /// Only this task.
    pub task: Option<String>,
    /// Only decisions made by this model id.
    pub model: Option<String>,
    /// `true` (default): only decisions with a label; `false`: every decision
    /// (e.g. to label them offline with a teacher model).
    pub labeled: Option<bool>,
    /// Only rows decided/labelled at or after this unix-ms time.
    pub since: Option<i64>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoreStats {
    pub requests: i64,
    pub decisions: i64,
    pub feedback: i64,
    pub labeled_decisions: i64,
}
