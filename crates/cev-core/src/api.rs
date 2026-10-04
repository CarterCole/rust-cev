//! Wire types. The request/response shapes match TypeSafe's System One API
//! (`POST /v1/systemone`) so existing Jev SDKs work against cev unchanged.
//! cev-specific additions live in `request_id`, `latency_ms` and `x_cev`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemOneRequest {
    /// Free text or any JSON value (program state).
    pub state: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub questions: IndexMap<String, Question>,
    /// Skip persisting this request for feedback (cev extension).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_store: bool,
    /// Position-bias correction; defaults to the server setting (cev extension).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debias: Option<crate::compile::Debias>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    /// "Is this true?" -> probability in [0, 1].
    Noul,
    /// "Which of these options?" -> one option plus a distribution.
    Choice,
    /// "Which level?" -> expected level index plus a distribution.
    Score,
}

impl QuestionType {
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionType::Noul => "noul",
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub kind: QuestionType,
    pub instructions: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Value>,
    /// Stable name for the online-learning adapter (cev extension). When
    /// omitted, a key is derived from the type, instructions and option names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
    pub request_id: String,
    pub latency_ms: f64,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_cev: Option<AnswerMeta>,
    },
    Choice {
        choice: String,
        confidence: f64,
        probabilities: IndexMap<String, f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_cev: Option<AnswerMeta>,
    },
    Score {
        score: f64,
        confidence: f64,
        legend: IndexMap<String, String>,
        probabilities: IndexMap<String, f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x_cev: Option<AnswerMeta>,
    },
}

/// Per-answer metadata (cev extension).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnswerMeta {
    /// Id of this single decision; send it back with `/v1/feedback`.
    pub decision_id: String,
    pub task: String,
    /// True when an online adapter adjusted this answer.
    pub adapted: bool,
    /// Distribution before the adapter, over the same keys as `probabilities`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_probabilities: Option<Vec<f64>>,
}

/// A ground-truth label sent back for a past answer (or a new example).
///
/// * noul: `true`/`false`, `"yes"`/`"no"`, or a probability in [0, 1]
/// * choice: the option name, or `{"option": p, ...}`
/// * score: a level index (fractional values split between neighbours),
///   or `{"0": p, "1": p, ...}`
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Label {
    Bool(bool),
    Number(f64),
    Text(String),
    Distribution(IndexMap<String, f64>),
}

/// Feedback on one stored decision. Identify it by `decision_id`, or by
/// `request_id` + `question_id`. Feedback is always stored with the decision's
/// full context so it can be exported for training, whether or not online
/// learning is enabled.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeedbackRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question_id: Option<String>,
    /// The correct answer. Omit to only leave a comment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<Label>,
    /// Importance weight for this label (default 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    /// Arbitrary caller metadata (who labelled it, source system, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeedbackResponse {
    pub feedback_id: String,
    pub decision_id: String,
    pub task: String,
    /// Log loss of the answer that was served, on this label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_loss: Option<f64>,
    /// Whether the label updated an online adapter.
    pub learned: bool,
    pub task_examples: u64,
    pub adapter_active: bool,
}

/// Labelled examples: answer the questions, then learn from the labels.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExamplesRequest {
    pub examples: Vec<Example>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Example {
    pub state: Value,
    pub questions: IndexMap<String, Question>,
    pub labels: IndexMap<String, Label>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub backbone: String,
    pub hidden_size: usize,
    pub max_options: usize,
    pub prompt_version: String,
}
