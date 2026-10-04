//! Typed System One decisions for Rust programs.
//!
//! Put a model where an `if` or a `match` goes. The answer comes back as a
//! real value of your type, so the compiler checks that every case is handled
//! and the model can never return something outside the enum.
//!
//! ```no_run
//! # async fn demo() -> Result<(), cev::Error> {
//! #[derive(cev::Choice, Debug, Clone, Copy, PartialEq)]
//! #[cev(instructions = "Which team should handle this ticket?")]
//! enum Team {
//!     /// Payments, invoices and refunds
//!     Billing,
//!     /// Bugs, crashes and error messages
//!     Tech,
//!     /// New purchases and upgrades
//!     Sales,
//! }
//!
//! let cev = cev::Cev::http("http://127.0.0.1:8080");
//! let ticket = serde_json::json!({"subject": "Charged twice", "body": "..."});
//!
//! if *cev.check(&ticket, "Is the customer asking for a refund?").await? {
//!     // ...
//! }
//!
//! let team = cev.pick::<Team>(&ticket).await?;
//! match *team {
//!     Team::Billing => {}
//!     Team::Tech => {}
//!     Team::Sales => {}
//! }
//!
//! // Later, when you learn the truth, send it back (typed):
//! team.correct(Team::Tech).await?;
//! # Ok(()) }
//! ```
//!
//! Several questions about the same state share one forward pass:
//!
//! ```no_run
//! # #[derive(cev::Choice, Clone, Copy)] enum Team { A, B }
//! # #[derive(cev::Choice, Clone, Copy)] enum Severity { Low, High }
//! # async fn demo(cev: cev::Cev, ticket: serde_json::Value) -> Result<(), cev::Error> {
//! let mut q = cev.ask(&ticket);
//! let team = q.choose::<Team>("Which team?");
//! let sev = q.score::<Severity>("How severe?");
//! let refund = q.check("Refund requested?");
//! let a = q.send().await?;
//! let (team, sev, refund) = (a.get(team)?, a.get(sev)?, a.get(refund)?);
//! # Ok(()) }
//! ```

pub use cev_core::{Debias, FeedbackResponse, Label, SystemOneRequest, SystemOneResponse};
pub use cev_derive::Choice;

use cev_core::{Answer, FeedbackRequest, Question, QuestionType};
use indexmap::IndexMap;
use serde::Serialize;
use serde_json::{Value, json};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

/// One option of a [`Choice`] enum.
#[derive(Debug, Clone, Copy)]
pub struct OptionDef {
    pub name: &'static str,
    pub description: Option<&'static str>,
}

/// A closed set of outcomes the model can pick from. Derive it on a fieldless
/// enum with `#[derive(cev::Choice)]`.
pub trait Choice: Sized + 'static {
    const OPTIONS: &'static [OptionDef];
    /// Default question text, from `#[cev(instructions = "...")]`.
    const INSTRUCTIONS: Option<&'static str> = None;
    /// Adapter/task key, from `#[cev(task = "...")]`.
    const TASK: Option<&'static str> = None;
    fn from_index(i: usize) -> Option<Self>;
    fn index(&self) -> usize;

    fn option_name(&self) -> &'static str {
        Self::OPTIONS[self.index()].name
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::OPTIONS.iter().position(|o| o.name == name).and_then(Self::from_index)
    }

    /// `criteria` for a choice question.
    fn choice_criteria() -> Value {
        Value::Object(
            Self::OPTIONS
                .iter()
                .map(|o| (o.name.to_string(), o.description.map_or(Value::Null, |d| Value::String(d.into()))))
                .collect(),
        )
    }

    /// `criteria` for a score question: variants are levels, lowest first.
    fn score_criteria() -> Value {
        Value::Array(Self::OPTIONS.iter().map(|o| Value::String(o.description.unwrap_or(o.name).into())).collect())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cev server returned {status}: {message}")]
    Api { status: u16, message: String },
    #[error("transport: {0}")]
    Transport(String),
    #[error("unexpected answer: {0}")]
    Protocol(String),
    #[error("{0}: this decision was not stored (no_store), so it cannot take feedback")]
    NotStored(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone)]
enum Transport {
    #[cfg(feature = "http")]
    Http { base: String, key: Option<String>, client: reqwest::Client },
    #[cfg(feature = "local")]
    Local(Arc<cev_runtime::Cev>),
}

/// Client handle. Cheap to clone.
#[derive(Clone)]
pub struct Cev {
    transport: Arc<Transport>,
    no_store: bool,
    debias: Option<Debias>,
}

impl Cev {
    /// Talk to a `cev` server.
    #[cfg(feature = "http")]
    pub fn http(base_url: impl Into<String>) -> Self {
        let base = base_url.into().trim_end_matches('/').to_string();
        Self { transport: Arc::new(Transport::Http { base, key: None, client: reqwest::Client::new() }), no_store: false, debias: None }
    }

    #[cfg(feature = "http")]
    pub fn with_api_key(self, key: impl Into<String>) -> Self {
        match &*self.transport {
            Transport::Http { base, client, .. } => Self {
                transport: Arc::new(Transport::Http { base: base.clone(), key: Some(key.into()), client: client.clone() }),
                no_store: self.no_store,
                debias: self.debias,
            },
            #[allow(unreachable_patterns)]
            _ => self,
        }
    }

    /// Run decisions in-process (no server), e.g. with a `cev_model::Engine`.
    #[cfg(feature = "local")]
    pub fn local(runtime: cev_runtime::Cev) -> Self {
        Self::from_runtime(Arc::new(runtime))
    }

    #[cfg(feature = "local")]
    pub fn from_runtime(runtime: Arc<cev_runtime::Cev>) -> Self {
        Self { transport: Arc::new(Transport::Local(runtime)), no_store: false, debias: None }
    }

    /// Don't log decisions made through this handle (they can't take feedback).
    pub fn without_storage(mut self) -> Self {
        self.no_store = true;
        self
    }

    /// Override the server's position-bias correction for this handle.
    pub fn with_debias(mut self, debias: Debias) -> Self {
        self.debias = Some(debias);
        self
    }

    /// Send a raw System One request.
    pub async fn system_one(&self, req: SystemOneRequest) -> Result<SystemOneResponse> {
        match &*self.transport {
            #[cfg(feature = "http")]
            Transport::Http { .. } => self.post("/v1/systemone", &req).await,
            #[cfg(feature = "local")]
            Transport::Local(rt) => {
                let rt = rt.clone();
                tokio::task::spawn_blocking(move || rt.decide(&req))
                    .await
                    .map_err(|e| Error::Transport(e.to_string()))?
                    .map_err(local_err)
            }
        }
    }

    /// Send feedback for a decision by id.
    pub async fn feedback(&self, decision_id: &str, label: Label, comment: Option<String>) -> Result<FeedbackResponse> {
        let req = FeedbackRequest {
            decision_id: Some(decision_id.to_string()),
            request_id: None,
            question_id: None,
            label: Some(label),
            weight: None,
            comment,
            metadata: None,
        };
        match &*self.transport {
            #[cfg(feature = "http")]
            Transport::Http { .. } => self.post("/v1/feedback", &req).await,
            #[cfg(feature = "local")]
            Transport::Local(rt) => {
                let rt = rt.clone();
                tokio::task::spawn_blocking(move || rt.feedback(&req))
                    .await
                    .map_err(|e| Error::Transport(e.to_string()))?
                    .map_err(local_err)
            }
        }
    }

    #[cfg(feature = "http")]
    #[cfg_attr(not(feature = "local"), allow(irrefutable_let_patterns))]
    async fn post<B: Serialize, T: serde::de::DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let Transport::Http { base, key, client } = &*self.transport else { unreachable!() };
        let mut rb = client.post(format!("{base}{path}")).json(body);
        if let Some(k) = key {
            rb = rb.bearer_auth(k);
        }
        let resp = rb.send().await.map_err(|e| Error::Transport(e.to_string()))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| Error::Transport(e.to_string()))?;
        if !status.is_success() {
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(String::from))
                .unwrap_or(text);
            return Err(Error::Api { status: status.as_u16(), message });
        }
        serde_json::from_str(&text).map_err(|e| Error::Protocol(e.to_string()))
    }

    /// Start a multi-question request about `state`.
    pub fn ask(&self, state: &impl Serialize) -> Ask {
        Ask { cev: self.clone(), state: serde_json::to_value(state).unwrap_or(Value::Null), questions: IndexMap::new() }
    }

    /// Yes/no as a `bool` (probability ≥ 0.5). The probability is on `.probability()`.
    pub async fn check(&self, state: &impl Serialize, instructions: &str) -> Result<Decided<bool>> {
        let mut q = self.ask(state);
        let h = q.check(instructions);
        q.send().await?.get(h)
    }

    /// Pick one variant of `T`, using `T`'s `#[cev(instructions)]`.
    pub async fn pick<T: Choice>(&self, state: &impl Serialize) -> Result<Decided<T>> {
        let instructions = T::INSTRUCTIONS.unwrap_or("Which option applies?");
        self.choose::<T>(state, instructions).await
    }

    /// Pick one variant of `T` for the given question.
    pub async fn choose<T: Choice>(&self, state: &impl Serialize, instructions: &str) -> Result<Decided<T>> {
        let mut q = self.ask(state);
        let h = q.choose::<T>(instructions);
        q.send().await?.get(h)
    }

    /// Place `state` on the ordered levels of `T` (first variant = lowest).
    /// `.score` holds the expected level; the value is the most likely level.
    pub async fn score<T: Choice>(&self, state: &impl Serialize, instructions: &str) -> Result<Decided<T>> {
        let mut q = self.ask(state);
        let h = q.score::<T>(instructions);
        q.send().await?.get(h)
    }
}

#[cfg(feature = "local")]
fn local_err(e: cev_runtime::CevError) -> Error {
    let status = match e {
        cev_runtime::CevError::BadRequest(_) => 400,
        cev_runtime::CevError::NotFound(_) => 404,
        cev_runtime::CevError::Internal(_) => 500,
    };
    Error::Api { status, message: e.to_string() }
}

/// Typed handle to one question of an [`Ask`].
pub struct Handle<T> {
    id: String,
    _t: PhantomData<fn() -> T>,
}

/// Builder for several questions about one state (one forward pass).
pub struct Ask {
    cev: Cev,
    state: Value,
    questions: IndexMap<String, Question>,
}

impl Ask {
    fn push<T>(&mut self, kind: QuestionType, instructions: &str, criteria: Option<Value>, task: Option<&str>) -> Handle<T> {
        let id = format!("q{}", self.questions.len());
        self.questions.insert(
            id.clone(),
            Question { kind, instructions: instructions.to_string(), criteria, task: task.map(String::from) },
        );
        Handle { id, _t: PhantomData }
    }

    pub fn check(&mut self, instructions: &str) -> Handle<bool> {
        self.push(QuestionType::Noul, instructions, None, None)
    }

    /// Yes/no with descriptions of what each side means.
    pub fn check_with(&mut self, instructions: &str, yes: &str, no: &str) -> Handle<bool> {
        self.push(QuestionType::Noul, instructions, Some(json!({"true": yes, "false": no})), None)
    }

    pub fn choose<T: Choice>(&mut self, instructions: &str) -> Handle<T> {
        self.push(QuestionType::Choice, instructions, Some(T::choice_criteria()), T::TASK)
    }

    pub fn score<T: Choice>(&mut self, instructions: &str) -> Handle<Scored<T>> {
        self.push(QuestionType::Score, instructions, Some(T::score_criteria()), T::TASK)
    }

    pub async fn send(self) -> Result<Answers> {
        let req = SystemOneRequest { state: self.state, model: None, questions: self.questions, no_store: self.cev.no_store, debias: self.cev.debias };
        let resp = self.cev.system_one(req).await?;
        Ok(Answers { cev: self.cev, resp })
    }
}

/// Marker for score handles; resolves to `Decided<T>` with `score` set.
pub struct Scored<T>(PhantomData<T>);

pub struct Answers {
    cev: Cev,
    resp: SystemOneResponse,
}

/// Converts an answer into a typed [`Decided`] value.
pub trait FromAnswer: Sized {
    type Out;
    fn from_answer(a: &Answer, cev: &Cev, request_id: &str) -> Result<Decided<Self::Out>>;
}

fn meta(x: &Option<cev_core::AnswerMeta>) -> (String, bool) {
    x.as_ref().map(|m| (m.decision_id.clone(), m.adapted)).unwrap_or_default()
}

impl FromAnswer for bool {
    type Out = bool;
    fn from_answer(a: &Answer, cev: &Cev, request_id: &str) -> Result<Decided<bool>> {
        let Answer::Noul { noul, x_cev } = a else { return Err(Error::Protocol("expected a noul answer".into())) };
        let (decision_id, adapted) = meta(x_cev);
        Ok(Decided {
            value: *noul >= 0.5,
            confidence: (2.0 * noul - 1.0).abs(),
            probabilities: IndexMap::from([("false".into(), 1.0 - noul), ("true".into(), *noul)]),
            score: None,
            decision_id,
            request_id: request_id.into(),
            adapted,
            cev: cev.clone(),
        })
    }
}

impl<T: Choice> FromAnswer for T {
    type Out = T;
    fn from_answer(a: &Answer, cev: &Cev, request_id: &str) -> Result<Decided<T>> {
        let Answer::Choice { choice, confidence, probabilities, x_cev } = a else {
            return Err(Error::Protocol("expected a choice answer".into()));
        };
        let value = T::from_name(choice).ok_or_else(|| Error::Protocol(format!("unknown option `{choice}`")))?;
        let (decision_id, adapted) = meta(x_cev);
        Ok(Decided {
            value,
            confidence: *confidence,
            probabilities: probabilities.clone(),
            score: None,
            decision_id,
            request_id: request_id.into(),
            adapted,
            cev: cev.clone(),
        })
    }
}

impl<T: Choice> FromAnswer for Scored<T> {
    type Out = T;
    fn from_answer(a: &Answer, cev: &Cev, request_id: &str) -> Result<Decided<T>> {
        let Answer::Score { score, confidence, probabilities, x_cev, .. } = a else {
            return Err(Error::Protocol("expected a score answer".into()));
        };
        // Most likely level (ties -> lower).
        let best = probabilities
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |b, (i, (_, &p))| if p > b.1 { (i, p) } else { b })
            .0;
        let value = T::from_index(best).ok_or_else(|| Error::Protocol("level out of range".into()))?;
        let (decision_id, adapted) = meta(x_cev);
        let probabilities = probabilities
            .iter()
            .enumerate()
            .map(|(i, (_, &p))| (T::OPTIONS.get(i).map_or_else(|| i.to_string(), |o| o.name.to_string()), p))
            .collect();
        Ok(Decided {
            value,
            confidence: *confidence,
            probabilities,
            score: Some(*score),
            decision_id,
            request_id: request_id.into(),
            adapted,
            cev: cev.clone(),
        })
    }
}

impl Answers {
    pub fn get<T: FromAnswer>(&self, h: Handle<T>) -> Result<Decided<T::Out>> {
        let a = self.resp.answers.get(&h.id).ok_or_else(|| Error::Protocol(format!("missing answer `{}`", h.id)))?;
        T::from_answer(a, &self.cev, &self.resp.request_id)
    }

    pub fn raw(&self) -> &SystemOneResponse {
        &self.resp
    }
}

/// A typed decision. Derefs to the value, so `match *d { ... }` and
/// `if *d { ... }` work directly.
#[derive(Clone)]
pub struct Decided<T> {
    pub value: T,
    /// 0 = uniform, 1 = certain (TypeSafe's definition).
    pub confidence: f64,
    pub probabilities: IndexMap<String, f64>,
    /// Expected level, for score questions.
    pub score: Option<f64>,
    pub decision_id: String,
    pub request_id: String,
    /// True when an online adapter shaped this answer.
    pub adapted: bool,
    cev: Cev,
}

impl<T> Deref for Decided<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Decided<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decided")
            .field("value", &self.value)
            .field("confidence", &self.confidence)
            .field("probabilities", &self.probabilities)
            .field("score", &self.score)
            .field("decision_id", &self.decision_id)
            .finish()
    }
}

impl<T> Decided<T> {
    pub fn into_inner(self) -> T {
        self.value
    }

    /// The value only when confidence clears `threshold`; route the rest to a
    /// human or a slower path.
    pub fn confident(&self, threshold: f64) -> Option<&T> {
        (self.confidence >= threshold).then_some(&self.value)
    }

    fn id(&self) -> Result<&str> {
        if self.decision_id.is_empty() {
            return Err(Error::NotStored("feedback"));
        }
        Ok(&self.decision_id)
    }

    /// Attach a free-text note to this decision without labelling it.
    pub async fn comment(&self, text: &str) -> Result<()> {
        let req = FeedbackRequest {
            decision_id: Some(self.id()?.to_string()),
            request_id: None,
            question_id: None,
            label: None,
            weight: None,
            comment: Some(text.to_string()),
            metadata: None,
        };
        match &*self.cev.transport {
            #[cfg(feature = "http")]
            Transport::Http { .. } => self.cev.post::<_, FeedbackResponse>("/v1/feedback", &req).await.map(|_| ()),
            #[cfg(feature = "local")]
            Transport::Local(rt) => rt.feedback(&req).map(|_| ()).map_err(local_err),
        }
    }
}

impl Decided<bool> {
    pub fn probability(&self) -> f64 {
        self.probabilities["true"]
    }

    /// Report the true answer.
    pub async fn correct(&self, actual: bool) -> Result<FeedbackResponse> {
        self.cev.feedback(self.id()?, Label::Bool(actual), None).await
    }

    /// The answer was right.
    pub async fn confirm(&self) -> Result<FeedbackResponse> {
        self.correct(self.value).await
    }
}

impl<T: Choice> Decided<T> {
    /// Report the true answer. Works for both choice and score decisions.
    pub async fn correct(&self, actual: T) -> Result<FeedbackResponse> {
        let label = match self.score {
            Some(_) => Label::Number(actual.index() as f64),
            None => Label::Text(actual.option_name().to_string()),
        };
        self.cev.feedback(self.id()?, label, None).await
    }

    /// The answer was right.
    pub async fn confirm(&self) -> Result<FeedbackResponse> {
        let label = match self.score {
            Some(_) => Label::Number(self.value.index() as f64),
            None => Label::Text(self.value.option_name().to_string()),
        };
        self.cev.feedback(self.id()?, label, None).await
    }
}
