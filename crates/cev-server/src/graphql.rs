//! GraphQL API over the same runtime. Dynamic parts (state, criteria, labels,
//! metadata) are `JSON` scalars; everything else is typed.

use crate::AppState;
use async_graphql::{Context, EmptySubscription, Enum, InputObject, Json, Object, Result, Schema, SimpleObject};
use cev_core::{Answer, FeedbackRequest, Label, Question, QuestionType, SystemOneRequest};
use indexmap::IndexMap;
use serde_json::Value;

pub type CevSchema = Schema<QueryRoot, MutationRoot, EmptySubscription>;

pub fn schema(state: AppState) -> CevSchema {
    Schema::build(QueryRoot, MutationRoot, EmptySubscription).data(state).finish()
}

fn state<'a>(ctx: &Context<'a>) -> &'a AppState {
    ctx.data_unchecked::<AppState>()
}

#[derive(Enum, Copy, Clone, Eq, PartialEq)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

impl From<Kind> for QuestionType {
    fn from(k: Kind) -> Self {
        match k {
            Kind::Noul => QuestionType::Noul,
            Kind::Choice => QuestionType::Choice,
            Kind::Score => QuestionType::Score,
        }
    }
}

impl From<QuestionType> for Kind {
    fn from(k: QuestionType) -> Self {
        match k {
            QuestionType::Noul => Kind::Noul,
            QuestionType::Choice => Kind::Choice,
            QuestionType::Score => Kind::Score,
        }
    }
}

#[derive(Enum, Copy, Clone, Eq, PartialEq)]
pub enum DebiasMode {
    None,
    Calibrate,
    Permute,
    Full,
}

impl From<DebiasMode> for cev_core::Debias {
    fn from(d: DebiasMode) -> Self {
        match d {
            DebiasMode::None => cev_core::Debias::None,
            DebiasMode::Calibrate => cev_core::Debias::Calibrate,
            DebiasMode::Permute => cev_core::Debias::Permute,
            DebiasMode::Full => cev_core::Debias::Full,
        }
    }
}

#[derive(InputObject)]
pub struct QuestionInput {
    pub id: String,
    #[graphql(name = "type")]
    pub kind: Kind,
    pub instructions: String,
    /// noul: `{true, false}`; choice: `{name: description}`; score: `[levels]`.
    pub criteria: Option<Json<Value>>,
    pub task: Option<String>,
}

#[derive(SimpleObject)]
pub struct OptionProbability {
    pub option: String,
    pub probability: f64,
}

#[derive(SimpleObject)]
pub struct DecisionAnswer {
    pub question_id: String,
    #[graphql(name = "type")]
    pub kind: Kind,
    pub decision_id: String,
    pub task: String,
    pub adapted: bool,
    /// choice: the chosen option; noul: "true"/"false"; score: nearest level.
    pub answer: String,
    pub noul: Option<f64>,
    pub score: Option<f64>,
    pub confidence: Option<f64>,
    pub probabilities: Vec<OptionProbability>,
}

#[derive(SimpleObject)]
pub struct DecideResult {
    pub request_id: String,
    pub model: String,
    pub latency_ms: f64,
    pub input_tokens: i64,
    pub answers: Vec<DecisionAnswer>,
}

#[derive(SimpleObject)]
pub struct FeedbackResult {
    pub feedback_id: String,
    pub decision_id: String,
    pub task: String,
    pub served_loss: Option<f64>,
    pub learned: bool,
    pub task_examples: u64,
    pub adapter_active: bool,
}

#[derive(SimpleObject)]
pub struct Task {
    pub task: String,
    pub examples: u64,
    pub buffered: u64,
    pub active: bool,
    pub temperature: f32,
    pub base_loss: f64,
    pub adapted_loss: f64,
}

impl From<cev_runtime::TaskInfo> for Task {
    fn from(t: cev_runtime::TaskInfo) -> Self {
        Self {
            task: t.task,
            examples: t.examples,
            buffered: t.buffered as u64,
            active: t.active,
            temperature: t.temperature,
            base_loss: t.base_loss,
            adapted_loss: t.adapted_loss,
        }
    }
}

#[derive(SimpleObject)]
pub struct StoredDecision {
    pub decision_id: String,
    pub request_id: String,
    pub question_id: String,
    pub model: String,
    pub task: String,
    #[graphql(name = "type")]
    pub kind: Kind,
    pub question: Json<Value>,
    pub options: Vec<String>,
    pub served: Vec<f64>,
    pub adapted: bool,
    pub created_at: i64,
    /// Program state the decision was made on.
    pub state: Option<Json<Value>>,
    /// Exact prompt text the model saw.
    pub prompt: Option<String>,
    pub feedback: Vec<Json<Value>>,
}

#[derive(SimpleObject)]
pub struct Model {
    pub id: String,
    pub backbone: String,
    pub hidden_size: u64,
    pub max_options: u64,
    pub prompt_version: String,
}

#[derive(SimpleObject)]
pub struct Stats {
    pub requests: i64,
    pub decisions: i64,
    pub feedback: i64,
    pub labeled_decisions: i64,
}

fn to_answer(qid: &str, a: Answer) -> DecisionAnswer {
    let probs = |m: &IndexMap<String, f64>| m.iter().map(|(k, v)| OptionProbability { option: k.clone(), probability: *v }).collect();
    let (kind, answer, noul, score, confidence, probabilities, meta) = match a {
        Answer::Noul { noul, x_cev } => (
            Kind::Noul,
            (noul >= 0.5).to_string(),
            Some(noul),
            None,
            None,
            vec![
                OptionProbability { option: "false".into(), probability: 1.0 - noul },
                OptionProbability { option: "true".into(), probability: noul },
            ],
            x_cev,
        ),
        Answer::Choice { choice, confidence, probabilities, x_cev } => {
            (Kind::Choice, choice, None, None, Some(confidence), probs(&probabilities), x_cev)
        }
        Answer::Score { score, confidence, probabilities, x_cev, .. } => (
            Kind::Score,
            (score.round() as i64).to_string(),
            None,
            Some(score),
            Some(confidence),
            probs(&probabilities),
            x_cev,
        ),
    };
    let meta = meta.unwrap_or(cev_core::AnswerMeta { decision_id: String::new(), task: String::new(), adapted: false, base_probabilities: None });
    DecisionAnswer {
        question_id: qid.to_string(),
        kind,
        decision_id: meta.decision_id,
        task: meta.task,
        adapted: meta.adapted,
        answer,
        noul,
        score,
        confidence,
        probabilities,
    }
}

fn gql_err(e: cev_runtime::CevError) -> async_graphql::Error {
    async_graphql::Error::new(e.to_string())
}

pub struct QueryRoot;

#[Object]
impl QueryRoot {
    async fn model(&self, ctx: &Context<'_>) -> Model {
        let m = state(ctx).cev.model_info();
        Model { id: m.id, backbone: m.backbone, hidden_size: m.hidden_size as u64, max_options: m.max_options as u64, prompt_version: m.prompt_version }
    }

    async fn stats(&self, ctx: &Context<'_>) -> Result<Stats> {
        let s = state(ctx).blocking(|c| c.stats()).await.map_err(gql_err)?;
        Ok(Stats { requests: s.requests, decisions: s.decisions, feedback: s.feedback, labeled_decisions: s.labeled_decisions })
    }

    async fn tasks(&self, ctx: &Context<'_>) -> Vec<Task> {
        state(ctx).cev.tasks().into_iter().map(Task::from).collect()
    }

    async fn task(&self, ctx: &Context<'_>, name: String) -> Option<Task> {
        state(ctx).cev.task(&name).map(Task::from)
    }

    async fn decision(&self, ctx: &Context<'_>, id: String) -> Result<Option<StoredDecision>> {
        let s = state(ctx);
        s.blocking(move |c| {
            let Some((d, fb)) = c.decision(&id)? else { return Ok(None) };
            let req = c.store().request(&d.request_id)?;
            Ok(Some(stored(d, fb, req)))
        })
        .await
        .map_err(gql_err)
    }

    async fn decisions(&self, ctx: &Context<'_>, task: Option<String>, #[graphql(default = 50)] limit: usize, #[graphql(default = 0)] offset: usize) -> Result<Vec<StoredDecision>> {
        state(ctx)
            .blocking(move |c| {
                let rows = c.store().recent_decisions(task.as_deref(), limit.min(1000), offset)?;
                rows.into_iter()
                    .map(|d| {
                        let fb = c.store().feedback_for(&d.decision_id)?;
                        Ok(stored(d, fb, None))
                    })
                    .collect::<Result<Vec<_>, cev_runtime::CevError>>()
            })
            .await
            .map_err(gql_err)
    }
}

fn stored(d: cev_runtime::store::DecisionRecord, fb: Vec<cev_runtime::store::FeedbackRecord>, req: Option<cev_runtime::store::RequestRecord>) -> StoredDecision {
    StoredDecision {
        prompt: req.as_ref().map(|r| format!("{}{}", r.prefix, d.suffix)),
        state: req.map(|r| Json(r.state)),
        feedback: fb.into_iter().map(|f| Json(serde_json::to_value(f).unwrap_or_default())).collect(),
        question: Json(serde_json::to_value(&d.question).unwrap_or_default()),
        options: d.options.into_iter().map(|o| o.name).collect(),
        decision_id: d.decision_id,
        request_id: d.request_id,
        question_id: d.question_id,
        model: d.model,
        task: d.task,
        kind: d.kind.into(),
        served: d.served,
        adapted: d.adapted,
        created_at: d.created_at,
    }
}

pub struct MutationRoot;

#[Object]
impl MutationRoot {
    /// Answer typed questions about `state` in one parallel pass.
    async fn decide(
        &self,
        ctx: &Context<'_>,
        state_: Json<Value>,
        questions: Vec<QuestionInput>,
        #[graphql(default = false)] no_store: bool,
        debias: Option<DebiasMode>,
    ) -> Result<DecideResult> {
        let mut qs = IndexMap::new();
        for q in questions {
            let id = q.id.clone();
            if qs.insert(id.clone(), Question { kind: q.kind.into(), instructions: q.instructions, criteria: q.criteria.map(|c| c.0), task: q.task }).is_some() {
                return Err(format!("duplicate question id `{id}`").into());
            }
        }
        let resp = state(ctx)
            .decide(SystemOneRequest { state: state_.0, model: None, questions: qs, no_store, debias: debias.map(Into::into) })
            .await
            .map_err(gql_err)?;
        Ok(DecideResult {
            request_id: resp.request_id,
            model: resp.model,
            latency_ms: resp.latency_ms,
            input_tokens: resp.usage.input_tokens as i64,
            answers: resp.answers.into_iter().map(|(k, a)| to_answer(&k, a)).collect(),
        })
    }

    /// Label a stored decision. `label`: option name, true/false, level index,
    /// or `{option: probability}`.
    #[allow(clippy::too_many_arguments)]
    async fn feedback(
        &self,
        ctx: &Context<'_>,
        decision_id: Option<String>,
        request_id: Option<String>,
        question_id: Option<String>,
        label: Option<Json<Value>>,
        weight: Option<f64>,
        comment: Option<String>,
        metadata: Option<Json<Value>>,
    ) -> Result<FeedbackResult> {
        let label: Option<Label> = label.map(|l| serde_json::from_value(l.0)).transpose()?;
        let req = FeedbackRequest { decision_id, request_id, question_id, label, weight, comment, metadata: metadata.map(|m| m.0) };
        let r = state(ctx).blocking(move |c| c.feedback(&req)).await.map_err(gql_err)?;
        Ok(FeedbackResult {
            feedback_id: r.feedback_id,
            decision_id: r.decision_id,
            task: r.task,
            served_loss: r.served_loss,
            learned: r.learned,
            task_examples: r.task_examples,
            adapter_active: r.adapter_active,
        })
    }

    /// Forget a task's online adapter (labels are kept).
    async fn reset_task(&self, ctx: &Context<'_>, name: String) -> Result<bool> {
        state(ctx).blocking(move |c| c.reset_task(&name)).await.map_err(gql_err)
    }
}
