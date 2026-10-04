//! The decision service: compile → backbone → adapters → typed answers,
//! with every decision logged and every label learned from.
//!
//! This is what `cev-server` exposes over HTTP/GraphQL and what the `cev`
//! SDK embeds in-process.

use crate::adapter::{Adapter, LearnConfig, Normalizer, Sample};
use crate::store::{DecisionRecord, ExportFilter, ExportRow, FeedbackRecord, RequestRecord, Store, StoreStats, now_ms};
use cev_core::{
    Answer, AnswerMeta, Backend, BackendOutput, CompiledRequest, Debias, ExamplesRequest, Readout, FeedbackRequest, FeedbackResponse, ModelInfo, SystemOneRequest,
    SystemOneResponse, Usage, compile, math, target::target,
};
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use web_time::Instant;

#[derive(Debug, thiserror::Error)]
pub enum CevError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    NotFound(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

pub type CevResult<T> = Result<T, CevError>;

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Learn from labels as they arrive. When false, labels are only stored.
    pub online_learning: bool,
    /// Persist hidden features with each decision (needed for online learning
    /// on feedback; costs 4 bytes x hidden size per decision).
    pub store_features: bool,
    pub learn: LearnConfig,
    /// Default position-bias correction (requests may override).
    pub debias: Debias,
    /// Most option rotations evaluated per question when permuting.
    pub max_permutations: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { online_learning: true, store_features: true, learn: LearnConfig::default(), debias: Debias::Full, max_permutations: 4 }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskInfo {
    pub task: String,
    pub examples: u64,
    pub buffered: usize,
    pub active: bool,
    pub temperature: f32,
    /// Decayed prequential log loss on labels, before learning from them.
    pub base_loss: f64,
    pub adapted_loss: f64,
}

pub struct Cev {
    backend: Arc<dyn Backend>,
    store: Store,
    cfg: RuntimeConfig,
    adapters: Mutex<HashMap<String, Adapter>>,
    normalizer: Mutex<Normalizer>,
    /// Content-free calibration logits by question-variant prompt. They do not
    /// depend on the state, so repeated question sets pay for them once.
    calibration: Mutex<LruMap<Vec<f32>>>,
}

/// Small LRU keyed by prompt text (eviction scans; entries are few and tiny).
struct LruMap<V> {
    cap: usize,
    map: HashMap<String, (u64, V)>,
    tick: u64,
}

impl<V: Clone> LruMap<V> {
    fn new(cap: usize) -> Self {
        Self { cap, map: HashMap::new(), tick: 0 }
    }

    fn get(&mut self, k: &str) -> Option<V> {
        self.tick += 1;
        let t = self.tick;
        self.map.get_mut(k).map(|e| {
            e.0 = t;
            e.1.clone()
        })
    }

    fn put(&mut self, k: String, v: V) {
        self.tick += 1;
        if self.map.len() >= self.cap
            && !self.map.contains_key(&k)
            && let Some(old) = self.map.iter().min_by_key(|(_, e)| e.0).map(|(k, _)| k.clone())
        {
            self.map.remove(&old);
        }
        self.map.insert(k, (self.tick, v));
    }
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::now_v7().simple())
}

impl Cev {
    pub fn new(backend: Arc<dyn Backend>, store: Store, cfg: RuntimeConfig) -> anyhow::Result<Self> {
        let model = backend.id().to_string();
        let dim = backend.hidden_size();
        let normalizer = store.load_normalizer(&model)?.filter(|n| n.mean.len() == dim).unwrap_or_else(|| Normalizer::new(dim));
        let mut adapters = HashMap::new();
        for (task, mut a) in store.load_adapters(&model)? {
            if a.dim == dim {
                a.set_buffer(store.replay(&model, &task, cfg.learn.buffer)?);
                adapters.insert(task, a);
            }
        }
        tracing::info!(model, adapters = adapters.len(), "runtime ready");
        Ok(Self { backend, store, cfg, adapters: Mutex::new(adapters), normalizer: Mutex::new(normalizer), calibration: Mutex::new(LruMap::new(8192)) })
    }

    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.backend.id().into(),
            object: "model",
            backbone: self.backend.backbone().into(),
            hidden_size: self.backend.hidden_size(),
            max_options: self.backend.codes().len(),
            prompt_version: cev_core::compile::PROMPT_VERSION.into(),
        }
    }

    /// Answer every question of a request. Blocking (runs the network).
    pub fn decide(&self, req: &SystemOneRequest) -> CevResult<SystemOneResponse> {
        let started = Instant::now();
        let compiled = compile(req, self.backend.codes(), self.backend.format()).map_err(|e| CevError::BadRequest(e.to_string()))?;
        let out = self.run_debiased(&compiled, req.debias.unwrap_or(self.cfg.debias))?;
        let model = self.backend.id().to_string();
        let request_id = new_id("req");
        let now = now_ms();

        let mut answers = indexmap::IndexMap::new();
        let mut records = Vec::with_capacity(compiled.questions.len());
        {
            let adapters = self.adapters.lock();
            let mut norm = self.normalizer.lock();
            for (q, r) in compiled.questions.iter().zip(&out.readouts) {
                norm.observe(&r.hidden);
                let names: Vec<String> = q.options.iter().map(|o| o.name.clone()).collect();
                let base = math::softmax(&r.logits);
                let adapter = adapters.get(&q.task).filter(|a| self.cfg.online_learning && a.is_active(&self.cfg.learn));
                let served = match adapter {
                    Some(a) => a.predict(&names, &r.logits, &norm.features(&r.hidden)),
                    None => base.clone(),
                };
                let decision_id = new_id("dec");
                let meta = AnswerMeta {
                    decision_id: decision_id.clone(),
                    task: q.task.clone(),
                    adapted: adapter.is_some(),
                    base_probabilities: adapter.map(|_| base.iter().map(|p| (p * 1e6).round() / 1e6).collect()),
                };
                answers.insert(q.id.clone(), math::answer(q, &served, Some(meta)));
                records.push(DecisionRecord {
                    decision_id,
                    request_id: request_id.clone(),
                    question_id: q.id.clone(),
                    model: model.clone(),
                    task: q.task.clone(),
                    kind: q.kind,
                    question: req.questions[&q.id].clone(),
                    options: q.options.clone(),
                    codes: q.codes.clone(),
                    suffix: q.suffix.clone(),
                    base_logits: r.logits.clone(),
                    served,
                    adapted: adapter.is_some(),
                    hidden: self.cfg.store_features.then(|| r.hidden.clone()),
                    created_at: now,
                });
            }
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1e3;
        if !req.no_store {
            self.store.insert_request(
                &RequestRecord {
                    request_id: request_id.clone(),
                    model: model.clone(),
                    state: req.state.clone(),
                    prefix: compiled.prefix.clone(),
                    input_tokens: out.input_tokens,
                    latency_ms,
                    created_at: now,
                },
                &records,
            )?;
        } else {
            for a in answers.values_mut() {
                strip_decision_id(a);
            }
        }
        Ok(SystemOneResponse {
            model,
            answers,
            usage: Usage { input_tokens: out.input_tokens, output_tokens: 0 },
            request_id,
            latency_ms: (latency_ms * 100.0).round() / 100.0,
        })
    }

    /// Run the backbone with position-bias correction. Returns one readout per
    /// question whose `logits` are debiased log-scores in canonical option
    /// order and whose `hidden` comes from the unrotated prompt.
    fn run_debiased(&self, compiled: &CompiledRequest, debias: Debias) -> anyhow::Result<BackendOutput> {
        if debias == Debias::None {
            return self.backend.run(compiled);
        }
        let fmt = self.backend.format();
        let mut variants = Vec::new();
        let mut spans = Vec::with_capacity(compiled.questions.len());
        for q in &compiled.questions {
            let start = variants.len();
            for shift in debias.shifts(q.options.len(), self.cfg.max_permutations) {
                variants.push(if shift == 0 { q.clone() } else { q.rotated(shift, fmt) });
            }
            spans.push(start..variants.len());
        }
        let main = CompiledRequest { prefix: compiled.prefix.clone(), questions: variants };
        // Calibration twins: same question variants over content-free evidence.
        // Cached ones are reused; the rest run fused with the request.
        let mut cf: Vec<Option<Vec<f32>>> = vec![None; main.questions.len()];
        let mut missing = Vec::new();
        if debias.calibrates() {
            let mut cache = self.calibration.lock();
            for (i, q) in main.questions.iter().enumerate() {
                match cache.get(&q.suffix) {
                    Some(z) => cf[i] = Some(z),
                    None => missing.push(i),
                }
            }
        }
        let out = if missing.is_empty() {
            self.backend.run(&main)?
        } else {
            let twin = CompiledRequest {
                prefix: cev_core::compile::content_free_prefix(fmt),
                questions: missing.iter().map(|&i| main.questions[i].clone()).collect(),
            };
            let mut outs = self.backend.run_many(&[main.clone(), twin])?;
            let twin_out = outs.pop().expect("twin output");
            let mut cache = self.calibration.lock();
            for (&i, r) in missing.iter().zip(twin_out.readouts) {
                cache.put(main.questions[i].suffix.clone(), r.logits.clone());
                cf[i] = Some(r.logits);
            }
            outs.pop().expect("main output")
        };
        let readouts = spans
            .into_iter()
            .map(|span| {
                let k = main.questions[span.start].options.len();
                let n = span.len() as f64;
                let mut score = vec![0f64; k];
                for v in span.clone() {
                    let lp = math::log_softmax(&out.readouts[v].logits);
                    let base = cf[v].as_ref().map(|z| math::log_softmax(z));
                    for (pos, &opt) in main.questions[v].order.iter().enumerate() {
                        score[opt] += (lp[pos] - base.as_ref().map_or(0.0, |b| b[pos])) / n;
                    }
                }
                Readout { logits: score.into_iter().map(|v| v as f32).collect(), hidden: out.readouts[span.start].hidden.clone() }
            })
            .collect();
        Ok(BackendOutput {
            readouts,
            input_tokens: out.input_tokens,
            prefix_cached: out.prefix_cached,
        })
    }

    /// Record feedback on a stored decision and (optionally) learn from it.
    pub fn feedback(&self, fb: &FeedbackRequest) -> CevResult<FeedbackResponse> {
        let d = match (&fb.decision_id, &fb.request_id, &fb.question_id) {
            (Some(id), _, _) => self.store.decision(id)?,
            (None, Some(r), Some(q)) => self.store.decision_by_question(r, q)?,
            _ => return Err(CevError::BadRequest("give `decision_id`, or `request_id` and `question_id`".into())),
        }
        .ok_or_else(|| CevError::NotFound("decision not found (was it made with no_store?)".into()))?;
        if fb.label.is_none() && fb.comment.is_none() {
            return Err(CevError::BadRequest("feedback needs a `label` or a `comment`".into()));
        }
        let weight = fb.weight.unwrap_or(1.0);
        if !(weight.is_finite() && weight > 0.0) {
            return Err(CevError::BadRequest("`weight` must be positive".into()));
        }
        let y = fb
            .label
            .as_ref()
            .map(|l| target(d.kind, &d.options, l))
            .transpose()
            .map_err(CevError::BadRequest)?;
        let feedback_id = new_id("fb");
        self.store.insert_feedback(&FeedbackRecord {
            feedback_id: feedback_id.clone(),
            decision_id: d.decision_id.clone(),
            label: fb.label.clone(),
            target: y.clone(),
            weight,
            comment: fb.comment.clone(),
            metadata: fb.metadata.clone(),
            created_at: now_ms(),
        })?;
        let served_loss = y.as_ref().map(|y| math::log_loss(&d.served, y));
        let learned = match (&y, &d.hidden) {
            (Some(y), Some(h)) if self.cfg.online_learning && d.model == self.backend.id() => {
                self.learn(&d.task, Sample {
                    decision_id: d.decision_id.clone(),
                    names: d.options.iter().map(|o| o.name.clone()).collect(),
                    logits: d.base_logits.clone(),
                    hidden: h.clone(),
                    target: y.clone(),
                    weight: weight as f32,
                })?;
                true
            }
            _ => false,
        };
        let info = self.task(&d.task);
        Ok(FeedbackResponse {
            feedback_id,
            decision_id: d.decision_id,
            task: d.task,
            served_loss,
            learned,
            task_examples: info.as_ref().map_or(0, |t| t.examples),
            adapter_active: info.is_some_and(|t| t.active),
        })
    }

    fn learn(&self, task: &str, s: Sample) -> CevResult<()> {
        let norm = self.normalizer.lock().clone();
        let mut adapters = self.adapters.lock();
        let a = adapters.entry(task.to_string()).or_insert_with(|| Adapter::new(self.backend.hidden_size()));
        a.learn(s, &norm, &self.cfg.learn);
        self.store.save_adapter(self.backend.id(), task, a)?;
        drop(adapters);
        self.store.save_normalizer(self.backend.id(), &norm)?;
        Ok(())
    }

    /// Decide each example, then apply its labels as feedback.
    pub fn examples(&self, req: &ExamplesRequest) -> CevResult<Vec<(SystemOneResponse, Vec<FeedbackResponse>)>> {
        let mut out = Vec::with_capacity(req.examples.len());
        for ex in &req.examples {
            for k in ex.labels.keys() {
                if !ex.questions.contains_key(k) {
                    return Err(CevError::BadRequest(format!("label for unknown question `{k}`")));
                }
            }
            let resp = self.decide(&SystemOneRequest {
                state: ex.state.clone(),
                model: None,
                questions: ex.questions.clone(),
                no_store: false,
                debias: None,
            })?;
            let mut fbs = Vec::new();
            for (qid, label) in &ex.labels {
                fbs.push(self.feedback(&FeedbackRequest {
                    decision_id: None,
                    request_id: Some(resp.request_id.clone()),
                    question_id: Some(qid.clone()),
                    label: Some(label.clone()),
                    weight: None,
                    comment: None,
                    metadata: Some(serde_json::json!({"source": "examples"})),
                })?);
            }
            out.push((resp, fbs));
        }
        Ok(out)
    }

    pub fn tasks(&self) -> Vec<TaskInfo> {
        let adapters = self.adapters.lock();
        let mut v: Vec<TaskInfo> = adapters.iter().map(|(k, a)| self.info(k, a)).collect();
        v.sort_by(|a, b| a.task.cmp(&b.task));
        v
    }

    pub fn task(&self, task: &str) -> Option<TaskInfo> {
        self.adapters.lock().get(task).map(|a| self.info(task, a))
    }

    fn info(&self, task: &str, a: &Adapter) -> TaskInfo {
        TaskInfo {
            task: task.to_string(),
            examples: a.examples,
            buffered: a.buffer_len(),
            active: self.cfg.online_learning && a.is_active(&self.cfg.learn),
            temperature: a.temperature(),
            base_loss: a.base_loss,
            adapted_loss: a.adapted_loss,
        }
    }

    /// Drop a task's adapter (labels stay in the log).
    pub fn reset_task(&self, task: &str) -> CevResult<bool> {
        let had = self.adapters.lock().remove(task).is_some();
        Ok(self.store.delete_adapter(self.backend.id(), task)? || had)
    }

    pub fn decision(&self, id: &str) -> CevResult<Option<(DecisionRecord, Vec<FeedbackRecord>)>> {
        Ok(match self.store.decision(id)? {
            Some(d) => {
                let fb = self.store.feedback_for(id)?;
                Some((d, fb))
            }
            None => None,
        })
    }

    pub fn export(&self, f: &ExportFilter, emit: impl FnMut(ExportRow) -> anyhow::Result<()>) -> CevResult<usize> {
        Ok(self.store.export(f, emit)?)
    }

    pub fn stats(&self) -> CevResult<StoreStats> {
        Ok(self.store.stats()?)
    }
}

fn strip_decision_id(a: &mut Answer) {
    let meta = match a {
        Answer::Noul { x_cev, .. } | Answer::Choice { x_cev, .. } | Answer::Score { x_cev, .. } => x_cev,
    };
    if let Some(m) = meta {
        m.decision_id.clear();
    }
}
