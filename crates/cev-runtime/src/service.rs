//! The decision service: compile → backbone → adapters → typed answers,
//! with every decision logged and every label learned from.
//!
//! This is what `cev-server` exposes over HTTP/GraphQL and what the `cev`
//! SDK embeds in-process.

use crate::adapter::{Adapter, LearnConfig, Normalizer, Sample};
use crate::store::{DecisionRecord, ExportFilter, ExportRow, FeedbackRecord, RequestRecord, Store, StoreStats, now_ms};
use cev_core::{
    Answer, AnswerMeta, Backend, BackendOutput, CompiledRequest, Debias, ExamplesRequest, FeedbackRequest, FeedbackResponse, ModelInfo, Readout, SystemOneRequest,
    SystemOneResponse, Usage, compile, content_free, math, target::target,
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
    Internal(anyhow::Error),
}

/// Backend errors are internal unless they carry a [`cev_core::InputError`]:
/// those were caused by the request and are the client's to fix.
impl From<anyhow::Error> for CevError {
    fn from(e: anyhow::Error) -> Self {
        if e.chain().any(|c| c.is::<cev_core::InputError>()) {
            CevError::BadRequest(format!("{e:#}"))
        } else {
            CevError::Internal(e)
        }
    }
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
    /// Default position-bias correction (requests may override). `Auto` is
    /// the backend's own default: `full` for zero-shot code readouts, `none`
    /// for trained decision models.
    pub debias: Debias,
    /// Most option rotations evaluated per question when permuting.
    pub max_permutations: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { online_learning: true, store_features: true, learn: LearnConfig::default(), debias: Debias::Auto, max_permutations: 4 }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskInfo {
    pub task: String,
    /// Feedback bucket; empty for the default one.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub namespace: String,
    pub examples: u64,
    pub buffered: usize,
    pub active: bool,
    pub temperature: f32,
    /// Decayed prequential log loss on labels, before learning from them.
    pub base_loss: f64,
    pub adapted_loss: f64,
    /// The same comparison on the Brier score.
    pub base_brier: f64,
    pub adapted_brier: f64,
}

/// What [`Cev::import_adapters`] did.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ImportStats {
    pub imported: usize,
    /// Left alone because the task already had an adapter here.
    pub skipped: usize,
    /// Imported with the probe restarted, because the hidden size differs.
    pub probe_reset: usize,
}

pub struct Cev {
    backend: Arc<dyn Backend>,
    store: Store,
    cfg: RuntimeConfig,
    /// Online adapters by (namespace, task).
    adapters: Mutex<HashMap<(String, String), Adapter>>,
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

/// Wire namespace to stored namespace: absent or blank is the default bucket.
fn namespace_key(ns: Option<&str>) -> CevResult<String> {
    let ns = ns.map(str::trim).unwrap_or_default();
    if ns.len() > 200 {
        return Err(CevError::BadRequest("`namespace` is longer than 200 bytes".into()));
    }
    Ok(ns.to_string())
}

fn namespace_wire(ns: &str) -> Option<String> {
    (!ns.is_empty()).then(|| ns.to_string())
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
        for (namespace, task, mut a) in store.load_adapters(&model)? {
            if a.dim == dim {
                a.set_buffer(store.replay(&model, &namespace, &task, cfg.learn.buffer)?);
                adapters.insert((namespace, task), a);
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
        let caps = self.backend.capabilities();
        ModelInfo {
            id: self.backend.id().into(),
            object: "model",
            backbone: self.backend.backbone().into(),
            family: caps.family.as_str().into(),
            modalities: caps.modalities().iter().map(|m| m.to_string()).collect(),
            hidden_size: self.backend.hidden_size(),
            max_options: caps.max_options,
            prompt_version: caps.prompt_version().into(),
        }
    }

    /// Answer every question of a request. Blocking (runs the network).
    pub fn decide(&self, req: &SystemOneRequest) -> CevResult<SystemOneResponse> {
        let started = Instant::now();
        let namespace = namespace_key(req.namespace.as_deref())?;
        let caps = self.backend.capabilities();
        let compiled = compile(req, caps.max_options).map_err(|e| CevError::BadRequest(e.to_string()))?;
        if !compiled.media.images.is_empty() && !caps.images {
            return Err(CevError::BadRequest(format!("`images`: the {} backend `{}` takes text only", caps.family.as_str(), self.backend.id())));
        }
        if compiled.media.audio.is_some() && !caps.audio {
            return Err(CevError::BadRequest(format!("`audio`: the {} backend `{}` does not take audio", caps.family.as_str(), self.backend.id())));
        }
        if compiled.media.audio.is_some() && !compiled.media.images.is_empty() && caps.family == cev_core::Family::D1Omni {
            return Err(CevError::BadRequest("a request carries images or audio, not both".into()));
        }
        let debias = req.debias.unwrap_or(self.cfg.debias).resolve(caps.debias);
        let out = self.run_debiased(&compiled, debias)?;
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
                let adapter = adapters.get(&(namespace.clone(), q.task.clone())).filter(|a| self.cfg.online_learning && a.is_active(&self.cfg.learn));
                let served = match adapter {
                    Some(a) => a.predict(&names, &r.logits, &norm.features(&r.hidden)),
                    None => base.clone(),
                };
                let decision_id = new_id("dec");
                let meta = AnswerMeta {
                    decision_id: decision_id.clone(),
                    task: q.task.clone(),
                    namespace: namespace_wire(&namespace),
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
                    namespace: namespace.clone(),
                    kind: q.kind,
                    question: req.questions[&q.id].clone(),
                    options: q.options.clone(),
                    codes: r.codes.clone(),
                    suffix: r.suffix.clone(),
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
                    prefix: out.prefix.clone(),
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
    /// order and whose `hidden` (and prompt) come from the unrotated variant.
    fn run_debiased(&self, compiled: &CompiledRequest, debias: Debias) -> anyhow::Result<BackendOutput> {
        if debias == Debias::None {
            return self.backend.run(compiled);
        }
        let mut variants = Vec::new();
        let mut spans = Vec::with_capacity(compiled.questions.len());
        for q in &compiled.questions {
            let start = variants.len();
            for shift in debias.shifts(q.options.len(), self.cfg.max_permutations) {
                variants.push(if shift == 0 { q.clone() } else { q.rotated(shift) });
            }
            spans.push(start..variants.len());
        }
        let main = CompiledRequest { questions: variants, ..compiled.clone() };
        // Calibration twins: same question variants over content-free evidence.
        // Cached ones are reused; the rest run fused with the request.
        let keys: Vec<String> = main.questions.iter().map(|q| q.fingerprint()).collect();
        let mut cf: Vec<Option<Vec<f32>>> = vec![None; main.questions.len()];
        let mut missing = Vec::new();
        if debias.calibrates() {
            let mut cache = self.calibration.lock();
            for (i, key) in keys.iter().enumerate() {
                match cache.get(key) {
                    Some(z) => cf[i] = Some(z),
                    None => missing.push(i),
                }
            }
        }
        let out = if missing.is_empty() {
            self.backend.run(&main)?
        } else {
            let mut twin = content_free(&main);
            twin.questions = missing.iter().map(|&i| main.questions[i].clone()).collect();
            let mut outs = self.backend.run_many(&[main.clone(), twin])?;
            let twin_out = outs.pop().expect("twin output");
            let mut cache = self.calibration.lock();
            for (&i, r) in missing.iter().zip(twin_out.readouts) {
                cache.put(keys[i].clone(), r.logits.clone());
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
                let base = &out.readouts[span.start];
                Readout { logits: score.into_iter().map(|v| v as f32).collect(), hidden: base.hidden.clone(), suffix: base.suffix.clone(), codes: base.codes.clone() }
            })
            .collect();
        Ok(BackendOutput { readouts, input_tokens: out.input_tokens, prefix_cached: out.prefix_cached, prefix: out.prefix })
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
                self.learn(&d.namespace, &d.task, Sample {
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
        let info = self.task(&d.namespace, &d.task);
        Ok(FeedbackResponse {
            feedback_id,
            decision_id: d.decision_id,
            namespace: namespace_wire(&d.namespace),
            task: d.task,
            served_loss,
            learned,
            task_examples: info.as_ref().map_or(0, |t| t.examples),
            adapter_active: info.is_some_and(|t| t.active),
        })
    }

    fn learn(&self, namespace: &str, task: &str, s: Sample) -> CevResult<()> {
        let norm = self.normalizer.lock().clone();
        let mut adapters = self.adapters.lock();
        let a = adapters
            .entry((namespace.to_string(), task.to_string()))
            .or_insert_with(|| Adapter::with_probe(self.backend.hidden_size(), self.cfg.learn.probe_dim));
        a.learn(s, &norm, &self.cfg.learn);
        self.store.save_adapter(self.backend.id(), namespace, task, a)?;
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
                namespace: ex.namespace.clone(),
                images: ex.images.clone(),
                audio: ex.audio.clone(),
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

    /// Every adapter, or only those of one namespace (`""` is the default).
    pub fn tasks(&self, namespace: Option<&str>) -> Vec<TaskInfo> {
        let adapters = self.adapters.lock();
        let mut v: Vec<TaskInfo> = adapters
            .iter()
            .filter(|((ns, _), _)| namespace.is_none_or(|n| n.trim() == ns))
            .map(|((ns, t), a)| self.info(ns, t, a))
            .collect();
        v.sort_by(|a, b| (&a.namespace, &a.task).cmp(&(&b.namespace, &b.task)));
        v
    }

    /// One task's adapter in a namespace (`""` is the default).
    pub fn task(&self, namespace: &str, task: &str) -> Option<TaskInfo> {
        let ns = namespace.trim();
        self.adapters.lock().get(&(ns.to_string(), task.to_string())).map(|a| self.info(ns, task, a))
    }

    fn info(&self, namespace: &str, task: &str, a: &Adapter) -> TaskInfo {
        TaskInfo {
            task: task.to_string(),
            namespace: namespace.to_string(),
            examples: a.examples,
            buffered: a.buffer_len(),
            active: self.cfg.online_learning && a.is_active(&self.cfg.learn),
            temperature: a.temperature(),
            base_loss: a.base_loss,
            adapted_loss: a.adapted_loss,
            base_brier: a.base_brier,
            adapted_brier: a.adapted_brier,
        }
    }

    /// Take adapters fit elsewhere (another database, or an older checkpoint)
    /// as starting points for this backbone, whatever model they came from.
    ///
    /// They are not served until they beat the base model on labels received
    /// here (see [`Adapter::transplant`]). Tasks that already have an adapter
    /// are kept unless `overwrite` is set.
    pub fn import_adapters(&self, rows: &[crate::store::AdapterRow], overwrite: bool) -> CevResult<ImportStats> {
        let model = self.backend.id();
        let dim = self.backend.hidden_size();
        let mut stats = ImportStats::default();
        let mut adapters = self.adapters.lock();
        for row in rows {
            let namespace = namespace_key(Some(&row.namespace))?;
            let key = (namespace.clone(), row.task.clone());
            if adapters.contains_key(&key) && !overwrite {
                stats.skipped += 1;
                continue;
            }
            let mut a = row.adapter.transplant(dim, self.cfg.learn.probe_dim);
            a.set_buffer(self.store.replay(model, &namespace, &row.task, self.cfg.learn.buffer)?);
            self.store.save_adapter(model, &namespace, &row.task, &a)?;
            stats.probe_reset += usize::from(row.adapter.dim != dim);
            adapters.insert(key, a);
            stats.imported += 1;
        }
        Ok(stats)
    }

    /// Drop a task's adapter in one namespace (labels stay in the log).
    pub fn reset_task(&self, namespace: &str, task: &str) -> CevResult<bool> {
        let ns = namespace.trim();
        let had = self.adapters.lock().remove(&(ns.to_string(), task.to_string())).is_some();
        Ok(self.store.delete_adapter(self.backend.id(), ns, task)? || had)
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
