//! In-memory store (browser builds, where SQLite is not available). Same
//! API and ordering semantics as the SQLite store; nothing survives a reload.

use super::*;
use crate::adapter::{Adapter, Normalizer, Sample};
use anyhow::{Result, bail};
use parking_lot::Mutex;
use std::collections::HashMap;

#[derive(Default)]
struct Inner {
    requests: HashMap<String, RequestRecord>,
    /// Insertion order (the SQLite rowid order).
    decisions: Vec<DecisionRecord>,
    by_id: HashMap<String, usize>,
    feedback: Vec<FeedbackRecord>,
    adapters: HashMap<(String, String), String>,
    normalizers: HashMap<String, String>,
}

impl Inner {
    /// Feedback of a decision, latest first; with `labels_first`, labelled
    /// feedback outranks newer comment-only feedback.
    fn latest_feedback(&self, decision_id: &str, labels_first: bool) -> Option<&FeedbackRecord> {
        self.feedback
            .iter()
            .enumerate()
            .filter(|(_, f)| f.decision_id == decision_id)
            .max_by_key(|(i, f)| (labels_first && f.target.is_some(), f.created_at, *i))
            .map(|(_, f)| f)
    }
}

#[derive(Default)]
pub struct Store {
    inner: Mutex<Inner>,
}

impl Store {
    pub fn in_memory() -> Result<Self> {
        Ok(Self::default())
    }

    pub fn insert_request(&self, r: &RequestRecord, decisions: &[DecisionRecord]) -> Result<()> {
        let mut s = self.inner.lock();
        if s.requests.contains_key(&r.request_id) {
            bail!("duplicate request id");
        }
        s.requests.insert(r.request_id.clone(), r.clone());
        for d in decisions {
            let i = s.decisions.len();
            s.by_id.insert(d.decision_id.clone(), i);
            s.decisions.push(d.clone());
        }
        Ok(())
    }

    pub fn decision(&self, decision_id: &str) -> Result<Option<DecisionRecord>> {
        let s = self.inner.lock();
        Ok(s.by_id.get(decision_id).map(|&i| s.decisions[i].clone()))
    }

    pub fn decision_by_question(&self, request_id: &str, question_id: &str) -> Result<Option<DecisionRecord>> {
        let s = self.inner.lock();
        Ok(s.decisions.iter().find(|d| d.request_id == request_id && d.question_id == question_id).cloned())
    }

    pub fn request(&self, request_id: &str) -> Result<Option<RequestRecord>> {
        Ok(self.inner.lock().requests.get(request_id).cloned())
    }

    /// Decisions of a request, in question order.
    pub fn decisions_of(&self, request_id: &str) -> Result<Vec<DecisionRecord>> {
        Ok(self.inner.lock().decisions.iter().filter(|d| d.request_id == request_id).cloned().collect())
    }

    /// Most recent decisions, optionally for one task.
    pub fn recent_decisions(&self, task: Option<&str>, limit: usize, offset: usize) -> Result<Vec<DecisionRecord>> {
        let s = self.inner.lock();
        let mut rows: Vec<(usize, &DecisionRecord)> =
            s.decisions.iter().enumerate().filter(|(_, d)| task.is_none_or(|t| d.task == t)).collect();
        rows.sort_by_key(|(i, d)| std::cmp::Reverse((d.created_at, *i)));
        Ok(rows.into_iter().skip(offset).take(limit).map(|(_, d)| d.clone()).collect())
    }

    pub fn insert_feedback(&self, f: &FeedbackRecord) -> Result<()> {
        let mut s = self.inner.lock();
        if !s.by_id.contains_key(&f.decision_id) {
            bail!("feedback for unknown decision");
        }
        s.feedback.push(f.clone());
        Ok(())
    }

    pub fn feedback_for(&self, decision_id: &str) -> Result<Vec<FeedbackRecord>> {
        let s = self.inner.lock();
        let mut rows: Vec<FeedbackRecord> = s.feedback.iter().filter(|f| f.decision_id == decision_id).cloned().collect();
        rows.sort_by_key(|f| f.created_at);
        Ok(rows)
    }

    /// Latest label per decision for a task, oldest first, as replay samples.
    pub fn replay(&self, model: &str, task: &str, limit: usize) -> Result<Vec<Sample>> {
        let s = self.inner.lock();
        let mut rows: Vec<(i64, Sample)> = s
            .decisions
            .iter()
            .filter(|d| d.model == model && d.task == task)
            .filter_map(|d| {
                let f = s.latest_feedback(&d.decision_id, true)?;
                Some((f.created_at, Sample {
                    decision_id: d.decision_id.clone(),
                    names: d.options.iter().map(|o| o.name.clone()).collect(),
                    logits: d.base_logits.clone(),
                    hidden: d.hidden.clone()?,
                    target: f.target.clone()?,
                    weight: f.weight as f32,
                }))
            })
            .collect();
        rows.sort_by_key(|(t, _)| *t);
        let skip = rows.len().saturating_sub(limit);
        Ok(rows.into_iter().skip(skip).map(|(_, s)| s).collect())
    }

    pub fn save_adapter(&self, model: &str, task: &str, a: &Adapter) -> Result<()> {
        self.inner.lock().adapters.insert((model.into(), task.into()), serde_json::to_string(a)?);
        Ok(())
    }

    pub fn load_adapters(&self, model: &str) -> Result<Vec<(String, Adapter)>> {
        let s = self.inner.lock();
        Ok(s.adapters
            .iter()
            .filter(|((m, _), _)| m == model)
            .filter_map(|((_, t), d)| serde_json::from_str(d).ok().map(|a| (t.clone(), a)))
            .collect())
    }

    pub fn delete_adapter(&self, model: &str, task: &str) -> Result<bool> {
        Ok(self.inner.lock().adapters.remove(&(model.to_string(), task.to_string())).is_some())
    }

    pub fn save_normalizer(&self, model: &str, n: &Normalizer) -> Result<()> {
        self.inner.lock().normalizers.insert(model.into(), serde_json::to_string(n)?);
        Ok(())
    }

    pub fn load_normalizer(&self, model: &str) -> Result<Option<Normalizer>> {
        Ok(self.inner.lock().normalizers.get(model).and_then(|s| serde_json::from_str(s).ok()))
    }

    /// Stream training rows, using the latest label per decision (comment-only
    /// feedback never hides an earlier label).
    pub fn export(&self, f: &ExportFilter, mut emit: impl FnMut(ExportRow) -> Result<()>) -> Result<usize> {
        let labeled = f.labeled.unwrap_or(true);
        let s = self.inner.lock();
        let mut n = 0;
        for d in &s.decisions {
            if f.limit.is_some_and(|l| n >= l) {
                break;
            }
            let fb = s.latest_feedback(&d.decision_id, true);
            if labeled && fb.is_none_or(|fb| fb.target.is_none()) {
                continue;
            }
            if f.task.as_ref().is_some_and(|t| *t != d.task) || f.model.as_ref().is_some_and(|m| *m != d.model) {
                continue;
            }
            if f.since.is_some_and(|t| fb.map_or(d.created_at, |fb| fb.created_at) < t) {
                continue;
            }
            let Some(r) = s.requests.get(&d.request_id) else { continue };
            emit(ExportRow {
                decision_id: d.decision_id.clone(),
                request_id: d.request_id.clone(),
                question_id: d.question_id.clone(),
                model: d.model.clone(),
                task: d.task.clone(),
                kind: d.kind,
                prompt: format!("{}{}", r.prefix, d.suffix),
                codes: d.codes.clone(),
                options: d.options.iter().map(|o| o.name.clone()).collect(),
                target: fb.and_then(|fb| fb.target.clone()),
                label: fb.and_then(|fb| fb.label.clone()),
                weight: fb.map_or(1.0, |fb| fb.weight),
                served: d.served.clone(),
                base_probabilities: cev_core::math::softmax(&d.base_logits),
                adapted: d.adapted,
                comment: fb.and_then(|fb| fb.comment.clone()),
                metadata: fb.and_then(|fb| fb.metadata.clone()),
                state: r.state.clone(),
                question: d.question.clone(),
                feedback_id: fb.map(|fb| fb.feedback_id.clone()),
                decided_at: d.created_at,
                labeled_at: fb.map(|fb| fb.created_at),
            })?;
            n += 1;
        }
        Ok(n)
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let s = self.inner.lock();
        let labeled: std::collections::HashSet<&str> =
            s.feedback.iter().filter(|f| f.target.is_some()).map(|f| f.decision_id.as_str()).collect();
        Ok(StoreStats {
            requests: s.requests.len() as i64,
            decisions: s.decisions.len() as i64,
            feedback: s.feedback.len() as i64,
            labeled_decisions: labeled.len() as i64,
        })
    }
}
