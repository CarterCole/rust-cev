//! SQLite-backed store (native targets).

use super::*;
use crate::adapter::{Adapter, Normalizer, Sample};
use anyhow::Result;
use cev_core::OptionSpec;
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;
use std::path::Path;

const SCHEMA: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS requests (
    request_id   TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    state        TEXT NOT NULL,
    prefix       TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    latency_ms   REAL NOT NULL,
    created_at   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id  TEXT PRIMARY KEY,
    request_id   TEXT NOT NULL REFERENCES requests(request_id) ON DELETE CASCADE,
    question_id  TEXT NOT NULL,
    model        TEXT NOT NULL,
    task         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    question     TEXT NOT NULL,
    options      TEXT NOT NULL,
    codes        TEXT NOT NULL,
    suffix       TEXT NOT NULL,
    base_logits  BLOB NOT NULL,
    served       TEXT NOT NULL,
    adapted      INTEGER NOT NULL,
    hidden       BLOB,
    created_at   INTEGER NOT NULL,
    UNIQUE (request_id, question_id)
);
CREATE INDEX IF NOT EXISTS decisions_task ON decisions(model, task, created_at);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id  TEXT PRIMARY KEY,
    decision_id  TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
    label        TEXT,
    target       TEXT,
    weight       REAL NOT NULL,
    comment      TEXT,
    metadata     TEXT,
    created_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_decision ON feedback(decision_id, created_at);
CREATE TABLE IF NOT EXISTS adapters (
    model      TEXT NOT NULL,
    task       TEXT NOT NULL,
    data       TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (model, task)
);
CREATE TABLE IF NOT EXISTS normalizers (
    model TEXT PRIMARY KEY,
    data  TEXT NOT NULL
);
"#;

fn f32_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn blob_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub struct Store {
    conn: Mutex<Connection>,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::init(Connection::open(path)?)
    }

    pub fn in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn insert_request(&self, r: &RequestRecord, decisions: &[DecisionRecord]) -> Result<()> {
        let mut c = self.conn.lock();
        let tx = c.transaction()?;
        tx.execute(
            "INSERT INTO requests VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![r.request_id, r.model, r.state.to_string(), r.prefix, r.input_tokens as i64, r.latency_ms, r.created_at],
        )?;
        {
            let mut st = tx.prepare(
                "INSERT INTO decisions VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            )?;
            for d in decisions {
                st.execute(params![
                    d.decision_id,
                    d.request_id,
                    d.question_id,
                    d.model,
                    d.task,
                    d.kind.as_str(),
                    serde_json::to_string(&d.question)?,
                    serde_json::to_string(&d.options)?,
                    serde_json::to_string(&d.codes)?,
                    d.suffix,
                    f32_blob(&d.base_logits),
                    serde_json::to_string(&d.served)?,
                    d.adapted,
                    d.hidden.as_deref().map(f32_blob),
                    d.created_at,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    const DECISION_COLS: &'static str = "decision_id, request_id, question_id, model, task, kind, question, options, codes, suffix, base_logits, served, adapted, hidden, created_at";

    fn row_decision(row: &rusqlite::Row) -> rusqlite::Result<DecisionRecord> {
        fn de<T: serde::de::DeserializeOwned>(s: String) -> rusqlite::Result<T> {
            serde_json::from_str(&s).map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }
        Ok(DecisionRecord {
            decision_id: row.get(0)?,
            request_id: row.get(1)?,
            question_id: row.get(2)?,
            model: row.get(3)?,
            task: row.get(4)?,
            kind: de(format!("\"{}\"", row.get::<_, String>(5)?))?,
            question: de(row.get(6)?)?,
            options: de(row.get(7)?)?,
            codes: de(row.get(8)?)?,
            suffix: row.get(9)?,
            base_logits: blob_f32(&row.get::<_, Vec<u8>>(10)?),
            served: de(row.get(11)?)?,
            adapted: row.get(12)?,
            hidden: row.get::<_, Option<Vec<u8>>>(13)?.map(|b| blob_f32(&b)),
            created_at: row.get(14)?,
        })
    }

    pub fn decision(&self, decision_id: &str) -> Result<Option<DecisionRecord>> {
        let c = self.conn.lock();
        let sql = format!("SELECT {} FROM decisions WHERE decision_id = ?1", Self::DECISION_COLS);
        Ok(c.query_row(&sql, [decision_id], Self::row_decision).optional()?)
    }

    pub fn decision_by_question(&self, request_id: &str, question_id: &str) -> Result<Option<DecisionRecord>> {
        let c = self.conn.lock();
        let sql = format!("SELECT {} FROM decisions WHERE request_id = ?1 AND question_id = ?2", Self::DECISION_COLS);
        Ok(c.query_row(&sql, [request_id, question_id], Self::row_decision).optional()?)
    }

    pub fn request(&self, request_id: &str) -> Result<Option<RequestRecord>> {
        let c = self.conn.lock();
        Ok(c.query_row(
            "SELECT request_id, model, state, prefix, input_tokens, latency_ms, created_at FROM requests WHERE request_id = ?1",
            [request_id],
            |r| {
                Ok(RequestRecord {
                    request_id: r.get(0)?,
                    model: r.get(1)?,
                    state: serde_json::from_str(&r.get::<_, String>(2)?).unwrap_or(Value::Null),
                    prefix: r.get(3)?,
                    input_tokens: r.get::<_, i64>(4)? as usize,
                    latency_ms: r.get(5)?,
                    created_at: r.get(6)?,
                })
            },
        )
        .optional()?)
    }

    /// Decisions of a request, in question order.
    pub fn decisions_of(&self, request_id: &str) -> Result<Vec<DecisionRecord>> {
        let c = self.conn.lock();
        let sql = format!("SELECT {} FROM decisions WHERE request_id = ?1 ORDER BY rowid", Self::DECISION_COLS);
        let mut st = c.prepare(&sql)?;
        let rows = st.query_map([request_id], Self::row_decision)?.collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// Most recent decisions, optionally for one task.
    pub fn recent_decisions(&self, task: Option<&str>, limit: usize, offset: usize) -> Result<Vec<DecisionRecord>> {
        let c = self.conn.lock();
        let sql = format!(
            "SELECT {} FROM decisions WHERE (?1 IS NULL OR task = ?1) ORDER BY created_at DESC, rowid DESC LIMIT ?2 OFFSET ?3",
            Self::DECISION_COLS
        );
        let mut st = c.prepare(&sql)?;
        let rows = st
            .query_map(params![task, limit as i64, offset as i64], Self::row_decision)?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    pub fn insert_feedback(&self, f: &FeedbackRecord) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO feedback VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                f.feedback_id,
                f.decision_id,
                f.label.as_ref().map(serde_json::to_string).transpose()?,
                f.target.as_ref().map(serde_json::to_string).transpose()?,
                f.weight,
                f.comment,
                f.metadata.as_ref().map(Value::to_string),
                f.created_at,
            ],
        )?;
        Ok(())
    }

    pub fn feedback_for(&self, decision_id: &str) -> Result<Vec<FeedbackRecord>> {
        let c = self.conn.lock();
        let mut st = c.prepare(
            "SELECT feedback_id, decision_id, label, target, weight, comment, metadata, created_at
             FROM feedback WHERE decision_id = ?1 ORDER BY created_at, rowid",
        )?;
        let rows = st
            .query_map([decision_id], |r| {
                let js = |i: usize| -> rusqlite::Result<Option<String>> { r.get(i) };
                Ok(FeedbackRecord {
                    feedback_id: r.get(0)?,
                    decision_id: r.get(1)?,
                    label: js(2)?.and_then(|s| serde_json::from_str(&s).ok()),
                    target: js(3)?.and_then(|s| serde_json::from_str(&s).ok()),
                    weight: r.get(4)?,
                    comment: r.get(5)?,
                    metadata: js(6)?.and_then(|s| serde_json::from_str(&s).ok()),
                    created_at: r.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    /// Latest label per decision for a task, oldest first, as replay samples.
    pub fn replay(&self, model: &str, task: &str, limit: usize) -> Result<Vec<Sample>> {
        let c = self.conn.lock();
        let mut st = c.prepare(
            "SELECT d.decision_id, d.options, d.base_logits, d.hidden, f.target, f.weight FROM feedback f
             JOIN decisions d ON d.decision_id = f.decision_id
             WHERE d.model = ?1 AND d.task = ?2 AND f.target IS NOT NULL AND d.hidden IS NOT NULL
               AND f.rowid = (SELECT f2.rowid FROM feedback f2 WHERE f2.decision_id = f.decision_id
                              AND f2.target IS NOT NULL ORDER BY f2.created_at DESC, f2.rowid DESC LIMIT 1)
             ORDER BY f.created_at DESC, f.rowid DESC LIMIT ?3",
        )?;
        let mut rows: Vec<Sample> = st
            .query_map(params![model, task, limit as i64], |r| {
                let options: Vec<OptionSpec> = serde_json::from_str(&r.get::<_, String>(1)?).unwrap_or_default();
                Ok(Sample {
                    decision_id: r.get(0)?,
                    names: options.into_iter().map(|o| o.name).collect(),
                    logits: blob_f32(&r.get::<_, Vec<u8>>(2)?),
                    hidden: blob_f32(&r.get::<_, Vec<u8>>(3)?),
                    target: serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or_default(),
                    weight: r.get::<_, f64>(5)? as f32,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        rows.reverse();
        Ok(rows)
    }

    pub fn save_adapter(&self, model: &str, task: &str, a: &Adapter) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO adapters VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(model, task) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at",
            params![model, task, serde_json::to_string(a)?, now_ms()],
        )?;
        Ok(())
    }

    pub fn load_adapters(&self, model: &str) -> Result<Vec<(String, Adapter)>> {
        let c = self.conn.lock();
        let mut st = c.prepare("SELECT task, data FROM adapters WHERE model = ?1")?;
        let rows = st
            .query_map([model], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows
            .into_iter()
            .filter_map(|(t, d)| serde_json::from_str(&d).ok().map(|a| (t, a)))
            .collect())
    }

    pub fn delete_adapter(&self, model: &str, task: &str) -> Result<bool> {
        Ok(self.conn.lock().execute("DELETE FROM adapters WHERE model = ?1 AND task = ?2", [model, task])? > 0)
    }

    pub fn save_normalizer(&self, model: &str, n: &Normalizer) -> Result<()> {
        self.conn.lock().execute(
            "INSERT INTO normalizers VALUES (?1, ?2) ON CONFLICT(model) DO UPDATE SET data = excluded.data",
            params![model, serde_json::to_string(n)?],
        )?;
        Ok(())
    }

    pub fn load_normalizer(&self, model: &str) -> Result<Option<Normalizer>> {
        let c = self.conn.lock();
        let s: Option<String> = c
            .query_row("SELECT data FROM normalizers WHERE model = ?1", [model], |r| r.get(0))
            .optional()?;
        Ok(s.and_then(|s| serde_json::from_str(&s).ok()))
    }

    /// Stream training rows, using the latest label per decision (comment-only
    /// feedback never hides an earlier label).
    pub fn export(&self, f: &ExportFilter, mut emit: impl FnMut(ExportRow) -> Result<()>) -> Result<usize> {
        let labeled = f.labeled.unwrap_or(true);
        let c = self.conn.lock();
        let sql = format!(
            "SELECT {cols}, r.prefix, r.state, fb.feedback_id, fb.label, fb.target, fb.weight, fb.comment, fb.metadata, fb.created_at
             FROM decisions d JOIN requests r ON r.request_id = d.request_id
             {join} JOIN feedback fb ON fb.rowid = (SELECT f2.rowid FROM feedback f2 WHERE f2.decision_id = d.decision_id
                 ORDER BY (f2.target IS NOT NULL) DESC, f2.created_at DESC, f2.rowid DESC LIMIT 1)
             WHERE (?1 IS NULL OR d.task = ?1) AND (?2 IS NULL OR d.model = ?2)
               AND (?3 IS NULL OR COALESCE(fb.created_at, d.created_at) >= ?3)
               {label_filter}
             ORDER BY d.created_at, d.rowid LIMIT ?4",
            cols = Self::DECISION_COLS.split(", ").map(|c| format!("d.{c}")).collect::<Vec<_>>().join(", "),
            join = if labeled { "" } else { "LEFT" },
            label_filter = if labeled { "AND fb.target IS NOT NULL" } else { "" },
        );
        let mut st = c.prepare(&sql)?;
        let limit = f.limit.map(|l| l as i64).unwrap_or(-1);
        let mut rows = st.query(params![f.task, f.model, f.since, limit])?;
        let mut n = 0;
        while let Some(r) = rows.next()? {
            let d = Self::row_decision(r)?;
            let prefix: String = r.get(15)?;
            let js = |i: usize| -> rusqlite::Result<Option<String>> { r.get(i) };
            let row = ExportRow {
                prompt: format!("{prefix}{}", d.suffix),
                state: serde_json::from_str(&r.get::<_, String>(16)?).unwrap_or(Value::Null),
                feedback_id: r.get(17)?,
                label: js(18)?.and_then(|s| serde_json::from_str(&s).ok()),
                target: js(19)?.and_then(|s| serde_json::from_str(&s).ok()),
                weight: r.get::<_, Option<f64>>(20)?.unwrap_or(1.0),
                comment: r.get(21)?,
                metadata: js(22)?.and_then(|s| serde_json::from_str(&s).ok()),
                labeled_at: r.get(23)?,
                base_probabilities: cev_core::math::softmax(&d.base_logits),
                options: d.options.iter().map(|o| o.name.clone()).collect(),
                decision_id: d.decision_id,
                request_id: d.request_id,
                question_id: d.question_id,
                model: d.model,
                task: d.task,
                kind: d.kind,
                codes: d.codes,
                served: d.served,
                adapted: d.adapted,
                question: d.question,
                decided_at: d.created_at,
            };
            emit(row)?;
            n += 1;
        }
        Ok(n)
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let c = self.conn.lock();
        let q = |sql: &str| -> Result<i64> { Ok(c.query_row(sql, [], |r| r.get(0))?) };
        Ok(StoreStats {
            requests: q("SELECT COUNT(*) FROM requests")?,
            decisions: q("SELECT COUNT(*) FROM decisions")?,
            feedback: q("SELECT COUNT(*) FROM feedback")?,
            labeled_decisions: q("SELECT COUNT(DISTINCT decision_id) FROM feedback WHERE target IS NOT NULL")?,
        })
    }
}
