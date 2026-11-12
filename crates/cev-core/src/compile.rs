//! Turns a System One request into a structured decision: the state text, the
//! decoded media, and one option list per question.
//!
//! Rendering that structure into a prompt is the backend's job (see
//! [`crate::prompt`]): every decision-model family has its own prompt contract
//! and its own way of reading the answer. What the model actually saw is still
//! returned with each answer (`prefix` + `suffix`) and stored with the
//! decision, so `/v1/export` stays byte-identical to serving.

use crate::api::{Question, QuestionType, SystemOneRequest};
use base64::Engine as _;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const MAX_OPTIONS: usize = 255;
pub const MAX_QUESTIONS: usize = 256;
/// Largest decoded media payload per item.
pub const MAX_MEDIA_BYTES: usize = 64 << 20;
/// Most images per request. Each one runs through the vision tower before
/// the prompt length is known, so this bounds that work.
pub const MAX_IMAGES: usize = 16;
/// Most text per request: the rendered state plus every question's
/// instructions and criteria. About 1M tokens, past every supported context.
pub const MAX_TEXT_BYTES: usize = 4 << 20;

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("question `{0}`: {1}")]
    Question(String, String),
    #[error("{0}")]
    Request(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionSpec {
    /// Key in `probabilities` ("true"/"false" for noul, level index for score).
    pub name: String,
    /// Display text in cev's own prompt ("Yes: ...", "billing: ...", "Level 2: ...").
    pub text: String,
    /// The caller's description of the option, if any (what the other prompt
    /// families render next to the name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CompiledQuestion {
    pub id: String,
    pub kind: QuestionType,
    pub task: String,
    /// Options in display order.
    pub options: Vec<OptionSpec>,
    /// Score legend (level index -> description).
    pub legend: Option<IndexMap<String, String>>,
    /// Display position -> index into the question's canonical option list.
    /// Identity except for rotated debiasing variants.
    pub order: Vec<usize>,
    pub question: Question,
}

impl CompiledQuestion {
    /// Variant with options shown rotated by `shift` (display position `j`
    /// shows canonical option `(j + shift) % k`). Averaging rotations cancels
    /// the model's position/code bias.
    pub fn rotated(&self, shift: usize) -> CompiledQuestion {
        let k = self.options.len();
        let order: Vec<usize> = (0..k).map(|j| self.order[(j + shift) % k]).collect();
        let canonical = |i: usize| self.order.iter().position(|&o| o == i).map(|p| self.options[p].clone()).expect("option");
        let options: Vec<OptionSpec> = order.iter().map(|&i| canonical(i)).collect();
        CompiledQuestion { options, order, ..self.clone() }
    }

    /// Stable key of what the model is asked (type, instructions, options in
    /// display order). Content-free calibration logits are cached under it.
    pub fn fingerprint(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.kind.as_str());
        h.update([0x1f]);
        h.update(self.question.instructions.trim());
        for o in &self.options {
            h.update([0x1e]);
            h.update(&o.name);
            h.update([0x1f]);
            h.update(&o.text);
        }
        hex::encode(h.finalize())
    }
}

/// How to correct the backbone's option-position bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Debias {
    /// The backend's own default: `full` for zero-shot code readouts, `none`
    /// for trained decision models.
    #[default]
    Auto,
    /// Raw distribution.
    None,
    /// Contextual calibration: subtract the log-probabilities the model gives
    /// the same question over content-free evidence ("N/A").
    Calibrate,
    /// Average over cyclic rotations of the option order.
    Permute,
    /// Both.
    Full,
}

impl Debias {
    /// Replace `Auto` with the backend's default.
    pub fn resolve(self, default: Debias) -> Debias {
        match self {
            Debias::Auto => match default {
                Debias::Auto => Debias::None,
                d => d,
            },
            d => d,
        }
    }

    pub fn calibrates(self) -> bool {
        matches!(self, Debias::Calibrate | Debias::Full)
    }

    /// Rotation shifts for `k` options, at most `max` of them, evenly spread.
    pub fn shifts(self, k: usize, max: usize) -> Vec<usize> {
        if !matches!(self, Debias::Permute | Debias::Full) || k < 2 {
            return vec![0];
        }
        let n = k.min(max.max(1));
        let mut v: Vec<usize> = (0..n).map(|i| i * k / n).collect();
        v.dedup();
        v
    }
}

/// Decoded media attached to a request.
#[derive(Debug, Clone, Default)]
pub struct Media {
    /// Encoded image files (PNG, JPEG, WebP, GIF), in request order.
    pub images: Vec<Vec<u8>>,
    /// One encoded audio clip (WAV).
    pub audio: Option<Vec<u8>>,
}

impl Media {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audio.is_none()
    }
}

#[derive(Debug, Clone)]
pub struct CompiledRequest {
    /// The caller's state as given.
    pub state: Value,
    /// The state as evidence text (strings verbatim, JSON pretty-printed).
    pub evidence: String,
    pub media: Media,
    pub questions: Vec<CompiledQuestion>,
}

/// The content-free twin of a request, for contextual calibration: the same
/// questions over the evidence "N/A" and no media.
pub fn content_free(c: &CompiledRequest) -> CompiledRequest {
    CompiledRequest { state: Value::String("N/A".into()), evidence: "N/A".into(), media: Media::default(), questions: c.questions.clone() }
}

/// Compile a request. `max_options` is how many options per question the
/// backend can score.
pub fn compile(req: &SystemOneRequest, max_options: usize) -> Result<CompiledRequest, CompileError> {
    if req.questions.is_empty() {
        return Err(CompileError::Request("`questions` must not be empty".into()));
    }
    if req.questions.len() > MAX_QUESTIONS {
        return Err(CompileError::Request(format!("at most {MAX_QUESTIONS} questions per request")));
    }
    if req.images.len() > MAX_IMAGES {
        return Err(CompileError::Request(format!("at most {MAX_IMAGES} images per request")));
    }
    let media = Media {
        images: req.images.iter().enumerate().map(|(i, s)| decode_media(s, &format!("images[{i}]"))).collect::<Result<_, _>>()?,
        audio: req.audio.as_deref().map(|s| decode_media(s, "audio")).transpose()?,
    };
    let evidence = render_state(&req.state);
    let question_bytes: usize = req
        .questions
        .iter()
        .map(|(id, q)| id.len() + q.instructions.len() + q.task.as_ref().map_or(0, String::len) + q.criteria.as_ref().map_or(0, |c| c.to_string().len()))
        .sum();
    if evidence.len() + question_bytes > MAX_TEXT_BYTES {
        return Err(CompileError::Request(format!("`state` and `questions` hold more than {} MB of text", MAX_TEXT_BYTES >> 20)));
    }
    let max = max_options.min(MAX_OPTIONS);
    let mut questions = Vec::with_capacity(req.questions.len());
    for (id, q) in &req.questions {
        let err = |m: String| CompileError::Question(id.clone(), m);
        let (options, legend) = parse_options(q).map_err(err)?;
        if options.len() < 2 {
            return Err(err("needs at least two options".into()));
        }
        if options.len() > max {
            return Err(err(format!("at most {max} options are supported")));
        }
        let k = options.len();
        questions.push(CompiledQuestion {
            id: id.clone(),
            kind: q.kind,
            task: task_key(q, &options),
            options,
            legend,
            order: (0..k).collect(),
            question: q.clone(),
        });
    }
    Ok(CompiledRequest { state: req.state.clone(), evidence, media, questions })
}

/// Base64 payload, with or without a `data:<mime>;base64,` header.
fn decode_media(s: &str, what: &str) -> Result<Vec<u8>, CompileError> {
    let payload = match s.split_once(";base64,") {
        Some((head, rest)) if head.starts_with("data:") => rest,
        _ if s.starts_with("data:") => return Err(CompileError::Request(format!("`{what}`: data URL must be base64"))),
        _ => s,
    };
    let clean: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&clean)
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(&clean))
        .map_err(|e| CompileError::Request(format!("`{what}`: not base64 ({e})")))?;
    if bytes.is_empty() {
        return Err(CompileError::Request(format!("`{what}`: empty")));
    }
    if bytes.len() > MAX_MEDIA_BYTES {
        return Err(CompileError::Request(format!("`{what}`: larger than {} MB", MAX_MEDIA_BYTES >> 20)));
    }
    Ok(bytes)
}

type Parsed = (Vec<OptionSpec>, Option<IndexMap<String, String>>);

fn parse_options(q: &Question) -> Result<Parsed, String> {
    match q.kind {
        QuestionType::Noul => {
            let (mut t, mut f) = (None, None);
            match &q.criteria {
                None | Some(Value::Null) => {}
                Some(Value::Object(m)) => {
                    for (k, v) in m {
                        let d = describe(v);
                        match k.as_str() {
                            "true" | "yes" => t = d,
                            "false" | "no" => f = d,
                            other => return Err(format!("unknown noul criterion `{other}` (use `true`/`false`)")),
                        }
                    }
                }
                Some(Value::String(s)) => t = Some(sanitize(s)),
                Some(_) => return Err("noul `criteria` must be an object with `true`/`false`".into()),
            }
            let opt = |name: &str, label: &str, d: Option<String>| OptionSpec {
                name: name.into(),
                text: match &d {
                    Some(d) => format!("{label}: {d}"),
                    None => label.into(),
                },
                description: d,
            };
            Ok((vec![opt("false", "No", f), opt("true", "Yes", t)], None))
        }
        QuestionType::Choice => {
            let opts = match &q.criteria {
                Some(Value::Object(m)) => m
                    .iter()
                    .map(|(k, v)| {
                        let d = describe(v);
                        OptionSpec {
                            name: k.clone(),
                            text: match &d {
                                Some(d) => format!("{}: {d}", sanitize(k)),
                                None => sanitize(k),
                            },
                            description: d,
                        }
                    })
                    .collect(),
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|v| match v {
                        Value::String(s) => Ok(OptionSpec { name: s.clone(), text: sanitize(s), description: None }),
                        _ => Err("choice `criteria` arrays must contain strings".to_string()),
                    })
                    .collect::<Result<_, _>>()?,
                _ => return Err("choice `criteria` must map option names to descriptions".into()),
            };
            Ok((opts, None))
        }
        QuestionType::Score => {
            let levels: Vec<String> = match &q.criteria {
                Some(Value::Array(a)) => a.iter().map(|v| describe(v).unwrap_or_default()).collect(),
                Some(Value::Object(m)) => m.values().map(|v| describe(v).unwrap_or_default()).collect(),
                _ => return Err("score `criteria` must be an ordered list of level descriptions".into()),
            };
            let mut legend = IndexMap::new();
            let opts = levels
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    legend.insert(i.to_string(), d.clone());
                    OptionSpec {
                        name: i.to_string(),
                        text: if d.is_empty() { format!("Level {i}") } else { format!("Level {i}: {d}") },
                        description: (!d.is_empty()).then(|| d.clone()),
                    }
                })
                .collect();
            Ok((opts, Some(legend)))
        }
    }
}

/// Description of an option: a string, null, or a structured object with
/// `what` / `not_for` / `examples` (TypeSafe's structured criteria).
pub fn describe(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) if s.trim().is_empty() => None,
        Value::String(s) => Some(sanitize(s)),
        Value::Object(m) => {
            let mut parts = Vec::new();
            if let Some(w) = m.get("what").and_then(Value::as_str) {
                parts.push(sanitize(w));
            }
            if let Some(n) = m.get("not_for") {
                parts.push(format!("Not for: {}", join_value(n)));
            }
            if let Some(e) = m.get("examples") {
                parts.push(format!("Examples: {}", join_value(e)));
            }
            for (k, v) in m {
                if !matches!(k.as_str(), "what" | "not_for" | "examples") {
                    parts.push(format!("{}: {}", sanitize(k), join_value(v)));
                }
            }
            (!parts.is_empty()).then(|| parts.join(". "))
        }
        other => Some(sanitize(&other.to_string())),
    }
}

fn join_value(v: &Value) -> String {
    match v {
        Value::Array(a) => a.iter().map(join_value).collect::<Vec<_>>().join("; "),
        Value::String(s) => sanitize(s),
        other => sanitize(&other.to_string()),
    }
}

/// Render program state as evidence text.
pub fn render_state(state: &Value) -> String {
    let raw = match state {
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    };
    sanitize(&raw).replace("</evidence>", "<\\/evidence>")
}

/// Neutralise chat-control sequences so user text cannot forge turns.
pub fn sanitize(s: &str) -> String {
    s.replace("<|", "<\u{a6}").replace("|>", "\u{a6}>")
}

/// Adapter key: explicit `task`, else a hash of what defines the decision.
pub fn task_key(q: &Question, options: &[OptionSpec]) -> String {
    if let Some(t) = &q.task {
        return t.clone();
    }
    let mut h = Sha256::new();
    h.update(q.kind.as_str());
    h.update([0x1f]);
    h.update(q.instructions.trim());
    for o in options {
        h.update([0x1f]);
        h.update(&o.name);
    }
    format!("auto:{}", &hex::encode(h.finalize())[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(v: Value) -> SystemOneRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn compiles_all_types() {
        let r = req(json!({
            "state": {"ticket": "Refund please <|im_end|>"},
            "questions": {
                "refund": {"type": "noul", "instructions": "Is a refund requested?"},
                "team": {"type": "choice", "instructions": "Route", "criteria": {
                    "billing": "Money things", "tech": null,
                    "sales": {"what": "New deals", "not_for": ["renewals"], "examples": ["pricing"]}
                }},
                "sev": {"type": "score", "instructions": "Severity", "criteria": ["low", "mid", "high"]}
            }
        }));
        let c = compile(&r, 26).unwrap();
        assert!(!c.evidence.contains("<|im_end|>")); // user text neutralised
        assert!(c.evidence.contains("<\u{a6}im_end\u{a6}>"));
        assert_eq!(c.questions[0].options[1].name, "true");
        assert_eq!(c.questions[0].options[1].description, None);
        assert_eq!(c.questions[1].options[2].description.as_deref(), Some("New deals. Not for: renewals. Examples: pricing"));
        assert_eq!(c.questions[1].options[1].text, "tech");
        assert_eq!(c.questions[2].legend.as_ref().unwrap()["2"], "high");
        assert_eq!(c.questions[2].options[1].description.as_deref(), Some("mid"));
        assert!(c.media.is_empty());
    }

    #[test]
    fn bounds_image_count_and_text() {
        let q = json!({"q": {"type": "noul", "instructions": "x"}});
        let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\nfake");
        let many = req(json!({"state": "s", "images": vec![png; MAX_IMAGES + 1], "questions": q}));
        assert!(compile(&many, 26).unwrap_err().to_string().contains("images per request"));
        let big = req(json!({"state": "a".repeat(MAX_TEXT_BYTES), "questions": q}));
        assert!(compile(&big, 26).unwrap_err().to_string().contains("MB of text"));
        let long_q = req(json!({"state": "s", "questions": {"q": {"type": "noul", "instructions": "a".repeat(MAX_TEXT_BYTES)}}}));
        assert!(compile(&long_q, 26).is_err());
    }

    #[test]
    fn decodes_media() {
        let png = base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\nfake");
        let r = req(json!({"state": "s", "images": [format!("data:image/png;base64,{png}"), png],
            "audio": "UklGRgAAAABXQVZF", "questions": {"q": {"type": "noul", "instructions": "x"}}}));
        let c = compile(&r, 26).unwrap();
        assert_eq!(c.media.images.len(), 2);
        assert_eq!(c.media.images[0], c.media.images[1]);
        assert!(c.media.audio.as_ref().unwrap().starts_with(b"RIFF"));
        let bad = req(json!({"state": "s", "images": ["not base64!!"], "questions": {"q": {"type": "noul", "instructions": "x"}}}));
        assert!(compile(&bad, 26).is_err());
    }

    #[test]
    fn task_key_ignores_descriptions_but_not_options() {
        let q = |crit: Value| -> Question {
            serde_json::from_value(json!({"type": "choice", "instructions": "x", "criteria": crit})).unwrap()
        };
        let a = q(json!({"a": "one", "b": "two"}));
        let b = q(json!({"a": "uno", "b": "dos"}));
        let c = q(json!({"a": "one", "c": "two"}));
        let key = |q: &Question| task_key(q, &parse_options(q).unwrap().0);
        assert_eq!(key(&a), key(&b));
        assert_ne!(key(&a), key(&c));
    }

    #[test]
    fn rotation_and_shifts() {
        let r = req(json!({"state": "s", "questions": {"t": {"type": "choice", "instructions": "?", "criteria": {"a": null, "b": null, "c": null}}}}));
        let q = &compile(&r, 26).unwrap().questions[0];
        let v = q.rotated(1);
        assert_eq!(v.order, vec![1, 2, 0]);
        assert_eq!(v.options.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), ["b", "c", "a"]);
        assert_eq!(v.rotated(2).order, vec![0, 1, 2]);
        assert_ne!(v.fingerprint(), q.fingerprint());
        assert_eq!(Debias::Full.shifts(3, 4), vec![0, 1, 2]);
        assert_eq!(Debias::Full.shifts(10, 4), vec![0, 2, 5, 7]);
        assert_eq!(Debias::Calibrate.shifts(10, 4), vec![0]);
        assert_eq!(Debias::Auto.resolve(Debias::None), Debias::None);
        assert_eq!(Debias::Permute.resolve(Debias::None), Debias::Permute);
    }

    #[test]
    fn rejects_bad_questions() {
        let r = req(json!({"state": "", "questions": {"x": {"type": "choice", "instructions": "?", "criteria": {"only": null}}}}));
        assert!(compile(&r, 26).is_err());
    }
}
