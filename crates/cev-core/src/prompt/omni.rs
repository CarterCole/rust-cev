//! Liquid d1-omni's record encoding: reserved tokens delimit the state, the
//! question and each option, a `<|mask|>` marker opens every option, and a
//! decision head scores the trunk's hidden state at each marker.
//!
//! ```text
//!   <bos> <|reserved_7|> state <|reserved_8|> instructions
//!         <|reserved_9|> <|mask|> option_0 <|reserved_10|> ... <|reserved_11|>
//! ```
//!
//! Option texts are budgeted in tokens (see [`budget`]), so the backend
//! assembles the final token sequence from the [`Parts`] rendered here.

use crate::api::QuestionType;
use crate::compile::{CompiledQuestion, CompiledRequest, sanitize};
use serde_json::Value;

pub const PROMPT_VERSION: &str = "d1-omni-v1";

pub const STATE: &str = "<|reserved_7|>";
pub const QUESTION: &str = "<|reserved_8|>";
pub const OPTION: &str = "<|reserved_9|>";
pub const OPTION_END: &str = "<|reserved_10|>";
pub const DECIDE: &str = "<|reserved_11|>";
pub const MARKER: &str = "<|mask|>";
pub const MAX_SCORE_LEVELS: usize = 10;

/// How the record is shaped for the media it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Text,
    Image,
    Audio,
}

/// `json.dumps(v, ensure_ascii=False)` for non-strings.
fn criterion(v: &str) -> String {
    sanitize(v)
}

/// The state as text: strings verbatim, JSON otherwise. With audio, a `null`
/// state is the empty object, as the audio questions were trained.
pub fn state_text(state: &Value, mode: Mode) -> String {
    match state {
        Value::Null if mode == Mode::Audio => "{}".into(),
        Value::Null => String::new(),
        Value::String(s) => sanitize(s),
        other => sanitize(&super::py_json(other)),
    }
}

/// Option texts in the model's order (a noul is `[false, true]`). After an
/// image or audio prefix the noul reads `false: no` / `true: yes`, and audio
/// choices are written `option_000: text`.
pub fn option_texts(q: &CompiledQuestion, mode: Mode) -> Vec<String> {
    match q.kind {
        QuestionType::Choice => q
            .options
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let text = match &o.description {
                    Some(d) if !d.is_empty() => format!("{}: {}", sanitize(&o.name), criterion(d)),
                    _ => sanitize(&o.name),
                };
                if mode == Mode::Audio {
                    let body = o.description.as_deref().filter(|d| !d.is_empty()).map(criterion).unwrap_or_else(|| sanitize(&o.name));
                    format!("option_{i:03}: {body}")
                } else {
                    text
                }
            })
            .collect(),
        QuestionType::Score => q.options.iter().map(|o| format!("level {}: {}", o.name, criterion(o.description.as_deref().unwrap_or("")))).collect(),
        QuestionType::Noul => q
            .options
            .iter()
            .map(|o| {
                let is_true = o.name == "true";
                let fallback = match (mode, is_true) {
                    (Mode::Text, false) => "no, the statement does not hold",
                    (Mode::Text, true) => "yes, the statement holds",
                    (_, false) => "no",
                    (_, true) => "yes",
                };
                let d = match mode {
                    Mode::Text => o.description.as_deref().filter(|d| !d.is_empty()).map(criterion).unwrap_or_else(|| fallback.into()),
                    _ => fallback.into(),
                };
                format!("{}: {d}", if is_true { "true" } else { "false" })
            })
            .collect(),
    }
}

/// Everything the backend needs to assemble one question's tokens.
#[derive(Debug, Clone)]
pub struct Parts {
    pub state: String,
    pub instructions: String,
    pub options: Vec<String>,
}

pub fn parts(req: &CompiledRequest, q: &CompiledQuestion, mode: Mode) -> Parts {
    Parts { state: state_text(&req.state, mode), instructions: sanitize(q.question.instructions.trim()), options: option_texts(q, mode) }
}

/// Token budget of the option block for `k` options within `max_len`
/// positions: `(block, per_option)`, as `prompt.encode` shares it out.
pub fn budget(k: usize, max_len: usize, per_option: usize) -> (usize, usize) {
    let block = 96.max((k * per_option + 32).min(max_len / 2));
    let per = 2.max((block.saturating_sub(3 * k)) / k.max(1));
    (block, per)
}

/// Human-readable form of the record (what is stored and exported); the
/// token-level truncation happens in the backend.
pub fn prefix(req: &CompiledRequest, mode: Mode) -> String {
    format!("{STATE}{}", state_text(&req.state, mode))
}

pub fn suffix(p: &Parts) -> String {
    let mut s = format!("{QUESTION}{}", p.instructions);
    for o in &p.options {
        s.push_str(&format!("{OPTION}{MARKER} {o}{OPTION_END}"));
    }
    s.push_str(DECIDE);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::compile;
    use serde_json::json;

    #[test]
    fn renders_options_per_mode() {
        let r = serde_json::from_value(json!({
            "state": {"a": 1},
            "questions": {
                "team": {"type": "choice", "instructions": "Team?", "criteria": {"returns": "Exchanges", "billing": null}},
                "refund": {"type": "noul", "instructions": "Refund?"},
                "mood": {"type": "score", "instructions": "Mood?", "criteria": ["Calm", "Angry"]}
            }
        }))
        .unwrap();
        let c = compile(&r, 255).unwrap();
        assert_eq!(option_texts(&c.questions[0], Mode::Text), ["returns: Exchanges", "billing"]);
        assert_eq!(option_texts(&c.questions[0], Mode::Audio), ["option_000: Exchanges", "option_001: billing"]);
        assert_eq!(option_texts(&c.questions[1], Mode::Text), ["false: no, the statement does not hold", "true: yes, the statement holds"]);
        assert_eq!(option_texts(&c.questions[1], Mode::Image), ["false: no", "true: yes"]);
        assert_eq!(option_texts(&c.questions[2], Mode::Text), ["level 0: Calm", "level 1: Angry"]);
        assert_eq!(state_text(&c.state, Mode::Text), "{\"a\": 1}");
        assert_eq!(state_text(&Value::Null, Mode::Audio), "{}");
        assert_eq!(budget(2, 16384, 24), (96, 45));
        assert_eq!(budget(255, 16384, 24), (6152, 21));
        let p = parts(&c, &c.questions[0], Mode::Text);
        assert_eq!(suffix(&p), "<|reserved_8|>Team?<|reserved_9|><|mask|> returns: Exchanges<|reserved_10|><|reserved_9|><|mask|> billing<|reserved_10|><|reserved_11|>");
    }
}
