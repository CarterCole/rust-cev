//! Kev's record encoding: Qwen special tokens delimit the state, each
//! question, each option and the decision slot; a pointer head scores every
//! option's closing token against the decision token.
//!
//! ```text
//!   <|fim_prefix|> state
//!   <|fim_middle|> instructions <|box_start|> option <|box_end|> ... <|fim_suffix|>
//! ```
//!
//! Special tokens are tokenizer boundaries, so rendering this as text and
//! tokenizing it gives the token-level layout `kev.model.encode` builds.
//! Caller text goes through [`sanitize`], the same `<|x|>` -> `<¦x¦>` rewrite
//! Kev applies, so it can never contain a delimiter.

use crate::api::QuestionType;
use crate::compile::{CompiledQuestion, CompiledRequest, sanitize};
use serde_json::Value;

pub const PROMPT_VERSION: &str = "kev-record-v1";

pub const STATE: &str = "<|fim_prefix|>";
pub const QUESTION: &str = "<|fim_middle|>";
pub const OPTION: &str = "<|box_start|>";
pub const OPTION_END: &str = "<|box_end|>";
pub const DECIDE: &str = "<|fim_suffix|>";

/// Kev's `render`: strings as they are, objects as `key: value` lines, arrays
/// as `- item` lines, nested with two-space indents.
pub fn render(v: &Value) -> String {
    render_at(v, 0)
}

fn scalar(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => super::py_json(&Value::Number(n.clone())),
        Value::String(s) => s.clone(),
        other => render_at(other, 0),
    }
}

fn render_at(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Array(a) => a.iter().map(|x| format!("{pad}- {}", render_at(x, indent + 1).trim_start())).collect::<Vec<_>>().join("\n"),
        Value::Object(m) => m
            .iter()
            .map(|(k, x)| match x {
                Value::Array(_) | Value::Object(_) => format!("{pad}{k}:\n{}", render_at(x, indent + 1)),
                _ => format!("{pad}{k}: {}", scalar(x)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => scalar(other),
    }
}

/// `name` or `name: description`.
fn option_text(name: &str, description: Option<&str>) -> String {
    match description {
        Some(d) if !d.is_empty() => format!("{name}: {d}"),
        _ => name.to_string(),
    }
}

/// The text of each option, in display order.
pub fn option_texts(q: &CompiledQuestion) -> Vec<String> {
    q.options
        .iter()
        .map(|o| match q.kind {
            QuestionType::Noul => option_text(if o.name == "true" { "yes" } else { "no" }, o.description.as_deref()),
            QuestionType::Choice => option_text(&o.name, o.description.as_deref()),
            QuestionType::Score => o.description.clone().unwrap_or_default(),
        })
        .map(|t| sanitize(&t))
        .collect()
}

pub fn prefix(req: &CompiledRequest) -> String {
    format!("{STATE}{}", sanitize(&render(&req.state)))
}

pub fn suffix(q: &CompiledQuestion) -> String {
    let mut s = format!("{QUESTION}{}", sanitize(q.question.instructions.trim()));
    for o in option_texts(q) {
        s.push_str(OPTION);
        s.push_str(&o);
        s.push_str(OPTION_END);
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
    fn renders_state_like_kev() {
        let v = json!({"subject": "Late", "items": [{"sku": "A1", "qty": 2}, "note"], "ok": true, "n": null, "price": 2.5});
        assert_eq!(render(&v), "subject: Late\nitems:\n  - sku: A1\n    qty: 2\n  - note\nok: True\nn: \nprice: 2.5");
        assert_eq!(render(&json!("plain <|fim_prefix|>")), "plain <|fim_prefix|>");
    }

    #[test]
    fn renders_record() {
        let r = serde_json::from_value(json!({
            "state": "Shoes late <|fim_suffix|>",
            "questions": {
                "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"returns": "Exchanges", "billing": null}},
                "urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "needs a human"}},
                "mood": {"type": "score", "instructions": "How angry?", "criteria": ["Calm", "Angry"]}
            }
        }))
        .unwrap();
        let c = compile(&r, 255).unwrap();
        assert_eq!(prefix(&c), "<|fim_prefix|>Shoes late <\u{a6}fim_suffix\u{a6}>");
        assert_eq!(suffix(&c.questions[0]), "<|fim_middle|>Which team?<|box_start|>returns: Exchanges<|box_end|><|box_start|>billing<|box_end|><|fim_suffix|>");
        assert_eq!(suffix(&c.questions[1]), "<|fim_middle|>Urgent?<|box_start|>no<|box_end|><|box_start|>yes: needs a human<|box_end|><|fim_suffix|>");
        assert_eq!(suffix(&c.questions[2]), "<|fim_middle|>How angry?<|box_start|>Calm<|box_end|><|box_start|>Angry<|box_end|><|fim_suffix|>");
    }
}
