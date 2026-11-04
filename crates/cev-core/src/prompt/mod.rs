//! Prompt contracts of the decision-model families cev can serve.
//!
//! A compiled request ([`crate::CompiledRequest`]) is structure; each family
//! renders it into the exact text its weights were trained on. Every renderer
//! produces a shared `prefix` (system turn, media placeholders and state) and
//! one `suffix` per question, so a backend can compute the state once and
//! fork a branch per question. Caller text is always passed through
//! [`crate::compile::sanitize`] first, so it can never forge a control token.
//!
//! | module | models | answer readout |
//! |---|---|---|
//! | [`cev`] | any Qwen3 / Qwen3.5 checkpoint (zero-shot, or fine-tuned on `/v1/export`) | LM logits of one-token codes `A`, `B`, ... |
//! | [`nimble`] | Bespoke Nimble (Qwen3.5 + LoRA) | LM logits of one-token codes over a JSON schema prompt |
//! | [`d3`] | vLLM Semantic Router d3 (Qwen3.5) | a separate 255-way readout matrix over the codes |
//! | [`kev`] | Kev (Qwen3.5-Base + LoRA) | pointer head: each option's closing token against the question's last token |
//! | [`d1`] | Liquid d1 (LFM2.5-VL) | LM logits of option codes / yes-no / digits, max-pooled over spellings |
//! | [`omni`] | Liquid d1-omni (LFM2.5 encoder) | a decision head over `<|mask|>` markers, one per option |

pub mod cev;
pub mod d1;
pub mod d3;
pub mod kev;
pub mod nimble;
pub mod omni;

pub use cev::PromptFormat;

/// Which family a backend speaks; reported in `/v1/models` and stored with
/// each decision so exports can be trained on with the right template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Family {
    /// cev's own headless code readout.
    Cev,
    Nimble,
    Kev,
    D3,
    D1,
    D1Omni,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Cev => "cev",
            Family::Nimble => "nimble",
            Family::Kev => "kev",
            Family::D3 => "d3",
            Family::D1 => "d1",
            Family::D1Omni => "d1-omni",
        }
    }

    /// Bump when a family's prompt text changes; adapters and calibrations are
    /// keyed to it.
    pub fn prompt_version(self) -> &'static str {
        match self {
            Family::Cev => cev::PROMPT_VERSION,
            Family::Nimble => nimble::PROMPT_VERSION,
            Family::Kev => kev::PROMPT_VERSION,
            Family::D3 => d3::PROMPT_VERSION,
            Family::D1 => d1::PROMPT_VERSION,
            Family::D1Omni => omni::PROMPT_VERSION,
        }
    }
}

/// `<|im_start|>` ... ChatML pieces shared by the Qwen-family templates.
pub(crate) const IM_START: &str = "<|im_start|>";
pub(crate) const IM_END: &str = "<|im_end|>";

/// The assistant turn opener of a Qwen3.5 (non-thinking) template, where the
/// answer code is the next token.
pub(crate) const QWEN35_ASSISTANT: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// JSON text with `<` and `>` escaped, as Python's `json.dumps(...).replace(...)`
/// does in the Nimble and d3 contracts: a state can never close a tag.
pub(crate) fn json_compact(v: &serde_json::Value) -> String {
    py_json(v).replace('<', "\\u003c").replace('>', "\\u003e")
}

/// `json.dumps(v, ensure_ascii=False)`: Python's default separators (`", "`,
/// `": "`), which differ from serde_json's compact form.
pub fn py_json(v: &serde_json::Value) -> String {
    use serde_json::Value::*;
    match v {
        Null => "null".into(),
        Bool(b) => b.to_string(),
        Number(n) => py_number(n),
        String(s) => serde_json::to_string(s).unwrap_or_default(),
        Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Object(m) => format!(
            "{{{}}}",
            m.iter().map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap_or_default(), py_json(v))).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// Python's `repr(float)` / `json.dumps` spelling of a number.
fn py_number(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    let f = n.as_f64().unwrap_or(0.0);
    if f.fract() == 0.0 && f.abs() < 1e16 {
        return format!("{f:.1}");
    }
    let s = n.to_string();
    // serde prints 1e-7 as "1e-7"; Python as "1e-07".
    if let Some((m, e)) = s.split_once('e') {
        let (sign, digits) = e.strip_prefix('-').map_or(("+", e), |d| ("-", d));
        return format!("{m}e{sign}{digits:0>2}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_spelling() {
        assert_eq!(py_json(&json!({"a": [1, 2.5, true, null], "b": "x<y"})), r#"{"a": [1, 2.5, true, null], "b": "x<y"}"#);
        assert_eq!(py_json(&json!(3.0)), "3.0");
        assert_eq!(py_json(&json!("é")), "\"é\"");
        assert_eq!(json_compact(&json!("<b>")), "\"\\u003cb\\u003e\"");
    }
}
