//! Liquid d1's prompt (LFM2.5-VL): no system turn, the state (JSON
//! pretty-printed) then `QUESTION:`, and a question block whose answer is read
//! from the LM's next-token distribution:
//!
//! * choice: one-token codes (the labels themselves when they are single
//!   letters, else `A`, `B`, ..., else `00`, `01`, ...), scored as the best of
//!   `code` and ` code`;
//! * noul: the best of `yes`/`Yes`/`YES` against `no`/`No`/`NO`;
//! * score: the digits `0`..`9` (2 to 10 levels).

use super::{IM_END, IM_START};
use crate::api::QuestionType;
use crate::compile::{CompiledQuestion, CompiledRequest, sanitize};
use serde_json::Value;

pub const PROMPT_VERSION: &str = "d1-prompt-v1";

/// What the chat template writes for one image; the backend expands it into
/// the tile layout the processor would.
pub const IMAGE_PLACEHOLDER: &str = "<image>";
pub const MAX_SCORE_LEVELS: usize = 10;

/// The block that precedes `QUESTION:` (the `json_only` style).
pub fn state_block(state: &Value) -> String {
    match state {
        Value::String(s) => format!("{}\n\n", sanitize(s)),
        other => format!("{}\n\n", sanitize(&serde_json::to_string_pretty(other).unwrap_or_default())),
    }
}

/// Shared prefix. `bos` is the tokenizer's beginning-of-text token. A `null`
/// state with images means the pictures are the whole state.
pub fn prefix(req: &CompiledRequest, bos: &str) -> String {
    let images: String = req.media.images.iter().map(|_| IMAGE_PLACEHOLDER).collect();
    let body = match &req.state {
        Value::Null if !req.media.images.is_empty() => String::new(),
        state => format!("{}\nQUESTION:\n", state_block(state)),
    };
    format!("{bos}{IM_START}user\n{images}{body}")
}

/// The codes d1 shows for a choice question's labels: the labels when they
/// are single letters, `A`.. up to 26, else `00`..
pub fn option_codes(labels: &[&str]) -> Vec<String> {
    let labs: Vec<&str> = labels.iter().map(|l| l.trim()).collect();
    if !labs.is_empty() && labs.iter().all(|k| k.chars().count() == 1 && k.chars().all(char::is_alphabetic)) {
        return labs.iter().map(|s| s.to_string()).collect();
    }
    if labs.len() <= 26 {
        return (0..labs.len()).map(|i| ((b'A' + i as u8) as char).to_string()).collect();
    }
    (0..labs.len()).map(|i| format!("{i:02}")).collect()
}

/// Codes tried when a label's own code is not one token: letters, digit pairs,
/// lower-case letters, `#i`, then letter pairs.
pub fn fallback_pool() -> Vec<String> {
    let mut v: Vec<String> = (b'A'..=b'Z').map(|c| (c as char).to_string()).collect();
    v.extend((0..100).map(|i| format!("{i:02}")));
    v.extend((b'a'..=b'z').map(|c| (c as char).to_string()));
    v.extend((0..200).map(|i| format!("#{i}")));
    v.extend((b'A'..=b'Z').flat_map(|a| (b'A'..=b'Z').map(move |b| format!("{}{}", a as char, b as char))));
    v
}

pub const YES_FORMS: [&str; 3] = ["yes", "Yes", "YES"];
pub const NO_FORMS: [&str; 3] = ["no", "No", "NO"];

fn option_line(code: &str, label: &str, description: Option<&str>) -> String {
    let text = description.map(str::to_string).unwrap_or_else(|| label.replace('_', " "));
    format!("{code} {}", sanitize(&text))
}

/// The question block, up to the answer slot. `codes` label a choice's
/// options in display order (ignored for noul and score).
pub fn question_block(q: &CompiledQuestion, codes: &[String]) -> String {
    let instructions = sanitize(q.question.instructions.trim());
    match q.kind {
        QuestionType::Choice => {
            let lines: Vec<String> = q.options.iter().zip(codes).map(|(o, c)| option_line(c, &o.name, o.description.as_deref())).collect();
            format!("{instructions}\n\nOptions:\n{}\n\nReply with the option code only.", lines.join("\n"))
        }
        QuestionType::Noul => {
            let described = q.options.iter().any(|o| o.description.is_some());
            let extra = if described {
                let get = |name: &str| q.options.iter().find(|o| o.name == name).and_then(|o| o.description.clone()).map(|d| sanitize(&d)).unwrap_or_else(|| "None".into());
                format!("\nYes: {}\nNo: {}", get("true"), get("false"))
            } else {
                String::new()
            };
            format!("{instructions}{extra}\n\nReply with yes or no only.")
        }
        QuestionType::Score => {
            // Levels are shown in canonical order whatever the display order:
            // the digit *is* the level.
            let mut levels: Vec<(usize, String)> = q.options.iter().map(|o| (o.name.parse().unwrap_or(0), o.description.clone().unwrap_or_default())).collect();
            levels.sort_by_key(|(i, _)| *i);
            let legend: Vec<String> = levels.iter().map(|(i, d)| format!("{i} {}", sanitize(d))).collect();
            format!("{instructions}\n\n{}\n\nReply with a single digit 0-{} only.", legend.join("\n"), levels.len() - 1)
        }
    }
}

pub fn suffix(q: &CompiledQuestion, codes: &[String]) -> String {
    format!("{}{IM_END}\n{IM_START}assistant\n", question_block(q, codes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::compile;
    use serde_json::json;

    #[test]
    fn renders_like_prompt_py() {
        let r = serde_json::from_value(json!({
            "state": {"text": "I was charged twice"},
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"returns": "Exchanges, refunds", "billing": null}},
                "refund": {"type": "noul", "instructions": "Is this a refund request?"},
                "escalate": {"type": "noul", "instructions": "Escalate?", "criteria": {"true": "angry", "false": "calm"}},
                "mood": {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Frustrated", "Very angry"]}
            }
        }))
        .unwrap();
        let c = compile(&r, 255).unwrap();
        assert_eq!(prefix(&c, "<|startoftext|>"), "<|startoftext|><|im_start|>user\n{\n  \"text\": \"I was charged twice\"\n}\n\n\nQUESTION:\n");
        let codes: Vec<String> = ["A", "B"].iter().map(|s| s.to_string()).collect();
        assert_eq!(suffix(&c.questions[0], &codes), "Which team?\n\nOptions:\nA Exchanges, refunds\nB billing\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(question_block(&c.questions[1], &codes), "Is this a refund request?\n\nReply with yes or no only.");
        assert_eq!(question_block(&c.questions[2], &codes), "Escalate?\nYes: angry\nNo: calm\n\nReply with yes or no only.");
        assert_eq!(question_block(&c.questions[3], &codes), "How frustrated?\n\n0 Calm\n1 Frustrated\n2 Very angry\n\nReply with a single digit 0-2 only.");
        assert_eq!(option_codes(&["a", "b"]), ["a", "b"]);
        assert_eq!(option_codes(&["billing", "tech"]), ["A", "B"]);
        assert_eq!(option_codes(&vec!["x_"; 27])[26], "26");
        let pics = compile(&serde_json::from_value(json!({"state": null, "images": ["AAAA", "AAAA"], "questions": {"q": {"type": "noul", "instructions": "cat?"}}})).unwrap(), 255).unwrap();
        assert_eq!(prefix(&pics, "<|startoftext|>"), "<|startoftext|><|im_start|>user\n<image><image>");
    }
}
