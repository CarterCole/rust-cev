//! Converts feedback labels into target distributions over a question's options.

use crate::api::{Label, QuestionType};
use crate::compile::OptionSpec;

pub fn target(kind: QuestionType, options: &[OptionSpec], label: &Label) -> Result<Vec<f64>, String> {
    let k = options.len();
    let one_hot = |i: usize| {
        let mut y = vec![0.0; k];
        y[i] = 1.0;
        y
    };
    let by_name = |s: &str| {
        options
            .iter()
            .position(|o| o.name == s)
            .or_else(|| options.iter().position(|o| o.name.eq_ignore_ascii_case(s)))
    };
    let y = match (kind, label) {
        (QuestionType::Noul, Label::Bool(b)) => one_hot(*b as usize),
        (QuestionType::Noul, Label::Number(p)) if (0.0..=1.0).contains(p) => vec![1.0 - p, *p],
        (QuestionType::Noul, Label::Text(s)) => match s.to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" => one_hot(1),
            "false" | "no" | "0" => one_hot(0),
            _ => return Err(format!("noul label must be true/false, got `{s}`")),
        },
        (QuestionType::Score, Label::Number(v)) => {
            let max = (k - 1) as f64;
            if !(0.0..=max).contains(v) {
                return Err(format!("score label must be within 0..={max}"));
            }
            // Split fractional levels so the target's expectation equals `v`.
            let (lo, frac) = (v.floor() as usize, v - v.floor());
            let mut y = one_hot(lo);
            if frac > 0.0 {
                y[lo] = 1.0 - frac;
                y[lo + 1] = frac;
            }
            y
        }
        (QuestionType::Score, Label::Text(s)) => match s.parse::<f64>() {
            Ok(v) => return target(kind, options, &Label::Number(v)),
            Err(_) => one_hot(by_name(s).ok_or_else(|| format!("unknown level `{s}`"))?),
        },
        (QuestionType::Choice, Label::Text(s)) => {
            one_hot(by_name(s).ok_or_else(|| format!("unknown option `{s}`"))?)
        }
        (_, Label::Distribution(m)) => {
            let mut y = vec![0.0; k];
            for (name, p) in m {
                let i = by_name(name).ok_or_else(|| format!("unknown option `{name}`"))?;
                if !(p.is_finite() && *p >= 0.0) {
                    return Err("probabilities must be non-negative".into());
                }
                y[i] = *p;
            }
            let s: f64 = y.iter().sum();
            if s <= 0.0 {
                return Err("distribution sums to zero".into());
            }
            y.iter().map(|v| v / s).collect()
        }
        (kind, l) => return Err(format!("label {l:?} does not fit a {} question", kind.as_str())),
    };
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(n: &[&str]) -> Vec<OptionSpec> {
        n.iter().map(|s| OptionSpec { name: s.to_string(), text: s.to_string() }).collect()
    }

    #[test]
    fn labels() {
        let noul = opts(&["false", "true"]);
        assert_eq!(target(QuestionType::Noul, &noul, &Label::Bool(true)).unwrap(), vec![0.0, 1.0]);
        assert_eq!(target(QuestionType::Noul, &noul, &Label::Text("No".into())).unwrap(), vec![1.0, 0.0]);
        let lv = opts(&["0", "1", "2"]);
        assert_eq!(target(QuestionType::Score, &lv, &Label::Number(1.25)).unwrap(), vec![0.0, 0.75, 0.25]);
        assert!(target(QuestionType::Score, &lv, &Label::Number(3.0)).is_err());
        let ch = opts(&["billing", "tech"]);
        assert_eq!(target(QuestionType::Choice, &ch, &Label::Text("tech".into())).unwrap(), vec![0.0, 1.0]);
        assert!(target(QuestionType::Choice, &ch, &Label::Bool(true)).is_err());
    }
}
