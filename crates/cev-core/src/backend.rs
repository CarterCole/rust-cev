//! The seam between request handling and the neural network.

use crate::compile::{CompiledRequest, PromptFormat};

/// What the network produced for one question.
#[derive(Debug, Clone)]
pub struct Readout {
    /// Raw logits of the question's answer codes, in option order.
    pub logits: Vec<f32>,
    /// Final hidden state at the answer position (the adapter's features).
    pub hidden: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BackendOutput {
    pub readouts: Vec<Readout>,
    pub input_tokens: usize,
    /// Whether the state prefix came from the prefix cache.
    pub prefix_cached: bool,
}

pub trait Backend: Send + Sync {
    /// Model id reported in responses, e.g. `cev-qwen3-1.7b`.
    fn id(&self) -> &str;
    /// Backbone weights identifier (HF repo or path).
    fn backbone(&self) -> &str;
    fn hidden_size(&self) -> usize;
    /// Ordered code vocabulary; see [`crate::compile::compile`].
    fn codes(&self) -> &[String];
    fn format(&self) -> &PromptFormat;
    fn run(&self, req: &CompiledRequest) -> anyhow::Result<BackendOutput>;
    /// Several requests at once; backends may fuse them into one pass.
    fn run_many(&self, reqs: &[CompiledRequest]) -> anyhow::Result<Vec<BackendOutput>> {
        reqs.iter().map(|r| self.run(r)).collect()
    }
}

/// Deterministic stand-in for tests and wiring: logits come from keyword
/// overlap between the evidence and each option line, features from hashed
/// character trigrams of the full prompt.
pub struct MockBackend {
    codes: Vec<String>,
    format: PromptFormat,
    pub dim: usize,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new(64)
    }
}

impl MockBackend {
    pub fn new(dim: usize) -> Self {
        Self {
            codes: (b'A'..=b'Z').map(|c| (c as char).to_string()).collect(),
            format: PromptFormat { think_stub: false },
            dim,
        }
    }

    fn features(&self, text: &str) -> Vec<f32> {
        let mut h = vec![0f32; self.dim];
        let b = text.to_lowercase().into_bytes();
        for w in b.windows(3) {
            let x = w.iter().fold(2166136261u32, |a, &c| (a ^ c as u32).wrapping_mul(16777619));
            h[x as usize % self.dim] += if x & 1 == 0 { 1.0 } else { -1.0 };
        }
        let n = h.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        h.iter().map(|v| v / n * (self.dim as f32).sqrt()).collect()
    }
}

impl Backend for MockBackend {
    fn id(&self) -> &str {
        "cev-mock"
    }
    fn backbone(&self) -> &str {
        "mock"
    }
    fn hidden_size(&self) -> usize {
        self.dim
    }
    fn codes(&self) -> &[String] {
        &self.codes
    }
    fn format(&self) -> &PromptFormat {
        &self.format
    }
    fn run(&self, req: &CompiledRequest) -> anyhow::Result<BackendOutput> {
        let evidence = req.prefix.to_lowercase();
        let readouts = req
            .questions
            .iter()
            .map(|q| {
                let logits = q
                    .options
                    .iter()
                    .map(|o| {
                        o.text
                            .to_lowercase()
                            .split(|c: char| !c.is_alphanumeric())
                            .filter(|w| w.len() > 2 && evidence.contains(w))
                            .count() as f32
                    })
                    .collect();
                Readout { logits, hidden: self.features(&format!("{}{}", req.prefix, q.suffix)) }
            })
            .collect();
        Ok(BackendOutput { readouts, input_tokens: evidence.len() / 4, prefix_cached: false })
    }
}
