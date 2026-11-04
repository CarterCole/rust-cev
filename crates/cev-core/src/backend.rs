//! The seam between request handling and the neural network.

use crate::compile::{CompiledQuestion, CompiledRequest, Debias};
use crate::prompt::{Family, PromptFormat, cev};

/// What the network produced for one question.
#[derive(Debug, Clone)]
pub struct Readout {
    /// Raw logits of the question's options, in display order.
    pub logits: Vec<f32>,
    /// Final hidden state at the answer position (the adapter's features).
    pub hidden: Vec<f32>,
    /// Prompt text of this question after the shared prefix: what the model saw.
    pub suffix: String,
    /// Answer codes, one per option in display order (empty when the family
    /// reads options without codes).
    pub codes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BackendOutput {
    pub readouts: Vec<Readout>,
    pub input_tokens: usize,
    /// Whether the state prefix came from the prefix cache.
    pub prefix_cached: bool,
    /// Shared prompt prefix of the request (system turn, media placeholders,
    /// state); full prompt of question `i` = `prefix + readouts[i].suffix`.
    pub prefix: String,
}

/// What a backend can take and how it should be used by default.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub family: Family,
    pub images: bool,
    pub audio: bool,
    /// Options per question the readout can score.
    pub max_options: usize,
    /// Position-bias correction applied when a request does not say.
    pub debias: Debias,
}

impl Capabilities {
    pub fn prompt_version(&self) -> &'static str {
        self.family.prompt_version()
    }

    pub fn modalities(&self) -> Vec<&'static str> {
        let mut v = vec!["text"];
        if self.images {
            v.push("image");
        }
        if self.audio {
            v.push("audio");
        }
        v
    }
}

pub trait Backend: Send + Sync {
    /// Model id reported in responses, e.g. `cev-qwen3-1.7b`.
    fn id(&self) -> &str;
    /// Backbone weights identifier (HF repo or path).
    fn backbone(&self) -> &str;
    fn hidden_size(&self) -> usize;
    fn capabilities(&self) -> Capabilities;
    fn run(&self, req: &CompiledRequest) -> anyhow::Result<BackendOutput>;
    /// Several requests at once; backends may fuse them into one pass.
    fn run_many(&self, reqs: &[CompiledRequest]) -> anyhow::Result<Vec<BackendOutput>> {
        reqs.iter().map(|r| self.run(r)).collect()
    }
}

/// Deterministic stand-in for tests and wiring: logits come from keyword
/// overlap between the evidence and each option line, features from hashed
/// character trigrams of the full prompt. Speaks cev's own prompt.
pub struct MockBackend {
    codes: Vec<String>,
    format: PromptFormat,
    pub dim: usize,
    /// Pretend to accept images and audio (they only count toward tokens).
    pub media: bool,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new(64)
    }
}

impl MockBackend {
    pub fn new(dim: usize) -> Self {
        Self { codes: cev::candidate_codes().take(26).collect(), format: PromptFormat { think_stub: false }, dim, media: false }
    }

    pub fn with_media(mut self) -> Self {
        self.media = true;
        self
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

    fn score(evidence: &str, q: &CompiledQuestion) -> Vec<f32> {
        q.options
            .iter()
            .map(|o| o.text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2 && evidence.contains(w)).count() as f32)
            .collect()
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
    fn capabilities(&self) -> Capabilities {
        Capabilities { family: Family::Cev, images: self.media, audio: self.media, max_options: self.codes.len(), debias: Debias::Full }
    }
    fn run(&self, req: &CompiledRequest) -> anyhow::Result<BackendOutput> {
        let prefix = self.format.request_prefix(req);
        let evidence = req.evidence.to_lowercase();
        let readouts = req
            .questions
            .iter()
            .map(|q| {
                let suffix = self.format.question_suffix(q, &self.codes);
                Readout {
                    logits: Self::score(&evidence, q),
                    hidden: self.features(&format!("{prefix}{suffix}")),
                    codes: self.codes[..q.options.len()].to_vec(),
                    suffix,
                }
            })
            .collect();
        let media_tokens = req.media.images.len() * 256 + req.media.audio.as_ref().map_or(0, |a| a.len() / 2560);
        Ok(BackendOutput { readouts, input_tokens: evidence.len() / 4 + media_tokens, prefix_cached: false, prefix })
    }
}
