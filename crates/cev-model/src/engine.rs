//! `Backend` implementation over the Qwen3 decoder.

#[cfg(not(target_arch = "wasm32"))]
use crate::hub;
use crate::qwen3::{Config, LayerKv, PrefixKv, Qwen3, Tensors};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cev_core::{Backend, BackendOutput, CompiledRequest, PromptFormat, Readout};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// Directory or Hugging Face repo id.
    pub model: String,
    /// `auto`, `cpu`, `metal` or `cuda`.
    pub device: String,
    /// `auto`, `f32`, `f16` or `bf16`.
    pub dtype: String,
    pub max_context: usize,
    pub prefill_chunk: usize,
    /// Most new tokens per packed forward pass (bounds the attention mask).
    pub pass_tokens: usize,
    /// Token budget for cached state prefixes.
    pub prefix_cache_tokens: usize,
    /// Name reported in responses; defaults to `cev-<model name>`.
    pub name: Option<String>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            model: "Qwen/Qwen3-1.7B".into(),
            device: "auto".into(),
            dtype: "auto".into(),
            max_context: 32768,
            prefill_chunk: 512,
            pass_tokens: 2048,
            prefix_cache_tokens: 65536,
            name: None,
        }
    }
}

/// Present exactly in int8 checkpoints.
const QUANT_MARKER: &str = "model.embed_tokens.weight.scale";

pub struct Engine {
    model: Qwen3,
    tokenizer: Tokenizer,
    codes: Vec<String>,
    /// LM-head rows of the code tokens, (n_codes, hidden), f32.
    code_head: Tensor,
    format: PromptFormat,
    name: String,
    backbone: String,
    opts: EngineOptions,
    cache: Mutex<PrefixCache>,
}

#[cfg(not(target_arch = "wasm32"))]
fn pick_device(s: &str) -> Result<Device> {
    Ok(match s {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        "cuda" => Device::new_cuda(0)?,
        "auto" => {
            if candle_core::utils::metal_is_available() {
                Device::new_metal(0)?
            } else if candle_core::utils::cuda_is_available() {
                Device::new_cuda(0)?
            } else {
                Device::Cpu
            }
        }
        other => bail!("unknown device `{other}`"),
    })
}

impl Engine {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(opts: EngineOptions) -> Result<Self> {
        let files = hub::resolve(&opts.model)?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer).map_err(anyhow::Error::msg)?;
        let tc = ["tokenizer_config.json", "chat_template.jinja"]
            .iter()
            .map(|f| std::fs::read_to_string(files.dir.join(f)).unwrap_or_default())
            .collect::<String>();
        let config = std::fs::read(&files.config)?;
        let names = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files.weights)? };
        if names.tensors().iter().any(|(name, _)| name == QUANT_MARKER) {
            // int8 checkpoint (examples/quantize.rs): CPU only.
            tracing::info!(dir = %files.dir.display(), "loading int8 weights (cpu)");
            let mut tensors = Tensors::new();
            for w in &files.weights {
                tensors.extend(candle_core::safetensors::load(w, &Device::Cpu)?);
            }
            let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu);
            return Self::build(&config, tokenizer, &tc, vb, Some(&tensors), opts);
        }
        let device = pick_device(&opts.device)?;
        // f32 is exact (matches transformers) and on Metal costs little for
        // small models; bigger checkpoints default to bf16 to halve memory.
        let weight_bytes: u64 = files.weights.iter().filter_map(|w| std::fs::metadata(w).ok()).map(|m| m.len()).sum();
        let dtype = match opts.dtype.as_str() {
            "auto" if device.is_cuda() => DType::BF16,
            "auto" if device.is_metal() && weight_bytes > 4 << 30 => DType::BF16,
            "auto" | "f32" => DType::F32,
            "bf16" => DType::BF16,
            "f16" => DType::F16,
            other => bail!("unknown dtype `{other}`"),
        };
        tracing::info!(dir = %files.dir.display(), ?device, ?dtype, "loading weights");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)? };
        let engine = Self::build(&config, tokenizer, &tc, vb, None, opts)?;
        engine.warmup()?;
        Ok(engine)
    }

    /// Build from files already in memory (the browser has no filesystem):
    /// `config.json`, `tokenizer.json`, the text of `tokenizer_config.json`
    /// (may be empty), and weights (f32, or an int8 checkpoint), e.g. from
    /// [`crate::weights::StreamLoader`]. Runs on the CPU.
    pub fn from_parts(config: &[u8], tokenizer: &[u8], tokenizer_config: &str, tensors: HashMap<String, Tensor>, opts: EngineOptions) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(tokenizer).map_err(anyhow::Error::msg)?;
        let quant = tensors.contains_key(QUANT_MARKER).then(|| tensors.clone());
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        Self::build(config, tokenizer, tokenizer_config, vb, quant.as_ref(), opts)
    }

    fn build(config: &[u8], tokenizer: Tokenizer, tokenizer_config: &str, vb: VarBuilder, quant: Option<&Tensors>, opts: EngineOptions) -> Result<Self> {
        let cfg: Config = serde_json::from_slice(config).context("config.json")?;
        let raw: serde_json::Value = serde_json::from_slice(config)?;
        let arch = raw["model_type"].as_str().unwrap_or_default();
        if arch != "qwen3" {
            bail!("unsupported model_type `{arch}` (cev currently supports qwen3)");
        }
        if let Some(t) = cfg.rope_parameters.as_ref().and_then(|r| r.rope_type.as_deref()).filter(|t| *t != "default") {
            bail!("unsupported rope_type `{t}`");
        }
        let model = Qwen3::new(cfg, vb, quant, opts.max_context)?;

        // Hybrid-thinking checkpoints get an empty <think></think> block; the
        // `-Instruct-2507` style checkpoints have no thinking mode.
        let think_stub = tokenizer_config.contains("enable_thinking")
            || (tokenizer.token_to_id("<think>").is_some() && !opts.model.contains("Instruct-2507"));
        let format = PromptFormat { think_stub };
        let (codes, code_ids) = code_vocab(&tokenizer, &format)?;
        let code_head = model.head_rows(&code_ids)?.to_dtype(DType::F32)?;
        let base = opts.model.trim_end_matches('/').rsplit('/').next().unwrap_or("model").to_lowercase();
        Ok(Self {
            name: opts.name.clone().unwrap_or_else(|| format!("cev-{base}")),
            backbone: opts.model.clone(),
            cache: Mutex::new(PrefixCache::new(opts.prefix_cache_tokens)),
            model,
            tokenizer,
            codes,
            code_head,
            format,
            opts,
        })
    }

    /// Compile GPU kernels up front so the first real request is not slow.
    #[cfg(not(target_arch = "wasm32"))]
    fn warmup(&self) -> Result<()> {
        let t = web_time::Instant::now();
        let req = serde_json::from_value(serde_json::json!({
            "state": "warmup",
            "questions": {"a": {"type": "noul", "instructions": "ok?"}, "b": {"type": "score", "instructions": "level?", "criteria": ["a", "b", "c"]}}
        }))?;
        self.run(&cev_core::compile(&req, &self.codes, &self.format)?)?;
        self.cache.lock().entries.clear();
        tracing::info!(ms = t.elapsed().as_millis() as u64, "warmed up");
        Ok(())
    }

    fn encode(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(s, false).map_err(anyhow::Error::msg)?.get_ids().to_vec())
    }
}

/// Codes the model answers with: A..Z, then two-letter codes, keeping only
/// those that are one token and tokenize identically after the answer prompt.
fn code_vocab(tok: &Tokenizer, fmt: &PromptFormat) -> Result<(Vec<String>, Vec<u32>)> {
    let enc = |s: &str| -> Result<Vec<u32>> {
        Ok(tok.encode(s, false).map_err(anyhow::Error::msg)?.get_ids().to_vec())
    };
    let probe = cev_core::compile(
        &serde_json::from_value(serde_json::json!({
            "state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}
        }))?,
        &["A".into(), "B".into()],
        fmt,
    )?
    .full_prompt(0);
    let base = enc(&probe)?;
    let letters = (b'A'..=b'Z').map(|c| (c as char).to_string());
    let pairs = (b'A'..=b'Z').flat_map(|a| (b'A'..=b'Z').map(move |b| format!("{}{}", a as char, b as char)));
    let (mut codes, mut ids) = (Vec::new(), Vec::new());
    for code in letters.chain(pairs) {
        if codes.len() == cev_core::compile::MAX_OPTIONS {
            break;
        }
        let one = enc(&code)?;
        if one.len() != 1 || ids.contains(&one[0]) {
            continue;
        }
        let joined = enc(&format!("{probe}{code}"))?;
        if joined.len() == base.len() + 1 && joined[..base.len()] == base[..] && joined[base.len()] == one[0] {
            codes.push(code);
            ids.push(one[0]);
        }
    }
    if codes.len() < 26 {
        bail!("tokenizer only supports {} single-token answer codes", codes.len());
    }
    Ok((codes, ids))
}

fn lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl Backend for Engine {
    fn id(&self) -> &str {
        &self.name
    }
    fn backbone(&self) -> &str {
        &self.backbone
    }
    fn hidden_size(&self) -> usize {
        self.model.cfg.hidden_size
    }
    fn codes(&self) -> &[String] {
        &self.codes
    }
    fn format(&self) -> &PromptFormat {
        &self.format
    }

    fn run(&self, req: &CompiledRequest) -> Result<BackendOutput> {
        Ok(self.run_many(std::slice::from_ref(req))?.remove(0))
    }

    /// Answer several requests (e.g. a request and its content-free
    /// calibration twin) in as few forward passes as possible: usually one.
    fn run_many(&self, reqs: &[CompiledRequest]) -> Result<Vec<BackendOutput>> {
        let t0 = web_time::Instant::now();
        let mut plans = reqs.iter().map(|r| self.plan(r)).collect::<Result<Vec<_>>>()?;
        let t_tok = t0.elapsed();

        // Prefill everything but the last chunk of uncached prefixes; the tail
        // rides along in the packed pass with the questions.
        for p in plans.iter_mut().filter(|p| p.cached.is_none()) {
            let tail = p.prefix.len().min(self.opts.prefill_chunk.max(1));
            let head = p.prefix.len() - tail;
            if head > 0 {
                p.past = Some(Arc::new(self.model.prefill(&p.prefix[..head], self.opts.prefill_chunk)?));
            }
            p.tail = head..p.prefix.len();
        }
        let t_prefill = t0.elapsed();

        // Work items: (plan, question), packed greedily under a token budget.
        let mut items: Vec<(usize, usize)> = Vec::new();
        for (pi, p) in plans.iter().enumerate() {
            items.extend((0..p.suffixes.len()).map(|qi| (pi, qi)));
        }
        let mut readouts: Vec<Vec<Option<Readout>>> = plans.iter().map(|p| vec![None; p.suffixes.len()]).collect();
        let mut passes = 0;
        let mut rest = &items[..];
        while !rest.is_empty() {
            let mut budget = self.opts.pass_tokens;
            // Tails still pending go first and count against the budget.
            for p in &plans {
                budget = budget.saturating_sub(p.tail.len());
            }
            let mut n = 0;
            while n < rest.len() {
                let (pi, qi) = rest[n];
                let len = plans[pi].suffixes[qi].len();
                if n > 0 && len > budget {
                    break;
                }
                budget = budget.saturating_sub(len);
                n += 1;
            }
            let (batch, tail_rest) = rest.split_at(n);
            rest = tail_rest;
            self.pass(&mut plans, batch, &mut readouts)?;
            passes += 1;
        }
        tracing::debug!(
            tokenize_ms = t_tok.as_secs_f64() * 1e3,
            prefill_ms = (t_prefill - t_tok).as_secs_f64() * 1e3,
            pass_ms = (t0.elapsed() - t_prefill).as_secs_f64() * 1e3,
            passes,
            requests = reqs.len(),
            "forward"
        );
        Ok(plans
            .into_iter()
            .zip(readouts)
            .map(|(p, r)| BackendOutput {
                readouts: r.into_iter().map(|x| x.expect("every question answered")).collect(),
                input_tokens: p.prefix.len() + p.suffixes.iter().map(Vec::len).sum::<usize>(),
                prefix_cached: p.cached.is_some(),
            })
            .collect())
    }
}

/// Per-request execution state.
struct Plan {
    prefix: Vec<u32>,
    suffixes: Vec<Vec<u32>>,
    n_options: Vec<usize>,
    /// Cache hit for the whole prefix.
    cached: Option<Arc<PrefixKv>>,
    /// K/V this request's tokens may read: the cached prefix, or the part
    /// computed so far.
    past: Option<Arc<PrefixKv>>,
    /// Prefix tokens not yet computed (they go into the next pass).
    tail: std::ops::Range<usize>,
}

impl Engine {
    fn plan(&self, req: &CompiledRequest) -> Result<Plan> {
        // Tokenize every full prompt independently, then split at the longest
        // token prefix shared with the prefix text, so BPE merges across the
        // boundary can never change what the model sees.
        let prefix_ids = self.encode(&req.prefix)?;
        let fulls: Vec<Vec<u32>> = (0..req.questions.len()).map(|i| self.encode(&req.full_prompt(i))).collect::<Result<_>>()?;
        let split = fulls.iter().map(|f| lcp(&prefix_ids, f)).min().unwrap_or(0).min(prefix_ids.len());
        let split = split.min(fulls.iter().map(|f| f.len() - 1).min().unwrap_or(0));
        if split == 0 {
            bail!("empty shared prefix");
        }
        let longest = fulls.iter().map(Vec::len).max().unwrap_or(0);
        if longest >= self.model.max_len {
            bail!("prompt is {longest} tokens; the context limit is {}", self.model.max_len);
        }
        let prefix = fulls[0][..split].to_vec();
        let cached = self.cache.lock().get(&prefix);
        Ok(Plan {
            suffixes: fulls.iter().map(|f| f[split..].to_vec()).collect(),
            n_options: req.questions.iter().map(|q| q.options.len()).collect(),
            past: cached.clone(),
            tail: prefix.len()..prefix.len(),
            cached,
            prefix,
        })
    }

    /// One packed forward pass: pending prefix tails plus `batch` questions.
    fn pass(&self, plans: &mut [Plan], batch: &[(usize, usize)], readouts: &mut [Vec<Option<Readout>>]) -> Result<()> {
        // Past K/V: every plan's computed prefix, concatenated in plan order.
        let mut past_off = Vec::with_capacity(plans.len());
        let mut past_len = 0;
        for p in plans.iter() {
            past_off.push(past_len);
            past_len += p.past.as_ref().map_or(0, |kv| kv.len);
        }
        let pasts: Vec<&PrefixKv> = plans.iter().filter_map(|p| p.past.as_deref()).collect();
        let past_layers: Option<Vec<LayerKv>> = match pasts.len() {
            0 => None,
            1 => Some(pasts[0].layers.clone()),
            _ => Some(
                (0..pasts[0].layers.len())
                    .map(|l| {
                        let ks: Vec<&Tensor> = pasts.iter().map(|p| &p.layers[l].k).collect();
                        let vs: Vec<&Tensor> = pasts.iter().map(|p| &p.layers[l].v).collect();
                        Ok(LayerKv { k: Tensor::cat(&ks, 2)?, v: Tensor::cat(&vs, 2)? })
                    })
                    .collect::<candle_core::Result<_>>()?,
            ),
        };

        // New tokens, each tagged with (plan, segment); segment 0 = prefix tail.
        let (mut tokens, mut pos, mut owner, mut seg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut keep = Vec::new();
        let mut tail_at = vec![None; plans.len()];
        for (pi, p) in plans.iter().enumerate() {
            if p.tail.is_empty() {
                continue;
            }
            tail_at[pi] = Some((tokens.len(), p.tail.len()));
            keep.push((tokens.len(), p.tail.len()));
            for (j, &t) in p.prefix[p.tail.clone()].iter().enumerate() {
                tokens.push(t);
                pos.push((p.tail.start + j) as u32);
                owner.push(pi);
                seg.push(0u32);
            }
        }
        let mut gather = Vec::with_capacity(batch.len());
        for (k, &(pi, qi)) in batch.iter().enumerate() {
            let p = &plans[pi];
            for (j, &t) in p.suffixes[qi].iter().enumerate() {
                tokens.push(t);
                pos.push((p.prefix.len() + j) as u32);
                owner.push(pi);
                seg.push(k as u32 + 1);
            }
            gather.push((tokens.len() - 1) as u32);
        }

        // Mask: a token sees its own plan's past, its plan's tail (causally
        // if it is a tail token), and earlier tokens of its own question.
        let n = tokens.len();
        let width = past_len + n;
        let mut mask = vec![f32::NEG_INFINITY; n * width];
        for i in 0..n {
            let row = &mut mask[i * width..(i + 1) * width];
            let pi = owner[i];
            let plen = plans[pi].past.as_ref().map_or(0, |kv| kv.len);
            row[past_off[pi]..past_off[pi] + plen].fill(0.0);
            if let Some((start, len)) = tail_at[pi] {
                let end = if seg[i] == 0 { i + 1 } else { start + len };
                row[past_len + start..past_len + end].fill(0.0);
            }
            if seg[i] != 0 {
                let mut j = i;
                loop {
                    row[past_len + j] = 0.0;
                    if j == 0 || seg[j - 1] != seg[i] {
                        break;
                    }
                    j -= 1;
                }
            }
        }

        let (hidden, kept) = self.model.forward_packed(past_layers.as_deref(), &tokens, &pos, mask, &gather, &keep)?;
        // Tails are now computed: extend each plan's past and cache the prefix.
        let mut kept = kept.into_iter();
        for p in plans.iter_mut().filter(|p| !p.tail.is_empty()) {
            let tail_kv = kept.next().expect("kept tail");
            let layers = match &p.past {
                Some(head) => head
                    .layers
                    .iter()
                    .zip(tail_kv)
                    .map(|(h, t)| Ok(LayerKv { k: Tensor::cat(&[&h.k, &t.k], 2)?, v: Tensor::cat(&[&h.v, &t.v], 2)? }))
                    .collect::<candle_core::Result<_>>()?,
                None => tail_kv,
            };
            let kv = Arc::new(PrefixKv { layers, len: p.prefix.len() });
            self.cache.lock().put(p.prefix.clone(), kv.clone());
            p.past = Some(kv);
            p.tail = p.prefix.len()..p.prefix.len();
        }
        let logits = hidden.matmul(&self.code_head.t()?)?;
        let hidden: Vec<Vec<f32>> = hidden.to_vec2()?;
        let logits: Vec<Vec<f32>> = logits.to_vec2()?;
        for ((&(pi, qi), h), z) in batch.iter().zip(hidden).zip(logits) {
            let k = plans[pi].n_options[qi];
            readouts[pi][qi] = Some(Readout { logits: z[..k].to_vec(), hidden: h });
        }
        Ok(())
    }
}

/// LRU of prefilled prefixes, bounded by total cached tokens.
struct PrefixCache {
    budget: usize,
    entries: Vec<(Vec<u32>, Arc<PrefixKv>)>,
}

impl PrefixCache {
    fn new(budget: usize) -> Self {
        Self { budget, entries: Vec::new() }
    }

    fn get(&mut self, key: &[u32]) -> Option<Arc<PrefixKv>> {
        let i = self.entries.iter().position(|(k, _)| k == key)?;
        let e = self.entries.remove(i);
        let kv = e.1.clone();
        self.entries.push(e);
        Some(kv)
    }

    fn put(&mut self, key: Vec<u32>, kv: Arc<PrefixKv>) {
        if key.len() > self.budget {
            return;
        }
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, kv));
        while self.entries.iter().map(|(k, _)| k.len()).sum::<usize>() > self.budget {
            self.entries.remove(0);
        }
    }
}
