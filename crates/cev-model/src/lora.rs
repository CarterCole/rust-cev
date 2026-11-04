//! PEFT LoRA adapters merged into base weights at load time:
//! `W' = W + (B · A) · α / r`, computed in f32 and rounded once to the
//! weight dtype (the same bits Kev's and mlx-lm's loaders produce).

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct AdapterConfig {
    pub base_model_name_or_path: Option<String>,
    pub r: usize,
    pub lora_alpha: f64,
    #[serde(default)]
    pub use_rslora: bool,
    #[serde(default)]
    pub target_modules: Vec<String>,
    #[serde(default)]
    pub revision: Option<String>,
}

impl AdapterConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let s = std::fs::read(dir.join("adapter_config.json")).with_context(|| format!("{}: adapter_config.json", dir.display()))?;
        Ok(serde_json::from_slice(&s)?)
    }

    pub fn scale(&self) -> f64 {
        if self.use_rslora { self.lora_alpha / (self.r as f64).sqrt() } else { self.lora_alpha / self.r as f64 }
    }
}

/// `module path -> (A, B)` of an adapter, with PEFT's `base_model.model.`
/// wrapper prefix and the backbone's own prefix (`model.`, `language_model.`,
/// ...) stripped down to `layers.N....`.
pub struct Adapter {
    pub cfg: AdapterConfig,
    pub deltas: HashMap<String, (Tensor, Tensor)>,
}

fn strip_to_layers(name: &str) -> Option<&str> {
    name.find("layers.").map(|i| &name[i..])
}

impl Adapter {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let cfg = AdapterConfig::load(dir)?;
        let file = dir.join("adapter_model.safetensors");
        let tensors = candle_core::safetensors::load(&file, dev).with_context(|| format!("{}", file.display()))?;
        let mut deltas: HashMap<String, (Option<Tensor>, Option<Tensor>)> = HashMap::new();
        for (name, t) in tensors {
            let (stem, is_a) = if let Some(s) = name.strip_suffix(".lora_A.weight") {
                (s, true)
            } else if let Some(s) = name.strip_suffix(".lora_B.weight") {
                (s, false)
            } else {
                bail!("unsupported adapter tensor `{name}` (only LoRA A/B weights are merged)");
            };
            let key = strip_to_layers(stem).with_context(|| format!("adapter tensor `{name}` is not under a `layers.N.` module"))?.to_string();
            let e = deltas.entry(key).or_default();
            if is_a {
                e.0 = Some(t);
            } else {
                e.1 = Some(t);
            }
        }
        let deltas = deltas
            .into_iter()
            .map(|(k, (a, b))| Ok((k.clone(), (a.with_context(|| format!("{k}: missing lora_A"))?, b.with_context(|| format!("{k}: missing lora_B"))?))))
            .collect::<Result<_>>()?;
        Ok(Self { cfg, deltas })
    }

    /// Merge into `weight` (the dense matrix of `module`, e.g.
    /// `layers.3.self_attn.q_proj`), if the adapter targets it.
    pub fn merge(&self, module: &str, weight: &Tensor) -> Result<Tensor> {
        let Some((a, b)) = self.deltas.get(module) else { return Ok(weight.clone()) };
        let delta = (b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)? * self.cfg.scale())?;
        if delta.dims() != weight.dims() {
            bail!("{module}: adapter delta {:?} does not match weight {:?}", delta.dims(), weight.dims());
        }
        Ok((weight.to_dtype(DType::F32)? + delta)?.to_dtype(weight.dtype())?)
    }

    pub fn targets(&self) -> impl Iterator<Item = &str> {
        self.deltas.keys().map(String::as_str)
    }
}

/// Load every tensor of `shards` in `dtype`, merging `adapter` into the
/// matrices it targets under `root` (the text model's prefix, e.g.
/// `model.language_model`; other towers such as `mtp.layers.*` are left
/// alone), and return them on `dev` by name for a `VarBuilder::from_tensors`.
///
/// The merge runs on the CPU in f32, one matrix at a time, and only the
/// result is moved to `dev`: exact, and independent of the accelerator (on
/// Metal, merging a 9B model's matrices in place came out as zeros).
/// `adapter` must be loaded on the CPU.
pub fn merged_tensors(shards: &[std::path::PathBuf], dtype: DType, dev: &Device, adapter: &Adapter, root: &str) -> Result<HashMap<String, Tensor>> {
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(shards)? };
    let mut out = HashMap::new();
    let mut merged = std::collections::HashSet::new();
    let prefix = format!("{root}.");
    for (name, _) in st.tensors() {
        let t = st.load(&name, &Device::Cpu)?;
        let module = name.strip_prefix(&prefix).and_then(|n| n.strip_suffix(".weight")).and_then(strip_to_layers);
        let t = match module {
            Some(m) if adapter.deltas.contains_key(m) => {
                merged.insert(m.to_string());
                // f32 base + f32 delta, rounded once to the serving dtype.
                adapter.merge(m, &t.to_dtype(DType::F32)?)?
            }
            _ => t,
        };
        let t = if t.dtype().is_float() { t.to_dtype(dtype)? } else { t };
        out.insert(name, t.to_device(dev)?);
    }
    if merged.len() != adapter.deltas.len() {
        let missing: Vec<&str> = adapter.targets().filter(|m| !merged.contains(*m)).collect();
        bail!("{} adapter matrices found no weight to merge into under `{root}`: {:?}", missing.len(), &missing[..missing.len().min(5)]);
    }
    tracing::info!(merged = merged.len(), scale = adapter.cfg.scale(), "merged LoRA adapter");
    Ok(out)
}
