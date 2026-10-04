//! Locate model files: a local directory, the Hugging Face cache (shared with
//! Python tooling), or a download from huggingface.co into cev's own cache.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

pub struct ModelFiles {
    pub dir: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

fn hf_cache() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return p.into();
    }
    std::env::var_os("HF_HOME")
        .map(|h| PathBuf::from(h).join("hub"))
        .unwrap_or_else(|| home().join(".cache/huggingface/hub"))
}

fn weights_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
        let mut files: Vec<String> = v["weight_map"]
            .as_object()
            .context("bad safetensors index")?
            .values()
            .filter_map(|f| f.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    Ok(vec![dir.join("model.safetensors")])
}

fn complete(dir: &Path) -> Option<ModelFiles> {
    let config = dir.join("config.json");
    let tokenizer = dir.join("tokenizer.json");
    let weights = weights_in(dir).ok()?;
    (config.exists() && tokenizer.exists() && weights.iter().all(|w| w.exists())).then(|| ModelFiles {
        dir: dir.to_path_buf(),
        config,
        tokenizer,
        weights,
    })
}

/// `model` is a directory path or a Hugging Face repo id such as `Qwen/Qwen3-1.7B`.
pub fn resolve(model: &str) -> Result<ModelFiles> {
    let p = Path::new(model);
    if p.is_dir() {
        return complete(p).with_context(|| format!("{model}: needs config.json, tokenizer.json and safetensors"));
    }
    let slug = model.replace('/', "--");
    let snaps = hf_cache().join(format!("models--{slug}/snapshots"));
    if let Ok(rd) = std::fs::read_dir(&snaps) {
        for e in rd.flatten() {
            if let Some(f) = complete(&e.path()) {
                return Ok(f);
            }
        }
    }
    let dir = home().join(".cache/cev/models").join(&slug);
    if let Some(f) = complete(&dir) {
        return Ok(f);
    }
    download(model, &dir)?;
    complete(&dir).context("download incomplete")
}

fn download(repo: &str, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let get = |file: &str| -> Result<()> {
        let dest = dir.join(file);
        if dest.exists() {
            return Ok(());
        }
        let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
        tracing::info!(%url, "downloading");
        let mut req = ureq::get(&url);
        if let Ok(tok) = std::env::var("HF_TOKEN") {
            req = req.header("Authorization", &format!("Bearer {tok}"));
        }
        let mut resp = req.call().with_context(|| format!("GET {url}"))?;
        if resp.status() != 200 {
            bail!("GET {url}: HTTP {}", resp.status());
        }
        let tmp = dest.with_extension("part");
        let mut out = std::fs::File::create(&tmp)?;
        std::io::copy(&mut resp.body_mut().as_reader(), &mut out)?;
        std::fs::rename(tmp, dest)?;
        Ok(())
    };
    get("config.json")?;
    get("tokenizer.json")?;
    if get("model.safetensors.index.json").is_ok() {
        for w in weights_in(dir)? {
            get(w.file_name().and_then(|f| f.to_str()).context("bad shard name")?)?;
        }
    } else {
        get("model.safetensors")?;
    }
    Ok(())
}
