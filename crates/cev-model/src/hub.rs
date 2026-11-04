//! Locate model files: a local directory, the Hugging Face cache (shared with
//! Python tooling), or a download from huggingface.co into cev's own cache.
//!
//! A decision model may span two repositories (a LoRA adapter plus its base),
//! so the unit here is one repository at one revision with a list of files.

use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Files of one repository (or local directory).
#[derive(Debug, Clone)]
pub struct RepoFiles {
    pub repo: String,
    pub dir: PathBuf,
}

impl RepoFiles {
    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.dir.join(name).is_file()
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        std::fs::read(self.dir.join(name)).with_context(|| format!("{}: reading {name}", self.dir.display()))
    }

    /// The safetensors shards of the checkpoint, from the index when there is one.
    pub fn weights(&self) -> Result<Vec<PathBuf>> {
        weights_in(&self.dir)
    }
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

fn hf_cache() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return p.into();
    }
    std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")).unwrap_or_else(|| home().join(".cache/huggingface/hub"))
}

pub fn cev_cache() -> PathBuf {
    std::env::var_os("CEV_CACHE").map(PathBuf::from).unwrap_or_else(|| home().join(".cache/cev/models"))
}

fn weights_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
        let mut files: Vec<String> = v["weight_map"].as_object().context("bad safetensors index")?.values().filter_map(|f| f.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    if dir.join("model.safetensors").exists() {
        return Ok(vec![dir.join("model.safetensors")]);
    }
    // Any other single safetensors file (adapters: adapter_model.safetensors).
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors") && !p.file_name().is_some_and(|f| f.to_string_lossy().starts_with("readout")))
        .collect();
    found.sort();
    if found.is_empty() {
        bail!("{}: no safetensors weights", dir.display());
    }
    Ok(found)
}

/// Marker in `needed`/`optional` lists for "the model weights" (index + shards
/// or `model.safetensors`).
pub const WEIGHTS: &str = "<weights>";

fn complete(dir: &Path, needed: &[&str]) -> bool {
    needed.iter().all(|f| if *f == WEIGHTS { weights_in(dir).is_ok_and(|w| w.iter().all(|p| p.exists())) } else { dir.join(f).is_file() })
}

/// Files of `repo` (a directory path or a Hugging Face repo id, optionally
/// `@revision`). `needed` files must exist (downloaded if missing); `optional`
/// ones are fetched when the repository has them.
pub fn repo(repo: &str, needed: &[&str], optional: &[&str]) -> Result<RepoFiles> {
    let p = Path::new(repo);
    if p.is_dir() {
        if !complete(p, needed) {
            bail!("{repo}: missing one of {}", needed.join(", "));
        }
        return Ok(RepoFiles { repo: repo.to_string(), dir: p.to_path_buf() });
    }
    let (id, revision) = match repo.split_once('@') {
        Some((r, rev)) => (r, Some(rev)),
        None => (repo, None),
    };
    let slug = id.replace('/', "--");
    // The Hugging Face cache (hf download / transformers) first.
    let snaps = hf_cache().join(format!("models--{slug}/snapshots"));
    if let Ok(rd) = std::fs::read_dir(&snaps) {
        let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        dirs.sort_by_key(|d| std::cmp::Reverse(d.metadata().and_then(|m| m.modified()).ok()));
        for d in dirs {
            if revision.is_some_and(|r| !d.file_name().is_some_and(|f| f.to_string_lossy().starts_with(r))) {
                continue;
            }
            if complete(&d, needed) && optional.iter().all(|f| d.join(f).exists() || !remote_has(id, revision, f)) {
                return Ok(RepoFiles { repo: id.to_string(), dir: d });
            }
        }
    }
    let dir = cev_cache().join(match revision {
        Some(r) => format!("{slug}@{r}"),
        None => slug,
    });
    download(id, revision, &dir, needed, optional)?;
    if !complete(&dir, needed) {
        bail!("{repo}: download incomplete");
    }
    Ok(RepoFiles { repo: id.to_string(), dir })
}

fn url(repo: &str, revision: Option<&str>, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{}/{file}", revision.unwrap_or("main"))
}

fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(ureq::config::Config::builder().http_status_as_error(false).build())
}

fn with_token(mut req: ureq::RequestBuilder<ureq::typestate::WithoutBody>) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", &format!("Bearer {tok}"));
    }
    req
}

/// Whether the repository has `file` (a HEAD request; false when offline).
pub fn remote_has(repo: &str, revision: Option<&str>, file: &str) -> bool {
    let u = url(repo, revision, file);
    with_token(agent().head(&u)).call().map(|r| r.status().is_success() || r.status().is_redirection()).unwrap_or(false)
}

fn fetch(repo: &str, revision: Option<&str>, dir: &Path, file: &str, required: bool) -> Result<bool> {
    let dest = dir.join(file);
    if dest.exists() {
        return Ok(true);
    }
    let u = url(repo, revision, file);
    let mut resp = with_token(agent().get(&u)).call().with_context(|| format!("GET {u}"))?;
    if resp.status() == 404 {
        if required {
            bail!("GET {u}: HTTP 404");
        }
        return Ok(false);
    }
    if !resp.status().is_success() {
        bail!("GET {u}: HTTP {}", resp.status());
    }
    tracing::info!(%u, "downloading");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("part");
    let mut out = std::fs::File::create(&tmp)?;
    std::io::copy(&mut resp.body_mut().with_config().limit(u64::MAX).reader(), &mut out)?;
    std::fs::rename(tmp, dest)?;
    Ok(true)
}

fn download(repo: &str, revision: Option<&str>, dir: &Path, needed: &[&str], optional: &[&str]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut seen = HashSet::new();
    for (list, required) in [(needed, true), (optional, false)] {
        for f in list {
            if !seen.insert(*f) {
                continue;
            }
            if *f == WEIGHTS {
                if fetch(repo, revision, dir, "model.safetensors.index.json", false)? {
                    for w in weights_in(dir)? {
                        fetch(repo, revision, dir, w.file_name().and_then(|f| f.to_str()).context("bad shard name")?, true)?;
                    }
                } else if !fetch(repo, revision, dir, "model.safetensors", required)? && required {
                    bail!("{repo}: no model.safetensors");
                }
            } else {
                fetch(repo, revision, dir, f, required)?;
            }
        }
    }
    Ok(())
}

/// A plain backbone: `config.json`, `tokenizer.json` and weights.
pub fn backbone(model: &str) -> Result<RepoFiles> {
    repo(model, &["config.json", "tokenizer.json", WEIGHTS], &["tokenizer_config.json", "chat_template.jinja", "generation_config.json", "preprocessor_config.json"])
}

/// Short names for the decision models cev knows, as on ollama.com.
pub fn alias(model: &str) -> String {
    let (name, tag) = match model.split_once(':') {
        Some((n, t)) if !n.contains('/') => (n, Some(t)),
        _ => (model, None),
    };
    let pick = |default: &str, table: &[(&str, &str)]| -> Option<String> {
        let t = tag.unwrap_or(default);
        table.iter().find(|(k, _)| *k == t).map(|(_, v)| v.to_string())
    };
    let hit = match name {
        "nimble" | "nimble-v3" => pick("9b", &[("9b", "bespokelabs/Bespoke-Nimble-9B-v3"), ("27b", "bespokelabs/Bespoke-Nimble-27B-v3"), ("latest", "bespokelabs/Bespoke-Nimble-9B-v3")]),
        "kev" => pick("9b", &[("0.8b", "jaredpalmer/kev-0.8b"), ("4b", "jaredpalmer/kev-4b"), ("9b", "jaredpalmer/kev-9b"), ("27b", "jaredpalmer/kev-27b"), ("latest", "jaredpalmer/kev-9b")]),
        "d3" => pick("4b", &[("2b", "vllm-sr/d3-nano"), ("4b", "vllm-sr/d3-mini"), ("8b", "vllm-sr/d3-flash"), ("9b", "vllm-sr/d3-flash"), ("27b", "vllm-sr/d3"), ("latest", "vllm-sr/d3-mini")]),
        "d1" => pick("3b", &[("2b", "LiquidAI/d1-2B"), ("3b", "LiquidAI/d1-3B"), ("latest", "LiquidAI/d1-3B")]),
        "d1-omni" => pick("600m", &[("600m", "LiquidAI/d1-omni-600M"), ("2b", "LiquidAI/d1-Omni-2B"), ("latest", "LiquidAI/d1-omni-600M")]),
        _ => None,
    };
    hit.unwrap_or_else(|| model.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases() {
        assert_eq!(alias("kev"), "jaredpalmer/kev-9b");
        assert_eq!(alias("kev:0.8b"), "jaredpalmer/kev-0.8b");
        assert_eq!(alias("d3:2b"), "vllm-sr/d3-nano");
        assert_eq!(alias("d1-omni"), "LiquidAI/d1-omni-600M");
        assert_eq!(alias("nimble-v3"), "bespokelabs/Bespoke-Nimble-9B-v3");
        assert_eq!(alias("Qwen/Qwen3-1.7B"), "Qwen/Qwen3-1.7B");
        assert_eq!(alias("kev:99b"), "kev:99b");
    }
}
