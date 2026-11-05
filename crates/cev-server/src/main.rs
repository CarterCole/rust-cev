use anyhow::Result;
use clap::Parser;
use cev_core::{Backend, MockBackend};
use cev_model::EngineOptions;
use cev_runtime::{Cev, LearnConfig, RuntimeConfig, Store};
use std::sync::Arc;

/// cev: typed System One decisions with feedback and online learning.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Model: a Hugging Face repo id, a local directory, or a short name of a
    /// decision model (`kev`, `kev:4b`, `nimble-v3`, `d3`, `d3:2b`, `d1`,
    /// `d1-omni`). Plain Qwen3 / Qwen3.5 checkpoints run cev's own prompt.
    #[arg(long, env = "CEV_MODEL", default_value = "Qwen/Qwen3-1.7B")]
    model: String,
    /// Name reported as `model` in responses.
    #[arg(long, env = "CEV_MODEL_NAME")]
    model_name: Option<String>,
    /// auto | cpu | metal | cuda
    #[arg(long, env = "CEV_DEVICE", default_value = "auto")]
    device: String,
    /// auto | f32 | f16 | bf16
    #[arg(long, env = "CEV_DTYPE", default_value = "auto")]
    dtype: String,
    #[arg(long, env = "CEV_MAX_CONTEXT", default_value_t = 32768)]
    max_context: usize,
    /// SQLite file holding decisions, feedback and adapters.
    #[arg(long, env = "CEV_DB", default_value = "cev.db")]
    db: String,
    #[arg(long, env = "CEV_ADDR", default_value = "127.0.0.1:8080")]
    addr: String,
    /// Require `Authorization: Bearer <key>`.
    #[arg(long, env = "CEV_API_KEY")]
    api_key: Option<String>,
    /// Concurrent forward passes.
    #[arg(long, env = "CEV_CONCURRENCY", default_value_t = 1)]
    concurrency: usize,
    /// Store labels only; do not adapt online.
    #[arg(long, env = "CEV_NO_ONLINE_LEARNING")]
    no_online_learning: bool,
    /// Do not persist hidden features (disables learning from later feedback).
    #[arg(long, env = "CEV_NO_STORE_FEATURES")]
    no_store_features: bool,
    /// Labels before an adapter may be served.
    #[arg(long, env = "CEV_MIN_EXAMPLES", default_value_t = 8)]
    min_examples: u64,
    /// Position-bias correction: auto | none | calibrate | permute | full.
    /// `auto` is the model family's default: `full` for zero-shot code
    /// readouts (plain Qwen checkpoints), `none` for trained decision models.
    #[arg(long, env = "CEV_DEBIAS", default_value = "auto")]
    debias: String,
    /// Most option rotations per question when permuting.
    #[arg(long, env = "CEV_MAX_PERMUTATIONS", default_value_t = 4)]
    max_permutations: usize,
    /// Serve the browser page from this directory (the repository's `web/`)
    /// at `/`; it then runs its decisions on this server.
    #[arg(long, env = "CEV_WEB")]
    web: Option<std::path::PathBuf>,
    /// Use a keyword-matching mock instead of a neural backbone (for wiring tests).
    #[arg(long)]
    mock: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=info".into()))
        .init();
    let args = Args::parse();
    let backend: Arc<dyn Backend> = if args.mock {
        Arc::new(MockBackend::default())
    } else {
        let opts = EngineOptions {
            model: args.model.clone(),
            device: args.device.clone(),
            dtype: args.dtype.clone(),
            max_context: args.max_context,
            name: args.model_name.clone(),
            ..Default::default()
        };
        tokio::task::spawn_blocking(move || cev_model::load(opts)).await??
    };
    let caps = backend.capabilities();
    tracing::info!(model = backend.id(), family = caps.family.as_str(), max_options = caps.max_options, modalities = ?caps.modalities(), "backbone loaded");
    let cfg = RuntimeConfig {
        online_learning: !args.no_online_learning,
        store_features: !args.no_store_features,
        learn: LearnConfig { min_examples: args.min_examples, ..Default::default() },
        debias: serde_json::from_value(serde_json::Value::String(args.debias.clone()))
            .map_err(|_| anyhow::anyhow!("--debias must be auto, none, calibrate, permute or full"))?,
        max_permutations: args.max_permutations,
    };
    let cev = Cev::new(backend, Store::open(&args.db)?, cfg)?;
    if let Some(dir) = &args.web {
        anyhow::ensure!(dir.join("index.html").is_file(), "--web {}: no index.html there", dir.display());
    }
    let app = cev_server::app_serving(cev_server::AppState::new(cev, args.concurrency, args.api_key), args.web.as_deref());
    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    match &args.web {
        Some(dir) => tracing::info!("listening on http://{} (page from {} at /, GraphiQL at /graphql)", args.addr, dir.display()),
        None => tracing::info!("listening on http://{} (GraphiQL at /graphql)", args.addr),
    }
    axum::serve(listener, app).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); }).await?;
    Ok(())
}
