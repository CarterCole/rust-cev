use anyhow::Result;
use clap::Parser;
use cev_core::{Backend, MockBackend};
use cev_model::{Engine, EngineOptions};
use cev_runtime::{Cev, LearnConfig, RuntimeConfig, Store};
use std::sync::Arc;

/// cev: typed System One decisions with feedback and online learning.
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Backbone: Hugging Face repo id or local directory (Qwen3 safetensors).
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
    /// Position-bias correction: none | calibrate | permute | full.
    #[arg(long, env = "CEV_DEBIAS", default_value = "full")]
    debias: String,
    /// Most option rotations per question when permuting.
    #[arg(long, env = "CEV_MAX_PERMUTATIONS", default_value_t = 4)]
    max_permutations: usize,
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
        Arc::new(tokio::task::spawn_blocking(move || Engine::load(opts)).await??)
    };
    tracing::info!(model = backend.id(), codes = backend.codes().len(), "backbone loaded");
    let cfg = RuntimeConfig {
        online_learning: !args.no_online_learning,
        store_features: !args.no_store_features,
        learn: LearnConfig { min_examples: args.min_examples, ..Default::default() },
        debias: serde_json::from_value(serde_json::Value::String(args.debias.clone()))
            .map_err(|_| anyhow::anyhow!("--debias must be none, calibrate, permute or full"))?,
        max_permutations: args.max_permutations,
    };
    let cev = Cev::new(backend, Store::open(&args.db)?, cfg)?;
    let app = cev_server::app(cev_server::AppState::new(cev, args.concurrency, args.api_key));
    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    tracing::info!("listening on http://{} (GraphiQL at /graphql)", args.addr);
    axum::serve(listener, app).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); }).await?;
    Ok(())
}
