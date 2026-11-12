use anyhow::Result;
use clap::Parser;
use cev_core::{Backend, MockBackend};
use cev_model::EngineOptions;
use cev_runtime::{AdapterRow, Cev, LearnConfig, RuntimeConfig, Store};
use std::io::{Read, Write};
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
    /// Host names the server answers to (repeat, or comma-separate in the
    /// env var); `*` answers any. Default: on a loopback address only
    /// localhost, 127.0.0.1 and [::1], which stops DNS-rebinding pages from
    /// reading the API; on any other address, any host (set `--api-key`).
    #[arg(long = "allowed-host", env = "CEV_ALLOWED_HOSTS", value_delimiter = ',')]
    allowed_hosts: Vec<String>,
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
    /// How many standard errors ahead of the base model an adapter must be
    /// before it is served. 0 serves any adapter that is ahead on average.
    #[arg(long, env = "CEV_GATE_Z", default_value_t = 0.8)]
    gate_z: f64,
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
    /// Write every stored adapter as NDJSON to FILE (`-` for stdout), then exit.
    #[arg(long, value_name = "FILE")]
    export_adapters: Option<String>,
    /// Load adapters from an NDJSON FILE written by --export-adapters (`-` for
    /// stdin) as starting points for this backbone, then exit. They are not
    /// served until they beat the base model on new labels.
    #[arg(long, value_name = "FILE")]
    import_adapters: Option<String>,
    /// With --import-adapters: only take adapters fit on this model id. Needed
    /// when the file mixes several.
    #[arg(long, value_name = "MODEL")]
    import_from: Option<String>,
    /// With --import-adapters: replace adapters that already exist.
    #[arg(long)]
    import_overwrite: bool,
}

fn export_adapters(db: &str, path: &str) -> Result<()> {
    if !std::path::Path::new(db).exists() {
        anyhow::bail!("no database at {db}");
    }
    let rows = Store::open(db)?.all_adapters()?;
    let mut out: Box<dyn Write> = if path == "-" { Box::new(std::io::stdout().lock()) } else { Box::new(std::io::BufWriter::new(std::fs::File::create(path)?)) };
    for row in &rows {
        serde_json::to_writer(&mut out, row)?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    eprintln!("exported {} adapters", rows.len());
    Ok(())
}

fn read_adapter_rows(path: &str, from: Option<&str>) -> Result<Vec<AdapterRow>> {
    let text = if path == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    } else {
        std::fs::read_to_string(path)?
    };
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        rows.push(serde_json::from_str::<AdapterRow>(line).map_err(|e| anyhow::anyhow!("{path}:{}: {e}", i + 1))?);
    }
    match from {
        Some(model) => rows.retain(|r| r.model == model),
        None => {
            let mut models: Vec<&str> = rows.iter().map(|r| r.model.as_str()).collect();
            models.sort_unstable();
            models.dedup();
            if models.len() > 1 {
                anyhow::bail!("{path} holds adapters from {} models ({}); pick one with --import-from", models.len(), models.join(", "));
            }
        }
    }
    if rows.is_empty() {
        anyhow::bail!("no adapters to import from {path}");
    }
    Ok(rows)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=info".into()))
        .init();
    let args = Args::parse();
    if let Some(path) = &args.export_adapters {
        return export_adapters(&args.db, path);
    }
    let import_rows = args.import_adapters.as_deref().map(|path| read_adapter_rows(path, args.import_from.as_deref())).transpose()?;
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
        learn: LearnConfig { min_examples: args.min_examples, gate_z: args.gate_z, ..Default::default() },
        debias: serde_json::from_value(serde_json::Value::String(args.debias.clone()))
            .map_err(|_| anyhow::anyhow!("--debias must be auto, none, calibrate, permute or full"))?,
        max_permutations: args.max_permutations,
    };
    let cev = Cev::new(backend, Store::open(&args.db)?, cfg)?;
    if let Some(rows) = import_rows {
        let stats = cev.import_adapters(&rows, args.import_overwrite)?;
        println!("{}", serde_json::to_string(&stats)?);
        return Ok(());
    }
    if let Some(dir) = &args.web {
        anyhow::ensure!(dir.join("index.html").is_file(), "--web {}: no index.html there", dir.display());
    }
    let listener = tokio::net::TcpListener::bind(&args.addr).await?;
    let mut state = cev_server::AppState::new(cev, args.concurrency, args.api_key);
    let hosts: Vec<String> = args.allowed_hosts.iter().map(|h| h.trim().to_string()).filter(|h| !h.is_empty()).collect();
    if hosts.is_empty() {
        if listener.local_addr()?.ip().is_loopback() {
            state = state.with_allowed_hosts(cev_server::LOOPBACK_HOSTS.map(String::from));
        }
    } else if !hosts.iter().any(|h| h == "*") {
        state = state.with_allowed_hosts(hosts);
    }
    let app = cev_server::app_serving(state, args.web.as_deref());
    match &args.web {
        Some(dir) => tracing::info!("listening on http://{} (page from {} at /, GraphiQL at /graphql)", args.addr, dir.display()),
        None => tracing::info!("listening on http://{} (GraphiQL at /graphql)", args.addr),
    }
    axum::serve(listener, app).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); }).await?;
    Ok(())
}
