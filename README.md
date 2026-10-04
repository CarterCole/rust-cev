# cev

**Calibrated Enum Verdicts.** Typed, calibrated decisions in one forward pass, served from Rust. It learns from feedback while it runs.

cev is a from-scratch Rust take on TypeSafe's [Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev) models. You send program state plus typed questions. You get back typed answers with probabilities, and the model never generates text.

- **noul** asks "is this true?" and returns a probability.
- **choice** asks "which of these?" and returns an option plus a distribution.
- **score** asks "which level?" and returns the expected level plus a distribution.

The wire format is compatible with `POST /v1/systemone`, so existing Jev SDKs work. On top of that, cev adds:

- **A decision log.** Every answer has a `decision_id` and is stored with its full context: the state, the question, the exact prompt text, the logits, and what was served.
- **Feedback.** Labels are attached by id and can always be exported as training data.
- **Online learning.** A per-task adapter learns from labels as they arrive, in Rust, without retraining the backbone.
- **A typed Rust SDK.** `#[derive(cev::Choice)]` on an enum turns a model decision into a `match`.

## How it works

```text
state ─┐
       ├─ compile ─▶ "<system> <evidence>{state}</evidence>"   shared prefix (LRU-cached)
       │             "Question … Options: A. … B. …"            one suffix per question (+ rotated variants),
questions                                                        all packed into ONE block-masked pass
                               │
                      Qwen3 (candle, Metal/CUDA/CPU)
                               │  next-token logits of the answer codes only ("A", "B", …, up to 255)
                               ▼
          z = debiased log-scores (rotations averaged, minus content-free "N/A" baseline)
          h = hidden state at the answer position
                               │
          per-task adapter:  z' = z·e^-τ + b + W·norm(h)     (only served once it wins prequentially)
                               ▼
               typed answer + probabilities + decision_id  ──▶  SQLite log
                                                                   ▲
                               /v1/feedback {decision_id, label} ──┘──▶ adapter update, /v1/export ──▶ PyTorch LoRA
```

- **Backbone.** Any Qwen3 safetensors checkpoint: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` (default), `Qwen/Qwen3-4B-Instruct-2507`, or a directory produced by `training/train_lora.py`. There is no custom head. The answer is the LM's own next-token distribution restricted to one-token option codes, so zero-shot works on day one and an out-of-type answer is impossible. The code vocabulary is checked against the tokenizer at startup.
- **One pass per request.** Everything goes into a single forward pass under an additive mask:
  - the (uncached tail of the) state,
  - every question,
  - every debiasing variant.

  Each question sees the state and itself only, with positions restarting per question. That is exactly equivalent to asking each question alone. States longer than 512 tokens prefill their head in chunks first. Prefixes are LRU-cached across requests. On Metal, attention uses candle's fused SDPA kernel with native GQA.
- **Parity.** In f32, cev matches HF transformers to |Δp| < 5e-5 on CPU and Metal (`training/parity.py`).
- **Confidence.** Uses TypeSafe's definitions. For choice: `(p_max − 1/K)/(1 − 1/K)`. For score: `1 − E|level − mode|/(K−1)`.

### Debiasing

Small instruct models have a strong option-position bias. Zero-shot Qwen3-1.7B picks "A" on 28 of 30 severity tickets. cev corrects this at inference time with two standard methods, set by `--debias` or per request with `"debias"`:

- **`calibrate`: contextual calibration.** cev subtracts the log-probabilities the model gives the same question over content-free evidence ("N/A"). Those logits don't depend on the state, so they are cached per question and are free after the first request with a given question set.
- **`permute`: rotation averaging.** cev averages over up to `--max-permutations` (default 4) cyclic rotations of the option order. The rotations are packed into the same pass.
- **`full`** (default) applies both. **`none`** returns the raw distribution.

Zero-shot accuracy on small built-in tasks (`scripts/eval_zero_shot.py`):

| Qwen3-1.7B | severity (30) | "is the user blocked?" (30) | routing (12) |
|---|---:|---:|---:|
| none | 0.40 | 0.70 | 0.67 |
| calibrate | 0.77 | 0.80 | 0.92 |
| full | **0.80** | **0.97** | **0.92** |

### Online learning

The backbone stays frozen. Each task gets a residual layer, keyed by `task`, which is either given explicitly or derived from the type, instructions, and option names:

```text
z'_k = z_k · e^{-τ} + b_k + w_k · x,     x = standardize(h)/√d
```

- `τ` learns calibration.
- `b` learns label priors.
- `w` is a linear probe on the backbone's own representation of the question.

At zero, the layer is exactly the base model. After each label, cev refits over a replay buffer of recent labels (SGD warm start). It serves the adapter only once it beats the base model prequentially. That means every label is scored by both the base model and the adapter *before* either learns from it, so the comparison is always on unseen data. `GET /v1/tasks` shows both losses.

Measured with a 3-level severity scale (`scripts/demo_online.py`):

- **Raw Qwen3-0.6B** (`--debias none`). 21 labels raised held-out accuracy from 0.33 to 0.89. The adapter learned T = 2.47, so the raw model was overconfident.
- **Offline LoRA fine-tune.** The same 21 labels (`training/train_lora.py`), served back through cev, reached 1.00.
- **Debiased Qwen3-1.7B.** The base model was already at 0.89, and the adapter correctly stayed inactive because it did not beat the base prequentially (0.528 vs 0.509 loss). The gate prevents regressions.

The held-out set is only 9 tickets from the same distribution, so treat these as tests of the loop, not benchmarks.

### Offline training (PyTorch)

Online adapters are cheap and immediate. For bigger shifts, fine-tune the backbone on the log:

```bash
cd training && uv sync
uv run python train_lora.py --base Qwen/Qwen3-1.7B --source http://127.0.0.1:8080 --out ./ckpt/v1
cev --model ./ckpt/v1
```

`/v1/export` returns one row per labelled decision. Each row has the exact served `prompt`, the answer `codes`, the `target` distribution, and the weight, comment, and metadata. That keeps training byte-identical to serving. The trainer applies soft cross-entropy on the code logits only (LoRA r=16), holds out a deterministic 10% of rows for before/after evaluation, merges the adapter, and writes safetensors that cev loads directly.

`/v1/export?labeled=false` also returns unlabelled decisions, for example to label them offline with a bigger teacher model.

## Run

```bash
cargo run --release -p cev-server --features metal -- --model Qwen/Qwen3-1.7B   # macOS
cargo run --release -p cev-server --features cuda  -- --model Qwen/Qwen3-4B-Instruct-2507
cargo run --release -p cev-server -- --mock        # no model; keyword stub for wiring tests
```

From crates.io, `cargo install cev-server --features metal` (or `cuda`) installs the same binary as `cev`.

Weights come from a local directory or the Hugging Face cache, or are downloaded into `~/.cache/cev/models`. The `--dtype auto` default picks f32 on CPU, and on Metal for checkpoints up to 4 GB; otherwise it picks bf16.

Other flags:

| Flag | Default | Purpose |
|---|---|---|
| `--db` | `cev.db` | SQLite file for the decision log |
| `--addr` | `127.0.0.1:8080` | Listen address |
| `--api-key` | none | Require a bearer token |
| `--no-online-learning` | off | Only store labels; don't adapt |
| `--no-store-features` | off | Don't store hidden states |
| `--min-examples` | 8 | Labels needed before an adapter can be served |
| `--debias` | `full` | Debias mode: `none`, `calibrate`, `permute` or `full` |

Each flag has a matching `CEV_*` environment variable.

Latency on an M-series Mac, f32, 3 questions, end to end over HTTP (`scripts/bench.py`). "Cached" means the state prefix was seen before.

| model | debias | short state (~270 tok): cached / uncached | long state (~2.3k tok): cached / uncached |
|---|---|---:|---:|
| Qwen3-0.6B | none, calibrate | 35 / 49 ms | 59 / 439 ms |
| Qwen3-0.6B | full | 78 / 91 ms | 129 / 510 ms |
| Qwen3-1.7B | none, calibrate | 70 / 100 ms | 93 / 831 ms |
| Qwen3-1.7B | full | 158 / 193 ms | 212 / 1000 ms |

### In the browser

`crates/cev-wasm` compiles the engine, the runtime and the online learner to WebAssembly, and `web/` is a static page around it: ask typed questions, click the right answer to send a label, and watch the adapter take over.

```bash
scripts/build_web.sh && python3 -m http.server -d web 8787    # then open http://localhost:8787
```

- The page starts on the mock backend. "Load Qwen3-0.6B" streams `model.safetensors` (1.5 GB) into the wasm module, converting each tensor to f32 as it arrives. Weights come from `web/models/Qwen3-0.6B/` if that exists (a symlink to a Hugging Face snapshot directory works), otherwise from huggingface.co, cached in the browser afterwards.
- Answers are the same as the native CPU build.
- f32 Qwen3-0.6B needs about 3 GB of the 4 GB a wasm32 module can address, so f32 is desktop-only and 0.6B-only.
- The decision log is in memory (SQLite is not built for wasm) and is gone on reload. "Export labels" downloads the same NDJSON as `/v1/export`.

**int8.** `cev-model`'s `quantize` example writes an int8 copy of a checkpoint (per-row scales, inputs quantized to 13 bits at run time). It loads anywhere a model directory does (`cev --model <dir>`, CPU only), and the page offers it when it is in `web/models/`:

```bash
cargo run --release -p cev-model --example quantize -- Qwen/Qwen3-0.6B web/models/Qwen3-0.6B-q8
```

Single-threaded wasm on an M-series Mac, short state and 3 questions (`node scripts/bench_web.mjs <model dir>`, same V8 as Chrome):

| Qwen3-0.6B | file | wasm memory | calibrate: first / new state / same state | full: first / new state / same state |
|---|---:|---:|---:|---:|
| f32 | 1.5 GB | 2.9 GB | 10.6 / 5.2 / 3.3 s | 20.5 / 12.2 / 10.1 s |
| int8 | 0.75 GB | 1.3 GB | 10.1 / 5.0 / 3.2 s | 20.0 / 11.7 / 9.7 s |

int8 halves the download and memory but is not faster: 70–85% of the time is the matmul kernel either way, and 16-bit × 8-bit integer products cost about what f32 does in 128-bit SIMD. Against f32 on the built-in zero-shot tasks (`scripts/eval_zero_shot.py <int8 url> <f32 url>`), int8 picks the same answer on 69–71 of 72 questions depending on debias mode, with a mean probability difference of 0.015–0.04 and accuracy within a few points either way.

## API

### REST

| Method | Path | |
|---|---|---|
| POST | `/v1/systemone` | Jev-compatible decide. Extensions: `task` per question, `no_store`, `debias`; `request_id`, `latency_ms`, and `x_cev.decision_id` in the response |
| GET | `/v1/models` | Backbone, code vocabulary size, prompt version |
| POST | `/v1/feedback` | `{decision_id \| request_id+question_id, label?, weight?, comment?, metadata?}` |
| POST | `/v1/examples` | `{examples: [{state, questions, labels}]}`: decide, then learn from the labels |
| GET | `/v1/decisions?task=&limit=&offset=` | Recent decisions |
| GET | `/v1/decisions/{id}` | One decision with its state, exact prompt, and all feedback |
| GET | `/v1/tasks`, `/v1/tasks/{task}` | Adapter status: labels, temperature, prequential losses, active |
| DELETE | `/v1/tasks/{task}` | Drop an adapter; its labels stay in the log |
| GET | `/v1/export?task=&labeled=&since=&limit=` | NDJSON training rows |
| GET | `/v1/stats`, `/health` | Counts and liveness |

Labels:

| Question type | Accepted labels |
|---|---|
| noul | `true`/`false`, `"yes"`/`"no"`, or a probability |
| choice | The option name, or `{"option": p}` |
| score | A level index (fractional values split between neighbours), or `{"0": p, …}` |

```bash
curl -s localhost:8080/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"ticket": "The app crashes when I tap Log in"},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "tech": "Bugs and crashes"}},
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["Cosmetic", "Degraded", "Blocking"]}
  }}'
curl -s localhost:8080/v1/feedback -H 'content-type: application/json' \
  -d '{"decision_id": "dec_…", "label": 2, "comment": "user is fully blocked", "metadata": {"by": "agent-7"}}'
```

### GraphQL

`POST /graphql` (GraphiQL UI at `GET /graphql`):

```graphql
mutation {
  decide(state: {ticket: "The app crashes on login"}, questions: [
    {id: "team", type: CHOICE, instructions: "Which team?", criteria: {billing: "Payments", tech: "Bugs"}, task: "router"}
    {id: "refund", type: NOUL, instructions: "Refund requested?"}
  ]) { requestId answers { questionId decisionId answer confidence probabilities { option probability } } }
}
mutation { feedback(decisionId: "dec_…", label: "tech", comment: "…") { learned taskExamples adapterActive } }
query { tasks { task examples active baseLoss adaptedLoss } decision(id: "dec_…") { state prompt feedback } }
```

## Typed decisions in Rust

`cargo add cev-rs`. The crate is published as `cev-rs` and imported as `cev`.

```rust
#[derive(cev::Choice, Debug, Clone, Copy, PartialEq)]
#[cev(instructions = "Which team should handle this ticket?", task = "router.team")]
enum Team {
    /// Payments, invoices, refunds
    Billing,
    /// Bugs, crashes, error messages
    Engineering,
    #[cev(rename = "sales", description = "Pricing and upgrades")]
    Sales,
}

let cev = cev::Cev::http("http://127.0.0.1:8080");      // or Cev::local(runtime) in-process

if *cev.check(&ticket, "Is the customer asking for a refund?").await? { /* … */ }

let team = cev.pick::<Team>(&ticket).await?;              // Decided<Team>, derefs to Team
match *team {
    Team::Billing => {}
    Team::Engineering => {}
    Team::Sales => {}
}
if team.confident(0.8).is_none() { /* route to a human */ }
team.correct(Team::Engineering).await?;                   // typed feedback via decision_id

// several questions, one forward pass
let mut q = cev.ask(&ticket);
let (t, s, r) = (q.choose::<Team>("Which team?"), q.score::<Severity>("How severe?"), q.check("Refund?"));
let a = q.send().await?;
let (team, severity, refund) = (a.get(t)?, a.get(s)?, a.get(r)?);
```

How the derive maps an enum:

- Variant names become option names (snake_case, or overridden with `rename`).
- Doc comments become option descriptions.
- For scores, variant order is the level order.

A full in-process example: `cargo run --release -p cev-rs --features local,metal --example router -- Qwen/Qwen3-1.7B`.

## Layout

| Crate | |
|---|---|
| `cev-core` | Wire types, prompt compiler, answer math, label → target, the `Backend` trait and a mock |
| `cev-model` | candle Qwen3 (prefill-only, shared-prefix KV, packed block-masked questions), weight resolution |
| `cev-runtime` | The decision service: SQLite log (in-memory on wasm), feedback, online adapters, export |
| `cev-server` | `cev` binary: axum REST + async-graphql |
| `cev-wasm`, `web/` | Browser demo: the same pipeline as a wasm module in a web worker, in-memory log |
| `cev-rs`, `cev-derive` | Typed SDK, imported as `cev` (`http` and `local` transports) and `#[derive(Choice)]` |
| `training/` | PyTorch: `train_lora.py` (fine-tune from export), `parity.py` (check cev against transformers) |

Tests:

- `cargo test --workspace --features cev-rs/local` runs everything against the mock backend, with no weights needed.
- `training/parity.py` checks the real network against transformers.

## Status / next

- Qwen3 dense only. Next: quantized (GGUF) Qwen3, Qwen3-MoE, and Qwen3.5 (hybrid DeltaNet), which the kev weights use.
- An explicit "insufficient evidence" option, and building the attention mask on the GPU for very long states.
- Larger option sets through an embedding shortlist before scoring.
- Per-tenant isolation for adapters and logs.
