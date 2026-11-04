#!/usr/bin/env bash
# Latency of each decision model through the server, on this machine's
# accelerator: starts `cev --model <name>`, waits for /health, runs
# scripts/bench.py (3 questions; short and ~2k-token states; prefix cached and
# uncached), stops it. Models are downloaded on first use.
#
#   scripts/bench_models.sh [features] [model ...]
#   scripts/bench_models.sh metal kev:0.8b d3:2b d1 d1-omni Qwen/Qwen3.5-0.8B
set -euo pipefail
cd "$(dirname "$0")/.."
features="${1:-metal}"
shift || true
models=("$@")
[ ${#models[@]} -eq 0 ] && models=(kev:0.8b d3:2b d1 d1-omni Qwen/Qwen3.5-0.8B)
cargo build --release -p cev-server --features "$features" >/dev/null
for m in "${models[@]}"; do
  ./target/release/cev --model "$m" --db "$(mktemp -d)/bench.db" --addr 127.0.0.1:8099 >"${TMPDIR:-/tmp}/cev-bench-$$.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 600); do curl -sf http://127.0.0.1:8099/health >/dev/null 2>&1 && break; sleep 1; done
  echo "== $m: $(curl -s http://127.0.0.1:8099/v1/models | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d["id"], d["family"], d["modalities"])')"
  python3 scripts/bench.py http://127.0.0.1:8099 || true
  kill "$pid"; wait "$pid" 2>/dev/null || true
done
