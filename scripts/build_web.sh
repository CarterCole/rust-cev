#!/bin/sh
# Build the browser demo into web/pkg, then serve web/ with any static server:
#
#     scripts/build_web.sh && python3 -m http.server -d web 8787
#
# Weights come from web/models/<name>/ when present (e.g. a symlink to a
# Hugging Face snapshot directory), otherwise from huggingface.co.
set -eu
cd "$(dirname "$0")/.."

rustup target list --installed | grep -q wasm32-unknown-unknown || rustup target add wasm32-unknown-unknown
cargo build -p cev-wasm --release --target wasm32-unknown-unknown

# wasm-bindgen's CLI must match the crate version exactly; keep a private copy.
want=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/[^0-9.]/, ""); print; exit }' Cargo.lock)
bindgen=wasm-bindgen
if [ "$(wasm-bindgen --version 2>/dev/null | awk '{print $2}')" != "$want" ]; then
    bindgen=target/wasm-tools/bin/wasm-bindgen
    [ "$($bindgen --version 2>/dev/null | awk '{print $2}')" = "$want" ] ||
        cargo install wasm-bindgen-cli --version "$want" --root target/wasm-tools --locked
fi
"$bindgen" --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/cev_wasm.wasm
ls -lh web/pkg/cev_wasm_bg.wasm
