#!/usr/bin/env bash
# One-shot installer for soda-tts. Builds the Rust binaries, fetches model
# files, converts ONNX -> safetensors (for the candle backend), and verifies
# the conversion. Idempotent: rerun freely.
#
# Prereqs: rust toolchain, curl, and `uv` (https://docs.astral.sh/uv) -- uv
# manages the small Python toolchain in an ephemeral env per invocation, so
# nothing gets installed to your system Python.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# --- 0. Sanity checks ------------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: need '$1' on PATH"; echo "$2"; exit 1; }; }
need cargo  "Install rust via https://rustup.rs/"
need curl   "Install curl from your package manager"
need uv     "Install uv via 'curl -LsSf https://astral.sh/uv/install.sh | sh' (or see https://docs.astral.sh/uv/)"

# `uv run --with X --with Y python ...` runs the script in a temp venv that
# contains those packages, with no global install. Idempotent.
UV_RUN=(uv run --quiet --with onnx --with safetensors --with numpy python)

echo "==> [1/4] cargo build --release"
cargo build --release \
    --bin speak --bin speak_candle --bin download-model

echo
echo "==> [2/4] downloading ONNX model + voice styles (~400 MB)"
echo "    (skipped if models/onnx/vocoder.onnx already exists)"
if [ ! -f models/onnx/vocoder.onnx ]; then
    ./target/release/download-model
else
    echo "    models/onnx/vocoder.onnx is already present, skipping"
fi

echo
echo "==> [3/4] converting ONNX -> safetensors for the candle backend"
"${UV_RUN[@]}" tools/convert_weights.py

echo
echo "==> [4/4] verifying every tensor is byte-identical to the ONNX source"
"${UV_RUN[@]}" tools/verify_weights.py | tail -1

echo
echo "==> setup complete. Try:"
echo
echo "    ./target/release/speak --text 'Hello, world.' --out hello.wav"
echo "    ./target/release/speak_candle --text 'Hello, world.' --out hello.wav"
echo
