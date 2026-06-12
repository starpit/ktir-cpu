#!/usr/bin/env bash
# End-to-end model sweep: run each compiled model under ~/.cache/cudaforge/ktir/
# through the Rust interpreter and report ms per full-model pass.
#
# DECODE: the SmolLM2-135M models below are all single-token decode (one is fp16,
# the rest are quantizations — they run identically because the quantization is
# baked into the weights, not new ops).
#
# PREFILL: add prefill model dirs to PREFILL_MODELS once a prefill model is
# compiled with cudaforge (the runner is shape-agnostic — same harness). None are
# present in the cache today, so that leg is empty until a bundle is generated.
#
# Usage:  ./bench_e2e_sweep.sh            # default decode sweep
#         ITERS=20 ./bench_e2e_sweep.sh   # more passes per model
set -euo pipefail
cd "$(dirname "$0")/rust"
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"
ITERS="${ITERS:-10}"

DECODE_MODELS=(
  smollm2-135m
  smollm2-135m-awq-gemm
  smollm2-135m-bnb-nf4-dq
  smollm2-135m-ct-int4-sym
  smollm2-135m-fp8-dynamic-per-tensor
  smollm2-135m-ggml
  smollm2-135m-mlx-affine-b4-g64
  smollm2-135m-mlx-affine-b4-g64-qembed
  smollm2-135m-nvfp4
)
PREFILL_MODELS=()  # populate once a prefill model is compiled

cargo build --release --bench smollm2_bench >/dev/null 2>&1
BIN=$(printf '%s\n' target/release/deps/smollm2_bench-* | grep -v '\.d$' | head -1)

run_leg() {
  local label="$1"; shift
  local models=("$@")
  [ ${#models[@]} -eq 0 ] && { echo "  ($label: no models present)"; return; }
  echo "=== $label e2e (Rust, $ITERS passes each) ==="
  for m in "${models[@]}"; do
    # Models run one at a time — never concurrently — so timings are clean.
    r=$(MODEL="$m" SMOLLM2_ITERS="$ITERS" "$BIN" 2>&1 | grep -oE '[0-9.]+ ms/pass.*' | head -1)
    printf "  %-42s %s\n" "$m" "${r:-<error / absent>}"
  done
}

run_leg DECODE "${DECODE_MODELS[@]}"
run_leg PREFILL "${PREFILL_MODELS[@]}"
