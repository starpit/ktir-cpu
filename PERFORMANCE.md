# KTIR Rust Performance

A trackable record of the **Rust** execution layer's performance over time, with
the two compute backends broken out:

- **AMX** — Apple Accelerate (`cblas_sgemm`, the AMX matrix coprocessor, f32) for
  matmul, and the naive/CPU interpreter for elementwise / layernorm.
- **Metal** — the M5 GPU: the NAX tensor engine (`NaxGemm`, f16) for matmul, the
  fused map-window MSL kernels for elementwise, and head-parallel attention via
  the segmented executor.

The Python reference interpreter numbers are carried forward verbatim (they are
current and are not re-run here).

## Machine

| | |
|---|---|
| Host | Apple **M5** (Mac17,2) — has AMX/Accelerate **and** a Metal GPU (NAX tensor engine) |
| Toolchain | `rustc 1.96.0` (stable-aarch64-apple-darwin) |
| Metal | `cfg(metal)` is auto-enabled on macOS by `build.rs` (no feature flag needed); the `--features metal` flag is for cross builds. **Every macOS build is a Metal build.** |

## How to regenerate

```bash
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"
cd rust
# RULE: run benches STRICTLY ONE AT A TIME (build first; never two benches, or a
# bench + a build, concurrently — contention skews wall-clock). A warm-up pass is
# excluded inside every harness. Use --test-threads=1 for the cargo-test benches.

# --- Per-kernel ---------------------------------------------------------------
# AMX/CPU path (the interpreter; matmul tiles route to Accelerate below the NAX gate):
cargo test --release -p ktir-cpu --test bench_py_vs_rust \
    -- --ignored --nocapture --test-threads=1
# AMX-vs-Metal matmul PRIMITIVE (blas::sgemm_rowmajor vs NaxGemm::run, full GEMM shape):
cargo test --release -p ktir-cpu --features metal --test bench_amx_vs_metal \
    -- --ignored --nocapture --test-threads=1

# --- E2E whole-model (ms/pass) ------------------------------------------------
# per-node (the optimized interpreter, no whole-program fusion):
MODEL=smollm2-135m         SMOLLM2_ITERS=40 cargo run  --release -p ktir-cpu --bench smollm2_bench --features metal
MODEL=smollm2-135m-prefill SMOLLM2_ITERS=20 cargo run  --release -p ktir-cpu --bench smollm2_bench --features metal
MODEL=llama-3.2-1b         SMOLLM2_ITERS=3  cargo run  --release -p ktir-cpu --bench smollm2_bench --features metal
MODEL=llama-3.2-1b-prefill SMOLLM2_ITERS=2  cargo run  --release -p ktir-cpu --bench smollm2_bench --features metal

# fused-AMX (segmented executor, GPU offloads OFF):
BUNDLE=<model> ITERS=<n> KTIR_NO_GPU_GEMM=1 KTIR_NO_GPU_MAP=1 \
    cargo test --release -p ktir-cpu --features metal --test fuse_run_smollm2 \
    segmented_mspass -- --ignored --nocapture --test-threads=1
# fused-Metal (segmented executor, GPU offloads ON — drop the KTIR_NO_GPU_* toggles):
BUNDLE=<model> ITERS=<n> \
    cargo test --release -p ktir-cpu --features metal --test fuse_run_smollm2 \
    segmented_mspass -- --ignored --nocapture --test-threads=1
# BUNDLE ∈ {smollm2-135m, smollm2-135m-prefill, llama-3.2-1b, llama-3.2-1b-prefill}
# (bundles live under ~/.cache/cudaforge/ktir/<bundle>/; not in the repo).
```

GPU-offload toggles (read by `comm_sched` / `metal_backend`):

| Env | Effect when set |
|---|---|
| `KTIR_NO_GPU_GEMM=1` | K-loop GEMMs stay on Accelerate (no NAX) |
| `KTIR_NO_GPU_MAP=1`  | map-window elementwise stays on the CPU interpreter |
| `KTIR_GPU_PLAIN_MATMUL` / `KTIR_GPU_REDUCE` / `KTIR_GPU_TRANSPOSE` | opt-in attention-island offloads (default OFF; a net loss on tiny attention tensors) |

---

## Latest snapshot (2026-06-14, commit `0017fc3`)

### Per-kernel

The first three rows are the Rust **interpreter** path (`execute_function`); on
this M5 it routes matmul tiles to Accelerate (the per-tile 32×128@128×512 blocks
are below the NAX gate `NAX_MIN_BLOCKS=32`) and elementwise/layernorm to the CPU,
so they are the **AMX/CPU** column. There is no Metal path for these kernels
*through the interpreter* at these shapes, so the last two columns time the matmul
**primitives directly** at the kernel's logical full-GEMM shape (the apples-to-
apples AMX-vs-Metal comparison the kernel's `linalg.matmul` would make if it were
one big GEMM instead of a tiled SPMD K-loop).

| Kernel | Python | Rust AMX/CPU | speedup vs Py | Rust Metal | AMX→Metal |
|---|---:|---:|---:|---:|---:|
| vector_add (n=4096, f16) | 861 µs | **630.9 µs** | 1.37× | CPU only at this size¹ | — |
| matmul (64×2048×8192, f16) — interpreter (tiled SPMD K-loop) | 11.95 s | **428.5 ms** | 27.9× | n/a (tiles below NAX gate)² | — |
| layernorm (1151×8192, f16) | 41.9 s | **695.1 ms** | 60.3× | CPU only at this size¹ | — |
| **matmul PRIMITIVE 64×2048×8192** (sgemm vs NaxGemm) | — | **3.75 ms** (573 GFLOP/s) | — | **2.93 ms** (733 GFLOP/s) | **1.28×** |
| **matmul PRIMITIVE 512×4096×4096** (prefill-scale) | — | **12.53 ms** (1371 GFLOP/s) | — | **6.55 ms** (2622 GFLOP/s) | **1.91×** |

¹ The map-window GPU offload only fires inside a single-core fused function; the
  standalone elementwise/layernorm kernels run on a multi-core SPMD grid and stay
  on the CPU interpreter (`KTIR_NO_GPU_MAP` path). At n=4096 / 1151×8192 the GPU
  dispatch latency would not pay off anyway.
² The matmul kernel's inner tiles are tiny, so even on Metal they route to
  Accelerate. The 428.5 ms is the *interpreter* path (32-core SPMD × K-tiles, a
  small Accelerate call + marshaling per tile) — ~100× the 3.75 ms raw primitive,
  i.e. per-tile dispatch/marshal overhead dominates this microbench. The
  primitive rows below it are the real AMX-vs-Metal matmul comparison.

### E2E whole-model (ms/pass; lower is better)

Three Rust paths: **per-node** (optimized interpreter, no whole-program fusion),
**fused-AMX** (segmented executor, GPU offloads OFF), **fused-Metal** (segmented
executor, GPU offloads ON: K-loop NAX GEMMs + map-window kernels + resident
weight cache; head-parallel attention via native-grid segments).

| Model / mode | Python | Rust per-node | Rust fused-AMX | Rust fused-Metal | fused Metal→AMX | golden max-abs | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| smollm2-135m **decode** | 2397 ms | 704.9 ms | **245.5 ms** | 418.6 ms | 0.59× (Metal slower) | 0.0014 | tiny M=1 GEMMs; GPU dispatch-bound → AMX wins |
| smollm2-135m **prefill** (M=8) | — | 1271.8 ms | **679.4 ms** | 813.6 ms | 0.83× (Metal slower) | 0.0035 | still too small to clear the GPU crossover |
| llama-3.2-1b **decode** | — | 3089.2 ms | **8717.2 ms** | 24918.4 ms | 0.35× (Metal slower) | 0.0026 | M=1; every layer pays NAX dispatch latency → big loss |
| llama-3.2-1b **prefill** (M=32) | — | 29132.1 ms | 7976.8 ms | **6269.6 ms** | **1.27× (Metal wins)** | 0.0033 | large M fills the GPU → the offloads finally pay off |

E2E speedups vs Python (smollm2-135m decode, the only mode with a Python e2e
number): per-node **3.4×**, fused-AMX **9.8×**, fused-Metal **5.7×**.

Iters were kept modest on the big model per the one-at-a-time rule: llama decode
ITERS=3, llama prefill ITERS=2 (a 2-sample median ≈ the faster of two passes).
All golden max-abs figures are the known-current correctness numbers (f16/GPU
noise, well inside the gates: smollm2 ≤0.0035, llama ≤0.05).

---

## Interpretation

- **Metal wins exactly where the use case lives — big-model prefill.** On
  llama-3.2-1b prefill (M=32, the GEMMs are large enough to fill the GPU and dwarf
  its ~300 µs/dispatch submission latency) fused-Metal is **1.27× faster** than
  fused-AMX, and the raw NAX matmul primitive is **1.9× faster** than Accelerate
  at prefill scale (512×4096×4096). This is the throughput target.
- **Metal loses on decode (M=1) regardless of model size.** Single-token decode
  produces tall-skinny 1×K@K×N GEMMs that leave the GPU mostly idle while still
  paying full dispatch+sync latency per layer, so fused-Metal is *slower* than
  fused-AMX on every decode config (0.59× smollm2, 0.35× llama-1B). For decode,
  **fused-AMX is the right path**; the GPU offloads should be left off.
- **Fusion beats per-node almost everywhere; the exception is llama decode.**
  Segmenting/fusing the program cuts per-node dispatch+marshal overhead (smollm2
  decode 705→246 ms, prefill 1272→679 ms; llama-1B prefill 29.1 s→8.0 s with AMX).
  The one inversion is llama-1B *decode*, where per-node (3.1 s) beats fused-AMX
  (8.7 s) — the fused [1,1] segments serialize the big weight GEMMs that the
  per-node path runs at its native token-parallel grid; worth a follow-up.
- **Emulation-overhead caveat.** The matmul microbench shows the interpreter's
  tiled SPMD path (428 ms) is ~100× the raw primitive (3.75 ms): most per-kernel
  time on small tensors is dispatch/marshal, not arithmetic. The fastest backend
  is only as fast as the path feeding it — which is why fusion (keeping data
  resident, batching GEMMs) matters more than the backend choice on small shapes.

---

## History

Append-only. Each snapshot is one block; newest on top. `n/a` = not measured that
snapshot; `—` = not applicable.

### 2026-06-14 · commit `0017fc3` · Apple M5

Per-kernel (Rust AMX/CPU interpreter): vector_add 630.9 µs (1.37× Py) · matmul
428.5 ms (27.9× Py) · layernorm 695.1 ms (60.3× Py). matmul primitive
AMX→Metal: 1.28× at 64×2048×8192, 1.91× at 512×4096×4096.

E2E ms/pass (per-node / fused-AMX / fused-Metal):

| Model/mode | per-node | fused-AMX | fused-Metal |
|---|---:|---:|---:|
| smollm2-135m decode  | 704.9 | 245.5 | 418.6 |
| smollm2-135m prefill | 1271.8 | 679.4 | 813.6 |
| llama-3.2-1b decode  | 3089.2 | 8717.2 | 24918.4 |
| llama-3.2-1b prefill | 29132.1 | 7976.8 | **6269.6** |

Headline: fused-Metal wins on llama-1B prefill (1.27× vs fused-AMX); AMX wins on
all decode and on small-model prefill (GPU dispatch-bound at M=1 / small M).

### v2 (earliest reference, pre-workspace-split) · Apple M5

The prior tracked Rust baseline, for regression context. Per-kernel speedups vs
Python: vector_add 5.6× · matmul 27.4× · layernorm 59×. E2E SmolLM2-135M decode
= **245 ms/pass** (9.8× vs Python). Note: that 245 ms matches this snapshot's
**fused-AMX** smollm2 decode (245.5 ms) — i.e. the historical "245" was the
segmented/fused-AMX path, not the per-node interpreter (which is 705 ms here on
the current Metal-by-default build). No AMX-vs-Metal split was recorded at v2.
