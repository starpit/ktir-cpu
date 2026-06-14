# Multi-core (Spyre-grid) GPU execution — design & plan

> "Spyre gpus" here = the **multi-core SPMD grid** (compute tiles). This is the
> plan to actually run that grid on the Metal GPU, instead of collapsing it.

## Why this is critical (load-bearing, not a side optimization)

Spyre's value proposition is **multi-core parallelism**. Today the emulator
leverages it for **nothing** on the GPU:

- **Non-attention nodes** — the optimizer collapses the grid to `[1,1]` and runs
  one big GEMM. Fast + correct, but deliberately discards the multi-core
  structure ("pretend it's one giant core").
- **Attention** (the genuinely multi-core part) — keeps its `[32,1]` head-parallel
  grid but runs **on the CPU interpreter, core-by-core, serially.**

So the multi-core SPMD path is never exercised on the Metal GPU. This work makes
it a first-class, GPU-resident citizen. It is also the **foundation for
multi-device** sharding (a layer above: the compiler shards per device; the
emulator runs each shard, which is itself a multi-core grid program).

## The measured bottleneck (this is a PREFILL win)

llama-3.2-1b **prefill** (`KTIR_SEG_DIAG`, committed `dc4def2`):

| segment class | per pass | where |
|---|---:|---|
| native attention (16 segs, grid `[32,1]`) | **~3040 ms (≈85%)** | CPU interpreter |
| fused GEMM/map | ~530–660 ms | GPU (NAX/AMX) |

- ~10K tiny per-(head,op) interpreter executions, incl. **1024 tiny GEMMs**
  (32 matmuls/head × 32 heads).
- **Decode** = `1 fused + 0 native` segments → single-core attention, already
  fast (~591 ms), **unaffected** (the batched path gates on `num_cores > 1`).
- Ceiling: attention → GEMM floor ⇒ **~5–6× e2e prefill** (3570 ms → ~600–700 ms).

## Why it's tractable (the key findings)

1. **One-op seed.** `ktdp.get_compute_tile_id` (→ `ctx.get_grid_id`) is the ONLY
   per-core divergence — ops, arg→HBM bindings, dispatch are byte-identical
   across cores. Replace that one scalar with a per-batch **iota** `[0..N)` and
   the whole SPMD body becomes batched ops, with **zero attention knowledge**.
2. **Comm-free.** `is_comm_op` matches only `ktdp.reduce`, which attention never
   uses → cores fully independent → batching is order-independent and correct.
   Per-head HBM slices are disjoint → the batched scatter is race-free.
3. **The gate inverts cleanly.** The `num_cores == 1` gate that makes full-shape
   reconstruction *wrong* at multi-core becomes *correct* once `[num_cores]` is an
   explicit leading batch/head dim — the batch dim **is** the head tiling.
4. **A proof-of-concept already exists.** `execute_function_gpu` /
   `try_combine_matmul` already stacks per-core A-panels into one tall GEMM for
   *shared-weight* matmuls; it bails on the **per-head gather** — which is the one
   piece of new machinery this needs.
5. **Collapse caveat.** Shared-weight layer GEMMs are *faster collapsed* to one
   big GEMM on a single M5 (NAX wants big GEMMs) — keep collapsing those. The
   grid-as-batch leverage pays only for **disjoint, non-collapsible** per-core
   work (attention, and the general SPMD case).

## Architecture

A new `execute_function_batched` (sibling of `execute_function_gpu`,
`interpreter.rs:529`), invoked FIRST from the `Segment::Native` arm
(`segmented.rs:229-259`) and the resident native path; on `Err` it falls back to
the unchanged per-core `execute_function` / `comm_sched` — so a wrong shape never
yields a wrong answer (pure accelerator with transparent fall-through).

- **Gate:** `num_cores > 1` AND comm-free (`ops.iter().all(|o| !is_comm_op(...))`)
  AND region-free AND NAX present. Does NOT touch `gpu_base` / the single-core
  fused path / decode / the K-loop full-M GEMM offload.
- **`BatchedContext`:** per SSA name, either a **core-invariant** scalar/index
  (bound identically to every core) or a **batched** `[num_cores, ...]` value.
  `get_compute_tile_id` → the iota seed (1-D: `b`; k-D: `grid.linear_to_grid(b)`
  per dim).
- **The new machinery — per-pid gather/scatter:** `construct_access_tile` with the
  per-batch index vector → eval `base_map` per batch elem → a **vector of
  base_ptrs**; `ktdp.load` → **batched gather** stacking all heads' slices into the
  `[num_cores, ...]` leading dim (core-invariant bases — mask, GQA-shared K/V —
  are a read-broadcast); `ktdp.store` → **batched scatter** (disjoint per head).

## Increment ladder (each golden-gated; Err → interpreter fallback)

- **INC-0** — per-op timer in the native path to attribute the 3040 ms across op
  classes (confirm matmuls dominate before optimizing). No behavior change.
- **INC-1 (FIRST)** — batch ONLY the two matmul classes (QK^T, A@V) via
  `NaxGemm::run_batched`; everything else per-core fallback. Builds the iota +
  gather/scatter. Collapses 1024 tiny GEMMs → a few. Flip the gate for comm-free
  multi-core; single-core path untouched.
- **INC-2** — batched softmax: row-reduce max/sum folded to `B*rows` dispatch
  width (generalize `run_reduce_gpu`); sub/exp/div as flat-gid map windows
  (`render_kernel` is rank-agnostic).
- **INC-3** — batched transpose (`run_transpose_gpu` is any-rank) + batched
  elementwise (scale mulf, mask addf, chunk-combine) over `[N, ...]`.
- **INC-4** — whole-segment GPU-resident: gather once, chain QK^T→softmax→A@V
  on-device (UnifiedBuffer / scratch ping-pong, `run_chain` pattern
  `metal_backend.rs:3512`), scatter once. NOT LX.
- **INC-5 (optional)** — one fused batched-softmax kernel (max+sub+exp+sum+div,
  threadgroup-per-(batch,row)) if dispatch count still matters after INC-4.

## Correctness model (verified — workflow `wan3dglws`)

**HOLDS:** per-head independence (comm-free), grid→batch index mapping, GQA
grouping (`(hpid/4)*64` = safe read-broadcast over 8 disjoint kv-heads), causal
mask, output layout (`view1[r, hpid*64]`), decode M=1 unaffected, K-loop full-M
GEMM untouched.

**3 to resolve before/during INC-1:**
1. **f16 drift** — tiny `m=1` attention GEMMs through `run_batched` half-round
   inputs; match the interpreter's f32-accumulate discipline or size-gate them
   (golden tol 0.05).
2. **1/√d scale** — keep it a `Scalar` (the `arith.mulf` Tile×Scalar arm
   broadcasts over any `[N,...]`) rather than a splatted Tile, OR make
   `tensor.splat` batch-aware.
3. **(framing)** `execute_function_gpu` is the existing *multi-core peer*, not
   "the single-core path." Assert the real invariant: `gpu_base` /
   `matmul_loop_schedule` (single-core fused + decode + K-loop GEMM) are untouched.

## Constraints / risks

- **LX pressure:** `[num_cores, ...]` tiles are N× a per-core tile → blow the
  per-core LX cap. Keep batched intermediates HBM-backed (INC-1) / device-resident
  (INC-4), not LX; `dies_at` ports but the budget must be N-scaled.
- **`run_batched` upload:** uses `newBufferWithBytes` (fresh host→device per call,
  `metal_backend.rs:3669`); INC-1 pays one upload of the gathered inputs, INC-4
  switches to UnifiedBuffer/scratch.
- **Ragged diagonal chunk:** the `(qrow+1)` K/V load varies per unrolled query
  row — batch each of the 8 unrolled blocks over the 32 heads (no row-loop
  reconstruction).
- **k-D grid iota:** general path must decompose via `linear_to_grid` per dim
  (attention is `[32,1]` = 1-D, safe for INC-1).
- **Gather/scatter disjointness:** correctness of order-independent batching needs
  disjoint per-head slices (RFC 0682: overlapping coordinate sets are
  *unspecified*); verify, or restrict to the proven affine-stride output layout.
  GQA-shared K/V is read-only → overlap safe.
- **RFC 0682 conformance:** execution-backend choice ONLY — no new `ktdp` ops, no
  `construct_memory_view` allocation, no op-semantic changes; Err→interpreter
  fallback guarantees a wrong shape never yields a wrong answer.

## Key file references

- `comm_sched.rs:238-239` (`gpu_base` gate), `:33-35` (`is_comm_op`),
  `:563-583` (per-core loop)
- `interpreter.rs:529` (`execute_function_gpu`), `:594` (`try_combine_matmul`)
- `metal_backend.rs:3649` (`NaxGemm::run_batched`), `:3714` (grid z = num_cores),
  `:3512` (`run_chain`)
- `ktdp_extra.rs:59-77` (`get_compute_tile_id`), `context.rs:80-87` (`get_grid_id`)
- `ktdp.rs:106-179` (`construct_access_tile`), `ops_memory.rs:828/887`
  (`load`/`store`)
- `segmented.rs:229-259` (`Segment::Native` arm); the resident native arm in
  `resident.rs`
- `~/.cache/cudaforge/ktir/llama-3.2-1b-prefill/node111.mlir` (representative
  attention node, grid `[32,1]`)

## Provenance

Design + adversarial verification: workflow `wan3dglws`. Bottleneck measurement:
`KTIR_SEG_DIAG` (resident seg-diag), committed `dc4def2`. Status: design complete,
green-light pending the 3 fixes above; **not yet implemented** (INC-0 is the next
step).
