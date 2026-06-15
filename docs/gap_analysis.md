# KTIR Spec RFC vs. `ktir_cpu` Implementation — Gap Analysis

**Date**: 2026-04-22
**Spec**: [RFC 0682 — KTIR Spec](https://github.com/torch-spyre/RFCs/blob/main/0682-KtirSpec/0682-KtirSpecRFC.md)

**Legend**: ✅ implemented — 🟡 partial — ❌ not implemented

---

## A. `ktdp` Dialect Operations

| # | Spec Operation | Status | Notes |
|---|---------------|--------|-------|
| 1 | `ktdp.construct_distributed_memory_view` | ✅ | Handler and parser implemented in `ktir_cpu/dialects/ktdp_ops.py`; produces `DistributedMemRef` (composition of N per-partition `MemRef`s). Per-partition routing at access time via `MemoryOps.distributed_tile_access` → `DistributedTileRef`. Tests in `tests/test_distributed_view.py`. |
| 2 | `ktdp.construct_indirect_access_tile` | ✅ | Handler and parser implemented in `ktir_cpu/dialects/ktdp_ops.py`; tests passing in `tests/test_indirect_access.py`. Both `ktdp.load` (gather, via `MemoryOps.indirect_load`) and `ktdp.store` (scatter, via `MemoryOps.indirect_store`) accept `IndirectAccessTile` (#44 closed). |

## B. `ktdp` Types & Attributes

| # | Spec Item | Status | Notes |
|---|-----------|--------|-------|
| 3 | `AccessTileType` with dynamic dimensions (`?`) | ❌ | The spec allows `access_tile<? x 64 x index>` (partially/fully dynamic shapes). The parser only extracts static integer dimensions — dynamic `?` dimensions are silently dropped. |
| 4 | `MemorySpaceAttr` (generic) | 🟡 | The parser extracts `SpyreMemorySpaceAttr` (`HBM`/`LX`), but the spec describes `MemorySpaceAttr` as a generic extensible wrapper that could encapsulate other hardware backends. The implementation hardcodes Spyre-specific memory spaces only. |

## C. Affine/Polyhedral Attributes

| # | Spec Attribute | Status | Notes |
|---|---------------|--------|-------|
| 5 | `coordinate_set` on `construct_memory_view` | 🟡 | Parsed and stored in `TileRef`. Not yet used to enforce coordinate constraints during load/store dispatch — the spec-required disjointness/overlap semantics are not enforced. |
| 6 | `access_tile_set` on `construct_access_tile` | ✅ | Parsed, stored in `AccessTile`, and used by `ktdp.load`/`ktdp.store` to enumerate coordinate tuples via the affine evaluator. |
| 7 | `access_tile_order` on `construct_access_tile` | ✅ | Parsed and stored; `ktdp.load`/`ktdp.store` apply the traversal order when iterating over coordinates. |
| 8 | `base_map` on `construct_access_tile` | ✅ | Parsed (identity map synthesized if absent); evaluated in `MemoryOps.tile_access` via the affine expression evaluator. |

**Note**: #6–8 and #33–35 are resolved — the interpreter uses affine coordinate sets and traversal order for load/store. Remaining gap is #5: `coordinate_set` on memory views is preserved but not enforced during access, so overlapping/disjoint distribution semantics are not yet checked.

## D. SCF Control-Flow Operations

The spec lists these SCF operations as "currently contemplated":

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 9 | `scf.reduce` | ❌ | Only `ktdp.reduce` exists for inter-core comm, which is semantically different |
| 10 | `scf.reduce.return` | ❌ | |
| 11 | `scf.parallel` | ❌ | |
| 12 | `scf.forall` | ❌ | |

Currently implemented: `scf.for`, `scf.if`, `scf.yield`.

## E. Standard MLIR Dialect Operations

This section is best read as a **coverage backlog**, not a list of equally
strong RFC violations. The RFC explicitly defines the `ktdp` surface and
explicitly calls out only a small subset of non-`ktdp` ops. Missing
`linalg.add`, `tensor.extract_slice`, and `memref.subview` are more directly
grounded in the RFC text than every absent op from the broader Arith/Math
dialects.

### Arith dialect

The spec references the [full Arith dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/). Currently implemented: `addf`, `subf`, `mulf`, `divf`, `addi`, `subi`, `muli`, `divui`, `remui`, `constant`, `maxf`, `maxnumf`, `extf`, `truncf`, `index_cast`, `sitofp`, `cmpi`, `select`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 13 | `arith.cmpf` | ❌ | float compare — only `arith.cmpi` (int compare) exists |
| 14 | `arith.negf` | ❌ | |
| 15 | `arith.absf` | ❌ | |
| 16 | `arith.minf` | ❌ | only `maxf` / `maxnumf` exist |
| 17 | `arith.minnumf` | ❌ | |
| 18 | `arith.fptosi`, `arith.fptoui`, `arith.uitofp` | 🟡 | only `sitofp` exists |
| 19 | `arith.divsi`, `arith.remsi`, `arith.andi`, `arith.ori`, `arith.xori`, `arith.ceildivsi`, `arith.floordivsi` | ❌ | only unsigned variants `divui`/`remui` exist |

### Math dialect

The spec references the [full Math dialect](https://mlir.llvm.org/docs/Dialects/MathOps/). Currently implemented: `math.exp`, `math.sqrt`, `math.log`, `math.rsqrt`, `math.log2`, `math.log1p`, `math.tanh`, `math.sin`, `math.cos`, `math.absf`, `math.absi`, `math.ceil`, `math.floor`, `math.erf`, `math.powf`, `math.fma`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 20 | `math.log2`, `math.log1p` | ✅ | |
| 21 | `math.tanh`, `math.sin`, `math.cos` | ✅ | |
| 22 | `math.rsqrt` | ✅ | |
| 23 | `math.absf`, `math.absi`, `math.ceil`, `math.floor` | ✅ | |
| 24 | `math.erf`, `math.powf`, `math.fma` | ✅ | `math.erf` uses polynomial approximation (no scipy) |

### Linalg dialect

The spec references the [full Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/). Currently implemented: `linalg.reduce`, `linalg.matmul`, `linalg.generic`, `linalg.broadcast`, `linalg.transpose`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 25 | `linalg.add` | ❌ | Used in the spec's primary matrix-add example — won't execute today |
| 26 | `linalg.generic` | ✅ | Full `bb0` block handling in `ktir_cpu/dialects/linalg_ops.py` |
| 27 | `linalg.map`, `linalg.broadcast`, `linalg.transpose` | 🟡 | `broadcast` and `transpose` implemented; `map` still missing |

### Tensor dialect

Currently implemented: `tensor.splat`, `tensor.extract`, `tensor.extract_slice`, `tensor.expand_shape`, `tensor.collapse_shape`.

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 28 | `tensor.extract_slice` | ✅ | Parser captures `[offsets][sizes][strides]` (static + dynamic SSA); handler materializes the strided sub-view. Closed in the fusion increment-2 work. |
| 29 | `tensor.insert_slice`, `tensor.collapse_shape` | 🟡 | `collapse_shape` implemented; `insert_slice` still missing |

### MemRef dialect

The spec explicitly mentions `memref.subview` for view-based transformations. **The entire `memref` dialect is absent from the implementation.**

| # | Operation | Status | Notes |
|---|-----------|--------|-------|
| 30 | `memref.subview` | ❌ | Spec explicitly calls this out |
| 31 | All other `memref` operations | ❌ | No `memref` dialect module exists |

## F. Semantic/Behavioral Gaps

| # | Gap | Status | Details |
|---|-----|--------|---------|
| 32 | `construct_memory_view` doesn't support mixed static+dynamic sizes/strides | 🟡 | The spec supports dynamic SSA operands for sizes/strides (`$sizes` as variadic index). The parser only handles static integer literals in `sizes: [96, 64]`. |
| 33 | `construct_access_tile` ignores base coordinate computation | ✅ | `base_map` is parsed and evaluated via the affine expression engine in `MemoryOps.tile_access`. |
| 34 | `ktdp.load` only implements rectangular slice semantics | ✅ | Now enumerates coordinates from `access_tile_set` and applies `access_tile_order`; supports general polyhedral regions. |
| 35 | `ktdp.store` only implements rectangular slice semantics | ✅ | Same coordinate-set enumeration as load. |
| 36 | `module { }` is tolerated, but module-level structure is not modeled | 🟡 | The parser can find `func.func` inside a `module { ... }` wrapper, but it does not model module-level attributes, declarations, or non-function top-level constructs. |

## G. Parser Limitations

| # | Gap | Status | Details |
|---|-----|--------|---------|
| 37 | No affine expression evaluation | ✅ | Full affine map and integer set parsing and evaluation implemented in `ktir_cpu/parser_ast.py` (`parse_affine_map`, `parse_affine_set`, `eval_affine_map`, `enumerate_affine_set`). |
| 38 | No `#alias = affine_set<...>` / `#alias = affine_map<...>` support | ✅ | Parser pre-scans module scope and populates an `aliases` dict; dialect parsers resolve aliases via `parse_ctx.aliases`. |
| 39 | `func.func` signature parsing is limited | 🟡 | The parser handles the basic typed signatures used in the shipped examples, but not the full MLIR function-signature space (richer types, attributes, or more complex declarative forms). |

## H. Priority Summary

### High Priority
Blocks running spec-compliant KTIR programs:

- **#25**: ❌ `linalg.add` — used in the spec's primary example and won't execute
- **#5**: 🟡 `coordinate_set` on memory views preserved but not enforced

### Medium Priority
Limits dialect coverage for real-world kernels:

- **#9–12**: ❌ SCF parallel/reduce operations
- **#13–19**: ❌/🟡 Many standard arith ops (cmpf, negf, absf, minf, signed int ops)
- **#20–24**: ✅ All math ops now implemented (log2, log1p, tanh, sin, cos, rsqrt, absf, ceil, floor, erf, powf, fma)
- **#28**: ✅ `tensor.extract_slice` implemented; **#30–31**: ❌ entire `memref` dialect
- **#32**: 🟡 Dynamic sizes/strides not supported

### Lower Priority
Extensibility and completeness:

- **#3, #4**: ❌/🟡 Dynamic access tile dimensions, generic `MemorySpaceAttr`
- **#27, #29**: 🟡 Remaining linalg/tensor ops (`linalg.map`, `tensor.insert_slice`)
- **#36, #39**: 🟡 Module-level handling, full function signatures

### Resolved
- **#2**: ✅ `construct_indirect_access_tile`
- **#6, #7, #8**: ✅ `access_tile_set`, `access_tile_order`, `base_map`
- **#20–24**: ✅ All math ops (rsqrt, log2, log1p, tanh, sin, cos, absf, ceil, floor, erf, powf, fma)
- **#26**: ✅ `linalg.generic`
- **#33, #34, #35**: ✅ Access tile coordinate semantics
- **#37, #38**: ✅ Affine expression evaluation and alias support

## I. Status as of 2026-04-22

Significant progress since the original writeup:

- `ktdp.construct_indirect_access_tile` (#2) is fully implemented with passing tests.
- The entire affine/polyhedral foundation (#6–8, #33–35, #37, #38) is now in
  place: affine maps and integer sets are parsed, evaluated, and used by
  `ktdp.load`/`ktdp.store` to enumerate coordinate tuples.
- `linalg.generic` (#26) is implemented with full `bb0` block handling.
- `linalg.broadcast`, `linalg.transpose` (#27 partial), `tensor.collapse_shape`
  (#29 partial), and `math.log` are now implemented.

Remaining notable gaps:

- `linalg.add` (#25) is still missing — the RFC's canonical matrix-add example
  cannot execute without it.
- `coordinate_set` on memory views (#5) is preserved in the IR but not used
  to enforce coordinate constraints during dispatch.
- SCF parallel/reduce ops (#9–12) and the entire `memref` dialect (#30–31)
  remain unimplemented. `tensor.extract_slice` (#28) is now implemented.

## J. Prioritized Conformance Roadmap

The initial phases of this roadmap are complete. The conformance target was established as "execute the RFC-defined `ktdp` subset plus the specific non-`ktdp` ops used by compiler-generated kernels." Spec gap tests were added to make missing coverage explicit. The access-tile foundation was then rebuilt: `base_map`, `access_tile_set`, `access_tile_order`, and `coordinate_set` are now parsed and preserved; a full affine/integer-set evaluator was implemented; and `ktdp.load`/`ktdp.store` iterate over affine coordinate tuples rather than rectangular subviews. `ktdp.construct_indirect_access_tile` was also completed as part of this work.

### Add The Missing KTDP Ops 🟡

Goal: cover the RFC-defined `ktdp` surface.

- ✅ Implement `ktdp.construct_indirect_access_tile`.
- ✅ Implement `ktdp.construct_distributed_memory_view`.
- Add validation rules for:
  matching dimensionalities,
  allowed direct versus indirect dimensions,
  and the RFC restriction that indirect indices are not further affine-scaled.
- Preserve dynamic `access_tile` dimensions (`?`) in the IR even if runtime
  support is initially partial.

### Close The RFC-Explicit Non-KTDP Gaps ❌

Goal: support the rest of the ops the RFC explicitly calls out.

- Add `linalg.add` so the RFC's canonical matrix-add example can execute
  without translation.
- ~~Add `tensor.extract_slice`.~~ Done (fusion increment 2).
- Add `memref.subview` and the minimal `memref` dialect support required to
  interpret it.
- Add the missing SCF ops explicitly named by the RFC:
  `scf.reduce`,
  `scf.reduce.return`,
  `scf.parallel`,
  and `scf.forall`.

### Widen Dialect Coverage Opportunistically ❌

Goal: improve practicality for real compiler output without pretending every
MLIR op is equally important for RFC conformance.

- Add only the Arith/Math/Linalg ops actually observed in upstream-generated
  KTIR or required by target workloads.
- Track these as "compiler coverage" rather than "spec blockers."
- Keep a small compatibility matrix in docs that separates:
  `RFC core`,
  `example coverage`,
  and `observed compiler output coverage`.

## K. Runtime / Simulation Correctness

These items concern the CPU simulator's fidelity rather than missing KTIR
spec surface.

### K1. Multi-round communication re-execution

**Status**: ✅ Fixed in PR-B (grid-network branch, issue #50).

`execute_with_communication` now uses a generator-based cooperative scheduler.
Each core runs as a Python generator via `CoreExecutionStack`; blocking `recv`
operations suspend the generator (`yield RecvRequest(src)`) until the expected
tile is delivered. No BSP replay — each core executes exactly once.

See `docs/cross_core_scheduling.md` for the full design.

### K2. Cyclic communication correctness

**Status**: ✅ Fixed in PR-B (grid-network branch, issue #50).

`CommOps.reduce` is now a generator that yields `RecvRequest` per ring round.
The scheduler drives it to completion via `gen.send(tile)`, consuming each
message exactly once in order. No duplicate sends, no message loss.
Bidirectional exchanges (both cores send then recv) are handled correctly
because `send_to` is fire-and-forget — the sender enqueues and continues
without blocking, so symmetric patterns never deadlock.

### K3. Multi-cast load modeling

**Status**: ❌ Not modeled. No existing kernels require it.

There is currently no model for multi-cast loads where one ring-bus
transaction serves multiple cores simultaneously. Two variations exist:

- LX-to-LX memory transfer (unicast or multi-cast)
- HBM-to-LX multi-cast load

The kernel optimizer would need to annotate `ktdp.load` with a
participating-core group attribute so the latency calculator can account for
the shared transaction cost. This is a future design question.

### K4. Head-parallel attention under whole-program fusion

**Status**: ✅ Fixed (partial fusion in `ktir_optimizer::fusion::plan_segments`).

Whole-program function fusion (`fuse_program`) collapses a multi-node KTIR
program into ONE function and stamps that function's grid from the first node
(`[1,1]`). Head-parallel ATTENTION nodes select their head with
`ktdp.get_compute_tile_id` against a multi-head grid (`[32,1]` for Llama-3.2-1B,
`[9,1]` for SmolLM2-135M); at the collapsed `[1,1]` grid that primitive returns
0, so only head 0's slice was computed and the prefill result diverged from
golden (Llama-1B prefill 0.0598, over the 0.05 gate).

`plan_segments` does PARTIAL fusion: it leaves each attention node (detected by
a non-trivial grid PLUS the attention op signature — a `linalg.transpose` and
the softmax `linalg.reduce { arith.maximumf }`) as a standalone segment run at
its NATIVE multi-core grid (the proven-correct per-node SPMD path), and fuses
each maximal run of consecutive non-attention nodes into a single `[1,1]`
function, threading intermediates through HBM between segments in program order.
The fused segments keep all GPU offloads (matmul-loop GEMM, map-window fusion,
weight cache); the segment grid is forced to `[1,1]` so the token-parallel
`[8,1]` matmul nodes folded in collapse to one reconstructed GEMM (the
single-core K-loop offload). Result: Llama-1B prefill 0.00326, SmolLM2-135M
prefill 0.00348 (both matching the per-node oracle ~0.003), decode unchanged.

### K5. Multi-core attention on the GPU (head-batched SPMD)

**Status**: 🟡 Partial (INC-0 + INC-1; `interpreter::execute_function_batched`).

K4 runs each head-parallel attention node correctly but on the CPU interpreter,
core-by-core: ~10K tiny per-`(head, op)` executions per pass including 1024 tiny
GEMMs (32 matmuls/head × 32 heads), which `KTIR_SEG_DIAG` measured as ≈85% of the
Llama-3.2-1B prefill pass. `execute_function_batched` is the inverse-sense sibling
of `execute_function_gpu`: the `[num_cores]` grid dim IS the head tiling, so it
steps every head's core in lockstep and collapses the per-head QK^T / A·V matmul
class into one batched `NaxGemm::run_batched` dispatch each (grid z = head), with
the per-pid gather/scatter falling out of the unchanged per-core `ktdp.load` /
`ktdp.store`. It INHERITS the `execute_function_gpu` region-free + comm-free gate
(plus `num_cores > 1`), so a flash-attention `scf.for` node (K6) or any
region-bearing body returns `Err` → transparent per-core interpreter fallback; the
single-core fused / decode / K-loop full-M GEMM paths are untouched. Wired
wired into the `Segment::Native` arms of `segmented.rs` and `resident.rs` behind
the opt-in `KTIR_BATCHED_ATTN` (DEFAULT-OFF, see perf note below);
`KTIR_NATIVE_OP_TIMER` attributes native-segment time across op classes (INC-0).

RFC 0682: execution-backend choice only — no new `ktdp` ops, no
`construct_memory_view` allocation, no op-semantic change. Bit-faithful (same
`nax_matmul` f16-in/f32-accumulate kernel as the golden-validated per-core path):
Llama-1B prefill golden 0.00403 with all 16 native attention nodes batched (512
matmul collapses); resident parity 0.00403; SmolLM2-135M prefill 0.00345; decode
unchanged.

**Perf (measured). Batched Metal/NAX BEATS AMX 2–3× for attention compute — but
two conditions gate capturing it, and INC-1 as built meets neither, so it ships
opt-in (default-off) for now.**

Compute-only measurement (`batched_attention::batched_nax_vs_amx_compute_only`;
operands resident, output never read back, GPU hardware-timestamp kernel time vs
best-of-N batched Accelerate sgemm — the steady-state a resident pipeline sees), 32
heads, head_dim 64, on `starpit/rust`:

| shape | NAX batched | AMX | NAX speedup |
|---|--:|--:|--:|
| large-layer control m=512 k=2048 n=2048 | 1.17 ms | 2.26 ms | **1.93×** |
| ATTN QK^T row-batched (m=cap) cap=256 | 0.16 ms | 0.35 ms | **2.26×** |
| ATTN QK^T row-batched cap=1024 | 2.30 ms | 4.76 ms | **2.07×** |
| ATTN QK^T row-batched cap=4096 | 35.7 ms | 110 ms | **3.08×** |
| ATTN A·V row-batched cap=1024 | 3.22 ms | 4.90 ms | **1.52×** |
| ATTN QK^T **unrolled m=1** cap=4096 | 0.79 ms | 0.47 ms | 0.60× |
| ATTN QK^T unrolled m=1 cap=256 | 0.26 ms | 0.016 ms | 0.06× |

So head-batched NAX is the **faster compute** for attention, consistent with the
size-gated backend / `fused_gpu_vs_cpu` 1.86× — *when* the work has GPU parallelism.
The two conditions:

1. **Row-batched (`m=cap`), not unrolled (`m=1`).** The cached bundles emit attention
   one query row at a time (`m=1` GEMV) — the only regime where NAX loses (0.03–0.60×,
   no row parallelism / low occupancy). A row-batched emit, or TODO #2's flash-attention
   tiling (which produces `m=block` score GEMMs), flips it to 2–3×.
2. **Resident operands/scores (no host round-trip).** INC-1 keeps softmax/transpose
   per-core on the CPU, so the `[cap,cap]` scores ship host↔device between the GPU
   matmuls; that transfer eats the compute win. The full-node A/B on llama-3.2-1b
   prefill is **0.90×** (4368 ms per-core vs 4877 ms batched) for exactly this reason —
   NOT because the matmul is slower (it is 2–3× faster), but because INC-1 pays the
   round-trip and runs on the bundles' unrolled `m=1` form.

(A superseded transfer-inclusive sweep, `batched_vs_percore_gemm_crossover`, showed
0.01–0.51× — but it timed `run_batched`'s per-call upload + full-scores readback,
i.e. data movement a resident pipeline avoids; kept only as the host-round-trip
datapoint.) **The win is captured by INC-4** (device-resident QK^T→softmax→A·V chain:
gather once, keep scores in a `UnifiedBuffer`, scatter once) **driven by a row-batched
attention shape** (FA-style tiling, not per-qrow unroll). INC-1 is the correct,
golden-faithful foundation + measurement harness for that; it stays opt-in
(`KTIR_BATCHED_ATTN`, default-off) until INC-4 lands. (Long-context prefill —
`[num_cores,…]` K/V > 2 MB LX — also still falls back, no `dies_at`/`forget` reclaim yet.)

### K6. Long-context attention LX overflow (flash-attention IR-rewrite pass)

**Status**: ✅ Implemented + fires on the REAL cached prefill nodes
(`ktir_optimizer::flash_attn`; cap-tiles the re-rolled context block, semantics
preserved on llama-3.2-1b-prefill + smollm2-135m-prefill node111, weight-free).

The `[m, cap]` attention scores tile is an INTRA-node tile that overflows the 2 MB
LX as the KV length (`cap`) grows; segmentation cannot help because attention is
one node (K4/K5 tile the head dim, not the cap dim). `flash_attn` has TWO
structural recognizers, both fail-safe (`None` on any deviation):
* `recognize_attention` / `tile_attention` — the clean single-block canonical
  idiom (QK^T → scale → (mask) → softmax → A·V), rewritten to an `scf.for` over KV
  blocks with online softmax (iter-args carry running max / sum / acc, avoiding the
  unregistered `tensor.insert_slice`). Used for synthetic / clean nodes.
* `recognize_rerolled_attention` / `tile_rerolled_attention` — the **head_rewrite
  OUTPUT** (the REAL model node). `head_rewrite` (K7, runs FIRST) re-rolls the
  unrolled per-head GQA lowering into a whole-tensor two-block form: a CONTEXT
  `[m, cap]` QK^T (the tile that overflows) + a small `[m, m]` masked DIAGONAL +
  online-softmax combine + two A·V, all stored as one `arith.addf(ovc, ovd)`. This
  recognizer ANCHORS on exactly that addf-of-two-matmuls store value (the hop the
  single-block recognizer bails on), recovers cap/m/d/gqac/hdc/scale/ninf purely
  STRUCTURALLY from the IR, and the tiler cap-tiles ONLY the CONTEXT block into KV
  blocks (`blk = choose_block_budgeted` so the per-block `[m, blk]` tile fits LX),
  leaving the `[m, m]` diagonal whole; both partials are re-based onto the global
  max and combined. Grid `[H,1,1]` and the per-head `get_compute_tile_id`/`divui
  gqac`/`muli hdc` arithmetic are preserved verbatim. ONLY RFC-0682 ops (ktdp
  load/store + Arith/Math/LinAlg/Tensor + ONE `scf.for`); NO `tensor.insert_slice`.

Both are gated by Contract (B)'s `attention_needs_flash` (`fusion.rs`):
`ReRolledIsland::scores_bytes == HeadAttnIsland::scores_bytes` (`m*cap*bytes`,
context tile only), so the monotone predicate routes a node to head-reroll XOR
flash, never both. Wired in `program::module_from_nodes` before segmentation
(covers both run paths).

Golden: single-block tiled-vs-naive ≈3e-5, forced-fire through the fusion path
2e-4. **REAL-IR weight-free semantics gate** (`flash_attn_golden.rs`,
`flash_rerolled_equals_head_rewrite_*`, `--ignored` + `metal`): for BOTH
llama-3.2-1b-prefill and smollm2-135m-prefill node111, after `head_rewrite` +
forced tiny budget, `flash_attn` FIRES (count > 0 — a no-op is a FAIL), the
flash-tiled module run through the UNCHANGED `interpreter::execute_function` equals
the head-rewritten reference within 0.05 (worst max-abs 0.0143 llama / 0.0096
smollm2), and the per-block context tile is strictly smaller than the full
`[m, cap]` footprint (the real long-context fix). Below the cap (the real caps are
tiny, 64) the pass NO-OPS, so the production `program::execute` golden is unchanged
(0.00138). **Follow-up**: the post-merge region-aware INC that lets a cap-tiled
node ALSO be head-batched (compose K5 + K6).

### K7. Head-parallel attention RE-ROLL (below-cap head IR-rewrite pass)

**Status**: ✅ Implemented + deployed (`ktir_optimizer::head_rewrite`; fires on the
real cached prefill nodes; measured wall-clock win).

The real cached prefill attention nodes (`node111.mlir`) are the head-parallel SPMD
lowering K6's recognizer explicitly does NOT match: a `grid = [H, 1]` whose per-core
body is `m` MANUALLY UNROLLED query rows, each a two-block (square context +
ragged-causal diagonal) online softmax — ~100 interpreter ops per row × `m` rows.
`head_rewrite` is the head analogue of `flash_attn` for the cap dim:
`recognize_head_attention` matches that unrolled idiom from STRUCTURAL invariants
only (grid `[H,1]` H>1, no top-level `scf.*`, exactly `m == view0.rows` stores, the
two-matmul-pair QK^T/AV signature per row, and the diagonal access tile verified to
grow EXACTLY `r+1` rows — causal growth checked, not assumed; GQA divisor / head dim
/ scale / -inf all READ from the ops), fail-safe `None` on any deviation.
`rewrite_head_attention` RE-ROLLS the `m` rows into ONE pass of whole-`[m,*]` tensor
ops: one `[m,d]` Q load, a `[m,cap]` context block (masked by the broadcast per-head
context mask), a `[m,m]` diagonal block carrying a STATIC lower-triangular -inf mask
(0 on/below diagonal — the exact re-association of the ragged per-row diagonal),
online softmax over the two blocks, `Wc·Vc + Wd·Vd`, one `[m,d]` store. It emits ONLY
RFC-0682 ops (`ktdp` load/store + Arith/Math/LinAlg + tensor; no new ktdp ops, no
`construct_memory_view` allocation) and PRESERVES the grid `[H,1]` and the per-head
GQA column arithmetic (`get_compute_tile_id` → `divui gqac` → `muli hdc`) as SSA, so
every core still selects its own head/KV slice. Pure re-roll → emits NO `scf.*`, so
it stays region-free and never trips the batched-executor's region-free gate.

It is the BELOW-cap arm of Contract (B): gated by the SAME `attention_needs_flash`
predicate as `flash_attn`, it fires IFF the re-rolled `[m,cap]` scores tile FITS LX,
returning `None` (leave naive for `flash_attn`'s cap-tiling) on a long-context
overflow — the documented disjoint head-vs-cap partition. Wired in
`program::module_from_nodes` before segmentation (covers both run paths), ahead of
the `flash_attn` pass.

Golden (semantics gate, `tests/head_rewrite_golden.rs`, weight-free, arbitrary
inputs through the UNCHANGED `execute_function`): re-rolled-vs-original on the REAL
node111 is 1e-5 (smollm2-135m-prefill) / 5e-5 (llama-3.2-1b-prefill), both ≪ 0.05;
`program::execute` whole-model golden unchanged at 0.00138 (decode bundle, where the
single-core grid correctly fails the H>1 gate and the pass no-ops). **Honest perf**:
the re-roll is a real wall-clock win — best-of-20 release, `execute_function` on the
real node WITH vs WITHOUT the pass: smollm2-135m-prefill 15.8 ms → 2.9 ms (≈5.4×),
llama-3.2-1b-prefill 258 ms → 19 ms (≈13.3×), from amortizing interpreter dispatch /
allocation over the `m` rows. Follow-up: the post-merge region-aware INC that lets a
re-rolled head node ALSO be GPU head-batched (compose with K5).

### Suggested Execution Order

If we want the fastest path to meaningful conformance progress:

1. ✅ Build the first-class access-tile IR and affine evaluator.
2. ✅ Rework `ktdp.load` / `ktdp.store` around that representation.
3. ✅ Add `construct_indirect_access_tile`.
4. ✅ Add `construct_distributed_memory_view`.
5. 🟡 `linalg.add` ✅ and `tensor.extract_slice` ✅ done; `memref.subview` ❌ still missing.
6. ❌ Fill in the missing RFC-listed SCF ops.
7. ❌ Expand broader Arith/Math/Linalg coverage as compiler demand appears.

### Definition Of "Good Enough" For A First Conformance Milestone

A strong first milestone would be:

- ✅ affine attributes are preserved and exercised in tests
- ✅ `ktdp.load` / `ktdp.store` operate over real coordinate collections
- ✅ all RFC-defined `ktdp` ops parse and execute
- ❌ the RFC matrix-add example can run with only mechanical syntax adaptation
- ❌ the repo has explicit tests for unsupported versus supported RFC surface
