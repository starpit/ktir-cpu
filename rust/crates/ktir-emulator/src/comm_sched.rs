// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Cross-core communication scheduler — port of
//! `GridExecutor.execute_with_communication` + `CoreExecutionStack` from
//! `ktir_emulator/grid.py`, and the ring all-reduce from `ktir_emulator/ops/comm_ops.py`.
//!
//! This is the one genuine redesign in the port. Python models a blocked core
//! as a generator that `yield`s `RecvRequest`; the scheduler parks it and
//! resumes via `gen.send(tile)`. Rust has no generators, so each core is an
//! explicit resumable [`CoreRunner`] state machine: it runs straight-line ops
//! via [`crate::interpreter::execute_op`] until it hits a **comm op**, which is
//! driven through the locked [`CommOp`]/[`CommStep`] protocol (comm.rs). Sends
//! go through `CoreContext::send_to` (drained into a message buffer after each
//! step); a recv parks the core until the matching tile is delivered. Per the
//! spec, comm only happens at the top level, so nested regions stay synchronous.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::rc::Rc;

use crate::comm::{CommOp, CommStep, RecvRequest};
use crate::context::CoreContext;
use crate::dialects::Dispatch;
use crate::env::{ExecutionEnv, GridExecutor};
use crate::interpreter::execute_op;
use crate::ir::{Attr, Operation, Value};
use crate::memory::SpyreMemoryHierarchy;
use crate::tile::Tile;

// ---------------------------------------------------------------------------
// Per-function derived-plan cache.
//
// The LX-liveness map (`compute_dies_at`) and — under Metal — the matmul-loop
// schedule (`matmul_loop_schedule`) and map-window fusion plan
// (`map_fusion_plan`) are PURE functions of a function's `ops`: nothing about
// them changes between forward passes. But the resident decode loop re-derives
// all three on EVERY pass (and every segment), and the flamegraph shows that
// re-derivation — window analysis + MSL codegen + liveness walks — dominating
// CPU time while the GPU sits idle. We memoize each by a structural fingerprint
// of `ops`, so a repeated pass over the same program pays a single cheap hash
// instead of the full analysis. (The compiled Metal pipeline was already cached
// separately, keyed by MSL source, in `metal::cached_dispatch`.)
// ---------------------------------------------------------------------------

/// Hash EVERY field the plans depend on — op_type, result, operands,
/// result_type, attributes, and nested regions — recursively. A collision would
/// require two functions with structurally identical IR (down to attribute
/// payloads), in which case reusing the plan is correct anyway; an accidental
/// 64-bit clash across the handful of distinct functions in a session is
/// negligible. Attributes are an unordered map, so they're folded with XOR.
fn hash_ops(ops: &[Operation], h: &mut std::collections::hash_map::DefaultHasher) {
    use std::hash::{Hash, Hasher};
    ops.len().hash(h);
    for op in ops {
        op.op_type.hash(h);
        op.result.hash(h);
        op.operands.hash(h);
        op.result_type.hash(h);
        let mut attr_acc: u64 = 0;
        for (k, v) in &op.attributes {
            let mut e = std::collections::hash_map::DefaultHasher::new();
            k.hash(&mut e);
            hash_attr(v, &mut e);
            attr_acc ^= e.finish();
        }
        attr_acc.hash(h);
        for r in &op.regions {
            hash_ops(r, h);
        }
    }
}

fn hash_attr(a: &Attr, h: &mut std::collections::hash_map::DefaultHasher) {
    use std::hash::Hash;
    std::mem::discriminant(a).hash(h);
    match a {
        Attr::Int(x) => x.hash(h),
        Attr::IntList(x) => x.hash(h),
        Attr::Float(x) => x.to_bits().hash(h),
        Attr::FloatList(x) => x.iter().for_each(|f| f.to_bits().hash(h)),
        Attr::Str(x) => x.hash(h),
        Attr::StrList(x) => x.hash(h),
        Attr::Bool(x) => x.hash(h),
        // Rare, non-hot attrs (ktdp views / affine maps): Debug is exact and cheap
        // at this frequency.
        Attr::Dtype(x) => format!("{x:?}").hash(h),
        Attr::AffineMap(x) => format!("{x:?}").hash(h),
        Attr::AffineMapList(x) => format!("{x:?}").hash(h),
        Attr::AffineSet(x) => format!("{x:?}").hash(h),
    }
}

/// Structural fingerprint of a function's ops — the key into the per-segment
/// plan caches. Exposed so the resident executor can precompute it once per
/// segment (its segments are stable for its lifetime) instead of paying the
/// deep ops-tree hash on every forward pass.
pub(crate) fn plan_key(ops: &[Operation]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    hash_ops(ops, &mut h);
    h.finish()
}

thread_local! {
    static DIES_AT_CACHE: std::cell::RefCell<HashMap<u64, Rc<Vec<Vec<u32>>>>> =
        std::cell::RefCell::new(HashMap::new());
    /// Per-function SSA-name intern table, keyed by `plan_key`. Shared across all
    /// forward passes of a function so each name allocates an id exactly once for
    /// the session (the table interns dynamically as ops execute).
    static INTERN_CACHE: std::cell::RefCell<
        HashMap<u64, Rc<crate::machine_state::memory::UnsafeShared<crate::context::InternTable>>>,
    > = std::cell::RefCell::new(HashMap::new());
}

/// The shared intern table for the function identified by `key` ([`plan_key`]).
fn cached_intern_table(
    key: u64,
) -> Rc<crate::machine_state::memory::UnsafeShared<crate::context::InternTable>> {
    INTERN_CACHE.with(|c| {
        Rc::clone(c.borrow_mut().entry(key).or_insert_with(|| {
            Rc::new(crate::machine_state::memory::UnsafeShared::new(
                crate::context::InternTable::new(),
            ))
        }))
    })
}

/// Memoized [`compute_dies_at`] keyed by `key` ([`plan_key`]), with the dead SSA
/// NAMES pre-resolved to intern ids. The liveness reclaim then runs `forget_id`
/// per dead value with no per-op name hashing (the names are interned once for the
/// function, into the SAME shared table the value store uses). Cached because both
/// the analysis and the resolution are pure functions of the ops.
fn cached_dies_at(
    ops: &[Operation],
    key: u64,
    intern: &Rc<crate::machine_state::memory::UnsafeShared<crate::context::InternTable>>,
) -> Rc<Vec<Vec<u32>>> {
    DIES_AT_CACHE.with(|c| {
        if let Some(v) = c.borrow().get(&key) {
            return Rc::clone(v);
        }
        let names = compute_dies_at(ops);
        let ids: Vec<Vec<u32>> = names
            .iter()
            .map(|row| row.iter().map(|n| intern.borrow_mut().intern(n)).collect())
            .collect();
        let v = Rc::new(ids);
        c.borrow_mut().insert(key, Rc::clone(&v));
        v
    })
}

#[cfg(metal)]
type MapPlan = (
    HashMap<usize, crate::metal::MapRegionKernel>,
    std::collections::HashSet<usize>,
);

#[cfg(metal)]
thread_local! {
    static MATMUL_SCHED_CACHE: std::cell::RefCell<
        HashMap<u64, Rc<HashMap<String, crate::metal::MatmulLoopInfo>>>,
    > = std::cell::RefCell::new(HashMap::new());
    static MAP_PLAN_CACHE: std::cell::RefCell<HashMap<u64, Rc<MapPlan>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Memoized [`crate::metal::matmul_loop_schedule`] keyed by `key` ([`plan_key`]).
#[cfg(metal)]
fn cached_matmul_schedule(
    ops: &[Operation],
    key: u64,
) -> Rc<HashMap<String, crate::metal::MatmulLoopInfo>> {
    MATMUL_SCHED_CACHE.with(|c| {
        if let Some(v) = c.borrow().get(&key) {
            return Rc::clone(v);
        }
        let v = Rc::new(crate::metal::matmul_loop_schedule(ops));
        c.borrow_mut().insert(key, Rc::clone(&v));
        v
    })
}

/// Memoized [`crate::metal::map_fusion_plan`] keyed by `key` ([`plan_key`]).
#[cfg(metal)]
fn cached_map_fusion_plan(ops: &[Operation], key: u64) -> Rc<MapPlan> {
    MAP_PLAN_CACHE.with(|c| {
        if let Some(v) = c.borrow().get(&key) {
            return Rc::clone(v);
        }
        let v = Rc::new(crate::metal::map_fusion_plan(ops));
        c.borrow_mut().insert(key, Rc::clone(&v));
        v
    })
}

/// True if `op_type` is a cross-core comm op (driven by the scheduler rather
/// than the normal handler table). Keyed registry analogue.
pub fn is_comm_op(op_type: &str) -> bool {
    matches!(op_type, "ktdp.reduce")
}

/// Construct the [`CommOp`] state machine for a comm op, reading its operands
/// from `ctx`. Mirrors `ktdp__reduce` (operands `[tile, core_group]`).
fn make_comm_op(op: &Operation, ctx: &CoreContext) -> Result<Box<dyn CommOp>, String> {
    match op.op_type.as_str() {
        "ktdp.reduce" => {
            let tile = match ctx.get_value(&op.operands[0])? {
                Value::Tile(t) => t.clone(),
                other => {
                    return Err(format!(
                        "ktdp.reduce: operand 0 must be a Tile, got {other:?}"
                    ));
                }
            };
            let core_group = read_core_group(ctx.get_value(&op.operands[1])?)?;
            Ok(Box::new(RingReduce::new(tile, core_group)))
        }
        other => Err(format!("not a comm op: {other}")),
    }
}

/// A core group is a tuple/list of core ids. Accepts `Value::Tuple` of
/// `Index`/int scalars.
fn read_core_group(v: &Value) -> Result<Vec<usize>, String> {
    let items = match v {
        Value::Tuple(items) => items,
        other => {
            return Err(format!(
                "ktdp.reduce: core_group must be a tuple, got {other:?}"
            ));
        }
    };
    items
        .iter()
        .map(|it| match it {
            Value::Index(i) => Ok(*i as usize),
            Value::Scalar(s) => s
                .as_i64()
                .map(|i| i as usize)
                .ok_or_else(|| "ktdp.reduce: core_group element not an int".to_string()),
            other => Err(format!("ktdp.reduce: bad core_group element {other:?}")),
        })
        .collect()
}

/// Ring all-reduce (sum) — one core's view. Port of `RingReduceBackend`.
///
/// Each core sends to `(idx+1) % N` and receives from `(idx-1) % N`, running
/// `N-1` rounds. The accumulator folds in each received tile; the *received*
/// tile (not the accumulator) is forwarded next round, so each starting tile
/// visits every core exactly once. After `N-1` rounds every core holds the full
/// sum. Cores outside the group return their tile unchanged without comm.
struct RingReduce {
    tile: Tile,
    core_group: Vec<usize>,
    // resolved on first step (when we know our core_id):
    state: RingState,
}

enum RingState {
    Init,
    /// Mid-ring: accumulator, tile to forward next round, rounds remaining,
    /// next/prev core ids.
    Running {
        result: Tile,
        to_forward: Tile,
        rounds_left: usize,
        next_core: usize,
        prev_core: usize,
    },
}

impl RingReduce {
    fn new(tile: Tile, core_group: Vec<usize>) -> Self {
        RingReduce {
            tile,
            core_group,
            state: RingState::Init,
        }
    }
}

/// Element-wise sum of two tiles (the default `reduce_fn`, `ArithOps.addf`).
fn tile_add(a: &Tile, b: &Tile) -> Result<Tile, String> {
    if a.shape != b.shape {
        return Err(format!(
            "ktdp.reduce: tile shape mismatch {:?} vs {:?}",
            a.shape, b.shape
        ));
    }
    let data = a
        .as_f32()
        .iter()
        .zip(b.as_f32().iter())
        .map(|(x, y)| x + y)
        .collect();
    Ok(Tile::compute(data, a.dtype, a.shape.clone()))
}

impl CommOp for RingReduce {
    fn step(&mut self, ctx: &mut CoreContext, incoming: Option<Tile>) -> Result<CommStep, String> {
        match &mut self.state {
            RingState::Init => {
                // Not in the group: identity passthrough, no comm.
                let Some(my_idx) = self.core_group.iter().position(|&c| c == ctx.core_id) else {
                    return Ok(CommStep::Done(Box::new(Some(Value::Tile(
                        self.tile.clone(),
                    )))));
                };
                let n = self.core_group.len();
                if n <= 1 {
                    return Ok(CommStep::Done(Box::new(Some(Value::Tile(
                        self.tile.clone(),
                    )))));
                }
                let next_core = self.core_group[(my_idx + 1) % n];
                let prev_core = self.core_group[(my_idx + n - 1) % n];
                // Round 1: send local tile onward, then wait for prev.
                ctx.send_to(next_core, self.tile.clone());
                self.state = RingState::Running {
                    result: self.tile.clone(),
                    to_forward: self.tile.clone(),
                    rounds_left: n - 1,
                    next_core,
                    prev_core,
                };
                Ok(CommStep::Recv(RecvRequest { src: prev_core }))
            }
            RingState::Running {
                result,
                to_forward,
                rounds_left,
                next_core,
                prev_core,
            } => {
                let received = incoming.ok_or("ktdp.reduce: resumed without an incoming tile")?;
                *result = tile_add(result, &received)?;
                *to_forward = received;
                *rounds_left -= 1;
                if *rounds_left == 0 {
                    return Ok(CommStep::Done(Box::new(Some(Value::Tile(result.clone())))));
                }
                ctx.send_to(*next_core, to_forward.clone());
                Ok(CommStep::Recv(RecvRequest { src: *prev_core }))
            }
        }
    }
}

/// One core's resumable execution: runs top-level ops until it blocks on a recv
/// or finishes. The Rust analogue of `CoreExecutionStack`.
struct CoreRunner {
    ctx: CoreContext,
    op_idx: usize,
    /// `Some` while suspended inside a comm op: `(machine, result_name)`.
    active: Option<(Box<dyn CommOp>, Option<String>)>,
    /// `dies_at[i]` = function-scope SSA tiles whose LAST use is top-level op `i`
    /// (counting uses nested in regions). After running op `i` they are dead, so
    /// their LX is reclaimed. Without this, a whole-program-fused function would
    /// hold every node's tiles resident at once and blow the 2 MB LX budget; the
    /// per-node runner gets the same effect for free via a fresh memory hierarchy
    /// per call. Shared (identical for every core).
    dies_at: Rc<Vec<Vec<u32>>>,
    /// Structural fingerprint of `ops` (hashed once in
    /// [`execute_with_communication`]); the key into the per-segment Metal plan
    /// caches, so [`step`](Self::step) never re-hashes the ops tree.
    #[cfg(metal)]
    plan_key: u64,
}

enum Poll {
    Block(usize), // waiting on a recv from this core
    Done,
}

impl CoreRunner {
    /// Advance: feed `incoming` to a suspended comm op (if any), then run
    /// straight-line ops until the next block or completion.
    fn step(
        &mut self,
        ops: &[Operation],
        env: &ExecutionEnv,
        mut incoming: Option<Tile>,
    ) -> Result<Poll, String> {
        // Resume a suspended comm op first.
        if let Some((comm, result_name)) = &mut self.active {
            match comm.step(&mut self.ctx, incoming.take())? {
                CommStep::Recv(req) => return Ok(Poll::Block(req.src)),
                CommStep::Done(val) => {
                    let name = result_name.clone();
                    self.active = None;
                    bind_result(&mut self.ctx, name.as_deref(), *val)?;
                }
            }
        }
        // MLX-style GPU offload: recognized matmul K-loops run as one GPU GEMM
        // instead of the interpreter's K-tiled loop. Gated to SINGLE-CORE
        // functions: there the K-loop's M-offset is 0 / the forwarded source
        // gives the full M, so reconstructing the whole GEMM is correct. A
        // multi-core grid M-tiles across cores (each core a block offset by its
        // tile id), so the full-shape reconstruction would be wrong — those keep
        // the interpreter loop. Skipped under a latency tracker (the model must
        // see each op). The fused whole-program function is grid [1,1].
        //
        // The same single-core/tracker-free conditions also gate the (opt-in)
        // attention-island offloads (plain matmul / reduce / transpose) below,
        // each behind its own KTIR_GPU_* toggle for A/B measurement.
        #[cfg(metal)]
        let gpu_base = env.tracker.is_none() && env.grid.num_cores == 1;
        #[cfg(metal)]
        let gpu_offload = gpu_base && std::env::var_os("KTIR_NO_GPU_GEMM").is_none();
        // Attention-island offloads (plain matmul / reduce / transpose). These are
        // OPT-IN (default OFF): measured on the SmolLM2-135M decode + 8-token
        // prefill bundles, per-op GPU dispatch of attention's TINY tensors (M=1
        // GEMMs, <=576-wide reduces, <=64x64 transposes) is a net LOSS — each pays
        // ~250us of GPU dispatch+sync that swamps the few-us CPU compute (decode
        // 0.53-0.90x, prefill 0.89-1.01x vs all-CPU). They are correct and golden-
        // faithful, and would win for much larger attention tensors, so they ship
        // gated behind KTIR_GPU_* env toggles (presence ENABLES) rather than
        // regressing the default path. Plain matmul additionally honors
        // KTIR_NO_GPU_GEMM (a clean GEMM-free baseline disables it too).
        #[cfg(metal)]
        let gpu_plain_matmul = gpu_offload && std::env::var_os("KTIR_GPU_PLAIN_MATMUL").is_some();
        #[cfg(metal)]
        let gpu_reduce = gpu_base && std::env::var_os("KTIR_GPU_REDUCE").is_some();
        #[cfg(metal)]
        let gpu_transpose = gpu_base && std::env::var_os("KTIR_GPU_TRANSPOSE").is_some();
        // Structural fingerprint shared by both Metal plan caches below — hashed
        // ONCE per segment in `execute_with_communication`, threaded in here.
        #[cfg(metal)]
        let plan_key = self.plan_key;
        #[cfg(metal)]
        let matmul_sched: Rc<HashMap<String, crate::metal::MatmulLoopInfo>> = if gpu_offload {
            cached_matmul_schedule(ops, plan_key)
        } else {
            Rc::new(HashMap::new())
        };
        // MLX-style map-window fusion: each maximal run of fusable elementwise/
        // cast/broadcast ops runs as ONE fused GPU kernel (instead of op-by-op on
        // the interpreter). Same gating as the matmul-loop offload above. The plan
        // maps each window's TRIGGER op (its last op) -> the compiled kernel, and a
        // SKIP set of all window op indices; non-trigger window ops are subsumed by
        // the fused kernel (their values come from it) and are not executed.
        #[cfg(metal)]
        let map_plan: Rc<MapPlan> = if gpu_offload && std::env::var_os("KTIR_NO_GPU_MAP").is_none()
        {
            cached_map_fusion_plan(ops, plan_key)
        } else {
            Rc::new((HashMap::new(), std::collections::HashSet::new()))
        };
        #[cfg(metal)]
        let (map_triggers, map_skip) = (&map_plan.0, &map_plan.1);

        // Window op indices whose liveness reclaim is deferred to the window's
        // trigger (so a fused kernel's live-ins survive until it has read them).
        #[cfg(metal)]
        let mut pending_skip: Vec<usize> = Vec::new();

        // Run remaining top-level ops.
        while self.op_idx < ops.len() {
            let op = &ops[self.op_idx];
            let this_idx = self.op_idx;
            self.op_idx += 1;
            // DIAGNOSTIC (KTIR_GEMM_DIAG): the decisive check the per-GEMM
            // KTIR_GEMM_CHECK can't do. KTIR_GEMM_CHECK compares the GPU GEMM to a
            // CPU sgemm on the *same recognized operands*, so it can never catch a
            // recognizer that reconstructs the WRONG (m,k,n,a_root,b_root). Here we
            // instead run the loop's ACTUAL scf.for body on the interpreter into
            // out_ssa, capture that result, then run the GPU offload (overwriting
            // out_ssa), and compare the two — a divergence pinpoints a recognizer
            // mis-derivation (the GPU computed a correct-but-WRONG A@B vs the loop's
            // real result).
            #[cfg(metal)]
            if op.op_type == "scf.for"
                && std::env::var_os("KTIR_GEMM_DIAG").is_some()
                && let Some(info) = op.result.as_deref().and_then(|r| matmul_sched.get(r))
            {
                // Run the real K-loop on the interpreter, capturing its result.
                execute_op(op, &mut self.ctx, env)?;
                let interp = match self.ctx.get_value(&info.out_ssa) {
                    Ok(Value::Tile(t)) => Some(t.as_f32().into_owned()),
                    _ => None,
                };
                // Run the GPU offload (overwrites out_ssa with the GPU result).
                if crate::metal::run_matmul_loop_gpu(info, &mut self.ctx).is_ok()
                    && let (Some(interp), Ok(Value::Tile(gpu))) =
                        (interp, self.ctx.get_value(&info.out_ssa))
                {
                    let d = interp
                        .iter()
                        .zip(gpu.as_f32().iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    if d > 0.05 {
                        eprintln!(
                            "  [gemm-diag] DIVERGENT loop {} -> GPU vs interp max diff {d:.4}  \
                             recognized m={} k={} n={} a_root={} b_root={}  \
                             interp_len={} gpu_len={}",
                            info.out_ssa,
                            info.m,
                            info.k,
                            info.n,
                            info.a_root,
                            info.b_root,
                            interp.len(),
                            gpu.len(),
                        );
                    }
                }
                if let Some(dead) = self.dies_at.get(this_idx) {
                    for &id in dead {
                        self.ctx.forget_id(id);
                    }
                }
                continue;
            }
            // Offload a recognized K-loop as a full-M GEMM (NAX or AMX). On Ok the
            // interpreter skips the loop body. On Err the fallback is the loop's
            // own `scf.for` below — correct at grid [1,1] ONLY for m==1 (decode: it
            // computes the single row). For m>1 (prefill) that body would compute
            // ONLY row 0 and silently drop the rest, so a failed offload is FATAL —
            // fail loud rather than emit a row-0-only result.
            #[cfg(metal)]
            if op.op_type == "scf.for"
                && let Some(info) = op.result.as_deref().and_then(|r| matmul_sched.get(r))
            {
                match crate::metal::run_matmul_loop_gpu(info, &mut self.ctx) {
                    Ok(()) => {
                        if let Some(dead) = self.dies_at.get(this_idx) {
                            for &id in dead {
                                self.ctx.forget_id(id);
                            }
                        }
                        continue;
                    }
                    Err(e) if info.m > 1 => {
                        return Err(format!(
                            "metal: full-M GEMM offload failed for m={} ({}): {e}; \
                             refusing the row-0-only interpreter fallback",
                            info.m, info.out_ssa
                        ));
                    }
                    // m == 1: the interpreter scf.for below computes the single row
                    // correctly, so fall through to it.
                    Err(_) => {}
                }
            }
            // Map-window GPU fusion: at a window's TRIGGER op, run the whole window
            // as one fused kernel (its loads/plumbing already ran, populating the
            // live-ins). A trigger failure is FATAL — the rest of the window's ops
            // were skipped, so there's no interpreter result to fall back to.
            //
            // Liveness for window ops is DEFERRED to the trigger: a live-in tile
            // whose last use is an earlier (skipped) window op must not be freed
            // before the fused kernel reads it. So skipped ops accumulate their
            // indices in `pending_skip` and we reclaim the whole window's dead
            // values only AFTER the kernel has run.
            #[cfg(metal)]
            if let Some(mrk) = map_triggers.get(&this_idx) {
                crate::metal::run_map_region_gpu(mrk, &mut self.ctx)?;
                pending_skip.push(this_idx);
                for idx in pending_skip.drain(..) {
                    if let Some(dead) = self.dies_at.get(idx) {
                        for &id in dead {
                            self.ctx.forget_id(id);
                        }
                    }
                }
                continue;
            }
            // A non-trigger op inside a fused window: its value is subsumed by the
            // fused kernel run at the trigger, so don't execute it. Defer its
            // liveness reclaim to the trigger (see above).
            #[cfg(metal)]
            if map_skip.contains(&this_idx) {
                pending_skip.push(this_idx);
                continue;
            }
            // Attention-island offloads: a PLAIN (not scf.for-nested) matmul,
            // a softmax row reduce, or a transpose runs on the GPU instead of the
            // interpreter. Each falls through to `execute_op` on any failure
            // (no device, unsupported shape) so correctness is preserved. These
            // ops are window boundaries (never inside a fused map window), so
            // they never collide with the map_skip/trigger handling above.
            #[cfg(metal)]
            if (gpu_plain_matmul
                && op.op_type == "linalg.matmul"
                && crate::metal::run_plain_matmul_gpu(op, &mut self.ctx).is_ok())
                || (gpu_reduce
                    && op.op_type == "linalg.reduce"
                    && crate::metal::run_reduce_gpu(op, &mut self.ctx).is_ok())
                || (gpu_transpose
                    && op.op_type == "linalg.transpose"
                    && crate::metal::run_transpose_gpu(op, &mut self.ctx).is_ok())
            {
                if let Some(dead) = self.dies_at.get(this_idx) {
                    for &id in dead {
                        self.ctx.forget_id(id);
                    }
                }
                continue;
            }
            if is_comm_op(&op.op_type) {
                // Charge the comm op's latency once (it doesn't go through
                // execute_op). Cost is derived from the operand tile + grid size.
                if let Some(tracker) = env.tracker {
                    let operands: Vec<Option<Value>> = op
                        .operands
                        .iter()
                        .map(|n| self.ctx.get_value(n).ok().cloned())
                        .collect();
                    // Comm ops aren't in the dispatch table; their class is Comm.
                    tracker.borrow_mut().record_op(
                        self.ctx.core_id,
                        &op.op_type,
                        crate::latency::LatencyCategory::Comm,
                        &None,
                        &operands,
                    );
                }
                let mut comm = make_comm_op(op, &self.ctx)?;
                match comm.step(&mut self.ctx, None)? {
                    CommStep::Recv(req) => {
                        self.active = Some((comm, op.result.clone()));
                        return Ok(Poll::Block(req.src));
                    }
                    CommStep::Done(val) => bind_result(&mut self.ctx, op.result.as_deref(), *val)?,
                }
            } else {
                execute_op(op, &mut self.ctx, env)?;
            }
            // Reclaim LX for every value that just went dead at this op.
            if let Some(dead) = self.dies_at.get(this_idx) {
                for &id in dead {
                    self.ctx.forget_id(id);
                }
            }
        }
        Ok(Poll::Done)
    }
}

/// Compute, for a top-level op list, which SSA values become dead after each op
/// — i.e. `dies_at[i]` lists every value whose LAST use (as an operand, counting
/// uses nested in regions) is op `i`. A value never read after definition dies at
/// its own op. Used to reclaim LX as a fused function streams through, instead of
/// holding every intermediate resident. Values defined inside regions are managed
/// by region scope pop and are not tracked here.
fn compute_dies_at(ops: &[Operation]) -> Vec<Vec<String>> {
    // Recursively record the highest TOP-LEVEL index at which each name is used.
    // Uses come from operands AND from SSA names embedded in string attributes
    // (e.g. tensor.extract_slice's `slice_offsets`, a dynamic `sizes_dyn`,
    // scf.for `iter_args`/`iter_var`) — missing those frees a value too early.
    // Over-counting (a bound name read as a use) only keeps a value alive
    // longer, which is safe; under-counting corrupts execution.
    fn note_uses(op: &Operation, top_idx: usize, last_use: &mut HashMap<String, usize>) {
        for operand in &op.operands {
            if operand.starts_with('%') {
                last_use.insert(operand.clone(), top_idx);
            }
        }
        for attr in op.attributes.values() {
            match attr {
                crate::ir::Attr::Str(s) if s.starts_with('%') => {
                    last_use.insert(s.clone(), top_idx);
                }
                crate::ir::Attr::StrList(xs) => {
                    for x in xs {
                        if x.starts_with('%') {
                            last_use.insert(x.clone(), top_idx);
                        }
                    }
                }
                _ => {}
            }
        }
        for region in &op.regions {
            for inner in region {
                note_uses(inner, top_idx, last_use);
            }
        }
    }
    let mut last_use: HashMap<String, usize> = HashMap::new();
    for (i, op) in ops.iter().enumerate() {
        note_uses(op, i, &mut last_use);
    }
    // A defined-but-never-used value dies at its own op (still tracked LX to free).
    for (i, op) in ops.iter().enumerate() {
        if let Some(r) = &op.result {
            last_use.entry(r.clone()).or_insert(i);
        }
    }
    let mut dies_at = vec![Vec::new(); ops.len()];
    for (name, idx) in last_use {
        dies_at[idx].push(name);
    }
    dies_at
}

/// Bind a comm op's result value to its SSA name, tracking LX for Tiles
/// (mirrors the binding `execute_op` / `_store` perform).
fn bind_result(
    ctx: &mut CoreContext,
    name: Option<&str>,
    val: Option<Value>,
) -> Result<(), String> {
    if let (Some(name), Some(val)) = (name, val) {
        if let Value::Tile(t) = &val {
            ctx.track_lx(name, t.size_bytes() as i64)?;
        }
        ctx.set_value(name, val);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Persistent worker pool for the cores of a comm-free multi-core grid (the
// attention nodes, grid=[H,1]). The H cores are independent — each reads the
// shared (read-only) weights/inputs and writes DISJOINT outputs — so they run
// concurrently instead of one-at-a-time. The pool is spawned once per thread and
// reused across every segment/pass (no per-segment thread spawning); workers
// drain a shared job queue, and `run_all` submits the cores and BLOCKS until all
// finish, so the raw pointers each job carries outlive the workers' use of them.
// ---------------------------------------------------------------------------

/// One core's work: type-erased raw pointers to its in-place runner, the ops
/// slice, and the env (all `!Send` via interior `Rc`, shared read-only/disjointly).
#[derive(Clone, Copy)]
struct CoreWork {
    runner: *mut CoreRunner,
    ops: *const [Operation],
    env: *const std::ffi::c_void,
}
// SAFETY: each item targets a distinct in-place runner; the shared state it
// touches is disjoint-or-read-only, and `run_all` blocks until every job is done.
unsafe impl Send for CoreWork {}

fn run_core(w: CoreWork) -> Result<(), String> {
    // SAFETY: distinct runner per job; ops/env read-only; submitter blocks on join.
    let runner = unsafe { &mut *w.runner };
    let ops = unsafe { &*w.ops };
    let env = unsafe { &*(w.env as *const ExecutionEnv) };
    match runner.step(ops, env, None)? {
        Poll::Done => Ok(()),
        Poll::Block(_) => Err("parallel core unexpectedly blocked in a comm-free segment".into()),
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

struct WorkerPool {
    queue: std::sync::Arc<(
        std::sync::Mutex<std::collections::VecDeque<Job>>,
        std::sync::Condvar,
    )>,
}

impl WorkerPool {
    fn new(n: usize) -> Self {
        let queue = std::sync::Arc::new((
            std::sync::Mutex::new(std::collections::VecDeque::new()),
            std::sync::Condvar::new(),
        ));
        for _ in 0..n.max(1) {
            let q = queue.clone();
            // Daemon worker: loops forever, killed at process exit (handle dropped).
            std::thread::spawn(move || {
                loop {
                    let job: Job = {
                        let (lock, cv) = &*q;
                        let mut g = lock.lock().unwrap();
                        while g.is_empty() {
                            g = cv.wait(g).unwrap();
                        }
                        g.pop_front().unwrap()
                    };
                    job();
                }
            });
        }
        WorkerPool { queue }
    }

    /// Submit every core and block until all complete; results in submission order.
    fn run_all(&self, items: Vec<CoreWork>) -> Vec<Result<(), String>> {
        let n = items.len();
        if n == 0 {
            return Vec::new();
        }
        let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<(), String>)>();
        {
            let (lock, cv) = &*self.queue;
            let mut g = lock.lock().unwrap();
            for (i, w) in items.into_iter().enumerate() {
                let tx = tx.clone();
                g.push_back(Box::new(move || {
                    let _ = tx.send((i, run_core(w)));
                }));
            }
            cv.notify_all();
        }
        let mut results: Vec<Result<(), String>> = (0..n).map(|_| Ok(())).collect();
        for _ in 0..n {
            let (i, r) = rx.recv().unwrap();
            results[i] = r;
        }
        results
    }
}

thread_local! {
    static WORKER_POOL: std::cell::OnceCell<WorkerPool> = const { std::cell::OnceCell::new() };
    /// Set while a caller GUARANTEES the cores will not mutate shared state that the
    /// worker pool races on — specifically that every HBM stick is PRE-ALLOCATED, so
    /// no core calls `hbm.allocate()` during the parallel section. The resident
    /// executor sets this (it allocates every tensor's stick once at construction);
    /// `execute_function` (fresh HBM, lazy per-op allocation) leaves it false, so the
    /// grid runs serial there — the only path where concurrent allocation corrupted
    /// the heap (the layernorm SIGABRT). Default OFF (safe).
    static PARALLEL_SAFE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark the current thread's grid execution as parallel-safe (pre-allocated HBM),
/// returning the prior value so callers can restore it (RAII guard). See
/// [`PARALLEL_SAFE`].
pub fn set_parallel_safe(on: bool) -> bool {
    PARALLEL_SAFE.with(|c| c.replace(on))
}

/// Drive all cores to completion, resolving cross-core recvs. Port of
/// `GridExecutor.execute_with_communication`. Cores with no comm op simply run
/// to completion on the first advance.
pub fn execute_with_communication(
    grid: &GridExecutor,
    mem: &SpyreMemoryHierarchy,
    ops: &[Operation],
    input_ptrs: &[(String, Value)],
    dispatch: &Dispatch,
    tracker: Option<&std::cell::RefCell<crate::latency::LatencyTracker>>,
    precomputed_key: Option<u64>,
) -> Result<(), String> {
    let env = match tracker {
        Some(t) => ExecutionEnv::with_tracker(dispatch, grid, t),
        None => ExecutionEnv::new(dispatch, grid),
    };
    let num_cores = grid.num_cores.max(1);

    // Structural fingerprint of this segment's ops — shared by every per-segment
    // plan cache (liveness + Metal schedule/fusion). The resident executor owns
    // its segments for its whole lifetime and precomputes this once per segment
    // (instance-scoped, so no cross-program pointer aliasing), passing it in;
    // other callers pass None and we hash the ops tree here.
    let plan_key = precomputed_key.unwrap_or_else(|| plan_key(ops));

    // SSA-name -> id table for this function, SHARED across passes (keyed by
    // plan_key), so each name interns once for the session — the per-core value
    // table is then a flat id-indexed Vec instead of a per-op-allocating HashMap.
    let intern = cached_intern_table(plan_key);

    // Liveness for LX reclaim (identical for every core) — see `CoreRunner::dies_at`.
    // Memoized across passes (dead names pre-resolved to intern ids).
    let dies_at = cached_dies_at(ops, plan_key, &intern);

    let mut runners: BTreeMap<usize, CoreRunner> = BTreeMap::new();
    for core_id in 0..num_cores {
        let mut ctx = CoreContext::with_intern(
            core_id,
            grid.linear_to_grid(core_id),
            Rc::clone(&mem.hbm),
            mem.get_lx(core_id),
            mem.lx_scratchpads.clone(),
            Rc::clone(&intern),
        );
        // The `unique_sticks` latency sideband is only read when metering; skip
        // its per-element stick `HashSet` on untracked runs (resident decode/
        // prefill), where it's pure overhead on the load/store gather hot path.
        ctx.set_track_sticks(env.tracker.is_some());
        for (name, val) in input_ptrs {
            ctx.set_value(name, val.clone());
        }
        runners.insert(
            core_id,
            CoreRunner {
                ctx,
                op_idx: 0,
                active: None,
                dies_at: Rc::clone(&dies_at),
                #[cfg(metal)]
                plan_key,
            },
        );
    }

    let mut messages: HashMap<(usize, usize), VecDeque<Tile>> = HashMap::new();
    let mut waiting: HashMap<usize, usize> = HashMap::new(); // core -> src it waits on

    // Helper: advance one core and route its sends / record its block state.
    fn advance(
        core_id: usize,
        incoming: Option<Tile>,
        runners: &mut BTreeMap<usize, CoreRunner>,
        messages: &mut HashMap<(usize, usize), VecDeque<Tile>>,
        waiting: &mut HashMap<usize, usize>,
        ops: &[Operation],
        env: &ExecutionEnv,
    ) -> Result<(), String> {
        let runner = runners.get_mut(&core_id).expect("live core");
        let poll = runner.step(ops, env, incoming)?;
        for (dst, tile) in runner.ctx.drain_outbox() {
            messages.entry((core_id, dst)).or_default().push_back(tile);
        }
        match poll {
            Poll::Block(src) => {
                waiting.insert(core_id, src);
            }
            Poll::Done => {
                runners.remove(&core_id);
            }
        }
        Ok(())
    }

    // Comm-free multi-core grid (the attention nodes, grid=[H,1] with no
    // send/recv): the H cores are INDEPENDENT — each reads the shared (read-only)
    // weights/inputs and writes DISJOINT outputs — so run them across CPU threads
    // instead of the serial loop below. Core 0 runs serially FIRST to warm the
    // shared intern table (so the parallel cores only hit it, never insert) and to
    // materialize the shared output allocation; cores 1..N run on the persistent
    // pool, each driving its in-place `CoreRunner` via a raw pointer (the shared
    // cells are borrowed, never cloned, during a step, and `UnsafeShared` drops the
    // borrow flag so disjoint concurrent access is sound). Skipped for the
    // latency-metered path (`tracker` is Some), which must stay serial.
    // The multi-core worker-pool path shares HBM/intern/LX across threads via
    // `UnsafeShared`. That is sound ONLY when no core allocates in the shared HBM
    // during the parallel section (concurrent `hbm.allocate()` races the allocator →
    // heap corruption, the layernorm SIGABRT). The resident executor PRE-ALLOCATES
    // every stick and sets `PARALLEL_SAFE`, so its attention cores only write disjoint
    // pre-existing sticks — safe and ~3.5x faster. `execute_function` (fresh HBM, lazy
    // allocation) leaves it off → serial. `KTIR_PARALLEL_CORES` force-enables it.
    let parallel_ok = PARALLEL_SAFE.with(std::cell::Cell::get)
        || std::env::var_os("KTIR_PARALLEL_CORES").is_some();
    if num_cores > 1
        && env.tracker.is_none()
        && parallel_ok
        && !ops.iter().any(|o| is_comm_op(&o.op_type))
    {
        advance(
            0,
            None,
            &mut runners,
            &mut messages,
            &mut waiting,
            ops,
            &env,
        )?;
        let env_ptr = &env as *const ExecutionEnv as *const std::ffi::c_void;
        let ops_ptr = ops as *const [Operation];
        let works: Vec<CoreWork> = (1..num_cores)
            .filter_map(|c| {
                runners.get_mut(&c).map(|r| CoreWork {
                    runner: r as *mut CoreRunner,
                    ops: ops_ptr,
                    env: env_ptr,
                })
            })
            .collect();
        let results = WORKER_POOL.with(|cell| {
            let pool = cell.get_or_init(|| {
                let n = std::thread::available_parallelism()
                    .map(|x| x.get())
                    .unwrap_or(8);
                WorkerPool::new(n)
            });
            pool.run_all(works)
        });
        for r in results {
            r?;
        }
        return Ok(());
    }

    // Initial pass: run every core to its first block (or completion).
    for core_id in 0..num_cores {
        advance(
            core_id,
            None,
            &mut runners,
            &mut messages,
            &mut waiting,
            ops,
            &env,
        )?;
    }

    // Deliver messages and resume until all cores finish.
    while !runners.is_empty() {
        let live: Vec<usize> = runners.keys().copied().collect();
        let mut progressed = false;
        for core_id in live {
            if let Some(&src) = waiting.get(&core_id)
                && let Some(q) = messages.get_mut(&(src, core_id))
                && let Some(tile) = q.pop_front()
            {
                if q.is_empty() {
                    messages.remove(&(src, core_id));
                }
                waiting.remove(&core_id);
                advance(
                    core_id,
                    Some(tile),
                    &mut runners,
                    &mut messages,
                    &mut waiting,
                    ops,
                    &env,
                )?;
                progressed = true;
            }
        }
        if !progressed {
            let desc = waiting
                .iter()
                .map(|(c, s)| format!("core {c} waiting on recv from core {s}"))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(format!("Deadlock detected: {desc}"));
        }
    }
    Ok(())
}

/// Run a function's ops for a SINGLE compute-tile (`tile` = one grid core) to
/// completion. The compute-tile dataflow executor uses this to stream ONE
/// token-row through a node — `get_compute_tile_id` returns `tile`, so the node's
/// `scf.for` computes that row's slice, and the GEMM runs on the CPU (cblas/AMX),
/// NOT as a batched GPU dispatch (grid >1 so the matmul-loop offload is off, just
/// as for the head-parallel path). Comm-free only.
pub fn execute_function_single_tile(
    grid: &GridExecutor,
    mem: &SpyreMemoryHierarchy,
    ops: &[Operation],
    input_ptrs: &[(String, Value)],
    dispatch: &Dispatch,
    tile: usize,
    precomputed_key: Option<u64>,
) -> Result<(), String> {
    let env = ExecutionEnv::new(dispatch, grid);
    let pk = precomputed_key.unwrap_or_else(|| plan_key(ops));
    let intern = cached_intern_table(pk);
    let dies_at = cached_dies_at(ops, pk, &intern);
    let mut ctx = CoreContext::with_intern(
        tile,
        grid.linear_to_grid(tile),
        Rc::clone(&mem.hbm),
        mem.get_lx(tile),
        mem.lx_scratchpads.clone(),
        Rc::clone(&intern),
    );
    ctx.set_track_sticks(false);
    for (name, val) in input_ptrs {
        ctx.set_value(name, val.clone());
    }
    let mut runner = CoreRunner {
        ctx,
        op_idx: 0,
        active: None,
        dies_at: Rc::clone(&dies_at),
        #[cfg(metal)]
        plan_key: pk,
    };
    match runner.step(ops, &env, None)? {
        Poll::Done => Ok(()),
        Poll::Block(_) => Err("single-tile: unexpected block in a comm-free node".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::DType;

    /// Run a scheduler over per-core seed bindings, returning each core's final
    /// scope value for `result_name`. A test harness that keeps the contexts so
    /// results stay inspectable (unlike `execute_with_communication`, whose
    /// cores write to shared HBM rather than returning values).
    fn run_capturing(
        grid: &GridExecutor,
        mem: &SpyreMemoryHierarchy,
        ops: &[Operation],
        seeds: &[Vec<(String, Value)>],
        result_name: &str,
    ) -> Vec<Option<Value>> {
        let dispatch = Dispatch::new();
        let env = ExecutionEnv::new(&dispatch, grid);
        let n = grid.num_cores;
        // This harness reads results back as SSA values from the captured
        // contexts (not from HBM), so it must NOT reclaim dead values — keep
        // every value alive (empty schedule = no reclaim).
        let dies_at = Rc::new(Vec::new());
        let mut runners: Vec<CoreRunner> = (0..n)
            .map(|core_id| {
                let mut ctx = CoreContext::new(
                    core_id,
                    grid.linear_to_grid(core_id),
                    Rc::clone(&mem.hbm),
                    mem.get_lx(core_id),
                    mem.lx_scratchpads.clone(),
                );
                for (name, val) in &seeds[core_id] {
                    ctx.set_value(name, val.clone());
                }
                CoreRunner {
                    ctx,
                    op_idx: 0,
                    active: None,
                    dies_at: Rc::clone(&dies_at),
                    #[cfg(metal)]
                    plan_key: plan_key(ops),
                }
            })
            .collect();

        let mut messages: HashMap<(usize, usize), VecDeque<Tile>> = HashMap::new();
        let mut waiting: HashMap<usize, usize> = HashMap::new();
        let mut done = vec![false; n];

        let advance_one = |runners: &mut Vec<CoreRunner>,
                           messages: &mut HashMap<(usize, usize), VecDeque<Tile>>,
                           waiting: &mut HashMap<usize, usize>,
                           done: &mut [bool],
                           core_id: usize,
                           incoming: Option<Tile>|
         -> Result<(), String> {
            let poll = runners[core_id].step(ops, &env, incoming)?;
            for (dst, tile) in runners[core_id].ctx.drain_outbox() {
                messages.entry((core_id, dst)).or_default().push_back(tile);
            }
            match poll {
                Poll::Block(src) => {
                    waiting.insert(core_id, src);
                }
                Poll::Done => {
                    done[core_id] = true;
                }
            }
            Ok(())
        };

        for core_id in 0..n {
            advance_one(
                &mut runners,
                &mut messages,
                &mut waiting,
                &mut done,
                core_id,
                None,
            )
            .unwrap();
        }
        let mut guard = 0;
        while done.iter().any(|d| !d) {
            let mut progressed = false;
            for core_id in 0..n {
                if done[core_id] {
                    continue;
                }
                if let Some(&src) = waiting.get(&core_id)
                    && let Some(q) = messages.get_mut(&(src, core_id))
                    && let Some(tile) = q.pop_front()
                {
                    if q.is_empty() {
                        messages.remove(&(src, core_id));
                    }
                    waiting.remove(&core_id);
                    advance_one(
                        &mut runners,
                        &mut messages,
                        &mut waiting,
                        &mut done,
                        core_id,
                        Some(tile),
                    )
                    .unwrap();
                    progressed = true;
                }
            }
            assert!(progressed, "deadlock: {waiting:?}");
            guard += 1;
            assert!(guard < 1000, "runaway scheduler");
        }

        runners
            .iter()
            .map(|r| r.ctx.get_value(result_name).ok().cloned())
            .collect()
    }

    #[test]
    fn ring_reduce_4_cores_sums_to_all() {
        // Worked example from RingReduceBackend: starting 1,2,3,4 -> every core 10.
        let grid = GridExecutor::new((4, 1, 1));
        let mem = SpyreMemoryHierarchy::new(4);
        let group = Value::Tuple((0..4i64).map(Value::Index).collect());
        let seeds: Vec<Vec<(String, Value)>> = (0..4)
            .map(|c| {
                vec![
                    (
                        "t".into(),
                        Value::Tile(Tile::compute(vec![(c + 1) as f32], DType::F32, vec![1])),
                    ),
                    ("g".into(), group.clone()),
                ]
            })
            .collect();
        let ops = vec![Operation::new(Some("%r"), "ktdp.reduce", &["%t", "%g"])];
        let results = run_capturing(&grid, &mem, &ops, &seeds, "%r");
        for (c, r) in results.iter().enumerate() {
            match r {
                Some(Value::Tile(t)) => assert_eq!(t.as_f32().to_vec(), vec![10.0], "core {c}"),
                other => panic!("core {c}: expected Tile([10]), got {other:?}"),
            }
        }
    }

    #[test]
    fn ring_reduce_3_cores_vectors() {
        // 3 cores, 2-element tiles: [1,10],[2,20],[3,30] -> all [6,60].
        let grid = GridExecutor::new((3, 1, 1));
        let mem = SpyreMemoryHierarchy::new(3);
        let group = Value::Tuple((0..3i64).map(Value::Index).collect());
        let starts = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let seeds: Vec<Vec<(String, Value)>> = (0..3)
            .map(|c| {
                vec![
                    (
                        "t".into(),
                        Value::Tile(Tile::compute(starts[c].to_vec(), DType::F32, vec![2])),
                    ),
                    ("g".into(), group.clone()),
                ]
            })
            .collect();
        let ops = vec![Operation::new(Some("%r"), "ktdp.reduce", &["%t", "%g"])];
        let results = run_capturing(&grid, &mem, &ops, &seeds, "%r");
        for (c, r) in results.iter().enumerate() {
            match r {
                Some(Value::Tile(t)) => {
                    assert_eq!(t.as_f32().to_vec(), vec![6.0, 60.0], "core {c}")
                }
                other => panic!("core {c}: {other:?}"),
            }
        }
    }

    #[test]
    fn no_comm_ops_runs_each_core_to_completion() {
        let grid = GridExecutor::new((3, 1, 1));
        let mem = SpyreMemoryHierarchy::new(3);
        let dispatch = Dispatch::new();
        let ops = vec![
            Operation::new(Some("%a"), "arith.constant", &[])
                .with_attr("value", crate::ir::Attr::Int(7)),
            Operation::new(Some("%b"), "arith.addi", &["%a", "%a"]),
        ];
        execute_with_communication(&grid, &mem, &ops, &[], &dispatch, None, None).unwrap();
    }

    #[test]
    fn core_outside_group_is_identity() {
        // 2-core grid, group = {0} only; core 1 isn't in the group -> identity,
        // core 0 is a singleton group -> identity. Neither blocks.
        let grid = GridExecutor::new((2, 1, 1));
        let mem = SpyreMemoryHierarchy::new(2);
        let group = Value::Tuple(vec![Value::Index(0)]);
        let seeds: Vec<Vec<(String, Value)>> = (0..2)
            .map(|c| {
                vec![
                    (
                        "t".into(),
                        Value::Tile(Tile::compute(vec![(c + 1) as f32], DType::F32, vec![1])),
                    ),
                    ("g".into(), group.clone()),
                ]
            })
            .collect();
        let ops = vec![Operation::new(Some("%r"), "ktdp.reduce", &["%t", "%g"])];
        let results = run_capturing(&grid, &mem, &ops, &seeds, "%r");
        // each core keeps its own value (no reduction)
        match &results[0] {
            Some(Value::Tile(t)) => assert_eq!(t.as_f32().to_vec(), vec![1.0]),
            o => panic!("{o:?}"),
        }
        match &results[1] {
            Some(Value::Tile(t)) => assert_eq!(t.as_f32().to_vec(), vec![2.0]),
            o => panic!("{o:?}"),
        }
    }
}
