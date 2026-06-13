// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Cross-core communication scheduler — port of
//! `GridExecutor.execute_with_communication` + `CoreExecutionStack` from
//! `ktir_cpu/grid.py`, and the ring all-reduce from `ktir_cpu/ops/comm_ops.py`.
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
use crate::ir::{Operation, Value};
use crate::memory::SpyreMemoryHierarchy;
use crate::tile::Tile;

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
        .data
        .iter()
        .zip(b.data.iter())
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
    dies_at: Rc<Vec<Vec<String>>>,
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
        #[cfg(metal)]
        let gpu_offload = env.tracker.is_none()
            && env.grid.num_cores == 1
            && std::env::var_os("KTIR_NO_GPU_GEMM").is_none();
        #[cfg(metal)]
        let matmul_sched = if gpu_offload {
            crate::metal_backend::matmul_loop_schedule(ops)
        } else {
            std::collections::HashMap::new()
        };
        // MLX-style map-window fusion: each maximal run of fusable elementwise/
        // cast/broadcast ops runs as ONE fused GPU kernel (instead of op-by-op on
        // the interpreter). Same gating as the matmul-loop offload above. The plan
        // maps each window's TRIGGER op (its last op) -> the compiled kernel, and a
        // SKIP set of all window op indices; non-trigger window ops are subsumed by
        // the fused kernel (their values come from it) and are not executed.
        #[cfg(metal)]
        let (map_triggers, map_skip) = if gpu_offload
            && std::env::var_os("KTIR_NO_GPU_MAP").is_none()
        {
            crate::metal_backend::map_fusion_plan(ops)
        } else {
            (std::collections::HashMap::new(), std::collections::HashSet::new())
        };

        // Window op indices whose liveness reclaim is deferred to the window's
        // trigger (so a fused kernel's live-ins survive until it has read them).
        #[cfg(metal)]
        let mut pending_skip: Vec<usize> = Vec::new();

        // Run remaining top-level ops.
        while self.op_idx < ops.len() {
            let op = &ops[self.op_idx];
            let this_idx = self.op_idx;
            self.op_idx += 1;
            // Offload a recognized K-loop to a single GPU GEMM; on any failure
            // fall through to the interpreter (correctness preserved).
            #[cfg(metal)]
            if op.op_type == "scf.for"
                && let Some(info) = op.result.as_deref().and_then(|r| matmul_sched.get(r))
                && crate::metal_backend::run_matmul_loop_gpu(info, &mut self.ctx).is_ok()
            {
                if let Some(dead) = self.dies_at.get(this_idx) {
                    for name in dead {
                        self.ctx.forget(name);
                    }
                }
                continue;
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
                crate::metal_backend::run_map_region_gpu(mrk, &mut self.ctx)?;
                pending_skip.push(this_idx);
                for idx in pending_skip.drain(..) {
                    if let Some(dead) = self.dies_at.get(idx) {
                        for name in dead {
                            self.ctx.forget(name);
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
                for name in dead {
                    self.ctx.forget(name);
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
) -> Result<(), String> {
    let env = match tracker {
        Some(t) => ExecutionEnv::with_tracker(dispatch, grid, t),
        None => ExecutionEnv::new(dispatch, grid),
    };
    let num_cores = grid.num_cores.max(1);

    // Liveness for LX reclaim (identical for every core) — see `CoreRunner::dies_at`.
    let dies_at = Rc::new(compute_dies_at(ops));

    let mut runners: BTreeMap<usize, CoreRunner> = BTreeMap::new();
    for core_id in 0..num_cores {
        let mut ctx = CoreContext::new(
            core_id,
            grid.linear_to_grid(core_id),
            Rc::clone(&mem.hbm),
            mem.get_lx(core_id),
            mem.lx_scratchpads.clone(),
        );
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
                Some(Value::Tile(t)) => assert_eq!(t.data.to_vec(), vec![10.0], "core {c}"),
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
                Some(Value::Tile(t)) => assert_eq!(t.data.to_vec(), vec![6.0, 60.0], "core {c}"),
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
        execute_with_communication(&grid, &mem, &ops, &[], &dispatch, None).unwrap();
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
            Some(Value::Tile(t)) => assert_eq!(t.data.to_vec(), vec![1.0]),
            o => panic!("{o:?}"),
        }
        match &results[1] {
            Some(Value::Tile(t)) => assert_eq!(t.data.to_vec(), vec![2.0]),
            o => panic!("{o:?}"),
        }
    }
}
