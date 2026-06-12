// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Execution orchestrator — port of `ktir_cpu/interpreter.py` + the per-op
//! driver from `grid.py`.
//!
//! This slice locks the execution contract every dialect handler builds
//! against: the handler signature `(op, &mut CoreContext, &ExecutionEnv)`, the
//! single-op driver `execute_op` (binds results, tracks Tile LX usage), and the
//! synchronous `execute_region` callback handlers use for scf bodies. The
//! multi-core comm scheduler (top-level only — see `comm.rs`) and HBM
//! input/output marshalling in `execute_function` are implement-phase fills
//! against these locked seams.

use std::collections::HashMap;
use std::rc::Rc;

use crate::codec;
use crate::context::CoreContext;
use crate::dialects::Dispatch;
use crate::dtypes::DType;
use crate::env::{ExecutionEnv, GridExecutor};
use crate::ir::{IRModule, Operation, Scalar, Value};
use crate::memory::{SpyreMemoryHierarchy, STICK_BYTES};

/// Execute one operation: dispatch, then bind its result (tracking LX for
/// Tiles). Mirrors `_execute_op`. Comm ops (which suspend) are driven by the
/// top-level scheduler, not here — see `comm.rs`.
pub fn execute_op(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let handler = env
        .dispatch
        .handler(&op.op_type)
        .ok_or_else(|| format!("no handler registered for op '{}'", op.op_type))?;
    let produced = handler(op, ctx, env)?;

    // Latency: record this op's cost before binding its result (operands are
    // still bound in scope; the result is not yet). Mirrors `_execute_op`.
    if let Some(tracker) = env.tracker {
        let operands: Vec<Option<Value>> = op
            .operands
            .iter()
            .map(|n| ctx.get_value(n).ok().cloned())
            .collect();
        let category = env.dispatch.latency_category(&op.op_type);
        tracker
            .borrow_mut()
            .record_op(ctx.core_id, &op.op_type, category, &produced, &operands);
    }

    // Multi-result op (`%a, %b = ...`): bind each name to one tuple element.
    // Mirrors Python `if isinstance(op.result, list) and isinstance(result, tuple)`.
    if let Some(crate::ir::Attr::StrList(names)) = op.attributes.get("result_names")
        && names.len() > 1
    {
        let Some(Value::Tuple(vals)) = &produced else {
            return Err(format!(
                "op '{}' has {} result names but did not produce a tuple",
                op.op_type,
                names.len()
            ));
        };
        if vals.len() != names.len() {
            return Err(format!(
                "op '{}': {} result names but produced {} values",
                op.op_type,
                names.len(),
                vals.len()
            ));
        }
        for (name, val) in names.iter().zip(vals) {
            if let Value::Tile(t) = val {
                ctx.track_lx(name, t.size_bytes() as i64)?;
            }
            ctx.set_value(name, val.clone());
        }
        return Ok(produced);
    }

    if let Some(name) = &op.result {
        match produced {
            Some(val) => {
                // Tiles occupy LX; bookkeeping values (TileRef, index, ...) don't.
                if let Value::Tile(t) = &val {
                    ctx.track_lx(name, t.size_bytes() as i64)?;
                }
                ctx.set_value(name, val.clone());
                return Ok(Some(val));
            }
            None => {
                return Err(format!(
                    "op '{}' has result {name} but produced no value",
                    op.op_type
                ));
            }
        }
    }
    Ok(produced)
}

/// Run a straight-line list of operations against a context. The function-body
/// driver and the body of `execute_region`.
pub fn execute_ops(
    ops: &[Operation],
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<(), String> {
    let mut i = 0;
    while i < ops.len() {
        // Peephole: fold `matmul` + a following elementwise op into one fused
        // NAX kernel (no host elementwise pass, one GPU dispatch). Only fires
        // when there's no latency tracker — the analytical model must still see
        // each op individually — so it speeds up pure execution / validation
        // without changing the latency report.
        #[cfg(metal)]
        if env.tracker.is_none()
            && let Some(advance) = try_fuse_matmul_epilogue(ops, i, ctx)?
        {
            i += advance;
            continue;
        }
        execute_op(&ops[i], ctx, env)?;
        i += 1;
    }
    Ok(())
}

/// Try to fuse `ops[i]` (a 2-operand `linalg.matmul` producing `%c`) with the
/// immediately following elementwise op that consumes `%c` (`linalg.add/mul/
/// sub/max/min`), running both as one fused NAX kernel and binding the
/// elementwise result. Returns `Some(2)` on a fuse, `None` to fall through to
/// normal op-by-op execution. Conservative: only fuses when `%c` is used by
/// nothing but that consumer, the shapes line up, and the size gate picks NAX.
#[cfg(metal)]
fn try_fuse_matmul_epilogue(
    ops: &[Operation],
    i: usize,
    ctx: &mut CoreContext,
) -> Result<Option<usize>, String> {
    use crate::metal_backend::Epilogue;

    let mm = &ops[i];
    if mm.op_type != "linalg.matmul" || mm.operands.len() != 2 {
        return Ok(None);
    }
    let Some(cname) = mm.result.as_deref() else { return Ok(None) };
    let Some(ep) = ops.get(i + 1) else { return Ok(None) };
    let Some(dname) = ep.result.as_deref() else { return Ok(None) };

    // The consumer must be a fusable binary elementwise op with `%c` as one
    // operand; `%e` is the other. For non-commutative ops the kernel computes
    // `c BINOP e`, so `%c` must be the FIRST operand.
    let Some(epi) = Epilogue::from_binary_op(&ep.op_type) else { return Ok(None) };
    if ep.operands.len() != 2 {
        return Ok(None);
    }
    let commutative = matches!(epi, Epilogue::ADD | Epilogue::MUL | Epilogue::MAX | Epilogue::MIN);
    let ename = if ep.operands[0] == cname {
        ep.operands[1].as_str()
    } else if ep.operands[1] == cname && commutative {
        ep.operands[0].as_str()
    } else {
        return Ok(None);
    };

    // `%c` must be dead after the consumer (else we'd still have to materialize
    // it). Reject if it reappears later or is used twice by the consumer itself.
    if ename == cname {
        return Ok(None);
    }
    let reused = ops[i + 2..].iter().any(|o| o.operands.iter().any(|x| x == cname))
        || ops[i + 1].operands.iter().filter(|x| x.as_str() == cname).count() > 1;
    if reused {
        return Ok(None);
    }

    // Pull A, B, E tiles; check 2-D, compatible inner dim, and E matching C.
    let (a, b, e) = (
        as_tile(ctx, &mm.operands[0])?,
        as_tile(ctx, &mm.operands[1])?,
        as_tile(ctx, ename)?,
    );
    if a.shape.len() != 2 || b.shape.len() != 2 || a.shape[1] != b.shape[0] {
        return Ok(None);
    }
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    if e.shape != [m, n] {
        return Ok(None);
    }
    let (a_data, b_data, e_data, dtype) =
        (a.data.clone(), b.data.clone(), e.data.clone(), a.dtype);

    // Fuse only if the gate picks NAX and the kernel runs; else fall through.
    let Some(out) = crate::metal_backend::metal_gemm_fused(m, k, n, &a_data, &b_data, &e_data, epi)
    else {
        return Ok(None);
    };
    let tile = crate::tile::Tile::compute(out, dtype, vec![m, n]);
    ctx.track_lx(dname, tile.size_bytes() as i64)?;
    ctx.set_value(dname, crate::ir::Value::Tile(tile));
    Ok(Some(2))
}

/// Borrow an SSA value as a `Tile`, or `Err` if it isn't one.
#[cfg(metal)]
fn as_tile<'a>(ctx: &'a CoreContext, name: &str) -> Result<&'a crate::tile::Tile, String> {
    match ctx.get_value(name)? {
        crate::ir::Value::Tile(t) => Ok(t),
        _ => Err(format!("fuse: {name} is not a tile")),
    }
}

/// Synchronous nested-region executor — the callback handlers use for scf.for
/// bodies / scf.if branches. Mirrors `execute_region`. Per the spec, comm ops
/// cannot appear in nested regions, so this never suspends. The caller (the op
/// handler) owns `push_scope`/`pop_scope`.
pub fn execute_region(
    ops: &[Operation],
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<(), String> {
    execute_ops(ops, ctx, env)
}

/// Build a single-core context for `grid_pos`/`core_id` over a fresh memory
/// hierarchy — the common setup for executing a `grid = [1]` function or a unit
/// test. Returns `(context, dispatch)` ready for `execute_ops`.
pub fn single_core_context() -> CoreContext {
    let mem = SpyreMemoryHierarchy::new(1);
    CoreContext::new(
        0,
        (0, 0, 0),
        Rc::clone(&mem.hbm),
        mem.get_lx(0),
        mem.lx_scratchpads.clone(),
    )
}

/// A function argument: a tensor (marshalled into HBM) or a scalar (bound
/// directly). Mirrors the `np.ndarray` vs scalar split in `execute_function`.
#[derive(Clone, Debug)]
pub enum Arg {
    /// f32 host data, narrowed to `dtype` on the way into HBM. The dtype-agnostic
    /// oracle path — convenient, but for an all-f16 model it pays an f32→f16
    /// narrow per input (and f16→f32 widen per output) and 2× host memory.
    Tensor { data: Vec<f32>, shape: Vec<usize>, dtype: DType },
    /// Pre-encoded typed bytes (already in `dtype` layout, e.g. f16), copied
    /// straight into HBM with no conversion — mirrors Spyre's typed host→AIU DMA.
    /// Use this to avoid the f32 round-trip for typed (f16/…) host buffers.
    TensorBytes { data: Vec<u8>, shape: Vec<usize>, dtype: DType },
    Scalar(Scalar),
}

/// A tensor read back from HBM after execution.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Execute a function with tensor + scalar arguments and return every tensor
/// argument read back from HBM. Port of `KTIRInterpreter.execute_function`
/// (latency tracking is an optional add-on, see `latency.rs`). Cores are driven
/// by the comm scheduler (`comm_sched`), so cross-core collectives work; cores
/// with no comm op just run their body to completion against shared HBM.
pub fn execute_function(
    module: &IRModule,
    func_name: &str,
    args: &[(&str, Arg)],
) -> Result<HashMap<String, Output>, String> {
    let func = module.get_function(func_name)?;
    let (gx, gy, gz) = func.grid;
    let num_cores = gx * gy * gz;

    let mem = SpyreMemoryHierarchy::new(num_cores.max(1));
    let grid = GridExecutor::new(func.grid);
    let dispatch = Dispatch::new();

    let (input_ptrs, tensor_meta) = marshal_inputs(&mem, args);

    // Drive all cores via the comm scheduler (cores with no comm op simply run
    // to completion; ring/collective ops suspend and resume through it).
    crate::comm_sched::execute_with_communication(
        &grid,
        &mem,
        &func.operations,
        &input_ptrs,
        &dispatch,
        None,
    )?;

    read_back(&mem, tensor_meta)
}

/// Like [`execute_function`], but records per-op latency and returns the report
/// alongside the outputs. Port of running `KTIRInterpreter` with a
/// `latency_config`. Every op (including region-nested ops, via the shared
/// `ExecutionEnv`) is metered; comm ops are charged by the scheduler.
pub fn execute_function_with_latency(
    module: &IRModule,
    func_name: &str,
    args: &[(&str, Arg)],
    config: crate::latency::HardwareConfig,
) -> Result<(HashMap<String, Output>, crate::latency::LatencyReport), String> {
    use std::cell::RefCell;

    let func = module.get_function(func_name)?;
    let (gx, gy, gz) = func.grid;
    let num_cores = gx * gy * gz;

    let mem = SpyreMemoryHierarchy::new(num_cores.max(1));
    let grid = GridExecutor::new(func.grid);
    let dispatch = Dispatch::new();
    let tracker = RefCell::new(crate::latency::LatencyTracker::new(config));

    let (input_ptrs, tensor_meta) = marshal_inputs(&mem, args);

    crate::comm_sched::execute_with_communication(
        &grid,
        &mem,
        &func.operations,
        &input_ptrs,
        &dispatch,
        Some(&tracker),
    )?;

    let outputs = read_back(&mem, tensor_meta)?;
    let report = tracker.borrow().report();
    Ok((outputs, report))
}

/// Tensor read-back metadata: `(name, stick, n_elements, shape, dtype)`.
type TensorMeta = (String, i64, usize, Vec<usize>, DType);

/// Marshal tensor args into HBM and return `(input_ptrs, tensor_meta)`.
/// Shared by the plain and latency-tracked execution paths.
fn marshal_inputs(
    mem: &SpyreMemoryHierarchy,
    args: &[(&str, Arg)],
) -> (Vec<(String, Value)>, Vec<TensorMeta>) {
    let mut input_ptrs: Vec<(String, Value)> = Vec::new();
    let mut tensor_meta: Vec<TensorMeta> = Vec::new();
    // Allocate an HBM stick, write `bytes`, and record the read-back metadata.
    fn place(
        mem: &SpyreMemoryHierarchy,
        input_ptrs: &mut Vec<(String, Value)>,
        tensor_meta: &mut Vec<TensorMeta>,
        name: &str,
        bytes: Vec<u8>,
        shape: &[usize],
        dtype: DType,
    ) {
        let stick = {
            let mut hbm = mem.hbm.borrow_mut();
            let stick = hbm.allocate(bytes.len() as i64);
            hbm.write_bytes(stick * STICK_BYTES, &bytes);
            stick
        };
        input_ptrs.push((name.to_string(), Value::Index(stick)));
        tensor_meta.push((name.to_string(), stick, shape.iter().product(), shape.to_vec(), dtype));
    }
    for (name, arg) in args {
        match arg {
            // f32 host data: narrow to `dtype` on the way in.
            Arg::Tensor { data, shape, dtype } => {
                place(mem, &mut input_ptrs, &mut tensor_meta, name, codec::encode(data, *dtype), shape, *dtype)
            }
            // Pre-encoded typed bytes: straight to HBM, no conversion.
            Arg::TensorBytes { data, shape, dtype } => {
                place(mem, &mut input_ptrs, &mut tensor_meta, name, data.clone(), shape, *dtype)
            }
            Arg::Scalar(s) => input_ptrs.push((name.to_string(), Value::Scalar(*s))),
        }
    }
    (input_ptrs, tensor_meta)
}

/// Opt-in GPU/Spyre-faithful (**f16**) execution of a pure-SPMD grid: step all
/// cores in lockstep and COMBINE their shared-weight `linalg.matmul`s into one
/// zero-copy NAX dispatch (the grid's many small matmuls become one tall one —
/// 1.2–2.6× over a serial AMX loop). Restricted to no-comm, straight-line
/// (region-free) functions on an M5; returns `Err` otherwise so the caller can
/// fall back to [`execute_function`].
///
/// The NAX kernel runs in **f16** — Spyre's matmul precision, and exactly what
/// the interpreter rounds every tile to (`f32` accumulate → f16). So results
/// match the f32/`execute_function` path to f16 tolerance (only the GEMM
/// accumulation order differs). Kept opt-in for now while the lockstep executor
/// is young; it is precision-faithful, not a lossy mode.
#[cfg(metal)]
pub fn execute_function_gpu(
    module: &IRModule,
    func_name: &str,
    args: &[(&str, Arg)],
) -> Result<HashMap<String, Output>, String> {
    use crate::metal_backend::NaxGemm;

    let func = module.get_function(func_name)?;
    let (gx, gy, gz) = func.grid;
    let num_cores = gx * gy * gz;
    let ops = &func.operations;

    // Applicable only to a multi-core, comm-free, straight-line SPMD body.
    if num_cores <= 1
        || ops
            .iter()
            .any(|o| crate::comm_sched::is_comm_op(&o.op_type) || !o.regions.is_empty())
    {
        return Err("execute_function_gpu: not a pure-SPMD straight-line grid".into());
    }
    let gemm = NaxGemm::new()?; // Err on non-M5 / no device -> caller falls back

    let mem = SpyreMemoryHierarchy::new(num_cores);
    let grid = GridExecutor::new(func.grid);
    let dispatch = Dispatch::new();
    let env = ExecutionEnv::new(&dispatch, &grid);
    let (input_ptrs, tensor_meta) = marshal_inputs(&mem, args);

    let mut ctxs: Vec<CoreContext> = (0..num_cores)
        .map(|c| {
            let mut ctx = CoreContext::new(
                c,
                grid.linear_to_grid(c),
                Rc::clone(&mem.hbm),
                mem.get_lx(c),
                mem.lx_scratchpads.clone(),
            );
            for (name, val) in &input_ptrs {
                ctx.set_value(name, val.clone());
            }
            ctx
        })
        .collect();

    for op in ops {
        // Combine a shared-weight 2-operand matmul across all cores into one
        // dispatch; fall through to per-core execution if it doesn't apply.
        if op.op_type == "linalg.matmul"
            && op.operands.len() == 2
            && try_combine_matmul(op, &mut ctxs, &gemm)?
        {
            continue;
        }
        for ctx in &mut ctxs {
            execute_op(op, ctx, &env)?;
        }
    }
    read_back(&mem, tensor_meta)
}

/// Combine `op` (a 2-operand `linalg.matmul`) across all cores when every core's
/// weight operand B is identical: stack the per-core A panels into one tall
/// GEMM, run it zero-copy on NAX, and scatter the row-blocks back. Returns
/// `Ok(true)` if combined, `Ok(false)` to fall back to per-core execution.
#[cfg(metal)]
fn try_combine_matmul(
    op: &Operation,
    ctxs: &mut [CoreContext],
    gemm: &crate::metal_backend::NaxGemm,
) -> Result<bool, String> {
    use crate::metal_backend::Epilogue;
    use crate::tile::Tile;

    let result = match op.result.as_deref() {
        Some(r) => r,
        None => return Ok(false),
    };
    // Read core 0's operands to fix the shapes and the shared weights.
    let (a0, b0) = (as_tile(&ctxs[0], &op.operands[0])?, as_tile(&ctxs[0], &op.operands[1])?);
    if a0.shape.len() != 2 || b0.shape.len() != 2 || a0.shape[1] != b0.shape[0] {
        return Ok(false);
    }
    let (m, k, n) = (a0.shape[0], a0.shape[1], b0.shape[1]);
    let dtype = a0.dtype;
    let shared_b = b0.data.clone();
    let a_shape = a0.shape.clone();
    let b_shape = b0.shape.clone();

    // Gather A panels; bail (fall back) unless every core shares B exactly.
    let mut a_stack = Vec::with_capacity(ctxs.len() * m * k);
    for ctx in ctxs.iter() {
        let a = as_tile(ctx, &op.operands[0])?;
        let b = as_tile(ctx, &op.operands[1])?;
        if a.shape != a_shape || b.shape != b_shape || b.data != shared_b {
            return Ok(false);
        }
        a_stack.extend_from_slice(&a.data);
    }

    // One zero-copy NAX dispatch for the whole grid's matmul.
    let ua = gemm.unified_from(&a_stack)?;
    let ub = gemm.unified_from(&shared_b)?;
    let mut uc = gemm.unified(ctxs.len() * m * n)?;
    gemm.matmul_unified(ctxs.len() * m, k, n, &ua, &ub, &mut uc, None, Epilogue::NONE)?;

    // Scatter each core's row-block back as its matmul result.
    let c = uc.as_slice();
    for (i, ctx) in ctxs.iter_mut().enumerate() {
        let block = c[i * m * n..(i + 1) * m * n].to_vec();
        let tile = Tile::compute(block, dtype, vec![m, n]);
        ctx.track_lx(result, tile.size_bytes() as i64)?;
        ctx.set_value(result, Value::Tile(tile));
    }
    Ok(true)
}

/// Read every tensor arg back out of HBM into an `Output`.
fn read_back(
    mem: &SpyreMemoryHierarchy,
    tensor_meta: Vec<TensorMeta>,
) -> Result<HashMap<String, Output>, String> {
    let mut outputs = HashMap::new();
    for (name, stick, n, shape, dtype) in tensor_meta {
        let nbytes = n * dtype.bytes_per_elem();
        let bytes = mem.hbm.borrow().read_bytes(stick * STICK_BYTES, nbytes);
        outputs.insert(name, Output { data: codec::decode(&bytes, n, dtype), shape, dtype });
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::dtypes::DType;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::ir::{Attr, Operation, Scalar};
    use crate::tile::Tile;

    fn run(ops: &[Operation]) -> CoreContext {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        execute_ops(ops, &mut ctx, &env).unwrap();
        ctx
    }

    #[test]
    fn scalar_constant_fold_chain() {
        let ops = vec![
            Operation::new(Some("%a"), "arith.constant", &[]).with_attr("value", Attr::Float(2.0)),
            Operation::new(Some("%b"), "arith.constant", &[]).with_attr("value", Attr::Float(3.0)),
            Operation::new(Some("%c"), "arith.addf", &["%a", "%b"]),
            Operation::new(Some("%d"), "arith.mulf", &["%c", "%a"]),
        ];
        let ctx = run(&ops);
        match ctx.get_value("%d").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 10.0),
            other => panic!("expected F32(10.0), got {other:?}"),
        }
    }

    #[test]
    fn elementwise_tile_add_tracks_lx() {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        ctx.set_value("%x", Value::Tile(Tile::compute(vec![1.0, 2.0, 3.0], DType::F32, vec![3])));
        ctx.set_value("%y", Value::Tile(Tile::compute(vec![10.0, 20.0, 30.0], DType::F32, vec![3])));
        let ops = vec![Operation::new(Some("%z"), "arith.addf", &["%x", "%y"])];
        execute_ops(&ops, &mut ctx, &env).unwrap();
        match ctx.get_value("%z").unwrap() {
            Value::Tile(t) => assert_eq!(t.data, vec![11.0, 22.0, 33.0]),
            other => panic!("expected tile, got {other:?}"),
        }
        // the result Tile was tracked in LX (3 * f32 = 12 bytes)
        assert_eq!(ctx.lx.borrow().used, 12);
    }

    #[test]
    fn unknown_op_errors() {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        let ops = vec![Operation::new(Some("%z"), "ktdp.not_yet", &[])];
        let err = execute_ops(&ops, &mut ctx, &env).unwrap_err();
        assert!(err.contains("no handler registered"));
    }
}
