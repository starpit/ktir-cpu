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
    for op in ops {
        execute_op(op, ctx, env)?;
    }
    Ok(())
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
    Tensor { data: Vec<f32>, shape: Vec<usize>, dtype: DType },
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

    // Allocate tensor inputs in HBM (stick index bound as the pointer); scalars
    // bind directly. Record tensor metadata for read-back.
    let mut input_ptrs: Vec<(String, Value)> = Vec::new();
    let mut tensor_meta: Vec<(String, i64, usize, Vec<usize>, DType)> = Vec::new(); // name, stick, n, shape, dtype
    for (name, arg) in args {
        match arg {
            Arg::Tensor { data, shape, dtype } => {
                let bytes = codec::encode(data, *dtype);
                let stick = {
                    let mut hbm = mem.hbm.borrow_mut();
                    let stick = hbm.allocate(bytes.len() as i64);
                    hbm.write_bytes(stick * STICK_BYTES, &bytes);
                    stick
                };
                input_ptrs.push((name.to_string(), Value::Index(stick)));
                tensor_meta.push((
                    name.to_string(),
                    stick,
                    shape.iter().product(),
                    shape.clone(),
                    *dtype,
                ));
            }
            Arg::Scalar(s) => input_ptrs.push((name.to_string(), Value::Scalar(*s))),
        }
    }

    // Drive all cores via the comm scheduler (cores with no comm op simply run
    // to completion; ring/collective ops suspend and resume through it).
    crate::comm_sched::execute_with_communication(
        &grid,
        &mem,
        &func.operations,
        &input_ptrs,
        &dispatch,
    )?;

    // Read tensor args back from HBM.
    let mut outputs = HashMap::new();
    for (name, stick, n, shape, dtype) in tensor_meta {
        let nbytes = n * dtype.bytes_per_elem();
        let bytes = mem.hbm.borrow().read_bytes(stick * STICK_BYTES, nbytes);
        outputs.insert(
            name,
            Output { data: codec::decode(&bytes, n, dtype), shape, dtype },
        );
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
        let env = ExecutionEnv { dispatch: &dispatch, grid: &grid };
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
        let env = ExecutionEnv { dispatch: &dispatch, grid: &grid };
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
        let env = ExecutionEnv { dispatch: &dispatch, grid: &grid };
        let mut ctx = single_core_context();
        let ops = vec![Operation::new(Some("%z"), "ktdp.not_yet", &[])];
        let err = execute_ops(&ops, &mut ctx, &env).unwrap_err();
        assert!(err.contains("no handler registered"));
    }
}
