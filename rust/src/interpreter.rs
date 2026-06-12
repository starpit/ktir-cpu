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

use crate::context::CoreContext;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Value};

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
    use crate::memory::SpyreMemoryHierarchy;
    use std::rc::Rc;
    let mem = SpyreMemoryHierarchy::new(1);
    CoreContext::new(
        0,
        (0, 0, 0),
        Rc::clone(&mem.hbm),
        mem.get_lx(0),
        mem.lx_scratchpads.clone(),
    )
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
