// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `scf` dialect handlers — port of `ktir_emulator/dialects/scf_ops.py` plus the
//! `ControlOps` loop/conditional helpers in `ktir_emulator/ops/control_ops.py`.
//!
//! Covers `scf.for` (induction var + iter_args + yield, with per-iteration
//! `push_scope`/`pop_scope` and iter_arg rebinding in the *parent* scope),
//! `scf.if` (then/else regions), and `scf.yield`.
//!
//! REGION/YIELD SEAM. The Python `execute_region` returns whatever the body's
//! last op produced; `scf.yield` returns a `_YieldResult` sentinel the loop
//! driver unwraps. The Rust contract's [`interpreter::execute_region`] returns
//! `Result<(), String>` and discards op results, so it cannot carry a yield out
//! of the body. We therefore run region bodies through a thin local executor
//! that drives [`interpreter::execute_op`] op-by-op and captures the value the
//! terminating `scf.yield` produces. Semantics are identical: the handler still
//! owns `push_scope`/`pop_scope`, and comm ops cannot appear in regions (so the
//! body never suspends), matching the spec.
//!
//! `scf.yield` is modeled as returning a `Value::Tuple(values)` (the
//! `_YieldResult` analogue). It has no SSA result name, so the value is not
//! bound into scope — it is observed only by the enclosing for/if driver.

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::env::ExecutionEnv;
use crate::interpreter::execute_op;
use crate::ir::{Attr, Operation, Scalar, Value};

pub fn register(d: &mut Dispatch) {
    d.register("scf.for", LatencyCategory::Zero, scf_for);
    d.register("scf.if", LatencyCategory::Zero, scf_if);
    d.register("scf.yield", LatencyCategory::Zero, scf_yield);
}

// ---------------------------------------------------------------------------
// scf.yield
// ---------------------------------------------------------------------------

/// `scf.yield %a, %b, ...` — gather the operand values and hand them back to
/// the enclosing loop/conditional driver. Mirrors `ControlOps.yield_op`: the
/// returned `Value::Tuple` is the `_YieldResult` sentinel analogue.
fn scf_yield(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let values: Vec<Value> = op
        .operands
        .iter()
        .map(|name| ctx.get_value(name).cloned())
        .collect::<Result<_, _>>()?;
    Ok(Some(Value::Tuple(values)))
}

// ---------------------------------------------------------------------------
// scf.if
// ---------------------------------------------------------------------------

/// `scf.if %cond { then } else { else }` — execute the selected branch in its
/// own scope. Mirrors `ControlOps.if_op`.
///
/// The branch body gets its own scope; body-local LX is freed on `pop_scope`.
/// If the branch yields Tile values, their LX is freed by `pop_scope` too — the
/// driver in `execute_op` re-tracks the bound result afterward.
fn scf_if(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("scf.if: missing condition operand".into());
    }
    let condition = as_bool(ctx.get_value(&op.operands[0])?, "scf.if")?;

    let region: &[Operation] = if condition {
        op.regions.first().map(Vec::as_slice).unwrap_or(&[])
    } else {
        op.regions.get(1).map(Vec::as_slice).unwrap_or(&[])
    };

    if region.is_empty() {
        return Ok(None);
    }

    // Branch body gets its own scope; body-local LX is freed on pop.
    ctx.push_scope();
    let result = run_region(region, ctx, env);
    ctx.pop_scope();
    let yielded = result?;

    // Mirror `unwrap_yield`: a single yielded value passes through bare; a
    // multi-value yield stays a tuple; no yield -> None.
    Ok(unwrap_yield(yielded))
}

// ---------------------------------------------------------------------------
// scf.for
// ---------------------------------------------------------------------------

/// `%r = scf.for %i = %lb to %ub step %step iter_args(%a = %init, ...) { body }`
///
/// Counted loop with optional loop-carried state. Mirrors `ControlOps.for_op`
/// and the `scf__for` handler glue: iter_args are bound in the *parent* scope
/// (they persist across iterations); each iteration body runs in a fresh scope
/// whose body-local LX is freed on `pop_scope`. Yielded values are fed back as
/// the next iteration's iter_arg bindings, with LX untracked/retracked across
/// the rebinding.
///
/// Returns the final iter_arg value (single) or a `Value::Tuple` (multiple);
/// `None` when there are no iter_args.
fn scf_for(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.len() < 3 {
        return Err(format!(
            "scf.for expects at least 3 operands (lb, ub, step), got {}",
            op.operands.len()
        ));
    }
    let lb = as_i64(ctx.get_value(&op.operands[0])?, "scf.for lb")?;
    let ub = as_i64(ctx.get_value(&op.operands[1])?, "scf.for ub")?;
    let step = as_i64(ctx.get_value(&op.operands[2])?, "scf.for step")?;

    let iter_var = match op.attributes.get("iter_var") {
        Some(Attr::Str(s)) => s.clone(),
        _ => "%i".to_string(),
    };

    let body_region: &[Operation] = op.regions.first().map(Vec::as_slice).unwrap_or(&[]);

    let iter_arg_names: Vec<String> = match op.attributes.get("iter_args") {
        Some(Attr::StrList(v)) => v.clone(),
        _ => Vec::new(),
    };
    let iter_init_operands = &op.operands[3..];
    let iter_init_values: Vec<Value> = iter_init_operands
        .iter()
        .map(|name| ctx.get_value(name).cloned())
        .collect::<Result<_, _>>()?;

    let result = for_op(
        ctx,
        lb,
        ub,
        step,
        &iter_var,
        body_region,
        env,
        &iter_arg_names,
        iter_init_values,
    )?;

    // for_op returns a Vec of final iter_arg values; unwrap when there is
    // exactly one (the common case for a single result var). Mirrors `scf__for`.
    match result {
        None => Ok(None),
        Some(mut vals) => {
            if vals.len() == 1 {
                Ok(Some(vals.pop().unwrap()))
            } else if vals.len() == iter_arg_names.len() {
                Ok(Some(Value::Tuple(vals)))
            } else {
                Err(format!(
                    "scf.for: expected {} results, got {}",
                    iter_arg_names.len(),
                    vals.len()
                ))
            }
        }
    }
}

/// Port of `ControlOps.for_op`. Returns the list of final iter_arg values, or
/// `None` when there are no iter_args.
#[allow(clippy::too_many_arguments)]
fn for_op(
    ctx: &mut CoreContext,
    lower_bound: i64,
    upper_bound: i64,
    step: i64,
    iter_var_name: &str,
    body_region: &[Operation],
    env: &ExecutionEnv,
    iter_arg_names: &[String],
    iter_init_values: Vec<Value>,
) -> Result<Option<Vec<Value>>, String> {
    // Bind initial iter_arg values in the *parent* scope. These persist across
    // iterations; body-local values do not.
    let mut current_values = iter_init_values;
    for (name, val) in iter_arg_names.iter().zip(current_values.iter()) {
        ctx.set_value(name, val.clone());
        if let Value::Tile(t) = val {
            ctx.track_lx(name, t.size_bytes() as i64)?;
        }
    }

    // `max(step, 1)` mirrors the Python guard against non-positive steps.
    let step = step.max(1);
    let mut i = lower_bound;
    while i < upper_bound {
        // New scope for this iteration's body-local values; pop frees their LX.
        ctx.push_scope();

        // Bind the iteration variable (a plain index, like Python's `int`).
        ctx.set_value(iter_var_name, Value::Index(i));

        // Execute the body, capturing the terminating yield (if any).
        let result = run_region(body_region, ctx, env);

        // Save yielded values before pop_scope() discards them.
        let yielded_values: Option<Vec<Value>> = match &result {
            Ok(Some(Value::Tuple(vals))) if !iter_arg_names.is_empty() => Some(vals.clone()),
            _ => None,
        };

        // Pop body scope — frees LX for all body-local Tiles, including any
        // Tiles that were yielded (they lived in this scope).
        ctx.pop_scope();
        result?; // surface any body error after the scope is cleaned up.

        // Re-bind yielded values as iter_args in the parent scope: untrack the
        // old Tile's LX and track the new one.
        if let Some(yielded) = yielded_values {
            for (name, val) in iter_arg_names.iter().zip(yielded.iter()) {
                ctx.untrack_lx(name);
                ctx.set_value(name, val.clone());
                if let Value::Tile(t) = val {
                    ctx.track_lx(name, t.size_bytes() as i64)?;
                }
            }
            current_values = yielded;
        }

        i += step;
    }

    if current_values.is_empty() {
        Ok(None)
    } else {
        Ok(Some(current_values))
    }
}

// ---------------------------------------------------------------------------
// region execution + yield plumbing
// ---------------------------------------------------------------------------

/// Drive a region body op-by-op, returning the value produced by its
/// terminating `scf.yield` (a `Value::Tuple`), or `None` if it does not yield.
///
/// This is the contract's `execute_region` with one addition: it threads the
/// terminator's value back out. Comm ops cannot appear in regions, so this
/// never suspends — matching `interpreter::execute_region`'s guarantee. The
/// caller owns `push_scope`/`pop_scope`.
fn run_region(
    ops: &[Operation],
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let mut last_yield = None;
    for op in ops {
        let produced = execute_op(op, ctx, env)?;
        if op.op_type == "scf.yield" {
            last_yield = produced;
        }
    }
    Ok(last_yield)
}

/// Mirror `_helpers.unwrap_yield`: a single yielded value passes through bare,
/// a multi-value yield stays a tuple, and a non-yield (None / empty) is `None`.
fn unwrap_yield(result: Option<Value>) -> Option<Value> {
    match result {
        Some(Value::Tuple(mut vals)) => match vals.len() {
            0 => None,
            1 => Some(vals.pop().unwrap()),
            _ => Some(Value::Tuple(vals)),
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------
// operand coercion helpers
// ---------------------------------------------------------------------------

fn as_i64(v: &Value, name: &str) -> Result<i64, String> {
    match v {
        Value::Index(i) => Ok(*i),
        Value::Scalar(s) => s.as_i64().ok_or_else(|| format!("{name}: non-int scalar")),
        other => Err(format!("{name}: expected index/int, got {other:?}")),
    }
}

fn as_bool(v: &Value, name: &str) -> Result<bool, String> {
    match v {
        Value::Scalar(Scalar::Bool(b)) => Ok(*b),
        Value::Scalar(Scalar::I32(i)) => Ok(*i != 0),
        Value::Scalar(Scalar::I64(i)) => Ok(*i != 0),
        Value::Index(i) => Ok(*i != 0),
        other => Err(format!("{name}: expected boolean condition, got {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::dtypes::DType;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};
    use crate::tile::Tile;

    fn run(ops: &[Operation], ctx: &mut CoreContext) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        execute_ops(ops, ctx, &env)
    }

    /// A `scf.for` op with a body region (and optional iter_args).
    #[allow(clippy::too_many_arguments)]
    fn for_op_ir(
        result: Option<&str>,
        lb: &str,
        ub: &str,
        step: &str,
        iter_var: &str,
        iter_inits: &[&str],
        iter_args: &[&str],
        body: Vec<Operation>,
    ) -> Operation {
        let mut operands = vec![lb, ub, step];
        operands.extend_from_slice(iter_inits);
        let mut op = Operation::new(result, "scf.for", &operands)
            .with_attr("iter_var", Attr::Str(iter_var.into()));
        if !iter_args.is_empty() {
            op = op.with_attr(
                "iter_args",
                Attr::StrList(iter_args.iter().map(|s| s.to_string()).collect()),
            );
        }
        op.regions = vec![body];
        op
    }

    // --- scf.for: counting ------------------------------------------------

    #[test]
    fn for_counts_iterations_via_iter_arg() {
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(5));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        // body: %s = addi %acc %one ; yield %s   (counts iterations)
        let one =
            Operation::new(Some("%one"), "arith.constant", &[]).with_attr("value", Attr::Int(1));
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%one"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let body = vec![one, add, yld];
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            body,
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 5), // 5 iterations
            other => panic!("expected I64(5), got {other:?}"),
        }
    }

    #[test]
    fn for_step_2_visits_three_times() {
        // 0..6 step 2 -> 3 iterations.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(6));
        ctx.set_value("%step", Value::Index(2));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        let one =
            Operation::new(Some("%one"), "arith.constant", &[]).with_attr("value", Attr::Int(1));
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%one"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![one, add, yld],
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 3),
            other => panic!("expected I64(3), got {other:?}"),
        }
    }

    #[test]
    fn for_induction_var_is_visible_in_body() {
        // running sum of the induction variable: acc += i over 0..4.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(4));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        // %s = addi %acc %i ; yield %s   (i is an Index, addi accepts it)
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%i"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![add, yld],
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            // 0 + (0+1+2+3) = 6
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 6),
            other => panic!("expected I64(6), got {other:?}"),
        }
    }

    // --- scf.for: iter_args -----------------------------------------------

    #[test]
    fn for_iter_args_running_sum() {
        // Mirrors test_for_op_iter_args_running_sum: acc starts 0, += i, 0..4.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(4));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%i"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![add, yld],
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 6),
            other => panic!("expected I64(6), got {other:?}"),
        }
    }

    #[test]
    fn for_no_iters_returns_none_and_leaves_no_result() {
        // ub == lb: zero iterations, no iter_args, no result binding.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(3));
        ctx.set_value("%ub", Value::Index(3));
        ctx.set_value("%step", Value::Index(1));
        let body = vec![Operation::new(None, "scf.yield", &[])];
        let f = for_op_ir(None, "%lb", "%ub", "%step", "%i", &[], &[], body);
        run(&[f], &mut ctx).unwrap();
        assert!(ctx.get_value("%i").is_err()); // induction var scope is gone
    }

    #[test]
    fn for_multi_iter_args_yields_tuple() {
        // Two scalar accumulators advanced independently.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(3));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%a0", Value::Scalar(Scalar::I64(0)));
        ctx.set_value("%b0", Value::Scalar(Scalar::I64(10)));
        let one =
            Operation::new(Some("%one"), "arith.constant", &[]).with_attr("value", Attr::Int(1));
        let na = Operation::new(Some("%na"), "arith.addi", &["%a", "%one"]);
        let nb = Operation::new(Some("%nb"), "arith.addi", &["%b", "%one"]);
        let yld = Operation::new(None, "scf.yield", &["%na", "%nb"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%a0", "%b0"],
            &["%a", "%b"],
            vec![one, na, nb, yld],
        );
        run(&[f], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Tuple(vals) => {
                assert_eq!(vals.len(), 2);
                assert!(matches!(vals[0], Value::Scalar(Scalar::I64(3)))); // 0+3
                assert!(matches!(vals[1], Value::Scalar(Scalar::I64(13)))); // 10+3
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn for_tile_iter_arg_lx_is_conserved() {
        // A Tile iter_arg: LX usage after the loop equals exactly one tile's
        // worth — the per-iteration body tile and old iter_arg tiles are freed.
        let mut ctx = single_core_context();
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(3));
        ctx.set_value("%step", Value::Index(1));
        // init tile: 4 x f32 = 16 bytes
        ctx.set_value(
            "%init",
            Value::Tile(Tile::compute(vec![0.0; 4], DType::F32, vec![4])),
        );
        ctx.track_lx("%init", 16).unwrap();
        let one = Operation::new(Some("%one"), "arith.constant", &[])
            .with_attr("value", Attr::Float(1.0));
        // each iteration yields a fresh tile via addf %acc %acc.
        let add = Operation::new(Some("%s"), "arith.addf", &["%acc", "%acc"]);
        let yld = Operation::new(None, "scf.yield", &["%s"]);
        let f = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![one, add, yld],
        );
        let used_before = ctx.lx.borrow().used;
        assert_eq!(used_before, 16); // only %init is tracked
        run(&[f], &mut ctx).unwrap();
        // After the loop, three tile-sized bindings remain, each 16 bytes (48):
        //   %init (still bound from setup), %acc (iter_arg, retracked each
        //   iteration), and %r (the loop result, tracked by execute_op).
        // The body-local %s tile is freed on pop_scope every iteration, so LX
        // does not grow with the iteration count — the conservation property.
        assert_eq!(ctx.lx.borrow().used, 48);
    }

    // --- scf.if -----------------------------------------------------------

    #[test]
    fn if_true_runs_then_branch() {
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(true)));
        let c =
            Operation::new(Some("%t"), "arith.constant", &[]).with_attr("value", Attr::Float(7.0));
        let yld = Operation::new(None, "scf.yield", &["%t"]);
        let e =
            Operation::new(Some("%f"), "arith.constant", &[]).with_attr("value", Attr::Float(9.0));
        let eyld = Operation::new(None, "scf.yield", &["%f"]);
        let mut iff = Operation::new(Some("%r"), "scf.if", &["%cond"]);
        iff.regions = vec![vec![c, yld], vec![e, eyld]];
        run(&[iff], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 7.0),
            other => panic!("expected F32(7.0), got {other:?}"),
        }
    }

    #[test]
    fn if_false_runs_else_branch() {
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(false)));
        let c =
            Operation::new(Some("%t"), "arith.constant", &[]).with_attr("value", Attr::Float(7.0));
        let yld = Operation::new(None, "scf.yield", &["%t"]);
        let e =
            Operation::new(Some("%f"), "arith.constant", &[]).with_attr("value", Attr::Float(9.0));
        let eyld = Operation::new(None, "scf.yield", &["%f"]);
        let mut iff = Operation::new(Some("%r"), "scf.if", &["%cond"]);
        iff.regions = vec![vec![c, yld], vec![e, eyld]];
        run(&[iff], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 9.0),
            other => panic!("expected F32(9.0), got {other:?}"),
        }
    }

    #[test]
    fn if_empty_branch_returns_none() {
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(false)));
        // then has a body, else is empty -> condition false selects empty -> None.
        let c =
            Operation::new(Some("%t"), "arith.constant", &[]).with_attr("value", Attr::Float(7.0));
        let yld = Operation::new(None, "scf.yield", &["%t"]);
        // no result name: op produces None, nothing bound.
        let mut iff = Operation::new(None, "scf.if", &["%cond"]);
        iff.regions = vec![vec![c, yld]]; // only a then-region
        run(&[iff], &mut ctx).unwrap();
        // no panic, and no stray binding leaked from the (unrun) then-branch.
        assert!(ctx.get_value("%t").is_err());
    }

    #[test]
    fn if_branch_local_lx_is_freed() {
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(true)));
        ctx.set_value(
            "%x",
            Value::Tile(Tile::compute(vec![1.0, 2.0], DType::F32, vec![2])),
        );
        // then: %y = addf %x %x ; yield nothing (no result) -> body-local tile freed.
        let add = Operation::new(Some("%y"), "arith.addf", &["%x", "%x"]);
        let yld = Operation::new(None, "scf.yield", &[]);
        let mut iff = Operation::new(None, "scf.if", &["%cond"]);
        iff.regions = vec![vec![add, yld]];
        let before = ctx.lx.borrow().used;
        run(&[iff], &mut ctx).unwrap();
        // %y (body-local) was freed on pop_scope; LX usage unchanged.
        assert_eq!(ctx.lx.borrow().used, before);
        assert!(ctx.get_value("%y").is_err());
    }

    #[test]
    fn yield_gathers_multiple_operands() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", Value::Scalar(Scalar::I64(1)));
        ctx.set_value("%b", Value::Scalar(Scalar::I64(2)));
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        let op = Operation::new(None, "scf.yield", &["%a", "%b"]);
        let out = scf_yield(&op, &mut ctx, &env).unwrap();
        match out {
            Some(Value::Tuple(vals)) => {
                assert_eq!(vals.len(), 2);
                assert!(matches!(vals[0], Value::Scalar(Scalar::I64(1))));
                assert!(matches!(vals[1], Value::Scalar(Scalar::I64(2))));
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn nested_for_inside_if() {
        // if(true) { %r = for ... acc += i ; yield acc } yield %r
        let mut ctx = single_core_context();
        ctx.set_value("%cond", Value::Scalar(Scalar::Bool(true)));
        ctx.set_value("%lb", Value::Index(0));
        ctx.set_value("%ub", Value::Index(4));
        ctx.set_value("%step", Value::Index(1));
        ctx.set_value("%init", Value::Scalar(Scalar::I64(0)));
        let add = Operation::new(Some("%s"), "arith.addi", &["%acc", "%i"]);
        let fyld = Operation::new(None, "scf.yield", &["%s"]);
        let inner_for = for_op_ir(
            Some("%r"),
            "%lb",
            "%ub",
            "%step",
            "%i",
            &["%init"],
            &["%acc"],
            vec![add, fyld],
        );
        let oyld = Operation::new(None, "scf.yield", &["%r"]);
        let mut iff = Operation::new(Some("%out"), "scf.if", &["%cond"]);
        iff.regions = vec![vec![inner_for, oyld]];
        run(&[iff], &mut ctx).unwrap();
        match ctx.get_value("%out").unwrap() {
            Value::Scalar(Scalar::I64(v)) => assert_eq!(*v, 6),
            other => panic!("expected I64(6), got {other:?}"),
        }
    }
}
