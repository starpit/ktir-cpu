// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `linalg` dialect handlers — Rust port of `ktir_cpu/dialects/linalg_ops.py`.
//!
//! Ports the structured-op family: `matmul`, `batch_matmul`, `generic`,
//! `reduce`, `transpose`, `broadcast`, `fill`, `index`, and the `yield`
//! terminator. `generic` and `reduce` are *zero-cost orchestrators*: the cost
//! lives in the ops of their combiner region, which we execute via
//! `execute_region` exactly as Python executes them through `env.execute_region`.
//!
//! ## Region / yield handling
//!
//! The locked `interpreter::execute_region` returns `()`, not a value — so this
//! module threads the yielded value through a per-scope sentinel SSA binding,
//! [`YIELD_KEY`]. `linalg.yield %v` binds `%v` under that key in the current
//! scope; [`run_region`] reads it back out before the caller pops the scope.
//! This mirrors Python's `_YieldResult` / `unwrap_yield` plumbing, kept local to
//! linalg since the shared scf yield seam is not yet in the Rust tree.
//!
//! ## N-dimensional tiles
//!
//! `Tile` stores a flat `Vec<f32>` + a `shape`; this module carries the
//! row-major index arithmetic NumPy gives for free in Python (strides, broadcast,
//! transpose, axis reductions) as small local helpers.

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::interpreter::execute_region;
use crate::ir::{Attr, Operation, Scalar, Value};
use crate::tile::Tile;

/// Sentinel scope key under which `linalg.yield` parks its yielded value so the
/// region driver can recover it after `execute_region` (which itself returns
/// `()`). Chosen to never collide with a real SSA name.
const YIELD_KEY: &str = "__linalg_yield__";

/// Sentinel scope key holding the current `linalg.generic` iteration shape, so
/// `linalg.index` can build its broadcasting index array. Mirrors the Python
/// `__linalg_shape__` binding.
const SHAPE_KEY: &str = "__linalg_shape__";

pub fn register(d: &mut Dispatch) {
    // generic/matmul carry real float compute cost in Python (LC.COMPUTE_FLOAT /
    // LC.COMPUTE_MATMUL); map both onto ComputeFloat, the closest present
    // variant. reduce is LC.ZERO (cost lives in its region's ops).
    d.register("linalg.matmul", LatencyCategory::ComputeFloat, matmul);
    d.register("linalg.batch_matmul", LatencyCategory::ComputeFloat, batch_matmul);
    d.register("linalg.generic", LatencyCategory::ComputeFloat, generic);
    d.register("linalg.reduce", LatencyCategory::Zero, reduce);
    d.register("linalg.transpose", LatencyCategory::Zero, transpose);
    d.register("linalg.broadcast", LatencyCategory::Zero, broadcast);
    d.register("linalg.fill", LatencyCategory::Zero, fill);
    d.register("linalg.index", LatencyCategory::Zero, index);
    d.register("linalg.yield", LatencyCategory::Zero, yield_op);
}

// ===========================================================================
// fill / broadcast / transpose
// ===========================================================================

/// `%r = linalg.fill ins(%scalar) outs(%init)` — fill a tile with a scalar.
fn fill(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let scalar = ctx.get_value(&op.operands[0])?;
    let scalar_val = as_f32(scalar, "linalg.fill scalar")?;
    let out = expect_tile(ctx.get_value(&op.operands[1])?, "linalg.fill outs")?;
    let data = vec![scalar_val; out.data.len()];
    Ok(Some(Value::Tile(Tile::compute(data, out.dtype, out.shape.clone()))))
}

/// `%r = linalg.broadcast ins(%x) outs(%init) dimensions = [...]`.
///
/// Expands `dimensions` on the input then broadcasts to the outs shape. Mirrors
/// `np.expand_dims` over sorted dims followed by `np.broadcast_to`.
fn broadcast(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let inp = expect_tile(ctx.get_value(&op.operands[0])?, "linalg.broadcast ins")?.clone();
    let out = expect_tile(ctx.get_value(&op.operands[1])?, "linalg.broadcast outs")?;
    let out_shape = out.shape.clone();
    let out_dtype = inp.dtype;

    let mut dims = int_list_attr(op, "dimensions").cloned().unwrap_or_default();
    dims.sort_unstable();

    // Build the input's expanded shape: start from inp.shape, insert size-1 axes
    // at each broadcast dimension (sorted, so earlier inserts don't shift later).
    let mut shape: Vec<usize> = inp.shape.clone();
    for &d in &dims {
        let d = d as usize;
        if d > shape.len() {
            return Err(format!("linalg.broadcast: dim {d} out of range for shape {shape:?}"));
        }
        shape.insert(d, 1);
    }

    let data = broadcast_to(&inp.data, &shape, &out_shape)
        .ok_or_else(|| format!("linalg.broadcast: cannot broadcast {shape:?} to {out_shape:?}"))?;
    Ok(Some(Value::Tile(Tile::compute(data, out_dtype, out_shape))))
}

/// `%r = linalg.transpose ins(%x) outs(%y) permutation = [...]`.
fn transpose(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let inp = expect_tile(ctx.get_value(&op.operands[0])?, "linalg.transpose ins")?.clone();
    let perm = int_list_attr(op, "permutation")
        .ok_or("linalg.transpose: missing permutation attribute")?
        .iter()
        .map(|&p| p as usize)
        .collect::<Vec<_>>();
    if perm.len() != inp.shape.len() {
        return Err(format!(
            "linalg.transpose: permutation rank {} != input rank {}",
            perm.len(),
            inp.shape.len()
        ));
    }

    let new_shape: Vec<usize> = perm.iter().map(|&p| inp.shape[p]).collect();
    let in_strides = row_major_strides(&inp.shape);
    let mut data = vec![0.0f32; inp.data.len()];
    // For each output multi-index, the source element is at the permuted axes:
    // out[idx] = in[ idx mapped back through perm ].
    for (out_lin, slot) in data.iter_mut().enumerate() {
        let out_idx = unravel(out_lin, &new_shape);
        // out_idx[k] is the coordinate along input axis perm[k].
        let mut src = 0usize;
        for (k, &coord) in out_idx.iter().enumerate() {
            src += coord * in_strides[perm[k]];
        }
        *slot = inp.data[src];
    }
    Ok(Some(Value::Tile(Tile::compute(data, inp.dtype, new_shape))))
}

// ===========================================================================
// matmul / batch_matmul
// ===========================================================================

/// `%r = linalg.matmul ins(%A, %B) outs(%C)` -> `C + A @ B`.
fn matmul(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let a = expect_tile(ctx.get_value(&op.operands[0])?, "linalg.matmul A")?.clone();
    let b = expect_tile(ctx.get_value(&op.operands[1])?, "linalg.matmul B")?.clone();
    let mut result = matmul2d(&a, &b)?;

    // Accumulate into outs (operands[2] = C) when present: result = C + A@B.
    if op.operands.len() > 2
        && let Value::Tile(c) = ctx.get_value(&op.operands[2])? {
            if c.shape != result.shape {
                return Err(format!(
                    "linalg.matmul: outs shape {:?} != A@B shape {:?}",
                    c.shape, result.shape
                ));
            }
            for (r, &cv) in result.data.iter_mut().zip(&c.data) {
                *r += cv;
            }
            result.dtype = c.dtype;
        }
    Ok(Some(Value::Tile(result)))
}

/// `%r = linalg.batch_matmul ins(%A, %B) outs(%C)` over the leading batch dim.
fn batch_matmul(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let a = expect_tile(ctx.get_value(&op.operands[0])?, "linalg.batch_matmul A")?.clone();
    let b = expect_tile(ctx.get_value(&op.operands[1])?, "linalg.batch_matmul B")?.clone();
    if a.shape.len() != 3 || b.shape.len() != 3 {
        return Err(format!(
            "linalg.batch_matmul: expected 3-D operands, got {:?} and {:?}",
            a.shape, b.shape
        ));
    }
    let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
    if b.shape[0] != batch || b.shape[1] != k {
        return Err(format!(
            "linalg.batch_matmul: incompatible shapes {:?} and {:?}",
            a.shape, b.shape
        ));
    }
    let n = b.shape[2];
    let mut data = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let r_off = bi * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a.data[a_off + i * k + kk] * b.data[b_off + kk * n + j];
                }
                data[r_off + i * n + j] = acc;
            }
        }
    }
    let mut result = Tile::compute(data, a.dtype, vec![batch, m, n]);

    if op.operands.len() > 2
        && let Value::Tile(c) = ctx.get_value(&op.operands[2])? {
            for (r, &cv) in result.data.iter_mut().zip(&c.data) {
                *r += cv;
            }
            result.dtype = c.dtype;
        }
    Ok(Some(Value::Tile(result)))
}

/// 2-D matmul `A @ B` keeping A's dtype. A is [M, K], B is [K, N].
fn matmul2d(a: &Tile, b: &Tile) -> Result<Tile, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(format!(
            "linalg.matmul: expected 2-D operands, got {:?} and {:?}",
            a.shape, b.shape
        ));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    if b.shape[0] != k {
        return Err(format!(
            "linalg.matmul: inner dims disagree: {:?} @ {:?}",
            a.shape, b.shape
        ));
    }
    let n = b.shape[1];
    let mut data = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a.data[i * k + kk] * b.data[kk * n + j];
            }
            data[i * n + j] = acc;
        }
    }
    Ok(Tile::compute(data, a.dtype, vec![m, n]))
}

// ===========================================================================
// reduce
// ===========================================================================

/// `%r = linalg.reduce ins(%x) outs(%init) dimensions = [d] { <combiner> }`.
///
/// Zero-cost orchestrator: the cost belongs to the combiner region's ops, not
/// the reduce. Both surface forms feed a pairwise tree fold of the combiner
/// region (`tree_fold`); shorthand (`{ arith.addf }`) synthesizes a one-op
/// region so it takes the identical path. Relies on the combiner being
/// associative (MLIR's `linalg.reduce` legalization already guarantees this).
fn reduce(op: &Operation, ctx: &mut CoreContext, env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let tile = match ctx.get_value(&op.operands[0])? {
        Value::Tile(t) => t.clone(),
        // Already a scalar — nothing to reduce, pass it through.
        other => return Ok(Some(other.clone())),
    };

    // Resolve the combiner region (capturing bb0 arg names).
    let (mut bb0_names, mut body_ops) = resolve_region_body(op);

    // Combiner op name: explicit form has it as the region's first non-yield op;
    // shorthand stores it in `reduce_fn`. Default to arith.addf.
    let reduce_fn = match op.attributes.get("reduce_fn") {
        Some(Attr::Str(s)) => Some(s.clone()),
        _ => None,
    }
    .or_else(|| {
        body_ops
            .iter()
            .find(|o| o.op_type != "linalg.yield")
            .map(|o| o.op_type.clone())
    })
    .unwrap_or_else(|| "arith.addf".to_string());

    // Shorthand has no region — synthesize the explicit-form block:
    //   (%in, %out) { %s = <reduce_fn> %in, %out; linalg.yield %s }
    if body_ops.is_empty() {
        bb0_names = vec!["__reduce_in__".to_string(), "__reduce_acc__".to_string()];
        body_ops = vec![
            Operation::new(Some("__reduce_combined__"), &reduce_fn, &["__reduce_in__", "__reduce_acc__"]),
            Operation::new(None, "linalg.yield", &["__reduce_combined__"]),
        ];
    }

    // dim: axis to reduce; absent -> collapse all to a scalar.
    let dim = match op.attributes.get("dim") {
        Some(Attr::Int(d)) => Some(*d as usize),
        _ => None,
    };

    let (folded_data, folded_shape) = tree_fold(&tile, dim, &bb0_names, &body_ops, ctx, env)?;

    // Squeeze the reduced axis. With no dim, the fold collapsed onto axis 0 of a
    // flattened view, leaving a length-1 vector -> scalar.
    let result = match dim {
        None => Value::Scalar(Scalar::F32(folded_data[0])),
        Some(d) => {
            let mut reduced_shape = folded_shape.clone();
            reduced_shape.remove(d); // extent was 1 along d after the fold
            if reduced_shape.is_empty() {
                Value::Scalar(Scalar::F32(folded_data[0]))
            } else {
                Value::Tile(Tile::compute(folded_data, tile.dtype, reduced_shape))
            }
        }
    };

    // MLIR writes the result back into the outs buffer; downstream ops may
    // reference it by the outs SSA name. Bind both so either reference resolves.
    if let Some(Attr::Str(outs_var)) = op.attributes.get("outs_var") {
        ctx.set_value(outs_var, result.clone());
    }

    Ok(Some(result))
}

/// Reduce `tile` along `dim` by folding the combiner region pairwise.
///
/// Splits the reduced axis in half, combines the two halves with one
/// *vectorised* region call, and repeats — `ceil(log2(N))` region executions
/// rather than `N` sequential folds. Odd lengths carry the unpaired slice into
/// the next round. Returns `(data, shape)` with extent 1 along `dim`.
fn tree_fold(
    tile: &Tile,
    dim: Option<usize>,
    bb0_names: &[String],
    body_ops: &[Operation],
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    // Reduce to scalar (no dim) -> flatten everything onto one axis first.
    let (mut acc, mut shape, axis) = match dim {
        None => (tile.data.clone(), vec![tile.data.len()], 0usize),
        Some(d) => {
            if d >= tile.shape.len() {
                return Err(format!(
                    "linalg.reduce: dim {d} out of range for shape {:?}",
                    tile.shape
                ));
            }
            (tile.data.clone(), tile.shape.clone(), d)
        }
    };

    let mut n = shape[axis];
    while n > 1 {
        let half = n / 2;
        let (left_data, left_shape) = slice_along(&acc, &shape, axis, 0, half);
        let (right_data, right_shape) = slice_along(&acc, &shape, axis, half, 2 * half);

        let combined = run_combiner(
            bb0_names,
            body_ops,
            Tile::compute(left_data, tile.dtype, left_shape.clone()),
            Tile::compute(right_data, tile.dtype, right_shape),
            ctx,
            env,
        )?;
        let mut combined_data = match combined {
            Value::Tile(t) => t.data,
            Value::Scalar(s) => vec![as_f32(&Value::Scalar(s), "reduce combiner")?],
            other => {
                return Err(format!(
                    "linalg.reduce: combiner yielded {other:?}, expected tile/scalar"
                ))
            }
        };
        let mut combined_shape = left_shape;

        if n % 2 == 1 {
            // Odd: concatenate the leftover slice along the reduced axis.
            let (tail_data, _tail_shape) = slice_along(&acc, &shape, axis, 2 * half, n);
            combined_data = concat_along(&combined_data, &combined_shape, &tail_data, axis);
            combined_shape[axis] += 1;
        }

        acc = combined_data;
        shape = combined_shape;
        n = shape[axis];
    }

    Ok((acc, shape))
}

/// Run the combiner region once on two equal-shaped operands, returning the
/// yielded value. Binds the bb0 args in an isolated scope and dispatches the
/// region via `execute_region` (so each combiner op fires through the normal
/// driver and is charged latency under its own category).
fn run_combiner(
    bb0_names: &[String],
    body_ops: &[Operation],
    lhs: Tile,
    rhs: Tile,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Value, String> {
    run_region(
        ctx,
        env,
        |ctx| {
            if let Some(name) = bb0_names.first() {
                ctx.set_value(name, Value::Tile(lhs.clone()));
            }
            if let Some(name) = bb0_names.get(1) {
                ctx.set_value(name, Value::Tile(rhs.clone()));
            }
            Ok(())
        },
        body_ops,
    )?
    .ok_or_else(|| "linalg.reduce: combiner region did not yield".to_string())
}

// ===========================================================================
// generic / index / yield
// ===========================================================================

/// `%r = linalg.generic ins(...) outs(%init) { ^bb0(...): <body> }`.
///
/// Broadcasts each input to the outs iteration space per its indexing map
/// (inserting size-1 axes for missing dims), binds the bb0 block-arg names, then
/// runs the region body once over the full arrays and broadcasts the yielded
/// value back to the outs shape.
fn generic(op: &Operation, ctx: &mut CoreContext, env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let n_ins = match op.attributes.get("n_ins") {
        Some(Attr::Int(n)) => *n as usize,
        _ => 0,
    };
    let indexing_maps = indexing_maps_attr(op);

    // Snapshot input values (clone to drop the borrow on ctx before we mutate).
    let ins_vals: Vec<Value> = (0..n_ins)
        .map(|i| ctx.get_value(&op.operands[i]).cloned())
        .collect::<Result<_, _>>()?;
    let outs_val = expect_tile(ctx.get_value(&op.operands[n_ins])?, "linalg.generic outs")?.clone();
    let out_shape = outs_val.shape.clone();
    let out_ndim = out_shape.len();
    let out_dtype = outs_val.dtype;

    let (bb0_names, body_ops) = resolve_region_body(op);
    if bb0_names.is_empty() {
        return Err("linalg.generic: cannot determine bb0 argument names".into());
    }

    let result = run_region(
        ctx,
        env,
        |ctx| {
            // Store output shape so linalg.index can build index arrays.
            ctx.set_value(
                SHAPE_KEY,
                Value::Tuple(out_shape.iter().map(|&d| Value::Index(d as i64)).collect()),
            );

            // Broadcast each input to the iteration space and bind to its bb0 arg.
            for (i, val) in ins_vals.iter().enumerate() {
                let arg_val = match val {
                    Value::Tile(t) => {
                        let imap = indexing_maps.get(i).cloned().unwrap_or_default();
                        // With an explicit indexing map, insert size-1 axes for
                        // any output dim the map does not reference (Python's
                        // np.expand_dims loop). With no map, fall back to plain
                        // right-aligned NumPy broadcasting against out_shape.
                        let mut shape: Vec<usize> = t.shape.clone();
                        if !imap.is_empty() {
                            for d in 0..out_ndim {
                                if !imap.contains(&d) && d <= shape.len() {
                                    shape.insert(d, 1);
                                }
                            }
                        }
                        let data = broadcast_to(&t.data, &shape, &out_shape).ok_or_else(|| {
                            format!(
                                "linalg.generic: cannot broadcast input {i} {shape:?} to {out_shape:?}"
                            )
                        })?;
                        Value::Tile(Tile::compute(data, t.dtype, out_shape.clone()))
                    }
                    other => other.clone(),
                };
                if let Some(name) = bb0_names.get(i) {
                    ctx.set_value(name, arg_val);
                }
            }

            // Bind the outs bb0 arg — in MLIR semantics outs is the initial value
            // of the output block argument.
            if n_ins < bb0_names.len() {
                ctx.set_value(
                    &bb0_names[n_ins],
                    Value::Tile(Tile::compute(outs_val.data.clone(), outs_val.dtype, out_shape.clone())),
                );
            }
            Ok(())
        },
        &body_ops,
    )?;

    // Broadcast the yielded value back to the outs shape.
    let out_tile = match result {
        Some(Value::Tile(t)) => {
            let data = broadcast_to(&t.data, &t.shape, &out_shape).ok_or_else(|| {
                format!("linalg.generic: yield shape {:?} not broadcastable to {out_shape:?}", t.shape)
            })?;
            Tile::compute(data, out_dtype, out_shape)
        }
        Some(other) => {
            let v = as_f32(&other, "linalg.generic yield")?;
            Tile::compute(vec![v; out_shape.iter().product()], out_dtype, out_shape)
        }
        None => return Err("linalg.generic: region did not yield".into()),
    };
    Ok(Some(Value::Tile(out_tile)))
}

/// `%r = linalg.index <dim>` — a broadcasting index array for iteration `dim`.
fn index(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let dim = match op.attributes.get("dim") {
        Some(Attr::Int(d)) => *d as usize,
        _ => 0,
    };
    let out_shape = match ctx.get_value(SHAPE_KEY)? {
        Value::Tuple(items) => items
            .iter()
            .map(|v| match v {
                Value::Index(i) => Ok(*i as usize),
                other => Err(format!("linalg.index: bad shape entry {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?,
        other => return Err(format!("linalg.index: {SHAPE_KEY} is {other:?}, expected shape tuple")),
    };
    if dim >= out_shape.len() {
        return Err(format!("linalg.index: dim {dim} out of range for shape {out_shape:?}"));
    }
    // arange(out_shape[dim]) reshaped to [1,...,out_shape[dim],...,1].
    let mut shape = vec![1usize; out_shape.len()];
    shape[dim] = out_shape[dim];
    let arange: Vec<f32> = (0..out_shape[dim]).map(|i| i as f32).collect();
    Ok(Some(Value::Tile(Tile::compute(arange, DType::I32, shape))))
}

/// `linalg.yield %v` — park the yielded value under [`YIELD_KEY`] in the current
/// scope so the enclosing region driver can recover it (see [`run_region`]).
fn yield_op(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    if let Some(name) = op.operands.first() {
        let v = ctx.get_value(name)?.clone();
        ctx.set_value(YIELD_KEY, v);
    }
    Ok(None)
}

// ===========================================================================
// region helpers (yield threading)
// ===========================================================================

/// Run `body_ops` in a fresh scope after `bind` populates the block args, then
/// recover the value parked by `linalg.yield`. Owns the `push_scope` /
/// `pop_scope` pair so callers cannot leak a scope on error.
fn run_region(
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
    bind: impl FnOnce(&mut CoreContext) -> Result<(), String>,
    body_ops: &[Operation],
) -> Result<Option<Value>, String> {
    ctx.push_scope();
    let outcome = (|| {
        bind(ctx)?;
        execute_region(body_ops, ctx, env)?;
        // Recover the yielded value (if any) before the scope is torn down.
        Ok(if ctx.has_value(YIELD_KEY) {
            Some(ctx.get_value(YIELD_KEY)?.clone())
        } else {
            None
        })
    })();
    ctx.pop_scope();
    outcome
}

/// Resolve a linalg op's region into `(bb0_names, body_ops)`.
///
/// Block-argument names are found in priority order, mirroring Python:
///   1. a `bb0_names` string-list attribute (mlir_frontend / `^bb0(...)` path);
///   2. the operand names of the region's first non-yield op (inline-block form,
///      e.g. `linalg.reduce`'s `(%in, %out) { %s = addf %in, %out }`).
///
/// Returns `([], [])` when the op has no region (reduce shorthand synthesizes
/// one). A synthetic `region.bb0_args` op, if present, is dropped from the body.
fn resolve_region_body(op: &Operation) -> (Vec<String>, Vec<Operation>) {
    let region: &[Operation] = op.regions.first().map(|r| r.as_slice()).unwrap_or(&[]);
    let body_ops: Vec<Operation> = region
        .iter()
        .filter(|o| o.op_type != "region.bb0_args")
        .cloned()
        .collect();

    if let Some(Attr::StrList(names)) = op.attributes.get("bb0_names") {
        return (names.clone(), body_ops);
    }
    if let Some(first) = body_ops.first() {
        return (first.operands.clone(), body_ops);
    }
    (Vec::new(), Vec::new())
}

// ===========================================================================
// shape / ndarray helpers (the NumPy ops Python gets for free)
// ===========================================================================

/// Row-major strides for `shape` (element strides, not bytes).
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a flat row-major index into a multi-index for `shape`.
fn unravel(mut lin: usize, shape: &[usize]) -> Vec<usize> {
    let strides = row_major_strides(shape);
    let mut idx = vec![0usize; shape.len()];
    for (k, &s) in strides.iter().enumerate() {
        idx[k] = lin / s;
        lin %= s;
    }
    idx
}

/// Broadcast `data` (logical `from_shape`) to `to_shape`, NumPy rules: right-
/// aligned by rank, each axis must match or be 1. Returns the expanded flat
/// data, or `None` if incompatible.
fn broadcast_to(data: &[f32], from_shape: &[usize], to_shape: &[usize]) -> Option<Vec<f32>> {
    if from_shape.len() > to_shape.len() {
        return None;
    }
    // Right-align ranks by left-padding from_shape with leading 1s.
    let pad = to_shape.len() - from_shape.len();
    let mut src_shape = vec![1usize; pad];
    src_shape.extend_from_slice(from_shape);

    for (s, t) in src_shape.iter().zip(to_shape) {
        if *s != *t && *s != 1 {
            return None;
        }
    }

    // Fast path: already the exact shape.
    if src_shape == to_shape {
        return Some(data.to_vec());
    }

    let src_strides = row_major_strides(&src_shape);
    let total: usize = to_shape.iter().product();
    let mut out = vec![0.0f32; total];
    for (lin, slot) in out.iter_mut().enumerate() {
        let idx = unravel(lin, to_shape);
        // Map each output coord to the source coord (0 where the src axis is 1).
        let mut src = 0usize;
        for (k, &coord) in idx.iter().enumerate() {
            let c = if src_shape[k] == 1 { 0 } else { coord };
            src += c * src_strides[k];
        }
        *slot = data[src];
    }
    Some(out)
}

/// Slice `data` (logical `shape`) along `axis` for `[lo, hi)`. Returns the
/// sliced flat data and its shape.
fn slice_along(data: &[f32], shape: &[usize], axis: usize, lo: usize, hi: usize) -> (Vec<f32>, Vec<usize>) {
    let mut out_shape = shape.to_vec();
    out_shape[axis] = hi - lo;
    let strides = row_major_strides(shape);
    let total: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; total];
    for (lin, slot) in out.iter_mut().enumerate() {
        let mut idx = unravel(lin, &out_shape);
        idx[axis] += lo; // shift into the source's coordinate frame
        let src: usize = idx.iter().zip(&strides).map(|(&c, &s)| c * s).sum();
        *slot = data[src];
    }
    (out, out_shape)
}

/// Concatenate `a` and `b` along `axis`. `a_shape` is the shape of `a`; `b` is
/// assumed to share that shape except along `axis` (the leftover odd slice the
/// tree fold carries). Result extent along `axis` is `a_shape[axis] + b_extent`.
fn concat_along(a: &[f32], a_shape: &[usize], b: &[f32], axis: usize) -> Vec<f32> {
    // Recover b's extent along `axis` from its element count and a's other axes.
    let outer: usize = a_shape.iter().enumerate().filter(|(i, _)| *i != axis).map(|(_, &d)| d).product();
    let b_extent = b.len().checked_div(outer).unwrap_or(0);

    let mut out_shape = a_shape.to_vec();
    out_shape[axis] = a_shape[axis] + b_extent;
    let total: usize = out_shape.iter().product();
    let a_strides = row_major_strides(a_shape);
    let mut b_shape = a_shape.to_vec();
    b_shape[axis] = b_extent;
    let b_strides = row_major_strides(&b_shape);

    let mut out = vec![0.0f32; total];
    for (lin, slot) in out.iter_mut().enumerate() {
        let idx = unravel(lin, &out_shape);
        if idx[axis] < a_shape[axis] {
            let src: usize = idx.iter().zip(&a_strides).map(|(&c, &s)| c * s).sum();
            *slot = a[src];
        } else {
            let mut bidx = idx.clone();
            bidx[axis] -= a_shape[axis];
            let src: usize = bidx.iter().zip(&b_strides).map(|(&c, &s)| c * s).sum();
            *slot = b[src];
        }
    }
    out
}

// ===========================================================================
// value / attribute helpers
// ===========================================================================

fn expect_tile<'a>(v: &'a Value, ctx: &str) -> Result<&'a Tile, String> {
    match v {
        Value::Tile(t) => Ok(t),
        other => Err(format!("{ctx}: expected Tile, got {other:?}")),
    }
}

/// Coerce a scalar-ish value to f32. Mirrors Python's `float(scalar)`.
fn as_f32(v: &Value, ctx: &str) -> Result<f32, String> {
    match v {
        Value::Scalar(Scalar::F32(x)) => Ok(*x),
        Value::Scalar(Scalar::I32(x)) => Ok(*x as f32),
        Value::Scalar(Scalar::I64(x)) => Ok(*x as f32),
        Value::Scalar(Scalar::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Index(i) => Ok(*i as f32),
        other => Err(format!("{ctx}: expected scalar, got {other:?}")),
    }
}

fn int_list_attr<'a>(op: &'a Operation, key: &str) -> Option<&'a Vec<i64>> {
    match op.attributes.get(key) {
        Some(Attr::IntList(v)) => Some(v),
        _ => None,
    }
}

/// Read `indexing_maps`: the Python parser stores, per input, the list of output
/// dims its affine map references. The closed `Attr` enum has no nested-list
/// variant, so the integrator supplies these via `Attr::StrList` of
/// comma-separated dim lists (e.g. `"0,1"`) per input, or omits the attribute —
/// in which case handlers fall back to NumPy right-aligned broadcasting, which
/// covers the common elementwise / scalar-broadcast cases the Python tests use.
fn indexing_maps_attr(op: &Operation) -> Vec<Vec<usize>> {
    match op.attributes.get("indexing_maps") {
        Some(Attr::StrList(maps)) => maps
            .iter()
            .map(|m| {
                m.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};

    fn run(ops: &[Operation], ctx: &mut CoreContext) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv { dispatch: &dispatch, grid: &grid };
        execute_ops(ops, ctx, &env)
    }

    fn tile(data: Vec<f32>, shape: Vec<usize>) -> Value {
        Value::Tile(Tile::compute(data, DType::F32, shape))
    }

    fn get_tile(ctx: &CoreContext, name: &str) -> Tile {
        match ctx.get_value(name).unwrap() {
            Value::Tile(t) => t.clone(),
            other => panic!("expected tile, got {other:?}"),
        }
    }

    // --- fill -------------------------------------------------------------

    #[test]
    fn fill_broadcasts_scalar() {
        let mut ctx = single_core_context();
        ctx.set_value("%s", Value::Scalar(Scalar::F32(7.0)));
        ctx.set_value("%init", tile(vec![0.0; 6], vec![2, 3]));
        run(&[Operation::new(Some("%r"), "linalg.fill", &["%s", "%init"])], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.data, vec![7.0; 6]);
        assert_eq!(t.shape, vec![2, 3]);
    }

    #[test]
    fn fill_from_index_scalar() {
        let mut ctx = single_core_context();
        ctx.set_value("%s", Value::Index(3));
        ctx.set_value("%init", tile(vec![0.0; 2], vec![2]));
        run(&[Operation::new(Some("%r"), "linalg.fill", &["%s", "%init"])], &mut ctx).unwrap();
        assert_eq!(get_tile(&ctx, "%r").data, vec![3.0, 3.0]);
    }

    // --- transpose --------------------------------------------------------

    #[test]
    fn transpose_2d() {
        let mut ctx = single_core_context();
        // [[1,2,3],[4,5,6]] -> transpose [1,0] -> [[1,4],[2,5],[3,6]]
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        ctx.set_value("%y", tile(vec![0.0; 6], vec![3, 2]));
        let op = Operation::new(Some("%r"), "linalg.transpose", &["%x", "%y"])
            .with_attr("permutation", Attr::IntList(vec![1, 0]));
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_identity_permutation() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%y", tile(vec![0.0; 4], vec![2, 2]));
        let op = Operation::new(Some("%r"), "linalg.transpose", &["%x", "%y"])
            .with_attr("permutation", Attr::IntList(vec![0, 1]));
        run(&[op], &mut ctx).unwrap();
        assert_eq!(get_tile(&ctx, "%r").data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn transpose_3d_permutation() {
        let mut ctx = single_core_context();
        // shape [2,1,3], permute [1,2,0] -> shape [1,3,2]; out[a,b,c]=in[c,a,b].
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        ctx.set_value("%x", tile(data, vec![2, 1, 3]));
        ctx.set_value("%y", tile(vec![0.0; 6], vec![1, 3, 2]));
        let op = Operation::new(Some("%r"), "linalg.transpose", &["%x", "%y"])
            .with_attr("permutation", Attr::IntList(vec![1, 2, 0]));
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![1, 3, 2]);
        // in[i,j,k] at i*3+k (j=0). out flat: (0,0,0)->in[0,0,0]=0 (0,0,1)->in[1,0,0]=3
        // (0,1,0)->in[0,0,1]=1 (0,1,1)->in[1,0,1]=4 (0,2,0)->in[0,0,2]=2 (0,2,1)->in[1,0,2]=5
        assert_eq!(t.data, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    // --- broadcast --------------------------------------------------------

    #[test]
    fn broadcast_along_dim() {
        let mut ctx = single_core_context();
        // ins [3], broadcast dim 1 -> expand to [3,1] -> [3,4]: rows constant.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0], vec![3]));
        ctx.set_value("%y", tile(vec![0.0; 12], vec![3, 4]));
        let op = Operation::new(Some("%r"), "linalg.broadcast", &["%x", "%y"])
            .with_attr("dimensions", Attr::IntList(vec![1]));
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![3, 4]);
        assert_eq!(t.data, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn broadcast_leading_dim() {
        let mut ctx = single_core_context();
        // ins [4], broadcast dim 0 -> [1,4] -> [3,4]: each row identical.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        ctx.set_value("%y", tile(vec![0.0; 12], vec![3, 4]));
        let op = Operation::new(Some("%r"), "linalg.broadcast", &["%x", "%y"])
            .with_attr("dimensions", Attr::IntList(vec![0]));
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    // --- matmul -----------------------------------------------------------

    #[test]
    fn matmul_plain() {
        let mut ctx = single_core_context();
        // A=[[1,2],[3,4]], B=[[5,6],[7,8]] -> [[19,22],[43,50]]
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%b", tile(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]));
        run(&[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_accumulates_outs() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%b", tile(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]));
        ctx.set_value("%c", tile(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]));
        run(&[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b", "%c"])], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.data, vec![20.0, 23.0, 44.0, 51.0]);
    }

    #[test]
    fn matmul_nonsquare() {
        let mut ctx = single_core_context();
        // A [2x3], B [3x2] -> [2x2]
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        ctx.set_value("%b", tile(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]));
        run(&[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 2]);
        // row0: [58, 64], row1: [139, 154]
        assert_eq!(t.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_rejects_inner_dim_mismatch() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", tile(vec![1.0, 2.0], vec![1, 2]));
        ctx.set_value("%b", tile(vec![1.0, 2.0, 3.0], vec![3, 1]));
        let err = run(&[Operation::new(Some("%r"), "linalg.matmul", &["%a", "%b"])], &mut ctx).unwrap_err();
        assert!(err.contains("inner dims disagree"));
    }

    #[test]
    fn batch_matmul_two_batches() {
        let mut ctx = single_core_context();
        // batch0: [[1,2],[3,4]] @ I = same. batch1: I @ [[5,6],[7,8]] = same.
        ctx.set_value("%a", tile(vec![1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], vec![2, 2, 2]));
        ctx.set_value("%b", tile(vec![1.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]));
        run(&[Operation::new(Some("%r"), "linalg.batch_matmul", &["%a", "%b"])], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 2, 2]);
        assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    // --- reduce -----------------------------------------------------------

    fn addf_combiner_region() -> Vec<Operation> {
        // (%in, %out) { %s = arith.addf %in, %out ; linalg.yield %s }
        vec![
            Operation::new(Some("%s"), "arith.addf", &["%in", "%out"]),
            Operation::new(None, "linalg.yield", &["%s"]),
        ]
    }

    #[test]
    fn reduce_all_to_scalar_explicit_region() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        let mut op = Operation::new(Some("%r"), "linalg.reduce", &["%x"]);
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 10.0),
            other => panic!("expected scalar 10.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_all_odd_length() {
        let mut ctx = single_core_context();
        // 5 elements exercises the odd-carry path in the tree fold.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]));
        let mut op = Operation::new(Some("%r"), "linalg.reduce", &["%x"]);
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 15.0),
            other => panic!("expected scalar 15.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_along_dim_keeps_other_axis() {
        let mut ctx = single_core_context();
        // [[1,2,3],[4,5,6]] reduce dim=1 -> [6, 15]
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        let mut op = Operation::new(Some("%r"), "linalg.reduce", &["%x"]).with_attr("dim", Attr::Int(1));
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2]);
        assert_eq!(t.data, vec![6.0, 15.0]);
    }

    #[test]
    fn reduce_along_dim0() {
        let mut ctx = single_core_context();
        // [[1,2,3],[4,5,6]] reduce dim=0 -> [5,7,9]
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        let mut op = Operation::new(Some("%r"), "linalg.reduce", &["%x"]).with_attr("dim", Attr::Int(0));
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_dim1_odd_extent() {
        let mut ctx = single_core_context();
        // [[1,2,3],[4,5,6]] dim=1 odd extent 3 -> [6,15] exercises odd carry on a 2-D fold.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]));
        let mut op = Operation::new(Some("%r"), "linalg.reduce", &["%x"]).with_attr("dim", Attr::Int(1));
        op.regions.push(addf_combiner_region());
        run(&[op], &mut ctx).unwrap();
        assert_eq!(get_tile(&ctx, "%r").data, vec![6.0, 15.0]);
    }

    #[test]
    fn reduce_shorthand_synthesizes_region() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![2.0, 4.0, 6.0, 8.0], vec![4]));
        // Shorthand: reduce_fn attribute, no region.
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.addf".into()));
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 20.0),
            other => panic!("expected 20.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_mul_combiner() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![4]));
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.mulf".into()));
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 24.0),
            other => panic!("expected 24.0, got {other:?}"),
        }
    }

    #[test]
    fn reduce_binds_outs_var() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0], vec![3]));
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.addf".into()))
            .with_attr("outs_var", Attr::Str("%acc".into()));
        run(&[op], &mut ctx).unwrap();
        // Both %r and %acc resolve to the reduced scalar.
        match ctx.get_value("%acc").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 6.0),
            other => panic!("expected 6.0 via outs_var, got {other:?}"),
        }
    }

    #[test]
    fn reduce_scalar_input_passthrough() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", Value::Scalar(Scalar::F32(42.0)));
        let op = Operation::new(Some("%r"), "linalg.reduce", &["%x"])
            .with_attr("reduce_fn", Attr::Str("arith.addf".into()));
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%r").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 42.0),
            other => panic!("expected 42.0 passthrough, got {other:?}"),
        }
    }

    // --- generic ----------------------------------------------------------

    #[test]
    fn generic_elementwise_add() {
        let mut ctx = single_core_context();
        // ^bb0(%a, %b, %out): %s = addf %a, %b ; yield %s
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0], vec![3]));
        ctx.set_value("%y", tile(vec![10.0, 20.0, 30.0], vec![3]));
        ctx.set_value("%init", tile(vec![0.0; 3], vec![3]));
        let mut op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%y", "%init"])
            .with_attr("n_ins", Attr::Int(2))
            .with_attr("bb0_names", Attr::StrList(vec!["%a".into(), "%b".into(), "%out".into()]));
        op.regions.push(vec![
            Operation::new(Some("%s"), "arith.addf", &["%a", "%b"]),
            Operation::new(None, "linalg.yield", &["%s"]),
        ]);
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.data, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn generic_uses_outs_block_arg() {
        let mut ctx = single_core_context();
        // ^bb0(%a, %out): %s = addf %a, %out ; yield %s  — accumulate into outs.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0], vec![3]));
        ctx.set_value("%init", tile(vec![100.0, 200.0, 300.0], vec![3]));
        let mut op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%init"])
            .with_attr("n_ins", Attr::Int(1))
            .with_attr("bb0_names", Attr::StrList(vec!["%a".into(), "%out".into()]));
        op.regions.push(vec![
            Operation::new(Some("%s"), "arith.addf", &["%a", "%out"]),
            Operation::new(None, "linalg.yield", &["%s"]),
        ]);
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.data, vec![101.0, 202.0, 303.0]);
    }

    #[test]
    fn generic_broadcasts_input_via_indexing_map() {
        let mut ctx = single_core_context();
        // out [2,3]. input %x shape [3] maps to dim 1 only (indexing_maps "1"),
        // so it broadcasts across rows. addf with the [2,3] outs (all zero).
        ctx.set_value("%x", tile(vec![10.0, 20.0, 30.0], vec![3]));
        ctx.set_value("%init", tile(vec![0.0; 6], vec![2, 3]));
        let mut op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%init"])
            .with_attr("n_ins", Attr::Int(1))
            .with_attr("bb0_names", Attr::StrList(vec!["%a".into(), "%out".into()]))
            .with_attr("indexing_maps", Attr::StrList(vec!["1".into(), "0,1".into()]));
        op.regions.push(vec![
            Operation::new(Some("%s"), "arith.addf", &["%a", "%out"]),
            Operation::new(None, "linalg.yield", &["%s"]),
        ]);
        run(&[op], &mut ctx).unwrap();
        let t = get_tile(&ctx, "%r");
        assert_eq!(t.shape, vec![2, 3]);
        // Each row is [10,20,30].
        assert_eq!(t.data, vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn generic_yield_passthrough() {
        let mut ctx = single_core_context();
        // body just yields the input arg unchanged.
        ctx.set_value("%x", tile(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        ctx.set_value("%init", tile(vec![0.0; 4], vec![2, 2]));
        let mut op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%init"])
            .with_attr("n_ins", Attr::Int(1))
            .with_attr("bb0_names", Attr::StrList(vec!["%a".into(), "%out".into()]));
        op.regions.push(vec![Operation::new(None, "linalg.yield", &["%a"])]);
        run(&[op], &mut ctx).unwrap();
        assert_eq!(get_tile(&ctx, "%r").data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn generic_requires_bb0_names() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", tile(vec![1.0], vec![1]));
        ctx.set_value("%init", tile(vec![0.0], vec![1]));
        let op = Operation::new(Some("%r"), "linalg.generic", &["%x", "%init"])
            .with_attr("n_ins", Attr::Int(1));
        // No region, no bb0_names -> error.
        let err = run(&[op], &mut ctx).unwrap_err();
        assert!(err.contains("cannot determine bb0"));
    }

    // --- index ------------------------------------------------------------

    #[test]
    fn index_builds_arange() {
        let mut ctx = single_core_context();
        ctx.set_value(SHAPE_KEY, Value::Tuple(vec![Value::Index(2), Value::Index(3)]));
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv { dispatch: &dispatch, grid: &grid };
        let op = Operation::new(Some("%i"), "linalg.index", &[]).with_attr("dim", Attr::Int(1));
        let v = super::index(&op, &mut ctx, &env).unwrap().unwrap();
        match v {
            Value::Tile(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                assert_eq!(t.data, vec![0.0, 1.0, 2.0]);
                assert_eq!(t.dtype, DType::I32);
            }
            other => panic!("expected index tile, got {other:?}"),
        }
    }

    #[test]
    fn index_dim0() {
        let mut ctx = single_core_context();
        ctx.set_value(SHAPE_KEY, Value::Tuple(vec![Value::Index(4), Value::Index(2)]));
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv { dispatch: &dispatch, grid: &grid };
        let op = Operation::new(Some("%i"), "linalg.index", &[]).with_attr("dim", Attr::Int(0));
        let v = super::index(&op, &mut ctx, &env).unwrap().unwrap();
        match v {
            Value::Tile(t) => {
                assert_eq!(t.shape, vec![4, 1]);
                assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0]);
            }
            other => panic!("got {other:?}"),
        }
    }

    // --- helper unit tests ------------------------------------------------

    #[test]
    fn broadcast_to_rules() {
        // [1,3] -> [2,3]
        assert_eq!(
            broadcast_to(&[1.0, 2.0, 3.0], &[1, 3], &[2, 3]).unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        // [3,1] -> [3,2]
        assert_eq!(
            broadcast_to(&[1.0, 2.0, 3.0], &[3, 1], &[3, 2]).unwrap(),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
        // rank-extend: [3] -> [2,3]
        assert_eq!(
            broadcast_to(&[1.0, 2.0, 3.0], &[3], &[2, 3]).unwrap(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        // incompatible
        assert!(broadcast_to(&[1.0, 2.0], &[2], &[3]).is_none());
    }

    #[test]
    fn slice_and_concat_roundtrip() {
        let shape = vec![2, 4];
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let (left, ls) = slice_along(&data, &shape, 1, 0, 2);
        let (right, _rs) = slice_along(&data, &shape, 1, 2, 4);
        assert_eq!(ls, vec![2, 2]);
        assert_eq!(left, vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(right, vec![2.0, 3.0, 6.0, 7.0]);
        let cat = concat_along(&left, &ls, &right, 1);
        assert_eq!(cat, data);
    }

    #[test]
    fn strides_and_unravel() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(unravel(7, &[2, 4]), vec![1, 3]);
        assert_eq!(unravel(0, &[2, 3]), vec![0, 0]);
    }
}
