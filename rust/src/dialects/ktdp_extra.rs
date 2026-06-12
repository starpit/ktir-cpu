// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `ktdp` dialect handlers — grid + distributed + indirect constructors.
//!
//! Port of the remaining `ktir_cpu/dialects/ktdp_ops.py` handlers not covered by
//! `dialects/ktdp.rs` (which owns `construct_memory_view` / `construct_access_tile`),
//! together with the grid helpers from `ktir_cpu/ops/grid_ops.py`:
//!
//!   * `ktdp.get_compute_tile_id` -> grid coordinate(s) of the executing core.
//!   * `ktdp.coreid`              -> core ids matching a masked grid tuple.
//!   * `ktdp.construct_distributed_memory_view` -> `DistributedMemRef`.
//!   * `ktdp.construct_indirect_access_tile`    -> `IndirectAccessTile`.
//!
//! These build the descriptor `Value`s faithfully; the matching distributed /
//! indirect LOAD/STORE resolution lives in the memory-ops subsystem.

use std::collections::HashMap;

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Attr, Operation, Scalar, Value};
use crate::memref::{DimSubscript, DistributedMemRef, IndirectAccessTile, MemRef};

pub fn register(d: &mut Dispatch) {
    d.register("ktdp.get_compute_tile_id", LatencyCategory::Zero, get_compute_tile_id);
    d.register("ktdp.coreid", LatencyCategory::Zero, coreid);
    d.register(
        "ktdp.construct_distributed_memory_view",
        LatencyCategory::Zero,
        construct_distributed_memory_view,
    );
    d.register(
        "ktdp.construct_indirect_access_tile",
        LatencyCategory::Zero,
        construct_indirect_access_tile,
    );
}

/// `%g = ktdp.get_compute_tile_id : index`  (single-result form)
/// `%x, %y = ktdp.get_compute_tile_id : index, index`  (multi-result form)
///
/// Port of `ktdp__get_compute_tile_id`. The single-result form returns
/// `GridOps.gridid(context, 0)` — the executing core's grid x coordinate. The
/// multi-result form returns one grid coordinate per result dimension
/// (`d = 0..N`) as a tuple.
///
/// Python detects the multi-result case via `isinstance(op.result, str)`. The
/// Rust `Operation.result` is a single `Option<String>`, so the parser records
/// the result count in a `num_results` attribute; absent (or `1`) means the
/// single-result form. Mirrors `GridOps.gridid` == `context.get_grid_id(dim)`.
fn get_compute_tile_id(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let num_dims = match op.attributes.get("num_results") {
        Some(Attr::Int(n)) if *n >= 1 => *n as usize,
        Some(Attr::Int(n)) => return Err(format!("get_compute_tile_id: invalid num_results {n}")),
        _ => 1,
    };

    if num_dims == 1 {
        return Ok(Some(Value::Index(ctx.get_grid_id(0) as i64)));
    }
    let ids = (0..num_dims)
        .map(|d| Value::Index(ctx.get_grid_id(d) as i64))
        .collect();
    Ok(Some(Value::Tuple(ids)))
}

/// `%ids = ktdp.coreid %x, %y, %z`
///
/// Port of `ktdp__coreid` -> `GridOps.coreid`. The operands resolve to grid
/// coordinates (`-1` = wildcard "all cores in that dimension"); the result is
/// the list of linear core ids matching the masked tuple, in linear order.
///
/// Mirrors `GridOps.coreid`: pad the coords to 3 dims with trailing zeros, then
/// `grid_executor.get_cores_in_group((x, y, z))`. `get_cores_in_group` is not
/// surfaced on the Rust `GridExecutor`, so the wildcard match is performed here
/// over `env.grid` using its linear<->grid transforms.
fn coreid(op: &Operation, ctx: &mut CoreContext, env: &ExecutionEnv) -> Result<Option<Value>, String> {
    let mut coords: Vec<i64> = op
        .operands
        .iter()
        .map(|name| ctx.get_value(name).and_then(|v| scalar_i64(v, "coreid coord")))
        .collect::<Result<_, _>>()?;

    // Pad to 3 dims with trailing zeros, then read (x, y, z).
    while coords.len() < 3 {
        coords.push(0);
    }
    let mask = (coords[0], coords[1], coords[2]);

    let ids = cores_in_group(env, mask);
    Ok(Some(Value::Tuple(
        ids.into_iter().map(|id| Value::Index(id as i64)).collect(),
    )))
}

/// Linear core ids whose grid position matches `mask`. A `-1` in any axis is a
/// wildcard. Mirrors `GridExecutor.get_cores_in_group`.
fn cores_in_group(env: &ExecutionEnv, mask: (i64, i64, i64)) -> Vec<usize> {
    let mut out = Vec::new();
    for id in 0..env.grid.num_cores {
        let (x, y, z) = env.grid.linear_to_grid(id);
        let matches = (mask.0 == -1 || mask.0 == x as i64)
            && (mask.1 == -1 || mask.1 == y as i64)
            && (mask.2 == -1 || mask.2 == z as i64);
        if matches {
            out.push(id);
        }
    }
    out
}

/// `%R = ktdp.construct_distributed_memory_view (%a, %b, ... : types) : memref<...>`
///
/// Port of `ktdp__construct_distributed_memory_view`. Composes N per-partition
/// `MemRef`s (each carrying its own `coordinate_set` = B_i in global coords) into
/// one `DistributedMemRef`. Does NOT allocate or move data — partition routing
/// happens at access time in `distributed_tile_access`.
///
/// `DistributedMemRef::new` enforces the Python `__post_init__` invariants
/// (non-empty, every partition has a coordinate_set, matching dtypes).
fn construct_distributed_memory_view(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let partitions: Vec<MemRef> = op
        .operands
        .iter()
        .enumerate()
        .map(|(i, name)| match ctx.get_value(name)? {
            Value::MemRef(m) => Ok(m.clone()),
            other => Err(format!(
                "construct_distributed_memory_view: operand {i} is {other:?}, expected MemRef"
            )),
        })
        .collect::<Result<_, _>>()?;

    let shape = int_list(op, "shape")?
        .iter()
        .map(|&n| n as usize)
        .collect::<Vec<_>>();
    let dtype = dtype_attr(op, "dtype")?;

    let dist = DistributedMemRef::new(partitions, shape, dtype)?;
    Ok(Some(Value::DistMemRef(dist)))
}

/// `%t = ktdp.construct_indirect_access_tile intermediate_variables(...) %X[...] {...}`
///
/// Port of `ktdp__construct_indirect_access_tile`. Builds the gather/scatter
/// descriptor: a primary memory view (`%X`), N index views (one per indirect
/// dim), and one `DimSubscript` per output dimension. The indirect LOAD/STORE
/// (`indirect_load` / `indirect_store`) that consumes this lives in memory-ops.
///
/// Attribute encoding (parser-populated):
/// - `shape`: `IntList` — output access-tile shape.
/// - `variables_space_set`: `AffineSet` — domain of the intermediate vars.
/// - `variables_space_order`: `AffineMap` (optional) — iteration order; normalized to `None` when identity, matching the Python parser.
/// - `dim_kinds`: `StrList` — per-dim kind, one of `"direct"` / `"direct_expr"` / `"indirect"`.
/// - `dim_data`: `IntList` — per-dim payload parallel to `dim_kinds`: variable index for `direct`, index-view index for `indirect`, ignored for `direct_expr`.
/// - `dim_map_N`: `AffineMap` for the Nth `direct_expr` dim, left-to-right.
///
/// `op.operands[0]` is the primary memref; `op.operands[1..]` are the index
/// views, in indirect-dim order. Mirrors the Python handler's construction of
/// `IndirectAccessTile(parent_ref, shape, dim_subscripts, index_views, vss, vso)`.
fn construct_indirect_access_tile(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("construct_indirect_access_tile: missing primary memref operand".into());
    }
    let parent_ref = match ctx.get_value(&op.operands[0])? {
        Value::MemRef(m) => m.clone(),
        other => {
            return Err(format!(
                "construct_indirect_access_tile: parent is {other:?}, expected MemRef"
            ))
        }
    };

    let index_views: Vec<MemRef> = op.operands[1..]
        .iter()
        .enumerate()
        .map(|(i, name)| match ctx.get_value(name)? {
            Value::MemRef(m) => Ok(m.clone()),
            other => Err(format!(
                "construct_indirect_access_tile: index_view {i} is {other:?}, expected MemRef"
            )),
        })
        .collect::<Result<_, _>>()?;

    let shape = int_list(op, "shape")?
        .iter()
        .map(|&n| n as usize)
        .collect::<Vec<_>>();

    let variables_space_set = match op.attributes.get("variables_space_set") {
        Some(Attr::AffineSet(s)) => s.clone(),
        _ => {
            return Err(
                "construct_indirect_access_tile: missing/invalid 'variables_space_set' attribute"
                    .into(),
            )
        }
    };

    let variables_space_order = match op.attributes.get("variables_space_order") {
        // Python normalizes an identity order to None.
        Some(Attr::AffineMap(m)) if !m.is_identity() => Some(m.clone()),
        _ => None,
    };

    let dim_subscripts = parse_dim_subscripts(op, shape.len())?;

    let iat = IndirectAccessTile {
        parent_ref,
        shape,
        dim_subscripts,
        index_views,
        variables_space_set,
        variables_space_order,
        extra: HashMap::new(),
    };
    Ok(Some(Value::IndirectAccessTile(iat)))
}

/// Build the per-output-dim `DimSubscript` list from the `dim_kinds` / `dim_data`
/// / `dim_map_N` attributes. Mirrors the Python `dim_subscripts` resolution loop,
/// reduced to the structural `DimSubscript` enum the Rust IR exposes.
fn parse_dim_subscripts(op: &Operation, ndims: usize) -> Result<Vec<DimSubscript>, String> {
    let kinds = match op.attributes.get("dim_kinds") {
        Some(Attr::StrList(v)) => v.clone(),
        _ => {
            return Err(
                "construct_indirect_access_tile: missing/invalid 'dim_kinds' attribute".into(),
            )
        }
    };
    if kinds.len() != ndims {
        return Err(format!(
            "construct_indirect_access_tile: dim_kinds has {} entries but shape has {ndims} dims",
            kinds.len()
        ));
    }

    let data = match op.attributes.get("dim_data") {
        Some(Attr::IntList(v)) => v.clone(),
        // dim_data may be omitted only when no dim needs a payload.
        None => vec![0; ndims],
        Some(other) => {
            return Err(format!(
                "construct_indirect_access_tile: 'dim_data' is {other:?}, expected IntList"
            ))
        }
    };
    if data.len() != ndims {
        return Err(format!(
            "construct_indirect_access_tile: dim_data has {} entries but shape has {ndims} dims",
            data.len()
        ));
    }

    let mut subs = Vec::with_capacity(ndims);
    let mut expr_cursor = 0usize;
    for (d, kind) in kinds.iter().enumerate() {
        let sub = match kind.as_str() {
            "direct" => DimSubscript::Direct { var_index: data[d] as usize },
            "indirect" => DimSubscript::Indirect { view: data[d] as usize },
            "direct_expr" => {
                let key = format!("dim_map_{expr_cursor}");
                let map = match op.attributes.get(&key) {
                    Some(Attr::AffineMap(m)) => m.clone(),
                    _ => {
                        return Err(format!(
                            "construct_indirect_access_tile: dim {d} is direct_expr but \
                             attribute '{key}' is missing/invalid"
                        ))
                    }
                };
                expr_cursor += 1;
                DimSubscript::DirectExpr { map }
            }
            other => {
                return Err(format!(
                    "construct_indirect_access_tile: dim {d} has unknown kind {other:?} \
                     (expected direct/direct_expr/indirect)"
                ))
            }
        };
        subs.push(sub);
    }
    Ok(subs)
}

// --- attribute helpers ---------------------------------------------------

fn int_list<'a>(op: &'a Operation, key: &str) -> Result<&'a Vec<i64>, String> {
    match op.attributes.get(key) {
        Some(Attr::IntList(v)) => Ok(v),
        Some(other) => Err(format!("{}: attr '{key}' is {other:?}, expected IntList", op.op_type)),
        None => Err(format!("{}: missing required attribute '{key}'", op.op_type)),
    }
}

fn dtype_attr(op: &Operation, key: &str) -> Result<DType, String> {
    match op.attributes.get(key) {
        Some(Attr::Dtype(d)) => Ok(*d),
        Some(Attr::Str(s)) => DType::parse(s),
        _ => Err(format!("{}: missing/invalid dtype attribute '{key}'", op.op_type)),
    }
}

fn scalar_i64(v: &Value, ctx: &str) -> Result<i64, String> {
    match v {
        Value::Index(i) => Ok(*i),
        Value::Scalar(Scalar::I32(i)) => Ok(*i as i64),
        Value::Scalar(Scalar::I64(i)) => Ok(*i),
        other => Err(format!("{ctx}: expected index/int, got {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use crate::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};
    use crate::memref::{CoordinateSet, MemorySpace};

    fn run_on(
        ops: &[Operation],
        ctx: &mut CoreContext,
        grid: (usize, usize, usize),
    ) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new(grid);
        let env = ExecutionEnv::new(&dispatch, &grid);
        execute_ops(ops, ctx, &env)
    }

    /// Inclusive box `[lo, hi]` as an `AffineSet` over `lo.len()` dims:
    /// for each axis i, `d_i - lo_i >= 0` and `hi_i - d_i >= 0`.
    fn box_set(lo: &[i64], hi: &[i64]) -> AffineSet {
        let mut constraints = Vec::new();
        for i in 0..lo.len() {
            constraints.push(Constraint {
                expr: AffineExpr::Sub(
                    Rc::new(AffineExpr::Dim(i)),
                    Rc::new(AffineExpr::Const(lo[i])),
                ),
                kind: ConstraintKind::GreaterEq,
            });
            constraints.push(Constraint {
                expr: AffineExpr::Sub(
                    Rc::new(AffineExpr::Const(hi[i])),
                    Rc::new(AffineExpr::Dim(i)),
                ),
                kind: ConstraintKind::GreaterEq,
            });
        }
        AffineSet { num_dims: lo.len(), num_syms: 0, constraints }
    }

    fn hbm_part(base_stick: i64, lo: &[i64], hi: &[i64]) -> MemRef {
        MemRef {
            base_ptr: base_stick,
            shape: vec![4, 4],
            strides: vec![4, 1],
            space: MemorySpace::Hbm,
            dtype: DType::F16,
            coordinate_set: Some(box_set(lo, hi)),
        }
    }

    fn lx_view(shape: Vec<usize>) -> MemRef {
        MemRef {
            base_ptr: 0,
            shape,
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F16,
            coordinate_set: None,
        }
    }

    fn vss_2d() -> AffineSet {
        // 2-d variable space, trivially satisfiable constraint.
        AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::Dim(0),
                kind: ConstraintKind::GreaterEq,
            }],
        }
    }

    // --- get_compute_tile_id -------------------------------------------------

    #[test]
    fn compute_tile_id_single_returns_grid_x() {
        // core 5 in a (4,2,1) grid => x = 5 % 4 = 1.
        let g = GridExecutor::new((4, 2, 1));
        let (gx, gy, gz) = g.linear_to_grid(5);
        let mut ctx = single_core_context();
        ctx.grid_pos = (gx, gy, gz);

        let op = Operation::new(Some("%g"), "ktdp.get_compute_tile_id", &[]);
        run_on(&[op], &mut ctx, (4, 2, 1)).unwrap();
        match ctx.get_value("%g").unwrap() {
            Value::Index(i) => assert_eq!(*i, gx as i64),
            other => panic!("expected Index, got {other:?}"),
        }
        assert_eq!(gx, 1);
    }

    #[test]
    fn compute_tile_id_multi_returns_tuple_of_coords() {
        let mut ctx = single_core_context();
        ctx.grid_pos = (1, 2, 3);
        let op = Operation::new(Some("%g"), "ktdp.get_compute_tile_id", &[])
            .with_attr("num_results", Attr::Int(3));
        run_on(&[op], &mut ctx, (4, 4, 4)).unwrap();
        match ctx.get_value("%g").unwrap() {
            Value::Tuple(t) => {
                let got: Vec<i64> = t
                    .iter()
                    .map(|v| match v {
                        Value::Index(i) => *i,
                        o => panic!("expected Index, got {o:?}"),
                    })
                    .collect();
                assert_eq!(got, vec![1, 2, 3]);
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    // --- coreid -------------------------------------------------------------

    #[test]
    fn coreid_wildcard_x_returns_full_row() {
        // grid (4, 2, 1); mask (-1, 1, 0) => all x with y=1, z=0.
        let mut ctx = single_core_context();
        ctx.set_value("%x", Value::Index(-1));
        ctx.set_value("%y", Value::Index(1));
        ctx.set_value("%z", Value::Index(0));
        let op = Operation::new(Some("%ids"), "ktdp.coreid", &["%x", "%y", "%z"]);
        run_on(&[op], &mut ctx, (4, 2, 1)).unwrap();

        // y=1 => linear ids 4,5,6,7 (z*(nx*ny)+y*nx+x = 0 + 4 + x).
        let g = GridExecutor::new((4, 2, 1));
        let expect: Vec<i64> = (0..4).map(|x| g.grid_to_linear(x, 1, 0) as i64).collect();
        match ctx.get_value("%ids").unwrap() {
            Value::Tuple(t) => {
                let got: Vec<i64> = t
                    .iter()
                    .map(|v| match v {
                        Value::Index(i) => *i,
                        o => panic!("expected Index, got {o:?}"),
                    })
                    .collect();
                assert_eq!(got, expect);
                assert_eq!(got, vec![4, 5, 6, 7]);
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn coreid_exact_match_is_single_core() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", Value::Index(2));
        ctx.set_value("%y", Value::Index(0));
        let op = Operation::new(Some("%ids"), "ktdp.coreid", &["%x", "%y"]);
        // only 2 operands: padded to (2, 0, 0).
        run_on(&[op], &mut ctx, (4, 2, 1)).unwrap();
        match ctx.get_value("%ids").unwrap() {
            Value::Tuple(t) => {
                assert_eq!(t.len(), 1);
                // grid_to_linear(2, 0, 0) = 2.
                assert!(matches!(t[0], Value::Index(2)));
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn coreid_all_wildcards_returns_every_core() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", Value::Index(-1));
        ctx.set_value("%y", Value::Index(-1));
        ctx.set_value("%z", Value::Index(-1));
        let op = Operation::new(Some("%ids"), "ktdp.coreid", &["%x", "%y", "%z"]);
        run_on(&[op], &mut ctx, (2, 2, 1)).unwrap();
        match ctx.get_value("%ids").unwrap() {
            Value::Tuple(t) => assert_eq!(t.len(), 4),
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    // --- construct_distributed_memory_view ----------------------------------

    #[test]
    fn distributed_view_composes_partitions() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", Value::MemRef(hbm_part(0, &[0, 0], &[3, 3])));
        ctx.set_value("%b", Value::MemRef(hbm_part(16, &[4, 0], &[7, 3])));

        let op = Operation::new(
            Some("%R"),
            "ktdp.construct_distributed_memory_view",
            &["%a", "%b"],
        )
        .with_attr("shape", Attr::IntList(vec![8, 4]))
        .with_attr("dtype", Attr::Str("f16".into()));
        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();

        match ctx.get_value("%R").unwrap() {
            Value::DistMemRef(d) => {
                assert_eq!(d.partitions.len(), 2);
                assert_eq!(d.shape, vec![8, 4]);
                assert_eq!(d.dtype, DType::F16);
                // partition routing: global coord [1,1] -> partition 0.
                let (i0, _) = d.find_partition(&[1, 1], &[]).unwrap();
                assert_eq!(i0, 0);
                // global coord [5,1] -> partition 1.
                let (i1, _) = d.find_partition(&[5, 1], &[]).unwrap();
                assert_eq!(i1, 1);
            }
            other => panic!("expected DistMemRef, got {other:?}"),
        }
    }

    #[test]
    fn distributed_view_rejects_non_memref_operand() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", Value::MemRef(hbm_part(0, &[0, 0], &[3, 3])));
        ctx.set_value("%b", Value::Index(7));
        let op = Operation::new(
            Some("%R"),
            "ktdp.construct_distributed_memory_view",
            &["%a", "%b"],
        )
        .with_attr("shape", Attr::IntList(vec![8, 4]))
        .with_attr("dtype", Attr::Str("f16".into()));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("expected MemRef"));
    }

    #[test]
    fn distributed_view_requires_coordinate_set() {
        // A partition without a coordinate_set is rejected by DistributedMemRef::new.
        let mut ctx = single_core_context();
        let mut p = hbm_part(0, &[0, 0], &[3, 3]);
        p.coordinate_set = None;
        ctx.set_value("%a", Value::MemRef(p));
        let op = Operation::new(
            Some("%R"),
            "ktdp.construct_distributed_memory_view",
            &["%a"],
        )
        .with_attr("shape", Attr::IntList(vec![4, 4]))
        .with_attr("dtype", Attr::Str("f16".into()));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("coordinate_set"));
    }

    #[test]
    fn distributed_view_dtype_mismatch_is_rejected() {
        let mut ctx = single_core_context();
        ctx.set_value("%a", Value::MemRef(hbm_part(0, &[0, 0], &[3, 3])));
        let op = Operation::new(
            Some("%R"),
            "ktdp.construct_distributed_memory_view",
            &["%a"],
        )
        .with_attr("shape", Attr::IntList(vec![4, 4]))
        // partition is f16 but the view claims f32.
        .with_attr("dtype", Attr::Str("f32".into()));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("dtype"));
    }

    // --- construct_indirect_access_tile -------------------------------------

    #[test]
    fn indirect_tile_builds_descriptor() {
        // X[ind(IDX[%m,%k]), (%k)] over intermediate vars (%m, %k).
        let mut ctx = single_core_context();
        ctx.set_value("%X", Value::MemRef(lx_view(vec![16, 16])));
        ctx.set_value("%IDX", Value::MemRef(lx_view(vec![4, 4])));

        let op = Operation::new(
            Some("%t"),
            "ktdp.construct_indirect_access_tile",
            &["%X", "%IDX"],
        )
        .with_attr("shape", Attr::IntList(vec![4, 4]))
        .with_attr("variables_space_set", Attr::AffineSet(vss_2d()))
        .with_attr(
            "dim_kinds",
            Attr::StrList(vec!["indirect".into(), "direct".into()]),
        )
        // dim 0: indirect via index_view 0; dim 1: direct via var index 1.
        .with_attr("dim_data", Attr::IntList(vec![0, 1]));

        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();

        match ctx.get_value("%t").unwrap() {
            Value::IndirectAccessTile(iat) => {
                assert_eq!(iat.shape, vec![4, 4]);
                assert_eq!(iat.index_views.len(), 1);
                assert_eq!(iat.dim_subscripts.len(), 2);
                assert!(matches!(iat.dim_subscripts[0], DimSubscript::Indirect { view: 0 }));
                assert!(matches!(iat.dim_subscripts[1], DimSubscript::Direct { var_index: 1 }));
                assert!(iat.variables_space_order.is_none());
                assert_eq!(iat.parent_ref.shape, vec![16, 16]);
            }
            other => panic!("expected IndirectAccessTile, got {other:?}"),
        }
    }

    #[test]
    fn indirect_tile_direct_expr_pulls_map() {
        let mut ctx = single_core_context();
        ctx.set_value("%X", Value::MemRef(lx_view(vec![16])));

        let op = Operation::new(Some("%t"), "ktdp.construct_indirect_access_tile", &["%X"])
            .with_attr("shape", Attr::IntList(vec![4]))
            .with_attr("variables_space_set", Attr::AffineSet(vss_2d()))
            .with_attr("dim_kinds", Attr::StrList(vec!["direct_expr".into()]))
            .with_attr("dim_data", Attr::IntList(vec![0]))
            .with_attr("dim_map_0", Attr::AffineMap(AffineMap::identity(1)));

        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::IndirectAccessTile(iat) => {
                assert_eq!(iat.dim_subscripts.len(), 1);
                match &iat.dim_subscripts[0] {
                    DimSubscript::DirectExpr { map } => assert!(map.is_identity()),
                    other => panic!("expected DirectExpr, got {other:?}"),
                }
            }
            other => panic!("expected IndirectAccessTile, got {other:?}"),
        }
    }

    #[test]
    fn indirect_tile_nonidentity_order_is_kept() {
        let mut ctx = single_core_context();
        ctx.set_value("%X", Value::MemRef(lx_view(vec![16, 16])));
        ctx.set_value("%IDX", Value::MemRef(lx_view(vec![4, 4])));

        // swap order (d0,d1) -> (d1,d0) is not identity, so it must be retained.
        let swap = AffineMap {
            num_dims: 2,
            num_syms: 0,
            exprs: vec![AffineExpr::Dim(1), AffineExpr::Dim(0)],
        };
        let op = Operation::new(
            Some("%t"),
            "ktdp.construct_indirect_access_tile",
            &["%X", "%IDX"],
        )
        .with_attr("shape", Attr::IntList(vec![4, 4]))
        .with_attr("variables_space_set", Attr::AffineSet(vss_2d()))
        .with_attr("variables_space_order", Attr::AffineMap(swap.clone()))
        .with_attr("dim_kinds", Attr::StrList(vec!["indirect".into(), "direct".into()]))
        .with_attr("dim_data", Attr::IntList(vec![0, 1]));

        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::IndirectAccessTile(iat) => {
                assert_eq!(iat.variables_space_order.as_ref().unwrap(), &swap);
            }
            other => panic!("expected IndirectAccessTile, got {other:?}"),
        }
    }

    #[test]
    fn indirect_tile_identity_order_normalized_to_none() {
        let mut ctx = single_core_context();
        ctx.set_value("%X", Value::MemRef(lx_view(vec![16])));
        let op = Operation::new(Some("%t"), "ktdp.construct_indirect_access_tile", &["%X"])
            .with_attr("shape", Attr::IntList(vec![4]))
            .with_attr("variables_space_set", Attr::AffineSet(vss_2d()))
            .with_attr("variables_space_order", Attr::AffineMap(AffineMap::identity(2)))
            .with_attr("dim_kinds", Attr::StrList(vec!["direct".into()]))
            .with_attr("dim_data", Attr::IntList(vec![0]));
        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::IndirectAccessTile(iat) => assert!(iat.variables_space_order.is_none()),
            other => panic!("expected IndirectAccessTile, got {other:?}"),
        }
    }

    #[test]
    fn indirect_tile_rejects_unknown_kind() {
        let mut ctx = single_core_context();
        ctx.set_value("%X", Value::MemRef(lx_view(vec![16])));
        let op = Operation::new(Some("%t"), "ktdp.construct_indirect_access_tile", &["%X"])
            .with_attr("shape", Attr::IntList(vec![4]))
            .with_attr("variables_space_set", Attr::AffineSet(vss_2d()))
            .with_attr("dim_kinds", Attr::StrList(vec!["bogus".into()]));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("unknown kind"));
    }

    #[test]
    fn indirect_tile_dim_kinds_count_must_match_shape() {
        let mut ctx = single_core_context();
        ctx.set_value("%X", Value::MemRef(lx_view(vec![16, 16])));
        let op = Operation::new(Some("%t"), "ktdp.construct_indirect_access_tile", &["%X"])
            .with_attr("shape", Attr::IntList(vec![4, 4]))
            .with_attr("variables_space_set", Attr::AffineSet(vss_2d()))
            // one kind but shape has two dims.
            .with_attr("dim_kinds", Attr::StrList(vec!["direct".into()]))
            .with_attr("dim_data", Attr::IntList(vec![0]));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("dim_kinds"));
    }

    #[test]
    fn indirect_tile_rejects_non_memref_parent() {
        let mut ctx = single_core_context();
        ctx.set_value("%X", Value::Index(3));
        let op = Operation::new(Some("%t"), "ktdp.construct_indirect_access_tile", &["%X"])
            .with_attr("shape", Attr::IntList(vec![4]))
            .with_attr("variables_space_set", Attr::AffineSet(vss_2d()))
            .with_attr("dim_kinds", Attr::StrList(vec!["direct".into()]));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("expected MemRef"));
    }

    #[test]
    fn coordinate_set_import_surface_is_available() {
        // Smoke test that CoordinateSet is the right import surface for memref.
        let cs = CoordinateSet::Points(vec![vec![0, 0]]);
        assert!(matches!(cs, CoordinateSet::Points(_)));
    }
}
