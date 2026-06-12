// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Memory load/store data path — Rust port of the `MemoryOps.load` /
//! `MemoryOps.store` helpers from `ktir_cpu/ops/memory_ops.py` plus the
//! `ktdp.load` / `ktdp.store` handlers from `ktir_cpu/dialects/ktdp_ops.py`.
//!
//! `MemRef` / `TileRef` in this crate carry **byte-addressed** bases (see
//! `memref.rs::to_tile_ref` / `tile_access`), so the `_MemAccessor` stick/intra
//! split that Python performs is folded into the absolute-byte reads against
//! `HBMSimulator::read_bytes` / `LXScratchpad::read_bytes`.
//!
//! Tile storage is a flat `Vec<f32>` (see `tile.rs`). Bytes are decoded into
//! f32 at the load boundary per the source dtype (f16 half-precision decode,
//! f32 bit-cast, i32/i64 integer decode widened to f32) and re-encoded
//! symmetrically on store. The f16 round trip is implemented inline
//! (round-to-nearest-even) with no `half` crate dependency.
//!
//! Two paths, mirroring the Python source:
//!   * **fast path** — `coordinate_set` absent and the tile is contiguous
//!     (row-major). A single span read/write of the whole footprint.
//!   * **slow path** — a `coordinate_set` is present (or the tile is strided):
//!     enumerate the local coords via the affine set, optionally reorder them
//!     through `coordinate_order`, linearize to flat element offsets, read one
//!     contiguous span, and gather/scatter via fancy indexing.
//!
//! HBM loads/stores compute `unique_sticks` (the distinct 128-byte sticks the
//! transfer touches); LX has no stick concept and reports `None`/`0`.
//!
//! Beyond the single-allocation path this module also owns the distributed and
//! indirect data paths:
//!   * **distributed** (`distributed_tile_access` / `distributed_load` /
//!     `distributed_store`) — gather/scatter across the surviving partitions of
//!     a `DistributedMemRef`, mirroring `MemoryOps.distributed_*`.
//!   * **indirect** (`indirect_load` / `indirect_store`) — gather/scatter via
//!     index views, mirroring `MemoryOps.indirect_*`.

use crate::affine::{eval_bound, AffineMap, AffineSet, BoxSet, SymBoxSet};
use crate::context::CoreContext;
use crate::dialects::{Dispatch, LatencyCategory};
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Value};
use crate::memory::STICK_BYTES;
use crate::memref::{
    AccessTile, CoordinateSet, DimSubscript, DistributedMemRef, DistributedTileRef,
    IndirectAccessTile, MemorySpace, ParentRef, TileRef,
};
use crate::tile::Tile;

pub fn register(d: &mut Dispatch) {
    d.register("ktdp.load", LatencyCategory::Memory, load);
    d.register("ktdp.store", LatencyCategory::Memory, store);
}

// ===========================================================================
// ktdp.load / ktdp.store handlers
// ===========================================================================

/// `%t = ktdp.load %access_tile` — gather the access tile's footprint into LX.
///
/// Mirrors `ktdp__load`. Three shapes of operand are accepted:
///   * a single-allocation `AccessTile` (`ParentRef::Tile`) — the original
///     fast/slow gather path; when the access tile carries a `coordinate_set`,
///     enumerate its coords (reordered through `coordinate_order`) before the
///     slow gather, otherwise load the whole contiguous/strided tile;
///   * a distributed `AccessTile` (`ParentRef::Dist`) — gather across the
///     surviving partitions (`distributed_load`);
///   * an `IndirectAccessTile` — a gather through index views
///     (`indirect_load`).
fn load(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("ktdp.load: missing access-tile operand".into());
    }
    match ctx.get_value(&op.operands[0])?.clone() {
        Value::IndirectAccessTile(iat) => {
            let tile = indirect_load(ctx, &iat, None)?;
            Ok(Some(Value::Tile(tile)))
        }
        Value::AccessTile(access) => match &access.parent_ref {
            ParentRef::Tile(tr) => {
                let tile_ref = tr.clone();
                let coords = enumerated_coords(&access);
                let result_shape = if coords.is_some() {
                    Some(access.shape.clone())
                } else {
                    None
                };
                let tile = load_data(ctx, &tile_ref, coords.as_deref(), result_shape)?;
                Ok(Some(Value::Tile(tile)))
            }
            ParentRef::Dist(dist) => {
                let dist = dist.clone();
                let tile = distributed_load(ctx, &dist, Some(access.shape.clone()))?;
                Ok(Some(Value::Tile(tile)))
            }
        },
        other => Err(format!(
            "ktdp.load: expected an AccessTile or IndirectAccessTile, got {other:?}"
        )),
    }
}

/// `ktdp.store %tile, %access_tile` — scatter a tile back to its footprint.
///
/// Stores have no IR result; the handler computes `unique_sticks` (the latency
/// sideband Python returns) but binds nothing. Mirrors `ktdp__store`. The
/// second operand may be a single-allocation `AccessTile`, a distributed
/// `AccessTile` (`ParentRef::Dist`), or an `IndirectAccessTile`.
fn store(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    if op.operands.len() < 2 {
        return Err(format!(
            "ktdp.store expects 2 operands (tile, access_tile), got {}",
            op.operands.len()
        ));
    }
    let tile = match ctx.get_value(&op.operands[0])? {
        Value::Tile(t) => t.clone(),
        other => return Err(format!("ktdp.store expects a Tile, got {other:?}")),
    };

    // Compute the unique-stick sideband (used by the latency tracker). We do
    // not bind it as an SSA result — the op has none.
    match ctx.get_value(&op.operands[1])?.clone() {
        Value::IndirectAccessTile(iat) => {
            let _unique_sticks = indirect_store(ctx, &tile, &iat)?;
        }
        Value::AccessTile(access) => match &access.parent_ref {
            ParentRef::Tile(tr) => {
                let tile_ref = tr.clone();
                let coords = enumerated_coords(&access);
                let _unique_sticks = store_data(ctx, &tile, &tile_ref, coords.as_deref())?;
            }
            ParentRef::Dist(dist) => {
                let dist = dist.clone();
                let _unique_sticks = distributed_store(ctx, &tile, &dist)?;
            }
        },
        other => {
            return Err(format!(
                "ktdp.store: expected an AccessTile or IndirectAccessTile, got {other:?}"
            ))
        }
    }
    Ok(None)
}

/// Resolve the access tile's coordinate list, if it carries a coordinate_set.
/// Enumerate over `access.shape`, then reorder each point through
/// `coordinate_order` when present (mirrors `css.enumerate` + `cso.eval`).
fn enumerated_coords(access: &AccessTile) -> Option<Vec<Vec<i64>>> {
    let css = access.coordinate_set.as_ref()?;
    // Fast-path bypass: a coordinate set covering the full `[0, shape)` box with
    // an identity iteration order selects exactly the whole tile in row-major
    // order — identical to a plain contiguous load/store. Returning `None` takes
    // that fast path instead of enumerating every point (the O(2^n) `is_full`
    // vertex check vs O(∏ shape) enumeration + element-wise gather/scatter).
    let order_is_identity = access
        .coordinate_order
        .as_ref()
        .is_none_or(|m| m.is_identity());
    if order_is_identity && css.is_full(&access.shape) {
        return None;
    }
    let mut coords = css.enumerate(&access.shape, &[]);
    if let Some(order) = &access.coordinate_order {
        coords = coords.iter().map(|pt| order.eval(pt, &[])).collect();
    }
    Some(coords)
}

// ===========================================================================
// Distributed memory views — port of MemoryOps.distributed_* (RFC 0682 §3.3)
//
// Naming used throughout:
//   x   = global_base = base_map.eval(indices) — global origin of the access
//   A   = access_tile_set, in local coords 0..access_shape-1; None means the
//         full box [0, access_shape)
//   x+A = global footprint of the access tile
//   B_i = partition i's coordinate_set, in global coords
//   C_i = (x + A) ∩ B_i — global coords covered by both; per-survivor set
//   p_i = min(B_i) — partition i's origin in global coords
//
// distributed_load consumes C_i and p_i directly:
//   load coords (partition-local) = C_i - p_i
//   output coords (access-local)  = C_i - x
// ===========================================================================

/// Port of `MemoryOps.distributed_tile_access`. Resolve partition routing once
/// and return a [`DistributedTileRef`] whose survivors each carry a
/// per-survivor `coordinate_set` (`C_i`) and `partition_origin` (`p_i`).
///
/// Fast path: when partition `B_i` lowers to a [`BoxSet`] and `x + A` is a box,
/// compute `C_i = B_i ∩ (x + A)` in O(ndim). Slow path: enumerate `B_i` over
/// the global shape and filter by membership in `x + A`. Empty intersections
/// are skipped. Raises if no partition covers the access region.
pub fn distributed_tile_access(
    dist_ref: &DistributedMemRef,
    access_shape: &[usize],
    base_map: &AffineMap,
    indices: &[i64],
    access_tile_set: Option<&AffineSet>,
) -> Result<DistributedTileRef, String> {
    let x = base_map.eval(indices, &[]);
    let ndim = dist_ref.shape.len();
    if x.len() != ndim {
        return Err(format!(
            "distributed_tile_access: base_map produced {} coords but view has {} dims",
            x.len(),
            ndim
        ));
    }

    // Pre-compute (x + A) as an (inclusive) BoxSet when possible. None ⇒ A is
    // the implicit full box [0, access_shape). The inclusive box spans
    // [x, x + access_shape - 1] per axis.
    let xa_box: Option<BoxSet> = match access_tile_set {
        None => Some(BoxSet::new(
            x.clone(),
            (0..ndim).map(|d| x[d] + access_shape[d] as i64 - 1).collect(),
        )),
        // Lower A to an inclusive box (if axis-aligned) then translate by x.
        Some(aset) => lower_to_box(aset).map(|b| {
            BoxSet::new(
                (0..ndim).map(|d| b.lo[d] + x[d]).collect(),
                (0..ndim).map(|d| b.hi[d] + x[d]).collect(),
            )
        }),
    };

    // Slow-path membership: point ∈ x + A.
    let in_xa = |p: &[i64]| -> bool {
        match access_tile_set {
            None => (0..ndim).all(|d| {
                let local = p[d] - x[d];
                0 <= local && local < access_shape[d] as i64
            }),
            Some(aset) => {
                let local: Vec<i64> = (0..ndim).map(|d| p[d] - x[d]).collect();
                aset.contains(&local, &[])
            }
        }
    };

    let mut survivors: Vec<TileRef> = Vec::new();
    for part in &dist_ref.partitions {
        // Every distributed partition carries a coordinate_set (enforced at
        // construction). It is stored as an AffineSet (B_i in global coords).
        let b_set = part
            .coordinate_set
            .as_ref()
            .ok_or_else(|| "distributed_tile_access: partition missing coordinate_set".to_string())?;

        // Try the box fast path: B_i lowers to a box and x+A is a box.
        let b_box = lower_to_box(b_set);
        let (coordinate_set_out, p_i): (CoordinateSet, Vec<i64>) =
            match (b_box.as_ref(), xa_box.as_ref()) {
                (Some(bbox), Some(xa)) => match bbox.intersect(xa) {
                    None => continue, // empty intersection
                    Some(ci) => (CoordinateSet::Box(ci), bbox.origin().to_vec()),
                },
                _ => {
                    // Slow path: enumerate B_i and filter by membership in x+A.
                    let b_pts = b_set.enumerate(&dist_ref.shape, &[]);
                    if b_pts.is_empty() {
                        continue;
                    }
                    let p_i: Vec<i64> = (0..ndim)
                        .map(|d| b_pts.iter().map(|pt| pt[d]).min().unwrap())
                        .collect();
                    let ci_pts: Vec<Vec<i64>> =
                        b_pts.into_iter().filter(|pt| in_xa(pt)).collect();
                    if ci_pts.is_empty() {
                        continue;
                    }
                    (CoordinateSet::Points(ci_pts), p_i)
                }
            };

        survivors.push(TileRef {
            base_ptr: part.byte_address(),
            shape: part.shape.clone(),
            strides: part.strides.clone(),
            dtype: part.dtype,
            memref: Box::new(part.clone()),
            coordinate_set: Some(coordinate_set_out),
            partition_origin: Some(p_i),
        });
    }

    if survivors.is_empty() {
        return Err(format!(
            "distributed_tile_access: no partition covers access region \
             global_base={x:?} shape={access_shape:?}"
        ));
    }
    Ok(DistributedTileRef {
        partitions: survivors,
        shape: dist_ref.shape.clone(),
        dtype: dist_ref.dtype,
        global_base: Some(x),
    })
}

/// Lower an [`AffineSet`] to an **inclusive** [`BoxSet`] (`[lo, hi]`), or
/// `None` when the set is not axis-aligned / not representable as a box.
///
/// `SymBoxSet::try_from_affine_set` yields a half-open `[lo, hi)` box; for
/// distributed routing the partition / access sets are concrete, so we resolve
/// with no symbols and shrink the exclusive upper bound to inclusive (`hi - 1`).
fn lower_to_box(aset: &AffineSet) -> Option<BoxSet> {
    let sym = SymBoxSet::try_from_affine_set(aset)?;
    if !sym.is_concrete() {
        return None;
    }
    let lo: Vec<i64> = sym.lo.iter().map(|b| eval_bound(b, &[])).collect();
    let hi: Vec<i64> = sym.hi.iter().map(|b| eval_bound(b, &[]) - 1).collect();
    Some(BoxSet::new(lo, hi))
}

/// Port of `MemoryOps._subtile_ref`. Build a `TileRef` covering exactly the
/// global-coordinate `box` within `survivor`. Inherits the survivor's strides
/// verbatim; `shape` shrinks to the box extent and `base_ptr` shifts to the
/// box's partition-local origin (`box.lo - p_i`, scaled by bpe).
fn subtile_ref(survivor: &TileRef, b: &BoxSet) -> TileRef {
    let ndim = survivor.shape.len();
    let zero = vec![0i64; ndim];
    let p_i = survivor.partition_origin.as_deref().unwrap_or(&zero);
    let local_lo: Vec<i64> = (0..ndim).map(|d| b.lo[d] - p_i[d]).collect();
    // Inclusive box -> extent is hi - lo + 1.
    let sub_shape: Vec<usize> = (0..ndim).map(|d| (b.hi[d] - b.lo[d] + 1) as usize).collect();
    let bpe = survivor.dtype.bytes_per_elem() as i64;
    let byte_offset: i64 =
        (0..ndim).map(|d| local_lo[d] * survivor.strides[d]).sum::<i64>() * bpe;
    TileRef {
        base_ptr: survivor.base_ptr + byte_offset,
        shape: sub_shape,
        strides: survivor.strides.clone(),
        dtype: survivor.dtype,
        memref: survivor.memref.clone(),
        coordinate_set: None,
        partition_origin: None,
    }
}

/// Port of `MemoryOps.distributed_load`. Gather across surviving partitions
/// into a single LX-resident [`Tile`].
///
/// Fast path (BoxSet `C_i`): build a sub-`TileRef` of the partition covering
/// exactly `C_i`, delegate the read to [`load_data`], and slot its data into a
/// rectangular slice of the output buffer. Slow path (`Points` `C_i`):
/// per-coord scatter — translate `C_i` to partition-local coords, read one
/// span, and scatter each element into the access-local position.
pub fn distributed_load(
    ctx: &mut CoreContext,
    dist_tile_ref: &DistributedTileRef,
    result_shape: Option<Vec<usize>>,
) -> Result<Tile, String> {
    let ndim = dist_tile_ref.shape.len();
    let zero_x = vec![0i64; ndim];
    let x = dist_tile_ref.global_base.as_deref().unwrap_or(&zero_x);
    let out_shape = result_shape.unwrap_or_else(|| dist_tile_ref.shape.clone());
    let out_len: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_len];
    let out_strides = row_major_strides(&out_shape);

    let mut total_unique_sticks = 0usize;
    let mut any_hbm = false;

    for survivor in &dist_tile_ref.partitions {
        let cs = survivor
            .coordinate_set
            .as_ref()
            .ok_or_else(|| "distributed_load: survivor missing coordinate_set".to_string())?;
        match cs {
            CoordinateSet::Box(b) => {
                // Fast path: rectangular sub-tile, then copy into out[C_i - x].
                let sub = subtile_ref(survivor, b);
                let tile = load_data(ctx, &sub, None, None)?;
                // access-local rectangle = C_i - x; copy row-major from tile.
                let access_lo: Vec<i64> = (0..ndim).map(|d| b.lo[d] - x[d]).collect();
                let sub_shape = &sub.shape;
                copy_rect_into(&mut out, &out_strides, &access_lo, sub_shape, &tile.data);
                if let Some(s) = tile.unique_sticks {
                    total_unique_sticks += s;
                    any_hbm = true;
                }
            }
            CoordinateSet::Points(ci) => {
                let zero_p = vec![0i64; ndim];
                let p_i = survivor.partition_origin.as_deref().unwrap_or(&zero_p);
                let local_coords: Vec<Vec<i64>> = ci
                    .iter()
                    .map(|c| (0..ndim).map(|d| c[d] - p_i[d]).collect())
                    .collect();
                let access_coords: Vec<Vec<i64>> = ci
                    .iter()
                    .map(|c| (0..ndim).map(|d| c[d] - x[d]).collect())
                    .collect();
                let space = survivor.memref.space;
                let stick_bytes = stick_bytes_for(space);
                let (offsets, unique_sticks) = flat_memory_offsets(
                    survivor.base_ptr,
                    &survivor.shape,
                    &survivor.strides,
                    survivor.dtype,
                    Some(&local_coords),
                    stick_bytes,
                );
                let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
                let raw = read_raw(ctx, space, survivor.base_ptr, span * survivor.dtype.bytes_per_elem());
                let flat = decode(&raw, survivor.dtype, span);
                for (ac, &off) in access_coords.iter().zip(&offsets) {
                    let lin = lin_index(ac, &out_strides);
                    out[lin] = flat[off as usize];
                }
                if let Some(s) = unique_sticks {
                    total_unique_sticks += s;
                    any_hbm = true;
                }
            }
            CoordinateSet::Affine(_) => {
                return Err(
                    "distributed_load: survivor carries an un-lowered AffineSet \
                     coordinate_set (distributed_tile_access emits Box/Points only)"
                        .into(),
                )
            }
        }
    }

    write_to_lx(ctx, &out, dist_tile_ref.dtype);
    Ok(Tile {
        data: out,
        dtype: dist_tile_ref.dtype,
        shape: out_shape,
        unique_sticks: if any_hbm { Some(total_unique_sticks) } else { None },
        index_unique_sticks: None,
    })
}

/// Port of `MemoryOps.distributed_store`. Scatter a tile to surviving
/// partitions, symmetric to [`distributed_load`]. Returns the aggregate
/// `unique_sticks` (HBM stick cost; `0` for all-LX).
pub fn distributed_store(
    ctx: &mut CoreContext,
    tile: &Tile,
    dist_tile_ref: &DistributedTileRef,
) -> Result<usize, String> {
    let ndim = dist_tile_ref.shape.len();
    let zero_x = vec![0i64; ndim];
    let x = dist_tile_ref.global_base.as_deref().unwrap_or(&zero_x);
    let src_strides = row_major_strides(&tile.shape);

    let mut total_unique_sticks = 0usize;
    for survivor in &dist_tile_ref.partitions {
        let cs = survivor
            .coordinate_set
            .as_ref()
            .ok_or_else(|| "distributed_store: survivor missing coordinate_set".to_string())?;
        match cs {
            CoordinateSet::Box(b) => {
                let sub = subtile_ref(survivor, b);
                // Slice the source tile rectangularly at C_i - x (row-major copy).
                let access_lo: Vec<i64> = (0..ndim).map(|d| b.lo[d] - x[d]).collect();
                let src = gather_rect(&tile.data, &src_strides, &access_lo, &sub.shape);
                let sub_tile = Tile::compute(src, survivor.dtype, sub.shape.clone());
                total_unique_sticks += store_data(ctx, &sub_tile, &sub, None)?;
            }
            CoordinateSet::Points(ci) => {
                let zero_p = vec![0i64; ndim];
                let p_i = survivor.partition_origin.as_deref().unwrap_or(&zero_p);
                let local_coords: Vec<Vec<i64>> = ci
                    .iter()
                    .map(|c| (0..ndim).map(|d| c[d] - p_i[d]).collect())
                    .collect();
                let access_coords: Vec<Vec<i64>> = ci
                    .iter()
                    .map(|c| (0..ndim).map(|d| c[d] - x[d]).collect())
                    .collect();
                let space = survivor.memref.space;
                let stick_bytes = stick_bytes_for(space);
                let (offsets, unique_sticks) = flat_memory_offsets(
                    survivor.base_ptr,
                    &survivor.shape,
                    &survivor.strides,
                    survivor.dtype,
                    Some(&local_coords),
                    stick_bytes,
                );
                let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
                let raw = read_raw(ctx, space, survivor.base_ptr, span * survivor.dtype.bytes_per_elem());
                let mut flat = decode(&raw, survivor.dtype, span);
                for (ac, &off) in access_coords.iter().zip(&offsets) {
                    let lin = lin_index(ac, &src_strides);
                    flat[off as usize] = tile.data[lin];
                }
                let new_raw = encode(&flat, survivor.dtype);
                write_raw(ctx, space, survivor.base_ptr, &new_raw);
                if let Some(s) = unique_sticks {
                    total_unique_sticks += s;
                }
            }
            CoordinateSet::Affine(_) => {
                return Err(
                    "distributed_store: survivor carries an un-lowered AffineSet \
                     coordinate_set (distributed_tile_access emits Box/Points only)"
                        .into(),
                )
            }
        }
    }
    Ok(total_unique_sticks)
}

// ===========================================================================
// Indirect access tiles — port of MemoryOps.indirect_load / indirect_store
// ===========================================================================

/// Port of `MemoryOps.indirect_load`. Enumerate the variable space (in
/// `variables_space_order` order), resolve each coordinate tuple (direct dims
/// from the variable point, indirect dims via index-view lookups), and delegate
/// the gather to [`load_data`]. Stamps `index_unique_sticks` on the result.
pub fn indirect_load(
    ctx: &mut CoreContext,
    iat: &IndirectAccessTile,
    result_shape: Option<Vec<usize>>,
) -> Result<Tile, String> {
    if let Some(vso) = &iat.variables_space_order
        && !vso.is_permutation() {
            return Err(format!(
                "indirect_load: variables_space_order must permute its input \
                 dimensions; got non-permutation map: {vso:?}"
            ));
        }

    let (idx_values, idx_unique_sticks) = resolve_idx_reads(ctx, iat)?;
    let coords = build_indirect_coords(iat, &idx_values)?;

    let out_shape = result_shape.unwrap_or_else(|| iat.shape.clone());
    let tile_ref = iat.parent_ref.to_tile_ref();
    let mut tile = load_data(ctx, &tile_ref, Some(&coords), Some(out_shape))?;
    tile.index_unique_sticks = Some(idx_unique_sticks);
    Ok(tile)
}

/// Port of `MemoryOps.indirect_store`. Mirror of [`indirect_load`]: enumerate,
/// resolve, build coords, then delegate the scatter to [`store_data`]. Returns
/// the aggregate stick cost (`data_sticks + idx_unique_sticks`).
pub fn indirect_store(
    ctx: &mut CoreContext,
    tile: &Tile,
    iat: &IndirectAccessTile,
) -> Result<usize, String> {
    if tile.shape != iat.shape {
        return Err(format!(
            "indirect_store: source tile shape {:?} does not match IAT shape {:?}",
            tile.shape, iat.shape
        ));
    }
    if let Some(vso) = &iat.variables_space_order
        && !vso.is_permutation() {
            return Err(format!(
                "indirect_store: variables_space_order must permute its input \
                 dimensions; got non-permutation map: {vso:?}"
            ));
        }

    let (idx_values, idx_unique_sticks) = resolve_idx_reads(ctx, iat)?;
    let coords = build_indirect_coords(iat, &idx_values)?;
    let tile_ref = iat.parent_ref.to_tile_ref();
    let data_sticks = store_data(ctx, tile, &tile_ref, Some(&coords))?;
    Ok(data_sticks + idx_unique_sticks)
}

/// Port of `_enumerate_in_vso_order`. Enumerate variable-space points; if a
/// non-identity `variables_space_order` is set, sort the points by the map's
/// image (lexicographic on the result vector) so idx reads and coord build stay
/// in lockstep (RFC 0682 §473). Callers must already have rejected
/// non-permutation maps.
fn enumerate_in_vso_order(iat: &IndirectAccessTile) -> Vec<Vec<i64>> {
    let mut points = iat.variables_space_set.enumerate(&iat.shape, &[]);
    if let Some(vso) = &iat.variables_space_order
        && !vso.is_identity() {
            points.sort_by_key(|a| vso.eval(a, &[]));
        }
    points
}

/// Port of `_resolve_idx_reads`. For every indirect dimension, read the index
/// value its index view holds at each enumerated point, returning a map from
/// `view -> values` (one entry per enumerated point, in pt order) plus the
/// total distinct HBM sticks touched by those reads.
///
/// Rust IR model note: `DimSubscript::Indirect { view }` carries no per-dim
/// subscript expressions, so the index view is addressed by the enumeration
/// point itself, projected to the view's element layout via the view's strides
/// (`offset = Σ pt[d] * stride[d]`, over the view's rank). This matches the
/// Python `_resolve_idx_reads` for the common identity-subscript case the parser
/// produces (e.g. `IDX[%m, %k]`).
fn resolve_idx_reads(
    ctx: &CoreContext,
    iat: &IndirectAccessTile,
) -> Result<(std::collections::HashMap<usize, Vec<i64>>, usize), String> {
    let points = enumerate_in_vso_order(iat);

    // The distinct index views used by indirect dims, in first-seen order.
    let mut view_idxs: Vec<usize> = Vec::new();
    for sub in &iat.dim_subscripts {
        if let DimSubscript::Indirect { view } = sub
            && !view_idxs.contains(view) {
                view_idxs.push(*view);
            }
    }

    let mut per_view_values: std::collections::HashMap<usize, Vec<i64>> =
        std::collections::HashMap::new();
    let mut total_sticks = 0usize;

    for &iv_idx in &view_idxs {
        let iv = iat
            .index_views
            .get(iv_idx)
            .ok_or_else(|| format!("indirect: index_view {iv_idx} out of range"))?;
        let rank = iv.strides.len();
        let bpe = iv.dtype.bytes_per_elem();
        let base = iv.byte_address();
        let space = iv.space;
        let stick_bytes = stick_bytes_for(space);
        let mut sticks: std::collections::HashSet<i64> = std::collections::HashSet::new();

        // For every enumerated point (and every indirect dim that uses this
        // view), read one index value. Indirect dims sharing a view append in
        // pt-major, dim-minor order — matching build_indirect_coords.
        let mut values: Vec<i64> = Vec::new();
        for pt in &points {
            for sub in &iat.dim_subscripts {
                if let DimSubscript::Indirect { view } = sub {
                    if *view != iv_idx {
                        continue;
                    }
                    // Project the point onto the view's rank (leading dims).
                    let offset: i64 = (0..rank)
                        .map(|d| pt.get(d).copied().unwrap_or(0) * iv.strides[d])
                        .sum();
                    let byte_addr = base + offset * bpe as i64;
                    if let Some(sb) = stick_bytes {
                        sticks.insert(byte_addr / sb);
                    }
                    let raw = read_raw(ctx, space, byte_addr, bpe);
                    let v = decode(&raw, iv.dtype, 1)[0];
                    values.push(v as i64);
                }
            }
        }
        per_view_values.insert(iv_idx, values);
        if stick_bytes.is_some() {
            total_sticks += sticks.len();
        }
    }

    Ok((per_view_values, total_sticks))
}

/// Port of `_build_indirect_coords`. For each enumerated point, build the
/// parent-tensor coordinate tuple: `Direct` dims read the variable point,
/// `DirectExpr` dims evaluate their affine map over the point, and `Indirect`
/// dims consume the next pre-resolved index value (pt-major, dim-minor order).
/// Rejects negative indirect indices (NumPy would silently wrap).
fn build_indirect_coords(
    iat: &IndirectAccessTile,
    idx_values: &std::collections::HashMap<usize, Vec<i64>>,
) -> Result<Vec<Vec<i64>>, String> {
    let points = enumerate_in_vso_order(iat);
    // Per-view consumption cursors (positional, in lockstep with resolve_idx_reads).
    let mut cursors: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    let mut coords: Vec<Vec<i64>> = Vec::with_capacity(points.len());
    for pt in &points {
        let mut coord: Vec<i64> = Vec::with_capacity(iat.dim_subscripts.len());
        for sub in &iat.dim_subscripts {
            match sub {
                DimSubscript::Direct { var_index } => {
                    let v = *pt.get(*var_index).ok_or_else(|| {
                        format!("indirect: direct var_index {var_index} out of range")
                    })?;
                    coord.push(v);
                }
                DimSubscript::DirectExpr { map } => {
                    let r = map.eval(pt, &[]);
                    coord.push(r[0]);
                }
                DimSubscript::Indirect { view } => {
                    let cur = cursors.entry(*view).or_insert(0);
                    let vals = idx_values
                        .get(view)
                        .ok_or_else(|| format!("indirect: no resolved values for view {view}"))?;
                    let raw_idx = *vals.get(*cur).ok_or_else(|| {
                        format!("indirect: ran out of resolved values for view {view}")
                    })?;
                    *cur += 1;
                    if raw_idx < 0 {
                        return Err(format!(
                            "indirect index {raw_idx} from index_view {view} is negative"
                        ));
                    }
                    coord.push(raw_idx);
                }
            }
        }
        coords.push(coord);
    }
    Ok(coords)
}

// ===========================================================================
// Row-major helpers for distributed rectangular slice copies
// ===========================================================================

/// Row-major (C-order) element strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1] as i64;
    }
    strides
}

/// Linear flat index of `coord` under `strides`.
fn lin_index(coord: &[i64], strides: &[i64]) -> usize {
    coord.iter().zip(strides).map(|(&c, &s)| c * s).sum::<i64>() as usize
}

/// Copy the row-major `src` (extent `sub_shape`) into `out` at the rectangle
/// whose origin is `lo` (access-local coords), under `out_strides`.
fn copy_rect_into(
    out: &mut [f32],
    out_strides: &[i64],
    lo: &[i64],
    sub_shape: &[usize],
    src: &[f32],
) {
    let mut i = 0usize;
    rect_iter(sub_shape, &mut |rel| {
        let abs: Vec<i64> = (0..rel.len()).map(|d| lo[d] + rel[d]).collect();
        out[lin_index(&abs, out_strides)] = src[i];
        i += 1;
    });
}

/// Gather the rectangle of `src` (origin `lo`, extent `sub_shape`, strides
/// `src_strides`) into a fresh row-major buffer.
fn gather_rect(src: &[f32], src_strides: &[i64], lo: &[i64], sub_shape: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(sub_shape.iter().product());
    rect_iter(sub_shape, &mut |rel| {
        let abs: Vec<i64> = (0..rel.len()).map(|d| lo[d] + rel[d]).collect();
        out.push(src[lin_index(&abs, src_strides)]);
    });
    out
}

/// Iterate the cartesian rectangle `[0, shape)` in row-major order, calling `f`
/// with each relative coordinate.
fn rect_iter(shape: &[usize], f: &mut impl FnMut(&[i64])) {
    if shape.is_empty() {
        f(&[]);
        return;
    }
    if shape.contains(&0) {
        return;
    }
    let mut idx = vec![0i64; shape.len()];
    loop {
        f(&idx);
        let mut d = shape.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            if (idx[d] as usize) < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

// ===========================================================================
// Core data path — port of MemoryOps.load / MemoryOps.store
// ===========================================================================

/// Port of `MemoryOps.load`. Reads the tile footprint from HBM/LX, decodes per
/// dtype into an f32 `Tile`, and writes the decoded tile into the executing
/// core's LX scratchpad. All loaded tiles land in LX regardless of source.
pub fn load_data(
    ctx: &mut CoreContext,
    tile_ref: &TileRef,
    coords: Option<&[Vec<i64>]>,
    result_shape: Option<Vec<usize>>,
) -> Result<Tile, String> {
    let dtype = tile_ref.dtype;
    let bpe = dtype.bytes_per_elem();
    let space = tile_ref.memref.space;
    let stick_bytes = stick_bytes_for(space);

    // Fast path: contiguous tile, no coord filtering.
    if coords.is_none() && is_contiguous(&tile_ref.shape, &tile_ref.strides) {
        let n: usize = tile_ref.shape.iter().product();
        let raw = read_raw(ctx, space, tile_ref.base_ptr, n * bpe);
        let data = decode(&raw, dtype, n);
        write_to_lx(ctx, &data, dtype);
        let unique_sticks = stick_bytes.map(|sb| {
            let end = tile_ref.base_ptr + (n * bpe) as i64;
            ((end + sb - 1) / sb - tile_ref.base_ptr / sb) as usize
        });
        return Ok(Tile {
            data,
            dtype,
            shape: tile_ref.shape.clone(),
            unique_sticks,
            index_unique_sticks: None,
        });
    }

    // Slow path: linearize coords/shape -> flat element offsets, single span
    // read, fancy-index gather.
    let (offsets, unique_sticks) = flat_memory_offsets(
        tile_ref.base_ptr,
        &tile_ref.shape,
        &tile_ref.strides,
        dtype,
        coords,
        stick_bytes,
    );
    let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
    let raw = read_raw(ctx, space, tile_ref.base_ptr, span * bpe);
    let flat = decode(&raw, dtype, span);
    let gathered: Vec<f32> = offsets.iter().map(|&o| flat[o as usize]).collect();

    let out_shape = result_shape.unwrap_or_else(|| tile_ref.shape.clone());
    write_to_lx(ctx, &gathered, dtype);
    Ok(Tile {
        data: gathered,
        dtype,
        shape: out_shape,
        unique_sticks,
        index_unique_sticks: None,
    })
}

/// Port of `MemoryOps.store`. Encodes the tile's f32 data per dtype and writes
/// it to HBM/LX. Returns `unique_sticks` (distinct HBM sticks touched; `0` for
/// LX) — the latency sideband.
pub fn store_data(
    ctx: &mut CoreContext,
    tile: &Tile,
    tile_ref: &TileRef,
    coords: Option<&[Vec<i64>]>,
) -> Result<usize, String> {
    let dtype = tile_ref.dtype;
    let bpe = dtype.bytes_per_elem();
    let space = tile_ref.memref.space;
    let stick_bytes = stick_bytes_for(space);

    // Fast path: contiguous tile, no coord filtering.
    if coords.is_none() && is_contiguous(&tile_ref.shape, &tile_ref.strides) {
        let raw = encode(&tile.data, dtype);
        write_raw(ctx, space, tile_ref.base_ptr, &raw);
        return Ok(match stick_bytes {
            None => 0,
            Some(sb) => {
                let n: usize = tile_ref.shape.iter().product();
                let end = tile_ref.base_ptr + (n * bpe) as i64;
                ((end + sb - 1) / sb - tile_ref.base_ptr / sb) as usize
            }
        });
    }

    // Slow path: read-modify-write via scatter offsets.
    let (offsets, unique_sticks) = flat_memory_offsets(
        tile_ref.base_ptr,
        &tile_ref.shape,
        &tile_ref.strides,
        dtype,
        coords,
        stick_bytes,
    );
    let span = offsets.iter().copied().max().map(|m| m + 1).unwrap_or(1) as usize;
    let raw = read_raw(ctx, space, tile_ref.base_ptr, span * bpe);
    let mut flat = decode(&raw, dtype, span);
    // Scatter the C-order tile data into the span; last-writer-wins on coord
    // collisions (matches NumPy assignment).
    for (i, &o) in offsets.iter().enumerate() {
        flat[o as usize] = tile.data[i];
    }
    let new_raw = encode(&flat, dtype);
    write_raw(ctx, space, tile_ref.base_ptr, &new_raw);
    Ok(unique_sticks.unwrap_or(0))
}

// ===========================================================================
// Memory-space dispatch (folds the _MemAccessor stick/intra split)
// ===========================================================================

fn stick_bytes_for(space: MemorySpace) -> Option<i64> {
    match space {
        MemorySpace::Hbm => Some(STICK_BYTES),
        MemorySpace::Lx { .. } => None,
    }
}

/// Read `len` raw bytes at absolute `byte_addr` from the right backing store.
/// HBM is shared; LX routes via `lx_core_id` (None => executing core's own LX).
fn read_raw(ctx: &CoreContext, space: MemorySpace, byte_addr: i64, len: usize) -> Vec<u8> {
    match space {
        MemorySpace::Hbm => ctx.hbm.borrow().read_bytes(byte_addr, len),
        MemorySpace::Lx { core_id } => {
            let lx = ctx.get_lx(core_id.map(|c| c as usize));
            
            lx.borrow().read_bytes(byte_addr, len)
        }
    }
}

// NB: the `let bytes = ...; bytes` form keeps the LX `RefCell` borrow scoped to
// the read, dropping it before the value is returned.

/// Write raw bytes at absolute `byte_addr` to the right backing store.
fn write_raw(ctx: &mut CoreContext, space: MemorySpace, byte_addr: i64, data: &[u8]) {
    match space {
        MemorySpace::Hbm => ctx.hbm.borrow_mut().write_bytes(byte_addr, data),
        MemorySpace::Lx { core_id } => {
            let lx = ctx.get_lx(core_id.map(|c| c as usize));
            lx.borrow_mut().write_bytes(byte_addr, data);
        }
    }
}

/// Port of `MemoryOps._write_to_lx`: reserve a stick-aligned span in the
/// executing core's LX and write the decoded tile there. All loaded tiles land
/// in LX regardless of source memory space. The bytes written are the dtype's
/// native encoding so a subsequent LX-sourced load round-trips exactly.
fn write_to_lx(ctx: &mut CoreContext, data: &[f32], dtype: DType) {
    let raw = encode(data, dtype);
    let size = raw.len() as i64;
    let lx = ctx.get_lx(None);
    let lx_ptr = {
        let mut lxm = lx.borrow_mut();
        let ptr = lxm.next_ptr;
        let advanced = ptr + size;
        lxm.next_ptr = (advanced + STICK_BYTES - 1) & !(STICK_BYTES - 1);
        ptr
    };
    lx.borrow_mut().write_bytes(lx_ptr, &raw);
}

// ===========================================================================
// Offset linearization + contiguity (port of _flat_memory_offsets / _is_contiguous)
// ===========================================================================

/// Port of `MemoryOps._is_contiguous`: row-major C-order check.
pub fn is_contiguous(shape: &[usize], strides: &[i64]) -> bool {
    let mut expected: i64 = 1;
    for (&dim, &stride) in shape.iter().rev().zip(strides.iter().rev()) {
        if stride != expected {
            return false;
        }
        expected *= dim as i64;
    }
    true
}

/// Port of `MemoryOps._flat_memory_offsets`. Linearizes N-d coords (or the full
/// shape when `coords` is None) into flat element offsets, and counts distinct
/// HBM sticks when `stick_bytes` is set.
fn flat_memory_offsets(
    base_ptr: i64,
    shape: &[usize],
    strides: &[i64],
    dtype: DType,
    coords: Option<&[Vec<i64>]>,
    stick_bytes: Option<i64>,
) -> (Vec<i64>, Option<usize>) {
    let bpe = dtype.bytes_per_elem() as i64;
    let mut offsets = Vec::new();
    let mut sticks: Option<std::collections::HashSet<i64>> =
        stick_bytes.map(|_| std::collections::HashSet::new());

    let mut emit = |coord: &[i64]| {
        let o: i64 = coord.iter().zip(strides).map(|(&c, &s)| c * s).sum();
        offsets.push(o);
        if let (Some(set), Some(sb)) = (sticks.as_mut(), stick_bytes) {
            set.insert((base_ptr + o * bpe) / sb);
        }
    };

    match coords {
        Some(cs) => {
            for c in cs {
                emit(c);
            }
        }
        None => {
            // np.ndindex(*shape): row-major, rightmost dim innermost.
            ndindex(shape, &mut emit);
        }
    }

    (offsets, sticks.map(|s| s.len()))
}

/// Iterate the cartesian index space of `shape` in row-major order.
fn ndindex(shape: &[usize], f: &mut impl FnMut(&[i64])) {
    if shape.is_empty() {
        f(&[]);
        return;
    }
    if shape.contains(&0) {
        return;
    }
    let mut idx = vec![0i64; shape.len()];
    loop {
        f(&idx);
        let mut d = shape.len();
        loop {
            if d == 0 {
                return;
            }
            d -= 1;
            idx[d] += 1;
            if (idx[d] as usize) < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

// ===========================================================================
// dtype <-> bytes (decode to f32, encode from f32)
// ===========================================================================

/// Decode `n` elements of `dtype` from the front of `raw` into f32 values.
/// Missing/short bytes decode as zero (matches the simulator's zero-padding).
fn decode(raw: &[u8], dtype: DType, n: usize) -> Vec<f32> {
    let bpe = dtype.bytes_per_elem();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * bpe;
        let chunk = raw.get(off..off + bpe);
        let v = match (dtype, chunk) {
            (DType::F16, Some(b)) => widen_f16(u16::from_le_bytes([b[0], b[1]])),
            (DType::F32, Some(b)) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (DType::I32, Some(b)) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            (DType::I64, Some(b)) => {
                i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
            }
            (DType::Bool, Some(b)) => {
                if b[0] != 0 {
                    1.0
                } else {
                    0.0
                }
            }
            (_, _) => 0.0, // short/missing bytes -> zero
        };
        out.push(v);
    }
    out
}

/// Encode f32 element values into `dtype`'s native little-endian byte layout.
fn encode(data: &[f32], dtype: DType) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * dtype.bytes_per_elem());
    for &x in data {
        match dtype {
            DType::F16 => out.extend_from_slice(&narrow_f16(x).to_le_bytes()),
            DType::F32 => out.extend_from_slice(&x.to_le_bytes()),
            DType::I32 => out.extend_from_slice(&(x.round() as i32).to_le_bytes()),
            DType::I64 => out.extend_from_slice(&(x.round() as i64).to_le_bytes()),
            DType::Bool => out.push(if x != 0.0 { 1 } else { 0 }),
        }
    }
    out
}

// ===========================================================================
// f16 <-> f32 (round-to-nearest-even; mirrors arith.rs narrow_f16/widen_f16)
// ===========================================================================

// These were standalone bit-manipulation duplicates of the codec's f16 round
// trip (the consolidation the codec module flagged as a follow-up). They now
// delegate to `codec`, so `decode` gets codec's 64K f16→f32 lookup table — the
// hot path of every `ktdp.load` — and there is a single f16 implementation.
fn narrow_f16(x: f32) -> u16 {
    crate::codec::f32_to_f16_bits(x)
}
fn widen_f16(h: u16) -> f32 {
    crate::codec::f16_bits_to_f32(h)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use crate::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};
    use crate::ir::Attr;
    use crate::memref::{MemRef, MemorySpace};

    fn run(ops: &[Operation], ctx: &mut CoreContext) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        execute_ops(ops, ctx, &env)
    }

    // ---- pure unit tests for the byte codecs ----

    #[test]
    fn f16_roundtrips_exact_representables() {
        for &v in &[0.0f32, 1.0, -2.0, 0.5, 1024.0, -0.25, 3.5] {
            let h = narrow_f16(v);
            assert_eq!(widen_f16(h), v, "f16 round trip for {v}");
        }
    }

    #[test]
    fn encode_decode_roundtrip_per_dtype() {
        for dt in [DType::F32, DType::I32, DType::I64, DType::F16] {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let raw = encode(&data, dt);
            assert_eq!(raw.len(), 4 * dt.bytes_per_elem());
            let back = decode(&raw, dt, 4);
            assert_eq!(back, data, "round trip dtype {dt}");
        }
    }

    #[test]
    fn decode_zero_pads_short_input() {
        // Only 4 bytes available but 2 f32 elements requested -> second is 0.
        let raw = 7.0f32.to_le_bytes().to_vec();
        assert_eq!(decode(&raw, DType::F32, 2), vec![7.0, 0.0]);
    }

    #[test]
    fn contiguity_check() {
        assert!(is_contiguous(&[4, 4], &[4, 1]));
        assert!(is_contiguous(&[4], &[1]));
        assert!(!is_contiguous(&[4], &[4])); // strided column
        assert!(!is_contiguous(&[2, 3], &[1, 2]));
    }

    #[test]
    fn flat_offsets_full_shape_rowmajor() {
        let (offsets, sticks) =
            flat_memory_offsets(0, &[2, 2], &[2, 1], DType::F32, None, None);
        assert_eq!(offsets, vec![0, 1, 2, 3]);
        assert_eq!(sticks, None);
    }

    #[test]
    fn flat_offsets_strided_column_counts_sticks() {
        // f16 column of a 4x4 matrix: base byte 4, strides [4], shape [4].
        // offsets 0,4,8,12 -> byte addrs 4,12,20,28 all in stick 0.
        let (offsets, sticks) = flat_memory_offsets(
            4,
            &[4],
            &[4],
            DType::F16,
            None,
            Some(STICK_BYTES),
        );
        assert_eq!(offsets, vec![0, 4, 8, 12]);
        assert_eq!(sticks, Some(1));
    }

    // ---- helpers to build views ----

    fn hbm_memref(base_ptr: i64, shape: Vec<usize>, strides: Vec<i64>, dtype: DType) -> MemRef {
        MemRef {
            base_ptr,
            shape,
            strides,
            space: MemorySpace::Hbm,
            dtype,
            coordinate_set: None,
        }
    }

    // ---- load fast path ----

    #[test]
    fn load_contiguous_hbm_decodes_f32() {
        let mut ctx = single_core_context();
        // Allocate a 4-element f32 region in HBM.
        let stick = ctx.hbm.borrow_mut().allocate(4 * 4);
        let byte_addr = stick * STICK_BYTES;
        let payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        ctx.hbm.borrow_mut().write_bytes(byte_addr, &payload);

        let m = hbm_memref(stick, vec![4], vec![1], DType::F32);
        let tr = m.to_tile_ref();
        let tile = load_data(&mut ctx, &tr, None, None).unwrap();
        assert_eq!(tile.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tile.shape, vec![4]);
        // 16 bytes from a stick boundary -> exactly 1 stick.
        assert_eq!(tile.unique_sticks, Some(1));
    }

    #[test]
    fn load_lx_decodes_f16() {
        let mut ctx = single_core_context();
        let raw: Vec<u8> = [1.0f32, 2.0, 4.0]
            .iter()
            .flat_map(|x| narrow_f16(*x).to_le_bytes())
            .collect();
        ctx.lx.borrow_mut().write_bytes(0, &raw);

        let m = MemRef {
            base_ptr: 0,
            shape: vec![3],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F16,
            coordinate_set: None,
        };
        let tile = load_data(&mut ctx, &m.to_tile_ref(), None, None).unwrap();
        assert_eq!(tile.data, vec![1.0, 2.0, 4.0]);
        assert_eq!(tile.unique_sticks, None); // LX: no sticks
    }

    // ---- load slow path (coords) ----

    #[test]
    fn load_with_coords_gathers() {
        let mut ctx = single_core_context();
        // 4x4 f32 matrix values 0..15 in HBM.
        let stick = ctx.hbm.borrow_mut().allocate(16 * 4);
        let byte_addr = stick * STICK_BYTES;
        let payload: Vec<u8> = (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect();
        ctx.hbm.borrow_mut().write_bytes(byte_addr, &payload);

        let m = hbm_memref(stick, vec![4, 4], vec![4, 1], DType::F32);
        let tr = m.to_tile_ref();
        // Gather the diagonal: (0,0),(1,1),(2,2),(3,3) -> 0,5,10,15.
        let coords = vec![vec![0, 0], vec![1, 1], vec![2, 2], vec![3, 3]];
        let tile = load_data(&mut ctx, &tr, Some(&coords), Some(vec![4])).unwrap();
        assert_eq!(tile.data, vec![0.0, 5.0, 10.0, 15.0]);
    }

    // ---- store round trips ----

    #[test]
    fn store_contiguous_hbm_roundtrips() {
        let mut ctx = single_core_context();
        let stick = ctx.hbm.borrow_mut().allocate(4 * 4);
        let m = hbm_memref(stick, vec![4], vec![1], DType::F32);
        let tr = m.to_tile_ref();

        let tile = Tile::compute(vec![10.0, 20.0, 30.0, 40.0], DType::F32, vec![4]);
        let sticks = store_data(&mut ctx, &tile, &tr, None).unwrap();
        assert_eq!(sticks, 1);

        let back = load_data(&mut ctx, &tr, None, None).unwrap();
        assert_eq!(back.data, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn store_lx_returns_zero_sticks() {
        let mut ctx = single_core_context();
        let m = MemRef {
            base_ptr: 256,
            shape: vec![3],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        let tr = m.to_tile_ref();
        let tile = Tile::compute(vec![1.0, 2.0, 3.0], DType::F32, vec![3]);
        let sticks = store_data(&mut ctx, &tile, &tr, None).unwrap();
        assert_eq!(sticks, 0);
        let back = load_data(&mut ctx, &tr, None, None).unwrap();
        assert_eq!(back.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn store_with_coords_scatters_rmw() {
        let mut ctx = single_core_context();
        // Pre-fill a 4-element f32 LX region with zeros, then scatter into
        // offsets 0 and 2 via coords on a [2] logical tile with stride [2].
        let m = MemRef {
            base_ptr: 512,
            shape: vec![2],
            strides: vec![2],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        ctx.lx.borrow_mut().write_bytes(512, &[0u8; 16]); // 4 f32 zeros
        let tr = m.to_tile_ref();
        let tile = Tile::compute(vec![7.0, 9.0], DType::F32, vec![2]);
        // coords (0) and (1) over strides [2] -> flat offsets 0 and 2.
        let coords = vec![vec![0], vec![1]];
        store_data(&mut ctx, &tile, &tr, Some(&coords)).unwrap();

        let raw = ctx.lx.borrow().read_bytes(512, 16);
        let vals = decode(&raw, DType::F32, 4);
        assert_eq!(vals, vec![7.0, 0.0, 9.0, 0.0]);
    }

    // ---- end-to-end: load -> addf -> store through the dispatch table ----

    fn ident1() -> Attr {
        Attr::AffineMap(AffineMap::identity(1))
    }

    #[test]
    fn vector_add_load_addf_store_end_to_end() {
        let mut ctx = single_core_context();

        // Two input vectors of length 4 in HBM, plus an output region.
        let n = 4usize;
        let a_stick = ctx.hbm.borrow_mut().allocate((n * 4) as i64);
        let b_stick = ctx.hbm.borrow_mut().allocate((n * 4) as i64);
        let out_stick = ctx.hbm.borrow_mut().allocate((n * 4) as i64);
        let a_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let b_bytes: Vec<u8> = [10.0f32, 20.0, 30.0, 40.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        ctx.hbm.borrow_mut().write_bytes(a_stick * STICK_BYTES, &a_bytes);
        ctx.hbm.borrow_mut().write_bytes(b_stick * STICK_BYTES, &b_bytes);

        // Bind the three base pointers (as stick indices) plus an index 0.
        ctx.set_value("%pa", Value::Index(a_stick));
        ctx.set_value("%pb", Value::Index(b_stick));
        ctx.set_value("%pout", Value::Index(out_stick));
        ctx.set_value("%i", Value::Index(0));

        let view = |res: &str, ptr: &str| {
            Operation::new(Some(res), "ktdp.construct_memory_view", &[ptr])
                .with_attr("shape", Attr::IntList(vec![n as i64]))
                .with_attr("strides", Attr::IntList(vec![1]))
                .with_attr("memory_space", Attr::Str("HBM".into()))
                .with_attr("dtype", Attr::Str("f32".into()))
        };
        let access = |res: &str, view: &str| {
            Operation::new(Some(res), "ktdp.construct_access_tile", &[view, "%i"])
                .with_attr("shape", Attr::IntList(vec![n as i64]))
                .with_attr("base_map", ident1())
        };

        let ops = vec![
            view("%va", "%pa"),
            view("%vb", "%pb"),
            view("%vout", "%pout"),
            access("%aa", "%va"),
            access("%ab", "%vb"),
            access("%aout", "%vout"),
            Operation::new(Some("%ta"), "ktdp.load", &["%aa"]),
            Operation::new(Some("%tb"), "ktdp.load", &["%ab"]),
            Operation::new(Some("%tc"), "arith.addf", &["%ta", "%tb"]),
            Operation::new(None, "ktdp.store", &["%tc", "%aout"]),
        ];
        run(&ops, &mut ctx).unwrap();

        // The stored output region should hold the element-wise sum.
        let raw = ctx.hbm.borrow().read_bytes(out_stick * STICK_BYTES, n * 4);
        let vals = decode(&raw, DType::F32, n);
        assert_eq!(vals, vec![11.0, 22.0, 33.0, 44.0]);
    }

    // ---- end-to-end with a coordinate_set on the access tile ----

    #[test]
    fn load_via_coordinate_set_through_handler() {
        let mut ctx = single_core_context();
        let n = 4usize;
        let stick = ctx.hbm.borrow_mut().allocate((n * 4) as i64);
        let payload: Vec<u8> = [5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        ctx.hbm.borrow_mut().write_bytes(stick * STICK_BYTES, &payload);

        ctx.set_value("%p", Value::Index(stick));
        ctx.set_value("%i", Value::Index(0));

        // coordinate_set { d0 : d0 >= 0 } over shape [4] selects all coords in
        // order -> behaves like a full contiguous gather.
        let css = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::Dim(0),
                kind: ConstraintKind::GreaterEq,
            }],
        };

        let ops = vec![
            Operation::new(Some("%v"), "ktdp.construct_memory_view", &["%p"])
                .with_attr("shape", Attr::IntList(vec![n as i64]))
                .with_attr("strides", Attr::IntList(vec![1]))
                .with_attr("memory_space", Attr::Str("HBM".into()))
                .with_attr("dtype", Attr::Str("f32".into())),
            Operation::new(Some("%a"), "ktdp.construct_access_tile", &["%v", "%i"])
                .with_attr("shape", Attr::IntList(vec![n as i64]))
                .with_attr("base_map", ident1())
                .with_attr("coordinate_set", Attr::AffineSet(css)),
            Operation::new(Some("%t"), "ktdp.load", &["%a"]),
        ];
        run(&ops, &mut ctx).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::Tile(t) => assert_eq!(t.data, vec![5.0, 6.0, 7.0, 8.0]),
            other => panic!("expected Tile, got {other:?}"),
        }
    }

    #[test]
    fn store_rejects_non_tile_first_operand() {
        let mut ctx = single_core_context();
        ctx.set_value("%x", Value::Index(3));
        ctx.set_value("%y", Value::Index(4));
        let op = Operation::new(None, "ktdp.store", &["%x", "%y"]);
        let err = run(&[op], &mut ctx).unwrap_err();
        assert!(err.contains("Tile"), "unexpected error: {err}");
    }

    #[test]
    fn ndindex_scalar_shape_emits_one_point() {
        let mut count = 0;
        ndindex(&[], &mut |_| count += 1);
        assert_eq!(count, 1);
        // Empty axis -> no points.
        let mut count2 = 0;
        ndindex(&[0, 3], &mut |_| count2 += 1);
        assert_eq!(count2, 0);
    }

    // =======================================================================
    // Distributed + indirect path tests
    // =======================================================================

    use crate::memref::{
        CoordinateSet, DimSubscript, DistributedMemRef, IndirectAccessTile, ParentRef,
    };

    /// Inclusive box `[lo, hi]` as an `AffineSet`: for each axis i,
    /// `d_i - lo_i >= 0` and `hi_i - d_i >= 0`.
    fn box_affine(lo: &[i64], hi: &[i64]) -> AffineSet {
        let mut constraints = Vec::new();
        for i in 0..lo.len() {
            constraints.push(Constraint {
                expr: AffineExpr::Sub(Rc::new(AffineExpr::Dim(i)), Rc::new(AffineExpr::Const(lo[i]))),
                kind: ConstraintKind::GreaterEq,
            });
            constraints.push(Constraint {
                expr: AffineExpr::Sub(Rc::new(AffineExpr::Const(hi[i])), Rc::new(AffineExpr::Dim(i))),
                kind: ConstraintKind::GreaterEq,
            });
        }
        AffineSet { num_dims: lo.len(), num_syms: 0, constraints }
    }

    // ---- lower_to_box ----

    #[test]
    fn lower_to_box_is_inclusive() {
        // affine [2,5] on one axis -> inclusive BoxSet lo=2 hi=5.
        let b = lower_to_box(&box_affine(&[2], &[5])).expect("lowerable");
        assert_eq!(b.lo, vec![2]);
        assert_eq!(b.hi, vec![5]);
        // non-axis-aligned -> None.
        let diag = AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::Add(Rc::new(AffineExpr::Dim(0)), Rc::new(AffineExpr::Dim(1))),
                kind: ConstraintKind::GreaterEq,
            }],
        };
        assert!(lower_to_box(&diag).is_none());
    }

    // ---- distributed_tile_access: 2-partition routing ----

    /// Two HBM partitions of a 1-D length-8 f32 tensor: B_0 owns coords [0,3],
    /// B_1 owns [4,7]. Each partition's data lives at its own stick.
    fn two_partition_dist(ctx: &mut CoreContext) -> (DistributedMemRef, i64, i64) {
        let s0 = ctx.hbm.borrow_mut().allocate(4 * 4);
        let s1 = ctx.hbm.borrow_mut().allocate(4 * 4);
        // Partition 0 holds global coords 0..3 -> values 0,1,2,3.
        let p0: Vec<u8> = [0.0f32, 1.0, 2.0, 3.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        // Partition 1 holds global coords 4..7 -> values 40,50,60,70.
        let p1: Vec<u8> = [40.0f32, 50.0, 60.0, 70.0].iter().flat_map(|x| x.to_le_bytes()).collect();
        ctx.hbm.borrow_mut().write_bytes(s0 * STICK_BYTES, &p0);
        ctx.hbm.borrow_mut().write_bytes(s1 * STICK_BYTES, &p1);

        let mk = |base: i64, lo: i64, hi: i64| MemRef {
            base_ptr: base,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Hbm,
            dtype: DType::F32,
            coordinate_set: Some(box_affine(&[lo], &[hi])),
        };
        let dist = DistributedMemRef::new(
            vec![mk(s0, 0, 3), mk(s1, 4, 7)],
            vec![8],
            DType::F32,
        )
        .unwrap();
        (dist, s0, s1)
    }

    #[test]
    fn distributed_tile_access_survivors_box_fastpath() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        // Access the full [0,8) window: x=0, access_shape=8, both partitions survive.
        let dtr = distributed_tile_access(
            &dist,
            &[8],
            &AffineMap::identity(1),
            &[0],
            None,
        )
        .unwrap();
        assert_eq!(dtr.partitions.len(), 2);
        assert_eq!(dtr.global_base, Some(vec![0]));
        // Each survivor carries a Box coordinate_set and partition origin.
        match &dtr.partitions[0].coordinate_set {
            Some(CoordinateSet::Box(b)) => {
                assert_eq!(b.lo, vec![0]);
                assert_eq!(b.hi, vec![3]);
            }
            other => panic!("expected Box C_0, got {other:?}"),
        }
        assert_eq!(dtr.partitions[0].partition_origin, Some(vec![0]));
        assert_eq!(dtr.partitions[1].partition_origin, Some(vec![4]));
    }

    #[test]
    fn distributed_tile_access_partial_window_drops_partition() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        // Access window [0,3) only -> only partition 0 survives.
        let dtr = distributed_tile_access(&dist, &[3], &AffineMap::identity(1), &[0], None).unwrap();
        assert_eq!(dtr.partitions.len(), 1);
        match &dtr.partitions[0].coordinate_set {
            Some(CoordinateSet::Box(b)) => {
                assert_eq!(b.lo, vec![0]);
                assert_eq!(b.hi, vec![2]); // C_0 = [0,3] ∩ [0,2] = [0,2]
            }
            other => panic!("expected Box, got {other:?}"),
        }
    }

    #[test]
    fn distributed_tile_access_no_coverage_errors() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        // Window starting at global 100 covers no partition.
        let err = distributed_tile_access(&dist, &[2], &AffineMap::identity(1), &[100], None)
            .unwrap_err();
        assert!(err.contains("no partition"), "unexpected: {err}");
    }

    // ---- distributed_load: 2-partition gather ----

    #[test]
    fn distributed_load_gathers_across_two_partitions() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        let dtr = distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();
        let tile = distributed_load(&mut ctx, &dtr, Some(vec![8])).unwrap();
        // Concatenation of both partitions in global-coord order.
        assert_eq!(tile.data, vec![0.0, 1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0]);
        assert_eq!(tile.shape, vec![8]);
        // Both partitions are HBM -> unique_sticks aggregated (1 each).
        assert_eq!(tile.unique_sticks, Some(2));
    }

    #[test]
    fn distributed_store_then_load_roundtrips_two_partitions() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        let dtr = distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();

        // Scatter a fresh 8-vector across both partitions.
        let tile = Tile::compute(
            vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            DType::F32,
            vec![8],
        );
        let sticks = distributed_store(&mut ctx, &tile, &dtr).unwrap();
        assert_eq!(sticks, 2); // one HBM stick per partition

        // Re-resolve (survivor TileRefs are consumed) and read back.
        let dtr2 = distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();
        let back = distributed_load(&mut ctx, &dtr2, Some(vec![8])).unwrap();
        assert_eq!(back.data, vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]);
    }

    // ---- distributed end-to-end through the ktdp.load dispatch handler ----

    #[test]
    fn distributed_load_through_access_tile_parent() {
        let mut ctx = single_core_context();
        let (dist, _, _) = two_partition_dist(&mut ctx);
        let dtr = distributed_tile_access(&dist, &[8], &AffineMap::identity(1), &[0], None).unwrap();
        // Wrap the DistributedTileRef in an AccessTile and load via the handler.
        let access = AccessTile {
            parent_ref: ParentRef::Dist(dtr),
            shape: vec![8],
            base_map: AffineMap::identity(1),
            coordinate_set: None,
            coordinate_order: None,
        };
        ctx.set_value("%a", Value::AccessTile(access));
        let op = Operation::new(Some("%t"), "ktdp.load", &["%a"]);
        run(&[op], &mut ctx).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::Tile(t) => assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0]),
            other => panic!("expected Tile, got {other:?}"),
        }
    }

    // ---- indirect gather ----

    /// 1-D vss over a single intermediate var (length 4), trivially satisfiable.
    fn vss_1d() -> AffineSet {
        AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![Constraint {
                expr: AffineExpr::Dim(0),
                kind: ConstraintKind::GreaterEq,
            }],
        }
    }

    #[test]
    fn indirect_gather_reads_through_index_view() {
        let mut ctx = single_core_context();
        // Parent X: 8 f32 values in LX at byte 0 -> 10,11,...,17.
        let x_data: Vec<u8> = (0..8).flat_map(|i| (10.0f32 + i as f32).to_le_bytes()).collect();
        ctx.lx.borrow_mut().write_bytes(0, &x_data);
        // Index view IDX: i32 values [3, 0, 5, 1] at byte 256.
        let idx_data: Vec<u8> = [3i32, 0, 5, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.lx.borrow_mut().write_bytes(256, &idx_data);

        let x_view = MemRef {
            base_ptr: 0,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        let idx_view = MemRef {
            base_ptr: 256,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::I32,
            coordinate_set: None,
        };

        // X[ ind(IDX[m]) ] over intermediate var m in [0,4): gather X at the
        // indices held in IDX -> X[3], X[0], X[5], X[1] = 13, 10, 15, 11.
        let iat = IndirectAccessTile {
            parent_ref: x_view,
            shape: vec![4],
            dim_subscripts: vec![DimSubscript::Indirect { view: 0 }],
            index_views: vec![idx_view],
            variables_space_set: vss_1d(),
            variables_space_order: None,
            extra: std::collections::HashMap::new(),
        };

        let tile = indirect_load(&mut ctx, &iat, None).unwrap();
        assert_eq!(tile.data, vec![13.0, 10.0, 15.0, 11.0]);
        assert_eq!(tile.shape, vec![4]);
        // LX index view -> no index sticks.
        assert_eq!(tile.index_unique_sticks, Some(0));
    }

    #[test]
    fn indirect_gather_negative_index_rejected() {
        let mut ctx = single_core_context();
        let x_data: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
        ctx.lx.borrow_mut().write_bytes(0, &x_data);
        // IDX holds a negative index -> must be rejected (no NumPy wrap).
        let idx_data: Vec<u8> = [-1i32, 0, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.lx.borrow_mut().write_bytes(256, &idx_data);

        let x_view = MemRef {
            base_ptr: 0,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        let idx_view = MemRef {
            base_ptr: 256,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::I32,
            coordinate_set: None,
        };
        let iat = IndirectAccessTile {
            parent_ref: x_view,
            shape: vec![4],
            dim_subscripts: vec![DimSubscript::Indirect { view: 0 }],
            index_views: vec![idx_view],
            variables_space_set: vss_1d(),
            variables_space_order: None,
            extra: std::collections::HashMap::new(),
        };
        let err = indirect_load(&mut ctx, &iat, None).unwrap_err();
        assert!(err.contains("negative"), "unexpected: {err}");
    }

    #[test]
    fn indirect_scatter_then_direct_load_roundtrips() {
        let mut ctx = single_core_context();
        // Destination X: 8 f32 zeros in HBM.
        let xs = ctx.hbm.borrow_mut().allocate(8 * 4);
        ctx.hbm.borrow_mut().write_bytes(xs * STICK_BYTES, &[0u8; 32]);
        // IDX in LX: scatter positions [2, 5, 0, 7].
        let idx_data: Vec<u8> = [2i32, 5, 0, 7].iter().flat_map(|v| v.to_le_bytes()).collect();
        ctx.lx.borrow_mut().write_bytes(512, &idx_data);

        let x_view = MemRef {
            base_ptr: xs,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Hbm,
            dtype: DType::F32,
            coordinate_set: None,
        };
        let idx_view = MemRef {
            base_ptr: 512,
            shape: vec![4],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::I32,
            coordinate_set: None,
        };
        let iat = IndirectAccessTile {
            parent_ref: x_view.clone(),
            shape: vec![4],
            dim_subscripts: vec![DimSubscript::Indirect { view: 0 }],
            index_views: vec![idx_view],
            variables_space_set: vss_1d(),
            variables_space_order: None,
            extra: std::collections::HashMap::new(),
        };

        // Scatter [100,200,300,400] to X[2],X[5],X[0],X[7].
        let src = Tile::compute(vec![100.0, 200.0, 300.0, 400.0], DType::F32, vec![4]);
        let sticks = indirect_store(&mut ctx, &src, &iat).unwrap();
        // Parent is HBM (one stick), idx view is LX (0) -> at least 1.
        assert!(sticks >= 1);

        // Direct full load of X confirms the scatter.
        let back = load_data(&mut ctx, &x_view.to_tile_ref(), None, None).unwrap();
        assert_eq!(back.data, vec![300.0, 0.0, 100.0, 0.0, 0.0, 200.0, 0.0, 400.0]);
    }

    #[test]
    fn indirect_load_rejects_non_permutation_vso() {
        let mut ctx = single_core_context();
        let x_view = MemRef {
            base_ptr: 0,
            shape: vec![8],
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F32,
            coordinate_set: None,
        };
        // vso (d0) -> (2*d0) is a scaling, not a permutation.
        let bad = AffineMap {
            num_dims: 1,
            num_syms: 0,
            exprs: vec![AffineExpr::Mul(
                Rc::new(AffineExpr::Const(2)),
                Rc::new(AffineExpr::Dim(0)),
            )],
        };
        let iat = IndirectAccessTile {
            parent_ref: x_view,
            shape: vec![4],
            dim_subscripts: vec![DimSubscript::Direct { var_index: 0 }],
            index_views: vec![],
            variables_space_set: vss_1d(),
            variables_space_order: Some(bad),
            extra: std::collections::HashMap::new(),
        };
        let err = indirect_load(&mut ctx, &iat, None).unwrap_err();
        assert!(err.contains("permut"), "unexpected: {err}");
    }
}
