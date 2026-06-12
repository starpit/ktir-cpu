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

use crate::context::CoreContext;
use crate::dialects::{Dispatch, LatencyCategory};
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Value};
use crate::memory::STICK_BYTES;
use crate::memref::{AccessTile, MemorySpace, ParentRef, TileRef};
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
/// Single-allocation path only (distributed / indirect access tiles are owned
/// by other subsystems). Mirrors `ktdp__load`: when the access tile carries a
/// `coordinate_set`, enumerate its coords (reordered through
/// `coordinate_order`) and take the slow gather path; otherwise load the whole
/// contiguous/strided tile.
fn load(op: &Operation, ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("ktdp.load: missing access-tile operand".into());
    }
    let access = access_tile(ctx.get_value(&op.operands[0])?, "ktdp.load")?;
    let tile_ref = single_tile_ref(&access, "ktdp.load")?;

    let coords = enumerated_coords(&access);
    let result_shape = if coords.is_some() {
        Some(access.shape.clone())
    } else {
        None
    };

    let tile = load_data(ctx, &tile_ref, coords.as_deref(), result_shape)?;
    Ok(Some(Value::Tile(tile)))
}

/// `ktdp.store %tile, %access_tile` — scatter a tile back to its footprint.
///
/// Stores have no IR result; the handler computes `unique_sticks` (the latency
/// sideband Python returns) but binds nothing. Mirrors `ktdp__store`.
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
    let access = access_tile(ctx.get_value(&op.operands[1])?, "ktdp.store")?;
    let tile_ref = single_tile_ref(&access, "ktdp.store")?;

    let coords = enumerated_coords(&access);
    // Compute the unique-stick sideband (used by the latency tracker). We do
    // not bind it as an SSA result — the op has none.
    let _unique_sticks = store_data(ctx, &tile, &tile_ref, coords.as_deref())?;
    Ok(None)
}

/// Resolve the access tile's coordinate list, if it carries a coordinate_set.
/// Enumerate over `access.shape`, then reorder each point through
/// `coordinate_order` when present (mirrors `css.enumerate` + `cso.eval`).
fn enumerated_coords(access: &AccessTile) -> Option<Vec<Vec<i64>>> {
    let css = access.coordinate_set.as_ref()?;
    let mut coords = css.enumerate(&access.shape, &[]);
    if let Some(order) = &access.coordinate_order {
        coords = coords.iter().map(|pt| order.eval(pt, &[])).collect();
    }
    Some(coords)
}

/// Extract the `AccessTile`, rejecting the indirect-access-tile variant (owned
/// by the indirect subsystem).
fn access_tile(v: &Value, who: &str) -> Result<AccessTile, String> {
    match v {
        Value::AccessTile(a) => Ok(a.clone()),
        Value::IndirectAccessTile(_) => Err(format!(
            "{who}: indirect access tiles are handled by the indirect load/store path"
        )),
        other => Err(format!("{who}: expected an AccessTile, got {other:?}")),
    }
}

/// Unwrap the single-allocation `TileRef` parent, rejecting the distributed
/// survivor list (owned by the distributed subsystem).
fn single_tile_ref(access: &AccessTile, who: &str) -> Result<TileRef, String> {
    match &access.parent_ref {
        ParentRef::Tile(tr) => Ok(tr.clone()),
        ParentRef::Dist(_) => Err(format!(
            "{who}: distributed access tiles are handled by the distributed load/store path"
        )),
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
fn is_contiguous(shape: &[usize], strides: &[i64]) -> bool {
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

/// Standard f32 -> f16 conversion (round-to-nearest-even), returning the f16
/// bit pattern. No `half` dependency.
fn narrow_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if x.is_nan() {
        return sign | 0x7e00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant_with_implicit = mant | 0x0080_0000;
        let shift = (14 - exp) as u32;
        let mut half_mant = mant_with_implicit >> shift;
        if shift >= 1 && (mant_with_implicit >> (shift - 1)) & 1 == 1 {
            half_mant += 1;
        }
        sign | half_mant as u16
    } else if exp >= 0x1f {
        sign | 0x7c00
    } else {
        let mut half = sign | ((exp as u16) << 10) | (mant >> 13) as u16;
        let round_bit = mant & 0x0000_1000;
        let sticky = mant & 0x0000_0fff;
        if round_bit != 0 && (sticky != 0 || (half & 1) == 1) {
            half += 1;
        }
        half
    }
}

/// Convert an f16 bit pattern back to f32.
fn widen_f16(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let f32_exp = (e + 127 - 15 + 1) as u32;
            sign | (f32_exp << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let f32_exp = exp + 127 - 15;
        sign | (f32_exp << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::{AffineExpr, AffineMap, AffineSet, Constraint, ConstraintKind};
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};
    use crate::ir::Attr;
    use crate::memref::{MemRef, MemorySpace};

    fn run(ops: &[Operation], ctx: &mut CoreContext) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv { dispatch: &dispatch, grid: &grid };
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
}
