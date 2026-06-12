#![allow(clippy::needless_range_loop, clippy::type_complexity)]
// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Port of `tests/test_distributed_view.py` — `construct_distributed_memory_view`
//! plus the `distributed_tile_access` / `distributed_load` / `distributed_store`
//! data path (RFC 0682 §3.3, implemented in `ops_memory.rs`).
//!
//! Translation notes (Python -> Rust)
//! ----------------------------------
//! * The Python suite drives whole MLIR kernels through `KTIRInterpreter`,
//!   seeding the partition memories with a monkey-patched `_prepare_execution`
//!   hook.  The Rust crate's `execute_function` does **not** expose a
//!   memory-seeding hook (it only marshals tensor args into HBM), so the
//!   2-partition copy table is driven at the ops layer instead: build a
//!   `DistributedMemRef` directly, seed each partition's strided block into the
//!   right backing store, then run `distributed_tile_access` -> `distributed_load`
//!   -> `distributed_store` and read the contiguous output back.  This exercises
//!   the SAME gather/scatter code (`ops_memory::distributed_*`) the kernel path
//!   reaches, and checks the SAME data-correctness invariant
//!   (`actual == reference_slice`).
//! * Python's HBM is *byte*-addressed in the test (`mem.write(byte_ptr, ...)`),
//!   while the Rust `MemRef.base_ptr` for HBM is a *stick* index.  Each HBM
//!   partition is therefore `allocate`d (yielding a stick) and seeded at
//!   `stick * STICK_BYTES`.  LX `base_ptr` is a byte address in both.
//! * Python's `coordinate_set` is lowered to a `BoxSet` by `parse_affine_set`;
//!   the box-form `affine_set` here is built as an `AffineSet` and lowered by
//!   `distributed_tile_access`'s `lower_to_box` fast path — same effect.
//! * Python `BoxSet` is half-open `[lo, hi)`.  The Rust crate's
//!   `distributed_tile_access` emits an **inclusive** `[lo, hi]` `BoxSet` for
//!   each survivor's `C_i`, so the structural assertions translate
//!   `BoxSet(lo, hi)` -> inclusive `lo..=hi-1`.
//! * `test_distributed_view_copy_rfc` (RFC §C.3 example file) is `xfail` in
//!   Python (per-core LX routing gap) and depends on an example `.mlir` file
//!   loaded by path; it has no faithful crate analogue here and is left as an
//!   `#[ignore]` stub (see `skipped`).
//! * The slow-path fixture test parses partition sets via `parse_affine_set_raw`
//!   to *force* the AffineSet enumeration path and asserts the survivor stores a
//!   `list` (Python type check).  The Rust survivor instead stores a
//!   `CoordinateSet::Points` when the fast box path is unavailable; the fixture
//!   here drives the slow path with non-axis-aligned (diagonal-masked) partition
//!   sets so `lower_to_box` returns `None`, and asserts `CoordinateSet::Points`.

use ktir_cpu::affine::{AffineExpr, AffineMap, AffineSet, BoxSet, Constraint, ConstraintKind};
use ktir_cpu::codec;
use ktir_cpu::context::CoreContext;
use ktir_cpu::dtypes::DType;
use ktir_cpu::interpreter::single_core_context;
use ktir_cpu::memory::STICK_BYTES;
use ktir_cpu::memref::{
    CoordinateSet, DistributedMemRef, DistributedTileRef, MemRef, MemorySpace,
};
use ktir_cpu::ops_memory::{distributed_load, distributed_store, distributed_tile_access};
use ktir_cpu::tile::Tile;

// ===========================================================================
// affine-set / memory helpers
// ===========================================================================

/// Inclusive box `[lo, hi]` (per axis) as an `AffineSet`: `d_i - lo_i >= 0`,
/// `hi_i - d_i >= 0`.  Mirrors the Python `_set_box` builder (which emits the
/// same inclusive box constraints), and is lowered to a `BoxSet` by
/// `distributed_tile_access`'s fast path.
fn box_affine(lo: &[i64], hi: &[i64]) -> AffineSet {
    let mut constraints = Vec::new();
    for i in 0..lo.len() {
        constraints.push(Constraint {
            expr: AffineExpr::Sub(
                Box::new(AffineExpr::Dim(i)),
                Box::new(AffineExpr::Const(lo[i])),
            ),
            kind: ConstraintKind::GreaterEq,
        });
        constraints.push(Constraint {
            expr: AffineExpr::Sub(
                Box::new(AffineExpr::Const(hi[i])),
                Box::new(AffineExpr::Dim(i)),
            ),
            kind: ConstraintKind::GreaterEq,
        });
    }
    AffineSet { num_dims: lo.len(), num_syms: 0, constraints }
}

/// Allocate a backing region for a partition and write `block` (logical
/// `[nrows, ncols]`, row-major source order) into it using element `strides`,
/// then return a `MemRef` describing the partition.
///
/// Mirrors Python `_write_strided`: element (i, j) lands at element offset
/// `i*strides[0] + j*strides[1]` from the partition base; holes are left zero.
/// HBM partitions are `allocate`d (base_ptr = stick index, data at
/// `stick * STICK_BYTES`); LX partitions use a caller-supplied byte address.
fn seed_partition(
    ctx: &mut CoreContext,
    block: &[Vec<f32>], // [nrows][ncols] row-major logical values
    strides: &[i64],
    space: MemorySpace,
    coordinate_set: AffineSet,
    lx_byte_addr: i64,
) -> MemRef {
    let nrows = block.len();
    let ncols = if nrows > 0 { block[0].len() } else { 0 };
    // Strided span (max element offset + 1).
    let mut span = 1usize;
    for i in 0..nrows {
        for j in 0..ncols {
            let off = (i as i64) * strides[0] + (j as i64) * strides[1];
            span = span.max(off as usize + 1);
        }
    }
    let mut buf = vec![0.0f32; span];
    for i in 0..nrows {
        for j in 0..ncols {
            let off = ((i as i64) * strides[0] + (j as i64) * strides[1]) as usize;
            buf[off] = block[i][j];
        }
    }
    let raw = codec::encode(&buf, DType::F16);

    match space {
        MemorySpace::Hbm => {
            let stick = ctx.hbm.borrow_mut().allocate(raw.len() as i64);
            ctx.hbm.borrow_mut().write_bytes(stick * STICK_BYTES, &raw);
            MemRef {
                base_ptr: stick,
                shape: vec![nrows, ncols],
                strides: strides.to_vec(),
                space: MemorySpace::Hbm,
                dtype: DType::F16,
                coordinate_set: Some(coordinate_set),
            }
        }
        MemorySpace::Lx { core_id } => {
            let lx = ctx.get_lx(core_id.map(|c| c as usize));
            lx.borrow_mut().write_bytes(lx_byte_addr, &raw);
            MemRef {
                base_ptr: lx_byte_addr,
                shape: vec![nrows, ncols],
                strides: strides.to_vec(),
                space: MemorySpace::Lx { core_id },
                dtype: DType::F16,
                coordinate_set: Some(coordinate_set),
            }
        }
    }
}

/// 4x4 reference tensor: `arange(16)` reshaped row-major, f16-exact.
fn reference_4x4() -> Vec<Vec<f32>> {
    (0..4).map(|r| (0..4).map(|c| (r * 4 + c) as f32).collect()).collect()
}

/// Extract the sub-block `[r0, r0+nr) x [c0, c0+nc)` of `full`.
fn slice_block(full: &[Vec<f32>], r0: usize, c0: usize, nr: usize, nc: usize) -> Vec<Vec<f32>> {
    (0..nr).map(|i| (0..nc).map(|j| full[r0 + i][c0 + j]).collect()).collect()
}

// ===========================================================================
// 2-partition distributed copy table (port of test_distributed_copy)
// ===========================================================================

#[derive(Clone)]
struct PartitionSpec {
    rows: (usize, usize), // inclusive global row range
    cols: (usize, usize), // inclusive global col range
    space: MemorySpace,
    strides: [i64; 2],
    lx_byte_addr: i64, // ignored for HBM
}

impl PartitionSpec {
    fn nrows(&self) -> usize {
        self.rows.1 - self.rows.0 + 1
    }
    fn ncols(&self) -> usize {
        self.cols.1 - self.cols.0 + 1
    }
}

#[derive(Clone)]
struct DistCopySpec {
    global_shape: (usize, usize),
    p0: PartitionSpec,
    p1: PartitionSpec,
    access_shape: (usize, usize),
    indices: [i64; 2],
    id: &'static str,
}

fn hbm() -> MemorySpace {
    MemorySpace::Hbm
}

/// LX partition on core 0 (matches Python's `interp.memory.get_lx(0)`).
fn lx0() -> MemorySpace {
    MemorySpace::Lx { core_id: Some(0) }
}

/// Build the 15-case copy table (mirrors `_CASES` in the Python module).
fn cases() -> Vec<DistCopySpec> {
    let lx_addr = 4096; // arbitrary LX byte address for an LX partition
    vec![
        // --- Row-band partitioning ---
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 1), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (2, 3), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            access_shape: (4, 4), indices: [0, 0], id: "row_hbm_hbm_full",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 1), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (2, 3), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            access_shape: (2, 4), indices: [0, 0], id: "row_hbm_hbm_partial_p1_pruned",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 1), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (2, 3), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            access_shape: (2, 2), indices: [1, 1], id: "row_hbm_hbm_subtile_nonzero",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 1), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (2, 3), cols: (0, 3), space: lx0(), strides: [1, 4], lx_byte_addr: lx_addr },
            access_shape: (4, 4), indices: [0, 0], id: "row_hbm_lx_col_packed_full",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 1), cols: (0, 3), space: lx0(), strides: [1, 4], lx_byte_addr: lx_addr },
            p1: PartitionSpec { rows: (2, 3), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            access_shape: (4, 4), indices: [0, 0], id: "row_lx_hbm_col_packed_full",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 1), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (2, 3), cols: (0, 3), space: lx0(), strides: [1, 4], lx_byte_addr: lx_addr },
            access_shape: (2, 2), indices: [1, 1], id: "row_hbm_lx_subtile_nonzero",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 0), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (1, 3), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            access_shape: (4, 4), indices: [0, 0], id: "row_hbm_hbm_unequal_full",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 0), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (1, 3), cols: (0, 3), space: hbm(), strides: [4, 1], lx_byte_addr: 0 },
            access_shape: (2, 4), indices: [1, 0], id: "row_hbm_hbm_unequal_partial_p0_pruned",
        },
        // --- Col-band partitioning ---
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 3), cols: (0, 1), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (0, 3), cols: (2, 3), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            access_shape: (4, 4), indices: [0, 0], id: "col_hbm_hbm_full",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 3), cols: (0, 1), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (0, 3), cols: (2, 3), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            access_shape: (4, 2), indices: [0, 0], id: "col_hbm_hbm_partial_p1_pruned",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 3), cols: (0, 1), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (0, 3), cols: (2, 3), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            access_shape: (2, 2), indices: [1, 1], id: "col_hbm_hbm_subtile_nonzero",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 3), cols: (0, 1), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (0, 3), cols: (2, 3), space: lx0(), strides: [1, 4], lx_byte_addr: lx_addr },
            access_shape: (4, 4), indices: [0, 0], id: "col_hbm_lx_col_packed_full",
        },
        // --- Mixed layout ---
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 3), cols: (0, 1), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (2, 3), cols: (2, 3), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            access_shape: (4, 2), indices: [0, 0], id: "mixed_left_block_only",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 3), cols: (0, 1), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            p1: PartitionSpec { rows: (2, 3), cols: (2, 3), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            access_shape: (2, 2), indices: [2, 0], id: "mixed_bottom_left_only",
        },
        DistCopySpec {
            global_shape: (4, 4),
            p0: PartitionSpec { rows: (0, 3), cols: (0, 1), space: lx0(), strides: [1, 4], lx_byte_addr: lx_addr },
            p1: PartitionSpec { rows: (2, 3), cols: (2, 3), space: hbm(), strides: [2, 1], lx_byte_addr: 0 },
            access_shape: (2, 2), indices: [2, 0], id: "mixed_lx_left_hbm_right_bottom_only",
        },
    ]
}

/// Seed the partitions, run access -> load -> store, and return (expected,
/// actual) flat row-major access-tile data.  Mirrors `_seed_and_run`.
fn run_copy(spec: &DistCopySpec) -> (Vec<f32>, Vec<f32>) {
    let mut ctx = single_core_context();
    let full = reference_4x4();
    let (p0, p1) = (&spec.p0, &spec.p1);

    let p0_block = slice_block(&full, p0.rows.0, p0.cols.0, p0.nrows(), p0.ncols());
    let p1_block = slice_block(&full, p1.rows.0, p1.cols.0, p1.nrows(), p1.ncols());

    let p0_set = box_affine(
        &[p0.rows.0 as i64, p0.cols.0 as i64],
        &[p0.rows.1 as i64, p0.cols.1 as i64],
    );
    let p1_set = box_affine(
        &[p1.rows.0 as i64, p1.cols.0 as i64],
        &[p1.rows.1 as i64, p1.cols.1 as i64],
    );

    let mr0 = seed_partition(&mut ctx, &p0_block, &p0.strides, p0.space, p0_set, p0.lx_byte_addr);
    let mr1 = seed_partition(&mut ctx, &p1_block, &p1.strides, p1.space, p1_set, p1.lx_byte_addr);

    let dist = DistributedMemRef::new(
        vec![mr0, mr1],
        vec![spec.global_shape.0, spec.global_shape.1],
        DType::F16,
    )
    .unwrap();

    let ac = [spec.access_shape.0, spec.access_shape.1];
    let base_map = AffineMap::identity(2);

    // Allocate a contiguous HBM output region for B and zero it.
    let n_out = ac[0] * ac[1];
    let out_stick = ctx.hbm.borrow_mut().allocate((n_out * 2) as i64);
    ctx.hbm.borrow_mut().write_bytes(out_stick * STICK_BYTES, &vec![0u8; n_out * 2]);

    // Load the access tile from the distributed view.
    let dtr = distributed_tile_access(&dist, &ac, &base_map, &spec.indices, None).unwrap();
    let data = distributed_load(&mut ctx, &dtr, Some(ac.to_vec())).unwrap();

    // Store it to contiguous HBM B (row-major, full box) via a plain TileRef.
    let b = MemRef {
        base_ptr: out_stick,
        shape: ac.to_vec(),
        strides: vec![ac[1] as i64, 1],
        space: MemorySpace::Hbm,
        dtype: DType::F16,
        coordinate_set: None,
    };
    ktir_cpu::ops_memory::store_data(&mut ctx, &data, &b.to_tile_ref(), None).unwrap();

    // Read B back.
    let raw = ctx.hbm.borrow().read_bytes(out_stick * STICK_BYTES, n_out * 2);
    let actual = codec::decode(&raw, n_out, DType::F16);

    // Expected = reference slice at indices, row-major flattened.
    let r0 = spec.indices[0] as usize;
    let c0 = spec.indices[1] as usize;
    let expected_block = slice_block(&full, r0, c0, ac[0], ac[1]);
    let expected: Vec<f32> = expected_block.into_iter().flatten().collect();
    (expected, actual)
}

#[test]
fn distributed_copy_all_cases() {
    for spec in cases() {
        let (expected, actual) = run_copy(&spec);
        assert_eq!(actual, expected, "case {} mismatch", spec.id);
    }
}

// ===========================================================================
// Structural: fast path produces an (inclusive) BoxSet in surviving partitions
// (port of test_distributed_tile_access_fast_path_emits_box_set)
// ===========================================================================

#[test]
fn distributed_tile_access_fast_path_emits_box_set() {
    let mut ctx = single_core_context();
    // 2 row-band partitions of a 4x4 tensor: P0 rows 0..1, P1 rows 2..3.
    let zeros = vec![vec![0.0f32; 4]; 2];
    let b0 = box_affine(&[0, 0], &[1, 3]);
    let b1 = box_affine(&[2, 0], &[3, 3]);
    let mr0 = seed_partition(&mut ctx, &zeros, &[4, 1], MemorySpace::Hbm, b0, 0);
    let mr1 = seed_partition(&mut ctx, &zeros, &[4, 1], MemorySpace::Hbm, b1, 0);
    let dist = DistributedMemRef::new(vec![mr0, mr1], vec![4, 4], DType::F16).unwrap();

    let base_map = AffineMap::identity(2);
    let out = distributed_tile_access(&dist, &[4, 4], &base_map, &[0, 0], None).unwrap();
    assert_eq!(out.partitions.len(), 2);
    for part in &out.partitions {
        assert!(
            matches!(part.coordinate_set, Some(CoordinateSet::Box(_))),
            "fast path must store a Box coordinate_set, got {:?}",
            part.coordinate_set
        );
    }
    // Rust C_i is inclusive: Python BoxSet(lo=(0,0), hi=(2,4)) == inclusive [0,1]x[0,3].
    match &out.partitions[0].coordinate_set {
        Some(CoordinateSet::Box(b)) => {
            assert_eq!(b.lo, vec![0, 0]);
            assert_eq!(b.hi, vec![1, 3]);
        }
        other => panic!("expected Box, got {other:?}"),
    }
    match &out.partitions[1].coordinate_set {
        Some(CoordinateSet::Box(b)) => {
            assert_eq!(b.lo, vec![2, 0]);
            assert_eq!(b.hi, vec![3, 3]);
        }
        other => panic!("expected Box, got {other:?}"),
    }
    // partition_origin == min(B_i)
    assert_eq!(out.partitions[0].partition_origin, Some(vec![0, 0]));
    assert_eq!(out.partitions[1].partition_origin, Some(vec![2, 0]));
}

// ===========================================================================
// Symbolic-shape variant: same fast-path BoxSet assertion, parametrised over
// partition row counts (port of test_distributed_tile_access_dynamic_shape_emits_box_set).
//
// The Rust crate does not specialise symbolic affine sets at the ops layer the
// way the Python test does (it hands already-concrete partitions to
// distributed_tile_access); the faithful equivalent here is to build the
// concrete partition sets directly for each row count and re-check the
// geometry + BoxSet survivor guard.
// ===========================================================================

#[test]
fn distributed_tile_access_parametrised_row_counts_emit_box_set() {
    for partition_rows in [2usize, 4, 8] {
        let mut ctx = single_core_context();
        let total_rows = 2 * partition_rows;
        // B0 = rows [0, partition_rows), cols [0,3]; B1 = rows [partition_rows, 2*pr).
        let b0 = box_affine(&[0, 0], &[partition_rows as i64 - 1, 3]);
        let b1 = box_affine(
            &[partition_rows as i64, 0],
            &[2 * partition_rows as i64 - 1, 3],
        );
        let zeros = vec![vec![0.0f32; 4]; partition_rows];
        let mr0 = seed_partition(&mut ctx, &zeros, &[4, 1], MemorySpace::Hbm, b0, 0);
        let mr1 = seed_partition(&mut ctx, &zeros, &[4, 1], MemorySpace::Hbm, b1, 0);
        let dist = DistributedMemRef::new(vec![mr0, mr1], vec![total_rows, 4], DType::F16).unwrap();

        let base_map = AffineMap::identity(2);
        let out = distributed_tile_access(&dist, &[total_rows, 4], &base_map, &[0, 0], None).unwrap();
        assert_eq!(out.partitions.len(), 2, "rows={partition_rows}");
        for part in &out.partitions {
            assert!(
                matches!(part.coordinate_set, Some(CoordinateSet::Box(_))),
                "rows={partition_rows}: dynamic fast path must store a Box"
            );
        }
        // Inclusive geometry: Python BoxSet(lo=(0,0), hi=(pr,4)) == [0,pr-1]x[0,3].
        let pr = partition_rows as i64;
        match &out.partitions[0].coordinate_set {
            Some(CoordinateSet::Box(b)) => {
                assert_eq!((&b.lo[..], &b.hi[..]), (&[0i64, 0][..], &[pr - 1, 3][..]))
            }
            other => panic!("rows={partition_rows}: expected Box, got {other:?}"),
        }
        match &out.partitions[1].coordinate_set {
            Some(CoordinateSet::Box(b)) => {
                assert_eq!((&b.lo[..], &b.hi[..]), (&[pr, 0][..], &[2 * pr - 1, 3][..]))
            }
            other => panic!("rows={partition_rows}: expected Box, got {other:?}"),
        }
        assert_eq!(out.partitions[0].partition_origin, Some(vec![0, 0]));
        assert_eq!(out.partitions[1].partition_origin, Some(vec![pr, 0]));
    }
}

// ===========================================================================
// fast/slow-path fixture (port of test_distributed_tile_access_fast_path /
// _slow_path).
//
// 256x512 view, 4 row-band partitions of 64x512; access tile 32x128.
// Fixture: for each `indices`, the surviving partitions, each described by
// (C_i extent as half-open (lo, hi), expected partition_origin).
// ===========================================================================

const SHAPE: (usize, usize) = (256, 512);
const PARTITION_ROWS: [(i64, i64); 4] = [(0, 64), (64, 128), (128, 192), (192, 256)];
const ACCESS_SHAPE: (usize, usize) = (32, 128);

/// Fixture entries: (id, indices, [(C_i_lo, C_i_hi half-open, origin), ...]).
fn fixture() -> Vec<(&'static str, [i64; 2], Vec<([i64; 2], [i64; 2], [i64; 2])>)> {
    vec![
        ("single_partition", [10, 0], vec![([10, 0], [42, 128], [0, 0])]),
        (
            "cross_boundary",
            [50, 64],
            vec![([50, 64], [64, 192], [0, 0]), ([64, 64], [82, 192], [64, 0])],
        ),
        ("last_partition", [200, 256], vec![([200, 256], [232, 384], [192, 0])]),
        ("origin", [0, 0], vec![([0, 0], [32, 128], [0, 0])]),
    ]
}

/// Build 4 row-band partitions; `diagonal_mask` adds a non-axis-aligned
/// constraint that defeats `lower_to_box`, forcing the slow (Points) path while
/// leaving the box region's membership unchanged for these fixtures (the extra
/// `d0 + d1 >= 0` constraint is satisfied by every in-range coord).
fn build_partitions(diagonal_mask: bool) -> DistributedMemRef {
    let (_, ncols) = SHAPE;
    let mut parts = Vec::new();
    for (r0, r1) in PARTITION_ROWS {
        // inclusive box [r0, r1-1] x [0, ncols-1]
        let mut set = box_affine(&[r0, 0], &[r1 - 1, ncols as i64 - 1]);
        if diagonal_mask {
            // d0 + d1 >= 0 — always true for non-negative coords, but not
            // axis-aligned, so SymBoxSet::try_from_affine_set / lower_to_box bails.
            set.constraints.push(Constraint {
                expr: AffineExpr::Add(Box::new(AffineExpr::Dim(0)), Box::new(AffineExpr::Dim(1))),
                kind: ConstraintKind::GreaterEq,
            });
        }
        parts.push(MemRef {
            base_ptr: 0,
            shape: vec![(r1 - r0) as usize, ncols],
            strides: vec![ncols as i64, 1],
            space: MemorySpace::Hbm,
            dtype: DType::F16,
            coordinate_set: Some(set),
        });
    }
    DistributedMemRef::new(parts, vec![SHAPE.0, SHAPE.1], DType::F16).unwrap()
}

/// Run distributed_tile_access and collect, per survivor:
/// (sorted point list of C_i, partition_origin, whether C_i is a Box).
fn run_and_collect(
    dist: &DistributedMemRef,
    indices: [i64; 2],
) -> Vec<(Vec<Vec<i64>>, Vec<i64>, bool)> {
    let base_map = AffineMap::identity(2);
    let ac = [ACCESS_SHAPE.0, ACCESS_SHAPE.1];
    let out = distributed_tile_access(dist, &ac, &base_map, &indices, None).unwrap();
    out.partitions
        .into_iter()
        .map(|part| {
            let (pts, is_box) = match part.coordinate_set.as_ref().unwrap() {
                CoordinateSet::Box(b) => (enumerate_box(b), true),
                CoordinateSet::Points(p) => {
                    let mut p = p.clone();
                    p.sort();
                    (p, false)
                }
                CoordinateSet::Affine(_) => panic!("unexpected un-lowered AffineSet"),
            };
            (pts, part.partition_origin.unwrap(), is_box)
        })
        .collect()
}

/// Enumerate an inclusive `BoxSet` into a sorted row-major point list.
fn enumerate_box(b: &BoxSet) -> Vec<Vec<i64>> {
    let mut pts = Vec::new();
    for r in b.lo[0]..=b.hi[0] {
        for c in b.lo[1]..=b.hi[1] {
            pts.push(vec![r, c]);
        }
    }
    pts.sort();
    pts
}

/// Expand a half-open box `[lo, hi)` into a sorted row-major point list.
fn expected_points(lo: [i64; 2], hi: [i64; 2]) -> Vec<Vec<i64>> {
    let mut pts = Vec::new();
    for r in lo[0]..hi[0] {
        for c in lo[1]..hi[1] {
            pts.push(vec![r, c]);
        }
    }
    pts.sort();
    pts
}

#[test]
fn distributed_tile_access_fast_path_fixture() {
    for (case_id, indices, expected) in fixture() {
        let dist = build_partitions(false);
        let got = run_and_collect(&dist, indices);
        assert_eq!(got.len(), expected.len(), "{case_id}: partition count mismatch");
        for ((pts_got, origin_got, is_box), (exp_lo, exp_hi, exp_origin)) in
            got.iter().zip(expected.iter())
        {
            assert!(*is_box, "{case_id}: fast path must emit a Box");
            assert_eq!(origin_got, &exp_origin.to_vec(), "{case_id}: origin");
            assert_eq!(
                pts_got,
                &expected_points(*exp_lo, *exp_hi),
                "{case_id}: C_i mismatch at origin {origin_got:?}"
            );
        }
    }
}

#[test]
fn distributed_tile_access_slow_path_fixture() {
    for (case_id, indices, expected) in fixture() {
        // diagonal_mask=true defeats box lowering -> Points (slow) path.
        let dist = build_partitions(true);
        let got = run_and_collect(&dist, indices);
        assert_eq!(got.len(), expected.len(), "{case_id}: partition count mismatch");
        for ((pts_got, origin_got, is_box), (exp_lo, exp_hi, exp_origin)) in
            got.iter().zip(expected.iter())
        {
            assert!(!*is_box, "{case_id}: slow path must emit a point list");
            assert_eq!(origin_got, &exp_origin.to_vec(), "{case_id}: origin");
            assert_eq!(
                pts_got,
                &expected_points(*exp_lo, *exp_hi),
                "{case_id}: C_i mismatch at origin {origin_got:?}"
            );
        }
    }
}

// ===========================================================================
// distributed_store fast path: writes touch ONLY the C_i rectangle
// (port of test_distributed_store_does_not_trample_outside_C_i and the
//  column-packed variant)
// ===========================================================================

/// Build a 2-partition (16x16) view of 8x16 row-band partitions, seed each
/// whole partition with the sentinel, run a 4x4 distributed store at (2,4) into
/// P0 only, and return (P0 logical grid, P1 logical grid, payload).  The
/// `strides` choose row-major vs column-packed layout.
fn trample_setup(strides: [i64; 2]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let part_shape = (8usize, 16usize);
    let sentinel = -7.0f32;
    let mut ctx = single_core_context();

    // Seed both partitions fully with the sentinel.
    let sentinel_block: Vec<Vec<f32>> = vec![vec![sentinel; part_shape.1]; part_shape.0];
    let b0 = box_affine(&[0, 0], &[7, 15]);
    let b1 = box_affine(&[8, 0], &[15, 15]);
    let mr0 = seed_partition(&mut ctx, &sentinel_block, &strides, MemorySpace::Hbm, b0, 0);
    let mr1 = seed_partition(&mut ctx, &sentinel_block, &strides, MemorySpace::Hbm, b1, 0);
    let (p0_stick, p1_stick) = (mr0.base_ptr, mr1.base_ptr);
    let dist = DistributedMemRef::new(vec![mr0, mr1], vec![16, 16], DType::F16).unwrap();

    // 4x4 access at (2,4) — fully inside P0; C_i = [2,6)x[4,8).
    let base_map = AffineMap::identity(2);
    let resolved: DistributedTileRef =
        distributed_tile_access(&dist, &[4, 4], &base_map, &[2, 4], None).unwrap();
    assert_eq!(resolved.partitions.len(), 1, "P1 should be pruned");
    match resolved.partitions[0].coordinate_set.as_ref().unwrap() {
        CoordinateSet::Box(b) => {
            // Inclusive: Python [2,6)x[4,8) == [2,5]x[4,7].
            assert_eq!(b.lo, vec![2, 4]);
            assert_eq!(b.hi, vec![5, 7]);
        }
        other => panic!("expected Box C_0, got {other:?}"),
    }

    // Payload arange(1..17) reshaped 4x4.
    let payload: Vec<Vec<f32>> = (0..4)
        .map(|i| (0..4).map(|j| (1 + i * 4 + j) as f32).collect())
        .collect();
    let payload_flat: Vec<f32> = payload.clone().into_iter().flatten().collect();
    let tile = Tile::compute(payload_flat, DType::F16, vec![4, 4]);
    distributed_store(&mut ctx, &tile, &resolved).unwrap();

    // Reconstruct each partition's logical grid through its strides.
    let read_logical = |stick: i64| -> Vec<Vec<f32>> {
        let mut span = 1usize;
        for i in 0..part_shape.0 {
            for j in 0..part_shape.1 {
                let off = (i as i64) * strides[0] + (j as i64) * strides[1];
                span = span.max(off as usize + 1);
            }
        }
        let raw = ctx.hbm.borrow().read_bytes(stick * STICK_BYTES, span * 2);
        let flat = codec::decode(&raw, span, DType::F16);
        (0..part_shape.0)
            .map(|i| {
                (0..part_shape.1)
                    .map(|j| flat[((i as i64) * strides[0] + (j as i64) * strides[1]) as usize])
                    .collect()
            })
            .collect()
    };
    (read_logical(p0_stick), read_logical(p1_stick), payload)
}

/// Assert the store touched only C_i = rows 2..6, cols 4..8 of P0, with P1
/// untouched (sentinel everywhere).
fn assert_only_ci_written(p0: &[Vec<f32>], p1: &[Vec<f32>], payload: &[Vec<f32>], sentinel: f32) {
    for row in p1 {
        for &v in row {
            assert_eq!(v, sentinel, "P1 was trampled");
        }
    }
    for r in 0..8 {
        for c in 0..16 {
            let in_ci = (2..6).contains(&r) && (4..8).contains(&c);
            if in_ci {
                let want = payload[r - 2][c - 4];
                assert_eq!(p0[r][c], want, "C_i value wrong at ({r},{c})");
            } else {
                assert_eq!(p0[r][c], sentinel, "trampled P0 cell at ({r},{c})");
            }
        }
    }
}

#[test]
fn distributed_store_does_not_trample_outside_c_i_row_major() {
    let (p0, p1, payload) = trample_setup([16, 1]);
    assert_only_ci_written(&p0, &p1, &payload, -7.0);
}

#[test]
fn distributed_store_does_not_trample_outside_c_i_col_packed() {
    // strides=[1, NROWS=8] -> column-packed; the sub-tile must inherit these
    // verbatim rather than synthesise row-major sub-strides.
    let (p0, p1, payload) = trample_setup([1, 8]);
    assert_only_ci_written(&p0, &p1, &payload, -7.0);
}

// ===========================================================================
// RFC §C.3 reference example — per-core LX routing.
//
// Python marks this xfail (per-core LX gap) and loads an example .mlir file by
// path.  There is no faithful crate analogue at the ops layer here, so it is
// left as an ignored stub.
// ===========================================================================

#[test]
#[ignore = "RFC C.3 per-core LX routing example: xfail in Python; no ops-layer analogue"]
fn distributed_view_copy_rfc() {
    // Intentionally empty — see module/skipped notes.
}
