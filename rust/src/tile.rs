// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Tile data value — minimal port of `Tile` from `ktir_cpu/ir_types.py`.
//!
//! SLICE-1 STORAGE DECISION: data is a flat `Vec<f32>` + shape. The production
//! port should swap this for `ndarray::ArrayD` behind the same API, and decide
//! the dtype question (see below). f16 tiles are widened to f32 here; the real
//! port pulls in `half::f16` at the load/store boundary.
//!
//! THE DTYPE FORK (settle before the dialect ops balloon):
//!   A) one `ArrayD<f32>` storage, `dtype: DType` tag, reinterpret on store.
//!      Simple; every compute handler is written once. Loses exact integer /
//!      f16-rounding fidelity unless handled explicitly.
//!   B) a `TileData` enum (`F16(ArrayD<f16>)`, `I32(ArrayD<i32>)`, ...).
//!      Faithful to NumPy per-dtype behavior, but multiplies every handler by
//!      N variants — tame with a macro over dtypes.
//! Recommendation: start with (A) + an explicit `cast` at load/store, escalate
//! individual ops to (B) only where integer/f16 semantics actually bite. The
//! `unique_sticks` bookkeeping below is storage-independent either way.

use crate::dtypes::DType;
use std::rc::Rc;

/// A tensor of element data — `load` result / compute-op operand.
///
/// `data` is an `Rc<[f32]>`: tiles are immutable once built (every op produces a
/// fresh result via [`Tile::compute`]), so cloning a Tile — which the
/// interpreter does constantly (binding op results, threading scf iter_args,
/// passing operands) — is a refcount bump, not a deep copy of the element data.
/// The two spots that fold into an existing result (`linalg.matmul`/`batch_matmul`
/// accumulate) own their tile uniquely and reach through `Rc::make_mut`, which is
/// copy-free at refcount 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    pub data: Rc<[f32]>,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Distinct HBM sticks touched by the load that produced this tile.
    /// `None` for compute-produced tiles (mirrors the Python field).
    pub unique_sticks: Option<usize>,
    /// Distinct sticks touched by index-tensor reads during an indirect
    /// load/store. `None` for direct loads and compute-produced tiles.
    pub index_unique_sticks: Option<usize>,
}

impl Tile {
    /// Construct a compute-produced tile (no stick bookkeeping).
    ///
    /// The data is rounded to `dtype`'s representable set — exactly what a NumPy
    /// `np.float16`/`np.int32`/... array does on assignment. Because every op
    /// handler builds its result through this constructor, this gives per-op
    /// rounding for free: an f16 compute *chain* rounds after each step the way
    /// NumPy does, rather than accumulating in f32 and rounding only at store.
    pub fn compute(mut data: Vec<f32>, dtype: DType, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "tile data length must equal product of shape"
        );
        crate::codec::round_to_dtype(&mut data, dtype);
        Tile {
            data: data.into(),
            dtype,
            shape,
            unique_sticks: None,
            index_unique_sticks: None,
        }
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len() * self.dtype.bytes_per_elem()
    }
}
