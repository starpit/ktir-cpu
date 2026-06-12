// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Grid metadata and the handler-facing execution environment — port of the
//! `GridExecutor` coordinate logic from `ktir_cpu/grid.py` plus `ExecutionEnv`
//! from `dialects/registry.py`.
//!
//! `GridExecutor` here holds grid shape + coordinate transforms only; the
//! per-core `CoreContext`s and the comm scheduler live in the interpreter
//! driver, which keeps grid metadata and mutable core state un-aliased (avoiding
//! the borrow fight the design notes flagged). `ExecutionEnv` carries the
//! shared, read-only resources a handler needs: the dispatch table (to run
//! nested regions) and grid metadata.

use crate::dialects::Dispatch;

/// Grid shape and linear<->(x,y,z) transforms. Mirrors `GridExecutor`'s
/// `_linear_to_grid` / `_grid_to_linear`.
pub struct GridExecutor {
    pub grid_shape: (usize, usize, usize),
    pub num_cores: usize,
}

impl GridExecutor {
    pub fn new(grid_shape: (usize, usize, usize)) -> Self {
        let (nx, ny, nz) = grid_shape;
        GridExecutor {
            grid_shape,
            num_cores: nx * ny * nz,
        }
    }

    /// Linear core id -> (x, y, z). Mirrors `_linear_to_grid`.
    pub fn linear_to_grid(&self, core_id: usize) -> (usize, usize, usize) {
        let (nx, ny, _nz) = self.grid_shape;
        let z = core_id / (nx * ny);
        let rem = core_id % (nx * ny);
        (rem % nx, rem / nx, z)
    }

    /// (x, y, z) -> linear core id. Mirrors `_grid_to_linear`.
    pub fn grid_to_linear(&self, x: usize, y: usize, z: usize) -> usize {
        let (nx, ny, _nz) = self.grid_shape;
        z * (nx * ny) + y * nx + x
    }
}

/// Read-only resources passed to every handler. Mirrors `ExecutionEnv`.
/// Borrows the dispatch table (handlers run nested regions through it) and grid
/// metadata; mutates nothing.
pub struct ExecutionEnv<'a> {
    pub dispatch: &'a Dispatch,
    pub grid: &'a GridExecutor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_grid_roundtrip() {
        let g = GridExecutor::new((4, 2, 3));
        assert_eq!(g.num_cores, 24);
        for id in 0..g.num_cores {
            let (x, y, z) = g.linear_to_grid(id);
            assert_eq!(g.grid_to_linear(x, y, z), id);
        }
    }
}
