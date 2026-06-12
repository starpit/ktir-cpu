// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Affine maps and sets — Rust port of `ktir_cpu/affine.py`.
//!
//! `AffineMap` is a pure function over dimension + symbol values (frozen
//! dataclass in Python -> immutable value type here). `BoxSet` is the
//! axis-aligned fast path with O(ndim) containment; `AffineSet` is the general
//! constraint-based set. Only the pieces the arith slice exercises are filled
//! in; the constraint solver in `AffineSet` is stubbed where noted.

/// Recursive affine-expression AST: `Dim`, `Sym`, `Const`, and the operators
/// MLIR affine exprs support. Box-recursive — the standard shape for this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AffineExpr {
    Dim(usize),
    Sym(usize),
    Const(i64),
    Add(Box<AffineExpr>, Box<AffineExpr>),
    Mul(Box<AffineExpr>, Box<AffineExpr>),
    FloorDiv(Box<AffineExpr>, Box<AffineExpr>),
    Mod(Box<AffineExpr>, Box<AffineExpr>),
}

impl AffineExpr {
    /// Evaluate against concrete dimension and symbol values.
    pub fn eval(&self, dims: &[i64], syms: &[i64]) -> i64 {
        match self {
            AffineExpr::Dim(i) => dims[*i],
            AffineExpr::Sym(i) => syms[*i],
            AffineExpr::Const(c) => *c,
            AffineExpr::Add(a, b) => a.eval(dims, syms) + b.eval(dims, syms),
            AffineExpr::Mul(a, b) => a.eval(dims, syms) * b.eval(dims, syms),
            // MLIR affine floordiv/mod are Euclidean (floor toward -inf).
            AffineExpr::FloorDiv(a, b) => a.eval(dims, syms).div_euclid(b.eval(dims, syms)),
            AffineExpr::Mod(a, b) => a.eval(dims, syms).rem_euclid(b.eval(dims, syms)),
        }
    }
}

/// A pure multi-result affine map: `(d0, d1)[s0] -> (expr, expr, ...)`.
/// Frozen/immutable, like the Python `@dataclass(frozen=True)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineMap {
    pub num_dims: usize,
    pub num_syms: usize,
    pub exprs: Vec<AffineExpr>,
}

impl AffineMap {
    /// Identity map of rank `n` — synthesized when MLIR omits `base_map`,
    /// matching `AccessTile.base_map` ("synthesized as identity if absent").
    pub fn identity(n: usize) -> Self {
        AffineMap {
            num_dims: n,
            num_syms: 0,
            exprs: (0..n).map(AffineExpr::Dim).collect(),
        }
    }

    /// Evaluate every result expression. Mirrors `AffineMap.eval`.
    pub fn eval(&self, dims: &[i64], syms: &[i64]) -> Vec<i64> {
        debug_assert_eq!(dims.len(), self.num_dims, "dim arity mismatch");
        debug_assert_eq!(syms.len(), self.num_syms, "sym arity mismatch");
        self.exprs.iter().map(|e| e.eval(dims, syms)).collect()
    }
}

/// Axis-aligned integer box `[lo, hi]` inclusive — the fast path of
/// `CoordinateSet`. O(ndim) containment and intersection, like Python `BoxSet`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoxSet {
    pub lo: Vec<i64>,
    pub hi: Vec<i64>,
}

impl BoxSet {
    pub fn new(lo: Vec<i64>, hi: Vec<i64>) -> Self {
        assert_eq!(lo.len(), hi.len(), "BoxSet lo/hi rank mismatch");
        BoxSet { lo, hi }
    }

    /// `min(coordinate_set)` = lower corner; used as the partition origin
    /// (`p_i`) in `distributed_tile_access`.
    pub fn origin(&self) -> &[i64] {
        &self.lo
    }

    pub fn contains(&self, point: &[i64]) -> bool {
        point.len() == self.lo.len()
            && point
                .iter()
                .zip(&self.lo)
                .zip(&self.hi)
                .all(|((&p, &lo), &hi)| lo <= p && p <= hi)
    }

    /// Per-axis intersection; `None` if the boxes are disjoint on any axis.
    /// Mirrors `BoxSet.intersect` (which returns an empty box on no overlap).
    pub fn intersect(&self, other: &BoxSet) -> Option<BoxSet> {
        assert_eq!(self.lo.len(), other.lo.len(), "BoxSet rank mismatch");
        let mut lo = Vec::with_capacity(self.lo.len());
        let mut hi = Vec::with_capacity(self.hi.len());
        for i in 0..self.lo.len() {
            let l = self.lo[i].max(other.lo[i]);
            let h = self.hi[i].min(other.hi[i]);
            if l > h {
                return None;
            }
            lo.push(l);
            hi.push(h);
        }
        Some(BoxSet { lo, hi })
    }
}

/// General affine set: a conjunction of affine constraints `expr >= 0` or
/// `expr == 0`. Containment substitutes the point and checks every constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AffineSet {
    pub num_dims: usize,
    pub num_syms: usize,
    pub constraints: Vec<Constraint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    pub expr: AffineExpr,
    pub kind: ConstraintKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintKind {
    /// `expr >= 0`
    GreaterEq,
    /// `expr == 0`
    Equal,
}

impl AffineSet {
    /// Point membership: every constraint must hold. Mirrors `AffineSet.contains`.
    pub fn contains(&self, point: &[i64], syms: &[i64]) -> bool {
        self.constraints.iter().all(|c| {
            let v = c.expr.eval(point, syms);
            match c.kind {
                ConstraintKind::GreaterEq => v >= 0,
                ConstraintKind::Equal => v == 0,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_eval_and_identity() {
        // (d0, d1)[s0] -> (d0 + s0, d1 * 2)
        let m = AffineMap {
            num_dims: 2,
            num_syms: 1,
            exprs: vec![
                AffineExpr::Add(
                    Box::new(AffineExpr::Dim(0)),
                    Box::new(AffineExpr::Sym(0)),
                ),
                AffineExpr::Mul(
                    Box::new(AffineExpr::Dim(1)),
                    Box::new(AffineExpr::Const(2)),
                ),
            ],
        };
        assert_eq!(m.eval(&[5, 7], &[10]), vec![15, 14]);
        assert_eq!(AffineMap::identity(3).eval(&[1, 2, 3], &[]), vec![1, 2, 3]);
    }

    #[test]
    fn euclidean_floordiv_and_mod() {
        let fd = AffineExpr::FloorDiv(
            Box::new(AffineExpr::Dim(0)),
            Box::new(AffineExpr::Const(4)),
        );
        let m = AffineExpr::Mod(Box::new(AffineExpr::Dim(0)), Box::new(AffineExpr::Const(4)));
        // -1 floordiv 4 == -1, -1 mod 4 == 3 (matches MLIR / Python semantics)
        assert_eq!(fd.eval(&[-1], &[]), -1);
        assert_eq!(m.eval(&[-1], &[]), 3);
    }

    #[test]
    fn box_contains_and_intersect() {
        let a = BoxSet::new(vec![0, 0], vec![3, 3]);
        let b = BoxSet::new(vec![2, 2], vec![5, 5]);
        assert!(a.contains(&[1, 2]));
        assert!(!a.contains(&[4, 0]));
        assert_eq!(a.origin(), &[0, 0]);
        assert_eq!(a.intersect(&b), Some(BoxSet::new(vec![2, 2], vec![3, 3])));

        let disjoint = BoxSet::new(vec![10, 10], vec![11, 11]);
        assert_eq!(a.intersect(&disjoint), None);
    }

    #[test]
    fn affine_set_membership() {
        // { (d0) : d0 >= 0, 7 - d0 >= 0 }  ==  0 <= d0 <= 7
        let set = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::Dim(0),
                    kind: ConstraintKind::GreaterEq,
                },
                Constraint {
                    expr: AffineExpr::Add(
                        Box::new(AffineExpr::Const(7)),
                        Box::new(AffineExpr::Mul(
                            Box::new(AffineExpr::Const(-1)),
                            Box::new(AffineExpr::Dim(0)),
                        )),
                    ),
                    kind: ConstraintKind::GreaterEq,
                },
            ],
        };
        assert!(set.contains(&[0], &[]));
        assert!(set.contains(&[7], &[]));
        assert!(!set.contains(&[8], &[]));
        assert!(!set.contains(&[-1], &[]));
    }
}
