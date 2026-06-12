// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Row-major single-precision GEMM, with an optional BLAS backend.
//!
//! `sgemm_rowmajor(m, k, n, a, b)` computes `C = A·B` for row-major `A` (m×k)
//! and `B` (k×n), returning a flat row-major `C` (m×n).
//!
//! - **macOS:** dispatches to Apple's `cblas_sgemm` (Accelerate, AMX-backed) by
//!   default — Accelerate ships with the OS, so it's on with no feature flag.
//! - **Linux / other:** a portable naive triple loop by default; enable a
//!   provider feature (`openblas-system` / `mkl` / `blis` / `openblas`) to route
//!   through that library's `cblas_sgemm` instead.
//!
//! BLAS is deterministic and — since NumPy's matmul is itself BLAS-backed —
//! tends to *tighten* parity with the reference. Both paths take the same
//! flat-`Vec<f32>` tile storage, so `linalg` matmul just calls `sgemm_rowmajor`.

/// `C(m×n) = A(m×k) · B(k×n)`, all row-major and contiguous. Naive loop —
/// the cross-platform default (and the parity oracle for the BLAS path).
#[cfg(not(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
)))]
pub fn sgemm_rowmajor(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    naive_sgemm(m, k, n, a, b)
}

/// `C(m×n) = A(m×k) · B(k×n)` via the linked BLAS `cblas_sgemm` (Accelerate on
/// macOS, or the selected provider — the cblas ABI is identical across them).
#[cfg(any(
    target_os = "macos",
    feature = "openblas",
    feature = "mkl",
    feature = "blis"
))]
pub fn sgemm_rowmajor(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemm};
    let mut c = vec![0.0f32; m * n];
    // SAFETY: a has m*k elements, b has k*n, c has m*n; leading dimensions match
    // the row-major contiguous layout (lda=k, ldb=n, ldc=n). All non-negative.
    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
    c
}

/// Portable reference GEMM — always available (also the oracle for the
/// accelerate path's parity test).
pub fn naive_sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_matches_known_product() {
        // [[1,2,3],[4,5,6]] · [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        assert_eq!(
            sgemm_rowmajor(2, 3, 2, &a, &b),
            vec![58.0, 64.0, 139.0, 154.0]
        );
    }

    /// With a BLAS backend active, `cblas_sgemm` must agree with the naive oracle.
    #[cfg(any(
        target_os = "macos",
        feature = "openblas",
        feature = "mkl",
        feature = "blis"
    ))]
    #[test]
    fn blas_matches_naive() {
        let m = 7;
        let k = 5;
        let n = 3;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 9) as f32 - 4.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 - 3.0).collect();
        assert_eq!(
            sgemm_rowmajor(m, k, n, &a, &b),
            naive_sgemm(m, k, n, &a, &b)
        );
    }
}
