#!/usr/bin/env python3
"""Python-side timing harness for the Python-vs-Rust ktir_cpu comparison.

Times ONLY interp.execute_function (parse/load excluded, warm-up excluded),
matching the methodology of the Rust #[ignore]'d release benches.

Run from repo root:
    uv run python bench_py_vs_rust.py
"""
import time
import numpy as np

from ktir_cpu import KTIRInterpreter

ROOT = "/Users/moosevan/git/ktir-cpu"


def time_call(fn, iters, warmup=3):
    """Return mean seconds/run over `iters` timed calls, after `warmup` runs."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def bench_vector_add(iters):
    interp = KTIRInterpreter()
    interp.load(f"{ROOT}/examples/triton-ktir/vector_add_ktir.mlir")
    x_ptr, y_ptr, output_ptr, BLOCK_SIZE = interp.arg_names("add_kernel")
    n = 4096
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n).astype(np.float16)
    y = rng.standard_normal(n).astype(np.float16)

    def call():
        out = np.zeros(n, dtype=np.float16)
        interp.execute_function("add_kernel", **{
            x_ptr: x, y_ptr: y, output_ptr: out, BLOCK_SIZE: 128,
        })

    return time_call(call, iters), f"n={n} f16, grid[32]"


def bench_matmul(iters):
    interp = KTIRInterpreter()
    interp.load(f"{ROOT}/examples/triton-ktir/matmul_fwd_ktir.mlir")
    a_ptr, b_ptr, c_ptr, K, BM, BN, BK = interp.arg_names("matmul_kernel")
    M, N, K_val = 64, 8192, 2048
    rng = np.random.default_rng(42)
    A = rng.standard_normal((M, K_val)).astype(np.float16)
    B = rng.standard_normal((K_val, N)).astype(np.float16)

    def call():
        C = np.zeros((M, N), dtype=np.float16)
        interp.execute_function("matmul_kernel", **{
            a_ptr: A, b_ptr: B, c_ptr: C,
            K: K_val, BM: 32, BN: 512, BK: 128,
        })

    return time_call(call, iters), f"M=64,K=2048,N=8192 f16, grid[2,16]"


def bench_layernorm(iters):
    interp = KTIRInterpreter()
    interp.load(f"{ROOT}/examples/triton-ktir/layernorm_fwd_ktir.mlir")
    X, Y, W, B, Mean, Rstd, N, eps, BLOCK_SIZE = interp.arg_names("_layer_norm_fwd_fused")
    rows, cols = 1151, 8192
    rng = np.random.default_rng(42)
    X_data = rng.standard_normal((rows, cols)).astype(np.float16)
    W_data = np.ones((rows, cols), dtype=np.float16)
    B_data = np.zeros((rows, cols), dtype=np.float16)

    def call():
        Y_data = np.zeros((rows, cols), dtype=np.float16)
        Mean_data = np.zeros(rows, dtype=np.float16)
        Rstd_data = np.zeros(rows, dtype=np.float16)
        interp.execute_function("_layer_norm_fwd_fused", **{
            X: X_data, Y: Y_data, W: W_data, B: B_data,
            Mean: Mean_data, Rstd: Rstd_data,
            N: 8192, eps: np.float16(1e-5), BLOCK_SIZE: 1024,
        })

    return time_call(call, iters), f"1151x8192 f16, grid[32]"


if __name__ == "__main__":
    # Iteration counts mirror the Rust benches where reasonable; heavier
    # kernels use fewer iters to keep total wall time sane.
    benches = [
        ("vector_add", bench_vector_add, 500),
        ("matmul", bench_matmul, 20),
        ("layernorm", bench_layernorm, 20),
    ]
    print(f"{'kernel':<12} {'iters':>6} {'µs/run':>14}   size")
    for name, fn, iters in benches:
        secs, size = fn(iters)
        print(f"{name:<12} {iters:>6} {secs * 1e6:>14.1f}   {size}")
