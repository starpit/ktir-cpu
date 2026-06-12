#!/usr/bin/env python3
"""MLX f16 fidelity probe — the gate for an MLX execution-backend pivot.

The ktir_cpu reference computes f16 tiles with numpy float16 (round-to-nearest-
even, per-op rounding). An MLX backend is only viable if MLX's f16 matches that
bit-for-bit. This probes exactly that, on the operations that matter:

  1. f16 round-trip: every f16 bit pattern, and a dense f32 sweep (rounding).
  2. elementwise chain with per-op f16 rounding (add/mul, the reduce combiners).
  3. reduction (sum) in f16 — the RMSNorm / softmax-denominator shape.
  4. matmul in f16 — the prefill-dominant op.

For each, compute via MLX (float16) and via numpy (float16) and diff the bits.
Green light = zero mismatches on (1)/(2); (3)/(4) are accumulation-order
sensitive, so we report max ULP/abs diff rather than demanding bit-equality.

Run:  uv run --with mlx --with numpy python mlx_fidelity_probe.py
"""
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    raise SystemExit("mlx not installed — run with: uv run --with mlx python mlx_fidelity_probe.py")


def bits16(a_np_f16):
    return a_np_f16.view(np.uint16)


def mlx_f16_roundtrip(x_f32):
    """f32 -> f16 -> f32 through MLX."""
    a = mx.array(x_f32, dtype=mx.float32).astype(mx.float16)
    return np.array(a.astype(mx.float32))


def np_f16_roundtrip(x_f32):
    return x_f32.astype(np.float16).astype(np.float32)


def probe_roundtrip_all_patterns():
    # Every f16 bit pattern -> f32, via the half bits both frameworks produce.
    patterns = np.arange(0, 1 << 16, dtype=np.uint16)
    as_f16 = patterns.view(np.float16)
    finite = np.isfinite(as_f16.astype(np.float32))
    x = as_f16.astype(np.float32)[finite]
    mlx_bits = bits16(mlx_f16_roundtrip(x).astype(np.float16))
    np_bits = bits16(np_f16_roundtrip(x).astype(np.float16))
    mism = int(np.sum(mlx_bits != np_bits))
    print(f"[1a] f16 round-trip, all finite patterns ({x.size}): {mism} bit mismatches")
    return mism == 0


def probe_roundtrip_rounding():
    # Dense f32 sweep across magnitudes — exercises real RNE rounding + overflow.
    rng = np.random.default_rng(0)
    x = np.concatenate([
        np.linspace(-70000, 70000, 2_000_00).astype(np.float32),
        rng.standard_normal(200_000).astype(np.float32) * 10.0,
        (rng.standard_normal(200_000).astype(np.float32) * 1e-4),  # subnormal-ish
    ])
    mlx_bits = bits16(mlx_f16_roundtrip(x).astype(np.float16))
    np_bits = bits16(np_f16_roundtrip(x).astype(np.float16))
    mism = int(np.sum(mlx_bits != np_bits))
    print(f"[1b] f32->f16 rounding, dense sweep ({x.size}): {mism} bit mismatches")
    if mism:
        i = np.argmax(mlx_bits != np_bits)
        print(f"     first mismatch: x={x[i]:.8g}  mlx=0x{mlx_bits[i]:04x}  np=0x{np_bits[i]:04x}")
    return mism == 0


def probe_elementwise_chain():
    # (a*b + c) with f16 rounding after each op — the per-op rounding the
    # interpreter does via Tile::compute / round_to_dtype.
    rng = np.random.default_rng(1)
    a = rng.standard_normal(100_000).astype(np.float16)
    b = rng.standard_normal(100_000).astype(np.float16)
    c = rng.standard_normal(100_000).astype(np.float16)

    # numpy: round (cast to f16) after each op
    np_mul = (a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)
    np_out = (np_mul.astype(np.float32) + c.astype(np.float32)).astype(np.float16)

    ma, mb, mc = (mx.array(t) for t in (a, b, c))
    mx_mul = (ma * mb).astype(mx.float16)
    mx_out = (mx_mul + mc).astype(mx.float16)
    mx_bits = bits16(np.array(mx_out))
    mism = int(np.sum(mx_bits != bits16(np_out)))
    print(f"[2] elementwise chain (a*b+c), per-op f16 round ({a.size}): {mism} bit mismatches")
    return mism == 0


def probe_reduction():
    # Sum over 576 (hidden dim), f16. Accumulation order differs, so report ULP.
    rng = np.random.default_rng(2)
    x = rng.standard_normal((256, 576)).astype(np.float16)
    np_sum = x.astype(np.float32).sum(axis=1).astype(np.float16)
    mx_sum = mx.array(x).sum(axis=1).astype(mx.float16)
    mx_sum = np.array(mx_sum)
    ulp = np.abs(bits16(mx_sum).astype(np.int32) - bits16(np_sum).astype(np.int32))
    print(f"[3] f16 reduce-sum over 576: max {int(ulp.max())} ULP, "
          f"{int(np.sum(ulp != 0))}/{ulp.size} differ (order-sensitive)")


def probe_matmul():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((64, 2048)).astype(np.float16)
    B = rng.standard_normal((2048, 512)).astype(np.float16)
    np_c = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
    mx_c = np.array((mx.array(A) @ mx.array(B)).astype(mx.float16))
    absd = np.abs(mx_c.astype(np.float32) - np_c.astype(np.float32))
    rel = absd / (np.abs(np_c.astype(np.float32)) + 1e-3)
    print(f"[4] f16 matmul 64x2048x512: max abs {absd.max():.4f}, max rel {rel.max():.4f} "
          f"(accumulation-order sensitive)")


if __name__ == "__main__":
    print(f"MLX default device: {mx.default_device()}\n")
    g1a = probe_roundtrip_all_patterns()
    g1b = probe_roundtrip_rounding()
    g2 = probe_elementwise_chain()
    probe_reduction()
    probe_matmul()
    print()
    if g1a and g1b and g2:
        print("VERDICT: MLX f16 is bit-identical to numpy f16 on round-trip + per-op "
              "rounding. Reduction/matmul differ only by accumulation order (expected, "
              "same as the current Metal GEMM path). GREEN LIGHT for fidelity-preserving "
              "fusion: lower elementwise/rounding to MLX, keep the tree-fold order where "
              "exact reductions matter.")
    else:
        print("VERDICT: f16 rounding diverges — MLX would need rounding pinned before use. "
              "Investigate the first-mismatch cases above.")
