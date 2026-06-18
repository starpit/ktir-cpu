#include <metal_stdlib>
using namespace metal;

inline float simd_epilogue(float v, float ev, uint binop, uint act) {
    switch (binop) {
        case 1: v = v + ev; break;  case 2: v = v * ev; break;
        case 3: v = v - ev; break;  case 4: v = max(v, ev); break;
        case 5: v = min(v, ev); break;  default: break;
    }
    switch (act) {
        case 1: v = max(v, 0.0f); break;  case 2: v = tanh(v); break;
        case 3: v = exp(v); break;  case 4: v = 1.0f/(1.0f+exp(-v)); break;
        default: break;
    }
    return v;
}

[[kernel]] void matmul(
    device const float* a_in [[buffer(0)]],
    device const float* b_in [[buffer(1)]],
    device float* c_out      [[buffer(2)]],
    constant uint3& dims     [[buffer(3)]],
    device const float* e_in [[buffer(4)]],
    constant uint2& epi      [[buffer(5)]],
    uint3 tg  [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_simdgroup]])
{
    const uint M = dims.x, N = dims.y, K = dims.z;
    a_in  += tg.z * M * K;   // batch index (grid z): independent same-shape GEMM
    b_in  += tg.z * K * N;
    c_out += tg.z * M * N;
    e_in  += tg.z * M * N;
    const uint r0 = tg.y * 8u;   // output 8x8 tile base row
    const uint c0 = tg.x * 8u;   // base col
    threadgroup float a_tg[64];
    threadgroup float b_tg[64];
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += 8u) {
        for (uint i = lid; i < 64u; i += 32u) {
            uint r = i / 8u, c = i % 8u;
            uint gm = r0 + r, gkA = k0 + c;
            a_tg[i] = (gm < M && gkA < K) ? a_in[gm * K + gkA] : 0.0f;
            uint gkB = k0 + r, gn = c0 + c;
            b_tg[i] = (gkB < K && gn < N) ? b_in[KTIR_TRANSPOSE_B ? (gn * K + gkB) : (gkB * N + gn)] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 fa, fb;
        simdgroup_load(fa, a_tg, 8);
        simdgroup_load(fb, b_tg, 8);
        simdgroup_multiply_accumulate(acc, fa, fb, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float c_tg[64];
    simdgroup_store(acc, c_tg, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint binop = epi.x, act = epi.y;
    for (uint i = lid; i < 64u; i += 32u) {
        uint r = i / 8u, c = i % 8u;
        uint gm = r0 + r, gn = c0 + c;
        if (gm < M && gn < N) {
            float ev = (binop != 0u) ? e_in[gm * N + gn] : 0.0f;
            c_out[gm * N + gn] = simd_epilogue(c_tg[i], ev, binop, act);
        }
    }
}
