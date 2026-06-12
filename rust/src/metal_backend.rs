// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Experimental KTIR -> Metal Shading Language backend (`--features metal`).
//!
//! This is the *codegen* half of a GPU execution backend: it lowers a KTIR
//! `IRModule` to an MSL kernel string. It is pure string emission — no GPU, no
//! Metal bindings — so it builds and unit-tests anywhere, and the emitted shader
//! can be diffed/inspected directly. The runtime half (compile the MSL, dispatch
//! on a `MTLDevice`, read back, and validate against `interpreter::execute_function`
//! as the golden oracle) is a later slice that needs a real Metal device.
//!
//! SCOPE (slice 1): the per-tile **element-wise** pattern — the shape of a
//! Triton-style `vector_add`: load N tiles, apply one element-wise compute op,
//! store the result. Each GPU thread handles one element; the Spyre grid +
//! BLOCK_SIZE collapse into a flat `thread_position_in_grid`. This is the GPU
//! "hello world" and proves the IR->MSL pipeline end to end.
//!
//! NOT YET (the roadmap): multi-op fusion, `linalg.matmul` (-> MPS), reductions
//! (threadgroup memory), cross-core comm (the global-sync problem), distributed
//! and indirect access. Each is its own slice.

use std::collections::HashMap;

use crate::dtypes::DType;
use crate::ir::{IRFunction, IRModule, Operation};

// =========================================================================
// Matmul acceleration tier — the GPU analogue of the BLAS auto-select.
//
// On Apple Silicon a GEMM can run three ways, best-first:
//   * Nax       — `mpp::tensor_ops::matmul2d` (Metal Performance Primitives),
//                 which drives the M5+ Neural Accelerators. NOT engaged
//                 automatically by MPS — it must be written in the shader.
//   * Simdgroup — `simdgroup_matrix<T,8,8>` + `simdgroup_multiply_accumulate`,
//                 the matrix instructions on Apple7+ (M1..M4) GPUs.
//   * Naive     — a plain per-element loop. The portable floor (also non-Apple).
//
// We pick the highest tier the device supports (capability), capped by the
// highest tier whose kernel we actually emit today (`HIGHEST_IMPLEMENTED`), so
// the backend degrades gracefully as the accelerated kernel slices land —
// exactly like the BLAS providers degrade to the naive matmul.
// =========================================================================

/// GPU matmul acceleration tier, ordered worst -> best.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatmulTier {
    Naive,
    Simdgroup,
    Nax,
}

/// The highest tier whose kernel we actually emit today. The general tiled NAX
/// GEMM ([`NaxGemm`] / [`run_nax_matmul`], `mpp::tensor_ops::matmul2d`) is
/// implemented and validated on the M5, so this is `Nax`. Note the tiers are
/// NOT all implemented in order — `Simdgroup` has no kernel yet — so
/// [`effective_matmul_tier`] selects the best *implemented* tier a device
/// supports rather than a simple linear cap (see [`tier_implemented`]).
pub const HIGHEST_IMPLEMENTED: MatmulTier = MatmulTier::Nax;

/// Whether a tier's GPU kernel is emitted today. `Naive` (the CPU/BLAS floor)
/// and `Nax` (the M5 tensor engine) are implemented; the pre-NAX `Simdgroup`
/// (`simdgroup_matrix`) kernel is a future slice, so a non-NAX Apple GPU falls
/// back to `Naive` rather than claiming a tier we can't run.
pub fn tier_implemented(tier: MatmulTier) -> bool {
    matches!(tier, MatmulTier::Naive | MatmulTier::Nax)
}

/// The matmul tier a Metal device *supports*, parsed from its name (mirrors
/// scratchy's `detect_device` name-parse → `AppleSiliconGen` → `is_nax_capable`):
///   * Apple `M5`+  -> Nax (Apple9 gen 17+, first with the Neural Accelerator)
///   * any other Apple GPU (M1..M4, Apple7+) -> Simdgroup
///   * non-Apple / unknown -> Naive
pub fn device_matmul_tier(device_name: &str) -> MatmulTier {
    if let Some(generation) = apple_m_generation(device_name) {
        return if generation >= 5 { MatmulTier::Nax } else { MatmulTier::Simdgroup };
    }
    if device_name.contains("Apple") {
        // An Apple GPU we couldn't pin to an M-number — assume Apple7+ matrix units.
        return MatmulTier::Simdgroup;
    }
    MatmulTier::Naive
}

/// The tier actually used for a device: the highest *implemented* tier that
/// the device's capability supports. Because `Simdgroup` isn't implemented yet,
/// a pre-NAX Apple GPU (capability `Simdgroup`) resolves to `Naive`, while an
/// M5 (capability `Nax`) resolves to `Nax`.
pub fn effective_matmul_tier(device_name: &str) -> MatmulTier {
    let cap = device_matmul_tier(device_name);
    [MatmulTier::Nax, MatmulTier::Simdgroup, MatmulTier::Naive]
        .into_iter()
        .find(|&t| t <= cap && tier_implemented(t))
        .unwrap_or(MatmulTier::Naive)
}

/// Parse the `M<n>` generation from an Apple GPU name like `"Apple M5 Pro"`.
/// Returns `None` for non-Apple-Silicon names. Forward-compatible: an `M6`
/// reads as 6 (>= 5 -> Nax), unlike scratchy's fixed M1..M5 match.
fn apple_m_generation(name: &str) -> Option<u32> {
    let rest = name.split('M').nth(1)?; // text after the first 'M'
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    (!digits.is_empty() && name.contains("Apple"))
        .then(|| digits.parse().ok())
        .flatten()
}

/// One kernel-argument buffer: the KTIR pointer-arg name, whether it's written,
/// and its element dtype. The order of [`MslKernel::buffers`] is the MSL
/// `[[buffer(i)]]` binding order — the runtime must supply data in this order.
#[derive(Clone, Debug, PartialEq)]
pub struct BufferBinding {
    pub name: String,
    pub is_output: bool,
    pub dtype: DType,
}

/// A lowered Metal kernel: the MSL source, the kernel name, and its buffer
/// bindings in `[[buffer(i)]]` order.
#[derive(Clone, Debug)]
pub struct MslKernel {
    pub source: String,
    pub name: String,
    pub buffers: Vec<BufferBinding>,
}

/// Lower `func_name` to an MSL kernel string. Errors if the function isn't the
/// supported element-wise shape (with a message pointing at what tripped it).
pub fn emit_msl(module: &IRModule, func_name: &str) -> Result<String, String> {
    Ok(emit_kernel(module, func_name)?.source)
}

/// Lower `func_name` to a full [`MslKernel`] (source + buffer bindings).
pub fn emit_kernel(module: &IRModule, func_name: &str) -> Result<MslKernel, String> {
    let f = module.get_function(func_name)?;
    let defs = def_map(f);

    // The kernel's "root" is its single store: `ktdp.store %value, %access_tile`.
    let store = f
        .operations
        .iter()
        .find(|o| o.op_type == "ktdp.store")
        .ok_or("metal: no ktdp.store — only element-wise store kernels are supported in slice 1")?;
    if store.operands.len() < 2 {
        return Err("metal: ktdp.store needs (value, access_tile) operands".into());
    }
    let out_buf = trace_buffer(&store.operands[1], &defs)
        .ok_or("metal: could not trace the store target back to a pointer argument")?;

    // The stored value must come from a single element-wise compute op whose
    // operands are loaded tiles.
    let compute = defs
        .get(strip(&store.operands[0]))
        .ok_or("metal: stored value has no defining op")?;
    let expr = lower_compute(compute, &defs)?;
    let dtype = buffer_dtype(&out_buf, f);

    // Inputs in first-seen order; the output buffer last. (De-dup: a buffer may
    // be both read and written, though vector_add's aren't.)
    let mut buffers: Vec<BufferBinding> = Vec::new();
    for b in collect_input_buffers(compute, &defs) {
        if !buffers.iter().any(|x| x.name == b) {
            let bdt = buffer_dtype(&b, f);
            buffers.push(BufferBinding { name: b, is_output: false, dtype: bdt });
        }
    }
    buffers.push(BufferBinding { name: out_buf, is_output: true, dtype });

    let source = render_kernel(func_name, &buffers, &expr);
    Ok(MslKernel { source, name: func_name.to_string(), buffers })
}

// --- dataflow ------------------------------------------------------------

/// `result-name (no %) -> defining op`.
fn def_map(f: &IRFunction) -> HashMap<String, &Operation> {
    let mut m = HashMap::new();
    for op in &f.operations {
        if let Some(r) = &op.result {
            m.insert(strip(r).to_string(), op);
        }
    }
    m
}

fn strip(name: &str) -> &str {
    name.trim_start_matches('%')
}

/// Follow an SSA value back to the pointer-argument buffer it ultimately reads
/// or writes: `load`/`store` access tile -> `construct_access_tile` -> its view
/// -> `construct_memory_view` -> the `%ptr` argument. Returns the arg name.
fn trace_buffer(name: &str, defs: &HashMap<String, &Operation>) -> Option<String> {
    let mut cur = strip(name).to_string();
    // Walk defining ops until we hit a name with no def (a function argument).
    for _ in 0..16 {
        let Some(op) = defs.get(cur.as_str()) else {
            return Some(cur); // no def -> it's a function argument (the pointer)
        };
        // Each of these ops carries the thing-we-want as operand 0.
        match op.op_type.as_str() {
            "ktdp.construct_access_tile" | "ktdp.construct_memory_view" | "ktdp.load" => {
                cur = strip(&op.operands[0]).to_string();
            }
            // Any other defining op isn't part of a load/store->buffer chain.
            _ => return None,
        }
    }
    None
}

/// Buffers feeding a compute op's tile operands (each operand is a load).
fn collect_input_buffers(compute: &Operation, defs: &HashMap<String, &Operation>) -> Vec<String> {
    compute
        .operands
        .iter()
        .filter_map(|o| trace_buffer(o, defs))
        .collect()
}

/// Element dtype of a buffer, read from the `construct_memory_view` that
/// produced it. Defaults to `f16` — the common KTIR tile dtype.
fn buffer_dtype(buf: &str, f: &IRFunction) -> DType {
    for op in &f.operations {
        if op.op_type == "ktdp.construct_memory_view"
            && op.operands.first().map(|p| strip(p)) == Some(buf)
            && let Some(crate::ir::Attr::Str(dt)) = op.attributes.get("dtype")
            && let Ok(parsed) = DType::parse(dt)
        {
            return parsed;
        }
    }
    DType::F16
}

/// The MSL scalar type for a KTIR dtype.
fn msl_type(dt: DType) -> &'static str {
    match dt {
        DType::F16 => "half",
        DType::F32 => "float",
        DType::I32 => "int",
        DType::I64 => "long",
        DType::Bool => "bool",
    }
}

// --- compute lowering ----------------------------------------------------

/// Lower a single element-wise compute op into an MSL expression over `gid`.
/// Operands resolve to `<buffer>[gid]`.
fn lower_compute(op: &Operation, defs: &HashMap<String, &Operation>) -> Result<String, String> {
    let operand = |i: usize| -> Result<String, String> {
        let name = op
            .operands
            .get(i)
            .ok_or_else(|| format!("metal: {} missing operand {i}", op.op_type))?;
        let buf = trace_buffer(name, defs)
            .ok_or_else(|| format!("metal: operand {name} is not a loaded buffer"))?;
        Ok(format!("{buf}[gid]"))
    };

    // Binary element-wise float ops -> infix operator.
    let binop = |sym: &str| -> Result<String, String> {
        Ok(format!("{} {} {}", operand(0)?, sym, operand(1)?))
    };
    // Unary math ops -> MSL intrinsic call.
    let unary = |func: &str| -> Result<String, String> { Ok(format!("{func}({})", operand(0)?)) };

    match op.op_type.as_str() {
        "arith.addf" => binop("+"),
        "arith.subf" => binop("-"),
        "arith.mulf" => binop("*"),
        "arith.divf" => binop("/"),
        "arith.maximumf" | "arith.maxf" => Ok(format!("max({}, {})", operand(0)?, operand(1)?)),
        "arith.minimumf" | "arith.minf" => Ok(format!("min({}, {})", operand(0)?, operand(1)?)),
        "arith.negf" => Ok(format!("-{}", operand(0)?)),
        "arith.absf" | "math.absf" => unary("abs"),
        "math.exp" => unary("exp"),
        "math.log" => unary("log"),
        "math.sqrt" => unary("sqrt"),
        "math.sin" => unary("sin"),
        "math.cos" => unary("cos"),
        "math.tanh" => unary("tanh"),
        "linalg.add" => binop("+"),
        "linalg.mul" => binop("*"),
        "linalg.sub" => binop("-"),
        other => Err(format!(
            "metal: compute op {other:?} not lowerable in slice 1 (element-wise only)"
        )),
    }
}

// --- rendering -----------------------------------------------------------

fn render_kernel(name: &str, buffers: &[BufferBinding], expr: &str) -> String {
    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
    s.push_str(&format!("kernel void {name}(\n"));
    for (i, b) in buffers.iter().enumerate() {
        let qual = if b.is_output { "device" } else { "device const" };
        s.push_str(&format!(
            "    {qual} {}* {} [[buffer({i})]],\n",
            msl_type(b.dtype),
            b.name
        ));
    }
    s.push_str("    uint gid [[thread_position_in_grid]]\n) {\n");
    // The output buffer is the last entry.
    let out = &buffers.last().unwrap().name;
    s.push_str(&format!("    {out}[gid] = {expr};\n"));
    s.push_str("}\n");
    s
}

// =========================================================================
// Runtime dispatch (slice 2) — compile the MSL and run it on a Metal device.
// =========================================================================

/// Compile `kernel`'s MSL, upload `inputs` (in `kernel.buffers` non-output
/// order, as f32 — encoded to each buffer's dtype), dispatch one thread per
/// output element, and read `out_len` elements back as f32.
///
/// Returns `Err("no Metal device …")` when no GPU is available (e.g. headless
/// CI), so callers can skip gracefully.
pub fn run_kernel(
    kernel: &MslKernel,
    inputs: &[Vec<f32>],
    out_len: usize,
) -> Result<Vec<f32>, String> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
        MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
    let opts = objc2_metal::MTLCompileOptions::new();
    let src = NSString::from_str(&kernel.source);
    let library = device
        .newLibraryWithSource_options_error(&src, Some(&opts))
        .map_err(|e| format!("metal: MSL compile failed: {e:?}"))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(&kernel.name))
        .ok_or_else(|| format!("metal: kernel {:?} not found", kernel.name))?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("metal: pipeline build failed: {e:?}"))?;
    let queue = device.newCommandQueue().ok_or("metal: newCommandQueue returned nil")?;

    let res = MTLResourceOptions::StorageModeShared;
    let mut gpu_buffers = Vec::with_capacity(kernel.buffers.len());
    let mut input_iter = inputs.iter();
    let mut out_dtype = DType::F16;
    for b in &kernel.buffers {
        let buf = if b.is_output {
            out_dtype = b.dtype;
            let len = (out_len * b.dtype.bytes_per_elem()).max(1);
            device
                .newBufferWithLength_options(len, res)
                .ok_or("metal: output buffer alloc failed")?
        } else {
            let data = input_iter.next().ok_or("metal: too few inputs for kernel buffers")?;
            let bytes = crate::codec::encode(data, b.dtype);
            // SAFETY: `bytes` lives until the copy completes inside this call.
            unsafe {
                device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len().max(1),
                        res,
                    )
                    .ok_or("metal: input buffer alloc failed")?
            }
        };
        gpu_buffers.push(buf);
    }

    let cb = queue.commandBuffer().ok_or("metal: commandBuffer returned nil")?;
    let enc = cb.computeCommandEncoder().ok_or("metal: computeCommandEncoder returned nil")?;
    enc.setComputePipelineState(&pipeline);
    for (i, buf) in gpu_buffers.iter().enumerate() {
        unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
    }
    let tg = pipeline.maxTotalThreadsPerThreadgroup().min(out_len).max(1);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: out_len, height: 1, depth: 1 },
        MTLSize { width: tg, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    // Read the output buffer (last) back and decode to f32.
    let out = gpu_buffers.last().unwrap();
    let nbytes = out_len * out_dtype.bytes_per_elem();
    let raw = unsafe {
        std::slice::from_raw_parts(out.contents().as_ptr() as *const u8, nbytes)
    }
    .to_vec();
    Ok(crate::codec::decode(&raw, out_len, out_dtype))
}

/// Compile MSL source as **Metal 4** (`MTLLanguageVersion::Version4_0`,
/// `MathMode::Safe`) — the options Metal Performance Primitives (`mpp::tensor_ops`,
/// the M5 NAX path) require. Compiled from source at runtime because the offline
/// `xcrun metal` toolchain miscompiles MPP (per scratchy's findings). Returns
/// `Ok(())` if the source compiles on the system device, else the compiler error.
pub fn compile_metal4(source: &str) -> Result<(), String> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLMathMode,
    };

    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
    let opts = objc2_metal::MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe);
    opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&opts))
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

// =========================================================================
// NAX (M5 Neural Accelerator) single-tile GEMM
// =========================================================================
//
// The M5's matmul tier. One simdgroup computes a fixed 16×32×16 output tile
// with `mpp::tensor_ops::matmul2d` — the Metal Performance Primitives op that
// dispatches to the NAX tensor engine. This is the irreducible NAX unit; a
// general GEMM tiles the problem into these (a later slice). It exists now to
// prove the engine produces correct results through our runtime and to measure
// the speedup, gating whether `HIGHEST_IMPLEMENTED` can rise to `Nax`.
//
// Inputs/outputs are host `f32` (row-major); A and B are converted to `bfloat`
// in threadgroup memory inside the shader, so the host never touches bf16. The
// op runs with `transpose_b`, so B (logical K×N) is consumed as its transpose
// Bᵀ (N×K) — the fill loop transposes while converting.

/// Fixed NAX tile dims: `C[M×N] = A[M×K] · B[K×N]`.
pub const NAX_TILE_M: usize = 16;
pub const NAX_TILE_N: usize = 32;
pub const NAX_TILE_K: usize = 16;

/// MSL for the single-tile NAX GEMM (pure MPP, no external headers): cooperative
/// fill of threadgroup A/B → load into `matmul2d` register cooperative tensors
/// via the BaseNAXFrag lane layout → `run` → store with the same layout. Mirrors
/// scratchy's proven register-fragment `mma` (the metal::tensor `run` overload
/// has a different, unvalidated output layout — see the kernel body).
const NAX_MATMUL_TILE_SRC: &str = "\
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_tensor>
using namespace metal;

// C[16x32] = A[16x16] . B[16x32], all row-major device float.
[[kernel]] void nax_matmul_tile(
    device const float* a_in [[buffer(0)]],   // M x K = 16 x 16
    device const float* b_in [[buffer(1)]],   // K x N = 16 x 32
    device float* c_out      [[buffer(2)]],   // M x N = 16 x 32
    uint lid [[thread_index_in_simdgroup]])
{
    threadgroup bfloat a_tg[16 * 16];   // [M, K] row-major
    threadgroup bfloat b_tg[32 * 16];   // [N, K] = transpose(B), row-major
    // Cooperative fill across the 32 simdgroup lanes.
    for (uint i = lid; i < 16u * 16u; i += 32u) {
        a_tg[i] = bfloat(a_in[i]);                  // A[m,k] at m*16+k
    }
    for (uint i = lid; i < 32u * 16u; i += 32u) {
        uint n = i / 16u;                           // 0..31
        uint k = i % 16u;                           // 0..15
        b_tg[n * 16u + k] = bfloat(b_in[k * 32u + n]);   // Bt[n,k] = B[k,n]
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        /*transpose_a=*/false, /*transpose_b=*/true, /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Register-fragment path (scratchy's proven `mma`): load A and B into the
    // input cooperative tensors via the validated BaseNAXFrag lane layout, run,
    // and store the destination with the SAME layout — internally consistent,
    // unlike the metal::tensor `run` overload whose output layout differs.
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto ct_c = gemm_op.template
        get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();

    // BaseNAXFrag lane→coord: within a 16x16 fragment, lane L element e maps to
    // (row fm + (e>>2)*8, col fn + e%4). N=32/M-as-two-frags pack as [.., 8+..].
    const short qid = (short)lid >> 2;
    const short fm = (qid & 4) | (((short)lid >> 1) & 3);
    const short fn = ((qid & 2) | ((short)lid & 1)) * 4;

    for (short e = 0; e < 8; ++e) {
        short r = fm + (e >> 2) * 8;
        short c = fn + (e % 4);
        ct_a[e] = a_tg[r * 16 + c];               // A[M,K], 1 fragment
        ct_b[e]     = b_tg[r * 16 + c];           // B[N,K] n-frag 0 (n 0..15)
        ct_b[8 + e] = b_tg[(r + 16) * 16 + c];    // B[N,K] n-frag 1 (n 16..31)
        ct_c[e] = 0.0f;
        ct_c[8 + e] = 0.0f;
    }

    gemm_op.run(ct_a, ct_b, ct_c);

    for (short e = 0; e < 8; ++e) {
        short r = fm + (e >> 2) * 8;
        short c = fn + (e % 4);
        c_out[r * 32 + c]      = ct_c[e];         // C[M,N] n 0..15
        c_out[r * 32 + c + 16] = ct_c[8 + e];     // C[M,N] n 16..31
    }
}
";

/// Run one NAX tile: `C[16×32] = A[16×16] · B[16×32]` on the M5 tensor engine.
/// `a` is row-major 16×16, `b` is row-major 16×32; returns row-major 16×32.
/// `Err("no Metal device …")` when no GPU is available, so callers can skip.
pub fn run_nax_matmul_tile(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
        MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    assert_eq!(a.len(), NAX_TILE_M * NAX_TILE_K, "A must be 16×16");
    assert_eq!(b.len(), NAX_TILE_K * NAX_TILE_N, "B must be 16×32");
    let out_len = NAX_TILE_M * NAX_TILE_N;

    let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
    let opts = objc2_metal::MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe);
    opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(NAX_MATMUL_TILE_SRC), Some(&opts))
        .map_err(|e| format!("metal: NAX MSL compile failed: {e:?}"))?;
    let function = library
        .newFunctionWithName(&NSString::from_str("nax_matmul_tile"))
        .ok_or("metal: kernel nax_matmul_tile not found")?;
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| format!("metal: pipeline build failed: {e:?}"))?;
    let queue = device.newCommandQueue().ok_or("metal: newCommandQueue returned nil")?;

    let res = MTLResourceOptions::StorageModeShared;
    let mk_in = |data: &[f32]| -> Result<_, String> {
        let bytes: &[u8] = bytemuck_cast(data);
        // SAFETY: `bytes` lives until the copy completes inside this call.
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                    bytes.len(),
                    res,
                )
                .ok_or_else(|| "metal: input buffer alloc failed".to_string())
        }
    };
    let a_buf = mk_in(a)?;
    let b_buf = mk_in(b)?;
    let c_buf = device
        .newBufferWithLength_options(out_len * 4, res)
        .ok_or("metal: output buffer alloc failed")?;

    let cb = queue.commandBuffer().ok_or("metal: commandBuffer returned nil")?;
    let enc = cb.computeCommandEncoder().ok_or("metal: computeCommandEncoder returned nil")?;
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
    }
    // One simdgroup (32 threads), one threadgroup.
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: 32, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let raw = unsafe {
        std::slice::from_raw_parts(c_buf.contents().as_ptr() as *const f32, out_len)
    };
    Ok(raw.to_vec())
}

/// Reinterpret an `&[f32]` as bytes without a dependency. (The runtime copies
/// it immediately into a Metal buffer.)
fn bytemuck_cast(data: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain-old-data; the returned slice covers exactly the same
    // bytes and borrows for the same lifetime.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

// =========================================================================
// General tiled NAX GEMM — arbitrary M, N, K
// =========================================================================
//
// One simdgroup per threadgroup computes one 16×32 output tile of C; the grid
// is ceil(M/16) × ceil(N/32) threadgroups. Each threadgroup walks K in steps of
// 16, staging A[16×16] and Bᵀ[32×16] sub-tiles into threadgroup memory (with
// bounds guards that zero-pad ragged edges), loading them into the `matmul2d`
// register cooperative tensors via the BaseNAXFrag layout, and accumulating
// into a persistent destination tensor across the K loop. The final tile is
// stored with per-element guards so partial M/N edges write only valid cells.
//
// This is the validated single-tile core (`run_nax_matmul_tile`) generalized:
// same fragment layout, now with a K-accumulation loop and edge handling.

/// MSL for the general NAX GEMM. `dims = (M, N, K)`.
///
/// Two levels of tiling. **Threadgroup**: 4 simdgroups (128 threads) arranged
/// 2×2 cooperatively stage the A[64×16] and Bᵀ[128×16] panels for a 64×128
/// output block — 128 threads share each device load. **Register**: each
/// simdgroup then computes its 32×64 sub-block as a 2×2 grid of 16×32
/// `matmul2d` tiles, loading 2 A row-fragments and 2 B column-fragment-pairs
/// per K-step and running all 4 products from them. Ragged M/N/K are zero-padded
/// on stage and guarded on store.
const NAX_MATMUL_SRC: &str = "\
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr uint BK   = 16;
constant constexpr uint TG_M = 64;    // threadgroup block rows  (2 simdgroups x 32)
constant constexpr uint TG_N = 128;   // threadgroup block cols  (2 simdgroups x 64)
constant constexpr uint SG_M = 32;    // simdgroup sub-block rows
constant constexpr uint SG_N = 64;    // simdgroup sub-block cols

[[kernel]] void nax_matmul(
    device const float* a_in [[buffer(0)]],   // M x K row-major
    device const float* b_in [[buffer(1)]],   // K x N row-major
    device float* c_out      [[buffer(2)]],   // M x N row-major
    constant uint3& dims     [[buffer(3)]],   // (M, N, K)
    uint2 tg  [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    const uint M = dims.x, N = dims.y, K = dims.z;
    const uint tm0 = tg.y * TG_M;          // threadgroup block base row
    const uint tn0 = tg.x * TG_N;          // threadgroup block base column
    const uint sm  = sgid / 2u;            // simdgroup's row slot (0,1)
    const uint sn  = sgid % 2u;            // simdgroup's col slot (0,1)
    const uint m0  = tm0 + sm * SG_M;      // this simdgroup's base row
    const uint n0  = tn0 + sn * SG_N;      // this simdgroup's base column
    const uint tid = sgid * 32u + lid;     // flat thread id in threadgroup (0..127)

    threadgroup bfloat a_tg[TG_M * BK];    // [TG_M, K-step] staging
    threadgroup bfloat b_tg[TG_N * BK];    // [TG_N, K-step] = transpose(B) staging

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        /*transpose_a=*/false, /*transpose_b=*/true, /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto a0 = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto a1 = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto b0 = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto b1 = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto c00 = gemm_op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto c01 = gemm_op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto c10 = gemm_op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto c11 = gemm_op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();

    const short qid = (short)lid >> 2;
    const short fm = (qid & 4) | (((short)lid >> 1) & 3);
    const short fn = ((qid & 2) | ((short)lid & 1)) * 4;

    for (short e = 0; e < 8; ++e) {
        c00[e] = 0.0f; c00[8 + e] = 0.0f;  c01[e] = 0.0f; c01[8 + e] = 0.0f;
        c10[e] = 0.0f; c10[8 + e] = 0.0f;  c11[e] = 0.0f; c11[8 + e] = 0.0f;
    }

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // All 128 threads cooperatively stage the threadgroup panels.
        for (uint i = tid; i < TG_M * BK; i += 128u) {
            uint r = i / BK, c = i % BK;              // r: M (in block), c: K
            uint gm = tm0 + r, gk = k0 + c;
            a_tg[i] = (gm < M && gk < K) ? bfloat(a_in[gm * K + gk]) : bfloat(0);
        }
        for (uint i = tid; i < TG_N * BK; i += 128u) {
            uint n = i / BK, c = i % BK;              // n: N (in block), c: K
            uint gn = tn0 + n, gk = k0 + c;
            b_tg[i] = (gn < N && gk < K) ? bfloat(b_in[gk * N + gn]) : bfloat(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // This simdgroup's sub-panel base within the shared staging.
        uint ar = sm * SG_M;     // row offset into a_tg (0 or 32)
        uint bn = sn * SG_N;     // col offset into b_tg (0 or 64)
        for (short e = 0; e < 8; ++e) {
            short r = fm + (e >> 2) * 8;
            short c = fn + (e % 4);
            a0[e] = a_tg[(ar + r) * BK + c];
            a1[e] = a_tg[(ar + r + 16) * BK + c];
            b0[e]     = b_tg[(bn + r) * BK + c];
            b0[8 + e] = b_tg[(bn + r + 16) * BK + c];
            b1[e]     = b_tg[(bn + r + 32) * BK + c];
            b1[8 + e] = b_tg[(bn + r + 48) * BK + c];
        }
        gemm_op.run(a0, b0, c00);
        gemm_op.run(a0, b1, c01);
        gemm_op.run(a1, b0, c10);
        gemm_op.run(a1, b1, c11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store this simdgroup's 2x2 tile block (rows m0+{0,16}, cols n0+{0,16,32,48}).
    for (short e = 0; e < 8; ++e) {
        short r = fm + (e >> 2) * 8;
        short c = fn + (e % 4);
        uint r0 = m0 + (uint)r;
        uint r1 = r0 + 16u;
        uint c0a = n0 + (uint)c;          uint c0b = c0a + 16u;   // tj=0 -> cols 0..31
        uint c1a = n0 + 32u + (uint)c;    uint c1b = c1a + 16u;   // tj=1 -> cols 32..63
        if (r0 < M) {
            if (c0a < N) c_out[r0 * N + c0a] = c00[e];
            if (c0b < N) c_out[r0 * N + c0b] = c00[8 + e];
            if (c1a < N) c_out[r0 * N + c1a] = c01[e];
            if (c1b < N) c_out[r0 * N + c1b] = c01[8 + e];
        }
        if (r1 < M) {
            if (c0a < N) c_out[r1 * N + c0a] = c10[e];
            if (c0b < N) c_out[r1 * N + c0b] = c10[8 + e];
            if (c1a < N) c_out[r1 * N + c1a] = c11[e];
            if (c1b < N) c_out[r1 * N + c1b] = c11[8 + e];
        }
    }
}
";

/// A compiled, reusable NAX GEMM context — builds the device/pipeline/queue
/// once so repeated `run` calls (and benchmarks) exclude compile cost. Created
/// with [`NaxGemm::new`]; `Err` if no Metal device or MPP won't compile (e.g.
/// a pre-M5 GPU without the NAX tensor engine).
#[cfg(feature = "metal")]
pub struct NaxGemm {
    device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    pipeline:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>,
    queue: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
}

#[cfg(feature = "metal")]
impl NaxGemm {
    /// Compile the general NAX GEMM kernel on the system default device.
    pub fn new() -> Result<Self, String> {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
        };
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        let opts = objc2_metal::MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(NAX_MATMUL_SRC), Some(&opts))
            .map_err(|e| format!("metal: NAX GEMM compile failed: {e:?}"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str("nax_matmul"))
            .ok_or("metal: kernel nax_matmul not found")?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| format!("metal: pipeline build failed: {e:?}"))?;
        let queue = device.newCommandQueue().ok_or("metal: newCommandQueue returned nil")?;
        Ok(Self { device, pipeline, queue })
    }

    /// `C(m×n) = A(m×k) · B(k×n)`, all row-major. A/B/C are f32 on the host;
    /// the kernel computes in bf16 (the NAX engine's input precision), so the
    /// result agrees with an f32 oracle only to bf16 tolerance.
    pub fn run(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;

        assert_eq!(a.len(), m * k, "A must be m×k");
        assert_eq!(b.len(), k * n, "B must be k×n");
        let out_len = m * n;
        let res = MTLResourceOptions::StorageModeShared;

        let mk_in = |data: &[f32]| -> Result<_, String> {
            let bytes: &[u8] = bytemuck_cast(data);
            // SAFETY: `bytes` lives until the copy completes inside this call.
            unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len().max(1),
                        res,
                    )
                    .ok_or_else(|| "metal: input buffer alloc failed".to_string())
            }
        };
        let a_buf = mk_in(a)?;
        let b_buf = mk_in(b)?;
        let c_buf = self
            .device
            .newBufferWithLength_options((out_len * 4).max(1), res)
            .ok_or("metal: output buffer alloc failed")?;
        let dims = [m as u32, n as u32, k as u32];
        // SAFETY: copies 12 bytes that live for the duration of this call.
        let dims_buf = unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(&dims),
                    res,
                )
                .ok_or("metal: dims buffer alloc failed")?
        };

        let cb = self.queue.commandBuffer().ok_or("metal: commandBuffer returned nil")?;
        let enc = cb.computeCommandEncoder().ok_or("metal: computeCommandEncoder returned nil")?;
        enc.setComputePipelineState(&self.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&dims_buf), 0, 3);
        }
        // One threadgroup (4 simdgroups, 128 threads) per 64×128 output block.
        let m_blocks = m.div_ceil(64);
        let n_blocks = n.div_ceil(128);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: n_blocks, height: m_blocks, depth: 1 },
            MTLSize { width: 128, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        let raw =
            unsafe { std::slice::from_raw_parts(c_buf.contents().as_ptr() as *const f32, out_len) };
        Ok(raw.to_vec())
    }

    /// Pure GPU kernel time (seconds) for one GEMM, from the command buffer's
    /// hardware timestamps — excludes buffer allocation, host→device copies,
    /// and readback. Buffers are allocated once and reused across `iters`
    /// dispatches (one command buffer), so this isolates kernel throughput from
    /// per-call CPU overhead. Returns the *total* GPU time over `iters`.
    pub fn gpu_time_seconds(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        iters: u32,
    ) -> Result<f64, String> {
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
            MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;

        let res = MTLResourceOptions::StorageModeShared;
        let mk_in = |data: &[f32]| -> Result<_, String> {
            let bytes: &[u8] = bytemuck_cast(data);
            // SAFETY: `bytes` lives until the copy completes inside this call.
            unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len().max(1),
                        res,
                    )
                    .ok_or_else(|| "alloc".to_string())
            }
        };
        let a_buf = mk_in(a)?;
        let b_buf = mk_in(b)?;
        let c_buf = self.device.newBufferWithLength_options((m * n * 4).max(1), res).ok_or("alloc")?;
        let dims = [m as u32, n as u32, k as u32];
        let dims_buf = unsafe {
            self.device
                .newBufferWithBytes_length_options(
                    NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(&dims),
                    res,
                )
                .ok_or("alloc")?
        };
        let m_blocks = m.div_ceil(64);
        let n_blocks = n.div_ceil(128);

        let cb = self.queue.commandBuffer().ok_or("cb")?;
        for _ in 0..iters {
            let enc = cb.computeCommandEncoder().ok_or("enc")?;
            enc.setComputePipelineState(&self.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 0, 3);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_blocks, height: m_blocks, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        // Hardware GPU timestamps (CFTimeInterval seconds) for the whole buffer.
        Ok(cb.GPUEndTime() - cb.GPUStartTime())
    }
}

/// Convenience: compile + run a general NAX GEMM once. For repeated calls or
/// benchmarks build a [`NaxGemm`] and reuse it (compiles the kernel once).
#[cfg(feature = "metal")]
pub fn run_nax_matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    NaxGemm::new()?.run(m, k, n, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;

    /// Minimal Metal Performance Primitives probe — confirms the M5 NAX
    /// toolchain (`mpp::tensor_ops::matmul2d` + the MPP framework include)
    /// compiles through our `objc2-metal` runtime as Metal 4.
    const MPP_PROBE: &str = "\
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
kernel void mpp_probe(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 16, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    (void)op;
    c[gid.y * 16 + gid.x] = a[gid.x] + b[gid.y];
}
";

    #[test]
    fn mpp_tensor_ops_compiles_as_metal4() {
        match compile_metal4(MPP_PROBE) {
            Ok(()) => eprintln!("MPP (mpp::tensor_ops) compiles as Metal 4 on this device ✓"),
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping MPP compile probe");
            }
            Err(e) => panic!("MPP shader failed to compile as Metal 4:\n{e}"),
        }
    }

    /// The NAX tensor engine produces a correct GEMM through our runtime.
    /// Small-integer inputs (exact in bf16) let us assert *exact* equality with
    /// the naive oracle. Two identity probes pin the fragment layout: with
    /// A = I, `B[k,n] = n` must yield `C[m,n] = n` (column mapping) and
    /// `B[k,n] = k` must yield `C[m,n] = m` (row mapping) — together these catch
    /// any cooperative-tensor axis swap or scramble in the BaseNAXFrag layout.
    #[test]
    fn nax_matmul_tile_matches_oracle() {
        // A[m,k] = (m + k) % 3, B[k,n] = (k + 2*n) % 4  — products ≤ 6, sums
        // over K=16 ≤ 96: all exact in bf16 and f32, and distinct per (m,n).
        let a: Vec<f32> =
            (0..NAX_TILE_M * NAX_TILE_K).map(|i| ((i / 16 + i % 16) % 3) as f32).collect();
        let b: Vec<f32> = (0..NAX_TILE_K * NAX_TILE_N)
            .map(|i| ((i / 32 + 2 * (i % 32)) % 4) as f32)
            .collect();

        let got = match run_nax_matmul_tile(&a, &b) {
            Ok(v) => v,
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping NAX matmul test");
                return;
            }
            Err(e) => panic!("NAX matmul failed: {e}"),
        };
        let want = crate::blas::naive_sgemm(NAX_TILE_M, NAX_TILE_K, NAX_TILE_N, &a, &b);
        assert_eq!(got, want, "NAX tile must match the naive oracle exactly");

        // Identity probes — A = I; column then row coordinate.
        let mut ai = vec![0.0f32; NAX_TILE_M * NAX_TILE_K];
        for d in 0..NAX_TILE_M {
            ai[d * NAX_TILE_K + d] = 1.0;
        }
        let col_probe = run_nax_matmul_tile(
            &ai,
            &(0..NAX_TILE_K * NAX_TILE_N).map(|i| (i % NAX_TILE_N) as f32).collect::<Vec<_>>(),
        )
        .unwrap();
        let row_probe = run_nax_matmul_tile(
            &ai,
            &(0..NAX_TILE_K * NAX_TILE_N).map(|i| (i / NAX_TILE_N) as f32).collect::<Vec<_>>(),
        )
        .unwrap();
        for m in 0..NAX_TILE_M {
            for n in 0..NAX_TILE_N {
                assert_eq!(col_probe[m * NAX_TILE_N + n], n as f32, "column map at ({m},{n})");
                assert_eq!(row_probe[m * NAX_TILE_N + n], m as f32, "row map at ({m},{n})");
            }
        }
        eprintln!("NAX matmul2d tile matches the oracle exactly (+ row/col layout) ✓");
    }

    /// The general tiled NAX GEMM is correct across shapes — including ragged
    /// M/N/K that exercise the zero-pad edge guards and multi-tile K
    /// accumulation. Small-integer inputs are exact in bf16, so we assert exact
    /// equality with the naive oracle.
    #[test]
    fn nax_matmul_general_matches_oracle() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => {
                eprintln!("no Metal device — skipping general NAX GEMM test");
                return;
            }
            Err(e) => panic!("NAX GEMM compile failed: {e}"),
        };
        // Exact tile (16,16,32); ragged in every dim; multi-K; K not /16; thin.
        let shapes = [
            (16usize, 16usize, 32usize),
            (1, 1, 1),
            (17, 33, 5),    // ragged M, N, K all
            (48, 16, 64),   // multi-tile, clean
            (50, 20, 70),   // multi-tile, ragged
            (7, 100, 3),    // wide N
            (100, 7, 3),    // tall M
        ];
        for (m, k, n) in shapes {
            // Small ints exact in bf16: a in 0..3, b in 0..4. Sum over K stays
            // well under bf16's 256 exact-integer limit for these K.
            let a: Vec<f32> = (0..m * k).map(|i| (i % 3) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32).collect();
            let got = ctx.run(m, k, n, &a, &b).unwrap();
            let want = crate::blas::naive_sgemm(m, k, n, &a, &b);
            assert_eq!(got, want, "NAX GEMM mismatch at shape ({m},{k},{n})");
        }
        eprintln!("general NAX GEMM matches the oracle across {} shapes ✓", shapes.len());
    }

    /// Random (non-bf16-exact) data: the NAX GEMM agrees with the f32 oracle to
    /// bf16 tolerance. Documents the precision the engine actually delivers.
    #[test]
    fn nax_matmul_general_bf16_tolerance() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        let (m, k, n) = (64usize, 48usize, 96usize);
        // Deterministic pseudo-random in [-1, 1].
        let prng = |i: usize| ((i.wrapping_mul(2654435761) % 2000) as f32 / 1000.0) - 1.0;
        let a: Vec<f32> = (0..m * k).map(prng).collect();
        let b: Vec<f32> = (0..k * n).map(|i| prng(i + 7)).collect();
        let got = ctx.run(m, k, n, &a, &b).unwrap();
        let want = crate::blas::naive_sgemm(m, k, n, &a, &b);
        let mut max_rel = 0.0f32;
        for (g, w) in got.iter().zip(&want) {
            let denom = w.abs().max(1.0);
            max_rel = max_rel.max((g - w).abs() / denom);
        }
        // bf16 has 8 mantissa bits; K=48 accumulation in f32 keeps error modest.
        assert!(max_rel < 0.05, "max relative error {max_rel} exceeds bf16 tolerance");
        eprintln!("NAX GEMM vs f32 oracle: max relative error {max_rel:.4} (bf16) ✓");
    }

    /// Real benchmark: NAX vs naive vs the linked BLAS (Accelerate on macOS) on
    /// a sizeable GEMM. Prints GFLOP/s for each so the speedup is concrete.
    /// `--ignored` because it's a perf measurement, not a correctness gate.
    #[test]
    #[ignore = "benchmark; run with --ignored --nocapture"]
    fn bench_nax_vs_blas() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping benchmark: {e}");
                return;
            }
        };
        let (m, k, n) = (1024usize, 1024usize, 1024usize);
        let prng = |i: usize| (i.wrapping_mul(2654435761) % 1000) as f32 / 1000.0;
        let a: Vec<f32> = (0..m * k).map(prng).collect();
        let b: Vec<f32> = (0..k * n).map(|i| prng(i + 3)).collect();
        let flops = 2.0 * m as f64 * k as f64 * n as f64;

        let bench = |label: &str, iters: u32, mut f: Box<dyn FnMut()>| {
            f(); // warm up
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!("{label:>12}: {:7.2} ms   {:7.1} GFLOP/s", secs * 1e3, flops / secs / 1e9);
        };

        // GPU-only kernel time (excludes alloc/copy/readback): 50 dispatches on
        // reused buffers, timed by hardware timestamps. Isolates kernel speed.
        let gpu_total = ctx.gpu_time_seconds(m, k, n, &a, &b, 50).unwrap();
        let gpu_each = gpu_total / 50.0;
        eprintln!(
            "{:>12}: {:7.2} ms   {:7.1} GFLOP/s   (GPU kernel only)",
            "NAX-gpu", gpu_each * 1e3, flops / gpu_each / 1e9
        );

        let (a1, b1) = (a.clone(), b.clone());
        bench("NAX-wall", 20, Box::new(move || {
            ctx.run(m, k, n, &a1, &b1).unwrap();
        }));
        let (a2, b2) = (a.clone(), b.clone());
        bench("BLAS/accel", 20, Box::new(move || {
            std::hint::black_box(crate::blas::sgemm_rowmajor(m, k, n, &a2, &b2));
        }));
        let (a3, b3) = (a.clone(), b.clone());
        bench("naive", 1, Box::new(move || {
            std::hint::black_box(crate::blas::naive_sgemm(m, k, n, &a3, &b3));
        }));
    }

    #[test]
    fn lowers_vector_add_to_msl() {
        let src = include_str!("../../examples/triton-ktir/vector_add_ktir.mlir");
        let module = parse_module(src).unwrap();
        let msl = emit_msl(&module, "add_kernel").expect("emit MSL");

        // Structural checks on the emitted shader.
        assert!(msl.contains("#include <metal_stdlib>"));
        assert!(msl.contains("kernel void add_kernel("));
        assert!(msl.contains("thread_position_in_grid"));
        // Three f16 buffers: two read-only inputs, one writable output.
        assert!(msl.contains("device const half* x_ptr [[buffer(0)]]"));
        assert!(msl.contains("device const half* y_ptr [[buffer(1)]]"));
        assert!(msl.contains("device half* output_ptr [[buffer(2)]]"));
        // The element-wise add, with the output buffer on the LHS.
        assert!(
            msl.contains("output_ptr[gid] = x_ptr[gid] + y_ptr[gid];"),
            "unexpected body:\n{msl}"
        );
    }

    #[test]
    fn gpu_matches_oracle_vector_add() {
        use crate::dtypes::DType;
        use crate::interpreter::{execute_function, Arg};
        use crate::ir::Scalar;

        let src = include_str!("../../examples/triton-ktir/vector_add_ktir.mlir");
        let module = parse_module(src).unwrap();
        let kernel = emit_kernel(&module, "add_kernel").unwrap();

        let n = 4096usize;
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();

        let gpu = match run_kernel(&kernel, &[x.clone(), y.clone()], n) {
            Ok(g) => g,
            // No GPU in this environment (e.g. headless CI) — skip, don't fail.
            Err(e) if e.contains("no Metal device") => {
                eprintln!("skipping GPU validation: {e}");
                return;
            }
            Err(e) => panic!("GPU run failed: {e}"),
        };

        // Oracle: the same kernel through the CPU interpreter.
        let args = [
            ("x_ptr", Arg::Tensor { data: x, shape: vec![n], dtype: DType::F16 }),
            ("y_ptr", Arg::Tensor { data: y, shape: vec![n], dtype: DType::F16 }),
            ("output_ptr", Arg::Tensor { data: vec![0.0; n], shape: vec![n], dtype: DType::F16 }),
            ("BLOCK_SIZE", Arg::Scalar(Scalar::I64(128))),
        ];
        let oracle = execute_function(&module, "add_kernel", &args).unwrap();
        let oracle = &oracle.get("output_ptr").unwrap().data;

        assert_eq!(gpu.len(), n);
        for i in 0..n {
            assert!(
                (gpu[i] - oracle[i]).abs() < 1e-2,
                "GPU vs oracle mismatch at {i}: gpu={}, oracle={}",
                gpu[i],
                oracle[i]
            );
        }
        eprintln!("GPU output matches the interpreter oracle over {n} elements ✓");
    }

    #[test]
    fn rejects_non_elementwise() {
        // matmul_small has a linalg.matmul -> not lowerable in slice 1.
        let src = include_str!("../../examples/latency/matmul_small.mlir");
        if let Ok(module) = parse_module(src) {
            let name = module.functions.keys().next().unwrap().clone();
            assert!(emit_msl(&module, &name).is_err());
        }
    }

    #[test]
    fn matmul_tier_detection() {
        use super::MatmulTier::*;
        // M5+ -> NAX (Neural Accelerator).
        assert_eq!(device_matmul_tier("Apple M5"), Nax);
        assert_eq!(device_matmul_tier("Apple M5 Pro"), Nax);
        assert_eq!(device_matmul_tier("Apple M6 Max"), Nax); // forward-compatible
        // M1..M4 Apple GPUs -> simdgroup matrix units.
        assert_eq!(device_matmul_tier("Apple M1"), Simdgroup);
        assert_eq!(device_matmul_tier("Apple M3 Max"), Simdgroup);
        assert_eq!(device_matmul_tier("Apple M4"), Simdgroup);
        // An Apple GPU with no M-number still gets the matrix path.
        assert_eq!(device_matmul_tier("Apple Paravirtual device"), Simdgroup);
        // Non-Apple -> naive floor.
        assert_eq!(device_matmul_tier("Intel UHD Graphics 630"), Naive);
        assert_eq!(device_matmul_tier("AMD Radeon Pro 5500M"), Naive);
        // Effective tier is the best *implemented* tier the device supports.
        // M5 (capability Nax) -> Nax, the highest implemented tier.
        assert_eq!(effective_matmul_tier("Apple M5"), HIGHEST_IMPLEMENTED);
        assert_eq!(effective_matmul_tier("Apple M5"), Nax);
        // Pre-NAX Apple GPUs are capability Simdgroup, but that kernel isn't
        // implemented yet, so they fall back to Naive (not the unimplemented tier).
        assert_eq!(effective_matmul_tier("Apple M4"), Naive);
        assert_eq!(effective_matmul_tier("Apple M1"), Naive);
        // Non-Apple stays at the naive floor.
        assert_eq!(effective_matmul_tier("Intel UHD Graphics 630"), Naive);
        assert!(!tier_implemented(Simdgroup));
    }

    #[test]
    fn reports_device_tier_on_real_gpu() {
        use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("no Metal device — skipping live tier check");
            return;
        };
        let name = device.name().to_string();
        let cap = device_matmul_tier(&name);
        eprintln!("device {name:?}: capability tier = {cap:?}, using = {:?}", effective_matmul_tier(&name));
        // This machine is an Apple GPU, so it must be at least the simdgroup tier.
        assert!(cap >= MatmulTier::Simdgroup, "expected an Apple GPU, got {name:?}");
    }
}
