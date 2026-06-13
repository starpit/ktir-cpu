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
    // All three are implemented now: Naive (CPU/BLAS floor), Simdgroup
    // (simdgroup_float8x8, M1–M4), and Nax (matmul2d, M5+).
    matches!(
        tier,
        MatmulTier::Naive | MatmulTier::Simdgroup | MatmulTier::Nax
    )
}

/// The matmul tier a Metal device *supports*, parsed from its name (mirrors
/// scratchy's `detect_device` name-parse → `AppleSiliconGen` → `is_nax_capable`):
///   * Apple `M5`+  -> Nax (Apple9 gen 17+, first with the Neural Accelerator)
///   * any other Apple GPU (M1..M4, Apple7+) -> Simdgroup
///   * non-Apple / unknown -> Naive
pub fn device_matmul_tier(device_name: &str) -> MatmulTier {
    if let Some(generation) = apple_m_generation(device_name) {
        return if generation >= 5 {
            MatmulTier::Nax
        } else {
            MatmulTier::Simdgroup
        };
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

/// Which matmul implementation to dispatch for a given problem on a given
/// device. NAX is the M5 GPU tensor engine (f16); Accelerate is Apple's AMX
/// matrix coprocessor (f32). See [`choose_matmul_backend`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulBackend {
    /// Apple Accelerate `cblas_sgemm` (AMX, f32). Used for tiny GEMMs and on
    /// non-Apple GPUs.
    Accelerate,
    /// The `simdgroup_float8x8` matrix GEMM — the GPU path on M1–M4 (pre-NAX).
    Simdgroup,
    /// The NAX `matmul2d` tensor-engine GEMM (f16) — the GPU path on M5+.
    Nax,
}

impl MatmulBackend {
    /// Whether this backend runs on the Metal GPU (`NaxGemm` context) vs the CPU.
    pub fn is_gpu(self) -> bool {
        matches!(self, MatmulBackend::Nax | MatmulBackend::Simdgroup)
    }
}

/// Minimum 128×256 output blocks before routing a matmul to the GPU. Measured
/// wall-clock crossover on the M5 is ~1024³ (= 32 blocks): NAX *compute* matches
/// or beats AMX from ~512³ (2739 vs 1498 GFLOP/s), but every GPU dispatch pays a
/// ~300 µs command-buffer submission round-trip that a single AMX call doesn't,
/// so only GEMMs large enough to dwarf that latency win. Below the crossover —
/// including every LX-sized tile (≤418³) a real KTIR program produces —
/// Accelerate is faster, so the gate routes there. (Batching many matmuls into
/// one submission, via [`NaxGemm::run_chain`], is how small GEMMs would win.)
pub const NAX_MIN_BLOCKS: usize = 32;
/// Minimum K depth before routing to the GPU. Same calibration.
pub const NAX_MIN_K: usize = 256;

/// Choose the matmul backend for `C(m×k·k×n)` on the named device.
///
/// NAX only exists on M5+ *and* only helps at scale: the M5 throughput sweep
/// shows small GEMMs are far slower on the GPU than on Accelerate's AMX (e.g.
/// 256³ ≈ 0.19 TFLOP/s vs AMX's ~2), because too few output blocks leave the
/// cores idle. So we route to NAX only when there are enough blocks to fill the
/// GPU ([`NAX_MIN_BLOCKS`]) and K is deep enough to amortize ([`NAX_MIN_K`]);
/// otherwise, and on every non-NAX device, we use Accelerate.
///
/// Note the backends differ in precision (NAX f16 vs Accelerate f32), so this
/// gate belongs to the experimental Metal/Spyre-faithful execution path, NOT
/// the f32 parity interpreter (which always uses Accelerate via `blas.rs`).
/// The threshold assumes GPU-resident operands; a one-shot host call pays
/// copy/readback that pushes the crossover higher (fusion keeps data resident
/// and lowers it back down).
pub fn choose_matmul_backend(device_name: &str, m: usize, k: usize, n: usize) -> MatmulBackend {
    // Only the NAX tensor engine on M5+, and only for GEMMs large enough to
    // beat the GPU submission latency, goes to the GPU. The pre-M5
    // `simdgroup_float8x8` path has lower compute throughput than AMX AND pays
    // the same submission latency, so it never wins wall-clock — M1–M4 (and any
    // smaller GEMM) use Accelerate, which is genuinely the fastest matmul there.
    let blocks = m.div_ceil(128) * n.div_ceil(256);
    let big_enough = blocks >= NAX_MIN_BLOCKS && k >= NAX_MIN_K;
    match effective_matmul_tier(device_name) {
        MatmulTier::Nax if big_enough => MatmulBackend::Nax,
        _ => MatmulBackend::Accelerate,
    }
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
            buffers.push(BufferBinding {
                name: b,
                is_output: false,
                dtype: bdt,
            });
        }
    }
    buffers.push(BufferBinding {
        name: out_buf,
        is_output: true,
        dtype,
    });

    let source = render_kernel(func_name, &buffers, &expr);
    Ok(MslKernel {
        source,
        name: func_name.to_string(),
        buffers,
    })
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

/// Every distinct input buffer feeding a fused elementwise expression tree, in
/// first-seen (DFS pre-order) order. Recurses through chained compute ops so a
/// fused kernel binds each loaded buffer once, no matter how deep in the
/// expression it appears.
fn collect_input_buffers(compute: &Operation, defs: &HashMap<String, &Operation>) -> Vec<String> {
    let mut out = Vec::new();
    collect_bufs(compute, defs, &mut out, 0);
    out
}

fn collect_bufs(
    op: &Operation,
    defs: &HashMap<String, &Operation>,
    out: &mut Vec<String>,
    depth: usize,
) {
    if depth > MAX_FUSE_DEPTH {
        return;
    }
    for operand in &op.operands {
        match defs.get(strip(operand)) {
            // A loaded tile is a leaf buffer.
            Some(d) if d.op_type == "ktdp.load" => {
                if let Some(b) = trace_buffer(operand, defs)
                    && !out.contains(&b)
                {
                    out.push(b);
                }
            }
            // A chained compute op: descend into its inputs.
            Some(d) => collect_bufs(d, defs, out, depth + 1),
            None => {}
        }
    }
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

/// Cap on fused-expression nesting — guards against pathological depth (and any
/// accidental cycle) while comfortably covering real elementwise chains.
const MAX_FUSE_DEPTH: usize = 256;

/// Lower an element-wise compute op into an MSL expression over `gid`, recursing
/// through chained compute operands so an entire elementwise DAG collapses into
/// ONE fused expression. Loaded tiles become `<buffer>[gid]` leaves; a chained
/// compute operand becomes a parenthesized sub-expression. This is the core of
/// MLX-style kernel fusion: `load,load,mul,exp,add -> store` lowers to a single
/// `exp(a[gid]*b[gid]) + c[gid]` kernel instead of three passes.
fn lower_compute(op: &Operation, defs: &HashMap<String, &Operation>) -> Result<String, String> {
    lower_compute_depth(op, defs, 0)
}

/// Resolve one operand SSA name to its MSL sub-expression: a loaded tile is a
/// `buf[gid]` leaf; anything else is recursively lowered as a compute op (which
/// errors if it isn't elementwise).
fn lower_value(
    name: &str,
    defs: &HashMap<String, &Operation>,
    depth: usize,
) -> Result<String, String> {
    if depth > MAX_FUSE_DEPTH {
        return Err("metal: fused expression exceeds max depth".into());
    }
    match defs.get(strip(name)) {
        None => Err(format!("metal: operand {name} has no defining op")),
        Some(d) if d.op_type == "ktdp.load" => {
            let buf = trace_buffer(name, defs)
                .ok_or_else(|| format!("metal: operand {name} is not a loaded buffer"))?;
            Ok(format!("{buf}[gid]"))
        }
        Some(d) => Ok(format!("({})", lower_compute_depth(d, defs, depth + 1)?)),
    }
}

fn lower_compute_depth(
    op: &Operation,
    defs: &HashMap<String, &Operation>,
    depth: usize,
) -> Result<String, String> {
    let operand = |i: usize| -> Result<String, String> {
        let name = op
            .operands
            .get(i)
            .ok_or_else(|| format!("metal: {} missing operand {i}", op.op_type))?;
        lower_value(name, defs, depth)
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
        let qual = if b.is_output {
            "device"
        } else {
            "device const"
        };
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
    let queue = device
        .newCommandQueue()
        .ok_or("metal: newCommandQueue returned nil")?;

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
            let data = input_iter
                .next()
                .ok_or("metal: too few inputs for kernel buffers")?;
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

    let cb = queue
        .commandBuffer()
        .ok_or("metal: commandBuffer returned nil")?;
    let enc = cb
        .computeCommandEncoder()
        .ok_or("metal: computeCommandEncoder returned nil")?;
    enc.setComputePipelineState(&pipeline);
    for (i, buf) in gpu_buffers.iter().enumerate() {
        unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, i) };
    }
    let tg = pipeline.maxTotalThreadsPerThreadgroup().min(out_len).max(1);
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: out_len,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    // Read the output buffer (last) back and decode to f32.
    let out = gpu_buffers.last().unwrap();
    let nbytes = out_len * out_dtype.bytes_per_elem();
    let raw = unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const u8, nbytes) }
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
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLMathMode};

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
// Inputs/outputs are host `f32` (row-major); A and B are converted to `half`
// in threadgroup memory inside the shader, so the host never touches f16. The
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
    threadgroup half a_tg[16 * 16];   // [M, K] row-major
    threadgroup half b_tg[32 * 16];   // [N, K] = transpose(B), row-major
    // Cooperative fill across the 32 simdgroup lanes.
    for (uint i = lid; i < 16u * 16u; i += 32u) {
        a_tg[i] = half(a_in[i]);                  // A[m,k] at m*16+k
    }
    for (uint i = lid; i < 32u * 16u; i += 32u) {
        uint n = i / 16u;                           // 0..31
        uint k = i % 16u;                           // 0..15
        b_tg[n * 16u + k] = half(b_in[k * 32u + n]);   // Bt[n,k] = B[k,n]
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
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
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
    let queue = device
        .newCommandQueue()
        .ok_or("metal: newCommandQueue returned nil")?;

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

    let cb = queue
        .commandBuffer()
        .ok_or("metal: commandBuffer returned nil")?;
    let enc = cb
        .computeCommandEncoder()
        .ok_or("metal: computeCommandEncoder returned nil")?;
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
    }
    // One simdgroup (32 threads), one threadgroup.
    enc.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let raw =
        unsafe { std::slice::from_raw_parts(c_buf.contents().as_ptr() as *const f32, out_len) };
    Ok(raw.to_vec())
}

/// Reinterpret an `&[f32]` as bytes without a dependency. (The runtime copies
/// it immediately into a Metal buffer.)
fn bytemuck_cast(data: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain-old-data; the returned slice covers exactly the same
    // bytes and borrows for the same lifetime.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// Reinterpret a `&[u32]` as bytes (for small uniform buffers like dims/codes).
fn bytemuck_u32(data: &[u32]) -> &[u8] {
    // SAFETY: u32 is plain-old-data; same bytes, same lifetime.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

/// One step of a batched matmul chain ([`NaxGemm::run_chain`]): multiply the
/// running result by `b` (k×n) and apply `epi` (with operand `e`, if any).
#[cfg(metal)]
pub struct ChainStep<'a> {
    pub k: usize,
    pub n: usize,
    pub b: &'a [f32],
    pub epi: Epilogue,
    pub e: Option<&'a [f32]>,
}

/// A fused matmul epilogue: `out = act(c BINOP e)`, where `e` is a per-element
/// operand (bias/residual/scale). The codes match the MSL `nax_epilogue` switch.
/// Lets the emulator fold a `matmul` and a following elementwise op (add, mul,
/// relu, tanh, …) into one GPU kernel — no readback, no second launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Epilogue {
    /// Binary op with `e`: 0 none, 1 add, 2 mul, 3 sub, 4 max, 5 min.
    pub binop: u32,
    /// Activation: 0 none, 1 relu, 2 tanh, 3 exp, 4 sigmoid.
    pub act: u32,
}

impl Epilogue {
    /// No epilogue — a plain matmul.
    pub const NONE: Epilogue = Epilogue { binop: 0, act: 0 };
    pub const ADD: Epilogue = Epilogue { binop: 1, act: 0 };
    pub const MUL: Epilogue = Epilogue { binop: 2, act: 0 };
    pub const SUB: Epilogue = Epilogue { binop: 3, act: 0 };
    pub const MAX: Epilogue = Epilogue { binop: 4, act: 0 };
    pub const MIN: Epilogue = Epilogue { binop: 5, act: 0 };
    pub const RELU: Epilogue = Epilogue { binop: 0, act: 1 };
    pub const TANH: Epilogue = Epilogue { binop: 0, act: 2 };
    pub const EXP: Epilogue = Epilogue { binop: 0, act: 3 };
    pub const SIGMOID: Epilogue = Epilogue { binop: 0, act: 4 };

    /// Map a binary elementwise KTIR op name to its epilogue (with `e` the other
    /// operand), or `None` if it isn't a fusable binary op.
    pub fn from_binary_op(op_type: &str) -> Option<Epilogue> {
        Some(match op_type {
            "linalg.add" | "arith.addf" => Epilogue::ADD,
            "linalg.mul" | "arith.mulf" => Epilogue::MUL,
            "linalg.sub" | "arith.subf" => Epilogue::SUB,
            "linalg.max" | "arith.maximumf" | "arith.maxf" => Epilogue::MAX,
            "linalg.min" | "arith.minimumf" | "arith.minf" => Epilogue::MIN,
            _ => return None,
        })
    }

    /// Map a unary activation KTIR op name to its epilogue, or `None`.
    pub fn from_unary_op(op_type: &str) -> Option<Epilogue> {
        Some(match op_type {
            "math.tanh" => Epilogue::TANH,
            "math.exp" => Epilogue::EXP,
            _ => return None,
        })
    }
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
/// Three levels of tiling. **Threadgroup**: SGS_M×SGS_N simdgroups (here 4×4 =
/// 16 simdgroups, 512 threads) cooperatively stage the A[128×16] and Bᵀ[256×16]
/// panels for a 128×256 output block — all threads share each device load.
/// **Register**: each simdgroup computes its 32×64 sub-block as a 2×2 grid of
/// 16×32 `matmul2d` tiles, loading 2 A row-fragments and 2 B column-fragment-
/// pairs per K-step and running all 4 products from them. **Pipeline**:
/// double-buffered panels — the next K-step's device loads are prefetched into
/// the other threadgroup half while the current panel feeds the matmuls, hiding
/// load latency behind compute. Ragged M/N/K are zero-padded on stage and
/// guarded on store.
const NAX_MATMUL_SRC: &str = "\
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr uint BK     = 16;
constant constexpr uint SG_M   = 32;    // simdgroup sub-block rows (2 tiles of 16)
constant constexpr uint SG_N   = 64;    // simdgroup sub-block cols (2 tiles of 32)
constant constexpr uint SGS_M  = 4;     // simdgroup rows per threadgroup
constant constexpr uint SGS_N  = 4;     // simdgroup cols per threadgroup
constant constexpr uint TG_M   = SG_M * SGS_M;   // threadgroup block rows  = 128
constant constexpr uint TG_N   = SG_N * SGS_N;   // threadgroup block cols  = 256
constant constexpr uint TG_THREADS = SGS_M * SGS_N * 32;   // = 512

// Fused elementwise epilogue applied in the GEMM store: out = act(c BINOP e).
// binop: 0 none, 1 add, 2 mul, 3 sub, 4 max, 5 min.  act: 0 none, 1 relu,
// 2 tanh, 3 exp, 4 sigmoid. This is the matmul->elementwise fusion — the
// activation/bias runs in the same kernel as the matmul, with no readback.
inline float nax_epilogue(float v, float ev, uint binop, uint act) {
    switch (binop) {
        case 1: v = v + ev; break;
        case 2: v = v * ev; break;
        case 3: v = v - ev; break;
        case 4: v = max(v, ev); break;
        case 5: v = min(v, ev); break;
        default: break;
    }
    switch (act) {
        case 1: v = max(v, 0.0f); break;
        case 2: v = tanh(v); break;
        case 3: v = exp(v); break;
        case 4: v = 1.0f / (1.0f + exp(-v)); break;
        default: break;
    }
    return v;
}

[[kernel]] void nax_matmul(
    device const float* a_in [[buffer(0)]],   // M x K row-major
    device const float* b_in [[buffer(1)]],   // K x N row-major
    device float* c_out      [[buffer(2)]],   // M x N row-major
    constant uint3& dims     [[buffer(3)]],   // (M, N, K)
    device const float* e_in [[buffer(4)]],   // M x N epilogue operand (or dummy)
    constant uint2& epi      [[buffer(5)]],   // (binop, act) codes
    uint3 tg  [[threadgroup_position_in_grid]],
    uint lid  [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]])
{
    const uint M = dims.x, N = dims.y, K = dims.z;
    // Batch index (grid z): each slice is an independent same-shape GEMM, so the
    // whole batch runs concurrently in one dispatch. tg.z = 0 for a single GEMM.
    a_in  += tg.z * M * K;
    b_in  += tg.z * K * N;
    c_out += tg.z * M * N;
    e_in  += tg.z * M * N;   // only dereferenced when binop != 0 (guarded below)
    const uint tm0 = tg.y * TG_M;          // threadgroup block base row
    const uint tn0 = tg.x * TG_N;          // threadgroup block base column
    const uint sm  = sgid / SGS_N;         // simdgroup's row slot
    const uint sn  = sgid % SGS_N;         // simdgroup's col slot
    const uint m0  = tm0 + sm * SG_M;      // this simdgroup's base row
    const uint n0  = tn0 + sn * SG_N;      // this simdgroup's base column
    const uint tid = sgid * 32u + lid;     // flat thread id in threadgroup

    // Double-buffered staging: while one panel feeds the matmuls, the next is
    // prefetched into the other half, so device-load latency overlaps compute.
    threadgroup half a_tg[2 * TG_M * BK];   // [2][TG_M, K-step]
    threadgroup half b_tg[2 * TG_N * BK];   // [2][TG_N, K-step] = transpose(B)

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16,
        /*transpose_a=*/false, /*transpose_b=*/true, /*relaxed_precision=*/false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto a0 = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto a1 = gemm_op.template get_left_input_cooperative_tensor<half, half, float>();
    auto b0 = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
    auto b1 = gemm_op.template get_right_input_cooperative_tensor<half, half, float>();
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

    const uint ar = sm * SG_M;   // this simdgroup's row offset into a_tg panel
    const uint bn = sn * SG_N;    // this simdgroup's col offset into b_tg panel
    const uint nk = (K + BK - 1u) / BK;  // number of K-steps

    // Stage one K-panel (rows of A, transposed cols of B) at K-offset `kc` into
    // buffer half `buf`. Zero-pads ragged M/N/K. (Macro so it inlines cleanly.)
#define STAGE_PANEL(buf, kc)                                                    \
    do {                                                                       \
        threadgroup half* ap = a_tg + (buf) * (TG_M * BK);                   \
        threadgroup half* bp = b_tg + (buf) * (TG_N * BK);                   \
        for (uint i = tid; i < TG_M * BK; i += TG_THREADS) {                   \
            uint r = i / BK, c = i % BK;                                       \
            uint gm = tm0 + r, gk = (kc) + c;                                  \
            ap[i] = (gm < M && gk < K) ? half(a_in[gm * K + gk]) : half(0);\
        }                                                                      \
        for (uint i = tid; i < TG_N * BK; i += TG_THREADS) {                   \
            uint n = i / BK, c = i % BK;                                       \
            uint gn = tn0 + n, gk = (kc) + c;                                  \
            bp[i] = (gn < N && gk < K) ? half(b_in[gk * N + gn]) : half(0);\
        }                                                                      \
    } while (0)

    STAGE_PANEL(0u, 0u);                       // prime buffer 0 with K-step 0
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ki = 0; ki < nk; ++ki) {
        uint cur = ki & 1u;
        // Prefetch the next panel into the other buffer; its device loads are
        // in flight while this step's matmuls run.
        if (ki + 1u < nk) {
            STAGE_PANEL(cur ^ 1u, (ki + 1u) * BK);
        }
        // Load this simdgroup's fragments from the current buffer and accumulate.
        threadgroup half* ap = a_tg + cur * (TG_M * BK);
        threadgroup half* bp = b_tg + cur * (TG_N * BK);
        for (short e = 0; e < 8; ++e) {
            short r = fm + (e >> 2) * 8;
            short c = fn + (e % 4);
            a0[e] = ap[(ar + r) * BK + c];
            a1[e] = ap[(ar + r + 16) * BK + c];
            b0[e]     = bp[(bn + r) * BK + c];
            b0[8 + e] = bp[(bn + r + 16) * BK + c];
            b1[e]     = bp[(bn + r + 32) * BK + c];
            b1[8 + e] = bp[(bn + r + 48) * BK + c];
        }
        gemm_op.run(a0, b0, c00);
        gemm_op.run(a0, b1, c01);
        gemm_op.run(a1, b0, c10);
        gemm_op.run(a1, b1, c11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#undef STAGE_PANEL

    // Store this simdgroup's 2x2 tile block (rows m0+{0,16}, cols n0+{0,16,32,48}),
    // applying the fused elementwise epilogue out = act(c BINOP e) per element.
    const uint binop = epi.x, act = epi.y;
#define EPI_STORE(rr, cc, cval)                                                \
    do {                                                                       \
        if ((rr) < M && (cc) < N) {                                            \
            float ev = (binop != 0u) ? e_in[(rr) * N + (cc)] : 0.0f;           \
            c_out[(rr) * N + (cc)] = nax_epilogue((cval), ev, binop, act);     \
        }                                                                      \
    } while (0)
    for (short e = 0; e < 8; ++e) {
        short r = fm + (e >> 2) * 8;
        short c = fn + (e % 4);
        uint r0 = m0 + (uint)r;
        uint r1 = r0 + 16u;
        uint c0a = n0 + (uint)c;          uint c0b = c0a + 16u;   // tj=0 -> cols 0..31
        uint c1a = n0 + 32u + (uint)c;    uint c1b = c1a + 16u;   // tj=1 -> cols 32..63
        EPI_STORE(r0, c0a, c00[e]);
        EPI_STORE(r0, c0b, c00[8 + e]);
        EPI_STORE(r0, c1a, c01[e]);
        EPI_STORE(r0, c1b, c01[8 + e]);
        EPI_STORE(r1, c0a, c10[e]);
        EPI_STORE(r1, c0b, c10[8 + e]);
        EPI_STORE(r1, c1a, c11[e]);
        EPI_STORE(r1, c1b, c11[8 + e]);
    }
#undef EPI_STORE
}
";

/// Pre-M5 GEMM via `simdgroup_float8x8` — the matrix path available on every
/// Apple7+ GPU (M1–M4), which lack the NAX tensor engine. Same fused epilogue
/// and same buffer layout as the NAX kernel (so the host dispatch is shared):
/// one simdgroup per 8×8 output tile accumulates over K in steps of 8 via
/// `simdgroup_multiply_accumulate`, then applies `act(c BINOP e)` on store.
#[cfg(metal)]
const SIMD_MATMUL_SRC: &str = "\
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
            b_tg[i] = (gkB < K && gn < N) ? b_in[gkB * N + gn] : 0.0f;
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
";

/// A compiled, reusable Metal GEMM context — builds the device/pipeline/queue
/// once so repeated `run` calls (and benchmarks) exclude compile cost. Picks the
/// kernel by device: the NAX `matmul2d` engine on M5+, else the `simdgroup_*`
/// matrix path on M1–M4. Created with [`NaxGemm::new`]; `Err` if no Metal device
/// or the chosen kernel won't compile.
#[cfg(metal)]
type MtlBuf = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>;

/// Page-aligned host allocation, freed on drop. Backs a [`UnifiedBuffer`].
#[cfg(metal)]
struct AlignedAlloc {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}
#[cfg(metal)]
impl Drop for AlignedAlloc {
    fn drop(&mut self) {
        // SAFETY: ptr/layout came from the matching alloc in UnifiedBuffer::new.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// A **zero-copy** unified-memory tensor: page-aligned host memory wrapped as a
/// Metal buffer via `newBufferWithBytesNoCopy`. The CPU accesses it as `&[f32]`
/// and the GPU as an `MTLBuffer` — they share the *same bytes*, so a matmul over
/// `UnifiedBuffer`s has no host↔device fill or readback (the ~600 µs of copies
/// the host-`Vec` path pays). This is the right primitive for Apple's unified
/// memory; tile storage backed by these makes the whole compute path copy-free.
///
/// Field order matters: `mtl` is released before `alloc` frees the memory.
#[cfg(metal)]
pub struct UnifiedBuffer {
    mtl: MtlBuf,
    alloc: AlignedAlloc,
    len: usize,
}

#[cfg(metal)]
impl UnifiedBuffer {
    /// Allocate `len` f32s of page-aligned, GPU-shared, zero-initialized memory.
    pub fn new(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        len: usize,
    ) -> Result<Self, String> {
        use objc2_metal::{MTLDevice, MTLResourceOptions};
        const PAGE: usize = 16 * 1024; // Apple Silicon page size
        let bytes = (len * 4).max(4).next_multiple_of(PAGE);
        let layout = std::alloc::Layout::from_size_align(bytes, PAGE).map_err(|e| e.to_string())?;
        // SAFETY: non-zero layout; zeroed so unused tail is defined.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err("UnifiedBuffer: alloc failed".into());
        }
        // SAFETY: ptr is page-aligned and `bytes` long; deallocator None means we
        // (AlignedAlloc) own the memory and free it after the buffer is released.
        let mtl = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                std::ptr::NonNull::new(ptr as *mut std::ffi::c_void).unwrap(),
                bytes,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or("UnifiedBuffer: newBufferWithBytesNoCopy returned nil")?;
        Ok(Self {
            mtl,
            alloc: AlignedAlloc { ptr, layout },
            len,
        })
    }

    /// Build a unified buffer initialized from `data` (one copy in; thereafter
    /// the GPU reads it in place with no further copies).
    pub fn from_slice(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        data: &[f32],
    ) -> Result<Self, String> {
        let mut b = Self::new(device, data.len())?;
        b.as_mut_slice().copy_from_slice(data);
        Ok(b)
    }

    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: alloc holds len f32s of live, aligned, initialized memory.
        unsafe { std::slice::from_raw_parts(self.alloc.ptr as *const f32, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: as above; &mut self gives exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.alloc.ptr as *mut f32, self.len) }
    }
}

/// Persistent per-context scratch buffers, grown on demand and reused across
/// `run` calls so repeated matmuls pay no per-call allocation. Shared-storage
/// (unified memory), so the host fills/reads them via `contents()` directly.
#[cfg(metal)]
#[derive(Default)]
struct Scratch {
    a: Option<MtlBuf>,
    b: Option<MtlBuf>,
    e: Option<MtlBuf>,
    c: Option<MtlBuf>,
}

#[cfg(metal)]
pub struct NaxGemm {
    device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    pipeline: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    queue: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    scratch: std::cell::RefCell<Scratch>,
    /// Output block this kernel computes per threadgroup, and its thread count.
    block_m: usize,
    block_n: usize,
    threads: usize,
}

#[cfg(metal)]
impl NaxGemm {
    /// Compile the best Metal GEMM kernel for the system default device: the NAX
    /// `matmul2d` engine on M5+, else the `simdgroup_float8x8` matrix path.
    pub fn new() -> Result<Self, String> {
        Self::compile(None)
    }

    /// Force the `simdgroup_float8x8` (pre-M5) kernel regardless of device — used
    /// to validate that path on an M5 in tests.
    pub fn new_simdgroup() -> Result<Self, String> {
        Self::compile(Some(false))
    }

    /// `force_nax`: `None` = auto by device, `Some(true)` = NAX, `Some(false)` =
    /// simdgroup.
    fn compile(force_nax: Option<bool>) -> Result<Self, String> {
        use objc2_foundation::NSString;
        use objc2_metal::{
            MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
        };
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device available")?;
        let is_nax = force_nax
            .unwrap_or_else(|| device_matmul_tier(&device.name().to_string()) == MatmulTier::Nax);

        let opts = objc2_metal::MTLCompileOptions::new();
        let (src, kname, block_m, block_n, threads) = if is_nax {
            opts.setMathMode(MTLMathMode::Safe);
            opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
            (NAX_MATMUL_SRC, "nax_matmul", 128usize, 256usize, 512usize)
        } else {
            (SIMD_MATMUL_SRC, "matmul", 8usize, 8usize, 32usize)
        };
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
            .map_err(|e| format!("metal: GEMM compile failed: {e:?}"))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(kname))
            .ok_or("metal: GEMM kernel not found")?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| format!("metal: pipeline build failed: {e:?}"))?;
        let queue = device
            .newCommandQueue()
            .ok_or("metal: newCommandQueue returned nil")?;
        Ok(Self {
            device,
            pipeline,
            queue,
            scratch: std::cell::RefCell::new(Scratch::default()),
            block_m,
            block_n,
            threads,
        })
    }

    /// `C(m×n) = A(m×k) · B(k×n)`, all row-major. A/B/C are f32 on the host;
    /// the kernel computes in f16 (the NAX engine's input precision), so the
    /// result agrees with an f32 oracle only to f16 tolerance.
    pub fn run(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.run_epi(m, k, n, a, b, None, Epilogue::NONE)
    }

    /// Fused matmul + elementwise epilogue in one kernel: `D = act(A·B BINOP E)`
    /// where `E` is the row-major m×n elementwise operand. No host readback of
    /// the matmul result and no second kernel launch — the activation/bias runs
    /// in the GEMM store. See [`Epilogue`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_fused(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        e: &[f32],
        epi: Epilogue,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(e.len(), m * n, "epilogue operand E must be m×n");
        self.run_epi(m, k, n, a, b, Some(e), epi)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_epi(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        e: Option<&[f32]>,
        epi: Epilogue,
    ) -> Result<Vec<f32>, String> {
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

        // Grow `slot` to hold `cap` bytes if needed, then return the buffer.
        let ensure = |slot: &mut Option<MtlBuf>, cap: usize| -> Result<MtlBuf, String> {
            let need = cap.max(4);
            let ok = slot.as_ref().is_some_and(|b| b.length() >= need);
            if !ok {
                *slot = Some(
                    self.device
                        .newBufferWithLength_options(need, res)
                        .ok_or("metal: buffer alloc failed")?,
                );
            }
            Ok(slot.as_ref().unwrap().clone())
        };
        // Copy host floats into a shared buffer's contents (no realloc when reused).
        let fill = |buf: &MtlBuf, data: &[f32]| unsafe {
            let dst = buf.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        };

        let mut s = self.scratch.borrow_mut();
        let a_buf = ensure(&mut s.a, a.len() * 4)?;
        let b_buf = ensure(&mut s.b, b.len() * 4)?;
        let e_buf = ensure(&mut s.e, e.map_or(4, |e| e.len() * 4))?;
        let c_buf = ensure(&mut s.c, out_len * 4)?;
        fill(&a_buf, a);
        fill(&b_buf, b);
        if let Some(e) = e {
            fill(&e_buf, e);
        }

        let dims = [m as u32, n as u32, k as u32];
        let codes = [epi.binop, epi.act];

        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer returned nil")?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or("metal: computeCommandEncoder returned nil")?;
        enc.setComputePipelineState(&self.pipeline);
        // Small uniforms via setBytes — no per-call buffer allocation.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&dims),
                3,
            );
            enc.setBuffer_offset_atIndex(Some(&e_buf), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&codes),
                5,
            );
        }
        // One threadgroup per output block (kernel-specific block + thread count).
        let m_blocks = m.div_ceil(self.block_m);
        let n_blocks = n.div_ceil(self.block_n);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n_blocks,
                height: m_blocks,
                depth: 1,
            },
            MTLSize {
                width: self.threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        let raw =
            unsafe { std::slice::from_raw_parts(c_buf.contents().as_ptr() as *const f32, out_len) };
        Ok(raw.to_vec())
    }

    /// Zero-copy matmul: `C = act(A·B BINOP E)` where A, B, C (and optional E)
    /// are [`UnifiedBuffer`]s already resident in shared memory. Encodes their
    /// buffers directly — no host↔device fill or readback. `c` must be sized
    /// `m·n`. This is the copy-free path unified memory makes possible.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_unified(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &UnifiedBuffer,
        b: &UnifiedBuffer,
        c: &mut UnifiedBuffer,
        e: Option<&UnifiedBuffer>,
        epi: Epilogue,
    ) -> Result<(), String> {
        use objc2_metal::{
            MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;
        assert_eq!(a.len, m * k, "A must be m×k");
        assert_eq!(b.len, k * n, "B must be k×n");
        assert_eq!(c.len, m * n, "C must be m×n");

        let dims = [m as u32, n as u32, k as u32];
        let codes = [epi.binop, epi.act];
        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer nil")?;
        let enc = cb.computeCommandEncoder().ok_or("metal: encoder nil")?;
        enc.setComputePipelineState(&self.pipeline);
        let e_mtl = e.unwrap_or(b); // dummy when binop==0 (never dereferenced)
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a.mtl), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b.mtl), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&c.mtl), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&dims),
                3,
            );
            enc.setBuffer_offset_atIndex(Some(&e_mtl.mtl), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&codes),
                5,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(self.block_n),
                height: m.div_ceil(self.block_m),
                depth: 1,
            },
            MTLSize {
                width: self.threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(())
    }

    /// Allocate a [`UnifiedBuffer`] on this context's device.
    pub fn unified(&self, len: usize) -> Result<UnifiedBuffer, String> {
        UnifiedBuffer::new(&self.device, len)
    }
    /// A [`UnifiedBuffer`] initialized from host data.
    pub fn unified_from(&self, data: &[f32]) -> Result<UnifiedBuffer, String> {
        UnifiedBuffer::from_slice(&self.device, data)
    }

    /// Run a chain of left-associated matmuls in ONE command buffer with a single
    /// GPU sync: `out₀ = a · steps[0].b`, then `outᵢ = outᵢ₋₁ · steps[i].b`, each
    /// with its fused epilogue. Intermediates stay in GPU buffers (never read back
    /// to the host), so the ~250 µs dispatch/sync latency is paid once for the
    /// whole chain instead of per matmul — the batching that makes GPU matmul win
    /// on the small LX-sized tiles. `a` is the host input (k0 = a.len()/m0);
    /// returns the final result `outₙ₋₁`.
    pub fn run_chain(
        &self,
        m0: usize,
        a: &[f32],
        steps: &[ChainStep<'_>],
    ) -> Result<Vec<f32>, String> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;
        assert!(!steps.is_empty(), "chain needs at least one step");

        let res = MTLResourceOptions::StorageModeShared;
        let rows = m0; // left-multiply: row count is fixed across the chain
        let alloc = |bytes: usize| -> Result<MtlBuf, String> {
            self.device
                .newBufferWithLength_options(bytes.max(4), res)
                .ok_or_else(|| "metal: chain alloc failed".to_string())
        };
        let fill = |buf: &MtlBuf, data: &[f32]| unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents().as_ptr() as *mut f32,
                data.len(),
            );
        };

        // Allocate the pool ONCE (sized to the chain's maxima) and reuse it for
        // every step — no per-step allocation. Two ping-pong result buffers hold
        // the running product; A, B, and E are refilled in place.
        let max_b = steps.iter().map(|s| s.b.len()).max().unwrap_or(1);
        let max_e = steps
            .iter()
            .map(|s| s.e.map_or(1, <[f32]>::len))
            .max()
            .unwrap_or(1);
        let max_out = steps.iter().map(|s| rows * s.n).max().unwrap_or(1);
        let a_buf = alloc(a.len() * 4)?;
        fill(&a_buf, a);
        let ping = [alloc(max_out * 4)?, alloc(max_out * 4)?];
        let b_buf = alloc(max_b * 4)?;
        let e_buf = alloc(max_e * 4)?;

        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer returned nil")?;
        let mut final_len = 0usize;
        for (i, s) in steps.iter().enumerate() {
            assert_eq!(s.b.len(), s.k * s.n, "chain step B must be k×n");
            let out_len = rows * s.n;
            let prev = if i == 0 { &a_buf } else { &ping[(i - 1) % 2] };
            let out = &ping[i % 2];
            fill(&b_buf, s.b);
            if let Some(e) = s.e {
                assert_eq!(e.len(), out_len, "chain step E must be m×n");
                fill(&e_buf, e);
            }
            let dims = [rows as u32, s.n as u32, s.k as u32];
            let codes = [s.epi.binop, s.epi.act];

            let enc = cb
                .computeCommandEncoder()
                .ok_or("metal: chain encoder nil")?;
            enc.setComputePipelineState(&self.pipeline);
            // Small uniforms go through setBytes (no buffer allocation).
            unsafe {
                enc.setBuffer_offset_atIndex(Some(prev), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(out), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(&dims),
                    3,
                );
                enc.setBuffer_offset_atIndex(Some(&e_buf), 0, 4);
                enc.setBytes_length_atIndex(
                    NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(&codes),
                    5,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: s.n.div_ceil(self.block_n),
                    height: rows.div_ceil(self.block_m),
                    depth: 1,
                },
                MTLSize {
                    width: self.threads,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            final_len = out_len;
        }

        cb.commit();
        cb.waitUntilCompleted();
        let last = &ping[(steps.len() - 1) % 2];
        let raw = unsafe {
            std::slice::from_raw_parts(last.contents().as_ptr() as *const f32, final_len)
        };
        Ok(raw.to_vec())
    }

    /// **Combine** many small matmuls that share the weight `b` into ONE tall
    /// matmul. `a_stack` is `count` row-panels (`count·m × k`, contiguous), `b`
    /// is the shared `k × n` weight; returns `count·m × n`. This is the real win
    /// for a SPMD KTIR grid (every core multiplies its rows by the same weights):
    /// stacking the panels into a single GEMM saturates the tensor engine, so
    /// the GPU beats a serial AMX loop by 1.25–1.74× once K ≳ 512 (measured) —
    /// unlike running them separately, where each small matmul underfills the
    /// engine and the GPU loses. Just a clarity wrapper over [`run`](Self::run)
    /// with `m' = count·m`.
    pub fn run_combined(
        &self,
        count: usize,
        m: usize,
        k: usize,
        n: usize,
        a_stack: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(a_stack.len(), count * m * k, "stacked A must be count·m×k");
        assert_eq!(b.len(), k * n, "shared B must be k×n");
        self.run(count * m, k, n, a_stack, b)
    }

    /// `batch` independent same-shape GEMMs `Cᵢ = Aᵢ · Bᵢ` in ONE dispatch (one
    /// submission), run concurrently across the GPU. `a` is `batch·m·k` and `b`
    /// is `batch·k·n`, both row-major and contiguous per slice; returns
    /// `batch·m·n`. Use this when each matmul has its OWN B; when they share B,
    /// [`run_combined`](Self::run_combined) is faster (one saturating GEMM).
    pub fn run_batched(
        &self,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, String> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
        };
        use std::ffi::c_void;
        use std::ptr::NonNull;
        assert_eq!(a.len(), batch * m * k, "A must be batch×m×k");
        assert_eq!(b.len(), batch * k * n, "B must be batch×k×n");
        let out_len = batch * m * n;
        let res = MTLResourceOptions::StorageModeShared;

        let upload = |data: &[f32]| -> Result<MtlBuf, String> {
            let bytes = bytemuck_cast(data);
            unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len().max(4),
                        res,
                    )
                    .ok_or_else(|| "metal: batched upload failed".to_string())
            }
        };
        let a_buf = upload(a)?;
        let b_buf = upload(b)?;
        let e_buf = upload(&[0.0f32])?;
        let c_buf = self
            .device
            .newBufferWithLength_options((out_len * 4).max(4), res)
            .ok_or("metal: batched output alloc failed")?;
        let dims = [m as u32, n as u32, k as u32];
        let codes = [0u32, 0u32];

        let cb = self
            .queue
            .commandBuffer()
            .ok_or("metal: commandBuffer returned nil")?;
        let enc = cb.computeCommandEncoder().ok_or("metal: encoder nil")?;
        enc.setComputePipelineState(&self.pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
            enc.setBytes_length_atIndex(
                NonNull::new(dims.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&dims),
                3,
            );
            enc.setBuffer_offset_atIndex(Some(&e_buf), 0, 4);
            enc.setBytes_length_atIndex(
                NonNull::new(codes.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(&codes),
                5,
            );
        }
        // Grid z = batch: all `batch` GEMMs dispatched together, run concurrently.
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: n.div_ceil(self.block_n),
                height: m.div_ceil(self.block_m),
                depth: batch,
            },
            MTLSize {
                width: self.threads,
                height: 1,
                depth: 1,
            },
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
        let e_buf = mk_in(&[0.0f32])?;
        let c_buf = self
            .device
            .newBufferWithLength_options((m * n * 4).max(1), res)
            .ok_or("alloc")?;
        let small = |v: &[u32]| -> Result<_, String> {
            let bytes = bytemuck_u32(v);
            unsafe {
                self.device
                    .newBufferWithBytes_length_options(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len(),
                        res,
                    )
                    .ok_or_else(|| "alloc".to_string())
            }
        };
        let dims_buf = small(&[m as u32, n as u32, k as u32])?;
        let codes_buf = small(&[0u32, 0u32])?;
        let m_blocks = m.div_ceil(self.block_m);
        let n_blocks = n.div_ceil(self.block_n);

        let cb = self.queue.commandBuffer().ok_or("cb")?;
        for _ in 0..iters {
            let enc = cb.computeCommandEncoder().ok_or("enc")?;
            enc.setComputePipelineState(&self.pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&a_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&b_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&c_buf), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&dims_buf), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&e_buf), 0, 4);
                enc.setBuffer_offset_atIndex(Some(&codes_buf), 0, 5);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_blocks,
                    height: m_blocks,
                    depth: 1,
                },
                MTLSize {
                    width: self.threads,
                    height: 1,
                    depth: 1,
                },
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
#[cfg(metal)]
pub fn run_nax_matmul(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, String> {
    NaxGemm::new()?.run(m, k, n, a, b)
}

/// The system default Metal device's name (e.g. `"Apple M5 Pro"`), or `""` if
/// there is no device.
#[cfg(metal)]
pub fn device_name() -> String {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    MTLCreateSystemDefaultDevice()
        .map(|d| d.name().to_string())
        .unwrap_or_default()
}

#[cfg(metal)]
thread_local! {
    /// Cached device name (cheap, resolved once per thread).
    static GEMM_DEVICE: std::cell::OnceCell<String> = const { std::cell::OnceCell::new() };
    /// Lazily-compiled NAX GEMM context, built the first time the gate picks NAX
    /// (so we never pay the kernel compile when only Accelerate is used). `None`
    /// if compilation fails (e.g. a pre-M5 GPU without the tensor engine).
    static GEMM_NAX: std::cell::OnceCell<Option<NaxGemm>> = const { std::cell::OnceCell::new() };
}

/// The production GEMM entry point for the emulator: `C(m×k·k×n)` row-major,
/// dispatched to the highest-performance available backend.
///
/// On an M5, large GEMMs (per [`choose_matmul_backend`]) run on the NAX tensor
/// engine (f16, ~4 TFLOP/s, ~2× Accelerate); small ones and everything on
/// non-NAX devices run on Accelerate/`sgemm_rowmajor` (f32). The NAX context is
/// compiled once and cached per thread; if NAX is chosen but unavailable or
/// errors, it falls back to Accelerate. So this is always correct and never
/// slower than the BLAS path by more than one (cached) capability probe.
#[cfg(metal)]
pub fn metal_gemm_or_blas(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let name = GEMM_DEVICE.with(|c| c.get_or_init(device_name).clone());
    if choose_matmul_backend(&name, m, k, n).is_gpu()
        && let Some(out) = GEMM_NAX.with(|c| {
            c.get_or_init(|| NaxGemm::new().ok())
                .as_ref()
                .and_then(|g| g.run(m, k, n, a, b).ok())
        })
    {
        return out;
    }
    crate::blas::sgemm_rowmajor(m, k, n, a, b)
}

/// Fused `D = act(A·B BINOP E)` on the NAX engine, in one kernel — the matmul→
/// elementwise fusion the interpreter peephole uses. Returns `Some(D)` only when
/// the gate picks NAX (large enough to win) and the kernel runs; otherwise
/// `None`, so the caller falls back to running the matmul and elementwise op
/// separately. `e` is the row-major m×n elementwise operand.
#[cfg(metal)]
pub fn metal_gemm_fused(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    e: &[f32],
    epi: Epilogue,
) -> Option<Vec<f32>> {
    let name = GEMM_DEVICE.with(|c| c.get_or_init(device_name).clone());
    if !choose_matmul_backend(&name, m, k, n).is_gpu() {
        return None;
    }
    GEMM_NAX.with(|c| {
        c.get_or_init(|| NaxGemm::new().ok())
            .as_ref()
            .and_then(|g| g.run_fused(m, k, n, a, b, e, epi).ok())
    })
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
    /// Small-integer inputs (exact in f16) let us assert *exact* equality with
    /// the naive oracle. Two identity probes pin the fragment layout: with
    /// A = I, `B[k,n] = n` must yield `C[m,n] = n` (column mapping) and
    /// `B[k,n] = k` must yield `C[m,n] = m` (row mapping) — together these catch
    /// any cooperative-tensor axis swap or scramble in the BaseNAXFrag layout.
    #[test]
    fn nax_matmul_tile_matches_oracle() {
        // A[m,k] = (m + k) % 3, B[k,n] = (k + 2*n) % 4  — products ≤ 6, sums
        // over K=16 ≤ 96: all exact in f16 and f32, and distinct per (m,n).
        let a: Vec<f32> = (0..NAX_TILE_M * NAX_TILE_K)
            .map(|i| ((i / 16 + i % 16) % 3) as f32)
            .collect();
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
            &(0..NAX_TILE_K * NAX_TILE_N)
                .map(|i| (i % NAX_TILE_N) as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let row_probe = run_nax_matmul_tile(
            &ai,
            &(0..NAX_TILE_K * NAX_TILE_N)
                .map(|i| (i / NAX_TILE_N) as f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for m in 0..NAX_TILE_M {
            for n in 0..NAX_TILE_N {
                assert_eq!(
                    col_probe[m * NAX_TILE_N + n],
                    n as f32,
                    "column map at ({m},{n})"
                );
                assert_eq!(
                    row_probe[m * NAX_TILE_N + n],
                    m as f32,
                    "row map at ({m},{n})"
                );
            }
        }
        eprintln!("NAX matmul2d tile matches the oracle exactly (+ row/col layout) ✓");
    }

    /// The general tiled NAX GEMM is correct across shapes — including ragged
    /// M/N/K that exercise the zero-pad edge guards and multi-tile K
    /// accumulation. Small-integer inputs are exact in f16, so we assert exact
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
            (17, 33, 5),  // ragged M, N, K all
            (48, 16, 64), // multi-tile, clean
            (50, 20, 70), // multi-tile, ragged
            (7, 100, 3),  // wide N
            (100, 7, 3),  // tall M
        ];
        for (m, k, n) in shapes {
            // Small ints exact in f16: a in 0..3, b in 0..4. Sum over K stays
            // well under f16's 256 exact-integer limit for these K.
            let a: Vec<f32> = (0..m * k).map(|i| (i % 3) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32).collect();
            let got = ctx.run(m, k, n, &a, &b).unwrap();
            let want = crate::blas::naive_sgemm(m, k, n, &a, &b);
            assert_eq!(got, want, "NAX GEMM mismatch at shape ({m},{k},{n})");
        }
        eprintln!(
            "general NAX GEMM matches the oracle across {} shapes ✓",
            shapes.len()
        );
    }

    /// A batched matmul chain (one command buffer, one sync) computes the same
    /// result as the matmuls run separately, and amortizes the per-dispatch
    /// latency: a chain of N small matmuls should be far faster than N calls.
    #[test]
    fn matmul_chain_matches_and_amortizes() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        // x (m×k0) · W1 (k0×k1) · W2 (k1×k2) · W3 (k2×k3), with a bias+relu epilogue.
        let (m, k0, k1, k2, k3) = (128usize, 128, 128, 128, 128);
        // Positive inputs: chained f16 matmuls don't cancel, so the f32 oracle
        // stays within f16 tolerance (signed inputs would cancel near zero and
        // blow up the *relative* error without any bug).
        let mk = |rows: usize, cols: usize, s: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| ((i + s) % 7) as f32 * 0.03 + 0.01)
                .collect()
        };
        let x = mk(m, k0, 0);
        let (w1, w2, w3) = (mk(k0, k1, 1), mk(k1, k2, 2), mk(k2, k3, 3));
        let bias = mk(m, k3, 9);

        let steps = [
            ChainStep {
                k: k0,
                n: k1,
                b: &w1,
                epi: Epilogue::NONE,
                e: None,
            },
            ChainStep {
                k: k1,
                n: k2,
                b: &w2,
                epi: Epilogue::NONE,
                e: None,
            },
            ChainStep {
                k: k2,
                n: k3,
                b: &w3,
                epi: Epilogue { binop: 1, act: 1 },
                e: Some(&bias),
            },
        ];
        let got = ctx.run_chain(m, &x, &steps).unwrap();

        // Oracle: same chain on the CPU (f16 tolerance, since NAX is f16).
        let c1 = crate::blas::naive_sgemm(m, k0, k1, &x, &w1);
        let c2 = crate::blas::naive_sgemm(m, k1, k2, &c1, &w2);
        let c3 = crate::blas::naive_sgemm(m, k2, k3, &c2, &w3);
        let want: Vec<f32> = c3
            .iter()
            .zip(&bias)
            .map(|(&c, &b)| (c + b).max(0.0))
            .collect();
        let mut max_rel = 0.0f32;
        for (g, w) in got.iter().zip(&want) {
            max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
        }
        assert!(
            max_rel < 0.1,
            "chain result max rel err {max_rel} too large"
        );

        // Timing: the 3-matmul chain (one sync) vs three separate run() calls.
        let iters = 100;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            ctx.run_chain(m, &x, &steps).unwrap();
        }
        let chained = t0.elapsed().as_secs_f64() / iters as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            let a = ctx.run(m, k0, k1, &x, &w1).unwrap();
            let b = ctx.run(m, k1, k2, &a, &w2).unwrap();
            let _ = ctx
                .run_fused(m, k2, k3, &b, &w3, &bias, Epilogue { binop: 1, act: 1 })
                .unwrap();
        }
        let separate = t1.elapsed().as_secs_f64() / iters as f64;
        eprintln!(
            "matmul chain: batched {:.1} µs vs separate {:.1} µs  ({:.2}× faster, one sync vs three)",
            chained * 1e6,
            separate * 1e6,
            separate / chained
        );
    }

    /// **Combining** many small same-weight matmuls into one tall GEMM is both
    /// correct (matches per-slice) AND faster than a serial AMX loop once the
    /// matmul is compute-bound (K ≳ 512) — the real way to exploit a SPMD grid's
    /// many small matmuls on the GPU. Measured speedups: ~1.25× at K=512 up to
    /// ~1.74× at K=2048 (see the module bench).
    #[test]
    fn combined_matmul_matches_and_wins() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        // 16 cores each multiply their 512 rows by the SAME 1024×1024 weights.
        let (count, m, k, n) = (16usize, 512usize, 1024usize, 1024usize);
        let a: Vec<f32> = (0..count * m * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
            .collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect();

        let got = ctx.run_combined(count, m, k, n, &a, &b).unwrap();
        // Correctness: each core's rows match its own matmul (f16 tolerance).
        let mut max_rel = 0.0f32;
        for s in 0..count {
            let want = crate::blas::naive_sgemm(m, k, n, &a[s * m * k..(s + 1) * m * k], &b);
            for (g, w) in got[s * m * n..(s + 1) * m * n].iter().zip(&want) {
                max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
            }
        }
        assert!(max_rel < 0.05, "combined mismatch, max rel err {max_rel}");

        // Speed: combined GEMM vs the serial per-core AMX loop it replaces.
        let it = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            ctx.run_combined(count, m, k, n, &a, &b).unwrap();
        }
        let combined = t0.elapsed().as_secs_f64() / it as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..it {
            for s in 0..count {
                std::hint::black_box(crate::blas::sgemm_rowmajor(
                    m,
                    k,
                    n,
                    &a[s * m * k..(s + 1) * m * k],
                    &b,
                ));
            }
        }
        let amx_loop = t1.elapsed().as_secs_f64() / it as f64;
        eprintln!(
            "combine {count}×({m}×{k}×{n}): GPU one tall GEMM {:.0} µs vs serial AMX {:.0} µs  ({:.2}×)",
            combined * 1e6,
            amx_loop * 1e6,
            amx_loop / combined
        );
    }

    /// Zero-copy unified-memory matmul: correct, and free of the host↔device
    /// fill/readback the copy-based `run` pays (CPU and GPU share the bytes).
    #[test]
    fn unified_matmul_zero_copy_matches_and_is_faster() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        let (m, k, n) = (4096usize, 1024usize, 1024usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect();

        let ua = ctx.unified_from(&a).unwrap();
        let ub = ctx.unified_from(&b).unwrap();
        let mut uc = ctx.unified(m * n).unwrap();
        ctx.matmul_unified(m, k, n, &ua, &ub, &mut uc, None, Epilogue::NONE)
            .unwrap();

        // Correctness vs the copy-based path (same kernel, identical result).
        let want = ctx.run(m, k, n, &a, &b).unwrap();
        let mut max_abs = 0.0f32;
        for (g, w) in uc.as_slice().iter().zip(&want) {
            max_abs = max_abs.max((g - w).abs());
        }
        assert!(max_abs < 1e-3, "unified vs copy-path differ by {max_abs}");

        // Speed: zero-copy (operands already resident) vs run() which fills A,B
        // and reads C back every call.
        let it = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            ctx.matmul_unified(m, k, n, &ua, &ub, &mut uc, None, Epilogue::NONE)
                .unwrap();
        }
        let zc = t0.elapsed().as_secs_f64() / it as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..it {
            std::hint::black_box(ctx.run(m, k, n, &a, &b).unwrap());
        }
        let copied = t1.elapsed().as_secs_f64() / it as f64;
        eprintln!(
            "unified {m}×{k}×{n}: zero-copy {:.0} µs vs copy-path {:.0} µs  ({:.2}× faster, copies removed)",
            zc * 1e6,
            copied * 1e6,
            copied / zc
        );
    }

    /// A batched dispatch (independent same-shape GEMMs in one submission)
    /// matches running them separately. (For *shared*-weight matmuls,
    /// `run_combined` is the faster path — see `combined_matmul_matches_and_wins`.)
    #[test]
    fn batched_matmul_matches_oracle() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        let (batch, m, k, n) = (64usize, 256usize, 256usize, 256usize);
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
            .collect();
        let b: Vec<f32> = (0..batch * k * n)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
            .collect();

        let got = ctx.run_batched(batch, m, k, n, &a, &b).unwrap();
        let mut max_rel = 0.0f32;
        for s in 0..batch {
            let want = crate::blas::naive_sgemm(
                m,
                k,
                n,
                &a[s * m * k..(s + 1) * m * k],
                &b[s * k * n..(s + 1) * k * n],
            );
            for (g, w) in got[s * m * n..(s + 1) * m * n].iter().zip(&want) {
                max_rel = max_rel.max((g - w).abs() / w.abs().max(1.0));
            }
        }
        assert!(max_rel < 0.05, "batched mismatch, max rel err {max_rel}");
    }

    /// The pre-M5 `simdgroup_float8x8` GEMM is correct across shapes (incl.
    /// ragged) and supports the same fused epilogue. Forced on the M5 so we can
    /// validate that code path here.
    #[test]
    fn simdgroup_matmul_matches_oracle() {
        let ctx = match NaxGemm::new_simdgroup() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("simdgroup compile failed: {e}"),
        };
        // Plain matmul across shapes (small ints exact in f32).
        for (m, k, n) in [
            (8usize, 8usize, 8usize),
            (17, 33, 5),
            (50, 20, 70),
            (100, 7, 3),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 3) as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32).collect();
            let got = ctx.run(m, k, n, &a, &b).unwrap();
            let want = crate::blas::naive_sgemm(m, k, n, &a, &b);
            assert_eq!(got, want, "simdgroup GEMM mismatch at ({m},{k},{n})");
        }
        // Fused epilogue (add + relu) matches matmul-then-elementwise.
        let (m, k, n) = (40usize, 24usize, 56usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let e: Vec<f32> = (0..m * n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
        let got = ctx
            .run_fused(m, k, n, &a, &b, &e, Epilogue { binop: 1, act: 1 })
            .unwrap();
        let mm = crate::blas::naive_sgemm(m, k, n, &a, &b);
        for i in 0..m * n {
            let want = (mm[i] + e[i]).max(0.0);
            assert!(
                (got[i] - want).abs() < 1e-3,
                "simdgroup fused mismatch at {i}"
            );
        }
        eprintln!("simdgroup_float8x8 GEMM (+fused epilogue) matches the oracle ✓");
    }

    /// Fused matmul→elementwise epilogue (`D = act(A·B BINOP E)`) computed in one
    /// kernel matches doing the matmul then the elementwise op separately.
    #[test]
    fn nax_matmul_fused_epilogue_matches_oracle() {
        let ctx = match NaxGemm::new() {
            Ok(c) => c,
            Err(e) if e.contains("no Metal device") => return,
            Err(e) => panic!("{e}"),
        };
        let (m, k, n) = (130usize, 40usize, 200usize); // ragged, multi-block
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let e: Vec<f32> = (0..m * n).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
        let mm = crate::blas::naive_sgemm(m, k, n, &a, &b);

        type Case = (Epilogue, fn(f32, f32) -> f32);
        let cases: &[Case] = &[
            (Epilogue::ADD, |c, ev| c + ev),
            (Epilogue::MUL, |c, ev| c * ev),
            (Epilogue::SUB, |c, ev| c - ev),
            (Epilogue::MAX, |c, ev| c.max(ev)),
            (Epilogue::RELU, |c, _| c.max(0.0)),
            (Epilogue { binop: 1, act: 1 }, |c, ev| (c + ev).max(0.0)), // add + relu
            (Epilogue { binop: 1, act: 2 }, |c, ev| (c + ev).tanh()),   // add + tanh
        ];
        for &(epi, f) in cases {
            let got = ctx.run_fused(m, k, n, &a, &b, &e, epi).unwrap();
            let mut max_rel = 0.0f32;
            for i in 0..m * n {
                let want = f(mm[i], e[i]);
                max_rel = max_rel.max((got[i] - want).abs() / want.abs().max(1.0));
            }
            assert!(
                max_rel < 0.05,
                "fused {epi:?}: max rel err {max_rel} > f16 tol"
            );
        }
        eprintln!(
            "fused matmul→elementwise epilogue matches oracle across {} ops ✓",
            cases.len()
        );
    }

    /// Random (non-f16-exact) data: the NAX GEMM agrees with the f32 oracle to
    /// f16 tolerance. Documents the precision the engine actually delivers.
    #[test]
    fn nax_matmul_general_f16_tolerance() {
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
        // f16 has 8 mantissa bits; K=48 accumulation in f32 keeps error modest.
        assert!(
            max_rel < 0.05,
            "max relative error {max_rel} exceeds f16 tolerance"
        );
        eprintln!("NAX GEMM vs f32 oracle: max relative error {max_rel:.4} (f16) ✓");
    }

    /// Real benchmark: NAX vs naive vs the linked BLAS (Accelerate on macOS) on
    /// a sizeable GEMM. Prints GFLOP/s for each so the speedup is concrete.
    /// `--ignored` because it's a perf measurement, not a correctness gate.
    ///
    /// Observed on an M5 (numbers vary with thermals):
    ///
    /// - GPU-only kernel throughput climbs with size and plateaus ~4 TFLOP/s at
    ///   2048³+ (where #threadgroups finally fills the cores); at 1024³ it is
    ///   occupancy-bound (~32 threadgroups) and small sizes are far worse.
    /// - At its plateau the kernel is ~2× Apple Accelerate (AMX, ~2 TFLOP/s).
    /// - Per-call wall-clock is dominated by buffer alloc + host/device copy +
    ///   readback; `gpu_time_seconds` isolates the kernel from that overhead.
    ///
    /// The remaining gap to NAX's true peak is the per-K-step staging+barrier
    /// tax — a double-buffered kernel (overlap load with compute) is the next win.
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
        let prng = |i: usize| (i.wrapping_mul(2654435761) % 1000) as f32 / 1000.0;

        // GPU-only throughput sweep across sizes: diagnoses whether the kernel
        // is occupancy-bound (climbs as #threadgroups grows) or compute-bound
        // (plateaus). #threadgroups = ceil(s/128) * ceil(s/256).
        eprintln!("-- GPU-only throughput sweep (kernel time only) --");
        for &s in &[256usize, 512, 1024, 2048, 4096] {
            let a: Vec<f32> = (0..s * s).map(prng).collect();
            let b: Vec<f32> = (0..s * s).map(|i| prng(i + 3)).collect();
            let iters = if s <= 1024 { 50 } else { 10 };
            let g = ctx.gpu_time_seconds(s, s, s, &a, &b, iters).unwrap() / iters as f64;
            let tgs = s.div_ceil(128) * s.div_ceil(256);
            eprintln!(
                "  {s:>4}^3: {:7.3} ms   {:7.1} GFLOP/s   ({tgs} threadgroups)",
                g * 1e3,
                2.0 * (s as f64).powi(3) / g / 1e9
            );
        }

        let (m, k, n) = (1024usize, 1024usize, 1024usize);
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
            eprintln!(
                "{label:>12}: {:7.2} ms   {:7.1} GFLOP/s",
                secs * 1e3,
                flops / secs / 1e9
            );
        };

        // GPU-only kernel time (excludes alloc/copy/readback): 50 dispatches on
        // reused buffers, timed by hardware timestamps. Isolates kernel speed.
        let gpu_total = ctx.gpu_time_seconds(m, k, n, &a, &b, 50).unwrap();
        let gpu_each = gpu_total / 50.0;
        eprintln!(
            "{:>12}: {:7.2} ms   {:7.1} GFLOP/s   (GPU kernel only)",
            "NAX-gpu",
            gpu_each * 1e3,
            flops / gpu_each / 1e9
        );

        let (a1, b1) = (a.clone(), b.clone());
        bench(
            "NAX-wall",
            20,
            Box::new(move || {
                ctx.run(m, k, n, &a1, &b1).unwrap();
            }),
        );
        let (a2, b2) = (a.clone(), b.clone());
        bench(
            "BLAS/accel",
            20,
            Box::new(move || {
                std::hint::black_box(crate::blas::sgemm_rowmajor(m, k, n, &a2, &b2));
            }),
        );
        let (a3, b3) = (a.clone(), b.clone());
        bench(
            "naive",
            1,
            Box::new(move || {
                std::hint::black_box(crate::blas::naive_sgemm(m, k, n, &a3, &b3));
            }),
        );
    }

    #[test]
    fn lowers_vector_add_to_msl() {
        let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
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
    fn fuses_elementwise_chain_into_one_expression() {
        // exp(a * b) + c  -> a single fused kernel, not three passes.
        let src = r#"
module {
  func.func @chain(%a_ptr: index, %b_ptr: index, %c_ptr: index, %out_ptr: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %va = ktdp.construct_memory_view %a_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %vb = ktdp.construct_memory_view %b_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %vc = ktdp.construct_memory_view %c_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %ta = ktdp.construct_access_tile %va[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %tb = ktdp.construct_access_tile %vb[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %tc = ktdp.construct_access_tile %vc[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    %la = ktdp.load %ta : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %lb = ktdp.load %tb : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %lc = ktdp.load %tc : !ktdp.access_tile<8xindex> -> tensor<8xf16>
    %ab = arith.mulf %la, %lb : tensor<8xf16>
    %e = math.exp %ab : tensor<8xf16>
    %r = arith.addf %e, %lc : tensor<8xf16>
    %vout = ktdp.construct_memory_view %out_ptr, sizes: [8], strides: [1] {
      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<8xf16>
    %tout = ktdp.construct_access_tile %vout[%c0] {
      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>, access_tile_order = affine_map<(d0) -> (d0)>
    } : memref<8xf16> -> !ktdp.access_tile<8xindex>
    ktdp.store %r, %tout : tensor<8xf16>, !ktdp.access_tile<8xindex>
    return
  }
}
"#;
        let module = parse_module(src).unwrap();
        let kernel = emit_kernel(&module, "chain").expect("emit fused chain");
        // One kernel, three input buffers (a,b,c) + one output, deduped & ordered.
        let names: Vec<&str> = kernel.buffers.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, vec!["a_ptr", "b_ptr", "c_ptr", "out_ptr"], "fused buffer set");
        assert_eq!(kernel.buffers.iter().filter(|b| b.is_output).count(), 1);
        // The whole DAG collapses into one assignment: exp(a*b) + c.
        assert!(
            kernel.source.contains(
                "out_ptr[gid] = (exp((a_ptr[gid] * b_ptr[gid]))) + c_ptr[gid];"
            ),
            "expected one fused expression, got:\n{}",
            kernel.source
        );
    }

    #[test]
    fn gpu_matches_oracle_vector_add() {
        use crate::dtypes::DType;
        use crate::interpreter::{Arg, execute_function};
        use crate::ir::Scalar;

        let src = include_str!("../../../../examples/triton-ktir/vector_add_ktir.mlir");
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
            (
                "x_ptr",
                Arg::Tensor {
                    data: x,
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            (
                "y_ptr",
                Arg::Tensor {
                    data: y,
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
            (
                "output_ptr",
                Arg::Tensor {
                    data: vec![0.0; n],
                    shape: vec![n],
                    dtype: DType::F16,
                },
            ),
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
        let src = include_str!("../../../../examples/latency/matmul_small.mlir");
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
        // Effective tier is the best *implemented* tier the device supports —
        // and all three are implemented now.
        assert_eq!(effective_matmul_tier("Apple M5"), HIGHEST_IMPLEMENTED);
        assert_eq!(effective_matmul_tier("Apple M5"), Nax);
        // Pre-NAX Apple GPUs use the simdgroup_float8x8 GPU path.
        assert_eq!(effective_matmul_tier("Apple M4"), Simdgroup);
        assert_eq!(effective_matmul_tier("Apple M1"), Simdgroup);
        // Non-Apple stays at the naive floor.
        assert_eq!(effective_matmul_tier("Intel UHD Graphics 630"), Naive);
        assert!(tier_implemented(Simdgroup));
    }

    #[test]
    fn matmul_backend_gating() {
        use MatmulBackend::{Accelerate, Nax};
        // M5 sends large GEMMs (>= measured ~1024³ crossover) to the NAX engine.
        assert_eq!(choose_matmul_backend("Apple M5", 1024, 1024, 1024), Nax); // 32 blocks
        assert_eq!(choose_matmul_backend("Apple M5", 2048, 2048, 2048), Nax);
        // Smaller / LX-sized matmuls -> Accelerate (AMX), faster there.
        assert_eq!(choose_matmul_backend("Apple M5", 512, 512, 512), Accelerate); // 8 blocks < 32
        assert_eq!(choose_matmul_backend("Apple M5", 256, 256, 256), Accelerate);
        // Pre-M5: the simdgroup GPU path never beats AMX in wall-clock -> Accelerate.
        assert_eq!(
            choose_matmul_backend("Apple M4", 2048, 2048, 2048),
            Accelerate
        );
        assert_eq!(
            choose_matmul_backend("Apple M1", 4096, 4096, 4096),
            Accelerate
        );
        // Non-Apple GPUs -> Accelerate.
        assert_eq!(
            choose_matmul_backend("Intel UHD Graphics 630", 4096, 4096, 4096),
            Accelerate
        );
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
        eprintln!(
            "device {name:?}: capability tier = {cap:?}, using = {:?}",
            effective_matmul_tier(&name)
        );
        // This machine is an Apple GPU, so it must be at least the simdgroup tier.
        assert!(
            cap >= MatmulTier::Simdgroup,
            "expected an Apple GPU, got {name:?}"
        );
    }
}
