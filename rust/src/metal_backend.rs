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

/// The highest tier whose kernel codegen is implemented today. Rises to
/// `Simdgroup` then `Nax` as those kernel slices (+ the Metal 4 runtime) land;
/// until then the tiered selection degrades to the naive floor.
pub const HIGHEST_IMPLEMENTED: MatmulTier = MatmulTier::Naive;

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

/// The tier actually used for a device: its capability, capped at what we emit.
pub fn effective_matmul_tier(device_name: &str) -> MatmulTier {
    device_matmul_tier(device_name).min(HIGHEST_IMPLEMENTED)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_module;

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
        // Effective tier is capped at what we actually emit today.
        assert_eq!(effective_matmul_tier("Apple M5"), HIGHEST_IMPLEMENTED);
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
