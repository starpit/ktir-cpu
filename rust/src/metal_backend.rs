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

use crate::ir::{IRFunction, IRModule, Operation};

/// Lower `func_name` to an MSL kernel string. Errors if the function isn't the
/// supported element-wise shape (with a message pointing at what tripped it).
pub fn emit_msl(module: &IRModule, func_name: &str) -> Result<String, String> {
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
    let elem_ty = buffer_dtype(&out_buf, f, &defs);

    // Inputs in first-seen order; the output buffer last. (De-dup: a buffer may
    // be both read and written, though vector_add's aren't.)
    let mut buffers: Vec<(String, bool)> = Vec::new(); // (name, is_output)
    for b in collect_input_buffers(compute, &defs) {
        if !buffers.iter().any(|(n, _)| *n == b) {
            buffers.push((b, false));
        }
    }
    buffers.push((out_buf, true));

    Ok(render_kernel(func_name, &buffers, &elem_ty, &expr))
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

/// Element type of a buffer, read from the `construct_memory_view` dtype that
/// produced it. Defaults to `half` (f16) — the common KTIR tile dtype.
fn buffer_dtype(buf: &str, f: &IRFunction, _defs: &HashMap<String, &Operation>) -> String {
    // Find a construct_memory_view whose pointer operand is this buffer.
    for op in &f.operations {
        if op.op_type == "ktdp.construct_memory_view"
            && op.operands.first().map(|p| strip(p)) == Some(buf)
            && let Some(crate::ir::Attr::Str(dt)) = op.attributes.get("dtype")
        {
            return msl_type(dt);
        }
    }
    "half".to_string()
}

fn msl_type(ktir_dtype: &str) -> String {
    match ktir_dtype {
        "f16" | "fp16" | "float16" => "half",
        "f32" | "float32" => "float",
        "i32" | "si32" | "index" => "int",
        "i64" | "si64" => "long",
        _ => "half",
    }
    .to_string()
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

fn render_kernel(name: &str, buffers: &[(String, bool)], elem_ty: &str, expr: &str) -> String {
    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
    s.push_str(&format!("kernel void {name}(\n"));
    for (i, (buf, is_out)) in buffers.iter().enumerate() {
        let qual = if *is_out { "device" } else { "device const" };
        s.push_str(&format!(
            "    {qual} {elem_ty}* {buf} [[buffer({i})]],\n"
        ));
    }
    s.push_str("    uint gid [[thread_position_in_grid]]\n) {\n");
    // The output buffer is the last entry.
    let out = &buffers.last().unwrap().0;
    s.push_str(&format!("    {out}[gid] = {expr};\n"));
    s.push_str("}\n");
    s
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
    fn rejects_non_elementwise() {
        // matmul_small has a linalg.matmul -> not lowerable in slice 1.
        let src = include_str!("../../examples/latency/matmul_small.mlir");
        if let Ok(module) = parse_module(src) {
            let name = module.functions.keys().next().unwrap().clone();
            assert!(emit_msl(&module, &name).is_err());
        }
    }
}
