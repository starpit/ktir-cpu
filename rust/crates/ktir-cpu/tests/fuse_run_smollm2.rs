// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! REAL-MODEL fuse-then-run e2e (fusion increment 2): take the SmolLM2-135M KTIR
//! bundle scratchy emits, build one `ProgramSpec` from `manifest.json`, run
//! `ktir_optimizer::fusion::fuse_program` to collapse all 452 nodes into a single
//! function whose forwardable HBM intermediates become SSA / `tensor.extract_slice`,
//! then execute that fused function through the SAME interpreter and compare to
//! golden. This is the pressure test for increment 2: the model's tiled edges are
//! `construct_access_tile %view[%c0, %k7]` loads INSIDE an `scf.for` K-loop, so it
//! exercises the region-aware analysis, the nested extract_slice rewrite, and the
//! scf.for attribute renaming all at once.
//!
//! The bundle is machine-specific and not in the repo, so the test SKIPS when
//! absent. `--ignored` because it runs a whole model.

use ktir_cpu::dtypes::DType;
use ktir_cpu::interpreter::{Arg, execute_function};
use ktir_cpu::ir::IRModule;
use ktir_cpu::parser::parse_module;
use ktir_optimizer::fusion::{Binding, NodeSpec, ProgramSpec, fuse_program};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn bundle_dir() -> Option<PathBuf> {
    bundle_dir_named("smollm2-135m")
}

fn bundle_dir_named(model: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let dir = PathBuf::from(home).join(".cache/cudaforge/ktir").join(model);
    dir.join("manifest.json").is_file().then_some(dir)
}

/// A whole bundle fused into one function, plus the metadata to run it.
struct Fused {
    func: ktir_cpu::ir::IRFunction,
    /// tensor id -> (rows, cols, is_source)
    shape: HashMap<u64, (usize, usize, bool)>,
    result_id: u64,
    mask_id: Option<u64>,
    n_nodes: usize,
}

/// Load a bundle's manifest + per-node MLIR, build the ProgramSpec, and fuse the
/// whole program into one function (decode or prefill — same path).
fn fuse_bundle(dir: &std::path::Path) -> Fused {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();

    let mut shape: HashMap<u64, (usize, usize, bool)> = HashMap::new();
    let mut sources: HashSet<u64> = HashSet::new();
    for t in manifest["tensors"].as_array().unwrap() {
        let id = t["id"].as_u64().unwrap();
        let is_src = t["is_source"].as_bool().unwrap_or(false);
        shape.insert(
            id,
            (
                t["rows"].as_u64().unwrap() as usize,
                t["cols"].as_u64().unwrap() as usize,
                is_src,
            ),
        );
        if is_src {
            sources.insert(id);
        }
    }
    let result_id = manifest["result"].as_u64().unwrap();
    let mask_id = manifest["attn_mask"].as_u64();
    if let Some(m) = mask_id {
        sources.insert(m);
    }

    let mut module = IRModule::default();
    let mut nodes: Vec<NodeSpec> = Vec::new();
    for node in manifest["nodes"].as_array().unwrap() {
        let func = node["fn"].as_str().unwrap().to_string();
        let mlir = node["mlir"].as_str().unwrap();
        let src = std::fs::read_to_string(dir.join(mlir)).unwrap();
        let parsed = parse_module(&src).unwrap_or_else(|e| panic!("parse {mlir}: {e}"));
        for (_, f) in parsed.functions {
            module.add_function(f);
        }
        let bindings = node["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| Binding {
                arg: format!("%{}", a["name"].as_str().unwrap()),
                tensor: a["tensor"].as_u64().unwrap(),
                is_output: a["is_output"].as_bool().unwrap_or(false),
            })
            .collect();
        nodes.push(NodeSpec { func, bindings });
    }
    let n_nodes = nodes.len();
    let spec = ProgramSpec {
        nodes,
        sources,
        results: HashSet::from([result_id]),
    };
    let func = fuse_program(&module, &spec).expect("fuse bundle");
    Fused { func, shape, result_id, mask_id, n_nodes }
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Parse `%t<id>_ptr` -> id. The fused function names every pointer arg this way.
fn tensor_id_of(arg: &str) -> u64 {
    arg.trim_start_matches('%')
        .trim_start_matches('t')
        .trim_end_matches("_ptr")
        .parse()
        .unwrap_or_else(|_| panic!("unexpected fused arg name {arg:?}"))
}

#[test]
#[ignore = "real-model fuse-then-run; needs the ~/.cache/cudaforge/ktir/smollm2-135m bundle. \
            Run with --ignored --nocapture"]
fn smollm2_135m_fused_matches_golden() {
    let Some(dir) = bundle_dir() else {
        eprintln!("SmolLM2 bundle absent — skipping");
        return;
    };
    let b = fuse_bundle(&dir);
    let (shape, result_id, mask_id, n_nodes) = (&b.shape, b.result_id, b.mask_id, b.n_nodes);
    let fused = b.func;
    let n_ops = fused.operations.len();
    let n_slices = fused
        .operations
        .iter()
        .filter(|o| o.op_type == "tensor.extract_slice")
        .count();
    // Count HBM round-trips remaining vs. what 452 unfused nodes would pay.
    let n_loads = count_op(&fused.operations, "ktdp.load");
    let n_stores = count_op(&fused.operations, "ktdp.store");
    let fused_ptrs = fused.arguments.len();
    eprintln!(
        "fused: {n_ops} ops, {fused_ptrs} pointer args (vs {} tensors), \
         {n_slices} extract_slice, {n_loads} ktdp.load, {n_stores} ktdp.store",
        shape.len()
    );

    // Provide a buffer for EVERY pointer arg the fused function still declares:
    // sources preloaded from t{id}.bin (mask = zeros), results + any
    // non-forwarded intermediate = zero-initialized (written then read within
    // the fused run).
    let mut args: Vec<(String, Arg)> = Vec::new();
    for (name, _) in &fused.arguments {
        let id = tensor_id_of(name);
        let (rows, cols, is_src) = shape[&id];
        let data = if is_src && Some(id) != mask_id {
            read_f32(&dir.join(format!("t{id}.bin")))
        } else {
            vec![0.0f32; rows * cols]
        };
        args.push((
            name.trim_start_matches('%').to_string(),
            Arg::Tensor { data, shape: vec![rows, cols], dtype: DType::F16 },
        ));
    }
    let arg_refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();

    let mut fused_module = IRModule::default();
    fused_module.add_function(fused);
    let out = execute_function(&fused_module, "fused", &arg_refs).expect("run fused SmolLM2");

    let got = &out
        .get(&format!("t{result_id}_ptr"))
        .expect("result tensor read back")
        .data;
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(got.len(), golden.len(), "result length");

    let mut max_abs = 0.0f32;
    let mut finite = 0usize;
    for (a, b) in got.iter().zip(&golden) {
        if a.is_finite() {
            finite += 1;
        }
        max_abs = max_abs.max((a - b).abs());
    }
    let total = got.len();
    eprintln!(
        "SmolLM2-135M FUSED ({n_nodes} nodes -> 1 fn): result vs golden \
         {finite}/{total} finite, max abs diff {max_abs:.4} (f16 compute)"
    );
    assert_eq!(finite, got.len(), "all result elements finite");
    // Same tolerance basis as the per-node oracle (e2e_smollm2): f16 compute +
    // GEMM reduction-order noise. Fusion only removes HBM round-trips, so the
    // fused result should track golden as closely as the unfused path.
    assert!(
        max_abs < 0.2,
        "fused result diverges from golden by {max_abs} — fusion changed semantics"
    );
}

/// PREFILL readiness: fuse the M=8 prefill bundle and confirm EVERY scf.for
/// K-loop is recognized as a single full-shape GEMM (the grid/token-parallel and
/// K-tiling decomposition collapses to one [8,k]@[k,n] matmul). This is the
/// proof that prefill is a first-class target, not a deferred one — the Metal
/// executor ignores the Spyre SPMD grid and reconstructs the whole GEMM.
#[cfg(metal)]
#[test]
#[ignore = "real-model prefill recognition; needs ~/.cache/cudaforge/ktir/smollm2-135m-prefill. \
            Run with --ignored --nocapture"]
fn prefill_matmul_loops_all_recognized() {
    let Some(dir) = bundle_dir_named("smollm2-135m-prefill") else {
        eprintln!("SmolLM2 prefill bundle absent — skipping");
        return;
    };
    let b = fuse_bundle(&dir);
    let (total, recognized) = ktir_cpu::metal_backend::count_matmul_loops(&b.func.operations);
    eprintln!(
        "prefill fused ({} nodes -> 1 fn): {} ops, {total} scf.for K-loops, \
         {recognized} recognized as GEMMs",
        b.n_nodes,
        b.func.operations.len()
    );
    assert!(total > 0, "expected matmul K-loops in the fused prefill function");
    assert_eq!(
        total, recognized,
        "every prefill K-loop must collapse to one GEMM (M=8) — {} unrecognized",
        total - recognized
    );
}

fn count_op(ops: &[ktir_cpu::ir::Operation], ty: &str) -> usize {
    ops.iter()
        .map(|o| (o.op_type == ty) as usize + count_op(&o.regions.concat(), ty))
        .sum()
}
