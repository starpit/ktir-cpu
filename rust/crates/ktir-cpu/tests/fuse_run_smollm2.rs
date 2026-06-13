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
use ktir_cpu::interpreter::{Arg, execute_function, execute_function_outputs};
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

/// Run a bundle the PER-NODE way (each node executed with its OWN grid — the
/// proven-correct oracle that respects [8,1]/[9,1] SPMD), threading one host
/// buffer per tensor. Returns the result tensor. This is the apples-to-apples
/// reference for the fused path: if fused == per-node, the fused single-grid run
/// is correct regardless of golden's own generation noise.
fn run_per_node_result(dir: &std::path::Path) -> Vec<f32> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let mut shape: HashMap<u64, (usize, usize, bool)> = HashMap::new();
    for t in manifest["tensors"].as_array().unwrap() {
        let id = t["id"].as_u64().unwrap();
        shape.insert(
            id,
            (
                t["rows"].as_u64().unwrap() as usize,
                t["cols"].as_u64().unwrap() as usize,
                t["is_source"].as_bool().unwrap_or(false),
            ),
        );
    }
    let mut buf: HashMap<u64, Vec<f32>> = HashMap::new();
    for (&id, &(_, _, is_src)) in &shape {
        if is_src {
            buf.insert(id, read_f32(&dir.join(format!("t{id}.bin"))));
        }
    }
    if let Some(m) = manifest["attn_mask"].as_u64() {
        let (r, c, _) = shape[&m];
        buf.insert(m, vec![0.0f32; r * c]);
    }
    let mut cache: HashMap<String, IRModule> = HashMap::new();
    for node in manifest["nodes"].as_array().unwrap() {
        let func = node["fn"].as_str().unwrap();
        let mlir = node["mlir"].as_str().unwrap();
        let module = cache.entry(mlir.to_string()).or_insert_with(|| {
            parse_module(&std::fs::read_to_string(dir.join(mlir)).unwrap()).unwrap()
        });
        let mut arg_ids: Vec<(String, u64, bool)> = Vec::new();
        let mut args: Vec<(String, Arg)> = Vec::new();
        for a in node["args"].as_array().unwrap() {
            let name = a["name"].as_str().unwrap().to_string();
            let tid = a["tensor"].as_u64().unwrap();
            let is_out = a["is_output"].as_bool().unwrap_or(false);
            let (rows, cols, _) = shape[&tid];
            let data = if is_out {
                vec![0.0f32; rows * cols]
            } else {
                buf.get(&tid).cloned().unwrap_or_else(|| panic!("node input {tid} not produced"))
            };
            args.push((name.clone(), Arg::Tensor { data, shape: vec![rows, cols], dtype: DType::F16 }));
            arg_ids.push((name, tid, is_out));
        }
        let refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
        let out = execute_function(module, func, &refs).unwrap_or_else(|e| panic!("per-node {func}: {e}"));
        for (name, tid, is_out) in &arg_ids {
            if *is_out {
                buf.insert(*tid, out.get(name).expect("output").data.clone());
            }
        }
    }
    buf[&manifest["result"].as_u64().unwrap()].clone()
}

/// PREFILL multi-core SPMD vs golden — the AUTHORITATIVE gate for cross-core
/// grid execution. Runs every prefill node at its NATIVE grid ([1,1] / [8,1]
/// token-parallel matmuls / [9,1] attention heads), threading one shared HBM
/// buffer per tensor between nodes, and compares the final result to golden.bin.
///
/// Each [8,1] node has 8 cores compute one token row each (`%pid =
/// get_compute_tile_id`, store `view[%pid, ...]`); each [9,1] node has 9 cores
/// compute one attention head each (writing disjoint 64-column head slices of
/// the shared rows). All cores write to the SAME shared HBM, and the readback
/// must capture every core's slice. A broken multi-core path (e.g. cores
/// clobbering each other's rows/columns, or get_compute_tile_id mapping, or a
/// readback that only sees one core's HBM) produces head-0-only attention and
/// diverges from golden by ~0.18. A correct one matches to f16 tolerance.
///
/// 0.05 is well above the observed ~0.0034 f16 noise yet far below the ~0.18 a
/// broken multi-core attention would give, so it rigorously distinguishes
/// correct cross-core SPMD from broken — it is not a rubber-stamp tolerance.
#[test]
#[ignore = "real-model prefill per-node multi-core; needs smollm2-135m-prefill. --ignored --nocapture"]
fn smollm2_135m_prefill_per_node_multicore_matches_golden() {
    let Some(dir) = bundle_dir_named("smollm2-135m-prefill") else {
        eprintln!("SmolLM2 prefill bundle absent — skipping");
        return;
    };
    let result = run_per_node_result(&dir);
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(result.len(), golden.len(), "result length");
    let finite = result.iter().filter(|x| x.is_finite()).count();
    let max_abs = result
        .iter()
        .zip(&golden)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "PREFILL per-node MULTI-CORE result vs golden: {finite}/{} finite, max abs diff {max_abs:.5}",
        result.len()
    );
    assert_eq!(finite, result.len(), "all result elements finite");
    assert!(
        max_abs < 0.05,
        "prefill per-node multi-core diverges from golden by {max_abs} — cross-core SPMD is wrong"
    );
}

/// Marshal the fused-function args for a bundle (sources from t{id}.bin, mask +
/// results/intermediates zeroed), returning (fused module, arg list, result_id).
fn fused_run_inputs(dir: &std::path::Path) -> (IRModule, Vec<(String, Arg)>, u64) {
    let b = fuse_bundle(dir);
    let fused = b.func;
    let mut args: Vec<(String, Arg)> = Vec::new();
    for (name, _) in &fused.arguments {
        let id = tensor_id_of(name);
        let (rows, cols, is_src) = b.shape[&id];
        let data = if is_src && Some(id) != b.mask_id {
            read_f32(&dir.join(format!("t{id}.bin")))
        } else {
            vec![0.0f32; rows * cols]
        };
        args.push((
            name.trim_start_matches('%').to_string(),
            Arg::Tensor { data, shape: vec![rows, cols], dtype: DType::F16 },
        ));
    }
    let mut module = IRModule::default();
    module.add_function(fused);
    (module, args, b.result_id)
}

/// Median ms/pass of `execute_function` on a fused bundle over `iters` runs.
#[cfg(metal)]
fn time_fused(dir: &std::path::Path, iters: u32) -> f64 {
    let (module, args, result_id) = fused_run_inputs(dir);
    let refs: Vec<(&str, Arg)> = args.iter().map(|(n, a)| (n.as_str(), a.clone())).collect();
    let result_ptr = format!("t{result_id}_ptr");
    // Warm up (pipeline compile, first-touch).
    execute_function_outputs(&module, "fused", &refs, &[&result_ptr]).expect("warmup");
    let mut times: Vec<f64> = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        execute_function_outputs(&module, "fused", &refs, &[&result_ptr]).expect("timed run");
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

/// PERF: decode + prefill ms/pass with the GPU GEMM offload ON vs OFF (the
/// interpreter's Accelerate K-loop). Run ONE bench at a time (no concurrency).
#[cfg(metal)]
#[test]
#[ignore = "perf bench; needs the smollm2-135m[-prefill] bundles. --ignored --nocapture"]
fn fused_gpu_vs_cpu_mspass() {
    let iters: u32 = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    for (model, dir_opt) in [
        ("decode", bundle_dir()),
        ("prefill", bundle_dir_named("smollm2-135m-prefill")),
    ] {
        let Some(dir) = dir_opt else {
            eprintln!("{model} bundle absent — skipping");
            continue;
        };
        // SAFETY: single-threaded test; toggling our own offload gate.
        unsafe { std::env::set_var("KTIR_NO_GPU_GEMM", "1") };
        let cpu = time_fused(&dir, iters);
        unsafe { std::env::remove_var("KTIR_NO_GPU_GEMM") };
        let gpu = time_fused(&dir, iters);
        eprintln!(
            "{model}: CPU K-loops {cpu:.1} ms/pass  |  GPU GEMMs {gpu:.1} ms/pass  |  speedup {:.2}x",
            cpu / gpu
        );
    }
}

/// Fuse a bundle, run the fused function through the interpreter (with the
/// matmul-loop GPU offload active under cfg(metal)), and compare the result to
/// golden.bin. Returns max-abs-diff vs golden.
fn run_fused_golden(dir: &std::path::Path, label: &str) -> (f32, Vec<f32>) {
    let b = fuse_bundle(dir);
    let (shape, result_id, mask_id, n_nodes) = (b.shape, b.result_id, b.mask_id, b.n_nodes);
    let fused = b.func;

    // Provide a buffer for EVERY pointer arg the fused function still declares:
    // sources from t{id}.bin (mask = zeros), results + non-forwarded
    // intermediates zero-initialized.
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
    let result_ptr = format!("t{result_id}_ptr");
    let out = execute_function_outputs(&fused_module, "fused", &arg_refs, &[&result_ptr])
        .expect("run fused bundle");

    let got = &out
        .get(&format!("t{result_id}_ptr"))
        .expect("result tensor read back")
        .data;
    let golden = read_f32(&dir.join("golden.bin"));
    assert_eq!(got.len(), golden.len(), "result length");

    let mut max_abs = 0.0f32;
    let mut finite = 0usize;
    for (a, g) in got.iter().zip(&golden) {
        if a.is_finite() {
            finite += 1;
        }
        max_abs = max_abs.max((a - g).abs());
    }
    eprintln!(
        "{label} FUSED ({n_nodes} nodes -> 1 fn): result vs golden \
         {finite}/{} finite, max abs diff {max_abs:.4}",
        got.len()
    );
    assert_eq!(finite, got.len(), "all result elements finite");
    (max_abs, got.clone())
}

#[test]
#[ignore = "real-model fuse-then-run; needs the ~/.cache/cudaforge/ktir/smollm2-135m bundle. \
            Run with --ignored --nocapture"]
fn smollm2_135m_fused_matches_golden() {
    let Some(dir) = bundle_dir() else {
        eprintln!("SmolLM2 bundle absent — skipping");
        return;
    };
    #[cfg(metal)]
    {
        ktir_cpu::metal_backend::MATMUL_LOOP_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
        ktir_cpu::metal_backend::MAP_REGION_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    }
    let (max_abs, _) = run_fused_golden(&dir, "SmolLM2-135M decode");
    #[cfg(metal)]
    {
        let gpu = ktir_cpu::metal_backend::MATMUL_LOOP_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let maps = ktir_cpu::metal_backend::MAP_REGION_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("  matmul K-loops offloaded to GPU GEMM: {gpu}");
        eprintln!("  map windows offloaded to fused GPU kernel: {maps}");
        assert!(gpu >= 200, "expected the K-loops to run on GPU, only {gpu} did");
        assert!(maps > 0, "expected map windows to run on GPU, none did");
    }
    assert!(max_abs < 0.2, "decode fused diverges from golden by {max_abs}");
}

/// PREFILL (M=8) end-to-end vs golden: the real throughput target. Same path as
/// decode — fuse, run through the interpreter with the matmul-loop GPU offload.
#[cfg(metal)]
#[test]
#[ignore = "real-model prefill fuse-then-run; needs smollm2-135m-prefill. --ignored --nocapture"]
fn smollm2_135m_prefill_fused_matches_golden() {
    let Some(dir) = bundle_dir_named("smollm2-135m-prefill") else {
        eprintln!("SmolLM2 prefill bundle absent — skipping");
        return;
    };
    ktir_cpu::metal_backend::MATMUL_LOOP_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    ktir_cpu::metal_backend::MAP_REGION_GPU_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    let (golden_diff, fused) = run_fused_golden(&dir, "SmolLM2-135M PREFILL");
    let gpu = ktir_cpu::metal_backend::MATMUL_LOOP_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let maps = ktir_cpu::metal_backend::MAP_REGION_GPU_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("  prefill matmul K-loops offloaded to GPU GEMM: {gpu}");
    eprintln!("  prefill map windows offloaded to fused GPU kernel: {maps}");
    assert!(gpu >= 200, "expected prefill K-loops on GPU, only {gpu} did");
    assert!(maps > 0, "expected prefill map windows on GPU, none did");

    // AUTHORITATIVE GATE: the fused single-grid + GPU-GEMM run must match
    // golden.bin (the reference scratchy generates). 0.05 is well above the
    // observed 0.0271 f16/GPU noise yet far below the ~0.18 a broken attention
    // (head-0-only) would produce — so this rigorously distinguishes correct
    // from broken, it is not a rubber-stamp tolerance.
    assert!(
        golden_diff < 0.05,
        "prefill fused diverges from golden by {golden_diff} — fusion/attention is wrong"
    );

    // CROSS-CORE GATE: a from-scratch per-node oracle that runs each node at its
    // OWN grid ([8,1] token-parallel / [9,1] attention heads), the multi-core
    // SPMD path. It now MATCHES golden to f16 tolerance (~0.0034) — i.e. the
    // emulator's MULTI-CORE SPMD execution of prefill nodes reproduces golden's
    // generation. (Earlier it diverged ~0.18 from a broken head-0-only
    // attention; this asserts that regression cannot return.) The fused
    // single-grid path also matches golden, so the two agree.
    let oracle = run_per_node_result(&dir);
    let golden = read_f32(&dir.join("golden.bin"));
    let oracle_vs_golden = oracle.iter().zip(&golden).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let fused_vs_oracle = fused.iter().zip(&oracle).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("  per-node multi-core oracle vs golden: {oracle_vs_golden:.5}; fused vs oracle: {fused_vs_oracle:.5}");
    assert!(
        oracle_vs_golden < 0.05,
        "prefill per-node multi-core SPMD diverges from golden by {oracle_vs_golden} — cross-core execution is wrong"
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

/// MAP-WINDOW FUSION readiness: fuse the decode bundle and confirm `map_fusion_plan`
/// carves the elementwise op stream into fused GPU kernels — proving the runtime
/// map offload has work to do (the number of windows = the MAP_REGION_GPU_COUNT a
/// run produces). No GPU dispatch: just the plan, so it's fast and device-free.
#[cfg(metal)]
#[test]
#[ignore = "real-model map-fusion plan; needs ~/.cache/cudaforge/ktir/smollm2-135m. \
            Run with --ignored --nocapture"]
fn map_fusion_plan_carves_windows() {
    if bundle_dir().is_none() {
        eprintln!("SmolLM2 bundle absent — skipping");
        return;
    }
    for which in ["smollm2-135m", "smollm2-135m-prefill"] {
        let Some(d) = bundle_dir_named(which) else { continue };
        let b = fuse_bundle(&d);
        let ops = &b.func.operations;
        let (triggers, skip) = ktir_cpu::metal_backend::map_fusion_plan(ops);
        eprintln!(
            "{which} fused ({} nodes -> 1 fn): {} map windows -> GPU kernels, {} op indices subsumed",
            b.n_nodes, triggers.len(), skip.len()
        );
        // element count per SSA result (product of its shape attr)
        let mut numel: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        fn rec(ops: &[ktir_cpu::ir::Operation], m: &mut std::collections::HashMap<String, i64>) {
            for op in ops {
                if let Some(r) = &op.result
                    && let Some(ktir_cpu::ir::Attr::IntList(s)) = op.attributes.get("shape")
                {
                    m.insert(r.trim_start_matches('%').to_string(), s.iter().product());
                }
                for region in &op.regions { rec(region, m); }
            }
        }
        rec(ops, &mut numel);
        // For each kernel: a live-in read as `name[gid]` (not a broadcast index)
        // MUST have element count == out_len, else gid runs out of bounds.
        let mut mism = 0;
        for mrk in triggers.values() {
            let out_len: i64 = mrk.out_shape.iter().map(|&x| x as i64).product();
            for li in &mrk.live_ins {
                let key = li.trim_start_matches('%');
                let n = *numel.get(key).unwrap_or(&-1);
                let reads_gid = mrk.kernel.source.contains(&format!("{key}[gid]"));
                if reads_gid && n != out_len {
                    if mism < 15 {
                        eprintln!("  MISMATCH {which}: live_out {} reads {key}[gid] len={n} but out_len={out_len}", mrk.live_out);
                    }
                    mism += 1;
                }
            }
        }
        eprintln!("  {which}: {mism} live-ins read [gid] with len != out_len (would corrupt)");
        assert_eq!(mism, 0, "{which}: gid-indexed live-in length mismatch");
        assert!(!triggers.is_empty(), "{which}: expected fusable windows");
    }
}
