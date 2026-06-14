// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! RESIDENT GPU execution — the production serving path that keeps weights
//! resident across passes and segments.
//!
//! The prior [`crate::segmented::execute_segmented`] is correct but pays, on
//! EVERY pass, a full weight MARSHAL: each segment call allocates a fresh
//! [`SpyreMemoryHierarchy`] and re-encodes + re-writes every weight tensor's f16
//! bytes into a brand-new HBM (and on a multi-segment program — Llama prefill's
//! interleaved attention nodes — that marshal runs once per segment, many times
//! per pass). On Llama-1B (~2 GB of f16 weights) that is 2–4 s/pass of pure
//! data movement plus the alloc/free churn of 2 GB per pass — the regression
//! that made fused-Metal decode 8× SLOWER than CPU.
//!
//! [`ResidentExecutor`] fixes that structurally:
//!
//! * ONE persistent [`SpyreMemoryHierarchy`]. Every logical tensor is allocated
//!   an HBM stick ONCE, at construction, so the pointer the IR binds for a weight
//!   is STABLE across passes. Sources (weights / the attention mask / the input)
//!   are written once in [`ResidentExecutor::set_source`]; they are never
//!   re-marshaled.
//! * The GPU side is already resident: the thread-local `WEIGHT_CACHE` in
//!   `metal_backend` decodes+uploads each GEMM weight to a [`UnifiedBuffer`] at
//!   most once and serves every subsequent pass from it (keyed by a content
//!   fingerprint, so it can never serve a stale weight). With the HBM bytes now
//!   stable too, the whole weight working set is uploaded exactly once.
//! * PER PASS [`ResidentExecutor::run`] only (a) rewrites the changing program
//!   INPUT activation(s) in place, (b) zeroes the result / intermediate / scratch
//!   sticks (small — kB, not GB), and (c) runs each planned segment against the
//!   SAME persistent HBM via [`crate::interpreter::execute_function_in`] — no
//!   `SpyreMemoryHierarchy::new`, no `marshal_inputs`, no fresh 2 GB alloc.
//!   Intermediates stay in the one persistent HBM across segments; the fused
//!   K-loop GEMMs (NaxGemm on resident weight buffers), fused map-window MSL
//!   kernels, and reductions chain on it exactly as before, but with zero weight
//!   re-marshal in the loop.
//!
//! The attention islands (native segments) run at their native head-parallel
//! grid against the SAME persistent HBM — so an attention segment does NOT
//! trigger a full per-segment weight re-marshal either (the previous segmented
//! path re-marshaled every attention input into a fresh HBM per node).
//!
//! Correctness is identical to `execute_segmented`: the same segment plan, the
//! same per-op handlers and GPU offloads, the same f16 precision. The only
//! difference is WHERE the bytes live (one resident HBM vs a fresh one per call)
//! — so the golden results are unchanged.

use crate::dtypes::DType;
use crate::interpreter::{Arg, Output, TensorMeta, execute_function_in};
use crate::ir::{Attr, IRModule, Operation, Value};
use crate::memory::{STICK_BYTES, SpyreMemoryHierarchy};
use ktir_optimizer::fusion::{ProgramSpec, Segment, plan_segments};
use std::collections::HashMap;

/// Recover the logical tensor id from a fused pointer-arg name `%t<id>_ptr`.
fn tensor_id_of_arg(arg: &str) -> Result<u64, String> {
    arg.trim_start_matches('%')
        .trim_start_matches('t')
        .trim_end_matches("_ptr")
        .parse()
        .map_err(|_| format!("unexpected fused pointer-arg name {arg:?}"))
}

/// Normalize a caller key (`t<id>`, `%t<id>`, `%t<id>_ptr`, bare `<id>`) -> id.
fn tensor_id_of_key(key: &str) -> Result<u64, String> {
    let s = key.trim_start_matches('%');
    let s = s.strip_prefix('t').unwrap_or(s);
    let s = s.strip_suffix("_ptr").unwrap_or(s);
    s.parse()
        .map_err(|_| format!("cannot parse tensor id from arg/output key {key:?}"))
}

/// The integer element-shape attribute on a `construct_memory_view` op.
fn view_shape_of(op: &Operation) -> Option<Vec<usize>> {
    match op.attributes.get("shape") {
        Some(Attr::IntList(v)) if !v.is_empty() => Some(v.iter().map(|&x| x as usize).collect()),
        _ => None,
    }
}

/// Walk ops (recursing into regions) recording every memory view's shape keyed by
/// the tensor id its pointer operand binds. Mirrors `segmented::collect_view_shapes`.
fn collect_view_shapes(
    ops: &[Operation],
    arg_to_tensor: &HashMap<&str, u64>,
    shapes: &mut HashMap<u64, Vec<usize>>,
) {
    for op in ops {
        if op.op_type == "ktdp.construct_memory_view"
            && let Some(ptr) = op.operands.first()
            && let Some(&tid) = arg_to_tensor.get(ptr.as_str())
            && let Some(shape) = view_shape_of(op)
        {
            shapes.entry(tid).or_insert(shape);
        }
        for rg in &op.regions {
            collect_view_shapes(rg, arg_to_tensor, shapes);
        }
    }
}

/// Derive `tensor_id -> element-shape` for every logical tensor the program
/// touches (from each node's `construct_memory_view` shapes). Same derivation the
/// segmented executor uses — the shapes live in the IR, no external manifest.
fn derive_shapes(module: &IRModule, spec: &ProgramSpec) -> Result<HashMap<u64, Vec<usize>>, String> {
    let mut shapes: HashMap<u64, Vec<usize>> = HashMap::new();
    for node in &spec.nodes {
        let func = module.get_function(&node.func)?;
        let arg_to_tensor: HashMap<&str, u64> =
            node.bindings.iter().map(|b| (b.arg.as_str(), b.tensor)).collect();
        collect_view_shapes(&func.operations, &arg_to_tensor, &mut shapes);
    }
    Ok(shapes)
}

/// A persistent resident-execution context for one KTIR program.
///
/// Build once with [`ResidentExecutor::new`], write the source weights once with
/// [`ResidentExecutor::set_source`] (or [`ResidentExecutor::set_sources`]), then
/// call [`ResidentExecutor::run`] per pass. Weights are NEVER re-marshaled.
pub struct ResidentExecutor {
    /// OWNED (not borrowed) so the executor holds its ENTIRE `Rc` graph
    /// exclusively — see the `unsafe impl Send` below.
    module: IRModule,
    segments: Vec<Segment>,
    shapes: HashMap<u64, Vec<usize>>,
    /// The one persistent HBM (and per-core LX). Sticks are allocated once and
    /// reused across every pass.
    mem: SpyreMemoryHierarchy,
    /// tensor id -> its fixed HBM stick. Stable across passes (so the IR's weight
    /// pointers don't move).
    stick: HashMap<u64, i64>,
    /// tensor id -> element count (product of its derived shape).
    numel: HashMap<u64, usize>,
    /// Which tensor ids are program SOURCES (weights / mask / input) — written
    /// once via `set_source`, NOT zeroed per pass.
    sources: std::collections::HashSet<u64>,
    /// The program's final result tensor ids (for default readback).
    results: std::collections::HashSet<u64>,
    /// The model dtype the per-node oracle threads (F16). All sticks are sized and
    /// read back at this dtype.
    dtype: DType,
}

// SAFETY: `ResidentExecutor` owns its ENTIRE object graph exclusively. The
// `IRModule` (with its `Rc<AffineExpr>`s) is moved in and never shared; the
// `SpyreMemoryHierarchy`'s `Rc<RefCell<..>>`s are created and held only here; the
// per-core contexts that clone those `Rc`s during `run()` are created AND dropped
// inside that one call, on the calling thread. No `Rc` clone of any of these
// allocations ever exists outside the executor, so moving the whole executor to
// another thread transfers every `Rc` together — no non-atomic refcount is ever
// touched from two threads at once. We impl `Send` (move between threads) but
// deliberately NOT `Sync`: the executor is internally single-threaded
// (`Rc`/`RefCell`) and must never be shared by `&` across threads. A serving
// worker owns one and calls `run()` serially — exactly this contract. (This is
// why the module is OWNED, not borrowed: a borrowed `&IRModule` shared by two
// executors on two threads could race its `Rc<AffineExpr>` refcounts.)
unsafe impl Send for ResidentExecutor {}

impl ResidentExecutor {
    /// Plan `spec` into segments, derive every tensor's shape from the IR, and
    /// allocate ONE persistent HBM stick per tensor (stable address across
    /// passes). Sources are not yet written — call [`set_source`](Self::set_source)
    /// / [`set_sources`](Self::set_sources) before [`run`](Self::run).
    pub fn new(module: IRModule, spec: &ProgramSpec) -> Result<Self, String> {
        let segments = plan_segments(&module, spec)?;
        let shapes = derive_shapes(&module, spec)?;
        let dtype = DType::F16;
        let bpe = dtype.bytes_per_elem();

        // The set of every tensor id any segment references (a pointer arg of a
        // fused segment, or a binding of a native node). We allocate a stick for
        // each so its address is fixed for the whole executor lifetime.
        let mut ids: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for seg in &segments {
            match seg {
                Segment::Fused(fs) => {
                    for (arg, _) in &fs.func.arguments {
                        ids.insert(tensor_id_of_arg(arg)?);
                    }
                }
                Segment::Native(node) => {
                    for b in &node.bindings {
                        ids.insert(b.tensor);
                    }
                }
            }
        }
        // Also pin the requested-result tensors (a result might be produced by a
        // node and threaded out without appearing as a pointer arg elsewhere).
        for &r in &spec.results {
            ids.insert(r);
        }

        let mem = SpyreMemoryHierarchy::new(largest_grid(&module, &segments));
        let mut stick: HashMap<u64, i64> = HashMap::new();
        let mut numel: HashMap<u64, usize> = HashMap::new();
        {
            let mut hbm = mem.hbm.borrow_mut();
            for &tid in &ids {
                let shape = shapes
                    .get(&tid)
                    .cloned()
                    .ok_or_else(|| format!("no shape derivable for tensor t{tid}"))?;
                let n: usize = shape.iter().product();
                let s = hbm.allocate((n * bpe).max(bpe) as i64);
                stick.insert(tid, s);
                numel.insert(tid, n);
            }
        }
        // The LX scratchpads are shared per-core resident memory (`mem` holds one
        // per core); they are reset per core at the start of each segment run, so
        // no per-pass HBM churn beyond the activation rewrites above.

        Ok(ResidentExecutor {
            module,
            segments,
            shapes,
            mem,
            stick,
            numel,
            sources: spec.sources.clone(),
            results: spec.results.clone(),
            dtype,
        })
    }

    /// Write one SOURCE tensor's f32 data into its resident HBM stick ONCE
    /// (encoded to the model dtype). Call this for every weight / the mask / the
    /// input before the first [`run`](Self::run). The bytes persist for the
    /// executor's lifetime — the per-pass loop never rewrites a source unless you
    /// explicitly do so (e.g. a changing decode input via [`set_input`](Self::set_input)).
    pub fn set_source(&mut self, tensor: u64, data: &[f32]) -> Result<(), String> {
        let s = *self
            .stick
            .get(&tensor)
            .ok_or_else(|| format!("set_source: t{tensor} is not a tensor this program uses"))?;
        let bytes = crate::codec::encode(data, self.dtype);
        self.mem.hbm.borrow_mut().write_bytes(s * STICK_BYTES, &bytes);
        Ok(())
    }

    /// Write many sources at once (keyed by the canonical `t<id>` / `%t<id>` /
    /// `%t<id>_ptr` / bare `<id>` name). Unknown keys (a tensor this program does
    /// not reference) are skipped — the caller can hand the whole weight set.
    pub fn set_sources(&mut self, args: &[(&str, Arg)]) -> Result<(), String> {
        for (key, arg) in args {
            let tid = tensor_id_of_key(key)?;
            if !self.stick.contains_key(&tid) {
                continue;
            }
            let data = arg_to_f32(arg)?;
            self.set_source(tid, &data)?;
        }
        Ok(())
    }

    /// Rewrite a changing INPUT activation in place (decode threads a new token
    /// each pass). Identical to [`set_source`](Self::set_source) but named for the
    /// per-pass intent. The stick is unchanged, so the resident weights are
    /// untouched.
    pub fn set_input(&mut self, tensor: u64, data: &[f32]) -> Result<(), String> {
        self.set_source(tensor, data)
    }

    /// Run ONE forward pass against the resident HBM and read back `outputs`
    /// (canonical `t<id>` keys; empty = the program's declared result tensors).
    ///
    /// Zeroes every NON-source stick first (results / intermediates / scratch) so
    /// a pass never reads a stale value from the previous pass, then runs each
    /// segment in order: a fused segment at grid `[1,1]` (carrying the GPU K-loop
    /// GEMM / map-window / resident-weight-cache offloads), a native attention
    /// node at its native head-parallel grid — both against the SAME persistent
    /// HBM, so intermediates flow segment-to-segment with no marshal.
    pub fn run(&mut self, outputs: &[&str]) -> Result<HashMap<String, Output>, String> {
        self.zero_non_sources();

        // KTIR_SEG_DIAG: accumulate fused (GPU GEMM/map) vs native (CPU-interpreter
        // attention) wall-time per pass — to see how much of e2e is the attention
        // islands still on the interpreter.
        let diag = std::env::var_os("KTIR_SEG_DIAG").is_some();
        let (mut t_fused, mut t_native, mut n_fused, mut n_native) = (0.0f64, 0.0f64, 0usize, 0usize);
        for seg in &self.segments {
            // Reset every core's LX scratchpad before each segment run. The
            // persistent `mem` reuses the SAME LX across segments/passes, but each
            // function run is a self-contained SPMD execution that bump-allocates
            // LX from empty (and whose `used` watermark must start at 0). Without
            // this, `used` accumulates the live-out tracking of every prior
            // segment and eventually trips the LX capacity guard. (A fresh-`mem`
            // `execute_function` got this for free; the resident `mem` must do it
            // explicitly.) HBM is NOT cleared — that's the resident weight set.
            for lx in &self.mem.lx_scratchpads {
                lx.borrow_mut().clear();
            }
            let seg_t0 = std::time::Instant::now();
            match seg {
                Segment::Fused(fs) => {
                    // Bind every pointer arg to its resident stick, and collect
                    // the boundary OUTPUTs to read back (so intermediates flow via
                    // HBM, not host).
                    let mut input_ptrs: Vec<(String, Value)> = Vec::with_capacity(fs.func.arguments.len());
                    for (arg_name, _) in &fs.func.arguments {
                        let tid = tensor_id_of_arg(arg_name)?;
                        let bare = arg_name.trim_start_matches('%').to_string();
                        let s = *self.stick.get(&tid).ok_or_else(|| {
                            format!("fused arg t{tid} has no resident stick")
                        })?;
                        input_ptrs.push((bare, Value::Index(s)));
                    }
                    // Read back only this segment's boundary outputs (selective —
                    // never decode the hundreds of resident weight pointers).
                    let read: Vec<TensorMeta> = fs
                        .outputs
                        .iter()
                        .map(|&tid| self.meta_for(tid))
                        .collect::<Result<_, _>>()?;
                    // run + readback against the persistent HBM; the readback
                    // values are already resident in HBM for the next segment, so
                    // we don't need to copy them anywhere — the next segment reads
                    // the same sticks. The readback only validates production.
                    let _ = execute_function_in(
                        &self.mem,
                        &fs.func.operations,
                        (1, 1, 1),
                        &input_ptrs,
                        &read,
                    )?;
                }
                Segment::Native(node) => {
                    let func = self.module.get_function(&node.func)?;
                    let grid = func.grid;
                    let mut input_ptrs: Vec<(String, Value)> = Vec::with_capacity(node.bindings.len());
                    let mut read: Vec<TensorMeta> = Vec::new();
                    for b in &node.bindings {
                        let name = b.arg.trim_start_matches('%').to_string();
                        let s = *self.stick.get(&b.tensor).ok_or_else(|| {
                            format!("native attn arg t{} has no resident stick", b.tensor)
                        })?;
                        input_ptrs.push((name, Value::Index(s)));
                        if b.is_output {
                            read.push(self.meta_for(b.tensor)?);
                        }
                    }
                    let _ = execute_function_in(
                        &self.mem,
                        &func.operations,
                        grid,
                        &input_ptrs,
                        &read,
                    )?;
                }
            }
            if diag {
                let dt = seg_t0.elapsed().as_secs_f64() * 1e3;
                match seg {
                    Segment::Fused(_) => {
                        t_fused += dt;
                        n_fused += 1;
                    }
                    Segment::Native(_) => {
                        t_native += dt;
                        n_native += 1;
                    }
                }
            }
        }
        if diag {
            eprintln!(
                "  [resident-seg-diag] {n_fused} fused {t_fused:.1}ms (GPU GEMM/map) | \
                 {n_native} native {t_native:.1}ms (CPU-interp attention)"
            );
        }

        // Read back the requested outputs (default: the program results) from the
        // resident HBM.
        let want: Vec<u64> = if outputs.is_empty() {
            let mut v: Vec<u64> = self.results.iter().copied().collect();
            v.sort_unstable();
            v
        } else {
            outputs.iter().map(|k| tensor_id_of_key(k)).collect::<Result<_, _>>()?
        };
        let read: Vec<TensorMeta> = want.iter().map(|&tid| self.meta_for(tid)).collect::<Result<_, _>>()?;
        // One readback pass (decode the wanted sticks to host f32).
        let mut result = HashMap::new();
        for (name, stick, n, shape, dtype) in read {
            let nbytes = n * dtype.bytes_per_elem();
            let bytes = self.mem.hbm.borrow().read_bytes(stick * STICK_BYTES, nbytes);
            let data = crate::codec::decode(&bytes, n, dtype);
            result.insert(name, Output { data, shape, dtype, raw: bytes });
        }
        Ok(result)
    }

    /// Zero every NON-source stick (results, intermediates, scratch) so a pass
    /// starts clean. Sources (weights / mask / input) keep their resident bytes.
    /// Cheap: these are activations (kB) not weights (GB).
    fn zero_non_sources(&mut self) {
        let mut hbm = self.mem.hbm.borrow_mut();
        let bpe = self.dtype.bytes_per_elem();
        for (&tid, &s) in &self.stick {
            if self.sources.contains(&tid) {
                continue;
            }
            let n = *self.numel.get(&tid).unwrap_or(&0);
            if n == 0 {
                continue;
            }
            let zeros = vec![0u8; n * bpe];
            hbm.write_bytes(s * STICK_BYTES, &zeros);
        }
    }

    /// Build the `(name, stick, numel, shape, dtype)` readback tuple for a tensor,
    /// keyed by the canonical `t<id>` name.
    fn meta_for(&self, tid: u64) -> Result<TensorMeta, String> {
        let s = *self.stick.get(&tid).ok_or_else(|| format!("t{tid} has no resident stick"))?;
        let n = *self.numel.get(&tid).unwrap_or(&0);
        let shape = self.shapes.get(&tid).cloned().unwrap_or_else(|| vec![n]);
        Ok((format!("t{tid}"), s, n, shape, self.dtype))
    }
}

/// The largest core count any segment runs at — sizes the persistent LX array so
/// a native attention node's grid has an LX per core. Fused segments are `[1,1]`;
/// native attention nodes run at their own grid.
fn largest_grid(module: &IRModule, segments: &[Segment]) -> usize {
    let mut n = 1usize;
    for seg in segments {
        if let Segment::Native(node) = seg
            && let Ok(f) = module.get_function(&node.func)
        {
            let (gx, gy, gz) = f.grid;
            n = n.max(gx * gy * gz);
        }
    }
    n.max(1)
}

/// Decode an [`Arg`] to host f32 (the form `set_source` writes).
fn arg_to_f32(arg: &Arg) -> Result<Vec<f32>, String> {
    match arg {
        Arg::Tensor { data, .. } => Ok(data.clone()),
        Arg::TensorBytes { data, shape, dtype } => {
            Ok(crate::codec::decode(data, shape.iter().product(), *dtype))
        }
        Arg::Scalar(_) => Err("resident: scalar args unsupported".into()),
    }
}

/// Execute a whole KTIR program with RESIDENT weights — the convenience
/// single-shot analogue of [`crate::segmented::execute_segmented`].
///
/// Builds a [`ResidentExecutor`], writes every source from `args` ONCE, runs one
/// pass, and reads back `outputs`. For a multi-pass loop (decode), construct a
/// [`ResidentExecutor`] directly and call [`ResidentExecutor::run`] per pass so
/// the weights are uploaded exactly once across all passes.
///
/// `args` keys are the canonical tensor names (`t<id>` / `%t<id>` / `%t<id>_ptr`
/// / bare `<id>`); `outputs` names the tensors to return (empty = the program's
/// declared results). The returned map is keyed by `t<id>`.
pub fn execute_resident(
    module: IRModule,
    spec: &ProgramSpec,
    args: &[(&str, Arg)],
    outputs: &[&str],
) -> Result<HashMap<String, Output>, String> {
    let mut exec = ResidentExecutor::new(module, spec)?;
    exec.set_sources(args)?;
    exec.run(outputs)
}
