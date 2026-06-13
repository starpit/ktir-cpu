// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Cross-function fusion: collapse a multi-function KTIR program whose nodes
//! thread intermediates through HBM into a single function, forwarding each
//! whole-tensor producer→consumer edge as an SSA value (dropping the
//! `store`/`load` pair). This is a KTIR→KTIR transform — it removes real HBM
//! traffic the hardware would otherwise pay (RFC 0682 §"further optimizations
//! within that decomposition"; intermediates "reused across producer-consumer
//! operations").
//!
//! INCREMENT 1 (this file): only **whole-tensor** edges are forwarded — where
//! the consumer's `ktdp.load` reads the entire producer tensor (access-tile
//! shape == memory-view shape). Tiled consumers (a sub-tile read inside a loop)
//! are left as HBM `store`/`load` for now; they need `tensor.extract_slice`
//! against the producer value, which is increment 2.
//!
//! What's eliminated regardless of fusing an edge: the *inter-function* boundary
//! itself. The fused single function holds all intermediates in one resident
//! context, so even a non-forwarded edge no longer pays the per-call
//! marshal/read-back the multi-call runner imposed.

use ktir_core::ir::{Attr, IRFunction, IRModule, Operation};
use std::collections::{HashMap, HashSet};

/// One function argument's binding to a logical tensor and its direction.
#[derive(Clone, Debug)]
pub struct Binding {
    /// The function arg name (e.g. `%t335_ptr`).
    pub arg: String,
    /// Logical tensor id the arg points at.
    pub tensor: u64,
    /// True if the node writes this tensor (an output), false if it reads it.
    pub is_output: bool,
}

/// One node in the program: a function name + how its args bind to tensors.
#[derive(Clone, Debug)]
pub struct NodeSpec {
    pub func: String,
    pub bindings: Vec<Binding>,
}

/// A whole multi-function program to fuse, in execution order.
#[derive(Clone, Debug)]
pub struct ProgramSpec {
    pub nodes: Vec<NodeSpec>,
    /// Tensors provided from outside (weights / inputs) — stay HBM args.
    pub sources: HashSet<u64>,
    /// Final result tensors — stay HBM (stored, read back by the caller).
    pub results: HashSet<u64>,
}

/// Fuse `spec`'s nodes (functions in `module`) into a single `IRFunction`.
///
/// The fused function's args are the source + result tensor pointers (one per
/// distinct tensor, named `%t<id>_ptr`); intermediates produced and consumed
/// whole-tensor are forwarded as SSA and need no pointer.
pub fn fuse_program(module: &IRModule, spec: &ProgramSpec) -> Result<IRFunction, String> {
    // Tensors that are produced by some node AND consumed by another, and are
    // neither a source nor a final result: candidates for SSA forwarding.
    let mut produced_by: HashMap<u64, usize> = HashMap::new();
    let mut consumed: HashSet<u64> = HashSet::new();
    for (i, node) in spec.nodes.iter().enumerate() {
        for b in &node.bindings {
            if b.is_output {
                produced_by.insert(b.tensor, i);
            } else {
                consumed.insert(b.tensor);
            }
        }
    }
    let is_intermediate = |t: u64| {
        produced_by.contains_key(&t)
            && consumed.contains(&t)
            && !spec.sources.contains(&t)
            && !spec.results.contains(&t)
    };

    // Analyze every node once (region-aware) and cache — `all_consumers_*`
    // would otherwise re-walk every consumer per producer (O(nodes²)).
    let analyses: Vec<Analysis> = spec
        .nodes
        .iter()
        .map(|n| module.get_function(&n.func).map(analyze))
        .collect::<Result<_, _>>()?;

    // `produced[T]` = (fused-function SSA value holding tensor T, its full shape),
    // recorded once its producing node is inlined and its whole-tensor store
    // forwarded. The shape pins the layout a tiled consumer slices into.
    let mut produced: HashMap<u64, (String, Vec<i64>)> = HashMap::new();
    // Pointer args the fused function still needs (sources, results, and any
    // intermediate edge we could not forward), keyed by tensor id.
    let mut needed_args: Vec<(u64, String)> = Vec::new();
    let mut have_arg: HashSet<u64> = HashSet::new();
    let mut body: Vec<Operation> = Vec::new();

    for (ni, node) in spec.nodes.iter().enumerate() {
        let an = &analyses[ni];
        let arg_to_tensor: HashMap<&str, &Binding> =
            node.bindings.iter().map(|b| (b.arg.as_str(), b)).collect();

        // ----- decide which of this node's pointer args get forwarded -----
        // An INPUT arg is forwarded iff its producer is resident AND *every*
        // load through it reads the producer's full-shape layout in a way we can
        // model — whole-tensor (alias) or a contiguous sub-tile (extract_slice).
        let mut forwarded_args: HashSet<String> = HashSet::new();
        let mut loads_by_arg: HashMap<&str, Vec<&LoadChain>> = HashMap::new();
        for ld in &an.loads {
            loads_by_arg.entry(ld.arg.as_str()).or_default().push(ld);
        }
        for (arg, lds) in &loads_by_arg {
            let Some(b) = arg_to_tensor.get(*arg) else { continue };
            if b.is_output || !is_intermediate(b.tensor) {
                continue;
            }
            let Some((_, pshape)) = produced.get(&b.tensor) else {
                continue; // producer not resident -> keep HBM load
            };
            if lds
                .iter()
                .all(|l| &l.view_shape == pshape && (l.whole_tensor || l.sliceable))
            {
                forwarded_args.insert((*arg).to_string());
            }
        }
        // An OUTPUT arg is forwarded iff the producer writes the whole tensor and
        // every consuming node can forward it (same full-shape + whole/sliceable).
        // Record the resident SSA so later nodes can forward off it.
        for st in &an.stores {
            let Some(b) = arg_to_tensor.get(st.arg.as_str()) else { continue };
            if b.is_output
                && st.whole_tensor
                && is_intermediate(b.tensor)
                && all_consumers_forwardable(spec, &analyses, b.tensor, &st.view_shape)
            {
                forwarded_args.insert(st.arg.clone());
                produced.insert(b.tensor, (prefixed(ni, &st.stored), st.view_shape.clone()));
            }
        }

        // ----- turn the forwarding decision into concrete drop/rename ops -----
        let mut rename: HashMap<String, String> = HashMap::new();
        // Op result SSAs to drop entirely (views/tiles on forwarded args, and
        // whole-tensor loads whose value is aliased to the producer).
        let mut drop_results: HashSet<String> = HashSet::new();
        // ktdp.store ops whose tile operand is in here are dropped.
        let mut drop_store_tiles: HashSet<String> = HashSet::new();
        // load result SSA -> the extract_slice that replaces it (tiled forward).
        let mut slice_at_load: HashMap<String, SliceForward> = HashMap::new();

        // Drop the construct_memory_view of every forwarded arg (its HBM pointer
        // is gone), then the access tiles built on those views.
        for (vssa, (arg, _)) in &an.views {
            if forwarded_args.contains(arg) {
                drop_results.insert(vssa.clone());
            }
        }
        for (tssa, ti) in &an.tiles {
            if drop_results.contains(&ti.view) {
                drop_results.insert(tssa.clone());
            }
        }
        // Loads on dropped tiles: alias (whole) or slice (tiled).
        for ld in &an.loads {
            if !drop_results.contains(&ld.tile) {
                continue;
            }
            let Some(b) = arg_to_tensor.get(ld.arg.as_str()) else { continue };
            let Some((val, _)) = produced.get(&b.tensor) else { continue };
            if ld.whole_tensor {
                rename.insert(ld.loaded.clone(), val.clone());
                drop_results.insert(ld.loaded.clone());
            } else {
                slice_at_load.insert(
                    ld.loaded.clone(),
                    SliceForward {
                        source: val.clone(),
                        loaded: ld.loaded.clone(),
                        offsets: ld.offsets.clone(),
                        sizes: ld.tile_shape.clone(),
                    },
                );
            }
        }
        // Stores on dropped tiles: drop the store itself.
        for st in &an.stores {
            if drop_results.contains(&st.tile) {
                drop_store_tiles.insert(st.tile.clone());
            }
        }

        // Non-forwarded args keep an HBM pointer, shared by tensor id under the
        // canonical name; map this node's arg name onto it.
        for b in &node.bindings {
            if forwarded_args.contains(&b.arg) {
                continue;
            }
            let canon = format!("%t{}_ptr", b.tensor);
            rename.insert(b.arg.clone(), canon.clone());
            if have_arg.insert(b.tensor) {
                needed_args.push((b.tensor, canon));
            }
        }

        // Emit the node's ops (recursively, into regions), renamed, dropping the
        // forwarded chains and substituting tiled loads with their extract_slice.
        body.extend(emit_ops(
            &module.get_function(&node.func)?.operations,
            ni,
            &rename,
            &drop_results,
            &drop_store_tiles,
            &slice_at_load,
        ));
    }

    // Fused function args, in a deterministic order: sources, then results.
    needed_args.sort_by_key(|(t, _)| {
        let cls = if spec.sources.contains(t) {
            0
        } else if spec.results.contains(t) {
            2
        } else {
            1
        };
        (cls, *t)
    });
    let args: Vec<(String, String)> = needed_args
        .into_iter()
        .map(|(_, name)| (name, "index".to_string()))
        .collect();
    body.push(Operation::new(None, "func.return", &[]));

    Ok(IRFunction {
        name: "fused".to_string(),
        arguments: args,
        operations: body,
        grid: spec
            .nodes
            .first()
            .map(|n| module.get_function(&n.func).map(|f| f.grid).unwrap_or((1, 1, 1)))
            .unwrap_or((1, 1, 1)),
        return_type: None,
    })
}

/// True if every node consuming `tensor` reads it in a way we can forward off a
/// resident SSA value of shape `pshape`: every load through that arg must read
/// the producer's full-shape layout (`view_shape == pshape`) as a whole tensor
/// or a contiguous sub-tile. Any other read (a different view shape, a
/// non-identity base_map, indirect/gather access, or no load at all) keeps the
/// producer store and that consumer's HBM load. `analyses[i]` is node `i`'s
/// cached analysis.
fn all_consumers_forwardable(
    spec: &ProgramSpec,
    analyses: &[Analysis],
    tensor: u64,
    pshape: &[i64],
) -> bool {
    for (i, node) in spec.nodes.iter().enumerate() {
        for b in &node.bindings {
            if !b.is_output && b.tensor == tensor {
                let lds: Vec<&LoadChain> =
                    analyses[i].loads.iter().filter(|l| l.arg == b.arg).collect();
                if lds.is_empty() {
                    return false; // consumed but no recognizable load -> can't forward
                }
                if !lds
                    .iter()
                    .all(|l| l.view_shape == pshape && (l.whole_tensor || l.sliceable))
                {
                    return false;
                }
            }
        }
    }
    true
}

/// A tiled forwarded load rewritten as a `tensor.extract_slice` of the
/// producer's resident SSA value. Built at emit time so its offset operands
/// resolve through the node's final rename map; the `source` is already in the
/// fused namespace (the producer node prefixed it) and is emitted verbatim.
struct SliceForward {
    source: String,
    loaded: String,
    offsets: Vec<String>,
    sizes: Vec<i64>,
}

impl SliceForward {
    fn build(&self, ni: usize, rename: &HashMap<String, String>) -> Operation {
        let res = resolve(ni, &self.loaded, rename);
        let offsets: Vec<String> = self.offsets.iter().map(|o| resolve(ni, o, rename)).collect();
        let sizes: Vec<String> = self.sizes.iter().map(|n| n.to_string()).collect();
        let strides: Vec<String> = self.sizes.iter().map(|_| "1".to_string()).collect();
        Operation::new(Some(&res), "tensor.extract_slice", &[self.source.as_str()])
            .with_attr("slice_offsets", Attr::StrList(offsets))
            .with_attr("slice_sizes", Attr::StrList(sizes))
            .with_attr("slice_strides", Attr::StrList(strides))
    }
}

// --- per-function analysis (region-aware) ----------------------------------

struct LoadChain {
    arg: String,
    loaded: String,
    /// View SSA the access tile is built on (dropped when the arg is forwarded).
    #[allow(dead_code)]
    view: String,
    /// Access-tile SSA — identifies the tile op to drop and the load to rewrite.
    tile: String,
    whole_tensor: bool,
    /// The access tile's index operands (`construct_access_tile %view[%i, %j]`).
    /// With an identity `base_map` these are the slice's per-axis start offsets.
    offsets: Vec<String>,
    /// The access tile's logical shape — the slice sizes for a tiled forward.
    tile_shape: Vec<i64>,
    /// The memory-view's shape — must equal the producer's stored shape for the
    /// forward to index the right layout.
    view_shape: Vec<i64>,
    /// True when the access tile reads a contiguous box at `offsets` (identity
    /// `base_map`, no reordering) — the only shape a plain `extract_slice` models.
    sliceable: bool,
}
struct StoreChain {
    arg: String,
    stored: String,
    tile: String,
    whole_tensor: bool,
    view_shape: Vec<i64>,
}

/// A construct_access_tile's decoded fields.
struct TileInfo {
    view: String,
    offsets: Vec<String>,
    shape: Vec<i64>,
    base_identity: bool,
    has_order: bool,
}

#[derive(Default)]
struct Analysis {
    loads: Vec<LoadChain>,
    stores: Vec<StoreChain>,
    /// view SSA -> (arg pointer it interprets, view shape).
    views: HashMap<String, (String, Vec<i64>)>,
    /// access-tile SSA -> decoded tile.
    tiles: HashMap<String, TileInfo>,
}

/// Trace every `ktdp.load`/`ktdp.store` — at any region depth — back through its
/// access tile and memory view to the function arg pointer it touches. The real
/// model issues its tiled loads INSIDE an `scf.for`, so the walk must recurse
/// into op regions; views/tiles are collected across all depths first (a tile in
/// a loop body is built on a view declared at function top level).
fn analyze(func: &IRFunction) -> Analysis {
    let mut a = Analysis::default();
    collect_views_tiles(&func.operations, &mut a);
    // Borrow-split: read views/tiles while pushing into loads/stores.
    let Analysis { views, tiles, loads, stores } = &mut a;
    collect_loads_stores(&func.operations, views, tiles, loads, stores);
    a
}

fn shape_attr_of(op: &Operation) -> Vec<i64> {
    match op.attributes.get("shape") {
        Some(Attr::IntList(v)) => v.clone(),
        _ => Vec::new(),
    }
}

fn collect_views_tiles(ops: &[Operation], a: &mut Analysis) {
    for op in ops {
        match op.op_type.as_str() {
            "ktdp.construct_memory_view" => {
                if let (Some(res), Some(arg)) = (&op.result, op.operands.first()) {
                    a.views.insert(res.clone(), (arg.clone(), shape_attr_of(op)));
                }
            }
            "ktdp.construct_access_tile" => {
                if let (Some(res), Some(view)) = (&op.result, op.operands.first()) {
                    a.tiles.insert(
                        res.clone(),
                        TileInfo {
                            view: view.clone(),
                            offsets: op.operands[1..].to_vec(),
                            shape: shape_attr_of(op),
                            base_identity: base_map_is_identity(op),
                            has_order: op.attributes.contains_key("coordinate_order"),
                        },
                    );
                }
            }
            _ => {}
        }
        for rg in &op.regions {
            collect_views_tiles(rg, a);
        }
    }
}

fn collect_loads_stores(
    ops: &[Operation],
    views: &HashMap<String, (String, Vec<i64>)>,
    tiles: &HashMap<String, TileInfo>,
    loads: &mut Vec<LoadChain>,
    stores: &mut Vec<StoreChain>,
) {
    for op in ops {
        match op.op_type.as_str() {
            "ktdp.load" => {
                if let (Some(loaded), Some(tile_ssa)) = (&op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile_ssa)
                    && let Some((arg, vshape)) = views.get(&ti.view)
                {
                    let whole = !ti.shape.is_empty() && &ti.shape == vshape;
                    let sliceable = ti.base_identity
                        && !ti.has_order
                        && !ti.offsets.is_empty()
                        && ti.offsets.len() == ti.shape.len();
                    loads.push(LoadChain {
                        arg: arg.clone(),
                        loaded: loaded.clone(),
                        view: ti.view.clone(),
                        tile: tile_ssa.clone(),
                        whole_tensor: whole,
                        offsets: ti.offsets.clone(),
                        tile_shape: ti.shape.clone(),
                        view_shape: vshape.clone(),
                        sliceable,
                    });
                }
            }
            "ktdp.store" => {
                if let (Some(stored), Some(tile_ssa)) = (op.operands.first(), op.operands.get(1))
                    && let Some(ti) = tiles.get(tile_ssa)
                    && let Some((arg, vshape)) = views.get(&ti.view)
                {
                    let whole = !ti.shape.is_empty() && &ti.shape == vshape;
                    stores.push(StoreChain {
                        arg: arg.clone(),
                        stored: stored.clone(),
                        tile: tile_ssa.clone(),
                        whole_tensor: whole,
                        view_shape: vshape.clone(),
                    });
                }
            }
            _ => {}
        }
        for rg in &op.regions {
            collect_loads_stores(rg, views, tiles, loads, stores);
        }
    }
}

/// True when a `construct_access_tile` op's `base_map` is the identity (so the
/// access reads a contiguous box starting at its index operands). An absent
/// `base_map` is identity by construction (the emulator synthesizes one).
fn base_map_is_identity(tile_op: &Operation) -> bool {
    match tile_op.attributes.get("base_map") {
        Some(Attr::AffineMap(m)) => m.is_identity(),
        _ => true,
    }
}

// --- SSA renaming + recursive emit -----------------------------------------

/// `%foo` -> `%nN_foo` (node-local rename to avoid collisions across inlined nodes).
fn prefixed(ni: usize, ssa: &str) -> String {
    format!("%n{ni}_{}", ssa.trim_start_matches('%'))
}

/// Resolve an operand/result name through the rename map: an explicit mapping
/// wins (arg→canonical/forwarded); otherwise an SSA value gets the node prefix.
fn resolve(ni: usize, name: &str, rename: &HashMap<String, String>) -> String {
    if let Some(mapped) = rename.get(name) {
        return mapped.clone();
    }
    if name.starts_with('%') {
        prefixed(ni, name)
    } else {
        name.to_string() // non-SSA token (rare in operands)
    }
}

/// Some ops carry SSA names in ATTRIBUTES, not just operands — `scf.for`'s
/// induction variable (`iter_var`) and loop-carried names (`iter_args`), and any
/// op's multi-result `result_names` / a view's dynamic `sizes_dyn`. These must be
/// renamed in lockstep with the op stream, or a fused loop body would reference a
/// differently-prefixed induction variable than the one the loop binds.
fn rename_attrs(
    op: &Operation,
    ni: usize,
    rename: &HashMap<String, String>,
) -> std::collections::HashMap<String, Attr> {
    let mut attrs = op.attributes.clone();
    for key in ["iter_var", "iter_args", "result_names", "sizes_dyn"] {
        match attrs.get(key) {
            Some(Attr::Str(s)) => {
                attrs.insert(key.to_string(), Attr::Str(resolve(ni, s, rename)));
            }
            Some(Attr::StrList(xs)) => {
                let mapped = xs.iter().map(|s| resolve(ni, s, rename)).collect();
                attrs.insert(key.to_string(), Attr::StrList(mapped));
            }
            _ => {}
        }
    }
    attrs
}

/// Emit a node's ops into the fused body, recursing into regions: rename every
/// SSA (operands, results, SSA-bearing attributes, nested regions), drop the
/// forwarded view/tile/load/store chains, and substitute each tiled forwarded
/// load with its `extract_slice`. Per-node `func.return`s are dropped (the fused
/// function gets a single trailing return).
fn emit_ops(
    ops: &[Operation],
    ni: usize,
    rename: &HashMap<String, String>,
    drop_results: &HashSet<String>,
    drop_store_tiles: &HashSet<String>,
    slice_at_load: &HashMap<String, SliceForward>,
) -> Vec<Operation> {
    let mut out = Vec::new();
    for op in ops {
        if op.op_type == "func.return" {
            continue;
        }
        if let Some(r) = &op.result
            && drop_results.contains(r)
        {
            continue;
        }
        if op.op_type == "ktdp.store"
            && let Some(tile) = op.operands.get(1)
            && drop_store_tiles.contains(tile)
        {
            continue;
        }
        if op.op_type == "ktdp.load"
            && let Some(r) = &op.result
            && let Some(sf) = slice_at_load.get(r)
        {
            out.push(sf.build(ni, rename));
            continue;
        }
        out.push(Operation {
            result: op.result.as_ref().map(|r| resolve(ni, r, rename)),
            op_type: op.op_type.clone(),
            operands: op.operands.iter().map(|o| resolve(ni, o, rename)).collect(),
            attributes: rename_attrs(op, ni, rename),
            result_type: op.result_type.clone(),
            regions: op
                .regions
                .iter()
                .map(|rg| {
                    emit_ops(rg, ni, rename, drop_results, drop_store_tiles, slice_at_load)
                })
                .collect(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktir_core::ir::Attr;

    /// Build a node function: load whole tensor from `in_arg`, "compute"
    /// (identity copy via op `%out = <op> %loaded`), store whole to `out_arg`.
    /// `whole` toggles whether the access-tile shape matches the view (full) or
    /// is a sub-tile (forces the HBM fallback).
    fn copy_node(name: &str, in_arg: &str, out_arg: &str, shape: i64, whole: bool) -> IRFunction {
        let tile_shape = if whole { shape } else { shape / 2 };
        let mk_view = |res: &str, arg: &str| {
            Operation::new(Some(res), "ktdp.construct_memory_view", &[arg])
                .with_attr("shape", Attr::IntList(vec![shape]))
                .with_attr("strides", Attr::IntList(vec![1]))
                .with_attr("memory_space", Attr::Str("HBM".into()))
                .with_attr("dtype", Attr::Str("f16".into()))
        };
        let mk_tile = |res: &str, view: &str| {
            Operation::new(Some(res), "ktdp.construct_access_tile", &[view])
                .with_attr("shape", Attr::IntList(vec![tile_shape]))
        };
        IRFunction {
            name: name.to_string(),
            arguments: vec![
                (in_arg.to_string(), "index".into()),
                (out_arg.to_string(), "index".into()),
            ],
            grid: (1, 1, 1),
            return_type: None,
            operations: vec![
                mk_view("%vin", in_arg),
                mk_tile("%tin", "%vin"),
                Operation::new(Some("%loaded"), "ktdp.load", &["%tin"]),
                Operation::new(Some("%y"), "math.exp", &["%loaded"]),
                mk_view("%vout", out_arg),
                mk_tile("%tout", "%vout"),
                Operation::new(None, "ktdp.store", &["%y", "%tout"]),
                Operation::new(None, "func.return", &[]),
            ],
        }
    }

    /// Consumer that reads a contiguous sub-tile of `in_arg` at a dynamic offset
    /// (`construct_access_tile %vin[%c0]`, identity base_map) — the tiled edge
    /// increment 2 forwards via `tensor.extract_slice`. Produces a `tile`-sized
    /// result stored whole to `out_arg`.
    fn tiled_consumer(name: &str, in_arg: &str, out_arg: &str, shape: i64, tile: i64) -> IRFunction {
        IRFunction {
            name: name.to_string(),
            arguments: vec![
                (in_arg.to_string(), "index".into()),
                (out_arg.to_string(), "index".into()),
            ],
            grid: (1, 1, 1),
            return_type: None,
            operations: vec![
                Operation::new(Some("%c0"), "arith.constant", &[]).with_attr("value", Attr::Int(0)),
                Operation::new(Some("%vin"), "ktdp.construct_memory_view", &[in_arg])
                    .with_attr("shape", Attr::IntList(vec![shape]))
                    .with_attr("strides", Attr::IntList(vec![1]))
                    .with_attr("memory_space", Attr::Str("HBM".into()))
                    .with_attr("dtype", Attr::Str("f16".into())),
                // access tile at offset %c0, size `tile` (a sub-tile of the view).
                Operation::new(Some("%tin"), "ktdp.construct_access_tile", &["%vin", "%c0"])
                    .with_attr("shape", Attr::IntList(vec![tile])),
                Operation::new(Some("%loaded"), "ktdp.load", &["%tin"]),
                Operation::new(Some("%y"), "math.exp", &["%loaded"]),
                Operation::new(Some("%vout"), "ktdp.construct_memory_view", &[out_arg])
                    .with_attr("shape", Attr::IntList(vec![tile]))
                    .with_attr("strides", Attr::IntList(vec![1]))
                    .with_attr("memory_space", Attr::Str("HBM".into()))
                    .with_attr("dtype", Attr::Str("f16".into())),
                Operation::new(Some("%tout"), "ktdp.construct_access_tile", &["%vout"])
                    .with_attr("shape", Attr::IntList(vec![tile])),
                Operation::new(None, "ktdp.store", &["%y", "%tout"]),
                Operation::new(None, "func.return", &[]),
            ],
        }
    }

    fn module(funcs: Vec<IRFunction>) -> IRModule {
        let mut m = IRModule::default();
        for f in funcs {
            m.add_function(f);
        }
        m
    }

    /// a: src(1) -> t(2);  b: t(2) -> result(3).  t is a whole-tensor edge.
    fn two_node_spec() -> ProgramSpec {
        ProgramSpec {
            nodes: vec![
                NodeSpec {
                    func: "a".into(),
                    bindings: vec![
                        Binding { arg: "%in".into(), tensor: 1, is_output: false },
                        Binding { arg: "%out".into(), tensor: 2, is_output: true },
                    ],
                },
                NodeSpec {
                    func: "b".into(),
                    bindings: vec![
                        Binding { arg: "%in".into(), tensor: 2, is_output: false },
                        Binding { arg: "%out".into(), tensor: 3, is_output: true },
                    ],
                },
            ],
            sources: HashSet::from([1]),
            results: HashSet::from([3]),
        }
    }

    #[test]
    fn whole_tensor_edge_is_forwarded_no_hbm() {
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            copy_node("b", "%in", "%out", 16, true),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();

        // The intermediate t2's store AND load are gone: no HBM round-trip.
        let loads = fused.operations.iter().filter(|o| o.op_type == "ktdp.load").count();
        let stores = fused.operations.iter().filter(|o| o.op_type == "ktdp.store").count();
        assert_eq!(loads, 1, "only the source load survives");
        assert_eq!(stores, 1, "only the result store survives");

        // The fused function only needs the source (t1) + result (t3) pointers.
        let arg_names: Vec<&str> = fused.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(arg_names, vec!["%t1_ptr", "%t3_ptr"], "no pointer for intermediate t2");

        // b's exp consumes a's exp result directly (SSA forwarded).
        let b_exp = fused
            .operations
            .iter()
            .find(|o| o.op_type == "math.exp" && o.result.as_deref() == Some("%n1_y"))
            .expect("b's exp present");
        assert_eq!(b_exp.operands, vec!["%n0_y"], "b's exp reads a's stored SSA value");
    }

    #[test]
    fn unsliceable_tiled_edge_falls_back_to_hbm() {
        // b reads a sub-tile with NO index operands (offsets empty) — not a
        // contiguous extract_slice we can place, so it stays an HBM round-trip.
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            copy_node("b", "%in", "%out", 16, false),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();
        let loads = fused.operations.iter().filter(|o| o.op_type == "ktdp.load").count();
        let stores = fused.operations.iter().filter(|o| o.op_type == "ktdp.store").count();
        let slices = fused.operations.iter().filter(|o| o.op_type == "tensor.extract_slice").count();
        // a still stores t2, b still loads it (resident HBM within the fused fn).
        assert_eq!(loads, 2, "source + tiled intermediate load both kept");
        assert_eq!(stores, 2, "intermediate + result stores both kept");
        assert_eq!(slices, 0, "no extract_slice emitted for the unsliceable edge");
        // The intermediate pointer is still a fused-function arg.
        let arg_names: Vec<&str> = fused.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert!(arg_names.contains(&"%t2_ptr"), "intermediate kept as HBM arg: {arg_names:?}");
    }

    #[test]
    fn tiled_edge_forwards_via_extract_slice() {
        // a writes t2 whole; b reads a contiguous sub-tile of t2 at offset %c0.
        // The edge forwards: a's store and b's load are gone, replaced by a
        // tensor.extract_slice of a's resident SSA value — no HBM round-trip.
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            tiled_consumer("b", "%in", "%out", 16, 8),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();

        // Only the source load (a) and the result store (b) survive.
        let loads = fused.operations.iter().filter(|o| o.op_type == "ktdp.load").count();
        let stores = fused.operations.iter().filter(|o| o.op_type == "ktdp.store").count();
        assert_eq!(loads, 1, "intermediate load replaced by extract_slice");
        assert_eq!(stores, 1, "intermediate store dropped (producer resident)");

        // The extract_slice reads a's stored value at the tile offset/size.
        let slice = fused
            .operations
            .iter()
            .find(|o| o.op_type == "tensor.extract_slice")
            .expect("extract_slice emitted for the tiled edge");
        assert_eq!(slice.operands, vec!["%n0_y"], "slices a's resident producer SSA");
        assert_eq!(slice.result.as_deref(), Some("%n1_loaded"));
        assert_eq!(
            slice.attributes.get("slice_offsets"),
            Some(&Attr::StrList(vec!["%n1_c0".into()])),
            "offset is b's renamed index operand"
        );
        assert_eq!(
            slice.attributes.get("slice_sizes"),
            Some(&Attr::StrList(vec!["8".into()]))
        );
        assert_eq!(
            slice.attributes.get("slice_strides"),
            Some(&Attr::StrList(vec!["1".into()]))
        );

        // b's exp consumes the slice (downstream SSA lines up).
        let b_exp = fused
            .operations
            .iter()
            .find(|o| o.op_type == "math.exp" && o.result.as_deref() == Some("%n1_y"))
            .expect("b's exp present");
        assert_eq!(b_exp.operands, vec!["%n1_loaded"]);

        // No HBM pointer for the forwarded intermediate t2.
        let arg_names: Vec<&str> = fused.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(arg_names, vec!["%t1_ptr", "%t3_ptr"], "no t2 pointer: {arg_names:?}");
    }

    #[test]
    fn ssa_renaming_avoids_collisions() {
        // Both nodes use identical internal SSA names (%loaded, %y); after fusion
        // they must be distinct (prefixed).
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            copy_node("b", "%in", "%out", 16, true),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();
        let exps: Vec<&str> = fused
            .operations
            .iter()
            .filter(|o| o.op_type == "math.exp")
            .filter_map(|o| o.result.as_deref())
            .collect();
        assert_eq!(exps, vec!["%n0_y", "%n1_y"], "node-prefixed, no collision");
    }
}
