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

    // `produced[T]` = the fused-function SSA value currently holding tensor T,
    // once its producing node has been inlined and its store forwarded.
    let mut produced: HashMap<u64, String> = HashMap::new();
    // Pointer args the fused function still needs (sources, results, and any
    // intermediate edge we could not forward), keyed by tensor id.
    let mut needed_args: Vec<(u64, String)> = Vec::new();
    let mut have_arg: HashSet<u64> = HashSet::new();
    let mut body: Vec<Operation> = Vec::new();

    for (ni, node) in spec.nodes.iter().enumerate() {
        let func = module.get_function(&node.func)?;
        let analysis = analyze(func);
        let arg_to_tensor: HashMap<&str, &Binding> =
            node.bindings.iter().map(|b| (b.arg.as_str(), b)).collect();

        // Build the SSA rename map for this node: internal defs get an `nN_`
        // prefix; arg pointers map to either a forwarded value, a dropped chain,
        // or the canonical fused-function arg `%t<id>_ptr`.
        let mut rename: HashMap<String, String> = HashMap::new();
        let mut dropped: HashSet<usize> = HashSet::new(); // op indices to skip

        // Forward whole-tensor intermediate INPUTS: drop view→tile→load chain,
        // alias the loaded SSA to the producer value.
        for ld in &analysis.loads {
            let b = match arg_to_tensor.get(ld.arg.as_str()) {
                Some(b) => *b,
                None => continue,
            };
            if !b.is_output
                && ld.whole_tensor
                && is_intermediate(b.tensor)
                && let Some(val) = produced.get(&b.tensor)
            {
                rename.insert(ld.loaded.clone(), val.clone());
                dropped.insert(ld.load_idx);
                dropped.insert(ld.tile_idx);
                if let Some(vidx) = ld.view_idx {
                    dropped.insert(vidx);
                }
            }
        }

        // Forward whole-tensor intermediate OUTPUTS: drop the store (and its
        // view/tile), record the stored value as the producer of that tensor.
        for st in &analysis.stores {
            let b = match arg_to_tensor.get(st.arg.as_str()) {
                Some(b) => *b,
                None => continue,
            };
            if b.is_output && st.whole_tensor && is_intermediate(b.tensor) {
                // Forwarded only if every consumer reads it whole-tensor;
                // increment 1 forwards optimistically and the consumer side
                // falls back to HBM if its own load is not whole-tensor. To stay
                // correct, only drop the store when we can also keep an HBM copy
                // is unnecessary here: a non-whole consumer simply won't find the
                // tensor in `produced` and will keep its (HBM) load — which means
                // we must NOT drop the store in that case. Conservatively keep
                // the store unless we can prove all consumers forward.
                if all_consumers_whole(module, spec, b.tensor) {
                    dropped.insert(st.store_idx);
                    dropped.insert(st.tile_idx);
                    if let Some(vidx) = st.view_idx {
                        dropped.insert(vidx);
                    }
                    produced.insert(b.tensor, prefixed(ni, &st.stored));
                }
            }
        }

        // Any arg whose chain we did NOT drop and that points at a tensor still
        // needing HBM (source, result, or unforwarded intermediate) becomes a
        // fused-function arg, shared by tensor id under the canonical name.
        for b in &node.bindings {
            let forwarded_in = analysis
                .loads
                .iter()
                .any(|l| l.arg == b.arg && rename.contains_key(&l.loaded));
            let forwarded_out = produced.contains_key(&b.tensor) && b.is_output;
            if !forwarded_in && !forwarded_out {
                let canon = format!("%t{}_ptr", b.tensor);
                rename.insert(b.arg.clone(), canon.clone());
                if have_arg.insert(b.tensor) {
                    needed_args.push((b.tensor, canon));
                }
            }
        }

        // Emit the node's ops, renamed, skipping dropped ones.
        for (idx, op) in func.operations.iter().enumerate() {
            if dropped.contains(&idx) {
                continue;
            }
            body.push(rename_op(op, ni, &rename));
        }
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

/// True if every node that consumes `tensor` does so with a whole-tensor load.
fn all_consumers_whole(module: &IRModule, spec: &ProgramSpec, tensor: u64) -> bool {
    for node in &spec.nodes {
        for b in &node.bindings {
            if !b.is_output && b.tensor == tensor {
                let Ok(func) = module.get_function(&node.func) else {
                    return false;
                };
                let a = analyze(func);
                let whole = a
                    .loads
                    .iter()
                    .find(|l| l.arg == b.arg)
                    .map(|l| l.whole_tensor)
                    .unwrap_or(false);
                if !whole {
                    return false;
                }
            }
        }
    }
    true
}

// --- per-function analysis -------------------------------------------------

struct LoadChain {
    arg: String,
    loaded: String,
    whole_tensor: bool,
    load_idx: usize,
    tile_idx: usize,
    view_idx: Option<usize>,
}
struct StoreChain {
    arg: String,
    stored: String,
    whole_tensor: bool,
    store_idx: usize,
    tile_idx: usize,
    view_idx: Option<usize>,
}
#[derive(Default)]
struct Analysis {
    loads: Vec<LoadChain>,
    stores: Vec<StoreChain>,
}

/// Trace each top-level `ktdp.load`/`ktdp.store` back through its access-tile and
/// memory-view to the function arg pointer it touches, and decide whether it
/// covers the whole tensor (access-tile shape == memory-view shape).
fn analyze(func: &IRFunction) -> Analysis {
    // result-SSA -> op index, for the top-level ops.
    let mut def: HashMap<&str, usize> = HashMap::new();
    for (i, op) in func.operations.iter().enumerate() {
        if let Some(r) = &op.result {
            def.insert(r.as_str(), i);
        }
    }
    let shape_of = |idx: usize| -> Option<&Vec<i64>> {
        match func.operations[idx].attributes.get("shape") {
            Some(Attr::IntList(v)) => Some(v),
            _ => None,
        }
    };

    let mut a = Analysis::default();
    for (i, op) in func.operations.iter().enumerate() {
        match op.op_type.as_str() {
            "ktdp.load" => {
                let tile = op.operands.first();
                let tile_idx = tile.and_then(|t| def.get(t.as_str())).copied();
                if let (Some(loaded), Some(tile_idx)) = (op.result.as_ref(), tile_idx) {
                    let tile_op = &func.operations[tile_idx];
                    let view_idx = tile_op.operands.first().and_then(|v| def.get(v.as_str())).copied();
                    let arg = view_idx
                        .and_then(|vi| func.operations[vi].operands.first().cloned());
                    if let Some(arg) = arg {
                        let whole = match (shape_of(tile_idx), view_idx.and_then(shape_of)) {
                            (Some(ts), Some(vs)) => ts == vs,
                            _ => false,
                        };
                        a.loads.push(LoadChain {
                            arg,
                            loaded: loaded.clone(),
                            whole_tensor: whole,
                            load_idx: i,
                            tile_idx,
                            view_idx,
                        });
                    }
                }
            }
            "ktdp.store" => {
                // store %value, %tile
                let stored = op.operands.first().cloned();
                let tile_idx = op.operands.get(1).and_then(|t| def.get(t.as_str())).copied();
                if let (Some(stored), Some(tile_idx)) = (stored, tile_idx) {
                    let tile_op = &func.operations[tile_idx];
                    let view_idx = tile_op.operands.first().and_then(|v| def.get(v.as_str())).copied();
                    let arg = view_idx
                        .and_then(|vi| func.operations[vi].operands.first().cloned());
                    if let Some(arg) = arg {
                        let whole = match (shape_of(tile_idx), view_idx.and_then(shape_of)) {
                            (Some(ts), Some(vs)) => ts == vs,
                            _ => false,
                        };
                        a.stores.push(StoreChain {
                            arg,
                            stored,
                            whole_tensor: whole,
                            store_idx: i,
                            tile_idx,
                            view_idx,
                        });
                    }
                }
            }
            _ => {}
        }
    }
    a
}

// --- SSA renaming ----------------------------------------------------------

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

/// Deep-copy an op with all SSA names (result, operands, nested regions) renamed.
fn rename_op(op: &Operation, ni: usize, rename: &HashMap<String, String>) -> Operation {
    Operation {
        result: op.result.as_ref().map(|r| resolve(ni, r, rename)),
        op_type: op.op_type.clone(),
        operands: op.operands.iter().map(|o| resolve(ni, o, rename)).collect(),
        attributes: op.attributes.clone(),
        result_type: op.result_type.clone(),
        regions: op
            .regions
            .iter()
            .map(|region| region.iter().map(|o| rename_op(o, ni, rename)).collect())
            .collect(),
    }
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
    fn tiled_edge_falls_back_to_hbm() {
        // b loads t2 as a sub-tile (not whole) -> cannot forward; keep HBM.
        let m = module(vec![
            copy_node("a", "%in", "%out", 16, true),
            copy_node("b", "%in", "%out", 16, false),
        ]);
        let fused = fuse_program(&m, &two_node_spec()).unwrap();
        let loads = fused.operations.iter().filter(|o| o.op_type == "ktdp.load").count();
        let stores = fused.operations.iter().filter(|o| o.op_type == "ktdp.store").count();
        // a still stores t2, b still loads it (resident HBM within the fused fn).
        assert_eq!(loads, 2, "source + tiled intermediate load both kept");
        assert_eq!(stores, 2, "intermediate + result stores both kept");
        // The intermediate pointer is still a fused-function arg.
        let arg_names: Vec<&str> = fused.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert!(arg_names.contains(&"%t2_ptr"), "intermediate kept as HBM arg: {arg_names:?}");
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
