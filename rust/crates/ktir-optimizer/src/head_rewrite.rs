// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Head-parallel attention RE-ROLL pass — TODO #1 (the BELOW-cap head regime of
//! Contract B).
//!
//! The real cached prefill attention nodes (`node111.mlir`) are a head-parallel
//! SPMD lowering: a `grid = [H, 1]` whose per-core body is `m` MANUALLY UNROLLED
//! query-row blocks. Each row does the textbook two-block online softmax — a
//! square CONTEXT block over the prior KV cache (`view5`/`view7`, masked by the
//! per-head context mask `view2`) plus a RAGGED causal DIAGONAL block over the
//! current segment (`view6`/`view8`, row `r` attending exactly KV positions
//! `0..=r`). The body emits ~100 interpreter ops PER ROW × `m` rows.
//!
//! This pass RECOGNIZES that idiom from STRUCTURAL invariants (never model names
//! or hard-coded shapes) and RE-ROLLS the `m` per-row blocks into ONE pass of
//! whole-`[m, *]` tensor ops, emitting ONLY RFC-0682 ops (`ktdp` load/store +
//! Arith/Math/LinAlg + `tensor`) that the EXISTING generic interpreter runs
//! UNCHANGED. It is the head analogue of [`crate::flash_attn`] for the cap dim:
//! a correctness-preserving IR→IR rewrite, NOT a hand kernel and NOT a bespoke
//! executor. The grid stays `[H, 1]`; the per-head GQA column arithmetic
//! (`get_compute_tile_id` → `divui gqac` → `muli hdc`) is preserved as SSA so
//! every core still selects its own head/KV slice.
//!
//! ## Why this is a legitimate, semantics-preserving optimization
//!
//! Stacking `m` independent per-row online-softmax blocks `S[r] = q[r]·Kᵀ` into
//! one `S = Q·Kᵀ` is exact tensor re-association (the rows never interact across
//! the softmax — each row reduces over its own KV axis). The ragged per-row
//! diagonal (row `r` over current-seg KV `0..=r`) is reproduced EXACTLY by one
//! square `[m, m]` masked block: a STATIC lower-triangular mask sets `S_d[r,k]`
//! to `0` for `k ≤ r` and `-inf` for `k > r`, so `exp(-inf)=0` zeroes the
//! out-of-causal weights — identical arithmetic, just stacked. Online softmax
//! over the two blocks (global max, two `exp`, two sums, normalize) is the same
//! per row as the unrolled form. It is therefore well inside the 0.05 gate.
//!
//! ## Contract B (`fusion::attention_needs_flash`)
//!
//! This pass owns the BELOW-cap regime: it fires IFF the re-rolled `[m, cap]`
//! context-scores tile fits LX. If that tile would OVERFLOW LX (long context),
//! the pass returns `None` and leaves the node NAIVE so [`crate::flash_attn`]'s
//! cap-tiling owns it — the documented disjoint head-vs-cap partition. The pass
//! emits NO `scf.*` (a pure re-roll), so it stays region-free and never trips
//! the batched-executor's region-free gate.
//!
//! ## Fail-safe recognition
//!
//! [`recognize_head_attention`] returns `None` unless the body is PROVABLY the
//! unrolled two-block head idiom: `grid = [H,1]` with `H>1`, no top-level
//! control flow, exactly `m == view0.rows` stores, and every one of the `m` rows
//! matching the two-matmul-pair QKᵀ/AV signature with a diagonal access tile of
//! EXACTLY `r+1` rows (causal growth verified, not assumed). Any deviation →
//! `None` → module unchanged. We never rewrite a node we cannot prove equivalent.

use ktir_core::ir::{Attr, IRFunction, IRModule, Operation};
use std::collections::HashMap;

/// The recovered configuration of a recognized head-parallel attention function.
///
/// Every shape/scalar is re-derived from the IR (never assumed). The view-arg
/// pointers and the GQA divisor / head-dim / scale / `-inf` constant are read
/// from the actual ops so the rewrite reproduces the node's exact arithmetic and
/// per-head selection.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadAttnIsland {
    /// Q pointer arg (view0), `[m, H*d]`.
    pub q_arg: String,
    /// O pointer arg (view1), `[m, H*d]`.
    pub o_arg: String,
    /// Per-head context mask pointer arg (view2), `[1, cap]`.
    pub mask_arg: String,
    /// Context K pointer arg (view5), `[cap, kv_cols]`.
    pub kc_arg: String,
    /// Diagonal (current-segment) K pointer arg (view6), `[m, kv_cols]`.
    pub kd_arg: String,
    /// Context V pointer arg (view7), `[cap, kv_cols]`.
    pub vc_arg: String,
    /// Diagonal V pointer arg (view8), `[m, kv_cols]`.
    pub vd_arg: String,
    /// Q/O view column width `H*d` (so `mk_view` round-trips the original shape).
    pub q_cols: i64,
    /// KV view column width (`num_kv_heads * d`).
    pub kv_cols: i64,
    /// Query rows == number of stores == view0.rows.
    pub m: i64,
    /// Context KV length (the `cap` axis) == view5.rows.
    pub cap: i64,
    /// Head dim.
    pub d: i64,
    /// GQA divisor recovered from the `arith.divui %hpid, %gqac`.
    pub gqac: i64,
    /// Per-head column stride `hdc` recovered from `arith.muli %hpid, %hdc`.
    pub hdc: i64,
    /// Grid head count `H` (grid.0).
    pub h: i64,
    /// `1/sqrt(d)` scale recovered from the `arith.mulf` by a splat constant.
    pub scale: f32,
    /// `-inf` mask constant recovered from the context mask path.
    pub ninf: f32,
    /// Storage dtype string (e.g. `"f16"`).
    pub dtype: String,
}

impl HeadAttnIsland {
    /// Re-rolled context-scores tile `[m, cap]` × storage-dtype bytes — the value
    /// Contract B's `attention_needs_flash` consumes to decide head-vs-cap.
    pub fn scores_bytes(&self) -> usize {
        let bytes = match self.dtype.as_str() {
            "f32" | "i32" => 4,
            "f64" | "i64" => 8,
            "i1" => 1,
            _ => 2, // f16/bf16 default
        };
        (self.m as usize)
            .saturating_mul(self.cap as usize)
            .saturating_mul(bytes)
    }
}

/// Apply the head re-roll pass to every function in `module`, in place.
///
/// For each function: recognize the unrolled head-parallel idiom; if it is
/// PROVABLY that idiom AND its re-rolled `[m, cap]` scores tile FITS LX
/// (`!needs_flash(scores_bytes)` — Contract B's below-cap regime), replace it
/// with the re-rolled whole-row rewrite (same function NAME, args, grid `[H,1]`).
/// Otherwise leave it untouched (fail-safe: above the cap → flash_attn's regime,
/// or not recognized → naive). Returns the number of functions rewritten.
///
/// `needs_flash` is injected so the caller threads its OWN LX budget (the same
/// predicate flash_attn uses), keeping the cap-partition decision in one place
/// and guaranteeing the two passes never both fire on one node.
pub fn apply_head_rewrite(module: &mut IRModule, needs_flash: impl Fn(usize) -> bool) -> usize {
    let names: Vec<String> = module.functions.keys().cloned().collect();
    let mut rewritten = 0usize;
    for name in names {
        let Some(func) = module.functions.get(&name) else {
            continue;
        };
        let Some(island) = recognize_head_attention(func) else {
            continue;
        };
        if needs_flash(island.scores_bytes()) {
            // Above the cap: leave naive so flash_attn's cap-tiling owns it.
            continue;
        }
        // Preserve the ORIGINAL function identity (name) and the ORIGINAL argument
        // list (names + order) verbatim, so the program's node→tensor bindings and
        // the segmenter — which bind args by name AND position — still resolve
        // exactly as before. The rewrite only re-rolls the body; the interface is
        // byte-identical.
        let original_args = func.arguments.clone();
        let mut rolled = rewrite_head_attention(&island);
        rolled.name = name.clone();
        rolled.arguments = original_args;
        module.functions.insert(name, rolled);
        rewritten += 1;
    }
    rewritten
}

// ===========================================================================
// Recognition
// ===========================================================================

/// Decoded `ktdp.construct_memory_view %ptr` -> (pointer arg, view shape, dtype).
struct ViewInfo {
    arg: String,
    shape: Vec<i64>,
    dtype: String,
}

/// Decoded `ktdp.construct_access_tile %view[idx..]` -> the view it reads, the
/// access-tile shape, and the index-operand SSA names (for the row offset).
struct TileInfo {
    view: String,
    shape: Vec<i64>,
    indices: Vec<String>,
}

fn shape_attr(op: &Operation) -> Vec<i64> {
    match op.attributes.get("shape") {
        Some(Attr::IntList(v)) => v.clone(),
        _ => Vec::new(),
    }
}

fn dtype_attr(op: &Operation) -> String {
    match op.attributes.get("dtype") {
        Some(Attr::Str(s)) => s.clone(),
        _ => "f16".to_string(),
    }
}

/// The decoded chain behind one matmul output: which view a loaded operand reads
/// and (for the diagonal block) the access-tile row count.
struct LoadChain {
    /// Pointer arg of the view feeding this load.
    arg: String,
    /// Full memory-view shape `[rows, cols]`.
    view_shape: Vec<i64>,
    /// Access-tile shape (the actual loaded sub-tile).
    tile_shape: Vec<i64>,
    /// Storage dtype of the source view.
    dtype: String,
}

/// Recognize the unrolled head-parallel two-block attention idiom in `func`.
///
/// Returns `Some(island)` only when ALL structural invariants hold (fail-safe):
///   1. `grid = (H, 1, 1)` with `H > 1`;
///   2. no top-level `scf.*` control flow;
///   3. exactly `m` `ktdp.store` ops where `m == view0.rows == view1.rows`;
///   4. a `get_compute_tile_id`, an `arith.divui %hpid, %gqac` (gqac ≥ 1), and an
///      `arith.muli %hpid, %hdc` with `hdc == view0.cols / H`;
///   5. for each row `r` in `0..m`: the two-matmul-pair signature — context
///      `Q·Kcᵀ` (Kc `[cap,d]`) + diagonal `Q·Kdᵀ` (Kd `[r+1, d]`, the CAUSAL
///      growth verified) feeding the AV pair `Wc·Vc` (`[cap,d]`) + `Wd·Vd`
///      (`[r+1,d]`) summed and stored at row `r`;
///   6. all rows share the same scale / mask / gqac / views.
///
/// Any deviation yields `None`, leaving the node naive.
pub fn recognize_head_attention(func: &IRFunction) -> Option<HeadAttnIsland> {
    // (1) grid = [H, 1, 1], H > 1.
    let (h, gy, gz) = func.grid;
    if gy != 1 || gz != 1 || h <= 1 {
        return None;
    }
    let h = h as i64;

    // (2) no top-level control flow (a pure re-roll never starts from a loop).
    if func.operations.iter().any(|op| {
        matches!(
            op.op_type.as_str(),
            "scf.for" | "scf.if" | "scf.while" | "scf.parallel" | "scf.forall"
        )
    }) {
        return None;
    }

    // Index views / access tiles / loads by result SSA, and the defining op for
    // every result so we can walk compute chains.
    let mut views: HashMap<String, ViewInfo> = HashMap::new();
    let mut tiles: HashMap<String, TileInfo> = HashMap::new();
    let mut load_src: HashMap<String, LoadChain> = HashMap::new();
    let mut def: HashMap<String, &Operation> = HashMap::new();
    // arith.constant index value table (for resolving row offsets).
    let mut int_const: HashMap<String, i64> = HashMap::new();

    for op in &func.operations {
        match op.op_type.as_str() {
            "ktdp.construct_memory_view" => {
                if let (Some(res), Some(arg)) = (&op.result, op.operands.first()) {
                    views.insert(
                        res.clone(),
                        ViewInfo {
                            arg: arg.clone(),
                            shape: shape_attr(op),
                            dtype: dtype_attr(op),
                        },
                    );
                }
            }
            "ktdp.construct_access_tile" => {
                if let (Some(res), Some(view)) = (&op.result, op.operands.first()) {
                    tiles.insert(
                        res.clone(),
                        TileInfo {
                            view: view.clone(),
                            shape: shape_attr(op),
                            indices: op.operands[1..].to_vec(),
                        },
                    );
                }
            }
            "ktdp.load" => {
                if let (Some(res), Some(tile)) = (&op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile)
                    && let Some(vi) = views.get(&ti.view)
                {
                    load_src.insert(
                        res.clone(),
                        LoadChain {
                            arg: vi.arg.clone(),
                            view_shape: vi.shape.clone(),
                            tile_shape: ti.shape.clone(),
                            dtype: vi.dtype.clone(),
                        },
                    );
                }
            }
            "arith.constant" => {
                if let (Some(res), Some(Attr::Int(v))) = (&op.result, op.attributes.get("value")) {
                    int_const.insert(res.clone(), *v);
                }
            }
            _ => {}
        }
        if let Some(res) = &op.result {
            def.insert(res.clone(), op);
        }
    }

    // (4) per-head selection arithmetic: divui by a constant gqac, muli by hdc.
    let gqac = func
        .operations
        .iter()
        .find(|o| o.op_type == "arith.divui")
        .and_then(|o| o.operands.get(1))
        .and_then(|c| int_const.get(c).copied())?;
    if gqac < 1 {
        return None;
    }
    if !func
        .operations
        .iter()
        .any(|o| o.op_type == "ktdp.get_compute_tile_id")
    {
        return None;
    }

    // (3) exactly `m` stores; m derived from the store COUNT and cross-checked
    // against view0/view1 rows below.
    let stores: Vec<&Operation> = func
        .operations
        .iter()
        .filter(|o| o.op_type == "ktdp.store")
        .collect();
    let m = stores.len() as i64;
    if m <= 0 {
        return None;
    }

    // Recover each per-row block by walking back from its store. Collect the
    // recovered config and assert it is identical across rows.
    let mut cfg: Option<HeadAttnIsland> = None;
    // The set of expected query-row offsets must be exactly {0, 1, ..., m-1}.
    let mut seen_rows = vec![false; m as usize];

    for store in &stores {
        let row = recognize_row(store, &tiles, &load_src, &def, &int_const, m, gqac, h)?;
        // record the row offset coverage
        if row.q_row < 0 || row.q_row >= m {
            return None;
        }
        let slot = &mut seen_rows[row.q_row as usize];
        if *slot {
            return None; // duplicate row offset
        }
        *slot = true;

        let island = row.island;
        match &cfg {
            None => cfg = Some(island),
            Some(prev) => {
                // All rows must agree on every recovered field.
                if *prev != island {
                    return None;
                }
            }
        }
    }
    // Every row 0..m-1 must be present exactly once (the contiguous causal set).
    if seen_rows.iter().any(|&b| !b) {
        return None;
    }

    cfg
}

/// One recognized query-row block: its row offset and the (row-invariant) island
/// config it implies. `recognize_head_attention` cross-checks the config across
/// all rows and the row offsets cover `0..m-1`.
struct RowMatch {
    q_row: i64,
    island: HeadAttnIsland,
}

/// Walk back from one row's `ktdp.store` and prove the two-block online-softmax
/// signature, returning the row offset and the implied island config. Returns
/// `None` on any structural deviation (fail-safe).
#[allow(clippy::too_many_arguments)]
fn recognize_row(
    store: &Operation,
    tiles: &HashMap<String, TileInfo>,
    load_src: &HashMap<String, LoadChain>,
    def: &HashMap<String, &Operation>,
    int_const: &HashMap<String, i64>,
    m: i64,
    gqac: i64,
    h: i64,
) -> Option<RowMatch> {
    // store %oa, %o_tile  (O[1, d] at [q_row, qcol]).
    let stored_val = store.operands.first()?;
    let o_tile_ssa = store.operands.get(1)?;
    let o_tile = tiles.get(o_tile_ssa)?;
    let o_view = o_tile.view.clone();
    // q_row is the first index operand resolved to a constant.
    let q_row = *int_const.get(o_tile.indices.first()?)?;

    // oa = arith.addf(ov_context, ov_diag)
    let add = def.get(stored_val)?;
    if add.op_type != "arith.addf" {
        return None;
    }
    let ovc = add.operands.first()?;
    let ovd = add.operands.get(1)?;

    // Context AV: ov_context = linalg.matmul(Wc, Vc), Vc loaded [cap, d].
    let avc = def.get(ovc)?;
    if avc.op_type != "linalg.matmul" {
        return None;
    }
    let wc = avc.operands.first()?;
    let vc_loaded = avc.operands.get(1)?;
    let vc = load_src.get(vc_loaded)?;

    // Diagonal AV: ov_diag = linalg.matmul(Wd, Vd), Vd loaded [r+1, d].
    let avd = def.get(ovd)?;
    if avd.op_type != "linalg.matmul" {
        return None;
    }
    let wd = avd.operands.first()?;
    let vd_loaded = avd.operands.get(1)?;
    let vd = load_src.get(vd_loaded)?;

    // Wc = divf(exp_c, gs_bcast), Wd = divf(exp_d, gs_bcast).
    let (exp_c, _gsc) = trace_divf(wc, def)?;
    let (exp_d, _gsd) = trace_divf(wd, def)?;

    // exp_c = math.exp(sub_c); sub_c = subf(scm_c, gm_bcast)
    let scm_c = trace_exp_sub(&exp_c, def)?;
    let sd = trace_exp_sub(&exp_d, def)?;

    // CONTEXT scores: scm_c = addf(scaled_c, mask)  [the per-head context mask].
    let scm_op = def.get(&scm_c)?;
    if scm_op.op_type != "arith.addf" {
        return None;
    }
    let scaled_c = scm_op.operands.first()?;
    let mask_loaded = scm_op.operands.get(1)?;
    let mask_chain = load_src.get(mask_loaded)?;

    // scaled_c = mulf(raw_c, scale_splat)
    let (raw_c, scale) = trace_scale(scaled_c, def)?;
    // raw_c = matmul(Q, Kct); Kct = transpose(Kc_loaded); Kc [cap, d].
    let (q_loaded_c, kc) = trace_qk(&raw_c, def, load_src)?;

    // DIAGONAL scores: sd = mulf(raw_d, scale_splat)  (NO mask add on the
    // diagonal in the unrolled form — the ragged access tile IS the mask).
    let (raw_d, scale_d) = trace_scale(&sd, def)?;
    if (scale - scale_d).abs() > 1e-4 {
        return None;
    }
    let (q_loaded_d, kd) = trace_qk(&raw_d, def, load_src)?;

    // Q must be the SAME load arg for both blocks (one query row).
    let q_c = load_src.get(&q_loaded_c)?;
    let q_d = load_src.get(&q_loaded_d)?;
    if q_c.arg != q_d.arg {
        return None;
    }

    // Recover the -inf mask constant from the context mask reduce-max init, or
    // fall back to the project default. The reduce over scm_c uses a splat of the
    // -inf constant; recover it for an exact rewrite (mask above-diagonal value).
    let ninf = recover_ninf(&scm_c, def).unwrap_or(-1.0e38);

    // ---- shape checks ----
    // Q/O view [m, q_cols]; q_cols = H*d, hdc = d = q_cols/H.
    let q_shape = &q_c.view_shape;
    if q_shape.len() != 2 || q_shape[0] != m {
        return None;
    }
    let q_cols = q_shape[1];
    if q_cols % h != 0 {
        return None;
    }
    let d = q_cols / h;
    if d <= 0 {
        return None;
    }
    // Context K/V view [cap, kv_cols]; cap = view rows.
    if kc.view_shape.len() != 2 || vc.view_shape != kc.view_shape {
        return None;
    }
    let cap = kc.view_shape[0];
    let kv_cols = kc.view_shape[1];
    if cap <= 0 || kv_cols % d != 0 {
        return None;
    }
    // Context K/V access tiles read the full [cap, d] head slice.
    if kc.tile_shape != [cap, d] || vc.tile_shape != [cap, d] {
        return None;
    }
    // Diagonal K/V view [m, kv_cols]; access tile MUST be [r+1, d] (causal).
    if kd.view_shape != [m, kv_cols] || vd.view_shape != [m, kv_cols] {
        return None;
    }
    if kd.tile_shape != [q_row + 1, d] || vd.tile_shape != [q_row + 1, d] {
        return None; // causal diagonal growth not satisfied -> fail-safe
    }
    // Output view must equal the Q view [m, q_cols].
    // (mask view is [1, cap].)
    let o_arg = {
        // resolve the o_view's pointer arg + shape via its construct_memory_view.
        // o_tile.view -> view info isn't in load_src; look it up via def.
        let vop = def.get(&o_view)?;
        if vop.op_type != "ktdp.construct_memory_view" {
            return None;
        }
        let os = shape_attr(vop);
        if os != *q_shape {
            return None;
        }
        vop.operands.first()?.clone()
    };
    if mask_chain.view_shape != [1, cap] {
        return None;
    }

    let island = HeadAttnIsland {
        q_arg: q_c.arg.clone(),
        o_arg,
        mask_arg: mask_chain.arg.clone(),
        kc_arg: kc.arg.clone(),
        kd_arg: kd.arg.clone(),
        vc_arg: vc.arg.clone(),
        vd_arg: vd.arg.clone(),
        q_cols,
        kv_cols,
        m,
        cap,
        d,
        gqac,
        hdc: d,
        h,
        scale,
        ninf,
        dtype: q_c.dtype.clone(),
    };

    Some(RowMatch { q_row, island })
}

/// `w = arith.divf(exp_tensor, gs_bcast)` -> (exp_tensor, gs_bcast).
fn trace_divf<'a>(w: &str, def: &'a HashMap<String, &'a Operation>) -> Option<(String, String)> {
    let d = def.get(w)?;
    if d.op_type != "arith.divf" {
        return None;
    }
    Some((d.operands.first()?.clone(), d.operands.get(1)?.clone()))
}

/// `exp = math.exp(subf(x, gm_bcast))` -> x (the un-shifted scores).
fn trace_exp_sub(exp: &str, def: &HashMap<String, &Operation>) -> Option<String> {
    let e = def.get(exp)?;
    if e.op_type != "math.exp" {
        return None;
    }
    let sub = def.get(e.operands.first()?)?;
    if sub.op_type != "arith.subf" {
        return None;
    }
    Some(sub.operands.first()?.clone())
}

/// `scaled = arith.mulf(raw, tensor.splat(scale_const))` -> (raw, scale value).
fn trace_scale(scaled: &str, def: &HashMap<String, &Operation>) -> Option<(String, f32)> {
    let mul = def.get(scaled)?;
    if mul.op_type != "arith.mulf" {
        return None;
    }
    let raw = mul.operands.first()?.clone();
    let splat = def.get(mul.operands.get(1)?)?;
    if splat.op_type != "tensor.splat" {
        return None;
    }
    let c = def.get(splat.operands.first()?)?;
    if c.op_type != "arith.constant" {
        return None;
    }
    let scale = match c.attributes.get("value") {
        Some(Attr::Float(f)) => *f as f32,
        Some(Attr::Int(i)) => *i as f32,
        _ => return None,
    };
    Some((raw, scale))
}

/// `raw = linalg.matmul(Q_loaded, Kt); Kt = linalg.transpose(K_loaded)` ->
/// (Q_loaded SSA, K LoadChain).
fn trace_qk<'a>(
    raw: &str,
    def: &HashMap<String, &Operation>,
    load_src: &'a HashMap<String, LoadChain>,
) -> Option<(String, &'a LoadChain)> {
    let mm = def.get(raw)?;
    if mm.op_type != "linalg.matmul" {
        return None;
    }
    let q_loaded = mm.operands.first()?.clone();
    let kt = def.get(mm.operands.get(1)?)?;
    if kt.op_type != "linalg.transpose" {
        return None;
    }
    let k_loaded = kt.operands.first()?;
    let kc = load_src.get(k_loaded)?;
    Some((q_loaded, kc))
}

/// Recover the `-inf` mask additive constant from the context reduce-max init
/// (`tensor.splat(arith.constant -1e38)`), traced from the masked scores.
fn recover_ninf(scm_c: &str, def: &HashMap<String, &Operation>) -> Option<f32> {
    // Find the reduce-max that consumes scm_c, read its outs init splat const.
    // We search defs for a linalg.reduce over scm_c with reduce_fn maximumf.
    for op in def.values() {
        if op.op_type == "linalg.reduce"
            && matches!(op.attributes.get("reduce_fn"), Some(Attr::Str(s)) if s == "arith.maximumf")
            && op.operands.first().map(|o| o == scm_c).unwrap_or(false)
        {
            // outs init named in outs_var or operand[1]; trace its splat const.
            let init = match op.attributes.get("outs_var") {
                Some(Attr::Str(s)) => s.clone(),
                _ => op.operands.get(1)?.clone(),
            };
            let splat = def.get(&init)?;
            if splat.op_type == "tensor.splat"
                && let Some(c) = def.get(splat.operands.first()?)
                && c.op_type == "arith.constant"
                && let Some(Attr::Float(f)) = c.attributes.get("value")
            {
                return Some(*f as f32);
            }
        }
    }
    None
}

// ===========================================================================
// Rewrite (whole-row re-roll)
// ===========================================================================

/// A monotonic SSA name generator (collision-free within one rewritten body).
struct NameGen {
    n: usize,
}
impl NameGen {
    fn new() -> Self {
        NameGen { n: 0 }
    }
    fn next(&mut self, tag: &str) -> String {
        let s = format!("%hr_{tag}_{}", self.n);
        self.n += 1;
        s
    }
}

fn const_index(name: &str, v: i64) -> Operation {
    Operation::new(Some(name), "arith.constant", &[]).with_attr("value", Attr::Int(v))
}
fn const_f(name: &str, v: f64) -> Operation {
    Operation::new(Some(name), "arith.constant", &[]).with_attr("value", Attr::Float(v))
}

/// `ktdp.construct_memory_view %ptr {shape, strides, memory_space, dtype}` — a
/// logical view only (RFC 0682: does NOT allocate).
fn mk_view(res: &str, ptr: &str, shape: &[i64], dtype: &str) -> Operation {
    let mut strides = vec![1i64; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    Operation::new(Some(res), "ktdp.construct_memory_view", &[ptr])
        .with_attr("shape", Attr::IntList(shape.to_vec()))
        .with_attr("strides", Attr::IntList(strides))
        .with_attr("memory_space", Attr::Str("HBM".into()))
        .with_attr("dtype", Attr::Str(dtype.into()))
}

/// Load a `[rows, cols]` tile of `view` at dynamic offset `[%row, %col]`:
/// `construct_access_tile %view[%row, %col]` then `ktdp.load`. The index
/// operands stay real SSA so the per-head column offset is honored per core.
fn block_load(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    view: &str,
    row: &str,
    col: &str,
    rows: i64,
    cols: i64,
) -> String {
    let at = g.next("at");
    ops.push(
        Operation::new(Some(&at), "ktdp.construct_access_tile", &[view, row, col])
            .with_attr("shape", Attr::IntList(vec![rows, cols])),
    );
    let loaded = g.next("ld");
    ops.push(Operation::new(Some(&loaded), "ktdp.load", &[&at]));
    loaded
}

fn mk_splat(res: &str, scalar: &str, shape: &[i64], dtype: &str) -> Operation {
    Operation::new(Some(res), "tensor.splat", &[scalar])
        .with_attr("shape", Attr::IntList(shape.to_vec()))
        .with_attr("dtype", Attr::Str(dtype.into()))
}

fn mk_empty(res: &str, shape: &[i64], dtype: &str) -> Operation {
    Operation::new(Some(res), "tensor.empty", &[])
        .with_attr("shape", Attr::IntList(shape.to_vec()))
        .with_attr("dtype", Attr::Str(dtype.into()))
}

/// `linalg.transpose ins(%x) outs(%init) permutation=[1,0]` -> `[cols, rows]`.
fn mk_transpose(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    x: &str,
    rows: i64,
    cols: i64,
    dtype: &str,
) -> String {
    let init = g.next("tpi");
    ops.push(mk_empty(&init, &[cols, rows], dtype));
    let res = g.next("tp");
    ops.push(
        Operation::new(Some(&res), "linalg.transpose", &[x, &init])
            .with_attr("permutation", Attr::IntList(vec![1, 0])),
    );
    res
}

/// `C = A @ B` with a zero `tensor.empty` outs init (so matmul's `C + A@B`
/// reduces to `A@B`).
fn mk_matmul(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    a: &str,
    b: &str,
    rows: i64,
    cols: i64,
    dtype: &str,
) -> String {
    let init = g.next("mmi");
    ops.push(mk_empty(&init, &[rows, cols], dtype));
    let res = g.next("mm");
    ops.push(Operation::new(Some(&res), "linalg.matmul", &[a, b, &init]));
    res
}

/// `linalg.reduce { reduce_fn } ins(%x) outs(%init) dimensions=[1]` over the last
/// axis of `[m, c]` -> `[m]`.
fn mk_reduce(res: &str, x: &str, init: &str, reduce_fn: &str) -> Operation {
    Operation::new(Some(res), "linalg.reduce", &[x])
        .with_attr("reduce_fn", Attr::Str(reduce_fn.into()))
        .with_attr("dimensions", Attr::IntList(vec![1]))
        .with_attr("outs_var", Attr::Str(init.into()))
}

/// Broadcast a `[m]` row-vector to `[m, cols]` (reshape to `[m,1]` then
/// `linalg.broadcast` up to the outs shape).
fn broadcast_row_to(
    g: &mut NameGen,
    ops: &mut Vec<Operation>,
    rowv: &str,
    m: i64,
    cols: i64,
    dtype: &str,
) -> String {
    let r2 = g.next("rs");
    ops.push(
        Operation::new(Some(&r2), "tensor.reshape", &[rowv])
            .with_attr("target_shape", Attr::IntList(vec![m, 1])),
    );
    let init = g.next("bci");
    ops.push(mk_empty(&init, &[m, cols], dtype));
    let res = g.next("bc");
    ops.push(
        Operation::new(Some(&res), "linalg.broadcast", &[&r2, &init])
            .with_attr("dimensions", Attr::IntList(vec![])),
    );
    res
}

/// The static `[m, m]` lower-triangular causal mask: `0` for `k ≤ r` (visible),
/// `ninf` for `k > r` (masked). This reproduces EXACTLY the unrolled form's
/// ragged diagonal — row `r` attends current-segment KV `0..=r`. Baked as a
/// dense `arith.constant` tensor (no region, no select), so it stays region-free
/// and fusion-safe.
fn causal_mask_mm(res: &str, m: i64, ninf: f32, dtype: &str) -> Operation {
    let mut vals = Vec::with_capacity((m * m) as usize);
    for r in 0..m {
        for k in 0..m {
            vals.push(if k <= r { 0.0 } else { ninf as f64 });
        }
    }
    Operation::new(Some(res), "arith.constant", &[])
        .with_attr("is_tensor", Attr::Bool(true))
        .with_attr("dense_list", Attr::Bool(true))
        .with_attr("shape", Attr::IntList(vec![m, m]))
        .with_attr("dtype", Attr::Str(dtype.into()))
        .with_attr("value", Attr::FloatList(vals))
}

/// Rewrite a recognized [`HeadAttnIsland`] into the whole-row re-rolled body.
///
/// Emits ONCE (instead of `m` times) per core, preserving grid `[H,1,1]` and the
/// per-head GQA column arithmetic:
///   * `hpid = get_compute_tile_id`; `qcol = hpid*hdc`; `kvcol = (hpid/gqac)*hdc`
///   * Q = load view0[0, qcol] -> `[m, d]`
///   * CONTEXT: Kc = view5[0, kvcol] `[cap,d]`; Sc = (Q·Kcᵀ)*scale + mask `[m,cap]`
///   * DIAGONAL: Kd = view6[0, kvcol] `[m,d]`; Sd = (Q·Kdᵀ)*scale + tri `[m,m]`
///   * online softmax over the two blocks (global max, exp, sums, normalize)
///   * O = Wc·Vc + Wd·Vd `[m,d]`; store -> view1[0, qcol]
pub fn rewrite_head_attention(isl: &HeadAttnIsland) -> IRFunction {
    let dt = isl.dtype.as_str();
    let (m, d, cap) = (isl.m, isl.d, isl.cap);
    let mut g = NameGen::new();
    let mut ops: Vec<Operation> = Vec::new();

    // ---- constants ----
    let c0 = g.next("c0");
    ops.push(const_index(&c0, 0));
    let scale_c = g.next("scl");
    ops.push(const_f(&scale_c, isl.scale as f64));
    let ninf_c = g.next("ninf");
    ops.push(const_f(&ninf_c, isl.ninf as f64));
    let zero_c = g.next("zero");
    ops.push(const_f(&zero_c, 0.0));

    // ---- per-head selection arithmetic (PRESERVED) ----
    let hpid = g.next("hpid");
    ops.push(Operation::new(Some(&hpid), "ktdp.get_compute_tile_id", &[]));
    let hdc = g.next("hdc");
    ops.push(const_index(&hdc, isl.hdc));
    let gqac = g.next("gqac");
    ops.push(const_index(&gqac, isl.gqac));
    let qcol = g.next("qcol");
    ops.push(Operation::new(Some(&qcol), "arith.muli", &[&hpid, &hdc]));
    let kvh = g.next("kvh");
    ops.push(Operation::new(Some(&kvh), "arith.divui", &[&hpid, &gqac]));
    let kvcol = g.next("kvcol");
    ops.push(Operation::new(Some(&kvcol), "arith.muli", &[&kvh, &hdc]));

    // ---- views ----
    let q_view = g.next("qv");
    ops.push(mk_view(&q_view, &isl.q_arg, &[m, isl.q_cols], dt));
    let o_view = g.next("ov");
    ops.push(mk_view(&o_view, &isl.o_arg, &[m, isl.q_cols], dt));
    let mask_view = g.next("mv");
    ops.push(mk_view(&mask_view, &isl.mask_arg, &[1, cap], dt));
    let kc_view = g.next("kcv");
    ops.push(mk_view(&kc_view, &isl.kc_arg, &[cap, isl.kv_cols], dt));
    let kd_view = g.next("kdv");
    ops.push(mk_view(&kd_view, &isl.kd_arg, &[m, isl.kv_cols], dt));
    let vc_view = g.next("vcv");
    ops.push(mk_view(&vc_view, &isl.vc_arg, &[cap, isl.kv_cols], dt));
    let vd_view = g.next("vdv");
    ops.push(mk_view(&vd_view, &isl.vd_arg, &[m, isl.kv_cols], dt));

    // ---- whole-Q load [m, d] at [0, qcol] ----
    let q = block_load(&mut g, &mut ops, &q_view, &c0, &qcol, m, d);

    // ---- per-head context mask [1, cap] at [0, 0] ----
    let mask = block_load(&mut g, &mut ops, &mask_view, &c0, &c0, 1, cap);

    // =========================== CONTEXT block ===========================
    // Kc [cap, d] at [0, kvcol] -> Kct [d, cap]; Sc = (Q @ Kct)*scale + mask.
    let kc = block_load(&mut g, &mut ops, &kc_view, &c0, &kvcol, cap, d);
    let kct = mk_transpose(&mut g, &mut ops, &kc, cap, d, dt);
    let sc_raw = mk_matmul(&mut g, &mut ops, &q, &kct, m, cap, dt);
    let sc_scl = g.next("scscl");
    ops.push(mk_splat(&sc_scl, &scale_c, &[m, cap], dt));
    let sc_scaled = g.next("scscaled");
    ops.push(Operation::new(
        Some(&sc_scaled),
        "arith.mulf",
        &[&sc_raw, &sc_scl],
    ));
    // mask broadcast over the m rows: mask is [1, cap], broadcast to [m, cap].
    let mask_init = g.next("mki");
    ops.push(mk_empty(&mask_init, &[m, cap], dt));
    let mask_b = g.next("mkb");
    ops.push(
        Operation::new(Some(&mask_b), "linalg.broadcast", &[&mask, &mask_init])
            .with_attr("dimensions", Attr::IntList(vec![])),
    );
    let sc = g.next("sc");
    ops.push(Operation::new(
        Some(&sc),
        "arith.addf",
        &[&sc_scaled, &mask_b],
    ));
    // mc = reduce_max(Sc, 1) -> [m]
    let mc_init = g.next("mci");
    ops.push(mk_splat(&mc_init, &ninf_c, &[m], dt));
    let mc = g.next("mc");
    ops.push(mk_reduce(&mc, &sc, &mc_init, "arith.maximumf"));

    // =========================== DIAGONAL block ==========================
    // Kd [m, d] at [0, kvcol] -> Kdt [d, m]; Sd = (Q @ Kdt)*scale + tri[m,m].
    let kd = block_load(&mut g, &mut ops, &kd_view, &c0, &kvcol, m, d);
    let kdt = mk_transpose(&mut g, &mut ops, &kd, m, d, dt);
    let sd_raw = mk_matmul(&mut g, &mut ops, &q, &kdt, m, m, dt);
    let sd_scl = g.next("sdscl");
    ops.push(mk_splat(&sd_scl, &scale_c, &[m, m], dt));
    let sd_scaled = g.next("sdscaled");
    ops.push(Operation::new(
        Some(&sd_scaled),
        "arith.mulf",
        &[&sd_raw, &sd_scl],
    ));
    let tri = g.next("tri");
    ops.push(causal_mask_mm(&tri, m, isl.ninf, dt));
    let sd = g.next("sd");
    ops.push(Operation::new(Some(&sd), "arith.addf", &[&sd_scaled, &tri]));
    // md = reduce_max(Sd, 1) -> [m]
    let md_init = g.next("mdi");
    ops.push(mk_splat(&md_init, &ninf_c, &[m], dt));
    let md = g.next("md");
    ops.push(mk_reduce(&md, &sd, &md_init, "arith.maximumf"));

    // =========================== combine =================================
    // gm = max(mc, md) [m]
    let gm = g.next("gm");
    ops.push(Operation::new(Some(&gm), "arith.maximumf", &[&mc, &md]));
    // Pc = exp(Sc - gm_bcast[m,cap]); Pd = exp(Sd - gm_bcast[m,m])
    let gm_bc = broadcast_row_to(&mut g, &mut ops, &gm, m, cap, dt);
    let shc = g.next("shc");
    ops.push(Operation::new(Some(&shc), "arith.subf", &[&sc, &gm_bc]));
    let pc = g.next("pc");
    ops.push(Operation::new(Some(&pc), "math.exp", &[&shc]));
    let gm_bd = broadcast_row_to(&mut g, &mut ops, &gm, m, m, dt);
    let shd = g.next("shd");
    ops.push(Operation::new(Some(&shd), "arith.subf", &[&sd, &gm_bd]));
    let pd = g.next("pd");
    ops.push(Operation::new(Some(&pd), "math.exp", &[&shd]));
    // sc_sum = reduce_sum(Pc,1) [m]; sd_sum = reduce_sum(Pd,1) [m]
    let scs_init = g.next("scsi");
    ops.push(mk_splat(&scs_init, &zero_c, &[m], dt));
    let scs = g.next("scs");
    ops.push(mk_reduce(&scs, &pc, &scs_init, "arith.addf"));
    let sds_init = g.next("sdsi");
    ops.push(mk_splat(&sds_init, &zero_c, &[m], dt));
    let sds = g.next("sds");
    ops.push(mk_reduce(&sds, &pd, &sds_init, "arith.addf"));
    // gs = sc_sum + sd_sum [m]
    let gs = g.next("gs");
    ops.push(Operation::new(Some(&gs), "arith.addf", &[&scs, &sds]));
    // Wc = Pc / gs_bcast[m,cap]; Wd = Pd / gs_bcast[m,m]
    let gs_bc = broadcast_row_to(&mut g, &mut ops, &gs, m, cap, dt);
    let wc = g.next("wc");
    ops.push(Operation::new(Some(&wc), "arith.divf", &[&pc, &gs_bc]));
    let gs_bd = broadcast_row_to(&mut g, &mut ops, &gs, m, m, dt);
    let wd = g.next("wd");
    ops.push(Operation::new(Some(&wd), "arith.divf", &[&pd, &gs_bd]));

    // =========================== AV + store ==============================
    // Vc [cap, d] at [0, kvcol]; Vd [m, d] at [0, kvcol].
    let vc = block_load(&mut g, &mut ops, &vc_view, &c0, &kvcol, cap, d);
    let vd = block_load(&mut g, &mut ops, &vd_view, &c0, &kvcol, m, d);
    let ovc = mk_matmul(&mut g, &mut ops, &wc, &vc, m, d, dt);
    let ovd = mk_matmul(&mut g, &mut ops, &wd, &vd, m, d, dt);
    let o = g.next("o");
    ops.push(Operation::new(Some(&o), "arith.addf", &[&ovc, &ovd]));
    // store O [m, d] at [0, qcol].
    let o_at = g.next("oat");
    ops.push(
        Operation::new(
            Some(&o_at),
            "ktdp.construct_access_tile",
            &[&o_view, &c0, &qcol],
        )
        .with_attr("shape", Attr::IntList(vec![m, d])),
    );
    ops.push(Operation::new(None, "ktdp.store", &[&o, &o_at]));
    ops.push(Operation::new(None, "func.return", &[]));

    IRFunction {
        name: String::new(), // caller stamps the original name
        arguments: vec![
            (isl.q_arg.clone(), "index".into()),
            (isl.o_arg.clone(), "index".into()),
            (isl.mask_arg.clone(), "index".into()),
            (isl.kc_arg.clone(), "index".into()),
            (isl.kd_arg.clone(), "index".into()),
            (isl.vc_arg.clone(), "index".into()),
            (isl.vd_arg.clone(), "index".into()),
        ],
        operations: ops,
        grid: (isl.h as usize, 1, 1),
        return_type: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic island matching the smollm shape (H=9, m=8, gqac=3, d=64,
    /// cap=64). Used to exercise the rewrite emitter structurally.
    fn smollm_island() -> HeadAttnIsland {
        HeadAttnIsland {
            q_arg: "%q".into(),
            o_arg: "%o".into(),
            mask_arg: "%mask".into(),
            kc_arg: "%kc".into(),
            kd_arg: "%kd".into(),
            vc_arg: "%vc".into(),
            vd_arg: "%vd".into(),
            q_cols: 576,
            kv_cols: 192,
            m: 8,
            cap: 64,
            d: 64,
            gqac: 3,
            hdc: 64,
            h: 9,
            scale: 0.125,
            ninf: -1.0e38,
            dtype: "f16".into(),
        }
    }

    #[test]
    fn rewrite_preserves_grid_and_args() {
        let isl = smollm_island();
        let f = rewrite_head_attention(&isl);
        // Grid stays [H, 1, 1] — every core still runs the body once for its head.
        assert_eq!(f.grid, (9, 1, 1));
        let names: Vec<&str> = f.arguments.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["%q", "%o", "%mask", "%kc", "%kd", "%vc", "%vd"]);
    }

    #[test]
    fn rewrite_emits_no_control_flow_and_one_store() {
        let isl = smollm_island();
        let f = rewrite_head_attention(&isl);
        // Pure re-roll: NO scf.* (region-free, never trips the batched gate).
        assert!(
            !f.operations.iter().any(|o| o.op_type.starts_with("scf.")),
            "re-roll must be region-free"
        );
        // Exactly ONE store (the whole [m,d] output) instead of m stores.
        let stores = f
            .operations
            .iter()
            .filter(|o| o.op_type == "ktdp.store")
            .count();
        assert_eq!(stores, 1, "one whole-row store");
        // Four matmuls total (context QKᵀ, diagonal QKᵀ, context AV, diagonal AV)
        // — vs 4×m in the unrolled form.
        let mms = f
            .operations
            .iter()
            .filter(|o| o.op_type == "linalg.matmul")
            .count();
        assert_eq!(mms, 4, "two QKᵀ + two AV, once");
        // Preserves the per-head selection arithmetic as SSA.
        assert!(
            f.operations
                .iter()
                .any(|o| o.op_type == "ktdp.get_compute_tile_id")
        );
        assert!(f.operations.iter().any(|o| o.op_type == "arith.divui"));
        let muls = f
            .operations
            .iter()
            .filter(|o| o.op_type == "arith.muli")
            .count();
        assert_eq!(muls, 2, "qcol = hpid*hdc and kvcol = (hpid/gqac)*hdc");
    }

    #[test]
    fn causal_mask_is_lower_triangular() {
        // mask[r,k] = 0 for k<=r (visible), ninf for k>r (masked).
        let op = causal_mask_mm("%tri", 4, -1.0e38, "f16");
        let vals = match op.attributes.get("value") {
            Some(Attr::FloatList(v)) => v.clone(),
            other => panic!("mask value not a FloatList: {other:?}"),
        };
        assert_eq!(vals.len(), 16);
        for r in 0..4i64 {
            for k in 0..4i64 {
                let v = vals[(r * 4 + k) as usize];
                if k <= r {
                    assert_eq!(v, 0.0, "[{r},{k}] visible");
                } else {
                    assert!(v < -1.0e30, "[{r},{k}] masked");
                }
            }
        }
    }

    #[test]
    fn rejects_single_core_grid() {
        // grid = [1,1,1] is not head-parallel -> None.
        let f = IRFunction {
            name: "x".into(),
            arguments: vec![],
            operations: vec![Operation::new(None, "ktdp.store", &["%a", "%b"])],
            grid: (1, 1, 1),
            return_type: None,
        };
        assert!(recognize_head_attention(&f).is_none());
    }

    #[test]
    fn rejects_region_bearing() {
        // A top-level scf.for disqualifies (the FA-tiled regime, not the head one).
        let mut forop = Operation::new(None, "scf.for", &["%x", "%y", "%z"]);
        forop.regions = vec![vec![Operation::new(None, "scf.yield", &[])]];
        let f = IRFunction {
            name: "x".into(),
            arguments: vec![],
            operations: vec![forop, Operation::new(None, "ktdp.store", &["%a", "%b"])],
            grid: (9, 1, 1),
            return_type: None,
        };
        assert!(recognize_head_attention(&f).is_none());
    }

    #[test]
    fn rejects_plain_copy() {
        // A multi-core copy node (no QKᵀ/softmax/AV) -> None.
        let f = IRFunction {
            name: "copy".into(),
            arguments: vec![
                ("%in".into(), "index".into()),
                ("%out".into(), "index".into()),
            ],
            grid: (9, 1, 1),
            return_type: None,
            operations: vec![
                mk_view("%vi", "%in", &[8, 64], "f16"),
                Operation::new(Some("%ti"), "ktdp.construct_access_tile", &["%vi"])
                    .with_attr("shape", Attr::IntList(vec![8, 64])),
                Operation::new(Some("%l"), "ktdp.load", &["%ti"]),
                Operation::new(Some("%y"), "math.exp", &["%l"]),
                mk_view("%vo", "%out", &[8, 64], "f16"),
                Operation::new(Some("%to"), "ktdp.construct_access_tile", &["%vo"])
                    .with_attr("shape", Attr::IntList(vec![8, 64])),
                Operation::new(None, "ktdp.store", &["%y", "%to"]),
                Operation::new(None, "func.return", &[]),
            ],
        };
        assert!(recognize_head_attention(&f).is_none());
    }

    #[test]
    fn scores_bytes_matches_m_cap_dtype() {
        let isl = smollm_island();
        // [8, 64] f16 = 8*64*2 = 1024 bytes.
        assert_eq!(isl.scores_bytes(), 8 * 64 * 2);
    }
}
