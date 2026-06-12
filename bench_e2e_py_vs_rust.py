#!/usr/bin/env python3
"""End-to-end Python timing harness for the Python-vs-Rust ktir_cpu comparison.

Unlike bench_py_vs_rust.py (which times a single kernel's execute_function), this
runs a WHOLE model bundle through the interpreter — every node, with per-node
dispatch, argument marshaling, and inter-node tensor threading — so the number
reflects the interpreter itself, not just one kernel's compute.

Mirrors rust/benches/smollm2_bench.rs / tests/e2e_smollm2.rs exactly: same node
sweep, same attn_mask seeding, sources reset each pass. Reports ms per full-model
pass (one warm-up pass excluded; parse is cached per unique .mlir).

Run from repo root:  uv run python bench_e2e_py_vs_rust.py
Env: SMOLLM2_ITERS (timed passes, default 5), BUNDLE (default smollm2-135m).
"""
import json
import os
import time

import numpy as np

from ktir_cpu import KTIRInterpreter

BUNDLE = os.environ.get("BUNDLE", "smollm2-135m")
DIR = os.path.expanduser(f"~/.cache/cudaforge/ktir/{BUNDLE}")


def read_f32(path):
    return np.fromfile(path, dtype="<f4")


def main():
    manifest_path = os.path.join(DIR, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"bundle {BUNDLE} absent at {DIR} — skipping")
        return
    manifest = json.load(open(manifest_path))

    # tensor id -> (rows, cols, is_source)
    shape = {}
    for t in manifest["tensors"]:
        shape[t["id"]] = (t["rows"], t["cols"], t.get("is_source", False))

    # sources preloaded; attn_mask (runtime input) seeded with the all-zeros
    # causal mask, matching the Rust harness.
    sources = {}
    for tid, (_, _, is_src) in shape.items():
        if is_src:
            sources[tid] = read_f32(os.path.join(DIR, f"t{tid}.bin"))
    mask_id = manifest.get("attn_mask")
    if mask_id is not None:
        r, c, _ = shape[mask_id]
        sources[mask_id] = np.zeros(r * c, dtype="<f4")

    nodes = manifest["nodes"]
    # One interpreter per unique .mlir, parsed once (parse excluded from timing).
    interps = {}
    for node in nodes:
        name = node["mlir"]
        if name not in interps:
            interp = KTIRInterpreter()
            interp.load(open(os.path.join(DIR, name)).read())
            interps[name] = interp

    def one_pass():
        buf = {tid: v.copy() for tid, v in sources.items()}
        for node in nodes:
            interp = interps[node["mlir"]]
            func = node["fn"]
            kwargs = {}
            outs = []
            for a in node["args"]:
                tid = a["tensor"]
                r, c, _ = shape[tid]
                is_out = a.get("is_output", False)
                if is_out:
                    data = np.zeros(r * c, dtype=np.float16)
                    outs.append((a["name"], tid))
                else:
                    data = buf[tid].astype(np.float16)
                kwargs[a["name"]] = data.reshape(r, c)
            result = interp.execute_function(func, **kwargs)
            for nm, tid in outs:
                buf[tid] = np.asarray(result[nm], dtype="<f4").reshape(-1)
        return buf

    iters = int(os.environ.get("SMOLLM2_ITERS", "5"))
    one_pass()  # warm-up (cache effects, lazy init) — excluded
    t0 = time.perf_counter()
    for _ in range(iters):
        one_pass()
    ms = (time.perf_counter() - t0) / iters * 1e3
    print(f"{BUNDLE} e2e (Python): {ms:.1f} ms/pass  ({len(nodes)} nodes, {iters} passes)")


if __name__ == "__main__":
    main()
