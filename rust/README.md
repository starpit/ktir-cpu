# ktir-cpu (Rust port)

A Rust port of the Python `ktir_cpu` validation interpreter (RFC 0682). A
compiling vertical cut that establishes the foundation types and proves the
architecture end-to-end. Module names mirror the Python package so the two can
be diffed against each other as the port grows.

## What's here

| Module | Mirrors | Status |
|---|---|---|
| `dtypes` | `dtypes.py` | `DType` enum + alias parsing — done |
| `affine` | `affine.py` | `AffineExpr`/`AffineMap`/`BoxSet`/`AffineSet` eval + containment |
| `ir` | `ir_types.py` (op half) | `Value`, `Scalar`, `Attr`, `Operation`, `IRFunction`, `IRModule` |
| `memref` | `ir_types.py` (memref half) | `MemRef`, `TileRef`, distributed variants, `MemorySpace`/`CoordinateSet`/`ParentRef` enums, `AccessTile` |
| `memory` | `memory.py` | `STICK_BYTES` (HBM stick granularity) |
| `tile` | `Tile` | minimal flat-`Vec<f32>` storage (see dtype fork note) |
| `dialects` | `dialects/registry.py` | explicit-table `Dispatch` |
| `dialects::arith` | `dialects/arith_ops.py` | `constant`, `addf`, `mulf`, `addi` |
| `dialects::ktdp` | `dialects/ktdp_ops.py` | `construct_memory_view`, `construct_access_tile` (single-allocation path) |
| `dialects::func` | (terminator) | `return` / `func.return` |
| `parser` | `parser.py` | module/func/grid/args extraction, multi-line op tokenizer, structural op parse, `arith.constant` + infix index-arith |
| `interpreter` | `interpreter.py` + `grid.py` scope | `Scope` + straight-line `execute_ops` |

The parser handles the **real** `examples/triton-ktir/vector_add_ktir.mlir`
structurally (correct func/grid/args and one `Operation` per multi-line
`ktdp.construct_*`), and parses+executes straight-line arith functions
end-to-end. See the test module in `parser.rs`.

`Value` is the keystone — the tagged union replacing Python's `Any` across every
SSA binding and handler return. `MemorySpace` as an enum makes the Python
`__post_init__` "lx_core_id only valid for LX" check structurally impossible to
violate.

## Not yet ported (next slices)

- Parser depth: nested regions (scf bodies), and dialect-specific **attribute**
  parsing for the affine attrs (`coordinate_set`, `base_map`, `memory_space`,
  `sizes:`/`strides:`). Until then a parsed `ktdp.construct_*` op has correct
  op_type/operands/result_type but not its affine attributes, so it parses
  structurally but is not yet executable end-to-end.
- The distributed memory path: `construct_distributed_memory_view`,
  `distributed_tile_access`, and `ktdp.load`/`store` (need a real
  `HBMSimulator` in `memory.rs`).
- `grid.rs`, `latency.rs`.
- The comm-op suspension model: regions returning `StepResult::{Done,Yield,
  AwaitRecv}` (replaces Python's generator that `yield`s `RecvRequest`).
- The dtype-storage fork in `tile.rs` (flat `Vec<f32>` → `ndarray::ArrayD`,
  plus the single-`f32` vs `TileData`-enum decision). See the header note there.
- Symbolic `access_tile_set` resolution (rejected for now, matching the Python
  `NotImplementedError`).

## Building

The repo's `~/.cargo/bin` rustup shims are broken (they point at a missing
`rustup`), so invoke the stable toolchain directly and put it on PATH for
cargo's own `rustc` call:

```sh
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"
cargo test
cargo clippy --all-targets
```

22 tests, clippy-clean.
