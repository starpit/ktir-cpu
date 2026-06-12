# ktir-cpu (Rust port)

A Rust port of the Python `ktir_cpu` validation interpreter (RFC 0682). This is
**slice 1**: a compiling vertical cut that establishes the foundation types and
proves the architecture end-to-end. Module names mirror the Python package so
the two can be diffed against each other as the port grows.

## What's here

| Module | Mirrors | Status |
|---|---|---|
| `dtypes` | `dtypes.py` | `DType` enum + alias parsing — done |
| `affine` | `affine.py` | `AffineExpr`/`AffineMap`/`BoxSet`/`AffineSet` eval + containment |
| `ir` | `ir_types.py` (op half) | `Value`, `Scalar`, `Attr`, `Operation`, `IRFunction`, `IRModule` |
| `tile` | `Tile` | minimal flat-`Vec<f32>` storage (see dtype fork note) |
| `dialects` | `dialects/registry.py` | explicit-table `Dispatch` |
| `dialects::arith` | `dialects/arith_ops.py` | `constant`, `addf`, `mulf`, `addi` |
| `interpreter` | `interpreter.py` + `grid.py` scope | `Scope` + straight-line `execute_ops` |

`Value` is the keystone — the tagged union replacing Python's `Any` across every
SSA binding and handler return.

## Not yet ported (next slices)

- `memref.rs` / remaining `tile.rs` — `MemRef`, `TileRef`, distributed variants,
  the `CoordinateSet` / `ParentRef` enums.
- `parser/` — the regex MLIR tokenizer (`parser.py` + `parser_ast.py`).
- `memory.rs`, `grid.rs`, `latency.rs`.
- The comm-op suspension model: regions returning `StepResult::{Done,Yield,
  AwaitRecv}` (replaces Python's generator that `yield`s `RecvRequest`).
- The dtype-storage fork in `tile.rs` (flat `Vec<f32>` → `ndarray::ArrayD`,
  plus the single-`f32` vs `TileData`-enum decision). See the header note there.

## Building

The repo's `~/.cargo/bin` rustup shims are broken (they point at a missing
`rustup`), so invoke the stable toolchain directly and put it on PATH for
cargo's own `rustc` call:

```sh
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"
cargo test
cargo clippy --all-targets
```

11 tests, clippy-clean.
