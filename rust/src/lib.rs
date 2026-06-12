// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! KTIR CPU validation interpreter — Rust port of the Python `ktir_cpu` package
//! (RFC 0682). Module names mirror the Python package for diffability.
//!
//! The execution contract is locked: handlers have signature
//! `(op, &mut CoreContext, &ExecutionEnv) -> Result<Option<Value>, String>`,
//! run nested regions via `interpreter::execute_region`, and the cross-core
//! comm seam lives in `comm` (only the top-level driver suspends). Dialect
//! implementations and the memory load/store data path build against these.

// Links the selected BLAS backend (Accelerate / OpenBLAS / MKL / BLIS) when a
// provider feature is on. `blas-src` must be referenced once at the crate root
// for its linker directives to take effect. See blas.rs.
#[cfg(feature = "blas")]
extern crate blas_src;

pub mod affine;
pub mod blas;
pub mod codec;
pub mod comm;
pub mod comm_sched;
pub mod context;
pub mod dialects;
pub mod dtypes;
pub mod env;
pub mod fxhash;
pub mod interpreter;
pub mod ir;
pub mod latency;
pub mod memory;
pub mod memref;
pub mod ops_memory;
pub mod parser;
pub mod parser_ast;
pub mod tile;
