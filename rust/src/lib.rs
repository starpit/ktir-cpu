// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! KTIR CPU validation interpreter — Rust port of the Python `ktir_cpu` package
//! (RFC 0682). This is slice 1: a compiling vertical cut through dtypes, affine,
//! IR, a minimal tile, dialect dispatch, the `arith` dialect, and the execution
//! loop. Module names mirror the Python package for diffability.

pub mod affine;
pub mod dialects;
pub mod dtypes;
pub mod interpreter;
pub mod ir;
pub mod memory;
pub mod memref;
pub mod tile;
