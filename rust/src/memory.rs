// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Memory simulation constants — slice of `ktir_cpu/memory.py`.
//!
//! Only the HBM stick granularity is needed so far: it drives `byte_address`
//! and `split_addr` on `MemRef`. The full `HBMSimulator` / `LXScratchpad` /
//! `SpyreMemoryHierarchy` land in a later slice.

/// HBM "stick" (cache block) size in bytes. Mirrors `HBMSimulator.STICK_BYTES`.
pub const STICK_BYTES: usize = 128;
