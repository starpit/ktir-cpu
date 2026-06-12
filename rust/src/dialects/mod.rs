// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Dialect dispatch — Rust port of the handler registry in
//! `ktir_cpu/dialects/registry.py`.
//!
//! Python registers handlers at import time via a `@register` decorator that
//! mutates a global dict. Rust uses the explicit-table approach (Option A from
//! the design sketch): each dialect module exposes `register(&mut Dispatch)`,
//! and [`Dispatch::new`] assembles them. Greppable, no macro magic, and the
//! registration is visible rather than hidden in attribute macros.

pub mod arith;
pub mod func;
pub mod ktdp;

use std::collections::HashMap;

use crate::context::CoreContext;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Value};

/// Handler signature. Mirrors Python's `HandlerFn = (op, context, env) -> Any`:
/// reads operands via `ctx.get_value`, runs nested regions via the dispatch
/// table in `env`, and returns the value to bind to `op.result` (or `None`).
pub type HandlerFn =
    fn(&Operation, &mut CoreContext, &ExecutionEnv) -> Result<Option<Value>, String>;

/// Op-name -> handler table, plus the parallel latency-category table that the
/// Python registry keeps in lockstep.
pub struct Dispatch {
    handlers: HashMap<&'static str, HandlerFn>,
    latency: HashMap<&'static str, LatencyCategory>,
}

/// Subset of `ktir_cpu/latency.py`'s `LatencyCategory` StrEnum needed so far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyCategory {
    Zero,
    ComputeFloat,
    ComputeInt,
}

impl Dispatch {
    /// Build the table by letting each dialect register its ops.
    pub fn new() -> Self {
        let mut d = Dispatch {
            handlers: HashMap::new(),
            latency: HashMap::new(),
        };
        arith::register(&mut d);
        func::register(&mut d);
        ktdp::register(&mut d);
        d
    }

    /// Called by dialect modules. Mirrors the `@register(name, latency_category)` decorator.
    pub fn register(&mut self, op_name: &'static str, cat: LatencyCategory, f: HandlerFn) {
        self.handlers.insert(op_name, f);
        self.latency.insert(op_name, cat);
    }

    /// Look up a handler — mirrors `dispatch(op_name)`.
    pub fn handler(&self, op_name: &str) -> Option<HandlerFn> {
        self.handlers.get(op_name).copied()
    }

    /// Latency category, defaulting to `Zero` — mirrors `get_latency_category`.
    pub fn latency_category(&self, op_name: &str) -> LatencyCategory {
        self.latency
            .get(op_name)
            .copied()
            .unwrap_or(LatencyCategory::Zero)
    }
}

impl Default for Dispatch {
    fn default() -> Self {
        Self::new()
    }
}
