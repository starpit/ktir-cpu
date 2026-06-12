// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Execution orchestrator — slice of `ktir_cpu/interpreter.py` + the per-core
//! scope from `ktir_cpu/grid.py`'s `CoreContext`.
//!
//! This drives a flat op list through the dispatch table. The full port grows
//! `Scope` into a stack-of-scopes `CoreContext` and threads grid/memory/latency
//! through an `ExecutionEnv`; regions return a `StepResult` (Done/Yield/
//! AwaitRecv) so the comm ops can suspend a core — see the design notes.

use std::collections::HashMap;

use crate::dialects::Dispatch;
use crate::ir::{Operation, Value};

/// Per-core SSA binding map. Stand-in for `CoreContext`'s value scope.
#[derive(Default)]
pub struct Scope {
    values: HashMap<String, Value>,
}

impl Scope {
    pub fn new() -> Self {
        Scope::default()
    }

    /// Bind an SSA name (with or without leading `%`) to a value.
    pub fn set(&mut self, name: &str, val: Value) {
        self.values.insert(normalize(name), val);
    }

    /// Look up an SSA value, erroring if unbound — mirrors a scope miss raising.
    pub fn get(&self, name: &str) -> Result<&Value, String> {
        self.values
            .get(&normalize(name))
            .ok_or_else(|| format!("undefined SSA value: {name}"))
    }
}

fn normalize(name: &str) -> String {
    name.trim_start_matches('%').to_string()
}

/// Run a straight-line op list against a scope using the dispatch table.
/// Each op's result (if any) is bound to its `result` name.
pub fn execute_ops(
    ops: &[Operation],
    dispatch: &Dispatch,
    scope: &mut Scope,
) -> Result<(), String> {
    for op in ops {
        let handler = dispatch
            .handler(&op.op_type)
            .ok_or_else(|| format!("no handler registered for op '{}'", op.op_type))?;
        let produced = handler(op, scope)?;
        match (&op.result, produced) {
            (Some(name), Some(val)) => scope.set(name, val),
            (Some(name), None) => {
                return Err(format!("op '{}' has result {name} but produced no value", op.op_type))
            }
            (None, _) => {} // value-less op (e.g. store); ignore any return
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::DType;
    use crate::ir::{Attr, Scalar};
    use crate::tile::Tile;

    #[test]
    fn scalar_constant_fold_chain() {
        // %a = constant 2.0 ; %b = constant 3.0 ; %c = addf %a,%b ; %d = mulf %c,%a
        let ops = vec![
            Operation::new(Some("%a"), "arith.constant", &[]).with_attr("value", Attr::Float(2.0)),
            Operation::new(Some("%b"), "arith.constant", &[]).with_attr("value", Attr::Float(3.0)),
            Operation::new(Some("%c"), "arith.addf", &["%a", "%b"]),
            Operation::new(Some("%d"), "arith.mulf", &["%c", "%a"]),
        ];
        let dispatch = Dispatch::new();
        let mut scope = Scope::new();
        execute_ops(&ops, &dispatch, &mut scope).unwrap();
        // (2+3)*2 = 10
        match scope.get("%d").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 10.0),
            other => panic!("expected F32(10.0), got {other:?}"),
        }
    }

    #[test]
    fn elementwise_tile_add() {
        let mut scope = Scope::new();
        scope.set(
            "%x",
            Value::Tile(Tile::compute(vec![1.0, 2.0, 3.0], DType::F32, vec![3])),
        );
        scope.set(
            "%y",
            Value::Tile(Tile::compute(vec![10.0, 20.0, 30.0], DType::F32, vec![3])),
        );
        let ops = vec![Operation::new(Some("%z"), "arith.addf", &["%x", "%y"])];
        execute_ops(&ops, &Dispatch::new(), &mut scope).unwrap();
        match scope.get("%z").unwrap() {
            Value::Tile(t) => assert_eq!(t.data, vec![11.0, 22.0, 33.0]),
            other => panic!("expected tile, got {other:?}"),
        }
    }

    #[test]
    fn unknown_op_errors() {
        let ops = vec![Operation::new(Some("%z"), "ktdp.not_yet", &[])];
        let err = execute_ops(&ops, &Dispatch::new(), &mut Scope::new()).unwrap_err();
        assert!(err.contains("no handler registered"));
    }
}
