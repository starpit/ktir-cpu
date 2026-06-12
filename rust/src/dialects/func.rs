// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `func` dialect — the function terminator. `return` carries operands but has
//! no SSA result; in this slice it's a value-less no-op so straight-line
//! functions execute cleanly. (Capturing the returned values into the
//! interpreter's function-result handling lands with the grid/interpreter slice.)

use super::{Dispatch, LatencyCategory};
use crate::context::CoreContext;
use crate::env::ExecutionEnv;
use crate::ir::{Operation, Value};

pub fn register(d: &mut Dispatch) {
    d.register("return", LatencyCategory::Zero, ret);
    d.register("func.return", LatencyCategory::Zero, ret);
}

fn ret(_op: &Operation, _ctx: &mut CoreContext, _env: &ExecutionEnv) -> Result<Option<Value>, String> {
    Ok(None)
}
