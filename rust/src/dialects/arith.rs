// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `arith` dialect handlers — partial port of `ktir_cpu/dialects/arith_ops.py`.
//!
//! Slice covers `constant`, `addf`, `mulf`, `addi`. Float ops work
//! element-wise on tiles *or* on scalars, mirroring how the Python handlers
//! accept either a NumPy array or a Python scalar.

use super::{Dispatch, LatencyCategory};
use crate::dtypes::DType;
use crate::interpreter::Scope;
use crate::ir::{Attr, Operation, Scalar, Value};
use crate::tile::Tile;

/// Register every handler this module owns. Called by `Dispatch::new`.
pub fn register(d: &mut Dispatch) {
    d.register("arith.constant", LatencyCategory::Zero, constant);
    d.register("arith.addf", LatencyCategory::ComputeFloat, addf);
    d.register("arith.mulf", LatencyCategory::ComputeFloat, mulf);
    d.register("arith.addi", LatencyCategory::ComputeInt, addi);
}

/// `%c = arith.constant <value> : <type>` — value carried in the `value` attr.
fn constant(op: &Operation, _scope: &mut Scope) -> Result<Option<Value>, String> {
    let v = op
        .attributes
        .get("value")
        .ok_or("arith.constant missing 'value' attribute")?;
    let val = match v {
        Attr::Float(f) => Value::Scalar(Scalar::F32(*f as f32)),
        Attr::Int(i) => Value::Scalar(Scalar::I64(*i)),
        Attr::Bool(b) => Value::Scalar(Scalar::Bool(*b)),
        other => return Err(format!("arith.constant: bad value attr {other:?}")),
    };
    Ok(Some(val))
}

fn addf(op: &Operation, scope: &mut Scope) -> Result<Option<Value>, String> {
    binary_float(op, scope, "arith.addf", |a, b| a + b)
}

fn mulf(op: &Operation, scope: &mut Scope) -> Result<Option<Value>, String> {
    binary_float(op, scope, "arith.mulf", |a, b| a * b)
}

fn addi(op: &Operation, scope: &mut Scope) -> Result<Option<Value>, String> {
    let (a, b) = two_operands(op, scope, "arith.addi")?;
    let (x, y) = (scalar_i64(a, "arith.addi")?, scalar_i64(b, "arith.addi")?);
    Ok(Some(Value::Scalar(Scalar::I64(x + y))))
}

// --- helpers -------------------------------------------------------------

fn two_operands<'s>(
    op: &Operation,
    scope: &'s Scope,
    name: &str,
) -> Result<(&'s Value, &'s Value), String> {
    if op.operands.len() != 2 {
        return Err(format!("{name} expects 2 operands, got {}", op.operands.len()));
    }
    let a = scope.get(&op.operands[0])?;
    let b = scope.get(&op.operands[1])?;
    Ok((a, b))
}

/// Float binary op accepting scalar+scalar or tile+tile (element-wise).
fn binary_float(
    op: &Operation,
    scope: &mut Scope,
    name: &str,
    f: fn(f32, f32) -> f32,
) -> Result<Option<Value>, String> {
    let (a, b) = two_operands(op, scope, name)?;
    match (a, b) {
        (Value::Scalar(x), Value::Scalar(y)) => {
            let (x, y) = (
                x.as_f32().ok_or_else(|| format!("{name}: non-float scalar"))?,
                y.as_f32().ok_or_else(|| format!("{name}: non-float scalar"))?,
            );
            Ok(Some(Value::Scalar(Scalar::F32(f(x, y)))))
        }
        (Value::Tile(x), Value::Tile(y)) => {
            if x.shape != y.shape {
                return Err(format!(
                    "{name}: shape mismatch {:?} vs {:?}",
                    x.shape, y.shape
                ));
            }
            let data: Vec<f32> = x.data.iter().zip(&y.data).map(|(&p, &q)| f(p, q)).collect();
            let dtype = if x.dtype == DType::F16 || y.dtype == DType::F16 {
                DType::F16
            } else {
                DType::F32
            };
            Ok(Some(Value::Tile(Tile::compute(data, dtype, x.shape.clone()))))
        }
        _ => Err(format!("{name}: operand kinds not both scalar or both tile")),
    }
}

fn scalar_i64(v: &Value, name: &str) -> Result<i64, String> {
    match v {
        Value::Scalar(s) => s.as_i64().ok_or_else(|| format!("{name}: non-int scalar")),
        Value::Index(i) => Ok(*i),
        _ => Err(format!("{name}: expected scalar/index operand")),
    }
}
