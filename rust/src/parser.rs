// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! KTIR MLIR text parser — Rust port of `ktir_cpu/parser.py` (the `KTIRParser`
//! class). Scope of this slice:
//!
//!   * module / `func.func` extraction via brace matching, grid attribute,
//!     function arguments — done, faithful to the Python regex logic;
//!   * the multi-line operation tokenizer (`tokenize_ops`) — ports the
//!     brace-balance + type-terminal + SSA-start flush heuristic, enough to
//!     group the real multi-line `ktdp.construct_*` ops in `examples/`;
//!   * structural op parse: result, op_type, operands, result type, plus the
//!     `arith.constant` value attribute and the infix index-arith shorthand.
//!
//! DEFERRED (later slices): nested regions (scf bodies), and dialect-specific
//! attribute parsing for the affine attrs (`coordinate_set`, `base_map`,
//! `memory_space`, `sizes:`/`strides:`). Until then a parsed `ktdp.construct_*`
//! op carries correct op_type / operands / result_type but not its affine
//! attributes, so it parses structurally but is not yet executable. The Python
//! original uses the `regex` crate's equivalent; this slice stays dependency-
//! free with manual scanning (regex is the production tool to adopt here).

use crate::ir::{Attr, IRFunction, IRModule, Operation, Scalar, Value};

/// Parse a full module's MLIR text into an [`IRModule`]. Mirrors `parse_module`.
pub fn parse_module(text: &str) -> Result<IRModule, String> {
    let text = strip_comments(text);
    let mut module = IRModule::default();
    for (name, args, grid, body) in find_functions(&text)? {
        let operations = tokenize_ops(&body)
            .iter()
            .filter_map(|op_text| parse_operation(op_text).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        module.add_function(IRFunction {
            name,
            arguments: args,
            operations,
            grid,
            return_type: None,
        });
    }
    Ok(module)
}

// --- phase 1: structure --------------------------------------------------

/// Strip `// ...` line comments. Mirrors `_preprocess_text`.
fn strip_comments(text: &str) -> String {
    text.lines()
        .map(|line| match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

type ParsedFn = (String, Vec<(String, String)>, (usize, usize, usize), String);

/// Locate each `func.func @name(args)`, extract its header (args + grid) and
/// brace-matched body. Mirrors the `func.func` loop in `parse_module`.
fn find_functions(text: &str) -> Result<Vec<ParsedFn>, String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut search = 0;
    while let Some(rel) = text[search..].find("func.func") {
        let start = search + rel;
        let after = start + "func.func".len();
        // @name
        let at = text[after..]
            .find('@')
            .ok_or("func.func missing '@name'")?
            + after;
        let name_start = at + 1;
        let name_end = name_start
            + text[name_start..]
                .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                .ok_or("unterminated function name")?;
        let name = text[name_start..name_end].to_string();

        // (args)
        let lparen = text[name_end..]
            .find('(')
            .ok_or("function missing arg list")?
            + name_end;
        let rparen = matching(bytes, lparen, b'(', b')')
            .ok_or("unbalanced function arg parens")?;
        let args = parse_args(&text[lparen + 1..rparen]);

        // After `)` comes `-> rettype attributes { grid = ... } { body }`.
        // The body is the LAST top-level brace block before the enclosing
        // `module {` close; intermediate blocks (the attributes block) are
        // skipped. Mirrors `_extract_brace_body`.
        let (body_open, body_close) = last_top_level_block(bytes, rparen + 1)
            .ok_or("function missing body")?;
        // The grid attribute lives in the header span up to the body block —
        // which still contains the skipped `attributes { grid = ... }`.
        let grid = parse_grid(&text[rparen..body_open]);
        let body = text[body_open + 1..body_close].to_string();

        out.push((name, args, grid, body));
        search = body_close + 1;
    }
    Ok(out)
}

/// Scan from `start`, skipping over each top-level `{...}` block, and return
/// the `(open, close)` indices of the LAST one before an unmatched `}` (the
/// enclosing scope's close) or end of input. Mirrors `_extract_brace_body`.
fn last_top_level_block(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut pos = start;
    let mut last = None;
    while pos < bytes.len() {
        match bytes[pos] {
            b'{' => {
                let close = matching(bytes, pos, b'{', b'}')?;
                last = Some((pos, close));
                pos = close + 1;
            }
            b'}' => break, // closing brace of an outer scope (e.g. module {})
            _ => pos += 1,
        }
    }
    last
}

/// Index of the byte matching the opener at `open_idx`, honoring nesting.
fn matching(bytes: &[u8], open_idx: usize, open: u8, close: u8) -> Option<usize> {
    debug_assert_eq!(bytes[open_idx], open);
    let mut depth = 0;
    for (i, &b) in bytes.iter().enumerate().skip(open_idx) {
        if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// `%name: type, ...` -> (name, type) pairs. Mirrors `_parse_function_args`.
fn parse_args(args_src: &str) -> Vec<(String, String)> {
    args_src
        .split(',')
        .filter_map(|seg| {
            let seg = seg.trim();
            let colon = seg.find(':')?;
            let name = seg[..colon].trim();
            if !name.starts_with('%') {
                return None;
            }
            Some((name.to_string(), seg[colon + 1..].trim().to_string()))
        })
        .collect()
}

/// `grid = [X]` / `[X, Y]` / `[X, Y, Z]`, missing dims default to 1.
/// Mirrors `_parse_grid_attribute`.
fn parse_grid(header: &str) -> (usize, usize, usize) {
    let Some(g) = header.find("grid") else {
        return (1, 1, 1);
    };
    let tail = &header[g..];
    let (Some(lb), Some(rb)) = (tail.find('['), tail.find(']')) else {
        return (1, 1, 1);
    };
    let nums: Vec<usize> = tail[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    (
        nums.first().copied().unwrap_or(1),
        nums.get(1).copied().unwrap_or(1),
        nums.get(2).copied().unwrap_or(1),
    )
}

// --- phase 2: tokenize ops ----------------------------------------------

/// Group body text into complete operation strings. Ports the brace-balance +
/// type-terminal + SSA-start flush heuristic from `_tokenize_operations`.
/// Region extraction (scf bodies) is deferred; this slice targets straight-line
/// bodies.
fn tokenize_ops(body: &str) -> Vec<String> {
    let mut results = Vec::new();
    let mut current: Vec<String> = Vec::new();

    let flush = |current: &mut Vec<String>, results: &mut Vec<String>| {
        if !current.is_empty() {
            results.push(current.join(" "));
            current.clear();
        }
    };

    for line in body.lines() {
        let stripped = line.trim();

        // Blank line flushes when braces are balanced.
        if stripped.is_empty() {
            if brace_balance(&current.join(" ")) == 0 {
                flush(&mut current, &mut results);
            }
            continue;
        }

        let accumulated = current.join(" ");
        if !current.is_empty()
            && brace_balance(&accumulated) == 0
            && !stripped.starts_with("->")
        {
            let prev_done = is_op_complete(&accumulated) || starts_ssa_assign(stripped);
            let next_cannot_start = stripped == "{";
            if prev_done && !next_cannot_start {
                flush(&mut current, &mut results);
            }
        }
        current.push(stripped.to_string());
    }
    flush(&mut current, &mut results);
    results
}

fn brace_balance(text: &str) -> i32 {
    text.bytes()
        .map(|b| match b {
            b'{' => 1,
            b'}' => -1,
            _ => 0,
        })
        .sum()
}

/// Does `text` start with `%name =` (or `%a, %b =`)? Mirrors the `starts_ssa`
/// regex in the tokenizer.
fn starts_ssa_assign(text: &str) -> bool {
    let Some(eq) = text.find('=') else {
        return false;
    };
    let lhs = &text[..eq];
    !lhs.is_empty()
        && lhs.trim_end().ends_with(|c: char| c.is_alphanumeric() || c == '_')
        && lhs.split(',').all(|p| p.trim().starts_with('%'))
}

/// Structural-completeness check. Mirrors `_is_op_complete`: a terminal type
/// annotation (`: T` / `-> T`), or a void terminator (`return`, `*.yield`).
fn is_op_complete(text: &str) -> bool {
    let text = text.trim_end();
    if text.is_empty() {
        return false;
    }
    // void terminators as op names (line start, or after `= `)
    let op_head = text.rsplit("= ").next().unwrap_or(text).trim_start();
    if op_head.starts_with("return") || op_head.split_whitespace().next().is_some_and(|t| t.ends_with(".yield")) {
        return true;
    }
    ends_with_type_terminal(text)
}

/// True when `text` ends with a `: T` / `-> T` type annotation, where T ends in
/// `>` (tensor/memref/!ktdp...) or `index` or `iNN`/`fNN`/`uNN`. Mirrors
/// `_TYPE_TERMINAL_RE`.
fn ends_with_type_terminal(text: &str) -> bool {
    if !text.contains(':') && !text.contains("->") {
        return false;
    }
    let last = text.trim_end();
    if last.ends_with('>') || last.ends_with("index") {
        return true;
    }
    let tok = last.rsplit(|c: char| c.is_whitespace() || c == ':' || c == '>').next().unwrap_or("");
    let mut chars = tok.chars();
    matches!(chars.next(), Some('i' | 'u' | 'f'))
        && !tok[1..].is_empty()
        && tok[1..].chars().all(|c| c.is_ascii_digit())
}

// --- phase 3: parse one op ----------------------------------------------

/// Parse a complete operation string. Mirrors `_parse_operation_text` +
/// `_parse_general_operation`, plus the constant and infix special cases.
fn parse_operation(text: &str) -> Result<Option<Operation>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    if let Some(op) = parse_index_binary(text) {
        return Ok(Some(op));
    }

    // optional `%result = `
    let (result, rest) = match split_assignment(text) {
        Some((r, rest)) => (Some(r.to_string()), rest),
        None => (None, text),
    };
    let rest = rest.trim();

    let op_type = rest
        .split_whitespace()
        .next()
        .ok_or("operation missing op_type")?
        .to_string();
    let after_op = rest[op_type.len()..].trim();

    let result_type = extract_result_type(after_op);
    let operands = extract_operands(after_op, result.as_deref());

    let mut attributes = std::collections::HashMap::new();
    if op_type == "arith.constant" {
        attributes.insert("value".to_string(), parse_constant_value(after_op)?);
    }

    Ok(Some(Operation {
        result,
        op_type,
        operands,
        attributes,
        result_type,
        regions: Vec::new(),
    }))
}

/// Infix index arithmetic: `%r = %a [*+-] %b : type`. Mirrors `_parse_index_binary`.
fn parse_index_binary(text: &str) -> Option<Operation> {
    let (lhs, rhs) = split_assignment(text)?;
    let rhs = rhs.trim();
    if !rhs.starts_with('%') {
        return None;
    }
    let before_colon = rhs.split(':').next().unwrap_or(rhs).trim();
    for (sym, op_name) in [('*', "arith.muli"), ('+', "arith.addi"), ('-', "arith.subi")] {
        if let Some(pos) = before_colon.find(sym) {
            let a = before_colon[..pos].trim();
            let b = before_colon[pos + 1..].trim();
            if a.starts_with('%') && b.starts_with('%') && !a[1..].contains(char::is_whitespace) {
                let rty = rhs.split(':').nth(1).map(|s| s.trim().to_string());
                return Some(Operation {
                    result: Some(lhs.to_string()),
                    op_type: op_name.to_string(),
                    operands: vec![a.to_string(), b.to_string()],
                    attributes: std::collections::HashMap::new(),
                    result_type: rty,
                    regions: Vec::new(),
                });
            }
        }
    }
    None
}

/// Split `%result = rest` -> `(%result, rest)`, only when the LHS is a single
/// SSA name (so we don't trip on `==` or attribute `=`).
fn split_assignment(text: &str) -> Option<(&str, &str)> {
    let eq = text.find('=')?;
    let lhs = text[..eq].trim();
    let rhs = &text[eq + 1..];
    if lhs.starts_with('%')
        && !lhs.contains(char::is_whitespace)
        && !rhs.starts_with('=')
    {
        Some((lhs, rhs))
    } else {
        None
    }
}

/// All `%name` operands before the type, with `{...}` blocks removed and the
/// result name excluded. Mirrors `_extract_operands`.
fn extract_operands(text: &str, result: Option<&str>) -> Vec<String> {
    let cleaned = remove_brace_blocks(text);
    let mut out = Vec::new();
    let bytes = cleaned.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let mut j = i + 1;
            while j < bytes.len()
                && (bytes[j].is_ascii_alphanumeric() || matches!(bytes[j], b'_' | b'$' | b'.'))
            {
                j += 1;
            }
            let name = &cleaned[i..j];
            if Some(name) != result && name.len() > 1 && !out.iter().any(|o| o == name) {
                out.push(name.to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Result type from `-> T` (preferred) or the last `: T` outside braces.
/// Mirrors `_extract_result_type`.
fn extract_result_type(text: &str) -> Option<String> {
    if let Some(arrow) = text.rfind("->") {
        let t = text[arrow + 2..].trim();
        if !t.is_empty() {
            return Some(t.to_string());
        }
    }
    let cleaned = remove_brace_blocks(text);
    let colon = cleaned.rfind(':')?;
    let t = cleaned[colon + 1..].trim();
    if t.is_empty() || t.starts_with('%') {
        None
    } else {
        Some(t.to_string())
    }
}

/// Remove every `{...}` block (one level of nesting collapsed repeatedly).
fn remove_brace_blocks(text: &str) -> String {
    let mut s = text.to_string();
    while let Some(open) = s.find('{') {
        if let Some(close) = matching(s.as_bytes(), open, b'{', b'}') {
            s.replace_range(open..=close, " ");
        } else {
            s.truncate(open); // unbalanced: drop the tail
            break;
        }
    }
    s
}

/// Parse the literal of `arith.constant <lit> : <type>` into a value `Attr`.
fn parse_constant_value(after_op: &str) -> Result<Attr, String> {
    let lit = after_op
        .split(':')
        .next()
        .and_then(|s| s.split_whitespace().next())
        .ok_or("arith.constant: missing literal")?;
    match lit {
        "true" => Ok(Attr::Bool(true)),
        "false" => Ok(Attr::Bool(false)),
        _ if lit.contains('.') || lit.contains('e') || lit.contains('E') => lit
            .parse::<f64>()
            .map(Attr::Float)
            .map_err(|_| format!("arith.constant: bad float literal {lit:?}")),
        _ => lit
            .parse::<i64>()
            .map(Attr::Int)
            .map_err(|_| format!("arith.constant: bad int literal {lit:?}")),
    }
}

/// Convenience: a parsed `arith.constant` value attr -> a [`Value`] for tests /
/// the interpreter's constant-folding entry. (Real placement is the handler.)
pub fn constant_attr_to_value(attr: &Attr) -> Option<Value> {
    match attr {
        Attr::Float(f) => Some(Value::Scalar(Scalar::F32(*f as f32))),
        Attr::Int(i) => Some(Value::Scalar(Scalar::I64(*i))),
        Attr::Bool(b) => Some(Value::Scalar(Scalar::Bool(*b))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};
    use crate::ir::Value;

    const VECTOR_ADD: &str = include_str!(
        "../../examples/triton-ktir/vector_add_ktir.mlir"
    );

    #[test]
    fn parses_real_vector_add_structurally() {
        let module = parse_module(VECTOR_ADD).unwrap();
        let f = module.get_function("add_kernel").unwrap();
        assert_eq!(f.grid, (32, 1, 1));
        assert_eq!(f.arg_names(), vec!["x_ptr", "y_ptr", "output_ptr", "BLOCK_SIZE"]);

        // The multi-line construct ops must each tokenize to exactly one op.
        let types: Vec<&str> = f.operations.iter().map(|o| o.op_type.as_str()).collect();
        assert_eq!(types.iter().filter(|t| **t == "ktdp.construct_memory_view").count(), 3);
        assert_eq!(types.iter().filter(|t| **t == "ktdp.construct_access_tile").count(), 3);
        assert_eq!(types.iter().filter(|t| **t == "ktdp.load").count(), 2);
        assert_eq!(types.iter().filter(|t| **t == "ktdp.store").count(), 1);
        assert!(types.contains(&"arith.addf"));
        assert_eq!(types.last(), Some(&"return"));

        // Operand/type wiring on a representative multi-line op.
        let view = f.operations.iter().find(|o| o.result.as_deref() == Some("%x_view")).unwrap();
        assert_eq!(view.operands, vec!["%x_ptr"]);
        assert_eq!(view.result_type.as_deref(), Some("memref<4096xf16>"));

        let at = f.operations.iter().find(|o| o.result.as_deref() == Some("%x_tile")).unwrap();
        assert_eq!(at.operands, vec!["%x_view", "%offset"]);
        assert_eq!(at.result_type.as_deref(), Some("!ktdp.access_tile<128xindex>"));
    }

    #[test]
    fn infix_index_arith_lowers_to_arith_op() {
        let module = parse_module(VECTOR_ADD).unwrap();
        let f = module.get_function("add_kernel").unwrap();
        let off = f.operations.iter().find(|o| o.result.as_deref() == Some("%offset")).unwrap();
        // `arith.muli %core_id, %BLOCK_SIZE : index`
        assert_eq!(off.op_type, "arith.muli");
        assert_eq!(off.operands, vec!["%core_id", "%BLOCK_SIZE"]);
    }

    #[test]
    fn parse_then_execute_arith_function() {
        let src = r#"
            module {
              func.func @f() attributes {grid = [1]} {
                %a = arith.constant 2.0 : f32
                %b = arith.constant 3.0 : f32
                %c = arith.addf %a, %b : f32
                %d = arith.mulf %c, %a : f32
                return
              }
            }
        "#;
        let module = parse_module(src).unwrap();
        let f = module.get_function("f").unwrap();

        let dispatch = Dispatch::new();
        let grid = GridExecutor::new(f.grid);
        let env = ExecutionEnv { dispatch: &dispatch, grid: &grid };
        let mut ctx = single_core_context();
        execute_ops(&f.operations, &mut ctx, &env).unwrap();
        match ctx.get_value("%d").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 10.0), // (2+3)*2
            other => panic!("expected F32(10.0), got {other:?}"),
        }
    }
}
