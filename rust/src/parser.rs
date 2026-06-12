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
//!     `arith.constant` value attribute and the infix index-arith shorthand;
//!   * dialect-specific attribute parsing for the `ktdp.construct_*` ops —
//!     ports `parse_construct_memory_view` / `parse_construct_access_tile`
//!     from `ktir_cpu/dialects/ktdp_ops.py`, lifting the real affine attrs
//!     (`coordinate_set`, `base_map`, `access_tile_set`, `access_tile_order`),
//!     the `sizes:`/`strides:` segments, the `dtype`, and the
//!     `#ktdp.spyre_memory_space<HBM|LX, core=N>` memory space into the typed
//!     [`Attr`] enum. The affine text is parsed by the recursive-descent parser
//!     in [`crate::parser_ast`].
//!
//!   * nested regions (scf bodies, linalg.reduce/generic combiners) — extracted
//!     by `tokenize_ops` (`_line_opens_region`) and recursively parsed into
//!     `op.regions`;
//!   * general `{ key = value }` attribute blocks AND bare `key = value` attrs
//!     (`permutation = [..]`, `dimensions = [..]`), plus `shape`/`dtype` derived
//!     from a `tensor<...>` result type.
//!
//! DEFERRED: dynamic/SSA memref sizes (lazy `?`-dim resolution). The Python
//! original uses the `regex` crate's equivalent; this stays dependency-free with
//! manual scanning (regex is the production tool to adopt here).

use crate::ir::{Attr, IRFunction, IRModule, Operation, Scalar, Value};
use crate::parser_ast::{is_full_set, is_identity_map, parse_affine_map, parse_affine_set};

/// Parse a full module's MLIR text into an [`IRModule`]. Mirrors `parse_module`.
pub fn parse_module(text: &str) -> Result<IRModule, String> {
    let text = strip_comments(text);
    let text = expand_attr_aliases(&text);
    let mut module = IRModule::default();
    for (name, args, grid, body) in find_functions(&text)? {
        let operations = parse_operations(&body)?;
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

/// Resolve module-level named attribute aliases. MLIR lets a module declare
/// `#name = <value>` at top scope and then reference `#name` inside op
/// attributes (e.g. `coordinate_set = #A_HBM_coord_set`). Port of the Python
/// `KTIRParser` "module-level pre-scan" (`parser.py`): collect every
/// `#name = keyword<...>` declaration and textually substitute each `#name`
/// reference with its expansion. The declaration lines themselves are blanked
/// (line structure preserved) so they are not re-parsed as ops.
///
/// Only `keyword<...>` values (`affine_set<...>` / `affine_map<...>` and the
/// like) are expanded — these are the alias forms the dialect ops use; depth is
/// tracked across `<`/`>` while skipping the `>=` / `->` operators that appear
/// inside affine bodies (same walk as `named_attr_value`).
fn expand_attr_aliases(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut aliases: Vec<(String, String)> = Vec::new();
    let mut blanked = text.to_string();

    // Find `#name = keyword<...>` declarations. A declaration begins at a `#`
    // whose token is followed (after whitespace) by `=` then a `keyword<`.
    let mut i = 0;
    while let Some(rel) = text[i..].find('#') {
        let hash = i + rel;
        // `#name` token: `#` then word chars / dots.
        let name_end = hash
            + 1
            + text[hash + 1..]
                .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
                .unwrap_or(text.len() - hash - 1);
        let name = &text[hash..name_end];
        let after = text[name_end..].trim_start();
        if name.len() > 1 && after.starts_with('=') {
            // Candidate declaration. Extract the balanced `keyword<...>` value.
            let val_region = &text[name_end..];
            if let Some(value) = keyword_value(val_region) {
                // Compute the absolute end of the declaration (after the value).
                let val_off = val_region.find(&value).unwrap();
                let decl_end = name_end + val_off + value.len();
                aliases.push((name.to_string(), value.clone()));
                // Blank the declaration span in `blanked` (preserve newlines).
                blank_span(&mut blanked, hash, decl_end);
                i = decl_end;
                continue;
            }
        }
        i = name_end;
    }
    let _ = bytes;

    if aliases.is_empty() {
        return blanked;
    }
    // Substitute references. Longest names first so a prefix alias never
    // shadows a longer one. Only replace whole `#name` tokens (the char after
    // the name must not continue the identifier).
    aliases.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
    for (name, value) in &aliases {
        blanked = replace_alias_token(&blanked, name, value);
    }
    blanked
}

/// Extract a leading `= keyword<...>` value from `text` (text starts at the
/// alias name's end). Returns the `keyword<...>` substring, balancing `<`/`>`
/// while skipping `>=` / `->`.
fn keyword_value(text: &str) -> Option<String> {
    let eq = text.find('=')?;
    let rest = text[eq + 1..].trim_start();
    let kw_lt = rest.find('<')?;
    if !rest[..kw_lt]
        .trim()
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return None;
    }
    let bytes = rest.as_bytes();
    let mut i = kw_lt;
    let mut depth = 0i32;
    while i < bytes.len() {
        if bytes[i] == b'>' && i + 1 < bytes.len() && bytes[i + 1] == b'=' {
            i += 2;
            continue;
        }
        if bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            i += 2;
            continue;
        }
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    // Include the leading keyword by returning from rest start.
                    return Some(rest[..=i].to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Overwrite `[start, end)` of `s` with spaces, preserving newlines so line
/// numbers (and the `strip_comments` line-count invariant) stay intact.
fn blank_span(s: &mut String, start: usize, end: usize) {
    let mut out = String::with_capacity(s.len());
    out.push_str(&s[..start]);
    for ch in s[start..end].chars() {
        out.push(if ch == '\n' { '\n' } else { ' ' });
    }
    out.push_str(&s[end..]);
    *s = out;
}

/// Replace every whole-token occurrence of `#name` in `text` with `value`.
/// A match is a whole token when the following char does not continue the
/// identifier (`[\w.]`).
fn replace_alias_token(text: &str, name: &str, value: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(rel) = text[i..].find(name) {
        let pos = i + rel;
        let after = pos + name.len();
        let boundary = text[after..]
            .chars()
            .next()
            .map(|c| !(c.is_alphanumeric() || c == '_' || c == '.'))
            .unwrap_or(true);
        out.push_str(&text[i..pos]);
        if boundary {
            out.push_str(value);
        } else {
            out.push_str(name);
        }
        i = after;
    }
    out.push_str(&text[i..]);
    out
}

// --- phase 1: structure --------------------------------------------------

/// Strip `// ...` line comments, preserving line structure (newline count).
/// Mirrors `_preprocess_text`. Uses `split('\n')` rather than `lines()` so a
/// trailing newline is kept — the line count is invariant under stripping.
pub fn strip_comments(text: &str) -> String {
    text.split('\n')
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
        let at = text[after..].find('@').ok_or("func.func missing '@name'")? + after;
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
        let rparen = matching(bytes, lparen, b'(', b')').ok_or("unbalanced function arg parens")?;
        let args = parse_args(&text[lparen + 1..rparen]);

        // After `)` comes `-> rettype attributes { grid = ... } { body }`.
        // The body is the LAST top-level brace block before the enclosing
        // `module {` close; intermediate blocks (the attributes block) are
        // skipped. Mirrors `_extract_brace_body`.
        let (body_open, body_close) =
            last_top_level_block(bytes, rparen + 1).ok_or("function missing body")?;
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

/// Parse a function/region body into operations, recursively parsing each
/// op's region bodies into `op.regions`. Mirrors `_parse_operations`.
fn parse_operations(body: &str) -> Result<Vec<Operation>, String> {
    let mut ops = Vec::new();
    for (op_text, regions) in tokenize_ops(body) {
        let Some(mut op) = parse_operation(&op_text)? else {
            continue;
        };
        for region_body in &regions {
            op.regions.push(parse_operations(region_body)?);
        }
        ops.push(op);
    }
    Ok(ops)
}

// --- phase 2: tokenize ops ----------------------------------------------

/// A tokenized op: its text plus any region bodies (the `{ ... }` blocks that
/// contain operations, e.g. `scf.for` / `linalg.reduce` combiner). Mirrors the
/// `(op_text, [region_bodies])` pairs from `_tokenize_operations`.
type TokenizedOp = (String, Vec<String>);

/// Group body text into complete operations, extracting region bodies. Ports
/// `_tokenize_operations` including `_line_opens_region` /
/// `_extract_region_from_lines`: a `{` that opens a block containing `%` SSA
/// references is a region (recursively parsed); other `{ }` blocks are inline
/// attribute blocks kept in the op text.
fn tokenize_ops(body: &str) -> Vec<TokenizedOp> {
    let lines: Vec<&str> = body.lines().collect();
    let mut results: Vec<TokenizedOp> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_regions: Vec<String> = Vec::new();

    let flush =
        |current: &mut Vec<String>, regions: &mut Vec<String>, results: &mut Vec<TokenizedOp>| {
            if !current.is_empty() {
                results.push((current.join(" "), std::mem::take(regions)));
                current.clear();
            }
        };

    let mut i = 0;
    while i < lines.len() {
        let stripped = lines[i].trim();

        // Blank line flushes when braces are balanced.
        if stripped.is_empty() {
            if brace_balance(&current.join(" ")) == 0 {
                flush(&mut current, &mut current_regions, &mut results);
            }
            i += 1;
            continue;
        }

        let accumulated = current.join(" ");
        if !current.is_empty() && brace_balance(&accumulated) == 0 && !stripped.starts_with("->") {
            let prev_done = is_op_complete(&accumulated) || starts_ssa_assign(stripped);
            let next_cannot_start = stripped == "{";
            if prev_done && !next_cannot_start {
                flush(&mut current, &mut current_regions, &mut results);
            }
        }
        current.push(stripped.to_string());

        // Does this line open a region body? (ends with `{`, block has `%` refs)
        if line_opens_region(stripped, &lines, i) {
            // Drop the trailing `{` from the op text.
            let last = current.last_mut().unwrap();
            *last = last.trim_end_matches('{').trim_end().to_string();
            if last.is_empty() {
                current.pop();
            }
            if let Some((region_body, end_line, trailing)) = extract_region_from_lines(&lines, i) {
                current_regions.push(region_body);
                if !trailing.is_empty() {
                    current.push(trailing);
                }
                i = end_line + 1;
                continue;
            }
        }
        i += 1;
    }
    flush(&mut current, &mut current_regions, &mut results);
    results
}

/// A line opens a region iff it ends with `{` and the brace-balanced block it
/// opens contains a `%` SSA reference (regions hold ops; attribute blocks don't).
/// Mirrors `_line_opens_region`.
fn line_opens_region(stripped: &str, lines: &[&str], idx: usize) -> bool {
    if !stripped.ends_with('{') {
        return false;
    }
    let mut depth = 1i32;
    for line in &lines[idx + 1..] {
        depth += brace_balance(line);
        if line.contains('%') {
            return true;
        }
        if depth <= 0 {
            break;
        }
    }
    false
}

/// Extract a region body from the line after the one ending in `{` to its
/// matching `}`. Returns `(region_body, closing_line_index, trailing_text)`
/// where trailing is any text after `}` on its line (belongs to the outer op).
/// Mirrors `_extract_region_from_lines`.
fn extract_region_from_lines(lines: &[&str], open_line: usize) -> Option<(String, usize, String)> {
    let mut depth = 1i32;
    let mut body_lines: Vec<String> = Vec::new();
    let mut i = open_line + 1;
    while i < lines.len() {
        let line = lines[i];
        let stripped = line.trim();
        for (ci, ch) in stripped.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        let before = stripped[..ci].trim();
                        if !before.is_empty() {
                            body_lines.push(before.to_string());
                        }
                        let after = stripped[ci + 1..].trim().to_string();
                        return Some((body_lines.join("\n"), i, after));
                    }
                }
                _ => {}
            }
        }
        body_lines.push(line.to_string());
        i += 1;
    }
    None
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
        && lhs
            .trim_end()
            .ends_with(|c: char| c.is_alphanumeric() || c == '_')
        && lhs.split(',').all(|p| p.trim().starts_with('%'))
}

/// Structural-completeness check. Mirrors `_is_op_complete`: a terminal type
/// annotation (`: T` / `-> T`), or a void terminator (`return`, `*.yield`).
pub fn is_op_complete(text: &str) -> bool {
    let text = text.trim_end();
    if text.is_empty() {
        return false;
    }
    // void terminators as op names (line start, or after `= `)
    let op_head = text.rsplit("= ").next().unwrap_or(text).trim_start();
    if op_head.starts_with("return")
        || op_head
            .split_whitespace()
            .next()
            .is_some_and(|t| t.ends_with(".yield"))
    {
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
    let tok = last
        .rsplit(|c: char| c.is_whitespace() || c == ':' || c == '>')
        .next()
        .unwrap_or("");
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

    // optional `%result = ` or multi-result `%a, %b = ` (e.g. the 2-D form of
    // `ktdp.get_compute_tile_id`, or an scf.for with several iter_args).
    let (result_names, rest) = match split_assignment_multi(text) {
        Some((names, rest)) => (names, rest),
        None => (Vec::new(), text),
    };
    let result = result_names.first().cloned();
    let rest = rest.trim();

    // The op name is the leading `dialect.op` identifier; it ends at the first
    // non-identifier char (whitespace, or `(` in the no-operand form like
    // `tensor.empty()`). Mirrors the `[a-z_][a-z0-9_.]*` capture in the Python
    // `_parse_general_operation` regex.
    let token = rest
        .split_whitespace()
        .next()
        .ok_or("operation missing op_type")?;
    let op_len = token
        .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .unwrap_or(token.len());
    let op_type = token[..op_len].to_string();
    // Lines that don't begin with a `dialect.op` identifier (e.g. block labels
    // `^bb0(%a: f32):`) are not operations — skip them, as the Python
    // `_parse_general_operation` regex does by failing to match and returning None.
    if op_type.is_empty() {
        return Ok(None);
    }
    let after_op = rest[op_type.len()..].trim();

    let result_type = extract_result_type(after_op);

    // scf.for needs structured operands `[lb, ub, step, ...inits]` and the
    // `iter_var` / `iter_args` attributes (the generic %-scan would mis-order
    // them and never bind the induction variable). Mirrors Python parse_scf_for.
    if op_type == "scf.for" {
        let (operands, mut attributes) = parse_scf_for_op(after_op)
            .ok_or("scf.for: could not parse `%iv = %lb to %ub step %step`")?;
        set_multi_result(&mut attributes, &result_names);
        return Ok(Some(Operation {
            result,
            op_type,
            operands,
            attributes,
            result_type,
            regions: Vec::new(),
        }));
    }

    let operands = extract_operands(after_op, result.as_deref());

    let mut attributes = std::collections::HashMap::new();
    if op_type == "arith.constant" {
        attributes.insert("value".to_string(), parse_constant_value(after_op)?);
    } else if op_type == "ktdp.construct_memory_view" {
        // The construct ops carry their real attributes across the whole op
        // text (including the `{ ... }` block and the trailing memref type),
        // so we parse from `text`, not just `after_op`.
        parse_construct_memory_view_attrs(text, result_type.as_deref(), &mut attributes)?;
    } else if op_type == "ktdp.construct_access_tile" {
        parse_construct_access_tile_attrs(
            text,
            result_type.as_deref(),
            &operands,
            &mut attributes,
        )?;
    } else {
        // General attributes: the `{ key = value, ... }` block AND bare
        // `key = value` attributes (MLIR named ops carry `permutation = [..]`,
        // `dimensions = [..]` bare). Mirrors `_extract_attributes` +
        // `_parse_bare_attr`. Bare attrs fill in keys the block doesn't have.
        attributes = parse_attr_block(after_op);
        for (k, v) in parse_bare_attrs(after_op) {
            attributes.entry(k).or_insert(v);
        }
        // `linalg.reduce { arith.maximumf }` shorthand: the `{ }` holds a bare
        // combiner op name (no `=`, no region). Lift it to `reduce_fn` so the
        // handler uses the right combiner instead of defaulting to addf.
        if op_type == "linalg.reduce"
            && !attributes.contains_key("reduce_fn")
            && let Some(combiner) = reduce_shorthand_combiner(after_op)
        {
            attributes.insert("reduce_fn".to_string(), Attr::Str(combiner));
        }
        // Derive `shape`/`dtype` from a `tensor<...>` result type when the op
        // doesn't carry them explicitly (tensor.splat/empty/generate read these).
        // Mirrors the `_result_shape`/`_result_dtype` population in
        // `_parse_general_operation`.
        if let Some(rt) = &result_type
            && let Some((shape, dt)) = parse_tensor_type(rt).or_else(|| parse_memref_type(rt))
        {
            attributes
                .entry("shape".to_string())
                .or_insert(Attr::IntList(shape));
            attributes
                .entry("dtype".to_string())
                .or_insert(Attr::Str(dt));
        }
    }

    set_multi_result(&mut attributes, &result_names);

    Ok(Some(Operation {
        result,
        op_type,
        operands,
        attributes,
        result_type,
        regions: Vec::new(),
    }))
}

/// For a multi-result op (`%a, %b = ...`), record every result name and the
/// count so the interpreter can bind each, and the handler (e.g.
/// `ktdp.get_compute_tile_id`) can return the right number of values.
fn set_multi_result(attrs: &mut std::collections::HashMap<String, Attr>, names: &[String]) {
    if names.len() > 1 {
        attrs.insert("result_names".to_string(), Attr::StrList(names.to_vec()));
        attrs.insert("num_results".to_string(), Attr::Int(names.len() as i64));
    }
}

/// First `%name` in `s`, with its end offset.
fn first_ssa(s: &str) -> Option<(&str, usize)> {
    let start = s.find('%')?;
    let bytes = s.as_bytes();
    let mut end = start + 1;
    while end < bytes.len()
        && (bytes[end].is_ascii_alphanumeric() || matches!(bytes[end], b'_' | b'$' | b'.'))
    {
        end += 1;
    }
    Some((&s[start..end], end))
}

/// Parse `%iv = %lb to %ub step %step iter_args(%a = %i, ...)` (region body
/// already stripped) into `([lb, ub, step, ...inits], {iter_var, iter_args})`.
/// Port of Python `parse_scf_for`.
fn parse_scf_for_op(rest: &str) -> Option<(Vec<String>, std::collections::HashMap<String, Attr>)> {
    let (iter_var, iv_end) = first_ssa(rest)?;
    let after_iv = &rest[iv_end..];
    let after_eq = &after_iv[after_iv.find('=')? + 1..];
    let (lb, _) = first_ssa(after_eq)?;
    let after_to = &after_eq[after_eq.find(" to ")? + 4..];
    let (ub, _) = first_ssa(after_to)?;
    let after_step = &after_to[after_to.find(" step ")? + 6..];
    let (step, _) = first_ssa(after_step)?;

    let mut operands = vec![lb.to_string(), ub.to_string(), step.to_string()];
    let mut iter_args: Vec<String> = Vec::new();
    if let Some(open) = rest.find("iter_args(") {
        let inner = &rest[open + "iter_args(".len()..];
        if let Some(close) = inner.find(')') {
            // pairs `%name = %init`, comma-separated.
            for pair in inner[..close].split(',') {
                if let Some((name, _)) = first_ssa(pair) {
                    let after = &pair[pair.find('=').unwrap_or(0) + 1..];
                    if let Some((init, _)) = first_ssa(after) {
                        iter_args.push(name.to_string());
                        operands.push(init.to_string());
                    }
                }
            }
        }
    }
    let mut attrs = std::collections::HashMap::new();
    attrs.insert("iter_var".to_string(), Attr::Str(iter_var.to_string()));
    if !iter_args.is_empty() {
        attrs.insert("iter_args".to_string(), Attr::StrList(iter_args));
    }
    Some((operands, attrs))
}

/// Split `%a, %b, ... = rest` (or single `%a = rest`) into the result names and
/// the RHS, only when the LHS is `%`-names separated by commas (so we don't trip
/// on `==` or attribute `=`). Generalizes [`split_assignment`] to multi-result.
fn split_assignment_multi(text: &str) -> Option<(Vec<String>, &str)> {
    let eq = text.find('=')?;
    let lhs = text[..eq].trim();
    let rhs = &text[eq + 1..];
    if rhs.starts_with('=') || !lhs.starts_with('%') {
        return None;
    }
    let mut names = Vec::new();
    for part in lhs.split(',') {
        let p = part.trim();
        // Each part must be exactly one SSA name (no spaces / extra tokens).
        if !p.starts_with('%') || p.contains(char::is_whitespace) {
            return None;
        }
        names.push(p.to_string());
    }
    Some((names, rhs))
}

// --- phase 4: ktdp construct-op attribute parsing -----------------------
//
// Ports `parse_construct_memory_view` / `parse_construct_access_tile` from
// `ktir_cpu/dialects/ktdp_ops.py`. The structural pass above already fills in
// result / operands / result_type; here we lift the affine + shape attributes
// into the typed `Attr` enum so the construct ops are executable.

/// Populate `ktdp.construct_memory_view` attributes from its op text. Mirrors
/// the body of `parse_construct_memory_view`:
///   * `sizes: [...]`   -> `Attr::IntList` (`shape`)
///   * `strides: [...]` -> `Attr::IntList` (`strides`, default `[1]`)
///   * `#ktdp.spyre_memory_space<S[, core=N]>` -> `memory_space` (`Attr::Str`)
///     plus an optional `lx_core_id` (`Attr::Int`)
///   * memref element type -> `dtype` (`Attr::Str`)
///   * `coordinate_set = affine_set<...>` -> `Attr::AffineSet`
///
/// Sizes/strides that are SSA names (dynamic dims) are not representable in the
/// integer `Attr::IntList`; they already appear as operands from the structural
/// pass and are resolved at execution time, so the size attribute is omitted in
/// that case (rather than guessing a literal). This matches the executor's
/// "lazily resolve SSA sizes" contract.
fn parse_construct_memory_view_attrs(
    text: &str,
    result_type: Option<&str>,
    attrs: &mut std::collections::HashMap<String, Attr>,
) -> Result<(), String> {
    // sizes: [...] — static (all literal ints) -> `shape` IntList. Dynamic
    // (any `%ssa` or `?`) -> `sizes_dyn` StrList of raw tokens, resolved at
    // execution time by the handler (the Python "lazily resolve SSA sizes"
    // contract).
    if let Some(list) = bracket_segment(text, "sizes") {
        if let Some(ints) = parse_int_list(&list) {
            attrs.insert("shape".to_string(), Attr::IntList(ints));
        } else {
            let tokens: Vec<String> = list
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            attrs.insert("sizes_dyn".to_string(), Attr::StrList(tokens));
        }
    }

    // strides: [...] — default [1] (matching the Python default).
    let strides = bracket_segment(text, "strides")
        .and_then(|l| parse_int_list(&l))
        .unwrap_or_else(|| vec![1]);
    attrs.insert("strides".to_string(), Attr::IntList(strides));

    // #ktdp.spyre_memory_space<S[, core = N]> — default HBM.
    let (memory_space, lx_core_id) = parse_memory_space(text);
    attrs.insert("memory_space".to_string(), Attr::Str(memory_space));
    if let Some(core) = lx_core_id {
        attrs.insert("lx_core_id".to_string(), Attr::Int(core));
    }

    // dtype from the memref<...> result type's trailing element type.
    let dtype = result_type
        .and_then(parse_memref_dtype)
        .ok_or("construct_memory_view: could not parse dtype from memref<> type")?;
    attrs.insert("dtype".to_string(), Attr::Str(dtype));

    // coordinate_set = affine_set<...>
    if let Some(raw) = named_attr_value(text, "coordinate_set") {
        let set = parse_affine_set(&raw)?;
        attrs.insert("coordinate_set".to_string(), Attr::AffineSet(set));
    }

    Ok(())
}

/// Populate `ktdp.construct_access_tile` attributes from its op text. Mirrors
/// `parse_construct_access_tile`:
///   * access-tile shape from `!ktdp.access_tile<NxMx...xindex>` (`Attr::IntList`)
///   * `base_map = affine_map<...>` -> `Attr::AffineMap` (synthesized identity
///     of rank `max(1, operands-1)` when absent)
///   * `access_tile_set = affine_set<...>` -> `coordinate_set` (`Attr::AffineSet`),
///     dropped when it is full over the tile box
///   * `access_tile_order = affine_map<...>` -> `coordinate_order`
///     (`Attr::AffineMap`), dropped when it is the identity
fn parse_construct_access_tile_attrs(
    text: &str,
    result_type: Option<&str>,
    operands: &[String],
    attrs: &mut std::collections::HashMap<String, Attr>,
) -> Result<(), String> {
    // Shape + `index` element-type validation from the access_tile<...> type.
    let inner = result_type
        .and_then(access_tile_inner)
        .ok_or("construct_access_tile: missing !ktdp.access_tile<> result type")?;
    let (shape, elem) = parse_access_tile_inner(&inner)?;
    if elem != "index" {
        return Err(format!(
            "AccessTileType element type must be 'index', got {elem:?}"
        ));
    }
    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    attrs.insert("shape".to_string(), Attr::IntList(shape_i64));

    // base_map — synthesize identity of rank max(1, operands-1) when absent.
    let base_map = match named_attr_value(text, "base_map") {
        Some(raw) => parse_affine_map(&raw)?,
        None => {
            let n = operands.len().saturating_sub(1).max(1);
            let dims: Vec<String> = (0..n).map(|i| format!("d{i}")).collect();
            let csv = dims.join(", ");
            parse_affine_map(&format!("affine_map<({csv}) -> ({csv})>"))?
        }
    };
    attrs.insert("base_map".to_string(), Attr::AffineMap(base_map));

    // access_tile_set -> coordinate_set; dropped when full over the tile box.
    if let Some(raw) = named_attr_value(text, "access_tile_set") {
        let set = parse_affine_set(&raw)?;
        if !is_full_set(&set, &shape) {
            attrs.insert("coordinate_set".to_string(), Attr::AffineSet(set));
        }
    }

    // access_tile_order -> coordinate_order; dropped when identity.
    if let Some(raw) = named_attr_value(text, "access_tile_order") {
        let map = parse_affine_map(&raw)?;
        if !is_identity_map(&map) {
            attrs.insert("coordinate_order".to_string(), Attr::AffineMap(map));
        }
    }

    Ok(())
}

/// Find a `keyword: [ ... ]` segment (e.g. `sizes: [4096]`) and return the
/// inner list text. Mirrors the `sizes\s*:\s*\[([^\]]+)\]` regex.
fn bracket_segment(text: &str, keyword: &str) -> Option<String> {
    let key_pos = text.find(keyword)?;
    let after = &text[key_pos + keyword.len()..];
    // Expect `:` then `[`.
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let rest = rest.strip_prefix('[')?;
    let close = rest.find(']')?;
    Some(rest[..close].to_string())
}

/// Parse a comma-separated list of integer literals. Returns `None` when any
/// element is not a literal int (i.e. an SSA name / dynamic dim).
fn parse_int_list(list: &str) -> Option<Vec<i64>> {
    let mut out = Vec::new();
    for tok in list.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        out.push(tok.parse::<i64>().ok()?);
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Parse `#ktdp.spyre_memory_space<S[, core = N]>` -> (memory_space, lx_core_id).
/// Defaults to `("HBM", None)` when absent. Mirrors the
/// `#ktdp\.spyre_memory_space<\s*(\w+)(?:\s*,\s*core\s*=\s*(\d+))?\s*>` regex.
fn parse_memory_space(text: &str) -> (String, Option<i64>) {
    let marker = "#ktdp.spyre_memory_space<";
    let Some(start) = text.find(marker) else {
        return ("HBM".to_string(), None);
    };
    let after = &text[start + marker.len()..];
    let Some(close) = after.find('>') else {
        return ("HBM".to_string(), None);
    };
    let body = after[..close].trim();
    // body is `S` or `S, core = N`.
    let mut parts = body.splitn(2, ',');
    let space = parts.next().unwrap_or("HBM").trim().to_string();
    let core = parts.next().and_then(|p| {
        // `core = N`
        let eq = p.find('=')?;
        p[eq + 1..].trim().parse::<i64>().ok()
    });
    (space, core)
}

/// Extract the element dtype (last `x`-segment) from a `memref<...>` type
/// string, e.g. `memref<4096xf16>` -> `f16`. Mirrors the memref-type split.
/// Parse a `tensor<DxDx...xELT[, encoding]>` type into `(static_shape, dtype)`.
///
/// Anchored at `tensor<` (trailing context after `>` is ignored, like Python's
/// `re.match`). Leading `Nx` / `?x` dimension tokens are consumed one at a time
/// — so the element type's own letters (notably `index`, which *ends in* `x`)
/// are never mistaken for a dim separator. Dynamic `?` dims are dropped from the
/// static shape; the dtype stops at the first `,`/`>`/whitespace (so an encoding
/// attribute like `tensor<4x4xf32, #enc>` yields `f32`). E.g.
/// `tensor<1x4xf16>` -> `([1, 4], "f16")`, `tensor<2xindex>` -> `([2], "index")`.
/// Parse a `tensor<...>` type into `(shape, dtype)`, or `None` if not a tensor
/// type. Port of Python `parser_utils.parse_tensor_type`: anchors at the start
/// (so trailing context after `>` is ignored, e.g. `tensor<4xf32> loc(...)`) and
/// takes the dtype as the leading element type, stopping at `,` (so an encoding
/// attribute like `tensor<4x4xf32, #enc>` yields `f32`). Public so the port
/// tests can exercise it directly, as the Python suite does.
pub fn parse_tensor_type(ty: &str) -> Option<(Vec<i64>, String)> {
    // Drop whitespace so `tensor< 2 x f32 >` tokenizes like `tensor<2xf32>`.
    let compact: String = ty.chars().filter(|c| !c.is_whitespace()).collect();
    let mut s = compact.strip_prefix("tensor<")?;
    let mut shape = Vec::new();
    loop {
        // A dimension token is `\d+` or `?`, immediately followed by `x`.
        if let Some(after) = s.strip_prefix('?') {
            if let Some(rest) = after.strip_prefix('x') {
                s = rest; // dynamic dim — drop from static shape
                continue;
            }
            break;
        }
        let digits = s.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && s.as_bytes().get(digits) == Some(&b'x') {
            shape.push(s[..digits].parse::<i64>().ok()?);
            s = &s[digits + 1..];
            continue;
        }
        break;
    }
    // Requires at least one *static* dim. `tensor<f32>` (rank-0) and
    // `tensor<?xf16>` (all-dynamic) both yield None, matching the Python helper.
    if shape.is_empty() {
        return None;
    }
    // The element type is the leading run of alphanumerics (stops at `,`/`>`).
    let dtype_end = s
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(s.len());
    let dtype = &s[..dtype_end];
    if dtype.is_empty() {
        return None;
    }
    Some((shape, dtype.to_string()))
}

/// Parse a `memref<DxDx...xELT>` type into `(static_shape, dtype)`, or `None` if
/// not a memref type. Mirrors `parse_tensor_type` but for the `memref<...>`
/// prefix; the Python `KTIRParser` derives `shape`/`dtype` from a memref result
/// type the same way (needed by `ktdp.construct_distributed_memory_view`, whose
/// shape lives only in its `memref<192x64xf16>` result type).
pub fn parse_memref_type(ty: &str) -> Option<(Vec<i64>, String)> {
    let compact: String = ty.chars().filter(|c| !c.is_whitespace()).collect();
    let mut s = compact.strip_prefix("memref<")?;
    let mut shape = Vec::new();
    loop {
        if let Some(after) = s.strip_prefix('?') {
            if let Some(rest) = after.strip_prefix('x') {
                s = rest; // dynamic dim — drop from static shape
                continue;
            }
            break;
        }
        let digits = s.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && s.as_bytes().get(digits) == Some(&b'x') {
            shape.push(s[..digits].parse::<i64>().ok()?);
            s = &s[digits + 1..];
            continue;
        }
        break;
    }
    if shape.is_empty() {
        return None;
    }
    // The element type is the leading run of alphanumerics (stops at `,`/`>`).
    let dtype_end = s
        .find(|c: char| !c.is_ascii_alphanumeric())
        .unwrap_or(s.len());
    let dtype = &s[..dtype_end];
    if dtype.is_empty() {
        return None;
    }
    Some((shape, dtype.to_string()))
}

fn parse_memref_dtype(result_type: &str) -> Option<String> {
    let inner = result_type
        .trim()
        .strip_prefix("memref<")?
        .strip_suffix('>')?;
    let dtype = inner.rsplit('x').next()?.trim();
    if dtype.is_empty() {
        None
    } else {
        Some(dtype.to_string())
    }
}

/// Inner text of a `!ktdp.access_tile<...>` type, e.g.
/// `!ktdp.access_tile<128xindex>` -> `128xindex`.
fn access_tile_inner(result_type: &str) -> Option<String> {
    let inner = result_type
        .trim()
        .strip_prefix("!ktdp.access_tile<")?
        .strip_suffix('>')?;
    Some(inner.to_string())
}

/// Split `NxMx...x<elem>` into its dimension list and element type. The element
/// type may itself contain `x` (e.g. `index`), so we walk the leading `\d+x`
/// run rather than a naive `split('x')`. Mirrors the
/// `^(\d+(?:x\d+)*)x([a-zA-Z_]\w*)$` regex.
fn parse_access_tile_inner(inner: &str) -> Result<(Vec<usize>, String), String> {
    let mut dims = Vec::new();
    let mut rest = inner;
    loop {
        // Consume a `\d+` run.
        let digits_end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if digits_end == 0 {
            break;
        }
        let num: usize = rest[..digits_end]
            .parse()
            .map_err(|_| format!("Malformed access_tile dims in {inner:?}"))?;
        // A dimension is only a dimension if an `x` separator follows it.
        match rest[digits_end..].strip_prefix('x') {
            Some(after) => {
                dims.push(num);
                rest = after;
            }
            None => break,
        }
    }
    if dims.is_empty() || rest.is_empty() {
        return Err(format!(
            "Malformed access_tile type {inner:?}: expected '<dims>x<elem>'"
        ));
    }
    Ok((dims, rest.to_string()))
}

/// Extract a `key = value` attribute value, where `value` is a `keyword<...>`
/// expression (`affine_set<...>` / `affine_map<...>`). Counts `<`/`>` depth
/// while skipping `>=` and `->`, so the constraint operators inside the body do
/// not prematurely close the value. Mirrors `extract_named_attr`'s `keyword<...>`
/// branch in `parser_utils.py`.
fn named_attr_value(text: &str, key: &str) -> Option<String> {
    // Find `key` followed (after optional whitespace) by `=`.
    let mut search = 0;
    let (rest, _val_start) = loop {
        let rel = text[search..].find(key)?;
        let kpos = search + rel;
        // Ensure a word boundary before the key (avoid matching inside a name).
        let prev_ok = kpos == 0
            || !text.as_bytes()[kpos - 1].is_ascii_alphanumeric()
                && text.as_bytes()[kpos - 1] != b'_';
        let after_key = &text[kpos + key.len()..];
        let trimmed = after_key.trim_start();
        if prev_ok && trimmed.starts_with('=') {
            let eq_rel = after_key.find('=').unwrap();
            let val = after_key[eq_rel + 1..].trim_start();
            break (val, kpos);
        }
        search = kpos + key.len();
    };

    // Walk a `keyword<...>` value, counting bracket depth, skipping `>=`/`->`.
    let kw_lt = rest.find('<')?;
    // The portion before `<` must be a bare keyword token (e.g. `affine_set`).
    if !rest[..kw_lt]
        .trim()
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return None;
    }
    let bytes = rest.as_bytes();
    let mut i = kw_lt;
    let mut depth = 0i32;
    while i < bytes.len() {
        let ch = bytes[i] as char;
        if ch == '>' && i + 1 < bytes.len() && bytes[i + 1] == b'=' {
            i += 2; // `>=` constraint operator
            continue;
        }
        if ch == '-' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            i += 2; // `->` affine-map arrow
            continue;
        }
        if ch == '<' {
            depth += 1;
        } else if ch == '>' {
            depth -= 1;
            if depth == 0 {
                return Some(rest[..=i].to_string());
            }
        }
        i += 1;
    }
    None
}

/// Infix index arithmetic: `%r = %a [*+-] %b : type`. Mirrors `_parse_index_binary`.
fn parse_index_binary(text: &str) -> Option<Operation> {
    let (lhs, rhs) = split_assignment(text)?;
    let rhs = rhs.trim();
    if !rhs.starts_with('%') {
        return None;
    }
    let before_colon = rhs.split(':').next().unwrap_or(rhs).trim();
    for (sym, op_name) in [
        ('*', "arith.muli"),
        ('+', "arith.addi"),
        ('-', "arith.subi"),
    ] {
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
    if lhs.starts_with('%') && !lhs.contains(char::is_whitespace) && !rhs.starts_with('=') {
        Some((lhs, rhs))
    } else {
        None
    }
}

/// All `%name` operands before the type, with `{...}` blocks removed and the
/// result name excluded. Mirrors `_extract_operands` (= `find_ssa_names` minus
/// the result). Operands are POSITIONAL, so repeats are kept — `%y = mulf %x,
/// %x` must yield `[%x, %x]` (squaring in RMSNorm/LayerNorm variance), not a
/// deduped `[%x]`.
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
            if Some(name) != result && name.len() > 1 {
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

/// Parse the outermost `{ key = value, ... }` attribute block of an op into a
/// typed attribute map. Mirrors `parse_attr_block`. Entries whose value can't be
/// classified are stored as `Attr::Str` (verbatim). No block -> empty map.
fn parse_attr_block(after_op: &str) -> std::collections::HashMap<String, Attr> {
    let mut attrs = std::collections::HashMap::new();
    let bytes = after_op.as_bytes();
    let Some(open) = after_op.find('{') else {
        return attrs;
    };
    let Some(close) = matching(bytes, open, b'{', b'}') else {
        return attrs;
    };
    for entry in split_top_level(&after_op[open + 1..close], ',') {
        let entry = entry.trim();
        let Some(eq) = entry.find('=') else { continue };
        let key = entry[..eq].trim();
        let val = entry[eq + 1..].trim();
        if key.is_empty() || val.is_empty() {
            continue;
        }
        if let Some(attr) = parse_attr_value(val) {
            attrs.insert(key.to_string(), attr);
        }
    }
    attrs
}

/// Extract the combiner op name from a `linalg.reduce { <dialect.op> }`
/// shorthand block — a `{ }` whose content is a single `dialect.op` identifier
/// (no `=`, no `%`). Returns `None` for the explicit-region form or no block.
fn reduce_shorthand_combiner(after_op: &str) -> Option<String> {
    let b = after_op.as_bytes();
    let open = after_op.find('{')?;
    let close = matching(b, open, b'{', b'}')?;
    let inner = after_op[open + 1..close].trim();
    if inner.contains('=') || inner.contains('%') || inner.contains('{') {
        return None; // attribute block or region, not a combiner shorthand
    }
    // A single `dialect.op` token (letters/digits/_/.), e.g. `arith.maximumf`.
    if !inner.is_empty()
        && inner.contains('.')
        && inner
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.'))
    {
        Some(inner.to_string())
    } else {
        None
    }
}

/// Scan for bare `key = value` attributes at top level (outside `()`, `{}`,
/// `<>`) — MLIR named ops attach `permutation = [..]`, `dimensions = [..]`, etc.
/// without an enclosing `{ }`. Mirrors `_parse_bare_attr`. At depth 0 a `=` is
/// always an attribute assignment (the result `%x =` is already stripped, and
/// `>=`/`<=` only occur inside `<>` at depth > 0).
fn parse_bare_attrs(text: &str) -> std::collections::HashMap<String, Attr> {
    let mut attrs = std::collections::HashMap::new();
    let b = text.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'(' | b'{' | b'<' | b'[' => depth += 1,
            b')' | b'}' | b'>' | b']' => depth -= 1,
            b'=' if depth == 0 && b.get(i + 1) != Some(&b'=') && i > 0 && b[i - 1] != b'=' => {
                // Walk back over whitespace to capture the key identifier.
                let mut ks = i;
                while ks > 0 && b[ks - 1].is_ascii_whitespace() {
                    ks -= 1;
                }
                let ke = ks;
                while ks > 0
                    && (b[ks - 1].is_ascii_alphanumeric() || matches!(b[ks - 1], b'_' | b'.'))
                {
                    ks -= 1;
                }
                let key = &text[ks..ke];
                // Read the value after `=`.
                let mut vs = i + 1;
                while vs < b.len() && b[vs].is_ascii_whitespace() {
                    vs += 1;
                }
                let (raw, end) = read_attr_value(text, vs);
                if !key.is_empty()
                    && let Some(attr) = parse_attr_value(raw)
                {
                    attrs.insert(key.to_string(), attr);
                }
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    attrs
}

/// Read a bare attribute value starting at `start`: a balanced `[..]` list, a
/// balanced `keyword<..>` (affine map/set, memory space), or a plain token up to
/// the next whitespace / `,` / top-level `:`. Returns `(value_str, end_index)`.
fn read_attr_value(text: &str, start: usize) -> (&str, usize) {
    let b = text.as_bytes();
    if start >= b.len() {
        return ("", start);
    }
    if b[start] == b'['
        && let Some(close) = matching(b, start, b'[', b']')
    {
        return (&text[start..=close], close + 1);
    }
    // keyword<...> (affine_map<>, affine_set<>, #ktdp...<>): balance <> while
    // skipping `->` and `>=` so constraint operators don't close early.
    if let Some(lt) = text[start..].find('<') {
        let head = &text[start..start + lt];
        if head
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '_' | '.' | '#'))
            && !head.is_empty()
        {
            let mut depth = 0i32;
            let vb = text.as_bytes();
            let mut j = start + lt;
            while j < vb.len() {
                match vb[j] {
                    b'<' => depth += 1,
                    b'>' if vb.get(j + 1) == Some(&b'=') => {} // `>=`, not a close
                    b'-' if vb.get(j + 1) == Some(&b'>') => j += 1, // skip `->`
                    b'>' => {
                        depth -= 1;
                        if depth == 0 {
                            return (&text[start..=j], j + 1);
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
        }
    }
    // Plain token up to whitespace / comma / colon.
    let end = text[start..]
        .find(|c: char| c.is_whitespace() || c == ',' || c == ':')
        .map(|o| start + o)
        .unwrap_or(b.len());
    (&text[start..end], end)
}

/// Split `s` on `sep` at top level only — commas inside `[]`, `<>`, `()` or `{}`
/// are not separators (affine maps, lists, nested types contain them).
fn split_top_level(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '[' | '<' | '(' | '{' => depth += 1,
            ']' | '>' | ')' | '}' => depth -= 1,
            _ if c == sep && depth == 0 => {
                out.push(s[start..i].to_string());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(s[start..].to_string());
    out
}

/// Classify a single attribute value into an [`Attr`].
fn parse_attr_value(val: &str) -> Option<Attr> {
    let val = val.trim();
    // `affine_map<...>` / `affine_set<...>`
    if val.starts_with("affine_map<") {
        return parse_affine_map(val).ok().map(Attr::AffineMap);
    }
    if val.starts_with("affine_set<") {
        return parse_affine_set(val).ok().map(Attr::AffineSet);
    }
    // `[a, b, ...]` list -> IntList unless any element is float-shaped.
    if let Some(inner) = val.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let items: Vec<String> = split_top_level(inner, ',')
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if items
            .iter()
            .any(|s| s.contains('.') || s.contains('e') || s.contains('E'))
        {
            let vals: Option<Vec<f64>> = items.iter().map(|s| s.parse().ok()).collect();
            return vals.map(Attr::FloatList);
        }
        let vals: Option<Vec<i64>> = items.iter().map(|s| s.parse().ok()).collect();
        return vals.map(Attr::IntList);
    }
    // Strip a trailing `: type` annotation MLIR attaches to typed attrs
    // (`42 : i32` -> `42`), then classify the bare token.
    let core = val.split(':').next().unwrap_or(val).trim();
    match core {
        "true" => return Some(Attr::Bool(true)),
        "false" => return Some(Attr::Bool(false)),
        _ => {}
    }
    if let Ok(i) = core.parse::<i64>() {
        return Some(Attr::Int(i));
    }
    if (core.contains('.') || core.contains('e') || core.contains('E'))
        && let Ok(f) = core.parse::<f64>()
    {
        return Some(Attr::Float(f));
    }
    Some(Attr::Str(core.to_string()))
}

/// Parse the literal of `arith.constant <lit> : <type>` into a value `Attr`.
///
/// Handles scalar `true`/`false`, decimal ints/floats, hex bit-pattern literals
/// (`0xFF80` — used for ±inf/NaN encodings), and `dense<...>` tensor constants
/// (splat scalar or `[..]` list). Mirrors `parse_numeric` + the dense-payload
/// handling in `parser_utils.py`.
fn parse_constant_value(after_op: &str) -> Result<Attr, String> {
    // Attribute-block form: `arith.constant { value = 42 : i32 } : index`.
    // The value lives in the `{ }` block, not as a bare literal.
    if after_op.trim_start().starts_with('{')
        && let Some(Attr::Int(_) | Attr::Float(_) | Attr::Bool(_)) =
            parse_attr_block(after_op).get("value")
    {
        return Ok(parse_attr_block(after_op).remove("value").unwrap());
    }
    // The literal runs from after the op name to the `:` type annotation; for
    // `dense<...>` it may contain `[ , ]`, so take everything before the LAST
    // top-level `:` rather than the first whitespace token.
    let head = after_op.split_whitespace().next().unwrap_or("").trim();
    if let Some(inner) = head
        .strip_prefix("dense<")
        .and_then(|s| s.strip_suffix('>'))
    {
        return parse_dense_payload(inner);
    }
    let lit = head;
    match lit {
        "true" => Ok(Attr::Bool(true)),
        "false" => Ok(Attr::Bool(false)),
        _ => parse_scalar_numeric(lit),
    }
}

/// A single numeric literal: hex bit-pattern, decimal float, or decimal int.
fn parse_scalar_numeric(lit: &str) -> Result<Attr, String> {
    if let Some(hex) = lit.strip_prefix("0x").or_else(|| lit.strip_prefix("0X")) {
        return i64::from_str_radix(hex, 16)
            .map(Attr::Int)
            .map_err(|_| format!("arith.constant: bad hex literal {lit:?}"));
    }
    if lit.contains('.') || lit.contains('e') || lit.contains('E') {
        return lit
            .parse::<f64>()
            .map(Attr::Float)
            .map_err(|_| format!("arith.constant: bad float literal {lit:?}"));
    }
    lit.parse::<i64>()
        .map(Attr::Int)
        .map_err(|_| format!("arith.constant: bad int literal {lit:?}"))
}

/// `dense<...>` payload: a `[a, b, ...]` list -> `FloatList`, or a splat scalar
/// -> `Float`/`Int`. The result type (carried separately) tells the consumer
/// the shape/dtype; here we only lift the values.
fn parse_dense_payload(inner: &str) -> Result<Attr, String> {
    let inner = inner.trim();
    if let Some(list) = inner.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let vals = list
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(parse_f64_lit)
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Attr::FloatList(vals));
    }
    match parse_scalar_numeric(inner)? {
        Attr::Int(i) => Ok(Attr::Float(i as f64)), // splat — normalize to float payload
        other => Ok(other),
    }
}

fn parse_f64_lit(s: &str) -> Result<f64, String> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        return i64::from_str_radix(hex, 16)
            .map(|i| i as f64)
            .map_err(|_| format!("dense: bad hex element {s:?}"));
    }
    s.parse::<f64>()
        .map_err(|_| format!("dense: bad element {s:?}"))
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

    const VECTOR_ADD: &str = include_str!("../../examples/triton-ktir/vector_add_ktir.mlir");

    // RUST-ONLY (not in the Python suite): regression for the operand-dedup bug.
    // MLIR operands are positional, so a repeated operand (`%y = mulf %x, %x`,
    // the squaring step in RMSNorm/LayerNorm variance) must be kept, not deduped.
    #[test]
    fn extract_operands_keeps_positional_repeats() {
        let ops = extract_operands("%x, %x : tensor<1x1024xf16>", Some("%y"));
        assert_eq!(ops, vec!["%x".to_string(), "%x".to_string()]);
    }

    #[test]
    fn parses_real_vector_add_structurally() {
        let module = parse_module(VECTOR_ADD).unwrap();
        let f = module.get_function("add_kernel").unwrap();
        assert_eq!(f.grid, (32, 1, 1));
        assert_eq!(
            f.arg_names(),
            vec!["x_ptr", "y_ptr", "output_ptr", "BLOCK_SIZE"]
        );

        // The multi-line construct ops must each tokenize to exactly one op.
        let types: Vec<&str> = f.operations.iter().map(|o| o.op_type.as_str()).collect();
        assert_eq!(
            types
                .iter()
                .filter(|t| **t == "ktdp.construct_memory_view")
                .count(),
            3
        );
        assert_eq!(
            types
                .iter()
                .filter(|t| **t == "ktdp.construct_access_tile")
                .count(),
            3
        );
        assert_eq!(types.iter().filter(|t| **t == "ktdp.load").count(), 2);
        assert_eq!(types.iter().filter(|t| **t == "ktdp.store").count(), 1);
        assert!(types.contains(&"arith.addf"));
        assert_eq!(types.last(), Some(&"return"));

        // Operand/type wiring on a representative multi-line op.
        let view = f
            .operations
            .iter()
            .find(|o| o.result.as_deref() == Some("%x_view"))
            .unwrap();
        assert_eq!(view.operands, vec!["%x_ptr"]);
        assert_eq!(view.result_type.as_deref(), Some("memref<4096xf16>"));

        let at = f
            .operations
            .iter()
            .find(|o| o.result.as_deref() == Some("%x_tile"))
            .unwrap();
        assert_eq!(at.operands, vec!["%x_view", "%offset"]);
        assert_eq!(
            at.result_type.as_deref(),
            Some("!ktdp.access_tile<128xindex>")
        );
    }

    #[test]
    fn infix_index_arith_lowers_to_arith_op() {
        let module = parse_module(VECTOR_ADD).unwrap();
        let f = module.get_function("add_kernel").unwrap();
        let off = f
            .operations
            .iter()
            .find(|o| o.result.as_deref() == Some("%offset"))
            .unwrap();
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
        let env = ExecutionEnv::new(&dispatch, &grid);
        let mut ctx = single_core_context();
        execute_ops(&f.operations, &mut ctx, &env).unwrap();
        match ctx.get_value("%d").unwrap() {
            Value::Scalar(Scalar::F32(v)) => assert_eq!(*v, 10.0), // (2+3)*2
            other => panic!("expected F32(10.0), got {other:?}"),
        }
    }

    // --- ktdp construct-op attribute parsing --------------------------------

    use crate::affine::{AffineExpr, ConstraintKind};

    /// Fetch the attribute map for the op binding `result`, failing the test if
    /// it is missing.
    fn attrs_of<'a>(
        f: &'a IRFunction,
        result: &str,
    ) -> &'a std::collections::HashMap<String, Attr> {
        &f.operations
            .iter()
            .find(|o| o.result.as_deref() == Some(result))
            .unwrap_or_else(|| panic!("no op binding {result}"))
            .attributes
    }

    #[test]
    fn construct_memory_view_carries_real_attributes() {
        let module = parse_module(VECTOR_ADD).unwrap();
        let f = module.get_function("add_kernel").unwrap();
        let a = attrs_of(f, "%x_view");

        // sizes: [4096] -> shape
        assert_eq!(a.get("shape"), Some(&Attr::IntList(vec![4096])));
        // strides: [1]
        assert_eq!(a.get("strides"), Some(&Attr::IntList(vec![1])));
        // memref<4096xf16> -> dtype f16
        assert_eq!(a.get("dtype"), Some(&Attr::Str("f16".to_string())));
        // #ktdp.spyre_memory_space<HBM>
        assert_eq!(a.get("memory_space"), Some(&Attr::Str("HBM".to_string())));
        // No per-core LX tag here.
        assert_eq!(a.get("lx_core_id"), None);

        // coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 4095 >= 0)>
        match a.get("coordinate_set") {
            Some(Attr::AffineSet(set)) => {
                assert_eq!(set.num_dims, 1);
                assert_eq!(set.constraints.len(), 2);
                // 0 <= d0 <= 4095
                assert!(set.contains(&[0], &[] as &[i64]));
                assert!(set.contains(&[4095], &[] as &[i64]));
                assert!(!set.contains(&[4096], &[] as &[i64]));
            }
            other => panic!("expected AffineSet coordinate_set, got {other:?}"),
        }
    }

    #[test]
    fn construct_access_tile_carries_real_attributes() {
        let module = parse_module(VECTOR_ADD).unwrap();
        let f = module.get_function("add_kernel").unwrap();
        let a = attrs_of(f, "%x_tile");

        // access_tile<128xindex> -> shape [128]
        assert_eq!(a.get("shape"), Some(&Attr::IntList(vec![128])));

        // base_map is absent in the source -> synthesized identity over 1 dim
        // (operands = [%x_view, %offset], so n = max(1, 2-1) = 1).
        match a.get("base_map") {
            Some(Attr::AffineMap(m)) => {
                assert_eq!(m.num_dims, 1);
                assert_eq!(m.exprs, vec![AffineExpr::Dim(0)]);
            }
            other => panic!("expected AffineMap base_map, got {other:?}"),
        }

        // access_tile_set is 0 <= d0 <= 127, which is FULL over the 128-extent
        // tile, so it is normalised away (no coordinate_set attribute).
        assert_eq!(a.get("coordinate_set"), None);

        // access_tile_order = identity map -> normalised away.
        assert_eq!(a.get("coordinate_order"), None);
    }

    #[test]
    fn construct_access_tile_keeps_nontrivial_coordinate_set() {
        // A genuinely restricting set (only even-indexed first half) must NOT be
        // dropped, and a permuting order map must be preserved.
        let src = r#"
            module {
              func.func @k(%p: index) attributes {grid = [1]} {
                %t = ktdp.construct_access_tile %v[%p] {
                  access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 == 0)>,
                  access_tile_order = affine_map<(d0, d1) -> (d1, d0)>,
                  base_map = affine_map<(d0, d1) -> (d0, d1)>
                } : memref<8x8xf16> -> !ktdp.access_tile<4x4xindex>
                return
              }
            }
        "#;
        let module = parse_module(src).unwrap();
        let f = module.get_function("k").unwrap();
        let a = attrs_of(f, "%t");

        assert_eq!(a.get("shape"), Some(&Attr::IntList(vec![4, 4])));

        // d1 == 0 excludes most of the 4x4 box, so the set is retained.
        match a.get("coordinate_set") {
            Some(Attr::AffineSet(set)) => {
                assert_eq!(set.num_dims, 2);
                assert_eq!(set.constraints[2].kind, ConstraintKind::Equal);
                assert!(set.contains(&[2, 0], &[] as &[i64]));
                assert!(!set.contains(&[2, 1], &[] as &[i64]));
            }
            other => panic!("expected retained AffineSet, got {other:?}"),
        }

        // The (d0,d1)->(d1,d0) order map is a permutation, not identity: kept.
        match a.get("coordinate_order") {
            Some(Attr::AffineMap(m)) => assert_eq!(m.eval(&[1, 2], &[]), vec![2, 1]),
            other => panic!("expected retained AffineMap order, got {other:?}"),
        }
    }

    #[test]
    fn construct_memory_view_parses_lx_core_and_strides() {
        // Per-core LX memory space and a multi-dim strided view.
        let src = r#"
            module {
              func.func @k(%p: index) attributes {grid = [1]} {
                %v = ktdp.construct_memory_view %p, sizes: [16, 32], strides: [32, 1] {
                  coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 15 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
                  memory_space = #ktdp.spyre_memory_space<LX, core = 3>
                } : memref<16x32xf32>
                return
              }
            }
        "#;
        let module = parse_module(src).unwrap();
        let f = module.get_function("k").unwrap();
        let a = attrs_of(f, "%v");

        assert_eq!(a.get("shape"), Some(&Attr::IntList(vec![16, 32])));
        assert_eq!(a.get("strides"), Some(&Attr::IntList(vec![32, 1])));
        assert_eq!(a.get("dtype"), Some(&Attr::Str("f32".to_string())));
        assert_eq!(a.get("memory_space"), Some(&Attr::Str("LX".to_string())));
        assert_eq!(a.get("lx_core_id"), Some(&Attr::Int(3)));
        assert!(matches!(a.get("coordinate_set"), Some(Attr::AffineSet(_))));
    }

    #[test]
    fn all_construct_ops_in_vector_add_carry_attributes() {
        // Regression guard: every construct op in the real example must end up
        // with the load-bearing attributes populated.
        let module = parse_module(VECTOR_ADD).unwrap();
        let f = module.get_function("add_kernel").unwrap();
        for op in &f.operations {
            match op.op_type.as_str() {
                "ktdp.construct_memory_view" => {
                    let a = &op.attributes;
                    assert!(a.contains_key("shape"), "view missing shape");
                    assert!(a.contains_key("strides"), "view missing strides");
                    assert!(a.contains_key("dtype"), "view missing dtype");
                    assert!(a.contains_key("memory_space"), "view missing memory_space");
                    assert!(
                        a.contains_key("coordinate_set"),
                        "view missing coordinate_set"
                    );
                }
                "ktdp.construct_access_tile" => {
                    let a = &op.attributes;
                    assert!(a.contains_key("shape"), "tile missing shape");
                    assert!(a.contains_key("base_map"), "tile missing base_map");
                }
                _ => {}
            }
        }
    }
}
