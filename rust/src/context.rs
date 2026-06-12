// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Per-core execution state — port of `CoreContext` from `ktir_cpu/grid.py`.
//!
//! Replaces the slice-1 `Scope`. Holds the region-scoped SSA value stack, the
//! LX bump-allocator with watermark rewinding, and the grid position. Comm
//! wiring (`send_to` / remote `get_lx`) is present as the locked seam; the
//! scheduler that fills it is implement-phase.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::ir::Value;
use crate::memory::{HBMSimulator, LXScratchpad};

/// Per-core execution context. One per core; handlers receive `&mut CoreContext`.
pub struct CoreContext {
    pub core_id: usize,
    /// (x, y, z) position; derived from `core_id`, immutable.
    pub grid_pos: (usize, usize, usize),
    pub hbm: Rc<RefCell<HBMSimulator>>,
    pub lx: Rc<RefCell<LXScratchpad>>,
    /// All cores' LX, for remote `get_lx` during comm.
    all_lx: Vec<Rc<RefCell<LXScratchpad>>>,
    /// Region-scoped SSA map; index 0 is the function body. Inner scopes shadow.
    scope_stack: Vec<HashMap<String, Value>>,
    /// SSA name -> LX bytes; single source of truth for `lx.used`.
    lx_bytes: HashMap<String, i64>,
    /// Bump-allocator watermarks; `len == scope_stack.len() - 1`.
    lx_next_ptr_stack: Vec<i64>,
}

impl CoreContext {
    pub fn new(
        core_id: usize,
        grid_pos: (usize, usize, usize),
        hbm: Rc<RefCell<HBMSimulator>>,
        lx: Rc<RefCell<LXScratchpad>>,
        all_lx: Vec<Rc<RefCell<LXScratchpad>>>,
    ) -> Self {
        CoreContext {
            core_id,
            grid_pos,
            hbm,
            lx,
            all_lx,
            scope_stack: vec![HashMap::new()],
            lx_bytes: HashMap::new(),
            lx_next_ptr_stack: Vec::new(),
        }
    }

    /// Grid coordinate for a dimension (0=x, 1=y, 2=z). Mirrors `get_grid_id`.
    pub fn get_grid_id(&self, dim: usize) -> usize {
        match dim {
            0 => self.grid_pos.0,
            1 => self.grid_pos.1,
            2 => self.grid_pos.2,
            _ => 0,
        }
    }

    /// Bind an SSA value in the topmost scope. Mirrors `set_value`.
    pub fn set_value(&mut self, name: &str, value: Value) {
        self.scope_stack
            .last_mut()
            .expect("scope stack never empty")
            .insert(normalize(name), value);
    }

    /// Look up an SSA value, searching scopes top-to-bottom. Mirrors `get_value`.
    pub fn get_value(&self, name: &str) -> Result<&Value, String> {
        let key = normalize(name);
        for scope in self.scope_stack.iter().rev() {
            if let Some(v) = scope.get(&key) {
                return Ok(v);
            }
        }
        Err(format!("undefined SSA value: {name}"))
    }

    pub fn has_value(&self, name: &str) -> bool {
        let key = normalize(name);
        self.scope_stack.iter().any(|s| s.contains_key(&key))
    }

    /// Enter a region: snapshot the LX watermark, push a fresh scope.
    pub fn push_scope(&mut self) {
        self.lx_next_ptr_stack.push(self.lx.borrow().next_ptr);
        self.scope_stack.push(HashMap::new());
    }

    /// Exit the current region: untrack its values, rewind LX to the watermark.
    /// Panics if called on the function-body scope. Mirrors `pop_scope`.
    pub fn pop_scope(&mut self) {
        assert!(self.scope_stack.len() > 1, "cannot pop function-body scope");
        let scope = self.scope_stack.pop().unwrap();
        for name in scope.keys() {
            self.untrack_lx(name);
        }
        let watermark = self.lx_next_ptr_stack.pop().unwrap();
        self.lx.borrow_mut().next_ptr = watermark;
    }

    /// Record an SSA value occupying `size_bytes` in LX. Mirrors `track_lx`;
    /// returns an error instead of raising `MemoryError` on overflow.
    pub fn track_lx(&mut self, name: &str, size_bytes: i64) -> Result<(), String> {
        let (used, capacity) = {
            let lx = self.lx.borrow();
            (lx.used, lx.capacity)
        };
        if used + size_bytes > capacity {
            return Err(format!(
                "LX capacity exceeded on core {}: {} + {} > {}",
                self.core_id, used, size_bytes, capacity
            ));
        }
        self.lx.borrow_mut().used += size_bytes;
        self.lx_bytes.insert(normalize(name), size_bytes);
        Ok(())
    }

    /// Free LX for `name`. No-op if untracked. Mirrors `untrack_lx`.
    pub fn untrack_lx(&mut self, name: &str) {
        if let Some(sz) = self.lx_bytes.remove(&normalize(name)) {
            self.lx.borrow_mut().used -= sz;
        }
    }

    /// Return the LX for a core: local fast path, else a remote handle.
    /// Mirrors `get_lx`.
    pub fn get_lx(&self, core_id: Option<usize>) -> Rc<RefCell<LXScratchpad>> {
        match core_id {
            None => Rc::clone(&self.lx),
            Some(id) if id == self.core_id => Rc::clone(&self.lx),
            Some(id) => Rc::clone(&self.all_lx[id]),
        }
    }

    /// Reset for the next execution round. Mirrors `clear_values`.
    pub fn clear_values(&mut self) {
        self.scope_stack = vec![HashMap::new()];
        self.lx_bytes.clear();
        self.lx_next_ptr_stack.clear();
        self.lx.borrow_mut().clear();
    }
}

fn normalize(name: &str) -> String {
    name.trim_start_matches('%').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Scalar;
    use crate::memory::SpyreMemoryHierarchy;

    fn ctx() -> CoreContext {
        let mem = SpyreMemoryHierarchy::new(2);
        CoreContext::new(
            0,
            (0, 0, 0),
            Rc::clone(&mem.hbm),
            mem.get_lx(0),
            mem.lx_scratchpads.clone(),
        )
    }

    #[test]
    fn scopes_shadow_and_resolve_outward() {
        let mut c = ctx();
        c.set_value("%a", Value::Index(1));
        c.push_scope();
        c.set_value("%b", Value::Index(2));
        assert!(matches!(c.get_value("%a").unwrap(), Value::Index(1))); // outer visible
        assert!(matches!(c.get_value("%b").unwrap(), Value::Index(2)));
        c.pop_scope();
        assert!(c.get_value("%b").is_err()); // inner gone
        assert!(c.has_value("%a"));
    }

    #[test]
    fn lx_tracking_and_watermark_rewind() {
        let mut c = ctx();
        c.track_lx("%t", 256).unwrap();
        assert_eq!(c.lx.borrow().used, 256);
        c.push_scope();
        c.set_value("%inner", Value::Scalar(Scalar::I64(0)));
        c.track_lx("%inner", 128).unwrap();
        assert_eq!(c.lx.borrow().used, 384);
        c.pop_scope(); // frees %inner
        assert_eq!(c.lx.borrow().used, 256);
    }

    #[test]
    fn lx_overflow_is_an_error() {
        let mut c = ctx();
        let cap = c.lx.borrow().capacity;
        assert!(c.track_lx("%big", cap + 1).is_err());
    }
}
