// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Memory hierarchy — port of `ktir_cpu/memory.py`.
//!
//! Storage model: Python keys a dict by byte address -> ndarray. Here each
//! allocation is a contiguous byte buffer keyed by its base byte address; reads
//! interpret bytes per the requested dtype at the load/store boundary (the
//! load/store data path itself is an implement-phase fill in the ktdp dialect).
//! Cross-core sharing uses `Rc<RefCell<>>` to mirror Python reference semantics
//! — the scheduler is single-threaded/cooperative, so no `Arc`/`Mutex` needed.

use crate::dtypes::DType;
use crate::fxhash::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;

/// HBM "stick" (cache block) size in bytes. Defined in `ktir-core` (memref byte
/// addressing needs it); re-exported here so `crate::memory::STICK_BYTES`
/// resolves unchanged.
pub use crate::memref::STICK_BYTES;

/// Shared, byte-addressed HBM with stick-granular addressing.
#[derive(Debug)]
pub struct HBMSimulator {
    pub size_bytes: i64,
    /// base byte address -> raw allocation bytes
    allocations: FxHashMap<i64, Vec<u8>>,
    /// next unallocated byte address (stick-aligned); starts at 0x10000
    pub next_ptr: i64,
}

impl Default for HBMSimulator {
    fn default() -> Self {
        HBMSimulator::new(128)
    }
}

impl HBMSimulator {
    pub fn new(size_gb: i64) -> Self {
        HBMSimulator {
            size_bytes: size_gb * 1024 * 1024 * 1024,
            allocations: FxHashMap::default(),
            next_ptr: 0x10000,
        }
    }

    /// Allocate `size` bytes, advance `next_ptr` to the next stick boundary,
    /// return the stick address (`ptr / STICK_BYTES`). Mirrors `allocate`.
    pub fn allocate(&mut self, size: i64) -> i64 {
        debug_assert_eq!(
            self.next_ptr % STICK_BYTES,
            0,
            "next_ptr must be stick-aligned"
        );
        let ptr = self.next_ptr;
        self.allocations
            .entry(ptr)
            .or_insert_with(|| vec![0u8; size.max(0) as usize]);
        let advanced = ptr + size;
        self.next_ptr = (advanced + STICK_BYTES - 1) & !(STICK_BYTES - 1);
        ptr / STICK_BYTES
    }

    /// Read `len` raw bytes starting at absolute `byte_addr`, zero-padding past
    /// the end of the containing allocation. Byte-level analogue of `_read_flat`.
    pub fn read_bytes(&self, byte_addr: i64, len: usize) -> Vec<u8> {
        read_bytes(&self.allocations, byte_addr, len)
    }

    /// Read `n` elements of `dtype` and decode to f32 in one pass, straight from
    /// the backing buffer — no intermediate byte `Vec`/memmove. `codec::decode`
    /// zero-pads when the read runs past the allocation. The contiguous-load fast
    /// path (the common case) uses this.
    pub fn read_decoded(&self, byte_addr: i64, n: usize, dtype: DType) -> Vec<f32> {
        read_decoded(&self.allocations, byte_addr, n, dtype)
    }

    /// Write raw bytes at absolute `byte_addr`, growing/creating the allocation.
    pub fn write_bytes(&mut self, byte_addr: i64, data: &[u8]) {
        write_bytes(&mut self.allocations, byte_addr, data);
    }
}

/// LX live-set budget for fused-segment planning: 7/8 of the 2 MB per-core LX,
/// leaving headroom for a node's transient temporaries. Passed to
/// `ktir_optimizer::fusion::plan_segments_budgeted` so a fused segment's
/// co-resident `[m, *]` intermediates never overflow LX (without it, a whole
/// transformer MLP fuses into one segment and overflows at larger token counts —
/// llama m=32). 2 MB matches `LXScratchpad::new(.., 2)` below.
pub const LX_FUSION_BUDGET_BYTES: usize = (2 * 1024 * 1024) * 7 / 8;

/// The effective LX fusion budget — [`LX_FUSION_BUDGET_BYTES`] unless overridden
/// by `KTIR_LX_FUSION_BUDGET` (bytes). The override lets a host with a different
/// LX size tune fusion, and lets tests force splitting on small models.
pub fn lx_fusion_budget() -> usize {
    std::env::var("KTIR_LX_FUSION_BUDGET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(LX_FUSION_BUDGET_BYTES)
}

/// Per-core local scratchpad. Plain byte addressing, no stick concept.
#[derive(Debug)]
pub struct LXScratchpad {
    pub capacity: i64,
    pub used: i64,
    pub core_id: usize,
    allocations: FxHashMap<i64, Vec<u8>>,
    pub next_ptr: i64,
}

impl LXScratchpad {
    pub fn new(core_id: usize, size_mb: i64) -> Self {
        LXScratchpad {
            capacity: size_mb * 1024 * 1024,
            used: 0,
            core_id,
            allocations: FxHashMap::default(),
            next_ptr: 0,
        }
    }

    pub fn read_bytes(&self, ptr: i64, len: usize) -> Vec<u8> {
        read_bytes(&self.allocations, ptr, len)
    }

    /// Decode `n` elements of `dtype` directly from the backing buffer (no
    /// intermediate byte `Vec`). See [`HBMSimulator::read_decoded`].
    pub fn read_decoded(&self, ptr: i64, n: usize, dtype: DType) -> Vec<f32> {
        read_decoded(&self.allocations, ptr, n, dtype)
    }

    pub fn write_bytes(&mut self, ptr: i64, data: &[u8]) {
        write_bytes(&mut self.allocations, ptr, data);
    }

    /// Reset for the next execution round. Mirrors `clear`.
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.next_ptr = 0;
        self.used = 0;
    }
}

/// Shared HBM + one LX per core. Mirrors `SpyreMemoryHierarchy`.
pub struct SpyreMemoryHierarchy {
    pub num_cores: usize,
    pub hbm: Rc<RefCell<HBMSimulator>>,
    pub lx_scratchpads: Vec<Rc<RefCell<LXScratchpad>>>,
}

impl SpyreMemoryHierarchy {
    pub fn new(num_cores: usize) -> Self {
        let lx = (0..num_cores)
            .map(|c| Rc::new(RefCell::new(LXScratchpad::new(c, 2))))
            .collect();
        SpyreMemoryHierarchy {
            num_cores,
            hbm: Rc::new(RefCell::new(HBMSimulator::default())),
            lx_scratchpads: lx,
        }
    }

    /// Route to a core's LX. Mirrors `get_lx`.
    pub fn get_lx(&self, core_id: usize) -> Rc<RefCell<LXScratchpad>> {
        Rc::clone(&self.lx_scratchpads[core_id])
    }
}

// --- shared byte-buffer helpers (port of _find_allocation/_read_flat/_write_flat) ---

/// Find the allocation containing `ptr`, returning `(base, len)`.
fn find_allocation(allocs: &FxHashMap<i64, Vec<u8>>, ptr: i64) -> Option<(i64, usize)> {
    if let Some(buf) = allocs.get(&ptr) {
        return Some((ptr, buf.len()));
    }
    allocs
        .iter()
        .find(|(base, buf)| ptr > **base && ptr < **base + buf.len() as i64)
        .map(|(&base, buf)| (base, buf.len()))
}

fn read_bytes(allocs: &FxHashMap<i64, Vec<u8>>, ptr: i64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    if let Some((base, _)) = find_allocation(allocs, ptr) {
        let buf = &allocs[&base];
        let off = (ptr - base) as usize;
        let avail = buf.len().saturating_sub(off);
        let n = avail.min(len);
        out[..n].copy_from_slice(&buf[off..off + n]);
    }
    out // zero-padded past allocation end (matches Python)
}

/// Read + decode in one pass: `codec::decode` reads the backing bytes directly
/// (zero-padding a short tail itself), so the contiguous load fast path skips
/// the intermediate byte `Vec` and its memmove.
fn read_decoded(allocs: &FxHashMap<i64, Vec<u8>>, ptr: i64, n: usize, dtype: DType) -> Vec<f32> {
    match find_allocation(allocs, ptr) {
        Some((base, _)) => {
            let buf = &allocs[&base];
            let off = (ptr - base) as usize;
            let end = (off + n * dtype.bytes_per_elem()).min(buf.len());
            crate::codec::decode(&buf[off..end], n, dtype)
        }
        None => crate::codec::decode(&[], n, dtype),
    }
}

fn write_bytes(allocs: &mut FxHashMap<i64, Vec<u8>>, ptr: i64, data: &[u8]) {
    if let Some((base, buflen)) = find_allocation(allocs, ptr) {
        let off = (ptr - base) as usize;
        let needed = off + data.len();
        let buf = allocs.get_mut(&base).unwrap();
        if needed > buflen {
            buf.resize(needed, 0);
        }
        buf[off..off + data.len()].copy_from_slice(data);
    } else {
        allocs.insert(ptr, data.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_is_stick_aligned() {
        let mut hbm = HBMSimulator::default();
        let s0 = hbm.allocate(100); // 100 bytes -> rounds to 128
        let s1 = hbm.allocate(10);
        assert_eq!(s0, 0x10000 / STICK_BYTES);
        assert_eq!(hbm.next_ptr % STICK_BYTES, 0);
        assert_eq!(s1 - s0, 1); // next stick
    }

    #[test]
    fn write_then_read_roundtrips_with_zero_pad() {
        let mut lx = LXScratchpad::new(0, 2);
        lx.write_bytes(64, &[1, 2, 3, 4]);
        assert_eq!(lx.read_bytes(64, 4), vec![1, 2, 3, 4]);
        // reading past the end zero-pads
        assert_eq!(lx.read_bytes(64, 6), vec![1, 2, 3, 4, 0, 0]);
        // unmapped reads are all zero
        assert_eq!(lx.read_bytes(4096, 3), vec![0, 0, 0]);
    }

    #[test]
    fn hierarchy_shares_hbm_routes_lx() {
        let mem = SpyreMemoryHierarchy::new(4);
        assert_eq!(mem.num_cores, 4);
        mem.get_lx(2).borrow_mut().write_bytes(0, &[9]);
        assert_eq!(mem.get_lx(2).borrow().read_bytes(0, 1), vec![9]);
        assert_eq!(mem.get_lx(0).borrow().read_bytes(0, 1), vec![0]); // independent
    }
}
