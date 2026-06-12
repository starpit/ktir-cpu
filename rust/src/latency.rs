// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Latency model — port of `ktir_cpu/latency.py`. This slice locks the
//! `LatencyCategory` enum (the dispatch table pairs every op with one) and the
//! `HardwareConfig` cost parameters with their computed roofline properties.
//! `LatencyTracker` / `LatencyReport` (per-op accounting + bottleneck
//! classification) are an implement-phase fill against these locked types.

/// Cost class assigned to each op at registration. Mirrors the `LatencyCategory`
/// StrEnum — all seven members.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyCategory {
    Zero,
    Memory,
    ComputeFloat,
    ComputeTranscendental,
    ComputeInt,
    ComputeMatmul,
    Comm,
}

impl LatencyCategory {
    /// The `StrEnum` string value, for parity with the Python registry.
    pub fn as_str(self) -> &'static str {
        match self {
            LatencyCategory::Zero => "zero",
            LatencyCategory::Memory => "memory",
            LatencyCategory::ComputeFloat => "compute_float",
            LatencyCategory::ComputeTranscendental => "compute_transcendental",
            LatencyCategory::ComputeInt => "compute_int",
            LatencyCategory::ComputeMatmul => "compute_matmul",
            LatencyCategory::Comm => "comm",
        }
    }
}

/// Hardware cost parameters. Mirrors `HardwareConfig` defaults exactly.
#[derive(Clone, Copy, Debug)]
pub struct HardwareConfig {
    pub num_cores: usize,
    pub clock_ghz: f64,
    pub hbm_bandwidth_tb_s: f64,
    pub ring_bandwidth_tb_s: f64,
    pub simd_elements_per_cycle: u32,
    pub systolic_flops_per_cycle: u64,
    pub transcendental_penalty: u32,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        HardwareConfig {
            num_cores: 32,
            clock_ghz: 1.0,
            hbm_bandwidth_tb_s: 1.0,
            ring_bandwidth_tb_s: 4.0,
            simd_elements_per_cycle: 64,
            systolic_flops_per_cycle: 2 * 64 * 64 * 64, // 524288
            transcendental_penalty: 4,
        }
    }
}

impl HardwareConfig {
    /// `(hbm_bandwidth_tb_s * 1e12) / (clock_ghz * 1e9) / num_cores`.
    pub fn hbm_bytes_per_cycle_per_core(&self) -> f64 {
        (self.hbm_bandwidth_tb_s * 1e12) / (self.clock_ghz * 1e9) / self.num_cores as f64
    }

    /// `ring_bandwidth_tb_s * 1e12 / (clock_ghz * 1e9)`.
    pub fn ring_bytes_per_cycle(&self) -> f64 {
        self.ring_bandwidth_tb_s * 1e12 / (self.clock_ghz * 1e9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_roofline() {
        let c = HardwareConfig::default();
        assert_eq!(c.systolic_flops_per_cycle, 524288);
        // 1e12 / 1e9 / 32 = 1000/32
        assert!((c.hbm_bytes_per_cycle_per_core() - 1000.0 / 32.0).abs() < 1e-9);
        assert!((c.ring_bytes_per_cycle() - 4000.0).abs() < 1e-9);
    }

    #[test]
    fn category_strings_match_python() {
        assert_eq!(LatencyCategory::ComputeMatmul.as_str(), "compute_matmul");
        assert_eq!(LatencyCategory::Comm.as_str(), "comm");
    }
}
