// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! dtype <-> byte codecs for the HBM/LX boundary. Tile data is flat `Vec<f32>`
//! (see `tile.rs`); these encode/decode it to the raw bytes the memory
//! simulator stores, per the source dtype. `f16` uses an inline IEEE
//! half-precision round-trip (round-to-nearest-even) — no `half` dependency.
//!
//! (Note: `arith.rs` and `ops_memory.rs` currently carry their own private
//! copies of the f16 round-trip; consolidating them onto this module is a
//! follow-up cleanup.)

use crate::dtypes::DType;

/// Decode IEEE-754 half-precision bits to f32.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let val: f32 = if exp == 0 {
        // subnormal / zero
        (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -val } else { val }
}

/// Encode f32 to IEEE-754 half-precision bits (round-to-nearest-even).
pub fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7fffff;
    if f.is_nan() {
        return sign | 0x7e00;
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if exp <= 0 {
        // subnormal / underflow
        if exp < -10 {
            return sign;
        }
        let mant_full = mant | 0x800000;
        let shift = (14 - exp) as u32;
        let mut half_mant = mant_full >> shift;
        // round-to-nearest-even
        let rem = mant_full & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half_mant & 1) == 1) {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }
    let mut half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1) {
        half_mant += 1;
        // carry into exponent handled naturally by the +1 spilling into exp bits
    }
    sign | ((exp as u16) << 10) | half_mant
}

/// Encode a flat f32 tile into raw bytes for memory, per `dtype`.
pub fn encode(data: &[f32], dtype: DType) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * dtype.bytes_per_elem());
    for &v in data {
        match dtype {
            DType::F16 => out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes()),
            DType::F32 => out.extend_from_slice(&v.to_le_bytes()),
            DType::I32 => out.extend_from_slice(&(v as i32).to_le_bytes()),
            DType::I64 => out.extend_from_slice(&(v as i64).to_le_bytes()),
            DType::Bool => out.push((v != 0.0) as u8),
        }
    }
    out
}

/// Decode `n` elements of `dtype` from raw bytes into a flat f32 tile.
/// Zero-pads if `bytes` is short (matches the memory sim's zero-fill).
pub fn decode(bytes: &[u8], n: usize, dtype: DType) -> Vec<f32> {
    let bpe = dtype.bytes_per_elem();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * bpe;
        let chunk = bytes.get(off..off + bpe);
        let v = match (dtype, chunk) {
            (_, None) => 0.0,
            (DType::F16, Some(c)) => f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])),
            (DType::F32, Some(c)) => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            (DType::I32, Some(c)) => i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32,
            (DType::I64, Some(c)) => {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            }
            (DType::Bool, Some(c)) => (c[0] != 0) as i32 as f32,
        };
        out.push(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_exact_for_small_ints() {
        for v in [0.0f32, 1.0, -1.0, 2.5, 128.0, -0.5, 1024.0] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            assert_eq!(back, v, "f16 round-trip for {v}");
        }
    }

    #[test]
    fn f16_inf_encoding() {
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f16_bits_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(0xfc00), f32::NEG_INFINITY);
    }

    #[test]
    fn encode_decode_roundtrips_each_dtype() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        for dt in [DType::F16, DType::F32, DType::I32, DType::I64] {
            let bytes = encode(&data, dt);
            assert_eq!(bytes.len(), 4 * dt.bytes_per_elem());
            assert_eq!(decode(&bytes, 4, dt), data, "round-trip {dt}");
        }
    }

    #[test]
    fn decode_zero_pads_short_input() {
        assert_eq!(decode(&[], 3, DType::F32), vec![0.0, 0.0, 0.0]);
    }
}
