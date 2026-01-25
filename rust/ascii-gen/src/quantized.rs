//! Helper utilities for loading weight-only quantized exports.
//!
//! The Python export path supports weight-only quantization for linear weights:
//! - INT8 symmetric per-row: `{name}.int_data` (int8) + `{name}.scale` (float32, per-row)
//! - INT4 symmetric per-row (packed): `{name}.int_data` (uint8 packed u4/u4) + `{name}.scale`
//!
//! These helpers focus on *dequantize-on-load* (turn quantized weights into float32 tensors),
//! so the rest of the Candle model can remain unchanged.

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};

/// Dequantize a symmetric per-row INT8 weight matrix to float32.
///
/// - `int_data` is row-major with shape `[out_features, in_features]`.
/// - `scales` is per-row with shape `[out_features]`.
///
/// Returns a row-major float32 buffer with the same shape.
///
/// # Errors
/// Returns an error if the provided slices do not match the expected shapes, or if shape
/// computations overflow.
pub fn dequantize_int8_per_row(
    int_data: &[i8],
    out_features: usize,
    in_features: usize,
    scales: &[f32],
) -> Result<Vec<f32>> {
    let expected_len = out_features
        .checked_mul(in_features)
        .context("out_features * in_features overflow")?;

    if int_data.len() != expected_len {
        bail!(
            "int8 int_data length mismatch: expected {expected_len}, got {}",
            int_data.len()
        );
    }
    if scales.len() != out_features {
        bail!(
            "int8 scale length mismatch: expected {out_features}, got {}",
            scales.len()
        );
    }

    let mut out = vec![0.0f32; expected_len];
    for (row, &scale) in scales.iter().enumerate() {
        let row_start = row * in_features;
        for col in 0..in_features {
            let idx = row_start + col;
            out[idx] = f32::from(int_data[idx]) * scale;
        }
    }

    Ok(out)
}

/// Dequantize a symmetric per-row INT4 (packed) weight matrix to float32.
///
/// The INT4 export stores two unsigned 4-bit values (0..15) per byte, where each nibble is
/// `q + 8` for signed `q` in `[-8, 7]`.
///
/// - `packed` is row-major with shape `[out_features, packed_in_features]`
///   where `packed_in_features = ceil(orig_in_features / 2)`.
/// - `orig_in_features` is the original `in_features` before padding to even.
/// - `scales` is per-row with shape `[out_features]`.
///
/// Returns a row-major float32 buffer with shape `[out_features, orig_in_features]`.
///
/// # Errors
/// Returns an error if the provided slices do not match the expected shapes, or if shape
/// computations overflow.
pub fn dequantize_int4_packed_per_row(
    packed: &[u8],
    out_features: usize,
    packed_in_features: usize,
    orig_in_features: usize,
    scales: &[f32],
) -> Result<Vec<f32>> {
    if scales.len() != out_features {
        bail!(
            "int4 scale length mismatch: expected {out_features}, got {}",
            scales.len()
        );
    }

    let expected_packed_len = out_features
        .checked_mul(packed_in_features)
        .context("out_features * packed_in_features overflow")?;
    if packed.len() != expected_packed_len {
        bail!(
            "int4 int_data length mismatch: expected {expected_packed_len}, got {}",
            packed.len()
        );
    }

    let expected_packed_cols = orig_in_features
        .checked_add(1)
        .context("orig_in_features + 1 overflow")?
        / 2;
    if packed_in_features != expected_packed_cols {
        bail!(
            "int4 packed_in_features mismatch: expected {expected_packed_cols}, got {packed_in_features}"
        );
    }

    let out_len = out_features
        .checked_mul(orig_in_features)
        .context("out_features * orig_in_features overflow")?;
    let mut out = vec![0.0f32; out_len];

    for (row, &scale) in scales.iter().enumerate() {
        let out_row_start = row * orig_in_features;
        let packed_row_start = row * packed_in_features;

        for packed_col in 0..packed_in_features {
            let byte = packed[packed_row_start + packed_col];
            let lo = byte & 0x0f;
            let hi = byte >> 4;

            let base_col = packed_col * 2;
            if base_col < orig_in_features {
                let q = i16::from(lo) - 8;
                out[out_row_start + base_col] = f32::from(q) * scale;
            }

            let col1 = base_col + 1;
            if col1 < orig_in_features {
                let q = i16::from(hi) - 8;
                out[out_row_start + col1] = f32::from(q) * scale;
            }
        }
    }

    Ok(out)
}

/// Convenience: dequantize INT8 and immediately build a Candle tensor on `device`.
///
/// # Errors
/// Returns an error if the input shapes are inconsistent or if tensor creation fails.
pub fn dequantize_int8_tensor(
    int_data: &[i8],
    out_features: usize,
    in_features: usize,
    scales: &[f32],
    device: &Device,
) -> Result<Tensor> {
    let deq = dequantize_int8_per_row(int_data, out_features, in_features, scales)?;
    Tensor::from_vec(deq, (out_features, in_features), device)
        .context("create dequantized int8 tensor")
}

/// Convenience: dequantize packed INT4 and immediately build a Candle tensor on `device`.
///
/// # Errors
/// Returns an error if the input shapes are inconsistent or if tensor creation fails.
pub fn dequantize_int4_tensor(
    packed: &[u8],
    out_features: usize,
    packed_in_features: usize,
    orig_in_features: usize,
    scales: &[f32],
    device: &Device,
) -> Result<Tensor> {
    let deq = dequantize_int4_packed_per_row(
        packed,
        out_features,
        packed_in_features,
        orig_in_features,
        scales,
    )?;
    Tensor::from_vec(deq, (out_features, orig_in_features), device)
        .context("create dequantized int4 tensor")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantize_int8_matches_expected() {
        let int_data: [i8; 6] = [1, -2, 3, 4, 0, -1];
        let scales: [f32; 2] = [0.5, 2.0];
        let out = dequantize_int8_per_row(&int_data, 2, 3, &scales).unwrap();
        assert_eq!(out, vec![0.5, -1.0, 1.5, 8.0, 0.0, -2.0]);
    }

    #[test]
    fn dequantize_int4_even_in_features_matches_expected() {
        // out=1, in=4 => packed_in=2
        // q values: [-8, -1, 0, 7], scale=0.5
        let packed: [u8; 2] = [0x70, 0xF8];
        let scales: [f32; 1] = [0.5];
        let out = dequantize_int4_packed_per_row(&packed, 1, 2, 4, &scales).unwrap();
        assert_eq!(out, vec![-4.0, -0.5, 0.0, 3.5]);
    }

    #[test]
    fn dequantize_int4_odd_in_features_drops_padding() {
        // out=1, orig_in=3 => packed_in=2 (pads one nibble with q=0)
        // q values: [1, 2, 3], pad=0. scale=0.25.
        let packed: [u8; 2] = [0xA9, 0x8B];
        let scales: [f32; 1] = [0.25];
        let out = dequantize_int4_packed_per_row(&packed, 1, 2, 3, &scales).unwrap();
        assert_eq!(out, vec![0.25, 0.5, 0.75]);
    }

    #[test]
    fn dequantize_int4_tensor_builds_expected_shape() {
        let packed: [u8; 2] = [0xA9, 0x8B];
        let scales: [f32; 1] = [0.25];
        let device = Device::Cpu;
        let t = dequantize_int4_tensor(&packed, 1, 2, 3, &scales, &device).unwrap();
        assert_eq!(t.dims(), &[1, 3]);
    }
}
