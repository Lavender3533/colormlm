//! DeepSeek-V4 mHC/final-boundary CPU `f32` reference primitives.
//!
//! 这些函数刻意保持朴素、确定的标量运算顺序。它们不是生产优化核，而是后续
//! Vulkan shader 的数值 oracle。所有张量都使用紧密 row-major `f32` slice；
//! 调用者必须显式提供行数和 hidden 宽度，任何 shape 或非有限值都会 fail closed。

use std::error::Error;
use std::fmt;

pub const NATIVE_HC_STREAMS: usize = 4;
pub const NATIVE_HC_MIX_WIDTH: usize = 24;
pub const NATIVE_SINKHORN_ITERS: usize = 20;
pub const NATIVE_NORM_EPS: f32 = 1.0e-6;
pub const NATIVE_HC_EPS: f32 = 1.0e-6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativePrimitiveError {
    InvalidDimension {
        name: &'static str,
        value: usize,
    },
    Shape {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        name: &'static str,
        index: usize,
    },
    ArithmeticOverflow {
        name: &'static str,
    },
}

impl fmt::Display for NativePrimitiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { name, value } => {
                write!(formatter, "{name} 必须大于零，实际为 {value}")
            }
            Self::Shape {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} 元素数不匹配：期望 {expected}，实际 {actual}"
            ),
            Self::NonFinite { name, index } => {
                write!(formatter, "{name}[{index}] 含 NaN/Inf")
            }
            Self::ArithmeticOverflow { name } => write!(formatter, "{name} shape 乘法溢出"),
        }
    }
}

impl Error for NativePrimitiveError {}

#[derive(Debug, Clone, PartialEq)]
pub struct HcSplitOutput {
    /// `[rows, 4]`，对应 attention/FFN branch 的 reduce 系数。
    pub pre: Vec<f32>,
    /// `[rows, 4]`，对应 branch expand 系数。
    pub post: Vec<f32>,
    /// `[rows, 4, 4]`，布局为 `[row, residual_stream, output_stream]`。
    pub comb: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HcPreOutput {
    /// `[rows, hidden]` 的单流 branch 输入。
    pub reduced: Vec<f32>,
    /// `[rows, 4]`，供随后的 `hc_post` 使用。
    pub post: Vec<f32>,
    /// `[rows, 4, 4]`，供随后的 `hc_post` 使用。
    pub comb: Vec<f32>,
}

fn checked_len(
    left: usize,
    right: usize,
    name: &'static str,
) -> Result<usize, NativePrimitiveError> {
    left.checked_mul(right)
        .ok_or(NativePrimitiveError::ArithmeticOverflow { name })
}

fn require_dimension(name: &'static str, value: usize) -> Result<(), NativePrimitiveError> {
    if value == 0 {
        return Err(NativePrimitiveError::InvalidDimension { name, value });
    }
    Ok(())
}

fn require_shape(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), NativePrimitiveError> {
    if actual != expected {
        return Err(NativePrimitiveError::Shape {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), NativePrimitiveError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(NativePrimitiveError::NonFinite { name, index });
    }
    Ok(())
}

fn require_finite_value(
    name: &'static str,
    index: usize,
    value: f32,
) -> Result<f32, NativePrimitiveError> {
    if !value.is_finite() {
        return Err(NativePrimitiveError::NonFinite { name, index });
    }
    Ok(value)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// 拆分 `[rows,24]` HC logits，并复现固定四流、20 轮 Sinkhorn 顺序。
///
/// `mixes` 必须已经包含 Python `hc_pre` 中 projection 后的 RMS 比例；即本函数
/// 直接对应 `hc_split_sinkhorn`，不会再次做 projection 或 RMS 缩放。
pub fn hc_split_sinkhorn(
    mixes: &[f32],
    rows: usize,
    hc_scale: &[f32],
    hc_base: &[f32],
) -> Result<HcSplitOutput, NativePrimitiveError> {
    require_dimension("rows", rows)?;
    require_shape(
        "mixes",
        mixes.len(),
        checked_len(rows, NATIVE_HC_MIX_WIDTH, "mixes")?,
    )?;
    require_shape("hc_scale", hc_scale.len(), 3)?;
    require_shape("hc_base", hc_base.len(), NATIVE_HC_MIX_WIDTH)?;
    require_finite("mixes", mixes)?;
    require_finite("hc_scale", hc_scale)?;
    require_finite("hc_base", hc_base)?;

    let stream_values = checked_len(rows, NATIVE_HC_STREAMS, "HC stream output")?;
    let comb_values = checked_len(stream_values, NATIVE_HC_STREAMS, "HC comb output")?;
    let mut pre = vec![0.0f32; stream_values];
    let mut post = vec![0.0f32; stream_values];
    let mut comb = vec![0.0f32; comb_values];

    for row in 0..rows {
        let mix_offset = row * NATIVE_HC_MIX_WIDTH;
        let stream_offset = row * NATIVE_HC_STREAMS;
        let comb_offset = row * NATIVE_HC_STREAMS * NATIVE_HC_STREAMS;

        for stream in 0..NATIVE_HC_STREAMS {
            let pre_logit = require_finite_value(
                "pre_logits",
                stream_offset + stream,
                mixes[mix_offset + stream] * hc_scale[0] + hc_base[stream],
            )?;
            pre[stream_offset + stream] = require_finite_value(
                "pre",
                stream_offset + stream,
                sigmoid(pre_logit) + NATIVE_HC_EPS,
            )?;

            let post_logit = require_finite_value(
                "post_logits",
                stream_offset + stream,
                mixes[mix_offset + NATIVE_HC_STREAMS + stream] * hc_scale[1]
                    + hc_base[NATIVE_HC_STREAMS + stream],
            )?;
            post[stream_offset + stream] =
                require_finite_value("post", stream_offset + stream, 2.0 * sigmoid(post_logit))?;
        }

        for input_stream in 0..NATIVE_HC_STREAMS {
            for output_stream in 0..NATIVE_HC_STREAMS {
                let local = input_stream * NATIVE_HC_STREAMS + output_stream;
                let logit = mixes[mix_offset + 2 * NATIVE_HC_STREAMS + local] * hc_scale[2]
                    + hc_base[2 * NATIVE_HC_STREAMS + local];
                comb[comb_offset + local] =
                    require_finite_value("comb_logits", comb_offset + local, logit)?;
            }
        }

        // Python 首轮先做稳定 row-softmax，再加 eps；这一轮与后续普通
        // row normalization 不可合并，否则会改变官方运算顺序。
        for input_stream in 0..NATIVE_HC_STREAMS {
            let row_start = comb_offset + input_stream * NATIVE_HC_STREAMS;
            let row_end = row_start + NATIVE_HC_STREAMS;
            let row_max = comb[row_start..row_end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0f32;
            for value in &mut comb[row_start..row_end] {
                *value = (*value - row_max).exp();
                denominator += *value;
            }
            require_finite_value("comb_row_softmax_sum", input_stream, denominator)?;
            for value in &mut comb[row_start..row_end] {
                *value = *value / denominator + NATIVE_HC_EPS;
            }
        }
        normalize_comb_columns(&mut comb[comb_offset..comb_offset + 16])?;

        for _ in 1..NATIVE_SINKHORN_ITERS {
            normalize_comb_rows(&mut comb[comb_offset..comb_offset + 16])?;
            normalize_comb_columns(&mut comb[comb_offset..comb_offset + 16])?;
        }
    }

    require_finite("pre", &pre)?;
    require_finite("post", &post)?;
    require_finite("comb", &comb)?;
    Ok(HcSplitOutput { pre, post, comb })
}

fn normalize_comb_rows(comb: &mut [f32]) -> Result<(), NativePrimitiveError> {
    debug_assert_eq!(comb.len(), 16);
    for input_stream in 0..NATIVE_HC_STREAMS {
        let start = input_stream * NATIVE_HC_STREAMS;
        let denominator = comb[start..start + NATIVE_HC_STREAMS]
            .iter()
            .copied()
            .sum::<f32>()
            + NATIVE_HC_EPS;
        require_finite_value("comb_row_sum", input_stream, denominator)?;
        for value in &mut comb[start..start + NATIVE_HC_STREAMS] {
            *value /= denominator;
        }
    }
    Ok(())
}

fn normalize_comb_columns(comb: &mut [f32]) -> Result<(), NativePrimitiveError> {
    debug_assert_eq!(comb.len(), 16);
    let mut denominators = [0.0f32; NATIVE_HC_STREAMS];
    for output_stream in 0..NATIVE_HC_STREAMS {
        for input_stream in 0..NATIVE_HC_STREAMS {
            denominators[output_stream] += comb[input_stream * NATIVE_HC_STREAMS + output_stream];
        }
        denominators[output_stream] += NATIVE_HC_EPS;
        require_finite_value(
            "comb_column_sum",
            output_stream,
            denominators[output_stream],
        )?;
    }
    for input_stream in 0..NATIVE_HC_STREAMS {
        for output_stream in 0..NATIVE_HC_STREAMS {
            comb[input_stream * NATIVE_HC_STREAMS + output_stream] /= denominators[output_stream];
        }
    }
    Ok(())
}

/// 对 `[rows,4,hidden]` mHC streams 执行官方 HC-pre reduction。
///
/// `projected_mixes` 是 `F.linear(x.flatten(2), hc_fn)` 的 `[rows,24]` F32
/// 结果；本函数只补上 Python 参考中的 RMS 比例、split/Sinkhorn 与 reduction。
pub fn hc_pre_from_projection(
    streams: &[f32],
    rows: usize,
    hidden: usize,
    projected_mixes: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
) -> Result<HcPreOutput, NativePrimitiveError> {
    require_dimension("rows", rows)?;
    require_dimension("hidden", hidden)?;
    let streams_per_row = checked_len(NATIVE_HC_STREAMS, hidden, "HC streams_per_row")?;
    require_shape(
        "streams",
        streams.len(),
        checked_len(rows, streams_per_row, "HC streams")?,
    )?;
    require_shape(
        "projected_mixes",
        projected_mixes.len(),
        checked_len(rows, NATIVE_HC_MIX_WIDTH, "HC projected_mixes")?,
    )?;
    require_finite("streams", streams)?;
    require_finite("projected_mixes", projected_mixes)?;

    let mut scaled_mixes = vec![0.0f32; projected_mixes.len()];
    for row in 0..rows {
        let stream_offset = row * streams_per_row;
        let mut sum_squares = 0.0f32;
        for value in &streams[stream_offset..stream_offset + streams_per_row] {
            sum_squares += *value * *value;
        }
        let mean_square = require_finite_value(
            "hc_pre_mean_square",
            row,
            sum_squares / streams_per_row as f32,
        )?;
        let inverse_rms = require_finite_value(
            "hc_pre_inverse_rms",
            row,
            (mean_square + NATIVE_NORM_EPS).sqrt().recip(),
        )?;
        let mix_offset = row * NATIVE_HC_MIX_WIDTH;
        for index in 0..NATIVE_HC_MIX_WIDTH {
            scaled_mixes[mix_offset + index] = require_finite_value(
                "hc_pre_scaled_mixes",
                mix_offset + index,
                projected_mixes[mix_offset + index] * inverse_rms,
            )?;
        }
    }

    let split = hc_split_sinkhorn(&scaled_mixes, rows, hc_scale, hc_base)?;
    let mut reduced = vec![0.0f32; checked_len(rows, hidden, "HC reduced")?];
    for row in 0..rows {
        for dimension in 0..hidden {
            let mut sum = 0.0f32;
            for stream in 0..NATIVE_HC_STREAMS {
                let coefficient = split.pre[row * NATIVE_HC_STREAMS + stream];
                let value = streams[(row * NATIVE_HC_STREAMS + stream) * hidden + dimension];
                sum += coefficient * value;
            }
            reduced[row * hidden + dimension] =
                require_finite_value("hc_pre_reduced", row * hidden + dimension, sum)?;
        }
    }
    Ok(HcPreOutput {
        reduced,
        post: split.post,
        comb: split.comb,
    })
}

/// 复现官方 HC-post：单流 branch 与四流 residual 合并成 `[rows,4,hidden]`。
pub fn hc_post(
    branch: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    rows: usize,
    hidden: usize,
) -> Result<Vec<f32>, NativePrimitiveError> {
    require_dimension("rows", rows)?;
    require_dimension("hidden", hidden)?;
    let branch_len = checked_len(rows, hidden, "HC branch")?;
    let stream_len = checked_len(
        checked_len(rows, NATIVE_HC_STREAMS, "HC stream rows")?,
        hidden,
        "HC residual",
    )?;
    require_shape("branch", branch.len(), branch_len)?;
    require_shape("residual", residual.len(), stream_len)?;
    require_shape(
        "post",
        post.len(),
        checked_len(rows, NATIVE_HC_STREAMS, "HC post")?,
    )?;
    require_shape("comb", comb.len(), checked_len(rows, 16, "HC comb")?)?;
    require_finite("branch", branch)?;
    require_finite("residual", residual)?;
    require_finite("post", post)?;
    require_finite("comb", comb)?;

    let mut merged = vec![0.0f32; stream_len];
    for row in 0..rows {
        for output_stream in 0..NATIVE_HC_STREAMS {
            for dimension in 0..hidden {
                let mut residual_sum = 0.0f32;
                for input_stream in 0..NATIVE_HC_STREAMS {
                    let coefficient =
                        comb[row * 16 + input_stream * NATIVE_HC_STREAMS + output_stream];
                    let value =
                        residual[(row * NATIVE_HC_STREAMS + input_stream) * hidden + dimension];
                    residual_sum += coefficient * value;
                }
                let branch_value = post[row * NATIVE_HC_STREAMS + output_stream]
                    * branch[row * hidden + dimension];
                let output_index = (row * NATIVE_HC_STREAMS + output_stream) * hidden + dimension;
                merged[output_index] = require_finite_value(
                    "hc_post_merged",
                    output_index,
                    branch_value + residual_sum,
                )?;
            }
        }
    }
    Ok(merged)
}

/// 官方 RMSNorm 的 F32 参考：F32 mean-square、`eps=1e-6`、先归一化再乘权重。
pub fn official_rms_norm(
    input: &[f32],
    rows: usize,
    hidden: usize,
    weight: &[f32],
) -> Result<Vec<f32>, NativePrimitiveError> {
    require_dimension("rows", rows)?;
    require_dimension("hidden", hidden)?;
    require_shape(
        "input",
        input.len(),
        checked_len(rows, hidden, "RMSNorm input")?,
    )?;
    require_shape("weight", weight.len(), hidden)?;
    require_finite("input", input)?;
    require_finite("weight", weight)?;

    let mut output = vec![0.0f32; input.len()];
    for row in 0..rows {
        let offset = row * hidden;
        let mut sum_squares = 0.0f32;
        for value in &input[offset..offset + hidden] {
            sum_squares += *value * *value;
        }
        let mean_square =
            require_finite_value("rms_norm_mean_square", row, sum_squares / hidden as f32)?;
        let inverse_rms = require_finite_value(
            "rms_norm_inverse_rms",
            row,
            (mean_square + NATIVE_NORM_EPS).sqrt().recip(),
        )?;
        for dimension in 0..hidden {
            let normalized = input[offset + dimension] * inverse_rms;
            output[offset + dimension] = require_finite_value(
                "rms_norm_output",
                offset + dimension,
                weight[dimension] * normalized,
            )?;
        }
    }
    Ok(output)
}

/// 按 round-to-nearest-even 做一次 F32 → BF16 → F32，拒绝 NaN/Inf。
pub fn bf16_round_trip(value: f32) -> Result<f32, NativePrimitiveError> {
    require_finite_value("bf16_round_trip_input", 0, value)?;
    let bits = value.to_bits();
    let rounding_bias = 0x7fffu32 + ((bits >> 16) & 1);
    let bf16 = bits.wrapping_add(rounding_bias) >> 16;
    let output = f32::from_bits(bf16 << 16);
    require_finite_value("bf16_round_trip_output", 0, output)
}

/// `bf16_round_trip` 的严格 slice 版本；错误 index 指向原始输入位置。
pub fn bf16_round_trip_slice(values: &[f32]) -> Result<Vec<f32>, NativePrimitiveError> {
    require_finite("bf16_round_trip_input", values)?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let bits = value.to_bits();
            let rounding_bias = 0x7fffu32 + ((bits >> 16) & 1);
            let output = f32::from_bits((bits.wrapping_add(rounding_bias) >> 16) << 16);
            require_finite_value("bf16_round_trip_output", index, output)
        })
        .collect()
}

#[cfg(test)]
mod native_primitives_tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "actual={actual:.9e}, expected={expected:.9e}, error={error:.9e}, tolerance={tolerance:.9e}"
        );
    }

    #[test]
    fn native_primitives_split_sinkhorn_is_deterministic_and_balanced() {
        let mixes: Vec<f32> = (0..NATIVE_HC_MIX_WIDTH)
            .map(|index| (index as f32 - 11.5) * 0.125)
            .collect();
        let scale = [0.7, -0.3, 0.2];
        let base: Vec<f32> = (0..NATIVE_HC_MIX_WIDTH)
            .map(|index| (index as f32 % 5.0 - 2.0) * 0.03125)
            .collect();

        let first = hc_split_sinkhorn(&mixes, 1, &scale, &base).unwrap();
        let second = hc_split_sinkhorn(&mixes, 1, &scale, &base).unwrap();
        assert_eq!(first, second);
        // 由 local_s14_primitives/hc.py 的 torch.float32 路径生成并冻结。
        let python_pre = [0.255_641_88, 0.278_885_84, 0.303_381_83, 0.329_047_68];
        let python_post = [1.170_202_3, 1.090_377_7, 1.087_277_4, 1.084_175_3];
        let python_comb = [
            0.262_246_52,
            0.262_246_52,
            0.233_156_19,
            0.242_349_79,
            0.252_298_18,
            0.252_298_18,
            0.262_246_52,
            0.233_156_23,
            0.242_727_18,
            0.242_727_19,
            0.252_298_18,
            0.262_246_49,
            0.242_727_18,
            0.242_727_19,
            0.252_298_18,
            0.262_246_49,
        ];
        for (actual, expected) in first.pre.iter().zip(python_pre) {
            assert_close(*actual, expected, 2.0e-7);
        }
        for (actual, expected) in first.post.iter().zip(python_post) {
            assert_close(*actual, expected, 2.0e-7);
        }
        for (actual, expected) in first.comb.iter().zip(python_comb) {
            assert_close(*actual, expected, 3.0e-7);
        }
        assert!(first
            .pre
            .iter()
            .all(|value| *value > NATIVE_HC_EPS && *value < 1.1));
        assert!(first.post.iter().all(|value| *value > 0.0 && *value < 2.0));
        for input_stream in 0..NATIVE_HC_STREAMS {
            let sum: f32 = first.comb
                [input_stream * NATIVE_HC_STREAMS..(input_stream + 1) * NATIVE_HC_STREAMS]
                .iter()
                .sum();
            assert_close(sum, 1.0, 3.0e-6);
        }
        for output_stream in 0..NATIVE_HC_STREAMS {
            let sum: f32 = (0..NATIVE_HC_STREAMS)
                .map(|input_stream| first.comb[input_stream * NATIVE_HC_STREAMS + output_stream])
                .sum();
            assert_close(sum, 1.0, 3.0e-6);
        }
    }

    #[test]
    fn native_primitives_hc_pre_zero_projection_reduces_four_streams() {
        let streams = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let projected = [0.0f32; NATIVE_HC_MIX_WIDTH];
        let scale = [1.0f32; 3];
        let base = [0.0f32; NATIVE_HC_MIX_WIDTH];
        let output = hc_pre_from_projection(&streams, 1, 2, &projected, &scale, &base).unwrap();
        let coefficient = 0.5 + NATIVE_HC_EPS;
        assert_close(output.reduced[0], coefficient * 16.0, 2.0e-6);
        assert_close(output.reduced[1], coefficient * 20.0, 2.0e-6);
        assert_eq!(output.post, vec![1.0; NATIVE_HC_STREAMS]);
    }

    #[test]
    fn native_primitives_hc_post_uses_input_output_comb_layout() {
        let branch = [10.0, 20.0];
        let residual = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let post = [1.0, 2.0, 3.0, 4.0];
        let mut comb = [0.0f32; 16];
        // output stream 0 receives residual stream 3; output stream 1 receives 2, etc.
        for input_stream in 0..NATIVE_HC_STREAMS {
            let output_stream = NATIVE_HC_STREAMS - 1 - input_stream;
            comb[input_stream * NATIVE_HC_STREAMS + output_stream] = 1.0;
        }
        let merged = hc_post(&branch, &residual, &post, &comb, 1, 2).unwrap();
        assert_eq!(merged, vec![17.0, 28.0, 25.0, 46.0, 33.0, 64.0, 41.0, 82.0]);
    }

    #[test]
    fn native_primitives_official_rms_norm_matches_scalar_formula() {
        let input = [1.0, -2.0, 3.0, -4.0, 0.5, -0.25, 0.125, -0.0625];
        let weight = [0.5, 1.0, 1.5, 2.0];
        let output = official_rms_norm(&input, 2, 4, &weight).unwrap();
        for row in 0..2 {
            let values = &input[row * 4..row * 4 + 4];
            let mean_square = values.iter().map(|value| value * value).sum::<f32>() / 4.0;
            let inverse_rms = (mean_square + NATIVE_NORM_EPS).sqrt().recip();
            for dimension in 0..4 {
                assert_eq!(
                    output[row * 4 + dimension].to_bits(),
                    (weight[dimension] * (values[dimension] * inverse_rms)).to_bits()
                );
            }
        }
    }

    #[test]
    fn native_primitives_bf16_round_trip_is_ties_to_even() {
        // 1.0 与下一 BF16 之间的正中点，低位偶数，向 1.0 舍入。
        assert_eq!(
            bf16_round_trip(1.003_906_25).unwrap().to_bits(),
            1.0f32.to_bits()
        );
        // 0x3f81 与 0x3f82 之间的正中点，0x3f81 低位为奇数，向 0x3f82 舍入。
        assert_eq!(
            bf16_round_trip(1.011_718_75).unwrap().to_bits(),
            1.015_625f32.to_bits()
        );
        assert_eq!(
            bf16_round_trip_slice(&[-0.0, 1.0, -2.0]).unwrap(),
            vec![-0.0, 1.0, -2.0]
        );
    }

    #[test]
    fn native_primitives_reject_shape_and_non_finite_inputs() {
        let shape_error = official_rms_norm(&[1.0, 2.0], 1, 3, &[1.0; 3]).unwrap_err();
        assert!(matches!(
            shape_error,
            NativePrimitiveError::Shape {
                name: "input",
                expected: 3,
                actual: 2
            }
        ));

        let finite_error = hc_post(
            &[1.0],
            &[0.0, 0.0, f32::NAN, 0.0],
            &[1.0; 4],
            &[0.25; 16],
            1,
            1,
        )
        .unwrap_err();
        assert_eq!(
            finite_error,
            NativePrimitiveError::NonFinite {
                name: "residual",
                index: 2
            }
        );
        assert!(bf16_round_trip(f32::INFINITY).is_err());
    }
}
