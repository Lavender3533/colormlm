//! L1/L2 `wo_b` 的 OpenBLAS/BF16 midpoint 归约顺序审计。
//!
//! 该程序只读取冻结的 position0 参考输入与真实 FP8 payload，在 CPU 上使用
//! 与 Vulkan exact shader 相同的 8 条 strided FMA lane。它枚举最终 8-lane
//! 顺序，寻找同时满足 L1/L2 判别行 BF16-RNE 结果的候选；不接入生产图。

use anyhow::{bail, Context, Result};
use polaris_s14_runner::{Position0Asset, Position0WholeTokenManifest};
use serde::Serialize;
use std::{env, fs, path::Path, path::PathBuf};

const N: usize = 4096;
const K: usize = 8192;
const LANES: usize = 8;
const CURRENT_ORDER: [usize; LANES] = [2, 3, 4, 5, 6, 0, 7, 1];
const LEGACY_WIDE_ORDER: [usize; LANES] = [0, 1, 3, 4, 6, 2, 5, 7];

#[derive(Debug, Serialize)]
struct CaseEvidence {
    layer: usize,
    row: usize,
    expected_bf16: String,
    openblas_raw_f32_bits: String,
    partial_f32_bits: Vec<String>,
    current_order_f32_bits: String,
    current_order_bf16: String,
    legacy_order_f32_bits: String,
    legacy_order_bf16: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    format: &'static str,
    reduction_model: &'static str,
    cases: Vec<CaseEvidence>,
    bf16_satisfying_order_count: usize,
    bf16_satisfying_orders_first_64: Vec<[usize; LANES]>,
    raw_f32_satisfying_order_count: usize,
    raw_f32_satisfying_orders_first_64: Vec<[usize; LANES]>,
    claim_limit: &'static str,
}

struct Case {
    layer: usize,
    row: usize,
    expected: u16,
    openblas_raw: f32,
    partial: [f32; LANES],
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let payload = fs::read(path).with_context(|| format!("读取 {}", path.display()))?;
    if payload.len() % 4 != 0 {
        bail!("{} 不是完整 F32LE", path.display());
    }
    Ok(payload
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("4-byte chunk")))
        .collect())
}

fn read_u16(path: &Path) -> Result<Vec<u16>> {
    let payload = fs::read(path).with_context(|| format!("读取 {}", path.display()))?;
    if payload.len() % 2 != 0 {
        bail!("{} 不是完整 U16LE", path.display());
    }
    Ok(payload
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("2-byte chunk")))
        .collect())
}

fn asset<'a>(
    manifest: &'a Position0WholeTokenManifest,
    layer: usize,
    name: &str,
) -> Result<&'a Position0Asset> {
    manifest
        .layers
        .get(layer)
        .with_context(|| format!("manifest 缺少 L{layer}"))?
        .assets
        .non_expert
        .iter()
        .find(|entry| entry.tensor == name)
        .with_context(|| format!("manifest 缺少 {name}"))
}

fn e4m3fn(code: u8) -> f32 {
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 7;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * 0.001_953_125
    } else if exponent == 15 && mantissa == 7 {
        f32::NAN
    } else {
        (1.0 + f32::from(mantissa) * 0.125) * 2.0f32.powi(i32::from(exponent) - 7)
    };
    if code & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn ue8m0(code: u8) -> f32 {
    2.0f32.powi(i32::from(code) - 127)
}

fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let bias = 0x7fffu32 + ((bits >> 16) & 1);
    ((bits.wrapping_add(bias)) >> 16) as u16
}

fn reduce(partial: &[f32; LANES], order: &[usize; LANES]) -> f32 {
    let mut total = partial[order[0]];
    for &lane in &order[1..] {
        total += partial[lane];
    }
    total
}

fn build_case(
    repo_root: &Path,
    manifest: &Position0WholeTokenManifest,
    layer: usize,
    row: usize,
) -> Result<Case> {
    if row >= N {
        bail!("L{layer} row {row} 越界");
    }
    let reference_root = repo_root.join(format!(
        ".tmp-polaris-runs/l{layer}-stage-reference-20260802"
    ));
    let input = read_f32(&reference_root.join("wo_a_qdq.f32le.bin"))?;
    let expected = read_u16(&reference_root.join("attention_branch.bf16le.bin"))?;
    let openblas_raw = read_f32(&reference_root.join("wo_b_openblas_raw.f32le.bin"))?;
    if input.len() != K || expected.len() != N || openblas_raw.len() != N {
        bail!("L{layer} 冻结参考 shape 漂移");
    }
    let weight_asset = asset(manifest, layer, &format!("layers.{layer}.attn.wo_b.weight"))?;
    let scale_asset = asset(manifest, layer, &format!("layers.{layer}.attn.wo_b.scale"))?;
    let weight = fs::read(&weight_asset.path)?;
    let scale = fs::read(&scale_asset.path)?;
    if weight.len() != N * K || scale.len() != (N / 128) * (K / 128) {
        bail!("L{layer} wo_b payload shape 漂移");
    }
    let mut partial = [0.0f32; LANES];
    let row_base = row * K;
    let scale_row_base = (row / 128) * (K / 128);
    for lane in 0..LANES {
        let mut acc = 0.0f32;
        for k in (lane..K).step_by(LANES) {
            let decoded = e4m3fn(weight[row_base + k]);
            let factor = ue8m0(scale[scale_row_base + k / 128]);
            if !decoded.is_finite() || !factor.is_finite() {
                bail!("L{layer} row{row} 含非法 FP8/UE8M0 code");
            }
            acc = (decoded * factor).mul_add(input[k], acc);
        }
        partial[lane] = acc;
    }
    Ok(Case {
        layer,
        row,
        expected: expected[row],
        openblas_raw: openblas_raw[row],
        partial,
    })
}

fn enumerate(
    prefix: &mut Vec<usize>,
    used: &mut [bool; LANES],
    cases: &[Case],
    bf16_count: &mut usize,
    bf16_first: &mut Vec<[usize; LANES]>,
    raw_count: &mut usize,
    raw_first: &mut Vec<[usize; LANES]>,
) {
    if prefix.len() == LANES {
        let order: [usize; LANES] = prefix.as_slice().try_into().expect("8 lanes");
        if cases
            .iter()
            .all(|case| bf16_rne(reduce(&case.partial, &order)) == case.expected)
        {
            *bf16_count += 1;
            if bf16_first.len() < 64 {
                bf16_first.push(order);
            }
        }
        if cases
            .iter()
            .all(|case| reduce(&case.partial, &order).to_bits() == case.openblas_raw.to_bits())
        {
            *raw_count += 1;
            if raw_first.len() < 64 {
                raw_first.push(order);
            }
        }
        return;
    }
    for lane in 0..LANES {
        if !used[lane] {
            used[lane] = true;
            prefix.push(lane);
            enumerate(
                prefix, used, cases, bf16_count, bf16_first, raw_count, raw_first,
            );
            prefix.pop();
            used[lane] = false;
        }
    }
}

fn main() -> Result<()> {
    let repo_root = env::var_os("POLARIS_REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
        .canonicalize()?;
    let manifest_path = repo_root.join(
        "fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let cases = vec![
        build_case(&repo_root, &manifest, 1, 3364)?,
        build_case(&repo_root, &manifest, 2, 2609)?,
    ];
    let evidence = cases
        .iter()
        .map(|case| CaseEvidence {
            layer: case.layer,
            row: case.row,
            expected_bf16: format!("0x{:04x}", case.expected),
            openblas_raw_f32_bits: format!("0x{:08x}", case.openblas_raw.to_bits()),
            partial_f32_bits: case
                .partial
                .iter()
                .map(|value| format!("0x{:08x}", value.to_bits()))
                .collect(),
            current_order_f32_bits: format!(
                "0x{:08x}",
                reduce(&case.partial, &CURRENT_ORDER).to_bits()
            ),
            current_order_bf16: format!(
                "0x{:04x}",
                bf16_rne(reduce(&case.partial, &CURRENT_ORDER))
            ),
            legacy_order_f32_bits: format!(
                "0x{:08x}",
                reduce(&case.partial, &LEGACY_WIDE_ORDER).to_bits()
            ),
            legacy_order_bf16: format!(
                "0x{:04x}",
                bf16_rne(reduce(&case.partial, &LEGACY_WIDE_ORDER))
            ),
        })
        .collect();
    let mut bf16_count = 0usize;
    let mut bf16_first = Vec::new();
    let mut raw_count = 0usize;
    let mut raw_first = Vec::new();
    enumerate(
        &mut Vec::with_capacity(LANES),
        &mut [false; LANES],
        &cases,
        &mut bf16_count,
        &mut bf16_first,
        &mut raw_count,
        &mut raw_first,
    );
    let report = AuditReport {
        format: "polaris-s14-fp8-wo-b-midpoint-audit-v1",
        reduction_model: "8 strided FMA lanes followed by sequential lane-order sum",
        cases: evidence,
        bf16_satisfying_order_count: bf16_count,
        bf16_satisfying_orders_first_64: bf16_first,
        raw_f32_satisfying_order_count: raw_count,
        raw_f32_satisfying_orders_first_64: raw_first,
        claim_limit: "audit-only two discriminating rows; a candidate order must still pass full L1/L2 Vulkan BF16 gates",
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
