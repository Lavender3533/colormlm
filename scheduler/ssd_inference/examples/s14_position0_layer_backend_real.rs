//! FullDepth43 position0 任意单层的真实 GPU 数值门。
//!
//! 从冻结的四流 BF16 hidden 启动指定层，严格比较该层的 attention、KV、
//! route、MoE、HC 和 layer output；绝不签发 whole-token 完成回执。

use anyhow::{bail, Context, Result};
use polaris_s14_runner::{DecoderStateV1, Position0WholeTokenManifest};
use sha2::{Digest, Sha256};
use ssd_inference::{
    s14_position0_hybrid_weight_arena::S14Position0HybridArenaLayout,
    s14_position0_layer_backend::S14Position0L0Backend,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_position0_whole_token::{
        Position0GpuBootstrap, Position0GpuCandidate, Position0LayerBackend,
    },
    s14_whole_token_device::WholeTokenDeviceState,
    VulkanContext,
};
use std::{env, fs, path::Path, path::PathBuf, time::Instant};

fn read_bf16(path: &Path) -> Result<Vec<u16>> {
    let bytes = fs::read(path).with_context(|| format!("读取 {}", path.display()))?;
    if bytes.len() % 2 != 0 {
        bail!("{} 不是完整 BF16LE", path.display());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes([word[0], word[1]]))
        .collect())
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("读取 {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!("{} 不是完整 F32LE", path.display());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("4-byte chunk")))
        .collect())
}

fn exact_bf16(name: &str, actual: &[u16], expected_path: &Path) -> Result<()> {
    let expected = read_bf16(expected_path)?;
    if actual.len() != expected.len() {
        bail!(
            "{name} shape 漂移: actual={} expected={}",
            actual.len(),
            expected.len()
        );
    }
    let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    if mismatches != 0 {
        let (first_index, first_actual, first_expected) = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(index, (actual, expected))| (index, *actual, *expected))
            .expect("mismatch exists");
        bail!(
            "{name} 未逐元素精确: mismatches={mismatches}/{} first_index={first_index} actual=0x{first_actual:04x} expected=0x{first_expected:04x}",
            actual.len(),
        );
    }
    Ok(())
}

fn exact_f32(name: &str, actual: &[f32], expected_path: &Path) -> Result<()> {
    let expected = read_f32(expected_path)?;
    if actual.len() != expected.len() {
        bail!(
            "{name} shape 漂移: actual={} expected={}",
            actual.len(),
            expected.len()
        );
    }
    let mismatches = actual
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    if mismatches != 0 {
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let (first_index, first_actual, first_expected) = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .find(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(index, (actual, expected))| (index, *actual, *expected))
            .expect("mismatch exists");
        bail!(
            "{name} 未逐元素精确: mismatches={mismatches}/{} max_abs={max_abs} first_index={first_index} actual={first_actual:.12} expected={first_expected:.12} actual_bits=0x{:08x} expected_bits=0x{:08x}",
            actual.len()
            ,first_actual.to_bits(), first_expected.to_bits()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let layer_index = env::var("POLARIS_LAYER_INDEX")
        .unwrap_or_else(|_| "1".into())
        .parse::<usize>()
        .context("POLARIS_LAYER_INDEX 必须是 0..42")?;
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let reference_root = env::var_os("POLARIS_LAYER_STAGE_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "D:/project/大模型ssd化/.tmp-polaris-runs/l{layer_index}-stage-reference-20260802"
            ))
        });
    let payload_root = PathBuf::from("D:/models/Polaris-S14/range_cache");
    let layer_input = fs::read(reference_root.join("layer_input.bf16le.bin"))?;
    let mut hc_aux = fs::read(reference_root.join("post_ffn.f32le.bin"))?;
    hc_aux.extend_from_slice(&fs::read(reference_root.join("comb_ffn.f32le.bin"))?);
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let layer = manifest
        .layers
        .get(layer_index)
        .ok_or_else(|| anyhow::anyhow!("manifest 缺少 layer index {layer_index}"))?;
    let weights = S14Position0HybridWeightPlan::build(&manifest)?;
    let physical = S14Position0HybridArenaLayout::build(&weights)?;
    let static_layout = physical
        .static_layers
        .get(layer_index)
        .ok_or_else(|| anyhow::anyhow!("physical layout 缺少 layer index {layer_index}"))?;
    let ctx = VulkanContext::init()?;
    let host = DecoderStateV1::new(4096, manifest.input_token_id)?;
    let mut device = WholeTokenDeviceState::new(&ctx, host.native_arena.bytes(), 0)?;

    let init_started = Instant::now();
    let mut backend = S14Position0L0Backend::new_gpu_layer_numeric(
        &ctx,
        &manifest,
        &weights,
        static_layout,
        layer_index,
        &payload_root,
        &layer_input,
        &hc_aux,
    )?;
    let init_ms = init_started.elapsed().as_secs_f64() * 1000.0;
    let command = device.begin_candidate(&ctx, 0)?;
    {
        let bootstrap = Position0GpuBootstrap {
            candidate: Position0GpuCandidate {
                ctx: &ctx,
                candidate_state: device.candidate_buffer()?,
                sticky_status: device.sticky_status_buffer()?,
                committed_host_state: &host,
                base_epoch: 0,
                candidate_bank: 1,
            },
            prologue_command: command,
        };
        backend.submit_numeric_layer_input(&bootstrap)?;
    }
    device.mark_candidate_in_flight()?;
    let compute_started = Instant::now();
    let receipt = {
        let candidate = Position0GpuCandidate {
            ctx: &ctx,
            candidate_state: device.candidate_buffer()?,
            sticky_status: device.sticky_status_buffer()?,
            committed_host_state: &host,
            base_epoch: 0,
            candidate_bank: 1,
        };
        backend.submit_layer(&candidate, layer)?;
        backend.wait_l0_numeric(&candidate)?
    };
    let compute_ms = compute_started.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "layer={} route_actual={:?} route_expected={:?}",
        layer.layer, receipt.route_weights, layer.route_weights
    );

    let expected_ids: Vec<u32> = layer.expert_ids.iter().copied().map(u32::from).collect();
    if receipt.route_ids.as_slice() != expected_ids
        || receipt.sticky_status != 0
        || !receipt.kv_candidate_exact
    {
        bail!("L{} route/KV/status 未闭合", layer.layer);
    }
    exact_f32(
        "attention_input",
        &receipt.attention_input_f32_values,
        &reference_root.join("attention_input_quant.f32le.bin"),
    )?;
    exact_bf16(
        "query_final",
        &receipt.query_final_bf16_bits,
        &reference_root.join("query_final.bf16le.bin"),
    )?;
    exact_bf16(
        "key_value_final",
        &receipt.key_value_final_bf16_bits,
        &reference_root.join("key_value_final.bf16le.bin"),
    )?;
    exact_bf16(
        "attention_output",
        &receipt.attention_output_bf16_bits,
        &reference_root.join("attention_output.bf16le.bin"),
    )?;
    exact_f32(
        "wo_a_qdq",
        &receipt.wo_a_qdq_f32_values,
        &reference_root.join("wo_a_qdq.f32le.bin"),
    )?;
    exact_bf16(
        "attention_branch",
        &receipt.attention_branch_bf16_bits,
        &reference_root.join("attention_branch.bf16le.bin"),
    )?;
    exact_bf16(
        "post_attention",
        &receipt.post_attention_bf16_bits,
        &reference_root.join("post_attention.bf16le.bin"),
    )?;
    exact_f32(
        "ffn_input",
        &receipt.ffn_input_f32_values,
        &reference_root.join("ffn_input_quant.f32le.bin"),
    )?;
    exact_f32(
        "hc_post",
        &receipt.hc_post_f32_values,
        &reference_root.join("post_ffn.f32le.bin"),
    )?;
    exact_f32(
        "hc_comb",
        &receipt.hc_comb_f32_values,
        &reference_root.join("comb_ffn.f32le.bin"),
    )?;
    exact_bf16(
        "moe_branch",
        &receipt.moe_bf16_bits,
        &reference_root.join("moe_branch.bf16le.bin"),
    )?;
    exact_bf16(
        "layer_output",
        &receipt.hidden_bf16_bits,
        &reference_root.join("layer_output.bf16le.bin"),
    )?;
    let expected_hidden = fs::read(reference_root.join("layer_output.f32le.bin"))?;
    let expected_hidden_sha = format!("{:x}", Sha256::digest(&expected_hidden));
    if receipt.hidden_f32_le_sha256 != expected_hidden_sha {
        bail!(
            "L{} layer_output SHA 漂移: actual={} expected={expected_hidden_sha}",
            layer.layer,
            receipt.hidden_f32_le_sha256
        );
    }

    device.rollback_external_candidate(&ctx)?;
    drop(backend);
    device.destroy(&ctx)?;
    println!(
        "status=pass layer={} init_ms={init_ms:.3} compute_wait_ms={compute_ms:.3} route_ids={:?} route_weight_sum={:.9} hidden_sha={} decoder_state_committed=false",
        layer.layer,
        receipt.route_ids,
        receipt.route_weights.iter().sum::<f32>(),
        receipt.hidden_f32_le_sha256,
    );
    Ok(())
}
