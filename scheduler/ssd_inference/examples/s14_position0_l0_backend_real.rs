//! FullDepth43 position0 的真实 L0 GPU 后端数值门。
//!
//! 只验证 BOS embedding → L0 完整图 → route/KV/hidden 回读；不会把 L0
//! 结果冒充 43 层 token，也不会提交 DecoderState。

use anyhow::{bail, Result};
use polaris_s14_runner::{DecoderStateV1, Position0WholeTokenManifest};
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
use std::{env, fs, path::PathBuf, time::Instant};

#[derive(Debug)]
struct NumericComparison {
    mismatches: usize,
    total: usize,
    max_abs_error: f32,
    mean_abs_error: f64,
    relative_l2: f64,
    cosine: f64,
}

fn summarize_error(actual: &[f32], expected: &[f32]) -> Result<NumericComparison> {
    if actual.len() != expected.len() || actual.is_empty() {
        bail!(
            "numeric comparison shape drift: actual={} expected={}",
            actual.len(),
            expected.len()
        );
    }
    let mut mismatches = 0usize;
    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f64;
    let mut sum_sq_error = 0.0f64;
    let mut actual_sum_sq = 0.0f64;
    let mut expected_sum_sq = 0.0f64;
    let mut dot = 0.0f64;
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.to_bits() != expected.to_bits() {
            mismatches += 1;
        }
        let error = (actual - expected).abs();
        max_abs_error = max_abs_error.max(error);
        sum_abs_error += f64::from(error);
        sum_sq_error += f64::from(error) * f64::from(error);
        actual_sum_sq += f64::from(*actual) * f64::from(*actual);
        expected_sum_sq += f64::from(*expected) * f64::from(*expected);
        dot += f64::from(*actual) * f64::from(*expected);
    }
    Ok(NumericComparison {
        mismatches,
        total: actual.len(),
        max_abs_error,
        mean_abs_error: sum_abs_error / actual.len() as f64,
        relative_l2: sum_sq_error.sqrt() / expected_sum_sq.sqrt(),
        cosine: dot / (actual_sum_sq * expected_sum_sq).sqrt(),
    })
}

fn compare_f32le(actual: &[f32], path: &PathBuf) -> Result<NumericComparison> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        bail!("{} 不是完整 F32LE", path.display());
    }
    let expected = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect::<Vec<_>>();
    summarize_error(actual, &expected)
}

fn compare_bf16le(actual: &[u16], path: &PathBuf) -> Result<NumericComparison> {
    let bytes = fs::read(path)?;
    if bytes.len() % 2 != 0 {
        bail!("{} 不是完整 BF16LE", path.display());
    }
    let expected = bytes
        .chunks_exact(2)
        .map(|word| f32::from_bits(u32::from(u16::from_le_bytes([word[0], word[1]])) << 16))
        .collect::<Vec<_>>();
    let actual = actual
        .iter()
        .map(|bits| f32::from_bits(u32::from(*bits) << 16))
        .collect::<Vec<_>>();
    summarize_error(&actual, &expected)
}

fn main() -> Result<()> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let payload_root = PathBuf::from("D:/models/Polaris-S14/range_cache");
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let weights = S14Position0HybridWeightPlan::build(&manifest)?;
    let physical = S14Position0HybridArenaLayout::build(&weights)?;
    let ctx = VulkanContext::init()?;
    let host = DecoderStateV1::new(4096, manifest.input_token_id)?;
    let mut device = WholeTokenDeviceState::new(&ctx, host.native_arena.bytes(), 0)?;
    let force_reference_route = env::var_os("POLARIS_L0_FORCE_REFERENCE_ROUTE").is_some();

    let started = Instant::now();
    let mut backend = if force_reference_route {
        S14Position0L0Backend::new_gpu_numeric_route_override(
            &ctx,
            &manifest,
            &weights,
            &physical.static_layers[0],
            &payload_root,
            manifest.layers[0].route_weights.as_slice().try_into()?,
        )?
    } else {
        S14Position0L0Backend::new_gpu(
            &ctx,
            &manifest,
            &weights,
            &physical.static_layers[0],
            &payload_root,
        )?
    };
    let init_ms = started.elapsed().as_secs_f64() * 1000.0;
    let verified = backend
        .verified_payload_stats()
        .ok_or_else(|| anyhow::anyhow!("L0 GPU owner 未发布 verified stats"))?;

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
        backend.submit_embedding(&bootstrap, &manifest.embedding_row)?;
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
        backend.submit_layer(&candidate, &manifest.layers[0])?;
        backend.wait_l0_numeric(&candidate)?
    };
    let compute_ms = compute_started.elapsed().as_secs_f64() * 1000.0;

    if receipt.route_ids != [254, 222, 245, 200, 53, 35]
        || receipt.finite_hidden_elements != 4 * 4096
        || receipt.nonzero_hidden_elements == 0
        || !receipt.kv_candidate_exact
        || receipt.sticky_status != 0
    {
        bail!("L0 numeric receipt 未闭合: {receipt:?}");
    }
    let expected_route = &manifest.layers[0].route_weights;
    let route_max_abs_error = receipt
        .route_weights
        .iter()
        .zip(expected_route)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    // 路由投影在生产图内经过 BF16 边界；允许不超过 3e-4 的单权重误差，
    // 但专家身份和 1.5 总权重仍必须严格闭合。
    if route_max_abs_error > 3.0e-4 {
        bail!("L0 route weight 与 reference 漂移: max_abs={route_max_abs_error}");
    }
    let stage_reference_root = env::var_os("POLARIS_L0_STAGE_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("D:/project/大模型ssd化/.tmp-polaris-runs/l0-stage-reference-20260802")
        });
    let attention_input_comparison = compare_f32le(
        &receipt.attention_input_f32_values,
        &stage_reference_root.join("attention_input_quant.f32le.bin"),
    )?;
    let attention_branch_comparison = compare_bf16le(
        &receipt.attention_branch_bf16_bits,
        &stage_reference_root.join("attention_branch.bf16le.bin"),
    )?;
    let post_attention_comparison = compare_bf16le(
        &receipt.post_attention_bf16_bits,
        &stage_reference_root.join("post_attention.bf16le.bin"),
    )?;
    let query_final_comparison = compare_bf16le(
        &receipt.query_final_bf16_bits,
        &stage_reference_root.join("query_final.bf16le.bin"),
    )?;
    let key_value_final_comparison = compare_bf16le(
        &receipt.key_value_final_bf16_bits,
        &stage_reference_root.join("key_value_final.bf16le.bin"),
    )?;
    let attention_output_comparison = compare_bf16le(
        &receipt.attention_output_bf16_bits,
        &stage_reference_root.join("attention_output.bf16le.bin"),
    )?;
    let wo_a_qdq_comparison = compare_f32le(
        &receipt.wo_a_qdq_f32_values,
        &stage_reference_root.join("wo_a_qdq.f32le.bin"),
    )?;
    println!(
        "stage_diagnostic attention_input={attention_input_comparison:?} attention_input_sha={} query_final={query_final_comparison:?} key_value_final={key_value_final_comparison:?} attention_output={attention_output_comparison:?} wo_a_qdq={wo_a_qdq_comparison:?} attention_branch={attention_branch_comparison:?} attention_branch_sha={} post_attention={post_attention_comparison:?} post_attention_sha={}",
        receipt.attention_input_f32_le_sha256,
        receipt.attention_branch_bf16_le_sha256,
        receipt.post_attention_bf16_le_sha256,
    );
    for (name, comparison) in [
        ("attention_input", &attention_input_comparison),
        ("query_final", &query_final_comparison),
        ("key_value_final", &key_value_final_comparison),
        ("attention_output", &attention_output_comparison),
        ("wo_a_qdq", &wo_a_qdq_comparison),
        ("attention_branch", &attention_branch_comparison),
        ("post_attention", &post_attention_comparison),
    ] {
        if comparison.mismatches != 0 {
            bail!(
                "L0 stage {name} 未达到逐元素精确门: mismatches={}/{} max_abs={} relative_l2={}",
                comparison.mismatches,
                comparison.total,
                comparison.max_abs_error,
                comparison.relative_l2,
            );
        }
    }
    let hidden_reference_match =
        receipt.hidden_f32_le_sha256 == manifest.layers[0].reference.layer_output_f32_le_sha256;
    let moe_reference_match =
        receipt.moe_f32_le_sha256 == manifest.layers[0].reference.moe_branch_f32_le_sha256;
    let expected_ffn_bytes = fs::read(&manifest.layers[0].capture.input_path)?;
    if expected_ffn_bytes.len() != receipt.ffn_input_f32_values.len() * 4 {
        bail!(
            "L0 FFN input capture 长度漂移: actual={} expected={}",
            receipt.ffn_input_f32_values.len() * 4,
            expected_ffn_bytes.len()
        );
    }
    let expected_ffn = expected_ffn_bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect::<Vec<_>>();
    let mut ffn_mismatches = 0usize;
    let mut ffn_max_abs_error = 0.0f32;
    let mut ffn_sum_abs_error = 0.0f64;
    let mut ffn_sum_sq_error = 0.0f64;
    let mut ffn_actual_sum_sq = 0.0f64;
    let mut ffn_expected_sum_sq = 0.0f64;
    let mut ffn_dot = 0.0f64;
    for (actual, expected) in receipt.ffn_input_f32_values.iter().zip(&expected_ffn) {
        if actual.to_bits() != expected.to_bits() {
            ffn_mismatches += 1;
        }
        let error = (actual - expected).abs();
        ffn_max_abs_error = ffn_max_abs_error.max(error);
        ffn_sum_abs_error += f64::from(error);
        ffn_sum_sq_error += f64::from(error) * f64::from(error);
        ffn_actual_sum_sq += f64::from(*actual) * f64::from(*actual);
        ffn_expected_sum_sq += f64::from(*expected) * f64::from(*expected);
        ffn_dot += f64::from(*actual) * f64::from(*expected);
    }
    let ffn_mean_abs_error = ffn_sum_abs_error / expected_ffn.len() as f64;
    let ffn_relative_l2 = ffn_sum_sq_error.sqrt() / ffn_expected_sum_sq.sqrt();
    let ffn_cosine = ffn_dot / (ffn_actual_sum_sq * ffn_expected_sum_sq).sqrt();
    let ffn_reference_match =
        receipt.ffn_input_f32_le_sha256 == manifest.layers[0].capture.input_sha256;
    let expected_moe_bytes = fs::read(&manifest.layers[0].capture.k1_moe_output_path)?;
    if expected_moe_bytes.len() != receipt.moe_bf16_bits.len() * 2 {
        bail!(
            "L0 MoE BF16 capture 长度漂移: actual={} expected={}",
            receipt.moe_bf16_bits.len() * 2,
            expected_moe_bytes.len()
        );
    }
    let expected_moe_bf16 = expected_moe_bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    let mut moe_bf16_mismatches = 0usize;
    let mut moe_bf16_max_abs_error = 0.0f32;
    let mut moe_bf16_sum_abs_error = 0.0f64;
    let mut moe_bf16_sum_sq_error = 0.0f64;
    let mut moe_bf16_actual_sum_sq = 0.0f64;
    let mut moe_bf16_expected_sum_sq = 0.0f64;
    let mut moe_bf16_dot = 0.0f64;
    for (actual, expected) in receipt.moe_bf16_bits.iter().zip(&expected_moe_bf16) {
        if actual != expected {
            moe_bf16_mismatches += 1;
        }
        let actual = f32::from_bits(u32::from(*actual) << 16);
        let expected = f32::from_bits(u32::from(*expected) << 16);
        let error = (actual - expected).abs();
        moe_bf16_max_abs_error = moe_bf16_max_abs_error.max(error);
        moe_bf16_sum_abs_error += f64::from(error);
        moe_bf16_sum_sq_error += f64::from(error) * f64::from(error);
        moe_bf16_actual_sum_sq += f64::from(actual) * f64::from(actual);
        moe_bf16_expected_sum_sq += f64::from(expected) * f64::from(expected);
        moe_bf16_dot += f64::from(actual) * f64::from(expected);
    }
    let moe_bf16_mean_abs_error = moe_bf16_sum_abs_error / expected_moe_bf16.len() as f64;
    let moe_bf16_relative_l2 = moe_bf16_sum_sq_error.sqrt() / moe_bf16_expected_sum_sq.sqrt();
    let moe_bf16_cosine = moe_bf16_dot / (moe_bf16_actual_sum_sq * moe_bf16_expected_sum_sq).sqrt();
    let moe_bf16_reference_match =
        receipt.moe_bf16_le_sha256 == manifest.layers[0].capture.k1_moe_output_sha256;
    if !hidden_reference_match {
        bail!(
            "L0 hidden SHA 与 reference 漂移: route_mode={} actual={} expected={} l2={:.9} mean={:.9} max_abs={:.9}; route_actual={:?} route_expected={:?}; ffn_match={} ffn_sha={} mismatches={}/{} max_abs_error={:.9} mean_abs_error={:.9} relative_l2={:.9} cosine={:.12}; moe_f32_match={} moe_sha={} moe_l2={:.9} moe_mean={:.9} moe_max_abs={:.9}; moe_bf16_match={} moe_bf16_sha={} mismatches={}/{} max_abs_error={:.9} mean_abs_error={:.9} relative_l2={:.9} cosine={:.12}",
            if force_reference_route { "reference_override" } else { "native_gpu" },
            receipt.hidden_f32_le_sha256,
            manifest.layers[0].reference.layer_output_f32_le_sha256,
            receipt.hidden_l2,
            receipt.hidden_mean,
            receipt.hidden_max_abs,
            receipt.route_weights,
            expected_route,
            ffn_reference_match,
            receipt.ffn_input_f32_le_sha256,
            ffn_mismatches,
            expected_ffn.len(),
            ffn_max_abs_error,
            ffn_mean_abs_error,
            ffn_relative_l2,
            ffn_cosine,
            moe_reference_match,
            receipt.moe_f32_le_sha256,
            receipt.moe_l2,
            receipt.moe_mean,
            receipt.moe_max_abs,
            moe_bf16_reference_match,
            receipt.moe_bf16_le_sha256,
            moe_bf16_mismatches,
            expected_moe_bf16.len(),
            moe_bf16_max_abs_error,
            moe_bf16_mean_abs_error,
            moe_bf16_relative_l2,
            moe_bf16_cosine,
        );
    }

    device.rollback_external_candidate(&ctx)?;
    drop(backend);
    device.destroy(&ctx)?;
    println!(
        "status=pass init_ms={init_ms:.3} l0_compute_wait_ms={compute_ms:.3} \
         verified_requests={} verified_misses={} verified_sha_bytes={} route_ids={:?} \
         route_weight_sum={:.8} route_max_abs_error={route_max_abs_error:.9} \
         hidden_reference_match={hidden_reference_match} hidden_l2={:.9} hidden_mean={:.9} \
         hidden_max_abs={:.9} finite_hidden={} nonzero_hidden={} kv_candidate_exact={} \
         compute_value={} decoder_state_committed=false",
        verified.requests,
        verified.misses,
        verified.sha256_bytes,
        receipt.route_ids,
        receipt.route_weights.iter().sum::<f32>(),
        receipt.hidden_l2,
        receipt.hidden_mean,
        receipt.hidden_max_abs,
        receipt.finite_hidden_elements,
        receipt.nonzero_hidden_elements,
        receipt.kv_candidate_exact,
        receipt.compute_value,
    );
    Ok(())
}
