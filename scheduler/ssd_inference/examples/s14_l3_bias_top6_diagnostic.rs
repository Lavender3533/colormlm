//! L3 真实 BiasTop6 顺序交换的独立 Vulkan 归因门。
//!
//! 该程序消费 CPU 参考路径显式捕获的量化前 `ffn_input.f32le.bin`，
//! 使用真实 BF16 router weight 与 F32 bias，在同一 command buffer 中执行
//! GPU matvec 和 route postprocess。报告保留 256 个 expert 的 CPU/GPU logits、
//! score、bias、ranking，以及 GPU top-6 的全部 selected/ranking 输出。

use anyhow::{bail, Context, Result};
use ash::vk;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use ssd_inference::{
    s14_route_postprocess::{
        postprocess_s14_route, sqrt_softplus_f32, S14RouteBias, S14RoutePostprocessKind,
    },
    s14_route_postprocess_gpu::{
        validate_route_postprocess_gpu_status, S14RouteBufferSlice, S14RoutePostprocessGpuBindings,
        S14RoutePostprocessGpuMode, S14RoutePostprocessGpuPipeline, S14_ROUTE_GPU_TOP_K,
    },
    s14_vulkan::{S14Bf16MatvecShape, S14NumericPipelines},
    GpuBuffer, VulkanContext,
};
use std::{
    cmp::Ordering,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

const N: usize = 256;
const K: usize = 4096;
const LAYER: u8 = 3;
const MANIFEST_RELATIVE: &str =
    "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json";
const DEFAULT_CAPTURE: &str = "../../.tmp-polaris-tests/l3-bias-top6-cpu-20260802";
const WEIGHT_TENSOR: &str = "layers.3.ffn.gate.weight";
const BIAS_TENSOR: &str = "layers.3.ffn.gate.bias";

#[derive(Debug, Serialize)]
struct ExpertRow {
    expert: usize,
    cpu_logit: f32,
    gpu_logit: f32,
    logit_abs_error: f32,
    cpu_selected_score: f32,
    selected_score_from_gpu_logit: f32,
    bias: f32,
    cpu_ranking_score: f32,
    ranking_score_from_gpu_logit: f32,
}

#[derive(Debug, Serialize)]
struct GpuTop6Row {
    slot: usize,
    expert: u32,
    weight: f32,
    selected_score: f32,
    ranking_score: f32,
}

#[derive(Debug, Serialize)]
struct PairGap {
    first_expert: usize,
    second_expert: usize,
    cpu_ranking_gap: f32,
    gpu_ranking_gap: f32,
    cpu_exact_tie: bool,
    gpu_exact_tie: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    format: &'static str,
    status: &'static str,
    gpu: String,
    manifest: String,
    capture_root: String,
    manifest_expected_ids: Vec<u32>,
    cpu_capture_ids: Vec<u32>,
    rust_cpu_ids_from_captured_logits: Vec<u32>,
    gpu_ids: Vec<u32>,
    gpu_top6: Vec<GpuTop6Row>,
    experts: Vec<ExpertRow>,
    max_abs_gpu_vs_cpu_logit: f32,
    gpu_matvec_wall_ms: f64,
    gpu_route_wall_included: bool,
    pair_213_vs_79: PairGap,
    weight_sha256: String,
    bias_sha256: String,
    input_sha256: String,
    cpu_logits_sha256: String,
    diagnosis: Vec<&'static str>,
    claim_limit: &'static str,
}

fn main() -> Result<()> {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MANIFEST_RELATIVE);
    let capture_root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_CAPTURE));
    let output_path = std::env::args().nth(2).map(PathBuf::from);
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let layer = manifest["layers"]
        .as_array()
        .and_then(|layers| layers.get(LAYER as usize))
        .context("position0 manifest 缺少 L3")?;
    let manifest_expected_ids = json_u32_vec(&layer["expert_ids"])?;
    let weight_asset = router_asset(layer, WEIGHT_TENSOR)?;
    let bias_asset = router_asset(layer, BIAS_TENSOR)?;
    let weight_bytes = read_verified_asset(weight_asset, N * K * 2)?;
    let bias_bytes = read_verified_asset(bias_asset, N * 4)?;
    let bias: Vec<f32> = bias_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let input_path = capture_root.join("ffn_input.f32le.bin");
    let cpu_logits_path = capture_root.join("router_logits.f32le.bin");
    let cpu_scores_path = capture_root.join("router_scores.f32le.bin");
    let cpu_ranking_path = capture_root.join("router_ranking_scores.f32le.bin");
    let input_bytes = fs::read(&input_path)?;
    let cpu_logits_bytes = fs::read(&cpu_logits_path)?;
    read_f32_exact(&input_bytes, K, "ffn_input")?;
    let cpu_logits = read_f32_exact(&cpu_logits_bytes, N, "router_logits")?;
    let cpu_scores = read_f32_exact(&fs::read(cpu_scores_path)?, N, "router_scores")?;
    let cpu_ranking = read_f32_exact(&fs::read(cpu_ranking_path)?, N, "router_ranking_scores")?;
    for expert in 0..N {
        if (cpu_scores[expert] + bias[expert]).to_bits() != cpu_ranking[expert].to_bits() {
            bail!("CPU capture ranking[{expert}] 不是 score+bias 的逐位结果");
        }
    }

    let cpu_capture_ids = rank_ids(&cpu_ranking);
    let rust_cpu = postprocess_s14_route(
        LAYER,
        &cpu_logits,
        S14RoutePostprocessKind::ScoreTop6 {
            bias: Some(S14RouteBias::F32(&bias)),
        },
    )?;
    let rust_cpu_ids: Vec<u32> = rust_cpu
        .expert_ids
        .iter()
        .map(|id| u32::from(*id))
        .collect();

    let ctx = VulkanContext::init()?;
    let numeric = S14NumericPipelines::new(&ctx)?;
    let route = S14RoutePostprocessGpuPipeline::new(&ctx)?;
    let weight_buffer = host_buffer(&ctx, weight_bytes.len() as u64)?;
    let input_buffer = host_buffer(&ctx, input_bytes.len() as u64)?;
    let logits_buffer = host_buffer(&ctx, (N * 4) as u64)?;
    let bias_buffer = host_buffer(&ctx, bias_bytes.len() as u64)?;
    let ids_buffer = host_buffer(&ctx, (S14_ROUTE_GPU_TOP_K * 4) as u64)?;
    let weights_buffer = host_buffer(&ctx, (S14_ROUTE_GPU_TOP_K * 4) as u64)?;
    let selected_buffer = host_buffer(&ctx, (S14_ROUTE_GPU_TOP_K * 4) as u64)?;
    let ranking_buffer = host_buffer(&ctx, (S14_ROUTE_GPU_TOP_K * 4) as u64)?;
    let status_buffer = host_buffer(&ctx, 4)?;
    unsafe {
        weight_buffer.write_at(0, &weight_bytes);
        input_buffer.write_at(0, &input_bytes);
        bias_buffer.write_at(0, &bias_bytes);
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let matvec = numeric.bind_bf16_matvec(
        &ctx,
        S14Bf16MatvecShape::new(N as u32, K as u32, 1)?,
        &weight_buffer,
        &input_buffer,
        &logits_buffer,
    )?;
    let route_dispatch = route.bind_with_offsets(
        &ctx,
        S14RoutePostprocessGpuMode::BiasTop6,
        S14RoutePostprocessGpuBindings {
            logits: S14RouteBufferSlice::new(&logits_buffer, 0),
            aux: S14RouteBufferSlice::new(&bias_buffer, 0),
            expert_ids: S14RouteBufferSlice::new(&ids_buffer, 0),
            weights: S14RouteBufferSlice::new(&weights_buffer, 0),
            selected_scores: S14RouteBufferSlice::new(&selected_buffer, 0),
            ranking_scores: S14RouteBufferSlice::new(&ranking_buffer, 0),
            status: S14RouteBufferSlice::new(&status_buffer, 0),
        },
    )?;
    let gpu_wall_ms = execute(&ctx, &numeric, &route, &matvec, &route_dispatch)?;
    let status = mapped_one::<u32>(&status_buffer);
    validate_route_postprocess_gpu_status(status)?;
    let gpu_logits = mapped_vec::<f32>(&logits_buffer, N);
    let gpu_ids = mapped_vec::<u32>(&ids_buffer, S14_ROUTE_GPU_TOP_K);
    let gpu_weights = mapped_vec::<f32>(&weights_buffer, S14_ROUTE_GPU_TOP_K);
    let gpu_selected = mapped_vec::<f32>(&selected_buffer, S14_ROUTE_GPU_TOP_K);
    let gpu_ranking = mapped_vec::<f32>(&ranking_buffer, S14_ROUTE_GPU_TOP_K);

    let gpu_cpu = postprocess_s14_route(
        LAYER,
        &gpu_logits,
        S14RoutePostprocessKind::ScoreTop6 {
            bias: Some(S14RouteBias::F32(&bias)),
        },
    )?;
    let gpu_cpu_ids: Vec<u32> = gpu_cpu.expert_ids.iter().map(|id| u32::from(*id)).collect();
    if gpu_ids != gpu_cpu_ids {
        bail!("GPU top-k 与相同 GPU logits 的 CPU 后处理不一致");
    }

    let mut experts = Vec::with_capacity(N);
    let mut gpu_all_ranking = Vec::with_capacity(N);
    let mut max_abs_gpu_vs_cpu_logit = 0.0f32;
    for expert in 0..N {
        let gpu_score = sqrt_softplus_f32(gpu_logits[expert]);
        let ranking = gpu_score + bias[expert];
        gpu_all_ranking.push(ranking);
        let error = (gpu_logits[expert] - cpu_logits[expert]).abs();
        max_abs_gpu_vs_cpu_logit = max_abs_gpu_vs_cpu_logit.max(error);
        experts.push(ExpertRow {
            expert,
            cpu_logit: cpu_logits[expert],
            gpu_logit: gpu_logits[expert],
            logit_abs_error: error,
            cpu_selected_score: cpu_scores[expert],
            selected_score_from_gpu_logit: gpu_score,
            bias: bias[expert],
            cpu_ranking_score: cpu_ranking[expert],
            ranking_score_from_gpu_logit: ranking,
        });
    }
    let pair_213_vs_79 = PairGap {
        first_expert: 213,
        second_expert: 79,
        cpu_ranking_gap: cpu_ranking[213] - cpu_ranking[79],
        gpu_ranking_gap: gpu_all_ranking[213] - gpu_all_ranking[79],
        cpu_exact_tie: cpu_ranking[213].to_bits() == cpu_ranking[79].to_bits(),
        gpu_exact_tie: gpu_all_ranking[213].to_bits() == gpu_all_ranking[79].to_bits(),
    };
    let gpu_top6 = (0..S14_ROUTE_GPU_TOP_K)
        .map(|slot| GpuTop6Row {
            slot,
            expert: gpu_ids[slot],
            weight: gpu_weights[slot],
            selected_score: gpu_selected[slot],
            ranking_score: gpu_ranking[slot],
        })
        .collect();

    let report = Report {
        format: "polaris-l3-bias-top6-diagnostic-v1",
        status: "pass",
        gpu: ctx.gpu_name.clone(),
        manifest: manifest_path.to_string_lossy().into_owned(),
        capture_root: capture_root.to_string_lossy().into_owned(),
        manifest_expected_ids,
        cpu_capture_ids: cpu_capture_ids.iter().map(|id| *id as u32).collect(),
        rust_cpu_ids_from_captured_logits: rust_cpu_ids,
        gpu_ids,
        gpu_top6,
        experts,
        max_abs_gpu_vs_cpu_logit,
        gpu_matvec_wall_ms: gpu_wall_ms,
        gpu_route_wall_included: true,
        pair_213_vs_79,
        weight_sha256: sha256_bytes(&weight_bytes),
        bias_sha256: sha256_bytes(&bias_bytes),
        input_sha256: sha256_bytes(&input_bytes),
        cpu_logits_sha256: sha256_bytes(&cpu_logits_bytes),
        diagnosis: vec![
            "GPU top-k equals deterministic CPU postprocess when both consume the same GPU logits",
            "CPU capture and GPU independently rank expert 213 before expert 79",
            "213 and 79 are not an exact F32 tie, so the lower-ID tie-break is not active",
            "the position0 manifest expected order is stale for this frozen L3 input; route shader changes are not justified",
        ],
        claim_limit: "single real L3 router-input numerical diagnosis; not a whole-token or quality result",
    };
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    print!("{encoded}");
    if let Some(path) = output_path {
        fs::write(path, encoded.as_bytes())?;
    }

    route_dispatch.binder.destroy(&ctx);
    matvec.binder.destroy(&ctx);
    status_buffer.destroy(&ctx);
    ranking_buffer.destroy(&ctx);
    selected_buffer.destroy(&ctx);
    weights_buffer.destroy(&ctx);
    ids_buffer.destroy(&ctx);
    bias_buffer.destroy(&ctx);
    logits_buffer.destroy(&ctx);
    input_buffer.destroy(&ctx);
    weight_buffer.destroy(&ctx);
    route.destroy(&ctx);
    numeric.destroy(&ctx);
    Ok(())
}

fn execute(
    ctx: &VulkanContext,
    numeric: &S14NumericPipelines,
    route: &S14RoutePostprocessGpuPipeline,
    matvec: &ssd_inference::s14_vulkan::S14Bf16MatvecDispatch,
    route_dispatch: &ssd_inference::s14_route_postprocess_gpu::S14RoutePostprocessGpuDispatch,
) -> Result<f64> {
    unsafe {
        let pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_graphics),
            None,
        )?;
        let command = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        numeric.cmd_bf16_matvec(ctx, command, matvec);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            &[],
            &[],
        );
        route.cmd(ctx, command, route_dispatch);
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        let started = Instant::now();
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
        Ok(wall_ms)
    }
}

fn rank_ids(values: &[f32]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..values.len()).collect();
    ids.sort_unstable_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    ids.truncate(S14_ROUTE_GPU_TOP_K);
    ids
}

fn router_asset<'a>(layer: &'a Value, tensor: &str) -> Result<&'a Value> {
    layer["assets"]["router"]
        .as_array()
        .and_then(|assets| assets.iter().find(|asset| asset["tensor"] == tensor))
        .with_context(|| format!("L3 缺少 {tensor}"))
}

fn read_verified_asset(asset: &Value, expected_bytes: usize) -> Result<Vec<u8>> {
    let path = PathBuf::from(asset["path"].as_str().context("asset path missing")?);
    let bytes = fs::read(&path)?;
    let expected_sha = asset["sha256"].as_str().context("asset sha256 missing")?;
    if bytes.len() != expected_bytes || sha256_bytes(&bytes) != expected_sha {
        bail!("真实资产 size/SHA 漂移: {}", path.display());
    }
    Ok(bytes)
}

fn read_f32_exact(bytes: &[u8], count: usize, label: &str) -> Result<Vec<f32>> {
    if bytes.len() != count * 4 {
        bail!("{label} 字节数漂移: {} != {}", bytes.len(), count * 4);
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{label} 含 NaN/Inf");
    }
    Ok(values)
}

fn json_u32_vec(value: &Value) -> Result<Vec<u32>> {
    value
        .as_array()
        .context("expected JSON array")?
        .iter()
        .map(|value| {
            u32::try_from(value.as_u64().context("expected JSON u64")?)
                .context("JSON integer exceeds u32")
        })
        .collect()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn host_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
}

fn mapped_one<T: Copy>(buffer: &GpuBuffer) -> T {
    unsafe { *(buffer.mapped() as *const T) }
}

fn mapped_vec<T: Copy>(buffer: &GpuBuffer, count: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const T, count).to_vec() }
}
