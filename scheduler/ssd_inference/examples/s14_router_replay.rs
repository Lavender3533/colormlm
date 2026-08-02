//! FullDepth43 真实 router 的单 arena、单 command buffer Vulkan 回放门。
//!
//! 输入 manifest 由 `whole_token_runtime/build_router_replay_manifest.py`
//! 生成并冻结 SHA。现有 capture 是 activation-quant 后输入，因此这里只
//! 证明 GPU/CPU 同输入数值与 43 层持久调度，不冒充正式 route 一致性。

use anyhow::{bail, Context, Result};
use ash::vk;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ssd_inference::{
    s14_vulkan::{S14Bf16MatvecDispatch, S14Bf16MatvecShape, S14NumericPipelines},
    GpuBuffer, VulkanContext,
};
use std::{fs, path::PathBuf, time::Instant};

const LAYERS: usize = 43;
const N: usize = 256;
const K: usize = 4096;
const WEIGHT_BYTES: usize = N * K * 2;
const INPUT_BYTES: usize = K * 4;
const OUTPUT_BYTES: usize = N * 4;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: String,
    revision: String,
    profile: String,
    position: u32,
    catalog_path: String,
    catalog_sha256: String,
    capture_root: String,
    layers: Vec<LayerManifest>,
    summary: ManifestSummary,
    input_semantics: String,
    claim_limit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestSummary {
    layer_count: usize,
    router_weight_bytes: usize,
    input_bytes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerManifest {
    layer: usize,
    weight_path: PathBuf,
    weight_bytes: usize,
    weight_sha256: String,
    input_path: PathBuf,
    input_bytes: usize,
    input_sha256: String,
    observed_route_source: String,
    observed_expert_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct LayerError {
    layer: usize,
    max_abs_error: f32,
    gpu_raw_top6: Vec<usize>,
    observed_official_experts: Vec<u32>,
    observation_only: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    format: &'static str,
    gpu: String,
    manifest_path: String,
    manifest_sha256: String,
    position: u32,
    layer_count: usize,
    router_weight_bytes: usize,
    input_bytes: usize,
    output_bytes: usize,
    command_submit_count: usize,
    dispatch_count: usize,
    submit_wall_ms: f64,
    router_kernel_ms: f64,
    cpu_reference_wall_ms: f64,
    global_max_abs_error: f32,
    layers: Vec<LayerError>,
    input_semantics: String,
    claim_limit: &'static str,
}

fn main() -> Result<()> {
    let manifest_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: s14_router_replay <router_replay_manifest.json> [report.json]")?;
    let report_path = std::env::args().nth(2).map(PathBuf::from);
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest(&manifest)?;

    let mut weights = Vec::with_capacity(LAYERS * WEIGHT_BYTES);
    let mut inputs = Vec::with_capacity(LAYERS * K);
    for (layer, row) in manifest.layers.iter().enumerate() {
        let weight = read_verified(&row.weight_path, WEIGHT_BYTES, &row.weight_sha256)?;
        weights.extend_from_slice(&weight);
        let input = read_verified(&row.input_path, INPUT_BYTES, &row.input_sha256)?;
        inputs.extend(
            input
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())),
        );
        if inputs[layer * K..(layer + 1) * K]
            .iter()
            .any(|value| !value.is_finite())
        {
            bail!("L{layer} input contains NaN/Inf");
        }
    }

    let ctx = VulkanContext::init()?;
    let pipelines = S14NumericPipelines::new(&ctx)?;
    let weight_bytes = weights.len() as u64;
    let input_bytes = std::mem::size_of_val(inputs.as_slice()) as u64;
    let output_bytes = (LAYERS * OUTPUT_BYTES) as u64;
    let weight_upload = GpuBuffer::new_staging(&ctx, weight_bytes)?;
    let input_upload = GpuBuffer::new_staging(&ctx, input_bytes)?;
    let weight_arena = GpuBuffer::new_vram(
        &ctx,
        weight_bytes,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let input_arena = GpuBuffer::new_vram(
        &ctx,
        input_bytes,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let output_arena = GpuBuffer::new_vram(
        &ctx,
        output_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    let readback = GpuBuffer::new(
        &ctx,
        output_bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;
    unsafe {
        weight_upload.write_at(0, &weights);
        input_upload.write_at(0, bytemuck::cast_slice(&inputs));
    }

    let shape = S14Bf16MatvecShape::new(N as u32, K as u32, 1)?;
    let mut dispatches = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        dispatches.push(pipelines.bind_bf16_matvec_arenas(
            &ctx,
            shape,
            &weight_arena,
            weight_bytes,
            (layer * WEIGHT_BYTES) as u64,
            &input_arena,
            input_bytes,
            (layer * INPUT_BYTES) as u64,
            &output_arena,
            output_bytes,
            (layer * OUTPUT_BYTES) as u64,
        )?);
    }

    let physical = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let timestamp_period_ns = physical.limits.timestamp_period as f64;
    let queue_properties = unsafe {
        ctx.instance
            .get_physical_device_queue_family_properties(ctx.physical)
    };
    let timestamp_bits = queue_properties[ctx.qf_graphics as usize].timestamp_valid_bits;
    if timestamp_bits == 0 {
        bail!("graphics queue does not support timestamps");
    }
    let (submit_wall_ms, router_kernel_ms) = execute(
        &ctx,
        &pipelines,
        &dispatches,
        &weight_upload,
        &input_upload,
        &weight_arena,
        &input_arena,
        &output_arena,
        &readback,
        weight_bytes,
        input_bytes,
        output_bytes,
        timestamp_period_ns,
        timestamp_bits,
    )?;

    let gpu_output =
        unsafe { std::slice::from_raw_parts(readback.mapped() as *const f32, LAYERS * N) };
    if gpu_output.iter().any(|value| !value.is_finite()) {
        bail!("router GPU output contains NaN/Inf");
    }
    let cpu_started = Instant::now();
    let cpu_rows: Vec<Vec<f32>> = (0..LAYERS)
        .into_par_iter()
        .map(|layer| {
            cpu_reference(
                &weights[layer * WEIGHT_BYTES..(layer + 1) * WEIGHT_BYTES],
                &inputs[layer * K..(layer + 1) * K],
            )
        })
        .collect();
    let cpu_reference_wall_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
    let mut global_max_abs_error = 0.0f32;
    let mut layer_errors = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let gpu = &gpu_output[layer * N..(layer + 1) * N];
        let max_abs_error = gpu
            .iter()
            .zip(&cpu_rows[layer])
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f32, f32::max);
        global_max_abs_error = global_max_abs_error.max(max_abs_error);
        layer_errors.push(LayerError {
            layer,
            max_abs_error,
            gpu_raw_top6: top6(gpu),
            observed_official_experts: manifest.layers[layer].observed_expert_ids.clone(),
            observation_only: true,
        });
    }
    if global_max_abs_error > 2.0e-3 {
        bail!("43-layer router GPU/CPU max abs error {global_max_abs_error} exceeds 0.002");
    }

    let report = Report {
        format: "polaris-fulldepth43-router-replay-v1",
        gpu: ctx.gpu_name.clone(),
        manifest_path: manifest_path.to_string_lossy().into_owned(),
        manifest_sha256,
        position: manifest.position,
        layer_count: LAYERS,
        router_weight_bytes: weights.len(),
        input_bytes: input_bytes as usize,
        output_bytes: output_bytes as usize,
        command_submit_count: 1,
        dispatch_count: LAYERS,
        submit_wall_ms,
        router_kernel_ms,
        cpu_reference_wall_ms,
        global_max_abs_error,
        layers: layer_errors,
        input_semantics: manifest.input_semantics,
        claim_limit: "43 real router weights and captures in one persistent Vulkan graph; capture is post-route activation-quant input, so this is numerical/runtime evidence, not formal route or model-quality evidence.",
    };
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    print!("{encoded}");
    if let Some(path) = report_path {
        fs::write(path, encoded.as_bytes())?;
    }

    for dispatch in dispatches.iter().rev() {
        dispatch.binder.destroy(&ctx);
    }
    readback.destroy(&ctx);
    output_arena.destroy(&ctx);
    input_arena.destroy(&ctx);
    weight_arena.destroy(&ctx);
    input_upload.destroy(&ctx);
    weight_upload.destroy(&ctx);
    pipelines.destroy(&ctx);
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.format != "polaris-fulldepth43-router-replay-manifest-v1"
        || manifest.revision != "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
        || manifest.profile != "fulldepth43_native_top6"
        || manifest.layers.len() != LAYERS
        || manifest.summary.layer_count != LAYERS
        || manifest.summary.router_weight_bytes != LAYERS * WEIGHT_BYTES
        || manifest.summary.input_bytes != LAYERS * INPUT_BYTES
    {
        bail!("router replay manifest identity/summary drift");
    }
    if manifest.catalog_path.is_empty()
        || manifest.catalog_sha256.len() != 64
        || manifest.capture_root.is_empty()
        || manifest.claim_limit.is_empty()
    {
        bail!("router replay manifest proof fields are incomplete");
    }
    for (layer, row) in manifest.layers.iter().enumerate() {
        if row.layer != layer
            || row.weight_bytes != WEIGHT_BYTES
            || row.input_bytes != INPUT_BYTES
            || row.observed_expert_ids.len() != 6
            || row.observed_route_source.is_empty()
        {
            bail!("L{layer} router replay contract drift");
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    dispatches: &[S14Bf16MatvecDispatch],
    weight_upload: &GpuBuffer,
    input_upload: &GpuBuffer,
    weight_arena: &GpuBuffer,
    input_arena: &GpuBuffer,
    output_arena: &GpuBuffer,
    readback: &GpuBuffer,
    weight_bytes: u64,
    input_bytes: u64,
    output_bytes: u64,
    timestamp_period_ns: f64,
    timestamp_bits: u32,
) -> Result<(f64, f64)> {
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
        let queries = ctx.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2),
            None,
        )?;
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_copy_buffer(
            command,
            weight_upload.handle(),
            weight_arena.handle(),
            &[vk::BufferCopy::default().size(weight_bytes)],
        );
        ctx.device.cmd_copy_buffer(
            command,
            input_upload.handle(),
            input_arena.handle(),
            &[vk::BufferCopy::default().size(input_bytes)],
        );
        let transfer_to_compute = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[transfer_to_compute],
            &[],
            &[],
        );
        ctx.device.cmd_reset_query_pool(command, queries, 0, 2);
        ctx.device
            .cmd_write_timestamp(command, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
        for dispatch in dispatches {
            pipelines.cmd_bf16_matvec(ctx, command, dispatch);
        }
        ctx.device
            .cmd_write_timestamp(command, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
        let compute_to_transfer = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_to_transfer],
            &[],
            &[],
        );
        ctx.device.cmd_copy_buffer(
            command,
            output_arena.handle(),
            readback.handle(),
            &[vk::BufferCopy::default().size(output_bytes)],
        );
        ctx.device.end_command_buffer(command)?;
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        let commands = [command];
        let started = Instant::now();
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let submit_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut ticks = [0u64; 2];
        ctx.device.get_query_pool_results(
            queries,
            0,
            &mut ticks,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;
        let mask = if timestamp_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << timestamp_bits) - 1
        };
        let elapsed_ticks = ticks[1].wrapping_sub(ticks[0]) & mask;
        let kernel_ms = elapsed_ticks as f64 * timestamp_period_ns / 1_000_000.0;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_query_pool(queries, None);
        ctx.device.destroy_command_pool(pool, None);
        Ok((submit_wall_ms, kernel_ms))
    }
}

fn read_verified(path: &PathBuf, expected_bytes: usize, expected_sha: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if bytes.len() != expected_bytes || sha256_bytes(&bytes) != expected_sha {
        bail!("payload size/SHA drift: {}", path.display());
    }
    Ok(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn cpu_reference(weight: &[u8], input: &[f32]) -> Vec<f32> {
    (0..N)
        .map(|row| {
            let mut lanes = [0.0f32; 64];
            for lane in 0..64 {
                let mut acc = 0.0f32;
                for dim in (lane..K).step_by(64) {
                    let offset = (row * K + dim) * 2;
                    let bf16 = u16::from_le_bytes([weight[offset], weight[offset + 1]]);
                    acc += f32::from_bits((bf16 as u32) << 16) * input[dim];
                }
                lanes[lane] = acc;
            }
            let mut stride = 32;
            while stride > 0 {
                for lane in 0..stride {
                    lanes[lane] += lanes[lane + stride];
                }
                stride >>= 1;
            }
            lanes[0]
        })
        .collect()
}

fn top6(values: &[f32]) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..values.len()).collect();
    ids.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
    ids.truncate(6);
    ids
}
