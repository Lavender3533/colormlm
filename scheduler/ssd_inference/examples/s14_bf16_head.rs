//! Polaris / DeepSeek-V4 真实 BF16 全词表 head Vulkan 短门。
//!
//! 默认读取本机已经缓存的固定 revision `head.weight`，一次上传
//! 后相邻测量 K=1/4/8。这个程序只证明 head 的数值与热态速度，
//! 不将 head-only tok/s 写成整模型 tok/s。

use anyhow::{bail, Context, Result};
use ash::vk;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder},
    device::VulkanContext,
};
use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};

const VOCAB: usize = 129_280;
const HIDDEN: usize = 4_096;
const HEAD_BYTES: usize = VOCAB * HIDDEN * 2;
const MAX_GROUPS_X: u32 = 65_535;
const DEFAULT_HEAD: &str = "D:/models/Polaris-S14/range_cache/ac7f39b6146436528a1c856bec3e95865f29bca6f4c0d6861fdbe6085192e494.bin";
const DEFAULT_SHA256: &str = "029e3c5293b29cc426e21d87795e15efa4d363f27b2bc4a9e3aef7d79f047919";
const SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/s14_bf16_head.spv"));

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    vocab: u32,
    hidden_size: u32,
    batch: u32,
    groups_x: u32,
}

#[derive(Debug, Serialize)]
struct BatchResult {
    batch: usize,
    warm_iterations: usize,
    wall_ms_samples: Vec<f64>,
    median_wall_ms: f64,
    equivalent_head_tokens_per_second: f64,
    argmax_token_ids: Vec<usize>,
    max_logits: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct Report {
    format: &'static str,
    gpu: String,
    head_path: String,
    head_bytes: usize,
    head_sha256: String,
    upload_wall_ms: f64,
    cpu_reference_wall_ms: f64,
    cpu_reference_argmax: usize,
    cpu_reference_max_logit: f32,
    k1_gpu_cpu_argmax_equal: bool,
    k1_argmax_logit_abs_error: f32,
    batches: Vec<BatchResult>,
    claim_limit: &'static str,
}

fn main() -> Result<()> {
    let head_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_HEAD));
    let report_path = std::env::args().nth(2).map(PathBuf::from);
    let expected_sha =
        std::env::var("POLARIS_HEAD_SHA256").unwrap_or_else(|_| DEFAULT_SHA256.to_owned());
    validate_sha256(&expected_sha)?;

    let file = File::open(&head_path)
        .with_context(|| format!("open real BF16 head {}", head_path.display()))?;
    let metadata = file.metadata()?;
    if metadata.len() != HEAD_BYTES as u64 {
        bail!(
            "BF16 head byte drift: expected={} actual={}",
            HEAD_BYTES,
            metadata.len()
        );
    }
    let mmap = unsafe { Mmap::map(&file)? };
    let observed_sha = sha256_bytes(&mmap);
    if observed_sha != expected_sha {
        bail!("BF16 head SHA-256 drift");
    }

    let hidden8 = deterministic_hidden(8);
    let cpu_started = Instant::now();
    let cpu_logits = cpu_reference(&mmap, &hidden8[..HIDDEN]);
    let cpu_reference_wall_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
    let (cpu_argmax, cpu_max) = argmax(&cpu_logits)?;

    let ctx = VulkanContext::init()?;
    if !ctx.gpu_name.contains("5700") {
        bail!("real head gate requires RX 5700 XT; found {}", ctx.gpu_name);
    }
    let staging = GpuBuffer::new_staging(&ctx, HEAD_BYTES as u64)?;
    let head = GpuBuffer::new_vram(
        &ctx,
        HEAD_BYTES as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    unsafe { staging.write_at(0, &mmap) };
    let upload_wall_ms = upload(&ctx, &staging, &head, HEAD_BYTES as u64)?;
    staging.destroy(&ctx);

    let hidden_buffer = GpuBuffer::new(
        &ctx,
        (hidden8.len() * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;
    unsafe { hidden_buffer.write_at(0, f32_bytes(&hidden8)) };
    let logits_buffer = GpuBuffer::new(
        &ctx,
        (8 * VOCAB * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;
    let pipeline = ComputePipeline::new(&ctx, SHADER, 3, std::mem::size_of::<Push>() as u32)?;
    let binder = DescriptorBinder::new(
        &ctx,
        &pipeline,
        &[
            (&head, HEAD_BYTES as u64),
            (&hidden_buffer, (hidden8.len() * 4) as u64),
            (&logits_buffer, (8 * VOCAB * 4) as u64),
        ],
    )?;

    let pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_graphics),
            None,
        )?
    };
    let command = unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0]
    };

    let mut batches = Vec::new();
    for &batch in &[1usize, 4, 8] {
        dispatch(&ctx, command, &pipeline, &binder, batch)?;
        let mut samples = Vec::new();
        for _ in 0..7 {
            samples.push(dispatch(&ctx, command, &pipeline, &binder, batch)?);
        }
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        let gpu_logits = unsafe {
            std::slice::from_raw_parts(logits_buffer.mapped() as *const f32, batch * VOCAB)
        };
        let mut ids = Vec::with_capacity(batch);
        let mut maxima = Vec::with_capacity(batch);
        for row in gpu_logits.chunks_exact(VOCAB) {
            let (id, value) = argmax(row)?;
            ids.push(id);
            maxima.push(value);
        }
        batches.push(BatchResult {
            batch,
            warm_iterations: 7,
            wall_ms_samples: samples,
            median_wall_ms: median,
            equivalent_head_tokens_per_second: batch as f64 * 1000.0 / median,
            argmax_token_ids: ids,
            max_logits: maxima,
        });
    }

    let k1 = &batches[0];
    let report = Report {
        format: "polaris-s14-real-bf16-head-v1",
        gpu: ctx.gpu_name.clone(),
        head_path: canonical_utf8(&head_path)?,
        head_bytes: HEAD_BYTES,
        head_sha256: observed_sha,
        upload_wall_ms,
        cpu_reference_wall_ms,
        cpu_reference_argmax: cpu_argmax,
        cpu_reference_max_logit: cpu_max,
        k1_gpu_cpu_argmax_equal: k1.argmax_token_ids[0] == cpu_argmax,
        k1_argmax_logit_abs_error: (k1.max_logits[0] - cpu_max).abs(),
        batches,
        claim_limit: "Real fixed-revision BF16 full-vocabulary head only; not an end-to-end model token/s or quality claim.",
    };
    if !report.k1_gpu_cpu_argmax_equal || report.k1_argmax_logit_abs_error > 2.0e-3 {
        bail!(
            "GPU/CPU head mismatch: cpu={} gpu={} logit_abs_error={}",
            report.cpu_reference_argmax,
            report.batches[0].argmax_token_ids[0],
            report.k1_argmax_logit_abs_error
        );
    }
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    print!("{encoded}");
    if let Some(path) = report_path {
        std::fs::write(&path, encoded.as_bytes())?;
    }

    unsafe { ctx.device.destroy_command_pool(pool, None) };
    binder.destroy(&ctx);
    pipeline.destroy(&ctx);
    logits_buffer.destroy(&ctx);
    hidden_buffer.destroy(&ctx);
    head.destroy(&ctx);
    Ok(())
}

fn deterministic_hidden(batch: usize) -> Vec<f32> {
    let mut state = 0x7a17_5eed_d00d_beefu64;
    (0..batch * HIDDEN)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32;
            (unit - 0.5) * 0.25
        })
        .collect()
}

fn cpu_reference(head: &[u8], hidden: &[f32]) -> Vec<f32> {
    (0..VOCAB)
        .into_par_iter()
        .map(|row| {
            let base = row * HIDDEN * 2;
            let bytes = &head[base..base + HIDDEN * 2];
            let mut lanes = [0.0f32; 64];
            for lane in 0..64 {
                let mut acc = 0.0f32;
                for dim in (lane..HIDDEN).step_by(64) {
                    let offset = dim * 2;
                    let bf16 = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
                    acc += f32::from_bits((bf16 as u32) << 16) * hidden[dim];
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

fn upload(ctx: &VulkanContext, staging: &GpuBuffer, head: &GpuBuffer, bytes: u64) -> Result<f64> {
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
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_copy_buffer(
            command,
            staging.handle(),
            head.handle(),
            &[vk::BufferCopy::default().size(bytes)],
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
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
        Ok(elapsed)
    }
}

fn dispatch(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    binder: &DescriptorBinder,
    batch: usize,
) -> Result<f64> {
    let groups_x = MAX_GROUPS_X.min(VOCAB as u32);
    let groups_y = (VOCAB as u32).div_ceil(groups_x);
    let push = Push {
        vocab: VOCAB as u32,
        hidden_size: HIDDEN as u32,
        batch: batch as u32,
        groups_x,
    };
    unsafe {
        ctx.device
            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())?;
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device
            .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[binder.set],
            &[],
        );
        let bytes = std::slice::from_raw_parts(
            &push as *const Push as *const u8,
            std::mem::size_of::<Push>(),
        );
        ctx.device.cmd_push_constants(
            command,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytes,
        );
        ctx.device.cmd_dispatch(command, groups_x, groups_y, 1);
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
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        ctx.device.destroy_fence(fence, None);
        Ok(elapsed)
    }
}

fn argmax(values: &[f32]) -> Result<(usize, f32)> {
    values
        .iter()
        .copied()
        .enumerate()
        .try_fold(None, |best, (index, value)| {
            if !value.is_finite() {
                bail!("head produced non-finite logit at {index}");
            }
            Ok(match best {
                None => Some((index, value)),
                Some((_, current)) if value > current => Some((index, value)),
                Some(previous) => Some(previous),
            })
        })?
        .context("empty logits")
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("head SHA-256 must be 64 lowercase hex characters");
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

fn canonical_utf8(path: &Path) -> Result<String> {
    path.canonicalize()?
        .to_str()
        .map(str::to_owned)
        .context("head path is not UTF-8")
}
