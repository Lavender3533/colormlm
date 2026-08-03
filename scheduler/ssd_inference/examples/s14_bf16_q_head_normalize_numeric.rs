//! 冻结真实 L0 query 上的 PyTorch-BF16 Q-head 归一化 Vulkan 数值门。

use anyhow::{bail, Context, Result};
use ash::vk;
use sha2::{Digest, Sha256};
use ssd_inference::{
    s14_bf16_q_head_normalize::{
        validate_bf16_q_head_normalize_status, S14Bf16QHeadNormalizePipeline,
        S14Bf16QHeadNormalizeShape,
    },
    GpuBuffer, VulkanContext,
};
use std::{env, fs, path::PathBuf, time::Instant};

const EPSILON: f32 = 1.0e-6;
const RAW_SHA256: &str = "84e23bebca0472450ee4496f1c97c45f00ae8a49f1dbb07f83f15238cc835bdd";
const FINAL_SHA256: &str = "d9d3d36187cd03577c408ab685f1cb6e20edec0152448f0e85c9764a771b5b94";

fn main() -> Result<()> {
    let root = env::var_os("POLARIS_L0_STAGE_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("D:/project/大模型ssd化/.tmp-polaris-runs/l0-stage-reference-20260802")
        });
    let raw = read_frozen(&root.join("query_raw.bf16le.bin"), RAW_SHA256)?;
    let expected = read_frozen(&root.join("query_final.bf16le.bin"), FINAL_SHA256)?;
    let shape = S14Bf16QHeadNormalizeShape::new();
    if raw.len() as u64 != shape.input_bf16_bytes()
        || expected.len() as u64 != shape.output_bf16_bytes()
    {
        bail!(
            "frozen Q-head payload size drifted: raw={} expected={}",
            raw.len(),
            expected.len()
        );
    }

    let ctx = VulkanContext::init()?;
    let pipeline = S14Bf16QHeadNormalizePipeline::new(&ctx)?;
    let input = host_buffer(&ctx, shape.input_bf16_bytes())?;
    let inverse = host_buffer(&ctx, shape.inverse_rms_f32_bytes())?;
    let output = host_buffer(&ctx, shape.output_bf16_bytes())?;
    let status = host_buffer(&ctx, shape.status_bytes())?;
    unsafe {
        input.write_at(0, &raw);
        status.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(&ctx, EPSILON, &input, &inverse, &output, &status)?;
    if pipeline
        .bind(&ctx, EPSILON, &input, &inverse, &input, &status)
        .is_ok()
    {
        bail!("Q-head normalize accepted input/output aliasing");
    }

    let started = Instant::now();
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let status_code = unsafe { *(status.mapped() as *const u32) };
    validate_bf16_q_head_normalize_status(status_code)?;
    let actual =
        unsafe { std::slice::from_raw_parts(output.mapped(), shape.output_bf16_bytes() as usize) };
    let mismatches = actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .filter(|(left, right)| left != right)
        .count();
    if mismatches != 0 {
        let first = actual
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .position(|(left, right)| left != right)
            .expect("nonzero mismatch count");
        bail!(
            "Q-head PyTorch-BF16 mismatch count={mismatches}/{} first={first} actual=0x{:04x} expected=0x{:04x}",
            shape.scalar_count(),
            u16::from_le_bytes(actual[first * 2..first * 2 + 2].try_into()?),
            u16::from_le_bytes(expected[first * 2..first * 2 + 2].try_into()?),
        );
    }

    println!(
        "status=pass shape={}x{} mismatches=0/{} raw_sha={} final_sha={} wall_ms={wall_ms:.4}",
        shape.rows(),
        shape.hidden(),
        shape.scalar_count(),
        RAW_SHA256,
        FINAL_SHA256,
    );

    dispatch.binder.destroy(&ctx);
    status.destroy(&ctx);
    output.destroy(&ctx);
    inverse.destroy(&ctx);
    input.destroy(&ctx);
    pipeline.destroy(&ctx);
    Ok(())
}

fn read_frozen(path: &PathBuf, expected_sha: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let actual_sha = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha != expected_sha {
        bail!(
            "frozen Q-head SHA drifted: path={} actual={} expected={}",
            path.display(),
            actual_sha,
            expected_sha
        );
    }
    Ok(bytes)
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

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14Bf16QHeadNormalizePipeline,
    dispatch: &ssd_inference::s14_bf16_q_head_normalize::S14Bf16QHeadNormalizeDispatch,
) -> Result<()> {
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
    let fence = unsafe {
        ctx.device
            .create_fence(&vk::FenceCreateInfo::default(), None)?
    };
    unsafe {
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        pipeline.cmd(ctx, command, dispatch);
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    Ok(())
}
