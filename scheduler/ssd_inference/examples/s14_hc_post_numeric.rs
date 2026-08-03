//! DeepSeek-V4 单 token mHC-post 的秒级 Vulkan 数值门。
//!
//! 正向门要求生产尺寸 4×4096 的 BF16 输出逐元素等于 CPU `hc_post` oracle；
//! 负向门验证非法 shape 被 host 拒绝、NaN 使整块输出清零且状态禁止发布。

use anyhow::{bail, Result};
use ash::vk;
use polaris_s14_runner::{bf16_round_trip_slice, hc_post};
use ssd_inference::{
    s14_hc_post::{
        validate_hc_post_status, S14HcPostDispatch, S14HcPostPipeline, S14HcPostShape,
        S14_HC_POST_STATUS_NON_FINITE_INPUT, S14_HC_POST_STREAMS,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const HIDDEN: usize = 4096;

fn main() -> Result<()> {
    let invalid_shapes_rejected = [0, 3, 4098]
        .into_iter()
        .all(|hidden| S14HcPostShape::new(hidden).is_err());
    if !invalid_shapes_rejected {
        bail!("S14 HC-post accepted an invalid shape");
    }

    let ctx = VulkanContext::init()?;
    let pipeline = S14HcPostPipeline::new(&ctx)?;
    let shape = S14HcPostShape::new(HIDDEN as u32)?;
    let streams = S14_HC_POST_STREAMS as usize;

    let branch_bits: Vec<u16> = (0..HIDDEN)
        .map(|index| to_bf16_bits(((index * 17 % 61) as f32 - 30.0) / 64.0))
        .collect();
    let residual_bits: Vec<u16> = (0..streams * HIDDEN)
        .map(|index| to_bf16_bits(((index * 29 % 79) as f32 - 39.0) / 96.0))
        .collect();
    let branch: Vec<f32> = branch_bits.iter().map(|value| from_bf16(*value)).collect();
    let residual: Vec<f32> = residual_bits
        .iter()
        .map(|value| from_bf16(*value))
        .collect();
    let mut post = vec![0.5f32, -0.75, 1.25, 1.5];
    let comb: Vec<f32> = (0..streams * streams)
        .map(|index| ((index * 7 % 13) as f32 - 4.0) / 32.0)
        .collect();
    let reference_f32 = hc_post(&branch, &residual, &post, &comb, 1, HIDDEN)?;
    let reference_bits: Vec<u16> = bf16_round_trip_slice(&reference_f32)?
        .iter()
        .map(|value| (value.to_bits() >> 16) as u16)
        .collect();

    let branch_buffer = host_buffer(&ctx, shape.branch_bf16_bytes()?)?;
    let residual_buffer = host_buffer(&ctx, shape.residual_bf16_bytes()?)?;
    let post_buffer = host_buffer(&ctx, shape.post_f32_bytes())?;
    let comb_buffer = host_buffer(&ctx, shape.comb_f32_bytes())?;
    let output_buffer = host_buffer(&ctx, shape.output_bf16_bytes()?)?;
    let status_buffer = host_buffer(&ctx, shape.status_bytes())?;
    unsafe {
        branch_buffer.write_at(0, bytemuck::cast_slice(&branch_bits));
        residual_buffer.write_at(0, bytemuck::cast_slice(&residual_bits));
        post_buffer.write_at(0, bytemuck::cast_slice(&post));
        comb_buffer.write_at(0, bytemuck::cast_slice(&comb));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(
        &ctx,
        shape,
        &branch_buffer,
        &residual_buffer,
        &post_buffer,
        &comb_buffer,
        &output_buffer,
        &status_buffer,
    )?;
    if pipeline
        .bind(
            &ctx,
            shape,
            &branch_buffer,
            &residual_buffer,
            &post_buffer,
            &comb_buffer,
            &residual_buffer,
            &status_buffer,
        )
        .is_ok()
    {
        bail!("S14 HC-post accepted output/residual aliasing");
    }

    let started = Instant::now();
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    let valid_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let status = mapped_u32(&status_buffer);
    validate_hc_post_status(status)?;
    let output = mapped_u16(&output_buffer, reference_bits.len());
    let mismatches = output
        .iter()
        .zip(&reference_bits)
        .filter(|(actual, expected)| actual != expected)
        .count();
    if mismatches != 0 {
        let first = output
            .iter()
            .zip(&reference_bits)
            .position(|(actual, expected)| actual != expected)
            .unwrap();
        bail!(
            "S14 HC-post BF16 mismatch count={mismatches}, first={first}, actual=0x{:04x}, expected=0x{:04x}",
            output[first],
            reference_bits[first]
        );
    }

    // NaN 负向门：禁止保留任何局部旧值或局部新值。
    post[2] = f32::NAN;
    let sentinel = vec![0x3f80u16; reference_bits.len()];
    unsafe {
        post_buffer.write_at(0, bytemuck::cast_slice(&post));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    let invalid_status = mapped_u32(&status_buffer);
    if invalid_status != S14_HC_POST_STATUS_NON_FINITE_INPUT
        || validate_hc_post_status(invalid_status).is_ok()
    {
        bail!("S14 HC-post NaN status did not fail closed: 0x{invalid_status:08x}");
    }
    let invalid_output = mapped_u16(&output_buffer, reference_bits.len());
    let nonzero_after_reject = invalid_output.iter().filter(|value| **value != 0).count();
    if nonzero_after_reject != 0 {
        bail!("S14 HC-post NaN path leaked {nonzero_after_reject} non-zero outputs");
    }

    // sticky 负向门：同一 WholeTokenCandidate 内后续合法 dispatch 不能洗掉失败。
    post[2] = 1.25;
    unsafe {
        post_buffer.write_at(0, bytemuck::cast_slice(&post));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
    }
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    let sticky_status = mapped_u32(&status_buffer);
    let sticky_output = mapped_u16(&output_buffer, reference_bits.len());
    if sticky_status != S14_HC_POST_STATUS_NON_FINITE_INPUT
        || sticky_output.iter().any(|value| *value != 0)
    {
        bail!("S14 HC-post failure status was not sticky: 0x{sticky_status:08x}");
    }

    println!(
        "status=pass hidden={HIDDEN} streams={streams} outputs={} bf16_mismatches=0 valid_status=0 invalid_shape_rejected=3 alias_rejected=1 nan_status=0x{invalid_status:08x} nan_nonzero_outputs=0 sticky_status=0x{sticky_status:08x} wall_ms={valid_wall_ms:.4}",
        reference_bits.len()
    );

    dispatch.binder.destroy(&ctx);
    status_buffer.destroy(&ctx);
    output_buffer.destroy(&ctx);
    comb_buffer.destroy(&ctx);
    post_buffer.destroy(&ctx);
    residual_buffer.destroy(&ctx);
    branch_buffer.destroy(&ctx);
    pipeline.destroy(&ctx);
    Ok(())
}

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14HcPostPipeline,
    dispatch: &S14HcPostDispatch,
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

fn mapped_u16(buffer: &GpuBuffer, len: usize) -> Vec<u16> {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const u16, len).to_vec() }
}

fn mapped_u32(buffer: &GpuBuffer) -> u32 {
    unsafe { *(buffer.mapped() as *const u32) }
}

fn from_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    if exponent == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
