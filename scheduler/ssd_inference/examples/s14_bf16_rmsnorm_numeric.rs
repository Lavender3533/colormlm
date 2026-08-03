//! FullDepth43 通用 BF16 RMSNorm 的生产尺寸 Vulkan 数值门。

use anyhow::{bail, Result};
use ash::vk;
use polaris_s14_runner::{bf16_round_trip_slice, official_rms_norm};
use ssd_inference::{
    s14_bf16_rmsnorm::{
        validate_bf16_rmsnorm_status, S14Bf16RmsNormDispatch, S14Bf16RmsNormPipeline,
        S14Bf16RmsNormShape, S14_BF16_RMSNORM_STATUS_NON_FINITE_INPUT,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const EPSILON: f32 = 1.0e-6;

fn main() -> Result<()> {
    let invalid_shapes = [(0, 512), (65, 512), (1, 0), (1, 3), (1, 4098)];
    if !invalid_shapes
        .into_iter()
        .all(|(rows, hidden)| S14Bf16RmsNormShape::new(rows, hidden).is_err())
    {
        bail!("S14 BF16 RMSNorm accepted an invalid shape");
    }

    let ctx = VulkanContext::init()?;
    let pipeline = S14Bf16RmsNormPipeline::new(&ctx)?;
    let (hidden_ms, hidden_alias_rejected) = run_valid_case(&ctx, &pipeline, 1, 4096, false)?;
    let (q_heads_ms, q_alias_rejected) = run_valid_case(&ctx, &pipeline, 64, 512, true)?;
    run_nan_and_sticky_case(&ctx, &pipeline)?;

    println!(
        "status=pass hidden_shape=1x4096 hidden_bf16_mismatches=0 hidden_wall_ms={hidden_ms:.4} q_shape=64x512 q_bf16_mismatches=0 q_wall_ms={q_heads_ms:.4} invalid_shape_rejected=5 alias_rejected={} nan_status=0x00000001 nan_nonzero_outputs=0 sticky_status=0x00000001",
        u32::from(hidden_alias_rejected) + u32::from(q_alias_rejected)
    );
    pipeline.destroy(&ctx);
    Ok(())
}

fn run_valid_case(
    ctx: &VulkanContext,
    pipeline: &S14Bf16RmsNormPipeline,
    rows: usize,
    hidden: usize,
    unit_weight: bool,
) -> Result<(f64, bool)> {
    let shape = S14Bf16RmsNormShape::new(rows as u32, hidden as u32)?;
    let input_bits: Vec<u16> = (0..rows * hidden)
        .map(|index| to_bf16_bits(((index * 17 % 97) as f32 - 48.0) / 64.0))
        .collect();
    let weight_bits: Vec<u16> = if unit_weight {
        vec![to_bf16_bits(1.0); hidden]
    } else {
        (0..hidden)
            .map(|index| to_bf16_bits(0.75 + (index % 19) as f32 / 64.0))
            .collect()
    };
    let input: Vec<f32> = input_bits.iter().map(|value| from_bf16(*value)).collect();
    let weight: Vec<f32> = weight_bits.iter().map(|value| from_bf16(*value)).collect();
    let reference = official_rms_norm(&input, rows, hidden, &weight)?;
    let reference_bits: Vec<u16> = bf16_round_trip_slice(&reference)?
        .iter()
        .map(|value| (value.to_bits() >> 16) as u16)
        .collect();

    let input_buffer = host_buffer(ctx, shape.input_bf16_bytes()?)?;
    let weight_buffer = host_buffer(ctx, shape.weight_bf16_bytes()?)?;
    let inverse_rms_buffer = host_buffer(ctx, shape.inverse_rms_f32_bytes()?)?;
    let output_buffer = host_buffer(ctx, shape.output_bf16_bytes()?)?;
    let status_buffer = host_buffer(ctx, shape.status_bytes())?;
    unsafe {
        input_buffer.write_at(0, bytemuck::cast_slice(&input_bits));
        weight_buffer.write_at(0, bytemuck::cast_slice(&weight_bits));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(
        ctx,
        shape,
        EPSILON,
        &input_buffer,
        &weight_buffer,
        &inverse_rms_buffer,
        &output_buffer,
        &status_buffer,
    )?;
    let alias_rejected = pipeline
        .bind(
            ctx,
            shape,
            EPSILON,
            &input_buffer,
            &weight_buffer,
            &inverse_rms_buffer,
            &input_buffer,
            &status_buffer,
        )
        .is_err();
    if !alias_rejected {
        bail!("S14 BF16 RMSNorm accepted input/output aliasing");
    }

    let started = Instant::now();
    dispatch_once(ctx, pipeline, &dispatch)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    validate_bf16_rmsnorm_status(mapped_u32(&status_buffer))?;
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
            "S14 BF16 RMSNorm {rows}x{hidden} mismatch count={mismatches}, first={first}, actual=0x{:04x}, expected=0x{:04x}",
            output[first],
            reference_bits[first]
        );
    }

    dispatch.binder.destroy(ctx);
    status_buffer.destroy(ctx);
    output_buffer.destroy(ctx);
    inverse_rms_buffer.destroy(ctx);
    weight_buffer.destroy(ctx);
    input_buffer.destroy(ctx);
    Ok((wall_ms, alias_rejected))
}

fn run_nan_and_sticky_case(ctx: &VulkanContext, pipeline: &S14Bf16RmsNormPipeline) -> Result<()> {
    let rows = 64usize;
    let hidden = 512usize;
    let shape = S14Bf16RmsNormShape::new(rows as u32, hidden as u32)?;
    let mut input_bits = vec![to_bf16_bits(0.25); rows * hidden];
    input_bits[123] = 0x7fc1;
    let weight_bits = vec![to_bf16_bits(1.0); hidden];
    let sentinel = vec![0x3f80u16; rows * hidden];

    let input_buffer = host_buffer(ctx, shape.input_bf16_bytes()?)?;
    let weight_buffer = host_buffer(ctx, shape.weight_bf16_bytes()?)?;
    let inverse_rms_buffer = host_buffer(ctx, shape.inverse_rms_f32_bytes()?)?;
    let output_buffer = host_buffer(ctx, shape.output_bf16_bytes()?)?;
    let status_buffer = host_buffer(ctx, shape.status_bytes())?;
    unsafe {
        input_buffer.write_at(0, bytemuck::cast_slice(&input_bits));
        weight_buffer.write_at(0, bytemuck::cast_slice(&weight_bits));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(
        ctx,
        shape,
        EPSILON,
        &input_buffer,
        &weight_buffer,
        &inverse_rms_buffer,
        &output_buffer,
        &status_buffer,
    )?;
    dispatch_once(ctx, pipeline, &dispatch)?;
    let nan_status = mapped_u32(&status_buffer);
    let rejected_output = mapped_u16(&output_buffer, rows * hidden);
    if nan_status != S14_BF16_RMSNORM_STATUS_NON_FINITE_INPUT
        || validate_bf16_rmsnorm_status(nan_status).is_ok()
        || rejected_output.iter().any(|value| *value != 0)
    {
        bail!("S14 BF16 RMSNorm NaN path did not fail closed: 0x{nan_status:08x}");
    }

    input_bits[123] = to_bf16_bits(0.25);
    unsafe {
        input_buffer.write_at(0, bytemuck::cast_slice(&input_bits));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
    }
    dispatch_once(ctx, pipeline, &dispatch)?;
    let sticky_status = mapped_u32(&status_buffer);
    let sticky_output = mapped_u16(&output_buffer, rows * hidden);
    if sticky_status != S14_BF16_RMSNORM_STATUS_NON_FINITE_INPUT
        || sticky_output.iter().any(|value| *value != 0)
    {
        bail!("S14 BF16 RMSNorm failure status was not sticky: 0x{sticky_status:08x}");
    }

    dispatch.binder.destroy(ctx);
    status_buffer.destroy(ctx);
    output_buffer.destroy(ctx);
    inverse_rms_buffer.destroy(ctx);
    weight_buffer.destroy(ctx);
    input_buffer.destroy(ctx);
    Ok(())
}

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14Bf16RmsNormPipeline,
    dispatch: &S14Bf16RmsNormDispatch,
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
