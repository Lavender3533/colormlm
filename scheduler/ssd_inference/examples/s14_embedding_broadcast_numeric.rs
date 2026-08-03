//! FullDepth43 BOS embedding `[4096] -> [4,4096]` 的真实 Vulkan 数值门。

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    s14_embedding_broadcast::{
        validate_embedding_broadcast_status, S14EmbeddingBroadcastDispatch,
        S14EmbeddingBroadcastPipeline, S14EmbeddingBroadcastShape,
        S14_EMBEDDING_STATUS_NON_FINITE_INPUT, S14_EMBEDDING_STREAMS,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const HIDDEN: usize = 4096;

fn main() -> Result<()> {
    let invalid_shapes_rejected = [0, 3, 4098]
        .into_iter()
        .all(|hidden| S14EmbeddingBroadcastShape::new(hidden).is_err());
    if !invalid_shapes_rejected {
        bail!("S14 embedding broadcast accepted an invalid shape");
    }

    let ctx = VulkanContext::init()?;
    let pipeline = S14EmbeddingBroadcastPipeline::new(&ctx)?;
    let shape = S14EmbeddingBroadcastShape::new(HIDDEN as u32)?;
    let row: Vec<u16> = (0..HIDDEN)
        .map(|index| to_bf16_bits(((index * 29 % 97) as f32 - 48.0) / 64.0))
        .collect();
    let expected: Vec<u16> = (0..S14_EMBEDDING_STREAMS)
        .flat_map(|_| row.iter().copied())
        .collect();

    let row_buffer = host_buffer(&ctx, shape.row_bf16_bytes())?;
    let output_buffer = host_buffer(&ctx, shape.output_bf16_bytes())?;
    let status_buffer = host_buffer(&ctx, shape.status_bytes())?;
    unsafe {
        row_buffer.write_at(0, bytemuck::cast_slice(&row));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(&ctx, shape, &row_buffer, &output_buffer, &status_buffer)?;
    if pipeline
        .bind(&ctx, shape, &row_buffer, &output_buffer, &row_buffer)
        .is_ok()
    {
        bail!("S14 embedding broadcast accepted aliased input/status");
    }

    let started = Instant::now();
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    validate_embedding_broadcast_status(mapped_u32(&status_buffer))?;
    let output = mapped_u16(&output_buffer, expected.len());
    let mismatches = output
        .iter()
        .zip(&expected)
        .filter(|(actual, expected)| actual != expected)
        .count();
    if mismatches != 0 {
        bail!("S14 embedding broadcast BF16 mismatch count={mismatches}");
    }

    // 非有限输入必须清零整块输出，并让错误在同一 candidate 内保持 sticky。
    let mut invalid_row = row.clone();
    invalid_row[777] = 0x7f80;
    let sentinel = vec![0x3f80u16; expected.len()];
    unsafe {
        row_buffer.write_at(0, bytemuck::cast_slice(&invalid_row));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    let invalid_status = mapped_u32(&status_buffer);
    if invalid_status != S14_EMBEDDING_STATUS_NON_FINITE_INPUT
        || validate_embedding_broadcast_status(invalid_status).is_ok()
    {
        bail!("S14 embedding broadcast NaN/Inf status drift: 0x{invalid_status:08x}");
    }
    if mapped_u16(&output_buffer, expected.len())
        .iter()
        .any(|value| *value != 0)
    {
        bail!("S14 embedding broadcast leaked output after invalid input");
    }

    unsafe {
        row_buffer.write_at(0, bytemuck::cast_slice(&row));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
    }
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    if mapped_u32(&status_buffer) != S14_EMBEDDING_STATUS_NON_FINITE_INPUT
        || mapped_u16(&output_buffer, expected.len())
            .iter()
            .any(|value| *value != 0)
    {
        bail!("S14 embedding broadcast failure status was not sticky");
    }

    println!(
        "status=pass gpu=\"{}\" hidden={HIDDEN} streams={} outputs={} bf16_mismatches=0 invalid_shape_rejected=3 alias_rejected=1 invalid_status=0x{invalid_status:08x} invalid_nonzero_outputs=0 sticky=pass wall_ms={wall_ms:.4}",
        ctx.gpu_name,
        S14_EMBEDDING_STREAMS,
        expected.len()
    );

    dispatch.binder.destroy(&ctx);
    status_buffer.destroy(&ctx);
    output_buffer.destroy(&ctx);
    row_buffer.destroy(&ctx);
    pipeline.destroy(&ctx);
    Ok(())
}

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14EmbeddingBroadcastPipeline,
    dispatch: &S14EmbeddingBroadcastDispatch,
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

fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    if exponent == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
