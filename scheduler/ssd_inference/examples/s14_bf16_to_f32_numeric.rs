//! FullDepth43 position0 attention BF16→grouped-wo_a F32 的真实 Vulkan 数值门。

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    s14_bf16_to_f32::{
        validate_bf16_to_f32_status, S14Bf16ToF32Dispatch, S14Bf16ToF32Pipeline, S14Bf16ToF32Shape,
        S14_BF16_TO_F32_STATUS_NON_FINITE_INPUT,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const SCALARS: usize = 64 * 512;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipeline = S14Bf16ToF32Pipeline::new(&ctx)?;
    let shape = S14Bf16ToF32Shape::new(SCALARS as u32)?;
    let input: Vec<u16> = (0..SCALARS)
        .map(|index| to_bf16_bits(((index * 17 % 113) as f32 - 56.0) / 41.0))
        .collect();
    let expected: Vec<f32> = input.iter().map(|value| from_bf16(*value)).collect();

    let input_buffer = host_buffer(&ctx, shape.input_bf16_bytes())?;
    let output_buffer = host_buffer(&ctx, shape.output_f32_bytes())?;
    let status_buffer = host_buffer(&ctx, shape.status_bytes())?;
    unsafe {
        input_buffer.write_at(0, bytemuck::cast_slice(&input));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(&ctx, shape, &input_buffer, &output_buffer, &status_buffer)?;

    let started = Instant::now();
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    validate_bf16_to_f32_status(mapped_u32(&status_buffer))?;
    let output = mapped_f32(&output_buffer, SCALARS);
    let mismatches = output
        .iter()
        .zip(&expected)
        .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
        .count();
    if mismatches != 0 {
        bail!("S14 BF16->F32 bit mismatch count={mismatches}");
    }

    let mut invalid_input = input.clone();
    invalid_input[777] = 0x7f80;
    let sentinel = vec![1.0f32; SCALARS];
    unsafe {
        input_buffer.write_at(0, bytemuck::cast_slice(&invalid_input));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    dispatch_once(&ctx, &pipeline, &dispatch)?;
    let invalid_status = mapped_u32(&status_buffer);
    if invalid_status != S14_BF16_TO_F32_STATUS_NON_FINITE_INPUT
        || validate_bf16_to_f32_status(invalid_status).is_ok()
    {
        bail!("S14 BF16->F32 invalid status drift: 0x{invalid_status:08x}");
    }
    if mapped_f32(&output_buffer, SCALARS)
        .iter()
        .any(|value| value.to_bits() != 0)
    {
        bail!("S14 BF16->F32 leaked output after invalid input");
    }

    println!(
        "status=pass gpu=\"{}\" scalars={SCALARS} f32_bit_mismatches=0 invalid_status=0x{invalid_status:08x} invalid_nonzero_outputs=0 wall_ms={wall_ms:.4}",
        ctx.gpu_name
    );

    dispatch.binder.destroy(&ctx);
    status_buffer.destroy(&ctx);
    output_buffer.destroy(&ctx);
    input_buffer.destroy(&ctx);
    pipeline.destroy(&ctx);
    Ok(())
}

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14Bf16ToF32Pipeline,
    dispatch: &S14Bf16ToF32Dispatch,
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

fn mapped_f32(buffer: &GpuBuffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const f32, len).to_vec() }
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
