//! FullDepth43 group64/group128 E4M3FN activation QDQ 的 Vulkan 数值门。

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_e4m3_qdq::{
        validate_e4m3_qdq_status, S14E4m3QdqDispatch, S14E4m3QdqPipeline, S14E4m3QdqShape,
        S14_E4M3_QDQ_STATUS_NON_FINITE_INPUT,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

fn main() -> Result<()> {
    let invalid = [
        (0, 448, 64),
        (65, 448, 64),
        (1, 447, 64),
        (1, 448, 32),
        (1, 8256, 128),
    ];
    if !invalid
        .into_iter()
        .all(|(rows, hidden, group)| S14E4m3QdqShape::new(rows, hidden, group).is_err())
    {
        bail!("S14 E4M3 QDQ accepted an invalid shape");
    }

    let ctx = VulkanContext::init()?;
    let pipeline = S14E4m3QdqPipeline::new(&ctx)?;
    let group64_ms = run_valid_case(&ctx, &pipeline, 1, 448, 64)?;
    let group128_ms = run_valid_case(&ctx, &pipeline, 1, 8192, 128)?;
    let single_arena_ms = run_single_arena_case(&ctx, &pipeline)?;
    run_nan_and_sticky_case(&ctx, &pipeline)?;
    println!(
        "status=pass group64_shape=1x448 group64_f32_bit_mismatches=0 group64_wall_ms={group64_ms:.4} group128_shape=1x8192 group128_f32_bit_mismatches=0 group128_wall_ms={group128_ms:.4} single_arena_f32_bit_mismatches=0 single_arena_wall_ms={single_arena_ms:.4} invalid_shape_rejected=5 alias_rejected=3 nan_status=0x00000001 nan_nonzero_outputs=0 sticky_status=0x00000001"
    );
    pipeline.destroy(&ctx);
    Ok(())
}

fn run_single_arena_case(ctx: &VulkanContext, pipeline: &S14E4m3QdqPipeline) -> Result<f64> {
    let shape = S14E4m3QdqShape::new(1, 8192, 128)?;
    let input_bits: Vec<u16> = (0..8192)
        .map(|index| to_bf16_bits(((index * 41 % 1021) as f32 - 510.0) / 37.0))
        .collect();
    let input: Vec<f32> = input_bits.iter().map(|value| from_bf16(*value)).collect();
    let reference = activation_qdq_reference(&input, 8192, 128)?;
    let alignment = unsafe {
        ctx.instance
            .get_physical_device_properties(ctx.physical)
            .limits
            .min_storage_buffer_offset_alignment
            .max(256)
    };
    let input_offset = 0;
    let scales_offset = align_up(input_offset + shape.input_bf16_bytes()?, alignment)?;
    let output_offset = align_up(scales_offset + shape.scale_f32_bytes()?, alignment)?;
    let status_offset = align_up(output_offset + shape.output_f32_bytes()?, alignment)?;
    let arena_bytes = align_up(status_offset + shape.status_bytes(), alignment)?;
    let arena = host_buffer(ctx, arena_bytes)?;
    unsafe {
        arena.write_at(input_offset as usize, bytemuck::cast_slice(&input_bits));
        arena.write_at(status_offset as usize, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind_slices(
        ctx,
        shape,
        StorageBufferSlice {
            buffer: &arena,
            offset: input_offset,
        },
        StorageBufferSlice {
            buffer: &arena,
            offset: scales_offset,
        },
        StorageBufferSlice {
            buffer: &arena,
            offset: output_offset,
        },
        StorageBufferSlice {
            buffer: &arena,
            offset: status_offset,
        },
    )?;
    if pipeline
        .bind_slices(
            ctx,
            shape,
            StorageBufferSlice {
                buffer: &arena,
                offset: input_offset,
            },
            StorageBufferSlice {
                buffer: &arena,
                offset: scales_offset,
            },
            StorageBufferSlice {
                buffer: &arena,
                offset: input_offset,
            },
            StorageBufferSlice {
                buffer: &arena,
                offset: status_offset,
            },
        )
        .is_ok()
    {
        bail!("S14 E4M3 QDQ accepted overlapping arena slices");
    }
    let started = Instant::now();
    dispatch_once(ctx, pipeline, &dispatch)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let status =
        unsafe { *((arena.mapped() as *const u8).add(status_offset as usize) as *const u32) };
    validate_e4m3_qdq_status(status)?;
    let output = unsafe {
        std::slice::from_raw_parts(
            (arena.mapped() as *const u8).add(output_offset as usize) as *const f32,
            reference.len(),
        )
    };
    if let Some(first) = output
        .iter()
        .zip(&reference)
        .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        bail!(
            "S14 E4M3 QDQ single-arena mismatch first={first}, actual={:.9e}, expected={:.9e}",
            output[first],
            reference[first]
        );
    }
    dispatch.binder.destroy(ctx);
    arena.destroy(ctx);
    Ok(wall_ms)
}

fn run_valid_case(
    ctx: &VulkanContext,
    pipeline: &S14E4m3QdqPipeline,
    rows: usize,
    hidden: usize,
    group_size: usize,
) -> Result<f64> {
    let shape = S14E4m3QdqShape::new(rows as u32, hidden as u32, group_size as u32)?;
    let input_bits: Vec<u16> = (0..rows * hidden)
        .map(|index| {
            let value = match index % group_size {
                0 => 1.0,
                1 => 1.0625,
                2 => 1.1875,
                3 => -1.0625,
                4 => -1.1875,
                _ => ((index * 37 % 997) as f32 - 498.0) / 32.0,
            };
            to_bf16_bits(value)
        })
        .collect();
    let input: Vec<f32> = input_bits.iter().map(|value| from_bf16(*value)).collect();
    let reference = activation_qdq_reference(&input, hidden, group_size)?;

    let input_buffer = host_buffer(ctx, shape.input_bf16_bytes()?)?;
    let scale_buffer = host_buffer(ctx, shape.scale_f32_bytes()?)?;
    let output_buffer = host_buffer(ctx, shape.output_f32_bytes()?)?;
    let status_buffer = host_buffer(ctx, shape.status_bytes())?;
    unsafe {
        input_buffer.write_at(0, bytemuck::cast_slice(&input_bits));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(
        ctx,
        shape,
        &input_buffer,
        &scale_buffer,
        &output_buffer,
        &status_buffer,
    )?;
    if pipeline
        .bind(
            ctx,
            shape,
            &input_buffer,
            &scale_buffer,
            &input_buffer,
            &status_buffer,
        )
        .is_ok()
    {
        bail!("S14 E4M3 QDQ accepted input/output aliasing");
    }
    let started = Instant::now();
    dispatch_once(ctx, pipeline, &dispatch)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    validate_e4m3_qdq_status(mapped_u32(&status_buffer))?;
    let output = mapped_f32(&output_buffer, reference.len());
    let mismatches = output
        .iter()
        .zip(&reference)
        .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
        .count();
    if mismatches != 0 {
        let first = output
            .iter()
            .zip(&reference)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits())
            .unwrap();
        bail!(
            "S14 E4M3 QDQ group{group_size} mismatch count={mismatches}, first={first}, actual={:.9e}/0x{:08x}, expected={:.9e}/0x{:08x}",
            output[first],
            output[first].to_bits(),
            reference[first],
            reference[first].to_bits()
        );
    }
    dispatch.binder.destroy(ctx);
    status_buffer.destroy(ctx);
    output_buffer.destroy(ctx);
    scale_buffer.destroy(ctx);
    input_buffer.destroy(ctx);
    Ok(wall_ms)
}

fn run_nan_and_sticky_case(ctx: &VulkanContext, pipeline: &S14E4m3QdqPipeline) -> Result<()> {
    let shape = S14E4m3QdqShape::new(1, 448, 64)?;
    let mut input_bits = vec![to_bf16_bits(0.25); 448];
    input_bits[123] = 0x7fc1;
    let sentinel = vec![1.0f32; 448];
    let input_buffer = host_buffer(ctx, shape.input_bf16_bytes()?)?;
    let scale_buffer = host_buffer(ctx, shape.scale_f32_bytes()?)?;
    let output_buffer = host_buffer(ctx, shape.output_f32_bytes()?)?;
    let status_buffer = host_buffer(ctx, shape.status_bytes())?;
    unsafe {
        input_buffer.write_at(0, bytemuck::cast_slice(&input_bits));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(
        ctx,
        shape,
        &input_buffer,
        &scale_buffer,
        &output_buffer,
        &status_buffer,
    )?;
    dispatch_once(ctx, pipeline, &dispatch)?;
    let nan_status = mapped_u32(&status_buffer);
    if nan_status != S14_E4M3_QDQ_STATUS_NON_FINITE_INPUT
        || validate_e4m3_qdq_status(nan_status).is_ok()
        || mapped_f32(&output_buffer, 448)
            .iter()
            .any(|value| value.to_bits() != 0)
    {
        bail!("S14 E4M3 QDQ NaN path did not fail closed: 0x{nan_status:08x}");
    }

    input_bits[123] = to_bf16_bits(0.25);
    unsafe {
        input_buffer.write_at(0, bytemuck::cast_slice(&input_bits));
        output_buffer.write_at(0, bytemuck::cast_slice(&sentinel));
    }
    dispatch_once(ctx, pipeline, &dispatch)?;
    let sticky_status = mapped_u32(&status_buffer);
    if sticky_status != S14_E4M3_QDQ_STATUS_NON_FINITE_INPUT
        || mapped_f32(&output_buffer, 448)
            .iter()
            .any(|value| value.to_bits() != 0)
    {
        bail!("S14 E4M3 QDQ status was not sticky: 0x{sticky_status:08x}");
    }
    dispatch.binder.destroy(ctx);
    status_buffer.destroy(ctx);
    output_buffer.destroy(ctx);
    scale_buffer.destroy(ctx);
    input_buffer.destroy(ctx);
    Ok(())
}

fn activation_qdq_reference(values: &[f32], hidden: usize, group_size: usize) -> Result<Vec<f32>> {
    if values.is_empty()
        || hidden == 0
        || hidden % group_size != 0
        || values.len() % hidden != 0
        || values.iter().any(|value| !value.is_finite())
    {
        bail!("invalid activation QDQ reference input");
    }
    let mut output = Vec::with_capacity(values.len());
    for row in values.chunks_exact(hidden) {
        for block in row.chunks_exact(group_size) {
            let amax = block
                .iter()
                .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
                .max(1.0e-4);
            let exponent = (amax / 448.0).log2().ceil() as i32;
            let scale = 2.0f32.powi(exponent);
            if !scale.is_finite() || scale <= 0.0 {
                bail!("invalid activation QDQ scale");
            }
            output.extend(block.iter().map(|value| {
                let normalized = (*value / scale).clamp(-448.0, 448.0);
                quant_dequant_e4m3fn(normalized) * scale
            }));
        }
    }
    Ok(output)
}

fn quant_dequant_e4m3fn(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return value;
    }
    let step = if magnitude < 2.0f32.powi(-6) {
        2.0f32.powi(-9)
    } else {
        2.0f32.powf(magnitude.log2().floor() - 3.0)
    };
    let quantized = (magnitude / step).round_ties_even() * step;
    quantized.min(448.0).copysign(value)
}

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14E4m3QdqPipeline,
    dispatch: &S14E4m3QdqDispatch,
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

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| anyhow::anyhow!("single-arena alignment overflow"))
}
