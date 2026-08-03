//! FullDepth43 position0 共享-KV attention 的秒级 Vulkan 数值门。

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    s14_position0_attention::{
        S14Position0AttentionPipeline, S14_POSITION0_HEADS, S14_POSITION0_HEAD_DIM,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipeline = S14Position0AttentionPipeline::new(&ctx)?;
    let heads = S14_POSITION0_HEADS as usize;
    let dim = S14_POSITION0_HEAD_DIM as usize;
    let query: Vec<u16> = (0..heads * dim)
        .map(|index| to_bf16_bits(((index * 19 % 41) as f32 - 20.0) / 128.0))
        .collect();
    let kv: Vec<u16> = (0..dim)
        .map(|index| to_bf16_bits(((index * 23 % 37) as f32 - 18.0) / 96.0))
        .collect();
    let sink: Vec<f32> = (0..heads).map(|head| (head as f32 - 31.5) / 32.0).collect();
    let reference = cpu_reference(&query, &kv, &sink, heads, dim);

    let query_buffer = host_buffer(&ctx, (query.len() * 2) as u64)?;
    let kv_buffer = host_buffer(&ctx, (kv.len() * 2) as u64)?;
    let sink_buffer = host_buffer(&ctx, (sink.len() * 4) as u64)?;
    let output_buffer = host_buffer(&ctx, (query.len() * 2) as u64)?;
    unsafe {
        query_buffer.write_at(0, bytemuck::cast_slice(&query));
        kv_buffer.write_at(0, bytemuck::cast_slice(&kv));
        sink_buffer.write_at(0, bytemuck::cast_slice(&sink));
    }
    let dispatch = pipeline.bind(
        &ctx,
        &query_buffer,
        &kv_buffer,
        &sink_buffer,
        &output_buffer,
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
    let fence = unsafe {
        ctx.device
            .create_fence(&vk::FenceCreateInfo::default(), None)?
    };
    let started = Instant::now();
    unsafe {
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        pipeline.cmd(&ctx, command, &dispatch);
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let output =
        unsafe { std::slice::from_raw_parts(output_buffer.mapped() as *const u16, query.len()) };
    let mismatches = output
        .iter()
        .zip(&reference)
        .filter(|(actual, expected)| actual != expected)
        .count();
    if mismatches != 0 {
        bail!("position0 attention BF16 mismatch count={mismatches}");
    }
    println!(
        "status=pass heads={heads} head_dim={dim} outputs={} mismatches=0 wall_ms={wall_ms:.4}",
        output.len()
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    dispatch.binder.destroy(&ctx);
    output_buffer.destroy(&ctx);
    sink_buffer.destroy(&ctx);
    kv_buffer.destroy(&ctx);
    query_buffer.destroy(&ctx);
    pipeline.destroy(&ctx);
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

fn cpu_reference(query: &[u16], kv: &[u16], sink: &[f32], heads: usize, dim: usize) -> Vec<u16> {
    let mut output = vec![0u16; query.len()];
    let scale = (dim as f32).sqrt().recip();
    for head in 0..heads {
        let mut lanes = [0.0f32; 64];
        for lane in 0..64 {
            for d in (lane..dim).step_by(64) {
                lanes[lane] += from_bf16(query[head * dim + d]) * from_bf16(kv[d]);
            }
        }
        let mut stride = 32;
        while stride > 0 {
            for lane in 0..stride {
                lanes[lane] += lanes[lane + stride];
            }
            stride >>= 1;
        }
        let probability = sigmoid(lanes[0] * scale - sink[head]);
        for d in 0..dim {
            output[head * dim + d] = to_bf16_bits(probability * from_bf16(kv[d]));
        }
    }
    output
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
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
