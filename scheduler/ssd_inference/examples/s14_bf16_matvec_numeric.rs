//! Polaris 通用 BF16 matvec 的无模型、秒级 Vulkan 数值门。
//!
//! 覆盖 K=1/4/8 的共享权重扫描，以及 N=65537 的二维 dispatch 展平。
//! 这只证明通用投影算子，不代表完整模型速度或质量。

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    s14_vulkan::{S14Bf16MatvecShape, S14NumericPipelines},
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipelines = S14NumericPipelines::new(&ctx)?;
    let mut reports = Vec::new();

    for &batch in &[1u32, 4, 8] {
        reports.push(run_case(&ctx, &pipelines, 257, 128, batch)?);
    }
    reports.push(run_case(&ctx, &pipelines, 65_537, 2, 1)?);

    for report in reports {
        println!(
            "BF16 matvec N={} K={} B={} max_abs_error={:.8} wall_ms={:.4}",
            report.n, report.k, report.batch, report.max_abs_error, report.wall_ms
        );
    }
    pipelines.destroy(&ctx);
    Ok(())
}

struct CaseReport {
    n: u32,
    k: u32,
    batch: u32,
    max_abs_error: f32,
    wall_ms: f64,
}

fn run_case(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    n: u32,
    k: u32,
    batch: u32,
) -> Result<CaseReport> {
    let shape = S14Bf16MatvecShape::new(n, k, batch)?;
    let weight = deterministic_bf16_weights(n as usize, k as usize);
    let input = deterministic_input(batch as usize, k as usize);
    let reference = cpu_reference(&weight, &input, n as usize, k as usize, batch as usize);

    let weight_buffer = host_buffer(ctx, shape.bf16_weight_bytes()?)?;
    let input_buffer = host_buffer(ctx, shape.fp32_input_bytes()?)?;
    let output_buffer = host_buffer(ctx, shape.fp32_output_bytes()?)?;
    unsafe {
        weight_buffer.write_at(0, bytemuck::cast_slice(&weight));
        input_buffer.write_at(0, bytemuck::cast_slice(&input));
    }
    let dispatch =
        pipelines.bind_bf16_matvec(ctx, shape, &weight_buffer, &input_buffer, &output_buffer)?;

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
        pipelines.cmd_bf16_matvec(ctx, command, &dispatch);
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
    let output = unsafe {
        std::slice::from_raw_parts(
            output_buffer.mapped() as *const f32,
            n as usize * batch as usize,
        )
    };
    let max_abs_error = output
        .iter()
        .zip(&reference)
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);
    if output.iter().any(|value| !value.is_finite()) || max_abs_error > 2.0e-4 {
        bail!("BF16 matvec mismatch for N={n} K={k} B={batch}: max_abs_error={max_abs_error}");
    }

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    dispatch.binder.destroy(ctx);
    output_buffer.destroy(ctx);
    input_buffer.destroy(ctx);
    weight_buffer.destroy(ctx);

    Ok(CaseReport {
        n,
        k,
        batch,
        max_abs_error,
        wall_ms,
    })
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

fn deterministic_bf16_weights(n: usize, k: usize) -> Vec<u16> {
    (0..n * k)
        .map(|index| {
            let value = ((index.wrapping_mul(17) % 31) as f32 - 15.0) / 64.0;
            (value.to_bits() >> 16) as u16
        })
        .collect()
}

fn deterministic_input(batch: usize, k: usize) -> Vec<f32> {
    (0..batch * k)
        .map(|index| ((index.wrapping_mul(13) % 29) as f32 - 14.0) / 128.0)
        .collect()
}

fn cpu_reference(weight: &[u16], input: &[f32], n: usize, k: usize, batch: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * n];
    for row in 0..n {
        let mut lanes = vec![[0.0f32; 64]; batch];
        for lane in 0..64 {
            for dim in (lane..k).step_by(64) {
                let w = f32::from_bits((weight[row * k + dim] as u32) << 16);
                for batch_index in 0..batch {
                    lanes[batch_index][lane] += w * input[batch_index * k + dim];
                }
            }
        }
        let mut stride = 32;
        while stride > 0 {
            for lane in 0..stride {
                for batch_index in 0..batch {
                    lanes[batch_index][lane] += lanes[batch_index][lane + stride];
                }
            }
            stride >>= 1;
        }
        for batch_index in 0..batch {
            output[batch_index * n + row] = lanes[batch_index][0];
        }
    }
    output
}
