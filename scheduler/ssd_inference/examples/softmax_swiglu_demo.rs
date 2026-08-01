//! Softmax + SwiGLU correctness check.

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, SOFTMAX_SPV, SWIGLU_SPV},
    device::VulkanContext,
};
use std::time::Instant;

#[repr(C)]
struct PushDim { dim: u32 }

fn cpu_softmax(x: &[f32], dim: usize, n_vec: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; n_vec * dim];
    for v in 0..n_vec {
        let xv = &x[v * dim..(v + 1) * dim];
        let yv = &mut y[v * dim..(v + 1) * dim];
        let m = xv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f64;
        for i in 0..dim {
            let e = ((xv[i] - m) as f64).exp();
            yv[i] = e as f32;
            sum += e;
        }
        let inv = 1.0 / sum;
        for v in yv.iter_mut() { *v = (*v as f64 * inv) as f32; }
    }
    y
}

fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| {
        let silu = g / (1.0 + (-g as f64).exp() as f32);
        silu * u
    }).collect()
}

fn run_softmax(ctx: &VulkanContext, dim: usize, n_vec: usize) -> Result<()> {
    let pipe = ComputePipeline::new(ctx, SOFTMAX_SPV, 2,
        std::mem::size_of::<PushDim>() as u32)?;
    let bytes = (n_vec * dim * 4) as u64;
    let mk = |usage| GpuBuffer::new(ctx, bytes, usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true);
    let x_buf = mk(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let y_buf = mk(vk::BufferUsageFlags::STORAGE_BUFFER)?;

    let mut s: u64 = 0xBAD_BEEF;
    let x_host: Vec<f32> = (0..n_vec * dim).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5) * 5.0
    }).collect();
    unsafe { x_buf.write_at(0, bytemuck_cast(&x_host)); }

    let binder = DescriptorBinder::new(ctx, &pipe, &[(&x_buf, bytes), (&y_buf, bytes)])?;
    unsafe {
        let pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics), None)?;
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1))?[0];
        ctx.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe.layout, 0, &[binder.set], &[]);
        let p = PushDim { dim: dim as u32 };
        let pb = std::slice::from_raw_parts(&p as *const _ as *const u8, 4);
        ctx.device.cmd_push_constants(cb, pipe.layout,
            vk::ShaderStageFlags::COMPUTE, 0, pb);
        ctx.device.cmd_dispatch(cb, n_vec as u32, 1, 1);
        ctx.device.end_command_buffer(cb)?;
        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let cpu = cpu_softmax(&x_host, dim, n_vec);
        let gpu = std::slice::from_raw_parts(y_buf.mapped() as *const f32, n_vec * dim);
        let mut max_abs = 0.0_f32;
        for i in 0..n_vec * dim {
            let e = (gpu[i] - cpu[i]).abs();
            if e > max_abs { max_abs = e; }
        }
        // Sum should be ~1.0 per vector
        let mut max_sum_err = 0.0_f32;
        for v in 0..n_vec {
            let s: f32 = gpu[v * dim..(v + 1) * dim].iter().sum();
            let e = (s - 1.0).abs();
            if e > max_sum_err { max_sum_err = e; }
        }
        println!("  softmax  n_vec={:<3} dim={:<5} GPU {:.3} ms  max_abs={:.2e}  max_sum_err={:.2e}",
            n_vec, dim, gpu_ms, max_abs, max_sum_err);
        if max_abs > 1e-5 || max_sum_err > 1e-5 {
            bail!("softmax mismatch");
        }
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    binder.destroy(ctx);
    y_buf.destroy(ctx); x_buf.destroy(ctx);
    pipe.destroy(ctx);
    Ok(())
}

fn run_swiglu(ctx: &VulkanContext, n: usize) -> Result<()> {
    let pipe = ComputePipeline::new(ctx, SWIGLU_SPV, 3,
        std::mem::size_of::<PushDim>() as u32)?;
    let bytes = (n * 4) as u64;
    let mk = |usage| GpuBuffer::new(ctx, bytes, usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true);
    let g_buf = mk(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let u_buf = mk(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let y_buf = mk(vk::BufferUsageFlags::STORAGE_BUFFER)?;

    let mut s: u64 = 0xFEED_F00D;
    let mut next = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5) * 4.0
    };
    let gate_host: Vec<f32> = (0..n).map(|_| next()).collect();
    let up_host:   Vec<f32> = (0..n).map(|_| next()).collect();
    unsafe {
        g_buf.write_at(0, bytemuck_cast(&gate_host));
        u_buf.write_at(0, bytemuck_cast(&up_host));
    }

    let binder = DescriptorBinder::new(ctx, &pipe, &[
        (&g_buf, bytes), (&u_buf, bytes), (&y_buf, bytes),
    ])?;

    unsafe {
        let pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics), None)?;
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1))?[0];
        ctx.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe.layout, 0, &[binder.set], &[]);
        let p = PushDim { dim: n as u32 };
        let pb = std::slice::from_raw_parts(&p as *const _ as *const u8, 4);
        ctx.device.cmd_push_constants(cb, pipe.layout,
            vk::ShaderStageFlags::COMPUTE, 0, pb);
        ctx.device.cmd_dispatch(cb, ((n + 255) / 256) as u32, 1, 1);
        ctx.device.end_command_buffer(cb)?;
        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let cpu = cpu_swiglu(&gate_host, &up_host);
        let gpu = std::slice::from_raw_parts(y_buf.mapped() as *const f32, n);
        let mut max_abs = 0.0_f32;
        for i in 0..n {
            let e = (gpu[i] - cpu[i]).abs();
            if e > max_abs { max_abs = e; }
        }
        println!("  swiglu   n={:<8}              GPU {:.3} ms  max_abs={:.2e}",
            n, gpu_ms, max_abs);
        if max_abs > 1e-5 { bail!("swiglu mismatch"); }
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    binder.destroy(ctx);
    y_buf.destroy(ctx); u_buf.destroy(ctx); g_buf.destroy(ctx);
    pipe.destroy(ctx);
    Ok(())
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
    }
}

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    println!("=== Softmax + SwiGLU correctness ===\nGPU: {}\n", ctx.gpu_name);

    // Softmax: typical use cases — router (128 expert), attention scores (4096 ctx)
    run_softmax(&ctx, 128, 1)?;
    run_softmax(&ctx, 128, 32)?;   // 32 tokens × 128 routes
    run_softmax(&ctx, 4096, 1)?;   // single attention row, 4k ctx
    run_softmax(&ctx, 1024, 32)?;  // batch of attention rows

    println!();
    // SwiGLU: typical use is intermediate dim per expert (2048 for 30B, 1536 for 235B)
    run_swiglu(&ctx, 2048)?;       // 30B per-expert
    run_swiglu(&ctx, 4096)?;       // 235B per-expert
    run_swiglu(&ctx, 2048 * 32)?;  // 32 expert outputs at once

    println!("\n✅ All ops match CPU reference (max abs err < 1e-5)");
    Ok(())
}
