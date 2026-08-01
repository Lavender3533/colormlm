//! RMSNorm correctness check: GPU vs CPU reference.
//! Tests with realistic dims (Qwen3 30B hidden=2048, 235B hidden=4096).
//!
//! Run: cargo run --release -p ssd_inference --example rmsnorm_demo

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, RMSNORM_SPV},
    device::VulkanContext,
};
use std::time::Instant;

#[repr(C)]
struct PushRMS { hidden: u32, eps: f32 }

fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32, n_vec: usize, hidden: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; n_vec * hidden];
    for v in 0..n_vec {
        let xv = &x[v * hidden..(v + 1) * hidden];
        let yv = &mut y[v * hidden..(v + 1) * hidden];
        let sum_sq: f64 = xv.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv_rms = 1.0 / ((sum_sq / hidden as f64) + eps as f64).sqrt();
        for i in 0..hidden {
            yv[i] = (xv[i] as f64 * inv_rms * w[i] as f64) as f32;
        }
    }
    y
}

fn run_test(ctx: &VulkanContext, pipe: &ComputePipeline,
            n_vec: usize, hidden: usize, eps: f32) -> Result<()> {
    let bytes_x = (n_vec * hidden * 4) as u64;
    let bytes_w = (hidden * 4) as u64;
    let bytes_y = (n_vec * hidden * 4) as u64;

    let make = |sz, usage| GpuBuffer::new(ctx, sz, usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true);
    let x_buf = make(bytes_x, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let w_buf = make(bytes_w, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let y_buf = make(bytes_y, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    // Fill X with random small values, W with random scale ~1.0
    let mut s: u64 = 0xCAFE_F00D;
    let mut next = |s: &mut u64| -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((*s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5)
    };
    let (x_host, w_host);
    unsafe {
        let xs = std::slice::from_raw_parts_mut(x_buf.mapped() as *mut f32, n_vec * hidden);
        let ws = std::slice::from_raw_parts_mut(w_buf.mapped() as *mut f32, hidden);
        for v in xs.iter_mut() { *v = next(&mut s) * 2.0; }
        for v in ws.iter_mut() { *v = 1.0 + next(&mut s) * 0.1; }
        x_host = xs.to_vec(); w_host = ws.to_vec();
    }

    let binder = DescriptorBinder::new(ctx, pipe, &[
        (&x_buf, bytes_x), (&w_buf, bytes_w), (&y_buf, bytes_y),
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
        let push = PushRMS { hidden: hidden as u32, eps };
        let push_bytes = std::slice::from_raw_parts(
            &push as *const _ as *const u8, std::mem::size_of::<PushRMS>());
        ctx.device.cmd_push_constants(cb, pipe.layout,
            vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
        ctx.device.cmd_dispatch(cb, n_vec as u32, 1, 1);
        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let cpu_y = cpu_rmsnorm(&x_host, &w_host, eps, n_vec, hidden);
        let gpu_y = std::slice::from_raw_parts(y_buf.mapped() as *const f32, n_vec * hidden);
        let mut max_rel = 0.0_f32;
        let mut max_abs = 0.0_f32;
        let mut argmax = 0;
        for i in 0..(n_vec * hidden) {
            let abs_err = (gpu_y[i] - cpu_y[i]).abs();
            let rel_err = if cpu_y[i].abs() > 1e-6 { abs_err / cpu_y[i].abs() } else { abs_err };
            if rel_err > max_rel { max_rel = rel_err; argmax = i; }
            if abs_err > max_abs { max_abs = abs_err; }
        }

        println!("  n_vec={:>4}  hidden={:>4}  GPU {:.3} ms  max_abs_err={:.2e}  max_rel_err={:.2e}",
            n_vec, hidden, gpu_ms, max_abs, max_rel);
        if max_rel > 1e-3 {
            println!("    cpu[{}]={}  gpu[{}]={}", argmax, cpu_y[argmax], argmax, gpu_y[argmax]);
            bail!("RMSNorm rel err exceeds 1e-3");
        }

        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    binder.destroy(ctx);
    y_buf.destroy(ctx); w_buf.destroy(ctx); x_buf.destroy(ctx);
    Ok(())
}

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    println!("=== RMSNorm correctness ===\nGPU: {}\n", ctx.gpu_name);
    let pipe = ComputePipeline::new(&ctx, RMSNORM_SPV, 3,
        std::mem::size_of::<PushRMS>() as u32)?;

    // Realistic LLM dims
    run_test(&ctx, &pipe, 1, 2048, 1e-6)?;   // 30B single token
    run_test(&ctx, &pipe, 1, 4096, 1e-6)?;   // 235B single token
    run_test(&ctx, &pipe, 32, 2048, 1e-6)?;  // 30B prefill batch
    run_test(&ctx, &pipe, 1, 5120, 1e-5)?;   // larger model
    println!("\n✅ RMSNorm matches CPU reference (max rel err < 1e-3) across all sizes");

    pipe.destroy(&ctx);
    Ok(())
}
