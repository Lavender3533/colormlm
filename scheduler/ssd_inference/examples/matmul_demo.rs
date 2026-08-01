//! fp32 matmul correctness + perf bench.
//!
//! 1. Small (M=128, N=128, K=256): compare GPU vs CPU reference, max abs err < 1e-2.
//! 2. Medium (M=512, N=512, K=512): GFLOPS measurement.
//! 3. Large (M=2048, N=2048, K=2048): saturating GPU.
//!
//! Run: cargo run --release -p ssd_inference --example matmul_demo

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, MATMUL_FP32_NAIVE_SPV},
    device::VulkanContext,
};
use std::time::Instant;

fn cpu_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

#[repr(C)]
struct PushMM { m: u32, n: u32, k: u32 }

fn run_matmul(
    ctx: &VulkanContext,
    pipe: &ComputePipeline,
    label: &str,
    m: usize, n: usize, k: usize,
    verify: bool,
) -> Result<()> {
    let bytes_a = (m * k * 4) as u64;
    let bytes_b = (k * n * 4) as u64;
    let bytes_c = (m * n * 4) as u64;

    // HOST_VISIBLE all three to skip staging dance for this scaffold
    let make = |sz, usage| GpuBuffer::new(ctx, sz, usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true);
    let buf_a = make(bytes_a, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let buf_b = make(bytes_b, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let buf_c = make(bytes_c, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    // Fill A and B with deterministic small values (avoid fp32 sum overflow)
    unsafe {
        let a = std::slice::from_raw_parts_mut(buf_a.mapped() as *mut f32, m * k);
        let b = std::slice::from_raw_parts_mut(buf_b.mapped() as *mut f32, k * n);
        let mut s: u64 = 0xC0FFEE;
        let mut next = |s: &mut u64| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((*s >> 32) as u32) as f32) / (u32::MAX as f32) - 0.5
        };
        for x in a.iter_mut() { *x = next(&mut s) * 0.1; }
        for x in b.iter_mut() { *x = next(&mut s) * 0.1; }
    }

    let binder = DescriptorBinder::new(ctx, pipe, &[
        (&buf_a, bytes_a), (&buf_b, bytes_b), (&buf_c, bytes_c),
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
        let push = PushMM { m: m as u32, n: n as u32, k: k as u32 };
        let push_bytes = std::slice::from_raw_parts(
            &push as *const _ as *const u8, std::mem::size_of::<PushMM>());
        ctx.device.cmd_push_constants(cb, pipe.layout,
            vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
        let wg_x = ((n + 15) / 16) as u32;
        let wg_y = ((m + 15) / 16) as u32;
        ctx.device.cmd_dispatch(cb, wg_x, wg_y, 1);
        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t0 = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = flops / dt_ms / 1e6;
        println!("\n[{}] M={} N={} K={}", label, m, n, k);
        println!("  GPU dispatch+sync: {:.2} ms   {:.1} GFLOPS",
            dt_ms, gflops);

        if verify {
            let a = std::slice::from_raw_parts(buf_a.mapped() as *const f32, m * k);
            let b = std::slice::from_raw_parts(buf_b.mapped() as *const f32, k * n);
            let c_gpu = std::slice::from_raw_parts(buf_c.mapped() as *const f32, m * n);
            let t_cpu = Instant::now();
            let c_cpu = cpu_reference(a, b, m, n, k);
            let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
            let mut max_err = 0.0_f32;
            for i in 0..(m * n) {
                let e = (c_gpu[i] - c_cpu[i]).abs();
                if e > max_err { max_err = e; }
            }
            println!("  CPU reference:     {:.2} ms", cpu_ms);
            println!("  max abs err:       {:.2e}", max_err);
            if max_err > 1e-2 {
                bail!("matmul mismatch: max err {:.2e} > 1e-2", max_err);
            }
            println!("  ✅ correctness OK");
        }

        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }

    binder.destroy(ctx);
    buf_c.destroy(ctx);
    buf_b.destroy(ctx);
    buf_a.destroy(ctx);
    Ok(())
}

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    println!("=== matmul_demo: naive fp32 GEMM on the new engine ===");
    println!("GPU: {}\n", ctx.gpu_name);

    // Build pipeline once, reuse across sizes.
    let pipe = ComputePipeline::new(&ctx, MATMUL_FP32_NAIVE_SPV, 3,
        std::mem::size_of::<PushMM>() as u32)?;

    // Small: verify correctness
    run_matmul(&ctx, &pipe, "small", 128, 128, 256, true)?;
    // Medium: timing only
    run_matmul(&ctx, &pipe, "medium", 512, 512, 512, false)?;
    // Large: saturating
    run_matmul(&ctx, &pipe, "large", 2048, 2048, 2048, false)?;

    pipe.destroy(&ctx);
    println!("\n5700 XT theoretical fp32 peak: ~7.5 TFLOPS");
    println!("Naive matmul typical: 5-15% of peak (no shared mem / no tile).");
    println!("This is intentional — correctness scaffold for forward pass next.");
    Ok(())
}
