//! Transfer ‖ compute parallelism with REAL compute shader.
//!
//! Replaces parallel_qs's VRAM↔VRAM copy stand-in with actual
//! `vector_add` dispatches. Tests if family 0 (compute) and family 2
//! (dedicated transfer) are truly parallel when the compute side does
//! real work (memory-bound) instead of just memory-controller traffic.
//!
//! Run: cargo run --release -p ssd_inference --example parallel_xfer_compute

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, VECTOR_ADD_SPV},
    device::VulkanContext,
};
use std::time::Instant;

const UPLOAD_MB:    usize = 64;
const UPLOAD_ITERS: usize = 32;            // 32 × 64 MB = 2 GB transfer total
const COMPUTE_N:    usize = 1 << 22;       // 4M elements per dispatch (16 MB per buffer)
const COMPUTE_DISPATCHES: usize = 200;     // tune so wall_G ~ wall_T

fn record_copies(
    ctx: &VulkanContext, qfi: u32, src: &GpuBuffer, dst: &GpuBuffer, n: usize,
) -> Result<(vk::CommandPool, vk::CommandBuffer)> {
    unsafe {
        let pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(qfi)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT), None)?;
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
        ctx.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let region = [vk::BufferCopy::default().size(src.size())];
        for _ in 0..n { ctx.device.cmd_copy_buffer(cb, src.handle(), dst.handle(), &region); }
        ctx.device.end_command_buffer(cb)?;
        Ok((pool, cb))
    }
}

fn record_compute(
    ctx: &VulkanContext, qfi: u32, pipe: &ComputePipeline, set: vk::DescriptorSet,
    n_elems: u32, n_dispatches: usize,
) -> Result<(vk::CommandPool, vk::CommandBuffer)> {
    unsafe {
        let pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(qfi)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT), None)?;
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
        ctx.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe.layout, 0, &[set], &[]);
        let n_bytes = n_elems.to_le_bytes();
        ctx.device.cmd_push_constants(cb, pipe.layout,
            vk::ShaderStageFlags::COMPUTE, 0, &n_bytes);
        let wg = ((n_elems + 63) / 64) as u32;
        for _ in 0..n_dispatches {
            ctx.device.cmd_dispatch(cb, wg, 1, 1);
        }
        ctx.device.end_command_buffer(cb)?;
        Ok((pool, cb))
    }
}

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    if !ctx.has_dedicated_transfer() {
        bail!("no dedicated transfer queue — can't test cross-queue parallelism");
    }
    println!("GPU: {}  | qf_graphics={}, qf_transfer={}",
        ctx.gpu_name, ctx.qf_graphics, ctx.qf_transfer);

    // ── Buffers for transfer (T): host→vram ──
    let host_src = GpuBuffer::new_staging(&ctx, (UPLOAD_MB * 1024 * 1024) as u64)?;
    unsafe { std::ptr::write_bytes(host_src.mapped(), 0xAA, UPLOAD_MB * 1024 * 1024); }
    let vram_dst = GpuBuffer::new_vram(&ctx, (UPLOAD_MB * 1024 * 1024) as u64,
        vk::BufferUsageFlags::TRANSFER_DST)?;

    // ── Buffers for compute (G): three vector_add SSBOs in VRAM ──
    let comp_bytes = (COMPUTE_N * 4) as u64;
    let buf_a = GpuBuffer::new_vram(&ctx, comp_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
    let buf_b = GpuBuffer::new_vram(&ctx, comp_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
    let buf_c = GpuBuffer::new_vram(&ctx, comp_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)?;
    let pipe = ComputePipeline::new(&ctx, VECTOR_ADD_SPV, 3, 4)?;
    let binder = DescriptorBinder::new(&ctx, &pipe, &[
        (&buf_a, comp_bytes), (&buf_b, comp_bytes), (&buf_c, comp_bytes),
    ])?;

    let fence_t = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    let fence_g = unsafe { ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };

    // ── A) T alone ──
    let (pool_t, cb_t) = record_copies(&ctx, ctx.qf_transfer, &host_src, &vram_dst, UPLOAD_ITERS)?;
    let t0 = Instant::now();
    let cbs = [cb_t];
    unsafe {
        ctx.device.queue_submit(ctx.q_transfer,
            &[vk::SubmitInfo::default().command_buffers(&cbs)], fence_t)?;
        ctx.device.wait_for_fences(&[fence_t], true, u64::MAX)?;
        ctx.device.reset_fences(&[fence_t])?;
    }
    let wall_t = t0.elapsed().as_secs_f64() * 1000.0;
    let bw_t = (UPLOAD_MB * UPLOAD_ITERS) as f64 / (wall_t / 1000.0) / 1024.0;
    println!("\n[A] T alone:  {:.1} ms   ({:.2} GB/s, {} × {} MB upload)",
        wall_t, bw_t, UPLOAD_ITERS, UPLOAD_MB);

    // ── B) G alone ──
    let (pool_g, cb_g) = record_compute(&ctx, ctx.qf_graphics, &pipe, binder.set,
        COMPUTE_N as u32, COMPUTE_DISPATCHES)?;
    let t0 = Instant::now();
    let cbs = [cb_g];
    unsafe {
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cbs)], fence_g)?;
        ctx.device.wait_for_fences(&[fence_g], true, u64::MAX)?;
        ctx.device.reset_fences(&[fence_g])?;
    }
    let wall_g = t0.elapsed().as_secs_f64() * 1000.0;
    let gflops = (COMPUTE_N * COMPUTE_DISPATCHES) as f64 / (wall_g / 1000.0) / 1e9;
    println!("[B] G alone:  {:.1} ms   ({:.1} G adds/s, {} dispatches × {} elems)",
        wall_g, gflops, COMPUTE_DISPATCHES, COMPUTE_N);

    // ── C) T ‖ G  (re-record cmd bufs since previous were ONE_TIME_SUBMIT) ──
    unsafe {
        ctx.device.destroy_command_pool(pool_t, None);
        ctx.device.destroy_command_pool(pool_g, None);
    }
    let (pool_t2, cb_t2) = record_copies(&ctx, ctx.qf_transfer, &host_src, &vram_dst, UPLOAD_ITERS)?;
    let (pool_g2, cb_g2) = record_compute(&ctx, ctx.qf_graphics, &pipe, binder.set,
        COMPUTE_N as u32, COMPUTE_DISPATCHES)?;
    let t0 = Instant::now();
    let cbs_t = [cb_t2]; let cbs_g = [cb_g2];
    unsafe {
        ctx.device.queue_submit(ctx.q_transfer,
            &[vk::SubmitInfo::default().command_buffers(&cbs_t)], fence_t)?;
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cbs_g)], fence_g)?;
        ctx.device.wait_for_fences(&[fence_t, fence_g], true, u64::MAX)?;
    }
    let wall_c = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[C] T ‖ G:    {:.1} ms", wall_c);

    let max_tg = wall_t.max(wall_g);
    let sum_tg = wall_t + wall_g;
    let par_score = (sum_tg - wall_c) / (sum_tg - max_tg).max(0.001);
    println!("\n=== Verdict ===");
    println!("  max(T,G) = {:.1} ms   (perfect parallel)", max_tg);
    println!("  T + G    = {:.1} ms   (fully serial)", sum_tg);
    println!("  measured = {:.1} ms   parallelism = {:.0}%",
        wall_c, par_score * 100.0);
    if par_score > 0.7 {
        println!("  ✅  REAL compute shader on family 0 runs in parallel with");
        println!("       PCIe transfer on family 2. Cross-queue concurrency works");
        println!("       under realistic engine workloads.");
    } else if par_score > 0.3 {
        println!("  ⚠️  Partial overlap — some shared bottleneck (likely VRAM bus)");
    } else {
        println!("  ❌  Effectively serial — investigate.");
    }

    // ── Cleanup ──
    unsafe {
        ctx.device.destroy_command_pool(pool_t2, None);
        ctx.device.destroy_command_pool(pool_g2, None);
        ctx.device.destroy_fence(fence_t, None);
        ctx.device.destroy_fence(fence_g, None);
    }
    binder.destroy(&ctx);
    pipe.destroy(&ctx);
    buf_c.destroy(&ctx); buf_b.destroy(&ctx); buf_a.destroy(&ctx);
    vram_dst.destroy(&ctx); host_src.destroy(&ctx);

    Ok(())
}
