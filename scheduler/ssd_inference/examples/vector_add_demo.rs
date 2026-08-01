//! Vector add compute shader smoke test.
//!
//! Goal: prove the compute pipeline path end-to-end.
//!   1. Allocate two HOST_VISIBLE input buffers and one host-readable output.
//!   2. Fill A with `i`, B with `2*i`.
//!   3. Dispatch `vector_add.comp` to compute C = A + B.
//!   4. Read C back and assert C[i] == 3*i for all i.
//!
//! Run: cargo run --release -p ssd_inference --example vector_add_demo

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, VECTOR_ADD_SPV},
    device::VulkanContext,
};
use std::time::Instant;

const N: usize = 1 << 20; // 1M elements = 4 MB per buffer

fn main() -> Result<()> {
    println!("=== vector_add_demo: first compute shader on the new engine ===\n");

    let ctx = VulkanContext::init()?;
    println!("GPU: {}", ctx.gpu_name);
    println!("SPIR-V bytecode: {} bytes\n", VECTOR_ADD_SPV.len());

    // Allocate buffers. We use HOST_VISIBLE for inputs/outputs to skip the
    // staging dance — vector_add is a smoke test, not a perf bench.
    let bytes = (N * std::mem::size_of::<f32>()) as u64;
    let make_io = |usage| GpuBuffer::new(&ctx, bytes,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true);
    let buf_a = make_io(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let buf_b = make_io(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let buf_c = make_io(vk::BufferUsageFlags::STORAGE_BUFFER)?;

    // Fill inputs
    unsafe {
        let a = std::slice::from_raw_parts_mut(buf_a.mapped() as *mut f32, N);
        let b = std::slice::from_raw_parts_mut(buf_b.mapped() as *mut f32, N);
        for i in 0..N {
            a[i] = i as f32;
            b[i] = 2.0 * i as f32;
        }
    }

    // Build compute pipeline (3 storage bindings + 4 byte push constant for `n`)
    let pipe = ComputePipeline::new(&ctx, VECTOR_ADD_SPV, 3, 4)?;

    // Descriptor set
    let binder = DescriptorBinder::new(&ctx, &pipe, &[
        (&buf_a, bytes),
        (&buf_b, bytes),
        (&buf_c, bytes),
    ])?;

    // Record a one-shot command buffer
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
        // push constant: u32 n
        let n_bytes = (N as u32).to_le_bytes();
        ctx.device.cmd_push_constants(cb, pipe.layout,
            vk::ShaderStageFlags::COMPUTE, 0, &n_bytes);
        // dispatch ceil(N / 64) workgroups
        let wg = ((N + 63) / 64) as u32;
        ctx.device.cmd_dispatch(cb, wg, 1, 1);
        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t0 = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;

        println!("Dispatch + sync: {:.3} ms ({} elements, {} workgroups, wg_size=64)",
            dt_ms, N, wg);

        // Verify
        let c = std::slice::from_raw_parts(buf_c.mapped() as *const f32, N);
        let mut wrong = 0;
        for i in 0..N {
            let expected = 3.0 * i as f32;
            if (c[i] - expected).abs() > 1e-3 { wrong += 1; }
        }
        if wrong == 0 {
            println!("✅ Verified: C[i] == 3*i for all {} elements", N);
        } else {
            bail!("❌ {} mismatches out of {}", wrong, N);
        }
        println!("First 8 results: {:?}", &c[..8]);

        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }

    binder.destroy(&ctx);
    pipe.destroy(&ctx);
    buf_c.destroy(&ctx);
    buf_b.destroy(&ctx);
    buf_a.destroy(&ctx);

    println!("\n🎉 First compute shader works end-to-end on the new engine.");
    Ok(())
}
