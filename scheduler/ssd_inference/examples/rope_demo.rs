//! RoPE correctness vs CPU reference (single token, single head).

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, ROPE_SPV},
    device::VulkanContext,
};
use std::time::Instant;

#[repr(C)]
struct PushRope { n_heads: u32, head_dim: u32, base_pos: u32, theta: f32 }

fn cpu_rope(x: &mut [f32], n_heads: usize, head_dim: usize, n_tokens: usize,
            base_pos: usize, theta: f32) {
    let half = head_dim / 2;
    for tok in 0..n_tokens {
        for head in 0..n_heads {
            let base = (tok * n_heads + head) * head_dim;
            for pair in 0..half {
                let pos = (base_pos + tok) as f32;
                let freq = 1.0 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let angle = pos * freq;
                let (s, c) = angle.sin_cos();
                let i0 = base + 2 * pair;
                let i1 = i0 + 1;
                let v0 = x[i0]; let v1 = x[i1];
                x[i0] = v0 * c - v1 * s;
                x[i1] = v0 * s + v1 * c;
            }
        }
    }
}

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    println!("=== RoPE correctness ===\nGPU: {}\n", ctx.gpu_name);

    // Realistic Qwen3 dims: head_dim=128, n_heads=32, theta=1e6
    let configs = [
        ("30B Q decode", 32usize, 128usize, 1usize, 0usize, 1.0e6_f32),
        ("30B K decode (GQA n_kv=4)", 4, 128, 1, 7, 1.0e6),
        ("30B Q prefill 32 tok", 32, 128, 32, 0, 1.0e6),
        ("standard rope theta", 16, 64, 4, 0, 10000.0),
    ];

    let pipe = ComputePipeline::new(&ctx, ROPE_SPV, 1,
        std::mem::size_of::<PushRope>() as u32)?;

    for (label, n_heads, head_dim, n_tokens, base_pos, theta) in configs {
        let n = n_tokens * n_heads * head_dim;
        let bytes = (n * 4) as u64;
        let buf = GpuBuffer::new(&ctx, bytes, vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;

        // Init
        let mut s: u64 = 0xC0DE;
        let mut x_init: Vec<f32> = (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5) * 0.5
        }).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(x_init.as_ptr(), buf.mapped() as *mut f32, n);
        }
        let mut cpu = x_init.clone();
        cpu_rope(&mut cpu, n_heads, head_dim, n_tokens, base_pos, theta);

        let binder = DescriptorBinder::new(&ctx, &pipe, &[(&buf, bytes)])?;
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
            let p = PushRope {
                n_heads: n_heads as u32, head_dim: head_dim as u32,
                base_pos: base_pos as u32, theta,
            };
            let pb = std::slice::from_raw_parts(
                &p as *const _ as *const u8, std::mem::size_of::<PushRope>());
            ctx.device.cmd_push_constants(cb, pipe.layout,
                vk::ShaderStageFlags::COMPUTE, 0, pb);
            ctx.device.cmd_dispatch(cb, n_heads as u32, n_tokens as u32, 1);
            ctx.device.end_command_buffer(cb)?;

            let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cb_arr = [cb];
            let t = Instant::now();
            ctx.device.queue_submit(ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

            let gpu = std::slice::from_raw_parts(buf.mapped() as *const f32, n);
            let mut max_abs = 0.0_f32;
            for i in 0..n {
                let e = (gpu[i] - cpu[i]).abs();
                if e > max_abs { max_abs = e; }
            }
            println!("  {:<28}  n_tok={} n_heads={} head_dim={}  GPU {:.3} ms  max_abs={:.2e}",
                label, n_tokens, n_heads, head_dim, gpu_ms, max_abs);
            if max_abs > 1e-4 {
                bail!("RoPE mismatch (max abs err {:.2e})", max_abs);
            }
            ctx.device.destroy_fence(fence, None);
            ctx.device.destroy_command_pool(pool, None);
        }
        binder.destroy(&ctx);
        buf.destroy(&ctx);
        let _ = x_init;
    }
    pipe.destroy(&ctx);
    println!("\n✅ RoPE matches CPU reference across all configs (max abs err < 1e-4)");
    Ok(())
}
