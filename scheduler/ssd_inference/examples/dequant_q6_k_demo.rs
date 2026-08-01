//! Q6_K dequant correctness check on a real GGUF tensor (output.weight aka lm_head).
//!
//! Pipeline:
//!   1. Read Qwen3-30B `output.weight` byte slice (Q6_K).
//!   2. CPU reference port of ggml's `dequantize_row_q6_K`.
//!   3. Run dequant_q6_k.comp on GPU; compare max abs err.
//!
//! Run: cargo run --release -p ssd_inference --example dequant_q6_k_demo

use anyhow::{bail, Result};
use ash::vk;
use gguf_reader::GgufFile;
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, DEQUANT_Q6_K_SPV},
    device::VulkanContext,
};
use std::time::Instant;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 210;

fn fp16_to_fp32(half: u16) -> f32 {
    let s = (half >> 15) & 1;
    let e = (half >> 10) & 0x1F;
    let m = half & 0x3FF;
    let bits: u32 = if e == 0 {
        if m == 0 { (s as u32) << 31 }
        else {
            let mut e2: i32 = -14;
            let mut m2 = m as u32;
            while (m2 & 0x400) == 0 { m2 <<= 1; e2 -= 1; }
            m2 &= 0x3FF;
            ((s as u32) << 31) | (((e2 + 127) as u32) << 23) | (m2 << 13)
        }
    } else if e == 0x1F {
        ((s as u32) << 31) | (0xFF << 23) | ((m as u32) << 13)
    } else {
        ((s as u32) << 31) | ((e as u32 - 15 + 127) << 23) | ((m as u32) << 13)
    };
    f32::from_bits(bits)
}

fn cpu_dequant_q6_k(blocks: &[u8]) -> Vec<f32> {
    assert!(blocks.len() % BLOCK_BYTES == 0);
    let nb = blocks.len() / BLOCK_BYTES;
    let mut y = Vec::with_capacity(nb * QK_K);
    for i in 0..nb {
        let blk = &blocks[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc: &[i8] = unsafe {
            std::slice::from_raw_parts(blk[192..208].as_ptr() as *const i8, 16)
        };
        let d = fp16_to_fp32(u16::from_le_bytes([blk[208], blk[209]]));

        // Match ggml-quants.c dequantize_row_q6_K layout
        for n in 0..2 {
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;
            let y_base = i * QK_K + n * 128;
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[ql_base + l]      & 0xF) | (((qh[qh_base + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0xF) | (((qh[qh_base + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_base + l]      >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i32 - 32;
                // Push to y at the right indices. We're appending, so use direct index via len.
                // Better: pre-resize.
                let _ = (q1, q2, q3, q4, is, sc_base, y_base);
            }
        }
        // Pre-allocated path: fill into a 256-slot scratch then push.
        let mut blk_y = vec![0f32; 256];
        for n in 0..2 {
            let ql_base = n * 64;
            let qh_base = n * 32;
            let sc_base = n * 8;
            let y_base = n * 128;
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[ql_base + l]      & 0xF) | (((qh[qh_base + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_base + l + 32] & 0xF) | (((qh[qh_base + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_base + l]      >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i32 - 32;
                blk_y[y_base + l]      = d * sc[sc_base + is + 0] as f32 * q1 as f32;
                blk_y[y_base + l + 32] = d * sc[sc_base + is + 2] as f32 * q2 as f32;
                blk_y[y_base + l + 64] = d * sc[sc_base + is + 4] as f32 * q3 as f32;
                blk_y[y_base + l + 96] = d * sc[sc_base + is + 6] as f32 * q4 as f32;
            }
        }
        y.extend_from_slice(&blk_y);
    }
    y
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());

    println!("=== Q6_K dequant correctness check ===\n");
    let g = GgufFile::open(&path)?;

    // Use output.weight (lm_head, Q6_K) — this exact tensor is used at the very
    // end of forward pass, so getting it right matters most.
    let target = "output.weight";
    let info = g.tensor_info(target)
        .ok_or_else(|| anyhow::anyhow!("missing {target}"))?;
    println!("Target: {} dtype={:?} shape={:?}", target, info.ggml_dtype, info.shape.dims());

    let bytes = g.tensor_bytes(target)?;
    if bytes.len() % BLOCK_BYTES != 0 {
        bail!("size {} not multiple of {} (Q6_K block)", bytes.len(), BLOCK_BYTES);
    }

    // Sample first 64K blocks (16M weights, 64MB output) to keep runtime sane.
    let n_blocks = bytes.len() / BLOCK_BYTES;
    let sample_blocks = n_blocks.min(64 * 1024);
    let sample_bytes = &bytes[..sample_blocks * BLOCK_BYTES];
    let sample_weights = sample_blocks * QK_K;
    println!("Sampling {} blocks of {} ({} fp32 weights = {:.1} MB)",
        sample_blocks, n_blocks, sample_weights,
        sample_weights as f64 * 4.0 / 1024.0 / 1024.0);

    let t = Instant::now();
    let cpu_y = cpu_dequant_q6_k(sample_bytes);
    println!("CPU dequant: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0);

    let ctx = VulkanContext::init()?;
    println!("GPU: {}\n", ctx.gpu_name);

    let q_buf = GpuBuffer::new(&ctx, sample_bytes.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
    let y_buf = GpuBuffer::new(&ctx, (sample_weights * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
    unsafe { q_buf.write_at(0, sample_bytes); }

    let pipe = ComputePipeline::new(&ctx, DEQUANT_Q6_K_SPV, 2, 4)?;
    let binder = DescriptorBinder::new(&ctx, &pipe, &[
        (&q_buf, sample_bytes.len() as u64),
        (&y_buf, (sample_weights * 4) as u64),
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
        let push = (sample_blocks as u32).to_le_bytes();
        ctx.device.cmd_push_constants(cb, pipe.layout,
            vk::ShaderStageFlags::COMPUTE, 0, &push);
        ctx.device.cmd_dispatch(cb, sample_blocks as u32, 1, 1);
        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("GPU dequant: {:.2} ms ({} blocks)", gpu_ms, sample_blocks);
        println!("  bandwidth: {:.2} GB/s out", (sample_weights * 4) as f64 / gpu_ms / 1e6);

        let gpu_y = std::slice::from_raw_parts(y_buf.mapped() as *const f32, sample_weights);
        let mut max_err = 0.0_f32;
        let mut argmax = 0usize;
        for i in 0..sample_weights {
            let e = (gpu_y[i] - cpu_y[i]).abs();
            if e > max_err { max_err = e; argmax = i; }
        }
        println!("\nmax abs err: {:.6e} at index {}", max_err, argmax);
        if max_err > 1e-3 {
            println!("  CPU val: {}\n  GPU val: {}", cpu_y[argmax], gpu_y[argmax]);
            println!("  CPU [0..8]: {:?}\n  GPU [0..8]: {:?}", &cpu_y[..8], &gpu_y[..8]);
            bail!("Q6_K dequant mismatch");
        }
        println!("OK Q6_K dequant matches CPU reference (max err < 1e-3)");
        println!("First 8 weights: {:?}", &cpu_y[..8]);

        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }

    binder.destroy(&ctx);
    pipe.destroy(&ctx);
    y_buf.destroy(&ctx);
    q_buf.destroy(&ctx);
    Ok(())
}
