//! Q4_K_M dequant correctness check on a real GGUF expert weight.
//!
//! Pipeline:
//!   1. Read 30B Q4_K_M expert byte slice (gguf_reader, no candle dequant).
//!   2. Compute CPU reference using port of ggml's `dequantize_row_q4_K`.
//!   3. Run dequant_q4_k.comp on GPU, compare vs CPU reference.
//!
//! Run: cargo run --release -p ssd_inference --example dequant_q4_k_demo

use anyhow::{bail, Result};
use ash::vk;
use gguf_reader::{ExpertKind, GgufFile};
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, DEQUANT_Q4_K_SPV},
    device::VulkanContext,
};
use std::time::Instant;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 144;

// ── CPU reference: line-for-line port of ggml's dequantize_row_q4_K ──

fn fp16_to_fp32(half: u16) -> f32 {
    let s = (half >> 15) & 1;
    let e = (half >> 10) & 0x1F;
    let m = half & 0x3FF;
    let bits: u32 = if e == 0 {
        if m == 0 { (s as u32) << 31 }
        else {
            // subnormal
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

fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4);
        (d, m)
    }
}

fn cpu_dequant_q4_k(blocks: &[u8]) -> Vec<f32> {
    assert!(blocks.len() % BLOCK_BYTES == 0);
    let nb = blocks.len() / BLOCK_BYTES;
    let mut y = Vec::with_capacity(nb * QK_K);
    for i in 0..nb {
        let blk = &blocks[i * BLOCK_BYTES..(i + 1) * BLOCK_BYTES];
        let d_half = u16::from_le_bytes([blk[0], blk[1]]);
        let dmin_half = u16::from_le_bytes([blk[2], blk[3]]);
        let d = fp16_to_fp32(d_half);
        let dmin = fp16_to_fp32(dmin_half);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&blk[4..16]);
        let qs = &blk[16..144];

        let mut is = 0;
        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is + 0, &scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let d1 = d * sc1 as f32; let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32; let m2f = dmin * m2 as f32;
            let q_off = j / 2; // 32-byte qs slice starts here
            for l in 0..32 { y.push(d1 * (qs[q_off + l] & 0xF) as f32 - m1f); }
            for l in 0..32 { y.push(d2 * (qs[q_off + l]  >> 4) as f32 - m2f); }
            is += 2;
        }
    }
    y
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());

    println!("=== Q4_K_M dequant correctness check ===\n");
    let g = GgufFile::open(&path)?;
    println!("File: {}  ({} tensors)", path, g.n_tensors());

    // Pick one expert weight (first MoE gate of layer 0)
    let exps = g.list_expert_tensors();
    let target = exps.iter()
        .find(|t| t.kind == ExpertKind::GateExps && t.layer == 0)
        .ok_or_else(|| anyhow::anyhow!("no layer 0 gate exps"))?;
    println!("Target tensor: {}  shape={:?}  size={} bytes",
        target.name, target.shape, target.byte_size);

    // Read just one expert slot from the packed tensor
    let bytes_per_slot = (target.byte_size as usize) / 128;
    let exp_bytes = g.expert_slot_bytes(target.layer, target.kind, 7, 128)?;
    if exp_bytes.len() != bytes_per_slot {
        bail!("size mismatch");
    }
    if exp_bytes.len() % BLOCK_BYTES != 0 {
        bail!("expert size {} not multiple of 144 (Q4_K block)", exp_bytes.len());
    }
    let n_blocks = exp_bytes.len() / BLOCK_BYTES;
    let n_weights = n_blocks * QK_K;
    println!("Single expert slot: {} bytes = {} Q4_K blocks = {} fp32 weights",
        exp_bytes.len(), n_blocks, n_weights);

    // ── CPU dequant ──
    let t = Instant::now();
    let cpu_y = cpu_dequant_q4_k(exp_bytes);
    let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("CPU dequant: {:.2} ms", cpu_ms);

    // ── GPU dequant ──
    let ctx = VulkanContext::init()?;
    println!("GPU: {}\n", ctx.gpu_name);

    let q_buf = GpuBuffer::new(&ctx, exp_bytes.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
    let y_buf = GpuBuffer::new(&ctx, (n_weights * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
    unsafe { q_buf.write_at(0, exp_bytes); }

    let pipe = ComputePipeline::new(&ctx, DEQUANT_Q4_K_SPV, 2, 4)?;
    let binder = DescriptorBinder::new(&ctx, &pipe, &[
        (&q_buf, exp_bytes.len() as u64),
        (&y_buf, (n_weights * 4) as u64),
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
        let n_blk_bytes = (n_blocks as u32).to_le_bytes();
        ctx.device.cmd_push_constants(cb, pipe.layout,
            vk::ShaderStageFlags::COMPUTE, 0, &n_blk_bytes);
        ctx.device.cmd_dispatch(cb, n_blocks as u32, 1, 1);
        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("GPU dequant: {:.2} ms ({} blocks, 256 threads/wg)",
            gpu_ms, n_blocks);
        println!("  GPU output bandwidth: {:.2} GB/s",
            (n_weights * 4) as f64 / gpu_ms / 1e6);

        // Compare
        let gpu_y = std::slice::from_raw_parts(y_buf.mapped() as *const f32, n_weights);
        let mut max_err = 0.0_f32;
        let mut argmax = 0;
        for i in 0..n_weights {
            let e = (gpu_y[i] - cpu_y[i]).abs();
            if e > max_err { max_err = e; argmax = i; }
        }
        println!("\nmax abs err: {:.6e}  at index {}", max_err, argmax);
        if max_err > 1e-4 {
            println!("  CPU value: {}", cpu_y[argmax]);
            println!("  GPU value: {}", gpu_y[argmax]);
            println!("  First 8 CPU: {:?}", &cpu_y[..8]);
            println!("  First 8 GPU: {:?}", &gpu_y[..8]);
            bail!("Q4_K_M dequant mismatch");
        }
        println!("✅ GPU dequant matches CPU reference (max err < 1e-4)");
        println!("\nFirst 8 weights: {:?}", &cpu_y[..8]);

        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }

    binder.destroy(&ctx);
    pipe.destroy(&ctx);
    y_buf.destroy(&ctx);
    q_buf.destroy(&ctx);
    Ok(())
}
