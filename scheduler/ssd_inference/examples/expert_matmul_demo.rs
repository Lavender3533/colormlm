//! End-to-end "real expert × fake input" demo:
//!   GGUF → seek_read → staging → VRAM → dequant_q4_k → matmul_fp32 → output
//!
//! Uses a Qwen3-30B layer 0 expert 7 ffn_gate weight as the matrix B,
//! a deterministic fake hidden state as A, runs C = A * B on GPU.
//!
//! Validates NO crash + finite outputs (no NaN/Inf). This is the minimum
//! viable inference op chain for the new engine.
//!
//! Run: cargo run --release -p ssd_inference --example expert_matmul_demo

use anyhow::{bail, Result};
use ash::vk;
use gguf_reader::{ExpertKind, GgufFile};
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, DEQUANT_Q4_K_SPV, MATMUL_FP32_NAIVE_SPV},
    device::VulkanContext,
};
use std::time::Instant;

const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;

#[repr(C)]
struct PushMM { m: u32, n: u32, k: u32 }

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());

    println!("=== expert_matmul_demo: GGUF → dequant → matmul end-to-end ===\n");
    let g = GgufFile::open(&path)?;

    // Pick layer 0, expert slot 7, gate proj
    let exps = g.list_expert_tensors();
    let target = exps.iter()
        .find(|t| t.kind == ExpertKind::GateExps && t.layer == 0)
        .ok_or_else(|| anyhow::anyhow!("no layer 0 gate exps"))?;
    println!("Source tensor: {}  shape={:?}", target.name, target.shape);
    // shape is [n_experts=128, hidden=768, intermediate=2048]
    // Per-expert weight is hidden x intermediate = 768 x 2048
    let hidden: usize = target.shape[1];
    let intermediate: usize = target.shape[2];
    let m: usize = 1;
    let k: usize = hidden;
    let n: usize = intermediate;
    println!("Per-expert: hidden={} intermediate={}  → matmul A[{}x{}] B[{}x{}] = C[{}x{}]",
        hidden, intermediate, m, k, k, n, m, n);

    let exp_bytes = g.expert_slot_bytes(target.layer, target.kind, 7, 128)?;
    if exp_bytes.len() % Q4_K_BLOCK_BYTES != 0 {
        bail!("expert size {} not multiple of 144", exp_bytes.len());
    }
    let n_blocks = exp_bytes.len() / Q4_K_BLOCK_BYTES;
    let n_weights = n_blocks * QK_K;
    if n_weights != hidden * intermediate {
        bail!("expected {} weights, got {}", hidden * intermediate, n_weights);
    }
    println!("Expert slot: {} bytes ({} Q4_K blocks → {} fp32 weights)",
        exp_bytes.len(), n_blocks, n_weights);

    let ctx = VulkanContext::init()?;
    println!("GPU: {}\n", ctx.gpu_name);

    // ── Buffers ──
    let q_buf = GpuBuffer::new(&ctx, exp_bytes.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
    unsafe { q_buf.write_at(0, exp_bytes); }

    // Dequantized weight matrix B: (k × n) fp32
    let b_bytes = (n_weights * 4) as u64;
    let b_buf = GpuBuffer::new(&ctx, b_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::HOST_VISIBLE, false)?;

    // Input A (1 × k) — deterministic fake hidden state, host visible
    let a_bytes = (m * k * 4) as u64;
    let a_buf = GpuBuffer::new(&ctx, a_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
    unsafe {
        let a = std::slice::from_raw_parts_mut(a_buf.mapped() as *mut f32, m * k);
        let mut s: u64 = 0xC0FFEE;
        for x in a.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *x = (((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5) * 0.1;
        }
    }

    // Output C (1 × n)
    let c_bytes = (m * n * 4) as u64;
    let c_buf = GpuBuffer::new(&ctx, c_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;

    // ── Pipelines ──
    let pipe_dq = ComputePipeline::new(&ctx, DEQUANT_Q4_K_SPV, 2, 4)?;
    let binder_dq = DescriptorBinder::new(&ctx, &pipe_dq, &[
        (&q_buf, exp_bytes.len() as u64),
        (&b_buf, b_bytes),
    ])?;

    let pipe_mm = ComputePipeline::new(&ctx, MATMUL_FP32_NAIVE_SPV, 3,
        std::mem::size_of::<PushMM>() as u32)?;
    let binder_mm = DescriptorBinder::new(&ctx, &pipe_mm, &[
        (&a_buf, a_bytes), (&b_buf, b_bytes), (&c_buf, c_bytes),
    ])?;

    // ── Record one cmd buf with: dequant → barrier → matmul ──
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

        // Dequant: dispatch 1 wg per Q4_K block
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe_dq.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe_dq.layout, 0, &[binder_dq.set], &[]);
        let n_blk_bytes = (n_blocks as u32).to_le_bytes();
        ctx.device.cmd_push_constants(cb, pipe_dq.layout,
            vk::ShaderStageFlags::COMPUTE, 0, &n_blk_bytes);
        ctx.device.cmd_dispatch(cb, n_blocks as u32, 1, 1);

        // Memory barrier: dequant write → matmul read on b_buf
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        ctx.device.cmd_pipeline_barrier(cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier], &[], &[]);

        // Matmul
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe_mm.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe_mm.layout, 0, &[binder_mm.set], &[]);
        let push = PushMM { m: m as u32, n: n as u32, k: k as u32 };
        let push_bytes = std::slice::from_raw_parts(
            &push as *const _ as *const u8, std::mem::size_of::<PushMM>());
        ctx.device.cmd_push_constants(cb, pipe_mm.layout,
            vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
        let wg_x = ((n + 15) / 16) as u32;
        let wg_y = ((m + 15) / 16) as u32;
        ctx.device.cmd_dispatch(cb, wg_x, wg_y, 1);

        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("Pipeline: dequant ({} blocks) → barrier → matmul (1×{}×{})",
            n_blocks, k, n);
        println!("Total dispatch+sync: {:.2} ms", total_ms);

        // Inspect output
        let c = std::slice::from_raw_parts(c_buf.mapped() as *const f32, m * n);
        let mut sum = 0.0_f32;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut nan_count = 0;
        for &v in c {
            if v.is_nan() || v.is_infinite() { nan_count += 1; continue; }
            sum += v;
            if v < min { min = v; }
            if v > max { max = v; }
        }
        let mean = sum / (m * n) as f32;
        println!("Output stats: min={:.6e}  max={:.6e}  mean={:.6e}", min, max, mean);
        println!("First 8 outputs: {:?}", &c[..8]);
        if nan_count > 0 {
            bail!("❌ {} NaN/Inf in output", nan_count);
        }
        println!("✅ All {} outputs finite. End-to-end inference op chain WORKS.", m * n);

        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }

    binder_mm.destroy(&ctx); binder_dq.destroy(&ctx);
    pipe_mm.destroy(&ctx); pipe_dq.destroy(&ctx);
    c_buf.destroy(&ctx); a_buf.destroy(&ctx);
    b_buf.destroy(&ctx); q_buf.destroy(&ctx);

    println!("\n🎉 Real Q4_K_M expert × input → fp32 output, on GPU.");
    println!("   This is the minimum viable inference op chain for the new engine.");
    Ok(())
}
