//! End-to-end attention block correctness:
//!   scaled_dot (Q×K^T / √d, causal mask) → softmax → attn_v (×V) → output
//!
//! Compares against a CPU reference that implements the same math.
//! Uses GQA (n_q_heads != n_kv_heads) to exercise head mapping.

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    buffer::GpuBuffer,
    compute::{ComputePipeline, DescriptorBinder, ATTN_V_SPV, SCALED_DOT_SPV, SOFTMAX_SPV},
    device::VulkanContext,
};
use std::time::Instant;

#[repr(C)]
struct PushSD { n_tok: u32, n_q: u32, n_kv: u32, head_dim: u32, seq_len: u32, base_pos: u32, scale: f32 }
#[repr(C)]
struct PushAV { n_tok: u32, n_q: u32, n_kv: u32, head_dim: u32, seq_len: u32 }
#[repr(C)]
struct PushSm { dim: u32 }

fn cpu_attention(
    q: &[f32], k: &[f32], v: &[f32],
    n_tok: usize, n_q: usize, n_kv: usize, head_dim: usize,
    seq_len: usize, base_pos: usize,
) -> Vec<f32> {
    let scale = (head_dim as f32).sqrt().recip();
    let mut out = vec![0.0_f32; n_tok * n_q * head_dim];
    for h in 0..n_q {
        let kv_h = h * n_kv / n_q;
        for i in 0..n_tok {
            let q_pos = base_pos + i;
            // Compute raw scores [seq_len]
            let mut scores = vec![0.0_f32; seq_len];
            for j in 0..seq_len {
                if j > q_pos { scores[j] = f32::NEG_INFINITY; continue; }
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[(i * n_q + h) * head_dim + d]
                         * k[(j * n_kv + kv_h) * head_dim + d];
                }
                scores[j] = dot * scale;
            }
            // softmax
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f64;
            for s in scores.iter_mut() {
                let e = ((*s - m) as f64).exp();
                *s = e as f32;
                sum += e;
            }
            let inv = 1.0 / sum;
            for s in scores.iter_mut() { *s = (*s as f64 * inv) as f32; }
            // V mix
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for j in 0..seq_len {
                    acc += scores[j] * v[(j * n_kv + kv_h) * head_dim + d];
                }
                out[(i * n_q + h) * head_dim + d] = acc;
            }
        }
    }
    out
}

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    println!("=== Attention block correctness ===\nGPU: {}\n", ctx.gpu_name);

    // Realistic-ish GQA shape (small enough for fast CPU ref)
    let n_tok = 4usize;       // prefill 4 tokens
    let n_q = 8usize;         // 8 query heads
    let n_kv = 2usize;        // 2 KV heads (GQA 4:1)
    let head_dim = 64usize;
    let seq_len = 8usize;     // current cache length (>= base_pos + n_tok)
    let base_pos = 4usize;    // queries are at positions [4, 5, 6, 7]
    let scale = (head_dim as f32).sqrt().recip();
    println!("config: n_tok={n_tok} n_q={n_q} n_kv={n_kv} head_dim={head_dim} seq_len={seq_len} base_pos={base_pos}");

    // Random fp32 Q, K, V
    let mut s: u64 = 0xC0DE_FEED;
    let mut next = || -> f32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5) * 0.5
    };
    let q_host: Vec<f32> = (0..n_tok * n_q * head_dim).map(|_| next()).collect();
    let k_host: Vec<f32> = (0..seq_len * n_kv * head_dim).map(|_| next()).collect();
    let v_host: Vec<f32> = (0..seq_len * n_kv * head_dim).map(|_| next()).collect();

    let cpu_t = Instant::now();
    let cpu_out = cpu_attention(&q_host, &k_host, &v_host,
        n_tok, n_q, n_kv, head_dim, seq_len, base_pos);
    let cpu_ms = cpu_t.elapsed().as_secs_f64() * 1000.0;
    println!("CPU reference: {:.2} ms\n", cpu_ms);

    // GPU buffers
    let mk = |sz: u64, usage| GpuBuffer::new(&ctx, sz, usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL, true);
    let q_buf = mk((q_host.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let k_buf = mk((k_host.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let v_buf = mk((v_host.len() * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let scores_n = n_q * n_tok * seq_len;
    let scores_buf = GpuBuffer::new(&ctx, (scores_n * 4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::HOST_VISIBLE, false)?;
    let out_buf = mk((n_tok * n_q * head_dim * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER)?;

    unsafe {
        q_buf.write_at(0, slice_as_bytes(&q_host));
        k_buf.write_at(0, slice_as_bytes(&k_host));
        v_buf.write_at(0, slice_as_bytes(&v_host));
    }

    let pipe_sd = ComputePipeline::new(&ctx, SCALED_DOT_SPV, 3,
        std::mem::size_of::<PushSD>() as u32)?;
    let pipe_sm = ComputePipeline::new(&ctx, SOFTMAX_SPV, 2,
        std::mem::size_of::<PushSm>() as u32)?;
    let pipe_av = ComputePipeline::new(&ctx, ATTN_V_SPV, 3,
        std::mem::size_of::<PushAV>() as u32)?;

    let bind_sd = DescriptorBinder::new(&ctx, &pipe_sd, &[
        (&q_buf, q_buf.size()), (&k_buf, k_buf.size()), (&scores_buf, scores_buf.size()),
    ])?;
    let bind_sm = DescriptorBinder::new(&ctx, &pipe_sm, &[
        (&scores_buf, scores_buf.size()), (&scores_buf, scores_buf.size()),
    ])?; // softmax in-place
    let bind_av = DescriptorBinder::new(&ctx, &pipe_av, &[
        (&scores_buf, scores_buf.size()), (&v_buf, v_buf.size()), (&out_buf, out_buf.size()),
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

        // 1) scaled dot
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe_sd.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe_sd.layout, 0, &[bind_sd.set], &[]);
        let p_sd = PushSD {
            n_tok: n_tok as u32, n_q: n_q as u32, n_kv: n_kv as u32,
            head_dim: head_dim as u32, seq_len: seq_len as u32,
            base_pos: base_pos as u32, scale,
        };
        let pb = std::slice::from_raw_parts(
            &p_sd as *const _ as *const u8, std::mem::size_of::<PushSD>());
        ctx.device.cmd_push_constants(cb, pipe_sd.layout,
            vk::ShaderStageFlags::COMPUTE, 0, pb);
        ctx.device.cmd_dispatch(cb, n_q as u32, n_tok as u32, 1);

        // barrier
        let bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        ctx.device.cmd_pipeline_barrier(cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(), &[bar], &[], &[]);

        // 2) softmax (n_q * n_tok rows, each of length seq_len)
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe_sm.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe_sm.layout, 0, &[bind_sm.set], &[]);
        let p_sm = PushSm { dim: seq_len as u32 };
        let pb = std::slice::from_raw_parts(
            &p_sm as *const _ as *const u8, std::mem::size_of::<PushSm>());
        ctx.device.cmd_push_constants(cb, pipe_sm.layout,
            vk::ShaderStageFlags::COMPUTE, 0, pb);
        ctx.device.cmd_dispatch(cb, (n_q * n_tok) as u32, 1, 1);

        ctx.device.cmd_pipeline_barrier(cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(), &[bar], &[], &[]);

        // 3) attn_v
        ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe_av.pipeline);
        ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
            pipe_av.layout, 0, &[bind_av.set], &[]);
        let p_av = PushAV {
            n_tok: n_tok as u32, n_q: n_q as u32, n_kv: n_kv as u32,
            head_dim: head_dim as u32, seq_len: seq_len as u32,
        };
        let pb = std::slice::from_raw_parts(
            &p_av as *const _ as *const u8, std::mem::size_of::<PushAV>());
        ctx.device.cmd_push_constants(cb, pipe_av.layout,
            vk::ShaderStageFlags::COMPUTE, 0, pb);
        ctx.device.cmd_dispatch(cb, n_q as u32, n_tok as u32, 1);

        ctx.device.end_command_buffer(cb)?;

        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        let t = Instant::now();
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let gpu_out = std::slice::from_raw_parts(out_buf.mapped() as *const f32,
            n_tok * n_q * head_dim);
        let mut max_abs = 0.0_f32;
        let mut argmax = 0;
        for i in 0..gpu_out.len() {
            let e = (gpu_out[i] - cpu_out[i]).abs();
            if e > max_abs { max_abs = e; argmax = i; }
        }
        println!("GPU pipeline: scaled_dot → softmax → attn_v: {:.2} ms", gpu_ms);
        println!("max abs err: {:.2e}  at index {} (cpu={}, gpu={})",
            max_abs, argmax, cpu_out[argmax], gpu_out[argmax]);
        if max_abs > 1e-5 {
            println!("first 8 cpu: {:?}", &cpu_out[..8]);
            println!("first 8 gpu: {:?}", &gpu_out[..8]);
            bail!("attention mismatch");
        }
        println!("✅ GPU attention block matches CPU reference");
        println!("\nFirst 8 outputs: {:?}", &gpu_out[..8]);

        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }

    bind_av.destroy(&ctx); bind_sm.destroy(&ctx); bind_sd.destroy(&ctx);
    pipe_av.destroy(&ctx); pipe_sm.destroy(&ctx); pipe_sd.destroy(&ctx);
    out_buf.destroy(&ctx); scores_buf.destroy(&ctx);
    v_buf.destroy(&ctx); k_buf.destroy(&ctx); q_buf.destroy(&ctx);
    Ok(())
}

fn slice_as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
