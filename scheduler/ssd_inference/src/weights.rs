//! One-shot startup load of every non-expert tensor into VRAM as fp32.
//!
//! Strategy:
//!   - F32 tensors: staging upload, no dequant.
//!   - Q4_K / Q6_K tensors: GPU dequant (one dispatch per tensor).
//! Expert tensors (`ffn_*_exps`) stay on disk and load through `ExpertLoader`.
//!
//! Memory budget for Qwen3-30B-A3B Q4_K_M (verified with `dump_metadata`):
//!   embed         151936 × 2048 × 4 = 1.18 GB
//!   lm_head       151936 × 2048 × 4 = 1.18 GB
//!   attn weights  48 layers × ~73 MB ≈ 3.5 GB
//!   norms         <50 MB
//!   total         ≈ 5.9 GB out of 8 GB VRAM
//!   leaves room for KV cache (~100 MB) + expert pool (~500 MB) + activations.

use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use candle_core::quantized::GgmlDType;
use gguf_reader::GgufFile;
use std::time::Instant;

use crate::buffer::GpuBuffer;
use crate::compute::{ComputePipeline, DescriptorBinder, DEQUANT_Q4_K_SPV, DEQUANT_Q6_K_SPV};
use crate::device::VulkanContext;
use crate::model::{ModelConfig, TensorNames};

const QK_K: usize = 256;
const Q4K_BYTES: usize = 144;
const Q6K_BYTES: usize = 210;pub struct LayerWeights {
    pub attn_norm: GpuBuffer,    // [d_model] fp32
    pub attn_q_norm: GpuBuffer,  // [head_dim] fp32
    pub attn_k_norm: GpuBuffer,  // [head_dim] fp32
    pub attn_q: GpuBuffer,       // [n_q_heads*head_dim, d_model] fp32
    pub attn_k: GpuBuffer,       // [n_kv_heads*head_dim, d_model] fp32
    pub attn_v: GpuBuffer,       // [n_kv_heads*head_dim, d_model] fp32
    pub attn_o: GpuBuffer,       // [d_model, n_q_heads*head_dim] fp32
    pub ffn_norm: GpuBuffer,     // [d_model] fp32
    pub router: GpuBuffer,       // [n_experts, d_model] fp32
}

pub struct LoadedWeights {
    /// Embedding stays in HOST RAM (1.18 GB fp32 for Qwen3-30B). Per-token
    /// rows get staged to GPU on demand at prefill time. This frees the
    /// equivalent VRAM for attention weights.
    pub embed_host: Vec<f32>,
    /// lm_head also stays on host. Final argmax is one CPU matvec
    /// (vocab × d_model fp32 dot products), ~150 ms on a modern CPU.
    pub lm_head_host: Vec<f32>,
    pub d_model: u32,
    pub vocab: u32,
    pub out_norm: GpuBuffer,     // [d_model] fp32
    pub layers: Vec<LayerWeights>,
    pub bytes_loaded: u64,
}

pub struct WeightLoader {
    q4k_pipe: ComputePipeline,
    q6k_pipe: ComputePipeline,
    cmd_pool: vk::CommandPool,
}

impl WeightLoader {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let q4k_pipe = ComputePipeline::new(ctx, DEQUANT_Q4_K_SPV, 2, 4)?;
        let q6k_pipe = ComputePipeline::new(ctx, DEQUANT_Q6_K_SPV, 2, 4)?;
        let cmd_pool = unsafe { ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None)? };
        Ok(Self { q4k_pipe, q6k_pipe, cmd_pool })
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe { ctx.device.destroy_command_pool(self.cmd_pool, None); }
        self.q4k_pipe.destroy(ctx);
        self.q6k_pipe.destroy(ctx);
    }

    /// Load ONE tensor into a fresh DEVICE_LOCAL fp32 GpuBuffer. Returns the
    /// buffer plus the number of bytes read from the GGUF file.
    pub fn load_tensor(&self, ctx: &VulkanContext, g: &GgufFile, name: &str) -> Result<(GpuBuffer, u64)> {
        let info = g.tensor_info(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let bytes = g.tensor_bytes(name)?;
        let n_weights: usize = info.shape.dims().iter().product();
        let dst_size = (n_weights * 4) as u64;
        let dt = info.ggml_dtype;

        let dst = GpuBuffer::new_vram(ctx, dst_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;

        match dt {
            GgmlDType::F32 => {
                // Direct upload: staging → VRAM via cmd_copy_buffer.
                let staging = GpuBuffer::new_staging(ctx, bytes.len() as u64)?;
                unsafe { staging.write_at(0, bytes); }
                self.copy_buf_to_buf(ctx, &staging, &dst, bytes.len() as u64)?;
                staging.destroy(ctx);
            }
            GgmlDType::Q4K => {
                if bytes.len() % Q4K_BYTES != 0 {
                    bail!("{name}: Q4K size {} not multiple of {}", bytes.len(), Q4K_BYTES);
                }
                let n_blocks = bytes.len() / Q4K_BYTES;
                if n_blocks * QK_K != n_weights {
                    bail!("{name}: Q4K weight count mismatch ({} blocks * 256 != {})", n_blocks, n_weights);
                }
                self.dequant_to_dst(ctx, &self.q4k_pipe, bytes, n_blocks as u32, &dst, dst_size)?;
            }
            GgmlDType::Q6K => {
                if bytes.len() % Q6K_BYTES != 0 {
                    bail!("{name}: Q6K size {} not multiple of {}", bytes.len(), Q6K_BYTES);
                }
                let n_blocks = bytes.len() / Q6K_BYTES;
                if n_blocks * QK_K != n_weights {
                    bail!("{name}: Q6K weight count mismatch", );
                }
                self.dequant_to_dst(ctx, &self.q6k_pipe, bytes, n_blocks as u32, &dst, dst_size)?;
            }
            other => bail!("{name}: unsupported dtype {:?}", other),
        }

        Ok((dst, bytes.len() as u64))
    }

    fn dequant_to_dst(
        &self,
        ctx: &VulkanContext,
        pipe: &ComputePipeline,
        bytes: &[u8],
        n_blocks: u32,
        dst: &GpuBuffer,
        dst_size: u64,
    ) -> Result<()> {
        // Q packed bytes: HOST_VISIBLE staging serves as both upload buffer and
        // the GPU-readable Q source (tiny perf hit but avoids an intermediate copy).
        let staging = GpuBuffer::new(ctx, bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true)?;
        unsafe { staging.write_at(0, bytes); }

        let binder = DescriptorBinder::new(ctx, pipe, &[
            (&staging, bytes.len() as u64),
            (dst, dst_size),
        ])?;

        unsafe {
            let cb = ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1))?[0];
            ctx.device.begin_command_buffer(cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
                pipe.layout, 0, &[binder.set], &[]);
            let push = n_blocks.to_le_bytes();
            ctx.device.cmd_push_constants(cb, pipe.layout,
                vk::ShaderStageFlags::COMPUTE, 0, &push);
            ctx.device.cmd_dispatch(cb, n_blocks, 1, 1);
            ctx.device.end_command_buffer(cb)?;

            let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cb_arr = [cb];
            ctx.device.queue_submit(ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            ctx.device.destroy_fence(fence, None);
            ctx.device.free_command_buffers(self.cmd_pool, &[cb]);
        }

        binder.destroy(ctx);
        staging.destroy(ctx);
        Ok(())
    }

    fn copy_buf_to_buf(
        &self,
        ctx: &VulkanContext,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        size: u64,
    ) -> Result<()> {
        unsafe {
            let cb = ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1))?[0];
            ctx.device.begin_command_buffer(cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let region = [vk::BufferCopy { src_offset: 0, dst_offset: 0, size }];
            ctx.device.cmd_copy_buffer(cb, src.handle(), dst.handle(), &region);
            ctx.device.end_command_buffer(cb)?;

            let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cb_arr = [cb];
            ctx.device.queue_submit(ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            ctx.device.destroy_fence(fence, None);
            ctx.device.free_command_buffers(self.cmd_pool, &[cb]);
        }
        Ok(())
    }
}

impl LoadedWeights {
    pub fn load(ctx: &VulkanContext, g: &GgufFile, cfg: &ModelConfig, verbose: bool) -> Result<Self> {
        let loader = WeightLoader::new(ctx)?;
        let t = Instant::now();
        let mut bytes_loaded = 0u64;

        let log = |name: &str, bytes: u64| {
            if verbose { println!("  {:<42} {:>10} bytes", name, bytes); }
        };

        // ── Embedding stays on host (saves 1.18 GB VRAM) ──
        let embed_t = Instant::now();
        let embed_info = g.tensor_info(TensorNames::EMBED)
            .ok_or_else(|| anyhow!("missing embed"))?;
        let embed_bytes = g.tensor_bytes(TensorNames::EMBED)?;
        let embed_host = match embed_info.ggml_dtype {
            GgmlDType::F32 => {
                // 4 bytes per fp32 — direct cast
                let n = embed_bytes.len() / 4;
                let mut out = vec![0f32; n];
                unsafe { std::ptr::copy_nonoverlapping(
                    embed_bytes.as_ptr() as *const f32, out.as_mut_ptr(), n); }
                out
            }
            GgmlDType::Q4K => cpu_dequant_q4_k(embed_bytes),
            GgmlDType::Q6K => cpu_dequant_q6_k(embed_bytes),
            other => bail!("unsupported embed dtype {:?}", other),
        };
        bytes_loaded += embed_bytes.len() as u64;
        println!("  embed (host RAM): {:.1} MB Q → {:.1} MB fp32 in {:.2} s",
            embed_bytes.len() as f64 / 1024.0 / 1024.0,
            embed_host.len() as f64 * 4.0 / 1024.0 / 1024.0,
            embed_t.elapsed().as_secs_f64());

        let (out_norm, b) = loader.load_tensor(ctx, g, TensorNames::OUT_NORM)?;  bytes_loaded += b; log(TensorNames::OUT_NORM, b);

        // ── lm_head also on host (saves another 1.18 GB VRAM) ──
        let lm_t = Instant::now();
        let lm_info = g.tensor_info(TensorNames::LM_HEAD)
            .ok_or_else(|| anyhow!("missing lm_head"))?;
        let lm_bytes = g.tensor_bytes(TensorNames::LM_HEAD)?;
        let lm_head_host = match lm_info.ggml_dtype {
            GgmlDType::F32 => {
                let n = lm_bytes.len() / 4;
                let mut out = vec![0f32; n];
                unsafe { std::ptr::copy_nonoverlapping(
                    lm_bytes.as_ptr() as *const f32, out.as_mut_ptr(), n); }
                out
            }
            GgmlDType::Q4K => cpu_dequant_q4_k(lm_bytes),
            GgmlDType::Q6K => cpu_dequant_q6_k(lm_bytes),
            other => bail!("unsupported lm_head dtype {:?}", other),
        };
        bytes_loaded += lm_bytes.len() as u64;
        println!("  lm_head (host RAM): {:.1} MB Q → {:.1} MB fp32 in {:.2} s",
            lm_bytes.len() as f64 / 1024.0 / 1024.0,
            lm_head_host.len() as f64 * 4.0 / 1024.0 / 1024.0,
            lm_t.elapsed().as_secs_f64());

        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        for l in 0..cfg.n_layer {
            let (attn_norm, b)   = loader.load_tensor(ctx, g, &TensorNames::attn_norm(l))?;   bytes_loaded += b;
            let (attn_q_norm, b) = loader.load_tensor(ctx, g, &TensorNames::attn_q_norm(l))?; bytes_loaded += b;
            let (attn_k_norm, b) = loader.load_tensor(ctx, g, &TensorNames::attn_k_norm(l))?; bytes_loaded += b;
            let (attn_q, b)      = loader.load_tensor(ctx, g, &TensorNames::attn_q(l))?;      bytes_loaded += b;
            let (attn_k, b)      = loader.load_tensor(ctx, g, &TensorNames::attn_k(l))?;      bytes_loaded += b;
            let (attn_v, b)      = loader.load_tensor(ctx, g, &TensorNames::attn_v(l))?;      bytes_loaded += b;
            let (attn_o, b)      = loader.load_tensor(ctx, g, &TensorNames::attn_o(l))?;      bytes_loaded += b;
            let (ffn_norm, b)    = loader.load_tensor(ctx, g, &TensorNames::ffn_norm(l))?;    bytes_loaded += b;
            let (router, b)      = loader.load_tensor(ctx, g, &TensorNames::router(l))?;      bytes_loaded += b;

            layers.push(LayerWeights {
                attn_norm, attn_q_norm, attn_k_norm,
                attn_q, attn_k, attn_v, attn_o,
                ffn_norm, router,
            });
            if verbose && (l + 1) % 8 == 0 {
                println!("  ... layer {}/{} done", l + 1, cfg.n_layer);
            }
        }

        let dt = t.elapsed().as_secs_f64();
        println!("Loaded {:.1} MB GGUF in {:.2} s", bytes_loaded as f64 / 1024.0 / 1024.0, dt);

        loader.destroy(ctx);

        Ok(Self {
            embed_host, lm_head_host,
            d_model: cfg.d_model, vocab: cfg.vocab,
            out_norm, layers, bytes_loaded,
        })
    }

    /// Get the fp32 embedding row for a token id (host slice).
    pub fn embed_row(&self, token_id: u32) -> &[f32] {
        let d = self.d_model as usize;
        let start = token_id as usize * d;
        &self.embed_host[start..start + d]
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        self.out_norm.destroy(ctx);
        for l in &self.layers {
            l.attn_norm.destroy(ctx);
            l.attn_q_norm.destroy(ctx);
            l.attn_k_norm.destroy(ctx);
            l.attn_q.destroy(ctx);
            l.attn_k.destroy(ctx);
            l.attn_v.destroy(ctx);
            l.attn_o.destroy(ctx);
            l.ffn_norm.destroy(ctx);
            l.router.destroy(ctx);
        }
    }
}

// ── CPU dequant helpers (also used by examples) ──

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

fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4)  | ((scales[j]     >> 6) << 4);
        (d, m)
    }
}

pub fn cpu_dequant_q4_k(blocks: &[u8]) -> Vec<f32> {
    let nb = blocks.len() / Q4K_BYTES;
    let mut y = vec![0f32; nb * QK_K];
    for i in 0..nb {
        let blk = &blocks[i * Q4K_BYTES..(i + 1) * Q4K_BYTES];
        let d = fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = fp16_to_fp32(u16::from_le_bytes([blk[2], blk[3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&blk[4..16]);
        let qs = &blk[16..144];
        let mut is = 0;
        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is + 0, &scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let d1 = d * sc1 as f32; let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32; let m2f = dmin * m2 as f32;
            let q_off = j / 2;
            for l in 0..32 { y[i * QK_K + j + l]      = d1 * (qs[q_off + l] & 0xF) as f32 - m1f; }
            for l in 0..32 { y[i * QK_K + j + 32 + l] = d2 * (qs[q_off + l]  >> 4) as f32 - m2f; }
            is += 2;
        }
    }
    y
}

pub fn cpu_dequant_q6_k(blocks: &[u8]) -> Vec<f32> {
    let nb = blocks.len() / Q6K_BYTES;
    let mut y = vec![0f32; nb * QK_K];
    for i in 0..nb {
        let blk = &blocks[i * Q6K_BYTES..(i + 1) * Q6K_BYTES];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc: &[i8] = unsafe {
            std::slice::from_raw_parts(blk[192..208].as_ptr() as *const i8, 16)
        };
        let d = fp16_to_fp32(u16::from_le_bytes([blk[208], blk[209]]));
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
                y[y_base + l]      = d * sc[sc_base + is + 0] as f32 * q1 as f32;
                y[y_base + l + 32] = d * sc[sc_base + is + 2] as f32 * q2 as f32;
                y[y_base + l + 64] = d * sc[sc_base + is + 4] as f32 * q3 as f32;
                y[y_base + l + 96] = d * sc[sc_base + is + 6] as f32 * q4 as f32;
            }
        }
    }
    y
}

/// Single-row fused Q6K dot: dot(dequant(q6k_row), x). Used for lm_head argmax.
pub fn fused_dot_q6k_single(row_q: &[u8], x: &[f32], n_in: usize) -> f32 {
    let bpr = n_in / QK_K;
    let mut acc = 0f32;
    for b in 0..bpr {
        let blk = &row_q[b * Q6K_BYTES..(b + 1) * Q6K_BYTES];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc: &[i8] = unsafe { std::slice::from_raw_parts(blk[192..208].as_ptr() as *const i8, 16) };
        let d_val = fp16_to_fp32(u16::from_le_bytes([blk[208], blk[209]]));
        let xb = b * QK_K;
        for nn in 0..2usize {
            let qlb = nn * 64; let qhb = nn * 32; let scb = nn * 8; let yb = nn * 128;
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[qlb+l]&0xF)|((qh[qhb+l]>>0&3)<<4)) as i32 - 32;
                let q2 = ((ql[qlb+l+32]&0xF)|((qh[qhb+l]>>2&3)<<4)) as i32 - 32;
                let q3 = ((ql[qlb+l]>>4)|((qh[qhb+l]>>4&3)<<4)) as i32 - 32;
                let q4 = ((ql[qlb+l+32]>>4)|((qh[qhb+l]>>6&3)<<4)) as i32 - 32;
                acc += d_val * sc[scb+is] as f32 * q1 as f32 * x[xb+yb+l];
                acc += d_val * sc[scb+is+2] as f32 * q2 as f32 * x[xb+yb+l+32];
                acc += d_val * sc[scb+is+4] as f32 * q3 as f32 * x[xb+yb+l+64];
                acc += d_val * sc[scb+is+6] as f32 * q4 as f32 * x[xb+yb+l+96];
            }
        }
    }
    acc
}
pub fn fused_matvec_q4k(w_q: &[u8], x: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    let bpr = n_in / QK_K;
    let row_bytes = bpr * Q4K_BYTES;
    let mut y = vec![0f32; n_out];
    for n in 0..n_out {
        let row = &w_q[n * row_bytes..(n + 1) * row_bytes];
        let mut acc = 0f32;
        for b in 0..bpr {
            let blk = &row[b * Q4K_BYTES..(b + 1) * Q4K_BYTES];
            let d = fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]]));
            let dmin = fp16_to_fp32(u16::from_le_bytes([blk[2], blk[3]]));
            let mut scales = [0u8; 12];
            scales.copy_from_slice(&blk[4..16]);
            let qs = &blk[16..144];
            let xb = b * QK_K;
            let mut is = 0;
            for j in (0..QK_K).step_by(64) {
                let (sc1, m1) = get_scale_min_k4(is, &scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
                let d1 = d * sc1 as f32; let m1f = dmin * m1 as f32;
                let d2 = d * sc2 as f32; let m2f = dmin * m2 as f32;
                let qo = j / 2;
                for l in 0..32 {
                    acc += (d1 * (qs[qo + l] & 0xF) as f32 - m1f) * x[xb + j + l];
                }
                for l in 0..32 {
                    acc += (d2 * (qs[qo + l] >> 4) as f32 - m2f) * x[xb + j + 32 + l];
                }
                is += 2;
            }
        }
        y[n] = acc;
    }
    y
}

/// Fused Q6K dequant + matvec.
pub fn fused_matvec_q6k(w_q: &[u8], x: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    let bpr = n_in / QK_K;
    let row_bytes = bpr * Q6K_BYTES;
    let mut y = vec![0f32; n_out];
    for n in 0..n_out {
        let row = &w_q[n * row_bytes..(n + 1) * row_bytes];
        let mut acc = 0f32;
        for b in 0..bpr {
            let blk = &row[b * Q6K_BYTES..(b + 1) * Q6K_BYTES];
            let ql = &blk[0..128];
            let qh = &blk[128..192];
            let sc: &[i8] = unsafe { std::slice::from_raw_parts(blk[192..208].as_ptr() as *const i8, 16) };
            let d_val = fp16_to_fp32(u16::from_le_bytes([blk[208], blk[209]]));
            let xb = b * QK_K;
            for nn in 0..2usize {
                let qlb = nn * 64; let qhb = nn * 32; let scb = nn * 8; let yb = nn * 128;
                for l in 0..32usize {
                    let is = l / 16;
                    let q1 = ((ql[qlb+l]&0xF)|((qh[qhb+l]>>0&3)<<4)) as i32 - 32;
                    let q2 = ((ql[qlb+l+32]&0xF)|((qh[qhb+l]>>2&3)<<4)) as i32 - 32;
                    let q3 = ((ql[qlb+l]>>4)|((qh[qhb+l]>>4&3)<<4)) as i32 - 32;
                    let q4 = ((ql[qlb+l+32]>>4)|((qh[qhb+l]>>6&3)<<4)) as i32 - 32;
                    acc += d_val * sc[scb+is] as f32 * q1 as f32 * x[xb+yb+l];
                    acc += d_val * sc[scb+is+2] as f32 * q2 as f32 * x[xb+yb+l+32];
                    acc += d_val * sc[scb+is+4] as f32 * q3 as f32 * x[xb+yb+l+64];
                    acc += d_val * sc[scb+is+6] as f32 * q4 as f32 * x[xb+yb+l+96];
                }
            }
        }
        y[n] = acc;
    }
    y
}
