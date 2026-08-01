//! Streaming weight loader v2: pre-loads attention Q4K into RAM, uploads
//! Q4K bytes directly to VRAM (no fp32 dequant). Forward uses fused_q4k_matvec.
//!
//! VRAM per layer: ~41 MB Q4K (vs 272 MB fp32 in v1) → massive VRAM savings.
//! RAM usage: ~3.8 GB for attention Q4K + ~5 GB embed/lm_head.

use anyhow::{anyhow, bail, Result};
use ash::vk;
use candle_core::quantized::GgmlDType;
use gguf_reader::MultiGgufFile;
use std::time::Instant;

use crate::buffer::GpuBuffer;
use crate::device::VulkanContext;
use crate::model::{ModelConfig, TensorNames};
use crate::weights::{cpu_dequant_q4_k, cpu_dequant_q6_k};

const QK_K: usize = 256;
const Q4K_BYTES: usize = 144;
const Q5K_BYTES: usize = 176;
const Q6K_BYTES: usize = 210;

/// Pre-loaded attention Q4K bytes for one layer (in RAM).
pub struct LayerQBytes {
    pub attn_q: Vec<u8>,     // Q4K bytes OR fp32 bytes
    pub attn_k: Vec<u8>,
    pub attn_v: Vec<u8>,
    pub attn_o: Vec<u8>,
    pub attn_norm: Vec<u8>,
    pub attn_q_norm: Vec<u8>,
    pub attn_k_norm: Vec<u8>,
    pub ffn_norm: Vec<u8>,
    pub router: Vec<u8>,
    pub is_q4k: bool,        // true = Q4K bytes, false = fp32 bytes
}

/// Reusable VRAM buffers: Q4K bytes (not fp32!) for attention, fp32 for norms/router.
pub struct StreamingLayerBufs {
    pub attn_q: GpuBuffer,       // Q4K bytes in VRAM
    pub attn_k: GpuBuffer,
    pub attn_v: GpuBuffer,
    pub attn_o: GpuBuffer,
    // fp32 fallback buffers for non-Q4K layers
    pub attn_q_fp32: GpuBuffer,
    pub attn_k_fp32: GpuBuffer,
    pub attn_v_fp32: GpuBuffer,
    pub attn_o_fp32: GpuBuffer,
    pub attn_norm: GpuBuffer,
    pub attn_q_norm: GpuBuffer,
    pub attn_k_norm: GpuBuffer,
    pub ffn_norm: GpuBuffer,
    pub router: GpuBuffer,
}

pub struct StreamingWeights {
    pub embed_host: Vec<f32>,
    pub lm_head_host: Vec<f32>,
    pub d_model: u32,
    pub vocab: u32,
    pub out_norm: GpuBuffer,
    pub layer_bufs: StreamingLayerBufs,
    pub layers_ram: Vec<LayerQBytes>,
    staging: GpuBuffer,
    cmd_pool: vk::CommandPool,
    pub current_layer: Option<u32>,
}

impl StreamingWeights {
    pub fn new(ctx: &VulkanContext, mg: &MultiGgufFile, cfg: &ModelConfig) -> Result<Self> {
        let d = cfg.d_model as u64;
        let nq = cfg.n_q_heads as u64;
        let nkv = cfg.n_kv_heads as u64;
        let hd = cfg.head_dim as u64;
        let n_exp = cfg.n_experts as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        let q4k_size = |n_weights: u64| (n_weights / QK_K as u64) * Q4K_BYTES as u64;

        let layer_bufs = StreamingLayerBufs {
            attn_q:      GpuBuffer::new_vram(ctx, q4k_size(nq * hd * d), usage)?,
            attn_k:      GpuBuffer::new_vram(ctx, q4k_size(nkv * hd * d), usage)?,
            attn_v:      GpuBuffer::new_vram(ctx, q4k_size(nkv * hd * d), usage)?,
            attn_o:      GpuBuffer::new_vram(ctx, q4k_size(d * nq * hd), usage)?,
            attn_q_fp32: GpuBuffer::new_vram(ctx, nq * hd * d * 4, usage)?,
            attn_k_fp32: GpuBuffer::new_vram(ctx, nkv * hd * d * 4, usage)?,
            attn_v_fp32: GpuBuffer::new_vram(ctx, nkv * hd * d * 4, usage)?,
            attn_o_fp32: GpuBuffer::new_vram(ctx, d * nq * hd * 4, usage)?,
            attn_norm:   GpuBuffer::new_vram(ctx, d * 4, usage)?,
            attn_q_norm: GpuBuffer::new_vram(ctx, hd * 4, usage)?,
            attn_k_norm: GpuBuffer::new_vram(ctx, hd * 4, usage)?,
            ffn_norm:    GpuBuffer::new_vram(ctx, d * 4, usage)?,
            router:      GpuBuffer::new_vram(ctx, n_exp * d * 4, usage)?,
        };

        // Staging: big enough for largest single tensor (attn_q Q6K or router fp32)
        let max_staging = std::cmp::max(nq * hd * d * 4, n_exp * d * 4); // fp32 attn_q is biggest
        let staging = GpuBuffer::new(ctx, max_staging,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::empty(), true)?;

        let cmd_pool = unsafe { ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None)? };

        // Pre-load embed + lm_head to host RAM
        let t0 = Instant::now();
        let embed_bytes = mg.tensor_bytes(TensorNames::EMBED)?;
        let embed_info = mg.tensor_info(TensorNames::EMBED).unwrap();
        let embed_host = cpu_dequant_any(embed_bytes, embed_info.ggml_dtype)?;
        println!("  embed (host): {:.1} MB → {:.1} MB fp32 in {:.2}s",
            embed_bytes.len() as f64 / 1e6, embed_host.len() as f64 * 4.0 / 1e6, t0.elapsed().as_secs_f64());

        let t1 = Instant::now();
        let lm_bytes = mg.tensor_bytes(TensorNames::LM_HEAD)?;
        let lm_info = mg.tensor_info(TensorNames::LM_HEAD).unwrap();
        let lm_head_host = cpu_dequant_any(lm_bytes, lm_info.ggml_dtype)?;
        println!("  lm_head (host): {:.1} MB → {:.1} MB fp32 in {:.2}s",
            lm_bytes.len() as f64 / 1e6, lm_head_host.len() as f64 * 4.0 / 1e6, t1.elapsed().as_secs_f64());

        // out_norm (small, permanent VRAM)
        let norm_bytes = mg.tensor_bytes(TensorNames::OUT_NORM)?;
        let out_norm = upload_f32_buf(ctx, norm_bytes, usage)?;

        // Pre-load ALL attention Q bytes into RAM
        let t2 = Instant::now();
        let mut layers_ram = Vec::with_capacity(cfg.n_layer as usize);
        let mut total_ram_bytes = 0u64;
        let mut n_q4k_layers = 0u32;
        let mut n_fp32_layers = 0u32;
        for l in 0..cfg.n_layer {
            let load_info = |name: &str| -> Result<(Vec<u8>, GgmlDType)> {
                let info = mg.tensor_info(name)
                    .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
                let bytes = mg.tensor_bytes(name)?;
                Ok((bytes.to_vec(), info.ggml_dtype))
            };

            let (aq_raw, aq_dt) = load_info(&TensorNames::attn_q(l))?;
            let (ak_raw, ak_dt) = load_info(&TensorNames::attn_k(l))?;
            let (av_raw, av_dt) = load_info(&TensorNames::attn_v(l))?;
            let (ao_raw, ao_dt) = load_info(&TensorNames::attn_o(l))?;

            let all_q4k = aq_dt == GgmlDType::Q4K && ak_dt == GgmlDType::Q4K
                       && av_dt == GgmlDType::Q4K && ao_dt == GgmlDType::Q4K;

            let (aq, ak, av, ao) = if all_q4k {
                n_q4k_layers += 1;
                (aq_raw, ak_raw, av_raw, ao_raw)
            } else {
                n_fp32_layers += 1;
                // CPU dequant to fp32 bytes for non-Q4K layers
                let to_fp32_bytes = |raw: &[u8], dt: GgmlDType| -> Vec<u8> {
                    let fp32 = cpu_dequant_any(raw, dt).unwrap();
                    unsafe { std::slice::from_raw_parts(fp32.as_ptr() as *const u8, fp32.len() * 4).to_vec() }
                };
                (to_fp32_bytes(&aq_raw, aq_dt), to_fp32_bytes(&ak_raw, ak_dt),
                 to_fp32_bytes(&av_raw, av_dt), to_fp32_bytes(&ao_raw, ao_dt))
            };

            let an = mg.tensor_bytes(&TensorNames::attn_norm(l))?.to_vec();
            let aqn = mg.tensor_bytes(&TensorNames::attn_q_norm(l))?.to_vec();
            let akn = mg.tensor_bytes(&TensorNames::attn_k_norm(l))?.to_vec();
            let fn_ = mg.tensor_bytes(&TensorNames::ffn_norm(l))?.to_vec();
            let rt = mg.tensor_bytes(&TensorNames::router(l))?.to_vec();

            total_ram_bytes += (aq.len() + ak.len() + av.len() + ao.len()
                + an.len() + aqn.len() + akn.len() + fn_.len() + rt.len()) as u64;

            layers_ram.push(LayerQBytes {
                attn_q: aq, attn_k: ak, attn_v: av, attn_o: ao,
                attn_norm: an, attn_q_norm: aqn, attn_k_norm: akn,
                ffn_norm: fn_, router: rt,
                is_q4k: all_q4k,
            });
        }
        println!("  attention (RAM): {:.1} MB for {} layers ({} Q4K + {} fp32) in {:.2}s",
            total_ram_bytes as f64 / 1e6, cfg.n_layer, n_q4k_layers, n_fp32_layers,
            t2.elapsed().as_secs_f64());

        Ok(Self {
            embed_host, lm_head_host,
            d_model: cfg.d_model, vocab: cfg.vocab,
            out_norm, layer_bufs, layers_ram, staging,
            cmd_pool, current_layer: None,
        })
    }

    /// Upload one layer's Q4K bytes from RAM to VRAM. No GPU dequant — just memcpy.
    /// Norms/router are fp32, uploaded directly.
    pub fn upload_layer(&mut self, ctx: &VulkanContext, layer: u32) -> Result<()> {
        if self.current_layer == Some(layer) { return Ok(()); }

        let lr = &self.layers_ram[layer as usize];

        if lr.is_q4k {
            self.upload_raw(ctx, &lr.attn_q, &self.layer_bufs.attn_q)?;
            self.upload_raw(ctx, &lr.attn_k, &self.layer_bufs.attn_k)?;
            self.upload_raw(ctx, &lr.attn_v, &self.layer_bufs.attn_v)?;
            self.upload_raw(ctx, &lr.attn_o, &self.layer_bufs.attn_o)?;
        } else {
            self.upload_raw(ctx, &lr.attn_q, &self.layer_bufs.attn_q_fp32)?;
            self.upload_raw(ctx, &lr.attn_k, &self.layer_bufs.attn_k_fp32)?;
            self.upload_raw(ctx, &lr.attn_v, &self.layer_bufs.attn_v_fp32)?;
            self.upload_raw(ctx, &lr.attn_o, &self.layer_bufs.attn_o_fp32)?;
        }

        self.upload_f32_or_dequant(ctx, &lr.attn_norm, &self.layer_bufs.attn_norm)?;
        self.upload_f32_or_dequant(ctx, &lr.attn_q_norm, &self.layer_bufs.attn_q_norm)?;
        self.upload_f32_or_dequant(ctx, &lr.attn_k_norm, &self.layer_bufs.attn_k_norm)?;
        self.upload_f32_or_dequant(ctx, &lr.ffn_norm, &self.layer_bufs.ffn_norm)?;
        self.upload_f32_or_dequant(ctx, &lr.router, &self.layer_bufs.router)?;

        self.current_layer = Some(layer);
        Ok(())
    }

    fn upload_raw(&self, ctx: &VulkanContext, bytes: &[u8], dst: &GpuBuffer) -> Result<()> {
        let sz = bytes.len() as u64;
        assert!(sz <= self.staging.size(), "upload_raw: {} > staging {}", sz, self.staging.size());
        unsafe { self.staging.write_at(0, bytes); }
        self.cmd_copy(ctx, &self.staging, dst, sz)
    }

    fn upload_f32_or_dequant(&self, ctx: &VulkanContext, bytes: &[u8], dst: &GpuBuffer) -> Result<()> {
        // Norms/router are always F32 in Qwen3 GGUF, direct upload
        assert!(bytes.len() as u64 <= self.staging.size());
        unsafe { self.staging.write_at(0, bytes); }
        self.cmd_copy(ctx, &self.staging, dst, bytes.len() as u64)
    }

    fn cmd_copy(&self, ctx: &VulkanContext, src: &GpuBuffer, dst: &GpuBuffer, size: u64) -> Result<()> {
        unsafe {
            let cb = ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1))?[0];
            ctx.device.begin_command_buffer(cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            ctx.device.cmd_copy_buffer(cb, src.handle(), dst.handle(),
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
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

    pub fn embed_row(&self, token_id: u32) -> &[f32] {
        let d = self.d_model as usize;
        let start = token_id as usize * d;
        &self.embed_host[start..start + d]
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        self.out_norm.destroy(ctx);
        self.layer_bufs.attn_q.destroy(ctx);
        self.layer_bufs.attn_k.destroy(ctx);
        self.layer_bufs.attn_v.destroy(ctx);
        self.layer_bufs.attn_o.destroy(ctx);
        self.layer_bufs.attn_q_fp32.destroy(ctx);
        self.layer_bufs.attn_k_fp32.destroy(ctx);
        self.layer_bufs.attn_v_fp32.destroy(ctx);
        self.layer_bufs.attn_o_fp32.destroy(ctx);
        self.layer_bufs.attn_norm.destroy(ctx);
        self.layer_bufs.attn_q_norm.destroy(ctx);
        self.layer_bufs.attn_k_norm.destroy(ctx);
        self.layer_bufs.ffn_norm.destroy(ctx);
        self.layer_bufs.router.destroy(ctx);
        self.staging.destroy(ctx);
        unsafe { ctx.device.destroy_command_pool(self.cmd_pool, None); }
    }
}

fn upload_f32_buf(ctx: &VulkanContext, bytes: &[u8], usage: vk::BufferUsageFlags) -> Result<GpuBuffer> {
    let dst = GpuBuffer::new_vram(ctx, bytes.len() as u64, usage)?;
    let staging = GpuBuffer::new_staging(ctx, bytes.len() as u64)?;
    unsafe { staging.write_at(0, bytes); }
    unsafe {
        let cmd_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT), None)?;
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1))?[0];
        ctx.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        ctx.device.cmd_copy_buffer(cb, staging.handle(), dst.handle(),
            &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes.len() as u64 }]);
        ctx.device.end_command_buffer(cb)?;
        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }
    staging.destroy(ctx);
    Ok(dst)
}

fn cpu_dequant_any(bytes: &[u8], dtype: GgmlDType) -> Result<Vec<f32>> {
    match dtype {
        GgmlDType::F32 => {
            let n = bytes.len() / 4;
            let mut out = vec![0f32; n];
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const f32, out.as_mut_ptr(), n); }
            Ok(out)
        }
        GgmlDType::Q4K => Ok(cpu_dequant_q4_k(bytes)),
        GgmlDType::Q6K => Ok(cpu_dequant_q6_k(bytes)),
        GgmlDType::Q5K => Ok(cpu_dequant_q5_k(bytes)),
        GgmlDType::Q2K => Ok(cpu_dequant_q2_k(bytes)),
        GgmlDType::Q3K => Ok(cpu_dequant_q3_k(bytes)),
        other => bail!("unsupported dtype for CPU dequant: {:?}", other),
    }
}

pub fn cpu_dequant_q5_k(blocks: &[u8]) -> Vec<f32> {
    let nb = blocks.len() / Q5K_BYTES;
    let mut y = vec![0f32; nb * QK_K];
    for i in 0..nb {
        let blk = &blocks[i * Q5K_BYTES..(i + 1) * Q5K_BYTES];
        let d = fp16_to_fp32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = fp16_to_fp32(u16::from_le_bytes([blk[2], blk[3]]));
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&blk[4..16]);
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let mut is = 0;
        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = get_scale_min_k4(is, &scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
            let d1 = d * sc1 as f32; let m1f = dmin * m1 as f32;
            let d2 = d * sc2 as f32; let m2f = dmin * m2 as f32;
            let q_off = j / 2;
            for l in 0..32 {
                let hbit = ((qh[(j + l) / 8] >> ((j + l) % 8)) & 1) as u8;
                y[i * QK_K + j + l] = d1 * ((qs[q_off + l] & 0xF) | (hbit << 4)) as f32 - m1f;
            }
            for l in 0..32 {
                let hbit = ((qh[(j + 32 + l) / 8] >> ((j + 32 + l) % 8)) & 1) as u8;
                y[i * QK_K + j + 32 + l] = d2 * ((qs[q_off + l] >> 4) | (hbit << 4)) as f32 - m2f;
            }
            is += 2;
        }
    }
    y
}

pub fn cpu_dequant_q2_k(blocks: &[u8]) -> Vec<f32> {
    const Q2K_BPB: usize = 84;
    let nb = blocks.len() / Q2K_BPB;
    let mut y = vec![0f32; nb * QK_K];
    let mut yi = 0usize;
    for i in 0..nb {
        let blk = &blocks[i * Q2K_BPB..(i + 1) * Q2K_BPB];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = fp16_to_fp32(u16::from_le_bytes([blk[80], blk[81]]));
        let dmin = fp16_to_fp32(u16::from_le_bytes([blk[82], blk[83]]));
        let mut is = 0usize;
        let mut q = &qs[..];
        for _ in 0..2 {
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = scales[is]; is += 1;
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                for l in 0..16 { y[yi] = dl * ((q[l] >> shift) & 3) as f32 - ml; yi += 1; }
                let sc2 = scales[is]; is += 1;
                let dl2 = d * (sc2 & 0xF) as f32;
                let ml2 = dmin * (sc2 >> 4) as f32;
                for l in 0..16 { y[yi] = dl2 * ((q[16 + l] >> shift) & 3) as f32 - ml2; yi += 1; }
                shift += 2;
            }
            q = &qs[32..];
        }
    }
    y
}

pub fn cpu_dequant_q3_k(blocks: &[u8]) -> Vec<f32> {
    const Q3K_BPB: usize = 110;
    let nb = blocks.len() / Q3K_BPB;
    let mut y = vec![0f32; nb * QK_K];
    let mut yi = 0usize;
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    for i in 0..nb {
        let blk = &blocks[i * Q3K_BPB..(i + 1) * Q3K_BPB];
        let hm = &blk[0..32];
        let qs = &blk[32..96];
        let raw_scales = &blk[96..108];
        let d_all = fp16_to_fp32(u16::from_le_bytes([blk[108], blk[109]]));
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([raw_scales[0], raw_scales[1], raw_scales[2], raw_scales[3]]);
        aux[1] = u32::from_le_bytes([raw_scales[4], raw_scales[5], raw_scales[6], raw_scales[7]]);
        aux[2] = u32::from_le_bytes([raw_scales[8], raw_scales[9], raw_scales[10], raw_scales[11]]);
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        let scales_bytes: [u8; 16] = unsafe { std::mem::transmute(aux) };
        let scales: &[i8; 16] = unsafe { &*(scales_bytes.as_ptr() as *const [i8; 16]) };
        let mut is = 0usize;
        let mut m: u8 = 1;
        let mut q_off = 0usize;
        for _ in 0..2 {
            let mut shift = 0u32;
            for _ in 0..4 {
                let dl = d_all * (scales[is] - 32) as f32; is += 1;
                for l in 0..16 {
                    let q2 = (qs[q_off + l] >> shift) & 3;
                    let h = if (hm[l] & m) != 0 { 0i8 } else { -4 };
                    y[yi] = dl * (q2 as i8 + h) as f32; yi += 1;
                }
                let dl2 = d_all * (scales[is] - 32) as f32; is += 1;
                for l in 0..16 {
                    let q2 = (qs[q_off + 16 + l] >> shift) & 3;
                    let h = if (hm[16 + l] & m) != 0 { 0i8 } else { -4 };
                    y[yi] = dl2 * (q2 as i8 + h) as f32; yi += 1;
                }
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
    }
    y
}

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
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Re-quantize fp32 weights to Q4_K format.
/// Simple min-max quantization per 32-element sub-block.
fn requant_to_q4k(fp32: &[f32]) -> Vec<u8> {
    let n_blocks = fp32.len() / QK_K;
    let mut out = vec![0u8; n_blocks * Q4K_BYTES];
    for bi in 0..n_blocks {
        let blk = &fp32[bi * QK_K..(bi + 1) * QK_K];
        let dst = &mut out[bi * Q4K_BYTES..(bi + 1) * Q4K_BYTES];

        // Find global d and dmin across all sub-blocks
        let mut max_scale = 0f32;
        let mut max_min = 0f32;
        let mut sub_d = [0f32; 8];
        let mut sub_m = [0f32; 8];

        for j in 0..8 {
            let sub = &blk[j * 32..(j + 1) * 32];
            let mn = sub.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            sub_d[j] = (mx - mn) / 15.0;
            sub_m[j] = -mn;
            if sub_d[j] > max_scale { max_scale = sub_d[j]; }
            if sub_m[j] > max_min { max_min = sub_m[j]; }
        }

        let d = max_scale / 63.0;
        let dmin = max_min / 63.0;

        // Encode d, dmin as fp16
        dst[0..2].copy_from_slice(&f32_to_fp16(d).to_le_bytes());
        dst[2..4].copy_from_slice(&f32_to_fp16(dmin).to_le_bytes());

        // Encode scales[12]
        let mut scales = [0u8; 12];
        for j in 0..8 {
            let sc = if d > 0.0 { (sub_d[j] / d + 0.5) as u8 } else { 0 };
            let mn = if dmin > 0.0 { (sub_m[j] / dmin + 0.5) as u8 } else { 0 };
            let sc = sc.min(63);
            let mn = mn.min(63);
            if j < 4 {
                scales[j] = (scales[j] & !63) | sc;
                scales[j + 4] = (scales[j + 4] & !63) | mn;
            } else {
                // Pack high bits
                scales[j + 4] = (scales[j + 4] & !0x0F) | (sc & 0x0F);
                scales[j + 4] = (scales[j + 4] & !0xF0) | ((mn & 0x0F) << 4);
                scales[j - 4] = (scales[j - 4] & !0xC0) | ((sc >> 4) << 6);
                scales[j] = (scales[j] & !0xC0) | ((mn >> 4) << 6);
            }
        }
        dst[4..16].copy_from_slice(&scales);

        // Encode qs[128]: 4-bit quantized values
        for j in 0..4 {
            let sc_lo = if j < 4 { scales[2*j] & 63 } else {
                let sj4 = scales[2*j + 4]; let sjm4 = scales[2*j - 4];
                (sj4 & 0xF) | ((sjm4 >> 6) << 4)
            };
            let mn_lo = if j < 4 { scales[2*j + 4] & 63 } else {
                let sj4 = scales[2*j + 4]; let sj = scales[2*j];
                (sj4 >> 4) | ((sj >> 6) << 4)
            };
            let sc_hi_idx = 2 * j + 1;
            let sc_hi = if sc_hi_idx < 4 { scales[sc_hi_idx] & 63 } else {
                let sj4 = scales[sc_hi_idx + 4]; let sjm4 = scales[sc_hi_idx - 4];
                (sj4 & 0xF) | ((sjm4 >> 6) << 4)
            };
            let mn_hi = if sc_hi_idx < 4 { scales[sc_hi_idx + 4] & 63 } else {
                let sj4 = scales[sc_hi_idx + 4]; let sj = scales[sc_hi_idx];
                (sj4 >> 4) | ((sj >> 6) << 4)
            };

            let d1 = d * sc_lo as f32;
            let m1 = dmin * mn_lo as f32;
            let d2 = d * sc_hi as f32;
            let m2 = dmin * mn_hi as f32;

            for l in 0..32 {
                let v1 = blk[j * 64 + l];
                let q_lo = if d1 > 0.0 { ((v1 + m1) / d1 + 0.5) as u8 } else { 0 };
                let v2 = blk[j * 64 + 32 + l];
                let q_hi = if d2 > 0.0 { ((v2 + m2) / d2 + 0.5) as u8 } else { 0 };
                dst[16 + j * 32 + l] = (q_lo.min(15)) | ((q_hi.min(15)) << 4);
            }
        }
    }
    out
}

fn f32_to_fp16(v: f32) -> u16 {
    let bits = v.to_bits();
    let s = (bits >> 31) & 1;
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let m = bits & 0x7FFFFF;
    if e > 15 { return ((s as u16) << 15) | 0x7C00; } // inf
    if e < -14 {
        let shift = -14 - e;
        if shift > 10 { return (s as u16) << 15; } // zero
        let m2 = (0x400 | (m >> 13)) >> shift as u32;
        return ((s as u16) << 15) | m2 as u16;
    }
    ((s as u16) << 15) | (((e + 15) as u16) << 10) | ((m >> 13) as u16)
}
