//! 235B end-to-end inference: streaming attention (GPU) + CPU MoE (Q2K/Q3K).
//!
//! Usage:
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release -p ssd_inference --example first_token_235b -- \
//!       path/to/Qwen3-235B-...-00001-of-00002.gguf "The capital of France is" 4

use anyhow::{anyhow, ensure, Result};
use ash::vk;
use gguf_reader::{ExpertKind, MultiGgufFile};
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

use ssd_inference::buffer::GpuBuffer;
use ssd_inference::compute::{ComputePipeline, DescriptorArena};
use ssd_inference::device::VulkanContext;
use ssd_inference::expert_reader::ExpertReader;
use ssd_inference::kv_cache::KvCache;
use ssd_inference::model::ModelConfig;
use ssd_inference::pipelines::Pipelines;
use ssd_inference::streaming_weights::{
    cpu_dequant_q2_k, cpu_dequant_q3_k, cpu_dequant_q5_k, StreamingWeights,
};
use ssd_inference::tokenizer::Tok;
use ssd_inference::weights::{cpu_dequant_q4_k, cpu_dequant_q6_k};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: first_token_235b <gguf-shard-1> [prompt] [max_tokens]"))?;
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "The capital of France is".into());
    let max_tokens: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    println!("Opening {}", path);
    let mg = MultiGgufFile::open(&path)?;
    let cfg = ModelConfig::from_multi_gguf(&mg)?;
    let tok = Tok::from_multi_gguf(&mg)?;
    let token_ids = tok.encode(&prompt)?;
    println!(
        "Model: {} | layers={} d={} experts={} top_k={}",
        cfg.arch, cfg.n_layer, cfg.d_model, cfg.n_experts, cfg.top_k
    );
    println!("Prompt tokens ({}): {:?}\n", token_ids.len(), token_ids);

    let ctx = VulkanContext::init()?;
    println!(
        "GPU: {} ({:.1} GB)",
        ctx.gpu_name,
        ctx.vram_size() as f64 / 1e9
    );

    let t0 = Instant::now();
    let mut sw = StreamingWeights::new(&ctx, &mg, &cfg)?;
    let reader = ExpertReader::from_multi_gguf(&mg, cfg.n_experts)?;
    let pipes = Pipelines::build(&ctx)?;
    let kv = KvCache::new(&ctx, &cfg, 128)?;

    // VramPool: cache expert Q bytes in VRAM
    // Scan for actual max expert size to optimize slot usage
    let mut max_expert_bytes = 0u64;
    for l in 0..cfg.n_layer {
        for kind in [
            ExpertKind::GateExps,
            ExpertKind::UpExps,
            ExpertKind::DownExps,
        ] {
            if let Some(sz) = reader.expert_size(l, kind, 0) {
                max_expert_bytes = max_expert_bytes.max(sz as u64);
            }
        }
    }
    ensure!(
        max_expert_bytes > 0,
        "model contains no indexed expert pages; refusing to create a zero-sized staging ring"
    );
    // Budget: VRAM total - attn_q4k(41MB) - KV(100MB) - scratch(100MB) - expert_fp32(24MB)
    let pool_budget_mb = 5000u64; // ~5 GB for pool (freed by fused Q4K attention)
    let pool_slots = (pool_budget_mb * 1024 * 1024 / max_expert_bytes).min(4096) as u32;
    let mut vram_pool =
        ssd_inference::vram_pool::VramPool::new(&ctx, pool_slots, max_expert_bytes)?;
    let mut expert_loader = ssd_inference::expert_loader::ExpertLoader::new_with_staging_bytes(
        &ctx,
        12,
        cfg.n_experts,
        max_expert_bytes,
    )?;

    println!(
        "Init: {:.2}s (pool: {} slots × {:.1} MB = {:.1} GB, max_exp={:.1} MB)\n",
        t0.elapsed().as_secs_f64(),
        pool_slots,
        max_expert_bytes as f64 / 1e6,
        pool_slots as f64 * max_expert_bytes as f64 / 1e9,
        max_expert_bytes as f64 / 1e6
    );

    let d = cfg.d_model as usize;

    // Scratch buffers
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;
    let nq = cfg.n_q_heads as u64;
    let nkv = cfg.n_kv_heads as u64;
    let hd = cfg.head_dim as u64;
    let d64 = d as u64;
    let inter = cfg.moe_intermediate as usize;
    let inter64 = inter as u64;

    let h_a = GpuBuffer::new_vram(&ctx, d64 * 4, usage)?;
    let h_b = GpuBuffer::new_vram(&ctx, d64 * 4, usage)?;
    let x = GpuBuffer::new_vram(&ctx, d64 * 4, usage)?;
    let q = GpuBuffer::new_vram(&ctx, nq * hd * 4, usage)?;
    let k_new = GpuBuffer::new_vram(&ctx, nkv * hd * 4, usage)?;
    let v_new = GpuBuffer::new_vram(&ctx, nkv * hd * 4, usage)?;
    let scores = GpuBuffer::new_vram(&ctx, nq * kv.max_seq as u64 * 4, usage)?;
    let probs = GpuBuffer::new_vram(&ctx, nq * kv.max_seq as u64 * 4, usage)?;
    let attn_out = GpuBuffer::new_vram(&ctx, nq * hd * 4, usage)?;
    let attn_proj = GpuBuffer::new_vram(&ctx, d64 * 4, usage)?;
    let router_logits = GpuBuffer::new_vram(&ctx, cfg.n_experts as u64 * 4, usage)?;

    // MoE GPU buffers: dequant from VRAM pool → fp32 → matvec
    let expert_fp32_buf = GpuBuffer::new_vram(&ctx, d64 * inter64 * 4, usage)?;
    let gate_out = GpuBuffer::new_vram(&ctx, inter64 * 4, usage)?;
    let up_out = GpuBuffer::new_vram(&ctx, inter64 * 4, usage)?;
    let swiglu_out = GpuBuffer::new_vram(&ctx, inter64 * 4, usage)?;
    let down_out = GpuBuffer::new_vram(&ctx, d64 * 4, usage)?;

    let readback = GpuBuffer::new(
        &ctx,
        std::cmp::max(d64 * 4, cfg.n_experts as u64 * 4),
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;
    let upload = GpuBuffer::new(
        &ctx,
        d64 * 4,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;

    let mut arena = DescriptorArena::new(&ctx, 2500, 5)?;

    let cmd_pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?
    };
    let cb = unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0]
    };
    let fence = unsafe {
        ctx.device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
            None,
        )?
    };

    // Helper: record + sync submit
    let record_sync = |ctx: &VulkanContext,
                       cb: vk::CommandBuffer,
                       fence: vk::Fence,
                       arena: &mut DescriptorArena,
                       f: &dyn Fn(vk::CommandBuffer, &mut DescriptorArena) -> Result<()>|
     -> Result<()> {
        unsafe {
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            arena.reset(ctx)?;
            ctx.device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            ctx.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        f(cb, arena)?;
        unsafe {
            ctx.device.end_command_buffer(cb)?;
            ctx.device.reset_fences(&[fence])?;
            let cb_arr = [cb];
            ctx.device.queue_submit(
                ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&cb_arr)],
                fence,
            )?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        Ok(())
    };

    let barrier = |ctx: &VulkanContext, cb: vk::CommandBuffer| unsafe {
        let bar = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
        ctx.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[bar],
            &[],
            &[],
        );
    };

    let mut pos = 0u32;
    let mut h_a_ref = &h_a;
    let mut h_b_ref = &h_b;

    // ── Main loop: prefill + decode ──
    let mut generated: Vec<u32> = Vec::new();
    let total_steps = token_ids.len() + max_tokens;
    let t_start = Instant::now();

    for ti in 0..total_steps {
        // Pick the token to feed: prompt token or previously generated
        let token_id = if ti < token_ids.len() {
            token_ids[ti]
        } else {
            *generated.last().unwrap()
        };

        // Embed
        let row = sw.embed_row(token_id);
        unsafe {
            let dst = std::slice::from_raw_parts_mut(upload.mapped() as *mut f32, d);
            dst.copy_from_slice(row);
        }
        record_sync(&ctx, cb, fence, &mut arena, &|cb, _| {
            unsafe {
                ctx.device.cmd_copy_buffer(
                    cb,
                    upload.handle(),
                    h_a_ref.handle(),
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: d64 * 4,
                    }],
                );
            }
            Ok(())
        })?;

        // Forward through all layers
        for l in 0..cfg.n_layer {
            sw.upload_layer(&ctx, l)?;

            let cur_seq = pos + 1;
            let h_bytes = d64 * 4;
            let q_bytes = nq * hd * 4;
            let kv_bytes = nkv * hd * 4;
            let scores_bytes = nq * kv.max_seq as u64 * 4;
            let attn_out_bytes = nq * hd * 4;
            let kv_layer_size = kv.max_seq as u64 * nkv * hd * 4;
            let router_bytes = cfg.n_experts as u64 * 4;
            let kv_write_off = pos as u64 * nkv * hd * 4;
            let lb = &sw.layer_bufs;
            let lr = &sw.layers_ram[l as usize];

            record_sync(&ctx, cb, fence, &mut arena, &|cb, ar| {
                dispatch(
                    &ctx,
                    &pipes.rmsnorm,
                    ar,
                    cb,
                    &[
                        (h_a_ref, 0, h_bytes),
                        (&lb.attn_norm, 0, d64 * 4),
                        (&x, 0, h_bytes),
                    ],
                    &pack_u32_f32(d64 as u32, cfg.rms_eps),
                    1,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);

                if lr.is_q4k {
                    let aq_b = lr.attn_q.len() as u64;
                    let ak_b = lr.attn_k.len() as u64;
                    let av_b = lr.attn_v.len() as u64;
                    dispatch(
                        &ctx,
                        &pipes.fused_q4k,
                        ar,
                        cb,
                        &[(&lb.attn_q, 0, aq_b), (&x, 0, h_bytes), (&q, 0, q_bytes)],
                        &pack_u32_3((nq * hd) as u32, d64 as u32, 0),
                        (nq * hd) as u32,
                        1,
                        1,
                    )?;
                    dispatch(
                        &ctx,
                        &pipes.fused_q4k,
                        ar,
                        cb,
                        &[
                            (&lb.attn_k, 0, ak_b),
                            (&x, 0, h_bytes),
                            (&k_new, 0, kv_bytes),
                        ],
                        &pack_u32_3((nkv * hd) as u32, d64 as u32, 0),
                        (nkv * hd) as u32,
                        1,
                        1,
                    )?;
                    dispatch(
                        &ctx,
                        &pipes.fused_q4k,
                        ar,
                        cb,
                        &[
                            (&lb.attn_v, 0, av_b),
                            (&x, 0, h_bytes),
                            (&v_new, 0, kv_bytes),
                        ],
                        &pack_u32_3((nkv * hd) as u32, d64 as u32, 0),
                        (nkv * hd) as u32,
                        1,
                        1,
                    )?;
                } else {
                    dispatch(
                        &ctx,
                        &pipes.matvec,
                        ar,
                        cb,
                        &[
                            (&x, 0, h_bytes),
                            (&lb.attn_q_fp32, 0, nq * hd * d64 * 4),
                            (&q, 0, q_bytes),
                        ],
                        &pack_u32_2((nq * hd) as u32, d64 as u32),
                        (nq * hd) as u32,
                        1,
                        1,
                    )?;
                    dispatch(
                        &ctx,
                        &pipes.matvec,
                        ar,
                        cb,
                        &[
                            (&x, 0, h_bytes),
                            (&lb.attn_k_fp32, 0, nkv * hd * d64 * 4),
                            (&k_new, 0, kv_bytes),
                        ],
                        &pack_u32_2((nkv * hd) as u32, d64 as u32),
                        (nkv * hd) as u32,
                        1,
                        1,
                    )?;
                    dispatch(
                        &ctx,
                        &pipes.matvec,
                        ar,
                        cb,
                        &[
                            (&x, 0, h_bytes),
                            (&lb.attn_v_fp32, 0, nkv * hd * d64 * 4),
                            (&v_new, 0, kv_bytes),
                        ],
                        &pack_u32_2((nkv * hd) as u32, d64 as u32),
                        (nkv * hd) as u32,
                        1,
                        1,
                    )?;
                }
                barrier(&ctx, cb);

                dispatch(
                    &ctx,
                    &pipes.rmsnorm,
                    ar,
                    cb,
                    &[
                        (&q, 0, q_bytes),
                        (&lb.attn_q_norm, 0, hd * 4),
                        (&q, 0, q_bytes),
                    ],
                    &pack_u32_f32(hd as u32, cfg.rms_eps),
                    cfg.n_q_heads,
                    1,
                    1,
                )?;
                dispatch(
                    &ctx,
                    &pipes.rmsnorm,
                    ar,
                    cb,
                    &[
                        (&k_new, 0, kv_bytes),
                        (&lb.attn_k_norm, 0, hd * 4),
                        (&k_new, 0, kv_bytes),
                    ],
                    &pack_u32_f32(hd as u32, cfg.rms_eps),
                    cfg.n_kv_heads,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);

                dispatch(
                    &ctx,
                    &pipes.rope,
                    ar,
                    cb,
                    &[(&q, 0, q_bytes)],
                    &pack_rope(cfg.n_q_heads, cfg.head_dim, pos, cfg.rope_theta),
                    cfg.n_q_heads,
                    1,
                    1,
                )?;
                dispatch(
                    &ctx,
                    &pipes.rope,
                    ar,
                    cb,
                    &[(&k_new, 0, kv_bytes)],
                    &pack_rope(cfg.n_kv_heads, cfg.head_dim, pos, cfg.rope_theta),
                    cfg.n_kv_heads,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);

                let k_dst = kv.k_offset(l) + kv_write_off;
                let v_dst = kv.v_offset(l) + kv_write_off;
                unsafe {
                    ctx.device.cmd_copy_buffer(
                        cb,
                        k_new.handle(),
                        kv.buf.handle(),
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: k_dst,
                            size: kv_bytes,
                        }],
                    );
                    ctx.device.cmd_copy_buffer(
                        cb,
                        v_new.handle(),
                        kv.buf.handle(),
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: v_dst,
                            size: kv_bytes,
                        }],
                    );
                }
                barrier(&ctx, cb);

                let scale = 1.0 / (cfg.head_dim as f32).sqrt();
                dispatch(
                    &ctx,
                    &pipes.scaled_dot,
                    ar,
                    cb,
                    &[
                        (&q, 0, q_bytes),
                        (&kv.buf, kv.k_offset(l), kv_layer_size),
                        (&scores, 0, scores_bytes),
                    ],
                    &pack_scaled_dot(
                        1,
                        cfg.n_q_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        cur_seq,
                        pos,
                        scale,
                    ),
                    cfg.n_q_heads,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);
                dispatch(
                    &ctx,
                    &pipes.softmax,
                    ar,
                    cb,
                    &[(&scores, 0, scores_bytes), (&probs, 0, scores_bytes)],
                    &cur_seq.to_le_bytes().to_vec(),
                    cfg.n_q_heads,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);
                dispatch(
                    &ctx,
                    &pipes.attn_v,
                    ar,
                    cb,
                    &[
                        (&probs, 0, scores_bytes),
                        (&kv.buf, kv.v_offset(l), kv_layer_size),
                        (&attn_out, 0, attn_out_bytes),
                    ],
                    &pack_attnv(1, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, cur_seq),
                    cfg.n_q_heads,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);

                if lr.is_q4k {
                    let ao_b = lr.attn_o.len() as u64;
                    dispatch(
                        &ctx,
                        &pipes.fused_q4k,
                        ar,
                        cb,
                        &[
                            (&lb.attn_o, 0, ao_b),
                            (&attn_out, 0, attn_out_bytes),
                            (&attn_proj, 0, h_bytes),
                        ],
                        &pack_u32_3(d64 as u32, (nq * hd) as u32, 0),
                        d64 as u32,
                        1,
                        1,
                    )?;
                } else {
                    dispatch(
                        &ctx,
                        &pipes.matvec,
                        ar,
                        cb,
                        &[
                            (&attn_out, 0, attn_out_bytes),
                            (&lb.attn_o_fp32, 0, d64 * (nq * hd) * 4),
                            (&attn_proj, 0, h_bytes),
                        ],
                        &pack_u32_2(d64 as u32, (nq * hd) as u32),
                        d64 as u32,
                        1,
                        1,
                    )?;
                }
                barrier(&ctx, cb);

                dispatch(
                    &ctx,
                    &pipes.residual,
                    ar,
                    cb,
                    &[
                        (h_a_ref, 0, h_bytes),
                        (&attn_proj, 0, h_bytes),
                        (h_b_ref, 0, h_bytes),
                    ],
                    &(d64 as u32).to_le_bytes().to_vec(),
                    ((d64 + 255) / 256) as u32,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);

                dispatch(
                    &ctx,
                    &pipes.rmsnorm,
                    ar,
                    cb,
                    &[
                        (h_b_ref, 0, h_bytes),
                        (&lb.ffn_norm, 0, d64 * 4),
                        (&x, 0, h_bytes),
                    ],
                    &pack_u32_f32(d64 as u32, cfg.rms_eps),
                    1,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);
                dispatch(
                    &ctx,
                    &pipes.matvec,
                    ar,
                    cb,
                    &[
                        (&x, 0, h_bytes),
                        (&lb.router, 0, cfg.n_experts as u64 * d64 * 4),
                        (&router_logits, 0, router_bytes),
                    ],
                    &pack_u32_2(cfg.n_experts, d64 as u32),
                    cfg.n_experts,
                    1,
                    1,
                )?;
                barrier(&ctx, cb);

                unsafe {
                    ctx.device.cmd_copy_buffer(
                        cb,
                        router_logits.handle(),
                        readback.handle(),
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: router_bytes,
                        }],
                    );
                }
                Ok(())
            })?;

            // MoE
            let n_exp = cfg.n_experts as usize;
            let mut router_vals = vec![0f32; n_exp];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    readback.mapped() as *const f32,
                    router_vals.as_mut_ptr(),
                    n_exp,
                );
            }
            let picks = host_pick_topk(&router_vals, n_exp, cfg.top_k as usize);

            // Collect which experts need loading
            let mut to_load: Vec<(ExpertKind, u32)> = Vec::new();
            for &(eidx, _) in &picks {
                for kind in [
                    ExpertKind::GateExps,
                    ExpertKind::UpExps,
                    ExpertKind::DownExps,
                ] {
                    let key = ssd_inference::vram_pool::ExpertKey {
                        layer: l,
                        kind: kind as u8,
                        slot: eidx,
                    };
                    if vram_pool.lookup(key).is_none() {
                        to_load.push((kind, eidx));
                    }
                }
            }
            for &(kind, eidx) in &to_load {
                expert_loader.enqueue(&ctx, &reader, &mut vram_pool, l, kind, eidx)?;
            }
            expert_loader.wait_all(&ctx, &mut vram_pool)?;

            for &(eidx, weight) in &picks {
                let nwgu = d * inter;
                let nbgu = nwgu / 256;
                let nwd = inter * d;
                let nbd = nwd / 256;
                let gs = reader.expert_size(l, ExpertKind::GateExps, eidx).unwrap();
                let us = reader.expert_size(l, ExpertKind::UpExps, eidx).unwrap();
                let ds = reader.expert_size(l, ExpertKind::DownExps, eidx).unwrap();
                let gq = detect_qtype(gs, nbgu);
                let uq = detect_qtype(us, nbgu);
                let dq = detect_qtype(ds, nbd);
                let gk = ssd_inference::vram_pool::ExpertKey {
                    layer: l,
                    kind: ExpertKind::GateExps as u8,
                    slot: eidx,
                };
                let uk = ssd_inference::vram_pool::ExpertKey {
                    layer: l,
                    kind: ExpertKind::UpExps as u8,
                    slot: eidx,
                };
                let dk = ssd_inference::vram_pool::ExpertKey {
                    layer: l,
                    kind: ExpertKind::DownExps as u8,
                    slot: eidx,
                };
                let gs_idx = vram_pool.lookup(gk).unwrap();
                let us_idx = vram_pool.lookup(uk).unwrap();
                let ds_idx = vram_pool.lookup(dk).unwrap();
                let go = vram_pool.slot_offset(gs_idx);
                let uo = vram_pool.slot_offset(us_idx);
                let doff = vram_pool.slot_offset(ds_idx);
                let pool_buf = &vram_pool.buffer;

                record_sync(&ctx, cb, fence, &mut arena, &|cb, ar| {
                    gpu_dequant_dispatch_off(
                        &ctx,
                        &pipes,
                        ar,
                        cb,
                        gq,
                        nbgu as u32,
                        pool_buf,
                        go,
                        gs as u64,
                        &expert_fp32_buf,
                        0,
                        nwgu as u64 * 4,
                    )?;
                    barrier(&ctx, cb);
                    dispatch(
                        &ctx,
                        &pipes.matvec,
                        ar,
                        cb,
                        &[
                            (&x, 0, d64 * 4),
                            (&expert_fp32_buf, 0, nwgu as u64 * 4),
                            (&gate_out, 0, inter64 * 4),
                        ],
                        &pack_u32_2(inter as u32, d as u32),
                        inter as u32,
                        1,
                        1,
                    )?;
                    barrier(&ctx, cb);
                    gpu_dequant_dispatch_off(
                        &ctx,
                        &pipes,
                        ar,
                        cb,
                        uq,
                        nbgu as u32,
                        pool_buf,
                        uo,
                        us as u64,
                        &expert_fp32_buf,
                        0,
                        nwgu as u64 * 4,
                    )?;
                    barrier(&ctx, cb);
                    dispatch(
                        &ctx,
                        &pipes.matvec,
                        ar,
                        cb,
                        &[
                            (&x, 0, d64 * 4),
                            (&expert_fp32_buf, 0, nwgu as u64 * 4),
                            (&up_out, 0, inter64 * 4),
                        ],
                        &pack_u32_2(inter as u32, d as u32),
                        inter as u32,
                        1,
                        1,
                    )?;
                    barrier(&ctx, cb);
                    dispatch(
                        &ctx,
                        &pipes.swiglu,
                        ar,
                        cb,
                        &[
                            (&gate_out, 0, inter64 * 4),
                            (&up_out, 0, inter64 * 4),
                            (&swiglu_out, 0, inter64 * 4),
                        ],
                        &(inter as u32).to_le_bytes().to_vec(),
                        ((inter64 + 255) / 256) as u32,
                        1,
                        1,
                    )?;
                    barrier(&ctx, cb);
                    gpu_dequant_dispatch_off(
                        &ctx,
                        &pipes,
                        ar,
                        cb,
                        dq,
                        nbd as u32,
                        pool_buf,
                        doff,
                        ds as u64,
                        &expert_fp32_buf,
                        0,
                        nwd as u64 * 4,
                    )?;
                    barrier(&ctx, cb);
                    dispatch(
                        &ctx,
                        &pipes.matvec,
                        ar,
                        cb,
                        &[
                            (&swiglu_out, 0, inter64 * 4),
                            (&expert_fp32_buf, 0, nwd as u64 * 4),
                            (&down_out, 0, d64 * 4),
                        ],
                        &pack_u32_2(d as u32, inter as u32),
                        d as u32,
                        1,
                        1,
                    )?;
                    barrier(&ctx, cb);
                    dispatch(
                        &ctx,
                        &pipes.weighted,
                        ar,
                        cb,
                        &[(h_b_ref, 0, d64 * 4), (&down_out, 0, d64 * 4)],
                        &pack_u32_f32(d as u32, weight),
                        ((d64 + 255) / 256) as u32,
                        1,
                        1,
                    )?;
                    Ok(())
                })?;
            }

            if l % 10 == 9 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            std::mem::swap(&mut h_a_ref, &mut h_b_ref);
        }

        pos += 1;

        // Sample next token
        record_sync(&ctx, cb, fence, &mut arena, &|cb, ar| {
            dispatch(
                &ctx,
                &pipes.rmsnorm,
                ar,
                cb,
                &[
                    (h_a_ref, 0, d64 * 4),
                    (&sw.out_norm, 0, d64 * 4),
                    (&x, 0, d64 * 4),
                ],
                &pack_u32_f32(d64 as u32, cfg.rms_eps),
                1,
                1,
                1,
            )?;
            Ok(())
        })?;
        let x_final = read_buf_f32(&ctx, cmd_pool, &x, 0, d, &readback)?;

        let lmh = &sw.lm_head_host;
        let logits: Vec<f32> = (0..cfg.vocab as usize)
            .into_par_iter()
            .map(|i| {
                let row = &lmh[i * d..(i + 1) * d];
                let mut acc = 0f32;
                for k in 0..d {
                    acc += x_final[k] * row[k];
                }
                acc
            })
            .collect();

        // Temperature + top-p sampling
        let temperature = 0.7f32;
        let next_id = sample_top_p(&logits, temperature, 0.9) as u32;
        generated.push(next_id);

        let tok_s = tok.id_to_str(next_id).unwrap_or_else(|| "?".into());
        if ti == token_ids.len() - 1 {
            eprintln!();
            print!("{}", prompt);
        }
        if ti >= token_ids.len() - 1 {
            print!("{}", tok_s.replace('Ġ', " "));
            std::io::stdout().flush()?;
        } else {
            eprint!(".");
        }

        if next_id == cfg.eos_token_id {
            break;
        }
    }
    println!();
    let total = t_start.elapsed().as_secs_f64();
    let n_gen = generated.len();
    println!(
        "Generated {} tokens in {:.1}s ({:.2} t/s)",
        n_gen,
        total,
        n_gen as f64 / total
    );

    // Cleanup
    arena.destroy(&ctx);
    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(cmd_pool, None);
    }
    for b in [
        &h_a,
        &h_b,
        &x,
        &q,
        &k_new,
        &v_new,
        &scores,
        &probs,
        &attn_out,
        &attn_proj,
        &router_logits,
        &readback,
        &upload,
        &expert_fp32_buf,
        &gate_out,
        &up_out,
        &swiglu_out,
        &down_out,
    ] {
        b.destroy(&ctx);
    }
    kv.destroy(&ctx);
    expert_loader.destroy(&ctx);
    vram_pool.destroy(&ctx);
    pipes.destroy(&ctx);
    sw.destroy(&ctx);

    Ok(())
}

// ── Helpers ──

fn dispatch(
    ctx: &VulkanContext,
    pipe: &ComputePipeline,
    arena: &mut DescriptorArena,
    cb: vk::CommandBuffer,
    bufs: &[(&GpuBuffer, u64, u64)],
    push: &[u8],
    dx: u32,
    dy: u32,
    dz: u32,
) -> Result<()> {
    let set = arena.alloc_set(ctx, pipe, bufs)?;
    unsafe {
        ctx.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
        ctx.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipe.layout,
            0,
            &[set],
            &[],
        );
        if !push.is_empty() {
            ctx.device
                .cmd_push_constants(cb, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
        }
        ctx.device.cmd_dispatch(cb, dx, dy, dz);
    }
    Ok(())
}

fn read_buf_f32(
    ctx: &VulkanContext,
    cmd_pool: vk::CommandPool,
    buf: &GpuBuffer,
    offset: u64,
    n: usize,
    readback: &GpuBuffer,
) -> Result<Vec<f32>> {
    let bytes = (n * 4) as u64;
    unsafe {
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        ctx.device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device.cmd_copy_buffer(
            cb,
            buf.handle(),
            readback.handle(),
            &[vk::BufferCopy {
                src_offset: offset,
                dst_offset: 0,
                size: bytes,
            }],
        );
        ctx.device.end_command_buffer(cb)?;
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(cmd_pool, &[cb]);
    }
    let mut out = vec![0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(readback.mapped() as *const f32, out.as_mut_ptr(), n);
    }
    Ok(out)
}

fn host_pick_topk(router: &[f32], n_exp: usize, k: usize) -> Vec<(u32, f32)> {
    let max = router.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exps: Vec<f32> = router.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in exps.iter_mut() {
        *e /= sum;
    }
    let mut idx: Vec<usize> = (0..n_exp).collect();
    idx.sort_unstable_by(|&a, &b| {
        exps[b]
            .partial_cmp(&exps[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut picks: Vec<(u32, f32)> = idx[..k].iter().map(|&i| (i as u32, exps[i])).collect();
    let s: f32 = picks.iter().map(|p| p.1).sum();
    for p in picks.iter_mut() {
        p.1 /= s;
    }
    picks
}

fn dequant_expert(bytes: &[u8], byte_size: usize, n_weights: usize) -> Vec<f32> {
    let q2k_bpb = 84;
    let q3k_bpb = 110;
    let q4k_bpb = 144;
    let q6k_bpb = 210;
    let n_blocks = n_weights / 256;
    if byte_size == n_blocks * q2k_bpb {
        return cpu_dequant_q2_k(bytes);
    }
    if byte_size == n_blocks * q3k_bpb {
        return cpu_dequant_q3_k(bytes);
    }
    if byte_size == n_blocks * q4k_bpb {
        return cpu_dequant_q4_k(bytes);
    }
    if byte_size == n_blocks * q6k_bpb {
        return cpu_dequant_q6_k(bytes);
    }
    panic!(
        "unknown expert quant: {} bytes for {} weights",
        byte_size, n_weights
    );
}

#[derive(Clone, Copy, PartialEq)]
enum Qtype {
    Q2K,
    Q3K,
    Q4K,
    Q6K,
}

fn detect_qtype(byte_size: usize, n_blocks: usize) -> Qtype {
    if byte_size == n_blocks * 84 {
        Qtype::Q2K
    } else if byte_size == n_blocks * 110 {
        Qtype::Q3K
    } else if byte_size == n_blocks * 144 {
        Qtype::Q4K
    } else if byte_size == n_blocks * 210 {
        Qtype::Q6K
    } else {
        panic!("unknown quant: {} bytes for {} blocks", byte_size, n_blocks)
    }
}

fn gpu_dequant_dispatch(
    ctx: &VulkanContext,
    pipes: &Pipelines,
    arena: &mut DescriptorArena,
    cb: vk::CommandBuffer,
    qt: Qtype,
    n_blocks: u32,
    src: &GpuBuffer,
    src_bytes: u64,
    dst: &GpuBuffer,
    dst_bytes: u64,
) -> Result<()> {
    gpu_dequant_dispatch_off(
        ctx, pipes, arena, cb, qt, n_blocks, src, 0, src_bytes, dst, 0, dst_bytes,
    )
}

fn gpu_dequant_dispatch_off(
    ctx: &VulkanContext,
    pipes: &Pipelines,
    arena: &mut DescriptorArena,
    cb: vk::CommandBuffer,
    qt: Qtype,
    n_blocks: u32,
    src: &GpuBuffer,
    src_off: u64,
    src_bytes: u64,
    dst: &GpuBuffer,
    dst_off: u64,
    dst_bytes: u64,
) -> Result<()> {
    let (pipe, push) = match qt {
        Qtype::Q2K => (&pipes.dq_q2k, n_blocks.to_le_bytes().to_vec()),
        Qtype::Q3K => {
            let mut p = Vec::with_capacity(8);
            p.extend_from_slice(&n_blocks.to_le_bytes());
            p.extend_from_slice(&28u32.to_le_bytes());
            (&pipes.dq_q3k, p)
        }
        Qtype::Q4K => (&pipes.dq_q4k, n_blocks.to_le_bytes().to_vec()),
        Qtype::Q6K => (&pipes.dq_q6k, n_blocks.to_le_bytes().to_vec()),
    };
    dispatch(
        ctx,
        pipe,
        arena,
        cb,
        &[(src, src_off, src_bytes), (dst, dst_off, dst_bytes)],
        &push,
        n_blocks,
        1,
        1,
    )
}

fn cpu_matvec(w: &[f32], x: &[f32], n_out: usize, n_in: usize) -> Vec<f32> {
    let mut y = vec![0f32; n_out];
    for n in 0..n_out {
        let row = &w[n * n_in..(n + 1) * n_in];
        let mut acc = 0f32;
        for k in 0..n_in {
            acc += row[k] * x[k];
        }
        y[n] = acc;
    }
    y
}

fn pack_u32_2(a: u32, b: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&a.to_le_bytes());
    v.extend_from_slice(&b.to_le_bytes());
    v
}

fn pack_u32_3(a: u32, b: u32, c: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(12);
    v.extend_from_slice(&a.to_le_bytes());
    v.extend_from_slice(&b.to_le_bytes());
    v.extend_from_slice(&c.to_le_bytes());
    v
}

fn pack_u32_f32(a: u32, b: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&a.to_le_bytes());
    v.extend_from_slice(&b.to_le_bytes());
    v
}

fn pack_rope(n_heads: u32, head_dim: u32, base_pos: u32, theta: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&n_heads.to_le_bytes());
    v.extend_from_slice(&head_dim.to_le_bytes());
    v.extend_from_slice(&base_pos.to_le_bytes());
    v.extend_from_slice(&theta.to_le_bytes());
    v
}

fn pack_scaled_dot(
    n_tok: u32,
    n_q: u32,
    n_kv: u32,
    hd: u32,
    seq_len: u32,
    base_pos: u32,
    scale: f32,
) -> Vec<u8> {
    let mut v = Vec::with_capacity(28);
    for x in [n_tok, n_q, n_kv, hd, seq_len, base_pos] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v.extend_from_slice(&scale.to_le_bytes());
    v
}

fn pack_attnv(n_tok: u32, n_q: u32, n_kv: u32, hd: u32, seq_len: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(20);
    for x in [n_tok, n_q, n_kv, hd, seq_len] {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    use rand::Rng;
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / temperature))
        .collect();
    let max = scaled.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = scaled.iter().map(|&(i, v)| (i, (v - max).exp())).collect();
    let sum: f32 = probs.iter().map(|x| x.1).sum();
    for p in probs.iter_mut() {
        p.1 /= sum;
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumsum = 0f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    let kept = &probs[..cutoff];
    let kept_sum: f32 = kept.iter().map(|x| x.1).sum();

    let _ = rand::random::<f32>(); // ensure rand compiles
    let r: f32 = rand::random::<f32>() * kept_sum;
    let mut acc = 0f32;
    for &(idx, p) in kept {
        acc += p;
        if acc >= r {
            return idx;
        }
    }
    kept.last().unwrap().0
}
