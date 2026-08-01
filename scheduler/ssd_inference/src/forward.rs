//! Transformer forward pass scheduler.
//!
//! Drives one prefill of N tokens through all layers of Qwen3-MoE and returns
//! the next-token prediction (greedy argmax of last token's logits).
//!
//! Per layer:
//!   1. ATTN cmd buffer:
//!      embedding lookup (layer 0 only) → rmsnorm(attn_norm) → Q/K/V matmul
//!      → q_norm/k_norm (per-head rmsnorm) → RoPE → KV cache append
//!      → scaled_dot → softmax → attn_v → o_proj → residual
//!      → ffn_norm → router matmul
//!   2. Host: read router scores, softmax, top-k, pick expert ids per token
//!   3. MoE cmd buffer: for each (token, picked_expert):
//!      ensure expert in VRAM → dequant gate/up/down → matmul × matmul → swiglu
//!      → matmul → weighted_add into hidden
//!
//! After all 48 layers: out_norm → lm_head → host argmax.

use crate::ggml_bridge;
use anyhow::{anyhow, Result};
use ash::vk;
use gguf_reader::{ExpertKind, GgufFile};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use std::sync::mpsc;

use predictor::{ActivationRecord, MatrixBuilder};
use scheduler_core::{Scheduler, SchedulerConfig, SchedulerCommand};
use expert_cache::ExpertCache;

use crate::buffer::GpuBuffer;
use crate::compute::{ComputePipeline, DescriptorBinder, DescriptorArena};
use crate::device::VulkanContext;
use crate::engine::Engine;
use crate::kv_cache::KvCache;
use crate::model::ModelConfig;
use crate::pipelines::Pipelines;
use crate::vram_pool::ExpertKey;
use crate::weights::LoadedWeights;

const Q4K_BYTES: u64 = 144;
const Q6K_BYTES: u64 = 210;
const QK_K: u64 = 256;

struct GpuExpertInfo {
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    gate_bytes: u64,
    up_bytes: u64,
    down_bytes: u64,
    gate_q6: bool,
    down_q6: bool,
    weight: f32,
}

pub struct Forward<'a> {
    pub ctx: &'a VulkanContext,
    pub engine: &'a mut Engine,
    pub weights: &'a LoadedWeights,
    pub kv: &'a KvCache,
    pub cfg: &'a ModelConfig,
    pub pipes: &'a Pipelines,
    cmd_pool: vk::CommandPool,
    pub pos: u32,
    /// Last token's expert picks per layer: [layer][expert_idx]
    /// Used to predict next token's experts for background prefetch
    prev_picks: Vec<Vec<u32>>,
    // Predictor: online co-occurrence matrix training + prefetch scheduler
    predictor_builder: MatrixBuilder,
    scheduler: Scheduler,
    token_counter: u32,
    predict_hits: u64,
    predict_total: u64,
    // Scratch buffers (n_tok=1)
    h_a: GpuBuffer,
    h_b: GpuBuffer,
    x: GpuBuffer,
    q: GpuBuffer,
    k_new: GpuBuffer,
    v_new: GpuBuffer,
    scores: GpuBuffer,
    probs: GpuBuffer,
    attn_out: GpuBuffer,
    attn_proj: GpuBuffer,
    router_logits: GpuBuffer,
    // MoE scratch — holds ALL 8 experts simultaneously for batched dispatch
    expert_gate_dq: GpuBuffer,
    expert_up_dq: GpuBuffer,
    expert_down_dq: GpuBuffer,
    g_all: GpuBuffer,   // 8 × inter fp32
    u_all: GpuBuffer,   // 8 × inter fp32
    s_all: GpuBuffer,   // 8 × inter fp32
    d_all: GpuBuffer,   // 8 × d_model fp32
    logits: GpuBuffer,
    readback: GpuBuffer,
    rb_router_off: u64,
    upload: GpuBuffer,
    /// HOST_VISIBLE STORAGE for expert Q bytes — GPU reads via PCIe
    /// Persistent fence for async GPU‖CPU pipeline
    pipe_fence: vk::Fence,
    pipe_cb: vk::CommandBuffer,
    pipe_pending: bool,
    expert_staging_stride: u64,
    arena: DescriptorArena,
}

impl<'a> Forward<'a> {
    pub fn new(
        ctx: &'a VulkanContext,
        engine: &'a mut Engine,
        weights: &'a LoadedWeights,
        kv: &'a KvCache,
        cfg: &'a ModelConfig,
        pipes: &'a Pipelines,
    ) -> Result<Self> {
        let cmd_pool = unsafe { ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None)? };

        let d = cfg.d_model as u64;
        let nq = cfg.n_q_heads as u64;
        let nkv = cfg.n_kv_heads as u64;
        let hd = cfg.head_dim as u64;
        let inter = cfg.moe_intermediate as u64;
        let vocab = cfg.vocab as u64;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                  | vk::BufferUsageFlags::TRANSFER_DST
                  | vk::BufferUsageFlags::TRANSFER_SRC;

        let make = |bytes: u64| GpuBuffer::new_vram(ctx, bytes, usage);

        // Pre-allocate 24 expert staging buffers (8 experts × 3 weights).
        // Max expert Q size: max(Q4K, Q6K) for n_weights = d*inter = 2048*768 = 1,572,864
        // Q6K: 1,290,240 bytes. Q4K: 884,736 bytes. Use Q6K as upper bound.
        let max_exp_bytes = ((d * inter) / QK_K as u64) * Q6K_BYTES;
        // Readback staging — router logits + also reused for lm_head readback
        let readback_router_off = 0u64;
        let readback_total = std::cmp::max(d * 4, cfg.n_experts as u64 * 4);
        let readback = GpuBuffer::new(ctx, readback_total,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
        let upload = GpuBuffer::new(ctx, d * 4,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;

        // Persistent fence + CB for async pipeline
        let pipe_fence = unsafe { ctx.device.create_fence(
            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)? };
        let pipe_cb = unsafe { ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1))?[0] };

        // Predictor + scheduler
        let predictor_builder = MatrixBuilder::new(cfg.n_layer as u16, cfg.n_experts as u16);
        let initial_matrix = predictor_builder.build_snapshot();
        let cache = Arc::new(ExpertCache::new(
            cfg.n_layer as u16, cfg.n_experts as u16,
            engine.config.vram_pool_slots as u32,
            1024, // RAM tier capacity (bookkeeping only)
        ));
        let sched_config = SchedulerConfig { prefetch_k_prime: 16, enabled: true };
        let scheduler = Scheduler::new(initial_matrix, cache, sched_config);

        Ok(Self {
            ctx, engine, weights, kv, cfg, pipes, cmd_pool,
            pos: 0,
            prev_picks: vec![Vec::new(); cfg.n_layer as usize],
            predictor_builder,
            scheduler,
            token_counter: 0,
            predict_hits: 0,
            predict_total: 0,
            h_a:           make(d * 4)?,
            h_b:           make(d * 4)?,
            x:             make(d * 4)?,
            q:             make(nq * hd * 4)?,
            k_new:         make(nkv * hd * 4)?,
            v_new:         make(nkv * hd * 4)?,
            scores:        make(nq * kv.max_seq as u64 * 4)?,
            probs:         make(nq * kv.max_seq as u64 * 4)?,
            attn_out:      make(nq * hd * 4)?,
            attn_proj:     make(d * 4)?,
            router_logits: make(cfg.n_experts as u64 * 4)?,
            expert_gate_dq: make(d * inter * 4)?,
            expert_up_dq:   make(d * inter * 4)?,
            expert_down_dq: make(inter * d * 4)?,
            g_all: make(8 * inter * 4)?,
            u_all: make(8 * inter * 4)?,
            s_all: make(8 * inter * 4)?,
            d_all: make(8 * d * 4)?,
            logits: make(vocab * 4)?,
            readback,
            rb_router_off: readback_router_off,
            upload, pipe_fence, pipe_cb, pipe_pending: false,
            expert_staging_stride: max_exp_bytes,
            // Attention: ~15 dispatches/layer × 48 = 720, MoE: ~35/layer × 48 = 1680, total ~2400
            arena: DescriptorArena::new(ctx, 2500, 5)?,
        })
    }

    // ---------- low-level helpers ----------

    fn record_sync<F>(&mut self, f: F) -> Result<()>
    where F: FnOnce(vk::CommandBuffer, &mut Self, &mut DescriptorArena) -> Result<()>
    {
        // Wait for any pending async work
        self.wait_pipe_fence()?;

        let mut arena = std::mem::replace(&mut self.arena, unsafe { std::mem::zeroed() });
        arena.reset(self.ctx)?;
        unsafe {
            self.ctx.device.reset_command_buffer(self.pipe_cb, vk::CommandBufferResetFlags::empty())?;
            self.ctx.device.begin_command_buffer(self.pipe_cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            f(self.pipe_cb, self, &mut arena)?;
            self.ctx.device.end_command_buffer(self.pipe_cb)?;

            self.ctx.device.reset_fences(&[self.pipe_fence])?;
            let cb_arr = [self.pipe_cb];
            self.ctx.device.queue_submit(self.ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&cb_arr)], self.pipe_fence)?;
            self.ctx.device.wait_for_fences(&[self.pipe_fence], true, u64::MAX)?;
        }
        self.arena = arena;
        Ok(())
    }

    /// Submit CB without waiting — GPU works while CPU prepares next batch.
    /// Caller must call wait_pipe_fence() before reading results or reusing buffers.
    fn record_async<F>(&mut self, f: F) -> Result<()>
    where F: FnOnce(vk::CommandBuffer, &mut Self, &mut DescriptorArena) -> Result<()>
    {
        self.wait_pipe_fence()?;

        let mut arena = std::mem::replace(&mut self.arena, unsafe { std::mem::zeroed() });
        arena.reset(self.ctx)?;
        unsafe {
            self.ctx.device.reset_command_buffer(self.pipe_cb, vk::CommandBufferResetFlags::empty())?;
            self.ctx.device.begin_command_buffer(self.pipe_cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            f(self.pipe_cb, self, &mut arena)?;
            self.ctx.device.end_command_buffer(self.pipe_cb)?;

            self.ctx.device.reset_fences(&[self.pipe_fence])?;
            let cb_arr = [self.pipe_cb];
            self.ctx.device.queue_submit(self.ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&cb_arr)], self.pipe_fence)?;
        }
        self.arena = arena;
        self.pipe_pending = true;
        Ok(())
    }

    fn wait_pipe_fence(&mut self) -> Result<()> {
        if self.pipe_pending {
            unsafe { self.ctx.device.wait_for_fences(&[self.pipe_fence], true, u64::MAX)?; }
            self.pipe_pending = false;
        }
        Ok(())
    }

    fn barrier(ctx: &VulkanContext, cb: vk::CommandBuffer) {
        unsafe {
            let bar = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
            ctx.device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[bar], &[], &[]);
        }
    }

    fn dispatch_full_s(
        ctx: &VulkanContext,
        pipe: &ComputePipeline,
        arena: &mut DescriptorArena,
        cb: vk::CommandBuffer,
        bufs: &[(&GpuBuffer, u64, u64)],
        push: &[u8],
        dx: u32, dy: u32, dz: u32,
    ) -> Result<()> {
        let set = arena.alloc_set(ctx, pipe, bufs)?;
        unsafe {
            ctx.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe.pipeline);
            ctx.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE,
                pipe.layout, 0, &[set], &[]);
            if !push.is_empty() {
                ctx.device.cmd_push_constants(cb, pipe.layout,
                    vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            ctx.device.cmd_dispatch(cb, dx, dy, dz);
        }
        Ok(())
    }

    fn dispatch_s(
        ctx: &VulkanContext,
        pipe: &ComputePipeline,
        arena: &mut DescriptorArena,
        cb: vk::CommandBuffer,
        bufs: &[(&GpuBuffer, u64)],
        push: &[u8],
        dx: u32, dy: u32, dz: u32,
    ) -> Result<()> {
        let triples: Vec<(&GpuBuffer, u64, u64)> = bufs.iter().map(|(b, s)| (*b, 0, *s)).collect();
        Self::dispatch_full_s(ctx, pipe, arena, cb, &triples, push, dx, dy, dz)
    }

    // ---------- public API ----------

    /// Process one token through all 48 layers. Updates KV cache, advances pos.
    pub fn step(&mut self, token_id: u32) -> Result<()> {
        let cfg = self.cfg.clone();
        let d = cfg.d_model as usize;
        let d_bytes = (d * 4) as u64;

        // Embed: CPU → upload staging → GPU
        let row = self.weights.embed_row(token_id);
        unsafe {
            let dst = std::slice::from_raw_parts_mut(self.upload.mapped() as *mut f32, d);
            dst.copy_from_slice(row);
        }
        self.record_sync(|cb, this, _b| {
            unsafe { this.ctx.device.cmd_copy_buffer(cb, this.upload.handle(), this.h_a.handle(),
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: d_bytes }]); }
            Ok(())
        })?;

        let skip_moe = std::env::var("MOE_OFF").is_ok();
        let use_cpu_moe = std::env::var("MOE_CPU_BRIDGE").is_ok();
        let n_exp = cfg.n_experts as usize;
        let inter = cfg.moe_intermediate as usize;

        let mut t_attn = 0.0f64;
        let mut t_load = 0.0f64;
        let mut t_moe_gpu = 0.0f64;
        let mut t_moe_cpu = 0.0f64;
        let mut total_hits = 0u32;
        let mut total_misses = 0u32;
        let mut gpu_experts = 0u32;
        let mut cpu_experts = 0u32;

        // Predicted experts queued for prefetch (from previous layer's prediction)
        let mut prefetch_list: Vec<(u32, ExpertKind, u32)> = Vec::new();
        let prefetch_enabled = self.token_counter >= 3; // warmup: skip prefetch first 3 tokens

        for l in 0..cfg.n_layer {
            // Prefetch predicted experts before attention
            if prefetch_enabled && !prefetch_list.is_empty() {
                let t1 = Instant::now();
                self.engine.loader.batch_enqueue(
                    &self.engine.ctx, &self.engine.reader, &mut self.engine.vram,
                    &prefetch_list,
                )?;
                self.engine.flush()?;
                t_load += t1.elapsed().as_secs_f64();
                prefetch_list.clear();
            }

            let t0 = Instant::now();
            self.run_layer_fused(l, false)?;
            t_attn += t0.elapsed().as_secs_f64();

            if !skip_moe {
                let rb = self.readback.mapped();
                let mut router = vec![0f32; n_exp];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (rb as *const f32).add(self.rb_router_off as usize / 4),
                        router.as_mut_ptr(), n_exp);
                }
                let picks = host_pick_topk(&router, 1, n_exp, cfg.top_k as usize);
                self.prev_picks[l as usize] = picks[0].iter().map(|&(idx, _)| idx).collect();

                // Predictor: observe + predict next layer
                {
                    let expert_ids: Vec<u16> = picks[0].iter().map(|&(idx, _)| idx as u16).collect();
                    let mut record: ActivationRecord = bytemuck::Zeroable::zeroed();
                    record.token_idx = self.pos;
                    record.layer = l as u16;
                    record.n_experts_used = expert_ids.len() as u8;
                    for (i, &e) in expert_ids.iter().enumerate().take(16) {
                        record.expert_ids[i] = e;
                        record.expert_weights[i] = picks[0][i].1;
                    }
                    self.predictor_builder.observe(&record);

                    let cmds = self.scheduler.on_layer_complete(l as u16, &expert_ids);
                    for cmd in &cmds {
                        if let SchedulerCommand::PrefetchToVram { expert, .. } = cmd {
                            let next_l = expert.layer as u32;
                            let slot = expert.expert as u32;
                            for &kind in &[ExpertKind::GateExps, ExpertKind::UpExps, ExpertKind::DownExps] {
                                prefetch_list.push((next_l, kind, slot));
                            }
                        }
                    }
                }

                if use_cpu_moe {
                    let t2 = Instant::now();
                    self.run_moe_cpu(l, &picks[0])?;
                    t_moe_cpu += t2.elapsed().as_secs_f64();
                } else {
                    // Hybrid dispatch: VRAM hit -> GPU, miss -> sync load + GPU
                    let mut gpu_batch: Vec<GpuExpertInfo> = Vec::new();
                    let mut miss_load: Vec<(u32, ExpertKind, u32)> = Vec::new();

                    for &(expert_idx, weight) in &picks[0] {
                        let gate_key = ExpertKey { layer: l, kind: ExpertKind::GateExps as u8, slot: expert_idx };
                        let up_key = ExpertKey { layer: l, kind: ExpertKind::UpExps as u8, slot: expert_idx };
                        let down_key = ExpertKey { layer: l, kind: ExpertKind::DownExps as u8, slot: expert_idx };

                        let all_hit = self.engine.vram.lookup(gate_key).is_some()
                            && self.engine.vram.lookup(up_key).is_some()
                            && self.engine.vram.lookup(down_key).is_some();

                        if all_hit {
                            total_hits += 3;
                            gpu_experts += 1;
                        } else {
                            // Queue sync load for misses
                            for &kind in &[ExpertKind::GateExps, ExpertKind::UpExps, ExpertKind::DownExps] {
                                miss_load.push((l, kind, expert_idx));
                            }
                            total_misses += 3;
                            cpu_experts += 1;
                        }
                    }

                    // Sync-load all misses in one batch
                    if !miss_load.is_empty() {
                        let t1 = Instant::now();
                        self.engine.loader.batch_enqueue(
                            &self.engine.ctx, &self.engine.reader, &mut self.engine.vram,
                            &miss_load,
                        )?;
                        self.engine.flush()?;
                        t_load += t1.elapsed().as_secs_f64();
                    }

                    // Now ALL experts are in VRAM, build GPU batch for all
                    for &(expert_idx, weight) in &picks[0] {
                        let gate_size = self.engine.reader.expert_size(l, ExpertKind::GateExps, expert_idx).unwrap();
                        let up_size = self.engine.reader.expert_size(l, ExpertKind::UpExps, expert_idx).unwrap();
                        let down_size = self.engine.reader.expert_size(l, ExpertKind::DownExps, expert_idx).unwrap();

                        let gate_key = ExpertKey { layer: l, kind: ExpertKind::GateExps as u8, slot: expert_idx };
                        let up_key = ExpertKey { layer: l, kind: ExpertKind::UpExps as u8, slot: expert_idx };
                        let down_key = ExpertKey { layer: l, kind: ExpertKind::DownExps as u8, slot: expert_idx };

                        let gate_slot = self.engine.vram.lookup(gate_key).unwrap();
                        let up_slot = self.engine.vram.lookup(up_key).unwrap();
                        let down_slot = self.engine.vram.lookup(down_key).unwrap();

                        let (_, gq6) = infer_quant(gate_size, d * inter);
                        let (_, dq6) = infer_quant(down_size, inter * d);

                        gpu_batch.push(GpuExpertInfo {
                            gate_off: self.engine.vram.slot_offset(gate_slot),
                            up_off: self.engine.vram.slot_offset(up_slot),
                            down_off: self.engine.vram.slot_offset(down_slot),
                            gate_bytes: gate_size as u64,
                            up_bytes: up_size as u64,
                            down_bytes: down_size as u64,
                            gate_q6: gq6, down_q6: dq6,
                            weight,
                        });
                    }

                    let t2 = Instant::now();
                    self.run_moe_gpu(l, &gpu_batch)?;
                    t_moe_gpu += t2.elapsed().as_secs_f64();
                }
            }
            std::mem::swap(&mut self.h_a, &mut self.h_b);
        }
        let hit_rate = if total_hits + total_misses > 0 { total_hits as f64 / (total_hits + total_misses) as f64 * 100.0 } else { 0.0 };
        eprintln!("  step timing: attn={:.1}ms load={:.1}ms moe_gpu={:.1}ms moe_cpu={:.1}ms total={:.1}ms vram_hit={:.0}% ({}/{}) gpu_exp={} cpu_exp={}",
            t_attn * 1000.0, t_load * 1000.0, t_moe_gpu * 1000.0, t_moe_cpu * 1000.0,
            (t_attn + t_load + t_moe_gpu + t_moe_cpu) * 1000.0,
            hit_rate, total_hits, total_hits + total_misses,
            gpu_experts, cpu_experts);
        // GPU MoE: result already in h_b (swapped to h_a), no upload needed
        self.pos += 1;
        self.token_counter += 1;

        // Refresh predictor snapshot every 50 tokens
        if self.token_counter % 50 == 0 && self.token_counter > 0 {
            let snap = self.predictor_builder.build_snapshot();
            self.scheduler.swap_matrix(snap);
            self.predictor_builder.clear_per_token();
        }

        Ok(())
    }

    /// After step(), read the hidden state, apply out_norm, and do CPU lm_head argmax.
    pub fn get_next_token(&mut self) -> Result<u32> {
        let cfg = self.cfg.clone();
        let d = cfg.d_model as u64;

        self.record_sync(|cb, this, arena| {
            let push_rms = pack_u32_f32(d as u32, this.cfg.rms_eps);
            Self::dispatch_s(this.ctx, &this.pipes.rmsnorm, arena, cb,
                &[(&this.h_a, d * 4), (&this.weights.out_norm, d * 4), (&this.x, d * 4)],
                &push_rms, 1, 1, 1)?;
            Ok(())
        })?;

        let x_host = self.read_buf_f32(&self.x, 0, cfg.d_model as usize)?;

        // lm_head: rayon parallel fp32 dot (pre-dequantized in host RAM)
        let lmh = &self.weights.lm_head_host;
        let dsz = cfg.d_model as usize;
        let chunk_sz = 4096usize;
        let (best_i, _) = (0..cfg.vocab as usize)
            .into_par_iter()
            .chunks(chunk_sz)
            .map(|chunk| {
                let mut bi = chunk[0]; let mut bv = f32::NEG_INFINITY;
                for i in chunk { let row = &lmh[i*dsz..(i+1)*dsz];
                    let mut acc = 0f32; for k in 0..dsz { acc += x_host[k]*row[k]; }
                    if acc > bv { bv = acc; bi = i; }
                }
                (bi, bv)
            }).reduce(|| (0, f32::NEG_INFINITY), |a, b| if a.1 >= b.1 { a } else { b });
        Ok(best_i as u32)
    }

    /// Convenience: process N prompt tokens, return next token id.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        for &tid in tokens {
            self.step(tid)?;
        }
        self.get_next_token()
    }

    /// Single CB per layer: attention + router + readback router only.
    /// MoE results stay on GPU — no x/hb readback needed.
    fn run_layer_fused(&mut self, layer: u32, _upload_first: bool) -> Result<()> {
        let cfg = self.cfg.clone();
        let d = cfg.d_model as u64;
        let nq = cfg.n_q_heads as u64;
        let nkv = cfg.n_kv_heads as u64;
        let hd = cfg.head_dim as u64;
        let max_seq = self.kv.max_seq as u64;
        let base_pos = self.pos;
        let cur_seq = base_pos + 1;
        let h_bytes = d * 4;
        let q_bytes = nq * hd * 4;
        let kv_bytes = nkv * hd * 4;
        let scores_bytes = nq * max_seq * 4;
        let attn_out_bytes = nq * hd * 4;
        let kv_layer_size = max_seq * nkv * hd * 4;
        let router_bytes = cfg.n_experts as u64 * 4;
        let li = layer as usize;
        let kv_write_off = base_pos as u64 * nkv * hd * 4;

        let push_norm = pack_u32_f32(d as u32, cfg.rms_eps);
        let push_qkv_q = pack_u32_2((nq*hd) as u32, d as u32);
        let push_qkv_k = pack_u32_2((nkv*hd) as u32, d as u32);
        let push_qnorm = pack_u32_f32(hd as u32, cfg.rms_eps);
        let push_rope_q = pack_rope(cfg.n_q_heads, cfg.head_dim, base_pos, cfg.rope_theta);
        let push_rope_k = pack_rope(cfg.n_kv_heads, cfg.head_dim, base_pos, cfg.rope_theta);
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let push_sd = pack_scaled_dot(1, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, cur_seq, base_pos, scale);
        let push_av = pack_attnv(1, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, cur_seq);
        let push_sm = cur_seq.to_le_bytes().to_vec();
        let push_o = pack_u32_2(d as u32, (nq*hd) as u32);
        let push_r = (d as u32).to_le_bytes().to_vec();
        let push_fn = pack_u32_f32(d as u32, cfg.rms_eps);
        let push_rt = pack_u32_2(cfg.n_experts, d as u32);
        let rb_rt = self.rb_router_off;

        self.record_sync(|cb, this, bi| {
            Self::dispatch_s(this.ctx, &this.pipes.rmsnorm, bi, cb,
                &[(&this.h_a, h_bytes), (&this.weights.layers[li].attn_norm, d*4), (&this.x, h_bytes)],
                &push_norm, 1, 1, 1)?;
            Self::barrier(this.ctx, cb);

            Self::dispatch_s(this.ctx, &this.pipes.matvec, bi, cb,
                &[(&this.x, h_bytes), (&this.weights.layers[li].attn_q, nq*hd*d*4), (&this.q, q_bytes)],
                &push_qkv_q, (nq*hd) as u32, 1, 1)?;
            Self::dispatch_s(this.ctx, &this.pipes.matvec, bi, cb,
                &[(&this.x, h_bytes), (&this.weights.layers[li].attn_k, nkv*hd*d*4), (&this.k_new, kv_bytes)],
                &push_qkv_k, (nkv*hd) as u32, 1, 1)?;
            Self::dispatch_s(this.ctx, &this.pipes.matvec, bi, cb,
                &[(&this.x, h_bytes), (&this.weights.layers[li].attn_v, nkv*hd*d*4), (&this.v_new, kv_bytes)],
                &push_qkv_k, (nkv*hd) as u32, 1, 1)?;
            Self::barrier(this.ctx, cb);

            Self::dispatch_s(this.ctx, &this.pipes.rmsnorm, bi, cb,
                &[(&this.q, q_bytes), (&this.weights.layers[li].attn_q_norm, hd*4), (&this.q, q_bytes)],
                &push_qnorm, cfg.n_q_heads, 1, 1)?;
            Self::dispatch_s(this.ctx, &this.pipes.rmsnorm, bi, cb,
                &[(&this.k_new, kv_bytes), (&this.weights.layers[li].attn_k_norm, hd*4), (&this.k_new, kv_bytes)],
                &push_qnorm, cfg.n_kv_heads, 1, 1)?;
            Self::barrier(this.ctx, cb);

            Self::dispatch_s(this.ctx, &this.pipes.rope, bi, cb, &[(&this.q, q_bytes)], &push_rope_q, cfg.n_q_heads, 1, 1)?;
            Self::dispatch_s(this.ctx, &this.pipes.rope, bi, cb, &[(&this.k_new, kv_bytes)], &push_rope_k, cfg.n_kv_heads, 1, 1)?;
            Self::barrier(this.ctx, cb);

            let k_dst = this.kv.k_offset(layer) + kv_write_off;
            let v_dst = this.kv.v_offset(layer) + kv_write_off;
            unsafe {
                this.ctx.device.cmd_copy_buffer(cb, this.k_new.handle(), this.kv.buf.handle(),
                    &[vk::BufferCopy { src_offset: 0, dst_offset: k_dst, size: kv_bytes }]);
                this.ctx.device.cmd_copy_buffer(cb, this.v_new.handle(), this.kv.buf.handle(),
                    &[vk::BufferCopy { src_offset: 0, dst_offset: v_dst, size: kv_bytes }]);
            }
            Self::barrier(this.ctx, cb);

            Self::dispatch_full_s(this.ctx, &this.pipes.scaled_dot, bi, cb, &[
                (&this.q, 0, q_bytes), (&this.kv.buf, this.kv.k_offset(layer), kv_layer_size),
                (&this.scores, 0, scores_bytes),
            ], &push_sd, cfg.n_q_heads, 1, 1)?;
            Self::barrier(this.ctx, cb);

            Self::dispatch_s(this.ctx, &this.pipes.softmax, bi, cb,
                &[(&this.scores, scores_bytes), (&this.probs, scores_bytes)], &push_sm, cfg.n_q_heads, 1, 1)?;
            Self::barrier(this.ctx, cb);

            Self::dispatch_full_s(this.ctx, &this.pipes.attn_v, bi, cb, &[
                (&this.probs, 0, scores_bytes), (&this.kv.buf, this.kv.v_offset(layer), kv_layer_size),
                (&this.attn_out, 0, attn_out_bytes),
            ], &push_av, cfg.n_q_heads, 1, 1)?;
            Self::barrier(this.ctx, cb);

            Self::dispatch_s(this.ctx, &this.pipes.matvec, bi, cb,
                &[(&this.attn_out, attn_out_bytes), (&this.weights.layers[li].attn_o, d*(nq*hd)*4), (&this.attn_proj, h_bytes)],
                &push_o, d as u32, 1, 1)?;
            Self::barrier(this.ctx, cb);

            Self::dispatch_s(this.ctx, &this.pipes.residual, bi, cb,
                &[(&this.h_a, h_bytes), (&this.attn_proj, h_bytes), (&this.h_b, h_bytes)],
                &push_r, (d as u32 + 255) / 256, 1, 1)?;
            Self::barrier(this.ctx, cb);

            // FFN norm + router
            Self::dispatch_s(this.ctx, &this.pipes.rmsnorm, bi, cb,
                &[(&this.h_b, h_bytes), (&this.weights.layers[li].ffn_norm, d*4), (&this.x, h_bytes)],
                &push_fn, 1, 1, 1)?;
            Self::barrier(this.ctx, cb);
            Self::dispatch_s(this.ctx, &this.pipes.matvec, bi, cb,
                &[(&this.x, h_bytes), (&this.weights.layers[li].router, cfg.n_experts as u64 * d * 4),
                  (&this.router_logits, router_bytes)],
                &push_rt, cfg.n_experts, 1, 1)?;
            Self::barrier(this.ctx, cb);

            // Copy router → readback staging (only router needed on CPU for top-k)
            unsafe {
                this.ctx.device.cmd_copy_buffer(cb, this.router_logits.handle(), this.readback.handle(),
                    &[vk::BufferCopy { src_offset: 0, dst_offset: rb_rt, size: router_bytes }]);
            }
            Ok(())
        })
    }

    /// GPU MoE: for each expert, run gate+up matvec → swiglu → down matvec → weighted_add.
    /// All data stays on GPU. Expert Q bytes are in VramPool, input x is in self.x,
    /// accumulation target is self.h_b.
    fn run_moe_gpu(&mut self, _layer: u32, experts: &[GpuExpertInfo]) -> Result<()> {
        let d = self.cfg.d_model as u64;
        let inter = self.cfg.moe_intermediate as u64;
        let h_bytes = d * 4;
        let inter_bytes = inter * 4;
        let n_exp = experts.len();

        self.record_async(|cb, this, bi| {
            // Phase 1: ALL gate + up matvecs (all independent, no barriers)
            for (ei, exp) in experts.iter().enumerate() {
                let g_off = ei as u64 * inter_bytes;
                let u_off = ei as u64 * inter_bytes;
                if !exp.gate_q6 {
                    Self::dispatch_full_s(this.ctx, &this.pipes.fused_q4k, bi, cb, &[
                        (&this.engine.vram.buffer, exp.gate_off, exp.gate_bytes),
                        (&this.x, 0, h_bytes),
                        (&this.g_all, g_off, inter_bytes),
                    ], &pack_u32_3(inter as u32, d as u32, 0), inter as u32, 1, 1)?;
                    Self::dispatch_full_s(this.ctx, &this.pipes.fused_q4k, bi, cb, &[
                        (&this.engine.vram.buffer, exp.up_off, exp.up_bytes),
                        (&this.x, 0, h_bytes),
                        (&this.u_all, u_off, inter_bytes),
                    ], &pack_u32_3(inter as u32, d as u32, 0), inter as u32, 1, 1)?;
                } else {
                    let n_blocks = (d * inter / QK_K) as u32;
                    Self::dispatch_full_s(this.ctx, &this.pipes.dq_q6k, bi, cb, &[
                        (&this.engine.vram.buffer, exp.gate_off, exp.gate_bytes),
                        (&this.expert_gate_dq, 0, d * inter * 4),
                    ], &n_blocks.to_le_bytes().to_vec(), n_blocks, 1, 1)?;
                    Self::dispatch_full_s(this.ctx, &this.pipes.dq_q6k, bi, cb, &[
                        (&this.engine.vram.buffer, exp.up_off, exp.up_bytes),
                        (&this.expert_up_dq, 0, d * inter * 4),
                    ], &n_blocks.to_le_bytes().to_vec(), n_blocks, 1, 1)?;
                    Self::barrier(this.ctx, cb);
                    Self::dispatch_full_s(this.ctx, &this.pipes.matvec, bi, cb, &[
                        (&this.x, 0, h_bytes), (&this.expert_gate_dq, 0, d * inter * 4),
                        (&this.g_all, g_off, inter_bytes),
                    ], &pack_u32_2(inter as u32, d as u32), inter as u32, 1, 1)?;
                    Self::dispatch_full_s(this.ctx, &this.pipes.matvec, bi, cb, &[
                        (&this.x, 0, h_bytes), (&this.expert_up_dq, 0, d * inter * 4),
                        (&this.u_all, u_off, inter_bytes),
                    ], &pack_u32_2(inter as u32, d as u32), inter as u32, 1, 1)?;
                }
            }
            Self::barrier(this.ctx, cb);

            // Phase 2: ALL swiglu
            for ei in 0..n_exp {
                let off = ei as u64 * inter_bytes;
                Self::dispatch_full_s(this.ctx, &this.pipes.swiglu, bi, cb, &[
                    (&this.g_all, off, inter_bytes),
                    (&this.u_all, off, inter_bytes),
                    (&this.s_all, off, inter_bytes),
                ], &(inter as u32).to_le_bytes().to_vec(), ((inter + 255) / 256) as u32, 1, 1)?;
            }
            Self::barrier(this.ctx, cb);

            // Phase 3: ALL down matvecs
            for (ei, exp) in experts.iter().enumerate() {
                let s_off = ei as u64 * inter_bytes;
                let d_off = ei as u64 * h_bytes;
                if !exp.down_q6 {
                    Self::dispatch_full_s(this.ctx, &this.pipes.fused_q4k, bi, cb, &[
                        (&this.engine.vram.buffer, exp.down_off, exp.down_bytes),
                        (&this.s_all, s_off, inter_bytes),
                        (&this.d_all, d_off, h_bytes),
                    ], &pack_u32_3(d as u32, inter as u32, 0), d as u32, 1, 1)?;
                } else {
                    let n_blocks = (inter * d / QK_K) as u32;
                    Self::dispatch_full_s(this.ctx, &this.pipes.dq_q6k, bi, cb, &[
                        (&this.engine.vram.buffer, exp.down_off, exp.down_bytes),
                        (&this.expert_down_dq, 0, inter * d * 4),
                    ], &n_blocks.to_le_bytes().to_vec(), n_blocks, 1, 1)?;
                    Self::barrier(this.ctx, cb);
                    Self::dispatch_full_s(this.ctx, &this.pipes.matvec, bi, cb, &[
                        (&this.s_all, s_off, inter_bytes),
                        (&this.expert_down_dq, 0, inter * d * 4),
                        (&this.d_all, d_off, h_bytes),
                    ], &pack_u32_2(d as u32, inter as u32), d as u32, 1, 1)?;
                }
            }
            Self::barrier(this.ctx, cb);

            // Phase 4: weighted sum of all expert outputs
            let mut push_wsum = Vec::with_capacity(40);
            push_wsum.extend_from_slice(&(d as u32).to_le_bytes());
            push_wsum.extend_from_slice(&(n_exp as u32).to_le_bytes());
            for exp in experts.iter() {
                push_wsum.extend_from_slice(&exp.weight.to_le_bytes());
            }
            for _ in n_exp..8 {
                push_wsum.extend_from_slice(&0.0f32.to_le_bytes());
            }
            Self::dispatch_s(this.ctx, &this.pipes.wsum8, bi, cb, &[
                (&this.h_b, h_bytes), (&this.d_all, 8 * h_bytes),
            ], &push_wsum, ((d + 255) / 256) as u32, 1, 1)?;

            Ok(())
        })
    }

    /// CPU MoE via ggml bridge: read x from GPU, do all expert matvecs on CPU,
    /// upload weighted result back to GPU h_b.
    fn run_moe_cpu(
        &mut self,
        layer: u32,
        picks: &[(u32, f32)],
    ) -> Result<()> {
        let d = self.cfg.d_model as usize;
        let inter = self.cfg.moe_intermediate as usize;

        let x_host = self.read_buf_f32(&self.x, 0, d)?;
        let h_b_host = self.read_buf_f32(&self.h_b, 0, d)?;

        let mut accum = vec![0f32; d];
        for &(expert_idx, weight) in picks {
            let gate_sz = self.engine.reader.expert_size(layer, ExpertKind::GateExps, expert_idx).unwrap();
            let up_sz = self.engine.reader.expert_size(layer, ExpertKind::UpExps, expert_idx).unwrap();
            let down_sz = self.engine.reader.expert_size(layer, ExpertKind::DownExps, expert_idx).unwrap();

            let mut gate_bytes = vec![0u8; gate_sz];
            let mut up_bytes = vec![0u8; up_sz];
            let mut down_bytes = vec![0u8; down_sz];
            self.engine.reader.read_into(layer, ExpertKind::GateExps, expert_idx, &mut gate_bytes)?;
            self.engine.reader.read_into(layer, ExpertKind::UpExps, expert_idx, &mut up_bytes)?;
            self.engine.reader.read_into(layer, ExpertKind::DownExps, expert_idx, &mut down_bytes)?;

            let (_, gate_q6) = infer_quant(gate_sz, d * inter);
            let (_, down_q6) = infer_quant(down_sz, inter * d);

            let mut gate_out = vec![0f32; inter];
            let mut up_out = vec![0f32; inter];

            if gate_q6 {
                let gate_fp32 = crate::weights::cpu_dequant_q6_k(&gate_bytes);
                let up_fp32 = crate::weights::cpu_dequant_q6_k(&up_bytes);
                cpu_matvec_fp32(&gate_fp32, &x_host, &mut gate_out, d, inter);
                cpu_matvec_fp32(&up_fp32, &x_host, &mut up_out, d, inter);
            } else {
                ggml_bridge::cpu_matvec_q4k(&gate_bytes, &x_host, &mut gate_out, d, inter);
                ggml_bridge::cpu_matvec_q4k(&up_bytes, &x_host, &mut up_out, d, inter);
            }

            let mut swiglu = vec![0f32; inter];
            for i in 0..inter {
                let g = gate_out[i];
                let sig = 1.0 / (1.0 + (-g).exp());
                swiglu[i] = up_out[i] * g * sig;
            }

            let mut down_out = vec![0f32; d];
            if down_q6 {
                let down_fp32 = crate::weights::cpu_dequant_q6_k(&down_bytes);
                cpu_matvec_fp32(&down_fp32, &swiglu, &mut down_out, inter, d);
            } else {
                ggml_bridge::cpu_matvec_q4k(&down_bytes, &swiglu, &mut down_out, inter, d);
            }

            for i in 0..d {
                accum[i] += weight * down_out[i];
            }
        }

        let mut result = h_b_host;
        for i in 0..d {
            result[i] += accum[i];
        }

        unsafe { self.upload.write_at(0, std::slice::from_raw_parts(result.as_ptr() as *const u8, d * 4)); }
        self.record_sync(|cb, this, _b| {
            unsafe { this.ctx.device.cmd_copy_buffer(cb, this.upload.handle(), this.h_b.handle(),
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: (d * 4) as u64 }]); }
            Ok(())
        })?;

        Ok(())
    }

    fn read_buf_f32(&self, buf: &GpuBuffer, src_offset: u64, n_floats: usize) -> Result<Vec<f32>> {
        let bytes = (n_floats * 4) as u64;
        assert!(bytes <= self.readback.size(), "readback staging too small: need {} have {}", bytes, self.readback.size());
        unsafe {
            let cb = self.ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1))?[0];
            self.ctx.device.begin_command_buffer(cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let region = [vk::BufferCopy { src_offset, dst_offset: 0, size: bytes }];
            self.ctx.device.cmd_copy_buffer(cb, buf.handle(), self.readback.handle(), &region);
            self.ctx.device.end_command_buffer(cb)?;
            let fence = self.ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cb_arr = [cb];
            self.ctx.device.queue_submit(self.ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
            self.ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.ctx.device.destroy_fence(fence, None);
            self.ctx.device.free_command_buffers(self.cmd_pool, &[cb]);
        }
        let mut out = vec![0f32; n_floats];
        unsafe {
            std::ptr::copy_nonoverlapping(self.readback.mapped() as *const f32, out.as_mut_ptr(), n_floats);
        }
        Ok(out)
    }
}

impl<'a> Drop for Forward<'a> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            self.ctx.device.destroy_fence(self.pipe_fence, None);
            self.ctx.device.destroy_command_pool(self.cmd_pool, None);
        }
        self.arena.destroy(self.ctx);
        for b in [&self.h_a, &self.h_b, &self.x, &self.q, &self.k_new, &self.v_new,
                  &self.scores, &self.probs, &self.attn_out, &self.attn_proj,
                  &self.router_logits,
                  &self.expert_gate_dq, &self.expert_up_dq, &self.expert_down_dq,
                  &self.g_all, &self.u_all, &self.s_all, &self.d_all, &self.logits,
                  &self.readback, &self.upload] {
            b.destroy(self.ctx);
        }
    }
}

// ── Push-constant packers ──

fn pack_u32_2(a: u32, b: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&a.to_le_bytes());
    v.extend_from_slice(&b.to_le_bytes());
    v
}

fn pack_u32_f32(a: u32, b: f32) -> Vec<u8> {
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

fn pack_rope(n_heads: u32, head_dim: u32, base_pos: u32, theta: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&n_heads.to_le_bytes());
    v.extend_from_slice(&head_dim.to_le_bytes());
    v.extend_from_slice(&base_pos.to_le_bytes());
    v.extend_from_slice(&theta.to_le_bytes());
    v
}

fn pack_scaled_dot(n_tok: u32, n_q: u32, n_kv: u32, hd: u32, seq_len: u32, base_pos: u32, scale: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(28);
    for x in [n_tok, n_q, n_kv, hd, seq_len, base_pos] { v.extend_from_slice(&x.to_le_bytes()); }
    v.extend_from_slice(&scale.to_le_bytes());
    v
}

fn pack_attnv(n_tok: u32, n_q: u32, n_kv: u32, hd: u32, seq_len: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(20);
    for x in [n_tok, n_q, n_kv, hd, seq_len] { v.extend_from_slice(&x.to_le_bytes()); }
    v
}

/// Infer quant type from byte size. Returns (n_blocks, is_q6k).
fn cpu_matvec_fp32(w: &[f32], x: &[f32], out: &mut [f32], n_in: usize, n_out: usize) {
    for row in 0..n_out {
        let mut acc = 0f32;
        let w_row = &w[row * n_in..(row + 1) * n_in];
        for k in 0..n_in {
            acc += w_row[k] * x[k];
        }
        out[row] = acc;
    }
}

fn infer_quant(byte_size: usize, n_weights: usize) -> (usize, bool) {
    let n_blocks = n_weights / QK_K as usize;
    let expected_q4k = n_blocks * Q4K_BYTES as usize;
    let expected_q6k = n_blocks * Q6K_BYTES as usize;
    if byte_size == expected_q6k {
        (n_blocks, true)
    } else if byte_size == expected_q4k {
        (n_blocks, false)
    } else {
        panic!("expert byte_size {byte_size} matches neither Q4K ({expected_q4k}) nor Q6K ({expected_q6k}) for {n_weights} weights");
    }
}

/// Per-token expert pick: (expert_idx, normalized_weight).
/// Qwen3-MoE: softmax over all 128 router logits, then take top-K, then
/// renormalize the K weights to sum to 1.
fn host_pick_topk(router_flat: &[f32], n_tok: usize, n_exp: usize, k: usize) -> Vec<Vec<(u32, f32)>> {
    let mut out = Vec::with_capacity(n_tok);
    for t in 0..n_tok {
        let row = &router_flat[t * n_exp .. (t + 1) * n_exp];
        // softmax
        let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for e in exps.iter_mut() { *e /= sum; }
        // top-k
        let mut idx: Vec<usize> = (0..n_exp).collect();
        idx.sort_unstable_by(|&a, &b| exps[b].partial_cmp(&exps[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut picks: Vec<(u32, f32)> = idx[..k].iter().map(|&i| (i as u32, exps[i])).collect();
        // renormalize the K weights so they sum to 1
        let s: f32 = picks.iter().map(|p| p.1).sum();
        for p in picks.iter_mut() { p.1 /= s; }
        out.push(picks);
    }
    out
}

fn _silence_unused(_g: &GgufFile) -> u64 {
    // Suppress unused-import warnings if GgufFile becomes truly unused.
    _g.file_size()
}

// Avoid `anyhow` unused-import warning if `bail!` removed
fn _err() -> anyhow::Error { anyhow!("") }
