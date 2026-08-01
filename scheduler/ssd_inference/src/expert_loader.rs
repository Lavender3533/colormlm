//! Pipelined expert uploader: pulls bytes from GGUF mmap → memcpy into a
//! ring of staging buffers → submit copy on the dedicated transfer queue
//! → fence-tracked completion.
//!
//! Designed for "fire many, await later" usage: caller `enqueue`s a batch of
//! expert loads, then `wait_all` to drain. Pipeline depth bounds in-flight
//! work to avoid unbounded memory pressure.

use crate::buffer::GpuBuffer;
use crate::device::VulkanContext;
use crate::expert_reader::ExpertReader;
use crate::vram_pool::{ExpertKey, VramPool};
use anyhow::Result;
use ash::vk;
use gguf_reader::ExpertKind;
use std::collections::HashMap;

const STAGING_BYTES_PER_SLOT: u64 = 1400 * 1024; // 1.37 MB default

#[derive(Clone, Copy)]
struct PendingTransfer {
    key: ExpertKey,
    vram_slot: u32,
}

pub struct ExpertLoader {
    pipeline_depth: usize,
    staging_bytes: u64,
    stagings: Vec<GpuBuffer>,
    cmd_pool: vk::CommandPool,
    cmd_buffers: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    next_pip: usize,
    n_in_flight: usize,
    pending: Vec<Option<PendingTransfer>>,
    n_experts_total: u32,

    // Stats
    pub stat_loads: u64,
    pub stat_bytes: u64,
}

impl ExpertLoader {
    pub fn new(ctx: &VulkanContext, pipeline_depth: usize, n_experts_total: u32) -> Result<Self> {
        Self::new_with_staging_bytes(ctx, pipeline_depth, n_experts_total, STAGING_BYTES_PER_SLOT)
    }

    /// Construct a loader whose staging slots can hold the model's largest
    /// expert page. Giant donor pages (for example K3) are much larger than
    /// the historical 1.37 MiB default, so the engine must pass its slot size.
    pub fn new_with_staging_bytes(
        ctx: &VulkanContext,
        pipeline_depth: usize,
        n_experts_total: u32,
        staging_bytes: u64,
    ) -> Result<Self> {
        if pipeline_depth == 0 {
            return Err(anyhow::anyhow!("pipeline depth must be greater than zero"));
        }
        if staging_bytes == 0 {
            return Err(anyhow::anyhow!(
                "staging slot size must be greater than zero"
            ));
        }
        let mut stagings = Vec::with_capacity(pipeline_depth);
        for _ in 0..pipeline_depth {
            stagings.push(GpuBuffer::new_staging(ctx, staging_bytes)?);
        }
        unsafe {
            let cmd_pool = ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_transfer)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmd_buffers = ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(pipeline_depth as u32),
            )?;
            let mut fences = Vec::with_capacity(pipeline_depth);
            for _ in 0..pipeline_depth {
                fences.push(ctx.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?);
            }
            Ok(Self {
                pipeline_depth,
                staging_bytes,
                stagings,
                cmd_pool,
                cmd_buffers,
                fences,
                next_pip: 0,
                n_in_flight: 0,
                pending: vec![None; pipeline_depth],
                n_experts_total,
                stat_loads: 0,
                stat_bytes: 0,
            })
        }
    }

    fn validate_transfer_size(&self, expert_size: usize, pool: &VramPool) -> Result<()> {
        if expert_size as u64 > self.staging_bytes {
            return Err(anyhow::anyhow!(
                "expert page is too large for staging slot: {} > {} bytes",
                expert_size,
                self.staging_bytes
            ));
        }
        if expert_size as u64 > pool.slot_bytes() {
            return Err(anyhow::anyhow!(
                "expert page is too large for VRAM slot: {} > {} bytes",
                expert_size,
                pool.slot_bytes()
            ));
        }
        Ok(())
    }

    fn complete_pipeline_slot(
        &mut self,
        ctx: &VulkanContext,
        pool: &mut VramPool,
        pip: usize,
    ) -> Result<()> {
        if let Some(pending) = self.pending[pip] {
            unsafe {
                ctx.device
                    .wait_for_fences(&[self.fences[pip]], true, u64::MAX)?;
            }
            pool.mark_ready(pending.vram_slot, pending.key)?;
            self.pending[pip] = None;
            self.n_in_flight = self.pending.iter().filter(|item| item.is_some()).count();
        }
        Ok(())
    }

    fn complete_pending_key(
        &mut self,
        ctx: &VulkanContext,
        pool: &mut VramPool,
        key: ExpertKey,
    ) -> Result<Option<u32>> {
        let pip = self
            .pending
            .iter()
            .position(|pending| pending.map(|item| item.key) == Some(key));
        if let Some(pip) = pip {
            self.complete_pipeline_slot(ctx, pool, pip)?;
            return Ok(pool.lookup(key));
        }
        Ok(None)
    }

    /// Enqueue one expert load. Blocks if pipeline is full (waits for oldest
    /// in-flight to complete and reclaims its slot).
    ///
    /// Returns the VRAM slot index the expert was placed into.
    pub fn enqueue(
        &mut self,
        ctx: &VulkanContext,
        reader: &ExpertReader,
        pool: &mut VramPool,
        layer: u32,
        kind: ExpertKind,
        expert_slot: u32,
    ) -> Result<u32> {
        let key = ExpertKey {
            layer,
            kind: kind as u8,
            slot: expert_slot,
        };

        // Already loaded? Return cached slot.
        if let Some(idx) = pool.lookup(key) {
            return Ok(idx);
        }
        if let Some(idx) = self.complete_pending_key(ctx, pool, key)? {
            return Ok(idx);
        }

        let expert_size = reader
            .expert_size(layer, kind, expert_slot)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "expert ({},{:?},{}) not in reader index",
                    layer,
                    kind,
                    expert_slot
                )
            })?;
        self.validate_transfer_size(expert_size, pool)?;

        let pip = self.next_pip;
        self.next_pip = (self.next_pip + 1) % self.pipeline_depth;

        let t_fence = std::time::Instant::now();
        self.complete_pipeline_slot(ctx, pool, pip)?;
        let fence_us = t_fence.elapsed().as_micros();

        // Allocate a hidden VRAM slot (may evict). It is not visible through
        // lookup until this pipeline slot's fence has completed.
        let (vram_slot_idx, _evicted) = pool.reserve_loading(key)?;

        // Read expert bytes directly into staging via File::seek + read_exact
        let t_read = std::time::Instant::now();
        let read_result = unsafe {
            let staging_ptr = self.stagings[pip].mapped();
            let dest = std::slice::from_raw_parts_mut(staging_ptr, expert_size);
            reader.read_into(layer, kind, expert_slot, dest)
        };
        if let Err(error) = read_result {
            pool.cancel_loading(vram_slot_idx, key);
            return Err(error);
        }
        let read_us = t_read.elapsed().as_micros();

        // Record copy cmd
        let t_submit = std::time::Instant::now();
        let cb = self.cmd_buffers[pip];
        let submit_result: Result<()> = unsafe {
            (|| {
                ctx.device
                    .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
                ctx.device.begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let region = [vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(pool.slot_offset(vram_slot_idx))
                    .size(expert_size as u64)];
                ctx.device.cmd_copy_buffer(
                    cb,
                    self.stagings[pip].handle(),
                    pool.buffer.handle(),
                    &region,
                );
                ctx.device.end_command_buffer(cb)?;

                let cb_arr = [cb];
                ctx.device.reset_fences(&[self.fences[pip]])?;
                ctx.device.queue_submit(
                    ctx.q_transfer,
                    &[vk::SubmitInfo::default().command_buffers(&cb_arr)],
                    self.fences[pip],
                )?;
                Ok(())
            })()
        };
        if let Err(error) = submit_result {
            pool.cancel_loading(vram_slot_idx, key);
            return Err(error);
        }
        self.pending[pip] = Some(PendingTransfer {
            key,
            vram_slot: vram_slot_idx,
        });
        let submit_us = t_submit.elapsed().as_micros();

        if std::env::var("IO_TIMING").is_ok() && self.stat_loads < 50 {
            eprintln!(
                "    enqueue L{} {:?} E{}: fence={}us read={}us({} KB) submit={}us",
                layer,
                kind,
                expert_slot,
                fence_us,
                read_us,
                expert_size / 1024,
                submit_us
            );
        }

        self.n_in_flight = self.pending.iter().filter(|item| item.is_some()).count();
        self.stat_loads += 1;
        self.stat_bytes += expert_size as u64;
        Ok(vram_slot_idx)
    }

    /// Wait for all in-flight transfers to complete.
    pub fn wait_all(&mut self, ctx: &VulkanContext, pool: &mut VramPool) -> Result<()> {
        for pip in 0..self.pipeline_depth {
            self.complete_pipeline_slot(ctx, pool, pip)?;
        }
        Ok(())
    }

    /// Batch-load multiple experts in waves no deeper than the staging ring.
    ///
    /// A prior implementation assigned every miss a ring index before issuing
    /// any transfer. More misses than `pipeline_depth` therefore reused the
    /// same mapped staging memory and allowed later reads to overwrite earlier
    /// ones. Explicit waves preserve disk/transfer overlap without aliasing.
    pub fn batch_enqueue(
        &mut self,
        ctx: &VulkanContext,
        reader: &ExpertReader,
        pool: &mut VramPool,
        experts: &[(u32, ExpertKind, u32)],
    ) -> Result<Vec<u32>> {
        if experts.is_empty() {
            return Ok(Vec::new());
        }

        // Settle earlier asynchronous work so duplicate detection observes the
        // authoritative Ready index before this batch is planned.
        self.wait_all(ctx, pool)?;

        let mut distinct = HashMap::<ExpertKey, usize>::new();
        let mut unique = Vec::<(u32, ExpertKind, u32)>::new();
        let mut input_to_unique = Vec::with_capacity(experts.len());
        for &(layer, kind, slot) in experts {
            let key = ExpertKey {
                layer,
                kind: kind as u8,
                slot,
            };
            let next = unique.len();
            let unique_idx = *distinct.entry(key).or_insert_with(|| {
                unique.push((layer, kind, slot));
                next
            });
            input_to_unique.push(unique_idx);
        }

        if unique.len() > pool.capacity() as usize {
            return Err(anyhow::anyhow!(
                "batch requests {} distinct experts but VRAM pool has only {} slots",
                unique.len(),
                pool.capacity()
            ));
        }

        let mut unique_slots = vec![0u32; unique.len()];
        let wave_count = unique.len().div_ceil(self.pipeline_depth);
        for (wave_idx, wave) in unique.chunks(self.pipeline_depth).enumerate() {
            let base = wave_idx * self.pipeline_depth;
            for (offset, &(layer, kind, slot)) in wave.iter().enumerate() {
                unique_slots[base + offset] = self.enqueue(ctx, reader, pool, layer, kind, slot)?;
            }
            // A later wave would reuse staging slots, so publish and drain the
            // current wave first. The final wave remains asynchronous and is
            // made visible by the caller's normal `wait_all`/`flush` barrier.
            if wave_idx + 1 < wave_count {
                self.wait_all(ctx, pool)?;
            }
        }

        Ok(input_to_unique
            .into_iter()
            .map(|unique_idx| unique_slots[unique_idx])
            .collect())
    }

    /// Enqueue a transfer from pre-read bytes (already in CPU RAM).
    /// Skips disk I/O — just copies to staging and submits transfer.
    pub fn enqueue_from_buf(
        &mut self,
        ctx: &VulkanContext,
        pool: &mut VramPool,
        layer: u32,
        kind: ExpertKind,
        expert_slot: u32,
        data: &[u8],
    ) -> Result<u32> {
        let key = ExpertKey {
            layer,
            kind: kind as u8,
            slot: expert_slot,
        };
        if let Some(idx) = pool.lookup(key) {
            return Ok(idx);
        }
        if let Some(idx) = self.complete_pending_key(ctx, pool, key)? {
            return Ok(idx);
        }

        let expert_size = data.len();
        self.validate_transfer_size(expert_size, pool)?;

        let pip = self.next_pip;
        self.next_pip = (self.next_pip + 1) % self.pipeline_depth;

        self.complete_pipeline_slot(ctx, pool, pip)?;

        let (vram_slot_idx, _evicted) = pool.reserve_loading(key)?;

        unsafe {
            self.stagings[pip].write_at(0, data);
        }

        let cb = self.cmd_buffers[pip];
        let submit_result: Result<()> = unsafe {
            (|| {
                ctx.device
                    .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
                ctx.device.begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let region = [vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(pool.slot_offset(vram_slot_idx))
                    .size(expert_size as u64)];
                ctx.device.cmd_copy_buffer(
                    cb,
                    self.stagings[pip].handle(),
                    pool.buffer.handle(),
                    &region,
                );
                ctx.device.end_command_buffer(cb)?;

                let cb_arr = [cb];
                ctx.device.reset_fences(&[self.fences[pip]])?;
                ctx.device.queue_submit(
                    ctx.q_transfer,
                    &[vk::SubmitInfo::default().command_buffers(&cb_arr)],
                    self.fences[pip],
                )?;
                Ok(())
            })()
        };
        if let Err(error) = submit_result {
            pool.cancel_loading(vram_slot_idx, key);
            return Err(error);
        }

        self.pending[pip] = Some(PendingTransfer {
            key,
            vram_slot: vram_slot_idx,
        });
        self.n_in_flight = self.pending.iter().filter(|item| item.is_some()).count();
        self.stat_loads += 1;
        self.stat_bytes += expert_size as u64;
        Ok(vram_slot_idx)
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            // Only submitted transfers need a wait. A failed queue submission
            // can leave its reset fence unsignaled without a PendingTransfer;
            // waiting every ring fence here would then deadlock teardown.
            let active_fences: Vec<_> = self
                .pending
                .iter()
                .enumerate()
                .filter_map(|(pip, pending)| pending.map(|_| self.fences[pip]))
                .collect();
            if !active_fences.is_empty() {
                let _ = ctx.device.wait_for_fences(&active_fences, true, u64::MAX);
            }
            for &f in &self.fences {
                ctx.device.destroy_fence(f, None);
            }
            ctx.device.destroy_command_pool(self.cmd_pool, None);
        }
        for s in &self.stagings {
            s.destroy(ctx);
        }
    }
}
