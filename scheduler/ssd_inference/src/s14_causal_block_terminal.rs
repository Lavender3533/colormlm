//! K=4/8 causal-block 的 production checkpoint arena 与 batched terminal/head 数据面。
//!
//! 本模块不生成或伪造 K 份 candidate state。调用方必须绑定由43层图真实产生的 K 个
//! 完整 device checkpoint、同源 K-row normalized head input、32个 production head chunk
//! 以及 producer timeline。recorder 在一个 command submit 中复制 K-prefix checkpoints、
//! 用 `production_batched(K)` 扫描一次32个 chunk、回读 K 行 top-1，并以 owned lease
//! 保活 arena/timeline；任一步失败都先 drain 已提交工作再释放 reservation。

use crate::{
    compute::StorageBufferSlice,
    s14_causal_block_layer::{
        S14CausalBlockDeviceCheckpointStorage, S14CausalBlockDeviceFutureReceipt,
        S14CausalBlockFinalOutput, S14CausalBlockHiddenBinding, S14CausalBlockOwnedDeviceFuture,
        S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
    },
    s14_causal_block_vulkan_backend::{
        S14CausalBlockVulkanFutureLease, S14CausalBlockVulkanFutureLeasePool,
    },
    s14_head_chunk_argmax::{
        decode_batched_head_argmax, S14HeadArgmaxResult, S14HeadChunkArgmaxDispatch,
        S14HeadChunkArgmaxPipeline, S14HeadChunkArgmaxRecorder, S14HeadChunkArgmaxRecordingReceipt,
        S14HeadChunkArgmaxShape, S14HeadChunkWorkspace, S14_HEAD_ARGMAX_WORDS,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::BatchedWholeTokenOutput;
use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
};

const CHECKPOINT_ALIGNMENT: u64 = 256;
const MAX_BLOCK_SIZE: usize = 8;
const TERMINAL_WAIT_TIMEOUT_NS: u64 = 60_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockCheckpointArenaLayout {
    pub checkpoint_state_bytes: u64,
    pub checkpoint_stride_bytes: u64,
    pub max_block_size: usize,
    pub slots: usize,
    pub slot_bytes: u64,
    pub arena_bytes: u64,
}

impl S14CausalBlockCheckpointArenaLayout {
    pub fn build(checkpoint_state_bytes: u64, slots: usize) -> Result<Self> {
        if checkpoint_state_bytes == 0 || slots == 0 {
            bail!("causal-block checkpoint arena state bytes/slots 不能为0");
        }
        let checkpoint_stride_bytes = align_up(checkpoint_state_bytes, CHECKPOINT_ALIGNMENT)?;
        let slot_bytes = checkpoint_stride_bytes
            .checked_mul(MAX_BLOCK_SIZE as u64)
            .context("causal-block checkpoint slot bytes overflow")?;
        let arena_bytes = slot_bytes
            .checked_mul(slots as u64)
            .context("causal-block checkpoint arena bytes overflow")?;
        Ok(Self {
            checkpoint_state_bytes,
            checkpoint_stride_bytes,
            max_block_size: MAX_BLOCK_SIZE,
            slots,
            slot_bytes,
            arena_bytes,
        })
    }

    fn slot_offset(self, slot: usize) -> Result<u64> {
        if slot >= self.slots {
            bail!("causal-block checkpoint slot 越界");
        }
        self.slot_bytes
            .checked_mul(slot as u64)
            .context("causal-block checkpoint slot offset overflow")
    }

    fn used_bytes(self, block_size: usize) -> Result<u64> {
        if !matches!(block_size, 4 | 8) {
            bail!("causal-block checkpoint arena 只接受 K=4/8");
        }
        self.checkpoint_stride_bytes
            .checked_mul(block_size.saturating_sub(1) as u64)
            .and_then(|prefix| prefix.checked_add(self.checkpoint_state_bytes))
            .context("causal-block checkpoint used bytes overflow")
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14CausalBlockCheckpointArenaTelemetry {
    pub acquisitions: u64,
    pub committed_leases: u64,
    pub releases: u64,
    pub cancellations: u64,
    pub rejected_releases: u64,
    pub in_use: usize,
    pub free: usize,
}

#[derive(Clone, Copy, Debug)]
struct LeaseEntry {
    slot: usize,
    block_size: usize,
    receipt: Option<S14CausalBlockDeviceFutureReceipt>,
}

#[derive(Debug)]
struct LeaseLedger {
    free: Vec<bool>,
    active: BTreeMap<u64, LeaseEntry>,
    next_lease_id: u64,
    next_timeline_value: u64,
    telemetry: S14CausalBlockCheckpointArenaTelemetry,
}

impl LeaseLedger {
    fn new(slots: usize) -> Self {
        Self {
            free: vec![true; slots],
            active: BTreeMap::new(),
            next_lease_id: 1,
            next_timeline_value: 1,
            telemetry: S14CausalBlockCheckpointArenaTelemetry {
                free: slots,
                ..S14CausalBlockCheckpointArenaTelemetry::default()
            },
        }
    }

    fn reserve(&mut self, block_size: usize) -> Result<(u64, usize)> {
        if !matches!(block_size, 4 | 8) {
            bail!("causal-block checkpoint reservation 只接受 K=4/8");
        }
        let slot = self
            .free
            .iter()
            .position(|&free| free)
            .context("causal-block checkpoint arena 没有空闲 slot")?;
        let lease_id = self.next_lease_id;
        self.next_lease_id = self
            .next_lease_id
            .checked_add(1)
            .context("causal-block checkpoint lease id overflow")?;
        self.free[slot] = false;
        self.active.insert(
            lease_id,
            LeaseEntry {
                slot,
                block_size,
                receipt: None,
            },
        );
        self.telemetry.acquisitions += 1;
        self.refresh_counts();
        Ok((lease_id, slot))
    }

    fn next_timeline_value(&mut self) -> Result<u64> {
        let value = self.next_timeline_value;
        self.next_timeline_value = self
            .next_timeline_value
            .checked_add(1)
            .context("causal-block checkpoint timeline value overflow")?;
        Ok(value)
    }

    fn commit(&mut self, lease_id: u64, receipt: S14CausalBlockDeviceFutureReceipt) -> Result<()> {
        let entry = self
            .active
            .get_mut(&lease_id)
            .context("causal-block checkpoint reservation 不存在")?;
        if entry.receipt.is_some() || entry.block_size != receipt.block_size {
            bail!("causal-block checkpoint reservation/receipt 漂移");
        }
        entry.receipt = Some(receipt);
        self.telemetry.committed_leases += 1;
        Ok(())
    }

    fn cancel(&mut self, lease_id: u64) {
        let Some(entry) = self.active.get(&lease_id).copied() else {
            return;
        };
        if entry.receipt.is_some() {
            self.telemetry.rejected_releases += 1;
            return;
        }
        self.active.remove(&lease_id);
        self.free[entry.slot] = true;
        self.telemetry.cancellations += 1;
        self.refresh_counts();
    }

    fn release(&mut self, lease_id: u64, receipt: S14CausalBlockDeviceFutureReceipt) {
        let Some(entry) = self.active.get(&lease_id).copied() else {
            self.telemetry.rejected_releases += 1;
            return;
        };
        if entry.receipt != Some(receipt) {
            self.telemetry.rejected_releases += 1;
            return;
        }
        self.active.remove(&lease_id);
        self.free[entry.slot] = true;
        self.telemetry.releases += 1;
        self.refresh_counts();
    }

    fn refresh_counts(&mut self) {
        self.telemetry.in_use = self.active.len();
        self.telemetry.free = self.free.iter().filter(|&&free| free).count();
    }
}

/// arena buffer 与 ready timeline 的真正 owner。owned future 中的 lease 持有本 pool 的
/// `Arc`，因此 recorder 被销毁后 receipt 的 buffer/semaphore 仍保持有效。
pub struct S14CausalBlockCheckpointArenaPool {
    ctx: Arc<VulkanContext>,
    layout: S14CausalBlockCheckpointArenaLayout,
    arena: GpuBuffer,
    ready_timeline: vk::Semaphore,
    ledger: Mutex<LeaseLedger>,
}

impl fmt::Debug for S14CausalBlockCheckpointArenaPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockCheckpointArenaPool")
            .field("layout", &self.layout)
            .field("arena", &self.arena.handle())
            .field("ready_timeline", &self.ready_timeline)
            .field("telemetry", &self.telemetry().ok())
            .finish()
    }
}

impl S14CausalBlockCheckpointArenaPool {
    pub fn new(
        ctx: Arc<VulkanContext>,
        checkpoint_state_bytes: u64,
        slots: usize,
    ) -> Result<Arc<Self>> {
        if !ctx.timeline_semaphore {
            bail!("causal-block checkpoint arena 要求 timeline semaphore");
        }
        let layout = S14CausalBlockCheckpointArenaLayout::build(checkpoint_state_bytes, slots)?;
        let usage = vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let arena = GpuBuffer::new_vram(&ctx, layout.arena_bytes, usage)
            .context("allocate causal-block checkpoint arena")?;
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let ready_timeline = match unsafe {
            ctx.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
                None,
            )
        } {
            Ok(semaphore) => semaphore,
            Err(error) => {
                arena.destroy(&ctx);
                return Err(error.into());
            }
        };
        Ok(Arc::new(Self {
            ctx,
            layout,
            arena,
            ready_timeline,
            ledger: Mutex::new(LeaseLedger::new(slots)),
        }))
    }

    pub fn layout(&self) -> S14CausalBlockCheckpointArenaLayout {
        self.layout
    }

    pub fn telemetry(&self) -> Result<S14CausalBlockCheckpointArenaTelemetry> {
        Ok(self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("causal-block checkpoint ledger poisoned"))?
            .telemetry)
    }

    fn acquire(self: &Arc<Self>, block_size: usize) -> Result<CheckpointReservation> {
        let (lease_id, slot) = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("causal-block checkpoint ledger poisoned"))?
            .reserve(block_size)?;
        Ok(CheckpointReservation {
            pool: Arc::clone(self),
            lease_id,
            slot,
            block_size,
            armed: true,
        })
    }

    fn next_ready_value(&self) -> Result<u64> {
        self.ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("causal-block checkpoint ledger poisoned"))?
            .next_timeline_value()
    }
}

impl S14CausalBlockVulkanFutureLeasePool for S14CausalBlockCheckpointArenaPool {
    fn release_future_lease(&self, lease_id: u64, receipt: S14CausalBlockDeviceFutureReceipt) {
        if let Ok(mut ledger) = self.ledger.lock() {
            ledger.release(lease_id, receipt);
        }
    }
}

impl Drop for S14CausalBlockCheckpointArenaPool {
    fn drop(&mut self) {
        debug_assert!(
            self.ledger
                .get_mut()
                .is_ok_and(|ledger| ledger.active.is_empty()),
            "checkpoint arena pool dropped with live leases"
        );
        unsafe {
            self.ctx.device.destroy_semaphore(self.ready_timeline, None);
        }
        self.arena.destroy(&self.ctx);
    }
}

struct CheckpointReservation {
    pool: Arc<S14CausalBlockCheckpointArenaPool>,
    lease_id: u64,
    slot: usize,
    block_size: usize,
    armed: bool,
}

impl CheckpointReservation {
    fn arena_offset(&self) -> Result<u64> {
        self.pool.layout.slot_offset(self.slot)
    }

    fn commit(
        mut self,
        receipt: S14CausalBlockDeviceFutureReceipt,
    ) -> Result<S14CausalBlockOwnedDeviceFuture> {
        let expected_offset = self.arena_offset()?;
        if receipt.block_size != self.block_size
            || receipt.checkpoint_arena != self.pool.arena.handle()
            || receipt.checkpoint_arena_offset != expected_offset
            || receipt.checkpoint_arena_bytes != self.pool.layout.used_bytes(self.block_size)?
            || receipt.checkpoint_stride_bytes != self.pool.layout.checkpoint_stride_bytes
            || receipt.checkpoint_state_bytes != self.pool.layout.checkpoint_state_bytes
            || receipt.ready_timeline != self.pool.ready_timeline
        {
            bail!("causal-block checkpoint reservation receipt identity 漂移");
        }
        self.pool
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("causal-block checkpoint ledger poisoned"))?
            .commit(self.lease_id, receipt)?;
        let pool: Arc<dyn S14CausalBlockVulkanFutureLeasePool> = self.pool.clone();
        let lease = S14CausalBlockVulkanFutureLease::new(receipt, self.lease_id, pool)
            .map_err(anyhow::Error::msg)?;
        self.armed = false;
        Ok(S14CausalBlockOwnedDeviceFuture::new(lease))
    }
}

impl Drop for CheckpointReservation {
    fn drop(&mut self) {
        if self.armed {
            if let Ok(mut ledger) = self.pool.ledger.lock() {
                ledger.cancel(self.lease_id);
            }
            self.armed = false;
        }
    }
}

#[derive(Clone, Copy)]
pub struct S14CausalBlockCheckpointSource<'a> {
    pub state: StorageBufferSlice<'a>,
}

/// 真实数据面的强输入；任何缺项都 fail-closed。`normalized_head_rows` 必须由
/// `final_hidden` 的 batched terminal/RMSNorm producer 生成，本模块不从 host 伪造。
pub struct S14CausalBlockTerminalInputBinding<'a> {
    pub base_position: u32,
    pub final_hidden: S14CausalBlockHiddenBinding,
    pub normalized_head_rows: StorageBufferSlice<'a>,
    pub checkpoints: &'a [S14CausalBlockCheckpointSource<'a>],
    pub head_chunks: &'a [StorageBufferSlice<'a>],
    pub producer_timeline: vk::Semaphore,
    pub producer_timeline_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TerminalWorkspaceLayout {
    normalized_offset: u64,
    logits_offset: u64,
    accumulator_offset: u64,
    total_bytes: u64,
}

impl TerminalWorkspaceLayout {
    fn build(shape: S14HeadChunkArgmaxShape) -> Result<Self> {
        let normalized_offset = 0;
        let logits_offset = align_up(shape.normalized_input_bytes()?, CHECKPOINT_ALIGNMENT)?;
        let accumulator_offset = align_up(
            logits_offset
                .checked_add(shape.max_chunk_logits_bytes()?)
                .context("causal-block head logits end overflow")?,
            CHECKPOINT_ALIGNMENT,
        )?;
        let total_bytes = align_up(
            accumulator_offset
                .checked_add(shape.argmax_bytes()?)
                .context("causal-block head accumulator end overflow")?,
            CHECKPOINT_ALIGNMENT,
        )?;
        Ok(Self {
            normalized_offset,
            logits_offset,
            accumulator_offset,
            total_bytes,
        })
    }
}

/// GPU 侧真实导出。只有调用方再提供同源的 K 份 host candidate output，才能构造
/// `S14CausalBlockFinalOutput`；这一步不会合成 checkpoint 或 prediction。
pub struct S14CausalBlockTerminalGpuExport {
    pub base_position: u32,
    pub block_size: usize,
    pub final_hidden: S14CausalBlockHiddenBinding,
    pub head_recording: S14HeadChunkArgmaxRecordingReceipt,
    pub head_results: Vec<S14HeadArgmaxResult>,
    pub device_future: S14CausalBlockOwnedDeviceFuture,
    pub ready_timeline_value: u64,
}

impl S14CausalBlockTerminalGpuExport {
    pub fn bind_host_output(
        self,
        output: BatchedWholeTokenOutput,
    ) -> Result<S14CausalBlockFinalOutput> {
        if output.forward_calls != 1
            || output.positions.len() != self.block_size
            || self.head_results.len() != self.block_size
            || output
                .positions
                .iter()
                .zip(&self.head_results)
                .any(|(position, head)| position.predicted_token_id != head.token_id)
        {
            bail!("causal-block GPU head 与真实 host candidate output 不同源");
        }
        Ok(S14CausalBlockFinalOutput {
            output,
            head_recording: self.head_recording,
            head_results: self.head_results,
            device_future: self.device_future,
            batched_head_submit_calls: 1,
            checkpoint_export_calls: 1,
            serial_token_forward_calls: 0,
        })
    }
}

/// 持久 command/workspace/readback/pipeline owner。checkpoint arena/timeline 由 pool 独立
/// 持有，保证 sealed future 离开 recorder 生命周期后仍可发布或 rollback。
pub struct S14CausalBlockBatchedTerminalRecorder {
    ctx: Arc<VulkanContext>,
    pool: Arc<S14CausalBlockCheckpointArenaPool>,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    workspace: Option<GpuBuffer>,
    readback: Option<GpuBuffer>,
    head_pipeline: Option<S14HeadChunkArgmaxPipeline>,
}

impl fmt::Debug for S14CausalBlockBatchedTerminalRecorder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockBatchedTerminalRecorder")
            .field("checkpoint_pool", &self.pool)
            .finish_non_exhaustive()
    }
}

impl S14CausalBlockBatchedTerminalRecorder {
    pub fn new(
        ctx: Arc<VulkanContext>,
        pool: Arc<S14CausalBlockCheckpointArenaPool>,
    ) -> Result<Self> {
        if !Arc::ptr_eq(&ctx, &pool.ctx) {
            bail!("causal-block terminal recorder/pool 不属于同一 Vulkan context");
        }
        let max_shape = S14HeadChunkArgmaxShape::production_batched(MAX_BLOCK_SIZE as u32)?;
        let workspace_layout = TerminalWorkspaceLayout::build(max_shape)?;
        let workspace = GpuBuffer::new_vram(
            &ctx,
            workspace_layout.total_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .context("allocate causal-block terminal workspace")?;
        let readback = match GpuBuffer::new(
            &ctx,
            max_shape.argmax_bytes()?,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                workspace.destroy(&ctx);
                return Err(error.context("allocate causal-block head readback"));
            }
        };
        let head_pipeline = match S14HeadChunkArgmaxPipeline::new(&ctx) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                readback.destroy(&ctx);
                workspace.destroy(&ctx);
                return Err(error.context("create causal-block batched head pipeline"));
            }
        };
        let (command_pool, command) = match allocate_terminal_command(&ctx) {
            Ok(resources) => resources,
            Err(error) => {
                head_pipeline.destroy(&ctx);
                readback.destroy(&ctx);
                workspace.destroy(&ctx);
                return Err(error);
            }
        };
        Ok(Self {
            ctx,
            pool,
            command_pool,
            command,
            workspace: Some(workspace),
            readback: Some(readback),
            head_pipeline: Some(head_pipeline),
        })
    }

    pub fn record_gpu_export(
        &mut self,
        input: S14CausalBlockTerminalInputBinding<'_>,
    ) -> Result<S14CausalBlockTerminalGpuExport> {
        validate_terminal_input(&self.pool, &input)?;
        let block_size = input.final_hidden.block_size;
        let shape = S14HeadChunkArgmaxShape::production_batched(block_size as u32)?;
        let layout = TerminalWorkspaceLayout::build(shape)?;
        let workspace_buffer = self
            .workspace
            .as_ref()
            .context("causal-block terminal workspace 已销毁")?;
        let readback = self
            .readback
            .as_ref()
            .context("causal-block terminal readback 已销毁")?;
        let head_pipeline = self
            .head_pipeline
            .as_ref()
            .context("causal-block terminal head pipeline 已销毁")?;
        let workspace = S14HeadChunkWorkspace::new(
            workspace_buffer,
            layout.normalized_offset,
            layout.logits_offset,
            layout.accumulator_offset,
        );
        let mut dispatches =
            Vec::<S14HeadChunkArgmaxDispatch<'_>>::with_capacity(shape.chunk_count() as usize);
        for (chunk, &weight) in input.head_chunks.iter().enumerate() {
            match head_pipeline.bind_chunk(&self.ctx, shape, chunk as u32, weight, workspace) {
                Ok(dispatch) => dispatches.push(dispatch),
                Err(error) => {
                    destroy_dispatches(&self.ctx, dispatches);
                    return Err(error.context("bind causal-block production batched head chunk"));
                }
            }
        }

        let reservation = self.pool.acquire(block_size)?;
        let arena_offset = reservation.arena_offset()?;
        let ready_value = self.pool.next_ready_value()?;
        unsafe {
            self.ctx
                .device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
            self.ctx.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.ctx.device.cmd_copy_buffer(
                self.command,
                input.normalized_head_rows.buffer.handle(),
                workspace_buffer.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(input.normalized_head_rows.offset)
                    .dst_offset(layout.normalized_offset)
                    .size(shape.normalized_input_bytes()?)],
            );
            for (lane, source) in input.checkpoints.iter().enumerate() {
                let destination = arena_offset
                    .checked_add(self.pool.layout.checkpoint_stride_bytes * lane as u64)
                    .context("causal-block checkpoint destination overflow")?;
                self.ctx.device.cmd_copy_buffer(
                    self.command,
                    source.state.buffer.handle(),
                    self.pool.arena.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(source.state.offset)
                        .dst_offset(destination)
                        .size(self.pool.layout.checkpoint_state_bytes)],
                );
            }
            transfer_to_compute_barrier(&self.ctx, self.command, workspace_buffer, layout, shape);

            let mut head_recorder = S14HeadChunkArgmaxRecorder::new(shape)?;
            head_recorder.cmd_reset(&self.ctx, self.command, &dispatches[0])?;
            for dispatch in &dispatches {
                head_recorder.cmd_chunk(&self.ctx, self.command, head_pipeline, dispatch)?;
            }
            let head_recording = head_recorder.finish_recording()?;
            compute_to_transfer_barrier(&self.ctx, self.command, workspace_buffer, layout, shape);
            self.ctx.device.cmd_copy_buffer(
                self.command,
                workspace_buffer.handle(),
                readback.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(layout.accumulator_offset)
                    .size(shape.argmax_bytes()?)],
            );
            self.ctx.device.end_command_buffer(self.command)?;
            if let Err(error) = submit_terminal(
                &self.ctx,
                self.command,
                input.producer_timeline,
                input.producer_timeline_value,
                self.pool.ready_timeline,
                ready_value,
            ) {
                destroy_dispatches(&self.ctx, dispatches);
                return Err(error);
            }
            if let Err(error) = wait_timeline(
                &self.ctx,
                self.pool.ready_timeline,
                ready_value,
                TERMINAL_WAIT_TIMEOUT_NS,
            ) {
                // 已提交 command 的 descriptor/source/arena 不能提前释放。等待失败时以
                // queue idle 作为错误路径 drain，不把半完成 future 发布给 orchestrator。
                let drain = self.ctx.device.queue_wait_idle(self.ctx.q_graphics);
                destroy_dispatches(&self.ctx, dispatches);
                if let Err(drain_error) = drain {
                    return Err(anyhow::anyhow!(
                        "causal-block terminal timeline wait失败: {error}; queue drain失败: {drain_error}"
                    ));
                }
                return Err(error);
            }
            destroy_dispatches(&self.ctx, dispatches);

            let word_count = block_size
                .checked_mul(S14_HEAD_ARGMAX_WORDS)
                .context("causal-block head readback word count overflow")?;
            let words = std::slice::from_raw_parts(readback.mapped().cast::<u32>(), word_count);
            let head_results = decode_batched_head_argmax(&head_recording, words)?;
            let receipt = S14CausalBlockDeviceFutureReceipt {
                base_position: input.base_position,
                block_size,
                checkpoint_count: block_size,
                storage: S14CausalBlockDeviceCheckpointStorage::PrefixCheckpoints,
                checkpoint_arena: self.pool.arena.handle(),
                checkpoint_arena_offset: arena_offset,
                checkpoint_arena_bytes: self.pool.layout.used_bytes(block_size)?,
                checkpoint_stride_bytes: self.pool.layout.checkpoint_stride_bytes,
                checkpoint_state_bytes: self.pool.layout.checkpoint_state_bytes,
                final_hidden: input.final_hidden,
                ready_timeline: self.pool.ready_timeline,
                ready_timeline_value: ready_value,
            };
            let device_future = reservation.commit(receipt)?;
            Ok(S14CausalBlockTerminalGpuExport {
                base_position: input.base_position,
                block_size,
                final_hidden: input.final_hidden,
                head_recording,
                head_results,
                device_future,
                ready_timeline_value: ready_value,
            })
        }
    }
}

impl Drop for S14CausalBlockBatchedTerminalRecorder {
    fn drop(&mut self) {
        unsafe {
            let _ = self.ctx.device.queue_wait_idle(self.ctx.q_graphics);
        }
        if let Some(pipeline) = self.head_pipeline.take() {
            pipeline.destroy(&self.ctx);
        }
        if let Some(readback) = self.readback.take() {
            readback.destroy(&self.ctx);
        }
        if let Some(workspace) = self.workspace.take() {
            workspace.destroy(&self.ctx);
        }
        unsafe {
            self.ctx
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

fn validate_terminal_input(
    pool: &S14CausalBlockCheckpointArenaPool,
    input: &S14CausalBlockTerminalInputBinding<'_>,
) -> Result<()> {
    let block_size = input.final_hidden.block_size;
    let shape = S14HeadChunkArgmaxShape::production_batched(block_size as u32)?;
    let expected_hidden_bytes = (block_size as u64)
        .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64)
        .and_then(|elements| elements.checked_mul(2))
        .context("causal-block terminal hidden bytes overflow")?;
    if !matches!(block_size, 4 | 8)
        || input.final_hidden.buffer == vk::Buffer::null()
        || input.final_hidden.bytes != expected_hidden_bytes
        || input.checkpoints.len() != block_size
        || input.head_chunks.len() != shape.chunk_count() as usize
        || input.producer_timeline == vk::Semaphore::null()
        || input.producer_timeline_value == 0
        || input.producer_timeline == pool.ready_timeline
    {
        bail!("causal-block terminal K/hidden/checkpoint/head/timeline binding 不完整");
    }
    input
        .base_position
        .checked_add(block_size as u32)
        .context("causal-block terminal position overflow")?;
    validate_slice(
        input.normalized_head_rows,
        shape.normalized_input_bytes()?,
        "batched normalized head rows",
    )?;
    for (lane, source) in input.checkpoints.iter().enumerate() {
        validate_slice(
            source.state,
            pool.layout.checkpoint_state_bytes,
            &format!("checkpoint lane {lane}"),
        )?;
        if source.state.buffer.handle() == pool.arena.handle() {
            bail!("causal-block checkpoint source 不能 alias lease arena");
        }
    }
    for (chunk, &weight) in input.head_chunks.iter().enumerate() {
        validate_slice(
            weight,
            shape.chunk(chunk as u32)?.weight_bytes(shape)?,
            &format!("head chunk {chunk}"),
        )?;
    }
    Ok(())
}

fn validate_slice(slice: StorageBufferSlice<'_>, bytes: u64, name: &str) -> Result<()> {
    if slice.buffer.handle() == vk::Buffer::null()
        || slice.offset % 4 != 0
        || slice
            .offset
            .checked_add(bytes)
            .is_none_or(|end| end > slice.buffer.size())
    {
        bail!("causal-block terminal {name} slice 越界/未对齐");
    }
    Ok(())
}

fn allocate_terminal_command(ctx: &VulkanContext) -> Result<(vk::CommandPool, vk::CommandBuffer)> {
    unsafe {
        let pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let command = match ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        ) {
            Ok(commands) => commands[0],
            Err(error) => {
                ctx.device.destroy_command_pool(pool, None);
                return Err(error.into());
            }
        };
        Ok((pool, command))
    }
}

unsafe fn transfer_to_compute_barrier(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    workspace: &GpuBuffer,
    layout: TerminalWorkspaceLayout,
    shape: S14HeadChunkArgmaxShape,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(workspace.handle())
        .offset(layout.normalized_offset)
        .size(shape.normalized_input_bytes().unwrap_or(vk::WHOLE_SIZE));
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[barrier],
        &[],
    );
}

unsafe fn compute_to_transfer_barrier(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    workspace: &GpuBuffer,
    layout: TerminalWorkspaceLayout,
    shape: S14HeadChunkArgmaxShape,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .buffer(workspace.handle())
        .offset(layout.accumulator_offset)
        .size(shape.argmax_bytes().unwrap_or(vk::WHOLE_SIZE));
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[barrier],
        &[],
    );
}

unsafe fn submit_terminal(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    producer_timeline: vk::Semaphore,
    producer_value: u64,
    ready_timeline: vk::Semaphore,
    ready_value: u64,
) -> Result<()> {
    let commands = [command];
    let waits = [producer_timeline];
    let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
    let signals = [ready_timeline];
    let wait_values = [producer_value];
    let signal_values = [ready_value];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_values)
        .signal_semaphore_values(&signal_values);
    let submit = vk::SubmitInfo::default()
        .command_buffers(&commands)
        .wait_semaphores(&waits)
        .wait_dst_stage_mask(&wait_stages)
        .signal_semaphores(&signals)
        .push_next(&mut timeline);
    ctx.device
        .queue_submit(ctx.q_graphics, &[submit], vk::Fence::null())?;
    Ok(())
}

fn wait_timeline(
    ctx: &VulkanContext,
    timeline: vk::Semaphore,
    value: u64,
    timeout_ns: u64,
) -> Result<()> {
    let semaphores = [timeline];
    let values = [value];
    let wait = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    unsafe { ctx.device.wait_semaphores(&wait, timeout_ns)? };
    Ok(())
}

fn destroy_dispatches(ctx: &VulkanContext, dispatches: Vec<S14HeadChunkArgmaxDispatch<'_>>) {
    for dispatch in dispatches {
        dispatch.destroy(ctx);
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let mask = alignment
        .checked_sub(1)
        .context("causal-block terminal alignment 不能为0")?;
    value
        .checked_add(mask)
        .map(|sum| sum & !mask)
        .context("causal-block terminal align overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn k_prefix_layout_and_lease_ledger_reject_stale_release() {
        let layout = S14CausalBlockCheckpointArenaLayout::build(46_000_003, 2).unwrap();
        assert_eq!(layout.checkpoint_stride_bytes % CHECKPOINT_ALIGNMENT, 0);
        assert!(layout.used_bytes(4).unwrap() <= layout.slot_bytes);
        assert!(layout.used_bytes(8).unwrap() <= layout.slot_bytes);

        let mut ledger = LeaseLedger::new(2);
        let (lease_id, slot) = ledger.reserve(8).unwrap();
        let receipt = S14CausalBlockDeviceFutureReceipt {
            base_position: 17,
            block_size: 8,
            checkpoint_count: 8,
            storage: S14CausalBlockDeviceCheckpointStorage::PrefixCheckpoints,
            checkpoint_arena: vk::Buffer::from_raw(1),
            checkpoint_arena_offset: layout.slot_offset(slot).unwrap(),
            checkpoint_arena_bytes: layout.used_bytes(8).unwrap(),
            checkpoint_stride_bytes: layout.checkpoint_stride_bytes,
            checkpoint_state_bytes: layout.checkpoint_state_bytes,
            final_hidden: S14CausalBlockHiddenBinding {
                buffer: vk::Buffer::from_raw(2),
                offset: 0,
                bytes: 8 * S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64 * 2,
                block_size: 8,
                generation: 86,
            },
            ready_timeline: vk::Semaphore::from_raw(3),
            ready_timeline_value: 1,
        };
        ledger.commit(lease_id, receipt).unwrap();
        let mut stale = receipt;
        stale.ready_timeline_value = 2;
        ledger.release(lease_id, stale);
        assert_eq!(ledger.telemetry.in_use, 1);
        assert_eq!(ledger.telemetry.rejected_releases, 1);
        ledger.release(lease_id, receipt);
        assert_eq!(ledger.telemetry.in_use, 0);
        assert_eq!(ledger.telemetry.free, 2);
        assert_eq!(ledger.telemetry.releases, 1);
    }
}
