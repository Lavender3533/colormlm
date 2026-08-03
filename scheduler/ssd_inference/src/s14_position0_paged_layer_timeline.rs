//! FullDepth43 position0 分页候选的 fail-closed timeline 编排。
//!
//! 43 层只负责建立 layer tickets；L42 后调用 `seal_layers` 不做 host wait，
//! 同一 compute timeline 仍可继续承载 final compute-only、head transfer/compute
//! 与最终归约。manifest replay 成功路径只在 `finish_candidate` 等待最终
//! compute；online route 模式另外在每层 router 后等待仅48-byte probe，以便物化
//! 当层真实专家页。错误路径使用
//! `drain_all` 在一次 `vkWaitSemaphores` 中联合排空 transfer/compute，包括已经
//! 提交 transfer、尚未提交 compute 的 orphan。

use crate::{
    s14_dual_queue_timeline::{S14DualQueueTimeline, S14LayerTicket, S14TimelineDrainReceipt},
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::FULL_DEPTH_LAYERS;

pub const S14_POSITION0_PAGED_LAYER_BANKS: usize = 2;
pub const S14_PRODUCTION_PAGED_MAX_POSITION: u32 = 2051;

/// timeline/事务所有权已经与 ratio4/128 边界解耦。position128 是首个
/// 环形 window 覆盖位置；position129..254复用同一已验证逻辑，position255
/// 同轮完成ratio4 block63与ratio128 block1。后续公式推进到固定ratio4 cache
/// 已有512块的最后可消费位置2050；position2051在分页history中追加第513块。
/// timeline只承诺该位置的candidate/commit编排边界；position2052+继续fail-closed。
pub fn validate_production_paged_position(position: u32) -> Result<()> {
    if position > S14_PRODUCTION_PAGED_MAX_POSITION {
        bail!(
            "production paged whole-token timeline 当前闭合到 position{}；position{position} fail-closed",
            S14_PRODUCTION_PAGED_MAX_POSITION
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0PagedLayerTimelineState {
    Open,
    TailOpen,
    FinalSubmitted,
    Poisoned,
    Drained,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0PendingTransferKind {
    Layer { index: usize, layer: u8 },
    Head { chunk: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0PendingTransfer {
    pub kind: S14Position0PendingTransferKind,
    pub bank: usize,
    pub transfer_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0PagedLayerTimelineStats {
    pub prologue_compute_value: Option<u64>,
    pub submitted_layers: usize,
    pub submitted_head_chunks: u64,
    pub tail_compute_segments: u64,
    pub producer_transfer_waits: u64,
    pub device_bank_reuse_waits: u64,
    pub token_host_waits: u64,
    pub drain_host_waits: u64,
    pub router_probe_host_waits: u64,
    pub last_transfer_value: u64,
    pub last_compute_value: u64,
    pub pending_transfer: bool,
    pub pending_router_probe: bool,
    pub state: S14Position0PagedLayerTimelineState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0PagedCandidateReceipt {
    pub prologue_compute_value: u64,
    pub layers: usize,
    pub layer_final_compute_value: u64,
    pub final_compute_value: u64,
    pub head_chunks: u64,
    pub tail_compute_segments: u64,
    pub producer_transfer_waits: u64,
    pub device_bank_reuse_waits: u64,
    pub token_host_waits: u64,
    pub completed_transfer_value: u64,
    pub completed_compute_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0PagedDrainReceipt {
    pub transfer_value: u64,
    pub compute_value: u64,
    pub host_wait_calls: u32,
    pub orphan_transfer: bool,
}

/// 只有 graphics queue fence 已完成后才能签发的 router probe 强回执。
///
/// probe 不递增 token compute timeline；它在同一 graphics queue 中位于上一层
/// continuation 之后，因而 fence 同时证明上一层已完成且本层的
/// IDs/weights 可以被主机读取。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0RouterProbeCompletionReceipt {
    pub layer: u8,
    pub index: usize,
    pub bank: usize,
    pub preceding_compute_value: u64,
    pub host_wait_calls: u32,
}

#[derive(Debug)]
struct PagedCandidateProgress {
    state: S14Position0PagedLayerTimelineState,
    layer_tickets: Vec<S14LayerTicket>,
    prologue_compute_value: Option<u64>,
    pending: Option<S14Position0PendingTransfer>,
    pending_router_probe: Option<S14Position0RouterProbeCompletionReceipt>,
    head_chunks: u64,
    tail_compute_segments: u64,
    final_compute_value: Option<u64>,
    producer_transfer_waits: u64,
    device_bank_reuse_waits: u64,
    token_host_waits: u64,
    drain_host_waits: u64,
    router_probe_host_waits: u64,
}

impl PagedCandidateProgress {
    fn new() -> Self {
        Self {
            state: S14Position0PagedLayerTimelineState::Open,
            layer_tickets: Vec::with_capacity(FULL_DEPTH_LAYERS.len()),
            prologue_compute_value: None,
            pending: None,
            pending_router_probe: None,
            head_chunks: 0,
            tail_compute_segments: 0,
            final_compute_value: None,
            producer_transfer_waits: 0,
            device_bank_reuse_waits: 0,
            token_host_waits: 0,
            drain_host_waits: 0,
            router_probe_host_waits: 0,
        }
    }

    fn expected_layer_identity(&self) -> Result<(usize, u8)> {
        if self.state != S14Position0PagedLayerTimelineState::Open
            || self.pending.is_some()
            || self.prologue_compute_value.is_none()
        {
            bail!("position0 paged layer timeline 不能开始下一层");
        }
        let index = self.layer_tickets.len();
        let layer = FULL_DEPTH_LAYERS
            .get(index)
            .copied()
            .ok_or_else(|| anyhow!("position0 paged 43 层已完成，必须先 seal_layers"))?;
        Ok((index, layer))
    }

    fn expected_layer(&self) -> Result<u8> {
        if self.pending_router_probe.is_some() {
            bail!("position0 online router probe 尚未绑定 continuation");
        }
        Ok(self.expected_layer_identity()?.1)
    }

    fn layer_bank(&self) -> usize {
        self.layer_tickets.len() % S14_POSITION0_PAGED_LAYER_BANKS
    }

    fn layer_reuse_after_compute(&self) -> Option<u64> {
        self.layer_tickets
            .len()
            .checked_sub(S14_POSITION0_PAGED_LAYER_BANKS)
            .and_then(|index| self.layer_tickets.get(index))
            .map(|ticket| ticket.compute_value)
    }

    fn poison(&mut self) {
        if matches!(
            self.state,
            S14Position0PagedLayerTimelineState::Open
                | S14Position0PagedLayerTimelineState::TailOpen
                | S14Position0PagedLayerTimelineState::FinalSubmitted
        ) {
            self.state = S14Position0PagedLayerTimelineState::Poisoned;
        }
    }
}

pub struct S14Position0PagedLayerTimeline {
    position: u32,
    timeline: S14DualQueueTimeline,
    progress: PagedCandidateProgress,
    layer_staging_last_transfer: [Option<u64>; S14_POSITION0_PAGED_LAYER_BANKS],
    head_staging_last_transfer: [Option<u64>; S14_POSITION0_PAGED_LAYER_BANKS],
    head_last_compute: [Option<u64>; S14_POSITION0_PAGED_LAYER_BANKS],
    router_probe_fence: vk::Fence,
    /// `queue_submit` 已成功、但对应 fence wait 尚未成功返回。正常 probe wait 后立即
    /// 清零；错误 drain 必须额外收敛它，不能只等待 dual timeline 的旧 compute 值。
    router_probe_fence_pending: bool,
}

impl S14Position0PagedLayerTimeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Self::new_for_position(ctx, 0)
    }

    pub fn new_for_position(ctx: &VulkanContext, position: u32) -> Result<Self> {
        validate_production_paged_position(position)?;
        let timeline = S14DualQueueTimeline::new(ctx)?;
        let router_probe_fence = match unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                timeline.destroy(ctx);
                return Err(error.into());
            }
        };
        Ok(Self {
            position,
            timeline,
            progress: PagedCandidateProgress::new(),
            layer_staging_last_transfer: [None; S14_POSITION0_PAGED_LAYER_BANKS],
            head_staging_last_transfer: [None; S14_POSITION0_PAGED_LAYER_BANKS],
            head_last_compute: [None; S14_POSITION0_PAGED_LAYER_BANKS],
            router_probe_fence,
            router_probe_fence_pending: false,
        })
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn stats(&self) -> S14Position0PagedLayerTimelineStats {
        S14Position0PagedLayerTimelineStats {
            prologue_compute_value: self.progress.prologue_compute_value,
            submitted_layers: self.progress.layer_tickets.len(),
            submitted_head_chunks: self.progress.head_chunks,
            tail_compute_segments: self.progress.tail_compute_segments,
            producer_transfer_waits: self.progress.producer_transfer_waits,
            device_bank_reuse_waits: self.progress.device_bank_reuse_waits,
            token_host_waits: self.progress.token_host_waits,
            drain_host_waits: self.progress.drain_host_waits,
            router_probe_host_waits: self.progress.router_probe_host_waits,
            last_transfer_value: self.timeline.last_transfer_value(),
            last_compute_value: self.timeline.last_compute_value(),
            pending_transfer: self.progress.pending.is_some(),
            pending_router_probe: self.progress.pending_router_probe.is_some()
                || self.router_probe_fence_pending,
            state: self.progress.state,
        }
    }

    /// 提交 device candidate prologue + embedding 首段。它必须是本 candidate 的第一个
    /// compute timeline signal；后续43层与 final tail 都沿用同一 timeline。
    ///
    /// # Safety
    ///
    /// command 已结束录制，且只写 inactive candidate/本 token workspace；资源至少存活到
    /// `finish_candidate` 或 `drain_all`。
    pub unsafe fn submit_prologue_compute_only(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<u64> {
        if self.progress.state != S14Position0PagedLayerTimelineState::Open
            || self.progress.pending.is_some()
            || !self.progress.layer_tickets.is_empty()
            || self.progress.prologue_compute_value.is_some()
            || self.timeline.last_compute_value() != 0
        {
            return self.poison_error(anyhow!("position0 prologue compute phase 漂移"));
        }
        let value = match self.timeline.submit_compute_only(ctx, command) {
            Ok(value) => value,
            Err(error) => return self.poison_error(error.context("submit prologue embedding")),
        };
        self.progress.prologue_compute_value = Some(value);
        Ok(value)
    }

    /// 提交当前层的 router-only command 并等待它完成。返回值是后续
    /// `complete_router_probe_after_wait` 的强完成依据，不允许在 matching layer
    /// continuation 进入层 timeline 前开始下一个 probe。
    ///
    /// # Safety
    ///
    /// `command` 已结束录制，引用资源至少存活到本调用返回。
    pub unsafe fn submit_router_probe_and_wait(
        &mut self,
        ctx: &VulkanContext,
        layer: u8,
        command: vk::CommandBuffer,
    ) -> Result<S14Position0RouterProbeCompletionReceipt> {
        let (index, expected) = match self.progress.expected_layer_identity() {
            Ok(identity) => identity,
            Err(error) => return self.poison_error(error),
        };
        if self.progress.pending_router_probe.is_some()
            || self.router_probe_fence_pending
            || layer != expected
        {
            return self.poison_error(anyhow!(
                "position0 router probe 顺序/phase 漂移: expected=L{expected} actual=L{layer}"
            ));
        }
        if let Err(error) = ctx.device.reset_fences(&[self.router_probe_fence]) {
            return self.poison_error(error.into());
        }
        let commands = [command];
        if let Err(error) = ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            self.router_probe_fence,
        ) {
            return self.poison_error(error.into());
        }
        self.router_probe_fence_pending = true;
        self.progress.router_probe_host_waits += 1;
        if let Err(error) = ctx
            .device
            .wait_for_fences(&[self.router_probe_fence], true, u64::MAX)
        {
            return self.poison_error(error.into());
        }
        self.router_probe_fence_pending = false;
        let receipt = S14Position0RouterProbeCompletionReceipt {
            layer,
            index,
            bank: self.progress.layer_bank(),
            preceding_compute_value: self.timeline.last_compute_value(),
            host_wait_calls: 1,
        };
        self.progress.pending_router_probe = Some(receipt);
        Ok(receipt)
    }

    /// 只提交当前层 transfer，并保留 pending。该两阶段接口使调用者能在 compute
    /// 录制/提交失败后显式 `abort_pending`，再由 `drain_all` 一次性收敛 orphan。
    ///
    /// # Safety
    ///
    /// command 已结束录制，引用资源至少存活到 pending compute 完成或 drain 完成。
    pub unsafe fn submit_next_layer_transfer<F>(
        &mut self,
        ctx: &VulkanContext,
        layer: u8,
        transfer_command: vk::CommandBuffer,
        stage: F,
    ) -> Result<S14Position0PendingTransfer>
    where
        F: FnOnce(usize) -> Result<()>,
    {
        let router_probe = self.progress.pending_router_probe;
        let expected = match router_probe {
            Some(receipt)
                if receipt.index == self.progress.layer_tickets.len()
                    && receipt.bank == self.progress.layer_bank() =>
            {
                receipt.layer
            }
            Some(_) => {
                return self.poison_error(anyhow!(
                    "position0 router probe completion receipt 层/index/bank 漂移"
                ));
            }
            None => match self.progress.expected_layer() {
                Ok(layer) => layer,
                Err(error) => return self.poison_error(error),
            },
        };
        if layer != expected {
            return self.poison_error(anyhow!(
                "position0 paged layer 顺序漂移: expected=L{expected} actual=L{layer}"
            ));
        }
        let index = self.progress.layer_tickets.len();
        let bank = self.progress.layer_bank();
        if router_probe.is_none() {
            if let Some(value) = self.layer_staging_last_transfer[bank] {
                if let Err(error) = self.timeline.wait_transfer(ctx, value, u64::MAX) {
                    return self.poison_error(error.context("wait paged layer staging transfer"));
                }
                self.progress.producer_transfer_waits += 1;
            }
        }
        if let Err(error) = stage(bank) {
            return self.poison_error(error.context(format!("stage real paged L{layer}")));
        }
        let reuse_after_compute = self.progress.layer_reuse_after_compute();
        let transfer_value =
            match self
                .timeline
                .submit_transfer(ctx, transfer_command, reuse_after_compute)
            {
                Ok(value) => value,
                Err(error) => {
                    return self
                        .poison_error(error.context(format!("submit paged transfer L{layer}")));
                }
            };
        self.layer_staging_last_transfer[bank] = Some(transfer_value);
        if reuse_after_compute.is_some() {
            self.progress.device_bank_reuse_waits += 1;
        }
        let pending = S14Position0PendingTransfer {
            kind: S14Position0PendingTransferKind::Layer { index, layer },
            bank,
            transfer_value,
        };
        self.progress.pending = Some(pending);
        self.progress.pending_router_probe = None;
        Ok(pending)
    }

    /// # Safety
    ///
    /// compute command 已结束录制，资源至少存活到最终 wait/drain。
    pub unsafe fn submit_pending_layer_compute(
        &mut self,
        ctx: &VulkanContext,
        pending: S14Position0PendingTransfer,
        compute_command: vk::CommandBuffer,
    ) -> Result<S14LayerTicket> {
        if self.progress.state != S14Position0PagedLayerTimelineState::Open
            || self.progress.pending != Some(pending)
        {
            return self.poison_error(anyhow!("position0 pending layer transfer ticket 漂移"));
        }
        let (index, layer) = match pending.kind {
            S14Position0PendingTransferKind::Layer { index, layer } => (index, layer),
            S14Position0PendingTransferKind::Head { .. } => {
                return self.poison_error(anyhow!("head pending 不能提交为 layer compute"));
            }
        };
        if index != self.progress.layer_tickets.len() || layer != FULL_DEPTH_LAYERS[index] {
            return self.poison_error(anyhow!("position0 pending layer index/order 漂移"));
        }
        let compute_value =
            match self
                .timeline
                .submit_compute(ctx, compute_command, pending.transfer_value)
            {
                Ok(value) => value,
                Err(error) => {
                    return self
                        .poison_error(error.context(format!("submit paged compute L{layer}")));
                }
            };
        let ticket = S14LayerTicket {
            transfer_value: pending.transfer_value,
            compute_value,
        };
        self.progress.layer_tickets.push(ticket);
        self.progress.pending = None;
        Ok(ticket)
    }

    /// 两阶段 layer API 的正常组合。
    ///
    /// # Safety
    ///
    /// 两个 command 均已结束录制，资源至少存活到最终 wait/drain。
    pub unsafe fn stage_and_submit_next<F>(
        &mut self,
        ctx: &VulkanContext,
        layer: u8,
        transfer_command: vk::CommandBuffer,
        compute_command: vk::CommandBuffer,
        stage: F,
    ) -> Result<S14LayerTicket>
    where
        F: FnOnce(usize) -> Result<()>,
    {
        let pending = self.submit_next_layer_transfer(ctx, layer, transfer_command, stage)?;
        self.submit_pending_layer_compute(ctx, pending, compute_command)
    }

    /// 43 层封口，不等待 GPU；final tail 必须继续使用同一 timeline。
    pub fn seal_layers(&mut self) -> Result<u64> {
        if self.progress.state != S14Position0PagedLayerTimelineState::Open
            || self.progress.pending.is_some()
            || self.progress.pending_router_probe.is_some()
            || self.progress.layer_tickets.len() != FULL_DEPTH_LAYERS.len()
            || self.progress.prologue_compute_value.is_none()
        {
            return self.poison_error(anyhow!(
                "position0 layer seal 要求完整 43 层且无 pending transfer"
            ));
        }
        let layer_final_compute = self.progress.layer_tickets[42].compute_value;
        self.progress.state = S14Position0PagedLayerTimelineState::TailOpen;
        Ok(layer_final_compute)
    }

    /// final HC、最终归约等不依赖新 transfer 的段。
    ///
    /// # Safety
    ///
    /// command 已结束录制，资源至少存活到最终 wait/drain。
    pub unsafe fn submit_tail_compute_only(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<u64> {
        if self.progress.state != S14Position0PagedLayerTimelineState::TailOpen
            || self.progress.pending.is_some()
        {
            return self.poison_error(anyhow!("position0 final compute-only phase 漂移"));
        }
        let value = match self.timeline.submit_compute_only(ctx, command) {
            Ok(value) => value,
            Err(error) => return self.poison_error(error.context("submit final compute-only")),
        };
        self.progress.tail_compute_segments += 1;
        Ok(value)
    }

    /// 提交候选最后一个 compute 段（生产中必须包含最终 argmax/token readback），
    /// 并永久封口 tail。成功后禁止继续提交 head 或普通 tail compute。
    ///
    /// # Safety
    ///
    /// command 已结束录制，资源至少存活到 `finish_candidate`/`drain_all`。
    pub unsafe fn submit_final_compute(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<u64> {
        if self.progress.state != S14Position0PagedLayerTimelineState::TailOpen
            || self.progress.pending.is_some()
            || self.progress.final_compute_value.is_some()
        {
            return self.poison_error(anyhow!("position0 final compute phase 漂移"));
        }
        let value = match self.timeline.submit_compute_only(ctx, command) {
            Ok(value) => value,
            Err(error) => {
                return self.poison_error(error.context("submit terminal argmax compute"));
            }
        };
        self.progress.tail_compute_segments += 1;
        self.progress.final_compute_value = Some(value);
        self.progress.state = S14Position0PagedLayerTimelineState::FinalSubmitted;
        Ok(value)
    }

    /// 两阶段 head transfer。调用顺序必须为 chunk0,1,...；双 bank staging 与设备页
    /// 分别等待各自上一次 transfer/compute。
    ///
    /// # Safety
    ///
    /// command 已结束录制，资源至少存活到 pending compute 或 drain 完成。
    pub unsafe fn submit_next_head_transfer<F>(
        &mut self,
        ctx: &VulkanContext,
        chunk: u64,
        transfer_command: vk::CommandBuffer,
        stage: F,
    ) -> Result<S14Position0PendingTransfer>
    where
        F: FnOnce(usize) -> Result<()>,
    {
        if self.progress.state != S14Position0PagedLayerTimelineState::TailOpen
            || self.progress.pending.is_some()
            || chunk != self.progress.head_chunks
        {
            return self.poison_error(anyhow!(
                "position0 head chunk 顺序/phase 漂移: expected={} actual={chunk}",
                self.progress.head_chunks
            ));
        }
        let bank = chunk as usize % S14_POSITION0_PAGED_LAYER_BANKS;
        if let Some(value) = self.head_staging_last_transfer[bank] {
            if let Err(error) = self.timeline.wait_transfer(ctx, value, u64::MAX) {
                return self.poison_error(error.context("wait head staging transfer"));
            }
            self.progress.producer_transfer_waits += 1;
        }
        if let Err(error) = stage(bank) {
            return self.poison_error(error.context(format!("stage real head chunk {chunk}")));
        }
        let reuse_after_compute = self.head_last_compute[bank];
        let transfer_value =
            match self
                .timeline
                .submit_transfer(ctx, transfer_command, reuse_after_compute)
            {
                Ok(value) => value,
                Err(error) => {
                    return self
                        .poison_error(error.context(format!("submit head transfer {chunk}")));
                }
            };
        self.head_staging_last_transfer[bank] = Some(transfer_value);
        if reuse_after_compute.is_some() {
            self.progress.device_bank_reuse_waits += 1;
        }
        let pending = S14Position0PendingTransfer {
            kind: S14Position0PendingTransferKind::Head { chunk },
            bank,
            transfer_value,
        };
        self.progress.pending = Some(pending);
        Ok(pending)
    }

    /// # Safety
    ///
    /// compute command 已结束录制，资源至少存活到最终 wait/drain。
    pub unsafe fn submit_pending_head_compute(
        &mut self,
        ctx: &VulkanContext,
        pending: S14Position0PendingTransfer,
        compute_command: vk::CommandBuffer,
    ) -> Result<S14LayerTicket> {
        if self.progress.state != S14Position0PagedLayerTimelineState::TailOpen
            || self.progress.pending != Some(pending)
        {
            return self.poison_error(anyhow!("position0 pending head transfer ticket 漂移"));
        }
        let chunk = match pending.kind {
            S14Position0PendingTransferKind::Head { chunk } => chunk,
            S14Position0PendingTransferKind::Layer { .. } => {
                return self.poison_error(anyhow!("layer pending 不能提交为 head compute"));
            }
        };
        if chunk != self.progress.head_chunks {
            return self.poison_error(anyhow!("position0 pending head chunk 漂移"));
        }
        let compute_value =
            match self
                .timeline
                .submit_compute(ctx, compute_command, pending.transfer_value)
            {
                Ok(value) => value,
                Err(error) => {
                    return self
                        .poison_error(error.context(format!("submit head compute {chunk}")));
                }
            };
        self.head_last_compute[pending.bank] = Some(compute_value);
        self.progress.head_chunks += 1;
        self.progress.pending = None;
        Ok(S14LayerTicket {
            transfer_value: pending.transfer_value,
            compute_value,
        })
    }

    /// # Safety
    ///
    /// 两个 command 均已结束录制，资源至少存活到最终 wait/drain。
    pub unsafe fn stage_and_submit_head<F>(
        &mut self,
        ctx: &VulkanContext,
        chunk: u64,
        transfer_command: vk::CommandBuffer,
        compute_command: vk::CommandBuffer,
        stage: F,
    ) -> Result<S14LayerTicket>
    where
        F: FnOnce(usize) -> Result<()>,
    {
        let pending = self.submit_next_head_transfer(ctx, chunk, transfer_command, stage)?;
        self.submit_pending_head_compute(ctx, pending, compute_command)
    }

    /// 标记 pending transfer 之后的主机/录制失败。不会等待，也不会签发 receipt。
    pub fn abort_pending(&mut self) -> Result<()> {
        if (self.progress.pending.is_none() && self.progress.pending_router_probe.is_none())
            || !matches!(
                self.progress.state,
                S14Position0PagedLayerTimelineState::Open
                    | S14Position0PagedLayerTimelineState::TailOpen
            )
        {
            bail!("position0 没有可 abort 的 pending transfer");
        }
        self.progress.poison();
        self.progress.pending_router_probe = None;
        Ok(())
    }

    /// 成功路径的唯一 host wait；只接受 final tail 的最后 compute ticket。
    pub fn finish_candidate(
        &mut self,
        ctx: &VulkanContext,
    ) -> Result<S14Position0PagedCandidateReceipt> {
        let final_compute_value = self.progress.final_compute_value.unwrap_or(0);
        if self.progress.state != S14Position0PagedLayerTimelineState::FinalSubmitted
            || self.progress.pending.is_some()
            || self.progress.layer_tickets.len() != FULL_DEPTH_LAYERS.len()
            || self.progress.tail_compute_segments == 0
            || final_compute_value == 0
            || final_compute_value != self.timeline.last_compute_value()
        {
            return self.poison_error(anyhow!("position0 candidate final ticket/phase 不完整"));
        }
        if let Err(error) = self
            .timeline
            .wait_compute(ctx, final_compute_value, u64::MAX)
        {
            return self.poison_error(error.context("wait final candidate compute"));
        }
        self.progress.token_host_waits += 1;
        let completed = match self.timeline.completed_values(ctx) {
            Ok(values) => values,
            Err(error) => {
                // final wait 已成功，所有资源都已收敛；标记 Drained，禁止上层再做第二次等待。
                self.progress.state = S14Position0PagedLayerTimelineState::Drained;
                return Err(error).context("read final candidate timeline");
            }
        };
        if completed.0 != self.timeline.last_transfer_value()
            || completed.1 != final_compute_value
            || self.progress.token_host_waits != 1
        {
            self.progress.state = S14Position0PagedLayerTimelineState::Drained;
            bail!("position0 final candidate completion drift: completed={completed:?}");
        }
        self.progress.state = S14Position0PagedLayerTimelineState::Finished;
        Ok(S14Position0PagedCandidateReceipt {
            prologue_compute_value: self.progress.prologue_compute_value.unwrap_or(0),
            layers: FULL_DEPTH_LAYERS.len(),
            layer_final_compute_value: self.progress.layer_tickets[42].compute_value,
            final_compute_value,
            head_chunks: self.progress.head_chunks,
            tail_compute_segments: self.progress.tail_compute_segments,
            producer_transfer_waits: self.progress.producer_transfer_waits,
            device_bank_reuse_waits: self.progress.device_bank_reuse_waits,
            token_host_waits: self.progress.token_host_waits,
            completed_transfer_value: completed.0,
            completed_compute_value: completed.1,
        })
    }

    /// 错误路径联合排空。通常只需 dual timeline 的一次 host wait；如果 router probe
    /// `queue_submit` 已成功、原 fence wait 却报错，则先重试该 fence，再联合等待
    /// transfer/compute timeline，禁止把旧 compute 值误当成 probe 完成证明。
    pub fn drain_all(&mut self, ctx: &VulkanContext) -> Result<S14Position0PagedDrainReceipt> {
        if matches!(
            self.progress.state,
            S14Position0PagedLayerTimelineState::Finished
                | S14Position0PagedLayerTimelineState::Drained
        ) {
            bail!("position0 candidate 已 finished/drained");
        }
        self.progress.poison();
        let orphan_transfer = self.progress.pending.is_some();
        let mut router_probe_recovery_waits = 0u32;
        if self.router_probe_fence_pending {
            unsafe {
                ctx.device
                    .wait_for_fences(&[self.router_probe_fence], true, u64::MAX)
                    .context("drain pending router probe fence")?;
            }
            self.router_probe_fence_pending = false;
            router_probe_recovery_waits = 1;
        }
        let S14TimelineDrainReceipt {
            transfer_value,
            compute_value,
            host_wait_calls: timeline_host_waits,
        } = self.timeline.drain_all(ctx, u64::MAX)?;
        let host_wait_calls = timeline_host_waits
            .checked_add(router_probe_recovery_waits)
            .ok_or_else(|| anyhow!("position0 drain host wait counter overflow"))?;
        self.progress.drain_host_waits += u64::from(host_wait_calls);
        self.progress.state = S14Position0PagedLayerTimelineState::Drained;
        Ok(S14Position0PagedDrainReceipt {
            transfer_value,
            compute_value,
            host_wait_calls,
            orphan_transfer,
        })
    }

    pub fn destroy(mut self, ctx: &VulkanContext) {
        if !matches!(
            self.progress.state,
            S14Position0PagedLayerTimelineState::Finished
                | S14Position0PagedLayerTimelineState::Drained
        ) && (self.timeline.last_transfer_value() != 0
            || self.timeline.last_compute_value() != 0
            || self.router_probe_fence_pending)
        {
            if self.drain_all(ctx).is_err() {
                // drain API 已把错误交给显式调用方；析构兜底不能继续释放可能仍被
                // command 引用的 semaphore/fence，因此退化为 device-wide idle。
                unsafe {
                    let _ = ctx.device.device_wait_idle();
                }
            }
        }
        self.timeline.destroy(ctx);
        unsafe {
            ctx.device.destroy_fence(self.router_probe_fence, None);
        }
    }

    fn poison_error<T>(&mut self, error: anyhow::Error) -> Result<T> {
        self.progress.poison();
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_depth_schedule_seals_without_finishing_candidate() {
        let mut progress = PagedCandidateProgress::new();
        assert!(progress.expected_layer().is_err());
        progress.prologue_compute_value = Some(1);
        for (index, &layer) in FULL_DEPTH_LAYERS.iter().enumerate() {
            assert_eq!(progress.expected_layer().unwrap(), layer);
            assert_eq!(progress.layer_bank(), index % 2);
            assert_eq!(
                progress.layer_reuse_after_compute(),
                index.checked_sub(2).map(|prior| prior as u64 + 2)
            );
            progress.layer_tickets.push(S14LayerTicket {
                transfer_value: index as u64 + 1,
                compute_value: index as u64 + 2,
            });
        }
        assert!(progress.expected_layer().is_err());
        assert_eq!(progress.layer_tickets.len(), 43);
        assert_eq!(progress.state, S14Position0PagedLayerTimelineState::Open);
    }

    #[test]
    fn poisoned_progress_never_reopens() {
        let mut progress = PagedCandidateProgress::new();
        progress.poison();
        assert_eq!(
            progress.state,
            S14Position0PagedLayerTimelineState::Poisoned
        );
        assert!(progress.expected_layer().is_err());
    }

    #[test]
    fn continuous_timeline_gate_allows_first_paged_ratio4_append_only() {
        validate_production_paged_position(0).unwrap();
        validate_production_paged_position(1).unwrap();
        validate_production_paged_position(2).unwrap();
        validate_production_paged_position(3).unwrap();
        validate_production_paged_position(4).unwrap();
        validate_production_paged_position(7).unwrap();
        validate_production_paged_position(15).unwrap();
        validate_production_paged_position(126).unwrap();
        validate_production_paged_position(127).unwrap();
        validate_production_paged_position(128).unwrap();
        validate_production_paged_position(129).unwrap();
        validate_production_paged_position(254).unwrap();
        validate_production_paged_position(255).unwrap();
        validate_production_paged_position(256).unwrap();
        validate_production_paged_position(2047).unwrap();
        validate_production_paged_position(2050).unwrap();
        validate_production_paged_position(2051).unwrap();
        assert!(validate_production_paged_position(2052).is_err());
        assert!(validate_production_paged_position(u32::MAX).is_err());
    }
}
