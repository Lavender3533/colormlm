//! 真实 43 层计划到 paged Vulkan timeline 的最小 fail-closed 桥。
//!
//! 桥只编排权重 staging、descriptor reconfigure 与 transfer/compute ticket；hidden
//! 始终由计划中的 GPU workspace A/B 槽承载。层间不做 host compute wait，设备页复用
//! 依赖由 `S14Position0PagedLayerTimeline` 在 transfer submit 上等待 L(n-2) compute。
//! L42 后把同一个 timeline 借给 tail，final/head 不需要新建同步域。

use crate::{
    s14_dual_queue_timeline::S14LayerTicket,
    s14_position0_paged_layer_timeline::{
        validate_production_paged_position, S14Position0PagedLayerTimeline,
        S14Position0PendingTransfer, S14Position0PendingTransferKind,
    },
    s14_position0_synchronous_layer_pager::{
        S14Position0DeviceHiddenBinding, S14Position0StaticPage, S14Position0SynchronousLayerPlan,
        S14_POSITION0_SYNCHRONOUS_BANKS, S14_POSITION0_SYNCHRONOUS_LAYER_COUNT,
    },
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0PagedLayerBridgeState {
    Open,
    Poisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0PagedLayerStageReceipt {
    pub layer: u8,
    pub bank: usize,
    pub static_uploaded_bytes: u64,
    pub routed_uploaded_bytes: u64,
    /// 必须为零。staging 回调只能读取权重 payload，不能读取 GPU hidden。
    pub hidden_host_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0PagedLayerBridgeReceipt {
    pub layer: u8,
    pub index: usize,
    pub bank: usize,
    pub stage: S14Position0PagedLayerStageReceipt,
    pub ticket: S14LayerTicket,
    pub hidden: S14Position0DeviceHiddenBinding,
}

/// 把真实 timeline 抽成可隔离测试的最小端口。生产实现仍直接调用原 Vulkan API。
pub trait S14Position0PagedLayerTimelinePort {
    type Context;
    type Command: Copy;

    /// # Safety
    ///
    /// 与 `S14Position0PagedLayerTimeline::submit_next_layer_transfer` 相同。
    unsafe fn submit_next_layer_transfer<F>(
        &mut self,
        ctx: &Self::Context,
        layer: u8,
        transfer_command: Self::Command,
        stage: F,
    ) -> Result<S14Position0PendingTransfer>
    where
        F: FnOnce(usize) -> Result<()>;

    /// # Safety
    ///
    /// 与 `S14Position0PagedLayerTimeline::submit_pending_layer_compute` 相同。
    unsafe fn submit_pending_layer_compute(
        &mut self,
        ctx: &Self::Context,
        pending: S14Position0PendingTransfer,
        compute_command: Self::Command,
    ) -> Result<S14LayerTicket>;

    fn abort_pending(&mut self) -> Result<()>;

    fn seal_layers(&mut self) -> Result<u64>;
}

impl S14Position0PagedLayerTimelinePort for S14Position0PagedLayerTimeline {
    type Context = VulkanContext;
    type Command = vk::CommandBuffer;

    unsafe fn submit_next_layer_transfer<F>(
        &mut self,
        ctx: &Self::Context,
        layer: u8,
        transfer_command: Self::Command,
        stage: F,
    ) -> Result<S14Position0PendingTransfer>
    where
        F: FnOnce(usize) -> Result<()>,
    {
        unsafe {
            S14Position0PagedLayerTimeline::submit_next_layer_transfer(
                self,
                ctx,
                layer,
                transfer_command,
                stage,
            )
        }
    }

    unsafe fn submit_pending_layer_compute(
        &mut self,
        ctx: &Self::Context,
        pending: S14Position0PendingTransfer,
        compute_command: Self::Command,
    ) -> Result<S14LayerTicket> {
        unsafe {
            S14Position0PagedLayerTimeline::submit_pending_layer_compute(
                self,
                ctx,
                pending,
                compute_command,
            )
        }
    }

    fn abort_pending(&mut self) -> Result<()> {
        S14Position0PagedLayerTimeline::abort_pending(self)
    }

    fn seal_layers(&mut self) -> Result<u64> {
        S14Position0PagedLayerTimeline::seal_layers(self)
    }
}

pub struct S14Position0PagedLayerBridge<'timeline, 'plans, T>
where
    T: S14Position0PagedLayerTimelinePort,
{
    timeline: &'timeline mut T,
    plans: &'plans [S14Position0SynchronousLayerPlan],
    position: u32,
    next_index: usize,
    state: S14Position0PagedLayerBridgeState,
}

impl<'timeline, 'plans, T> S14Position0PagedLayerBridge<'timeline, 'plans, T>
where
    T: S14Position0PagedLayerTimelinePort,
{
    pub fn new(
        timeline: &'timeline mut T,
        plans: &'plans [S14Position0SynchronousLayerPlan],
    ) -> Result<Self> {
        Self::new_for_position(timeline, plans, 0)
    }

    pub fn new_for_position(
        timeline: &'timeline mut T,
        plans: &'plans [S14Position0SynchronousLayerPlan],
        position: u32,
    ) -> Result<Self> {
        validate_production_paged_position(position)?;
        validate_full_depth_plans(plans)?;
        Ok(Self {
            timeline,
            plans,
            position,
            next_index: 0,
            state: S14Position0PagedLayerBridgeState::Open,
        })
    }

    pub fn state(&self) -> S14Position0PagedLayerBridgeState {
        self.state
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn next_layer(&self) -> Option<u8> {
        (self.state == S14Position0PagedLayerBridgeState::Open)
            .then(|| self.plans.get(self.next_index).map(|plan| plan.layer))
            .flatten()
    }

    /// 录入下一层，但不等待上一层 compute。transfer timeline 仅在复用同一 bank 时
    /// 等待 L(n-2) 的 compute ticket；graphics queue 保持 hidden 的逐层顺序。
    ///
    /// `stage` 只把计划中的权重 payload 写入当前 staging bank；`reconfigure` 只更新
    /// 当前 static/routed 页和 GPU hidden A/B descriptor。
    ///
    /// # Safety
    ///
    /// 两个 command 必须已结束录制；descriptor、staging、paged arena 和 workspace
    /// 至少存活到最终 candidate wait/drain。
    pub unsafe fn submit_next_layer<Stage, Reconfigure>(
        &mut self,
        ctx: &T::Context,
        transfer_command: T::Command,
        compute_command: T::Command,
        stage: Stage,
        reconfigure: Reconfigure,
    ) -> Result<S14Position0PagedLayerBridgeReceipt>
    where
        Stage: FnOnce(
            &S14Position0SynchronousLayerPlan,
            usize,
        ) -> Result<S14Position0PagedLayerStageReceipt>,
        Reconfigure: FnOnce(&S14Position0SynchronousLayerPlan, usize) -> Result<()>,
    {
        self.ensure_open()?;
        let plan = match self.plans.get(self.next_index) {
            Some(plan) => plan,
            None => return self.poison(anyhow!("position0 paged bridge 已提交完整 43 层")),
        };
        let expected_static = match plan.upload_request().expected_static_upload_bytes() {
            Ok(bytes) => bytes,
            Err(error) => return self.poison(error),
        };
        let expected_routed = match plan.upload_request().expected_routed_upload_bytes() {
            Ok(bytes) => bytes,
            Err(error) => return self.poison(error),
        };
        let index = self.next_index;
        let mut stage_receipt = None;
        let pending = match unsafe {
            self.timeline
                .submit_next_layer_transfer(ctx, plan.layer, transfer_command, |bank| {
                    validate_runtime_bank(plan, bank)?;
                    let receipt = stage(plan, bank)?;
                    validate_stage_receipt(plan, bank, receipt, expected_static, expected_routed)?;
                    stage_receipt = Some(receipt);
                    Ok(())
                })
        } {
            Ok(pending) => pending,
            Err(error) => {
                return self.poison(error.context(format!("stage/submit L{}", plan.layer)));
            }
        };
        if pending.kind
            != (S14Position0PendingTransferKind::Layer {
                index,
                layer: plan.layer,
            })
            || pending.bank != plan.routed_bank
        {
            let error = anyhow!("L{} paged pending ticket 身份/bank 漂移", plan.layer);
            return self.abort_pending(error);
        }
        if let Err(error) = reconfigure(plan, pending.bank) {
            return self.abort_pending(error.context(format!("reconfigure L{}", plan.layer)));
        }
        let ticket = match unsafe {
            self.timeline
                .submit_pending_layer_compute(ctx, pending, compute_command)
        } {
            Ok(ticket) => ticket,
            Err(error) => {
                return self.poison(error.context(format!("submit compute L{}", plan.layer)));
            }
        };
        if ticket.transfer_value != pending.transfer_value || ticket.compute_value == 0 {
            return self.poison(anyhow!("L{} paged compute ticket 漂移", plan.layer));
        }
        let receipt = S14Position0PagedLayerBridgeReceipt {
            layer: plan.layer,
            index,
            bank: pending.bank,
            stage: stage_receipt.expect("timeline 成功必已调用 stage"),
            ticket,
            hidden: plan.hidden,
        };
        self.next_index += 1;
        Ok(receipt)
    }

    /// Online router probe 需要在当层 continuation 进入 bridge 之前取得
    /// timeline 的强完成回执。借用不能逸出 bridge，层计数仍只能由
    /// `submit_next_layer` 推进。
    pub fn timeline_mut(&mut self) -> &mut T {
        self.timeline
    }

    /// 43 层封口并把原 timeline 借给 final/head。没有新 semaphore，也没有 host wait。
    pub fn seal_layers(mut self) -> Result<S14Position0PagedLayerTail<'timeline, T>> {
        self.ensure_open()?;
        if self.next_index != self.plans.len()
            || self.next_index != usize::from(S14_POSITION0_SYNCHRONOUS_LAYER_COUNT)
        {
            return self.poison(anyhow!(
                "position0 paged bridge seal 过早: submitted={} expected=43",
                self.next_index
            ));
        }
        let layer_final_compute_value = match self.timeline.seal_layers() {
            Ok(value) => value,
            Err(error) => return self.poison(error.context("seal paged 43 layers")),
        };
        let final_hidden = self.plans[42].hidden.output;
        Ok(S14Position0PagedLayerTail {
            timeline: self.timeline,
            position: self.position,
            layer_final_compute_value,
            final_hidden,
        })
    }

    fn ensure_open(&self) -> Result<()> {
        if self.state != S14Position0PagedLayerBridgeState::Open {
            bail!("position0 paged layer bridge 已 poisoned");
        }
        Ok(())
    }

    fn abort_pending<U>(&mut self, error: anyhow::Error) -> Result<U> {
        let abort = self.timeline.abort_pending();
        self.state = S14Position0PagedLayerBridgeState::Poisoned;
        match abort {
            Ok(()) => Err(error),
            Err(abort_error) => Err(anyhow!(
                "{error:#}; abort pending 同时失败: {abort_error:#}"
            )),
        }
    }

    fn poison<U>(&mut self, error: anyhow::Error) -> Result<U> {
        self.state = S14Position0PagedLayerBridgeState::Poisoned;
        Err(error)
    }
}

pub struct S14Position0PagedLayerTail<'timeline, T>
where
    T: S14Position0PagedLayerTimelinePort,
{
    timeline: &'timeline mut T,
    position: u32,
    layer_final_compute_value: u64,
    final_hidden: crate::s14_position0_synchronous_layer_pager::S14Position0DeviceHiddenSlot,
}

impl<'timeline, T> S14Position0PagedLayerTail<'timeline, T>
where
    T: S14Position0PagedLayerTimelinePort,
{
    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn layer_final_compute_value(&self) -> u64 {
        self.layer_final_compute_value
    }

    pub fn final_hidden(
        &self,
    ) -> crate::s14_position0_synchronous_layer_pager::S14Position0DeviceHiddenSlot {
        self.final_hidden
    }

    pub fn timeline_mut(&mut self) -> &mut T {
        self.timeline
    }

    pub fn into_timeline(self) -> &'timeline mut T {
        self.timeline
    }
}

fn validate_full_depth_plans(plans: &[S14Position0SynchronousLayerPlan]) -> Result<()> {
    if plans.len() != usize::from(S14_POSITION0_SYNCHRONOUS_LAYER_COUNT) {
        bail!("position0 paged bridge 要求严格 43 层 plan");
    }
    for (index, plan) in plans.iter().enumerate() {
        plan.validate()
            .with_context(|| format!("validate paged bridge L{} plan", plan.layer))?;
        if usize::from(plan.layer) != index {
            bail!("position0 paged bridge layer/index 漂移 at {index}");
        }
        if index > 0 && plans[index - 1].hidden.output != plan.hidden.input {
            bail!("position0 paged bridge hidden 在 L{} 断链", plan.layer);
        }
    }
    Ok(())
}

fn validate_runtime_bank(plan: &S14Position0SynchronousLayerPlan, bank: usize) -> Result<()> {
    let expected = usize::from(plan.layer) % S14_POSITION0_SYNCHRONOUS_BANKS;
    if bank != expected || plan.routed_bank != bank {
        bail!("L{} paged runtime routed bank 漂移", plan.layer);
    }
    if let S14Position0StaticPage::Streamed { bank: static_bank } = plan.static_page {
        if static_bank != bank {
            bail!("L{} paged runtime static bank 漂移", plan.layer);
        }
    }
    Ok(())
}

fn validate_stage_receipt(
    plan: &S14Position0SynchronousLayerPlan,
    bank: usize,
    receipt: S14Position0PagedLayerStageReceipt,
    expected_static: u64,
    expected_routed: u64,
) -> Result<()> {
    if receipt.layer != plan.layer
        || receipt.bank != bank
        || receipt.static_uploaded_bytes != expected_static
        || receipt.routed_uploaded_bytes != expected_routed
        || receipt.hidden_host_bytes != 0
    {
        bail!(
            "L{} paged stage receipt 漂移: actual={receipt:?} expected_static={expected_static} expected_routed={expected_routed}",
            plan.layer
        );
    }
    Ok(())
}
