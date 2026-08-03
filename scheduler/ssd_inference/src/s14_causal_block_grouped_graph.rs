//! K=4/8 causal-block grouped MoE 的持久 Vulkan command graph。
//!
//! command pools、43 对 command buffers、双 staging 与 timeline 均在 graph 生命周期
//! 分配；block 热路径只 reset pools、重写动态 union bytes，并对每层提交一次 transfer
//! 与一次 grouped compute。具体 shader/pipeline 由窄 recorder ABI 注入，回执必须证明
//! 覆盖全部唯一专家与 K×top-6 assignments，且串行 whole-token 调用数为0。

use crate::{
    s14_causal_block_layer::{
        S14CausalBlockHiddenBinding, S14CausalBlockLayerRangePlan, S14CausalBlockUnionBankBinding,
    },
    s14_causal_block_union_materializer::{
        S14CausalBlockMaterializedUnion, S14CausalBlockUnionStageReceipt,
    },
    s14_dual_queue_timeline::{S14DualQueueTimeline, S14TimelineDrainReceipt},
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{LayerCausalBatchPlan, RouteDecision, FULL_DEPTH_LAYERS};
use std::sync::Arc;

const GRAPH_LAYER_COUNT: usize = 43;
const STAGING_BANKS: usize = 2;
const GRAPH_WAIT_TIMEOUT_NS: u64 = 60_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRecordedGroupedMoe {
    pub output_hidden: S14CausalBlockHiddenBinding,
    pub grouped_expert_work_items: usize,
    pub lane_assignments: usize,
    pub recorder_calls: u32,
    pub serial_token_forward_calls: u32,
}

/// 真实 grouped shader/pipeline recorder。实现只能向给定的单个 command buffer 录制
/// 整层工作，不拥有 queue submit 权，也不能调用 `S14Runtime::step`。
pub trait S14CausalBlockGroupedMoeRecorder {
    /// Reset recorder-owned sticky state before any layer of a new block can be
    /// submitted. No Vulkan queue work may still reference the previous block.
    fn begin_block(&mut self, base_position: u32, block_size: usize) -> Result<()>;

    fn record_grouped_moe(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        union_bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockRecordedGroupedMoe>;

    /// 证明返回的 hidden 仍由 recorder 自身保活。production 组合层不能只接收一个
    /// 可复制的 `vk::Buffer` 裸句柄；具体 recorder 必须把该 binding 反查到自己的
    /// output arena。默认 fail-closed，避免旧测试 recorder 被误当成 production owner。
    fn owns_output_hidden(&self, _binding: S14CausalBlockHiddenBinding) -> bool {
        false
    }

    /// Called only after the grouped graph timeline has drained. Implementors
    /// must validate sticky numeric status and may now release descriptor pools.
    fn finish_block_after_drain(&mut self, aborted: bool) -> Result<()>;

    /// Explicitly release persistent pipelines/workspaces after graph destroy.
    fn destroy(&mut self) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14CausalBlockGroupedGraphTelemetry {
    pub block_index: u64,
    pub resource_allocations_this_block: u64,
    pub command_pool_resets_this_block: u32,
    pub union_transfer_submit_calls: u32,
    pub grouped_compute_submit_calls: u32,
    pub host_staging_reuse_wait_calls: u32,
    pub drain_wait_calls: u32,
    pub serial_token_forward_calls: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockGroupedLayerReceipt {
    pub layer: u8,
    pub transfer_timeline_value: u64,
    pub compute_timeline_value: u64,
    pub staged_bytes: u64,
    pub host_range_copies: usize,
    pub gpu_upload_copy_regions: u32,
    pub grouped_submit_calls: u32,
    pub grouped_expert_work_items: usize,
    pub lane_assignments: usize,
    pub serial_token_forward_calls: u32,
    pub output_hidden: S14CausalBlockHiddenBinding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActiveBlock {
    base_position: u32,
    block_size: usize,
    bank_index: usize,
    next_layer: usize,
}

#[derive(Clone, Copy, Debug)]
struct PendingLayer {
    layer: u8,
    bank: S14CausalBlockUnionBankBinding,
    transfer_value: u64,
    stage: S14CausalBlockUnionStageReceipt,
}

/// Runtime 常驻 command graph。调用方必须在销毁前调用 `destroy`；该方法会先 drain。
pub struct S14CausalBlockGroupedGraph {
    ctx: Arc<VulkanContext>,
    transfer_pool: vk::CommandPool,
    compute_pool: vk::CommandPool,
    transfer_commands: [vk::CommandBuffer; GRAPH_LAYER_COUNT],
    compute_commands: [vk::CommandBuffer; GRAPH_LAYER_COUNT],
    staging: [GpuBuffer; STAGING_BANKS],
    timeline: Option<S14DualQueueTimeline>,
    staging_transfer_values: [u64; STAGING_BANKS],
    union_bank_compute_values: [u64; STAGING_BANKS],
    active: Option<ActiveBlock>,
    pending: Option<PendingLayer>,
    blocks_started: u64,
    telemetry: S14CausalBlockGroupedGraphTelemetry,
}

impl S14CausalBlockGroupedGraph {
    pub fn new(ctx: Arc<VulkanContext>, max_union_bytes: u64) -> Result<Self> {
        if max_union_bytes == 0 || !ctx.timeline_semaphore {
            bail!("causal-block grouped graph 要求非零 union 容量与 timeline semaphore");
        }
        let (transfer_pool, compute_pool, transfer_commands, compute_commands) =
            allocate_graph_commands(&ctx)?;
        let timeline = match S14DualQueueTimeline::new(&ctx) {
            Ok(timeline) => timeline,
            Err(error) => {
                destroy_command_pools(&ctx, transfer_pool, compute_pool);
                return Err(error.context("创建 causal-block grouped timeline"));
            }
        };
        let staging0 = match GpuBuffer::new_staging(&ctx, max_union_bytes) {
            Ok(buffer) => buffer,
            Err(error) => {
                timeline.destroy(&ctx);
                destroy_command_pools(&ctx, transfer_pool, compute_pool);
                return Err(error.context("创建 causal-block union staging0"));
            }
        };
        let staging1 = match GpuBuffer::new_staging(&ctx, max_union_bytes) {
            Ok(buffer) => buffer,
            Err(error) => {
                staging0.destroy(&ctx);
                timeline.destroy(&ctx);
                destroy_command_pools(&ctx, transfer_pool, compute_pool);
                return Err(error.context("创建 causal-block union staging1"));
            }
        };
        Ok(Self {
            ctx,
            transfer_pool,
            compute_pool,
            transfer_commands,
            compute_commands,
            staging: [staging0, staging1],
            timeline: Some(timeline),
            staging_transfer_values: [0; STAGING_BANKS],
            union_bank_compute_values: [0; STAGING_BANKS],
            active: None,
            pending: None,
            blocks_started: 0,
            telemetry: S14CausalBlockGroupedGraphTelemetry::default(),
        })
    }

    pub fn begin_block(
        &mut self,
        base_position: u32,
        block_size: usize,
        bank_index: usize,
    ) -> Result<S14CausalBlockGroupedGraphTelemetry> {
        if self.active.is_some() || self.pending.is_some() {
            bail!("causal-block grouped graph 已有 active block/layer");
        }
        if !matches!(block_size, 4 | 8) || bank_index >= STAGING_BANKS {
            bail!("causal-block grouped graph K/bank 非法");
        }
        base_position
            .checked_add(block_size as u32)
            .context("causal-block grouped graph position overflow")?;
        unsafe {
            self.ctx
                .device
                .reset_command_pool(self.transfer_pool, vk::CommandPoolResetFlags::empty())?;
            self.ctx
                .device
                .reset_command_pool(self.compute_pool, vk::CommandPoolResetFlags::empty())?;
        }
        self.telemetry = S14CausalBlockGroupedGraphTelemetry {
            block_index: self.blocks_started,
            resource_allocations_this_block: 0,
            command_pool_resets_this_block: 2,
            ..S14CausalBlockGroupedGraphTelemetry::default()
        };
        self.blocks_started = self
            .blocks_started
            .checked_add(1)
            .context("causal-block grouped graph block counter overflow")?;
        self.active = Some(ActiveBlock {
            base_position,
            block_size,
            bank_index,
            next_layer: 0,
        });
        Ok(self.telemetry)
    }

    /// 物化结果写入 ping-pong staging，并录制/提交整层唯一一次 union copy。
    pub fn upload_union_layer(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        materialized: &S14CausalBlockMaterializedUnion,
    ) -> Result<u64> {
        let active = self.active.context("causal-block grouped graph 未 begin")?;
        if self.pending.is_some() || active.next_layer >= GRAPH_LAYER_COUNT {
            bail!("causal-block grouped graph layer upload 顺序非法");
        }
        let expected_layer = FULL_DEPTH_LAYERS[active.next_layer];
        if materialized.layer != expected_layer
            || bank.bank_index != active.bank_index
            || bank.buffer == vk::Buffer::null()
            || bank.allocated_bank_bytes < materialized.union_expert_bytes
        {
            bail!("causal-block grouped graph union layer/bank identity 漂移");
        }
        let staging_index = active.next_layer % STAGING_BANKS;
        let old_transfer = self.staging_transfer_values[staging_index];
        if old_transfer != 0 {
            self.timeline()?
                .wait_transfer(&self.ctx, old_transfer, GRAPH_WAIT_TIMEOUT_NS)
                .context("等待 causal-block staging 复用 transfer")?;
            self.telemetry.host_staging_reuse_wait_calls = self
                .telemetry
                .host_staging_reuse_wait_calls
                .checked_add(1)
                .context("causal-block staging wait counter overflow")?;
        }
        let stage = materialized.stage_into_gpu(&self.staging[staging_index])?;
        if stage.gpu_upload_copy_regions != 1 {
            bail!("causal-block union upload 必须精确一个 copy region");
        }
        let command = self.transfer_commands[active.next_layer];
        unsafe {
            self.ctx.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.ctx.device.cmd_copy_buffer(
                command,
                self.staging[staging_index].handle(),
                bank.buffer,
                &[materialized.upload_copy_region()],
            );
            self.ctx.device.end_command_buffer(command)?;
        }
        let reuse_after_compute = match self.union_bank_compute_values[bank.bank_index] {
            0 => None,
            value => Some(value),
        };
        let ctx = Arc::clone(&self.ctx);
        let transfer_value = unsafe {
            self.timeline_mut()?
                .submit_transfer(&ctx, command, reuse_after_compute)?
        };
        self.staging_transfer_values[staging_index] = transfer_value;
        self.pending = Some(PendingLayer {
            layer: expected_layer,
            bank,
            transfer_value,
            stage,
        });
        self.telemetry.union_transfer_submit_calls = self
            .telemetry
            .union_transfer_submit_calls
            .checked_add(1)
            .context("causal-block transfer submit counter overflow")?;
        Ok(transfer_value)
    }

    /// 在一个 compute command buffer 中录制整层所有唯一专家并只 submit 一次。
    pub fn record_and_submit_grouped_moe<R: S14CausalBlockGroupedMoeRecorder>(
        &mut self,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
        recorder: &mut R,
    ) -> Result<S14CausalBlockGroupedLayerReceipt> {
        let active = self.active.context("causal-block grouped graph 未 begin")?;
        let pending = self
            .pending
            .context("causal-block grouped graph 缺少 union upload")?;
        if active.next_layer >= GRAPH_LAYER_COUNT
            || pending.layer != FULL_DEPTH_LAYERS[active.next_layer]
            || batch_plan.layer != pending.layer
            || range_plan.layer != pending.layer
            || routes.len() != active.block_size
            || post_attention_hidden.block_size != active.block_size
        {
            bail!("causal-block grouped graph compute layer/K identity 漂移");
        }
        batch_plan
            .validate_against(routes)
            .context("causal-block grouped plan 不能无损重建 K×top-6")?;
        if range_plan.unique_experts != batch_plan.unique_experts
            || range_plan.union_expert_bytes != batch_plan.union_expert_bytes
        {
            bail!("causal-block grouped graph batch/range union 漂移");
        }
        let command = self.compute_commands[active.next_layer];
        unsafe {
            self.ctx.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let recorded = recorder.record_grouped_moe(
            &self.ctx,
            command,
            pending.bank,
            post_attention_hidden,
            routes,
            batch_plan,
            range_plan,
        )?;
        validate_recorded_grouped_moe(recorded, post_attention_hidden, batch_plan)?;
        unsafe { self.ctx.device.end_command_buffer(command)? };
        let ctx = Arc::clone(&self.ctx);
        let compute_value = unsafe {
            self.timeline_mut()?
                .submit_compute(&ctx, command, pending.transfer_value)?
        };
        self.union_bank_compute_values[pending.bank.bank_index] = compute_value;
        self.telemetry.grouped_compute_submit_calls = self
            .telemetry
            .grouped_compute_submit_calls
            .checked_add(1)
            .context("causal-block grouped submit counter overflow")?;
        self.telemetry.serial_token_forward_calls = self
            .telemetry
            .serial_token_forward_calls
            .checked_add(recorded.serial_token_forward_calls)
            .context("causal-block serial token counter overflow")?;
        self.pending = None;
        self.active = Some(ActiveBlock {
            next_layer: active.next_layer + 1,
            ..active
        });
        Ok(S14CausalBlockGroupedLayerReceipt {
            layer: pending.layer,
            transfer_timeline_value: pending.transfer_value,
            compute_timeline_value: compute_value,
            staged_bytes: pending.stage.staged_bytes,
            host_range_copies: pending.stage.host_range_copies,
            gpu_upload_copy_regions: pending.stage.gpu_upload_copy_regions,
            grouped_submit_calls: 1,
            grouped_expert_work_items: recorded.grouped_expert_work_items,
            lane_assignments: recorded.lane_assignments,
            serial_token_forward_calls: recorded.serial_token_forward_calls,
            output_hidden: recorded.output_hidden,
        })
    }

    pub fn seal_and_drain(&mut self) -> Result<S14TimelineDrainReceipt> {
        let active = self.active.context("causal-block grouped graph 未 begin")?;
        if self.pending.is_some()
            || active.next_layer != GRAPH_LAYER_COUNT
            || self.telemetry.union_transfer_submit_calls != GRAPH_LAYER_COUNT as u32
            || self.telemetry.grouped_compute_submit_calls != GRAPH_LAYER_COUNT as u32
            || self.telemetry.serial_token_forward_calls != 0
        {
            bail!("causal-block grouped graph 禁止 seal 不完整/串行43层图");
        }
        let drain = self
            .timeline()?
            .drain_all(&self.ctx, GRAPH_WAIT_TIMEOUT_NS)
            .context("drain causal-block grouped graph")?;
        self.telemetry.drain_wait_calls = self
            .telemetry
            .drain_wait_calls
            .checked_add(drain.host_wait_calls)
            .context("causal-block drain wait counter overflow")?;
        self.active = None;
        Ok(drain)
    }

    pub fn drain_and_abort(&mut self) -> Result<S14TimelineDrainReceipt> {
        if self.active.is_none() && self.pending.is_none() {
            bail!("causal-block grouped graph 没有 active work 可 abort");
        }
        let drain = self
            .timeline()?
            .drain_all(&self.ctx, GRAPH_WAIT_TIMEOUT_NS)
            .context("abort drain causal-block grouped graph")?;
        self.telemetry.drain_wait_calls = self
            .telemetry
            .drain_wait_calls
            .checked_add(drain.host_wait_calls)
            .context("causal-block abort wait counter overflow")?;
        self.pending = None;
        self.active = None;
        Ok(drain)
    }

    pub fn telemetry(&self) -> S14CausalBlockGroupedGraphTelemetry {
        self.telemetry
    }

    /// 销毁前会 drain；调用者不得在此后继续使用 recorder 引用的 graph 资源。
    pub fn destroy(mut self) -> Result<()> {
        if self.active.is_some() || self.pending.is_some() {
            self.drain_and_abort()?;
        }
        if let Some(timeline) = self.timeline.take() {
            timeline.destroy(&self.ctx);
        }
        for staging in &self.staging {
            staging.destroy(&self.ctx);
        }
        destroy_command_pools(&self.ctx, self.transfer_pool, self.compute_pool);
        Ok(())
    }

    fn timeline(&self) -> Result<&S14DualQueueTimeline> {
        self.timeline
            .as_ref()
            .context("causal-block grouped timeline 已销毁")
    }

    fn timeline_mut(&mut self) -> Result<&mut S14DualQueueTimeline> {
        self.timeline
            .as_mut()
            .context("causal-block grouped timeline 已销毁")
    }
}

fn validate_recorded_grouped_moe(
    recorded: S14CausalBlockRecordedGroupedMoe,
    input_hidden: S14CausalBlockHiddenBinding,
    batch_plan: &LayerCausalBatchPlan,
) -> Result<()> {
    if recorded.recorder_calls != 1
        || recorded.serial_token_forward_calls != 0
        || recorded.grouped_expert_work_items != batch_plan.experts.len()
        || recorded.lane_assignments != batch_plan.assignments
        || recorded.output_hidden.buffer == vk::Buffer::null()
        || recorded.output_hidden.block_size != input_hidden.block_size
        || recorded.output_hidden.bytes != input_hidden.bytes
        || recorded.output_hidden.generation != input_hidden.generation.checked_add(1).unwrap_or(0)
    {
        bail!("causal-block grouped recorder 未证明一次完整 union/K×top-6 dispatch");
    }
    Ok(())
}

fn allocate_graph_commands(
    ctx: &VulkanContext,
) -> Result<(
    vk::CommandPool,
    vk::CommandPool,
    [vk::CommandBuffer; GRAPH_LAYER_COUNT],
    [vk::CommandBuffer; GRAPH_LAYER_COUNT],
)> {
    unsafe {
        let transfer_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_transfer)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let compute_pool = match ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        ) {
            Ok(pool) => pool,
            Err(error) => {
                ctx.device.destroy_command_pool(transfer_pool, None);
                return Err(error.into());
            }
        };
        let transfer = match ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(transfer_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(GRAPH_LAYER_COUNT as u32),
        ) {
            Ok(commands) => commands,
            Err(error) => {
                destroy_command_pools(ctx, transfer_pool, compute_pool);
                return Err(error.into());
            }
        };
        let compute = match ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(compute_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(GRAPH_LAYER_COUNT as u32),
        ) {
            Ok(commands) => commands,
            Err(error) => {
                destroy_command_pools(ctx, transfer_pool, compute_pool);
                return Err(error.into());
            }
        };
        let transfer_commands = transfer.try_into().map_err(|commands: Vec<_>| {
            destroy_command_pools(ctx, transfer_pool, compute_pool);
            anyhow::anyhow!(
                "causal-block transfer command count 漂移: {}",
                commands.len()
            )
        })?;
        let compute_commands = compute.try_into().map_err(|commands: Vec<_>| {
            destroy_command_pools(ctx, transfer_pool, compute_pool);
            anyhow::anyhow!(
                "causal-block compute command count 漂移: {}",
                commands.len()
            )
        })?;
        Ok((
            transfer_pool,
            compute_pool,
            transfer_commands,
            compute_commands,
        ))
    }
}

fn destroy_command_pools(
    ctx: &VulkanContext,
    transfer_pool: vk::CommandPool,
    compute_pool: vk::CommandPool,
) {
    unsafe {
        ctx.device.destroy_command_pool(compute_pool, None);
        ctx.device.destroy_command_pool(transfer_pool, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use polaris_s14_runner::{build_layer_causal_batch_plan, router_kind_for_layer};

    #[test]
    fn grouped_receipt_requires_one_recorder_full_union_and_zero_serial_forward() {
        let routes = (0..4)
            .map(|lane| RouteDecision {
                layer: 7,
                kind: router_kind_for_layer(7).unwrap(),
                expert_ids: (lane..lane + 6).map(|expert| expert as u16).collect(),
                weights: vec![0.25; 6],
            })
            .collect::<Vec<_>>();
        let batch = build_layer_causal_batch_plan(&routes).unwrap();
        let input = hidden(4, 10);
        let valid = S14CausalBlockRecordedGroupedMoe {
            output_hidden: hidden(4, 11),
            grouped_expert_work_items: batch.experts.len(),
            lane_assignments: batch.assignments,
            recorder_calls: 1,
            serial_token_forward_calls: 0,
        };
        validate_recorded_grouped_moe(valid, input, &batch).unwrap();

        for invalid in [
            S14CausalBlockRecordedGroupedMoe {
                recorder_calls: 4,
                ..valid
            },
            S14CausalBlockRecordedGroupedMoe {
                serial_token_forward_calls: 4,
                ..valid
            },
            S14CausalBlockRecordedGroupedMoe {
                grouped_expert_work_items: batch.experts.len() - 1,
                ..valid
            },
            S14CausalBlockRecordedGroupedMoe {
                lane_assignments: batch.assignments - 1,
                ..valid
            },
        ] {
            assert!(validate_recorded_grouped_moe(invalid, input, &batch).is_err());
        }
    }

    fn hidden(block_size: usize, generation: u64) -> S14CausalBlockHiddenBinding {
        S14CausalBlockHiddenBinding {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            bytes: block_size as u64 * 4 * 4096 * 2,
            block_size,
            generation,
        }
    }
}
