//! S14 K=4 production 单层的两阶段 Vulkan owner。
//!
//! 在线 top-6 必须先完成，CPU 才能按真实 expert identity 做 Range proof/SHA/mmap。
//! 因此诚实的单层边界不是一个跨磁盘 I/O 的 Vulkan command buffer，而是：
//!
//! 1. HC/QKV/attention/router 的一个 K-row command buffer/submit；
//! 2. route 返回后物化实际 Range，把唯一一次 upload copy 与 grouped MoE 录进同一个
//!    command buffer/submit。
//!
//! 本模块实现第二阶段的 production adapter，并冻结整层总计两次 submit 的强回执。
//! union bank 由 `Arc<GpuBuffer>` lease 保活；grouped output 必须能反查到 recorder 自有
//! output arena，不能用可复制的裸 `vk::Buffer` 冒充资源所有权。

use crate::{
    s14_causal_block_grouped_graph::{
        S14CausalBlockGroupedMoeRecorder, S14CausalBlockRecordedGroupedMoe,
    },
    s14_causal_block_grouped_moe_recorder::{
        S14CausalBlockGroupedMoeStaticLayerResources, S14CausalBlockGroupedMoeVulkanRecorder,
    },
    s14_causal_block_layer::{
        S14CausalBlockAttentionRouterOutput, S14CausalBlockGroupedMoeOutput,
        S14CausalBlockHiddenBinding, S14CausalBlockLayerInput, S14CausalBlockLayerRangePlan,
        S14CausalBlockRangeEvidenceReceipt, S14CausalBlockUnionBankBinding,
        S14CausalBlockUnionMaterializeReceipt,
    },
    s14_causal_block_moe_adapter::S14CausalBlockVulkanMoeAdapter,
    s14_causal_block_resources::S14CausalBlockUnionBankPlan,
    s14_causal_block_union_materializer::{
        build_causal_block_union_identity_plan, S14CausalBlockMaterializedUnion,
        S14CausalBlockUnionMaterializer, S14CausalBlockUnionStageReceipt,
    },
    s14_dynamic_page_cache_readiness::DynamicPageFetchMode,
    s14_dynamic_routed_page_plan::FullDepthExpertCatalog,
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{GraphProfile, LayerCausalBatchPlan, RouteDecision, FULL_DEPTH_LAYERS};
use std::{fmt, path::Path, path::PathBuf, sync::Arc};

const K4: usize = 4;
const SINGLE_LAYER_INDEX: usize = 0;
const WAIT_TIMEOUT_NS: u64 = 60_000_000_000;

/// Runtime union bank 的强 lease。binding 只能从持有真实 `GpuBuffer` 的对象生成。
#[derive(Clone)]
pub struct S14CausalBlockOwnedUnionBank {
    buffer: Arc<GpuBuffer>,
    bank_index: usize,
}

impl fmt::Debug for S14CausalBlockOwnedUnionBank {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockOwnedUnionBank")
            .field("buffer", &self.buffer.handle())
            .field("bank_index", &self.bank_index)
            .field("allocated_bytes", &self.buffer.size())
            .finish()
    }
}

impl S14CausalBlockOwnedUnionBank {
    pub fn new(buffer: Arc<GpuBuffer>, bank_index: usize) -> Result<Self> {
        let plan = S14CausalBlockUnionBankPlan::build(K4)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if bank_index >= 2
            || buffer.handle() == vk::Buffer::null()
            || buffer.size() < plan.allocated_bank_bytes
        {
            bail!("single-layer owned union bank identity/K4 capacity 非法");
        }
        Ok(Self { buffer, bank_index })
    }

    pub fn binding(&self) -> S14CausalBlockUnionBankBinding {
        S14CausalBlockUnionBankBinding {
            bank_index: self.bank_index,
            buffer: self.buffer.handle(),
            allocated_bank_bytes: self.buffer.size(),
        }
    }

    fn owns(&self, binding: S14CausalBlockUnionBankBinding) -> bool {
        binding == self.binding()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14CausalBlockSingleLayerTelemetry {
    /// 上游 production HC/QKV 强回执证明的 K-row route submit。
    pub route_producer_submit_calls: u32,
    /// Range upload copy 与 grouped MoE 共用的 command buffer 数。
    pub fused_range_grouped_command_buffers: u32,
    pub fused_range_grouped_submit_calls: u32,
    pub total_layer_submit_calls: u32,
    pub layer_recorder_calls: u32,
    pub union_upload_copy_regions: u32,
    pub grouped_recorder_calls: u32,
    pub serial_token_forward_calls: u32,
    pub actual_physical_ranges: usize,
    pub actual_union_bytes: u64,
    pub union_buffer_owned: bool,
    pub output_buffer_owned: bool,
    pub drained: bool,
}

impl S14CausalBlockSingleLayerTelemetry {
    pub fn validate(self) -> Result<()> {
        if self.route_producer_submit_calls != 1
            || self.fused_range_grouped_command_buffers != 1
            || self.fused_range_grouped_submit_calls != 1
            || self.total_layer_submit_calls != 2
            || self.layer_recorder_calls != 1
            || self.union_upload_copy_regions != 1
            || self.grouped_recorder_calls != 1
            || self.serial_token_forward_calls != 0
            || self.actual_physical_ranges == 0
            || self.actual_union_bytes == 0
            || !self.union_buffer_owned
            || !self.output_buffer_owned
            || !self.drained
        {
            bail!("single-layer K4 telemetry 未闭合一次 recorder/两段 submit/owned lease");
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CapturedRoute {
    input_layer: u8,
    post_attention_hidden: S14CausalBlockHiddenBinding,
    routes: Vec<RouteDecision>,
}

#[derive(Debug)]
struct PendingMaterialized {
    captured: CapturedRoute,
    range_plan: S14CausalBlockLayerRangePlan,
    materialized: S14CausalBlockMaterializedUnion,
    stage: S14CausalBlockUnionStageReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Active {
        base_position: u32,
        block_size: usize,
        completed_layers: usize,
    },
    Destroyed,
}

/// 单层第二阶段 owner。`recorder`、staging、command pool/buffer/fence、实际 Range mmap
/// lease 与 union-bank `Arc` 全部在同一个对象内保活。
pub struct S14CausalBlockProductionSingleLayerMoeAdapter<R: S14CausalBlockGroupedMoeRecorder> {
    ctx: Arc<VulkanContext>,
    catalog: FullDepthExpertCatalog,
    cache_root: PathBuf,
    materializer: S14CausalBlockUnionMaterializer,
    recorder: Option<R>,
    union: S14CausalBlockOwnedUnionBank,
    staging: Option<GpuBuffer>,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    in_flight: bool,
    recorder_finished: bool,
    captured: Option<CapturedRoute>,
    pending: Option<PendingMaterialized>,
    phase: Phase,
    telemetry: S14CausalBlockSingleLayerTelemetry,
}

impl<R: S14CausalBlockGroupedMoeRecorder> fmt::Debug
    for S14CausalBlockProductionSingleLayerMoeAdapter<R>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockProductionSingleLayerMoeAdapter")
            .field("cache_root", &self.cache_root)
            .field("union", &self.union)
            .field("phase", &self.phase)
            .field("in_flight", &self.in_flight)
            .field("telemetry", &self.telemetry)
            .finish_non_exhaustive()
    }
}

impl<R: S14CausalBlockGroupedMoeRecorder> S14CausalBlockProductionSingleLayerMoeAdapter<R> {
    pub fn new(
        ctx: Arc<VulkanContext>,
        catalog: FullDepthExpertCatalog,
        cache_root: &Path,
        fetch_mode: DynamicPageFetchMode,
        recorder: R,
        union: S14CausalBlockOwnedUnionBank,
    ) -> Result<Self> {
        let plan = S14CausalBlockUnionBankPlan::build(K4)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if !union.owns(union.binding())
            || union.binding().allocated_bank_bytes < plan.allocated_bank_bytes
        {
            bail!("single-layer union lease 与 K4 plan 漂移");
        }
        let staging = GpuBuffer::new_staging(&ctx, plan.allocated_bank_bytes)?;
        let materializer = S14CausalBlockUnionMaterializer::new(cache_root, fetch_mode)?;
        let command_pool = match unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(error) => {
                staging.destroy(&ctx);
                return Err(error.into());
            }
        };
        let command = match unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(commands) => commands[0],
            Err(error) => {
                unsafe { ctx.device.destroy_command_pool(command_pool, None) };
                staging.destroy(&ctx);
                return Err(error.into());
            }
        };
        let fence = match unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                unsafe { ctx.device.destroy_command_pool(command_pool, None) };
                staging.destroy(&ctx);
                return Err(error.into());
            }
        };
        Ok(Self {
            ctx,
            catalog,
            cache_root: cache_root.to_path_buf(),
            materializer,
            recorder: Some(recorder),
            union,
            staging: Some(staging),
            command_pool,
            command,
            fence,
            in_flight: false,
            recorder_finished: false,
            captured: None,
            pending: None,
            phase: Phase::Idle,
            telemetry: S14CausalBlockSingleLayerTelemetry::default(),
        })
    }

    pub fn telemetry(&self) -> S14CausalBlockSingleLayerTelemetry {
        self.telemetry
    }

    fn begin_inner(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> Result<()> {
        if self.phase != Phase::Idle
            || block_size != K4
            || !self.union.owns(bank)
            || base_position == 0
            || base_position
                .checked_add(block_size as u32)
                .is_none_or(|end| end > 127)
        {
            bail!("single-layer K4 begin phase/position/owned union lease 非法");
        }
        self.recorder_mut()?
            .begin_block(base_position, block_size)?;
        self.phase = Phase::Active {
            base_position,
            block_size,
            completed_layers: 0,
        };
        self.telemetry = S14CausalBlockSingleLayerTelemetry::default();
        self.captured = None;
        self.pending = None;
        self.recorder_finished = false;
        Ok(())
    }

    fn capture_inner(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
        output: &S14CausalBlockAttentionRouterOutput,
    ) -> Result<()> {
        let (base_position, block_size, completed_layers) = self.active_identity()?;
        let layer = FULL_DEPTH_LAYERS[SINGLE_LAYER_INDEX];
        if completed_layers != 0
            || self.captured.is_some()
            || self.pending.is_some()
            || input.base_position != base_position
            || input.layer != layer
            || input.input_token_ids.len() != block_size
            || output.forward_calls != 1
            || output.routes.len() != block_size
            || output.post_attention_hidden.block_size != block_size
            || output.post_attention_hidden.bytes != input.input_hidden.bytes
            || output.post_attention_hidden.generation
                != input
                    .input_hidden
                    .generation
                    .checked_add(1)
                    .context("single-layer attention generation overflow")?
            || (output.post_attention_hidden.buffer == input.input_hidden.buffer
                && output.post_attention_hidden.offset == input.input_hidden.offset)
        {
            bail!("single-layer route producer output 与 K4 layer/hidden identity 漂移");
        }
        for route in &output.routes {
            route
                .validate_for(GraphProfile::FullDepth43NativeTop6)
                .context("single-layer online top-6 route 非法")?;
            if route.layer != layer {
                bail!("single-layer route layer 漂移");
            }
        }
        self.captured = Some(CapturedRoute {
            input_layer: layer,
            post_attention_hidden: output.post_attention_hidden,
            routes: output.routes.clone(),
        });
        // HC/QKV adapter 的强回执已在 Vulkan backend 中校验 command_graph_submit_calls=1。
        self.telemetry.route_producer_submit_calls = 1;
        Ok(())
    }

    fn materialize_inner(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockUnionMaterializeReceipt> {
        let (base_position, block_size, completed_layers) = self.active_identity()?;
        let captured = self
            .captured
            .take()
            .context("single-layer actual Range 前缺少 route producer output")?;
        if completed_layers != 0
            || self.pending.is_some()
            || !self.union.owns(bank)
            || range_plan.layer != captured.input_layer
            || range_plan.block_size != block_size
        {
            self.captured = Some(captured);
            bail!("single-layer actual Range plan/bank/K identity 漂移");
        }
        let identity = build_causal_block_union_identity_plan(
            &self.catalog,
            &self.cache_root,
            base_position,
            &captured.routes,
            range_plan,
        )?;
        let materialized = self.materializer.materialize(&identity)?;
        if materialized.layer != range_plan.layer
            || materialized.unique_experts != range_plan.unique_experts
            || materialized.union_expert_bytes != range_plan.union_expert_bytes
            || materialized.telemetry.physical_ranges != range_plan.physical_ranges
            || materialized.telemetry.proof_assets != range_plan.physical_ranges
            || materialized.telemetry.gpu_upload_copy_regions != 1
        {
            bail!("single-layer actual Range proof/SHA/mmap 回执漂移");
        }
        let stage = materialized.stage_into_gpu(self.staging()?)?;
        if stage.staged_bytes != range_plan.union_expert_bytes
            || stage.host_range_copies != range_plan.physical_ranges
            || stage.gpu_upload_copy_regions != 1
        {
            bail!("single-layer actual Range staging 回执漂移");
        }
        let telemetry = materialized.telemetry;
        self.pending = Some(PendingMaterialized {
            captured,
            range_plan: range_plan.clone(),
            materialized,
            stage,
        });
        Ok(S14CausalBlockUnionMaterializeReceipt {
            layer: range_plan.layer,
            bank_index: bank.bank_index,
            unique_experts: range_plan.unique_experts,
            physical_ranges: range_plan.physical_ranges,
            uploaded_bytes: range_plan.union_expert_bytes,
            // 这里表示一次真实物化；GPU copy 延迟到 grouped 同一个 command buffer。
            materialize_calls: 1,
            range_evidence: S14CausalBlockRangeEvidenceReceipt {
                proof_assets: telemetry.proof_assets,
                explicit_fetch_lane_plans: telemetry.explicit_fetch_lane_plans,
                mmap_requests: telemetry.mmap_requests_this_call,
                mmap_hits: telemetry.mmap_hits_this_call,
                mmap_misses: telemetry.mmap_misses_this_call,
                sha256_bytes: telemetry.sha256_bytes_this_call,
                staging_range_copies: stage.host_range_copies,
                gpu_upload_copy_regions: stage.gpu_upload_copy_regions,
            },
        })
    }

    fn grouped_inner(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockGroupedMoeOutput> {
        let (_, block_size, completed_layers) = self.active_identity()?;
        let pending = self
            .pending
            .take()
            .context("single-layer grouped MoE 前缺少实际 Range lease")?;
        if completed_layers != 0
            || !self.union.owns(bank)
            || pending.captured.post_attention_hidden != post_attention_hidden
            || pending.captured.routes != routes
            || pending.range_plan != *range_plan
            || batch_plan.block_size != block_size
            || batch_plan.layer != range_plan.layer
        {
            self.pending = Some(pending);
            bail!("single-layer grouped MoE 与 captured route/actual Range identity 漂移");
        }
        batch_plan
            .validate_against(routes)
            .context("single-layer grouped MoE 不能无损重建 K×top-6")?;
        self.reset_command_owner()?;
        unsafe {
            self.ctx.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.ctx.device.cmd_copy_buffer(
                self.command,
                self.staging()?.handle(),
                self.union.binding().buffer,
                &[pending.materialized.upload_copy_region()],
            );
            transfer_to_compute_barrier(&self.ctx, self.command);
        }
        let ctx = Arc::clone(&self.ctx);
        let command = self.command;
        let recorded: S14CausalBlockRecordedGroupedMoe = self.recorder_mut()?.record_grouped_moe(
            &ctx,
            command,
            bank,
            post_attention_hidden,
            routes,
            batch_plan,
            range_plan,
        )?;
        self.validate_recorded(&recorded, batch_plan, &pending)?;
        unsafe {
            self.ctx.device.end_command_buffer(self.command)?;
            let commands = [self.command];
            self.ctx.device.queue_submit(
                self.ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                self.fence,
            )?;
        }
        self.in_flight = true;
        unsafe {
            self.ctx
                .device
                .wait_for_fences(&[self.fence], true, WAIT_TIMEOUT_NS)?
        };
        self.in_flight = false;
        self.recorder_mut()?.finish_block_after_drain(true)?;
        self.recorder_finished = true;
        self.phase = Phase::Active {
            base_position: self.active_identity()?.0,
            block_size,
            completed_layers: 1,
        };
        self.telemetry = S14CausalBlockSingleLayerTelemetry {
            route_producer_submit_calls: 1,
            fused_range_grouped_command_buffers: 1,
            fused_range_grouped_submit_calls: 1,
            total_layer_submit_calls: 2,
            layer_recorder_calls: 1,
            union_upload_copy_regions: pending.stage.gpu_upload_copy_regions,
            grouped_recorder_calls: recorded.recorder_calls,
            serial_token_forward_calls: recorded.serial_token_forward_calls,
            actual_physical_ranges: pending.range_plan.physical_ranges,
            actual_union_bytes: pending.materialized.union_expert_bytes,
            union_buffer_owned: self.union.owns(bank),
            output_buffer_owned: self
                .recorder_ref()?
                .owns_output_hidden(recorded.output_hidden),
            drained: true,
        };
        self.telemetry.validate()?;
        Ok(S14CausalBlockGroupedMoeOutput {
            output_hidden: recorded.output_hidden,
            grouped_submit_calls: 1,
            serial_token_forward_calls: 0,
            unique_experts: recorded.grouped_expert_work_items,
        })
    }

    fn validate_recorded(
        &self,
        recorded: &S14CausalBlockRecordedGroupedMoe,
        batch_plan: &LayerCausalBatchPlan,
        pending: &PendingMaterialized,
    ) -> Result<()> {
        if recorded.recorder_calls != 1
            || recorded.serial_token_forward_calls != 0
            || recorded.grouped_expert_work_items != batch_plan.unique_experts
            || recorded.lane_assignments != batch_plan.assignments
            || recorded.output_hidden.block_size != K4
            || recorded.output_hidden.generation
                != pending
                    .captured
                    .post_attention_hidden
                    .generation
                    .checked_add(1)
                    .context("single-layer grouped generation overflow")?
            || !self
                .recorder_ref()?
                .owns_output_hidden(recorded.output_hidden)
        {
            bail!("single-layer grouped recorder coverage/output owner 回执漂移");
        }
        Ok(())
    }

    fn active_identity(&self) -> Result<(u32, usize, usize)> {
        match self.phase {
            Phase::Active {
                base_position,
                block_size,
                completed_layers,
            } => Ok((base_position, block_size, completed_layers)),
            _ => bail!("single-layer adapter 当前没有 active K4 layer"),
        }
    }

    fn reset_command_owner(&mut self) -> Result<()> {
        if self.in_flight {
            bail!("single-layer fused command owner 尚未 drain");
        }
        unsafe {
            self.ctx.device.reset_fences(&[self.fence])?;
            self.ctx
                .device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<()> {
        if self.in_flight {
            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(&[self.fence], true, WAIT_TIMEOUT_NS)?
            };
            self.in_flight = false;
        }
        if self.phase != Phase::Idle && !self.recorder_finished {
            self.recorder_mut()?.finish_block_after_drain(true)?;
            self.recorder_finished = true;
        }
        Ok(())
    }

    fn abort_inner(&mut self, completed_layers: usize) -> Result<()> {
        if self.phase == Phase::Idle {
            return Ok(());
        }
        let expected = self.active_identity()?.2;
        self.drain()?;
        self.captured = None;
        self.pending = None;
        self.phase = Phase::Idle;
        if completed_layers != expected {
            bail!(
                "single-layer abort completed_layers 漂移: reported={completed_layers} expected={expected}"
            );
        }
        Ok(())
    }

    fn destroy_inner(&mut self) -> Result<()> {
        if self.phase == Phase::Destroyed {
            return Ok(());
        }
        if self.phase != Phase::Idle {
            let completed = self.active_identity()?.2;
            self.abort_inner(completed)?;
        }
        if let Some(recorder) = self.recorder.as_mut() {
            recorder.destroy()?;
        }
        self.recorder = None;
        unsafe {
            self.ctx.device.destroy_fence(self.fence, None);
            self.ctx
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        if let Some(staging) = self.staging.take() {
            staging.destroy(&self.ctx);
        }
        self.phase = Phase::Destroyed;
        Ok(())
    }

    fn staging(&self) -> Result<&GpuBuffer> {
        self.staging.as_ref().context("single-layer staging 已销毁")
    }

    fn recorder_ref(&self) -> Result<&R> {
        self.recorder
            .as_ref()
            .context("single-layer grouped recorder 已销毁")
    }

    fn recorder_mut(&mut self) -> Result<&mut R> {
        self.recorder
            .as_mut()
            .context("single-layer grouped recorder 已销毁")
    }
}

impl<R: S14CausalBlockGroupedMoeRecorder> S14CausalBlockVulkanMoeAdapter
    for S14CausalBlockProductionSingleLayerMoeAdapter<R>
{
    fn begin_block(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> std::result::Result<(), String> {
        self.begin_inner(bank, base_position, block_size)
            .map_err(|error| format!("{error:#}"))
    }

    fn capture_attention_router_output(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
        output: &S14CausalBlockAttentionRouterOutput,
    ) -> std::result::Result<(), String> {
        self.capture_inner(input, output)
            .map_err(|error| format!("{error:#}"))
    }

    fn materialize_union_ranges(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> std::result::Result<S14CausalBlockUnionMaterializeReceipt, String> {
        self.materialize_inner(bank, range_plan)
            .map_err(|error| format!("{error:#}"))
    }

    fn run_grouped_moe(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> std::result::Result<S14CausalBlockGroupedMoeOutput, String> {
        self.grouped_inner(bank, post_attention_hidden, routes, batch_plan, range_plan)
            .map_err(|error| format!("{error:#}"))
    }

    fn seal_and_drain(&mut self, _completed_layers: usize) -> std::result::Result<(), String> {
        Err("single-layer K4 owner 禁止冒充 FullDepth43 seal".into())
    }

    fn drain_and_abort(&mut self, completed_layers: usize) -> std::result::Result<(), String> {
        self.abort_inner(completed_layers)
            .map_err(|error| format!("{error:#}"))
    }

    fn finish_validated_block(&mut self) -> std::result::Result<(), String> {
        Err("single-layer K4 owner 没有可发布的 whole-token future".into())
    }

    fn destroy(&mut self) -> std::result::Result<(), String> {
        self.destroy_inner().map_err(|error| format!("{error:#}"))
    }
}

pub type S14CausalBlockConcreteSingleLayerMoeAdapter =
    S14CausalBlockProductionSingleLayerMoeAdapter<S14CausalBlockGroupedMoeVulkanRecorder>;

pub fn build_s14_causal_block_concrete_single_layer_moe_adapter(
    ctx: Arc<VulkanContext>,
    catalog: FullDepthExpertCatalog,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
    static_layer: S14CausalBlockGroupedMoeStaticLayerResources,
    union: S14CausalBlockOwnedUnionBank,
) -> Result<S14CausalBlockConcreteSingleLayerMoeAdapter> {
    if static_layer.layer != FULL_DEPTH_LAYERS[SINGLE_LAYER_INDEX] {
        bail!("single-layer concrete adapter 只接受 FullDepth43 第一层 static owner");
    }
    let recorder = S14CausalBlockGroupedMoeVulkanRecorder::new_with_static_layer(
        Arc::clone(&ctx),
        static_layer,
    )?;
    S14CausalBlockProductionSingleLayerMoeAdapter::new(
        ctx, catalog, cache_root, fetch_mode, recorder, union,
    )
}

unsafe fn transfer_to_compute_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k4_receipt_requires_exactly_two_submits_owned_buffers_and_zero_serial_forward() {
        let valid = S14CausalBlockSingleLayerTelemetry {
            route_producer_submit_calls: 1,
            fused_range_grouped_command_buffers: 1,
            fused_range_grouped_submit_calls: 1,
            total_layer_submit_calls: 2,
            layer_recorder_calls: 1,
            union_upload_copy_regions: 1,
            grouped_recorder_calls: 1,
            serial_token_forward_calls: 0,
            actual_physical_ranges: 144,
            actual_union_bytes: 320_864_256,
            union_buffer_owned: true,
            output_buffer_owned: true,
            drained: true,
        };
        valid.validate().unwrap();
        for invalid in [
            S14CausalBlockSingleLayerTelemetry {
                total_layer_submit_calls: 3,
                ..valid
            },
            S14CausalBlockSingleLayerTelemetry {
                serial_token_forward_calls: 4,
                ..valid
            },
            S14CausalBlockSingleLayerTelemetry {
                union_buffer_owned: false,
                ..valid
            },
            S14CausalBlockSingleLayerTelemetry {
                output_buffer_owned: false,
                ..valid
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
}
