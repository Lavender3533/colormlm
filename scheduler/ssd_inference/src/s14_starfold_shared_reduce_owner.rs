//! S14 StarFold B4 shared expert 与 exact slot-order reduce 的 Vulkan owner。
//!
//! routed W2 已在同一 graphics queue 提交后，本 owner 用一个 command 完成四个 lane 的
//! shared FP8 W1/W3、一次 batched official prepare、四个 shared FP8 W2，最后严格按
//! routed slot0→slot5→shared 的顺序归约。owner 常驻持有 pipeline/command/fence/workspace，
//! 每次提交新建的 descriptor pool 会一直保活到 fence 完成。

use crate::{
    compute::{ComputePipeline, DescriptorBinder, ExternalStorageBuffer},
    s14_position0_hybrid_weight_arena::S14Position0StaticLayerLayout,
    s14_position0_paged_weight_arena::{
        S14Position0PagedWeightArena, S14Position0StaticLayerBinding,
    },
    s14_starfold_b4_owner::{S14StarfoldB4RoutedLayerReceipt, S14StarfoldRoutedDownBinding},
    s14_starfold_cache::{STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_expert_schedule::{
        S14StarfoldExpertProjection, S14_STARFOLD_HIDDEN, S14_STARFOLD_INTERMEDIATE,
    },
    s14_starfold_mxfp4_tile::S14StarfoldMxfp4ExternalSlice,
    s14_vulkan::{
        S14MatvecShape, S14_BATCHED_OFFICIAL_EXPERT_PREPARE_SPV, S14_EXACT_ORDER_BLOCK_REDUCE_SPV,
        S14_FP8_MATVEC_SPV,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use std::sync::Arc;

const F32_BYTES: u64 = 4;
const WORKSPACE_MIN_ALIGNMENT: u64 = 256;
const POSITIONS: u32 = STARFOLD_B4_LANES as u32;
const ROUTED_BRANCHES_PER_POSITION: u32 = STARFOLD_TOP_K as u32;

const W1_BINDER_BASE: usize = 0;
const W3_BINDER_BASE: usize = W1_BINDER_BASE + STARFOLD_B4_LANES;
const PREPARE_BINDER: usize = W3_BINDER_BASE + STARFOLD_B4_LANES;
const W2_BINDER_BASE: usize = PREPARE_BINDER + 1;
const REDUCE_BINDER: usize = W2_BINDER_BASE + STARFOLD_B4_LANES;
const SUBMISSION_BINDERS: usize = REDUCE_BINDER + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldSharedReduceReceipt {
    pub layer: u16,
    pub base_position: u64,
    pub positions: u32,
    /// shared expert 的 route weight 恒为官方语义要求的 1.0；这里是实际写入数量。
    pub exact_shared_route_weights: u32,
    pub shared_w1_dispatch_calls: u32,
    pub shared_w3_dispatch_calls: u32,
    pub shared_prepare_dispatch_calls: u32,
    pub shared_w2_dispatch_calls: u32,
    pub exact_reduce_dispatch_calls: u32,
    pub queue_submit_calls: u32,
    pub serial_token_forward_calls: u32,
    /// 若本层来自 static stream bank，上层在此 fence 完成前不得覆盖该 bank。
    pub static_stream_bank: Option<usize>,
    /// 该 handle 在 owner 下一次提交时可能被 reset/reuse，不是永久 timeline identity。
    pub fence: vk::Fence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S14StarfoldSharedWorkspaceLayout {
    gate: u64,
    up: u64,
    route_weights: u64,
    hidden: u64,
    shared_down: u64,
    bytes: u64,
}

impl S14StarfoldSharedWorkspaceLayout {
    fn build(alignment: u64) -> Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            bail!("S14 StarFold shared workspace alignment 必须为非零二次幂");
        }
        let positions = u64::from(POSITIONS);
        let intermediate_bytes = positions
            .checked_mul(u64::from(S14_STARFOLD_INTERMEDIATE))
            .and_then(|elements| elements.checked_mul(F32_BYTES))
            .context("S14 StarFold shared intermediate bytes overflow")?;
        let down_bytes = positions
            .checked_mul(u64::from(S14_STARFOLD_HIDDEN))
            .and_then(|elements| elements.checked_mul(F32_BYTES))
            .context("S14 StarFold shared down bytes overflow")?;
        let route_weight_bytes = positions
            .checked_mul(F32_BYTES)
            .context("S14 StarFold shared route-weight bytes overflow")?;

        let mut cursor = 0u64;
        let gate = take(&mut cursor, intermediate_bytes, alignment)?;
        let up = take(&mut cursor, intermediate_bytes, alignment)?;
        let route_weights = take(&mut cursor, route_weight_bytes, alignment)?;
        let hidden = take(&mut cursor, intermediate_bytes, alignment)?;
        let shared_down = take(&mut cursor, down_bytes, alignment)?;
        let bytes = align_up(cursor, alignment)?;
        Ok(Self {
            gate,
            up,
            route_weights,
            hidden,
            shared_down,
            bytes,
        })
    }
}

#[derive(Clone, Copy)]
struct ResolvedStaticSharedWeights {
    buffer: vk::Buffer,
    logical_bytes: u64,
    stream_bank: Option<usize>,
    w1_weight: u64,
    w1_scale: u64,
    w3_weight: u64,
    w3_scale: u64,
    w2_weight: u64,
    w2_scale: u64,
}

pub struct S14StarfoldSharedReduceOwner {
    context: Arc<VulkanContext>,
    /// 强持有权重 arena，保证 resident/static-stream buffer handle 的生命周期。
    static_arena: Arc<S14Position0PagedWeightArena>,
    workspace: Option<GpuBuffer>,
    workspace_layout: S14StarfoldSharedWorkspaceLayout,
    fp8_pipeline: Option<ComputePipeline>,
    prepare_pipeline: Option<ComputePipeline>,
    reduce_pipeline: Option<ComputePipeline>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    binders: Vec<DescriptorBinder>,
    in_flight: bool,
    destroyed: bool,
}

impl S14StarfoldSharedReduceOwner {
    pub fn new(
        context: Arc<VulkanContext>,
        static_arena: Arc<S14Position0PagedWeightArena>,
    ) -> Result<Self> {
        let alignment = storage_alignment(&context).max(WORKSPACE_MIN_ALIGNMENT);
        let workspace_layout = S14StarfoldSharedWorkspaceLayout::build(alignment)?;
        let workspace = GpuBuffer::new_vram(
            &context,
            workspace_layout.bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .context("分配 S14 StarFold shared/reduce workspace")?;

        let fp8_pipeline = match ComputePipeline::new(&context, S14_FP8_MATVEC_SPV, 4, 8)
            .context("创建 S14 StarFold shared FP8 pipeline")
        {
            Ok(pipeline) => pipeline,
            Err(error) => {
                workspace.destroy(&context);
                return Err(error);
            }
        };
        let prepare_pipeline =
            match ComputePipeline::new(&context, S14_BATCHED_OFFICIAL_EXPERT_PREPARE_SPV, 4, 8)
                .context("创建 S14 StarFold shared prepare pipeline")
            {
                Ok(pipeline) => pipeline,
                Err(error) => {
                    fp8_pipeline.destroy(&context);
                    workspace.destroy(&context);
                    return Err(error);
                }
            };
        let reduce_pipeline =
            match ComputePipeline::new(&context, S14_EXACT_ORDER_BLOCK_REDUCE_SPV, 3, 4)
                .context("创建 S14 StarFold exact reduce pipeline")
            {
                Ok(pipeline) => pipeline,
                Err(error) => {
                    prepare_pipeline.destroy(&context);
                    fp8_pipeline.destroy(&context);
                    workspace.destroy(&context);
                    return Err(error);
                }
            };

        let command_pool = match unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .context("创建 S14 StarFold shared/reduce command pool")
        {
            Ok(pool) => pool,
            Err(error) => {
                reduce_pipeline.destroy(&context);
                prepare_pipeline.destroy(&context);
                fp8_pipeline.destroy(&context);
                workspace.destroy(&context);
                return Err(error);
            }
        };
        let command_buffer = match unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .context("分配 S14 StarFold shared/reduce command buffer")
        {
            Ok(buffers) if buffers.len() == 1 => buffers[0],
            Ok(_) => {
                unsafe { context.device.destroy_command_pool(command_pool, None) };
                reduce_pipeline.destroy(&context);
                prepare_pipeline.destroy(&context);
                fp8_pipeline.destroy(&context);
                workspace.destroy(&context);
                bail!("S14 StarFold shared/reduce command buffer 数量漂移");
            }
            Err(error) => {
                unsafe { context.device.destroy_command_pool(command_pool, None) };
                reduce_pipeline.destroy(&context);
                prepare_pipeline.destroy(&context);
                fp8_pipeline.destroy(&context);
                workspace.destroy(&context);
                return Err(error);
            }
        };
        let fence = match unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("创建 S14 StarFold shared/reduce fence")
        {
            Ok(fence) => fence,
            Err(error) => {
                unsafe { context.device.destroy_command_pool(command_pool, None) };
                reduce_pipeline.destroy(&context);
                prepare_pipeline.destroy(&context);
                fp8_pipeline.destroy(&context);
                workspace.destroy(&context);
                return Err(error);
            }
        };

        Ok(Self {
            context,
            static_arena,
            workspace: Some(workspace),
            workspace_layout,
            fp8_pipeline: Some(fp8_pipeline),
            prepare_pipeline: Some(prepare_pipeline),
            reduce_pipeline: Some(reduce_pipeline),
            command_pool,
            command_buffer,
            fence,
            binders: Vec::with_capacity(SUBMISSION_BINDERS),
            in_flight: false,
            destroyed: false,
        })
    }

    /// 在 routed owner 的 W2 submission 之后调用。两者都提交到 `q_graphics`，本 command
    /// 开头的 compute write→read barrier 与 queue submission order 一起建立真实依赖。
    ///
    /// 对 streamed static layer，调用方在回执 fence 完成前不得向 `static_stream_bank`
    /// 指定的 bank 上传下一层；arena 当前没有独立 pin API，owner 只能保证 handle 保活。
    pub fn submit_after_routed_w2(
        &mut self,
        routed_receipt: &S14StarfoldB4RoutedLayerReceipt,
        input_f32: S14StarfoldMxfp4ExternalSlice,
        routed_down: S14StarfoldRoutedDownBinding,
        output_f32: S14StarfoldMxfp4ExternalSlice,
    ) -> Result<S14StarfoldSharedReduceReceipt> {
        if self.destroyed {
            bail!("S14 StarFold shared/reduce owner 已销毁");
        }
        self.wait_previous()?;
        validate_routed_receipt(routed_receipt)?;

        let input_bytes = b4_hidden_bytes()?;
        let routed_bytes = routed_down_bytes()?;
        let output_bytes = input_bytes;
        let intermediate_bytes = shared_intermediate_bytes()?;
        validate_external_slice(input_f32, input_bytes, "B4 shared input")?;
        validate_routed_down(routed_down, routed_bytes)?;
        validate_external_slice(output_f32, output_bytes, "B4 exact-reduce output")?;
        require_non_overlapping(
            routed_down.branches,
            routed_bytes,
            output_f32,
            output_bytes,
            "routed down",
            "exact-reduce output",
        )?;
        require_non_overlapping(
            input_f32,
            input_bytes,
            routed_down.branches,
            routed_bytes,
            "shared input",
            "routed down",
        )?;

        let layer = u8::try_from(routed_receipt.layer)
            .context("S14 StarFold shared/reduce layer 超出 u8")?;
        let static_weights = resolve_static_shared_weights(&self.static_arena, layer)?;
        let alignment = storage_alignment(&self.context);
        validate_required_offsets(
            alignment,
            input_f32,
            routed_down.branches,
            output_f32,
            static_weights,
            self.workspace_layout,
        )?;

        let workspace = self
            .workspace
            .as_ref()
            .context("S14 StarFold shared/reduce workspace 已销毁")?;
        let fp8_pipeline = self
            .fp8_pipeline
            .as_ref()
            .context("S14 StarFold shared FP8 pipeline 已销毁")?;
        let prepare_pipeline = self
            .prepare_pipeline
            .as_ref()
            .context("S14 StarFold shared prepare pipeline 已销毁")?;
        let reduce_pipeline = self
            .reduce_pipeline
            .as_ref()
            .context("S14 StarFold exact reduce pipeline 已销毁")?;
        let binders = create_submission_binders(
            &self.context,
            fp8_pipeline,
            prepare_pipeline,
            reduce_pipeline,
            workspace,
            self.workspace_layout,
            static_weights,
            input_f32,
            routed_down.branches,
            output_f32,
        )?;

        if let Err(error) = unsafe {
            self.context
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .context("reset S14 StarFold shared/reduce command buffer")
        {
            destroy_binders(&self.context, binders);
            return Err(error);
        }
        if let Err(error) = unsafe {
            self.context.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .context("begin S14 StarFold shared/reduce command buffer")
        {
            destroy_binders(&self.context, binders);
            return Err(error);
        }

        let route_weight_bytes = exact_shared_route_weight_bytes();
        unsafe {
            predecessor_compute_write_to_read_barrier(
                &self.context,
                self.command_buffer,
                routed_down.branches.buffer,
                routed_down.branches.offset,
                routed_bytes,
            );
            self.context.device.cmd_update_buffer(
                self.command_buffer,
                workspace.handle(),
                self.workspace_layout.route_weights,
                &route_weight_bytes,
            );
            transfer_write_to_compute_read_barrier(
                &self.context,
                self.command_buffer,
                workspace.handle(),
                self.workspace_layout.route_weights,
                route_weight_bytes.len() as u64,
            );

            let w1_shape = S14MatvecShape {
                n: S14_STARFOLD_INTERMEDIATE,
                k: S14_STARFOLD_HIDDEN,
            };
            let w2_shape = S14MatvecShape {
                n: S14_STARFOLD_HIDDEN,
                k: S14_STARFOLD_INTERMEDIATE,
            };
            for lane in 0..STARFOLD_B4_LANES {
                record_fp8_matvec(
                    &self.context,
                    self.command_buffer,
                    fp8_pipeline,
                    binders[W1_BINDER_BASE + lane].set,
                    w1_shape,
                );
            }
            for lane in 0..STARFOLD_B4_LANES {
                record_fp8_matvec(
                    &self.context,
                    self.command_buffer,
                    fp8_pipeline,
                    binders[W3_BINDER_BASE + lane].set,
                    w1_shape,
                );
            }
            compute_write_to_read_barriers(
                &self.context,
                self.command_buffer,
                &[
                    (
                        workspace.handle(),
                        self.workspace_layout.gate,
                        intermediate_bytes,
                    ),
                    (
                        workspace.handle(),
                        self.workspace_layout.up,
                        intermediate_bytes,
                    ),
                ],
            );
            record_batched_prepare(
                &self.context,
                self.command_buffer,
                prepare_pipeline,
                binders[PREPARE_BINDER].set,
                POSITIONS,
                S14_STARFOLD_INTERMEDIATE,
            );
            compute_write_to_read_barriers(
                &self.context,
                self.command_buffer,
                &[(
                    workspace.handle(),
                    self.workspace_layout.hidden,
                    intermediate_bytes,
                )],
            );
            for lane in 0..STARFOLD_B4_LANES {
                record_fp8_matvec(
                    &self.context,
                    self.command_buffer,
                    fp8_pipeline,
                    binders[W2_BINDER_BASE + lane].set,
                    w2_shape,
                );
            }
            compute_write_to_read_barriers(
                &self.context,
                self.command_buffer,
                &[(
                    workspace.handle(),
                    self.workspace_layout.shared_down,
                    output_bytes,
                )],
            );
            record_exact_reduce(
                &self.context,
                self.command_buffer,
                reduce_pipeline,
                binders[REDUCE_BINDER].set,
                POSITIONS,
            );
            publish_compute_output_barrier(
                &self.context,
                self.command_buffer,
                output_f32.buffer,
                output_f32.offset,
                output_bytes,
            );
        }

        if let Err(error) = unsafe { self.context.device.end_command_buffer(self.command_buffer) }
            .context("end S14 StarFold shared/reduce command buffer")
        {
            destroy_binders(&self.context, binders);
            return Err(error);
        }
        if let Err(error) = unsafe {
            self.context.device.queue_submit(
                self.context.q_graphics,
                &[vk::SubmitInfo::default()
                    .command_buffers(std::slice::from_ref(&self.command_buffer))],
                self.fence,
            )
        }
        .context("提交 S14 StarFold shared/reduce command")
        {
            destroy_binders(&self.context, binders);
            return Err(error);
        }

        debug_assert!(self.binders.is_empty());
        self.binders = binders;
        self.in_flight = true;
        Ok(S14StarfoldSharedReduceReceipt {
            layer: routed_receipt.layer,
            base_position: routed_receipt.base_position,
            positions: POSITIONS,
            exact_shared_route_weights: POSITIONS,
            shared_w1_dispatch_calls: POSITIONS,
            shared_w3_dispatch_calls: POSITIONS,
            shared_prepare_dispatch_calls: 1,
            shared_w2_dispatch_calls: POSITIONS,
            exact_reduce_dispatch_calls: 1,
            queue_submit_calls: 1,
            serial_token_forward_calls: 0,
            static_stream_bank: static_weights.stream_bank,
            fence: self.fence,
        })
    }

    /// 等待当前 submission，并在 fence 完成后释放本次 descriptor pools。
    pub fn wait(&mut self) -> Result<()> {
        if self.destroyed {
            bail!("S14 StarFold shared/reduce owner 已销毁");
        }
        self.wait_previous()
    }

    pub fn try_destroy(&mut self) -> Result<()> {
        if self.destroyed {
            return Ok(());
        }
        if self.in_flight {
            let complete = unsafe { self.context.device.get_fence_status(self.fence) }
                .context("查询 S14 StarFold shared/reduce fence")?;
            if !complete {
                bail!("S14 StarFold shared/reduce command 仍在执行");
            }
            self.finish_previous()?;
        }
        self.destroy_resources();
        Ok(())
    }

    fn wait_previous(&mut self) -> Result<()> {
        if !self.in_flight {
            return Ok(());
        }
        unsafe {
            self.context
                .device
                .wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)
        }
        .context("等待 S14 StarFold shared/reduce fence")?;
        self.finish_previous()
    }

    fn finish_previous(&mut self) -> Result<()> {
        let binder_count = self.binders.len();
        unsafe {
            self.context
                .device
                .reset_fences(std::slice::from_ref(&self.fence))
        }
        .context("reset S14 StarFold shared/reduce fence")?;
        for binder in self.binders.drain(..).rev() {
            binder.destroy(&self.context);
        }
        self.in_flight = false;
        if binder_count != SUBMISSION_BINDERS {
            bail!("S14 StarFold shared/reduce descriptor lifetime owner 数量漂移");
        }
        Ok(())
    }

    fn destroy_resources(&mut self) {
        if self.destroyed {
            return;
        }
        for binder in self.binders.drain(..).rev() {
            binder.destroy(&self.context);
        }
        unsafe {
            self.context.device.destroy_fence(self.fence, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        if let Some(pipeline) = self.reduce_pipeline.take() {
            pipeline.destroy(&self.context);
        }
        if let Some(pipeline) = self.prepare_pipeline.take() {
            pipeline.destroy(&self.context);
        }
        if let Some(pipeline) = self.fp8_pipeline.take() {
            pipeline.destroy(&self.context);
        }
        if let Some(workspace) = self.workspace.take() {
            workspace.destroy(&self.context);
        }
        self.fence = vk::Fence::null();
        self.command_pool = vk::CommandPool::null();
        self.command_buffer = vk::CommandBuffer::null();
        self.in_flight = false;
        self.destroyed = true;
    }
}

impl Drop for S14StarfoldSharedReduceOwner {
    fn drop(&mut self) {
        if self.destroyed {
            return;
        }
        let _ = unsafe { self.context.device.device_wait_idle() };
        self.destroy_resources();
    }
}

#[allow(clippy::too_many_arguments)]
fn create_submission_binders(
    context: &VulkanContext,
    fp8_pipeline: &ComputePipeline,
    prepare_pipeline: &ComputePipeline,
    reduce_pipeline: &ComputePipeline,
    workspace: &GpuBuffer,
    layout: S14StarfoldSharedWorkspaceLayout,
    weights: ResolvedStaticSharedWeights,
    input: S14StarfoldMxfp4ExternalSlice,
    routed_down: S14StarfoldMxfp4ExternalSlice,
    output: S14StarfoldMxfp4ExternalSlice,
) -> Result<Vec<DescriptorBinder>> {
    let mut binders = Vec::with_capacity(SUBMISSION_BINDERS);
    let result = (|| -> Result<()> {
        let w1_shape =
            S14MatvecShape::new(S14_STARFOLD_INTERMEDIATE, S14_STARFOLD_HIDDEN)?.validate_fp8()?;
        let w2_shape =
            S14MatvecShape::new(S14_STARFOLD_HIDDEN, S14_STARFOLD_INTERMEDIATE)?.validate_fp8()?;
        let input_lane_bytes = w1_shape.fp32_input_bytes()?;
        let intermediate_lane_bytes = w1_shape.fp32_output_bytes()?;
        let down_lane_bytes = w2_shape.fp32_output_bytes()?;
        let workspace_view = ExternalStorageBuffer {
            buffer: workspace.handle(),
            capacity: layout.bytes,
        };
        let static_view = ExternalStorageBuffer {
            buffer: weights.buffer,
            capacity: weights.logical_bytes,
        };
        let input_view = external(input);

        for (weight_offset, scale_offset, output_base) in [
            (weights.w1_weight, weights.w1_scale, layout.gate),
            (weights.w3_weight, weights.w3_scale, layout.up),
        ] {
            for lane in 0..STARFOLD_B4_LANES {
                let lane = lane as u64;
                let input_offset = input
                    .offset
                    .checked_add(lane * input_lane_bytes)
                    .context("S14 StarFold shared input lane offset overflow")?;
                let output_offset = output_base
                    .checked_add(lane * intermediate_lane_bytes)
                    .context("S14 StarFold shared W1/W3 lane offset overflow")?;
                binders.push(
                    DescriptorBinder::new_with_external_offsets(
                        context,
                        fp8_pipeline,
                        &[
                            (input_view, input_offset, input_lane_bytes),
                            (static_view, weight_offset, w1_shape.fp8_weight_bytes()?),
                            (static_view, scale_offset, w1_shape.fp8_scale_bytes()?),
                            (workspace_view, output_offset, intermediate_lane_bytes),
                        ],
                    )
                    .context("绑定 S14 StarFold shared W1/W3 FP8 buffers")?,
                );
            }
        }

        binders.push(
            DescriptorBinder::new_with_external_offsets(
                context,
                prepare_pipeline,
                &[
                    (workspace_view, layout.gate, shared_intermediate_bytes()?),
                    (workspace_view, layout.up, shared_intermediate_bytes()?),
                    (
                        workspace_view,
                        layout.route_weights,
                        u64::from(POSITIONS) * F32_BYTES,
                    ),
                    (workspace_view, layout.hidden, shared_intermediate_bytes()?),
                ],
            )
            .context("绑定 S14 StarFold shared batched prepare buffers")?,
        );

        for lane in 0..STARFOLD_B4_LANES {
            let lane = lane as u64;
            let input_offset = layout
                .hidden
                .checked_add(lane * intermediate_lane_bytes)
                .context("S14 StarFold shared W2 input lane offset overflow")?;
            let output_offset = layout
                .shared_down
                .checked_add(lane * down_lane_bytes)
                .context("S14 StarFold shared W2 output lane offset overflow")?;
            binders.push(
                DescriptorBinder::new_with_external_offsets(
                    context,
                    fp8_pipeline,
                    &[
                        (workspace_view, input_offset, intermediate_lane_bytes),
                        (static_view, weights.w2_weight, w2_shape.fp8_weight_bytes()?),
                        (static_view, weights.w2_scale, w2_shape.fp8_scale_bytes()?),
                        (workspace_view, output_offset, down_lane_bytes),
                    ],
                )
                .context("绑定 S14 StarFold shared W2 FP8 buffers")?,
            );
        }

        binders.push(
            DescriptorBinder::new_with_external_offsets(
                context,
                reduce_pipeline,
                &[
                    (
                        external(routed_down),
                        routed_down.offset,
                        routed_down_bytes()?,
                    ),
                    (workspace_view, layout.shared_down, b4_hidden_bytes()?),
                    (external(output), output.offset, b4_hidden_bytes()?),
                ],
            )
            .context("绑定 S14 StarFold exact slot-order reduce buffers")?,
        );
        if binders.len() != SUBMISSION_BINDERS {
            bail!("S14 StarFold shared/reduce descriptor 数量漂移");
        }
        Ok(())
    })();
    match result {
        Ok(()) => Ok(binders),
        Err(error) => {
            destroy_binders(context, binders);
            Err(error)
        }
    }
}

fn validate_routed_receipt(receipt: &S14StarfoldB4RoutedLayerReceipt) -> Result<()> {
    let identity_matches = receipt.w1.layer == receipt.layer
        && receipt.w3.layer == receipt.layer
        && receipt.prepare.layer == receipt.layer
        && receipt.w2.layer == receipt.layer
        && receipt.w1.base_position == receipt.base_position
        && receipt.w3.base_position == receipt.base_position
        && receipt.prepare.base_position == receipt.base_position
        && receipt.w2.base_position == receipt.base_position;
    let projections_match = receipt.w1.projection == S14StarfoldExpertProjection::W1
        && receipt.w3.projection == S14StarfoldExpertProjection::W3
        && receipt.w2.projection == S14StarfoldExpertProjection::W2;
    let submissions_exist = receipt.w1.queue_submit_calls > 0
        && receipt.w3.queue_submit_calls > 0
        && receipt.prepare.queue_submit_calls > 0
        && receipt.w2.queue_submit_calls > 0;
    let expert_identity_matches = receipt.w1.unique_experts == receipt.unique_experts
        && receipt.w3.unique_experts == receipt.unique_experts
        && receipt.w2.unique_experts == receipt.unique_experts;
    let no_serial_fallback = receipt.serial_token_forward_calls == 0
        && receipt.w1.serial_token_forward_calls == 0
        && receipt.w3.serial_token_forward_calls == 0
        && receipt.prepare.serial_token_forward_calls == 0
        && receipt.w2.serial_token_forward_calls == 0;
    if !identity_matches
        || !projections_match
        || !submissions_exist
        || !expert_identity_matches
        || !no_serial_fallback
        || receipt.unique_experts == 0
        || receipt.packed_uploads == 0
        || receipt.lane_dispatches == 0
        || receipt.prepare.exact_route_weights != (STARFOLD_B4_LANES * STARFOLD_TOP_K) as u32
        || receipt.prepare.prepare_dispatch_calls != 1
    {
        bail!("S14 StarFold shared/reduce 缺少同层 routed W2 完成提交凭据");
    }
    Ok(())
}

fn validate_routed_down(binding: S14StarfoldRoutedDownBinding, required: u64) -> Result<()> {
    if binding.positions != POSITIONS
        || binding.branches_per_position != ROUTED_BRANCHES_PER_POSITION
    {
        bail!("S14 StarFold exact reduce 要求精确 B4×top-6 routed down");
    }
    validate_external_slice(binding.branches, required, "B4×top-6 routed down")
}

fn validate_external_slice(
    slice: S14StarfoldMxfp4ExternalSlice,
    required: u64,
    label: &str,
) -> Result<()> {
    if slice.buffer == vk::Buffer::null()
        || slice.capacity_bytes == 0
        || slice.logical_bytes < required
    {
        bail!("S14 StarFold {label} external slice handle/capacity/payload 非法");
    }
    let end = slice
        .offset
        .checked_add(slice.logical_bytes)
        .with_context(|| format!("S14 StarFold {label} range overflow"))?;
    if end > slice.capacity_bytes {
        bail!("S14 StarFold {label} external slice 越界");
    }
    Ok(())
}

fn require_non_overlapping(
    left: S14StarfoldMxfp4ExternalSlice,
    left_bytes: u64,
    right: S14StarfoldMxfp4ExternalSlice,
    right_bytes: u64,
    left_label: &str,
    right_label: &str,
) -> Result<()> {
    if left.buffer != right.buffer {
        return Ok(());
    }
    let left_end = left
        .offset
        .checked_add(left_bytes)
        .with_context(|| format!("S14 StarFold {left_label} alias range overflow"))?;
    let right_end = right
        .offset
        .checked_add(right_bytes)
        .with_context(|| format!("S14 StarFold {right_label} alias range overflow"))?;
    if left.offset < right_end && right.offset < left_end {
        bail!("S14 StarFold {left_label}/{right_label} binding 重叠");
    }
    Ok(())
}

fn resolve_static_shared_weights(
    arena: &S14Position0PagedWeightArena,
    layer: u8,
) -> Result<ResolvedStaticSharedWeights> {
    let ready = arena.ready_static_layer(layer)?;
    let (buffer, layout, stream_bank) = match ready {
        S14Position0StaticLayerBinding::Resident { buffer, layout } => (buffer, layout, None),
        S14Position0StaticLayerBinding::Streamed {
            bank,
            buffer,
            layout,
        } => (buffer, layout, Some(bank)),
    };
    if layout.layer != layer
        || layout.requested_bytes == 0
        || layout.requested_bytes > buffer.size()
    {
        bail!("S14 StarFold ready static layer identity/capacity 漂移");
    }
    let w1_shape =
        S14MatvecShape::new(S14_STARFOLD_INTERMEDIATE, S14_STARFOLD_HIDDEN)?.validate_fp8()?;
    let w2_shape =
        S14MatvecShape::new(S14_STARFOLD_HIDDEN, S14_STARFOLD_INTERMEDIATE)?.validate_fp8()?;
    Ok(ResolvedStaticSharedWeights {
        buffer: buffer.handle(),
        logical_bytes: layout.requested_bytes,
        stream_bank,
        w1_weight: static_asset(
            layout,
            layer,
            "ffn.shared_experts.w1.weight",
            w1_shape.fp8_weight_bytes()?,
        )?,
        w1_scale: static_asset(
            layout,
            layer,
            "ffn.shared_experts.w1.scale",
            w1_shape.fp8_scale_bytes()?,
        )?,
        w3_weight: static_asset(
            layout,
            layer,
            "ffn.shared_experts.w3.weight",
            w1_shape.fp8_weight_bytes()?,
        )?,
        w3_scale: static_asset(
            layout,
            layer,
            "ffn.shared_experts.w3.scale",
            w1_shape.fp8_scale_bytes()?,
        )?,
        w2_weight: static_asset(
            layout,
            layer,
            "ffn.shared_experts.w2.weight",
            w2_shape.fp8_weight_bytes()?,
        )?,
        w2_scale: static_asset(
            layout,
            layer,
            "ffn.shared_experts.w2.scale",
            w2_shape.fp8_scale_bytes()?,
        )?,
    })
}

fn static_asset(
    layout: &S14Position0StaticLayerLayout,
    layer: u8,
    suffix: &str,
    expected_bytes: u64,
) -> Result<u64> {
    let tensor = format!("layers.{layer}.{suffix}");
    let mut matches = layout.assets.iter().filter(|asset| asset.tensor == tensor);
    let asset = matches
        .next()
        .with_context(|| format!("S14 StarFold static layer 缺少 {tensor}"))?;
    if matches.next().is_some() {
        bail!("S14 StarFold static layer 重复资产: {tensor}");
    }
    let end = asset
        .local_offset
        .checked_add(asset.bytes)
        .with_context(|| format!("S14 StarFold static asset range overflow: {tensor}"))?;
    if asset.bytes != expected_bytes || end > layout.requested_bytes {
        bail!("S14 StarFold static asset bytes/capacity 漂移: {tensor}");
    }
    Ok(asset.local_offset)
}

fn validate_required_offsets(
    alignment: u64,
    input: S14StarfoldMxfp4ExternalSlice,
    routed: S14StarfoldMxfp4ExternalSlice,
    output: S14StarfoldMxfp4ExternalSlice,
    weights: ResolvedStaticSharedWeights,
    layout: S14StarfoldSharedWorkspaceLayout,
) -> Result<()> {
    for (offset, label) in [
        (input.offset, "shared input"),
        (routed.offset, "routed down"),
        (output.offset, "exact-reduce output"),
        (weights.w1_weight, "shared W1 weight"),
        (weights.w1_scale, "shared W1 scale"),
        (weights.w3_weight, "shared W3 weight"),
        (weights.w3_scale, "shared W3 scale"),
        (weights.w2_weight, "shared W2 weight"),
        (weights.w2_scale, "shared W2 scale"),
        (layout.gate, "workspace gate"),
        (layout.up, "workspace up"),
        (layout.route_weights, "workspace route weights"),
        (layout.hidden, "workspace hidden"),
        (layout.shared_down, "workspace shared down"),
    ] {
        if offset % alignment != 0 {
            bail!(
                "S14 StarFold {label} descriptor offset 未对齐: offset={offset} alignment={alignment}"
            );
        }
    }
    Ok(())
}

fn exact_shared_route_weight_bytes() -> [u8; 16] {
    let mut bytes = [0u8; 16];
    for chunk in bytes.chunks_exact_mut(4) {
        chunk.copy_from_slice(&1.0f32.to_bits().to_le_bytes());
    }
    bytes
}

fn external(slice: S14StarfoldMxfp4ExternalSlice) -> ExternalStorageBuffer {
    ExternalStorageBuffer {
        buffer: slice.buffer,
        capacity: slice.capacity_bytes,
    }
}

fn shared_intermediate_bytes() -> Result<u64> {
    u64::from(POSITIONS)
        .checked_mul(u64::from(S14_STARFOLD_INTERMEDIATE))
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .context("S14 StarFold shared intermediate bytes overflow")
}

fn b4_hidden_bytes() -> Result<u64> {
    u64::from(POSITIONS)
        .checked_mul(u64::from(S14_STARFOLD_HIDDEN))
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .context("S14 StarFold B4 hidden bytes overflow")
}

fn routed_down_bytes() -> Result<u64> {
    u64::from(POSITIONS)
        .checked_mul(u64::from(ROUTED_BRANCHES_PER_POSITION))
        .and_then(|branches| branches.checked_mul(u64::from(S14_STARFOLD_HIDDEN)))
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .context("S14 StarFold routed down bytes overflow")
}

fn storage_alignment(context: &VulkanContext) -> u64 {
    unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical)
            .limits
            .min_storage_buffer_offset_alignment
    }
    .max(1)
}

fn take(cursor: &mut u64, bytes: u64, alignment: u64) -> Result<u64> {
    let offset = align_up(*cursor, alignment)?;
    *cursor = offset
        .checked_add(bytes)
        .context("S14 StarFold shared workspace region overflow")?;
    Ok(offset)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(anyhow!("S14 StarFold alignment 必须为非零二次幂"));
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("S14 StarFold alignment overflow")
}

fn destroy_binders(context: &VulkanContext, binders: Vec<DescriptorBinder>) {
    for binder in binders.into_iter().rev() {
        binder.destroy(context);
    }
}

unsafe fn record_fp8_matvec(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    shape: S14MatvecShape,
) {
    unsafe {
        context.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        context.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[set],
            &[],
        );
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&shape.n.to_le_bytes());
        push[4..].copy_from_slice(&shape.k.to_le_bytes());
        context.device.cmd_push_constants(
            command,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        context.device.cmd_dispatch(command, shape.n, 1, 1);
    }
}

unsafe fn record_batched_prepare(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    branches: u32,
    n: u32,
) {
    unsafe {
        context.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        context.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[set],
            &[],
        );
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&branches.to_le_bytes());
        push[4..].copy_from_slice(&n.to_le_bytes());
        context.device.cmd_push_constants(
            command,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        context
            .device
            .cmd_dispatch(command, n.div_ceil(128), branches, 1);
    }
}

unsafe fn record_exact_reduce(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    positions: u32,
) {
    unsafe {
        context.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.pipeline,
        );
        context.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            0,
            &[set],
            &[],
        );
        context.device.cmd_push_constants(
            command,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &positions.to_le_bytes(),
        );
        let output_elements = positions * S14_STARFOLD_HIDDEN;
        context
            .device
            .cmd_dispatch(command, output_elements.div_ceil(256), 1, 1);
    }
}

unsafe fn predecessor_compute_write_to_read_barrier(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    bytes: u64,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(bytes);
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&barrier),
            &[],
        );
    }
}

unsafe fn transfer_write_to_compute_read_barrier(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    bytes: u64,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(bytes);
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&barrier),
            &[],
        );
    }
}

unsafe fn compute_write_to_read_barriers(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    ranges: &[(vk::Buffer, u64, u64)],
) {
    let barriers = ranges
        .iter()
        .map(|(buffer, offset, bytes)| {
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(*buffer)
                .offset(*offset)
                .size(*bytes)
        })
        .collect::<Vec<_>>();
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &barriers,
            &[],
        );
    }
}

unsafe fn publish_compute_output_barrier(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    bytes: u64,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(bytes);
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&barrier),
            &[],
        );
    }
}
