//! 任意已提交 base_position、K=4 block-major prefix checkpoint 的真实 Vulkan producer。
//!
//! 每层 HC/QKV 得到 K 行 `hc_branch_f32` 与已按各自绝对 position 完成 RoPE 的 KV 后，
//! 本 producer 在同一个 command buffer 中把 source lane `s` 的写集累积到 prefix
//! `s..K`。compressor 投影每个
//! source lane 只计算一次，随后把正式 recipe 的 writeback 行复制到其余 prefix。ratio4
//! 的 lane0/1 在这里闭合；lane2/3 只先写 window KV，并由 boundary state owner 在同一
//! command 中完成 remainder/finalize/rollover。没有结构 receipt 能替代这些 Vulkan 录制。

use crate::{
    compute::{DescriptorBinder, StorageBufferSlice},
    s14_causal_block_prefix_arena::S14CausalBlockPrefixCheckpointArena,
    s14_causal_block_prefix_state::S14CausalBlockPrefixStateProgram,
    s14_causal_block_production_evidence::{
        S14CausalBlockProductionEvidenceSnapshot, S14CausalBlockRatio4SegmentedRecordingReceipt,
    },
    s14_position0_state_writeback::{
        S14Position0ApeAddPipeline, S14Position0StateRecordingOp, S14Position0StateRowKind,
    },
    s14_vulkan::S14NumericPipelines,
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{MaterializedTokenSource, COMPRESS_RATIOS, FULL_DEPTH_LAYERS};
use std::{
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};

const K4: usize = 4;
const K8: usize = 8;
const HIDDEN: u64 = 4096;
const KV: u64 = 512;
const F32_BYTES: u64 = 4;
const BF16_BYTES: u64 = 2;

pub type S14CausalBlockSharedPrefixStateProgram = Arc<Mutex<S14CausalBlockPrefixStateProgram>>;

#[derive(Clone, Copy)]
pub struct S14CausalBlockPrefixLayerInputs<'a> {
    pub command: vk::CommandBuffer,
    pub layer: u8,
    pub static_weights: StorageBufferSlice<'a>,
    pub static_logical_bytes: u64,
    pub hc_branch_f32: StorageBufferSlice<'a>,
    /// 已按 `base_position + lane` 完成 position RoPE 的 K 行 KV。raw KV 只允许
    /// 供本轮 attention 使用，禁止写入跨 prefix/window 的历史状态。
    pub rotated_current_kv_bf16: StorageBufferSlice<'a>,
}

impl fmt::Debug for S14CausalBlockPrefixLayerInputs<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockPrefixLayerInputs")
            .field("command", &self.command)
            .field("layer", &self.layer)
            .field("static_weights", &self.static_weights.buffer.handle())
            .field("static_logical_bytes", &self.static_logical_bytes)
            .field("hc_branch_f32", &self.hc_branch_f32.buffer.handle())
            .field(
                "rotated_current_kv_bf16",
                &self.rotated_current_kv_bf16.buffer.handle(),
            )
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockPrefixLayerRecordingReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub layer: u8,
    pub window_writebacks: usize,
    pub completed_lane_applications: usize,
    pub deferred_ratio4_lane_applications: usize,
    pub compressor_source_lane_evaluations: usize,
    pub serial_token_forward_calls: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProducerPhase {
    Ready { next_layer_index: usize },
    Sealed,
    Poisoned,
    Destroyed,
}

/// 强持有 scratch、pipeline、prefix arena 与共享顺序 program。调用方必须保证
/// `release_completed_layer` 只在包含本层录制的 fence 完成后执行。
pub struct S14CausalBlockPrefixStateProducer {
    context: Arc<VulkanContext>,
    base_position: u32,
    block_size: usize,
    source: MaterializedTokenSource,
    arena: Arc<S14CausalBlockPrefixCheckpointArena>,
    program: S14CausalBlockSharedPrefixStateProgram,
    workspace: Option<GpuBuffer>,
    numeric: Option<S14NumericPipelines>,
    ape: Option<S14Position0ApeAddPipeline>,
    pending_binders: Mutex<Vec<DescriptorBinder>>,
    phase: Mutex<ProducerPhase>,
}

impl fmt::Debug for S14CausalBlockPrefixStateProducer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockPrefixStateProducer")
            .field("context", &Arc::as_ptr(&self.context))
            .field("base_position", &self.base_position)
            .field("arena", &Arc::as_ptr(&self.arena))
            .field("workspace", &self.workspace.as_ref().map(GpuBuffer::handle))
            .field("phase", &self.phase.lock().ok().map(|phase| *phase))
            .finish_non_exhaustive()
    }
}

impl S14CausalBlockPrefixStateProducer {
    pub fn new(
        context: Arc<VulkanContext>,
        arena: Arc<S14CausalBlockPrefixCheckpointArena>,
        program: S14CausalBlockSharedPrefixStateProgram,
    ) -> Result<Self> {
        Self::new_with_source(
            context,
            arena,
            program,
            MaterializedTokenSource::SpeculativeDraft,
        )
    }

    /// base0 只由显式 teacher-forced bootstrap 构造；普通构造器仍保持 draft/nonzero 合同。
    pub fn new_forced_prefill(
        context: Arc<VulkanContext>,
        arena: Arc<S14CausalBlockPrefixCheckpointArena>,
        program: S14CausalBlockSharedPrefixStateProgram,
    ) -> Result<Self> {
        Self::new_with_source(
            context,
            arena,
            program,
            MaterializedTokenSource::ForcedPrefill,
        )
    }

    fn new_with_source(
        context: Arc<VulkanContext>,
        arena: Arc<S14CausalBlockPrefixCheckpointArena>,
        program: S14CausalBlockSharedPrefixStateProgram,
        source: MaterializedTokenSource,
    ) -> Result<Self> {
        let base_position = arena.base_position();
        let block_size = arena.layout().block_size;
        if !Arc::ptr_eq(&context, arena.context())
            || (source == MaterializedTokenSource::SpeculativeDraft && base_position == 0)
            || !matches!(block_size, K4 | K8)
            || (source == MaterializedTokenSource::SpeculativeDraft && block_size != K4)
        {
            bail!("K-block prefix producer context/base/K/source 与 arena 漂移");
        }
        let workspace_bytes = {
            let program = program
                .lock()
                .map_err(|_| anyhow!("K4 prefix state program mutex poisoned"))?;
            validate_program_identity(&program, &arena, source)?;
            let first = program.recipe(0, FULL_DEPTH_LAYERS[0])?;
            for layer in FULL_DEPTH_LAYERS {
                for lane in 0..block_size {
                    let recipe = program.recipe(lane, layer)?;
                    if recipe.workspace_bytes != first.workspace_bytes
                        || recipe.candidate_state_bytes != arena.layout().checkpoint_state_bytes
                        || recipe.position != base_position + lane as u32
                    {
                        bail!("K4 prefix producer L{layer}/lane{lane} recipe ABI 漂移");
                    }
                }
            }
            first.workspace_bytes
        };

        let workspace = device_buffer(&context, workspace_bytes)?;
        let numeric = match S14NumericPipelines::new(&context) {
            Ok(value) => value,
            Err(error) => {
                workspace.destroy(&context);
                return Err(error.context("构造 K4 prefix numeric pipelines"));
            }
        };
        let ape = match S14Position0ApeAddPipeline::new(&context) {
            Ok(value) => value,
            Err(error) => {
                numeric.destroy(&context);
                workspace.destroy(&context);
                return Err(error.context("构造 K4 prefix APE pipeline"));
            }
        };
        Ok(Self {
            context,
            base_position,
            block_size,
            source,
            arena,
            program,
            workspace: Some(workspace),
            numeric: Some(numeric),
            ape: Some(ape),
            pending_binders: Mutex::new(Vec::new()),
            phase: Mutex::new(ProducerPhase::Ready {
                next_layer_index: 0,
            }),
        })
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn arena(&self) -> &Arc<S14CausalBlockPrefixCheckpointArena> {
        &self.arena
    }

    pub fn source(&self) -> MaterializedTokenSource {
        self.source
    }

    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn shared_program(&self) -> &S14CausalBlockSharedPrefixStateProgram {
        &self.program
    }

    /// 在当前 HC/QKV layer command 内录制真实 K-prefix 写集。
    ///
    /// # Safety
    /// `command` 必须处于 recording；HC/KV source 的 shader write 必须先发生，且所有
    /// source/static buffer 必须保活到 command fence 完成。
    pub unsafe fn record_layer(
        &self,
        inputs: S14CausalBlockPrefixLayerInputs<'_>,
    ) -> Result<S14CausalBlockPrefixLayerRecordingReceipt> {
        let result = self.record_layer_inner(inputs);
        if result.is_err() {
            self.poison();
        }
        result
    }

    unsafe fn record_layer_inner(
        &self,
        inputs: S14CausalBlockPrefixLayerInputs<'_>,
    ) -> Result<S14CausalBlockPrefixLayerRecordingReceipt> {
        let mut phase = self.lock_phase()?;
        let ProducerPhase::Ready { next_layer_index } = *phase else {
            bail!("K4 prefix producer 当前不可录制 layer: {phase:?}");
        };
        let expected_layer = *FULL_DEPTH_LAYERS
            .get(next_layer_index)
            .context("K4 prefix producer 已录满43层")?;
        if inputs.command == vk::CommandBuffer::null() || inputs.layer != expected_layer {
            bail!(
                "K4 prefix producer layer/command 顺序漂移: expected=L{expected_layer} observed=L{}",
                inputs.layer
            );
        }
        validate_slice(
            inputs.static_weights,
            inputs.static_logical_bytes,
            "static weights",
        )?;
        if inputs.static_weights.offset != 0 {
            bail!("K4 prefix producer static recipe 只接受 layer-local offset=0 页");
        }
        validate_slice(
            inputs.hc_branch_f32,
            self.block_size as u64 * HIDDEN * F32_BYTES,
            "HC branch F32",
        )?;
        validate_slice(
            inputs.rotated_current_kv_bf16,
            self.block_size as u64 * KV * BF16_BYTES,
            "rotated current KV BF16",
        )?;
        if !self.pending_binders()?.is_empty() {
            bail!("K4 prefix producer 上一层 binder 尚未在 fence 后释放");
        }

        let workspace = self.workspace()?;
        let numeric = self.numeric()?;
        let ape = self.ape()?;
        let mut program = self
            .program
            .lock()
            .map_err(|_| anyhow!("K4 prefix state program mutex poisoned"))?;
        let ratio = *COMPRESS_RATIOS
            .get(usize::from(inputs.layer))
            .context("K4 prefix compressor ratio layer 越界")?;
        if !matches!(ratio, 0 | 4 | 128) {
            bail!("K4 prefix producer L{} 未知 ratio {ratio}", inputs.layer);
        }
        // ratio4 以 B4 segment 为顺序事务单位。通用 producer 只闭合首个 boundary
        // 之前的 lanes；从首 boundary 开始的所有 application 均交给 segmented owner。
        // owner 会依次执行 segment0 boundary/tail、segment1 pre-boundary/boundary/tail，
        // 因而 K8 不会被错误地当成一个普通 contiguous prefix，也无需退回两次43层。
        let ratio4_boundary_lane = if ratio == 4 {
            Some(((K4 as u32 - 1) - (self.base_position % K4 as u32)) as usize)
        } else {
            None
        };
        let generic_lanes = ratio4_boundary_lane.unwrap_or(self.block_size);
        let mut window_writebacks = 0usize;
        let mut completed = 0usize;
        let mut compressor_evaluations = 0usize;

        for source_lane in 0..self.block_size {
            let targets = source_lane..self.block_size;
            for prefix_index in targets.clone() {
                let deferred_ratio4_owner_lane =
                    ratio4_boundary_lane.is_some_and(|boundary_lane| source_lane >= boundary_lane);
                // K8 的 deferred window 也必须由 segmented owner 按 segment 顺序写入；
                // 否则一次性提前写完8行会在128行滑窗附近把未来行暴露给 segment0。
                // K4 保持既有物理写法，owner 只接管状态登记，避免改变已通过的单段路径。
                let owner_records_window = self.block_size > K4 && deferred_ratio4_owner_lane;
                if !deferred_ratio4_owner_lane {
                    program.begin_lane_application(prefix_index, inputs.layer, source_lane)?;
                }
                if !owner_records_window {
                    let source_offset = inputs
                        .rotated_current_kv_bf16
                        .offset
                        .checked_add(source_lane as u64 * KV * BF16_BYTES)
                        .context("K-block prefix rotated-current-KV source offset overflow")?;
                    program.state_layout(source_lane)?.record_row_writeback_at(
                        &self.context,
                        inputs.command,
                        inputs.rotated_current_kv_bf16.buffer,
                        source_offset,
                        self.arena.buffer(),
                        self.arena.prefix_offset(prefix_index)?,
                        inputs.layer,
                        S14Position0StateRowKind::WindowKv,
                    )?;
                    window_writebacks += 1;
                }
                if !deferred_ratio4_owner_lane {
                    program.mark_window_recorded(prefix_index, inputs.layer)?;
                }
            }

            if source_lane >= generic_lanes {
                continue;
            }
            let recipe = program.recipe(source_lane, inputs.layer)?.clone();
            if recipe.static_layer_bytes != inputs.static_logical_bytes
                || recipe.position != self.base_position + source_lane as u32
                || recipe.compress_ratio != ratio
            {
                bail!(
                    "K4 prefix producer L{} lane{source_lane} static/position/ratio 漂移",
                    inputs.layer
                );
            }
            if !recipe.compressor_ops.is_empty() {
                copy_hc_lane(
                    &self.context,
                    inputs.command,
                    inputs.hc_branch_f32,
                    workspace,
                    compressor_input_offset(&recipe)?,
                    source_lane,
                )?;
                let mut recorded = recipe.record_compressor_remainder_at(
                    &self.context,
                    inputs.command,
                    numeric,
                    ape,
                    inputs.static_weights.buffer,
                    workspace,
                    self.arena.buffer(),
                    self.arena.prefix_offset(source_lane)?,
                )?;
                self.pending_binders()?.append(&mut recorded);
                compressor_evaluations += 1;
                for prefix_index in source_lane + 1..self.block_size {
                    for op in &recipe.compressor_ops {
                        if let S14Position0StateRecordingOp::Writeback {
                            target,
                            source_offset,
                            ..
                        } = op
                        {
                            program.state_layout(source_lane)?.record_row_writeback_at(
                                &self.context,
                                inputs.command,
                                workspace,
                                *source_offset,
                                self.arena.buffer(),
                                self.arena.prefix_offset(prefix_index)?,
                                inputs.layer,
                                *target,
                            )?;
                        }
                    }
                }
            }

            for prefix_index in targets {
                program.mark_remainder_recorded(prefix_index, inputs.layer)?;
                program.mark_boundary_finalized(prefix_index, inputs.layer)?;
                recipe.record_rollover_at(
                    &self.context,
                    inputs.command,
                    self.arena.buffer(),
                    self.arena.prefix_offset(prefix_index)?,
                )?;
                program.mark_rollover_recorded(prefix_index, inputs.layer)?;
                program.seal_lane_application(prefix_index, inputs.layer)?;
                completed += 1;
            }
        }

        *phase = ProducerPhase::Ready {
            next_layer_index: next_layer_index + 1,
        };
        Ok(S14CausalBlockPrefixLayerRecordingReceipt {
            base_position: self.base_position,
            block_size: self.block_size,
            layer: inputs.layer,
            window_writebacks,
            completed_lane_applications: completed,
            deferred_ratio4_lane_applications: ratio4_boundary_lane
                .map(|boundary_lane| {
                    (boundary_lane..self.block_size)
                        .map(|lane| self.block_size - lane)
                        .sum()
                })
                .unwrap_or(0),
            compressor_source_lane_evaluations: compressor_evaluations,
            serial_token_forward_calls: 0,
        })
    }

    /// 仅在包含上一层录制的 fence 完成后调用。
    pub fn release_completed_layer(&self) -> Result<()> {
        self.destroy_pending()
    }

    /// 只能在本层 fence、sticky status 与 route decode 全部成功后调用。
    pub(crate) fn record_completed_ratio4_layer(
        &self,
        layer: u8,
        receipt: S14CausalBlockRatio4SegmentedRecordingReceipt,
    ) -> Result<()> {
        self.arena.record_completed_ratio4_layer(layer, receipt)
    }

    pub fn production_evidence_snapshot(&self) -> Result<S14CausalBlockProductionEvidenceSnapshot> {
        self.arena.production_evidence_snapshot()
    }

    /// 43层 fence 都完成后 seal 430 次累积 application，并把同源 receipt 发布给 arena。
    pub fn seal_and_publish(&self) -> Result<()> {
        self.destroy_pending()?;
        let mut phase = self.lock_phase()?;
        if *phase
            != (ProducerPhase::Ready {
                next_layer_index: FULL_DEPTH_LAYERS.len(),
            })
        {
            bail!("K4 prefix producer 不能在未完成43层时 seal: {phase:?}");
        }
        let receipt = {
            let mut program = self
                .program
                .lock()
                .map_err(|_| anyhow!("K4 prefix state program mutex poisoned"))?;
            for prefix in 0..self.block_size {
                program.seal_prefix(prefix)?;
            }
            program.seal_block()?
        };
        self.arena.publish_prefix_program_seal_receipt(receipt)?;
        *phase = ProducerPhase::Sealed;
        Ok(())
    }

    pub fn abort_block(&self) {
        self.poison();
        let _ = self.destroy_pending();
    }

    pub fn destroy(&mut self) -> Result<()> {
        self.destroy_pending()?;
        {
            let mut phase = self.lock_phase()?;
            if *phase == ProducerPhase::Destroyed {
                return Ok(());
            }
            *phase = ProducerPhase::Destroyed;
        }
        if let Some(ape) = self.ape.take() {
            ape.destroy(&self.context);
        }
        if let Some(numeric) = self.numeric.take() {
            numeric.destroy(&self.context);
        }
        if let Some(workspace) = self.workspace.take() {
            workspace.destroy(&self.context);
        }
        Ok(())
    }

    fn poison(&self) {
        self.arena.abort();
        if let Ok(mut phase) = self.phase.lock() {
            *phase = ProducerPhase::Poisoned;
        }
    }

    fn workspace(&self) -> Result<&GpuBuffer> {
        self.workspace
            .as_ref()
            .context("K4 prefix workspace 已销毁")
    }

    fn numeric(&self) -> Result<&S14NumericPipelines> {
        self.numeric.as_ref().context("K4 prefix numeric 已销毁")
    }

    fn ape(&self) -> Result<&S14Position0ApeAddPipeline> {
        self.ape.as_ref().context("K4 prefix APE 已销毁")
    }

    fn lock_phase(&self) -> Result<MutexGuard<'_, ProducerPhase>> {
        self.phase
            .lock()
            .map_err(|_| anyhow!("K4 prefix producer phase poisoned"))
    }

    fn pending_binders(&self) -> Result<MutexGuard<'_, Vec<DescriptorBinder>>> {
        self.pending_binders
            .lock()
            .map_err(|_| anyhow!("K4 prefix producer binder owner poisoned"))
    }

    fn destroy_pending(&self) -> Result<()> {
        let mut pending = self.pending_binders()?;
        for binder in pending.drain(..).rev() {
            binder.destroy(&self.context);
        }
        Ok(())
    }
}

fn validate_program_identity(
    program: &S14CausalBlockPrefixStateProgram,
    arena: &S14CausalBlockPrefixCheckpointArena,
    source: MaterializedTokenSource,
) -> Result<()> {
    let base_position = arena.base_position();
    let block_size = arena.layout().block_size;
    if (source == MaterializedTokenSource::SpeculativeDraft && base_position == 0)
        || program.base_position() != base_position
        || !matches!(block_size, K4 | K8)
        || (source == MaterializedTokenSource::SpeculativeDraft && block_size != K4)
        || program.block_size() != block_size
        || program.identities().len() != block_size
        || program
            .identities()
            .iter()
            .enumerate()
            .any(|(lane, identity)| {
                identity.prefix_index != lane
                    || identity.input_position != base_position + lane as u32
                    || identity.checkpoint_position != base_position + lane as u32 + 1
                    || identity.candidate_state_bytes != arena.layout().checkpoint_state_bytes
            })
    {
        bail!("K-block prefix producer program/base/checkpoint ABI 漂移");
    }
    Ok(())
}

fn compressor_input_offset(
    recipe: &crate::s14_position0_state_writeback::S14Position0LayerStateRecordingRecipe,
) -> Result<u64> {
    let mut found = None;
    for op in &recipe.compressor_ops {
        if let S14Position0StateRecordingOp::Projection {
            input_offset, k, ..
        } = op
        {
            if u64::from(*k) != HIDDEN {
                bail!("L{} compressor projection K 漂移", recipe.layer);
            }
            match found {
                Some(expected) if expected != *input_offset => {
                    bail!(
                        "L{} compressor projection input offset 不一致",
                        recipe.layer
                    )
                }
                None => found = Some(*input_offset),
                _ => {}
            }
        }
    }
    found.context("compressor recipe 缺少 projection input")
}

unsafe fn copy_hc_lane(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    source: StorageBufferSlice<'_>,
    target: &GpuBuffer,
    target_offset: u64,
    lane: usize,
) -> Result<()> {
    let bytes = HIDDEN * F32_BYTES;
    let source_offset = source
        .offset
        .checked_add(lane as u64 * bytes)
        .context("K4 prefix HC lane source offset overflow")?;
    if source_offset
        .checked_add(bytes)
        .is_none_or(|end| end > source.buffer.size())
        || target_offset
            .checked_add(bytes)
            .is_none_or(|end| end > target.size())
        || source_offset % 4 != 0
        || target_offset % 4 != 0
    {
        bail!("K4 prefix HC lane copy range 越界或未对齐");
    }
    let acquire = [
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .buffer(source.buffer.handle())
            .offset(source_offset)
            .size(bytes),
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .buffer(target.handle())
            .offset(target_offset)
            .size(bytes),
    ];
    context.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &acquire,
        &[],
    );
    context.device.cmd_copy_buffer(
        command,
        source.buffer.handle(),
        target.handle(),
        &[vk::BufferCopy::default()
            .src_offset(source_offset)
            .dst_offset(target_offset)
            .size(bytes)],
    );
    let publish = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(target.handle())
        .offset(target_offset)
        .size(bytes);
    context.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[publish],
        &[],
    );
    Ok(())
}

fn validate_slice(slice: StorageBufferSlice<'_>, bytes: u64, label: &str) -> Result<()> {
    if slice.buffer.handle() == vk::Buffer::null()
        || bytes == 0
        || slice.offset % 4 != 0
        || slice
            .offset
            .checked_add(bytes)
            .is_none_or(|end| end > slice.buffer.size())
    {
        bail!("K4 prefix producer {label} slice 越界或未对齐");
    }
    Ok(())
}

fn device_buffer(context: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        context,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
        false,
    )
}
