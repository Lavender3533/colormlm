//! base_position=1、K=4 block-major prefix checkpoint 的真实 Vulkan producer。
//!
//! 每层 HC/QKV 得到 K 行 `hc_branch_f32` 与 `kv_raw_bf16` 后，本 producer 在同一个
//! command buffer 中把 source lane `s` 的写集累积到 prefix `s..K`。compressor 投影每个
//! source lane 只计算一次，随后把正式 recipe 的 writeback 行复制到其余 prefix。ratio4
//! 的 lane0/1 在这里闭合；lane2/3 只先写 window KV，并由 boundary state owner 在同一
//! command 中完成 remainder/finalize/rollover。没有结构 receipt 能替代这些 Vulkan 录制。

use crate::{
    compute::{DescriptorBinder, StorageBufferSlice},
    s14_causal_block_prefix_arena::S14CausalBlockPrefixCheckpointArena,
    s14_causal_block_prefix_state::S14CausalBlockPrefixStateProgram,
    s14_causal_block_production_evidence::S14CausalBlockProductionEvidenceSnapshot,
    s14_position0_state_writeback::{
        S14Position0ApeAddPipeline, S14Position0StateRecordingOp, S14Position0StateRowKind,
    },
    s14_vulkan::S14NumericPipelines,
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{COMPRESS_RATIOS, FULL_DEPTH_LAYERS};
use std::{
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};

const BASE_POSITION: u32 = 1;
const BLOCK_SIZE: usize = 4;
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
    pub current_kv_bf16: StorageBufferSlice<'a>,
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
            .field("current_kv_bf16", &self.current_kv_bf16.buffer.handle())
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
        if !Arc::ptr_eq(&context, arena.context())
            || arena.base_position() != BASE_POSITION
            || arena.layout().block_size != BLOCK_SIZE
        {
            bail!("K4 prefix producer context/base/K 与 arena 漂移");
        }
        let workspace_bytes = {
            let program = program
                .lock()
                .map_err(|_| anyhow!("K4 prefix state program mutex poisoned"))?;
            validate_program_identity(&program, &arena)?;
            let first = program.recipe(0, FULL_DEPTH_LAYERS[0])?;
            for layer in FULL_DEPTH_LAYERS {
                for lane in 0..BLOCK_SIZE {
                    let recipe = program.recipe(lane, layer)?;
                    if recipe.workspace_bytes != first.workspace_bytes
                        || recipe.candidate_state_bytes != arena.layout().checkpoint_state_bytes
                        || recipe.position != BASE_POSITION + lane as u32
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
            BLOCK_SIZE as u64 * HIDDEN * F32_BYTES,
            "HC branch F32",
        )?;
        validate_slice(
            inputs.current_kv_bf16,
            BLOCK_SIZE as u64 * KV * BF16_BYTES,
            "current KV BF16",
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

        let generic_lanes = if ratio == 4 { 2 } else { BLOCK_SIZE };
        let mut window_writebacks = 0usize;
        let mut completed = 0usize;
        let mut compressor_evaluations = 0usize;

        for source_lane in 0..BLOCK_SIZE {
            let targets = source_lane..BLOCK_SIZE;
            for prefix_index in targets.clone() {
                // ratio4 prefix3 的 lane2 尚未经过 boundary finalize/rollover 时，不能
                // 同时打开 lane3 application。lane3 KV 仍在这里真实写入；boundary
                // owner 在 lane2 seal 后再登记 begin/window receipt。
                let deferred_ratio4_lane3 = ratio == 4 && source_lane == 3;
                if !deferred_ratio4_lane3 {
                    program.begin_lane_application(prefix_index, inputs.layer, source_lane)?;
                }
                let source_offset = inputs
                    .current_kv_bf16
                    .offset
                    .checked_add(source_lane as u64 * KV * BF16_BYTES)
                    .context("K4 prefix current-KV source offset overflow")?;
                program.state_layout(source_lane)?.record_row_writeback_at(
                    &self.context,
                    inputs.command,
                    inputs.current_kv_bf16.buffer,
                    source_offset,
                    self.arena.buffer(),
                    self.arena.prefix_offset(prefix_index)?,
                    inputs.layer,
                    S14Position0StateRowKind::WindowKv,
                )?;
                if !deferred_ratio4_lane3 {
                    program.mark_window_recorded(prefix_index, inputs.layer)?;
                }
                window_writebacks += 1;
            }

            if source_lane >= generic_lanes {
                continue;
            }
            let recipe = program.recipe(source_lane, inputs.layer)?.clone();
            if recipe.static_layer_bytes != inputs.static_logical_bytes
                || recipe.position != BASE_POSITION + source_lane as u32
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
                for prefix_index in source_lane + 1..BLOCK_SIZE {
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
            base_position: BASE_POSITION,
            block_size: BLOCK_SIZE,
            layer: inputs.layer,
            window_writebacks,
            completed_lane_applications: completed,
            deferred_ratio4_lane_applications: if ratio == 4 { 3 } else { 0 },
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
        receipt: crate::s14_causal_block_ratio4_boundary::S14CausalBlockRatio4BoundaryRecordingReceipt,
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
            for prefix in 0..BLOCK_SIZE {
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
) -> Result<()> {
    if program.base_position() != BASE_POSITION
        || program.block_size() != BLOCK_SIZE
        || program.identities().len() != BLOCK_SIZE
        || program
            .identities()
            .iter()
            .enumerate()
            .any(|(lane, identity)| {
                identity.prefix_index != lane
                    || identity.input_position != BASE_POSITION + lane as u32
                    || identity.checkpoint_position != BASE_POSITION + lane as u32 + 1
                    || identity.candidate_state_bytes != arena.layout().checkpoint_state_bytes
            })
    {
        bail!("K4 prefix producer program/base/checkpoint ABI 漂移");
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
