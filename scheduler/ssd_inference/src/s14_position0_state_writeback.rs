//! FullDepth43 连续 token 的紧凑状态写回与唯一-fence读回。
//!
//! 层后端把真实 window KV、main compressor 与 ratio4 indexer remainder 写入
//! `WholeTokenDeviceState` 的 inactive candidate bank。本模块只读取
//! `Position0StateTxn::stage_layer` 允许发布的精确行，并在 terminal command 中把这些
//! 小写集合并复制到一个 host-visible ledger。调用方必须在整 token 唯一 host wait 后
//! 才能调用 `snapshot`；随后可用 `stage_payloads` 把真实 GPU payload 暂存到
//! `WholeTokenCandidate`，最终仍由原子 commit 决定是否发布。
//!
//! 每个 `position % 4 == 3` 的边界必须原子写入 ratio4 main/indexer compressed
//! BF16 KV，并把 active half row4..7 滚入 prefix row0..3；每个
//! `position % 128 == 127` 的边界还必须写入 ratio128 main compressed BF16 KV。

use crate::{
    compute::{ComputePipeline, DescriptorBinder},
    s14_position0_layer_program::{
        S14Position0CompressorProgram, S14Position0FullDepthLayerProgram, S14Position0LayerProgram,
        S14Position0WeightArena, S14TokenLayerStateAccess,
    },
    s14_position0_workspace::{S14Position0WorkspaceLayout, S14Position0WorkspaceSlot},
    s14_ratio4_history_paging::S14Ratio4HistoryPublishPlan,
    s14_vulkan::{S14Bf16MatvecShape, S14NumericPipelines},
    GpuBuffer, VulkanContext, WEIGHTED_ADD_SPV,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    BufferSlice, DType, GraphProfile, NativeState, Position0CompressorInput, WholeTokenCandidate,
    FULL_DEPTH_LAYERS,
};
use std::ops::Range;

pub const S14_POSITION0_STATE_WRITEBACK_BYTES: u64 = 340_992;
pub const S14_POSITION3_STATE_READBACK_BYTES: u64 = 367_872;
pub const S14_POSITION3_LAYER_DIRTY_BYTES: u64 = 1_228_032;
pub const S14_POSITION3_READBACK_COPY_COUNT: usize = 209;
pub const S14_POSITION3_DEVICE_COPY_COUNT: usize = 545;
const S14_POSITION0_HIDDEN: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0StateRowKind {
    WindowKv,
    MainCompressorKv,
    MainCompressorScore,
    IndexerCompressorKv,
    IndexerCompressorScore,
    MainCompressedKv,
    IndexerCompressedKv,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0StateRolloverCopy {
    pub kind: S14Position0StateRowKind,
    pub source_range: Range<u64>,
    pub target_range: Range<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0StateRow {
    pub kind: S14Position0StateRowKind,
    pub state_range: Range<u64>,
    pub readback_range: Range<u64>,
    pub dtype: DType,
    pub elements: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0LayerStateLayout {
    pub layer: u8,
    pub compress_ratio: u16,
    pub position: u32,
    /// candidate bank 已从 committed bank 同步、可供本 token attention 读取的 window 行。
    pub committed_window_state_range: Range<u64>,
    pub rows: Vec<S14Position0StateRow>,
    /// position3 ratio4 active half row4..7 到 prefix row0..3 的精确 device 内复制。
    /// 这些 target 属于 dirty write-set，但无需再次读回 host；host TokenStateTxn 从
    /// committed arena 原子重建相同 rollover。
    pub rollover_copies: Vec<S14Position0StateRolloverCopy>,
}

impl S14Position0LayerStateLayout {
    pub fn state_ranges(&self) -> Vec<Range<u64>> {
        self.rows
            .iter()
            .map(|row| row.state_range.clone())
            .chain(
                self.rollover_copies
                    .iter()
                    .map(|copy| copy.target_range.clone()),
            )
            .collect()
    }

    pub fn device_copy_count(&self) -> usize {
        self.rows.len() + self.rollover_copies.len()
    }

    pub fn row(&self, kind: S14Position0StateRowKind) -> Result<&S14Position0StateRow> {
        let mut matches = self.rows.iter().filter(|row| row.kind == kind);
        let row = matches
            .next()
            .ok_or_else(|| anyhow!("L{} state ledger 缺少 {kind:?}", self.layer))?;
        if matches.next().is_some() {
            bail!("L{} state ledger 重复 {kind:?}", self.layer);
        }
        Ok(row)
    }
}

/// 复用现有 `weighted_add` shader 执行 position0 的
/// `score = wgate(hidden) + ape[0]`。它只加真实 APE payload，不生成常量或 fixture。
pub struct S14Position0ApeAddPipeline {
    pipeline: ComputePipeline,
}

pub struct S14Position0ApeAddDispatch {
    pub binder: DescriptorBinder,
    elements: u32,
}

impl S14Position0ApeAddPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, WEIGHTED_ADD_SPV, 2, 8)?,
        })
    }

    /// `ape_offset` 必须已经指向固定 revision APE 的 position0 行；ratio4/128
    /// 都是 row 0。descriptor 会同时验证真实 buffer 容量和设备 offset 对齐。
    pub fn bind(
        &self,
        ctx: &VulkanContext,
        score_buffer: &GpuBuffer,
        score_offset: u64,
        ape_buffer: &GpuBuffer,
        ape_offset: u64,
        elements: u32,
    ) -> Result<S14Position0ApeAddDispatch> {
        if !matches!(elements, 256 | 512 | 1024) {
            bail!("position0 APE add elements 必须是 256/512/1024");
        }
        let bytes = u64::from(elements) * 4;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (score_buffer, score_offset, bytes),
                (ape_buffer, ape_offset, bytes),
            ],
        )?;
        Ok(S14Position0ApeAddDispatch { binder, elements })
    }

    /// # Safety
    /// `command` 必须处于 recording 状态；wgate matvec 对 score 的写入必须先以 compute
    /// barrier 对本 dispatch 可见。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14Position0ApeAddDispatch,
    ) {
        ctx.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.pipeline,
        );
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.layout,
            0,
            &[dispatch.binder.set],
            &[],
        );
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&dispatch.elements.to_le_bytes());
        push[4..].copy_from_slice(&1.0f32.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push,
        );
        ctx.device
            .cmd_dispatch(command, dispatch.elements.div_ceil(256), 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0StateWritebackLayout {
    pub position: u32,
    pub layers: Vec<S14Position0LayerStateLayout>,
    pub readback_bytes: u64,
    pub candidate_state_bytes: u64,
}

/// 完全由 `NativeState` 的 slice ABI 与当前 position 写集推导的紧凑 ledger。
/// 这里不把某个固定 position 的手算常量作为运行时真相。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0StateWritebackAbiLedger {
    pub readback_copy_count: usize,
    pub device_copy_count: usize,
    pub readback_bytes: u64,
    pub layer_dirty_bytes: u64,
}

impl S14Position0StateWritebackLayout {
    pub fn build(state: &NativeState) -> Result<Self> {
        state
            .validate_for(GraphProfile::FullDepth43NativeTop6)
            .context("validate FullDepth43 native state")?;
        let mut cursor = 0u64;
        let mut layers = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
        for layer in FULL_DEPTH_LAYERS {
            let kv = unique(
                state.kv.iter().filter(|entry| entry.layer == layer),
                &format!("L{layer} window KV"),
            )?;
            let access = S14TokenLayerStateAccess::for_ratio(state.position, kv.compress_ratio)?;
            let mut rows = Vec::with_capacity(match kv.compress_ratio {
                0 => 1,
                4 => 5 + usize::from(access.compressed_block_ready) * 2,
                128 => 3 + usize::from(access.compressed_block_ready),
                ratio => bail!("L{layer} 未知 compressor ratio {ratio}"),
            });
            let mut rollover_copies = Vec::new();
            let committed_window_state_range =
                slice_rows_range(&kv.cache, &access.committed_window_rows, DType::Bf16, 512)?;
            push_row(
                &mut rows,
                &mut cursor,
                S14Position0StateRowKind::WindowKv,
                &kv.cache,
                access.candidate_window_row,
                DType::Bf16,
                512,
            )?;

            match kv.compress_ratio {
                0 => {}
                4 => {
                    let remainder_row = u32::from(
                        access
                            .remainder_row
                            .ok_or_else(|| anyhow!("L{layer} ratio4 remainder row 缺失"))?,
                    );
                    let compressor = unique(
                        state
                            .compressors
                            .iter()
                            .filter(|entry| entry.layer == layer && entry.compress_ratio == 4),
                        &format!("L{layer} ratio4 main compressor"),
                    )?;
                    let indexer = unique(
                        state.indexers.iter().filter(|entry| entry.layer == layer),
                        &format!("L{layer} ratio4 indexer compressor"),
                    )?;
                    let history_publish =
                        S14Ratio4HistoryPublishPlan::build(state, layer, state.position)?;
                    if access.compressed_count != history_publish.candidate_logical_len
                        || access.compressed_block_ready
                            != history_publish.appended_target.is_some()
                    {
                        bail!("L{layer} ratio4 history logical-length/access 合同漂移");
                    }
                    for (kind, slice, elements) in [
                        (
                            S14Position0StateRowKind::MainCompressorKv,
                            &compressor.kv_state,
                            1024,
                        ),
                        (
                            S14Position0StateRowKind::MainCompressorScore,
                            &compressor.score_state,
                            1024,
                        ),
                        (
                            S14Position0StateRowKind::IndexerCompressorKv,
                            &indexer.compressor_kv_state,
                            256,
                        ),
                        (
                            S14Position0StateRowKind::IndexerCompressorScore,
                            &indexer.compressor_score_state,
                            256,
                        ),
                    ] {
                        push_row(
                            &mut rows,
                            &mut cursor,
                            kind,
                            slice,
                            remainder_row,
                            DType::F32,
                            elements,
                        )?;
                    }
                    if access.compressed_block_ready {
                        let block = access.compressed_block_index.ok_or_else(|| {
                            anyhow!("L{layer} ratio4 compressed block index 缺失")
                        })?;
                        let history_target =
                            history_publish.appended_target.as_ref().ok_or_else(|| {
                                anyhow!("L{layer} ratio4 boundary history target 缺失")
                            })?;
                        if history_target.logical_block != block {
                            bail!("L{layer} ratio4 boundary block/history target 漂移");
                        }
                        let boundary_row_start = rows.len();
                        push_row(
                            &mut rows,
                            &mut cursor,
                            S14Position0StateRowKind::MainCompressedKv,
                            &kv.cache,
                            128u32.checked_add(block).ok_or_else(|| {
                                anyhow!("L{layer} ratio4 main compressed row overflow")
                            })?,
                            DType::Bf16,
                            512,
                        )?;
                        push_row(
                            &mut rows,
                            &mut cursor,
                            S14Position0StateRowKind::IndexerCompressedKv,
                            &indexer.kv_cache,
                            block,
                            DType::Bf16,
                            128,
                        )?;
                        if rows[boundary_row_start].state_range != history_target.main_state_range
                            || rows[boundary_row_start + 1].state_range
                                != history_target.indexer_state_range
                        {
                            bail!("L{layer} ratio4 boundary 分页 target/state ledger 漂移");
                        }
                        let (source_rows, target_rows) = access
                            .overlap_rollover_rows
                            .clone()
                            .ok_or_else(|| anyhow!("L{layer} ratio4 rollover rows 缺失"))?;
                        for (kind, slice, elements) in [
                            (
                                S14Position0StateRowKind::MainCompressorKv,
                                &compressor.kv_state,
                                1024,
                            ),
                            (
                                S14Position0StateRowKind::MainCompressorScore,
                                &compressor.score_state,
                                1024,
                            ),
                            (
                                S14Position0StateRowKind::IndexerCompressorKv,
                                &indexer.compressor_kv_state,
                                256,
                            ),
                            (
                                S14Position0StateRowKind::IndexerCompressorScore,
                                &indexer.compressor_score_state,
                                256,
                            ),
                        ] {
                            append_rollover_copies(
                                &mut rollover_copies,
                                kind,
                                slice,
                                &source_rows,
                                &target_rows,
                                elements,
                            )?;
                        }
                    } else if access.compressed_block_index.is_some()
                        || access.overlap_rollover_rows.is_some()
                    {
                        bail!("L{layer} 非边界 ratio4 不得声明 compressed/rollover");
                    }
                }
                128 => {
                    let remainder_row = u32::from(
                        access
                            .remainder_row
                            .ok_or_else(|| anyhow!("L{layer} ratio128 remainder row 缺失"))?,
                    );
                    let compressor = unique(
                        state
                            .compressors
                            .iter()
                            .filter(|entry| entry.layer == layer && entry.compress_ratio == 128),
                        &format!("L{layer} ratio128 main compressor"),
                    )?;
                    for (kind, slice) in [
                        (
                            S14Position0StateRowKind::MainCompressorKv,
                            &compressor.kv_state,
                        ),
                        (
                            S14Position0StateRowKind::MainCompressorScore,
                            &compressor.score_state,
                        ),
                    ] {
                        push_row(
                            &mut rows,
                            &mut cursor,
                            kind,
                            slice,
                            remainder_row,
                            DType::F32,
                            512,
                        )?;
                    }
                    if access.compressed_block_ready {
                        let block = access.compressed_block_index.ok_or_else(|| {
                            anyhow!("L{layer} ratio128 compressed block index 缺失")
                        })?;
                        push_row(
                            &mut rows,
                            &mut cursor,
                            S14Position0StateRowKind::MainCompressedKv,
                            &kv.cache,
                            128u32.checked_add(block).ok_or_else(|| {
                                anyhow!("L{layer} ratio128 main compressed row overflow")
                            })?,
                            DType::Bf16,
                            512,
                        )?;
                    } else if access.compressed_block_index.is_some() {
                        bail!("L{layer} 非边界 ratio128 不得声明 compressed block");
                    }
                }
                _ => unreachable!(),
            }
            layers.push(S14Position0LayerStateLayout {
                layer,
                compress_ratio: kv.compress_ratio,
                position: state.position,
                committed_window_state_range,
                rows,
                rollover_copies,
            });
        }
        validate_non_overlapping_state_ranges(&layers, state.arena_bytes)?;
        Ok(Self {
            position: state.position,
            layers,
            readback_bytes: cursor,
            candidate_state_bytes: state.arena_bytes,
        })
    }

    pub fn layer(&self, layer: u8) -> Option<&S14Position0LayerStateLayout> {
        self.layers
            .get(layer as usize)
            .filter(|entry| entry.layer == layer)
    }

    pub fn copy_count(&self) -> usize {
        self.layers.iter().map(|entry| entry.rows.len()).sum()
    }

    pub fn device_copy_count(&self) -> usize {
        self.layers
            .iter()
            .map(S14Position0LayerStateLayout::device_copy_count)
            .sum()
    }

    pub fn abi_ledger(&self) -> Result<S14Position0StateWritebackAbiLedger> {
        let merged = merge_device_dirty_ranges(
            self.layers
                .iter()
                .flat_map(S14Position0LayerStateLayout::state_ranges)
                .collect(),
            self.candidate_state_bytes,
        )?;
        Ok(S14Position0StateWritebackAbiLedger {
            readback_copy_count: self.copy_count(),
            device_copy_count: self.device_copy_count(),
            readback_bytes: self.readback_bytes,
            layer_dirty_bytes: dirty_range_bytes(&merged)?,
        })
    }

    fn copies(&self) -> Vec<vk::BufferCopy> {
        self.layers
            .iter()
            .flat_map(|entry| &entry.rows)
            .map(|row| {
                vk::BufferCopy::default()
                    .src_offset(row.state_range.start)
                    .dst_offset(row.readback_range.start)
                    .size(row.state_range.end - row.state_range.start)
            })
            .collect()
    }

    /// 把 workspace 中已经完成真实 projection（score 还必须完成真实 APE add）的单行
    /// 复制到 inactive candidate bank。它不提交、不等待，也不允许调用者自选 target offset。
    /// 返回值可直接交给 `WholeTokenDeviceState::mark_candidate_dirty`。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态；source 的 shader write 必须属于同一 timeline。
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record_row_writeback(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        source: &GpuBuffer,
        source_offset: u64,
        candidate_state: &GpuBuffer,
        layer: u8,
        kind: S14Position0StateRowKind,
    ) -> Result<Range<u64>> {
        let layer_layout = self
            .layer(layer)
            .ok_or_else(|| anyhow!("position0 state layout 缺少 L{layer}"))?;
        let row = layer_layout.row(kind)?;
        record_exact_row_writeback(
            ctx,
            command,
            source,
            source_offset,
            candidate_state,
            layer,
            kind,
            &row.state_range,
            source.size(),
            self.candidate_state_bytes,
        )?;
        Ok(row.state_range.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum S14Position0StateRecordingOp {
    Projection {
        target: S14Position0StateRowKind,
        tensor: String,
        weight_offset: u64,
        weight_bytes: u64,
        n: u32,
        k: u32,
        input_offset: u64,
        output_offset: u64,
    },
    ApeAdd {
        target: S14Position0StateRowKind,
        tensor: String,
        ape_offset: u64,
        ape_asset_bytes: u64,
        elements: u32,
        score_offset: u64,
    },
    Writeback {
        target: S14Position0StateRowKind,
        source_offset: u64,
        state_range: Range<u64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0LayerStateRecordingRecipe {
    pub layer: u8,
    pub index: usize,
    pub position: u32,
    pub compress_ratio: u16,
    pub static_layer_bytes: u64,
    pub workspace_bytes: u64,
    pub candidate_state_bytes: u64,
    /// attention 从已同步到 candidate bank 的 committed window 前缀读取。
    pub committed_window_state_range: Range<u64>,
    pub window_kv_source_offset: u64,
    pub window_kv_state_range: Range<u64>,
    pub compressor_ops: Vec<S14Position0StateRecordingOp>,
    pub rollover_copies: Vec<S14Position0StateRolloverCopy>,
    pub state_ranges_written: Vec<Range<u64>>,
}

impl S14Position0LayerStateRecordingRecipe {
    /// 非 ratio4 边界的一体化入口：先记录 compressor remainder，再记录 rollover（通常为空）。
    /// 每个 `position % 4 == 3` 的 ratio4 backend 必须改用 `record_compressor_remainder`，
    /// 在其返回后完成对应 compressed main/indexer finalize 与写回，最后再调用
    /// `record_rollover`；否则会先覆盖当前压缩块所需的 active half。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态；`static_arena` 必须是当前层 program 指定的
    /// resident/streamed static 页，`workspace` 必须包含当前层 HC branch F32。
    pub unsafe fn record_compressor(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        numeric: &S14NumericPipelines,
        ape: &S14Position0ApeAddPipeline,
        static_arena: &GpuBuffer,
        workspace: &GpuBuffer,
        candidate_state: &GpuBuffer,
    ) -> Result<Vec<DescriptorBinder>> {
        if self.compress_ratio == 4 && self.position % 4 == 3 {
            bail!(
                "L{} position{} ratio4 边界必须拆分记录 compressor remainder/finalize/rollover",
                self.layer,
                self.position
            );
        }
        let mut binders = self.record_compressor_remainder(
            ctx,
            command,
            numeric,
            ape,
            static_arena,
            workspace,
            candidate_state,
        )?;
        if let Err(error) = self.record_rollover(ctx, command, candidate_state) {
            for binder in binders.drain(..) {
                binder.destroy(ctx);
            }
            return Err(error);
        }
        Ok(binders)
    }

    /// 在 attention HC-pre/RMSNorm 之后、Q/KV QDQ 复用 compressor workspace 之前调用。
    /// ratio0 自动为空操作；ratio4/128 的投影、真实 APE row 与 remainder state writeback
    /// 均由 recipe 固定。此阶段绝不录制 rollover，因此任意 ratio4 边界 backend 可在返回后
    /// 从完整 row0..7 finalize 当前 compressed main/indexer block。
    ///
    /// 若 descriptor 绑定或录制前验证失败，本调用会销毁此前创建的全部 binder 并返回错误；
    /// 调用成功后 binder 的销毁责任与旧 `record_compressor` 返回值一致。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态；`static_arena` 必须是当前层 program 指定的
    /// resident/streamed static 页，`workspace` 必须包含当前层 HC branch F32。
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record_compressor_remainder(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        numeric: &S14NumericPipelines,
        ape: &S14Position0ApeAddPipeline,
        static_arena: &GpuBuffer,
        workspace: &GpuBuffer,
        candidate_state: &GpuBuffer,
    ) -> Result<Vec<DescriptorBinder>> {
        let mut binders = Vec::with_capacity(self.compressor_ops.len());
        let result = (|| -> Result<()> {
            for op in &self.compressor_ops {
                match op {
                    S14Position0StateRecordingOp::Projection {
                        weight_offset,
                        n,
                        k,
                        input_offset,
                        output_offset,
                        ..
                    } => {
                        let dispatch = numeric.bind_bf16_matvec_arenas(
                            ctx,
                            S14Bf16MatvecShape::new(*n, *k, 1)?,
                            static_arena,
                            self.static_layer_bytes,
                            *weight_offset,
                            workspace,
                            self.workspace_bytes,
                            *input_offset,
                            workspace,
                            self.workspace_bytes,
                            *output_offset,
                        )?;
                        numeric.cmd_bf16_matvec(ctx, command, &dispatch);
                        state_compute_barrier(ctx, command);
                        binders.push(dispatch.binder);
                    }
                    S14Position0StateRecordingOp::ApeAdd {
                        ape_offset,
                        elements,
                        score_offset,
                        ..
                    } => {
                        let dispatch = ape.bind(
                            ctx,
                            workspace,
                            *score_offset,
                            static_arena,
                            *ape_offset,
                            *elements,
                        )?;
                        ape.cmd(ctx, command, &dispatch);
                        state_compute_barrier(ctx, command);
                        binders.push(dispatch.binder);
                    }
                    S14Position0StateRecordingOp::Writeback {
                        target,
                        source_offset,
                        state_range,
                    } => {
                        record_exact_row_writeback(
                            ctx,
                            command,
                            workspace,
                            *source_offset,
                            candidate_state,
                            self.layer,
                            *target,
                            state_range,
                            self.workspace_bytes,
                            self.candidate_state_bytes,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            for binder in binders.drain(..) {
                binder.destroy(ctx);
            }
            return Err(error);
        }
        Ok(binders)
    }

    /// 单独录制 ratio4 边界 active half row4..7 → prefix row0..3 的 device 内
    /// rollover。必须在 compressed main/indexer finalize 及其 candidate 写回之后调用。
    /// position0..2 以及 ratio0/128 recipe 的 copy 集为空，因此保持兼容的空操作。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态；candidate 中 remainder 与当前边界 compressed
    /// 写回必须已经按同一 timeline happens-before 本调用。
    pub unsafe fn record_rollover(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        candidate_state: &GpuBuffer,
    ) -> Result<()> {
        if !self.rollover_copies.is_empty() && (self.position % 4 != 3 || self.compress_ratio != 4)
        {
            bail!(
                "L{} ratio4 rollover 仅允许 position%4==3 的 ratio4 recipe",
                self.layer
            );
        }
        record_ratio4_rollover(
            ctx,
            command,
            candidate_state,
            self.candidate_state_bytes,
            self.layer,
            &self.rollover_copies,
        )
    }

    /// 在当前层真实 `wkv → norm → rope/QDQ` 得到 `KeyValueBf16` 后调用。
    /// 返回精确 dirty range，供 device candidate ledger 使用。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态且 KeyValueBf16 的 shader write 已发生。
    pub unsafe fn record_window_kv(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: &GpuBuffer,
        candidate_state: &GpuBuffer,
    ) -> Result<Range<u64>> {
        record_exact_row_writeback(
            ctx,
            command,
            workspace,
            self.window_kv_source_offset,
            candidate_state,
            self.layer,
            S14Position0StateRowKind::WindowKv,
            &self.window_kv_state_range,
            self.workspace_bytes,
            self.candidate_state_bytes,
        )?;
        Ok(self.window_kv_state_range.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0FullDepthStateRecordingProgram {
    pub position: u32,
    pub layers: Vec<S14Position0LayerStateRecordingRecipe>,
    pub state_layout: S14Position0StateWritebackLayout,
    pub workspace_bytes: u64,
    merged_layer_state_ranges: Vec<Range<u64>>,
}

impl S14Position0FullDepthStateRecordingProgram {
    pub fn build(
        graph: &S14Position0FullDepthLayerProgram,
        workspace: &S14Position0WorkspaceLayout,
        state: &NativeState,
    ) -> Result<Self> {
        workspace.validate()?;
        if graph.layers.len() != FULL_DEPTH_LAYERS.len()
            || graph.workspace_used_bytes != workspace.used_bytes()
        {
            bail!("position0 state recording graph/workspace 层数或字节漂移");
        }
        let state_layout = S14Position0StateWritebackLayout::build(state)?;
        let mut layers = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
        for (&expected, layer_graph) in FULL_DEPTH_LAYERS.iter().zip(&graph.layers) {
            if layer_graph.layer != expected || layer_graph.index != expected as usize {
                bail!("position0 state recording layer graph 顺序漂移 at L{expected}");
            }
            let layer_state = state_layout
                .layer(expected)
                .ok_or_else(|| anyhow!("position0 state recording 缺少 L{expected} layout"))?;
            layers.push(build_layer_recording_recipe(
                layer_graph,
                layer_state,
                workspace,
                state_layout.candidate_state_bytes,
            )?);
        }
        let merged_layer_state_ranges = merge_device_dirty_ranges(
            layers
                .iter()
                .flat_map(|layer| layer.state_ranges_written.iter().cloned())
                .collect(),
            state_layout.candidate_state_bytes,
        )?;
        let output = Self {
            position: state.position,
            layers,
            state_layout,
            workspace_bytes: workspace.used_bytes(),
            merged_layer_state_ranges,
        };
        let abi = output.state_layout.abi_ledger()?;
        if output.copy_count() != abi.device_copy_count
            || output.state_layout.copy_count() != abi.readback_copy_count
            || output.state_layout.device_copy_count() != abi.device_copy_count
            || output.state_layout.readback_bytes != abi.readback_bytes
            || dirty_range_bytes(&output.merged_layer_state_ranges)? != abi.layer_dirty_bytes
        {
            bail!("token full-depth state recording 总写集漂移");
        }
        Ok(output)
    }

    pub fn layer(&self, layer: u8) -> Option<&S14Position0LayerStateRecordingRecipe> {
        self.layers
            .get(layer as usize)
            .filter(|entry| entry.layer == layer)
    }

    pub fn copy_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.state_ranges_written.len())
            .sum()
    }

    /// 43层 window KV、compressor remainder、position3 compressed KV 与 rollover
    /// 的精确、排序且相邻合并后的 device dirty write-set；不含 terminal final HC。
    pub fn merged_layer_state_ranges(&self) -> &[Range<u64>] {
        &self.merged_layer_state_ranges
    }

    /// 把43层精确状态写集与本 token 最终 `hc.streams` 的完整 BF16 slice 合并为 candidate
    /// bank 的唯一 dirty contract。调用方应逐 range 注册，禁止再用 `0..arena_bytes`。
    ///
    /// 本函数重新构建状态布局并逐字段核验，因而不能把另一个 position/profile/arena 的 HC
    /// 范围误配到当前 recording program。
    pub fn merged_device_dirty_write_set(&self, state: &NativeState) -> Result<Vec<Range<u64>>> {
        let rebuilt = S14Position0StateWritebackLayout::build(state)?;
        if rebuilt != self.state_layout || state.position != self.position {
            bail!("token device dirty write-set state/layout/position 漂移");
        }
        let hc = &state.hc.streams;
        let hc_end = hc
            .offset
            .checked_add(hc.bytes)
            .ok_or_else(|| anyhow!("position0 HC streams range overflow"))?;
        let expected_hc_bytes = 4u64 * u64::from(S14_POSITION0_HIDDEN) * 2;
        if hc.dtype != DType::Bf16
            || hc.shape != [1, 1, 4, S14_POSITION0_HIDDEN]
            || hc.bytes != expected_hc_bytes
            || hc.offset % 4 != 0
            || hc.bytes % 4 != 0
            || hc_end > self.state_layout.candidate_state_bytes
        {
            bail!("position0 HC streams dirty range shape/alignment 漂移");
        }

        let mut ranges = self.merged_layer_state_ranges.clone();
        ranges.push(hc.offset..hc_end);
        let merged = merge_device_dirty_ranges(ranges, self.state_layout.candidate_state_bytes)?;
        let expected_total = self
            .state_layout
            .abi_ledger()?
            .layer_dirty_bytes
            .checked_add(hc.bytes)
            .ok_or_else(|| anyhow!("token device dirty byte total overflow"))?;
        if dirty_range_bytes(&merged)? != expected_total {
            bail!("position0 device dirty write-set 与 layer/HC 精确字节不一致");
        }
        Ok(merged)
    }
}

fn merge_device_dirty_ranges(
    mut ranges: Vec<Range<u64>>,
    candidate_state_bytes: u64,
) -> Result<Vec<Range<u64>>> {
    if ranges.is_empty() || candidate_state_bytes == 0 {
        bail!("position0 device dirty write-set 不能为空");
    }
    for range in &ranges {
        if range.start >= range.end
            || range.start % 4 != 0
            || range.end % 4 != 0
            || range.end > candidate_state_bytes
        {
            bail!("position0 device dirty range 非空/对齐/边界合同漂移");
        }
    }
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    Ok(merged)
}

fn dirty_range_bytes(ranges: &[Range<u64>]) -> Result<u64> {
    ranges.iter().try_fold(0u64, |total, range| {
        let bytes = range
            .end
            .checked_sub(range.start)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| anyhow!("position0 dirty range byte count invalid"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("position0 dirty range byte total overflow"))
    })
}

fn build_layer_recording_recipe(
    layer_graph: &S14Position0LayerProgram,
    layer_state: &S14Position0LayerStateLayout,
    workspace: &S14Position0WorkspaceLayout,
    candidate_state_bytes: u64,
) -> Result<S14Position0LayerStateRecordingRecipe> {
    if layer_graph.layer != layer_state.layer
        || layer_graph.index != usize::from(layer_graph.layer)
        || layer_graph.static_layer_bytes == 0
        || candidate_state_bytes == 0
    {
        bail!(
            "L{} state recording layer/layout 合同漂移",
            layer_graph.layer
        );
    }

    let expected_ratio = match layer_graph.compressor {
        S14Position0CompressorProgram::None => 0,
        S14Position0CompressorProgram::Ratio4WithIndexer => 4,
        S14Position0CompressorProgram::Ratio128 => 128,
    };
    let access = layer_graph.state_access(layer_state.position)?;
    if layer_state.compress_ratio != expected_ratio {
        bail!(
            "L{} state recording compressor ratio 漂移: graph={expected_ratio} state={}",
            layer_graph.layer,
            layer_state.compress_ratio
        );
    }

    let window_source = workspace.region(S14Position0WorkspaceSlot::KeyValueBf16);
    let window_row = layer_state.row(S14Position0StateRowKind::WindowKv)?;
    let window_bytes = window_row.state_range.end - window_row.state_range.start;
    if window_source.logical_bytes != window_bytes
        || window_row.dtype != DType::Bf16
        || window_row.elements != 512
        || window_row.state_range.end > candidate_state_bytes
    {
        bail!("L{} window KV source/target layout 漂移", layer_graph.layer);
    }

    let mut compressor_ops = Vec::with_capacity(match layer_graph.compressor {
        S14Position0CompressorProgram::None => 0,
        S14Position0CompressorProgram::Ratio4WithIndexer => 10,
        S14Position0CompressorProgram::Ratio128 => 5,
    });
    match layer_graph.compressor {
        S14Position0CompressorProgram::None => {}
        S14Position0CompressorProgram::Ratio4WithIndexer => {
            append_compressor_recording_ops(
                &mut compressor_ops,
                layer_graph,
                layer_state,
                workspace,
                &format!("layers.{}.attn.compressor", layer_graph.layer),
                4,
                access
                    .ape_row
                    .ok_or_else(|| anyhow!("L{} ratio4 APE row 缺失", layer_graph.layer))?,
                1024,
                S14Position0StateRowKind::MainCompressorKv,
                S14Position0StateRowKind::MainCompressorScore,
            )?;
            append_compressor_recording_ops(
                &mut compressor_ops,
                layer_graph,
                layer_state,
                workspace,
                &format!("layers.{}.attn.indexer.compressor", layer_graph.layer),
                4,
                access
                    .ape_row
                    .ok_or_else(|| anyhow!("L{} indexer APE row 缺失", layer_graph.layer))?,
                256,
                S14Position0StateRowKind::IndexerCompressorKv,
                S14Position0StateRowKind::IndexerCompressorScore,
            )?;
        }
        S14Position0CompressorProgram::Ratio128 => {
            append_compressor_recording_ops(
                &mut compressor_ops,
                layer_graph,
                layer_state,
                workspace,
                &format!("layers.{}.attn.compressor", layer_graph.layer),
                128,
                access
                    .ape_row
                    .ok_or_else(|| anyhow!("L{} ratio128 APE row 缺失", layer_graph.layer))?,
                512,
                S14Position0StateRowKind::MainCompressorKv,
                S14Position0StateRowKind::MainCompressorScore,
            )?;
        }
    }

    let expected_ops = match layer_graph.compressor {
        S14Position0CompressorProgram::None => 0,
        S14Position0CompressorProgram::Ratio4WithIndexer => 10,
        S14Position0CompressorProgram::Ratio128 => 5,
    };
    if compressor_ops.len() != expected_ops {
        bail!("L{} compressor recording op 数量漂移", layer_graph.layer);
    }
    let state_ranges_written = layer_state.state_ranges();
    if state_ranges_written
        .iter()
        .any(|range| range.start >= range.end || range.end > candidate_state_bytes)
    {
        bail!("L{} state recording target range 越界", layer_graph.layer);
    }

    Ok(S14Position0LayerStateRecordingRecipe {
        layer: layer_graph.layer,
        index: layer_graph.index,
        position: layer_state.position,
        compress_ratio: expected_ratio,
        static_layer_bytes: layer_graph.static_layer_bytes,
        workspace_bytes: workspace.used_bytes(),
        candidate_state_bytes,
        committed_window_state_range: layer_state.committed_window_state_range.clone(),
        window_kv_source_offset: window_source.offset,
        window_kv_state_range: window_row.state_range.clone(),
        compressor_ops,
        rollover_copies: layer_state.rollover_copies.clone(),
        state_ranges_written,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_compressor_recording_ops(
    ops: &mut Vec<S14Position0StateRecordingOp>,
    layer_graph: &S14Position0LayerProgram,
    layer_state: &S14Position0LayerStateLayout,
    workspace: &S14Position0WorkspaceLayout,
    tensor_prefix: &str,
    ape_rows: u32,
    ape_row: u16,
    elements: u32,
    kv_kind: S14Position0StateRowKind,
    score_kind: S14Position0StateRowKind,
) -> Result<()> {
    let matrix_bytes = u64::from(elements)
        .checked_mul(u64::from(S14_POSITION0_HIDDEN))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| anyhow!("L{} compressor matrix bytes overflow", layer_graph.layer))?;
    let ape_asset_bytes = u64::from(ape_rows)
        .checked_mul(u64::from(elements))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| anyhow!("L{} compressor APE bytes overflow", layer_graph.layer))?;
    let wkv_tensor = format!("{tensor_prefix}.wkv.weight");
    let wgate_tensor = format!("{tensor_prefix}.wgate.weight");
    let ape_tensor = format!("{tensor_prefix}.ape");
    let wkv = exact_static_weight(layer_graph, &wkv_tensor, matrix_bytes)?;
    let wgate = exact_static_weight(layer_graph, &wgate_tensor, matrix_bytes)?;
    let ape = exact_static_weight(layer_graph, &ape_tensor, ape_asset_bytes)?;

    let input = workspace.region(S14Position0WorkspaceSlot::HcBranchF32);
    let projection = workspace.region(S14Position0WorkspaceSlot::CompressorProjectionF32);
    let score = workspace.region(S14Position0WorkspaceSlot::CompressorScoreF32);
    let row_bytes = u64::from(elements) * 4;
    if u32::from(ape_row) >= ape_rows {
        bail!("L{} {tensor_prefix} APE row 越界", layer_graph.layer);
    }
    let ape_offset = ape
        .offset
        .checked_add(u64::from(ape_row) * row_bytes)
        .ok_or_else(|| {
            anyhow!(
                "L{} {tensor_prefix} APE row offset overflow",
                layer_graph.layer
            )
        })?;
    let ape_row_end = ape_offset.checked_add(row_bytes).ok_or_else(|| {
        anyhow!(
            "L{} {tensor_prefix} APE row end overflow",
            layer_graph.layer
        )
    })?;
    let ape_asset_end = ape.offset.checked_add(ape.bytes).ok_or_else(|| {
        anyhow!(
            "L{} {tensor_prefix} APE asset end overflow",
            layer_graph.layer
        )
    })?;
    if ape_row_end > ape_asset_end {
        bail!("L{} {tensor_prefix} APE row range 漂移", layer_graph.layer);
    }
    let kv_row = layer_state.row(kv_kind)?;
    let score_row = layer_state.row(score_kind)?;
    if input.logical_bytes != u64::from(S14_POSITION0_HIDDEN) * 4
        || projection.logical_bytes < row_bytes
        || score.logical_bytes < row_bytes
        || kv_row.dtype != DType::F32
        || score_row.dtype != DType::F32
        || kv_row.elements != elements as usize
        || score_row.elements != elements as usize
        || kv_row.state_range.end - kv_row.state_range.start != row_bytes
        || score_row.state_range.end - score_row.state_range.start != row_bytes
    {
        bail!(
            "L{} {tensor_prefix} compressor workspace/state shape 漂移",
            layer_graph.layer
        );
    }

    ops.extend([
        S14Position0StateRecordingOp::Projection {
            target: kv_kind,
            tensor: wkv.tensor.clone(),
            weight_offset: wkv.offset,
            weight_bytes: wkv.bytes,
            n: elements,
            k: S14_POSITION0_HIDDEN,
            input_offset: input.offset,
            output_offset: projection.offset,
        },
        S14Position0StateRecordingOp::Projection {
            target: score_kind,
            tensor: wgate.tensor.clone(),
            weight_offset: wgate.offset,
            weight_bytes: wgate.bytes,
            n: elements,
            k: S14_POSITION0_HIDDEN,
            input_offset: input.offset,
            output_offset: score.offset,
        },
        S14Position0StateRecordingOp::ApeAdd {
            target: score_kind,
            tensor: ape.tensor.clone(),
            ape_offset,
            ape_asset_bytes: ape.bytes,
            elements,
            score_offset: score.offset,
        },
        S14Position0StateRecordingOp::Writeback {
            target: kv_kind,
            source_offset: projection.offset,
            state_range: kv_row.state_range.clone(),
        },
        S14Position0StateRecordingOp::Writeback {
            target: score_kind,
            source_offset: score.offset,
            state_range: score_row.state_range.clone(),
        },
    ]);
    Ok(())
}

fn exact_static_weight<'a>(
    layer_graph: &'a S14Position0LayerProgram,
    tensor: &str,
    expected_bytes: u64,
) -> Result<&'a crate::s14_position0_layer_program::S14Position0WeightBinding> {
    let binding = unique(
        layer_graph
            .static_weights
            .iter()
            .filter(|binding| binding.tensor == tensor),
        &format!("L{} static tensor {tensor}", layer_graph.layer),
    )?;
    let end = binding
        .offset
        .checked_add(binding.bytes)
        .ok_or_else(|| anyhow!("L{} {tensor} static range overflow", layer_graph.layer))?;
    if binding.arena != S14Position0WeightArena::StaticLayer(layer_graph.layer)
        || binding.bytes != expected_bytes
        || end > layer_graph.static_layer_bytes
    {
        bail!(
            "L{} {tensor} static binding range/bytes 漂移",
            layer_graph.layer
        );
    }
    Ok(binding)
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_exact_row_writeback(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    source: &GpuBuffer,
    source_offset: u64,
    candidate_state: &GpuBuffer,
    layer: u8,
    kind: S14Position0StateRowKind,
    state_range: &Range<u64>,
    source_logical_bytes: u64,
    candidate_logical_bytes: u64,
) -> Result<()> {
    let bytes = state_range
        .end
        .checked_sub(state_range.start)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| anyhow!("L{layer} {kind:?} target range 非法"))?;
    let source_end = source_offset
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("L{layer} {kind:?} source range overflow"))?;
    if source_offset % 4 != 0
        || state_range.start % 4 != 0
        || source_end > source_logical_bytes
        || source_end > source.size()
        || state_range.end > candidate_logical_bytes
        || state_range.end > candidate_state.size()
    {
        bail!("L{layer} {kind:?} source/candidate logical or physical range 漂移");
    }

    let acquire = [
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .buffer(source.handle())
            .offset(source_offset)
            .size(bytes),
        vk::BufferMemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .buffer(candidate_state.handle())
            .offset(state_range.start)
            .size(bytes),
    ];
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &acquire,
        &[],
    );
    ctx.device.cmd_copy_buffer(
        command,
        source.handle(),
        candidate_state.handle(),
        &[vk::BufferCopy::default()
            .src_offset(source_offset)
            .dst_offset(state_range.start)
            .size(bytes)],
    );
    let publish = [
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .buffer(source.handle())
            .offset(source_offset)
            .size(bytes),
        vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)
            .buffer(candidate_state.handle())
            .offset(state_range.start)
            .size(bytes),
    ];
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &publish,
        &[],
    );
    Ok(())
}

unsafe fn record_ratio4_rollover(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    candidate_state: &GpuBuffer,
    candidate_logical_bytes: u64,
    layer: u8,
    copies: &[S14Position0StateRolloverCopy],
) -> Result<()> {
    if copies.is_empty() {
        return Ok(());
    }

    let mut occupied = Vec::<Range<u64>>::with_capacity(copies.len() * 2);
    let mut regions = Vec::with_capacity(copies.len());
    let mut acquire = Vec::with_capacity(copies.len() * 2);
    let mut publish = Vec::with_capacity(copies.len() * 2);
    for copy in copies {
        if !matches!(
            copy.kind,
            S14Position0StateRowKind::MainCompressorKv
                | S14Position0StateRowKind::MainCompressorScore
                | S14Position0StateRowKind::IndexerCompressorKv
                | S14Position0StateRowKind::IndexerCompressorScore
        ) {
            bail!("L{layer} ratio4 rollover kind 漂移: {:?}", copy.kind);
        }
        let source_bytes = copy
            .source_range
            .end
            .checked_sub(copy.source_range.start)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| anyhow!("L{layer} ratio4 rollover source range 非法"))?;
        let target_bytes = copy
            .target_range
            .end
            .checked_sub(copy.target_range.start)
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| anyhow!("L{layer} ratio4 rollover target range 非法"))?;
        if source_bytes != target_bytes
            || source_bytes % 4 != 0
            || copy.source_range.start % 4 != 0
            || copy.target_range.start % 4 != 0
            || copy.source_range.end > candidate_logical_bytes
            || copy.target_range.end > candidate_logical_bytes
            || copy.source_range.end > candidate_state.size()
            || copy.target_range.end > candidate_state.size()
        {
            bail!("L{layer} ratio4 rollover source/target 字节、对齐或边界漂移");
        }
        for range in [&copy.source_range, &copy.target_range] {
            if occupied
                .iter()
                .any(|other| range.start < other.end && other.start < range.end)
            {
                bail!("L{layer} ratio4 rollover source/target 全局交叉重叠");
            }
            occupied.push(range.clone());
        }

        regions.push(
            vk::BufferCopy::default()
                .src_offset(copy.source_range.start)
                .dst_offset(copy.target_range.start)
                .size(source_bytes),
        );
        acquire.extend([
            vk::BufferMemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::TRANSFER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(candidate_state.handle())
                .offset(copy.source_range.start)
                .size(source_bytes),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::TRANSFER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .buffer(candidate_state.handle())
                .offset(copy.target_range.start)
                .size(target_bytes),
        ]);
        publish.extend([
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)
                .buffer(candidate_state.handle())
                .offset(copy.source_range.start)
                .size(source_bytes),
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)
                .buffer(candidate_state.handle())
                .offset(copy.target_range.start)
                .size(target_bytes),
        ]);
    }

    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &acquire,
        &[],
    );
    ctx.device.cmd_copy_buffer(
        command,
        candidate_state.handle(),
        candidate_state.handle(),
        &regions,
    );
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[],
        &publish,
        &[],
    );
    Ok(())
}

unsafe fn state_compute_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

#[derive(Debug, Clone)]
pub enum S14Position0CompressorPayload {
    None,
    Ratio4 {
        main_kv: Vec<f32>,
        main_score: Vec<f32>,
        indexer_kv: Vec<f32>,
        indexer_score: Vec<f32>,
    },
    Ratio4Boundary {
        main_kv: Vec<f32>,
        main_score: Vec<f32>,
        indexer_kv: Vec<f32>,
        indexer_score: Vec<f32>,
        main_compressed_kv_bf16: Vec<u16>,
        indexer_compressed_kv_bf16: Vec<u16>,
    },
    Ratio128 {
        main_kv: Vec<f32>,
        main_score: Vec<f32>,
    },
    Ratio128Boundary {
        main_kv: Vec<f32>,
        main_score: Vec<f32>,
        main_compressed_kv_bf16: Vec<u16>,
    },
}

#[derive(Debug, Clone)]
pub struct S14Position0LayerStatePayload {
    pub layer: u8,
    pub window_kv_bf16: Vec<u16>,
    pub compressor: S14Position0CompressorPayload,
    pub state_ranges_written: Vec<Range<u64>>,
}

pub struct S14Position0StateReadback {
    buffer: GpuBuffer,
    layout: S14Position0StateWritebackLayout,
    recorded: bool,
}

impl S14Position0StateReadback {
    pub fn new(ctx: &VulkanContext, state: &NativeState) -> Result<Self> {
        let layout = S14Position0StateWritebackLayout::build(state)?;
        let buffer = GpuBuffer::new(
            ctx,
            layout.readback_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )
        .context("allocate compact position0 state readback")?;
        Ok(Self {
            buffer,
            layout,
            recorded: false,
        })
    }

    pub fn layout(&self) -> &S14Position0StateWritebackLayout {
        &self.layout
    }

    /// 把 candidate bank 的精确 position0 写集追加到 terminal command。该 command 的
    /// 最终 timeline fence 必须是整 token 唯一 host wait；本函数不会提交或等待。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态，并且按 timeline happens-before 所有43层状态写入。
    pub unsafe fn record(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        candidate_state: &GpuBuffer,
    ) -> Result<()> {
        if self.recorded {
            bail!("position0 state readback 已录制，禁止重复录制");
        }
        if candidate_state.size() < self.layout.candidate_state_bytes {
            bail!("position0 candidate state buffer 小于 NativeState arena");
        }
        let source_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .buffer(candidate_state.handle())
            .offset(0)
            .size(self.layout.candidate_state_bytes);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[source_barrier],
            &[],
        );
        let copies = self.layout.copies();
        ctx.device.cmd_copy_buffer(
            command,
            candidate_state.handle(),
            self.buffer.handle(),
            &copies,
        );
        let host_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .buffer(self.buffer.handle())
            .offset(0)
            .size(self.layout.readback_bytes);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[host_barrier],
            &[],
        );
        self.recorded = true;
        Ok(())
    }

    /// 只能在包含 `record` 的 final timeline fence 已完成后调用。
    pub fn snapshot(&self) -> Result<Vec<S14Position0LayerStatePayload>> {
        if !self.recorded {
            bail!("position0 state readback 尚未录制");
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.buffer.mapped() as *const u8,
                self.layout.readback_bytes as usize,
            )
        };
        decode_payloads(&self.layout, bytes)
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.buffer.destroy(ctx);
    }
}

/// 把唯一-fence之后解码出的真实 payload 暂存到 host candidate。完整性、层序、finite、
/// ratio 与 shape 在此再次 fail-closed；本函数不会提交 token。
pub fn stage_payloads(
    candidate: &mut WholeTokenCandidate,
    payloads: &[S14Position0LayerStatePayload],
) -> Result<()> {
    if payloads.len() != FULL_DEPTH_LAYERS.len() {
        bail!("position0 state payload 不是43层");
    }
    for (&expected, payload) in FULL_DEPTH_LAYERS.iter().zip(payloads) {
        if payload.layer != expected {
            bail!(
                "position0 state payload 层序漂移: expected L{expected}, actual L{}",
                payload.layer
            );
        }
        let input = match &payload.compressor {
            S14Position0CompressorPayload::None => Position0CompressorInput::None,
            S14Position0CompressorPayload::Ratio4 {
                main_kv,
                main_score,
                indexer_kv,
                indexer_score,
            } => Position0CompressorInput::Ratio4 {
                main_kv,
                main_score,
                indexer_kv,
                indexer_score,
            },
            S14Position0CompressorPayload::Ratio4Boundary {
                main_kv,
                main_score,
                indexer_kv,
                indexer_score,
                main_compressed_kv_bf16,
                indexer_compressed_kv_bf16,
            } => Position0CompressorInput::Ratio4Boundary {
                main_kv,
                main_score,
                indexer_kv,
                indexer_score,
                main_compressed_kv_bf16,
                indexer_compressed_kv_bf16,
            },
            S14Position0CompressorPayload::Ratio128 {
                main_kv,
                main_score,
            } => Position0CompressorInput::Ratio128 {
                main_kv,
                main_score,
            },
            S14Position0CompressorPayload::Ratio128Boundary {
                main_kv,
                main_score,
                main_compressed_kv_bf16,
            } => Position0CompressorInput::Ratio128Boundary {
                main_kv,
                main_score,
                main_compressed_kv_bf16,
            },
        };
        candidate
            .stage_layer_state(expected, &payload.window_kv_bf16, input)
            .with_context(|| format!("stage L{expected} token state"))?;
        candidate
            .complete_layer(expected)
            .with_context(|| format!("complete L{expected} after state staging"))?;
    }
    Ok(())
}

fn decode_payloads(
    layout: &S14Position0StateWritebackLayout,
    bytes: &[u8],
) -> Result<Vec<S14Position0LayerStatePayload>> {
    if bytes.len() != layout.readback_bytes as usize {
        bail!("position0 state readback 字节数漂移");
    }
    layout
        .layers
        .iter()
        .map(|layer| {
            let window_kv_bf16 =
                decode_bf16(bytes, layer.row(S14Position0StateRowKind::WindowKv)?)?;
            let compressor = match layer.compress_ratio {
                0 => S14Position0CompressorPayload::None,
                4 => {
                    let main_kv = decode_f32(
                        bytes,
                        layer.row(S14Position0StateRowKind::MainCompressorKv)?,
                    )?;
                    let main_score = decode_f32(
                        bytes,
                        layer.row(S14Position0StateRowKind::MainCompressorScore)?,
                    )?;
                    let indexer_kv = decode_f32(
                        bytes,
                        layer.row(S14Position0StateRowKind::IndexerCompressorKv)?,
                    )?;
                    let indexer_score = decode_f32(
                        bytes,
                        layer.row(S14Position0StateRowKind::IndexerCompressorScore)?,
                    )?;
                    if layout.position % 4 == 3 {
                        S14Position0CompressorPayload::Ratio4Boundary {
                            main_kv,
                            main_score,
                            indexer_kv,
                            indexer_score,
                            main_compressed_kv_bf16: decode_bf16(
                                bytes,
                                layer.row(S14Position0StateRowKind::MainCompressedKv)?,
                            )?,
                            indexer_compressed_kv_bf16: decode_bf16(
                                bytes,
                                layer.row(S14Position0StateRowKind::IndexerCompressedKv)?,
                            )?,
                        }
                    } else {
                        S14Position0CompressorPayload::Ratio4 {
                            main_kv,
                            main_score,
                            indexer_kv,
                            indexer_score,
                        }
                    }
                }
                128 => {
                    let main_kv = decode_f32(
                        bytes,
                        layer.row(S14Position0StateRowKind::MainCompressorKv)?,
                    )?;
                    let main_score = decode_f32(
                        bytes,
                        layer.row(S14Position0StateRowKind::MainCompressorScore)?,
                    )?;
                    if layout.position % 128 == 127 {
                        S14Position0CompressorPayload::Ratio128Boundary {
                            main_kv,
                            main_score,
                            main_compressed_kv_bf16: decode_bf16(
                                bytes,
                                layer.row(S14Position0StateRowKind::MainCompressedKv)?,
                            )?,
                        }
                    } else {
                        S14Position0CompressorPayload::Ratio128 {
                            main_kv,
                            main_score,
                        }
                    }
                }
                ratio => bail!("L{} snapshot 未知 ratio {ratio}", layer.layer),
            };
            Ok(S14Position0LayerStatePayload {
                layer: layer.layer,
                window_kv_bf16,
                compressor,
                state_ranges_written: layer.state_ranges(),
            })
        })
        .collect()
}

fn decode_bf16(bytes: &[u8], row: &S14Position0StateRow) -> Result<Vec<u16>> {
    if row.dtype != DType::Bf16
        || row.readback_range.end - row.readback_range.start != (row.elements * 2) as u64
    {
        bail!("BF16 state row layout 漂移");
    }
    let payload = checked_bytes(bytes, &row.readback_range)?;
    let output = payload
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    if output
        .iter()
        .any(|bits| !f32::from_bits(u32::from(*bits) << 16).is_finite())
    {
        bail!("position0 BF16 state payload contains NaN/Inf");
    }
    Ok(output)
}

fn decode_f32(bytes: &[u8], row: &S14Position0StateRow) -> Result<Vec<f32>> {
    if row.dtype != DType::F32
        || row.readback_range.end - row.readback_range.start != (row.elements * 4) as u64
    {
        bail!("F32 state row layout 漂移");
    }
    let payload = checked_bytes(bytes, &row.readback_range)?;
    let output = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        bail!("position0 F32 state payload contains NaN/Inf");
    }
    Ok(output)
}

fn checked_bytes<'a>(bytes: &'a [u8], range: &Range<u64>) -> Result<&'a [u8]> {
    let start = usize::try_from(range.start)?;
    let end = usize::try_from(range.end)?;
    bytes
        .get(start..end)
        .ok_or_else(|| anyhow!("position0 state readback range 越界"))
}

fn slice_rows_range(
    slice: &BufferSlice,
    rows: &Range<u32>,
    dtype: DType,
    elements: usize,
) -> Result<Range<u64>> {
    if slice.dtype != dtype
        || slice.shape.len() != 3
        || slice.shape[0] != 1
        || rows.start > rows.end
        || rows.end > slice.shape[1]
        || slice.shape[2] as usize != elements
    {
        bail!("committed window state rows shape/dtype 漂移");
    }
    let dtype_bytes = match dtype {
        DType::Bf16 => 2,
        DType::F32 => 4,
    };
    let row_bytes = u64::try_from(elements)?
        .checked_mul(dtype_bytes)
        .ok_or_else(|| anyhow!("committed window row bytes overflow"))?;
    let expected_slice_bytes = u64::from(slice.shape[1])
        .checked_mul(row_bytes)
        .ok_or_else(|| anyhow!("committed window slice bytes overflow"))?;
    if slice.bytes != expected_slice_bytes {
        bail!("committed window slice byte ledger 漂移");
    }
    let start = slice
        .offset
        .checked_add(u64::from(rows.start) * row_bytes)
        .ok_or_else(|| anyhow!("committed window state range start overflow"))?;
    let end = slice
        .offset
        .checked_add(u64::from(rows.end) * row_bytes)
        .ok_or_else(|| anyhow!("committed window state range end overflow"))?;
    Ok(start..end)
}

fn push_row(
    rows: &mut Vec<S14Position0StateRow>,
    cursor: &mut u64,
    kind: S14Position0StateRowKind,
    slice: &BufferSlice,
    row: u32,
    dtype: DType,
    elements: usize,
) -> Result<()> {
    if slice.dtype != dtype || slice.shape.len() != 3 || slice.shape[0] != 1 {
        bail!("{kind:?} state slice shape/dtype 漂移");
    }
    let rows_count = u64::from(slice.shape[1]);
    if rows_count == 0 || u64::from(row) >= rows_count || slice.bytes % rows_count != 0 {
        bail!("{kind:?} state slice row layout 漂移");
    }
    let row_bytes = slice.bytes / rows_count;
    let dtype_bytes = match dtype {
        DType::Bf16 => 2,
        DType::F32 => 4,
    };
    let expected_bytes = u64::try_from(elements)?
        .checked_mul(dtype_bytes)
        .ok_or_else(|| anyhow!("{kind:?} state row bytes overflow"))?;
    if row_bytes != expected_bytes || slice.shape[2] as usize != elements {
        bail!("{kind:?} state row width 漂移");
    }
    let state_start = slice
        .offset
        .checked_add(u64::from(row) * row_bytes)
        .ok_or_else(|| anyhow!("{kind:?} state row offset overflow"))?;
    let state_end = state_start
        .checked_add(row_bytes)
        .ok_or_else(|| anyhow!("{kind:?} state row end overflow"))?;
    let readback_end = cursor
        .checked_add(row_bytes)
        .ok_or_else(|| anyhow!("{kind:?} readback row end overflow"))?;
    rows.push(S14Position0StateRow {
        kind,
        state_range: state_start..state_end,
        readback_range: *cursor..readback_end,
        dtype,
        elements,
    });
    *cursor = readback_end;
    Ok(())
}

fn append_rollover_copies(
    copies: &mut Vec<S14Position0StateRolloverCopy>,
    kind: S14Position0StateRowKind,
    slice: &BufferSlice,
    source_rows: &Range<u16>,
    target_rows: &Range<u16>,
    elements: usize,
) -> Result<()> {
    if slice.dtype != DType::F32
        || slice.shape.len() != 3
        || slice.shape[0] != 1
        || slice.shape[2] as usize != elements
        || source_rows != &(4..8)
        || target_rows != &(0..4)
        || source_rows.end > slice.shape[1] as u16
        || target_rows.end > slice.shape[1] as u16
        || slice.offset % 4 != 0
    {
        bail!("{kind:?} ratio4 rollover slice/row/dtype 合同漂移");
    }
    let row_count = u64::from(slice.shape[1]);
    let row_bytes = u64::try_from(elements)?
        .checked_mul(4)
        .ok_or_else(|| anyhow!("{kind:?} ratio4 rollover row bytes overflow"))?;
    let expected_slice_bytes = row_count
        .checked_mul(row_bytes)
        .ok_or_else(|| anyhow!("{kind:?} ratio4 rollover slice bytes overflow"))?;
    if row_count == 0 || slice.bytes != expected_slice_bytes {
        bail!("{kind:?} ratio4 rollover slice byte ledger 漂移");
    }

    for (source_row, target_row) in source_rows.clone().zip(target_rows.clone()) {
        let source_start = slice
            .offset
            .checked_add(u64::from(source_row) * row_bytes)
            .ok_or_else(|| anyhow!("{kind:?} ratio4 rollover source offset overflow"))?;
        let source_end = source_start
            .checked_add(row_bytes)
            .ok_or_else(|| anyhow!("{kind:?} ratio4 rollover source end overflow"))?;
        let target_start = slice
            .offset
            .checked_add(u64::from(target_row) * row_bytes)
            .ok_or_else(|| anyhow!("{kind:?} ratio4 rollover target offset overflow"))?;
        let target_end = target_start
            .checked_add(row_bytes)
            .ok_or_else(|| anyhow!("{kind:?} ratio4 rollover target end overflow"))?;
        if source_start < target_end && target_start < source_end {
            bail!("{kind:?} ratio4 rollover source/target overlap");
        }
        copies.push(S14Position0StateRolloverCopy {
            kind,
            source_range: source_start..source_end,
            target_range: target_start..target_end,
        });
    }
    Ok(())
}

fn unique<'a, T>(mut values: impl Iterator<Item = &'a T>, label: &str) -> Result<&'a T> {
    let value = values.next().ok_or_else(|| anyhow!("{label} 缺失"))?;
    if values.next().is_some() {
        bail!("{label} 不唯一");
    }
    Ok(value)
}

fn validate_non_overlapping_state_ranges(
    layers: &[S14Position0LayerStateLayout],
    arena_bytes: u64,
) -> Result<()> {
    let mut ranges = layers
        .iter()
        .flat_map(S14Position0LayerStateLayout::state_ranges)
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| range.start);
    let mut previous_end = 0;
    for range in ranges {
        if range.start < previous_end || range.start >= range.end || range.end > arena_bytes {
            bail!("position0 state writeback target ranges overlap/out-of-bounds");
        }
        previous_end = range.end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s14_position0_weight_plan::S14Position0HybridWeightPlan;
    use polaris_s14_runner::{DecoderStateV1, Position0WholeTokenManifest};
    use std::{collections::BTreeMap, path::PathBuf};

    fn real_graph_and_state() -> (
        S14Position0FullDepthLayerProgram,
        S14Position0WorkspaceLayout,
        NativeState,
    ) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        let manifest = Position0WholeTokenManifest::load(&path).unwrap();
        let weights = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let workspace = S14Position0WorkspaceLayout::build(2 * 1024 * 1024).unwrap();
        let graph =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        let state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        (graph, workspace, state)
    }

    fn write_f32(bytes: &mut [u8], range: &Range<u64>, value: f32) {
        for chunk in bytes[range.start as usize..range.end as usize].chunks_exact_mut(4) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
    }

    #[test]
    fn full_depth_layout_covers_exact_position0_write_set() {
        let state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        assert_eq!(layout.layers.len(), 43);
        assert_eq!(layout.copy_count(), 167);
        assert_eq!(layout.readback_bytes, S14_POSITION0_STATE_WRITEBACK_BYTES);
        assert_eq!(layout.layer(0).unwrap().rows.len(), 1);
        assert_eq!(layout.layer(2).unwrap().rows.len(), 5);
        assert_eq!(layout.layer(3).unwrap().rows.len(), 3);
        assert_eq!(layout.layer(4).unwrap().rows.len(), 5);
        assert_eq!(layout.layer(42).unwrap().rows.len(), 5);
    }

    #[test]
    fn position1_layout_reads_committed_row0_and_writes_exact_candidate_rows() {
        let (graph, workspace, mut state) = real_graph_and_state();
        state.position = 1;
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        assert_eq!(layout.position, 1);

        let ratio4 = layout.layer(2).unwrap();
        let ratio4_kv = &state.kv[2].cache;
        assert_eq!(
            ratio4.committed_window_state_range,
            ratio4_kv.offset..ratio4_kv.offset + 1024
        );
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::WindowKv)
                .unwrap()
                .state_range,
            ratio4_kv.offset + 1024..ratio4_kv.offset + 2048
        );
        let main = state
            .compressors
            .iter()
            .find(|entry| entry.layer == 2)
            .unwrap();
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::MainCompressorKv)
                .unwrap()
                .state_range,
            main.kv_state.offset + 5 * 4096..main.kv_state.offset + 6 * 4096
        );

        let ratio128 = layout.layer(3).unwrap();
        let main = state
            .compressors
            .iter()
            .find(|entry| entry.layer == 3)
            .unwrap();
        assert_eq!(
            ratio128
                .row(S14Position0StateRowKind::MainCompressorKv)
                .unwrap()
                .state_range,
            main.kv_state.offset + 2048..main.kv_state.offset + 4096
        );

        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        let layer2 = program.layer(2).unwrap();
        assert_eq!(layer2.position, 1);
        assert_eq!(
            layer2.committed_window_state_range,
            ratio4_kv.offset..ratio4_kv.offset + 1024
        );
        let ape = graph.layers[2]
            .static_weights
            .iter()
            .find(|binding| binding.tensor == "layers.2.attn.compressor.ape")
            .unwrap();
        assert!(matches!(
            &layer2.compressor_ops[2],
            S14Position0StateRecordingOp::ApeAdd {
                ape_offset,
                elements: 1024,
                ..
            } if *ape_offset == ape.offset + 4096
        ));

        state.position = 3;
        let boundary = S14Position0StateWritebackLayout::build(&state).unwrap();
        assert_eq!(boundary.readback_bytes, S14_POSITION3_STATE_READBACK_BYTES);
    }

    #[test]
    fn position3_layout_and_program_cover_compressed_targets_and_rollover() {
        let (graph, workspace, mut state) = real_graph_and_state();
        state.position = 3;
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        assert_eq!(layout.position, 3);
        assert_eq!(layout.copy_count(), S14_POSITION3_READBACK_COPY_COUNT);
        assert_eq!(layout.device_copy_count(), S14_POSITION3_DEVICE_COPY_COUNT);
        assert_eq!(layout.readback_bytes, S14_POSITION3_STATE_READBACK_BYTES);

        let ratio4 = layout.layer(2).unwrap();
        assert_eq!(ratio4.rows.len(), 7);
        assert_eq!(ratio4.rollover_copies.len(), 16);
        assert_eq!(ratio4.state_ranges().len(), 23);
        let kv = &state.kv[2].cache;
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            kv.offset + 128 * 1024..kv.offset + 129 * 1024
        );
        let indexer = state
            .indexers
            .iter()
            .find(|entry| entry.layer == 2)
            .unwrap();
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::IndexerCompressedKv)
                .unwrap()
                .state_range,
            indexer.kv_cache.offset..indexer.kv_cache.offset + 256
        );
        let compressor = state
            .compressors
            .iter()
            .find(|entry| entry.layer == 2 && entry.compress_ratio == 4)
            .unwrap();
        for (kind, slice, row_bytes) in [
            (
                S14Position0StateRowKind::MainCompressorKv,
                &compressor.kv_state,
                4096,
            ),
            (
                S14Position0StateRowKind::MainCompressorScore,
                &compressor.score_state,
                4096,
            ),
            (
                S14Position0StateRowKind::IndexerCompressorKv,
                &indexer.compressor_kv_state,
                1024,
            ),
            (
                S14Position0StateRowKind::IndexerCompressorScore,
                &indexer.compressor_score_state,
                1024,
            ),
        ] {
            let copies = ratio4
                .rollover_copies
                .iter()
                .filter(|copy| copy.kind == kind)
                .collect::<Vec<_>>();
            assert_eq!(copies.len(), 4);
            for (row, copy) in copies.into_iter().enumerate() {
                let source = slice.offset + (4 + row as u64) * row_bytes;
                let target = slice.offset + row as u64 * row_bytes;
                assert_eq!(copy.source_range, source..source + row_bytes);
                assert_eq!(copy.target_range, target..target + row_bytes);
            }
        }

        let ratio128 = layout.layer(3).unwrap();
        assert_eq!(ratio128.rows.len(), 3);
        assert!(ratio128.rollover_copies.is_empty());

        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        assert_eq!(program.copy_count(), S14_POSITION3_DEVICE_COPY_COUNT);
        assert_eq!(
            program.state_layout.copy_count(),
            S14_POSITION3_READBACK_COPY_COUNT
        );
        assert_eq!(
            dirty_range_bytes(program.merged_layer_state_ranges()).unwrap(),
            S14_POSITION3_LAYER_DIRTY_BYTES
        );
        assert_eq!(program.layer(2).unwrap().state_ranges_written.len(), 23);
        let device_ranges = program.merged_device_dirty_write_set(&state).unwrap();
        assert_eq!(dirty_range_bytes(&device_ranges).unwrap(), 1_260_800);
    }

    #[test]
    fn position4_and_second_ratio4_boundary_reuse_generic_ledgers() {
        let (graph, workspace, mut state) = real_graph_and_state();

        state.position = 4;
        let regular = S14Position0StateWritebackLayout::build(&state).unwrap();
        assert_eq!(regular.copy_count(), 167);
        assert_eq!(regular.device_copy_count(), 167);
        assert_eq!(regular.readback_bytes, S14_POSITION0_STATE_WRITEBACK_BYTES);
        let ratio4 = regular.layer(2).unwrap();
        assert_eq!(ratio4.rows.len(), 5);
        assert!(ratio4.rollover_copies.is_empty());
        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        assert_eq!(program.copy_count(), 167);

        state.position = 7;
        let boundary = S14Position0StateWritebackLayout::build(&state).unwrap();
        assert_eq!(boundary.copy_count(), S14_POSITION3_READBACK_COPY_COUNT);
        assert_eq!(
            boundary.device_copy_count(),
            S14_POSITION3_DEVICE_COPY_COUNT
        );
        assert_eq!(boundary.readback_bytes, S14_POSITION3_STATE_READBACK_BYTES);
        let ratio4 = boundary.layer(2).unwrap();
        let kv = &state.kv[2].cache;
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            kv.offset + 129 * 1024..kv.offset + 130 * 1024
        );
        let indexer = state
            .indexers
            .iter()
            .find(|entry| entry.layer == 2)
            .unwrap();
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::IndexerCompressedKv)
                .unwrap()
                .state_range,
            indexer.kv_cache.offset + 256..indexer.kv_cache.offset + 512
        );
        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        assert_eq!(program.copy_count(), S14_POSITION3_DEVICE_COPY_COUNT);
    }

    #[test]
    fn position127_ratio128_boundary_registers_abi_derived_writeback_ledger() {
        let (graph, workspace, mut state) = real_graph_and_state();
        state.position = 127;
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        let abi = layout.abi_ledger().unwrap();
        assert_eq!(
            abi,
            S14Position0StateWritebackAbiLedger {
                readback_copy_count: 229,
                device_copy_count: 565,
                readback_bytes: 388_352,
                layer_dirty_bytes: 1_248_512,
            }
        );

        let ratio128 = layout.layer(3).unwrap();
        assert_eq!(ratio128.rows.len(), 4);
        assert!(ratio128.rollover_copies.is_empty());
        let kv = &state.kv[3].cache;
        assert_eq!(
            ratio128
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            kv.offset + 128 * 1024..kv.offset + 129 * 1024
        );

        let ratio4 = layout.layer(2).unwrap();
        let kv = &state.kv[2].cache;
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            kv.offset + 159 * 1024..kv.offset + 160 * 1024
        );

        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        assert_eq!(program.copy_count(), abi.device_copy_count);
        assert_eq!(
            dirty_range_bytes(program.merged_layer_state_ranges()).unwrap(),
            abi.layer_dirty_bytes
        );

        let payloads = decode_payloads(&layout, &vec![0u8; abi.readback_bytes as usize]).unwrap();
        assert!(matches!(
            payloads[2].compressor,
            S14Position0CompressorPayload::Ratio4Boundary { .. }
        ));
        assert!(matches!(
            &payloads[3].compressor,
            S14Position0CompressorPayload::Ratio128Boundary {
                main_compressed_kv_bf16,
                ..
            } if main_compressed_kv_bf16.len() == 512
        ));
    }

    #[test]
    fn position128_ring_wrap_reads_full_committed_window_and_writes_physical_row0() {
        let (graph, workspace, mut state) = real_graph_and_state();
        state.position = 128;
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        let abi = layout.abi_ledger().unwrap();
        for layer in FULL_DEPTH_LAYERS {
            let kv = state.kv.iter().find(|entry| entry.layer == layer).unwrap();
            let layer_layout = layout.layer(layer).unwrap();
            assert_eq!(
                layer_layout.committed_window_state_range,
                kv.cache.offset..kv.cache.offset + 128 * 1024
            );
            assert_eq!(
                layer_layout
                    .row(S14Position0StateRowKind::WindowKv)
                    .unwrap()
                    .state_range,
                kv.cache.offset..kv.cache.offset + 1024
            );
            assert!(layer_layout
                .row(S14Position0StateRowKind::MainCompressedKv)
                .is_err());
        }
        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        assert_eq!(program.copy_count(), abi.device_copy_count);
        let payloads = decode_payloads(&layout, &vec![0u8; abi.readback_bytes as usize]).unwrap();
        assert!(matches!(
            payloads[2].compressor,
            S14Position0CompressorPayload::Ratio4 { .. }
        ));
        assert!(matches!(
            payloads[3].compressor,
            S14Position0CompressorPayload::Ratio128 { .. }
        ));
    }

    #[test]
    fn position255_writes_physical_row127_and_both_compressor_boundary_targets() {
        let (graph, workspace, mut state) = real_graph_and_state();
        state.position = 255;
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        let abi = layout.abi_ledger().unwrap();

        for layer in FULL_DEPTH_LAYERS {
            let kv = state.kv.iter().find(|entry| entry.layer == layer).unwrap();
            let layer_layout = layout.layer(layer).unwrap();
            assert_eq!(
                layer_layout.committed_window_state_range,
                kv.cache.offset..kv.cache.offset + 128 * 1024
            );
            assert_eq!(
                layer_layout
                    .row(S14Position0StateRowKind::WindowKv)
                    .unwrap()
                    .state_range,
                kv.cache.offset + 127 * 1024..kv.cache.offset + 128 * 1024
            );
        }

        let ratio4 = layout.layer(2).unwrap();
        let ratio4_kv = state.kv.iter().find(|entry| entry.layer == 2).unwrap();
        let ratio4_indexer = state
            .indexers
            .iter()
            .find(|entry| entry.layer == 2)
            .unwrap();
        assert!(!ratio4.rollover_copies.is_empty());
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            ratio4_kv.cache.offset + 191 * 1024..ratio4_kv.cache.offset + 192 * 1024
        );
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::IndexerCompressedKv)
                .unwrap()
                .state_range,
            ratio4_indexer.kv_cache.offset + 63 * 256..ratio4_indexer.kv_cache.offset + 64 * 256
        );

        let ratio128 = layout.layer(3).unwrap();
        let ratio128_kv = state.kv.iter().find(|entry| entry.layer == 3).unwrap();
        assert!(ratio128.rollover_copies.is_empty());
        assert_eq!(
            ratio128
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            ratio128_kv.cache.offset + 129 * 1024..ratio128_kv.cache.offset + 130 * 1024
        );

        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        assert_eq!(program.copy_count(), abi.device_copy_count);
        assert_eq!(
            dirty_range_bytes(program.merged_layer_state_ranges()).unwrap(),
            abi.layer_dirty_bytes
        );
        let payloads = decode_payloads(&layout, &vec![0u8; abi.readback_bytes as usize]).unwrap();
        assert!(matches!(
            payloads[2].compressor,
            S14Position0CompressorPayload::Ratio4Boundary { .. }
        ));
        assert!(matches!(
            payloads[3].compressor,
            S14Position0CompressorPayload::Ratio128Boundary { .. }
        ));
    }

    #[test]
    fn position2047_uses_last_fixed_ratio4_cache_row_and_ratio128_block15() {
        let (graph, workspace, mut state) = real_graph_and_state();
        state.position = 2047;
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        let abi = layout.abi_ledger().unwrap();

        let ratio4 = layout.layer(2).unwrap();
        let ratio4_kv = state.kv.iter().find(|entry| entry.layer == 2).unwrap();
        let ratio4_indexer = state
            .indexers
            .iter()
            .find(|entry| entry.layer == 2)
            .unwrap();
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            ratio4_kv.cache.offset + 639 * 1024..ratio4_kv.cache.offset + 640 * 1024
        );
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::IndexerCompressedKv)
                .unwrap()
                .state_range,
            ratio4_indexer.kv_cache.offset + 511 * 256..ratio4_indexer.kv_cache.offset + 512 * 256
        );

        let ratio128 = layout.layer(3).unwrap();
        let ratio128_kv = state.kv.iter().find(|entry| entry.layer == 3).unwrap();
        assert_eq!(
            ratio128
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            ratio128_kv.cache.offset + 143 * 1024..ratio128_kv.cache.offset + 144 * 1024
        );

        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        assert_eq!(program.copy_count(), abi.device_copy_count);
        assert_eq!(
            dirty_range_bytes(program.merged_layer_state_ranges()).unwrap(),
            abi.layer_dirty_bytes
        );
    }

    #[test]
    fn position2051_writes_ratio4_block512_to_second_history_page() {
        let (graph, workspace, mut state) = real_graph_and_state();
        state.position = 2051;
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        let ratio4 = layout.layer(2).unwrap();
        let ratio4_kv = state.kv.iter().find(|entry| entry.layer == 2).unwrap();
        let ratio4_indexer = state
            .indexers
            .iter()
            .find(|entry| entry.layer == 2)
            .unwrap();
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::MainCompressedKv)
                .unwrap()
                .state_range,
            ratio4_kv.cache.offset + 640 * 1024..ratio4_kv.cache.offset + 641 * 1024
        );
        assert_eq!(
            ratio4
                .row(S14Position0StateRowKind::IndexerCompressedKv)
                .unwrap()
                .state_range,
            ratio4_indexer.kv_cache.offset + 512 * 256..ratio4_indexer.kv_cache.offset + 513 * 256
        );

        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        assert_eq!(program.copy_count(), layout.device_copy_count());
    }

    #[test]
    fn compact_payload_decode_preserves_ratio4_and_ratio128_rows() {
        let state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        let mut bytes = vec![0u8; layout.readback_bytes as usize];
        for layer in &layout.layers {
            for row in &layer.rows {
                match row.dtype {
                    DType::Bf16 => {
                        let bits = ((layer.layer as f32 + 1.0).to_bits() >> 16) as u16;
                        for chunk in bytes
                            [row.readback_range.start as usize..row.readback_range.end as usize]
                            .chunks_exact_mut(2)
                        {
                            chunk.copy_from_slice(&bits.to_le_bytes());
                        }
                    }
                    DType::F32 => write_f32(
                        &mut bytes,
                        &row.readback_range,
                        layer.layer as f32 + row.kind as u8 as f32 + 1.0,
                    ),
                }
            }
        }
        let payloads = decode_payloads(&layout, &bytes).unwrap();
        assert_eq!(payloads.len(), 43);
        assert!(matches!(
            payloads[2].compressor,
            S14Position0CompressorPayload::Ratio4 { .. }
        ));
        assert!(matches!(
            payloads[3].compressor,
            S14Position0CompressorPayload::Ratio128 { .. }
        ));
        assert_eq!(payloads[2].state_ranges_written.len(), 5);
        assert_eq!(payloads[3].state_ranges_written.len(), 3);
    }

    #[test]
    fn position3_payload_decode_exposes_compressed_ratio4_boundary() {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 3;
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        let bytes = vec![0u8; layout.readback_bytes as usize];
        let payloads = decode_payloads(&layout, &bytes).unwrap();
        match &payloads[2].compressor {
            S14Position0CompressorPayload::Ratio4Boundary {
                main_compressed_kv_bf16,
                indexer_compressed_kv_bf16,
                ..
            } => {
                assert_eq!(main_compressed_kv_bf16.len(), 512);
                assert_eq!(indexer_compressed_kv_bf16.len(), 128);
            }
            other => panic!("position3 ratio4 payload kind 漂移: {other:?}"),
        }
        assert_eq!(payloads[2].state_ranges_written.len(), 23);
        assert!(matches!(
            payloads[3].compressor,
            S14Position0CompressorPayload::Ratio128 { .. }
        ));
        assert_eq!(payloads[3].state_ranges_written.len(), 3);
    }

    #[test]
    fn nan_payload_fails_closed_before_host_candidate_staging() {
        let state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        let layout = S14Position0StateWritebackLayout::build(&state).unwrap();
        let mut bytes = vec![0u8; layout.readback_bytes as usize];
        let score = layout
            .layer(2)
            .unwrap()
            .row(S14Position0StateRowKind::MainCompressorScore)
            .unwrap();
        bytes[score.readback_range.start as usize..score.readback_range.start as usize + 4]
            .copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(decode_payloads(&layout, &bytes).is_err());
    }

    #[test]
    fn decoded_payloads_stage_all_43_layers_without_committing() {
        let state = DecoderStateV1::new(4096, 0).unwrap();
        let layout = S14Position0StateWritebackLayout::build(&state.native).unwrap();
        let bytes = vec![0u8; layout.readback_bytes as usize];
        let payloads = decode_payloads(&layout, &bytes).unwrap();
        let mut candidate = state.begin_token(0, 0, 0).unwrap();
        stage_payloads(&mut candidate, &payloads).unwrap();
        assert_eq!(candidate.position0_written_layer_count(), 43);
        assert_eq!(
            candidate.position0_staged_bytes(),
            S14_POSITION0_STATE_WRITEBACK_BYTES as usize
        );
        assert_eq!(state.position, 0);
        assert_eq!(state.commit_epoch, 0);
    }

    #[test]
    fn position3_boundary_staging_failure_rolls_back_committed_decoder_state() {
        let mut state = DecoderStateV1::new(4096, 0).unwrap();
        for predicted_token_id in [5, 223, 7] {
            let layout = S14Position0StateWritebackLayout::build(&state.native).unwrap();
            let bytes = vec![0u8; layout.readback_bytes as usize];
            let payloads = decode_payloads(&layout, &bytes).unwrap();
            let mut candidate = state
                .begin_token(state.commit_epoch, state.position, state.input_token_id)
                .unwrap();
            stage_payloads(&mut candidate, &payloads).unwrap();
            candidate.stage_hc_state(&vec![0u16; 4 * 4096]).unwrap();
            candidate.complete_final(predicted_token_id).unwrap();
            candidate.commit(&mut state).unwrap();
        }
        assert_eq!(state.position, 3);
        let before = state.clone();

        let layout = S14Position0StateWritebackLayout::build(&state.native).unwrap();
        let bytes = vec![0u8; layout.readback_bytes as usize];
        let mut invalid_payloads = decode_payloads(&layout, &bytes).unwrap();
        match &mut invalid_payloads[2].compressor {
            S14Position0CompressorPayload::Ratio4Boundary {
                main_compressed_kv_bf16,
                ..
            } => {
                main_compressed_kv_bf16.pop();
                assert_eq!(main_compressed_kv_bf16.len(), 511);
            }
            other => panic!("position3 ratio4 payload kind 漂移: {other:?}"),
        }
        let mut candidate = state
            .begin_token(state.commit_epoch, state.position, state.input_token_id)
            .unwrap();
        assert!(stage_payloads(&mut candidate, &invalid_payloads).is_err());
        drop(candidate);
        assert_eq!(state, before);

        let valid_payloads = decode_payloads(&layout, &bytes).unwrap();
        let mut candidate = state
            .begin_token(state.commit_epoch, state.position, state.input_token_id)
            .unwrap();
        stage_payloads(&mut candidate, &valid_payloads).unwrap();
        assert_eq!(
            candidate.position0_staged_bytes(),
            S14_POSITION3_LAYER_DIRTY_BYTES as usize
        );
        drop(candidate);
        assert_eq!(state, before);
    }

    #[test]
    fn real_layer_graph_builds_exact_full_depth_recording_recipes() {
        let (graph, workspace, state) = real_graph_and_state();
        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();

        assert_eq!(
            program
                .layers
                .iter()
                .map(|layer| layer.layer)
                .collect::<Vec<_>>(),
            FULL_DEPTH_LAYERS
        );
        assert_eq!(program.copy_count(), 167);
        assert_eq!(
            program.state_layout.readback_bytes,
            S14_POSITION0_STATE_WRITEBACK_BYTES
        );
        assert_eq!(
            program
                .layers
                .iter()
                .fold(BTreeMap::<u16, usize>::new(), |mut counts, layer| {
                    *counts.entry(layer.compress_ratio).or_default() += 1;
                    counts
                }),
            BTreeMap::from([(0, 2), (4, 21), (128, 20)])
        );

        let window_source = workspace
            .region(S14Position0WorkspaceSlot::KeyValueBf16)
            .offset;
        for layer in [&program.layers[0], &program.layers[1]] {
            assert!(layer.compressor_ops.is_empty());
            assert_eq!(layer.state_ranges_written.len(), 1);
            assert_eq!(layer.window_kv_source_offset, window_source);
            assert_eq!(
                layer.window_kv_state_range,
                program
                    .state_layout
                    .layer(layer.layer)
                    .unwrap()
                    .row(S14Position0StateRowKind::WindowKv)
                    .unwrap()
                    .state_range
            );
        }

        let ratio4 = program.layer(2).unwrap();
        assert_eq!(ratio4.compress_ratio, 4);
        assert_eq!(ratio4.compressor_ops.len(), 10);
        assert!(matches!(
            &ratio4.compressor_ops[0],
            S14Position0StateRecordingOp::Projection {
                target: S14Position0StateRowKind::MainCompressorKv,
                tensor,
                n: 1024,
                k: S14_POSITION0_HIDDEN,
                ..
            } if tensor == "layers.2.attn.compressor.wkv.weight"
        ));
        assert!(matches!(
            &ratio4.compressor_ops[2],
            S14Position0StateRecordingOp::ApeAdd {
                target: S14Position0StateRowKind::MainCompressorScore,
                tensor,
                ape_asset_bytes: 16_384,
                elements: 1024,
                ..
            } if tensor == "layers.2.attn.compressor.ape"
        ));
        assert!(matches!(
            &ratio4.compressor_ops[5],
            S14Position0StateRecordingOp::Projection {
                target: S14Position0StateRowKind::IndexerCompressorKv,
                tensor,
                n: 256,
                k: S14_POSITION0_HIDDEN,
                ..
            } if tensor == "layers.2.attn.indexer.compressor.wkv.weight"
        ));
        assert!(matches!(
            &ratio4.compressor_ops[7],
            S14Position0StateRecordingOp::ApeAdd {
                target: S14Position0StateRowKind::IndexerCompressorScore,
                tensor,
                ape_asset_bytes: 4096,
                elements: 256,
                ..
            } if tensor == "layers.2.attn.indexer.compressor.ape"
        ));
        assert_eq!(
            ratio4
                .compressor_ops
                .iter()
                .filter_map(|op| match op {
                    S14Position0StateRecordingOp::Writeback { target, .. } => Some(*target),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec![
                S14Position0StateRowKind::MainCompressorKv,
                S14Position0StateRowKind::MainCompressorScore,
                S14Position0StateRowKind::IndexerCompressorKv,
                S14Position0StateRowKind::IndexerCompressorScore,
            ]
        );

        let ratio128 = program.layer(3).unwrap();
        assert_eq!(ratio128.compress_ratio, 128);
        assert_eq!(ratio128.compressor_ops.len(), 5);
        assert!(matches!(
            &ratio128.compressor_ops[0],
            S14Position0StateRecordingOp::Projection {
                target: S14Position0StateRowKind::MainCompressorKv,
                n: 512,
                k: S14_POSITION0_HIDDEN,
                ..
            }
        ));
        assert!(matches!(
            &ratio128.compressor_ops[2],
            S14Position0StateRecordingOp::ApeAdd {
                target: S14Position0StateRowKind::MainCompressorScore,
                ape_asset_bytes: 262_144,
                elements: 512,
                ..
            }
        ));
        assert!(!ratio128.compressor_ops.iter().any(|op| matches!(
            op,
            S14Position0StateRecordingOp::Projection {
                target: S14Position0StateRowKind::IndexerCompressorKv
                    | S14Position0StateRowKind::IndexerCompressorScore,
                ..
            } | S14Position0StateRecordingOp::ApeAdd {
                target: S14Position0StateRowKind::IndexerCompressorKv
                    | S14Position0StateRowKind::IndexerCompressorScore,
                ..
            } | S14Position0StateRecordingOp::Writeback {
                target: S14Position0StateRowKind::IndexerCompressorKv
                    | S14Position0StateRowKind::IndexerCompressorScore,
                ..
            }
        )));

        let tail = program.layer(42).unwrap();
        assert_eq!(tail.compress_ratio, 4);
        assert_eq!(tail.compressor_ops.len(), 10);
    }

    #[test]
    fn merged_device_dirty_write_set_covers_167_layer_writes_and_final_hc_only() {
        let (graph, workspace, state) = real_graph_and_state();
        let program =
            S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state).unwrap();
        let layer_ranges = program.merged_layer_state_ranges();

        assert_eq!(program.copy_count(), 167);
        assert!(!layer_ranges.is_empty());
        assert!(layer_ranges
            .windows(2)
            .all(|pair| pair[0].end < pair[1].start));
        assert!(layer_ranges.iter().all(|range| {
            range.start < range.end
                && range.start % 4 == 0
                && range.end % 4 == 0
                && range.end <= state.arena_bytes
        }));
        assert_eq!(
            dirty_range_bytes(layer_ranges).unwrap(),
            S14_POSITION0_STATE_WRITEBACK_BYTES
        );
        for write in program
            .layers
            .iter()
            .flat_map(|layer| &layer.state_ranges_written)
        {
            assert!(layer_ranges
                .iter()
                .any(|merged| merged.start <= write.start && merged.end >= write.end));
        }

        let device_ranges = program.merged_device_dirty_write_set(&state).unwrap();
        let hc = state.hc.streams.offset..state.hc.streams.offset + state.hc.streams.bytes;
        assert!(!device_ranges.is_empty());
        assert!(device_ranges
            .windows(2)
            .all(|pair| pair[0].end < pair[1].start));
        assert!(device_ranges.iter().all(|range| {
            range.start < range.end
                && range.start % 4 == 0
                && range.end % 4 == 0
                && range.end <= state.arena_bytes
        }));
        assert!(device_ranges
            .iter()
            .any(|merged| merged.start <= hc.start && merged.end >= hc.end));
        for write in program
            .layers
            .iter()
            .flat_map(|layer| &layer.state_ranges_written)
        {
            assert!(device_ranges
                .iter()
                .any(|merged| merged.start <= write.start && merged.end >= write.end));
        }

        let exact_dirty_bytes = dirty_range_bytes(&device_ranges).unwrap();
        assert_eq!(
            exact_dirty_bytes,
            S14_POSITION0_STATE_WRITEBACK_BYTES + state.hc.streams.bytes
        );
        assert_eq!(exact_dirty_bytes, 373_760);
        assert!(exact_dirty_bytes < state.arena_bytes / 10);
    }

    #[test]
    fn recording_recipe_rejects_static_compressor_range_drift() {
        let (mut graph, workspace, state) = real_graph_and_state();
        let layer = &mut graph.layers[2];
        let static_layer_bytes = layer.static_layer_bytes;
        let binding = layer
            .static_weights
            .iter_mut()
            .find(|binding| binding.tensor == "layers.2.attn.compressor.wkv.weight")
            .unwrap();
        binding.offset = static_layer_bytes;

        let error = S14Position0FullDepthStateRecordingProgram::build(&graph, &workspace, &state)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("static binding range/bytes 漂移"));
    }
}
