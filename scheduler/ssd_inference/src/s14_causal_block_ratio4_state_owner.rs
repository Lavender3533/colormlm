//! base_position=1、K=4 ratio4 boundary 的 production 强状态 owner。
//!
//! 这个 adapter 不创建状态 fixture。它把 position1 authoritative state 已复制出的同一
//! prefix checkpoint arena 作为唯一 candidate owner，复用 position1..4 的正式
//! state-recording recipe：prefix2/prefix3 都在 position3 完成 remainder、main/indexer
//! finalize 与 rollover，prefix3 再写 position4 remainder。position4 dynamic index-head
//! 和 raw index-query 也在同一 command 中由真实权重投影得到。

use crate::{
    compute::{ComputePipeline, DescriptorBinder, StorageBufferSlice},
    s14_causal_block_hc_qkv_recorder::S14_CAUSAL_BLOCK_FP8_MATVEC_EXACT_SPV,
    s14_causal_block_prefix_arena::S14CausalBlockPrefixCheckpointArena,
    s14_causal_block_prefix_producer::S14CausalBlockSharedPrefixStateProgram,
    s14_causal_block_prefix_state::S14CausalBlockPrefixStateProgram,
    s14_causal_block_ratio4_boundary::{
        S14CausalBlockRatio4BoundaryFinalizeReceipt, S14CausalBlockRatio4BoundaryStateRecorder,
        S14CausalBlockRatio4CandidateStateBinding, S14CausalBlockRatio4Position4PreludeReceipt,
        S14CausalBlockRatio4RolloverReceipt, S14CausalBlockRatio4StateWorkspaceBindings,
    },
    s14_f32_to_bf16::{S14F32ToBf16Pipeline, S14F32ToBf16Shape},
    s14_position0_layer_program::{
        S14Position0FullDepthLayerProgram, S14Position0LayerProgram, S14Position0WeightArena,
    },
    s14_position0_state_writeback::{
        S14Position0ApeAddPipeline, S14Position0LayerStateRecordingRecipe,
        S14Position0StateRecordingOp, S14Position0StateRowKind, S14Position0StateWritebackLayout,
    },
    s14_ratio4_compressor_finalize::{
        S14Ratio4CompressorBoundary, S14Ratio4CompressorFinalizePipelines, S14Ratio4CompressorKind,
    },
    s14_vulkan::{S14Bf16MatvecShape, S14NumericPipelines},
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{DType, NativeState, COMPRESS_RATIOS, FULL_DEPTH_LAYERS};
use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
};

const BASE_POSITION: u32 = 1;
const BLOCK_SIZE: usize = 4;
const POSITION3_LANE: usize = 2;
const POSITION4_LANE: usize = 3;
const PREFIX2: usize = 2;
const PREFIX3: usize = 3;
const HIDDEN: u32 = 4096;
const INDEX_HEADS: u32 = 64;
const QUERY_LOW: u32 = 1024;
const RAW_INDEX_QUERY: u32 = 8192;
const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockRatio4StaticOffsets {
    pub layer: u8,
    pub main_compressor_norm_weight: u64,
    pub indexer_compressor_norm_weight: u64,
    pub index_weights_proj_weight: u64,
    pub index_query_weight: u64,
    pub index_query_scale: u64,
}

impl S14CausalBlockRatio4StaticOffsets {
    pub fn from_layer_program(program: &S14Position0LayerProgram) -> Result<Self> {
        let layer = program.layer;
        if COMPRESS_RATIOS.get(usize::from(layer)).copied() != Some(4) {
            bail!("L{layer} 不是 ratio4 层");
        }
        Ok(Self {
            layer,
            main_compressor_norm_weight: exact_static_offset(
                program,
                &format!("layers.{layer}.attn.compressor.norm.weight"),
                512 * BF16_BYTES,
            )?,
            indexer_compressor_norm_weight: exact_static_offset(
                program,
                &format!("layers.{layer}.attn.indexer.compressor.norm.weight"),
                128 * BF16_BYTES,
            )?,
            index_weights_proj_weight: exact_static_offset(
                program,
                &format!("layers.{layer}.attn.indexer.weights_proj.weight"),
                u64::from(INDEX_HEADS) * u64::from(HIDDEN) * BF16_BYTES,
            )?,
            index_query_weight: exact_static_offset(
                program,
                &format!("layers.{layer}.attn.indexer.wq_b.weight"),
                u64::from(RAW_INDEX_QUERY) * u64::from(QUERY_LOW),
            )?,
            index_query_scale: exact_static_offset(
                program,
                &format!("layers.{layer}.attn.indexer.wq_b.scale"),
                u64::from(RAW_INDEX_QUERY / 128) * u64::from(QUERY_LOW / 128),
            )?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct StateOffsets {
    main_kv_state: u64,
    main_score_state: u64,
    indexer_kv_state: u64,
    indexer_score_state: u64,
    first_compressed_kv: u64,
    first_indexer_row: u64,
}

#[derive(Clone, Copy, Debug)]
struct ScratchLayout {
    main_a: u64,
    main_b: u64,
    main_c: u64,
    inverse_rms: u64,
    index_head_f32: u64,
    raw_index_query_f32: u64,
    bytes: u64,
}

impl ScratchLayout {
    fn build(compressor_workspace_bytes: u64) -> Result<Self> {
        let mut cursor = align_up(compressor_workspace_bytes, ALIGNMENT)?;
        let mut take = |bytes: u64| -> Result<u64> {
            cursor = align_up(cursor, ALIGNMENT)?;
            let offset = cursor;
            cursor = cursor
                .checked_add(bytes)
                .context("ratio4 owner scratch overflow")?;
            Ok(offset)
        };
        let output = Self {
            main_a: take(512 * BF16_BYTES)?,
            main_b: take(512 * BF16_BYTES)?,
            main_c: take(512 * BF16_BYTES)?,
            inverse_rms: take(F32_BYTES)?,
            index_head_f32: take(u64::from(INDEX_HEADS) * F32_BYTES)?,
            raw_index_query_f32: take(u64::from(RAW_INDEX_QUERY) * F32_BYTES)?,
            bytes: 0,
        };
        Ok(Self {
            bytes: align_up(cursor, ALIGNMENT)?,
            ..output
        })
    }
}

struct SharedCore {
    context: Arc<VulkanContext>,
    prefix_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
    prefix_program: S14CausalBlockSharedPrefixStateProgram,
    workspace: GpuBuffer,
    scratch: ScratchLayout,
    compressed_rope: GpuBuffer,
    numeric: S14NumericPipelines,
    ape: S14Position0ApeAddPipeline,
    finalize: S14Ratio4CompressorFinalizePipelines,
    f32_to_bf16: S14F32ToBf16Pipeline,
    fp8_exact: ComputePipeline,
    pending_binders: Mutex<Vec<DescriptorBinder>>,
    active_layer: Mutex<Option<u8>>,
}

impl fmt::Debug for SharedCore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockRatio4SharedStateCore")
            .field("context", &Arc::as_ptr(&self.context))
            .field("prefix_arena", &Arc::as_ptr(&self.prefix_arena))
            .field("workspace", &self.workspace.handle())
            .field("scratch", &self.scratch)
            .field(
                "active_layer",
                &self.active_layer.lock().ok().and_then(|value| *value),
            )
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerPhase {
    Ready,
    Finalized,
    RolledOver,
    Position4Ready,
    Poisoned,
}

pub struct S14CausalBlockRatio4ProductionStateOwner {
    core: Arc<SharedCore>,
    layer: u8,
    prefix2_base: u64,
    prefix3_base: u64,
    candidate_logical_bytes: u64,
    recipes: [S14Position0LayerStateRecordingRecipe; BLOCK_SIZE],
    compressor_input_offset: u64,
    state: StateOffsets,
    static_offsets: S14CausalBlockRatio4StaticOffsets,
    phase: Mutex<LayerPhase>,
}

impl fmt::Debug for S14CausalBlockRatio4ProductionStateOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockRatio4ProductionStateOwner")
            .field("layer", &self.layer)
            .field("prefix2_base", &self.prefix2_base)
            .field("prefix3_base", &self.prefix3_base)
            .field("candidate_logical_bytes", &self.candidate_logical_bytes)
            .field("state", &self.state)
            .field("static_offsets", &self.static_offsets)
            .field("phase", &self.phase.lock().ok().map(|value| *value))
            .finish_non_exhaustive()
    }
}

#[must_use = "bundle/provider 销毁并等待最后 command 后必须显式 destroy ratio4 state owners"]
pub struct S14CausalBlockRatio4ProductionStateOwners {
    context: Arc<VulkanContext>,
    core: Option<Arc<SharedCore>>,
    states: BTreeMap<u8, Arc<S14CausalBlockRatio4ProductionStateOwner>>,
}

impl fmt::Debug for S14CausalBlockRatio4ProductionStateOwners {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockRatio4ProductionStateOwners")
            .field("state_layers", &self.states.keys().collect::<Vec<_>>())
            .field("core_present", &self.core.is_some())
            .finish()
    }
}

impl S14CausalBlockRatio4ProductionStateOwners {
    pub fn build(
        context: Arc<VulkanContext>,
        prefix_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
        prefix_program: S14CausalBlockSharedPrefixStateProgram,
        graph: &S14Position0FullDepthLayerProgram,
        authoritative: &NativeState,
    ) -> Result<Self> {
        let program = prefix_program
            .lock()
            .map_err(|_| anyhow!("ratio4 owner prefix program poisoned"))?;
        validate_global_inputs(&context, &prefix_arena, &program, graph, authoritative)?;
        let ratio4_layers = FULL_DEPTH_LAYERS
            .iter()
            .copied()
            .filter(|&layer| COMPRESS_RATIOS[usize::from(layer)] == 4)
            .collect::<Vec<_>>();
        let first_layer = *ratio4_layers.first().context("ratio4 layer set 为空")?;
        let first_recipe = program.recipe(0, first_layer)?;
        let scratch = ScratchLayout::build(first_recipe.workspace_bytes)?;
        let core = Arc::new(SharedCore::new(
            Arc::clone(&context),
            Arc::clone(&prefix_arena),
            Arc::clone(&prefix_program),
            scratch,
        )?);
        let prefix2_base = prefix_arena.prefix_offset(PREFIX2)?;
        let prefix3_base = prefix_arena.prefix_offset(PREFIX3)?;
        let mut states = BTreeMap::new();
        for layer in ratio4_layers {
            let recipes = [
                program.recipe(0, layer)?.clone(),
                program.recipe(1, layer)?.clone(),
                program.recipe(2, layer)?.clone(),
                program.recipe(3, layer)?.clone(),
            ];
            let layer_program = graph
                .layers
                .get(usize::from(layer))
                .filter(|program| program.layer == layer)
                .with_context(|| format!("ratio4 owner graph 缺少 L{layer}"))?;
            let static_offsets =
                S14CausalBlockRatio4StaticOffsets::from_layer_program(layer_program)?;
            let state = build_state_offsets(authoritative, layer, &recipes)?;
            let compressor_input_offset = validate_recipes(
                layer,
                &recipes,
                first_recipe.workspace_bytes,
                prefix_arena.layout().checkpoint_state_bytes,
                layer_program.static_layer_bytes,
            )?;
            states.insert(
                layer,
                Arc::new(S14CausalBlockRatio4ProductionStateOwner {
                    core: Arc::clone(&core),
                    layer,
                    prefix2_base,
                    prefix3_base,
                    candidate_logical_bytes: prefix_arena.layout().checkpoint_state_bytes,
                    recipes,
                    compressor_input_offset,
                    state,
                    static_offsets,
                    phase: Mutex::new(LayerPhase::Ready),
                }),
            );
        }
        Ok(Self {
            context,
            core: Some(core),
            states,
        })
    }

    pub fn trait_states(&self) -> BTreeMap<u8, Arc<dyn S14CausalBlockRatio4BoundaryStateRecorder>> {
        self.states
            .iter()
            .map(|(&layer, state)| {
                let erased: Arc<dyn S14CausalBlockRatio4BoundaryStateRecorder> = state.clone();
                (layer, erased)
            })
            .collect()
    }

    pub fn destroy(&mut self) -> Result<()> {
        self.states.clear();
        let core = self.core.take().context("ratio4 state owners 已销毁")?;
        match Arc::try_unwrap(core) {
            Ok(core) => {
                core.destroy(&self.context)?;
                Ok(())
            }
            Err(core) => {
                let refs = Arc::strong_count(&core);
                self.core = Some(core);
                bail!("ratio4 state core 仍被 provider/resource 持有: refs={refs}")
            }
        }
    }
}

impl S14CausalBlockRatio4BoundaryStateRecorder for S14CausalBlockRatio4ProductionStateOwner {
    fn candidate_state_owner(&self) -> &Arc<GpuBuffer> {
        self.core.prefix_arena.buffer()
    }

    fn candidate_state_binding(&self) -> S14CausalBlockRatio4CandidateStateBinding {
        S14CausalBlockRatio4CandidateStateBinding {
            layer: self.layer,
            base_position: BASE_POSITION,
            block_size: BLOCK_SIZE as u32,
            candidate_base_offset: self.prefix3_base,
            candidate_logical_bytes: self.candidate_logical_bytes,
            first_compressed_kv_offset: self.prefix3_base + self.state.first_compressed_kv,
            first_indexer_row_offset: self.prefix3_base + self.state.first_indexer_row,
            position3_recipe_position: self.recipes[POSITION3_LANE].position,
            position3_recipe_compress_ratio: self.recipes[POSITION3_LANE].compress_ratio,
            position4_recipe_position: self.recipes[POSITION4_LANE].position,
            position4_recipe_compress_ratio: self.recipes[POSITION4_LANE].compress_ratio,
            compressed_rope_position: S14Ratio4CompressorBoundary::new(3)
                .expect("validated position3 boundary")
                .compressed_position,
        }
    }

    unsafe fn record_remainder_and_finalize(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
    ) -> Result<S14CausalBlockRatio4BoundaryFinalizeReceipt> {
        self.begin_phase(ctx, command, workspace, LayerPhase::Ready)?;
        let mut binders = Vec::new();
        let result = (|| -> Result<()> {
            // lane0/1 的真实 window+remainder，以及 lane2 的 begin+window，均由通用
            // prefix producer 先录制。这里不能重复它们，只接管 boundary remainder。
            for (prefix, candidate_base) in
                [(PREFIX2, self.prefix2_base), (PREFIX3, self.prefix3_base)]
            {
                self.copy_hc_lane(ctx, command, workspace.hc_branch_f32, POSITION3_LANE)?;
                binders.extend(self.recipes[POSITION3_LANE].record_compressor_remainder_at(
                    ctx,
                    command,
                    &self.core.numeric,
                    &self.core.ape,
                    workspace.static_weights.buffer,
                    &self.core.workspace,
                    self.core.prefix_arena.buffer(),
                    candidate_base,
                )?);
                self.mark_position3_remainder(prefix)?;
                binders.extend(self.record_finalize(ctx, command, workspace, candidate_base)?);
                self.mark_position3_finalized(prefix)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.core.append_binders(binders)?;
                self.set_phase(LayerPhase::Finalized)?;
                Ok(S14CausalBlockRatio4BoundaryFinalizeReceipt {
                    base_position: BASE_POSITION,
                    block_size: BLOCK_SIZE as u32,
                    boundary_position: 3,
                    pre_boundary_remainder_record_calls: 3,
                    main_finalize_writeback_calls: 1,
                    indexer_finalize_writeback_calls: 1,
                    compressed_main_rows_written: 1,
                    compressed_indexer_rows_written: 1,
                    serial_token_forward_calls: 0,
                    cpu_fallback_calls: 0,
                })
            }
            Err(error) => {
                destroy_binders(ctx, &mut binders);
                self.poison();
                Err(error)
            }
        }
    }

    unsafe fn record_rollover_after_finalize(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        _workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
    ) -> Result<S14CausalBlockRatio4RolloverReceipt> {
        self.require_active_phase(ctx, command, LayerPhase::Finalized)?;
        let result = (|| -> Result<()> {
            for (prefix, candidate_base) in
                [(PREFIX2, self.prefix2_base), (PREFIX3, self.prefix3_base)]
            {
                self.recipes[POSITION3_LANE].record_rollover_at(
                    ctx,
                    command,
                    self.core.prefix_arena.buffer(),
                    candidate_base,
                )?;
                self.mark_position3_rolled_over_and_sealed(prefix)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.set_phase(LayerPhase::RolledOver)?;
                Ok(S14CausalBlockRatio4RolloverReceipt {
                    base_position: BASE_POSITION,
                    block_size: BLOCK_SIZE as u32,
                    boundary_position: 3,
                    rollover_record_calls: 1,
                    serial_token_forward_calls: 0,
                    cpu_fallback_calls: 0,
                })
            }
            Err(error) => {
                self.poison();
                Err(error)
            }
        }
    }

    unsafe fn record_position4_remainder_and_index_head(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
    ) -> Result<S14CausalBlockRatio4Position4PreludeReceipt> {
        self.validate_call(ctx, command, workspace)?;
        self.require_active_phase(ctx, command, LayerPhase::RolledOver)?;
        let mut binders = Vec::new();
        let result = (|| -> Result<()> {
            // 通用 producer 已在同一 command 中真实写入 position4 window KV；必须等
            // prefix3 的 position3 lane rollover/seal 后，才能串行打开 lane3 receipt。
            self.begin_position4_window()?;
            self.copy_hc_lane(ctx, command, workspace.hc_branch_f32, POSITION4_LANE)?;
            binders.extend(self.recipes[POSITION4_LANE].record_compressor_remainder_at(
                ctx,
                command,
                &self.core.numeric,
                &self.core.ape,
                workspace.static_weights.buffer,
                &self.core.workspace,
                self.core.prefix_arena.buffer(),
                self.prefix3_base,
            )?);
            self.mark_position4_remainder()?;
            // position4 不是 boundary；显式闭合 no-op finalize/rollover，不能只改计数。
            self.recipes[POSITION4_LANE].record_rollover_at(
                ctx,
                command,
                self.core.prefix_arena.buffer(),
                self.prefix3_base,
            )?;
            self.mark_position4_noop_boundary_and_seal()?;
            binders.extend(self.record_position4_projections(ctx, command, workspace)?);
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.core.append_binders(binders)?;
                self.set_phase(LayerPhase::Position4Ready)?;
                self.core.finish_layer(self.layer)?;
                Ok(S14CausalBlockRatio4Position4PreludeReceipt {
                    base_position: BASE_POSITION,
                    block_size: BLOCK_SIZE as u32,
                    position: 4,
                    remainder_record_calls: 1,
                    index_head_weight_projection_calls: 1,
                    serial_token_forward_calls: 0,
                    cpu_fallback_calls: 0,
                })
            }
            Err(error) => {
                destroy_binders(ctx, &mut binders);
                self.poison();
                Err(error)
            }
        }
    }
}

impl S14CausalBlockRatio4ProductionStateOwner {
    fn with_prefix_program(
        &self,
        apply: impl FnOnce(&mut S14CausalBlockPrefixStateProgram) -> Result<()>,
    ) -> Result<()> {
        let mut program = self
            .core
            .prefix_program
            .lock()
            .map_err(|_| anyhow!("ratio4 owner prefix program poisoned"))?;
        apply(&mut program)
    }

    fn mark_position3_remainder(&self, prefix: usize) -> Result<()> {
        self.with_prefix_program(|program| program.mark_remainder_recorded(prefix, self.layer))
    }

    fn mark_position3_finalized(&self, prefix: usize) -> Result<()> {
        self.with_prefix_program(|program| program.mark_boundary_finalized(prefix, self.layer))
    }

    fn mark_position3_rolled_over_and_sealed(&self, prefix: usize) -> Result<()> {
        self.with_prefix_program(|program| {
            program.mark_rollover_recorded(prefix, self.layer)?;
            program.seal_lane_application(prefix, self.layer)
        })
    }

    fn mark_position4_remainder(&self) -> Result<()> {
        self.with_prefix_program(|program| program.mark_remainder_recorded(PREFIX3, self.layer))
    }

    fn begin_position4_window(&self) -> Result<()> {
        self.with_prefix_program(|program| {
            program.begin_lane_application(PREFIX3, self.layer, POSITION4_LANE)?;
            program.mark_window_recorded(PREFIX3, self.layer)
        })
    }

    fn mark_position4_noop_boundary_and_seal(&self) -> Result<()> {
        self.with_prefix_program(|program| {
            program.mark_boundary_finalized(PREFIX3, self.layer)?;
            program.mark_rollover_recorded(PREFIX3, self.layer)?;
            program.seal_lane_application(PREFIX3, self.layer)
        })
    }

    fn begin_phase(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
        expected: LayerPhase,
    ) -> Result<()> {
        self.validate_call(ctx, command, workspace)?;
        if *self.lock_phase()? != expected {
            bail!("L{} ratio4 state owner phase 不是 {expected:?}", self.layer);
        }
        self.core.begin_layer(self.layer)
    }

    fn require_active_phase(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        expected: LayerPhase,
    ) -> Result<()> {
        if !std::ptr::eq(ctx, self.core.context.as_ref()) || command == vk::CommandBuffer::null() {
            bail!("L{} ratio4 state owner context/command 漂移", self.layer);
        }
        if *self.lock_phase()? != expected {
            bail!("L{} ratio4 state owner phase 不是 {expected:?}", self.layer);
        }
        self.core.require_active_layer(self.layer)
    }

    fn validate_call(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
    ) -> Result<()> {
        self.candidate_state_binding()
            .validate(self.candidate_state_owner())?;
        if !std::ptr::eq(ctx, self.core.context.as_ref())
            || command == vk::CommandBuffer::null()
            || workspace.static_weights.offset != 0
            || workspace.static_logical_bytes != self.recipes[0].static_layer_bytes
            || workspace.index_weights_proj_weight_offset
                != self.static_offsets.index_weights_proj_weight
            || workspace.index_query_weight_offset != self.static_offsets.index_query_weight
            || workspace.index_query_scale_offset != self.static_offsets.index_query_scale
        {
            bail!(
                "L{} ratio4 state owner context/static recipe 漂移",
                self.layer
            );
        }
        validate_slice(
            workspace.hc_branch_f32,
            BLOCK_SIZE as u64 * HIDDEN as u64 * F32_BYTES,
            "HC branch F32",
        )?;
        validate_slice(
            workspace.position4_query_low_f32,
            QUERY_LOW as u64 * F32_BYTES,
            "position4 query-low",
        )?;
        validate_slice(
            workspace.raw_index_query_bf16,
            RAW_INDEX_QUERY as u64 * BF16_BYTES,
            "raw index-query",
        )?;
        validate_slice(
            workspace.position4_head_weights_bf16,
            INDEX_HEADS as u64 * BF16_BYTES,
            "index-head",
        )?;
        validate_slice(workspace.sticky_status_u32, 4, "sticky status")?;
        Ok(())
    }

    unsafe fn copy_hc_lane(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        source: StorageBufferSlice<'_>,
        lane: usize,
    ) -> Result<()> {
        let bytes = u64::from(HIDDEN) * F32_BYTES;
        let source_offset = source
            .offset
            .checked_add(lane as u64 * bytes)
            .context("ratio4 HC lane source offset overflow")?;
        validate_slice(
            StorageBufferSlice {
                buffer: source.buffer,
                offset: source_offset,
            },
            bytes,
            "HC lane",
        )?;
        let target_end = self
            .compressor_input_offset
            .checked_add(bytes)
            .context("ratio4 compressor input range overflow")?;
        if target_end > self.recipes[0].workspace_bytes {
            bail!("ratio4 compressor input offset 越界");
        }
        let acquire = [
            vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(source.buffer.handle())
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
                .buffer(self.core.workspace.handle())
                .offset(self.compressor_input_offset)
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
            source.buffer.handle(),
            self.core.workspace.handle(),
            &[vk::BufferCopy::default()
                .src_offset(source_offset)
                .dst_offset(self.compressor_input_offset)
                .size(bytes)],
        );
        let publish = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .buffer(self.core.workspace.handle())
            .offset(self.compressor_input_offset)
            .size(bytes);
        ctx.device.cmd_pipeline_barrier(
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

    unsafe fn record_finalize(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
        candidate_base: u64,
    ) -> Result<Vec<DescriptorBinder>> {
        let mut binders = Vec::with_capacity(8);
        let result = (|| -> Result<()> {
            let status = workspace.sticky_status_u32;
            for (kind, kv, score, norm, output) in [
                (
                    S14Ratio4CompressorKind::Main,
                    self.state.main_kv_state,
                    self.state.main_score_state,
                    self.static_offsets.main_compressor_norm_weight,
                    self.state.first_compressed_kv,
                ),
                (
                    S14Ratio4CompressorKind::Indexer,
                    self.state.indexer_kv_state,
                    self.state.indexer_score_state,
                    self.static_offsets.indexer_compressor_norm_weight,
                    self.state.first_indexer_row,
                ),
            ] {
                let pool = self.core.finalize.pool.bind_slices(
                    ctx,
                    kind,
                    StorageBufferSlice {
                        buffer: self.core.prefix_arena.buffer(),
                        offset: candidate_base + kv,
                    },
                    StorageBufferSlice {
                        buffer: self.core.prefix_arena.buffer(),
                        offset: candidate_base + score,
                    },
                    StorageBufferSlice {
                        buffer: &self.core.workspace,
                        offset: self.core.scratch.main_a,
                    },
                    status,
                )?;
                self.core.finalize.pool.cmd(ctx, command, &pool);
                compute_barrier(ctx, command);
                binders.push(pool.binder);

                let rms = self.core.finalize.rmsnorm.bind_slices(
                    ctx,
                    kind.rmsnorm_shape()?,
                    1.0e-6,
                    StorageBufferSlice {
                        buffer: &self.core.workspace,
                        offset: self.core.scratch.main_a,
                    },
                    StorageBufferSlice {
                        buffer: workspace.static_weights.buffer,
                        offset: norm,
                    },
                    StorageBufferSlice {
                        buffer: &self.core.workspace,
                        offset: self.core.scratch.inverse_rms,
                    },
                    StorageBufferSlice {
                        buffer: &self.core.workspace,
                        offset: self.core.scratch.main_b,
                    },
                    status,
                )?;
                self.core.finalize.rmsnorm.cmd(ctx, command, &rms);
                compute_barrier(ctx, command);
                binders.push(rms.binder);

                let rope = self.core.finalize.rope.bind_slices(
                    ctx,
                    kind,
                    StorageBufferSlice {
                        buffer: &self.core.workspace,
                        offset: self.core.scratch.main_b,
                    },
                    StorageBufferSlice::whole(&self.core.compressed_rope),
                    StorageBufferSlice {
                        buffer: &self.core.workspace,
                        offset: self.core.scratch.main_c,
                    },
                    status,
                )?;
                self.core.finalize.rope.cmd(ctx, command, &rope);
                compute_barrier(ctx, command);
                binders.push(rope.binder);

                match kind {
                    S14Ratio4CompressorKind::Main => {
                        let qdq = self.core.finalize.main_qdq_bf16.bind_slices(
                            ctx,
                            StorageBufferSlice {
                                buffer: &self.core.workspace,
                                offset: self.core.scratch.main_c,
                            },
                            StorageBufferSlice {
                                buffer: self.core.prefix_arena.buffer(),
                                offset: candidate_base + output,
                            },
                            status,
                        )?;
                        self.core.finalize.main_qdq_bf16.cmd(ctx, command, &qdq);
                        compute_barrier(ctx, command);
                        binders.push(qdq.binder);
                    }
                    S14Ratio4CompressorKind::Indexer => {
                        let qdq = self.core.finalize.indexer_hadamard_qdq.bind_slices(
                            ctx,
                            StorageBufferSlice {
                                buffer: &self.core.workspace,
                                offset: self.core.scratch.main_c,
                            },
                            StorageBufferSlice {
                                buffer: self.core.prefix_arena.buffer(),
                                offset: candidate_base + output,
                            },
                            status,
                        )?;
                        self.core
                            .finalize
                            .indexer_hadamard_qdq
                            .cmd(ctx, command, &qdq);
                        compute_barrier(ctx, command);
                        binders.push(qdq.binder);
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            destroy_binders(ctx, &mut binders);
            return Err(error);
        }
        Ok(binders)
    }

    unsafe fn record_position4_projections(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        workspace: S14CausalBlockRatio4StateWorkspaceBindings<'_>,
    ) -> Result<Vec<DescriptorBinder>> {
        let mut binders = Vec::with_capacity(4);
        let result = (|| -> Result<()> {
            let index_head = self.core.numeric.bind_bf16_matvec_arenas(
                ctx,
                S14Bf16MatvecShape::new(INDEX_HEADS, HIDDEN, 1)?,
                workspace.static_weights.buffer,
                workspace.static_logical_bytes,
                self.static_offsets.index_weights_proj_weight,
                workspace.hc_branch_f32.buffer,
                workspace.hc_branch_f32.buffer.size(),
                workspace.hc_branch_f32.offset + POSITION4_LANE as u64 * HIDDEN as u64 * F32_BYTES,
                &self.core.workspace,
                self.core.scratch.bytes,
                self.core.scratch.index_head_f32,
            )?;
            self.core.numeric.cmd_bf16_matvec(ctx, command, &index_head);
            compute_barrier(ctx, command);
            binders.push(index_head.binder);
            let head_bf16 = self.core.f32_to_bf16.bind_slices(
                ctx,
                S14F32ToBf16Shape::new(INDEX_HEADS)?,
                StorageBufferSlice {
                    buffer: &self.core.workspace,
                    offset: self.core.scratch.index_head_f32,
                },
                workspace.position4_head_weights_bf16,
                workspace.sticky_status_u32,
            )?;
            self.core.f32_to_bf16.cmd(ctx, command, &head_bf16);
            compute_barrier(ctx, command);
            binders.push(head_bf16.binder);

            let raw_query = DescriptorBinder::new_with_offsets(
                ctx,
                &self.core.fp8_exact,
                &[
                    (
                        workspace.position4_query_low_f32.buffer,
                        workspace.position4_query_low_f32.offset,
                        u64::from(QUERY_LOW) * F32_BYTES,
                    ),
                    (
                        workspace.static_weights.buffer,
                        self.static_offsets.index_query_weight,
                        u64::from(RAW_INDEX_QUERY) * u64::from(QUERY_LOW),
                    ),
                    (
                        workspace.static_weights.buffer,
                        self.static_offsets.index_query_scale,
                        u64::from(RAW_INDEX_QUERY / 128) * u64::from(QUERY_LOW / 128),
                    ),
                    (
                        &self.core.workspace,
                        self.core.scratch.raw_index_query_f32,
                        u64::from(RAW_INDEX_QUERY) * F32_BYTES,
                    ),
                ],
            )?;
            record_fp8_exact(ctx, command, &self.core.fp8_exact, &raw_query);
            compute_barrier(ctx, command);
            binders.push(raw_query);
            let raw_bf16 = self.core.f32_to_bf16.bind_slices(
                ctx,
                S14F32ToBf16Shape::new(RAW_INDEX_QUERY)?,
                StorageBufferSlice {
                    buffer: &self.core.workspace,
                    offset: self.core.scratch.raw_index_query_f32,
                },
                workspace.raw_index_query_bf16,
                workspace.sticky_status_u32,
            )?;
            self.core.f32_to_bf16.cmd(ctx, command, &raw_bf16);
            compute_barrier(ctx, command);
            binders.push(raw_bf16.binder);
            Ok(())
        })();
        if let Err(error) = result {
            destroy_binders(ctx, &mut binders);
            return Err(error);
        }
        Ok(binders)
    }

    fn lock_phase(&self) -> Result<std::sync::MutexGuard<'_, LayerPhase>> {
        self.phase
            .lock()
            .map_err(|_| anyhow!("L{} ratio4 state phase poisoned", self.layer))
    }

    fn set_phase(&self, phase: LayerPhase) -> Result<()> {
        *self.lock_phase()? = phase;
        Ok(())
    }

    fn poison(&self) {
        if let Ok(mut phase) = self.phase.lock() {
            *phase = LayerPhase::Poisoned;
        }
        self.core.abort_layer(self.layer);
    }
}

impl SharedCore {
    fn new(
        context: Arc<VulkanContext>,
        prefix_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
        prefix_program: S14CausalBlockSharedPrefixStateProgram,
        scratch: ScratchLayout,
    ) -> Result<Self> {
        let mut guard = CoreBuildGuard::new(Arc::clone(&context));
        guard.workspace = Some(device_buffer(&context, scratch.bytes)?);
        guard.compressed_rope = Some(host_storage_buffer(&context, 64 * F32_BYTES)?);
        let rope = S14Ratio4CompressorBoundary::new(3)?.rope_cos_sin()?;
        unsafe {
            guard
                .compressed_rope
                .as_ref()
                .expect("compressed rope allocated")
                .write_at(0, bytemuck::cast_slice(&rope));
        }
        guard.numeric = Some(S14NumericPipelines::new(&context)?);
        guard.ape = Some(S14Position0ApeAddPipeline::new(&context)?);
        guard.finalize = Some(S14Ratio4CompressorFinalizePipelines::new(&context)?);
        guard.f32_to_bf16 = Some(S14F32ToBf16Pipeline::new(&context)?);
        guard.fp8_exact = Some(ComputePipeline::new(
            &context,
            S14_CAUSAL_BLOCK_FP8_MATVEC_EXACT_SPV,
            4,
            12,
        )?);
        Ok(Self {
            context,
            prefix_arena,
            prefix_program,
            workspace: guard.workspace.take().unwrap(),
            scratch,
            compressed_rope: guard.compressed_rope.take().unwrap(),
            numeric: guard.numeric.take().unwrap(),
            ape: guard.ape.take().unwrap(),
            finalize: guard.finalize.take().unwrap(),
            f32_to_bf16: guard.f32_to_bf16.take().unwrap(),
            fp8_exact: guard.fp8_exact.take().unwrap(),
            pending_binders: Mutex::new(Vec::new()),
            active_layer: Mutex::new(None),
        })
    }

    fn begin_layer(&self, layer: u8) -> Result<()> {
        let mut active = self
            .active_layer
            .lock()
            .map_err(|_| anyhow!("ratio4 core active-layer poisoned"))?;
        if active.is_some() {
            bail!("ratio4 core 仍有 active layer: {active:?}");
        }
        self.destroy_pending()?;
        *active = Some(layer);
        Ok(())
    }

    fn require_active_layer(&self, layer: u8) -> Result<()> {
        if *self
            .active_layer
            .lock()
            .map_err(|_| anyhow!("ratio4 core active-layer poisoned"))?
            != Some(layer)
        {
            bail!("ratio4 core active layer 不是 L{layer}");
        }
        Ok(())
    }

    fn finish_layer(&self, layer: u8) -> Result<()> {
        let mut active = self
            .active_layer
            .lock()
            .map_err(|_| anyhow!("ratio4 core active-layer poisoned"))?;
        if *active != Some(layer) {
            bail!("ratio4 core finish layer 不是 L{layer}");
        }
        *active = None;
        Ok(())
    }

    fn abort_layer(&self, layer: u8) {
        if let Ok(mut active) = self.active_layer.lock() {
            if *active == Some(layer) {
                *active = None;
            }
        }
    }

    fn append_binders(&self, mut binders: Vec<DescriptorBinder>) -> Result<()> {
        self.pending_binders
            .lock()
            .map_err(|_| anyhow!("ratio4 pending binders poisoned"))?
            .append(&mut binders);
        Ok(())
    }

    fn destroy_pending(&self) -> Result<()> {
        let mut pending = self
            .pending_binders
            .lock()
            .map_err(|_| anyhow!("ratio4 pending binders poisoned"))?;
        for binder in pending.drain(..) {
            binder.destroy(&self.context);
        }
        Ok(())
    }

    fn destroy(self, context: &VulkanContext) -> Result<()> {
        if !std::ptr::eq(context, self.context.as_ref()) {
            bail!("ratio4 core destroy context 漂移");
        }
        if self
            .active_layer
            .lock()
            .map_err(|_| anyhow!("ratio4 core active-layer poisoned"))?
            .is_some()
        {
            bail!("ratio4 core 仍有 active command，禁止 destroy");
        }
        self.destroy_pending()?;
        self.fp8_exact.destroy(context);
        self.f32_to_bf16.destroy(context);
        self.finalize.destroy(context);
        self.ape.destroy(context);
        self.numeric.destroy(context);
        self.compressed_rope.destroy(context);
        self.workspace.destroy(context);
        Ok(())
    }
}

struct CoreBuildGuard {
    context: Arc<VulkanContext>,
    workspace: Option<GpuBuffer>,
    compressed_rope: Option<GpuBuffer>,
    numeric: Option<S14NumericPipelines>,
    ape: Option<S14Position0ApeAddPipeline>,
    finalize: Option<S14Ratio4CompressorFinalizePipelines>,
    f32_to_bf16: Option<S14F32ToBf16Pipeline>,
    fp8_exact: Option<ComputePipeline>,
}

impl CoreBuildGuard {
    fn new(context: Arc<VulkanContext>) -> Self {
        Self {
            context,
            workspace: None,
            compressed_rope: None,
            numeric: None,
            ape: None,
            finalize: None,
            f32_to_bf16: None,
            fp8_exact: None,
        }
    }
}

impl Drop for CoreBuildGuard {
    fn drop(&mut self) {
        if let Some(value) = self.fp8_exact.take() {
            value.destroy(&self.context);
        }
        if let Some(value) = self.f32_to_bf16.take() {
            value.destroy(&self.context);
        }
        if let Some(value) = self.finalize.take() {
            value.destroy(&self.context);
        }
        if let Some(value) = self.ape.take() {
            value.destroy(&self.context);
        }
        if let Some(value) = self.numeric.take() {
            value.destroy(&self.context);
        }
        if let Some(value) = self.compressed_rope.take() {
            value.destroy(&self.context);
        }
        if let Some(value) = self.workspace.take() {
            value.destroy(&self.context);
        }
    }
}

fn validate_global_inputs(
    context: &Arc<VulkanContext>,
    prefix_arena: &Arc<S14CausalBlockPrefixCheckpointArena>,
    prefix_program: &S14CausalBlockPrefixStateProgram,
    graph: &S14Position0FullDepthLayerProgram,
    authoritative: &NativeState,
) -> Result<()> {
    if !Arc::ptr_eq(context, prefix_arena.context())
        || prefix_arena.base_position() != BASE_POSITION
        || prefix_arena.layout().block_size != BLOCK_SIZE
        || prefix_program.base_position() != BASE_POSITION
        || prefix_program.block_size() != BLOCK_SIZE
        || authoritative.position != BASE_POSITION
        || authoritative.arena_bytes != prefix_arena.layout().checkpoint_state_bytes
        || graph.layers.len() != FULL_DEPTH_LAYERS.len()
    {
        bail!("ratio4 owner context/base/K/arena/43层 identity 漂移");
    }
    Ok(())
}

fn validate_recipes(
    layer: u8,
    recipes: &[S14Position0LayerStateRecordingRecipe; BLOCK_SIZE],
    workspace_bytes: u64,
    candidate_bytes: u64,
    static_bytes: u64,
) -> Result<u64> {
    for (lane, recipe) in recipes.iter().enumerate() {
        if recipe.layer != layer
            || recipe.position != BASE_POSITION + lane as u32
            || recipe.compress_ratio != 4
            || recipe.workspace_bytes != workspace_bytes
            || recipe.candidate_state_bytes != candidate_bytes
            || recipe.static_layer_bytes != static_bytes
            || recipe.compressor_ops.len() != 10
        {
            bail!(
                "L{layer} ratio4 position{} recipe identity 漂移",
                BASE_POSITION + lane as u32
            );
        }
    }
    if recipes[POSITION3_LANE].rollover_copies.len() != 16
        || !recipes[0].rollover_copies.is_empty()
        || !recipes[1].rollover_copies.is_empty()
        || !recipes[POSITION4_LANE].rollover_copies.is_empty()
    {
        bail!("L{layer} ratio4 position1..4 rollover recipe 漂移");
    }
    let mut input = None;
    for recipe in recipes {
        for op in &recipe.compressor_ops {
            if let S14Position0StateRecordingOp::Projection {
                input_offset, k, ..
            } = op
            {
                if *k != HIDDEN {
                    bail!("L{layer} ratio4 compressor projection K 漂移");
                }
                match input {
                    Some(expected) if expected != *input_offset => {
                        bail!("L{layer} ratio4 compressor HC input offset 漂移")
                    }
                    None => input = Some(*input_offset),
                    _ => {}
                }
            }
        }
    }
    input.context("ratio4 recipe 缺少 compressor projection")
}

fn build_state_offsets(
    authoritative: &NativeState,
    layer: u8,
    recipes: &[S14Position0LayerStateRecordingRecipe; BLOCK_SIZE],
) -> Result<StateOffsets> {
    let main = unique(
        authoritative
            .compressors
            .iter()
            .filter(|entry| entry.layer == layer && entry.compress_ratio == 4),
        &format!("L{layer} ratio4 main compressor"),
    )?;
    let indexer = unique(
        authoritative
            .indexers
            .iter()
            .filter(|entry| entry.layer == layer),
        &format!("L{layer} ratio4 indexer"),
    )?;
    if main.kv_state.dtype != DType::F32
        || main.kv_state.shape != [1, 8, 1024]
        || main.score_state.dtype != DType::F32
        || main.score_state.shape != [1, 8, 1024]
        || indexer.compressor_kv_state.dtype != DType::F32
        || indexer.compressor_kv_state.shape != [1, 8, 256]
        || indexer.compressor_score_state.dtype != DType::F32
        || indexer.compressor_score_state.shape != [1, 8, 256]
    {
        bail!("L{layer} ratio4 compressor state dtype/shape 漂移");
    }
    let mut position3 = authoritative.clone();
    position3.position = 3;
    let layout = S14Position0StateWritebackLayout::build(&position3)?;
    let layer_layout = layout
        .layer(layer)
        .context("position3 ratio4 state layout 缺层")?;
    let main_target = layer_layout.row(S14Position0StateRowKind::MainCompressedKv)?;
    let indexer_target = layer_layout.row(S14Position0StateRowKind::IndexerCompressedKv)?;
    if main_target.state_range.end - main_target.state_range.start != 512 * BF16_BYTES
        || indexer_target.state_range.end - indexer_target.state_range.start != 128 * BF16_BYTES
        || !recipes[POSITION3_LANE]
            .state_ranges_written
            .contains(&main_target.state_range)
        || !recipes[POSITION3_LANE]
            .state_ranges_written
            .contains(&indexer_target.state_range)
    {
        bail!("L{layer} ratio4 position3 compressed target/recipe 漂移");
    }
    Ok(StateOffsets {
        main_kv_state: main.kv_state.offset,
        main_score_state: main.score_state.offset,
        indexer_kv_state: indexer.compressor_kv_state.offset,
        indexer_score_state: indexer.compressor_score_state.offset,
        first_compressed_kv: main_target.state_range.start,
        first_indexer_row: indexer_target.state_range.start,
    })
}

fn exact_static_offset(
    program: &S14Position0LayerProgram,
    tensor: &str,
    expected_bytes: u64,
) -> Result<u64> {
    let binding = unique(
        program
            .static_weights
            .iter()
            .filter(|entry| entry.tensor == tensor),
        tensor,
    )?;
    if binding.arena != S14Position0WeightArena::StaticLayer(program.layer)
        || binding.bytes != expected_bytes
        || binding
            .offset
            .checked_add(binding.bytes)
            .is_none_or(|end| end > program.static_layer_bytes)
    {
        bail!("L{} {tensor} static offset/bytes 漂移", program.layer);
    }
    Ok(binding.offset)
}

fn unique<'a, T>(mut values: impl Iterator<Item = &'a T>, label: &str) -> Result<&'a T> {
    let value = values.next().with_context(|| format!("{label} 缺失"))?;
    if values.next().is_some() {
        bail!("{label} 不唯一");
    }
    Ok(value)
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
        bail!("ratio4 owner {label} slice 越界或未对齐");
    }
    Ok(())
}

unsafe fn record_fp8_exact(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    binder: &DescriptorBinder,
) {
    ctx.device
        .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
    ctx.device.cmd_bind_descriptor_sets(
        command,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.layout,
        0,
        &[binder.set],
        &[],
    );
    let mut push = [0u8; 12];
    push[..4].copy_from_slice(&RAW_INDEX_QUERY.to_le_bytes());
    push[4..8].copy_from_slice(&QUERY_LOW.to_le_bytes());
    push[8..].copy_from_slice(&1u32.to_le_bytes());
    ctx.device.cmd_push_constants(
        command,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &push,
    );
    ctx.device.cmd_dispatch(command, RAW_INDEX_QUERY, 1, 1);
}

unsafe fn compute_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
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

fn destroy_binders(ctx: &VulkanContext, binders: &mut Vec<DescriptorBinder>) {
    for binder in binders.drain(..) {
        binder.destroy(ctx);
    }
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

fn host_storage_buffer(context: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        context,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(),
        true,
    )
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("ratio4 owner alignment overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_layout_separates_compressor_and_position4_projection_outputs() {
        let layout = ScratchLayout::build(2 * 1024 * 1024).unwrap();
        assert_eq!(layout.main_a % ALIGNMENT, 0);
        assert!(layout.main_a >= 2 * 1024 * 1024);
        assert!(layout.main_a < layout.main_b);
        assert!(layout.main_b < layout.main_c);
        assert!(layout.main_c < layout.inverse_rms);
        assert!(layout.inverse_rms < layout.index_head_f32);
        assert!(layout.index_head_f32 < layout.raw_index_query_f32);
        assert!(layout.raw_index_query_f32 + RAW_INDEX_QUERY as u64 * F32_BYTES <= layout.bytes);
    }
}
