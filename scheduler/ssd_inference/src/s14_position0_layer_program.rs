//! FullDepth43/native-top6 position0 的参数化逐层绑定与录制计划。
//!
//! 本模块不拥有 Vulkan 对象，也不执行数值计算。它把真实 manifest、Hybrid 权重计划和
//! 单 arena workspace 收敛成 L0..L42 的唯一层程序，供 concrete backend 逐项绑定已有
//! offset API。计划本身保持 fail-closed：43 层、原生 top-6、A/B hidden ping-pong、
//! candidate state 写集和 transfer/compute timeline 任一漂移都会拒绝。

use crate::{
    s14_position0_hybrid_weight_arena::{
        S14Position0HybridArenaLayout, S14Position0PhysicalAssetPlacement,
        S14Position0StaticLayerLayout,
    },
    s14_position0_weight_plan::{
        S14Position0AssetPlacement, S14Position0HybridWeightPlan, S14_POSITION0_ROLLING_BANKS,
    },
    s14_position0_workspace::{
        S14Position0WorkspaceLayout, S14Position0WorkspaceRegion, S14Position0WorkspaceSlot,
    },
};
use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{
    Position0Asset, Position0Layer, Position0WholeTokenManifest, EXPERTS_PER_TOKEN,
    FULL_DEPTH_LAYERS,
};
use std::{collections::BTreeMap, ops::Range, path::PathBuf};

pub const S14_POSITION0_LAYER_PROGRAM_PROFILE: &str = "FullDepth43/native-top6";
pub const S14_POSITION0_ROUTED_TENSORS_PER_EXPERT: usize = 6;
pub const S14_STATE_WINDOW_ROWS: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum S14Position0RouteProgram {
    PhysicalTid2Eid,
    BiasTop6,
}

impl S14Position0RouteProgram {
    fn from_manifest(layer: &Position0Layer) -> Result<Self> {
        match layer.route_source.as_str() {
            "current_token_tid2eid_physical_i64" if layer.layer < 3 => Ok(Self::PhysicalTid2Eid),
            "sqrtsoftplus_plus_bias_top6" if layer.layer >= 3 => Ok(Self::BiasTop6),
            _ => bail!("L{} route program/source 漂移", layer.layer),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0CompressorProgram {
    None,
    Ratio4WithIndexer,
    Ratio128,
}

impl S14Position0CompressorProgram {
    fn from_ratio(layer: u8, ratio: u16) -> Result<Self> {
        match ratio {
            0 if layer < 2 => Ok(Self::None),
            4 => Ok(Self::Ratio4WithIndexer),
            128 => Ok(Self::Ratio128),
            _ => bail!("L{layer} compressor program/ratio 漂移: {ratio}"),
        }
    }
}

/// 单层在一个 token position 上对递归 state 的精确读写行合同。
///
/// 描述任意 token position 的 window/remainder/压缩边界。数值 backend 必须依据
/// `compressed_block_ready` 选择普通 remainder 或完成块路径，并用
/// `window_start_row/window_count` 以逻辑时间顺序读取环形 window。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14TokenLayerStateAccess {
    pub position: u32,
    pub committed_window_rows: Range<u32>,
    pub candidate_window_row: u32,
    /// 写入当前 token 后，attention 应消费的逻辑 window 行数与首个物理行。
    pub window_count: u32,
    pub window_start_row: u32,
    pub ape_row: Option<u16>,
    pub remainder_row: Option<u16>,
    pub compressed_block_ready: bool,
    /// 写入当前 token 后可供本轮 attention 消费的 compressed block 总数。
    pub compressed_count: u32,
    pub compressed_block_index: Option<u32>,
    pub compressed_rope_position: Option<u32>,
    /// ratio4 边界完成后需把 active half row4..7 镜像到 prefix row0..3。
    pub overlap_rollover_rows: Option<(Range<u16>, Range<u16>)>,
}

impl S14TokenLayerStateAccess {
    pub fn for_ratio(position: u32, compress_ratio: u16) -> Result<Self> {
        let next_position = position
            .checked_add(1)
            .ok_or_else(|| anyhow!("S14 layer-state access position overflow"))?;
        let candidate_window_row = position % S14_STATE_WINDOW_ROWS;
        let committed_window_rows = 0..position.min(S14_STATE_WINDOW_ROWS);
        let window_count = next_position.min(S14_STATE_WINDOW_ROWS);
        let window_start_row = if next_position <= S14_STATE_WINDOW_ROWS {
            0
        } else {
            next_position % S14_STATE_WINDOW_ROWS
        };
        let (ape_row, remainder_row, compressed_block_ready) = match compress_ratio {
            0 => (None, None, false),
            4 => {
                let row = (position % 4) as u16;
                (Some(row), Some(4 + row), next_position % 4 == 0)
            }
            128 => {
                let row = (position % 128) as u16;
                (Some(row), Some(row), next_position % 128 == 0)
            }
            ratio => bail!("S14 未注册 compressor ratio {ratio}"),
        };
        let compressed_count = match compress_ratio {
            0 => 0,
            ratio => next_position / u32::from(ratio),
        };
        Ok(Self {
            position,
            committed_window_rows,
            candidate_window_row,
            window_count,
            window_start_row,
            ape_row,
            remainder_row,
            compressed_block_ready,
            compressed_count,
            compressed_block_index: if compressed_block_ready {
                Some(position / u32::from(compress_ratio))
            } else {
                None
            },
            compressed_rope_position: if compressed_block_ready {
                Some(position + 1 - u32::from(compress_ratio))
            } else {
                None
            },
            overlap_rollover_rows: (compress_ratio == 4 && compressed_block_ready)
                .then_some((4..8, 0..4)),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0StateWriteKind {
    WindowKv,
    MainCompressorKvRemainder,
    MainCompressorScoreRemainder,
    IndexerCompressorKvRemainder,
    IndexerCompressorScoreRemainder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0WeightClass {
    Static,
    Router,
    Shared,
    Routed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0WeightArena {
    /// Paged arena 可把它解析为常驻层页或当前 parity 的 static stream bank；offset
    /// 始终是逐层物理布局的 local offset，不能使用 hybrid resident 全局 offset。
    StaticLayer(u8),
    RoutedBank(usize),
}

/// 一条 descriptor 所需的精确权重子范围。`offset` 始终相对于 `arena`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0WeightBinding {
    pub tensor: String,
    pub kind: String,
    pub expert_id: Option<u16>,
    pub path: PathBuf,
    pub sha256: String,
    pub offset: u64,
    pub bytes: u64,
    pub arena: S14Position0WeightArena,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0HiddenBinding {
    pub input_slot: S14Position0WorkspaceSlot,
    pub input: S14Position0WorkspaceRegion,
    pub output_slot: S14Position0WorkspaceSlot,
    pub output: S14Position0WorkspaceRegion,
}

/// timeline 数值是 candidate 内的相对序号，不是进程全局 semaphore value。
/// prologue embedding 占 compute ordinal 1；层内禁止 host wait。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0LayerTimelineProgram {
    pub transfer_signal_ordinal: u64,
    pub compute_wait_transfer_ordinal: u64,
    pub compute_signal_ordinal: u64,
    pub bank_reuse_after_compute_ordinal: Option<u64>,
    pub host_wait_allowed: bool,
}

/// 现有数值原语到逐层图的映射。枚举表示录制阶段，不携带固定 capture 或输出。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0LayerPrimitive {
    /// `S14NumericPipelines::{bind_hc_normalize_input_arena,
    /// bind_hc_split_reduce_norm_arenas}`。
    AttentionHcPre,
    /// `S14Bf16RmsNormPipeline::bind_slices` 与 attention packed-FP8/grouped API。
    AttentionProjections,
    /// ratio4/128 的 compressor；ratio4 还包括 indexer 投影。
    CompressorIndexer,
    /// `S14Position0AttentionPipeline::bind_slices`。
    Position0Attention,
    /// `S14NumericPipelines::{bind_grouped_fp8_bf16_weight_arenas,
    /// bind_fp8_arenas}`。
    AttentionOutputProjection,
    /// `S14HcPostPipeline::bind_slices`。
    AttentionHcPost,
    /// 与 attention HC-pre 相同的 arena offset API，使用 FFN HC 参数。
    FfnHcPre,
    /// `S14NumericPipelines::bind_bf16_matvec_arenas`。
    RouterProjection,
    /// `S14RoutePostprocessGpuPipeline::bind_with_offsets`，mode 由 route program 决定。
    RoutePostprocess,
    /// `S14NumericPipelines::bind_ragged_mxfp4_arenas`，三投影共用 routed bank。
    RoutedTop6Matvec,
    /// `S14NumericPipelines::bind_batched_official_expert_prepare_arena`。
    RoutedOfficialPrepare,
    /// shared expert 的 FP8 w1/w3/w2 与 SwiGLU。
    SharedExpert,
    /// `S14NumericPipelines::bind_exact_order_block_reduce_arena`。
    ExactOrderReduce,
    /// `S14HcPostPipeline::bind_slices`，输出到下一 hidden stream。
    FfnHcPost,
    /// candidate state buffer 的 window KV 与 compressor remainder 写回。
    CandidateStateWriteback,
}

const COMMON_PREFIX: [S14Position0LayerPrimitive; 2] = [
    S14Position0LayerPrimitive::AttentionHcPre,
    S14Position0LayerPrimitive::AttentionProjections,
];

const COMMON_SUFFIX: [S14Position0LayerPrimitive; 12] = [
    S14Position0LayerPrimitive::Position0Attention,
    S14Position0LayerPrimitive::AttentionOutputProjection,
    S14Position0LayerPrimitive::AttentionHcPost,
    S14Position0LayerPrimitive::FfnHcPre,
    S14Position0LayerPrimitive::RouterProjection,
    S14Position0LayerPrimitive::RoutePostprocess,
    S14Position0LayerPrimitive::RoutedTop6Matvec,
    S14Position0LayerPrimitive::RoutedOfficialPrepare,
    S14Position0LayerPrimitive::SharedExpert,
    S14Position0LayerPrimitive::ExactOrderReduce,
    S14Position0LayerPrimitive::FfnHcPost,
    S14Position0LayerPrimitive::CandidateStateWriteback,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0LayerProgram {
    pub layer: u8,
    pub index: usize,
    pub routed_bank: usize,
    pub static_layer_bytes: u64,
    pub route: S14Position0RouteProgram,
    pub compressor: S14Position0CompressorProgram,
    pub hidden: S14Position0HiddenBinding,
    pub static_weights: Vec<S14Position0WeightBinding>,
    pub router_weights: Vec<S14Position0WeightBinding>,
    pub shared_weights: Vec<S14Position0WeightBinding>,
    pub routed_weights: Vec<S14Position0WeightBinding>,
    pub routed_expert_ids: [u16; EXPERTS_PER_TOKEN],
    pub state_writes: Vec<S14Position0StateWriteKind>,
    pub primitives: Vec<S14Position0LayerPrimitive>,
    pub timeline: S14Position0LayerTimelineProgram,
}

impl S14Position0LayerProgram {
    pub fn weights(&self, class: S14Position0WeightClass) -> &[S14Position0WeightBinding] {
        match class {
            S14Position0WeightClass::Static => &self.static_weights,
            S14Position0WeightClass::Router => &self.router_weights,
            S14Position0WeightClass::Shared => &self.shared_weights,
            S14Position0WeightClass::Routed => &self.routed_weights,
        }
    }

    pub fn state_access(&self, position: u32) -> Result<S14TokenLayerStateAccess> {
        let ratio = match self.compressor {
            S14Position0CompressorProgram::None => 0,
            S14Position0CompressorProgram::Ratio4WithIndexer => 4,
            S14Position0CompressorProgram::Ratio128 => 128,
        };
        S14TokenLayerStateAccess::for_ratio(position, ratio)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0FullDepthLayerProgram {
    pub profile: &'static str,
    pub prologue_compute_signal_ordinal: u64,
    pub terminal_host_waits: u64,
    pub workspace_used_bytes: u64,
    pub layers: Vec<S14Position0LayerProgram>,
}

impl S14Position0FullDepthLayerProgram {
    pub fn build(
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        workspace: &S14Position0WorkspaceLayout,
    ) -> Result<Self> {
        manifest
            .validate()
            .map_err(|error| anyhow!("position0 manifest invalid: {error}"))?;
        weights.validate(manifest)?;
        workspace.validate()?;
        let physical = S14Position0HybridArenaLayout::build(weights)?;

        let mut layers = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
        for (index, ((&expected_layer, layer), routed_plan)) in FULL_DEPTH_LAYERS
            .iter()
            .zip(&manifest.layers)
            .zip(&weights.routed_layers)
            .enumerate()
        {
            if layer.layer != expected_layer || routed_plan.layer != expected_layer {
                bail!("position0 layer/weight plan 顺序漂移 at index {index}");
            }
            let routed_bank = index % S14_POSITION0_ROLLING_BANKS;
            if routed_plan.bank != routed_bank {
                bail!("L{expected_layer} routed bank 漂移");
            }
            let compressor =
                S14Position0CompressorProgram::from_ratio(expected_layer, layer.compress_ratio)?;
            let route = S14Position0RouteProgram::from_manifest(layer)?;
            let static_layout = physical
                .static_layers
                .get(index)
                .ok_or_else(|| anyhow!("physical static layout 缺少 L{expected_layer}"))?;
            if static_layout.layer != expected_layer {
                bail!("physical static layout 层序漂移 at L{expected_layer}");
            }
            let (input_slot, output_slot) = hidden_slots(index);
            let hidden = S14Position0HiddenBinding {
                input_slot,
                input: workspace.region(input_slot),
                output_slot,
                output: workspace.region(output_slot),
            };
            let routed_expert_ids =
                layer.expert_ids.clone().try_into().map_err(|_| {
                    anyhow!("L{expected_layer} route 不是严格 top-{EXPERTS_PER_TOKEN}")
                })?;
            layers.push(S14Position0LayerProgram {
                layer: expected_layer,
                index,
                routed_bank,
                static_layer_bytes: static_layout.requested_bytes,
                route,
                compressor,
                hidden,
                static_weights: static_bindings(
                    &layer.assets.non_expert,
                    &weights.resident.assets,
                    static_layout,
                )
                .with_context(|| format!("bind L{expected_layer} static weights"))?,
                router_weights: static_bindings(
                    &layer.assets.router,
                    &weights.resident.assets,
                    static_layout,
                )
                .with_context(|| format!("bind L{expected_layer} router weights"))?,
                shared_weights: static_bindings(
                    &layer.assets.shared,
                    &weights.resident.assets,
                    static_layout,
                )
                .with_context(|| format!("bind L{expected_layer} shared weights"))?,
                routed_weights: routed_bindings(
                    &layer.assets.routed,
                    &routed_plan.assets,
                    routed_bank,
                )
                .with_context(|| format!("bind L{expected_layer} routed weights"))?,
                routed_expert_ids,
                state_writes: state_writes(compressor),
                primitives: primitives(compressor),
                timeline: layer_timeline(index),
            });
        }

        let program = Self {
            profile: S14_POSITION0_LAYER_PROGRAM_PROFILE,
            prologue_compute_signal_ordinal: 1,
            terminal_host_waits: 1,
            workspace_used_bytes: workspace.used_bytes(),
            layers,
        };
        program.validate(manifest, weights, workspace)?;
        Ok(program)
    }

    pub fn layer(&self, layer: u8) -> Option<&S14Position0LayerProgram> {
        self.layers
            .get(layer as usize)
            .filter(|entry| entry.layer == layer)
    }

    /// L0 已由首段 backend 录制后，主线继续消费的 L1..L42；此切片不能独立提交 token。
    pub fn remaining_after_l0(&self) -> &[S14Position0LayerProgram] {
        self.layers.get(1..).unwrap_or_default()
    }

    pub fn validate(
        &self,
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        workspace: &S14Position0WorkspaceLayout,
    ) -> Result<()> {
        manifest
            .validate()
            .map_err(|error| anyhow!("position0 manifest invalid: {error}"))?;
        weights.validate(manifest)?;
        workspace.validate()?;
        let physical = S14Position0HybridArenaLayout::build(weights)?;
        if self.profile != S14_POSITION0_LAYER_PROGRAM_PROFILE
            || self.prologue_compute_signal_ordinal != 1
            || self.terminal_host_waits != 1
            || self.workspace_used_bytes != workspace.used_bytes()
            || self.layers.len() != FULL_DEPTH_LAYERS.len()
            || self.remaining_after_l0().len() != 42
        {
            bail!("position0 FullDepth43 program 总账漂移");
        }

        let mut ratios = BTreeMap::<u16, usize>::new();
        let mut routes = BTreeMap::<S14Position0RouteProgram, usize>::new();
        for (index, (((&expected_layer, manifest_layer), routed_plan), layer)) in FULL_DEPTH_LAYERS
            .iter()
            .zip(&manifest.layers)
            .zip(&weights.routed_layers)
            .zip(&self.layers)
            .enumerate()
        {
            let expected_bank = index % S14_POSITION0_ROLLING_BANKS;
            let static_layout = physical
                .static_layers
                .get(index)
                .ok_or_else(|| anyhow!("physical static layout 缺少 L{expected_layer}"))?;
            let expected_route = S14Position0RouteProgram::from_manifest(manifest_layer)?;
            let expected_compressor = S14Position0CompressorProgram::from_ratio(
                expected_layer,
                manifest_layer.compress_ratio,
            )?;
            let (input_slot, output_slot) = hidden_slots(index);
            if layer.layer != expected_layer
                || layer.index != index
                || layer.routed_bank != expected_bank
                || layer.static_layer_bytes != static_layout.requested_bytes
                || routed_plan.bank != expected_bank
                || layer.route != expected_route
                || layer.compressor != expected_compressor
                || layer.hidden.input_slot != input_slot
                || layer.hidden.output_slot != output_slot
                || layer.hidden.input != workspace.region(input_slot)
                || layer.hidden.output != workspace.region(output_slot)
                || layer.routed_expert_ids.as_slice() != manifest_layer.expert_ids.as_slice()
                || layer.state_writes != state_writes(expected_compressor)
                || layer.primitives != primitives(expected_compressor)
                || layer.timeline != layer_timeline(index)
                || layer.timeline.host_wait_allowed
            {
                bail!("L{expected_layer} layer program 合同漂移");
            }
            if index > 0 && self.layers[index - 1].hidden.output != layer.hidden.input {
                bail!("L{expected_layer} hidden A/B ping-pong 断链");
            }
            validate_binding_class(
                &layer.static_weights,
                &manifest_layer.assets.non_expert,
                S14Position0WeightArena::StaticLayer(expected_layer),
                Some(static_layout),
                &weights.resident.assets,
            )?;
            validate_binding_class(
                &layer.router_weights,
                &manifest_layer.assets.router,
                S14Position0WeightArena::StaticLayer(expected_layer),
                Some(static_layout),
                &weights.resident.assets,
            )?;
            validate_binding_class(
                &layer.shared_weights,
                &manifest_layer.assets.shared,
                S14Position0WeightArena::StaticLayer(expected_layer),
                Some(static_layout),
                &weights.resident.assets,
            )?;
            validate_binding_class(
                &layer.routed_weights,
                &manifest_layer.assets.routed,
                S14Position0WeightArena::RoutedBank(expected_bank),
                None,
                &routed_plan.assets,
            )?;
            if layer.router_weights.len() != 2
                || layer.shared_weights.len() != 6
                || layer.routed_weights.len()
                    != EXPERTS_PER_TOKEN * S14_POSITION0_ROUTED_TENSORS_PER_EXPERT
            {
                bail!("L{expected_layer} router/shared/routed binding 数量漂移");
            }
            for &expert in &layer.routed_expert_ids {
                if layer
                    .routed_weights
                    .iter()
                    .filter(|binding| binding.expert_id == Some(expert))
                    .count()
                    != S14_POSITION0_ROUTED_TENSORS_PER_EXPERT
                {
                    bail!("L{expected_layer}/E{expert} routed 三投影权重/scale 不完整");
                }
            }
            *ratios.entry(manifest_layer.compress_ratio).or_default() += 1;
            *routes.entry(expected_route).or_default() += 1;
        }
        if ratios != BTreeMap::from([(0, 2), (4, 21), (128, 20)])
            || routes
                != BTreeMap::from([
                    (S14Position0RouteProgram::PhysicalTid2Eid, 3),
                    (S14Position0RouteProgram::BiasTop6, 40),
                ])
        {
            bail!("position0 layer class 分布漂移");
        }
        Ok(())
    }

    /// 录制模板只负责不可跳过的调用顺序；实际 descriptor、barrier 和 command buffer
    /// 由 recorder 持有。层间没有 host wait，L42 后由调用者继续同一 compute timeline。
    pub fn record_full_depth(
        &self,
        recorder: &mut dyn S14Position0LayerProgramRecorder,
    ) -> Result<()> {
        for layer in &self.layers {
            for class in [
                S14Position0WeightClass::Static,
                S14Position0WeightClass::Router,
                S14Position0WeightClass::Shared,
                S14Position0WeightClass::Routed,
            ] {
                recorder.bind_weights(layer.layer, class, layer.weights(class))?;
            }
            recorder.bind_workspace(layer.layer, layer.hidden)?;
            recorder.bind_state_writes(layer.layer, &layer.state_writes)?;
            recorder.record_layer(
                layer.layer,
                &layer.primitives,
                layer.route,
                layer.compressor,
                layer.timeline,
            )?;
        }
        Ok(())
    }
}

/// Concrete Vulkan backend 的最小适配面。实现者不能在此回调中等待 host。
pub trait S14Position0LayerProgramRecorder {
    fn bind_weights(
        &mut self,
        layer: u8,
        class: S14Position0WeightClass,
        bindings: &[S14Position0WeightBinding],
    ) -> Result<()>;

    fn bind_workspace(&mut self, layer: u8, hidden: S14Position0HiddenBinding) -> Result<()>;

    fn bind_state_writes(&mut self, layer: u8, writes: &[S14Position0StateWriteKind])
        -> Result<()>;

    fn record_layer(
        &mut self,
        layer: u8,
        primitives: &[S14Position0LayerPrimitive],
        route: S14Position0RouteProgram,
        compressor: S14Position0CompressorProgram,
        timeline: S14Position0LayerTimelineProgram,
    ) -> Result<()>;
}

fn hidden_slots(index: usize) -> (S14Position0WorkspaceSlot, S14Position0WorkspaceSlot) {
    if index % 2 == 0 {
        (
            S14Position0WorkspaceSlot::HiddenStreamsA,
            S14Position0WorkspaceSlot::HiddenStreamsB,
        )
    } else {
        (
            S14Position0WorkspaceSlot::HiddenStreamsB,
            S14Position0WorkspaceSlot::HiddenStreamsA,
        )
    }
}

fn state_writes(compressor: S14Position0CompressorProgram) -> Vec<S14Position0StateWriteKind> {
    let mut writes = vec![S14Position0StateWriteKind::WindowKv];
    match compressor {
        S14Position0CompressorProgram::None => {}
        S14Position0CompressorProgram::Ratio4WithIndexer => writes.extend([
            S14Position0StateWriteKind::MainCompressorKvRemainder,
            S14Position0StateWriteKind::MainCompressorScoreRemainder,
            S14Position0StateWriteKind::IndexerCompressorKvRemainder,
            S14Position0StateWriteKind::IndexerCompressorScoreRemainder,
        ]),
        S14Position0CompressorProgram::Ratio128 => writes.extend([
            S14Position0StateWriteKind::MainCompressorKvRemainder,
            S14Position0StateWriteKind::MainCompressorScoreRemainder,
        ]),
    }
    writes
}

fn primitives(compressor: S14Position0CompressorProgram) -> Vec<S14Position0LayerPrimitive> {
    let mut output = Vec::with_capacity(COMMON_PREFIX.len() + COMMON_SUFFIX.len() + 1);
    output.extend(COMMON_PREFIX);
    if compressor != S14Position0CompressorProgram::None {
        output.push(S14Position0LayerPrimitive::CompressorIndexer);
    }
    output.extend(COMMON_SUFFIX);
    output
}

fn layer_timeline(index: usize) -> S14Position0LayerTimelineProgram {
    let transfer = index as u64 + 1;
    let compute = index as u64 + 2;
    S14Position0LayerTimelineProgram {
        transfer_signal_ordinal: transfer,
        compute_wait_transfer_ordinal: transfer,
        compute_signal_ordinal: compute,
        bank_reuse_after_compute_ordinal: index
            .checked_sub(S14_POSITION0_ROLLING_BANKS)
            .map(|prior| prior as u64 + 2),
        host_wait_allowed: false,
    }
}

fn static_bindings(
    assets: &[Position0Asset],
    source_placements: &[S14Position0AssetPlacement],
    layout: &S14Position0StaticLayerLayout,
) -> Result<Vec<S14Position0WeightBinding>> {
    assets
        .iter()
        .map(|asset| {
            let source = exact_source_placement(asset, source_placements)?;
            let physical = exact_physical_placement(source, &layout.assets)?;
            Ok(S14Position0WeightBinding {
                tensor: source.tensor.clone(),
                kind: source.kind.clone(),
                expert_id: source.expert_id,
                path: source.path.clone(),
                sha256: source.sha256.clone(),
                offset: physical.local_offset,
                bytes: physical.bytes,
                arena: S14Position0WeightArena::StaticLayer(layout.layer),
            })
        })
        .collect()
}

fn routed_bindings(
    assets: &[Position0Asset],
    placements: &[S14Position0AssetPlacement],
    bank: usize,
) -> Result<Vec<S14Position0WeightBinding>> {
    bindings(
        assets,
        placements,
        S14Position0WeightArena::RoutedBank(bank),
    )
}

fn bindings(
    assets: &[Position0Asset],
    placements: &[S14Position0AssetPlacement],
    arena: S14Position0WeightArena,
) -> Result<Vec<S14Position0WeightBinding>> {
    assets
        .iter()
        .map(|asset| {
            let placement = exact_source_placement(asset, placements)?;
            Ok(S14Position0WeightBinding {
                tensor: placement.tensor.clone(),
                kind: placement.kind.clone(),
                expert_id: placement.expert_id,
                path: placement.path.clone(),
                sha256: placement.sha256.clone(),
                offset: placement.offset,
                bytes: placement.bytes,
                arena,
            })
        })
        .collect()
}

fn exact_source_placement<'a>(
    asset: &Position0Asset,
    placements: &'a [S14Position0AssetPlacement],
) -> Result<&'a S14Position0AssetPlacement> {
    let mut matches = placements.iter().filter(|placement| {
        placement.tensor == asset.tensor
            && placement.kind == asset.kind
            && placement.expert_id == asset.expert_id
            && placement.path == asset.path
            && placement.sha256 == asset.sha256
            && placement.bytes == asset.bytes
    });
    let placement = matches
        .next()
        .ok_or_else(|| anyhow!("weight plan 缺少精确资产: {}", asset.tensor))?;
    if matches.next().is_some() {
        bail!("weight plan 资产身份不唯一: {}", asset.tensor);
    }
    Ok(placement)
}

fn exact_physical_placement<'a>(
    source: &S14Position0AssetPlacement,
    placements: &'a [S14Position0PhysicalAssetPlacement],
) -> Result<&'a S14Position0PhysicalAssetPlacement> {
    let mut matches = placements.iter().filter(|placement| {
        placement.tensor == source.tensor
            && placement.source_offset == source.offset
            && placement.bytes == source.bytes
    });
    let placement = matches
        .next()
        .ok_or_else(|| anyhow!("physical static layout 缺少精确资产: {}", source.tensor))?;
    if matches.next().is_some() {
        bail!("physical static 资产身份不唯一: {}", source.tensor);
    }
    Ok(placement)
}

fn validate_binding_class(
    bindings: &[S14Position0WeightBinding],
    assets: &[Position0Asset],
    arena: S14Position0WeightArena,
    static_layout: Option<&S14Position0StaticLayerLayout>,
    source_placements: &[S14Position0AssetPlacement],
) -> Result<()> {
    if bindings.len() != assets.len() {
        bail!("weight binding 数量漂移");
    }
    for (binding, asset) in bindings.iter().zip(assets) {
        let source = exact_source_placement(asset, source_placements)?;
        if binding.tensor != asset.tensor
            || binding.kind != asset.kind
            || binding.expert_id != asset.expert_id
            || binding.path != asset.path
            || binding.sha256 != asset.sha256
            || binding.bytes != asset.bytes
            || binding.bytes == 0
            || binding.offset % 256 != 0
            || binding.arena != arena
        {
            bail!("weight binding 身份/范围漂移: {}", asset.tensor);
        }
        if let Some(layout) = static_layout {
            let physical = exact_physical_placement(source, &layout.assets)?;
            let end = binding
                .offset
                .checked_add(binding.bytes)
                .ok_or_else(|| anyhow!("static binding range overflow: {}", asset.tensor))?;
            if end > layout.requested_bytes || binding.offset != physical.local_offset {
                bail!("static binding local range 漂移: {}", asset.tensor);
            }
        } else if binding.offset != source.offset {
            bail!("routed binding source offset 漂移: {}", asset.tensor);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_inputs() -> (
        Position0WholeTokenManifest,
        S14Position0HybridWeightPlan,
        S14Position0WorkspaceLayout,
    ) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        let manifest = Position0WholeTokenManifest::load(&path).unwrap();
        let weights = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let workspace = S14Position0WorkspaceLayout::build(2 * 1024 * 1024).unwrap();
        (manifest, weights, workspace)
    }

    #[test]
    fn real_manifest_builds_all_43_layers_and_l1_l42_continuation() {
        let (manifest, weights, workspace) = real_inputs();
        let program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        assert_eq!(program.layers.len(), 43);
        assert_eq!(program.remaining_after_l0().len(), 42);
        assert_eq!(program.remaining_after_l0()[0].layer, 1);
        assert_eq!(program.remaining_after_l0()[41].layer, 42);

        let mut ratios = BTreeMap::new();
        let mut routes = BTreeMap::new();
        for (index, layer) in program.layers.iter().enumerate() {
            assert_eq!(layer.layer as usize, index);
            assert_eq!(layer.routed_bank, index % 2);
            assert_eq!(layer.router_weights.len(), 2);
            assert_eq!(layer.shared_weights.len(), 6);
            assert_eq!(layer.routed_weights.len(), 36);
            assert_eq!(
                layer.static_weights.len(),
                match layer.compressor {
                    S14Position0CompressorProgram::None => 21,
                    S14Position0CompressorProgram::Ratio4WithIndexer => 32,
                    S14Position0CompressorProgram::Ratio128 => 25,
                }
            );
            assert_eq!(
                layer.state_writes.len(),
                match layer.compressor {
                    S14Position0CompressorProgram::None => 1,
                    S14Position0CompressorProgram::Ratio4WithIndexer => 5,
                    S14Position0CompressorProgram::Ratio128 => 3,
                }
            );
            *ratios
                .entry(manifest.layers[index].compress_ratio)
                .or_insert(0usize) += 1;
            *routes.entry(layer.route).or_insert(0usize) += 1;
        }
        assert_eq!(ratios, BTreeMap::from([(0, 2), (4, 21), (128, 20)]));
        assert_eq!(
            routes,
            BTreeMap::from([
                (S14Position0RouteProgram::PhysicalTid2Eid, 3),
                (S14Position0RouteProgram::BiasTop6, 40),
            ])
        );
    }

    #[test]
    fn generic_state_access_tracks_window_wrap_and_repeated_compression_boundaries() {
        let ratio0 = S14TokenLayerStateAccess::for_ratio(1, 0).unwrap();
        assert_eq!(ratio0.committed_window_rows, 0..1);
        assert_eq!(ratio0.candidate_window_row, 1);
        assert_eq!((ratio0.window_start_row, ratio0.window_count), (0, 2));
        assert_eq!(ratio0.compressed_count, 0);
        assert_eq!(ratio0.ape_row, None);
        assert_eq!(ratio0.remainder_row, None);

        let ratio4 = S14TokenLayerStateAccess::for_ratio(1, 4).unwrap();
        assert_eq!(ratio4.committed_window_rows, 0..1);
        assert_eq!(ratio4.candidate_window_row, 1);
        assert_eq!(ratio4.compressed_count, 0);
        assert_eq!(ratio4.ape_row, Some(1));
        assert_eq!(ratio4.remainder_row, Some(5));

        let ratio128 = S14TokenLayerStateAccess::for_ratio(1, 128).unwrap();
        assert_eq!(ratio128.ape_row, Some(1));
        assert_eq!(ratio128.remainder_row, Some(1));
        assert_eq!(ratio128.compressed_count, 0);

        let boundary = S14TokenLayerStateAccess::for_ratio(3, 4).unwrap();
        assert_eq!(boundary.committed_window_rows, 0..3);
        assert_eq!(boundary.candidate_window_row, 3);
        assert_eq!(boundary.ape_row, Some(3));
        assert_eq!(boundary.remainder_row, Some(7));
        assert!(boundary.compressed_block_ready);
        assert_eq!(boundary.compressed_count, 1);
        assert_eq!(boundary.compressed_block_index, Some(0));
        assert_eq!(boundary.compressed_rope_position, Some(0));
        assert_eq!(boundary.overlap_rollover_rows, Some((4..8, 0..4)));

        let position4 = S14TokenLayerStateAccess::for_ratio(4, 4).unwrap();
        assert_eq!(position4.candidate_window_row, 4);
        assert_eq!((position4.window_start_row, position4.window_count), (0, 5));
        assert_eq!(
            (position4.ape_row, position4.remainder_row),
            (Some(0), Some(4))
        );
        assert!(!position4.compressed_block_ready);
        assert_eq!(position4.compressed_count, 1);

        let position7 = S14TokenLayerStateAccess::for_ratio(7, 4).unwrap();
        assert_eq!(position7.compressed_count, 2);
        assert_eq!(position7.compressed_block_index, Some(1));
        assert_eq!(position7.compressed_rope_position, Some(4));
        assert_eq!(position7.overlap_rollover_rows, Some((4..8, 0..4)));

        let position127 = S14TokenLayerStateAccess::for_ratio(127, 128).unwrap();
        assert_eq!(position127.candidate_window_row, 127);
        assert_eq!(
            (position127.window_start_row, position127.window_count),
            (0, 128)
        );
        assert_eq!(position127.compressed_count, 1);
        assert_eq!(position127.compressed_block_index, Some(0));
        assert_eq!(position127.compressed_rope_position, Some(0));

        let position128 = S14TokenLayerStateAccess::for_ratio(128, 128).unwrap();
        assert_eq!(position128.candidate_window_row, 0);
        assert_eq!(
            (position128.window_start_row, position128.window_count),
            (1, 128)
        );
        assert_eq!(position128.compressed_count, 1);
        assert!(!position128.compressed_block_ready);
        assert_eq!(position128.compressed_block_index, None);

        assert!(S14TokenLayerStateAccess::for_ratio(u32::MAX, 4).is_err());
    }

    #[test]
    fn real_bindings_are_exact_static_local_or_routed_subranges() {
        let (manifest, weights, workspace) = real_inputs();
        let program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        for layer in &program.layers {
            for class in [
                S14Position0WeightClass::Static,
                S14Position0WeightClass::Router,
                S14Position0WeightClass::Shared,
            ] {
                assert!(layer
                    .weights(class)
                    .iter()
                    .all(|binding| binding.arena
                        == S14Position0WeightArena::StaticLayer(layer.layer)));
            }
            assert!(layer.routed_weights.iter().all(|binding| {
                binding.arena == S14Position0WeightArena::RoutedBank(layer.routed_bank)
            }));
            for &expert in &layer.routed_expert_ids {
                let expert_bindings = layer
                    .routed_weights
                    .iter()
                    .filter(|binding| binding.expert_id == Some(expert))
                    .collect::<Vec<_>>();
                assert_eq!(expert_bindings.len(), 6);
                for suffix in [
                    "w1.scale",
                    "w1.weight",
                    "w2.scale",
                    "w2.weight",
                    "w3.scale",
                    "w3.weight",
                ] {
                    assert_eq!(
                        expert_bindings
                            .iter()
                            .filter(|binding| binding.tensor.ends_with(suffix))
                            .count(),
                        1
                    );
                }
            }
        }
    }

    #[test]
    fn hidden_and_timeline_form_one_unbroken_candidate() {
        let (manifest, weights, workspace) = real_inputs();
        let program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        for (index, layer) in program.layers.iter().enumerate() {
            assert_eq!(layer.timeline.transfer_signal_ordinal, index as u64 + 1);
            assert_eq!(
                layer.timeline.compute_wait_transfer_ordinal,
                index as u64 + 1
            );
            assert_eq!(layer.timeline.compute_signal_ordinal, index as u64 + 2);
            assert!(!layer.timeline.host_wait_allowed);
            assert_eq!(
                layer.timeline.bank_reuse_after_compute_ordinal,
                index.checked_sub(2).map(|prior| prior as u64 + 2)
            );
            if index > 0 {
                assert_eq!(program.layers[index - 1].hidden.output, layer.hidden.input);
            }
        }
        assert_eq!(
            program.layers[0].hidden.input_slot,
            S14Position0WorkspaceSlot::HiddenStreamsA
        );
        assert_eq!(
            program.layers[42].hidden.output_slot,
            S14Position0WorkspaceSlot::HiddenStreamsB
        );
        assert_eq!(program.terminal_host_waits, 1);
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Event {
        Weight(u8, S14Position0WeightClass),
        Workspace(u8),
        State(u8),
        Record(u8),
    }

    #[derive(Default)]
    struct Probe {
        events: Vec<Event>,
    }

    impl S14Position0LayerProgramRecorder for Probe {
        fn bind_weights(
            &mut self,
            layer: u8,
            class: S14Position0WeightClass,
            bindings: &[S14Position0WeightBinding],
        ) -> Result<()> {
            if bindings.is_empty() {
                bail!("empty binding class");
            }
            self.events.push(Event::Weight(layer, class));
            Ok(())
        }

        fn bind_workspace(&mut self, layer: u8, _: S14Position0HiddenBinding) -> Result<()> {
            self.events.push(Event::Workspace(layer));
            Ok(())
        }

        fn bind_state_writes(
            &mut self,
            layer: u8,
            writes: &[S14Position0StateWriteKind],
        ) -> Result<()> {
            if writes.is_empty() {
                bail!("empty state writes");
            }
            self.events.push(Event::State(layer));
            Ok(())
        }

        fn record_layer(
            &mut self,
            layer: u8,
            primitives: &[S14Position0LayerPrimitive],
            _: S14Position0RouteProgram,
            _: S14Position0CompressorProgram,
            timeline: S14Position0LayerTimelineProgram,
        ) -> Result<()> {
            if primitives.is_empty() || timeline.host_wait_allowed {
                bail!("invalid record contract");
            }
            self.events.push(Event::Record(layer));
            Ok(())
        }
    }

    #[test]
    fn recorder_visits_every_real_layer_once_in_strict_order() {
        let (manifest, weights, workspace) = real_inputs();
        let program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        let mut probe = Probe::default();
        program.record_full_depth(&mut probe).unwrap();
        assert_eq!(probe.events.len(), 43 * 7);
        for (index, events) in probe.events.chunks_exact(7).enumerate() {
            let layer = index as u8;
            assert_eq!(
                events,
                [
                    Event::Weight(layer, S14Position0WeightClass::Static),
                    Event::Weight(layer, S14Position0WeightClass::Router),
                    Event::Weight(layer, S14Position0WeightClass::Shared),
                    Event::Weight(layer, S14Position0WeightClass::Routed),
                    Event::Workspace(layer),
                    Event::State(layer),
                    Event::Record(layer),
                ]
            );
        }
    }

    #[test]
    fn generated_program_drift_fails_closed() {
        let (manifest, weights, workspace) = real_inputs();
        let mut program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        program.layers[1].routed_bank = 0;
        assert!(program.validate(&manifest, &weights, &workspace).is_err());

        let mut program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        program.layers.remove(17);
        assert!(program.validate(&manifest, &weights, &workspace).is_err());

        let mut program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        program.layers[28].state_writes.pop();
        assert!(program.validate(&manifest, &weights, &workspace).is_err());

        let mut program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        program.layers[3].static_weights[0].offset += 256;
        assert!(program.validate(&manifest, &weights, &workspace).is_err());

        let mut program =
            S14Position0FullDepthLayerProgram::build(&manifest, &weights, &workspace).unwrap();
        program.layers[42].routed_weights[0].offset += 256;
        assert!(program.validate(&manifest, &weights, &workspace).is_err());
    }
}
