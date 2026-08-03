//! FullDepth43 position0 真实后端的 L0 首段。
//!
//! 这个模块刻意不宣称已经闭合 43 层或 final head。它先把最容易
//! 被 fixture 污染的 L0 边界冻结为可审计合同：
//!
//! - embedding 只接受 manifest 中经 SHA 验证的 BOS 权重行；
//! - L0 route 顺序必须与真实 `tid2eid[BOS]` 物理行和 routed 权重页同时闭合；
//! - routed/shared MoE 的六元素 offset 只从当前权重 plan 派生，不从 capture
//!   hidden 或历史输出派生；
//! - 未闭合 L1..L42、final head 和 post-fence receipts 前，capabilities 保持
//!   fail-closed，因而不会被 `run_bos_position0` 误当成可交付后端。

use crate::{
    compute::{DescriptorBinder, StorageBufferSlice},
    s14_bf16_q_head_normalize::S14Bf16QHeadNormalizePipeline,
    s14_bf16_rmsnorm::S14Bf16RmsNormPipeline,
    s14_bf16_rmsnorm::S14Bf16RmsNormShape,
    s14_bf16_to_f32::{S14Bf16ToF32Pipeline, S14Bf16ToF32Shape},
    s14_dual_queue_timeline::S14DualQueueTimeline,
    s14_dynamic_routed_packing::S14DynamicRoutedUploadPlan,
    s14_dynamic_routed_page_plan::{
        DynamicRoutedPagePlan, MaterializedDynamicRoutedPagePlan, OnlineTop6,
        DYNAMIC_ROUTED_RANGE_COUNT,
    },
    s14_e4m3_qdq::{S14E4m3QdqPipeline, S14E4m3QdqShape},
    s14_embedding_broadcast::{S14EmbeddingBroadcastPipeline, S14EmbeddingBroadcastShape},
    s14_f32_to_bf16::{S14F32ToBf16Pipeline, S14F32ToBf16Shape},
    s14_hc_post::{S14HcPostPipeline, S14HcPostShape},
    s14_input_asset_plan::{S14InputAssetPlan, S14PositionExecutionPlan},
    s14_position0_attention::S14Position0AttentionPipeline,
    s14_position0_hybrid_upload::{
        S14Position0HeadChunkReceipt, S14Position0HybridUploader, S14Position0LayerCopyReceipt,
        S14Position0RoutedUploadReceipt, S14Position0StaticLayerUploadReceipt,
    },
    s14_position0_hybrid_weight_arena::{
        S14Position0PhysicalAssetPlacement, S14Position0StaticLayerLayout,
    },
    s14_position0_layer_program::S14Position0FullDepthLayerProgram,
    s14_position0_mapped_assets::{VerifiedMappedAssetStats, VerifiedMappedAssetStore},
    s14_position0_paged_weight_arena::{
        S14Position0PagedWeightArena, S14Position0StaticLayerBinding,
    },
    s14_position0_state_writeback::{
        S14Position0ApeAddPipeline, S14Position0FullDepthStateRecordingProgram,
        S14Position0LayerStateRecordingRecipe,
    },
    s14_position0_synchronous_layer_pager::{
        S14Position0LayerUploadReceipt, S14Position0SynchronousLayerBackend,
        S14Position0SynchronousLayerPlan,
    },
    s14_position0_weight_plan::{S14Position0HybridWeightPlan, S14Position0LayerWeightPlan},
    s14_position0_whole_token::{
        Position0BackendCapabilities, Position0BackendCompletion, Position0BackendError,
        Position0GpuBootstrap, Position0GpuCandidate, Position0GraphReceipt, Position0LayerBackend,
    },
    s14_position0_workspace::{S14Position0WorkspaceLayout, S14Position0WorkspaceSlot},
    s14_position1_attention::{position_rope_cos_sin, S14Position1AttentionPipeline},
    s14_position3_attention::S14Position3AttentionPipeline,
    s14_ratio128_compressor_finalize::{
        S14Ratio128CompressorBoundary, S14Ratio128CompressorFinalizePipelines,
        S14_RATIO128_RMS_EPSILON,
    },
    s14_ratio4_compressor_finalize::{
        S14Ratio4CompressorBoundary, S14Ratio4CompressorFinalizePipelines, S14Ratio4CompressorKind,
    },
    s14_ratio4_global_topk::{S14Ratio4GlobalTopKPipeline, S14Ratio4PagedGlobalTopKBindings},
    s14_ratio4_history_paging::{S14Ratio4HistoryLayout, S14Ratio4HistoryPublishPlan},
    s14_ratio4_main_page_gather::{
        build_ratio4_main_page_table, S14Ratio4MainGatherShape, S14Ratio4MainPageGatherPipeline,
        S14Ratio4MaterializedMainPage, S14_RATIO4_GATHER_ROW_WORDS,
    },
    s14_route_postprocess_gpu::{
        S14RouteBufferSlice, S14RoutePostprocessGpuBindings, S14RoutePostprocessGpuMode,
        S14RoutePostprocessGpuPipeline, S14_ROUTE_GPU_LOGITS_BYTES,
        S14_ROUTE_GPU_PHYSICAL_IDS_BYTES, S14_ROUTE_GPU_STATUS_BYTES,
    },
    s14_route_slot_align::{
        S14RouteSlotAlignBindings, S14RouteSlotAlignPipeline, S14RouteSlotAlignSlice,
    },
    s14_sparse_attention::{
        S14Ratio4IndexQueryShape, S14SparseAttentionPipelines, S14SparseAttentionShape,
        S14_INDEX_HEADS,
    },
    s14_vulkan::{
        s14_batched_official_prepare_buffer_bytes, s14_exact_order_block_reduce_buffer_bytes,
        S14Bf16MatvecShape, S14F32MatvecShape, S14GroupedMatvecShape, S14HcPreShape,
        S14MatvecShape, S14NumericPipelines, S14RaggedBranchOffsets, S14RaggedMatvecShape,
        S14RaggedProjection,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    DType, GraphProfile, NativeState, Position0Asset, Position0Final, Position0Layer,
    Position0WholeTokenManifest, EXPERTS_PER_TOKEN, FULL_DEPTH_LAYERS, N_ROUTED_EXPERTS,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

pub const S14_POSITION0_L0: u8 = 0;
pub const S14_POSITION0_L0_INPUT_TOKEN: u32 = 0;
pub const S14_POSITION0_L0_WORKSPACE_BYTES: u64 = 512 * 1024 * 1024;

/// Production dynamic routing splits one layer at the only host-visible data
/// dependency: online top-6. The probe leaves every attention/router workspace
/// value device-resident; the continuation may only be recorded after the
/// matching 36 physical ranges have been proof-checked and materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum S14LayerCommandSegment {
    CompleteManifestReplay,
    RouterProbe,
    DynamicMoeContinuation,
}

impl S14LayerCommandSegment {
    const fn records_prefix(self) -> bool {
        !matches!(self, Self::DynamicMoeContinuation)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct S14Position0RouterProbeReceipt {
    pub route: OnlineTop6,
    /// Only IDs and weights cross the device/host boundary. Attention/HC/KV
    /// intermediates remain in the shared device workspace for continuation.
    pub readback_bytes: u64,
}

/// 已经在数值门中单独验证、L0 必须严格串行的 GPU 阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0L0Stage {
    EmbeddingBroadcast,
    AttentionHcPre,
    QueryAndKeyValue,
    Position0Attention,
    AttentionOutputProjection,
    AttentionHcPost,
    FfnHcPre,
    RouterProjection,
    PhysicalRoutePostprocess,
    RoutedTop6Moe,
    SharedMoe,
    ExactOrderReduce,
    FfnHcPost,
    WindowKvWriteback,
}

pub const S14_POSITION0_L0_STAGES: [S14Position0L0Stage; 14] = [
    S14Position0L0Stage::EmbeddingBroadcast,
    S14Position0L0Stage::AttentionHcPre,
    S14Position0L0Stage::QueryAndKeyValue,
    S14Position0L0Stage::Position0Attention,
    S14Position0L0Stage::AttentionOutputProjection,
    S14Position0L0Stage::AttentionHcPost,
    S14Position0L0Stage::FfnHcPre,
    S14Position0L0Stage::RouterProjection,
    S14Position0L0Stage::PhysicalRoutePostprocess,
    S14Position0L0Stage::RoutedTop6Moe,
    S14Position0L0Stage::SharedMoe,
    S14Position0L0Stage::ExactOrderReduce,
    S14Position0L0Stage::FfnHcPost,
    S14Position0L0Stage::WindowKvWriteback,
];

/// 当前原语 API 与单 arena workspace 之间尚缺的 offset 绑定。
///
/// 这是编译阶段的硬阻塞清单，不是数值降级许可。主代理补齐后，
/// concrete recorder 才能不新建几十个 Vulkan allocation 而直接绑定 workspace。
pub const S14_POSITION0_L0_REQUIRED_OFFSET_APIS: [&str; 0] = [];

const HIDDEN: u32 = 4096;
const INTERMEDIATE: u32 = 2048;
const HC_FLAT: u32 = 4 * HIDDEN;
const TOP6: u32 = EXPERTS_PER_TOKEN as u32;
const NORM_EPS: f32 = 1.0e-6;
const POSITION1_ROPE_ROW_BYTES: u64 = 32 * 2 * 4;
const POSITION1_ROPE_RATIO0_OFFSET: u64 = 0;
const POSITION1_ROPE_YARN_OFFSET: u64 = POSITION1_ROPE_ROW_BYTES;
const POSITION1_ROPE_BUFFER_BYTES: u64 = POSITION1_ROPE_ROW_BYTES * 2;
const RATIO4_COMPRESSED_ROPE_BYTES: u64 = POSITION1_ROPE_ROW_BYTES;
const RATIO128_COMPRESSED_ROPE_BYTES: u64 = POSITION1_ROPE_ROW_BYTES;
const AUX_EMBEDDING_OFFSET: u64 = 0;
const AUX_METADATA_OFFSET: u64 = 8_192;
const AUX_PHYSICAL_IDS_OFFSET: u64 = 8_448;
const AUX_Q_RMS_ONES_OFFSET: u64 = 8_704;
const AUX_SHARED_ROUTE_ONE_OFFSET: u64 = 9_728;
const AUX_NUMERIC_ROUTE_OVERRIDE_OFFSET: u64 = 9_984;
const AUX_LAYER_INPUT_OFFSET: u64 = 10_240;
const AUX_LAYER_INPUT_BYTES: u64 = 4 * 4096 * 2;
const AUX_HC_OVERRIDE_OFFSET: u64 = AUX_LAYER_INPUT_OFFSET + AUX_LAYER_INPUT_BYTES;
const AUX_HC_OVERRIDE_BYTES: u64 = 20 * 4;
const AUX_LOGICAL_BYTES: u64 = 43_264;
const READBACK_HIDDEN_OFFSET: u64 = 0;
const READBACK_ROUTE_IDS_OFFSET: u64 = 32_768;
const READBACK_ROUTE_WEIGHTS_OFFSET: u64 = 33_024;
const READBACK_KV_OFFSET: u64 = 33_280;
const READBACK_CANDIDATE_KV_OFFSET: u64 = 34_304;
const READBACK_MOE_OFFSET: u64 = 35_328;
const READBACK_FFN_INPUT_OFFSET: u64 = 51_712;
const READBACK_ATTENTION_INPUT_OFFSET: u64 = 68_096;
const READBACK_ATTENTION_BRANCH_OFFSET: u64 = 84_480;
const READBACK_POST_ATTENTION_OFFSET: u64 = 92_672;
const READBACK_QUERY_FINAL_OFFSET: u64 = 125_440;
const READBACK_KEY_VALUE_FINAL_OFFSET: u64 = 190_976;
const READBACK_ATTENTION_OUTPUT_OFFSET: u64 = 192_000;
const READBACK_WO_A_QDQ_OFFSET: u64 = 257_536;
const READBACK_HC_POST_OFFSET: u64 = 290_304;
const READBACK_HC_COMB_OFFSET: u64 = 290_320;
const READBACK_LOGICAL_BYTES: u64 = 290_560;

const ROUTED_SUFFIXES: [&str; 6] = [
    "w1.weight",
    "w1.scale",
    "w3.weight",
    "w3.scale",
    "w2.weight",
    "w2.scale",
];

/// L0 计算图对当前物理权重页的唯一视图。
///
/// `static_offsets` 是 L0 static buffer 内 offset；`routed_metadata` 是双页
/// routed bank 内 offset。两者都不允许从 capture manifest 反推。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0L0GraphPlan {
    pub layer: u8,
    pub layer_index: usize,
    pub route_mode: S14RoutePostprocessGpuMode,
    pub static_logical_bytes: u64,
    pub routed_logical_bytes: u64,
    pub static_offsets: BTreeMap<String, u64>,
    pub routed_metadata: [S14RaggedBranchOffsets; EXPERTS_PER_TOKEN],
    pub routed_expert_ids: [u16; EXPERTS_PER_TOKEN],
    pub routed_route_weight_bits: [u32; EXPERTS_PER_TOKEN],
    pub workspace: S14Position0WorkspaceLayout,
    pub stages: [S14Position0L0Stage; 14],
}

impl S14Position0L0GraphPlan {
    pub fn build(
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        static_layout: &S14Position0StaticLayerLayout,
    ) -> Result<Self> {
        Self::build_layer(manifest, weights, static_layout, 0)
    }

    /// 用同一组已验证的 Vulkan 原语构建任意 position0 层。L0 构造器仍是
    /// `build` 的严格包装，后续 FullDepth43 后端按 index 参数化调用本入口。
    pub fn build_layer(
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        static_layout: &S14Position0StaticLayerLayout,
        layer_index: usize,
    ) -> Result<Self> {
        manifest
            .validate()
            .map_err(|error| anyhow!("position0 manifest invalid: {error}"))?;
        weights.validate(manifest)?;
        let layer = manifest
            .layers
            .get(layer_index)
            .ok_or_else(|| anyhow!("position0 manifest 缺少 layer index {layer_index}"))?;
        let routed = weights.routed_layers.get(layer_index).ok_or_else(|| {
            anyhow!("position0 weight plan 缺少 layer index {layer_index} routed 页")
        })?;
        validate_layer_identity(manifest, layer, routed, static_layout, layer_index)?;

        let route_mode = match layer.route_source.as_str() {
            "current_token_tid2eid_physical_i64" if layer.layer < 3 => {
                S14RoutePostprocessGpuMode::PhysicalIds
            }
            "sqrtsoftplus_plus_bias_top6" if layer.layer >= 3 => {
                S14RoutePostprocessGpuMode::BiasTop6
            }
            _ => bail!("L{} route source/mode 漂移", layer.layer),
        };

        let static_offsets = static_layout
            .assets
            .iter()
            .map(|asset| (asset.tensor.clone(), asset.local_offset))
            .collect::<BTreeMap<_, _>>();
        let routed_expert_ids: [u16; EXPERTS_PER_TOKEN] = layer
            .expert_ids
            .clone()
            .try_into()
            .map_err(|_| anyhow!("L0 route 不是严格 top-{EXPERTS_PER_TOKEN}"))?;
        let routed_metadata: [S14RaggedBranchOffsets; EXPERTS_PER_TOKEN] = routed_expert_ids
            .iter()
            .map(|&expert| routed_offsets_for_expert(routed, layer.layer, expert))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow!("L0 routed metadata 不是严格 top-6"))?;
        let routed_route_weight_bits: [u32; EXPERTS_PER_TOKEN] = layer
            .route_weights
            .iter()
            .map(|weight| f32::to_bits(*weight))
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| anyhow!("L0 route weights 不是严格 top-6"))?;
        let workspace = S14Position0WorkspaceLayout::build(S14_POSITION0_L0_WORKSPACE_BYTES)?;
        validate_l0_gpu_shape_contracts(routed.used_bytes, &routed_metadata, &workspace)?;
        Ok(Self {
            layer: layer.layer,
            layer_index,
            route_mode,
            static_logical_bytes: static_layout.requested_bytes,
            routed_logical_bytes: routed.used_bytes,
            static_offsets,
            routed_metadata,
            routed_expert_ids,
            routed_route_weight_bits,
            workspace,
            stages: S14_POSITION0_L0_STAGES,
        })
    }

    pub fn static_offset(&self, tensor: &str) -> Result<u64> {
        self.static_offsets
            .get(tensor)
            .copied()
            .ok_or_else(|| anyhow!("L0 static 页缺少 tensor: {tensor}"))
    }

    pub fn static_offset_suffix(&self, suffix: &str) -> Result<u64> {
        self.static_offset(&format!("layers.{}.{}", self.layer, suffix))
    }

    /// 从经 SHA 验证的真实 `tid2eid` tensor 解码当前 input token 的物理行。
    ///
    /// 本入口只闭合 token→row、shape、范围与唯一性；online dynamic routed
    /// continuation 会以返回的专家身份重新规划并物化36个真实Range，不能在这里
    /// 与冻结position0 manifest的专家页比较。
    pub fn decode_tid2eid_row_for_token(
        &self,
        asset: &Position0Asset,
        payload: &[u8],
        input_token_id: u32,
    ) -> Result<[u32; EXPERTS_PER_TOKEN]> {
        if self.route_mode != S14RoutePostprocessGpuMode::PhysicalIds
            || asset.tensor != format!("layers.{}.ffn.gate.tid2eid", self.layer)
            || asset.dtype != "I64"
            || asset.shape.as_slice() != [129_280, EXPERTS_PER_TOKEN as u64]
            || asset.bytes != 129_280 * EXPERTS_PER_TOKEN as u64 * 8
            || payload.len() as u64 != asset.bytes
        {
            bail!("L{} tid2eid asset shape/dtype/bytes 漂移", self.layer);
        }
        let row_bytes = EXPERTS_PER_TOKEN
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or_else(|| anyhow!("L{} tid2eid row bytes overflow", self.layer))?;
        let row_start = usize::try_from(input_token_id)
            .ok()
            .and_then(|token| token.checked_mul(row_bytes))
            .ok_or_else(|| anyhow!("L{} tid2eid token row offset overflow", self.layer))?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or_else(|| anyhow!("L{} tid2eid token row end overflow", self.layer))?;
        let row = payload.get(row_start..row_end).ok_or_else(|| {
            anyhow!(
                "L{} tid2eid token row 越界: token={input_token_id}",
                self.layer
            )
        })?;
        let mut output = [0u32; EXPERTS_PER_TOKEN];
        let mut seen = BTreeSet::new();
        for (slot, bytes) in row.chunks_exact(8).enumerate() {
            let expert = i64::from_le_bytes(bytes.try_into().expect("8-byte chunk"));
            if !(0..N_ROUTED_EXPERTS as i64).contains(&expert) {
                bail!(
                    "L{} tid2eid token expert 越界: token={input_token_id} slot={slot} expert={expert}",
                    self.layer
                );
            }
            let expert = expert as u16;
            if !seen.insert(expert) {
                bail!(
                    "L{} tid2eid token expert 重复: token={input_token_id} expert={expert}",
                    self.layer
                );
            }
            output[slot] = u32::from(expert);
        }
        Ok(output)
    }

    /// 历史BOS/manifest replay门仍要求row0与冻结专家页逐slot一致。
    pub fn decode_and_validate_tid2eid_bos(
        &self,
        asset: &Position0Asset,
        payload: &[u8],
    ) -> Result<[u32; EXPERTS_PER_TOKEN]> {
        let output =
            self.decode_tid2eid_row_for_token(asset, payload, S14_POSITION0_L0_INPUT_TOKEN)?;
        for (slot, (&expert, &manifest_expert)) in
            output.iter().zip(self.routed_expert_ids.iter()).enumerate()
        {
            if expert != u32::from(manifest_expert) {
                bail!(
                    "L{} routed 权重页与真实 tid2eid BOS漂移: slot={slot} page={manifest_expert} tid2eid={expert}",
                    self.layer
                );
            }
        }
        Ok(output)
    }
}

fn validate_l0_gpu_shape_contracts(
    routed_bytes: u64,
    routed_metadata: &[S14RaggedBranchOffsets; EXPERTS_PER_TOKEN],
    workspace: &S14Position0WorkspaceLayout,
) -> Result<()> {
    const HIDDEN: u32 = 4096;
    const HC_FLAT: u32 = 4 * HIDDEN;
    const INTERMEDIATE: u32 = 2048;
    const TOP6: u32 = EXPERTS_PER_TOKEN as u32;

    if S14RoutePostprocessGpuMode::PhysicalIds.aux_bytes() != S14_ROUTE_GPU_PHYSICAL_IDS_BYTES {
        bail!("L0 physical route GPU ABI 漂移");
    }

    S14HcPreShape::new(HIDDEN)?;
    S14F32MatvecShape::new(24, HC_FLAT, 1)?;
    S14Bf16RmsNormShape::new(1, 1024)?;
    S14Bf16RmsNormShape::new(1, 512)?;
    S14F32ToBf16Shape::new(1024)?;
    S14F32ToBf16Shape::new(32_768)?;
    S14Bf16ToF32Shape::new(32_768)?;
    S14E4m3QdqShape::new(1, 448, 64)?;
    S14E4m3QdqShape::new(1, 8192, 128)?;
    S14GroupedMatvecShape::new(8, 1024, HIDDEN)?.validate_fp8_bf16_weight()?;
    S14Bf16MatvecShape::new(256, HIDDEN, 1)?;
    S14HcPostShape::new(HIDDEN)?;

    let routed_shape = |n, k, projection| {
        S14RaggedMatvecShape::new(
            TOP6,
            if projection == S14RaggedProjection::W2 {
                1
            } else {
                TOP6
            },
            n,
            k,
            projection,
        )
    };
    routed_shape(INTERMEDIATE, HIDDEN, S14RaggedProjection::W1)?
        .validate_mxfp4(routed_bytes, routed_metadata)?;
    routed_shape(INTERMEDIATE, HIDDEN, S14RaggedProjection::W3)?
        .validate_mxfp4(routed_bytes, routed_metadata)?;
    routed_shape(HIDDEN, INTERMEDIATE, S14RaggedProjection::W2)?
        .validate_mxfp4(routed_bytes, routed_metadata)?;

    S14MatvecShape::new(INTERMEDIATE, HIDDEN)?.validate_fp8()?;
    S14MatvecShape::new(HIDDEN, INTERMEDIATE)?.validate_fp8()?;
    let (expert_matrix_bytes, route_weight_bytes) =
        s14_batched_official_prepare_buffer_bytes(TOP6, INTERMEDIATE)?;
    let (routed_down_bytes, reduced_bytes) = s14_exact_order_block_reduce_buffer_bytes(1)?;
    let requirements = [
        (
            S14Position0WorkspaceSlot::HiddenStreamsA,
            4 * HIDDEN as u64 * 2,
        ),
        (
            S14Position0WorkspaceSlot::RouterLogitsF32,
            S14_ROUTE_GPU_LOGITS_BYTES,
        ),
        (
            S14Position0WorkspaceSlot::RouterIdsU32,
            S14_ROUTE_GPU_PHYSICAL_IDS_BYTES,
        ),
        (
            S14Position0WorkspaceSlot::RouterWeightsF32,
            route_weight_bytes,
        ),
        (
            S14Position0WorkspaceSlot::ExpertHiddenF32,
            expert_matrix_bytes,
        ),
        (S14Position0WorkspaceSlot::ExpertDownF32, routed_down_bytes),
        (S14Position0WorkspaceSlot::MoeAccumulatorF32, reduced_bytes),
        (
            S14Position0WorkspaceSlot::HeadArgmax,
            S14_ROUTE_GPU_STATUS_BYTES,
        ),
    ];
    for (slot, required) in requirements {
        let region = workspace.region(slot);
        if region.logical_bytes < required {
            bail!(
                "L0 workspace slot 过小: slot={slot:?} actual={} required={required}",
                region.logical_bytes
            );
        }
    }
    Ok(())
}

fn validate_layer_identity(
    manifest: &Position0WholeTokenManifest,
    layer: &Position0Layer,
    routed: &S14Position0LayerWeightPlan,
    static_layout: &S14Position0StaticLayerLayout,
    layer_index: usize,
) -> Result<()> {
    if manifest.position != 0
        || manifest.input_token_id != S14_POSITION0_L0_INPUT_TOKEN
        || FULL_DEPTH_LAYERS.get(layer_index).copied() != Some(layer.layer)
        || routed.layer != layer.layer
        || routed.bank != layer_index % 2
        || static_layout.layer != layer.layer
    {
        bail!("L{} manifest/weight/static identity 漂移", layer.layer);
    }
    let route_valid = if layer.layer < 3 {
        layer.route_source == "current_token_tid2eid_physical_i64"
    } else {
        layer.route_source == "sqrtsoftplus_plus_bias_top6"
    };
    if !route_valid {
        bail!("L{} route source 漂移", layer.layer);
    }
    if layer.expert_ids.len() != EXPERTS_PER_TOKEN
        || layer.assets.routed.len() != EXPERTS_PER_TOKEN * ROUTED_SUFFIXES.len()
    {
        bail!("L{} routed 页不是完整 top-6 三投影", layer.layer);
    }

    let manifest_static = layer
        .assets
        .non_expert
        .iter()
        .chain(&layer.assets.router)
        .chain(&layer.assets.shared)
        .map(|asset| asset.tensor.as_str())
        .collect::<BTreeSet<_>>();
    let physical_static = static_layout
        .assets
        .iter()
        .map(|asset| asset.tensor.as_str())
        .collect::<BTreeSet<_>>();
    if manifest_static != physical_static {
        bail!("L{} static 物理页与 manifest tensor 集不一致", layer.layer);
    }
    for asset in &static_layout.assets {
        validate_physical_asset(asset, static_layout.requested_bytes)?;
    }
    Ok(())
}

fn validate_physical_asset(
    asset: &S14Position0PhysicalAssetPlacement,
    logical_bytes: u64,
) -> Result<()> {
    let end = asset
        .local_offset
        .checked_add(asset.bytes)
        .ok_or_else(|| anyhow!("physical asset range overflow: {}", asset.tensor))?;
    if asset.bytes == 0 || end > logical_bytes || asset.local_offset % 256 != 0 {
        bail!("physical asset placement 非法: {}", asset.tensor);
    }
    Ok(())
}

fn routed_offsets_for_expert(
    routed: &S14Position0LayerWeightPlan,
    layer: u8,
    expert: u16,
) -> Result<S14RaggedBranchOffsets> {
    let prefix = format!("layers.{layer}.ffn.experts.{expert}.");
    let mut offsets = [None; ROUTED_SUFFIXES.len()];
    for placement in &routed.assets {
        let Some(suffix) = placement.tensor.strip_prefix(&prefix) else {
            continue;
        };
        if let Some(index) = ROUTED_SUFFIXES
            .iter()
            .position(|expected| *expected == suffix)
        {
            offsets[index] = Some(
                u32::try_from(placement.offset)
                    .with_context(|| format!("L0/E{expert} {suffix} offset 超出 u32"))?,
            );
        }
    }
    let [w1, s1, w3, s3, w2, s2] = offsets
        .map(|value| value.ok_or_else(|| anyhow!("L0/E{expert} routed 页缺少三投影权重/缩放")));
    Ok(S14RaggedBranchOffsets {
        w1: w1?,
        s1: s1?,
        w3: w3?,
        s3: s3?,
        w2: w2?,
        s2: s2?,
    })
}

/// L0 唯一的 Vulkan owner。所有 descriptor 都至少存活到对应 compute timeline
/// 完成；析构时先 drain device，避免错误路径释放仍被 command 引用的资源。
struct S14Position0L0GpuOwner<'ctx> {
    ctx: &'ctx VulkanContext,
    static_arena: Option<GpuBuffer>,
    routed_arena: Option<GpuBuffer>,
    workspace: Option<GpuBuffer>,
    external_static_arena: Option<&'ctx GpuBuffer>,
    external_routed_arena: Option<&'ctx GpuBuffer>,
    external_workspace: Option<&'ctx GpuBuffer>,
    immutable: Option<GpuBuffer>,
    paged_immutables: Vec<GpuBuffer>,
    active_paged_immutable: Option<usize>,
    readback: Option<GpuBuffer>,
    embedding_pipeline: Option<S14EmbeddingBroadcastPipeline>,
    numeric: Option<S14NumericPipelines>,
    numeric_exact: Option<S14NumericPipelines>,
    sparse_index_query_numeric_exact: Option<S14NumericPipelines>,
    rmsnorm: Option<S14Bf16RmsNormPipeline>,
    q_head_normalize: Option<S14Bf16QHeadNormalizePipeline>,
    f32_to_bf16: Option<S14F32ToBf16Pipeline>,
    bf16_to_f32: Option<S14Bf16ToF32Pipeline>,
    qdq: Option<S14E4m3QdqPipeline>,
    attention: Option<S14Position0AttentionPipeline>,
    position1_attention: Option<S14Position1AttentionPipeline>,
    position3_attention: Option<S14Position3AttentionPipeline>,
    sparse_attention: Option<S14SparseAttentionPipelines>,
    position1_rope: Option<GpuBuffer>,
    ratio4_compressed_rope: Option<GpuBuffer>,
    ratio4_finalize: Option<S14Ratio4CompressorFinalizePipelines>,
    ratio4_global_topk: Option<S14Ratio4GlobalTopKPipeline>,
    ratio4_main_gather: Option<S14Ratio4MainPageGatherPipeline>,
    ratio128_compressed_rope: Option<GpuBuffer>,
    ratio128_finalize: Option<S14Ratio128CompressorFinalizePipelines>,
    hc_post: Option<S14HcPostPipeline>,
    ape_add: Option<S14Position0ApeAddPipeline>,
    route: Option<S14RoutePostprocessGpuPipeline>,
    route_slot_align: Option<S14RouteSlotAlignPipeline>,
    timeline: Option<S14DualQueueTimeline>,
    command_pool: vk::CommandPool,
    layer_command: vk::CommandBuffer,
    owns_command_pool: bool,
    binders: Vec<DescriptorBinder>,
    verified: VerifiedMappedAssetStats,
    last_compute_value: u64,
    layer_kv_offset: Option<u64>,
    numeric_route_override: bool,
    reference_route_replay: bool,
    numeric_layer_input: bool,
    numeric_hc_override: bool,
    external_timeline_drained: bool,
}

impl<'ctx> S14Position0L0GpuOwner<'ctx> {
    fn new_external(
        ctx: &'ctx VulkanContext,
        manifest: &Position0WholeTokenManifest,
        graph: &S14Position0L0GraphPlan,
        store: &mut VerifiedMappedAssetStore,
        static_arena: &'ctx GpuBuffer,
        routed_arena: &'ctx GpuBuffer,
        workspace: &'ctx GpuBuffer,
        command_pool: vk::CommandPool,
        layer_command: vk::CommandBuffer,
    ) -> Result<Self> {
        if static_arena.size() < graph.static_logical_bytes
            || routed_arena.size() < graph.routed_logical_bytes
            || workspace.size() < S14_POSITION0_L0_WORKSPACE_BYTES
        {
            bail!("L{} external owner buffer 容量不足", graph.layer);
        }
        if command_pool == vk::CommandPool::null() || layer_command == vk::CommandBuffer::null() {
            bail!(
                "L{} external owner persistent command handle为空",
                graph.layer
            );
        }
        let layer = manifest
            .layers
            .get(graph.layer_index)
            .ok_or_else(|| anyhow!("position0 manifest 缺少 L{}", graph.layer))?;
        let embedding = store.map_verified_batch(std::slice::from_ref(&manifest.embedding_row))?;
        let physical_ids = if graph.route_mode == S14RoutePostprocessGpuMode::PhysicalIds {
            let name = format!("layers.{}.ffn.gate.tid2eid", graph.layer);
            let asset = layer
                .assets
                .router
                .iter()
                .find(|asset| asset.tensor == name)
                .ok_or_else(|| anyhow!("L{} manifest 缺少 tid2eid", graph.layer))?;
            let mapped = store.map_verified_batch(std::slice::from_ref(asset))?;
            graph.decode_and_validate_tid2eid_bos(asset, mapped[0].bytes())?
        } else {
            graph.routed_expert_ids.map(u32::from)
        };
        let immutable = host_storage_buffer(ctx, AUX_LOGICAL_BYTES)?;
        unsafe {
            std::ptr::write_bytes(immutable.mapped(), 0, AUX_LOGICAL_BYTES as usize);
            immutable.write_at(AUX_EMBEDDING_OFFSET as usize, embedding[0].bytes());
            let mut metadata = Vec::with_capacity(EXPERTS_PER_TOKEN * 6 * 4);
            for branch in graph.routed_metadata {
                for word in branch.words() {
                    metadata.extend_from_slice(&word.to_le_bytes());
                }
            }
            immutable.write_at(AUX_METADATA_OFFSET as usize, &metadata);
            immutable.write_at(
                AUX_PHYSICAL_IDS_OFFSET as usize,
                bytemuck::cast_slice(&physical_ids),
            );
            immutable.write_at(
                AUX_NUMERIC_ROUTE_OVERRIDE_OFFSET as usize,
                bytemuck::cast_slice(&graph.routed_route_weight_bits),
            );
            immutable.write_at(
                AUX_Q_RMS_ONES_OFFSET as usize,
                bytemuck::cast_slice(&vec![0x3f80u16; 512]),
            );
            immutable.write_at(AUX_SHARED_ROUTE_ONE_OFFSET as usize, &1.0f32.to_le_bytes());
        }
        let readback = GpuBuffer::new(
            ctx,
            READBACK_LOGICAL_BYTES,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        Ok(Self {
            ctx,
            static_arena: None,
            routed_arena: None,
            workspace: None,
            external_static_arena: Some(static_arena),
            external_routed_arena: Some(routed_arena),
            external_workspace: Some(workspace),
            immutable: Some(immutable),
            paged_immutables: Vec::new(),
            active_paged_immutable: None,
            readback: Some(readback),
            embedding_pipeline: Some(S14EmbeddingBroadcastPipeline::new(ctx)?),
            numeric: Some(S14NumericPipelines::new(ctx)?),
            numeric_exact: None,
            // ratio4 index-query 的 K=1024 投影有独立冻结归约顺序；
            // 不能经 `numeric_exact.unwrap_or(numeric)` 静默回落到快速 128-lane 树。
            sparse_index_query_numeric_exact: Some(S14NumericPipelines::new_exact_audit(ctx)?),
            rmsnorm: Some(S14Bf16RmsNormPipeline::new(ctx)?),
            q_head_normalize: Some(S14Bf16QHeadNormalizePipeline::new(ctx)?),
            f32_to_bf16: Some(S14F32ToBf16Pipeline::new(ctx)?),
            bf16_to_f32: Some(S14Bf16ToF32Pipeline::new(ctx)?),
            qdq: Some(S14E4m3QdqPipeline::new(ctx)?),
            attention: Some(S14Position0AttentionPipeline::new(ctx)?),
            position1_attention: Some(S14Position1AttentionPipeline::new(ctx)?),
            position3_attention: Some(S14Position3AttentionPipeline::new(ctx)?),
            sparse_attention: Some(S14SparseAttentionPipelines::new(ctx)?),
            position1_rope: Some(build_position_rope_buffer(ctx, 0)?),
            ratio4_compressed_rope: Some(build_ratio4_compressed_rope_buffer(ctx)?),
            ratio4_finalize: Some(S14Ratio4CompressorFinalizePipelines::new(ctx)?),
            ratio4_global_topk: Some(S14Ratio4GlobalTopKPipeline::new(ctx)?),
            ratio4_main_gather: Some(S14Ratio4MainPageGatherPipeline::new(ctx)?),
            ratio128_compressed_rope: Some(build_ratio128_compressed_rope_buffer(ctx)?),
            ratio128_finalize: Some(S14Ratio128CompressorFinalizePipelines::new(ctx)?),
            hc_post: Some(S14HcPostPipeline::new(ctx)?),
            ape_add: Some(S14Position0ApeAddPipeline::new(ctx)?),
            route: Some(S14RoutePostprocessGpuPipeline::new(ctx)?),
            route_slot_align: Some(S14RouteSlotAlignPipeline::new(ctx)?),
            timeline: Some(S14DualQueueTimeline::new(ctx)?),
            command_pool,
            layer_command,
            owns_command_pool: false,
            binders: Vec::new(),
            verified: store.stats(),
            last_compute_value: 0,
            layer_kv_offset: None,
            numeric_route_override: false,
            reference_route_replay: true,
            numeric_layer_input: false,
            numeric_hc_override: false,
            external_timeline_drained: false,
        })
    }

    fn new(
        ctx: &'ctx VulkanContext,
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        static_layout: &S14Position0StaticLayerLayout,
        graph: &S14Position0L0GraphPlan,
        payload_root: &Path,
        numeric_route_override: Option<[f32; EXPERTS_PER_TOKEN]>,
        numeric_layer_input: Option<&[u8]>,
        numeric_hc_override: Option<&[u8]>,
    ) -> Result<Self> {
        let mut owner = Self {
            ctx,
            static_arena: None,
            routed_arena: None,
            workspace: None,
            external_static_arena: None,
            external_routed_arena: None,
            external_workspace: None,
            immutable: None,
            paged_immutables: Vec::new(),
            active_paged_immutable: None,
            readback: None,
            embedding_pipeline: None,
            numeric: None,
            numeric_exact: None,
            sparse_index_query_numeric_exact: None,
            rmsnorm: None,
            q_head_normalize: None,
            f32_to_bf16: None,
            bf16_to_f32: None,
            qdq: None,
            attention: None,
            position1_attention: None,
            position3_attention: None,
            sparse_attention: None,
            position1_rope: None,
            ratio4_compressed_rope: None,
            ratio4_finalize: None,
            ratio4_global_topk: None,
            ratio4_main_gather: None,
            ratio128_compressed_rope: None,
            ratio128_finalize: None,
            hc_post: None,
            ape_add: None,
            route: None,
            route_slot_align: None,
            timeline: None,
            command_pool: vk::CommandPool::null(),
            layer_command: vk::CommandBuffer::null(),
            owns_command_pool: false,
            binders: Vec::new(),
            verified: VerifiedMappedAssetStats::default(),
            last_compute_value: 0,
            layer_kv_offset: None,
            numeric_route_override: numeric_route_override.is_some(),
            reference_route_replay: false,
            numeric_layer_input: numeric_layer_input.is_some(),
            numeric_hc_override: numeric_hc_override.is_some(),
            external_timeline_drained: false,
        };

        let layer = manifest
            .layers
            .get(graph.layer_index)
            .ok_or_else(|| anyhow!("position0 manifest 缺少 L{}", graph.layer))?;
        let routed_plan = weights
            .routed_layers
            .get(graph.layer_index)
            .ok_or_else(|| anyhow!("position0 weight plan 缺少 L{} routed 页", graph.layer))?;
        let mut store = VerifiedMappedAssetStore::new(payload_root)?;

        let static_assets = static_layout
            .assets
            .iter()
            .map(|placement| {
                let asset = find_layer_asset(layer, &placement.tensor)?;
                if asset.bytes != placement.bytes {
                    bail!(
                        "L{} static placement 字节漂移: {}",
                        graph.layer,
                        placement.tensor
                    );
                }
                Ok((asset, placement.local_offset))
            })
            .collect::<Result<Vec<_>>>()?;
        owner.static_arena = Some(upload_verified_arena(
            ctx,
            &mut store,
            &static_assets,
            static_layout.requested_bytes,
            &format!("L{} static", graph.layer),
        )?);

        let routed_assets = routed_plan
            .assets
            .iter()
            .map(|placement| {
                let asset = layer
                    .assets
                    .routed
                    .iter()
                    .find(|asset| asset.tensor == placement.tensor)
                    .ok_or_else(|| {
                        anyhow!("L{} routed manifest 缺少 {}", graph.layer, placement.tensor)
                    })?;
                if asset.bytes != placement.bytes {
                    bail!(
                        "L{} routed placement 字节漂移: {}",
                        graph.layer,
                        placement.tensor
                    );
                }
                Ok((asset, placement.offset))
            })
            .collect::<Result<Vec<_>>>()?;
        owner.routed_arena = Some(upload_verified_arena(
            ctx,
            &mut store,
            &routed_assets,
            routed_plan.used_bytes,
            &format!("L{} routed", graph.layer),
        )?);

        let embedding = store.map_verified_batch(std::slice::from_ref(&manifest.embedding_row))?;
        let physical_ids = if graph.route_mode == S14RoutePostprocessGpuMode::PhysicalIds {
            let tid2eid_name = format!("layers.{}.ffn.gate.tid2eid", graph.layer);
            let tid2eid = layer
                .assets
                .router
                .iter()
                .find(|asset| asset.tensor == tid2eid_name)
                .ok_or_else(|| anyhow!("L{} manifest 缺少 tid2eid", graph.layer))?;
            let mapped_tid2eid = store.map_verified_batch(std::slice::from_ref(tid2eid))?;
            graph.decode_and_validate_tid2eid_bos(tid2eid, mapped_tid2eid[0].bytes())?
        } else {
            graph.routed_expert_ids.map(u32::from)
        };
        if let Some(input) = numeric_layer_input {
            if input.len() as u64 != AUX_LAYER_INPUT_BYTES {
                bail!(
                    "L{} numeric layer input 字节漂移: actual={} expected={AUX_LAYER_INPUT_BYTES}",
                    graph.layer,
                    input.len()
                );
            }
        }
        if let Some(aux) = numeric_hc_override {
            if aux.len() as u64 != AUX_HC_OVERRIDE_BYTES {
                bail!(
                    "L{} numeric HC override 字节漂移: actual={} expected={AUX_HC_OVERRIDE_BYTES}",
                    graph.layer,
                    aux.len()
                );
            }
        }

        let immutable = host_storage_buffer(ctx, AUX_LOGICAL_BYTES)?;
        unsafe {
            std::ptr::write_bytes(immutable.mapped(), 0, AUX_LOGICAL_BYTES as usize);
            immutable.write_at(AUX_EMBEDDING_OFFSET as usize, embedding[0].bytes());
            let mut metadata = Vec::with_capacity(EXPERTS_PER_TOKEN * 6 * 4);
            for branch in graph.routed_metadata {
                for word in branch.words() {
                    metadata.extend_from_slice(&word.to_le_bytes());
                }
            }
            immutable.write_at(AUX_METADATA_OFFSET as usize, &metadata);
            immutable.write_at(
                AUX_PHYSICAL_IDS_OFFSET as usize,
                bytemuck::cast_slice(&physical_ids),
            );
            let ones = vec![0x3f80u16; 512];
            immutable.write_at(AUX_Q_RMS_ONES_OFFSET as usize, bytemuck::cast_slice(&ones));
            immutable.write_at(AUX_SHARED_ROUTE_ONE_OFFSET as usize, &1.0f32.to_le_bytes());
            if let Some(route_weights) = numeric_route_override {
                immutable.write_at(
                    AUX_NUMERIC_ROUTE_OVERRIDE_OFFSET as usize,
                    bytemuck::cast_slice(&route_weights),
                );
            }
            if let Some(input) = numeric_layer_input {
                immutable.write_at(AUX_LAYER_INPUT_OFFSET as usize, input);
            }
            if let Some(aux) = numeric_hc_override {
                immutable.write_at(AUX_HC_OVERRIDE_OFFSET as usize, aux);
            }
        }
        owner.immutable = Some(immutable);
        owner.workspace = Some(GpuBuffer::new_vram(
            ctx,
            S14_POSITION0_L0_WORKSPACE_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?);
        owner.readback = Some(GpuBuffer::new(
            ctx,
            READBACK_LOGICAL_BYTES,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?);

        owner.embedding_pipeline = Some(S14EmbeddingBroadcastPipeline::new(ctx)?);
        owner.numeric = Some(S14NumericPipelines::new(ctx)?);
        if numeric_layer_input.is_some() {
            // 冻结参考门只在已证明确有 BF16 midpoint 差异的投影上选择
            // audit pipeline，其余投影保持生产归约树，避免修一处又漂移 Q/KV。
            owner.numeric_exact = Some(S14NumericPipelines::new_exact_audit(ctx)?);
        }
        owner.rmsnorm = Some(S14Bf16RmsNormPipeline::new(ctx)?);
        owner.q_head_normalize = Some(S14Bf16QHeadNormalizePipeline::new(ctx)?);
        owner.f32_to_bf16 = Some(S14F32ToBf16Pipeline::new(ctx)?);
        owner.bf16_to_f32 = Some(S14Bf16ToF32Pipeline::new(ctx)?);
        owner.qdq = Some(S14E4m3QdqPipeline::new(ctx)?);
        owner.attention = Some(S14Position0AttentionPipeline::new(ctx)?);
        owner.position1_attention = Some(S14Position1AttentionPipeline::new(ctx)?);
        owner.position3_attention = Some(S14Position3AttentionPipeline::new(ctx)?);
        owner.sparse_attention = Some(S14SparseAttentionPipelines::new(ctx)?);
        owner.position1_rope = Some(build_position_rope_buffer(ctx, 0)?);
        owner.ratio4_compressed_rope = Some(build_ratio4_compressed_rope_buffer(ctx)?);
        owner.ratio4_finalize = Some(S14Ratio4CompressorFinalizePipelines::new(ctx)?);
        owner.ratio4_global_topk = Some(S14Ratio4GlobalTopKPipeline::new(ctx)?);
        owner.ratio4_main_gather = Some(S14Ratio4MainPageGatherPipeline::new(ctx)?);
        owner.ratio128_compressed_rope = Some(build_ratio128_compressed_rope_buffer(ctx)?);
        owner.ratio128_finalize = Some(S14Ratio128CompressorFinalizePipelines::new(ctx)?);
        owner.hc_post = Some(S14HcPostPipeline::new(ctx)?);
        owner.ape_add = Some(S14Position0ApeAddPipeline::new(ctx)?);
        owner.route = Some(S14RoutePostprocessGpuPipeline::new(ctx)?);
        owner.route_slot_align = Some(S14RouteSlotAlignPipeline::new(ctx)?);
        owner.timeline = Some(S14DualQueueTimeline::new(ctx)?);

        owner.command_pool = unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        owner.layer_command = unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(owner.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };
        owner.owns_command_pool = true;
        owner.verified = store.stats();
        Ok(owner)
    }

    fn static_arena(&self) -> &GpuBuffer {
        self.external_static_arena
            .or(self.static_arena.as_ref())
            .expect("L0 static owner")
    }

    fn routed_arena(&self) -> &GpuBuffer {
        self.external_routed_arena
            .or(self.routed_arena.as_ref())
            .expect("L0 routed owner")
    }

    fn workspace(&self) -> &GpuBuffer {
        self.external_workspace
            .or(self.workspace.as_ref())
            .expect("L0 workspace owner")
    }

    fn immutable(&self) -> &GpuBuffer {
        match self.active_paged_immutable {
            Some(index) => self
                .paged_immutables
                .get(index)
                .expect("paged immutable layer owner"),
            None => self.immutable.as_ref().expect("L0 immutable owner"),
        }
    }

    fn position1_rope(&self) -> &GpuBuffer {
        self.position1_rope.as_ref().expect("position1 RoPE owner")
    }

    fn ratio4_compressed_rope(&self) -> &GpuBuffer {
        self.ratio4_compressed_rope
            .as_ref()
            .expect("ratio4 compressed RoPE owner")
    }

    fn ratio128_compressed_rope(&self) -> &GpuBuffer {
        self.ratio128_compressed_rope
            .as_ref()
            .expect("ratio128 compressed RoPE owner")
    }

    /// 异步录制43层时，descriptor 不能继续引用随后会被主机覆盖的同一个 immutable。
    /// 每层保留约43KiB只读元数据快照，总量不足2MiB。
    fn prepare_paged_immutable_snapshots(
        &mut self,
        graphs: &[S14Position0L0GraphPlan],
    ) -> Result<()> {
        if !self.paged_immutables.is_empty() || graphs.len() != FULL_DEPTH_LAYERS.len() {
            bail!("position0 paged immutable snapshots 重复或层数漂移");
        }
        let mut snapshots = Vec::<GpuBuffer>::with_capacity(graphs.len());
        for graph in graphs {
            let buffer = match host_storage_buffer(self.ctx, AUX_LOGICAL_BYTES) {
                Ok(buffer) => buffer,
                Err(error) => {
                    for buffer in snapshots {
                        buffer.destroy(self.ctx);
                    }
                    return Err(error).context("allocate paged immutable snapshot");
                }
            };
            unsafe {
                std::ptr::write_bytes(buffer.mapped(), 0, AUX_LOGICAL_BYTES as usize);
                let mut metadata = Vec::with_capacity(EXPERTS_PER_TOKEN * 6 * 4);
                for branch in graph.routed_metadata {
                    for word in branch.words() {
                        metadata.extend_from_slice(&word.to_le_bytes());
                    }
                }
                buffer.write_at(AUX_METADATA_OFFSET as usize, &metadata);
                buffer.write_at(
                    AUX_PHYSICAL_IDS_OFFSET as usize,
                    bytemuck::cast_slice(&graph.routed_expert_ids.map(u32::from)),
                );
                buffer.write_at(
                    AUX_NUMERIC_ROUTE_OVERRIDE_OFFSET as usize,
                    bytemuck::cast_slice(&graph.routed_route_weight_bits),
                );
                buffer.write_at(
                    AUX_Q_RMS_ONES_OFFSET as usize,
                    bytemuck::cast_slice(&vec![0x3f80u16; 512]),
                );
                buffer.write_at(AUX_SHARED_ROUTE_ONE_OFFSET as usize, &1.0f32.to_le_bytes());
            }
            snapshots.push(buffer);
        }
        self.paged_immutables = snapshots;
        Ok(())
    }

    fn select_paged_immutable(&mut self, layer: u8) -> Result<()> {
        let index = usize::from(layer);
        if self.paged_immutables.len() != FULL_DEPTH_LAYERS.len()
            || self.paged_immutables.get(index).is_none()
        {
            bail!("position0 paged immutable L{layer} 尚未准备");
        }
        self.active_paged_immutable = Some(index);
        Ok(())
    }

    fn readback(&self) -> &GpuBuffer {
        self.readback.as_ref().expect("L0 readback owner")
    }

    /// 同步首 token 桥只替换权重页引用；workspace 由 paged arena 唯一拥有并跨43层保留。
    /// 调用方必须已经等待上一层 compute，保证旧 descriptor 与滚动页不再 in-flight。
    fn attach_external_layer_buffers(
        &mut self,
        static_arena: &'ctx GpuBuffer,
        routed_arena: &'ctx GpuBuffer,
        workspace: &'ctx GpuBuffer,
        graph: &S14Position0L0GraphPlan,
    ) -> Result<()> {
        if static_arena.size() < graph.static_logical_bytes
            || routed_arena.size() < graph.routed_logical_bytes
            || workspace.size() < S14_POSITION0_L0_WORKSPACE_BYTES
        {
            bail!("L{} external paged buffer 容量不足", graph.layer);
        }
        if self.last_compute_value != 0 && !self.binders.is_empty() {
            bail!(
                "L{} external reconfigure 前仍有 in-flight descriptor",
                graph.layer
            );
        }
        self.external_static_arena = Some(static_arena);
        self.external_routed_arena = Some(routed_arena);
        self.external_workspace = Some(workspace);

        let mut metadata = Vec::with_capacity(EXPERTS_PER_TOKEN * 6 * 4);
        for branch in graph.routed_metadata {
            for word in branch.words() {
                metadata.extend_from_slice(&word.to_le_bytes());
            }
        }
        unsafe {
            self.immutable()
                .write_at(AUX_METADATA_OFFSET as usize, &metadata);
            self.immutable().write_at(
                AUX_NUMERIC_ROUTE_OVERRIDE_OFFSET as usize,
                bytemuck::cast_slice(&graph.routed_route_weight_bits),
            );
        }
        self.layer_kv_offset = None;
        Ok(())
    }
}

impl Drop for S14Position0L0GpuOwner<'_> {
    fn drop(&mut self) {
        if !self.external_timeline_drained {
            unsafe {
                let _ = self.ctx.device.device_wait_idle();
            }
        }
        for binder in self.binders.drain(..) {
            binder.destroy(self.ctx);
        }
        if let Some(pipeline) = self.route.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.route_slot_align.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.ape_add.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.hc_post.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.attention.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.position1_attention.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.position3_attention.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipelines) = self.sparse_attention.take() {
            pipelines.destroy(self.ctx);
        }
        if let Some(pipelines) = self.ratio4_finalize.take() {
            pipelines.destroy(self.ctx);
        }
        if let Some(pipeline) = self.ratio4_main_gather.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.ratio4_global_topk.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipelines) = self.ratio128_finalize.take() {
            pipelines.destroy(self.ctx);
        }
        if let Some(pipeline) = self.qdq.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.bf16_to_f32.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.f32_to_bf16.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.rmsnorm.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.q_head_normalize.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.numeric.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.numeric_exact.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.sparse_index_query_numeric_exact.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(pipeline) = self.embedding_pipeline.take() {
            pipeline.destroy(self.ctx);
        }
        if let Some(timeline) = self.timeline.take() {
            timeline.destroy(self.ctx);
        }
        unsafe {
            if self.owns_command_pool && self.command_pool != vk::CommandPool::null() {
                self.ctx
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
        }
        for buffer in [
            self.readback.take(),
            self.ratio128_compressed_rope.take(),
            self.ratio4_compressed_rope.take(),
            self.position1_rope.take(),
            self.immutable.take(),
            self.workspace.take(),
            self.routed_arena.take(),
            self.static_arena.take(),
        ]
        .into_iter()
        .flatten()
        {
            buffer.destroy(self.ctx);
        }
        for buffer in self.paged_immutables.drain(..) {
            buffer.destroy(self.ctx);
        }
    }
}

fn find_layer_asset<'a>(layer: &'a Position0Layer, tensor: &str) -> Result<&'a Position0Asset> {
    layer
        .assets
        .non_expert
        .iter()
        .chain(&layer.assets.router)
        .chain(&layer.assets.shared)
        .find(|asset| asset.tensor == tensor)
        .ok_or_else(|| anyhow!("L0 manifest 缺少 static tensor: {tensor}"))
}

fn host_storage_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
}

fn build_position_rope_buffer(ctx: &VulkanContext, position: u32) -> Result<GpuBuffer> {
    let buffer = host_storage_buffer(ctx, POSITION1_ROPE_BUFFER_BYTES)?;
    write_position_rope_buffer(&buffer, position)?;
    Ok(buffer)
}

fn build_ratio4_compressed_rope_buffer(ctx: &VulkanContext) -> Result<GpuBuffer> {
    let buffer = host_storage_buffer(ctx, RATIO4_COMPRESSED_ROPE_BYTES)?;
    write_ratio4_compressed_rope_buffer(&buffer, 3)?;
    Ok(buffer)
}

fn build_ratio128_compressed_rope_buffer(ctx: &VulkanContext) -> Result<GpuBuffer> {
    let buffer = host_storage_buffer(ctx, RATIO128_COMPRESSED_ROPE_BYTES)?;
    write_ratio128_compressed_rope_buffer(&buffer, 127)?;
    Ok(buffer)
}

fn write_ratio4_compressed_rope_buffer(buffer: &GpuBuffer, position: u32) -> Result<()> {
    let boundary = S14Ratio4CompressorBoundary::new(position)?;
    let rope = boundary.rope_cos_sin()?;
    unsafe {
        buffer.write_at(0, bytemuck::cast_slice(&rope));
    }
    Ok(())
}

fn write_ratio128_compressed_rope_buffer(buffer: &GpuBuffer, position: u32) -> Result<()> {
    if buffer.size() < RATIO128_COMPRESSED_ROPE_BYTES {
        bail!("ratio128 compressed RoPE buffer 容量不足");
    }
    let boundary = S14Ratio128CompressorBoundary::new(position)?;
    let rope = boundary.rope_cos_sin()?;
    unsafe {
        buffer.write_at(0, bytemuck::cast_slice(&rope));
    }
    Ok(())
}

fn write_position_rope_buffer(buffer: &GpuBuffer, position: u32) -> Result<()> {
    if buffer.size() < POSITION1_ROPE_BUFFER_BYTES {
        bail!("position RoPE buffer 容量不足");
    }
    let ratio0 = position_rope_cos_sin(position, 0)?;
    let yarn = position_rope_cos_sin(position, 4)?;
    unsafe {
        buffer.write_at(
            POSITION1_ROPE_RATIO0_OFFSET as usize,
            bytemuck::cast_slice(&ratio0),
        );
        buffer.write_at(
            POSITION1_ROPE_YARN_OFFSET as usize,
            bytemuck::cast_slice(&yarn),
        );
    }
    Ok(())
}

fn upload_verified_arena(
    ctx: &VulkanContext,
    store: &mut VerifiedMappedAssetStore,
    placements: &[(&Position0Asset, u64)],
    logical_bytes: u64,
    label: &str,
) -> Result<GpuBuffer> {
    if placements.is_empty() || logical_bytes == 0 {
        bail!("{label} upload 不能为空");
    }
    let assets = placements
        .iter()
        .map(|(asset, _)| (*asset).clone())
        .collect::<Vec<_>>();
    let mapped = store.map_verified_batch(&assets)?;
    let staging = GpuBuffer::new_staging(ctx, logical_bytes)
        .with_context(|| format!("allocate {label} staging"))?;
    unsafe {
        std::ptr::write_bytes(staging.mapped(), 0, logical_bytes as usize);
        for ((asset, offset), payload) in placements.iter().zip(&mapped) {
            let end = offset
                .checked_add(asset.bytes)
                .ok_or_else(|| anyhow!("{label} placement overflow: {}", asset.tensor))?;
            if end > logical_bytes || payload.bytes().len() as u64 != asset.bytes {
                staging.destroy(ctx);
                bail!("{label} placement 越界/字节漂移: {}", asset.tensor);
            }
            staging.write_at(*offset as usize, payload.bytes());
        }
    }
    let device = match GpuBuffer::new_vram(
        ctx,
        logical_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC,
    ) {
        Ok(buffer) => buffer,
        Err(error) => {
            staging.destroy(ctx);
            return Err(error).with_context(|| format!("allocate {label} VRAM"));
        }
    };
    let result = upload_and_wait(ctx, &staging, &device, logical_bytes, label);
    staging.destroy(ctx);
    if let Err(error) = result {
        device.destroy(ctx);
        return Err(error);
    }
    Ok(device)
}

fn upload_and_wait(
    ctx: &VulkanContext,
    staging: &GpuBuffer,
    device: &GpuBuffer,
    bytes: u64,
    label: &str,
) -> Result<()> {
    let pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_graphics),
            None,
        )?
    };
    let command = match unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    } {
        Ok(commands) => commands[0],
        Err(error) => {
            unsafe { ctx.device.destroy_command_pool(pool, None) };
            return Err(error.into());
        }
    };
    let fence = match unsafe {
        ctx.device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    } {
        Ok(fence) => fence,
        Err(error) => {
            unsafe { ctx.device.destroy_command_pool(pool, None) };
            return Err(error.into());
        }
    };
    let result = (|| -> Result<()> {
        unsafe {
            ctx.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            ctx.device.cmd_copy_buffer(
                command,
                staging.handle(),
                device.handle(),
                &[vk::BufferCopy::default().size(bytes)],
            );
            ctx.device.end_command_buffer(command)?;
            let commands = [command];
            ctx.device.queue_submit(
                ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                fence,
            )?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        Ok(())
    })()
    .with_context(|| format!("upload {label}"));
    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    result
}

/// 首枚完整 token 使用的同步 Vulkan 层适配器。
///
/// 它复用已经逐层验证的 recorder 与 paged arena；每层显式等待上一层后才覆盖滚动页，
/// 因此不是最终性能路径，但 hidden 始终留在同一 GPU workspace。后续双页 timeline 只需
/// 替换该适配器，不改变 layer graph、状态 recipe 或 terminal 数学。
fn same_runtime_asset_identity(left: &Position0Asset, right: &Position0Asset) -> bool {
    left.tensor == right.tensor
        && left.kind == right.kind
        && left.expert_id == right.expert_id
        && left.dtype == right.dtype
        && left.shape == right.shape
        && left.bytes == right.bytes
        && left.range_key == right.range_key
        && left.cache_key == right.cache_key
        && left.path == right.path
        && left.sha256 == right.sha256
        && left.proof_path == right.proof_path
        && left.proof_sha256 == right.proof_sha256
        && left.hash_authority == right.hash_authority
        && left.payload_rehashed_by_builder == right.payload_rehashed_by_builder
        && left.source == right.source
}

/// 跨 token 常驻、且不借用 session candidate 的 host/Vulkan upload资源。
///
/// uploader的6个staging buffer、command pool/buffer/fence，只读SHA mmap store和
/// 43层graph plan都在runtime load时建立。每个token只重置游标并重绑动态页。
pub struct S14Position0PersistentHostResources {
    uploader: S14Position0HybridUploader,
    store: VerifiedMappedAssetStore,
    graphs: Vec<S14Position0L0GraphPlan>,
    backend_command_pool: vk::CommandPool,
    backend_layer_command: vk::CommandBuffer,
    steps_started: u64,
    active_token: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0PersistentHostStepTelemetry {
    pub step_index: u64,
    pub reused_from_previous_step: bool,
    /// 这些资源全部在runtime load时创建；token路径必须恒为0。
    pub resource_allocations_this_step: u64,
    pub uploader_staging_allocations_this_step: u64,
    pub mapped_store_allocations_this_step: u64,
    pub graph_plan_allocations_this_step: u64,
    pub backend_command_allocations_this_step: u64,
}

impl S14Position0PersistentHostResources {
    pub fn new(
        ctx: &VulkanContext,
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        arena: &S14Position0PagedWeightArena,
        payload_root: &Path,
    ) -> Result<Self> {
        weights.validate(manifest)?;
        let physical = &arena.plan().physical;
        if physical.static_layers.len() != FULL_DEPTH_LAYERS.len() {
            bail!("persistent position0 paged arena 不是43层");
        }
        let mut graphs = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
        for index in 0..FULL_DEPTH_LAYERS.len() {
            graphs.push(build_position0_layer_graph_plan(
                manifest,
                weights,
                &physical.static_layers[index],
                index,
            )?);
        }
        let mut store = VerifiedMappedAssetStore::new(payload_root)?;
        let mut uploader = S14Position0HybridUploader::new(ctx, weights)?;
        if let Err(error) = uploader.upload_static_once(ctx, manifest, weights, &mut store, arena) {
            uploader.destroy(ctx);
            return Err(error);
        }
        let backend_command_pool = match unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(error) => {
                uploader.destroy(ctx);
                return Err(error.into());
            }
        };
        let backend_layer_command = match unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(backend_command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(commands) => commands[0],
            Err(error) => {
                unsafe {
                    ctx.device.destroy_command_pool(backend_command_pool, None);
                }
                uploader.destroy(ctx);
                return Err(error.into());
            }
        };
        Ok(Self {
            uploader,
            store,
            graphs,
            backend_command_pool,
            backend_layer_command,
            steps_started: 0,
            active_token: false,
        })
    }

    /// 单token runtime完成并排空后，把同一verified uploader/store移交给K-block terminal/provider。
    /// backend command pool与graph快照不跨模式复用，避免旧descriptor继续借用position0 candidate。
    pub fn into_causal_block_upload_parts(
        self,
        ctx: &VulkanContext,
        weights: &S14Position0HybridWeightPlan,
    ) -> Result<(S14Position0HybridUploader, VerifiedMappedAssetStore)> {
        if self.active_token {
            bail!("active position0 token存在时禁止移交causal-block uploader/store");
        }
        let Self {
            mut uploader,
            store,
            graphs: _,
            backend_command_pool,
            backend_layer_command: _,
            steps_started: _,
            active_token,
        } = self;
        debug_assert!(!active_token);
        unsafe {
            ctx.device.destroy_command_pool(backend_command_pool, None);
        }
        if let Err(error) = uploader.begin_persistent_token(weights, false) {
            uploader.destroy(ctx);
            return Err(error.context("把 position0 uploader 切换到 causal-block token 起点"));
        }
        Ok((uploader, store))
    }

    fn begin_token(
        &mut self,
        weights: &S14Position0HybridWeightPlan,
    ) -> Result<S14Position0PersistentHostStepTelemetry> {
        if self.active_token {
            bail!("persistent host resources已有active token");
        }
        let reused = self.steps_started != 0;
        self.uploader.begin_persistent_token(weights, !reused)?;
        let receipt = S14Position0PersistentHostStepTelemetry {
            step_index: self.steps_started,
            reused_from_previous_step: reused,
            resource_allocations_this_step: 0,
            uploader_staging_allocations_this_step: 0,
            mapped_store_allocations_this_step: 0,
            graph_plan_allocations_this_step: 0,
            backend_command_allocations_this_step: 0,
        };
        self.steps_started = self
            .steps_started
            .checked_add(1)
            .ok_or_else(|| anyhow!("persistent host step counter overflow"))?;
        self.active_token = true;
        Ok(receipt)
    }

    fn finish_token(&mut self, weights: &S14Position0HybridWeightPlan) -> Result<()> {
        if !self.active_token || !self.uploader.is_complete(weights) {
            bail!("persistent host resources完成回执早于完整token");
        }
        self.active_token = false;
        Ok(())
    }

    fn abort_token_after_drain(&mut self) -> Result<()> {
        if !self.active_token {
            bail!("persistent host resources没有可回滚token");
        }
        self.uploader.abort_persistent_token_after_drain();
        self.steps_started = self
            .steps_started
            .checked_sub(1)
            .ok_or_else(|| anyhow!("persistent host step counter underflow"))?;
        self.active_token = false;
        Ok(())
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        unsafe {
            ctx.device
                .destroy_command_pool(self.backend_command_pool, None);
        }
        self.uploader.destroy(ctx);
    }
}

pub struct S14Position0SynchronousVulkanLayerAdapter<'ctx, 'persistent> {
    ctx: &'ctx VulkanContext,
    manifest: &'ctx Position0WholeTokenManifest,
    weights: &'ctx S14Position0HybridWeightPlan,
    arena: &'ctx S14Position0PagedWeightArena,
    resources: &'persistent mut S14Position0PersistentHostResources,
    persistent_step: S14Position0PersistentHostStepTelemetry,
    state_recording: S14Position0FullDepthStateRecordingProgram,
    gpu: Option<S14Position0L0GpuOwner<'ctx>>,
    candidate: Position0GpuCandidate<'ctx>,
    current_layer: Option<u8>,
    pending_router_probe_layer: Option<u8>,
    pending_online_route: Option<OnlineTop6>,
    pending_dynamic_routed_bank: Option<(u8, usize)>,
    bound_embedding: Position0Asset,
    input_execution: Option<S14PositionExecutionPlan>,
    embedding_submitted: bool,
    payload_root: PathBuf,
}

impl<'ctx, 'persistent> S14Position0SynchronousVulkanLayerAdapter<'ctx, 'persistent> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &'ctx VulkanContext,
        manifest: &'ctx Position0WholeTokenManifest,
        weights: &'ctx S14Position0HybridWeightPlan,
        arena: &'ctx S14Position0PagedWeightArena,
        payload_root: &Path,
        resources: &'persistent mut S14Position0PersistentHostResources,
        candidate: Position0GpuCandidate<'ctx>,
    ) -> Result<Self> {
        if !std::ptr::eq(ctx, candidate.ctx) {
            bail!("position0 synchronous adapter Vulkan context 漂移");
        }
        validate_production_layer_position(candidate.committed_host_state.position)?;
        weights.validate(manifest)?;
        let layer_program =
            S14Position0FullDepthLayerProgram::build(manifest, weights, arena.workspace_layout())?;
        let state_recording = S14Position0FullDepthStateRecordingProgram::build(
            &layer_program,
            arena.workspace_layout(),
            &candidate.committed_host_state.native,
        )?;
        // 从构造起只借用 paged arena，禁止为复用单层 recorder 再申请一份 L0/static/workspace。
        let static_buffer = match arena.static_layer(0)? {
            S14Position0StaticLayerBinding::Resident { buffer, .. }
            | S14Position0StaticLayerBinding::Streamed { buffer, .. } => buffer,
        };
        let mut gpu = S14Position0L0GpuOwner::new_external(
            ctx,
            manifest,
            &resources.graphs[0],
            &mut resources.store,
            static_buffer,
            arena.routed(0)?,
            arena.workspace(),
            resources.backend_command_pool,
            resources.backend_layer_command,
        )?;
        gpu.prepare_paged_immutable_snapshots(&resources.graphs)?;
        // 所有可能失败的candidate-scoped Vulkan构造已经完成；最后才打开
        // persistent token，保证构造失败不会遗留active uploader游标。
        let persistent_step = resources.begin_token(weights)?;
        Ok(Self {
            ctx,
            manifest,
            weights,
            arena,
            resources,
            persistent_step,
            state_recording,
            gpu: Some(gpu),
            candidate,
            current_layer: None,
            pending_router_probe_layer: None,
            pending_online_route: None,
            pending_dynamic_routed_bank: None,
            bound_embedding: manifest.embedding_row.clone(),
            input_execution: None,
            embedding_submitted: false,
            payload_root: payload_root.to_path_buf(),
        })
    }

    pub fn persistent_step_telemetry(&self) -> S14Position0PersistentHostStepTelemetry {
        self.persistent_step
    }

    /// 把 catalog 派生的任意 token embedding Range 绑定到下一次 prologue。
    ///
    /// payload 在写入 host-coherent immutable 区前仍由 `VerifiedMappedAssetStore`
    /// 完整 SHA；position/RoPE/window/remainder 参数与同一计划一起保存，后续
    /// sparse-attention/state recorder 必须消费这份合同，不能回退到隐式 row0。
    pub fn bind_input_prologue(
        &mut self,
        input: &S14InputAssetPlan,
        embedding: &Position0Asset,
    ) -> Result<()> {
        if self.embedding_submitted || self.current_layer.is_some() {
            bail!("S14 input prologue 已提交或 layer 已开始");
        }
        input
            .position_execution
            .validate_for_position(input.position)?;
        validate_production_layer_position(input.position)?;
        if input.position != self.candidate.committed_host_state.position
            || input.input_token_id != self.candidate.committed_host_state.input_token_id
        {
            bail!("S14 input plan 与 DecoderState position/token/RoPE/window 漂移");
        }
        input
            .embedding
            .validate_resolved_position0_asset(embedding, None)?;
        let mapped = self
            .resources
            .store
            .map_verified_batch(std::slice::from_ref(embedding))?;
        if mapped.len() != 1 || mapped[0].bytes().len() != 8_192 {
            bail!("S14 input embedding payload 不是精确 8 KiB 行");
        }

        // L0--L2不按router score选逻辑expert，而是读取当前input token对应的
        // tid2eid物理行。三行必须在任何probe command录制前，从完整SHA验证过的
        // payload写入各自immutable快照；禁止沿用构造期BOS/row0。
        let mut current_token_physical_rows = Vec::with_capacity(3);
        for graph in self
            .resources
            .graphs
            .iter()
            .filter(|graph| graph.route_mode == S14RoutePostprocessGpuMode::PhysicalIds)
        {
            let layer = self
                .manifest
                .layers
                .get(graph.layer_index)
                .ok_or_else(|| anyhow!("S14 input tid2eid缺少L{} manifest", graph.layer))?;
            let tensor = format!("layers.{}.ffn.gate.tid2eid", graph.layer);
            let asset = layer
                .assets
                .router
                .iter()
                .find(|asset| asset.tensor == tensor)
                .ok_or_else(|| anyhow!("S14 input tid2eid缺少L{}资产", graph.layer))?;
            let mapped_tid2eid = self
                .resources
                .store
                .map_verified_batch(std::slice::from_ref(asset))?;
            if mapped_tid2eid.len() != 1 {
                bail!("S14 input tid2eid L{}映射数量漂移", graph.layer);
            }
            let ids = graph.decode_tid2eid_row_for_token(
                asset,
                mapped_tid2eid[0].bytes(),
                input.input_token_id,
            )?;
            current_token_physical_rows.push((graph.layer_index, graph.layer, ids));
        }
        if current_token_physical_rows.len() != 3
            || current_token_physical_rows
                .iter()
                .map(|(_, layer, _)| *layer)
                .collect::<Vec<_>>()
                != [0, 1, 2]
        {
            bail!("S14 input tid2eid physical层集合不是严格L0--L2");
        }
        let gpu = self.gpu.as_mut().expect("synchronous GPU owner");
        // 当前 token 尚未录制任何 command；安全更新这一 token 唯一的两套
        // ratio0/YARN RoPE 表，43层 descriptor 都只读同一稳定 payload。
        write_position_rope_buffer(gpu.position1_rope(), input.position)?;
        // ratio4 finalize 使用压缩块自身的位置，而不是当前 token 的普通 RoPE。
        // 每个 position%4==3 边界都必须刷新；沿用构造期 position3 identity 会让
        // position7/11/... 把后续 compressed block 旋转到错误坐标。
        if input.position % 4 == 3 {
            write_ratio4_compressed_rope_buffer(gpu.ratio4_compressed_rope(), input.position)?;
        }
        if input.position % 128 == 127 {
            write_ratio128_compressed_rope_buffer(gpu.ratio128_compressed_rope(), input.position)?;
        }
        unsafe {
            gpu.immutable()
                .write_at(AUX_EMBEDDING_OFFSET as usize, mapped[0].bytes());
            for (layer_index, layer, ids) in current_token_physical_rows {
                let immutable = gpu
                    .paged_immutables
                    .get(layer_index)
                    .ok_or_else(|| anyhow!("S14 input tid2eid L{layer} immutable快照尚未准备"))?;
                immutable.write_at(AUX_PHYSICAL_IDS_OFFSET as usize, bytemuck::cast_slice(&ids));
            }
        }
        self.bound_embedding = embedding.clone();
        self.input_execution = Some(input.position_execution.clone());
        Ok(())
    }

    pub fn input_execution(&self) -> Option<&S14PositionExecutionPlan> {
        self.input_execution.as_ref()
    }

    /// `prologue_command` 已由 whole-token candidate 录入 committed→inactive 修复与 sticky 清零。
    pub fn submit_embedding(
        &mut self,
        prologue_command: vk::CommandBuffer,
        embedding: &Position0Asset,
    ) -> Result<u64> {
        unsafe { self.record_embedding_prologue(prologue_command, embedding)? };
        let gpu = self.gpu.as_mut().expect("synchronous GPU owner");
        let value = unsafe {
            gpu.timeline
                .as_mut()
                .expect("synchronous timeline")
                .submit_compute_only(self.ctx, prologue_command)?
        };
        gpu.last_compute_value = value;
        Ok(value)
    }

    /// 在 whole-token candidate 已 begin 的 prologue command 中追加真实 embedding，
    /// 结束录制但不 submit、不 wait；由共享 paged timeline 接管其生命周期。
    ///
    /// # Safety
    /// `prologue_command` 必须处于 recording，且引用资源活到 candidate final wait/drain。
    pub unsafe fn record_embedding_prologue(
        &mut self,
        prologue_command: vk::CommandBuffer,
        embedding: &Position0Asset,
    ) -> Result<()> {
        if self.embedding_submitted
            || !same_runtime_asset_identity(embedding, &self.bound_embedding)
        {
            bail!("position0 synchronous embedding 身份/phase 漂移");
        }
        let gpu = self.gpu.as_mut().expect("synchronous GPU owner");
        let shape = S14EmbeddingBroadcastShape::new(HIDDEN)?;
        let dispatch = gpu
            .embedding_pipeline
            .as_ref()
            .expect("embedding pipeline")
            .bind_slices(
                self.ctx,
                shape,
                StorageBufferSlice {
                    buffer: gpu.immutable(),
                    offset: AUX_EMBEDDING_OFFSET,
                },
                StorageBufferSlice {
                    buffer: gpu.workspace(),
                    offset: self
                        .resources
                        .graphs
                        .first()
                        .expect("L0 graph")
                        .workspace
                        .region(S14Position0WorkspaceSlot::HiddenStreamsA)
                        .offset,
                },
                StorageBufferSlice::whole(self.candidate.sticky_status),
            )?;
        gpu.embedding_pipeline
            .as_ref()
            .expect("embedding pipeline")
            .cmd(self.ctx, prologue_command, &dispatch);
        self.ctx.device.end_command_buffer(prologue_command)?;
        gpu.binders.push(dispatch.binder);
        self.embedding_submitted = true;
        Ok(())
    }

    pub fn payload_root(&self) -> &Path {
        &self.payload_root
    }

    /// 录制下一层 static/routed 双 bank transfer，不读 payload、不 submit、不 wait。
    ///
    /// # Safety
    /// `command` 必须是尚未 begin 的 transfer command，资源活到共享 timeline 排空。
    pub unsafe fn record_next_layer_transfer(
        &mut self,
        command: vk::CommandBuffer,
    ) -> Result<S14Position0LayerCopyReceipt> {
        self.ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let receipt = self.resources.uploader.record_next_layer_copies(
            self.ctx,
            command,
            self.manifest,
            self.weights,
            self.arena,
        );
        match receipt {
            Ok(receipt) => {
                self.ctx.device.end_command_buffer(command)?;
                Ok(receipt)
            }
            Err(error) => Err(error).context("record position0 paged layer transfer"),
        }
    }

    /// timeline 确认 staging bank 可复用后，填充已录制层 copy 的 verified payload。
    pub fn stage_recorded_layer(
        &mut self,
        receipt: S14Position0LayerCopyReceipt,
        timeline_bank: usize,
    ) -> Result<S14Position0LayerCopyReceipt> {
        self.resources.uploader.stage_recorded_layer(
            self.manifest,
            self.weights,
            &mut self.resources.store,
            self.arena,
            receipt,
            timeline_bank,
        )
    }

    /// Dynamic route probe 之前只准备当前层 static 权重。32个resident层为零拷贝
    /// receipt；streamed层在这里完成必要上传。routed bank 此时保持未触碰，避免
    /// 为了执行router先搬运manifest中的错误专家页。
    pub fn prepare_router_probe_static(
        &mut self,
        plan: &S14Position0SynchronousLayerPlan,
    ) -> Result<S14Position0StaticLayerUploadReceipt> {
        if self.pending_router_probe_layer.is_some()
            || self.pending_online_route.is_some()
            || self.pending_dynamic_routed_bank.is_some()
        {
            bail!("position0 router probe static prepare phase 漂移");
        }
        let receipt = self.resources.uploader.prepare_next_static_layer(
            self.ctx,
            self.manifest,
            self.weights,
            &mut self.resources.store,
            self.arena,
        )?;
        let bank_matches = if receipt.resident_hit {
            receipt.bank.is_none()
        } else {
            receipt.bank == Some(plan.routed_bank)
        };
        if receipt.layer != plan.layer || !bank_matches {
            bail!("position0 router probe static receipt layer/bank 漂移");
        }
        Ok(receipt)
    }

    /// 为下一层录制真实 compute command。每层绑定独立 immutable 快照，因而可以在
    /// GPU 尚未执行前一次性录完43层，主机不会把早期 descriptor 的元数据覆盖掉。
    ///
    /// # Safety
    /// `command` 必须尚未 begin；descriptor、arena、candidate 与 snapshots 活到 final wait。
    pub unsafe fn record_paged_layer_compute(
        &mut self,
        plan: &S14Position0SynchronousLayerPlan,
        command: vk::CommandBuffer,
    ) -> Result<()> {
        if self.candidate.committed_host_state.position != 0 {
            bail!("position1+禁止manifest route replay，必须执行online router probe");
        }
        if self.pending_router_probe_layer.is_some()
            || self.pending_online_route.is_some()
            || self.pending_dynamic_routed_bank.is_some()
        {
            bail!("position0 manifest replay 禁止跨过未闭合的 online router probe");
        }
        let expected = self
            .current_layer
            .map_or(0, |layer| layer.saturating_add(1));
        if !self.embedding_submitted || plan.layer != expected {
            bail!(
                "position0 paged layer record 顺序漂移: expected=L{expected} actual=L{}",
                plan.layer
            );
        }
        let index = usize::from(plan.layer);
        let graph = self
            .resources
            .graphs
            .get(index)
            .ok_or_else(|| anyhow!("position0 paged graph 缺少 L{}", plan.layer))?;
        let static_buffer = match self.arena.static_layer(plan.layer)? {
            S14Position0StaticLayerBinding::Resident { buffer, .. }
            | S14Position0StaticLayerBinding::Streamed { buffer, .. } => buffer,
        };
        let recipe = self
            .state_recording
            .layer(plan.layer)
            .ok_or_else(|| anyhow!("position0 state recipe 缺少 L{}", plan.layer))?;
        let gpu = self.gpu.as_mut().expect("paged GPU owner");
        gpu.select_paged_immutable(plan.layer)?;
        gpu.attach_external_layer_buffers(
            static_buffer,
            self.arena.routed(plan.routed_bank)?,
            self.arena.workspace(),
            graph,
        )?;
        let mut binders = record_layer_command_on(
            gpu,
            graph,
            &self.candidate,
            Some(recipe),
            self.input_execution.as_ref(),
            command,
            false,
            S14LayerCommandSegment::CompleteManifestReplay,
        )?;
        gpu.binders.append(&mut binders);
        self.current_layer = Some(plan.layer);
        Ok(())
    }

    /// 录制当前层的 attention/HC/router 前半段，并在 command 尾部只发布
    /// `top6 ids + route weights` 的 48-byte host-visible probe。routed MoE、shared、
    /// FFN HC-post 与状态写回均不会被录入该 command。
    ///
    /// # Safety
    /// `command` 必须尚未 begin。调用方必须提交并等待该 command，再调用
    /// [`Self::complete_router_probe_after_wait`]；在 continuation 完成前不得复用
    /// workspace、static bank 或 candidate state。
    pub unsafe fn record_paged_layer_router_probe(
        &mut self,
        plan: &S14Position0SynchronousLayerPlan,
        command: vk::CommandBuffer,
    ) -> Result<()> {
        let expected = self
            .current_layer
            .map_or(0, |layer| layer.saturating_add(1));
        if !self.embedding_submitted
            || self.pending_router_probe_layer.is_some()
            || self.pending_online_route.is_some()
            || self.pending_dynamic_routed_bank.is_some()
            || plan.layer != expected
        {
            bail!(
                "position0 router probe 顺序/phase 漂移: expected=L{expected} actual=L{}",
                plan.layer
            );
        }
        let index = usize::from(plan.layer);
        let graph = self
            .resources
            .graphs
            .get(index)
            .ok_or_else(|| anyhow!("position0 router probe graph 缺少 L{}", plan.layer))?;
        let static_buffer = match self.arena.static_layer(plan.layer)? {
            S14Position0StaticLayerBinding::Resident { buffer, .. }
            | S14Position0StaticLayerBinding::Streamed { buffer, .. } => buffer,
        };
        let recipe = self
            .state_recording
            .layer(plan.layer)
            .ok_or_else(|| anyhow!("position0 router probe state recipe 缺少 L{}", plan.layer))?;
        let gpu = self.gpu.as_mut().expect("paged GPU owner");
        gpu.select_paged_immutable(plan.layer)?;
        gpu.attach_external_layer_buffers(
            static_buffer,
            self.arena.routed(plan.routed_bank)?,
            self.arena.workspace(),
            graph,
        )?;
        let mut binders = record_layer_command_on(
            gpu,
            graph,
            &self.candidate,
            Some(recipe),
            self.input_execution.as_ref(),
            command,
            false,
            S14LayerCommandSegment::RouterProbe,
        )?;
        gpu.binders.append(&mut binders);
        self.pending_router_probe_layer = Some(plan.layer);
        Ok(())
    }

    /// 只允许在调用方已经等待 router probe command 后读取。这里不再把在线专家
    /// 与 position0 manifest 比较；它只验证物理 top-6 合同，并把结果绑定到后续
    /// dynamic page plan。删除 manifest 比对本身不构成动态路由，continuation 仍要求
    /// 完整36个物理Range的 materialization 证据。
    pub fn complete_router_probe_after_wait(
        &mut self,
        layer: u8,
    ) -> Result<S14Position0RouterProbeReceipt> {
        if self.pending_router_probe_layer != Some(layer) || self.pending_online_route.is_some() {
            bail!("position0 router probe readback phase/layer 漂移");
        }
        let sticky = unsafe { *(self.candidate.sticky_status.mapped() as *const u32) };
        if sticky != 0 {
            bail!("L{layer} router probe sticky status 非零: 0x{sticky:08x}");
        }
        let gpu = self.gpu.as_ref().expect("paged GPU owner");
        let raw_ids = unsafe {
            std::slice::from_raw_parts(
                gpu.readback()
                    .mapped()
                    .add(READBACK_ROUTE_IDS_OFFSET as usize) as *const u32,
                EXPERTS_PER_TOKEN,
            )
        };
        let raw_weights = unsafe {
            std::slice::from_raw_parts(
                gpu.readback()
                    .mapped()
                    .add(READBACK_ROUTE_WEIGHTS_OFFSET as usize) as *const f32,
                EXPERTS_PER_TOKEN,
            )
        };
        let mut expert_ids = [0u16; EXPERTS_PER_TOKEN];
        let mut route_weights = [0f32; EXPERTS_PER_TOKEN];
        let mut seen = BTreeSet::new();
        for slot in 0..EXPERTS_PER_TOKEN {
            let expert = u16::try_from(raw_ids[slot])
                .map_err(|_| anyhow!("L{layer} router probe expert id 不能表示为u16"))?;
            if expert >= N_ROUTED_EXPERTS || !seen.insert(expert) {
                bail!("L{layer} router probe top-6 id 越界或重复: {raw_ids:?}");
            }
            let weight = raw_weights[slot];
            if !weight.is_finite() || weight < 0.0 {
                bail!("L{layer} router probe route weight 非法: {raw_weights:?}");
            }
            expert_ids[slot] = expert;
            route_weights[slot] = weight;
        }
        let route = OnlineTop6 {
            layer,
            position: u64::from(self.candidate.committed_host_state.position),
            expert_ids,
            route_weights,
        };
        self.pending_online_route = Some(route.clone());
        Ok(S14Position0RouterProbeReceipt {
            route,
            readback_bytes: (EXPERTS_PER_TOKEN
                * (std::mem::size_of::<u32>() + std::mem::size_of::<f32>()))
                as u64,
        })
    }

    /// 将已物化的online top-6写入当前rolling bank的host staging，并把copy录入
    /// whole-token timeline使用的transfer command。这里不submit、不wait；matching
    /// continuation必须等待外部timeline返回的transfer ticket。
    ///
    /// # Safety
    ///
    /// `transfer_command` 必须尚未begin，且全部引用资源存活到timeline完成或drain。
    pub unsafe fn record_dynamic_routed_after_probe(
        &mut self,
        plan: &S14Position0SynchronousLayerPlan,
        dynamic_plan: &DynamicRoutedPagePlan,
        materialized: &MaterializedDynamicRoutedPagePlan,
        transfer_command: vk::CommandBuffer,
    ) -> Result<S14Position0RoutedUploadReceipt> {
        let observed = self
            .pending_online_route
            .as_ref()
            .ok_or_else(|| anyhow!("position0 dynamic routed upload 缺少router probe"))?;
        validate_dynamic_routed_continuation(plan, observed, dynamic_plan, materialized)?;
        if self.pending_dynamic_routed_bank.is_some() {
            bail!("position0 dynamic routed bank 已发布，禁止重复上传");
        }
        let dynamic_upload = S14DynamicRoutedUploadPlan::build(dynamic_plan, materialized)?;
        let receipt = self.resources.uploader.record_next_dynamic_routed_layer(
            self.ctx,
            transfer_command,
            self.manifest,
            self.weights,
            self.arena,
            dynamic_plan,
            &dynamic_upload,
        )?;
        if receipt.layer != plan.layer || receipt.bank != plan.routed_bank {
            bail!("position0 dynamic routed upload receipt layer/bank 漂移");
        }
        // continuation 不得再读 manifest 的专家身份/布局。一次性用同一
        // canonical packing plan 发布 ragged offsets、online IDs 与 route weights，
        // 保证 descriptor 视图与 rolling bank 的物理字节完全同源。
        let immutable = dynamic_upload.layout.immutable_bytes();
        let gpu = self.gpu.as_ref().expect("paged GPU owner");
        unsafe {
            gpu.immutable()
                .write_at(AUX_METADATA_OFFSET as usize, &immutable.ragged_metadata_le);
            gpu.immutable()
                .write_at(AUX_PHYSICAL_IDS_OFFSET as usize, &immutable.route_ids_le);
            gpu.immutable().write_at(
                AUX_NUMERIC_ROUTE_OVERRIDE_OFFSET as usize,
                &immutable.route_weights_le,
            );
        }
        self.pending_dynamic_routed_bank = Some((receipt.layer, receipt.bank));
        Ok(receipt)
    }

    /// 在 matching online route 的36个物理Range已经从 cache proof 物化后，录制
    /// QDQ→routed top-6→shared→FFN HC-post→state writeback 后半段。调用方负责在
    /// command 提交前把 `materialized` 按 graph 固定slot布局上传到 `plan.routed_bank`；
    /// 本入口以完整tensor/expert/bytes身份再次绑定该事实，禁止复用manifest专家页。
    ///
    /// # Safety
    /// `command` 必须尚未 begin；probe command 必须已完成，dynamic routed bank 的
    /// payload copy 必须在该 command 执行前完成并通过队列依赖可见。
    pub unsafe fn record_paged_layer_dynamic_moe_continuation(
        &mut self,
        plan: &S14Position0SynchronousLayerPlan,
        dynamic_plan: &DynamicRoutedPagePlan,
        materialized: &MaterializedDynamicRoutedPagePlan,
        command: vk::CommandBuffer,
    ) -> Result<()> {
        let observed = self
            .pending_online_route
            .as_ref()
            .ok_or_else(|| anyhow!("position0 dynamic MoE 缺少已完成的 router probe"))?;
        validate_dynamic_routed_continuation(plan, observed, dynamic_plan, materialized)?;
        if self.pending_router_probe_layer != Some(plan.layer) {
            bail!("position0 dynamic MoE probe layer 漂移");
        }
        if self.pending_dynamic_routed_bank != Some((plan.layer, plan.routed_bank)) {
            bail!("position0 dynamic MoE 缺少已验证的rolling bank publication");
        }
        let index = usize::from(plan.layer);
        let graph = self
            .resources
            .graphs
            .get(index)
            .ok_or_else(|| anyhow!("position0 dynamic MoE graph 缺少 L{}", plan.layer))?;
        let recipe = self
            .state_recording
            .layer(plan.layer)
            .ok_or_else(|| anyhow!("position0 dynamic MoE state recipe 缺少 L{}", plan.layer))?;
        let gpu = self.gpu.as_mut().expect("paged GPU owner");
        if gpu.active_paged_immutable != Some(index) {
            bail!("position0 dynamic MoE continuation 与 probe immutable 快照漂移");
        }
        let mut binders = record_layer_command_on(
            gpu,
            graph,
            &self.candidate,
            Some(recipe),
            self.input_execution.as_ref(),
            command,
            false,
            S14LayerCommandSegment::DynamicMoeContinuation,
        )?;
        gpu.binders.append(&mut binders);
        self.pending_router_probe_layer = None;
        self.pending_online_route = None;
        self.pending_dynamic_routed_bank = None;
        self.current_layer = Some(plan.layer);
        Ok(())
    }

    pub fn validate_recorded_layer_binding(
        &self,
        plan: &S14Position0SynchronousLayerPlan,
        timeline_bank: usize,
    ) -> Result<()> {
        if self.current_layer != Some(plan.layer)
            || plan.routed_bank != timeline_bank
            || usize::from(plan.layer) % 2 != timeline_bank
        {
            bail!("position0 paged recorded layer/bank 漂移");
        }
        Ok(())
    }

    /// 录制下一块真实 head staging→device copy，不 submit、不 wait。
    ///
    /// # Safety
    /// `command` 必须尚未 begin，双 bank 资源活到 terminal timeline 完成。
    pub unsafe fn record_next_head_transfer(
        &mut self,
        command: vk::CommandBuffer,
    ) -> Result<S14Position0HeadChunkReceipt> {
        self.ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let receipt = self.resources.uploader.record_next_head_chunk_copy(
            self.ctx,
            command,
            self.manifest,
            self.weights,
            self.arena,
        );
        match receipt {
            Ok(receipt) => {
                self.ctx.device.end_command_buffer(command)?;
                Ok(receipt)
            }
            Err(error) => Err(error).context("record position0 paged head transfer"),
        }
    }

    pub fn stage_recorded_head(
        &mut self,
        receipt: S14Position0HeadChunkReceipt,
        timeline_bank: usize,
    ) -> Result<S14Position0HeadChunkReceipt> {
        self.resources.uploader.stage_recorded_head_chunk(
            self.manifest,
            self.weights,
            &mut self.resources.store,
            receipt,
            timeline_bank,
        )
    }

    /// 同步首 token 终端链按顺序取真实 head chunk。该入口沿用 uploader 的
    /// verified mapping 与 fence wait，只用于首枚 token；异步生产路径使用双 bank
    /// stager，不能把这里的32次 transfer wait 计作最终性能。
    pub fn upload_next_head_chunk(&mut self) -> Result<S14Position0HeadChunkReceipt> {
        self.resources.uploader.upload_next_head_chunk(
            self.ctx,
            self.manifest,
            self.weights,
            &mut self.resources.store,
            self.arena,
        )
    }

    /// 只允许在 pager 已等待 L42 后读取；它是整43层唯一的 hidden D2H 诊断边界，
    /// 不参与下一层或 final head 计算。
    pub fn final_hidden_bf16(&self) -> Result<Vec<u16>> {
        if self.current_layer != Some(42) {
            bail!("position0 synchronous final hidden 只能在 L42 后读取");
        }
        let gpu = self.gpu.as_ref().expect("synchronous GPU owner");
        if gpu.last_compute_value != 0 || !gpu.binders.is_empty() {
            bail!("position0 synchronous final hidden 读取早于 L42 wait");
        }
        let words = unsafe {
            std::slice::from_raw_parts(
                gpu.readback().mapped().add(READBACK_HIDDEN_OFFSET as usize) as *const u16,
                (4 * HIDDEN) as usize,
            )
        };
        if words
            .iter()
            .any(|bits| !f32::from_bits(u32::from(*bits) << 16).is_finite())
        {
            bail!("position0 synchronous L42 hidden 含 NaN/Inf");
        }
        Ok(words.to_vec())
    }

    pub fn sticky_status(&self) -> u32 {
        unsafe { *(self.candidate.sticky_status.mapped() as *const u32) }
    }

    /// 同步/错误路径先由GPU owner执行device-idle收敛，再重置persistent游标。
    pub fn abort(mut self) -> Result<()> {
        if let Some(gpu) = self.gpu.take() {
            drop(gpu);
        }
        self.resources.abort_token_after_drain()
    }

    /// 同步成功路径完成当前persistent token；GPU owner仍以device-idle保证内部
    /// command已收敛，uploader/store/graph资源不销毁。
    pub fn finish(mut self) -> Result<S14Position0PersistentHostStepTelemetry> {
        if let Some(gpu) = self.gpu.take() {
            drop(gpu);
        }
        match self.resources.finish_token(self.weights) {
            Ok(()) => Ok(self.persistent_step),
            Err(failure) => match self.resources.abort_token_after_drain() {
                Ok(()) => Err(failure),
                Err(cleanup) => Err(anyhow!(
                    "{failure:#}; persistent host token abort 同时失败: {cleanup:#}"
                )),
            },
        }
    }

    /// 共享外部 timeline 已完成最终 wait后的成功快路。此时descriptor与pipeline
    /// 不再被GPU command引用，不需要额外device-wide idle；persistent资源只完成游标。
    pub fn finish_after_external_timeline_drained(
        mut self,
    ) -> Result<S14Position0PersistentHostStepTelemetry> {
        if let Some(mut gpu) = self.gpu.take() {
            gpu.external_timeline_drained = true;
            drop(gpu);
        }
        match self.resources.finish_token(self.weights) {
            Ok(()) => Ok(self.persistent_step),
            Err(failure) => match self.resources.abort_token_after_drain() {
                Ok(()) => Err(failure),
                Err(cleanup) => Err(anyhow!(
                    "{failure:#}; persistent host token abort 同时失败: {cleanup:#}"
                )),
            },
        }
    }

    /// 共享外部 timeline 已drain后的失败快路。candidate-scoped owner销毁后只
    /// 重置token游标；跨token staging/fence/mmap/graph保持常驻。
    pub fn abort_after_external_timeline_drained(mut self) -> Result<()> {
        if let Some(mut gpu) = self.gpu.take() {
            gpu.external_timeline_drained = true;
            drop(gpu);
        }
        self.resources.abort_token_after_drain()
    }
}

fn validate_production_layer_position(position: u32) -> Result<()> {
    if position > 2051 {
        bail!("production layer backend 当前只闭合首个ratio4分页边界position2051；position{position} 仍 fail-closed");
    }
    Ok(())
}

fn validate_dynamic_routed_continuation(
    layer_plan: &S14Position0SynchronousLayerPlan,
    observed: &OnlineTop6,
    dynamic_plan: &DynamicRoutedPagePlan,
    materialized: &MaterializedDynamicRoutedPagePlan,
) -> Result<()> {
    if observed.layer != layer_plan.layer
        || dynamic_plan.layer != layer_plan.layer
        || observed.position != dynamic_plan.position
        || observed.expert_ids != dynamic_plan.expert_ids
        || observed
            .route_weights
            .iter()
            .map(|weight| weight.to_bits())
            .ne(dynamic_plan
                .route_weights
                .iter()
                .map(|weight| weight.to_bits()))
    {
        bail!("dynamic routed continuation 的 probe/plan identity 漂移");
    }
    if materialized.assets.len() != DYNAMIC_ROUTED_RANGE_COUNT
        || materialized.mapped_assets.len() != DYNAMIC_ROUTED_RANGE_COUNT
        || dynamic_plan.pages.len() * 2 != DYNAMIC_ROUTED_RANGE_COUNT
    {
        bail!("dynamic routed continuation 不是完整36个物理Range");
    }
    for (physical_index, (asset, mapped)) in materialized
        .assets
        .iter()
        .zip(&materialized.mapped_assets)
        .enumerate()
    {
        let page = &dynamic_plan.pages[physical_index / 2];
        let range = if physical_index % 2 == 0 {
            &page.weight
        } else {
            &page.scale
        };
        if page.route_slot >= EXPERTS_PER_TOKEN
            || page.expert_id != observed.expert_ids[page.route_slot]
            || asset.tensor != range.tensor
            || asset.expert_id != Some(page.expert_id)
            || asset.bytes != range.bytes
            || mapped.tensor() != range.tensor
            || mapped.bytes().len() as u64 != range.bytes
            || asset.sha256 != mapped.expected_sha256()
            || !asset.payload_rehashed_by_builder
        {
            bail!("dynamic routed continuation physical Range #{physical_index} identity 漂移");
        }
    }
    Ok(())
}

impl S14Position0SynchronousLayerBackend for S14Position0SynchronousVulkanLayerAdapter<'_, '_> {
    type ComputeTicket = u64;

    fn wait_compute(&mut self, layer: u8, ticket: Self::ComputeTicket) -> Result<()> {
        if self.current_layer != Some(layer) {
            bail!("position0 synchronous wait layer 漂移");
        }
        let gpu = self.gpu.as_mut().expect("synchronous GPU owner");
        if gpu.last_compute_value != ticket || ticket == 0 {
            bail!("L{layer} synchronous compute ticket 漂移");
        }
        gpu.timeline
            .as_ref()
            .expect("synchronous timeline")
            .wait_compute(self.ctx, ticket, u64::MAX)?;
        let sticky = unsafe { *(self.candidate.sticky_status.mapped() as *const u32) };
        if sticky != 0 {
            bail!("L{layer} synchronous sticky status 非零: 0x{sticky:08x}");
        }
        let actual_ids = unsafe {
            std::slice::from_raw_parts(
                gpu.readback()
                    .mapped()
                    .add(READBACK_ROUTE_IDS_OFFSET as usize) as *const u32,
                EXPERTS_PER_TOKEN,
            )
        };
        let expected_ids = self.resources.graphs[usize::from(layer)]
            .routed_expert_ids
            .map(u32::from);
        if !gpu.reference_route_replay && actual_ids != expected_ids {
            let mut actual_set = actual_ids.to_vec();
            let mut expected_set = expected_ids.to_vec();
            actual_set.sort_unstable();
            expected_set.sort_unstable();
            if actual_set != expected_set {
                bail!(
                    "L{layer} synchronous route expert 集合漂移: actual={actual_ids:?} expected={expected_ids:?}"
                );
            }
        }
        for binder in gpu.binders.drain(..) {
            binder.destroy(self.ctx);
        }
        gpu.last_compute_value = 0;
        Ok(())
    }

    fn upload_weights(
        &mut self,
        request: crate::s14_position0_synchronous_layer_pager::S14Position0LayerUploadRequest<'_>,
    ) -> Result<S14Position0LayerUploadReceipt> {
        let uploader = &mut self.resources.uploader;
        let before = uploader.stats().transfer_submits;
        let static_receipt = uploader.prepare_next_static_layer(
            self.ctx,
            self.manifest,
            self.weights,
            &mut self.resources.store,
            self.arena,
        )?;
        let routed_receipt = uploader.upload_next_routed_layer(
            self.ctx,
            self.manifest,
            self.weights,
            &mut self.resources.store,
            self.arena,
        )?;
        if static_receipt.layer != request.layer || routed_receipt.layer != request.layer {
            bail!("position0 synchronous upload layer receipt 漂移");
        }
        let waits = uploader
            .stats()
            .transfer_submits
            .checked_sub(before)
            .ok_or_else(|| anyhow!("position0 synchronous transfer counter 倒退"))?;
        Ok(S14Position0LayerUploadReceipt {
            static_uploaded_bytes: static_receipt.bytes,
            routed_uploaded_bytes: routed_receipt.bytes,
            host_wait_calls: u32::try_from(waits)?,
        })
    }

    fn reconfigure_layer(&mut self, plan: &S14Position0SynchronousLayerPlan) -> Result<()> {
        let index = usize::from(plan.layer);
        let graph = self
            .resources
            .graphs
            .get(index)
            .ok_or_else(|| anyhow!("position0 synchronous graph 缺少 L{}", plan.layer))?;
        if plan.layer == 0 {
            if self.current_layer.is_some() {
                bail!("position0 synchronous L0 重复 reconfigure");
            }
            // L0 在 external owner 构造时已经绑定；embedding 与 L0 compute 在同一队列
            // 顺序执行，不能为无变化的页销毁仍在飞行的 embedding descriptor。
            self.current_layer = Some(0);
            return Ok(());
        }
        let static_buffer = match self.arena.static_layer(plan.layer)? {
            S14Position0StaticLayerBinding::Resident { buffer, .. }
            | S14Position0StaticLayerBinding::Streamed { buffer, .. } => buffer,
        };
        let gpu = self.gpu.as_mut().expect("synchronous GPU owner");
        gpu.attach_external_layer_buffers(
            static_buffer,
            self.arena.routed(plan.routed_bank)?,
            self.arena.workspace(),
            graph,
        )?;
        self.current_layer = Some(plan.layer);
        Ok(())
    }

    fn submit_layer(&mut self, layer: u8) -> Result<Self::ComputeTicket> {
        if !self.embedding_submitted || self.current_layer != Some(layer) {
            bail!("L{layer} synchronous submit phase 漂移");
        }
        let index = usize::from(layer);
        let graph = &self.resources.graphs[index];
        let recipe = self
            .state_recording
            .layer(layer)
            .ok_or_else(|| anyhow!("position0 state recipe 缺少 L{layer}"))?;
        let gpu = self.gpu.as_mut().expect("synchronous GPU owner");
        let mut binders = record_layer_command(
            gpu,
            graph,
            &self.candidate,
            Some(recipe),
            self.input_execution.as_ref(),
        )?;
        let value = unsafe {
            gpu.timeline
                .as_mut()
                .expect("synchronous timeline")
                .submit_compute_only(self.ctx, gpu.layer_command)?
        };
        gpu.last_compute_value = value;
        gpu.binders.append(&mut binders);
        Ok(value)
    }
}

fn workspace_offset(graph: &S14Position0L0GraphPlan, slot: S14Position0WorkspaceSlot) -> u64 {
    graph.workspace.region(slot).offset
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct S14Ratio4PagedWorkspace {
    page_scores: u64,
    page_indices: u64,
    global_score_banks: [u64; 2],
    global_index_banks: [u64; 2],
    page_table: u64,
    packed_main: u64,
}

fn ratio4_paged_workspace(graph: &S14Position0L0GraphPlan) -> Result<S14Ratio4PagedWorkspace> {
    const WORDS_512_BYTES: u64 = 512 * 4;
    const PAGE_TABLE_BYTES: u64 = 2 * 2 * 4;
    const PACKED_MAIN_BYTES: u64 = 512 * 512 * 2;

    fn reserve(cursor: &mut u64, bytes: u64) -> Result<u64> {
        let start = cursor
            .checked_add(255)
            .map(|value| value & !255)
            .ok_or_else(|| anyhow!("ratio4 paged workspace alignment overflow"))?;
        *cursor = start
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("ratio4 paged workspace range overflow"))?;
        Ok(start)
    }

    let mut cursor = graph.workspace.used_bytes();
    let page_scores = reserve(&mut cursor, WORDS_512_BYTES)?;
    let page_indices = reserve(&mut cursor, WORDS_512_BYTES)?;
    let global_score_banks = [
        reserve(&mut cursor, WORDS_512_BYTES)?,
        reserve(&mut cursor, WORDS_512_BYTES)?,
    ];
    let global_index_banks = [
        reserve(&mut cursor, WORDS_512_BYTES)?,
        reserve(&mut cursor, WORDS_512_BYTES)?,
    ];
    let page_table = reserve(&mut cursor, PAGE_TABLE_BYTES)?;
    let packed_main = reserve(&mut cursor, PACKED_MAIN_BYTES)?;
    if cursor > graph.workspace.capacity_bytes() {
        bail!(
            "L{} ratio4 paged workspace不足: end={cursor} capacity={}",
            graph.layer,
            graph.workspace.capacity_bytes()
        );
    }
    Ok(S14Ratio4PagedWorkspace {
        page_scores,
        page_indices,
        global_score_banks,
        global_index_banks,
        page_table,
        packed_main,
    })
}

unsafe fn l0_compute_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
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

unsafe fn compute_to_transfer_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

unsafe fn compute_read_write_to_transfer_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

unsafe fn transfer_to_transfer_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum S14ProductionAttentionMode {
    Position0,
    PreCompressionWindow {
        previous_kv_offset: u64,
        previous_count: u32,
        rope_offset: u64,
    },
    Ratio4FirstCompressedBlock {
        previous_kv_offset: u64,
        compressed_kv_offset: u64,
        rope_offset: u64,
    },
    Ratio4Sparse {
        previous_kv_offset: u64,
        window_start: u32,
        previous_count: u32,
        compressed_kv_offset: u64,
        compressed_count: u32,
        rope_offset: u64,
    },
    Ratio4PagedSparse {
        previous_kv_offset: u64,
        window_start: u32,
        previous_count: u32,
        compressed_kv_offset: u64,
        logical_compressed_count: u32,
        selected_count: u32,
        rope_offset: u64,
    },
    RingWindow {
        previous_kv_offset: u64,
        window_start: u32,
        previous_count: u32,
        rope_offset: u64,
    },
    Ratio128Deterministic {
        previous_kv_offset: u64,
        window_start: u32,
        previous_count: u32,
        compressed_kv_offset: u64,
        compressed_count: u32,
        rope_offset: u64,
    },
}

#[allow(clippy::too_many_arguments)]
fn production_attention_mode(
    position: u32,
    compress_ratio: u16,
    kv_offset: u64,
    kv_bytes: u64,
    arena_bytes: u64,
    state_recipe: Option<&S14Position0LayerStateRecordingRecipe>,
    position_execution: Option<&S14PositionExecutionPlan>,
) -> Result<S14ProductionAttentionMode> {
    let row0_end = kv_offset
        .checked_add(1024)
        .ok_or_else(|| anyhow!("attention committed window row0 overflow"))?;
    if kv_bytes < 1024 || row0_end > arena_bytes {
        bail!("attention window KV row0 range 漂移");
    }

    match position {
        0 => {
            if position_execution.is_some_and(|execution| {
                execution.rope_position != 0
                    || execution.window_slot != 0
                    || execution.active_window_tokens != 1
            }) {
                bail!("position0 attention RoPE/window 合同漂移");
            }
            if let Some(recipe) = state_recipe {
                if recipe.position != 0
                    || recipe.committed_window_state_range != (kv_offset..kv_offset)
                    || recipe.window_kv_state_range != (kv_offset..row0_end)
                {
                    bail!("position0 attention/state recipe 行合同漂移");
                }
            }
            Ok(S14ProductionAttentionMode::Position0)
        }
        1..=3 => {
            let execution = position_execution
                .ok_or_else(|| anyhow!("window attention 缺少已验证 position execution plan"))?;
            if execution.rope_position != position
                || execution.window_slot != position
                || execution.active_window_tokens != position + 1
            {
                bail!("position{position} attention RoPE/window 合同漂移");
            }
            let previous_end = kv_offset
                .checked_add(u64::from(position) * 1024)
                .ok_or_else(|| anyhow!("attention committed window prefix overflow"))?;
            let candidate_end = previous_end
                .checked_add(1024)
                .ok_or_else(|| anyhow!("attention candidate window row overflow"))?;
            if kv_bytes < u64::from(position + 1) * 1024 || candidate_end > arena_bytes {
                bail!("position{position} attention window KV prefix/current range漂移");
            }
            let recipe = state_recipe
                .ok_or_else(|| anyhow!("window attention 缺少 state recording recipe"))?;
            if recipe.position != position
                || recipe.committed_window_state_range != (kv_offset..previous_end)
                || recipe.window_kv_state_range != (previous_end..candidate_end)
            {
                bail!("position{position} attention committed prefix/candidate row 合同漂移");
            }
            let rope_offset = match compress_ratio {
                0 => POSITION1_ROPE_RATIO0_OFFSET,
                4 | 128 => POSITION1_ROPE_YARN_OFFSET,
                ratio => bail!("position{position} attention 未注册 compress ratio {ratio}"),
            };
            if position == 3 && compress_ratio == 4 {
                let compressed_kv_offset = kv_offset
                    .checked_add(128 * 512 * 2)
                    .ok_or_else(|| anyhow!("position3 ratio4 compressed KV offset overflow"))?;
                let compressed_end = compressed_kv_offset
                    .checked_add(512 * 2)
                    .ok_or_else(|| anyhow!("position3 ratio4 compressed KV range overflow"))?;
                let kv_end = kv_offset
                    .checked_add(kv_bytes)
                    .ok_or_else(|| anyhow!("position3 ratio4 KV cache range overflow"))?;
                if compressed_end > kv_end || compressed_end > arena_bytes {
                    bail!("position3 ratio4 first compressed KV row range漂移");
                }
                Ok(S14ProductionAttentionMode::Ratio4FirstCompressedBlock {
                    previous_kv_offset: kv_offset,
                    compressed_kv_offset,
                    rope_offset,
                })
            } else {
                Ok(S14ProductionAttentionMode::PreCompressionWindow {
                    previous_kv_offset: kv_offset,
                    previous_count: position,
                    rope_offset,
                })
            }
        }
        4..=2051 => {
            let execution = position_execution
                .ok_or_else(|| anyhow!("position{position} attention 缺少已验证 execution plan"))?;
            if execution.rope_position != position
                || execution.window_slot != position % 128
                || execution.active_window_tokens != (position + 1).min(128)
            {
                bail!("position{position} attention RoPE/window 合同漂移");
            }
            let previous_count = position.min(127);
            let window_start = if position < 128 {
                0
            } else {
                (position + 1) % 128
            };
            let committed_rows = position.min(128);
            let committed_end = kv_offset
                .checked_add(u64::from(committed_rows) * 1024)
                .ok_or_else(|| anyhow!("position{position} committed window range overflow"))?;
            let candidate_row = u64::from(position % 128);
            let candidate_start = kv_offset
                .checked_add(candidate_row * 1024)
                .ok_or_else(|| anyhow!("position{position} candidate window offset overflow"))?;
            let candidate_end = candidate_start
                .checked_add(1024)
                .ok_or_else(|| anyhow!("position{position} candidate window range overflow"))?;
            let kv_end = kv_offset
                .checked_add(kv_bytes)
                .ok_or_else(|| anyhow!("position{position} KV cache range overflow"))?;
            if committed_end > kv_end || candidate_end > kv_end || candidate_end > arena_bytes {
                bail!("position{position} attention window KV range 漂移");
            }
            let recipe = state_recipe
                .ok_or_else(|| anyhow!("position{position} attention 缺少 state recipe"))?;
            if recipe.position != position
                || recipe.committed_window_state_range != (kv_offset..committed_end)
                || recipe.window_kv_state_range != (candidate_start..candidate_end)
            {
                bail!("position{position} attention committed/candidate window 合同漂移");
            }
            let rope_offset = match compress_ratio {
                0 => POSITION1_ROPE_RATIO0_OFFSET,
                4 | 128 => POSITION1_ROPE_YARN_OFFSET,
                ratio => bail!("position{position} attention 未注册 compress ratio {ratio}"),
            };
            if compress_ratio == 4 {
                let compressed_count = (position + 1) / 4;
                let compressed_kv_offset =
                    kv_offset.checked_add(128 * 512 * 2).ok_or_else(|| {
                        anyhow!("position{position} ratio4 compressed KV offset overflow")
                    })?;
                let compressed_end = compressed_kv_offset
                    .checked_add(u64::from(compressed_count) * 512 * 2)
                    .ok_or_else(|| {
                        anyhow!("position{position} ratio4 compressed KV range overflow")
                    })?;
                if compressed_count == 0 || compressed_end > kv_end || compressed_end > arena_bytes
                {
                    bail!("position{position} ratio4 compressed KV count/range 漂移");
                }
                if compressed_count <= 512 {
                    Ok(S14ProductionAttentionMode::Ratio4Sparse {
                        previous_kv_offset: kv_offset,
                        window_start,
                        previous_count,
                        compressed_kv_offset,
                        compressed_count,
                        rope_offset,
                    })
                } else if position == 2051 && compressed_count == 513 {
                    Ok(S14ProductionAttentionMode::Ratio4PagedSparse {
                        previous_kv_offset: kv_offset,
                        window_start,
                        previous_count,
                        compressed_kv_offset,
                        logical_compressed_count: compressed_count,
                        selected_count: 512,
                        rope_offset,
                    })
                } else {
                    bail!("position{position} ratio4 paged attention 尚未闭合")
                }
            } else if compress_ratio == 128 && position >= 127 {
                let compressed_count = (position + 1) / 128;
                let compressed_kv_offset =
                    kv_offset.checked_add(128 * 512 * 2).ok_or_else(|| {
                        anyhow!("position{position} ratio128 compressed KV offset overflow")
                    })?;
                let compressed_end = compressed_kv_offset
                    .checked_add(u64::from(compressed_count) * 512 * 2)
                    .ok_or_else(|| {
                        anyhow!("position{position} ratio128 compressed KV range overflow")
                    })?;
                if compressed_count != (position + 1) / 128
                    || compressed_end > kv_end
                    || compressed_end > arena_bytes
                {
                    bail!("position{position} ratio128 compressed KV count/range 漂移");
                }
                Ok(S14ProductionAttentionMode::Ratio128Deterministic {
                    previous_kv_offset: kv_offset,
                    window_start,
                    previous_count,
                    compressed_kv_offset,
                    compressed_count,
                    rope_offset,
                })
            } else if position >= 128 && compress_ratio == 0 {
                Ok(S14ProductionAttentionMode::RingWindow {
                    previous_kv_offset: kv_offset,
                    window_start,
                    previous_count,
                    rope_offset,
                })
            } else {
                Ok(S14ProductionAttentionMode::PreCompressionWindow {
                    previous_kv_offset: kv_offset,
                    previous_count,
                    rope_offset,
                })
            }
        }
        _ => {
            bail!("production layer attention position{position} 尚未闭合后续环形窗口")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct S14Ratio4BoundaryStateBindings {
    main_kv_state: u64,
    main_score_state: u64,
    indexer_kv_state: u64,
    indexer_score_state: u64,
    main_compressed_kv: u64,
    indexer_compressed_kv: u64,
}

fn ratio4_boundary_state_bindings(
    state: &NativeState,
    layer: u8,
    recipe: &S14Position0LayerStateRecordingRecipe,
) -> Result<S14Ratio4BoundaryStateBindings> {
    if state.position != recipe.position || recipe.layer != layer || recipe.compress_ratio != 4 {
        bail!(
            "L{layer} ratio4 finalize recipe/state position或ratio漂移：state={} recipe={}",
            state.position,
            recipe.position
        );
    }
    let history_publish = S14Ratio4HistoryPublishPlan::build(state, layer, recipe.position)?;
    let history_target = history_publish.appended_target.as_ref().ok_or_else(|| {
        anyhow!(
            "L{layer} position{} ratio4 finalize 缺少分页 target",
            recipe.position
        )
    })?;
    let kv = unique_native_entry(
        state.kv.iter().filter(|entry| entry.layer == layer),
        &format!("L{layer} position{} KV", recipe.position),
    )?;
    let main = unique_native_entry(
        state
            .compressors
            .iter()
            .filter(|entry| entry.layer == layer && entry.compress_ratio == 4),
        &format!(
            "L{layer} position{} ratio4 main compressor",
            recipe.position
        ),
    )?;
    let indexer = unique_native_entry(
        state.indexers.iter().filter(|entry| entry.layer == layer),
        &format!("L{layer} position{} ratio4 indexer", recipe.position),
    )?;
    if kv.compress_ratio != 4
        || kv.cache.dtype != DType::Bf16
        || kv.cache.shape.len() != 3
        || kv.cache.shape[0] != 1
        || kv.cache.shape[2] != 512
        || main.kv_state.dtype != DType::F32
        || main.kv_state.shape != [1, 8, 1024]
        || main.score_state.dtype != DType::F32
        || main.score_state.shape != [1, 8, 1024]
        || indexer.compressor_kv_state.dtype != DType::F32
        || indexer.compressor_kv_state.shape != [1, 8, 256]
        || indexer.compressor_score_state.dtype != DType::F32
        || indexer.compressor_score_state.shape != [1, 8, 256]
        || indexer.kv_cache.dtype != DType::Bf16
        || indexer.kv_cache.shape.len() != 3
        || indexer.kv_cache.shape[0] != 1
        || indexer.kv_cache.shape[2] != 128
    {
        bail!(
            "L{layer} position{} ratio4 state dtype/shape 漂移",
            recipe.position
        );
    }
    let main_compressed_kv = history_target.main_state_range.start;
    let indexer_compressed_kv = history_target.indexer_state_range.start;
    let main_target = history_target.main_state_range.clone();
    let indexer_target = history_target.indexer_state_range.clone();
    let kv_end = kv
        .cache
        .offset
        .checked_add(kv.cache.bytes)
        .ok_or_else(|| anyhow!("L{layer} main cache range overflow"))?;
    let indexer_end = indexer
        .kv_cache
        .offset
        .checked_add(indexer.kv_cache.bytes)
        .ok_or_else(|| anyhow!("L{layer} indexer cache range overflow"))?;
    if main_target.end > kv_end
        || indexer_target.end > indexer_end
        || main_target.end > state.arena_bytes
        || indexer_target.end > state.arena_bytes
        || !recipe.state_ranges_written.contains(&main_target)
        || !recipe.state_ranges_written.contains(&indexer_target)
    {
        bail!(
            "L{layer} position{} ratio4 compressed target/dirty write-set 漂移",
            recipe.position
        );
    }
    Ok(S14Ratio4BoundaryStateBindings {
        main_kv_state: main.kv_state.offset,
        main_score_state: main.score_state.offset,
        indexer_kv_state: indexer.compressor_kv_state.offset,
        indexer_score_state: indexer.compressor_score_state.offset,
        main_compressed_kv,
        indexer_compressed_kv,
    })
}

fn ratio4_compressed_cache_rows(position: u32) -> Result<(u32, u32)> {
    let boundary = S14Ratio4CompressorBoundary::new(position)?;
    let main_row = 128u32
        .checked_add(boundary.cache_index)
        .ok_or_else(|| anyhow!("position{position} ratio4 main compressed row overflow"))?;
    Ok((main_row, boundary.cache_index))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct S14Ratio128BoundaryStateBindings {
    boundary: S14Ratio128CompressorBoundary,
    main_kv_state: u64,
    main_score_state: u64,
    main_compressed_kv: u64,
}

fn ratio128_boundary_state_bindings(
    state: &NativeState,
    layer: u8,
    recipe: &S14Position0LayerStateRecordingRecipe,
) -> Result<S14Ratio128BoundaryStateBindings> {
    if state.position != recipe.position || recipe.layer != layer || recipe.compress_ratio != 128 {
        bail!(
            "L{layer} ratio128 finalize recipe/state position或ratio漂移：state={} recipe={}",
            state.position,
            recipe.position
        );
    }
    let boundary = S14Ratio128CompressorBoundary::new(recipe.position)?;
    let compressed_row = 128u32.checked_add(boundary.cache_index).ok_or_else(|| {
        anyhow!(
            "L{layer} position{} ratio128 cache row overflow",
            recipe.position
        )
    })?;
    let kv = unique_native_entry(
        state.kv.iter().filter(|entry| entry.layer == layer),
        &format!("L{layer} position{} KV", recipe.position),
    )?;
    let main = unique_native_entry(
        state
            .compressors
            .iter()
            .filter(|entry| entry.layer == layer && entry.compress_ratio == 128),
        &format!(
            "L{layer} position{} ratio128 main compressor",
            recipe.position
        ),
    )?;
    if kv.compress_ratio != 128
        || kv.cache.dtype != DType::Bf16
        || kv.cache.shape.len() != 3
        || kv.cache.shape[0] != 1
        || kv.cache.shape[1] <= compressed_row
        || kv.cache.shape[2] != 512
        || main.kv_state.dtype != DType::F32
        || main.kv_state.shape != [1, 128, 512]
        || main.score_state.dtype != DType::F32
        || main.score_state.shape != [1, 128, 512]
    {
        bail!(
            "L{layer} position{} ratio128 state dtype/shape 漂移",
            recipe.position
        );
    }
    let main_compressed_kv = kv
        .cache
        .offset
        .checked_add(u64::from(compressed_row) * 512 * 2)
        .ok_or_else(|| anyhow!("L{layer} ratio128 compressed cache offset overflow"))?;
    let target_end = main_compressed_kv
        .checked_add(512 * 2)
        .ok_or_else(|| anyhow!("L{layer} ratio128 compressed target overflow"))?;
    let target = main_compressed_kv..target_end;
    let kv_end = kv
        .cache
        .offset
        .checked_add(kv.cache.bytes)
        .ok_or_else(|| anyhow!("L{layer} ratio128 cache range overflow"))?;
    if target.end > kv_end
        || target.end > state.arena_bytes
        || !recipe.state_ranges_written.contains(&target)
    {
        bail!(
            "L{layer} position{} ratio128 compressed target/dirty write-set 漂移",
            recipe.position
        );
    }
    Ok(S14Ratio128BoundaryStateBindings {
        boundary,
        main_kv_state: main.kv_state.offset,
        main_score_state: main.score_state.offset,
        main_compressed_kv,
    })
}

fn unique_native_entry<'a, T>(
    mut entries: impl Iterator<Item = &'a T>,
    label: &str,
) -> Result<&'a T> {
    let entry = entries.next().ok_or_else(|| anyhow!("{label} 缺失"))?;
    if entries.next().is_some() {
        bail!("{label} 重复");
    }
    Ok(entry)
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_ratio4_boundary_finalize(
    gpu: &S14Position0L0GpuOwner<'_>,
    graph: &S14Position0L0GraphPlan,
    candidate: &Position0GpuCandidate<'_>,
    recipe: &S14Position0LayerStateRecordingRecipe,
    command: vk::CommandBuffer,
    workspace: &GpuBuffer,
    scratch_bf16_a: u64,
    scratch_bf16_b: u64,
    scratch_bf16_c: u64,
    inverse_rms: u64,
) -> Result<Vec<DescriptorBinder>> {
    let ctx = gpu.ctx;
    let state = &candidate.committed_host_state.native;
    let bindings = ratio4_boundary_state_bindings(state, graph.layer, recipe)?;
    let pipelines = gpu
        .ratio4_finalize
        .as_ref()
        .expect("ratio4 finalize pipelines");
    let static_arena = gpu.static_arena();
    let status = StorageBufferSlice::whole(candidate.sticky_status);
    let mut binders = Vec::with_capacity(8);
    let result = (|| -> Result<()> {
        for (kind, kv_state, score_state, norm_suffix, output_offset) in [
            (
                S14Ratio4CompressorKind::Main,
                bindings.main_kv_state,
                bindings.main_score_state,
                "attn.compressor.norm.weight",
                bindings.main_compressed_kv,
            ),
            (
                S14Ratio4CompressorKind::Indexer,
                bindings.indexer_kv_state,
                bindings.indexer_score_state,
                "attn.indexer.compressor.norm.weight",
                bindings.indexer_compressed_kv,
            ),
        ] {
            let dispatch = pipelines.pool.bind_slices(
                ctx,
                kind,
                StorageBufferSlice {
                    buffer: candidate.candidate_state,
                    offset: kv_state,
                },
                StorageBufferSlice {
                    buffer: candidate.candidate_state,
                    offset: score_state,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch_bf16_a,
                },
                status,
            )?;
            pipelines.pool.cmd(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
            binders.push(dispatch.binder);

            let dispatch = pipelines.rmsnorm.bind_slices(
                ctx,
                kind.rmsnorm_shape()?,
                1.0e-6,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch_bf16_a,
                },
                StorageBufferSlice {
                    buffer: static_arena,
                    offset: graph.static_offset_suffix(norm_suffix)?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: inverse_rms,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch_bf16_b,
                },
                status,
            )?;
            pipelines.rmsnorm.cmd(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
            binders.push(dispatch.binder);

            let dispatch = pipelines.rope.bind_slices(
                ctx,
                kind,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch_bf16_b,
                },
                StorageBufferSlice {
                    buffer: gpu.ratio4_compressed_rope(),
                    offset: 0,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch_bf16_c,
                },
                status,
            )?;
            pipelines.rope.cmd(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
            binders.push(dispatch.binder);

            match kind {
                S14Ratio4CompressorKind::Main => {
                    let dispatch = pipelines.main_qdq_bf16.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: scratch_bf16_c,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: output_offset,
                        },
                        status,
                    )?;
                    pipelines.main_qdq_bf16.cmd(ctx, command, &dispatch);
                    l0_compute_barrier(ctx, command);
                    binders.push(dispatch.binder);
                }
                S14Ratio4CompressorKind::Indexer => {
                    let dispatch = pipelines.indexer_hadamard_qdq.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: scratch_bf16_c,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: output_offset,
                        },
                        status,
                    )?;
                    pipelines.indexer_hadamard_qdq.cmd(ctx, command, &dispatch);
                    l0_compute_barrier(ctx, command);
                    binders.push(dispatch.binder);
                }
            }
        }
        recipe.record_rollover(ctx, command, candidate.candidate_state)?;
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

#[allow(clippy::too_many_arguments)]
unsafe fn record_ratio128_boundary_finalize(
    gpu: &S14Position0L0GpuOwner<'_>,
    graph: &S14Position0L0GraphPlan,
    candidate: &Position0GpuCandidate<'_>,
    recipe: &S14Position0LayerStateRecordingRecipe,
    command: vk::CommandBuffer,
    workspace: &GpuBuffer,
    scratch_bf16_a: u64,
    scratch_bf16_b: u64,
    scratch_bf16_c: u64,
    inverse_rms: u64,
) -> Result<Vec<DescriptorBinder>> {
    let ctx = gpu.ctx;
    let bindings = ratio128_boundary_state_bindings(
        &candidate.committed_host_state.native,
        graph.layer,
        recipe,
    )?;
    let pipelines = gpu
        .ratio128_finalize
        .as_ref()
        .expect("ratio128 finalize pipelines");
    let static_arena = gpu.static_arena();
    let status = StorageBufferSlice::whole(candidate.sticky_status);
    let mut binders = Vec::with_capacity(4);
    let result = (|| -> Result<()> {
        let dispatch = pipelines.pool.bind_slices(
            ctx,
            StorageBufferSlice {
                buffer: candidate.candidate_state,
                offset: bindings.main_kv_state,
            },
            StorageBufferSlice {
                buffer: candidate.candidate_state,
                offset: bindings.main_score_state,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: scratch_bf16_a,
            },
            status,
        )?;
        pipelines.pool.cmd(ctx, command, &dispatch);
        l0_compute_barrier(ctx, command);
        binders.push(dispatch.binder);

        let dispatch = pipelines.rmsnorm.bind_slices(
            ctx,
            bindings.boundary.rmsnorm_shape()?,
            S14_RATIO128_RMS_EPSILON,
            StorageBufferSlice {
                buffer: workspace,
                offset: scratch_bf16_a,
            },
            StorageBufferSlice {
                buffer: static_arena,
                offset: graph.static_offset_suffix("attn.compressor.norm.weight")?,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: inverse_rms,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: scratch_bf16_b,
            },
            status,
        )?;
        pipelines.rmsnorm.cmd(ctx, command, &dispatch);
        l0_compute_barrier(ctx, command);
        binders.push(dispatch.binder);

        let dispatch = pipelines.rope.bind_slices(
            ctx,
            S14Ratio4CompressorKind::Main,
            StorageBufferSlice {
                buffer: workspace,
                offset: scratch_bf16_b,
            },
            StorageBufferSlice {
                buffer: gpu.ratio128_compressed_rope(),
                offset: 0,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: scratch_bf16_c,
            },
            status,
        )?;
        pipelines.rope.cmd(ctx, command, &dispatch);
        l0_compute_barrier(ctx, command);
        binders.push(dispatch.binder);

        // Main group64 QDQ 直接写 inactive candidate 的 ABI cache 行。sticky status
        // 在唯一 terminal fence 后统一验证；非零时 whole-token candidate 不得发布。
        let dispatch = pipelines.main_qdq_bf16.bind_slices(
            ctx,
            StorageBufferSlice {
                buffer: workspace,
                offset: scratch_bf16_c,
            },
            StorageBufferSlice {
                buffer: candidate.candidate_state,
                offset: bindings.main_compressed_kv,
            },
            status,
        )?;
        pipelines.main_qdq_bf16.cmd(ctx, command, &dispatch);
        l0_compute_barrier(ctx, command);
        binders.push(dispatch.binder);
        recipe.record_rollover(ctx, command, candidate.candidate_state)?;
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

fn record_layer_command(
    gpu: &mut S14Position0L0GpuOwner<'_>,
    graph: &S14Position0L0GraphPlan,
    candidate: &Position0GpuCandidate<'_>,
    state_recipe: Option<&S14Position0LayerStateRecordingRecipe>,
    position_execution: Option<&S14PositionExecutionPlan>,
) -> Result<Vec<DescriptorBinder>> {
    record_layer_command_on(
        gpu,
        graph,
        candidate,
        state_recipe,
        position_execution,
        gpu.layer_command,
        true,
        S14LayerCommandSegment::CompleteManifestReplay,
    )
}

fn record_layer_command_on(
    gpu: &mut S14Position0L0GpuOwner<'_>,
    graph: &S14Position0L0GraphPlan,
    candidate: &Position0GpuCandidate<'_>,
    state_recipe: Option<&S14Position0LayerStateRecordingRecipe>,
    position_execution: Option<&S14PositionExecutionPlan>,
    command: vk::CommandBuffer,
    reset_internal_pool: bool,
    segment: S14LayerCommandSegment,
) -> Result<Vec<DescriptorBinder>> {
    let ctx = gpu.ctx;
    let workspace = gpu.workspace();
    let static_arena = gpu.static_arena();
    let routed_arena = gpu.routed_arena();
    let immutable = gpu.immutable();
    let readback = gpu.readback();
    let numeric = gpu.numeric.as_ref().expect("L0 numeric pipeline");
    let numeric_exact = gpu.numeric_exact.as_ref().unwrap_or(numeric);
    let rmsnorm = gpu.rmsnorm.as_ref().expect("L0 RMSNorm pipeline");
    let q_head_normalize = gpu
        .q_head_normalize
        .as_ref()
        .expect("L0 Q-head normalize pipeline");
    let f32_to_bf16 = gpu.f32_to_bf16.as_ref().expect("L0 F32-to-BF16 pipeline");
    let bf16_to_f32 = gpu.bf16_to_f32.as_ref().expect("L0 BF16-to-F32 pipeline");
    let qdq = gpu.qdq.as_ref().expect("L0 QDQ pipeline");
    let hc_post = gpu.hc_post.as_ref().expect("L0 HC-post pipeline");
    let route = gpu.route.as_ref().expect("L0 route pipeline");
    let route_slot_align = gpu
        .route_slot_align
        .as_ref()
        .expect("L0 route slot-align pipeline");
    let workspace_bytes = S14_POSITION0_L0_WORKSPACE_BYTES;
    let mut binders = Vec::with_capacity(36);

    let kv = candidate
        .committed_host_state
        .native
        .kv
        .iter()
        .find(|state| state.layer == graph.layer)
        .ok_or_else(|| anyhow!("candidate state 缺少 L{} KV", graph.layer))?;
    if kv.cache.bytes < 1024
        || kv.cache.offset.checked_add(1024).is_none()
        || kv.cache.offset + 1024 > candidate.committed_host_state.native.arena_bytes
    {
        bail!("candidate L{} KV writeback range 漂移", graph.layer);
    }
    let layer_kv_offset = kv.cache.offset;
    let attention_mode = production_attention_mode(
        candidate.committed_host_state.position,
        kv.compress_ratio,
        kv.cache.offset,
        kv.cache.bytes,
        candidate.committed_host_state.native.arena_bytes,
        state_recipe,
        position_execution,
    )?;

    // Both command segments bind the same stable workspace offsets. The probe
    // publishes only router output; the continuation consumes these device
    // values without replaying attention or FFN HC-pre.
    let (hidden_input_slot, hidden_output_slot) = if graph.layer_index % 2 == 0 {
        (
            S14Position0WorkspaceSlot::HiddenStreamsA,
            S14Position0WorkspaceSlot::HiddenStreamsB,
        )
    } else {
        (
            S14Position0WorkspaceSlot::HiddenStreamsB,
            S14Position0WorkspaceSlot::HiddenStreamsA,
        )
    };
    let hidden_a = workspace_offset(graph, hidden_input_slot);
    let hidden_b = workspace_offset(graph, hidden_output_slot);
    let hc_branch_bf16 = workspace_offset(graph, S14Position0WorkspaceSlot::HcBranchBf16);
    let hc_branch_f32 = workspace_offset(graph, S14Position0WorkspaceSlot::HcBranchF32);
    let hc_aux = workspace_offset(graph, S14Position0WorkspaceSlot::HcAux);
    let scratch = workspace_offset(graph, S14Position0WorkspaceSlot::CompressorProjectionF32);
    let key_value_bf16 = workspace_offset(graph, S14Position0WorkspaceSlot::KeyValueBf16);
    let attention_branch_bf16 =
        workspace_offset(graph, S14Position0WorkspaceSlot::AttentionBranchBf16);
    let router_ids = workspace_offset(graph, S14Position0WorkspaceSlot::RouterIdsU32);
    let router_weights = workspace_offset(graph, S14Position0WorkspaceSlot::RouterWeightsF32);
    let router_selected =
        workspace_offset(graph, S14Position0WorkspaceSlot::RouterSelectedScoresF32);

    let result = (|| -> Result<()> {
        unsafe {
            if reset_internal_pool {
                ctx.device.reset_command_pool(
                    gpu.command_pool,
                    vk::CommandPoolResetFlags::RELEASE_RESOURCES,
                )?;
            }
            ctx.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            // 上一 submission 的 embedding 写入必须在本 command 首段对后续 shader 可见。
            l0_compute_barrier(ctx, command);
        }

        if segment.records_prefix() {
            let hc_shape = S14HcPreShape::new(HIDDEN)?;
            // 层边界按 A/B 交替；层内仍用另一槽保存 attention 残差，FFN HC-post
            // 完成后再做一次 32KiB device-local copy，把最终四流状态发布到下一层输入槽。
            let (hidden_input_slot, hidden_output_slot) = if graph.layer_index % 2 == 0 {
                (
                    S14Position0WorkspaceSlot::HiddenStreamsA,
                    S14Position0WorkspaceSlot::HiddenStreamsB,
                )
            } else {
                (
                    S14Position0WorkspaceSlot::HiddenStreamsB,
                    S14Position0WorkspaceSlot::HiddenStreamsA,
                )
            };
            let hidden_a = workspace_offset(graph, hidden_input_slot);
            let hidden_b = workspace_offset(graph, hidden_output_slot);
            let hc_norm = workspace_offset(graph, S14Position0WorkspaceSlot::HcNormalizedInput);
            let hc_branch_bf16 = workspace_offset(graph, S14Position0WorkspaceSlot::HcBranchBf16);
            let hc_branch_f32 = workspace_offset(graph, S14Position0WorkspaceSlot::HcBranchF32);
            let hc_aux = workspace_offset(graph, S14Position0WorkspaceSlot::HcAux);
            let hc_inverse = workspace_offset(graph, S14Position0WorkspaceSlot::HcInverseRms);
            let scratch =
                workspace_offset(graph, S14Position0WorkspaceSlot::CompressorProjectionF32);

            // 1. Attention HC-pre + RMSNorm。
            let dispatch = numeric.bind_hc_normalize_input_arena(
                ctx,
                hc_shape,
                NORM_EPS,
                workspace,
                workspace_bytes,
                hidden_a,
                hc_norm,
                hc_inverse,
            )?;
            unsafe {
                numeric.cmd_hc_normalize_input(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric.bind_f32_matvec_arenas(
                ctx,
                S14F32MatvecShape::new(24, HC_FLAT, 1)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("hc_attn_fn")?,
                workspace,
                workspace_bytes,
                hc_norm,
                workspace,
                workspace_bytes,
                scratch,
            )?;
            unsafe {
                numeric.cmd_f32_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric.bind_hc_split_reduce_norm_arenas(
                ctx,
                hc_shape,
                NORM_EPS,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("hc_attn_scale")?,
                graph.static_offset_suffix("hc_attn_base")?,
                graph.static_offset_suffix("attn_norm.weight")?,
                workspace,
                workspace_bytes,
                hidden_a,
                scratch,
                hc_branch_bf16,
                hc_branch_f32,
                hc_aux,
                hc_inverse,
            )?;
            unsafe {
                numeric.cmd_hc_split_reduce_norm(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);

            // compressor/indexer 状态必须使用与 Q/KV 相同的真实 HC branch。
            // 每个 ratio4 边界都必须严格保持 remainder -> finalize/write -> rollover；
            // ratio128 边界保持 remainder -> pool/norm/RoPE/QDQ -> inactive cache write。
            // 先 rollover 会覆盖 ratio4 当前块 pool 所需的 active half。
            if let Some(recipe) = state_recipe {
                let ratio4_boundary = recipe.compress_ratio == 4 && recipe.position % 4 == 3;
                let ratio128_boundary =
                    recipe.compress_ratio == 128 && recipe.position % 128 == 127;
                let mut state_binders = if ratio4_boundary || ratio128_boundary {
                    let mut remainder = unsafe {
                        recipe.record_compressor_remainder(
                            ctx,
                            command,
                            numeric,
                            gpu.ape_add.as_ref().expect("position0 APE add pipeline"),
                            static_arena,
                            workspace,
                            candidate.candidate_state,
                        )?
                    };
                    if ratio4_boundary {
                        let mut finalize = unsafe {
                            record_ratio4_boundary_finalize(
                                gpu,
                                graph,
                                candidate,
                                recipe,
                                command,
                                workspace,
                                workspace_offset(graph, S14Position0WorkspaceSlot::QueryLowBf16),
                                workspace_offset(graph, S14Position0WorkspaceSlot::KeyValueBf16),
                                workspace_offset(
                                    graph,
                                    S14Position0WorkspaceSlot::AttentionBranchBf16,
                                ),
                                hc_inverse,
                            )?
                        };
                        remainder.append(&mut finalize);
                    } else if ratio128_boundary {
                        let mut finalize = unsafe {
                            record_ratio128_boundary_finalize(
                                gpu,
                                graph,
                                candidate,
                                recipe,
                                command,
                                workspace,
                                workspace_offset(graph, S14Position0WorkspaceSlot::QueryLowBf16),
                                workspace_offset(graph, S14Position0WorkspaceSlot::KeyValueBf16),
                                workspace_offset(
                                    graph,
                                    S14Position0WorkspaceSlot::AttentionBranchBf16,
                                ),
                                hc_inverse,
                            )?
                        };
                        remainder.append(&mut finalize);
                    }
                    remainder
                } else {
                    unsafe {
                        recipe.record_compressor(
                            ctx,
                            command,
                            numeric,
                            gpu.ape_add.as_ref().expect("position0 APE add pipeline"),
                            static_arena,
                            workspace,
                            candidate.candidate_state,
                        )?
                    }
                };
                binders.append(&mut state_binders);
            }

            // ratio4 compressed indexer 的动态 head weight 必须从未量化的真实
            // attention branch 产生。先完成 [64,4096] BF16 投影，再让下方
            // branch QDQ 复用/overwrite HcBranchF32；不得用 QDQ 后的 branch 代替。
            let sparse_ratio4 = matches!(
                attention_mode,
                S14ProductionAttentionMode::Ratio4Sparse { .. }
                    | S14ProductionAttentionMode::Ratio4PagedSparse { .. }
            );
            let index_weights_f32 =
                workspace_offset(graph, S14Position0WorkspaceSlot::CompressorScoreF32);
            let index_weights_bf16 = index_weights_f32 + u64::from(S14_INDEX_HEADS) * 4;
            if sparse_ratio4 {
                let dispatch = numeric.bind_bf16_matvec_arenas(
                    ctx,
                    S14Bf16MatvecShape::new(64, HIDDEN, 1)?,
                    static_arena,
                    graph.static_logical_bytes,
                    graph.static_offset_suffix("attn.indexer.weights_proj.weight")?,
                    workspace,
                    workspace_bytes,
                    hc_branch_f32,
                    workspace,
                    workspace_bytes,
                    index_weights_f32,
                )?;
                unsafe {
                    numeric.cmd_bf16_matvec(ctx, command, &dispatch);
                    l0_compute_barrier(ctx, command);
                }
                binders.push(dispatch.binder);
                // 官方 F.linear(BF16, BF16) 在 scale 前先发布 BF16；scale 的第二次
                // BF16 RNE 由 sparse-indexer shader 完成，再转 F32 参与 score。
                let dispatch = f32_to_bf16.bind_slices(
                    ctx,
                    S14F32ToBf16Shape::new(S14_INDEX_HEADS)?,
                    StorageBufferSlice {
                        buffer: workspace,
                        offset: index_weights_f32,
                    },
                    StorageBufferSlice {
                        buffer: workspace,
                        offset: index_weights_bf16,
                    },
                    StorageBufferSlice::whole(candidate.sticky_status),
                )?;
                unsafe {
                    f32_to_bf16.cmd(ctx, command, &dispatch);
                    l0_compute_barrier(ctx, command);
                }
                binders.push(dispatch.binder);
            }

            // 2. Q/KV projection。attention 的多个 FP8 投影共享同一份原生 branch QDQ。
            let dispatch = qdq.bind_slices(
                ctx,
                S14E4m3QdqShape::new(1, HIDDEN, 128)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hc_branch_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hc_branch_f32,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                qdq.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            unsafe {
                compute_to_transfer_barrier(ctx, command);
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(hc_branch_f32)
                        .dst_offset(READBACK_ATTENTION_INPUT_OFFSET)
                        .size(HIDDEN as u64 * 4)],
                );
                transfer_to_compute_barrier(ctx, command);
            }

            let query_low_f32 = workspace_offset(graph, S14Position0WorkspaceSlot::QueryLowF32);
            let query_low_bf16 = workspace_offset(graph, S14Position0WorkspaceSlot::QueryLowBf16);
            let query_f32 = workspace_offset(graph, S14Position0WorkspaceSlot::QueryF32);
            let query_bf16 = workspace_offset(graph, S14Position0WorkspaceSlot::QueryBf16);
            let key_value_f32 = workspace_offset(graph, S14Position0WorkspaceSlot::KeyValueF32);
            let key_value_bf16 = workspace_offset(graph, S14Position0WorkspaceSlot::KeyValueBf16);
            let attention_bf16 = workspace_offset(graph, S14Position0WorkspaceSlot::AttentionBf16);

            let dispatch = numeric.bind_fp8_arenas(
                ctx,
                S14MatvecShape::new(1024, HIDDEN)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("attn.wq_a.weight")?,
                graph.static_offset_suffix("attn.wq_a.scale")?,
                workspace,
                workspace_bytes,
                hc_branch_f32,
                query_low_f32,
            )?;
            unsafe {
                numeric.cmd_fp8_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = f32_to_bf16.bind_slices(
                ctx,
                S14F32ToBf16Shape::new(1024)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_low_f32,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_low_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                f32_to_bf16.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = rmsnorm.bind_slices(
                ctx,
                S14Bf16RmsNormShape::new(1, 1024)?,
                NORM_EPS,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_low_bf16,
                },
                StorageBufferSlice {
                    buffer: static_arena,
                    offset: graph.static_offset_suffix("attn.q_norm.weight")?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hc_inverse,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hc_branch_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                rmsnorm.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = qdq.bind_slices(
                ctx,
                S14E4m3QdqShape::new(1, 1024, 128)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hc_branch_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_low_f32,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                qdq.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric_exact.bind_fp8_arenas(
                ctx,
                S14MatvecShape::new(32_768, 1024)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("attn.wq_b.weight")?,
                graph.static_offset_suffix("attn.wq_b.scale")?,
                workspace,
                workspace_bytes,
                query_low_f32,
                query_f32,
            )?;
            unsafe {
                numeric_exact.cmd_fp8_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = f32_to_bf16.bind_slices(
                ctx,
                S14F32ToBf16Shape::new(32_768)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_f32,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                f32_to_bf16.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = q_head_normalize.bind_slices(
                ctx,
                NORM_EPS,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: attention_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                q_head_normalize.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);

            // position4+ ratio4 真实 index query：官方 `_linear_fp8(qr)` 会先做
            // activation E4M3 QDQ，和主 wq_b 共享 query_low_f32 中的同一份真实 qr QDQ。
            // QueryF32/QueryBf16 在 attention query normalize 完成后已经死亡，可安全
            // 复用；AttentionF32 此时尚未被 wo 输出链使用，用作最终 BF16 index query。
            let index_query_bf16 = workspace_offset(graph, S14Position0WorkspaceSlot::AttentionF32);
            if sparse_ratio4 {
                let sparse = gpu
                    .sparse_attention
                    .as_ref()
                    .expect("position4+ sparse attention pipelines");
                let index_query_numeric_exact = gpu
                    .sparse_index_query_numeric_exact
                    .as_ref()
                    .or(gpu.numeric_exact.as_ref())
                    .ok_or_else(|| {
                        anyhow!("position4+ ratio4 index-query 缺少 K=1024 exact 归约 pipeline")
                    })?;
                let dispatch = index_query_numeric_exact.bind_fp8_arenas(
                    ctx,
                    S14MatvecShape::new(8_192, 1_024)?,
                    static_arena,
                    graph.static_logical_bytes,
                    graph.static_offset_suffix("attn.indexer.wq_b.weight")?,
                    graph.static_offset_suffix("attn.indexer.wq_b.scale")?,
                    workspace,
                    workspace_bytes,
                    query_low_f32,
                    query_f32,
                )?;
                unsafe {
                    index_query_numeric_exact.cmd_fp8_matvec(ctx, command, &dispatch);
                    l0_compute_barrier(ctx, command);
                }
                binders.push(dispatch.binder);
                let dispatch = f32_to_bf16.bind_slices(
                    ctx,
                    S14F32ToBf16Shape::new(8_192)?,
                    StorageBufferSlice {
                        buffer: workspace,
                        offset: query_f32,
                    },
                    StorageBufferSlice {
                        buffer: workspace,
                        offset: query_bf16,
                    },
                    StorageBufferSlice::whole(candidate.sticky_status),
                )?;
                unsafe {
                    f32_to_bf16.cmd(ctx, command, &dispatch);
                    l0_compute_barrier(ctx, command);
                }
                binders.push(dispatch.binder);
                let dispatch = sparse.index_query.bind_slices(
                    ctx,
                    StorageBufferSlice {
                        buffer: workspace,
                        offset: query_bf16,
                    },
                    StorageBufferSlice {
                        buffer: gpu.position1_rope(),
                        offset: POSITION1_ROPE_YARN_OFFSET,
                    },
                    StorageBufferSlice {
                        buffer: workspace,
                        offset: index_query_bf16,
                    },
                    StorageBufferSlice::whole(candidate.sticky_status),
                    S14Ratio4IndexQueryShape::new(
                        candidate.committed_host_state.position,
                        kv.compress_ratio,
                    )?,
                )?;
                unsafe {
                    sparse.index_query.cmd(ctx, command, &dispatch);
                    l0_compute_barrier(ctx, command);
                }
                binders.push(dispatch.binder);
            }
            unsafe {
                compute_to_transfer_barrier(ctx, command);
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(attention_bf16)
                        .dst_offset(READBACK_QUERY_FINAL_OFFSET)
                        .size(32_768 * 2)],
                );
                transfer_to_compute_barrier(ctx, command);
            }

            let dispatch = numeric.bind_fp8_arenas(
                ctx,
                S14MatvecShape::new(512, HIDDEN)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("attn.wkv.weight")?,
                graph.static_offset_suffix("attn.wkv.scale")?,
                workspace,
                workspace_bytes,
                hc_branch_f32,
                key_value_f32,
            )?;
            unsafe {
                numeric.cmd_fp8_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = f32_to_bf16.bind_slices(
                ctx,
                S14F32ToBf16Shape::new(512)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: key_value_f32,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: key_value_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                f32_to_bf16.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = rmsnorm.bind_slices(
                ctx,
                S14Bf16RmsNormShape::new(1, 512)?,
                NORM_EPS,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: key_value_bf16,
                },
                StorageBufferSlice {
                    buffer: static_arena,
                    offset: graph.static_offset_suffix("attn.kv_norm.weight")?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hc_inverse,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_low_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                rmsnorm.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = qdq.bind_slices(
                ctx,
                S14E4m3QdqShape::new(1, 448, 64)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_low_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: key_value_f32,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                qdq.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = f32_to_bf16.bind_slices(
                ctx,
                S14F32ToBf16Shape::new(448)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: key_value_f32,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: key_value_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                f32_to_bf16.cmd(ctx, command, &dispatch);
                compute_to_transfer_barrier(ctx, command);
                // 只把未量化的末64维从 KV RMSNorm 输出补回；前448维保持QDQ边界。
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    workspace.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(query_low_bf16 + 448 * 2)
                        .dst_offset(key_value_bf16 + 448 * 2)
                        .size(64 * 2)],
                );
                // 下一次 transfer 会读取刚补回的末 64 维，必须显式建立
                // transfer-write → transfer-read 依赖；compute barrier 不覆盖此处。
                transfer_to_transfer_barrier(ctx, command);
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(key_value_bf16)
                        .dst_offset(READBACK_KEY_VALUE_FINAL_OFFSET)
                        .size(512 * 2)],
                );
                transfer_to_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);

            // Window KV state stores keys after their own position RoPE.  The
            // attention kernels rotate the current raw row internally, so keep
            // their input unchanged and materialize a separate rotated row for
            // the later transactional state writeback.  Reusing QueryLowBf16 is
            // safe here: wq_b/q-head normalization have already consumed it and
            // no downstream stage reads it again.
            let current_rope_offset = match kv.compress_ratio {
                0 => POSITION1_ROPE_RATIO0_OFFSET,
                4 | 128 => POSITION1_ROPE_YARN_OFFSET,
                other => bail!("L{} unsupported KV RoPE ratio {other}", graph.layer),
            };
            let finalize = gpu
                .ratio4_finalize
                .as_ref()
                .expect("position-aware KV RoPE pipeline");
            let rotated_kv = finalize.rope.bind_slices(
                ctx,
                S14Ratio4CompressorKind::Main,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: key_value_bf16,
                },
                StorageBufferSlice {
                    buffer: gpu.position1_rope(),
                    offset: current_rope_offset,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_low_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                finalize.rope.cmd(ctx, command, &rotated_kv);
                l0_compute_barrier(ctx, command);
            }
            binders.push(rotated_kv.binder);

            // 3. position-aware attention。两条分支输出同一 query_bf16，后续
            // wo_a/wo_b/HC/router 链保持完全一致。
            match attention_mode {
                S14ProductionAttentionMode::Position0 => {
                    let attention = gpu
                        .attention
                        .as_ref()
                        .expect("position0 attention pipeline");
                    let dispatch = attention.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: attention_bf16,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: key_value_bf16,
                        },
                        StorageBufferSlice {
                            buffer: static_arena,
                            offset: graph.static_offset_suffix("attn.attn_sink")?,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: query_bf16,
                        },
                    )?;
                    unsafe {
                        attention.cmd(ctx, command, &dispatch);
                    }
                    binders.push(dispatch.binder);
                }
                S14ProductionAttentionMode::PreCompressionWindow {
                    previous_kv_offset,
                    previous_count,
                    rope_offset,
                } => {
                    let attention = gpu
                        .position1_attention
                        .as_ref()
                        .expect("position1 attention pipeline");
                    let dispatch = attention.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: attention_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: previous_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: key_value_bf16,
                        },
                        StorageBufferSlice {
                            buffer: static_arena,
                            offset: graph.static_offset_suffix("attn.attn_sink")?,
                        },
                        StorageBufferSlice {
                            buffer: gpu.position1_rope(),
                            offset: rope_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: query_bf16,
                        },
                        candidate.committed_host_state.position,
                        previous_count,
                    )?;
                    unsafe {
                        attention.cmd(ctx, command, &dispatch);
                    }
                    binders.push(dispatch.binder);
                }
                S14ProductionAttentionMode::RingWindow {
                    previous_kv_offset,
                    window_start,
                    previous_count,
                    rope_offset,
                } => {
                    let sparse = gpu
                        .sparse_attention
                        .as_ref()
                        .expect("position128 ring-window attention pipelines");
                    let shape = S14SparseAttentionShape::new(
                        candidate.committed_host_state.position,
                        window_start,
                        previous_count,
                        0,
                    )?;
                    // compressed_count=0 时两个 descriptor 只用于满足固定ABI，shader
                    // 不会读取；绑定已死亡的workspace scratch，避免越过ratio0 KV尾端。
                    let unused =
                        workspace_offset(graph, S14Position0WorkspaceSlot::CompressorProjectionF32);
                    let attention_dispatch = sparse.attention.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: attention_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: previous_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: key_value_bf16,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: unused,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: unused,
                        },
                        StorageBufferSlice {
                            buffer: static_arena,
                            offset: graph.static_offset_suffix("attn.attn_sink")?,
                        },
                        StorageBufferSlice {
                            buffer: gpu.position1_rope(),
                            offset: rope_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: query_bf16,
                        },
                        StorageBufferSlice::whole(candidate.sticky_status),
                        shape,
                    )?;
                    unsafe {
                        sparse.attention.cmd(ctx, command, &attention_dispatch);
                    }
                    binders.push(attention_dispatch.binder);
                }
                S14ProductionAttentionMode::Ratio4FirstCompressedBlock {
                    previous_kv_offset,
                    compressed_kv_offset,
                    rope_offset,
                } => {
                    let attention = gpu
                        .position3_attention
                        .as_ref()
                        .expect("position3 attention pipeline");
                    let dispatch = attention.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: attention_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: previous_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: key_value_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: compressed_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: static_arena,
                            offset: graph.static_offset_suffix("attn.attn_sink")?,
                        },
                        StorageBufferSlice {
                            buffer: gpu.position1_rope(),
                            offset: rope_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: query_bf16,
                        },
                        candidate.committed_host_state.position,
                        kv.compress_ratio,
                        3,
                        1,
                    )?;
                    unsafe {
                        attention.cmd(ctx, command, &dispatch);
                    }
                    binders.push(dispatch.binder);
                }
                S14ProductionAttentionMode::Ratio4Sparse {
                    previous_kv_offset,
                    window_start,
                    previous_count,
                    compressed_kv_offset,
                    compressed_count,
                    rope_offset,
                } => {
                    let indexer_state = unique_native_entry(
                        candidate
                            .committed_host_state
                            .native
                            .indexers
                            .iter()
                            .filter(|entry| entry.layer == graph.layer),
                        &format!("L{} ratio4 sparse indexer state", graph.layer),
                    )?;
                    if indexer_state.kv_cache.dtype != DType::Bf16
                        || indexer_state.kv_cache.shape.len() != 3
                        || indexer_state.kv_cache.shape[0] != 1
                        || indexer_state.kv_cache.shape[2] != 128
                        || indexer_state.kv_cache.shape[1] < compressed_count
                    {
                        bail!("L{} ratio4 sparse index cache shape 漂移", graph.layer);
                    }
                    let index_cache_end = indexer_state
                        .kv_cache
                        .offset
                        .checked_add(u64::from(compressed_count) * 128 * 2)
                        .ok_or_else(|| anyhow!("L{} index cache range overflow", graph.layer))?;
                    if index_cache_end
                        > indexer_state
                            .kv_cache
                            .offset
                            .checked_add(indexer_state.kv_cache.bytes)
                            .ok_or_else(|| anyhow!("L{} index cache bytes overflow", graph.layer))?
                        || index_cache_end > candidate.committed_host_state.native.arena_bytes
                    {
                        bail!("L{} ratio4 sparse index cache range 漂移", graph.layer);
                    }
                    let sparse = gpu
                        .sparse_attention
                        .as_ref()
                        .expect("position4+ sparse attention pipelines");
                    let index_scores = query_low_f32;
                    let index_ids =
                        workspace_offset(graph, S14Position0WorkspaceSlot::CompressorProjectionF32);
                    let index_dispatch = sparse.indexer.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: index_query_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: indexer_state.kv_cache.offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: index_weights_bf16,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: index_scores,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: index_ids,
                        },
                        StorageBufferSlice::whole(candidate.sticky_status),
                        compressed_count,
                    )?;
                    unsafe {
                        sparse.indexer.cmd(ctx, command, &index_dispatch);
                        l0_compute_barrier(ctx, command);
                    }
                    binders.push(index_dispatch.binder);

                    let shape = S14SparseAttentionShape::new(
                        candidate.committed_host_state.position,
                        window_start,
                        previous_count,
                        compressed_count,
                    )?;
                    let attention_dispatch = sparse.attention.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: attention_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: previous_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: key_value_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: compressed_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: index_ids,
                        },
                        StorageBufferSlice {
                            buffer: static_arena,
                            offset: graph.static_offset_suffix("attn.attn_sink")?,
                        },
                        StorageBufferSlice {
                            buffer: gpu.position1_rope(),
                            offset: rope_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: query_bf16,
                        },
                        StorageBufferSlice::whole(candidate.sticky_status),
                        shape,
                    )?;
                    unsafe {
                        sparse.attention.cmd(ctx, command, &attention_dispatch);
                    }
                    binders.push(attention_dispatch.binder);
                }
                S14ProductionAttentionMode::Ratio4PagedSparse {
                    previous_kv_offset,
                    window_start,
                    previous_count,
                    compressed_kv_offset,
                    logical_compressed_count,
                    selected_count,
                    rope_offset,
                } => {
                    if candidate.committed_host_state.position != 2051
                        || logical_compressed_count != 513
                        || selected_count != 512
                    {
                        bail!(
                            "L{} ratio4 paged production 只闭合position2051",
                            graph.layer
                        );
                    }
                    let history = S14Ratio4HistoryLayout::build(
                        &candidate.committed_host_state.native,
                        graph.layer,
                        logical_compressed_count,
                    )?;
                    if history.pages.len() != 2
                        || history.pages[0].logical_rows != (0..512)
                        || history.pages[1].logical_rows != (512..513)
                    {
                        bail!("L{} position2051 ratio4 history page合同漂移", graph.layer);
                    }
                    let indexer_state = unique_native_entry(
                        candidate
                            .committed_host_state
                            .native
                            .indexers
                            .iter()
                            .filter(|entry| entry.layer == graph.layer),
                        &format!("L{} ratio4 paged indexer state", graph.layer),
                    )?;
                    let scratch = ratio4_paged_workspace(graph)?;
                    let sparse = gpu
                        .sparse_attention
                        .as_ref()
                        .expect("position2051 sparse attention pipelines");
                    let global_topk = gpu
                        .ratio4_global_topk
                        .as_ref()
                        .expect("position2051 ratio4 global top-k pipeline");
                    let global_bindings = S14Ratio4PagedGlobalTopKBindings {
                        processed_index_query: StorageBufferSlice {
                            buffer: workspace,
                            offset: index_query_bf16,
                        },
                        indexer_history: StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: indexer_state.kv_cache.offset,
                        },
                        head_weights: StorageBufferSlice {
                            buffer: workspace,
                            offset: index_weights_bf16,
                        },
                        page_scores: StorageBufferSlice {
                            buffer: workspace,
                            offset: scratch.page_scores,
                        },
                        page_indices: StorageBufferSlice {
                            buffer: workspace,
                            offset: scratch.page_indices,
                        },
                        global_score_banks: scratch.global_score_banks.map(|offset| {
                            StorageBufferSlice {
                                buffer: workspace,
                                offset,
                            }
                        }),
                        global_index_banks: scratch.global_index_banks.map(|offset| {
                            StorageBufferSlice {
                                buffer: workspace,
                                offset,
                            }
                        }),
                        status: StorageBufferSlice::whole(candidate.sticky_status),
                    };
                    let global_recording = unsafe {
                        global_topk.record_paged_indexer_global_topk(
                            &sparse.indexer,
                            ctx,
                            command,
                            &history,
                            global_bindings,
                        )?
                    };
                    let receipt = global_recording.receipt();
                    if receipt.logical_count != logical_compressed_count
                        || receipt.selected_count != selected_count
                        || receipt.scanned_pages != 2
                    {
                        bail!(
                            "L{} position2051 ratio4 global top-k receipt漂移",
                            graph.layer
                        );
                    }
                    let selected_indices = global_recording.final_indices(&global_bindings);
                    let (_, mut global_binders) = global_recording.into_parts();
                    binders.append(&mut global_binders);

                    let source_word_count = logical_compressed_count
                        .checked_mul(S14_RATIO4_GATHER_ROW_WORDS)
                        .ok_or_else(|| {
                            anyhow!("L{} ratio4 main source words overflow", graph.layer)
                        })?;
                    let gather_shape = S14Ratio4MainGatherShape::new(
                        logical_compressed_count,
                        selected_count,
                        source_word_count,
                    )?;
                    let materialized_pages = history
                        .pages
                        .iter()
                        .map(|page| {
                            Ok(S14Ratio4MaterializedMainPage {
                                page_index: page.page_index,
                                source_word_offset: page
                                    .logical_rows
                                    .start
                                    .checked_mul(S14_RATIO4_GATHER_ROW_WORDS)
                                    .ok_or_else(|| {
                                        anyhow!("ratio4 main page word offset overflow")
                                    })?,
                                row_count: page.logical_len(),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let page_table =
                        build_ratio4_main_page_table(gather_shape, &materialized_pages)?;
                    let identity_indices = (0..selected_count).collect::<Vec<_>>();
                    unsafe {
                        compute_read_write_to_transfer_barrier(ctx, command);
                        ctx.device.cmd_update_buffer(
                            command,
                            workspace.handle(),
                            scratch.page_table,
                            bytemuck::cast_slice(&page_table),
                        );
                        ctx.device.cmd_update_buffer(
                            command,
                            workspace.handle(),
                            scratch.page_indices,
                            bytemuck::cast_slice(&identity_indices),
                        );
                        transfer_to_compute_barrier(ctx, command);
                    }

                    let gather = gpu
                        .ratio4_main_gather
                        .as_ref()
                        .expect("position2051 ratio4 main-page gather pipeline");
                    let gather_dispatch = gather.bind_slices(
                        ctx,
                        selected_indices,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: scratch.page_table,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: compressed_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: scratch.packed_main,
                        },
                        StorageBufferSlice::whole(candidate.sticky_status),
                        gather_shape,
                    )?;
                    unsafe {
                        gather.cmd(ctx, command, &gather_dispatch);
                        l0_compute_barrier(ctx, command);
                    }
                    binders.push(gather_dispatch.binder);

                    let shape = S14SparseAttentionShape::new(
                        candidate.committed_host_state.position,
                        window_start,
                        previous_count,
                        selected_count,
                    )?;
                    let attention_dispatch = sparse.attention.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: attention_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: previous_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: key_value_bf16,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: scratch.packed_main,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: scratch.page_indices,
                        },
                        StorageBufferSlice {
                            buffer: static_arena,
                            offset: graph.static_offset_suffix("attn.attn_sink")?,
                        },
                        StorageBufferSlice {
                            buffer: gpu.position1_rope(),
                            offset: rope_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: query_bf16,
                        },
                        StorageBufferSlice::whole(candidate.sticky_status),
                        shape,
                    )?;
                    unsafe {
                        sparse.attention.cmd(ctx, command, &attention_dispatch);
                    }
                    binders.push(attention_dispatch.binder);
                }
                S14ProductionAttentionMode::Ratio128Deterministic {
                    previous_kv_offset,
                    window_start,
                    previous_count,
                    compressed_kv_offset,
                    compressed_count,
                    rope_offset,
                } => {
                    let sparse = gpu
                        .sparse_attention
                        .as_ref()
                        .expect("position127 ratio128 sparse attention pipelines");
                    let shape = S14SparseAttentionShape::new_ratio128(
                        candidate.committed_host_state.position,
                        window_start,
                        previous_count,
                        compressed_count,
                    )?;
                    // ratio128 由 shader 按 block0..N-1 隐式消费；descriptor 仍绑定
                    // 一个容量充足的 scratch slice，但绝不读取 ratio4 indexer 排名。
                    let unused_indices =
                        workspace_offset(graph, S14Position0WorkspaceSlot::CompressorProjectionF32);
                    let attention_dispatch = sparse.attention.bind_slices(
                        ctx,
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: attention_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: previous_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: key_value_bf16,
                        },
                        StorageBufferSlice {
                            buffer: candidate.candidate_state,
                            offset: compressed_kv_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: unused_indices,
                        },
                        StorageBufferSlice {
                            buffer: static_arena,
                            offset: graph.static_offset_suffix("attn.attn_sink")?,
                        },
                        StorageBufferSlice {
                            buffer: gpu.position1_rope(),
                            offset: rope_offset,
                        },
                        StorageBufferSlice {
                            buffer: workspace,
                            offset: query_bf16,
                        },
                        StorageBufferSlice::whole(candidate.sticky_status),
                        shape,
                    )?;
                    unsafe {
                        sparse.attention.cmd(ctx, command, &attention_dispatch);
                    }
                    binders.push(attention_dispatch.binder);
                }
            }
            unsafe {
                l0_compute_barrier(ctx, command);
                compute_to_transfer_barrier(ctx, command);
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(query_bf16)
                        .dst_offset(READBACK_ATTENTION_OUTPUT_OFFSET)
                        .size(32_768 * 2)],
                );
                // Publish the separately RoPE-rotated current key as the exact
                // source consumed later by record_window_kv.  Historical rows
                // are therefore already position-rotated when position1/2/3
                // attention reads them on the next token.
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    workspace.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(query_low_bf16)
                        .dst_offset(key_value_bf16)
                        .size(512 * 2)],
                );
                transfer_to_transfer_barrier(ctx, command);
                transfer_to_compute_barrier(ctx, command);
            }
            let attention_f32 = workspace_offset(graph, S14Position0WorkspaceSlot::AttentionF32);
            let dispatch = bf16_to_f32.bind_slices(
                ctx,
                S14Bf16ToF32Shape::new(32_768)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: attention_f32,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                bf16_to_f32.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let grouped = workspace_offset(graph, S14Position0WorkspaceSlot::GroupedWoAF32);
            let dispatch = numeric.bind_grouped_fp8_bf16_weight_arenas(
                ctx,
                S14GroupedMatvecShape::new(8, 1024, HIDDEN)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("attn.wo_a.weight")?,
                graph.static_offset_suffix("attn.wo_a.scale")?,
                workspace,
                workspace_bytes,
                attention_f32,
                grouped,
            )?;
            unsafe {
                numeric.cmd_grouped_fp8_bf16_weight_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = f32_to_bf16.bind_slices(
                ctx,
                S14F32ToBf16Shape::new(8192)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: grouped,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                f32_to_bf16.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = qdq.bind_slices(
                ctx,
                S14E4m3QdqShape::new(1, 8192, 128)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: query_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: scratch,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: grouped,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                qdq.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
                compute_to_transfer_barrier(ctx, command);
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(grouped)
                        .dst_offset(READBACK_WO_A_QDQ_OFFSET)
                        .size(8_192 * 4)],
                );
                transfer_to_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let attention_branch_f32 =
                workspace_offset(graph, S14Position0WorkspaceSlot::AttentionBranchF32);
            let attention_branch_bf16 =
                workspace_offset(graph, S14Position0WorkspaceSlot::AttentionBranchBf16);
            let dispatch = numeric_exact.bind_fp8_arenas(
                ctx,
                S14MatvecShape::new(HIDDEN, 8192)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("attn.wo_b.weight")?,
                graph.static_offset_suffix("attn.wo_b.scale")?,
                workspace,
                workspace_bytes,
                grouped,
                attention_branch_f32,
            )?;
            unsafe {
                numeric_exact.cmd_fp8_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = f32_to_bf16.bind_slices(
                ctx,
                S14F32ToBf16Shape::new(HIDDEN)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: attention_branch_f32,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: attention_branch_bf16,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                f32_to_bf16.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = hc_post.bind_slices(
                ctx,
                S14HcPostShape::new(HIDDEN)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: attention_branch_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hidden_a,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hc_aux,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hc_aux + 16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: hidden_b,
                },
                StorageBufferSlice::whole(candidate.sticky_status),
            )?;
            unsafe {
                hc_post.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
                compute_to_transfer_barrier(ctx, command);
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    readback.handle(),
                    &[
                        vk::BufferCopy::default()
                            .src_offset(attention_branch_bf16)
                            .dst_offset(READBACK_ATTENTION_BRANCH_OFFSET)
                            .size(HIDDEN as u64 * 2),
                        vk::BufferCopy::default()
                            .src_offset(hidden_b)
                            .dst_offset(READBACK_POST_ATTENTION_OFFSET)
                            .size(4 * HIDDEN as u64 * 2),
                    ],
                );
                transfer_to_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);

            // 4. FFN HC-pre，router 必须先看未做 activation QDQ 的原始 F32 branch。
            let dispatch = numeric.bind_hc_normalize_input_arena(
                ctx,
                hc_shape,
                NORM_EPS,
                workspace,
                workspace_bytes,
                hidden_b,
                hc_norm,
                hc_inverse,
            )?;
            unsafe {
                numeric.cmd_hc_normalize_input(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric.bind_f32_matvec_arenas(
                ctx,
                S14F32MatvecShape::new(24, HC_FLAT, 1)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("hc_ffn_fn")?,
                workspace,
                workspace_bytes,
                hc_norm,
                workspace,
                workspace_bytes,
                scratch,
            )?;
            unsafe {
                numeric.cmd_f32_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric.bind_hc_split_reduce_norm_arenas(
                ctx,
                hc_shape,
                NORM_EPS,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("hc_ffn_scale")?,
                graph.static_offset_suffix("hc_ffn_base")?,
                graph.static_offset_suffix("ffn_norm.weight")?,
                workspace,
                workspace_bytes,
                hidden_b,
                scratch,
                hc_branch_bf16,
                hc_branch_f32,
                hc_aux,
                hc_inverse,
            )?;
            unsafe {
                numeric.cmd_hc_split_reduce_norm(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            // 逐层归因门可覆盖 PyTorch/libm 与 GLSL exp 在 1e-7 量级的 HC
            // post/comb 差异，以独立验证后续 route/MoE/HC-post。生产构造器永不启用。
            if gpu.numeric_hc_override {
                unsafe {
                    compute_to_transfer_barrier(ctx, command);
                    ctx.device.cmd_copy_buffer(
                        command,
                        immutable.handle(),
                        workspace.handle(),
                        &[vk::BufferCopy::default()
                            .src_offset(AUX_HC_OVERRIDE_OFFSET)
                            .dst_offset(hc_aux)
                            .size(AUX_HC_OVERRIDE_BYTES)],
                    );
                    transfer_to_compute_barrier(ctx, command);
                }
            }

            let router_logits = workspace_offset(graph, S14Position0WorkspaceSlot::RouterLogitsF32);
            let router_ids = workspace_offset(graph, S14Position0WorkspaceSlot::RouterIdsU32);
            let router_weights =
                workspace_offset(graph, S14Position0WorkspaceSlot::RouterWeightsF32);
            let router_selected =
                workspace_offset(graph, S14Position0WorkspaceSlot::RouterSelectedScoresF32);
            let router_ranking =
                workspace_offset(graph, S14Position0WorkspaceSlot::RouterRankingScoresF32);
            let dispatch = numeric.bind_bf16_matvec_arenas(
                ctx,
                S14Bf16MatvecShape::new(256, HIDDEN, 1)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix("ffn.gate.weight")?,
                workspace,
                workspace_bytes,
                hc_branch_f32,
                workspace,
                workspace_bytes,
                router_logits,
            )?;
            unsafe {
                numeric.cmd_bf16_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let route_aux = match graph.route_mode {
                S14RoutePostprocessGpuMode::PhysicalIds => {
                    S14RouteBufferSlice::new(immutable, AUX_PHYSICAL_IDS_OFFSET)
                }
                S14RoutePostprocessGpuMode::BiasTop6 => S14RouteBufferSlice::new(
                    static_arena,
                    graph.static_offset_suffix("ffn.gate.bias")?,
                ),
            };
            let dispatch = route.bind_with_offsets(
                ctx,
                graph.route_mode,
                S14RoutePostprocessGpuBindings {
                    logits: S14RouteBufferSlice::new(workspace, router_logits),
                    aux: route_aux,
                    expert_ids: S14RouteBufferSlice::new(workspace, router_ids),
                    weights: S14RouteBufferSlice::new(workspace, router_weights),
                    selected_scores: S14RouteBufferSlice::new(workspace, router_selected),
                    ranking_scores: S14RouteBufferSlice::new(workspace, router_ranking),
                    status: S14RouteBufferSlice::new(candidate.sticky_status, 0),
                },
            )?;
            unsafe {
                route.cmd(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            if segment == S14LayerCommandSegment::RouterProbe {
                unsafe {
                    compute_to_transfer_barrier(ctx, command);
                    ctx.device.cmd_copy_buffer(
                        command,
                        workspace.handle(),
                        readback.handle(),
                        &[
                            vk::BufferCopy::default()
                                .src_offset(router_ids)
                                .dst_offset(READBACK_ROUTE_IDS_OFFSET)
                                .size((EXPERTS_PER_TOKEN * std::mem::size_of::<u32>()) as u64),
                            vk::BufferCopy::default()
                                .src_offset(router_weights)
                                .dst_offset(READBACK_ROUTE_WEIGHTS_OFFSET)
                                .size((EXPERTS_PER_TOKEN * std::mem::size_of::<f32>()) as u64),
                        ],
                    );
                    let host_barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ);
                    ctx.device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[host_barrier],
                        &[],
                        &[],
                    );
                    ctx.device.end_command_buffer(command)?;
                }
                return Ok(());
            }
            // routed page 的六个权重槽按 manifest 专家身份固定，而 top-k 排名在
            // 极近分数下允许产生同集合的不同顺序。先按 expert ID 对齐在线权重，
            // 再进入专家计算；集合变化则通过 sticky status fail-closed。
            if !gpu.reference_route_replay {
                let dispatch = route_slot_align.bind(
                    ctx,
                    S14RouteSlotAlignBindings {
                        actual_ids: S14RouteSlotAlignSlice::new(workspace, router_ids),
                        actual_weights: S14RouteSlotAlignSlice::new(workspace, router_weights),
                        expected_ids: S14RouteSlotAlignSlice::new(
                            immutable,
                            AUX_PHYSICAL_IDS_OFFSET,
                        ),
                        aligned_weights: S14RouteSlotAlignSlice::new(workspace, router_selected),
                        status: S14RouteSlotAlignSlice::new(candidate.sticky_status, 0),
                    },
                )?;
                unsafe {
                    route_slot_align.cmd(ctx, command, &dispatch);
                    l0_compute_barrier(ctx, command);
                }
                binders.push(dispatch.binder);
            }
        }
        // 仅供 L0 数值归因门使用：保持真实 GPU route 的专家身份不变，
        // 只把同一 top-6 的 reference 权重覆盖回 workspace，以证明后续
        // MoE/HC 漂移是否完全来自 BF16 router 投影边界。生产构造器永不启用。
        let expert_route_weights = if segment == S14LayerCommandSegment::DynamicMoeContinuation {
            router_weights
        } else if gpu.numeric_route_override || gpu.reference_route_replay {
            unsafe {
                compute_to_transfer_barrier(ctx, command);
                ctx.device.cmd_copy_buffer(
                    command,
                    immutable.handle(),
                    workspace.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(AUX_NUMERIC_ROUTE_OVERRIDE_OFFSET)
                        .dst_offset(router_weights)
                        .size((EXPERTS_PER_TOKEN * std::mem::size_of::<f32>()) as u64)],
                );
                transfer_to_compute_barrier(ctx, command);
            }
            router_weights
        } else {
            router_selected
        };

        let dispatch = qdq.bind_slices(
            ctx,
            S14E4m3QdqShape::new(1, HIDDEN, 128)?,
            StorageBufferSlice {
                buffer: workspace,
                offset: hc_branch_bf16,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: scratch,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: hc_branch_f32,
            },
            StorageBufferSlice::whole(candidate.sticky_status),
        )?;
        unsafe {
            qdq.cmd(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);

        // 5. Routed top-6 + shared expert + exact 0→5→shared BF16 边界归约。
        let expert_gate = workspace_offset(graph, S14Position0WorkspaceSlot::ExpertGateF32);
        let expert_up = workspace_offset(graph, S14Position0WorkspaceSlot::ExpertUpF32);
        let expert_hidden = workspace_offset(graph, S14Position0WorkspaceSlot::ExpertHiddenF32);
        let expert_down = workspace_offset(graph, S14Position0WorkspaceSlot::ExpertDownF32);
        for (projection, output) in [
            (S14RaggedProjection::W1, expert_gate),
            (S14RaggedProjection::W3, expert_up),
        ] {
            let dispatch = numeric_exact.bind_ragged_mxfp4_arenas(
                ctx,
                S14RaggedMatvecShape::new(TOP6, TOP6, INTERMEDIATE, HIDDEN, projection)?,
                routed_arena,
                graph.routed_logical_bytes,
                immutable,
                AUX_LOGICAL_BYTES,
                AUX_METADATA_OFFSET,
                &graph.routed_metadata,
                workspace,
                workspace_bytes,
                hc_branch_f32,
                output,
            )?;
            unsafe {
                numeric_exact.cmd_ragged_mxfp4_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
        }
        let dispatch = numeric.bind_batched_official_expert_prepare_arena(
            ctx,
            TOP6,
            INTERMEDIATE,
            workspace,
            workspace_bytes,
            expert_gate,
            expert_up,
            expert_route_weights,
            expert_hidden,
        )?;
        unsafe {
            numeric.cmd_batched_official_expert_prepare(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        let dispatch = numeric_exact.bind_ragged_mxfp4_arenas(
            ctx,
            S14RaggedMatvecShape::new(TOP6, 1, HIDDEN, INTERMEDIATE, S14RaggedProjection::W2)?,
            routed_arena,
            graph.routed_logical_bytes,
            immutable,
            AUX_LOGICAL_BYTES,
            AUX_METADATA_OFFSET,
            &graph.routed_metadata,
            workspace,
            workspace_bytes,
            expert_hidden,
            expert_down,
        )?;
        unsafe {
            numeric_exact.cmd_ragged_mxfp4_matvec(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);

        let shared_gate = workspace_offset(graph, S14Position0WorkspaceSlot::SharedGateF32);
        let shared_up = workspace_offset(graph, S14Position0WorkspaceSlot::SharedUpF32);
        let shared_hidden = workspace_offset(graph, S14Position0WorkspaceSlot::SharedHiddenF32);
        let shared_down = workspace_offset(graph, S14Position0WorkspaceSlot::SharedDownF32);
        for (suffix, output) in [("w1", shared_gate), ("w3", shared_up)] {
            let dispatch = numeric_exact.bind_fp8_arenas(
                ctx,
                S14MatvecShape::new(INTERMEDIATE, HIDDEN)?,
                static_arena,
                graph.static_logical_bytes,
                graph.static_offset_suffix(&format!("ffn.shared_experts.{suffix}.weight"))?,
                graph.static_offset_suffix(&format!("ffn.shared_experts.{suffix}.scale"))?,
                workspace,
                workspace_bytes,
                hc_branch_f32,
                output,
            )?;
            unsafe {
                numeric_exact.cmd_fp8_matvec(ctx, command, &dispatch);
                l0_compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
        }
        unsafe {
            compute_to_transfer_barrier(ctx, command);
            ctx.device.cmd_copy_buffer(
                command,
                immutable.handle(),
                workspace.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(AUX_SHARED_ROUTE_ONE_OFFSET)
                    .dst_offset(router_selected)
                    .size(4)],
            );
            transfer_to_compute_barrier(ctx, command);
        }
        let dispatch = numeric.bind_batched_official_expert_prepare_arena(
            ctx,
            1,
            INTERMEDIATE,
            workspace,
            workspace_bytes,
            shared_gate,
            shared_up,
            router_selected,
            shared_hidden,
        )?;
        unsafe {
            numeric.cmd_batched_official_expert_prepare(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        let dispatch = numeric_exact.bind_fp8_arenas(
            ctx,
            S14MatvecShape::new(HIDDEN, INTERMEDIATE)?,
            static_arena,
            graph.static_logical_bytes,
            graph.static_offset_suffix("ffn.shared_experts.w2.weight")?,
            graph.static_offset_suffix("ffn.shared_experts.w2.scale")?,
            workspace,
            workspace_bytes,
            shared_hidden,
            shared_down,
        )?;
        unsafe {
            numeric_exact.cmd_fp8_matvec(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        let moe = workspace_offset(graph, S14Position0WorkspaceSlot::MoeAccumulatorF32);
        let dispatch = numeric.bind_exact_order_block_reduce_arena(
            ctx,
            1,
            workspace,
            workspace_bytes,
            expert_down,
            shared_down,
            moe,
        )?;
        unsafe {
            numeric.cmd_exact_order_block_reduce(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        let dispatch = f32_to_bf16.bind_slices(
            ctx,
            S14F32ToBf16Shape::new(HIDDEN)?,
            StorageBufferSlice {
                buffer: workspace,
                offset: moe,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: attention_branch_bf16,
            },
            StorageBufferSlice::whole(candidate.sticky_status),
        )?;
        unsafe {
            f32_to_bf16.cmd(ctx, command, &dispatch);
            l0_compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        let dispatch = hc_post.bind_slices(
            ctx,
            S14HcPostShape::new(HIDDEN)?,
            StorageBufferSlice {
                buffer: workspace,
                offset: attention_branch_bf16,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: hidden_b,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: hc_aux,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: hc_aux + 16,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: hidden_a,
            },
            StorageBufferSlice::whole(candidate.sticky_status),
        )?;
        unsafe {
            hc_post.cmd(ctx, command, &dispatch);
            compute_to_transfer_barrier(ctx, command);

            // FFN HC-post 的既有算子把最终状态写回 layer input 槽；发布到 layer
            // output 槽后，下一层无需 host readback 即可直接消费。
            ctx.device.cmd_copy_buffer(
                command,
                workspace.handle(),
                workspace.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(hidden_a)
                    .dst_offset(hidden_b)
                    .size(4 * HIDDEN as u64 * 2)],
            );
            transfer_to_transfer_barrier(ctx, command);

            // 6. L0 window KV 与专属 numeric readback。没有任何固定输出 token。
            if let Some(recipe) = state_recipe {
                recipe.record_window_kv(ctx, command, workspace, candidate.candidate_state)?;
            } else {
                ctx.device.cmd_copy_buffer(
                    command,
                    workspace.handle(),
                    candidate.candidate_state.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(key_value_bf16)
                        .dst_offset(kv.cache.offset)
                        .size(1024)],
                );
                let candidate_kv_barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(candidate.candidate_state.handle())
                    .offset(kv.cache.offset)
                    .size(1024);
                ctx.device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[candidate_kv_barrier],
                    &[],
                );
            }
            ctx.device.cmd_copy_buffer(
                command,
                workspace.handle(),
                readback.handle(),
                &[
                    vk::BufferCopy::default()
                        .src_offset(hidden_b)
                        .dst_offset(READBACK_HIDDEN_OFFSET)
                        .size(4 * HIDDEN as u64 * 2),
                    vk::BufferCopy::default()
                        .src_offset(router_ids)
                        .dst_offset(READBACK_ROUTE_IDS_OFFSET)
                        .size(24),
                    vk::BufferCopy::default()
                        .src_offset(expert_route_weights)
                        .dst_offset(READBACK_ROUTE_WEIGHTS_OFFSET)
                        .size(24),
                    vk::BufferCopy::default()
                        .src_offset(key_value_bf16)
                        .dst_offset(READBACK_KV_OFFSET)
                        .size(1024),
                    vk::BufferCopy::default()
                        .src_offset(moe)
                        .dst_offset(READBACK_MOE_OFFSET)
                        .size(HIDDEN as u64 * 4),
                    vk::BufferCopy::default()
                        .src_offset(hc_branch_f32)
                        .dst_offset(READBACK_FFN_INPUT_OFFSET)
                        .size(HIDDEN as u64 * 4),
                    vk::BufferCopy::default()
                        .src_offset(hc_aux)
                        .dst_offset(READBACK_HC_POST_OFFSET)
                        .size(16),
                    vk::BufferCopy::default()
                        .src_offset(hc_aux + 16)
                        .dst_offset(READBACK_HC_COMB_OFFSET)
                        .size(64),
                ],
            );
            ctx.device.cmd_copy_buffer(
                command,
                candidate.candidate_state.handle(),
                readback.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(kv.cache.offset)
                    .dst_offset(READBACK_CANDIDATE_KV_OFFSET)
                    .size(1024)],
            );
            let host_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            ctx.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[host_barrier],
                &[],
                &[],
            );
            ctx.device.end_command_buffer(command)?;
        }
        Ok(())
    })();

    if let Err(error) = result {
        if reset_internal_pool {
            unsafe {
                let _ = ctx.device.reset_command_pool(
                    gpu.command_pool,
                    vk::CommandPoolResetFlags::RELEASE_RESOURCES,
                );
            }
        }
        for binder in binders.drain(..) {
            binder.destroy(ctx);
        }
        return Err(error);
    }
    if segment != S14LayerCommandSegment::RouterProbe {
        gpu.layer_kv_offset = Some(layer_kv_offset);
    }
    Ok(binders)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14Position0L0BackendPhase {
    Ready,
    EmbeddingSubmitted,
    Layer0Submitted,
    Drained,
}

#[derive(Debug, Clone)]
pub struct S14Position0L0NumericReceipt {
    pub base_epoch: u64,
    pub candidate_bank: usize,
    pub compute_value: u64,
    pub route_ids: [u32; EXPERTS_PER_TOKEN],
    pub route_weights: [f32; EXPERTS_PER_TOKEN],
    pub finite_hidden_elements: usize,
    pub nonzero_hidden_elements: usize,
    pub hidden_bf16_bits: Vec<u16>,
    /// 将 GPU BF16 hidden 精确扩展成 F32 little-endian 后的 SHA-256。
    /// 该格式与 position0 manifest 的 `layer_output_f32_le_sha256` 一致。
    pub hidden_f32_le_sha256: String,
    pub hidden_l2: f64,
    pub hidden_mean: f64,
    pub hidden_max_abs: f32,
    pub attention_input_f32_values: Vec<f32>,
    pub attention_input_f32_le_sha256: String,
    pub attention_branch_bf16_bits: Vec<u16>,
    pub attention_branch_bf16_le_sha256: String,
    pub post_attention_bf16_bits: Vec<u16>,
    pub post_attention_bf16_le_sha256: String,
    pub query_final_bf16_bits: Vec<u16>,
    pub key_value_final_bf16_bits: Vec<u16>,
    pub attention_output_bf16_bits: Vec<u16>,
    pub wo_a_qdq_f32_values: Vec<f32>,
    pub ffn_input_f32_values: Vec<f32>,
    pub ffn_input_f32_le_sha256: String,
    pub hc_post_f32_values: [f32; 4],
    pub hc_comb_f32_values: [f32; 16],
    pub moe_f32_le_sha256: String,
    /// L0 MoE F32 accumulator 经与生产 shader 相同的 RNE 规则量化后的 BF16。
    /// 数值门用它与已验证的 `vulkan_moe_branch.bf16le.bin` 做逐元素归因。
    pub moe_bf16_bits: Vec<u16>,
    pub moe_bf16_le_sha256: String,
    pub moe_l2: f64,
    pub moe_mean: f64,
    pub moe_max_abs: f32,
    pub kv_candidate_exact: bool,
    pub sticky_status: u32,
}

/// 可编译、fail-closed 的 `Position0LayerBackend` 首段。
///
/// 当前类型先固化调度边界与 L0 真实权重身份。GPU owner/command recorder
/// 在同模块的下一步接入；在此之前不会对外声明 `all_layers` 或
/// `position0_state_outputs`。
pub struct S14Position0L0Backend<'ctx> {
    graph: S14Position0L0GraphPlan,
    state_recording: Option<S14Position0FullDepthStateRecordingProgram>,
    phase: S14Position0L0BackendPhase,
    base_epoch: Option<u64>,
    candidate_bank: Option<usize>,
    gpu: Option<S14Position0L0GpuOwner<'ctx>>,
}

impl<'ctx> S14Position0L0Backend<'ctx> {
    pub fn new(graph: S14Position0L0GraphPlan) -> Self {
        Self {
            graph,
            state_recording: None,
            phase: S14Position0L0BackendPhase::Ready,
            base_epoch: None,
            candidate_bank: None,
            gpu: None,
        }
    }

    pub fn new_gpu(
        ctx: &'ctx VulkanContext,
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        static_layout: &S14Position0StaticLayerLayout,
        payload_root: &Path,
    ) -> Result<Self> {
        Self::new_gpu_inner(
            ctx,
            manifest,
            weights,
            static_layout,
            0,
            payload_root,
            None,
            None,
            None,
        )
    }

    /// L0 数值门专用 A/B：真实执行 BOS→route→MoE→HC，但在 route 后用
    /// 已冻结 reference 的同一 top-6 权重覆盖 GPU 权重。它不能签发 whole-token
    /// 能力，也不会被生产 `new_gpu` 调用。
    pub fn new_gpu_numeric_route_override(
        ctx: &'ctx VulkanContext,
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        static_layout: &S14Position0StaticLayerLayout,
        payload_root: &Path,
        route_weights: [f32; EXPERTS_PER_TOKEN],
    ) -> Result<Self> {
        if route_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
            || (route_weights.iter().sum::<f32>() - 1.5).abs() > 2.0e-6
        {
            bail!("L0 numeric route override 非法");
        }
        Self::new_gpu_inner(
            ctx,
            manifest,
            weights,
            static_layout,
            0,
            payload_root,
            Some(route_weights),
            None,
            None,
        )
    }

    /// 任意单层的真实数值门：从冻结的四流 BF16 hidden 直接进入当前层，
    /// 仍执行真实 attention、route、top-6 MoE、HC 与该层 KV 写回。
    /// 该入口只用于逐层闭合，不能签发 whole-token 完成回执。
    pub fn new_gpu_layer_numeric(
        ctx: &'ctx VulkanContext,
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        static_layout: &S14Position0StaticLayerLayout,
        layer_index: usize,
        payload_root: &Path,
        layer_input_bf16le: &[u8],
        hc_aux_f32le: &[u8],
    ) -> Result<Self> {
        let route_weights: [f32; EXPERTS_PER_TOKEN] = manifest
            .layers
            .get(layer_index)
            .ok_or_else(|| anyhow!("manifest 缺少 layer index {layer_index}"))?
            .route_weights
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("L{layer_index} reference route 不是严格 top-6"))?;
        Self::new_gpu_inner(
            ctx,
            manifest,
            weights,
            static_layout,
            layer_index,
            payload_root,
            Some(route_weights),
            Some(layer_input_bf16le),
            Some(hc_aux_f32le),
        )
    }

    fn new_gpu_inner(
        ctx: &'ctx VulkanContext,
        manifest: &Position0WholeTokenManifest,
        weights: &S14Position0HybridWeightPlan,
        static_layout: &S14Position0StaticLayerLayout,
        layer_index: usize,
        payload_root: &Path,
        numeric_route_override: Option<[f32; EXPERTS_PER_TOKEN]>,
        numeric_layer_input: Option<&[u8]>,
        numeric_hc_override: Option<&[u8]>,
    ) -> Result<Self> {
        let graph =
            build_position0_layer_graph_plan(manifest, weights, static_layout, layer_index)?;
        let layer_program =
            S14Position0FullDepthLayerProgram::build(manifest, weights, &graph.workspace)?;
        let state_layout =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, HIDDEN)?;
        let state_recording = S14Position0FullDepthStateRecordingProgram::build(
            &layer_program,
            &graph.workspace,
            &state_layout,
        )?;
        let gpu = S14Position0L0GpuOwner::new(
            ctx,
            manifest,
            weights,
            static_layout,
            &graph,
            payload_root,
            numeric_route_override,
            numeric_layer_input,
            numeric_hc_override,
        )?;
        Ok(Self {
            graph,
            state_recording: Some(state_recording),
            phase: S14Position0L0BackendPhase::Ready,
            base_epoch: None,
            candidate_bank: None,
            gpu: Some(gpu),
        })
    }

    pub fn graph(&self) -> &S14Position0L0GraphPlan {
        &self.graph
    }

    pub fn phase(&self) -> S14Position0L0BackendPhase {
        self.phase
    }

    pub fn verified_payload_stats(&self) -> Option<VerifiedMappedAssetStats> {
        self.gpu.as_ref().map(|gpu| gpu.verified)
    }

    /// 把构造器中冻结的四流 BF16 hidden 放入当前层 workspace。
    /// 与 embedding 入口互斥；只服务逐层数值闭合。
    pub fn submit_numeric_layer_input(
        &mut self,
        bootstrap: &Position0GpuBootstrap<'_>,
    ) -> Result<()> {
        if self.phase != S14Position0L0BackendPhase::Ready {
            bail!("L{} numeric input 调度顺序漂移", self.graph.layer);
        }
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| anyhow!("L{} GPU owner 尚未绑定", self.graph.layer))?;
        if !gpu.numeric_layer_input || !std::ptr::eq(gpu.ctx, bootstrap.candidate.ctx) {
            bail!(
                "L{} numeric input 未绑定或 Vulkan context 漂移",
                self.graph.layer
            );
        }
        let input_slot = if self.graph.layer_index % 2 == 0 {
            S14Position0WorkspaceSlot::HiddenStreamsA
        } else {
            S14Position0WorkspaceSlot::HiddenStreamsB
        };
        let hidden_a = self.graph.workspace.region(input_slot).offset;
        unsafe {
            gpu.ctx.device.cmd_copy_buffer(
                bootstrap.prologue_command,
                gpu.immutable().handle(),
                gpu.workspace().handle(),
                &[vk::BufferCopy::default()
                    .src_offset(AUX_LAYER_INPUT_OFFSET)
                    .dst_offset(hidden_a)
                    .size(AUX_LAYER_INPUT_BYTES)],
            );
            transfer_to_compute_barrier(gpu.ctx, bootstrap.prologue_command);
            gpu.ctx
                .device
                .end_command_buffer(bootstrap.prologue_command)?;
        }
        let compute_value = unsafe {
            gpu.timeline
                .as_mut()
                .expect("layer timeline")
                .submit_compute_only(gpu.ctx, bootstrap.prologue_command)?
        };
        gpu.last_compute_value = compute_value;
        self.base_epoch = Some(bootstrap.candidate.base_epoch);
        self.candidate_bank = Some(bootstrap.candidate.candidate_bank);
        self.phase = S14Position0L0BackendPhase::EmbeddingSubmitted;
        Ok(())
    }

    /// L0 专属数值门。它只 drain 到 L0 timeline，不签发 whole-token 完成回执，
    /// 也不允许上层把该结果当作 43 层或 final token。
    pub fn wait_l0_numeric(
        &mut self,
        candidate: &Position0GpuCandidate<'_>,
    ) -> Result<S14Position0L0NumericReceipt> {
        if self.phase != S14Position0L0BackendPhase::Layer0Submitted
            || self.base_epoch != Some(candidate.base_epoch)
            || self.candidate_bank != Some(candidate.candidate_bank)
        {
            bail!("L0 numeric wait 调度/candidate 身份漂移");
        }
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| anyhow!("L0 GPU owner 尚未绑定"))?;
        if !std::ptr::eq(gpu.ctx, candidate.ctx) || gpu.last_compute_value == 0 {
            bail!("L0 numeric wait Vulkan context/timeline 漂移");
        }
        gpu.timeline.as_ref().expect("L0 timeline").wait_compute(
            gpu.ctx,
            gpu.last_compute_value,
            u64::MAX,
        )?;
        let sticky_status = unsafe { *(candidate.sticky_status.mapped() as *const u32) };
        if sticky_status != 0 {
            bail!("L0 GPU sticky status 非零: 0x{sticky_status:08x}");
        }
        let readback = gpu.readback();
        let route_ids = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_ROUTE_IDS_OFFSET as usize) as *const u32,
                EXPERTS_PER_TOKEN,
            )
        };
        let route_ids: [u32; EXPERTS_PER_TOKEN] = route_ids.try_into()?;
        let expected_ids = self.graph.routed_expert_ids.map(u32::from);
        if route_ids != expected_ids {
            bail!("L0 GPU route IDs 漂移: actual={route_ids:?} expected={expected_ids:?}");
        }
        let route_weights = unsafe {
            std::slice::from_raw_parts(
                readback
                    .mapped()
                    .add(READBACK_ROUTE_WEIGHTS_OFFSET as usize) as *const f32,
                EXPERTS_PER_TOKEN,
            )
        };
        let route_weights: [f32; EXPERTS_PER_TOKEN] = route_weights.try_into()?;
        if route_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
            || (route_weights.iter().sum::<f32>() - 1.5).abs() > 2.0e-5
        {
            bail!("L0 GPU route weights 非法: {route_weights:?}");
        }
        let hidden = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_HIDDEN_OFFSET as usize) as *const u16,
                (4 * HIDDEN) as usize,
            )
        };
        let finite_hidden_elements = hidden
            .iter()
            .filter(|bits| (**bits & 0x7f80) != 0x7f80)
            .count();
        let nonzero_hidden_elements = hidden.iter().filter(|bits| (**bits & 0x7fff) != 0).count();
        if finite_hidden_elements != hidden.len() || nonzero_hidden_elements == 0 {
            bail!(
                "L0 final hidden 非有限/全零: finite={finite_hidden_elements} nonzero={nonzero_hidden_elements} total={}",
                hidden.len()
            );
        }
        let mut hidden_f32_le = Vec::with_capacity(hidden.len() * 4);
        let hidden_bf16_bits = hidden.to_vec();
        let mut hidden_sum = 0.0f64;
        let mut hidden_sum_sq = 0.0f64;
        let mut hidden_max_abs = 0.0f32;
        for bits in hidden {
            let value = f32::from_bits(u32::from(*bits) << 16);
            hidden_f32_le.extend_from_slice(&value.to_le_bytes());
            hidden_sum += f64::from(value);
            hidden_sum_sq += f64::from(value) * f64::from(value);
            hidden_max_abs = hidden_max_abs.max(value.abs());
        }
        let hidden_f32_le_sha256 = format!("{:x}", Sha256::digest(&hidden_f32_le));
        let hidden_l2 = hidden_sum_sq.sqrt();
        let hidden_mean = hidden_sum / hidden.len() as f64;
        let attention_input_bytes = unsafe {
            std::slice::from_raw_parts(
                readback
                    .mapped()
                    .add(READBACK_ATTENTION_INPUT_OFFSET as usize),
                HIDDEN as usize * 4,
            )
        };
        let attention_input_f32_values =
            bytemuck::cast_slice::<u8, f32>(attention_input_bytes).to_vec();
        if attention_input_f32_values
            .iter()
            .any(|value| !value.is_finite())
        {
            bail!("L0 attention input 出现非有限值");
        }
        let attention_input_f32_le_sha256 = format!("{:x}", Sha256::digest(attention_input_bytes));
        let attention_branch_bytes = unsafe {
            std::slice::from_raw_parts(
                readback
                    .mapped()
                    .add(READBACK_ATTENTION_BRANCH_OFFSET as usize),
                HIDDEN as usize * 2,
            )
        };
        let attention_branch_bf16_bits =
            bytemuck::cast_slice::<u8, u16>(attention_branch_bytes).to_vec();
        let attention_branch_bf16_le_sha256 =
            format!("{:x}", Sha256::digest(attention_branch_bytes));
        let post_attention_bytes = unsafe {
            std::slice::from_raw_parts(
                readback
                    .mapped()
                    .add(READBACK_POST_ATTENTION_OFFSET as usize),
                4 * HIDDEN as usize * 2,
            )
        };
        let post_attention_bf16_bits =
            bytemuck::cast_slice::<u8, u16>(post_attention_bytes).to_vec();
        let post_attention_bf16_le_sha256 = format!("{:x}", Sha256::digest(post_attention_bytes));
        let query_final_bf16_bits = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_QUERY_FINAL_OFFSET as usize) as *const u16,
                32_768,
            )
            .to_vec()
        };
        let key_value_final_bf16_bits = unsafe {
            std::slice::from_raw_parts(
                readback
                    .mapped()
                    .add(READBACK_KEY_VALUE_FINAL_OFFSET as usize) as *const u16,
                512,
            )
            .to_vec()
        };
        let attention_output_bf16_bits = unsafe {
            std::slice::from_raw_parts(
                readback
                    .mapped()
                    .add(READBACK_ATTENTION_OUTPUT_OFFSET as usize) as *const u16,
                32_768,
            )
            .to_vec()
        };
        let wo_a_qdq_f32_values = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_WO_A_QDQ_OFFSET as usize) as *const f32,
                8_192,
            )
            .to_vec()
        };
        if wo_a_qdq_f32_values.iter().any(|value| !value.is_finite()) {
            bail!("L0 wo_a QDQ 输出出现非有限值");
        }
        let ffn_input_bytes = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_FFN_INPUT_OFFSET as usize),
                HIDDEN as usize * 4,
            )
        };
        let ffn_input_f32_values = bytemuck::cast_slice::<u8, f32>(ffn_input_bytes).to_vec();
        if ffn_input_f32_values.iter().any(|value| !value.is_finite()) {
            bail!("L0 FFN input 出现非有限值");
        }
        let ffn_input_f32_le_sha256 = format!("{:x}", Sha256::digest(ffn_input_bytes));
        let hc_post_f32_values: [f32; 4] = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_HC_POST_OFFSET as usize) as *const f32,
                4,
            )
        }
        .try_into()?;
        let hc_comb_f32_values: [f32; 16] = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_HC_COMB_OFFSET as usize) as *const f32,
                16,
            )
        }
        .try_into()?;
        let moe_bytes = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_MOE_OFFSET as usize),
                HIDDEN as usize * 4,
            )
        };
        let moe = bytemuck::cast_slice::<u8, f32>(moe_bytes);
        let mut moe_sum = 0.0f64;
        let mut moe_sum_sq = 0.0f64;
        let mut moe_max_abs = 0.0f32;
        let mut moe_bf16_bits = Vec::with_capacity(moe.len());
        for value in moe {
            if !value.is_finite() {
                bail!("L0 MoE accumulator 出现非有限值");
            }
            moe_sum += f64::from(*value);
            moe_sum_sq += f64::from(*value) * f64::from(*value);
            moe_max_abs = moe_max_abs.max(value.abs());
            let bits = value.to_bits();
            let rounded = bits + 0x0000_7fff + ((bits >> 16) & 1);
            moe_bf16_bits.push((rounded >> 16) as u16);
        }
        let moe_f32_le_sha256 = format!("{:x}", Sha256::digest(moe_bytes));
        let moe_bf16_le_sha256 = format!(
            "{:x}",
            Sha256::digest(bytemuck::cast_slice::<u16, u8>(&moe_bf16_bits))
        );
        let moe_l2 = moe_sum_sq.sqrt();
        let moe_mean = moe_sum / moe.len() as f64;
        let workspace_kv = unsafe {
            std::slice::from_raw_parts(readback.mapped().add(READBACK_KV_OFFSET as usize), 1024)
        };
        let candidate_kv = unsafe {
            std::slice::from_raw_parts(
                readback.mapped().add(READBACK_CANDIDATE_KV_OFFSET as usize),
                1024,
            )
        };
        let kv_candidate_exact = workspace_kv == candidate_kv;
        if !kv_candidate_exact {
            bail!("L0 KV candidate writeback 与 workspace 不一致");
        }
        for binder in gpu.binders.drain(..) {
            binder.destroy(gpu.ctx);
        }
        self.phase = S14Position0L0BackendPhase::Drained;
        Ok(S14Position0L0NumericReceipt {
            base_epoch: candidate.base_epoch,
            candidate_bank: candidate.candidate_bank,
            compute_value: gpu.last_compute_value,
            route_ids,
            route_weights,
            finite_hidden_elements,
            nonzero_hidden_elements,
            hidden_bf16_bits,
            hidden_f32_le_sha256,
            hidden_l2,
            hidden_mean,
            hidden_max_abs,
            attention_input_f32_values,
            attention_input_f32_le_sha256,
            attention_branch_bf16_bits,
            attention_branch_bf16_le_sha256,
            post_attention_bf16_bits,
            post_attention_bf16_le_sha256,
            query_final_bf16_bits,
            key_value_final_bf16_bits,
            attention_output_bf16_bits,
            wo_a_qdq_f32_values,
            ffn_input_f32_values,
            ffn_input_f32_le_sha256,
            hc_post_f32_values,
            hc_comb_f32_values,
            moe_f32_le_sha256,
            moe_bf16_bits,
            moe_bf16_le_sha256,
            moe_l2,
            moe_mean,
            moe_max_abs,
            kv_candidate_exact,
            sticky_status,
        })
    }
}

impl Position0LayerBackend for S14Position0L0Backend<'_> {
    fn capabilities(&self) -> Position0BackendCapabilities {
        Position0BackendCapabilities {
            embedding: self.gpu.is_some(),
            all_layers: false,
            final_head: false,
            payload_sha256: self.gpu.is_some(),
            route_receipts: false,
            position0_state_outputs: false,
        }
    }

    fn submit_embedding(
        &mut self,
        bootstrap: &Position0GpuBootstrap<'_>,
        embedding: &Position0Asset,
    ) -> Result<(), Position0BackendError> {
        if self.phase != S14Position0L0BackendPhase::Ready
            || embedding.tensor != "embed.weight[0:1]"
            || embedding.dtype != "BF16"
            || embedding.shape.as_slice() != [1, 4096]
            || embedding.bytes != 8192
            || bootstrap.candidate.committed_host_state.input_token_id
                != S14_POSITION0_L0_INPUT_TOKEN
        {
            return Err(Position0BackendError::Execution(
                "L0 embedding 调度/资产身份漂移".into(),
            ));
        }
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| Position0BackendError::Unavailable("L0 GPU owner 尚未绑定".into()))?;
        if !std::ptr::eq(gpu.ctx, bootstrap.candidate.ctx) {
            return Err(Position0BackendError::Execution(
                "L0 embedding Vulkan context 身份漂移".into(),
            ));
        }
        let shape = S14EmbeddingBroadcastShape::new(HIDDEN).map_err(position0_execution_error)?;
        let dispatch = gpu
            .embedding_pipeline
            .as_ref()
            .expect("L0 embedding pipeline")
            .bind_slices(
                gpu.ctx,
                shape,
                StorageBufferSlice {
                    buffer: gpu.immutable(),
                    offset: AUX_EMBEDDING_OFFSET,
                },
                StorageBufferSlice {
                    buffer: gpu.workspace(),
                    offset: self
                        .graph
                        .workspace
                        .region(S14Position0WorkspaceSlot::HiddenStreamsA)
                        .offset,
                },
                StorageBufferSlice::whole(bootstrap.candidate.sticky_status),
            )
            .map_err(position0_execution_error)?;
        let submit_result = unsafe {
            gpu.embedding_pipeline
                .as_ref()
                .expect("L0 embedding pipeline")
                .cmd(gpu.ctx, bootstrap.prologue_command, &dispatch);
            let result = gpu
                .ctx
                .device
                .end_command_buffer(bootstrap.prologue_command)
                .map_err(anyhow::Error::from)
                .and_then(|_| {
                    gpu.timeline
                        .as_mut()
                        .expect("L0 timeline")
                        .submit_compute_only(gpu.ctx, bootstrap.prologue_command)
                });
            result
        };
        let compute_value = match submit_result {
            Ok(value) => value,
            Err(error) => {
                dispatch.binder.destroy(gpu.ctx);
                return Err(position0_execution_error(error));
            }
        };
        gpu.last_compute_value = compute_value;
        gpu.binders.push(dispatch.binder);
        self.base_epoch = Some(bootstrap.candidate.base_epoch);
        self.candidate_bank = Some(bootstrap.candidate.candidate_bank);
        self.phase = S14Position0L0BackendPhase::EmbeddingSubmitted;
        Ok(())
    }

    fn submit_layer(
        &mut self,
        candidate: &Position0GpuCandidate<'_>,
        layer: &Position0Layer,
    ) -> Result<(), Position0BackendError> {
        if self.phase != S14Position0L0BackendPhase::EmbeddingSubmitted
            || layer.layer != self.graph.layer
            || self.base_epoch != Some(candidate.base_epoch)
            || self.candidate_bank != Some(candidate.candidate_bank)
        {
            return Err(Position0BackendError::Execution(format!(
                "L{} layer 调度顺序或 candidate 身份漂移",
                self.graph.layer
            )));
        }
        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| Position0BackendError::Unavailable("L0 GPU owner 尚未绑定".into()))?;
        if !std::ptr::eq(gpu.ctx, candidate.ctx) {
            return Err(Position0BackendError::Execution(
                "L0 layer Vulkan context 身份漂移".into(),
            ));
        }
        let state_recipe = self
            .state_recording
            .as_ref()
            .and_then(|program| program.layer(self.graph.layer));
        let mut binders = record_layer_command(gpu, &self.graph, candidate, state_recipe, None)
            .map_err(position0_execution_error)?;
        let submit_result = unsafe {
            gpu.timeline
                .as_mut()
                .expect("L0 timeline")
                .submit_compute_only(gpu.ctx, gpu.layer_command)
        };
        let compute_value = match submit_result {
            Ok(value) => value,
            Err(error) => {
                for binder in binders.drain(..) {
                    binder.destroy(gpu.ctx);
                }
                return Err(position0_execution_error(error));
            }
        };
        gpu.last_compute_value = compute_value;
        gpu.binders.extend(binders);
        self.phase = S14Position0L0BackendPhase::Layer0Submitted;
        Ok(())
    }

    fn submit_final(
        &mut self,
        _candidate: &Position0GpuCandidate<'_>,
        _final_section: &Position0Final,
    ) -> Result<(), Position0BackendError> {
        Err(Position0BackendError::Unavailable(
            "L1..L42 与 final head 未闭合".into(),
        ))
    }

    fn wait_candidate(&mut self) -> Result<Position0BackendCompletion, Position0BackendError> {
        Err(Position0BackendError::Unavailable(
            "禁止对 L0-only candidate 签发完成回执".into(),
        ))
    }

    fn finish_receipts(
        &mut self,
        _manifest: &Position0WholeTokenManifest,
    ) -> Result<Position0GraphReceipt, Position0BackendError> {
        Err(Position0BackendError::Unavailable(
            "L0-only 后端没有 43 层 post-fence receipts".into(),
        ))
    }

    fn abort_candidate(&mut self) -> Result<(), Position0BackendError> {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.timeline
                .as_ref()
                .expect("L0 timeline")
                .drain_all(gpu.ctx, u64::MAX)
                .map_err(position0_execution_error)?;
            for binder in gpu.binders.drain(..) {
                binder.destroy(gpu.ctx);
            }
        }
        self.phase = S14Position0L0BackendPhase::Drained;
        Ok(())
    }
}

fn position0_execution_error(error: anyhow::Error) -> Position0BackendError {
    Position0BackendError::Execution(error.to_string())
}

/// 给主代理的明确出口：这个文件接入 `lib.rs` 后可以直接跑的
/// 纯合同门，不读 capture hidden，也不触发 Vulkan 分配。
pub fn build_l0_graph_plan(
    manifest: &Position0WholeTokenManifest,
    weights: &S14Position0HybridWeightPlan,
    static_layout: &S14Position0StaticLayerLayout,
) -> Result<S14Position0L0GraphPlan> {
    if FULL_DEPTH_LAYERS.first().copied() != Some(S14_POSITION0_L0) {
        bail!("FullDepth43 层序不再以 L0 开始");
    }
    S14Position0L0GraphPlan::build(manifest, weights, static_layout)
}

pub fn build_position0_layer_graph_plan(
    manifest: &Position0WholeTokenManifest,
    weights: &S14Position0HybridWeightPlan,
    static_layout: &S14Position0StaticLayerLayout,
    layer_index: usize,
) -> Result<S14Position0L0GraphPlan> {
    S14Position0L0GraphPlan::build_layer(manifest, weights, static_layout, layer_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        s14_position0_hybrid_weight_arena::S14Position0HybridArenaLayout,
        s14_position0_mapped_assets::VerifiedMappedAssetStore,
    };
    use std::path::PathBuf;

    fn real_manifest_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        )
    }

    #[test]
    fn production_position1_through_126_attention_bind_exact_window_and_compressed_modes() {
        let kv_offset = 4096;
        let recipe = S14Position0LayerStateRecordingRecipe {
            layer: 2,
            index: 2,
            position: 1,
            compress_ratio: 4,
            static_layer_bytes: 1,
            workspace_bytes: 1,
            candidate_state_bytes: 16_384,
            committed_window_state_range: kv_offset..kv_offset + 1024,
            window_kv_source_offset: 0,
            window_kv_state_range: kv_offset + 1024..kv_offset + 2048,
            compressor_ops: Vec::new(),
            rollover_copies: Vec::new(),
            state_ranges_written: vec![kv_offset + 1024..kv_offset + 2048],
        };
        let execution = S14PositionExecutionPlan {
            rope_position: 1,
            window_slot: 1,
            active_window_tokens: 2,
            ape_rows: Vec::new(),
        };
        assert_eq!(
            production_attention_mode(
                1,
                4,
                kv_offset,
                4096,
                16_384,
                Some(&recipe),
                Some(&execution),
            )
            .unwrap(),
            S14ProductionAttentionMode::PreCompressionWindow {
                previous_kv_offset: kv_offset,
                previous_count: 1,
                rope_offset: POSITION1_ROPE_YARN_OFFSET,
            }
        );

        let mut drifted = recipe.clone();
        drifted.window_kv_state_range = kv_offset..kv_offset + 1024;
        assert!(production_attention_mode(
            1,
            4,
            kv_offset,
            4096,
            16_384,
            Some(&drifted),
            Some(&execution),
        )
        .is_err());
        let position2_recipe = S14Position0LayerStateRecordingRecipe {
            position: 2,
            committed_window_state_range: kv_offset..kv_offset + 2048,
            window_kv_state_range: kv_offset + 2048..kv_offset + 3072,
            state_ranges_written: vec![kv_offset + 2048..kv_offset + 3072],
            ..recipe.clone()
        };
        let position2_execution = S14PositionExecutionPlan {
            rope_position: 2,
            window_slot: 2,
            active_window_tokens: 3,
            ape_rows: Vec::new(),
        };
        assert_eq!(
            production_attention_mode(
                2,
                4,
                kv_offset,
                4096,
                16_384,
                Some(&position2_recipe),
                Some(&position2_execution),
            )
            .unwrap(),
            S14ProductionAttentionMode::PreCompressionWindow {
                previous_kv_offset: kv_offset,
                previous_count: 2,
                rope_offset: POSITION1_ROPE_YARN_OFFSET,
            }
        );
        assert!(production_attention_mode(
            2,
            4,
            kv_offset,
            4096,
            16_384,
            Some(&recipe),
            Some(&execution),
        )
        .is_err());
        let position3_recipe = S14Position0LayerStateRecordingRecipe {
            position: 3,
            committed_window_state_range: kv_offset..kv_offset + 3072,
            window_kv_state_range: kv_offset + 3072..kv_offset + 4096,
            state_ranges_written: vec![
                kv_offset + 3072..kv_offset + 4096,
                kv_offset + 128 * 512 * 2..kv_offset + 129 * 512 * 2,
            ],
            ..recipe.clone()
        };
        let position3_execution = S14PositionExecutionPlan {
            rope_position: 3,
            window_slot: 3,
            active_window_tokens: 4,
            ape_rows: Vec::new(),
        };
        assert_eq!(
            production_attention_mode(
                3,
                4,
                kv_offset,
                140_000,
                262_144,
                Some(&position3_recipe),
                Some(&position3_execution),
            )
            .unwrap(),
            S14ProductionAttentionMode::Ratio4FirstCompressedBlock {
                previous_kv_offset: kv_offset,
                compressed_kv_offset: kv_offset + 128 * 512 * 2,
                rope_offset: POSITION1_ROPE_YARN_OFFSET,
            }
        );
        for ratio in [0, 128] {
            let mut uncompressed_recipe = position3_recipe.clone();
            uncompressed_recipe.compress_ratio = ratio;
            assert_eq!(
                production_attention_mode(
                    3,
                    ratio,
                    kv_offset,
                    4096,
                    262_144,
                    Some(&uncompressed_recipe),
                    Some(&position3_execution),
                )
                .unwrap(),
                S14ProductionAttentionMode::PreCompressionWindow {
                    previous_kv_offset: kv_offset,
                    previous_count: 3,
                    rope_offset: if ratio == 0 {
                        POSITION1_ROPE_RATIO0_OFFSET
                    } else {
                        POSITION1_ROPE_YARN_OFFSET
                    },
                }
            );
        }
        for (position, compressed_count) in [(4u32, 1u32), (7, 2)] {
            let sparse_recipe = S14Position0LayerStateRecordingRecipe {
                position,
                committed_window_state_range: kv_offset..kv_offset + u64::from(position) * 1024,
                window_kv_state_range: kv_offset + u64::from(position) * 1024
                    ..kv_offset + u64::from(position + 1) * 1024,
                state_ranges_written: vec![
                    kv_offset + u64::from(position) * 1024
                        ..kv_offset + u64::from(position + 1) * 1024,
                ],
                ..recipe.clone()
            };
            let sparse_execution = S14PositionExecutionPlan {
                rope_position: position,
                window_slot: position,
                active_window_tokens: position + 1,
                ape_rows: Vec::new(),
            };
            assert_eq!(
                production_attention_mode(
                    position,
                    4,
                    kv_offset,
                    140_000,
                    262_144,
                    Some(&sparse_recipe),
                    Some(&sparse_execution),
                )
                .unwrap(),
                S14ProductionAttentionMode::Ratio4Sparse {
                    previous_kv_offset: kv_offset,
                    window_start: 0,
                    previous_count: position,
                    compressed_kv_offset: kv_offset + 128 * 512 * 2,
                    compressed_count,
                    rope_offset: POSITION1_ROPE_YARN_OFFSET,
                }
            );
        }
        for position in [4u32, 126] {
            let ratio128_recipe = S14Position0LayerStateRecordingRecipe {
                position,
                compress_ratio: 128,
                committed_window_state_range: kv_offset..kv_offset + u64::from(position) * 1024,
                window_kv_state_range: kv_offset + u64::from(position) * 1024
                    ..kv_offset + u64::from(position + 1) * 1024,
                state_ranges_written: vec![
                    kv_offset + u64::from(position) * 1024
                        ..kv_offset + u64::from(position + 1) * 1024,
                ],
                ..recipe.clone()
            };
            let execution = S14PositionExecutionPlan {
                rope_position: position,
                window_slot: position,
                active_window_tokens: position + 1,
                ape_rows: Vec::new(),
            };
            assert_eq!(
                production_attention_mode(
                    position,
                    128,
                    kv_offset,
                    140_000,
                    262_144,
                    Some(&ratio128_recipe),
                    Some(&execution),
                )
                .unwrap(),
                S14ProductionAttentionMode::PreCompressionWindow {
                    previous_kv_offset: kv_offset,
                    previous_count: position,
                    rope_offset: POSITION1_ROPE_YARN_OFFSET,
                }
            );
        }
        assert!(production_attention_mode(
            4,
            4,
            kv_offset,
            140_000,
            262_144,
            Some(&position3_recipe),
            Some(&position3_execution),
        )
        .is_err());
    }

    #[test]
    fn position127_ratio128_finalize_and_deterministic_attention_target_inactive_abi_row() {
        use crate::s14_ratio128_compressor_finalize::{
            S14Ratio128ProductionFinalizeStage, S14_RATIO128_PRODUCTION_FINALIZE_STAGES,
        };

        assert_eq!(
            S14_RATIO128_PRODUCTION_FINALIZE_STAGES,
            [
                S14Ratio128ProductionFinalizeStage::Pool,
                S14Ratio128ProductionFinalizeStage::RmsNorm,
                S14Ratio128ProductionFinalizeStage::CompressedRope,
                S14Ratio128ProductionFinalizeStage::MainQdq,
                S14Ratio128ProductionFinalizeStage::InactiveStateWrite,
            ]
        );

        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 127;
        let layer = 3;
        let kv = state.kv.iter().find(|entry| entry.layer == layer).unwrap();
        assert_eq!(kv.compress_ratio, 128);
        let target_start = kv.cache.offset + 128 * 512 * 2;
        let target = target_start..target_start + 512 * 2;
        let recipe = S14Position0LayerStateRecordingRecipe {
            layer,
            index: usize::from(layer),
            position: 127,
            compress_ratio: 128,
            static_layer_bytes: 1,
            workspace_bytes: S14_POSITION0_L0_WORKSPACE_BYTES,
            candidate_state_bytes: state.arena_bytes,
            committed_window_state_range: kv.cache.offset..kv.cache.offset + 127 * 1024,
            window_kv_source_offset: 0,
            window_kv_state_range: kv.cache.offset + 127 * 1024..kv.cache.offset + 128 * 1024,
            compressor_ops: Vec::new(),
            rollover_copies: Vec::new(),
            state_ranges_written: vec![target.clone()],
        };
        let bindings = ratio128_boundary_state_bindings(&state, layer, &recipe).unwrap();
        assert_eq!(bindings.boundary.cache_index, 0);
        assert_eq!(bindings.boundary.compressed_position, 0);
        assert_eq!(bindings.main_compressed_kv, target.start);
        let compressor = state
            .compressors
            .iter()
            .find(|entry| entry.layer == layer && entry.compress_ratio == 128)
            .unwrap();
        assert_eq!(bindings.main_kv_state, compressor.kv_state.offset);
        assert_eq!(bindings.main_score_state, compressor.score_state.offset);

        let mut missing_dirty_target = recipe.clone();
        missing_dirty_target.state_ranges_written.clear();
        assert!(ratio128_boundary_state_bindings(&state, layer, &missing_dirty_target).is_err());

        let execution = S14PositionExecutionPlan {
            rope_position: 127,
            window_slot: 127,
            active_window_tokens: 128,
            ape_rows: Vec::new(),
        };
        assert_eq!(
            production_attention_mode(
                127,
                128,
                kv.cache.offset,
                kv.cache.bytes,
                state.arena_bytes,
                Some(&recipe),
                Some(&execution),
            )
            .unwrap(),
            S14ProductionAttentionMode::Ratio128Deterministic {
                previous_kv_offset: kv.cache.offset,
                window_start: 0,
                previous_count: 127,
                compressed_kv_offset: target.start,
                compressed_count: 1,
                rope_offset: POSITION1_ROPE_YARN_OFFSET,
            }
        );
        assert!(validate_production_layer_position(127).is_ok());
        assert!(validate_production_layer_position(128).is_ok());
        assert!(validate_production_layer_position(254).is_ok());
        assert!(validate_production_layer_position(255).is_ok());
        assert!(validate_production_layer_position(2050).is_ok());
        assert!(validate_production_layer_position(2051).is_ok());
        assert!(validate_production_layer_position(2052).is_err());
    }

    #[test]
    fn position128_ring_wrap_uses_rows1_to127_current_row0_and_existing_compressed_blocks() {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 128;
        let execution = S14PositionExecutionPlan {
            rope_position: 128,
            window_slot: 0,
            active_window_tokens: 128,
            ape_rows: Vec::new(),
        };

        for (layer, ratio) in [(0u8, 0u16), (2, 4), (3, 128)] {
            let kv = state.kv.iter().find(|entry| entry.layer == layer).unwrap();
            assert_eq!(kv.compress_ratio, ratio);
            let recipe = S14Position0LayerStateRecordingRecipe {
                layer,
                index: usize::from(layer),
                position: 128,
                compress_ratio: ratio,
                static_layer_bytes: 1,
                workspace_bytes: S14_POSITION0_L0_WORKSPACE_BYTES,
                candidate_state_bytes: state.arena_bytes,
                committed_window_state_range: kv.cache.offset..kv.cache.offset + 128 * 1024,
                window_kv_source_offset: 0,
                window_kv_state_range: kv.cache.offset..kv.cache.offset + 1024,
                compressor_ops: Vec::new(),
                rollover_copies: Vec::new(),
                state_ranges_written: vec![kv.cache.offset..kv.cache.offset + 1024],
            };
            let mode = production_attention_mode(
                128,
                ratio,
                kv.cache.offset,
                kv.cache.bytes,
                state.arena_bytes,
                Some(&recipe),
                Some(&execution),
            )
            .unwrap();
            match (ratio, mode) {
                (
                    0,
                    S14ProductionAttentionMode::RingWindow {
                        window_start,
                        previous_count,
                        rope_offset,
                        ..
                    },
                ) => {
                    assert_eq!((window_start, previous_count), (1, 127));
                    assert_eq!(rope_offset, POSITION1_ROPE_RATIO0_OFFSET);
                }
                (
                    4,
                    S14ProductionAttentionMode::Ratio4Sparse {
                        window_start,
                        previous_count,
                        compressed_count,
                        rope_offset,
                        ..
                    },
                ) => {
                    assert_eq!(
                        (window_start, previous_count, compressed_count),
                        (1, 127, 32)
                    );
                    assert_eq!(rope_offset, POSITION1_ROPE_YARN_OFFSET);
                }
                (
                    128,
                    S14ProductionAttentionMode::Ratio128Deterministic {
                        window_start,
                        previous_count,
                        compressed_count,
                        rope_offset,
                        ..
                    },
                ) => {
                    assert_eq!(
                        (window_start, previous_count, compressed_count),
                        (1, 127, 1)
                    );
                    assert_eq!(rope_offset, POSITION1_ROPE_YARN_OFFSET);
                }
                other => panic!("position128 attention mode漂移: {other:?}"),
            }
        }
        assert!(validate_production_layer_position(128).is_ok());
        assert!(validate_production_layer_position(254).is_ok());
        assert!(validate_production_layer_position(255).is_ok());
        assert!(validate_production_layer_position(2050).is_ok());
        assert!(validate_production_layer_position(2051).is_ok());
        assert!(validate_production_layer_position(2052).is_err());
    }

    #[test]
    fn position254_ring_cycle_end_uses_rows127_to125_current_row126_and_existing_blocks() {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 254;
        let execution = S14PositionExecutionPlan {
            rope_position: 254,
            window_slot: 126,
            active_window_tokens: 128,
            ape_rows: Vec::new(),
        };

        for (layer, ratio) in [(0u8, 0u16), (2, 4), (3, 128)] {
            let kv = state.kv.iter().find(|entry| entry.layer == layer).unwrap();
            assert_eq!(kv.compress_ratio, ratio);
            let candidate_start = kv.cache.offset + 126 * 1024;
            let recipe = S14Position0LayerStateRecordingRecipe {
                layer,
                index: usize::from(layer),
                position: 254,
                compress_ratio: ratio,
                static_layer_bytes: 1,
                workspace_bytes: S14_POSITION0_L0_WORKSPACE_BYTES,
                candidate_state_bytes: state.arena_bytes,
                committed_window_state_range: kv.cache.offset..kv.cache.offset + 128 * 1024,
                window_kv_source_offset: 0,
                window_kv_state_range: candidate_start..candidate_start + 1024,
                compressor_ops: Vec::new(),
                rollover_copies: Vec::new(),
                state_ranges_written: vec![candidate_start..candidate_start + 1024],
            };
            let mode = production_attention_mode(
                254,
                ratio,
                kv.cache.offset,
                kv.cache.bytes,
                state.arena_bytes,
                Some(&recipe),
                Some(&execution),
            )
            .unwrap();
            match (ratio, mode) {
                (
                    0,
                    S14ProductionAttentionMode::RingWindow {
                        window_start,
                        previous_count,
                        rope_offset,
                        ..
                    },
                ) => {
                    assert_eq!((window_start, previous_count), (127, 127));
                    assert_eq!(rope_offset, POSITION1_ROPE_RATIO0_OFFSET);
                }
                (
                    4,
                    S14ProductionAttentionMode::Ratio4Sparse {
                        window_start,
                        previous_count,
                        compressed_count,
                        rope_offset,
                        ..
                    },
                ) => {
                    assert_eq!(
                        (window_start, previous_count, compressed_count),
                        (127, 127, 63)
                    );
                    assert_eq!(rope_offset, POSITION1_ROPE_YARN_OFFSET);
                }
                (
                    128,
                    S14ProductionAttentionMode::Ratio128Deterministic {
                        window_start,
                        previous_count,
                        compressed_count,
                        rope_offset,
                        ..
                    },
                ) => {
                    assert_eq!(
                        (window_start, previous_count, compressed_count),
                        (127, 127, 1)
                    );
                    assert_eq!(rope_offset, POSITION1_ROPE_YARN_OFFSET);
                }
                other => panic!("position254 attention mode漂移: {other:?}"),
            }
        }
    }

    #[test]
    fn position255_finalizes_ratio4_block63_and_ratio128_block1_before_attention() {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 255;
        let execution = S14PositionExecutionPlan {
            rope_position: 255,
            window_slot: 127,
            active_window_tokens: 128,
            ape_rows: Vec::new(),
        };

        for (layer, ratio) in [(0u8, 0u16), (2, 4), (3, 128)] {
            let kv = state.kv.iter().find(|entry| entry.layer == layer).unwrap();
            assert_eq!(kv.compress_ratio, ratio);
            let candidate_start = kv.cache.offset + 127 * 1024;
            let mut state_ranges_written = vec![candidate_start..candidate_start + 1024];
            if ratio == 4 {
                let indexer = state
                    .indexers
                    .iter()
                    .find(|entry| entry.layer == layer)
                    .unwrap();
                state_ranges_written
                    .push(kv.cache.offset + 191 * 1024..kv.cache.offset + 192 * 1024);
                state_ranges_written
                    .push(indexer.kv_cache.offset + 63 * 256..indexer.kv_cache.offset + 64 * 256);
            } else if ratio == 128 {
                state_ranges_written
                    .push(kv.cache.offset + 129 * 1024..kv.cache.offset + 130 * 1024);
            }
            let recipe = S14Position0LayerStateRecordingRecipe {
                layer,
                index: usize::from(layer),
                position: 255,
                compress_ratio: ratio,
                static_layer_bytes: 1,
                workspace_bytes: S14_POSITION0_L0_WORKSPACE_BYTES,
                candidate_state_bytes: state.arena_bytes,
                committed_window_state_range: kv.cache.offset..kv.cache.offset + 128 * 1024,
                window_kv_source_offset: 0,
                window_kv_state_range: candidate_start..candidate_start + 1024,
                compressor_ops: Vec::new(),
                rollover_copies: Vec::new(),
                state_ranges_written,
            };

            if ratio == 4 {
                let bindings = ratio4_boundary_state_bindings(&state, layer, &recipe).unwrap();
                let indexer = state
                    .indexers
                    .iter()
                    .find(|entry| entry.layer == layer)
                    .unwrap();
                assert_eq!(bindings.main_compressed_kv, kv.cache.offset + 191 * 1024);
                assert_eq!(
                    bindings.indexer_compressed_kv,
                    indexer.kv_cache.offset + 63 * 256
                );
            } else if ratio == 128 {
                let bindings = ratio128_boundary_state_bindings(&state, layer, &recipe).unwrap();
                assert_eq!(bindings.boundary.cache_index, 1);
                assert_eq!(bindings.boundary.compressed_position, 128);
                assert_eq!(bindings.main_compressed_kv, kv.cache.offset + 129 * 1024);
            }

            let mode = production_attention_mode(
                255,
                ratio,
                kv.cache.offset,
                kv.cache.bytes,
                state.arena_bytes,
                Some(&recipe),
                Some(&execution),
            )
            .unwrap();
            match (ratio, mode) {
                (
                    0,
                    S14ProductionAttentionMode::RingWindow {
                        window_start,
                        previous_count,
                        ..
                    },
                ) => assert_eq!((window_start, previous_count), (0, 127)),
                (
                    4,
                    S14ProductionAttentionMode::Ratio4Sparse {
                        window_start,
                        previous_count,
                        compressed_count,
                        ..
                    },
                ) => assert_eq!(
                    (window_start, previous_count, compressed_count),
                    (0, 127, 64)
                ),
                (
                    128,
                    S14ProductionAttentionMode::Ratio128Deterministic {
                        window_start,
                        previous_count,
                        compressed_count,
                        ..
                    },
                ) => assert_eq!(
                    (window_start, previous_count, compressed_count),
                    (0, 127, 2)
                ),
                other => panic!("position255 attention/finalize mode漂移: {other:?}"),
            }
        }
        assert!(validate_production_layer_position(255).is_ok());
        assert!(validate_production_layer_position(256).is_ok());
        assert!(validate_production_layer_position(2050).is_ok());
        assert!(validate_production_layer_position(2051).is_ok());
        assert!(validate_production_layer_position(2052).is_err());
    }

    #[test]
    fn position2047_finalizes_last_fixed_ratio4_block_and_ratio128_block15() {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 2047;
        let execution = S14PositionExecutionPlan {
            rope_position: 2047,
            window_slot: 127,
            active_window_tokens: 128,
            ape_rows: Vec::new(),
        };

        for (layer, ratio) in [(2u8, 4u16), (3, 128)] {
            let kv = state.kv.iter().find(|entry| entry.layer == layer).unwrap();
            let candidate_start = kv.cache.offset + 127 * 1024;
            let mut state_ranges_written = vec![candidate_start..candidate_start + 1024];
            if ratio == 4 {
                let indexer = state
                    .indexers
                    .iter()
                    .find(|entry| entry.layer == layer)
                    .unwrap();
                state_ranges_written
                    .push(kv.cache.offset + 639 * 1024..kv.cache.offset + 640 * 1024);
                state_ranges_written
                    .push(indexer.kv_cache.offset + 511 * 256..indexer.kv_cache.offset + 512 * 256);
            } else {
                state_ranges_written
                    .push(kv.cache.offset + 143 * 1024..kv.cache.offset + 144 * 1024);
            }
            let recipe = S14Position0LayerStateRecordingRecipe {
                layer,
                index: usize::from(layer),
                position: 2047,
                compress_ratio: ratio,
                static_layer_bytes: 1,
                workspace_bytes: S14_POSITION0_L0_WORKSPACE_BYTES,
                candidate_state_bytes: state.arena_bytes,
                committed_window_state_range: kv.cache.offset..kv.cache.offset + 128 * 1024,
                window_kv_source_offset: 0,
                window_kv_state_range: candidate_start..candidate_start + 1024,
                compressor_ops: Vec::new(),
                rollover_copies: Vec::new(),
                state_ranges_written,
            };

            if ratio == 4 {
                let bindings = ratio4_boundary_state_bindings(&state, layer, &recipe).unwrap();
                assert_eq!(bindings.main_compressed_kv, kv.cache.offset + 639 * 1024);
            } else {
                let bindings = ratio128_boundary_state_bindings(&state, layer, &recipe).unwrap();
                assert_eq!(bindings.boundary.cache_index, 15);
                assert_eq!(bindings.boundary.compressed_position, 1920);
                assert_eq!(bindings.main_compressed_kv, kv.cache.offset + 143 * 1024);
            }

            match (
                ratio,
                production_attention_mode(
                    2047,
                    ratio,
                    kv.cache.offset,
                    kv.cache.bytes,
                    state.arena_bytes,
                    Some(&recipe),
                    Some(&execution),
                )
                .unwrap(),
            ) {
                (
                    4,
                    S14ProductionAttentionMode::Ratio4Sparse {
                        window_start,
                        previous_count,
                        compressed_count,
                        ..
                    },
                ) => assert_eq!(
                    (window_start, previous_count, compressed_count),
                    (0, 127, 512)
                ),
                (
                    128,
                    S14ProductionAttentionMode::Ratio128Deterministic {
                        window_start,
                        previous_count,
                        compressed_count,
                        ..
                    },
                ) => assert_eq!(
                    (window_start, previous_count, compressed_count),
                    (0, 127, 16)
                ),
                other => panic!("position2047 fixed-cache boundary漂移: {other:?}"),
            }
        }
    }

    #[test]
    fn position2050_consumes_all_512_ratio4_blocks_and_rejects_block513_position() {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 2050;
        let execution = S14PositionExecutionPlan {
            rope_position: 2050,
            window_slot: 2,
            active_window_tokens: 128,
            ape_rows: Vec::new(),
        };

        for (layer, ratio) in [(0u8, 0u16), (2, 4), (3, 128)] {
            let kv = state.kv.iter().find(|entry| entry.layer == layer).unwrap();
            let candidate_start = kv.cache.offset + 2 * 1024;
            let recipe = S14Position0LayerStateRecordingRecipe {
                layer,
                index: usize::from(layer),
                position: 2050,
                compress_ratio: ratio,
                static_layer_bytes: 1,
                workspace_bytes: S14_POSITION0_L0_WORKSPACE_BYTES,
                candidate_state_bytes: state.arena_bytes,
                committed_window_state_range: kv.cache.offset..kv.cache.offset + 128 * 1024,
                window_kv_source_offset: 0,
                window_kv_state_range: candidate_start..candidate_start + 1024,
                compressor_ops: Vec::new(),
                rollover_copies: Vec::new(),
                state_ranges_written: vec![candidate_start..candidate_start + 1024],
            };
            let mode = production_attention_mode(
                2050,
                ratio,
                kv.cache.offset,
                kv.cache.bytes,
                state.arena_bytes,
                Some(&recipe),
                Some(&execution),
            )
            .unwrap();
            match (ratio, mode) {
                (
                    0,
                    S14ProductionAttentionMode::RingWindow {
                        window_start,
                        previous_count,
                        ..
                    },
                ) => assert_eq!((window_start, previous_count), (3, 127)),
                (
                    4,
                    S14ProductionAttentionMode::Ratio4Sparse {
                        window_start,
                        previous_count,
                        compressed_count,
                        ..
                    },
                ) => assert_eq!(
                    (window_start, previous_count, compressed_count),
                    (3, 127, 512)
                ),
                (
                    128,
                    S14ProductionAttentionMode::Ratio128Deterministic {
                        window_start,
                        previous_count,
                        compressed_count,
                        ..
                    },
                ) => assert_eq!(
                    (window_start, previous_count, compressed_count),
                    (3, 127, 16)
                ),
                other => panic!("position2050 fixed-cache tail漂移: {other:?}"),
            }
        }
        assert!(validate_production_layer_position(2050).is_ok());
        assert!(validate_production_layer_position(2051).is_ok());
        assert!(validate_production_layer_position(2052).is_err());
    }

    #[test]
    fn repeated_ratio4_boundaries_use_distinct_cache_rows_and_compressed_rope() {
        assert_eq!(ratio4_compressed_cache_rows(3).unwrap(), (128, 0));
        assert_eq!(ratio4_compressed_cache_rows(7).unwrap(), (129, 1));
        assert_eq!(ratio4_compressed_cache_rows(11).unwrap(), (130, 2));
        assert_eq!(ratio4_compressed_cache_rows(123).unwrap(), (158, 30));
        assert_eq!(ratio4_compressed_cache_rows(127).unwrap(), (159, 31));
        assert_eq!(ratio4_compressed_cache_rows(251).unwrap(), (190, 62));
        assert_eq!(ratio4_compressed_cache_rows(255).unwrap(), (191, 63));
        assert_eq!(ratio4_compressed_cache_rows(2047).unwrap(), (639, 511));
        assert_eq!(ratio4_compressed_cache_rows(2051).unwrap(), (640, 512));

        for position in [0, 4, 6, 126, 254, 256, 2050, 2052] {
            assert!(ratio4_compressed_cache_rows(position).is_err());
        }

        let first = S14Ratio4CompressorBoundary::new(3)
            .unwrap()
            .rope_cos_sin()
            .unwrap();
        let second = S14Ratio4CompressorBoundary::new(7)
            .unwrap()
            .rope_cos_sin()
            .unwrap();
        assert_ne!(first, second, "position7不得复用position3 compressed RoPE");
    }

    #[test]
    fn position2051_ratio4_finalize_binds_second_history_page_row0() {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 2051;
        let kv = state.kv.iter().find(|entry| entry.layer == 2).unwrap();
        let indexer = state
            .indexers
            .iter()
            .find(|entry| entry.layer == 2)
            .unwrap();
        let main_target = kv.cache.offset + 640 * 1024..kv.cache.offset + 641 * 1024;
        let indexer_target =
            indexer.kv_cache.offset + 512 * 256..indexer.kv_cache.offset + 513 * 256;
        let recipe = S14Position0LayerStateRecordingRecipe {
            layer: 2,
            index: 2,
            position: 2051,
            compress_ratio: 4,
            static_layer_bytes: 1,
            workspace_bytes: 1,
            candidate_state_bytes: state.arena_bytes,
            committed_window_state_range: kv.cache.offset..kv.cache.offset + 128 * 1024,
            window_kv_source_offset: 0,
            window_kv_state_range: kv.cache.offset + 3 * 1024..kv.cache.offset + 4 * 1024,
            compressor_ops: Vec::new(),
            rollover_copies: Vec::new(),
            state_ranges_written: vec![main_target.clone(), indexer_target.clone()],
        };
        let bindings = ratio4_boundary_state_bindings(&state, 2, &recipe).unwrap();
        assert_eq!(bindings.main_compressed_kv, main_target.start);
        assert_eq!(bindings.indexer_compressed_kv, indexer_target.start);
    }

    #[test]
    fn position2051_selects_two_page_ratio4_production_attention_contract() {
        let mut state =
            NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096).unwrap();
        state.position = 2051;
        let kv = state.kv.iter().find(|entry| entry.layer == 2).unwrap();
        let recipe = S14Position0LayerStateRecordingRecipe {
            layer: 2,
            index: 2,
            position: 2051,
            compress_ratio: 4,
            static_layer_bytes: 1,
            workspace_bytes: S14_POSITION0_L0_WORKSPACE_BYTES,
            candidate_state_bytes: state.arena_bytes,
            committed_window_state_range: kv.cache.offset..kv.cache.offset + 128 * 1024,
            window_kv_source_offset: 0,
            window_kv_state_range: kv.cache.offset + 3 * 1024..kv.cache.offset + 4 * 1024,
            compressor_ops: Vec::new(),
            rollover_copies: Vec::new(),
            state_ranges_written: Vec::new(),
        };
        let execution = S14PositionExecutionPlan {
            rope_position: 2051,
            window_slot: 3,
            active_window_tokens: 128,
            ape_rows: Vec::new(),
        };
        assert_eq!(
            production_attention_mode(
                2051,
                4,
                kv.cache.offset,
                kv.cache.bytes,
                state.arena_bytes,
                Some(&recipe),
                Some(&execution),
            )
            .unwrap(),
            S14ProductionAttentionMode::Ratio4PagedSparse {
                previous_kv_offset: kv.cache.offset,
                window_start: 4,
                previous_count: 127,
                compressed_kv_offset: kv.cache.offset + 128 * 1024,
                logical_compressed_count: 513,
                selected_count: 512,
                rope_offset: POSITION1_ROPE_YARN_OFFSET,
            }
        );
        assert!(validate_production_layer_position(2051).is_ok());
        assert!(validate_production_layer_position(2052).is_err());
    }

    #[test]
    fn position1_tid2eid_uses_current_input_token_row_and_fails_closed() {
        let manifest = Position0WholeTokenManifest::load(&real_manifest_path()).unwrap();
        let weights = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let physical = S14Position0HybridArenaLayout::build(&weights).unwrap();
        let graph = build_l0_graph_plan(&manifest, &weights, &physical.static_layers[0]).unwrap();
        let asset = manifest.layers[0]
            .assets
            .router
            .iter()
            .find(|asset| asset.tensor == "layers.0.ffn.gate.tid2eid")
            .unwrap();
        let mut payload = vec![0u8; usize::try_from(asset.bytes).unwrap()];
        let position1_token = 5u32;
        let expected = [17u32, 3, 255, 42, 99, 128];
        let row_bytes = EXPERTS_PER_TOKEN * std::mem::size_of::<i64>();
        let row_start = usize::try_from(position1_token).unwrap() * row_bytes;
        for (slot, expert) in expected.iter().copied().enumerate() {
            payload[row_start + slot * 8..row_start + (slot + 1) * 8]
                .copy_from_slice(&i64::from(expert).to_le_bytes());
        }

        assert_eq!(
            graph
                .decode_tid2eid_row_for_token(asset, &payload, position1_token)
                .unwrap(),
            expected
        );
        assert!(graph
            .decode_tid2eid_row_for_token(asset, &payload, 129_280)
            .is_err());

        payload[row_start + 8..row_start + 16]
            .copy_from_slice(&i64::from(expected[0]).to_le_bytes());
        assert!(graph
            .decode_tid2eid_row_for_token(asset, &payload, position1_token)
            .is_err());
    }

    /// 这是本机真实权重门：手动执行 `--ignored` 时会对 tid2eid
    /// payload 重做 SHA-256，然后解码 BOS 行。它不使用 capture route。
    #[test]
    #[ignore = "requires D:/models/Polaris-S14 real payloads"]
    fn real_l0_plan_closes_bos_tid2eid_and_routed_pages() {
        let manifest = Position0WholeTokenManifest::load(&real_manifest_path()).unwrap();
        let weights = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let physical = S14Position0HybridArenaLayout::build(&weights).unwrap();
        let graph = build_l0_graph_plan(&manifest, &weights, &physical.static_layers[0]).unwrap();
        let tid2eid = manifest.layers[0]
            .assets
            .router
            .iter()
            .find(|asset| asset.tensor == "layers.0.ffn.gate.tid2eid")
            .unwrap();
        let mut store = VerifiedMappedAssetStore::new(
            PathBuf::from("D:/models/Polaris-S14/range_cache").as_path(),
        )
        .unwrap();
        let mapped = store
            .map_verified_batch(std::slice::from_ref(tid2eid))
            .unwrap();
        let ids = graph
            .decode_and_validate_tid2eid_bos(tid2eid, mapped[0].bytes())
            .unwrap();
        assert_eq!(ids, [254, 222, 245, 200, 53, 35]);
        assert_eq!(store.stats().sha256_bytes, tid2eid.bytes);
        assert_eq!(graph.routed_metadata.len(), EXPERTS_PER_TOKEN);
        for metadata in graph.routed_metadata {
            assert!(metadata.words().into_iter().all(|offset| offset % 256 == 0));
        }
    }

    #[test]
    fn partial_backend_stays_fail_closed() {
        let manifest = Position0WholeTokenManifest::load(&real_manifest_path()).unwrap();
        let weights = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let physical = S14Position0HybridArenaLayout::build(&weights).unwrap();
        let graph = build_l0_graph_plan(&manifest, &weights, &physical.static_layers[0]).unwrap();
        let backend = S14Position0L0Backend::new(graph);
        assert_eq!(backend.phase(), S14Position0L0BackendPhase::Ready);
        assert_eq!(
            backend.capabilities(),
            Position0BackendCapabilities {
                embedding: false,
                all_layers: false,
                final_head: false,
                payload_sha256: false,
                route_receipts: false,
                position0_state_outputs: false,
            }
        );
    }

    #[test]
    fn all_43_layer_graphs_share_the_verified_numeric_contract() {
        let manifest = Position0WholeTokenManifest::load(&real_manifest_path()).unwrap();
        let weights = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let physical = S14Position0HybridArenaLayout::build(&weights).unwrap();
        let graphs = physical
            .static_layers
            .iter()
            .enumerate()
            .map(|(index, layout)| {
                build_position0_layer_graph_plan(&manifest, &weights, layout, index).unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(graphs.len(), FULL_DEPTH_LAYERS.len());
        for (index, graph) in graphs.iter().enumerate() {
            assert_eq!(graph.layer, FULL_DEPTH_LAYERS[index]);
            assert_eq!(graph.layer_index, index);
            assert_eq!(
                graph.routed_expert_ids,
                manifest.layers[index].expert_ids.as_slice()
            );
            assert_eq!(graph.routed_metadata.len(), EXPERTS_PER_TOKEN);
            assert!(graph
                .routed_metadata
                .iter()
                .all(|entry| entry.words().into_iter().all(|offset| offset % 256 == 0)));
            let expected_mode = if graph.layer < 3 {
                S14RoutePostprocessGpuMode::PhysicalIds
            } else {
                S14RoutePostprocessGpuMode::BiasTop6
            };
            assert_eq!(graph.route_mode, expected_mode);
        }
    }
}
