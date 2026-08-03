//! S14 causal-block HC/QKV/attention/router 的 production K-row Vulkan 数据面。
//!
//! 本 recorder 直接消费正式 `[K,4,4096]` BF16 hidden，并在一个 command buffer、一次
//! queue submit 中录制 HC-pre、共享权重 FP8 QKV、K-lane attention、wo、HC-post、FFN
//! HC-pre 与 router。它没有逐 token 入口；K 只允许 4/8。

use crate::{
    compute::{ComputePipeline, DescriptorBinder, StorageBufferSlice},
    s14_bf16_rmsnorm::{S14Bf16RmsNormPipeline, S14Bf16RmsNormShape},
    s14_bf16_to_f32::{S14Bf16ToF32Pipeline, S14Bf16ToF32Shape},
    s14_causal_block_attention_router::{
        S14CausalBlockAttentionRouterBindings, S14CausalBlockAttentionRouterRecorder,
        S14CausalBlockAttentionRouterShape, S14_CAUSAL_BLOCK_ATTENTION_HEADS,
        S14_CAUSAL_BLOCK_ATTENTION_HEAD_DIM, S14_CAUSAL_BLOCK_ROUTER_EXPERTS,
        S14_CAUSAL_BLOCK_ROUTER_HIDDEN, S14_CAUSAL_BLOCK_ROUTER_TOP_K,
    },
    s14_causal_block_hc_qkv_adapter::{
        S14CausalBlockHcQkvLayerRecorder, S14CausalBlockHcQkvLayerRecordingReceipt,
        S14CausalBlockHcQkvRecordedLayer,
    },
    s14_causal_block_layer::{
        S14CausalBlockAttentionRouterOutput, S14CausalBlockHiddenBinding, S14CausalBlockLayerInput,
        S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
    },
    s14_e4m3_qdq::{S14E4m3QdqPipeline, S14E4m3QdqShape},
    s14_f32_to_bf16::{S14F32ToBf16Pipeline, S14F32ToBf16Shape},
    s14_position0_layer_backend::S14Position0L0GraphPlan,
    s14_position0_paged_weight_arena::{
        S14Position0PagedWeightArena, S14Position0StaticLayerBinding,
    },
    s14_route_postprocess_gpu::S14RoutePostprocessGpuMode,
    s14_vulkan::{S14F32MatvecShape, S14NumericPipelines},
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{router_kind_for_layer, RouteDecision, FULL_DEPTH_LAYERS};
use std::{cell::RefCell, fmt, sync::Arc};

const HIDDEN: u32 = 4096;
const HC_FLAT: u32 = 4 * HIDDEN;
const Q_LOW: u32 = 1024;
const QUERY: u32 = 32_768;
const KV: u32 = 512;
const WO_GROUPS: u32 = 8;
const WO_GROUP_OUT: u32 = 1024;
const WO_A: u32 = WO_GROUPS * WO_GROUP_OUT;
const HC_MIXES: u32 = 24;
const HC_AUX: u32 = 20;
const NORM_EPS: f32 = 1.0e-6;
const ALIGN: u64 = 256;
const MAX_BATCH: u32 = 8;
const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;

pub const S14_CAUSAL_BLOCK_FP8_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_causal_block_fp8_matvec.spv"));
pub const S14_CAUSAL_BLOCK_FP8_MATVEC_EXACT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_causal_block_fp8_matvec_exact.spv"
));
pub const S14_CAUSAL_BLOCK_GROUPED_WO_A_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_causal_block_grouped_wo_a.spv"
));
pub const S14_CAUSAL_BLOCK_HC_NORMALIZE_INPUT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_causal_block_hc_normalize_input.spv"
));
pub const S14_CAUSAL_BLOCK_HC_SPLIT_REDUCE_NORM_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_causal_block_hc_split_reduce_norm.spv"
));
pub const S14_CAUSAL_BLOCK_HC_POST_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_causal_block_hc_post.spv"));
pub const S14_CAUSAL_BLOCK_KV_FINALIZE_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_causal_block_kv_finalize.spv"
));
pub const S14_Q_HEAD_NORMALIZE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_bf16_q_head_normalize.spv"));

#[derive(Clone)]
pub struct S14CausalBlockOwnedBufferSlice {
    pub buffer: Arc<GpuBuffer>,
    pub offset: u64,
    pub bytes: u64,
}

impl fmt::Debug for S14CausalBlockOwnedBufferSlice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockOwnedBufferSlice")
            .field("buffer", &self.buffer.handle())
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl S14CausalBlockOwnedBufferSlice {
    fn storage(&self) -> StorageBufferSlice<'_> {
        StorageBufferSlice {
            buffer: &self.buffer,
            offset: self.offset,
        }
    }

    fn validate_exact(&self, expected: u64, label: &str) -> Result<()> {
        let end = self
            .offset
            .checked_add(self.bytes)
            .context("slice range overflow")?;
        if self.bytes != expected || end > self.buffer.size() || self.offset % 4 != 0 {
            bail!(
                "causal-block HC/QKV {label} slice 非法: offset={} bytes={} expected={} capacity={}",
                self.offset,
                self.bytes,
                expected,
                self.buffer.size()
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockHcQkvWeightOffsets {
    pub hc_attn_fn: u64,
    pub hc_attn_scale: u64,
    pub hc_attn_base: u64,
    pub attn_norm: u64,
    pub wq_a_weight: u64,
    pub wq_a_scale: u64,
    pub q_norm: u64,
    pub wq_b_weight: u64,
    pub wq_b_scale: u64,
    pub wkv_weight: u64,
    pub wkv_scale: u64,
    pub kv_norm: u64,
    pub attention_sink: u64,
    pub wo_a_weight: u64,
    pub wo_a_scale: u64,
    pub wo_b_weight: u64,
    pub wo_b_scale: u64,
    pub hc_ffn_fn: u64,
    pub hc_ffn_scale: u64,
    pub hc_ffn_base: u64,
    pub ffn_norm: u64,
    pub router_weight: u64,
}

impl S14CausalBlockHcQkvWeightOffsets {
    pub fn from_position0_graph(graph: &S14Position0L0GraphPlan) -> Result<Self> {
        Ok(Self {
            hc_attn_fn: graph.static_offset_suffix("hc_attn_fn")?,
            hc_attn_scale: graph.static_offset_suffix("hc_attn_scale")?,
            hc_attn_base: graph.static_offset_suffix("hc_attn_base")?,
            attn_norm: graph.static_offset_suffix("attn_norm.weight")?,
            wq_a_weight: graph.static_offset_suffix("attn.wq_a.weight")?,
            wq_a_scale: graph.static_offset_suffix("attn.wq_a.scale")?,
            q_norm: graph.static_offset_suffix("attn.q_norm.weight")?,
            wq_b_weight: graph.static_offset_suffix("attn.wq_b.weight")?,
            wq_b_scale: graph.static_offset_suffix("attn.wq_b.scale")?,
            wkv_weight: graph.static_offset_suffix("attn.wkv.weight")?,
            wkv_scale: graph.static_offset_suffix("attn.wkv.scale")?,
            kv_norm: graph.static_offset_suffix("attn.kv_norm.weight")?,
            attention_sink: graph.static_offset_suffix("attn.attn_sink")?,
            wo_a_weight: graph.static_offset_suffix("attn.wo_a.weight")?,
            wo_a_scale: graph.static_offset_suffix("attn.wo_a.scale")?,
            wo_b_weight: graph.static_offset_suffix("attn.wo_b.weight")?,
            wo_b_scale: graph.static_offset_suffix("attn.wo_b.scale")?,
            hc_ffn_fn: graph.static_offset_suffix("hc_ffn_fn")?,
            hc_ffn_scale: graph.static_offset_suffix("hc_ffn_scale")?,
            hc_ffn_base: graph.static_offset_suffix("hc_ffn_base")?,
            ffn_norm: graph.static_offset_suffix("ffn_norm.weight")?,
            router_weight: graph.static_offset_suffix("ffn.gate.weight")?,
        })
    }
}

#[derive(Clone)]
pub struct S14CausalBlockHcQkvLayerResources {
    pub layer: u8,
    static_arena: Arc<S14Position0PagedWeightArena>,
    static_binding: S14CausalBlockHcQkvPagedStaticLayerBinding,
    pub weights: S14CausalBlockHcQkvWeightOffsets,
    pub route_mode: S14RoutePostprocessGpuMode,
    pub committed_window_kv_bf16: S14CausalBlockOwnedBufferSlice,
    pub rotated_current_block_kv_bf16: S14CausalBlockOwnedBufferSlice,
    pub rope_f32: S14CausalBlockOwnedBufferSlice,
    pub route_aux: S14CausalBlockOwnedBufferSlice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14CausalBlockHcQkvPagedStaticLayerLocation {
    Resident,
    Streamed { bank: usize },
}

/// 可跨过 provider 返回边界的 paged static 层 identity。权重 offset 均相对
/// `buffer_offset`；当前 paged arena 每层/每 stream bank 都是独立 buffer，因此 offset 为0。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockHcQkvPagedStaticLayerBinding {
    pub layer: u8,
    pub location: S14CausalBlockHcQkvPagedStaticLayerLocation,
    pub buffer_offset: u64,
    pub logical_bytes: u64,
}

#[derive(Clone, Copy)]
struct S14CausalBlockHcQkvResolvedStaticLayer<'a> {
    buffer: &'a GpuBuffer,
    buffer_offset: u64,
    logical_bytes: u64,
}

impl S14CausalBlockHcQkvLayerResources {
    #[allow(clippy::too_many_arguments)]
    pub fn from_ready_paged_static_layer(
        layer: u8,
        static_arena: Arc<S14Position0PagedWeightArena>,
        weights: S14CausalBlockHcQkvWeightOffsets,
        route_mode: S14RoutePostprocessGpuMode,
        committed_window_kv_bf16: S14CausalBlockOwnedBufferSlice,
        rotated_current_block_kv_bf16: S14CausalBlockOwnedBufferSlice,
        rope_f32: S14CausalBlockOwnedBufferSlice,
        route_aux: S14CausalBlockOwnedBufferSlice,
    ) -> Result<Self> {
        let static_binding = ready_paged_static_binding(&static_arena, layer)?;
        Ok(Self {
            layer,
            static_arena,
            static_binding,
            weights,
            route_mode,
            committed_window_kv_bf16,
            rotated_current_block_kv_bf16,
            rope_f32,
            route_aux,
        })
    }

    pub fn paged_weight_arena(&self) -> &Arc<S14Position0PagedWeightArena> {
        &self.static_arena
    }

    pub fn static_binding(&self) -> S14CausalBlockHcQkvPagedStaticLayerBinding {
        self.static_binding
    }

    /// 再次穿过 arena readiness 门。stream bank 若已被另一层覆写，这里会在 descriptor
    /// 创建与 command recording 前 fail-closed，旧 resource clone 不能继续使用旧页。
    fn resolve_ready_static_layer(&self) -> Result<S14CausalBlockHcQkvResolvedStaticLayer<'_>> {
        let ready = self.static_arena.ready_static_layer(self.layer)?;
        let (buffer, location, logical_bytes) = match ready {
            S14Position0StaticLayerBinding::Resident { buffer, layout } => (
                buffer,
                S14CausalBlockHcQkvPagedStaticLayerLocation::Resident,
                layout.requested_bytes,
            ),
            S14Position0StaticLayerBinding::Streamed {
                bank,
                buffer,
                layout,
            } => (
                buffer,
                S14CausalBlockHcQkvPagedStaticLayerLocation::Streamed { bank },
                layout.requested_bytes,
            ),
        };
        let observed = S14CausalBlockHcQkvPagedStaticLayerBinding {
            layer: self.layer,
            location,
            buffer_offset: 0,
            logical_bytes,
        };
        validate_paged_static_binding(self.static_binding, observed)?;
        Ok(S14CausalBlockHcQkvResolvedStaticLayer {
            buffer,
            buffer_offset: observed.buffer_offset,
            logical_bytes: observed.logical_bytes,
        })
    }
}

impl fmt::Debug for S14CausalBlockHcQkvLayerResources {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockHcQkvLayerResources")
            .field("layer", &self.layer)
            .field("static_arena", &Arc::as_ptr(&self.static_arena))
            .field("static_binding", &self.static_binding)
            .field("route_mode", &self.route_mode)
            .finish_non_exhaustive()
    }
}

/// 实现者负责把经过 proof/SHA 的当前层 static Range 与 committed KV/route aux 准备好；
/// 返回后这些资源必须保持有效，直到本层唯一 submit 已完成。
pub trait S14CausalBlockHcQkvResourceProvider: fmt::Debug {
    fn prepare_layer(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> std::result::Result<S14CausalBlockHcQkvLayerResources, String>;
}

#[derive(Clone)]
pub struct S14CausalBlockHiddenBank {
    pub buffer: Arc<GpuBuffer>,
    pub offset: u64,
    pub capacity_bytes: u64,
}

impl fmt::Debug for S14CausalBlockHiddenBank {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockHiddenBank")
            .field("buffer", &self.buffer.handle())
            .field("offset", &self.offset)
            .field("capacity_bytes", &self.capacity_bytes)
            .finish()
    }
}

impl S14CausalBlockHiddenBank {
    pub fn binding(
        &self,
        block_size: usize,
        generation: u64,
    ) -> Result<S14CausalBlockHiddenBinding> {
        let bytes = hidden_bytes(block_size)?;
        let end = self
            .offset
            .checked_add(bytes)
            .context("hidden bank range overflow")?;
        if self.offset % 4 != 0 || bytes > self.capacity_bytes || end > self.buffer.size() {
            bail!("causal-block HC/QKV hidden bank capacity/alignment 非法");
        }
        Ok(S14CausalBlockHiddenBinding {
            buffer: self.buffer.handle(),
            offset: self.offset,
            bytes,
            block_size,
            generation,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Region {
    offset: u64,
    bytes: u64,
}

#[derive(Clone, Debug)]
struct WorkspaceLayout {
    bytes: u64,
    hc_expanded: Region,
    hc_inverse: Region,
    hc_mixes: Region,
    hc_branch_bf16: Region,
    hc_branch_f32: Region,
    hc_aux: Region,
    q_low_f32: Region,
    q_low_bf16: Region,
    q_low_norm_bf16: Region,
    q_f32: Region,
    q_temp_bf16: Region,
    q_final_bf16: Region,
    q_inverse: Region,
    kv_f32: Region,
    kv_temp_bf16: Region,
    kv_norm_bf16: Region,
    kv_inverse: Region,
    kv_scales: Region,
    kv_raw_bf16: Region,
    attention_bf16: Region,
    attention_f32: Region,
    wo_a_f32: Region,
    wo_a_bf16: Region,
    wo_a_scales: Region,
    attention_branch_f32: Region,
    attention_branch_bf16: Region,
    router_logits: Region,
}

impl WorkspaceLayout {
    fn build() -> Result<Self> {
        let mut cursor = 0u64;
        let mut take = |bytes: u64| -> Result<Region> {
            cursor = align_up(cursor, ALIGN)?;
            let region = Region {
                offset: cursor,
                bytes,
            };
            cursor = cursor.checked_add(bytes).context("workspace overflow")?;
            Ok(region)
        };
        let batch = MAX_BATCH as u64;
        let layout = Self {
            hc_expanded: take(batch * HC_FLAT as u64 * F32_BYTES)?,
            hc_inverse: take(batch * F32_BYTES)?,
            hc_mixes: take(batch * HC_MIXES as u64 * F32_BYTES)?,
            hc_branch_bf16: take(batch * HIDDEN as u64 * BF16_BYTES)?,
            hc_branch_f32: take(batch * HIDDEN as u64 * F32_BYTES)?,
            hc_aux: take(batch * HC_AUX as u64 * F32_BYTES)?,
            q_low_f32: take(batch * Q_LOW as u64 * F32_BYTES)?,
            q_low_bf16: take(batch * Q_LOW as u64 * BF16_BYTES)?,
            q_low_norm_bf16: take(batch * Q_LOW as u64 * BF16_BYTES)?,
            q_f32: take(batch * QUERY as u64 * F32_BYTES)?,
            q_temp_bf16: take(batch * QUERY as u64 * BF16_BYTES)?,
            q_final_bf16: take(batch * QUERY as u64 * BF16_BYTES)?,
            q_inverse: take(batch * S14_CAUSAL_BLOCK_ATTENTION_HEADS as u64 * F32_BYTES)?,
            kv_f32: take(batch * KV as u64 * F32_BYTES)?,
            kv_temp_bf16: take(batch * KV as u64 * BF16_BYTES)?,
            kv_norm_bf16: take(batch * KV as u64 * BF16_BYTES)?,
            kv_inverse: take(batch * F32_BYTES)?,
            kv_scales: take(batch * 7 * F32_BYTES)?,
            kv_raw_bf16: take(batch * KV as u64 * BF16_BYTES)?,
            attention_bf16: take(batch * QUERY as u64 * BF16_BYTES)?,
            attention_f32: take(batch * QUERY as u64 * F32_BYTES)?,
            wo_a_f32: take(batch * WO_A as u64 * F32_BYTES)?,
            wo_a_bf16: take(batch * WO_A as u64 * BF16_BYTES)?,
            wo_a_scales: take(batch * (WO_A / 128) as u64 * F32_BYTES)?,
            attention_branch_f32: take(batch * HIDDEN as u64 * F32_BYTES)?,
            attention_branch_bf16: take(batch * HIDDEN as u64 * BF16_BYTES)?,
            router_logits: take(batch * S14_CAUSAL_BLOCK_ROUTER_EXPERTS as u64 * F32_BYTES)?,
            bytes: 0,
        };
        Ok(Self {
            bytes: align_up(cursor, ALIGN)?,
            ..layout
        })
    }

    fn slice<'a>(&self, buffer: &'a GpuBuffer, region: Region) -> StorageBufferSlice<'a> {
        StorageBufferSlice {
            buffer,
            offset: region.offset,
        }
    }
}

struct Pipelines {
    fp8: ComputePipeline,
    fp8_exact: ComputePipeline,
    grouped_wo_a: ComputePipeline,
    hc_normalize: ComputePipeline,
    hc_split: ComputePipeline,
    hc_post: ComputePipeline,
    q_head: ComputePipeline,
    kv_finalize: ComputePipeline,
    numeric: S14NumericPipelines,
    rmsnorm: S14Bf16RmsNormPipeline,
    f32_to_bf16: S14F32ToBf16Pipeline,
    bf16_to_f32: S14Bf16ToF32Pipeline,
    qdq: S14E4m3QdqPipeline,
}

impl Pipelines {
    fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            fp8: ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_FP8_MATVEC_SPV, 4, 12)?,
            fp8_exact: ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_FP8_MATVEC_EXACT_SPV, 4, 12)?,
            grouped_wo_a: ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_GROUPED_WO_A_SPV, 4, 16)?,
            hc_normalize: ComputePipeline::new(
                ctx,
                S14_CAUSAL_BLOCK_HC_NORMALIZE_INPUT_SPV,
                3,
                12,
            )?,
            hc_split: ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_HC_SPLIT_REDUCE_NORM_SPV, 9, 12)?,
            hc_post: ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_HC_POST_SPV, 5, 8)?,
            q_head: ComputePipeline::new(ctx, S14_Q_HEAD_NORMALIZE_SPV, 4, 16)?,
            kv_finalize: ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_KV_FINALIZE_SPV, 6, 8)?,
            numeric: S14NumericPipelines::new(ctx)?,
            rmsnorm: S14Bf16RmsNormPipeline::new(ctx)?,
            f32_to_bf16: S14F32ToBf16Pipeline::new(ctx)?,
            bf16_to_f32: S14Bf16ToF32Pipeline::new(ctx)?,
            qdq: S14E4m3QdqPipeline::new(ctx)?,
        })
    }

    fn destroy(self, ctx: &VulkanContext) {
        self.qdq.destroy(ctx);
        self.bf16_to_f32.destroy(ctx);
        self.f32_to_bf16.destroy(ctx);
        self.rmsnorm.destroy(ctx);
        self.numeric.destroy(ctx);
        self.kv_finalize.destroy(ctx);
        self.q_head.destroy(ctx);
        self.hc_post.destroy(ctx);
        self.hc_split.destroy(ctx);
        self.hc_normalize.destroy(ctx);
        self.grouped_wo_a.destroy(ctx);
        self.fp8_exact.destroy(ctx);
        self.fp8.destroy(ctx);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecorderPhase {
    Idle,
    Active {
        base_position: u32,
        block_size: usize,
    },
    LayersSealed,
    Destroyed,
}

pub struct S14CausalBlockProductionHcQkvLayerRecorder<P: S14CausalBlockHcQkvResourceProvider> {
    ctx: Arc<VulkanContext>,
    static_arena: Arc<S14Position0PagedWeightArena>,
    provider: P,
    hidden_banks: [S14CausalBlockHiddenBank; 2],
    layout: WorkspaceLayout,
    workspace: Option<GpuBuffer>,
    status: Option<GpuBuffer>,
    expert_ids: Option<GpuBuffer>,
    route_weights: Option<GpuBuffer>,
    pipelines: Option<Pipelines>,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    pending_binders: RefCell<Vec<DescriptorBinder>>,
    pending_attention: RefCell<Option<S14CausalBlockAttentionRouterRecorder>>,
    in_flight: bool,
    phase: RecorderPhase,
}

impl<P: S14CausalBlockHcQkvResourceProvider> fmt::Debug
    for S14CausalBlockProductionHcQkvLayerRecorder<P>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockProductionHcQkvLayerRecorder")
            .field("provider", &self.provider)
            .field("hidden_banks", &self.hidden_banks)
            .field("phase", &self.phase)
            .field("in_flight", &self.in_flight)
            .finish_non_exhaustive()
    }
}

impl<P: S14CausalBlockHcQkvResourceProvider> S14CausalBlockProductionHcQkvLayerRecorder<P> {
    pub fn new(
        ctx: Arc<VulkanContext>,
        static_arena: Arc<S14Position0PagedWeightArena>,
        provider: P,
        hidden_banks: [S14CausalBlockHiddenBank; 2],
    ) -> Result<Self> {
        let layout = WorkspaceLayout::build()?;
        let max_hidden = hidden_bytes(MAX_BATCH as usize)?;
        for bank in &hidden_banks {
            if bank.capacity_bytes < max_hidden
                || bank
                    .offset
                    .checked_add(max_hidden)
                    .is_none_or(|end| end > bank.buffer.size())
            {
                bail!("causal-block HC/QKV hidden bank 小于 K=8 production capacity");
            }
        }
        if hidden_banks[0].buffer.handle() == hidden_banks[1].buffer.handle()
            && ranges_overlap(
                hidden_banks[0].offset,
                max_hidden,
                hidden_banks[1].offset,
                max_hidden,
            )?
        {
            bail!("causal-block HC/QKV A/B hidden banks 重叠");
        }
        let workspace = device_buffer(&ctx, layout.bytes)?;
        let status = host_buffer(&ctx, 4)?;
        let route_bytes = MAX_BATCH as u64 * S14_CAUSAL_BLOCK_ROUTER_TOP_K as u64 * 4;
        let expert_ids = host_buffer(&ctx, route_bytes)?;
        let route_weights = host_buffer(&ctx, route_bytes)?;
        let pipelines = Pipelines::new(&ctx)?;
        let command_pool = unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let command = unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };
        let fence = unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        Ok(Self {
            ctx,
            static_arena,
            provider,
            hidden_banks,
            layout,
            workspace: Some(workspace),
            status: Some(status),
            expert_ids: Some(expert_ids),
            route_weights: Some(route_weights),
            pipelines: Some(pipelines),
            command_pool,
            command,
            fence,
            pending_binders: RefCell::new(Vec::new()),
            pending_attention: RefCell::new(None),
            in_flight: false,
            phase: RecorderPhase::Idle,
        })
    }

    pub fn hidden_binding(
        &self,
        bank: usize,
        block_size: usize,
        generation: u64,
    ) -> Result<S14CausalBlockHiddenBinding> {
        self.hidden_banks
            .get(bank)
            .context("causal-block HC/QKV hidden bank index 非法")?
            .binding(block_size, generation)
    }

    fn record_layer(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockHcQkvRecordedLayer> {
        let (base_position, block_size) = match self.phase {
            RecorderPhase::Active {
                base_position,
                block_size,
            } => (base_position, block_size),
            _ => bail!("causal-block HC/QKV recorder 当前没有 active block"),
        };
        if input.base_position != base_position || input.input_token_ids.len() != block_size {
            bail!("causal-block HC/QKV recorder input 与 active block 漂移");
        }
        let batch = u32::try_from(block_size).context("block size conversion")?;
        if !matches!(batch, 4 | 8) {
            bail!("causal-block HC/QKV recorder K 只允许4或8");
        }
        let (input_bank, output_bank) = self.resolve_hidden_banks(input.input_hidden)?;
        let output_hidden = self.hidden_banks[output_bank].binding(
            block_size,
            input
                .input_hidden
                .generation
                .checked_add(1)
                .context("generation overflow")?,
        )?;
        let resources = self
            .provider
            .prepare_layer(input)
            .map_err(anyhow::Error::msg)?;
        if !Arc::ptr_eq(&self.static_arena, resources.paged_weight_arena()) {
            bail!("causal-block HC/QKV provider 返回了非 production paged arena 的静态层");
        }
        self.validate_resources(&resources, input)?;
        self.reset_recording_owner()?;
        unsafe { self.status()?.write_at(0, &0u32.to_le_bytes()) };

        let record_result =
            self.record_command_graph(input, input_bank, output_bank, &resources, batch);
        if let Err(error) = record_result {
            self.destroy_pending();
            return Err(error);
        }
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
                .wait_for_fences(&[self.fence], true, u64::MAX)?
        };
        self.in_flight = false;
        let status = unsafe { *(self.status()?.mapped() as *const u32) };
        if status != 0 {
            self.destroy_pending();
            bail!("causal-block HC/QKV K-row graph sticky status=0x{status:08x}");
        }
        let routes = self.decode_routes(input.layer, block_size)?;
        self.destroy_pending();
        let output = S14CausalBlockAttentionRouterOutput {
            post_attention_hidden: output_hidden,
            routes,
            forward_calls: 1,
        };
        Ok(S14CausalBlockHcQkvRecordedLayer {
            receipt: S14CausalBlockHcQkvLayerRecordingReceipt {
                base_position,
                layer: input.layer,
                block_size,
                input_hidden: input.input_hidden,
                post_attention_hidden: output_hidden,
                layer_record_calls: 1,
                command_graph_submit_calls: 1,
                hc_qkv_projection_record_calls: 1,
                attention_recording_calls: 1,
                attention_output_post_record_calls: 1,
                ffn_hc_router_input_record_calls: 1,
                router_recording_calls: 1,
                serial_token_forward_calls: 0,
                hc_hidden_integration_complete: true,
            },
            output,
        })
    }

    fn record_command_graph(
        &self,
        input: &S14CausalBlockLayerInput<'_>,
        input_bank: usize,
        output_bank: usize,
        resources: &S14CausalBlockHcQkvLayerResources,
        batch: u32,
    ) -> Result<()> {
        unsafe {
            self.ctx.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let workspace = self.workspace()?;
        let status = self.status()?;
        let pipelines = self.pipelines()?;
        let input_hidden = StorageBufferSlice {
            buffer: &self.hidden_banks[input_bank].buffer,
            offset: self.hidden_banks[input_bank].offset,
        };
        let output_hidden = StorageBufferSlice {
            buffer: &self.hidden_banks[output_bank].buffer,
            offset: self.hidden_banks[output_bank].offset,
        };
        let static_layer = resources.resolve_ready_static_layer()?;
        let static_arena = static_layer.buffer;
        let static_base = static_layer.buffer_offset;
        let weights = resources.weights;

        self.record_hc_pre(
            batch,
            input_hidden,
            weights.hc_attn_fn,
            weights.hc_attn_scale,
            weights.hc_attn_base,
            weights.attn_norm,
            static_layer,
        )?;
        self.record_fp8(
            &pipelines.fp8,
            batch,
            Q_LOW,
            HIDDEN,
            static_arena,
            static_base + weights.wq_a_weight,
            static_base + weights.wq_a_scale,
            self.layout.slice(workspace, self.layout.hc_branch_f32),
            self.layout.slice(workspace, self.layout.q_low_f32),
        )?;
        self.record_f32_to_bf16(batch * Q_LOW, self.layout.q_low_f32, self.layout.q_low_bf16)?;
        self.record_rmsnorm(
            batch,
            Q_LOW,
            static_base + weights.q_norm,
            self.layout.q_low_bf16,
            self.layout.q_inverse,
            self.layout.q_low_norm_bf16,
            static_layer,
        )?;
        self.record_qdq(
            batch,
            Q_LOW,
            128,
            self.layout.q_low_norm_bf16,
            self.layout.q_low_f32,
        )?;
        self.record_fp8(
            &pipelines.fp8_exact,
            batch,
            QUERY,
            Q_LOW,
            static_arena,
            static_base + weights.wq_b_weight,
            static_base + weights.wq_b_scale,
            self.layout.slice(workspace, self.layout.q_low_f32),
            self.layout.slice(workspace, self.layout.q_f32),
        )?;
        self.record_f32_to_bf16(batch * QUERY, self.layout.q_f32, self.layout.q_temp_bf16)?;
        self.record_q_head(batch)?;

        self.record_fp8(
            &pipelines.fp8,
            batch,
            KV,
            HIDDEN,
            static_arena,
            static_base + weights.wkv_weight,
            static_base + weights.wkv_scale,
            self.layout.slice(workspace, self.layout.hc_branch_f32),
            self.layout.slice(workspace, self.layout.kv_f32),
        )?;
        self.record_f32_to_bf16(batch * KV, self.layout.kv_f32, self.layout.kv_temp_bf16)?;
        self.record_rmsnorm(
            batch,
            KV,
            static_base + weights.kv_norm,
            self.layout.kv_temp_bf16,
            self.layout.kv_inverse,
            self.layout.kv_norm_bf16,
            static_layer,
        )?;
        self.record_kv_finalize(batch, resources)?;

        let attention_shape = S14CausalBlockAttentionRouterShape::new(
            batch,
            input.base_position,
            input.base_position,
        )?;
        let attention = S14CausalBlockAttentionRouterRecorder::bind(
            &self.ctx,
            attention_shape,
            resources.route_mode,
            S14CausalBlockAttentionRouterBindings {
                query_bf16: self.layout.slice(workspace, self.layout.q_final_bf16),
                committed_window_kv_bf16: resources.committed_window_kv_bf16.storage(),
                current_block_kv_bf16: self.layout.slice(workspace, self.layout.kv_raw_bf16),
                sink_f32: StorageBufferSlice {
                    buffer: static_arena,
                    offset: static_base + weights.attention_sink,
                },
                rope_f32: resources.rope_f32.storage(),
                attention_output_bf16: self.layout.slice(workspace, self.layout.attention_bf16),
                router_weight_bf16: StorageBufferSlice {
                    buffer: static_arena,
                    offset: static_base + weights.router_weight,
                },
                router_input_f32: self.layout.slice(workspace, self.layout.hc_branch_f32),
                router_logits_f32: self.layout.slice(workspace, self.layout.router_logits),
                route_aux: resources.route_aux.storage(),
                expert_ids_u32: StorageBufferSlice::whole(self.expert_ids()?),
                route_weights_f32: StorageBufferSlice::whole(self.route_weights()?),
                sticky_status_u32: StorageBufferSlice::whole(status),
            },
        )?;
        unsafe { attention.record_attention(&self.ctx, self.command)? };
        *self.pending_attention.borrow_mut() = Some(attention);

        self.record_bf16_to_f32(
            batch * QUERY,
            self.layout.attention_bf16,
            self.layout.attention_f32,
        )?;
        self.record_grouped_wo_a(batch, static_layer, resources.weights)?;
        self.record_f32_to_bf16(batch * WO_A, self.layout.wo_a_f32, self.layout.wo_a_bf16)?;
        self.record_qdq(
            batch,
            WO_A,
            128,
            self.layout.wo_a_bf16,
            self.layout.wo_a_f32,
        )?;
        self.record_fp8(
            &pipelines.fp8_exact,
            batch,
            HIDDEN,
            WO_A,
            static_arena,
            static_base + weights.wo_b_weight,
            static_base + weights.wo_b_scale,
            self.layout.slice(workspace, self.layout.wo_a_f32),
            self.layout
                .slice(workspace, self.layout.attention_branch_f32),
        )?;
        self.record_f32_to_bf16(
            batch * HIDDEN,
            self.layout.attention_branch_f32,
            self.layout.attention_branch_bf16,
        )?;
        self.record_hc_post(batch, input_hidden, output_hidden)?;
        self.record_hc_pre(
            batch,
            output_hidden,
            weights.hc_ffn_fn,
            weights.hc_ffn_scale,
            weights.hc_ffn_base,
            weights.ffn_norm,
            static_layer,
        )?;
        let pending_attention = self.pending_attention.borrow();
        unsafe {
            pending_attention
                .as_ref()
                .context("attention recorder owner missing")?
                .record_router(&self.ctx, self.command)?;
        }
        Ok(())
    }

    fn record_hc_pre(
        &self,
        batch: u32,
        hidden: StorageBufferSlice<'_>,
        fn_offset: u64,
        scale_offset: u64,
        base_offset: u64,
        norm_offset: u64,
        static_layer: S14CausalBlockHcQkvResolvedStaticLayer<'_>,
    ) -> Result<()> {
        let workspace = self.workspace()?;
        let pipelines = self.pipelines()?;
        let hidden_bytes = batch as u64 * HC_FLAT as u64 * BF16_BYTES;
        let expanded_bytes = batch as u64 * HC_FLAT as u64 * F32_BYTES;
        let inverse_bytes = batch as u64 * F32_BYTES;
        let binder = DescriptorBinder::new_with_offsets(
            &self.ctx,
            &pipelines.hc_normalize,
            &[
                (hidden.buffer, hidden.offset, hidden_bytes),
                (workspace, self.layout.hc_expanded.offset, expanded_bytes),
                (workspace, self.layout.hc_inverse.offset, inverse_bytes),
            ],
        )?;
        unsafe {
            record_pipeline(&self.ctx, self.command, &pipelines.hc_normalize, binder.set);
            push_hc(&self.ctx, self.command, &pipelines.hc_normalize, batch);
            self.ctx.device.cmd_dispatch(self.command, 1, batch, 1);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(binder);
        let dispatch = pipelines.numeric.bind_f32_matvec_arenas(
            &self.ctx,
            S14F32MatvecShape::new(HC_MIXES, HC_FLAT, batch)?,
            static_layer.buffer,
            static_layer
                .buffer_offset
                .checked_add(static_layer.logical_bytes)
                .context("paged static binding end overflow")?,
            static_layer
                .buffer_offset
                .checked_add(fn_offset)
                .context("HC fn offset overflow")?,
            workspace,
            workspace.size(),
            self.layout.hc_expanded.offset,
            workspace,
            workspace.size(),
            self.layout.hc_mixes.offset,
        )?;
        unsafe {
            pipelines
                .numeric
                .cmd_f32_matvec(&self.ctx, self.command, &dispatch);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(dispatch.binder);
        let binder = DescriptorBinder::new_with_offsets(
            &self.ctx,
            &pipelines.hc_split,
            &[
                (hidden.buffer, hidden.offset, hidden_bytes),
                (
                    workspace,
                    self.layout.hc_mixes.offset,
                    batch as u64 * 24 * 4,
                ),
                (
                    static_layer.buffer,
                    static_layer
                        .buffer_offset
                        .checked_add(scale_offset)
                        .context("HC scale offset overflow")?,
                    3 * 4,
                ),
                (
                    static_layer.buffer,
                    static_layer
                        .buffer_offset
                        .checked_add(base_offset)
                        .context("HC base offset overflow")?,
                    24 * 4,
                ),
                (
                    static_layer.buffer,
                    static_layer
                        .buffer_offset
                        .checked_add(norm_offset)
                        .context("HC norm offset overflow")?,
                    HIDDEN as u64 * 2,
                ),
                (
                    workspace,
                    self.layout.hc_branch_bf16.offset,
                    batch as u64 * HIDDEN as u64 * 2,
                ),
                (
                    workspace,
                    self.layout.hc_branch_f32.offset,
                    batch as u64 * HIDDEN as u64 * 4,
                ),
                (workspace, self.layout.hc_aux.offset, batch as u64 * 20 * 4),
                (workspace, self.layout.hc_inverse.offset, inverse_bytes),
            ],
        )?;
        unsafe {
            record_pipeline(&self.ctx, self.command, &pipelines.hc_split, binder.set);
            push_hc(&self.ctx, self.command, &pipelines.hc_split, batch);
            self.ctx.device.cmd_dispatch(self.command, 1, batch, 1);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(binder);
        Ok(())
    }

    fn record_fp8(
        &self,
        pipeline: &ComputePipeline,
        batch: u32,
        n: u32,
        k: u32,
        weights: &GpuBuffer,
        weight_offset: u64,
        scale_offset: u64,
        input: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
    ) -> Result<()> {
        let weight_bytes = n as u64 * k as u64;
        let scale_bytes = n.div_ceil(128) as u64 * k.div_ceil(128) as u64;
        let binder = DescriptorBinder::new_with_offsets(
            &self.ctx,
            pipeline,
            &[
                (input.buffer, input.offset, batch as u64 * k as u64 * 4),
                (weights, weight_offset, weight_bytes),
                (weights, scale_offset, scale_bytes),
                (output.buffer, output.offset, batch as u64 * n as u64 * 4),
            ],
        )?;
        unsafe {
            record_pipeline(&self.ctx, self.command, pipeline, binder.set);
            let mut push = [0u8; 12];
            push[..4].copy_from_slice(&n.to_le_bytes());
            push[4..8].copy_from_slice(&k.to_le_bytes());
            push[8..].copy_from_slice(&batch.to_le_bytes());
            push_constants(&self.ctx, self.command, pipeline, &push);
            self.ctx.device.cmd_dispatch(self.command, n, batch, 1);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(binder);
        Ok(())
    }

    fn record_f32_to_bf16(&self, scalars: u32, input: Region, output: Region) -> Result<()> {
        let workspace = self.workspace()?;
        let pipelines = self.pipelines()?;
        let dispatch = pipelines.f32_to_bf16.bind_slices(
            &self.ctx,
            S14F32ToBf16Shape::new(scalars)?,
            self.layout.slice(workspace, input),
            self.layout.slice(workspace, output),
            StorageBufferSlice::whole(self.status()?),
        )?;
        unsafe {
            pipelines
                .f32_to_bf16
                .cmd(&self.ctx, self.command, &dispatch);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(dispatch.binder);
        Ok(())
    }

    fn record_bf16_to_f32(&self, scalars: u32, input: Region, output: Region) -> Result<()> {
        let workspace = self.workspace()?;
        let pipelines = self.pipelines()?;
        let dispatch = pipelines.bf16_to_f32.bind_slices(
            &self.ctx,
            S14Bf16ToF32Shape::new(scalars)?,
            self.layout.slice(workspace, input),
            self.layout.slice(workspace, output),
            StorageBufferSlice::whole(self.status()?),
        )?;
        unsafe {
            pipelines
                .bf16_to_f32
                .cmd(&self.ctx, self.command, &dispatch);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(dispatch.binder);
        Ok(())
    }

    fn record_rmsnorm(
        &self,
        rows: u32,
        hidden: u32,
        weight_offset: u64,
        input: Region,
        inverse: Region,
        output: Region,
        static_layer: S14CausalBlockHcQkvResolvedStaticLayer<'_>,
    ) -> Result<()> {
        let workspace = self.workspace()?;
        let pipelines = self.pipelines()?;
        let dispatch = pipelines.rmsnorm.bind_slices(
            &self.ctx,
            S14Bf16RmsNormShape::new(rows, hidden)?,
            NORM_EPS,
            self.layout.slice(workspace, input),
            StorageBufferSlice {
                buffer: static_layer.buffer,
                offset: weight_offset,
            },
            self.layout.slice(workspace, inverse),
            self.layout.slice(workspace, output),
            StorageBufferSlice::whole(self.status()?),
        )?;
        unsafe {
            pipelines.rmsnorm.cmd(&self.ctx, self.command, &dispatch);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(dispatch.binder);
        Ok(())
    }

    fn record_qdq(
        &self,
        rows: u32,
        hidden: u32,
        group: u32,
        input: Region,
        output: Region,
    ) -> Result<()> {
        let workspace = self.workspace()?;
        let pipelines = self.pipelines()?;
        let scales = if hidden == Q_LOW {
            self.layout.q_inverse
        } else {
            self.layout.wo_a_scales
        };
        let dispatch = pipelines.qdq.bind_slices(
            &self.ctx,
            S14E4m3QdqShape::new(rows, hidden, group)?,
            self.layout.slice(workspace, input),
            self.layout.slice(workspace, scales),
            self.layout.slice(workspace, output),
            StorageBufferSlice::whole(self.status()?),
        )?;
        unsafe {
            pipelines.qdq.cmd(&self.ctx, self.command, &dispatch);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(dispatch.binder);
        Ok(())
    }

    fn record_q_head(&self, batch: u32) -> Result<()> {
        let workspace = self.workspace()?;
        let status = self.status()?;
        let pipeline = &self.pipelines()?.q_head;
        let rows = batch * S14_CAUSAL_BLOCK_ATTENTION_HEADS;
        let bytes = batch as u64 * QUERY as u64 * 2;
        let binder = DescriptorBinder::new_with_offsets(
            &self.ctx,
            pipeline,
            &[
                (workspace, self.layout.q_temp_bf16.offset, bytes),
                (workspace, self.layout.q_inverse.offset, rows as u64 * 4),
                (workspace, self.layout.q_final_bf16.offset, bytes),
                (status, 0, 4),
            ],
        )?;
        unsafe {
            record_pipeline(&self.ctx, self.command, pipeline, binder.set);
            let mut push = [0u8; 16];
            push[..4].copy_from_slice(&rows.to_le_bytes());
            push[4..8].copy_from_slice(&S14_CAUSAL_BLOCK_ATTENTION_HEAD_DIM.to_le_bytes());
            push[8..12].copy_from_slice(&NORM_EPS.to_le_bytes());
            push[12..].copy_from_slice(&0u32.to_le_bytes());
            push_constants(&self.ctx, self.command, pipeline, &push);
            self.ctx.device.cmd_dispatch(self.command, rows, 1, 1);
            compute_barrier(&self.ctx, self.command);
            push[12..].copy_from_slice(&1u32.to_le_bytes());
            push_constants(&self.ctx, self.command, pipeline, &push);
            self.ctx
                .device
                .cmd_dispatch(self.command, (batch * QUERY / 2).div_ceil(256), 1, 1);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(binder);
        Ok(())
    }

    fn record_kv_finalize(
        &self,
        batch: u32,
        resources: &S14CausalBlockHcQkvLayerResources,
    ) -> Result<()> {
        let workspace = self.workspace()?;
        let status = self.status()?;
        let pipeline = &self.pipelines()?.kv_finalize;
        let bytes = batch as u64 * KV as u64 * 2;
        let binder = DescriptorBinder::new_with_offsets(
            &self.ctx,
            pipeline,
            &[
                (workspace, self.layout.kv_norm_bf16.offset, bytes),
                (
                    workspace,
                    self.layout.kv_scales.offset,
                    batch as u64 * 7 * 4,
                ),
                (
                    &resources.rope_f32.buffer,
                    resources.rope_f32.offset,
                    batch as u64 * 64 * 4,
                ),
                (workspace, self.layout.kv_raw_bf16.offset, bytes),
                (
                    &resources.rotated_current_block_kv_bf16.buffer,
                    resources.rotated_current_block_kv_bf16.offset,
                    bytes,
                ),
                (status, 0, 4),
            ],
        )?;
        unsafe {
            record_pipeline(&self.ctx, self.command, pipeline, binder.set);
            let mut push = [0u8; 8];
            push[..4].copy_from_slice(&batch.to_le_bytes());
            push[4..].copy_from_slice(&0u32.to_le_bytes());
            push_constants(&self.ctx, self.command, pipeline, &push);
            self.ctx.device.cmd_dispatch(self.command, batch * 7, 1, 1);
            compute_barrier(&self.ctx, self.command);
            push[4..].copy_from_slice(&1u32.to_le_bytes());
            push_constants(&self.ctx, self.command, pipeline, &push);
            self.ctx
                .device
                .cmd_dispatch(self.command, (batch * 256).div_ceil(128), 1, 1);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(binder);
        Ok(())
    }

    fn record_grouped_wo_a(
        &self,
        batch: u32,
        static_layer: S14CausalBlockHcQkvResolvedStaticLayer<'_>,
        weights: S14CausalBlockHcQkvWeightOffsets,
    ) -> Result<()> {
        let workspace = self.workspace()?;
        let pipeline = &self.pipelines()?.grouped_wo_a;
        let binder = DescriptorBinder::new_with_offsets(
            &self.ctx,
            pipeline,
            &[
                (
                    workspace,
                    self.layout.attention_f32.offset,
                    batch as u64 * QUERY as u64 * 4,
                ),
                (
                    static_layer.buffer,
                    static_layer
                        .buffer_offset
                        .checked_add(weights.wo_a_weight)
                        .context("wo_a weight offset overflow")?,
                    WO_A as u64 * HIDDEN as u64,
                ),
                (
                    static_layer.buffer,
                    static_layer
                        .buffer_offset
                        .checked_add(weights.wo_a_scale)
                        .context("wo_a scale offset overflow")?,
                    (WO_A / 128) as u64 * (HIDDEN / 128) as u64,
                ),
                (
                    workspace,
                    self.layout.wo_a_f32.offset,
                    batch as u64 * WO_A as u64 * 4,
                ),
            ],
        )?;
        unsafe {
            record_pipeline(&self.ctx, self.command, pipeline, binder.set);
            let mut push = [0u8; 16];
            push[..4].copy_from_slice(&WO_GROUPS.to_le_bytes());
            push[4..8].copy_from_slice(&WO_GROUP_OUT.to_le_bytes());
            push[8..12].copy_from_slice(&HIDDEN.to_le_bytes());
            push[12..].copy_from_slice(&batch.to_le_bytes());
            push_constants(&self.ctx, self.command, pipeline, &push);
            self.ctx.device.cmd_dispatch(self.command, WO_A, batch, 1);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(binder);
        Ok(())
    }

    fn record_hc_post(
        &self,
        batch: u32,
        residual: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
    ) -> Result<()> {
        let workspace = self.workspace()?;
        let status = self.status()?;
        let pipeline = &self.pipelines()?.hc_post;
        let binder = DescriptorBinder::new_with_offsets(
            &self.ctx,
            pipeline,
            &[
                (
                    workspace,
                    self.layout.attention_branch_bf16.offset,
                    batch as u64 * HIDDEN as u64 * 2,
                ),
                (
                    residual.buffer,
                    residual.offset,
                    batch as u64 * HC_FLAT as u64 * 2,
                ),
                (
                    workspace,
                    self.layout.hc_aux.offset,
                    batch as u64 * HC_AUX as u64 * 4,
                ),
                (
                    output.buffer,
                    output.offset,
                    batch as u64 * HC_FLAT as u64 * 2,
                ),
                (status, 0, 4),
            ],
        )?;
        unsafe {
            record_pipeline(&self.ctx, self.command, pipeline, binder.set);
            let mut push = [0u8; 8];
            push[..4].copy_from_slice(&HIDDEN.to_le_bytes());
            push[4..].copy_from_slice(&batch.to_le_bytes());
            push_constants(&self.ctx, self.command, pipeline, &push);
            self.ctx.device.cmd_dispatch(self.command, 1, batch, 1);
            compute_barrier(&self.ctx, self.command);
        }
        self.pending_binders.borrow_mut().push(binder);
        Ok(())
    }

    fn validate_resources(
        &self,
        resources: &S14CausalBlockHcQkvLayerResources,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<()> {
        let batch = input.input_token_ids.len() as u64;
        let static_layer = resources.resolve_ready_static_layer()?;
        let static_end = static_layer
            .buffer_offset
            .checked_add(static_layer.logical_bytes)
            .context("paged static layer range overflow")?;
        if resources.layer != input.layer
            || resources.static_binding.layer != input.layer
            || static_end > static_layer.buffer.size()
            || static_layer.logical_bytes == 0
        {
            bail!("causal-block HC/QKV layer/static arena identity 漂移");
        }
        resources
            .committed_window_kv_bf16
            .validate_exact(input.base_position as u64 * KV as u64 * 2, "committed KV")?;
        resources
            .rotated_current_block_kv_bf16
            .validate_exact(batch * KV as u64 * 2, "rotated current KV")?;
        resources.rope_f32.validate_exact(batch * 64 * 4, "RoPE")?;
        let aux_bytes = match resources.route_mode {
            S14RoutePostprocessGpuMode::BiasTop6 => S14_CAUSAL_BLOCK_ROUTER_EXPERTS as u64 * 4,
            S14RoutePostprocessGpuMode::PhysicalIds => {
                batch * S14_CAUSAL_BLOCK_ROUTER_TOP_K as u64 * 4
            }
        };
        resources.route_aux.validate_exact(aux_bytes, "route aux")?;
        let w = resources.weights;
        for (offset, bytes, label) in [
            (
                w.hc_attn_fn,
                HC_MIXES as u64 * HC_FLAT as u64 * 4,
                "hc_attn_fn",
            ),
            (
                w.hc_ffn_fn,
                HC_MIXES as u64 * HC_FLAT as u64 * 4,
                "hc_ffn_fn",
            ),
            (w.wq_a_weight, Q_LOW as u64 * HIDDEN as u64, "wq_a"),
            (w.wq_b_weight, QUERY as u64 * Q_LOW as u64, "wq_b"),
            (w.wkv_weight, KV as u64 * HIDDEN as u64, "wkv"),
            (w.wo_a_weight, WO_A as u64 * HIDDEN as u64, "wo_a"),
            (w.wo_b_weight, HIDDEN as u64 * WO_A as u64, "wo_b"),
            (
                w.router_weight,
                S14_CAUSAL_BLOCK_ROUTER_EXPERTS as u64 * HIDDEN as u64 * 2,
                "router",
            ),
        ] {
            let end = offset.checked_add(bytes).context("weight range overflow")?;
            if offset % 4 != 0 || end > static_layer.logical_bytes {
                bail!("causal-block HC/QKV {label} weight range 越界");
            }
        }
        Ok(())
    }

    fn resolve_hidden_banks(&self, input: S14CausalBlockHiddenBinding) -> Result<(usize, usize)> {
        let mut found = None;
        for (index, bank) in self.hidden_banks.iter().enumerate() {
            if input.buffer == bank.buffer.handle()
                && input.offset == bank.offset
                && input.bytes == hidden_bytes(input.block_size)?
            {
                found = Some(index);
            }
        }
        let input_bank =
            found.context("causal-block HC/QKV input hidden 不属于已注册 A/B banks")?;
        Ok((input_bank, 1 - input_bank))
    }

    fn decode_routes(&self, layer: u8, block_size: usize) -> Result<Vec<RouteDecision>> {
        let count = block_size * S14_CAUSAL_BLOCK_ROUTER_TOP_K as usize;
        let ids =
            unsafe { std::slice::from_raw_parts(self.expert_ids()?.mapped() as *const u32, count) };
        let weights = unsafe {
            std::slice::from_raw_parts(self.route_weights()?.mapped() as *const f32, count)
        };
        let kind = router_kind_for_layer(layer).context("router kind missing")?;
        (0..block_size)
            .map(|row| {
                let start = row * S14_CAUSAL_BLOCK_ROUTER_TOP_K as usize;
                let expert_ids = ids[start..start + S14_CAUSAL_BLOCK_ROUTER_TOP_K as usize]
                    .iter()
                    .map(|&id| u16::try_from(id).context("route physical ID overflow"))
                    .collect::<Result<Vec<_>>>()?;
                let route_weights =
                    weights[start..start + S14_CAUSAL_BLOCK_ROUTER_TOP_K as usize].to_vec();
                if route_weights.iter().any(|value| !value.is_finite()) {
                    bail!("causal-block HC/QKV route weight non-finite");
                }
                Ok(RouteDecision {
                    layer,
                    kind,
                    expert_ids,
                    weights: route_weights,
                })
            })
            .collect()
    }

    fn reset_recording_owner(&mut self) -> Result<()> {
        if self.in_flight
            || !self.pending_binders.borrow().is_empty()
            || self.pending_attention.borrow().is_some()
        {
            bail!("causal-block HC/QKV command owner 尚未 drain");
        }
        unsafe {
            self.ctx.device.reset_fences(&[self.fence])?;
            self.ctx
                .device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
        }
        Ok(())
    }

    fn destroy_pending(&mut self) {
        if let Some(attention) = self.pending_attention.get_mut().take() {
            attention.destroy(&self.ctx);
        }
        for binder in self.pending_binders.get_mut().drain(..).rev() {
            binder.destroy(&self.ctx);
        }
    }

    fn drain_pending(&mut self) -> Result<()> {
        if self.in_flight {
            unsafe {
                self.ctx
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?
            };
            self.in_flight = false;
        }
        self.destroy_pending();
        Ok(())
    }

    fn workspace(&self) -> Result<&GpuBuffer> {
        self.workspace.as_ref().context("workspace destroyed")
    }
    fn status(&self) -> Result<&GpuBuffer> {
        self.status.as_ref().context("status destroyed")
    }
    fn expert_ids(&self) -> Result<&GpuBuffer> {
        self.expert_ids.as_ref().context("route IDs destroyed")
    }
    fn route_weights(&self) -> Result<&GpuBuffer> {
        self.route_weights
            .as_ref()
            .context("route weights destroyed")
    }
    fn pipelines(&self) -> Result<&Pipelines> {
        self.pipelines.as_ref().context("pipelines destroyed")
    }
}

impl<P: S14CausalBlockHcQkvResourceProvider> S14CausalBlockHcQkvLayerRecorder
    for S14CausalBlockProductionHcQkvLayerRecorder<P>
{
    fn begin_block(
        &mut self,
        base_position: u32,
        block_size: usize,
    ) -> std::result::Result<(), String> {
        if self.phase != RecorderPhase::Idle || !matches!(block_size, 4 | 8) {
            return Err("causal-block HC/QKV concrete recorder begin phase/K 非法".into());
        }
        if base_position == 0
            || base_position
                .checked_add(block_size as u32)
                .is_none_or(|end| end > 127)
        {
            return Err("causal-block HC/QKV concrete recorder position 越界".into());
        }
        self.phase = RecorderPhase::Active {
            base_position,
            block_size,
        };
        Ok(())
    }

    fn record_k_lane_hc_qkv_attention_router(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> std::result::Result<S14CausalBlockHcQkvRecordedLayer, String> {
        self.record_layer(input)
            .map_err(|error| format!("{error:#}"))
    }

    fn seal_and_drain(&mut self, completed_layers: usize) -> std::result::Result<(), String> {
        if !matches!(self.phase, RecorderPhase::Active { .. })
            || completed_layers != FULL_DEPTH_LAYERS.len()
        {
            return Err("causal-block HC/QKV concrete recorder seal phase/layers 非法".into());
        }
        self.drain_pending().map_err(|error| format!("{error:#}"))?;
        self.phase = RecorderPhase::LayersSealed;
        Ok(())
    }

    fn drain_and_abort(&mut self, _completed_layers: usize) -> std::result::Result<(), String> {
        if self.phase == RecorderPhase::Destroyed {
            return Err("causal-block HC/QKV concrete recorder 已销毁".into());
        }
        self.drain_pending().map_err(|error| format!("{error:#}"))?;
        self.phase = RecorderPhase::Idle;
        Ok(())
    }

    fn finish_validated_block(&mut self) -> std::result::Result<(), String> {
        if self.phase != RecorderPhase::LayersSealed {
            return Err("causal-block HC/QKV concrete recorder 没有 sealed block".into());
        }
        self.phase = RecorderPhase::Idle;
        Ok(())
    }

    fn destroy(&mut self) -> std::result::Result<(), String> {
        if self.phase == RecorderPhase::Destroyed {
            return Ok(());
        }
        self.drain_pending().map_err(|error| format!("{error:#}"))?;
        unsafe {
            self.ctx.device.destroy_fence(self.fence, None);
            self.ctx
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        if let Some(pipelines) = self.pipelines.take() {
            pipelines.destroy(&self.ctx);
        }
        if let Some(buffer) = self.route_weights.take() {
            buffer.destroy(&self.ctx);
        }
        if let Some(buffer) = self.expert_ids.take() {
            buffer.destroy(&self.ctx);
        }
        if let Some(buffer) = self.status.take() {
            buffer.destroy(&self.ctx);
        }
        if let Some(buffer) = self.workspace.take() {
            buffer.destroy(&self.ctx);
        }
        self.phase = RecorderPhase::Destroyed;
        Ok(())
    }
}

unsafe fn record_pipeline(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
) {
    ctx.device
        .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
    ctx.device.cmd_bind_descriptor_sets(
        command,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.layout,
        0,
        &[set],
        &[],
    );
}

unsafe fn push_constants(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    bytes: &[u8],
) {
    ctx.device.cmd_push_constants(
        command,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
    );
}

unsafe fn push_hc(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    batch: u32,
) {
    let mut push = [0u8; 12];
    push[..4].copy_from_slice(&HIDDEN.to_le_bytes());
    push[4..8].copy_from_slice(&batch.to_le_bytes());
    push[8..].copy_from_slice(&NORM_EPS.to_le_bytes());
    push_constants(ctx, command, pipeline, &push);
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

fn ready_paged_static_binding(
    arena: &S14Position0PagedWeightArena,
    layer: u8,
) -> Result<S14CausalBlockHcQkvPagedStaticLayerBinding> {
    let (location, logical_bytes) = match arena.ready_static_layer(layer)? {
        S14Position0StaticLayerBinding::Resident { layout, .. } => (
            S14CausalBlockHcQkvPagedStaticLayerLocation::Resident,
            layout.requested_bytes,
        ),
        S14Position0StaticLayerBinding::Streamed { bank, layout, .. } => (
            S14CausalBlockHcQkvPagedStaticLayerLocation::Streamed { bank },
            layout.requested_bytes,
        ),
    };
    if logical_bytes == 0 {
        bail!("causal-block HC/QKV paged static L{layer} logical bytes 不能为0");
    }
    Ok(S14CausalBlockHcQkvPagedStaticLayerBinding {
        layer,
        location,
        buffer_offset: 0,
        logical_bytes,
    })
}

fn validate_paged_static_binding(
    expected: S14CausalBlockHcQkvPagedStaticLayerBinding,
    observed: S14CausalBlockHcQkvPagedStaticLayerBinding,
) -> Result<()> {
    if expected != observed {
        bail!(
            "causal-block HC/QKV paged static binding 漂移: expected={expected:?} observed={observed:?}"
        );
    }
    Ok(())
}

fn hidden_bytes(block_size: usize) -> Result<u64> {
    u64::try_from(
        block_size
            .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE)
            .context("hidden elements overflow")?,
    )
    .context("hidden bytes conversion")?
    .checked_mul(2)
    .context("hidden bytes overflow")
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .context("alignment overflow")
        .map(|v| v / alignment * alignment)
}

fn ranges_overlap(left: u64, left_bytes: u64, right: u64, right_bytes: u64) -> Result<bool> {
    let left_end = left
        .checked_add(left_bytes)
        .context("left range overflow")?;
    let right_end = right
        .checked_add(right_bytes)
        .context("right range overflow")?;
    Ok(left < right_end && right < left_end)
}

fn device_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::MemoryPropertyFlags::empty(),
        false,
    )
}

fn host_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_static_binding_rejects_stale_stream_layer_bank_or_offset() {
        let expected = S14CausalBlockHcQkvPagedStaticLayerBinding {
            layer: 7,
            location: S14CausalBlockHcQkvPagedStaticLayerLocation::Streamed { bank: 1 },
            buffer_offset: 0,
            logical_bytes: 4096,
        };
        validate_paged_static_binding(expected, expected).unwrap();

        for observed in [
            S14CausalBlockHcQkvPagedStaticLayerBinding {
                layer: 8,
                ..expected
            },
            S14CausalBlockHcQkvPagedStaticLayerBinding {
                location: S14CausalBlockHcQkvPagedStaticLayerLocation::Streamed { bank: 0 },
                ..expected
            },
            S14CausalBlockHcQkvPagedStaticLayerBinding {
                buffer_offset: 256,
                ..expected
            },
        ] {
            assert!(validate_paged_static_binding(expected, observed).is_err());
        }
    }
}
