//! S14 causal-block grouped MoE 的 concrete Vulkan recorder。
//!
//! 每层把 K 个 FFN HC-pre、K×top-6 routed expert、K 个 shared expert、
//! exact `slot 0→5→shared` reduce、BF16-RNE 与 HC-post 全部录入同一个 command
//! buffer。union bank 由 runtime 保活；本 owner 持有 pipeline、workspace、control、
//! output ping-pong 与所有尚未 drain 的 descriptor pool。

use crate::{
    compute::{DescriptorBinder, StorageBufferSlice},
    s14_causal_block_grouped_graph::{
        S14CausalBlockGroupedMoeRecorder, S14CausalBlockRecordedGroupedMoe,
    },
    s14_causal_block_hc_qkv_recorder::S14CausalBlockHiddenBank,
    s14_causal_block_layer::{
        S14CausalBlockHiddenBinding, S14CausalBlockLayerRangePlan, S14CausalBlockUnionBankBinding,
    },
    s14_causal_block_moe_adapter::S14CausalBlockProductionMoeAdapter,
    s14_dynamic_page_cache_readiness::DynamicPageFetchMode,
    s14_dynamic_routed_page_plan::{FullDepthExpertCatalog, RoutedProjection, RoutedRangePart},
    s14_e4m3_qdq::{S14E4m3QdqPipeline, S14E4m3QdqShape},
    s14_f32_to_bf16::{S14F32ToBf16Pipeline, S14F32ToBf16Shape},
    s14_hc_post::{S14HcPostPipeline, S14HcPostShape},
    s14_position0_hybrid_weight_arena::{
        S14Position0HybridWeightArena, S14Position0StaticLayerLayout,
    },
    s14_position0_paged_weight_arena::{
        S14Position0PagedWeightArena, S14Position0StaticLayerBinding,
    },
    s14_vulkan::{
        S14F32MatvecShape, S14HcPreShape, S14MatvecShape, S14NumericPipelines,
        S14RaggedBranchOffsets, S14RaggedMatvecShape, S14RaggedProjection,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    GraphProfile, LayerCausalBatchPlan, RouteDecision, EXPERTS_PER_TOKEN, EXPERT_PAGE_BYTES,
    FULL_DEPTH_LAYERS,
};
use std::{collections::BTreeMap, fmt, path::Path, sync::Arc};

const HIDDEN: u32 = 4096;
const HC_STREAMS: u64 = 4;
const HC_FLAT: u32 = HIDDEN * HC_STREAMS as u32;
const INTERMEDIATE: u32 = 2048;
const TOP6: u32 = EXPERTS_PER_TOKEN as u32;
const MAX_K: usize = 8;
const MAX_BRANCHES: usize = MAX_K * EXPERTS_PER_TOKEN;
const LAYERS: usize = 43;
const NORM_EPS: f32 = 1.0e-6;
const CONTROL_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug)]
struct StridedRegion {
    offset: u64,
    stride: u64,
}

impl StridedRegion {
    fn lane(self, lane: usize) -> Result<u64> {
        self.offset
            .checked_add(
                self.stride
                    .checked_mul(lane as u64)
                    .context("lane stride overflow")?,
            )
            .context("lane offset overflow")
    }
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceLayout {
    residual: StridedRegion,
    hc_norm: StridedRegion,
    hc_mixes: StridedRegion,
    hc_branch_bf16: StridedRegion,
    hc_branch_f32: StridedRegion,
    hc_aux: StridedRegion,
    hc_inverse: StridedRegion,
    qdq_scales: StridedRegion,
    route_weights: u64,
    shared_weights: u64,
    routed_gate: u64,
    routed_up: u64,
    routed_hidden: u64,
    routed_down: u64,
    shared_gate: u64,
    shared_up: u64,
    shared_hidden: u64,
    shared_down: u64,
    moe_f32: u64,
    moe_bf16: u64,
    bytes: u64,
}

impl WorkspaceLayout {
    fn build(alignment: u64) -> Result<Self> {
        let mut cursor = 0u64;
        let residual = take_strided(&mut cursor, 4 * HIDDEN as u64 * 2, MAX_K, alignment)?;
        let hc_norm = take_strided(&mut cursor, HC_FLAT as u64 * 4, MAX_K, alignment)?;
        let hc_mixes = take_strided(&mut cursor, 24 * 4, MAX_K, alignment)?;
        let hc_branch_bf16 = take_strided(&mut cursor, HIDDEN as u64 * 2, MAX_K, alignment)?;
        let hc_branch_f32 = take_strided(&mut cursor, HIDDEN as u64 * 4, MAX_K, alignment)?;
        let hc_aux = take_strided(&mut cursor, 20 * 4, MAX_K, alignment)?;
        let hc_inverse = take_strided(&mut cursor, 4, MAX_K, alignment)?;
        let qdq_scales = take_strided(&mut cursor, (HIDDEN / 128) as u64 * 4, MAX_K, alignment)?;
        let route_weights = take(&mut cursor, MAX_BRANCHES as u64 * 4, alignment)?;
        let shared_weights = take(&mut cursor, MAX_K as u64 * 4, alignment)?;
        let routed_gate = take(
            &mut cursor,
            MAX_BRANCHES as u64 * INTERMEDIATE as u64 * 4,
            alignment,
        )?;
        let routed_up = take(
            &mut cursor,
            MAX_BRANCHES as u64 * INTERMEDIATE as u64 * 4,
            alignment,
        )?;
        let routed_hidden = take(
            &mut cursor,
            MAX_BRANCHES as u64 * INTERMEDIATE as u64 * 4,
            alignment,
        )?;
        let routed_down = take(
            &mut cursor,
            MAX_BRANCHES as u64 * HIDDEN as u64 * 4,
            alignment,
        )?;
        let shared_gate = take(
            &mut cursor,
            MAX_K as u64 * INTERMEDIATE as u64 * 4,
            alignment,
        )?;
        let shared_up = take(
            &mut cursor,
            MAX_K as u64 * INTERMEDIATE as u64 * 4,
            alignment,
        )?;
        let shared_hidden = take(
            &mut cursor,
            MAX_K as u64 * INTERMEDIATE as u64 * 4,
            alignment,
        )?;
        let shared_down = take(&mut cursor, MAX_K as u64 * HIDDEN as u64 * 4, alignment)?;
        let moe_f32 = take(&mut cursor, MAX_K as u64 * HIDDEN as u64 * 4, alignment)?;
        let moe_bf16 = take(&mut cursor, MAX_K as u64 * HIDDEN as u64 * 2, alignment)?;
        Ok(Self {
            residual,
            hc_norm,
            hc_mixes,
            hc_branch_bf16,
            hc_branch_f32,
            hc_aux,
            hc_inverse,
            qdq_scales,
            route_weights,
            shared_weights,
            routed_gate,
            routed_up,
            routed_hidden,
            routed_down,
            shared_gate,
            shared_up,
            shared_hidden,
            shared_down,
            moe_f32,
            moe_bf16,
            bytes: align_up(cursor, alignment)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ControlLayout {
    routed_metadata: u64,
    shared_metadata: u64,
    route_weights: u64,
    shared_weights: u64,
    layer_stride: u64,
    status: u64,
    bytes: u64,
}

impl ControlLayout {
    fn build(alignment: u64) -> Result<Self> {
        let mut cursor = 0u64;
        let routed_metadata = take(&mut cursor, MAX_BRANCHES as u64 * 24, alignment)?;
        let shared_metadata = take(&mut cursor, MAX_K as u64 * 24, alignment)?;
        let route_weights = take(&mut cursor, MAX_BRANCHES as u64 * 4, alignment)?;
        let shared_weights = take(&mut cursor, MAX_K as u64 * 4, alignment)?;
        let layer_stride = align_up(cursor, alignment)?;
        let status = layer_stride
            .checked_mul(LAYERS as u64)
            .context("control status overflow")?;
        let bytes = align_up(
            status.checked_add(4).context("control bytes overflow")?,
            alignment,
        )?;
        Ok(Self {
            routed_metadata,
            shared_metadata,
            route_weights,
            shared_weights,
            layer_stride,
            status,
            bytes,
        })
    }

    fn layer_base(self, index: usize) -> Result<u64> {
        self.layer_stride
            .checked_mul(index as u64)
            .context("control layer offset overflow")
    }
}

#[derive(Debug)]
struct ActiveBlock {
    block_size: usize,
    next_layer: usize,
}

#[derive(Debug)]
struct LayerRecordPlan {
    routed_metadata: Vec<S14RaggedBranchOffsets>,
    shared_metadata: Vec<S14RaggedBranchOffsets>,
    route_weight_bits: Vec<u32>,
}

struct StaticLayerView<'a> {
    buffer: &'a GpuBuffer,
    logical_bytes: u64,
    hc_fn: u64,
    hc_scale: u64,
    hc_base: u64,
    ffn_norm: u64,
    shared: S14RaggedBranchOffsets,
}

#[derive(Clone)]
pub struct S14CausalBlockGroupedMoeStaticLayerResources {
    pub layer: u8,
    pub buffer: Arc<GpuBuffer>,
    pub logical_bytes: u64,
    pub hc_fn: u64,
    pub hc_scale: u64,
    pub hc_base: u64,
    pub ffn_norm: u64,
    pub shared: S14RaggedBranchOffsets,
}

impl fmt::Debug for S14CausalBlockGroupedMoeStaticLayerResources {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockGroupedMoeStaticLayerResources")
            .field("layer", &self.layer)
            .field("buffer", &self.buffer.handle())
            .field("logical_bytes", &self.logical_bytes)
            .finish_non_exhaustive()
    }
}

enum StaticLayerSource {
    Hybrid(Arc<S14Position0HybridWeightArena>),
    /// RX 5700 XT production owner。resident 与 streamed static 层都由同一个 paged arena
    /// 保活；上游 provider 必须在本层 recorder 被调用前把 streamed bank 填成当前层。
    Paged(Arc<S14Position0PagedWeightArena>),
    Direct(BTreeMap<u8, S14CausalBlockGroupedMoeStaticLayerResources>),
}

pub struct S14CausalBlockGroupedMoeVulkanRecorder {
    ctx: Arc<VulkanContext>,
    static_layers: StaticLayerSource,
    numeric: Option<S14NumericPipelines>,
    numeric_exact: Option<S14NumericPipelines>,
    qdq: Option<S14E4m3QdqPipeline>,
    f32_to_bf16: Option<S14F32ToBf16Pipeline>,
    hc_post: Option<S14HcPostPipeline>,
    workspace: Option<GpuBuffer>,
    control: Option<GpuBuffer>,
    outputs: [Option<GpuBuffer>; 2],
    /// production whole-layer A/B owner。存在时 grouped MoE 必须把 HC-post
    /// 写回与 post-attention bank 相反的注册 bank，供下一层直接继续。
    external_outputs: Option<[S14CausalBlockHiddenBank; 2]>,
    workspace_layout: WorkspaceLayout,
    control_layout: ControlLayout,
    binders: Vec<DescriptorBinder>,
    active: Option<ActiveBlock>,
    destroyed: bool,
}

/// Resident-loader 的最短 production 注入类型；构造成功后可直接传给
/// `S14CausalBlockVulkanBackend::with_moe_adapter`。
pub type S14CausalBlockConcreteMoeAdapter =
    S14CausalBlockProductionMoeAdapter<S14CausalBlockGroupedMoeVulkanRecorder>;

pub fn build_s14_causal_block_concrete_moe_adapter(
    ctx: Arc<VulkanContext>,
    catalog: FullDepthExpertCatalog,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
    static_arena: Arc<S14Position0HybridWeightArena>,
) -> Result<S14CausalBlockConcreteMoeAdapter> {
    let recorder = S14CausalBlockGroupedMoeVulkanRecorder::new(Arc::clone(&ctx), static_arena)?;
    S14CausalBlockProductionMoeAdapter::new(ctx, catalog, cache_root, fetch_mode, recorder)
}

pub fn build_s14_causal_block_paged_moe_adapter(
    ctx: Arc<VulkanContext>,
    catalog: FullDepthExpertCatalog,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
    static_arena: Arc<S14Position0PagedWeightArena>,
    hidden_banks: [S14CausalBlockHiddenBank; 2],
) -> Result<S14CausalBlockConcreteMoeAdapter> {
    let recorder = S14CausalBlockGroupedMoeVulkanRecorder::new_paged(
        Arc::clone(&ctx),
        static_arena,
        hidden_banks,
    )?;
    S14CausalBlockProductionMoeAdapter::new(ctx, catalog, cache_root, fetch_mode, recorder)
}

impl fmt::Debug for S14CausalBlockGroupedMoeVulkanRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S14CausalBlockGroupedMoeVulkanRecorder")
            .field("active", &self.active)
            .field("pending_binders", &self.binders.len())
            .field("destroyed", &self.destroyed)
            .finish_non_exhaustive()
    }
}

impl S14CausalBlockGroupedMoeVulkanRecorder {
    pub fn new(
        ctx: Arc<VulkanContext>,
        static_arena: Arc<S14Position0HybridWeightArena>,
    ) -> Result<Self> {
        Self::new_with_static_source(ctx, StaticLayerSource::Hybrid(static_arena))
    }

    pub fn new_paged(
        ctx: Arc<VulkanContext>,
        static_arena: Arc<S14Position0PagedWeightArena>,
        hidden_banks: [S14CausalBlockHiddenBank; 2],
    ) -> Result<Self> {
        let mut recorder =
            Self::new_with_static_source(ctx, StaticLayerSource::Paged(static_arena))?;
        let required = MAX_K as u64 * HC_STREAMS * HIDDEN as u64 * 2;
        for bank in &hidden_banks {
            if bank.offset % 4 != 0
                || bank.capacity_bytes < required
                || bank
                    .offset
                    .checked_add(required)
                    .is_none_or(|end| end > bank.buffer.size())
            {
                recorder.destroy_inner();
                bail!("grouped MoE production hidden bank capacity/alignment 非法");
            }
        }
        if hidden_banks[0].buffer.handle() == hidden_banks[1].buffer.handle()
            && ranges_overlap(
                hidden_banks[0].offset,
                required,
                hidden_banks[1].offset,
                required,
            )?
        {
            recorder.destroy_inner();
            bail!("grouped MoE production hidden A/B banks 重叠");
        }
        // Common constructor allocates private compatibility outputs. production改为
        // 复用同一A/B owner后立即释放它们，避免第三套hidden与下一层身份断裂。
        for output in recorder.outputs.iter_mut().rev() {
            if let Some(buffer) = output.take() {
                buffer.destroy(&recorder.ctx);
            }
        }
        recorder.external_outputs = Some(hidden_banks);
        Ok(recorder)
    }

    pub fn new_with_static_layer(
        ctx: Arc<VulkanContext>,
        resources: S14CausalBlockGroupedMoeStaticLayerResources,
    ) -> Result<Self> {
        validate_direct_static_layer(&resources)?;
        let mut layers = BTreeMap::new();
        layers.insert(resources.layer, resources);
        Self::new_with_static_source(ctx, StaticLayerSource::Direct(layers))
    }

    fn new_with_static_source(
        ctx: Arc<VulkanContext>,
        static_layers: StaticLayerSource,
    ) -> Result<Self> {
        let alignment = storage_alignment(&ctx);
        let workspace_layout = WorkspaceLayout::build(alignment)?;
        let control_layout = ControlLayout::build(alignment)?;
        let mut owner = Self {
            ctx,
            static_layers,
            numeric: None,
            numeric_exact: None,
            qdq: None,
            f32_to_bf16: None,
            hc_post: None,
            workspace: None,
            control: None,
            outputs: [None, None],
            external_outputs: None,
            workspace_layout,
            control_layout,
            binders: Vec::new(),
            active: None,
            destroyed: false,
        };
        owner.numeric = Some(S14NumericPipelines::new(&owner.ctx)?);
        owner.numeric_exact = Some(S14NumericPipelines::new_exact_audit(&owner.ctx)?);
        owner.qdq = Some(S14E4m3QdqPipeline::new(&owner.ctx)?);
        owner.f32_to_bf16 = Some(S14F32ToBf16Pipeline::new(&owner.ctx)?);
        owner.hc_post = Some(S14HcPostPipeline::new(&owner.ctx)?);
        owner.workspace = Some(new_device_buffer(&owner.ctx, workspace_layout.bytes)?);
        owner.control = Some(new_control_buffer(&owner.ctx, control_layout.bytes)?);
        let output_bytes = MAX_K as u64 * HC_STREAMS * HIDDEN as u64 * 2;
        owner.outputs[0] = Some(new_device_buffer(&owner.ctx, output_bytes)?);
        owner.outputs[1] = Some(new_device_buffer(&owner.ctx, output_bytes)?);
        Ok(owner)
    }

    fn record_inner(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        union_bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
        layer_index: usize,
        binders: &mut Vec<DescriptorBinder>,
    ) -> Result<S14CausalBlockRecordedGroupedMoe> {
        if !std::ptr::eq(ctx, self.ctx.as_ref()) {
            bail!("grouped MoE recorder VulkanContext owner 漂移");
        }
        let k = routes.len();
        let expected_hidden_bytes = k as u64 * HC_STREAMS * HIDDEN as u64 * 2;
        if !matches!(k, 4 | 8)
            || post_attention_hidden.buffer == vk::Buffer::null()
            || post_attention_hidden.offset % 4 != 0
            || post_attention_hidden.block_size != k
            || post_attention_hidden.bytes != expected_hidden_bytes
            || union_bank.buffer == vk::Buffer::null()
            || union_bank.allocated_bank_bytes < range_plan.union_expert_bytes
        {
            bail!("grouped MoE external hidden/union binding 非法");
        }
        let static_view = self.static_layer_view(range_plan.layer)?;
        let plan = build_layer_record_plan(routes, batch_plan, range_plan, static_view.shared)?;
        let workspace = self.workspace()?;
        let control = self.control()?;
        let (output, output_base) = self.output_bank(post_attention_hidden, layer_index)?;
        let numeric = self.numeric.as_ref().context("numeric pipeline 已销毁")?;
        let numeric_exact = self
            .numeric_exact
            .as_ref()
            .context("exact numeric pipeline 已销毁")?;
        let qdq = self.qdq.as_ref().context("QDQ pipeline 已销毁")?;
        let f32_to_bf16 = self
            .f32_to_bf16
            .as_ref()
            .context("F32->BF16 pipeline 已销毁")?;
        let hc_post = self.hc_post.as_ref().context("HC-post pipeline 已销毁")?;

        let control_base = self.control_layout.layer_base(layer_index)?;
        let routed_meta_offset = control_base + self.control_layout.routed_metadata;
        let shared_meta_offset = control_base + self.control_layout.shared_metadata;
        let route_weight_src = control_base + self.control_layout.route_weights;
        let shared_weight_src = control_base + self.control_layout.shared_weights;
        unsafe {
            control.write_at(
                routed_meta_offset as usize,
                &metadata_bytes(&plan.routed_metadata),
            );
            control.write_at(
                shared_meta_offset as usize,
                &metadata_bytes(&plan.shared_metadata),
            );
            control.write_at(
                route_weight_src as usize,
                &u32_bytes(&plan.route_weight_bits),
            );
            control.write_at(
                shared_weight_src as usize,
                &u32_bytes(&vec![1.0f32.to_bits(); k]),
            );
            ctx.device.cmd_copy_buffer(
                command,
                post_attention_hidden.buffer,
                workspace.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(post_attention_hidden.offset)
                    .dst_offset(self.workspace_layout.residual.offset)
                    .size(expected_hidden_bytes)],
            );
            ctx.device.cmd_copy_buffer(
                command,
                control.handle(),
                workspace.handle(),
                &[
                    vk::BufferCopy::default()
                        .src_offset(route_weight_src)
                        .dst_offset(self.workspace_layout.route_weights)
                        .size((k * EXPERTS_PER_TOKEN * 4) as u64),
                    vk::BufferCopy::default()
                        .src_offset(shared_weight_src)
                        .dst_offset(self.workspace_layout.shared_weights)
                        .size((k * 4) as u64),
                ],
            );
            transfer_to_compute_barrier(ctx, command);
        }

        let hc_shape = S14HcPreShape::new(HIDDEN)?;
        for lane in 0..k {
            let residual = self.workspace_layout.residual.lane(lane)?;
            let hc_norm = self.workspace_layout.hc_norm.lane(lane)?;
            let mixes = self.workspace_layout.hc_mixes.lane(lane)?;
            let branch_bf16 = self.workspace_layout.hc_branch_bf16.lane(lane)?;
            let branch_f32 = self.workspace_layout.hc_branch_f32.lane(lane)?;
            let aux = self.workspace_layout.hc_aux.lane(lane)?;
            let inverse = self.workspace_layout.hc_inverse.lane(lane)?;
            let dispatch = numeric.bind_hc_normalize_input_arena(
                ctx,
                hc_shape,
                NORM_EPS,
                workspace,
                self.workspace_layout.bytes,
                residual,
                hc_norm,
                inverse,
            )?;
            unsafe {
                numeric.cmd_hc_normalize_input(ctx, command, &dispatch);
                compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric.bind_f32_matvec_arenas(
                ctx,
                S14F32MatvecShape::new(24, HC_FLAT, 1)?,
                static_view.buffer,
                static_view.logical_bytes,
                static_view.hc_fn,
                workspace,
                self.workspace_layout.bytes,
                hc_norm,
                workspace,
                self.workspace_layout.bytes,
                mixes,
            )?;
            unsafe {
                numeric.cmd_f32_matvec(ctx, command, &dispatch);
                compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric.bind_hc_split_reduce_norm_arenas(
                ctx,
                hc_shape,
                NORM_EPS,
                static_view.buffer,
                static_view.logical_bytes,
                static_view.hc_scale,
                static_view.hc_base,
                static_view.ffn_norm,
                workspace,
                self.workspace_layout.bytes,
                residual,
                mixes,
                branch_bf16,
                branch_f32,
                aux,
                inverse,
            )?;
            unsafe {
                numeric.cmd_hc_split_reduce_norm(ctx, command, &dispatch);
                compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
            let dispatch = qdq.bind_slices(
                ctx,
                S14E4m3QdqShape::new(1, HIDDEN, 128)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: branch_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.workspace_layout.qdq_scales.lane(lane)?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: branch_f32,
                },
                StorageBufferSlice {
                    buffer: control,
                    offset: self.control_layout.status,
                },
            )?;
            unsafe {
                qdq.cmd(ctx, command, &dispatch);
                compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
        }

        let branches = (k * EXPERTS_PER_TOKEN) as u32;
        for (projection, output_offset) in [
            (S14RaggedProjection::W1, self.workspace_layout.routed_gate),
            (S14RaggedProjection::W3, self.workspace_layout.routed_up),
        ] {
            let dispatch = numeric_exact.bind_ragged_mxfp4_external_weight_arena(
                ctx,
                S14RaggedMatvecShape::new(branches, TOP6, INTERMEDIATE, HIDDEN, projection)?,
                union_bank.buffer,
                union_bank.allocated_bank_bytes,
                range_plan.union_expert_bytes,
                control,
                self.control_layout.bytes,
                routed_meta_offset,
                &plan.routed_metadata,
                workspace,
                self.workspace_layout.bytes,
                self.workspace_layout.hc_branch_f32.offset,
                output_offset,
            )?;
            unsafe {
                numeric_exact.cmd_ragged_mxfp4_matvec(ctx, command, &dispatch);
                compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
        }
        let dispatch = numeric.bind_batched_official_expert_prepare_arena(
            ctx,
            branches,
            INTERMEDIATE,
            workspace,
            self.workspace_layout.bytes,
            self.workspace_layout.routed_gate,
            self.workspace_layout.routed_up,
            self.workspace_layout.route_weights,
            self.workspace_layout.routed_hidden,
        )?;
        unsafe {
            numeric.cmd_batched_official_expert_prepare(ctx, command, &dispatch);
            compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        let dispatch = numeric_exact.bind_ragged_mxfp4_external_weight_arena(
            ctx,
            S14RaggedMatvecShape::new(branches, 1, HIDDEN, INTERMEDIATE, S14RaggedProjection::W2)?,
            union_bank.buffer,
            union_bank.allocated_bank_bytes,
            range_plan.union_expert_bytes,
            control,
            self.control_layout.bytes,
            routed_meta_offset,
            &plan.routed_metadata,
            workspace,
            self.workspace_layout.bytes,
            self.workspace_layout.routed_hidden,
            self.workspace_layout.routed_down,
        )?;
        unsafe {
            numeric_exact.cmd_ragged_mxfp4_matvec(ctx, command, &dispatch);
            compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);

        for (projection, output_offset) in [
            (S14RaggedProjection::W1, self.workspace_layout.shared_gate),
            (S14RaggedProjection::W3, self.workspace_layout.shared_up),
        ] {
            let dispatch = numeric_exact.bind_ragged_fp8_arenas(
                ctx,
                S14RaggedMatvecShape::new(k as u32, 1, INTERMEDIATE, HIDDEN, projection)?,
                static_view.buffer,
                static_view.logical_bytes,
                control,
                self.control_layout.bytes,
                shared_meta_offset,
                &plan.shared_metadata,
                workspace,
                self.workspace_layout.bytes,
                self.workspace_layout.hc_branch_f32.offset,
                output_offset,
            )?;
            unsafe {
                numeric_exact.cmd_ragged_fp8_matvec(ctx, command, &dispatch);
                compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
        }
        let dispatch = numeric.bind_batched_official_expert_prepare_arena(
            ctx,
            k as u32,
            INTERMEDIATE,
            workspace,
            self.workspace_layout.bytes,
            self.workspace_layout.shared_gate,
            self.workspace_layout.shared_up,
            self.workspace_layout.shared_weights,
            self.workspace_layout.shared_hidden,
        )?;
        unsafe {
            numeric.cmd_batched_official_expert_prepare(ctx, command, &dispatch);
            compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        let dispatch = numeric_exact.bind_ragged_fp8_arenas(
            ctx,
            S14RaggedMatvecShape::new(k as u32, 1, HIDDEN, INTERMEDIATE, S14RaggedProjection::W2)?,
            static_view.buffer,
            static_view.logical_bytes,
            control,
            self.control_layout.bytes,
            shared_meta_offset,
            &plan.shared_metadata,
            workspace,
            self.workspace_layout.bytes,
            self.workspace_layout.shared_hidden,
            self.workspace_layout.shared_down,
        )?;
        unsafe {
            numeric_exact.cmd_ragged_fp8_matvec(ctx, command, &dispatch);
            compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);

        let dispatch = numeric.bind_exact_order_block_reduce_arena(
            ctx,
            k as u32,
            workspace,
            self.workspace_layout.bytes,
            self.workspace_layout.routed_down,
            self.workspace_layout.shared_down,
            self.workspace_layout.moe_f32,
        )?;
        unsafe {
            numeric.cmd_exact_order_block_reduce(ctx, command, &dispatch);
            compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        let dispatch = f32_to_bf16.bind_slices(
            ctx,
            S14F32ToBf16Shape::new(k as u32 * HIDDEN)?,
            StorageBufferSlice {
                buffer: workspace,
                offset: self.workspace_layout.moe_f32,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: self.workspace_layout.moe_bf16,
            },
            StorageBufferSlice {
                buffer: control,
                offset: self.control_layout.status,
            },
        )?;
        unsafe {
            f32_to_bf16.cmd(ctx, command, &dispatch);
            compute_barrier(ctx, command);
        }
        binders.push(dispatch.binder);
        for lane in 0..k {
            let dispatch = hc_post.bind_slices(
                ctx,
                S14HcPostShape::new(HIDDEN)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.workspace_layout.moe_bf16 + lane as u64 * HIDDEN as u64 * 2,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.workspace_layout.residual.lane(lane)?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.workspace_layout.hc_aux.lane(lane)?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.workspace_layout.hc_aux.lane(lane)? + 16,
                },
                StorageBufferSlice {
                    buffer: output,
                    offset: output_base + lane as u64 * HC_STREAMS * HIDDEN as u64 * 2,
                },
                StorageBufferSlice {
                    buffer: control,
                    offset: self.control_layout.status,
                },
            )?;
            unsafe {
                hc_post.cmd(ctx, command, &dispatch);
                compute_barrier(ctx, command);
            }
            binders.push(dispatch.binder);
        }
        unsafe { publish_and_host_barrier(ctx, command) };
        Ok(S14CausalBlockRecordedGroupedMoe {
            output_hidden: S14CausalBlockHiddenBinding {
                buffer: output.handle(),
                offset: output_base,
                bytes: expected_hidden_bytes,
                block_size: k,
                generation: post_attention_hidden
                    .generation
                    .checked_add(1)
                    .context("grouped MoE hidden generation overflow")?,
            },
            grouped_expert_work_items: batch_plan.unique_experts,
            lane_assignments: batch_plan.assignments,
            recorder_calls: 1,
            serial_token_forward_calls: 0,
        })
    }

    fn static_layer_view(&self, layer: u8) -> Result<StaticLayerView<'_>> {
        if let StaticLayerSource::Direct(layers) = &self.static_layers {
            let resources = layers
                .get(&layer)
                .context("direct static layer resources 缺层")?;
            validate_direct_static_layer(resources)?;
            return Ok(StaticLayerView {
                buffer: &resources.buffer,
                logical_bytes: resources.logical_bytes,
                hc_fn: resources.hc_fn,
                hc_scale: resources.hc_scale,
                hc_base: resources.hc_base,
                ffn_norm: resources.ffn_norm,
                shared: resources.shared,
            });
        }
        let index = FULL_DEPTH_LAYERS
            .iter()
            .position(|&candidate| candidate == layer)
            .context("static layer 越出 FullDepth43")?;
        let (buffer, layout, source_name) = match &self.static_layers {
            StaticLayerSource::Hybrid(static_arena) => {
                let layout = static_arena
                    .layout()
                    .static_layers
                    .get(index)
                    .context("hybrid static layout 缺层")?;
                (static_arena.static_layer(layer)?, layout, "hybrid")
            }
            StaticLayerSource::Paged(static_arena) => {
                match static_arena.ready_static_layer(layer)? {
                    S14Position0StaticLayerBinding::Resident { buffer, layout }
                    | S14Position0StaticLayerBinding::Streamed { buffer, layout, .. } => {
                        (buffer, layout, "paged")
                    }
                }
            }
            StaticLayerSource::Direct(_) => unreachable!(),
        };
        if layout.layer != layer {
            bail!("{source_name} static layer identity 漂移");
        }
        let hc_fn = static_asset(layout, layer, "hc_ffn_fn", 24 * HC_FLAT as u64 * 4)?;
        let hc_scale = static_asset(layout, layer, "hc_ffn_scale", 3 * 4)?;
        let hc_base = static_asset(layout, layer, "hc_ffn_base", 24 * 4)?;
        let ffn_norm = static_asset(layout, layer, "ffn_norm.weight", HIDDEN as u64 * 2)?;
        let w1 = S14MatvecShape::new(INTERMEDIATE, HIDDEN)?.validate_fp8()?;
        let w2 = S14MatvecShape::new(HIDDEN, INTERMEDIATE)?.validate_fp8()?;
        let offset = |suffix: &str, bytes: u64| static_asset(layout, layer, suffix, bytes);
        let shared = S14RaggedBranchOffsets {
            w1: to_u32(offset(
                "ffn.shared_experts.w1.weight",
                w1.fp8_weight_bytes()?,
            )?)?,
            s1: to_u32(offset(
                "ffn.shared_experts.w1.scale",
                w1.fp8_scale_bytes()?,
            )?)?,
            w3: to_u32(offset(
                "ffn.shared_experts.w3.weight",
                w1.fp8_weight_bytes()?,
            )?)?,
            s3: to_u32(offset(
                "ffn.shared_experts.w3.scale",
                w1.fp8_scale_bytes()?,
            )?)?,
            w2: to_u32(offset(
                "ffn.shared_experts.w2.weight",
                w2.fp8_weight_bytes()?,
            )?)?,
            s2: to_u32(offset(
                "ffn.shared_experts.w2.scale",
                w2.fp8_scale_bytes()?,
            )?)?,
        };
        Ok(StaticLayerView {
            buffer,
            logical_bytes: layout.requested_bytes,
            hc_fn,
            hc_scale,
            hc_base,
            ffn_norm,
            shared,
        })
    }

    fn workspace(&self) -> Result<&GpuBuffer> {
        self.workspace
            .as_ref()
            .context("grouped MoE workspace 已销毁")
    }

    fn control(&self) -> Result<&GpuBuffer> {
        self.control.as_ref().context("grouped MoE control 已销毁")
    }

    fn output_bank(
        &self,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        layer_index: usize,
    ) -> Result<(&GpuBuffer, u64)> {
        if let Some(banks) = &self.external_outputs {
            let mut output = None;
            let mut post_attention_is_registered = false;
            for bank in banks {
                if bank.buffer.handle() == post_attention_hidden.buffer
                    && bank.offset == post_attention_hidden.offset
                {
                    post_attention_is_registered = true;
                } else {
                    if output.is_some() {
                        bail!("grouped MoE production hidden bank identity 非唯一");
                    }
                    output = Some((bank.buffer.as_ref(), bank.offset));
                }
            }
            if !post_attention_is_registered {
                bail!("grouped MoE post-attention hidden 不属于 production A/B banks");
            }
            return output.context("grouped MoE production output bank 缺失");
        }
        let output = self.outputs[layer_index % 2]
            .as_ref()
            .context("grouped MoE output buffer 已销毁")?;
        Ok((output, 0))
    }

    fn release_binders(&mut self) {
        for binder in self.binders.drain(..).rev() {
            binder.destroy(&self.ctx);
        }
    }

    fn destroy_inner(&mut self) {
        if self.destroyed {
            return;
        }
        self.release_binders();
        for output in self.outputs.iter_mut().rev() {
            if let Some(buffer) = output.take() {
                buffer.destroy(&self.ctx);
            }
        }
        self.external_outputs.take();
        if let Some(buffer) = self.control.take() {
            buffer.destroy(&self.ctx);
        }
        if let Some(buffer) = self.workspace.take() {
            buffer.destroy(&self.ctx);
        }
        if let Some(pipeline) = self.hc_post.take() {
            pipeline.destroy(&self.ctx);
        }
        if let Some(pipeline) = self.f32_to_bf16.take() {
            pipeline.destroy(&self.ctx);
        }
        if let Some(pipeline) = self.qdq.take() {
            pipeline.destroy(&self.ctx);
        }
        if let Some(pipeline) = self.numeric_exact.take() {
            pipeline.destroy(&self.ctx);
        }
        if let Some(pipeline) = self.numeric.take() {
            pipeline.destroy(&self.ctx);
        }
        self.active = None;
        self.destroyed = true;
    }
}

impl S14CausalBlockGroupedMoeRecorder for S14CausalBlockGroupedMoeVulkanRecorder {
    fn begin_block(&mut self, _base_position: u32, block_size: usize) -> Result<()> {
        if self.destroyed
            || self.active.is_some()
            || !self.binders.is_empty()
            || !matches!(block_size, 4 | 8)
        {
            bail!("grouped MoE recorder begin 生命周期/K 非法");
        }
        let status = self.control_layout.status as usize;
        unsafe { self.control()?.write_at(status, &0u32.to_le_bytes()) };
        self.active = Some(ActiveBlock {
            block_size,
            next_layer: 0,
        });
        Ok(())
    }

    fn record_grouped_moe(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        union_bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockRecordedGroupedMoe> {
        let active = self
            .active
            .as_ref()
            .context("grouped MoE recorder 未 begin")?;
        let layer_index = active.next_layer;
        if active.block_size != routes.len()
            || FULL_DEPTH_LAYERS.get(layer_index).copied() != Some(range_plan.layer)
        {
            bail!("grouped MoE recorder block/layer 顺序漂移");
        }
        let mut layer_binders = Vec::new();
        let result = self.record_inner(
            ctx,
            command,
            union_bank,
            post_attention_hidden,
            routes,
            batch_plan,
            range_plan,
            layer_index,
            &mut layer_binders,
        );
        if let Err(error) = result {
            for binder in layer_binders.into_iter().rev() {
                binder.destroy(ctx);
            }
            return Err(error);
        }
        self.binders.extend(layer_binders);
        self.active.as_mut().unwrap().next_layer += 1;
        result
    }

    fn owns_output_hidden(&self, binding: S14CausalBlockHiddenBinding) -> bool {
        matches!(binding.block_size, 4 | 8)
            && binding.bytes == binding.block_size as u64 * HC_STREAMS * HIDDEN as u64 * 2
            && (self.external_outputs.as_ref().is_some_and(|banks| {
                banks.iter().any(|bank| {
                    bank.buffer.handle() == binding.buffer
                        && bank.offset == binding.offset
                        && binding.bytes <= bank.capacity_bytes
                })
            }) || (binding.offset == 0
                && self.outputs.iter().flatten().any(|output| {
                    output.handle() == binding.buffer && output.size() >= binding.bytes
                })))
    }

    fn finish_block_after_drain(&mut self, aborted: bool) -> Result<()> {
        let active = self
            .active
            .take()
            .context("grouped MoE recorder 没有 active block")?;
        let complete = active.next_layer == FULL_DEPTH_LAYERS.len();
        let code = unsafe {
            std::ptr::read_volatile(
                self.control()?
                    .mapped()
                    .add(self.control_layout.status as usize) as *const u32,
            )
        };
        self.release_binders();
        if !aborted && !complete {
            bail!("grouped MoE recorder 禁止发布不完整 FullDepth43 block");
        }
        if code != 0 {
            bail!("grouped MoE recorder sticky numeric status 非零: 0x{code:08x}");
        }
        Ok(())
    }

    fn destroy(&mut self) -> Result<()> {
        if self.active.is_some() || !self.binders.is_empty() {
            bail!("grouped MoE recorder 必须 timeline drain 后销毁");
        }
        self.destroy_inner();
        Ok(())
    }
}

impl Drop for S14CausalBlockGroupedMoeVulkanRecorder {
    fn drop(&mut self) {
        self.destroy_inner();
    }
}

fn build_layer_record_plan(
    routes: &[RouteDecision],
    batch_plan: &LayerCausalBatchPlan,
    range_plan: &S14CausalBlockLayerRangePlan,
    shared: S14RaggedBranchOffsets,
) -> Result<LayerRecordPlan> {
    batch_plan
        .validate_against(routes)
        .map_err(|error| anyhow!(error.to_string()))?;
    if !matches!(routes.len(), 4 | 8)
        || range_plan.block_size != routes.len()
        || range_plan.layer != batch_plan.layer
        || range_plan.unique_experts != batch_plan.unique_experts
        || range_plan.physical_ranges != range_plan.unique_experts * 6
        || range_plan.ranges.len() != range_plan.physical_ranges
    {
        bail!("grouped MoE record plan K/layer/range count 漂移");
    }
    let mut cursor = 0u64;
    let mut experts = BTreeMap::new();
    for chunk in range_plan.ranges.chunks_exact(6) {
        let expert = chunk[0].expert_id;
        let expected = [
            (RoutedProjection::W1, RoutedRangePart::Weight),
            (RoutedProjection::W1, RoutedRangePart::Scale),
            (RoutedProjection::W2, RoutedRangePart::Weight),
            (RoutedProjection::W2, RoutedRangePart::Scale),
            (RoutedProjection::W3, RoutedRangePart::Weight),
            (RoutedProjection::W3, RoutedRangePart::Scale),
        ];
        let mut offsets = [0u32; 6];
        for (index, (range, identity)) in chunk.iter().zip(expected).enumerate() {
            if range.expert_id != expert
                || range.projection != identity.0
                || range.part != identity.1
                || range.bytes != routed_range_bytes(identity.1)
            {
                bail!("grouped MoE union canonical Range identity/bytes 漂移");
            }
            offsets[index] = to_u32(cursor)?;
            cursor = cursor
                .checked_add(range.bytes)
                .context("union offset overflow")?;
        }
        let metadata = S14RaggedBranchOffsets {
            w1: offsets[0],
            s1: offsets[1],
            w3: offsets[4],
            s3: offsets[5],
            w2: offsets[2],
            s2: offsets[3],
        };
        if experts.insert(expert, metadata).is_some() {
            bail!("grouped MoE union expert 重复");
        }
    }
    if cursor != range_plan.union_expert_bytes
        || cursor != range_plan.unique_experts as u64 * EXPERT_PAGE_BYTES
        || !batch_plan
            .experts
            .iter()
            .map(|entry| entry.expert_id)
            .eq(experts.keys().copied())
    {
        bail!("grouped MoE union expert order/bytes 与 batch plan 漂移");
    }
    let mut routed_metadata = Vec::with_capacity(routes.len() * EXPERTS_PER_TOKEN);
    let mut route_weight_bits = Vec::with_capacity(routes.len() * EXPERTS_PER_TOKEN);
    for route in routes {
        route
            .validate_for(GraphProfile::FullDepth43NativeTop6)
            .map_err(|error| anyhow!(error.to_string()))?;
        if route.layer != range_plan.layer {
            bail!("grouped MoE route layer 漂移");
        }
        for (&expert, &weight) in route.expert_ids.iter().zip(&route.weights) {
            routed_metadata.push(*experts.get(&expert).context("route expert 不在 union")?);
            route_weight_bits.push(weight.to_bits());
        }
    }
    if routed_metadata.len() != routes.len() * EXPERTS_PER_TOKEN
        || route_weight_bits.len() != routed_metadata.len()
    {
        bail!("grouped MoE K×top-6 route-slot 展开不完整");
    }
    Ok(LayerRecordPlan {
        routed_metadata,
        shared_metadata: vec![shared; routes.len()],
        route_weight_bits,
    })
}

fn routed_range_bytes(part: RoutedRangePart) -> u64 {
    match part {
        RoutedRangePart::Weight => 4_194_304,
        RoutedRangePart::Scale => 262_144,
    }
}

fn static_asset(
    layout: &S14Position0StaticLayerLayout,
    layer: u8,
    suffix: &str,
    expected_bytes: u64,
) -> Result<u64> {
    let tensor = format!("layers.{layer}.{suffix}");
    let asset = layout
        .assets
        .iter()
        .find(|asset| asset.tensor == tensor)
        .with_context(|| format!("static layer 缺少 {tensor}"))?;
    if asset.bytes != expected_bytes
        || asset
            .local_offset
            .checked_add(asset.bytes)
            .unwrap_or(u64::MAX)
            > layout.requested_bytes
    {
        bail!("static asset bytes/capacity 漂移: {tensor}");
    }
    Ok(asset.local_offset)
}

fn validate_direct_static_layer(
    resources: &S14CausalBlockGroupedMoeStaticLayerResources,
) -> Result<()> {
    if !FULL_DEPTH_LAYERS.contains(&resources.layer)
        || resources.logical_bytes == 0
        || resources.logical_bytes > resources.buffer.size()
    {
        bail!("direct grouped MoE static layer identity/capacity 非法");
    }
    let w1 = S14MatvecShape::new(INTERMEDIATE, HIDDEN)?.validate_fp8()?;
    let w2 = S14MatvecShape::new(HIDDEN, INTERMEDIATE)?.validate_fp8()?;
    for (offset, bytes, label) in [
        (resources.hc_fn, 24 * HC_FLAT as u64 * 4, "hc_fn"),
        (resources.hc_scale, 3 * 4, "hc_scale"),
        (resources.hc_base, 24 * 4, "hc_base"),
        (resources.ffn_norm, HIDDEN as u64 * 2, "ffn_norm"),
        (
            resources.shared.w1 as u64,
            w1.fp8_weight_bytes()?,
            "shared.w1",
        ),
        (
            resources.shared.s1 as u64,
            w1.fp8_scale_bytes()?,
            "shared.s1",
        ),
        (
            resources.shared.w3 as u64,
            w1.fp8_weight_bytes()?,
            "shared.w3",
        ),
        (
            resources.shared.s3 as u64,
            w1.fp8_scale_bytes()?,
            "shared.s3",
        ),
        (
            resources.shared.w2 as u64,
            w2.fp8_weight_bytes()?,
            "shared.w2",
        ),
        (
            resources.shared.s2 as u64,
            w2.fp8_scale_bytes()?,
            "shared.s2",
        ),
    ] {
        let end = offset
            .checked_add(bytes)
            .with_context(|| format!("direct static {label} range overflow"))?;
        if offset % 4 != 0 || end > resources.logical_bytes {
            bail!("direct grouped MoE static {label} range 越界");
        }
    }
    Ok(())
}

fn new_device_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new_vram(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
    )
}

fn new_control_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(),
        true,
    )
}

fn storage_alignment(ctx: &VulkanContext) -> u64 {
    unsafe {
        ctx.instance
            .get_physical_device_properties(ctx.physical)
            .limits
            .min_storage_buffer_offset_alignment
    }
    .max(CONTROL_ALIGNMENT)
}

fn take(cursor: &mut u64, bytes: u64, alignment: u64) -> Result<u64> {
    *cursor = align_up(*cursor, alignment)?;
    let offset = *cursor;
    *cursor = cursor
        .checked_add(bytes)
        .context("workspace region overflow")?;
    Ok(offset)
}

fn take_strided(
    cursor: &mut u64,
    bytes: u64,
    lanes: usize,
    alignment: u64,
) -> Result<StridedRegion> {
    let stride = align_up(bytes, alignment)?;
    let offset = take(
        cursor,
        stride
            .checked_mul(lanes as u64)
            .context("strided region overflow")?,
        alignment,
    )?;
    Ok(StridedRegion { offset, stride })
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("alignment 必须为非零二次幂");
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .context("alignment overflow")
}

fn ranges_overlap(left: u64, left_bytes: u64, right: u64, right_bytes: u64) -> Result<bool> {
    let left_end = left
        .checked_add(left_bytes)
        .context("left hidden range overflow")?;
    let right_end = right
        .checked_add(right_bytes)
        .context("right hidden range overflow")?;
    Ok(left < right_end && right < left_end)
}

fn to_u32(value: u64) -> Result<u32> {
    u32::try_from(value).context("ragged metadata offset 超过 u32")
}

fn metadata_bytes(metadata: &[S14RaggedBranchOffsets]) -> Vec<u8> {
    metadata
        .iter()
        .flat_map(|entry| entry.words())
        .flat_map(u32::to_le_bytes)
        .collect()
}

fn u32_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().copied().flat_map(u32::to_le_bytes).collect()
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

unsafe fn publish_and_host_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let publish = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[publish],
        &[],
        &[],
    );
    let host = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::HOST,
        vk::DependencyFlags::empty(),
        &[host],
        &[],
        &[],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s14_causal_block_layer::S14CausalBlockPhysicalRange;
    use polaris_s14_runner::{build_layer_causal_batch_plan, router_kind_for_layer};

    #[test]
    fn synthetic_k4_union_metadata_preserves_route_slots_and_exact_bf16_order() {
        let routes = [
            [9, 3, 12, 7, 1, 5],
            [2, 8, 4, 10, 6, 0],
            [13, 15, 11, 14, 16, 17],
            [18, 19, 20, 21, 22, 23],
        ]
        .into_iter()
        .enumerate()
        .map(|(_lane, ids)| RouteDecision {
            layer: 7,
            kind: router_kind_for_layer(7).unwrap(),
            expert_ids: ids.to_vec(),
            weights: vec![0.20, 0.22, 0.24, 0.26, 0.28, 0.30],
        })
        .collect::<Vec<_>>();
        let batch = build_layer_causal_batch_plan(&routes).unwrap();
        let mut ranges = Vec::new();
        for expert in 0..24u16 {
            for (projection, part) in [
                (RoutedProjection::W1, RoutedRangePart::Weight),
                (RoutedProjection::W1, RoutedRangePart::Scale),
                (RoutedProjection::W2, RoutedRangePart::Weight),
                (RoutedProjection::W2, RoutedRangePart::Scale),
                (RoutedProjection::W3, RoutedRangePart::Weight),
                (RoutedProjection::W3, RoutedRangePart::Scale),
            ] {
                ranges.push(S14CausalBlockPhysicalRange {
                    expert_id: expert,
                    projection,
                    part,
                    tensor: format!("e{expert:?}"),
                    range_key: format!("e{expert:?}-{projection:?}-{part:?}"),
                    bytes: routed_range_bytes(part),
                });
            }
        }
        let range = S14CausalBlockLayerRangePlan {
            layer: 7,
            block_size: 4,
            unique_experts: 24,
            physical_ranges: 144,
            union_expert_bytes: 24 * EXPERT_PAGE_BYTES,
            ranges,
        };
        let shared = S14RaggedBranchOffsets {
            w1: 4,
            s1: 8,
            w3: 12,
            s3: 16,
            w2: 20,
            s2: 24,
        };
        let plan = build_layer_record_plan(&routes, &batch, &range, shared).unwrap();
        assert_eq!(plan.routed_metadata.len(), 24);
        assert_eq!(plan.route_weight_bits.len(), 24);
        for (index, (&expert, &weight)) in routes
            .iter()
            .flat_map(|route| route.expert_ids.iter().zip(&route.weights))
            .enumerate()
        {
            let base = expert as u32 * EXPERT_PAGE_BYTES as u32;
            let metadata = plan.routed_metadata[index];
            assert_eq!(metadata.w1, base);
            assert_eq!(metadata.w2, base + 4_456_448);
            assert_eq!(metadata.w3, base + 8_912_896);
            assert_eq!(plan.route_weight_bits[index], weight.to_bits());
        }
        assert_eq!(plan.shared_metadata, vec![shared; 4]);

        let slots = [0.1f32, 0.2, -0.3, 0.4, 0.5, -0.6];
        let shared_value = 0.7f32;
        let exact = slots
            .into_iter()
            .map(round_bf16)
            .fold(0.0f32, |sum, value| sum + value)
            + round_bf16(shared_value);
        let illegal_late_round = slots.into_iter().sum::<f32>() + shared_value;
        assert_ne!(bf16_rne(exact), bf16_rne(illegal_late_round));
    }

    fn bf16_rne(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
    }

    fn round_bf16(value: f32) -> f32 {
        f32::from_bits((bf16_rne(value) as u32) << 16)
    }
}
