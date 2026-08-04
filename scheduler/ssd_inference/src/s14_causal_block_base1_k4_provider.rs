//! position1、K=4 production causal-block 的 concrete HC/QKV resource provider。
//!
//! 本模块不执行 whole-token，也不生成 terminal/checkpoint。它只闭合 bundle 之前缺失的
//! concrete provider：真实 position1 committed state、按实际 input token 解码的 L0--L2
//! `tid2eid`、经 SHA 验证的 L3--L42 router bias、K 行真实 RoPE、43 层 current-KV
//! 输出与 paged static uploader。所有小型 allocation 都由显式 owner 管理；bundle 销毁前
//! 禁止销毁 owner。

use crate::{
    s14_causal_block_hc_qkv_recorder::{
        S14CausalBlockHcQkvLayerResources, S14CausalBlockHcQkvResourceProvider,
        S14CausalBlockHcQkvWeightOffsets, S14CausalBlockOwnedBufferSlice,
    },
    s14_causal_block_layer::S14CausalBlockLayerInput,
    s14_causal_block_prefix_producer::S14CausalBlockPrefixStateProducer,
    s14_causal_block_production_bundle::{
        S14CausalBlockProductionHcQkvResourceProvider, S14CausalBlockProductionTerminalAssets,
    },
    s14_causal_block_ratio4_boundary::S14CausalBlockRatio4BoundaryStateRecorder,
    s14_causal_block_terminal_owner::{
        S14CausalBlockTerminalHeadLeaseOwner, S14CausalBlockTerminalHeadUploadState,
    },
    s14_position0_hybrid_upload::S14Position0CausalBlockUploadLease,
    s14_position0_layer_backend::{build_position0_layer_graph_plan, S14Position0L0GraphPlan},
    s14_position0_paged_weight_arena::{S14Position0PagedArenaPlan, S14Position0PagedWeightArena},
    s14_position0_state_writeback::S14Position0StateWritebackLayout,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_position1_attention::position_rope_cos_sin,
    s14_route_postprocess_gpu::S14RoutePostprocessGpuMode,
    s14_starfold_prefetch_pipeline::{
        S14StarfoldPrefetchLayerIdentity, S14StarfoldStaticMaterializeReceipt,
        S14StarfoldStaticPageClass, S14StarfoldStaticPageIntent, S14StarfoldStaticSsdIntent,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    MaterializedTokenSource, NativeState, Position0Asset, Position0WholeTokenManifest,
    COMPRESS_RATIOS, EXPERTS_PER_TOKEN, FULL_DEPTH_LAYERS,
};
use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
};

pub const S14_BASE1_K4_BASE_POSITION: u32 = 1;
pub const S14_BASE1_K4_BLOCK_SIZE: usize = 4;

const KV_ELEMENTS_PER_LANE: u64 = 512;
const ROPE_ELEMENTS_PER_LANE: u64 = 64;
const ROUTER_EXPERTS: u64 = 256;
const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const AUX_ALIGNMENT: u64 = 256;

/// 外部 runtime 已提交的 authoritative device state。`native_state` 与 device slice
/// 必须是同一个 position1 checkpoint；本模块只借用其 committed window 行，不复制或
/// 伪造状态。
pub struct S14Base1K4AuthoritativeStateBinding {
    pub native_state: NativeState,
    pub device_state: S14CausalBlockOwnedBufferSlice,
}

/// ratio4 producer 必须为21个真实 ratio4 层逐层提供 candidate state recorder。
/// recorder 自己拥有 prefix checkpoint arena 内的目标 slice 与数值写回生命周期。
pub struct S14Base1K4ProviderInputs {
    pub context: Arc<VulkanContext>,
    pub manifest: Arc<Position0WholeTokenManifest>,
    pub weight_plan: Arc<S14Position0HybridWeightPlan>,
    pub paged_arena: Arc<S14Position0PagedWeightArena>,
    pub head_upload: Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>,
    pub upload_lease: S14Position0CausalBlockUploadLease,
    /// 当前块的真实起始位置。首块为1；后续块必须来自已提交checkpoint。
    pub base_position: u32,
    /// Provider 构造时冻结的 token 来源，禁止逐层把 prefill 冒充 draft。
    pub source: MaterializedTokenSource,
    pub authoritative: S14Base1K4AuthoritativeStateBinding,
    pub input_token_ids: [u32; S14_BASE1_K4_BLOCK_SIZE],
    pub ratio4_boundary_states: BTreeMap<u8, Arc<dyn S14CausalBlockRatio4BoundaryStateRecorder>>,
    pub prefix_state_producer: S14CausalBlockPrefixStateProducer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProviderPhase {
    Ready,
    Preparing { next_layer_index: usize },
    Complete,
    Poisoned,
}

/// provider 使用的两块小型 allocation 的显式 owner。`GpuBuffer` 没有 RAII Drop，
/// 因此 orchestrator 必须先销毁 production bundle，再调用本 owner 的 `destroy()`。
#[must_use = "bundle 销毁后必须显式 destroy base1/K4 HC/QKV external resources"]
pub struct S14Base1K4HcQkvExternalResources {
    context: Arc<VulkanContext>,
    aux: Option<Arc<GpuBuffer>>,
    current_kv: Option<Arc<GpuBuffer>>,
}

impl fmt::Debug for S14Base1K4HcQkvExternalResources {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14Base1K4HcQkvExternalResources")
            .field("context", &Arc::as_ptr(&self.context))
            .field("aux_present", &self.aux.is_some())
            .field("current_kv_present", &self.current_kv.is_some())
            .finish()
    }
}

impl S14Base1K4HcQkvExternalResources {
    pub fn destroy(&mut self) -> Result<()> {
        if self.aux.is_none() && self.current_kv.is_none() {
            return Ok(());
        }
        let aux_refs = self.aux.as_ref().map_or(0, Arc::strong_count);
        let current_refs = self.current_kv.as_ref().map_or(0, Arc::strong_count);
        if aux_refs != 1 || current_refs != 1 {
            bail!(
                "base1/K4 HC/QKV resources 仍被 provider/recorder 持有: aux_refs={aux_refs} current_kv_refs={current_refs}"
            );
        }
        let current_kv = Arc::try_unwrap(
            self.current_kv
                .take()
                .context("base1/K4 current-KV owner 缺失")?,
        )
        .map_err(|_| anyhow!("base1/K4 current-KV Arc ownership 漂移"))?;
        let aux = Arc::try_unwrap(self.aux.take().context("base1/K4 aux owner 缺失")?)
            .map_err(|_| anyhow!("base1/K4 aux Arc ownership 漂移"))?;
        current_kv.destroy(&self.context);
        aux.destroy(&self.context);
        Ok(())
    }
}

/// 已经能直接传入 `build_s14_causal_block_production_bundle` 的 concrete provider。
pub struct S14Base1K4ProductionHcQkvProvider {
    context: Arc<VulkanContext>,
    manifest: Arc<Position0WholeTokenManifest>,
    weight_plan: Arc<S14Position0HybridWeightPlan>,
    paged_arena: Arc<S14Position0PagedWeightArena>,
    head_upload: Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>,
    terminal_upload_lease: Mutex<Option<S14Position0CausalBlockUploadLease>>,
    base_position: u32,
    source: MaterializedTokenSource,
    state_layout: S14Position0StateWritebackLayout,
    authoritative_device_state: S14CausalBlockOwnedBufferSlice,
    input_token_ids: [u32; S14_BASE1_K4_BLOCK_SIZE],
    graphs: Vec<S14Position0L0GraphPlan>,
    weights: Vec<S14CausalBlockHcQkvWeightOffsets>,
    rope_by_ratio: BTreeMap<u16, S14CausalBlockOwnedBufferSlice>,
    route_aux_by_layer: Vec<S14CausalBlockOwnedBufferSlice>,
    current_kv_by_layer: Vec<S14CausalBlockOwnedBufferSlice>,
    ratio4_boundary_states: BTreeMap<u8, Arc<dyn S14CausalBlockRatio4BoundaryStateRecorder>>,
    prefix_state_producer: Option<S14CausalBlockPrefixStateProducer>,
    phase: ProviderPhase,
}

impl fmt::Debug for S14Base1K4ProductionHcQkvProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14Base1K4ProductionHcQkvProvider")
            .field("context", &Arc::as_ptr(&self.context))
            .field("base_position", &self.base_position)
            .field("terminal_upload_lease", &self.terminal_upload_lease)
            .field("input_token_ids", &self.input_token_ids)
            .field("graph_count", &self.graphs.len())
            .field("ratio4_state_count", &self.ratio4_boundary_states.len())
            .field("phase", &self.phase)
            .finish_non_exhaustive()
    }
}

/// 构建 concrete provider，并一次性上传 resident static 权重。
///
/// 失败后调用方必须丢弃同一个 `head_upload`/paged arena；static uploader 是 one-shot，
/// 不允许在部分失败后重解释为全新 runtime。
pub fn build_s14_base1_k4_production_hc_qkv_provider(
    inputs: S14Base1K4ProviderInputs,
) -> Result<(
    S14Base1K4ProductionHcQkvProvider,
    S14Base1K4HcQkvExternalResources,
)> {
    let S14Base1K4ProviderInputs {
        context,
        manifest,
        weight_plan,
        paged_arena,
        head_upload,
        upload_lease,
        base_position,
        source,
        authoritative,
        input_token_ids,
        ratio4_boundary_states,
        prefix_state_producer,
    } = inputs;

    validate_authoritative_state(&authoritative, base_position, source)?;
    if upload_lease.base_position() != base_position || upload_lease.source() != source {
        bail!(
            "base1/K4 uploader lease 与 provider base/source 漂移: lease={upload_lease:?} base={base_position} source={source:?}"
        );
    }
    manifest
        .validate()
        .map_err(|error| anyhow!("position0 manifest invalid: {error}"))?;
    weight_plan.validate(&manifest)?;
    let expected_arena_plan = S14Position0PagedArenaPlan::build(&weight_plan)?;
    if paged_arena.plan() != &expected_arena_plan {
        bail!("base1/K4 paged arena 与 manifest/weight plan 不是同源布局");
    }

    let graphs = build_graphs(&manifest, &weight_plan, &paged_arena)?;
    let weights = graphs
        .iter()
        .map(S14CausalBlockHcQkvWeightOffsets::from_position0_graph)
        .collect::<Result<Vec<_>>>()?;
    validate_ratio4_state_owners(&context, &ratio4_boundary_states, base_position)?;
    if !Arc::ptr_eq(&context, prefix_state_producer.context())
        || prefix_state_producer.arena().base_position() != base_position
        || prefix_state_producer.arena().layout().block_size != S14_BASE1_K4_BLOCK_SIZE
        || prefix_state_producer.source() != source
    {
        bail!("base1/K4 prefix state producer context/base/K 漂移");
    }

    let state_layout = S14Position0StateWritebackLayout::build(&authoritative.native_state)?;
    validate_exact_slice(
        &authoritative.device_state,
        authoritative.native_state.arena_bytes,
        "authoritative device state",
    )?;

    let route_aux_payloads = {
        let mut upload = head_upload
            .lock()
            .map_err(|_| anyhow!("base1/K4 head upload/store mutex poisoned"))?;
        let payloads =
            build_route_aux_payloads(&manifest, &graphs, input_token_ids, &mut upload.store)?;
        let S14CausalBlockTerminalHeadUploadState { uploader, store } = &mut *upload;
        if !uploader.resident_static_uploaded() {
            let receipt = uploader.upload_static_once(
                &context,
                &manifest,
                &weight_plan,
                store,
                paged_arena.as_ref(),
            )?;
            if !receipt.complete {
                bail!("base1/K4 resident static upload receipt 不完整");
            }
        }
        payloads
    };

    let rope_payloads = build_rope_payloads(base_position)?;
    let aux_layout = AuxLayout::build(&route_aux_payloads, &rope_payloads)?;
    let aux = Arc::new(host_storage_buffer(&context, aux_layout.bytes)?);
    let current_kv_bytes = current_kv_total_bytes()?;
    let current_kv = match GpuBuffer::new_vram(
        &context,
        current_kv_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
    ) {
        Ok(buffer) => Arc::new(buffer),
        Err(error) => {
            aux.destroy(&context);
            return Err(error.context("allocate base1/K4 43-layer current-KV arena"));
        }
    };

    let resource_build = (|| -> Result<_> {
        let rope_by_ratio = write_rope_payloads(&aux, &aux_layout, &rope_payloads)?;
        let route_aux_by_layer = write_route_aux_payloads(&aux, &aux_layout, &route_aux_payloads)?;
        let current_kv_by_layer = build_current_kv_slices(&current_kv)?;
        Ok((rope_by_ratio, route_aux_by_layer, current_kv_by_layer))
    })();
    let (rope_by_ratio, route_aux_by_layer, current_kv_by_layer) = match resource_build {
        Ok(value) => value,
        Err(error) => {
            current_kv.destroy(&context);
            aux.destroy(&context);
            return Err(error);
        }
    };

    let provider = S14Base1K4ProductionHcQkvProvider {
        context: Arc::clone(&context),
        manifest,
        weight_plan,
        paged_arena,
        head_upload,
        terminal_upload_lease: Mutex::new(Some(upload_lease)),
        base_position,
        source,
        state_layout,
        authoritative_device_state: authoritative.device_state,
        input_token_ids,
        graphs,
        weights,
        rope_by_ratio,
        route_aux_by_layer,
        current_kv_by_layer,
        ratio4_boundary_states,
        prefix_state_producer: Some(prefix_state_producer),
        phase: ProviderPhase::Ready,
    };
    let mut owner = S14Base1K4HcQkvExternalResources {
        context,
        aux: Some(aux),
        current_kv: Some(current_kv),
    };
    if let Err(error) = provider.validate_internal(S14_BASE1_K4_BLOCK_SIZE) {
        drop(provider);
        let cleanup = owner.destroy();
        return Err(anyhow!(
            "校验 base1/K4 production provider: {error:#}; owner cleanup={cleanup:?}"
        ));
    }
    Ok((provider, owner))
}

impl S14CausalBlockHcQkvResourceProvider for S14Base1K4ProductionHcQkvProvider {
    fn materialized_token_source(&self) -> MaterializedTokenSource {
        self.source
    }

    fn prepare_layer(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> std::result::Result<S14CausalBlockHcQkvLayerResources, String> {
        self.prepare_layer_inner(input).map_err(|error| {
            self.phase = ProviderPhase::Poisoned;
            format!("base1/K4 production HC/QKV provider: {error:#}")
        })
    }
}

impl S14CausalBlockProductionHcQkvResourceProvider for S14Base1K4ProductionHcQkvProvider {
    fn paged_weight_arena(&self) -> &Arc<S14Position0PagedWeightArena> {
        &self.paged_arena
    }

    fn validate_production_bundle(&self, block_size: usize) -> std::result::Result<(), String> {
        self.validate_internal(block_size)
            .map_err(|error| error.to_string())
    }

    fn validate_post_prefix_handoff(&self, block_size: usize) -> std::result::Result<(), String> {
        self.validate_post_prefix_handoff_inner(block_size)
            .map_err(|error| error.to_string())
    }

    fn validate_committed_block_rebind(
        &self,
        base_position: u32,
        input_token_ids: &[u32],
        checkpoint_state_bytes: u64,
    ) -> std::result::Result<(), String> {
        self.validate_committed_rebind_inner(base_position, input_token_ids, checkpoint_state_bytes)
            .map_err(|error| error.to_string())
    }

    fn plan_starfold_static_prefetch(
        &self,
        layer: S14StarfoldPrefetchLayerIdentity,
    ) -> std::result::Result<Option<S14StarfoldStaticSsdIntent>, String> {
        self.plan_static_prefetch_inner(layer)
            .map(Some)
            .map_err(|error| format!("{error:#}"))
    }

    fn materialize_starfold_static_prefetch(
        &mut self,
        intent: &S14StarfoldStaticSsdIntent,
    ) -> std::result::Result<S14StarfoldStaticMaterializeReceipt, String> {
        self.materialize_static_prefetch_inner(intent)
            .map_err(|error| format!("{error:#}"))
    }

    fn terminal_assets(
        &self,
    ) -> std::result::Result<S14CausalBlockProductionTerminalAssets, String> {
        self.terminal_assets_inner().map_err(|error| {
            let progress = match self.head_upload.lock() {
                Ok(upload) => {
                    let snapshot = upload
                        .uploader
                        .causal_block_progress_snapshot(&self.weight_plan);
                    format!(
                        "active_lease={:?} phase={:?} static_complete={} static={}/{} routed={} pending_layer={:?} head={}/{} pending_head={:?}",
                        snapshot.active_lease,
                        snapshot.lease_phase,
                        snapshot.static_complete,
                        snapshot.next_runtime_static_layer,
                        snapshot.runtime_static_layer_count,
                        snapshot.next_routed_layer,
                        snapshot.pending_layer,
                        snapshot.next_head_chunk,
                        snapshot.head_chunk_count,
                        snapshot.pending_head_chunk,
                    )
                }
                Err(_) => "unavailable=head_upload_mutex_poisoned".to_owned(),
            };
            format!("{error:#}; persistent_uploader_progress={{ {progress} }}")
        })
    }

    fn abort_block_upload_lease_after_drain(&self) -> std::result::Result<(), String> {
        let mut lease_owner = self
            .terminal_upload_lease
            .lock()
            .map_err(|_| "K4 abort upload lease mutex poisoned".to_owned())?;
        let Some(lease) = lease_owner.as_ref().copied() else {
            // 完整43层后的 terminal handoff 会 one-shot take lease，并由 terminal owner
            // 自己负责 head stream abort。只有未完成 provider 丢失 lease 才是合同错误。
            return if self.phase == ProviderPhase::Complete {
                Ok(())
            } else {
                Err("K4 pre-terminal abort 时 block-scoped upload lease 已被消费".to_owned())
            };
        };
        let mut upload = self
            .head_upload
            .lock()
            .map_err(|_| "K4 abort terminal head upload/store mutex poisoned".to_owned())?;
        upload
            .uploader
            .abort_causal_block_lease_after_drain(&self.weight_plan, &lease)
            .map_err(|error| format!("K4 abort persistent uploader lease: {error:#}"))?;
        drop(upload);
        *lease_owner = None;
        Ok(())
    }

    fn take_prefix_state_producer(
        &mut self,
    ) -> std::result::Result<S14CausalBlockPrefixStateProducer, String> {
        self.prefix_state_producer
            .take()
            .ok_or_else(|| "base1/K4 prefix state producer 已被消费".to_owned())
    }
}

impl Drop for S14Base1K4ProductionHcQkvProvider {
    fn drop(&mut self) {
        if let Some(mut producer) = self.prefix_state_producer.take() {
            let _ = producer.destroy();
        }
    }
}

fn static_page_intent(
    class: S14StarfoldStaticPageClass,
    asset: &Position0Asset,
) -> Result<S14StarfoldStaticPageIntent> {
    if asset.expert_id.is_some() {
        bail!("K4 static prefetch 禁止 routed expert asset")
    }
    S14StarfoldStaticPageIntent::new(
        class,
        format!("{}:{}:{}", asset.tensor, asset.cache_key, asset.range_key),
        asset.bytes,
    )
}

impl S14Base1K4ProductionHcQkvProvider {
    fn plan_static_prefetch_inner(
        &self,
        layer: S14StarfoldPrefetchLayerIdentity,
    ) -> Result<S14StarfoldStaticSsdIntent> {
        let manifest_layer = self
            .manifest
            .layers
            .get(usize::from(layer.layer_ordinal()))
            .context("K4 static prefetch layer ordinal 越出 manifest")?;
        if u16::from(manifest_layer.layer) != layer.layer() {
            bail!("K4 static prefetch layer identity 与 manifest 漂移");
        }
        let mut pages = Vec::with_capacity(
            manifest_layer.assets.non_expert.len()
                + manifest_layer.assets.router.len()
                + manifest_layer.assets.shared.len(),
        );
        for asset in &manifest_layer.assets.non_expert {
            let class = if asset.tensor.contains("norm") {
                S14StarfoldStaticPageClass::Normalization
            } else {
                S14StarfoldStaticPageClass::Attention
            };
            pages.push(static_page_intent(class, asset)?);
        }
        for asset in &manifest_layer.assets.router {
            pages.push(static_page_intent(
                S14StarfoldStaticPageClass::Router,
                asset,
            )?);
        }
        for asset in &manifest_layer.assets.shared {
            pages.push(static_page_intent(
                S14StarfoldStaticPageClass::SharedExpert,
                asset,
            )?);
        }
        S14StarfoldStaticSsdIntent::new(layer, pages)
    }

    fn materialize_static_prefetch_inner(
        &mut self,
        intent: &S14StarfoldStaticSsdIntent,
    ) -> Result<S14StarfoldStaticMaterializeReceipt> {
        let expected = self.plan_static_prefetch_inner(intent.layer())?;
        if &expected != intent {
            bail!("K4 static prefetch intent 与 provider manifest 漂移");
        }
        let manifest_layer = &self.manifest.layers[usize::from(intent.layer().layer_ordinal())];
        let assets = manifest_layer
            .assets
            .non_expert
            .iter()
            .chain(&manifest_layer.assets.router)
            .chain(&manifest_layer.assets.shared)
            .cloned()
            .collect::<Vec<_>>();
        let mapped = {
            let mut upload = self
                .head_upload
                .lock()
                .map_err(|_| anyhow!("K4 static prefetch head upload/store mutex poisoned"))?;
            upload
                .store
                .map_verified_batch(&assets)
                .context("K4 static L+2 proof/SHA/mmap 热化失败")?
        };
        let bytes = mapped.iter().try_fold(0u64, |sum, asset| {
            let bytes = u64::try_from(asset.bytes().len())
                .context("K4 static prefetch mapped asset bytes 超出 u64")?;
            sum.checked_add(bytes)
                .context("K4 static prefetch mapped bytes overflow")
        })?;
        let receipt = S14StarfoldStaticMaterializeReceipt {
            layer: intent.layer(),
            assets: mapped.len(),
            bytes,
        };
        receipt.validate_for(intent)?;
        Ok(receipt)
    }

    fn terminal_assets_inner(&self) -> Result<S14CausalBlockProductionTerminalAssets> {
        if self.phase != ProviderPhase::Complete {
            bail!(
                "K4 terminal assets 要求 provider phase=Complete，actual={:?}",
                self.phase
            );
        }
        if self.prefix_state_producer.is_some() {
            bail!("K4 terminal assets 要求 prefix producer 已唯一移交给 recorder");
        }
        self.weight_plan.validate(&self.manifest)?;
        let expected_arena_plan = S14Position0PagedArenaPlan::build(&self.weight_plan)?;
        if self.paged_arena.plan() != &expected_arena_plan {
            bail!("K4 terminal assets 的 manifest/weight plan/paged arena 不同源");
        }
        let mut lease_owner = self
            .terminal_upload_lease
            .lock()
            .map_err(|_| anyhow!("K4 terminal upload lease mutex poisoned"))?;
        let lease = lease_owner
            .as_ref()
            .copied()
            .context("K4 terminal_assets upload lease 已被 one-shot 消费")?;
        let upload = self
            .head_upload
            .lock()
            .map_err(|_| anyhow!("K4 terminal head upload/store mutex poisoned"))?;
        if !upload.uploader.resident_static_uploaded() {
            bail!("K4 terminal head uploader resident static 尚未上传");
        }
        upload
            .uploader
            .validate_causal_block_terminal_head_stream_for_lease(&self.weight_plan, &lease)
            .context("K4 terminal head uploader 尚未完成本 block static stream")?;
        drop(upload);
        let lease = lease_owner
            .take()
            .context("K4 terminal upload lease one-shot take 漂移")?;
        drop(lease_owner);
        Ok(S14CausalBlockProductionTerminalAssets {
            manifest: Arc::clone(&self.manifest),
            weight_plan: Arc::clone(&self.weight_plan),
            head_upload: S14CausalBlockTerminalHeadLeaseOwner::new(
                Arc::clone(&self.head_upload),
                lease,
            ),
        })
    }

    fn validate_committed_rebind_inner(
        &self,
        base_position: u32,
        input_token_ids: &[u32],
        checkpoint_state_bytes: u64,
    ) -> Result<()> {
        self.validate_internal(S14_BASE1_K4_BLOCK_SIZE)?;
        validate_ratio4_state_owners(&self.context, &self.ratio4_boundary_states, base_position)?;
        let prefix = self
            .prefix_state_producer
            .as_ref()
            .context("K4 committed rebind 缺少 prefix producer")?;
        if base_position != self.base_position
            || input_token_ids != self.input_token_ids
            || checkpoint_state_bytes != self.authoritative_device_state.bytes
            || !Arc::ptr_eq(&self.context, prefix.context())
            || prefix.arena().base_position() != base_position
            || prefix.arena().layout().checkpoint_state_bytes != checkpoint_state_bytes
        {
            bail!("K4 committed rebind provider 的 base/input/checkpoint/prefix identity 漂移");
        }
        Ok(())
    }

    fn prepare_layer_inner(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockHcQkvLayerResources> {
        let next_layer_index = match self.phase {
            ProviderPhase::Ready => 0,
            ProviderPhase::Preparing { next_layer_index } => next_layer_index,
            ProviderPhase::Complete => bail!("43层已经全部准备，禁止重复 prepare"),
            ProviderPhase::Poisoned => bail!("provider 已 poisoned"),
        };
        let &expected_layer = FULL_DEPTH_LAYERS
            .get(next_layer_index)
            .context("base1/K4 next layer index 越界")?;
        if input.base_position != self.base_position
            || input.layer != expected_layer
            || input.input_token_ids != self.input_token_ids
            || input.source != self.source
        {
            bail!("base1/K4 input base/layer/token/source 漂移");
        }

        let static_receipt = {
            let upload_lease = self
                .terminal_upload_lease
                .lock()
                .map_err(|_| anyhow!("base1/K4 terminal upload lease mutex poisoned"))?
                .as_ref()
                .copied()
                .context("base1/K4 static prepare 时 upload lease 已被 terminal 消费")?;
            let mut upload = self
                .head_upload
                .lock()
                .map_err(|_| anyhow!("base1/K4 head upload/store mutex poisoned"))?;
            let S14CausalBlockTerminalHeadUploadState { uploader, store } = &mut *upload;
            uploader.prepare_next_static_layer_for_causal_block(
                &upload_lease,
                &self.context,
                &self.manifest,
                &self.weight_plan,
                store,
                self.paged_arena.as_ref(),
            )?
        };
        if static_receipt.layer != expected_layer {
            bail!("base1/K4 static uploader 层序漂移");
        }

        let graph = self
            .graphs
            .get(next_layer_index)
            .context("base1/K4 graph 缺层")?;
        let layer_state = self
            .state_layout
            .layer(expected_layer)
            .context("base1/K4 state layout 缺层")?;
        let committed_range = &layer_state.committed_window_state_range;
        let committed_offset = self
            .authoritative_device_state
            .offset
            .checked_add(committed_range.start)
            .context("base1/K4 committed KV offset overflow")?;
        let committed = S14CausalBlockOwnedBufferSlice {
            buffer: Arc::clone(&self.authoritative_device_state.buffer),
            offset: committed_offset,
            bytes: committed_range.end - committed_range.start,
        };
        let rope = self
            .rope_by_ratio
            .get(&COMPRESS_RATIOS[usize::from(expected_layer)])
            .context("base1/K4 layer RoPE ratio 缺失")?
            .clone();
        let mut resources = S14CausalBlockHcQkvLayerResources::from_ready_paged_static_layer(
            expected_layer,
            Arc::clone(&self.paged_arena),
            self.weights[next_layer_index],
            graph.route_mode,
            committed,
            self.current_kv_by_layer[next_layer_index].clone(),
            rope,
            self.route_aux_by_layer[next_layer_index].clone(),
        )?;
        if COMPRESS_RATIOS[usize::from(expected_layer)] == 4 {
            resources = resources.with_ratio4_boundary_state(
                self.ratio4_boundary_states
                    .get(&expected_layer)
                    .context("base1/K4 ratio4 layer 缺少 candidate state owner")?
                    .clone(),
            )?;
        }

        let following = next_layer_index + 1;
        self.phase = if following == FULL_DEPTH_LAYERS.len() {
            ProviderPhase::Complete
        } else {
            ProviderPhase::Preparing {
                next_layer_index: following,
            }
        };
        Ok(resources)
    }

    fn validate_internal(&self, block_size: usize) -> Result<()> {
        self.validate_core_internal(block_size)?;
        let prefix = self
            .prefix_state_producer
            .as_ref()
            .context("base1/K4 provider 缺少 prefix producer")?;
        if prefix.source() != self.source || prefix.arena().base_position() != self.base_position {
            bail!("base1/K4 provider prefix source/base identity 漂移");
        }
        Ok(())
    }

    fn validate_post_prefix_handoff_inner(&self, block_size: usize) -> Result<()> {
        self.validate_core_internal(block_size)?;
        if self.prefix_state_producer.is_some() {
            bail!("base1/K4 provider post-handoff 仍持有 prefix producer");
        }
        Ok(())
    }

    fn validate_core_internal(&self, block_size: usize) -> Result<()> {
        if block_size != S14_BASE1_K4_BLOCK_SIZE
            || self.state_layout.position != self.base_position
            || self.graphs.len() != FULL_DEPTH_LAYERS.len()
            || self.weights.len() != FULL_DEPTH_LAYERS.len()
            || self.route_aux_by_layer.len() != FULL_DEPTH_LAYERS.len()
            || self.current_kv_by_layer.len() != FULL_DEPTH_LAYERS.len()
            || self.phase != ProviderPhase::Ready
        {
            bail!("base1/K4 provider core readiness K/position/43层/phase 漂移");
        }
        let lease_owner = self
            .terminal_upload_lease
            .lock()
            .map_err(|_| anyhow!("base1/K4 terminal upload lease mutex poisoned"))?;
        let lease = lease_owner
            .as_ref()
            .context("base1/K4 provider core 缺少 block-scoped upload lease")?;
        if lease.base_position() != self.base_position || lease.source() != self.source {
            bail!("base1/K4 provider core uploader lease base/source 漂移");
        }
        drop(lease_owner);
        validate_ratio4_state_owners(
            &self.context,
            &self.ratio4_boundary_states,
            self.base_position,
        )?;
        if self
            .graphs
            .iter()
            .zip(FULL_DEPTH_LAYERS)
            .any(|(graph, layer)| graph.layer != layer)
        {
            bail!("base1/K4 provider graph 层序漂移");
        }
        for ratio in [0u16, 4, 128] {
            validate_exact_slice(
                self.rope_by_ratio
                    .get(&ratio)
                    .context("base1/K4 provider RoPE ratio 缺失")?,
                rope_bytes()?,
                "K-row RoPE",
            )?;
        }
        Ok(())
    }
}

fn validate_authoritative_state(
    binding: &S14Base1K4AuthoritativeStateBinding,
    base_position: u32,
    _source: MaterializedTokenSource,
) -> Result<()> {
    binding
        .native_state
        .validate_for(polaris_s14_runner::GraphProfile::FullDepth43NativeTop6)
        .context("validate base1/K4 authoritative native state")?;
    // generation base0 的 StarWave committed-origin 强校验位于 adapter 显式入口；provider
    // 只绑定 native/device 物理 state，不能单独授权 launch。
    if binding.native_state.position != base_position || binding.native_state.poisoned {
        bail!("K4 authoritative state/source 必须匹配真实 base 且未 poisoned");
    }
    Ok(())
}

fn build_graphs(
    manifest: &Position0WholeTokenManifest,
    weights: &S14Position0HybridWeightPlan,
    paged_arena: &S14Position0PagedWeightArena,
) -> Result<Vec<S14Position0L0GraphPlan>> {
    FULL_DEPTH_LAYERS
        .iter()
        .enumerate()
        .map(|(index, &layer)| {
            let layout = paged_arena
                .plan()
                .physical
                .static_layers
                .get(index)
                .with_context(|| format!("base1/K4 paged arena 缺少 L{layer} static layout"))?;
            let graph = build_position0_layer_graph_plan(manifest, weights, layout, index)?;
            if graph.layer != layer || graph.layer_index != index {
                bail!("base1/K4 L{layer} graph identity 漂移");
            }
            Ok(graph)
        })
        .collect()
}

fn validate_ratio4_state_owners(
    _context: &Arc<VulkanContext>,
    states: &BTreeMap<u8, Arc<dyn S14CausalBlockRatio4BoundaryStateRecorder>>,
    base_position: u32,
) -> Result<()> {
    let expected = FULL_DEPTH_LAYERS
        .iter()
        .copied()
        .filter(|&layer| COMPRESS_RATIOS[usize::from(layer)] == 4)
        .collect::<Vec<_>>();
    if states.keys().copied().collect::<Vec<_>>() != expected {
        bail!("base1/K4 ratio4 candidate state 必须精确覆盖21个真实层");
    }
    for (&layer, state) in states {
        let binding = state.candidate_state_binding();
        binding.validate(state.candidate_state_owner())?;
        if binding.layer != layer
            || binding.base_position != base_position
            || binding.block_size as usize != S14_BASE1_K4_BLOCK_SIZE
        {
            bail!("base1/K4 L{layer} ratio4 candidate state context/base/K 漂移");
        }
    }
    Ok(())
}

fn build_route_aux_payloads(
    manifest: &Position0WholeTokenManifest,
    graphs: &[S14Position0L0GraphPlan],
    input_token_ids: [u32; S14_BASE1_K4_BLOCK_SIZE],
    store: &mut crate::s14_position0_mapped_assets::VerifiedMappedAssetStore,
) -> Result<Vec<Vec<u8>>> {
    let mut payloads = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
    for (index, graph) in graphs.iter().enumerate() {
        let layer = manifest
            .layers
            .get(index)
            .with_context(|| format!("base1/K4 manifest 缺少 L{}", graph.layer))?;
        let (suffix, expected_dtype, expected_shape) = match graph.route_mode {
            S14RoutePostprocessGpuMode::PhysicalIds => (
                "ffn.gate.tid2eid",
                "I64",
                vec![129_280, EXPERTS_PER_TOKEN as u64],
            ),
            S14RoutePostprocessGpuMode::BiasTop6 => ("ffn.gate.bias", "F32", vec![ROUTER_EXPERTS]),
        };
        let tensor = format!("layers.{}.{}", graph.layer, suffix);
        let asset = unique_asset(
            layer.assets.iter().filter(|asset| asset.tensor == tensor),
            &tensor,
        )?;
        if asset.dtype != expected_dtype || asset.shape != expected_shape {
            bail!("base1/K4 {tensor} dtype/shape 漂移");
        }
        let mapped = store.map_verified_batch(&[asset.clone()])?;
        let bytes = mapped
            .first()
            .context("base1/K4 route aux verified mmap 缺失")?
            .bytes();
        let payload = match graph.route_mode {
            S14RoutePostprocessGpuMode::PhysicalIds => {
                let mut ids = Vec::with_capacity(S14_BASE1_K4_BLOCK_SIZE * EXPERTS_PER_TOKEN);
                for token in input_token_ids {
                    ids.extend(graph.decode_tid2eid_row_for_token(asset, bytes, token)?);
                }
                bytemuck::cast_slice(&ids).to_vec()
            }
            S14RoutePostprocessGpuMode::BiasTop6 => {
                if bytes.len() as u64 != ROUTER_EXPERTS * F32_BYTES
                    || bytemuck::cast_slice::<u8, f32>(bytes)
                        .iter()
                        .any(|value| !value.is_finite())
                {
                    bail!("base1/K4 {tensor} bytes/finite 漂移");
                }
                bytes.to_vec()
            }
        };
        payloads.push(payload);
    }
    Ok(payloads)
}

fn build_rope_payloads(base_position: u32) -> Result<BTreeMap<u16, Vec<u8>>> {
    let mut output = BTreeMap::new();
    for ratio in [0u16, 4, 128] {
        let mut rows =
            Vec::<f32>::with_capacity(S14_BASE1_K4_BLOCK_SIZE * ROPE_ELEMENTS_PER_LANE as usize);
        for lane in 0..S14_BASE1_K4_BLOCK_SIZE {
            let position = base_position
                .checked_add(lane as u32)
                .context("base1/K4 RoPE position overflow")?;
            rows.extend(position_rope_cos_sin(position, u32::from(ratio))?);
        }
        if rows.iter().any(|value| !value.is_finite()) {
            bail!("base1/K4 ratio{ratio} RoPE 包含非有限值");
        }
        output.insert(ratio, bytemuck::cast_slice(&rows).to_vec());
    }
    Ok(output)
}

#[derive(Debug)]
struct AuxLayout {
    rope_offsets: BTreeMap<u16, u64>,
    route_offsets: Vec<u64>,
    bytes: u64,
}

impl AuxLayout {
    fn build(route_payloads: &[Vec<u8>], rope_payloads: &BTreeMap<u16, Vec<u8>>) -> Result<Self> {
        if route_payloads.len() != FULL_DEPTH_LAYERS.len() || rope_payloads.len() != 3 {
            bail!("base1/K4 aux payload count 漂移");
        }
        let mut cursor = 0u64;
        let mut rope_offsets = BTreeMap::new();
        for ratio in [0u16, 4, 128] {
            cursor = align_up(cursor, AUX_ALIGNMENT)?;
            rope_offsets.insert(ratio, cursor);
            cursor = cursor
                .checked_add(u64::try_from(
                    rope_payloads
                        .get(&ratio)
                        .context("base1/K4 rope payload 缺失")?
                        .len(),
                )?)
                .context("base1/K4 rope layout overflow")?;
        }
        let mut route_offsets = Vec::with_capacity(route_payloads.len());
        for payload in route_payloads {
            cursor = align_up(cursor, AUX_ALIGNMENT)?;
            route_offsets.push(cursor);
            cursor = cursor
                .checked_add(u64::try_from(payload.len())?)
                .context("base1/K4 route aux layout overflow")?;
        }
        Ok(Self {
            rope_offsets,
            route_offsets,
            bytes: align_up(cursor, AUX_ALIGNMENT)?,
        })
    }
}

fn write_rope_payloads(
    buffer: &Arc<GpuBuffer>,
    layout: &AuxLayout,
    payloads: &BTreeMap<u16, Vec<u8>>,
) -> Result<BTreeMap<u16, S14CausalBlockOwnedBufferSlice>> {
    let mut slices = BTreeMap::new();
    for ratio in [0u16, 4, 128] {
        let payload = payloads.get(&ratio).context("base1/K4 rope payload 缺失")?;
        let offset = *layout
            .rope_offsets
            .get(&ratio)
            .context("base1/K4 rope offset 缺失")?;
        write_mapped(buffer, offset, payload, "RoPE")?;
        slices.insert(
            ratio,
            S14CausalBlockOwnedBufferSlice {
                buffer: Arc::clone(buffer),
                offset,
                bytes: u64::try_from(payload.len())?,
            },
        );
    }
    Ok(slices)
}

fn write_route_aux_payloads(
    buffer: &Arc<GpuBuffer>,
    layout: &AuxLayout,
    payloads: &[Vec<u8>],
) -> Result<Vec<S14CausalBlockOwnedBufferSlice>> {
    payloads
        .iter()
        .zip(&layout.route_offsets)
        .map(|(payload, &offset)| {
            write_mapped(buffer, offset, payload, "route aux")?;
            Ok(S14CausalBlockOwnedBufferSlice {
                buffer: Arc::clone(buffer),
                offset,
                bytes: u64::try_from(payload.len())?,
            })
        })
        .collect()
}

fn build_current_kv_slices(buffer: &Arc<GpuBuffer>) -> Result<Vec<S14CausalBlockOwnedBufferSlice>> {
    let layer_bytes = current_kv_layer_bytes()?;
    FULL_DEPTH_LAYERS
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let offset = layer_bytes
                .checked_mul(index as u64)
                .context("base1/K4 current-KV layer offset overflow")?;
            Ok(S14CausalBlockOwnedBufferSlice {
                buffer: Arc::clone(buffer),
                offset,
                bytes: layer_bytes,
            })
        })
        .collect()
}

fn host_storage_buffer(context: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        context,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(),
        true,
    )
}

fn write_mapped(buffer: &GpuBuffer, offset: u64, payload: &[u8], label: &str) -> Result<()> {
    let end = offset
        .checked_add(u64::try_from(payload.len())?)
        .with_context(|| format!("base1/K4 {label} write overflow"))?;
    if buffer.mapped().is_null() || offset % 4 != 0 || end > buffer.size() {
        bail!("base1/K4 {label} mapped/alignment/capacity 漂移");
    }
    unsafe { buffer.write_at(usize::try_from(offset)?, payload) };
    Ok(())
}

fn validate_exact_slice(
    slice: &S14CausalBlockOwnedBufferSlice,
    expected_bytes: u64,
    label: &str,
) -> Result<()> {
    let end = slice
        .offset
        .checked_add(slice.bytes)
        .with_context(|| format!("{label} range overflow"))?;
    if slice.buffer.handle() == vk::Buffer::null()
        || slice.offset % 4 != 0
        || slice.bytes != expected_bytes
        || end > slice.buffer.size()
    {
        bail!("base1/K4 {label} handle/alignment/bytes/capacity 漂移");
    }
    Ok(())
}

fn unique_asset<'a>(
    mut assets: impl Iterator<Item = &'a Position0Asset>,
    label: &str,
) -> Result<&'a Position0Asset> {
    let first = assets
        .next()
        .with_context(|| format!("base1/K4 manifest 缺少 {label}"))?;
    if assets.next().is_some() {
        bail!("base1/K4 manifest {label} 不唯一");
    }
    Ok(first)
}

fn rope_bytes() -> Result<u64> {
    (S14_BASE1_K4_BLOCK_SIZE as u64)
        .checked_mul(ROPE_ELEMENTS_PER_LANE)
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .context("base1/K4 rope bytes overflow")
}

fn current_kv_layer_bytes() -> Result<u64> {
    (S14_BASE1_K4_BLOCK_SIZE as u64)
        .checked_mul(KV_ELEMENTS_PER_LANE)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .context("base1/K4 current-KV layer bytes overflow")
}

fn current_kv_total_bytes() -> Result<u64> {
    current_kv_layer_bytes()?
        .checked_mul(FULL_DEPTH_LAYERS.len() as u64)
        .context("base1/K4 current-KV arena bytes overflow")
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("base1/K4 alignment overflow")
}
