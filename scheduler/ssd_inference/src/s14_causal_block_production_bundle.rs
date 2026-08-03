//! S14 causal-block 三段 concrete Vulkan 数据面的 production 组合根。
//!
//! 本模块只负责 owner 接线与生命周期，不创建、填充或替代任何模型权重。所有静态
//! arena、HC/QKV layer provider、hidden banks 与 terminal production source 都必须显式
//! 绑定到同一个 `Arc<VulkanContext>`。factory 在首次 Vulkan allocation 前校验 K、
//! FullDepth catalog、arena ledger 与 provider readiness；任一缺项均 fail-closed。

use crate::{
    s14_causal_block_grouped_moe_recorder::build_s14_causal_block_paged_moe_adapter_with_shared_store,
    s14_causal_block_hc_qkv_adapter::S14CausalBlockProductionHcQkvAdapter,
    s14_causal_block_hc_qkv_recorder::{
        S14CausalBlockHcQkvResourceProvider, S14CausalBlockHiddenBank,
        S14CausalBlockProductionHcQkvLayerRecorder,
    },
    s14_causal_block_layer::{
        S14CausalBlockAbortReceipt, S14CausalBlockAttentionRouterOutput,
        S14CausalBlockBeginReceipt, S14CausalBlockCheckpointBackend, S14CausalBlockFinalOutput,
        S14CausalBlockFullDepthBackend, S14CausalBlockGroupedMoeOutput,
        S14CausalBlockHiddenBinding, S14CausalBlockLayerBackend, S14CausalBlockLayerInput,
        S14CausalBlockLayerRangePlan, S14CausalBlockPostSealReceipt, S14CausalBlockSealReceipt,
        S14CausalBlockUnionBankBinding, S14CausalBlockUnionMaterializeReceipt,
    },
    s14_causal_block_moe_adapter::S14CausalBlockVulkanMoeAdapter,
    s14_causal_block_prefix_producer::S14CausalBlockPrefixStateProducer,
    s14_causal_block_terminal::{
        S14CausalBlockBatchedTerminalRecorder, S14CausalBlockCheckpointArenaPool,
        S14CausalBlockCheckpointArenaTelemetry,
    },
    s14_causal_block_terminal_adapter::{
        s14_causal_block_terminal_production_channel, S14CausalBlockHostCandidateFinalizer,
        S14CausalBlockTerminalProductionAdapter, S14CausalBlockTerminalProductionPublisher,
        S14CausalBlockTerminalProductionSource, S14CausalBlockTerminalProviderTelemetry,
    },
    s14_causal_block_terminal_owner::{
        S14CausalBlockProductionTerminalResourceOwner, S14CausalBlockTerminalHeadLeaseOwner,
        S14CausalBlockTerminalPublishReceipt,
    },
    s14_causal_block_union_materializer::S14CausalBlockSharedMappedAssetStore,
    s14_causal_block_vulkan_backend::S14CausalBlockVulkanBackend,
    s14_dynamic_page_cache_readiness::DynamicPageFetchMode,
    s14_dynamic_routed_page_plan::{FullDepthExpertCatalog, OnlineTop6},
    s14_head_chunk_argmax::S14_HEAD_CHUNK_COUNT,
    s14_position0_paged_weight_arena::{
        S14Position0PagedArenaPlan, S14Position0PagedWeightArena, S14Position0StaticLayerBinding,
        S14_POSITION0_STATIC_STREAM_BANKS,
    },
    s14_position0_weight_plan::{S14Position0HybridWeightPlan, S14_POSITION0_ROLLING_BANKS},
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    LayerCausalBatchPlan, Position0WholeTokenManifest, RouteDecision, EXPERTS_PER_TOKEN,
    FULL_DEPTH_LAYERS, N_ROUTED_EXPERTS,
};
use std::{
    fmt,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

const HIDDEN_BANK_COUNT: usize = 2;

/// FullDepth43 provider 在最后一层完成后交给 StarFold terminal 的同源只读 owner。
/// trait 没有默认构造，production provider 必须显式返回其真实 manifest/weight plan 与
/// 已执行本 block 静态层上传的同一个 uploader/store。
pub struct S14CausalBlockProductionTerminalAssets {
    pub manifest: Arc<Position0WholeTokenManifest>,
    pub weight_plan: Arc<S14Position0HybridWeightPlan>,
    pub head_upload: S14CausalBlockTerminalHeadLeaseOwner,
}

impl fmt::Debug for S14CausalBlockProductionTerminalAssets {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockProductionTerminalAssets")
            .field("manifest", &Arc::as_ptr(&self.manifest))
            .field("weight_plan", &Arc::as_ptr(&self.weight_plan))
            .field("head_upload", &self.head_upload)
            .finish()
    }
}

/// 将 loader 已在某个 context 下创建的资源与其 owner identity 一并移动到 factory。
///
/// 这是 owner contract，不会把外来 Vulkan handle 重新解释成当前 device 的资源。调用方只应
/// 在资源确由传入 `context` 创建且仍由 `value` 强所有时构造它。
pub struct S14CausalBlockContextBound<T> {
    context: Arc<VulkanContext>,
    value: T,
}

impl<T> S14CausalBlockContextBound<T> {
    pub fn new(context: Arc<VulkanContext>, value: T) -> Self {
        Self { context, value }
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub(crate) fn into_parts(self) -> (Arc<VulkanContext>, T) {
        (self.context, self.value)
    }
}

impl<T: fmt::Debug> fmt::Debug for S14CausalBlockContextBound<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockContextBound")
            .field("context", &Arc::as_ptr(&self.context))
            .field("value", &self.value)
            .finish()
    }
}

/// HC/QKV provider 的 builder-time readiness 边界。实现必须验证 FullDepth43 静态权重、
/// K-row KV/RoPE/route-aux 容量与 source owner 生命周期；不能在这里返回占位资源。
pub trait S14CausalBlockProductionHcQkvResourceProvider:
    S14CausalBlockHcQkvResourceProvider
{
    /// HC/QKV 必须从与 grouped-MoE 相同的 paged arena 取得 static/routed owner。
    /// `prepare_layer` 对 streamed static 层还必须先完成 verified proof/SHA/mmap/upload，
    /// 使 arena 的 `ready_static_layer(layer)` 在返回资源前成立。
    fn paged_weight_arena(&self) -> &Arc<S14Position0PagedWeightArena>;

    fn validate_production_bundle(&self, block_size: usize) -> Result<(), String>;

    /// `take_prefix_state_producer` 成功后的第二阶段合同。此时 provider 核心资源必须仍
    /// 完整且 ready，但 producer 必须已经唯一移交给 recorder；不能复用 pre-handoff
    /// readiness 把合法的所有权转移误判为缺失。
    fn validate_post_prefix_handoff(&self, block_size: usize) -> Result<(), String>;

    /// 常驻 K4 execution owners 不变时，校验新 provider 绑定到本次 committed
    /// checkpoint 的 position/input/state ABI。实现必须只读且 fail-closed。
    fn validate_committed_block_rebind(
        &self,
        base_position: u32,
        input_token_ids: &[u32],
        checkpoint_state_bytes: u64,
    ) -> Result<(), String>;

    /// 只能在本 block 的43层 static upload 全部完成后导出。禁止 mock/default 或重新创建
    /// uploader；terminal 必须继续消费 provider 当前持有的同一个 Arc owner。
    fn terminal_assets(&self) -> Result<S14CausalBlockProductionTerminalAssets, String>;

    /// K-prefix producer 必须与HC/QKV recorder进入同一layer command；factory只允许在
    /// recorder构造完成后消费一次，不能在terminal begin前预制host checkpoint。
    fn take_prefix_state_producer(&mut self) -> Result<S14CausalBlockPrefixStateProducer, String>;
}

/// 同一个 production bundle 的一次性 post-seal producer。调用发生时43层已 seal/drain，
/// final terminal/head 尚未提交；实现者必须用传入 publisher 发布一份真实 owner 保活的
/// source，不能在 hook 内循环调用单 token forward。
trait S14CausalBlockProductionPostSealTerminalProducer: fmt::Debug {
    fn block_size(&self) -> usize;
    fn base_position(&self) -> u32;

    fn publish_terminal_source(
        self: Box<Self>,
        publisher: &S14CausalBlockProductionTerminalPublisher,
        completed_layers: usize,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<(), String>;
}

#[derive(Debug)]
struct S14CausalBlockProductionOwnedTerminalProducer {
    owner: Arc<S14CausalBlockProductionTerminalResourceOwner>,
    host_candidates: Box<dyn S14CausalBlockHostCandidateFinalizer>,
}

impl S14CausalBlockProductionPostSealTerminalProducer
    for S14CausalBlockProductionOwnedTerminalProducer
{
    fn block_size(&self) -> usize {
        self.owner.block_size()
    }

    fn base_position(&self) -> u32 {
        self.host_candidates.base_position()
    }

    fn publish_terminal_source(
        self: Box<Self>,
        publisher: &S14CausalBlockProductionTerminalPublisher,
        completed_layers: usize,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<(), String> {
        let Self {
            owner,
            host_candidates,
        } = *self;
        let receipt = owner.record_and_publish(
            publisher,
            base_position,
            final_hidden,
            routes_by_position.to_vec(),
            host_candidates,
        )?;
        validate_terminal_publish_receipt(
            receipt,
            completed_layers,
            base_position,
            final_hidden.block_size,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockProductionBundleShape {
    block_size: usize,
    checkpoint_state_bytes: u64,
    checkpoint_slots: usize,
}

impl S14CausalBlockProductionBundleShape {
    pub fn new(
        block_size: usize,
        checkpoint_state_bytes: u64,
        checkpoint_slots: usize,
    ) -> Result<Self> {
        if !matches!(block_size, 4 | 8) {
            bail!("S14 production bundle K 只允许4或8");
        }
        if checkpoint_state_bytes == 0 || checkpoint_slots == 0 {
            bail!("S14 production bundle checkpoint state bytes/slots 不能为0");
        }
        Ok(Self {
            block_size,
            checkpoint_state_bytes,
            checkpoint_slots,
        })
    }

    pub fn block_size(self) -> usize {
        self.block_size
    }

    pub fn checkpoint_state_bytes(self) -> u64 {
        self.checkpoint_state_bytes
    }

    pub fn checkpoint_slots(self) -> usize {
        self.checkpoint_slots
    }
}

/// factory 的完整必需输入。字段没有 `Option`：缺 catalog、arena、provider、hidden banks 或
/// context 时无法构造 production bundle，更不会得到部分接线的 backend。
pub struct S14CausalBlockProductionBundleInputs<P> {
    pub context: Arc<VulkanContext>,
    pub shape: S14CausalBlockProductionBundleShape,
    pub hc_qkv_provider: S14CausalBlockContextBound<P>,
    pub hidden_banks: S14CausalBlockContextBound<[S14CausalBlockHiddenBank; HIDDEN_BANK_COUNT]>,
    pub catalog: FullDepthExpertCatalog,
    pub cache_root: PathBuf,
    pub fetch_mode: DynamicPageFetchMode,
    pub union_mapped_store: S14CausalBlockSharedMappedAssetStore,
    pub static_arena: S14CausalBlockContextBound<Arc<S14Position0PagedWeightArena>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BundlePhase {
    Idle,
    Recording {
        base_position: u32,
        completed_layers: usize,
    },
    LayersSealed {
        base_position: u32,
    },
    TerminalSourcePublished {
        base_position: u32,
    },
    AwaitingExportAck,
    Destroying,
    Poisoned,
    Destroyed,
}

type SharedBundlePhase = Arc<Mutex<BundlePhase>>;

/// 只允许在同 bundle 的 FullDepth43 seal 后发布一次同 identity terminal source。所有 clone
/// 共享 lifecycle；bundle 开始销毁后，旧 publisher 无法向已经失去 consumer 的 channel 发布。
#[derive(Clone)]
pub(crate) struct S14CausalBlockProductionTerminalPublisher {
    context: Arc<VulkanContext>,
    block_size: usize,
    phase: SharedBundlePhase,
    inner: S14CausalBlockTerminalProductionPublisher,
}

impl fmt::Debug for S14CausalBlockProductionTerminalPublisher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockProductionTerminalPublisher")
            .field("block_size", &self.block_size)
            .field("phase", &self.phase.lock().ok().map(|phase| *phase))
            .field("telemetry", &self.inner.telemetry().ok())
            .finish()
    }
}

impl S14CausalBlockProductionTerminalPublisher {
    pub(crate) fn publish(
        &self,
        source: S14CausalBlockContextBound<S14CausalBlockTerminalProductionSource>,
    ) -> Result<(), String> {
        let (source_context, source) = source.into_parts();
        if !Arc::ptr_eq(&self.context, &source_context) {
            return Err("terminal production source 与 bundle VulkanContext owner 漂移".into());
        }
        let mut phase = lock_phase(&self.phase)?;
        let base_position = match *phase {
            BundlePhase::LayersSealed { base_position } => base_position,
            _ => {
                return Err("terminal production source 只能在同一 FullDepth43 seal 后发布".into())
            }
        };
        if source.base_position != base_position
            || source.final_hidden.block_size != self.block_size
            || source.completed_layers != FULL_DEPTH_LAYERS.len()
        {
            return Err(
                "terminal production source 与 bundle base/K/FullDepth43 identity 漂移".into(),
            );
        }
        self.inner.publish(source)?;
        *phase = BundlePhase::TerminalSourcePublished { base_position };
        Ok(())
    }

    fn telemetry(&self) -> Result<S14CausalBlockTerminalProviderTelemetry, String> {
        self.inner.telemetry()
    }
}

/// 完整 production backend 与其 terminal producer channel/checkpoint pool 的唯一 owner。
#[must_use = "production bundle 必须由 orchestrator 持有，并在 VulkanContext 前显式 destroy"]
pub struct S14CausalBlockProductionBundle {
    context: Arc<VulkanContext>,
    paged_arena: Arc<S14Position0PagedWeightArena>,
    block_size: usize,
    backend: Option<S14CausalBlockVulkanBackend>,
    terminal_publisher: S14CausalBlockProductionTerminalPublisher,
    checkpoint_pool: Arc<S14CausalBlockCheckpointArenaPool>,
    hidden_banks: [S14CausalBlockHiddenBank; HIDDEN_BANK_COUNT],
    post_seal_terminal_producer: Option<Box<dyn S14CausalBlockProductionPostSealTerminalProducer>>,
    phase: SharedBundlePhase,
}

impl fmt::Debug for S14CausalBlockProductionBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockProductionBundle")
            .field("context", &Arc::as_ptr(&self.context))
            .field("block_size", &self.block_size)
            .field("phase", &self.phase.lock().ok().map(|phase| *phase))
            .field("backend_present", &self.backend.is_some())
            .field(
                "post_seal_terminal_producer_present",
                &self.post_seal_terminal_producer.is_some(),
            )
            .field("checkpoint_pool", &self.checkpoint_pool)
            .finish()
    }
}

impl S14CausalBlockProductionBundle {
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    /// 为下一次 block 安装真实 terminal owner 与未完成 host candidate。调用方不接触 publisher；
    /// bundle 只会在同一 FullDepth43 seal 后消费并发布这份 one-shot producer。
    pub fn install_terminal_resources_for_next_block(
        &mut self,
        owner: Arc<S14CausalBlockProductionTerminalResourceOwner>,
        host_candidates: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    ) -> Result<(), String> {
        if !self.is_idle() || self.post_seal_terminal_producer.is_some() {
            return Err("production bundle 只能在 idle 安装一个 terminal owner".into());
        }
        if self.terminal_publisher.telemetry()?.pending {
            return Err("production bundle terminal provider 已有 pending source".into());
        }
        if !Arc::ptr_eq(&self.context, owner.context())
            || !Arc::ptr_eq(&self.paged_arena, owner.paged_weight_arena())
            || owner.block_size() != self.block_size
            || owner.checkpoint_state_bytes()
                != self.checkpoint_pool.layout().checkpoint_state_bytes
            || host_candidates.block_size() != self.block_size
            || host_candidates
                .base_position()
                .checked_add(self.block_size as u32)
                .is_none()
        {
            return Err(
                "production terminal owner/finalizer 与 bundle context/K/checkpoint ABI 漂移"
                    .into(),
            );
        }
        self.post_seal_terminal_producer =
            Some(Box::new(S14CausalBlockProductionOwnedTerminalProducer {
                owner,
                host_candidates,
            }));
        Ok(())
    }

    pub fn checkpoint_pool(&self) -> &Arc<S14CausalBlockCheckpointArenaPool> {
        &self.checkpoint_pool
    }

    pub fn checkpoint_telemetry(&self) -> Result<S14CausalBlockCheckpointArenaTelemetry> {
        self.checkpoint_pool.telemetry()
    }

    pub fn initial_hidden_binding(
        &self,
        bank: usize,
        generation: u64,
    ) -> Result<S14CausalBlockHiddenBinding> {
        self.hidden_banks
            .get(bank)
            .context("S14 production bundle initial hidden bank index 非法")?
            .binding(self.block_size, generation)
    }

    pub fn is_idle(&self) -> bool {
        self.phase
            .lock()
            .is_ok_and(|phase| *phase == BundlePhase::Idle)
    }

    /// 只在没有 active/sealed/unvalidated block 时执行。顺序固定为：关闭 publisher 并回滚
    /// pending source；销毁已 drain 的 MoE graph/timeline/recorder；销毁 HC/QKV owner；最后
    /// drop terminal recorder。checkpoint pool 可由已导出的 future lease 独立保活。
    pub fn destroy(&mut self) -> Result<(), String> {
        {
            let mut phase = lock_phase(&self.phase)?;
            if *phase == BundlePhase::Destroyed {
                return Ok(());
            }
            if *phase != BundlePhase::Idle {
                return Err("S14 production bundle 只能在 idle/已验收 future 状态销毁".into());
            }
            *phase = BundlePhase::Destroying;
        }

        if let Err(error) = self.terminal_publisher.inner.rollback_pending() {
            set_phase(&self.phase, BundlePhase::Poisoned);
            return Err(format!(
                "terminal provider pending source rollback 失败: {error}"
            ));
        }
        self.post_seal_terminal_producer.take();
        let backend = self
            .backend
            .as_mut()
            .ok_or_else(|| "S14 production backend 已销毁".to_owned())?;
        if let Err(error) = backend.destroy_moe_adapter() {
            set_phase(&self.phase, BundlePhase::Poisoned);
            return Err(format!("production bundle MoE destroy 失败: {error}"));
        }
        if let Err(error) = backend.destroy_hc_qkv_adapter() {
            set_phase(&self.phase, BundlePhase::Poisoned);
            return Err(format!("production bundle HC/QKV destroy 失败: {error}"));
        }
        drop(self.backend.take());
        set_phase(&self.phase, BundlePhase::Destroyed);
        Ok(())
    }

    fn backend_mut(&mut self) -> Result<&mut S14CausalBlockVulkanBackend, String> {
        self.backend
            .as_mut()
            .ok_or_else(|| "S14 production backend 已销毁".to_owned())
    }
}

pub fn build_s14_causal_block_production_bundle<P>(
    inputs: S14CausalBlockProductionBundleInputs<P>,
) -> Result<S14CausalBlockProductionBundle>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    let S14CausalBlockProductionBundleInputs {
        context,
        shape,
        hc_qkv_provider,
        hidden_banks,
        catalog,
        cache_root,
        fetch_mode,
        union_mapped_store,
        static_arena,
    } = inputs;
    if !context.timeline_semaphore {
        bail!("S14 production bundle 要求 Vulkan timeline semaphore");
    }
    validate_context_owner(&context, hc_qkv_provider.context(), "HC/QKV provider")?;
    validate_context_owner(&context, hidden_banks.context(), "hidden banks")?;
    validate_context_owner(&context, static_arena.context(), "paged static arena")?;

    if !Arc::ptr_eq(
        hc_qkv_provider.value().paged_weight_arena(),
        static_arena.value(),
    ) {
        bail!("S14 production bundle HC/QKV 与 grouped-MoE paged arena identity 漂移");
    }

    let (_, mut provider) = hc_qkv_provider.into_parts();
    provider
        .validate_production_bundle(shape.block_size)
        .map_err(anyhow::Error::msg)
        .context("HC/QKV provider builder readiness 拒绝")?;
    let (_, hidden_banks) = hidden_banks.into_parts();
    validate_hidden_banks(&hidden_banks)?;
    let (_, static_arena) = static_arena.into_parts();
    validate_static_arena(&static_arena)?;
    let bundle_paged_arena = Arc::clone(&static_arena);
    validate_full_depth_catalog(&catalog)?;

    // terminal recorder 先构造；其 Drop 会 queue-drain 并释放自身资源，便于后续任一 factory
    // 步骤失败时保持可回滚。pool 与 recorder 由同一 Arc context 直接创建。
    let checkpoint_pool = S14CausalBlockCheckpointArenaPool::new(
        Arc::clone(&context),
        shape.checkpoint_state_bytes,
        shape.checkpoint_slots,
    )?;
    let terminal_recorder = S14CausalBlockBatchedTerminalRecorder::new(
        Arc::clone(&context),
        Arc::clone(&checkpoint_pool),
    )?;

    let hc_static_arena = Arc::clone(&static_arena);
    let mut moe_adapter = build_s14_causal_block_paged_moe_adapter_with_shared_store(
        Arc::clone(&context),
        catalog,
        &cache_root,
        fetch_mode,
        static_arena,
        hidden_banks.clone(),
        union_mapped_store,
    )?;
    let prefix_state_producer = provider
        .take_prefix_state_producer()
        .map_err(anyhow::Error::msg)
        .context("消费同源K-prefix state producer")?;
    let hc_recorder = match S14CausalBlockProductionHcQkvLayerRecorder::new(
        Arc::clone(&context),
        hc_static_arena,
        provider,
        hidden_banks.clone(),
    ) {
        Ok(recorder) => recorder,
        Err(error) => {
            let mut prefix_state_producer = prefix_state_producer;
            let prefix_cleanup = prefix_state_producer.destroy();
            let cleanup = moe_adapter.destroy();
            return match cleanup {
                Ok(()) if prefix_cleanup.is_ok() => Err(error.context("构造 HC/QKV recorder")),
                Err(cleanup_error) => Err(anyhow!(
                    "构造 HC/QKV recorder 失败: {error:#}; MoE rollback 失败: {cleanup_error}; prefix rollback={prefix_cleanup:?}"
                )),
                Ok(()) => Err(anyhow!(
                    "构造 HC/QKV recorder 失败: {error:#}; prefix rollback={prefix_cleanup:?}"
                )),
            };
        }
    }
    .with_prefix_state_producer(prefix_state_producer)
    .context("安装同command K-prefix state producer")?;
    let hc_adapter = S14CausalBlockProductionHcQkvAdapter::new(hc_recorder);
    let (publisher, provider) = s14_causal_block_terminal_production_channel();
    let terminal_adapter =
        S14CausalBlockTerminalProductionAdapter::new(terminal_recorder, provider);

    let mut backend = S14CausalBlockVulkanBackend::with_moe_adapter(moe_adapter);
    backend
        .install_hc_qkv_adapter(hc_adapter)
        .map_err(anyhow::Error::msg)
        .context("安装 production HC/QKV adapter")?;
    if let Err(error) = backend.install_terminal_recorder(terminal_adapter) {
        let moe_cleanup = backend.destroy_moe_adapter();
        let hc_cleanup = backend.destroy_hc_qkv_adapter();
        bail!(
            "安装 production terminal adapter 失败: {error}; MoE cleanup={moe_cleanup:?}; HC cleanup={hc_cleanup:?}"
        );
    }

    let phase = Arc::new(Mutex::new(BundlePhase::Idle));
    let terminal_publisher = S14CausalBlockProductionTerminalPublisher {
        context: Arc::clone(&context),
        block_size: shape.block_size,
        phase: Arc::clone(&phase),
        inner: publisher,
    };
    Ok(S14CausalBlockProductionBundle {
        context,
        paged_arena: bundle_paged_arena,
        block_size: shape.block_size,
        backend: Some(backend),
        terminal_publisher,
        checkpoint_pool,
        hidden_banks,
        post_seal_terminal_producer: None,
        phase,
    })
}

impl S14CausalBlockLayerBackend for S14CausalBlockProductionBundle {
    fn run_k_lane_attention_router(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockAttentionRouterOutput, String> {
        validate_recording_input(&self.phase, self.block_size, input)?;
        self.backend_mut()?.run_k_lane_attention_router(input)
    }

    fn materialize_union_ranges(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockUnionMaterializeReceipt, String> {
        validate_recording_k(&self.phase, self.block_size, range_plan.block_size)?;
        self.backend_mut()?
            .materialize_union_ranges(bank, range_plan)
    }

    fn run_grouped_moe(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockGroupedMoeOutput, String> {
        validate_recording_k(&self.phase, self.block_size, routes.len())?;
        let output = self.backend_mut()?.run_grouped_moe(
            bank,
            post_attention_hidden,
            routes,
            batch_plan,
            range_plan,
        )?;
        let mut phase = lock_phase(&self.phase)?;
        let BundlePhase::Recording {
            completed_layers, ..
        } = &mut *phase
        else {
            return Err("grouped MoE 完成时 production bundle phase 漂移".into());
        };
        *completed_layers = completed_layers
            .checked_add(1)
            .ok_or_else(|| "production bundle completed layer counter overflow".to_owned())?;
        Ok(output)
    }
}

impl S14CausalBlockFullDepthBackend for S14CausalBlockProductionBundle {
    fn begin_full_depth_block(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> Result<S14CausalBlockBeginReceipt, String> {
        if block_size != self.block_size {
            return Err(format!(
                "production bundle 固定 K={}，拒绝 K={block_size}",
                self.block_size
            ));
        }
        let phase_owner = Arc::clone(&self.phase);
        let mut phase = lock_phase(&phase_owner)?;
        if *phase != BundlePhase::Idle {
            return Err("production bundle 已有未释放 block/future".into());
        }
        let producer = self.post_seal_terminal_producer.as_ref().ok_or_else(|| {
            "production bundle 开始前缺少真实 terminal owner/finalizer".to_owned()
        })?;
        if producer.block_size() != block_size || producer.base_position() != base_position {
            return Err("production terminal producer 与本次 block K/base position 漂移".into());
        }
        let receipt =
            self.backend_mut()?
                .begin_full_depth_block(bank, base_position, block_size)?;
        *phase = BundlePhase::Recording {
            base_position,
            completed_layers: 0,
        };
        Ok(receipt)
    }

    fn seal_full_depth_layers(
        &mut self,
        completed_layers: usize,
    ) -> Result<S14CausalBlockSealReceipt, String> {
        let phase_owner = Arc::clone(&self.phase);
        let mut phase = lock_phase(&phase_owner)?;
        let base_position = match *phase {
            BundlePhase::Recording {
                base_position,
                completed_layers: observed,
            } if observed == completed_layers && completed_layers == FULL_DEPTH_LAYERS.len() => {
                base_position
            }
            _ => return Err("production bundle 禁止 seal 不完整/phase漂移的 FullDepth43".into()),
        };
        let receipt = self
            .backend_mut()?
            .seal_full_depth_layers(completed_layers)?;
        *phase = BundlePhase::LayersSealed { base_position };
        Ok(receipt)
    }

    fn drain_and_abort_full_depth_block(
        &mut self,
        completed_layers: usize,
    ) -> Result<S14CausalBlockAbortReceipt, String> {
        let phase_owner = Arc::clone(&self.phase);
        let mut phase = lock_phase(&phase_owner)?;
        if matches!(
            *phase,
            BundlePhase::Idle | BundlePhase::Destroying | BundlePhase::Destroyed
        ) {
            return Err("production bundle 当前没有可 abort 的 block".into());
        }
        let result = self
            .backend_mut()?
            .drain_and_abort_full_depth_block(completed_layers);
        self.post_seal_terminal_producer.take();
        match result {
            Ok(receipt) => {
                *phase = BundlePhase::Idle;
                Ok(receipt)
            }
            Err(error) => {
                *phase = BundlePhase::Poisoned;
                Err(error)
            }
        }
    }
}

impl S14CausalBlockCheckpointBackend for S14CausalBlockProductionBundle {
    fn publish_terminal_source_after_full_depth_seal(
        &mut self,
        completed_layers: usize,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockPostSealReceipt, String> {
        {
            let phase = lock_phase(&self.phase)?;
            if *phase != (BundlePhase::LayersSealed { base_position }) {
                return Err("production post-seal hook 不在同一 LayersSealed block".into());
            }
        }
        if completed_layers != FULL_DEPTH_LAYERS.len()
            || final_hidden.block_size != self.block_size
            || routes_by_position.len() != self.block_size
            || routes_by_position
                .iter()
                .any(|routes| routes.len() != FULL_DEPTH_LAYERS.len())
        {
            return Err("production post-seal hook K/43层/routes identity 漂移".into());
        }
        let before = self.terminal_publisher.telemetry()?;
        if before.pending {
            return Err("production post-seal hook 开始前已有 pending terminal source".into());
        }
        let producer = self
            .post_seal_terminal_producer
            .take()
            .ok_or_else(|| "production bundle 缺少一次性 post-seal terminal producer".to_owned())?;
        producer.publish_terminal_source(
            &self.terminal_publisher,
            completed_layers,
            base_position,
            final_hidden,
            routes_by_position,
        )?;
        let after = self.terminal_publisher.telemetry()?;
        let expected_published = before
            .published
            .checked_add(1)
            .ok_or_else(|| "terminal publisher telemetry overflow".to_owned())?;
        if after.published != expected_published
            || after.rejected != before.rejected
            || after.take_attempts != before.take_attempts
            || !after.pending
            || !matches!(
                *lock_phase(&self.phase)?,
                BundlePhase::TerminalSourcePublished {
                    base_position: published_base
                } if published_base == base_position
            )
        {
            return Err(
                "production post-seal producer 未精确发布一份同 block terminal source".into(),
            );
        }
        Ok(S14CausalBlockPostSealReceipt {
            hook_calls: 1,
            completed_layers,
            base_position,
            block_size: self.block_size,
            published_terminal_sources: 1,
            serial_token_forward_calls: 0,
        })
    }

    fn run_batched_final_head_and_export_checkpoints(
        &mut self,
        completed_layers: usize,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockFinalOutput, String> {
        let phase_owner = Arc::clone(&self.phase);
        let mut phase = lock_phase(&phase_owner)?;
        if !matches!(*phase, BundlePhase::TerminalSourcePublished { .. })
            || final_hidden.block_size != self.block_size
            || routes_by_position.len() != self.block_size
        {
            return Err("production bundle terminal 输入与 fixed-K sealed block 漂移".into());
        }
        let output = self
            .backend_mut()?
            .run_batched_final_head_and_export_checkpoints(
                completed_layers,
                final_hidden,
                routes_by_position,
            )?;
        *phase = BundlePhase::AwaitingExportAck;
        Ok(output)
    }

    fn acknowledge_export_validated(&mut self) -> Result<(), String> {
        let phase_owner = Arc::clone(&self.phase);
        let mut phase = lock_phase(&phase_owner)?;
        if *phase != BundlePhase::AwaitingExportAck {
            return Err("production bundle 没有等待验收的 terminal export".into());
        }
        self.backend_mut()?.acknowledge_export_validated()?;
        *phase = BundlePhase::Idle;
        Ok(())
    }
}

fn validate_terminal_publish_receipt(
    receipt: S14CausalBlockTerminalPublishReceipt,
    completed_layers: usize,
    base_position: u32,
    block_size: usize,
) -> Result<(), String> {
    if receipt.completed_layers != completed_layers
        || completed_layers != FULL_DEPTH_LAYERS.len()
        || receipt.base_position != base_position
        || receipt.block_size != block_size
        || receipt.producer_timeline_value == 0
        || receipt.checkpoint_count != block_size
        || receipt.head_chunk_count != S14_HEAD_CHUNK_COUNT as usize
        || receipt.predicted_tokens_prebuilt
    {
        return Err(
            "production terminal publish receipt K/43层/resource identity 漂移或预造预测token"
                .into(),
        );
    }
    Ok(())
}

pub(crate) fn validate_context_owner(
    expected: &Arc<VulkanContext>,
    observed: &Arc<VulkanContext>,
    label: &str,
) -> Result<()> {
    if !Arc::ptr_eq(expected, observed) {
        bail!("S14 production bundle {label} 不属于同一 Arc<VulkanContext>");
    }
    Ok(())
}

pub(crate) fn validate_hidden_banks(
    banks: &[S14CausalBlockHiddenBank; HIDDEN_BANK_COUNT],
) -> Result<()> {
    let bindings = [banks[0].binding(8, 0)?, banks[1].binding(8, 0)?];
    if bindings
        .iter()
        .any(|binding| binding.buffer == vk::Buffer::null())
    {
        bail!("S14 production bundle hidden bank handle 为空");
    }
    let left_end = bindings[0]
        .offset
        .checked_add(bindings[0].bytes)
        .context("hidden bank A range overflow")?;
    let right_end = bindings[1]
        .offset
        .checked_add(bindings[1].bytes)
        .context("hidden bank B range overflow")?;
    if bindings[0].buffer == bindings[1].buffer
        && bindings[0].offset < right_end
        && bindings[1].offset < left_end
    {
        bail!("S14 production bundle hidden A/B banks 重叠");
    }
    Ok(())
}

pub(crate) fn validate_static_arena(arena: &S14Position0PagedWeightArena) -> Result<()> {
    let plan = arena.plan();
    let layout = &plan.physical;
    let resident_layers = arena.resident_static_layers();
    let streamed_layers = arena.streamed_static_layers();
    let mut resident_requested_bytes = 0u64;
    let mut streamed_requested_bytes = 0u64;

    validate_paged_plan_ledger(
        plan,
        resident_layers,
        streamed_layers,
        arena.recurring_static_upload_bytes(),
    )?;
    arena
        .workspace_layout()
        .validate()
        .context("S14 production bundle paged workspace layout")?;
    if arena.workspace().handle() == vk::Buffer::null()
        || arena.workspace().size() != plan.workspace_bytes
        || arena.workspace_layout().capacity_bytes() != plan.workspace_bytes
        || arena.resident_small().handle() == vk::Buffer::null()
        || arena.resident_small().size() != layout.resident_small.requested_bytes
    {
        bail!("S14 production bundle paged workspace/resident-small arena 不完整");
    }

    for (index, (&layer, placement)) in FULL_DEPTH_LAYERS
        .iter()
        .zip(&layout.static_layers)
        .enumerate()
    {
        let (buffer, binding_layout, resident) = match arena.static_layer(layer)? {
            S14Position0StaticLayerBinding::Resident { buffer, layout } => {
                resident_requested_bytes = resident_requested_bytes
                    .checked_add(layout.requested_bytes)
                    .context("paged resident static bytes overflow")?;
                (buffer, layout, true)
            }
            S14Position0StaticLayerBinding::Streamed {
                bank,
                buffer,
                layout,
            } => {
                if bank != index % S14_POSITION0_STATIC_STREAM_BANKS
                    || buffer.handle() != arena.static_stream_bank(bank)?.handle()
                {
                    bail!("S14 production bundle paged static L{layer} stream bank identity 漂移");
                }
                streamed_requested_bytes = streamed_requested_bytes
                    .checked_add(layout.requested_bytes)
                    .context("paged streamed static bytes overflow")?;
                (buffer, layout, false)
            }
        };
        if placement.layer != layer
            || binding_layout.layer != layer
            || binding_layout != placement
            || placement.requested_bytes == 0
            || buffer.handle() == vk::Buffer::null()
            || buffer.size() < placement.requested_bytes
        {
            bail!("S14 production bundle paged static L{layer} identity/capacity 漂移");
        }
        if resident != (index < resident_layers) {
            bail!("S14 production bundle paged static resident prefix 漂移");
        }
    }
    if resident_requested_bytes
        .checked_add(streamed_requested_bytes)
        .context("paged static requested bytes overflow")?
        != layout.static_requested_bytes
        || resident_requested_bytes != arena.allocated_static_resident_bytes()
        || streamed_requested_bytes != arena.recurring_static_upload_bytes()
        || arena.allocated_essential_bytes() != plan.essential_device_bytes
        || arena.allocated_device_bytes()
            != plan
                .essential_device_bytes
                .checked_add(resident_requested_bytes)
                .context("paged allocated device bytes overflow")?
    {
        bail!("S14 production bundle paged static/essential/allocated byte ledger 漂移");
    }

    let mut static_handles = Vec::with_capacity(S14_POSITION0_STATIC_STREAM_BANKS);
    for bank in 0..S14_POSITION0_STATIC_STREAM_BANKS {
        let buffer = arena.static_stream_bank(bank)?;
        if buffer.handle() == vk::Buffer::null() || buffer.size() != plan.static_stream_bank_bytes {
            bail!("S14 production bundle paged static stream bank {bank} 不完整");
        }
        static_handles.push(buffer.handle());
    }
    if static_handles[0] == static_handles[1] {
        bail!("S14 production bundle paged static stream 双 bank 发生别名");
    }

    let mut routed_handles = Vec::with_capacity(S14_POSITION0_ROLLING_BANKS);
    let mut head_handles = Vec::with_capacity(S14_POSITION0_ROLLING_BANKS);
    for bank in 0..S14_POSITION0_ROLLING_BANKS {
        let routed = arena.routed(bank)?;
        let head = arena.head_chunk(bank)?;
        if routed.handle() == vk::Buffer::null()
            || routed.size() != layout.routed_bank_bytes
            || head.handle() == vk::Buffer::null()
            || head.size() != layout.head_chunk_bytes
        {
            bail!("S14 production bundle paged routed/head rolling bank {bank} 不完整");
        }
        routed_handles.push(routed.handle());
        head_handles.push(head.handle());
    }
    if routed_handles[0] == routed_handles[1] || head_handles[0] == head_handles[1] {
        bail!("S14 production bundle paged routed/head 双 bank 发生别名");
    }
    Ok(())
}

fn validate_paged_plan_ledger(
    plan: &S14Position0PagedArenaPlan,
    resident_layers: usize,
    streamed_layers: usize,
    recurring_static_upload_bytes: u64,
) -> Result<()> {
    let physical = &plan.physical;
    let static_stream_bytes = plan
        .static_stream_bank_bytes
        .checked_mul(S14_POSITION0_STATIC_STREAM_BANKS as u64)
        .context("paged static stream bytes overflow")?;
    let routed_bytes = physical
        .routed_bank_bytes
        .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
        .context("paged routed bytes overflow")?;
    let head_bytes = physical
        .head_chunk_bytes
        .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
        .context("paged head bytes overflow")?;
    let expected_essential = plan
        .workspace_bytes
        .checked_add(physical.resident_small.requested_bytes)
        .and_then(|bytes| bytes.checked_add(routed_bytes))
        .and_then(|bytes| bytes.checked_add(head_bytes))
        .and_then(|bytes| bytes.checked_add(static_stream_bytes))
        .context("paged essential bytes overflow")?;
    let expected_recurring = physical
        .static_layers
        .iter()
        .skip(resident_layers)
        .try_fold(0u64, |sum, layer| {
            sum.checked_add(layer.requested_bytes)
                .context("paged recurring static bytes overflow")
        })?;
    if physical.static_layers.len() != FULL_DEPTH_LAYERS.len()
        || resident_layers + streamed_layers != FULL_DEPTH_LAYERS.len()
        || resident_layers > FULL_DEPTH_LAYERS.len()
        || plan.workspace_bytes == 0
        || plan.static_stream_bank_bytes == 0
        || plan.static_stream_device_bytes != static_stream_bytes
        || plan.essential_device_bytes != expected_essential
        || recurring_static_upload_bytes != expected_recurring
    {
        bail!("S14 production bundle paged arena plan/essential/recurring ledger 不完整");
    }
    Ok(())
}

pub(crate) fn validate_full_depth_catalog(catalog: &FullDepthExpertCatalog) -> Result<()> {
    let expert_count = usize::from(N_ROUTED_EXPERTS);
    for layer in FULL_DEPTH_LAYERS {
        let mut start = 0usize;
        while start < expert_count {
            let first = if start + EXPERTS_PER_TOKEN <= expert_count {
                start
            } else {
                expert_count - EXPERTS_PER_TOKEN
            };
            let mut expert_ids = [0u16; EXPERTS_PER_TOKEN];
            for (slot, expert) in expert_ids.iter_mut().enumerate() {
                *expert = u16::try_from(first + slot).context("catalog expert ID overflow")?;
            }
            catalog
                .plan(OnlineTop6 {
                    layer,
                    position: 1,
                    expert_ids,
                    route_weights: [1.0 / EXPERTS_PER_TOKEN as f32; EXPERTS_PER_TOKEN],
                })
                .with_context(|| {
                    format!(
                        "S14 production bundle catalog L{layer} experts {first}..{} 拒绝",
                        first + EXPERTS_PER_TOKEN - 1
                    )
                })?;
            start = start.saturating_add(EXPERTS_PER_TOKEN);
        }
    }
    Ok(())
}

fn validate_recording_input(
    phase: &SharedBundlePhase,
    block_size: usize,
    input: &S14CausalBlockLayerInput<'_>,
) -> Result<(), String> {
    let phase = lock_phase(phase)?;
    match *phase {
        BundlePhase::Recording { base_position, .. }
            if input.base_position == base_position
                && input.input_token_ids.len() == block_size
                && input.input_hidden.block_size == block_size =>
        {
            Ok(())
        }
        _ => Err("production bundle HC/QKV input 与 fixed-K active block 漂移".into()),
    }
}

fn validate_recording_k(
    phase: &SharedBundlePhase,
    block_size: usize,
    observed: usize,
) -> Result<(), String> {
    if observed != block_size || !matches!(*lock_phase(phase)?, BundlePhase::Recording { .. }) {
        return Err("production bundle layer input 与 fixed-K recording phase 漂移".into());
    }
    Ok(())
}

fn lock_phase(phase: &SharedBundlePhase) -> Result<MutexGuard<'_, BundlePhase>, String> {
    phase
        .lock()
        .map_err(|_| "S14 production bundle lifecycle poisoned".to_owned())
}

fn set_phase(phase: &SharedBundlePhase, next: BundlePhase) {
    if let Ok(mut phase) = phase.lock() {
        *phase = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_rejects_non_production_k_before_resource_construction() {
        let error = S14CausalBlockProductionBundleShape::new(6, 1, 1).unwrap_err();
        assert!(error.to_string().contains("K 只允许4或8"));
    }

    #[test]
    fn post_seal_terminal_receipt_requires_one_real_k4_publication() {
        let valid = S14CausalBlockTerminalPublishReceipt {
            base_position: 1,
            block_size: 4,
            completed_layers: FULL_DEPTH_LAYERS.len(),
            producer_timeline_value: 1,
            normalized_head_rows_offset: 65_536,
            checkpoint_count: 4,
            head_chunk_count: S14_HEAD_CHUNK_COUNT as usize,
            predicted_tokens_prebuilt: false,
        };
        validate_terminal_publish_receipt(valid, FULL_DEPTH_LAYERS.len(), 1, 4).unwrap();

        let mut invalid = valid;
        invalid.predicted_tokens_prebuilt = true;
        assert!(
            validate_terminal_publish_receipt(invalid, FULL_DEPTH_LAYERS.len(), 1, 4,).is_err()
        );
    }

    #[test]
    fn paged_plan_ledger_accepts_full_depth_resident_plus_streamed_split() {
        use crate::s14_position0_weight_plan::S14Position0HybridWeightPlan;
        use polaris_s14_runner::Position0WholeTokenManifest;

        let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        let manifest = Position0WholeTokenManifest::load(&manifest_path).unwrap();
        let weights = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        let plan = S14Position0PagedArenaPlan::build(&weights).unwrap();
        let resident_layers = 32;
        let streamed_layers = FULL_DEPTH_LAYERS.len() - resident_layers;
        let recurring = plan
            .physical
            .static_layers
            .iter()
            .skip(resident_layers)
            .map(|layer| layer.requested_bytes)
            .sum();
        validate_paged_plan_ledger(&plan, resident_layers, streamed_layers, recurring).unwrap();

        assert!(
            validate_paged_plan_ledger(&plan, resident_layers, streamed_layers, recurring - 1,)
                .is_err()
        );
    }
}
