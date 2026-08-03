//! S14 causal-block 三段 concrete Vulkan 数据面的 production 组合根。
//!
//! 本模块只负责 owner 接线与生命周期，不创建、填充或替代任何模型权重。所有静态
//! arena、HC/QKV layer provider、hidden banks 与 terminal production source 都必须显式
//! 绑定到同一个 `Arc<VulkanContext>`。factory 在首次 Vulkan allocation 前校验 K、
//! FullDepth catalog、arena ledger 与 provider readiness；任一缺项均 fail-closed。

use crate::{
    s14_causal_block_grouped_moe_recorder::build_s14_causal_block_concrete_moe_adapter,
    s14_causal_block_hc_qkv_adapter::{
        S14CausalBlockProductionHcQkvAdapter, S14CausalBlockVulkanHcQkvAdapter,
    },
    s14_causal_block_hc_qkv_recorder::{
        S14CausalBlockHcQkvResourceProvider, S14CausalBlockHiddenBank,
        S14CausalBlockProductionHcQkvLayerRecorder,
    },
    s14_causal_block_layer::{
        S14CausalBlockAbortReceipt, S14CausalBlockAttentionRouterOutput,
        S14CausalBlockBeginReceipt, S14CausalBlockCheckpointBackend,
        S14CausalBlockFinalOutput, S14CausalBlockFullDepthBackend,
        S14CausalBlockGroupedMoeOutput, S14CausalBlockHiddenBinding,
        S14CausalBlockLayerBackend, S14CausalBlockLayerInput, S14CausalBlockLayerRangePlan,
        S14CausalBlockSealReceipt, S14CausalBlockUnionBankBinding,
        S14CausalBlockUnionMaterializeReceipt,
    },
    s14_causal_block_moe_adapter::S14CausalBlockVulkanMoeAdapter,
    s14_causal_block_terminal::{
        S14CausalBlockBatchedTerminalRecorder, S14CausalBlockCheckpointArenaPool,
        S14CausalBlockCheckpointArenaTelemetry,
    },
    s14_causal_block_terminal_adapter::{
        s14_causal_block_terminal_production_channel,
        S14CausalBlockTerminalProductionAdapter, S14CausalBlockTerminalProductionPublisher,
        S14CausalBlockTerminalProductionSource, S14CausalBlockTerminalProviderTelemetry,
    },
    s14_causal_block_vulkan_backend::S14CausalBlockVulkanBackend,
    s14_dynamic_page_cache_readiness::DynamicPageFetchMode,
    s14_dynamic_routed_page_plan::{FullDepthExpertCatalog, OnlineTop6},
    s14_position0_hybrid_weight_arena::{
        S14Position0HybridWeightArena, S14_POSITION0_HYBRID_ALLOCATION_COUNT,
    },
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    LayerCausalBatchPlan, RouteDecision, EXPERTS_PER_TOKEN, FULL_DEPTH_LAYERS,
    N_ROUTED_EXPERTS,
};
use std::{
    fmt,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

const HIDDEN_BANK_COUNT: usize = 2;
const HYBRID_ROLLING_BANKS: usize = 2;

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

    fn into_parts(self) -> (Arc<VulkanContext>, T) {
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
    fn validate_production_bundle(&self, block_size: usize) -> Result<(), String>;
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
    pub static_arena: S14CausalBlockContextBound<Arc<S14Position0HybridWeightArena>>,
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
    AwaitingExportAck,
    Destroying,
    Poisoned,
    Destroyed,
}

type SharedBundlePhase = Arc<Mutex<BundlePhase>>;

/// 只允许在同 bundle 的 FullDepth43 seal 后发布一次同 identity terminal source。所有 clone
/// 共享 lifecycle；bundle 开始销毁后，旧 publisher 无法向已经失去 consumer 的 channel 发布。
#[derive(Clone)]
pub struct S14CausalBlockProductionTerminalPublisher {
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
    pub fn publish(
        &self,
        source: S14CausalBlockContextBound<S14CausalBlockTerminalProductionSource>,
    ) -> Result<(), String> {
        let (source_context, source) = source.into_parts();
        if !Arc::ptr_eq(&self.context, &source_context) {
            return Err("terminal production source 与 bundle VulkanContext owner 漂移".into());
        }
        let phase = lock_phase(&self.phase)?;
        let base_position = match *phase {
            BundlePhase::LayersSealed { base_position } => base_position,
            _ => return Err("terminal production source 只能在同一 FullDepth43 seal 后发布".into()),
        };
        if source.base_position != base_position
            || source.final_hidden.block_size != self.block_size
            || source.completed_layers != FULL_DEPTH_LAYERS.len()
        {
            return Err("terminal production source 与 bundle base/K/FullDepth43 identity 漂移".into());
        }
        self.inner.publish(source)
    }

    pub fn telemetry(&self) -> Result<S14CausalBlockTerminalProviderTelemetry, String> {
        self.inner.telemetry()
    }
}

/// 完整 production backend 与其 terminal producer channel/checkpoint pool 的唯一 owner。
#[must_use = "production bundle 必须由 orchestrator 持有，并在 VulkanContext 前显式 destroy"]
pub struct S14CausalBlockProductionBundle {
    context: Arc<VulkanContext>,
    block_size: usize,
    backend: Option<S14CausalBlockVulkanBackend>,
    terminal_publisher: S14CausalBlockProductionTerminalPublisher,
    checkpoint_pool: Arc<S14CausalBlockCheckpointArenaPool>,
    hidden_banks: [S14CausalBlockHiddenBank; HIDDEN_BANK_COUNT],
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

    pub fn terminal_publisher(&self) -> S14CausalBlockProductionTerminalPublisher {
        self.terminal_publisher.clone()
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
            return Err(format!("terminal provider pending source rollback 失败: {error}"));
        }
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
        static_arena,
    } = inputs;
    if !context.timeline_semaphore {
        bail!("S14 production bundle 要求 Vulkan timeline semaphore");
    }
    validate_context_owner(&context, hc_qkv_provider.context(), "HC/QKV provider")?;
    validate_context_owner(&context, hidden_banks.context(), "hidden banks")?;
    validate_context_owner(&context, static_arena.context(), "hybrid static arena")?;

    let (_, provider) = hc_qkv_provider.into_parts();
    provider
        .validate_production_bundle(shape.block_size)
        .map_err(anyhow::Error::msg)
        .context("HC/QKV provider builder readiness 拒绝")?;
    let (_, hidden_banks) = hidden_banks.into_parts();
    validate_hidden_banks(&hidden_banks)?;
    let (_, static_arena) = static_arena.into_parts();
    validate_static_arena(&static_arena)?;
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

    let mut moe_adapter = build_s14_causal_block_concrete_moe_adapter(
        Arc::clone(&context),
        catalog,
        &cache_root,
        fetch_mode,
        static_arena,
    )?;
    let hc_recorder = match S14CausalBlockProductionHcQkvLayerRecorder::new(
        Arc::clone(&context),
        provider,
        hidden_banks.clone(),
    ) {
        Ok(recorder) => recorder,
        Err(error) => {
            let cleanup = moe_adapter.destroy();
            return match cleanup {
                Ok(()) => Err(error.context("构造 HC/QKV recorder")),
                Err(cleanup_error) => Err(anyhow!(
                    "构造 HC/QKV recorder 失败: {error:#}; MoE rollback 失败: {cleanup_error}"
                )),
            };
        }
    };
    let hc_adapter = S14CausalBlockProductionHcQkvAdapter::new(hc_recorder);
    let (publisher, provider) = s14_causal_block_terminal_production_channel();
    let terminal_adapter = S14CausalBlockTerminalProductionAdapter::new(terminal_recorder, provider);

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
        block_size: shape.block_size,
        backend: Some(backend),
        terminal_publisher,
        checkpoint_pool,
        hidden_banks,
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
        self.backend_mut()?.materialize_union_ranges(bank, range_plan)
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
        let receipt = self.backend_mut()?.begin_full_depth_block(
            bank,
            base_position,
            block_size,
        )?;
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
        if matches!(*phase, BundlePhase::Idle | BundlePhase::Destroying | BundlePhase::Destroyed) {
            return Err("production bundle 当前没有可 abort 的 block".into());
        }
        match self
            .backend_mut()?
            .drain_and_abort_full_depth_block(completed_layers)
        {
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
    fn run_batched_final_head_and_export_checkpoints(
        &mut self,
        completed_layers: usize,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockFinalOutput, String> {
        let phase_owner = Arc::clone(&self.phase);
        let mut phase = lock_phase(&phase_owner)?;
        if !matches!(*phase, BundlePhase::LayersSealed { .. })
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

fn validate_context_owner(
    expected: &Arc<VulkanContext>,
    observed: &Arc<VulkanContext>,
    label: &str,
) -> Result<()> {
    if !Arc::ptr_eq(expected, observed) {
        bail!("S14 production bundle {label} 不属于同一 Arc<VulkanContext>");
    }
    Ok(())
}

fn validate_hidden_banks(
    banks: &[S14CausalBlockHiddenBank; HIDDEN_BANK_COUNT],
) -> Result<()> {
    let bindings = [banks[0].binding(8, 0)?, banks[1].binding(8, 0)?];
    if bindings.iter().any(|binding| binding.buffer == vk::Buffer::null()) {
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

fn validate_static_arena(arena: &S14Position0HybridWeightArena) -> Result<()> {
    let layout = arena.layout();
    if arena.allocation_count() != S14_POSITION0_HYBRID_ALLOCATION_COUNT
        || layout.static_layers.len() != FULL_DEPTH_LAYERS.len()
        || arena.requested_device_bytes() != layout.requested_device_bytes
        || arena.allocated_device_bytes() < arena.requested_device_bytes()
    {
        bail!("S14 production bundle hybrid arena allocation/ledger 不完整");
    }
    for (&layer, placement) in FULL_DEPTH_LAYERS.iter().zip(&layout.static_layers) {
        let buffer = arena.static_layer(layer)?;
        if placement.layer != layer
            || placement.requested_bytes == 0
            || buffer.handle() == vk::Buffer::null()
            || buffer.size() < placement.requested_bytes
        {
            bail!("S14 production bundle hybrid static layer L{layer} identity/capacity 漂移");
        }
    }
    if layout.resident_small.requested_bytes == 0
        || arena.resident_small().handle() == vk::Buffer::null()
        || arena.resident_small().size() < layout.resident_small.requested_bytes
    {
        bail!("S14 production bundle hybrid resident-small arena 不完整");
    }
    for bank in 0..HYBRID_ROLLING_BANKS {
        let routed = arena.routed(bank)?;
        let head = arena.head_chunk(bank)?;
        if routed.handle() == vk::Buffer::null()
            || routed.size() < layout.routed_bank_bytes
            || head.handle() == vk::Buffer::null()
            || head.size() < layout.head_chunk_bytes
        {
            bail!("S14 production bundle hybrid rolling arena bank {bank} 不完整");
        }
    }
    Ok(())
}

fn validate_full_depth_catalog(catalog: &FullDepthExpertCatalog) -> Result<()> {
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
                    format!("S14 production bundle catalog L{layer} experts {first}..{} 拒绝", first + EXPERTS_PER_TOKEN - 1)
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
}
