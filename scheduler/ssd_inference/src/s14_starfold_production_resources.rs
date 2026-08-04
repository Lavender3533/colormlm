//! S14Runtime 同源 StarFold K4 production resource builder。
//!
//! builder 只接受 `S14Runtime::into_starfold_owned_parts` 导出的唯一 runtime/windows，
//! 不创建第二套 microtile window，也不构造旧 grouped-MoE/union bank。HC/QKV provider、
//! hidden banks、paged arena 与 StarFold runtime 必须绑定同一 VulkanContext。

use crate::{
    s14_causal_block_hc_qkv_adapter::{
        S14CausalBlockProductionHcQkvAdapter, S14Position0CommittedGenerationProvenance,
    },
    s14_causal_block_hc_qkv_recorder::{
        S14CausalBlockHiddenBank, S14CausalBlockProductionHcQkvLayerRecorder,
    },
    s14_causal_block_layer::S14CausalBlockHiddenBinding,
    s14_causal_block_production_bundle::{
        validate_context_owner, validate_full_depth_catalog, validate_hidden_banks,
        validate_static_arena, S14CausalBlockContextBound,
        S14CausalBlockProductionHcQkvResourceProvider,
    },
    s14_dynamic_routed_page_plan::FullDepthExpertCatalog,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_runtime::S14Runtime,
    s14_starfold_k4_adapter::{
        S14StarfoldConcreteK4Stage, S14StarfoldK4FullDepthReceipt, S14StarfoldK4ProductionAdapter,
    },
    s14_starfold_k4_terminal_chain::{S14StarfoldK4CommitReceipt, S14StarfoldK4TerminalChainOwner},
    s14_starfold_prompt_prefill::{
        S14StarfoldSealedTeacherForcedPrefillProduct, S14StarfoldTeacherForcedBlockPlan,
    },
    s14_starfold_runtime::S14StarfoldRuntime,
    s14_starfold_terminal_endpoint::S14StarfoldTerminalEndpoint,
    s14_starwave_draft::S14StarwaveDraftProposal,
    s14_whole_token_device::WholeTokenDeviceState,
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{DecoderStateV1, MaterializedTokenSource};
use std::{fmt, sync::Arc};

const K4: usize = 4;

pub type S14StarfoldConcreteHcQkvAdapter<P> =
    S14CausalBlockProductionHcQkvAdapter<S14CausalBlockProductionHcQkvLayerRecorder<P>>;

pub type S14StarfoldConcreteProductionAdapter<P> =
    S14StarfoldK4ProductionAdapter<S14StarfoldConcreteK4Stage<S14StarfoldConcreteHcQkvAdapter<P>>>;

/// 只由当前真实 `Option` owner 与 StarFold runtime 物理窗口合同计数生成。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarfoldPersistentResourceInventory {
    pub base_runtime_owners: usize,
    pub starfold_runtime_owners: usize,
    pub paged_arena_owners: usize,
    pub expert_catalog_owners: usize,
    pub microtile_window_owners: usize,
    pub b4_owners: usize,
    pub hc_bridge_stage_owners: usize,
    pub direct_terminal_endpoint_owners: usize,
}

/// `S14Runtime` 一次性导出的同源物理 owner。`runtime` 私有且只能被 builder 消费；
/// 如果调用方在构建前丢弃 parts，Drop 会销毁唯一 StarFold runtime/windows。
pub struct S14StarfoldOwnedRuntimeParts {
    context: Arc<VulkanContext>,
    paged_arena: Option<Arc<S14Position0PagedWeightArena>>,
    expert_catalog: Option<Arc<FullDepthExpertCatalog>>,
    runtime: Option<S14StarfoldRuntime>,
    base_runtime: Option<S14Runtime>,
}

impl fmt::Debug for S14StarfoldOwnedRuntimeParts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14StarfoldOwnedRuntimeParts")
            .field("context", &Arc::as_ptr(&self.context))
            .field("paged_arena", &self.paged_arena.as_ref().map(Arc::as_ptr))
            .field("runtime_present", &self.runtime.is_some())
            .finish_non_exhaustive()
    }
}

impl S14StarfoldOwnedRuntimeParts {
    pub(crate) fn new(
        context: Arc<VulkanContext>,
        paged_arena: Arc<S14Position0PagedWeightArena>,
        expert_catalog: Arc<FullDepthExpertCatalog>,
        runtime: S14StarfoldRuntime,
        base_runtime: S14Runtime,
    ) -> Self {
        Self {
            context,
            paged_arena: Some(paged_arena),
            expert_catalog: Some(expert_catalog),
            runtime: Some(runtime),
            base_runtime: Some(base_runtime),
        }
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn paged_arena(&self) -> &Arc<S14Position0PagedWeightArena> {
        self.paged_arena
            .as_ref()
            .expect("S14 StarFold owned paged arena 已被消费")
    }

    pub fn expert_catalog(&self) -> &Arc<FullDepthExpertCatalog> {
        self.expert_catalog
            .as_ref()
            .expect("S14 StarFold owned catalog 已被消费")
    }

    pub fn build_k4<P>(
        self,
        inputs: S14StarfoldK4ProductionResourceInputs<P>,
    ) -> Result<S14StarfoldK4ProductionResources<P>>
    where
        P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
    {
        build_s14_starfold_k4_production_resources(self, inputs)
    }

    /// 首块 teacher-forced prefill 可为 K4/K8；generation 仍由调用方和 provenance 合同
    /// 锁定为 K4。旧 `build_k4` 保留为兼容窄入口。
    pub fn build_kblock<P>(
        self,
        inputs: S14StarfoldK4ProductionResourceInputs<P>,
        physical_block_size: usize,
    ) -> Result<S14StarfoldK4ProductionResources<P>>
    where
        P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
    {
        build_s14_starfold_kblock_production_resources(self, inputs, physical_block_size)
    }
}

impl Drop for S14StarfoldOwnedRuntimeParts {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            let _ = runtime.destroy();
        }
        self.paged_arena.take();
        self.expert_catalog.take();
        if let Some(runtime) = self.base_runtime.take() {
            let _ = runtime.destroy();
        }
    }
}

/// Runtime 外部已经按真实 committed state 构造好的 HC/QKV provider 与 hidden banks。
/// context-bound wrapper 防止把另一个 device 的 Vulkan handles 塞入同源 runtime。
pub struct S14StarfoldK4ProductionResourceInputs<P> {
    pub hc_qkv_provider: S14CausalBlockContextBound<P>,
    pub hidden_banks: S14CausalBlockContextBound<[S14CausalBlockHiddenBank; 2]>,
    /// 只允许首个 base0 SpeculativeGeneration block；其它 block 必须为 None。
    pub position0_generation_provenance: Option<S14Position0CommittedGenerationProvenance>,
}

/// FullDepth43/K4 的唯一资源组合根。begin/execute 都直接委托同一个 persistent adapter；
/// owner 内不存在旧 union bank 或第二套 StarFold runtime。
pub struct S14StarfoldK4ProductionResources<P>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    adapter: Option<S14StarfoldConcreteProductionAdapter<P>>,
    terminal_endpoint: Option<S14StarfoldTerminalEndpoint>,
    base_runtime: Option<S14Runtime>,
    context: Arc<VulkanContext>,
    // 必须能在销毁 base runtime 前显式释放这份共享引用；否则
    // `S14Runtime::destroy` 无法 `Arc::try_unwrap` 它内部的权威 arena。
    paged_arena: Option<Arc<S14Position0PagedWeightArena>>,
}

impl<P> fmt::Debug for S14StarfoldK4ProductionResources<P>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14StarfoldK4ProductionResources")
            .field("adapter_present", &self.adapter.is_some())
            .field(
                "terminal_endpoint_present",
                &self.terminal_endpoint.is_some(),
            )
            .field("base_runtime_present", &self.base_runtime.is_some())
            .field("context", &Arc::as_ptr(&self.context))
            .field("paged_arena", &self.paged_arena.as_ref().map(Arc::as_ptr))
            .finish()
    }
}

impl<P> S14StarfoldK4ProductionResources<P>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    pub fn begin_k4_block(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
        source: MaterializedTokenSource,
    ) -> Result<()> {
        self.adapter_mut()?.begin_block_with_source(
            base_position,
            input_token_ids,
            initial_hidden,
            source,
        )
    }

    pub fn execute_k4_full_depth(
        &mut self,
        base_position: u32,
        input_token_ids: &[u32],
        initial_hidden: S14CausalBlockHiddenBinding,
        source: MaterializedTokenSource,
    ) -> Result<S14StarfoldK4FullDepthReceipt> {
        self.adapter_mut()?.execute_full_depth_with_source(
            base_position,
            input_token_ids,
            initial_hidden,
            source,
        )
    }

    /// 同一 adapter 内连续完成 ForcedPrefill FullDepth seal、prefix arena 导出与
    /// finish/drain，返回不可手工拼装的强 product。
    pub fn execute_teacher_forced_prefill_product(
        &mut self,
        authoritative: &DecoderStateV1,
        block: &S14StarfoldTeacherForcedBlockPlan,
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<S14StarfoldSealedTeacherForcedPrefillProduct> {
        let full_depth = self
            .adapter_mut()?
            .execute_teacher_forced_prefill_full_depth(
                authoritative,
                block.physical_input_token_ids(),
                initial_hidden,
            )?;
        self.finish_teacher_forced_prefill_product(full_depth, block)
    }

    pub(crate) fn finish_teacher_forced_prefill_product(
        &mut self,
        full_depth: S14StarfoldK4FullDepthReceipt,
        block: &S14StarfoldTeacherForcedBlockPlan,
    ) -> Result<S14StarfoldSealedTeacherForcedPrefillProduct> {
        let prefix_product = self
            .adapter
            .as_ref()
            .context("S14 StarFold K4 production adapter 已销毁")?
            .starfold_teacher_forced_prefill_prefix_product()
            .context("导出同 adapter 已 drain ForcedPrefill prefix product")?;
        self.adapter_mut()?.finish_validated_block()?;
        S14StarfoldSealedTeacherForcedPrefillProduct::from_finished_adapter(
            full_depth,
            block.clone(),
            prefix_product,
        )
    }

    pub fn execute_generation_proposal(
        &mut self,
        authoritative: &DecoderStateV1,
        proposal: &S14StarwaveDraftProposal,
        initial_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<S14StarfoldK4FullDepthReceipt> {
        proposal
            .validate_for(authoritative)
            .map_err(|error| anyhow!(error.to_string()))
            .context("校验 StarWave generation proposal")?;
        if authoritative.position == 0 {
            self.adapter_mut()?.execute_generation_from_position0(
                authoritative,
                proposal,
                initial_hidden,
            )
        } else {
            self.execute_k4_full_depth(
                authoritative.position,
                proposal.input_token_ids(),
                initial_hidden,
                proposal.source(),
            )
        }
    }

    pub fn adapter_mut(&mut self) -> Result<&mut S14StarfoldConcreteProductionAdapter<P>> {
        self.adapter
            .as_mut()
            .context("S14 StarFold K4 production resources 已销毁")
    }

    /// 已 seal 的 FullDepth43 block 的 production 级 direct terminal + host/device 原子提交。
    /// owner 全部从本 root 的同一个 adapter/endpoint 导出，调用方不能注入另一套 terminal
    /// source、paged arena 或 checkpoint uploader。
    pub fn execute_and_commit_sealed_k4(
        &mut self,
        chain: &mut S14StarfoldK4TerminalChainOwner,
        full_depth: S14StarfoldK4FullDepthReceipt,
        draft_token_ids: &[u32; K4],
        authoritative: &mut DecoderStateV1,
        device_state: &mut WholeTokenDeviceState,
    ) -> Result<S14StarfoldK4CommitReceipt> {
        self.execute_and_commit_sealed_k4_with_commit_limit(
            chain,
            full_depth,
            draft_token_ids,
            K4,
            authoritative,
            device_state,
        )
    }

    /// 与默认入口相同的 direct terminal + 两阶段 commit，但显式限制实际发布的 prefix。
    pub fn execute_and_commit_sealed_k4_with_commit_limit(
        &mut self,
        chain: &mut S14StarfoldK4TerminalChainOwner,
        full_depth: S14StarfoldK4FullDepthReceipt,
        draft_token_ids: &[u32; K4],
        commit_limit: usize,
        authoritative: &mut DecoderStateV1,
        device_state: &mut WholeTokenDeviceState,
    ) -> Result<S14StarfoldK4CommitReceipt> {
        if full_depth.source != MaterializedTokenSource::SpeculativeDraft {
            bail!(
                "generation terminal 只接受 SpeculativeDraft FullDepth43，actual={:?}",
                full_depth.source
            );
        }
        let terminal_owners = self
            .adapter
            .as_ref()
            .context("S14 StarFold K4 production adapter 已销毁")?
            .starfold_terminal_block_owners(full_depth.final_hidden)
            .context("获取 S14 StarFold 同源 terminal block owners")?;
        let adapter = self
            .adapter
            .as_mut()
            .context("S14 StarFold K4 production adapter 已销毁")?;
        let terminal = self
            .terminal_endpoint
            .as_mut()
            .context("S14 StarFold direct terminal endpoint 已销毁")?;
        chain.execute_sealed_k4_with_commit_limit(
            adapter,
            terminal,
            terminal_owners,
            full_depth,
            draft_token_ids,
            commit_limit,
            authoritative,
            device_state,
        )
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn paged_arena(&self) -> &Arc<S14Position0PagedWeightArena> {
        self.paged_arena
            .as_ref()
            .expect("production paged arena 只能在最终 base runtime teardown 前释放")
    }

    pub fn resource_inventory(&self) -> S14StarfoldPersistentResourceInventory {
        let adapter = self.adapter.as_ref().map(|adapter| adapter.owner_counts());
        S14StarfoldPersistentResourceInventory {
            base_runtime_owners: usize::from(self.base_runtime.is_some()),
            starfold_runtime_owners: adapter.map_or(0, |counts| counts.runtime_owners),
            paged_arena_owners: usize::from(self.adapter.is_some()),
            expert_catalog_owners: adapter.map_or(0, |counts| counts.catalog_owners),
            microtile_window_owners: adapter.map_or(0, |counts| counts.microtile_window_owners),
            b4_owners: adapter.map_or(0, |counts| counts.b4_owners),
            hc_bridge_stage_owners: adapter.map_or(0, |counts| counts.stage_owners),
            direct_terminal_endpoint_owners: usize::from(self.terminal_endpoint.is_some()),
        }
    }

    pub fn finish_validated_block(&mut self) -> Result<()> {
        self.adapter_mut()?.finish_validated_block()
    }

    /// Session 退休 external owners 前先销毁会持有 provider/hidden Arc 的执行链；
    /// base runtime/paged arena 仍保留到 external destroy 完成之后。
    pub fn destroy_execution_owners(&mut self) -> Result<()> {
        self.terminal_endpoint.take();
        self.adapter
            .take()
            .map(|mut adapter| adapter.destroy())
            .transpose()
            .context("销毁 S14 StarFold K4 adapter")?;
        Ok(())
    }

    /// 请求成功结束时只退休 request-scoped adapter/stage/terminal，并把昂贵的
    /// StarFold 双窗口 runtime 拆回。与 `destroy_execution_owners` 不同，本入口绝不
    /// 销毁 microtile windows、proof store 或 transfer executor。
    pub(crate) fn detach_resident_starfold_runtime(&mut self) -> Result<S14StarfoldRuntime> {
        self.terminal_endpoint.take();
        let adapter = self
            .adapter
            .take()
            .context("归还 resident runtime 时 production adapter 已缺失")?;
        adapter
            .into_resident_runtime()
            .context("从请求级 K4 adapter 拆回 resident StarFold runtime")
    }

    /// external owners 已退休后，把拆回的 StarFold runtime 放回 base runtime，并
    /// 返回完整可再次 `new_session` 的 resident `S14Runtime`。
    pub(crate) fn restore_base_runtime_after_request(
        &mut self,
        starfold_runtime: S14StarfoldRuntime,
    ) -> Result<S14Runtime> {
        if self.adapter.is_some() || self.terminal_endpoint.is_some() {
            bail!("恢复 base runtime 前必须先拆除请求级 adapter/terminal");
        }
        drop(self.paged_arena.take());
        let mut runtime = self
            .base_runtime
            .take()
            .context("恢复请求后 resident root 时 base runtime 已缺失")?;
        runtime
            .restore_starfold_runtime(starfold_runtime)
            .context("把 StarFold 双窗口 owner 放回 base runtime")?;
        Ok(runtime)
    }

    pub fn destroy_base_runtime_owner(&mut self) -> Result<()> {
        if self.adapter.is_some() || self.terminal_endpoint.is_some() {
            bail!("销毁 base runtime 前必须先退出 StarFold adapter/terminal owners");
        }
        // base runtime 自己还持有权威 arena；先释放 production root 的观察引用，
        // 再让 runtime 销毁唯一 Vulkan allocation owner。
        drop(self.paged_arena.take());
        self.base_runtime
            .take()
            .map(S14Runtime::destroy)
            .transpose()
            .context("销毁 S14 base runtime owner")?;
        Ok(())
    }

    pub fn destroy(&mut self) -> Result<()> {
        let adapter_error = self.destroy_execution_owners().err();
        let runtime_error = self.destroy_base_runtime_owner().err();
        match (adapter_error, runtime_error) {
            (None, None) => Ok(()),
            (Some(error), None) => Err(error.context("销毁 S14 StarFold K4 adapter")),
            (None, Some(error)) => Err(error.context("销毁 S14 base runtime owner")),
            (Some(adapter), Some(runtime)) => Err(anyhow!(
                "销毁 S14 StarFold K4 adapter: {adapter:#}; base runtime: {runtime:#}"
            )),
        }
    }
}

impl<P> Drop for S14StarfoldK4ProductionResources<P>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

pub fn build_s14_starfold_k4_production_resources<P>(
    owned: S14StarfoldOwnedRuntimeParts,
    inputs: S14StarfoldK4ProductionResourceInputs<P>,
) -> Result<S14StarfoldK4ProductionResources<P>>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    build_s14_starfold_kblock_production_resources(owned, inputs, K4)
}

pub fn build_s14_starfold_kblock_production_resources<P>(
    mut owned: S14StarfoldOwnedRuntimeParts,
    inputs: S14StarfoldK4ProductionResourceInputs<P>,
    physical_block_size: usize,
) -> Result<S14StarfoldK4ProductionResources<P>>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    let position0_generation_provenance = inputs.position0_generation_provenance;
    if !matches!(physical_block_size, 4 | 8)
        || (position0_generation_provenance.is_some() && physical_block_size != K4)
    {
        bail!("S14 StarFold production builder 的 physical K/provenance 合同非法");
    }
    if !owned.context.timeline_semaphore || !owned.context.has_dedicated_transfer() {
        bail!("S14 StarFold production builder 要求 timeline semaphore 与独立 transfer queue");
    }
    validate_context_owner(
        &owned.context,
        inputs.hc_qkv_provider.context(),
        "StarFold HC/QKV provider",
    )?;
    validate_context_owner(
        &owned.context,
        inputs.hidden_banks.context(),
        "StarFold hidden banks",
    )?;
    if !Arc::ptr_eq(
        inputs.hc_qkv_provider.value().paged_weight_arena(),
        owned.paged_arena(),
    ) {
        bail!("S14 StarFold HC/QKV provider 与 runtime paged arena owner 漂移");
    }
    inputs
        .hc_qkv_provider
        .value()
        .validate_production_bundle(physical_block_size)
        .map_err(anyhow::Error::msg)
        .context("S14 StarFold HC/QKV provider readiness 拒绝")?;
    validate_hidden_banks(inputs.hidden_banks.value())?;
    validate_static_arena(owned.paged_arena())?;
    validate_full_depth_catalog(owned.expert_catalog())?;

    let (_, mut provider) = inputs.hc_qkv_provider.into_parts();
    let (_, hidden_banks) = inputs.hidden_banks.into_parts();
    let mut prefix = provider
        .take_prefix_state_producer()
        .map_err(anyhow::Error::msg)
        .context("消费 S14 StarFold 同源 K4 prefix producer")?;
    let checkpoint_state_bytes = prefix.arena().layout().checkpoint_state_bytes;
    if !Arc::ptr_eq(&owned.context, prefix.context())
        || prefix.arena().layout().block_size != physical_block_size
    {
        let cleanup = prefix.destroy();
        return Err(anyhow!(
            "S14 StarFold prefix producer context/K 漂移; cleanup={cleanup:?}"
        ));
    }
    if position0_generation_provenance.is_some() && prefix.arena().base_position() != 0 {
        let cleanup = prefix.destroy();
        return Err(anyhow!(
            "position0 committed-generation provenance 与非 base0 prefix 冲突; cleanup={cleanup:?}"
        ));
    }

    let recorder = match S14CausalBlockProductionHcQkvLayerRecorder::new(
        Arc::clone(&owned.context),
        Arc::clone(owned.paged_arena()),
        provider,
        hidden_banks.clone(),
    ) {
        Ok(recorder) => recorder,
        Err(error) => {
            let cleanup = prefix.destroy();
            return Err(anyhow!(
                "构造 S14 StarFold HC/QKV recorder: {error:#}; prefix cleanup={cleanup:?}"
            ));
        }
    };
    // context/K 已在上方校验；新 recorder 必为 idle 且尚未安装 producer。
    let recorder = recorder
        .with_prefix_state_producer(prefix)
        .context("安装 S14 StarFold 同 command prefix producer")?;
    let hc_qkv = S14CausalBlockProductionHcQkvAdapter::new(recorder);
    let stage = S14StarfoldConcreteK4Stage::new(
        Arc::clone(&owned.context),
        Arc::clone(owned.paged_arena()),
        hidden_banks,
        hc_qkv,
        position0_generation_provenance,
    )
    .context("构造 S14 StarFold concrete K4 stage")?;
    let runtime = owned
        .runtime
        .take()
        .context("S14 StarFold owned runtime 已被消费")?;
    let adapter = S14StarfoldK4ProductionAdapter::from_owned_runtime(
        Arc::clone(&owned.context),
        runtime,
        Arc::clone(owned.expert_catalog()),
        stage,
    )
    .context("接管 S14Runtime 唯一 StarFold runtime/windows")?;
    let terminal_endpoint = S14StarfoldTerminalEndpoint::new(
        Arc::clone(&owned.context),
        checkpoint_state_bytes,
        // direct terminal 只服务 generation K4；K8 prefill 的 prefix arena 不进入 head。
        K4,
    )
    .context("构造 S14 StarFold 同源 direct terminal endpoint")?;
    let paged_arena = owned
        .paged_arena
        .take()
        .context("S14 StarFold owned paged arena 已被消费")?;
    owned.expert_catalog.take();
    let base_runtime = owned
        .base_runtime
        .take()
        .context("S14 StarFold base runtime owner 已被消费")?;
    Ok(S14StarfoldK4ProductionResources {
        adapter: Some(adapter),
        terminal_endpoint: Some(terminal_endpoint),
        base_runtime: Some(base_runtime),
        context: Arc::clone(&owned.context),
        paged_arena: Some(paged_arena),
    })
}
