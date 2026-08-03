//! Polaris S14 StarFold production root/session 编排。
//!
//! 本模块只组合现有 production owners；不实现模型算法，不创建第二 runtime，也不提供
//! legacy/serial/CPU fallback。root 以显式 lease 在请求间转移唯一 runtime/factory；
//! 成功 close 会只退休请求级 owners 并把常驻 StarFold 双窗口原样归还。

use crate::{
    s14_causal_block_layer::S14CausalBlockHiddenBinding,
    s14_causal_block_production_bundle::S14CausalBlockProductionHcQkvResourceProvider,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_runtime::{S14Runtime, S14Session},
    s14_starfold_cache::{
        process_starfold_verified_lease_cache, STARFOLD_ONE_MIB, STARFOLD_TOP_K,
        STARFOLD_VERIFIED_LEASE_CACHE_CONTRACT_VERSION,
    },
    s14_starfold_k4_adapter::S14StarfoldK4FullDepthReceipt,
    s14_starfold_k4_rebind::{
        rebind_next_committed_k4_block, rebind_next_teacher_forced_prefill_block,
        S14StarfoldCommittedK4Anchor, S14StarfoldK4BlockMode, S14StarfoldNextK4RebindInputs,
        S14StarfoldTeacherForcedCommittedK4Anchor,
    },
    s14_starfold_k4_terminal_chain::{
        s14_starfold_decoder_state_sha256, S14StarfoldK4CommitReceipt,
        S14StarfoldK4TerminalChainOwner, S14_STARFOLD_K4_BLOCK_SIZE,
    },
    s14_starfold_packed_l2::{
        S14_STARFOLD_PACKED_L2_CONTRACT_VERSION, S14_STARFOLD_PACKED_L2_MAX_BYTES,
        S14_STARFOLD_PACKED_L2_MIB_BYTES,
    },
    s14_starfold_production_resources::{
        S14StarfoldK4ProductionResourceInputs, S14StarfoldK4ProductionResources,
        S14StarfoldPersistentResourceInventory,
    },
    s14_starfold_prompt_prefill::{
        S14StarfoldPrefillReadbackOwner, S14StarfoldPromptPrefillPlan,
        S14StarfoldSealedTeacherForcedBlockCommitReceipt, S14StarfoldTeacherForcedBlockPlan,
    },
    s14_starfold_runtime::S14_STARFOLD_WINDOW_COUNT,
    s14_starwave_draft::{
        propose_s14_starwave_draft, propose_s14_starwave_draft_with_navigator,
        S14StarwaveDraftProposal, S14StarwavePosition0CommittedOrigin,
    },
    s14_starwave_history_navigator::{
        observe_process_starwave_committed_sequence, process_starwave_transition_atlas_stats,
        S14StarwaveHistoryNavigator, S14_STARWAVE_TRANSITION_ATLAS_CAPACITY,
        S14_STARWAVE_TRANSITION_ATLAS_CONTRACT_VERSION,
    },
    s14_whole_token_device::{
        WholeTokenDetachedCommittedState, WholeTokenDeviceCommittedCheckpointBinding,
    },
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{DecoderStateV1, FULL_DEPTH_LAYERS, VOCAB_SIZE};
use std::{fmt, sync::Arc};

const K4: usize = S14_STARFOLD_K4_BLOCK_SIZE;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldProductionLeaseState {
    Ready,
    Busy,
    Exhausted,
}

/// factory 必须从真实 owner/Option/集合长度生成，session 不把“请求成功”硬编码成 inventory。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarfoldBlockOwnerInventory {
    pub provider_owners: usize,
    pub hidden_bank_owners: usize,
    pub prefix_arena_owners: usize,
    pub ratio4_state_owners: usize,
    pub upload_auxiliary_owners: usize,
}

pub trait S14StarfoldBlockExternalOwners: fmt::Debug {
    fn inventory(&self) -> S14StarfoldBlockOwnerInventory;
    fn destroy(&mut self, context: &VulkanContext) -> Result<()>;
}

/// runtime/load 到 concrete provider 的唯一暂留边界。实现输入不含 cache root/model config，
/// 因而不能启动第二模型；输出必须同时交付 provider、两块 hidden、initial binding 与显式
/// external owners。没有 default/mock/fixture 实现。
pub trait S14StarfoldBlockResourceFactory {
    type Provider: S14CausalBlockProductionHcQkvResourceProvider + 'static;
    type ExternalOwners: S14StarfoldBlockExternalOwners;

    /// 启动期只汇报 factory 真正持有/绑定的常驻 owners；request block owners 必须为零。
    fn resident_owner_inventory(
        &self,
        context: &Arc<VulkanContext>,
        paged_arena: &Arc<S14Position0PagedWeightArena>,
    ) -> Result<S14StarfoldResidentBlockFactoryInventory>;

    fn build_block(
        &mut self,
        request: S14StarfoldBlockResourceRequest<'_>,
    ) -> Result<S14StarfoldBlockResourceProduct<Self::Provider, Self::ExternalOwners>>;

    /// 上一个请求的 adapter/provider/external owners 已全部退休后，把常驻 uploader 的
    /// per-request 游标重新武装到 sequence0/base0。不得重建或替换 persistent owner。
    fn rearm_for_new_request(&mut self, context: &VulkanContext) -> Result<()>;

    /// 所有 request block/provider Arc 退出后、唯一 runtime/context 退出前退休常驻 loader owners。
    fn retire_persistent_owners(&mut self, context: &VulkanContext) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarfoldResidentBlockFactoryInventory {
    pub context_bindings: usize,
    pub paged_arena_bindings: usize,
    pub verified_mapped_asset_store_owners: usize,
    pub terminal_head_uploader_owners: usize,
    pub request_owned: S14StarfoldBlockOwnerInventory,
    pub forbidden: S14StarfoldForbiddenPathCounters,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldRequestOwnerReadiness {
    /// Provider/hidden/prefix/ratio4/upload auxiliaries 依赖真实 prompt checkpoint，不能启动期伪造。
    DeferredUntilPrompt,
    Active(S14StarfoldBlockOwnerInventory),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldResidentResourceContract {
    pub vulkan_context_owners: usize,
    pub graphics_queue_owners: usize,
    pub transfer_queue_owners: usize,
    pub paged_arena_owners: usize,
    pub starfold_window_owners: usize,
    pub starfold_window_bytes_each: u32,
    pub starfold_physical_allocation_bytes: u64,
    pub full_depth_layers: usize,
    pub physical_block_lanes: usize,
    pub routed_top_k: usize,
    pub verified_mapped_asset_store_owners: usize,
    pub verified_lease_cache_owners: usize,
    pub verified_lease_cache_capacity_entries: usize,
    pub verified_lease_cache_contract_version: u32,
    pub packed_l2_owners: usize,
    pub packed_l2_capacity_bytes: u64,
    pub packed_l2_contract_version: u32,
    pub starwave_transition_atlas_owners: usize,
    pub starwave_transition_atlas_capacity_entries: usize,
    pub starwave_transition_atlas_contract_version: u32,
    pub terminal_head_uploader_owners: usize,
    pub starwave_commit_owners: usize,
    pub request_owned: S14StarfoldRequestOwnerReadiness,
    pub forbidden: S14StarfoldForbiddenPathCounters,
}

pub struct S14StarfoldBlockResourceRequest<'a> {
    pub context: &'a Arc<VulkanContext>,
    pub paged_arena: &'a Arc<S14Position0PagedWeightArena>,
    pub authoritative: &'a DecoderStateV1,
    pub committed_device: WholeTokenDeviceCommittedCheckpointBinding<'a>,
    /// 从同一个 request-owned `WholeTokenDeviceState` 做出的 device→device snapshot。
    /// factory 必须消费进 prefix initialization owner；若在消费前失败，本 owner 自动销毁。
    pub committed_snapshot: S14StarfoldDetachedCheckpointOwner,
    /// 仅允许单-token prompt 的 base0 generation；非 base0 与 ForcedPrefill 必须为 None。
    pub position0_committed_origin: Option<S14StarwavePosition0CommittedOrigin>,
    pub block_sequence: u64,
    pub mode: S14StarfoldK4BlockMode,
    pub input_token_ids: [u32; K4],
    pub expected_initial_hidden_generation: u64,
}

/// detached checkpoint 的最窄 RAII 移交壳，防止 concrete factory 的失败分支泄漏 snapshot。
pub struct S14StarfoldDetachedCheckpointOwner {
    context: Arc<VulkanContext>,
    detached: Option<WholeTokenDetachedCommittedState>,
}

impl S14StarfoldDetachedCheckpointOwner {
    fn from_same_session(
        context: Arc<VulkanContext>,
        detached: WholeTokenDetachedCommittedState,
        committed: &WholeTokenDeviceCommittedCheckpointBinding<'_>,
    ) -> Result<Self> {
        if detached.state_bytes != committed.state_bytes()
            || detached.epoch != committed.epoch()
            || detached.active_bank != committed.active_bank()
            || detached.source_device != context.device.handle()
            || detached.source_graphics_queue != context.q_graphics
            || detached.source_graphics_queue_family != context.qf_graphics
        {
            detached.buffer.destroy(&context);
            bail!("detached checkpoint 与 request committed device/context 身份漂移");
        }
        Ok(Self {
            context,
            detached: Some(detached),
        })
    }

    pub fn take(&mut self) -> Result<WholeTokenDetachedCommittedState> {
        self.detached
            .take()
            .context("detached checkpoint owner 已被 factory 消费")
    }

    pub fn identity(&self) -> Result<(u64, u64, usize)> {
        let detached = self
            .detached
            .as_ref()
            .context("detached checkpoint owner 已被 factory 消费")?;
        Ok((detached.state_bytes, detached.epoch, detached.active_bank))
    }
}

impl Drop for S14StarfoldDetachedCheckpointOwner {
    fn drop(&mut self) {
        if let Some(detached) = self.detached.take() {
            detached.buffer.destroy(&self.context);
        }
    }
}

pub struct S14StarfoldBlockResourceProduct<P, E> {
    pub production: S14StarfoldK4ProductionResourceInputs<P>,
    pub initial_hidden: S14CausalBlockHiddenBinding,
    pub external_owners: E,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarfoldForbiddenPathCounters {
    pub legacy_union_calls: u64,
    pub legacy_grouped_moe_calls: u64,
    pub serial_position0_calls: u64,
    pub serial_token_forward_calls: u64,
    pub cpu_fallback_calls: u64,
    pub v38_calls: u64,
    pub v47_calls: u64,
    pub transformer_calls: u64,
    pub whole_model_fallback_calls: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarfoldProductionResourceInventory {
    pub persistent: S14StarfoldPersistentResourceInventory,
    pub active_block: S14StarfoldBlockOwnerInventory,
    pub pending_rebind_external_owner_sets: usize,
    pub retired_external_owner_sets: usize,
    pub decoder_state_owners: usize,
    pub whole_token_device_owners: usize,
    pub prefill_readback_owners: usize,
    pub forbidden: S14StarfoldForbiddenPathCounters,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldCheckpointIdentity {
    pub position: u32,
    pub epoch: u64,
    pub decoder_state_sha256: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct S14StarfoldCommittedLedgerDelta {
    pub requested_commit_limit: usize,
    pub proposal_safe_commit_limit: usize,
    pub effective_commit_limit: usize,
    pub committed_token_ids: Vec<u32>,
    pub checkpoint: S14StarfoldCheckpointIdentity,
    pub receipt: S14StarfoldK4CommitReceipt,
}

pub struct S14StarfoldProductionRoot<F: S14StarfoldBlockResourceFactory> {
    runtime: Option<S14Runtime>,
    factory: Option<F>,
    commit_chain: Option<S14StarfoldK4TerminalChainOwner>,
    starwave_eos_token_id: Option<u32>,
    lease: S14StarfoldProductionLeaseState,
}

impl<F: S14StarfoldBlockResourceFactory> S14StarfoldProductionRoot<F> {
    pub fn new(runtime: S14Runtime, factory: F) -> Self {
        Self {
            runtime: Some(runtime),
            factory: Some(factory),
            commit_chain: Some(S14StarfoldK4TerminalChainOwner::new()),
            starwave_eos_token_id: None,
            lease: S14StarfoldProductionLeaseState::Ready,
        }
    }

    /// 由同一个已验签 chat codec 注入 EOS；root Busy 后禁止改变生成协议。
    pub fn configure_starwave_eos_token_id(&mut self, eos_token_id: u32) -> Result<()> {
        if self.lease != S14StarfoldProductionLeaseState::Ready {
            bail!("S14 production root Busy/Exhausted 时禁止切换 StarWave EOS");
        }
        if eos_token_id >= polaris_s14_runner::VOCAB_SIZE {
            bail!("S14 StarWave EOS token 越出冻结 vocab");
        }
        match self.starwave_eos_token_id {
            Some(current) if current != eos_token_id => {
                bail!("S14 production root 禁止在请求间切换 EOS token")
            }
            _ => self.starwave_eos_token_id = Some(eos_token_id),
        }
        Ok(())
    }

    pub fn lease_state(&self) -> S14StarfoldProductionLeaseState {
        self.lease
    }

    /// API backend 构造前可读取的强拓扑回执。返回 `Ok` 即代表所有 persistent-ready
    /// 不变量已经由 root/factory 真实字段闭合；依赖 prompt 的 block owners 明确延迟。
    pub fn resident_resource_contract(&self) -> Result<S14StarfoldResidentResourceContract> {
        if self.lease != S14StarfoldProductionLeaseState::Ready {
            bail!("S14 production root 非 Ready，拒绝签发 resident resource contract");
        }
        let runtime = self
            .runtime
            .as_ref()
            .context("resident contract 缺少唯一 runtime owner")?;
        let factory = self
            .factory
            .as_ref()
            .context("resident contract 缺少唯一 block factory owner")?;
        let context = runtime.context();
        let paged_arena = runtime.paged_arena()?;
        let windows = runtime
            .starfold_window_contract()
            .context("resident contract 缺少 StarFold double-window runtime")?;
        let physical_allocation = runtime
            .starfold_physical_allocation_bytes()
            .context("resident contract 缺少 StarFold physical allocation")?;
        let factory_inventory = factory.resident_owner_inventory(context, paged_arena)?;
        let forbidden = factory_inventory.forbidden;
        let verified_lease_cache = process_starfold_verified_lease_cache();
        let packed_l2 = runtime
            .starfold_packed_l2_cache_stats()
            .context("resident contract 缺少 StarFold packed L2 owner")?;
        let _transition_atlas = process_starwave_transition_atlas_stats();
        let expected_physical_allocation = u64::from(windows.microtile_bytes)
            .checked_mul(u64::try_from(S14_STARFOLD_WINDOW_COUNT)?)
            .context("resident StarFold window bytes overflow")?;
        let valid_supertile = windows.microtile_bytes >= STARFOLD_ONE_MIB
            && windows.microtile_bytes <= 64 * STARFOLD_ONE_MIB
            && windows.microtile_bytes.is_power_of_two();
        let valid = context.q_graphics != vk::Queue::null()
            && context.q_transfer != vk::Queue::null()
            && context.has_dedicated_transfer()
            && windows.window_count == S14_STARFOLD_WINDOW_COUNT
            && valid_supertile
            && physical_allocation == expected_physical_allocation
            && FULL_DEPTH_LAYERS.len() == 43
            && K4 == 4
            && STARFOLD_TOP_K == 6
            && factory_inventory.context_bindings == 1
            && factory_inventory.paged_arena_bindings == 1
            && factory_inventory.verified_mapped_asset_store_owners == 1
            && packed_l2.capacity_bytes <= S14_STARFOLD_PACKED_L2_MAX_BYTES
            && packed_l2.capacity_bytes % S14_STARFOLD_PACKED_L2_MIB_BYTES == 0
            && packed_l2.resident_bytes <= packed_l2.capacity_bytes
            && (packed_l2.capacity_bytes != 0 || packed_l2.entries == 0)
            && factory_inventory.terminal_head_uploader_owners == 1
            && factory_inventory.request_owned == S14StarfoldBlockOwnerInventory::default()
            && forbidden == S14StarfoldForbiddenPathCounters::default()
            && self.commit_chain.is_some();
        if !valid {
            bail!("S14 resident root/factory owner topology 未闭合，拒绝启动 API backend");
        }
        Ok(S14StarfoldResidentResourceContract {
            vulkan_context_owners: usize::from(self.runtime.is_some()),
            graphics_queue_owners: usize::from(context.q_graphics != vk::Queue::null()),
            transfer_queue_owners: usize::from(context.q_transfer != vk::Queue::null()),
            paged_arena_owners: usize::from(runtime.paged_arena().is_ok()),
            starfold_window_owners: windows.window_count,
            starfold_window_bytes_each: windows.microtile_bytes,
            starfold_physical_allocation_bytes: physical_allocation,
            full_depth_layers: FULL_DEPTH_LAYERS.len(),
            physical_block_lanes: K4,
            routed_top_k: STARFOLD_TOP_K,
            verified_mapped_asset_store_owners: factory_inventory
                .verified_mapped_asset_store_owners,
            verified_lease_cache_owners: 1,
            verified_lease_cache_capacity_entries: verified_lease_cache.capacity_entries(),
            verified_lease_cache_contract_version: STARFOLD_VERIFIED_LEASE_CACHE_CONTRACT_VERSION,
            packed_l2_owners: 1,
            packed_l2_capacity_bytes: packed_l2.capacity_bytes,
            packed_l2_contract_version: S14_STARFOLD_PACKED_L2_CONTRACT_VERSION,
            starwave_transition_atlas_owners: 1,
            starwave_transition_atlas_capacity_entries: S14_STARWAVE_TRANSITION_ATLAS_CAPACITY,
            starwave_transition_atlas_contract_version:
                S14_STARWAVE_TRANSITION_ATLAS_CONTRACT_VERSION,
            terminal_head_uploader_owners: factory_inventory.terminal_head_uploader_owners,
            starwave_commit_owners: usize::from(self.commit_chain.is_some()),
            request_owned: S14StarfoldRequestOwnerReadiness::DeferredUntilPrompt,
            forbidden,
        })
    }

    pub fn begin_prompt_session(
        &mut self,
        prompt: &[u32],
        max_seq_len: u32,
    ) -> Result<S14StarfoldProductionSession<F>> {
        let plan = S14StarfoldPromptPrefillPlan::build(prompt)?;
        let navigator = self.starwave_eos_token_id.map(|eos_token_id| {
            crate::s14_starwave_draft::S14StarwaveProductionNavigatorAdapter::new(
                S14StarwaveHistoryNavigator::new(eos_token_id),
            )
        });
        if prompt.len() > max_seq_len as usize {
            bail!("prompt 长度超过 max_seq_len");
        }
        match self.lease {
            S14StarfoldProductionLeaseState::Ready => {
                self.lease = S14StarfoldProductionLeaseState::Busy;
            }
            S14StarfoldProductionLeaseState::Busy => bail!("S14 production root 正被请求独占"),
            S14StarfoldProductionLeaseState::Exhausted => {
                bail!("S14 production root adapter 不支持跨请求 reset，已 fail-closed")
            }
        }
        let mut runtime = match self.runtime.take() {
            Some(runtime) => runtime,
            None => {
                self.lease = S14StarfoldProductionLeaseState::Exhausted;
                bail!("S14 production root 缺少唯一 runtime，已 fail-closed");
            }
        };
        let factory = match self.factory.take() {
            Some(factory) => factory,
            None => {
                self.runtime = Some(runtime);
                self.lease = S14StarfoldProductionLeaseState::Exhausted;
                bail!("S14 production root 缺少唯一 block factory，已 fail-closed");
            }
        };
        let commit_chain = match self.commit_chain.take() {
            Some(chain) => chain,
            None => {
                self.runtime = Some(runtime);
                self.factory = Some(factory);
                self.lease = S14StarfoldProductionLeaseState::Exhausted;
                bail!("S14 production root 缺少唯一 StarWave commit chain owner，已 fail-closed");
            }
        };
        if let Err(error) = runtime.begin_starfold_verified_lease_request_epoch() {
            self.runtime = Some(runtime);
            self.factory = Some(factory);
            self.commit_chain = Some(commit_chain);
            self.lease = S14StarfoldProductionLeaseState::Ready;
            return Err(error.context("签发 prompt request verified lease validation epoch"));
        }
        let decoder = match runtime.new_session(plan.first_input_token_id, max_seq_len) {
            Ok(session) => session,
            Err(error) => {
                self.runtime = Some(runtime);
                self.factory = Some(factory);
                self.commit_chain = Some(commit_chain);
                self.lease = S14StarfoldProductionLeaseState::Ready;
                return Err(error.context("建立真实 position0 host/device checkpoint"));
            }
        };
        let prefill_readback = if plan.blocks.is_empty() {
            None
        } else {
            let state_bytes = decoder.authoritative_state().native_arena.bytes().len() as u64;
            match S14StarfoldPrefillReadbackOwner::new(Arc::clone(decoder.context()), state_bytes) {
                Ok(owner) => Some(owner),
                Err(error) => {
                    let cleanup = decoder.destroy();
                    self.runtime = Some(runtime);
                    self.factory = Some(factory);
                    self.commit_chain = Some(commit_chain);
                    self.lease = S14StarfoldProductionLeaseState::Ready;
                    return Err(anyhow!(
                        "构造常驻 prefill readback owner: {error:#}; decoder cleanup={cleanup:?}"
                    ));
                }
            }
        };
        // session 独占唯一 owner，直到显式 `return_prompt_session` 把它们归还。
        self.lease = S14StarfoldProductionLeaseState::Busy;
        let mut session = S14StarfoldProductionSession {
            factory: Some(factory),
            runtime: Some(runtime),
            decoder: Some(decoder),
            prefill_readback,
            resources: None,
            chain: commit_chain,
            current_external: None,
            pending_external: None,
            retired_external: Vec::new(),
            last_prefill_commit: None,
            navigator,
            forbidden: S14StarfoldForbiddenPathCounters::default(),
            closed: false,
        };
        if let Err(error) = session.prefill_prompt(&plan) {
            let cleanup = session.cleanup_inner();
            self.lease = S14StarfoldProductionLeaseState::Exhausted;
            return Err(anyhow!(
                "S14 prompt ForcedPrefill 失败: {error:#}; cleanup={cleanup:?}"
            ));
        }
        Ok(session)
    }

    /// 成功请求结束后的唯一 owner-return 边界。只有 clean/Idle session 可以归还；
    /// 任一 request owner、uploader phase 或 Vulkan context 漂移都会把 root 置为
    /// `Exhausted`，绝不通过改 lease 标志伪装可复用。
    pub fn return_prompt_session(
        &mut self,
        session: S14StarfoldProductionSession<F>,
    ) -> Result<()> {
        if self.lease != S14StarfoldProductionLeaseState::Busy
            || self.runtime.is_some()
            || self.factory.is_some()
            || self.commit_chain.is_some()
        {
            self.lease = S14StarfoldProductionLeaseState::Exhausted;
            bail!("S14 production root owner-return 前 lease/owner 空槽不合法");
        }
        let (runtime, mut factory, committed_sequence) = match session.into_resident_parts() {
            Ok(parts) => parts,
            Err(error) => {
                self.lease = S14StarfoldProductionLeaseState::Exhausted;
                return Err(error.context("拆回 S14 resident runtime/factory"));
            }
        };
        if let Err(error) = factory.rearm_for_new_request(runtime.context()) {
            let factory_cleanup = factory.retire_persistent_owners(runtime.context());
            let runtime_cleanup = runtime.destroy();
            self.lease = S14StarfoldProductionLeaseState::Exhausted;
            return Err(anyhow!(
                "S14 resident factory 新请求 rearm 失败: {error:#}; factory cleanup={factory_cleanup:?}; runtime cleanup={runtime_cleanup:?}"
            ));
        }
        self.runtime = Some(runtime);
        self.factory = Some(factory);
        self.commit_chain = Some(S14StarfoldK4TerminalChainOwner::new());
        self.lease = S14StarfoldProductionLeaseState::Ready;
        if let Err(error) = self.resident_resource_contract() {
            self.lease = S14StarfoldProductionLeaseState::Exhausted;
            return Err(error.context("owner-return 后 resident resource contract 未闭合"));
        }
        if let Some(committed_sequence) = committed_sequence {
            observe_process_starwave_committed_sequence(&committed_sequence);
        }
        Ok(())
    }
}

impl<F: S14StarfoldBlockResourceFactory> Drop for S14StarfoldProductionRoot<F> {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        let Some(mut factory) = self.factory.take() else {
            let _ = runtime.destroy();
            return;
        };
        if factory.retire_persistent_owners(runtime.context()).is_err() {
            // Drop 无法回传 cleanup 错误；保留同源 factory/runtime，绝不先 teardown context。
            std::mem::forget(factory);
            std::mem::forget(runtime);
            return;
        }
        drop(factory);
        let _ = runtime.destroy();
    }
}

pub struct S14StarfoldProductionSession<F>
where
    F: S14StarfoldBlockResourceFactory,
{
    factory: Option<F>,
    runtime: Option<S14Runtime>,
    decoder: Option<S14Session>,
    prefill_readback: Option<S14StarfoldPrefillReadbackOwner>,
    resources: Option<S14StarfoldK4ProductionResources<F::Provider>>,
    chain: S14StarfoldK4TerminalChainOwner,
    current_external: Option<F::ExternalOwners>,
    pending_external: Option<F::ExternalOwners>,
    retired_external: Vec<F::ExternalOwners>,
    last_prefill_commit: Option<S14StarfoldSealedTeacherForcedBlockCommitReceipt>,
    navigator: Option<
        crate::s14_starwave_draft::S14StarwaveProductionNavigatorAdapter<
            S14StarwaveHistoryNavigator,
        >,
    >,
    forbidden: S14StarfoldForbiddenPathCounters,
    closed: bool,
}

impl<F> S14StarfoldProductionSession<F>
where
    F: S14StarfoldBlockResourceFactory,
{
    pub fn checkpoint_identity(&self) -> Result<S14StarfoldCheckpointIdentity> {
        let state = self.decoder()?.authoritative_state();
        let sha = s14_starfold_decoder_state_sha256(state)?;
        Ok(S14StarfoldCheckpointIdentity {
            position: state.position,
            epoch: state.commit_epoch,
            decoder_state_sha256: *sha.as_bytes(),
        })
    }

    pub fn authoritative_state(&self) -> Result<&DecoderStateV1> {
        Ok(self.decoder()?.authoritative_state())
    }

    pub fn resource_inventory(&self) -> Result<S14StarfoldProductionResourceInventory> {
        let persistent = if let Some(resources) = &self.resources {
            resources.resource_inventory()
        } else {
            let runtime = self
                .runtime
                .as_ref()
                .context("production session 缺少 runtime owner")?;
            let starfold = runtime
                .starfold_window_contract()
                .filter(|_| runtime.starfold_physical_allocation_bytes().unwrap_or(0) != 0);
            S14StarfoldPersistentResourceInventory {
                base_runtime_owners: 1,
                starfold_runtime_owners: usize::from(starfold.is_some()),
                paged_arena_owners: usize::from(runtime.paged_arena().is_ok()),
                expert_catalog_owners: usize::from(starfold.is_some()),
                microtile_window_owners: starfold.map_or(0, |contract| contract.window_count),
                b4_owners: 0,
                hc_bridge_stage_owners: 0,
                direct_terminal_endpoint_owners: 0,
            }
        };
        let (decoder_state_owners, whole_token_device_owners) = self
            .decoder
            .as_ref()
            .map_or((0, 0), S14Session::authoritative_owner_counts);
        Ok(S14StarfoldProductionResourceInventory {
            persistent,
            active_block: self
                .current_external
                .as_ref()
                .map_or(S14StarfoldBlockOwnerInventory::default(), |owners| {
                    owners.inventory()
                }),
            retired_external_owner_sets: self.retired_external.len(),
            pending_rebind_external_owner_sets: usize::from(self.pending_external.is_some()),
            decoder_state_owners,
            whole_token_device_owners,
            prefill_readback_owners: usize::from(self.prefill_readback.is_some()),
            forbidden: self.forbidden,
        })
    }

    /// 返回 `decision.committed_token_ids` 的完整 ledger 增量。effective limit 始终是
    /// `min(requested_limit, proposal_safe_limit)`；只有无 EOS、无历史命中或确定性
    /// fallback 路径才会 fail-closed 为 lane0-only。
    pub fn commit_next_block(
        &mut self,
        requested_limit: usize,
    ) -> Result<S14StarfoldCommittedLedgerDelta> {
        if !(1..=K4).contains(&requested_limit) {
            bail!("generation commit limit 必须为 1..=4");
        }
        let proposal = self.propose_generation()?;
        let proposal_safe_limit = proposal.effective_commit_limit();
        let effective_limit = requested_limit.min(proposal_safe_limit);
        let ledger_before = self.decoder()?.committed_tokens().len();
        let full_depth = self.execute_generation_block(&proposal)?;
        let draft = *proposal.draft_token_ids();
        let receipt = {
            let resources = self
                .resources
                .as_mut()
                .context("generation 缺少 production resources")?;
            let decoder = self
                .decoder
                .as_mut()
                .context("generation 缺少 decoder session")?;
            let (authoritative, device) = decoder.authoritative_host_device_mut()?;
            resources.execute_and_commit_sealed_k4_with_commit_limit(
                &mut self.chain,
                full_depth,
                &draft,
                effective_limit,
                authoritative,
                device,
            )?
        };
        self.accumulate_generation_audit(&receipt)?;
        let committed_token_ids = receipt.decision.committed_token_ids.clone();
        let ledger = self.decoder()?.committed_tokens();
        let observed = ledger
            .get(ledger_before..)
            .context("generation ledger 未增长")?
            .iter()
            .map(|record| record.predicted_token_id)
            .collect::<Vec<_>>();
        if receipt.commit_limit != effective_limit
            || committed_token_ids != observed
            || committed_token_ids.len() != receipt.committed_tokens
        {
            bail!("generation decision/receipt/authoritative ledger 增量未闭合");
        }
        self.last_prefill_commit.take();
        Ok(S14StarfoldCommittedLedgerDelta {
            requested_commit_limit: requested_limit,
            proposal_safe_commit_limit: proposal_safe_limit,
            effective_commit_limit: effective_limit,
            committed_token_ids,
            checkpoint: self.checkpoint_identity()?,
            receipt,
        })
    }

    pub fn close(mut self) -> Result<()> {
        self.cleanup_inner()
    }

    fn into_resident_parts(mut self) -> Result<(S14Runtime, F, Option<Vec<u32>>)> {
        if self.closed {
            bail!("S14 production session 已关闭，不能归还 resident root");
        }
        let context = self
            .resources
            .as_ref()
            .map(|resources| Arc::clone(resources.context()))
            .or_else(|| {
                self.runtime
                    .as_ref()
                    .map(|runtime| Arc::clone(runtime.context()))
            })
            .context("归还 resident root 时缺少 Vulkan context")?;

        // 这里只冻结候选输入链；真正写入 atlas 必须等 root 完整取回 runtime/factory、
        // rearm 并重新验签成功之后。这样任何后续 owner cleanup 失败都不会污染星图。
        let committed_sequence = if self.navigator.is_some() {
            let decoder = self
                .decoder
                .as_ref()
                .context("冻结 committed transition atlas 输入时缺少 decoder")?;
            let authoritative = decoder.authoritative_state();
            let mut committed_sequence = authoritative
                .committed_tokens
                .iter()
                .map(|record| record.input_token_id)
                .collect::<Vec<_>>();
            committed_sequence.push(authoritative.input_token_id);
            Some(committed_sequence)
        } else {
            None
        };

        // 先释放持有 provider/hidden/prefix Arc 的 request execution graph，但保留
        // 唯一 StarFold runtime/windows；随后 external owners 才能安全退休。
        let detached_starfold = self
            .resources
            .as_mut()
            .map(S14StarfoldK4ProductionResources::detach_resident_starfold_runtime)
            .transpose()
            .context("拆回请求后的 resident StarFold runtime")?;
        if let Some(pending) = self.pending_external.take() {
            self.retired_external.push(pending);
        }
        if let Some(current) = self.current_external.take() {
            self.retired_external.push(current);
        }
        retire_all_external(&mut self.retired_external, &context)
            .context("归还 resident root 前退休全部 request external owners")?;

        if let Some(decoder) = self.decoder.take() {
            decoder
                .destroy()
                .context("归还 resident root 前销毁请求 decoder/device state")?;
        }
        if let Some(readback) = self.prefill_readback.take() {
            readback
                .destroy()
                .context("归还 resident root 前销毁请求 prefill readback")?;
        }
        let runtime = match (self.resources.as_mut(), detached_starfold) {
            (Some(resources), Some(starfold)) => resources
                .restore_base_runtime_after_request(starfold)
                .context("恢复完整 resident S14Runtime")?,
            (None, None) => self
                .runtime
                .take()
                .context("无 block 请求归还时缺少未拆分 runtime")?,
            _ => bail!("请求归还时 base/StarFold runtime owner 组合不完整"),
        };
        self.resources.take();
        let factory = self
            .factory
            .take()
            .context("归还 resident root 时 block factory 已缺失")?;
        self.closed = true;
        Ok((runtime, factory, committed_sequence))
    }

    fn prefill_prompt(&mut self, plan: &S14StarfoldPromptPrefillPlan) -> Result<()> {
        for block in &plan.blocks {
            let receipt = self.execute_prefill_block(block)?;
            self.forbidden.serial_token_forward_calls = self
                .forbidden
                .serial_token_forward_calls
                .checked_add(u64::from(receipt.commit.serial_token_forward_calls))
                .context("prefill serial audit overflow")?;
            self.last_prefill_commit = Some(receipt);
        }
        plan.validate_final_state(self.decoder()?.authoritative_state())?;
        if let Some(last) = &self.last_prefill_commit {
            self.chain = S14StarfoldK4TerminalChainOwner::after_teacher_forced_prefill(
                u64::try_from(plan.blocks.len()).context("prefill block count 超出 u64")?,
                last.commit.base_position,
            )?;
        }
        Ok(())
    }

    fn execute_prefill_block(
        &mut self,
        block: &S14StarfoldTeacherForcedBlockPlan,
    ) -> Result<S14StarfoldSealedTeacherForcedBlockCommitReceipt> {
        let product = self.build_block_resources(
            block.physical_input_token_ids,
            S14StarfoldK4BlockMode::TeacherForcedPrefill,
            None,
        )?;
        let S14StarfoldBlockResourceProduct {
            production,
            initial_hidden,
            external_owners,
        } = product;
        let sealed = if self.resources.is_none() {
            self.install_first_resources(production, external_owners)?;
            let authoritative = self
                .decoder
                .as_ref()
                .context("prefill 首块缺少 decoder")?
                .authoritative_state();
            let resources = self
                .resources
                .as_mut()
                .context("prefill 首块 resources 安装失败")?;
            resources.execute_teacher_forced_prefill_product(
                authoritative,
                block,
                initial_hidden,
            )?
        } else {
            self.pending_external = Some(external_owners);
            let previous = self
                .last_prefill_commit
                .clone()
                .context("后续 prefill block 缺少上一强 commit receipt")?;
            let full_depth = {
                let resources = self.resources.as_mut().expect("checked above");
                let decoder = self.decoder.as_mut().context("prefill 缺少 decoder")?;
                let (authoritative, device) = decoder.authoritative_host_device_mut()?;
                let launch = rebind_next_teacher_forced_prefill_block(
                    resources,
                    S14StarfoldTeacherForcedCommittedK4Anchor {
                        previous_commit: &previous,
                        committed_device: device,
                        authoritative,
                    },
                    S14StarfoldNextK4RebindInputs {
                        production,
                        input_token_ids: block.physical_input_token_ids,
                        initial_hidden,
                        mode: S14StarfoldK4BlockMode::TeacherForcedPrefill,
                    },
                )?;
                let rebound_external = self
                    .pending_external
                    .take()
                    .context("prefill rebind 成功后缺少 pending external owners")?;
                retire_rebound_external(
                    &mut self.current_external,
                    &mut self.retired_external,
                    resources.context(),
                    rebound_external,
                )?;
                launch.execute(resources)?
            };
            self.resources
                .as_mut()
                .expect("checked above")
                .finish_teacher_forced_prefill_product(full_depth, block)?
        };
        let context = Arc::clone(self.resources.as_ref().expect("installed").context());
        let decoder = self.decoder.as_mut().context("prefill 缺少 decoder")?;
        let (authoritative, device) = decoder.authoritative_host_device_mut()?;
        sealed.commit(
            &context,
            authoritative,
            device,
            self.prefill_readback
                .as_mut()
                .context("prefill 缺少常驻 readback owner")?,
        )
    }

    fn propose_generation(&mut self) -> Result<S14StarwaveDraftProposal> {
        let decoder = self.decoder.as_mut().context("generation 缺少 decoder")?;
        let (authoritative, device) = decoder.authoritative_host_device_mut()?;
        let position0_origin = if authoritative.position == 0 {
            let committed = device.committed_checkpoint_binding()?;
            Some(
                S14StarwavePosition0CommittedOrigin::from_committed_checkpoint(
                    authoritative,
                    &committed,
                )
                .map_err(|error| anyhow!(error.to_string()))?,
            )
        } else {
            None
        };
        match self.navigator.as_mut() {
            Some(navigator) => propose_s14_starwave_draft_with_navigator(
                authoritative,
                position0_origin,
                navigator,
            ),
            None => propose_s14_starwave_draft(authoritative, position0_origin, None),
        }
        .map_err(|error| anyhow!(error.to_string()))
    }

    fn execute_generation_block(
        &mut self,
        proposal: &S14StarwaveDraftProposal,
    ) -> Result<S14StarfoldK4FullDepthReceipt> {
        let product = self.build_block_resources(
            *proposal.input_token_ids(),
            S14StarfoldK4BlockMode::SpeculativeGeneration,
            proposal.position0_committed_origin(),
        )?;
        let S14StarfoldBlockResourceProduct {
            production,
            initial_hidden,
            external_owners,
        } = product;
        if self.resources.is_none() {
            self.install_first_resources(production, external_owners)?;
            let authoritative = self
                .decoder
                .as_ref()
                .context("generation 首块缺少 decoder")?
                .authoritative_state();
            let resources = self.resources.as_mut().expect("installed");
            return resources.execute_generation_proposal(authoritative, proposal, initial_hidden);
        }
        self.pending_external = Some(external_owners);
        if let Some(previous) = self.last_prefill_commit.clone() {
            let resources = self.resources.as_mut().expect("checked above");
            let decoder = self.decoder.as_mut().context("generation 缺少 decoder")?;
            let (authoritative, device) = decoder.authoritative_host_device_mut()?;
            let launch = rebind_next_teacher_forced_prefill_block(
                resources,
                S14StarfoldTeacherForcedCommittedK4Anchor {
                    previous_commit: &previous,
                    committed_device: device,
                    authoritative,
                },
                S14StarfoldNextK4RebindInputs {
                    production,
                    input_token_ids: *proposal.input_token_ids(),
                    initial_hidden,
                    mode: S14StarfoldK4BlockMode::SpeculativeGeneration,
                },
            )?;
            let rebound_external = self
                .pending_external
                .take()
                .context("prefill→generation rebind 成功后缺少 pending external owners")?;
            retire_rebound_external(
                &mut self.current_external,
                &mut self.retired_external,
                resources.context(),
                rebound_external,
            )?;
            return launch.execute(resources);
        }
        let previous = self
            .chain
            .last_commit()
            .cloned()
            .context("后续 generation block 缺少上一 terminal commit receipt")?;
        let resources = self.resources.as_mut().expect("checked above");
        let decoder = self.decoder.as_mut().context("generation 缺少 decoder")?;
        let (authoritative, device) = decoder.authoritative_host_device_mut()?;
        let launch = rebind_next_committed_k4_block(
            resources,
            S14StarfoldCommittedK4Anchor {
                previous_commit: &previous,
                committed_device: device,
                authoritative,
            },
            S14StarfoldNextK4RebindInputs {
                production,
                input_token_ids: *proposal.input_token_ids(),
                initial_hidden,
                mode: S14StarfoldK4BlockMode::SpeculativeGeneration,
            },
        )?;
        let rebound_external = self
            .pending_external
            .take()
            .context("generation rebind 成功后缺少 pending external owners")?;
        retire_rebound_external(
            &mut self.current_external,
            &mut self.retired_external,
            resources.context(),
            rebound_external,
        )?;
        launch.execute(resources)
    }

    fn build_block_resources(
        &mut self,
        input_token_ids: [u32; K4],
        mode: S14StarfoldK4BlockMode,
        position0_committed_origin: Option<S14StarwavePosition0CommittedOrigin>,
    ) -> Result<S14StarfoldBlockResourceProduct<F::Provider, F::ExternalOwners>> {
        if self.pending_external.is_some() || !self.retired_external.is_empty() {
            bail!(
                "上一 block rebind/external retirement 未闭合；production session 已 fail-closed，只允许 close"
            );
        }
        if input_token_ids.iter().any(|&token| token >= VOCAB_SIZE) {
            bail!("block factory input token 越出 vocab");
        }
        let (context, paged_arena) = if let Some(resources) = &self.resources {
            (
                Arc::clone(resources.context()),
                Arc::clone(resources.paged_arena()),
            )
        } else {
            let runtime = self.runtime.as_ref().context("首块 factory 缺少 runtime")?;
            (
                Arc::clone(runtime.context()),
                Arc::clone(runtime.paged_arena()?),
            )
        };
        let (block_sequence, expected_generation) = if let Some(previous) = self.chain.last_commit()
        {
            (
                // terminal receipt 是一基已提交 block count；factory request 使用零基序号。
                previous.block_sequence,
                previous
                    .sealed_full_depth
                    .final_hidden
                    .generation
                    .checked_add(1)
                    .context("hidden generation overflow")?,
            )
        } else if let Some(previous) = &self.last_prefill_commit {
            (
                u64::try_from(previous.commit.block_index + 1)
                    .context("prefill block sequence 超出 u64")?,
                previous
                    .sealed_full_depth
                    .final_hidden
                    .generation
                    .checked_add(1)
                    .context("prefill hidden generation overflow")?,
            )
        } else {
            (0, 0)
        };
        let decoder = self
            .decoder
            .as_mut()
            .context("block factory 缺少 decoder")?;
        let (authoritative, device) = decoder.authoritative_host_device_mut()?;
        let detached = device
            .snapshot_detached_committed_state(&context)
            .context("为 concrete block factory snapshot committed device checkpoint")?;
        let committed_device = device.committed_checkpoint_binding()?;
        let committed_snapshot = S14StarfoldDetachedCheckpointOwner::from_same_session(
            Arc::clone(&context),
            detached,
            &committed_device,
        )?;
        let product = self
            .factory
            .as_mut()
            .context("block factory 已被归还或关闭")?
            .build_block(S14StarfoldBlockResourceRequest {
                context: &context,
                paged_arena: &paged_arena,
                authoritative,
                committed_device,
                committed_snapshot,
                position0_committed_origin,
                block_sequence,
                mode,
                input_token_ids,
                expected_initial_hidden_generation: expected_generation,
            })?;
        let inventory = product.external_owners.inventory();
        if inventory.provider_owners != 1
            || inventory.hidden_bank_owners != 2
            || inventory.prefix_arena_owners != 1
        {
            let S14StarfoldBlockResourceProduct {
                production,
                initial_hidden: _,
                mut external_owners,
            } = product;
            drop(production);
            let cleanup = external_owners.destroy(&context);
            bail!("block factory 未交付唯一 provider/双 hidden/唯一 prefix owner; cleanup={cleanup:?}");
        }
        Ok(product)
    }

    fn install_first_resources(
        &mut self,
        production: S14StarfoldK4ProductionResourceInputs<F::Provider>,
        mut external: F::ExternalOwners,
    ) -> Result<()> {
        let cleanup_context = Arc::clone(
            self.decoder
                .as_ref()
                .context("首块安装缺少 decoder context")?
                .context(),
        );
        let runtime = self.runtime.take().context("首块安装缺少唯一 runtime")?;
        let owned = match runtime.into_starfold_owned_parts() {
            Ok(owned) => owned,
            Err(error) => {
                let cleanup = external.destroy(&cleanup_context);
                return Err(anyhow!(
                    "拆分唯一 runtime 为 StarFold owned parts: {error:#}; external cleanup={cleanup:?}"
                ));
            }
        };
        match owned.build_k4(production) {
            Ok(resources) => {
                self.resources = Some(resources);
                self.current_external = Some(external);
                Ok(())
            }
            Err(error) => {
                let cleanup = external.destroy(&cleanup_context);
                Err(anyhow!(
                    "构造首块 production resources: {error:#}; external cleanup={cleanup:?}"
                ))
            }
        }
    }

    fn accumulate_generation_audit(&mut self, receipt: &S14StarfoldK4CommitReceipt) -> Result<()> {
        self.forbidden.legacy_union_calls += u64::from(receipt.legacy_union_calls);
        self.forbidden.legacy_grouped_moe_calls += u64::from(receipt.legacy_grouped_moe_calls);
        self.forbidden.serial_token_forward_calls += u64::from(receipt.serial_token_forward_calls);
        self.forbidden.cpu_fallback_calls += u64::from(receipt.cpu_fallback_calls);
        if self.forbidden != S14StarfoldForbiddenPathCounters::default() {
            bail!("production session 观察到禁止 fallback/path counter");
        }
        Ok(())
    }

    fn decoder(&self) -> Result<&S14Session> {
        self.decoder.as_ref().context("production session 已关闭")
    }

    fn cleanup_inner(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let context = self
            .resources
            .as_ref()
            .map(|resources| Arc::clone(resources.context()))
            .or_else(|| {
                self.runtime
                    .as_ref()
                    .map(|runtime| Arc::clone(runtime.context()))
            })
            .or_else(|| {
                self.decoder
                    .as_ref()
                    .map(|session| Arc::clone(session.context()))
            });
        let mut errors = Vec::new();
        if let Some(resources) = self.resources.as_mut() {
            if let Err(error) = resources.destroy_execution_owners() {
                errors.push(format!("execution owners: {error:#}"));
            }
        }
        if !errors.is_empty() {
            bail!(
                "S14 production session cleanup 保留未排空 execution owners；拒绝退休 external/factory/runtime: {}",
                errors.join("; ")
            );
        }
        if let Some(pending) = self.pending_external.take() {
            self.retired_external.push(pending);
        }
        if let Some(current) = self.current_external.take() {
            self.retired_external.push(current);
        }
        if let Some(context) = context.as_ref() {
            let mut failed = Vec::new();
            for mut owner in self.retired_external.drain(..) {
                if let Err(error) = owner.destroy(context) {
                    errors.push(format!("external owners: {error:#}"));
                    failed.push(owner);
                }
            }
            self.retired_external = failed;
        }
        if !self.retired_external.is_empty() {
            bail!(
                "S14 production session cleanup 保留 {} 组未退休 external owners；拒绝提前销毁 factory/runtime",
                self.retired_external.len()
            );
        }
        if let Some(context) = context.as_ref() {
            if let Some(factory) = self.factory.as_mut() {
                if let Err(error) = factory.retire_persistent_owners(context) {
                    errors.push(format!("factory persistent owners: {error:#}"));
                }
            }
        }
        if !errors.is_empty() {
            bail!(
                "S14 production session cleanup 保留 factory persistent owners；拒绝提前销毁 runtime: {}",
                errors.join("; ")
            );
        }
        if let Some(resources) = self.resources.as_mut() {
            if let Err(error) = resources.destroy_base_runtime_owner() {
                errors.push(format!("base runtime: {error:#}"));
            }
        }
        self.resources.take();
        if let Some(decoder) = self.decoder.take() {
            if let Err(error) = decoder.destroy() {
                errors.push(format!("decoder device state: {error:#}"));
            }
        }
        if let Some(readback) = self.prefill_readback.take() {
            if let Err(error) = readback.destroy() {
                errors.push(format!("prefill readback owner: {error:#}"));
            }
        }
        if let Some(runtime) = self.runtime.take() {
            if let Err(error) = runtime.destroy() {
                errors.push(format!("unconverted runtime: {error:#}"));
            }
        }
        self.closed = true;
        if errors.is_empty() {
            Ok(())
        } else {
            bail!("S14 production session cleanup: {}", errors.join("; "))
        }
    }
}

impl<F> Drop for S14StarfoldProductionSession<F>
where
    F: S14StarfoldBlockResourceFactory,
{
    fn drop(&mut self) {
        let _ = self.cleanup_inner();
    }
}

fn retire_rebound_external<E: S14StarfoldBlockExternalOwners>(
    current: &mut Option<E>,
    retired: &mut Vec<E>,
    context: &VulkanContext,
    external: E,
) -> Result<()> {
    if let Some(owner) = current.replace(external) {
        retired.push(owner);
    }
    let index = 0;
    while index < retired.len() {
        match retired[index].destroy(context) {
            Ok(()) => {
                retired.remove(index);
            }
            Err(error) => return Err(error.context("销毁 rebind 后已退休 external owners")),
        }
    }
    Ok(())
}

fn retire_all_external<E: S14StarfoldBlockExternalOwners>(
    retired: &mut Vec<E>,
    context: &VulkanContext,
) -> Result<()> {
    let index = 0;
    while index < retired.len() {
        match retired[index].destroy(context) {
            Ok(()) => {
                retired.remove(index);
            }
            Err(error) => {
                return Err(error.context("销毁请求结束后 external owners"));
            }
        }
    }
    Ok(())
}
