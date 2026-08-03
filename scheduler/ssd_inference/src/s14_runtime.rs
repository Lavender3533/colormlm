//! Polaris S14 production whole-token 持久运行时。
//!
//! 该 facade 从 production example 提炼资源生命周期与单 token 事务：Vulkan
//! context、分页权重 arena、层计划、输入 planner 和专家 catalog 跨 token/请求常驻；
//! session 只拥有 DecoderState 与 device candidate banks。任何 step 失败都先 drain，
//! 再销毁借用 candidate 的 GPU owner，最后 rollback，绝不发布半枚 token。

use crate::{
    compute::StorageBufferSlice,
    s14_causal_block_layer::{S14CausalBlockPublishReceipt, S14CausalBlockSealedFuture},
    s14_causal_block_resources::S14CausalBlockUnionBanks,
    s14_dynamic_page_cache_readiness::{
        materialize_dynamic_page_plan, materialize_planned_range_asset, DynamicPageFetchMode,
    },
    s14_dynamic_routed_page_plan::FullDepthExpertCatalog,
    s14_head_chunk_argmax::S14_HEAD_CHUNK_COUNT,
    s14_input_asset_plan::S14InputAssetPlanner,
    s14_position0_hybrid_upload::S14Position0HybridUploader,
    s14_position0_layer_backend::{
        S14Position0PersistentHostResources, S14Position0PersistentHostStepTelemetry,
        S14Position0SynchronousVulkanLayerAdapter,
    },
    s14_position0_layer_program::S14Position0FullDepthLayerProgram,
    s14_position0_mapped_assets::VerifiedMappedAssetStore,
    s14_position0_paged_layer_bridge::{
        S14Position0PagedLayerBridge, S14Position0PagedLayerStageReceipt,
    },
    s14_position0_paged_layer_timeline::{
        validate_production_paged_position, S14Position0PagedLayerTimeline,
        S14Position0PagedLayerTimelineState,
    },
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_state_writeback::{
        stage_payloads, S14Position0FullDepthStateRecordingProgram, S14Position0StateReadback,
    },
    s14_position0_synchronous_layer_pager::{
        S14Position0DeviceHiddenSlot, S14Position0SynchronousLayerPlan,
    },
    s14_position0_synchronous_layer_plan::build_synchronous_layer_plans,
    s14_position0_terminal::S14Position0TerminalChain,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_position0_whole_token::Position0GpuCandidate,
    s14_position0_workspace::S14Position0WorkspaceSlot,
    s14_whole_token_device::{WholeTokenDetachedCommittedState, WholeTokenDeviceState},
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{DecoderStateV1, Position0WholeTokenManifest, VOCAB_SIZE};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc, time::Instant};

const LAYER_COUNT: usize = 43;
const LAYER_TRANSFER_START: usize = 0;
const HEAD_TRANSFER_START: usize = LAYER_TRANSFER_START + LAYER_COUNT;
const LAYER_PROBE_COMPUTE_START: usize = 0;
const LAYER_CONTINUATION_COMPUTE_START: usize = LAYER_PROBE_COMPUTE_START + LAYER_COUNT;
const PRELUDE_COMPUTE: usize = LAYER_CONTINUATION_COMPUTE_START + LAYER_COUNT;
const HEAD_COMPUTE_START: usize = PRELUDE_COMPUTE + 1;
const TERMINAL_COMPUTE: usize = HEAD_COMPUTE_START + S14_HEAD_CHUNK_COUNT as usize;
const TRANSFER_COMMAND_COUNT: usize = HEAD_TRANSFER_START + S14_HEAD_CHUNK_COUNT as usize;
const COMPUTE_COMMAND_COUNT: usize = TERMINAL_COMPUTE + 1;

/// `S14Position0L0GpuOwner` 同时借用 paged arena buffer，并创建绑定当前 session
/// candidate bank 的 descriptor binder。Rust不能让 `S14Runtime` 安全持有这个对自身
/// arena 的借用，因此该 owner 仍必须每 step 重建，直到改成handle/offset所有权模型。
pub const S14_CANDIDATE_SCOPED_BACKEND_REBUILT_EACH_STEP: bool = true;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14RuntimeCommandStepTelemetry {
    pub step_index: u64,
    pub reused_from_previous_step: bool,
    pub resource_allocations_this_step: u64,
    pub command_pool_resets_this_step: u64,
}

pub struct S14RuntimeCommandStep {
    pub transfer_commands: [vk::CommandBuffer; TRANSFER_COMMAND_COUNT],
    pub compute_commands: [vk::CommandBuffer; COMPUTE_COMMAND_COUNT],
    pub telemetry: S14RuntimeCommandStepTelemetry,
}

fn validate_persistent_step_contract(
    host: S14Position0PersistentHostStepTelemetry,
    command: S14RuntimeCommandStepTelemetry,
) -> Result<u64> {
    if host.step_index != command.step_index
        || host.reused_from_previous_step != command.reused_from_previous_step
        || host.reused_from_previous_step != (host.step_index != 0)
    {
        bail!("persistent host/command step identity 漂移");
    }
    if host.resource_allocations_this_step != 0
        || host.uploader_staging_allocations_this_step != 0
        || host.mapped_store_allocations_this_step != 0
        || host.graph_plan_allocations_this_step != 0
        || host.backend_command_allocations_this_step != 0
        || command.resource_allocations_this_step != 0
        || command.command_pool_resets_this_step != 2
    {
        bail!("persistent whole-token resource allocation/reset contract 漂移");
    }
    host.resource_allocations_this_step
        .checked_add(command.resource_allocations_this_step)
        .ok_or_else(|| anyhow!("persistent resource allocation counter overflow"))
}

/// Runtime级transfer/compute command pool与全部command buffer。第二个及后续
/// token只reset两个pool；不create pool、不allocate command buffer。
pub struct S14RuntimePersistentCommandResources {
    transfer_pool: vk::CommandPool,
    compute_pool: vk::CommandPool,
    transfer_commands: [vk::CommandBuffer; TRANSFER_COMMAND_COUNT],
    compute_commands: [vk::CommandBuffer; COMPUTE_COMMAND_COUNT],
    steps_started: u64,
    active_step: bool,
}

impl S14RuntimePersistentCommandResources {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        let (transfer_pool, compute_pool, transfer, compute) = allocate_commands(ctx)?;
        let transfer_commands: [vk::CommandBuffer; TRANSFER_COMMAND_COUNT] = match transfer
            .try_into()
        {
            Ok(commands) => commands,
            Err(commands) => {
                unsafe {
                    ctx.device.destroy_command_pool(compute_pool, None);
                    ctx.device.destroy_command_pool(transfer_pool, None);
                }
                bail!(
                    "persistent transfer command count漂移: actual={} expected={TRANSFER_COMMAND_COUNT}",
                    commands.len()
                );
            }
        };
        let compute_commands: [vk::CommandBuffer; COMPUTE_COMMAND_COUNT] = match compute.try_into()
        {
            Ok(commands) => commands,
            Err(commands) => {
                unsafe {
                    ctx.device.destroy_command_pool(compute_pool, None);
                    ctx.device.destroy_command_pool(transfer_pool, None);
                }
                bail!(
                    "persistent compute command count漂移: actual={} expected={COMPUTE_COMMAND_COUNT}",
                    commands.len()
                );
            }
        };
        Ok(Self {
            transfer_pool,
            compute_pool,
            transfer_commands,
            compute_commands,
            steps_started: 0,
            active_step: false,
        })
    }

    pub fn begin_step(&mut self, ctx: &VulkanContext) -> Result<S14RuntimeCommandStep> {
        if self.active_step {
            bail!("persistent command resources已有active step");
        }
        unsafe {
            ctx.device
                .reset_command_pool(self.transfer_pool, vk::CommandPoolResetFlags::empty())?;
            ctx.device
                .reset_command_pool(self.compute_pool, vk::CommandPoolResetFlags::empty())?;
        }
        let telemetry = S14RuntimeCommandStepTelemetry {
            step_index: self.steps_started,
            reused_from_previous_step: self.steps_started != 0,
            resource_allocations_this_step: 0,
            command_pool_resets_this_step: 2,
        };
        self.steps_started = self
            .steps_started
            .checked_add(1)
            .ok_or_else(|| anyhow!("persistent command step counter overflow"))?;
        self.active_step = true;
        Ok(S14RuntimeCommandStep {
            transfer_commands: self.transfer_commands,
            compute_commands: self.compute_commands,
            telemetry,
        })
    }

    pub fn finish_step(&mut self) -> Result<()> {
        if !self.active_step {
            bail!("persistent command resources没有active step");
        }
        self.active_step = false;
        Ok(())
    }

    pub fn abort_after_drain(&mut self) -> Result<()> {
        if !self.active_step {
            bail!("persistent command resources没有可回滚active step");
        }
        self.steps_started = self
            .steps_started
            .checked_sub(1)
            .ok_or_else(|| anyhow!("persistent command step counter underflow"))?;
        self.active_step = false;
        Ok(())
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_command_pool(self.compute_pool, None);
            ctx.device.destroy_command_pool(self.transfer_pool, None);
        }
    }
}

/// S14 runtime 的显式资产与 I/O 授权边界。
#[derive(Clone, Debug)]
pub struct S14RuntimeConfig {
    pub manifest_path: PathBuf,
    pub payload_root: PathBuf,
    pub catalog_path: PathBuf,
    pub paged_weight_arena_limit_bytes: Option<u64>,
    pub page_fetch_mode: DynamicPageFetchMode,
}

impl S14RuntimeConfig {
    /// 与 production example 相同的固定资产位置；默认严格 local-only。
    pub fn production_defaults() -> Self {
        let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            manifest_path: workspace_root.join(
                "fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
            ),
            payload_root: PathBuf::from("D:/models/Polaris-S14/range_cache"),
            catalog_path: PathBuf::from(
                "D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json",
            ),
            paged_weight_arena_limit_bytes: Some(3 * 1024 * 1024 * 1024),
            page_fetch_mode: DynamicPageFetchMode::LocalOnly,
        }
    }

    pub fn with_explicit_page_fetch(mut self, enabled: bool) -> Self {
        self.page_fetch_mode = if enabled {
            DynamicPageFetchMode::ExplicitFetch
        } else {
            DynamicPageFetchMode::LocalOnly
        };
        self
    }
}

/// 跨 token 常驻的昂贵 production 资源。
pub struct S14Runtime {
    ctx: Arc<VulkanContext>,
    manifest: Position0WholeTokenManifest,
    weights: S14Position0HybridWeightPlan,
    arena: Option<S14Position0PagedWeightArena>,
    causal_block_union_banks: Option<S14CausalBlockUnionBanks>,
    layer_program: S14Position0FullDepthLayerProgram,
    plans: Vec<S14Position0SynchronousLayerPlan>,
    input_planner: S14InputAssetPlanner,
    expert_catalog: FullDepthExpertCatalog,
    payload_root: PathBuf,
    page_fetch_mode: DynamicPageFetchMode,
    persistent_host_resources: Option<S14Position0PersistentHostResources>,
    persistent_command_resources: Option<S14RuntimePersistentCommandResources>,
}

/// 每个聊天请求独占的连续 decoder/device 状态。
pub struct S14Session {
    ctx: Arc<VulkanContext>,
    host: DecoderStateV1,
    device: Option<WholeTokenDeviceState>,
}

/// position0 `0→5`完成后移交给K=4 production builder的同源资源。
/// 这里只做owner转移，不创建prefix/hidden/terminal候选，也不读取host checkpoint。
pub struct S14Position0CommittedRuntimeParts {
    pub context: Arc<VulkanContext>,
    pub manifest: Position0WholeTokenManifest,
    pub weights: S14Position0HybridWeightPlan,
    pub paged_arena: Arc<S14Position0PagedWeightArena>,
    pub union_banks: S14CausalBlockUnionBanks,
    pub layer_program: S14Position0FullDepthLayerProgram,
    pub input_planner: S14InputAssetPlanner,
    pub payload_root: PathBuf,
    pub page_fetch_mode: DynamicPageFetchMode,
    pub uploader: S14Position0HybridUploader,
    pub mapped_store: VerifiedMappedAssetStore,
    pub authoritative: DecoderStateV1,
    pub committed_device: WholeTokenDetachedCommittedState,
    pub continuation: S14CausalBlockRuntimeContinuation,
}

/// K-block producer取得paged/union/head owner后仍保留的权威runtime/session壳。
/// 它只负责对sealed future调用既有两阶段发布路径，并从发布后的真实committed bank
/// 派生下一block起点；不重建position0，也不接受host fixture替换device checkpoint。
pub struct S14CausalBlockRuntimeContinuation {
    runtime: S14Runtime,
    session: S14Session,
}

/// 下一K=4/8 block只能从最近一次真实发布后的host/device同源状态构造。
pub struct S14NextCausalBlockCommittedState {
    pub authoritative: DecoderStateV1,
    pub committed_device: WholeTokenDetachedCommittedState,
    pub base_position: u32,
    pub input_token_id: u32,
    pub commit_epoch: u64,
}

#[derive(Clone, Debug)]
pub struct S14StepOutput {
    pub position: u32,
    pub input_token_id: u32,
    pub predicted_token_id: u32,
    pub max_logit: f32,
    pub commit_epoch: u64,
    pub active_bank: usize,
    pub l42_hidden_sha256: String,
    pub elapsed_ms: f64,
    pub online_top6_routes: u64,
    pub dynamic_physical_ranges: u64,
    pub static_uploaded_bytes: u64,
    pub routed_uploaded_bytes: u64,
    pub head_uploaded_bytes: u64,
    pub runtime_host_api_waits: u64,
    pub persistent_step_index: u64,
    pub persistent_resource_allocations_this_step: u64,
    pub persistent_command_pool_resets_this_step: u64,
    pub persistent_resources_reused_from_previous_step: bool,
    /// 当前精确阻塞：descriptor/pipeline owner仍借用arena buffer并绑定session candidate。
    pub candidate_scoped_backend_rebuilt_this_step: bool,
}

impl S14Runtime {
    pub fn load(config: S14RuntimeConfig) -> Result<Self> {
        let manifest =
            Position0WholeTokenManifest::load(&config.manifest_path).with_context(|| {
                format!("加载 S14 manifest 失败: {}", config.manifest_path.display())
            })?;
        let weights = S14Position0HybridWeightPlan::build(&manifest)?;
        let ctx = Arc::new(VulkanContext::init()?);
        if !ctx.timeline_semaphore || !ctx.has_dedicated_transfer() {
            bail!("S14 production runtime requires timeline semaphore and dedicated transfer");
        }
        // Union banks 是 K4/K8 block-major forward 的必要资源，必须在可选静态缓存
        // 贪心分配之前物理预留；否则单 token 缓存会吞掉 block runtime 的硬容量。
        let causal_block_union_banks = S14CausalBlockUnionBanks::new(&ctx)
            .map_err(anyhow::Error::new)
            .context("初始化 S14 K4/K8 causal-block union banks")?;
        let arena = match S14Position0PagedWeightArena::new(
            &ctx,
            &weights,
            config.paged_weight_arena_limit_bytes,
        ) {
            Ok(arena) => arena,
            Err(error) => {
                causal_block_union_banks.destroy(&ctx);
                return Err(error.context("初始化 S14 paged weight arena"));
            }
        };
        let setup = (|| -> Result<_> {
            let layer_program = S14Position0FullDepthLayerProgram::build(
                &manifest,
                &weights,
                arena.workspace_layout(),
            )?;
            let plans = build_synchronous_layer_plans(&layer_program, &arena)?;
            if plans.len() != LAYER_COUNT {
                bail!("S14 production layer plan count 漂移: {}", plans.len());
            }
            let input_planner =
                S14InputAssetPlanner::load_pinned(&config.catalog_path, &config.payload_root)?;
            let expert_catalog = FullDepthExpertCatalog::load(&config.catalog_path)?;
            Ok((layer_program, plans, input_planner, expert_catalog))
        })();
        let (layer_program, plans, input_planner, expert_catalog) = match setup {
            Ok(loaded) => loaded,
            Err(error) => {
                arena.destroy(&ctx);
                causal_block_union_banks.destroy(&ctx);
                return Err(error);
            }
        };
        let persistent_command_resources = match S14RuntimePersistentCommandResources::new(&ctx) {
            Ok(resources) => resources,
            Err(error) => {
                arena.destroy(&ctx);
                causal_block_union_banks.destroy(&ctx);
                return Err(error.context("初始化S14 persistent command resources"));
            }
        };
        let persistent_host_resources = match S14Position0PersistentHostResources::new(
            &ctx,
            &manifest,
            &weights,
            &arena,
            &config.payload_root,
        ) {
            Ok(resources) => resources,
            Err(error) => {
                persistent_command_resources.destroy(&ctx);
                arena.destroy(&ctx);
                causal_block_union_banks.destroy(&ctx);
                return Err(error.context("初始化S14 persistent uploader/store/graph resources"));
            }
        };
        Ok(Self {
            ctx,
            manifest,
            weights,
            arena: Some(arena),
            causal_block_union_banks: Some(causal_block_union_banks),
            layer_program,
            plans,
            input_planner,
            expert_catalog,
            payload_root: config.payload_root,
            page_fetch_mode: config.page_fetch_mode,
            persistent_host_resources: Some(persistent_host_resources),
            persistent_command_resources: Some(persistent_command_resources),
        })
    }

    pub fn new_session(&self, first_token_id: u32, max_seq_len: u32) -> Result<S14Session> {
        let host = DecoderStateV1::new(max_seq_len, first_token_id)?;
        let device = WholeTokenDeviceState::new(&self.ctx, host.native_arena.bytes(), 0)?;
        Ok(S14Session {
            ctx: Arc::clone(&self.ctx),
            host,
            device: Some(device),
        })
    }

    pub(crate) fn causal_block_union_banks(&self) -> Option<&S14CausalBlockUnionBanks> {
        self.causal_block_union_banks.as_ref()
    }

    pub(crate) fn expert_catalog(&self) -> &FullDepthExpertCatalog {
        &self.expert_catalog
    }

    pub(crate) fn owns_session(&self, session: &S14Session) -> bool {
        Arc::ptr_eq(&self.ctx, &session.ctx)
    }

    pub(crate) fn authoritative_state<'a>(
        &self,
        session: &'a S14Session,
    ) -> Option<&'a DecoderStateV1> {
        self.owns_session(session).then_some(&session.host)
    }

    pub(crate) fn session_host_device_mut<'a>(
        &self,
        session: &'a mut S14Session,
    ) -> Option<(
        &'a VulkanContext,
        &'a mut DecoderStateV1,
        &'a mut WholeTokenDeviceState,
    )> {
        if !self.owns_session(session) {
            return None;
        }
        let ctx = session.ctx.as_ref();
        let host = &mut session.host;
        let device = session.device.as_mut()?;
        Some((ctx, host, device))
    }

    pub fn step(&mut self, session: &mut S14Session) -> Result<S14StepOutput> {
        self.step_with_next_input(session, None)
    }

    /// 只允许在真实position0成功提交`0→5`后消费runtime/session。`expected_next_input`
    /// 是本次请求在position1实际要消费的token；它可以由prompt prefill覆盖模型的position0
    /// 预测，但ledger仍必须保留真实`0→5`，不能伪造position0输出。昂贵paged arena、union
    /// bank、verified uploader/store与active committed device bank保持同一Arc context。
    pub fn into_position0_committed_causal_block_parts(
        mut self,
        mut session: S14Session,
        expected_next_input: u32,
    ) -> Result<S14Position0CommittedRuntimeParts> {
        if expected_next_input >= VOCAB_SIZE {
            bail!("position0 committed exporter next input越界: {expected_next_input}");
        }
        if !self.owns_session(&session) {
            bail!("position0 committed exporter拒绝外来session");
        }
        session.host.validate()?;
        let record = session.host.committed_tokens.last();
        if session.host.position != 1
            || session.host.commit_epoch != 1
            || session.host.input_token_id != expected_next_input
            || session.host.active_fixed_bank != 1
            || session.host.committed_tokens.len() != 1
            || record.is_none_or(|value| {
                value.position != 0 || value.input_token_id != 0 || value.predicted_token_id != 5
            })
        {
            bail!(
                "causal-block exporter必须消费真实position0 0→5 committed state并绑定请求next input={expected_next_input}"
            );
        }
        let device = session
            .device
            .as_ref()
            .context("position0 committed exporter缺少device state")?;
        if device.epoch() != session.host.commit_epoch
            || device.active_bank() != usize::from(session.host.active_fixed_bank)
            || device.state_bytes() != session.host.native_arena.len() as u64
            || device.candidate_position().is_some()
        {
            bail!("position0 committed exporter host/device identity漂移");
        }

        let command_resources = self
            .persistent_command_resources
            .take()
            .context("position0 committed exporter缺少command owner")?;
        command_resources.destroy(&self.ctx);
        let host_resources = self
            .persistent_host_resources
            .take()
            .context("position0 committed exporter缺少verified host owner")?;
        let (uploader, mapped_store) =
            host_resources.into_causal_block_upload_parts(&self.ctx, &self.weights)?;
        let paged_arena = Arc::new(
            self.arena
                .take()
                .context("position0 committed exporter缺少paged arena")?,
        );
        let union_banks = self
            .causal_block_union_banks
            .take()
            .context("position0 committed exporter缺少union banks")?;
        let committed_device = match session
            .device
            .as_mut()
            .context("position0 committed exporter缺少device state")?
            .snapshot_detached_committed_state(&self.ctx)
        {
            Ok(value) => value,
            Err(error) => {
                uploader.destroy(&self.ctx);
                Arc::try_unwrap(paged_arena)
                    .map_err(|_| anyhow!("position0 exporter rollback paged arena Arc漂移"))?
                    .destroy(&self.ctx);
                union_banks.destroy(&self.ctx);
                return Err(error.context("snapshot position0 committed device bank"));
            }
        };
        let context = Arc::clone(&self.ctx);
        let manifest = self.manifest.clone();
        let weights = self.weights.clone();
        let layer_program = self.layer_program.clone();
        let input_planner = self.input_planner.clone();
        let payload_root = self.payload_root.clone();
        let page_fetch_mode = self.page_fetch_mode;
        let authoritative = session.host.clone();
        let continuation = S14CausalBlockRuntimeContinuation {
            runtime: self,
            session,
        };
        Ok(S14Position0CommittedRuntimeParts {
            context,
            manifest,
            weights,
            paged_arena,
            union_banks,
            layer_program,
            input_planner,
            payload_root,
            page_fetch_mode,
            uploader,
            mapped_store,
            authoritative,
            committed_device,
            continuation,
        })
    }

    /// 执行真实模型 step，但允许 prompt prefill 在原子提交时把下一输入替换为
    /// 下一枚真实 prompt token。预测 token 仍原样写入 ledger，不伪造模型输出。
    pub fn step_with_next_input(
        &mut self,
        session: &mut S14Session,
        forced_next_input: Option<u32>,
    ) -> Result<S14StepOutput> {
        if !Arc::ptr_eq(&self.ctx, &session.ctx) {
            bail!("S14 session 不属于当前 runtime");
        }
        if let Some(token_id) = forced_next_input {
            if token_id >= VOCAB_SIZE {
                bail!("forced next input token 越界: {token_id}");
            }
        }
        let arena = self
            .arena
            .as_ref()
            .ok_or_else(|| anyhow!("S14 runtime 已销毁"))?;
        let host = &mut session.host;
        let device = session
            .device
            .as_mut()
            .ok_or_else(|| anyhow!("S14 session 已销毁"))?;
        let position = host.position;
        validate_production_paged_position(position)?;
        if device.epoch() != host.commit_epoch
            || device.active_bank() != usize::from(host.active_fixed_bank)
        {
            bail!("S14 host/device position/epoch/bank 漂移");
        }
        let base_epoch = host.commit_epoch;
        let input_token_id = host.input_token_id;
        let input_plan =
            self.input_planner
                .plan(position, input_token_id, host.native.max_seq_len)?;
        let input_embedding = materialize_planned_range_asset(
            &input_plan.embedding,
            &self.payload_root,
            self.page_fetch_mode,
        )?;
        let state_recording = S14Position0FullDepthStateRecordingProgram::build(
            &self.layer_program,
            arena.workspace_layout(),
            &host.native,
        )?;

        let mut timeline_slot = None;
        let mut terminal_slot = None;
        let mut state_readback_slot = None;
        let mut command_step_slot = None;
        let mut command_step_active = false;
        let prologue_command =
            device.begin_candidate_for_position(&self.ctx, base_epoch, position)?;
        let mut rollback_guard = CandidateRollbackGuard::new(device, &self.ctx);
        device.arm_external_candidate()?;
        rollback_guard.mark_external_armed();
        let candidate_bank = 1usize - device.active_bank();
        let candidate = Position0GpuCandidate {
            ctx: &self.ctx,
            candidate_state: device.candidate_buffer()?,
            sticky_status: device.sticky_status_buffer()?,
            committed_host_state: host,
            base_epoch,
            candidate_bank,
        };

        let started = Instant::now();
        let persistent_host_resources = self
            .persistent_host_resources
            .as_mut()
            .ok_or_else(|| anyhow!("S14 persistent host resources已销毁"))?;
        let backend = S14Position0SynchronousVulkanLayerAdapter::new(
            &self.ctx,
            &self.manifest,
            &self.weights,
            arena,
            &self.payload_root,
            persistent_host_resources,
            candidate,
        )?;
        let persistent_host_step = backend.persistent_step_telemetry();
        let mut backend_slot = Some(backend);
        let mut completion_slot = None;
        let mut payloads_slot = None;
        let mut l42_hidden_sha_slot = None;
        let mut static_uploaded_bytes = 0u64;
        let mut routed_uploaded_bytes = 0u64;
        let mut router_probe_readback_bytes = 0u64;
        let mut router_probe_host_waits = 0u64;
        let mut streamed_static_transfer_fence_waits = 0u64;
        let dynamic_routed_transfer_fence_waits = 0u64;
        let mut dynamic_physical_ranges = 0u64;
        let mut head_uploaded_bytes = 0u64;

        let token_result = (|| -> Result<()> {
            let backend = backend_slot.as_mut().expect("candidate backend slot");
            backend.bind_input_prologue(&input_plan, &input_embedding)?;
            let input_execution = backend
                .input_execution()
                .ok_or_else(|| anyhow!("S14 input execution plan 未绑定"))?;
            if input_execution.rope_position != position
                || input_execution.window_slot != position % 128
                || input_execution.active_window_tokens != (position + 1).min(128)
            {
                bail!("S14 prologue position/RoPE/window 合同漂移");
            }
            let command_step = self
                .persistent_command_resources
                .as_mut()
                .ok_or_else(|| anyhow!("S14 persistent command resources已销毁"))?
                .begin_step(&self.ctx)?;
            command_step_slot = Some(command_step.telemetry);
            command_step_active = true;
            let transfer_commands = command_step.transfer_commands;
            let compute_commands = command_step.compute_commands;
            timeline_slot = Some(S14Position0PagedLayerTimeline::new_for_position(
                &self.ctx, position,
            )?);
            let timeline = timeline_slot.as_mut().expect("candidate timeline slot");

            unsafe {
                backend.record_embedding_prologue(prologue_command, &input_embedding)?;
                timeline.submit_prologue_compute_only(&self.ctx, prologue_command)?;
            }

            let mut bridge =
                S14Position0PagedLayerBridge::new_for_position(timeline, &self.plans, position)?;
            for (index, plan) in self.plans.iter().enumerate() {
                let transfer = transfer_commands[LAYER_TRANSFER_START + index];
                let probe_compute = compute_commands[LAYER_PROBE_COMPUTE_START + index];
                let continuation_compute =
                    compute_commands[LAYER_CONTINUATION_COMPUTE_START + index];

                let static_receipt = backend.prepare_router_probe_static(plan)?;
                static_uploaded_bytes = static_uploaded_bytes
                    .checked_add(static_receipt.bytes)
                    .ok_or_else(|| anyhow!("static upload byte counter overflow"))?;
                if static_receipt.bytes != 0 {
                    streamed_static_transfer_fence_waits += 1;
                }
                unsafe { backend.record_paged_layer_router_probe(plan, probe_compute)? };
                let probe_completion = unsafe {
                    bridge.timeline_mut().submit_router_probe_and_wait(
                        &self.ctx,
                        plan.layer,
                        probe_compute,
                    )?
                };
                if probe_completion.layer != plan.layer
                    || probe_completion.index != index
                    || probe_completion.bank != plan.routed_bank
                    || probe_completion.host_wait_calls != 1
                {
                    bail!("L{} router probe timeline 强回执漂移", plan.layer);
                }
                router_probe_host_waits += u64::from(probe_completion.host_wait_calls);
                let observed = backend.complete_router_probe_after_wait(probe_completion.layer)?;
                router_probe_readback_bytes += observed.readback_bytes;
                let observed_expert_ids = observed.route.expert_ids;
                let dynamic_plan = self.expert_catalog.plan(observed.route)?;
                let materialized = materialize_dynamic_page_plan(
                    &dynamic_plan,
                    &self.payload_root,
                    self.page_fetch_mode,
                )
                .map_err(anyhow::Error::new)
                .with_context(|| {
                    format!(
                        "position{position} L{} dynamic top-6 Range materialize 失败: experts={observed_expert_ids:?}",
                        plan.layer
                    )
                })?;
                dynamic_physical_ranges += u64::try_from(materialized.assets.len())?;
                let routed_receipt = unsafe {
                    backend.record_dynamic_routed_after_probe(
                        plan,
                        &dynamic_plan,
                        &materialized,
                        transfer,
                    )?
                };
                routed_uploaded_bytes += routed_receipt.bytes;

                unsafe {
                    backend.record_paged_layer_dynamic_moe_continuation(
                        plan,
                        &dynamic_plan,
                        &materialized,
                        continuation_compute,
                    )?;
                }
                backend.validate_recorded_layer_binding(plan, plan.routed_bank)?;
                let expected_layer = plan.layer;
                let expected_bank = plan.routed_bank;
                let receipt = unsafe {
                    bridge.submit_next_layer(
                        &self.ctx,
                        transfer,
                        continuation_compute,
                        |_runtime_plan, timeline_bank| {
                            if timeline_bank != expected_bank {
                                bail!("L{expected_layer} dynamic routed timeline bank 漂移");
                            }
                            Ok(S14Position0PagedLayerStageReceipt {
                                layer: expected_layer,
                                bank: expected_bank,
                                static_uploaded_bytes: static_receipt.bytes,
                                routed_uploaded_bytes: routed_receipt.bytes,
                                hidden_host_bytes: 0,
                            })
                        },
                        |runtime_plan, timeline_bank| {
                            if runtime_plan.layer != expected_layer
                                || timeline_bank != expected_bank
                            {
                                bail!("L{expected_layer} recorded descriptor/bank 漂移");
                            }
                            Ok(())
                        },
                    )?
                };
                if receipt.layer != expected_layer
                    || receipt.index != index
                    || receipt.bank != expected_bank
                    || receipt.stage.static_uploaded_bytes != static_receipt.bytes
                    || receipt.stage.routed_uploaded_bytes != routed_receipt.bytes
                {
                    bail!("L{expected_layer} dynamic routed bridge receipt 漂移");
                }
            }

            let mut tail = bridge.seal_layers()?;
            if tail.final_hidden() != S14Position0DeviceHiddenSlot::B {
                bail!("FullDepth43 L42 final hidden slot 漂移");
            }
            let l42_region = arena
                .workspace_layout()
                .region(S14Position0WorkspaceSlot::HiddenStreamsB);
            let l42_hidden = StorageBufferSlice {
                buffer: arena.workspace(),
                offset: l42_region.offset,
            };
            let candidate_hc = StorageBufferSlice {
                buffer: device.candidate_buffer()?,
                offset: host.native.hc.streams.offset,
            };
            state_readback_slot = Some(S14Position0StateReadback::new(&self.ctx, &host.native)?);
            terminal_slot = Some(S14Position0TerminalChain::new(
                &self.ctx, arena, l42_hidden,
            )?);
            let state_readback = state_readback_slot
                .as_mut()
                .expect("candidate state readback slot");
            let terminal = terminal_slot.as_mut().expect("candidate terminal slot");

            begin_graphics_command(&self.ctx, compute_commands[PRELUDE_COMPUTE])?;
            unsafe {
                terminal.record_prelude(&self.ctx, compute_commands[PRELUDE_COMPUTE])?;
                self.ctx
                    .device
                    .end_command_buffer(compute_commands[PRELUDE_COMPUTE])?;
                terminal.submit_recorded_prelude(
                    &self.ctx,
                    tail.timeline_mut(),
                    compute_commands[PRELUDE_COMPUTE],
                )?;
            }

            for chunk in 0..S14_HEAD_CHUNK_COUNT {
                let index = chunk as usize;
                let transfer = transfer_commands[HEAD_TRANSFER_START + index];
                let compute = compute_commands[HEAD_COMPUTE_START + index];
                let recorded = unsafe { backend.record_next_head_transfer(transfer)? };
                begin_graphics_command(&self.ctx, compute)?;
                unsafe {
                    terminal.record_head_chunk(&self.ctx, chunk, compute)?;
                    self.ctx.device.end_command_buffer(compute)?;
                    terminal.submit_recorded_head(
                        &self.ctx,
                        tail.timeline_mut(),
                        chunk,
                        transfer,
                        compute,
                        |timeline_bank| {
                            let staged = backend.stage_recorded_head(recorded, timeline_bank)?;
                            if staged.chunk != u64::from(chunk)
                                || staged.bank != timeline_bank
                                || staged.bytes != recorded.bytes
                            {
                                bail!("head chunk {chunk} staged receipt 漂移");
                            }
                            Ok(())
                        },
                    )?;
                }
                head_uploaded_bytes += recorded.bytes;
            }

            begin_graphics_command(&self.ctx, compute_commands[TERMINAL_COMPUTE])?;
            unsafe {
                terminal.record_terminal_commit_readback(
                    &self.ctx,
                    compute_commands[TERMINAL_COMPUTE],
                    candidate_hc,
                    state_readback,
                )?;
                self.ctx
                    .device
                    .end_command_buffer(compute_commands[TERMINAL_COMPUTE])?;
                terminal.submit_terminal(
                    &self.ctx,
                    tail.timeline_mut(),
                    compute_commands[TERMINAL_COMPUTE],
                )?;
            }
            let completion = terminal.finish_candidate(
                &self.ctx,
                tail.timeline_mut(),
                base_epoch,
                candidate_bank,
            )?;
            if completion.timeline.token_host_waits != 1
                || completion.timeline.layers != LAYER_COUNT
                || completion.timeline.head_chunks != u64::from(S14_HEAD_CHUNK_COUNT)
                || router_probe_host_waits != LAYER_COUNT as u64
                || router_probe_readback_bytes != LAYER_COUNT as u64 * 48
                || dynamic_physical_ranges != LAYER_COUNT as u64 * 36
            {
                bail!("production paged timeline completion contract 漂移");
            }
            let l42_hidden_sha = format!(
                "{:x}",
                Sha256::digest(bytemuck::cast_slice::<u16, u8>(&completion.hc_streams_bf16))
            );
            let payloads = state_readback.snapshot()?;
            drop(tail);
            terminal_slot
                .take()
                .expect("candidate terminal owner")
                .destroy(&self.ctx);
            state_readback_slot
                .take()
                .expect("candidate state readback owner")
                .destroy(&self.ctx);
            let finished_host_step = backend_slot
                .take()
                .expect("candidate backend owner")
                .finish_after_external_timeline_drained()?;
            if finished_host_step != persistent_host_step {
                bail!("persistent host step telemetry 漂移");
            }
            timeline_slot
                .take()
                .expect("candidate timeline owner")
                .destroy(&self.ctx);
            self.persistent_command_resources
                .as_mut()
                .ok_or_else(|| anyhow!("S14 persistent command resources已销毁"))?
                .finish_step()?;
            command_step_active = false;
            completion_slot = Some(completion);
            payloads_slot = Some(payloads);
            l42_hidden_sha_slot = Some(l42_hidden_sha);
            Ok(())
        })();

        if let Err(error) = token_result {
            let mut failure = error;
            let mut timeline_drained = match timeline_slot.as_ref() {
                Some(timeline) => matches!(
                    timeline.stats().state,
                    S14Position0PagedLayerTimelineState::Finished
                        | S14Position0PagedLayerTimelineState::Drained
                ),
                None => true,
            };
            if !timeline_drained {
                if let Some(timeline) = timeline_slot.as_mut() {
                    match timeline.drain_all(&self.ctx) {
                        Ok(_) => timeline_drained = true,
                        Err(cleanup) => {
                            failure = anyhow!(
                                "{failure:#}; candidate timeline drain 同时失败: {cleanup:#}"
                            );
                        }
                    }
                }
            }
            if timeline_drained {
                if let Some(terminal) = terminal_slot.take() {
                    terminal.destroy(&self.ctx);
                }
                if let Some(state_readback) = state_readback_slot.take() {
                    state_readback.destroy(&self.ctx);
                }
                if let Some(backend) = backend_slot.take() {
                    if let Err(cleanup) = backend.abort_after_external_timeline_drained() {
                        failure = anyhow!(
                            "{failure:#}; persistent host token abort 同时失败: {cleanup:#}"
                        );
                    }
                }
            } else {
                if let Some(backend) = backend_slot.take() {
                    if let Err(cleanup) = backend.abort() {
                        failure = anyhow!(
                            "{failure:#}; candidate backend device-idle abort 同时失败: {cleanup:#}"
                        );
                    }
                }
                if let Some(terminal) = terminal_slot.take() {
                    terminal.destroy(&self.ctx);
                }
                if let Some(state_readback) = state_readback_slot.take() {
                    state_readback.destroy(&self.ctx);
                }
            }
            if let Some(timeline) = timeline_slot.take() {
                timeline.destroy(&self.ctx);
            }
            if command_step_active {
                match self.persistent_command_resources.as_mut() {
                    Some(resources) => {
                        if let Err(cleanup) = resources.abort_after_drain() {
                            failure = anyhow!(
                                "{failure:#}; persistent command step abort 同时失败: {cleanup:#}"
                            );
                        }
                    }
                    None => {
                        failure = anyhow!(
                            "{failure:#}; persistent command step abort 同时失败: resources已销毁"
                        );
                    }
                }
            }
            drop(backend_slot);
            if let Err(cleanup) = rollback_guard.rollback_now() {
                failure = anyhow!("{failure:#}; candidate rollback 同时失败: {cleanup:#}");
            }
            return Err(failure);
        }

        drop(backend_slot);
        drop(terminal_slot);
        drop(state_readback_slot);
        drop(timeline_slot);
        let command_step = command_step_slot
            .take()
            .ok_or_else(|| anyhow!("candidate success 缺少 persistent command telemetry"))?;
        let persistent_resource_allocations_this_step =
            validate_persistent_step_contract(persistent_host_step, command_step)?;
        let completion = completion_slot
            .take()
            .ok_or_else(|| anyhow!("candidate success 缺少 terminal completion"))?;
        let payloads = payloads_slot
            .take()
            .ok_or_else(|| anyhow!("candidate success 缺少 state payloads"))?;
        let l42_hidden_sha256 = l42_hidden_sha_slot
            .take()
            .ok_or_else(|| anyhow!("candidate success 缺少 L42 SHA"))?;

        let mut host_candidate = host.begin_token(base_epoch, position, input_token_id)?;
        stage_payloads(&mut host_candidate, &payloads)?;
        host_candidate.stage_position0_hc_state(&completion.hc_streams_bf16)?;
        host_candidate.complete_final(completion.predicted_token_id)?;
        let mut next_host = host.clone();
        let token_record =
            host_candidate.commit_with_next_input(&mut next_host, forced_next_input)?;
        next_host.validate()?;
        let expected_epoch = base_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow!("continuous epoch overflow"))?;
        if next_host.commit_epoch != expected_epoch
            || next_host.position != position + 1
            || next_host.input_token_id
                != forced_next_input.unwrap_or(completion.predicted_token_id)
            || usize::from(next_host.active_fixed_bank) != candidate_bank
            || token_record.predicted_token_id != completion.predicted_token_id
        {
            bail!("S14 production host candidate 提交前合同漂移");
        }
        for range in state_recording.merged_device_dirty_write_set(&host.native)? {
            device.mark_candidate_dirty(range.start, range.end - range.start)?;
        }
        device.finish_external_candidate(base_epoch, candidate_bank)?;
        let prepared = device.prepare_candidate_commit_for_position(base_epoch, position)?;
        let device_receipt = device.publish_prepared_commit(prepared);
        *host = next_host;
        rollback_guard.disarm();
        if host.commit_epoch != expected_epoch
            || device_receipt.epoch != expected_epoch
            || device_receipt.active_bank != candidate_bank
            || usize::from(host.active_fixed_bank) != device_receipt.active_bank
            || device.active_bank() != device_receipt.active_bank
        {
            bail!("S14 production host/device 原子提交回执漂移");
        }

        let runtime_host_api_waits = completion.timeline.producer_transfer_waits
            + completion.timeline.token_host_waits
            + router_probe_host_waits
            + streamed_static_transfer_fence_waits
            + dynamic_routed_transfer_fence_waits;
        Ok(S14StepOutput {
            position,
            input_token_id,
            predicted_token_id: completion.predicted_token_id,
            max_logit: completion.max_logit,
            commit_epoch: host.commit_epoch,
            active_bank: device_receipt.active_bank,
            l42_hidden_sha256,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            online_top6_routes: router_probe_host_waits,
            dynamic_physical_ranges,
            static_uploaded_bytes,
            routed_uploaded_bytes,
            head_uploaded_bytes,
            runtime_host_api_waits,
            persistent_step_index: persistent_host_step.step_index,
            persistent_resource_allocations_this_step,
            persistent_command_pool_resets_this_step: command_step.command_pool_resets_this_step,
            persistent_resources_reused_from_previous_step: persistent_host_step
                .reused_from_previous_step,
            candidate_scoped_backend_rebuilt_this_step:
                S14_CANDIDATE_SCOPED_BACKEND_REBUILT_EACH_STEP,
        })
    }

    pub fn destroy(mut self) -> Result<()> {
        if let Some(resources) = self.persistent_command_resources.take() {
            resources.destroy(&self.ctx);
        }
        if let Some(resources) = self.persistent_host_resources.take() {
            resources.destroy(&self.ctx);
        }
        if let Some(arena) = self.arena.take() {
            arena.destroy(&self.ctx);
        }
        if let Some(banks) = self.causal_block_union_banks.take() {
            banks.destroy(&self.ctx);
        }
        Ok(())
    }
}

impl Drop for S14Runtime {
    fn drop(&mut self) {
        if let Some(resources) = self.persistent_command_resources.take() {
            resources.destroy(&self.ctx);
        }
        if let Some(resources) = self.persistent_host_resources.take() {
            resources.destroy(&self.ctx);
        }
        if let Some(arena) = self.arena.take() {
            arena.destroy(&self.ctx);
        }
        if let Some(banks) = self.causal_block_union_banks.take() {
            banks.destroy(&self.ctx);
        }
    }
}

impl S14CausalBlockRuntimeContinuation {
    pub fn context(&self) -> Arc<VulkanContext> {
        Arc::clone(&self.runtime.ctx)
    }

    pub fn authoritative_state(&self) -> &DecoderStateV1 {
        &self.session.host
    }

    /// 复用`S14Runtime::publish_causal_block_longest_prefix`；这里不复制提交算法。
    pub fn publish_longest_prefix(
        &mut self,
        sealed: S14CausalBlockSealedFuture,
    ) -> Result<S14CausalBlockPublishReceipt> {
        self.runtime
            .publish_causal_block_longest_prefix(&mut self.session, sealed)
            .map_err(|error| anyhow!(error.to_string()))
    }

    /// 从刚发布的session position/input/epoch与active device bank同源派生下一block。
    /// snapshot是device→device copy，只供prefix initializer消费；权威双bank继续留在session。
    pub fn snapshot_next_block(
        &mut self,
        block_size: usize,
    ) -> Result<S14NextCausalBlockCommittedState> {
        if !matches!(block_size, 4 | 8) {
            bail!("causal-block continuation只接受K=4/8，实际K={block_size}");
        }
        if !self.runtime.owns_session(&self.session) {
            bail!("causal-block continuation runtime/session context漂移");
        }
        self.session
            .host
            .validate()
            .context("validate causal-block continuation host state")?;
        let base_position = self.session.position();
        let input_token_id = self.session.input_token_id();
        let commit_epoch = self.session.commit_epoch();
        let block_end = base_position
            .checked_add(u32::try_from(block_size).context("causal-block K无法表示为u32")?)
            .context("causal-block continuation position overflow")?;
        if base_position == 0
            || block_end > self.session.host.native.max_seq_len
            || input_token_id >= VOCAB_SIZE
        {
            bail!(
                "causal-block continuation next block越界: base={base_position} K={block_size} input={input_token_id}"
            );
        }
        let device = self
            .session
            .device
            .as_mut()
            .context("causal-block continuation缺少device state")?;
        if device.epoch() != commit_epoch
            || device.active_bank() != usize::from(self.session.host.active_fixed_bank)
            || device.state_bytes() != self.session.host.native_arena.len() as u64
            || device.candidate_position().is_some()
        {
            bail!("causal-block continuation host/device committed identity漂移");
        }
        let committed_device = device
            .snapshot_detached_committed_state(&self.runtime.ctx)
            .context("snapshot next causal-block committed device bank")?;
        Ok(S14NextCausalBlockCommittedState {
            authoritative: self.session.host.clone(),
            committed_device,
            base_position,
            input_token_id,
            commit_epoch,
        })
    }

    pub fn destroy(self) -> Result<()> {
        let S14CausalBlockRuntimeContinuation { runtime, session } = self;
        session.destroy()?;
        runtime.destroy()
    }
}

impl S14Session {
    pub fn position(&self) -> u32 {
        self.host.position
    }

    pub fn commit_epoch(&self) -> u64 {
        self.host.commit_epoch
    }

    pub fn input_token_id(&self) -> u32 {
        self.host.input_token_id
    }

    pub fn committed_tokens(&self) -> &[polaris_s14_runner::TokenRecord] {
        &self.host.committed_tokens
    }

    pub fn destroy(mut self) -> Result<()> {
        if let Some(device) = self.device.take() {
            device.destroy(&self.ctx)?;
        }
        Ok(())
    }
}

impl Drop for S14Session {
    fn drop(&mut self) {
        if let Some(device) = self.device.take() {
            let _ = device.destroy(&self.ctx);
        }
    }
}

struct CandidateRollbackGuard<'ctx> {
    device: *mut WholeTokenDeviceState,
    ctx: &'ctx VulkanContext,
    active: bool,
    external_armed: bool,
}

impl<'ctx> CandidateRollbackGuard<'ctx> {
    fn new(device: &mut WholeTokenDeviceState, ctx: &'ctx VulkanContext) -> Self {
        Self {
            device,
            ctx,
            active: true,
            external_armed: false,
        }
    }

    fn mark_external_armed(&mut self) {
        self.external_armed = true;
    }

    fn disarm(&mut self) {
        self.active = false;
    }

    fn rollback_now(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        // SAFETY: guard 与当前 step 同线程、同作用域；外部 timeline 已由调用方先 drain。
        let result = unsafe {
            if self.external_armed {
                (*self.device).rollback_external_candidate(self.ctx)
            } else {
                (*self.device).rollback_candidate(self.ctx)
            }
        };
        if result.is_ok() {
            self.active = false;
        }
        result
    }
}

impl Drop for CandidateRollbackGuard<'_> {
    fn drop(&mut self) {
        let _ = self.rollback_now();
    }
}

type Commands = (
    vk::CommandPool,
    vk::CommandPool,
    Vec<vk::CommandBuffer>,
    Vec<vk::CommandBuffer>,
);

fn allocate_commands(ctx: &VulkanContext) -> Result<Commands> {
    unsafe {
        let transfer_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_transfer),
            None,
        )?;
        let compute_pool = match ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        ) {
            Ok(pool) => pool,
            Err(error) => {
                ctx.device.destroy_command_pool(transfer_pool, None);
                return Err(error.into());
            }
        };
        let transfer_commands = match ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(transfer_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count((HEAD_TRANSFER_START + S14_HEAD_CHUNK_COUNT as usize) as u32),
        ) {
            Ok(commands) => commands,
            Err(error) => {
                ctx.device.destroy_command_pool(compute_pool, None);
                ctx.device.destroy_command_pool(transfer_pool, None);
                return Err(error.into());
            }
        };
        let compute_commands = match ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(compute_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count((TERMINAL_COMPUTE + 1) as u32),
        ) {
            Ok(commands) => commands,
            Err(error) => {
                ctx.device.destroy_command_pool(compute_pool, None);
                ctx.device.destroy_command_pool(transfer_pool, None);
                return Err(error.into());
            }
        };
        Ok((
            transfer_pool,
            compute_pool,
            transfer_commands,
            compute_commands,
        ))
    }
}

fn begin_graphics_command(ctx: &VulkanContext, command: vk::CommandBuffer) -> Result<()> {
    unsafe {
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_defaults_are_local_only_and_point_at_fixed_assets() {
        let config = S14RuntimeConfig::production_defaults();
        assert_eq!(config.page_fetch_mode, DynamicPageFetchMode::LocalOnly);
        assert!(config.manifest_path.ends_with(std::path::Path::new(
            "fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json"
        )));
        assert_eq!(
            config.catalog_path,
            PathBuf::from("D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json")
        );
    }

    #[test]
    fn explicit_fetch_requires_explicit_builder_call() {
        let config = S14RuntimeConfig::production_defaults().with_explicit_page_fetch(true);
        assert_eq!(config.page_fetch_mode, DynamicPageFetchMode::ExplicitFetch);
    }

    #[test]
    fn second_step_reuses_persistent_resources_with_zero_allocations() {
        let host = S14Position0PersistentHostStepTelemetry {
            step_index: 1,
            reused_from_previous_step: true,
            resource_allocations_this_step: 0,
            uploader_staging_allocations_this_step: 0,
            mapped_store_allocations_this_step: 0,
            graph_plan_allocations_this_step: 0,
            backend_command_allocations_this_step: 0,
        };
        let command = S14RuntimeCommandStepTelemetry {
            step_index: 1,
            reused_from_previous_step: true,
            resource_allocations_this_step: 0,
            command_pool_resets_this_step: 2,
        };
        assert_eq!(validate_persistent_step_contract(host, command).unwrap(), 0);
        assert!(S14_CANDIDATE_SCOPED_BACKEND_REBUILT_EACH_STEP);
    }

    #[test]
    fn persistent_step_contract_rejects_hidden_reallocation() {
        let host = S14Position0PersistentHostStepTelemetry {
            step_index: 1,
            reused_from_previous_step: true,
            resource_allocations_this_step: 0,
            uploader_staging_allocations_this_step: 1,
            mapped_store_allocations_this_step: 0,
            graph_plan_allocations_this_step: 0,
            backend_command_allocations_this_step: 0,
        };
        let command = S14RuntimeCommandStepTelemetry {
            step_index: 1,
            reused_from_previous_step: true,
            resource_allocations_this_step: 0,
            command_pool_resets_this_step: 2,
        };
        assert!(validate_persistent_step_contract(host, command).is_err());
    }
}
