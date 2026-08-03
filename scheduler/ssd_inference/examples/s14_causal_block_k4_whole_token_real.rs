//! Polaris S14 production position1 K=4 whole-token 唯一真门。
//!
//! 本 example 只装配 `build_s14_causal_block_production_bundle` 与
//! `execute_causal_block_full_depth_with_checkpoints`；不包含任何 synthetic provider、
//! 固定 BOS replay 或逐 token fallback。当 position1 production bootstrap 尚未提供真实
//! authoritative/hidden/KV/RoPE/prefix/terminal owner 时，入口必须 fail-closed。

use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{
    DecoderStateV1, GraphProfile, LongestPrefixDecision, MaterializedTokenSource,
    Position0WholeTokenManifest, COMPRESS_RATIOS, EXPERTS_PER_TOKEN, EXPERT_PAGE_BYTES,
    FULL_DEPTH_LAYERS, VOCAB_SIZE,
};
use ssd_inference::{
    s14_causal_block_base1_k4_provider::{
        build_s14_base1_k4_production_hc_qkv_provider, S14Base1K4AuthoritativeStateBinding,
        S14Base1K4HcQkvExternalResources, S14Base1K4ProductionHcQkvProvider,
        S14Base1K4ProviderInputs,
    },
    s14_causal_block_hc_qkv_recorder::{
        S14CausalBlockHiddenBank,
        S14CausalBlockOwnedBufferSlice as S14CausalBlockHcQkvOwnedBufferSlice,
    },
    s14_causal_block_host_candidates::S14CausalBlockDeferredHostCandidateBatch,
    s14_causal_block_k4_input_hidden::S14CausalBlockK4InputHiddenOwner,
    s14_causal_block_layer::{
        execute_causal_block_full_depth_with_checkpoints, S14CausalBlockDeviceCheckpointStorage,
        S14CausalBlockPublishReceipt, S14CausalBlockSealedFuture,
        S14CausalBlockUnionBankBinding,
    },
    s14_causal_block_prefix_initializer::S14CausalBlockPrefixInitializationOwner,
    s14_causal_block_prefix_producer::S14CausalBlockPrefixStateProducer,
    s14_causal_block_prefix_state::S14CausalBlockPrefixStateProgram,
    s14_causal_block_production_bundle::{
        build_s14_causal_block_production_bundle, S14CausalBlockContextBound,
        S14CausalBlockProductionBundleInputs, S14CausalBlockProductionBundleShape,
        S14CausalBlockProductionHcQkvResourceProvider,
    },
    s14_causal_block_production_evidence::S14CausalBlockProductionEvidenceSnapshot,
    s14_causal_block_ratio4_boundary::S14CausalBlockRatio4BoundaryStateRecorder,
    s14_causal_block_ratio4_state_owner::S14CausalBlockRatio4ProductionStateOwners,
    s14_causal_block_resources::S14CausalBlockUnionBanks,
    s14_causal_block_terminal_adapter::S14CausalBlockHostCandidateFinalizer,
    s14_causal_block_terminal_owner::{
        S14CausalBlockOwnedBufferSlice, S14CausalBlockProductionTerminalResourceOwner,
        S14CausalBlockTerminalHeadUploadState, S14CausalBlockTerminalResourceOwnerInputs,
    },
    s14_dynamic_page_cache_readiness::{
        materialize_planned_range_asset, DynamicPageFetchMode,
    },
    s14_dynamic_routed_page_plan::FullDepthExpertCatalog,
    s14_input_asset_plan::S14InputAssetPlanner,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_layer_program::S14Position0FullDepthLayerProgram,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_runtime::{
        S14CausalBlockRuntimeContinuation, S14NextCausalBlockCommittedState, S14Runtime,
        S14RuntimeConfig,
    },
    VulkanContext,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const BLOCK_SIZE: usize = 4;

fn durable_checkpoint_path() -> PathBuf {
    env::var_os("POLARIS_S14_DURABLE_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../.tmp-polaris-runs/s14-k4-checkpoints/latest.s14ckpt")
        })
}
const BASE_POSITION: u32 = 1;
const REFERENCE_TOKEN_CHAIN: [u32; BLOCK_SIZE + 1] = [5, 223, 939, 21, 695];
const INITIAL_HIDDEN_BANK: usize = 0;
const INITIAL_HIDDEN_GENERATION: u64 = 0;
const EXPECTED_FINAL_HIDDEN_GENERATION: u64 = (FULL_DEPTH_LAYERS.len() * 2) as u64;
const EXPECTED_ONLINE_TOP6_ROWS: usize = BLOCK_SIZE * FULL_DEPTH_LAYERS.len();

fn main() -> Result<()> {
    let token_request = RealK4TokenRequest::from_cli()?;
    build_real_position1_k4_inputs(RealK4GateAssetPaths::production(), token_request)?.run()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RealK4TokenRequest {
    first_input_token_id: u32,
    draft_token_ids: [u32; BLOCK_SIZE],
    second_block_draft_token_ids: Option<[u32; BLOCK_SIZE]>,
}

impl RealK4TokenRequest {
    /// 无参数保留单块reference门；传入5个token执行任意首输入的单块门；传入9个
    /// token时，后4个是调用者提供给第二块的动态draft。K-lane输入始终由
    /// `[first,draft0..2]` 派生，第二块首输入只能来自第一块真实commit。
    fn from_cli() -> Result<Self> {
        let raw = env::args().skip(1).collect::<Vec<_>>();
        if !matches!(raw.len(), 0 | 5 | 9) {
            bail!(
                "用法: s14_causal_block_k4_whole_token_real [first draft0 draft1 draft2 draft3 [next_draft0 next_draft1 next_draft2 next_draft3]]"
            );
        }
        let parsed = if raw.is_empty() {
            REFERENCE_TOKEN_CHAIN.to_vec()
        } else {
            raw.iter()
                .enumerate()
                .map(|(index, value)| {
                    value
                        .parse::<u32>()
                        .with_context(|| format!("token参数{index}不是u32: {value:?}"))
                })
                .collect::<Result<Vec<_>>>()?
        };
        if parsed.iter().any(|&token| token >= VOCAB_SIZE) {
            bail!("K=4 token参数越过词表: {parsed:?}");
        }
        let second_block_draft_token_ids = (parsed.len() == 9)
            .then(|| [parsed[5], parsed[6], parsed[7], parsed[8]]);
        Ok(Self {
            first_input_token_id: parsed[0],
            draft_token_ids: [parsed[1], parsed[2], parsed[3], parsed[4]],
            second_block_draft_token_ids,
        })
    }

    fn input_token_ids(self) -> [u32; BLOCK_SIZE] {
        [
            self.first_input_token_id,
            self.draft_token_ids[0],
            self.draft_token_ids[1],
            self.draft_token_ids[2],
        ]
    }
}

/// 让 fail-closed position0 builder 与已完成的 concrete provider 之间不需要伪造类型。
trait RealK4GateExecutable {
    fn run(self: Box<Self>) -> Result<()>;
}

/// position0 production builder 完成后，顶层装配将所有强 owner 放入此结构。
/// `bundle_inputs.catalog` 与 `orchestrator_catalog` 必须由同一真实 catalog 文件各加载一次：
/// 前者被 bundle 消费，后者在43层 union Range 计划期间持有。
struct RealK4GateInputs<P> {
    bundle_inputs: S14CausalBlockProductionBundleInputs<P>,
    orchestrator_catalog: FullDepthExpertCatalog,
    authoritative: DecoderStateV1,
    terminal_owner: Arc<S14CausalBlockProductionTerminalResourceOwner>,
    host_finalizer: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    external_owner: Box<dyn RealK4ExternalResourceOwner>,
    input_token_ids: [u32; BLOCK_SIZE],
    draft_token_ids: [u32; BLOCK_SIZE],
    reusable: RealK4ReusableProduction,
    second_block_draft_token_ids: Option<[u32; BLOCK_SIZE]>,
}

/// 两个连续K=4 block之间唯一允许存活的production资源。
/// 首块ephemeral provider/prefix/hidden销毁后，第二块从continuation已提交bank重新构造。
struct RealK4ReusableProduction {
    manifest: Arc<Position0WholeTokenManifest>,
    weight_plan: Arc<S14Position0HybridWeightPlan>,
    paged_arena: Arc<S14Position0PagedWeightArena>,
    head_upload: Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>,
    layer_program: S14Position0FullDepthLayerProgram,
    input_planner: S14InputAssetPlanner,
    cache_root: PathBuf,
    catalog_path: PathBuf,
    union_banks: S14CausalBlockUnionBanks,
    continuation: S14CausalBlockRuntimeContinuation,
}

impl RealK4ReusableProduction {
    fn destroy(self) -> Result<()> {
        let context = self.continuation.context();
        let Self {
            manifest,
            weight_plan,
            paged_arena,
            head_upload,
            layer_program,
            input_planner,
            cache_root,
            catalog_path,
            union_banks,
            continuation,
        } = self;
        union_banks.destroy(&context);
        drop(manifest);
        drop(weight_plan);
        drop(layer_program);
        drop(input_planner);
        drop(cache_root);
        drop(catalog_path);

        let mut failures = Vec::new();
        match Arc::try_unwrap(head_upload) {
            Ok(head_upload) => match head_upload.into_inner() {
                Ok(state) => state.uploader.destroy(&context),
                Err(_) => failures.push("terminal head upload mutex poisoned".to_owned()),
            },
            Err(head_upload) => failures.push(format!(
                "terminal head upload 仍被持有: refs={}",
                Arc::strong_count(&head_upload)
            )),
        }
        match Arc::try_unwrap(paged_arena) {
            Ok(paged_arena) => paged_arena.destroy(&context),
            Err(paged_arena) => failures.push(format!(
                "paged arena 仍被持有: refs={}",
                Arc::strong_count(&paged_arena)
            )),
        }
        if let Err(error) = continuation.destroy() {
            failures.push(format!("runtime continuation: {error:#}"));
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(failures.join("; ")))
        }
    }
}

impl<P> RealK4GateExecutable for RealK4GateInputs<P>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    fn run(self: Box<Self>) -> Result<()> {
        run_real_position1_k4_gate(*self)
    }
}

/// bundle 只拥有 recorder/adapter；真实 arena、hidden/KV/head backing allocation 的顶层 owner
/// 必须在 bundle/future/terminal/union bank 全部退出后再销毁。
trait RealK4ExternalResourceOwner {
    fn destroy(self: Box<Self>, context: &Arc<VulkanContext>) -> Result<()>;
}

/// 唯一允许补齐的 production 缺口。实现者必须在同一 Vulkan context 内真实执行
/// position0 `0 -> 5`，并从该次 committed device state 同源导出 K=4 producer 的全部 owner。
/// 这里没有 fixed-BOS manifest replay、forced-prefill checkpoint 或 Python executor 入口。
trait RealK4Position0CommittedProductionBuilder {
    fn build(
        self: Box<Self>,
        request: RealK4Position0BootstrapRequest,
    ) -> Result<RealK4Position0ProductionSeed>;
}

struct RealK4Position0BootstrapRequest {
    manifest: Arc<Position0WholeTokenManifest>,
    weight_plan: Arc<S14Position0HybridWeightPlan>,
    cache_root: PathBuf,
    input_token_ids: [u32; BLOCK_SIZE],
    draft_token_ids: [u32; BLOCK_SIZE],
    catalog_path: PathBuf,
    second_block_draft_token_ids: Option<[u32; BLOCK_SIZE]>,
}

/// position0 production builder 的原子返回值。ratio4 recorder、prefix program/arena 与
/// terminal owner 必须由同一 producer timeline 构造；不能在 example 中事后拼装假 checkpoint。
struct RealK4Position0ProductionSeed {
    context: Arc<VulkanContext>,
    authoritative: DecoderStateV1,
    authoritative_device_state: S14CausalBlockHcQkvOwnedBufferSlice,
    paged_arena: Arc<S14Position0PagedWeightArena>,
    head_upload: Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>,
    hidden_banks: [S14CausalBlockHiddenBank; 2],
    ratio4_boundary_states: BTreeMap<u8, Arc<dyn S14CausalBlockRatio4BoundaryStateRecorder>>,
    prefix_state_producer: S14CausalBlockPrefixStateProducer,
    terminal_owner: Arc<S14CausalBlockProductionTerminalResourceOwner>,
    host_finalizer: S14CausalBlockDeferredHostCandidateBatch,
    union_banks: S14CausalBlockUnionBanks,
    layer_program: S14Position0FullDepthLayerProgram,
    input_planner: S14InputAssetPlanner,
    continuation: S14CausalBlockRuntimeContinuation,
    checkpoint_slots: usize,
    external_owner: Box<dyn RealK4ExternalResourceOwner>,
}

struct S14CommittedPosition0ProductionBuilder;

impl RealK4Position0CommittedProductionBuilder for S14CommittedPosition0ProductionBuilder {
    fn build(
        self: Box<Self>,
        request: RealK4Position0BootstrapRequest,
    ) -> Result<RealK4Position0ProductionSeed> {
        if request.input_token_ids[0] >= VOCAB_SIZE
            || request
                .input_token_ids
                .iter()
                .chain(request.draft_token_ids.iter())
                .any(|&token| token >= VOCAB_SIZE)
            || request.input_token_ids[1..] != request.draft_token_ids[..BLOCK_SIZE - 1]
        {
            bail!("position0 bootstrap 请求的任意首输入/K=4 draft闭包合同漂移");
        }
        let mut config = S14RuntimeConfig::production_defaults();
        config.payload_root = request.cache_root.clone();
        let mut runtime = S14Runtime::load(config).context("加载 S14 production runtime")?;
        let mut session = runtime
            .new_session(0, 127)
            .context("创建 position0 production session")?;
        let position0 = runtime
            .step_with_next_input(&mut session, Some(request.input_token_ids[0]))
            .context("执行真实 position0 whole-token 0→5")?;
        if position0.position != 0
            || position0.input_token_id != 0
            || position0.predicted_token_id != 5
            || position0.commit_epoch != 1
            || position0.active_bank != 1
            || position0.online_top6_routes != FULL_DEPTH_LAYERS.len() as u64
        {
            bail!("真实 position0 whole-token 未闭合 0→5 committed production contract");
        }
        let parts = runtime
            .into_position0_committed_causal_block_parts(session, request.input_token_ids[0])
            .context("导出 position0 committed causal-block parts")?;
        if parts.weights != *request.weight_plan || parts.payload_root != request.cache_root {
            bail!("runtime exporter 与 K=4 bootstrap manifest/weight/cache identity 漂移");
        }

        let context = Arc::clone(&parts.context);
        let authoritative = parts.authoritative.clone();
        let prefix_initialization = S14CausalBlockPrefixInitializationOwner::initialize(
            Arc::clone(&context),
            authoritative.clone(),
            parts.committed_device,
        )
        .context("初始化同源 K=4 prefix checkpoint arena")?;
        let prefix_arena = Arc::clone(prefix_initialization.prefix_arena()?);
        let prefix_program = Arc::new(Mutex::new(S14CausalBlockPrefixStateProgram::build(
            &parts.layer_program,
            parts.paged_arena.workspace_layout(),
            &authoritative.native,
            BLOCK_SIZE,
        )?));
        let prefix_state_producer = prefix_initialization
            .build_prefix_state_producer(Arc::clone(&prefix_program))
            .context("构造真实 K=4 prefix state producer")?;
        let ratio4_owners = S14CausalBlockRatio4ProductionStateOwners::build(
            Arc::clone(&context),
            Arc::clone(&prefix_arena),
            prefix_program,
            &parts.layer_program,
            &authoritative.native,
        )
        .context("构造21层 ratio4 production state owners")?;
        let ratio4_boundary_states = ratio4_owners.trait_states();

        let mut mapped_store = parts.mapped_store;
        let input_planner = parts.input_planner;
        let hidden_owner = S14CausalBlockK4InputHiddenOwner::build_at(
            Arc::clone(&context),
            &input_planner,
            &mut mapped_store,
            BASE_POSITION,
            authoritative.native.max_seq_len,
            request.input_token_ids,
            INITIAL_HIDDEN_GENERATION,
        )
        .context("构造 positions1..4 真实 embedding hidden 双 bank")?;
        let hidden_banks = hidden_owner.hidden_banks()?;
        let final_hidden = S14CausalBlockOwnedBufferSlice::new(
            Arc::clone(&hidden_banks[INITIAL_HIDDEN_BANK].buffer),
            hidden_banks[INITIAL_HIDDEN_BANK].offset,
        );
        let head_upload = Arc::new(Mutex::new(S14CausalBlockTerminalHeadUploadState {
            uploader: parts.uploader,
            store: mapped_store,
        }));
        let terminal_owner = S14CausalBlockProductionTerminalResourceOwner::new(
            S14CausalBlockTerminalResourceOwnerInputs {
                context: Arc::clone(&context),
                block_size: BLOCK_SIZE,
                final_hidden,
                prefix_checkpoint_arena: Arc::clone(&prefix_arena),
                paged_arena: Arc::clone(&parts.paged_arena),
                head_manifest: Arc::clone(&request.manifest),
                head_weight_plan: Arc::clone(&request.weight_plan),
                head_upload: Arc::clone(&head_upload),
            },
        )
        .context("构造同源 K=4 terminal owner")?;
        let host_finalizer = S14CausalBlockDeferredHostCandidateBatch::new(
            authoritative.clone(),
            request.draft_token_ids.to_vec(),
            Arc::clone(&prefix_arena),
        )?;
        let authoritative_device = prefix_initialization.authoritative_device_state()?;
        let authoritative_device_state = S14CausalBlockHcQkvOwnedBufferSlice {
            buffer: authoritative_device.buffer,
            offset: authoritative_device.offset,
            bytes: authoritative.native_arena.len() as u64,
        };
        let paged_arena = Arc::clone(&parts.paged_arena);
        let external_owner = Box::new(S14RealK4ProductionExternalOwner {
            ratio4_owners: Some(ratio4_owners),
            hidden_owner: Some(hidden_owner),
            prefix_initialization: Some(prefix_initialization),
        });
        Ok(RealK4Position0ProductionSeed {
            context,
            authoritative,
            authoritative_device_state,
            paged_arena,
            head_upload,
            hidden_banks,
            ratio4_boundary_states,
            prefix_state_producer,
            terminal_owner,
            host_finalizer,
            union_banks: parts.union_banks,
            layer_program: parts.layer_program,
            input_planner,
            continuation: parts.continuation,
            checkpoint_slots: BLOCK_SIZE,
            external_owner,
        })
    }
}

struct S14RealK4ProductionExternalOwner {
    ratio4_owners: Option<S14CausalBlockRatio4ProductionStateOwners>,
    hidden_owner: Option<S14CausalBlockK4InputHiddenOwner>,
    prefix_initialization: Option<S14CausalBlockPrefixInitializationOwner>,
}

impl RealK4ExternalResourceOwner for S14RealK4ProductionExternalOwner {
    fn destroy(mut self: Box<Self>, _context: &Arc<VulkanContext>) -> Result<()> {
        let mut failures = Vec::new();
        if let Some(mut owners) = self.ratio4_owners.take() {
            if let Err(error) = owners.destroy() {
                failures.push(format!("ratio4 owners: {error:#}"));
            }
        }
        if let Some(mut hidden) = self.hidden_owner.take() {
            if let Err(error) = hidden.destroy() {
                failures.push(format!("hidden banks: {error:#}"));
            }
        }
        if let Some(mut prefix) = self.prefix_initialization.take() {
            if let Err(error) = prefix.destroy() {
                failures.push(format!("prefix initialization: {error:#}"));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(failures.join("; ")))
        }
    }
}

struct CombinedRealK4ExternalResourceOwner {
    provider: Option<S14Base1K4HcQkvExternalResources>,
    production: Option<Box<dyn RealK4ExternalResourceOwner>>,
}

impl RealK4ExternalResourceOwner for CombinedRealK4ExternalResourceOwner {
    fn destroy(mut self: Box<Self>, context: &Arc<VulkanContext>) -> Result<()> {
        let provider_cleanup = self
            .provider
            .as_mut()
            .context("K=4 concrete provider external owner 缺失")?
            .destroy()
            .context("销毁 K=4 concrete provider external resources");
        let production_cleanup = self
            .production
            .take()
            .context("K=4 production external owner 缺失")?
            .destroy(context)
            .context("销毁 K=4 position0 production resources");
        merge_two_cleanups(provider_cleanup, production_cleanup)
    }
}

struct RealK4ContinuationBlockInputs {
    bundle_inputs: S14CausalBlockProductionBundleInputs<S14Base1K4ProductionHcQkvProvider>,
    orchestrator_catalog: FullDepthExpertCatalog,
    authoritative: DecoderStateV1,
    terminal_owner: Arc<S14CausalBlockProductionTerminalResourceOwner>,
    host_finalizer: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    external_owner: Box<dyn RealK4ExternalResourceOwner>,
    input_token_ids: [u32; BLOCK_SIZE],
    draft_token_ids: [u32; BLOCK_SIZE],
}

fn build_continuation_k4_block(
    reusable: &mut RealK4ReusableProduction,
    next: S14NextCausalBlockCommittedState,
    draft_token_ids: [u32; BLOCK_SIZE],
) -> Result<RealK4ContinuationBlockInputs> {
    let context = reusable.continuation.context();
    if next.authoritative != *reusable.continuation.authoritative_state()
        || next.authoritative.position != next.base_position
        || next.authoritative.input_token_id != next.input_token_id
        || next.authoritative.commit_epoch != next.commit_epoch
        || draft_token_ids.iter().any(|&token| token >= VOCAB_SIZE)
    {
        next.committed_device.buffer.destroy(&context);
        bail!("第二块拒绝非同源 selected checkpoint 或非法动态draft");
    }
    let input_token_ids = [
        next.input_token_id,
        draft_token_ids[0],
        draft_token_ids[1],
        draft_token_ids[2],
    ];
    let authoritative = next.authoritative;
    let base_position = next.base_position;

    let mut prefix_initialization = Some(
        S14CausalBlockPrefixInitializationOwner::initialize_at(
            Arc::clone(&context),
            authoritative.clone(),
            next.committed_device,
            base_position,
            BLOCK_SIZE,
        )
        .context("从 selected checkpoint 初始化第二块 prefix arena")?,
    );
    let mut ratio4_owners = None;
    let mut hidden_owner = None;

    let assembled = (|| -> Result<RealK4ContinuationBlockInputs> {
        let bundle_catalog = FullDepthExpertCatalog::load(&reusable.catalog_path)
            .context("加载第二块 production bundle catalog")?;
        let orchestrator_catalog = FullDepthExpertCatalog::load(&reusable.catalog_path)
            .context("加载第二块 production orchestrator catalog")?;
        let prefix = prefix_initialization
            .as_ref()
            .context("第二块 prefix initialization owner 缺失")?;
        let prefix_arena = Arc::clone(prefix.prefix_arena()?);
        let prefix_program = Arc::new(Mutex::new(S14CausalBlockPrefixStateProgram::build(
            &reusable.layer_program,
            reusable.paged_arena.workspace_layout(),
            &authoritative.native,
            BLOCK_SIZE,
        )?));
        let prefix_state_producer = prefix
            .build_prefix_state_producer(Arc::clone(&prefix_program))
            .context("构造第二块 prefix state producer")?;
        ratio4_owners = Some(
            S14CausalBlockRatio4ProductionStateOwners::build(
                Arc::clone(&context),
                Arc::clone(&prefix_arena),
                prefix_program,
                &reusable.layer_program,
                &authoritative.native,
            )
            .context("构造第二块 ratio4 production state owners")?,
        );
        let ratio4_boundary_states = ratio4_owners
            .as_ref()
            .context("第二块 ratio4 owner 缺失")?
            .trait_states();

        hidden_owner = Some({
            let mut head = reusable
                .head_upload
                .lock()
                .map_err(|_| anyhow!("第二块 terminal head upload mutex poisoned"))?;
            S14CausalBlockK4InputHiddenOwner::build_at(
                Arc::clone(&context),
                &reusable.input_planner,
                &mut head.store,
                base_position,
                authoritative.native.max_seq_len,
                input_token_ids,
                INITIAL_HIDDEN_GENERATION,
            )
            .context("构造第二块真实 embedding hidden 双 bank")?
        });
        let hidden_banks = hidden_owner
            .as_ref()
            .context("第二块 hidden owner 缺失")?
            .hidden_banks()?;
        let final_hidden = S14CausalBlockOwnedBufferSlice::new(
            Arc::clone(&hidden_banks[INITIAL_HIDDEN_BANK].buffer),
            hidden_banks[INITIAL_HIDDEN_BANK].offset,
        );
        let terminal_owner = S14CausalBlockProductionTerminalResourceOwner::new(
            S14CausalBlockTerminalResourceOwnerInputs {
                context: Arc::clone(&context),
                block_size: BLOCK_SIZE,
                final_hidden,
                prefix_checkpoint_arena: Arc::clone(&prefix_arena),
                paged_arena: Arc::clone(&reusable.paged_arena),
                head_manifest: Arc::clone(&reusable.manifest),
                head_weight_plan: Arc::clone(&reusable.weight_plan),
                head_upload: Arc::clone(&reusable.head_upload),
            },
        )
        .context("构造第二块 terminal owner")?;
        let host_finalizer = S14CausalBlockDeferredHostCandidateBatch::new(
            authoritative.clone(),
            draft_token_ids.to_vec(),
            Arc::clone(&prefix_arena),
        )?;
        let authoritative_device = prefix.authoritative_device_state()?;
        let authoritative_device_state = S14CausalBlockHcQkvOwnedBufferSlice {
            buffer: authoritative_device.buffer,
            offset: authoritative_device.offset,
            bytes: authoritative.native_arena.len() as u64,
        };
        let (provider, provider_external) = build_s14_base1_k4_production_hc_qkv_provider(
            S14Base1K4ProviderInputs {
                context: Arc::clone(&context),
                manifest: Arc::clone(&reusable.manifest),
                weight_plan: Arc::clone(&reusable.weight_plan),
                paged_arena: Arc::clone(&reusable.paged_arena),
                head_upload: Arc::clone(&reusable.head_upload),
                base_position,
                authoritative: S14Base1K4AuthoritativeStateBinding {
                    native_state: authoritative.native.clone(),
                    device_state: authoritative_device_state,
                },
                input_token_ids,
                ratio4_boundary_states,
                prefix_state_producer,
            },
        )
        .context("构造第二块通用base K=4 HC/QKV provider")?;
        let checkpoint_state_bytes = u64::try_from(authoritative.native_arena.len())?;
        let shape = S14CausalBlockProductionBundleShape::new(
            BLOCK_SIZE,
            checkpoint_state_bytes,
            BLOCK_SIZE,
        )?;
        let bundle_inputs = S14CausalBlockProductionBundleInputs {
            context: Arc::clone(&context),
            shape,
            hc_qkv_provider: S14CausalBlockContextBound::new(Arc::clone(&context), provider),
            hidden_banks: S14CausalBlockContextBound::new(Arc::clone(&context), hidden_banks),
            catalog: bundle_catalog,
            cache_root: reusable.cache_root.clone(),
            fetch_mode: DynamicPageFetchMode::ExplicitFetch,
            static_arena: S14CausalBlockContextBound::new(
                Arc::clone(&context),
                Arc::clone(&reusable.paged_arena),
            ),
        };
        let production_owner = Box::new(S14RealK4ProductionExternalOwner {
            ratio4_owners: ratio4_owners.take(),
            hidden_owner: hidden_owner.take(),
            prefix_initialization: prefix_initialization.take(),
        });
        Ok(RealK4ContinuationBlockInputs {
            bundle_inputs,
            orchestrator_catalog,
            authoritative,
            terminal_owner,
            host_finalizer: Box::new(host_finalizer),
            external_owner: Box::new(CombinedRealK4ExternalResourceOwner {
                provider: Some(provider_external),
                production: Some(production_owner),
            }),
            input_token_ids,
            draft_token_ids,
        })
    })();

    match assembled {
        Ok(block) => Ok(block),
        Err(error) => {
            let cleanup = Box::new(S14RealK4ProductionExternalOwner {
                ratio4_owners,
                hidden_owner,
                prefix_initialization,
            })
            .destroy(&context);
            match cleanup {
                Ok(()) => Err(error),
                Err(cleanup) => Err(anyhow!("{error:#}; 第二块构造cleanup: {cleanup:#}")),
            }
        }
    }
}

fn execute_and_commit_continuation_block(
    reusable: &mut RealK4ReusableProduction,
    inputs: RealK4ContinuationBlockInputs,
) -> Result<(S14CausalBlockPublishReceipt, LongestPrefixDecision)> {
    let RealK4ContinuationBlockInputs {
        bundle_inputs,
        orchestrator_catalog,
        authoritative,
        terminal_owner,
        host_finalizer,
        external_owner,
        input_token_ids,
        draft_token_ids,
    } = inputs;
    let base_position = authoritative.position;
    let context = Arc::clone(&bundle_inputs.context);
    let mut bundle_inputs = Some(bundle_inputs);
    let mut terminal_owner = Some(terminal_owner);
    let mut host_finalizer = Some(host_finalizer);
    let mut bundle = None;

    let execution = (|| -> Result<(S14CausalBlockPublishReceipt, LongestPrefixDecision)> {
        validate_unique_gate_inputs(
            bundle_inputs
                .as_ref()
                .context("第二块 bundle inputs 已被消费")?,
            &authoritative,
            input_token_ids[0],
            base_position,
        )?;
        bundle = Some(
            build_s14_causal_block_production_bundle(
                bundle_inputs
                    .take()
                    .context("第二块 bundle inputs 只允许消费一次")?,
            )
            .context("构造第二块 production bundle")?,
        );
        let production = bundle.as_mut().context("第二块 production bundle 缺失")?;
        let evidence_owner = Arc::clone(
            terminal_owner
                .as_ref()
                .context("第二块 terminal owner 缺失")?,
        );
        production
            .install_terminal_resources_for_next_block(
                terminal_owner
                    .take()
                    .context("第二块 terminal owner 只允许消费一次")?,
                host_finalizer
                    .take()
                    .context("第二块 host finalizer 只允许消费一次")?,
            )
            .map_err(anyhow::Error::msg)
            .context("安装第二块 terminal owner/finalizer")?;

        let union_plan = reusable
            .union_banks
            .plan(BLOCK_SIZE)
            .context("取得第二块 union bank plan")?;
        let bank_index = authoritative.commit_epoch as usize % 2;
        let union_buffer = reusable
            .union_banks
            .bank(bank_index)
            .context("取得第二块 union bank")?;
        let union_binding = S14CausalBlockUnionBankBinding {
            bank_index,
            buffer: union_buffer.handle(),
            allocated_bank_bytes: union_buffer.size(),
        };
        let initial_hidden = production
            .initial_hidden_binding(INITIAL_HIDDEN_BANK, INITIAL_HIDDEN_GENERATION)
            .context("绑定第二块 authoritative initial hidden")?;
        let sealed = execute_causal_block_full_depth_with_checkpoints(
            union_plan,
            union_binding,
            &orchestrator_catalog,
            &authoritative,
            &draft_token_ids,
            &input_token_ids,
            initial_hidden,
            MaterializedTokenSource::SpeculativeDraft,
            production,
        )
        .context("执行第二块 FullDepth43 production gate")?;
        validate_sealed_future(
            &sealed,
            &authoritative,
            initial_hidden,
            &draft_token_ids,
            base_position,
        )?;
        validate_production_evidence(
            evidence_owner.production_evidence_snapshot()?,
            base_position,
        )?;
        drop(evidence_owner);
        let decision = sealed.decision().clone();
        let publish = reusable
            .continuation
            .publish_longest_prefix(sealed)
            .context("发布第二块实际 selected checkpoint")?;
        Ok((publish, decision))
    })();

    let bundle_cleanup = match bundle.as_mut() {
        Some(production) => production
            .destroy()
            .map_err(anyhow::Error::msg)
            .context("销毁第二块 production bundle"),
        None => Ok(()),
    };
    drop(bundle);
    drop(bundle_inputs);
    drop(terminal_owner);
    drop(host_finalizer);
    drop(orchestrator_catalog);
    let external_cleanup = external_owner
        .destroy(&context)
        .context("销毁第二块临时 production owner");
    merge_block_execution_and_cleanup(execution, bundle_cleanup, external_cleanup)
}

fn validate_publish_receipt(
    receipt: &S14CausalBlockPublishReceipt,
    authoritative: &DecoderStateV1,
    base_position: u32,
    decision: &LongestPrefixDecision,
) -> Result<()> {
    let committed_count = decision.committed_token_ids.len();
    let expected_position = base_position
        .checked_add(u32::try_from(committed_count).context("committed count无法表示为u32")?)
        .context("published position overflow")?;
    let expected_epoch = u64::from(base_position)
        .checked_add(u64::try_from(committed_count).context("committed count无法表示为u64")?)
        .context("published epoch overflow")?;
    if !(1..=BLOCK_SIZE).contains(&committed_count)
        || receipt.host.base_position != base_position
        || receipt.host.committed_position != expected_position
        || receipt.host.committed_epoch != expected_epoch
        || receipt.host.checkpoint_index + 1 != committed_count
        || receipt.host.decision != *decision
        || receipt.device.position != expected_position
        || receipt.device.epoch != expected_epoch
        || receipt.device.accepted_tokens != committed_count
        || receipt.device.checkpoint_index + 1 != committed_count
        || authoritative.position != expected_position
        || authoritative.native.position != expected_position
        || authoritative.commit_epoch != expected_epoch
        || authoritative.input_token_id != decision.committed_token_ids[committed_count - 1]
        || usize::from(authoritative.active_fixed_bank) != receipt.device.active_bank
    {
        bail!(
            "causal-block publish receipt 未闭合: base={base_position} committed={committed_count} host={:?} device={:?}",
            receipt.host,
            receipt.device
        );
    }
    Ok(())
}

fn merge_block_execution_and_cleanup<T>(
    execution: Result<T>,
    bundle_cleanup: Result<()>,
    external_cleanup: Result<()>,
) -> Result<T> {
    match execution {
        Ok(value) => {
            merge_two_cleanups(bundle_cleanup, external_cleanup)?;
            Ok(value)
        }
        Err(error) => {
            let cleanup = merge_two_cleanups(bundle_cleanup, external_cleanup);
            match cleanup {
                Ok(()) => Err(error),
                Err(cleanup) => Err(anyhow!("execution: {error:#}; cleanup: {cleanup:#}")),
            }
        }
    }
}

fn run_real_position1_k4_gate<P>(inputs: RealK4GateInputs<P>) -> Result<()>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    let RealK4GateInputs {
        bundle_inputs,
        orchestrator_catalog,
        authoritative,
        terminal_owner,
        host_finalizer,
        external_owner,
        input_token_ids,
        draft_token_ids,
        reusable,
        second_block_draft_token_ids,
    } = inputs;
    let mut reusable = reusable;
    let context = Arc::clone(&bundle_inputs.context);
    let mut bundle_inputs = Some(bundle_inputs);
    let mut terminal_owner = Some(terminal_owner);
    let mut host_finalizer = Some(host_finalizer);
    let mut external_owner = Some(external_owner);
    let mut bundle = None;

    let execution = (|| -> Result<()> {
        validate_unique_gate_inputs(
            bundle_inputs
                .as_ref()
                .context("K=4 gate bundle inputs 已被消费")?,
            &authoritative,
            input_token_ids[0],
            BASE_POSITION,
        )?;

        bundle = Some(
            build_s14_causal_block_production_bundle(
                bundle_inputs
                    .take()
                    .context("K=4 gate bundle inputs 只允许消费一次")?,
            )
            .context("构造 S14 position1 K=4 production bundle")?,
        );
        let production = bundle.as_mut().context("K=4 production bundle 缺失")?;
        let evidence_owner = Arc::clone(
            terminal_owner
                .as_ref()
                .context("K=4 terminal owner 已在安装前被消费")?,
        );
        production
            .install_terminal_resources_for_next_block(
                terminal_owner
                    .take()
                    .context("K=4 terminal owner 只允许消费一次")?,
                host_finalizer
                    .take()
                    .context("K=4 host finalizer 只允许消费一次")?,
            )
            .map_err(anyhow::Error::msg)
            .context("安装同源 position1 K=4 terminal owner/finalizer")?;

        let union_plan = reusable
            .union_banks
            .plan(BLOCK_SIZE)
            .context("取得 K=4 union bank plan")?;
        let bank_index = authoritative.commit_epoch as usize % 2;
        let union_buffer = reusable
            .union_banks
            .bank(bank_index)
            .context("取得 K=4 union bank")?;
        let union_binding = S14CausalBlockUnionBankBinding {
            bank_index,
            buffer: union_buffer.handle(),
            allocated_bank_bytes: union_buffer.size(),
        };
        let initial_hidden = production
            .initial_hidden_binding(INITIAL_HIDDEN_BANK, INITIAL_HIDDEN_GENERATION)
            .context("绑定 position1 K=4 authoritative initial hidden")?;

        let sealed = execute_causal_block_full_depth_with_checkpoints(
            union_plan,
            union_binding,
            &orchestrator_catalog,
            &authoritative,
            &draft_token_ids,
            &input_token_ids,
            initial_hidden,
            MaterializedTokenSource::SpeculativeDraft,
            production,
        )
        .context("执行 S14 position1 K=4 real whole-token gate")?;

        validate_sealed_future(
            &sealed,
            &authoritative,
            initial_hidden,
            &draft_token_ids,
            BASE_POSITION,
        )?;
        let decision = sealed.decision().clone();
        validate_production_evidence(
            evidence_owner.production_evidence_snapshot()?,
            BASE_POSITION,
        )?;
        drop(evidence_owner);
        if let Some(next_drafts) = second_block_draft_token_ids {
            let first_publish = reusable
                .continuation
                .publish_longest_prefix(sealed)
                .context("发布第一块实际 selected checkpoint")?;
            validate_publish_receipt(
                &first_publish,
                reusable.continuation.authoritative_state(),
                BASE_POSITION,
                &decision,
            )?;
            let first_checkpoint = reusable
                .continuation
                .persist_committed_block_checkpoint(&durable_checkpoint_path())
                .context(
                    "第一块已commit，但durable checkpoint发布失败；可在当前continuation上重试",
                )?;
            println!(
                "checkpoint=committed position={} epoch={} bank={} bytes={} sha256={} path={}",
                first_checkpoint.position,
                first_checkpoint.commit_epoch,
                first_checkpoint.active_bank,
                first_checkpoint.arena_bytes,
                first_checkpoint.file_sha256,
                first_checkpoint.path.display(),
            );

            bundle
                .as_mut()
                .context("第一块 production bundle 缺失")?
                .destroy()
                .map_err(anyhow::Error::msg)
                .context("销毁第一块 production bundle")?;
            drop(bundle.take());
            external_owner
                .take()
                .context("第一块 external owner 缺失")?
                .destroy(&context)
                .context("销毁第一块临时 production owner")?;

            reusable
                .head_upload
                .lock()
                .map_err(|_| anyhow!("重开第二块 head stream 时 uploader mutex poisoned"))?
                .uploader
                .begin_next_causal_block_head_stream(&reusable.weight_plan)
                .context("重开第二块 causal-block head stream")?;

            let next = reusable
                .continuation
                .snapshot_next_block(BLOCK_SIZE)
                .context("从第一块真实提交态 snapshot 第二块起点")?;
            let second_base_position = next.base_position;
            let second = build_continuation_k4_block(&mut reusable, next, next_drafts)
                .context("从第一块 selected checkpoint 构造第二块")?;
            let (second_publish, second_decision) =
                execute_and_commit_continuation_block(&mut reusable, second)
                    .context("执行并提交第二个 production K=4 block")?;
            validate_publish_receipt(
                &second_publish,
                reusable.continuation.authoritative_state(),
                second_base_position,
                &second_decision,
            )?;
            let second_checkpoint = reusable
                .continuation
                .persist_committed_block_checkpoint(&durable_checkpoint_path())
                .context(
                    "第二块已commit，但durable checkpoint发布失败；可在当前continuation上重试",
                )?;
            println!(
                "status=pass mode=polaris_s14_production_k4_two_blocks first_base={} second_base={} final_position={} first_committed={:?} second_committed={:?} first_drafts={:?} second_drafts={:?} blocks=2 committed=true checkpoint_position={} checkpoint_epoch={} checkpoint_sha256={} checkpoint_path={}",
                BASE_POSITION,
                second_base_position,
                reusable.continuation.authoritative_state().position,
                decision.committed_token_ids,
                second_decision.committed_token_ids,
                draft_token_ids,
                next_drafts,
                second_checkpoint.position,
                second_checkpoint.commit_epoch,
                second_checkpoint.file_sha256,
                second_checkpoint.path.display(),
            );
        } else {
            let authoritative_before_rollback = authoritative.clone();
            let rollback = sealed
                .rollback(&authoritative)
                .context("K=4 sealed future rollback")?;
            if rollback.position != BASE_POSITION
                || rollback.commit_epoch != authoritative.commit_epoch
                || authoritative != authoritative_before_rollback
            {
                bail!("K=4 gate rollback 修改了 authoritative state");
            }
            println!(
                "status=pass mode=polaris_s14_production_k4_whole_token_real base_position={} inputs={:?} drafts={:?} committed_tokens={:?} fallback={:?} mismatch_index={:?} layers={} online_top6_rows={} grouped_moe_submits={} checkpoints=4 batched_head_submits=1 serial_token_forward_calls=0 committed=false",
                BASE_POSITION,
                input_token_ids,
                draft_token_ids,
                decision.committed_token_ids,
                decision.fallback_token_id,
                decision.mismatch_index,
                FULL_DEPTH_LAYERS.len(),
                EXPECTED_ONLINE_TOP6_ROWS,
                FULL_DEPTH_LAYERS.len(),
            );
        }
        Ok(())
    })();

    let bundle_cleanup = match bundle.as_mut() {
        Some(production) => production
            .destroy()
            .map_err(anyhow::Error::msg)
            .context("销毁 K=4 production bundle"),
        None => Ok(()),
    };
    drop(bundle);
    drop(bundle_inputs);
    drop(terminal_owner);
    drop(host_finalizer);
    drop(orchestrator_catalog);
    let external_cleanup = match external_owner.take() {
        Some(owner) => owner
            .destroy(&context)
            .context("销毁 K=4 external production assets"),
        None => Ok(()),
    };
    let reusable_cleanup = reusable
        .destroy()
        .context("销毁 K=4 reusable production assets");
    let runtime_cleanup = merge_two_cleanups(external_cleanup, reusable_cleanup);
    merge_execution_and_cleanup(execution, bundle_cleanup, runtime_cleanup)
}

fn validate_unique_gate_inputs<P>(
    bundle_inputs: &S14CausalBlockProductionBundleInputs<P>,
    authoritative: &DecoderStateV1,
    expected_first_input: u32,
    base_position: u32,
) -> Result<()>
where
    P: S14CausalBlockProductionHcQkvResourceProvider,
{
    authoritative.validate()?;
    let checkpoint_state_bytes = u64::try_from(authoritative.native_arena.len())
        .context("authoritative native arena bytes 无法表示为u64")?;
    let committed_tail = authoritative.committed_tokens.last();
    if base_position == 0
        || authoritative.position != base_position
        || authoritative.native.position != base_position
        || authoritative.commit_epoch != u64::from(base_position)
        || authoritative.input_token_id != expected_first_input
        || usize::try_from(base_position).ok() != Some(authoritative.committed_tokens.len())
        || committed_tail.is_none_or(|token| token.position + 1 != base_position)
        || bundle_inputs.shape.block_size() != BLOCK_SIZE
        || bundle_inputs.shape.checkpoint_state_bytes() != checkpoint_state_bytes
        || bundle_inputs.shape.checkpoint_slots() == 0
    {
        bail!(
            "K=4 真门必须从真实 committed state 开始: base={base_position} input={expected_first_input}"
        );
    }
    if !Arc::ptr_eq(
        &bundle_inputs.context,
        bundle_inputs.hc_qkv_provider.context(),
    ) || !Arc::ptr_eq(&bundle_inputs.context, bundle_inputs.hidden_banks.context())
        || !Arc::ptr_eq(&bundle_inputs.context, bundle_inputs.static_arena.context())
    {
        bail!("K=4 gate provider/hidden/paged arena 不属于同一 VulkanContext");
    }
    Ok(())
}

fn validate_sealed_future(
    sealed: &S14CausalBlockSealedFuture,
    authoritative: &DecoderStateV1,
    initial_hidden: ssd_inference::s14_causal_block_layer::S14CausalBlockHiddenBinding,
    draft_token_ids: &[u32; BLOCK_SIZE],
    base_position: u32,
) -> Result<()> {
    let layers = &sealed.layers;
    if layers.base_position != base_position
        || layers.block_size != BLOCK_SIZE
        || layers.completed_layers != FULL_DEPTH_LAYERS.len()
        || layers.layer_summaries.len() != FULL_DEPTH_LAYERS.len()
        || layers.attention_router_forward_calls != FULL_DEPTH_LAYERS.len() as u32
        || layers.union_range_materialize_calls != FULL_DEPTH_LAYERS.len() as u32
        || layers.grouped_moe_submit_calls != FULL_DEPTH_LAYERS.len() as u32
        || layers.serial_token_forward_calls != 0
        || !layers.lifecycle_sealed
        || !layers.head_ready
        || layers.checkpoint_commit_ready
        || layers.final_hidden.generation != EXPECTED_FINAL_HIDDEN_GENERATION
        || layers.final_hidden.buffer != initial_hidden.buffer
        || layers.final_hidden.offset != initial_hidden.offset
        || layers.final_hidden.bytes != initial_hidden.bytes
    {
        bail!("K=4 FullDepth43/head/hidden/zero-serial 强回执漂移");
    }
    for (summary, &expected_layer) in layers.layer_summaries.iter().zip(&FULL_DEPTH_LAYERS) {
        if summary.layer != expected_layer
            || !(1..=BLOCK_SIZE * EXPERTS_PER_TOKEN).contains(&summary.unique_experts)
            || summary.physical_ranges != summary.unique_experts * 6
            || summary.union_expert_bytes != summary.unique_experts as u64 * EXPERT_PAGE_BYTES
        {
            bail!("K=4 L{expected_layer} online top-6 union Range 回执漂移");
        }
    }
    if layers.routes_by_position.len() != BLOCK_SIZE {
        bail!("K=4 online top-6 position rows 不是4");
    }
    for (lane, routes) in layers.routes_by_position.iter().enumerate() {
        if routes.len() != FULL_DEPTH_LAYERS.len() {
            bail!("K=4 lane{lane} online top-6 不是43层");
        }
        for (route, &expected_layer) in routes.iter().zip(&FULL_DEPTH_LAYERS) {
            route
                .validate_for(GraphProfile::FullDepth43NativeTop6)
                .with_context(|| format!("K=4 lane{lane} L{expected_layer} route 非法"))?;
            if route.layer != expected_layer
                || route.expert_ids.len() != EXPERTS_PER_TOKEN
                || route.weights.len() != EXPERTS_PER_TOKEN
            {
                bail!("K=4 lane{lane} L{expected_layer} online top-6 identity 漂移");
            }
        }
    }

    let device = sealed.device_receipt();
    let expected_state_bytes = u64::try_from(authoritative.native_arena.len())?;
    if device.base_position != base_position
        || device.block_size != BLOCK_SIZE
        || device.checkpoint_count != BLOCK_SIZE
        || device.storage != S14CausalBlockDeviceCheckpointStorage::PrefixCheckpoints
        || device.checkpoint_state_bytes != expected_state_bytes
        || device.final_hidden != layers.final_hidden
        || device.ready_timeline_value == 0
    {
        bail!("K=4 device prefix checkpoint owner/receipt 漂移");
    }

    let decision = sealed.decision();
    let committed_count = decision.committed_token_ids.len();
    if !(1..=BLOCK_SIZE).contains(&committed_count) {
        bail!("K=4 longest-prefix committed token count越界: {committed_count}");
    }
    match decision.mismatch_index {
        Some(index) => {
            let fallback = decision
                .fallback_token_id
                .context("K=4 mismatch决策缺少fallback token")?;
            if index >= BLOCK_SIZE
                || decision.accepted_prefix.as_slice() != &draft_token_ids[..index]
                || decision.rejected_draft_suffix.as_slice() != &draft_token_ids[index..]
                || decision.committed_token_ids.len() != index + 1
                || decision.committed_token_ids[..index] != draft_token_ids[..index]
                || decision.committed_token_ids[index] != fallback
            {
                bail!("K=4 mismatch longest-prefix/draft/fallback闭包漂移");
            }
        }
        None => {
            if decision.accepted_prefix.as_slice() != draft_token_ids
                || decision.fallback_token_id.is_some()
                || !decision.rejected_draft_suffix.is_empty()
                || decision.committed_token_ids.as_slice() != draft_token_ids
            {
                bail!("K=4 full-accept longest-prefix/draft闭包漂移");
            }
        }
    }
    let selected = sealed.selected_prefix()?;
    let checkpoint = selected.checkpoint();
    let expected_active_fixed_bank =
        authoritative.active_fixed_bank ^ ((committed_count as u8) & 1);
    let expected_checkpoint_index = committed_count - 1;
    let expected_position = base_position + committed_count as u32;
    let expected_commit_epoch = authoritative.commit_epoch + committed_count as u64;
    let expected_input_token_id = decision.committed_token_ids[expected_checkpoint_index];
    let expected_committed_tokens_len = authoritative.committed_tokens.len() + committed_count;
    if selected.accepted_tokens() != committed_count
        || selected.checkpoint_index() != expected_checkpoint_index
        || checkpoint.position != expected_position
        || checkpoint.commit_epoch != expected_commit_epoch
        || checkpoint.input_token_id != expected_input_token_id
        || checkpoint.active_fixed_bank != expected_active_fixed_bank
        || checkpoint.committed_tokens.len() != expected_committed_tokens_len
    {
        bail!(
            "K=4 longest-prefix host/device checkpoint 与实际接受长度漂移: accepted_tokens={}/{} checkpoint_index={}/{} position={}/{} commit_epoch={}/{} input_token_id={}/{} active_fixed_bank={}/{} committed_tokens_len={}/{}",
            selected.accepted_tokens(),
            committed_count,
            selected.checkpoint_index(),
            expected_checkpoint_index,
            checkpoint.position,
            expected_position,
            checkpoint.commit_epoch,
            expected_commit_epoch,
            checkpoint.input_token_id,
            expected_input_token_id,
            checkpoint.active_fixed_bank,
            expected_active_fixed_bank,
            checkpoint.committed_tokens.len(),
            expected_committed_tokens_len,
        );
    }
    Ok(())
}

fn validate_production_evidence(
    evidence: S14CausalBlockProductionEvidenceSnapshot,
    base_position: u32,
) -> Result<()> {
    let expected_ratio4_layers = FULL_DEPTH_LAYERS
        .iter()
        .copied()
        .filter(|&layer| COMPRESS_RATIOS[usize::from(layer)] == 4)
        .collect::<Vec<_>>();
    let prefix = evidence
        .prefix_seal_receipt
        .context("K=4 production evidence 缺少 prefix seal receipt")?;
    if evidence.base_position != base_position
        || evidence.block_size != BLOCK_SIZE
        || evidence.expected_ratio4_layers != expected_ratio4_layers
        || evidence.ratio4_layer_evidence.len() != expected_ratio4_layers.len()
        || evidence.position3_finalize_writeback_rollover_layers != expected_ratio4_layers
        || evidence.position4_indexer_attention_layers != expected_ratio4_layers
        || prefix.base_position != base_position
        || prefix.block_size != BLOCK_SIZE
        || prefix.sealed_prefixes != BLOCK_SIZE
        || prefix.sealed_prefix_layers != BLOCK_SIZE * FULL_DEPTH_LAYERS.len()
        || prefix.cumulative_lane_applications
            != (BLOCK_SIZE * (BLOCK_SIZE + 1) / 2) * FULL_DEPTH_LAYERS.len()
        || evidence.serial_token_forward_calls != 0
        || evidence.cpu_fallback_calls != 0
        || !evidence.ratio4_prefix_evidence_complete
    {
        bail!("K=4 ratio4/indexer/prefix production evidence 强回执漂移: {evidence:?}");
    }
    Ok(())
}

fn merge_execution_and_cleanup(
    execution: Result<()>,
    bundle_cleanup: Result<()>,
    external_cleanup: Result<()>,
) -> Result<()> {
    let mut failures = Vec::new();
    if let Err(error) = execution {
        failures.push(format!("execution: {error:#}"));
    }
    if let Err(error) = bundle_cleanup {
        failures.push(format!("bundle cleanup: {error:#}"));
    }
    if let Err(error) = external_cleanup {
        failures.push(format!("external cleanup: {error:#}"));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(anyhow!(failures.join("; ")))
    }
}

fn merge_two_cleanups(left: Result<()>, right: Result<()>) -> Result<()> {
    match (left, right) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(left), Err(right)) => Err(anyhow!("{left:#}; {right:#}")),
    }
}

#[derive(Debug, Clone)]
struct RealK4GateAssetPaths {
    manifest: PathBuf,
    cache_root: PathBuf,
    catalog: PathBuf,
    incompatible_forced_checkpoint: PathBuf,
}

impl RealK4GateAssetPaths {
    fn production() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            manifest: root.join(
                "fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
            ),
            cache_root: PathBuf::from("D:/models/Polaris-S14/range_cache"),
            catalog: PathBuf::from(
                "D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json",
            ),
            incompatible_forced_checkpoint: PathBuf::from(
                "D:/models/Polaris-S14/checkpoints/causal-block-k4-forced.json",
            ),
        }
    }
}

fn build_real_position1_k4_inputs(
    paths: RealK4GateAssetPaths,
    token_request: RealK4TokenRequest,
) -> Result<Box<dyn RealK4GateExecutable>> {
    validate_real_asset_file(&paths.manifest, "position0 whole-token manifest")?;
    validate_real_asset_file(&paths.catalog, "FullDepth43 native top-6 catalog")?;
    if !paths.cache_root.is_dir() {
        bail!(
            "Polaris S14 real Range cache 目录不存在: {}",
            paths.cache_root.display()
        );
    }
    materialize_statically_known_embedding_pages(&paths, token_request)
        .context("补齐两块启动前可静态判定的 embedding Range 页")?;
    let manifest = Arc::new(
        Position0WholeTokenManifest::load(&paths.manifest)
            .context("加载真实 position0 manifest")?,
    );
    let weight_plan = Arc::new(
        S14Position0HybridWeightPlan::build(&manifest)
            .context("构造真实 production paged weight plan")?,
    );
    let bundle_catalog =
        FullDepthExpertCatalog::load(&paths.catalog).context("加载 production bundle catalog")?;
    let orchestrator_catalog = FullDepthExpertCatalog::load(&paths.catalog)
        .context("加载 production orchestrator catalog")?;

    if paths.incompatible_forced_checkpoint.is_file() {
        eprintln!(
            "忽略 incompatible forced checkpoint {}：它不是 position0 0→5 committed production state",
            paths.incompatible_forced_checkpoint.display()
        );
    }

    build_real_position1_k4_inputs_with_builder(
        RealK4Position0BootstrapRequest {
            manifest,
            weight_plan,
            cache_root: paths.cache_root,
            input_token_ids: token_request.input_token_ids(),
            draft_token_ids: token_request.draft_token_ids,
            catalog_path: paths.catalog,
            second_block_draft_token_ids: token_request.second_block_draft_token_ids,
        },
        bundle_catalog,
        orchestrator_catalog,
        Box::new(S14CommittedPosition0ProductionBuilder),
    )
}

/// GPU context 创建前先经 production Range transport 补齐两块已知输入页。
/// 第二块首 token 只能来自第一块真实 selected checkpoint，无法静态猜测；
/// 其余三个 lane 则来自调用者的 next_draft0..2，可在唯一 GPU 真门前完成
/// HTTPS 206、Content-Range、长度、proof 与 SHA 校验，避免持有显存时才发现缺页。
fn materialize_statically_known_embedding_pages(
    paths: &RealK4GateAssetPaths,
    token_request: RealK4TokenRequest,
) -> Result<()> {
    let planner = S14InputAssetPlanner::load_pinned(&paths.catalog, &paths.cache_root)
        .context("加载静态 embedding Range planner")?;
    let mut token_ids = token_request
        .input_token_ids()
        .into_iter()
        .collect::<BTreeSet<_>>();
    if let Some(next_drafts) = token_request.second_block_draft_token_ids {
        token_ids.extend(next_drafts[..BLOCK_SIZE - 1].iter().copied());
    }
    for (position, token_id) in token_ids.into_iter().enumerate() {
        let position = u32::try_from(position).context("静态 embedding position overflow")?;
        let plan = planner
            .plan(position, token_id, 127)
            .with_context(|| format!("规划静态 embedding token={token_id}"))?;
        materialize_planned_range_asset(
            &plan.embedding,
            &paths.cache_root,
            DynamicPageFetchMode::ExplicitFetch,
        )
        .with_context(|| format!("补齐静态 embedding token={token_id}"))?;
    }
    Ok(())
}

fn build_real_position1_k4_inputs_with_builder(
    request: RealK4Position0BootstrapRequest,
    bundle_catalog: FullDepthExpertCatalog,
    orchestrator_catalog: FullDepthExpertCatalog,
    builder: Box<dyn RealK4Position0CommittedProductionBuilder>,
) -> Result<Box<dyn RealK4GateExecutable>> {
    let manifest = Arc::clone(&request.manifest);
    let weight_plan = Arc::clone(&request.weight_plan);
    let cache_root = request.cache_root.clone();
    let catalog_path = request.catalog_path.clone();
    let input_token_ids = request.input_token_ids;
    let draft_token_ids = request.draft_token_ids;
    let second_block_draft_token_ids = request.second_block_draft_token_ids;
    let seed = builder
        .build(request)
        .context("构造同源 position0→position1 K=4 production seed")?;
    let RealK4Position0ProductionSeed {
        context,
        authoritative,
        authoritative_device_state,
        paged_arena,
        head_upload,
        hidden_banks,
        ratio4_boundary_states,
        prefix_state_producer,
        terminal_owner,
        host_finalizer,
        union_banks,
        layer_program,
        input_planner,
        continuation,
        checkpoint_slots,
        external_owner,
    } = seed;

    validate_position0_production_seed(
        &context,
        &authoritative,
        &paged_arena,
        &terminal_owner,
        checkpoint_slots,
        input_token_ids[0],
    )?;
    let checkpoint_state_bytes = u64::try_from(authoritative.native_arena.len())
        .context("position1 authoritative native arena bytes 无法表示为u64")?;
    let shape = S14CausalBlockProductionBundleShape::new(
        BLOCK_SIZE,
        checkpoint_state_bytes,
        checkpoint_slots,
    )?;
    let provider_build = build_s14_base1_k4_production_hc_qkv_provider(S14Base1K4ProviderInputs {
        context: Arc::clone(&context),
        manifest: Arc::clone(&manifest),
        weight_plan: Arc::clone(&weight_plan),
        paged_arena: Arc::clone(&paged_arena),
        head_upload: Arc::clone(&head_upload),
        base_position: authoritative.position,
        authoritative: S14Base1K4AuthoritativeStateBinding {
            native_state: authoritative.native.clone(),
            device_state: authoritative_device_state,
        },
        input_token_ids,
        ratio4_boundary_states,
        prefix_state_producer,
    });
    let (provider, provider_external) = match provider_build {
        Ok(value) => value,
        Err(error) => {
            drop(terminal_owner);
            drop(host_finalizer);
            drop(hidden_banks);
            let external_cleanup = external_owner.destroy(&context);
            let reusable_cleanup = RealK4ReusableProduction {
                manifest,
                weight_plan,
                paged_arena,
                head_upload,
                layer_program,
                input_planner,
                cache_root,
                catalog_path,
                union_banks,
                continuation,
            }
            .destroy();
            let cleanup = merge_two_cleanups(external_cleanup, reusable_cleanup);
            return match cleanup {
                Ok(()) => Err(error.context("构造 concrete base1/K=4 HC/QKV provider")),
                Err(cleanup) => Err(anyhow!(
                    "构造 concrete base1/K=4 HC/QKV provider: {error:#}; cleanup: {cleanup:#}"
                )),
            };
        }
    };

    let bundle_inputs = S14CausalBlockProductionBundleInputs {
        context: Arc::clone(&context),
        shape,
        hc_qkv_provider: S14CausalBlockContextBound::new(Arc::clone(&context), provider),
        hidden_banks: S14CausalBlockContextBound::new(Arc::clone(&context), hidden_banks),
        catalog: bundle_catalog,
        cache_root: cache_root.clone(),
        // 用户已显式授权外网 Range；仅在真实在线 top-6 命中缺页时
        // 通过现有 206/Content-Range/SHA transport 补页，不下载整模。
        fetch_mode: DynamicPageFetchMode::ExplicitFetch,
        static_arena: S14CausalBlockContextBound::new(
            Arc::clone(&context),
            Arc::clone(&paged_arena),
        ),
    };
    let combined_external = Box::new(CombinedRealK4ExternalResourceOwner {
        provider: Some(provider_external),
        production: Some(external_owner),
    });
    let reusable = RealK4ReusableProduction {
        manifest,
        weight_plan,
        paged_arena: Arc::clone(&paged_arena),
        head_upload,
        layer_program,
        input_planner,
        cache_root,
        catalog_path,
        union_banks,
        continuation,
    };
    Ok(Box::new(RealK4GateInputs::<
        S14Base1K4ProductionHcQkvProvider,
    > {
        bundle_inputs,
        orchestrator_catalog,
        authoritative,
        terminal_owner,
        host_finalizer: Box::new(host_finalizer),
        external_owner: combined_external,
        input_token_ids,
        draft_token_ids,
        reusable,
        second_block_draft_token_ids,
    }))
}

fn validate_position0_production_seed(
    context: &Arc<VulkanContext>,
    authoritative: &DecoderStateV1,
    paged_arena: &Arc<S14Position0PagedWeightArena>,
    terminal_owner: &Arc<S14CausalBlockProductionTerminalResourceOwner>,
    checkpoint_slots: usize,
    expected_first_input: u32,
) -> Result<()> {
    authoritative.validate()?;
    let state_bytes = u64::try_from(authoritative.native_arena.len())?;
    if authoritative.position != BASE_POSITION
        || authoritative.commit_epoch != u64::from(BASE_POSITION)
        || authoritative.input_token_id != expected_first_input
        || checkpoint_slots == 0
        || terminal_owner.block_size() != BLOCK_SIZE
        || terminal_owner.checkpoint_state_bytes() != state_bytes
        || !Arc::ptr_eq(context, terminal_owner.context())
        || !Arc::ptr_eq(paged_arena, terminal_owner.paged_weight_arena())
    {
        bail!("position0 production seed 的 context/state/paged/prefix/terminal identity 漂移");
    }
    Ok(())
}

fn validate_real_asset_file(path: &Path, label: &str) -> Result<()> {
    if !path.is_file() {
        bail!("Polaris S14 {label} 不存在: {}", path.display());
    }
    Ok(())
}
