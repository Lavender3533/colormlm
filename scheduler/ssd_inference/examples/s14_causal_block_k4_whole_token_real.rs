//! Polaris S14 production position1 K=4 whole-token 唯一真门。
//!
//! 本 example 只装配 `build_s14_causal_block_production_bundle` 与
//! `execute_causal_block_full_depth_with_checkpoints`；不包含任何 synthetic provider、
//! 固定 BOS replay 或逐 token fallback。当 position1 production bootstrap 尚未提供真实
//! authoritative/hidden/KV/RoPE/prefix/terminal owner 时，入口必须 fail-closed。

use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{
    DecoderStateV1, GraphProfile, MaterializedTokenSource, Position0WholeTokenManifest,
    COMPRESS_RATIOS, EXPERTS_PER_TOKEN, EXPERT_PAGE_BYTES, FULL_DEPTH_LAYERS,
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
        S14CausalBlockSealedFuture, S14CausalBlockUnionBankBinding,
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
    s14_dynamic_page_cache_readiness::DynamicPageFetchMode,
    s14_dynamic_routed_page_plan::FullDepthExpertCatalog,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_runtime::{S14Runtime, S14RuntimeConfig},
    VulkanContext,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const BLOCK_SIZE: usize = 4;
const BASE_POSITION: u32 = 1;
const INPUT_TOKEN_IDS: [u32; BLOCK_SIZE] = [5, 223, 939, 21];
const DRAFT_TOKEN_IDS: [u32; BLOCK_SIZE] = [223, 939, 21, 695];
const INITIAL_HIDDEN_BANK: usize = 0;
const INITIAL_HIDDEN_GENERATION: u64 = 0;
const EXPECTED_FINAL_HIDDEN_GENERATION: u64 = (FULL_DEPTH_LAYERS.len() * 2) as u64;
const EXPECTED_ONLINE_TOP6_ROWS: usize = BLOCK_SIZE * FULL_DEPTH_LAYERS.len();

fn main() -> Result<()> {
    build_real_position1_k4_inputs(RealK4GateAssetPaths::production())?.run()
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
    union_banks: S14CausalBlockUnionBanks,
    authoritative: DecoderStateV1,
    terminal_owner: Arc<S14CausalBlockProductionTerminalResourceOwner>,
    host_finalizer: Box<dyn S14CausalBlockHostCandidateFinalizer>,
    external_owner: Box<dyn RealK4ExternalResourceOwner>,
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
    checkpoint_slots: usize,
    external_owner: Box<dyn RealK4ExternalResourceOwner>,
}

struct S14CommittedPosition0ProductionBuilder;

impl RealK4Position0CommittedProductionBuilder for S14CommittedPosition0ProductionBuilder {
    fn build(
        self: Box<Self>,
        request: RealK4Position0BootstrapRequest,
    ) -> Result<RealK4Position0ProductionSeed> {
        if request.input_token_ids != INPUT_TOKEN_IDS || request.draft_token_ids != DRAFT_TOKEN_IDS
        {
            bail!("position0 bootstrap 请求的 K=4 input/draft 合同漂移");
        }
        let mut config = S14RuntimeConfig::production_defaults();
        config.payload_root = request.cache_root.clone();
        let mut runtime = S14Runtime::load(config).context("加载 S14 production runtime")?;
        let mut session = runtime
            .new_session(0, 127)
            .context("创建 position0 production session")?;
        let position0 = runtime
            .step(&mut session)
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
            .into_position0_committed_causal_block_parts(session)
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
            INPUT_TOKEN_IDS,
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
            DRAFT_TOKEN_IDS.to_vec(),
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
            head_upload: Some(Arc::clone(&head_upload)),
            paged_arena: Some(Arc::clone(&paged_arena)),
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
            checkpoint_slots: BLOCK_SIZE,
            external_owner,
        })
    }
}

struct S14RealK4ProductionExternalOwner {
    ratio4_owners: Option<S14CausalBlockRatio4ProductionStateOwners>,
    hidden_owner: Option<S14CausalBlockK4InputHiddenOwner>,
    prefix_initialization: Option<S14CausalBlockPrefixInitializationOwner>,
    head_upload: Option<Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>>,
    paged_arena: Option<Arc<S14Position0PagedWeightArena>>,
}

impl RealK4ExternalResourceOwner for S14RealK4ProductionExternalOwner {
    fn destroy(mut self: Box<Self>, context: &Arc<VulkanContext>) -> Result<()> {
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
        if let Some(head_upload) = self.head_upload.take() {
            match Arc::try_unwrap(head_upload) {
                Ok(head_upload) => match head_upload.into_inner() {
                    Ok(state) => state.uploader.destroy(context),
                    Err(_) => failures.push("terminal head upload mutex poisoned".to_owned()),
                },
                Err(head_upload) => failures.push(format!(
                    "terminal head upload 仍被持有: refs={}",
                    Arc::strong_count(&head_upload)
                )),
            }
        }
        if let Some(paged_arena) = self.paged_arena.take() {
            match Arc::try_unwrap(paged_arena) {
                Ok(paged_arena) => paged_arena.destroy(context),
                Err(paged_arena) => failures.push(format!(
                    "paged arena 仍被持有: refs={}",
                    Arc::strong_count(&paged_arena)
                )),
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

fn run_real_position1_k4_gate<P>(inputs: RealK4GateInputs<P>) -> Result<()>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    let RealK4GateInputs {
        bundle_inputs,
        orchestrator_catalog,
        union_banks,
        authoritative,
        terminal_owner,
        host_finalizer,
        external_owner,
    } = inputs;
    let context = Arc::clone(&bundle_inputs.context);
    let mut bundle_inputs = Some(bundle_inputs);
    let mut terminal_owner = Some(terminal_owner);
    let mut host_finalizer = Some(host_finalizer);
    let mut bundle = None;

    let execution = (|| -> Result<()> {
        validate_unique_gate_inputs(
            bundle_inputs
                .as_ref()
                .context("K=4 gate bundle inputs 已被消费")?,
            &authoritative,
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

        let union_plan = union_banks
            .plan(BLOCK_SIZE)
            .context("取得 K=4 union bank plan")?;
        let bank_index = authoritative.commit_epoch as usize % 2;
        let union_buffer = union_banks
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
            &DRAFT_TOKEN_IDS,
            &INPUT_TOKEN_IDS,
            initial_hidden,
            MaterializedTokenSource::SpeculativeDraft,
            production,
        )
        .context("执行 S14 position1 K=4 real whole-token gate")?;

        validate_sealed_future(&sealed, &authoritative, initial_hidden)?;
        validate_production_evidence(evidence_owner.production_evidence_snapshot()?)?;
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
            "status=pass mode=polaris_s14_production_k4_whole_token_real base_position={} positions=[1,2,3,4] inputs={:?} drafts={:?} layers={} online_top6_rows={} grouped_moe_submits={} ratio4_boundary_position=3 compressed_indexer_position=4 checkpoints=4 batched_head_submits=1 serial_token_forward_calls=0 committed=false",
            BASE_POSITION,
            INPUT_TOKEN_IDS,
            DRAFT_TOKEN_IDS,
            FULL_DEPTH_LAYERS.len(),
            EXPECTED_ONLINE_TOP6_ROWS,
            FULL_DEPTH_LAYERS.len(),
        );
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
    union_banks.destroy(&context);
    let external_cleanup = external_owner
        .destroy(&context)
        .context("销毁 K=4 external production assets");
    merge_execution_and_cleanup(execution, bundle_cleanup, external_cleanup)
}

fn validate_unique_gate_inputs<P>(
    bundle_inputs: &S14CausalBlockProductionBundleInputs<P>,
    authoritative: &DecoderStateV1,
) -> Result<()>
where
    P: S14CausalBlockProductionHcQkvResourceProvider,
{
    authoritative.validate()?;
    let checkpoint_state_bytes = u64::try_from(authoritative.native_arena.len())
        .context("authoritative native arena bytes 无法表示为u64")?;
    let committed_position0 = authoritative.committed_tokens.last();
    if authoritative.position != BASE_POSITION
        || authoritative.commit_epoch != u64::from(BASE_POSITION)
        || authoritative.input_token_id != INPUT_TOKEN_IDS[0]
        || authoritative.active_fixed_bank != 1
        || authoritative.committed_tokens.len() != 1
        || committed_position0.is_none_or(|token| {
            token.position != 0 || token.input_token_id != 0 || token.predicted_token_id != 5
        })
        || bundle_inputs.shape.block_size() != BLOCK_SIZE
        || bundle_inputs.shape.checkpoint_state_bytes() != checkpoint_state_bytes
        || bundle_inputs.shape.checkpoint_slots() == 0
    {
        bail!(
            "K=4 真门必须从真实 position0 提交态开始: position=1 epoch=1 input=5 bank=1 token0=0->5"
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
) -> Result<()> {
    let layers = &sealed.layers;
    if layers.base_position != BASE_POSITION
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
    if device.base_position != BASE_POSITION
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
    if decision.accepted_prefix.as_slice() != DRAFT_TOKEN_IDS
        || decision.fallback_token_id.is_some()
        || !decision.rejected_draft_suffix.is_empty()
    {
        bail!("K=4 真门预测未闭合 N=8 真实轨迹 position1..4: expected={DRAFT_TOKEN_IDS:?}");
    }
    let selected = sealed.selected_prefix()?;
    let checkpoint = selected.checkpoint();
    if selected.accepted_tokens() != BLOCK_SIZE
        || selected.checkpoint_index() != BLOCK_SIZE - 1
        || checkpoint.position != BASE_POSITION + BLOCK_SIZE as u32
        || checkpoint.commit_epoch != authoritative.commit_epoch + BLOCK_SIZE as u64
        || checkpoint.input_token_id != DRAFT_TOKEN_IDS[BLOCK_SIZE - 1]
        || checkpoint.active_fixed_bank != authoritative.active_fixed_bank
        || checkpoint.committed_tokens.len() != authoritative.committed_tokens.len() + BLOCK_SIZE
    {
        bail!("K=4 longest-prefix host/device checkpoint 不是完整 position1..4");
    }
    Ok(())
}

fn validate_production_evidence(evidence: S14CausalBlockProductionEvidenceSnapshot) -> Result<()> {
    let expected_ratio4_layers = FULL_DEPTH_LAYERS
        .iter()
        .copied()
        .filter(|&layer| COMPRESS_RATIOS[usize::from(layer)] == 4)
        .collect::<Vec<_>>();
    let prefix = evidence
        .prefix_seal_receipt
        .context("K=4 production evidence 缺少 prefix seal receipt")?;
    if evidence.base_position != BASE_POSITION
        || evidence.block_size != BLOCK_SIZE
        || evidence.expected_ratio4_layers != expected_ratio4_layers
        || evidence.ratio4_layer_evidence.len() != expected_ratio4_layers.len()
        || evidence.position3_finalize_writeback_rollover_layers != expected_ratio4_layers
        || evidence.position4_indexer_attention_layers != expected_ratio4_layers
        || prefix.base_position != BASE_POSITION
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
) -> Result<Box<dyn RealK4GateExecutable>> {
    validate_real_asset_file(&paths.manifest, "position0 whole-token manifest")?;
    validate_real_asset_file(&paths.catalog, "FullDepth43 native top-6 catalog")?;
    if !paths.cache_root.is_dir() {
        bail!(
            "Polaris S14 real Range cache 目录不存在: {}",
            paths.cache_root.display()
        );
    }
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
            input_token_ids: INPUT_TOKEN_IDS,
            draft_token_ids: DRAFT_TOKEN_IDS,
        },
        bundle_catalog,
        orchestrator_catalog,
        Box::new(S14CommittedPosition0ProductionBuilder),
    )
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
        checkpoint_slots,
        external_owner,
    } = seed;

    validate_position0_production_seed(
        &context,
        &authoritative,
        &paged_arena,
        &terminal_owner,
        checkpoint_slots,
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
        manifest,
        weight_plan,
        paged_arena: Arc::clone(&paged_arena),
        head_upload,
        authoritative: S14Base1K4AuthoritativeStateBinding {
            native_state: authoritative.native.clone(),
            device_state: authoritative_device_state,
        },
        input_token_ids: INPUT_TOKEN_IDS,
        ratio4_boundary_states,
        prefix_state_producer,
    });
    let (provider, provider_external) = match provider_build {
        Ok(value) => value,
        Err(error) => {
            drop(terminal_owner);
            drop(paged_arena);
            union_banks.destroy(&context);
            let cleanup = external_owner.destroy(&context);
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
        cache_root,
        fetch_mode: DynamicPageFetchMode::LocalOnly,
        static_arena: S14CausalBlockContextBound::new(
            Arc::clone(&context),
            Arc::clone(&paged_arena),
        ),
    };
    let combined_external = Box::new(CombinedRealK4ExternalResourceOwner {
        provider: Some(provider_external),
        production: Some(external_owner),
    });
    Ok(Box::new(RealK4GateInputs::<
        S14Base1K4ProductionHcQkvProvider,
    > {
        bundle_inputs,
        orchestrator_catalog,
        union_banks,
        authoritative,
        terminal_owner,
        host_finalizer: Box::new(host_finalizer),
        external_owner: combined_external,
    }))
}

fn validate_position0_production_seed(
    context: &Arc<VulkanContext>,
    authoritative: &DecoderStateV1,
    paged_arena: &Arc<S14Position0PagedWeightArena>,
    terminal_owner: &Arc<S14CausalBlockProductionTerminalResourceOwner>,
    checkpoint_slots: usize,
) -> Result<()> {
    authoritative.validate()?;
    let state_bytes = u64::try_from(authoritative.native_arena.len())?;
    if authoritative.position != BASE_POSITION
        || authoritative.commit_epoch != u64::from(BASE_POSITION)
        || authoritative.input_token_id != INPUT_TOKEN_IDS[0]
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
