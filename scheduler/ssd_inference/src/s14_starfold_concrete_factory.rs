//! 纯 S14 StarFold 的 concrete K4 block resource factory。
//!
//! `S14Runtime` 与请求级 `S14Session` 始终只由 production root/session 拥有；
//! concrete factory 只保留 manifest/weight plan/layer program/local embedding shard/head uploader 等
//! block assets，并且只从 request 携带的同源 authoritative/device checkpoint 构造。
//! 本模块不执行 token，不存在 union/grouped-MoE、CPU、serial-token 或 Transformer fallback。

use crate::{
    s14_causal_block_base1_k4_provider::{
        build_s14_base1_k4_production_hc_qkv_provider, S14Base1K4AuthoritativeStateBinding,
        S14Base1K4HcQkvExternalResources, S14Base1K4ProductionHcQkvProvider,
        S14Base1K4ProviderInputs,
    },
    s14_causal_block_hc_qkv_adapter::S14Position0CommittedGenerationProvenance,
    s14_causal_block_hc_qkv_recorder::S14CausalBlockOwnedBufferSlice,
    s14_causal_block_k4_input_hidden::S14CausalBlockK4InputHiddenOwner,
    s14_causal_block_prefix_initializer::S14CausalBlockPrefixInitializationOwner,
    s14_causal_block_prefix_state::S14CausalBlockPrefixStateProgram,
    s14_causal_block_production_bundle::S14CausalBlockContextBound,
    s14_causal_block_ratio4_state_owner::S14CausalBlockRatio4ProductionStateOwners,
    s14_causal_block_terminal_owner::S14CausalBlockTerminalHeadUploadState,
    s14_local_embedding_shard::S14LocalEmbeddingShard,
    s14_position0_hybrid_upload::S14Position0CausalBlockUploadLease,
    s14_position0_layer_program::S14Position0FullDepthLayerProgram,
    s14_position0_paged_weight_arena::{S14Position0PagedArenaPlan, S14Position0PagedWeightArena},
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_runtime::{S14Runtime, S14RuntimeCausalBlockWeightStorage, S14RuntimeConfig},
    s14_starfold_k4_rebind::S14StarfoldK4BlockMode,
    s14_starfold_production_resources::S14StarfoldK4ProductionResourceInputs,
    s14_starfold_production_session::{
        S14StarfoldBlockExternalOwners, S14StarfoldBlockOwnerInventory,
        S14StarfoldBlockResourceFactory, S14StarfoldBlockResourceProduct,
        S14StarfoldBlockResourceRequest, S14StarfoldForbiddenPathCounters,
        S14StarfoldProductionRoot, S14StarfoldResidentBlockFactoryInventory,
    },
    s14_starwave_draft::validate_s14_starwave_generation_origin,
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{
    DecoderStateV1, MaterializedTokenSource, Position0WholeTokenManifest, VOCAB_SIZE,
};
use std::sync::{Arc, Mutex, Weak};

pub const S14_STARFOLD_CONCRETE_K: usize = 4;
pub const S14_STARFOLD_INITIAL_HIDDEN_GENERATION: u64 = 0;

/// Provider 从 adapter 退出之后仍必须显式按顺序销毁的 block backing owners。
#[must_use = "adapter rebind/drop provider 后必须显式 destroy concrete block external owners"]
pub struct S14StarfoldConcreteBlockResources {
    context: Arc<VulkanContext>,
    provider_external: Option<S14Base1K4HcQkvExternalResources>,
    ratio4_owners: Option<S14CausalBlockRatio4ProductionStateOwners>,
    hidden_owner: Option<S14CausalBlockK4InputHiddenOwner>,
    prefix_initialization: Option<S14CausalBlockPrefixInitializationOwner>,
}

impl S14StarfoldBlockResourceFactory for S14StarfoldConcreteFactory {
    type Provider = S14Base1K4ProductionHcQkvProvider;
    type ExternalOwners = S14StarfoldConcreteBlockResources;

    fn resident_owner_inventory(
        &self,
        context: &Arc<VulkanContext>,
        paged_arena: &Arc<S14Position0PagedWeightArena>,
    ) -> Result<S14StarfoldResidentBlockFactoryInventory> {
        let bound_context = self
            .context
            .upgrade()
            .context("S14 concrete resident context owner 已退出")?;
        let bound_arena = self
            .paged_arena
            .upgrade()
            .context("S14 concrete resident paged arena owner 已退出")?;
        if !Arc::ptr_eq(&bound_context, context) || !Arc::ptr_eq(&bound_arena, paged_arena) {
            bail!("concrete resident inventory 与 production runtime 不同源");
        }
        Ok(S14StarfoldResidentBlockFactoryInventory {
            context_bindings: usize::from(self.context.strong_count() != 0),
            paged_arena_bindings: usize::from(self.paged_arena.strong_count() != 0),
            verified_mapped_asset_store_owners: usize::from(self.head_upload.is_some()),
            terminal_head_uploader_owners: usize::from(self.head_upload.is_some()),
            request_owned: S14StarfoldBlockOwnerInventory::default(),
            forbidden: self.forbidden,
        })
    }

    fn retire_persistent_owners(&mut self, context: &VulkanContext) -> Result<()> {
        let bound = self
            .context
            .upgrade()
            .context("S14 concrete persistent context owner 已退出")?;
        if !std::ptr::eq(bound.as_ref(), context) {
            bail!("concrete persistent owners 与 production context 不同源");
        }
        S14StarfoldConcreteFactory::retire_persistent_owners(self)
    }

    fn rearm_for_new_request(&mut self, context: &VulkanContext) -> Result<()> {
        let bound = self
            .context
            .upgrade()
            .context("S14 concrete persistent context owner 已退出")?;
        if !std::ptr::eq(bound.as_ref(), context) {
            bail!("concrete request rearm 与 production context 不同源");
        }
        self.rearm_request_lineage()
    }

    fn build_block(
        &mut self,
        mut request: S14StarfoldBlockResourceRequest<'_>,
    ) -> Result<S14StarfoldBlockResourceProduct<Self::Provider, Self::ExternalOwners>> {
        let context = self
            .context
            .upgrade()
            .context("S14 concrete context owner 已退出")?;
        let paged_arena = self
            .paged_arena
            .upgrade()
            .context("S14 concrete paged arena owner 已退出")?;
        if !Arc::ptr_eq(&context, request.context)
            || !Arc::ptr_eq(&paged_arena, request.paged_arena)
        {
            bail!("concrete factory request 与启动期 runtime context/paged arena 不同源");
        }
        let authoritative = request.authoritative.clone();
        let state_bytes = u64::try_from(authoritative.native_arena.len())
            .context("S14 concrete authoritative state bytes overflow")?;
        let snapshot_identity = request.committed_snapshot.identity()?;
        if request.committed_device.state_bytes() != state_bytes
            || request.committed_device.epoch() != authoritative.commit_epoch
            || request.committed_device.active_bank()
                != usize::from(authoritative.active_fixed_bank)
            || snapshot_identity
                != (
                    request.committed_device.state_bytes(),
                    request.committed_device.epoch(),
                    request.committed_device.active_bank(),
                )
        {
            bail!("concrete factory authoritative/committed binding/detached snapshot 身份漂移");
        }
        match request.mode {
            S14StarfoldK4BlockMode::TeacherForcedPrefill => {
                if request.position0_committed_origin.is_some() {
                    bail!("ForcedPrefill 禁止携带 generation position0 committed-origin");
                }
            }
            S14StarfoldK4BlockMode::SpeculativeGeneration => {
                validate_s14_starwave_generation_origin(
                    &authoritative,
                    request.position0_committed_origin,
                )
                .map_err(|error| anyhow!(error.to_string()))?;
            }
        }
        let source = request.mode.materialized_source();
        validate_block_request(
            &authoritative,
            &request.input_token_ids,
            request.expected_initial_hidden_generation,
        )?;
        let upload_lease =
            self.prepare_persistent_uploader(&authoritative, &source, request.block_sequence)?;
        let generation_position0_provenance = if request.mode
            == S14StarfoldK4BlockMode::SpeculativeGeneration
            && authoritative.position == 0
        {
            Some(
                S14Position0CommittedGenerationProvenance::validate(
                    &authoritative,
                    request.position0_committed_origin,
                )
                .map_err(anyhow::Error::msg)
                .context("构造 concrete base0 committed-generation provenance")?,
            )
        } else {
            None
        };

        let committed = request
            .committed_snapshot
            .take()
            .context("消费 request-owned detached committed snapshot")?;
        let mut prefix_initialization = match (request.mode, authoritative.position) {
            (S14StarfoldK4BlockMode::TeacherForcedPrefill, 0) => {
                S14CausalBlockPrefixInitializationOwner::initialize_forced_prefill(
                    Arc::clone(&context),
                    authoritative.clone(),
                    committed,
                )
            }
            (S14StarfoldK4BlockMode::TeacherForcedPrefill, _) => {
                S14CausalBlockPrefixInitializationOwner::initialize_forced_prefill_at(
                    Arc::clone(&context),
                    authoritative.clone(),
                    committed,
                )
            }
            (S14StarfoldK4BlockMode::SpeculativeGeneration, 0) => {
                S14CausalBlockPrefixInitializationOwner::initialize_generation_position0(
                    Arc::clone(&context),
                    authoritative.clone(),
                    committed,
                    generation_position0_provenance
                        .expect("validated above")
                        .origin(),
                )
            }
            (S14StarfoldK4BlockMode::SpeculativeGeneration, _) => {
                S14CausalBlockPrefixInitializationOwner::initialize_at(
                    Arc::clone(&context),
                    authoritative.clone(),
                    committed,
                    authoritative.position,
                    S14_STARFOLD_CONCRETE_K,
                )
            }
        }
        .context("构造 S14 concrete prefix initialization owner")?;

        let assembled = self.assemble_block_after_prefix(
            Arc::clone(&context),
            authoritative.clone(),
            request.input_token_ids,
            source.clone(),
            upload_lease,
            generation_position0_provenance,
            request.expected_initial_hidden_generation,
            paged_arena,
            &mut prefix_initialization,
        );
        let mut product = match assembled {
            Ok(product) => product,
            Err(error) => {
                let cleanup = prefix_initialization.destroy();
                return Err(anyhow!(
                    "装配 S14 concrete K4 block: {error:#}; prefix cleanup={cleanup:?}"
                ));
            }
        };
        product.external_owners.prefix_initialization = Some(prefix_initialization);
        self.prepared_blocks = self
            .prepared_blocks
            .checked_add(1)
            .context("S14 concrete prepared block sequence overflow")?;
        self.last_prepared_block_sequence = Some(request.block_sequence);
        self.last_prepared_base_position = Some(authoritative.position);
        self.last_prepared_source = Some(source);
        if self.forbidden != S14StarfoldForbiddenPathCounters::default() {
            let S14StarfoldBlockResourceProduct {
                production,
                initial_hidden: _,
                mut external_owners,
            } = product;
            drop(production);
            let cleanup = external_owners.destroy(&context);
            bail!("concrete factory 禁止路径计数非零; cleanup={cleanup:?}");
        }
        Ok(product)
    }
}

impl S14StarfoldConcreteBlockResources {
    /// 严格退休顺序：provider 的 aux/current-KV → ratio4 shared core → hidden A/B →
    /// prefix arena/authoritative snapshot。每一步失败都保留未退休 owner，可在真正
    /// release/drain 后重试，不把忙碌 Vulkan allocation 冒充已释放。
    fn retire_after_stage_release(&mut self) -> Result<()> {
        destroy_option_owner(
            &mut self.provider_external,
            |owner| owner.destroy(),
            "HC/QKV external resources",
        )?;
        destroy_option_owner(
            &mut self.ratio4_owners,
            |owner| owner.destroy(),
            "ratio4 production states",
        )?;
        destroy_option_owner(
            &mut self.hidden_owner,
            |owner| owner.destroy(),
            "K4 hidden banks",
        )?;
        destroy_option_owner(
            &mut self.prefix_initialization,
            |owner| owner.destroy(),
            "prefix initialization",
        )
    }

    pub fn retired(&self) -> bool {
        self.provider_external.is_none()
            && self.ratio4_owners.is_none()
            && self.hidden_owner.is_none()
            && self.prefix_initialization.is_none()
    }
}

impl std::fmt::Debug for S14StarfoldConcreteBlockResources {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("S14StarfoldConcreteBlockResources")
            .field("context", &Arc::as_ptr(&self.context))
            .field("provider_external", &self.provider_external.is_some())
            .field("ratio4_owners", &self.ratio4_owners.is_some())
            .field("hidden_owner", &self.hidden_owner.is_some())
            .field(
                "prefix_initialization",
                &self.prefix_initialization.is_some(),
            )
            .finish()
    }
}

impl S14StarfoldBlockExternalOwners for S14StarfoldConcreteBlockResources {
    fn inventory(&self) -> S14StarfoldBlockOwnerInventory {
        S14StarfoldBlockOwnerInventory {
            provider_owners: usize::from(self.provider_external.is_some()),
            hidden_bank_owners: usize::from(self.hidden_owner.is_some()) * 2,
            prefix_arena_owners: usize::from(self.prefix_initialization.is_some()),
            ratio4_state_owners: self
                .ratio4_owners
                .as_ref()
                .map_or(0, |owners| owners.state_owner_count()),
            upload_auxiliary_owners: usize::from(self.provider_external.is_some()),
        }
    }

    fn destroy(&mut self, context: &VulkanContext) -> Result<()> {
        if !std::ptr::eq(self.context.as_ref(), context) {
            bail!("concrete external owners 与 production context 不同源");
        }
        self.retire_after_stage_release()
    }
}

/// 常驻 block-assets factory。runtime/session authority 始终不进入本类型。
#[must_use = "必须移入 production root/session，并在请求结束后退休 head uploader"]
pub struct S14StarfoldConcreteFactory {
    context: Weak<VulkanContext>,
    paged_arena: Weak<S14Position0PagedWeightArena>,
    manifest: Arc<Position0WholeTokenManifest>,
    weight_plan: Arc<S14Position0HybridWeightPlan>,
    layer_program: S14Position0FullDepthLayerProgram,
    embedding_shard: S14LocalEmbeddingShard,
    head_upload: Option<Arc<Mutex<S14CausalBlockTerminalHeadUploadState>>>,
    uploader_prepared_block_sequence: Option<u64>,
    uploader_prepared_base_position: Option<u32>,
    uploader_prepared_source: Option<MaterializedTokenSource>,
    prepared_blocks: u64,
    last_prepared_block_sequence: Option<u64>,
    last_prepared_base_position: Option<u32>,
    last_prepared_source: Option<MaterializedTokenSource>,
    forbidden: S14StarfoldForbiddenPathCounters,
}

impl S14StarfoldConcreteFactory {
    /// 只加载 block assets，并与外部唯一 runtime 的 context/paged arena 做强同源绑定。
    pub fn load_for_runtime(config: &S14RuntimeConfig, runtime: &mut S14Runtime) -> Result<Self> {
        match config.causal_block_weight_storage {
            S14RuntimeCausalBlockWeightStorage::StarfoldDoubleMicrotile { microtile_bytes }
                if microtile_bytes > 0 => {}
            _ => bail!("S14 concrete factory 只接受 StarFold double-microtile production config"),
        }

        let manifest = Arc::new(
            Position0WholeTokenManifest::load(&config.manifest_path).with_context(|| {
                format!(
                    "加载 S14 concrete manifest: {}",
                    config.manifest_path.display()
                )
            })?,
        );
        manifest
            .validate()
            .map_err(|error| anyhow!("S14 concrete manifest 非法: {error}"))?;
        let weight_plan = Arc::new(S14Position0HybridWeightPlan::build(&manifest)?);
        weight_plan.validate(&manifest)?;
        let embedding_shard = S14LocalEmbeddingShard::open_verified(&config.embedding_shard_path)
            .context("加载 S14 concrete 本地 embedding shard")?;
        let context = Arc::clone(runtime.context());
        let paged_arena = Arc::clone(runtime.paged_arena()?);
        let expected_arena = S14Position0PagedArenaPlan::build(&weight_plan)?;
        if paged_arena.plan() != &expected_arena {
            bail!("S14 runtime 与 concrete manifest/weight paged arena 不同源");
        }
        let layer_program = S14Position0FullDepthLayerProgram::build(
            &manifest,
            &weight_plan,
            paged_arena.workspace_layout(),
        )
        .context("构造 S14 concrete FullDepth43 layer program")?;
        let (uploader, mapped_store) = runtime
            .take_starfold_upload_parts()
            .context("消费唯一 runtime terminal uploader/verified mapped store")?;
        let head_upload = Arc::new(Mutex::new(S14CausalBlockTerminalHeadUploadState {
            uploader,
            store: mapped_store,
        }));
        Ok(Self {
            context: Arc::downgrade(&context),
            paged_arena: Arc::downgrade(&paged_arena),
            manifest,
            weight_plan,
            layer_program,
            embedding_shard,
            head_upload: Some(head_upload),
            uploader_prepared_block_sequence: None,
            uploader_prepared_base_position: None,
            uploader_prepared_source: None,
            prepared_blocks: 0,
            last_prepared_block_sequence: None,
            last_prepared_base_position: None,
            last_prepared_source: None,
            forbidden: S14StarfoldForbiddenPathCounters::default(),
        })
    }

    /// 启动期唯一 loader：runtime 只 load 一次，factory 只从借用的 runtime 绑定 assets。
    pub fn load_production_root(
        config: S14RuntimeConfig,
    ) -> Result<S14StarfoldProductionRoot<Self>> {
        let mut runtime =
            S14Runtime::load(config.clone()).context("加载唯一 S14 StarFold runtime")?;
        match Self::load_for_runtime(&config, &mut runtime) {
            Ok(factory) => Ok(S14StarfoldProductionRoot::new(runtime, factory)),
            Err(error) => {
                let cleanup = runtime.destroy();
                Err(anyhow!(
                    "构造 S14 concrete block assets: {error:#}; runtime cleanup={cleanup:?}"
                ))
            }
        }
    }

    fn assemble_block_after_prefix(
        &mut self,
        context: Arc<VulkanContext>,
        authoritative: DecoderStateV1,
        input_token_ids: [u32; S14_STARFOLD_CONCRETE_K],
        source: MaterializedTokenSource,
        upload_lease: S14Position0CausalBlockUploadLease,
        position0_generation_provenance: Option<S14Position0CommittedGenerationProvenance>,
        hidden_generation: u64,
        paged_arena: Arc<S14Position0PagedWeightArena>,
        prefix_initialization: &mut S14CausalBlockPrefixInitializationOwner,
    ) -> Result<
        S14StarfoldBlockResourceProduct<
            S14Base1K4ProductionHcQkvProvider,
            S14StarfoldConcreteBlockResources,
        >,
    > {
        let prefix_arena = Arc::clone(prefix_initialization.prefix_arena()?);
        let prefix_program = Arc::new(Mutex::new(S14CausalBlockPrefixStateProgram::build(
            &self.layer_program,
            paged_arena.workspace_layout(),
            &authoritative.native,
            S14_STARFOLD_CONCRETE_K,
        )?));
        let mut prefix_producer = prefix_initialization
            .build_prefix_state_producer(Arc::clone(&prefix_program))
            .context("构造 S14 concrete prefix state producer")?;
        let mut ratio4_owners = match S14CausalBlockRatio4ProductionStateOwners::build(
            Arc::clone(&context),
            prefix_arena,
            prefix_program,
            &self.layer_program,
            &authoritative.native,
        ) {
            Ok(owners) => owners,
            Err(error) => {
                let cleanup = prefix_producer.destroy();
                return Err(anyhow!(
                    "构造 S14 concrete ratio4 states: {error:#}; producer cleanup={cleanup:?}"
                ));
            }
        };
        let ratio4_boundary_states = ratio4_owners.trait_states();

        let head_upload = Arc::clone(
            self.head_upload
                .as_ref()
                .context("S14 concrete terminal head uploader 已退休")?,
        );
        let mut hidden_owner = {
            match S14CausalBlockK4InputHiddenOwner::build_at_from_local_shard(
                Arc::clone(&context),
                &self.embedding_shard,
                authoritative.position,
                authoritative.native.max_seq_len,
                input_token_ids,
                hidden_generation,
            ) {
                Ok(owner) => owner,
                Err(error) => {
                    drop(ratio4_boundary_states);
                    let producer_cleanup = prefix_producer.destroy();
                    let ratio_cleanup = ratio4_owners.destroy();
                    return Err(anyhow!(
                        "构造 S14 concrete input hidden: {error:#}; producer={producer_cleanup:?}; ratio4={ratio_cleanup:?}"
                    ));
                }
            }
        };
        let (hidden_banks, initial_hidden) = match (
            hidden_owner.hidden_banks(),
            hidden_owner.binding(),
        ) {
            (Ok(banks), Ok(binding)) => (banks, binding),
            (banks, binding) => {
                drop(ratio4_boundary_states);
                let producer_cleanup = prefix_producer.destroy();
                let ratio_cleanup = ratio4_owners.destroy();
                let hidden_cleanup = hidden_owner.destroy();
                return Err(anyhow!(
                    "导出 S14 concrete hidden owners: banks={banks:?}; binding={binding:?}; producer={producer_cleanup:?}; ratio4={ratio_cleanup:?}; hidden={hidden_cleanup:?}"
                ));
            }
        };
        let authoritative_device = match prefix_initialization.authoritative_device_state() {
            Ok(binding) => binding,
            Err(error) => {
                drop(ratio4_boundary_states);
                let producer_cleanup = prefix_producer.destroy();
                let ratio_cleanup = ratio4_owners.destroy();
                let hidden_cleanup = hidden_owner.destroy();
                return Err(anyhow!(
                    "导出 S14 authoritative device binding: {error:#}; producer={producer_cleanup:?}; ratio4={ratio_cleanup:?}; hidden={hidden_cleanup:?}"
                ));
            }
        };
        let authoritative_device_state = S14CausalBlockOwnedBufferSlice {
            buffer: authoritative_device.buffer,
            offset: authoritative_device.offset,
            bytes: u64::try_from(authoritative.native_arena.len())?,
        };
        let provider = build_s14_base1_k4_production_hc_qkv_provider(S14Base1K4ProviderInputs {
            context: Arc::clone(&context),
            manifest: Arc::clone(&self.manifest),
            weight_plan: Arc::clone(&self.weight_plan),
            paged_arena,
            head_upload,
            base_position: authoritative.position,
            source: source.clone(),
            upload_lease,
            authoritative: S14Base1K4AuthoritativeStateBinding {
                native_state: authoritative.native.clone(),
                device_state: authoritative_device_state,
            },
            input_token_ids,
            ratio4_boundary_states,
            prefix_state_producer: prefix_producer,
        });
        let (provider, provider_external) = match provider {
            Ok(value) => value,
            Err(error) => {
                let hidden_cleanup = hidden_owner.destroy();
                let ratio_cleanup = ratio4_owners.destroy();
                return Err(anyhow!(
                    "构造 S14 concrete HC/QKV provider: {error:#}; hidden={hidden_cleanup:?}; ratio4={ratio_cleanup:?}"
                ));
            }
        };
        Ok(S14StarfoldBlockResourceProduct {
            initial_hidden,
            production: S14StarfoldK4ProductionResourceInputs {
                hc_qkv_provider: S14CausalBlockContextBound::new(Arc::clone(&context), provider),
                hidden_banks: S14CausalBlockContextBound::new(Arc::clone(&context), hidden_banks),
                position0_generation_provenance,
            },
            external_owners: S14StarfoldConcreteBlockResources {
                context,
                provider_external: Some(provider_external),
                ratio4_owners: Some(ratio4_owners),
                hidden_owner: Some(hidden_owner),
                prefix_initialization: None,
            },
        })
    }

    fn prepare_persistent_uploader(
        &mut self,
        authoritative: &DecoderStateV1,
        source: &MaterializedTokenSource,
        block_sequence: u64,
    ) -> Result<S14Position0CausalBlockUploadLease> {
        if self.uploader_prepared_block_sequence == Some(block_sequence) {
            if self.uploader_prepared_base_position != Some(authoritative.position)
                || self.uploader_prepared_source.as_ref() != Some(source)
            {
                bail!("同一 block sequence 重试时禁止切换 authoritative position/source");
            }
            let mut head = self
                .head_upload
                .as_ref()
                .context("S14 concrete terminal head uploader 已退休")?
                .lock()
                .map_err(|_| anyhow!("S14 concrete terminal head uploader mutex poisoned"))?;
            return head
                .uploader
                .issue_causal_block_upload_lease(
                    &self.weight_plan,
                    block_sequence,
                    authoritative.position,
                    *source,
                )
                .context("同一 block sequence 重取 uploader lease");
        }
        if self.prepared_blocks == 0 {
            if block_sequence != 0 || authoritative.position != 0 {
                bail!("首个 concrete block 必须是 sequence0/base0");
            }
        } else {
            let previous_base = self
                .last_prepared_base_position
                .context("S14 concrete previous block base 缺失")?;
            if authoritative.position <= previous_base {
                bail!("上一 concrete block 尚未发布新的 authoritative position");
            }
            let previous_sequence = self
                .last_prepared_block_sequence
                .context("S14 concrete previous block sequence 缺失")?;
            if self.uploader_prepared_block_sequence != Some(previous_sequence) {
                bail!("concrete uploader/prepared block sequence lineage 漂移");
            }
            let expected_sequence = previous_sequence
                .checked_add(1)
                .context("S14 concrete block sequence overflow")?;
            if block_sequence != expected_sequence || block_sequence != self.prepared_blocks {
                bail!("concrete block sequence 未按 production adapter lineage 单调推进");
            }
        }
        let mut head = self
            .head_upload
            .as_ref()
            .context("S14 concrete terminal head uploader 已退休")?
            .lock()
            .map_err(|_| anyhow!("S14 concrete terminal head uploader mutex poisoned"))?;
        let lease = head
            .uploader
            .issue_causal_block_upload_lease(
                &self.weight_plan,
                block_sequence,
                authoritative.position,
                *source,
            )
            .context("签发 concrete block-scoped uploader lease")?;
        drop(head);
        self.uploader_prepared_block_sequence = Some(block_sequence);
        self.uploader_prepared_base_position = Some(authoritative.position);
        self.uploader_prepared_source = Some(*source);
        Ok(lease)
    }

    /// 只在请求级 stage/provider/external owners 已全部退出后调用。它复用同一个
    /// terminal head uploader，并把 block lineage 清回新请求的 sequence0/base0。
    fn rearm_request_lineage(&mut self) -> Result<()> {
        if self.prepared_blocks == 0 {
            return Ok(());
        }
        let previous_sequence = self
            .last_prepared_block_sequence
            .context("request rearm 缺少上一 block sequence")?;
        if self.uploader_prepared_block_sequence != Some(previous_sequence) {
            bail!("request rearm 的 uploader/block sequence lineage 漂移");
        }
        let mut head = self
            .head_upload
            .as_ref()
            .context("S14 concrete terminal head uploader 已退休")?
            .lock()
            .map_err(|_| anyhow!("S14 concrete terminal head uploader mutex poisoned"))?;
        head.uploader
            .reset_causal_block_request_lineage(&self.weight_plan)
            .context("新请求前关闭上一 block uploader lease lineage")?;
        drop(head);
        self.uploader_prepared_block_sequence = None;
        self.uploader_prepared_base_position = None;
        self.uploader_prepared_source = None;
        self.prepared_blocks = 0;
        self.last_prepared_block_sequence = None;
        self.last_prepared_base_position = None;
        self.last_prepared_source = None;
        Ok(())
    }

    /// 只有 production resources/current block owners 已全部退出后才调用。
    pub fn retire_persistent_owners(&mut self) -> Result<()> {
        let context = self
            .context
            .upgrade()
            .context("S14 concrete context owner 已退出")?;
        let head = self
            .head_upload
            .as_ref()
            .context("S14 concrete persistent owners 已退休")?;
        if Arc::strong_count(head) != 1 {
            bail!(
                "terminal head uploader 仍被 StarFold block 持有: refs={}",
                Arc::strong_count(head)
            );
        }
        let head = Arc::try_unwrap(
            self.head_upload
                .take()
                .context("S14 concrete head uploader owner 漂移")?,
        )
        .map_err(|head| {
            let refs = Arc::strong_count(&head);
            self.head_upload = Some(head);
            anyhow!("terminal head uploader Arc ownership 漂移: refs={refs}")
        })?;
        let head = head
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        head.uploader.destroy(&context);
        drop(head.store);
        Ok(())
    }
}

impl Drop for S14StarfoldConcreteFactory {
    fn drop(&mut self) {
        let Some(head) = self.head_upload.take() else {
            return;
        };
        if Arc::strong_count(&head) != 1 {
            // provider 仍活跃时不能提前销毁；其余强 owner 会在 VulkanContext teardown 前
            // 由 production root 显式退休。Drop 不伪造成功。
            self.head_upload = Some(head);
            return;
        }
        if let Ok(head) = Arc::try_unwrap(head) {
            let head = head
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(context) = self.context.upgrade() {
                head.uploader.destroy(&context);
            }
        }
    }
}

fn validate_block_request(
    authoritative: &DecoderStateV1,
    input_token_ids: &[u32; S14_STARFOLD_CONCRETE_K],
    hidden_generation: u64,
) -> Result<()> {
    authoritative
        .validate()
        .map_err(|error| anyhow!("S14 concrete authoritative state 非法: {error}"))?;
    let end = authoritative
        .position
        .checked_add(S14_STARFOLD_CONCRETE_K as u32)
        .context("S14 concrete K4 position overflow")?;
    if input_token_ids[0] != authoritative.input_token_id
        || input_token_ids.iter().any(|&token| token >= VOCAB_SIZE)
        || end > authoritative.native.max_seq_len
        || (authoritative.position == 0
            && hidden_generation != S14_STARFOLD_INITIAL_HIDDEN_GENERATION)
    {
        bail!("S14 concrete K4 input/source/base/max_seq/hidden generation 合同漂移");
    }
    Ok(())
}

fn destroy_option_owner<T>(
    slot: &mut Option<T>,
    destroy: impl FnOnce(&mut T) -> Result<()>,
    label: &str,
) -> Result<()> {
    let Some(owner) = slot.as_mut() else {
        return Ok(());
    };
    destroy(owner).with_context(|| format!("退休 S14 concrete {label}"))?;
    slot.take();
    Ok(())
}
