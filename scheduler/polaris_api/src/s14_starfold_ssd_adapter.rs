//! `ssd_inference` owning StarFold production root/session 到 API commit ABI 的独立适配层。
//!
//! 本模块不构造模型资源、不生成 token，也不根据错误字符串分类。SSD 暂时返回
//! `anyhow::Error` 的调用只会按调用阶段收敛为 `Internal`；可判定的 lease/输入/position
//! 状态使用本模块的 typed error。startup resource contract 没有 default/mock 实现。

use crate::s14_engine::{
    S14ChatCodec, S14ResidentK4ResourceInventory, VerifiedS14ResidentK4Resources,
};
use crate::s14_starfold_commit_provider::{
    build_s14_starfold_production_chat_backend, S14StarfoldK4CommitLimit,
    S14StarfoldProductionChatBackend, S14StarfoldProductionSession,
    S14StarfoldProductionSessionFactory, S14StarfoldRuntimeCheckpointReceipt,
    S14StarfoldRuntimeCommittedBlockReceipt,
};
use crate::{
    EngineError, EngineErrorKind, ResidentChatBackend, ResidentChatEngine,
    S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS,
};
use ssd_inference::s14_starfold_concrete_factory::S14StarfoldConcreteFactory;
use ssd_inference::s14_starfold_production_session::{
    S14StarfoldBlockOwnerInventory, S14StarfoldBlockResourceFactory, S14StarfoldCheckpointIdentity,
    S14StarfoldCommittedLedgerDelta, S14StarfoldForbiddenPathCounters,
    S14StarfoldProductionLeaseState, S14StarfoldProductionResourceInventory,
    S14StarfoldProductionRoot, S14StarfoldProductionSession as SsdS14StarfoldProductionSession,
    S14StarfoldRequestOwnerReadiness, S14StarfoldResidentResourceContract,
};
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt,
    rc::Rc,
};

const SSD_STARFOLD_PHYSICAL_BLOCK_TOKENS: u32 = 4;

/// SSD→API 分类。分类只来自 typed 状态或明确的 adapter 前置条件，不解析错误文本。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldSsdAdapterErrorKind {
    Invalid,
    Busy,
    Exhausted,
    Unsupported,
    Internal,
}

/// 标记失败发生在哪个不可混淆的适配阶段。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldSsdAdapterStage {
    WorkerLoad,
    StartupContract,
    RootLease,
    PromptPrefill,
    BeginCheckpoint,
    SessionInventory,
    BeforeCommitCheckpoint,
    AtomicCommit,
    CommitReceipt,
    AfterCommitCheckpoint,
    Close,
}

impl S14StarfoldSsdAdapterStage {
    const fn name(self) -> &'static str {
        match self {
            Self::WorkerLoad => "worker_load",
            Self::StartupContract => "startup_contract",
            Self::RootLease => "root_lease",
            Self::PromptPrefill => "prompt_prefill",
            Self::BeginCheckpoint => "begin_checkpoint",
            Self::SessionInventory => "session_inventory",
            Self::BeforeCommitCheckpoint => "before_commit_checkpoint",
            Self::AtomicCommit => "atomic_commit",
            Self::CommitReceipt => "commit_receipt",
            Self::AfterCommitCheckpoint => "after_commit_checkpoint",
            Self::Close => "close",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldSsdAdapterError {
    pub kind: S14StarfoldSsdAdapterErrorKind,
    pub stage: S14StarfoldSsdAdapterStage,
    pub message: String,
    pub failed_position: Option<u32>,
}

impl S14StarfoldSsdAdapterError {
    pub fn new(
        kind: S14StarfoldSsdAdapterErrorKind,
        stage: S14StarfoldSsdAdapterStage,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            stage,
            message: message.into(),
            failed_position: None,
        }
    }

    pub fn at_position(mut self, position: u32) -> Self {
        self.failed_position = Some(position);
        self
    }

    pub fn into_engine_error(self) -> EngineError {
        let kind = match self.kind {
            S14StarfoldSsdAdapterErrorKind::Invalid => EngineErrorKind::InvalidRequest,
            S14StarfoldSsdAdapterErrorKind::Busy => EngineErrorKind::QueueFull,
            S14StarfoldSsdAdapterErrorKind::Exhausted => EngineErrorKind::RuntimeUnavailable,
            S14StarfoldSsdAdapterErrorKind::Unsupported => EngineErrorKind::UnsupportedPosition,
            S14StarfoldSsdAdapterErrorKind::Internal => EngineErrorKind::Internal,
        };
        EngineError {
            kind,
            message: format!("SSD StarFold {}: {}", self.stage.name(), self.message),
            failed_position: self.failed_position,
            retryable: self.kind == S14StarfoldSsdAdapterErrorKind::Busy,
        }
    }
}

impl fmt::Display for S14StarfoldSsdAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "SSD StarFold {}: {}",
            self.stage.name(),
            self.message
        )
    }
}

impl std::error::Error for S14StarfoldSsdAdapterError {}

impl From<S14StarfoldSsdAdapterError> for EngineError {
    fn from(error: S14StarfoldSsdAdapterError) -> Self {
        error.into_engine_error()
    }
}

/// API ready 前所需的 SSD startup resource proof 边界。
///
/// 实现必须从 concrete root 的真实 startup receipt/owner counters 导出 inventory，并用
/// session 的动态 inventory 复核 owner handoff。禁止返回手写“全1”或静态成功。SSD root
/// 尚未发布 startup receipt 的其他 root 不应为任何类型实现本 trait，worker 将 fail-closed。
pub trait S14StarfoldSsdStartupContract: 'static {
    fn startup_resource_inventory(
        &self,
    ) -> Result<S14ResidentK4ResourceInventory, S14StarfoldSsdAdapterError>;

    fn verify_session_inventory(
        &self,
        inventory: &S14StarfoldProductionResourceInventory,
    ) -> Result<(), S14StarfoldSsdAdapterError>;
}

/// 直接持有 SSD root 签发的真实 resident resource contract；不补写 request owner。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldSsdRootStartupContract {
    receipt: S14StarfoldResidentResourceContract,
}

impl S14StarfoldSsdRootStartupContract {
    pub fn from_root<F>(root: &S14StarfoldProductionRoot<F>) -> Result<Self, EngineError>
    where
        F: S14StarfoldBlockResourceFactory,
    {
        let receipt = root
            .resident_resource_contract()
            .map_err(|error| ssd_internal(S14StarfoldSsdAdapterStage::StartupContract, error))?;
        let contract = Self { receipt };
        // 在消费 root 前先执行一次完整 API schema 映射/验证。
        let inventory = contract
            .startup_resource_inventory()
            .map_err(EngineError::from)?;
        VerifiedS14ResidentK4Resources::verify(inventory)?;
        Ok(contract)
    }

    pub const fn receipt(&self) -> S14StarfoldResidentResourceContract {
        self.receipt
    }
}

impl S14StarfoldSsdStartupContract for S14StarfoldSsdRootStartupContract {
    fn startup_resource_inventory(
        &self,
    ) -> Result<S14ResidentK4ResourceInventory, S14StarfoldSsdAdapterError> {
        resident_inventory_from_ssd(self.receipt)
    }

    fn verify_session_inventory(
        &self,
        inventory: &S14StarfoldProductionResourceInventory,
    ) -> Result<(), S14StarfoldSsdAdapterError> {
        validate_ssd_session_inventory(self.receipt, inventory)
    }
}

/// 持有 SSD owning production root 的 API session factory。
pub struct S14StarfoldSsdProductionFactory<F, C>
where
    F: S14StarfoldBlockResourceFactory,
    C: S14StarfoldSsdStartupContract,
{
    root: Rc<RefCell<S14StarfoldProductionRoot<F>>>,
    startup_contract: Rc<C>,
    verified_resources: VerifiedS14ResidentK4Resources,
}

impl<F, C> S14StarfoldSsdProductionFactory<F, C>
where
    F: S14StarfoldBlockResourceFactory,
    C: S14StarfoldSsdStartupContract,
{
    pub fn new(
        root: S14StarfoldProductionRoot<F>,
        startup_contract: C,
    ) -> Result<Self, EngineError> {
        require_ready_lease(root.lease_state())?;
        let inventory = startup_contract
            .startup_resource_inventory()
            .map_err(EngineError::from)?;
        let verified_resources = VerifiedS14ResidentK4Resources::verify(inventory)?;
        Ok(Self {
            root: Rc::new(RefCell::new(root)),
            startup_contract: Rc::new(startup_contract),
            verified_resources,
        })
    }

    pub fn root(&self) -> Ref<'_, S14StarfoldProductionRoot<F>> {
        self.root.borrow()
    }

    pub fn root_mut(&self) -> RefMut<'_, S14StarfoldProductionRoot<F>> {
        self.root.borrow_mut()
    }
}

impl<F> S14StarfoldSsdProductionFactory<F, S14StarfoldSsdRootStartupContract>
where
    F: S14StarfoldBlockResourceFactory,
{
    pub fn from_ssd_root(root: S14StarfoldProductionRoot<F>) -> Result<Self, EngineError> {
        let startup_contract = S14StarfoldSsdRootStartupContract::from_root(&root)?;
        Self::new(root, startup_contract)
    }
}

/// 持有 SSD owning session，并缓存最后一次成功 begin/commit 的真实 checkpoint。
pub struct S14StarfoldSsdProductionSession<F, C>
where
    F: S14StarfoldBlockResourceFactory,
    C: S14StarfoldSsdStartupContract,
{
    session: Option<SsdS14StarfoldProductionSession<F>>,
    root: Rc<RefCell<S14StarfoldProductionRoot<F>>>,
    startup_contract: Rc<C>,
    committed_checkpoint: S14StarfoldRuntimeCheckpointReceipt,
    max_seq_len: u32,
}

impl<F, C> S14StarfoldSsdProductionSession<F, C>
where
    F: S14StarfoldBlockResourceFactory,
    C: S14StarfoldSsdStartupContract,
{
    fn session_ref(&self) -> Result<&SsdS14StarfoldProductionSession<F>, EngineError> {
        self.session.as_ref().ok_or_else(|| {
            S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Internal,
                S14StarfoldSsdAdapterStage::RootLease,
                "SSD production session 已归还 resident root",
            )
            .into_engine_error()
        })
    }

    fn session_mut(&mut self) -> Result<&mut SsdS14StarfoldProductionSession<F>, EngineError> {
        self.session.as_mut().ok_or_else(|| {
            S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Internal,
                S14StarfoldSsdAdapterStage::RootLease,
                "SSD production session 已归还 resident root",
            )
            .into_engine_error()
        })
    }
}

impl<F, C> S14StarfoldProductionSessionFactory for S14StarfoldSsdProductionFactory<F, C>
where
    F: S14StarfoldBlockResourceFactory + 'static,
    C: S14StarfoldSsdStartupContract,
{
    type Session = S14StarfoldSsdProductionSession<F, C>;

    fn resources(&self) -> &VerifiedS14ResidentK4Resources {
        &self.verified_resources
    }

    fn begin_prompt_prefilled_session(
        &mut self,
        prompt_token_ids: &[u32],
        max_seq_len: u32,
    ) -> Result<Self::Session, EngineError> {
        if prompt_token_ids.is_empty() || max_seq_len == 0 {
            return Err(S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Invalid,
                S14StarfoldSsdAdapterStage::PromptPrefill,
                "prompt 与 max_seq_len 必须非空/非零",
            )
            .into());
        }
        if prompt_token_ids.len() > max_seq_len as usize {
            return Err(S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Invalid,
                S14StarfoldSsdAdapterStage::PromptPrefill,
                "prompt 长度超过 max_seq_len",
            )
            .into());
        }
        let mut root = self.root.try_borrow_mut().map_err(|_| {
            S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Busy,
                S14StarfoldSsdAdapterStage::RootLease,
                "production root lease 正被另一个 owner 借用",
            )
            .into_engine_error()
        })?;
        require_ready_lease(root.lease_state())?;
        let session = root
            .begin_prompt_session(prompt_token_ids, max_seq_len)
            .map_err(|error| ssd_internal(S14StarfoldSsdAdapterStage::PromptPrefill, error))?;
        drop(root);

        let checkpoint = match session.checkpoint_identity() {
            Ok(identity) => checkpoint_from_ssd(identity),
            Err(error) => {
                return Err(close_after_begin_error(
                    session,
                    ssd_internal(S14StarfoldSsdAdapterStage::BeginCheckpoint, error).into(),
                ));
            }
        };
        let expected_position = u32::try_from(prompt_token_ids.len() - 1).map_err(|_| {
            S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Invalid,
                S14StarfoldSsdAdapterStage::BeginCheckpoint,
                "prompt position 超过 u32",
            )
            .into_engine_error()
        })?;
        if checkpoint.next_position() != expected_position
            || checkpoint.checkpoint_sha256() == [0; 32]
        {
            let error = S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Internal,
                S14StarfoldSsdAdapterStage::BeginCheckpoint,
                format!(
                    "prefill checkpoint 漂移：expected_position={expected_position} actual_position={} sha_present={}",
                    checkpoint.next_position(),
                    checkpoint.checkpoint_sha256() != [0; 32]
                ),
            )
            .at_position(checkpoint.next_position())
            .into_engine_error();
            return Err(close_after_begin_error(session, error));
        }
        if let Err(error) = verify_inventory(self.startup_contract.as_ref(), &session) {
            return Err(close_after_begin_error(session, error));
        }

        Ok(S14StarfoldSsdProductionSession {
            session: Some(session),
            root: Rc::clone(&self.root),
            startup_contract: Rc::clone(&self.startup_contract),
            committed_checkpoint: checkpoint,
            max_seq_len,
        })
    }
}

impl<F, C> S14StarfoldProductionSession for S14StarfoldSsdProductionSession<F, C>
where
    F: S14StarfoldBlockResourceFactory + 'static,
    C: S14StarfoldSsdStartupContract,
{
    fn committed_checkpoint_receipt(&self) -> S14StarfoldRuntimeCheckpointReceipt {
        self.committed_checkpoint
    }

    fn commit_next_starfold_block(
        &mut self,
        longest_prefix_limit: S14StarfoldK4CommitLimit,
    ) -> Result<S14StarfoldRuntimeCommittedBlockReceipt, EngineError> {
        let expected = self.committed_checkpoint;
        let observed_before = self
            .session_ref()?
            .checkpoint_identity()
            .map(checkpoint_from_ssd)
            .map_err(|error| {
                ssd_internal(S14StarfoldSsdAdapterStage::BeforeCommitCheckpoint, error)
            })?;
        if observed_before != expected {
            self.committed_checkpoint = observed_before;
            return Err(S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Internal,
                S14StarfoldSsdAdapterStage::BeforeCommitCheckpoint,
                "SSD session 在 commit 调用外推进了 authoritative checkpoint",
            )
            .at_position(observed_before.next_position())
            .into());
        }
        let physical_end = expected
            .next_position()
            .checked_add(SSD_STARFOLD_PHYSICAL_BLOCK_TOKENS)
            .ok_or_else(|| {
                S14StarfoldSsdAdapterError::new(
                    S14StarfoldSsdAdapterErrorKind::Unsupported,
                    S14StarfoldSsdAdapterStage::AtomicCommit,
                    "K4 physical position 溢出",
                )
                .at_position(expected.next_position())
                .into_engine_error()
            })?;
        if physical_end > self.max_seq_len {
            return Err(S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Unsupported,
                S14StarfoldSsdAdapterStage::AtomicCommit,
                format!(
                    "K4 physical block 需要 position end={physical_end}，超过 max_seq_len={}",
                    self.max_seq_len
                ),
            )
            .at_position(expected.next_position())
            .into());
        }

        let requested_limit = usize::from(longest_prefix_limit.max_committed_tokens());
        let delta = match self.session_mut()?.commit_next_block(requested_limit) {
            Ok(delta) => delta,
            Err(error) => {
                // SSD 可能在返回后置审计错误前已完成原子发布。只在 getter 成功时更新缓存；
                // getter 失败时保持旧值并返回 Internal，绝不 unwrap 或合成 checkpoint。
                if let Ok(identity) = self.session_ref()?.checkpoint_identity() {
                    self.committed_checkpoint = checkpoint_from_ssd(identity);
                }
                return Err(ssd_internal(S14StarfoldSsdAdapterStage::AtomicCommit, error).into());
            }
        };

        // SSD commit 已返回成功，立即缓存其真实 delta checkpoint，再执行任何后置校验。
        let committed = checkpoint_from_ssd(delta.checkpoint);
        self.committed_checkpoint = committed;
        let mapped = S14StarfoldRuntimeCommittedBlockReceipt::from_starfold_atomic_commit_receipt(
            &delta.receipt,
        )?;
        validate_delta(expected, committed, requested_limit, &delta, &mapped)?;

        let observed_after = self
            .session_ref()?
            .checkpoint_identity()
            .map(checkpoint_from_ssd)
            .map_err(|error| {
                ssd_internal(S14StarfoldSsdAdapterStage::AfterCommitCheckpoint, error)
            })?;
        if observed_after != committed {
            self.committed_checkpoint = observed_after;
            return Err(S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Internal,
                S14StarfoldSsdAdapterStage::AfterCommitCheckpoint,
                "delta checkpoint 与 commit 后 authoritative checkpoint 不一致",
            )
            .at_position(observed_after.next_position())
            .into());
        }
        verify_inventory(self.startup_contract.as_ref(), self.session_ref()?)?;
        Ok(mapped)
    }

    fn close(mut self) -> Result<(), EngineError> {
        let session = self.session.take().ok_or_else(|| {
            S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Internal,
                S14StarfoldSsdAdapterStage::Close,
                "SSD production session 已归还",
            )
            .into_engine_error()
        })?;
        let mut root = self.root.try_borrow_mut().map_err(|_| {
            S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Busy,
                S14StarfoldSsdAdapterStage::RootLease,
                "归还 production session 时 root lease 正被借用",
            )
            .into_engine_error()
        })?;
        root.return_prompt_session(session)
            .map_err(|error| ssd_internal(S14StarfoldSsdAdapterStage::Close, error).into())
    }
}

/// codec + SSD owning root + startup proof 的完整 backend 类型。
pub type S14StarfoldSsdProductionChatBackend<Codec, Factory, Contract> =
    S14StarfoldProductionChatBackend<Codec, S14StarfoldSsdProductionFactory<Factory, Contract>>;

pub type S14StarfoldConcreteSsdProductionFactory =
    S14StarfoldSsdProductionFactory<S14StarfoldConcreteFactory, S14StarfoldSsdRootStartupContract>;

pub type S14StarfoldConcreteSsdProductionSession =
    S14StarfoldSsdProductionSession<S14StarfoldConcreteFactory, S14StarfoldSsdRootStartupContract>;

pub type S14StarfoldConcreteSsdChatBackend<Codec> = S14StarfoldSsdProductionChatBackend<
    Codec,
    S14StarfoldConcreteFactory,
    S14StarfoldSsdRootStartupContract,
>;

pub fn build_s14_starfold_ssd_production_chat_backend<Codec, Factory, Contract>(
    codec: Codec,
    mut root: S14StarfoldProductionRoot<Factory>,
    startup_contract: Contract,
    max_seq_len: u32,
    default_max_tokens: u32,
) -> Result<S14StarfoldSsdProductionChatBackend<Codec, Factory, Contract>, EngineError>
where
    Codec: S14ChatCodec + 'static,
    Factory: S14StarfoldBlockResourceFactory + 'static,
    Contract: S14StarfoldSsdStartupContract,
{
    if let Some(eos_token_id) = codec.eos_token_id() {
        root.configure_starwave_eos_token_id(eos_token_id)
            .map_err(|error| ssd_internal(S14StarfoldSsdAdapterStage::StartupContract, error))?;
    }
    let factory = S14StarfoldSsdProductionFactory::new(root, startup_contract)?;
    build_s14_starfold_production_chat_backend(codec, factory, max_seq_len, default_max_tokens)
}

/// 直接消费 SSD root 自签 resident contract 的 production 构造入口。
pub fn build_s14_starfold_ssd_root_chat_backend<Codec, Factory>(
    codec: Codec,
    mut root: S14StarfoldProductionRoot<Factory>,
    max_seq_len: u32,
    default_max_tokens: u32,
) -> Result<
    S14StarfoldSsdProductionChatBackend<Codec, Factory, S14StarfoldSsdRootStartupContract>,
    EngineError,
>
where
    Codec: S14ChatCodec + 'static,
    Factory: S14StarfoldBlockResourceFactory + 'static,
{
    if let Some(eos_token_id) = codec.eos_token_id() {
        root.configure_starwave_eos_token_id(eos_token_id)
            .map_err(|error| ssd_internal(S14StarfoldSsdAdapterStage::StartupContract, error))?;
    }
    let factory = S14StarfoldSsdProductionFactory::from_ssd_root(root)?;
    build_s14_starfold_production_chat_backend(codec, factory, max_seq_len, default_max_tokens)
}

/// 在 resident worker 内加载 codec/root/startup proof，避免把 Vulkan owner 移出 owner 线程。
/// `main.rs` 在 SSD concrete startup receipt 落盘前不会调用本函数。
pub fn spawn_s14_starfold_ssd_production_worker<Loader, Codec, Factory, Contract>(
    queue_capacity: usize,
    max_seq_len: u32,
    default_max_tokens: u32,
    loader: Loader,
) -> Result<ResidentChatEngine, EngineError>
where
    Loader: FnOnce() -> Result<
            (Codec, S14StarfoldProductionRoot<Factory>, Contract),
            S14StarfoldSsdAdapterError,
        > + Send
        + 'static,
    Codec: S14ChatCodec + 'static,
    Factory: S14StarfoldBlockResourceFactory + 'static,
    Contract: S14StarfoldSsdStartupContract,
{
    ResidentChatEngine::spawn(queue_capacity, move || {
        let (codec, root, startup_contract) = loader().map_err(EngineError::from)?;
        let backend = build_s14_starfold_ssd_production_chat_backend(
            codec,
            root,
            startup_contract,
            max_seq_len,
            default_max_tokens,
        )?;
        Ok(Box::new(backend) as Box<dyn ResidentChatBackend>)
    })
}

/// resident worker 内直接加载 codec + SSD root，并消费 root 的真实 startup contract。
pub fn spawn_s14_starfold_ssd_root_worker<Loader, Codec, Factory>(
    queue_capacity: usize,
    max_seq_len: u32,
    default_max_tokens: u32,
    loader: Loader,
) -> Result<ResidentChatEngine, EngineError>
where
    Loader: FnOnce() -> Result<(Codec, S14StarfoldProductionRoot<Factory>), S14StarfoldSsdAdapterError>
        + Send
        + 'static,
    Codec: S14ChatCodec + 'static,
    Factory: S14StarfoldBlockResourceFactory + 'static,
{
    ResidentChatEngine::spawn(queue_capacity, move || {
        let (codec, root) = loader().map_err(EngineError::from)?;
        let backend =
            build_s14_starfold_ssd_root_chat_backend(codec, root, max_seq_len, default_max_tokens)?;
        Ok(Box::new(backend) as Box<dyn ResidentChatBackend>)
    })
}

fn resident_inventory_from_ssd(
    receipt: S14StarfoldResidentResourceContract,
) -> Result<S14ResidentK4ResourceInventory, S14StarfoldSsdAdapterError> {
    let (request_decoder_owners, request_owners_deferred_until_prompt) = match receipt.request_owned
    {
        S14StarfoldRequestOwnerReadiness::DeferredUntilPrompt => (0, true),
        S14StarfoldRequestOwnerReadiness::Active(_) => {
            return Err(S14StarfoldSsdAdapterError::new(
                S14StarfoldSsdAdapterErrorKind::Internal,
                S14StarfoldSsdAdapterStage::StartupContract,
                "root startup contract 不得把 request-owned decoder/block 冒充 resident owner",
            ));
        }
    };
    Ok(S14ResidentK4ResourceInventory {
        request_decoder_owners,
        request_owners_deferred_until_prompt,
        vulkan_context_owners: count_u32(receipt.vulkan_context_owners, "vulkan_context_owners")?,
        transfer_queue_owners: count_u32(receipt.transfer_queue_owners, "transfer_queue_owners")?,
        compute_queue_owners: count_u32(receipt.graphics_queue_owners, "graphics_queue_owners")?,
        paged_arena_owners: count_u32(receipt.paged_arena_owners, "paged_arena_owners")?,
        starfold_microtile_windows: count_u32(
            receipt.starfold_window_owners,
            "starfold_window_owners",
        )?,
        starfold_microtile_bytes: receipt.starfold_window_bytes_each,
        starfold_physical_allocation_bytes: receipt.starfold_physical_allocation_bytes,
        full_depth_layers: count_u32(receipt.full_depth_layers, "full_depth_layers")?,
        positions_per_physical_block: count_u32(
            receipt.physical_block_lanes,
            "physical_block_lanes",
        )?,
        routed_experts_per_position: count_u32(receipt.routed_top_k, "routed_top_k")?,
        verified_mapped_store_owners: count_u32(
            receipt.verified_mapped_asset_store_owners,
            "verified_mapped_asset_store_owners",
        )?,
        verified_lease_cache_owners: count_u32(
            receipt.verified_lease_cache_owners,
            "verified_lease_cache_owners",
        )?,
        verified_lease_cache_capacity_entries: count_u32(
            receipt.verified_lease_cache_capacity_entries,
            "verified_lease_cache_capacity_entries",
        )?,
        verified_lease_cache_contract_version: receipt.verified_lease_cache_contract_version,
        packed_l2_owners: count_u32(receipt.packed_l2_owners, "packed_l2_owners")?,
        packed_l2_capacity_bytes: receipt.packed_l2_capacity_bytes,
        packed_l2_contract_version: receipt.packed_l2_contract_version,
        starwave_transition_atlas_owners: count_u32(
            receipt.starwave_transition_atlas_owners,
            "starwave_transition_atlas_owners",
        )?,
        starwave_transition_atlas_capacity_entries: count_u32(
            receipt.starwave_transition_atlas_capacity_entries,
            "starwave_transition_atlas_capacity_entries",
        )?,
        starwave_transition_atlas_contract_version: receipt
            .starwave_transition_atlas_contract_version,
        terminal_head_uploader_owners: count_u32(
            receipt.terminal_head_uploader_owners,
            "terminal_head_uploader_owners",
        )?,
        starwave_commit_owners: count_u32(
            receipt.starwave_commit_owners,
            "starwave_commit_owners",
        )?,
        legacy_union_calls: receipt.forbidden.legacy_union_calls,
        legacy_grouped_moe_calls: receipt.forbidden.legacy_grouped_moe_calls,
        serial_position0_calls: receipt.forbidden.serial_position0_calls,
        serial_token_forward_calls: receipt.forbidden.serial_token_forward_calls,
        cpu_compute_fallback_calls: receipt.forbidden.cpu_fallback_calls,
        v38_fallback_calls: receipt.forbidden.v38_calls,
        v47_fallback_calls: receipt.forbidden.v47_calls,
        transformer_fallback_calls: receipt.forbidden.transformer_calls,
        whole_model_fallback_calls: receipt.forbidden.whole_model_fallback_calls,
    })
}

fn validate_ssd_session_inventory(
    startup: S14StarfoldResidentResourceContract,
    inventory: &S14StarfoldProductionResourceInventory,
) -> Result<(), S14StarfoldSsdAdapterError> {
    let persistent = inventory.persistent;
    let execution_owner_sets = [
        persistent.b4_owners,
        persistent.hc_bridge_stage_owners,
        persistent.direct_terminal_endpoint_owners,
    ];
    let execution_active = execution_owner_sets.iter().all(|&count| count == 1);
    let execution_deferred = execution_owner_sets.iter().all(|&count| count == 0);
    let active_block_valid = if execution_active {
        inventory.active_block.provider_owners == 1
            && inventory.active_block.hidden_bank_owners == 2
            && inventory.active_block.prefix_arena_owners == 1
            && inventory.active_block.ratio4_state_owners > 0
            && inventory.active_block.upload_auxiliary_owners > 0
    } else {
        execution_deferred && inventory.active_block == S14StarfoldBlockOwnerInventory::default()
    };
    let valid = persistent.base_runtime_owners == 1
        && persistent.starfold_runtime_owners == 1
        && persistent.paged_arena_owners == startup.paged_arena_owners
        && persistent.expert_catalog_owners == 1
        && persistent.microtile_window_owners == startup.starfold_window_owners
        && (execution_active || execution_deferred)
        && active_block_valid
        && inventory.pending_rebind_external_owner_sets == 0
        && inventory.retired_external_owner_sets == 0
        && inventory.decoder_state_owners == 1
        && inventory.whole_token_device_owners == 1
        && inventory.prefill_readback_owners <= 1
        && inventory.forbidden == startup.forbidden
        && inventory.forbidden == S14StarfoldForbiddenPathCounters::default();
    if !valid {
        return Err(S14StarfoldSsdAdapterError::new(
            S14StarfoldSsdAdapterErrorKind::Internal,
            S14StarfoldSsdAdapterStage::SessionInventory,
            format!(
                "owning session resource inventory 未闭合 startup/prompt handoff: {inventory:?}"
            ),
        ));
    }
    Ok(())
}

fn count_u32(value: usize, label: &'static str) -> Result<u32, S14StarfoldSsdAdapterError> {
    u32::try_from(value).map_err(|_| {
        S14StarfoldSsdAdapterError::new(
            S14StarfoldSsdAdapterErrorKind::Internal,
            S14StarfoldSsdAdapterStage::StartupContract,
            format!("{label} 超过 u32"),
        )
    })
}

fn require_ready_lease(lease: S14StarfoldProductionLeaseState) -> Result<(), EngineError> {
    match lease {
        S14StarfoldProductionLeaseState::Ready => Ok(()),
        S14StarfoldProductionLeaseState::Busy => Err(S14StarfoldSsdAdapterError::new(
            S14StarfoldSsdAdapterErrorKind::Busy,
            S14StarfoldSsdAdapterStage::RootLease,
            "production root 正被请求独占",
        )
        .into()),
        S14StarfoldProductionLeaseState::Exhausted => Err(S14StarfoldSsdAdapterError::new(
            S14StarfoldSsdAdapterErrorKind::Exhausted,
            S14StarfoldSsdAdapterStage::RootLease,
            "production root 因 owner-return/cleanup 失败已耗尽",
        )
        .into()),
    }
}

fn checkpoint_from_ssd(
    identity: S14StarfoldCheckpointIdentity,
) -> S14StarfoldRuntimeCheckpointReceipt {
    S14StarfoldRuntimeCheckpointReceipt::from_runtime_receipt(
        identity.position,
        identity.epoch,
        identity.decoder_state_sha256,
    )
}

fn verify_inventory<F, C>(
    contract: &C,
    session: &SsdS14StarfoldProductionSession<F>,
) -> Result<(), EngineError>
where
    F: S14StarfoldBlockResourceFactory,
    C: S14StarfoldSsdStartupContract,
{
    let inventory = session
        .resource_inventory()
        .map_err(|error| ssd_internal(S14StarfoldSsdAdapterStage::SessionInventory, error))?;
    contract
        .verify_session_inventory(&inventory)
        .map_err(EngineError::from)
}

fn validate_delta(
    expected: S14StarfoldRuntimeCheckpointReceipt,
    committed: S14StarfoldRuntimeCheckpointReceipt,
    requested_limit: usize,
    delta: &S14StarfoldCommittedLedgerDelta,
    mapped: &S14StarfoldRuntimeCommittedBlockReceipt,
) -> Result<(), EngineError> {
    let committed_tokens = delta.committed_token_ids.len();
    let committed_tokens_u32 = u32::try_from(committed_tokens).map_err(|_| {
        S14StarfoldSsdAdapterError::new(
            S14StarfoldSsdAdapterErrorKind::Internal,
            S14StarfoldSsdAdapterStage::CommitReceipt,
            "committed ledger delta 长度超过 u32",
        )
        .into_engine_error()
    })?;
    let committed_tokens_u64 = u64::from(committed_tokens_u32);
    let valid = delta.requested_commit_limit == requested_limit
        && (1..=usize::from(S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS))
            .contains(&delta.proposal_safe_commit_limit)
        && delta.effective_commit_limit == requested_limit.min(delta.proposal_safe_commit_limit)
        && (1..=delta.effective_commit_limit).contains(&committed_tokens)
        && delta.receipt.commit_limit == delta.effective_commit_limit
        && mapped.consumed() == expected
        && mapped.committed() == committed
        && mapped.committed_token_ids() == delta.committed_token_ids.as_slice()
        && committed
            .next_position()
            .checked_sub(expected.next_position())
            == Some(committed_tokens_u32)
        && committed
            .commit_epoch()
            .checked_sub(expected.commit_epoch())
            == Some(committed_tokens_u64)
        && expected.checkpoint_sha256() != [0; 32]
        && committed.checkpoint_sha256() != [0; 32]
        && expected.checkpoint_sha256() != committed.checkpoint_sha256();
    if !valid {
        return Err(S14StarfoldSsdAdapterError::new(
            S14StarfoldSsdAdapterErrorKind::Internal,
            S14StarfoldSsdAdapterStage::CommitReceipt,
            "SSD ledger delta 与原子 receipt 的 limit/token/position/epoch/SHA 未完整闭合",
        )
        .at_position(committed.next_position())
        .into());
    }
    Ok(())
}

fn close_after_begin_error<F>(
    session: SsdS14StarfoldProductionSession<F>,
    mut error: EngineError,
) -> EngineError
where
    F: S14StarfoldBlockResourceFactory,
{
    if let Err(cleanup) = session.close() {
        error.message = format!(
            "{}；同时 SSD owning session close 失败：{cleanup:#}",
            error.message
        );
    }
    error
}

fn ssd_internal(
    stage: S14StarfoldSsdAdapterStage,
    error: impl fmt::Display,
) -> S14StarfoldSsdAdapterError {
    S14StarfoldSsdAdapterError::new(
        S14StarfoldSsdAdapterErrorKind::Internal,
        stage,
        // `anyhow::Error` 的 alternate Display 会保留完整 context/source chain。
        // 真实性门价格很高，不能只上报最外层阶段名后再靠重复启动定位。
        format!("{error:#}"),
    )
}
