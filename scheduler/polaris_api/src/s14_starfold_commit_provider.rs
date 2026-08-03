//! 纯 S14 StarFold production root 到 resident K4 commit adapter 的 concrete 接线。
//!
//! 本模块只定义 `polaris_api` 所需的最窄 runtime ABI，不拥有模型实现，也不提供 mock、
//! fixture 或成功占位。`s14_starfold_ssd_adapter` 为 `ssd_inference` 的 owning
//! production root/session 实现本模块定义的 factory/session trait，并把原始原子 commit
//! receipt 无损搬入边界。

use crate::s14_engine::{S14ChatCodec, S14ResidentK4ChatBackend, VerifiedS14ResidentK4Resources};
use crate::s14_resident_k4_adapter::{
    S14ProductionK4CheckpointIdentity, S14ProductionK4CommitProvider, S14ProductionK4CommitRequest,
    S14ProductionK4CommittedBlock, S14RuntimeCommitK4Decoder,
    S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS,
};
use crate::{EngineError, EngineErrorKind};
use ssd_inference::s14_starfold_k4_terminal_chain::{
    S14StarfoldK4CommitReceipt, S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION,
};

/// 传给 StarWave 最长可靠前缀决策的硬上限。
///
/// 该值只能是 `1..=4`。production session 必须在选 checkpoint、发布 device/host state
/// 之前应用此上限；禁止先提交更长前缀，再只在回执或 API 输出里隐藏尾部。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4CommitLimit(u8);

impl S14StarfoldK4CommitLimit {
    fn new(max_committed_tokens: u8) -> Result<Self, EngineError> {
        if !(1..=S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS).contains(&max_committed_tokens) {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                format!(
                    "StarFold K4 committed-prefix 上限必须是 1..={}，实际为 {max_committed_tokens}",
                    S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS
                ),
            ));
        }
        Ok(Self(max_committed_tokens))
    }

    /// 应原样传给 StarWave 最长可靠前缀选择或其前置截断决策。
    pub const fn max_committed_tokens(self) -> u8 {
        self.0
    }
}

/// runtime 原始 committed checkpoint receipt。
///
/// SHA 必须是 runtime receipt 直接给出的 32 字节值，不能由 position、epoch、token 或
/// API adapter 重新计算。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldRuntimeCheckpointReceipt {
    next_position: u32,
    commit_epoch: u64,
    checkpoint_sha256: [u8; 32],
}

impl S14StarfoldRuntimeCheckpointReceipt {
    pub const fn from_runtime_receipt(
        next_position: u32,
        commit_epoch: u64,
        checkpoint_sha256: [u8; 32],
    ) -> Self {
        Self {
            next_position,
            commit_epoch,
            checkpoint_sha256,
        }
    }

    pub const fn next_position(self) -> u32 {
        self.next_position
    }

    pub const fn commit_epoch(self) -> u64 {
        self.commit_epoch
    }

    pub const fn checkpoint_sha256(self) -> [u8; 32] {
        self.checkpoint_sha256
    }

    fn into_production(self) -> S14ProductionK4CheckpointIdentity {
        S14ProductionK4CheckpointIdentity::from_runtime_commit(
            self.next_position,
            self.commit_epoch,
            self.checkpoint_sha256,
        )
    }
}

/// runtime 已完成 terminal、最长前缀选择及 host/device 原子发布后的权威回执。
///
/// `committed_token_ids` 必须是该次原子 commit 向 ledger 新增的完整切片，不能来自 draft、
/// GPU head candidate、teacher-forced 输入或截断后的 API 视图。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldRuntimeCommittedBlockReceipt {
    consumed: S14StarfoldRuntimeCheckpointReceipt,
    committed: S14StarfoldRuntimeCheckpointReceipt,
    committed_token_ids: Vec<u32>,
}

impl S14StarfoldRuntimeCommittedBlockReceipt {
    pub fn from_atomic_runtime_commit(
        consumed: S14StarfoldRuntimeCheckpointReceipt,
        committed: S14StarfoldRuntimeCheckpointReceipt,
        committed_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            consumed,
            committed,
            committed_token_ids,
        }
    }

    pub const fn consumed(&self) -> S14StarfoldRuntimeCheckpointReceipt {
        self.consumed
    }

    pub const fn committed(&self) -> S14StarfoldRuntimeCheckpointReceipt {
        self.committed
    }

    pub fn committed_token_ids(&self) -> &[u32] {
        &self.committed_token_ids
    }

    /// 从当前 `ssd_inference` StarFold terminal-chain 原子回执无损抽取 API 边界数据。
    /// checkpoint SHA 直接复制 receipt 内的原始 `[u8; 32]`，不会重新哈希。
    pub fn from_starfold_atomic_commit_receipt(
        receipt: &S14StarfoldK4CommitReceipt,
    ) -> Result<Self, EngineError> {
        let committed_token_ids = receipt.decision.committed_token_ids.clone();
        let valid = receipt.schema_version == S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION
            && (1..=usize::from(S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS))
                .contains(&receipt.commit_limit)
            && receipt.committed_tokens == committed_token_ids.len()
            && (1..=usize::from(S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS))
                .contains(&receipt.committed_tokens)
            && receipt.committed_tokens <= receipt.commit_limit
            && receipt.checkpoint_index.checked_add(1) == Some(receipt.committed_tokens)
            && receipt.host_device_checkpoint_bytes_verified
            && receipt.device_commit.host_device_bytes_verified
            && receipt.device_commit.position == receipt.committed_position
            && receipt.device_commit.epoch == receipt.committed_epoch
            && receipt.device_commit.active_bank == receipt.committed_active_bank
            && receipt.device_commit.accepted_tokens == receipt.committed_tokens
            && receipt.device_commit.checkpoint_index == receipt.checkpoint_index
            && receipt.terminal_head_submit_calls == 1
            && receipt.checkpoint_export_calls == 1
            && receipt.legacy_union_calls == 0
            && receipt.legacy_grouped_moe_calls == 0
            && receipt.serial_token_forward_calls == 0
            && receipt.cpu_fallback_calls == 0;
        if !valid {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                "ssd_inference StarFold 原子 commit receipt 身份、device 发布或禁用路径计数漂移",
            ));
        }
        Ok(Self::from_atomic_runtime_commit(
            S14StarfoldRuntimeCheckpointReceipt::from_runtime_receipt(
                receipt.base_position,
                receipt.base_commit_epoch,
                *receipt.base_checkpoint_sha256.as_bytes(),
            ),
            S14StarfoldRuntimeCheckpointReceipt::from_runtime_receipt(
                receipt.committed_position,
                receipt.committed_epoch,
                *receipt.committed_checkpoint_sha256.as_bytes(),
            ),
            committed_token_ids,
        ))
    }
}

/// 每个 HTTP 请求独占的纯 S14 production continuation。
pub trait S14StarfoldProductionSession {
    /// 返回当前 runtime 权威 committed checkpoint 的原始 receipt 身份。
    fn committed_checkpoint_receipt(&self) -> S14StarfoldRuntimeCheckpointReceipt;

    /// 执行下一块 FullDepth43 → direct terminal → StarWave 原子 commit。
    ///
    /// `longest_prefix_limit` 必须在最长前缀选择或选择前截断时生效；本方法只能在原子
    /// commit 完成后返回 `Ok`，并返回 runtime ledger 实际新增的全部 token。
    fn commit_next_starfold_block(
        &mut self,
        longest_prefix_limit: S14StarfoldK4CommitLimit,
    ) -> Result<S14StarfoldRuntimeCommittedBlockReceipt, EngineError>;

    fn close(self) -> Result<(), EngineError>;
}

/// 长驻纯 S14 production root/session factory 的注入边界。
///
/// 实现必须复用唯一 resident Vulkan/StarFold/StarWave owner。`begin_prompt_prefilled_session`
/// 必须对整个 prompt 做真实 teacher-forced prefill，并在返回前原子发布 position
/// `prompt_token_ids.len() - 1` 的 checkpoint；返回的 session 必须独占同一 root 的请求句柄，
/// 不得只创建 position0 session、延迟到首块生成或另起第二套 production owner。
pub trait S14StarfoldProductionSessionFactory: 'static {
    type Session: S14StarfoldProductionSession;

    fn resources(&self) -> &VerifiedS14ResidentK4Resources;

    fn begin_prompt_prefilled_session(
        &mut self,
        prompt_token_ids: &[u32],
        max_seq_len: u32,
    ) -> Result<Self::Session, EngineError>;
}

/// 持有可注入 production root/session factory 的 concrete commit provider。
pub struct S14StarfoldCommitProvider<F> {
    factory: F,
}

impl<F> S14StarfoldCommitProvider<F> {
    pub const fn new(factory: F) -> Self {
        Self { factory }
    }

    pub const fn factory(&self) -> &F {
        &self.factory
    }

    pub fn factory_mut(&mut self) -> &mut F {
        &mut self.factory
    }

    pub fn into_inner(self) -> F {
        self.factory
    }
}

/// 每个请求的 concrete wrapper；只保存 runtime session 与最后一次权威 receipt 身份。
pub struct S14StarfoldCommitRequest<S> {
    session: S,
    committed: S14StarfoldRuntimeCheckpointReceipt,
}

impl<F> S14ProductionK4CommitProvider for S14StarfoldCommitProvider<F>
where
    F: S14StarfoldProductionSessionFactory,
{
    type Request = S14StarfoldCommitRequest<F::Session>;

    fn resources(&self) -> &VerifiedS14ResidentK4Resources {
        self.factory.resources()
    }

    fn begin_committed_request(
        &mut self,
        prompt_token_ids: &[u32],
        max_seq_len: u32,
    ) -> Result<Self::Request, EngineError> {
        if prompt_token_ids.is_empty() {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "StarFold production prompt 不能为空",
            ));
        }
        let prompt_tokens = u32::try_from(prompt_token_ids.len()).map_err(|_| {
            EngineError::new(
                EngineErrorKind::InvalidRequest,
                "StarFold production prompt token 数量超过 u32 上限",
            )
        })?;
        if max_seq_len == 0 || prompt_tokens > max_seq_len {
            return Err(EngineError::new(
                EngineErrorKind::InvalidRequest,
                "StarFold production prompt 超过 max_seq_len",
            ));
        }
        let expected_position = prompt_tokens - 1;
        let session = self
            .factory
            .begin_prompt_prefilled_session(prompt_token_ids, max_seq_len)?;
        let committed = session.committed_checkpoint_receipt();
        if committed.next_position() != expected_position
            || committed.checkpoint_sha256() == [0; 32]
        {
            let mut error = EngineError::runtime_unavailable(format!(
                "StarFold production prompt prefill 回执未闭合：期望 position={expected_position}，实际 position={}，sha256_present={}",
                committed.next_position(),
                committed.checkpoint_sha256() != [0; 32]
            ));
            if let Err(cleanup) = session.close() {
                error.message = format!(
                    "{}；同时关闭 prefill session 失败：{}",
                    error.message, cleanup.message
                );
            }
            return Err(error);
        }
        Ok(S14StarfoldCommitRequest { session, committed })
    }
}

impl<S> S14ProductionK4CommitRequest for S14StarfoldCommitRequest<S>
where
    S: S14StarfoldProductionSession,
{
    fn committed_checkpoint(&self) -> S14ProductionK4CheckpointIdentity {
        self.committed.into_production()
    }

    fn commit_next_block(
        &mut self,
        max_committed_tokens: u8,
    ) -> Result<S14ProductionK4CommittedBlock, EngineError> {
        let limit = S14StarfoldK4CommitLimit::new(max_committed_tokens)?;
        let observed_before = self.session.committed_checkpoint_receipt();
        if observed_before != self.committed {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                "StarFold production session 在 block 调用外推进了 committed checkpoint",
            ));
        }

        let receipt = self.session.commit_next_starfold_block(limit)?;
        let observed_after = self.session.committed_checkpoint_receipt();
        validate_atomic_receipt(self.committed, observed_after, limit, &receipt)?;

        let consumed = receipt.consumed.into_production();
        let committed = receipt.committed.into_production();
        let committed_token_ids = receipt.committed_token_ids;
        self.committed = observed_after;
        Ok(S14ProductionK4CommittedBlock::from_runtime_commit(
            consumed,
            committed,
            committed_token_ids,
        ))
    }

    fn close(self) -> Result<(), EngineError> {
        let observed = self.session.committed_checkpoint_receipt();
        let identity_error = (observed != self.committed).then(|| {
            EngineError::new(
                EngineErrorKind::Internal,
                "StarFold production session close 前 committed checkpoint 漂移",
            )
        });
        let cleanup = self.session.close();
        match (identity_error, cleanup) {
            (None, Ok(())) => Ok(()),
            (Some(error), Ok(())) | (None, Err(error)) => Err(error),
            (Some(mut error), Err(cleanup)) => {
                error.message = format!("{}；同时 close 失败：{}", error.message, cleanup.message);
                Err(error)
            }
        }
    }
}

/// 生产构造入口：codec + 唯一纯 S14 root/session factory → resident K4 chat backend。
pub type S14StarfoldProductionChatBackend<C, F> =
    S14ResidentK4ChatBackend<C, S14RuntimeCommitK4Decoder<S14StarfoldCommitProvider<F>>>;

pub fn build_s14_starfold_production_chat_backend<C, F>(
    codec: C,
    factory: F,
    max_seq_len: u32,
    default_max_tokens: u32,
) -> Result<S14StarfoldProductionChatBackend<C, F>, EngineError>
where
    C: S14ChatCodec + 'static,
    F: S14StarfoldProductionSessionFactory,
{
    let provider = S14StarfoldCommitProvider::new(factory);
    let decoder = S14RuntimeCommitK4Decoder::new(provider);
    S14ResidentK4ChatBackend::new(codec, decoder, max_seq_len, default_max_tokens)
}

fn validate_atomic_receipt(
    expected: S14StarfoldRuntimeCheckpointReceipt,
    observed_after: S14StarfoldRuntimeCheckpointReceipt,
    limit: S14StarfoldK4CommitLimit,
    receipt: &S14StarfoldRuntimeCommittedBlockReceipt,
) -> Result<(), EngineError> {
    let committed_tokens = receipt.committed_token_ids.len();
    let committed_tokens_u32 = u32::try_from(committed_tokens).map_err(|_| {
        EngineError::new(
            EngineErrorKind::Internal,
            "StarFold runtime committed token 数量超过 u32",
        )
    })?;
    let committed_tokens_u64 = u64::from(committed_tokens_u32);
    let valid = receipt.consumed == expected
        && receipt.committed == observed_after
        && (1..=usize::from(limit.max_committed_tokens())).contains(&committed_tokens)
        && receipt
            .committed
            .next_position()
            .checked_sub(receipt.consumed.next_position())
            == Some(committed_tokens_u32)
        && receipt
            .committed
            .commit_epoch()
            .checked_sub(receipt.consumed.commit_epoch())
            == Some(committed_tokens_u64)
        && receipt.consumed.checkpoint_sha256() != [0; 32]
        && receipt.committed.checkpoint_sha256() != [0; 32]
        && receipt.consumed.checkpoint_sha256() != receipt.committed.checkpoint_sha256();
    if !valid {
        return Err(EngineError::new(
            EngineErrorKind::Internal,
            "StarFold runtime 原子 commit receipt 未闭合 consumed/committed checkpoint、完整 ledger 或最长前缀硬上限",
        ));
    }
    Ok(())
}
