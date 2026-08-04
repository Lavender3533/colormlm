//! StarFold runtime commit provider 到 resident K4 Chat backend 的最窄适配层。
//!
//! 本模块不执行模型、不选择 token，也不计算 checkpoint SHA。provider 必须回带 runtime
//! 已经原子提交的 token ledger 与提交前后 checkpoint 身份；adapter 只验证连续性并把
//! 二进制 checkpoint SHA 无损编码为 API 层使用的十六进制字符串。

use crate::s14_engine::{
    S14ResidentK4Checkpoint, S14ResidentK4CommittedBlock, S14ResidentK4Decoder,
    S14ResidentK4Request, VerifiedS14ResidentK4Resources,
};
use crate::{EngineError, EngineErrorKind};
use std::{fmt::Write as _, time::Instant};

pub const S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS: u8 = 4;

/// runtime 权威 committed state 的不可解释身份。
///
/// `checkpoint_sha256` 必须来自 runtime commit receipt（例如 StarFold terminal chain 的
/// committed checkpoint SHA），不得由 API adapter 根据 position/token 自行合成。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14ProductionK4CheckpointIdentity {
    next_position: u32,
    commit_epoch: u64,
    checkpoint_sha256: [u8; 32],
}

impl S14ProductionK4CheckpointIdentity {
    pub const fn from_runtime_commit(
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
}

/// runtime 一次原子 commit 的完整输出。token 必须取自已提交 ledger，而不是 draft、
/// candidate head、fixture 或 API 层采样结果。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14ProductionK4CommittedBlock {
    block_sequence: u64,
    base_position: u32,
    consumed: S14ProductionK4CheckpointIdentity,
    committed: S14ProductionK4CheckpointIdentity,
    committed_token_ids: Vec<u32>,
}

impl S14ProductionK4CommittedBlock {
    pub fn from_runtime_commit(
        block_sequence: u64,
        base_position: u32,
        consumed: S14ProductionK4CheckpointIdentity,
        committed: S14ProductionK4CheckpointIdentity,
        committed_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            block_sequence,
            base_position,
            consumed,
            committed,
            committed_token_ids,
        }
    }

    pub const fn block_sequence(&self) -> u64 {
        self.block_sequence
    }

    pub const fn base_position(&self) -> u32 {
        self.base_position
    }

    pub const fn consumed(&self) -> S14ProductionK4CheckpointIdentity {
        self.consumed
    }

    pub const fn committed(&self) -> S14ProductionK4CheckpointIdentity {
        self.committed
    }

    pub fn committed_token_ids(&self) -> &[u32] {
        &self.committed_token_ids
    }
}

/// 单个聊天请求所独占的真实 runtime continuation。
///
/// `commit_next_block` 必须先完成 terminal/head/最长可靠前缀/checkpoint 的原子发布，
/// 再返回本次真实 committed block。`max_committed_tokens` 是硬上限，不能在 runtime 中
/// 多提交 token 后只向 API 隐藏尾部。
pub trait S14ProductionK4CommitRequest {
    fn committed_checkpoint(&self) -> S14ProductionK4CheckpointIdentity;

    fn commit_next_block(
        &mut self,
        max_committed_tokens: u8,
    ) -> Result<S14ProductionK4CommittedBlock, EngineError>;

    fn close(self) -> Result<(), EngineError>;
}

/// 长驻 StarFold production root 的最窄 API。实现方必须复用同一个 Vulkan/paged
/// arena/catalog/双 1MiB microtile owner，并在 `begin_committed_request` 内真实完成 prompt
/// prefill；返回的初始 checkpoint position 必须等于 `prompt_token_ids.len() - 1`。
pub trait S14ProductionK4CommitProvider: 'static {
    type Request: S14ProductionK4CommitRequest;

    fn resources(&self) -> &VerifiedS14ResidentK4Resources;

    fn begin_committed_request(
        &mut self,
        prompt_token_ids: &[u32],
        max_seq_len: u32,
    ) -> Result<Self::Request, EngineError>;
}

/// production commit provider 的 concrete resident decoder adapter。
///
/// runtime root 只需实现 [`S14ProductionK4CommitProvider`]；本类型已经直接实现
/// [`S14ResidentK4Decoder`]，不会引入旧 bundle publisher 或第二模型实例。
pub struct S14RuntimeCommitK4Decoder<P> {
    provider: P,
}

impl<P> S14RuntimeCommitK4Decoder<P> {
    pub const fn new(provider: P) -> Self {
        Self { provider }
    }

    pub fn provider(&self) -> &P {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut P {
        &mut self.provider
    }

    pub fn into_inner(self) -> P {
        self.provider
    }
}

pub struct S14RuntimeCommitK4Request<R> {
    request: R,
    production_checkpoint: S14ProductionK4CheckpointIdentity,
    api_checkpoint: S14ResidentK4Checkpoint,
}

impl<P> S14ResidentK4Decoder for S14RuntimeCommitK4Decoder<P>
where
    P: S14ProductionK4CommitProvider,
{
    type Request = S14RuntimeCommitK4Request<P::Request>;

    fn resources(&self) -> &VerifiedS14ResidentK4Resources {
        self.provider.resources()
    }

    fn begin_request(
        &mut self,
        prompt_token_ids: &[u32],
        max_seq_len: u32,
    ) -> Result<Self::Request, EngineError> {
        let expected_position =
            u32::try_from(prompt_token_ids.len().saturating_sub(1)).map_err(|_| {
                EngineError::new(
                    EngineErrorKind::InvalidRequest,
                    "resident K4 prompt position 数量超过 u32 上限",
                )
            })?;
        let request = self
            .provider
            .begin_committed_request(prompt_token_ids, max_seq_len)?;
        let production_checkpoint = request.committed_checkpoint();
        let initial = to_api_checkpoint(production_checkpoint).and_then(|api_checkpoint| {
            if production_checkpoint.next_position() != expected_position {
                return Err(EngineError::runtime_unavailable(format!(
                    "production commit provider 未从真实 prompt prefill checkpoint 启动：期望 position={expected_position}，实际 position={}",
                    production_checkpoint.next_position()
                )));
            }
            Ok((production_checkpoint, api_checkpoint))
        });
        match initial {
            Ok((production_checkpoint, api_checkpoint)) => Ok(S14RuntimeCommitK4Request {
                request,
                production_checkpoint,
                api_checkpoint,
            }),
            Err(mut error) => {
                if let Err(cleanup) = request.close() {
                    error.message = format!(
                        "{}；同时 production K4 request close 失败：{}",
                        error.message, cleanup.message
                    );
                }
                Err(error)
            }
        }
    }
}

impl<R> S14ResidentK4Request for S14RuntimeCommitK4Request<R>
where
    R: S14ProductionK4CommitRequest,
{
    fn checkpoint(&self) -> &S14ResidentK4Checkpoint {
        &self.api_checkpoint
    }

    fn execute_next_block(
        &mut self,
        remaining_tokens: u32,
    ) -> Result<S14ResidentK4CommittedBlock, EngineError> {
        if remaining_tokens == 0 {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                "resident K4 adapter 收到 remaining_tokens=0",
            ));
        }
        let observed_before = self.request.committed_checkpoint();
        if observed_before != self.production_checkpoint {
            return Err(EngineError::new(
                EngineErrorKind::Internal,
                "production K4 request 在 block 调用外推进了 committed checkpoint",
            ));
        }
        let max_committed_tokens =
            remaining_tokens.min(u32::from(S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS)) as u8;
        let started = Instant::now();
        let block = self.request.commit_next_block(max_committed_tokens)?;
        let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let observed_after = self.request.committed_checkpoint();
        validate_runtime_block(
            self.production_checkpoint,
            observed_after,
            max_committed_tokens,
            &block,
        )?;

        let consumed = to_api_checkpoint(block.consumed)?;
        let committed = to_api_checkpoint(block.committed)?;
        let block_sequence = block.block_sequence;
        let base_position = block.base_position;
        let token_ids = block.committed_token_ids;
        self.production_checkpoint = observed_after;
        self.api_checkpoint = committed.clone();
        Ok(S14ResidentK4CommittedBlock {
            block_sequence,
            base_position,
            consumed,
            committed,
            token_ids,
            wall_ms,
        })
    }

    fn close(self) -> Result<(), EngineError> {
        let observed = self.request.committed_checkpoint();
        let identity_error = (observed != self.production_checkpoint).then(|| {
            EngineError::new(
                EngineErrorKind::Internal,
                "production K4 request close 前 committed checkpoint 漂移",
            )
        });
        let cleanup = self.request.close();
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

fn validate_runtime_block(
    expected: S14ProductionK4CheckpointIdentity,
    observed_after: S14ProductionK4CheckpointIdentity,
    max_committed_tokens: u8,
    block: &S14ProductionK4CommittedBlock,
) -> Result<(), EngineError> {
    let committed_tokens = block.committed_token_ids.len();
    let committed_tokens_u32 = u32::try_from(committed_tokens).map_err(|_| {
        EngineError::new(
            EngineErrorKind::Internal,
            "production K4 committed token 数量超过 u32",
        )
    })?;
    let committed_tokens_u64 = u64::from(committed_tokens_u32);
    let valid = block.block_sequence > 0
        && block.base_position == block.consumed.next_position()
        && block.consumed == expected
        && block.committed == observed_after
        && (1..=usize::from(max_committed_tokens)).contains(&committed_tokens)
        && committed_tokens <= usize::from(S14_PRODUCTION_K4_MAX_COMMITTED_TOKENS)
        && block
            .committed
            .next_position()
            .checked_sub(block.consumed.next_position())
            == Some(committed_tokens_u32)
        && block
            .committed
            .commit_epoch()
            .checked_sub(block.consumed.commit_epoch())
            == Some(committed_tokens_u64)
        && block.consumed.checkpoint_sha256() != [0; 32]
        && block.committed.checkpoint_sha256() != [0; 32]
        && block.consumed.checkpoint_sha256() != block.committed.checkpoint_sha256();
    if !valid {
        return Err(EngineError::new(
            EngineErrorKind::Internal,
            "production K4 commit receipt 未闭合真实 consumed/committed checkpoint、token ledger 或调用上限",
        ));
    }
    Ok(())
}

fn to_api_checkpoint(
    identity: S14ProductionK4CheckpointIdentity,
) -> Result<S14ResidentK4Checkpoint, EngineError> {
    if identity.checkpoint_sha256() == [0; 32] {
        return Err(EngineError::new(
            EngineErrorKind::Internal,
            "production K4 checkpoint SHA-256 为空",
        ));
    }
    let mut sha256 = String::with_capacity(64);
    for byte in identity.checkpoint_sha256() {
        write!(&mut sha256, "{byte:02x}").expect("写入 String 不会失败");
    }
    Ok(S14ResidentK4Checkpoint {
        position: identity.next_position(),
        commit_epoch: identity.commit_epoch(),
        sha256,
    })
}
