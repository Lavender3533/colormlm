//! 纯 S14 StarFold K4 的 terminal/head/checkpoint 提交链。
//!
//! 本模块只接受 43 层 StarFold seal、真实 production terminal owned future 与
//! `WholeTokenDeviceState`。它不构造 token，不调用旧 grouped-MoE/union，也不提供
//! CPU/旧模型 fallback。所有可失败工作在 host/device 权威状态发布前完成。

use crate::{
    s14_causal_block_hc_qkv_adapter::S14CausalBlockVulkanHcQkvAdapter,
    s14_causal_block_hc_qkv_recorder::S14CausalBlockStarfoldTerminalBlockOwners,
    s14_causal_block_host_candidates::S14CausalBlockDeferredHostCandidateBatch,
    s14_causal_block_layer::{
        S14CausalBlockDeviceCheckpointStorage, S14CausalBlockFinalOutput,
        S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
    },
    s14_head_chunk_argmax::S14HeadChunkArgmaxShape,
    s14_starfold_k4_adapter::{
        S14StarfoldConcreteK4Stage, S14StarfoldK4FullDepthReceipt, S14StarfoldK4ProductionAdapter,
    },
    s14_starfold_terminal_endpoint::{S14StarfoldTerminalBlockInputs, S14StarfoldTerminalEndpoint},
    s14_starwave_transaction::{S14StarwaveProofWriter, S14StarwaveSha256},
    s14_whole_token_device::{
        WholeTokenDeviceBlockCommitReceipt, WholeTokenDeviceCommittedCheckpointBinding,
        WholeTokenDeviceState,
    },
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk::{self, Handle};
use polaris_s14_runner::{
    decide_longest_prefix, BatchedWholeTokenPosition, DecoderStateV1, GraphProfile,
    LongestPrefixDecision, RouteDecision, RouterKind, BATCHED_CAUSAL_WHOLE_TOKEN_MODE,
    FULL_DEPTH_LAYERS, VOCAB_SIZE,
};
use std::sync::Arc;

pub const S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION: u32 = 1;
pub const S14_STARFOLD_K4_BLOCK_SIZE: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4CandidateCheckpointIdentity {
    pub checkpoint_index: usize,
    pub position: u32,
    pub commit_epoch: u64,
    pub active_fixed_bank: u8,
    pub teacher_forced_input_token_id: u32,
    pub predicted_token_id: u32,
    pub checkpoint_sha256: S14StarwaveSha256,
}

/// 一次 host/device K4 前缀原子提交的强回执。
#[derive(Clone, Debug)]
pub struct S14StarfoldK4CommitReceipt {
    pub schema_version: u32,
    pub block_sequence: u64,
    pub base_position: u32,
    pub base_commit_epoch: u64,
    pub base_checkpoint_sha256: S14StarwaveSha256,
    pub previous_commit_chain_sha256: S14StarwaveSha256,
    pub draft_token_ids: [u32; S14_STARFOLD_K4_BLOCK_SIZE],
    pub predicted_token_ids: [u32; S14_STARFOLD_K4_BLOCK_SIZE],
    /// caller 允许本次原子发布的最大 prefix 长度；始终为 1..=4。
    pub commit_limit: usize,
    pub decision: LongestPrefixDecision,
    /// 下一块 rebind 必须消费的同一份 FullDepth43 seal，不允许 caller 重报结构 identity。
    pub sealed_full_depth: S14StarfoldK4FullDepthReceipt,
    pub candidate_checkpoints: Vec<S14StarfoldK4CandidateCheckpointIdentity>,
    /// limit 内 mismatch 时包含 target fallback；否则为 limit 内 draft prefix。始终为 1..=4。
    pub committed_tokens: usize,
    pub checkpoint_index: usize,
    pub committed_position: u32,
    pub committed_epoch: u64,
    pub committed_active_bank: usize,
    pub committed_input_token_id: u32,
    pub committed_checkpoint_sha256: S14StarwaveSha256,
    pub host_device_checkpoint_bytes_verified: bool,
    pub device_commit: WholeTokenDeviceBlockCommitReceipt,
    pub device_ready_timeline_value: u64,
    pub device_checkpoint_arena_bytes: u64,
    pub physical_evidence_sha256: S14StarwaveSha256,
    pub commit_chain_sha256: S14StarwaveSha256,
    pub terminal_head_submit_calls: u32,
    pub checkpoint_export_calls: u32,
    pub legacy_union_calls: u32,
    pub legacy_grouped_moe_calls: u32,
    pub serial_token_forward_calls: u32,
    pub cpu_fallback_calls: u32,
    /// durable 文件发布还必须使用 active device readback 做逐字节校验，本提交不伪造该证明。
    pub durable_checkpoint_persisted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldK4NextBlockGate {
    AwaitingCommittedStateProviderAndInputHiddenRebind,
}

/// 下一个 K4 的真实 launch anchor。它绑定上一块刚发布的 host SHA 和 active device bank；
/// committed-state/HC-QKV provider 必须消费该 bank，token embedding owner 则独立生成新的
/// `[K,4,4096]` initial hidden；两者未完成 production 重绑前，本类型不会伪造 launch。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4NextBlockLaunchBinding {
    pub schema_version: u32,
    pub next_block_sequence: u64,
    pub previous_committed_position: u32,
    pub previous_committed_epoch: u64,
    pub previous_committed_checkpoint_sha256: S14StarwaveSha256,
    pub previous_commit_chain_sha256: S14StarwaveSha256,
    pub next_block_base_position: u32,
    pub next_block_base_epoch: u64,
    pub next_block_base_checkpoint_sha256: S14StarwaveSha256,
    pub input_token_id: u32,
    pub device_checkpoint_state_bytes: u64,
    pub device_checkpoint_epoch: u64,
    pub device_checkpoint_active_bank: usize,
    /// 本次借用 real active buffer 时生成的物理身份；不泄露可逃逸的 raw Vulkan handle。
    pub device_checkpoint_binding_sha256: S14StarwaveSha256,
    pub gate: S14StarfoldK4NextBlockGate,
    pub launch_binding_sha256: S14StarwaveSha256,
    pub legacy_union_calls: u32,
    pub legacy_grouped_moe_calls: u32,
    pub serial_token_forward_calls: u32,
    pub cpu_fallback_calls: u32,
}

/// `begin_second_k4` 的兼容回执；新代码应消费 `S14StarfoldK4NextBlockLaunchBinding`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldK4SecondBlockLaunchBinding {
    pub schema_version: u32,
    pub previous_committed_position: u32,
    pub previous_committed_epoch: u64,
    pub previous_committed_checkpoint_sha256: S14StarwaveSha256,
    pub previous_commit_chain_sha256: S14StarwaveSha256,
    pub second_block_base_position: u32,
    pub second_block_base_epoch: u64,
    pub second_block_base_checkpoint_sha256: S14StarwaveSha256,
    pub input_token_id: u32,
    pub device_checkpoint_state_bytes: u64,
    pub device_checkpoint_epoch: u64,
    pub device_checkpoint_active_bank: usize,
    pub device_checkpoint_binding_sha256: S14StarwaveSha256,
    pub gate: S14StarfoldK4SecondBlockGate,
    pub launch_binding_sha256: S14StarwaveSha256,
    pub legacy_union_calls: u32,
    pub legacy_grouped_moe_calls: u32,
    pub serial_token_forward_calls: u32,
    pub cpu_fallback_calls: u32,
}

pub type S14StarfoldK4SecondBlockGate = S14StarfoldK4NextBlockGate;

/// 连续 K4 commit 的唯一 chain owner。每次后续 execute 必须观察到 adapter 新增的一次真实
/// committed-state provider rebind；只调用 launch gate 而未重绑会 fail-closed。
#[derive(Debug, Default)]
pub struct S14StarfoldK4TerminalChainOwner {
    committed_blocks: u64,
    last_commit: Option<S14StarfoldK4CommitReceipt>,
    initial_last_finished_base_position: Option<u32>,
}

impl S14StarfoldK4TerminalChainOwner {
    pub fn new() -> Self {
        Self::default()
    }

    /// 最后一个 teacher-forced block 已经通过独立原子提交并完成 adapter rebind 后，
    /// generation chain 从全局 block lineage 继续计数；不伪造 generation receipt。
    pub fn after_teacher_forced_prefill(
        committed_prefill_blocks: u64,
        last_prefill_base_position: u32,
    ) -> Result<Self> {
        if committed_prefill_blocks == 0 {
            bail!("prefill lineage 必须至少包含一个已提交 block");
        }
        Ok(Self {
            committed_blocks: committed_prefill_blocks,
            last_commit: None,
            initial_last_finished_base_position: Some(last_prefill_base_position),
        })
    }

    pub fn last_commit(&self) -> Option<&S14StarfoldK4CommitReceipt> {
        self.last_commit.as_ref()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_sealed_k4<H>(
        &mut self,
        adapter: &mut S14StarfoldK4ProductionAdapter<S14StarfoldConcreteK4Stage<H>>,
        terminal: &mut S14StarfoldTerminalEndpoint,
        terminal_owners: S14CausalBlockStarfoldTerminalBlockOwners,
        full_depth: S14StarfoldK4FullDepthReceipt,
        draft_token_ids: &[u32; S14_STARFOLD_K4_BLOCK_SIZE],
        authoritative: &mut DecoderStateV1,
        device_state: &mut WholeTokenDeviceState,
    ) -> Result<S14StarfoldK4CommitReceipt>
    where
        H: S14CausalBlockVulkanHcQkvAdapter,
    {
        self.execute_sealed_k4_with_commit_limit(
            adapter,
            terminal,
            terminal_owners,
            full_depth,
            draft_token_ids,
            S14_STARFOLD_K4_BLOCK_SIZE,
            authoritative,
            device_state,
        )
    }

    /// 显式限制本次可发布的最长 prefix。terminal 仍计算完整 K4，但 host/device prepare
    /// 直接选择 limit 内 checkpoint，绝不会先发布完整 K4 再隐藏后缀。
    #[allow(clippy::too_many_arguments)]
    pub fn execute_sealed_k4_with_commit_limit<H>(
        &mut self,
        adapter: &mut S14StarfoldK4ProductionAdapter<S14StarfoldConcreteK4Stage<H>>,
        terminal: &mut S14StarfoldTerminalEndpoint,
        terminal_owners: S14CausalBlockStarfoldTerminalBlockOwners,
        full_depth: S14StarfoldK4FullDepthReceipt,
        draft_token_ids: &[u32; S14_STARFOLD_K4_BLOCK_SIZE],
        commit_limit: usize,
        authoritative: &mut DecoderStateV1,
        device_state: &mut WholeTokenDeviceState,
    ) -> Result<S14StarfoldK4CommitReceipt>
    where
        H: S14CausalBlockVulkanHcQkvAdapter,
    {
        let adapter_lineage_valid = adapter.validated_blocks() == self.committed_blocks
            && adapter.committed_rebinds() == self.committed_blocks
            && match self.last_commit.as_ref() {
                None => {
                    adapter.last_finished_base_position()
                        == self.initial_last_finished_base_position
                }
                Some(previous) => {
                    self.committed_blocks != 0
                        && previous.block_sequence == self.committed_blocks
                        && adapter.last_finished_base_position() == Some(previous.base_position)
                }
            };
        if !adapter_lineage_valid {
            return Err(abort_sealed_after_error(
                adapter,
                terminal,
                anyhow!(
                    "S14 StarFold adapter validated/rebind lineage 未证明本 block 消费上一 committed state"
                ),
            ));
        }
        if !(1..=S14_STARFOLD_K4_BLOCK_SIZE).contains(&commit_limit) {
            return Err(abort_sealed_after_error(
                adapter,
                terminal,
                anyhow!("S14 StarFold K4 commit_limit 必须为 1..=4"),
            ));
        }
        let base_state = authoritative.clone();
        let base_sha256 = match decoder_state_sha256(&base_state) {
            Ok(sha256) => sha256,
            Err(error) => {
                return Err(abort_sealed_after_error(adapter, terminal, error));
            }
        };
        let block_sequence = match self.committed_blocks.checked_add(1) {
            Some(sequence) => sequence,
            None => {
                return Err(abort_sealed_after_error(
                    adapter,
                    terminal,
                    anyhow!("S14 StarFold K4 block sequence overflow"),
                ));
            }
        };
        let previous_chain_sha256 = self
            .last_commit
            .as_ref()
            .map_or(S14StarwaveSha256::ZERO, |receipt| {
                receipt.commit_chain_sha256
            });

        if let Err(error) = validate_sealed_base(
            &full_depth,
            draft_token_ids,
            &base_state,
            base_sha256,
            self.last_commit.as_ref(),
            device_state,
        ) {
            return Err(abort_sealed_after_error(adapter, terminal, error));
        }

        let host_candidates = match S14CausalBlockDeferredHostCandidateBatch::new(
            base_state.clone(),
            draft_token_ids.to_vec(),
            Arc::clone(&terminal_owners.prefix_checkpoint_arena),
        ) {
            Ok(candidates) => candidates,
            Err(error) => {
                return Err(abort_sealed_after_error(
                    adapter,
                    terminal,
                    anyhow!("构造 S14 StarFold deferred host candidates 失败: {error:#}"),
                ));
            }
        };
        let terminal_context = Arc::clone(&terminal_owners.context);
        let terminal_assets = terminal_owners.terminal_assets;
        let final_output = match terminal.execute_block(S14StarfoldTerminalBlockInputs {
            context: Arc::clone(&terminal_owners.context),
            base_position: full_depth.base_position,
            final_hidden: full_depth.final_hidden,
            final_hidden_owner: terminal_owners.final_hidden,
            prefix_checkpoint_arena: terminal_owners.prefix_checkpoint_arena,
            paged_arena: terminal_owners.paged_arena,
            head_manifest: terminal_assets.manifest,
            head_weight_plan: terminal_assets.weight_plan,
            head_upload: terminal_assets.head_upload,
            routes_by_position: full_depth.routes_by_position.clone(),
            host_candidates: Box::new(host_candidates),
        }) {
            Ok(output) => output.terminal,
            Err(error) => {
                return Err(abort_sealed_after_error(
                    adapter,
                    terminal,
                    anyhow!("S14 StarFold K4 production terminal/head 失败: {error}"),
                ));
            }
        };

        let prepared_host = match validate_and_prepare_host_commit(
            &base_state,
            base_sha256,
            draft_token_ids,
            &full_depth,
            &final_output,
            commit_limit,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                drop(final_output);
                return Err(abort_sealed_after_error(adapter, terminal, error));
            }
        };

        let prepared_device = match device_state.prepare_starfold_block_prefix_commit(
            terminal_context.as_ref(),
            &final_output.device_future,
            prepared_host.committed_tokens,
            prepared_host.checkpoint_index,
            &prepared_host.committed_checkpoint,
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                drop(final_output);
                return Err(abort_sealed_after_error(
                    adapter,
                    terminal,
                    error.context("prepare S14 StarFold K4 device prefix checkpoint"),
                ));
            }
        };

        if authoritative != &base_state {
            let rollback = device_state.rollback_prepared_block_commit(prepared_device);
            drop(final_output);
            return Err(abort_sealed_after_error(
                adapter,
                terminal,
                anyhow!(
                    "S14 StarFold K4 terminal 阶段改写了 authoritative state; device rollback={rollback:?}"
                ),
            ));
        }

        let future_receipt = final_output.device_future.receipt();
        let commit_chain_sha256 = commit_chain_sha256(
            block_sequence,
            previous_chain_sha256,
            prepared_host.physical_evidence_sha256,
            prepared_host.committed_checkpoint_sha256,
            commit_limit,
            prepared_host.committed_tokens,
            prepared_host.checkpoint_index,
        );

        let receipt = S14StarfoldK4CommitReceipt {
            schema_version: S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION,
            block_sequence,
            base_position: base_state.position,
            base_commit_epoch: base_state.commit_epoch,
            base_checkpoint_sha256: base_sha256,
            previous_commit_chain_sha256: previous_chain_sha256,
            draft_token_ids: *draft_token_ids,
            predicted_token_ids: prepared_host.predicted_token_ids,
            commit_limit,
            decision: prepared_host.decision,
            sealed_full_depth: full_depth.clone(),
            candidate_checkpoints: prepared_host.candidate_checkpoints,
            committed_tokens: prepared_host.committed_tokens,
            checkpoint_index: prepared_host.checkpoint_index,
            committed_position: prepared_host.committed_checkpoint.position,
            committed_epoch: prepared_host.committed_checkpoint.commit_epoch,
            committed_active_bank: usize::from(
                prepared_host.committed_checkpoint.active_fixed_bank,
            ),
            committed_input_token_id: prepared_host.committed_checkpoint.input_token_id,
            committed_checkpoint_sha256: prepared_host.committed_checkpoint_sha256,
            host_device_checkpoint_bytes_verified: true,
            device_commit: WholeTokenDeviceBlockCommitReceipt {
                epoch: prepared_host.committed_checkpoint.commit_epoch,
                active_bank: usize::from(prepared_host.committed_checkpoint.active_fixed_bank),
                position: prepared_host.committed_checkpoint.position,
                accepted_tokens: prepared_host.committed_tokens,
                checkpoint_index: prepared_host.checkpoint_index,
                host_device_bytes_verified: true,
            },
            device_ready_timeline_value: future_receipt.ready_timeline_value,
            device_checkpoint_arena_bytes: future_receipt.checkpoint_arena_bytes,
            physical_evidence_sha256: prepared_host.physical_evidence_sha256,
            commit_chain_sha256,
            terminal_head_submit_calls: final_output.batched_head_submit_calls,
            checkpoint_export_calls: final_output.checkpoint_export_calls,
            legacy_union_calls: 0,
            legacy_grouped_moe_calls: 0,
            serial_token_forward_calls: 0,
            cpu_fallback_calls: 0,
            durable_checkpoint_persisted: false,
        };

        // 此调用仍可能失败，所以必须发生在 host/device 权威发布之前。
        if let Err(error) = adapter.finish_validated_block() {
            let rollback = device_state.rollback_prepared_block_commit(prepared_device);
            drop(final_output);
            return Err(abort_sealed_after_error(
                adapter,
                terminal,
                anyhow!(
                    "释放 S14 StarFold sealed block 失败: {error:#}; device rollback={rollback:?}"
                ),
            ));
        }

        // 自此以后只允许不可失败的 owner swap / 元数据发布。
        *authoritative = prepared_host.committed_checkpoint;
        let published_device = device_state.publish_prepared_block_commit(prepared_device);
        assert_published_identity(authoritative, &receipt, published_device);

        self.committed_blocks = block_sequence;
        self.last_commit = Some(receipt.clone());
        drop(final_output);
        Ok(receipt)
    }

    /// 从任意上一块真实提交生成下一块 anchor。返回的 device binding 来自当前 active bank；
    /// committed-state provider 与新 input hidden 完成 production 重绑前，本函数明确停在物理 gate。
    pub fn begin_next_k4(
        &self,
        authoritative: &DecoderStateV1,
        device_state: &WholeTokenDeviceState,
    ) -> Result<S14StarfoldK4NextBlockLaunchBinding> {
        if self.committed_blocks == 0 {
            bail!("begin_next_k4 要求至少一个已提交 K4 block");
        }
        let previous = self
            .last_commit
            .as_ref()
            .context("begin_next_k4 缺少上一块 commit receipt")?;
        if previous.block_sequence != self.committed_blocks {
            bail!("begin_next_k4 的 block sequence 与上一 commit receipt 漂移");
        }
        if !(1..=S14_STARFOLD_K4_BLOCK_SIZE).contains(&previous.commit_limit)
            || previous.committed_tokens > previous.commit_limit
            || previous.decision.committed_token_ids.len() != previous.committed_tokens
            || previous
                .checkpoint_index
                .checked_add(1)
                .is_none_or(|count| count != previous.committed_tokens)
        {
            bail!("上一块 commit limit/decision/checkpoint receipt 不闭合");
        }
        if !(1..=S14_STARFOLD_K4_BLOCK_SIZE).contains(&previous.committed_tokens) {
            bail!("上一块 committed token 数不在 1..=4");
        }
        let committed_tokens_u32 = u32::try_from(previous.committed_tokens)
            .context("上一块 committed token 数无法表示为 u32")?;
        let committed_tokens_u64 = u64::try_from(previous.committed_tokens)
            .context("上一块 committed token 数无法表示为 u64")?;
        let next_block_sequence = previous
            .block_sequence
            .checked_add(1)
            .context("S14 StarFold next block sequence overflow")?;
        let next_base_position = previous
            .base_position
            .checked_add(committed_tokens_u32)
            .context("S14 StarFold next base position overflow")?;
        let next_base_epoch = previous
            .base_commit_epoch
            .checked_add(committed_tokens_u64)
            .context("S14 StarFold next base epoch overflow")?;
        if next_base_position != previous.committed_position
            || next_base_epoch != previous.committed_epoch
        {
            bail!("上一块 commit receipt 未按实际 committed token 数推进 position/epoch");
        }
        authoritative
            .validate()
            .map_err(|error| anyhow!("下一块 base DecoderState 非法: {error}"))?;
        let base_sha256 = decoder_state_sha256(authoritative)?;
        if authoritative.position != next_base_position
            || authoritative.commit_epoch != next_base_epoch
            || usize::from(authoritative.active_fixed_bank) != previous.committed_active_bank
            || base_sha256 != previous.committed_checkpoint_sha256
        {
            bail!("下一块没有消费上一块真实 committed host checkpoint");
        }
        let device_checkpoint = device_state.committed_checkpoint_binding()?;
        let authoritative_state_bytes = u64::try_from(authoritative.native_arena.bytes().len())
            .context("下一块 host checkpoint state bytes 无法表示为 u64")?;
        if device_checkpoint.epoch() != previous.committed_epoch
            || device_checkpoint.active_bank() != previous.committed_active_bank
            || device_checkpoint.state_bytes() != authoritative_state_bytes
        {
            bail!("下一块 committed device checkpoint 与上一块回执漂移");
        }
        let device_checkpoint_binding_sha256 = next_block_device_binding_sha256(
            next_block_sequence,
            previous,
            authoritative.input_token_id,
            &device_checkpoint,
        );
        let launch_binding_sha256 = next_block_launch_sha256(
            next_block_sequence,
            previous,
            authoritative.input_token_id,
            device_checkpoint_binding_sha256,
        );
        Ok(S14StarfoldK4NextBlockLaunchBinding {
            schema_version: S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION,
            next_block_sequence,
            previous_committed_position: previous.committed_position,
            previous_committed_epoch: previous.committed_epoch,
            previous_committed_checkpoint_sha256: previous.committed_checkpoint_sha256,
            previous_commit_chain_sha256: previous.commit_chain_sha256,
            next_block_base_position: next_base_position,
            next_block_base_epoch: next_base_epoch,
            next_block_base_checkpoint_sha256: base_sha256,
            input_token_id: authoritative.input_token_id,
            device_checkpoint_state_bytes: device_checkpoint.state_bytes(),
            device_checkpoint_epoch: device_checkpoint.epoch(),
            device_checkpoint_active_bank: device_checkpoint.active_bank(),
            device_checkpoint_binding_sha256,
            gate: S14StarfoldK4NextBlockGate::AwaitingCommittedStateProviderAndInputHiddenRebind,
            launch_binding_sha256,
            legacy_union_calls: 0,
            legacy_grouped_moe_calls: 0,
            serial_token_forward_calls: 0,
            cpu_fallback_calls: 0,
        })
    }

    /// 兼容旧的两块 demo 调用；核心校验与 identity 均委托任意序列的 `begin_next_k4`。
    pub fn begin_second_k4(
        &self,
        authoritative: &DecoderStateV1,
        device_state: &WholeTokenDeviceState,
    ) -> Result<S14StarfoldK4SecondBlockLaunchBinding> {
        if self.committed_blocks != 1 {
            bail!("begin_second_k4 兼容入口只接受第一块之后的 launch");
        }
        let next = self.begin_next_k4(authoritative, device_state)?;
        Ok(S14StarfoldK4SecondBlockLaunchBinding {
            schema_version: next.schema_version,
            previous_committed_position: next.previous_committed_position,
            previous_committed_epoch: next.previous_committed_epoch,
            previous_committed_checkpoint_sha256: next.previous_committed_checkpoint_sha256,
            previous_commit_chain_sha256: next.previous_commit_chain_sha256,
            second_block_base_position: next.next_block_base_position,
            second_block_base_epoch: next.next_block_base_epoch,
            second_block_base_checkpoint_sha256: next.next_block_base_checkpoint_sha256,
            input_token_id: next.input_token_id,
            device_checkpoint_state_bytes: next.device_checkpoint_state_bytes,
            device_checkpoint_epoch: next.device_checkpoint_epoch,
            device_checkpoint_active_bank: next.device_checkpoint_active_bank,
            device_checkpoint_binding_sha256: next.device_checkpoint_binding_sha256,
            gate: next.gate,
            launch_binding_sha256: next.launch_binding_sha256,
            legacy_union_calls: next.legacy_union_calls,
            legacy_grouped_moe_calls: next.legacy_grouped_moe_calls,
            serial_token_forward_calls: next.serial_token_forward_calls,
            cpu_fallback_calls: next.cpu_fallback_calls,
        })
    }
}

struct PreparedHostCommit {
    predicted_token_ids: [u32; S14_STARFOLD_K4_BLOCK_SIZE],
    decision: LongestPrefixDecision,
    candidate_checkpoints: Vec<S14StarfoldK4CandidateCheckpointIdentity>,
    committed_tokens: usize,
    checkpoint_index: usize,
    committed_checkpoint: DecoderStateV1,
    committed_checkpoint_sha256: S14StarwaveSha256,
    physical_evidence_sha256: S14StarwaveSha256,
}

fn validate_sealed_base(
    full_depth: &S14StarfoldK4FullDepthReceipt,
    draft_token_ids: &[u32; S14_STARFOLD_K4_BLOCK_SIZE],
    base: &DecoderStateV1,
    base_sha256: S14StarwaveSha256,
    previous: Option<&S14StarfoldK4CommitReceipt>,
    device_state: &WholeTokenDeviceState,
) -> Result<()> {
    base.validate()
        .map_err(|error| anyhow!("base DecoderState 非法: {error}"))?;
    if draft_token_ids.iter().any(|&token| token >= VOCAB_SIZE) {
        bail!("S14 StarFold K4 draft token 越出冻结 vocab");
    }
    if full_depth.source != polaris_s14_runner::MaterializedTokenSource::SpeculativeDraft
        || full_depth.base_position != base.position
        || full_depth.block_size != S14_STARFOLD_K4_BLOCK_SIZE
        || full_depth.physical_input_token_ids.as_slice() != &draft_token_ids[..]
        || full_depth.completed_layers != FULL_DEPTH_LAYERS.len()
        || full_depth.layers.len() != FULL_DEPTH_LAYERS.len()
        || full_depth.routes_by_position.len() != S14_STARFOLD_K4_BLOCK_SIZE
        || full_depth.serial_token_forward_calls != 0
        || !full_depth.terminal_ready
        || full_depth.token_committed
    {
        bail!("S14 StarFold K4 FullDepth43 seal/K/terminal identity 非法");
    }
    let expected_hidden_bytes = (S14_STARFOLD_K4_BLOCK_SIZE as u64)
        .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64)
        .and_then(|elements| elements.checked_mul(2))
        .context("S14 StarFold K4 final hidden bytes overflow")?;
    if full_depth.final_hidden.buffer == vk::Buffer::null()
        || full_depth.final_hidden.offset % 4 != 0
        || full_depth.final_hidden.bytes != expected_hidden_bytes
        || full_depth.final_hidden.block_size != S14_STARFOLD_K4_BLOCK_SIZE
    {
        bail!("S14 StarFold K4 final hidden 不是 device [4,4,4096] BF16");
    }
    let seal = full_depth.checkpoint_seal;
    if seal.base_position != base.position
        || seal.block_size != S14_STARFOLD_K4_BLOCK_SIZE
        || seal.completed_layers != FULL_DEPTH_LAYERS.len()
        || seal.checkpoint_count != S14_STARFOLD_K4_BLOCK_SIZE
        || seal.prefix_program_seal_calls != 1
        || seal.checkpoint_commit_calls != 0
        || seal.serial_token_forward_calls != 0
    {
        bail!("S14 StarFold K4 checkpoint seal 没有闭合 4 个未提交 candidates");
    }
    for (lane, routes) in full_depth.routes_by_position.iter().enumerate() {
        if routes.len() != FULL_DEPTH_LAYERS.len() {
            bail!("S14 StarFold K4 lane {lane} route chain 不完整");
        }
        for (&expected_layer, route) in FULL_DEPTH_LAYERS.iter().zip(routes) {
            if route.layer != expected_layer {
                bail!("S14 StarFold K4 lane {lane} route layer 顺序漂移");
            }
            route
                .validate_for(GraphProfile::FullDepth43NativeTop6)
                .map_err(|error| anyhow!("S14 StarFold K4 lane {lane} route 非法: {error}"))?;
        }
    }
    let mut aggregate_packed_uploads = 0u32;
    let mut aggregate_packed_upload_bytes = 0u64;
    let mut aggregate_lane_dispatches = 0u32;
    for (layer_index, (&expected_layer, layer)) in
        FULL_DEPTH_LAYERS.iter().zip(&full_depth.layers).enumerate()
    {
        let checkpoint = layer.checkpoint;
        if layer.layer != expected_layer
            || layer.base_position != base.position
            || layer.routes.len() != S14_STARFOLD_K4_BLOCK_SIZE
            || checkpoint.base_position != base.position
            || checkpoint.layer != expected_layer
            || checkpoint.block_size != S14_STARFOLD_K4_BLOCK_SIZE
            || checkpoint.layer_record_calls != 1
            || checkpoint.command_graph_submit_calls != 1
            || checkpoint.hc_qkv_projection_record_calls != 1
            || checkpoint.attention_recording_calls != 1
            || checkpoint.ffn_hc_router_input_record_calls != 1
            || checkpoint.router_recording_calls != 1
            || checkpoint.serial_token_forward_calls != 0
            || !checkpoint.hc_hidden_integration_complete
            || layer.expert.layer != u16::from(expected_layer)
            || layer.expert.base_position != u64::from(base.position)
            || layer.expert.unique_experts == 0
            || layer.expert.unique_experts > 24
            || layer.expert.serial_token_forward_calls != 0
            || !layer.additional_experts.is_empty()
            || layer.hidden_commit.base_position != base.position
            || layer.hidden_commit.layer != expected_layer
            || layer.hidden_commit.routed_reduce_dispatch_calls == 0
            || layer.hidden_commit.hc_post_dispatch_calls == 0
            || layer.hidden_commit.queue_submit_calls == 0
            || layer.hidden_commit.serial_token_forward_calls != 0
        {
            bail!("S14 StarFold K4 layer {expected_layer} seal 回执漂移");
        }
        validate_hidden_binding(checkpoint.input_hidden, expected_hidden_bytes)?;
        validate_hidden_binding(checkpoint.post_attention_hidden, expected_hidden_bytes)?;
        validate_hidden_binding(layer.hidden_commit.next_hidden, expected_hidden_bytes)?;
        if checkpoint.post_attention_hidden.generation
            != checkpoint
                .input_hidden
                .generation
                .checked_add(1)
                .context("S14 StarFold attention hidden generation overflow")?
            || layer.hidden_commit.next_hidden.generation
                != checkpoint
                    .post_attention_hidden
                    .generation
                    .checked_add(1)
                    .context("S14 StarFold grouped hidden generation overflow")?
        {
            bail!("S14 StarFold K4 layer {expected_layer} hidden generation 不连续");
        }
        if layer_index > 0
            && checkpoint.input_hidden
                != full_depth.layers[layer_index - 1].hidden_commit.next_hidden
        {
            bail!("S14 StarFold K4 layer {expected_layer} 没有消费前层 next hidden");
        }
        for lane in 0..S14_STARFOLD_K4_BLOCK_SIZE {
            if layer.routes[lane] != full_depth.routes_by_position[lane][layer_index] {
                bail!("S14 StarFold K4 layer/lane route evidence 不同源");
            }
        }
        aggregate_packed_uploads = aggregate_packed_uploads
            .checked_add(layer.expert.packed_uploads)
            .context("S14 StarFold packed upload count overflow")?;
        aggregate_packed_upload_bytes = aggregate_packed_upload_bytes
            .checked_add(layer.expert.packed_upload_bytes)
            .context("S14 StarFold packed upload bytes overflow")?;
        aggregate_lane_dispatches = aggregate_lane_dispatches
            .checked_add(layer.expert.lane_dispatches)
            .context("S14 StarFold lane dispatch count overflow")?;
    }
    if full_depth
        .layers
        .last()
        .is_none_or(|layer| layer.hidden_commit.next_hidden != full_depth.final_hidden)
        || aggregate_packed_uploads != full_depth.packed_uploads
        || aggregate_packed_upload_bytes != full_depth.packed_upload_bytes
        || aggregate_lane_dispatches != full_depth.lane_dispatches
    {
        bail!("S14 StarFold K4 final hidden/physical aggregate 与 43 层链不闭合");
    }
    let base_state_bytes = u64::try_from(base.native_arena.bytes().len())
        .context("S14 StarFold base state bytes 无法表示为 u64")?;
    if device_state.epoch() != base.commit_epoch
        || device_state.active_bank() != usize::from(base.active_fixed_bank)
        || device_state.state_bytes() != base_state_bytes
    {
        bail!("S14 StarFold K4 host/device base epoch/bank/state bytes 漂移");
    }
    if let Some(previous) = previous {
        if previous.committed_position != base.position
            || previous.committed_epoch != base.commit_epoch
            || previous.committed_active_bank != usize::from(base.active_fixed_bank)
            || previous.committed_checkpoint_sha256 != base_sha256
        {
            bail!("后续 K4 没有从上一块真实 committed checkpoint 连续启动");
        }
    }
    Ok(())
}

fn validate_hidden_binding(
    binding: crate::s14_causal_block_layer::S14CausalBlockHiddenBinding,
    expected_bytes: u64,
) -> Result<()> {
    if binding.buffer == vk::Buffer::null()
        || binding.offset % 4 != 0
        || binding.bytes != expected_bytes
        || binding.block_size != S14_STARFOLD_K4_BLOCK_SIZE
    {
        bail!("S14 StarFold K4 hidden binding shape/handle 非法");
    }
    Ok(())
}

fn validate_and_prepare_host_commit(
    base: &DecoderStateV1,
    base_sha256: S14StarwaveSha256,
    draft_token_ids: &[u32; S14_STARFOLD_K4_BLOCK_SIZE],
    full_depth: &S14StarfoldK4FullDepthReceipt,
    final_output: &S14CausalBlockFinalOutput,
    commit_limit: usize,
) -> Result<PreparedHostCommit> {
    if !(1..=S14_STARFOLD_K4_BLOCK_SIZE).contains(&commit_limit) {
        bail!("S14 StarFold K4 commit_limit 必须为 1..=4");
    }
    if final_output.batched_head_submit_calls != 1
        || final_output.checkpoint_export_calls != 1
        || final_output.serial_token_forward_calls != 0
        || final_output.output.mode != BATCHED_CAUSAL_WHOLE_TOKEN_MODE
        || final_output.output.forward_calls != 1
        || final_output.output.positions.len() != S14_STARFOLD_K4_BLOCK_SIZE
    {
        bail!("S14 StarFold K4 terminal/head/checkpoint 强回执漂移");
    }
    let expected_shape =
        S14HeadChunkArgmaxShape::production_batched(S14_STARFOLD_K4_BLOCK_SIZE as u32)
            .context("构造 S14 StarFold K4 production head shape")?;
    if final_output.head_recording.shape != expected_shape
        || final_output.head_recording.submitted_chunks != expected_shape.chunk_count()
        || final_output.head_recording.expected_next_token != expected_shape.vocab
        || final_output.head_results.len() != S14_STARFOLD_K4_BLOCK_SIZE
    {
        bail!("S14 StarFold K4 head 未证明一次 K-row/32-chunk 扫描");
    }
    for (lane, (head, position)) in final_output
        .head_results
        .iter()
        .zip(&final_output.output.positions)
        .enumerate()
    {
        if head.token_id != position.predicted_token_id
            || head.token_id >= VOCAB_SIZE
            || !head.logit.is_finite()
        {
            bail!("S14 StarFold K4 GPU head lane {lane} token/logit 漂移");
        }
    }
    let device_receipt = final_output.device_future.receipt();
    device_receipt
        .validate(
            full_depth.base_position,
            S14_STARFOLD_K4_BLOCK_SIZE,
            full_depth.final_hidden,
        )
        .map_err(|error| anyhow!("S14 StarFold K4 device future 非法: {error}"))?;
    if device_receipt.storage != S14CausalBlockDeviceCheckpointStorage::PrefixCheckpoints
        || device_receipt.checkpoint_count != S14_STARFOLD_K4_BLOCK_SIZE
    {
        bail!("S14 StarFold K4 terminal 没有导出 4 个完整 prefix checkpoints");
    }
    if final_output
        .output
        .positions
        .iter()
        .zip(&full_depth.routes_by_position)
        .any(|(position, routes)| position.routes != *routes)
    {
        bail!("S14 StarFold K4 terminal checkpoint routes 与 43 层 seal 不同源");
    }
    validate_checkpoint_chain(base, draft_token_ids, &final_output.output.positions)?;

    let predicted = final_output
        .output
        .positions
        .iter()
        .map(|position| position.predicted_token_id)
        .collect::<Vec<_>>();
    let predicted_token_ids: [u32; S14_STARFOLD_K4_BLOCK_SIZE] = predicted
        .try_into()
        .map_err(|_| anyhow!("S14 StarFold K4 predicted token 数量漂移"))?;
    let full_decision = decide_longest_prefix(draft_token_ids, &predicted_token_ids)
        .map_err(|error| anyhow!("S14 StarFold K4 最长前缀决策失败: {error}"))?;
    let decision = if full_decision
        .mismatch_index
        .is_some_and(|mismatch_index| mismatch_index < commit_limit)
    {
        full_decision
    } else {
        let committed_token_ids = draft_token_ids[..commit_limit].to_vec();
        LongestPrefixDecision {
            accepted_prefix: committed_token_ids.clone(),
            fallback_token_id: None,
            rejected_draft_suffix: draft_token_ids[commit_limit..].to_vec(),
            committed_token_ids,
            mismatch_index: None,
        }
    };
    let committed_tokens = decision.committed_token_ids.len();
    if !(1..=commit_limit).contains(&committed_tokens) {
        bail!("S14 StarFold K4 最长前缀越出 commit_limit");
    }
    let checkpoint_index = decision
        .committed_token_ids
        .len()
        .checked_sub(1)
        .context("S14 StarFold K4 最长前缀缺少可提交 checkpoint")?;
    if checkpoint_index
        .checked_add(1)
        .context("S14 StarFold K4 checkpoint index overflow")?
        != committed_tokens
    {
        bail!("S14 StarFold K4 最长前缀/checkpoint index 不闭合");
    }

    let mut candidate_checkpoints = Vec::with_capacity(S14_STARFOLD_K4_BLOCK_SIZE);
    for (index, position) in final_output.output.positions.iter().enumerate() {
        candidate_checkpoints.push(S14StarfoldK4CandidateCheckpointIdentity {
            checkpoint_index: index,
            position: position.checkpoint.position,
            commit_epoch: position.checkpoint.commit_epoch,
            active_fixed_bank: position.checkpoint.active_fixed_bank,
            teacher_forced_input_token_id: position.checkpoint.input_token_id,
            predicted_token_id: position.predicted_token_id,
            checkpoint_sha256: decoder_state_sha256(&position.checkpoint)?,
        });
    }

    let mut committed_checkpoint = final_output.output.positions[checkpoint_index]
        .checkpoint
        .clone();
    if let Some(fallback) = decision.fallback_token_id {
        committed_checkpoint.input_token_id = fallback;
    }
    committed_checkpoint
        .validate()
        .map_err(|error| anyhow!("S14 StarFold K4 selected checkpoint 非法: {error}"))?;
    let added_records = committed_checkpoint
        .committed_tokens
        .get(base.committed_tokens.len()..)
        .context("S14 StarFold K4 selected checkpoint ledger 短于 base")?;
    let added_tokens = added_records
        .iter()
        .map(|record| record.predicted_token_id)
        .collect::<Vec<_>>();
    if added_tokens != decision.committed_token_ids {
        bail!("S14 StarFold K4 selected checkpoint ledger 与最长前缀不闭合");
    }
    if added_records.len() != committed_tokens {
        bail!("S14 StarFold K4 selected checkpoint ledger 增量越出 commit_limit");
    }
    let committed_tokens_u32 =
        u32::try_from(committed_tokens).context("S14 StarFold committed token 数无法表示为 u32")?;
    let committed_tokens_u64 =
        u64::try_from(committed_tokens).context("S14 StarFold committed token 数无法表示为 u64")?;
    let expected_committed_position = base
        .position
        .checked_add(committed_tokens_u32)
        .context("S14 StarFold committed position overflow")?;
    let expected_committed_epoch = base
        .commit_epoch
        .checked_add(committed_tokens_u64)
        .context("S14 StarFold committed epoch overflow")?;
    if committed_checkpoint.position != expected_committed_position
        || committed_checkpoint.commit_epoch != expected_committed_epoch
    {
        bail!("S14 StarFold selected checkpoint 未按实际 ledger 增量推进 position/epoch");
    }
    let committed_checkpoint_sha256 = decoder_state_sha256(&committed_checkpoint)?;
    let physical_evidence_sha256 = physical_evidence_sha256(
        base_sha256,
        draft_token_ids,
        full_depth,
        final_output,
        &candidate_checkpoints,
        commit_limit,
        committed_tokens,
        checkpoint_index,
        committed_checkpoint_sha256,
    );
    Ok(PreparedHostCommit {
        predicted_token_ids,
        decision,
        candidate_checkpoints,
        committed_tokens,
        checkpoint_index,
        committed_checkpoint,
        committed_checkpoint_sha256,
        physical_evidence_sha256,
    })
}

fn validate_checkpoint_chain(
    base: &DecoderStateV1,
    draft_token_ids: &[u32; S14_STARFOLD_K4_BLOCK_SIZE],
    positions: &[BatchedWholeTokenPosition],
) -> Result<()> {
    let end = base
        .position
        .checked_add(S14_STARFOLD_K4_BLOCK_SIZE as u32)
        .context("S14 StarFold K4 checkpoint end position overflow")?;
    if end > base.native.max_seq_len {
        bail!("S14 StarFold K4 checkpoint chain 越出 max_seq_len");
    }
    for (offset, position) in positions.iter().enumerate() {
        if position.predicted_token_id >= VOCAB_SIZE {
            bail!("S14 StarFold K4 checkpoint {offset} prediction 越界");
        }
        let checkpoint = &position.checkpoint;
        checkpoint
            .validate()
            .map_err(|error| anyhow!("S14 StarFold K4 checkpoint {offset} 非法: {error}"))?;
        let offset_u32 =
            u32::try_from(offset).context("S14 StarFold K4 checkpoint offset 无法表示为 u32")?;
        let offset_u64 =
            u64::try_from(offset).context("S14 StarFold K4 checkpoint offset 无法表示为 u64")?;
        let offset_u8 =
            u8::try_from(offset).context("S14 StarFold K4 checkpoint offset 无法表示为 u8")?;
        let offset_plus_one_u32 = offset_u32
            .checked_add(1)
            .context("S14 StarFold K4 checkpoint offset overflow")?;
        let offset_plus_one_u64 = offset_u64
            .checked_add(1)
            .context("S14 StarFold K4 checkpoint epoch offset overflow")?;
        let offset_plus_one_u8 = offset_u8
            .checked_add(1)
            .context("S14 StarFold K4 checkpoint bank offset overflow")?;
        let expected_position = base
            .position
            .checked_add(offset_plus_one_u32)
            .context("S14 StarFold K4 checkpoint position overflow")?;
        let expected_epoch = base
            .commit_epoch
            .checked_add(offset_plus_one_u64)
            .context("S14 StarFold K4 checkpoint epoch overflow")?;
        let expected_bank = base.active_fixed_bank ^ (offset_plus_one_u8 & 1);
        if checkpoint.position != expected_position
            || checkpoint.commit_epoch != expected_epoch
            || checkpoint.active_fixed_bank != expected_bank
            || checkpoint.input_token_id != draft_token_ids[offset]
            || checkpoint.native.max_seq_len != base.native.max_seq_len
        {
            bail!("S14 StarFold K4 checkpoint {offset} position/epoch/bank/teacher-force 漂移");
        }
        if checkpoint.committed_tokens[..base.committed_tokens.len()] != base.committed_tokens[..] {
            bail!("S14 StarFold K4 checkpoint {offset} 改写已提交 ledger");
        }
        let offset_plus_one = offset
            .checked_add(1)
            .context("S14 StarFold K4 ledger offset overflow")?;
        let expected_len = base
            .committed_tokens
            .len()
            .checked_add(offset_plus_one)
            .context("S14 StarFold K4 checkpoint ledger length overflow")?;
        if checkpoint.committed_tokens.len() != expected_len {
            bail!("S14 StarFold K4 checkpoint {offset} ledger 长度漂移");
        }
        if offset > 0 {
            let previous_len = expected_len
                .checked_sub(1)
                .context("S14 StarFold K4 previous checkpoint ledger length underflow")?;
            let previous_offset = offset
                .checked_sub(1)
                .context("S14 StarFold K4 previous checkpoint offset underflow")?;
            if checkpoint.committed_tokens[..previous_len]
                != positions[previous_offset].checkpoint.committed_tokens[..]
            {
                bail!("S14 StarFold K4 checkpoint {offset} 不是前一 checkpoint 的连续子代");
            }
        }
        let record = checkpoint
            .committed_tokens
            .last()
            .context("S14 StarFold K4 checkpoint 缺少新增 token record")?;
        let expected_input = if offset == 0 {
            base.input_token_id
        } else {
            draft_token_ids[offset - 1]
        };
        let expected_record_position = base
            .position
            .checked_add(offset_u32)
            .context("S14 StarFold K4 token record position overflow")?;
        if record.position != expected_record_position
            || record.input_token_id != expected_input
            || record.predicted_token_id != position.predicted_token_id
        {
            bail!("S14 StarFold K4 checkpoint {offset} ledger 与 GPU head 不同源");
        }
    }
    Ok(())
}

/// Production API 映射 checkpoint identity 的唯一 SHA helper。序列化域与 terminal
/// commit receipt 完全相同，调用方不得另行合成 position/epoch 摘要。
pub fn s14_starfold_decoder_state_sha256(state: &DecoderStateV1) -> Result<S14StarwaveSha256> {
    state
        .validate()
        .map_err(|error| anyhow!("DecoderState SHA 输入非法: {error}"))?;
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-decoder-checkpoint-v1");
    writer.write_u32(state.abi_version);
    writer.write_u64(state.commit_epoch);
    writer.write_u32(state.position);
    writer.write_u32(state.input_token_id);
    writer.write_u8(state.active_fixed_bank);
    writer.write_u8(match state.native.profile {
        GraphProfile::S14Top6 => 0,
        GraphProfile::FullDepth43NativeTop6 => 1,
    });
    writer.write_u32(state.native.position);
    writer.write_u32(state.native.max_seq_len);
    writer.write_u64(state.native.arena_bytes);
    writer.write_u8(state.native.poisoned as u8);
    writer.write_u64(state.native_arena.arena_id());
    writer.write_bytes(state.native_arena.bytes());
    writer.write_u64(state.committed_tokens.len() as u64);
    for record in &state.committed_tokens {
        writer.write_u32(record.position);
        writer.write_u32(record.input_token_id);
        writer.write_u32(record.predicted_token_id);
    }
    Ok(writer.finish())
}

fn decoder_state_sha256(state: &DecoderStateV1) -> Result<S14StarwaveSha256> {
    s14_starfold_decoder_state_sha256(state)
}

#[allow(clippy::too_many_arguments)]
fn physical_evidence_sha256(
    base_sha256: S14StarwaveSha256,
    draft_token_ids: &[u32; S14_STARFOLD_K4_BLOCK_SIZE],
    full_depth: &S14StarfoldK4FullDepthReceipt,
    final_output: &S14CausalBlockFinalOutput,
    checkpoints: &[S14StarfoldK4CandidateCheckpointIdentity],
    commit_limit: usize,
    committed_tokens: usize,
    checkpoint_index: usize,
    committed_checkpoint_sha256: S14StarwaveSha256,
) -> S14StarwaveSha256 {
    let mut writer =
        S14StarwaveProofWriter::new("polaris-s14-starfold-k4-terminal-physical-evidence-v1");
    writer.write_u32(S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION);
    writer.write_sha256(base_sha256);
    writer.write_u32(full_depth.base_position);
    writer.write_u64(full_depth.completed_layers as u64);
    writer.write_u64(full_depth.block_size as u64);
    writer.write_u64(full_depth.final_hidden.buffer.as_raw());
    writer.write_u64(full_depth.final_hidden.offset);
    writer.write_u64(full_depth.final_hidden.bytes);
    writer.write_u64(full_depth.final_hidden.generation);
    writer.write_u32(full_depth.packed_uploads);
    writer.write_u64(full_depth.packed_upload_bytes);
    writer.write_u32(full_depth.lane_dispatches);
    for layer in &full_depth.layers {
        writer.write_u8(layer.layer);
        writer.write_u32(layer.base_position);
        write_hidden_binding(&mut writer, layer.checkpoint.input_hidden);
        write_hidden_binding(&mut writer, layer.checkpoint.post_attention_hidden);
        writer.write_u32(layer.checkpoint.command_graph_submit_calls);
        writer.write_u32(layer.expert.unique_experts);
        writer.write_u32(layer.expert.packed_uploads);
        writer.write_u64(layer.expert.packed_upload_bytes);
        writer.write_u32(layer.expert.lane_dispatches);
        write_hidden_binding(&mut writer, layer.hidden_commit.next_hidden);
        writer.write_u32(layer.hidden_commit.routed_reduce_dispatch_calls);
        writer.write_u32(layer.hidden_commit.hc_post_dispatch_calls);
        writer.write_u32(layer.hidden_commit.queue_submit_calls);
    }
    for token in draft_token_ids {
        writer.write_u32(*token);
    }
    for routes in &full_depth.routes_by_position {
        writer.write_u64(routes.len() as u64);
        for route in routes {
            write_route(&mut writer, route);
        }
    }
    writer.write_u32(final_output.batched_head_submit_calls);
    writer.write_u32(final_output.checkpoint_export_calls);
    writer.write_u32(final_output.head_recording.submitted_chunks);
    for head in &final_output.head_results {
        writer.write_u32(head.token_id);
        writer.write_f32(head.logit);
    }
    let future = final_output.device_future.receipt();
    writer.write_u64(future.checkpoint_arena.as_raw());
    writer.write_u64(future.checkpoint_arena_offset);
    writer.write_u64(future.checkpoint_arena_bytes);
    writer.write_u64(future.checkpoint_stride_bytes);
    writer.write_u64(future.checkpoint_state_bytes);
    writer.write_u64(future.ready_timeline.as_raw());
    writer.write_u64(future.ready_timeline_value);
    for checkpoint in checkpoints {
        writer.write_u64(checkpoint.checkpoint_index as u64);
        writer.write_u32(checkpoint.position);
        writer.write_u64(checkpoint.commit_epoch);
        writer.write_u8(checkpoint.active_fixed_bank);
        writer.write_u32(checkpoint.teacher_forced_input_token_id);
        writer.write_u32(checkpoint.predicted_token_id);
        writer.write_sha256(checkpoint.checkpoint_sha256);
    }
    writer.write_u64(commit_limit as u64);
    writer.write_u64(committed_tokens as u64);
    writer.write_u64(checkpoint_index as u64);
    writer.write_sha256(committed_checkpoint_sha256);
    writer.write_u8(1); // StarFold prepare 已逐字节核对 selected host/device checkpoint。
    writer.finish()
}

fn commit_chain_sha256(
    block_sequence: u64,
    previous_commit_chain_sha256: S14StarwaveSha256,
    physical_evidence_sha256: S14StarwaveSha256,
    committed_checkpoint_sha256: S14StarwaveSha256,
    commit_limit: usize,
    committed_tokens: usize,
    checkpoint_index: usize,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starfold-k4-atomic-commit-chain-v1");
    writer.write_u32(S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION);
    writer.write_u64(block_sequence);
    writer.write_sha256(previous_commit_chain_sha256);
    writer.write_sha256(physical_evidence_sha256);
    writer.write_sha256(committed_checkpoint_sha256);
    writer.write_u64(commit_limit as u64);
    writer.write_u64(committed_tokens as u64);
    writer.write_u64(checkpoint_index as u64);
    writer.finish()
}

fn next_block_device_binding_sha256(
    next_block_sequence: u64,
    previous: &S14StarfoldK4CommitReceipt,
    input_token_id: u32,
    device: &WholeTokenDeviceCommittedCheckpointBinding<'_>,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starfold-next-k4-device-binding-v1");
    writer.write_u32(S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION);
    writer.write_u64(next_block_sequence);
    writer.write_u64(previous.block_sequence);
    writer.write_u32(previous.committed_position);
    writer.write_u64(previous.committed_epoch);
    writer.write_sha256(previous.committed_checkpoint_sha256);
    writer.write_sha256(previous.commit_chain_sha256);
    writer.write_u32(input_token_id);
    writer.write_u64(device.buffer().as_raw());
    writer.write_u64(device.state_bytes());
    writer.write_u64(device.epoch());
    writer.write_u64(device.active_bank() as u64);
    writer.finish()
}

fn next_block_launch_sha256(
    next_block_sequence: u64,
    previous: &S14StarfoldK4CommitReceipt,
    input_token_id: u32,
    device_checkpoint_binding_sha256: S14StarwaveSha256,
) -> S14StarwaveSha256 {
    let mut writer = S14StarwaveProofWriter::new("polaris-s14-starfold-next-k4-launch-binding-v1");
    writer.write_u32(S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION);
    writer.write_u64(next_block_sequence);
    writer.write_sha256(previous.commit_chain_sha256);
    writer.write_sha256(previous.committed_checkpoint_sha256);
    writer.write_u32(previous.committed_position);
    writer.write_u64(previous.committed_epoch);
    writer.write_u32(input_token_id);
    writer.write_sha256(device_checkpoint_binding_sha256);
    writer.finish()
}

fn write_route(writer: &mut S14StarwaveProofWriter, route: &RouteDecision) {
    writer.write_u8(route.layer);
    writer.write_u8(match route.kind {
        RouterKind::Hash => 0,
        RouterKind::Score => 1,
    });
    writer.write_u64(route.expert_ids.len() as u64);
    for expert in &route.expert_ids {
        writer.write_u32(u32::from(*expert));
    }
    writer.write_u64(route.weights.len() as u64);
    for weight in &route.weights {
        writer.write_f32(*weight);
    }
}

fn write_hidden_binding(
    writer: &mut S14StarwaveProofWriter,
    binding: crate::s14_causal_block_layer::S14CausalBlockHiddenBinding,
) {
    writer.write_u64(binding.buffer.as_raw());
    writer.write_u64(binding.offset);
    writer.write_u64(binding.bytes);
    writer.write_u64(binding.block_size as u64);
    writer.write_u64(binding.generation);
}

fn assert_published_identity(
    authoritative: &DecoderStateV1,
    receipt: &S14StarfoldK4CommitReceipt,
    published: WholeTokenDeviceBlockCommitReceipt,
) {
    assert!((1..=S14_STARFOLD_K4_BLOCK_SIZE).contains(&receipt.commit_limit));
    assert!((1..=receipt.commit_limit).contains(&receipt.committed_tokens));
    assert_eq!(
        receipt.decision.committed_token_ids.len(),
        receipt.committed_tokens
    );
    assert_eq!(authoritative.position, receipt.committed_position);
    assert_eq!(authoritative.commit_epoch, receipt.committed_epoch);
    assert_eq!(
        usize::from(authoritative.active_fixed_bank),
        receipt.committed_active_bank
    );
    assert_eq!(published.position, receipt.committed_position);
    assert_eq!(published.epoch, receipt.committed_epoch);
    assert_eq!(published.active_bank, receipt.committed_active_bank);
    assert_eq!(published.accepted_tokens, receipt.committed_tokens);
    assert_eq!(published.checkpoint_index, receipt.checkpoint_index);
    assert!(published.host_device_bytes_verified);
    assert!(receipt.host_device_checkpoint_bytes_verified);
}

fn abort_sealed_after_error<H>(
    adapter: &mut S14StarfoldK4ProductionAdapter<S14StarfoldConcreteK4Stage<H>>,
    terminal: &mut S14StarfoldTerminalEndpoint,
    original: anyhow::Error,
) -> anyhow::Error
where
    H: S14CausalBlockVulkanHcQkvAdapter,
{
    let terminal_abort = terminal.drain_and_abort_block();
    // Adapter 没有公开的 sealed-only abort；destroy 会真实 drain stage 后释放 owner，
    // 是当前唯一不会把失败 block 留在可复用状态中的 fail-closed 路径。
    let starfold_abort = adapter.destroy();
    anyhow!(
        "{original:#}; terminal drain={terminal_abort:?}; StarFold sealed owner destroy/drain={starfold_abort:?}"
    )
}
