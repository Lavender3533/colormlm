//! 纯 S14 StarFold 的直接 batched terminal endpoint。
//!
//! 该 endpoint 不进入旧 `S14CausalBlockProductionBundle` 生命周期；它把 FullDepth43
//! 已 seal 的 final hidden、K-prefix checkpoint 与 deferred host candidates 直接交给
//! 原生 terminal channel，再执行一次真实 K-row GPU head/checkpoint export。模块不接受
//! 预制 token，不提供 CPU/mock fallback，也不调用旧 union/grouped-MoE 路径。

use crate::{
    s14_causal_block_layer::{S14CausalBlockFinalOutput, S14CausalBlockHiddenBinding},
    s14_causal_block_prefix_arena::S14CausalBlockPrefixCheckpointArena,
    s14_causal_block_terminal::{
        S14CausalBlockBatchedTerminalRecorder, S14CausalBlockCheckpointArenaPool,
        S14CausalBlockCheckpointArenaTelemetry,
    },
    s14_causal_block_terminal_adapter::{
        s14_causal_block_terminal_production_channel, S14CausalBlockHostCandidateFinalizer,
        S14CausalBlockTerminalProductionAdapter, S14CausalBlockTerminalProductionPublisher,
        S14CausalBlockTerminalProviderTelemetry,
    },
    s14_causal_block_terminal_owner::{
        S14CausalBlockOwnedBufferSlice, S14CausalBlockProductionTerminalResourceOwner,
        S14CausalBlockTerminalHeadLeaseOwner, S14CausalBlockTerminalPublishReceipt,
        S14CausalBlockTerminalResourceOwnerInputs,
    },
    s14_causal_block_vulkan_backend::S14CausalBlockVulkanTerminalRecorder,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    VulkanContext,
};
use anyhow::{bail, Context, Result};
use polaris_s14_runner::{Position0WholeTokenManifest, RouteDecision, FULL_DEPTH_LAYERS};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldTerminalEndpointPhase {
    Idle,
    Executing,
    Poisoned,
}

/// 单个 StarFold block 的强所有权输入。`final_hidden_owner` 与 `final_hidden` 必须指向
/// 同一 Vulkan buffer/range；其余 owner 必须来自同一次 FullDepth43 production runtime。
pub struct S14StarfoldTerminalBlockInputs {
    pub context: Arc<VulkanContext>,
    pub base_position: u32,
    pub final_hidden: S14CausalBlockHiddenBinding,
    pub final_hidden_owner: S14CausalBlockOwnedBufferSlice,
    pub prefix_checkpoint_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
    pub paged_arena: Arc<S14Position0PagedWeightArena>,
    pub head_manifest: Arc<Position0WholeTokenManifest>,
    pub head_weight_plan: Arc<S14Position0HybridWeightPlan>,
    pub head_upload: S14CausalBlockTerminalHeadLeaseOwner,
    pub routes_by_position: Vec<Vec<RouteDecision>>,
    pub host_candidates: Box<dyn S14CausalBlockHostCandidateFinalizer>,
}

#[derive(Debug)]
pub struct S14StarfoldTerminalBlockOutput {
    pub publication: S14CausalBlockTerminalPublishReceipt,
    pub terminal: S14CausalBlockFinalOutput,
}

/// 跨 block 复用 checkpoint lease pool、terminal command/workspace 与 raw channel。
/// block 级 HC/norm owner 仍为 one-shot，并由 channel 强持有到 terminal 完成。
pub struct S14StarfoldTerminalEndpoint {
    context: Arc<VulkanContext>,
    checkpoint_pool: Arc<S14CausalBlockCheckpointArenaPool>,
    publisher: S14CausalBlockTerminalProductionPublisher,
    terminal: S14CausalBlockTerminalProductionAdapter,
    phase: S14StarfoldTerminalEndpointPhase,
}

impl std::fmt::Debug for S14StarfoldTerminalEndpoint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("S14StarfoldTerminalEndpoint")
            .field("context", &Arc::as_ptr(&self.context))
            .field("checkpoint_pool", &self.checkpoint_pool)
            .field("publisher", &self.publisher)
            .field("terminal", &self.terminal)
            .field("phase", &self.phase)
            .finish()
    }
}

impl S14StarfoldTerminalEndpoint {
    pub fn new(
        context: Arc<VulkanContext>,
        checkpoint_state_bytes: u64,
        checkpoint_slots: usize,
    ) -> Result<Self> {
        if !context.timeline_semaphore {
            bail!("StarFold terminal endpoint 要求 timeline semaphore");
        }
        let checkpoint_pool = S14CausalBlockCheckpointArenaPool::new(
            Arc::clone(&context),
            checkpoint_state_bytes,
            checkpoint_slots,
        )?;
        let recorder = S14CausalBlockBatchedTerminalRecorder::new(
            Arc::clone(&context),
            Arc::clone(&checkpoint_pool),
        )?;
        let (publisher, provider) = s14_causal_block_terminal_production_channel();
        let terminal = S14CausalBlockTerminalProductionAdapter::new(recorder, provider);
        Ok(Self {
            context,
            checkpoint_pool,
            publisher,
            terminal,
            phase: S14StarfoldTerminalEndpointPhase::Idle,
        })
    }

    pub fn phase(&self) -> S14StarfoldTerminalEndpointPhase {
        self.phase
    }

    pub fn checkpoint_pool_telemetry(&self) -> Result<S14CausalBlockCheckpointArenaTelemetry> {
        self.checkpoint_pool.telemetry()
    }

    pub fn terminal_channel_telemetry(
        &self,
    ) -> Result<S14CausalBlockTerminalProviderTelemetry, String> {
        self.publisher.telemetry()
    }

    /// 外层 host/device commit 在 terminal 成功后仍可能拒绝本 block。该入口幂等排空 raw
    /// channel；即使 source 已被 terminal 消费，rollback/drain 也只会得到空 pending。
    pub(crate) fn drain_and_abort_block(&mut self) -> Result<(), String> {
        let publisher_rollback = self.publisher.rollback_pending();
        let terminal_drain = self
            .terminal
            .drain_and_abort_batched_terminal(FULL_DEPTH_LAYERS.len());
        if publisher_rollback.is_ok() && terminal_drain.is_ok() {
            self.phase = S14StarfoldTerminalEndpointPhase::Idle;
            Ok(())
        } else {
            self.phase = S14StarfoldTerminalEndpointPhase::Poisoned;
            Err(format!(
                "StarFold terminal drain 失败: publisher={publisher_rollback:?}; terminal={terminal_drain:?}"
            ))
        }
    }

    pub fn execute_block(
        &mut self,
        inputs: S14StarfoldTerminalBlockInputs,
    ) -> Result<S14StarfoldTerminalBlockOutput, String> {
        if self.phase != S14StarfoldTerminalEndpointPhase::Idle {
            return Err(format!(
                "StarFold terminal endpoint 当前 phase={:?}，拒绝新 block",
                self.phase
            ));
        }
        self.phase = S14StarfoldTerminalEndpointPhase::Executing;
        let base_position = inputs.base_position;
        let block_size = inputs.final_hidden.block_size;
        let result = self.execute_block_inner(inputs);
        match result {
            Ok(output) => {
                self.phase = S14StarfoldTerminalEndpointPhase::Idle;
                Ok(output)
            }
            Err(error) => Err(self.rollback_after_error(base_position, block_size, error)),
        }
    }

    fn execute_block_inner(
        &mut self,
        inputs: S14StarfoldTerminalBlockInputs,
    ) -> Result<S14StarfoldTerminalBlockOutput, String> {
        validate_block_inputs(&self.context, &inputs).map_err(|error| error.to_string())?;
        let S14StarfoldTerminalBlockInputs {
            context,
            base_position,
            final_hidden,
            final_hidden_owner,
            prefix_checkpoint_arena,
            paged_arena,
            head_manifest,
            head_weight_plan,
            head_upload,
            routes_by_position,
            host_candidates,
        } = inputs;
        let block_size = final_hidden.block_size;
        let owner = S14CausalBlockProductionTerminalResourceOwner::new(
            S14CausalBlockTerminalResourceOwnerInputs {
                context,
                block_size,
                final_hidden: final_hidden_owner,
                prefix_checkpoint_arena,
                paged_arena,
                head_manifest,
                head_weight_plan,
                head_upload,
            },
        )
        .map_err(|error| format!("构造 StarFold terminal 强 owner 失败: {error:#}"))?;
        let publication = owner.record_and_publish_starfold(
            &self.publisher,
            base_position,
            final_hidden,
            routes_by_position.clone(),
            host_candidates,
        )?;
        let terminal = self.terminal.record_batched_terminal_head_and_checkpoints(
            FULL_DEPTH_LAYERS.len(),
            base_position,
            final_hidden,
            &routes_by_position,
        )?;
        validate_block_output(&publication, &terminal).map_err(|error| error.to_string())?;
        Ok(S14StarfoldTerminalBlockOutput {
            publication,
            terminal,
        })
    }

    fn rollback_after_error(
        &mut self,
        base_position: u32,
        block_size: usize,
        error: String,
    ) -> String {
        let publisher_rollback = self.publisher.rollback_pending();
        let terminal_drain = self
            .terminal
            .drain_and_abort_batched_terminal(FULL_DEPTH_LAYERS.len());
        let cleanup_ok = publisher_rollback.is_ok() && terminal_drain.is_ok();
        self.phase = if cleanup_ok {
            S14StarfoldTerminalEndpointPhase::Idle
        } else {
            S14StarfoldTerminalEndpointPhase::Poisoned
        };
        if cleanup_ok {
            error
        } else {
            format!(
                "StarFold terminal block base={base_position} K={block_size} 失败: {error}; \
                 publisher rollback={publisher_rollback:?}; terminal drain={terminal_drain:?}"
            )
        }
    }
}

fn validate_block_inputs(
    expected_context: &Arc<VulkanContext>,
    inputs: &S14StarfoldTerminalBlockInputs,
) -> Result<()> {
    if !Arc::ptr_eq(expected_context, &inputs.context)
        || !Arc::ptr_eq(expected_context, inputs.prefix_checkpoint_arena.context())
    {
        bail!("StarFold terminal block VulkanContext owner 漂移");
    }
    let owned = &inputs.final_hidden_owner;
    if owned.buffer.handle() != inputs.final_hidden.buffer
        || owned.offset != inputs.final_hidden.offset
        || inputs.final_hidden.block_size != inputs.prefix_checkpoint_arena.layout().block_size
    {
        bail!("StarFold terminal final hidden/prefix arena identity 漂移");
    }
    inputs
        .base_position
        .checked_add(inputs.final_hidden.block_size as u32)
        .context("StarFold terminal block position overflow")?;
    Ok(())
}

fn validate_block_output(
    publication: &S14CausalBlockTerminalPublishReceipt,
    terminal: &S14CausalBlockFinalOutput,
) -> Result<()> {
    let future = terminal.device_future.receipt();
    if publication.completed_layers != FULL_DEPTH_LAYERS.len()
        || publication.predicted_tokens_prebuilt
        || publication.checkpoint_count != publication.block_size
        || terminal.output.forward_calls != 1
        || terminal.output.positions.len() != publication.block_size
        || terminal.head_results.len() != publication.block_size
        || terminal.batched_head_submit_calls != 1
        || terminal.checkpoint_export_calls != 1
        || terminal.serial_token_forward_calls != 0
        || future.base_position != publication.base_position
        || future.block_size != publication.block_size
        || future.checkpoint_count != publication.checkpoint_count
    {
        bail!("StarFold direct terminal publication/GPU head/checkpoint 强回执不闭合");
    }
    Ok(())
}
