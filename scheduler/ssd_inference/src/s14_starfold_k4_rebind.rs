//! StarFold 任意连续真实 K4 block 的 committed-state 重绑定会话。
//!
//! 本模块不构造 fixture，也不重建 StarFold runtime/windows/B4/shared-reduce/HC bridge。
//! 每次上一块完成 terminal 原子发布和 adapter finish/drain 后，只把下一块 production
//! provider、hidden/prefix/ratio4 owners 原地装入常驻 stage，并返回一次性 launch gate。

use crate::{
    s14_causal_block_hc_qkv_recorder::S14CausalBlockHiddenBank,
    s14_causal_block_layer::{S14CausalBlockHiddenBinding, S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE},
    s14_causal_block_production_bundle::{
        validate_context_owner, validate_hidden_banks,
        S14CausalBlockProductionHcQkvResourceProvider,
    },
    s14_starfold_k4_adapter::S14StarfoldK4FullDepthReceipt,
    s14_starfold_k4_terminal_chain::{
        S14StarfoldK4CommitReceipt, S14_STARFOLD_K4_BLOCK_SIZE,
        S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION,
    },
    s14_starfold_production_resources::{
        S14StarfoldK4ProductionResourceInputs, S14StarfoldK4ProductionResources,
    },
    s14_starfold_prompt_prefill::S14StarfoldSealedTeacherForcedBlockCommitReceipt,
    s14_starwave_transaction::S14StarwaveSha256,
    s14_whole_token_device::{WholeTokenDeviceCommittedCheckpointBinding, WholeTokenDeviceState},
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{DecoderStateV1, MaterializedTokenSource, FULL_DEPTH_LAYERS, VOCAB_SIZE};
use std::{fmt, sync::Arc};

const K4: usize = S14_STARFOLD_K4_BLOCK_SIZE;
const BF16_BYTES: u64 = 2;

/// 下一块的显式 ledger 模式。mode 是 source 的唯一 authority，调用方不能自由组合二者。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldK4BlockMode {
    TeacherForcedPrefill,
    SpeculativeGeneration,
}

impl S14StarfoldK4BlockMode {
    pub const fn materialized_source(self) -> MaterializedTokenSource {
        match self {
            Self::TeacherForcedPrefill => MaterializedTokenSource::ForcedPrefill,
            Self::SpeculativeGeneration => MaterializedTokenSource::SpeculativeDraft,
        }
    }
}

/// 上一块 terminal/head 与选定 committed prefix 已经完成原子发布后的强 anchor。
/// 三个共享借用会保留到下一块真正 launch，期间 Rust 所有权禁止改写 host/device owner
/// 或替换 commit receipt。
pub struct S14StarfoldCommittedK4Anchor<'a> {
    pub previous_commit: &'a S14StarfoldK4CommitReceipt,
    pub committed_device: &'a WholeTokenDeviceState,
    pub authoritative: &'a DecoderStateV1,
}

/// 下一块已经由真实 committed checkpoint 构造好的可替换状态。固定 execution owners
/// 不在本输入中，因而接口没有重建 runtime/windows/B4/shared-reduce/HC bridge 的能力。
pub struct S14StarfoldNextK4RebindInputs<P> {
    pub production: S14StarfoldK4ProductionResourceInputs<P>,
    pub input_token_ids: [u32; K4],
    pub initial_hidden: S14CausalBlockHiddenBinding,
    pub mode: S14StarfoldK4BlockMode,
}

/// 一次性下一 K4 launch gate。只有 host/device committed identity 在 rebind 与 execute
/// 两次观察中完全一致，才能进入纯 StarFold FullDepth43 production 执行。
pub struct S14StarfoldNextK4Launch<'a> {
    previous_commit: &'a S14StarfoldK4CommitReceipt,
    authoritative: &'a DecoderStateV1,
    committed_device: &'a WholeTokenDeviceState,
    checkpoint: WholeTokenDeviceCommittedCheckpointBinding<'a>,
    context: Arc<VulkanContext>,
    paged_arena: Arc<crate::s14_position0_paged_weight_arena::S14Position0PagedWeightArena>,
    base_position: u32,
    input_token_ids: [u32; K4],
    initial_hidden: S14CausalBlockHiddenBinding,
    mode: S14StarfoldK4BlockMode,
    expected_validated_blocks: u64,
    previous_base_position: u32,
    rebind_generation: u64,
}

impl fmt::Debug for S14StarfoldNextK4Launch<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14StarfoldNextK4Launch")
            .field(
                "previous_block_sequence",
                &self.previous_commit.block_sequence,
            )
            .field("base_position", &self.base_position)
            .field("mode", &self.mode)
            .field("checkpoint", &self.checkpoint)
            .field("context", &Arc::as_ptr(&self.context))
            .field("paged_arena", &Arc::as_ptr(&self.paged_arena))
            .field("initial_hidden", &self.initial_hidden)
            .field("rebind_generation", &self.rebind_generation)
            .finish_non_exhaustive()
    }
}

impl S14StarfoldNextK4Launch<'_> {
    pub fn base_position(&self) -> u32 {
        self.base_position
    }

    pub fn committed_epoch(&self) -> u64 {
        self.checkpoint.epoch()
    }

    pub fn hidden_generation(&self) -> u64 {
        self.initial_hidden.generation
    }

    pub fn mode(&self) -> S14StarfoldK4BlockMode {
        self.mode
    }

    /// 消费 launch gate 执行下一真实 K4 block。这里循环的是 FullDepth43 层深度，
    /// 不存在 serial-token、CPU、旧 union/grouped-MoE 或 Transformer fallback。
    pub fn execute<P>(
        self,
        resources: &mut S14StarfoldK4ProductionResources<P>,
    ) -> Result<S14StarfoldK4FullDepthReceipt>
    where
        P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
    {
        validate_receipt_and_host(self.previous_commit, self.authoritative)?;
        let observed = self
            .committed_device
            .committed_checkpoint_binding()
            .context("下一 K4 launch 重新借用 committed device checkpoint")?;
        if observed.buffer() != self.checkpoint.buffer()
            || observed.state_bytes() != self.checkpoint.state_bytes()
            || observed.epoch() != self.checkpoint.epoch()
            || observed.active_bank() != self.checkpoint.active_bank()
            || observed.state_bytes() != self.authoritative.native_arena.bytes().len() as u64
            || self.previous_commit.device_checkpoint_arena_bytes < observed.state_bytes()
            || observed.epoch() != self.previous_commit.committed_epoch
            || observed.active_bank() != self.previous_commit.committed_active_bank
            || !Arc::ptr_eq(resources.context(), &self.context)
            || !Arc::ptr_eq(resources.paged_arena(), &self.paged_arena)
        {
            bail!("下一 K4 launch 前 committed checkpoint/context/paged arena 已漂移");
        }
        {
            let adapter = resources.adapter_mut()?;
            if adapter.committed_rebinds() != self.rebind_generation
                || adapter.validated_blocks() != self.expected_validated_blocks
                || adapter.last_finished_base_position() != Some(self.previous_base_position)
                || self.rebind_generation != self.expected_validated_blocks
            {
                bail!("下一 K4 launch gate 已陈旧或 adapter session lineage 漂移");
            }
        }
        let receipt = match self.mode {
            S14StarfoldK4BlockMode::TeacherForcedPrefill => resources
                .adapter_mut()?
                .execute_teacher_forced_prefill_full_depth(
                    self.authoritative,
                    &self.input_token_ids,
                    self.initial_hidden,
                )?,
            S14StarfoldK4BlockMode::SpeculativeGeneration => resources.execute_k4_full_depth(
                self.base_position,
                &self.input_token_ids,
                self.initial_hidden,
                self.mode.materialized_source(),
            )?,
        };
        if receipt.base_position != self.base_position
            || receipt.block_size != K4
            || receipt.completed_layers != FULL_DEPTH_LAYERS.len()
            || receipt.layers.len() != FULL_DEPTH_LAYERS.len()
            || receipt.routes_by_position.len() != K4
            || receipt.serial_token_forward_calls != 0
            || !receipt.terminal_ready
            || receipt.token_committed
        {
            bail!("下一 K4 FullDepth43 receipt identity 漂移");
        }
        Ok(receipt)
    }
}

/// teacher-forced 原子提交后的强 anchor；不伪装为 generation terminal receipt。
pub struct S14StarfoldTeacherForcedCommittedK4Anchor<'a> {
    pub previous_commit: &'a S14StarfoldSealedTeacherForcedBlockCommitReceipt,
    pub committed_device: &'a WholeTokenDeviceState,
    pub authoritative: &'a DecoderStateV1,
}

pub struct S14StarfoldTeacherForcedNextK4Launch<'a> {
    previous_commit: &'a S14StarfoldSealedTeacherForcedBlockCommitReceipt,
    authoritative: &'a DecoderStateV1,
    committed_device: &'a WholeTokenDeviceState,
    checkpoint: WholeTokenDeviceCommittedCheckpointBinding<'a>,
    context: Arc<VulkanContext>,
    paged_arena: Arc<crate::s14_position0_paged_weight_arena::S14Position0PagedWeightArena>,
    base_position: u32,
    input_token_ids: [u32; K4],
    initial_hidden: S14CausalBlockHiddenBinding,
    mode: S14StarfoldK4BlockMode,
    expected_validated_blocks: u64,
    previous_base_position: u32,
    rebind_generation: u64,
}

impl S14StarfoldTeacherForcedNextK4Launch<'_> {
    pub fn base_position(&self) -> u32 {
        self.base_position
    }

    pub fn execute<P>(
        self,
        resources: &mut S14StarfoldK4ProductionResources<P>,
    ) -> Result<S14StarfoldK4FullDepthReceipt>
    where
        P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
    {
        validate_teacher_forced_anchor(
            self.previous_commit,
            self.authoritative,
            self.committed_device,
        )?;
        let observed = self.committed_device.committed_checkpoint_binding()?;
        if observed.buffer() != self.checkpoint.buffer()
            || observed.state_bytes() != self.checkpoint.state_bytes()
            || observed.epoch() != self.checkpoint.epoch()
            || observed.active_bank() != self.checkpoint.active_bank()
            || !Arc::ptr_eq(resources.context(), &self.context)
            || !Arc::ptr_eq(resources.paged_arena(), &self.paged_arena)
        {
            bail!("teacher-forced 下一块 launch checkpoint/context 已漂移");
        }
        let adapter = resources.adapter_mut()?;
        if adapter.validated_blocks() != self.expected_validated_blocks
            || adapter.committed_rebinds() != self.rebind_generation
            || adapter.last_finished_base_position() != Some(self.previous_base_position)
            || self.rebind_generation != self.expected_validated_blocks
        {
            bail!("teacher-forced 下一块 adapter lineage 漂移");
        }
        match self.mode {
            S14StarfoldK4BlockMode::TeacherForcedPrefill => adapter
                .execute_teacher_forced_prefill_full_depth(
                    self.authoritative,
                    &self.input_token_ids,
                    self.initial_hidden,
                ),
            S14StarfoldK4BlockMode::SpeculativeGeneration => resources.execute_k4_full_depth(
                self.base_position,
                &self.input_token_ids,
                self.initial_hidden,
                MaterializedTokenSource::SpeculativeDraft,
            ),
        }
    }
}

/// prefill→prefill 与最后 prefill→首个 generation 共用的独立强重绑入口。
pub fn rebind_next_teacher_forced_prefill_block<'a, P>(
    resources: &mut S14StarfoldK4ProductionResources<P>,
    anchor: S14StarfoldTeacherForcedCommittedK4Anchor<'a>,
    inputs: S14StarfoldNextK4RebindInputs<P>,
) -> Result<S14StarfoldTeacherForcedNextK4Launch<'a>>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    let checkpoint = validate_teacher_forced_anchor(
        anchor.previous_commit,
        anchor.authoritative,
        anchor.committed_device,
    )?;
    let previous = &anchor.previous_commit.commit;
    let next_base_position = previous.committed_position;
    let expected_hidden_generation = anchor
        .previous_commit
        .sealed_full_depth
        .final_hidden
        .generation
        .checked_add(1)
        .context("teacher-forced 下一块 hidden generation overflow")?;
    validate_next_block_inputs(
        resources,
        &inputs,
        next_base_position,
        &checkpoint,
        expected_hidden_generation,
        anchor.authoritative,
    )?;
    let expected_validated_blocks = u64::try_from(previous.block_index)
        .context("prefill block index 超出 u64")?
        .checked_add(1)
        .context("prefill validated block count overflow")?;
    {
        let adapter = resources.adapter_mut()?;
        if adapter.validated_blocks() != expected_validated_blocks
            || adapter.committed_rebinds() + 1 != expected_validated_blocks
            || adapter.last_finished_base_position() != Some(previous.base_position)
        {
            bail!("teacher-forced receipt 与 adapter validated/rebind lineage 未闭合");
        }
    }
    let S14StarfoldNextK4RebindInputs {
        production,
        input_token_ids,
        initial_hidden,
        mode,
    } = inputs;
    if production.position0_generation_provenance.is_some() {
        bail!("teacher-forced 后续 rebind 禁止携带 position0 generation provenance");
    }
    let (_, mut provider) = production.hc_qkv_provider.into_parts();
    let (_, hidden_banks) = production.hidden_banks.into_parts();
    let mut prefix = provider
        .take_prefix_state_producer()
        .map_err(anyhow::Error::msg)
        .context("消费 teacher-forced 下一块 prefix producer")?;
    if !Arc::ptr_eq(resources.context(), prefix.context())
        || prefix.arena().base_position() != next_base_position
        || prefix.arena().layout().block_size != K4
        || prefix.arena().layout().checkpoint_state_bytes != checkpoint.state_bytes()
    {
        let cleanup = prefix.destroy();
        bail!("teacher-forced 下一块 prefix identity 漂移; cleanup={cleanup:?}");
    }
    let rebind_generation = match resources.adapter_mut() {
        Ok(adapter) => {
            adapter.rebind_committed_block_state(
                next_base_position,
                provider,
                hidden_banks,
                prefix,
            )?;
            adapter.committed_rebinds()
        }
        Err(error) => {
            let cleanup = prefix.destroy();
            return Err(anyhow!(
                "{error:#}; teacher-forced 下一块 prefix cleanup={cleanup:?}"
            ));
        }
    };
    if rebind_generation != expected_validated_blocks {
        bail!("teacher-forced rebind generation 未按 validated_blocks 单调推进");
    }
    Ok(S14StarfoldTeacherForcedNextK4Launch {
        previous_commit: anchor.previous_commit,
        authoritative: anchor.authoritative,
        committed_device: anchor.committed_device,
        checkpoint,
        context: Arc::clone(resources.context()),
        paged_arena: Arc::clone(resources.paged_arena()),
        base_position: next_base_position,
        input_token_ids,
        initial_hidden,
        mode,
        expected_validated_blocks,
        previous_base_position: previous.base_position,
        rebind_generation,
    })
}

/// 把上一块原子 commit 后的真实 state 原地重绑到常驻 StarFold stage，并签发下一块
/// 一次性 launch gate。next base 只取 receipt 的实际 committed position，上一块即使只
/// 提交 1..3 个 token，也不会按物理 K4 错误推进四个位置。
pub fn rebind_next_committed_k4_block<'a, P>(
    resources: &mut S14StarfoldK4ProductionResources<P>,
    anchor: S14StarfoldCommittedK4Anchor<'a>,
    inputs: S14StarfoldNextK4RebindInputs<P>,
) -> Result<S14StarfoldNextK4Launch<'a>>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    let checkpoint = validate_committed_anchor(&anchor)?;
    let previous = anchor.previous_commit;
    let next_base_position = previous.committed_position;
    let expected_hidden_generation = previous
        .sealed_full_depth
        .final_hidden
        .generation
        .checked_add(1)
        .context("下一 K4 initial hidden generation overflow")?;
    validate_next_block_inputs(
        resources,
        &inputs,
        next_base_position,
        &checkpoint,
        expected_hidden_generation,
        anchor.authoritative,
    )?;

    let expected_prior_rebinds = previous
        .block_sequence
        .checked_sub(1)
        .context("上一 K4 block sequence 必须从1开始")?;
    {
        let adapter = resources.adapter_mut()?;
        if adapter.validated_blocks() != previous.block_sequence
            || adapter.committed_rebinds() != expected_prior_rebinds
            || adapter.last_finished_base_position() != Some(previous.base_position)
        {
            bail!("下一 K4 rebind 要求上一 receipt、finish/drain 与 adapter lineage 单调闭合");
        }
    }

    let S14StarfoldNextK4RebindInputs {
        production,
        input_token_ids,
        initial_hidden,
        mode,
    } = inputs;
    if production.position0_generation_provenance.is_some() {
        bail!("后续 committed rebind 禁止复用 position0 generation provenance");
    }
    let (_, mut provider) = production.hc_qkv_provider.into_parts();
    let (_, hidden_banks) = production.hidden_banks.into_parts();
    let mut prefix = provider
        .take_prefix_state_producer()
        .map_err(anyhow::Error::msg)
        .context("消费下一 K4 committed prefix producer")?;
    let prefix_valid = Arc::ptr_eq(resources.context(), prefix.context())
        && prefix.arena().base_position() == next_base_position
        && prefix.arena().layout().block_size == K4
        && prefix.arena().layout().checkpoint_state_bytes == checkpoint.state_bytes();
    if !prefix_valid {
        let cleanup = prefix.destroy();
        bail!("下一 K4 prefix context/base/K/checkpoint ABI 漂移; cleanup={cleanup:?}");
    }

    let rebind_generation = match resources.adapter_mut() {
        Ok(adapter) => {
            adapter.rebind_committed_block_state(
                next_base_position,
                provider,
                hidden_banks,
                prefix,
            )?;
            adapter.committed_rebinds()
        }
        Err(error) => {
            let cleanup = prefix.destroy();
            return Err(anyhow!("{error:#}; 下一 K4 prefix cleanup={cleanup:?}"));
        }
    };
    if rebind_generation != previous.block_sequence {
        bail!("下一 K4 rebind generation 未与上一 commit block sequence 单调闭合");
    }
    Ok(S14StarfoldNextK4Launch {
        previous_commit: previous,
        authoritative: anchor.authoritative,
        committed_device: anchor.committed_device,
        checkpoint,
        context: Arc::clone(resources.context()),
        paged_arena: Arc::clone(resources.paged_arena()),
        base_position: next_base_position,
        input_token_ids,
        initial_hidden,
        mode,
        expected_validated_blocks: previous.block_sequence,
        previous_base_position: previous.base_position,
        rebind_generation,
    })
}

/// 兼容“第一块之后启动第二块”的便利包装；第二块 base 同样来自实际 committed position，
/// 不再假定第一块从 position1 开始或一定提交四个 token。
pub fn rebind_s14_starfold_second_k4_block<'a, P>(
    resources: &mut S14StarfoldK4ProductionResources<P>,
    anchor: S14StarfoldCommittedK4Anchor<'a>,
    inputs: S14StarfoldNextK4RebindInputs<P>,
) -> Result<S14StarfoldNextK4Launch<'a>>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    if anchor.previous_commit.block_sequence != 1 {
        bail!("第二 K4 便利包装只接受 block_sequence=1 的上一 commit receipt");
    }
    rebind_next_committed_k4_block(resources, anchor, inputs)
}

/// 旧的第二块类型名保留为公开重导出；核心语义与字段已经泛化为任意下一块。
pub use S14StarfoldCommittedK4Anchor as S14StarfoldFirstK4AtomicCommit;
pub use S14StarfoldNextK4Launch as S14StarfoldSecondK4Launch;
pub use S14StarfoldNextK4RebindInputs as S14StarfoldSecondK4RebindInputs;

fn validate_committed_anchor<'a>(
    anchor: &S14StarfoldCommittedK4Anchor<'a>,
) -> Result<WholeTokenDeviceCommittedCheckpointBinding<'a>> {
    validate_receipt_and_host(anchor.previous_commit, anchor.authoritative)?;
    let owner: &'a WholeTokenDeviceState = anchor.committed_device;
    let checkpoint = owner
        .committed_checkpoint_binding()
        .context("借用上一 K4 原子提交后的 committed device checkpoint")?;
    let previous = anchor.previous_commit;
    if checkpoint.buffer() == vk::Buffer::null()
        || checkpoint.state_bytes() == 0
        || checkpoint.state_bytes() != anchor.authoritative.native_arena.bytes().len() as u64
        || previous.device_checkpoint_arena_bytes < checkpoint.state_bytes()
        || checkpoint.epoch() != previous.committed_epoch
        || checkpoint.active_bank() != previous.committed_active_bank
        || checkpoint.epoch() != owner.epoch()
        || checkpoint.active_bank() != owner.active_bank()
        || checkpoint.state_bytes() != owner.state_bytes()
    {
        bail!("上一 K4 receipt/device/host committed checkpoint identity 未闭合");
    }
    Ok(checkpoint)
}

fn validate_teacher_forced_anchor<'a>(
    previous: &S14StarfoldSealedTeacherForcedBlockCommitReceipt,
    authoritative: &DecoderStateV1,
    device: &'a WholeTokenDeviceState,
) -> Result<WholeTokenDeviceCommittedCheckpointBinding<'a>> {
    authoritative
        .validate()
        .map_err(|error| anyhow!("teacher-forced authoritative checkpoint 非法: {error}"))?;
    let commit = &previous.commit;
    let logical_u32 =
        u32::try_from(commit.logical_lanes).context("teacher-forced logical lanes 超出 u32")?;
    let logical_u64 =
        u64::try_from(commit.logical_lanes).context("teacher-forced logical lanes 超出 u64")?;
    if !(1..=K4).contains(&commit.logical_lanes)
        || commit.physical_lanes != K4
        || commit.committed_position != commit.base_position + logical_u32
        || commit.committed_epoch != u64::from(commit.base_position) + logical_u64
        || commit.committed_active_bank != commit.committed_position as usize & 1
        || commit.device.position != commit.committed_position
        || commit.device.epoch != commit.committed_epoch
        || commit.device.active_bank != commit.committed_active_bank
        || commit.device.accepted_tokens != commit.logical_lanes
        || !commit.host_device_checkpoint_bytes_verified
        || commit.terminal_head_calls != 0
        || commit.serial_token_forward_calls != 0
        || previous.sealed_full_depth.base_position != commit.base_position
        || previous.sealed_full_depth.block_size != K4
        || previous.sealed_full_depth.source != MaterializedTokenSource::ForcedPrefill
        || previous.sealed_full_depth.serial_token_forward_calls != 0
        || authoritative.position != commit.committed_position
        || authoritative.commit_epoch != commit.committed_epoch
        || usize::from(authoritative.active_fixed_bank) != commit.committed_active_bank
        || authoritative.input_token_id != commit.committed_input_token_id
    {
        bail!("teacher-forced commit/full-depth/authoritative identity 未闭合");
    }
    let checkpoint = device.committed_checkpoint_binding()?;
    if checkpoint.epoch() != commit.committed_epoch
        || checkpoint.active_bank() != commit.committed_active_bank
        || checkpoint.state_bytes() != authoritative.native_arena.bytes().len() as u64
    {
        bail!("teacher-forced committed device identity 漂移");
    }
    Ok(checkpoint)
}

fn validate_receipt_and_host(
    previous: &S14StarfoldK4CommitReceipt,
    authoritative: &DecoderStateV1,
) -> Result<()> {
    authoritative
        .validate()
        .map_err(|error| anyhow!("上一 K4 authoritative checkpoint 非法: {error}"))?;
    validate_commit_receipt(previous)?;
    if authoritative.position != previous.committed_position
        || authoritative.native.position != previous.committed_position
        || authoritative.commit_epoch != previous.committed_epoch
        || usize::from(authoritative.active_fixed_bank) != previous.committed_active_bank
        || authoritative.input_token_id != previous.committed_input_token_id
        || previous.device_checkpoint_arena_bytes < authoritative.native_arena.bytes().len() as u64
    {
        bail!("上一 K4 commit receipt 与 authoritative host checkpoint identity 漂移");
    }
    Ok(())
}

fn validate_commit_receipt(previous: &S14StarfoldK4CommitReceipt) -> Result<()> {
    let full_depth = &previous.sealed_full_depth;
    let committed_tokens = u32::try_from(previous.committed_tokens)
        .context("上一 K4 committed token count 超出 u32")?;
    let expected_position = previous
        .base_position
        .checked_add(committed_tokens)
        .context("上一 K4 committed position overflow")?;
    let expected_epoch = previous
        .base_commit_epoch
        .checked_add(u64::from(committed_tokens))
        .context("上一 K4 committed epoch overflow")?;
    let expected_committed_tokens = previous
        .checkpoint_index
        .checked_add(1)
        .context("上一 K4 checkpoint index overflow")?;
    let selected = previous
        .candidate_checkpoints
        .get(previous.checkpoint_index)
        .context("上一 K4 commit receipt 缺少 selected checkpoint identity")?;
    let committed_input_token_id = previous
        .decision
        .committed_token_ids
        .last()
        .copied()
        .context("上一 K4 commit receipt 缺少 committed token ledger")?;
    let final_hidden_bytes = K4 as u64 * S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64 * BF16_BYTES;

    if previous.schema_version != S14_STARFOLD_TERMINAL_CHAIN_SCHEMA_VERSION
        || previous.block_sequence == 0
        || !(1..=K4).contains(&previous.commit_limit)
        || !(1..=K4).contains(&previous.committed_tokens)
        || previous.committed_tokens > previous.commit_limit
        || expected_committed_tokens != previous.committed_tokens
        || previous.committed_position != expected_position
        || previous.committed_epoch != expected_epoch
        || previous.committed_active_bank >= 2
        || previous.committed_input_token_id >= VOCAB_SIZE
        || previous.committed_input_token_id != committed_input_token_id
        || previous.candidate_checkpoints.len() != K4
        || previous
            .draft_token_ids
            .iter()
            .any(|&token| token >= VOCAB_SIZE)
        || previous
            .predicted_token_ids
            .iter()
            .any(|&token| token >= VOCAB_SIZE)
        || previous.decision.committed_token_ids.len() != previous.committed_tokens
        || previous
            .decision
            .mismatch_index
            .is_some_and(|index| index != previous.checkpoint_index)
        || !previous.host_device_checkpoint_bytes_verified
        || previous.device_ready_timeline_value == 0
        || previous.device_checkpoint_arena_bytes == 0
        || previous.base_checkpoint_sha256 == S14StarwaveSha256::ZERO
        || previous.committed_checkpoint_sha256 == S14StarwaveSha256::ZERO
        || previous.physical_evidence_sha256 == S14StarwaveSha256::ZERO
        || previous.commit_chain_sha256 == S14StarwaveSha256::ZERO
        || previous.terminal_head_submit_calls != 1
        || previous.checkpoint_export_calls != 1
        || previous.legacy_union_calls != 0
        || previous.legacy_grouped_moe_calls != 0
        || previous.serial_token_forward_calls != 0
        || previous.cpu_fallback_calls != 0
        || previous.device_commit.epoch != previous.committed_epoch
        || previous.device_commit.active_bank != previous.committed_active_bank
        || previous.device_commit.position != previous.committed_position
        || previous.device_commit.accepted_tokens != previous.committed_tokens
        || previous.device_commit.checkpoint_index != previous.checkpoint_index
        || !previous.device_commit.host_device_bytes_verified
        || selected.checkpoint_index != previous.checkpoint_index
        || selected.position != previous.committed_position
        || selected.commit_epoch != previous.committed_epoch
        || usize::from(selected.active_fixed_bank) != previous.committed_active_bank
        || selected.predicted_token_id != previous.predicted_token_ids[previous.checkpoint_index]
        || full_depth.base_position != previous.base_position
        || full_depth.block_size != K4
        || full_depth.completed_layers != FULL_DEPTH_LAYERS.len()
        || full_depth.layers.len() != FULL_DEPTH_LAYERS.len()
        || full_depth.routes_by_position.len() != K4
        || full_depth.checkpoint_seal.base_position != previous.base_position
        || full_depth.checkpoint_seal.block_size != K4
        || full_depth.checkpoint_seal.completed_layers != FULL_DEPTH_LAYERS.len()
        || full_depth.checkpoint_seal.checkpoint_count != K4
        || full_depth.checkpoint_seal.prefix_program_seal_calls != 1
        || full_depth.checkpoint_seal.checkpoint_commit_calls != 0
        || full_depth.checkpoint_seal.serial_token_forward_calls != 0
        || full_depth.final_hidden.buffer == vk::Buffer::null()
        || full_depth.final_hidden.offset % 4 != 0
        || full_depth.final_hidden.bytes != final_hidden_bytes
        || full_depth.final_hidden.block_size != K4
        || full_depth.serial_token_forward_calls != 0
        || !full_depth.terminal_ready
        || full_depth.token_committed
    {
        bail!("上一 K4 terminal/full-depth/device commit receipt 强身份不完整");
    }
    if previous
        .candidate_checkpoints
        .iter()
        .enumerate()
        .any(|(index, checkpoint)| {
            checkpoint.checkpoint_index != index
                || checkpoint.predicted_token_id != previous.predicted_token_ids[index]
                || checkpoint.predicted_token_id >= VOCAB_SIZE
                || checkpoint.checkpoint_sha256 == S14StarwaveSha256::ZERO
        })
    {
        bail!("上一 K4 candidate checkpoint ledger identity 漂移");
    }
    if full_depth.layers.iter().any(|layer| {
        layer.base_position != previous.base_position
            || layer.checkpoint.base_position != previous.base_position
            || layer.expert.base_position != u64::from(previous.base_position)
            || layer.hidden_commit.base_position != previous.base_position
            || layer.checkpoint.serial_token_forward_calls != 0
            || layer.expert.serial_token_forward_calls != 0
            || layer.hidden_commit.serial_token_forward_calls != 0
    }) || full_depth
        .layers
        .last()
        .is_none_or(|layer| layer.hidden_commit.next_hidden != full_depth.final_hidden)
        || full_depth.routes_by_position.iter().any(|routes| {
            routes.len() != FULL_DEPTH_LAYERS.len()
                || routes
                    .iter()
                    .zip(FULL_DEPTH_LAYERS)
                    .any(|(route, layer)| route.layer != layer)
        })
    {
        bail!("上一 K4 FullDepth43 layer/route/hidden chain identity 漂移");
    }
    Ok(())
}

fn validate_next_block_inputs<P>(
    resources: &S14StarfoldK4ProductionResources<P>,
    inputs: &S14StarfoldNextK4RebindInputs<P>,
    next_base_position: u32,
    checkpoint: &WholeTokenDeviceCommittedCheckpointBinding<'_>,
    expected_hidden_generation: u64,
    authoritative: &DecoderStateV1,
) -> Result<()>
where
    P: S14CausalBlockProductionHcQkvResourceProvider + 'static,
{
    validate_context_owner(
        resources.context(),
        inputs.production.hc_qkv_provider.context(),
        "下一 K4 HC/QKV provider",
    )?;
    validate_context_owner(
        resources.context(),
        inputs.production.hidden_banks.context(),
        "下一 K4 hidden banks",
    )?;
    let source = inputs.mode.materialized_source();
    let source_matches_mode = matches!(
        (inputs.mode, source),
        (
            S14StarfoldK4BlockMode::TeacherForcedPrefill,
            MaterializedTokenSource::ForcedPrefill
        ) | (
            S14StarfoldK4BlockMode::SpeculativeGeneration,
            MaterializedTokenSource::SpeculativeDraft
        )
    );
    if !source_matches_mode
        || inputs.input_token_ids[0] != authoritative.input_token_id
        || inputs
            .input_token_ids
            .iter()
            .any(|&token| token >= VOCAB_SIZE)
        || !Arc::ptr_eq(
            inputs
                .production
                .hc_qkv_provider
                .value()
                .paged_weight_arena(),
            resources.paged_arena(),
        )
    {
        bail!("下一 K4 mode/source/input/committed token/paged arena identity 漂移");
    }
    inputs
        .production
        .hc_qkv_provider
        .value()
        .validate_production_bundle(K4)
        .map_err(anyhow::Error::msg)
        .context("下一 K4 provider readiness 拒绝")?;
    inputs
        .production
        .hc_qkv_provider
        .value()
        .validate_committed_block_rebind(
            next_base_position,
            &inputs.input_token_ids,
            checkpoint.state_bytes(),
        )
        .map_err(anyhow::Error::msg)
        .context("下一 K4 provider committed-state identity 拒绝")?;
    validate_hidden_banks(inputs.production.hidden_banks.value())?;
    validate_initial_hidden_owner(
        inputs.production.hidden_banks.value(),
        inputs.initial_hidden,
        expected_hidden_generation,
    )
}

fn validate_initial_hidden_owner(
    banks: &[S14CausalBlockHiddenBank; 2],
    initial_hidden: S14CausalBlockHiddenBinding,
    expected_generation: u64,
) -> Result<()> {
    let expected_bytes = K4 as u64 * S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64 * BF16_BYTES;
    if initial_hidden.buffer == vk::Buffer::null()
        || initial_hidden.offset % 4 != 0
        || initial_hidden.bytes != expected_bytes
        || initial_hidden.block_size != K4
        || initial_hidden.generation != expected_generation
    {
        bail!("下一 K4 initial hidden ABI/generation 漂移");
    }
    let owners = banks
        .iter()
        .map(|bank| bank.binding(K4, expected_generation))
        .collect::<Result<Vec<_>>>()?;
    if owners
        .iter()
        .filter(|&&binding| binding == initial_hidden)
        .count()
        != 1
    {
        bail!("下一 K4 initial hidden 不唯一属于新 A/B owner");
    }
    Ok(())
}
