//! 纯 S14 StarFold K4/K8 teacher-forced prompt prefill 计划与提交门。
//!
//! `DecoderStateV1::position` 始终表示下一个待执行位置。给定 N 个 prompt token，本模块只
//! 执行 N-1 个已观测输入转移；计划器默认优先发出物理 K8，最后不超过4个转移时发出 K4。
//! 8-GiB 设备可通过显式环境变量把物理上限收紧为 K4，避免 K8 arena 退让到系统内存。
//! 每块只提交逻辑前缀 r，filler lane 永不进入权威 ledger/checkpoint。这里不执行
//! generation terminal/head，也不把 prompt 的 observed next input 描述成模型预测。

use crate::{
    s14_causal_block_hc_qkv_recorder::S14CausalBlockStarfoldPrefillPrefixProduct,
    s14_causal_block_prefix_arena::S14CausalBlockPrefixCheckpointArena,
    s14_starfold_k4_adapter::S14StarfoldK4FullDepthReceipt,
    s14_starfold_vulkan_windows::S14StarfoldTimelinePoint,
    s14_whole_token_device::{WholeTokenDeviceBlockCommitReceipt, WholeTokenDeviceState},
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    DecoderStateV1, GraphProfile, MaterializedTokenSource, NativeStateArena, TokenRecord,
    FULL_DEPTH_LAYERS, VOCAB_SIZE,
};
use std::sync::Arc;

/// 现有 K4 production adapter 的冻结 lane 数。保留旧符号供下游迁移期间显式验收，
/// 新计划代码必须读取 block 自身的 [`S14StarfoldPrefillPhysicalK`]，不得再用它推断任意块。
pub const S14_STARFOLD_PREFILL_PHYSICAL_K: usize = 4;
pub const S14_STARFOLD_PREFILL_PHYSICAL_K4: usize = 4;
pub const S14_STARFOLD_PREFILL_PHYSICAL_K8: usize = 8;
pub const S14_STARFOLD_PREFILL_MAX_K_ENV: &str = "POLARIS_S14_PREFILL_MAX_K";

/// teacher-forced prefill 的物理块宽度。只允许冻结的 K4/K8，禁止用裸 `usize` 把逻辑
/// lane 数冒充为物理宽度。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum S14StarfoldPrefillPhysicalK {
    K4 = S14_STARFOLD_PREFILL_PHYSICAL_K4 as u8,
    K8 = S14_STARFOLD_PREFILL_PHYSICAL_K8 as u8,
}

impl S14StarfoldPrefillPhysicalK {
    pub const fn lanes(self) -> usize {
        self as usize
    }

    /// 规划策略：能覆盖超过4个逻辑转移时优先 K8；最后1..=4个转移使用 K4。
    pub const fn preferred_for_remaining(remaining_logical_lanes: usize) -> Option<Self> {
        match remaining_logical_lanes {
            0 => None,
            1..=S14_STARFOLD_PREFILL_PHYSICAL_K4 => Some(Self::K4),
            _ => Some(Self::K8),
        }
    }
}

/// 不歧义的物理输入块。K8 不能通过隐式截断进入仅接收 `[u32; 4]` 的旧执行器；下游必须
/// 显式匹配 variant，或改用 [`Self::as_slice`] 接入真正的 K-block runtime。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldPrefillInputBlock {
    K4([u32; S14_STARFOLD_PREFILL_PHYSICAL_K4]),
    K8([u32; S14_STARFOLD_PREFILL_PHYSICAL_K8]),
}

impl S14StarfoldPrefillInputBlock {
    pub const fn filled(physical_k: S14StarfoldPrefillPhysicalK, filler_token_id: u32) -> Self {
        match physical_k {
            S14StarfoldPrefillPhysicalK::K4 => {
                Self::K4([filler_token_id; S14_STARFOLD_PREFILL_PHYSICAL_K4])
            }
            S14StarfoldPrefillPhysicalK::K8 => {
                Self::K8([filler_token_id; S14_STARFOLD_PREFILL_PHYSICAL_K8])
            }
        }
    }

    pub const fn physical_k(&self) -> S14StarfoldPrefillPhysicalK {
        match self {
            Self::K4(_) => S14StarfoldPrefillPhysicalK::K4,
            Self::K8(_) => S14StarfoldPrefillPhysicalK::K8,
        }
    }

    pub const fn as_slice(&self) -> &[u32] {
        match self {
            Self::K4(tokens) => tokens,
            Self::K8(tokens) => tokens,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u32] {
        match self {
            Self::K4(tokens) => tokens,
            Self::K8(tokens) => tokens,
        }
    }

    /// 旧 K4 adapter 的显式窄入口。K8 返回 `None`，调用方不得取前四 lane 伪装成功。
    pub const fn as_k4(&self) -> Option<&[u32; S14_STARFOLD_PREFILL_PHYSICAL_K4]> {
        match self {
            Self::K4(tokens) => Some(tokens),
            Self::K8(_) => None,
        }
    }

    pub const fn as_k8(&self) -> Option<&[u32; S14_STARFOLD_PREFILL_PHYSICAL_K8]> {
        match self {
            Self::K4(_) => None,
            Self::K8(tokens) => Some(tokens),
        }
    }
}

impl AsRef<[u32]> for S14StarfoldPrefillInputBlock {
    fn as_ref(&self) -> &[u32] {
        self.as_slice()
    }
}

/// 只能由同一个 production adapter 在 FullDepth seal、prefix arena 导出和
/// finish/drain 连续完成后构造；session 无法把不相关的 arena 与 receipt 手工配对。
pub struct S14StarfoldSealedTeacherForcedPrefillProduct {
    full_depth: S14StarfoldK4FullDepthReceipt,
    block: S14StarfoldTeacherForcedBlockPlan,
    prefix_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
    /// 同源 recorder 的历史 completion identity。producer 在 product 构造前已 drain，
    /// after-drain commit 不会再次等待该 timeline。
    producer_completion: S14StarfoldTimelinePoint,
}

#[derive(Clone, Debug)]
pub struct S14StarfoldSealedTeacherForcedBlockCommitReceipt {
    pub commit: S14StarfoldTeacherForcedBlockCommitReceipt,
    pub sealed_full_depth: S14StarfoldK4FullDepthReceipt,
}

impl S14StarfoldSealedTeacherForcedPrefillProduct {
    pub(crate) fn from_finished_adapter(
        full_depth: S14StarfoldK4FullDepthReceipt,
        block: S14StarfoldTeacherForcedBlockPlan,
        prefix: S14CausalBlockStarfoldPrefillPrefixProduct,
    ) -> Result<Self> {
        block.validate()?;
        let physical_k = block.physical_k();
        let physical_inputs = block.physical_input_token_ids();
        let S14CausalBlockStarfoldPrefillPrefixProduct {
            context: prefix_context,
            prefix_checkpoint_arena: prefix_arena,
            producer_ready: producer_completion,
            source: prefix_source,
        } = prefix;
        if full_depth.physical_input_token_ids.len() != physical_k {
            bail!(
                "StarFold sealed prefill receipt 仍是物理 K{}，无法消费计划的 K{}；必须接入对应 K-block adapter",
                full_depth.physical_input_token_ids.len(),
                physical_k
            );
        }
        if full_depth.source != MaterializedTokenSource::ForcedPrefill
            || prefix_source != MaterializedTokenSource::ForcedPrefill
            || full_depth.base_position != block.base_position
            || full_depth.base_position != prefix_arena.base_position()
            || full_depth.block_size != physical_k
            || full_depth.physical_input_token_ids.as_slice() != physical_inputs
            || full_depth.completed_layers != FULL_DEPTH_LAYERS.len()
            || full_depth.layers.len() != FULL_DEPTH_LAYERS.len()
            || full_depth.checkpoint_seal.checkpoint_count != physical_k
            || full_depth.serial_token_forward_calls != 0
            || !full_depth.terminal_ready
            || full_depth.token_committed
            || prefix_arena.layout().block_size != physical_k
            || !Arc::ptr_eq(&prefix_context, prefix_arena.context())
            || producer_completion.semaphore == vk::Semaphore::null()
            || producer_completion.generation == 0
            || producer_completion.value == 0
        {
            bail!("StarFold sealed prefill product FullDepth/prefix/recorder identity 漂移");
        }
        prefix_arena.validate_host_readback_ready()?;
        Ok(Self {
            full_depth,
            block,
            prefix_arena,
            producer_completion,
        })
    }

    pub fn full_depth(&self) -> &S14StarfoldK4FullDepthReceipt {
        &self.full_depth
    }

    pub fn commit(
        self,
        context: &VulkanContext,
        authoritative: &mut DecoderStateV1,
        device_state: &mut WholeTokenDeviceState,
        readback: &mut S14StarfoldPrefillReadbackOwner,
    ) -> Result<S14StarfoldSealedTeacherForcedBlockCommitReceipt> {
        let commit = commit_starfold_teacher_forced_prefill_block_inner(
            context,
            authoritative,
            device_state,
            readback,
            &self.block,
            &self.prefix_arena,
            self.producer_completion,
        )?;
        Ok(S14StarfoldSealedTeacherForcedBlockCommitReceipt {
            commit,
            sealed_full_depth: self.full_depth,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldTeacherForcedBlockPlan {
    pub block_index: usize,
    pub base_position: u32,
    pub physical_input_token_ids: S14StarfoldPrefillInputBlock,
    /// 只含逻辑 lane `0..r` 的 observed next input；长度就是唯一可提交前缀。
    pub observed_next_input_token_ids: Vec<u32>,
    pub logical_lanes: usize,
    pub filler_token_id: u32,
    pub source: MaterializedTokenSource,
}

impl S14StarfoldTeacherForcedBlockPlan {
    pub const fn physical_k_contract(&self) -> S14StarfoldPrefillPhysicalK {
        self.physical_input_token_ids.physical_k()
    }

    pub const fn physical_k(&self) -> usize {
        self.physical_k_contract().lanes()
    }

    pub const fn physical_input_token_ids(&self) -> &[u32] {
        self.physical_input_token_ids.as_slice()
    }

    pub fn checkpoint_index(&self) -> usize {
        self.logical_lanes - 1
    }

    pub fn committed_position(&self) -> u32 {
        self.base_position + self.logical_lanes as u32
    }

    pub fn committed_input_token_id(&self) -> u32 {
        self.observed_next_input_token_ids[self.logical_lanes - 1]
    }

    pub fn validate(&self) -> Result<()> {
        let physical_k = self.physical_k();
        let physical_input_token_ids = self.physical_input_token_ids();
        if self.source != MaterializedTokenSource::ForcedPrefill
            || !(1..=physical_k).contains(&self.logical_lanes)
            || self.observed_next_input_token_ids.len() != self.logical_lanes
            || self.filler_token_id >= VOCAB_SIZE
            || physical_input_token_ids
                .iter()
                .chain(&self.observed_next_input_token_ids)
                .any(|&token| token >= VOCAB_SIZE)
            || physical_input_token_ids[self.logical_lanes..]
                .iter()
                .any(|&token| token != self.filler_token_id)
            || self.observed_next_input_token_ids.last() != Some(&self.filler_token_id)
        {
            bail!("StarFold teacher-forced block 的 source/K/r/token/filler 合同漂移");
        }
        self.base_position
            .checked_add(self.logical_lanes as u32)
            .context("StarFold teacher-forced block position overflow")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldPromptPrefillPlan {
    pub prompt_len: usize,
    pub first_input_token_id: u32,
    pub final_position: u32,
    pub final_input_token_id: u32,
    pub blocks: Vec<S14StarfoldTeacherForcedBlockPlan>,
}

impl S14StarfoldPromptPrefillPlan {
    pub fn build(prompt_token_ids: &[u32]) -> Result<Self> {
        let max_physical_k = prefill_max_physical_k_from_env()?;
        Self::build_with_max_physical_k(prompt_token_ids, max_physical_k)
    }

    /// Deterministic planner entry used by runtime policy and focused fixtures.
    /// The logical prompt/checkpoint contract is identical for K4 and K8; only
    /// the physical batch width changes.
    pub fn build_with_max_physical_k(
        prompt_token_ids: &[u32],
        max_physical_k: S14StarfoldPrefillPhysicalK,
    ) -> Result<Self> {
        let &first_input_token_id = prompt_token_ids
            .first()
            .context("StarFold prompt prefill 要求至少一个 token")?;
        if prompt_token_ids.iter().any(|&token| token >= VOCAB_SIZE) {
            bail!("StarFold prompt prefill token 越出冻结 vocab");
        }
        let transition_count = prompt_token_ids.len() - 1;
        let final_position = u32::try_from(transition_count)
            .context("StarFold prompt prefill 长度超出 u32 position")?;
        let max_lanes = max_physical_k.lanes();
        let block_capacity = transition_count
            .checked_add(max_lanes - 1)
            .context("StarFold prompt block count overflow")?
            / max_lanes;
        let mut blocks = Vec::with_capacity(block_capacity);
        let mut base = 0usize;
        while base < transition_count {
            let remaining = transition_count - base;
            let physical_k = match max_physical_k {
                S14StarfoldPrefillPhysicalK::K4 => S14StarfoldPrefillPhysicalK::K4,
                S14StarfoldPrefillPhysicalK::K8 => {
                    S14StarfoldPrefillPhysicalK::preferred_for_remaining(remaining)
                        .expect("loop guarantees non-empty remainder")
                }
            };
            let logical_lanes = remaining.min(physical_k.lanes());
            let filler_token_id = prompt_token_ids[base + logical_lanes];
            let mut physical_input_token_ids =
                S14StarfoldPrefillInputBlock::filled(physical_k, filler_token_id);
            physical_input_token_ids.as_mut_slice()[..logical_lanes]
                .copy_from_slice(&prompt_token_ids[base..base + logical_lanes]);
            let block = S14StarfoldTeacherForcedBlockPlan {
                block_index: blocks.len(),
                base_position: u32::try_from(base)
                    .context("StarFold prompt block base position 超出 u32")?,
                physical_input_token_ids,
                observed_next_input_token_ids: prompt_token_ids[base + 1..base + logical_lanes + 1]
                    .to_vec(),
                logical_lanes,
                filler_token_id,
                source: MaterializedTokenSource::ForcedPrefill,
            };
            block.validate()?;
            blocks.push(block);
            base += logical_lanes;
        }
        let plan = Self {
            prompt_len: prompt_token_ids.len(),
            first_input_token_id,
            final_position,
            final_input_token_id: *prompt_token_ids.last().expect("non-empty checked above"),
            blocks,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// 验证同一物理上限下的规划、跨块 observed-next 连续性以及最终 checkpoint 身份。
    pub fn validate(&self) -> Result<()> {
        if self.prompt_len == 0
            || self.first_input_token_id >= VOCAB_SIZE
            || self.final_input_token_id >= VOCAB_SIZE
            || usize::try_from(self.final_position).ok() != self.prompt_len.checked_sub(1)
        {
            bail!("StarFold prompt prefill plan 长度/首尾 token 合同漂移");
        }

        let max_physical_k = if self
            .blocks
            .iter()
            .any(|block| block.physical_k_contract() == S14StarfoldPrefillPhysicalK::K8)
        {
            S14StarfoldPrefillPhysicalK::K8
        } else {
            S14StarfoldPrefillPhysicalK::K4
        };
        let mut expected_base = 0usize;
        let mut expected_input_token_id = self.first_input_token_id;
        for (block_index, block) in self.blocks.iter().enumerate() {
            block.validate()?;
            let remaining = self
                .prompt_len
                .checked_sub(1)
                .and_then(|transitions| transitions.checked_sub(expected_base))
                .context("StarFold prompt plan block base 超出 transition count")?;
            let expected_physical_k = match max_physical_k {
                S14StarfoldPrefillPhysicalK::K4 => S14StarfoldPrefillPhysicalK::K4,
                S14StarfoldPrefillPhysicalK::K8 => {
                    S14StarfoldPrefillPhysicalK::preferred_for_remaining(remaining)
                        .context("StarFold prompt plan 含零长度尾块")?
                }
            };
            if block.block_index != block_index
                || usize::try_from(block.base_position).ok() != Some(expected_base)
                || block.physical_k_contract() != expected_physical_k
                || block.physical_input_token_ids()[0] != expected_input_token_id
                || block.logical_lanes > remaining
            {
                bail!("StarFold prompt prefill plan block index/base/K/input 合同漂移");
            }
            expected_base = expected_base
                .checked_add(block.logical_lanes)
                .context("StarFold prompt plan logical lane 累加 overflow")?;
            expected_input_token_id = block.committed_input_token_id();
        }

        if expected_base != self.prompt_len - 1
            || u32::try_from(expected_base).ok() != Some(self.final_position)
            || expected_input_token_id != self.final_input_token_id
            || (self.prompt_len == 1 && !self.blocks.is_empty())
            || (self.prompt_len > 1 && self.blocks.is_empty())
        {
            bail!("StarFold prompt prefill plan 最终 position/input/block 链未闭合");
        }
        Ok(())
    }

    pub fn validate_final_state(&self, state: &DecoderStateV1) -> Result<()> {
        state
            .validate()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        if state.position != self.final_position
            || state.input_token_id != self.final_input_token_id
            || state.committed_tokens.len() != self.prompt_len - 1
        {
            bail!("StarFold prompt prefill 最终 position/input/N-1 ledger 未闭合");
        }
        Ok(())
    }
}

fn prefill_max_physical_k_from_env() -> Result<S14StarfoldPrefillPhysicalK> {
    let Some(raw) = std::env::var_os(S14_STARFOLD_PREFILL_MAX_K_ENV) else {
        return Ok(S14StarfoldPrefillPhysicalK::K8);
    };
    let value = raw
        .to_str()
        .context("S14 prefill max K 环境变量不是 UTF-8")?
        .trim();
    match value {
        "4" | "K4" | "k4" => Ok(S14StarfoldPrefillPhysicalK::K4),
        "8" | "K8" | "k8" | "" => Ok(S14StarfoldPrefillPhysicalK::K8),
        _ => bail!("{S14_STARFOLD_PREFILL_MAX_K_ENV} 只接受 4/K4 或 8/K8"),
    }
}

/// 单独的 teacher-forced ledger 语义；字段刻意不使用 predicted 命名。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldTeacherForcedTransitionReceipt {
    pub position: u32,
    pub input_token_id: u32,
    pub observed_next_input_token_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldPrefillArenaCommitIdentity {
    pub checkpoint_arena: vk::Buffer,
    pub checkpoint_arena_offset: u64,
    pub checkpoint_stride_bytes: u64,
    pub checkpoint_state_bytes: u64,
    pub selected_checkpoint_index: usize,
    pub producer_timeline: vk::Semaphore,
    pub producer_timeline_generation: u64,
    pub producer_timeline_value: u64,
    pub producer_drained_before_commit: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldTeacherForcedBlockCommitReceipt {
    pub block_index: usize,
    pub base_position: u32,
    pub logical_lanes: usize,
    pub physical_lanes: usize,
    pub transitions: Vec<S14StarfoldTeacherForcedTransitionReceipt>,
    pub committed_position: u32,
    pub committed_epoch: u64,
    pub committed_active_bank: usize,
    pub committed_input_token_id: u32,
    pub arena: S14StarfoldPrefillArenaCommitIdentity,
    pub device: WholeTokenDeviceBlockCommitReceipt,
    pub host_device_checkpoint_bytes_verified: bool,
    pub terminal_head_calls: u32,
    pub serial_token_forward_calls: u32,
}

/// Session 级持久 readback owner；跨 block 复用映射、command pool/buffer 与 fence。
#[must_use = "prefill session 结束后必须显式 destroy readback owner"]
pub struct S14StarfoldPrefillReadbackOwner {
    context: Arc<VulkanContext>,
    state_bytes: u64,
    readback: GpuBuffer,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    poisoned: bool,
}

impl S14StarfoldPrefillReadbackOwner {
    pub fn new(context: Arc<VulkanContext>, state_bytes: u64) -> Result<Self> {
        if state_bytes == 0 {
            bail!("prefill readback state bytes 不能为空");
        }
        let readback = GpuBuffer::new(
            &context,
            state_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        let pool = match unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        } {
            Ok(pool) => pool,
            Err(error) => {
                readback.destroy(&context);
                return Err(error.into());
            }
        };
        let command = match unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        } {
            Ok(commands) => commands[0],
            Err(error) => {
                unsafe { context.device.destroy_command_pool(pool, None) };
                readback.destroy(&context);
                return Err(error.into());
            }
        };
        let fence = match unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                unsafe { context.device.destroy_command_pool(pool, None) };
                readback.destroy(&context);
                return Err(error.into());
            }
        };
        Ok(Self {
            context,
            state_bytes,
            readback,
            command_pool: pool,
            command,
            fence,
            poisoned: false,
        })
    }

    pub fn destroy(self) -> Result<()> {
        if self.poisoned {
            bail!("prefill readback owner poisoned；资源保留到 Vulkan context teardown");
        }
        unsafe {
            self.context.device.destroy_fence(self.fence, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        self.readback.destroy(&self.context);
        Ok(())
    }

    /// 调用方持有的 sealed prefill product 已证明 recorder 完成 FullDepth43 并 drain；
    /// 此处只提交 arena→持久 readback copy，不再伪造或重复等待 producer timeline。
    fn read_selected_prefix_after_drain(
        &mut self,
        arena: &S14CausalBlockPrefixCheckpointArena,
        checkpoint_index: usize,
    ) -> Result<Vec<u8>> {
        arena.validate_host_readback_ready()?;
        if self.poisoned
            || !Arc::ptr_eq(&self.context, arena.context())
            || arena.layout().checkpoint_state_bytes != self.state_bytes
        {
            bail!("prefill readback owner context/state/phase 漂移");
        }
        let source_offset = arena.prefix_offset(checkpoint_index)?;
        let result = (|| -> Result<Vec<u8>> {
            unsafe {
                self.context
                    .device
                    .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
                self.context.device.reset_fences(&[self.fence])?;
                self.context.device.begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let source = vk::BufferMemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(arena.buffer().handle())
                    .offset(source_offset)
                    .size(self.state_bytes);
                self.context.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[source],
                    &[],
                );
                self.context.device.cmd_copy_buffer(
                    self.command,
                    arena.buffer().handle(),
                    self.readback.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(source_offset)
                        .size(self.state_bytes)],
                );
                self.context.device.end_command_buffer(self.command)?;
                let commands = [self.command];
                let submit = vk::SubmitInfo::default().command_buffers(&commands);
                self.context
                    .device
                    .queue_submit(self.context.q_graphics, &[submit], self.fence)?;
                self.context
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?;
            }
            Ok(unsafe {
                std::slice::from_raw_parts(
                    self.readback.mapped() as *const u8,
                    usize::try_from(self.state_bytes)?,
                )
                .to_vec()
            })
        })();
        if result.is_err()
            && unsafe { self.context.device.queue_wait_idle(self.context.q_graphics) }.is_err()
        {
            self.poisoned = true;
        }
        result
    }
}

fn commit_starfold_teacher_forced_prefill_block_inner(
    context: &VulkanContext,
    authoritative: &mut DecoderStateV1,
    device_state: &mut WholeTokenDeviceState,
    readback: &mut S14StarfoldPrefillReadbackOwner,
    block: &S14StarfoldTeacherForcedBlockPlan,
    prefix_arena: &Arc<S14CausalBlockPrefixCheckpointArena>,
    producer_ready: S14StarfoldTimelinePoint,
) -> Result<S14StarfoldTeacherForcedBlockCommitReceipt> {
    block.validate()?;
    authoritative
        .validate()
        .map_err(|error| anyhow::anyhow!("prefill base DecoderState 非法: {error}"))?;
    validate_base_identity(
        authoritative,
        device_state,
        block,
        prefix_arena,
        producer_ready,
    )?;
    // arena 已由同源 recorder seal，且 strong product 只会在 FullDepth43 drain 后导出。
    prefix_arena.validate_host_readback_ready()?;

    let checkpoint_bytes =
        readback.read_selected_prefix_after_drain(prefix_arena, block.checkpoint_index())?;
    let checkpoint = build_teacher_forced_checkpoint(authoritative, block, checkpoint_bytes)?;
    let transitions = build_transition_receipts(block)?;
    let prepared = device_state.prepare_starfold_teacher_forced_prefix_commit_after_drain(
        context,
        prefix_arena,
        block.logical_lanes,
        &checkpoint,
    )?;

    let arena_layout = prefix_arena.layout();
    let arena = S14StarfoldPrefillArenaCommitIdentity {
        checkpoint_arena: prefix_arena.buffer().handle(),
        checkpoint_arena_offset: prefix_arena.prefix_offset(0)?,
        checkpoint_stride_bytes: arena_layout.checkpoint_stride_bytes,
        checkpoint_state_bytes: arena_layout.checkpoint_state_bytes,
        selected_checkpoint_index: block.checkpoint_index(),
        producer_timeline: producer_ready.semaphore,
        producer_timeline_generation: producer_ready.generation,
        producer_timeline_value: producer_ready.value,
        producer_drained_before_commit: true,
    };
    let committed_position = checkpoint.position;
    let committed_epoch = checkpoint.commit_epoch;
    let committed_active_bank = usize::from(checkpoint.active_fixed_bank);
    let committed_input_token_id = checkpoint.input_token_id;

    // 此后只有不可失败的 owner swap / metadata publish。
    *authoritative = checkpoint;
    let device = device_state.publish_prepared_block_commit(prepared);
    assert_eq!(device.position, committed_position);
    assert_eq!(device.epoch, committed_epoch);
    assert_eq!(device.active_bank, committed_active_bank);
    assert_eq!(device.accepted_tokens, block.logical_lanes);
    assert_eq!(device.checkpoint_index, block.checkpoint_index());
    assert!(device.host_device_bytes_verified);

    Ok(S14StarfoldTeacherForcedBlockCommitReceipt {
        block_index: block.block_index,
        base_position: block.base_position,
        logical_lanes: block.logical_lanes,
        physical_lanes: block.physical_k(),
        transitions,
        committed_position,
        committed_epoch,
        committed_active_bank,
        committed_input_token_id,
        arena,
        device,
        host_device_checkpoint_bytes_verified: true,
        terminal_head_calls: 0,
        serial_token_forward_calls: 0,
    })
}

fn validate_base_identity(
    authoritative: &DecoderStateV1,
    device: &WholeTokenDeviceState,
    block: &S14StarfoldTeacherForcedBlockPlan,
    arena: &Arc<S14CausalBlockPrefixCheckpointArena>,
    producer_ready: S14StarfoldTimelinePoint,
) -> Result<()> {
    let layout = arena.layout();
    if authoritative.position != block.base_position
        || authoritative.native.position != block.base_position
        || authoritative.commit_epoch != u64::from(block.base_position)
        || usize::from(authoritative.active_fixed_bank) != (block.base_position as usize & 1)
        || authoritative.input_token_id != block.physical_input_token_ids()[0]
        || device.epoch() != authoritative.commit_epoch
        || device.active_bank() != usize::from(authoritative.active_fixed_bank)
        || device.state_bytes() != authoritative.native_arena.len() as u64
        || arena.base_position() != block.base_position
        || layout.block_size != block.physical_k()
        || layout.checkpoint_state_bytes != device.state_bytes()
        || producer_ready.semaphore == vk::Semaphore::null()
        || producer_ready.generation == 0
        || producer_ready.value == 0
    {
        bail!("StarFold prefill base host/device/arena/timeline identity 漂移");
    }
    if block.base_position == 0
        && (authoritative.commit_epoch != 0
            || authoritative.active_fixed_bank != 0
            || !authoritative.committed_tokens.is_empty())
    {
        bail!("StarFold ForcedPrefill base0 必须来自 position0/epoch0/bank0 空 ledger");
    }
    authoritative
        .position
        .checked_add(block.logical_lanes as u32)
        .filter(|&end| end <= authoritative.native.max_seq_len)
        .context("StarFold prefill block 越出 max_seq_len")?;
    Ok(())
}

fn build_transition_receipts(
    block: &S14StarfoldTeacherForcedBlockPlan,
) -> Result<Vec<S14StarfoldTeacherForcedTransitionReceipt>> {
    (0..block.logical_lanes)
        .map(|lane| {
            Ok(S14StarfoldTeacherForcedTransitionReceipt {
                position: block
                    .base_position
                    .checked_add(lane as u32)
                    .context("teacher-forced receipt position overflow")?,
                input_token_id: block.physical_input_token_ids()[lane],
                observed_next_input_token_id: block.observed_next_input_token_ids[lane],
            })
        })
        .collect()
}

fn build_teacher_forced_checkpoint(
    authoritative: &DecoderStateV1,
    block: &S14StarfoldTeacherForcedBlockPlan,
    checkpoint_bytes: Vec<u8>,
) -> Result<DecoderStateV1> {
    let next_position = block.committed_position();
    let next_epoch = authoritative
        .commit_epoch
        .checked_add(block.logical_lanes as u64)
        .context("teacher-forced checkpoint epoch overflow")?;
    let mut native = authoritative.native.clone();
    native.position = next_position;
    native.validate_for(GraphProfile::FullDepth43NativeTop6)?;
    let native_arena = NativeStateArena::from_verified_checkpoint_bytes(&native, checkpoint_bytes)
        .context("从同源 StarFold prefix arena 构造 prefill host checkpoint")?;
    let mut ledger = authoritative.committed_tokens.clone();
    for lane in 0..block.logical_lanes {
        // DecoderState ABI v1 没有 teacher-forced record tag。仅在 prefill checkpoint 中，旧
        // predicted_token_id 槽承载“已观测的下一输入”，绝不是模型预测；生成质量评估不得读取。
        ledger.push(TokenRecord {
            position: block.base_position + lane as u32,
            input_token_id: block.physical_input_token_ids()[lane],
            predicted_token_id: block.observed_next_input_token_ids[lane],
        });
    }
    let checkpoint = DecoderStateV1 {
        abi_version: authoritative.abi_version,
        commit_epoch: next_epoch,
        position: next_position,
        input_token_id: block.committed_input_token_id(),
        active_fixed_bank: authoritative.active_fixed_bank ^ ((block.logical_lanes as u8) & 1),
        committed_tokens: ledger,
        native,
        native_arena,
    };
    checkpoint
        .validate()
        .map_err(|error| anyhow::anyhow!("teacher-forced checkpoint 非法: {error}"))?;
    Ok(checkpoint)
}
