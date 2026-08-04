//! FullDepth43 whole-token candidate 的 Vulkan 状态所有权。
//!
//! 一个 token 的全部算子只允许写 inactive bank。开始候选时由 GPU 复制当前
//! committed bank；任何数值错误、最终 token 不一致或上层合同失败都只丢弃候选。
//! `commit_candidate` 是唯一切换 active bank/epoch 的位置。

use crate::{
    s14_causal_block_layer::{
        S14CausalBlockDeviceCheckpointStorage, S14CausalBlockDeviceFutureReceipt,
        S14CausalBlockOwnedDeviceFuture, S14CausalBlockSelectedPrefix,
    },
    s14_causal_block_prefix_arena::S14CausalBlockPrefixCheckpointArena,
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::DecoderStateV1;
use std::{marker::PhantomData, ops::Range};

const STATUS_BYTES: u64 = 4;
const UPDATE_CHUNK_BYTES: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidatePhase {
    Idle,
    /// candidate prologue 正在录制，尚未交给外部队列。
    Recording,
    /// prologue 已作为首个 compute segment 提交；后端可继续逐层提交。
    InFlight,
    Ready,
    Failed,
    /// 选中的 K-prefix checkpoint 已等待 producer timeline 并复制进 scratch；尚未发布。
    BlockPrepared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WholeTokenPreparedCommit {
    expected_epoch: u64,
    next_epoch: u64,
    next_bank: usize,
    position: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WholeTokenDeviceCommitReceipt {
    pub epoch: u64,
    pub active_bank: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct WholeTokenPreparedBlockCommit {
    expected_epoch: u64,
    next_epoch: u64,
    next_bank: usize,
    base_position: u32,
    next_position: u32,
    block_size: usize,
    accepted_tokens: usize,
    checkpoint_index: usize,
    host_device_bytes_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WholeTokenDeviceBlockCommitReceipt {
    pub epoch: u64,
    pub active_bank: usize,
    pub position: u32,
    pub accepted_tokens: usize,
    pub checkpoint_index: usize,
    pub host_device_bytes_verified: bool,
}

/// 当前 committed device checkpoint 的借用式强身份。`buffer` 的生命周期仍由
/// `WholeTokenDeviceState` 独占；本回执只允许下游重绑器在 owner 存活且 identity 未漂移时使用。
#[derive(Debug)]
pub struct WholeTokenDeviceCommittedCheckpointBinding<'owner> {
    buffer: vk::Buffer,
    state_bytes: u64,
    epoch: u64,
    active_bank: usize,
    owner: PhantomData<&'owner WholeTokenDeviceState>,
}

impl WholeTokenDeviceCommittedCheckpointBinding<'_> {
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn state_bytes(&self) -> u64 {
        self.state_bytes
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn active_bank(&self) -> usize {
        self.active_bank
    }
}

/// 从已排空单token runtime移交给causal-block builder的唯一 committed bank。
/// 调用方取得后必须把`buffer`纳入显式external owner生命周期。
pub struct WholeTokenDetachedCommittedState {
    pub buffer: GpuBuffer,
    pub state_bytes: u64,
    pub epoch: u64,
    pub active_bank: usize,
    /// 只用于后续 production owner 的同 context 身份核验，不拥有 device/queue。
    pub source_device: vk::Device,
    pub source_graphics_queue: vk::Queue,
    pub source_graphics_queue_family: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WholeTokenBlockPublishPlan {
    next_epoch: u64,
    next_bank: usize,
    next_position: u32,
    checkpoint_index: usize,
    checkpoint_offset: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WholeTokenBlockPrefixSource {
    base_position: u32,
    block_size: usize,
    checkpoint_arena: vk::Buffer,
    checkpoint_arena_offset: u64,
    checkpoint_arena_bytes: u64,
    checkpoint_stride_bytes: u64,
    checkpoint_state_bytes: u64,
    ready_timeline: vk::Semaphore,
    ready_timeline_value: u64,
}

pub struct WholeTokenDeviceState {
    banks: [GpuBuffer; 2],
    /// Block publish 总是先复制到第三块 scratch；即使接受偶数 token，也不覆盖当前 active。
    block_publish_scratch: GpuBuffer,
    /// 每个 block 发布前的 host/device 字节核验复用同一持久映射，禁止热路径重复分配。
    block_publish_readback: GpuBuffer,
    sticky_status: GpuBuffer,
    state_bytes: u64,
    active_bank: usize,
    epoch: u64,
    candidate_position: Option<u32>,
    phase: CandidatePhase,
    /// 两个 bank 初始一致；提交后只需把上个 token 的脏区同步到旧 bank。
    last_committed_dirty: Vec<Range<u64>>,
    repair_dirty: Vec<Range<u64>>,
    candidate_dirty: Vec<Range<u64>>,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}

impl WholeTokenDeviceState {
    pub fn new(ctx: &VulkanContext, initial_state: &[u8], epoch: u64) -> Result<Self> {
        Self::from_committed_checkpoint(ctx, initial_state, epoch, 0)
    }

    /// 从已通过 durable checkpoint 长度/SHA/身份校验的 active bank 镜像恢复。
    /// 两个 bank 初始写入同一份已提交字节，但 epoch/active bank 保留持久身份。
    pub fn from_committed_checkpoint(
        ctx: &VulkanContext,
        committed_state: &[u8],
        epoch: u64,
        active_bank: usize,
    ) -> Result<Self> {
        if active_bank > 1 {
            bail!("whole-token durable checkpoint active bank越界: {active_bank}");
        }
        Self::new_with_active_bank(ctx, committed_state, epoch, active_bank)
    }

    fn new_with_active_bank(
        ctx: &VulkanContext,
        initial_state: &[u8],
        epoch: u64,
        active_bank: usize,
    ) -> Result<Self> {
        if initial_state.is_empty() || initial_state.len() % 4 != 0 {
            bail!("whole-token device state 必须非空且 4-byte 对齐");
        }
        let state_bytes = u64::try_from(initial_state.len())?;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let bank0 = GpuBuffer::new_vram(ctx, state_bytes, usage)
            .context("allocate whole-token state bank0")?;
        let bank1 = match GpuBuffer::new_vram(ctx, state_bytes, usage) {
            Ok(buffer) => buffer,
            Err(error) => {
                bank0.destroy(ctx);
                return Err(error).context("allocate whole-token state bank1");
            }
        };
        let block_publish_scratch = match GpuBuffer::new_vram(ctx, state_bytes, usage) {
            Ok(buffer) => buffer,
            Err(error) => {
                bank1.destroy(ctx);
                bank0.destroy(ctx);
                return Err(error).context("allocate whole-token block publish scratch");
            }
        };
        let block_publish_readback = match GpuBuffer::new(
            ctx,
            state_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                block_publish_scratch.destroy(ctx);
                bank1.destroy(ctx);
                bank0.destroy(ctx);
                return Err(error).context("allocate whole-token block publish readback");
            }
        };
        let sticky_status = match GpuBuffer::new(
            ctx,
            STATUS_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                block_publish_readback.destroy(ctx);
                block_publish_scratch.destroy(ctx);
                bank1.destroy(ctx);
                bank0.destroy(ctx);
                return Err(error).context("allocate whole-token sticky status");
            }
        };
        let command_pool = unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let command = unsafe {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };
        let fence = unsafe {
            ctx.device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        let staging = GpuBuffer::new_staging(ctx, state_bytes)?;
        unsafe {
            staging.write_at(0, initial_state);
            ctx.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            for bank in [&bank0, &bank1] {
                ctx.device.cmd_copy_buffer(
                    command,
                    staging.handle(),
                    bank.handle(),
                    &[vk::BufferCopy::default().size(state_bytes)],
                );
            }
            ctx.device
                .cmd_fill_buffer(command, sticky_status.handle(), 0, STATUS_BYTES, 0);
            ctx.device.end_command_buffer(command)?;
            submit_and_wait(ctx, command, fence)?;
        }
        staging.destroy(ctx);
        Ok(Self {
            banks: [bank0, bank1],
            block_publish_scratch,
            block_publish_readback,
            sticky_status,
            state_bytes,
            active_bank,
            epoch,
            candidate_position: None,
            phase: CandidatePhase::Idle,
            last_committed_dirty: Vec::new(),
            repair_dirty: Vec::new(),
            candidate_dirty: Vec::new(),
            command_pool,
            command,
            fence,
        })
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn active_bank(&self) -> usize {
        self.active_bank
    }

    pub fn state_bytes(&self) -> u64 {
        self.state_bytes
    }

    /// 返回真实 active bank 的窄 binding，不制造 initial hidden，也不延长 buffer 生命周期。
    /// 第二个 StarFold K4 block 必须把本 binding 交给 committed-state/HC-QKV provider
    /// 重绑器；token embedding hidden 仍由独立 K4 input owner 生成。重绑器接入前只能
    /// 生成 launch gate，禁止伪造 host state、hidden 或任意 Vulkan handle。
    pub fn committed_checkpoint_binding(
        &self,
    ) -> Result<WholeTokenDeviceCommittedCheckpointBinding<'_>> {
        if self.phase != CandidatePhase::Idle {
            bail!("candidate 存在时禁止借用 committed device checkpoint");
        }
        let buffer = self.banks[self.active_bank].handle();
        if buffer == vk::Buffer::null() || self.state_bytes == 0 || self.active_bank >= 2 {
            bail!("committed device checkpoint binding 非法");
        }
        Ok(WholeTokenDeviceCommittedCheckpointBinding {
            buffer,
            state_bytes: self.state_bytes,
            epoch: self.epoch,
            active_bank: self.active_bank,
            owner: PhantomData,
        })
    }

    pub fn candidate_position(&self) -> Option<u32> {
        self.candidate_position
    }

    pub fn active_buffer(&self) -> &GpuBuffer {
        &self.banks[self.active_bank]
    }

    /// 在不消费双bank device state的前提下，把当前已提交bank真实复制为
    /// block-major prefix initializer可消费的detached snapshot。这是device→device
    /// copy，不经host checkpoint；保留的`WholeTokenDeviceState`继续负责第一个
    /// K-block selected checkpoint的两阶段发布。
    pub fn snapshot_detached_committed_state(
        &mut self,
        ctx: &VulkanContext,
    ) -> Result<WholeTokenDetachedCommittedState> {
        if self.phase != CandidatePhase::Idle
            || self.candidate_position.is_some()
            || self.active_bank > 1
        {
            bail!("whole-token device state非idle时禁止snapshot committed bank");
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let snapshot = GpuBuffer::new_vram(ctx, self.state_bytes, usage)
            .context("allocate whole-token committed snapshot")?;
        let copied = (|| -> Result<()> {
            unsafe {
                ctx.device.reset_command_pool(
                    self.command_pool,
                    vk::CommandPoolResetFlags::RELEASE_RESOURCES,
                )?;
                ctx.device.begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                ctx.device.cmd_copy_buffer(
                    self.command,
                    self.active_buffer().handle(),
                    snapshot.handle(),
                    &[vk::BufferCopy::default().size(self.state_bytes)],
                );
                ctx.device.end_command_buffer(self.command)?;
                submit_and_wait(ctx, self.command, self.fence)?;
            }
            Ok(())
        })();
        if let Err(error) = copied {
            snapshot.destroy(ctx);
            return Err(error.context("copy whole-token committed snapshot"));
        }
        Ok(WholeTokenDetachedCommittedState {
            buffer: snapshot,
            state_bytes: self.state_bytes,
            epoch: self.epoch,
            active_bank: self.active_bank,
            source_device: ctx.device.handle(),
            source_graphics_queue: ctx.q_graphics,
            source_graphics_queue_family: ctx.qf_graphics,
        })
    }

    /// position0成功提交且所有外部timeline已排空后，把active committed bank移交给K-block。
    /// inactive bank、block scratch、sticky status与单token command owner在这里真实销毁。
    pub fn into_detached_committed_state(
        self,
        ctx: &VulkanContext,
    ) -> Result<WholeTokenDetachedCommittedState> {
        if self.phase != CandidatePhase::Idle
            || self.candidate_position.is_some()
            || self.active_bank > 1
        {
            bail!("whole-token device state非idle时禁止移交committed bank");
        }
        let Self {
            banks: [bank0, bank1],
            block_publish_scratch,
            block_publish_readback,
            sticky_status,
            state_bytes,
            active_bank,
            epoch,
            candidate_position: _,
            phase: _,
            last_committed_dirty: _,
            repair_dirty: _,
            candidate_dirty: _,
            command_pool,
            command: _,
            fence,
        } = self;
        unsafe {
            ctx.device.destroy_fence(fence, None);
            ctx.device.destroy_command_pool(command_pool, None);
        }
        sticky_status.destroy(ctx);
        block_publish_readback.destroy(ctx);
        block_publish_scratch.destroy(ctx);
        let (buffer, inactive) = match active_bank {
            0 => (bank0, bank1),
            1 => (bank1, bank0),
            _ => unreachable!("active bank已在资源移交前校验"),
        };
        inactive.destroy(ctx);
        Ok(WholeTokenDetachedCommittedState {
            buffer,
            state_bytes,
            epoch,
            active_bank,
            source_device: ctx.device.handle(),
            source_graphics_queue: ctx.q_graphics,
            source_graphics_queue_family: ctx.qf_graphics,
        })
    }

    pub fn candidate_buffer(&self) -> Result<&GpuBuffer> {
        if !matches!(
            self.phase,
            CandidatePhase::Recording
                | CandidatePhase::InFlight
                | CandidatePhase::Ready
                | CandidatePhase::Failed
        ) {
            bail!("whole-token candidate 尚未开始或 block publish 已prepared");
        }
        Ok(&self.banks[1 - self.active_bank])
    }

    pub fn sticky_status_buffer(&self) -> Result<&GpuBuffer> {
        if !matches!(
            self.phase,
            CandidatePhase::Recording | CandidatePhase::InFlight
        ) {
            bail!("sticky status 只能在 recording/in-flight 阶段绑定");
        }
        Ok(&self.sticky_status)
    }

    /// 开始一个候选 command graph；返回的 command 已经处于 recording 状态。
    pub fn begin_candidate(
        &mut self,
        ctx: &VulkanContext,
        expected_epoch: u64,
    ) -> Result<vk::CommandBuffer> {
        self.begin_candidate_for_position(ctx, expected_epoch, 0)
    }

    /// position-aware candidate 入口。position1 prologue 会把上一提交的 row0 dirty set
    /// 从 active bank 修复到 inactive bank，后续 attention 应从 candidate bank 读取该行。
    pub fn begin_candidate_for_position(
        &mut self,
        ctx: &VulkanContext,
        expected_epoch: u64,
        position: u32,
    ) -> Result<vk::CommandBuffer> {
        validate_candidate_position(position)?;
        if self.phase != CandidatePhase::Idle {
            bail!("已有 whole-token candidate 未结束");
        }
        if expected_epoch != self.epoch {
            bail!(
                "whole-token device epoch stale: expected={} actual={}",
                expected_epoch,
                self.epoch
            );
        }
        unsafe {
            ctx.device.reset_command_pool(
                self.command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
            ctx.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let sync_ranges = coalesce_ranges(
                self.last_committed_dirty
                    .iter()
                    .chain(&self.repair_dirty)
                    .cloned()
                    .collect(),
            );
            if !sync_ranges.is_empty() {
                let regions: Vec<vk::BufferCopy> = sync_ranges
                    .iter()
                    .map(|range| {
                        vk::BufferCopy::default()
                            .src_offset(range.start)
                            .dst_offset(range.start)
                            .size(range.end - range.start)
                    })
                    .collect();
                ctx.device.cmd_copy_buffer(
                    self.command,
                    self.banks[self.active_bank].handle(),
                    self.banks[1 - self.active_bank].handle(),
                    &regions,
                );
            }
            ctx.device.cmd_fill_buffer(
                self.command,
                self.sticky_status.handle(),
                0,
                STATUS_BYTES,
                0,
            );
            let barriers = [
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(self.banks[1 - self.active_bank].handle())
                    .offset(0)
                    .size(self.state_bytes),
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(self.sticky_status.handle())
                    .offset(0)
                    .size(STATUS_BYTES),
            ];
            ctx.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
        // repair_dirty 只有在 candidate 真正发布后才能清除。外部后端可能在提交
        // bootstrap 之前失败；若此处提前清空，下一 token 会遗漏对 inactive bank 的修复。
        self.candidate_dirty.clear();
        self.candidate_position = Some(position);
        self.phase = CandidatePhase::Recording;
        Ok(self.command)
    }

    /// 把 host 侧小 write-set 录入 inactive bank。Vulkan `updateBuffer` 会在
    /// record 时复制数据；调用者无需延长 slice 生命周期。
    pub fn record_candidate_patch(
        &mut self,
        ctx: &VulkanContext,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        if self.phase != CandidatePhase::Recording {
            bail!("whole-token state patch 只能在 recording 阶段写入");
        }
        if bytes.is_empty() || offset % 4 != 0 || bytes.len() % 4 != 0 {
            bail!("whole-token state patch 必须非空且 offset/size 4-byte 对齐");
        }
        let end = offset
            .checked_add(u64::try_from(bytes.len())?)
            .ok_or_else(|| anyhow::anyhow!("whole-token state patch overflow"))?;
        if end > self.state_bytes {
            bail!("whole-token state patch 越出 candidate bank");
        }
        let candidate = &self.banks[1 - self.active_bank];
        unsafe {
            let pre = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .buffer(candidate.handle())
                .offset(offset)
                .size(bytes.len() as u64);
            ctx.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[pre],
                &[],
            );
            let mut cursor = 0usize;
            while cursor < bytes.len() {
                let take = (bytes.len() - cursor).min(UPDATE_CHUNK_BYTES);
                ctx.device.cmd_update_buffer(
                    self.command,
                    candidate.handle(),
                    offset + cursor as u64,
                    &bytes[cursor..cursor + take],
                );
                cursor += take;
            }
            let post = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(candidate.handle())
                .offset(offset)
                .size(bytes.len() as u64);
            ctx.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[post],
                &[],
            );
        }
        self.mark_candidate_dirty(offset, bytes.len() as u64)?;
        Ok(())
    }

    /// 注册由其他 compute shader 写入 candidate state 的区间。允许保守扩大，
    /// 但禁止漏报；下次 token 只同步这些区间，不再复制完整 46 MiB arena。
    /// 外部分段提交后 command 已经结束，因此这里只更新 host 元数据，不再录 Vulkan 命令。
    pub fn mark_candidate_dirty(&mut self, offset: u64, bytes: u64) -> Result<()> {
        if !matches!(
            self.phase,
            CandidatePhase::Recording | CandidatePhase::InFlight
        ) {
            bail!("whole-token dirty range 只能在 recording/in-flight 阶段注册");
        }
        if bytes == 0 || offset % 4 != 0 || bytes % 4 != 0 {
            bail!("whole-token dirty range 必须非空且 4-byte 对齐");
        }
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("whole-token dirty range overflow"))?;
        if end > self.state_bytes {
            bail!("whole-token dirty range 越出 candidate bank");
        }
        self.candidate_dirty.push(offset..end);
        Ok(())
    }

    /// 把 `begin_candidate` 返回的 prologue command 所有权移交给外部分段后端。
    ///
    /// 调用者必须先结束并提交该 command，且保证它在所有后续 candidate compute 之前执行。
    /// 后续 layer/final 可以使用任意数量的 transfer/compute submit，但全部只能绑定
    /// `candidate_buffer()` 与同一个 sticky status。此函数不等待主机。
    pub fn mark_candidate_in_flight(&mut self) -> Result<()> {
        if self.phase != CandidatePhase::Recording {
            bail!("whole-token candidate 只有 recording 阶段可以移交外部队列");
        }
        self.phase = CandidatePhase::InFlight;
        Ok(())
    }

    /// 在外部后端需要长期借用 candidate buffer 之前，预先把候选所有权移交给该后端。
    /// `begin_candidate` 返回的 prologue command 此时仍可处于 recording；调用方随后必须把它
    /// 作为共享 timeline 的第一个 compute segment 提交。自本调用成功起，任何错误都只能先
    /// drain 已提交工作（没有提交时 drain 可为空），再调用 `rollback_external_candidate`。
    pub fn arm_external_candidate(&mut self) -> Result<()> {
        if self.phase != CandidatePhase::Recording {
            bail!("whole-token candidate 只有 recording 阶段可以预先移交外部后端");
        }
        self.phase = CandidatePhase::InFlight;
        Ok(())
    }

    /// 外部 timeline 已在 token 末尾完成唯一一次 host wait 后调用。
    /// 本函数不提交也不等待，只读取 host-visible sticky 并把 candidate 置为 Ready。
    pub fn finish_external_candidate(
        &mut self,
        base_epoch: u64,
        candidate_bank: usize,
    ) -> Result<()> {
        if self.phase != CandidatePhase::InFlight {
            bail!("whole-token 外部 candidate 尚未 in-flight");
        }
        if base_epoch != self.epoch || candidate_bank != 1 - self.active_bank {
            bail!(
                "whole-token external completion 身份漂移: epoch={base_epoch}/{}, bank={candidate_bank}/{}",
                self.epoch,
                1 - self.active_bank
            );
        }
        let status = unsafe { *(self.sticky_status.mapped() as *const u32) };
        if status != 0 {
            self.phase = CandidatePhase::Failed;
            bail!("whole-token GPU sticky status=0x{status:08x}");
        }
        self.phase = CandidatePhase::Ready;
        Ok(())
    }

    pub fn submit_candidate(&mut self, ctx: &VulkanContext) -> Result<()> {
        if self.phase != CandidatePhase::Recording {
            bail!("whole-token candidate 不在 recording 阶段");
        }
        unsafe {
            ctx.device.end_command_buffer(self.command)?;
            submit_and_wait(ctx, self.command, self.fence)?;
        }
        let status = unsafe { *(self.sticky_status.mapped() as *const u32) };
        if status != 0 {
            self.phase = CandidatePhase::Failed;
            bail!("whole-token GPU sticky status=0x{status:08x}");
        }
        self.phase = CandidatePhase::Ready;
        Ok(())
    }

    /// 在发布 host/device 任一侧之前完成所有可能失败的 device commit 检查。
    pub fn prepare_candidate_commit(
        &self,
        expected_epoch: u64,
    ) -> Result<WholeTokenPreparedCommit> {
        if self.phase != CandidatePhase::Ready {
            bail!("whole-token candidate 尚未成功提交 GPU graph");
        }
        if expected_epoch != self.epoch {
            bail!("whole-token commit epoch stale");
        }
        if self.candidate_dirty.is_empty() {
            bail!("whole-token candidate 没有注册任何状态写入");
        }
        let position = self
            .candidate_position
            .ok_or_else(|| anyhow::anyhow!("whole-token candidate position 缺失"))?;
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("whole-token device epoch overflow"))?;
        Ok(WholeTokenPreparedCommit {
            expected_epoch,
            next_epoch,
            next_bank: 1 - self.active_bank,
            position,
        })
    }

    pub fn prepare_candidate_commit_for_position(
        &self,
        expected_epoch: u64,
        expected_position: u32,
    ) -> Result<WholeTokenPreparedCommit> {
        if self.candidate_position != Some(expected_position) {
            bail!("whole-token commit position stale");
        }
        self.prepare_candidate_commit(expected_epoch)
    }

    /// `prepared` 由当前对象在没有中间 mutation 时签发；因此发布阶段不再包含
    /// 可恢复错误。调用方可以先在 host clone 上完成全部验证，再同时发布两侧。
    pub fn publish_prepared_commit(
        &mut self,
        prepared: WholeTokenPreparedCommit,
    ) -> WholeTokenDeviceCommitReceipt {
        assert_eq!(self.phase, CandidatePhase::Ready);
        assert_eq!(self.epoch, prepared.expected_epoch);
        assert_eq!(self.candidate_position, Some(prepared.position));
        assert_eq!(prepared.next_bank, 1 - self.active_bank);
        assert!(!self.candidate_dirty.is_empty());
        self.active_bank = 1 - self.active_bank;
        self.epoch = prepared.next_epoch;
        self.last_committed_dirty = coalesce_ranges(std::mem::take(&mut self.candidate_dirty));
        self.repair_dirty.clear();
        self.candidate_position = None;
        self.phase = CandidatePhase::Idle;
        WholeTokenDeviceCommitReceipt {
            epoch: self.epoch,
            active_bank: self.active_bank,
        }
    }

    /// 等待 block producer timeline，把最长前缀选中的完整 device checkpoint 复制到
    /// 第三块 scratch。此阶段可以失败，但绝不覆盖当前 active bank。
    pub fn prepare_block_prefix_commit(
        &mut self,
        ctx: &VulkanContext,
        selected: &S14CausalBlockSelectedPrefix<'_>,
    ) -> Result<WholeTokenPreparedBlockCommit> {
        let device_future = selected.device_receipt();
        self.prepare_owned_block_prefix_commit(
            ctx,
            device_future,
            selected.accepted_tokens(),
            selected.checkpoint_index(),
            selected.checkpoint(),
            false,
        )
    }

    /// StarFold-neutral 的 device prefix prepare：只消费 terminal 返回的 owned future、
    /// 最长前缀选择与同一份 host checkpoint，不依赖旧 union/page-plan 包装。
    pub fn prepare_starfold_block_prefix_commit(
        &mut self,
        ctx: &VulkanContext,
        device_future: &S14CausalBlockOwnedDeviceFuture,
        accepted_tokens: usize,
        checkpoint_index: usize,
        checkpoint: &DecoderStateV1,
    ) -> Result<WholeTokenPreparedBlockCommit> {
        self.prepare_owned_block_prefix_commit(
            ctx,
            device_future.receipt(),
            accepted_tokens,
            checkpoint_index,
            checkpoint,
            true,
        )
    }

    /// Prompt prefill 的 after-drain 非 terminal device prepare。源 checkpoint 直接来自
    /// 同一份 sealed `S14CausalBlockPrefixCheckpointArena`；producer 已由 adapter drain，
    /// 因此本入口不接收/等待 timeline，也不构造 terminal `OwnedDeviceFuture`。
    pub(crate) fn prepare_starfold_teacher_forced_prefix_commit_after_drain(
        &mut self,
        ctx: &VulkanContext,
        prefix_arena: &S14CausalBlockPrefixCheckpointArena,
        committed_prefix: usize,
        checkpoint: &DecoderStateV1,
    ) -> Result<WholeTokenPreparedBlockCommit> {
        prefix_arena.validate_host_readback_ready()?;
        let layout = prefix_arena.layout();
        if !std::ptr::eq(ctx, prefix_arena.context().as_ref())
            || !matches!(layout.block_size, 4 | 8)
        {
            bail!(
                "StarFold after-drain prefill prefix arena/context/K identity 非法: block_size={}",
                layout.block_size
            );
        }
        let source = WholeTokenBlockPrefixSource {
            base_position: prefix_arena.base_position(),
            block_size: layout.block_size,
            checkpoint_arena: prefix_arena.buffer().handle(),
            checkpoint_arena_offset: prefix_arena.prefix_offset(0)?,
            checkpoint_arena_bytes: layout.used_bytes,
            checkpoint_stride_bytes: layout.checkpoint_stride_bytes,
            checkpoint_state_bytes: layout.checkpoint_state_bytes,
            ready_timeline: vk::Semaphore::null(),
            ready_timeline_value: 0,
        };
        self.prepare_block_prefix_from_source(
            ctx,
            source,
            committed_prefix,
            committed_prefix.saturating_sub(1),
            checkpoint,
            true,
        )
    }

    fn prepare_owned_block_prefix_commit(
        &mut self,
        ctx: &VulkanContext,
        device_future: S14CausalBlockDeviceFutureReceipt,
        accepted_tokens: usize,
        checkpoint_index: usize,
        checkpoint: &DecoderStateV1,
        verify_host_device_bytes: bool,
    ) -> Result<WholeTokenPreparedBlockCommit> {
        if self.phase != CandidatePhase::Idle {
            bail!("whole-token device 只有 idle 阶段可以 prepare block prefix");
        }
        if device_future.storage != S14CausalBlockDeviceCheckpointStorage::PrefixCheckpoints {
            bail!("owned block future 必须来自 PrefixCheckpoints storage");
        }
        device_future
            .validate(
                device_future.base_position,
                device_future.block_size,
                device_future.final_hidden,
            )
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let source = WholeTokenBlockPrefixSource {
            base_position: device_future.base_position,
            block_size: device_future.block_size,
            checkpoint_arena: device_future.checkpoint_arena,
            checkpoint_arena_offset: device_future.checkpoint_arena_offset,
            checkpoint_arena_bytes: device_future.checkpoint_arena_bytes,
            checkpoint_stride_bytes: device_future.checkpoint_stride_bytes,
            checkpoint_state_bytes: device_future.checkpoint_state_bytes,
            ready_timeline: device_future.ready_timeline,
            ready_timeline_value: device_future.ready_timeline_value,
        };
        self.prepare_block_prefix_from_source(
            ctx,
            source,
            accepted_tokens,
            checkpoint_index,
            checkpoint,
            verify_host_device_bytes,
        )
    }

    fn prepare_block_prefix_from_source(
        &mut self,
        ctx: &VulkanContext,
        source: WholeTokenBlockPrefixSource,
        accepted_tokens: usize,
        checkpoint_index: usize,
        checkpoint: &DecoderStateV1,
        verify_host_device_bytes: bool,
    ) -> Result<WholeTokenPreparedBlockCommit> {
        if self.phase != CandidatePhase::Idle {
            bail!("whole-token device 只有 idle 阶段可以 prepare block prefix");
        }
        let plan = build_block_publish_plan(
            self.active_bank,
            self.epoch,
            self.state_bytes,
            source,
            accepted_tokens,
            checkpoint,
        )?;
        if plan.checkpoint_index != checkpoint_index {
            bail!("selected host/device checkpoint index 不同源");
        }

        unsafe {
            ctx.device.reset_command_pool(
                self.command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
            ctx.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if source.ready_timeline == vk::Semaphore::null() {
                let drained_source = vk::BufferMemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(source.checkpoint_arena)
                    .offset(source.checkpoint_arena_offset)
                    .size(source.checkpoint_arena_bytes);
                ctx.device.cmd_pipeline_barrier(
                    self.command,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[drained_source],
                    &[],
                );
            }
            ctx.device.cmd_copy_buffer(
                self.command,
                source.checkpoint_arena,
                self.block_publish_scratch.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(plan.checkpoint_offset)
                    .size(self.state_bytes)],
            );
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::SHADER_WRITE
                        | vk::AccessFlags::TRANSFER_READ,
                )
                .buffer(self.block_publish_scratch.handle())
                .offset(0)
                .size(self.state_bytes);
            ctx.device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
            ctx.device.end_command_buffer(self.command)?;
            if source.ready_timeline == vk::Semaphore::null() {
                submit_and_wait(ctx, self.command, self.fence)?;
            } else {
                submit_and_wait_on_timeline(
                    ctx,
                    self.command,
                    self.fence,
                    source.ready_timeline,
                    source.ready_timeline_value,
                )?;
            }
        }
        if verify_host_device_bytes {
            self.verify_block_publish_scratch_matches(ctx, checkpoint.native_arena.bytes())?;
        }
        self.phase = CandidatePhase::BlockPrepared;
        Ok(WholeTokenPreparedBlockCommit {
            expected_epoch: self.epoch,
            next_epoch: plan.next_epoch,
            next_bank: plan.next_bank,
            base_position: source.base_position,
            next_position: plan.next_position,
            block_size: source.block_size,
            accepted_tokens,
            checkpoint_index: plan.checkpoint_index,
            host_device_bytes_verified: verify_host_device_bytes,
        })
    }

    /// StarFold terminal 的 host candidate 本应来自同一 prefix arena 的 production readback；
    /// 在 device publish 前再对选中 scratch 做一次逐字节核对，把这一点从 trait 假设升级为事实。
    fn verify_block_publish_scratch_matches(
        &mut self,
        ctx: &VulkanContext,
        expected: &[u8],
    ) -> Result<()> {
        if expected.len() as u64 != self.state_bytes {
            bail!("StarFold host/device checkpoint byte length 漂移");
        }
        (|| -> Result<()> {
            unsafe {
                ctx.device
                    .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
                ctx.device.begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                ctx.device.cmd_copy_buffer(
                    self.command,
                    self.block_publish_scratch.handle(),
                    self.block_publish_readback.handle(),
                    &[vk::BufferCopy::default().size(self.state_bytes)],
                );
                ctx.device.end_command_buffer(self.command)?;
                submit_and_wait(ctx, self.command, self.fence)?;
            }
            let observed = unsafe {
                std::slice::from_raw_parts(
                    self.block_publish_readback.mapped() as *const u8,
                    self.state_bytes as usize,
                )
            };
            if observed != expected {
                bail!("StarFold selected host/device checkpoint bytes 不同源");
            }
            Ok(())
        })()
    }

    /// 所有可能失败的 GPU copy/wait 已完成；该发布只做不可失败的 owner swap 与元数据切换。
    pub fn publish_prepared_block_commit(
        &mut self,
        prepared: WholeTokenPreparedBlockCommit,
    ) -> WholeTokenDeviceBlockCommitReceipt {
        assert_eq!(self.phase, CandidatePhase::BlockPrepared);
        assert_eq!(self.epoch, prepared.expected_epoch);
        assert!(matches!(prepared.block_size, 4 | 8));
        assert!((1..=prepared.block_size).contains(&prepared.accepted_tokens));
        assert_eq!(
            prepared.next_bank,
            self.active_bank ^ (prepared.accepted_tokens & 1)
        );
        assert_eq!(
            prepared.next_position,
            prepared.base_position + prepared.accepted_tokens as u32
        );
        std::mem::swap(
            &mut self.banks[prepared.next_bank],
            &mut self.block_publish_scratch,
        );
        self.active_bank = prepared.next_bank;
        self.epoch = prepared.next_epoch;
        self.last_committed_dirty = vec![0..self.state_bytes];
        self.repair_dirty.clear();
        self.candidate_dirty.clear();
        self.candidate_position = None;
        self.phase = CandidatePhase::Idle;
        WholeTokenDeviceBlockCommitReceipt {
            epoch: self.epoch,
            active_bank: self.active_bank,
            position: prepared.next_position,
            accepted_tokens: prepared.accepted_tokens,
            checkpoint_index: prepared.checkpoint_index,
            host_device_bytes_verified: prepared.host_device_bytes_verified,
        }
    }

    pub fn rollback_prepared_block_commit(
        &mut self,
        prepared: WholeTokenPreparedBlockCommit,
    ) -> Result<()> {
        if self.phase != CandidatePhase::BlockPrepared
            || self.epoch != prepared.expected_epoch
            || self.active_bank ^ (prepared.accepted_tokens & 1) != prepared.next_bank
        {
            bail!("prepared block rollback 身份漂移");
        }
        self.phase = CandidatePhase::Idle;
        Ok(())
    }

    /// 兼容旧调用者；新 whole-token orchestrator 使用 prepare/publish 两阶段。
    pub fn commit_candidate(&mut self, expected_epoch: u64) -> Result<u64> {
        let prepared = self.prepare_candidate_commit(expected_epoch)?;
        Ok(self.publish_prepared_commit(prepared).epoch)
    }

    /// 丢弃 inactive bank；下一候选会重新从 committed bank 复制。
    pub fn rollback_candidate(&mut self, ctx: &VulkanContext) -> Result<()> {
        match self.phase {
            CandidatePhase::Idle => bail!("没有可回滚的 whole-token candidate"),
            CandidatePhase::Recording => unsafe {
                ctx.device.reset_command_pool(
                    self.command_pool,
                    vk::CommandPoolResetFlags::RELEASE_RESOURCES,
                )?;
            },
            CandidatePhase::InFlight => {
                bail!(
                    "in-flight candidate 必须先由外部后端 drain，再使用 rollback_external_candidate"
                )
            }
            CandidatePhase::Ready | CandidatePhase::Failed => {}
            CandidatePhase::BlockPrepared => {
                bail!("block publish 已prepared；使用 rollback_prepared_block_commit")
            }
        }
        accumulate_repair_ranges(&mut self.repair_dirty, &mut self.candidate_dirty);
        self.candidate_position = None;
        self.phase = CandidatePhase::Idle;
        Ok(())
    }

    /// 丢弃已经由外部分段后端 drain 的 inactive bank。
    ///
    /// 调用方必须保证所有引用 bootstrap command、candidate bank、sticky status 的 transfer/
    /// compute submit 均已完成。该前置条件由 `Position0LayerBackend::abort_candidate` 或成功的
    /// token 末尾 wait 提供；这里不会产生第二次 host wait。
    pub fn rollback_external_candidate(&mut self, ctx: &VulkanContext) -> Result<()> {
        match self.phase {
            CandidatePhase::Idle => bail!("没有可回滚的 whole-token external candidate"),
            CandidatePhase::Recording | CandidatePhase::InFlight => unsafe {
                ctx.device.reset_command_pool(
                    self.command_pool,
                    vk::CommandPoolResetFlags::RELEASE_RESOURCES,
                )?;
            },
            CandidatePhase::Ready | CandidatePhase::Failed => {}
            CandidatePhase::BlockPrepared => {
                bail!("block publish 已prepared；使用 rollback_prepared_block_commit")
            }
        }
        accumulate_repair_ranges(&mut self.repair_dirty, &mut self.candidate_dirty);
        self.candidate_position = None;
        self.phase = CandidatePhase::Idle;
        Ok(())
    }

    /// 只用于数值门/检查点导出；热路径不得每 token 回读完整状态。
    pub fn read_active_for_audit(&mut self, ctx: &VulkanContext) -> Result<Vec<u8>> {
        if self.phase != CandidatePhase::Idle {
            bail!("candidate 存在时禁止审计回读 active bank");
        }
        let readback = GpuBuffer::new(
            ctx,
            self.state_bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?;
        unsafe {
            ctx.device.reset_command_pool(
                self.command_pool,
                vk::CommandPoolResetFlags::RELEASE_RESOURCES,
            )?;
            ctx.device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            ctx.device.cmd_copy_buffer(
                self.command,
                self.banks[self.active_bank].handle(),
                readback.handle(),
                &[vk::BufferCopy::default().size(self.state_bytes)],
            );
            ctx.device.end_command_buffer(self.command)?;
            submit_and_wait(ctx, self.command, self.fence)?;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(readback.mapped() as *const u8, self.state_bytes as usize)
                .to_vec()
        };
        readback.destroy(ctx);
        Ok(bytes)
    }

    pub fn destroy(self, ctx: &VulkanContext) -> Result<()> {
        if matches!(
            self.phase,
            CandidatePhase::Recording | CandidatePhase::InFlight
        ) {
            bail!("recording/in-flight candidate 存在时禁止销毁 whole-token device state");
        }
        unsafe {
            ctx.device.destroy_fence(self.fence, None);
            ctx.device.destroy_command_pool(self.command_pool, None);
        }
        self.sticky_status.destroy(ctx);
        self.block_publish_readback.destroy(ctx);
        self.block_publish_scratch.destroy(ctx);
        self.banks[1].destroy(ctx);
        self.banks[0].destroy(ctx);
        Ok(())
    }
}

fn validate_candidate_position(position: u32) -> Result<()> {
    if position > 2051 {
        bail!(
            "whole-token device state 当前闭合到首个ratio4分页续页position2051；position{position} fail-closed"
        );
    }
    Ok(())
}

fn build_block_publish_plan(
    active_bank: usize,
    base_epoch: u64,
    state_bytes: u64,
    source: WholeTokenBlockPrefixSource,
    accepted_tokens: usize,
    selected_host_checkpoint: &DecoderStateV1,
) -> Result<WholeTokenBlockPublishPlan> {
    if source.checkpoint_state_bytes != state_bytes
        || selected_host_checkpoint.native_arena.bytes().len() as u64 != state_bytes
        || source.checkpoint_arena == vk::Buffer::null()
        || source.checkpoint_stride_bytes < state_bytes
        || (source.ready_timeline == vk::Semaphore::null()) != (source.ready_timeline_value == 0)
    {
        bail!("block prefix publish 的 K/state/accepted identity 非法");
    }
    selected_host_checkpoint
        .validate()
        .map_err(|error| anyhow::anyhow!("selected host checkpoint 非法: {error}"))?;
    let (next_epoch, next_bank, next_position, checkpoint_index) = block_prefix_identity(
        active_bank,
        base_epoch,
        source.base_position,
        source.block_size,
        accepted_tokens,
    )?;
    if selected_host_checkpoint.commit_epoch != next_epoch
        || selected_host_checkpoint.position != next_position
        || usize::from(selected_host_checkpoint.active_fixed_bank) != next_bank
    {
        bail!("selected host/device prefix checkpoint 的 epoch/position/bank 不同源");
    }
    let checkpoint_offset = source
        .checkpoint_stride_bytes
        .checked_mul(checkpoint_index as u64)
        .and_then(|relative| source.checkpoint_arena_offset.checked_add(relative))
        .ok_or_else(|| anyhow::anyhow!("selected device checkpoint offset overflow"))?;
    let arena_end = source
        .checkpoint_arena_offset
        .checked_add(source.checkpoint_arena_bytes)
        .ok_or_else(|| anyhow::anyhow!("device checkpoint arena end overflow"))?;
    let checkpoint_end = checkpoint_offset
        .checked_add(state_bytes)
        .ok_or_else(|| anyhow::anyhow!("selected device checkpoint end overflow"))?;
    if checkpoint_end > arena_end {
        bail!("selected device checkpoint 越出 future arena");
    }
    Ok(WholeTokenBlockPublishPlan {
        next_epoch,
        next_bank,
        next_position,
        checkpoint_index,
        checkpoint_offset,
    })
}

fn block_prefix_identity(
    active_bank: usize,
    base_epoch: u64,
    base_position: u32,
    block_size: usize,
    accepted_tokens: usize,
) -> Result<(u64, usize, u32, usize)> {
    if active_bank >= 2
        || !matches!(block_size, 4 | 8)
        || !(1..=block_size).contains(&accepted_tokens)
    {
        bail!("block prefix identity 的 active bank/K/accepted 非法");
    }
    let next_epoch = base_epoch
        .checked_add(accepted_tokens as u64)
        .ok_or_else(|| anyhow::anyhow!("block prefix epoch overflow"))?;
    let next_position = base_position
        .checked_add(accepted_tokens as u32)
        .ok_or_else(|| anyhow::anyhow!("block prefix position overflow"))?;
    Ok((
        next_epoch,
        active_bank ^ (accepted_tokens & 1),
        next_position,
        accepted_tokens - 1,
    ))
}

fn coalesce_ranges(mut ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

fn accumulate_repair_ranges(
    repair_dirty: &mut Vec<Range<u64>>,
    candidate_dirty: &mut Vec<Range<u64>>,
) {
    repair_dirty.append(candidate_dirty);
    *repair_dirty = coalesce_ranges(std::mem::take(repair_dirty));
}

unsafe fn submit_and_wait(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    fence: vk::Fence,
) -> Result<()> {
    ctx.device.reset_fences(&[fence])?;
    let commands = [command];
    ctx.device.queue_submit(
        ctx.q_graphics,
        &[vk::SubmitInfo::default().command_buffers(&commands)],
        fence,
    )?;
    ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    Ok(())
}

unsafe fn submit_and_wait_on_timeline(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    wait_timeline: vk::Semaphore,
    wait_value: u64,
) -> Result<()> {
    if wait_timeline == vk::Semaphore::null() || wait_value == 0 {
        bail!("block prefix copy 要求非空 producer timeline/value");
    }
    ctx.device.reset_fences(&[fence])?;
    let wait_values = [wait_value];
    let mut timeline =
        vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
    let wait_semaphores = [wait_timeline];
    let wait_stages = [vk::PipelineStageFlags::TRANSFER];
    let commands = [command];
    let submit = vk::SubmitInfo::default()
        .push_next(&mut timeline)
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_stages)
        .command_buffers(&commands);
    ctx.device.queue_submit(ctx.q_graphics, &[submit], fence)?;
    ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        accumulate_repair_ranges, block_prefix_identity, coalesce_ranges,
        validate_candidate_position,
    };

    #[test]
    fn k4_k8_prefix_identity_advances_epoch_position_and_bank_by_accepted_parity() {
        for block_size in [4, 8] {
            for accepted in 1..=block_size {
                let (epoch, bank, position, checkpoint) =
                    block_prefix_identity(0, 20, 100, block_size, accepted).unwrap();
                assert_eq!(epoch, 20 + accepted as u64);
                assert_eq!(bank, accepted & 1);
                assert_eq!(position, 100 + accepted as u32);
                assert_eq!(checkpoint, accepted - 1);
            }
        }
        assert!(block_prefix_identity(0, 0, 0, 4, 0).is_err());
        assert!(block_prefix_identity(0, 0, 0, 4, 5).is_err());
        assert!(block_prefix_identity(0, u64::MAX, 0, 4, 1).is_err());
        assert!(block_prefix_identity(0, 0, u32::MAX, 4, 1).is_err());
    }

    #[test]
    fn device_candidate_allows_first_paged_ratio4_append_and_rejects_position2052() {
        validate_candidate_position(0).unwrap();
        validate_candidate_position(1).unwrap();
        validate_candidate_position(2).unwrap();
        validate_candidate_position(3).unwrap();
        validate_candidate_position(4).unwrap();
        validate_candidate_position(15).unwrap();
        validate_candidate_position(126).unwrap();
        validate_candidate_position(127).unwrap();
        validate_candidate_position(128).unwrap();
        validate_candidate_position(129).unwrap();
        validate_candidate_position(254).unwrap();
        validate_candidate_position(255).unwrap();
        validate_candidate_position(256).unwrap();
        validate_candidate_position(2047).unwrap();
        validate_candidate_position(2050).unwrap();
        validate_candidate_position(2051).unwrap();
        assert!(validate_candidate_position(2052).is_err());
    }

    #[test]
    fn repair_ranges_survive_a_second_failed_candidate() {
        let mut repair = vec![64..128, 256..320];
        let mut candidate = vec![96..160, 512..576];
        accumulate_repair_ranges(&mut repair, &mut candidate);
        assert_eq!(repair, vec![64..160, 256..320, 512..576]);
        assert!(candidate.is_empty());

        let mut no_new_writes = Vec::new();
        accumulate_repair_ranges(&mut repair, &mut no_new_writes);
        assert_eq!(repair, vec![64..160, 256..320, 512..576]);
    }

    #[test]
    fn adjacent_dirty_ranges_coalesce_for_bank_repair() {
        assert_eq!(
            coalesce_ranges(vec![16..32, 0..8, 8..16, 48..64]),
            vec![0..32, 48..64]
        );
    }
}
