//! StarFold verified microtile/constellation immutable payload 的双 staging Vulkan transfer
//! 录制器。
//!
//! 本模块拥有两个常驻 HOST_VISIBLE staging buffer、一个 transfer command pool、两个
//! command buffer 与对应 fence。显式 block-epoch 流水允许当前 window 在 GPU compute 时，
//! host 把下一份已通过 proof/SHA 的 payload 写入另一 staging 并录制 command；只有
//! Prepared 在同一 epoch 下 commit 后才会暴露给 `S14StarfoldRuntime::submit_upload`。
//! 本模块本身始终不 queue-submit，也不分配第三个 staging/window。

use crate::{
    s14_starfold_routed_executor::constellation_packet::{
        S14StarfoldConstellationPacket, S14StarfoldResidentWindowKey,
    },
    s14_starfold_runtime::{S14StarfoldUploadTicket, S14StarfoldVerifiedMicrotile},
    s14_starfold_vulkan_windows::{
        S14StarfoldBufferBarrier, S14StarfoldUploadRecording,
        S14StarfoldUploadTicket as VulkanUploadTicket, S14StarfoldWindowId,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

static NEXT_TRANSFER_EXECUTOR_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransferSlotState {
    Idle,
    /// immutable proof/SHA lease 已写入 staging 且 command buffer 已结束录制，但尚未授权
    /// runtime queue submit。该状态没有可等待 fence；只能 commit 或 cancel。
    Prepared {
        recording_serial: u64,
        block_epoch: u64,
        window_generation: u64,
    },
    /// fence 尚未完成时既可能在等待 runtime submit，也可能已经 in-flight。
    Armed {
        recording_serial: u64,
        /// `None` 只保留给旧的同步 record API；显式流水一律为 `Some(block_epoch)`。
        block_epoch: Option<u64>,
        window_generation: u64,
    },
}

struct TransferSlot {
    staging: GpuBuffer,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    state: TransferSlotState,
}

/// 交给唯一接入点 `S14StarfoldRuntime::submit_upload` 的录制结果。
///
/// 调用方必须把这里的 command buffer 与 fence 原样提交；不能为同一录制替换 fence，
/// 否则执行器无法证明 staging/command buffer 已经可以复用。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldRecordedTransfer {
    executor_id: u64,
    block_epoch: Option<u64>,
    window: S14StarfoldWindowId,
    window_generation: u64,
    recording_serial: u64,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    staging_buffer: vk::Buffer,
    destination_buffer: vk::Buffer,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
}

impl S14StarfoldRecordedTransfer {
    /// `Some` 表示该录制由显式 block-epoch 流水产生；`None` 是旧同步入口。
    pub const fn block_epoch(self) -> Option<u64> {
        self.block_epoch
    }

    pub const fn command_buffer(self) -> vk::CommandBuffer {
        self.command_buffer
    }

    pub const fn fence(self) -> vk::Fence {
        self.fence
    }

    pub const fn window(self) -> S14StarfoldWindowId {
        self.window
    }

    pub const fn window_generation(self) -> u64 {
        self.window_generation
    }

    pub const fn byte_len(self) -> vk::DeviceSize {
        self.byte_len
    }

    pub const fn staging_buffer(self) -> vk::Buffer {
        self.staging_buffer
    }

    pub const fn destination_buffer(self) -> vk::Buffer {
        self.destination_buffer
    }

    pub const fn destination_offset(self) -> vk::DeviceSize {
        self.destination_offset
    }
}

/// 已完成 proof/SHA 验证、host staging 写入和 command 录制，但还不能提交到 device 的
/// 有界流水所有权。
///
/// 对象额外强持有 verified proof lease；唯一 upload ticket 仍由调用方持有，因此正常背压
/// 不会吞掉窗口 reservation。调用方必须在同一 `block_epoch` 内二选一：
///
/// - [`S14StarfoldTransferExecutor::commit_prepared_upload`]；
/// - [`S14StarfoldTransferExecutor::cancel_prepared_upload`]。
///
/// 直接丢弃会让对应槽 fail-closed 地停留在 Prepared，防止未证明的跨 epoch 复用。
#[derive(Debug)]
pub struct S14StarfoldPreparedUpload {
    block_epoch: u64,
    proof: Arc<S14StarfoldVerifiedMicrotile>,
    recorded: S14StarfoldRecordedTransfer,
}

impl S14StarfoldPreparedUpload {
    pub const fn block_epoch(&self) -> u64 {
        self.block_epoch
    }

    pub const fn window(&self) -> S14StarfoldWindowId {
        self.recorded.window()
    }

    pub const fn window_generation(&self) -> u64 {
        self.recorded.window_generation()
    }

    pub const fn byte_len(&self) -> vk::DeviceSize {
        self.recorded.byte_len()
    }

    pub fn proof(&self) -> &Arc<S14StarfoldVerifiedMicrotile> {
        &self.proof
    }
}

/// Prepared 状态已经在同一 block epoch 下获准提交后的唯一所有权。
///
/// `into_recorded` 后应立即把调用方持有的原 ticket、command buffer 和 fence 原样交给
/// `S14StarfoldRuntime::submit_upload`；queue submit 失败时仍须调用
/// [`S14StarfoldTransferExecutor::abandon_unsubmitted`] 并取消窗口 reservation。
#[derive(Debug)]
pub struct S14StarfoldCommittedUpload {
    block_epoch: u64,
    proof: Arc<S14StarfoldVerifiedMicrotile>,
    recorded: S14StarfoldRecordedTransfer,
}

impl S14StarfoldCommittedUpload {
    pub const fn block_epoch(&self) -> u64 {
        self.block_epoch
    }

    pub const fn recorded(&self) -> S14StarfoldRecordedTransfer {
        self.recorded
    }

    pub fn proof(&self) -> &Arc<S14StarfoldVerifiedMicrotile> {
        &self.proof
    }

    pub fn into_recorded(self) -> S14StarfoldRecordedTransfer {
        self.recorded
    }
}

/// 星座 packet 已写入现有 A/B staging 并完成 command 录制，但尚未获准
/// queue submit 的有界所有权。
///
/// 本对象强持有 immutable packet，保证 host payload 与多专家 proof lease 在
/// Prepared 期间不会失活；物理 staging/window 仍是 executor 原有的 A/B 两个。
#[derive(Debug)]
pub struct S14StarfoldPreparedConstellationUpload {
    block_epoch: u64,
    packet: Arc<S14StarfoldConstellationPacket>,
    recorded: S14StarfoldRecordedTransfer,
}

impl S14StarfoldPreparedConstellationUpload {
    pub const fn block_epoch(&self) -> u64 {
        self.block_epoch
    }

    pub const fn window(&self) -> S14StarfoldWindowId {
        self.recorded.window()
    }

    pub const fn window_generation(&self) -> u64 {
        self.recorded.window_generation()
    }

    pub const fn byte_len(&self) -> vk::DeviceSize {
        self.recorded.byte_len()
    }

    pub fn packet(&self) -> &Arc<S14StarfoldConstellationPacket> {
        &self.packet
    }
}

/// 同一 block epoch 下已从 Prepared 推进到 Armed 的星座上传所有权。
#[derive(Debug)]
pub struct S14StarfoldCommittedConstellationUpload {
    block_epoch: u64,
    packet: Arc<S14StarfoldConstellationPacket>,
    recorded: S14StarfoldRecordedTransfer,
}

impl S14StarfoldCommittedConstellationUpload {
    pub const fn block_epoch(&self) -> u64 {
        self.block_epoch
    }

    pub const fn recorded(&self) -> S14StarfoldRecordedTransfer {
        self.recorded
    }

    pub fn packet(&self) -> &Arc<S14StarfoldConstellationPacket> {
        &self.packet
    }

    /// 提交给 runtime 时一并转移 packet lease，使 queue submit 成功后可直接
    /// 构造 ready packet，不需要二次 payload owner。
    pub fn into_parts(
        self,
    ) -> (
        S14StarfoldRecordedTransfer,
        Arc<S14StarfoldConstellationPacket>,
    ) {
        (self.recorded, self.packet)
    }
}

/// 常驻双 staging transfer 资源。显式流水热路径不等待 fence：目标槽位尚未完成时直接返回
/// 背压，由上层继续当前 compute 或稍后重试。旧 `record_verified_upload` 为兼容入口，仍只
/// 等待它即将复用的单个 transfer fence。
pub struct S14StarfoldTransferExecutor {
    executor_id: u64,
    context: Arc<VulkanContext>,
    staging_bytes: vk::DeviceSize,
    command_pool: vk::CommandPool,
    slots: [TransferSlot; 2],
    next_recording_serial: u64,
    /// 显式流水的唯一活动 block。存在时禁止旧同步入口混入 epoch-less recording。
    active_block_epoch: Option<u64>,
    destroyed: bool,
}

impl S14StarfoldTransferExecutor {
    /// 创建两个常驻 staging、一个 transfer-family command pool、两个 command buffer/fence。
    /// 本函数不提交 command，也不访问 GPU 数值。
    pub fn new(context: Arc<VulkanContext>, staging_bytes: vk::DeviceSize) -> Result<Self> {
        if staging_bytes == 0 {
            bail!("S14 StarFold transfer staging_bytes 不能为0");
        }
        if context.qf_transfer == vk::QUEUE_FAMILY_IGNORED {
            bail!("S14 StarFold transfer queue family 非法");
        }

        let staging_a = GpuBuffer::new_staging(&context, staging_bytes)
            .context("创建 S14 StarFold staging A")?;
        let staging_b = match GpuBuffer::new_staging(&context, staging_bytes)
            .context("创建 S14 StarFold staging B")
        {
            Ok(buffer) => buffer,
            Err(error) => {
                staging_a.destroy(&context);
                return Err(error);
            }
        };

        let command_pool = match unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.qf_transfer)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .context("创建 S14 StarFold transfer command pool")
        {
            Ok(pool) => pool,
            Err(error) => {
                staging_b.destroy(&context);
                staging_a.destroy(&context);
                return Err(error);
            }
        };

        let command_buffers = match unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(2),
            )
        }
        .context("分配 S14 StarFold transfer command buffers")
        {
            Ok(buffers) if buffers.len() == 2 => buffers,
            Ok(_) => {
                unsafe {
                    context.device.destroy_command_pool(command_pool, None);
                }
                staging_b.destroy(&context);
                staging_a.destroy(&context);
                bail!("S14 StarFold transfer command buffer 数量漂移");
            }
            Err(error) => {
                unsafe {
                    context.device.destroy_command_pool(command_pool, None);
                }
                staging_b.destroy(&context);
                staging_a.destroy(&context);
                return Err(error);
            }
        };

        let fence_a = match unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("创建 S14 StarFold transfer fence A")
        {
            Ok(fence) => fence,
            Err(error) => {
                unsafe {
                    context.device.destroy_command_pool(command_pool, None);
                }
                staging_b.destroy(&context);
                staging_a.destroy(&context);
                return Err(error);
            }
        };
        let fence_b = match unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("创建 S14 StarFold transfer fence B")
        {
            Ok(fence) => fence,
            Err(error) => {
                unsafe {
                    context.device.destroy_fence(fence_a, None);
                    context.device.destroy_command_pool(command_pool, None);
                }
                staging_b.destroy(&context);
                staging_a.destroy(&context);
                return Err(error);
            }
        };

        let executor_id = NEXT_TRANSFER_EXECUTOR_ID.fetch_add(1, Ordering::Relaxed);
        if executor_id == u64::MAX {
            unsafe {
                context.device.destroy_fence(fence_b, None);
                context.device.destroy_fence(fence_a, None);
                context.device.destroy_command_pool(command_pool, None);
            }
            staging_b.destroy(&context);
            staging_a.destroy(&context);
            bail!("S14 StarFold transfer executor id 溢出");
        }

        Ok(Self {
            executor_id,
            context,
            staging_bytes,
            command_pool,
            slots: [
                TransferSlot {
                    staging: staging_a,
                    command_buffer: command_buffers[0],
                    fence: fence_a,
                    state: TransferSlotState::Idle,
                },
                TransferSlot {
                    staging: staging_b,
                    command_buffer: command_buffers[1],
                    fence: fence_b,
                    state: TransferSlotState::Idle,
                },
            ],
            next_recording_serial: 1,
            active_block_epoch: None,
            destroyed: false,
        })
    }

    pub const fn staging_bytes(&self) -> vk::DeviceSize {
        self.staging_bytes
    }

    pub const fn active_block_epoch(&self) -> Option<u64> {
        self.active_block_epoch
    }

    /// 开启一个显式计算—传输流水 epoch。
    ///
    /// epoch 切换只允许发生在两个 transfer 槽都 idle 时；同一 epoch 的重复调用是幂等的。
    /// 该门使当前 microtile 的 device compute 与下一 microtile 的 host prepare 可以重叠，
    /// 同时禁止下一 block 的 recording 混入尚未退休的 A/B 槽。
    pub fn begin_block_epoch(&mut self, block_epoch: u64) -> Result<()> {
        if self.destroyed {
            bail!("S14 StarFold transfer executor 已销毁");
        }
        if self.active_block_epoch == Some(block_epoch) {
            return Ok(());
        }
        if let Some(active) = self.active_block_epoch {
            bail!(
                "S14 StarFold transfer block epoch 尚未结束: active={}, requested={}",
                active,
                block_epoch
            );
        }
        self.refresh_slot(0)?;
        self.refresh_slot(1)?;
        if self
            .slots
            .iter()
            .any(|slot| slot.state != TransferSlotState::Idle)
        {
            bail!("S14 StarFold transfer 仍有旧录制或 in-flight command，不能开始新 block epoch");
        }
        self.active_block_epoch = Some(block_epoch);
        Ok(())
    }

    /// 非阻塞准备一个已验证 microtile：写入目标 window 对应的 host staging，并完成 Vulkan
    /// command 录制，但不向 runtime 暴露可提交 command/fence。
    ///
    /// ticket 只能由 runtime 的 verified proof/SHA 路径构造；本入口再次核对 proof key、长度和
    /// recording range。目标槽不空闲时直接返回背压，不等待 fence，因此不会阻塞当前 GPU
    /// compute；有界容量始终为 A/B 两个槽。
    pub fn try_prepare_verified_upload(
        &mut self,
        block_epoch: u64,
        ticket: &S14StarfoldUploadTicket,
    ) -> Result<S14StarfoldPreparedUpload> {
        if !self.can_prepare_verified_upload(block_epoch, ticket)? {
            bail!(
                "S14 StarFold transfer {} 预取槽忙；保持当前 compute 并稍后重试",
                ticket.ticket().window()
            );
        }

        let recorded = self.record_into_idle_slot(ticket, Some(block_epoch), false)?;
        Ok(S14StarfoldPreparedUpload {
            block_epoch,
            proof: Arc::clone(ticket.proof()),
            recorded,
        })
    }

    /// 有界背压版本：只等待即将复用的同一 A/B transfer 槽，不等待另一槽或整 device。
    /// 适用于 production 顺序流；前一窗口的 compute 仍可与本次 host prepare 并发。
    pub fn prepare_verified_upload(
        &mut self,
        block_epoch: u64,
        ticket: &S14StarfoldUploadTicket,
    ) -> Result<S14StarfoldPreparedUpload> {
        self.require_active_epoch(block_epoch)?;
        validate_verified_ticket(ticket)?;
        let index = ticket.ticket().window().index();
        self.wait_slot(index)?;
        let recorded = self.record_into_idle_slot(ticket, Some(block_epoch), false)?;
        Ok(S14StarfoldPreparedUpload {
            block_epoch,
            proof: Arc::clone(ticket.proof()),
            recorded,
        })
    }

    /// 查询目标 A/B 槽是否可立即执行 host prepare。该调用只刷新目标 transfer fence，
    /// 不等待、不写 staging、不改变 ticket/slot 所有权。
    pub fn can_prepare_verified_upload(
        &mut self,
        block_epoch: u64,
        ticket: &S14StarfoldUploadTicket,
    ) -> Result<bool> {
        self.require_active_epoch(block_epoch)?;
        validate_verified_ticket(ticket)?;
        let index = ticket.ticket().window().index();
        self.refresh_slot(index)?;
        Ok(self.slots[index].state == TransferSlotState::Idle)
    }

    /// 非阻塞查询星座 packet 目标 A/B 槽是否可立即录制。这与 microtile
    /// 共用同一 active epoch 和同一组 transfer fence。
    pub fn can_prepare_constellation_upload(
        &mut self,
        block_epoch: u64,
        ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
        recording: S14StarfoldUploadRecording,
        packet: &Arc<S14StarfoldConstellationPacket>,
    ) -> Result<bool> {
        self.require_active_epoch(block_epoch)?;
        validate_constellation_ticket(ticket, recording, packet)?;
        let index = ticket.window().index();
        self.refresh_slot(index)?;
        Ok(self.slots[index].state == TransferSlotState::Idle)
    }

    /// 非阻塞准备一个 immutable constellation packet。槽忙时只返回背压，不分配
    /// 第三个 staging，也不建立第二套 window owner。
    pub fn try_prepare_constellation_upload(
        &mut self,
        block_epoch: u64,
        ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
        recording: S14StarfoldUploadRecording,
        packet: &Arc<S14StarfoldConstellationPacket>,
    ) -> Result<S14StarfoldPreparedConstellationUpload> {
        if !self.can_prepare_constellation_upload(block_epoch, ticket, recording, packet)? {
            bail!(
                "S14 StarFold constellation transfer {} 预取槽忙；保持当前 compute 并稍后重试",
                ticket.window()
            );
        }
        let recorded = self.record_immutable_payload_into_idle_slot(
            ticket,
            recording,
            packet.payload(),
            Some(block_epoch),
            false,
        )?;
        Ok(S14StarfoldPreparedConstellationUpload {
            block_epoch,
            packet: Arc::clone(packet),
            recorded,
        })
    }

    /// 有界背压地准备星座 packet：只等待 ticket 指定的同一 A/B 槽。
    pub fn prepare_constellation_upload(
        &mut self,
        block_epoch: u64,
        ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
        recording: S14StarfoldUploadRecording,
        packet: &Arc<S14StarfoldConstellationPacket>,
    ) -> Result<S14StarfoldPreparedConstellationUpload> {
        self.require_active_epoch(block_epoch)?;
        validate_constellation_ticket(ticket, recording, packet)?;
        let index = ticket.window().index();
        self.wait_slot(index)?;
        let recorded = self.record_immutable_payload_into_idle_slot(
            ticket,
            recording,
            packet.payload(),
            Some(block_epoch),
            false,
        )?;
        Ok(S14StarfoldPreparedConstellationUpload {
            block_epoch,
            packet: Arc::clone(packet),
            recorded,
        })
    }

    /// 把星座 Prepared 槽在同一 epoch 下原子推进为 Armed。本函数仍不
    /// queue-submit；返回值继续强持有 packet lease。
    pub fn commit_prepared_constellation_upload(
        &mut self,
        ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
        recording: S14StarfoldUploadRecording,
        packet: &Arc<S14StarfoldConstellationPacket>,
        prepared: S14StarfoldPreparedConstellationUpload,
    ) -> Result<S14StarfoldCommittedConstellationUpload> {
        let block_epoch = prepared.block_epoch;
        if let Err(error) = self.require_active_epoch(block_epoch) {
            let cleanup =
                self.abandon_prepared_recording(prepared.recorded, block_epoch, "constellation");
            return Err(anyhow::anyhow!(
                "{error:#}; constellation prepared owner cleanup={cleanup:?}"
            ));
        }
        if prepared.recorded.block_epoch() != Some(block_epoch) {
            let cleanup =
                self.abandon_prepared_recording(prepared.recorded, block_epoch, "constellation");
            bail!("S14 StarFold constellation prepared recording epoch 漂移; cleanup={cleanup:?}");
        }
        let index = match self.validate_prepared_constellation(&prepared, ticket, recording, packet)
        {
            Ok(index) => index,
            Err(error) => {
                let cleanup = self.abandon_prepared_recording(
                    prepared.recorded,
                    block_epoch,
                    "constellation",
                );
                return Err(anyhow::anyhow!(
                    "{error:#}; constellation prepared owner cleanup={cleanup:?}"
                ));
            }
        };
        self.slots[index].state = TransferSlotState::Armed {
            recording_serial: prepared.recorded.recording_serial,
            block_epoch: Some(block_epoch),
            window_generation: prepared.recorded.window_generation,
        };
        Ok(S14StarfoldCommittedConstellationUpload {
            block_epoch,
            packet: prepared.packet,
            recorded: prepared.recorded,
        })
    }

    /// 放弃尚未提交的星座 Prepared 槽。运行时仍须用自己持有的 ticket
    /// 同步取消 window reservation。
    pub fn cancel_prepared_constellation_upload(
        &mut self,
        ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
        recording: S14StarfoldUploadRecording,
        packet: &Arc<S14StarfoldConstellationPacket>,
        prepared: S14StarfoldPreparedConstellationUpload,
    ) -> Result<()> {
        let block_epoch = prepared.block_epoch;
        self.require_active_epoch(block_epoch)?;
        let index = self.validate_prepared_constellation(&prepared, ticket, recording, packet)?;
        unsafe {
            self.context.device.reset_command_buffer(
                self.slots[index].command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .context("取消 S14 StarFold constellation prepared command buffer")?;
        self.slots[index].state = TransferSlotState::Idle;
        Ok(())
    }

    /// 把 Prepared 槽在同一 block epoch 下原子推进为 Armed，随后才允许 runtime device
    /// submit。此函数本身仍不 queue-submit。
    pub fn commit_prepared_upload(
        &mut self,
        ticket: &S14StarfoldUploadTicket,
        prepared: S14StarfoldPreparedUpload,
    ) -> Result<S14StarfoldCommittedUpload> {
        let block_epoch = prepared.block_epoch;
        if let Err(error) = self.require_active_epoch(block_epoch) {
            let cleanup = self.abandon_prepared_owned(&prepared);
            return Err(anyhow::anyhow!(
                "{error:#}; prepared owner cleanup={cleanup:?}"
            ));
        }
        if prepared.recorded.block_epoch() != Some(block_epoch) {
            let cleanup = self.abandon_prepared_owned(&prepared);
            bail!("S14 StarFold prepared upload 的 recording epoch 漂移; cleanup={cleanup:?}");
        }
        let index = match self.validate_prepared(&prepared, ticket) {
            Ok(index) => index,
            Err(error) => {
                let cleanup = self.abandon_prepared_owned(&prepared);
                return Err(anyhow::anyhow!(
                    "{error:#}; prepared owner cleanup={cleanup:?}"
                ));
            }
        };
        self.slots[index].state = TransferSlotState::Armed {
            recording_serial: prepared.recorded.recording_serial,
            block_epoch: Some(block_epoch),
            window_generation: prepared.recorded.window_generation,
        };
        Ok(S14StarfoldCommittedUpload {
            block_epoch,
            proof: prepared.proof,
            recorded: prepared.recorded,
        })
    }

    fn abandon_prepared_owned(&mut self, prepared: &S14StarfoldPreparedUpload) -> Result<()> {
        self.abandon_prepared_recording(prepared.recorded, prepared.block_epoch, "microtile")
    }

    fn abandon_prepared_recording(
        &mut self,
        recorded: S14StarfoldRecordedTransfer,
        block_epoch: u64,
        payload_kind: &str,
    ) -> Result<()> {
        if recorded.executor_id != self.executor_id {
            bail!("S14 StarFold {payload_kind} prepared cleanup 来自其他 executor");
        }
        let index = recorded.window.index();
        let expected = TransferSlotState::Prepared {
            recording_serial: recorded.recording_serial,
            block_epoch,
            window_generation: recorded.window_generation,
        };
        if self.slots[index].state != expected {
            bail!("S14 StarFold {payload_kind} prepared cleanup slot identity 漂移");
        }
        unsafe {
            self.context.device.reset_command_buffer(
                self.slots[index].command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .with_context(|| format!("回收 S14 StarFold {payload_kind} Prepared command buffer"))?;
        self.slots[index].state = TransferSlotState::Idle;
        Ok(())
    }

    /// 放弃尚未提交的 Prepared 槽。成功后调用方必须用仍由自己持有的 ticket 同步取消
    /// runtime window reservation；本 executor 不拥有 Vulkan window 状态机。
    pub fn cancel_prepared_upload(
        &mut self,
        ticket: &S14StarfoldUploadTicket,
        prepared: S14StarfoldPreparedUpload,
    ) -> Result<()> {
        let block_epoch = prepared.block_epoch;
        self.require_active_epoch(block_epoch)?;
        let index = self.validate_prepared(&prepared, ticket)?;
        unsafe {
            self.context.device.reset_command_buffer(
                self.slots[index].command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .context("取消 S14 StarFold prepared command buffer")?;
        self.slots[index].state = TransferSlotState::Idle;
        Ok(())
    }

    /// 结束显式 block epoch。该操作非阻塞；任一 Prepared/Armed/in-flight 槽仍存在时拒绝
    /// 结束。调用方还必须在更高层证明该 block 的 compute receipt 已完成。
    pub fn finish_block_epoch(&mut self, block_epoch: u64) -> Result<()> {
        self.require_active_epoch(block_epoch)?;
        self.refresh_slot(0)?;
        self.refresh_slot(1)?;
        if self
            .slots
            .iter()
            .any(|slot| slot.state != TransferSlotState::Idle)
        {
            bail!("S14 StarFold transfer block epoch 仍有 prepared/armed/in-flight 槽");
        }
        self.active_block_epoch = None;
        Ok(())
    }

    /// projection/block 边界的唯一有界 drain：只等待两个 transfer fence，随后结束 epoch。
    /// 上层仍须独立持有并验证 compute completion；这里不调用 device-idle。
    pub fn drain_block_epoch(&mut self, block_epoch: u64) -> Result<()> {
        self.require_active_epoch(block_epoch)?;
        self.wait_slot(0)?;
        self.wait_slot(1)?;
        self.finish_block_epoch(block_epoch)
    }

    /// 把 ticket 中 SHA 已验证的 immutable mmap bytes 写入对应 A/B staging，并录制：
    /// acquire-from-compute（可选）→ host-write visibility → 精确 copy → release-to-compute。
    ///
    /// 返回值是接入 runtime submit 的完整 command/fence 对；本函数不 queue-submit。
    pub fn record_verified_upload(
        &mut self,
        ticket: &S14StarfoldUploadTicket,
    ) -> Result<S14StarfoldRecordedTransfer> {
        if self.destroyed {
            bail!("S14 StarFold transfer executor 已销毁");
        }
        if let Some(active) = self.active_block_epoch {
            bail!(
                "S14 StarFold 显式 block epoch {} 活动时禁止 epoch-less record API",
                active
            );
        }
        validate_verified_ticket(ticket)?;
        let index = ticket.ticket().window().index();
        self.wait_slot(index)?;
        self.record_into_idle_slot(ticket, None, true)
    }

    /// runtime submit 失败或 ticket 在 submit 前取消时，归还尚未提交的录制槽。
    ///
    /// # Safety
    /// `recorded.command_buffer()` 必须从未传给 queue submit，或外部已经证明 queue 不再引用它。
    pub unsafe fn abandon_unsubmitted(
        &mut self,
        recorded: S14StarfoldRecordedTransfer,
    ) -> Result<()> {
        let index = self.validate_recorded(recorded)?;
        unsafe {
            self.context.device.reset_command_buffer(
                self.slots[index].command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .context("放弃 S14 StarFold 未提交 command buffer")?;
        self.slots[index].state = TransferSlotState::Idle;
        Ok(())
    }

    /// 非阻塞清理：只有两个槽都 idle/已完成时才销毁资源。热路径无需调用。
    pub fn try_destroy(&mut self) -> Result<()> {
        if self.destroyed {
            return Ok(());
        }
        if let Some(active) = self.active_block_epoch {
            bail!(
                "S14 StarFold transfer block epoch {} 尚未显式结束，拒绝销毁 executor",
                active
            );
        }
        self.refresh_slot(0)?;
        self.refresh_slot(1)?;
        if self
            .slots
            .iter()
            .any(|slot| slot.state != TransferSlotState::Idle)
        {
            bail!("S14 StarFold transfer 仍有未提交或 in-flight command");
        }
        self.destroy_resources();
        Ok(())
    }

    fn require_active_epoch(&self, block_epoch: u64) -> Result<()> {
        if self.destroyed {
            bail!("S14 StarFold transfer executor 已销毁");
        }
        match self.active_block_epoch {
            Some(active) if active == block_epoch => Ok(()),
            Some(active) => bail!(
                "S14 StarFold transfer block epoch 漂移: active={}, requested={}",
                active,
                block_epoch
            ),
            None => bail!(
                "S14 StarFold transfer block epoch {} 尚未 begin",
                block_epoch
            ),
        }
    }

    fn record_into_idle_slot(
        &mut self,
        ticket: &S14StarfoldUploadTicket,
        block_epoch: Option<u64>,
        arm_immediately: bool,
    ) -> Result<S14StarfoldRecordedTransfer> {
        validate_verified_ticket(ticket)?;
        self.record_immutable_payload_into_idle_slot(
            ticket.ticket(),
            ticket.recording(),
            ticket.bytes(),
            block_epoch,
            arm_immediately,
        )
    }

    /// microtile proof 与 constellation packet 的唯一 host payload 录制内核。
    ///
    /// 调用方必须在进入前完成 payload key/proof 身份校验，并在返回的
    /// Prepared/Committed owner 中强持有对应 `Arc`。本函数只处理通用的
    /// immutable bytes→staging→copy command 状态转移。
    fn record_immutable_payload_into_idle_slot(
        &mut self,
        upload_ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
        recording: S14StarfoldUploadRecording,
        bytes: &[u8],
        block_epoch: Option<u64>,
        arm_immediately: bool,
    ) -> Result<S14StarfoldRecordedTransfer> {
        validate_recording(
            recording,
            upload_ticket.byte_len(),
            bytes.len(),
            self.staging_bytes,
            self.context.qf_transfer,
        )?;

        let index = upload_ticket.window().index();
        if self.slots[index].state != TransferSlotState::Idle {
            bail!(
                "S14 StarFold transfer {} staging/command buffer 不是 idle",
                upload_ticket.window()
            );
        }
        if !arm_immediately && block_epoch.is_none() {
            bail!("S14 StarFold Prepared transfer 必须绑定 block epoch");
        }

        let recording_serial = self.next_recording_serial;
        self.next_recording_serial = self
            .next_recording_serial
            .checked_add(1)
            .context("S14 StarFold transfer recording serial 溢出")?;
        let slot = &mut self.slots[index];
        if slot.staging.handle() == recording.buffer {
            bail!("S14 StarFold transfer staging 与 destination buffer 意外 alias");
        }

        // SAFETY: staging 由 GpuBuffer::new_staging 创建并保持映射；上面已证明长度不越界。
        unsafe {
            slot.staging.write_at(0, bytes);
        }
        unsafe {
            self.context
                .device
                .reset_command_buffer(slot.command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .context("reset S14 StarFold transfer command buffer")?;
        unsafe {
            self.context.device.begin_command_buffer(
                slot.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .context("begin S14 StarFold transfer command buffer")?;

        let record_result = unsafe {
            record_copy_commands(
                &self.context.device,
                slot.command_buffer,
                slot.staging.handle(),
                recording,
            );
            self.context.device.end_command_buffer(slot.command_buffer)
        };
        if let Err(error) = record_result {
            let _ = unsafe {
                self.context
                    .device
                    .reset_command_buffer(slot.command_buffer, vk::CommandBufferResetFlags::empty())
            };
            return Err(error).context("end S14 StarFold transfer command buffer");
        }

        slot.state = if arm_immediately {
            TransferSlotState::Armed {
                recording_serial,
                block_epoch,
                window_generation: upload_ticket.window_generation(),
            }
        } else {
            TransferSlotState::Prepared {
                recording_serial,
                block_epoch: block_epoch.expect("上面已证明 Prepared 绑定 block epoch"),
                window_generation: upload_ticket.window_generation(),
            }
        };
        Ok(S14StarfoldRecordedTransfer {
            executor_id: self.executor_id,
            block_epoch,
            window: upload_ticket.window(),
            window_generation: upload_ticket.window_generation(),
            recording_serial,
            command_buffer: slot.command_buffer,
            fence: slot.fence,
            staging_buffer: slot.staging.handle(),
            destination_buffer: recording.buffer,
            destination_offset: recording.dst_buffer_offset,
            byte_len: recording.byte_len,
        })
    }

    fn refresh_slot(&mut self, index: usize) -> Result<()> {
        match self.slots[index].state {
            TransferSlotState::Idle | TransferSlotState::Prepared { .. } => return Ok(()),
            TransferSlotState::Armed { .. } => {}
        }
        let completed = unsafe {
            self.context
                .device
                .get_fence_status(self.slots[index].fence)
        }
        .context("查询 S14 StarFold transfer fence")?;
        if !completed {
            return Ok(());
        }
        unsafe {
            self.context
                .device
                .reset_fences(std::slice::from_ref(&self.slots[index].fence))
        }
        .context("reset S14 StarFold transfer fence")?;
        self.slots[index].state = TransferSlotState::Idle;
        Ok(())
    }

    /// 只对即将复用的 A/B staging/command slot 施加背压；另一窗口的 compute/transfer
    /// 仍可并发，避免第三个 tile 因瞬时 fence 未完成而把整层执行当成错误退出。
    fn wait_slot(&mut self, index: usize) -> Result<()> {
        self.refresh_slot(index)?;
        match self.slots[index].state {
            TransferSlotState::Idle => return Ok(()),
            TransferSlotState::Prepared { block_epoch, .. } => bail!(
                "S14 StarFold transfer slot 处于 block epoch {} Prepared；没有可等待 fence",
                block_epoch
            ),
            TransferSlotState::Armed { .. } => {}
        }
        unsafe {
            self.context.device.wait_for_fences(
                std::slice::from_ref(&self.slots[index].fence),
                true,
                u64::MAX,
            )
        }
        .context("等待 S14 StarFold A/B transfer slot")?;
        self.refresh_slot(index)?;
        if self.slots[index].state != TransferSlotState::Idle {
            bail!("S14 StarFold transfer fence 完成后 slot 仍未退休");
        }
        Ok(())
    }

    fn validate_prepared(
        &self,
        prepared: &S14StarfoldPreparedUpload,
        ticket: &S14StarfoldUploadTicket,
    ) -> Result<usize> {
        let recorded = prepared.recorded;
        if recorded.executor_id != self.executor_id {
            bail!("S14 StarFold prepared upload 来自其他 executor");
        }
        if !Arc::ptr_eq(&prepared.proof, ticket.proof()) {
            bail!("S14 StarFold prepared upload 的 proof lease 与 ticket 不同源");
        }
        validate_ticket_matches_recorded(ticket, recorded)?;
        let index = recorded.window.index();
        let expected = TransferSlotState::Prepared {
            recording_serial: recorded.recording_serial,
            block_epoch: prepared.block_epoch,
            window_generation: recorded.window_generation,
        };
        let slot = &self.slots[index];
        if slot.state != expected || !recorded_matches_slot(recorded, slot) {
            bail!("S14 StarFold prepared upload 已过期或身份漂移");
        }
        Ok(index)
    }

    fn validate_prepared_constellation(
        &self,
        prepared: &S14StarfoldPreparedConstellationUpload,
        ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
        recording: S14StarfoldUploadRecording,
        packet: &Arc<S14StarfoldConstellationPacket>,
    ) -> Result<usize> {
        let recorded = prepared.recorded;
        if recorded.executor_id != self.executor_id {
            bail!("S14 StarFold constellation prepared upload 来自其他 executor");
        }
        if !Arc::ptr_eq(&prepared.packet, packet) {
            bail!("S14 StarFold constellation prepared upload 的 packet lease 与 ticket 不同源");
        }
        validate_constellation_matches_recorded(ticket, recording, packet, recorded)?;
        let index = recorded.window.index();
        let expected = TransferSlotState::Prepared {
            recording_serial: recorded.recording_serial,
            block_epoch: prepared.block_epoch,
            window_generation: recorded.window_generation,
        };
        let slot = &self.slots[index];
        if slot.state != expected || !recorded_matches_slot(recorded, slot) {
            bail!("S14 StarFold constellation prepared upload 已过期或身份漂移");
        }
        Ok(index)
    }

    fn validate_recorded(&self, recorded: S14StarfoldRecordedTransfer) -> Result<usize> {
        if recorded.executor_id != self.executor_id {
            bail!("S14 StarFold transfer recording 来自其他 executor");
        }
        let index = recorded.window.index();
        let expected = TransferSlotState::Armed {
            recording_serial: recorded.recording_serial,
            block_epoch: recorded.block_epoch,
            window_generation: recorded.window_generation,
        };
        let slot = &self.slots[index];
        if slot.state != expected || !recorded_matches_slot(recorded, slot) {
            bail!("S14 StarFold transfer recording 已过期或身份漂移");
        }
        Ok(index)
    }

    fn destroy_resources(&mut self) {
        if self.destroyed {
            return;
        }
        unsafe {
            self.context.device.destroy_fence(self.slots[1].fence, None);
            self.context.device.destroy_fence(self.slots[0].fence, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        self.slots[1].staging.destroy(&self.context);
        self.slots[0].staging.destroy(&self.context);
        self.active_block_epoch = None;
        self.destroyed = true;
    }
}

impl Drop for S14StarfoldTransferExecutor {
    fn drop(&mut self) {
        if self.destroyed {
            return;
        }
        // Drop 是停机兜底；正常运行应先 try_destroy，避免全 device wait。
        let _ = unsafe { self.context.device.device_wait_idle() };
        self.destroy_resources();
    }
}

fn recorded_matches_slot(recorded: S14StarfoldRecordedTransfer, slot: &TransferSlot) -> bool {
    slot.command_buffer == recorded.command_buffer
        && slot.fence == recorded.fence
        && slot.staging.handle() == recorded.staging_buffer
}

/// `S14StarfoldUploadTicket` 的字段私有且只能由 runtime verified 路径创建；这里仍复核
/// proof key/长度，防止后续重构把“只有类型名可信”误当成可提交证明。
fn validate_verified_ticket(ticket: &S14StarfoldUploadTicket) -> Result<()> {
    let upload = ticket.ticket();
    let proof = ticket.proof();
    if upload.key() != S14StarfoldResidentWindowKey::Microtile(proof.key()) {
        bail!("S14 StarFold upload ticket 与 proof page key 漂移");
    }
    if upload.byte_len() != proof.byte_len() {
        bail!(
            "S14 StarFold upload ticket 与 proof 长度漂移: ticket={}, proof={}",
            upload.byte_len(),
            proof.byte_len()
        );
    }
    if upload.byte_len()
        != u64::try_from(ticket.bytes().len()).context("S14 StarFold proof bytes 超出 u64")?
    {
        bail!("S14 StarFold upload ticket 与 immutable proof bytes 长度漂移");
    }
    Ok(())
}

fn validate_constellation_ticket(
    ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
    recording: S14StarfoldUploadRecording,
    packet: &Arc<S14StarfoldConstellationPacket>,
) -> Result<()> {
    packet
        .validate()
        .context("校验 S14 StarFold constellation immutable packet")?;
    if ticket.key() != S14StarfoldResidentWindowKey::Constellation(packet.key()) {
        bail!("S14 StarFold constellation upload ticket 与 packet key 漂移");
    }
    let payload_bytes = u64::try_from(packet.payload().len())
        .context("S14 StarFold constellation payload 超出 u64")?;
    if ticket.byte_len() != packet.payload_bytes
        || packet.payload_bytes != payload_bytes
        || recording.byte_len != ticket.byte_len()
    {
        bail!(
            "S14 StarFold constellation upload 长度漂移: ticket={}, recording={}, packet={}, host={}",
            ticket.byte_len(),
            recording.byte_len,
            packet.payload_bytes,
            payload_bytes
        );
    }
    Ok(())
}

fn validate_ticket_matches_recorded(
    ticket: &S14StarfoldUploadTicket,
    recorded: S14StarfoldRecordedTransfer,
) -> Result<()> {
    validate_verified_ticket(ticket)?;
    let upload = ticket.ticket();
    let recording = ticket.recording();
    if upload.window() != recorded.window
        || upload.window_generation() != recorded.window_generation
        || upload.byte_len() != recorded.byte_len
        || recording.buffer != recorded.destination_buffer
        || recording.dst_buffer_offset != recorded.destination_offset
        || recording.byte_len != recorded.byte_len
    {
        bail!("S14 StarFold prepared recording 与 upload ticket 身份漂移");
    }
    Ok(())
}

fn validate_constellation_matches_recorded(
    ticket: VulkanUploadTicket<S14StarfoldResidentWindowKey>,
    recording: S14StarfoldUploadRecording,
    packet: &Arc<S14StarfoldConstellationPacket>,
    recorded: S14StarfoldRecordedTransfer,
) -> Result<()> {
    validate_constellation_ticket(ticket, recording, packet)?;
    if ticket.window() != recorded.window
        || ticket.window_generation() != recorded.window_generation
        || ticket.byte_len() != recorded.byte_len
        || recording.buffer != recorded.destination_buffer
        || recording.dst_buffer_offset != recorded.destination_offset
        || recording.byte_len != recorded.byte_len
    {
        bail!("S14 StarFold constellation prepared recording 与 upload ticket 身份漂移");
    }
    Ok(())
}

fn validate_recording(
    recording: S14StarfoldUploadRecording,
    ticket_bytes: vk::DeviceSize,
    host_bytes: usize,
    staging_bytes: vk::DeviceSize,
    transfer_family: u32,
) -> Result<()> {
    let host_bytes = u64::try_from(host_bytes).context("S14 StarFold host bytes 超出 u64")?;
    if recording.buffer == vk::Buffer::null() {
        bail!("S14 StarFold upload destination buffer 为空");
    }
    if recording.byte_len == 0
        || recording.byte_len != ticket_bytes
        || recording.byte_len != host_bytes
        || recording.byte_len > staging_bytes
    {
        bail!(
            "S14 StarFold upload 长度漂移: recording={}, ticket={}, host={}, staging={}",
            recording.byte_len,
            ticket_bytes,
            host_bytes,
            staging_bytes
        );
    }
    if recording.dst_buffer_offset % 4 != 0 || recording.byte_len % 4 != 0 {
        bail!(
            "S14 StarFold vkCmdCopyBuffer range 必须4字节对齐: dst={}, bytes={}",
            recording.dst_buffer_offset,
            recording.byte_len
        );
    }
    recording
        .dst_buffer_offset
        .checked_add(recording.byte_len)
        .context("S14 StarFold upload destination range 溢出")?;
    validate_barrier(
        &recording.release_to_compute,
        recording,
        transfer_family,
        true,
    )?;
    if let Some(acquire) = &recording.acquire_from_compute {
        validate_barrier(acquire, recording, transfer_family, false)?;
    }
    Ok(())
}

fn validate_barrier(
    barrier: &S14StarfoldBufferBarrier,
    recording: S14StarfoldUploadRecording,
    transfer_family: u32,
    release: bool,
) -> Result<()> {
    if barrier.barrier.buffer != recording.buffer {
        bail!("S14 StarFold upload barrier buffer 与 destination 漂移");
    }
    let copy_end = recording
        .dst_buffer_offset
        .checked_add(recording.byte_len)
        .context("S14 StarFold copy range 溢出")?;
    let barrier_end = if barrier.barrier.size == vk::WHOLE_SIZE {
        u64::MAX
    } else {
        barrier
            .barrier
            .offset
            .checked_add(barrier.barrier.size)
            .context("S14 StarFold barrier range 溢出")?
    };
    if barrier.barrier.offset > recording.dst_buffer_offset || barrier_end < copy_end {
        bail!("S14 StarFold upload barrier 未覆盖完整 copy range");
    }

    let queue_family = if release {
        barrier.barrier.src_queue_family_index
    } else {
        barrier.barrier.dst_queue_family_index
    };
    if queue_family != vk::QUEUE_FAMILY_IGNORED && queue_family != transfer_family {
        bail!("S14 StarFold upload barrier 与 transfer queue family 漂移");
    }
    Ok(())
}

/// # Safety
/// command_buffer 必须处于 recording 状态，且 staging/destination 在执行期间保持有效。
unsafe fn record_copy_commands(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    staging: vk::Buffer,
    recording: S14StarfoldUploadRecording,
) {
    if let Some(acquire) = &recording.acquire_from_compute {
        unsafe {
            acquire.record(device, command_buffer);
        }
    }

    let host_to_transfer = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::HOST_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(staging)
        .offset(0)
        .size(recording.byte_len);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&host_to_transfer),
            &[],
        );
        device.cmd_copy_buffer(
            command_buffer,
            staging,
            recording.buffer,
            &[vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(recording.dst_buffer_offset)
                .size(recording.byte_len)],
        );
        recording.release_to_compute.record(device, command_buffer);
    }
}
