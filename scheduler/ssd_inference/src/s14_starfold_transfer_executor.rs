//! StarFold verified microtile 的双 staging Vulkan transfer 录制器。
//!
//! 本模块拥有两个常驻 HOST_VISIBLE staging buffer、一个 transfer command pool、两个
//! command buffer 与对应 fence。它只负责把已经由 runtime 校验过的 microtile 字节精确
//! 录制成 `staging -> StarFold window` copy；真正的 queue submit 仍由
//! `S14StarfoldRuntime::submit_upload` 完成。

use crate::{
    s14_starfold_runtime::S14StarfoldUploadTicket,
    s14_starfold_vulkan_windows::{
        S14StarfoldBufferBarrier, S14StarfoldUploadRecording, S14StarfoldWindowId,
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
    /// fence 尚未完成时既可能在等待 runtime submit，也可能已经 in-flight。
    Armed {
        recording_serial: u64,
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

/// 常驻双 staging transfer 资源。正常热路径不等待 fence：目标槽位尚未完成时直接返回错误，
/// 由上层继续使用另一个 StarFold window 或稍后重试。
pub struct S14StarfoldTransferExecutor {
    executor_id: u64,
    context: Arc<VulkanContext>,
    staging_bytes: vk::DeviceSize,
    command_pool: vk::CommandPool,
    slots: [TransferSlot; 2],
    next_recording_serial: u64,
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
            destroyed: false,
        })
    }

    pub const fn staging_bytes(&self) -> vk::DeviceSize {
        self.staging_bytes
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
        let upload_ticket = ticket.ticket();
        let recording = ticket.recording();
        let bytes = ticket.bytes();
        validate_recording(
            recording,
            upload_ticket.byte_len(),
            bytes.len(),
            self.staging_bytes,
            self.context.qf_transfer,
        )?;

        let index = upload_ticket.window().index();
        self.wait_slot(index)?;
        if self.slots[index].state != TransferSlotState::Idle {
            bail!(
                "S14 StarFold transfer {} staging/command buffer 仍在等待提交或执行完成",
                upload_ticket.window()
            );
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

        slot.state = TransferSlotState::Armed {
            recording_serial,
            window_generation: upload_ticket.window_generation(),
        };
        Ok(S14StarfoldRecordedTransfer {
            executor_id: self.executor_id,
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

    fn refresh_slot(&mut self, index: usize) -> Result<()> {
        if self.slots[index].state == TransferSlotState::Idle {
            return Ok(());
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
        if self.slots[index].state == TransferSlotState::Idle {
            return Ok(());
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

    fn validate_recorded(&self, recorded: S14StarfoldRecordedTransfer) -> Result<usize> {
        if recorded.executor_id != self.executor_id {
            bail!("S14 StarFold transfer recording 来自其他 executor");
        }
        let index = recorded.window.index();
        let expected = TransferSlotState::Armed {
            recording_serial: recorded.recording_serial,
            window_generation: recorded.window_generation,
        };
        let slot = &self.slots[index];
        if slot.state != expected
            || slot.command_buffer != recorded.command_buffer
            || slot.fence != recorded.fence
            || slot.staging.handle() != recorded.staging_buffer
        {
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
