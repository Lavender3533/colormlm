//! Starfold packed-row MXFP4 tile 的常驻 compute command owner。

use crate::{
    s14_starfold_mxfp4_tile::{
        S14StarfoldMxfp4ExternalSlice, S14StarfoldMxfp4ScaleAudit, S14StarfoldMxfp4TileBindings,
        S14StarfoldMxfp4TileDispatch, S14StarfoldMxfp4TileRecorder,
        S14StarfoldMxfp4TileRecordingReceipt, S14StarfoldMxfp4TileShape,
    },
    s14_starfold_routed_executor::constellation_packet::{
        S14StarfoldConstellationMemberReceipt, S14StarfoldConstellationPacket,
        S14StarfoldConstellationPacketReceipt,
        S14StarfoldConstellationReadyPacket, S14StarfoldResidentWindowKey,
        S14_STARFOLD_CONSTELLATION_CONTRACT_VERSION,
    },
    s14_starfold_runtime::S14StarfoldVerifiedMicrotile,
    s14_starfold_vulkan_windows::{
        S14StarfoldBufferBarrier, S14StarfoldComputeReceipt, S14StarfoldComputeRecording,
        S14StarfoldComputeTicket, S14StarfoldReadyBinding, S14StarfoldTimelinePoint,
        S14StarfoldVulkanConfig, S14StarfoldVulkanWindows, S14StarfoldWindowId,
    },
    VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

static NEXT_COMPUTE_OWNER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComputeSlotState {
    Idle,
    InFlight {
        submission_serial: u64,
        window_generation: u64,
    },
}

struct ComputeSlot {
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    state: ComputeSlotState,
    dispatches: Vec<S14StarfoldMxfp4TileDispatch>,
    proof: Option<Arc<S14StarfoldVerifiedMicrotile>>,
    constellation: Option<Arc<S14StarfoldConstellationPacket>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldMxfp4ComputeSubmissionReceipt {
    pub owner_id: u64,
    pub submission_serial: u64,
    pub window: S14StarfoldWindowId,
    pub window_generation: u64,
    pub consumer_id: u64,
    pub wait_transfer: S14StarfoldTimelinePoint,
    pub signal_compute: S14StarfoldTimelinePoint,
    pub compute: S14StarfoldComputeReceipt<S14StarfoldResidentWindowKey>,
    pub tiles: Vec<S14StarfoldMxfp4TileRecordingReceipt>,
    pub lane_dispatches: u32,
    pub acquire_barrier_calls: u32,
    pub output_barrier_calls: u32,
    pub release_barrier_calls: u32,
    pub begin_command_calls: u32,
    pub end_command_calls: u32,
    pub queue_submit_calls: u32,
}

impl S14StarfoldMxfp4ComputeSubmissionReceipt {
    pub fn validate(&self) -> Result<()> {
        if self.tiles.is_empty() || self.tiles.len() != self.lane_dispatches as usize {
            bail!("S14 Starfold MXFP4 compute 回执 lane/tile 数量漂移");
        }
        for tile in &self.tiles {
            tile.validate()?;
        }
        let expected_release = if self.compute.residency_retired { 1 } else { 0 };
        if self.owner_id == 0
            || self.submission_serial == 0
            || !matches!(
                self.compute.key,
                S14StarfoldResidentWindowKey::Microtile(_)
            )
            || self.window != self.compute.window
            || self.window_generation != self.compute.window_generation
            || self.consumer_id != self.compute.consumer_id
            || self.signal_compute != self.compute.completion
            || self.wait_transfer.generation != self.signal_compute.generation
            || self.acquire_barrier_calls > 1
            || self.output_barrier_calls != self.lane_dispatches
            || self.release_barrier_calls != expected_release
            || self.begin_command_calls != 1
            || self.end_command_calls != 1
            || self.queue_submit_calls != 1
        {
            bail!("S14 Starfold MXFP4 compute 回执不能证明完整 command 生命周期");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldConstellationComputeSubmissionReceipt {
    pub packet: S14StarfoldConstellationPacketReceipt,
    pub owner_id: u64,
    pub submission_serial: u64,
    pub consumer_id: u64,
    pub wait_transfer: S14StarfoldTimelinePoint,
    pub signal_compute: S14StarfoldTimelinePoint,
    pub compute: S14StarfoldComputeReceipt<S14StarfoldResidentWindowKey>,
    pub tiles: Vec<S14StarfoldMxfp4TileRecordingReceipt>,
    pub lane_dispatches: u32,
    pub acquire_barrier_calls: u32,
    pub output_barrier_calls: u32,
    pub release_barrier_calls: u32,
    pub begin_command_calls: u32,
    pub end_command_calls: u32,
    pub queue_submit_calls: u32,
}

impl S14StarfoldConstellationComputeSubmissionReceipt {
    pub fn validate(&self) -> Result<()> {
        self.packet.validate()?;
        if self.tiles.is_empty() || self.tiles.len() != self.lane_dispatches as usize {
            bail!("S14 StarFold 星座 compute lane/tile 回执数量漂移");
        }
        for tile in &self.tiles {
            tile.validate()?;
        }
        let expected_release = self.compute.residency_retired as u32;
        if self.owner_id == 0
            || self.submission_serial == 0
            || S14StarfoldResidentWindowKey::Constellation(self.packet.key) != self.compute.key
            || self.packet.window != self.compute.window
            || self.packet.window_generation != self.compute.window_generation
            || self.consumer_id != self.compute.consumer_id
            || self.signal_compute != self.compute.completion
            || self.wait_transfer.generation != self.signal_compute.generation
            || self.output_barrier_calls != self.lane_dispatches
            || self.acquire_barrier_calls > 1
            || self.release_barrier_calls != expected_release
            || self.begin_command_calls != 1
            || self.end_command_calls != 1
            || self.queue_submit_calls != 1
        {
            bail!("S14 StarFold 星座 compute 回执不能证明单 command 生命周期");
        }
        Ok(())
    }
}

/// 同一 resident tile 在一个 command 中服务一个 lane 的输入/输出切片。
#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldMxfp4LaneIo {
    pub lane: u8,
    pub input_f32: S14StarfoldMxfp4ExternalSlice,
    pub output_f32: S14StarfoldMxfp4ExternalSlice,
}

pub struct S14StarfoldMxfp4ComputeOwner {
    owner_id: u64,
    context: Arc<VulkanContext>,
    pipeline: Option<S14StarfoldMxfp4TileRecorder>,
    command_pool: vk::CommandPool,
    slots: [ComputeSlot; 2],
    next_submission_serial: u64,
    destroyed: bool,
}

impl S14StarfoldMxfp4ComputeOwner {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        if context.q_graphics == vk::Queue::null()
            || context.qf_graphics == vk::QUEUE_FAMILY_IGNORED
        {
            bail!("S14 Starfold MXFP4 compute queue/family 非法");
        }
        let pipeline = S14StarfoldMxfp4TileRecorder::new(&context)
            .context("创建 S14 Starfold MXFP4 常驻 tile pipeline")?;
        let command_pool = match unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .context("创建 S14 Starfold MXFP4 compute command pool")
        {
            Ok(pool) => pool,
            Err(error) => {
                pipeline.destroy(&context);
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
        .context("分配 S14 Starfold MXFP4 compute command buffers")
        {
            Ok(buffers) if buffers.len() == 2 => buffers,
            Ok(_) => {
                unsafe {
                    context.device.destroy_command_pool(command_pool, None);
                }
                pipeline.destroy(&context);
                bail!("S14 Starfold MXFP4 compute command buffer 数量漂移");
            }
            Err(error) => {
                unsafe {
                    context.device.destroy_command_pool(command_pool, None);
                }
                pipeline.destroy(&context);
                return Err(error);
            }
        };
        let fence_a = match create_fence(&context, "A") {
            Ok(fence) => fence,
            Err(error) => {
                unsafe {
                    context.device.destroy_command_pool(command_pool, None);
                }
                pipeline.destroy(&context);
                return Err(error);
            }
        };
        let fence_b = match create_fence(&context, "B") {
            Ok(fence) => fence,
            Err(error) => {
                unsafe {
                    context.device.destroy_fence(fence_a, None);
                    context.device.destroy_command_pool(command_pool, None);
                }
                pipeline.destroy(&context);
                return Err(error);
            }
        };
        let owner_id = NEXT_COMPUTE_OWNER_ID.fetch_add(1, Ordering::Relaxed);
        if owner_id == u64::MAX {
            unsafe {
                context.device.destroy_fence(fence_b, None);
                context.device.destroy_fence(fence_a, None);
                context.device.destroy_command_pool(command_pool, None);
            }
            pipeline.destroy(&context);
            bail!("S14 Starfold MXFP4 compute owner id 溢出");
        }
        Ok(Self {
            owner_id,
            context,
            pipeline: Some(pipeline),
            command_pool,
            slots: [
                ComputeSlot {
                    command_buffer: command_buffers[0],
                    fence: fence_a,
                    state: ComputeSlotState::Idle,
                    dispatches: Vec::new(),
                    proof: None,
                    constellation: None,
                },
                ComputeSlot {
                    command_buffer: command_buffers[1],
                    fence: fence_b,
                    state: ComputeSlotState::Idle,
                    dispatches: Vec::new(),
                    proof: None,
                    constellation: None,
                },
            ],
            next_submission_serial: 1,
            destroyed: false,
        })
    }

    pub const fn owner_id(&self) -> u64 {
        self.owner_id
    }

    /// 星座 packer 用同一 physical device 的真实 descriptor offset 对齐，避免用 256
    /// 之类经验常量制造设备相关失败。
    pub fn descriptor_offset_alignment(&self) -> u64 {
        let limits = unsafe {
            self.context
                .instance
                .get_physical_device_properties(self.context.physical)
                .limits
        };
        limits.min_storage_buffer_offset_alignment.max(1)
    }

    /// 一次 compute submission 复用同一 resident packed tile，连续处理所有命中专家的
    /// B4 lane。上传 consumer_count 因而应设为 1，而不是 lane 数；A/B window 在整批
    /// dispatch 完成后才允许退休或被 transfer 覆盖。
    #[allow(clippy::too_many_arguments)]
    pub fn submit_ready_tile_batch(
        &mut self,
        windows: &mut S14StarfoldVulkanWindows<S14StarfoldResidentWindowKey>,
        ready: S14StarfoldReadyBinding<S14StarfoldResidentWindowKey>,
        proof: Arc<S14StarfoldVerifiedMicrotile>,
        consumer_id: u64,
        shape: S14StarfoldMxfp4TileShape,
        tile_index: u32,
        lanes: &[S14StarfoldMxfp4LaneIo],
        scale_audit: S14StarfoldMxfp4ScaleAudit,
    ) -> Result<S14StarfoldMxfp4ComputeSubmissionReceipt> {
        if self.destroyed {
            bail!("S14 Starfold MXFP4 compute owner 已销毁");
        }
        let config = windows.config();
        validate_owner_config(&self.context, config, shape)?;
        let slot_index = ready.window().index();
        self.wait_slot(slot_index)?;
        if self.slots[slot_index].state != ComputeSlotState::Idle {
            bail!(
                "S14 Starfold MXFP4 {} compute command/fence 仍在执行",
                ready.window()
            );
        }

        let spec = shape.tile(tile_index)?;
        if ready.key() != S14StarfoldResidentWindowKey::Microtile(proof.key())
            || proof.byte_len() != ready.byte_len()
        {
            bail!("S14 Starfold MXFP4 ready binding 与 proof identity/bytes 漂移");
        }
        if lanes.is_empty() || lanes.len() > 4 {
            bail!("S14 Starfold MXFP4 resident tile 必须服务 1..=4 个 B4 lane");
        }
        let mut lane_mask = 0u8;
        for lane in lanes {
            if lane.lane >= 4 || lane_mask & (1u8 << lane.lane) != 0 {
                bail!("S14 Starfold MXFP4 lane 重复或越出 B4");
            }
            lane_mask |= 1u8 << lane.lane;
        }
        if ready.byte_len() != spec.payload_bytes {
            bail!(
                "S14 Starfold MXFP4 ready window payload 漂移: ready={} tile={}",
                ready.byte_len(),
                spec.payload_bytes
            );
        }
        let (ticket, recording) = windows
            .reserve_compute(ready, consumer_id)
            .map_err(anyhow::Error::new)
            .context("预留 S14 Starfold MXFP4 compute consumer")?;
        if let Err(error) = validate_compute_contract(config, ticket, recording, spec.payload_bytes)
        {
            return rollback_reservation(windows, ticket, error);
        }

        let mut dispatches = Vec::with_capacity(lanes.len());
        for lane in lanes {
            let bindings = S14StarfoldMxfp4TileBindings {
                input_f32: lane.input_f32,
                raw_window: S14StarfoldMxfp4ExternalSlice {
                    buffer: recording.buffer,
                    capacity_bytes: config.window_bytes,
                    offset: recording.buffer_offset,
                    logical_bytes: recording.byte_len,
                },
                output_f32: lane.output_f32,
                scale_audit,
            };
            match self
                .pipeline
                .as_ref()
                .context("S14 Starfold MXFP4 tile pipeline 已销毁")?
                .bind_external_tile(&self.context, shape, tile_index, bindings)
            {
                Ok(dispatch) => dispatches.push(dispatch),
                Err(error) => {
                    destroy_dispatches(&self.context, dispatches);
                    return rollback_reservation(windows, ticket, error);
                }
            }
        }

        let submission_serial = self.next_submission_serial;
        self.next_submission_serial = match self.next_submission_serial.checked_add(1) {
            Some(next) => next,
            None => {
                destroy_dispatches(&self.context, dispatches);
                return rollback_reservation(
                    windows,
                    ticket,
                    anyhow!("S14 Starfold MXFP4 compute submission serial 溢出"),
                );
            }
        };
        let command_buffer = self.slots[slot_index].command_buffer;
        let fence = self.slots[slot_index].fence;
        if let Err(error) = self.begin_command(command_buffer) {
            destroy_dispatches(&self.context, dispatches);
            return rollback_reservation(windows, ticket, error);
        }

        let recorded =
            unsafe { self.record_commands_batch(command_buffer, recording, lanes, &dispatches) };
        let tile_receipts = match recorded {
            Ok(receipts) => receipts,
            Err(error) => {
                let reset = self.reset_unsubmitted(command_buffer);
                destroy_dispatches(&self.context, dispatches);
                return rollback_after_recording(windows, ticket, error, reset);
            }
        };
        if let Err(error) = unsafe { self.context.device.end_command_buffer(command_buffer) }
            .context("结束 S14 Starfold MXFP4 compute command buffer")
        {
            let reset = self.reset_unsubmitted(command_buffer);
            destroy_dispatches(&self.context, dispatches);
            return rollback_after_recording(windows, ticket, error, reset);
        }

        let submit =
            unsafe { windows.submit_compute(&self.context.device, ticket, command_buffer, fence) }
                .map_err(anyhow::Error::new)
                .context("提交 S14 Starfold MXFP4 compute queue");
        let compute = match submit {
            Ok(receipt) => receipt,
            Err(error) => {
                let reset = self.reset_unsubmitted(command_buffer);
                destroy_dispatches(&self.context, dispatches);
                return rollback_after_recording(windows, ticket, error, reset);
            }
        };

        self.slots[slot_index].state = ComputeSlotState::InFlight {
            submission_serial,
            window_generation: ready.window_generation(),
        };
        self.slots[slot_index].dispatches = dispatches;
        self.slots[slot_index].proof = Some(proof);
        self.slots[slot_index].constellation = None;
        let receipt = S14StarfoldMxfp4ComputeSubmissionReceipt {
            owner_id: self.owner_id,
            submission_serial,
            window: ready.window(),
            window_generation: ready.window_generation(),
            consumer_id,
            wait_transfer: recording.wait_transfer,
            signal_compute: recording.signal_compute,
            compute,
            tiles: tile_receipts,
            lane_dispatches: lanes.len() as u32,
            acquire_barrier_calls: if recording.acquire_from_transfer.is_some() {
                1
            } else {
                0
            },
            output_barrier_calls: lanes.len() as u32,
            release_barrier_calls: if recording.release_to_transfer.is_some() {
                1
            } else {
                0
            },
            begin_command_calls: 1,
            end_command_calls: 1,
            queue_submit_calls: 1,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    /// 一个 resident 星座包只预留一次 window consumer，并在同一 command buffer 内依次
    /// dispatch 包中所有专家及其命中 lane。packet 本身持有每个专家的 proof/SHA lease，
    /// 直到对应 A/B fence 完成才会从 slot 释放。
    pub fn submit_ready_constellation_batch(
        &mut self,
        windows: &mut S14StarfoldVulkanWindows<S14StarfoldResidentWindowKey>,
        ready: S14StarfoldConstellationReadyPacket,
        consumer_id: u64,
    ) -> Result<S14StarfoldConstellationComputeSubmissionReceipt> {
        if self.destroyed {
            bail!("S14 StarFold 星座 compute owner 已销毁");
        }
        let (ready, packet) = ready.into_parts();
        packet.validate()?;
        let config = windows.config();
        if config.queues.compute_queue != self.context.q_graphics
            || config.queues.compute_family != self.context.qf_graphics
            || config.window_bytes != u64::from(packet.window_capacity_bytes)
        {
            bail!("S14 StarFold 星座 packet/window/compute queue 合同漂移");
        }
        let slot_index = ready.window().index();
        self.wait_slot(slot_index)?;
        if self.slots[slot_index].state != ComputeSlotState::Idle {
            bail!("S14 StarFold 星座 {} compute slot 仍在执行", ready.window());
        }
        if ready.key() != S14StarfoldResidentWindowKey::Constellation(packet.key())
            || ready.byte_len() != packet.payload_bytes
        {
            bail!("S14 StarFold 星座 ready binding identity/bytes 漂移");
        }

        let lane_dispatches = packet.members().iter().try_fold(0usize, |sum, member| {
            validate_owner_config(&self.context, config, member.shape)?;
            sum.checked_add(member.lanes().len())
                .context("S14 StarFold 星座 lane dispatch count overflow")
        })?;
        if lane_dispatches == 0 {
            bail!("S14 StarFold 星座没有 lane dispatch");
        }

        let (ticket, recording) = windows
            .reserve_compute(ready, consumer_id)
            .map_err(anyhow::Error::new)
            .context("预留 S14 StarFold 星座 compute consumer")?;
        if let Err(error) =
            validate_compute_contract(config, ticket, recording, packet.payload_bytes)
        {
            return rollback_reservation(windows, ticket, error);
        }

        let mut dispatches = Vec::with_capacity(lane_dispatches);
        for member in packet.members() {
            let Some(raw_offset) = recording.buffer_offset.checked_add(member.window_offset) else {
                destroy_dispatches(&self.context, dispatches);
                return rollback_reservation(
                    windows,
                    ticket,
                    anyhow!("S14 StarFold 星座 descriptor offset overflow"),
                );
            };
            for lane in member.lanes() {
                let bindings = S14StarfoldMxfp4TileBindings {
                    input_f32: lane.input_f32,
                    raw_window: S14StarfoldMxfp4ExternalSlice {
                        buffer: recording.buffer,
                        capacity_bytes: config.window_bytes,
                        offset: raw_offset,
                        logical_bytes: member.payload_bytes,
                    },
                    output_f32: lane.output_f32,
                    scale_audit: member.scale_audit,
                };
                match self
                    .pipeline
                    .as_ref()
                    .context("S14 StarFold MXFP4 tile pipeline 已销毁")?
                    .bind_external_tile(&self.context, member.shape, member.tile_index, bindings)
                {
                    Ok(dispatch) => dispatches.push(dispatch),
                    Err(error) => {
                        destroy_dispatches(&self.context, dispatches);
                        return rollback_reservation(windows, ticket, error);
                    }
                }
            }
        }
        if dispatches.len() != lane_dispatches || dispatches.is_empty() {
            destroy_dispatches(&self.context, dispatches);
            return rollback_reservation(
                windows,
                ticket,
                anyhow!("S14 StarFold 星座 descriptor/lane 数量漂移"),
            );
        }

        let submission_serial = self.next_submission_serial;
        self.next_submission_serial = match self.next_submission_serial.checked_add(1) {
            Some(next) => next,
            None => {
                destroy_dispatches(&self.context, dispatches);
                return rollback_reservation(
                    windows,
                    ticket,
                    anyhow!("S14 StarFold 星座 compute submission serial 溢出"),
                );
            }
        };
        let command_buffer = self.slots[slot_index].command_buffer;
        let fence = self.slots[slot_index].fence;
        if let Err(error) = self.begin_command(command_buffer) {
            destroy_dispatches(&self.context, dispatches);
            return rollback_reservation(windows, ticket, error);
        }
        let recorded = unsafe {
            self.record_constellation_commands(command_buffer, recording, &packet, &dispatches)
        };
        let (members, tile_receipts) = match recorded {
            Ok(receipts) => receipts,
            Err(error) => {
                let reset = self.reset_unsubmitted(command_buffer);
                destroy_dispatches(&self.context, dispatches);
                return rollback_after_recording(windows, ticket, error, reset);
            }
        };
        if let Err(error) = unsafe { self.context.device.end_command_buffer(command_buffer) }
            .context("结束 S14 StarFold 星座 compute command buffer")
        {
            let reset = self.reset_unsubmitted(command_buffer);
            destroy_dispatches(&self.context, dispatches);
            return rollback_after_recording(windows, ticket, error, reset);
        }

        let submit =
            unsafe { windows.submit_compute(&self.context.device, ticket, command_buffer, fence) }
                .map_err(anyhow::Error::new)
                .context("提交 S14 StarFold 星座 compute queue");
        let compute = match submit {
            Ok(receipt) => receipt,
            Err(error) => {
                let reset = self.reset_unsubmitted(command_buffer);
                destroy_dispatches(&self.context, dispatches);
                return rollback_after_recording(windows, ticket, error, reset);
            }
        };

        self.slots[slot_index].state = ComputeSlotState::InFlight {
            submission_serial,
            window_generation: ready.window_generation(),
        };
        self.slots[slot_index].dispatches = dispatches;
        self.slots[slot_index].constellation = Some(Arc::clone(&packet));
        let packet_receipt = S14StarfoldConstellationPacketReceipt {
            contract_version: S14_STARFOLD_CONSTELLATION_CONTRACT_VERSION,
            key: packet.key(),
            window: ready.window(),
            window_generation: ready.window_generation(),
            packet_bytes: packet.payload_bytes,
            logical_payload_bytes: packet.logical_payload_bytes,
            members,
            transfer_submit_calls: 1,
            compute_submit_calls: 1,
            serial_token_forward_calls: 0,
        };
        let lane_dispatches = u32::try_from(lane_dispatches)
            .context("S14 StarFold 星座 lane dispatch count 超出 u32")?;
        let receipt = S14StarfoldConstellationComputeSubmissionReceipt {
            packet: packet_receipt,
            owner_id: self.owner_id,
            submission_serial,
            consumer_id,
            wait_transfer: recording.wait_transfer,
            signal_compute: recording.signal_compute,
            compute,
            tiles: tile_receipts,
            lane_dispatches,
            acquire_barrier_calls: recording.acquire_from_transfer.is_some() as u32,
            output_barrier_calls: lane_dispatches,
            release_barrier_calls: recording.release_to_transfer.is_some() as u32,
            begin_command_calls: 1,
            end_command_calls: 1,
            queue_submit_calls: 1,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn try_destroy(&mut self) -> Result<()> {
        if self.destroyed {
            return Ok(());
        }
        self.refresh_slot(0)?;
        self.refresh_slot(1)?;
        if self
            .slots
            .iter()
            .any(|slot| slot.state != ComputeSlotState::Idle)
        {
            bail!("S14 Starfold MXFP4 compute 仍有 in-flight command");
        }
        self.destroy_resources();
        Ok(())
    }

    fn begin_command(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.context
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .context("reset S14 Starfold MXFP4 compute command buffer")?;
        unsafe {
            self.context.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .context("begin S14 Starfold MXFP4 compute command buffer")
    }

    unsafe fn record_commands_batch(
        &self,
        command_buffer: vk::CommandBuffer,
        recording: S14StarfoldComputeRecording,
        lanes: &[S14StarfoldMxfp4LaneIo],
        dispatches: &[S14StarfoldMxfp4TileDispatch],
    ) -> Result<Vec<S14StarfoldMxfp4TileRecordingReceipt>> {
        if lanes.len() != dispatches.len() || lanes.is_empty() {
            bail!("S14 Starfold MXFP4 batch command lane/descriptor 数漂移");
        }
        if let Some(acquire) = &recording.acquire_from_transfer {
            unsafe {
                acquire.record(&self.context.device, command_buffer);
            }
        }
        let mut receipts = Vec::with_capacity(lanes.len());
        for (lane, dispatch) in lanes.iter().zip(dispatches) {
            let tile = unsafe {
                self.pipeline
                    .as_ref()
                    .context("S14 Starfold MXFP4 tile pipeline 已销毁")?
                    .record_tile(&self.context, command_buffer, dispatch)
            }?;
            unsafe {
                record_output_barrier(&self.context.device, command_buffer, lane.output_f32, tile)?;
            }
            receipts.push(tile);
        }
        if let Some(release) = &recording.release_to_transfer {
            unsafe {
                release.record(&self.context.device, command_buffer);
            }
        }
        Ok(receipts)
    }

    unsafe fn record_constellation_commands(
        &self,
        command_buffer: vk::CommandBuffer,
        recording: S14StarfoldComputeRecording,
        packet: &S14StarfoldConstellationPacket,
        dispatches: &[S14StarfoldMxfp4TileDispatch],
    ) -> Result<(
        Vec<S14StarfoldConstellationMemberReceipt>,
        Vec<S14StarfoldMxfp4TileRecordingReceipt>,
    )> {
        let expected_dispatches = packet.members().iter().try_fold(0usize, |sum, member| {
            sum.checked_add(member.lanes().len())
                .context("S14 StarFold 星座 record lane count overflow")
        })?;
        if expected_dispatches == 0 || expected_dispatches != dispatches.len() {
            bail!("S14 StarFold 星座 record descriptor/lane 数量漂移");
        }
        if let Some(acquire) = &recording.acquire_from_transfer {
            unsafe { acquire.record(&self.context.device, command_buffer) };
        }
        let mut cursor = 0usize;
        let mut members = Vec::with_capacity(packet.members().len());
        let mut receipts = Vec::with_capacity(dispatches.len());
        for member in packet.members() {
            for lane in member.lanes() {
                let dispatch = dispatches
                    .get(cursor)
                    .context("S14 StarFold 星座 dispatch cursor 越界")?;
                let tile = unsafe {
                    self.pipeline
                        .as_ref()
                        .context("S14 StarFold MXFP4 tile pipeline 已销毁")?
                        .record_tile(&self.context, command_buffer, dispatch)
                }?;
                unsafe {
                    record_output_barrier(
                        &self.context.device,
                        command_buffer,
                        lane.output_f32,
                        tile,
                    )?;
                }
                receipts.push(tile);
                cursor += 1;
            }
            members.push(S14StarfoldConstellationMemberReceipt {
                expert_id: member.expert_id,
                source_key: member.source_key,
                window_offset: member.window_offset,
                payload_bytes: member.payload_bytes,
                lane_dispatches: u32::try_from(member.lanes().len())
                    .context("S14 StarFold 星座 member lane count 超出 u32")?,
            });
        }
        if cursor != dispatches.len() {
            bail!("S14 StarFold 星座 dispatch cursor 未完整消费");
        }
        if let Some(release) = &recording.release_to_transfer {
            unsafe { release.record(&self.context.device, command_buffer) };
        }
        Ok((members, receipts))
    }

    fn reset_unsubmitted(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.context
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .context("回滚 S14 Starfold MXFP4 未提交 command buffer")
    }

    fn refresh_slot(&mut self, index: usize) -> Result<()> {
        if self.slots[index].state == ComputeSlotState::Idle {
            return Ok(());
        }
        let complete = unsafe {
            self.context
                .device
                .get_fence_status(self.slots[index].fence)
        }
        .context("查询 S14 Starfold MXFP4 compute fence")?;
        if !complete {
            return Ok(());
        }
        unsafe {
            self.context
                .device
                .reset_fences(std::slice::from_ref(&self.slots[index].fence))
        }
        .context("reset S14 Starfold MXFP4 compute fence")?;
        if self.slots[index].dispatches.is_empty() {
            bail!("S14 Starfold MXFP4 in-flight slot 缺少 descriptor lifetime owner");
        }
        let has_single_proof = self.slots[index].proof.take().is_some();
        let has_constellation = self.slots[index].constellation.take().is_some();
        if has_single_proof == has_constellation {
            bail!("S14 Starfold MXFP4 in-flight slot 缺少 proof/SHA lifetime owner");
        }
        destroy_dispatches(
            &self.context,
            std::mem::take(&mut self.slots[index].dispatches),
        );
        self.slots[index].state = ComputeSlotState::Idle;
        Ok(())
    }

    /// A/B window 第三次复用时只等待同一窗口最老的 compute；另一个窗口及 transfer
    /// queue 可继续前进，因此不会把整条 StarFold 流水退化成逐 tile device-idle。
    fn wait_slot(&mut self, index: usize) -> Result<()> {
        self.refresh_slot(index)?;
        if self.slots[index].state == ComputeSlotState::Idle {
            return Ok(());
        }
        unsafe {
            self.context.device.wait_for_fences(
                std::slice::from_ref(&self.slots[index].fence),
                true,
                u64::MAX,
            )
        }
        .context("等待 S14 Starfold MXFP4 A/B compute slot")?;
        self.refresh_slot(index)?;
        if self.slots[index].state != ComputeSlotState::Idle {
            bail!("S14 Starfold MXFP4 compute fence 完成后 slot 仍未退休");
        }
        Ok(())
    }

    fn destroy_resources(&mut self) {
        if self.destroyed {
            return;
        }
        for slot in &mut self.slots {
            destroy_dispatches(&self.context, std::mem::take(&mut slot.dispatches));
            slot.proof.take();
            slot.constellation.take();
        }
        unsafe {
            self.context.device.destroy_fence(self.slots[1].fence, None);
            self.context.device.destroy_fence(self.slots[0].fence, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        if let Some(pipeline) = self.pipeline.take() {
            pipeline.destroy(&self.context);
        }
        self.command_pool = vk::CommandPool::null();
        self.slots[0].fence = vk::Fence::null();
        self.slots[1].fence = vk::Fence::null();
        self.destroyed = true;
    }
}

impl Drop for S14StarfoldMxfp4ComputeOwner {
    fn drop(&mut self) {
        if self.destroyed {
            return;
        }
        let _ = unsafe { self.context.device.device_wait_idle() };
        self.destroy_resources();
    }
}

fn destroy_dispatches(context: &VulkanContext, dispatches: Vec<S14StarfoldMxfp4TileDispatch>) {
    for dispatch in dispatches {
        dispatch.destroy(context);
    }
}

fn create_fence(context: &VulkanContext, label: &str) -> Result<vk::Fence> {
    unsafe {
        context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    }
    .with_context(|| format!("创建 S14 Starfold MXFP4 compute fence {label}"))
}

fn validate_owner_config(
    context: &VulkanContext,
    config: S14StarfoldVulkanConfig,
    shape: S14StarfoldMxfp4TileShape,
) -> Result<()> {
    if config.queues.compute_queue != context.q_graphics
        || config.queues.compute_family != context.qf_graphics
    {
        bail!("S14 Starfold MXFP4 command owner 与 window compute queue/family 漂移");
    }
    if config.window_bytes != u64::from(shape.window_capacity_bytes()) {
        bail!(
            "S14 Starfold MXFP4 shape/window capacity 漂移: shape={} windows={}",
            shape.window_capacity_bytes(),
            config.window_bytes
        );
    }
    Ok(())
}

fn validate_compute_contract<K: Copy>(
    config: S14StarfoldVulkanConfig,
    ticket: S14StarfoldComputeTicket<K>,
    recording: S14StarfoldComputeRecording,
    expected_payload_bytes: u64,
) -> Result<()> {
    if recording.buffer == vk::Buffer::null()
        || recording.buffer_offset != 0
        || recording.byte_len != expected_payload_bytes
        || ticket.byte_len() != expected_payload_bytes
        || recording.wait_transfer != ticket.transfer_ready()
        || recording.signal_compute != ticket.signal_compute()
        || recording.wait_transfer.semaphore != config.timelines.transfer
        || recording.signal_compute.semaphore != config.timelines.compute
        || recording.wait_transfer.generation != config.timelines.generation
        || recording.signal_compute.generation != config.timelines.generation
        || recording.release_to_transfer.is_some() != ticket.is_final_consumer()
    {
        bail!("S14 Starfold MXFP4 compute ticket/recording/payload 合同漂移");
    }
    if let Some(acquire) = &recording.acquire_from_transfer {
        validate_window_barrier(acquire, recording, config.queues.compute_family, true)?;
    }
    if let Some(release) = &recording.release_to_transfer {
        validate_window_barrier(release, recording, config.queues.compute_family, false)?;
    }
    Ok(())
}

fn validate_window_barrier(
    barrier: &S14StarfoldBufferBarrier,
    recording: S14StarfoldComputeRecording,
    compute_family: u32,
    acquire: bool,
) -> Result<()> {
    if barrier.barrier.buffer != recording.buffer {
        bail!("S14 Starfold MXFP4 window barrier buffer 漂移");
    }
    let recording_end = recording
        .buffer_offset
        .checked_add(recording.byte_len)
        .context("S14 Starfold MXFP4 recording range overflow")?;
    let barrier_end = if barrier.barrier.size == vk::WHOLE_SIZE {
        u64::MAX
    } else {
        barrier
            .barrier
            .offset
            .checked_add(barrier.barrier.size)
            .context("S14 Starfold MXFP4 window barrier range overflow")?
    };
    if barrier.barrier.offset > recording.buffer_offset || barrier_end < recording_end {
        bail!("S14 Starfold MXFP4 window barrier 未覆盖完整 resident payload");
    }
    let family = if acquire {
        barrier.barrier.dst_queue_family_index
    } else {
        barrier.barrier.src_queue_family_index
    };
    if family != vk::QUEUE_FAMILY_IGNORED && family != compute_family {
        bail!("S14 Starfold MXFP4 window barrier compute queue family 漂移");
    }
    if acquire {
        if barrier.dst_stage_mask != vk::PipelineStageFlags::COMPUTE_SHADER
            || !barrier
                .barrier
                .dst_access_mask
                .contains(vk::AccessFlags::SHADER_READ)
        {
            bail!("S14 Starfold MXFP4 acquire barrier 未发布 shader read 可见性");
        }
    } else if barrier.src_stage_mask != vk::PipelineStageFlags::COMPUTE_SHADER
        || !barrier
            .barrier
            .src_access_mask
            .contains(vk::AccessFlags::SHADER_READ)
    {
        bail!("S14 Starfold MXFP4 release barrier 未覆盖 shader read");
    }
    Ok(())
}

unsafe fn record_output_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    output: S14StarfoldMxfp4ExternalSlice,
    tile: S14StarfoldMxfp4TileRecordingReceipt,
) -> Result<()> {
    let offset = output
        .offset
        .checked_add(tile.global_output_byte_start)
        .context("S14 Starfold MXFP4 output barrier offset overflow")?;
    let size = tile
        .global_output_byte_end
        .checked_sub(tile.global_output_byte_start)
        .context("S14 Starfold MXFP4 output barrier range underflow")?;
    let end = offset
        .checked_add(size)
        .context("S14 Starfold MXFP4 output barrier range overflow")?;
    let logical_end = output
        .offset
        .checked_add(output.logical_bytes)
        .context("S14 Starfold MXFP4 output logical range overflow")?;
    if output.buffer == vk::Buffer::null() || size == 0 || end > logical_end {
        bail!("S14 Starfold MXFP4 output barrier 越出全局 output");
    }
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(output.buffer)
        .offset(offset)
        .size(size);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&barrier),
            &[],
        );
    }
    Ok(())
}

fn rollback_reservation<K: Copy + Eq, T>(
    windows: &mut S14StarfoldVulkanWindows<K>,
    ticket: S14StarfoldComputeTicket<K>,
    primary: anyhow::Error,
) -> Result<T> {
    match windows.cancel_compute(ticket).map_err(anyhow::Error::new) {
        Ok(()) => Err(primary),
        Err(rollback) => Err(anyhow!(
            "{primary:#}; compute reservation rollback: {rollback:#}"
        )),
    }
}

fn rollback_after_recording<K: Copy + Eq, T>(
    windows: &mut S14StarfoldVulkanWindows<K>,
    ticket: S14StarfoldComputeTicket<K>,
    primary: anyhow::Error,
    command_reset: Result<()>,
) -> Result<T> {
    let reservation = windows.cancel_compute(ticket).map_err(anyhow::Error::new);
    match (command_reset, reservation) {
        (Ok(()), Ok(())) => Err(primary),
        (command_reset, reservation) => Err(anyhow!(
            "{primary:#}; compute rollback: command={command_reset:?} reservation={reservation:?}"
        )),
    }
}
