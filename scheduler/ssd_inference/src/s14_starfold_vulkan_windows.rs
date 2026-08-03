//! StarFold 的 Vulkan 双窗口物理执行器。
//!
//! VkBuffer、VkDeviceMemory、queue 和 timeline semaphore 均由接入层创建并保活；
//! 本模块负责外部显存子分配、绑定、双窗口状态、timeline 提交和独占队列族所有权切换。

use crate::s14_vulkan_arena::{
    DeviceSize, S14ScratchLifetime, S14ScratchPlan, S14ScratchRequest, S14VulkanArena,
    S14VulkanArenaError, S14VulkanArenaSlice,
};
use ash::vk::{self, Handle};
use std::{
    collections::HashSet,
    error::Error,
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

pub const S14_STARFOLD_ONE_MIB: DeviceSize = 1024 * 1024;

static NEXT_EXECUTOR_ID: AtomicU64 = AtomicU64::new(1);

pub type S14StarfoldVulkanResult<T> = Result<T, S14StarfoldVulkanError>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S14StarfoldVulkanError {
    Arena(S14VulkanArenaError),
    NullHandle(&'static str),
    InvalidWindowBytes(DeviceSize),
    InvalidQueueFamily(u32),
    InvalidTimelineGeneration,
    SharedTimelineSemaphore,
    TimelineOverflow(&'static str),
    InvalidMemoryTypeIndex(u32),
    BufferTooSmall {
        role: S14StarfoldBufferRole,
        buffer_bytes: DeviceSize,
        required_bytes: DeviceSize,
    },
    InvalidMemoryRequirements {
        role: S14StarfoldBufferRole,
        size: DeviceSize,
        alignment: DeviceSize,
    },
    IncompatibleMemoryType {
        role: S14StarfoldBufferRole,
        memory_type_index: u32,
        memory_type_bits: u32,
    },
    DuplicateBuffer,
    DuplicateScratchRequestId(u64),
    BuffersAlreadyBound,
    BuffersNotBound,
    BufferBindingPoisoned,
    Vulkan(vk::Result),
    NoReusableWindow,
    InvalidUploadBytes {
        requested: DeviceSize,
        capacity: DeviceSize,
    },
    InvalidConsumerCount(u32),
    StaleUpload,
    StaleReadyBinding,
    ComputeReservationBusy,
    DuplicateConsumer(u64),
    StaleCompute,
    ScratchNotFound(u64),
    ScratchOutsideLifetime {
        request_id: u64,
        use_index: u64,
        lifetime: S14ScratchLifetime,
    },
    TimelineGenerationMismatch {
        expected: u64,
        actual: u64,
    },
    TimelinesIncomplete {
        required_transfer: u64,
        completed_transfer: u64,
        required_compute: u64,
        completed_compute: u64,
    },
    InternalInvariant(&'static str),
}

impl fmt::Display for S14StarfoldVulkanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arena(error) => write!(formatter, "StarFold Vulkan Arena: {error}"),
            Self::NullHandle(kind) => write!(formatter, "StarFold Vulkan {kind} 句柄不能为空"),
            Self::InvalidWindowBytes(bytes) => {
                write!(
                    formatter,
                    "StarFold Vulkan window_bytes 必须大于 0，收到 {bytes}"
                )
            }
            Self::InvalidQueueFamily(index) => {
                write!(formatter, "StarFold Vulkan queue family 非法: {index}")
            }
            Self::InvalidTimelineGeneration => {
                write!(formatter, "StarFold Vulkan timeline generation 必须大于 0")
            }
            Self::SharedTimelineSemaphore => write!(
                formatter,
                "StarFold Vulkan transfer/compute 必须使用不同的 timeline semaphore"
            ),
            Self::TimelineOverflow(kind) => {
                write!(formatter, "StarFold Vulkan {kind} timeline value 溢出")
            }
            Self::InvalidMemoryTypeIndex(index) => {
                write!(
                    formatter,
                    "StarFold Vulkan memory type index 超出 0..32: {index}"
                )
            }
            Self::BufferTooSmall {
                role,
                buffer_bytes,
                required_bytes,
            } => write!(
                formatter,
                "StarFold Vulkan {role} buffer 过小: buffer={buffer_bytes}, required={required_bytes}"
            ),
            Self::InvalidMemoryRequirements {
                role,
                size,
                alignment,
            } => write!(
                formatter,
                "StarFold Vulkan {role} memory requirements 非法: size={size}, alignment={alignment}"
            ),
            Self::IncompatibleMemoryType {
                role,
                memory_type_index,
                memory_type_bits,
            } => write!(
                formatter,
                "StarFold Vulkan {role} 不兼容 memory type {memory_type_index}: bits={memory_type_bits:#010x}"
            ),
            Self::DuplicateBuffer => write!(formatter, "StarFold Vulkan buffer 句柄重复"),
            Self::DuplicateScratchRequestId(request_id) => {
                write!(
                    formatter,
                    "StarFold Vulkan scratch request_id 重复: {request_id}"
                )
            }
            Self::BuffersAlreadyBound => write!(formatter, "StarFold Vulkan buffers 已绑定"),
            Self::BuffersNotBound => write!(formatter, "StarFold Vulkan buffers 尚未绑定"),
            Self::BufferBindingPoisoned => write!(
                formatter,
                "StarFold Vulkan buffer 绑定只完成了一部分；必须销毁外部 buffers 后重建执行器"
            ),
            Self::Vulkan(error) => write!(formatter, "StarFold Vulkan 调用失败: {error:?}"),
            Self::NoReusableWindow => write!(
                formatter,
                "StarFold Vulkan A/B 窗口都仍被上传、录制或驻留消费者占用"
            ),
            Self::InvalidUploadBytes {
                requested,
                capacity,
            } => write!(
                formatter,
                "StarFold Vulkan segment bytes 超出窗口: requested={requested}, capacity={capacity}"
            ),
            Self::InvalidConsumerCount(count) => write!(
                formatter,
                "StarFold Vulkan 每次驻留的 consumer_count 必须大于 0，收到 {count}"
            ),
            Self::StaleUpload => write!(
                formatter,
                "StarFold Vulkan 拒绝过期、取消或来自其他执行器的 upload 回调"
            ),
            Self::StaleReadyBinding => write!(
                formatter,
                "StarFold Vulkan ready binding 已过期或窗口驻留身份已漂移"
            ),
            Self::ComputeReservationBusy => write!(
                formatter,
                "StarFold Vulkan 同一窗口已有尚未提交或取消的 compute reservation"
            ),
            Self::DuplicateConsumer(consumer_id) => write!(
                formatter,
                "StarFold Vulkan 同一驻留权重不能重复服务 consumer {consumer_id}"
            ),
            Self::StaleCompute => write!(
                formatter,
                "StarFold Vulkan 拒绝过期、取消或来自其他执行器的 compute 回调"
            ),
            Self::ScratchNotFound(request_id) => {
                write!(formatter, "StarFold Vulkan scratch 不存在: {request_id}")
            }
            Self::ScratchOutsideLifetime {
                request_id,
                use_index,
                lifetime,
            } => write!(
                formatter,
                "StarFold Vulkan scratch {request_id} 在生命周期外使用: use={use_index}, lifetime={}..={} ",
                lifetime.first_use, lifetime.last_use
            ),
            Self::TimelineGenerationMismatch { expected, actual } => write!(
                formatter,
                "StarFold Vulkan timeline generation 漂移: expected={expected}, actual={actual}"
            ),
            Self::TimelinesIncomplete {
                required_transfer,
                completed_transfer,
                required_compute,
                completed_compute,
            } => write!(
                formatter,
                "StarFold Vulkan 仍有 GPU 引用: transfer={completed_transfer}/{required_transfer}, compute={completed_compute}/{required_compute}"
            ),
            Self::InternalInvariant(message) => {
                write!(formatter, "StarFold Vulkan 内部不变量失败: {message}")
            }
        }
    }
}

impl Error for S14StarfoldVulkanError {}

impl From<S14VulkanArenaError> for S14StarfoldVulkanError {
    fn from(error: S14VulkanArenaError) -> Self {
        Self::Arena(error)
    }
}

impl From<vk::Result> for S14StarfoldVulkanError {
    fn from(error: vk::Result) -> Self {
        Self::Vulkan(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum S14StarfoldWindowId {
    A,
    B,
}

impl S14StarfoldWindowId {
    pub const ALL: [Self; 2] = [Self::A, Self::B];

    pub const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }
}

impl fmt::Display for S14StarfoldWindowId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::A => "window A",
            Self::B => "window B",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum S14StarfoldBufferRole {
    Window(S14StarfoldWindowId),
    Scratch(u64),
}

impl fmt::Display for S14StarfoldBufferRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Window(window) => write!(formatter, "{window}"),
            Self::Scratch(request_id) => write!(formatter, "scratch {request_id}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldBufferSpec {
    pub buffer: vk::Buffer,
    /// 创建 VkBuffer 时传入的 size。
    pub buffer_bytes: DeviceSize,
    /// `vkGetBufferMemoryRequirements` 的原样结果。
    pub memory_requirements: vk::MemoryRequirements,
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldScratchBufferSpec {
    pub request_id: u64,
    pub buffer: S14StarfoldBufferSpec,
    pub lifetime: S14ScratchLifetime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldQueueBinding {
    pub transfer_queue: vk::Queue,
    pub transfer_family: u32,
    pub compute_queue: vk::Queue,
    pub compute_family: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldTimelineBinding {
    pub transfer: vk::Semaphore,
    pub compute: vk::Semaphore,
    /// semaphore 被销毁重建时必须递增；所有异步票据都会携带它。
    pub generation: u64,
    /// 创建执行器时已经保留或提交的最大 timeline 值。
    pub initial_transfer_value: u64,
    pub initial_compute_value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldVulkanConfig {
    pub window_bytes: DeviceSize,
    pub memory_type_index: u32,
    pub queues: S14StarfoldQueueBinding,
    pub timelines: S14StarfoldTimelineBinding,
}

impl S14StarfoldVulkanConfig {
    pub const fn one_mib(
        memory_type_index: u32,
        queues: S14StarfoldQueueBinding,
        timelines: S14StarfoldTimelineBinding,
    ) -> Self {
        Self {
            window_bytes: S14_STARFOLD_ONE_MIB,
            memory_type_index,
            queues,
            timelines,
        }
    }
}

/// `memory_offset` 是相对 VkDeviceMemory 起点的绝对 bind offset；
/// `memory_span_bytes` 是该 VkBuffer 的 requirements.size，不等于逻辑有效字节数。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldBufferBindContract {
    pub role: S14StarfoldBufferRole,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub memory_offset: DeviceSize,
    pub memory_span_bytes: DeviceSize,
    pub required_alignment: DeviceSize,
    pub buffer_bytes: DeviceSize,
    pub usable_bytes: DeviceSize,
    pub memory_type_bits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldScratchBindContract {
    pub binding: S14StarfoldBufferBindContract,
    pub lifetime: S14ScratchLifetime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldTimelinePoint {
    pub semaphore: vk::Semaphore,
    pub generation: u64,
    pub value: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldCompletedTimelines {
    pub generation: u64,
    pub transfer: u64,
    pub compute: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldBufferBarrier {
    pub src_stage_mask: vk::PipelineStageFlags,
    pub dst_stage_mask: vk::PipelineStageFlags,
    pub barrier: vk::BufferMemoryBarrier<'static>,
}

impl S14StarfoldBufferBarrier {
    /// # Safety
    ///
    /// command_buffer 必须处于 recording 状态并属于合同指定的 queue family。
    pub unsafe fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                self.src_stage_mask,
                self.dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&self.barrier),
                &[],
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldUploadRecording {
    pub buffer: vk::Buffer,
    pub dst_buffer_offset: DeviceSize,
    pub byte_len: DeviceSize,
    pub wait_compute: Option<S14StarfoldTimelinePoint>,
    pub signal_transfer: S14StarfoldTimelinePoint,
    /// 复用旧驻留窗口时，在新 copy 前录制。
    pub acquire_from_compute: Option<S14StarfoldBufferBarrier>,
    /// copy 后录制；把完整窗口释放给 compute queue。
    pub release_to_compute: S14StarfoldBufferBarrier,
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldComputeRecording {
    pub buffer: vk::Buffer,
    pub buffer_offset: DeviceSize,
    pub byte_len: DeviceSize,
    pub wait_transfer: S14StarfoldTimelinePoint,
    pub signal_compute: S14StarfoldTimelinePoint,
    /// 本次驻留的第一个 consumer 在 dispatch 前录制。
    pub acquire_from_transfer: Option<S14StarfoldBufferBarrier>,
    /// 本次驻留的最后一个 consumer 在 dispatch 后录制。
    pub release_to_transfer: Option<S14StarfoldBufferBarrier>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldWindowPhase {
    Vacant,
    UploadReserved,
    Resident,
    ComputeReserved,
    Retiring,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldWindowSnapshot<K> {
    pub window: S14StarfoldWindowId,
    pub phase: S14StarfoldWindowPhase,
    pub generation: u64,
    pub key: Option<K>,
    pub byte_len: DeviceSize,
    pub total_consumers: u32,
    pub issued_consumers: u32,
    pub transfer_ready: Option<S14StarfoldTimelinePoint>,
    pub reuse_after_compute: Option<S14StarfoldTimelinePoint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldUploadTicket<K> {
    executor_id: u64,
    window: S14StarfoldWindowId,
    window_generation: u64,
    key: K,
    byte_len: DeviceSize,
    consumer_count: u32,
    signal_transfer: S14StarfoldTimelinePoint,
}

impl<K: Copy> S14StarfoldUploadTicket<K> {
    pub const fn window(self) -> S14StarfoldWindowId {
        self.window
    }

    pub const fn window_generation(self) -> u64 {
        self.window_generation
    }

    pub const fn key(self) -> K {
        self.key
    }

    pub const fn byte_len(self) -> DeviceSize {
        self.byte_len
    }

    pub const fn consumer_count(self) -> u32 {
        self.consumer_count
    }

    pub const fn signal_transfer(self) -> S14StarfoldTimelinePoint {
        self.signal_transfer
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldReadyBinding<K> {
    executor_id: u64,
    window: S14StarfoldWindowId,
    window_generation: u64,
    key: K,
    byte_len: DeviceSize,
    transfer_ready: S14StarfoldTimelinePoint,
}

impl<K: Copy> S14StarfoldReadyBinding<K> {
    pub const fn window(self) -> S14StarfoldWindowId {
        self.window
    }

    pub const fn window_generation(self) -> u64 {
        self.window_generation
    }

    pub const fn key(self) -> K {
        self.key
    }

    pub const fn byte_len(self) -> DeviceSize {
        self.byte_len
    }

    pub const fn transfer_ready(self) -> S14StarfoldTimelinePoint {
        self.transfer_ready
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldComputeTicket<K> {
    executor_id: u64,
    window: S14StarfoldWindowId,
    window_generation: u64,
    key: K,
    consumer_id: u64,
    byte_len: DeviceSize,
    transfer_ready: S14StarfoldTimelinePoint,
    signal_compute: S14StarfoldTimelinePoint,
    final_consumer: bool,
}

impl<K: Copy> S14StarfoldComputeTicket<K> {
    pub const fn window(self) -> S14StarfoldWindowId {
        self.window
    }

    pub const fn window_generation(self) -> u64 {
        self.window_generation
    }

    pub const fn key(self) -> K {
        self.key
    }

    pub const fn consumer_id(self) -> u64 {
        self.consumer_id
    }

    pub const fn byte_len(self) -> DeviceSize {
        self.byte_len
    }

    pub const fn transfer_ready(self) -> S14StarfoldTimelinePoint {
        self.transfer_ready
    }

    pub const fn signal_compute(self) -> S14StarfoldTimelinePoint {
        self.signal_compute
    }

    pub const fn is_final_consumer(self) -> bool {
        self.final_consumer
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldComputeReceipt<K> {
    pub window: S14StarfoldWindowId,
    pub window_generation: u64,
    pub key: K,
    pub consumer_id: u64,
    pub completion: S14StarfoldTimelinePoint,
    pub residency_retired: bool,
}

/// 一条 compute command 同时消费两个驻留窗口（典型用途：MXFP4 weight + scale）。
/// 两个 ticket 共用同一个 compute timeline 点，确保只提交一次 queue work。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldComputePairTicket<K> {
    tickets: [S14StarfoldComputeTicket<K>; 2],
}

impl<K: Copy> S14StarfoldComputePairTicket<K> {
    pub const fn tickets(self) -> [S14StarfoldComputeTicket<K>; 2] {
        self.tickets
    }

    pub const fn signal_compute(self) -> S14StarfoldTimelinePoint {
        self.tickets[0].signal_compute
    }
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldComputePairRecording {
    pub recordings: [S14StarfoldComputeRecording; 2],
    /// transfer timeline 是同一 semaphore；等待最大值即可覆盖两个驻留上传。
    pub wait_transfer: S14StarfoldTimelinePoint,
    pub signal_compute: S14StarfoldTimelinePoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferBindState {
    Unbound,
    Bound,
    Poisoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RetiredResidency<K> {
    key: K,
    byte_len: DeviceSize,
    reuse_after: S14StarfoldTimelinePoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActiveCompute {
    consumer_id: u64,
    signal: S14StarfoldTimelinePoint,
    final_consumer: bool,
}

#[derive(Debug)]
struct UploadState<K> {
    key: K,
    byte_len: DeviceSize,
    consumer_count: u32,
    signal_transfer: S14StarfoldTimelinePoint,
    previous: Option<RetiredResidency<K>>,
}

#[derive(Debug)]
struct ResidentState<K> {
    key: K,
    byte_len: DeviceSize,
    consumer_count: u32,
    transfer_ready: S14StarfoldTimelinePoint,
    issued_consumers: HashSet<u64>,
    active_compute: Option<ActiveCompute>,
}

#[derive(Debug)]
enum WindowState<K> {
    Vacant,
    UploadReserved(UploadState<K>),
    Resident(ResidentState<K>),
    Retiring(RetiredResidency<K>),
}

#[derive(Debug)]
struct WindowSlot<K> {
    generation: u64,
    state: WindowState<K>,
}

#[derive(Debug)]
struct TimelineAllocator {
    binding: S14StarfoldTimelineBinding,
    next_transfer: u64,
    next_compute: u64,
    submitted_transfer: u64,
    submitted_compute: u64,
}

impl TimelineAllocator {
    fn transfer_point(&mut self) -> S14StarfoldVulkanResult<S14StarfoldTimelinePoint> {
        self.next_transfer = self
            .next_transfer
            .checked_add(1)
            .ok_or(S14StarfoldVulkanError::TimelineOverflow("transfer"))?;
        Ok(S14StarfoldTimelinePoint {
            semaphore: self.binding.transfer,
            generation: self.binding.generation,
            value: self.next_transfer,
        })
    }

    fn compute_point(&mut self) -> S14StarfoldVulkanResult<S14StarfoldTimelinePoint> {
        self.next_compute = self
            .next_compute
            .checked_add(1)
            .ok_or(S14StarfoldVulkanError::TimelineOverflow("compute"))?;
        Ok(S14StarfoldTimelinePoint {
            semaphore: self.binding.compute,
            generation: self.binding.generation,
            value: self.next_compute,
        })
    }
}

/// 单 owner 状态机。所有 reserve/cancel/submit 调用必须由同一个调度 owner 串行化。
#[derive(Debug)]
pub struct S14StarfoldVulkanWindows<K> {
    executor_id: u64,
    config: S14StarfoldVulkanConfig,
    bind_state: BufferBindState,
    window_slices: [S14VulkanArenaSlice; 2],
    window_bindings: [S14StarfoldBufferBindContract; 2],
    scratch_plan: S14ScratchPlan,
    scratch_bindings: Vec<S14StarfoldScratchBindContract>,
    slots: [WindowSlot<K>; 2],
    next_window_hint: usize,
    timelines: TimelineAllocator,
}

impl<K: Copy + Eq> S14StarfoldVulkanWindows<K> {
    pub fn allocate(
        arena: &mut S14VulkanArena,
        config: S14StarfoldVulkanConfig,
        window_buffers: [S14StarfoldBufferSpec; 2],
        scratch_buffers: &[S14StarfoldScratchBufferSpec],
    ) -> S14StarfoldVulkanResult<Self> {
        validate_config(config)?;
        validate_buffer_specs(config, &window_buffers, scratch_buffers)?;

        let first = arena.allocate(
            window_buffers[0].memory_requirements.size,
            window_buffers[0].memory_requirements.alignment,
        )?;
        let second = match arena.allocate(
            window_buffers[1].memory_requirements.size,
            window_buffers[1].memory_requirements.alignment,
        ) {
            Ok(slice) => slice,
            Err(error) => {
                arena.free(first)?;
                return Err(error.into());
            }
        };

        let scratch_requests = scratch_buffers
            .iter()
            .map(|scratch| S14ScratchRequest {
                request_id: scratch.request_id,
                size: scratch.buffer.memory_requirements.size,
                alignment: scratch.buffer.memory_requirements.alignment,
                lifetime: scratch.lifetime,
            })
            .collect::<Vec<_>>();
        let scratch_plan = match arena.plan_scratch(&scratch_requests) {
            Ok(plan) => plan,
            Err(error) => {
                arena.free(second)?;
                arena.free(first)?;
                return Err(error.into());
            }
        };

        let memory = arena.binding().memory();
        let window_slices = [first, second];
        let window_bindings = std::array::from_fn(|index| S14StarfoldBufferBindContract {
            role: S14StarfoldBufferRole::Window(S14StarfoldWindowId::ALL[index]),
            buffer: window_buffers[index].buffer,
            memory,
            memory_offset: window_slices[index].offset(),
            memory_span_bytes: window_buffers[index].memory_requirements.size,
            required_alignment: window_buffers[index].memory_requirements.alignment,
            buffer_bytes: window_buffers[index].buffer_bytes,
            usable_bytes: config.window_bytes,
            memory_type_bits: window_buffers[index].memory_requirements.memory_type_bits,
        });
        let mut scratch_bindings = Vec::with_capacity(scratch_buffers.len());
        for scratch in scratch_buffers {
            let placement = scratch_plan
                .placements()
                .iter()
                .find(|placement| placement.request_id == scratch.request_id)
                .ok_or(S14StarfoldVulkanError::InternalInvariant(
                    "Arena scratch placement 缺失",
                ))?;
            scratch_bindings.push(S14StarfoldScratchBindContract {
                binding: S14StarfoldBufferBindContract {
                    role: S14StarfoldBufferRole::Scratch(scratch.request_id),
                    buffer: scratch.buffer.buffer,
                    memory,
                    memory_offset: placement.offset,
                    memory_span_bytes: scratch.buffer.memory_requirements.size,
                    required_alignment: scratch.buffer.memory_requirements.alignment,
                    buffer_bytes: scratch.buffer.buffer_bytes,
                    usable_bytes: scratch.buffer.buffer_bytes,
                    memory_type_bits: scratch.buffer.memory_requirements.memory_type_bits,
                },
                lifetime: scratch.lifetime,
            });
        }
        scratch_bindings.sort_by_key(|contract| match contract.binding.role {
            S14StarfoldBufferRole::Scratch(request_id) => request_id,
            S14StarfoldBufferRole::Window(_) => 0,
        });

        let executor_id = NEXT_EXECUTOR_ID.fetch_add(1, Ordering::Relaxed);
        if executor_id == u64::MAX {
            arena.release_scratch_plan(scratch_plan)?;
            arena.free(second)?;
            arena.free(first)?;
            return Err(S14StarfoldVulkanError::TimelineOverflow("executor id"));
        }
        Ok(Self {
            executor_id,
            config,
            bind_state: BufferBindState::Unbound,
            window_slices,
            window_bindings,
            scratch_plan,
            scratch_bindings,
            slots: [
                WindowSlot {
                    generation: 0,
                    state: WindowState::Vacant,
                },
                WindowSlot {
                    generation: 0,
                    state: WindowState::Vacant,
                },
            ],
            next_window_hint: 0,
            timelines: TimelineAllocator {
                binding: config.timelines,
                next_transfer: config.timelines.initial_transfer_value,
                next_compute: config.timelines.initial_compute_value,
                submitted_transfer: config.timelines.initial_transfer_value,
                submitted_compute: config.timelines.initial_compute_value,
            },
        })
    }

    pub const fn config(&self) -> S14StarfoldVulkanConfig {
        self.config
    }

    pub const fn window_bindings(&self) -> &[S14StarfoldBufferBindContract; 2] {
        &self.window_bindings
    }

    pub fn scratch_bindings(&self) -> &[S14StarfoldScratchBindContract] {
        &self.scratch_bindings
    }

    /// 绑定所有外部 VkBuffer。任一绑定失败后执行器进入 poisoned，不能继续提交。
    ///
    /// # Safety
    ///
    /// specs 必须来自同一个 device，且 config.memory_type_index 必须对应外部 memory。
    pub unsafe fn bind_buffers(&mut self, device: &ash::Device) -> S14StarfoldVulkanResult<()> {
        match self.bind_state {
            BufferBindState::Bound => return Err(S14StarfoldVulkanError::BuffersAlreadyBound),
            BufferBindState::Poisoned => {
                return Err(S14StarfoldVulkanError::BufferBindingPoisoned);
            }
            BufferBindState::Unbound => {}
        }
        for contract in self
            .window_bindings
            .iter()
            .chain(self.scratch_bindings.iter().map(|scratch| &scratch.binding))
        {
            if let Err(error) = unsafe {
                device.bind_buffer_memory(contract.buffer, contract.memory, contract.memory_offset)
            } {
                self.bind_state = BufferBindState::Poisoned;
                return Err(error.into());
            }
        }
        self.bind_state = BufferBindState::Bound;
        Ok(())
    }

    pub fn scratch_at(
        &self,
        request_id: u64,
        use_index: u64,
    ) -> S14StarfoldVulkanResult<S14StarfoldScratchBindContract> {
        self.ensure_bound()?;
        let contract = self
            .scratch_bindings
            .iter()
            .find(|contract| contract.binding.role == S14StarfoldBufferRole::Scratch(request_id))
            .copied()
            .ok_or(S14StarfoldVulkanError::ScratchNotFound(request_id))?;
        if !(contract.lifetime.first_use..=contract.lifetime.last_use).contains(&use_index) {
            return Err(S14StarfoldVulkanError::ScratchOutsideLifetime {
                request_id,
                use_index,
                lifetime: contract.lifetime,
            });
        }
        Ok(contract)
    }

    pub fn reserve_upload(
        &mut self,
        key: K,
        byte_len: DeviceSize,
        consumer_count: u32,
    ) -> S14StarfoldVulkanResult<(S14StarfoldUploadTicket<K>, S14StarfoldUploadRecording)> {
        let selected = [self.next_window_hint, self.next_window_hint ^ 1]
            .into_iter()
            .find(|index| {
                matches!(
                    self.slots[*index].state,
                    WindowState::Vacant | WindowState::Retiring(_)
                )
            })
            .ok_or(S14StarfoldVulkanError::NoReusableWindow)?;
        self.reserve_upload_in(
            S14StarfoldWindowId::ALL[selected],
            key,
            byte_len,
            consumer_count,
        )
    }

    pub fn reserve_upload_in(
        &mut self,
        window: S14StarfoldWindowId,
        key: K,
        byte_len: DeviceSize,
        consumer_count: u32,
    ) -> S14StarfoldVulkanResult<(S14StarfoldUploadTicket<K>, S14StarfoldUploadRecording)> {
        self.ensure_bound()?;
        if byte_len == 0 || byte_len > self.config.window_bytes {
            return Err(S14StarfoldVulkanError::InvalidUploadBytes {
                requested: byte_len,
                capacity: self.config.window_bytes,
            });
        }
        if consumer_count == 0 {
            return Err(S14StarfoldVulkanError::InvalidConsumerCount(consumer_count));
        }
        let index = window.index();
        if !matches!(
            self.slots[index].state,
            WindowState::Vacant | WindowState::Retiring(_)
        ) {
            return Err(S14StarfoldVulkanError::NoReusableWindow);
        }
        let signal_transfer = self.timelines.transfer_point()?;
        let slot = &mut self.slots[index];
        let next_generation =
            slot.generation
                .checked_add(1)
                .ok_or(S14StarfoldVulkanError::TimelineOverflow(
                    "window generation",
                ))?;
        let previous = match slot.state {
            WindowState::Vacant => None,
            WindowState::Retiring(retired) => Some(retired),
            _ => return Err(S14StarfoldVulkanError::NoReusableWindow),
        };
        slot.generation = next_generation;
        slot.state = WindowState::UploadReserved(UploadState {
            key,
            byte_len,
            consumer_count,
            signal_transfer,
            previous,
        });
        self.next_window_hint = index ^ 1;

        let binding = self.window_bindings[index];
        let wait_compute = previous.map(|retired| retired.reuse_after);
        let recording = S14StarfoldUploadRecording {
            buffer: binding.buffer,
            dst_buffer_offset: 0,
            byte_len,
            wait_compute,
            signal_transfer,
            acquire_from_compute: previous.map(|_| self.acquire_compute_to_transfer(binding)),
            release_to_compute: self.release_transfer_to_compute(binding),
        };
        Ok((
            S14StarfoldUploadTicket {
                executor_id: self.executor_id,
                window,
                window_generation: next_generation,
                key,
                byte_len,
                consumer_count,
                signal_transfer,
            },
            recording,
        ))
    }

    pub fn cancel_upload(
        &mut self,
        ticket: S14StarfoldUploadTicket<K>,
    ) -> S14StarfoldVulkanResult<()> {
        let index = self.validate_upload(ticket)?;
        let previous = match &self.slots[index].state {
            WindowState::UploadReserved(upload) => upload.previous,
            _ => return Err(S14StarfoldVulkanError::StaleUpload),
        };
        self.slots[index].state = match previous {
            Some(retired) => WindowState::Retiring(retired),
            None => WindowState::Vacant,
        };
        Ok(())
    }

    /// 校验 I/O 回调身份并异步提交 transfer command buffer；不做 host wait。
    ///
    /// # Safety
    ///
    /// command_buffer 必须已结束录制，且按 ticket 对应 recording 录制了 copy/barrier。
    pub unsafe fn submit_upload(
        &mut self,
        device: &ash::Device,
        ticket: S14StarfoldUploadTicket<K>,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> S14StarfoldVulkanResult<S14StarfoldReadyBinding<K>> {
        let index = self.validate_upload(ticket)?;
        if command_buffer == vk::CommandBuffer::null() {
            return Err(S14StarfoldVulkanError::NullHandle(
                "transfer command buffer",
            ));
        }
        let previous = match &self.slots[index].state {
            WindowState::UploadReserved(upload) => upload.previous,
            _ => return Err(S14StarfoldVulkanError::StaleUpload),
        };
        let command_buffers = [command_buffer];
        let signal_semaphores = [self.config.timelines.transfer];
        let signal_values = [ticket.signal_transfer.value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let mut submit = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let wait_semaphores;
        let wait_stages;
        let wait_values;
        if let Some(retired) = previous {
            wait_semaphores = [self.config.timelines.compute];
            wait_stages = [vk::PipelineStageFlags::TRANSFER];
            wait_values = [retired.reuse_after.value];
            timeline = timeline.wait_semaphore_values(&wait_values);
            submit = submit
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages);
        }
        submit = submit.push_next(&mut timeline);
        unsafe {
            device.queue_submit(self.config.queues.transfer_queue, &[submit], fence)?;
        }
        self.timelines.submitted_transfer = ticket.signal_transfer.value;
        self.slots[index].state = WindowState::Resident(ResidentState {
            key: ticket.key,
            byte_len: ticket.byte_len,
            consumer_count: ticket.consumer_count,
            transfer_ready: ticket.signal_transfer,
            issued_consumers: HashSet::with_capacity(ticket.consumer_count as usize),
            active_compute: None,
        });
        Ok(S14StarfoldReadyBinding {
            executor_id: self.executor_id,
            window: ticket.window,
            window_generation: ticket.window_generation,
            key: ticket.key,
            byte_len: ticket.byte_len,
            transfer_ready: ticket.signal_transfer,
        })
    }

    pub fn resident_binding(&self, key: K) -> Option<S14StarfoldReadyBinding<K>> {
        self.slots.iter().enumerate().find_map(|(index, slot)| {
            let WindowState::Resident(resident) = &slot.state else {
                return None;
            };
            (resident.key == key).then_some(S14StarfoldReadyBinding {
                executor_id: self.executor_id,
                window: S14StarfoldWindowId::ALL[index],
                window_generation: slot.generation,
                key: resident.key,
                byte_len: resident.byte_len,
                transfer_ready: resident.transfer_ready,
            })
        })
    }

    pub fn reserve_compute(
        &mut self,
        binding: S14StarfoldReadyBinding<K>,
        consumer_id: u64,
    ) -> S14StarfoldVulkanResult<(S14StarfoldComputeTicket<K>, S14StarfoldComputeRecording)> {
        self.ensure_bound()?;
        let index = self.validate_ready(binding)?;
        let resident = match &self.slots[index].state {
            WindowState::Resident(resident) => resident,
            _ => return Err(S14StarfoldVulkanError::StaleReadyBinding),
        };
        if resident.active_compute.is_some() {
            return Err(S14StarfoldVulkanError::ComputeReservationBusy);
        }
        if resident.issued_consumers.contains(&consumer_id) {
            return Err(S14StarfoldVulkanError::DuplicateConsumer(consumer_id));
        }
        if resident.issued_consumers.len() >= resident.consumer_count as usize {
            return Err(S14StarfoldVulkanError::InternalInvariant(
                "resident consumer 计数超限",
            ));
        }
        let first_consumer = resident.issued_consumers.is_empty();
        let final_consumer =
            resident.issued_consumers.len() + 1 == resident.consumer_count as usize;
        let signal_compute = self.timelines.compute_point()?;
        let resident = match &mut self.slots[index].state {
            WindowState::Resident(resident) => resident,
            _ => return Err(S14StarfoldVulkanError::StaleReadyBinding),
        };
        resident.issued_consumers.insert(consumer_id);
        resident.active_compute = Some(ActiveCompute {
            consumer_id,
            signal: signal_compute,
            final_consumer,
        });

        let buffer = self.window_bindings[index];
        let recording = S14StarfoldComputeRecording {
            buffer: buffer.buffer,
            buffer_offset: 0,
            byte_len: binding.byte_len,
            wait_transfer: binding.transfer_ready,
            signal_compute,
            acquire_from_transfer: first_consumer.then(|| self.acquire_transfer_to_compute(buffer)),
            release_to_transfer: final_consumer.then(|| self.release_compute_to_transfer(buffer)),
        };
        Ok((
            S14StarfoldComputeTicket {
                executor_id: self.executor_id,
                window: binding.window,
                window_generation: binding.window_generation,
                key: binding.key,
                consumer_id,
                byte_len: binding.byte_len,
                transfer_ready: binding.transfer_ready,
                signal_compute,
                final_consumer,
            },
            recording,
        ))
    }

    /// 为同一条 command 原子预留两个不同驻留窗口。两个窗口共享一次 compute
    /// timeline signal；任一窗口校验失败都不会改变状态。
    pub fn reserve_compute_pair(
        &mut self,
        bindings: [S14StarfoldReadyBinding<K>; 2],
        consumer_id: u64,
    ) -> S14StarfoldVulkanResult<(
        S14StarfoldComputePairTicket<K>,
        S14StarfoldComputePairRecording,
    )> {
        self.ensure_bound()?;
        let indices = [
            self.validate_ready(bindings[0])?,
            self.validate_ready(bindings[1])?,
        ];
        if indices[0] == indices[1] {
            return Err(S14StarfoldVulkanError::InternalInvariant(
                "pair compute 必须消费两个不同窗口",
            ));
        }
        for index in indices {
            let resident = match &self.slots[index].state {
                WindowState::Resident(resident) => resident,
                _ => return Err(S14StarfoldVulkanError::StaleReadyBinding),
            };
            if resident.active_compute.is_some() {
                return Err(S14StarfoldVulkanError::ComputeReservationBusy);
            }
            if resident.issued_consumers.contains(&consumer_id) {
                return Err(S14StarfoldVulkanError::DuplicateConsumer(consumer_id));
            }
            if resident.issued_consumers.len() >= resident.consumer_count as usize {
                return Err(S14StarfoldVulkanError::InternalInvariant(
                    "pair compute resident consumer 计数超限",
                ));
            }
        }

        let signal_compute = self.timelines.compute_point()?;
        let tickets = std::array::from_fn(|slot| {
            let index = indices[slot];
            let resident = match &mut self.slots[index].state {
                WindowState::Resident(resident) => resident,
                _ => unreachable!("pair compute 已预校验 resident"),
            };
            let final_consumer =
                resident.issued_consumers.len() + 1 == resident.consumer_count as usize;
            resident.issued_consumers.insert(consumer_id);
            resident.active_compute = Some(ActiveCompute {
                consumer_id,
                signal: signal_compute,
                final_consumer,
            });
            S14StarfoldComputeTicket {
                executor_id: self.executor_id,
                window: bindings[slot].window,
                window_generation: bindings[slot].window_generation,
                key: bindings[slot].key,
                consumer_id,
                byte_len: bindings[slot].byte_len,
                transfer_ready: bindings[slot].transfer_ready,
                signal_compute,
                final_consumer,
            }
        });
        let recordings = std::array::from_fn(|slot| {
            let index = indices[slot];
            let binding = self.window_bindings[index];
            let ticket = tickets[slot];
            let first_consumer = match &self.slots[index].state {
                WindowState::Resident(resident) => resident.issued_consumers.len() == 1,
                _ => false,
            };
            S14StarfoldComputeRecording {
                buffer: binding.buffer,
                buffer_offset: 0,
                byte_len: ticket.byte_len,
                wait_transfer: ticket.transfer_ready,
                signal_compute,
                acquire_from_transfer: first_consumer
                    .then(|| self.acquire_transfer_to_compute(binding)),
                release_to_transfer: ticket
                    .final_consumer
                    .then(|| self.release_compute_to_transfer(binding)),
            }
        });
        let wait_transfer = if bindings[0].transfer_ready.value >= bindings[1].transfer_ready.value
        {
            bindings[0].transfer_ready
        } else {
            bindings[1].transfer_ready
        };
        Ok((
            S14StarfoldComputePairTicket { tickets },
            S14StarfoldComputePairRecording {
                recordings,
                wait_transfer,
                signal_compute,
            },
        ))
    }

    pub fn cancel_compute(
        &mut self,
        ticket: S14StarfoldComputeTicket<K>,
    ) -> S14StarfoldVulkanResult<()> {
        let index = self.validate_compute(ticket)?;
        let resident = match &mut self.slots[index].state {
            WindowState::Resident(resident) => resident,
            _ => return Err(S14StarfoldVulkanError::StaleCompute),
        };
        resident.active_compute = None;
        if !resident.issued_consumers.remove(&ticket.consumer_id) {
            return Err(S14StarfoldVulkanError::InternalInvariant(
                "取消 compute 时 consumer 缺失",
            ));
        }
        Ok(())
    }

    pub fn cancel_compute_pair(
        &mut self,
        pair: S14StarfoldComputePairTicket<K>,
    ) -> S14StarfoldVulkanResult<()> {
        let indices = [
            self.validate_compute(pair.tickets[0])?,
            self.validate_compute(pair.tickets[1])?,
        ];
        if indices[0] == indices[1] {
            return Err(S14StarfoldVulkanError::StaleCompute);
        }
        for (slot, index) in indices.into_iter().enumerate() {
            let resident = match &mut self.slots[index].state {
                WindowState::Resident(resident) => resident,
                _ => return Err(S14StarfoldVulkanError::StaleCompute),
            };
            resident.active_compute = None;
            if !resident
                .issued_consumers
                .remove(&pair.tickets[slot].consumer_id)
            {
                return Err(S14StarfoldVulkanError::InternalInvariant(
                    "取消 pair compute 时 consumer 缺失",
                ));
            }
        }
        Ok(())
    }

    /// 异步提交一个驻留消费者。最后一个 consumer 的 command buffer 必须包含
    /// recording.release_to_transfer，之后窗口即可被下一次 upload 预订；新 upload
    /// 会在 GPU 端等待本次 compute timeline，不会阻塞 host。
    ///
    /// # Safety
    ///
    /// command_buffer 必须已结束录制，并与 ticket/recording 完全对应。
    pub unsafe fn submit_compute(
        &mut self,
        device: &ash::Device,
        ticket: S14StarfoldComputeTicket<K>,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> S14StarfoldVulkanResult<S14StarfoldComputeReceipt<K>> {
        let index = self.validate_compute(ticket)?;
        if command_buffer == vk::CommandBuffer::null() {
            return Err(S14StarfoldVulkanError::NullHandle("compute command buffer"));
        }
        let command_buffers = [command_buffer];
        let wait_semaphores = [self.config.timelines.transfer];
        let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
        let signal_semaphores = [self.config.timelines.compute];
        let wait_values = [ticket.transfer_ready.value];
        let signal_values = [ticket.signal_compute.value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        unsafe {
            device.queue_submit(self.config.queues.compute_queue, &[submit], fence)?;
        }
        self.timelines.submitted_compute = ticket.signal_compute.value;
        if ticket.final_consumer {
            self.slots[index].state = WindowState::Retiring(RetiredResidency {
                key: ticket.key,
                byte_len: ticket.byte_len,
                reuse_after: ticket.signal_compute,
            });
        } else {
            let resident = match &mut self.slots[index].state {
                WindowState::Resident(resident) => resident,
                _ => {
                    return Err(S14StarfoldVulkanError::InternalInvariant(
                        "compute submit 后 resident 消失",
                    ));
                }
            };
            resident.active_compute = None;
        }
        Ok(S14StarfoldComputeReceipt {
            window: ticket.window,
            window_generation: ticket.window_generation,
            key: ticket.key,
            consumer_id: ticket.consumer_id,
            completion: ticket.signal_compute,
            residency_retired: ticket.final_consumer,
        })
    }

    /// 异步提交一条同时引用两个窗口的 compute command。两个驻留状态只在
    /// queue_submit 成功后一起推进，避免 weight/scale 生命周期分裂。
    ///
    /// # Safety
    /// command_buffer 必须已结束录制，并包含 pair recording 中两个窗口的 barrier。
    pub unsafe fn submit_compute_pair(
        &mut self,
        device: &ash::Device,
        pair: S14StarfoldComputePairTicket<K>,
        command_buffer: vk::CommandBuffer,
        fence: vk::Fence,
    ) -> S14StarfoldVulkanResult<[S14StarfoldComputeReceipt<K>; 2]> {
        let indices = [
            self.validate_compute(pair.tickets[0])?,
            self.validate_compute(pair.tickets[1])?,
        ];
        if indices[0] == indices[1]
            || pair.tickets[0].signal_compute != pair.tickets[1].signal_compute
        {
            return Err(S14StarfoldVulkanError::StaleCompute);
        }
        if command_buffer == vk::CommandBuffer::null() {
            return Err(S14StarfoldVulkanError::NullHandle(
                "pair compute command buffer",
            ));
        }
        let wait_value = pair.tickets[0]
            .transfer_ready
            .value
            .max(pair.tickets[1].transfer_ready.value);
        let command_buffers = [command_buffer];
        let wait_semaphores = [self.config.timelines.transfer];
        let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
        let signal_semaphores = [self.config.timelines.compute];
        let wait_values = [wait_value];
        let signal_values = [pair.signal_compute().value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        unsafe {
            device.queue_submit(self.config.queues.compute_queue, &[submit], fence)?;
        }
        self.timelines.submitted_compute = pair.signal_compute().value;
        let receipts = std::array::from_fn(|slot| {
            let ticket = pair.tickets[slot];
            S14StarfoldComputeReceipt {
                window: ticket.window,
                window_generation: ticket.window_generation,
                key: ticket.key,
                consumer_id: ticket.consumer_id,
                completion: ticket.signal_compute,
                residency_retired: ticket.final_consumer,
            }
        });
        for (slot, index) in indices.into_iter().enumerate() {
            let ticket = pair.tickets[slot];
            if ticket.final_consumer {
                self.slots[index].state = WindowState::Retiring(RetiredResidency {
                    key: ticket.key,
                    byte_len: ticket.byte_len,
                    reuse_after: ticket.signal_compute,
                });
            } else {
                let resident = match &mut self.slots[index].state {
                    WindowState::Resident(resident) => resident,
                    _ => {
                        return Err(S14StarfoldVulkanError::InternalInvariant(
                            "pair compute submit 后 resident 消失",
                        ));
                    }
                };
                resident.active_compute = None;
            }
        }
        Ok(receipts)
    }

    pub fn snapshot(&self, window: S14StarfoldWindowId) -> S14StarfoldWindowSnapshot<K> {
        let slot = &self.slots[window.index()];
        let (
            phase,
            key,
            byte_len,
            total_consumers,
            issued_consumers,
            transfer_ready,
            reuse_after_compute,
        ) = match &slot.state {
            WindowState::Vacant => (S14StarfoldWindowPhase::Vacant, None, 0, 0, 0, None, None),
            WindowState::UploadReserved(upload) => (
                S14StarfoldWindowPhase::UploadReserved,
                Some(upload.key),
                upload.byte_len,
                upload.consumer_count,
                0,
                None,
                upload.previous.map(|previous| previous.reuse_after),
            ),
            WindowState::Resident(resident) => (
                if resident.active_compute.is_some() {
                    S14StarfoldWindowPhase::ComputeReserved
                } else {
                    S14StarfoldWindowPhase::Resident
                },
                Some(resident.key),
                resident.byte_len,
                resident.consumer_count,
                resident.issued_consumers.len() as u32,
                Some(resident.transfer_ready),
                None,
            ),
            WindowState::Retiring(retired) => (
                S14StarfoldWindowPhase::Retiring,
                Some(retired.key),
                retired.byte_len,
                0,
                0,
                None,
                Some(retired.reuse_after),
            ),
        };
        S14StarfoldWindowSnapshot {
            window,
            phase,
            generation: slot.generation,
            key,
            byte_len,
            total_consumers,
            issued_consumers,
            transfer_ready,
            reuse_after_compute,
        }
    }

    pub fn submitted_timelines(&self) -> S14StarfoldCompletedTimelines {
        S14StarfoldCompletedTimelines {
            generation: self.config.timelines.generation,
            transfer: self.timelines.submitted_transfer,
            compute: self.timelines.submitted_compute,
        }
    }

    /// 在外部确认两个 timeline 已完成后，把窗口与 scratch owner 归还 Arena。
    pub fn release(
        self,
        arena: &mut S14VulkanArena,
        completed: S14StarfoldCompletedTimelines,
    ) -> S14StarfoldVulkanResult<()> {
        if completed.generation != self.config.timelines.generation {
            return Err(S14StarfoldVulkanError::TimelineGenerationMismatch {
                expected: self.config.timelines.generation,
                actual: completed.generation,
            });
        }
        if completed.transfer < self.timelines.submitted_transfer
            || completed.compute < self.timelines.submitted_compute
        {
            return Err(S14StarfoldVulkanError::TimelinesIncomplete {
                required_transfer: self.timelines.submitted_transfer,
                completed_transfer: completed.transfer,
                required_compute: self.timelines.submitted_compute,
                completed_compute: completed.compute,
            });
        }
        arena.release_scratch_plan(self.scratch_plan)?;
        arena.free(self.window_slices[1])?;
        arena.free(self.window_slices[0])?;
        Ok(())
    }

    fn ensure_bound(&self) -> S14StarfoldVulkanResult<()> {
        match self.bind_state {
            BufferBindState::Bound => Ok(()),
            BufferBindState::Unbound => Err(S14StarfoldVulkanError::BuffersNotBound),
            BufferBindState::Poisoned => Err(S14StarfoldVulkanError::BufferBindingPoisoned),
        }
    }

    fn validate_upload(
        &self,
        ticket: S14StarfoldUploadTicket<K>,
    ) -> S14StarfoldVulkanResult<usize> {
        self.ensure_bound()?;
        if ticket.executor_id != self.executor_id
            || ticket.signal_transfer.generation != self.config.timelines.generation
            || ticket.signal_transfer.semaphore != self.config.timelines.transfer
        {
            return Err(S14StarfoldVulkanError::StaleUpload);
        }
        let index = ticket.window.index();
        let slot = &self.slots[index];
        let matches = match &slot.state {
            WindowState::UploadReserved(upload) => {
                slot.generation == ticket.window_generation
                    && upload.key == ticket.key
                    && upload.byte_len == ticket.byte_len
                    && upload.consumer_count == ticket.consumer_count
                    && upload.signal_transfer == ticket.signal_transfer
            }
            _ => false,
        };
        if !matches {
            return Err(S14StarfoldVulkanError::StaleUpload);
        }
        Ok(index)
    }

    fn validate_ready(
        &self,
        binding: S14StarfoldReadyBinding<K>,
    ) -> S14StarfoldVulkanResult<usize> {
        if binding.executor_id != self.executor_id
            || binding.transfer_ready.generation != self.config.timelines.generation
            || binding.transfer_ready.semaphore != self.config.timelines.transfer
        {
            return Err(S14StarfoldVulkanError::StaleReadyBinding);
        }
        let index = binding.window.index();
        let slot = &self.slots[index];
        let matches = match &slot.state {
            WindowState::Resident(resident) => {
                slot.generation == binding.window_generation
                    && resident.key == binding.key
                    && resident.byte_len == binding.byte_len
                    && resident.transfer_ready == binding.transfer_ready
            }
            _ => false,
        };
        if !matches {
            return Err(S14StarfoldVulkanError::StaleReadyBinding);
        }
        Ok(index)
    }

    fn validate_compute(
        &self,
        ticket: S14StarfoldComputeTicket<K>,
    ) -> S14StarfoldVulkanResult<usize> {
        self.ensure_bound()?;
        if ticket.executor_id != self.executor_id
            || ticket.transfer_ready.generation != self.config.timelines.generation
            || ticket.signal_compute.generation != self.config.timelines.generation
            || ticket.transfer_ready.semaphore != self.config.timelines.transfer
            || ticket.signal_compute.semaphore != self.config.timelines.compute
        {
            return Err(S14StarfoldVulkanError::StaleCompute);
        }
        let index = ticket.window.index();
        let slot = &self.slots[index];
        let matches = match &slot.state {
            WindowState::Resident(resident) => {
                slot.generation == ticket.window_generation
                    && resident.key == ticket.key
                    && resident.byte_len == ticket.byte_len
                    && resident.transfer_ready == ticket.transfer_ready
                    && resident.active_compute
                        == Some(ActiveCompute {
                            consumer_id: ticket.consumer_id,
                            signal: ticket.signal_compute,
                            final_consumer: ticket.final_consumer,
                        })
            }
            _ => false,
        };
        if !matches {
            return Err(S14StarfoldVulkanError::StaleCompute);
        }
        Ok(index)
    }

    fn release_transfer_to_compute(
        &self,
        binding: S14StarfoldBufferBindContract,
    ) -> S14StarfoldBufferBarrier {
        self.barrier(
            binding,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::empty(),
            self.config.queues.transfer_family,
            self.config.queues.compute_family,
        )
    }

    fn acquire_transfer_to_compute(
        &self,
        binding: S14StarfoldBufferBindContract,
    ) -> S14StarfoldBufferBarrier {
        self.barrier(
            binding,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
            self.config.queues.transfer_family,
            self.config.queues.compute_family,
        )
    }

    fn release_compute_to_transfer(
        &self,
        binding: S14StarfoldBufferBindContract,
    ) -> S14StarfoldBufferBarrier {
        self.barrier(
            binding,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::empty(),
            self.config.queues.compute_family,
            self.config.queues.transfer_family,
        )
    }

    fn acquire_compute_to_transfer(
        &self,
        binding: S14StarfoldBufferBindContract,
    ) -> S14StarfoldBufferBarrier {
        self.barrier(
            binding,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            self.config.queues.compute_family,
            self.config.queues.transfer_family,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn barrier(
        &self,
        binding: S14StarfoldBufferBindContract,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        src_queue_family: u32,
        dst_queue_family: u32,
    ) -> S14StarfoldBufferBarrier {
        let (src_queue_family_index, dst_queue_family_index) =
            if src_queue_family == dst_queue_family {
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
            } else {
                (src_queue_family, dst_queue_family)
            };
        S14StarfoldBufferBarrier {
            src_stage_mask,
            dst_stage_mask,
            barrier: vk::BufferMemoryBarrier::default()
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask)
                .src_queue_family_index(src_queue_family_index)
                .dst_queue_family_index(dst_queue_family_index)
                .buffer(binding.buffer)
                .offset(0)
                .size(self.config.window_bytes),
        }
    }
}

fn validate_config(config: S14StarfoldVulkanConfig) -> S14StarfoldVulkanResult<()> {
    if config.window_bytes == 0 {
        return Err(S14StarfoldVulkanError::InvalidWindowBytes(
            config.window_bytes,
        ));
    }
    if config.memory_type_index >= 32 {
        return Err(S14StarfoldVulkanError::InvalidMemoryTypeIndex(
            config.memory_type_index,
        ));
    }
    if config.queues.transfer_queue == vk::Queue::null() {
        return Err(S14StarfoldVulkanError::NullHandle("transfer queue"));
    }
    if config.queues.compute_queue == vk::Queue::null() {
        return Err(S14StarfoldVulkanError::NullHandle("compute queue"));
    }
    for family in [config.queues.transfer_family, config.queues.compute_family] {
        if family == vk::QUEUE_FAMILY_IGNORED {
            return Err(S14StarfoldVulkanError::InvalidQueueFamily(family));
        }
    }
    if config.timelines.transfer == vk::Semaphore::null() {
        return Err(S14StarfoldVulkanError::NullHandle(
            "transfer timeline semaphore",
        ));
    }
    if config.timelines.compute == vk::Semaphore::null() {
        return Err(S14StarfoldVulkanError::NullHandle(
            "compute timeline semaphore",
        ));
    }
    if config.timelines.transfer == config.timelines.compute {
        return Err(S14StarfoldVulkanError::SharedTimelineSemaphore);
    }
    if config.timelines.generation == 0 {
        return Err(S14StarfoldVulkanError::InvalidTimelineGeneration);
    }
    Ok(())
}

fn validate_buffer_specs(
    config: S14StarfoldVulkanConfig,
    windows: &[S14StarfoldBufferSpec; 2],
    scratch: &[S14StarfoldScratchBufferSpec],
) -> S14StarfoldVulkanResult<()> {
    let mut buffers = HashSet::with_capacity(2 + scratch.len());
    let mut request_ids = HashSet::with_capacity(scratch.len());
    for (index, spec) in windows.iter().enumerate() {
        validate_buffer_spec(
            config,
            S14StarfoldBufferRole::Window(S14StarfoldWindowId::ALL[index]),
            *spec,
            config.window_bytes,
        )?;
        if !buffers.insert(spec.buffer.as_raw()) {
            return Err(S14StarfoldVulkanError::DuplicateBuffer);
        }
    }
    for spec in scratch {
        if !request_ids.insert(spec.request_id) {
            return Err(S14StarfoldVulkanError::DuplicateScratchRequestId(
                spec.request_id,
            ));
        }
        validate_buffer_spec(
            config,
            S14StarfoldBufferRole::Scratch(spec.request_id),
            spec.buffer,
            spec.buffer.buffer_bytes,
        )?;
        if !buffers.insert(spec.buffer.buffer.as_raw()) {
            return Err(S14StarfoldVulkanError::DuplicateBuffer);
        }
        if spec.lifetime.first_use > spec.lifetime.last_use {
            return Err(S14VulkanArenaError::InvalidScratchLifetime {
                first_use: spec.lifetime.first_use,
                last_use: spec.lifetime.last_use,
            }
            .into());
        }
    }
    Ok(())
}

fn validate_buffer_spec(
    config: S14StarfoldVulkanConfig,
    role: S14StarfoldBufferRole,
    spec: S14StarfoldBufferSpec,
    usable_bytes: DeviceSize,
) -> S14StarfoldVulkanResult<()> {
    if spec.buffer == vk::Buffer::null() {
        return Err(S14StarfoldVulkanError::NullHandle("buffer"));
    }
    if spec.buffer_bytes == 0 || spec.buffer_bytes < usable_bytes {
        return Err(S14StarfoldVulkanError::BufferTooSmall {
            role,
            buffer_bytes: spec.buffer_bytes,
            required_bytes: usable_bytes,
        });
    }
    let requirements = spec.memory_requirements;
    if requirements.size < spec.buffer_bytes
        || requirements.alignment == 0
        || !requirements.alignment.is_power_of_two()
    {
        return Err(S14StarfoldVulkanError::InvalidMemoryRequirements {
            role,
            size: requirements.size,
            alignment: requirements.alignment,
        });
    }
    let memory_type_mask = 1_u32 << config.memory_type_index;
    if requirements.memory_type_bits & memory_type_mask == 0 {
        return Err(S14StarfoldVulkanError::IncompatibleMemoryType {
            role,
            memory_type_index: config.memory_type_index,
            memory_type_bits: requirements.memory_type_bits,
        });
    }
    Ok(())
}
