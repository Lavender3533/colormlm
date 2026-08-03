//! S14 Vulkan 外部显存 Arena 的纯 host 子分配原型。
//!
//! 本模块不创建、绑定或销毁 Vulkan 对象。调用方先提供一段已分配的
//! `VkDeviceMemory + base_offset + capacity`，Arena 只返回对齐后的绝对 memory offset。
//! 普通切片用不可伪造的 arena/切片编号防止跨 Arena 释放和 double-free；scratch
//! 请求还可以按闭区间生命周期做别名规划，让互不重叠的中间量复用同一物理区间。

use ash::vk::{self, Handle};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    error::Error,
    fmt,
    ops::Bound::{Excluded, Unbounded},
    sync::atomic::{AtomicU64, Ordering},
};

pub type DeviceSize = vk::DeviceSize;

static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S14VulkanArenaError {
    NullExternalMemory,
    EmptyArena,
    InvalidAlignment(DeviceSize),
    ArithmeticOverflow,
    OutOfMemory {
        requested_bytes: DeviceSize,
        alignment: DeviceSize,
        free_bytes: DeviceSize,
        largest_free_range_bytes: DeviceSize,
    },
    ForeignOrReleasedSlice,
    InvalidScratchLifetime {
        first_use: u64,
        last_use: u64,
    },
    DuplicateScratchRequestId(u64),
    InternalInvariant(&'static str),
}

impl fmt::Display for S14VulkanArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NullExternalMemory => write!(formatter, "外部 VkDeviceMemory 不能为空"),
            Self::EmptyArena => write!(formatter, "Arena capacity 必须大于零"),
            Self::InvalidAlignment(alignment) => {
                write!(formatter, "alignment 必须是非零 2 的幂，收到 {alignment}")
            }
            Self::ArithmeticOverflow => write!(formatter, "Arena offset/size 算术溢出"),
            Self::OutOfMemory {
                requested_bytes,
                alignment,
                free_bytes,
                largest_free_range_bytes,
            } => write!(
                formatter,
                "Arena 无连续空间: request={requested_bytes}, alignment={alignment}, \
                 free={free_bytes}, largest={largest_free_range_bytes}"
            ),
            Self::ForeignOrReleasedSlice => {
                write!(formatter, "切片不属于本 Arena、已释放或元数据已漂移")
            }
            Self::InvalidScratchLifetime {
                first_use,
                last_use,
            } => write!(
                formatter,
                "scratch 生命周期非法: first_use={first_use}, last_use={last_use}"
            ),
            Self::DuplicateScratchRequestId(request_id) => {
                write!(formatter, "scratch request_id 重复: {request_id}")
            }
            Self::InternalInvariant(message) => {
                write!(formatter, "Arena 内部不变量失败: {message}")
            }
        }
    }
}

impl Error for S14VulkanArenaError {}

pub type S14VulkanArenaResult<T> = Result<T, S14VulkanArenaError>;

/// 一段由上层创建并保活的 Vulkan device memory。该类型没有所有权，也没有 Drop 行为。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14ExternalDeviceMemory {
    memory: vk::DeviceMemory,
    base_offset: DeviceSize,
    capacity: DeviceSize,
}

impl S14ExternalDeviceMemory {
    pub fn bind(
        memory: vk::DeviceMemory,
        base_offset: DeviceSize,
        capacity: DeviceSize,
    ) -> S14VulkanArenaResult<Self> {
        if memory == vk::DeviceMemory::null() {
            return Err(S14VulkanArenaError::NullExternalMemory);
        }
        if capacity == 0 {
            return Err(S14VulkanArenaError::EmptyArena);
        }
        base_offset
            .checked_add(capacity)
            .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
        Ok(Self {
            memory,
            base_offset,
            capacity,
        })
    }

    pub fn memory(self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn base_offset(self) -> DeviceSize {
        self.base_offset
    }

    pub fn capacity(self) -> DeviceSize {
        self.capacity
    }
}

/// Arena 返回的物理切片。offset 是相对 `VkDeviceMemory` 起点的绝对 memory offset。
/// 字段保持私有，避免调用方拼装一个可被 `free` 接受的伪切片。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14VulkanArenaSlice {
    arena_id: u64,
    allocation_id: u64,
    memory: vk::DeviceMemory,
    offset: DeviceSize,
    size: DeviceSize,
    alignment: DeviceSize,
}

impl S14VulkanArenaSlice {
    pub fn memory(self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn offset(self) -> DeviceSize {
        self.offset
    }

    pub fn size(self) -> DeviceSize {
        self.size
    }

    pub fn alignment(self) -> DeviceSize {
        self.alignment
    }

    pub fn end_offset(self) -> DeviceSize {
        // 构造时已经验证过，不可能溢出。
        self.offset + self.size
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14VulkanArenaStats {
    pub capacity_bytes: DeviceSize,
    pub active_bytes: DeviceSize,
    pub free_bytes: DeviceSize,
    pub largest_free_range_bytes: DeviceSize,
    pub active_slices: u64,
    pub peak_active_bytes: DeviceSize,
    pub peak_active_slices: u64,
    pub total_allocations: u64,
    pub total_frees: u64,
    pub failed_allocations: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActiveAllocation {
    relative_offset: DeviceSize,
    size: DeviceSize,
    alignment: DeviceSize,
}

#[derive(Clone, Debug)]
struct FreeList {
    capacity: DeviceSize,
    ranges: BTreeMap<DeviceSize, DeviceSize>,
}

impl FreeList {
    fn new(capacity: DeviceSize) -> Self {
        Self {
            capacity,
            ranges: BTreeMap::from([(0, capacity)]),
        }
    }

    fn free_bytes(&self) -> DeviceSize {
        self.ranges.values().copied().sum()
    }

    fn largest_range(&self) -> DeviceSize {
        self.ranges.values().copied().max().unwrap_or(0)
    }

    fn allocate(
        &mut self,
        base_offset: DeviceSize,
        size: DeviceSize,
        alignment: DeviceSize,
    ) -> S14VulkanArenaResult<DeviceSize> {
        let mut selected = None;
        for (&range_offset, &range_size) in &self.ranges {
            let range_end = range_offset
                .checked_add(range_size)
                .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
            let absolute_candidate = base_offset
                .checked_add(range_offset)
                .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
            let aligned_absolute = align_up(absolute_candidate, alignment)?;
            let aligned_relative = aligned_absolute.checked_sub(base_offset).ok_or(
                S14VulkanArenaError::InternalInvariant("aligned offset 落在 Arena base 之前"),
            )?;
            let allocation_end = aligned_relative
                .checked_add(size)
                .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
            if aligned_relative >= range_offset && allocation_end <= range_end {
                selected = Some((range_offset, range_end, aligned_relative, allocation_end));
                break;
            }
        }

        let Some((range_offset, range_end, allocation_offset, allocation_end)) = selected else {
            return Err(S14VulkanArenaError::OutOfMemory {
                requested_bytes: size,
                alignment,
                free_bytes: self.free_bytes(),
                largest_free_range_bytes: self.largest_range(),
            });
        };

        self.ranges.remove(&range_offset);
        if range_offset < allocation_offset {
            self.ranges
                .insert(range_offset, allocation_offset - range_offset);
        }
        if allocation_end < range_end {
            self.ranges
                .insert(allocation_end, range_end - allocation_end);
        }
        Ok(allocation_offset)
    }

    fn release(&mut self, offset: DeviceSize, size: DeviceSize) -> S14VulkanArenaResult<()> {
        let end = offset
            .checked_add(size)
            .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
        if size == 0 || end > self.capacity {
            return Err(S14VulkanArenaError::InternalInvariant(
                "释放区间越过 Arena 边界",
            ));
        }

        let previous = self
            .ranges
            .range(..=offset)
            .next_back()
            .map(|(&start, &length)| (start, length));
        if let Some((start, length)) = previous {
            let previous_end = start
                .checked_add(length)
                .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
            if previous_end > offset {
                return Err(S14VulkanArenaError::InternalInvariant(
                    "释放区间与前一空闲区间重叠",
                ));
            }
        }
        let next = self
            .ranges
            .range((Excluded(offset), Unbounded))
            .next()
            .map(|(&start, &length)| (start, length));
        if next.is_some_and(|(start, _)| start < end) {
            return Err(S14VulkanArenaError::InternalInvariant(
                "释放区间与后一空闲区间重叠",
            ));
        }

        let mut merged_start = offset;
        let mut merged_end = end;
        if let Some((start, length)) = previous {
            if start + length == offset {
                self.ranges.remove(&start);
                merged_start = start;
            }
        }
        if let Some((start, length)) = next {
            if start == end {
                self.ranges.remove(&start);
                merged_end = start
                    .checked_add(length)
                    .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
            }
        }
        self.ranges.insert(merged_start, merged_end - merged_start);
        Ok(())
    }
}

/// 固定外部分配之上的 host-only 子分配器。它不是线程安全容器；上层 owner 应以锁或
/// 单线程调度串行化 `allocate/free/plan_scratch`。
#[derive(Debug)]
pub struct S14VulkanArena {
    arena_id: u64,
    binding: S14ExternalDeviceMemory,
    default_alignment: DeviceSize,
    next_allocation_id: u64,
    free_list: FreeList,
    active: HashMap<u64, ActiveAllocation>,
    stats: S14VulkanArenaStats,
}

impl S14VulkanArena {
    pub fn new(
        binding: S14ExternalDeviceMemory,
        default_alignment: DeviceSize,
    ) -> S14VulkanArenaResult<Self> {
        validate_alignment(default_alignment)?;
        let arena_id = NEXT_ARENA_ID.fetch_add(1, Ordering::Relaxed);
        if arena_id == u64::MAX {
            return Err(S14VulkanArenaError::ArithmeticOverflow);
        }
        Ok(Self {
            arena_id,
            binding,
            default_alignment,
            next_allocation_id: 1,
            free_list: FreeList::new(binding.capacity),
            active: HashMap::new(),
            stats: S14VulkanArenaStats {
                capacity_bytes: binding.capacity,
                free_bytes: binding.capacity,
                largest_free_range_bytes: binding.capacity,
                ..S14VulkanArenaStats::default()
            },
        })
    }

    pub fn binding(&self) -> S14ExternalDeviceMemory {
        self.binding
    }

    pub fn default_alignment(&self) -> DeviceSize {
        self.default_alignment
    }

    /// `alignment` 会与 Arena 的默认对齐取较大值，且必须是 2 的幂。
    pub fn allocate(
        &mut self,
        size: DeviceSize,
        alignment: DeviceSize,
    ) -> S14VulkanArenaResult<S14VulkanArenaSlice> {
        if size == 0 {
            return Err(S14VulkanArenaError::EmptyArena);
        }
        validate_alignment(alignment)?;
        let alignment = alignment.max(self.default_alignment);
        let allocation_id = self.next_allocation_id;
        let next_allocation_id = allocation_id
            .checked_add(1)
            .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
        let next_active_bytes = self
            .stats
            .active_bytes
            .checked_add(size)
            .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
        let next_active_slices = self
            .stats
            .active_slices
            .checked_add(1)
            .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
        let relative_offset =
            match self
                .free_list
                .allocate(self.binding.base_offset, size, alignment)
            {
                Ok(offset) => offset,
                Err(error @ S14VulkanArenaError::OutOfMemory { .. }) => {
                    self.stats.failed_allocations = self.stats.failed_allocations.saturating_add(1);
                    return Err(error);
                }
                Err(error) => return Err(error),
            };
        self.next_allocation_id = next_allocation_id;
        let allocation = ActiveAllocation {
            relative_offset,
            size,
            alignment,
        };
        if self.active.insert(allocation_id, allocation).is_some() {
            return Err(S14VulkanArenaError::InternalInvariant(
                "allocation_id 意外重复",
            ));
        }
        self.stats.active_bytes = next_active_bytes;
        self.stats.active_slices = next_active_slices;
        self.stats.peak_active_bytes = self.stats.peak_active_bytes.max(self.stats.active_bytes);
        self.stats.peak_active_slices = self.stats.peak_active_slices.max(self.stats.active_slices);
        self.stats.total_allocations = self.stats.total_allocations.saturating_add(1);
        self.refresh_free_stats();

        Ok(S14VulkanArenaSlice {
            arena_id: self.arena_id,
            allocation_id,
            memory: self.binding.memory,
            offset: self
                .binding
                .base_offset
                .checked_add(relative_offset)
                .ok_or(S14VulkanArenaError::ArithmeticOverflow)?,
            size,
            alignment,
        })
    }

    pub fn free(&mut self, slice: S14VulkanArenaSlice) -> S14VulkanArenaResult<()> {
        if slice.arena_id != self.arena_id || slice.memory != self.binding.memory {
            return Err(S14VulkanArenaError::ForeignOrReleasedSlice);
        }
        let Some(allocation) = self.active.get(&slice.allocation_id).copied() else {
            return Err(S14VulkanArenaError::ForeignOrReleasedSlice);
        };
        let expected_offset = self
            .binding
            .base_offset
            .checked_add(allocation.relative_offset)
            .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
        if slice.offset != expected_offset
            || slice.size != allocation.size
            || slice.alignment != allocation.alignment
        {
            return Err(S14VulkanArenaError::ForeignOrReleasedSlice);
        }
        self.free_list
            .release(allocation.relative_offset, allocation.size)?;
        self.active.remove(&slice.allocation_id);
        self.stats.active_bytes = self
            .stats
            .active_bytes
            .checked_sub(allocation.size)
            .ok_or(S14VulkanArenaError::InternalInvariant("active bytes 下溢"))?;
        self.stats.active_slices = self
            .stats
            .active_slices
            .checked_sub(1)
            .ok_or(S14VulkanArenaError::InternalInvariant("active slices 下溢"))?;
        self.stats.total_frees = self.stats.total_frees.saturating_add(1);
        self.refresh_free_stats();
        Ok(())
    }

    pub fn stats(&self) -> S14VulkanArenaStats {
        self.stats
    }

    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }

    /// 规划一组 scratch：生命周期是闭区间，只有 `old.last_use < new.first_use` 才允许
    /// 物理别名。规划结果在主 Arena 中只保留一个连续 owner 切片。
    pub fn plan_scratch(
        &mut self,
        requests: &[S14ScratchRequest],
    ) -> S14VulkanArenaResult<S14ScratchPlan> {
        let layout = build_scratch_layout(requests, self.binding.capacity, self.default_alignment)?;
        if layout.reserved_span_bytes == 0 {
            return Ok(S14ScratchPlan {
                arena_id: self.arena_id,
                owner: None,
                placements: Vec::new(),
                stats: layout.stats,
            });
        }
        let owner = self.allocate(layout.reserved_span_bytes, layout.max_alignment)?;
        let placements = layout
            .placements
            .into_iter()
            .map(|placement| S14ScratchPlacement {
                request_id: placement.request_id,
                memory: owner.memory,
                offset: owner.offset + placement.relative_offset,
                size: placement.size,
                alignment: placement.alignment,
                lifetime: placement.lifetime,
            })
            .collect();
        Ok(S14ScratchPlan {
            arena_id: self.arena_id,
            owner: Some(owner),
            placements,
            stats: layout.stats,
        })
    }

    pub fn release_scratch_plan(&mut self, plan: S14ScratchPlan) -> S14VulkanArenaResult<()> {
        if plan.arena_id != self.arena_id {
            return Err(S14VulkanArenaError::ForeignOrReleasedSlice);
        }
        match plan.owner {
            Some(owner) => self.free(owner),
            None => Ok(()),
        }
    }

    fn refresh_free_stats(&mut self) {
        self.stats.free_bytes = self.free_list.free_bytes();
        self.stats.largest_free_range_bytes = self.free_list.largest_range();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14ScratchLifetime {
    pub first_use: u64,
    pub last_use: u64,
}

impl S14ScratchLifetime {
    pub fn new(first_use: u64, last_use: u64) -> S14VulkanArenaResult<Self> {
        if first_use > last_use {
            return Err(S14VulkanArenaError::InvalidScratchLifetime {
                first_use,
                last_use,
            });
        }
        Ok(Self {
            first_use,
            last_use,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14ScratchRequest {
    pub request_id: u64,
    pub size: DeviceSize,
    pub alignment: DeviceSize,
    pub lifetime: S14ScratchLifetime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14ScratchPlacement {
    pub request_id: u64,
    pub memory: vk::DeviceMemory,
    pub offset: DeviceSize,
    pub size: DeviceSize,
    pub alignment: DeviceSize,
    pub lifetime: S14ScratchLifetime,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14ScratchPlanStats {
    pub logical_requests: u64,
    pub logical_bytes: DeviceSize,
    pub peak_live_requests: u64,
    pub peak_live_bytes: DeviceSize,
    pub reserved_span_bytes: DeviceSize,
}

#[derive(Debug)]
pub struct S14ScratchPlan {
    arena_id: u64,
    owner: Option<S14VulkanArenaSlice>,
    placements: Vec<S14ScratchPlacement>,
    stats: S14ScratchPlanStats,
}

impl S14ScratchPlan {
    pub fn placements(&self) -> &[S14ScratchPlacement] {
        &self.placements
    }

    pub fn stats(&self) -> S14ScratchPlanStats {
        self.stats
    }

    pub fn reserved_slice(&self) -> Option<S14VulkanArenaSlice> {
        self.owner
    }
}

#[derive(Clone, Copy, Debug)]
struct RelativeScratchPlacement {
    request_id: u64,
    relative_offset: DeviceSize,
    size: DeviceSize,
    alignment: DeviceSize,
    lifetime: S14ScratchLifetime,
}

#[derive(Debug)]
struct ScratchLayout {
    placements: Vec<RelativeScratchPlacement>,
    max_alignment: DeviceSize,
    reserved_span_bytes: DeviceSize,
    stats: S14ScratchPlanStats,
}

fn build_scratch_layout(
    requests: &[S14ScratchRequest],
    capacity: DeviceSize,
    default_alignment: DeviceSize,
) -> S14VulkanArenaResult<ScratchLayout> {
    let mut seen = HashSet::with_capacity(requests.len());
    let mut sorted = Vec::with_capacity(requests.len());
    let mut logical_bytes = 0_u64;
    let mut max_alignment = default_alignment;
    for request in requests {
        if request.size == 0 {
            return Err(S14VulkanArenaError::EmptyArena);
        }
        validate_alignment(request.alignment)?;
        if request.lifetime.first_use > request.lifetime.last_use {
            return Err(S14VulkanArenaError::InvalidScratchLifetime {
                first_use: request.lifetime.first_use,
                last_use: request.lifetime.last_use,
            });
        }
        if !seen.insert(request.request_id) {
            return Err(S14VulkanArenaError::DuplicateScratchRequestId(
                request.request_id,
            ));
        }
        logical_bytes = logical_bytes
            .checked_add(request.size)
            .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
        max_alignment = max_alignment.max(request.alignment);
        sorted.push(*request);
    }
    sorted.sort_by_key(|request| {
        (
            request.lifetime.first_use,
            request.lifetime.last_use,
            request.request_id,
        )
    });

    let mut free_list = FreeList::new(capacity);
    let mut active: Vec<RelativeScratchPlacement> = Vec::new();
    let mut placements = Vec::with_capacity(sorted.len());
    let mut live_bytes = 0_u64;
    let mut peak_live_bytes = 0_u64;
    let mut peak_live_requests = 0_u64;
    let mut reserved_span_bytes = 0_u64;

    for request in sorted {
        let mut index = 0;
        while index < active.len() {
            if active[index].lifetime.last_use < request.lifetime.first_use {
                let expired = active.swap_remove(index);
                free_list.release(expired.relative_offset, expired.size)?;
                live_bytes = live_bytes.checked_sub(expired.size).ok_or(
                    S14VulkanArenaError::InternalInvariant("scratch live bytes 下溢"),
                )?;
            } else {
                index += 1;
            }
        }

        let alignment = request.alignment.max(default_alignment);
        let relative_offset = free_list.allocate(0, request.size, alignment)?;
        let placement = RelativeScratchPlacement {
            request_id: request.request_id,
            relative_offset,
            size: request.size,
            alignment,
            lifetime: request.lifetime,
        };
        live_bytes = live_bytes
            .checked_add(request.size)
            .ok_or(S14VulkanArenaError::ArithmeticOverflow)?;
        peak_live_bytes = peak_live_bytes.max(live_bytes);
        peak_live_requests = peak_live_requests.max(active.len() as u64 + 1);
        reserved_span_bytes = reserved_span_bytes.max(
            relative_offset
                .checked_add(request.size)
                .ok_or(S14VulkanArenaError::ArithmeticOverflow)?,
        );
        active.push(placement);
        placements.push(placement);
    }

    // 恢复 request 顺序无关、按稳定 request_id 查询友好的输出。
    placements.sort_by_key(|placement| placement.request_id);
    Ok(ScratchLayout {
        placements,
        max_alignment,
        reserved_span_bytes,
        stats: S14ScratchPlanStats {
            logical_requests: requests.len() as u64,
            logical_bytes,
            peak_live_requests,
            peak_live_bytes,
            reserved_span_bytes,
        },
    })
}

fn validate_alignment(alignment: DeviceSize) -> S14VulkanArenaResult<()> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(S14VulkanArenaError::InvalidAlignment(alignment));
    }
    Ok(())
}

fn align_up(value: DeviceSize, alignment: DeviceSize) -> S14VulkanArenaResult<DeviceSize> {
    validate_alignment(alignment)?;
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(S14VulkanArenaError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(base_offset: DeviceSize, capacity: DeviceSize) -> S14ExternalDeviceMemory {
        S14ExternalDeviceMemory::bind(vk::DeviceMemory::from_raw(0x5a14), base_offset, capacity)
            .unwrap()
    }

    #[test]
    fn aligned_suballocation_and_full_coalesce() {
        let mut arena = S14VulkanArena::new(binding(3, 2048), 16).unwrap();
        let first = arena.allocate(100, 64).unwrap();
        let second = arena.allocate(200, 256).unwrap();
        assert_eq!(first.offset() % 64, 0);
        assert_eq!(second.offset() % 256, 0);
        arena.free(first).unwrap();
        arena.free(second).unwrap();
        assert!(arena.is_idle());
        assert_eq!(arena.stats().free_bytes, 2048);
        assert_eq!(arena.stats().largest_free_range_bytes, 2048);
    }

    #[test]
    fn stale_and_foreign_slices_are_rejected() {
        let memory = binding(0, 1024);
        let mut first_arena = S14VulkanArena::new(memory, 16).unwrap();
        let mut second_arena = S14VulkanArena::new(memory, 16).unwrap();
        let slice = first_arena.allocate(64, 16).unwrap();
        assert_eq!(
            second_arena.free(slice),
            Err(S14VulkanArenaError::ForeignOrReleasedSlice)
        );
        first_arena.free(slice).unwrap();
        assert_eq!(
            first_arena.free(slice),
            Err(S14VulkanArenaError::ForeignOrReleasedSlice)
        );
    }

    #[test]
    fn stats_track_active_and_peak_slices() {
        let mut arena = S14VulkanArena::new(binding(0, 1024), 1).unwrap();
        let a = arena.allocate(128, 1).unwrap();
        let b = arena.allocate(256, 1).unwrap();
        arena.free(a).unwrap();
        let stats = arena.stats();
        assert_eq!(stats.active_bytes, 256);
        assert_eq!(stats.active_slices, 1);
        assert_eq!(stats.peak_active_bytes, 384);
        assert_eq!(stats.peak_active_slices, 2);
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_frees, 1);
        arena.free(b).unwrap();
    }

    #[test]
    fn scratch_reuses_non_overlapping_lifetimes() {
        let mut arena = S14VulkanArena::new(binding(128, 4096), 64).unwrap();
        let requests = [
            S14ScratchRequest {
                request_id: 10,
                size: 512,
                alignment: 256,
                lifetime: S14ScratchLifetime::new(0, 2).unwrap(),
            },
            S14ScratchRequest {
                request_id: 20,
                size: 512,
                alignment: 256,
                lifetime: S14ScratchLifetime::new(3, 5).unwrap(),
            },
            S14ScratchRequest {
                request_id: 30,
                size: 128,
                alignment: 64,
                lifetime: S14ScratchLifetime::new(1, 4).unwrap(),
            },
        ];
        let plan = arena.plan_scratch(&requests).unwrap();
        let placement = |id| {
            plan.placements()
                .iter()
                .find(|placement| placement.request_id == id)
                .copied()
                .unwrap()
        };
        assert_eq!(placement(10).offset, placement(20).offset);
        assert_ne!(placement(10).offset, placement(30).offset);
        assert_eq!(plan.stats().logical_bytes, 1152);
        assert_eq!(plan.stats().peak_live_bytes, 640);
        assert_eq!(arena.stats().active_slices, 1);
        arena.release_scratch_plan(plan).unwrap();
        assert!(arena.is_idle());
    }

    #[test]
    fn adjacent_lifetimes_do_not_alias() {
        let mut arena = S14VulkanArena::new(binding(0, 2048), 16).unwrap();
        let requests = [
            S14ScratchRequest {
                request_id: 1,
                size: 256,
                alignment: 16,
                lifetime: S14ScratchLifetime::new(0, 1).unwrap(),
            },
            S14ScratchRequest {
                request_id: 2,
                size: 256,
                alignment: 16,
                lifetime: S14ScratchLifetime::new(1, 2).unwrap(),
            },
        ];
        let plan = arena.plan_scratch(&requests).unwrap();
        assert_ne!(plan.placements()[0].offset, plan.placements()[1].offset);
        assert_eq!(plan.stats().peak_live_requests, 2);
        arena.release_scratch_plan(plan).unwrap();
    }

    #[test]
    fn allocation_failure_preserves_free_list() {
        let mut arena = S14VulkanArena::new(binding(0, 256), 16).unwrap();
        assert!(matches!(
            arena.allocate(512, 16),
            Err(S14VulkanArenaError::OutOfMemory { .. })
        ));
        assert_eq!(arena.stats().failed_allocations, 1);
        assert_eq!(arena.stats().free_bytes, 256);
        assert_eq!(arena.allocate(256, 16).unwrap().size(), 256);
    }
}
