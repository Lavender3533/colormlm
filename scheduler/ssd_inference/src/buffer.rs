//! Vulkan buffer + memory allocation helpers.
//!
//! Wraps a (VkBuffer, VkDeviceMemory, optional mapped pointer) tuple with
//! RAII cleanup. Higher layers compose these into VRAM pool / staging rings.

use crate::device::VulkanContext;
use anyhow::{anyhow, Result};
use ash::vk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Process-wide Vulkan buffer allocation telemetry.
///
/// The live fields describe buffers whose constructors completed successfully
/// and whose `destroy` method has not yet run.  The maximum and total fields are
/// monotonic for the lifetime of the process.  This is deliberately a small,
/// lock-free diagnostic ledger rather than a general-purpose allocator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuBufferAllocationSnapshot {
    pub live_allocated_bytes: u64,
    pub live_device_local_bytes: u64,
    pub live_host_visible_system_bytes: u64,
    pub live_other_bytes: u64,
    pub live_buffer_count: u64,
    pub max_single_allocation: u64,
    pub total_allocations: u64,
}

/// Vulkan allocation 的物理层级。DEVICE_LOCAL 视为 L1；非 DEVICE_LOCAL 的
/// HOST_VISIBLE heap 视为可被 GPU 直接访问的系统内存 L2。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBufferMemoryTier {
    DeviceLocal,
    HostVisibleSystem,
    Other,
}

impl GpuBufferMemoryTier {
    fn from_properties(properties: vk::MemoryPropertyFlags) -> Self {
        if properties.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            Self::DeviceLocal
        } else if properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            Self::HostVisibleSystem
        } else {
            Self::Other
        }
    }
}

struct AllocationLedger {
    live_allocated_bytes: AtomicU64,
    live_device_local_bytes: AtomicU64,
    live_host_visible_system_bytes: AtomicU64,
    live_other_bytes: AtomicU64,
    live_buffer_count: AtomicU64,
    max_single_allocation: AtomicU64,
    total_allocations: AtomicU64,
}

impl AllocationLedger {
    const fn new() -> Self {
        Self {
            live_allocated_bytes: AtomicU64::new(0),
            live_device_local_bytes: AtomicU64::new(0),
            live_host_visible_system_bytes: AtomicU64::new(0),
            live_other_bytes: AtomicU64::new(0),
            live_buffer_count: AtomicU64::new(0),
            max_single_allocation: AtomicU64::new(0),
            total_allocations: AtomicU64::new(0),
        }
    }

    fn register(&self, allocation_size: u64, tier: GpuBufferMemoryTier) {
        self.live_allocated_bytes
            .fetch_add(allocation_size, Ordering::Relaxed);
        self.tier_counter(tier)
            .fetch_add(allocation_size, Ordering::Relaxed);
        self.live_buffer_count.fetch_add(1, Ordering::Relaxed);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.max_single_allocation
            .fetch_max(allocation_size, Ordering::Relaxed);
    }

    fn release(&self, allocation_size: u64, tier: GpuBufferMemoryTier) {
        let previous_bytes = self
            .live_allocated_bytes
            .fetch_sub(allocation_size, Ordering::Relaxed);
        let previous_count = self.live_buffer_count.fetch_sub(1, Ordering::Relaxed);
        let previous_tier_bytes = self
            .tier_counter(tier)
            .fetch_sub(allocation_size, Ordering::Relaxed);
        debug_assert!(previous_bytes >= allocation_size);
        debug_assert!(previous_tier_bytes >= allocation_size);
        debug_assert!(previous_count >= 1);
    }

    fn snapshot(&self) -> GpuBufferAllocationSnapshot {
        GpuBufferAllocationSnapshot {
            live_allocated_bytes: self.live_allocated_bytes.load(Ordering::Relaxed),
            live_device_local_bytes: self.live_device_local_bytes.load(Ordering::Relaxed),
            live_host_visible_system_bytes: self
                .live_host_visible_system_bytes
                .load(Ordering::Relaxed),
            live_other_bytes: self.live_other_bytes.load(Ordering::Relaxed),
            live_buffer_count: self.live_buffer_count.load(Ordering::Relaxed),
            max_single_allocation: self.max_single_allocation.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
        }
    }

    fn tier_counter(&self, tier: GpuBufferMemoryTier) -> &AtomicU64 {
        match tier {
            GpuBufferMemoryTier::DeviceLocal => &self.live_device_local_bytes,
            GpuBufferMemoryTier::HostVisibleSystem => &self.live_host_visible_system_bytes,
            GpuBufferMemoryTier::Other => &self.live_other_bytes,
        }
    }
}

static GPU_BUFFER_ALLOCATION_LEDGER: AllocationLedger = AllocationLedger::new();

/// Return the current process-wide buffer allocation telemetry.
pub fn gpu_buffer_allocation_snapshot() -> GpuBufferAllocationSnapshot {
    GPU_BUFFER_ALLOCATION_LEDGER.snapshot()
}

fn release_accounting_once(
    ledger: &AllocationLedger,
    allocation_size: u64,
    tier: GpuBufferMemoryTier,
    destroyed: &AtomicBool,
) -> bool {
    if destroyed
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return false;
    }
    ledger.release(allocation_size, tier);
    true
}

fn allocation_error(
    constructor: &str,
    stage: &str,
    error: vk::Result,
    requested_logical_size: u64,
    requirement_size: u64,
    memory_type: Option<u32>,
) -> anyhow::Error {
    anyhow!(error).context(format!(
        "{constructor} {stage} failed: requested_logical_size={requested_logical_size}, \
         requirement_size={requirement_size}, memory_type={memory_type:?}, current_snapshot={:?}",
        gpu_buffer_allocation_snapshot()
    ))
}

pub struct GpuBuffer {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    size: u64,
    mapped: *mut u8,
    memory_tier: GpuBufferMemoryTier,
    destroyed: AtomicBool,
}

impl GpuBuffer {
    /// Create a buffer + back it with memory matching the given property flags.
    /// `map = true` keeps the memory persistently mapped (only meaningful for
    /// HOST_VISIBLE memory).
    pub fn new(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        must_have: vk::MemoryPropertyFlags,
        must_not_have: vk::MemoryPropertyFlags,
        map: bool,
    ) -> Result<Self> {
        unsafe {
            let buf = ctx.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = ctx.device.get_buffer_memory_requirements(buf);
            let mt = match ctx.find_memory_type(req.memory_type_bits, must_have, must_not_have) {
                Some(mt) => mt,
                None => {
                    ctx.device.destroy_buffer(buf, None);
                    return Err(anyhow!(
                        "GpuBuffer::new memory-type selection failed: \
                         requested_logical_size={size}, requirement_size={}, memory_type=None, \
                         required={must_have:?}, excluded={must_not_have:?}, current_snapshot={:?}",
                        req.size,
                        gpu_buffer_allocation_snapshot()
                    ));
                }
            };
            let mem = match ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            ) {
                Ok(mem) => mem,
                Err(error) => {
                    ctx.device.destroy_buffer(buf, None);
                    return Err(allocation_error(
                        "GpuBuffer::new",
                        "allocate_memory",
                        error,
                        size,
                        req.size,
                        Some(mt),
                    ));
                }
            };
            if let Err(error) = ctx.device.bind_buffer_memory(buf, mem, 0) {
                ctx.device.destroy_buffer(buf, None);
                ctx.device.free_memory(mem, None);
                return Err(allocation_error(
                    "GpuBuffer::new",
                    "bind_buffer_memory",
                    error,
                    size,
                    req.size,
                    Some(mt),
                ));
            }
            let mapped = if map {
                match ctx
                    .device
                    .map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty())
                {
                    Ok(mapped) => mapped as *mut u8,
                    Err(error) => {
                        ctx.device.destroy_buffer(buf, None);
                        ctx.device.free_memory(mem, None);
                        return Err(allocation_error(
                            "GpuBuffer::new",
                            "map_memory",
                            error,
                            size,
                            req.size,
                            Some(mt),
                        ));
                    }
                }
            } else {
                std::ptr::null_mut()
            };
            let memory_tier = GpuBufferMemoryTier::from_properties(
                ctx.mem_props.memory_types[mt as usize].property_flags,
            );
            GPU_BUFFER_ALLOCATION_LEDGER.register(req.size, memory_tier);
            Ok(Self {
                buf,
                mem,
                size: req.size,
                mapped,
                memory_tier,
                destroyed: AtomicBool::new(false),
            })
        }
    }

    /// HOST_VISIBLE | HOST_COHERENT staging buffer (plain RAM, mapped).
    pub fn new_staging(ctx: &VulkanContext, size: u64) -> Result<Self> {
        Self::new(
            ctx,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, // exclude rebar (write performance bad)
            true,
        )
    }

    /// DEVICE_LOCAL only buffer (pure VRAM).
    pub fn new_vram(ctx: &VulkanContext, size: u64, usage: vk::BufferUsageFlags) -> Result<Self> {
        Self::new(
            ctx,
            size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            false,
        )
    }

    /// GPU-addressable buffer backed by non-device-local HOST_VISIBLE memory.
    ///
    /// This is not a CPU staging-only allocation: callers may include
    /// `STORAGE_BUFFER` and use it directly from compute shaders.  It is the
    /// explicit L2 tier for large, low-reuse state that must remain visible to
    /// Vulkan without consuming the discrete GPU's DEVICE_LOCAL heap.
    pub fn new_host_storage(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::new(
            ctx,
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
        )
    }

    /// 大型 GPU 可寻址事务 buffer 的统一两级分配入口。先使用 DEVICE_LOCAL L1；
    /// 只有真实分配失败时才退让到 HOST_VISIBLE_SYSTEM L2，数值 ABI 与 usage 不变。
    pub fn new_device_first_storage(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        match Self::new_vram(ctx, size, usage) {
            Ok(buffer) => Ok(buffer),
            Err(device_local_error) => {
                Self::new_host_storage(ctx, size, usage).map_err(|host_error| {
                    anyhow!(
                        "DEVICE_LOCAL 与 HOST_VISIBLE_SYSTEM 两级分配均失败: \
                     device_local_error={device_local_error:#}; host_error={host_error:#}"
                    )
                })
            }
        }
    }

    /// DEVICE_LOCAL buffer with CONCURRENT sharing across multiple queue families.
    pub fn new_vram_shared(
        ctx: &VulkanContext,
        size: u64,
        usage: vk::BufferUsageFlags,
        queue_families: &[u32],
    ) -> Result<Self> {
        unsafe {
            let buf = ctx.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(queue_families),
                None,
            )?;
            let req = ctx.device.get_buffer_memory_requirements(buf);
            let mt = match ctx.find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            ) {
                Some(mt) => mt,
                None => {
                    ctx.device.destroy_buffer(buf, None);
                    return Err(anyhow!(
                        "GpuBuffer::new_vram_shared memory-type selection failed: \
                         requested_logical_size={size}, requirement_size={}, memory_type=None, \
                         current_snapshot={:?}",
                        req.size,
                        gpu_buffer_allocation_snapshot()
                    ));
                }
            };
            let mem = match ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            ) {
                Ok(mem) => mem,
                Err(error) => {
                    ctx.device.destroy_buffer(buf, None);
                    return Err(allocation_error(
                        "GpuBuffer::new_vram_shared",
                        "allocate_memory",
                        error,
                        size,
                        req.size,
                        Some(mt),
                    ));
                }
            };
            if let Err(error) = ctx.device.bind_buffer_memory(buf, mem, 0) {
                ctx.device.destroy_buffer(buf, None);
                ctx.device.free_memory(mem, None);
                return Err(allocation_error(
                    "GpuBuffer::new_vram_shared",
                    "bind_buffer_memory",
                    error,
                    size,
                    req.size,
                    Some(mt),
                ));
            }
            let memory_tier = GpuBufferMemoryTier::from_properties(
                ctx.mem_props.memory_types[mt as usize].property_flags,
            );
            GPU_BUFFER_ALLOCATION_LEDGER.register(req.size, memory_tier);
            Ok(Self {
                buf,
                mem,
                size: req.size,
                mapped: std::ptr::null_mut(),
                memory_tier,
                destroyed: AtomicBool::new(false),
            })
        }
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buf
    }
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn mapped(&self) -> *mut u8 {
        self.mapped
    }
    pub fn memory_tier(&self) -> GpuBufferMemoryTier {
        self.memory_tier
    }

    /// Copy bytes from a host slice into a mapped staging buffer at given offset.
    /// SAFETY: caller guarantees the buffer is HOST_VISIBLE (created with `map=true`).
    pub unsafe fn write_at(&self, offset: usize, src: &[u8]) {
        debug_assert!(
            !self.mapped.is_null(),
            "write_at called on non-mapped buffer"
        );
        debug_assert!(offset + src.len() <= self.size as usize);
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.mapped.add(offset), src.len());
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        if !release_accounting_once(
            &GPU_BUFFER_ALLOCATION_LEDGER,
            self.size,
            self.memory_tier,
            &self.destroyed,
        ) {
            return;
        }
        unsafe {
            if !self.mapped.is_null() {
                ctx.device.unmap_memory(self.mem);
            }
            ctx.device.destroy_buffer(self.buf, None);
            ctx.device.free_memory(self.mem, None);
        }
    }
}
