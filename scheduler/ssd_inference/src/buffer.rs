//! Vulkan buffer + memory allocation helpers.
//!
//! Wraps a (VkBuffer, VkDeviceMemory, optional mapped pointer) tuple with
//! RAII cleanup. Higher layers compose these into VRAM pool / staging rings.

use anyhow::{anyhow, Result};
use ash::vk;
use crate::device::VulkanContext;

pub struct GpuBuffer {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    size: u64,
    mapped: *mut u8,
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
            let mt = ctx.find_memory_type(req.memory_type_bits, must_have, must_not_have)
                .ok_or_else(|| anyhow!("no memory type matches {:?}/!{:?}",
                    must_have, must_not_have))?;
            let mem = ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            )?;
            ctx.device.bind_buffer_memory(buf, mem, 0)?;
            let mapped = if map {
                ctx.device.map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty())? as *mut u8
            } else { std::ptr::null_mut() };
            Ok(Self { buf, mem, size: req.size, mapped })
        }
    }

    /// HOST_VISIBLE | HOST_COHERENT staging buffer (plain RAM, mapped).
    pub fn new_staging(ctx: &VulkanContext, size: u64) -> Result<Self> {
        Self::new(ctx, size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, // exclude rebar (write performance bad)
            true)
    }

    /// DEVICE_LOCAL only buffer (pure VRAM).
    pub fn new_vram(ctx: &VulkanContext, size: u64, usage: vk::BufferUsageFlags) -> Result<Self> {
        Self::new(ctx, size, usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            false)
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
            let mt = ctx.find_memory_type(
                req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            ).ok_or_else(|| anyhow!("no DEVICE_LOCAL memory type"))?;
            let mem = ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt),
                None,
            )?;
            ctx.device.bind_buffer_memory(buf, mem, 0)?;
            Ok(Self { buf, mem, size: req.size, mapped: std::ptr::null_mut() })
        }
    }

    pub fn handle(&self) -> vk::Buffer { self.buf }
    pub fn size(&self) -> u64 { self.size }
    pub fn mapped(&self) -> *mut u8 { self.mapped }

    /// Copy bytes from a host slice into a mapped staging buffer at given offset.
    /// SAFETY: caller guarantees the buffer is HOST_VISIBLE (created with `map=true`).
    pub unsafe fn write_at(&self, offset: usize, src: &[u8]) {
        debug_assert!(!self.mapped.is_null(), "write_at called on non-mapped buffer");
        debug_assert!(offset + src.len() <= self.size as usize);
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.mapped.add(offset), src.len());
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            if !self.mapped.is_null() {
                ctx.device.unmap_memory(self.mem);
            }
            ctx.device.destroy_buffer(self.buf, None);
            ctx.device.free_memory(self.mem, None);
        }
    }
}
