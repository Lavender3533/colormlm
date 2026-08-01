//! Test if transfer queue (family 2) and graphics queue (family 0) run truly
//! in parallel on hardware.
//!
//! Methodology:
//!   - Allocate big host_src (PCIe upload), big vram_a / vram_b (VRAM-only).
//!   - Workload T (transfer):  family 2 does host_src → vram_dst   (PCIe-bound)
//!   - Workload G (graphics):  family 0 does vram_a → vram_b ×N   (VRAM-bound)
//!     We tune N so G's wall-clock time alone is ~50–100 ms,
//!     i.e. comparable to T's ~80 ms.
//!
//!   Three measurements:
//!     A) T alone           wall_T
//!     B) G alone           wall_G
//!     C) T and G submitted on their respective queues at the same time
//!         If parallel:    wall_C ≈ max(wall_T, wall_G)
//!         If serial:      wall_C ≈ wall_T + wall_G
//!
//! Caveats: G uses vkCmdCopyBuffer in VRAM, which exercises the GPU's memory
//! controller (not PCIe). It is not a real compute workload, but it IS handled
//! by the graphics/compute hardware path on family 0, so concurrency with
//! family 2's DMA engine is the relevant question and gets answered.
//!
//! Run: cargo run --release -p vulkan_xfer --example parallel_qs

use anyhow::{anyhow, bail, Result};
use ash::{vk, Entry, Instance, Device};
use std::ffi::CStr;
use std::time::Instant;

const UPLOAD_MB:    usize = 64;
const UPLOAD_ITERS: usize = 32;            // 32×64MB = 2 GB transfer total
const VRAM_MB:      usize = 64;
const VRAM_ITERS:   usize = 256;           // tune to make G ~ same wall-clock as T

unsafe fn cstr(buf: &[i8]) -> &str {
    CStr::from_ptr(buf.as_ptr()).to_str().unwrap_or("?")
}

unsafe fn pick_amd(instance: &Instance) -> Result<vk::PhysicalDevice> {
    for pd in instance.enumerate_physical_devices()? {
        let p = instance.get_physical_device_properties(pd);
        if p.vendor_id == 0x1002 || p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            println!("Using GPU: {}", cstr(&p.device_name));
            return Ok(pd);
        }
    }
    bail!("no AMD/discrete GPU found");
}

unsafe fn find_queue_family(
    instance: &Instance,
    pd: vk::PhysicalDevice,
    required: vk::QueueFlags,
    forbidden: vk::QueueFlags,
) -> Option<u32> {
    instance.get_physical_device_queue_family_properties(pd).iter().enumerate()
        .find_map(|(i, q)| {
            if q.queue_flags.contains(required) && !q.queue_flags.intersects(forbidden) {
                Some(i as u32)
            } else { None }
        })
}

unsafe fn find_memory_type(
    mem: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    must_have: vk::MemoryPropertyFlags,
    must_not_have: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem.memory_type_count {
        if (type_bits & (1 << i)) == 0 { continue; }
        let f = mem.memory_types[i as usize].property_flags;
        if f.contains(must_have) && !f.intersects(must_not_have) { return Some(i); }
    }
    None
}

struct GpuBuf { buf: vk::Buffer, mem: vk::DeviceMemory, size: u64, mapped: *mut u8 }

unsafe fn alloc_buffer(
    instance: &Instance, pd: vk::PhysicalDevice, device: &Device,
    size: u64, usage: vk::BufferUsageFlags,
    must_have: vk::MemoryPropertyFlags, must_not_have: vk::MemoryPropertyFlags,
    map: bool,
) -> Result<GpuBuf> {
    let mem_props = instance.get_physical_device_memory_properties(pd);
    let bi = vk::BufferCreateInfo::default().size(size).usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf = device.create_buffer(&bi, None)?;
    let req = device.get_buffer_memory_requirements(buf);
    let mt = find_memory_type(&mem_props, req.memory_type_bits, must_have, must_not_have)
        .ok_or_else(|| anyhow!("no memory type"))?;
    let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt);
    let mem = device.allocate_memory(&ai, None)?;
    device.bind_buffer_memory(buf, mem, 0)?;
    let mapped = if map {
        device.map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty())? as *mut u8
    } else { std::ptr::null_mut() };
    Ok(GpuBuf { buf, mem, size: req.size, mapped })
}

unsafe fn destroy_buffer(device: &Device, b: &GpuBuf) {
    if !b.mapped.is_null() { device.unmap_memory(b.mem); }
    device.destroy_buffer(b.buf, None);
    device.free_memory(b.mem, None);
}

/// Build a command buffer that records `n_copies` calls of vkCmdCopyBuffer(src, dst, full size).
unsafe fn record_copy_cmd(
    device: &Device, pool: vk::CommandPool,
    src: &GpuBuf, dst: &GpuBuf, n_copies: usize,
) -> Result<vk::CommandBuffer> {
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1);
    let cb = device.allocate_command_buffers(&alloc)?[0];
    device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
    let region = [vk::BufferCopy::default().size(src.size)];
    for _ in 0..n_copies {
        device.cmd_copy_buffer(cb, src.buf, dst.buf, &region);
    }
    device.end_command_buffer(cb)?;
    Ok(cb)
}

fn main() -> Result<()> {
    unsafe {
        let entry = Entry::load().map_err(|e| anyhow!("load Vulkan: {e}"))?;
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app_info), None)?;
        let pd = pick_amd(&instance)?;

        let qf_g = find_queue_family(&instance, pd, vk::QueueFlags::GRAPHICS, vk::QueueFlags::empty())
            .ok_or_else(|| anyhow!("no graphics queue"))?;
        let qf_t = find_queue_family(&instance, pd, vk::QueueFlags::TRANSFER,
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            .ok_or_else(|| anyhow!("no dedicated transfer queue"))?;
        println!("queues: graphics={qf_g}  dedicated_transfer={qf_t}");

        let prio = [1.0_f32];
        let qi = [
            vk::DeviceQueueCreateInfo::default().queue_family_index(qf_g).queue_priorities(&prio),
            vk::DeviceQueueCreateInfo::default().queue_family_index(qf_t).queue_priorities(&prio),
        ];
        let device = instance.create_device(pd,
            &vk::DeviceCreateInfo::default().queue_create_infos(&qi), None)?;
        let q_g = device.get_device_queue(qf_g, 0);
        let q_t = device.get_device_queue(qf_t, 0);

        // host_src: plain RAM staging
        let host_src = alloc_buffer(&instance, pd, &device, (UPLOAD_MB * 1024 * 1024) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?;
        std::ptr::write_bytes(host_src.mapped, 0xCD, UPLOAD_MB * 1024 * 1024);

        // vram_dst: PCIe-upload destination (DEVICE_LOCAL only)
        let vram_dst = alloc_buffer(&instance, pd, &device, (UPLOAD_MB * 1024 * 1024) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE, false)?;

        // vram_a/vram_b: pure VRAM, used for VRAM↔VRAM copies on family 0
        let vram_a = alloc_buffer(&instance, pd, &device, (VRAM_MB * 1024 * 1024) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE, false)?;
        let vram_b = alloc_buffer(&instance, pd, &device, (VRAM_MB * 1024 * 1024) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE, false)?;

        // Pools (reset between phases — recording into TRANSIENT lifetime)
        let pool_g_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_g).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool_t_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_t).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool_g = device.create_command_pool(&pool_g_ci, None)?;
        let pool_t = device.create_command_pool(&pool_t_ci, None)?;

        let fence_g = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let fence_t = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

        // ── A) T alone (transfer queue, host→device, UPLOAD_ITERS copies) ──
        let cb_t = record_copy_cmd(&device, pool_t, &host_src, &vram_dst, UPLOAD_ITERS)?;
        let t0 = Instant::now();
        let cbs = [cb_t];
        device.queue_submit(q_t, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence_t)?;
        device.wait_for_fences(&[fence_t], true, u64::MAX)?;
        let wall_t = t0.elapsed().as_secs_f64() * 1000.0;
        let bw_t = (UPLOAD_MB * UPLOAD_ITERS) as f64 / (wall_t / 1000.0) / 1024.0;
        println!("\n[A] T alone:  {:.1} ms   ({:.2} GB/s upload, {} × {} MB)",
            wall_t, bw_t, UPLOAD_ITERS, UPLOAD_MB);
        device.reset_fences(&[fence_t])?;
        device.reset_command_pool(pool_t, vk::CommandPoolResetFlags::empty())?;

        // ── B) G alone (graphics queue, VRAM↔VRAM, VRAM_ITERS copies) ──
        let cb_g = record_copy_cmd(&device, pool_g, &vram_a, &vram_b, VRAM_ITERS)?;
        let t0 = Instant::now();
        let cbs = [cb_g];
        device.queue_submit(q_g, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence_g)?;
        device.wait_for_fences(&[fence_g], true, u64::MAX)?;
        let wall_g = t0.elapsed().as_secs_f64() * 1000.0;
        let bw_g = (VRAM_MB * VRAM_ITERS) as f64 / (wall_g / 1000.0) / 1024.0;
        println!("[B] G alone:  {:.1} ms   ({:.2} GB/s vram↔vram, {} × {} MB)",
            wall_g, bw_g, VRAM_ITERS, VRAM_MB);
        device.reset_fences(&[fence_g])?;
        device.reset_command_pool(pool_g, vk::CommandPoolResetFlags::empty())?;

        // ── C) T and G concurrently (submit to both, then wait both) ──
        let cb_t = record_copy_cmd(&device, pool_t, &host_src, &vram_dst, UPLOAD_ITERS)?;
        let cb_g = record_copy_cmd(&device, pool_g, &vram_a, &vram_b, VRAM_ITERS)?;
        let t0 = Instant::now();
        let cbs_t = [cb_t]; let cbs_g = [cb_g];
        device.queue_submit(q_t, &[vk::SubmitInfo::default().command_buffers(&cbs_t)], fence_t)?;
        device.queue_submit(q_g, &[vk::SubmitInfo::default().command_buffers(&cbs_g)], fence_g)?;
        device.wait_for_fences(&[fence_t, fence_g], true, u64::MAX)?;
        let wall_c = t0.elapsed().as_secs_f64() * 1000.0;
        println!("[C] T ‖ G:    {:.1} ms", wall_c);

        // ── Verdict ──
        let max_tg = wall_t.max(wall_g);
        let sum_tg = wall_t + wall_g;
        let parallel_score = (sum_tg - wall_c) / (sum_tg - max_tg).max(0.001);
        println!("\n=== Verdict ===");
        println!("  max(T,G) = {:.1} ms   (perfect parallel)", max_tg);
        println!("  T + G    = {:.1} ms   (fully serial)", sum_tg);
        println!("  measured = {:.1} ms", wall_c);
        println!("  parallelism score = {:.0}%   (100% = perfect, 0% = serial)",
            parallel_score * 100.0);
        if parallel_score > 0.7 {
            println!("  ✅  Transfer and graphics queues run TRULY IN PARALLEL on hardware.");
        } else if parallel_score > 0.3 {
            println!("  ⚠️  Partial overlap — there is some shared resource bottleneck.");
        } else {
            println!("  ❌  Effectively serial — hardware does NOT parallelize these workloads.");
        }

        // ── Cleanup ──
        device.destroy_fence(fence_g, None);
        device.destroy_fence(fence_t, None);
        device.destroy_command_pool(pool_g, None);
        device.destroy_command_pool(pool_t, None);
        destroy_buffer(&device, &vram_b);
        destroy_buffer(&device, &vram_a);
        destroy_buffer(&device, &vram_dst);
        destroy_buffer(&device, &host_src);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}
