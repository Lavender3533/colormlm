//! Transfer-only bandwidth benchmark.
//!
//! Measures host→device copy throughput on three paths:
//!   1. Family 0 (graphics queue) staging copy        — what ggml-vulkan likely does
//!   2. Family 2 (dedicated transfer) staging copy    — should match #1's BW but free up graphics
//!   3. ReBAR direct write (HOST_VISIBLE | DEVICE_LOCAL) — no staging, zero-copy mmap
//!
//! Each path uploads N×64 MB to a device buffer and reports GB/s.
//!
//! Run: cargo run --release -p vulkan_xfer --example bw_transfer

use anyhow::{anyhow, bail, Result};
use ash::{vk, Entry, Instance, Device};
use std::ffi::CStr;
use std::time::Instant;

const UPLOAD_MB: usize = 64;
const N_ITERS:   usize = 32;
const BYTES:     usize = UPLOAD_MB * 1024 * 1024;

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
    let qf = instance.get_physical_device_queue_family_properties(pd);
    qf.iter().enumerate().find_map(|(i, q)| {
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
        if f.contains(must_have) && !f.intersects(must_not_have) {
            return Some(i);
        }
    }
    None
}

struct GpuBuf {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    size: u64,
    mapped: *mut u8, // null if not host-visible
}

unsafe fn alloc_buffer(
    instance: &Instance,
    pd: vk::PhysicalDevice,
    device: &Device,
    size: u64,
    usage: vk::BufferUsageFlags,
    must_have: vk::MemoryPropertyFlags,
    must_not_have: vk::MemoryPropertyFlags,
    map: bool,
) -> Result<GpuBuf> {
    let mem_props = instance.get_physical_device_memory_properties(pd);
    let bi = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buf = device.create_buffer(&bi, None)?;
    let req = device.get_buffer_memory_requirements(buf);
    let mem_type = find_memory_type(&mem_props, req.memory_type_bits, must_have, must_not_have)
        .ok_or_else(|| anyhow!("no matching memory type for must={:?}/must_not={:?}", must_have, must_not_have))?;
    let ai = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
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

fn main() -> Result<()> {
    unsafe {
        let entry = Entry::load().map_err(|e| anyhow!("load Vulkan: {e}"))?;
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let ci = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = entry.create_instance(&ci, None)?;
        let pd = pick_amd(&instance)?;

        let qf_graphics = find_queue_family(&instance, pd,
            vk::QueueFlags::GRAPHICS, vk::QueueFlags::empty())
            .ok_or_else(|| anyhow!("no graphics queue"))?;
        let qf_transfer = find_queue_family(&instance, pd,
            vk::QueueFlags::TRANSFER, vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            .ok_or_else(|| anyhow!("no dedicated transfer queue"))?;

        println!("queue family: graphics={}  dedicated_transfer={}", qf_graphics, qf_transfer);

        // Logical device with both queues
        let prio = [1.0_f32];
        let q_infos = [
            vk::DeviceQueueCreateInfo::default().queue_family_index(qf_graphics).queue_priorities(&prio),
            vk::DeviceQueueCreateInfo::default().queue_family_index(qf_transfer).queue_priorities(&prio),
        ];
        let dci = vk::DeviceCreateInfo::default().queue_create_infos(&q_infos);
        let device = instance.create_device(pd, &dci, None)?;
        let q_graphics = device.get_device_queue(qf_graphics, 0);
        let q_transfer = device.get_device_queue(qf_transfer, 0);

        // Source: host-visible coherent buffer with random data
        let host_src = alloc_buffer(&instance, pd, &device, BYTES as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL, // exclude rebar so this is plain RAM staging
            true)?;
        // Fill with bytes
        std::ptr::write_bytes(host_src.mapped, 0xAB, BYTES);

        // Dest: pure DEVICE_LOCAL (no host visibility)
        let dev_dst = alloc_buffer(&instance, pd, &device, BYTES as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            false)?;

        // ReBAR dest: DEVICE_LOCAL | HOST_VISIBLE (zero-copy direct write)
        let rebar = alloc_buffer(&instance, pd, &device, BYTES as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::empty(),
            true);
        let rebar_ok = rebar.is_ok();
        let rebar_buf = if rebar_ok { Some(rebar.unwrap()) } else { None };
        if !rebar_ok {
            println!("\n⚠️  ReBAR allocation failed — DEVICE_LOCAL|HOST_VISIBLE not available for {} MB", UPLOAD_MB);
        }

        // ───────────────────────────────────────────────────
        // Helper: bench a queue with vkCmdCopyBuffer N times
        // ───────────────────────────────────────────────────
        let bench = |label: &str, queue: vk::Queue, qfi: u32, src: &GpuBuf, dst: &GpuBuf| -> Result<f64> {
            let pool_ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(qfi)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let pool = device.create_command_pool(&pool_ci, None)?;

            let cb_alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(N_ITERS as u32);
            let cbs = device.allocate_command_buffers(&cb_alloc)?;

            let region = [vk::BufferCopy::default().size(src.size)];
            for &cb in &cbs {
                device.begin_command_buffer(cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
                device.cmd_copy_buffer(cb, src.buf, dst.buf, &region);
                device.end_command_buffer(cb)?;
            }

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            let t0 = Instant::now();
            let cb_arr: Vec<vk::CommandBuffer> = cbs.clone();
            let submit = vk::SubmitInfo::default().command_buffers(&cb_arr);
            device.queue_submit(queue, &[submit], fence)?;
            device.wait_for_fences(&[fence], true, u64::MAX)?;
            let dt = t0.elapsed().as_secs_f64();

            let total_bytes = (BYTES * N_ITERS) as f64;
            let gbs = total_bytes / dt / 1e9;
            println!("  {}  {} × {} MB in {:.2} ms = {:.2} GB/s",
                label, N_ITERS, UPLOAD_MB, dt * 1000.0, gbs);

            device.destroy_fence(fence, None);
            device.destroy_command_pool(pool, None);
            Ok(gbs)
        };

        println!("\n=== Test 1: Family 0 (graphics queue) staging copy ===");
        bench("graphics:", q_graphics, qf_graphics, &host_src, &dev_dst)?;

        println!("\n=== Test 2: Family 2 (dedicated transfer) staging copy ===");
        bench("transfer:", q_transfer, qf_transfer, &host_src, &dev_dst)?;

        if let Some(ref rb) = rebar_buf {
            println!("\n=== Test 3: ReBAR direct CPU memcpy (no GPU command) ===");
            let t0 = Instant::now();
            for _ in 0..N_ITERS {
                std::ptr::copy_nonoverlapping(host_src.mapped, rb.mapped, BYTES);
            }
            let dt = t0.elapsed().as_secs_f64();
            let gbs = (BYTES * N_ITERS) as f64 / dt / 1e9;
            println!("  rebar:    {} × {} MB memcpy in {:.2} ms = {:.2} GB/s",
                N_ITERS, UPLOAD_MB, dt * 1000.0, gbs);
        }

        // ── Cleanup ──
        if let Some(b) = rebar_buf { destroy_buffer(&device, &b); }
        destroy_buffer(&device, &dev_dst);
        destroy_buffer(&device, &host_src);
        device.destroy_device(None);
        instance.destroy_instance(None);

        println!("\nNote: Test 1 vs 2 should be similar (same PCIe). The win for");
        println!("dedicated transfer is overlap with compute, not raw BW (covered in next PoC).");
    }
    Ok(())
}
