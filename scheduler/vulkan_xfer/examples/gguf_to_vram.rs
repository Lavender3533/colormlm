//! End-to-end "fake 1T" expert upload throughput.
//!
//! Pipeline:
//!   GGUF mmap byte-slice → memcpy → host-visible staging buffer
//!                                     ↓ family 2 dedicated transfer queue
//!                                  VRAM expert slot pool
//!
//! Two modes:
//!   Mode A (simple):  for each expert load — memcpy → submit copy → wait fence.
//!   Mode B (pipelined): triple-buffered staging, fire copy N+1 while waiting on N.
//!
//! Reports:
//!   - Per-expert latency (avg, p50, p99)
//!   - Effective expert/s, GB/s, equivalent t/s
//!     (assuming Qwen3-30B-A3B: 8 active experts/layer × 48 layers = 384 expert loads / token)
//!
//! Run:
//!   cargo run --release -p vulkan_xfer --example gguf_to_vram \
//!     -- ../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf

use anyhow::{anyhow, bail, Result};
use ash::{vk, Entry, Instance, Device};
use gguf_reader::{ExpertKind, GgufFile};
use std::ffi::CStr;
use std::time::Instant;

const N_EXPERTS_TOTAL: u32 = 128;     // Qwen 30B-A3B
const N_LOADS_TO_TIME: usize = 4096;  // total expert loads to benchmark
const STAGING_BUFFER_MB: usize = 8;   // big enough for any single expert
const VRAM_POOL_SLOTS: usize = 16;    // ring-buffer in VRAM
const VRAM_SLOT_BYTES: usize = 2 * 1024 * 1024; // 2 MB max per expert (Q4_K_M down ~1.3MB)
const PIPELINE_DEPTH: usize = 3;      // triple-buffered staging

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
    bail!("no AMD/discrete GPU");
}
unsafe fn find_qf(instance: &Instance, pd: vk::PhysicalDevice,
    req: vk::QueueFlags, forbid: vk::QueueFlags) -> Option<u32>
{
    instance.get_physical_device_queue_family_properties(pd).iter().enumerate()
        .find_map(|(i, q)| if q.queue_flags.contains(req) && !q.queue_flags.intersects(forbid) {
            Some(i as u32) } else { None })
}
unsafe fn find_mt(mem: &vk::PhysicalDeviceMemoryProperties, bits: u32,
    must: vk::MemoryPropertyFlags, mustnt: vk::MemoryPropertyFlags) -> Option<u32>
{
    for i in 0..mem.memory_type_count {
        if (bits & (1 << i)) == 0 { continue; }
        let f = mem.memory_types[i as usize].property_flags;
        if f.contains(must) && !f.intersects(mustnt) { return Some(i); }
    }
    None
}

struct Buf { buf: vk::Buffer, mem: vk::DeviceMemory, size: u64, mapped: *mut u8 }

unsafe fn alloc(instance: &Instance, pd: vk::PhysicalDevice, device: &Device,
    size: u64, usage: vk::BufferUsageFlags,
    must: vk::MemoryPropertyFlags, mustnt: vk::MemoryPropertyFlags, map: bool) -> Result<Buf>
{
    let mp = instance.get_physical_device_memory_properties(pd);
    let buf = device.create_buffer(&vk::BufferCreateInfo::default()
        .size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
    let req = device.get_buffer_memory_requirements(buf);
    let mt = find_mt(&mp, req.memory_type_bits, must, mustnt)
        .ok_or_else(|| anyhow!("no memory type for must={:?}", must))?;
    let mem = device.allocate_memory(&vk::MemoryAllocateInfo::default()
        .allocation_size(req.size).memory_type_index(mt), None)?;
    device.bind_buffer_memory(buf, mem, 0)?;
    let mapped = if map {
        device.map_memory(mem, 0, req.size, vk::MemoryMapFlags::empty())? as *mut u8
    } else { std::ptr::null_mut() };
    Ok(Buf { buf, mem, size: req.size, mapped })
}
unsafe fn free_buf(device: &Device, b: &Buf) {
    if !b.mapped.is_null() { device.unmap_memory(b.mem); }
    device.destroy_buffer(b.buf, None);
    device.free_memory(b.mem, None);
}

/// Build a list of (layer, kind, slot, byte_size) tuples covering all experts
/// across all MoE layers. Used as the workload to drive uploads.
fn build_expert_workload(gguf: &GgufFile) -> Vec<(u32, ExpertKind, u32, usize)> {
    let mut work = Vec::new();
    for ts in gguf.list_expert_tensors() {
        let per_slot = (ts.byte_size as usize) / N_EXPERTS_TOTAL as usize;
        for slot in 0..N_EXPERTS_TOTAL {
            work.push((ts.layer, ts.kind, slot, per_slot));
        }
    }
    work
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());

    println!("=== End-to-end GGUF → VRAM upload throughput ===\n");
    println!("Opening {}", path);
    let gguf = GgufFile::open(&path)?;
    println!("  file size {:.1} MB | data_start {} | tensors {}",
        gguf.file_size() as f64 / 1024.0 / 1024.0, gguf.data_start(), gguf.n_tensors());

    let workload = build_expert_workload(&gguf);
    println!("  workload size: {} (layer, kind, slot) tuples", workload.len());
    let avg_size = workload.iter().map(|(_,_,_,s)| s).sum::<usize>() / workload.len();
    println!("  avg expert byte size: {} ({:.1} KB)", avg_size, avg_size as f64 / 1024.0);

    // ── Vulkan init ──
    unsafe {
        let entry = Entry::load().map_err(|e| anyhow!("load Vulkan: {e}"))?;
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default()
            .application_info(&vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2)), None)?;
        let pd = pick_amd(&instance)?;
        let qf_t = find_qf(&instance, pd, vk::QueueFlags::TRANSFER,
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            .ok_or_else(|| anyhow!("no dedicated transfer queue"))?;
        println!("\n  dedicated transfer queue family: {qf_t}");

        let prio = [1.0f32];
        let qi = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf_t).queue_priorities(&prio)];
        let device = instance.create_device(pd,
            &vk::DeviceCreateInfo::default().queue_create_infos(&qi), None)?;
        let queue = device.get_device_queue(qf_t, 0);

        // Staging buffers (PIPELINE_DEPTH copies)
        let mut stagings: Vec<Buf> = Vec::with_capacity(PIPELINE_DEPTH);
        for _ in 0..PIPELINE_DEPTH {
            stagings.push(alloc(&instance, pd, &device,
                (STAGING_BUFFER_MB * 1024 * 1024) as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL, true)?);
        }
        // VRAM expert pool
        let vram_pool = alloc(&instance, pd, &device,
            (VRAM_POOL_SLOTS * VRAM_SLOT_BYTES) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE, false)?;
        println!("  staging × {}: {} MB ea | vram pool: {} slots × {} MB = {} MB",
            PIPELINE_DEPTH, STAGING_BUFFER_MB, VRAM_POOL_SLOTS,
            VRAM_SLOT_BYTES / 1024 / 1024,
            VRAM_POOL_SLOTS * VRAM_SLOT_BYTES / 1024 / 1024);

        let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_t)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)?;
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool).level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(PIPELINE_DEPTH as u32);
        let cbs = device.allocate_command_buffers(&cb_alloc)?;
        let mut fences: Vec<vk::Fence> = Vec::with_capacity(PIPELINE_DEPTH);
        for _ in 0..PIPELINE_DEPTH {
            fences.push(device.create_fence(&vk::FenceCreateInfo::default()
                .flags(vk::FenceCreateFlags::SIGNALED), None)?); // signaled = "free"
        }

        // Pre-warm OS page cache for first 64 MB so first iters aren't disk-bound
        let _ = gguf.tensor_bytes("blk.0.ffn_gate_exps.weight").map(|b| {
            let mut s: u64 = 0;
            for c in b.chunks(4096) { s = s.wrapping_add(c[0] as u64); }
            std::hint::black_box(s);
        });

        // ── Mode A: simple — per-load: memcpy → submit → wait ──
        run_bench("Mode A (simple, sync per load)", &workload, &gguf, &device, queue,
                  &stagings, &cbs, &fences, &vram_pool, /*pipeline=*/ 1)?;

        // ── Mode B: pipelined depth-3 ──
        run_bench("Mode B (pipelined depth-3)", &workload, &gguf, &device, queue,
                  &stagings, &cbs, &fences, &vram_pool, /*pipeline=*/ PIPELINE_DEPTH)?;

        // ── Cleanup ──
        for f in &fences { device.destroy_fence(*f, None); }
        device.destroy_command_pool(pool, None);
        free_buf(&device, &vram_pool);
        for s in &stagings { free_buf(&device, s); }
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn run_bench(
    label: &str,
    workload: &[(u32, ExpertKind, u32, usize)],
    gguf: &GgufFile,
    device: &Device,
    queue: vk::Queue,
    stagings: &[Buf],
    cbs: &[vk::CommandBuffer],
    fences: &[vk::Fence],
    vram_pool: &Buf,
    pipeline: usize,
) -> Result<()> {
    let mut latencies_us: Vec<u64> = Vec::with_capacity(N_LOADS_TO_TIME);
    let mut total_bytes: usize = 0;

    // All fences should already be signaled (created with SIGNALED flag).
    // Loop invariant: each fence is signaled when its slot is "free".

    let t_start = Instant::now();

    for i in 0..N_LOADS_TO_TIME {
        let (layer, kind, slot, _sz) = workload[i % workload.len()];
        let bytes = gguf.expert_slot_bytes(layer, kind, slot, N_EXPERTS_TOTAL)?;
        let vram_slot = i % VRAM_POOL_SLOTS;
        let pip = i % pipeline;

        let iter_t0 = Instant::now();

        // Wait for the slot we're about to reuse to be free
        device.wait_for_fences(&[fences[pip]], true, u64::MAX)?;
        device.reset_fences(&[fences[pip]])?;

        // memcpy gguf bytes → staging
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), stagings[pip].mapped, bytes.len());

        // record cmd: copy staging[pip] → vram_pool[slot]
        let cb = cbs[pip];
        device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let region = [vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset((vram_slot * VRAM_SLOT_BYTES) as u64)
            .size(bytes.len() as u64)];
        device.cmd_copy_buffer(cb, stagings[pip].buf, vram_pool.buf, &region);
        device.end_command_buffer(cb)?;

        let cb_arr = [cb];
        device.queue_submit(queue,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fences[pip])?;

        // For pipeline=1, sync immediately so the per-load timing is accurate.
        // For pipeline>1, do NOT wait — let the next iter's top of loop wait
        // when this fence comes around again (PIPELINE_DEPTH iters later).
        if pipeline == 1 {
            device.wait_for_fences(&[fences[pip]], true, u64::MAX)?;
            // fence is now signaled; loop top will see it signaled, reset, submit again.
        }

        latencies_us.push(iter_t0.elapsed().as_micros() as u64);
        total_bytes += bytes.len();
    }

    // Drain
    for &f in fences { device.wait_for_fences(&[f], true, u64::MAX)?; }
    let total_s = t_start.elapsed().as_secs_f64();

    // Re-signal all fences for next benchmark run (so next caller sees free slots)
    for &f in fences {
        // submit empty batch with this fence to leave it signaled
        device.queue_submit(queue, &[vk::SubmitInfo::default()], f)?;
    }
    for &f in fences { device.wait_for_fences(&[f], true, u64::MAX)?; }

    latencies_us.sort();
    let p50 = latencies_us[latencies_us.len() / 2];
    let p99 = latencies_us[latencies_us.len() * 99 / 100];
    let avg = latencies_us.iter().sum::<u64>() / latencies_us.len() as u64;

    let throughput_gbs = total_bytes as f64 / total_s / 1e9;
    let throughput_eps = N_LOADS_TO_TIME as f64 / total_s;
    let experts_per_token = 8.0 * 48.0 * 3.0; // gate + up + down per layer
    let equiv_tps = throughput_eps / experts_per_token;

    println!("\n--- {} ---", label);
    println!("  total time:   {:.2} s   loads {} ({:.1} MB)",
        total_s, N_LOADS_TO_TIME, total_bytes as f64 / 1024.0 / 1024.0);
    println!("  per-load:     avg {} µs | p50 {} µs | p99 {} µs",
        avg, p50, p99);
    println!("  throughput:   {:.0} loads/s | {:.2} GB/s",
        throughput_eps, throughput_gbs);
    println!("  equivalent t/s (8×48×3 expert loads/token): {:.1}", equiv_tps);

    Ok(())
}
