//! Enumerate Vulkan queue families on this device.
//!
//! Critical for the new engine: we need a queue that supports TRANSFER
//! independently of GRAPHICS/COMPUTE so that PCIe upload can run on a
//! separate hardware queue and overlap GPU compute. AMD GCN/RDNA GPUs
//! traditionally expose a dedicated DMA queue family; Nvidia exposes one
//! too. This PoC just confirms that's true on RX 5700 XT.
//!
//! Run: cargo run --release --example list_queues -p vulkan_xfer

use anyhow::{anyhow, Result};
use ash::{vk, Entry};
use std::ffi::CStr;

fn flag_str(flags: vk::QueueFlags) -> String {
    let mut parts = Vec::new();
    if flags.contains(vk::QueueFlags::GRAPHICS)       { parts.push("GRAPHICS"); }
    if flags.contains(vk::QueueFlags::COMPUTE)        { parts.push("COMPUTE"); }
    if flags.contains(vk::QueueFlags::TRANSFER)       { parts.push("TRANSFER"); }
    if flags.contains(vk::QueueFlags::SPARSE_BINDING) { parts.push("SPARSE"); }
    if flags.contains(vk::QueueFlags::PROTECTED)      { parts.push("PROTECTED"); }
    if parts.is_empty() { "<none>".into() } else { parts.join(" | ") }
}

unsafe fn cstr(buf: &[i8]) -> &str {
    CStr::from_ptr(buf.as_ptr()).to_str().unwrap_or("<invalid utf8>")
}

fn main() -> Result<()> {
    unsafe {
        let entry = Entry::load().map_err(|e| anyhow!("load Vulkan loader: {e}"))?;

        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = entry.create_instance(&create_info, None)
            .map_err(|e| anyhow!("create instance: {e}"))?;

        let phys = instance.enumerate_physical_devices()?;
        if phys.is_empty() { return Err(anyhow!("no physical devices")); }

        for (i, pd) in phys.iter().enumerate() {
            let props = instance.get_physical_device_properties(*pd);
            let name = cstr(&props.device_name);
            let dt = match props.device_type {
                vk::PhysicalDeviceType::INTEGRATED_GPU => "iGPU",
                vk::PhysicalDeviceType::DISCRETE_GPU   => "dGPU",
                vk::PhysicalDeviceType::VIRTUAL_GPU    => "vGPU",
                vk::PhysicalDeviceType::CPU            => "CPU",
                _                                       => "other",
            };
            let api = props.api_version;
            println!("[{}] {} ({}) — Vulkan {}.{}.{} — vendor 0x{:04x} device 0x{:04x}",
                i, name, dt,
                vk::api_version_major(api),
                vk::api_version_minor(api),
                vk::api_version_patch(api),
                props.vendor_id, props.device_id);

            let qf = instance.get_physical_device_queue_family_properties(*pd);
            for (j, q) in qf.iter().enumerate() {
                let flags = q.queue_flags;
                let dedicated_transfer =
                    flags.contains(vk::QueueFlags::TRANSFER)
                    && !flags.contains(vk::QueueFlags::GRAPHICS)
                    && !flags.contains(vk::QueueFlags::COMPUTE);
                let async_compute =
                    flags.contains(vk::QueueFlags::COMPUTE)
                    && !flags.contains(vk::QueueFlags::GRAPHICS);
                let mut tags: Vec<&str> = vec![];
                if dedicated_transfer { tags.push("DEDICATED_TRANSFER ⭐"); }
                if async_compute { tags.push("ASYNC_COMPUTE ⭐"); }
                let tag_s = if tags.is_empty() { String::new() } else { format!(" [{}]", tags.join(", ")) };

                println!("    family {}  count={}  granularity={}x{}x{}  {}{}",
                    j, q.queue_count,
                    q.min_image_transfer_granularity.width,
                    q.min_image_transfer_granularity.height,
                    q.min_image_transfer_granularity.depth,
                    flag_str(flags),
                    tag_s);
            }

            // Memory heaps (so we know what RAM/VRAM looks like to Vulkan)
            let mem = instance.get_physical_device_memory_properties(*pd);
            println!("    -- {} memory heaps --", mem.memory_heap_count);
            for h in 0..mem.memory_heap_count as usize {
                let heap = mem.memory_heaps[h];
                let dev_local = heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
                println!("    heap {}  size={:.2} GB  {}",
                    h, heap.size as f64 / (1024.0 * 1024.0 * 1024.0),
                    if dev_local { "DEVICE_LOCAL (VRAM)" } else { "HOST (RAM)" });
            }
            println!("    -- {} memory types --", mem.memory_type_count);
            for t in 0..mem.memory_type_count as usize {
                let mt = mem.memory_types[t];
                let mut props_v = vec![];
                if mt.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)     { props_v.push("DEVICE_LOCAL"); }
                if mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)     { props_v.push("HOST_VISIBLE"); }
                if mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)    { props_v.push("HOST_COHERENT"); }
                if mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_CACHED)      { props_v.push("HOST_CACHED"); }
                if mt.property_flags.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED) { props_v.push("LAZY"); }
                if mt.property_flags.contains(vk::MemoryPropertyFlags::PROTECTED)        { props_v.push("PROTECTED"); }
                println!("    type {}  heap={}  flags={}", t, mt.heap_index, props_v.join(" | "));
            }
        }

        instance.destroy_instance(None);
    }
    Ok(())
}
