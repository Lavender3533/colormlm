//! Vulkan instance + device + queue selection.
//!
//! Scope: pick discrete GPU, expose graphics + dedicated transfer queue,
//! plus memory type lookup. Higher layers (engine, vram_pool) consume this.

use anyhow::{anyhow, Result};
use ash::{vk, Device, Entry, Instance};
use std::ffi::CStr;

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub physical: vk::PhysicalDevice,
    pub device: Device,

    pub qf_graphics: u32,
    pub qf_transfer: u32,
    pub q_graphics: vk::Queue,
    pub q_transfer: vk::Queue,

    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub gpu_name: String,
    /// Vulkan 1.2 timeline semaphore capability enabled at device creation.
    /// The whole-token streamer requires this to overlap transfer and compute
    /// without a host fence between layers.
    pub timeline_semaphore: bool,
}

impl VulkanContext {
    pub fn init() -> Result<Self> {
        unsafe {
            let entry = Entry::load().map_err(|e| anyhow!("load Vulkan loader: {e}"))?;
            let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )?;

            // Pick first AMD/discrete GPU
            let mut chosen = None;
            for pd in instance.enumerate_physical_devices()? {
                let p = instance.get_physical_device_properties(pd);
                if p.vendor_id == 0x1002 || p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    chosen = Some(pd);
                    break;
                }
            }
            let physical = chosen.ok_or_else(|| anyhow!("no AMD/discrete GPU"))?;
            let props = instance.get_physical_device_properties(physical);
            let gpu_name = CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();

            let qf_graphics = find_queue_family(
                &instance,
                physical,
                vk::QueueFlags::GRAPHICS,
                vk::QueueFlags::empty(),
            )
            .ok_or_else(|| anyhow!("no graphics queue"))?;
            let qf_transfer = find_queue_family(
                &instance,
                physical,
                vk::QueueFlags::TRANSFER,
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            )
            .unwrap_or(qf_graphics); // fallback if no dedicated transfer

            let prio = [1.0_f32];
            let mut qis = vec![vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qf_graphics)
                .queue_priorities(&prio)];
            if qf_transfer != qf_graphics {
                qis.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(qf_transfer)
                        .queue_priorities(&prio),
                );
            }
            let mut timeline_query = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
            {
                let mut features =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut timeline_query);
                instance.get_physical_device_features2(physical, &mut features);
            }
            let timeline_semaphore = timeline_query.timeline_semaphore == vk::TRUE;
            let mut timeline_enable = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
                .timeline_semaphore(timeline_semaphore);
            let mut device_info = vk::DeviceCreateInfo::default().queue_create_infos(&qis);
            if timeline_semaphore {
                device_info = device_info.push_next(&mut timeline_enable);
            }
            let device = instance.create_device(physical, &device_info, None)?;
            let q_graphics = device.get_device_queue(qf_graphics, 0);
            let q_transfer = device.get_device_queue(qf_transfer, 0);

            let mem_props = instance.get_physical_device_memory_properties(physical);

            Ok(Self {
                entry,
                instance,
                physical,
                device,
                qf_graphics,
                qf_transfer,
                q_graphics,
                q_transfer,
                mem_props,
                gpu_name,
                timeline_semaphore,
            })
        }
    }

    pub fn has_dedicated_transfer(&self) -> bool {
        self.qf_transfer != self.qf_graphics
    }

    pub fn find_memory_type(
        &self,
        type_bits: u32,
        must_have: vk::MemoryPropertyFlags,
        must_not_have: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self.mem_props.memory_type_count {
            if (type_bits & (1 << i)) == 0 {
                continue;
            }
            let f = self.mem_props.memory_types[i as usize].property_flags;
            if f.contains(must_have) && !f.intersects(must_not_have) {
                return Some(i);
            }
        }
        None
    }

    pub fn vram_size(&self) -> u64 {
        for i in 0..self.mem_props.memory_heap_count as usize {
            let h = self.mem_props.memory_heaps[i];
            if h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                return h.size;
            }
        }
        0
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn find_queue_family(
    instance: &Instance,
    pd: vk::PhysicalDevice,
    required: vk::QueueFlags,
    forbidden: vk::QueueFlags,
) -> Option<u32> {
    instance
        .get_physical_device_queue_family_properties(pd)
        .iter()
        .enumerate()
        .find_map(|(i, q)| {
            if q.queue_flags.contains(required) && !q.queue_flags.intersects(forbidden) {
                Some(i as u32)
            } else {
                None
            }
        })
}
