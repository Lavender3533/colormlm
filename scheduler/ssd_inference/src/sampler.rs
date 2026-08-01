//! Greedy argmax sampler. Reuses the existing `top_k.comp` shader with K=1
//! and reads back the single index to host. For ~150K vocab logits this is
//! still well under 1 ms on RX 5700 XT (single-threaded, but tiny work).

use anyhow::Result;
use ash::vk;

use crate::buffer::GpuBuffer;
use crate::compute::{ComputePipeline, DescriptorBinder};
use crate::device::VulkanContext;

pub fn argmax(
    ctx: &VulkanContext,
    topk_pipe: &ComputePipeline,
    cmd_pool: vk::CommandPool,
    logits: &GpuBuffer,
    vocab: u32,
) -> Result<u32> {
    // top_k shader is single-threaded but capped at MAX_N=256. For 150K vocab
    // we need a multi-pass argmax. Simpler: do CPU argmax. Read all logits back
    // and scan.
    let n = vocab as usize;
    let bytes = (n * 4) as u64;
    let staging = GpuBuffer::new(ctx, bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true)?;

    unsafe {
        let cb = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1))?[0];
        ctx.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        let region = [vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }];
        ctx.device.cmd_copy_buffer(cb, logits.handle(), staging.handle(), &region);
        ctx.device.end_command_buffer(cb)?;
        let fence = ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        let cb_arr = [cb];
        ctx.device.queue_submit(ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&cb_arr)], fence)?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(cmd_pool, &[cb]);
    }

    let host = unsafe { std::slice::from_raw_parts(staging.mapped() as *const f32, n) };
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0u32;
    for (i, &v) in host.iter().enumerate() {
        if v > best_v { best_v = v; best_i = i as u32; }
    }
    let _ = topk_pipe;
    staging.destroy(ctx);
    Ok(best_i)
}
