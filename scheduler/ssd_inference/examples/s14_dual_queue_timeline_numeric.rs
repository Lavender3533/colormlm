use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    s14_dual_queue_timeline::{S14DualQueueTimeline, S14LayerTicket},
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    if !ctx.timeline_semaphore {
        bail!("当前 GPU 未提供 timeline semaphore，不能运行整 token 双队列门");
    }
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let families = [ctx.qf_transfer, ctx.qf_graphics];
    let make_shared = |bytes| {
        if ctx.qf_transfer == ctx.qf_graphics {
            GpuBuffer::new_vram(&ctx, bytes, usage)
        } else {
            GpuBuffer::new_vram_shared(&ctx, bytes, usage, &families)
        }
    };
    let pages = [make_shared(4)?, make_shared(4)?];
    let output = make_shared(16)?;
    let readback = GpuBuffer::new(
        &ctx,
        16,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;
    let values = [11u32, 22, 33, 44];
    let mut staging = Vec::with_capacity(values.len());
    for value in values {
        let buffer = GpuBuffer::new_staging(&ctx, 4)?;
        unsafe { buffer.write_at(0, &value.to_le_bytes()) };
        staging.push(buffer);
    }

    let (transfer_pool, compute_pool, transfer_commands, compute_commands) = unsafe {
        let transfer_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_transfer)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let compute_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let transfer_commands = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(transfer_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(5),
        )?;
        let compute_commands = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(compute_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(5),
        )?;
        (
            transfer_pool,
            compute_pool,
            transfer_commands,
            compute_commands,
        )
    };

    for index in 0..4usize {
        unsafe {
            ctx.device.begin_command_buffer(
                transfer_commands[index],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            ctx.device.cmd_copy_buffer(
                transfer_commands[index],
                staging[index].handle(),
                pages[index % 2].handle(),
                &[vk::BufferCopy::default().size(4)],
            );
            ctx.device.end_command_buffer(transfer_commands[index])?;

            ctx.device.begin_command_buffer(
                compute_commands[index],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            ctx.device.cmd_copy_buffer(
                compute_commands[index],
                pages[index % 2].handle(),
                output.handle(),
                &[vk::BufferCopy::default()
                    .dst_offset((index * 4) as u64)
                    .size(4)],
            );
            ctx.device.end_command_buffer(compute_commands[index])?;
        }

        if index == 3 {
            unsafe {
                // final-only compute 段不依赖新 transfer，但必须成为唯一成功等待的最终票据。
                ctx.device.begin_command_buffer(
                    compute_commands[4],
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(output.handle())
                    .offset(0)
                    .size(16);
                ctx.device.cmd_pipeline_barrier(
                    compute_commands[4],
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
                ctx.device.cmd_copy_buffer(
                    compute_commands[4],
                    output.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default().size(16)],
                );
                ctx.device.end_command_buffer(compute_commands[4])?;

                // 错误路径探针：最后只有 transfer、没有 compute 时，联合 drain 也必须闭合。
                ctx.device.begin_command_buffer(
                    transfer_commands[4],
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                ctx.device.cmd_copy_buffer(
                    transfer_commands[4],
                    staging[0].handle(),
                    pages[0].handle(),
                    &[vk::BufferCopy::default().size(4)],
                );
                ctx.device.end_command_buffer(transfer_commands[4])?;
            }
        }
    }

    let started = Instant::now();
    let mut timeline = S14DualQueueTimeline::new(&ctx)?;
    let mut tickets = Vec::<S14LayerTicket>::with_capacity(4);
    for index in 0..4usize {
        let reuse = if index >= 2 {
            Some(tickets[index - 2].compute_value)
        } else {
            None
        };
        let transfer_value =
            unsafe { timeline.submit_transfer(&ctx, transfer_commands[index], reuse)? };
        let compute_value =
            unsafe { timeline.submit_compute(&ctx, compute_commands[index], transfer_value)? };
        tickets.push(S14LayerTicket {
            transfer_value,
            compute_value,
        });
    }
    let final_compute = unsafe { timeline.submit_compute_only(&ctx, compute_commands[4])? };
    timeline.wait_compute(&ctx, final_compute, u64::MAX)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let completed = timeline.completed_values(&ctx)?;
    let observed =
        unsafe { std::slice::from_raw_parts(readback.mapped() as *const u32, 4).to_vec() };
    if observed != values {
        bail!("双队列 ping-pong 页复用错误: expected={values:?}, actual={observed:?}");
    }
    if completed != (4, 5) || final_compute != 5 {
        bail!("timeline 完成值漂移: {completed:?}");
    }

    let mut orphan = S14DualQueueTimeline::new(&ctx)?;
    let orphan_transfer = unsafe { orphan.submit_transfer(&ctx, transfer_commands[4], None)? };
    let drain = orphan.drain_all(&ctx, u64::MAX)?;
    if orphan_transfer != 1
        || drain.transfer_value != 1
        || drain.compute_value != 0
        || drain.host_wait_calls != 1
        || orphan.completed_values(&ctx)? != (1, 0)
    {
        bail!("orphan transfer drain 漂移: {drain:?}");
    }
    println!(
        "status=pass timeline=true layers=4 final_segments=1 host_waits=1 transfer_value={} compute_value={} orphan_transfer_drain=pass orphan_drain_waits={} values={observed:?} wall_ms={wall_ms:.4}",
        completed.0, completed.1
        , drain.host_wait_calls
    );

    orphan.destroy(&ctx);
    timeline.destroy(&ctx);
    unsafe {
        ctx.device.destroy_command_pool(compute_pool, None);
        ctx.device.destroy_command_pool(transfer_pool, None);
    }
    readback.destroy(&ctx);
    output.destroy(&ctx);
    for page in &pages {
        page.destroy(&ctx);
    }
    for buffer in &staging {
        buffer.destroy(&ctx);
    }
    Ok(())
}
