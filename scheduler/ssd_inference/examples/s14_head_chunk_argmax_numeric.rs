//! K=4 BF16 head chunk + 跨 chunk GPU argmax 的最小 synthetic 数值门。
//!
//! 使用小 shape，但保持生产 ABI：head arena 非零 offset、三个临时量位于
//! 单一 workspace 的不同子范围、最后一块为非整块。正路径必须与 CPU top-1
//! 一致；NaN、乱序、重叠、越界和非法 chunk 都必须 fail-closed。

use anyhow::{bail, Context, Result};
use ash::vk;
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_head_chunk_argmax::{
        decode_batched_head_argmax, S14HeadChunkArgmaxPipeline, S14HeadChunkArgmaxRecorder,
        S14HeadChunkArgmaxShape, S14HeadChunkWorkspace, S14_HEAD_ARGMAX_WORDS,
        S14_HEAD_STATUS_NON_FINITE,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const VOCAB: u32 = 11;
const HIDDEN: u32 = 128;
const CHUNK_ROWS: u32 = 4;
const BATCH: u32 = 4;

#[derive(Debug, Clone, Copy)]
struct WorkspaceLayout {
    normalized: u64,
    logits: u64,
    accumulator: u64,
    total: u64,
}

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let alignment = u64::from(properties.limits.min_storage_buffer_offset_alignment.max(1));
    let shape = S14HeadChunkArgmaxShape::new_batched(VOCAB, HIDDEN, CHUNK_ROWS, BATCH)?;
    let head_base = alignment;
    let head_bytes = shape.head_total_bytes()?;
    let head_total = align_up(
        head_base
            .checked_add(head_bytes)
            .and_then(|value| value.checked_add(alignment))
            .context("synthetic head arena overflow")?,
        alignment,
    )?;
    let layout = workspace_layout(shape, alignment)?;

    let mut normalized = vec![0.0f32; BATCH as usize * HIDDEN as usize];
    for lane in 0..BATCH as usize {
        normalized[lane * HIDDEN as usize + lane] = 1.0;
    }
    let score_rows = [
        [-2.0f32, 1.0, 4.0, 4.0, 0.5, 3.0, -1.0, 9.0, 8.0, 2.0, 7.0],
        [0.0f32, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, -1.0],
        [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        [0.0f32, 1.0, 12.0, 3.0, 12.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ];
    let mut head = vec![0u16; VOCAB as usize * HIDDEN as usize];
    for (lane, scores) in score_rows.iter().enumerate() {
        for (token, score) in scores.iter().copied().enumerate() {
            head[token * HIDDEN as usize + lane] = to_bf16_bits(score);
        }
    }
    let cpu = (0..BATCH as usize)
        .map(|lane| {
            cpu_argmax(
                &head,
                &normalized[lane * HIDDEN as usize..(lane + 1) * HIDDEN as usize],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let expected_tokens = [7, 9, 10, 2];
    if cpu.iter().map(|value| value.0).collect::<Vec<_>>() != expected_tokens {
        bail!("synthetic CPU fixture drift");
    }

    let head_arena = host_buffer(&ctx, head_total)?;
    let workspace_buffer = host_buffer(&ctx, layout.total)?;
    unsafe {
        head_arena.write_at(head_base as usize, bytemuck::cast_slice(&head));
        workspace_buffer.write_at(
            layout.normalized as usize,
            bytemuck::cast_slice(&normalized),
        );
    }
    let workspace = S14HeadChunkWorkspace::new(
        &workspace_buffer,
        layout.normalized,
        layout.logits,
        layout.accumulator,
    );
    let pipeline = S14HeadChunkArgmaxPipeline::new(&ctx)?;
    let dispatches = (0..shape.chunk_count())
        .map(|chunk| {
            let spec = shape.chunk(chunk)?;
            let weight_offset = head_base
                .checked_add(spec.token_start as u64 * HIDDEN as u64 * 2)
                .context("synthetic head chunk offset overflow")?;
            if weight_offset == 0 || weight_offset % alignment != 0 {
                bail!("synthetic head chunk offset is zero/misaligned");
            }
            pipeline.bind_chunk(
                &ctx,
                shape,
                chunk,
                StorageBufferSlice {
                    buffer: &head_arena,
                    offset: weight_offset,
                },
                workspace,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?
    };
    let commands = unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(3),
        )?
    };

    let positive_started = Instant::now();
    let mut positive = S14HeadChunkArgmaxRecorder::new(shape)?;
    unsafe {
        begin(&ctx, commands[0])?;
        positive.cmd_reset(&ctx, commands[0], &dispatches[0])?;
        for dispatch in &dispatches {
            positive.cmd_chunk(&ctx, commands[0], &pipeline, dispatch)?;
        }
    }
    let receipt = positive.finish_recording()?;
    unsafe { ctx.device.end_command_buffer(commands[0])? };
    submit_and_wait(&ctx, commands[0])?;
    let positive_wall_ms = positive_started.elapsed().as_secs_f64() * 1000.0;
    let positive_words = mapped_words(&workspace_buffer, layout.accumulator, shape.batch);
    let gpu = decode_batched_head_argmax(&receipt, &positive_words)?;
    for lane in 0..BATCH as usize {
        if gpu[lane].token_id != cpu[lane].0 || gpu[lane].logit.to_bits() != cpu[lane].1.to_bits() {
            bail!("synthetic BF16 chunk head mismatch at batch lane {lane}");
        }
    }

    // 同一 command 内故意把 chunk1 放在 chunk0 前；必须在提交前 poison。
    let mut out_of_order = S14HeadChunkArgmaxRecorder::new(shape)?;
    unsafe {
        begin(&ctx, commands[1])?;
        out_of_order.cmd_reset(&ctx, commands[1], &dispatches[0])?;
        if out_of_order
            .cmd_chunk(&ctx, commands[1], &pipeline, &dispatches[1])
            .is_ok()
        {
            bail!("head recorder accepted out-of-order chunk");
        }
        ctx.device.end_command_buffer(commands[1])?;
    }

    // NaN 不能被普通比较吞掉；GPU sticky status 必须拒绝最终 token。
    normalized[2 * HIDDEN as usize + 2] = f32::NAN;
    unsafe {
        workspace_buffer.write_at(
            layout.normalized as usize,
            bytemuck::cast_slice(&normalized),
        );
    }
    let mut nan_recording = S14HeadChunkArgmaxRecorder::new(shape)?;
    unsafe {
        begin(&ctx, commands[2])?;
        nan_recording.cmd_reset(&ctx, commands[2], &dispatches[0])?;
        for dispatch in &dispatches {
            nan_recording.cmd_chunk(&ctx, commands[2], &pipeline, dispatch)?;
        }
    }
    let nan_receipt = nan_recording.finish_recording()?;
    unsafe { ctx.device.end_command_buffer(commands[2])? };
    submit_and_wait(&ctx, commands[2])?;
    let nan_words = mapped_words(&workspace_buffer, layout.accumulator, shape.batch);
    if nan_words[2 * S14_HEAD_ARGMAX_WORDS + 3] & S14_HEAD_STATUS_NON_FINITE == 0
        || decode_batched_head_argmax(&nan_receipt, &nan_words).is_ok()
    {
        bail!("head argmax accepted NaN path");
    }

    let overlap = S14HeadChunkWorkspace::new(
        &workspace_buffer,
        layout.normalized,
        layout.normalized,
        layout.accumulator,
    );
    if pipeline
        .bind_chunk(
            &ctx,
            shape,
            0,
            StorageBufferSlice {
                buffer: &head_arena,
                offset: head_base,
            },
            overlap,
        )
        .is_ok()
    {
        bail!("head primitive accepted overlapping workspace ranges");
    }
    if pipeline
        .bind_chunk(
            &ctx,
            shape,
            0,
            StorageBufferSlice {
                buffer: &head_arena,
                offset: head_arena.size(),
            },
            workspace,
        )
        .is_ok()
    {
        bail!("head primitive accepted out-of-bounds weight range");
    }
    if pipeline
        .bind_chunk(
            &ctx,
            shape,
            shape.chunk_count(),
            StorageBufferSlice {
                buffer: &head_arena,
                offset: head_base,
            },
            workspace,
        )
        .is_ok()
    {
        bail!("head primitive accepted invalid chunk/token range");
    }

    println!(
        "status=pass gpu={:?} batch={} vocab={} hidden={} chunk_rows={} chunks={} last_chunk_rows={} head_nonzero_offset={} workspace_nonzero_offsets=3 cpu_tokens={:?} gpu_tokens={:?} progress={:?} sticky_status={:?} out_of_order_rejected=1 overlap_rejected=1 out_of_bounds_rejected=1 invalid_range_rejected=1 nan_rejected=1 positive_wall_ms={positive_wall_ms:.4}",
        ctx.gpu_name,
        shape.batch,
        shape.vocab,
        shape.hidden,
        shape.chunk_rows,
        shape.chunk_count(),
        shape.chunk(shape.chunk_count() - 1)?.rows,
        head_base,
        cpu.iter().map(|value| value.0).collect::<Vec<_>>(),
        gpu.iter().map(|value| value.token_id).collect::<Vec<_>>(),
        positive_words
            .chunks_exact(S14_HEAD_ARGMAX_WORDS)
            .map(|row| row[2])
            .collect::<Vec<_>>(),
        positive_words
            .chunks_exact(S14_HEAD_ARGMAX_WORDS)
            .map(|row| row[3])
            .collect::<Vec<_>>(),
    );

    for dispatch in dispatches {
        dispatch.destroy(&ctx);
    }
    unsafe { ctx.device.destroy_command_pool(pool, None) };
    pipeline.destroy(&ctx);
    workspace_buffer.destroy(&ctx);
    head_arena.destroy(&ctx);
    Ok(())
}

fn workspace_layout(shape: S14HeadChunkArgmaxShape, alignment: u64) -> Result<WorkspaceLayout> {
    let mut cursor = alignment;
    let normalized = alloc(&mut cursor, shape.normalized_input_bytes()?, alignment)?;
    let logits = alloc(&mut cursor, shape.max_chunk_logits_bytes()?, alignment)?;
    let accumulator = alloc(&mut cursor, shape.argmax_bytes()?, alignment)?;
    Ok(WorkspaceLayout {
        normalized,
        logits,
        accumulator,
        total: align_up(cursor, alignment)?,
    })
}

unsafe fn begin(ctx: &VulkanContext, command: vk::CommandBuffer) -> Result<()> {
    ctx.device.begin_command_buffer(
        command,
        &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;
    Ok(())
}

fn submit_and_wait(ctx: &VulkanContext, command: vk::CommandBuffer) -> Result<()> {
    unsafe {
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        Ok(())
    }
}

fn cpu_argmax(head: &[u16], input: &[f32]) -> Result<(u32, f32)> {
    let mut best: Option<(u32, f32)> = None;
    for token in 0..VOCAB {
        let row = &head[token as usize * HIDDEN as usize..(token + 1) as usize * HIDDEN as usize];
        let mut logit = 0.0f32;
        for (&weight, &value) in row.iter().zip(input) {
            logit += f32::from_bits((weight as u32) << 16) * value;
        }
        if !logit.is_finite() {
            bail!("CPU fixture produced non-finite logit");
        }
        if best.is_none_or(|(best_token, best_value)| {
            logit > best_value || (logit == best_value && token < best_token)
        }) {
            best = Some((token, logit));
        }
    }
    best.context("empty synthetic logits")
}

fn host_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
}

fn mapped_words(buffer: &GpuBuffer, offset: u64, batch: u32) -> Vec<u32> {
    unsafe {
        let values = std::slice::from_raw_parts(
            (buffer.mapped() as *const u8).add(offset as usize) as *const u32,
            batch as usize * S14_HEAD_ARGMAX_WORDS,
        );
        values.to_vec()
    }
}

fn alloc(cursor: &mut u64, bytes: u64, alignment: u64) -> Result<u64> {
    let start = align_up(*cursor, alignment)?;
    *cursor = start
        .checked_add(bytes)
        .context("synthetic workspace size overflow")?;
    Ok(start)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid storage-buffer alignment {alignment}");
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("synthetic alignment overflow")
}

fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
