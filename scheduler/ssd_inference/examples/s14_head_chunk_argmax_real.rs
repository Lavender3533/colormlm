//! 真实 DeepSeek-V4 BF16 词表头的 32 段流式 GPU argmax 门。
//!
//! 只保留一个 4096 行 head bank，并在每段 GPU 完成后覆写下一段；因此不会把
//! 1.06 GiB 输出头常驻显存。输入使用 `s14_bf16_head` 已冻结的确定性向量，结果
//! 必须复现同一真实 head 的既有 CPU/GPU 基准 token 106762。

use anyhow::{bail, Context, Result};
use ash::vk;
use memmap2::Mmap;
use sha2::{Digest, Sha256};
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_head_chunk_argmax::{
        decode_batched_head_argmax, S14HeadChunkArgmaxPipeline, S14HeadChunkArgmaxRecorder,
        S14HeadChunkArgmaxShape, S14HeadChunkWorkspace, S14_HEAD_ARGMAX_WORDS,
    },
    GpuBuffer, VulkanContext,
};
use std::{fs::File, path::PathBuf, time::Instant};

const HEAD_PATH: &str =
    "D:/models/Polaris-S14/range_cache/ac7f39b6146436528a1c856bec3e95865f29bca6f4c0d6861fdbe6085192e494.bin";
const HEAD_SHA256: &str = "029e3c5293b29cc426e21d87795e15efa4d363f27b2bc4a9e3aef7d79f047919";
const EXPECTED_TOKEN: u32 = 106_762;
const EXPECTED_LOGIT: f32 = 4.39032;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(HEAD_PATH));
    let batch = std::env::var("POLARIS_HEAD_BATCH")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()
        .context("parse POLARIS_HEAD_BATCH")?
        .unwrap_or(1);
    let shape = S14HeadChunkArgmaxShape::production_batched(batch)?;
    if shape.chunk_count() != 32 || shape.chunk(31)?.rows != 2_304 {
        bail!("production head chunk contract drift");
    }
    let file = File::open(&path).with_context(|| format!("open head {}", path.display()))?;
    if file.metadata()?.len() != shape.head_total_bytes()? {
        bail!("real head byte count drift");
    }
    let head = unsafe { Mmap::map(&file)? };
    let observed_sha = sha256_bytes(&head);
    if observed_sha != HEAD_SHA256 {
        bail!("real head SHA-256 drift");
    }

    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let alignment = u64::from(properties.limits.min_storage_buffer_offset_alignment.max(1));
    let head_offset = alignment;
    let head_bank_bytes = align_up(
        head_offset
            .checked_add(shape.max_chunk_weight_bytes()?)
            .context("head bank byte overflow")?,
        alignment,
    )?;
    let normalized_offset = alignment;
    let logits_offset = align_up(
        normalized_offset + shape.normalized_input_bytes()?,
        alignment,
    )?;
    let accumulator_offset = align_up(logits_offset + shape.max_chunk_logits_bytes()?, alignment)?;
    let workspace_bytes = align_up(accumulator_offset + shape.argmax_bytes()?, alignment)?;

    let head_bank = host_buffer(&ctx, head_bank_bytes, false)?;
    let workspace_buffer = host_buffer(&ctx, workspace_bytes, true)?;
    let normalized_row = deterministic_hidden();
    let normalized = (0..shape.batch)
        .flat_map(|_| normalized_row.iter().copied())
        .collect::<Vec<_>>();
    unsafe {
        workspace_buffer.write_at(
            normalized_offset as usize,
            bytemuck::cast_slice(&normalized),
        );
    }
    let workspace = S14HeadChunkWorkspace::new(
        &workspace_buffer,
        normalized_offset,
        logits_offset,
        accumulator_offset,
    );
    let pipeline = S14HeadChunkArgmaxPipeline::new(&ctx)?;
    let mut recorder = S14HeadChunkArgmaxRecorder::new(shape)?;

    let pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?
    };
    let command = unsafe {
        ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0]
    };
    let fence = unsafe {
        ctx.device
            .create_fence(&vk::FenceCreateInfo::default(), None)?
    };
    let started = Instant::now();
    let mut upload_ms = 0.0f64;
    let mut compute_ms = 0.0f64;
    for chunk_index in 0..shape.chunk_count() {
        let spec = shape.chunk(chunk_index)?;
        let source_offset = spec.token_start as usize * shape.hidden as usize * 2;
        let source_bytes = spec.weight_bytes(shape)? as usize;
        let source = head
            .get(source_offset..source_offset + source_bytes)
            .context("real head chunk source range overflow")?;
        let upload_started = Instant::now();
        unsafe { head_bank.write_at(head_offset as usize, source) };
        upload_ms += upload_started.elapsed().as_secs_f64() * 1000.0;
        let dispatch = pipeline.bind_chunk(
            &ctx,
            shape,
            chunk_index,
            StorageBufferSlice {
                buffer: &head_bank,
                offset: head_offset,
            },
            workspace,
        )?;
        unsafe {
            ctx.device
                .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())?;
            ctx.device.reset_fences(&[fence])?;
            ctx.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if chunk_index == 0 {
                recorder.cmd_reset(&ctx, command, &dispatch)?;
            }
            recorder.cmd_chunk(&ctx, command, &pipeline, &dispatch)?;
            ctx.device.end_command_buffer(command)?;
            let commands = [command];
            let compute_started = Instant::now();
            ctx.device.queue_submit(
                ctx.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                fence,
            )?;
            ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
            compute_ms += compute_started.elapsed().as_secs_f64() * 1000.0;
        }
        dispatch.destroy(&ctx);
    }
    let receipt = recorder.finish_recording()?;
    let words = mapped_words(&workspace_buffer, accumulator_offset, shape.batch);
    let results = decode_batched_head_argmax(&receipt, &words)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    if results.iter().any(|result| {
        result.token_id != EXPECTED_TOKEN || (result.logit - EXPECTED_LOGIT).abs() > 1.0e-5
    }) {
        bail!("real streamed batched head mismatch");
    }
    println!(
        "status=pass gpu={:?} batch={} real_head_sha256={} vocab={} hidden={} chunks={} last_chunk_rows={} resident_head_bytes={} full_head_bytes={} tokens={:?} logits={:?} progress={:?} sticky_status={:?} upload_memcpy_ms={upload_ms:.4} compute_wait_ms={compute_ms:.4} wall_ms={wall_ms:.4} effective_rows_per_second={:.4} full_head_resident=false",
        ctx.gpu_name,
        shape.batch,
        observed_sha,
        shape.vocab,
        shape.hidden,
        shape.chunk_count(),
        shape.chunk(shape.chunk_count() - 1)?.rows,
        shape.max_chunk_weight_bytes()?,
        shape.head_total_bytes()?,
        results.iter().map(|result| result.token_id).collect::<Vec<_>>(),
        results.iter().map(|result| result.logit).collect::<Vec<_>>(),
        words.chunks_exact(S14_HEAD_ARGMAX_WORDS).map(|row| row[2]).collect::<Vec<_>>(),
        words.chunks_exact(S14_HEAD_ARGMAX_WORDS).map(|row| row[3]).collect::<Vec<_>>(),
        shape.batch as f64 * 1000.0 / wall_ms,
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    pipeline.destroy(&ctx);
    workspace_buffer.destroy(&ctx);
    head_bank.destroy(&ctx);
    Ok(())
}

fn deterministic_hidden() -> Vec<f32> {
    let mut state = 0x7a17_5eed_d00d_beefu64;
    (0..4_096)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as u32) as f32 / ((1u32 << 24) - 1) as f32;
            (unit - 0.5) * 0.25
        })
        .collect()
}

fn host_buffer(ctx: &VulkanContext, bytes: u64, transfer_dst: bool) -> Result<GpuBuffer> {
    let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER;
    if transfer_dst {
        usage |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    GpuBuffer::new(
        ctx,
        bytes,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
}

fn mapped_words(buffer: &GpuBuffer, offset: u64, batch: u32) -> Vec<u32> {
    let values = unsafe {
        std::slice::from_raw_parts(
            (buffer.mapped() as *const u8).add(offset as usize) as *const u32,
            batch as usize * S14_HEAD_ARGMAX_WORDS,
        )
    };
    values.to_vec()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .context("alignment overflow")
}
