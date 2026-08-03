//! position4+ ratio4 compressed indexer + sparse attention Vulkan 秒级数值门。

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_f32_to_bf16::{S14F32ToBf16Pipeline, S14F32ToBf16Shape},
    s14_position1_attention::position_rope_cos_sin,
    s14_sparse_attention::{
        reference_sparse_indexer, S14Ratio4SparseAttentionBindings, S14SparseAttentionPipelines,
        S14SparseAttentionShape, S14_INDEX_HEADS, S14_INDEX_HEAD_DIM, S14_WINDOW_ROWS,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const ATTENTION_HEADS: usize = 64;
const ATTENTION_DIM: usize = 512;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipelines = S14SparseAttentionPipelines::new(&ctx)?;
    let head_weight_publisher = S14F32ToBf16Pipeline::new(&ctx)?;
    let shape = S14SparseAttentionShape::new(7, 0, 7, 2)?;
    let rope = position_rope_cos_sin(shape.position, 4)?;

    let mut raw_index_query = vec![0u16; (S14_INDEX_HEADS * S14_INDEX_HEAD_DIM) as usize];
    raw_index_query[0] = to_bf16(1.0);
    raw_index_query[S14_INDEX_HEAD_DIM as usize] = to_bf16(-1.0);
    let mut index_cache = vec![0u16; 2 * S14_INDEX_HEAD_DIM as usize];
    index_cache[..S14_INDEX_HEAD_DIM as usize].fill(to_bf16(1.0));
    index_cache[S14_INDEX_HEAD_DIM as usize..].fill(to_bf16(-1.0));
    let mut head_weights = vec![0.0f32; S14_INDEX_HEADS as usize];
    // This value distinguishes the two official BF16 publication boundaries:
    // BF16(F32 value) -> 0x3f00, then BF16(* scale) -> 0x3bb5.  Omitting the
    // first boundary and rounding only once produces 0x3bb6.
    head_weights[0] = f32::from_bits(0x3f00_5703);
    head_weights[1] = 1.0;
    let head_weights_bf16: Vec<u16> = head_weights.iter().copied().map(to_bf16).collect();
    let scale = f32::from_bits(0x3c35_04f3);
    let published = f32::from_bits(u32::from(head_weights_bf16[0]) << 16);
    let double_rounded = to_bf16(published * scale);
    let single_rounded = to_bf16(head_weights[0] * scale);
    if head_weights_bf16[0] != 0x3f00
        || double_rounded != 0x3bb5
        || single_rounded != 0x3bb6
        || double_rounded == single_rounded
    {
        bail!(
            "weights_proj BF16 boundary fixture drift: first=0x{:04x} double=0x{double_rounded:04x} single=0x{single_rounded:04x}",
            head_weights_bf16[0]
        );
    }

    let query = vec![0u16; ATTENTION_HEADS * ATTENTION_DIM];
    let mut window = vec![0u16; S14_WINDOW_ROWS as usize * ATTENTION_DIM];
    for row in 0..7usize {
        window[row * ATTENTION_DIM] = to_bf16((row + 1) as f32);
    }
    let mut current = vec![0u16; ATTENTION_DIM];
    current[0] = to_bf16(8.0);
    let mut compressed = vec![0u16; 2 * ATTENTION_DIM];
    compressed[0] = to_bf16(9.0);
    compressed[ATTENTION_DIM] = to_bf16(10.0);
    let sink = vec![0.0f32; ATTENTION_HEADS];

    let raw_index_query_buffer = host_buffer(&ctx, bytes_u16(&raw_index_query))?;
    let rope_buffer = host_buffer(&ctx, bytes_f32(&rope))?;
    let processed_index_query_buffer = host_buffer(&ctx, bytes_u16(&raw_index_query))?;
    let index_cache_buffer = host_buffer(&ctx, bytes_u16(&index_cache))?;
    let head_weights_f32_buffer = host_buffer(&ctx, bytes_f32(&head_weights))?;
    let head_weights_buffer = host_buffer(&ctx, bytes_u16(&head_weights_bf16))?;
    let index_scores_buffer = host_buffer(&ctx, 2 * 4)?;
    let compressed_indices_buffer = host_buffer(&ctx, 2 * 4)?;
    let query_buffer = host_buffer(&ctx, bytes_u16(&query))?;
    let window_buffer = host_buffer(&ctx, bytes_u16(&window))?;
    let current_buffer = host_buffer(&ctx, bytes_u16(&current))?;
    let compressed_buffer = host_buffer(&ctx, bytes_u16(&compressed))?;
    let sink_buffer = host_buffer(&ctx, bytes_f32(&sink))?;
    let output_buffer = host_buffer(&ctx, bytes_u16(&query))?;
    let status_buffer = host_buffer(&ctx, 4)?;
    unsafe {
        raw_index_query_buffer.write_at(0, bytemuck::cast_slice(&raw_index_query));
        rope_buffer.write_at(0, bytemuck::cast_slice(&rope));
        index_cache_buffer.write_at(0, bytemuck::cast_slice(&index_cache));
        head_weights_f32_buffer.write_at(0, bytemuck::cast_slice(&head_weights));
        query_buffer.write_at(0, bytemuck::cast_slice(&query));
        window_buffer.write_at(0, bytemuck::cast_slice(&window));
        current_buffer.write_at(0, bytemuck::cast_slice(&current));
        compressed_buffer.write_at(0, bytemuck::cast_slice(&compressed));
        sink_buffer.write_at(0, bytemuck::cast_slice(&sink));
        status_buffer.write_at(0, &0u32.to_le_bytes());
    }
    let head_weight_dispatch = head_weight_publisher.bind(
        &ctx,
        S14F32ToBf16Shape::new(S14_INDEX_HEADS)?,
        &head_weights_f32_buffer,
        &head_weights_buffer,
        &status_buffer,
    )?;

    let pool = unsafe {
        ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_graphics),
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
    unsafe {
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        head_weight_publisher.cmd(&ctx, command, &head_weight_dispatch);
        let publish_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        ctx.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[publish_barrier],
            &[],
            &[],
        );
    }
    let recording = unsafe {
        pipelines.record_ratio4(
            &ctx,
            command,
            S14Ratio4SparseAttentionBindings {
                raw_index_query: StorageBufferSlice::whole(&raw_index_query_buffer),
                rope: StorageBufferSlice::whole(&rope_buffer),
                processed_index_query: StorageBufferSlice::whole(&processed_index_query_buffer),
                index_cache: StorageBufferSlice::whole(&index_cache_buffer),
                head_weights: StorageBufferSlice::whole(&head_weights_buffer),
                index_scores: StorageBufferSlice::whole(&index_scores_buffer),
                compressed_indices: StorageBufferSlice::whole(&compressed_indices_buffer),
                query: StorageBufferSlice::whole(&query_buffer),
                window_kv: StorageBufferSlice::whole(&window_buffer),
                current_kv: StorageBufferSlice::whole(&current_buffer),
                compressed_kv: StorageBufferSlice::whole(&compressed_buffer),
                sink: StorageBufferSlice::whole(&sink_buffer),
                output: StorageBufferSlice::whole(&output_buffer),
                status: StorageBufferSlice::whole(&status_buffer),
            },
            shape,
        )?
    };
    unsafe {
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

    let status = unsafe { *(status_buffer.mapped() as *const u32) };
    if status != 0 {
        bail!("compressed sparse attention sticky status 非零: 0x{status:08x}");
    }
    let published_weights = mapped_u16(&head_weights_buffer, head_weights_bf16.len());
    if published_weights != head_weights_bf16 {
        bail!(
            "weights_proj first BF16 publication mismatch: actual={published_weights:?} expected={head_weights_bf16:?}"
        );
    }
    let processed_query = mapped_u16(&processed_index_query_buffer, raw_index_query.len());
    let reference = reference_sparse_indexer(
        processed_query,
        &index_cache,
        &head_weights,
        shape.compressed_count,
    )?;
    let actual_indices = mapped_u32(&compressed_indices_buffer, 2);
    let actual_scores = mapped_f32(&index_scores_buffer, 2);
    if actual_indices != reference.indices || actual_scores != reference.scores {
        bail!(
            "compressed indexer mismatch: indices={actual_indices:?}/{:?} scores={actual_scores:?}/{:?}",
            reference.indices,
            reference.scores
        );
    }
    shape.kv_sequence(actual_indices)?;

    let output = mapped_u16(&output_buffer, query.len());
    let expected = to_bf16(5.0);
    let mut signed_zero_count = 0usize;
    for head in 0..ATTENTION_HEADS {
        for dimension in 0..ATTENTION_DIM {
            let expected_bits = if dimension == 0 { expected } else { 0 };
            let actual = output[head * ATTENTION_DIM + dimension];
            let signed_zero_equivalent = (actual & 0x7fff) == 0 && (expected_bits & 0x7fff) == 0;
            if actual == 0x8000 && expected_bits == 0 {
                signed_zero_count += 1;
            }
            if actual != expected_bits && !signed_zero_equivalent {
                bail!(
                    "attention mismatch head={head} dim={dimension}: actual=0x{actual:04x} expected=0x{expected_bits:04x}"
                );
            }
        }
    }
    println!(
        "status=pass position=7 compressed_count=2 weights_proj_first_bf16=0x{:04x} scaled_double_bf16=0x{double_rounded:04x} wrong_single_bf16=0x{single_rounded:04x} index_order={actual_indices:?} index_scores_exact=true attention_elements={} attention_bf16_numeric_exact=true signed_zero_count={signed_zero_count} sticky_status=0 wall_ms={wall_ms:.4}",
        published_weights[0],
        output.len(),
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    head_weight_dispatch.binder.destroy(&ctx);
    head_weight_publisher.destroy(&ctx);
    recording.destroy(&ctx);
    status_buffer.destroy(&ctx);
    output_buffer.destroy(&ctx);
    sink_buffer.destroy(&ctx);
    compressed_buffer.destroy(&ctx);
    current_buffer.destroy(&ctx);
    window_buffer.destroy(&ctx);
    query_buffer.destroy(&ctx);
    compressed_indices_buffer.destroy(&ctx);
    index_scores_buffer.destroy(&ctx);
    head_weights_buffer.destroy(&ctx);
    head_weights_f32_buffer.destroy(&ctx);
    index_cache_buffer.destroy(&ctx);
    processed_index_query_buffer.destroy(&ctx);
    rope_buffer.destroy(&ctx);
    raw_index_query_buffer.destroy(&ctx);
    pipelines.destroy(&ctx);
    Ok(())
}

fn host_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
}

fn bytes_u16(values: &[u16]) -> u64 {
    (values.len() * 2) as u64
}

fn bytes_f32(values: &[f32]) -> u64 {
    (values.len() * 4) as u64
}

fn mapped_u16(buffer: &GpuBuffer, len: usize) -> &[u16] {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const u16, len) }
}

fn mapped_u32(buffer: &GpuBuffer, len: usize) -> &[u32] {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const u32, len) }
}

fn mapped_f32(buffer: &GpuBuffer, len: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const f32, len) }
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
