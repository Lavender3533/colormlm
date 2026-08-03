//! position127/128/255 ratio128 deterministic sparse-attention Vulkan 秒级数值门。

use anyhow::{anyhow, bail, Result};
use ash::vk;
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_position1_attention::position_rope_cos_sin,
    s14_sparse_attention::{S14SparseAttentionPipeline, S14SparseAttentionShape, S14_WINDOW_ROWS},
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const HEADS: usize = 64;
const HEAD_DIM: usize = 512;

fn main() -> Result<()> {
    let position = std::env::args()
        .nth(1)
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|error| anyhow!("position参数解析失败: {error}"))
        })
        .transpose()?
        .unwrap_or(127);
    if !matches!(position, 127 | 128 | 255) {
        bail!("本数值门只允许position127、128或255");
    }
    let window_start = u32::from(position == 128);
    let compressed_count = if position == 255 { 2 } else { 1 };
    let ctx = VulkanContext::init()?;
    let pipeline = S14SparseAttentionPipeline::new(&ctx)?;
    let shape =
        S14SparseAttentionShape::new_ratio128(position, window_start, 127, compressed_count)?;
    let sequence = shape.kv_sequence(&[])?;
    let expected_sequence_len = 128 + compressed_count as usize;
    if sequence.len() != expected_sequence_len {
        bail!("ratio128 KV sequence length drift: {}", sequence.len());
    }

    let query = vec![0u16; HEADS * HEAD_DIM];
    let mut window = vec![0u16; S14_WINDOW_ROWS as usize * HEAD_DIM];
    if position != 128 {
        for row in 0..127usize {
            window[row * HEAD_DIM] = to_bf16((row + 1) as f32);
        }
        // position127和255的当前token都占物理row127；attention不得把旧row127
        // 当作previous window读取。
        window[127 * HEAD_DIM] = to_bf16(999.0);
    } else {
        // row0即将被当前token覆盖，attention必须从物理row1开始；sentinel可抓出
        // 任何把环形window误当连续row0..126的实现。
        window[0] = to_bf16(999.0);
        for row in 1..128usize {
            window[row * HEAD_DIM] = to_bf16(row as f32);
        }
    }
    let mut current = vec![0u16; HEAD_DIM];
    current[0] = to_bf16(128.0);
    let mut compressed = vec![0u16; compressed_count as usize * HEAD_DIM];
    compressed[0] = to_bf16(129.0);
    if compressed_count == 2 {
        compressed[HEAD_DIM] = to_bf16(130.0);
    }
    let sink = vec![0.0f32; HEADS];
    let rope = position_rope_cos_sin(position, 128)?;
    let unused_indices = vec![u32::MAX; compressed_count as usize];

    let query_buffer = host_buffer(&ctx, bytes_u16(&query))?;
    let window_buffer = host_buffer(&ctx, bytes_u16(&window))?;
    let current_buffer = host_buffer(&ctx, bytes_u16(&current))?;
    let compressed_buffer = host_buffer(&ctx, bytes_u16(&compressed))?;
    // ratio128 必须忽略该 descriptor；故意写入越界 sentinel，误读会触发 sticky status。
    let unused_indices_buffer = host_buffer(&ctx, bytes_u32(&unused_indices))?;
    let sink_buffer = host_buffer(&ctx, bytes_f32(&sink))?;
    let rope_buffer = host_buffer(&ctx, bytes_f32(&rope))?;
    let output_buffer = host_buffer(&ctx, bytes_u16(&query))?;
    let status_buffer = host_buffer(&ctx, 4)?;
    unsafe {
        query_buffer.write_at(0, bytemuck::cast_slice(&query));
        window_buffer.write_at(0, bytemuck::cast_slice(&window));
        current_buffer.write_at(0, bytemuck::cast_slice(&current));
        compressed_buffer.write_at(0, bytemuck::cast_slice(&compressed));
        unused_indices_buffer.write_at(0, bytemuck::cast_slice(&unused_indices));
        sink_buffer.write_at(0, bytemuck::cast_slice(&sink));
        rope_buffer.write_at(0, bytemuck::cast_slice(&rope));
        status_buffer.write_at(0, &0u32.to_le_bytes());
    }

    let dispatch = pipeline.bind_slices(
        &ctx,
        StorageBufferSlice::whole(&query_buffer),
        StorageBufferSlice::whole(&window_buffer),
        StorageBufferSlice::whole(&current_buffer),
        StorageBufferSlice::whole(&compressed_buffer),
        StorageBufferSlice::whole(&unused_indices_buffer),
        StorageBufferSlice::whole(&sink_buffer),
        StorageBufferSlice::whole(&rope_buffer),
        StorageBufferSlice::whole(&output_buffer),
        StorageBufferSlice::whole(&status_buffer),
        shape,
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
        pipeline.cmd(&ctx, command, &dispatch);
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
        bail!("ratio128 sparse attention sticky status 非零: 0x{status:08x}");
    }
    // query/sink 全零，KV行等概率且sink只进入分母：
    // position127/128: sum(1..=129) / (129 + 1 sink) = 64.5；
    // position255: sum(1..=130) / (130 + 1 sink) = 65.0。
    let output = mapped_u16(&output_buffer, query.len());
    let expected = to_bf16(if position == 255 { 65.0 } else { 64.5 });
    for head in 0..HEADS {
        for dimension in 0..HEAD_DIM {
            let expected_bits = if dimension == 0 { expected } else { 0 };
            let actual = output[head * HEAD_DIM + dimension];
            if actual != expected_bits && !((actual | expected_bits) & 0x7fff == 0) {
                bail!(
                    "ratio128 attention mismatch head={head} dim={dimension}: actual=0x{actual:04x} expected=0x{expected_bits:04x}"
                );
            }
        }
    }
    println!(
        "status=pass position={position} window_start={window_start} previous_window_rows=127 current_rows=1 ratio128_blocks={compressed_count} implicit_compressed_order=true index_sentinel_ignored=true ring_row0_sentinel_ignored={} ring_row127_sentinel_ignored={} attention_elements={} expected_first_bf16=0x{expected:04x} sticky_status=0 wall_ms={wall_ms:.4}",
        position == 128,
        position == 255,
        output.len(),
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    dispatch.binder.destroy(&ctx);
    pipeline.destroy(&ctx);
    status_buffer.destroy(&ctx);
    output_buffer.destroy(&ctx);
    rope_buffer.destroy(&ctx);
    sink_buffer.destroy(&ctx);
    unused_indices_buffer.destroy(&ctx);
    compressed_buffer.destroy(&ctx);
    current_buffer.destroy(&ctx);
    window_buffer.destroy(&ctx);
    query_buffer.destroy(&ctx);
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

fn bytes_u32(values: &[u32]) -> u64 {
    (values.len() * 4) as u64
}

fn mapped_u16(buffer: &GpuBuffer, len: usize) -> &[u16] {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const u16, len) }
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
