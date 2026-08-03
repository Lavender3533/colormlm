//! RX 5700 XT：同一 command 内逐页 ratio4 indexer + global top-512 数值门。

use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{GraphProfile, NativeState};
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_ratio4_global_topk::{
        validate_global_topk_status, S14Ratio4GlobalTopKPipeline, S14Ratio4PagedGlobalTopKBindings,
    },
    s14_ratio4_history_paging::S14Ratio4HistoryLayout,
    s14_sparse_attention::{
        reference_sparse_indexer, S14SparseIndexerPipeline, S14_INDEX_HEADS, S14_INDEX_HEAD_DIM,
        S14_INDEX_TOP_K,
    },
    GpuBuffer, VulkanContext,
};

const LOGICAL_COUNT: u32 = 513;
const TOP_K: usize = S14_INDEX_TOP_K as usize;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let alignment = u64::from(properties.limits.min_storage_buffer_offset_alignment.max(4));

    let mut state = NativeState::decode_layout_for(GraphProfile::FullDepth43NativeTop6, 4096)?;
    state.position = 2051;
    let history = S14Ratio4HistoryLayout::build(&state, 2, LOGICAL_COUNT)?;
    if history.pages.len() != 2
        || history.pages[0].logical_rows != (0..512)
        || history.pages[1].logical_rows != (512..513)
    {
        bail!("position2051 ratio4 history page fixture drift");
    }

    let query_len = (S14_INDEX_HEADS * S14_INDEX_HEAD_DIM) as usize;
    let mut query = vec![0u16; query_len];
    query[0] = to_bf16(1.0);
    let mut index_history = vec![0u16; LOGICAL_COUNT as usize * S14_INDEX_HEAD_DIM as usize];
    for logical in 0..512usize {
        index_history[logical * S14_INDEX_HEAD_DIM as usize] = to_bf16((512 - logical) as f32);
    }
    index_history[512 * S14_INDEX_HEAD_DIM as usize] = to_bf16(1024.0);
    let mut head_weights_f32 = vec![0.0f32; S14_INDEX_HEADS as usize];
    head_weights_f32[0] = 1.0;
    let head_weights_bf16 = head_weights_f32
        .iter()
        .copied()
        .map(to_bf16)
        .collect::<Vec<_>>();

    let mut cursor = alignment;
    let query_offset = alloc(&mut cursor, bytes_u16(&query), alignment)?;
    let history_offset = alloc(&mut cursor, bytes_u16(&index_history), alignment)?;
    let head_weights_offset = alloc(&mut cursor, bytes_u16(&head_weights_bf16), alignment)?;
    let page_scores = alloc(&mut cursor, (TOP_K * 4) as u64, alignment)?;
    let page_indices = alloc(&mut cursor, (TOP_K * 4) as u64, alignment)?;
    let bank0_scores = alloc(&mut cursor, (TOP_K * 4) as u64, alignment)?;
    let bank0_indices = alloc(&mut cursor, (TOP_K * 4) as u64, alignment)?;
    let bank1_scores = alloc(&mut cursor, (TOP_K * 4) as u64, alignment)?;
    let bank1_indices = alloc(&mut cursor, (TOP_K * 4) as u64, alignment)?;
    let status_offset = alloc(&mut cursor, 4, alignment)?;
    let workspace = GpuBuffer::new(
        &ctx,
        align_up(cursor, alignment)?,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;
    unsafe {
        workspace.write_at(query_offset as usize, bytemuck::cast_slice(&query));
        workspace.write_at(
            history_offset as usize,
            bytemuck::cast_slice(&index_history),
        );
        workspace.write_at(
            head_weights_offset as usize,
            bytemuck::cast_slice(&head_weights_bf16),
        );
    }

    let indexer = S14SparseIndexerPipeline::new(&ctx)?;
    let global_topk = S14Ratio4GlobalTopKPipeline::new(&ctx)?;
    let bindings = S14Ratio4PagedGlobalTopKBindings {
        processed_index_query: slice(&workspace, query_offset),
        indexer_history: slice(&workspace, history_offset),
        head_weights: slice(&workspace, head_weights_offset),
        page_scores: slice(&workspace, page_scores),
        page_indices: slice(&workspace, page_indices),
        global_score_banks: [
            slice(&workspace, bank0_scores),
            slice(&workspace, bank1_scores),
        ],
        global_index_banks: [
            slice(&workspace, bank0_indices),
            slice(&workspace, bank1_indices),
        ],
        status: slice(&workspace, status_offset),
    };

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
    unsafe {
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device
            .cmd_fill_buffer(command, workspace.handle(), status_offset, 4, 0);
        transfer_to_compute(&ctx, command);
    }
    let recording = unsafe {
        global_topk.record_paged_indexer_global_topk(&indexer, &ctx, command, &history, bindings)?
    };
    unsafe {
        compute_to_host(&ctx, command);
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    }

    let receipt = recording.receipt();
    if receipt.logical_count != LOGICAL_COUNT
        || receipt.selected_count != S14_INDEX_TOP_K
        || receipt.scanned_pages != 2
        || receipt.final_bank != 1
    {
        bail!("ratio4 paged recorder receipt drift: {receipt:?}");
    }
    let status = read_u32(&workspace, status_offset, 1)[0];
    validate_global_topk_status(status)?;
    let final_scores = recording.final_scores(&bindings);
    let final_indices = recording.final_indices(&bindings);
    let actual_scores = read_f32(&workspace, final_scores.offset, TOP_K);
    let actual_indices = read_u32(&workspace, final_indices.offset, TOP_K);

    let first = reference_sparse_indexer(
        &query,
        &index_history[..512 * S14_INDEX_HEAD_DIM as usize],
        &head_weights_f32,
        512,
    )?;
    let second = reference_sparse_indexer(
        &query,
        &index_history[512 * S14_INDEX_HEAD_DIM as usize..],
        &head_weights_f32,
        1,
    )?;
    let mut expected = first
        .scores
        .into_iter()
        .zip(first.indices)
        .chain(
            second
                .scores
                .into_iter()
                .zip(second.indices.into_iter().map(|index| 512 + index)),
        )
        .collect::<Vec<_>>();
    expected.sort_by(|(left_score, left_index), (right_score, right_index)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_index.cmp(right_index))
    });
    expected.truncate(TOP_K);
    let expected_scores = expected.iter().map(|entry| entry.0).collect::<Vec<_>>();
    let expected_indices = expected.iter().map(|entry| entry.1).collect::<Vec<_>>();
    if actual_indices != expected_indices
        || actual_scores
            .iter()
            .map(|value| value.to_bits())
            .ne(expected_scores.iter().map(|value| value.to_bits()))
    {
        bail!("ratio4 paged indexer/global top-k numeric mismatch");
    }
    println!(
        "status=pass gpu={:?} same_command=true logical_count={} selected_count={} pages={} final_bank={} first_indices={:?} last_index={} sticky_status={status}",
        ctx.gpu_name,
        receipt.logical_count,
        receipt.selected_count,
        receipt.scanned_pages,
        receipt.final_bank,
        &actual_indices[..6],
        actual_indices[TOP_K - 1],
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    recording.destroy(&ctx);
    global_topk.destroy(&ctx);
    indexer.destroy(&ctx);
    workspace.destroy(&ctx);
    Ok(())
}

fn slice(buffer: &GpuBuffer, offset: u64) -> StorageBufferSlice<'_> {
    StorageBufferSlice { buffer, offset }
}

fn read_u32(buffer: &GpuBuffer, offset: u64, count: usize) -> Vec<u32> {
    unsafe {
        std::slice::from_raw_parts(
            (buffer.mapped() as *const u8)
                .add(offset as usize)
                .cast::<u32>(),
            count,
        )
        .to_vec()
    }
}

fn read_f32(buffer: &GpuBuffer, offset: u64, count: usize) -> Vec<f32> {
    unsafe {
        std::slice::from_raw_parts(
            (buffer.mapped() as *const u8)
                .add(offset as usize)
                .cast::<f32>(),
            count,
        )
        .to_vec()
    }
}

unsafe fn transfer_to_compute(ctx: &VulkanContext, command: vk::CommandBuffer) {
    barrier(
        ctx,
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
    );
}

unsafe fn compute_to_host(ctx: &VulkanContext, command: vk::CommandBuffer) {
    barrier(
        ctx,
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::HOST,
        vk::AccessFlags::SHADER_WRITE,
        vk::AccessFlags::HOST_READ,
    );
}

unsafe fn barrier(
    ctx: &VulkanContext,
    command: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let memory = vk::MemoryBarrier::default()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    ctx.device.cmd_pipeline_barrier(
        command,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[memory],
        &[],
        &[],
    );
}

fn alloc(cursor: &mut u64, bytes: u64, alignment: u64) -> Result<u64> {
    let start = align_up(*cursor, alignment)?;
    *cursor = start
        .checked_add(bytes)
        .context("ratio4 global top-k workspace overflow")?;
    Ok(start)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .context("ratio4 global top-k alignment overflow")
}

fn bytes_u16(values: &[u16]) -> u64 {
    (values.len() * std::mem::size_of::<u16>()) as u64
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
