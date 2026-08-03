//! position2051+ ratio4 global top-512 -> actual pages -> contiguous main-KV GPU gate.

use anyhow::{bail, Context, Result};
use ash::vk;
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_ratio4_main_page_gather::{
        build_ratio4_main_page_table, validate_ratio4_main_gather_status, S14Ratio4MainGatherShape,
        S14Ratio4MainPageGatherPipeline, S14Ratio4MaterializedMainPage,
        S14_RATIO4_GATHER_ROW_WORDS, S14_RATIO4_GATHER_STATUS_BF16_NAN,
        S14_RATIO4_GATHER_STATUS_DUPLICATE_INDEX, S14_RATIO4_GATHER_STATUS_INDEX_OUT_OF_RANGE,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const LOGICAL_COUNT: u32 = 513;
const SELECTED_COUNT: u32 = 512;
const PAGE0_ROWS: u32 = 512;
const PAGE1_ROWS: u32 = 1;
const ROW_WORDS: u32 = S14_RATIO4_GATHER_ROW_WORDS;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let alignment = u64::from(properties.limits.min_storage_buffer_offset_alignment.max(4));
    let source_words = (PAGE0_ROWS + PAGE1_ROWS) * ROW_WORDS;
    let shape = S14Ratio4MainGatherShape::new(LOGICAL_COUNT, SELECTED_COUNT, source_words)?;
    let page_table = build_ratio4_main_page_table(
        shape,
        &[
            S14Ratio4MaterializedMainPage {
                page_index: 0,
                source_word_offset: 0,
                row_count: PAGE0_ROWS,
            },
            S14Ratio4MaterializedMainPage {
                page_index: 1,
                source_word_offset: PAGE0_ROWS * ROW_WORDS,
                row_count: PAGE1_ROWS,
            },
        ],
    )?;

    let mut cursor = alignment;
    let indices_offset = alloc(&mut cursor, u64::from(SELECTED_COUNT) * 4, alignment)?;
    let page_table_offset = alloc(&mut cursor, page_table.len() as u64 * 4, alignment)?;
    let source_offset = alloc(&mut cursor, u64::from(source_words) * 4, alignment)?;
    let output_offset = alloc(&mut cursor, shape.packed_main_bytes(), alignment)?;
    let status_offset = alloc(&mut cursor, 4, alignment)?;
    let workspace = GpuBuffer::new(
        &ctx,
        align_up(cursor, alignment)?,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;

    let valid_indices = std::iter::once(512u32).chain(0..511u32).collect::<Vec<_>>();
    let source = build_source_words();
    unsafe {
        workspace.write_at(
            page_table_offset as usize,
            bytemuck::cast_slice(&page_table),
        );
        workspace.write_at(source_offset as usize, bytemuck::cast_slice(&source));
        workspace.write_at(
            indices_offset as usize,
            bytemuck::cast_slice(&valid_indices),
        );
    }

    let pipeline = S14Ratio4MainPageGatherPipeline::new(&ctx)?;
    let dispatch = pipeline.bind_slices(
        &ctx,
        slice(&workspace, indices_offset),
        slice(&workspace, page_table_offset),
        slice(&workspace, source_offset),
        slice(&workspace, output_offset),
        slice(&workspace, status_offset),
        shape,
    )?;
    let command_pool = unsafe {
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
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0]
    };
    let fence = unsafe {
        ctx.device
            .create_fence(&vk::FenceCreateInfo::default(), None)?
    };

    let started = Instant::now();
    submit(
        &ctx,
        &pipeline,
        &dispatch,
        command,
        fence,
        &workspace,
        output_offset,
        shape.packed_main_bytes(),
        status_offset,
    )?;
    let valid_ms = started.elapsed().as_secs_f64() * 1000.0;
    let status = read_u32(&workspace, status_offset, 1)[0];
    validate_ratio4_main_gather_status(status)?;
    let output = read_u32(
        &workspace,
        output_offset,
        (SELECTED_COUNT * ROW_WORDS) as usize,
    );
    verify_output(&output, &valid_indices, &source)?;

    let mut duplicate = valid_indices.clone();
    duplicate[1] = duplicate[0];
    write_indices(&workspace, indices_offset, &duplicate);
    submit(
        &ctx,
        &pipeline,
        &dispatch,
        command,
        fence,
        &workspace,
        output_offset,
        shape.packed_main_bytes(),
        status_offset,
    )?;
    expect_status(
        &workspace,
        status_offset,
        S14_RATIO4_GATHER_STATUS_DUPLICATE_INDEX,
        "duplicate",
    )?;

    let mut out_of_range = valid_indices.clone();
    out_of_range[0] = LOGICAL_COUNT;
    write_indices(&workspace, indices_offset, &out_of_range);
    submit(
        &ctx,
        &pipeline,
        &dispatch,
        command,
        fence,
        &workspace,
        output_offset,
        shape.packed_main_bytes(),
        status_offset,
    )?;
    expect_status(
        &workspace,
        status_offset,
        S14_RATIO4_GATHER_STATUS_INDEX_OUT_OF_RANGE,
        "out_of_range",
    )?;

    write_indices(&workspace, indices_offset, &valid_indices);
    let page1_first_word = source_offset + u64::from(PAGE0_ROWS * ROW_WORDS) * 4;
    unsafe {
        workspace.write_at(page1_first_word as usize, &0x4000_7fc1u32.to_le_bytes());
    }
    submit(
        &ctx,
        &pipeline,
        &dispatch,
        command,
        fence,
        &workspace,
        output_offset,
        shape.packed_main_bytes(),
        status_offset,
    )?;
    expect_status(
        &workspace,
        status_offset,
        S14_RATIO4_GATHER_STATUS_BF16_NAN,
        "bf16_nan",
    )?;
    if read_u32(&workspace, output_offset, 1)[0] != 0 {
        bail!("ratio4 NaN row was not zeroed before rejection");
    }

    println!(
        "status=pass gpu={:?} logical_count={} selected_count={} pages=2 mapped_first=(page1,row0) valid_wall_ms={valid_ms:.4} duplicate=sticky out_of_range=sticky bf16_nan=sticky",
        ctx.gpu_name, LOGICAL_COUNT, SELECTED_COUNT
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(command_pool, None);
    }
    dispatch.destroy(&ctx);
    pipeline.destroy(&ctx);
    workspace.destroy(&ctx);
    Ok(())
}

fn build_source_words() -> Vec<u32> {
    let mut words = Vec::with_capacity(((PAGE0_ROWS + PAGE1_ROWS) * ROW_WORDS) as usize);
    for logical_row in 0..LOGICAL_COUNT {
        for word in 0..ROW_WORDS {
            let column = word * 2;
            let low = bf16_fixture(logical_row, column);
            let high = bf16_fixture(logical_row, column + 1);
            words.push(u32::from(low) | (u32::from(high) << 16));
        }
    }
    words
}

fn bf16_fixture(row: u32, column: u32) -> u16 {
    0x4000 | (((row * 131 + column) & 0x007f) as u16)
}

fn verify_output(output: &[u32], indices: &[u32], source: &[u32]) -> Result<()> {
    for (slot, &logical_row) in indices.iter().enumerate() {
        let source_start = logical_row as usize * ROW_WORDS as usize;
        let output_start = slot * ROW_WORDS as usize;
        if output[output_start..output_start + ROW_WORDS as usize]
            != source[source_start..source_start + ROW_WORDS as usize]
        {
            bail!("ratio4 gathered row mismatch at packed slot {slot}");
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn submit(
    ctx: &VulkanContext,
    pipeline: &S14Ratio4MainPageGatherPipeline,
    dispatch: &ssd_inference::s14_ratio4_main_page_gather::S14Ratio4MainPageGatherDispatch,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    workspace: &GpuBuffer,
    output_offset: u64,
    output_bytes: u64,
    status_offset: u64,
) -> Result<()> {
    unsafe {
        ctx.device
            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())?;
        ctx.device.reset_fences(&[fence])?;
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        ctx.device
            .cmd_fill_buffer(command, workspace.handle(), status_offset, 4, 0);
        ctx.device
            .cmd_fill_buffer(command, workspace.handle(), output_offset, output_bytes, 0);
        barrier(
            ctx,
            command,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        pipeline.cmd(ctx, command, dispatch);
        barrier(
            ctx,
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::HOST_READ,
        );
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
    }
    Ok(())
}

fn write_indices(buffer: &GpuBuffer, offset: u64, indices: &[u32]) {
    unsafe {
        buffer.write_at(offset as usize, bytemuck::cast_slice(indices));
    }
}

fn expect_status(buffer: &GpuBuffer, offset: u64, expected: u32, label: &str) -> Result<()> {
    let actual = read_u32(buffer, offset, 1)[0];
    if actual != expected || validate_ratio4_main_gather_status(actual).is_ok() {
        bail!("ratio4 {label} status mismatch: expected=0x{expected:08x} actual=0x{actual:08x}");
    }
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
        .context("ratio4 main gather workspace overflow")?;
    Ok(start)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .context("ratio4 main gather alignment overflow")
}
