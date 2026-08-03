//! S14 三条 arena offset API 的真实 Vulkan A/B 数值门。
//!
//! 每个 kernel 使用同一输入分别执行旧的独立-buffer API 与新的同-arena API，
//! 要求 F32 输出逐位一致，并验证重叠、越界与未对齐 offset 在提交前失败。

use anyhow::{bail, Context, Result};
use ash::vk;
use ssd_inference::{
    s14_vulkan::{
        s14_batched_official_prepare_buffer_bytes, s14_exact_order_block_reduce_buffer_bytes,
        S14NumericPipelines, S14RaggedBranchOffsets, S14RaggedMatvecShape, S14RaggedProjection,
    },
    GpuBuffer, VulkanContext,
};

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipelines = S14NumericPipelines::new(&ctx)?;
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let alignment = u64::from(properties.limits.min_storage_buffer_offset_alignment.max(1));

    let ragged = verify_ragged_mxfp4(&ctx, &pipelines, alignment)?;
    let batched = verify_batched_prepare(&ctx, &pipelines, alignment)?;
    let exact = verify_exact_reduce(&ctx, &pipelines, alignment)?;

    println!(
        "status=pass gpu=\"{}\" descriptor_alignment={} ragged_mxfp4_bit_mismatches={} batched_prepare_bit_mismatches={} exact_reduce_bit_mismatches={} bit_exact_ab=pass nonzero_offset_bindings=11 overlap_rejected=3 out_of_bounds_rejected=3 misaligned_rejected={} capture_inputs=false",
        ctx.gpu_name,
        alignment,
        ragged,
        batched,
        exact,
        if alignment > 1 { 3 } else { 0 }
    );
    pipelines.destroy(&ctx);
    Ok(())
}

fn verify_ragged_mxfp4(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    alignment: u64,
) -> Result<usize> {
    let shape = S14RaggedMatvecShape::new(1, 1, 1, 128, S14RaggedProjection::W1)?;
    let metadata = [S14RaggedBranchOffsets {
        w1: 0,
        s1: 64,
        w3: 0,
        s3: 0,
        w2: 0,
        s2: 0,
    }];
    let metadata_words = metadata[0].words();
    let mut weight_bytes = vec![0x22u8; 64]; // 两个 E2M1 +1.0
    weight_bytes.extend_from_slice(&[127u8; 4]); // UE8M0 scale = 1.0
    let x: Vec<f32> = (0..128)
        .map(|index| ((index * 13 % 31) as f32 - 15.0) / 16.0)
        .collect();

    let x_buffer = host_buffer(ctx, 512)?;
    let weight_buffer = host_buffer(ctx, weight_bytes.len() as u64)?;
    let metadata_buffer = host_buffer(ctx, 24)?;
    let y_buffer = host_buffer(ctx, 4)?;
    unsafe {
        x_buffer.write_at(0, bytemuck::cast_slice(&x));
        weight_buffer.write_at(0, &weight_bytes);
        metadata_buffer.write_at(0, bytemuck::cast_slice(&metadata_words));
        y_buffer.write_at(0, bytemuck::bytes_of(&f32::NAN));
    }
    let independent = pipelines.bind_ragged_mxfp4_weight_arena(
        ctx,
        shape,
        &x_buffer,
        &weight_buffer,
        weight_bytes.len() as u64,
        &metadata_buffer,
        &metadata,
        &y_buffer,
    )?;
    submit(ctx, |command| unsafe {
        pipelines.cmd_ragged_mxfp4_matvec(ctx, command, &independent)
    })?;
    let expected = mapped_f32(&y_buffer, 0, 1);

    let metadata_offset = alignment;
    let metadata_arena = host_buffer(ctx, metadata_offset + 24)?;
    let (workspace_offsets, workspace_bytes) = packed_offsets(&[512, 4], alignment)?;
    let workspace = host_buffer(ctx, workspace_bytes)?;
    unsafe {
        metadata_arena.write_at(
            metadata_offset as usize,
            bytemuck::cast_slice(&metadata_words),
        );
        workspace.write_at(workspace_offsets[0] as usize, bytemuck::cast_slice(&x));
        workspace.write_at(workspace_offsets[1] as usize, bytemuck::bytes_of(&f32::NAN));
    }
    let arena_dispatch = pipelines.bind_ragged_mxfp4_arenas(
        ctx,
        shape,
        &weight_buffer,
        weight_bytes.len() as u64,
        &metadata_arena,
        metadata_arena.size(),
        metadata_offset,
        &metadata,
        &workspace,
        workspace.size(),
        workspace_offsets[0],
        workspace_offsets[1],
    )?;
    submit(ctx, |command| unsafe {
        pipelines.cmd_ragged_mxfp4_matvec(ctx, command, &arena_dispatch)
    })?;
    let actual = mapped_f32(&workspace, workspace_offsets[1], 1);
    let mismatches = bit_mismatches(&expected, &actual);
    if mismatches != 0 {
        bail!("ragged MXFP4 independent/arena mismatch={mismatches}");
    }

    if pipelines
        .bind_ragged_mxfp4_arenas(
            ctx,
            shape,
            &weight_buffer,
            weight_bytes.len() as u64,
            &metadata_arena,
            metadata_arena.size(),
            metadata_offset,
            &metadata,
            &workspace,
            workspace.size(),
            workspace_offsets[0],
            workspace_offsets[0],
        )
        .is_ok()
    {
        bail!("ragged MXFP4 arena accepted overlapping activation input/output");
    }
    if pipelines
        .bind_ragged_mxfp4_arenas(
            ctx,
            shape,
            &weight_buffer,
            weight_bytes.len() as u64,
            &metadata_arena,
            metadata_arena.size(),
            metadata_offset,
            &metadata,
            &workspace,
            workspace.size(),
            workspace_offsets[0],
            workspace.size(),
        )
        .is_ok()
    {
        bail!("ragged MXFP4 arena accepted out-of-bounds output");
    }
    if alignment > 1
        && pipelines
            .bind_ragged_mxfp4_arenas(
                ctx,
                shape,
                &weight_buffer,
                weight_bytes.len() as u64,
                &metadata_arena,
                metadata_arena.size(),
                metadata_offset + 1,
                &metadata,
                &workspace,
                workspace.size(),
                workspace_offsets[0],
                workspace_offsets[1],
            )
            .is_ok()
    {
        bail!("ragged MXFP4 arena accepted misaligned metadata offset");
    }

    arena_dispatch.binder.destroy(ctx);
    independent.binder.destroy(ctx);
    workspace.destroy(ctx);
    metadata_arena.destroy(ctx);
    y_buffer.destroy(ctx);
    metadata_buffer.destroy(ctx);
    weight_buffer.destroy(ctx);
    x_buffer.destroy(ctx);
    Ok(mismatches)
}

fn verify_batched_prepare(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    alignment: u64,
) -> Result<usize> {
    let branches = 2;
    let n = 128;
    let (matrix_bytes, route_bytes) = s14_batched_official_prepare_buffer_bytes(branches, n)?;
    let elements = (branches * n) as usize;
    let gate: Vec<f32> = (0..elements)
        .map(|index| ((index * 7 % 41) as f32 - 20.0) / 8.0)
        .collect();
    let up: Vec<f32> = (0..elements)
        .map(|index| ((index * 11 % 43) as f32 - 21.0) / 7.0)
        .collect();
    let routes = [0.25f32, 0.375];

    let gate_buffer = host_buffer(ctx, matrix_bytes)?;
    let up_buffer = host_buffer(ctx, matrix_bytes)?;
    let route_buffer = host_buffer(ctx, route_bytes)?;
    let hidden_buffer = host_buffer(ctx, matrix_bytes)?;
    unsafe {
        gate_buffer.write_at(0, bytemuck::cast_slice(&gate));
        up_buffer.write_at(0, bytemuck::cast_slice(&up));
        route_buffer.write_at(0, bytemuck::cast_slice(&routes));
        hidden_buffer.write_at(0, &vec![0xff; matrix_bytes as usize]);
    }
    let independent = pipelines.bind_batched_official_expert_prepare(
        ctx,
        branches,
        n,
        &gate_buffer,
        &up_buffer,
        &route_buffer,
        &hidden_buffer,
    )?;
    submit(ctx, |command| unsafe {
        pipelines.cmd_batched_official_expert_prepare(ctx, command, &independent)
    })?;
    let expected = mapped_f32(&hidden_buffer, 0, elements);

    let (offsets, workspace_bytes) = packed_offsets(
        &[matrix_bytes, matrix_bytes, route_bytes, matrix_bytes],
        alignment,
    )?;
    let workspace = host_buffer(ctx, workspace_bytes)?;
    unsafe {
        workspace.write_at(offsets[0] as usize, bytemuck::cast_slice(&gate));
        workspace.write_at(offsets[1] as usize, bytemuck::cast_slice(&up));
        workspace.write_at(offsets[2] as usize, bytemuck::cast_slice(&routes));
        workspace.write_at(offsets[3] as usize, &vec![0x7f; matrix_bytes as usize]);
    }
    let arena_dispatch = pipelines.bind_batched_official_expert_prepare_arena(
        ctx,
        branches,
        n,
        &workspace,
        workspace.size(),
        offsets[0],
        offsets[1],
        offsets[2],
        offsets[3],
    )?;
    submit(ctx, |command| unsafe {
        pipelines.cmd_batched_official_expert_prepare(ctx, command, &arena_dispatch)
    })?;
    let actual = mapped_f32(&workspace, offsets[3], elements);
    let mismatches = bit_mismatches(&expected, &actual);
    if mismatches != 0 {
        bail!("batched prepare independent/arena mismatch={mismatches}");
    }
    if pipelines
        .bind_batched_official_expert_prepare_arena(
            ctx,
            branches,
            n,
            &workspace,
            workspace.size(),
            offsets[0],
            offsets[1],
            offsets[2],
            offsets[0],
        )
        .is_ok()
    {
        bail!("batched prepare arena accepted overlapping gate/hidden");
    }
    if pipelines
        .bind_batched_official_expert_prepare_arena(
            ctx,
            branches,
            n,
            &workspace,
            workspace.size(),
            offsets[0],
            offsets[1],
            offsets[2],
            workspace.size(),
        )
        .is_ok()
    {
        bail!("batched prepare arena accepted out-of-bounds hidden");
    }
    if alignment > 1
        && pipelines
            .bind_batched_official_expert_prepare_arena(
                ctx,
                branches,
                n,
                &workspace,
                workspace.size(),
                offsets[0] + 1,
                offsets[1],
                offsets[2],
                offsets[3],
            )
            .is_ok()
    {
        bail!("batched prepare arena accepted misaligned gate");
    }

    arena_dispatch.binder.destroy(ctx);
    independent.binder.destroy(ctx);
    workspace.destroy(ctx);
    hidden_buffer.destroy(ctx);
    route_buffer.destroy(ctx);
    up_buffer.destroy(ctx);
    gate_buffer.destroy(ctx);
    Ok(mismatches)
}

fn verify_exact_reduce(
    ctx: &VulkanContext,
    pipelines: &S14NumericPipelines,
    alignment: u64,
) -> Result<usize> {
    let positions = 1;
    let (routed_bytes, output_bytes) = s14_exact_order_block_reduce_buffer_bytes(positions)?;
    let routed_elements = (routed_bytes / 4) as usize;
    let output_elements = (output_bytes / 4) as usize;
    let routed: Vec<f32> = (0..routed_elements)
        .map(|index| ((index * 19 % 37) as f32 - 18.0) / 32.0)
        .collect();
    let shared: Vec<f32> = (0..output_elements)
        .map(|index| ((index * 23 % 41) as f32 - 20.0) / 24.0)
        .collect();

    let routed_buffer = host_buffer(ctx, routed_bytes)?;
    let shared_buffer = host_buffer(ctx, output_bytes)?;
    let output_buffer = host_buffer(ctx, output_bytes)?;
    unsafe {
        routed_buffer.write_at(0, bytemuck::cast_slice(&routed));
        shared_buffer.write_at(0, bytemuck::cast_slice(&shared));
        output_buffer.write_at(0, &vec![0xff; output_bytes as usize]);
    }
    let independent = pipelines.bind_exact_order_block_reduce(
        ctx,
        positions,
        &routed_buffer,
        &shared_buffer,
        &output_buffer,
    )?;
    submit(ctx, |command| unsafe {
        pipelines.cmd_exact_order_block_reduce(ctx, command, &independent)
    })?;
    let expected = mapped_f32(&output_buffer, 0, output_elements);

    let (offsets, workspace_bytes) =
        packed_offsets(&[routed_bytes, output_bytes, output_bytes], alignment)?;
    let workspace = host_buffer(ctx, workspace_bytes)?;
    unsafe {
        workspace.write_at(offsets[0] as usize, bytemuck::cast_slice(&routed));
        workspace.write_at(offsets[1] as usize, bytemuck::cast_slice(&shared));
        workspace.write_at(offsets[2] as usize, &vec![0x7f; output_bytes as usize]);
    }
    let arena_dispatch = pipelines.bind_exact_order_block_reduce_arena(
        ctx,
        positions,
        &workspace,
        workspace.size(),
        offsets[0],
        offsets[1],
        offsets[2],
    )?;
    submit(ctx, |command| unsafe {
        pipelines.cmd_exact_order_block_reduce(ctx, command, &arena_dispatch)
    })?;
    let actual = mapped_f32(&workspace, offsets[2], output_elements);
    let mismatches = bit_mismatches(&expected, &actual);
    if mismatches != 0 {
        bail!("exact reduce independent/arena mismatch={mismatches}");
    }
    if pipelines
        .bind_exact_order_block_reduce_arena(
            ctx,
            positions,
            &workspace,
            workspace.size(),
            offsets[0],
            offsets[1],
            offsets[1],
        )
        .is_ok()
    {
        bail!("exact reduce arena accepted overlapping shared/output");
    }
    if pipelines
        .bind_exact_order_block_reduce_arena(
            ctx,
            positions,
            &workspace,
            workspace.size(),
            offsets[0],
            offsets[1],
            workspace.size(),
        )
        .is_ok()
    {
        bail!("exact reduce arena accepted out-of-bounds output");
    }
    if alignment > 1
        && pipelines
            .bind_exact_order_block_reduce_arena(
                ctx,
                positions,
                &workspace,
                workspace.size(),
                offsets[0] + 1,
                offsets[1],
                offsets[2],
            )
            .is_ok()
    {
        bail!("exact reduce arena accepted misaligned routed-down");
    }

    arena_dispatch.binder.destroy(ctx);
    independent.binder.destroy(ctx);
    workspace.destroy(ctx);
    output_buffer.destroy(ctx);
    shared_buffer.destroy(ctx);
    routed_buffer.destroy(ctx);
    Ok(mismatches)
}

fn submit(ctx: &VulkanContext, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
    unsafe {
        let pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_graphics),
            None,
        )?;
        let command = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        record(command);
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    Ok(())
}

fn packed_offsets(sizes: &[u64], alignment: u64) -> Result<(Vec<u64>, u64)> {
    let mut cursor = alignment;
    let mut offsets = Vec::with_capacity(sizes.len());
    for size in sizes {
        cursor = align_up(cursor, alignment)?;
        offsets.push(cursor);
        cursor = cursor
            .checked_add(*size)
            .context("offset numeric workspace overflow")?;
    }
    Ok((offsets, cursor))
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid storage-buffer alignment {alignment}");
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("offset numeric alignment overflow")
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

fn mapped_f32(buffer: &GpuBuffer, offset: u64, len: usize) -> Vec<f32> {
    unsafe {
        std::slice::from_raw_parts(
            (buffer.mapped() as *const u8).add(offset as usize) as *const f32,
            len,
        )
        .to_vec()
    }
}

fn bit_mismatches(left: &[f32], right: &[f32]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count()
}
