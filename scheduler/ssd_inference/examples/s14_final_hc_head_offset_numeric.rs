//! final HC-head 的同 arena offset-aware Vulkan 数值门。
//!
//! 同一组生产尺寸输入分别走独立 buffer 与单一 `GpuBuffer` 非重叠子范围，
//! 要求 status、BF16 输出和 F32 aux 逐位一致；重叠与越界必须在提交前拒绝。

use anyhow::{bail, Context, Result};
use ash::vk;
use ssd_inference::{
    s14_final_hc_head::{
        validate_final_hc_head_status, S14FinalHcHeadBindings, S14FinalHcHeadBufferSlice,
        S14FinalHcHeadDispatch, S14FinalHcHeadPipeline, S14FinalHcHeadShape,
        S14_FINAL_HC_AUX_VALUES, S14_FINAL_HC_FLAT, S14_FINAL_HC_HIDDEN, S14_FINAL_HC_STREAMS,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
struct ArenaLayout {
    hidden: u64,
    hc_head_fn: u64,
    hc_head_scale: u64,
    hc_head_base: u64,
    output: u64,
    aux: u64,
    status: u64,
    total: u64,
}

impl ArenaLayout {
    fn new(shape: S14FinalHcHeadShape, alignment: u64) -> Result<Self> {
        let mut cursor = alignment;
        let hidden = alloc(&mut cursor, shape.hidden_bf16_bytes(), alignment)?;
        let hc_head_fn = alloc(&mut cursor, shape.hc_head_fn_f32_bytes(), alignment)?;
        let hc_head_scale = alloc(&mut cursor, shape.hc_head_scale_f32_bytes(), alignment)?;
        let hc_head_base = alloc(&mut cursor, shape.hc_head_base_f32_bytes(), alignment)?;
        let output = alloc(&mut cursor, shape.output_bf16_bytes(), alignment)?;
        let aux = alloc(&mut cursor, shape.aux_f32_bytes(), alignment)?;
        let status = alloc(&mut cursor, shape.status_bytes(), alignment)?;
        Ok(Self {
            hidden,
            hc_head_fn,
            hc_head_scale,
            hc_head_base,
            output,
            aux,
            status,
            total: cursor,
        })
    }

    fn bindings<'a>(&self, arena: &'a GpuBuffer) -> S14FinalHcHeadBindings<'a> {
        S14FinalHcHeadBindings {
            hidden: S14FinalHcHeadBufferSlice::new(arena, self.hidden),
            hc_head_fn: S14FinalHcHeadBufferSlice::new(arena, self.hc_head_fn),
            hc_head_scale: S14FinalHcHeadBufferSlice::new(arena, self.hc_head_scale),
            hc_head_base: S14FinalHcHeadBufferSlice::new(arena, self.hc_head_base),
            output: S14FinalHcHeadBufferSlice::new(arena, self.output),
            aux: S14FinalHcHeadBufferSlice::new(arena, self.aux),
            status: S14FinalHcHeadBufferSlice::new(arena, self.status),
        }
    }
}

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipeline = S14FinalHcHeadPipeline::new(&ctx)?;
    let shape = S14FinalHcHeadShape::production();
    let properties = unsafe { ctx.instance.get_physical_device_properties(ctx.physical) };
    let alignment = u64::from(properties.limits.min_storage_buffer_offset_alignment.max(1));
    let layout = ArenaLayout::new(shape, alignment)?;

    let hidden: Vec<u16> = (0..S14_FINAL_HC_FLAT as usize)
        .map(|index| {
            let centered = ((index * 17 + 13) % 251) as f32 - 125.0;
            to_bf16_bits(centered / 128.0)
        })
        .collect();
    let hc_head_fn: Vec<f32> = (0..S14_FINAL_HC_STREAMS as usize)
        .flat_map(|channel| {
            (0..S14_FINAL_HC_FLAT as usize).map(move |index| {
                let centered = ((index * 29 + channel * 43 + 7) % 257) as f32 - 128.0;
                centered / 4096.0
            })
        })
        .collect();
    let scale = [0.7f32];
    let base = [-0.3f32, 0.1, 0.4, -0.2];

    let hidden_buffer = host_buffer(&ctx, shape.hidden_bf16_bytes())?;
    let fn_buffer = host_buffer(&ctx, shape.hc_head_fn_f32_bytes())?;
    let scale_buffer = host_buffer(&ctx, shape.hc_head_scale_f32_bytes())?;
    let base_buffer = host_buffer(&ctx, shape.hc_head_base_f32_bytes())?;
    let output_buffer = host_buffer(&ctx, shape.output_bf16_bytes())?;
    let aux_buffer = host_buffer(&ctx, shape.aux_f32_bytes())?;
    let status_buffer = host_buffer(&ctx, shape.status_bytes())?;
    unsafe {
        hidden_buffer.write_at(0, bytemuck::cast_slice(&hidden));
        fn_buffer.write_at(0, bytemuck::cast_slice(&hc_head_fn));
        scale_buffer.write_at(0, bytemuck::cast_slice(&scale));
        base_buffer.write_at(0, bytemuck::cast_slice(&base));
        output_buffer.write_at(
            0,
            bytemuck::cast_slice(&vec![0x55aau16; S14_FINAL_HC_HIDDEN as usize]),
        );
        aux_buffer.write_at(
            0,
            bytemuck::cast_slice(&[12345.0f32; S14_FINAL_HC_AUX_VALUES as usize]),
        );
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let independent_dispatch = pipeline.bind(
        &ctx,
        shape,
        &hidden_buffer,
        &fn_buffer,
        &scale_buffer,
        &base_buffer,
        &output_buffer,
        &aux_buffer,
        &status_buffer,
    )?;
    let independent_wall_ms = dispatch_once(&ctx, &pipeline, &independent_dispatch)?;
    let independent_status = mapped_one::<u32>(&status_buffer, 0);
    validate_final_hc_head_status(independent_status)?;
    let independent_output = mapped_vec::<u16>(&output_buffer, 0, S14_FINAL_HC_HIDDEN as usize);
    let independent_aux = mapped_vec::<f32>(&aux_buffer, 0, S14_FINAL_HC_AUX_VALUES as usize);

    let arena = host_buffer(&ctx, layout.total)?;
    unsafe {
        arena.write_at(layout.hidden as usize, bytemuck::cast_slice(&hidden));
        arena.write_at(
            layout.hc_head_fn as usize,
            bytemuck::cast_slice(&hc_head_fn),
        );
        arena.write_at(layout.hc_head_scale as usize, bytemuck::cast_slice(&scale));
        arena.write_at(layout.hc_head_base as usize, bytemuck::cast_slice(&base));
        arena.write_at(
            layout.output as usize,
            bytemuck::cast_slice(&vec![0xa55au16; S14_FINAL_HC_HIDDEN as usize]),
        );
        arena.write_at(
            layout.aux as usize,
            bytemuck::cast_slice(&[-12345.0f32; S14_FINAL_HC_AUX_VALUES as usize]),
        );
        arena.write_at(layout.status as usize, bytemuck::bytes_of(&0u32));
    }
    let offsets = [
        layout.hidden,
        layout.hc_head_fn,
        layout.hc_head_scale,
        layout.hc_head_base,
        layout.output,
        layout.aux,
        layout.status,
    ];
    if offsets
        .iter()
        .any(|offset| *offset == 0 || *offset % alignment != 0)
    {
        bail!("final HC arena produced zero/misaligned descriptor offset");
    }

    let arena_dispatch = pipeline.bind_with_offsets(&ctx, shape, layout.bindings(&arena))?;
    let arena_wall_ms = dispatch_once(&ctx, &pipeline, &arena_dispatch)?;
    let arena_status = mapped_one::<u32>(&arena, layout.status);
    validate_final_hc_head_status(arena_status)?;
    let arena_output = mapped_vec::<u16>(&arena, layout.output, S14_FINAL_HC_HIDDEN as usize);
    let arena_aux = mapped_vec::<f32>(&arena, layout.aux, S14_FINAL_HC_AUX_VALUES as usize);
    let output_mismatches = independent_output
        .iter()
        .zip(&arena_output)
        .filter(|(left, right)| left != right)
        .count();
    let aux_bit_mismatches = independent_aux
        .iter()
        .zip(&arena_aux)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count();
    if independent_status != arena_status || output_mismatches != 0 || aux_bit_mismatches != 0 {
        bail!(
            "final HC offset parity drift: independent_status=0x{independent_status:08x} arena_status=0x{arena_status:08x} output_mismatches={output_mismatches} aux_bit_mismatches={aux_bit_mismatches}"
        );
    }

    let mut overlap = layout.bindings(&arena);
    overlap.output = overlap.hidden;
    if pipeline.bind_with_offsets(&ctx, shape, overlap).is_ok() {
        bail!("final HC offset API accepted overlapping hidden/output ranges");
    }
    let mut out_of_bounds = layout.bindings(&arena);
    out_of_bounds.status.offset = arena.size();
    if pipeline
        .bind_with_offsets(&ctx, shape, out_of_bounds)
        .is_ok()
    {
        bail!("final HC offset API accepted out-of-bounds status range");
    }

    println!(
        "status=pass gpu=\"{}\" descriptor_alignment={} nonzero_offsets=7 same_arena_parity=bit_exact output_bf16_mismatches=0 aux_f32_bit_mismatches=0 overlap_rejected=1 out_of_bounds_rejected=1 independent_wall_ms={independent_wall_ms:.4} arena_wall_ms={arena_wall_ms:.4}",
        ctx.gpu_name, alignment
    );

    arena_dispatch.binder.destroy(&ctx);
    independent_dispatch.binder.destroy(&ctx);
    arena.destroy(&ctx);
    status_buffer.destroy(&ctx);
    aux_buffer.destroy(&ctx);
    output_buffer.destroy(&ctx);
    base_buffer.destroy(&ctx);
    scale_buffer.destroy(&ctx);
    fn_buffer.destroy(&ctx);
    hidden_buffer.destroy(&ctx);
    pipeline.destroy(&ctx);
    Ok(())
}

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14FinalHcHeadPipeline,
    dispatch: &S14FinalHcHeadDispatch,
) -> Result<f64> {
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
        pipeline.cmd(ctx, command, dispatch);
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        let started = Instant::now();
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
        Ok(wall_ms)
    }
}

fn alloc(cursor: &mut u64, bytes: u64, alignment: u64) -> Result<u64> {
    let start = align_up(*cursor, alignment)?;
    *cursor = start
        .checked_add(bytes)
        .context("final HC arena size overflow")?;
    Ok(start)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("invalid storage-buffer alignment {alignment}");
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("final HC alignment overflow")
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

fn mapped_one<T: Copy>(buffer: &GpuBuffer, offset: u64) -> T {
    unsafe { *((buffer.mapped() as *const u8).add(offset as usize) as *const T) }
}

fn mapped_vec<T: Copy>(buffer: &GpuBuffer, offset: u64, len: usize) -> Vec<T> {
    unsafe {
        std::slice::from_raw_parts(
            (buffer.mapped() as *const u8).add(offset as usize) as *const T,
            len,
        )
        .to_vec()
    }
}

fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
