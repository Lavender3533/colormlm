//! DeepSeek-V4 四流 mHC-pre + RMSNorm 的秒级 Vulkan 数值门。
//!
//! 覆盖真实 dtype 边界：BF16 hidden、F32 hc_fn/scale/base、BF16 norm weight，
//! 并把 `expand → F32 matvec → split/Sinkhorn/reduce → BF16 → RMSNorm`
//! 放进一次 command buffer。合成数据只证明算子语义，不代表模型质量。

use anyhow::{bail, Result};
use ash::vk;
use polaris_s14_runner::{bf16_round_trip, hc_split_sinkhorn};
use ssd_inference::{
    s14_vulkan::{S14F32MatvecShape, S14HcPreShape, S14NumericPipelines},
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const HIDDEN: usize = 4096;
const STREAMS: usize = 4;
const MIXES: usize = 24;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipelines = S14NumericPipelines::new(&ctx)?;
    let shape = S14HcPreShape::new(HIDDEN as u32)?;
    let matvec_shape = S14F32MatvecShape::new(MIXES as u32, (STREAMS * HIDDEN) as u32, 1)?;

    let hidden_f32: Vec<f32> = (0..STREAMS * HIDDEN)
        .map(|i| ((i * 17 % 61) as f32 - 30.0) / 64.0)
        .collect();
    let hidden_bf16: Vec<u16> = hidden_f32
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect();
    let hidden_exact: Vec<f32> = hidden_bf16
        .iter()
        .map(|v| f32::from_bits((*v as u32) << 16))
        .collect();
    let hc_fn: Vec<f32> = (0..MIXES * STREAMS * HIDDEN)
        .map(|i| ((i * 29 % 47) as f32 - 23.0) / 2048.0)
        .collect();
    let hc_scale = [0.75f32, -0.5, 0.625];
    let hc_base: Vec<f32> = (0..MIXES).map(|i| (i as f32 - 11.5) / 128.0).collect();
    let norm_weight_f32: Vec<f32> = (0..HIDDEN).map(|i| 0.875 + (i % 9) as f32 / 64.0).collect();
    let norm_weight_bf16: Vec<u16> = norm_weight_f32
        .iter()
        .map(|v| (bf16_round_trip(*v).unwrap().to_bits() >> 16) as u16)
        .collect();
    let norm_weight_exact: Vec<f32> = norm_weight_bf16
        .iter()
        .map(|v| f32::from_bits((*v as u32) << 16))
        .collect();

    let hidden_buffer = host_buffer(&ctx, shape.hidden_bf16_bytes()?)?;
    let expanded_buffer = host_buffer(&ctx, shape.normalized_input_bytes()?)?;
    let inverse_rms_buffer = host_buffer(&ctx, 4)?;
    let weight_buffer = host_buffer(&ctx, matvec_shape.weight_bytes()?)?;
    let mixes_buffer = host_buffer(&ctx, matvec_shape.output_bytes()?)?;
    let scale_buffer = host_buffer(&ctx, 12)?;
    let base_buffer = host_buffer(&ctx, 96)?;
    let norm_weight_buffer = host_buffer(&ctx, shape.norm_weight_bytes()?)?;
    let output_bf16_buffer = host_buffer(&ctx, shape.output_bf16_bytes()?)?;
    let output_f32_buffer = host_buffer(&ctx, shape.output_f32_bytes()?)?;
    let aux_buffer = host_buffer(&ctx, 80)?;
    unsafe {
        hidden_buffer.write_at(0, bytemuck::cast_slice(&hidden_bf16));
        weight_buffer.write_at(0, bytemuck::cast_slice(&hc_fn));
        scale_buffer.write_at(0, bytemuck::cast_slice(&hc_scale));
        base_buffer.write_at(0, bytemuck::cast_slice(&hc_base));
        norm_weight_buffer.write_at(0, bytemuck::cast_slice(&norm_weight_bf16));
    }

    let prepare = pipelines.bind_hc_normalize_input(
        &ctx,
        shape,
        1.0e-6,
        &hidden_buffer,
        &expanded_buffer,
        &inverse_rms_buffer,
    )?;
    let projection = pipelines.bind_f32_matvec(
        &ctx,
        matvec_shape,
        &weight_buffer,
        &expanded_buffer,
        &mixes_buffer,
    )?;
    let reduce = pipelines.bind_hc_split_reduce_norm(
        &ctx,
        shape,
        1.0e-6,
        &hidden_buffer,
        &mixes_buffer,
        &scale_buffer,
        &base_buffer,
        &norm_weight_buffer,
        &output_bf16_buffer,
        &output_f32_buffer,
        &aux_buffer,
        &inverse_rms_buffer,
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
        pipelines.cmd_hc_normalize_input(&ctx, command, &prepare);
        shader_barrier(&ctx, command);
        pipelines.cmd_f32_matvec(&ctx, command, &projection);
        shader_barrier(&ctx, command);
        pipelines.cmd_hc_split_reduce_norm(&ctx, command, &reduce);
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

    let gpu_inverse_rms = unsafe { *(inverse_rms_buffer.mapped() as *const f32) };
    let gpu_mixes = mapped_f32(&mixes_buffer, MIXES);
    let gpu_output = mapped_f32(&output_f32_buffer, HIDDEN);
    let gpu_aux = mapped_f32(&aux_buffer, 20);

    let reference_mixes = matvec_reference(&hc_fn, &hidden_exact, MIXES, STREAMS * HIDDEN);
    let reference_inverse_rms = inverse_rms_reference(&hidden_exact);
    let scaled_mixes: Vec<f32> = reference_mixes
        .iter()
        .map(|value| *value * reference_inverse_rms)
        .collect();
    let split = hc_split_sinkhorn(&scaled_mixes, 1, &hc_scale, &hc_base)?;
    let mut reduced = vec![0.0f32; HIDDEN];
    for d in 0..HIDDEN {
        for stream in 0..STREAMS {
            reduced[d] += split.pre[stream] * hidden_exact[stream * HIDDEN + d];
        }
        reduced[d] = bf16_round_trip(reduced[d])?;
    }
    let norm_inverse = inverse_rms_reference(&reduced);
    let reference_output: Vec<f32> = reduced
        .iter()
        .zip(&norm_weight_exact)
        .map(|(value, weight)| bf16_round_trip(*value * norm_inverse * *weight).unwrap())
        .collect();
    let mut reference_aux = split.post;
    reference_aux.extend(split.comb);

    let mix_error = max_abs(&gpu_mixes, &reference_mixes);
    let inverse_error = (gpu_inverse_rms - reference_inverse_rms).abs();
    let aux_error = max_abs(&gpu_aux, &reference_aux);
    let output_error = max_abs(&gpu_output, &reference_output);
    if mix_error > 2.0e-5 || inverse_error > 2.0e-6 || aux_error > 2.0e-5 || output_error != 0.0 {
        bail!(
            "S14 HC mismatch: mix={mix_error} inv={inverse_error} aux={aux_error} output={output_error}"
        );
    }
    println!(
        "status=pass hidden={HIDDEN} wall_ms={wall_ms:.4} mix_max_abs={mix_error:.8} inverse_max_abs={inverse_error:.8} aux_max_abs={aux_error:.8} output_max_abs={output_error:.8}"
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    reduce.binder.destroy(&ctx);
    projection.binder.destroy(&ctx);
    prepare.binder.destroy(&ctx);
    for buffer in [
        aux_buffer,
        output_f32_buffer,
        output_bf16_buffer,
        norm_weight_buffer,
        base_buffer,
        scale_buffer,
        mixes_buffer,
        weight_buffer,
        inverse_rms_buffer,
        expanded_buffer,
        hidden_buffer,
    ] {
        buffer.destroy(&ctx);
    }
    pipelines.destroy(&ctx);
    Ok(())
}

unsafe fn shader_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
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

fn mapped_f32(buffer: &GpuBuffer, len: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const f32, len).to_vec() }
}

fn inverse_rms_reference(values: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 256];
    for lane in 0..256 {
        for index in (lane..values.len()).step_by(256) {
            lanes[lane] += values[index] * values[index];
        }
    }
    let mut stride = 128;
    while stride > 0 {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
        stride >>= 1;
    }
    (lanes[0] / values.len() as f32 + 1.0e-6).sqrt().recip()
}

fn matvec_reference(weight: &[f32], input: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    for row in 0..n {
        let mut lanes = [0.0f32; 64];
        for lane in 0..64 {
            for dim in (lane..k).step_by(64) {
                lanes[lane] += weight[row * k + dim] * input[dim];
            }
        }
        let mut stride = 32;
        while stride > 0 {
            for lane in 0..stride {
                lanes[lane] += lanes[lane + stride];
            }
            stride >>= 1;
        }
        output[row] = lanes[0];
    }
    output
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0, f32::max)
}
