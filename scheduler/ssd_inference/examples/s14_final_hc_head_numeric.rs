//! DeepSeek-V4 final HC-head 的生产尺寸 Vulkan/CPU 数值门。

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    s14_final_hc_head::{
        validate_final_hc_head_status, S14FinalHcHeadDispatch, S14FinalHcHeadPipeline,
        S14FinalHcHeadShape, S14_FINAL_HC_HIDDEN, S14_FINAL_HC_STATUS_NON_FINITE_INPUT,
        S14_FINAL_HC_STREAMS,
    },
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

const LANES: usize = 256;
const NORM_EPS: f32 = 1.0e-6;
const HC_EPS: f32 = 1.0e-6;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipeline = S14FinalHcHeadPipeline::new(&ctx)?;
    let shape = S14FinalHcHeadShape::production();
    let hidden_len = (S14_FINAL_HC_STREAMS * S14_FINAL_HC_HIDDEN) as usize;
    let hidden: Vec<u16> = (0..hidden_len)
        .map(|index| {
            let value = ((index * 17 + 13) % 251) as f32;
            to_bf16_bits((value - 125.0) / 128.0)
        })
        .collect();
    let hc_head_fn: Vec<f32> = (0..S14_FINAL_HC_STREAMS as usize)
        .flat_map(|output| {
            (0..hidden_len).map(move |index| {
                let value = ((index * 29 + output * 43 + 7) % 257) as f32;
                (value - 128.0) / 4096.0
            })
        })
        .collect();
    let scale = [0.7f32];
    let base = [-0.3f32, 0.1, 0.4, -0.2];
    let reference = cpu_reference(&hidden, &hc_head_fn, scale[0], &base);

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
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    let dispatch = pipeline.bind(
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

    let wall_ms = dispatch_once(&ctx, &pipeline, &dispatch, command)?;
    let status = read_status(&status_buffer);
    validate_final_hc_head_status(status)?;
    let output = mapped_u16(&output_buffer, S14_FINAL_HC_HIDDEN as usize);
    let mismatches = output
        .iter()
        .zip(&reference.output_bf16)
        .filter(|(actual, expected)| actual != expected)
        .count();
    let aux = mapped_f32(&aux_buffer, 8);
    let aux_expected: Vec<f32> = reference
        .normalized_logits
        .iter()
        .chain(&reference.pre)
        .copied()
        .collect();
    let aux_max_abs = aux
        .iter()
        .zip(&aux_expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if mismatches != 0 || aux_max_abs > 2.0e-5 {
        bail!(
            "final HC-head numeric drift: bf16_mismatches={mismatches}, aux_max_abs={aux_max_abs:.9}"
        );
    }

    // 非有限输入必须拒绝整个输出，不能留下部分 BF16 或 aux。
    let mut bad_hidden = hidden.clone();
    bad_hidden[0] = 0x7f80; // +Inf BF16
    let output_sentinel = vec![0x55aau16; S14_FINAL_HC_HIDDEN as usize];
    let aux_sentinel = [1.0f32; 8];
    unsafe {
        hidden_buffer.write_at(0, bytemuck::cast_slice(&bad_hidden));
        output_buffer.write_at(0, bytemuck::cast_slice(&output_sentinel));
        aux_buffer.write_at(0, bytemuck::cast_slice(&aux_sentinel));
        status_buffer.write_at(0, bytemuck::bytes_of(&0u32));
    }
    dispatch_once(&ctx, &pipeline, &dispatch, command)?;
    let rejected_status = read_status(&status_buffer);
    if rejected_status & S14_FINAL_HC_STATUS_NON_FINITE_INPUT == 0
        || mapped_u16(&output_buffer, S14_FINAL_HC_HIDDEN as usize)
            .iter()
            .any(|&value| value != 0)
        || mapped_f32(&aux_buffer, 8)
            .iter()
            .any(|&value| value.to_bits() != 0)
    {
        bail!("final HC-head did not fail closed for non-finite input");
    }

    // status 是 candidate 级 sticky：即使恢复合法输入，也必须继续拒绝。
    unsafe {
        hidden_buffer.write_at(0, bytemuck::cast_slice(&hidden));
        output_buffer.write_at(0, bytemuck::cast_slice(&output_sentinel));
        aux_buffer.write_at(0, bytemuck::cast_slice(&aux_sentinel));
    }
    dispatch_once(&ctx, &pipeline, &dispatch, command)?;
    if read_status(&status_buffer) != rejected_status
        || mapped_u16(&output_buffer, S14_FINAL_HC_HIDDEN as usize)
            .iter()
            .any(|&value| value != 0)
        || mapped_f32(&aux_buffer, 8)
            .iter()
            .any(|&value| value.to_bits() != 0)
    {
        bail!("final HC-head sticky status did not preserve fail-closed state");
    }

    println!(
        "status=pass hidden={} streams={} bf16_mismatches=0 aux_max_abs={aux_max_abs:.9} valid_wall_ms={wall_ms:.4} sticky_status=0x{rejected_status:08x}",
        S14_FINAL_HC_HIDDEN, S14_FINAL_HC_STREAMS
    );

    unsafe { ctx.device.destroy_command_pool(pool, None) };
    dispatch.binder.destroy(&ctx);
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

struct CpuReference {
    output_bf16: Vec<u16>,
    normalized_logits: [f32; 4],
    pre: [f32; 4],
}

fn cpu_reference(hidden: &[u16], hc_head_fn: &[f32], scale: f32, base: &[f32; 4]) -> CpuReference {
    let flat = hidden.len();
    assert_eq!(flat, 4 * S14_FINAL_HC_HIDDEN as usize);
    assert_eq!(hc_head_fn.len(), 4 * flat);
    let mut norm_partial = [0.0f32; LANES];
    let mut logit_partial = [[0.0f32; LANES]; 4];
    for lane in 0..LANES {
        for index in (lane..flat).step_by(LANES) {
            let value = from_bf16(hidden[index]);
            norm_partial[lane] += value * value;
            for output in 0..4 {
                logit_partial[output][lane] += value * hc_head_fn[output * flat + index];
            }
        }
    }
    let mut stride = LANES / 2;
    while stride > 0 {
        for lane in 0..stride {
            norm_partial[lane] += norm_partial[lane + stride];
            for output in 0..4 {
                logit_partial[output][lane] += logit_partial[output][lane + stride];
            }
        }
        stride >>= 1;
    }
    let inverse_rms = (norm_partial[0] / flat as f32 + NORM_EPS).sqrt().recip();
    let mut normalized_logits = [0.0f32; 4];
    let mut pre = [0.0f32; 4];
    for output in 0..4 {
        normalized_logits[output] = logit_partial[output][0] * inverse_rms;
        pre[output] = sigmoid(normalized_logits[output] * scale + base[output]) + HC_EPS;
    }
    let mut output_bf16 = vec![0u16; S14_FINAL_HC_HIDDEN as usize];
    for dimension in 0..S14_FINAL_HC_HIDDEN as usize {
        let mut sum = 0.0f32;
        for stream in 0..4 {
            sum +=
                pre[stream] * from_bf16(hidden[stream * S14_FINAL_HC_HIDDEN as usize + dimension]);
        }
        output_bf16[dimension] = to_bf16_bits(sum);
    }
    CpuReference {
        output_bf16,
        normalized_logits,
        pre,
    }
}

fn dispatch_once(
    ctx: &VulkanContext,
    pipeline: &S14FinalHcHeadPipeline,
    dispatch: &S14FinalHcHeadDispatch,
    command: vk::CommandBuffer,
) -> Result<f64> {
    let fence = unsafe {
        ctx.device
            .create_fence(&vk::FenceCreateInfo::default(), None)?
    };
    let started = Instant::now();
    unsafe {
        ctx.device
            .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())?;
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        pipeline.cmd(ctx, command, dispatch);
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        ctx.device.destroy_fence(fence, None);
    }
    Ok(started.elapsed().as_secs_f64() * 1000.0)
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

fn read_status(buffer: &GpuBuffer) -> u32 {
    unsafe { *(buffer.mapped() as *const u32) }
}

fn mapped_u16(buffer: &GpuBuffer, len: usize) -> &[u16] {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const u16, len) }
}

fn mapped_f32(buffer: &GpuBuffer, len: usize) -> &[f32] {
    unsafe { std::slice::from_raw_parts(buffer.mapped() as *const f32, len) }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
