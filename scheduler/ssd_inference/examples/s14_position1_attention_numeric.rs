//! FullDepth43 position1/2 pre-compression window sparse-attention Vulkan numeric gate.

use anyhow::{bail, Result};
use ash::vk;
use sha2::{Digest, Sha256};
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_position0_attention::{S14_POSITION0_HEADS, S14_POSITION0_HEAD_DIM},
    s14_position1_attention::{position_rope_cos_sin, S14Position1AttentionPipeline},
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let pipeline = S14Position1AttentionPipeline::new(&ctx)?;
    for position in [1, 2] {
        for ratio in [0, 4] {
            run_case(&ctx, &pipeline, position, ratio)?;
        }
    }
    pipeline.destroy(&ctx);
    Ok(())
}

fn run_case(
    ctx: &VulkanContext,
    pipeline: &S14Position1AttentionPipeline,
    position: u32,
    compress_ratio: u32,
) -> Result<()> {
    let heads = S14_POSITION0_HEADS as usize;
    let dim = S14_POSITION0_HEAD_DIM as usize;
    let previous_count = position as usize;
    let query: Vec<u16> = (0..heads * dim)
        .map(|index| to_bf16_bits(((index * 19 % 101) as f32 - 50.0) / 128.0))
        .collect();
    let previous_kv: Vec<u16> = (0..previous_count * dim)
        .map(|index| {
            let row = index / dim;
            let column = index % dim;
            to_bf16_bits(
                (((column * 23 + row * 31) % 89) as f32 - 44.0 + row as f32 * 0.125) / 96.0,
            )
        })
        .collect();
    let current_kv: Vec<u16> = (0..dim)
        .map(|index| to_bf16_bits(((index * 29 % 97) as f32 - 48.0) / 112.0))
        .collect();
    let sink: Vec<f32> = (0..heads).map(|head| (head as f32 - 31.5) / 24.0).collect();
    let rope = position_rope_cos_sin(position, compress_ratio)?;
    let reference = python_semantic_reference(
        &query,
        &previous_kv,
        &current_kv,
        &sink,
        &rope,
        heads,
        dim,
        previous_count,
    );

    let query_buffer = host_buffer(ctx, (query.len() * 2) as u64)?;
    let previous_buffer = host_buffer(ctx, (previous_kv.len() * 2) as u64)?;
    let current_buffer = host_buffer(ctx, (current_kv.len() * 2) as u64)?;
    let sink_buffer = host_buffer(ctx, (sink.len() * 4) as u64)?;
    let rope_buffer = host_buffer(ctx, (rope.len() * 4) as u64)?;
    let output_buffer = host_buffer(ctx, (query.len() * 2) as u64)?;
    unsafe {
        query_buffer.write_at(0, bytemuck::cast_slice(&query));
        previous_buffer.write_at(0, bytemuck::cast_slice(&previous_kv));
        current_buffer.write_at(0, bytemuck::cast_slice(&current_kv));
        sink_buffer.write_at(0, bytemuck::cast_slice(&sink));
        rope_buffer.write_at(0, bytemuck::cast_slice(&rope));
    }
    let dispatch = pipeline.bind_slices(
        ctx,
        StorageBufferSlice::whole(&query_buffer),
        StorageBufferSlice::whole(&previous_buffer),
        StorageBufferSlice::whole(&current_buffer),
        StorageBufferSlice::whole(&sink_buffer),
        StorageBufferSlice::whole(&rope_buffer),
        StorageBufferSlice::whole(&output_buffer),
        position,
        position,
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
        pipeline.cmd(ctx, command, &dispatch);
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
    let output =
        unsafe { std::slice::from_raw_parts(output_buffer.mapped() as *const u16, query.len()) };
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_abs_index = 0usize;
    let mut max_abs_actual_bits = 0u16;
    let mut max_abs_expected_bits = 0u16;
    for (index, (&actual, &expected)) in output.iter().zip(&reference).enumerate() {
        if actual != expected {
            mismatches += 1;
        }
        let error = (from_bf16(actual) - from_bf16(expected)).abs();
        if error > max_abs {
            max_abs = error;
            max_abs_index = index;
            max_abs_actual_bits = actual;
            max_abs_expected_bits = expected;
        }
    }
    let output_sha256 = sha256_bf16(output);
    let reference_sha256 = sha256_bf16(&reference);
    let rope_sha256 = sha256_f32(&rope);
    let expected_rope_sha256 = match (position, compress_ratio) {
        (1, 0) => "c6b5e399d5475db9335f36c09008cf21f8097c249d6e008d2b58d59e7f1b9f5c",
        (1, 4) => "40df0c6f80b63fb49e04cc5d65b0b32c1b5a6e72ae842b05b5bcfe7c49e0661b",
        (2, 0) => "715c9c9ef6edfd834d0289516ca2da7bd4a105280af742d7bfe782c7d36252d7",
        (2, 4) => "119e909b6dbbecc5afde2f3ce740e1c488061fa0009055ca3f9845426294f81d",
        _ => unreachable!(),
    };
    if rope_sha256 != expected_rope_sha256 {
        bail!(
            "position{position} attention ratio{compress_ratio} official RoPE SHA mismatch: actual={rope_sha256} expected={expected_rope_sha256}"
        );
    }
    if mismatches != 0 {
        if position == 2 && compress_ratio == 4 {
            trace_reference_element(
                &query,
                &previous_kv,
                &current_kv,
                &sink,
                &rope,
                heads,
                dim,
                previous_count,
                max_abs_index,
            );
        }
        bail!(
            "position{position} attention ratio{compress_ratio} mismatch_count={mismatches} max_abs={max_abs:e} max_abs_index={max_abs_index} actual_bits=0x{max_abs_actual_bits:04x} expected_bits=0x{max_abs_expected_bits:04x} output_sha256={output_sha256} reference_sha256={reference_sha256}"
        );
    }
    println!(
        "status=pass position={position} ratio={compress_ratio} rows={} elements={} mismatch_count=0 max_abs=0 output_sha256={output_sha256} reference_sha256={reference_sha256} rope_sha256={rope_sha256} wall_ms={wall_ms:.4}",
        previous_count + 1,
        output.len()
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    dispatch.binder.destroy(ctx);
    output_buffer.destroy(ctx);
    rope_buffer.destroy(ctx);
    sink_buffer.destroy(ctx);
    current_buffer.destroy(ctx);
    previous_buffer.destroy(ctx);
    query_buffer.destroy(ctx);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn trace_reference_element(
    query: &[u16],
    previous_kv: &[u16],
    current_kv: &[u16],
    sink: &[f32],
    rope: &[f32; 64],
    heads: usize,
    dim: usize,
    previous_count: usize,
    output_index: usize,
) {
    let head = output_index / dim;
    let d = output_index % dim;
    assert!(head < heads);
    let mut rotated_query = query.to_vec();
    let mut rotated_current = current_kv.to_vec();
    for current_head in 0..heads {
        apply_rope_bf16(
            &mut rotated_query[current_head * dim + dim - 64..current_head * dim + dim],
            rope,
            false,
        );
    }
    apply_rope_bf16(&mut rotated_current[dim - 64..], rope, false);

    let token_count = previous_count + 1;
    let scale = (dim as f32).sqrt().recip();
    let mut scores = vec![0.0f32; token_count];
    for (token, score) in scores.iter_mut().enumerate() {
        let mut lanes = [0.0f32; 64];
        for lane in 0..64 {
            for column in (lane..dim).step_by(64) {
                let q = from_bf16(rotated_query[head * dim + column]);
                let kv = if token < previous_count {
                    from_bf16(previous_kv[token * dim + column])
                } else {
                    from_bf16(rotated_current[column])
                };
                lanes[lane] += q * kv;
            }
        }
        let lane_bits: Vec<String> = lanes
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect();
        let mut stride = 32;
        while stride > 0 {
            for lane in 0..stride {
                lanes[lane] += lanes[lane + stride];
            }
            stride >>= 1;
        }
        *score = lanes[0] * scale;
        eprintln!(
            "trace token={token} lane_bits={} dot={:.9e}/0x{:08x} score={:.9e}/0x{:08x}",
            lane_bits.join(","),
            lanes[0],
            lanes[0].to_bits(),
            *score,
            score.to_bits(),
        );
    }
    let mut maximum = sink[head];
    for &score in &scores {
        maximum = maximum.max(score);
    }
    let sink_exp = (sink[head] - maximum).exp();
    let mut denominator = sink_exp;
    let mut probabilities = Vec::with_capacity(token_count);
    for &score in &scores {
        let probability = (score - maximum).exp();
        probabilities.push(probability);
        denominator += probability;
    }
    let reciprocal = denominator.recip();
    let kv: Vec<f32> = (0..token_count)
        .map(|token| {
            if token < previous_count {
                from_bf16(previous_kv[token * dim + d])
            } else {
                from_bf16(rotated_current[d])
            }
        })
        .collect();
    let mut value = 0.0f32;
    for token in 0..token_count {
        probabilities[token] *= reciprocal;
        let product = probabilities[token] * kv[token];
        value += product;
        eprintln!(
            "trace token={token} exp_arg={:.9e}/0x{:08x} probability={:.9e}/0x{:08x} kv={:.9e}/0x{:08x} product={:.9e}/0x{:08x} accum={:.9e}/0x{:08x}",
            scores[token] - maximum,
            (scores[token] - maximum).to_bits(),
            probabilities[token],
            probabilities[token].to_bits(),
            kv[token],
            kv[token].to_bits(),
            product,
            product.to_bits(),
            value,
            value.to_bits(),
        );
    }
    eprintln!(
        "trace head={head} d={d} scale={:.9e}/0x{:08x} sink={:.9e}/0x{:08x} maximum={:.9e}/0x{:08x} sink_exp={:.9e}/0x{:08x} denominator={:.9e}/0x{:08x} reciprocal={:.9e}/0x{:08x} value={:.9e}/0x{:08x} bf16=0x{:04x}",
        scale,
        scale.to_bits(),
        sink[head],
        sink[head].to_bits(),
        maximum,
        maximum.to_bits(),
        sink_exp,
        sink_exp.to_bits(),
        denominator,
        denominator.to_bits(),
        reciprocal,
        reciprocal.to_bits(),
        value,
        value.to_bits(),
        to_bf16_bits(value),
    );
}

/// CPU transcription of the Python sequence:
/// forward RoPE into BF16 q/current-KV -> sparse_attention(F32 math, BF16 output)
/// -> inverse RoPE into BF16。行、lane 与 softmax 累加顺序与 production shader 固定一致。
#[allow(clippy::too_many_arguments)]
fn python_semantic_reference(
    query: &[u16],
    previous_kv: &[u16],
    current_kv: &[u16],
    sink: &[f32],
    rope: &[f32; 64],
    heads: usize,
    dim: usize,
    previous_count: usize,
) -> Vec<u16> {
    let mut rotated_query = query.to_vec();
    let mut rotated_current = current_kv.to_vec();
    for head in 0..heads {
        apply_rope_bf16(
            &mut rotated_query[head * dim + dim - 64..head * dim + dim],
            rope,
            false,
        );
    }
    apply_rope_bf16(&mut rotated_current[dim - 64..], rope, false);

    let mut output = vec![0u16; query.len()];
    let scale = (dim as f32).sqrt().recip();
    for head in 0..heads {
        let token_count = previous_count + 1;
        let mut scores = vec![0.0f32; token_count];
        for (token, score) in scores.iter_mut().enumerate() {
            let mut lanes = [0.0f32; 64];
            for lane in 0..64 {
                for d in (lane..dim).step_by(64) {
                    let q = from_bf16(rotated_query[head * dim + d]);
                    let kv = if token < previous_count {
                        from_bf16(previous_kv[token * dim + d])
                    } else {
                        from_bf16(rotated_current[d])
                    };
                    lanes[lane] += q * kv;
                }
            }
            let mut stride = 32;
            while stride > 0 {
                for lane in 0..stride {
                    lanes[lane] += lanes[lane + stride];
                }
                stride >>= 1;
            }
            *score = lanes[0] * scale;
        }
        let mut maximum = sink[head];
        for &score in &scores {
            maximum = maximum.max(score);
        }
        let mut denominator = (sink[head] - maximum).exp();
        let mut probabilities = Vec::with_capacity(token_count);
        for &score in &scores {
            let probability = (score - maximum).exp();
            probabilities.push(probability);
            denominator += probability;
        }
        let reciprocal = denominator.recip();
        for probability in &mut probabilities {
            *probability *= reciprocal;
        }
        for d in 0..dim {
            let mut value = 0.0f32;
            for (token, &probability) in probabilities.iter().enumerate() {
                let kv = if token < previous_count {
                    from_bf16(previous_kv[token * dim + d])
                } else {
                    from_bf16(rotated_current[d])
                };
                value += probability * kv;
            }
            output[head * dim + d] = to_bf16_bits(value);
        }
        apply_rope_bf16(
            &mut output[head * dim + dim - 64..head * dim + dim],
            rope,
            true,
        );
    }
    output
}

fn apply_rope_bf16(values: &mut [u16], rope: &[f32; 64], inverse: bool) {
    for (pair, values) in values.chunks_exact_mut(2).enumerate() {
        let x = from_bf16(values[0]);
        let y = from_bf16(values[1]);
        let cosine = rope[pair * 2];
        let sine = rope[pair * 2 + 1];
        let (left, right) = if inverse {
            (x * cosine + y * sine, -x * sine + y * cosine)
        } else {
            (x * cosine - y * sine, x * sine + y * cosine)
        };
        values[0] = to_bf16_bits(left);
        values[1] = to_bf16_bits(right);
    }
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

fn from_bf16(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    if exponent == 0x7f80_0000 {
        return (bits >> 16) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn sha256_bf16(values: &[u16]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}
