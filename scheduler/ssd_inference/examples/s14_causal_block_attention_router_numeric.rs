//! K=4 causal-block attention/router 的 RX 5700 XT 秒级结构/数值门。
//!
//! 零 query/KV/router weight 使期望值可直接判定：attention 与 logits 全零，
//! BiasTop6 稳定选择 expert 0..5，归一化权重各0.25。重点验证一次 recorder
//! 真实写出4行，而不是循环调用4次 whole-token。

use anyhow::{bail, Result};
use ash::vk;
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_causal_block_attention_router::{
        validate_causal_block_attention_router_status, S14CausalBlockAttentionRouterBindings,
        S14CausalBlockAttentionRouterRecorder, S14CausalBlockAttentionRouterShape,
    },
    s14_position1_attention::position_rope_cos_sin,
    s14_route_postprocess_gpu::S14RoutePostprocessGpuMode,
    GpuBuffer, VulkanContext,
};
use std::time::Instant;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let shape = S14CausalBlockAttentionRouterShape::new(4, 1, 1)?;

    let query = host_buffer(&ctx, shape.query_bf16_bytes()?)?;
    let committed = host_buffer(&ctx, shape.committed_window_bf16_bytes()?)?;
    let current = host_buffer(&ctx, shape.current_block_kv_bf16_bytes()?)?;
    let sink = host_buffer(&ctx, shape.sink_f32_bytes())?;
    let rope = host_buffer(&ctx, shape.rope_f32_bytes()?)?;
    let attention_output = host_buffer(&ctx, shape.attention_output_bf16_bytes()?)?;
    let router_weight = host_buffer(&ctx, shape.router_weight_bf16_bytes())?;
    let router_input = host_buffer(&ctx, shape.router_input_f32_bytes()?)?;
    let router_logits = host_buffer(&ctx, shape.router_logits_f32_bytes()?)?;
    let route_aux = host_buffer(
        &ctx,
        shape.route_aux_bytes(S14RoutePostprocessGpuMode::BiasTop6)?,
    )?;
    let expert_ids = host_buffer(&ctx, shape.route_output_bytes()?)?;
    let route_weights = host_buffer(&ctx, shape.route_output_bytes()?)?;
    let status = host_buffer(&ctx, 4)?;

    let zero_query = vec![0u8; shape.query_bf16_bytes()? as usize];
    let zero_committed = vec![0u8; shape.committed_window_bf16_bytes()? as usize];
    let zero_current = vec![0u8; shape.current_block_kv_bf16_bytes()? as usize];
    let zero_sink = vec![0u8; shape.sink_f32_bytes() as usize];
    let mut rope_values = Vec::with_capacity(shape.block_size as usize * 64);
    for position in shape.base_position..shape.base_position + shape.block_size {
        rope_values.extend_from_slice(&position_rope_cos_sin(position, 0)?);
    }
    let poison_attention = vec![0xffu8; shape.attention_output_bf16_bytes()? as usize];
    let zero_router_weight = vec![0u8; shape.router_weight_bf16_bytes() as usize];
    let zero_router_input = vec![0u8; shape.router_input_f32_bytes()? as usize];
    let poison_logits = vec![0xffu8; shape.router_logits_f32_bytes()? as usize];
    let zero_aux = vec![0u8; shape.route_aux_bytes(S14RoutePostprocessGpuMode::BiasTop6)? as usize];
    let poison_ids = vec![0xffu8; shape.route_output_bytes()? as usize];
    let poison_weights = vec![0xffu8; shape.route_output_bytes()? as usize];
    unsafe {
        query.write_at(0, &zero_query);
        committed.write_at(0, &zero_committed);
        current.write_at(0, &zero_current);
        sink.write_at(0, &zero_sink);
        rope.write_at(0, bytemuck::cast_slice(&rope_values));
        attention_output.write_at(0, &poison_attention);
        router_weight.write_at(0, &zero_router_weight);
        router_input.write_at(0, &zero_router_input);
        router_logits.write_at(0, &poison_logits);
        route_aux.write_at(0, &zero_aux);
        expert_ids.write_at(0, &poison_ids);
        route_weights.write_at(0, &poison_weights);
        status.write_at(0, &0u32.to_le_bytes());
    }

    let recorder = S14CausalBlockAttentionRouterRecorder::bind(
        &ctx,
        shape,
        S14RoutePostprocessGpuMode::BiasTop6,
        S14CausalBlockAttentionRouterBindings {
            query_bf16: StorageBufferSlice::whole(&query),
            committed_window_kv_bf16: StorageBufferSlice::whole(&committed),
            current_block_kv_bf16: StorageBufferSlice::whole(&current),
            sink_f32: StorageBufferSlice::whole(&sink),
            rope_f32: StorageBufferSlice::whole(&rope),
            attention_output_bf16: StorageBufferSlice::whole(&attention_output),
            router_weight_bf16: StorageBufferSlice::whole(&router_weight),
            router_input_f32: StorageBufferSlice::whole(&router_input),
            router_logits_f32: StorageBufferSlice::whole(&router_logits),
            route_aux: StorageBufferSlice::whole(&route_aux),
            expert_ids_u32: StorageBufferSlice::whole(&expert_ids),
            route_weights_f32: StorageBufferSlice::whole(&route_weights),
            sticky_status_u32: StorageBufferSlice::whole(&status),
        },
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
    let receipt = unsafe {
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let receipt = recorder.record(&ctx, command)?;
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device.queue_submit(
            ctx.q_graphics,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            fence,
        )?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        receipt
    };
    receipt.validate()?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

    let status_code = unsafe { *(status.mapped() as *const u32) };
    validate_causal_block_attention_router_status(status_code)?;
    let attention_values = unsafe {
        std::slice::from_raw_parts(
            attention_output.mapped() as *const u16,
            (shape.attention_output_bf16_bytes()? / 2) as usize,
        )
    };
    if attention_values.iter().any(|&value| value & 0x7fff != 0) {
        bail!("K=4 attention 零输入输出包含非零数值");
    }
    let logits = unsafe {
        std::slice::from_raw_parts(
            router_logits.mapped() as *const f32,
            (shape.router_logits_f32_bytes()? / 4) as usize,
        )
    };
    if logits.iter().any(|&value| value.to_bits() != 0) {
        bail!("K=4 router 零权重 logits 不是逐元素+0");
    }
    let ids = unsafe {
        std::slice::from_raw_parts(
            expert_ids.mapped() as *const u32,
            (shape.route_output_bytes()? / 4) as usize,
        )
    };
    let weights = unsafe {
        std::slice::from_raw_parts(
            route_weights.mapped() as *const f32,
            (shape.route_output_bytes()? / 4) as usize,
        )
    };
    for row in 0..shape.block_size as usize {
        let start = row * 6;
        if ids[start..start + 6] != [0, 1, 2, 3, 4, 5] {
            bail!(
                "K=4 route row{row} expert IDs 漂移: {:?}",
                &ids[start..start + 6]
            );
        }
        if weights[start..start + 6]
            .iter()
            .any(|&weight| (weight - 0.25).abs() > 1.0e-6)
        {
            bail!(
                "K=4 route row{row} weights 漂移: {:?}",
                &weights[start..start + 6]
            );
        }
    }
    println!(
        "status=pass K={} device_rows={} attention_dispatches={} router_weight_scans={} route_dispatches={} serial_token_forward_calls={} hc_hidden_integration_complete={} wall_ms={wall_ms:.4}",
        shape.block_size,
        receipt.device_rows,
        receipt.attention_dispatch_calls,
        receipt.router_weight_scan_calls,
        receipt.route_postprocess_dispatch_calls,
        receipt.serial_token_forward_calls,
        receipt.hc_hidden_integration_complete,
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.destroy_command_pool(pool, None);
    }
    recorder.destroy(&ctx);
    status.destroy(&ctx);
    route_weights.destroy(&ctx);
    expert_ids.destroy(&ctx);
    route_aux.destroy(&ctx);
    router_logits.destroy(&ctx);
    router_input.destroy(&ctx);
    router_weight.destroy(&ctx);
    attention_output.destroy(&ctx);
    rope.destroy(&ctx);
    sink.destroy(&ctx);
    current.destroy(&ctx);
    committed.destroy(&ctx);
    query.destroy(&ctx);
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
