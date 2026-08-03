//! Concrete K=4 HC/QKV/attention/router recorder 的秒级 Vulkan 数值门。

use anyhow::{bail, Result};
use ash::vk;
use polaris_s14_runner::MaterializedTokenSource;
use ssd_inference::{
    s14_causal_block_hc_qkv_adapter::{
        S14CausalBlockProductionHcQkvAdapter, S14CausalBlockVulkanHcQkvAdapter,
    },
    s14_causal_block_hc_qkv_recorder::{
        S14CausalBlockHcQkvLayerResources, S14CausalBlockHcQkvResourceProvider,
        S14CausalBlockHcQkvWeightOffsets, S14CausalBlockHiddenBank, S14CausalBlockOwnedBufferSlice,
        S14CausalBlockProductionHcQkvLayerRecorder,
    },
    s14_causal_block_layer::S14CausalBlockLayerInput,
    s14_causal_block_vulkan_backend::S14CausalBlockVulkanBackend,
    s14_position1_attention::position_rope_cos_sin,
    s14_route_postprocess_gpu::S14RoutePostprocessGpuMode,
    GpuBuffer, VulkanContext,
};
use std::{fmt, sync::Arc, time::Instant};

const K: usize = 4;
const BASE_POSITION: u32 = 1;
const LAYER: u8 = 0;
const HIDDEN: usize = 4096;
const HC_STREAMS: usize = 4;
const KV: usize = 512;
const MAX_K: usize = 8;
const ROUTER_EXPERTS: usize = 256;
const STATIC_BYTES: usize = 8192 * HIDDEN;

#[derive(Clone)]
struct ZeroProvider {
    resources: S14CausalBlockHcQkvLayerResources,
}

impl fmt::Debug for ZeroProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ZeroProvider").finish()
    }
}

impl S14CausalBlockHcQkvResourceProvider for ZeroProvider {
    fn prepare_layer(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> std::result::Result<S14CausalBlockHcQkvLayerResources, String> {
        if input.layer != self.resources.layer {
            return Err("numeric provider layer identity drift".into());
        }
        Ok(self.resources.clone())
    }
}

fn main() -> Result<()> {
    let ctx = Arc::new(VulkanContext::init()?);
    let hidden_capacity = (MAX_K * HC_STREAMS * HIDDEN * 2) as u64;
    let hidden_a = Arc::new(host_buffer(&ctx, hidden_capacity)?);
    let hidden_b = Arc::new(host_buffer(&ctx, hidden_capacity)?);
    let static_arena = Arc::new(host_buffer(&ctx, STATIC_BYTES as u64)?);
    let committed = Arc::new(host_buffer(&ctx, BASE_POSITION as u64 * KV as u64 * 2)?);
    let rotated_current = Arc::new(host_buffer(&ctx, (K * KV * 2) as u64)?);
    let rope = Arc::new(host_buffer(&ctx, (K * 64 * 4) as u64)?);
    let route_aux = Arc::new(host_buffer(&ctx, (ROUTER_EXPERTS * 4) as u64)?);

    let mut rope_values = Vec::with_capacity(K * 64);
    for position in BASE_POSITION..BASE_POSITION + K as u32 {
        rope_values.extend_from_slice(&position_rope_cos_sin(position, 0)?);
    }
    unsafe {
        hidden_a.write_at(0, &vec![0u8; hidden_capacity as usize]);
        hidden_b.write_at(0, &vec![0xffu8; hidden_capacity as usize]);
        static_arena.write_at(0, &vec![0u8; STATIC_BYTES]);
        committed.write_at(0, &vec![0u8; committed.size() as usize]);
        rotated_current.write_at(0, &vec![0xffu8; rotated_current.size() as usize]);
        rope.write_at(0, bytemuck::cast_slice(&rope_values));
        route_aux.write_at(0, &vec![0u8; route_aux.size() as usize]);
    }

    // 所有只读 tensor 共用同一段零 arena；最大正式 Range 是 8192x4096 FP8。
    let weights = S14CausalBlockHcQkvWeightOffsets {
        hc_attn_fn: 0,
        hc_attn_scale: 0,
        hc_attn_base: 0,
        attn_norm: 0,
        wq_a_weight: 0,
        wq_a_scale: 0,
        q_norm: 0,
        wq_b_weight: 0,
        wq_b_scale: 0,
        wkv_weight: 0,
        wkv_scale: 0,
        kv_norm: 0,
        attention_sink: 0,
        wo_a_weight: 0,
        wo_a_scale: 0,
        wo_b_weight: 0,
        wo_b_scale: 0,
        hc_ffn_fn: 0,
        hc_ffn_scale: 0,
        hc_ffn_base: 0,
        ffn_norm: 0,
        router_weight: 0,
    };
    let resources = S14CausalBlockHcQkvLayerResources {
        layer: LAYER,
        static_arena: static_arena.clone(),
        static_logical_bytes: STATIC_BYTES as u64,
        weights,
        route_mode: S14RoutePostprocessGpuMode::BiasTop6,
        committed_window_kv_bf16: owned(&committed),
        rotated_current_block_kv_bf16: owned(&rotated_current),
        rope_f32: owned(&rope),
        route_aux: owned(&route_aux),
    };
    let hidden_banks = [
        S14CausalBlockHiddenBank {
            buffer: hidden_a.clone(),
            offset: 0,
            capacity_bytes: hidden_capacity,
        },
        S14CausalBlockHiddenBank {
            buffer: hidden_b.clone(),
            offset: 0,
            capacity_bytes: hidden_capacity,
        },
    ];

    let recorder = S14CausalBlockProductionHcQkvLayerRecorder::new(
        ctx.clone(),
        ZeroProvider {
            resources: resources.clone(),
        },
        hidden_banks.clone(),
    )?;
    let input_hidden = recorder.hidden_binding(0, K, 0)?;
    let token_ids = [11, 12, 13, 14];
    let input = S14CausalBlockLayerInput {
        base_position: BASE_POSITION,
        layer: LAYER,
        input_token_ids: &token_ids,
        input_hidden,
        source: MaterializedTokenSource::SpeculativeDraft,
    };
    let mut adapter = S14CausalBlockProductionHcQkvAdapter::new(recorder);
    adapter
        .begin_block(BASE_POSITION, K)
        .map_err(anyhow::Error::msg)?;
    let started = Instant::now();
    let recorded = adapter
        .run_k_lane_hc_qkv_attention_router(&input)
        .map_err(anyhow::Error::msg)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    recorded
        .receipt
        .validate(&input, &recorded.output)
        .map_err(anyhow::Error::msg)?;

    if recorded.receipt.command_graph_submit_calls != 1
        || recorded.receipt.serial_token_forward_calls != 0
        || !recorded.receipt.hc_hidden_integration_complete
    {
        bail!("K=4 recorder 强回执未证明单 command graph/submit 与完整 HC 接线");
    }
    assert_zero_finite_bf16(
        &hidden_b,
        (K * HC_STREAMS * HIDDEN) as usize,
        "post-attention hidden",
    )?;
    assert_zero_finite_bf16(&rotated_current, K * KV, "current KV")?;
    for (row, route) in recorded.output.routes.iter().enumerate() {
        if route.expert_ids != [0, 1, 2, 3, 4, 5]
            || route
                .weights
                .iter()
                .any(|weight| !weight.is_finite() || (*weight - 0.25).abs() > 1.0e-6)
        {
            bail!("K=4 route row{row} 数值漂移: {route:?}");
        }
    }
    // recorder 成功返回意味着同一 sticky status 已逐级通过 Q、raw current-KV、
    // attention output、FFN HC router input/logits 的真实 shader 非有限值检查。
    adapter.drain_and_abort(0).map_err(anyhow::Error::msg)?;
    adapter.destroy().map_err(anyhow::Error::msg)?;
    drop(adapter);

    // 用一个未开始 block 的同型 concrete recorder 实际穿过 backend 的对象安全注入边界。
    let injectable = S14CausalBlockProductionHcQkvLayerRecorder::new(
        ctx.clone(),
        ZeroProvider {
            resources: resources.clone(),
        },
        hidden_banks.clone(),
    )?;
    let mut backend = S14CausalBlockVulkanBackend::default();
    backend
        .install_hc_qkv_adapter(S14CausalBlockProductionHcQkvAdapter::new(injectable))
        .map_err(anyhow::Error::msg)?;
    backend
        .destroy_hc_qkv_adapter()
        .map_err(anyhow::Error::msg)?;
    drop(backend);

    println!(
        "status=pass K={K} command_buffers=1 submits={} serial_token_forward_calls={} hc_hidden_integration_complete={} q_finite=true current_kv_finite=true router_input_finite=true backend_injectable=true wall_ms={wall_ms:.3}",
        recorded.receipt.command_graph_submit_calls,
        recorded.receipt.serial_token_forward_calls,
        recorded.receipt.hc_hidden_integration_complete,
    );

    drop(hidden_banks);
    drop(resources);
    route_aux.destroy(&ctx);
    rope.destroy(&ctx);
    rotated_current.destroy(&ctx);
    committed.destroy(&ctx);
    static_arena.destroy(&ctx);
    hidden_b.destroy(&ctx);
    hidden_a.destroy(&ctx);
    Ok(())
}

fn owned(buffer: &Arc<GpuBuffer>) -> S14CausalBlockOwnedBufferSlice {
    S14CausalBlockOwnedBufferSlice {
        buffer: buffer.clone(),
        offset: 0,
        bytes: buffer.size(),
    }
}

fn assert_zero_finite_bf16(buffer: &GpuBuffer, elements: usize, label: &str) -> Result<()> {
    let values = unsafe { std::slice::from_raw_parts(buffer.mapped() as *const u16, elements) };
    if values.iter().any(|value| value & 0x7f80 == 0x7f80) {
        bail!("K=4 {label} 含 NaN/Inf BF16");
    }
    if values.iter().any(|value| value & 0x7fff != 0) {
        bail!("K=4 零输入 {label} 不是逐元素零");
    }
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
