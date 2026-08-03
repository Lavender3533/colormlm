//! K=4 单层 HC/QKV -> grouped MoE -> next-layer hidden 的真实 Vulkan 集成门。

use anyhow::{Context, Result, bail};
use ash::vk;
use polaris_s14_runner::{
    EXPERT_PAGE_BYTES, LayerCausalBatchPlan, MaterializedTokenSource, RouteDecision,
    build_layer_causal_batch_plan,
};
use ssd_inference::{
    GpuBuffer, VulkanContext,
    s14_causal_block_grouped_graph::{
        S14CausalBlockGroupedMoeRecorder, S14CausalBlockRecordedGroupedMoe,
    },
    s14_causal_block_grouped_moe_recorder::{
        S14CausalBlockGroupedMoeStaticLayerResources, S14CausalBlockGroupedMoeVulkanRecorder,
    },
    s14_causal_block_hc_qkv_adapter::S14CausalBlockProductionHcQkvAdapter,
    s14_causal_block_hc_qkv_recorder::{
        S14CausalBlockHcQkvLayerResources, S14CausalBlockHcQkvResourceProvider,
        S14CausalBlockHcQkvWeightOffsets, S14CausalBlockHiddenBank, S14CausalBlockOwnedBufferSlice,
        S14CausalBlockProductionHcQkvLayerRecorder,
    },
    s14_causal_block_layer::{
        S14CausalBlockAttentionRouterOutput, S14CausalBlockFullDepthBackend,
        S14CausalBlockGroupedMoeOutput, S14CausalBlockHiddenBinding, S14CausalBlockLayerBackend,
        S14CausalBlockLayerInput, S14CausalBlockLayerRangePlan, S14CausalBlockPhysicalRange,
        S14CausalBlockUnionBankBinding, S14CausalBlockUnionMaterializeReceipt,
    },
    s14_causal_block_moe_adapter::S14CausalBlockVulkanMoeAdapter,
    s14_causal_block_vulkan_backend::S14CausalBlockVulkanBackend,
    s14_dynamic_routed_page_plan::{RoutedProjection, RoutedRangePart},
    s14_position1_attention::position_rope_cos_sin,
    s14_route_postprocess_gpu::S14RoutePostprocessGpuMode,
    s14_vulkan::S14RaggedBranchOffsets,
};
use std::{
    collections::BTreeSet,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Instant,
};

const K: usize = 4;
const BASE_POSITION: u32 = 1;
const LAYER: u8 = 0;
const HIDDEN: usize = 4096;
const HC_STREAMS: usize = 4;
const KV: usize = 512;
const MAX_K: usize = 8;
const ROUTER_EXPERTS: usize = 256;
const HC_STATIC_BYTES: usize = 8192 * HIDDEN;
const MOE_STATIC_BYTES: usize = 2048 * HIDDEN;
const ROUTED_WEIGHT_BYTES: u64 = 4_194_304;
const ROUTED_SCALE_BYTES: u64 = 262_144;

#[derive(Clone)]
struct ZeroHcProvider {
    resources: S14CausalBlockHcQkvLayerResources,
}

impl fmt::Debug for ZeroHcProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ZeroHcProvider").finish()
    }
}

impl S14CausalBlockHcQkvResourceProvider for ZeroHcProvider {
    fn prepare_layer(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> std::result::Result<S14CausalBlockHcQkvLayerResources, String> {
        if input.layer != self.resources.layer {
            return Err("single-layer HC provider layer drift".into());
        }
        Ok(self.resources.clone())
    }
}

struct FixtureMoeAdapter {
    ctx: Arc<VulkanContext>,
    recorder: Option<S14CausalBlockGroupedMoeVulkanRecorder>,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    bank: Option<S14CausalBlockUnionBankBinding>,
    captured: Option<(S14CausalBlockHiddenBinding, Vec<RouteDecision>)>,
    materialized: bool,
    completed_layers: usize,
    submits: Arc<AtomicU32>,
    destroyed: bool,
}

impl fmt::Debug for FixtureMoeAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FixtureMoeAdapter")
            .field("completed_layers", &self.completed_layers)
            .field("materialized", &self.materialized)
            .finish_non_exhaustive()
    }
}

impl FixtureMoeAdapter {
    fn new(
        ctx: Arc<VulkanContext>,
        recorder: S14CausalBlockGroupedMoeVulkanRecorder,
        submits: Arc<AtomicU32>,
    ) -> Result<Self> {
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
        Ok(Self {
            ctx,
            recorder: Some(recorder),
            command_pool,
            command,
            fence,
            bank: None,
            captured: None,
            materialized: false,
            completed_layers: 0,
            submits,
            destroyed: false,
        })
    }

    fn recorder(&mut self) -> Result<&mut S14CausalBlockGroupedMoeVulkanRecorder> {
        self.recorder.as_mut().context("grouped recorder destroyed")
    }

    fn cleanup_active(&mut self, completed_layers: usize) -> Result<()> {
        if self.bank.is_some() {
            self.recorder()?.finish_block_after_drain(true)?;
            self.bank = None;
            self.captured = None;
            self.materialized = false;
        }
        if completed_layers != self.completed_layers {
            bail!(
                "fixture MoE abort completed_layers drift: reported={completed_layers} expected={}",
                self.completed_layers
            );
        }
        Ok(())
    }
}

impl S14CausalBlockVulkanMoeAdapter for FixtureMoeAdapter {
    fn begin_block(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> std::result::Result<(), String> {
        let result = (|| -> Result<()> {
            if self.destroyed || self.bank.is_some() || !matches!(block_size, 4 | 8) {
                bail!("fixture MoE begin lifecycle/K invalid");
            }
            self.recorder()?.begin_block(base_position, block_size)?;
            self.bank = Some(bank);
            self.completed_layers = 0;
            Ok(())
        })();
        result.map_err(|error| format!("{error:#}"))
    }

    fn capture_attention_router_output(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
        output: &S14CausalBlockAttentionRouterOutput,
    ) -> std::result::Result<(), String> {
        let result = (|| -> Result<()> {
            if self.bank.is_none()
                || self.captured.is_some()
                || input.layer != LAYER
                || output.routes.len() != K
                || output.forward_calls != 1
            {
                bail!("fixture MoE attention capture identity drift");
            }
            self.captured = Some((output.post_attention_hidden, output.routes.clone()));
            Ok(())
        })();
        result.map_err(|error| format!("{error:#}"))
    }

    fn materialize_union_ranges(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> std::result::Result<S14CausalBlockUnionMaterializeReceipt, String> {
        let result = (|| -> Result<S14CausalBlockUnionMaterializeReceipt> {
            if self.bank != Some(bank)
                || self.captured.is_none()
                || self.materialized
                || range_plan.layer != LAYER
                || range_plan.block_size != K
                || range_plan.union_expert_bytes > bank.allocated_bank_bytes
            {
                bail!("fixture zero union materialize identity/capacity drift");
            }
            self.materialized = true;
            Ok(S14CausalBlockUnionMaterializeReceipt {
                layer: LAYER,
                bank_index: bank.bank_index,
                unique_experts: range_plan.unique_experts,
                physical_ranges: range_plan.physical_ranges,
                uploaded_bytes: range_plan.union_expert_bytes,
                materialize_calls: 1,
            })
        })();
        result.map_err(|error| format!("{error:#}"))
    }

    fn run_grouped_moe(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> std::result::Result<S14CausalBlockGroupedMoeOutput, String> {
        let result = (|| -> Result<S14CausalBlockGroupedMoeOutput> {
            let (captured_hidden, captured_routes) = self
                .captured
                .as_ref()
                .context("fixture MoE missing routes")?;
            if self.bank != Some(bank)
                || !self.materialized
                || *captured_hidden != post_attention_hidden
                || captured_routes != routes
                || batch_plan.layer != LAYER
                || range_plan.layer != LAYER
            {
                bail!("fixture grouped MoE route/range/hidden drift");
            }
            unsafe {
                self.ctx.device.reset_fences(&[self.fence])?;
                self.ctx
                    .device
                    .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())?;
                self.ctx.device.begin_command_buffer(
                    self.command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
            }
            let ctx = Arc::clone(&self.ctx);
            let command = self.command;
            let recorded: S14CausalBlockRecordedGroupedMoe = self.recorder()?.record_grouped_moe(
                &ctx,
                command,
                bank,
                post_attention_hidden,
                routes,
                batch_plan,
                range_plan,
            )?;
            unsafe {
                self.ctx.device.end_command_buffer(self.command)?;
                let commands = [self.command];
                self.ctx.device.queue_submit(
                    self.ctx.q_graphics,
                    &[vk::SubmitInfo::default().command_buffers(&commands)],
                    self.fence,
                )?;
                self.ctx
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?;
            }
            if recorded.recorder_calls != 1
                || recorded.serial_token_forward_calls != 0
                || recorded.grouped_expert_work_items != batch_plan.unique_experts
                || recorded.lane_assignments != batch_plan.assignments
                || !self.recorder()?.owns_output_hidden(recorded.output_hidden)
            {
                bail!("fixture grouped recorder coverage receipt drift");
            }
            self.submits.fetch_add(1, Ordering::SeqCst);
            self.completed_layers = 1;
            self.captured = None;
            self.materialized = false;
            Ok(S14CausalBlockGroupedMoeOutput {
                output_hidden: recorded.output_hidden,
                grouped_submit_calls: 1,
                serial_token_forward_calls: 0,
                unique_experts: recorded.grouped_expert_work_items,
            })
        })();
        result.map_err(|error| format!("{error:#}"))
    }

    fn seal_and_drain(&mut self, _completed_layers: usize) -> std::result::Result<(), String> {
        Err("fixture single-layer adapter forbids FullDepth seal".into())
    }

    fn drain_and_abort(&mut self, completed_layers: usize) -> std::result::Result<(), String> {
        self.cleanup_active(completed_layers)
            .map_err(|error| format!("{error:#}"))
    }

    fn finish_validated_block(&mut self) -> std::result::Result<(), String> {
        Err("fixture single-layer adapter has no sealed block".into())
    }

    fn destroy(&mut self) -> std::result::Result<(), String> {
        let result = (|| -> Result<()> {
            if self.destroyed {
                return Ok(());
            }
            self.cleanup_active(self.completed_layers)?;
            if let Some(recorder) = self.recorder.as_mut() {
                recorder.destroy()?;
            }
            self.recorder = None;
            unsafe {
                self.ctx.device.destroy_fence(self.fence, None);
                self.ctx
                    .device
                    .destroy_command_pool(self.command_pool, None);
            }
            self.destroyed = true;
            Ok(())
        })();
        result.map_err(|error| format!("{error:#}"))
    }
}

fn main() -> Result<()> {
    let ctx = Arc::new(VulkanContext::init()?);
    let hidden_capacity = (MAX_K * HC_STREAMS * HIDDEN * 2) as u64;
    let hidden_a = Arc::new(host_buffer(&ctx, hidden_capacity)?);
    let hidden_b = Arc::new(host_buffer(&ctx, hidden_capacity)?);
    let hc_static = Arc::new(host_buffer(&ctx, HC_STATIC_BYTES as u64)?);
    let moe_static = Arc::new(host_buffer(&ctx, MOE_STATIC_BYTES as u64)?);
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
        hc_static.write_at(0, &vec![0u8; HC_STATIC_BYTES]);
        moe_static.write_at(0, &vec![0u8; MOE_STATIC_BYTES]);
        committed.write_at(0, &vec![0u8; committed.size() as usize]);
        rotated_current.write_at(0, &vec![0xffu8; rotated_current.size() as usize]);
        rope.write_at(0, bytemuck::cast_slice(&rope_values));
        route_aux.write_at(0, &vec![0u8; route_aux.size() as usize]);
    }

    let hc_resources = S14CausalBlockHcQkvLayerResources {
        layer: LAYER,
        static_arena: hc_static.clone(),
        static_logical_bytes: HC_STATIC_BYTES as u64,
        weights: zero_hc_offsets(),
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
    let hc_recorder = S14CausalBlockProductionHcQkvLayerRecorder::new(
        ctx.clone(),
        ZeroHcProvider {
            resources: hc_resources.clone(),
        },
        hidden_banks.clone(),
    )?;
    let input_hidden = hc_recorder.hidden_binding(0, K, 0)?;
    let hc_adapter = S14CausalBlockProductionHcQkvAdapter::new(hc_recorder);

    let moe_recorder = S14CausalBlockGroupedMoeVulkanRecorder::new_with_static_layer(
        ctx.clone(),
        S14CausalBlockGroupedMoeStaticLayerResources {
            layer: LAYER,
            buffer: moe_static.clone(),
            logical_bytes: MOE_STATIC_BYTES as u64,
            hc_fn: 0,
            hc_scale: 0,
            hc_base: 0,
            ffn_norm: 0,
            shared: S14RaggedBranchOffsets {
                w1: 0,
                s1: 0,
                w3: 0,
                s3: 0,
                w2: 0,
                s2: 0,
            },
        },
    )?;
    let grouped_submits = Arc::new(AtomicU32::new(0));
    let moe_adapter = FixtureMoeAdapter::new(ctx.clone(), moe_recorder, grouped_submits.clone())?;
    let mut backend = S14CausalBlockVulkanBackend::with_moe_adapter(moe_adapter);
    backend
        .install_hc_qkv_adapter(hc_adapter)
        .map_err(anyhow::Error::msg)?;

    let range_plan = zero_range_plan();
    let union = host_buffer(&ctx, range_plan.union_expert_bytes)?;
    unsafe { union.write_at(0, &vec![0u8; union.size() as usize]) };
    let bank = S14CausalBlockUnionBankBinding {
        bank_index: 0,
        buffer: union.handle(),
        allocated_bank_bytes: union.size(),
    };
    let token_ids = [31, 32, 33, 34];
    let input = S14CausalBlockLayerInput {
        base_position: BASE_POSITION,
        layer: LAYER,
        input_token_ids: &token_ids,
        input_hidden,
        source: MaterializedTokenSource::SpeculativeDraft,
    };

    let started = Instant::now();
    let begin = backend
        .begin_full_depth_block(bank, BASE_POSITION, K)
        .map_err(anyhow::Error::msg)?;
    let attention = backend
        .run_k_lane_attention_router(&input)
        .map_err(anyhow::Error::msg)?;
    let batch_plan = build_layer_causal_batch_plan(&attention.routes)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if batch_plan.assignments != K * 6 || batch_plan.unique_experts != 6 {
        bail!("single-layer route-slot plan drift");
    }
    let materialized = backend
        .materialize_union_ranges(bank, &range_plan)
        .map_err(anyhow::Error::msg)?;
    let grouped = backend
        .run_grouped_moe(
            bank,
            attention.post_attention_hidden,
            &attention.routes,
            &batch_plan,
            &range_plan,
        )
        .map_err(anyhow::Error::msg)?;

    if begin.serial_token_forward_calls != 0
        || attention.forward_calls != 1
        || grouped.grouped_submit_calls != 1
        || grouped.serial_token_forward_calls != 0
        || grouped.unique_experts != 6
        || grouped.output_hidden.generation != input_hidden.generation + 2
        || grouped.output_hidden.block_size != K
        || grouped.output_hidden == attention.post_attention_hidden
        || materialized.materialize_calls != 1
        || materialized.physical_ranges != 36
    {
        bail!("single-layer backend HC/QKV->MoE receipt/next hidden identity drift");
    }
    let abort = backend
        .drain_and_abort_full_depth_block(1)
        .map_err(anyhow::Error::msg)?;
    if !abort.drained || abort.completed_layers != 1 || !backend.is_idle() {
        bail!("single-layer backend drain/abort lifecycle drift");
    }
    backend.destroy_moe_adapter().map_err(anyhow::Error::msg)?;
    backend
        .destroy_hc_qkv_adapter()
        .map_err(anyhow::Error::msg)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;

    println!(
        "status=pass K={K} layers=1 hc_command_graph_submits=1 grouped_command_graph_submits={} total_submits={} serial_token_forward_calls=0 route_slots={} bf16_finite=true next_layer_generation={} next_layer_identity=true drained=true aborted=true wall_ms={wall_ms:.3}",
        grouped_submits.load(Ordering::SeqCst),
        1 + grouped_submits.load(Ordering::SeqCst),
        batch_plan.assignments,
        grouped.output_hidden.generation,
    );

    drop(backend);
    union.destroy(&ctx);
    drop(hidden_banks);
    drop(hc_resources);
    route_aux.destroy(&ctx);
    rope.destroy(&ctx);
    rotated_current.destroy(&ctx);
    committed.destroy(&ctx);
    moe_static.destroy(&ctx);
    hc_static.destroy(&ctx);
    hidden_b.destroy(&ctx);
    hidden_a.destroy(&ctx);
    Ok(())
}

fn zero_range_plan() -> S14CausalBlockLayerRangePlan {
    let mut ranges = Vec::with_capacity(36);
    for expert_id in 0u16..6 {
        for (projection, part, bytes) in [
            (
                RoutedProjection::W1,
                RoutedRangePart::Weight,
                ROUTED_WEIGHT_BYTES,
            ),
            (
                RoutedProjection::W1,
                RoutedRangePart::Scale,
                ROUTED_SCALE_BYTES,
            ),
            (
                RoutedProjection::W2,
                RoutedRangePart::Weight,
                ROUTED_WEIGHT_BYTES,
            ),
            (
                RoutedProjection::W2,
                RoutedRangePart::Scale,
                ROUTED_SCALE_BYTES,
            ),
            (
                RoutedProjection::W3,
                RoutedRangePart::Weight,
                ROUTED_WEIGHT_BYTES,
            ),
            (
                RoutedProjection::W3,
                RoutedRangePart::Scale,
                ROUTED_SCALE_BYTES,
            ),
        ] {
            ranges.push(S14CausalBlockPhysicalRange {
                expert_id,
                projection,
                part,
                tensor: format!("fixture.L{LAYER}.E{expert_id}.{projection:?}.{part:?}"),
                range_key: format!("fixture:{expert_id}:{projection:?}:{part:?}"),
                bytes,
            });
        }
    }
    let unique_experts = ranges
        .iter()
        .map(|range| range.expert_id)
        .collect::<BTreeSet<_>>()
        .len();
    S14CausalBlockLayerRangePlan {
        layer: LAYER,
        block_size: K,
        unique_experts,
        physical_ranges: ranges.len(),
        union_expert_bytes: unique_experts as u64 * EXPERT_PAGE_BYTES,
        ranges,
    }
}

fn zero_hc_offsets() -> S14CausalBlockHcQkvWeightOffsets {
    S14CausalBlockHcQkvWeightOffsets {
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
    }
}

fn owned(buffer: &Arc<GpuBuffer>) -> S14CausalBlockOwnedBufferSlice {
    S14CausalBlockOwnedBufferSlice {
        buffer: buffer.clone(),
        offset: 0,
        bytes: buffer.size(),
    }
}

fn host_buffer(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        ctx,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
}
