//! Polaris S14 production paged N=8 连续状态边界门。
//!
//! 43 层、terminal prelude、32 个 head chunk 与最终 readback 共用一个双队列
//! timeline。每层先执行 router-only probe，只回读 top-6 IDs/weights，再从固定
//! catalog 物化36个真实 Range 并接续 MoE continuation。当前首版为了闭合正确性，
//! 如实保留逐层 probe host wait；dynamic routed copy 已接入同一个双队列
//! timeline，不再为每层单独 submit+fence wait。
//! position0 产生 token5，position1 消费 token5 并产生 token223；position2..7
//! 继续在线生成，其中 position3/7 必须真实跨过两次 ratio4 compressed block 边界，
//! position4+ 必须运行在线 compressed indexer。这不是旧固定 manifest replay，
//! 也不预注册 position2+ 的输出 token。

use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{DecoderStateV1, Position0WholeTokenManifest};
use sha2::{Digest, Sha256};
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_dynamic_page_cache_readiness::{
        materialize_dynamic_page_plan, materialize_planned_range_asset, DynamicPageFetchMode,
    },
    s14_dynamic_routed_page_plan::FullDepthExpertCatalog,
    s14_head_chunk_argmax::S14_HEAD_CHUNK_COUNT,
    s14_input_asset_plan::S14InputAssetPlanner,
    s14_position0_layer_backend::{
        S14Position0PersistentHostResources, S14Position0SynchronousVulkanLayerAdapter,
    },
    s14_position0_layer_program::S14Position0FullDepthLayerProgram,
    s14_position0_paged_layer_bridge::{
        S14Position0PagedLayerBridge, S14Position0PagedLayerStageReceipt,
    },
    s14_position0_paged_layer_timeline::{
        validate_production_paged_position, S14Position0PagedLayerTimeline,
        S14Position0PagedLayerTimelineState,
    },
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_state_writeback::{
        stage_payloads, S14Position0FullDepthStateRecordingProgram, S14Position0StateReadback,
    },
    s14_position0_synchronous_layer_pager::S14Position0DeviceHiddenSlot,
    s14_position0_synchronous_layer_plan::build_synchronous_layer_plans,
    s14_position0_terminal::S14Position0TerminalChain,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_position0_whole_token::Position0GpuCandidate,
    s14_position0_workspace::S14Position0WorkspaceSlot,
    s14_runtime::S14RuntimePersistentCommandResources,
    s14_whole_token_device::WholeTokenDeviceState,
    VulkanContext,
};
use std::{path::PathBuf, time::Instant};

const LAYER_COUNT: usize = 43;
const LAYER_TRANSFER_START: usize = 0;
const HEAD_TRANSFER_START: usize = LAYER_TRANSFER_START + LAYER_COUNT;
const LAYER_PROBE_COMPUTE_START: usize = 0;
const LAYER_CONTINUATION_COMPUTE_START: usize = LAYER_PROBE_COMPUTE_START + LAYER_COUNT;
const PRELUDE_COMPUTE: usize = LAYER_CONTINUATION_COMPUTE_START + LAYER_COUNT;
const HEAD_COMPUTE_START: usize = PRELUDE_COMPUTE + 1;
const TERMINAL_COMPUTE: usize = HEAD_COMPUTE_START + S14_HEAD_CHUNK_COUNT as usize;
const EXPECTED_PREFIX_TOKENS: [u32; 2] = [5, 223];
const TARGET_TOKEN_COUNT: u32 = 8;

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest_path = root.join(
        "fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let payload_root = PathBuf::from("D:/models/Polaris-S14/range_cache");
    let catalog_path = PathBuf::from("D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json");
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let weights = S14Position0HybridWeightPlan::build(&manifest)?;
    let ctx = VulkanContext::init()?;
    if !ctx.timeline_semaphore || !ctx.has_dedicated_transfer() {
        bail!("production paged whole-token requires timeline semaphore and dedicated transfer");
    }

    let arena = S14Position0PagedWeightArena::new(&ctx, &weights, Some(3 * 1024 * 1024 * 1024))?;
    let layer_program =
        S14Position0FullDepthLayerProgram::build(&manifest, &weights, arena.workspace_layout())?;
    let plans = build_synchronous_layer_plans(&layer_program, &arena)?;
    let mut host = DecoderStateV1::new(4096, manifest.input_token_id)?;
    let input_planner = S14InputAssetPlanner::load_pinned(&catalog_path, &payload_root)?;
    let expert_catalog = FullDepthExpertCatalog::load(&catalog_path)?;
    let page_fetch_mode = match std::env::var("POLARIS_S14_FETCH_MISSING_PAGES") {
        Ok(value) if value == "1" => DynamicPageFetchMode::ExplicitFetch,
        Err(std::env::VarError::NotPresent) => DynamicPageFetchMode::LocalOnly,
        Ok(value) => bail!("POLARIS_S14_FETCH_MISSING_PAGES 只允许未设置或1，actual={value:?}"),
        Err(error) => return Err(error.into()),
    };
    let mut device = WholeTokenDeviceState::new(&ctx, host.native_arena.bytes(), 0)?;
    let mut persistent_host_resources =
        S14Position0PersistentHostResources::new(&ctx, &manifest, &weights, &arena, &payload_root)?;
    let mut persistent_command_resources = S14RuntimePersistentCommandResources::new(&ctx)?;
    let continuous_started = Instant::now();
    let mut output_tokens = Vec::with_capacity(TARGET_TOKEN_COUNT as usize);
    let mut total_online_top6_routes = 0u64;
    let mut total_dynamic_physical_ranges = 0u64;

    for requested_position in 0..TARGET_TOKEN_COUNT {
        validate_production_paged_position(requested_position)?;
        if host.position != requested_position
            || host.commit_epoch != u64::from(requested_position)
            || device.epoch() != host.commit_epoch
            || device.active_bank() != usize::from(host.active_fixed_bank)
        {
            bail!("continuous host/device position/epoch/bank 漂移");
        }
        let position = host.position;
        let base_epoch = host.commit_epoch;
        let input_token_id = host.input_token_id;
        let input_plan =
            input_planner.plan(host.position, host.input_token_id, host.native.max_seq_len)?;
        let input_embedding =
            materialize_planned_range_asset(&input_plan.embedding, &payload_root, page_fetch_mode)?;
        let state_recording = S14Position0FullDepthStateRecordingProgram::build(
            &layer_program,
            arena.workspace_layout(),
            &host.native,
        )?;

        // begin/arm 之后任何失败都必须先排空外部 timeline，销毁引用 candidate 的
        // Vulkan 资源，再回滚 inactive bank。所有 token 私有 owner 都放进 slot，保证
        // `?` 离开执行闭包后仍可按同一顺序收敛。
        let mut timeline_slot = None;
        let mut terminal_slot = None;
        let mut state_readback_slot = None;
        let mut command_step_slot = None;
        let mut command_step_active = false;
        let prologue_command = device.begin_candidate_for_position(&ctx, base_epoch, position)?;
        let mut rollback_guard = CandidateRollbackGuard::new(&mut device, &ctx);
        // 当前 adapter 在整个 graph 生命周期借用 candidate buffer；先转入 InFlight，
        // 后续任何失败都只能 drain/rollback，不能发布 inactive bank。
        device.arm_external_candidate()?;
        rollback_guard.mark_external_armed();
        let candidate_bank = 1usize - device.active_bank();
        let candidate = Position0GpuCandidate {
            ctx: &ctx,
            candidate_state: device.candidate_buffer()?,
            sticky_status: device.sticky_status_buffer()?,
            committed_host_state: &host,
            base_epoch,
            candidate_bank,
        };

        let started = Instant::now();
        let backend = S14Position0SynchronousVulkanLayerAdapter::new(
            &ctx,
            &manifest,
            &weights,
            &arena,
            &payload_root,
            &mut persistent_host_resources,
            candidate,
        )?;
        let persistent_host_step = backend.persistent_step_telemetry();
        let mut backend_slot = Some(backend);
        let mut completion_slot = None;
        let mut payloads_slot = None;
        let mut l42_hidden_sha_slot = None;
        let mut token_elapsed_ms = 0.0f64;
        let mut static_uploaded_bytes = 0u64;
        let mut routed_uploaded_bytes = 0u64;
        let mut router_probe_readback_bytes = 0u64;
        let mut router_probe_host_waits = 0u64;
        let mut streamed_static_transfer_fence_waits = 0u64;
        let dynamic_routed_transfer_fence_waits = 0u64;
        let mut dynamic_physical_ranges = 0u64;
        let mut head_uploaded_bytes = 0u64;
        let token_result = (|| -> Result<()> {
            let backend = backend_slot.as_mut().expect("candidate backend slot");
            backend.bind_input_prologue(&input_plan, &input_embedding)?;
            let input_execution = backend
                .input_execution()
                .ok_or_else(|| anyhow!("S14 input execution plan 未绑定"))?;
            if input_execution.rope_position != position
                || input_execution.window_slot != position % 128
                || input_execution.active_window_tokens != (position + 1).min(128)
            {
                bail!("S14 prologue position/RoPE/window 合同漂移");
            }
            let command_step = persistent_command_resources.begin_step(&ctx)?;
            command_step_slot = Some(command_step.telemetry);
            command_step_active = true;
            let transfer_commands = command_step.transfer_commands;
            let compute_commands = command_step.compute_commands;
            timeline_slot = Some(S14Position0PagedLayerTimeline::new_for_position(
                &ctx, position,
            )?);
            let timeline = timeline_slot.as_mut().expect("candidate timeline slot");

            unsafe {
                backend.record_embedding_prologue(prologue_command, &input_embedding)?;
                timeline.submit_prologue_compute_only(&ctx, prologue_command)?;
            }

            let mut bridge =
                S14Position0PagedLayerBridge::new_for_position(timeline, &plans, position)?;
            if bridge.position() != position {
                bail!("continuous paged bridge position 漂移");
            }
            for (index, plan) in plans.iter().enumerate() {
                let transfer = transfer_commands[LAYER_TRANSFER_START + index];
                let probe_compute = compute_commands[LAYER_PROBE_COMPUTE_START + index];
                let continuation_compute =
                    compute_commands[LAYER_CONTINUATION_COMPUTE_START + index];

                let static_receipt = backend.prepare_router_probe_static(plan)?;
                static_uploaded_bytes = static_uploaded_bytes
                    .checked_add(static_receipt.bytes)
                    .ok_or_else(|| anyhow!("static upload byte counter overflow"))?;
                if static_receipt.bytes != 0 {
                    streamed_static_transfer_fence_waits += 1;
                }

                unsafe { backend.record_paged_layer_router_probe(plan, probe_compute)? };
                let probe_completion = unsafe {
                    bridge.timeline_mut().submit_router_probe_and_wait(
                        &ctx,
                        plan.layer,
                        probe_compute,
                    )?
                };
                if probe_completion.layer != plan.layer
                    || probe_completion.index != index
                    || probe_completion.bank != plan.routed_bank
                    || probe_completion.host_wait_calls != 1
                {
                    bail!("L{} router probe timeline 强回执漂移", plan.layer);
                }
                router_probe_host_waits = router_probe_host_waits
                    .checked_add(u64::from(probe_completion.host_wait_calls))
                    .ok_or_else(|| anyhow!("router probe host wait counter overflow"))?;

                let observed = backend.complete_router_probe_after_wait(probe_completion.layer)?;
                router_probe_readback_bytes = router_probe_readback_bytes
                    .checked_add(observed.readback_bytes)
                    .ok_or_else(|| anyhow!("router probe readback byte counter overflow"))?;
                let observed_expert_ids = observed.route.expert_ids;
                let dynamic_plan = expert_catalog.plan(observed.route)?;
                let materialized =
                    materialize_dynamic_page_plan(&dynamic_plan, &payload_root, page_fetch_mode)
                        .map_err(anyhow::Error::new)
                        .with_context(|| {
                            format!(
                        "position{position} L{} dynamic top-6 Range materialize 失败: experts={:?}",
                        plan.layer, observed_expert_ids
                    )
                        })?;
                let physical_range_count = u64::try_from(materialized.assets.len())
                    .context("dynamic physical Range count 不能表示为u64")?;
                dynamic_physical_ranges = dynamic_physical_ranges
                    .checked_add(physical_range_count)
                    .ok_or_else(|| anyhow!("dynamic physical Range counter overflow"))?;
                println!(
                    "trace=online_top6_dynamic_ranges source=gpu_router_probe position={} layer={} expert_ids={:?} route_weight_bits={:?} physical_ranges={} rust_proof_sha_verified=true cpu_compute_fallbacks=0",
                    position,
                    plan.layer,
                    dynamic_plan.expert_ids,
                    dynamic_plan.route_weights.map(|weight| weight.to_bits()),
                    physical_range_count,
                );
                let routed_receipt = unsafe {
                    backend.record_dynamic_routed_after_probe(
                        plan,
                        &dynamic_plan,
                        &materialized,
                        transfer,
                    )?
                };
                routed_uploaded_bytes = routed_uploaded_bytes
                    .checked_add(routed_receipt.bytes)
                    .ok_or_else(|| anyhow!("routed upload byte counter overflow"))?;

                unsafe {
                    backend.record_paged_layer_dynamic_moe_continuation(
                        plan,
                        &dynamic_plan,
                        &materialized,
                        continuation_compute,
                    )?;
                }
                backend.validate_recorded_layer_binding(plan, plan.routed_bank)?;
                let expected_layer = plan.layer;
                let expected_bank = plan.routed_bank;
                let receipt = unsafe {
                    bridge.submit_next_layer(
                        &ctx,
                        transfer,
                        continuation_compute,
                        |_runtime_plan, timeline_bank| {
                            if timeline_bank != expected_bank {
                                bail!("L{expected_layer} dynamic routed timeline bank 漂移");
                            }
                            Ok(S14Position0PagedLayerStageReceipt {
                                layer: expected_layer,
                                bank: expected_bank,
                                static_uploaded_bytes: static_receipt.bytes,
                                routed_uploaded_bytes: routed_receipt.bytes,
                                hidden_host_bytes: 0,
                            })
                        },
                        |runtime_plan, timeline_bank| {
                            if runtime_plan.layer != expected_layer
                                || timeline_bank != expected_bank
                            {
                                bail!("L{expected_layer} recorded descriptor/bank 漂移");
                            }
                            Ok(())
                        },
                    )?
                };
                if receipt.layer != expected_layer
                    || receipt.index != index
                    || receipt.bank != expected_bank
                    || receipt.stage.static_uploaded_bytes != static_receipt.bytes
                    || receipt.stage.routed_uploaded_bytes != routed_receipt.bytes
                {
                    bail!("L{expected_layer} dynamic routed bridge receipt 漂移");
                }
            }

            let mut tail = bridge.seal_layers()?;
            if tail.position() != position {
                bail!("continuous paged tail position 漂移");
            }
            if tail.final_hidden() != S14Position0DeviceHiddenSlot::B {
                bail!("FullDepth43 L42 final hidden slot 漂移");
            }
            let l42_region = arena
                .workspace_layout()
                .region(S14Position0WorkspaceSlot::HiddenStreamsB);
            let l42_hidden = StorageBufferSlice {
                buffer: arena.workspace(),
                offset: l42_region.offset,
            };
            let candidate_hc = StorageBufferSlice {
                buffer: device.candidate_buffer()?,
                offset: host.native.hc.streams.offset,
            };
            state_readback_slot = Some(S14Position0StateReadback::new(&ctx, &host.native)?);
            terminal_slot = Some(S14Position0TerminalChain::new(&ctx, &arena, l42_hidden)?);
            let state_readback = state_readback_slot
                .as_mut()
                .expect("candidate state readback slot");
            let terminal = terminal_slot.as_mut().expect("candidate terminal slot");

            begin_graphics_command(&ctx, compute_commands[PRELUDE_COMPUTE])?;
            unsafe {
                terminal.record_prelude(&ctx, compute_commands[PRELUDE_COMPUTE])?;
                ctx.device
                    .end_command_buffer(compute_commands[PRELUDE_COMPUTE])?;
                terminal.submit_recorded_prelude(
                    &ctx,
                    tail.timeline_mut(),
                    compute_commands[PRELUDE_COMPUTE],
                )?;
            }

            for chunk in 0..S14_HEAD_CHUNK_COUNT {
                let index = chunk as usize;
                let transfer = transfer_commands[HEAD_TRANSFER_START + index];
                let compute = compute_commands[HEAD_COMPUTE_START + index];
                let recorded = unsafe { backend.record_next_head_transfer(transfer)? };
                if recorded.chunk != u64::from(chunk) || recorded.bank != index % 2 {
                    bail!("head chunk {chunk} recorded receipt 漂移");
                }
                begin_graphics_command(&ctx, compute)?;
                unsafe {
                    terminal.record_head_chunk(&ctx, chunk, compute)?;
                    ctx.device.end_command_buffer(compute)?;
                    terminal.submit_recorded_head(
                        &ctx,
                        tail.timeline_mut(),
                        chunk,
                        transfer,
                        compute,
                        |timeline_bank| {
                            let staged = backend.stage_recorded_head(recorded, timeline_bank)?;
                            if staged.chunk != u64::from(chunk)
                                || staged.bank != timeline_bank
                                || staged.bytes != recorded.bytes
                            {
                                bail!("head chunk {chunk} staged receipt 漂移");
                            }
                            Ok(())
                        },
                    )?;
                }
                head_uploaded_bytes = head_uploaded_bytes
                    .checked_add(recorded.bytes)
                    .ok_or_else(|| anyhow!("head upload byte counter overflow"))?;
            }

            begin_graphics_command(&ctx, compute_commands[TERMINAL_COMPUTE])?;
            unsafe {
                terminal.record_terminal_commit_readback(
                    &ctx,
                    compute_commands[TERMINAL_COMPUTE],
                    candidate_hc,
                    state_readback,
                )?;
                ctx.device
                    .end_command_buffer(compute_commands[TERMINAL_COMPUTE])?;
                terminal.submit_terminal(
                    &ctx,
                    tail.timeline_mut(),
                    compute_commands[TERMINAL_COMPUTE],
                )?;
            }
            let completion =
                terminal.finish_candidate(&ctx, tail.timeline_mut(), base_epoch, candidate_bank)?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            if let Some(&expected_output) = EXPECTED_PREFIX_TOKENS.get(position as usize) {
                if completion.predicted_token_id != expected_output {
                    bail!(
                        "Polaris S14 position{position} 已冻结前缀 token 漂移: actual={} expected={}",
                        completion.predicted_token_id,
                        expected_output
                    );
                }
            }
            let l42_hidden_sha = format!(
                "{:x}",
                Sha256::digest(bytemuck::cast_slice::<u16, u8>(&completion.hc_streams_bf16))
            );
            // 旧 6ad6...b60 指纹属于 manifest_reference_replay，不能作为
            // online_probe_dynamic_ranges 的 hidden 合同。在线 production 继续由真实
            // top-6身份、36个Range proof/SHA、43层覆盖、最终token和原子状态提交闭合；
            // 这里保留动态轨迹指纹作可观察性，不用旧 replay 指纹误拒绝新模型路径。
            if completion.timeline.token_host_waits != 1
                || completion.timeline.layers != LAYER_COUNT
                || completion.timeline.head_chunks != u64::from(S14_HEAD_CHUNK_COUNT)
                || router_probe_host_waits != LAYER_COUNT as u64
                || router_probe_readback_bytes != LAYER_COUNT as u64 * 48
                || dynamic_physical_ranges != LAYER_COUNT as u64 * 36
            {
                bail!("production paged timeline completion contract 漂移");
            }

            let payloads = state_readback.snapshot()?;
            drop(tail);
            terminal_slot
                .take()
                .expect("candidate terminal owner")
                .destroy(&ctx);
            state_readback_slot
                .take()
                .expect("candidate state readback owner")
                .destroy(&ctx);
            let finished_host_step = backend_slot
                .take()
                .expect("candidate backend owner")
                .finish_after_external_timeline_drained()?;
            if finished_host_step != persistent_host_step {
                bail!("persistent host step telemetry 漂移");
            }
            timeline_slot
                .take()
                .expect("candidate timeline owner")
                .destroy(&ctx);
            persistent_command_resources.finish_step()?;
            command_step_active = false;

            completion_slot = Some(completion);
            payloads_slot = Some(payloads);
            l42_hidden_sha_slot = Some(l42_hidden_sha);
            token_elapsed_ms = elapsed_ms;
            Ok(())
        })();

        if let Err(error) = token_result {
            let mut failure = error;
            let mut timeline_drained = match timeline_slot.as_ref() {
                Some(timeline) => matches!(
                    timeline.stats().state,
                    S14Position0PagedLayerTimelineState::Finished
                        | S14Position0PagedLayerTimelineState::Drained
                ),
                None => true,
            };
            if !timeline_drained {
                if let Some(timeline) = timeline_slot.as_mut() {
                    match timeline.drain_all(&ctx) {
                        Ok(_) => timeline_drained = true,
                        Err(cleanup) => {
                            failure = anyhow!(
                                "{failure:#}; candidate timeline drain 同时失败: {cleanup:#}"
                            );
                        }
                    }
                }
            }

            if timeline_drained {
                if let Some(terminal) = terminal_slot.take() {
                    terminal.destroy(&ctx);
                }
                if let Some(state_readback) = state_readback_slot.take() {
                    state_readback.destroy(&ctx);
                }
                if let Some(backend) = backend_slot.take() {
                    if let Err(cleanup) = backend.abort_after_external_timeline_drained() {
                        failure = anyhow!(
                            "{failure:#}; persistent host token abort 同时失败: {cleanup:#}"
                        );
                    }
                }
            } else {
                // timeline wait 失败时让 backend owner 退化到 device-wide idle，再释放
                // terminal/readback，避免销毁仍被 command 引用的 descriptor/buffer。
                if let Some(backend) = backend_slot.take() {
                    if let Err(cleanup) = backend.abort() {
                        failure = anyhow!(
                            "{failure:#}; candidate backend device-idle abort 同时失败: {cleanup:#}"
                        );
                    }
                }
                if let Some(terminal) = terminal_slot.take() {
                    terminal.destroy(&ctx);
                }
                if let Some(state_readback) = state_readback_slot.take() {
                    state_readback.destroy(&ctx);
                }
            }
            if let Some(timeline) = timeline_slot.take() {
                timeline.destroy(&ctx);
            }
            if command_step_active {
                if let Err(cleanup) = persistent_command_resources.abort_after_drain() {
                    failure =
                        anyhow!("{failure:#}; persistent command step abort 同时失败: {cleanup:#}");
                }
            }
            drop(backend_slot);

            let rollback = rollback_guard.rollback_now();
            if let Err(cleanup) = rollback {
                failure = anyhow!("{failure:#}; candidate rollback 同时失败: {cleanup:#}");
            }
            return Err(failure);
        }

        // 成功图路径已完成唯一 final wait，并销毁所有借用 host/device candidate 的 owner。
        // 显式结束这些 Option 的生命周期后，才允许可失败的 host clone 验证与 device 发布。
        drop(backend_slot);
        drop(terminal_slot);
        drop(state_readback_slot);
        drop(timeline_slot);
        let command_step = command_step_slot
            .take()
            .ok_or_else(|| anyhow!("candidate success 缺少 persistent command telemetry"))?;
        if command_step.step_index != persistent_host_step.step_index
            || command_step.reused_from_previous_step
                != persistent_host_step.reused_from_previous_step
            || persistent_host_step.resource_allocations_this_step != 0
            || command_step.resource_allocations_this_step != 0
            || command_step.command_pool_resets_this_step != 2
            || persistent_host_step.reused_from_previous_step != (position != 0)
        {
            bail!("persistent whole-token resource telemetry 漂移");
        }
        let persistent_resource_allocations_this_step = persistent_host_step
            .resource_allocations_this_step
            .checked_add(command_step.resource_allocations_this_step)
            .ok_or_else(|| anyhow!("persistent allocation counter overflow"))?;
        let completion = completion_slot
            .take()
            .ok_or_else(|| anyhow!("candidate success 缺少 terminal completion"))?;
        let payloads = payloads_slot
            .take()
            .ok_or_else(|| anyhow!("candidate success 缺少 state payloads"))?;
        let l42_hidden_sha = l42_hidden_sha_slot
            .take()
            .ok_or_else(|| anyhow!("candidate success 缺少 L42 SHA"))?;

        let mut host_candidate = host.begin_token(base_epoch, position, input_token_id)?;
        stage_payloads(&mut host_candidate, &payloads)?;
        host_candidate.stage_position0_hc_state(&completion.hc_streams_bf16)?;
        host_candidate.complete_final(completion.predicted_token_id)?;
        let mut next_host = host.clone();
        let token_record = host_candidate.commit(&mut next_host)?;
        next_host.validate()?;
        let expected_epoch = base_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow!("continuous epoch overflow"))?;
        if next_host.commit_epoch != expected_epoch
            || next_host.position != position + 1
            || next_host.input_token_id != completion.predicted_token_id
            || usize::from(next_host.active_fixed_bank) != candidate_bank
            || token_record.predicted_token_id != completion.predicted_token_id
        {
            bail!("FullDepth43 production host candidate 提交前合同漂移");
        }
        for range in state_recording.merged_device_dirty_write_set(&host.native)? {
            device.mark_candidate_dirty(range.start, range.end - range.start)?;
        }
        device.finish_external_candidate(base_epoch, candidate_bank)?;
        let prepared = device.prepare_candidate_commit_for_position(base_epoch, position)?;
        let device_receipt = device.publish_prepared_commit(prepared);
        host = next_host;
        rollback_guard.disarm();
        if host.commit_epoch != expected_epoch
            || host.position != position + 1
            || host.input_token_id != completion.predicted_token_id
            || token_record.predicted_token_id != completion.predicted_token_id
            || device_receipt.epoch != expected_epoch
            || device_receipt.active_bank != candidate_bank
            || usize::from(host.active_fixed_bank) != device_receipt.active_bank
            || device.active_bank() != device_receipt.active_bank
        {
            bail!("FullDepth43 production host/device 原子提交回执漂移");
        }

        let producer_waits = completion.timeline.producer_transfer_waits;
        let total_host_api_waits = producer_waits
            + completion.timeline.token_host_waits
            + router_probe_host_waits
            + streamed_static_transfer_fence_waits
            + dynamic_routed_transfer_fence_waits;
        total_online_top6_routes = total_online_top6_routes
            .checked_add(router_probe_host_waits)
            .ok_or_else(|| anyhow!("online top-6 route counter overflow"))?;
        total_dynamic_physical_ranges = total_dynamic_physical_ranges
            .checked_add(dynamic_physical_ranges)
            .ok_or_else(|| anyhow!("dynamic physical Range total overflow"))?;
        output_tokens.push(completion.predicted_token_id);
        println!(
            "status=pass mode=polaris_s14_production_paged_continuous_n4 routing=online_probe_dynamic_ranges position={} input_token={} layers={} online_top6_routes={} router_probe_host_waits={} router_probe_readback_bytes={} dynamic_physical_ranges={} rust_proof_sha_verified=true cpu_compute_fallbacks=0 streamed_static_transfer_fence_waits={} dynamic_routed_transfer_fence_waits={} head_uploader_fence_waits=0 static_cold_start_waits_included_uninstrumented=true producer_bank_reuse_waits={} token_final_waits={} runtime_host_api_waits={} cleanup_device_wait_idle_calls=0 device_bank_reuse_waits={} static_bytes={} routed_bytes={} head_bytes={} persistent_step_index={} persistent_resource_allocations_this_step={} persistent_command_pool_resets_this_step={} persistent_resources_reused_from_previous_step={} candidate_scoped_backend_rebuilt_this_step=true l42_hidden_sha={} output_token={} max_logit={:.9} decoder_state_committed=true commit_epoch={} active_bank={} elapsed_ms={token_elapsed_ms:.3}",
            position,
            input_token_id,
            completion.timeline.layers,
            router_probe_host_waits,
            router_probe_host_waits,
            router_probe_readback_bytes,
            dynamic_physical_ranges,
            streamed_static_transfer_fence_waits,
            dynamic_routed_transfer_fence_waits,
            producer_waits,
            completion.timeline.token_host_waits,
            total_host_api_waits,
            completion.timeline.device_bank_reuse_waits,
            static_uploaded_bytes,
            routed_uploaded_bytes,
            head_uploaded_bytes,
            persistent_host_step.step_index,
            persistent_resource_allocations_this_step,
            command_step.command_pool_resets_this_step,
            persistent_host_step.reused_from_previous_step,
            l42_hidden_sha,
            completion.predicted_token_id,
            completion.max_logit,
            host.commit_epoch,
            host.active_fixed_bank,
        );
    }

    if output_tokens.len() != TARGET_TOKEN_COUNT as usize
        || output_tokens[..EXPECTED_PREFIX_TOKENS.len()] != EXPECTED_PREFIX_TOKENS
        || host.position != TARGET_TOKEN_COUNT
        || host.commit_epoch != u64::from(TARGET_TOKEN_COUNT)
        || device.epoch() != host.commit_epoch
        || device.active_bank() != usize::from(host.active_fixed_bank)
        || total_online_top6_routes != u64::from(TARGET_TOKEN_COUNT) * LAYER_COUNT as u64
        || total_dynamic_physical_ranges != u64::from(TARGET_TOKEN_COUNT) * LAYER_COUNT as u64 * 36
    {
        bail!("Polaris S14 production paged N=4 连续状态合同漂移");
    }
    validate_production_paged_position(TARGET_TOKEN_COUNT)
        .context("position4 paged timeline/state transaction gate")?;
    let continuous_elapsed_ms = continuous_started.elapsed().as_secs_f64() * 1000.0;
    persistent_command_resources.destroy(&ctx);
    persistent_host_resources.destroy(&ctx);
    arena.destroy(&ctx);
    device.destroy(&ctx)?;
    println!(
        "status=pass mode=polaris_s14_production_paged_continuous_n8 routing=online_probe_dynamic_ranges positions={TARGET_TOKEN_COUNT} output_tokens={output_tokens:?} online_top6_routes={total_online_top6_routes} dynamic_physical_ranges={total_dynamic_physical_ranges} rust_proof_sha_verified=true cpu_compute_fallbacks=0 final_commit_epoch={} timeline_max_position=126 ratio4_boundaries_committed=2 ratio128_boundary_pending=true position4_numeric_backend=compressed_indexer_exact elapsed_ms={continuous_elapsed_ms:.3}",
        host.commit_epoch,
    );
    Ok(())
}

/// `begin_candidate` 之后的最后一道事务保险。正常错误路径仍先显式 drain 并调用
/// `rollback_now`；若构造 backend 或 host/device commit 中途用 `?` 退出，本 guard
/// 会按 candidate 当前所有权选择普通或 external rollback，避免常驻进程遗留 phase。
struct CandidateRollbackGuard<'ctx> {
    device: *mut WholeTokenDeviceState,
    ctx: &'ctx VulkanContext,
    active: bool,
    external_armed: bool,
}

impl<'ctx> CandidateRollbackGuard<'ctx> {
    fn new(device: &mut WholeTokenDeviceState, ctx: &'ctx VulkanContext) -> Self {
        Self {
            device,
            ctx,
            active: true,
            external_armed: false,
        }
    }

    fn mark_external_armed(&mut self) {
        self.external_armed = true;
    }

    fn disarm(&mut self) {
        self.active = false;
    }

    fn rollback_now(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        // SAFETY: guard 只在 `main` 当前 token 作用域内存活；device 在 guard 之后销毁，
        // 且所有调用都发生在同一线程。外部路径由调用方先完成 timeline drain。
        let result = unsafe {
            if self.external_armed {
                (*self.device).rollback_external_candidate(self.ctx)
            } else {
                (*self.device).rollback_candidate(self.ctx)
            }
        };
        if result.is_ok() {
            self.active = false;
        }
        result
    }
}

impl Drop for CandidateRollbackGuard<'_> {
    fn drop(&mut self) {
        let _ = self.rollback_now();
    }
}

fn begin_graphics_command(ctx: &VulkanContext, command: vk::CommandBuffer) -> Result<()> {
    unsafe {
        // 每个 command 在本 token 只录制一次；不能在已有同 pool command in-flight 后
        // 重置整个 pool。
        ctx.device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
    }
    Ok(())
}
