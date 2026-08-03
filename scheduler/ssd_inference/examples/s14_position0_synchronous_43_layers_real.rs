//! BOS embedding → FullDepth43/native-top6 的首个同步 GPU 连续层门。
//!
//! 该入口不执行 final head，也不提交 DecoderState；它只证明同一 paged workspace 中
//! L0→L42 的真实 hidden、路由与状态写回已经连续闭合。首版允许逐层同步等待。

use anyhow::{bail, Result};
use polaris_s14_runner::{DecoderStateV1, Position0WholeTokenManifest};
use sha2::{Digest, Sha256};
use ssd_inference::{
    compute::StorageBufferSlice,
    s14_position0_layer_backend::{
        S14Position0PersistentHostResources, S14Position0SynchronousVulkanLayerAdapter,
    },
    s14_position0_layer_program::S14Position0FullDepthLayerProgram,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_state_writeback::{
        stage_payloads, S14Position0FullDepthStateRecordingProgram, S14Position0StateReadback,
    },
    s14_position0_synchronous_layer_pager::S14Position0SynchronousLayerPager,
    s14_position0_synchronous_layer_plan::build_synchronous_layer_plans,
    s14_position0_terminal::S14Position0TerminalChain,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_position0_whole_token::Position0GpuCandidate,
    s14_position0_workspace::S14Position0WorkspaceSlot,
    s14_whole_token_device::WholeTokenDeviceState,
    VulkanContext,
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest_path = root.join(
        "fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let payload_root = PathBuf::from("D:/models/Polaris-S14/range_cache");
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let weights = S14Position0HybridWeightPlan::build(&manifest)?;
    let ctx = VulkanContext::init()?;
    // 同步桥还需要 static/routed/head staging；限制可选常驻静态缓存，避免贪心缓存
    // 吃掉 uploader 空间。该值只改变流式层数量，不改变任何模型数学。
    let arena = S14Position0PagedWeightArena::new(&ctx, &weights, Some(3 * 1024 * 1024 * 1024))?;
    let layer_program =
        S14Position0FullDepthLayerProgram::build(&manifest, &weights, arena.workspace_layout())?;
    let plans = build_synchronous_layer_plans(&layer_program, &arena)?;
    let mut host = DecoderStateV1::new(4096, manifest.input_token_id)?;
    let state_recording = S14Position0FullDepthStateRecordingProgram::build(
        &layer_program,
        arena.workspace_layout(),
        &host.native,
    )?;
    let mut device = WholeTokenDeviceState::new(&ctx, host.native_arena.bytes(), 0)?;
    let mut persistent_host_resources =
        S14Position0PersistentHostResources::new(&ctx, &manifest, &weights, &arena, &payload_root)?;
    let command = device.begin_candidate(&ctx, 0)?;
    // Prototype 在借出 candidate buffer 前标记 in-flight；任一后续错误仍由进程析构/显式
    // rollback 收敛，正式 Position0LayerBackend 接口会恢复“submit 成功后标记”的顺序。
    device.mark_candidate_in_flight()?;
    let candidate = Position0GpuCandidate {
        ctx: &ctx,
        candidate_state: device.candidate_buffer()?,
        sticky_status: device.sticky_status_buffer()?,
        committed_host_state: &host,
        base_epoch: 0,
        candidate_bank: 1,
    };

    let started = Instant::now();
    let mut backend = S14Position0SynchronousVulkanLayerAdapter::new(
        &ctx,
        &manifest,
        &weights,
        &arena,
        &payload_root,
        &mut persistent_host_resources,
        candidate,
    )?;
    backend.submit_embedding(command, &manifest.embedding_row)?;
    let mut pager = S14Position0SynchronousLayerPager::<u64>::new();
    for plan in &plans {
        pager.reconfigure_layer(&mut backend, plan)?;
    }
    let summary = pager.finish(&mut backend)?;
    let hidden = backend.final_hidden_bf16()?;
    if hidden.iter().all(|bits| bits & 0x7fff == 0) || backend.sticky_status() != 0 {
        bail!("FullDepth43 L42 hidden 全零或 sticky status 非零");
    }
    let hidden_sha = format!(
        "{:x}",
        Sha256::digest(bytemuck::cast_slice::<u16, u8>(&hidden))
    );
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let l42_region = arena
        .workspace_layout()
        .region(S14Position0WorkspaceSlot::HiddenStreamsB);
    let l42_hidden = StorageBufferSlice {
        buffer: arena.workspace(),
        offset: l42_region.offset,
    };
    let mut state_readback = S14Position0StateReadback::new(&ctx, &host.native)?;
    let candidate_hc = StorageBufferSlice {
        buffer: device.candidate_buffer()?,
        offset: host.native.hc.streams.offset,
    };
    let mut terminal = S14Position0TerminalChain::new(&ctx, &arena, l42_hidden)?;
    let terminal_started = Instant::now();
    let completion = terminal.execute_synchronous_reference(
        &ctx,
        0,
        1,
        candidate_hc,
        &mut state_readback,
        |expected_chunk| {
            let receipt = backend.upload_next_head_chunk()?;
            if receipt.chunk != u64::from(expected_chunk)
                || receipt.bank != expected_chunk as usize % 2
            {
                bail!(
                    "head chunk receipt 漂移: expected={expected_chunk} actual={:?}",
                    receipt
                );
            }
            Ok(())
        },
    )?;
    let terminal_ms = terminal_started.elapsed().as_secs_f64() * 1000.0;
    if completion.predicted_token_id != manifest.expected_output_token_id {
        bail!(
            "FullDepth43 output token 漂移: actual={} expected={}",
            completion.predicted_token_id,
            manifest.expected_output_token_id
        );
    }

    let payloads = state_readback.snapshot()?;
    terminal.destroy(&ctx);
    state_readback.destroy(&ctx);
    let persistent_host_step = backend.finish()?;
    if persistent_host_step.step_index != 0
        || persistent_host_step.reused_from_previous_step
        || persistent_host_step.resource_allocations_this_step != 0
    {
        bail!("synchronous persistent host telemetry 漂移");
    }

    let mut host_candidate = host.begin_token(0, 0, manifest.input_token_id)?;
    stage_payloads(&mut host_candidate, &payloads)?;
    host_candidate.stage_position0_hc_state(&completion.hc_streams_bf16)?;
    host_candidate.complete_final(completion.predicted_token_id)?;
    let mut next_host = host.clone();
    let token_record = host_candidate.commit(&mut next_host)?;
    next_host.validate()?;

    // 只发布43层真实写回与 terminal HC 覆盖的精确区间；禁止再把整个约46MiB
    // state arena 当作 dirty，从而让下一 token 的 candidate 修复保持亚 MiB 级。
    for range in state_recording.merged_device_dirty_write_set(&host.native)? {
        device.mark_candidate_dirty(range.start, range.end - range.start)?;
    }
    device.finish_external_candidate(0, 1)?;
    let prepared = device.prepare_candidate_commit(0)?;
    let device_receipt = device.publish_prepared_commit(prepared);
    host = next_host;

    if host.commit_epoch != 1
        || host.position != 1
        || host.input_token_id != completion.predicted_token_id
        || token_record.predicted_token_id != completion.predicted_token_id
        || device_receipt.epoch != 1
        || device_receipt.active_bank != 1
    {
        bail!("FullDepth43 host/device 原子提交回执漂移");
    }
    let total_ms = started.elapsed().as_secs_f64() * 1000.0;

    persistent_host_resources.destroy(&ctx);
    arena.destroy(&ctx);
    device.destroy(&ctx)?;
    println!(
        "status=pass layers={} route_execution=manifest_reference_replay compute_waits={} upload_waits={} static_bytes={} routed_bytes={} hidden_readback_bytes={} persistent_step_index={} persistent_resource_allocations_this_step={} persistent_resources_reused_from_previous_step={} candidate_scoped_backend_rebuilt_this_step=true l42_hidden_sha={} layer_elapsed_ms={elapsed_ms:.3} terminal_elapsed_ms={terminal_ms:.3} total_elapsed_ms={total_ms:.3} output_token={} max_logit={:.9} terminal_compute_waits={} decoder_state_committed=true commit_epoch={} active_bank={}",
        summary.completed_layers,
        summary.compute_wait_calls,
        summary.upload_wait_calls,
        summary.static_uploaded_bytes,
        summary.routed_uploaded_bytes,
        summary.hidden_readback_bytes,
        persistent_host_step.step_index,
        persistent_host_step.resource_allocations_this_step,
        persistent_host_step.reused_from_previous_step,
        hidden_sha,
        completion.predicted_token_id,
        completion.max_logit,
        completion.compute_host_waits,
        host.commit_epoch,
        host.active_fixed_bank,
    );
    Ok(())
}
