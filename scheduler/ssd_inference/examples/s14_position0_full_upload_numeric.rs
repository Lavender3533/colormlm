use anyhow::{bail, Result};
use ash::vk;
use polaris_s14_runner::Position0WholeTokenManifest;
use ssd_inference::{
    s14_dual_queue_timeline::S14DualQueueTimeline,
    s14_position0_mapped_assets::VerifiedMappedAssetStore,
    s14_position0_rolling_upload::S14Position0RollingUploader,
    s14_position0_weight_arena::S14Position0WeightArena,
    s14_position0_weight_plan::{S14Position0WeightPlan, S14_POSITION0_ROLLING_BANKS},
    GpuBuffer, VulkanContext,
};
use std::{path::PathBuf, time::Instant};

const SAMPLE_BYTES: u64 = 32;
const SAMPLES_PER_LAYER: usize = 3;

#[derive(Debug, Clone, Copy)]
struct PassMetrics {
    logical_bytes: u64,
    producer_transfer_waits: u64,
    producer_transfer_wait_ms: f64,
    wall_ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_pass(
    ctx: &VulkanContext,
    manifest: &Position0WholeTokenManifest,
    plan: &S14Position0WeightPlan,
    arena: &S14Position0WeightArena,
    uploader: &S14Position0RollingUploader,
    store: &mut VerifiedMappedAssetStore,
    timeline: &mut S14DualQueueTimeline,
    transfer_commands: &[vk::CommandBuffer],
    compute_commands: &[vk::CommandBuffer],
    readback: &GpuBuffer,
    last_transfer_by_bank: &mut [u64; S14_POSITION0_ROLLING_BANKS],
    last_compute_by_bank: &mut [u64; S14_POSITION0_ROLLING_BANKS],
    expected_samples: &mut Vec<u8>,
    collect_expected: bool,
    reset_commands: bool,
) -> Result<PassMetrics> {
    let started = Instant::now();
    let mut logical_bytes = 0u64;
    let mut producer_transfer_waits = 0u64;
    let mut producer_transfer_wait_ms = 0.0f64;
    let mut final_compute = 0u64;

    for index in 0..plan.layers.len() {
        let layer = &plan.layers[index];
        let assets = manifest.layers[index]
            .assets
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mapped = store.map_verified_batch(&assets)?;
        if collect_expected {
            let sample_indices = [0, mapped.len() / 2, mapped.len() - 1];
            for asset_index in sample_indices {
                if mapped[asset_index].bytes().len() < SAMPLE_BYTES as usize {
                    bail!("sample asset too small: {}", mapped[asset_index].tensor());
                }
                expected_samples
                    .extend_from_slice(&mapped[asset_index].bytes()[..SAMPLE_BYTES as usize]);
            }
        }

        let previous_transfer = last_transfer_by_bank[layer.bank];
        if previous_transfer != 0 {
            let wait_started = Instant::now();
            timeline.wait_transfer(ctx, previous_transfer, u64::MAX)?;
            producer_transfer_wait_ms += wait_started.elapsed().as_secs_f64() * 1000.0;
            producer_transfer_waits += 1;
        }
        logical_bytes = logical_bytes
            .checked_add(uploader.stage_verified_layer(layer, &mapped)?)
            .ok_or_else(|| anyhow::anyhow!("full upload logical byte ledger overflow"))?;

        unsafe {
            if reset_commands {
                ctx.device.reset_command_buffer(
                    transfer_commands[index],
                    vk::CommandBufferResetFlags::empty(),
                )?;
                ctx.device.reset_command_buffer(
                    compute_commands[index],
                    vk::CommandBufferResetFlags::empty(),
                )?;
            }
            ctx.device.begin_command_buffer(
                transfer_commands[index],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            uploader.record_layer_upload(ctx, transfer_commands[index], arena, layer)?;
            ctx.device.end_command_buffer(transfer_commands[index])?;

            ctx.device.begin_command_buffer(
                compute_commands[index],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            for (sample, asset_index) in [0, layer.assets.len() / 2, layer.assets.len() - 1]
                .into_iter()
                .enumerate()
            {
                let placement = &layer.assets[asset_index];
                ctx.device.cmd_copy_buffer(
                    compute_commands[index],
                    arena.rolling(layer.bank)?.handle(),
                    readback.handle(),
                    &[vk::BufferCopy::default()
                        .src_offset(placement.offset)
                        .dst_offset(((index * SAMPLES_PER_LAYER + sample) as u64) * SAMPLE_BYTES)
                        .size(SAMPLE_BYTES)],
                );
            }
            ctx.device.end_command_buffer(compute_commands[index])?;
        }

        let reuse_compute =
            (last_compute_by_bank[layer.bank] != 0).then_some(last_compute_by_bank[layer.bank]);
        let transfer_value =
            unsafe { timeline.submit_transfer(ctx, transfer_commands[index], reuse_compute)? };
        let compute_value =
            unsafe { timeline.submit_compute(ctx, compute_commands[index], transfer_value)? };
        last_transfer_by_bank[layer.bank] = transfer_value;
        last_compute_by_bank[layer.bank] = compute_value;
        final_compute = compute_value;
    }
    timeline.wait_compute(ctx, final_compute, u64::MAX)?;
    Ok(PassMetrics {
        logical_bytes,
        producer_transfer_waits,
        producer_transfer_wait_ms,
        wall_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn main() -> Result<()> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let plan = S14Position0WeightPlan::build(&manifest)?;
    let ctx = VulkanContext::init()?;
    let arena = S14Position0WeightArena::new(&ctx, &plan)?;
    let uploader = S14Position0RollingUploader::new(&ctx, &plan)?;
    let mut store = VerifiedMappedAssetStore::new(
        PathBuf::from("D:/models/Polaris-S14/range_cache").as_path(),
    )?;
    let layer_count = plan.layers.len();

    let (transfer_pool, compute_pool, transfer_commands, compute_commands) = unsafe {
        let transfer_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_transfer)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let compute_pool = ctx.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(ctx.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let allocate = |pool| {
            ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(layer_count as u32),
            )
        };
        (
            transfer_pool,
            compute_pool,
            allocate(transfer_pool)?,
            allocate(compute_pool)?,
        )
    };
    let readback_bytes = (layer_count * SAMPLES_PER_LAYER) as u64 * SAMPLE_BYTES;
    let readback = GpuBuffer::new(
        &ctx,
        readback_bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;

    let mut timeline = S14DualQueueTimeline::new(&ctx)?;
    let mut last_transfer_by_bank = [0u64; S14_POSITION0_ROLLING_BANKS];
    let mut last_compute_by_bank = [0u64; S14_POSITION0_ROLLING_BANKS];
    let mut expected_samples = Vec::with_capacity(readback_bytes as usize);
    let cold = run_pass(
        &ctx,
        &manifest,
        &plan,
        &arena,
        &uploader,
        &mut store,
        &mut timeline,
        &transfer_commands,
        &compute_commands,
        &readback,
        &mut last_transfer_by_bank,
        &mut last_compute_by_bank,
        &mut expected_samples,
        true,
        false,
    )?;
    let observed = unsafe {
        std::slice::from_raw_parts(readback.mapped() as *const u8, readback_bytes as usize)
    };
    if observed != expected_samples {
        bail!("position0 cold full upload sentinel mismatch");
    }

    let hot = run_pass(
        &ctx,
        &manifest,
        &plan,
        &arena,
        &uploader,
        &mut store,
        &mut timeline,
        &transfer_commands,
        &compute_commands,
        &readback,
        &mut last_transfer_by_bank,
        &mut last_compute_by_bank,
        &mut expected_samples,
        false,
        true,
    )?;
    let observed = unsafe {
        std::slice::from_raw_parts(readback.mapped() as *const u8, readback_bytes as usize)
    };
    if observed != expected_samples {
        bail!("position0 hot full upload sentinel mismatch");
    }
    let stats = store.stats();
    let completed = timeline.completed_values(&ctx)?;
    println!(
        "status=pass gpu={:?} layers={} logical_bytes={} mapped_entries={} map_hits={} map_misses={} sha256_bytes={} cold_wall_ms={:.4} hot_wall_ms={:.4} cold_producer_waits={} hot_producer_waits={} cold_producer_wait_ms={:.4} hot_producer_wait_ms={:.4} token_thread_final_waits=2 timeline_transfer={} timeline_compute={} sentinel_bytes={} payload_uploaded=true compute_scope=sentinel_only token_emitted=false",
        ctx.gpu_name,
        layer_count,
        cold.logical_bytes,
        store.len(),
        stats.hits,
        stats.misses,
        stats.sha256_bytes,
        cold.wall_ms,
        hot.wall_ms,
        cold.producer_transfer_waits,
        hot.producer_transfer_waits,
        cold.producer_transfer_wait_ms,
        hot.producer_transfer_wait_ms,
        completed.0,
        completed.1,
        readback_bytes,
    );

    if cold.logical_bytes != hot.logical_bytes {
        bail!("position0 cold/hot logical byte ledger drift");
    }
    timeline.destroy(&ctx);
    unsafe {
        ctx.device.destroy_command_pool(compute_pool, None);
        ctx.device.destroy_command_pool(transfer_pool, None);
    }
    readback.destroy(&ctx);
    uploader.destroy(&ctx);
    arena.destroy(&ctx);
    Ok(())
}
