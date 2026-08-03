use anyhow::{bail, Result};
use ash::vk;
use polaris_s14_runner::Position0WholeTokenManifest;
use ssd_inference::{
    s14_dual_queue_timeline::{S14DualQueueTimeline, S14LayerTicket},
    s14_position0_mapped_assets::VerifiedMappedAssetStore,
    s14_position0_rolling_upload::S14Position0RollingUploader,
    s14_position0_weight_arena::S14Position0WeightArena,
    s14_position0_weight_plan::S14Position0WeightPlan,
    GpuBuffer, VulkanContext,
};
use std::{path::PathBuf, time::Instant};

const SAMPLE_BYTES: u64 = 32;
const LAYERS: usize = 2;
const SAMPLES_PER_LAYER: usize = 3;

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

    let started = Instant::now();
    let mut verified_bytes = 0u64;
    let mut expected = Vec::<u8>::with_capacity(LAYERS * SAMPLES_PER_LAYER * SAMPLE_BYTES as usize);
    for index in 0..LAYERS {
        let layer = &manifest.layers[index];
        let assets = layer.assets.iter().cloned().collect::<Vec<_>>();
        let mapped = store.map_verified_batch(&assets)?;
        verified_bytes += uploader.stage_verified_layer(&plan.layers[index], &mapped)?;
        let sample_indices = [0, mapped.len() / 2, mapped.len() - 1];
        for asset_index in sample_indices {
            if mapped[asset_index].bytes().len() < SAMPLE_BYTES as usize {
                bail!("sample asset too small: {}", mapped[asset_index].tensor());
            }
            expected.extend_from_slice(&mapped[asset_index].bytes()[..SAMPLE_BYTES as usize]);
        }
    }
    let verify_and_stage_ms = started.elapsed().as_secs_f64() * 1000.0;
    let hot_started = Instant::now();
    let mut hot_staged_bytes = 0u64;
    for index in 0..LAYERS {
        let assets = manifest.layers[index]
            .assets
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mapped = store.map_verified_batch(&assets)?;
        hot_staged_bytes += uploader.stage_verified_layer(&plan.layers[index], &mapped)?;
    }
    let hot_reuse_stage_ms = hot_started.elapsed().as_secs_f64() * 1000.0;
    if hot_staged_bytes != verified_bytes {
        bail!("position0 hot staged byte ledger drift");
    }

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
                    .command_buffer_count(LAYERS as u32),
            )
        };
        (
            transfer_pool,
            compute_pool,
            allocate(transfer_pool)?,
            allocate(compute_pool)?,
        )
    };
    let readback_bytes = (LAYERS * SAMPLES_PER_LAYER) as u64 * SAMPLE_BYTES;
    let readback = GpuBuffer::new(
        &ctx,
        readback_bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )?;
    for index in 0..LAYERS {
        let layer = &plan.layers[index];
        let sample_indices = [0, layer.assets.len() / 2, layer.assets.len() - 1];
        unsafe {
            ctx.device.begin_command_buffer(
                transfer_commands[index],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            uploader.record_layer_upload(&ctx, transfer_commands[index], &arena, layer)?;
            ctx.device.end_command_buffer(transfer_commands[index])?;

            ctx.device.begin_command_buffer(
                compute_commands[index],
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            for (sample, &asset_index) in sample_indices.iter().enumerate() {
                let placement = &layer.assets[asset_index];
                if placement.bytes < SAMPLE_BYTES {
                    bail!("sample asset too small: {}", placement.tensor);
                }
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
    }

    let transfer_started = Instant::now();
    let mut timeline = S14DualQueueTimeline::new(&ctx)?;
    let mut tickets = Vec::<S14LayerTicket>::with_capacity(LAYERS);
    for index in 0..LAYERS {
        let transfer_value =
            unsafe { timeline.submit_transfer(&ctx, transfer_commands[index], None)? };
        let compute_value =
            unsafe { timeline.submit_compute(&ctx, compute_commands[index], transfer_value)? };
        tickets.push(S14LayerTicket {
            transfer_value,
            compute_value,
        });
    }
    timeline.wait_compute(&ctx, tickets.last().unwrap().compute_value, u64::MAX)?;
    let transfer_wall_ms = transfer_started.elapsed().as_secs_f64() * 1000.0;
    let actual = unsafe {
        std::slice::from_raw_parts(readback.mapped() as *const u8, readback_bytes as usize)
    };
    if actual != expected {
        bail!("position0 rolling upload sentinel mismatch");
    }
    println!(
        "status=pass gpu={:?} layers={} verified_bytes={} mapped_entries={} map_hits={} map_misses={} sha256_bytes={} verify_and_stage_ms={verify_and_stage_ms:.4} hot_reuse_stage_ms={hot_reuse_stage_ms:.4} transfer_wall_ms={transfer_wall_ms:.4} host_waits=1 sentinel_bytes={} payload_uploaded=true token_emitted=false",
        ctx.gpu_name,
        LAYERS,
        verified_bytes,
        store.len(),
        store.stats().hits,
        store.stats().misses,
        store.stats().sha256_bytes,
        readback_bytes,
    );

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
