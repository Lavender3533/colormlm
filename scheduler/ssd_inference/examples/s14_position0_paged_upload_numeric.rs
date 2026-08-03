use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::Position0WholeTokenManifest;
use ssd_inference::{
    s14_position0_hybrid_upload::{S14Position0HybridUploadTarget, S14Position0HybridUploader},
    s14_position0_mapped_assets::VerifiedMappedAssetStore,
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    GpuBuffer, VulkanContext,
};
use std::{path::PathBuf, time::Instant};

const SAMPLE_BYTES: u64 = 32;
const MAX_READBACK_BYTES: u64 = 4 * SAMPLE_BYTES;

fn main() -> Result<()> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let plan = S14Position0HybridWeightPlan::build(&manifest)?;
    let ctx = VulkanContext::init()?;
    // staging 是完整上传链的必要资源，必须先于可选静态缓存物理保留。
    let mut uploader =
        S14Position0HybridUploader::new(&ctx, &plan).context("allocate paged uploader")?;
    let readback = GpuBuffer::new(
        &ctx,
        MAX_READBACK_BYTES,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    )
    .context("allocate paged sentinel readback")?;
    let arena = S14Position0PagedWeightArena::new(&ctx, &plan, None)
        .context("allocate paged weight arena")?;
    let mut store =
        VerifiedMappedAssetStore::new(PathBuf::from("D:/models/Polaris-S14/range_cache").as_path())
            .context("upload startup resident static groups")?;

    let started = Instant::now();
    let static_receipt = uploader.upload_static_once(&ctx, &manifest, &plan, &mut store, &arena)?;
    let startup_static_ms = started.elapsed().as_secs_f64() * 1000.0;
    let resident_layers = arena.resident_static_layers();
    let mut streamed_receipt = None;
    for _ in 0..resident_layers.min(42) + usize::from(resident_layers < 43) {
        let receipt =
            uploader.prepare_next_static_layer(&ctx, &manifest, &plan, &mut store, &arena)?;
        if !receipt.resident_hit {
            streamed_receipt = Some(receipt);
            break;
        }
    }
    if resident_layers < 43 && streamed_receipt.is_none() {
        bail!("paged arena 有流式层但上传器没有产生流式 receipt");
    }
    let routed_receipt =
        uploader.upload_next_routed_layer(&ctx, &manifest, &plan, &mut store, &arena)?;
    let head_receipt =
        uploader.upload_next_head_chunk(&ctx, &manifest, &plan, &mut store, &arena)?;

    let resident_placement = plan
        .resident
        .assets
        .iter()
        .find(|placement| {
            arena
                .static_asset_destination(placement)
                .map(|destination| destination.resident_once)
                .unwrap_or(false)
        })
        .ok_or_else(|| anyhow!("paged upload 没有 resident sample"))?;
    let streamed_placement = streamed_receipt.as_ref().and_then(|receipt| {
        let prefix = format!("layers.{}.", receipt.layer);
        plan.resident
            .assets
            .iter()
            .find(|placement| placement.tensor.starts_with(&prefix))
    });
    let routed_placement = &plan.routed_layers[0].assets[0];
    let head_asset = manifest
        .final_section
        .assets
        .iter()
        .find(|asset| asset.tensor == "head.weight")
        .ok_or_else(|| anyhow!("head.weight missing"))?;

    let mut expected = Vec::new();
    let mut copies = Vec::<(vk::Buffer, u64, u64)>::new();
    let resident_asset = manifest
        .all_assets()
        .find(|asset| asset.tensor == resident_placement.tensor)
        .ok_or_else(|| anyhow!("resident sample missing"))?;
    let resident_mapped = store.map_verified_batch(std::slice::from_ref(resident_asset))?;
    expected.extend_from_slice(&resident_mapped[0].bytes()[..SAMPLE_BYTES as usize]);
    let resident_destination = arena.static_asset_destination(resident_placement)?;
    copies.push((
        resident_destination.buffer.handle(),
        resident_destination.offset,
        0,
    ));

    if let Some(placement) = streamed_placement {
        let asset = manifest
            .all_assets()
            .find(|asset| asset.tensor == placement.tensor)
            .ok_or_else(|| anyhow!("streamed sample missing"))?;
        let mapped = store.map_verified_batch(std::slice::from_ref(asset))?;
        let dst = expected.len() as u64;
        expected.extend_from_slice(&mapped[0].bytes()[..SAMPLE_BYTES as usize]);
        let destination = arena.static_asset_destination(placement)?;
        copies.push((destination.buffer.handle(), destination.offset, dst));
    }

    let routed_asset = manifest.layers[0]
        .assets
        .routed
        .iter()
        .find(|asset| asset.tensor == routed_placement.tensor)
        .ok_or_else(|| anyhow!("routed sample missing"))?;
    let routed_mapped = store.map_verified_batch(std::slice::from_ref(routed_asset))?;
    let routed_dst = expected.len() as u64;
    expected.extend_from_slice(&routed_mapped[0].bytes()[..SAMPLE_BYTES as usize]);
    copies.push((
        arena.routed(routed_receipt.bank)?.handle(),
        routed_placement.offset,
        routed_dst,
    ));

    let head_mapped = store.map_verified_batch(std::slice::from_ref(head_asset))?;
    let head_dst = expected.len() as u64;
    expected.extend_from_slice(&head_mapped[0].bytes()[..SAMPLE_BYTES as usize]);
    copies.push((arena.head_chunk(head_receipt.bank)?.handle(), 0, head_dst));

    let (pool, command, fence) = unsafe {
        let pool = ctx
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.qf_transfer),
                None,
            )
            .context("prepare static layer")?;
        let command = ctx.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let fence = ctx
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        ctx.device
            .begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .context("upload first routed layer")?;
        for (source, source_offset, destination_offset) in &copies {
            ctx.device.cmd_copy_buffer(
                command,
                *source,
                readback.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(*source_offset)
                    .dst_offset(*destination_offset)
                    .size(SAMPLE_BYTES)],
            );
        }
        ctx.device.end_command_buffer(command)?;
        let commands = [command];
        ctx.device
            .queue_submit(
                ctx.q_transfer,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                fence,
            )
            .context("upload first head chunk")?;
        ctx.device.wait_for_fences(&[fence], true, u64::MAX)?;
        (pool, command, fence)
    };
    let observed =
        unsafe { std::slice::from_raw_parts(readback.mapped() as *const u8, expected.len()) };
    if observed != expected {
        bail!("position0 paged upload GPU sentinel mismatch");
    }
    let stats = uploader.stats();
    println!(
        "status=pass gpu={:?} resident_static_layers={} streamed_static_layers={} startup_assets_uploaded={} startup_assets_deferred={} startup_static_bytes={} startup_static_ms={startup_static_ms:.4} first_streamed_layer={:?} first_streamed_bytes={} routed_layer={} routed_bytes={} head_chunk={} head_bytes={} transfer_submits={} mapped_entries={} sha256_bytes={} sentinel_bytes={} payload_uploaded=true token_emitted=false",
        ctx.gpu_name,
        resident_layers,
        arena.streamed_static_layers(),
        static_receipt.assets_uploaded_this_call,
        static_receipt.assets_deferred_to_streaming,
        static_receipt.bytes_uploaded_this_call,
        streamed_receipt.map(|receipt| receipt.layer),
        streamed_receipt.map(|receipt| receipt.bytes).unwrap_or(0),
        routed_receipt.layer,
        routed_receipt.bytes,
        head_receipt.chunk,
        head_receipt.bytes,
        stats.transfer_submits,
        store.len(),
        store.stats().sha256_bytes,
        expected.len(),
    );

    unsafe {
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(pool, &[command]);
        ctx.device.destroy_command_pool(pool, None);
    }
    readback.destroy(&ctx);
    uploader.destroy(&ctx);
    arena.destroy(&ctx);
    Ok(())
}
