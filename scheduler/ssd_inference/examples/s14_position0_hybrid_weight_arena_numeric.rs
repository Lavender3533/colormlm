use anyhow::{bail, Result};
use polaris_s14_runner::Position0WholeTokenManifest;
use ssd_inference::{
    s14_position0_hybrid_weight_arena::{
        S14Position0HybridArenaLayout, S14Position0HybridWeightArena,
        S14_POSITION0_HYBRID_ALLOCATION_COUNT, S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES,
    },
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    VulkanContext,
};
use std::{io::Write, path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let plan = S14Position0HybridWeightPlan::build(&manifest)?;
    let layout = S14Position0HybridArenaLayout::build(&plan)?;
    let ctx = VulkanContext::init()?;
    println!(
        "status=preflight gpu={:?} static_layer_buffers={} resident_small_buffers=1 routed_banks=2 head_chunks=2 allocation_count={} static_requested_bytes={} resident_small_requested_bytes={} routed_requested_bytes={} head_requested_bytes={} largest_requested_allocation_bytes={} requested_device_bytes={} vram_bytes={} nominal_heap_remainder_bytes={} minimum_workspace_reserve_bytes={}",
        ctx.gpu_name,
        layout.static_layers.len(),
        S14_POSITION0_HYBRID_ALLOCATION_COUNT,
        layout.static_requested_bytes,
        layout.resident_small.requested_bytes,
        plan.routed_device_bytes,
        plan.head_device_bytes,
        layout.largest_requested_allocation_bytes(),
        layout.requested_device_bytes,
        ctx.vram_size(),
        ctx.vram_size().saturating_sub(layout.requested_device_bytes),
        S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES,
    );
    std::io::stdout().flush()?;
    let started = Instant::now();
    let arena = S14Position0HybridWeightArena::new(&ctx, &plan)?;
    let allocation_ms = started.elapsed().as_secs_f64() * 1000.0;

    if arena.allocation_count() != S14_POSITION0_HYBRID_ALLOCATION_COUNT
        || arena.accounted_workspace_bytes() < S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES
    {
        arena.destroy(&ctx);
        bail!("hybrid arena allocation/reserve contract failed");
    }

    println!(
        "status=pass gpu={:?} static_layer_buffers={} resident_small_buffers=1 routed_banks=2 head_chunks=2 allocation_count={} static_requested_bytes={} resident_small_requested_bytes={} routed_requested_bytes={} head_requested_bytes={} largest_requested_allocation_bytes={} requested_device_bytes={} static_actual_bytes={} resident_small_actual_bytes={} routed_actual_bytes={} head_actual_bytes={} actual_device_bytes={} vram_bytes={} accounted_workspace_bytes={} minimum_workspace_reserve_bytes={} allocation_ms={allocation_ms:.4} payload_uploaded=false token_emitted=false",
        ctx.gpu_name,
        arena.layout().static_layers.len(),
        arena.allocation_count(),
        arena.layout().static_requested_bytes,
        arena.layout().resident_small.requested_bytes,
        plan.routed_device_bytes,
        plan.head_device_bytes,
        arena.layout().largest_requested_allocation_bytes(),
        arena.requested_device_bytes(),
        arena.allocated_static_bytes(),
        arena.allocated_resident_small_bytes(),
        arena.allocated_routed_bytes(),
        arena.allocated_head_bytes(),
        arena.allocated_device_bytes(),
        arena.vram_bytes(),
        arena.accounted_workspace_bytes(),
        S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES,
    );
    arena.destroy(&ctx);
    Ok(())
}
