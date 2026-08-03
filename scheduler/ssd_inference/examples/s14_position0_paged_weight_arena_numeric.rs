use anyhow::{bail, Result};
use polaris_s14_runner::Position0WholeTokenManifest;
use ssd_inference::{
    s14_position0_paged_weight_arena::S14Position0PagedWeightArena,
    s14_position0_weight_plan::S14Position0HybridWeightPlan, VulkanContext,
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let weight_plan = S14Position0HybridWeightPlan::build(&manifest)?;
    let ctx = VulkanContext::init()?;
    let started = Instant::now();
    let arena = S14Position0PagedWeightArena::new(&ctx, &weight_plan, None)?;
    let allocation_ms = started.elapsed().as_secs_f64() * 1000.0;
    if arena.resident_static_layers() + arena.streamed_static_layers() != 43 {
        bail!("paged static layer ledger drift");
    }
    println!(
        "status=pass gpu={:?} resident_static_layers={} streamed_static_layers={} essential_bytes={} resident_static_bytes={} allocated_device_bytes={} recurring_static_upload_bytes={} workspace_bytes={} vram_heap_bytes={} optional_stop={:?} allocation_ms={allocation_ms:.4} payload_uploaded=false token_emitted=false",
        ctx.gpu_name,
        arena.resident_static_layers(),
        arena.streamed_static_layers(),
        arena.allocated_essential_bytes(),
        arena.allocated_static_resident_bytes(),
        arena.allocated_device_bytes(),
        arena.recurring_static_upload_bytes(),
        arena.plan().workspace_bytes,
        ctx.vram_size(),
        arena.optional_stop(),
    );
    arena.destroy(&ctx);
    Ok(())
}
