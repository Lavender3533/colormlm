use anyhow::Result;
use polaris_s14_runner::Position0WholeTokenManifest;
use ssd_inference::{
    s14_position0_weight_arena::S14Position0WeightArena,
    s14_position0_weight_plan::S14Position0WeightPlan, VulkanContext,
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let plan = S14Position0WeightPlan::build(&manifest)?;
    let ctx = VulkanContext::init()?;
    let started = Instant::now();
    let arena = S14Position0WeightArena::new(&ctx, &plan)?;
    let allocation_ms = started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "status=pass gpu={:?} layers={} rolling_bank_bytes={} rolling_device_bytes={} resident_bytes={} requested_device_bytes={} allocated_device_bytes={} vram_bytes={} allocation_ms={allocation_ms:.4} payload_uploaded=false token_emitted=false",
        ctx.gpu_name,
        plan.layers.len(),
        plan.rolling_bank_bytes,
        plan.rolling_device_bytes,
        plan.resident.used_bytes,
        arena.requested_device_bytes(),
        arena.allocated_device_bytes(),
        ctx.vram_size(),
    );
    arena.destroy(&ctx);
    Ok(())
}
