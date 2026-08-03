//! 真实 RX 5700 XT 上的 FullDepth43 position0 L0 owner/recorder 数值门。
//! 只验证 embedding + L0；不执行 L1..L42、final head，也不产生 token。

use anyhow::{bail, Result};
use polaris_s14_runner::{DecoderStateV1, Position0WholeTokenManifest};
use ssd_inference::{
    s14_position0_hybrid_weight_arena::S14Position0HybridArenaLayout,
    s14_position0_layer_backend::S14Position0L0Backend,
    s14_position0_weight_plan::S14Position0HybridWeightPlan,
    s14_position0_whole_token::{
        Position0GpuBootstrap, Position0GpuCandidate, Position0LayerBackend,
    },
    s14_whole_token_device::WholeTokenDeviceState,
    VulkanContext,
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
    );
    let manifest = Position0WholeTokenManifest::load(&manifest_path)?;
    let weights = S14Position0HybridWeightPlan::build(&manifest)?;
    let physical = S14Position0HybridArenaLayout::build(&weights)?;
    let payload_root = PathBuf::from("D:/models/Polaris-S14/range_cache");
    let ctx = VulkanContext::init()?;
    let started = Instant::now();
    let mut backend = S14Position0L0Backend::new_gpu(
        &ctx,
        &manifest,
        &weights,
        &physical.static_layers[0],
        &payload_root,
    )?;
    let init_ms = started.elapsed().as_secs_f64() * 1000.0;
    let host = DecoderStateV1::new(4096, 0)?;
    let mut device = WholeTokenDeviceState::new(&ctx, host.native_arena.bytes(), 0)?;
    let prologue = device.begin_candidate(&ctx, 0)?;
    {
        let bootstrap = Position0GpuBootstrap {
            candidate: Position0GpuCandidate {
                ctx: &ctx,
                candidate_state: device.candidate_buffer()?,
                sticky_status: device.sticky_status_buffer()?,
                committed_host_state: &host,
                base_epoch: 0,
                candidate_bank: 1,
            },
            prologue_command: prologue,
        };
        backend.submit_embedding(&bootstrap, &manifest.embedding_row)?;
    }
    device.mark_candidate_in_flight()?;
    let compute_started = Instant::now();
    let receipt = {
        let candidate = Position0GpuCandidate {
            ctx: &ctx,
            candidate_state: device.candidate_buffer()?,
            sticky_status: device.sticky_status_buffer()?,
            committed_host_state: &host,
            base_epoch: 0,
            candidate_bank: 1,
        };
        backend.submit_layer(&candidate, &manifest.layers[0])?;
        backend.wait_l0_numeric(&candidate)?
    };
    let compute_ms = compute_started.elapsed().as_secs_f64() * 1000.0;
    if receipt.route_ids != [254, 222, 245, 200, 53, 35]
        || receipt.sticky_status != 0
        || !receipt.kv_candidate_exact
    {
        bail!("L0 numeric receipt contract failed: {receipt:?}");
    }
    let stats = backend
        .verified_payload_stats()
        .ok_or_else(|| anyhow::anyhow!("missing verified payload stats"))?;
    drop(backend);
    device.rollback_external_candidate(&ctx)?;
    device.destroy(&ctx)?;
    println!(
        "status=pass gpu={:?} scope=embedding_plus_l0_only route_ids={:?} route_weights={:?} hidden_finite={} hidden_nonzero={} kv_candidate_exact={} sticky_status={} verified_requests={} verified_sha256_bytes={} init_ms={init_ms:.4} compute_wall_ms={compute_ms:.4} token_emitted=false",
        ctx.gpu_name,
        receipt.route_ids,
        receipt.route_weights,
        receipt.finite_hidden_elements,
        receipt.nonzero_hidden_elements,
        receipt.kv_candidate_exact,
        receipt.sticky_status,
        stats.requests,
        stats.sha256_bytes,
    );
    Ok(())
}
