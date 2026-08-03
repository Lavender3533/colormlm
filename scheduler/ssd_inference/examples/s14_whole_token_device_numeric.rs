//! FullDepth43 双 bank Vulkan 状态的生产尺寸原子提交门。

use anyhow::{bail, Result};
use polaris_s14_runner::DecoderStateV1;
use ssd_inference::{s14_whole_token_device::WholeTokenDeviceState, VulkanContext};
use std::time::Instant;

fn main() -> Result<()> {
    let ctx = VulkanContext::init()?;
    let host = DecoderStateV1::new(4096, 0)?;
    let initial = host.native_arena.bytes();
    let kv = &host.native.kv[0].cache;
    let patch_offset = kv.offset;
    let patch = vec![0x5au8; 512 * 2];
    let rollback_patch_offset = patch_offset + 4096;
    let rollback_patch = vec![0xa5u8; 512 * 2];
    let mut expected = initial.to_vec();
    let start = patch_offset as usize;
    expected[start..start + patch.len()].copy_from_slice(&patch);

    let started = Instant::now();
    let mut device = WholeTokenDeviceState::new(&ctx, initial, 0)?;
    if device.read_active_for_audit(&ctx)? != initial {
        bail!("初始化 active bank 与 host arena 不一致");
    }

    device.begin_candidate(&ctx, 0)?;
    device.record_candidate_patch(&ctx, rollback_patch_offset, &rollback_patch)?;
    device.mark_candidate_dirty(0, 340_992)?;
    device.submit_candidate(&ctx)?;
    device.rollback_candidate(&ctx)?;
    if device.read_active_for_audit(&ctx)? != initial {
        bail!("rollback 污染 committed active bank");
    }

    device.begin_candidate(&ctx, 0)?;
    device.record_candidate_patch(&ctx, patch_offset, &patch)?;
    device.mark_candidate_dirty(0, 340_992)?;
    device.submit_candidate(&ctx)?;
    let next_epoch = device.commit_candidate(0)?;
    let committed = device.read_active_for_audit(&ctx)?;
    if committed != expected || next_epoch != 1 || device.active_bank() != 1 {
        bail!("commit 未原子发布 candidate bank/epoch");
    }
    if device.begin_candidate(&ctx, 0).is_ok() {
        bail!("stale epoch 未被拒绝");
    }

    device.begin_candidate(&ctx, 1)?;
    device.rollback_candidate(&ctx)?;
    if device.read_active_for_audit(&ctx)? != committed {
        bail!("recording rollback 污染 committed bank");
    }

    let mut candidate_ms = Vec::with_capacity(16);
    for _ in 0..16 {
        let candidate_started = Instant::now();
        device.begin_candidate(&ctx, 1)?;
        device.submit_candidate(&ctx)?;
        device.rollback_candidate(&ctx)?;
        candidate_ms.push(candidate_started.elapsed().as_secs_f64() * 1000.0);
    }
    candidate_ms.sort_by(f64::total_cmp);
    let candidate_copy_submit_p50_ms = candidate_ms[candidate_ms.len() / 2];

    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    println!(
        "status=pass state_bytes={} banks=2 rollback_unchanged=true commit_exact=true epoch={} active_bank={} candidate_copy_submit_p50_ms={candidate_copy_submit_p50_ms:.4} audit_wall_ms={wall_ms:.4}",
        device.state_bytes(),
        device.epoch(),
        device.active_bank(),
    );
    device.destroy(&ctx)?;
    Ok(())
}
