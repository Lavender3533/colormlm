//! FullDepth43 position0 双页 staging 与 rolling bank 上传。
//!
//! payload 必须先由 `VerifiedMappedAssetStore` 做完整 SHA-256 校验；本模块再次核对
//! tensor/path/bytes/SHA 身份后才写入对应 staging offset，并把同一精确 placement
//! 录入 transfer command。调用者用 `S14DualQueueTimeline` 负责页复用等待。

use crate::{
    s14_position0_mapped_assets::VerifiedMappedAsset,
    s14_position0_weight_arena::S14Position0WeightArena,
    s14_position0_weight_plan::{
        S14Position0LayerWeightPlan, S14Position0WeightPlan, S14_POSITION0_ROLLING_BANKS,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use std::sync::Arc;

pub struct S14Position0RollingUploader {
    staging: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    bank_bytes: u64,
}

impl S14Position0RollingUploader {
    pub fn new(ctx: &VulkanContext, plan: &S14Position0WeightPlan) -> Result<Self> {
        if plan.rolling_bank_bytes == 0 {
            bail!("position0 rolling staging bytes must be non-zero");
        }
        let staging0 = GpuBuffer::new_staging(ctx, plan.rolling_bank_bytes)?;
        let staging1 = match GpuBuffer::new_staging(ctx, plan.rolling_bank_bytes) {
            Ok(buffer) => buffer,
            Err(error) => {
                staging0.destroy(ctx);
                return Err(error);
            }
        };
        Ok(Self {
            staging: [staging0, staging1],
            bank_bytes: plan.rolling_bank_bytes,
        })
    }

    pub fn stage_verified_layer(
        &self,
        plan: &S14Position0LayerWeightPlan,
        mapped: &[Arc<VerifiedMappedAsset>],
    ) -> Result<u64> {
        if plan.bank >= S14_POSITION0_ROLLING_BANKS || mapped.len() != plan.assets.len() {
            bail!(
                "position0 staged layer count/bank drift: L{} bank={} mapped={} planned={}",
                plan.layer,
                plan.bank,
                mapped.len(),
                plan.assets.len()
            );
        }
        if plan.used_bytes > self.bank_bytes {
            bail!("position0 staged layer exceeds rolling bank capacity");
        }
        let staging = &self.staging[plan.bank];
        let mut copied = 0u64;
        for (placement, payload) in plan.assets.iter().zip(mapped) {
            let canonical_path = placement
                .path
                .canonicalize()
                .with_context(|| format!("resolve staged payload {}", placement.path.display()))?;
            let end = placement
                .offset
                .checked_add(placement.bytes)
                .ok_or_else(|| anyhow!("position0 staged asset range overflow"))?;
            if end > self.bank_bytes
                || payload.tensor() != placement.tensor
                || payload.path() != canonical_path
                || payload.expected_sha256() != placement.sha256
                || payload.bytes().len() as u64 != placement.bytes
            {
                bail!(
                    "position0 verified payload identity drift: {}",
                    placement.tensor
                );
            }
            let offset = usize::try_from(placement.offset)
                .map_err(|_| anyhow!("position0 staging offset does not fit usize"))?;
            unsafe { staging.write_at(offset, payload.bytes()) };
            copied = copied
                .checked_add(placement.bytes)
                .ok_or_else(|| anyhow!("position0 staged byte ledger overflow"))?;
        }
        if copied != plan.logical_bytes {
            bail!("position0 staged logical byte ledger drift");
        }
        Ok(copied)
    }

    /// `command` 必须来自 transfer queue family 且处于 recording 状态。
    pub unsafe fn record_layer_upload(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        arena: &S14Position0WeightArena,
        plan: &S14Position0LayerWeightPlan,
    ) -> Result<()> {
        if plan.bank >= S14_POSITION0_ROLLING_BANKS || plan.used_bytes > self.bank_bytes {
            bail!("position0 upload layer bank/capacity drift");
        }
        let mut copies = Vec::with_capacity(plan.assets.len());
        for placement in &plan.assets {
            let end = placement
                .offset
                .checked_add(placement.bytes)
                .ok_or_else(|| anyhow!("position0 upload asset range overflow"))?;
            if placement.offset % 4 != 0 || placement.bytes % 4 != 0 || end > self.bank_bytes {
                bail!("position0 upload asset is not Vulkan-copy aligned");
            }
            copies.push(
                vk::BufferCopy::default()
                    .src_offset(placement.offset)
                    .dst_offset(placement.offset)
                    .size(placement.bytes),
            );
        }
        ctx.device.cmd_copy_buffer(
            command,
            self.staging[plan.bank].handle(),
            arena.rolling(plan.bank)?.handle(),
            &copies,
        );
        Ok(())
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        for buffer in &self.staging {
            buffer.destroy(ctx);
        }
    }
}
