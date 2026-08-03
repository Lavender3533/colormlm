//! FullDepth43 position0 的真实 Vulkan 权重 arena owner。
//!
//! 两个 rolling bank 交替承载 L0..L42，resident arena 承载 BOS embedding 与
//! final head。分配失败时立即释放已成功的部分；本模块不读取 payload，也不把只分配
//! 显存解释成模型前向成功。

use crate::{
    s14_position0_weight_plan::{S14Position0WeightPlan, S14_POSITION0_ROLLING_BANKS},
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Result};
use ash::vk;

const MIN_RUNTIME_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

pub struct S14Position0WeightArena {
    rolling: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    resident: GpuBuffer,
    requested_device_bytes: u64,
    allocated_device_bytes: u64,
}

impl S14Position0WeightArena {
    pub fn new(ctx: &VulkanContext, plan: &S14Position0WeightPlan) -> Result<Self> {
        if plan.rolling_bank_bytes == 0
            || plan.resident.used_bytes == 0
            || plan.rolling_device_bytes
                != plan
                    .rolling_bank_bytes
                    .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
                    .ok_or_else(|| anyhow!("position0 rolling arena bytes overflow"))?
            || plan.device_weight_bytes
                != plan
                    .rolling_device_bytes
                    .checked_add(plan.resident.used_bytes)
                    .ok_or_else(|| anyhow!("position0 weight arena bytes overflow"))?
        {
            bail!("position0 weight plan ledger is not self-consistent");
        }
        let vram_bytes = ctx.vram_size();
        let required_with_reserve = plan
            .device_weight_bytes
            .checked_add(MIN_RUNTIME_RESERVE_BYTES)
            .ok_or_else(|| anyhow!("position0 VRAM requirement overflow"))?;
        if required_with_reserve > vram_bytes {
            bail!(
                "position0 weight arena exceeds VRAM budget: weights={} reserve={} vram={}",
                plan.device_weight_bytes,
                MIN_RUNTIME_RESERVE_BYTES,
                vram_bytes
            );
        }

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let make_vram = |bytes| {
            if ctx.qf_transfer == ctx.qf_graphics {
                GpuBuffer::new_vram(ctx, bytes, usage)
            } else {
                GpuBuffer::new_vram_shared(ctx, bytes, usage, &[ctx.qf_transfer, ctx.qf_graphics])
            }
        };

        let rolling0 = make_vram(plan.rolling_bank_bytes)?;
        let rolling1 = match make_vram(plan.rolling_bank_bytes) {
            Ok(buffer) => buffer,
            Err(error) => {
                rolling0.destroy(ctx);
                return Err(error);
            }
        };
        let resident = match make_vram(plan.resident.used_bytes) {
            Ok(buffer) => buffer,
            Err(error) => {
                rolling1.destroy(ctx);
                rolling0.destroy(ctx);
                return Err(error);
            }
        };
        let allocated_device_bytes = rolling0
            .size()
            .checked_add(rolling1.size())
            .and_then(|bytes| bytes.checked_add(resident.size()))
            .ok_or_else(|| anyhow!("position0 allocated arena ledger overflow"))?;
        Ok(Self {
            rolling: [rolling0, rolling1],
            resident,
            requested_device_bytes: plan.device_weight_bytes,
            allocated_device_bytes,
        })
    }

    pub fn rolling(&self, bank: usize) -> Result<&GpuBuffer> {
        self.rolling
            .get(bank)
            .ok_or_else(|| anyhow!("invalid position0 rolling bank {bank}"))
    }

    pub fn resident(&self) -> &GpuBuffer {
        &self.resident
    }

    pub fn requested_device_bytes(&self) -> u64 {
        self.requested_device_bytes
    }

    pub fn allocated_device_bytes(&self) -> u64 {
        self.allocated_device_bytes
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.resident.destroy(ctx);
        for buffer in &self.rolling {
            buffer.destroy(ctx);
        }
    }
}
