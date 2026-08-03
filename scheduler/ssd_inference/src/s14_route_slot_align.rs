//! 将在线 top-6 route 权重按当前 routed page 的专家槽位对齐。
//!
//! top-k 本身是集合；同一集合的输出顺序不改变加权 MoE 数学。但 paged
//! routed arena 的六个参数槽有固定专家身份，因此进入专家计算前必须按身份
//! 对齐权重。集合不一致时 fail-closed，禁止把权重乘到错误专家上。

use crate::compute::{ComputePipeline, DescriptorBinder};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_ROUTE_SLOT_ALIGN_BYTES: u64 = 6 * 4;
pub const S14_ROUTE_SLOT_ALIGN_STATUS_SET_MISMATCH: u32 = 32;
pub const S14_ROUTE_SLOT_ALIGN_STATUS_NON_FINITE_WEIGHT: u32 = 64;

pub const S14_ROUTE_SLOT_ALIGN_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_route_slot_align.spv"));

#[derive(Clone, Copy)]
pub struct S14RouteSlotAlignSlice<'a> {
    pub buffer: &'a GpuBuffer,
    pub offset: u64,
}

impl<'a> S14RouteSlotAlignSlice<'a> {
    pub const fn new(buffer: &'a GpuBuffer, offset: u64) -> Self {
        Self { buffer, offset }
    }
}

pub struct S14RouteSlotAlignBindings<'a> {
    pub actual_ids: S14RouteSlotAlignSlice<'a>,
    pub actual_weights: S14RouteSlotAlignSlice<'a>,
    pub expected_ids: S14RouteSlotAlignSlice<'a>,
    pub aligned_weights: S14RouteSlotAlignSlice<'a>,
    pub status: S14RouteSlotAlignSlice<'a>,
}

pub struct S14RouteSlotAlignPipeline {
    pipeline: ComputePipeline,
}

pub struct S14RouteSlotAlignDispatch {
    pub binder: DescriptorBinder,
}

impl S14RouteSlotAlignPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_ROUTE_SLOT_ALIGN_SPV, 5, 0)?,
        })
    }

    pub fn bind(
        &self,
        ctx: &VulkanContext,
        bindings: S14RouteSlotAlignBindings<'_>,
    ) -> Result<S14RouteSlotAlignDispatch> {
        let slices = [
            ("actual_ids", bindings.actual_ids),
            ("actual_weights", bindings.actual_weights),
            ("expected_ids", bindings.expected_ids),
            ("aligned_weights", bindings.aligned_weights),
        ];
        for (name, slice) in slices {
            let end = slice
                .offset
                .checked_add(S14_ROUTE_SLOT_ALIGN_BYTES)
                .ok_or_else(|| anyhow::anyhow!("route slot align {name} range overflow"))?;
            if end > slice.buffer.size() {
                bail!(
                    "route slot align {name} 越界: offset={} capacity={}",
                    slice.offset,
                    slice.buffer.size()
                );
            }
        }
        if bindings.status.offset + 4 > bindings.status.buffer.size() {
            bail!("route slot align status 越界");
        }
        let descriptors = [
            (
                bindings.actual_ids.buffer,
                bindings.actual_ids.offset,
                S14_ROUTE_SLOT_ALIGN_BYTES,
            ),
            (
                bindings.actual_weights.buffer,
                bindings.actual_weights.offset,
                S14_ROUTE_SLOT_ALIGN_BYTES,
            ),
            (
                bindings.expected_ids.buffer,
                bindings.expected_ids.offset,
                S14_ROUTE_SLOT_ALIGN_BYTES,
            ),
            (
                bindings.aligned_weights.buffer,
                bindings.aligned_weights.offset,
                S14_ROUTE_SLOT_ALIGN_BYTES,
            ),
            (bindings.status.buffer, bindings.status.offset, 4),
        ];
        Ok(S14RouteSlotAlignDispatch {
            binder: DescriptorBinder::new_with_offsets(ctx, &self.pipeline, &descriptors)?,
        })
    }

    /// # Safety
    ///
    /// 绑定的 buffer 必须在 command 完成前保持有效。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14RouteSlotAlignDispatch,
    ) {
        ctx.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.pipeline,
        );
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.layout,
            0,
            &[dispatch.binder.set],
            &[],
        );
        ctx.device.cmd_dispatch(command, 1, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

pub fn validate_route_slot_align_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    if code & S14_ROUTE_SLOT_ALIGN_STATUS_SET_MISMATCH != 0 {
        bail!("在线 top-6 集合与 routed page 专家集合不一致");
    }
    if code & S14_ROUTE_SLOT_ALIGN_STATUS_NON_FINITE_WEIGHT != 0 {
        bail!("在线 top-6 含 NaN/Inf 权重");
    }
    bail!("route slot align 未知 status: 0x{code:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_contract_is_disjoint_from_route_postprocess() {
        assert_eq!(S14_ROUTE_SLOT_ALIGN_STATUS_SET_MISMATCH, 32);
        assert_eq!(S14_ROUTE_SLOT_ALIGN_STATUS_NON_FINITE_WEIGHT, 64);
        assert!(validate_route_slot_align_status(0).is_ok());
        assert!(validate_route_slot_align_status(32).is_err());
        assert!(validate_route_slot_align_status(64).is_err());
    }
}
