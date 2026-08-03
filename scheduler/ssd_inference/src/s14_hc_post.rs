//! FullDepth43 单 token mHC-post Vulkan 原语。
//!
//! 输入严格采用官方边界：BF16 branch、BF16 四流 residual、F32 post/comb；
//! 输出为 BF16 四流状态。shader 在任何写回前验证整块输入和结果，非有限值会
//! 清零全部输出并写入 sticky 非零状态码。调用方须在一个 WholeTokenCandidate
//! 开始前清零状态，并在提交前统一校验；一次失败不会被后续成功 dispatch 覆盖。

use crate::compute::{
    storage_buffer_slices_overlap, ComputePipeline, DescriptorBinder, StorageBufferSlice,
};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_HC_POST_STREAMS: u32 = 4;
pub const S14_HC_POST_MAX_HIDDEN: u32 = 4096;
pub const S14_HC_POST_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_HC_POST_STATUS_NON_FINITE_MERGED: u32 = 2;
pub const S14_HC_POST_STATUS_NON_FINITE_BF16: u32 = 4;

pub const S14_HC_POST_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/s14_hc_post.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14HcPostShape {
    pub hidden: u32,
}

impl S14HcPostShape {
    pub fn new(hidden: u32) -> Result<Self> {
        if hidden == 0 || hidden > S14_HC_POST_MAX_HIDDEN || hidden % 2 != 0 {
            bail!("S14 HC-post hidden must be even and in 1..={S14_HC_POST_MAX_HIDDEN}");
        }
        let shape = Self { hidden };
        shape.branch_bf16_bytes()?;
        shape.residual_bf16_bytes()?;
        shape.output_bf16_bytes()?;
        Ok(shape)
    }

    pub fn branch_bf16_bytes(self) -> Result<u64> {
        checked_bytes(self.hidden as u64, 2, "S14 HC-post branch")
    }

    pub fn residual_bf16_bytes(self) -> Result<u64> {
        let elements = (self.hidden as u64)
            .checked_mul(S14_HC_POST_STREAMS as u64)
            .ok_or_else(|| anyhow::anyhow!("S14 HC-post residual shape overflow"))?;
        checked_bytes(elements, 2, "S14 HC-post residual")
    }

    pub fn post_f32_bytes(self) -> u64 {
        S14_HC_POST_STREAMS as u64 * 4
    }

    pub fn comb_f32_bytes(self) -> u64 {
        S14_HC_POST_STREAMS as u64 * S14_HC_POST_STREAMS as u64 * 4
    }

    pub fn output_bf16_bytes(self) -> Result<u64> {
        self.residual_bf16_bytes()
    }

    pub fn status_bytes(self) -> u64 {
        4
    }
}

pub struct S14HcPostPipeline {
    pipeline: ComputePipeline,
}

pub struct S14HcPostDispatch {
    pub binder: DescriptorBinder,
    pub shape: S14HcPostShape,
}

impl S14HcPostPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_HC_POST_SPV, 6, 4)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &self,
        ctx: &VulkanContext,
        shape: S14HcPostShape,
        branch: &GpuBuffer,
        residual: &GpuBuffer,
        post: &GpuBuffer,
        comb: &GpuBuffer,
        output: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14HcPostDispatch> {
        self.bind_slices(
            ctx,
            shape,
            StorageBufferSlice::whole(branch),
            StorageBufferSlice::whole(residual),
            StorageBufferSlice::whole(post),
            StorageBufferSlice::whole(comb),
            StorageBufferSlice::whole(output),
            StorageBufferSlice::whole(status),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_slices(
        &self,
        ctx: &VulkanContext,
        shape: S14HcPostShape,
        branch: StorageBufferSlice<'_>,
        residual: StorageBufferSlice<'_>,
        post: StorageBufferSlice<'_>,
        comb: StorageBufferSlice<'_>,
        output: StorageBufferSlice<'_>,
        status: StorageBufferSlice<'_>,
    ) -> Result<S14HcPostDispatch> {
        let shape = S14HcPostShape::new(shape.hidden)?;
        let branch_bytes = shape.branch_bf16_bytes()?;
        let residual_bytes = shape.residual_bf16_bytes()?;
        let post_bytes = shape.post_f32_bytes();
        let comb_bytes = shape.comb_f32_bytes();
        let output_bytes = shape.output_bf16_bytes()?;
        let status_bytes = shape.status_bytes();
        for (input, bytes) in [
            (branch, branch_bytes),
            (residual, residual_bytes),
            (post, post_bytes),
            (comb, comb_bytes),
            (status, status_bytes),
        ] {
            if storage_buffer_slices_overlap(output, output_bytes, input, bytes)? {
                bail!("S14 HC-post output must not alias an input or status buffer");
            }
        }
        for (input, bytes) in [
            (branch, branch_bytes),
            (residual, residual_bytes),
            (post, post_bytes),
            (comb, comb_bytes),
        ] {
            if storage_buffer_slices_overlap(status, status_bytes, input, bytes)? {
                bail!("S14 HC-post status must not alias an input buffer");
            }
        }
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.pipeline,
            &[
                (branch.buffer, branch.offset, branch_bytes),
                (residual.buffer, residual.offset, residual_bytes),
                (post.buffer, post.offset, post_bytes),
                (comb.buffer, comb.offset, comb_bytes),
                (output.buffer, output.offset, output_bytes),
                (status.buffer, status.offset, status_bytes),
            ],
        )?;
        Ok(S14HcPostDispatch { binder, shape })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14HcPostDispatch,
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
        ctx.device.cmd_push_constants(
            command,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &dispatch.shape.hidden.to_le_bytes(),
        );
        ctx.device.cmd_dispatch(command, 1, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

/// WholeTokenCandidate fence 完成后必须调用；任意非零状态都禁止发布四流输出。
pub fn validate_hc_post_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let known = S14_HC_POST_STATUS_NON_FINITE_INPUT
        | S14_HC_POST_STATUS_NON_FINITE_MERGED
        | S14_HC_POST_STATUS_NON_FINITE_BF16;
    if code & !known != 0 {
        bail!("S14 HC-post returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 HC-post rejected non-finite state, status=0x{code:08x}")
}

fn checked_bytes(elements: u64, element_bytes: u64, name: &str) -> Result<u64> {
    elements
        .checked_mul(element_bytes)
        .ok_or_else(|| anyhow::anyhow!("{name} byte size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_post_shape_accepts_production_width() {
        let shape = S14HcPostShape::new(4096).unwrap();
        assert_eq!(shape.branch_bf16_bytes().unwrap(), 8192);
        assert_eq!(shape.residual_bf16_bytes().unwrap(), 32768);
        assert_eq!(shape.output_bf16_bytes().unwrap(), 32768);
        assert_eq!(shape.post_f32_bytes(), 16);
        assert_eq!(shape.comb_f32_bytes(), 64);
    }

    #[test]
    fn hc_post_shape_and_status_fail_closed() {
        for hidden in [0, 3, 4098] {
            assert!(S14HcPostShape::new(hidden).is_err());
        }
        validate_hc_post_status(0).unwrap();
        for code in [1, 2, 4, 7, 8] {
            assert!(validate_hc_post_status(code).is_err());
        }
    }
}
