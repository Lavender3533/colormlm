//! DeepSeek-V4 / Polaris FullDepth43 的专用 final HC-head GPU 原语。
//!
//! 这条路径把 BF16 `[4,4096]` 四流状态收束为 BF16 `[4096]`，供随后
//! final RMSNorm 与 lm-head 使用。它不是普通 HC-pre：final checkpoint 只有
//! 四个 sigmoid 系数，没有 24 路 split，也没有 Sinkhorn。

use crate::compute::{ComputePipeline, DescriptorBinder};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_FINAL_HC_STREAMS: u32 = 4;
pub const S14_FINAL_HC_HIDDEN: u32 = 4096;
pub const S14_FINAL_HC_FLAT: u32 = S14_FINAL_HC_STREAMS * S14_FINAL_HC_HIDDEN;
pub const S14_FINAL_HC_AUX_VALUES: u32 = 2 * S14_FINAL_HC_STREAMS;

pub const S14_FINAL_HC_STATUS_NON_FINITE_INPUT: u32 = 1;
pub const S14_FINAL_HC_STATUS_NON_FINITE_PROJECTION: u32 = 2;
pub const S14_FINAL_HC_STATUS_NON_FINITE_REDUCED: u32 = 4;
pub const S14_FINAL_HC_STATUS_NON_FINITE_BF16: u32 = 8;

pub const S14_FINAL_HC_HEAD_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_final_hc_head.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14FinalHcHeadShape {
    pub streams: u32,
    pub hidden: u32,
}

impl S14FinalHcHeadShape {
    pub fn production() -> Self {
        Self {
            streams: S14_FINAL_HC_STREAMS,
            hidden: S14_FINAL_HC_HIDDEN,
        }
    }

    pub fn new(streams: u32, hidden: u32) -> Result<Self> {
        if streams != S14_FINAL_HC_STREAMS || hidden != S14_FINAL_HC_HIDDEN {
            bail!(
                "S14 final HC-head requires exact shape [{S14_FINAL_HC_STREAMS},{S14_FINAL_HC_HIDDEN}]"
            );
        }
        Ok(Self { streams, hidden })
    }

    pub fn hidden_bf16_bytes(self) -> u64 {
        S14_FINAL_HC_FLAT as u64 * 2
    }

    pub fn hc_head_fn_f32_bytes(self) -> u64 {
        S14_FINAL_HC_STREAMS as u64 * S14_FINAL_HC_FLAT as u64 * 4
    }

    pub fn hc_head_scale_f32_bytes(self) -> u64 {
        4
    }

    pub fn hc_head_base_f32_bytes(self) -> u64 {
        S14_FINAL_HC_STREAMS as u64 * 4
    }

    pub fn output_bf16_bytes(self) -> u64 {
        S14_FINAL_HC_HIDDEN as u64 * 2
    }

    pub fn aux_f32_bytes(self) -> u64 {
        S14_FINAL_HC_AUX_VALUES as u64 * 4
    }

    pub fn status_bytes(self) -> u64 {
        4
    }
}

pub struct S14FinalHcHeadPipeline {
    pipeline: ComputePipeline,
}

pub struct S14FinalHcHeadDispatch {
    pub binder: DescriptorBinder,
    pub shape: S14FinalHcHeadShape,
}

/// 一个 storage buffer 的显式子范围起点。实际 range 由 fixed-shape ABI 决定。
#[derive(Clone, Copy)]
pub struct S14FinalHcHeadBufferSlice<'a> {
    pub buffer: &'a GpuBuffer,
    pub offset: u64,
}

impl<'a> S14FinalHcHeadBufferSlice<'a> {
    pub const fn new(buffer: &'a GpuBuffer, offset: u64) -> Self {
        Self { buffer, offset }
    }
}

/// final HC-head 的七个 descriptor 子范围。允许共享同一底层 arena，但范围
/// 必须完全不重叠，且每个 offset 满足设备 storage-buffer alignment。
#[derive(Clone, Copy)]
pub struct S14FinalHcHeadBindings<'a> {
    pub hidden: S14FinalHcHeadBufferSlice<'a>,
    pub hc_head_fn: S14FinalHcHeadBufferSlice<'a>,
    pub hc_head_scale: S14FinalHcHeadBufferSlice<'a>,
    pub hc_head_base: S14FinalHcHeadBufferSlice<'a>,
    pub output: S14FinalHcHeadBufferSlice<'a>,
    pub aux: S14FinalHcHeadBufferSlice<'a>,
    pub status: S14FinalHcHeadBufferSlice<'a>,
}

impl S14FinalHcHeadPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_FINAL_HC_HEAD_SPV, 7, 0)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &self,
        ctx: &VulkanContext,
        shape: S14FinalHcHeadShape,
        hidden: &GpuBuffer,
        hc_head_fn: &GpuBuffer,
        hc_head_scale: &GpuBuffer,
        hc_head_base: &GpuBuffer,
        output: &GpuBuffer,
        aux: &GpuBuffer,
        status: &GpuBuffer,
    ) -> Result<S14FinalHcHeadDispatch> {
        self.bind_with_offsets(
            ctx,
            shape,
            S14FinalHcHeadBindings {
                hidden: S14FinalHcHeadBufferSlice::new(hidden, 0),
                hc_head_fn: S14FinalHcHeadBufferSlice::new(hc_head_fn, 0),
                hc_head_scale: S14FinalHcHeadBufferSlice::new(hc_head_scale, 0),
                hc_head_base: S14FinalHcHeadBufferSlice::new(hc_head_base, 0),
                output: S14FinalHcHeadBufferSlice::new(output, 0),
                aux: S14FinalHcHeadBufferSlice::new(aux, 0),
                status: S14FinalHcHeadBufferSlice::new(status, 0),
            },
        )
    }

    /// 绑定同一 arena 或独立 buffer 内的显式子范围。越界、重叠、未按设备要求
    /// 对齐都会在创建 descriptor set 前失败，不会提交不完整的 final 输出。
    pub fn bind_with_offsets(
        &self,
        ctx: &VulkanContext,
        shape: S14FinalHcHeadShape,
        bindings: S14FinalHcHeadBindings<'_>,
    ) -> Result<S14FinalHcHeadDispatch> {
        let shape = S14FinalHcHeadShape::new(shape.streams, shape.hidden)?;
        let requirements = [
            (
                bindings.hidden,
                shape.hidden_bf16_bytes(),
                "S14 final HC hidden",
            ),
            (
                bindings.hc_head_fn,
                shape.hc_head_fn_f32_bytes(),
                "S14 final HC hc_head_fn",
            ),
            (
                bindings.hc_head_scale,
                shape.hc_head_scale_f32_bytes(),
                "S14 final HC hc_head_scale",
            ),
            (
                bindings.hc_head_base,
                shape.hc_head_base_f32_bytes(),
                "S14 final HC hc_head_base",
            ),
            (
                bindings.output,
                shape.output_bf16_bytes(),
                "S14 final HC output",
            ),
            (bindings.aux, shape.aux_f32_bytes(), "S14 final HC aux"),
            (bindings.status, shape.status_bytes(), "S14 final HC status"),
        ];
        validate_final_hc_head_ranges(&requirements)?;

        let descriptors: Vec<(&GpuBuffer, u64, u64)> = requirements
            .iter()
            .map(|(slice, bytes, _)| (slice.buffer, slice.offset, *bytes))
            .collect();
        let binder = DescriptorBinder::new_with_offsets(ctx, &self.pipeline, &descriptors)?;
        Ok(S14FinalHcHeadDispatch { binder, shape })
    }

    /// # Safety
    ///
    /// `command` 必须处于 recording 状态，且 dispatch 的所有 buffer 在 fence
    /// 完成前保持有效。调用方必须在 fence 后检查 sticky status 才能使用输出。
    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14FinalHcHeadDispatch,
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

fn validate_final_hc_head_ranges(
    requirements: &[(S14FinalHcHeadBufferSlice<'_>, u64, &str)],
) -> Result<()> {
    for (slice, required, name) in requirements {
        let end = slice
            .offset
            .checked_add(*required)
            .ok_or_else(|| anyhow::anyhow!("{name} descriptor range overflow"))?;
        if end > slice.buffer.size() {
            bail!(
                "{name} slice out of bounds: offset={} bytes={} capacity={}",
                slice.offset,
                required,
                slice.buffer.size()
            );
        }
    }
    for left in 0..requirements.len() {
        for right in left + 1..requirements.len() {
            let (left_slice, left_bytes, left_name) = requirements[left];
            let (right_slice, right_bytes, right_name) = requirements[right];
            if left_slice.buffer.handle() != right_slice.buffer.handle() {
                continue;
            }
            let left_end = left_slice.offset + left_bytes;
            let right_end = right_slice.offset + right_bytes;
            if left_slice.offset < right_end && right_slice.offset < left_end {
                bail!(
                    "S14 final HC descriptor slices overlap: {left_name}=[{}, {}) {right_name}=[{}, {})",
                    left_slice.offset,
                    left_end,
                    right_slice.offset,
                    right_end
                );
            }
        }
    }
    Ok(())
}

/// fence 完成后必须调用；任何非零状态都禁止发布 final HC 输出。
pub fn validate_final_hc_head_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let known = S14_FINAL_HC_STATUS_NON_FINITE_INPUT
        | S14_FINAL_HC_STATUS_NON_FINITE_PROJECTION
        | S14_FINAL_HC_STATUS_NON_FINITE_REDUCED
        | S14_FINAL_HC_STATUS_NON_FINITE_BF16;
    if code & !known != 0 {
        bail!("S14 final HC-head returned unknown status bits 0x{code:08x}");
    }
    bail!("S14 final HC-head rejected candidate, status=0x{code:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_shape_has_exact_official_bytes() {
        let shape = S14FinalHcHeadShape::production();
        assert_eq!(shape.hidden_bf16_bytes(), 32_768);
        assert_eq!(shape.hc_head_fn_f32_bytes(), 262_144);
        assert_eq!(shape.hc_head_scale_f32_bytes(), 4);
        assert_eq!(shape.hc_head_base_f32_bytes(), 16);
        assert_eq!(shape.output_bf16_bytes(), 8_192);
        assert_eq!(shape.aux_f32_bytes(), 32);
        assert_eq!(shape.status_bytes(), 4);
    }

    #[test]
    fn shape_and_status_fail_closed() {
        for (streams, hidden) in [(0, 4096), (4, 0), (3, 4096), (4, 2048)] {
            assert!(S14FinalHcHeadShape::new(streams, hidden).is_err());
        }
        validate_final_hc_head_status(0).unwrap();
        for code in [1, 2, 4, 8, 15, 16] {
            assert!(validate_final_hc_head_status(code).is_err());
        }
    }
}
