//! Polaris S14 原生 route 后处理的 Vulkan 硬门。
//!
//! 本模块只接收当前 router 的 256 个 logits，以及二选一的在线数据：
//! L0..L2 的 `tid2eid` 物理行，或 L3..L42 的原生 bias。capture 中记录的
//! expected expert IDs 不是这个接口的输入。所有 descriptor 都显式携带 offset，
//! 因而可以安全地绑定 whole-token arena 内的子区间。

use crate::compute::{ComputePipeline, DescriptorBinder};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Result};
use ash::vk;

pub const S14_ROUTE_GPU_EXPERTS: usize = 256;
pub const S14_ROUTE_GPU_TOP_K: usize = 6;
pub const S14_ROUTE_GPU_LOGITS_BYTES: u64 = (S14_ROUTE_GPU_EXPERTS * 4) as u64;
pub const S14_ROUTE_GPU_BIAS_BYTES: u64 = (S14_ROUTE_GPU_EXPERTS * 4) as u64;
pub const S14_ROUTE_GPU_PHYSICAL_IDS_BYTES: u64 = (S14_ROUTE_GPU_TOP_K * 4) as u64;
pub const S14_ROUTE_GPU_OUTPUT_BYTES: u64 = (S14_ROUTE_GPU_TOP_K * 4) as u64;
pub const S14_ROUTE_GPU_STATUS_BYTES: u64 = 4;

pub const S14_ROUTE_GPU_STATUS_NON_FINITE_LOGIT_OR_SCORE: u32 = 1;
pub const S14_ROUTE_GPU_STATUS_NON_FINITE_BIAS: u32 = 2;
pub const S14_ROUTE_GPU_STATUS_INVALID_PHYSICAL_ID: u32 = 4;
pub const S14_ROUTE_GPU_STATUS_INVALID_NORMALIZATION: u32 = 8;
pub const S14_ROUTE_GPU_STATUS_INVALID_MODE: u32 = 16;
pub const S14_ROUTE_GPU_STATUS_KNOWN_MASK: u32 = S14_ROUTE_GPU_STATUS_NON_FINITE_LOGIT_OR_SCORE
    | S14_ROUTE_GPU_STATUS_NON_FINITE_BIAS
    | S14_ROUTE_GPU_STATUS_INVALID_PHYSICAL_ID
    | S14_ROUTE_GPU_STATUS_INVALID_NORMALIZATION
    | S14_ROUTE_GPU_STATUS_INVALID_MODE;

pub const S14_ROUTE_POSTPROCESS_GPU_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_route_postprocess_gpu.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum S14RoutePostprocessGpuMode {
    BiasTop6 = 0,
    PhysicalIds = 1,
}

impl S14RoutePostprocessGpuMode {
    pub const fn as_raw(self) -> u32 {
        self as u32
    }

    pub const fn aux_bytes(self) -> u64 {
        match self {
            Self::BiasTop6 => S14_ROUTE_GPU_BIAS_BYTES,
            Self::PhysicalIds => S14_ROUTE_GPU_PHYSICAL_IDS_BYTES,
        }
    }
}

#[derive(Clone, Copy)]
pub struct S14RouteBufferSlice<'a> {
    pub buffer: &'a GpuBuffer,
    pub offset: u64,
}

impl<'a> S14RouteBufferSlice<'a> {
    pub const fn new(buffer: &'a GpuBuffer, offset: u64) -> Self {
        Self { buffer, offset }
    }
}

#[derive(Clone, Copy)]
pub struct S14RoutePostprocessGpuBindings<'a> {
    pub logits: S14RouteBufferSlice<'a>,
    pub aux: S14RouteBufferSlice<'a>,
    pub expert_ids: S14RouteBufferSlice<'a>,
    pub weights: S14RouteBufferSlice<'a>,
    pub selected_scores: S14RouteBufferSlice<'a>,
    pub ranking_scores: S14RouteBufferSlice<'a>,
    pub status: S14RouteBufferSlice<'a>,
}

pub struct S14RoutePostprocessGpuPipeline {
    pipeline: ComputePipeline,
}

pub struct S14RoutePostprocessGpuDispatch {
    pub binder: DescriptorBinder,
    mode: S14RoutePostprocessGpuMode,
}

impl S14RoutePostprocessGpuDispatch {
    pub const fn mode(&self) -> S14RoutePostprocessGpuMode {
        self.mode
    }
}

impl S14RoutePostprocessGpuPipeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            pipeline: ComputePipeline::new(ctx, S14_ROUTE_POSTPROCESS_GPU_SPV, 7, 4)?,
        })
    }

    pub fn bind_with_offsets(
        &self,
        ctx: &VulkanContext,
        mode: S14RoutePostprocessGpuMode,
        bindings: S14RoutePostprocessGpuBindings<'_>,
    ) -> Result<S14RoutePostprocessGpuDispatch> {
        let ranges = [
            ("logits", bindings.logits, S14_ROUTE_GPU_LOGITS_BYTES),
            ("aux", bindings.aux, mode.aux_bytes()),
            (
                "expert_ids",
                bindings.expert_ids,
                S14_ROUTE_GPU_OUTPUT_BYTES,
            ),
            ("weights", bindings.weights, S14_ROUTE_GPU_OUTPUT_BYTES),
            (
                "selected_scores",
                bindings.selected_scores,
                S14_ROUTE_GPU_OUTPUT_BYTES,
            ),
            (
                "ranking_scores",
                bindings.ranking_scores,
                S14_ROUTE_GPU_OUTPUT_BYTES,
            ),
            ("status", bindings.status, S14_ROUTE_GPU_STATUS_BYTES),
        ];
        validate_pairwise_disjoint(&ranges)?;

        let descriptors: Vec<(&GpuBuffer, u64, u64)> = ranges
            .iter()
            .map(|(_, slice, bytes)| (slice.buffer, slice.offset, *bytes))
            .collect();
        let binder = DescriptorBinder::new_with_offsets(ctx, &self.pipeline, &descriptors)?;
        Ok(S14RoutePostprocessGpuDispatch { binder, mode })
    }

    pub unsafe fn cmd(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14RoutePostprocessGpuDispatch,
    ) {
        self.cmd_raw_mode_for_validation(ctx, command, dispatch, dispatch.mode.as_raw());
    }

    /// 只用于验证 shader 对非法 ABI mode 的 fail-closed 行为。
    ///
    /// # Safety
    ///
    /// `raw_mode` 不是生产 ABI。调用者必须把该 dispatch 当作一次性诊断命令，
    /// 并在读取任何输出前检查 sticky status。
    pub unsafe fn cmd_raw_mode_for_validation(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        dispatch: &S14RoutePostprocessGpuDispatch,
        raw_mode: u32,
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
            &raw_mode.to_le_bytes(),
        );
        ctx.device.cmd_dispatch(command, 1, 1, 1);
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.pipeline.destroy(ctx);
    }
}

fn validate_pairwise_disjoint(ranges: &[(&str, S14RouteBufferSlice<'_>, u64)]) -> Result<()> {
    for (name, slice, bytes) in ranges {
        let end = slice
            .offset
            .checked_add(*bytes)
            .ok_or_else(|| anyhow::anyhow!("S14 route {name} range overflow"))?;
        if end > slice.buffer.size() {
            bail!(
                "S14 route {name} slice 越界: offset={} bytes={} capacity={}",
                slice.offset,
                bytes,
                slice.buffer.size()
            );
        }
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            let (left_name, left_slice, left_bytes) = ranges[left];
            let (right_name, right_slice, right_bytes) = ranges[right];
            if left_slice.buffer.handle() != right_slice.buffer.handle() {
                continue;
            }
            let left_end = left_slice.offset + left_bytes;
            let right_end = right_slice.offset + right_bytes;
            if left_slice.offset < right_end && right_slice.offset < left_end {
                bail!(
                    "S14 route descriptor slices overlap: {left_name}=[{}, {}) {right_name}=[{}, {})",
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

pub fn validate_route_postprocess_gpu_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let unknown = code & !S14_ROUTE_GPU_STATUS_KNOWN_MASK;
    if unknown != 0 {
        bail!("S14 route GPU returned unknown status bits 0x{unknown:08x} (full=0x{code:08x})");
    }
    let mut reasons = Vec::new();
    if code & S14_ROUTE_GPU_STATUS_NON_FINITE_LOGIT_OR_SCORE != 0 {
        reasons.push("non-finite logit/derived score");
    }
    if code & S14_ROUTE_GPU_STATUS_NON_FINITE_BIAS != 0 {
        reasons.push("non-finite bias/ranking score");
    }
    if code & S14_ROUTE_GPU_STATUS_INVALID_PHYSICAL_ID != 0 {
        reasons.push("invalid/duplicate physical ID");
    }
    if code & S14_ROUTE_GPU_STATUS_INVALID_NORMALIZATION != 0 {
        reasons.push("invalid normalization");
    }
    if code & S14_ROUTE_GPU_STATUS_INVALID_MODE != 0 {
        reasons.push("invalid mode");
    }
    bail!(
        "S14 route GPU fail-closed, status=0x{code:08x}: {}",
        reasons.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_and_byte_contract_are_frozen() {
        assert_eq!(S14RoutePostprocessGpuMode::BiasTop6.as_raw(), 0);
        assert_eq!(S14RoutePostprocessGpuMode::PhysicalIds.as_raw(), 1);
        assert_eq!(S14_ROUTE_GPU_LOGITS_BYTES, 1024);
        assert_eq!(S14RoutePostprocessGpuMode::BiasTop6.aux_bytes(), 1024);
        assert_eq!(S14RoutePostprocessGpuMode::PhysicalIds.aux_bytes(), 24);
        assert_eq!(S14_ROUTE_GPU_OUTPUT_BYTES, 24);
        assert_eq!(S14_ROUTE_GPU_STATUS_BYTES, 4);
    }

    #[test]
    fn status_contract_rejects_every_failure_bit_and_unknown_bits() {
        validate_route_postprocess_gpu_status(0).unwrap();
        for code in [1, 2, 4, 8, 16, 3, 31, 32, 63] {
            assert!(validate_route_postprocess_gpu_status(code).is_err());
        }
    }
}
