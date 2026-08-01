//! S14 routed expert pages on top of the existing Vulkan `Loading -> Ready`
//! publication discipline.
//!
//! This is deliberately not a native DeepSeek executor. It only adapts an
//! already-produced official route to the existing fenced VRAM page pool.

use crate::compute::{ComputePipeline, DescriptorBinder};
use crate::{ExpertKey, GpuBuffer, VramPool, VulkanContext};
use anyhow::{anyhow, bail, Result};
use ash::vk;
use polaris_s14_runner::{GraphProfile, RouteDecision, EXPERT_PAGE_BYTES};

const S14_ROUTED_EXPERT_KIND: u8 = 0x53;

/// Packed I8(E2M1x2) + F8_E8M0 shader, compiled by `build.rs`.
pub const S14_MXFP4_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_mxfp4_matvec.spv"));

/// F8_E4M3FN + F8_E8M0 weight-only shader, compiled by `build.rs`.
pub const S14_FP8_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_fp8_matvec.spv"));

pub const S14_SWIGLU_LIMIT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_swiglu_limit.spv"));

pub const S14_ROUTE_MIX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/s14_route_mix.spv"));

const FP4_GROUP_SIZE: u32 = 32;
const FP8_TILE: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14MatvecShape {
    pub n: u32,
    pub k: u32,
}

impl S14MatvecShape {
    pub fn new(n: u32, k: u32) -> Result<Self> {
        if n == 0 || k == 0 {
            bail!("S14 matvec requires non-zero N and K");
        }
        Ok(Self { n, k })
    }

    pub fn validate_mxfp4(self) -> Result<Self> {
        if self.n == 0 || self.k == 0 {
            bail!("S14 matvec requires non-zero N and K");
        }
        if self.k % FP8_TILE != 0 {
            bail!("S14 MXFP4 K={} must be a multiple of 128", self.k);
        }
        self.mxfp4_weight_bytes()?;
        self.mxfp4_scale_bytes()?;
        self.fp32_input_bytes()?;
        self.fp32_output_bytes()?;
        Ok(self)
    }

    pub fn validate_fp8(self) -> Result<Self> {
        if self.n == 0 || self.k == 0 {
            bail!("S14 matvec requires non-zero N and K");
        }
        self.fp8_weight_bytes()?;
        self.fp8_scale_bytes()?;
        self.fp32_input_bytes()?;
        self.fp32_output_bytes()?;
        Ok(self)
    }

    pub fn fp32_input_bytes(self) -> Result<u64> {
        checked_bytes(self.k as u64, 4, "S14 input")
    }

    pub fn fp32_output_bytes(self) -> Result<u64> {
        checked_bytes(self.n as u64, 4, "S14 output")
    }

    pub fn mxfp4_weight_bytes(self) -> Result<u64> {
        if self.k % 2 != 0 {
            bail!("S14 MXFP4 K={} must be even", self.k);
        }
        let elements = checked_product(self.n, self.k, "S14 MXFP4 weight")?;
        Ok(elements / 2)
    }

    pub fn mxfp4_scale_bytes(self) -> Result<u64> {
        if self.k % FP4_GROUP_SIZE != 0 {
            bail!(
                "S14 MXFP4 K={} must be a multiple of {FP4_GROUP_SIZE}",
                self.k
            );
        }
        let groups = self.k / FP4_GROUP_SIZE;
        checked_product(self.n, groups, "S14 MXFP4 scale")
    }

    pub fn fp8_weight_bytes(self) -> Result<u64> {
        checked_product(self.n, self.k, "S14 FP8 weight")
    }

    pub fn fp8_scale_bytes(self) -> Result<u64> {
        let n_tiles = self.n.div_ceil(FP8_TILE);
        let k_tiles = self.k.div_ceil(FP8_TILE);
        checked_product(n_tiles, k_tiles, "S14 FP8 scale")
    }
}

fn checked_product(a: u32, b: u32, label: &str) -> Result<u64> {
    (a as u64)
        .checked_mul(b as u64)
        .ok_or_else(|| anyhow!("{label} byte count overflow"))
}

fn checked_bytes(elements: u64, element_bytes: u64, label: &str) -> Result<u64> {
    elements
        .checked_mul(element_bytes)
        .ok_or_else(|| anyhow!("{label} byte count overflow"))
}

fn storage_bytes(logical_bytes: u64) -> u64 {
    logical_bytes.div_ceil(4) * 4
}

fn require_capacity(buffer: &GpuBuffer, required: u64, label: &str) -> Result<()> {
    if buffer.size() < required {
        bail!(
            "{label} buffer is {} B, requires at least {required} B",
            buffer.size()
        );
    }
    Ok(())
}

/// Reject the UE8M0 NaN encoding before bytes are uploaded to Vulkan.
pub fn validate_ue8m0_codes(codes: &[u8]) -> Result<()> {
    if let Some(index) = codes.iter().position(|&code| code == 0xff) {
        bail!("UE8M0 scale contains 0xff NaN at byte {index}");
    }
    Ok(())
}

/// Reject the two E4M3FN NaN encodings before bytes are uploaded to Vulkan.
pub fn validate_e4m3fn_codes(codes: &[u8]) -> Result<()> {
    if let Some(index) = codes.iter().position(|&code| code == 0x7f || code == 0xff) {
        bail!("E4M3FN weight contains NaN code at byte {index}");
    }
    Ok(())
}

pub struct S14Mxfp4Dispatch {
    pub binder: DescriptorBinder,
    pub shape: S14MatvecShape,
}

pub struct S14Fp8Dispatch {
    pub binder: DescriptorBinder,
    pub shape: S14MatvecShape,
}

pub struct S14SwigluLimitDispatch {
    pub binder: DescriptorBinder,
    pub n: u32,
}

pub struct S14RouteMixDispatch {
    pub binder: DescriptorBinder,
    pub n: u32,
    pub route_weight: f32,
}

/// Pipelines used by the native S14 graph. They are independent from the GGUF
/// Q4_K pipelines because the Polaris checkpoint has a different byte ABI.
pub struct S14NumericPipelines {
    mxfp4_matvec: ComputePipeline,
    fp8_matvec: ComputePipeline,
    swiglu_limit: ComputePipeline,
    route_mix: ComputePipeline,
}

impl S14NumericPipelines {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Ok(Self {
            mxfp4_matvec: ComputePipeline::new(ctx, S14_MXFP4_MATVEC_SPV, 4, 8)?,
            fp8_matvec: ComputePipeline::new(ctx, S14_FP8_MATVEC_SPV, 4, 8)?,
            swiglu_limit: ComputePipeline::new(ctx, S14_SWIGLU_LIMIT_SPV, 3, 4)?,
            route_mix: ComputePipeline::new(ctx, S14_ROUTE_MIX_SPV, 2, 8)?,
        })
    }

    pub fn bind_mxfp4(
        &self,
        ctx: &VulkanContext,
        shape: S14MatvecShape,
        x: &GpuBuffer,
        packed_weight: &GpuBuffer,
        weight_scale: &GpuBuffer,
        y: &GpuBuffer,
    ) -> Result<S14Mxfp4Dispatch> {
        let shape = shape.validate_mxfp4()?;
        let x_bytes = shape.fp32_input_bytes()?;
        let weight_bytes = storage_bytes(shape.mxfp4_weight_bytes()?);
        let scale_bytes = storage_bytes(shape.mxfp4_scale_bytes()?);
        let y_bytes = shape.fp32_output_bytes()?;
        require_capacity(x, x_bytes, "S14 MXFP4 input")?;
        require_capacity(packed_weight, weight_bytes, "S14 MXFP4 weight")?;
        require_capacity(weight_scale, scale_bytes, "S14 MXFP4 scale")?;
        require_capacity(y, y_bytes, "S14 MXFP4 output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.mxfp4_matvec,
            &[
                (x, x_bytes),
                (packed_weight, weight_bytes),
                (weight_scale, scale_bytes),
                (y, y_bytes),
            ],
        )?;
        Ok(S14Mxfp4Dispatch { binder, shape })
    }

    pub fn bind_fp8(
        &self,
        ctx: &VulkanContext,
        shape: S14MatvecShape,
        x: &GpuBuffer,
        weight: &GpuBuffer,
        weight_scale: &GpuBuffer,
        y: &GpuBuffer,
    ) -> Result<S14Fp8Dispatch> {
        let shape = shape.validate_fp8()?;
        let x_bytes = shape.fp32_input_bytes()?;
        let weight_bytes = storage_bytes(shape.fp8_weight_bytes()?);
        let scale_bytes = storage_bytes(shape.fp8_scale_bytes()?);
        let y_bytes = shape.fp32_output_bytes()?;
        require_capacity(x, x_bytes, "S14 FP8 input")?;
        require_capacity(weight, weight_bytes, "S14 FP8 weight")?;
        require_capacity(weight_scale, scale_bytes, "S14 FP8 scale")?;
        require_capacity(y, y_bytes, "S14 FP8 output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.fp8_matvec,
            &[
                (x, x_bytes),
                (weight, weight_bytes),
                (weight_scale, scale_bytes),
                (y, y_bytes),
            ],
        )?;
        Ok(S14Fp8Dispatch { binder, shape })
    }

    pub fn bind_swiglu_limit(
        &self,
        ctx: &VulkanContext,
        n: u32,
        gate: &GpuBuffer,
        up: &GpuBuffer,
        y: &GpuBuffer,
    ) -> Result<S14SwigluLimitDispatch> {
        if n == 0 {
            bail!("S14 SwiGLU requires non-zero length");
        }
        let bytes = checked_bytes(n as u64, 4, "S14 SwiGLU")?;
        require_capacity(gate, bytes, "S14 SwiGLU gate")?;
        require_capacity(up, bytes, "S14 SwiGLU up")?;
        require_capacity(y, bytes, "S14 SwiGLU output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.swiglu_limit,
            &[(gate, bytes), (up, bytes), (y, bytes)],
        )?;
        Ok(S14SwigluLimitDispatch { binder, n })
    }

    pub fn bind_route_mix(
        &self,
        ctx: &VulkanContext,
        n: u32,
        route_weight: f32,
        expert: &GpuBuffer,
        routed: &GpuBuffer,
    ) -> Result<S14RouteMixDispatch> {
        if n == 0 || !route_weight.is_finite() || route_weight < 0.0 {
            bail!("S14 route mix requires non-zero length and finite non-negative weight");
        }
        let bytes = checked_bytes(n as u64, 4, "S14 route mix")?;
        require_capacity(expert, bytes, "S14 route mix expert")?;
        require_capacity(routed, bytes, "S14 route mix output")?;
        let binder =
            DescriptorBinder::new(ctx, &self.route_mix, &[(expert, bytes), (routed, bytes)])?;
        Ok(S14RouteMixDispatch {
            binder,
            n,
            route_weight,
        })
    }

    /// Record only the kernel dispatch. Upload, barriers, readback and fence
    /// ownership stay with the surrounding native forward command graph.
    pub unsafe fn cmd_mxfp4_matvec(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14Mxfp4Dispatch,
    ) {
        record_matvec(
            ctx,
            command_buffer,
            &self.mxfp4_matvec,
            dispatch.binder.set,
            dispatch.shape,
        );
    }

    /// Record the FP8 weight-only kernel. The activation is ordinary F32;
    /// quantized-activation FP8 GEMM remains a separate forward capability.
    pub unsafe fn cmd_fp8_matvec(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14Fp8Dispatch,
    ) {
        record_matvec(
            ctx,
            command_buffer,
            &self.fp8_matvec,
            dispatch.binder.set,
            dispatch.shape,
        );
    }

    pub unsafe fn cmd_swiglu_limit(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14SwigluLimitDispatch,
    ) {
        record_elementwise(
            ctx,
            command_buffer,
            &self.swiglu_limit,
            dispatch.binder.set,
            dispatch.n,
            None,
        );
    }

    pub unsafe fn cmd_route_mix(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14RouteMixDispatch,
    ) {
        record_elementwise(
            ctx,
            command_buffer,
            &self.route_mix,
            dispatch.binder.set,
            dispatch.n,
            Some(dispatch.route_weight),
        );
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        self.route_mix.destroy(ctx);
        self.swiglu_limit.destroy(ctx);
        self.fp8_matvec.destroy(ctx);
        self.mxfp4_matvec.destroy(ctx);
    }
}

unsafe fn record_elementwise(
    ctx: &VulkanContext,
    command_buffer: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    n: u32,
    scalar: Option<f32>,
) {
    ctx.device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.pipeline,
    );
    ctx.device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.layout,
        0,
        &[set],
        &[],
    );
    let mut push = [0u8; 8];
    push[..4].copy_from_slice(&n.to_le_bytes());
    let push_bytes = if let Some(value) = scalar {
        push[4..].copy_from_slice(&value.to_le_bytes());
        &push[..]
    } else {
        &push[..4]
    };
    ctx.device.cmd_push_constants(
        command_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_bytes,
    );
    ctx.device
        .cmd_dispatch(command_buffer, n.div_ceil(256), 1, 1);
}

unsafe fn record_matvec(
    ctx: &VulkanContext,
    command_buffer: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    shape: S14MatvecShape,
) {
    ctx.device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.pipeline,
    );
    ctx.device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline.layout,
        0,
        &[set],
        &[],
    );
    let mut push = [0u8; 8];
    push[..4].copy_from_slice(&shape.n.to_le_bytes());
    push[4..].copy_from_slice(&shape.k.to_le_bytes());
    ctx.device.cmd_push_constants(
        command_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &push,
    );
    ctx.device.cmd_dispatch(command_buffer, shape.n, 1, 1);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PageState {
    CachedReady { slot: u32 },
    Loading { slot: u32, key: ExpertKey },
}

/// A route-bound upload ticket. It can only be created after a validated
/// `RouteDecision`; an ABI sample or guessed E0 can never create this ticket.
#[derive(Debug)]
pub struct VulkanRouteLoad {
    pub layer: u8,
    pub expert_ids: Vec<u16>,
    pages: Vec<PageState>,
}

pub struct S14VulkanExpertPages<'a> {
    pool: &'a mut VramPool,
    profile: GraphProfile,
}

impl<'a> S14VulkanExpertPages<'a> {
    pub fn new(pool: &'a mut VramPool, profile: GraphProfile) -> Result<Self> {
        if pool.slot_bytes() < EXPERT_PAGE_BYTES {
            return Err(anyhow!(
                "S14 expert slot is {} B, requires at least {} B",
                pool.slot_bytes(),
                EXPERT_PAGE_BYTES
            ));
        }
        Ok(Self { pool, profile })
    }

    /// Reserve misses as hidden Loading pages. Cached Ready pages remain
    /// visible; newly reserved pages remain invisible until `publish_after_fence`.
    pub fn begin_after_official_route(&mut self, route: &RouteDecision) -> Result<VulkanRouteLoad> {
        route
            .validate_for(self.profile)
            .map_err(|error| anyhow!(error.to_string()))?;
        let mut pages = Vec::with_capacity(route.expert_ids.len());
        for &expert in &route.expert_ids {
            let key = ExpertKey {
                layer: route.layer as u32,
                kind: S14_ROUTED_EXPERT_KIND,
                slot: expert as u32,
            };
            if let Some(slot) = self.pool.lookup(key) {
                pages.push(PageState::CachedReady { slot });
                continue;
            }
            match self.pool.reserve_loading(key) {
                Ok((slot, _evicted)) => pages.push(PageState::Loading { slot, key }),
                Err(error) => {
                    for page in &pages {
                        if let PageState::Loading { slot, key } = *page {
                            self.pool.cancel_loading(slot, key);
                        }
                    }
                    return Err(error);
                }
            }
        }
        Ok(VulkanRouteLoad {
            layer: route.layer,
            expert_ids: route.expert_ids.clone(),
            pages,
        })
    }

    /// Call only after the transfer fence covering every Loading page has
    /// completed. This is the sole publication point.
    pub fn publish_after_fence(&mut self, ticket: VulkanRouteLoad) -> Result<Vec<u32>> {
        let mut slots = Vec::with_capacity(ticket.pages.len());
        for page in ticket.pages {
            match page {
                PageState::CachedReady { slot } => slots.push(slot),
                PageState::Loading { slot, key } => {
                    self.pool.mark_ready(slot, key)?;
                    slots.push(slot);
                }
            }
        }
        Ok(slots)
    }

    pub fn cancel(&mut self, ticket: VulkanRouteLoad) {
        for page in ticket.pages {
            if let PageState::Loading { slot, key } = page {
                self.pool.cancel_loading(slot, key);
            }
        }
    }
}

#[cfg(test)]
mod numeric_tests {
    use super::*;

    #[test]
    fn real_sample_shapes_have_exact_byte_contracts() {
        let w1 = S14MatvecShape::new(2048, 4096)
            .unwrap()
            .validate_mxfp4()
            .unwrap();
        assert_eq!(w1.mxfp4_weight_bytes().unwrap(), 4_194_304);
        assert_eq!(w1.mxfp4_scale_bytes().unwrap(), 262_144);

        let w2 = S14MatvecShape::new(4096, 2048)
            .unwrap()
            .validate_mxfp4()
            .unwrap();
        assert_eq!(w2.mxfp4_weight_bytes().unwrap(), 4_194_304);
        assert_eq!(w2.mxfp4_scale_bytes().unwrap(), 262_144);

        let wq_a = S14MatvecShape::new(1024, 4096)
            .unwrap()
            .validate_fp8()
            .unwrap();
        assert_eq!(wq_a.fp8_weight_bytes().unwrap(), 4_194_304);
        assert_eq!(wq_a.fp8_scale_bytes().unwrap(), 256);
    }

    #[test]
    fn rejects_invalid_quantized_codes() {
        validate_ue8m0_codes(&[0, 127, 254]).unwrap();
        assert!(validate_ue8m0_codes(&[127, 255]).is_err());
        validate_e4m3fn_codes(&[0x00, 0x01, 0x7e, 0xfe]).unwrap();
        assert!(validate_e4m3fn_codes(&[0x7f]).is_err());
        assert!(validate_e4m3fn_codes(&[0xff]).is_err());
    }

    #[test]
    fn mxfp4_rejects_non_native_k_alignment() {
        assert!(S14MatvecShape::new(1, 32)
            .unwrap()
            .validate_mxfp4()
            .is_err());
    }
}
