//! S14 routed expert pages on top of the existing Vulkan `Loading -> Ready`
//! publication discipline.
//!
//! This is deliberately not a native DeepSeek executor. It only adapts an
//! already-produced official route to the existing fenced VRAM page pool.

use crate::compute::{ComputePipeline, DescriptorBinder};
use crate::vram_pool::{LoadingReservation, SlotBinding, SlotLease};
use crate::{ExpertKey, GpuBuffer, VramPool, VulkanContext};
use anyhow::{anyhow, bail, Result};
use ash::vk;
use polaris_s14_runner::{GraphProfile, RouteDecision, EXPERT_PAGE_BYTES};

const S14_ROUTED_EXPERT_KIND: u8 = 0x53;
const S14_SHARED_EXPERT_KIND: u8 = 0x54;

/// Three FP8 [2048,4096]/[4096,2048] weights plus their 128x128 scales.
pub const S14_SHARED_EXPERT_PAGE_BYTES: u64 = 25_167_360;

/// Packed I8(E2M1x2) + F8_E8M0 shader, compiled by `build.rs`.
pub const S14_MXFP4_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_mxfp4_matvec.spv"));

/// F8_E4M3FN + F8_E8M0 weight-only shader, compiled by `build.rs`.
pub const S14_FP8_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_fp8_matvec.spv"));

/// Audit-only OpenBLAS-compatible reductions. These shaders deliberately use
/// fewer active lanes and must never replace the default production kernels.
pub const S14_MXFP4_MATVEC_EXACT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_mxfp4_matvec_exact.spv"));
pub const S14_FP8_MATVEC_EXACT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_fp8_matvec_exact.spv"));

/// Ragged branch kernels select w1/w3/w2 slices from one shared byte arena.
/// The audit variants preserve the exact single-row reduction order.
pub const S14_RAGGED_MXFP4_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_ragged_mxfp4_matvec.spv"));
pub const S14_RAGGED_MXFP4_MATVEC_EXACT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_ragged_mxfp4_matvec_exact.spv"
));
pub const S14_RAGGED_FP8_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_ragged_fp8_matvec.spv"));
pub const S14_RAGGED_FP8_MATVEC_EXACT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_ragged_fp8_matvec_exact.spv"));

/// Eight-group wo_a projection. The source remains packed E4M3+UE8M0, but
/// each decoded weight crosses the official BF16 boundary before the dot.
pub const S14_GROUPED_FP8_BF16_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_grouped_fp8_bf16_matvec.spv"));
pub const S14_GROUPED_FP8_BF16_MATVEC_EXACT_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_grouped_fp8_bf16_matvec_exact.spv"
));

/// Generic BF16 [N,K] x F32 [K] -> F32 [N] projection. The first production
/// consumer is the native 256x4096 router; compressor/indexer projections can
/// reuse the same persistent-pipeline ABI.
pub const S14_BF16_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_bf16_matvec.spv"));

pub const S14_SWIGLU_LIMIT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_swiglu_limit.spv"));

pub const S14_ROUTE_MIX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/s14_route_mix.spv"));

pub const S14_MOE_ACCUMULATE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_moe_accumulate.spv"));

pub const S14_OFFICIAL_EXPERT_PREPARE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_official_expert_prepare.spv"));

pub const S14_BF16_ACCUMULATE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_bf16_accumulate.spv"));

pub const S14_BATCHED_OFFICIAL_EXPERT_PREPARE_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_batched_official_expert_prepare.spv"
));

pub const S14_EXACT_ORDER_BLOCK_REDUCE_SPV: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/s14_exact_order_block_reduce.spv"
));

const FP4_GROUP_SIZE: u32 = 32;
const FP8_TILE: u32 = 128;
pub const S14_ROUTED_EXPERTS_PER_POSITION: u32 = 6;
pub const S14_BLOCK_HIDDEN: u32 = 4096;
pub const S14_RAGGED_METADATA_WORDS: u64 = 6;

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

/// BF16 projection shape with a small causal block sharing the same weight
/// scan. Batch is capped at eight, matching the K=1/4/8 runtime contracts and
/// keeping shader shared memory statically bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Bf16MatvecShape {
    pub n: u32,
    pub k: u32,
    pub batch: u32,
}

impl S14Bf16MatvecShape {
    pub fn new(n: u32, k: u32, batch: u32) -> Result<Self> {
        Self { n, k, batch }.validate()
    }

    pub fn validate(self) -> Result<Self> {
        if self.n == 0 || self.k == 0 || self.batch == 0 || self.batch > 8 {
            bail!("S14 BF16 matvec requires non-zero N/K and batch in 1..=8");
        }
        let elements = checked_product(self.n, self.k, "S14 BF16 weight")?;
        let input_elements = checked_product(self.batch, self.k, "S14 BF16 input")?;
        let output_elements = checked_product(self.batch, self.n, "S14 BF16 output")?;
        if elements > u32::MAX as u64
            || input_elements > u32::MAX as u64
            || output_elements > u32::MAX as u64
        {
            bail!("S14 BF16 matvec shader index exceeds u32");
        }
        if elements % 2 != 0 {
            bail!("S14 BF16 matvec requires an even N*K for packed-u32 storage");
        }
        self.bf16_weight_bytes()?;
        self.fp32_input_bytes()?;
        self.fp32_output_bytes()?;
        Ok(self)
    }

    pub fn bf16_weight_bytes(self) -> Result<u64> {
        checked_bytes(
            checked_product(self.n, self.k, "S14 BF16 weight")?,
            2,
            "S14 BF16 weight",
        )
    }

    pub fn fp32_input_bytes(self) -> Result<u64> {
        let elements = checked_product(self.batch, self.k, "S14 BF16 input")?;
        checked_bytes(elements, 4, "S14 BF16 input")
    }

    pub fn fp32_output_bytes(self) -> Result<u64> {
        let elements = checked_product(self.batch, self.n, "S14 BF16 output")?;
        checked_bytes(elements, 4, "S14 BF16 output")
    }
}

/// Selects the pair of byte offsets used by a ragged branch dispatch.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14RaggedProjection {
    W1 = 0,
    W3 = 1,
    W2 = 2,
}

impl TryFrom<u32> for S14RaggedProjection {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::W1),
            1 => Ok(Self::W3),
            2 => Ok(Self::W2),
            _ => bail!("S14 ragged projection {value} is not w1(0), w3(1), or w2(2)"),
        }
    }
}

/// GPU metadata ABI: six little-endian u32 byte offsets per branch.
///
/// u32 keeps the shader independent from optional Vulkan int64 features.
/// The host consequently rejects arenas larger than 4 GiB. All offsets must
/// be four-byte aligned because the shader reads the arena through uint words.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct S14RaggedBranchOffsets {
    pub w1: u32,
    pub s1: u32,
    pub w3: u32,
    pub s3: u32,
    pub w2: u32,
    pub s2: u32,
}

impl S14RaggedBranchOffsets {
    pub fn words(self) -> [u32; 6] {
        [self.w1, self.s1, self.w3, self.s3, self.w2, self.s2]
    }

    pub fn projection_offsets(self, projection: S14RaggedProjection) -> (u64, u64) {
        let (weight, scale) = match projection {
            S14RaggedProjection::W1 => (self.w1, self.s1),
            S14RaggedProjection::W3 => (self.w3, self.s3),
            S14RaggedProjection::W2 => (self.w2, self.s2),
        };
        (weight as u64, scale as u64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14RaggedMatvecShape {
    pub branches: u32,
    pub branches_per_input: u32,
    pub n: u32,
    pub k: u32,
    pub projection: S14RaggedProjection,
}

impl S14RaggedMatvecShape {
    pub fn new(
        branches: u32,
        branches_per_input: u32,
        n: u32,
        k: u32,
        projection: S14RaggedProjection,
    ) -> Result<Self> {
        let shape = Self {
            branches,
            branches_per_input,
            n,
            k,
            projection,
        };
        shape.validate_common()?;
        Ok(shape)
    }

    fn validate_common(self) -> Result<Self> {
        if self.branches == 0 || self.branches_per_input == 0 || self.n == 0 || self.k == 0 {
            bail!("S14 ragged matvec requires non-zero branches, branches_per_input, N and K");
        }
        self.input_rows()?;
        self.input_elements()?;
        self.output_elements()?;
        self.metadata_bytes()?;
        Ok(self)
    }

    pub fn input_rows(self) -> Result<u32> {
        self.branches
            .checked_add(self.branches_per_input - 1)
            .ok_or_else(|| anyhow!("S14 ragged input-row count overflow"))
            .map(|value| value / self.branches_per_input)
    }

    pub fn input_elements(self) -> Result<u32> {
        self.input_rows()?
            .checked_mul(self.k)
            .ok_or_else(|| anyhow!("S14 ragged input index overflow"))
    }

    pub fn output_elements(self) -> Result<u32> {
        self.branches
            .checked_mul(self.n)
            .ok_or_else(|| anyhow!("S14 ragged output index overflow"))
    }

    pub fn fp32_input_bytes(self) -> Result<u64> {
        checked_bytes(self.input_elements()? as u64, 4, "S14 ragged input")
    }

    pub fn fp32_output_bytes(self) -> Result<u64> {
        checked_bytes(self.output_elements()? as u64, 4, "S14 ragged output")
    }

    pub fn metadata_bytes(self) -> Result<u64> {
        let words = (self.branches as u64)
            .checked_mul(S14_RAGGED_METADATA_WORDS)
            .ok_or_else(|| anyhow!("S14 ragged metadata word count overflow"))?;
        checked_bytes(words, 4, "S14 ragged metadata")
    }

    pub fn validate_mxfp4(
        self,
        arena_logical_bytes: u64,
        metadata: &[S14RaggedBranchOffsets],
    ) -> Result<Self> {
        self.validate_common()?;
        let matvec = S14MatvecShape {
            n: self.n,
            k: self.k,
        }
        .validate_mxfp4()?;
        self.validate_metadata(
            arena_logical_bytes,
            metadata,
            storage_bytes(matvec.mxfp4_weight_bytes()?),
            storage_bytes(matvec.mxfp4_scale_bytes()?),
        )?;
        Ok(self)
    }

    pub fn validate_fp8(
        self,
        arena_logical_bytes: u64,
        metadata: &[S14RaggedBranchOffsets],
    ) -> Result<Self> {
        self.validate_common()?;
        let matvec = S14MatvecShape {
            n: self.n,
            k: self.k,
        }
        .validate_fp8()?;
        self.validate_metadata(
            arena_logical_bytes,
            metadata,
            storage_bytes(matvec.fp8_weight_bytes()?),
            storage_bytes(matvec.fp8_scale_bytes()?),
        )?;
        Ok(self)
    }

    fn validate_metadata(
        self,
        arena_logical_bytes: u64,
        metadata: &[S14RaggedBranchOffsets],
        weight_bytes: u64,
        scale_bytes: u64,
    ) -> Result<()> {
        const U32_ADDRESS_SPACE_BYTES: u64 = u32::MAX as u64 + 1;
        if arena_logical_bytes == 0
            || arena_logical_bytes > U32_ADDRESS_SPACE_BYTES
            || arena_logical_bytes % 4 != 0
        {
            bail!(
                "S14 ragged arena logical capacity {arena_logical_bytes} B must be non-zero, four-byte aligned, and at most 4 GiB"
            );
        }
        if metadata.len() != self.branches as usize {
            bail!(
                "S14 ragged metadata has {} branches, expected {}",
                metadata.len(),
                self.branches
            );
        }
        for (branch, offsets) in metadata.iter().copied().enumerate() {
            for (word, offset) in offsets.words().into_iter().enumerate() {
                if offset % 4 != 0 {
                    bail!(
                        "S14 ragged branch {branch} metadata word {word} offset {offset} is not four-byte aligned"
                    );
                }
                if offset as u64 >= arena_logical_bytes {
                    bail!(
                        "S14 ragged branch {branch} metadata word {word} offset {offset} exceeds arena capacity {arena_logical_bytes} B"
                    );
                }
            }
            let (weight_offset, scale_offset) = offsets.projection_offsets(self.projection);
            require_logical_subrange(
                arena_logical_bytes,
                weight_offset,
                weight_bytes,
                &format!("S14 ragged branch {branch} weight"),
            )?;
            require_logical_subrange(
                arena_logical_bytes,
                scale_offset,
                scale_bytes,
                &format!("S14 ragged branch {branch} scale"),
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14GroupedMatvecShape {
    pub groups: u32,
    pub n_per_group: u32,
    pub k: u32,
}

impl S14GroupedMatvecShape {
    pub fn new(groups: u32, n_per_group: u32, k: u32) -> Result<Self> {
        if groups == 0 || n_per_group == 0 || k == 0 {
            bail!("S14 grouped matvec requires non-zero groups, N and K");
        }
        Ok(Self {
            groups,
            n_per_group,
            k,
        })
    }

    pub fn validate_fp8_bf16_weight(self) -> Result<Self> {
        if self.k % FP8_TILE != 0 || self.n_per_group % FP8_TILE != 0 {
            bail!(
                "S14 grouped FP8/BF16 N={} and K={} must be multiples of {FP8_TILE}",
                self.n_per_group,
                self.k
            );
        }
        self.flat_n()?;
        self.fp32_input_bytes()?;
        self.fp8_weight_bytes()?;
        self.fp8_scale_bytes()?;
        self.fp32_output_bytes()?;
        Ok(self)
    }

    pub fn flat_n(self) -> Result<u32> {
        self.groups
            .checked_mul(self.n_per_group)
            .ok_or_else(|| anyhow!("S14 grouped output element count overflow"))
    }

    pub fn fp32_input_bytes(self) -> Result<u64> {
        let elements = checked_product(self.groups, self.k, "S14 grouped input")?;
        checked_bytes(elements, 4, "S14 grouped input")
    }

    pub fn fp8_weight_bytes(self) -> Result<u64> {
        checked_product(self.flat_n()?, self.k, "S14 grouped FP8 weight")
    }

    pub fn fp8_scale_bytes(self) -> Result<u64> {
        let n_tiles = self.flat_n()?.div_ceil(FP8_TILE);
        let k_tiles = self.k.div_ceil(FP8_TILE);
        checked_product(n_tiles, k_tiles, "S14 grouped FP8 scale")
    }

    pub fn fp32_output_bytes(self) -> Result<u64> {
        checked_bytes(self.flat_n()? as u64, 4, "S14 grouped output")
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

/// Return `(matrix_bytes, route_weight_bytes)` for the batched prepare ABI.
pub fn s14_batched_official_prepare_buffer_bytes(branches: u32, n: u32) -> Result<(u64, u64)> {
    if branches == 0 || n == 0 || n % FP8_TILE != 0 {
        bail!("S14 batched official prepare requires non-zero branches and positive 128-aligned N");
    }
    let elements = branches
        .checked_mul(n)
        .ok_or_else(|| anyhow!("S14 batched official prepare index overflow"))?;
    Ok((
        checked_bytes(elements as u64, 4, "S14 batched official prepare")?,
        checked_bytes(
            branches as u64,
            4,
            "S14 batched official prepare route weights",
        )?,
    ))
}

/// Return `(routed_bytes, shared_or_output_bytes)` for the fixed top-6 block.
pub fn s14_exact_order_block_reduce_buffer_bytes(positions: u32) -> Result<(u64, u64)> {
    if positions == 0 {
        bail!("S14 exact-order block reduce requires non-zero positions");
    }
    let output_elements = positions
        .checked_mul(S14_BLOCK_HIDDEN)
        .ok_or_else(|| anyhow!("S14 exact-order block output index overflow"))?;
    let routed_elements = positions
        .checked_mul(S14_ROUTED_EXPERTS_PER_POSITION)
        .and_then(|value| value.checked_mul(S14_BLOCK_HIDDEN))
        .ok_or_else(|| anyhow!("S14 exact-order routed-down index overflow"))?;
    Ok((
        checked_bytes(routed_elements as u64, 4, "S14 exact-order routed-down")?,
        checked_bytes(output_elements as u64, 4, "S14 exact-order shared/output")?,
    ))
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

fn require_logical_subrange(
    logical_capacity: u64,
    offset: u64,
    required: u64,
    label: &str,
) -> Result<()> {
    let end = offset
        .checked_add(required)
        .ok_or_else(|| anyhow!("{label} subrange overflow"))?;
    if end > logical_capacity {
        bail!("{label} subrange [{offset}, {end}) exceeds logical capacity {logical_capacity} B");
    }
    Ok(())
}

fn require_subrange_capacity(
    buffer: &GpuBuffer,
    logical_capacity: u64,
    offset: u64,
    required: u64,
    label: &str,
) -> Result<()> {
    if logical_capacity == 0 || logical_capacity > buffer.size() {
        bail!(
            "{label} logical capacity {logical_capacity} B is invalid for {} B allocation",
            buffer.size()
        );
    }
    require_logical_subrange(logical_capacity, offset, required, label)
}

fn require_storage_offset_alignment(ctx: &VulkanContext, offset: u64, label: &str) -> Result<()> {
    let alignment = unsafe {
        ctx.instance
            .get_physical_device_properties(ctx.physical)
            .limits
            .min_storage_buffer_offset_alignment
    }
    .max(1);
    if offset % alignment != 0 {
        bail!("{label} offset {offset} is not aligned to {alignment} B");
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

pub struct S14RaggedMxfp4Dispatch {
    pub binder: DescriptorBinder,
    pub shape: S14RaggedMatvecShape,
}

pub struct S14RaggedFp8Dispatch {
    pub binder: DescriptorBinder,
    pub shape: S14RaggedMatvecShape,
}

pub struct S14GroupedFp8Bf16Dispatch {
    pub binder: DescriptorBinder,
    pub shape: S14GroupedMatvecShape,
}

pub struct S14Bf16MatvecDispatch {
    pub binder: DescriptorBinder,
    pub shape: S14Bf16MatvecShape,
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

pub struct S14MoeAccumulateDispatch {
    pub binder: DescriptorBinder,
    pub n: u32,
    pub weight: f32,
}

pub struct S14OfficialExpertPrepareDispatch {
    pub binder: DescriptorBinder,
    pub n: u32,
    pub route_weight: f32,
}

pub struct S14Bf16AccumulateDispatch {
    pub binder: DescriptorBinder,
    pub n: u32,
}

pub struct S14BatchedOfficialExpertPrepareDispatch {
    pub binder: DescriptorBinder,
    pub branches: u32,
    pub n: u32,
}

pub struct S14ExactOrderBlockReduceDispatch {
    pub binder: DescriptorBinder,
    pub positions: u32,
}

/// Pipelines used by the native S14 graph. They are independent from the GGUF
/// Q4_K pipelines because the Polaris checkpoint has a different byte ABI.
pub struct S14NumericPipelines {
    mxfp4_matvec: ComputePipeline,
    fp8_matvec: ComputePipeline,
    ragged_mxfp4_matvec: ComputePipeline,
    ragged_fp8_matvec: ComputePipeline,
    grouped_fp8_bf16_matvec: ComputePipeline,
    bf16_matvec: ComputePipeline,
    swiglu_limit: ComputePipeline,
    route_mix: ComputePipeline,
    moe_accumulate: ComputePipeline,
    official_expert_prepare: ComputePipeline,
    bf16_accumulate: ComputePipeline,
    batched_official_expert_prepare: ComputePipeline,
    exact_order_block_reduce: ComputePipeline,
}

impl S14NumericPipelines {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        Self::new_with_matvec_spv(
            ctx,
            S14_MXFP4_MATVEC_SPV,
            S14_FP8_MATVEC_SPV,
            S14_RAGGED_MXFP4_MATVEC_SPV,
            S14_RAGGED_FP8_MATVEC_SPV,
            S14_GROUPED_FP8_BF16_MATVEC_SPV,
        )
    }

    /// Construct the slow, deterministic audit pipeline used by the
    /// FullDepth43 exact-writeback worker. Production callers use `new`.
    pub fn new_exact_audit(ctx: &VulkanContext) -> Result<Self> {
        Self::new_with_matvec_spv(
            ctx,
            S14_MXFP4_MATVEC_EXACT_SPV,
            S14_FP8_MATVEC_EXACT_SPV,
            S14_RAGGED_MXFP4_MATVEC_EXACT_SPV,
            S14_RAGGED_FP8_MATVEC_EXACT_SPV,
            S14_GROUPED_FP8_BF16_MATVEC_EXACT_SPV,
        )
    }

    fn new_with_matvec_spv(
        ctx: &VulkanContext,
        mxfp4_matvec_spv: &[u8],
        fp8_matvec_spv: &[u8],
        ragged_mxfp4_matvec_spv: &[u8],
        ragged_fp8_matvec_spv: &[u8],
        grouped_fp8_bf16_matvec_spv: &[u8],
    ) -> Result<Self> {
        Ok(Self {
            mxfp4_matvec: ComputePipeline::new(ctx, mxfp4_matvec_spv, 4, 8)?,
            fp8_matvec: ComputePipeline::new(ctx, fp8_matvec_spv, 4, 8)?,
            ragged_mxfp4_matvec: ComputePipeline::new(ctx, ragged_mxfp4_matvec_spv, 4, 16)?,
            ragged_fp8_matvec: ComputePipeline::new(ctx, ragged_fp8_matvec_spv, 4, 16)?,
            grouped_fp8_bf16_matvec: ComputePipeline::new(ctx, grouped_fp8_bf16_matvec_spv, 4, 12)?,
            bf16_matvec: ComputePipeline::new(ctx, S14_BF16_MATVEC_SPV, 3, 16)?,
            swiglu_limit: ComputePipeline::new(ctx, S14_SWIGLU_LIMIT_SPV, 3, 4)?,
            route_mix: ComputePipeline::new(ctx, S14_ROUTE_MIX_SPV, 2, 8)?,
            moe_accumulate: ComputePipeline::new(ctx, S14_MOE_ACCUMULATE_SPV, 2, 8)?,
            official_expert_prepare: ComputePipeline::new(
                ctx,
                S14_OFFICIAL_EXPERT_PREPARE_SPV,
                3,
                8,
            )?,
            bf16_accumulate: ComputePipeline::new(ctx, S14_BF16_ACCUMULATE_SPV, 2, 4)?,
            batched_official_expert_prepare: ComputePipeline::new(
                ctx,
                S14_BATCHED_OFFICIAL_EXPERT_PREPARE_SPV,
                4,
                8,
            )?,
            exact_order_block_reduce: ComputePipeline::new(
                ctx,
                S14_EXACT_ORDER_BLOCK_REDUCE_SPV,
                3,
                4,
            )?,
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

    /// Bind MXFP4 weight and scale views inside one larger storage arena.
    /// `arena_logical_bytes` is the caller-owned logical bound; it is checked
    /// separately from `GpuBuffer::size()` because Vulkan memory requirements
    /// may round the physical allocation up beyond the requested arena size.
    pub fn bind_mxfp4_weight_arena(
        &self,
        ctx: &VulkanContext,
        shape: S14MatvecShape,
        x: &GpuBuffer,
        arena: &GpuBuffer,
        arena_logical_bytes: u64,
        packed_weight_offset: u64,
        weight_scale_offset: u64,
        y: &GpuBuffer,
    ) -> Result<S14Mxfp4Dispatch> {
        let shape = shape.validate_mxfp4()?;
        let x_bytes = shape.fp32_input_bytes()?;
        let weight_bytes = storage_bytes(shape.mxfp4_weight_bytes()?);
        let scale_bytes = storage_bytes(shape.mxfp4_scale_bytes()?);
        let y_bytes = shape.fp32_output_bytes()?;
        require_capacity(x, x_bytes, "S14 MXFP4 input")?;
        require_subrange_capacity(
            arena,
            arena_logical_bytes,
            packed_weight_offset,
            weight_bytes,
            "S14 MXFP4 arena weight",
        )?;
        require_subrange_capacity(
            arena,
            arena_logical_bytes,
            weight_scale_offset,
            scale_bytes,
            "S14 MXFP4 arena scale",
        )?;
        require_storage_offset_alignment(ctx, packed_weight_offset, "S14 MXFP4 arena weight")?;
        require_storage_offset_alignment(ctx, weight_scale_offset, "S14 MXFP4 arena scale")?;
        require_capacity(y, y_bytes, "S14 MXFP4 output")?;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.mxfp4_matvec,
            &[
                (x, 0, x_bytes),
                (arena, packed_weight_offset, weight_bytes),
                (arena, weight_scale_offset, scale_bytes),
                (y, 0, y_bytes),
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

    /// Bind FP8 weight and scale views inside one larger storage arena.
    pub fn bind_fp8_weight_arena(
        &self,
        ctx: &VulkanContext,
        shape: S14MatvecShape,
        x: &GpuBuffer,
        arena: &GpuBuffer,
        arena_logical_bytes: u64,
        weight_offset: u64,
        weight_scale_offset: u64,
        y: &GpuBuffer,
    ) -> Result<S14Fp8Dispatch> {
        let shape = shape.validate_fp8()?;
        let x_bytes = shape.fp32_input_bytes()?;
        let weight_bytes = storage_bytes(shape.fp8_weight_bytes()?);
        let scale_bytes = storage_bytes(shape.fp8_scale_bytes()?);
        let y_bytes = shape.fp32_output_bytes()?;
        require_capacity(x, x_bytes, "S14 FP8 input")?;
        require_subrange_capacity(
            arena,
            arena_logical_bytes,
            weight_offset,
            weight_bytes,
            "S14 FP8 arena weight",
        )?;
        require_subrange_capacity(
            arena,
            arena_logical_bytes,
            weight_scale_offset,
            scale_bytes,
            "S14 FP8 arena scale",
        )?;
        require_storage_offset_alignment(ctx, weight_offset, "S14 FP8 arena weight")?;
        require_storage_offset_alignment(ctx, weight_scale_offset, "S14 FP8 arena scale")?;
        require_capacity(y, y_bytes, "S14 FP8 output")?;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.fp8_matvec,
            &[
                (x, 0, x_bytes),
                (arena, weight_offset, weight_bytes),
                (arena, weight_scale_offset, scale_bytes),
                (y, 0, y_bytes),
            ],
        )?;
        Ok(S14Fp8Dispatch { binder, shape })
    }

    /// Bind a multi-branch MXFP4 dispatch over one shared arena. `metadata`
    /// is the CPU mirror of `metadata_buffer`; validating it here prevents a
    /// malformed GPU offset from escaping the logical arena. The caller must
    /// upload these exact six-word records before recording the dispatch.
    pub fn bind_ragged_mxfp4_weight_arena(
        &self,
        ctx: &VulkanContext,
        shape: S14RaggedMatvecShape,
        x: &GpuBuffer,
        arena: &GpuBuffer,
        arena_logical_bytes: u64,
        metadata_buffer: &GpuBuffer,
        metadata: &[S14RaggedBranchOffsets],
        y: &GpuBuffer,
    ) -> Result<S14RaggedMxfp4Dispatch> {
        let shape = shape.validate_mxfp4(arena_logical_bytes, metadata)?;
        let x_bytes = shape.fp32_input_bytes()?;
        let metadata_bytes = shape.metadata_bytes()?;
        let y_bytes = shape.fp32_output_bytes()?;
        require_capacity(x, x_bytes, "S14 ragged MXFP4 input")?;
        require_capacity(arena, arena_logical_bytes, "S14 ragged MXFP4 arena")?;
        require_capacity(metadata_buffer, metadata_bytes, "S14 ragged MXFP4 metadata")?;
        require_capacity(y, y_bytes, "S14 ragged MXFP4 output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.ragged_mxfp4_matvec,
            &[
                (x, x_bytes),
                (arena, arena_logical_bytes),
                (metadata_buffer, metadata_bytes),
                (y, y_bytes),
            ],
        )?;
        Ok(S14RaggedMxfp4Dispatch { binder, shape })
    }

    /// FP8 counterpart of `bind_ragged_mxfp4_weight_arena`.
    pub fn bind_ragged_fp8_weight_arena(
        &self,
        ctx: &VulkanContext,
        shape: S14RaggedMatvecShape,
        x: &GpuBuffer,
        arena: &GpuBuffer,
        arena_logical_bytes: u64,
        metadata_buffer: &GpuBuffer,
        metadata: &[S14RaggedBranchOffsets],
        y: &GpuBuffer,
    ) -> Result<S14RaggedFp8Dispatch> {
        let shape = shape.validate_fp8(arena_logical_bytes, metadata)?;
        let x_bytes = shape.fp32_input_bytes()?;
        let metadata_bytes = shape.metadata_bytes()?;
        let y_bytes = shape.fp32_output_bytes()?;
        require_capacity(x, x_bytes, "S14 ragged FP8 input")?;
        require_capacity(arena, arena_logical_bytes, "S14 ragged FP8 arena")?;
        require_capacity(metadata_buffer, metadata_bytes, "S14 ragged FP8 metadata")?;
        require_capacity(y, y_bytes, "S14 ragged FP8 output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.ragged_fp8_matvec,
            &[
                (x, x_bytes),
                (arena, arena_logical_bytes),
                (metadata_buffer, metadata_bytes),
                (y, y_bytes),
            ],
        )?;
        Ok(S14RaggedFp8Dispatch { binder, shape })
    }

    pub fn bind_grouped_fp8_bf16_weight(
        &self,
        ctx: &VulkanContext,
        shape: S14GroupedMatvecShape,
        x: &GpuBuffer,
        weight: &GpuBuffer,
        weight_scale: &GpuBuffer,
        y: &GpuBuffer,
    ) -> Result<S14GroupedFp8Bf16Dispatch> {
        let shape = shape.validate_fp8_bf16_weight()?;
        let x_bytes = shape.fp32_input_bytes()?;
        let weight_bytes = storage_bytes(shape.fp8_weight_bytes()?);
        let scale_bytes = storage_bytes(shape.fp8_scale_bytes()?);
        let y_bytes = shape.fp32_output_bytes()?;
        require_capacity(x, x_bytes, "S14 grouped FP8/BF16 input")?;
        require_capacity(weight, weight_bytes, "S14 grouped FP8/BF16 weight")?;
        require_capacity(weight_scale, scale_bytes, "S14 grouped FP8/BF16 scale")?;
        require_capacity(y, y_bytes, "S14 grouped FP8/BF16 output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.grouped_fp8_bf16_matvec,
            &[
                (x, x_bytes),
                (weight, weight_bytes),
                (weight_scale, scale_bytes),
                (y, y_bytes),
            ],
        )?;
        Ok(S14GroupedFp8Bf16Dispatch { binder, shape })
    }

    pub fn bind_bf16_matvec(
        &self,
        ctx: &VulkanContext,
        shape: S14Bf16MatvecShape,
        weight: &GpuBuffer,
        x: &GpuBuffer,
        y: &GpuBuffer,
    ) -> Result<S14Bf16MatvecDispatch> {
        let shape = shape.validate()?;
        let weight_bytes = storage_bytes(shape.bf16_weight_bytes()?);
        let x_bytes = shape.fp32_input_bytes()?;
        let y_bytes = shape.fp32_output_bytes()?;
        require_capacity(weight, weight_bytes, "S14 BF16 matvec weight")?;
        require_capacity(x, x_bytes, "S14 BF16 matvec input")?;
        require_capacity(y, y_bytes, "S14 BF16 matvec output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.bf16_matvec,
            &[(weight, weight_bytes), (x, x_bytes), (y, y_bytes)],
        )?;
        Ok(S14Bf16MatvecDispatch { binder, shape })
    }

    /// Bind a BF16 matrix inside a caller-owned persistent weight arena. This
    /// is the intended 43-layer router path: upload the 86 MiB router set once,
    /// then keep one descriptor per layer without rebuilding Vulkan state.
    pub fn bind_bf16_matvec_weight_arena(
        &self,
        ctx: &VulkanContext,
        shape: S14Bf16MatvecShape,
        arena: &GpuBuffer,
        arena_logical_bytes: u64,
        weight_offset: u64,
        x: &GpuBuffer,
        y: &GpuBuffer,
    ) -> Result<S14Bf16MatvecDispatch> {
        let shape = shape.validate()?;
        let weight_bytes = storage_bytes(shape.bf16_weight_bytes()?);
        let x_bytes = shape.fp32_input_bytes()?;
        let y_bytes = shape.fp32_output_bytes()?;
        require_subrange_capacity(
            arena,
            arena_logical_bytes,
            weight_offset,
            weight_bytes,
            "S14 BF16 matvec arena weight",
        )?;
        require_storage_offset_alignment(ctx, weight_offset, "S14 BF16 matvec arena weight")?;
        require_capacity(x, x_bytes, "S14 BF16 matvec input")?;
        require_capacity(y, y_bytes, "S14 BF16 matvec output")?;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.bf16_matvec,
            &[
                (arena, weight_offset, weight_bytes),
                (x, 0, x_bytes),
                (y, 0, y_bytes),
            ],
        )?;
        Ok(S14Bf16MatvecDispatch { binder, shape })
    }

    /// Fully arena-backed variant used by the whole-token worker. Offsets are
    /// logical tensor starts, not Vulkan allocation sizes; all three are
    /// checked against both the caller's logical bound and device alignment.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_bf16_matvec_arenas(
        &self,
        ctx: &VulkanContext,
        shape: S14Bf16MatvecShape,
        weight_arena: &GpuBuffer,
        weight_arena_logical_bytes: u64,
        weight_offset: u64,
        input_arena: &GpuBuffer,
        input_arena_logical_bytes: u64,
        input_offset: u64,
        output_arena: &GpuBuffer,
        output_arena_logical_bytes: u64,
        output_offset: u64,
    ) -> Result<S14Bf16MatvecDispatch> {
        let shape = shape.validate()?;
        let weight_bytes = storage_bytes(shape.bf16_weight_bytes()?);
        let input_bytes = shape.fp32_input_bytes()?;
        let output_bytes = shape.fp32_output_bytes()?;
        require_subrange_capacity(
            weight_arena,
            weight_arena_logical_bytes,
            weight_offset,
            weight_bytes,
            "S14 BF16 weight arena",
        )?;
        require_subrange_capacity(
            input_arena,
            input_arena_logical_bytes,
            input_offset,
            input_bytes,
            "S14 BF16 input arena",
        )?;
        require_subrange_capacity(
            output_arena,
            output_arena_logical_bytes,
            output_offset,
            output_bytes,
            "S14 BF16 output arena",
        )?;
        require_storage_offset_alignment(ctx, weight_offset, "S14 BF16 weight arena")?;
        require_storage_offset_alignment(ctx, input_offset, "S14 BF16 input arena")?;
        require_storage_offset_alignment(ctx, output_offset, "S14 BF16 output arena")?;
        let binder = DescriptorBinder::new_with_offsets(
            ctx,
            &self.bf16_matvec,
            &[
                (weight_arena, weight_offset, weight_bytes),
                (input_arena, input_offset, input_bytes),
                (output_arena, output_offset, output_bytes),
            ],
        )?;
        Ok(S14Bf16MatvecDispatch { binder, shape })
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

    pub fn bind_moe_accumulate(
        &self,
        ctx: &VulkanContext,
        n: u32,
        weight: f32,
        expert: &GpuBuffer,
        accumulator: &GpuBuffer,
    ) -> Result<S14MoeAccumulateDispatch> {
        if n == 0 || !weight.is_finite() || weight < 0.0 {
            bail!("S14 MoE accumulate requires non-zero length and finite non-negative weight");
        }
        let bytes = checked_bytes(n as u64, 4, "S14 MoE accumulate")?;
        require_capacity(expert, bytes, "S14 MoE accumulate expert")?;
        require_capacity(accumulator, bytes, "S14 MoE accumulator")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.moe_accumulate,
            &[(expert, bytes), (accumulator, bytes)],
        )?;
        Ok(S14MoeAccumulateDispatch { binder, n, weight })
    }

    pub fn bind_official_expert_prepare(
        &self,
        ctx: &VulkanContext,
        n: u32,
        route_weight: f32,
        gate: &GpuBuffer,
        up: &GpuBuffer,
        hidden: &GpuBuffer,
    ) -> Result<S14OfficialExpertPrepareDispatch> {
        if n == 0 || n % 128 != 0 || !route_weight.is_finite() || route_weight < 0.0 {
            bail!(
                "S14 official expert prepare requires positive 128-aligned length and finite non-negative route weight"
            );
        }
        let bytes = checked_bytes(n as u64, 4, "S14 official expert prepare")?;
        require_capacity(gate, bytes, "S14 official prepare gate")?;
        require_capacity(up, bytes, "S14 official prepare up")?;
        require_capacity(hidden, bytes, "S14 official prepare hidden")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.official_expert_prepare,
            &[(gate, bytes), (up, bytes), (hidden, bytes)],
        )?;
        Ok(S14OfficialExpertPrepareDispatch {
            binder,
            n,
            route_weight,
        })
    }

    pub fn bind_bf16_accumulate(
        &self,
        ctx: &VulkanContext,
        n: u32,
        expert: &GpuBuffer,
        accumulator: &GpuBuffer,
    ) -> Result<S14Bf16AccumulateDispatch> {
        if n == 0 {
            bail!("S14 BF16 accumulate requires non-zero length");
        }
        let bytes = checked_bytes(n as u64, 4, "S14 BF16 accumulate")?;
        require_capacity(expert, bytes, "S14 BF16 accumulate expert")?;
        require_capacity(accumulator, bytes, "S14 BF16 accumulate output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.bf16_accumulate,
            &[(expert, bytes), (accumulator, bytes)],
        )?;
        Ok(S14Bf16AccumulateDispatch { binder, n })
    }

    pub fn bind_batched_official_expert_prepare(
        &self,
        ctx: &VulkanContext,
        branches: u32,
        n: u32,
        gate: &GpuBuffer,
        up: &GpuBuffer,
        route_weights: &GpuBuffer,
        hidden: &GpuBuffer,
    ) -> Result<S14BatchedOfficialExpertPrepareDispatch> {
        let (matrix_bytes, route_bytes) = s14_batched_official_prepare_buffer_bytes(branches, n)?;
        require_capacity(gate, matrix_bytes, "S14 batched official prepare gate")?;
        require_capacity(up, matrix_bytes, "S14 batched official prepare up")?;
        require_capacity(
            route_weights,
            route_bytes,
            "S14 batched official prepare route weights",
        )?;
        require_capacity(hidden, matrix_bytes, "S14 batched official prepare hidden")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.batched_official_expert_prepare,
            &[
                (gate, matrix_bytes),
                (up, matrix_bytes),
                (route_weights, route_bytes),
                (hidden, matrix_bytes),
            ],
        )?;
        Ok(S14BatchedOfficialExpertPrepareDispatch {
            binder,
            branches,
            n,
        })
    }

    pub fn bind_exact_order_block_reduce(
        &self,
        ctx: &VulkanContext,
        positions: u32,
        routed_down: &GpuBuffer,
        shared_down: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<S14ExactOrderBlockReduceDispatch> {
        let (routed_bytes, output_bytes) = s14_exact_order_block_reduce_buffer_bytes(positions)?;
        require_capacity(routed_down, routed_bytes, "S14 exact-order routed-down")?;
        require_capacity(shared_down, output_bytes, "S14 exact-order shared-down")?;
        require_capacity(output, output_bytes, "S14 exact-order output")?;
        let binder = DescriptorBinder::new(
            ctx,
            &self.exact_order_block_reduce,
            &[
                (routed_down, routed_bytes),
                (shared_down, output_bytes),
                (output, output_bytes),
            ],
        )?;
        Ok(S14ExactOrderBlockReduceDispatch { binder, positions })
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

    pub unsafe fn cmd_ragged_mxfp4_matvec(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14RaggedMxfp4Dispatch,
    ) {
        record_ragged_matvec(
            ctx,
            command_buffer,
            &self.ragged_mxfp4_matvec,
            dispatch.binder.set,
            dispatch.shape,
        );
    }

    pub unsafe fn cmd_ragged_fp8_matvec(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14RaggedFp8Dispatch,
    ) {
        record_ragged_matvec(
            ctx,
            command_buffer,
            &self.ragged_fp8_matvec,
            dispatch.binder.set,
            dispatch.shape,
        );
    }

    pub unsafe fn cmd_grouped_fp8_bf16_weight_matvec(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14GroupedFp8Bf16Dispatch,
    ) {
        record_grouped_matvec(
            ctx,
            command_buffer,
            &self.grouped_fp8_bf16_matvec,
            dispatch.binder.set,
            dispatch.shape,
        );
    }

    pub unsafe fn cmd_bf16_matvec(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14Bf16MatvecDispatch,
    ) {
        record_bf16_matvec(
            ctx,
            command_buffer,
            &self.bf16_matvec,
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
            256,
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
            256,
        );
    }

    pub unsafe fn cmd_moe_accumulate(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14MoeAccumulateDispatch,
    ) {
        record_elementwise(
            ctx,
            command_buffer,
            &self.moe_accumulate,
            dispatch.binder.set,
            dispatch.n,
            Some(dispatch.weight),
            256,
        );
    }

    pub unsafe fn cmd_official_expert_prepare(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14OfficialExpertPrepareDispatch,
    ) {
        record_elementwise(
            ctx,
            command_buffer,
            &self.official_expert_prepare,
            dispatch.binder.set,
            dispatch.n,
            Some(dispatch.route_weight),
            128,
        );
    }

    pub unsafe fn cmd_bf16_accumulate(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14Bf16AccumulateDispatch,
    ) {
        record_elementwise(
            ctx,
            command_buffer,
            &self.bf16_accumulate,
            dispatch.binder.set,
            dispatch.n,
            None,
            256,
        );
    }

    pub unsafe fn cmd_batched_official_expert_prepare(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14BatchedOfficialExpertPrepareDispatch,
    ) {
        record_batched_official_expert_prepare(
            ctx,
            command_buffer,
            &self.batched_official_expert_prepare,
            dispatch.binder.set,
            dispatch.branches,
            dispatch.n,
        );
    }

    pub unsafe fn cmd_exact_order_block_reduce(
        &self,
        ctx: &VulkanContext,
        command_buffer: vk::CommandBuffer,
        dispatch: &S14ExactOrderBlockReduceDispatch,
    ) {
        record_exact_order_block_reduce(
            ctx,
            command_buffer,
            &self.exact_order_block_reduce,
            dispatch.binder.set,
            dispatch.positions,
        );
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        self.exact_order_block_reduce.destroy(ctx);
        self.batched_official_expert_prepare.destroy(ctx);
        self.bf16_accumulate.destroy(ctx);
        self.official_expert_prepare.destroy(ctx);
        self.moe_accumulate.destroy(ctx);
        self.route_mix.destroy(ctx);
        self.swiglu_limit.destroy(ctx);
        self.bf16_matvec.destroy(ctx);
        self.grouped_fp8_bf16_matvec.destroy(ctx);
        self.ragged_fp8_matvec.destroy(ctx);
        self.ragged_mxfp4_matvec.destroy(ctx);
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
    local_size: u32,
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
        .cmd_dispatch(command_buffer, n.div_ceil(local_size), 1, 1);
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

unsafe fn record_bf16_matvec(
    ctx: &VulkanContext,
    command_buffer: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    shape: S14Bf16MatvecShape,
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
    const MAX_GROUPS_X: u32 = 65_535;
    let groups_x = shape.n.min(MAX_GROUPS_X);
    let groups_y = shape.n.div_ceil(groups_x);
    let mut push = [0u8; 16];
    push[..4].copy_from_slice(&shape.n.to_le_bytes());
    push[4..8].copy_from_slice(&shape.k.to_le_bytes());
    push[8..12].copy_from_slice(&shape.batch.to_le_bytes());
    push[12..].copy_from_slice(&groups_x.to_le_bytes());
    ctx.device.cmd_push_constants(
        command_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &push,
    );
    ctx.device
        .cmd_dispatch(command_buffer, groups_x, groups_y, 1);
}

unsafe fn record_ragged_matvec(
    ctx: &VulkanContext,
    command_buffer: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    shape: S14RaggedMatvecShape,
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
    let mut push = [0u8; 16];
    push[..4].copy_from_slice(&(shape.projection as u32).to_le_bytes());
    push[4..8].copy_from_slice(&shape.branches_per_input.to_le_bytes());
    push[8..12].copy_from_slice(&shape.n.to_le_bytes());
    push[12..].copy_from_slice(&shape.k.to_le_bytes());
    ctx.device.cmd_push_constants(
        command_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &push,
    );
    ctx.device
        .cmd_dispatch(command_buffer, shape.n, shape.branches, 1);
}

unsafe fn record_batched_official_expert_prepare(
    ctx: &VulkanContext,
    command_buffer: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    branches: u32,
    n: u32,
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
    push[..4].copy_from_slice(&branches.to_le_bytes());
    push[4..].copy_from_slice(&n.to_le_bytes());
    ctx.device.cmd_push_constants(
        command_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &push,
    );
    ctx.device
        .cmd_dispatch(command_buffer, n.div_ceil(FP8_TILE), branches, 1);
}

unsafe fn record_exact_order_block_reduce(
    ctx: &VulkanContext,
    command_buffer: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    positions: u32,
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
    ctx.device.cmd_push_constants(
        command_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &positions.to_le_bytes(),
    );
    let output_elements = positions * S14_BLOCK_HIDDEN;
    ctx.device
        .cmd_dispatch(command_buffer, output_elements.div_ceil(256), 1, 1);
}

unsafe fn record_grouped_matvec(
    ctx: &VulkanContext,
    command_buffer: vk::CommandBuffer,
    pipeline: &ComputePipeline,
    set: vk::DescriptorSet,
    shape: S14GroupedMatvecShape,
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
    let mut push = [0u8; 12];
    push[..4].copy_from_slice(&shape.groups.to_le_bytes());
    push[4..8].copy_from_slice(&shape.n_per_group.to_le_bytes());
    push[8..].copy_from_slice(&shape.k.to_le_bytes());
    ctx.device.cmd_push_constants(
        command_buffer,
        pipeline.layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        &push,
    );
    ctx.device
        .cmd_dispatch(command_buffer, shape.groups * shape.n_per_group, 1, 1);
}

#[derive(Debug)]
enum PageState {
    CachedReady(SlotLease),
    Loading(LoadingReservation),
}

fn acquire_page(pool: &mut VramPool, key: ExpertKey) -> Result<PageState> {
    if let Some(lease) = pool.lookup_and_pin(key)? {
        Ok(PageState::CachedReady(lease))
    } else {
        Ok(PageState::Loading(pool.reserve_loading_generation(key)?))
    }
}

fn loading_reservations(pages: &[PageState]) -> Vec<LoadingReservation> {
    pages
        .iter()
        .filter_map(|page| match page {
            PageState::CachedReady(_) => None,
            PageState::Loading(reservation) => Some(*reservation),
        })
        .collect()
}

fn validate_page_publish(pool: &VramPool, pages: &[PageState]) -> Result<()> {
    let reservations = loading_reservations(pages);
    if reservations.is_empty() {
        Ok(())
    } else {
        pool.validate_publish_batch(&reservations)
    }
}

fn publish_prevalidated_pages(pool: &mut VramPool, pages: Vec<PageState>) -> Vec<SlotLease> {
    let reservations = loading_reservations(&pages);
    let published = if reservations.is_empty() {
        Vec::new()
    } else {
        pool.publish_prevalidated_batch_and_pin(&reservations)
    };
    let mut published = published.into_iter();
    pages
        .into_iter()
        .map(|page| match page {
            PageState::CachedReady(lease) => lease,
            PageState::Loading(_) => published
                .next()
                .expect("prevalidated Loading page must have a published lease"),
        })
        .collect()
}

fn rollback_pages(pool: &mut VramPool, pages: Vec<PageState>) -> Result<()> {
    let mut first_error = None;
    for page in pages {
        let result = match page {
            PageState::CachedReady(lease) => pool.release_after_compute_fence(lease),
            PageState::Loading(reservation) => pool.cancel_reservation(reservation),
        };
        if first_error.is_none() {
            first_error = result.err();
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Routed-only compatibility ticket, now generation-safe and pinned through
/// the caller's compute fence.
#[derive(Debug)]
pub struct VulkanRouteLoad {
    pub layer: u8,
    pub expert_ids: Vec<u16>,
    pages: Vec<PageState>,
}

#[derive(Debug)]
pub struct VulkanRouteLease {
    pub layer: u8,
    pub expert_ids: Vec<u16>,
    leases: Vec<SlotLease>,
}

impl VulkanRouteLease {
    pub fn bindings(&self) -> Vec<SlotBinding> {
        self.leases.iter().map(SlotLease::binding).collect()
    }
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
            match acquire_page(self.pool, key) {
                Ok(page) => pages.push(page),
                Err(error) => {
                    let _ = rollback_pages(self.pool, pages);
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

    pub fn publish_after_fence(&mut self, ticket: VulkanRouteLoad) -> Result<VulkanRouteLease> {
        if let Err(error) = validate_page_publish(self.pool, &ticket.pages) {
            let _ = rollback_pages(self.pool, ticket.pages);
            return Err(error);
        }
        Ok(VulkanRouteLease {
            layer: ticket.layer,
            expert_ids: ticket.expert_ids,
            leases: publish_prevalidated_pages(self.pool, ticket.pages),
        })
    }

    pub fn release_after_compute_fence(&mut self, lease: VulkanRouteLease) -> Result<()> {
        for page in &lease.leases {
            self.pool.validate_binding(page.binding())?;
        }
        for page in lease.leases {
            self.pool.release_after_compute_fence(page)?;
        }
        Ok(())
    }

    pub fn cancel(&mut self, ticket: VulkanRouteLoad) -> Result<()> {
        rollback_pages(self.pool, ticket.pages)
    }
}

/// One transfer transaction for all six routed pages plus the layer's shared
/// FP8 page. No page is published until both pools pass generation preflight.
#[derive(Debug)]
pub struct RouteLoadBatch {
    pub layer: u8,
    pub expert_ids: Vec<u16>,
    pub route_weights: Vec<f32>,
    routed_pages: Vec<PageState>,
    shared_page: PageState,
}

#[derive(Debug)]
pub struct S14MoeComputeLeaseBatch {
    layer: u8,
    expert_ids: Vec<u16>,
    route_weights: Vec<f32>,
    routed: Vec<SlotLease>,
    shared: SlotLease,
}

impl S14MoeComputeLeaseBatch {
    pub fn layer(&self) -> u8 {
        self.layer
    }

    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    pub fn route_weights(&self) -> &[f32] {
        &self.route_weights
    }

    pub fn routed_bindings(&self) -> Vec<SlotBinding> {
        self.routed.iter().map(SlotLease::binding).collect()
    }

    pub fn shared_binding(&self) -> SlotBinding {
        self.shared.binding()
    }
}

pub struct S14VulkanMoePages<'a> {
    routed_pool: &'a mut VramPool,
    shared_pool: &'a mut VramPool,
    profile: GraphProfile,
}

impl<'a> S14VulkanMoePages<'a> {
    pub fn new(
        routed_pool: &'a mut VramPool,
        shared_pool: &'a mut VramPool,
        profile: GraphProfile,
    ) -> Result<Self> {
        if routed_pool.slot_bytes() < EXPERT_PAGE_BYTES {
            bail!(
                "S14 routed slot is {} B, requires at least {} B",
                routed_pool.slot_bytes(),
                EXPERT_PAGE_BYTES
            );
        }
        if shared_pool.slot_bytes() < S14_SHARED_EXPERT_PAGE_BYTES {
            bail!(
                "S14 shared slot is {} B, requires at least {} B",
                shared_pool.slot_bytes(),
                S14_SHARED_EXPERT_PAGE_BYTES
            );
        }
        Ok(Self {
            routed_pool,
            shared_pool,
            profile,
        })
    }

    pub fn begin_after_official_route(&mut self, route: &RouteDecision) -> Result<RouteLoadBatch> {
        route
            .validate_for(self.profile)
            .map_err(|error| anyhow!(error.to_string()))?;
        let mut routed_pages = Vec::with_capacity(route.expert_ids.len());
        for &expert in &route.expert_ids {
            let key = ExpertKey {
                layer: route.layer as u32,
                kind: S14_ROUTED_EXPERT_KIND,
                slot: expert as u32,
            };
            match acquire_page(self.routed_pool, key) {
                Ok(page) => routed_pages.push(page),
                Err(error) => {
                    let _ = rollback_pages(self.routed_pool, routed_pages);
                    return Err(error);
                }
            }
        }
        let shared_key = ExpertKey {
            layer: route.layer as u32,
            kind: S14_SHARED_EXPERT_KIND,
            slot: 0,
        };
        let shared_page = match acquire_page(self.shared_pool, shared_key) {
            Ok(page) => page,
            Err(error) => {
                let _ = rollback_pages(self.routed_pool, routed_pages);
                return Err(error);
            }
        };
        Ok(RouteLoadBatch {
            layer: route.layer,
            expert_ids: route.expert_ids.clone(),
            route_weights: route.weights.clone(),
            routed_pages,
            shared_page,
        })
    }

    /// Sole publication point after the transfer fence for the complete batch.
    pub fn publish_after_transfer_fence(
        &mut self,
        batch: RouteLoadBatch,
    ) -> Result<S14MoeComputeLeaseBatch> {
        let shared_pages = std::slice::from_ref(&batch.shared_page);
        if let Err(error) = validate_page_publish(self.routed_pool, &batch.routed_pages)
            .and_then(|_| validate_page_publish(self.shared_pool, shared_pages))
        {
            let _ = rollback_pages(self.routed_pool, batch.routed_pages);
            let _ = rollback_pages(self.shared_pool, vec![batch.shared_page]);
            return Err(error);
        }

        let routed = publish_prevalidated_pages(self.routed_pool, batch.routed_pages);
        let shared =
            publish_prevalidated_pages(self.shared_pool, vec![batch.shared_page]).remove(0);
        Ok(S14MoeComputeLeaseBatch {
            layer: batch.layer,
            expert_ids: batch.expert_ids,
            route_weights: batch.route_weights,
            routed,
            shared,
        })
    }

    pub fn release_after_compute_fence(&mut self, batch: S14MoeComputeLeaseBatch) -> Result<()> {
        for lease in &batch.routed {
            self.routed_pool.validate_binding(lease.binding())?;
        }
        self.shared_pool.validate_binding(batch.shared.binding())?;
        for lease in batch.routed {
            self.routed_pool.release_after_compute_fence(lease)?;
        }
        self.shared_pool.release_after_compute_fence(batch.shared)?;
        Ok(())
    }

    pub fn cancel(&mut self, batch: RouteLoadBatch) -> Result<()> {
        let routed_result = rollback_pages(self.routed_pool, batch.routed_pages);
        let shared_result = rollback_pages(self.shared_pool, vec![batch.shared_page]);
        routed_result.and(shared_result)
    }
}

#[cfg(test)]
mod numeric_tests {
    use super::*;

    fn route(ids: [u16; 6]) -> RouteDecision {
        RouteDecision {
            layer: 42,
            kind: polaris_s14_runner::RouterKind::Score,
            expert_ids: ids.to_vec(),
            weights: vec![0.25; 6],
        }
    }

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

        let wo_a = S14GroupedMatvecShape::new(8, 1024, 4096)
            .unwrap()
            .validate_fp8_bf16_weight()
            .unwrap();
        assert_eq!(wo_a.flat_n().unwrap(), 8192);
        assert_eq!(wo_a.fp32_input_bytes().unwrap(), 131_072);
        assert_eq!(wo_a.fp8_weight_bytes().unwrap(), 33_554_432);
        assert_eq!(wo_a.fp8_scale_bytes().unwrap(), 2_048);
        assert_eq!(wo_a.fp32_output_bytes().unwrap(), 32_768);

        let router = S14Bf16MatvecShape::new(256, 4096, 4).unwrap();
        assert_eq!(router.bf16_weight_bytes().unwrap(), 2_097_152);
        assert_eq!(router.fp32_input_bytes().unwrap(), 65_536);
        assert_eq!(router.fp32_output_bytes().unwrap(), 4_096);
    }

    #[test]
    fn bf16_matvec_rejects_odd_packed_storage() {
        assert!(S14Bf16MatvecShape::new(1, 3, 1).is_err());
        assert!(S14Bf16MatvecShape::new(256, 4096, 0).is_err());
        assert!(S14Bf16MatvecShape::new(256, 4096, 9).is_err());
        assert!(S14Bf16MatvecShape::new(u32::MAX, 2, 1).is_err());
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

    #[test]
    fn ragged_metadata_abi_and_k4_shapes_are_exact() {
        assert_eq!(std::mem::size_of::<S14RaggedBranchOffsets>(), 24);
        let shape = S14RaggedMatvecShape::new(4, 2, 2048, 4096, S14RaggedProjection::W1).unwrap();
        assert_eq!(shape.input_rows().unwrap(), 2);
        assert_eq!(shape.fp32_input_bytes().unwrap(), 32_768);
        assert_eq!(shape.fp32_output_bytes().unwrap(), 32_768);
        assert_eq!(shape.metadata_bytes().unwrap(), 96);

        let stride = 4_456_448u32;
        let metadata: Vec<_> = (0..4)
            .map(|branch| {
                let weight = branch * stride;
                let scale = weight + 4_194_304;
                S14RaggedBranchOffsets {
                    w1: weight,
                    s1: scale,
                    w3: weight,
                    s3: scale,
                    w2: weight,
                    s2: scale,
                }
            })
            .collect();
        shape.validate_mxfp4(stride as u64 * 4, &metadata).unwrap();
    }

    #[test]
    fn ragged_contract_rejects_bad_projection_alignment_bounds_and_overflow() {
        assert!(S14RaggedProjection::try_from(3).is_err());
        assert!(S14RaggedMatvecShape::new(u32::MAX, 1, 2, 128, S14RaggedProjection::W1).is_err());
        assert!(S14RaggedMatvecShape::new(u32::MAX, 2, 1, 128, S14RaggedProjection::W1).is_err());

        let shape = S14RaggedMatvecShape::new(1, 1, 128, 128, S14RaggedProjection::W3).unwrap();
        let mut metadata = [S14RaggedBranchOffsets {
            w1: 0,
            s1: 0,
            w3: 0,
            s3: 16_384,
            w2: 0,
            s2: 0,
        }];
        shape.validate_fp8(16_388, &metadata).unwrap();
        metadata[0].w3 = 2;
        assert!(shape.validate_fp8(16_388, &metadata).is_err());
        metadata[0].w3 = 8;
        assert!(shape.validate_fp8(16_388, &metadata).is_err());
        metadata[0].w3 = 0;
        assert!(shape.validate_fp8(16_384, &metadata).is_err());
        assert!(shape.validate_fp8(16_388, &[]).is_err());
        assert!(shape.validate_fp8(u32::MAX as u64 + 2, &metadata).is_err());
    }

    #[test]
    fn batched_prepare_and_exact_reduce_sizes_reject_index_overflow() {
        assert_eq!(
            s14_batched_official_prepare_buffer_bytes(4, 2048).unwrap(),
            (32_768, 16)
        );
        assert!(s14_batched_official_prepare_buffer_bytes(0, 2048).is_err());
        assert!(s14_batched_official_prepare_buffer_bytes(4, 2047).is_err());
        assert!(s14_batched_official_prepare_buffer_bytes(u32::MAX, 128).is_err());

        assert_eq!(
            s14_exact_order_block_reduce_buffer_bytes(1).unwrap(),
            (98_304, 16_384)
        );
        assert!(s14_exact_order_block_reduce_buffer_bytes(0).is_err());
        assert!(s14_exact_order_block_reduce_buffer_bytes(u32::MAX).is_err());
    }

    #[test]
    fn route_load_batch_rolls_back_every_reservation_on_capacity_failure() {
        let mut routed = VramPool::new_for_tests(5, EXPERT_PAGE_BYTES);
        let mut shared = VramPool::new_for_tests(1, S14_SHARED_EXPERT_PAGE_BYTES);
        {
            let mut pages =
                S14VulkanMoePages::new(&mut routed, &mut shared, GraphProfile::S14Top6).unwrap();
            assert!(pages
                .begin_after_official_route(&route([126, 12, 205, 149, 227, 174]))
                .is_err());
        }
        assert_eq!(routed.n_loaded(), 0);
        assert_eq!(shared.n_loaded(), 0);
    }

    #[test]
    fn route_load_batch_rejects_duplicate_or_weight_drift_before_reservation() {
        let mut routed = VramPool::new_for_tests(6, EXPERT_PAGE_BYTES);
        let mut shared = VramPool::new_for_tests(1, S14_SHARED_EXPERT_PAGE_BYTES);
        {
            let mut pages =
                S14VulkanMoePages::new(&mut routed, &mut shared, GraphProfile::S14Top6).unwrap();
            assert!(pages
                .begin_after_official_route(&route([126, 126, 205, 149, 227, 174]))
                .is_err());
            let mut drift = route([126, 12, 205, 149, 227, 174]);
            drift.weights[0] = 0.0;
            assert!(pages.begin_after_official_route(&drift).is_err());
        }
        assert_eq!(routed.n_loaded(), 0);
        assert_eq!(shared.n_loaded(), 0);
    }

    #[test]
    fn route_load_batch_capacity_failure_releases_cached_pin() {
        let resident = ExpertKey {
            layer: 42,
            kind: S14_ROUTED_EXPERT_KIND,
            slot: 126,
        };
        let mut routed = VramPool::new_for_tests(1, EXPERT_PAGE_BYTES);
        let reservation = routed.reserve_loading_generation(resident).unwrap();
        let lease = routed
            .publish_batch_and_pin(&[reservation])
            .unwrap()
            .remove(0);
        routed.release_after_compute_fence(lease).unwrap();
        let mut shared = VramPool::new_for_tests(1, S14_SHARED_EXPERT_PAGE_BYTES);
        {
            let mut pages =
                S14VulkanMoePages::new(&mut routed, &mut shared, GraphProfile::S14Top6).unwrap();
            assert!(pages
                .begin_after_official_route(&route([126, 12, 205, 149, 227, 174]))
                .is_err());
        }
        assert!(routed
            .reserve_loading_generation(ExpertKey {
                layer: 42,
                kind: S14_ROUTED_EXPERT_KIND,
                slot: 7,
            })
            .is_ok());
    }

    #[test]
    fn route_load_batch_publishes_all_then_holds_compute_fence_pins() {
        let mut routed = VramPool::new_for_tests(6, EXPERT_PAGE_BYTES);
        let mut shared = VramPool::new_for_tests(1, S14_SHARED_EXPERT_PAGE_BYTES);
        let mut pages =
            S14VulkanMoePages::new(&mut routed, &mut shared, GraphProfile::S14Top6).unwrap();
        let batch = pages
            .begin_after_official_route(&route([126, 12, 205, 149, 227, 174]))
            .unwrap();
        let compute = pages.publish_after_transfer_fence(batch).unwrap();
        assert_eq!(compute.routed_bindings().len(), 6);
        assert_eq!(compute.expert_ids(), &[126, 12, 205, 149, 227, 174]);

        assert!(pages
            .begin_after_official_route(&route([1, 2, 3, 4, 5, 6]))
            .is_err());
        pages.release_after_compute_fence(compute).unwrap();
        let replacement = pages
            .begin_after_official_route(&route([1, 2, 3, 4, 5, 6]))
            .unwrap();
        pages.cancel(replacement).unwrap();
    }

    #[test]
    fn stale_shared_generation_prevents_partial_routed_publication() {
        let ids = [126, 12, 205, 149, 227, 174];
        let mut routed = VramPool::new_for_tests(6, EXPERT_PAGE_BYTES);
        let mut shared = VramPool::new_for_tests(1, S14_SHARED_EXPERT_PAGE_BYTES);
        let mut pages =
            S14VulkanMoePages::new(&mut routed, &mut shared, GraphProfile::S14Top6).unwrap();
        let batch = pages.begin_after_official_route(&route(ids)).unwrap();
        let stale_shared = match &batch.shared_page {
            PageState::Loading(reservation) => *reservation,
            PageState::CachedReady(_) => panic!("first shared page must be Loading"),
        };
        pages.shared_pool.cancel_reservation(stale_shared).unwrap();
        let unrelated = pages
            .shared_pool
            .reserve_loading_generation(ExpertKey {
                layer: 42,
                kind: S14_SHARED_EXPERT_KIND,
                slot: 1,
            })
            .unwrap();

        assert!(pages.publish_after_transfer_fence(batch).is_err());
        for expert in ids {
            assert!(pages
                .routed_pool
                .lookup(ExpertKey {
                    layer: 42,
                    kind: S14_ROUTED_EXPERT_KIND,
                    slot: expert as u32,
                })
                .is_none());
        }
        pages.shared_pool.cancel_reservation(unrelated).unwrap();
    }
}
