//! FullDepth43 position0 hybrid 权重计划的真实 Vulkan 显存 owner。
//!
//! `S14Position0HybridWeightPlan::resident` 的逻辑总量约 6.7 GiB，但不能把它
//! 变成单个 Vulkan allocation。本模块把它重新打包为 43 个逐层静态 buffer 和
//! 一个很小的 embedding/final buffer；top-6 routed expert 与词表头各使用两个
//! 交替 bank。这里只分配显存并冻结物理布局，不上传 payload，也不产生 token。

use crate::{
    s14_position0_weight_plan::{
        S14Position0AssetPlacement, S14Position0HybridWeightPlan, S14_POSITION0_ROLLING_BANKS,
        S14_POSITION0_WEIGHT_ALIGNMENT,
    },
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::FULL_DEPTH_LAYERS;

pub const S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
pub const S14_POSITION0_HYBRID_ALLOCATION_COUNT: usize =
    FULL_DEPTH_LAYERS.len() + 1 + S14_POSITION0_ROLLING_BANKS * 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0PhysicalAssetPlacement {
    pub tensor: String,
    /// `S14Position0HybridWeightPlan::resident` 中的原始逻辑 offset。
    pub source_offset: u64,
    /// 拆分后的目标 buffer 内 offset。
    pub local_offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0StaticLayerLayout {
    pub layer: u8,
    pub requested_bytes: u64,
    pub assets: Vec<S14Position0PhysicalAssetPlacement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0ResidentSmallLayout {
    pub requested_bytes: u64,
    pub assets: Vec<S14Position0PhysicalAssetPlacement>,
}

/// 不接触 GPU 的确定物理布局。上传器可以使用 `source_offset -> local_offset`
/// 映射把 hybrid plan 的 resident payload 精确写入拆分后的 buffer。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0HybridArenaLayout {
    pub static_layers: Vec<S14Position0StaticLayerLayout>,
    pub resident_small: S14Position0ResidentSmallLayout,
    pub static_requested_bytes: u64,
    pub routed_bank_bytes: u64,
    pub head_chunk_bytes: u64,
    pub requested_device_bytes: u64,
}

impl S14Position0HybridArenaLayout {
    pub fn build(plan: &S14Position0HybridWeightPlan) -> Result<Self> {
        validate_hybrid_plan_ledger(plan)?;

        let mut layer_assets = (0..FULL_DEPTH_LAYERS.len())
            .map(|_| Vec::<&S14Position0AssetPlacement>::new())
            .collect::<Vec<_>>();
        let mut small_assets = Vec::<&S14Position0AssetPlacement>::new();

        for asset in &plan.resident.assets {
            if asset.tensor.starts_with("layers.") {
                let layer = parse_layer_id(&asset.tensor)
                    .ok_or_else(|| anyhow!("invalid resident layer tensor: {}", asset.tensor))?;
                let index = FULL_DEPTH_LAYERS
                    .iter()
                    .position(|&expected| expected == layer)
                    .ok_or_else(|| anyhow!("unexpected resident layer L{layer}"))?;
                layer_assets[index].push(asset);
            } else {
                small_assets.push(asset);
            }
        }

        let mut static_layers = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
        let mut static_requested_bytes = 0u64;
        for (&layer, assets) in FULL_DEPTH_LAYERS.iter().zip(layer_assets) {
            if assets.is_empty() {
                bail!("hybrid resident plan has no static assets for L{layer}");
            }
            let (assets, requested_bytes) = repack_assets(assets)
                .with_context(|| format!("repack hybrid static assets for L{layer}"))?;
            static_requested_bytes = static_requested_bytes
                .checked_add(requested_bytes)
                .ok_or_else(|| anyhow!("hybrid static allocation bytes overflow"))?;
            static_layers.push(S14Position0StaticLayerLayout {
                layer,
                requested_bytes,
                assets,
            });
        }

        if small_assets.is_empty() {
            bail!("hybrid resident plan has no embedding/final-small assets");
        }
        let (small_assets, resident_small_bytes) =
            repack_assets(small_assets).context("repack hybrid resident-small assets")?;
        let physical_resident_bytes = static_requested_bytes
            .checked_add(resident_small_bytes)
            .ok_or_else(|| anyhow!("hybrid physical resident bytes overflow"))?;
        if physical_resident_bytes != plan.resident.used_bytes {
            bail!(
                "hybrid resident split ledger drift: plan={} physical={}",
                plan.resident.used_bytes,
                physical_resident_bytes
            );
        }

        let requested_device_bytes = physical_resident_bytes
            .checked_add(plan.routed_device_bytes)
            .and_then(|bytes| bytes.checked_add(plan.head_device_bytes))
            .ok_or_else(|| anyhow!("hybrid physical device bytes overflow"))?;
        if requested_device_bytes != plan.device_weight_bytes {
            bail!(
                "hybrid physical device ledger drift: plan={} physical={}",
                plan.device_weight_bytes,
                requested_device_bytes
            );
        }

        Ok(Self {
            static_layers,
            resident_small: S14Position0ResidentSmallLayout {
                requested_bytes: resident_small_bytes,
                assets: small_assets,
            },
            static_requested_bytes,
            routed_bank_bytes: plan.routed_bank_bytes,
            head_chunk_bytes: plan.head_chunk_bytes,
            requested_device_bytes,
        })
    }

    pub fn largest_requested_allocation_bytes(&self) -> u64 {
        self.static_layers
            .iter()
            .map(|layer| layer.requested_bytes)
            .chain([
                self.resident_small.requested_bytes,
                self.routed_bank_bytes,
                self.head_chunk_bytes,
            ])
            .max()
            .unwrap_or(0)
    }
}

pub struct S14Position0HybridWeightArena {
    layout: S14Position0HybridArenaLayout,
    static_layers: Vec<GpuBuffer>,
    resident_small: GpuBuffer,
    routed: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    head_chunks: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    allocated_static_bytes: u64,
    allocated_resident_small_bytes: u64,
    allocated_routed_bytes: u64,
    allocated_head_bytes: u64,
    allocated_device_bytes: u64,
    vram_bytes: u64,
}

impl S14Position0HybridWeightArena {
    pub fn new(ctx: &VulkanContext, plan: &S14Position0HybridWeightPlan) -> Result<Self> {
        let layout = S14Position0HybridArenaLayout::build(plan)?;
        let vram_bytes = ctx.vram_size();
        let requested_with_reserve = layout
            .requested_device_bytes
            .checked_add(S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES)
            .ok_or_else(|| anyhow!("hybrid VRAM requirement overflow"))?;
        if requested_with_reserve > vram_bytes {
            bail!(
                "hybrid weight arena exceeds VRAM budget: weights={} reserve={} vram={}",
                layout.requested_device_bytes,
                S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES,
                vram_bytes
            );
        }

        let mut pending = PendingAllocations::new(ctx);
        for layer in &layout.static_layers {
            pending.allocate(layer.requested_bytes).with_context(|| {
                format!(
                    "allocate hybrid static L{} buffer ({} bytes)",
                    layer.layer, layer.requested_bytes
                )
            })?;
        }
        pending
            .allocate(layout.resident_small.requested_bytes)
            .context("allocate hybrid resident-small buffer")?;
        for bank in 0..S14_POSITION0_ROLLING_BANKS {
            pending
                .allocate(layout.routed_bank_bytes)
                .with_context(|| format!("allocate hybrid routed bank {bank}"))?;
        }
        for chunk in 0..S14_POSITION0_ROLLING_BANKS {
            pending
                .allocate(layout.head_chunk_bytes)
                .with_context(|| format!("allocate hybrid head chunk {chunk}"))?;
        }
        if pending.len() != S14_POSITION0_HYBRID_ALLOCATION_COUNT {
            bail!("hybrid physical allocation count drift");
        }

        let allocated_device_bytes = pending.allocated_bytes()?;
        let actual_with_reserve = allocated_device_bytes
            .checked_add(S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES)
            .ok_or_else(|| anyhow!("hybrid allocated VRAM requirement overflow"))?;
        if actual_with_reserve > vram_bytes {
            bail!(
                "hybrid allocated arena leaves less than workspace reserve: allocated={} reserve={} vram={}",
                allocated_device_bytes,
                S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES,
                vram_bytes
            );
        }

        let mut buffers = pending.finish();
        let static_layers = buffers.drain(..FULL_DEPTH_LAYERS.len()).collect::<Vec<_>>();
        let resident_small = buffers.remove(0);
        let routed = [buffers.remove(0), buffers.remove(0)];
        let head_chunks = [buffers.remove(0), buffers.remove(0)];
        debug_assert!(buffers.is_empty());

        let allocated_static_bytes = sum_buffer_bytes(&static_layers)?;
        let allocated_resident_small_bytes = resident_small.size();
        let allocated_routed_bytes = sum_buffer_bytes(&routed)?;
        let allocated_head_bytes = sum_buffer_bytes(&head_chunks)?;
        let category_total = allocated_static_bytes
            .checked_add(allocated_resident_small_bytes)
            .and_then(|bytes| bytes.checked_add(allocated_routed_bytes))
            .and_then(|bytes| bytes.checked_add(allocated_head_bytes))
            .ok_or_else(|| anyhow!("hybrid allocated category ledger overflow"))?;
        if category_total != allocated_device_bytes {
            bail!("hybrid allocated category ledger drift");
        }

        Ok(Self {
            layout,
            static_layers,
            resident_small,
            routed,
            head_chunks,
            allocated_static_bytes,
            allocated_resident_small_bytes,
            allocated_routed_bytes,
            allocated_head_bytes,
            allocated_device_bytes,
            vram_bytes,
        })
    }

    pub fn layout(&self) -> &S14Position0HybridArenaLayout {
        &self.layout
    }

    pub fn static_layer(&self, layer: u8) -> Result<&GpuBuffer> {
        let index = FULL_DEPTH_LAYERS
            .iter()
            .position(|&expected| expected == layer)
            .ok_or_else(|| anyhow!("invalid hybrid static layer L{layer}"))?;
        Ok(&self.static_layers[index])
    }

    pub fn resident_small(&self) -> &GpuBuffer {
        &self.resident_small
    }

    pub fn routed(&self, bank: usize) -> Result<&GpuBuffer> {
        self.routed
            .get(bank)
            .ok_or_else(|| anyhow!("invalid hybrid routed bank {bank}"))
    }

    pub fn head_chunk(&self, chunk: usize) -> Result<&GpuBuffer> {
        self.head_chunks
            .get(chunk)
            .ok_or_else(|| anyhow!("invalid hybrid head chunk {chunk}"))
    }

    pub fn allocation_count(&self) -> usize {
        self.static_layers.len() + 1 + self.routed.len() + self.head_chunks.len()
    }

    pub fn requested_device_bytes(&self) -> u64 {
        self.layout.requested_device_bytes
    }

    pub fn allocated_device_bytes(&self) -> u64 {
        self.allocated_device_bytes
    }

    pub fn allocated_static_bytes(&self) -> u64 {
        self.allocated_static_bytes
    }

    pub fn allocated_resident_small_bytes(&self) -> u64 {
        self.allocated_resident_small_bytes
    }

    pub fn allocated_routed_bytes(&self) -> u64 {
        self.allocated_routed_bytes
    }

    pub fn allocated_head_bytes(&self) -> u64 {
        self.allocated_head_bytes
    }

    /// 这是按 Vulkan DEVICE_LOCAL heap 总量减本 arena 实际 allocation 的保守账本余量；
    /// 不把它冒充 `VK_EXT_memory_budget` 的进程瞬时可用值。
    pub fn accounted_workspace_bytes(&self) -> u64 {
        self.vram_bytes.saturating_sub(self.allocated_device_bytes)
    }

    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        for buffer in &self.head_chunks {
            buffer.destroy(ctx);
        }
        for buffer in &self.routed {
            buffer.destroy(ctx);
        }
        self.resident_small.destroy(ctx);
        for buffer in &self.static_layers {
            buffer.destroy(ctx);
        }
    }
}

struct PendingAllocations<'a> {
    ctx: &'a VulkanContext,
    buffers: Vec<GpuBuffer>,
}

impl<'a> PendingAllocations<'a> {
    fn new(ctx: &'a VulkanContext) -> Self {
        Self {
            ctx,
            buffers: Vec::with_capacity(S14_POSITION0_HYBRID_ALLOCATION_COUNT),
        }
    }

    fn allocate(&mut self, bytes: u64) -> Result<()> {
        if bytes == 0 {
            bail!("hybrid arena refuses zero-byte allocation");
        }
        let committed_allocations = self.buffers.len();
        let committed_actual_bytes = self.allocated_bytes()?;
        let buffer = make_vram(self.ctx, bytes).with_context(|| {
            format!(
                "device allocation failed after {} buffers / {} actual bytes; next_requested_bytes={} heap_bytes={}",
                committed_allocations,
                committed_actual_bytes,
                bytes,
                self.ctx.vram_size()
            )
        })?;
        self.buffers.push(buffer);
        Ok(())
    }

    fn len(&self) -> usize {
        self.buffers.len()
    }

    fn allocated_bytes(&self) -> Result<u64> {
        sum_buffer_bytes(&self.buffers)
    }

    fn finish(mut self) -> Vec<GpuBuffer> {
        std::mem::take(&mut self.buffers)
    }
}

impl Drop for PendingAllocations<'_> {
    fn drop(&mut self) {
        for buffer in &self.buffers {
            buffer.destroy(self.ctx);
        }
    }
}

fn make_vram(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;
    if ctx.qf_transfer == ctx.qf_graphics {
        GpuBuffer::new_vram(ctx, bytes, usage)
    } else {
        GpuBuffer::new_vram_shared(ctx, bytes, usage, &[ctx.qf_transfer, ctx.qf_graphics])
    }
}

fn validate_hybrid_plan_ledger(plan: &S14Position0HybridWeightPlan) -> Result<()> {
    if plan.resident.used_bytes == 0
        || plan.routed_layers.len() != FULL_DEPTH_LAYERS.len()
        || plan.routed_bank_bytes == 0
        || plan.head_chunk_bytes == 0
        || plan.routed_device_bytes
            != plan
                .routed_bank_bytes
                .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
                .ok_or_else(|| anyhow!("hybrid routed bytes overflow"))?
        || plan.head_device_bytes
            != plan
                .head_chunk_bytes
                .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
                .ok_or_else(|| anyhow!("hybrid head bytes overflow"))?
        || plan.device_weight_bytes
            != plan
                .resident
                .used_bytes
                .checked_add(plan.routed_device_bytes)
                .and_then(|bytes| bytes.checked_add(plan.head_device_bytes))
                .ok_or_else(|| anyhow!("hybrid device bytes overflow"))?
    {
        bail!("hybrid weight plan ledger is not self-consistent");
    }
    Ok(())
}

fn repack_assets(
    assets: Vec<&S14Position0AssetPlacement>,
) -> Result<(Vec<S14Position0PhysicalAssetPlacement>, u64)> {
    let mut cursor = 0u64;
    let mut physical = Vec::with_capacity(assets.len());
    for asset in assets {
        if asset.bytes == 0
            || asset.offset % S14_POSITION0_WEIGHT_ALIGNMENT != 0
            || asset.offset.checked_add(asset.bytes).is_none()
        {
            bail!("invalid hybrid resident placement: {}", asset.tensor);
        }
        cursor = align_up(cursor, S14_POSITION0_WEIGHT_ALIGNMENT)?;
        physical.push(S14Position0PhysicalAssetPlacement {
            tensor: asset.tensor.clone(),
            source_offset: asset.offset,
            local_offset: cursor,
            bytes: asset.bytes,
        });
        cursor = cursor
            .checked_add(asset.bytes)
            .ok_or_else(|| anyhow!("hybrid physical asset overflow: {}", asset.tensor))?;
    }
    Ok((physical, align_up(cursor, S14_POSITION0_WEIGHT_ALIGNMENT)?))
}

fn parse_layer_id(tensor: &str) -> Option<u8> {
    let rest = tensor.strip_prefix("layers.")?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse().ok()
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("alignment must be a non-zero power of two");
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| anyhow!("alignment overflow"))
}

fn sum_buffer_bytes(buffers: &[GpuBuffer]) -> Result<u64> {
    buffers.iter().try_fold(0u64, |total, buffer| {
        total
            .checked_add(buffer.size())
            .ok_or_else(|| anyhow!("hybrid allocated bytes overflow"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use polaris_s14_runner::Position0WholeTokenManifest;
    use std::path::PathBuf;

    fn hybrid_plan() -> S14Position0HybridWeightPlan {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        let manifest = Position0WholeTokenManifest::load(&path).unwrap();
        S14Position0HybridWeightPlan::build(&manifest).unwrap()
    }

    #[test]
    fn real_hybrid_plan_splits_resident_into_43_small_allocations() {
        let plan = hybrid_plan();
        let layout = S14Position0HybridArenaLayout::build(&plan).unwrap();
        assert_eq!(layout.static_layers.len(), 43);
        assert_eq!(layout.static_layers[0].layer, 0);
        assert_eq!(layout.static_layers[42].layer, 42);
        assert_eq!(layout.resident_small.assets.len(), 5);
        assert!(layout.resident_small.requested_bytes < 512 * 1024);
        assert!(layout.largest_requested_allocation_bytes() < 192 * 1024 * 1024);
        assert_eq!(layout.requested_device_bytes, plan.device_weight_bytes);
        assert_eq!(
            layout.static_requested_bytes + layout.resident_small.requested_bytes,
            plan.resident.used_bytes
        );
        assert_eq!(S14_POSITION0_HYBRID_ALLOCATION_COUNT, 48);
    }

    #[test]
    fn physical_asset_offsets_are_aligned_and_capacity_bounded() {
        let layout = S14Position0HybridArenaLayout::build(&hybrid_plan()).unwrap();
        for layer in &layout.static_layers {
            for asset in &layer.assets {
                assert_eq!(asset.local_offset % S14_POSITION0_WEIGHT_ALIGNMENT, 0);
                assert!(asset.local_offset + asset.bytes <= layer.requested_bytes);
            }
        }
        for asset in &layout.resident_small.assets {
            assert_eq!(asset.local_offset % S14_POSITION0_WEIGHT_ALIGNMENT, 0);
            assert!(asset.local_offset + asset.bytes <= layout.resident_small.requested_bytes);
        }
    }

    #[test]
    fn tampered_hybrid_plan_fails_before_gpu_allocation() {
        let mut plan = hybrid_plan();
        plan.device_weight_bytes -= 1;
        assert!(S14Position0HybridArenaLayout::build(&plan).is_err());

        let mut plan = hybrid_plan();
        plan.resident.assets[1].tensor = "layers.99.invalid".to_owned();
        assert!(S14Position0HybridArenaLayout::build(&plan).is_err());
    }
}
