//! FullDepth43 position0 的 WDDM 自适应 Hybrid 权重 arena。
//!
//! 43 个静态层在空闲 8 GiB 卡上可能接近全常驻，但 Windows 桌面进程会动态占用
//! 设备内存。把 6.7 GiB 静态权重视为必须同时常驻会在分配后段直接失败。本模块先
//! 物理保留 whole-token workspace、routed/head 双 bank 与静态层双页，再按 L0→L42
//! 贪心常驻静态层；第一次可选分配失败后，其余层走双页流式路径。可选失败不是数值
//! 降级：每一层仍使用原生完整权重，只改变权重驻留位置。

use crate::{
    s14_position0_hybrid_weight_arena::{
        S14Position0HybridArenaLayout, S14Position0StaticLayerLayout,
        S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES,
    },
    s14_position0_weight_plan::{S14Position0HybridWeightPlan, S14_POSITION0_ROLLING_BANKS},
    s14_position0_workspace::S14Position0WorkspaceLayout,
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::FULL_DEPTH_LAYERS;
use std::sync::Mutex;

pub const S14_POSITION0_STATIC_STREAM_BANKS: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0PagedArenaPlan {
    pub physical: S14Position0HybridArenaLayout,
    pub static_stream_bank_bytes: u64,
    pub static_stream_device_bytes: u64,
    pub workspace_bytes: u64,
    /// 不含任何可选静态常驻层；这一部分必须先真实分配成功。
    pub essential_device_bytes: u64,
}

impl S14Position0PagedArenaPlan {
    pub fn build(plan: &S14Position0HybridWeightPlan) -> Result<Self> {
        let physical = S14Position0HybridArenaLayout::build(plan)?;
        let static_stream_bank_bytes = physical
            .static_layers
            .iter()
            .map(|layer| layer.requested_bytes)
            .max()
            .ok_or_else(|| anyhow!("position0 paged arena 没有静态层"))?;
        let static_stream_device_bytes = static_stream_bank_bytes
            .checked_mul(S14_POSITION0_STATIC_STREAM_BANKS as u64)
            .ok_or_else(|| anyhow!("position0 static stream bytes overflow"))?;
        let workspace_bytes = S14_POSITION0_MIN_WORKSPACE_RESERVE_BYTES;
        let essential_device_bytes = physical
            .resident_small
            .requested_bytes
            .checked_add(
                physical
                    .routed_bank_bytes
                    .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
                    .ok_or_else(|| anyhow!("position0 routed bytes overflow"))?,
            )
            .and_then(|bytes| {
                bytes.checked_add(
                    physical
                        .head_chunk_bytes
                        .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)?,
                )
            })
            .and_then(|bytes| bytes.checked_add(static_stream_device_bytes))
            .and_then(|bytes| bytes.checked_add(workspace_bytes))
            .ok_or_else(|| anyhow!("position0 paged essential bytes overflow"))?;
        Ok(Self {
            physical,
            static_stream_bank_bytes,
            static_stream_device_bytes,
            workspace_bytes,
            essential_device_bytes,
        })
    }
}

pub enum S14Position0StaticLayerBinding<'a> {
    Resident {
        buffer: &'a GpuBuffer,
        layout: &'a S14Position0StaticLayerLayout,
    },
    Streamed {
        bank: usize,
        buffer: &'a GpuBuffer,
        layout: &'a S14Position0StaticLayerLayout,
    },
}

pub struct S14Position0StaticAssetBinding<'a> {
    pub buffer: &'a GpuBuffer,
    pub destination_offset: u64,
    /// `false` 表示该资产属于当前 token 必须重装的 static stream bank。
    pub resident_once: bool,
    pub layer: Option<u8>,
    pub bank: Option<usize>,
}

pub struct S14Position0PagedWeightArena {
    plan: S14Position0PagedArenaPlan,
    workspace: GpuBuffer,
    workspace_layout: S14Position0WorkspaceLayout,
    resident_small: GpuBuffer,
    routed: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    head_chunks: [GpuBuffer; S14_POSITION0_ROLLING_BANKS],
    static_stream: [GpuBuffer; S14_POSITION0_STATIC_STREAM_BANKS],
    /// 与 physical.static_layers 同索引；`None` 表示该层必须流式上传。
    static_resident: Vec<Option<GpuBuffer>>,
    optional_stop: Option<String>,
    allocated_essential_bytes: u64,
    allocated_static_resident_bytes: u64,
    allocated_device_bytes: u64,
    /// 只由 verified mmap/SHA 上传器在同步 transfer 完成后发布。resident 层逐层置位；
    /// streamed bank 保存最后一次真实装入的层 identity，禁止 consumer 把空页或旧页当当前层。
    static_readiness: Mutex<S14Position0StaticReadiness>,
}

#[derive(Debug)]
struct S14Position0StaticReadiness {
    resident_layers: Vec<bool>,
    streamed_banks: [Option<u8>; S14_POSITION0_STATIC_STREAM_BANKS],
}

impl S14Position0PagedWeightArena {
    /// 先保证完整 token 所需的固定缓冲，再把剩余可分配显存用于静态层缓存。
    /// `static_cache_limit_bytes` 只限制可选静态缓存，不影响数值完整性。
    pub fn new(
        ctx: &VulkanContext,
        weight_plan: &S14Position0HybridWeightPlan,
        static_cache_limit_bytes: Option<u64>,
    ) -> Result<Self> {
        let plan = S14Position0PagedArenaPlan::build(weight_plan)?;
        let workspace_layout = S14Position0WorkspaceLayout::build(plan.workspace_bytes)
            .context("build position0 workspace sub-range layout")?;
        if plan.essential_device_bytes > ctx.vram_size() {
            bail!(
                "position0 paged essential arena exceeds VRAM heap: essential={} vram={}",
                plan.essential_device_bytes,
                ctx.vram_size()
            );
        }

        let mut pending = PendingBuffers::new(ctx);
        // 顺序很重要：workspace 和所有流式 bank 是完整前向的必要条件，静态缓存不是。
        pending
            .allocate(plan.workspace_bytes)
            .context("allocate position0 whole-token workspace")?;
        pending
            .allocate(plan.physical.resident_small.requested_bytes)
            .context("allocate position0 resident-small")?;
        for bank in 0..S14_POSITION0_ROLLING_BANKS {
            pending
                .allocate(plan.physical.routed_bank_bytes)
                .with_context(|| format!("allocate position0 routed bank {bank}"))?;
        }
        for bank in 0..S14_POSITION0_ROLLING_BANKS {
            pending
                .allocate(plan.physical.head_chunk_bytes)
                .with_context(|| format!("allocate position0 head bank {bank}"))?;
        }
        for bank in 0..S14_POSITION0_STATIC_STREAM_BANKS {
            pending
                .allocate(plan.static_stream_bank_bytes)
                .with_context(|| format!("allocate position0 static stream bank {bank}"))?;
        }
        if pending.allocated_bytes()? != plan.essential_device_bytes {
            bail!("position0 paged essential allocation ledger drift");
        }

        let essential = pending.finish();
        let mut essential = essential.into_iter();
        let workspace = essential.next().expect("workspace");
        let resident_small = essential.next().expect("resident-small");
        let routed = [
            essential.next().expect("routed0"),
            essential.next().expect("routed1"),
        ];
        let head_chunks = [
            essential.next().expect("head0"),
            essential.next().expect("head1"),
        ];
        let static_stream = [
            essential.next().expect("static-stream0"),
            essential.next().expect("static-stream1"),
        ];
        debug_assert!(essential.next().is_none());

        let mut static_resident = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
        let mut static_bytes = 0u64;
        let mut optional_stop = None;
        for layer in &plan.physical.static_layers {
            if let Some(limit) = static_cache_limit_bytes {
                if static_bytes
                    .checked_add(layer.requested_bytes)
                    .ok_or_else(|| anyhow!("position0 static cache bytes overflow"))?
                    > limit
                {
                    optional_stop = Some(format!(
                        "static cache limit reached before L{}: limit={limit}",
                        layer.layer
                    ));
                    break;
                }
            }
            match make_vram(ctx, layer.requested_bytes) {
                Ok(buffer) => {
                    static_bytes = static_bytes
                        .checked_add(buffer.size())
                        .ok_or_else(|| anyhow!("position0 static resident bytes overflow"))?;
                    static_resident.push(Some(buffer));
                }
                Err(error) => {
                    optional_stop = Some(format!(
                        "static cache allocation stopped before L{} ({} bytes): {error:#}",
                        layer.layer, layer.requested_bytes
                    ));
                    break;
                }
            }
        }
        static_resident.resize_with(FULL_DEPTH_LAYERS.len(), || None);
        let allocated_device_bytes = plan
            .essential_device_bytes
            .checked_add(static_bytes)
            .ok_or_else(|| anyhow!("position0 paged allocated bytes overflow"))?;

        Ok(Self {
            plan,
            workspace,
            workspace_layout,
            resident_small,
            routed,
            head_chunks,
            static_stream,
            static_resident,
            optional_stop,
            allocated_essential_bytes: allocated_device_bytes - static_bytes,
            allocated_static_resident_bytes: static_bytes,
            allocated_device_bytes,
            static_readiness: Mutex::new(S14Position0StaticReadiness {
                resident_layers: vec![false; FULL_DEPTH_LAYERS.len()],
                streamed_banks: [None; S14_POSITION0_STATIC_STREAM_BANKS],
            }),
        })
    }

    pub fn plan(&self) -> &S14Position0PagedArenaPlan {
        &self.plan
    }

    pub fn workspace(&self) -> &GpuBuffer {
        &self.workspace
    }

    pub fn workspace_layout(&self) -> &S14Position0WorkspaceLayout {
        &self.workspace_layout
    }

    pub fn resident_small(&self) -> &GpuBuffer {
        &self.resident_small
    }

    pub fn routed(&self, bank: usize) -> Result<&GpuBuffer> {
        self.routed
            .get(bank)
            .ok_or_else(|| anyhow!("invalid position0 routed bank {bank}"))
    }

    pub fn head_chunk(&self, bank: usize) -> Result<&GpuBuffer> {
        self.head_chunks
            .get(bank)
            .ok_or_else(|| anyhow!("invalid position0 head bank {bank}"))
    }

    pub fn static_layer(&self, layer: u8) -> Result<S14Position0StaticLayerBinding<'_>> {
        let index = FULL_DEPTH_LAYERS
            .iter()
            .position(|&expected| expected == layer)
            .ok_or_else(|| anyhow!("invalid position0 static layer L{layer}"))?;
        let layout = &self.plan.physical.static_layers[index];
        Ok(match self.static_resident[index].as_ref() {
            Some(buffer) => S14Position0StaticLayerBinding::Resident { buffer, layout },
            None => {
                let bank = index % S14_POSITION0_STATIC_STREAM_BANKS;
                S14Position0StaticLayerBinding::Streamed {
                    bank,
                    buffer: &self.static_stream[bank],
                    layout,
                }
            }
        })
    }

    /// production consumer 的 fail-closed 入口。只有 verified payload store 已完成
    /// proof/SHA/mmap 且同步 transfer 已 wait 成功后，上传器才会发布 readiness。
    pub fn ready_static_layer(&self, layer: u8) -> Result<S14Position0StaticLayerBinding<'_>> {
        let binding = self.static_layer(layer)?;
        let index = FULL_DEPTH_LAYERS
            .iter()
            .position(|&expected| expected == layer)
            .ok_or_else(|| anyhow!("invalid position0 static layer L{layer}"))?;
        let readiness = self
            .static_readiness
            .lock()
            .map_err(|_| anyhow!("position0 static readiness poisoned"))?;
        let ready = match &binding {
            S14Position0StaticLayerBinding::Resident { .. } => readiness.resident_layers[index],
            S14Position0StaticLayerBinding::Streamed { bank, .. } => {
                readiness.streamed_banks[*bank] == Some(layer)
            }
        };
        if !ready {
            bail!("position0 static L{layer} 尚无 verified proof/SHA/mmap/upload readiness");
        }
        Ok(binding)
    }

    /// 只能由 crate 内正式上传器在同步 copy+wait 成功后调用。
    pub(crate) fn publish_static_layer_ready(
        &self,
        layer: u8,
        resident_hit: bool,
        bank: Option<usize>,
        assets: usize,
        uploaded_bytes: u64,
    ) -> Result<()> {
        let index = FULL_DEPTH_LAYERS
            .iter()
            .position(|&expected| expected == layer)
            .ok_or_else(|| anyhow!("invalid position0 static readiness L{layer}"))?;
        let layout = &self.plan.physical.static_layers[index];
        let expected_payload_bytes = layout.assets.iter().try_fold(0u64, |sum, asset| {
            sum.checked_add(asset.bytes)
                .ok_or_else(|| anyhow!("position0 static L{layer} payload bytes overflow"))
        })?;
        if assets != layout.assets.len() {
            bail!("position0 static L{layer} readiness asset count 漂移");
        }
        let mut readiness = self
            .static_readiness
            .lock()
            .map_err(|_| anyhow!("position0 static readiness poisoned"))?;
        match self.static_layer(layer)? {
            S14Position0StaticLayerBinding::Resident { .. } => {
                if !resident_hit || bank.is_some() || uploaded_bytes != 0 {
                    bail!("position0 resident static L{layer} readiness receipt 漂移");
                }
                readiness.resident_layers[index] = true;
            }
            S14Position0StaticLayerBinding::Streamed {
                bank: expected_bank,
                ..
            } => {
                if resident_hit
                    || bank != Some(expected_bank)
                    || uploaded_bytes != expected_payload_bytes
                {
                    bail!("position0 streamed static L{layer} readiness receipt 漂移");
                }
                readiness.streamed_banks[expected_bank] = Some(layer);
            }
        }
        Ok(())
    }

    pub fn static_stream_bank(&self, bank: usize) -> Result<&GpuBuffer> {
        self.static_stream
            .get(bank)
            .ok_or_else(|| anyhow!("invalid position0 static stream bank {bank}"))
    }

    /// 把 hybrid 逻辑 placement 映射到真实分段 buffer。上传器必须使用这个
    /// destination offset，禁止继续把全局 resident offset 写进局部 layer buffer。
    pub fn static_asset(&self, tensor: &str) -> Result<S14Position0StaticAssetBinding<'_>> {
        if let Some(asset) = self
            .plan
            .physical
            .resident_small
            .assets
            .iter()
            .find(|asset| asset.tensor == tensor)
        {
            return Ok(S14Position0StaticAssetBinding {
                buffer: &self.resident_small,
                destination_offset: asset.local_offset,
                resident_once: true,
                layer: None,
                bank: None,
            });
        }
        for (index, layout) in self.plan.physical.static_layers.iter().enumerate() {
            let Some(asset) = layout.assets.iter().find(|asset| asset.tensor == tensor) else {
                continue;
            };
            return Ok(match self.static_resident[index].as_ref() {
                Some(buffer) => S14Position0StaticAssetBinding {
                    buffer,
                    destination_offset: asset.local_offset,
                    resident_once: true,
                    layer: Some(layout.layer),
                    bank: None,
                },
                None => {
                    let bank = index % S14_POSITION0_STATIC_STREAM_BANKS;
                    S14Position0StaticAssetBinding {
                        buffer: &self.static_stream[bank],
                        destination_offset: asset.local_offset,
                        resident_once: false,
                        layer: Some(layout.layer),
                        bank: Some(bank),
                    }
                }
            });
        }
        bail!("position0 static tensor 不在物理布局: {tensor}")
    }

    pub fn resident_static_layers(&self) -> usize {
        self.static_resident
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }

    pub fn streamed_static_layers(&self) -> usize {
        FULL_DEPTH_LAYERS.len() - self.resident_static_layers()
    }

    pub fn recurring_static_upload_bytes(&self) -> u64 {
        self.plan
            .physical
            .static_layers
            .iter()
            .zip(&self.static_resident)
            .filter_map(|(layout, resident)| resident.is_none().then_some(layout.requested_bytes))
            .sum()
    }

    pub fn optional_stop(&self) -> Option<&str> {
        self.optional_stop.as_deref()
    }

    pub fn allocated_essential_bytes(&self) -> u64 {
        self.allocated_essential_bytes
    }

    pub fn allocated_static_resident_bytes(&self) -> u64 {
        self.allocated_static_resident_bytes
    }

    pub fn allocated_device_bytes(&self) -> u64 {
        self.allocated_device_bytes
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        for buffer in self.static_resident.iter().flatten() {
            buffer.destroy(ctx);
        }
        for buffer in &self.static_stream {
            buffer.destroy(ctx);
        }
        for buffer in &self.head_chunks {
            buffer.destroy(ctx);
        }
        for buffer in &self.routed {
            buffer.destroy(ctx);
        }
        self.resident_small.destroy(ctx);
        self.workspace.destroy(ctx);
    }
}

struct PendingBuffers<'a> {
    ctx: &'a VulkanContext,
    buffers: Vec<GpuBuffer>,
}

impl<'a> PendingBuffers<'a> {
    fn new(ctx: &'a VulkanContext) -> Self {
        Self {
            ctx,
            buffers: Vec::with_capacity(8),
        }
    }

    fn allocate(&mut self, bytes: u64) -> Result<()> {
        self.buffers.push(make_vram(self.ctx, bytes)?);
        Ok(())
    }

    fn allocated_bytes(&self) -> Result<u64> {
        self.buffers.iter().try_fold(0u64, |sum, buffer| {
            sum.checked_add(buffer.size())
                .ok_or_else(|| anyhow!("position0 pending bytes overflow"))
        })
    }

    fn finish(mut self) -> Vec<GpuBuffer> {
        std::mem::take(&mut self.buffers)
    }
}

impl Drop for PendingBuffers<'_> {
    fn drop(&mut self) {
        for buffer in &self.buffers {
            buffer.destroy(self.ctx);
        }
    }
}

fn make_vram(ctx: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    if bytes == 0 {
        bail!("position0 paged arena refuses zero-byte allocation");
    }
    let usage = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;
    if ctx.qf_transfer == ctx.qf_graphics {
        GpuBuffer::new_vram(ctx, bytes, usage)
    } else {
        GpuBuffer::new_vram_shared(ctx, bytes, usage, &[ctx.qf_transfer, ctx.qf_graphics])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polaris_s14_runner::Position0WholeTokenManifest;
    use std::path::PathBuf;

    fn plan() -> S14Position0HybridWeightPlan {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        let manifest = Position0WholeTokenManifest::load(&path).unwrap();
        S14Position0HybridWeightPlan::build(&manifest).unwrap()
    }

    #[test]
    fn paged_essential_plan_preserves_full_depth_with_bounded_vram() {
        let weight_plan = plan();
        let paged = S14Position0PagedArenaPlan::build(&weight_plan).unwrap();
        assert_eq!(paged.physical.static_layers.len(), 43);
        assert!(paged.static_stream_bank_bytes < 192 * 1024 * 1024);
        assert_eq!(paged.workspace_bytes, 512 * 1024 * 1024);
        assert!(paged.essential_device_bytes < 1200 * 1024 * 1024);
        assert!(
            paged.essential_device_bytes + paged.physical.static_requested_bytes
                > weight_plan.device_weight_bytes
        );
    }

    #[test]
    fn paged_plan_keeps_two_banks_for_every_streamed_class() {
        let paged = S14Position0PagedArenaPlan::build(&plan()).unwrap();
        assert_eq!(S14_POSITION0_STATIC_STREAM_BANKS, 2);
        assert_eq!(S14_POSITION0_ROLLING_BANKS, 2);
        assert_eq!(
            paged.static_stream_device_bytes,
            paged.static_stream_bank_bytes * 2
        );
        assert!(paged.physical.routed_bank_bytes > 0);
        assert!(paged.physical.head_chunk_bytes > 0);
    }
}
