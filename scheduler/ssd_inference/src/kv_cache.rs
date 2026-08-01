//! KV cache: one big VRAM buffer holding K and V for every layer.
//!
//! Layout (row-major):
//!   [layer][kv ∈ {0=K, 1=V}][pos][kv_head][head_dim] fp32
//!
//! Per-layer K slab is contiguous: matches `scaled_dot.comp` expectation that
//! K is `[seq_len, n_kv_heads, head_dim]`. We expose a single VkBuffer; the
//! shader binding range is set per-layer via `DescriptorBufferInfo::offset`.

use anyhow::Result;
use ash::vk;

use crate::buffer::GpuBuffer;
use crate::device::VulkanContext;
use crate::model::ModelConfig;

pub struct KvCache {
    pub buf: GpuBuffer,
    pub max_seq: u32,
    pub layer_kv_bytes: u64,    // bytes per (layer, K-or-V) slab
    pub seq_n_kv_d_bytes: u64,  // bytes per token's K (or V) within a layer = n_kv * head_dim * 4
    pub n_layer: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

impl KvCache {
    pub fn new(ctx: &VulkanContext, cfg: &ModelConfig, max_seq: u32) -> Result<Self> {
        let seq_n_kv_d_bytes = max_seq as u64 * cfg.n_kv_heads as u64 * cfg.head_dim as u64 * 4;
        let layer_kv_bytes = seq_n_kv_d_bytes;
        let total = cfg.n_layer as u64 * 2 * layer_kv_bytes;
        let buf = GpuBuffer::new_vram(ctx, total,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
        Ok(Self {
            buf, max_seq,
            layer_kv_bytes,
            seq_n_kv_d_bytes: cfg.n_kv_heads as u64 * cfg.head_dim as u64 * 4,
            n_layer: cfg.n_layer,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Byte offset of layer L's K slab from buf start.
    pub fn k_offset(&self, layer: u32) -> u64 {
        layer as u64 * 2 * self.layer_kv_bytes
    }

    /// Byte offset of layer L's V slab from buf start.
    pub fn v_offset(&self, layer: u32) -> u64 {
        self.k_offset(layer) + self.layer_kv_bytes
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        self.buf.destroy(ctx);
    }
}
