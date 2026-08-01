//! Top-level Engine: ties together GGUF model, Vulkan context, VRAM pool,
//! and ExpertLoader. The current shape is "expert weight loader as a service" —
//! enough to drive prefetch experiments, not yet a full forward pass.

use crate::buffer::GpuBuffer;
use crate::device::VulkanContext;
use crate::expert_loader::ExpertLoader;
use crate::expert_reader::ExpertReader;
use crate::vram_pool::VramPool;
use anyhow::{Context, Result};
use ash::vk;
use gguf_reader::{ExpertKind, GgufFile};
use std::path::{Path, PathBuf};

pub struct EngineConfig {
    /// Number of VRAM expert slots to allocate (each slot_bytes big)
    pub vram_pool_slots: u32,
    /// Bytes per slot — must be ≥ largest single expert in the model
    pub vram_slot_bytes: u64,
    /// Pipelined upload depth (3 was the sweet spot in PoC)
    pub pipeline_depth: usize,
    /// n_experts per layer (model-dependent: Qwen3-30B-A3B = 128, MiMo-1T = 384)
    pub n_experts_total: u32,
}

impl EngineConfig {
    pub fn qwen3_30b() -> Self {
        Self {
            vram_pool_slots: 512,
            vram_slot_bytes: 1300 * 1024,
            pipeline_depth: 24,
            n_experts_total: 128,
        }
    }

    /// Qwen3-235B-A22B Q2_K_XL: 128 experts, but each expert is 2.6 MB (down).
    /// Default to 256 VRAM slots × 3 MB = 768 MB pool (well under 8 GB VRAM).
    pub fn qwen3_235b_q2() -> Self {
        Self {
            vram_pool_slots: 256,
            vram_slot_bytes: 3 * 1024 * 1024, // 3 MB covers Q2_K_XL down (2.6 MB)
            pipeline_depth: 3,
            n_experts_total: 128,
        }
    }
}

pub struct Engine {
    pub ctx: VulkanContext,
    pub gguf: GgufFile,
    pub reader: ExpertReader,
    pub vram: VramPool,
    pub loader: ExpertLoader,
    pub config: EngineConfig,
}

impl Engine {
    pub fn new<P: AsRef<Path>>(model_path: P, config: EngineConfig) -> Result<Self> {
        let model_path: PathBuf = model_path.as_ref().to_path_buf();
        let ctx = VulkanContext::init().context("vulkan init")?;
        let gguf = GgufFile::open(&model_path).context("open gguf")?;
        let reader = ExpertReader::from_gguf(&gguf, &model_path, config.n_experts_total)
            .context("build expert reader")?;
        let vram = VramPool::new(&ctx, config.vram_pool_slots, config.vram_slot_bytes)
            .context("alloc vram pool")?;
        let loader = ExpertLoader::new_with_staging_bytes(
            &ctx,
            config.pipeline_depth,
            config.n_experts_total,
            config.vram_slot_bytes,
        )
        .context("create expert loader")?;
        Ok(Self {
            ctx,
            gguf,
            reader,
            vram,
            loader,
            config,
        })
    }

    /// Get the VRAM slot for an expert, loading on demand. Synchronous: returns
    /// after the upload command has been submitted. Caller must `flush()` before
    /// using the data on a different queue.
    pub fn ensure_expert_in_vram(
        &mut self,
        layer: u32,
        kind: ExpertKind,
        expert_slot: u32,
    ) -> Result<u32> {
        self.loader.enqueue(
            &self.ctx,
            &self.reader,
            &mut self.vram,
            layer,
            kind,
            expert_slot,
        )
    }

    /// Wait for all queued uploads to finish.
    pub fn flush(&mut self) -> Result<()> {
        self.loader.wait_all(&self.ctx, &mut self.vram)
    }

    /// Load a batch of experts using parallel disk reads (rayon) + sequential VRAM transfers.
    /// Returns (n_hits, n_misses) for cache stats.
    pub fn load_experts_parallel(
        &mut self,
        experts: &[(u32, ExpertKind, u32)],
    ) -> Result<(u32, u32)> {
        use crate::vram_pool::ExpertKey;

        let mut hits = 0u32;
        let misses_count;
        let slots = self
            .loader
            .batch_enqueue(&self.ctx, &self.reader, &mut self.vram, experts)?;

        // Count hits/misses from the batch result
        misses_count = experts
            .iter()
            .enumerate()
            .filter(|(i, &(layer, kind, slot))| {
                let key = ExpertKey {
                    layer,
                    kind: kind as u8,
                    slot,
                };
                // If the slot was looked up (touched LRU) during batch, it was a hit before batch
                // Simple heuristic: check if the returned slot was loaded fresh
                false // batch_enqueue doesn't track this, counted below
            })
            .count() as u32;

        // TODO: proper hit/miss tracking from batch_enqueue
        Ok((0, experts.len() as u32))
    }

    pub fn stats(&self) -> EngineStats {
        EngineStats {
            vram_pool_loaded: self.vram.n_loaded(),
            vram_pool_capacity: self.vram.capacity(),
            vram_pool_total_bytes: self.vram.total_bytes(),
            uploads_total: self.loader.stat_loads,
            bytes_uploaded_total: self.loader.stat_bytes,
            gpu_name: self.ctx.gpu_name.clone(),
            has_dedicated_transfer: self.ctx.has_dedicated_transfer(),
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            // wait idle so it's safe to free everything
            let _ = self.ctx.device.device_wait_idle();
        }
        self.loader.destroy(&self.ctx);
        self.vram.destroy(&self.ctx);
        // ctx drops itself
    }
}

#[derive(Debug, Clone)]
pub struct EngineStats {
    pub vram_pool_loaded: u32,
    pub vram_pool_capacity: u32,
    pub vram_pool_total_bytes: u64,
    pub uploads_total: u64,
    pub bytes_uploaded_total: u64,
    pub gpu_name: String,
    pub has_dedicated_transfer: bool,
}

// Re-export for convenience
pub use gguf_reader::ExpertKind as Kind;

// Suppress dead-code warning during early development
#[allow(dead_code)]
fn _silence_unused_imports() {
    let _ = vk::DeviceMemory::null();
    let _ = std::ptr::null::<GpuBuffer>();
}
