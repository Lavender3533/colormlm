//! ssd_inference — new SSD-MoE inference engine for "石子入水" architecture.
//!
//! Layered design:
//!   - `device`        — Vulkan instance/device/queues
//!   - `buffer`        — VkBuffer + VkDeviceMemory wrapper with RAII
//!   - `vram_pool`     — fixed-slot VRAM pool with LRU eviction
//!   - `expert_loader` — pipelined GGUF→staging→VRAM uploader on transfer queue
//!   - `engine`        — top-level facade tying it all together
//!
//! Status: weight-loading subsystem only. Forward pass / compute shaders /
//! KV cache / tokenizer are upcoming work.

pub mod buffer;
pub mod compute;
pub mod device;
pub mod engine;
pub mod expert_loader;
pub mod expert_reader;
pub mod forward;
pub mod ggml_bridge;
pub mod ggml_ffi;
pub mod kv_cache;
pub mod model;
pub mod pipelines;
pub mod s14_vulkan;
pub mod sampler;
pub mod simd_dot;
pub mod streaming_weights;
pub mod tokenizer;
pub mod verified_payload_cache;
pub mod vram_pool;
pub mod weights;

pub use buffer::GpuBuffer;
pub use compute::{
    ComputePipeline, DescriptorBinder, ATTN_V_SPV, DEQUANT_Q4_K_SPV, DEQUANT_Q6_K_SPV,
    EMBEDDING_SPV, MATMUL_FP32_NAIVE_SPV, MATMUL_XWT_SPV, MATVEC_WT_SPV, RESIDUAL_ADD_SPV,
    RMSNORM_SPV, ROPE_SPV, SCALED_DOT_SPV, SOFTMAX_SPV, SWIGLU_SPV, TOP_K_SPV, VECTOR_ADD_SPV,
    WEIGHTED_ADD_SPV,
};
pub use device::VulkanContext;
pub use engine::{Engine, EngineConfig, EngineStats};
pub use expert_loader::ExpertLoader;
pub use expert_reader::{ExpertReader, MultiFileExpertReader};
pub use gguf_reader::ExpertKind;
pub use model::{ModelConfig, TensorNames};
pub use tokenizer::Tok;
pub use verified_payload_cache::{VerifiedPayloadCache, VerifiedPayloadCacheStats};
pub use vram_pool::{ExpertKey, VramPool};
pub use weights::{LayerWeights, LoadedWeights, WeightLoader};
