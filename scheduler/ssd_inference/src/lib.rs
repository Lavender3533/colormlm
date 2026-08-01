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

pub mod device;
pub mod buffer;
pub mod vram_pool;
pub mod expert_reader;
pub mod expert_loader;
pub mod compute;
pub mod engine;
pub mod model;
pub mod tokenizer;
pub mod weights;
pub mod pipelines;
pub mod kv_cache;
pub mod sampler;
pub mod forward;
pub mod simd_dot;
pub mod ggml_ffi;
pub mod ggml_bridge;
pub mod streaming_weights;
pub mod s14_vulkan;

pub use device::VulkanContext;
pub use buffer::GpuBuffer;
pub use vram_pool::{VramPool, ExpertKey};
pub use expert_reader::{ExpertReader, MultiFileExpertReader};
pub use expert_loader::ExpertLoader;
pub use compute::{ComputePipeline, DescriptorBinder,
    VECTOR_ADD_SPV, MATMUL_FP32_NAIVE_SPV, MATMUL_XWT_SPV, MATVEC_WT_SPV,
    DEQUANT_Q4_K_SPV, DEQUANT_Q6_K_SPV, RMSNORM_SPV,
    SOFTMAX_SPV, SWIGLU_SPV, EMBEDDING_SPV, ROPE_SPV,
    SCALED_DOT_SPV, ATTN_V_SPV,
    RESIDUAL_ADD_SPV, WEIGHTED_ADD_SPV, TOP_K_SPV};
pub use engine::{Engine, EngineConfig, EngineStats};
pub use model::{ModelConfig, TensorNames};
pub use tokenizer::Tok;
pub use weights::{LoadedWeights, LayerWeights, WeightLoader};
pub use gguf_reader::ExpertKind;
