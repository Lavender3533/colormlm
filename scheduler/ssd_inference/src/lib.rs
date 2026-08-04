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
pub mod s14_bf16_q_head_normalize;
pub mod s14_bf16_rmsnorm;
pub mod s14_bf16_to_f32;
pub mod s14_causal_block_attention_router;
pub mod s14_causal_block_base1_k4_provider;
pub mod s14_causal_block_grouped_graph;
pub mod s14_causal_block_grouped_moe_recorder;
pub mod s14_causal_block_hc_qkv_adapter;
pub mod s14_causal_block_hc_qkv_recorder;
pub mod s14_causal_block_host_candidates;
pub mod s14_causal_block_k4_input_hidden;
pub mod s14_causal_block_layer;
pub mod s14_causal_block_moe_adapter;
pub mod s14_causal_block_prefix_arena;
pub mod s14_causal_block_prefix_initializer;
pub mod s14_causal_block_prefix_producer;
pub mod s14_causal_block_prefix_state;
pub mod s14_causal_block_production_bundle;
pub mod s14_causal_block_production_evidence;
pub mod s14_causal_block_ratio4_boundary;
pub mod s14_causal_block_ratio4_state_owner;
pub mod s14_causal_block_resources;
pub mod s14_causal_block_route_prefetch;
pub mod s14_causal_block_single_layer;
pub mod s14_causal_block_terminal;
pub mod s14_causal_block_terminal_adapter;
pub mod s14_causal_block_terminal_owner;
pub mod s14_causal_block_union_materializer;
pub mod s14_causal_block_vulkan_backend;
pub mod s14_dual_helix;
pub mod s14_dual_queue_timeline;
pub mod s14_durable_checkpoint;
pub mod s14_dynamic_page_cache_readiness;
pub mod s14_dynamic_routed_packing;
pub mod s14_dynamic_routed_page_plan;
pub mod s14_e4m3_qdq;
pub mod s14_embedding_broadcast;
pub mod s14_f32_to_bf16;
pub mod s14_final_hc_head;
pub mod s14_hc_post;
pub mod s14_head_chunk_argmax;
pub mod s14_information_gain_router;
pub mod s14_input_asset_plan;
pub mod s14_local_embedding_shard;
pub mod s14_position0_attention;
pub mod s14_position0_hybrid_upload;
pub mod s14_position0_hybrid_weight_arena;
pub mod s14_position0_layer_backend;
pub mod s14_position0_layer_program;
pub mod s14_position0_mapped_assets;
pub mod s14_position0_paged_layer_bridge;
pub mod s14_position0_paged_layer_timeline;
pub mod s14_position0_paged_weight_arena;
pub mod s14_position0_rolling_upload;
pub mod s14_position0_state_writeback;
pub mod s14_position0_synchronous_layer_pager;
pub mod s14_position0_synchronous_layer_plan;
pub mod s14_position0_terminal;
pub mod s14_position0_weight_arena;
pub mod s14_position0_weight_plan;
pub mod s14_position0_whole_token;
pub mod s14_position0_workspace;
pub mod s14_position1_attention;
pub mod s14_position3_attention;
pub mod s14_ratio128_compressor_finalize;
pub mod s14_ratio4_compressor_finalize;
pub mod s14_ratio4_global_topk;
pub mod s14_ratio4_history_paging;
pub mod s14_ratio4_main_page_gather;
pub mod s14_route_postprocess;
pub mod s14_route_postprocess_gpu;
pub mod s14_route_slot_align;
pub mod s14_runtime;
pub mod s14_sparse_attention;
pub mod s14_starfold_b4_owner;
pub mod s14_starfold_cache;
pub mod s14_starfold_concrete_factory;
pub mod s14_starfold_expert_schedule;
pub mod s14_starfold_hc_bridge;
pub mod s14_starfold_k4_adapter;
pub mod s14_starfold_k4_physical_transaction;
pub mod s14_starfold_k4_rebind;
pub mod s14_starfold_k4_terminal_chain;
pub mod s14_starfold_mxfp4_compute;
pub mod s14_starfold_mxfp4_stream;
pub mod s14_starfold_mxfp4_tile;
pub mod s14_starfold_packed_l2;
pub mod s14_starfold_prefetch_pipeline;
pub mod s14_starfold_prepare_owner;
pub mod s14_starfold_production_resources;
pub mod s14_starfold_production_session;
pub mod s14_starfold_prompt_prefill;
pub mod s14_starfold_routed_executor;
pub mod s14_starfold_runtime;
pub mod s14_starfold_shared_reduce_owner;
pub mod s14_starfold_terminal_endpoint;
pub mod s14_starfold_transfer_executor;
pub mod s14_starfold_vulkan_windows;
pub mod s14_stargraph_engine;
pub mod s14_starwave_draft;
pub mod s14_starwave_history_navigator;
pub mod s14_starwave_transaction;
pub mod s14_vulkan;
pub mod s14_vulkan_arena;
pub mod s14_whole_token_device;
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
