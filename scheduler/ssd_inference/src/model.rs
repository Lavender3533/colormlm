//! Model config + tensor name catalog for Qwen3-MoE.
//!
//! Pulls hyperparameters from GGUF metadata (verified against Qwen3-30B-A3B
//! Q4_K_M Thinking 2507) and provides string builders for every tensor
//! we need to look up by name during weight loading or forward pass.

use anyhow::{anyhow, Result};
use gguf_reader::{GgufFile, MultiGgufFile};

/// All hyperparameters needed to drive a Qwen3-MoE forward pass.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,           // e.g. "qwen3moe"
    pub n_layer: u32,
    pub d_model: u32,           // = n_q_heads * head_dim for Qwen3
    pub n_q_heads: u32,
    pub n_kv_heads: u32,        // GQA: n_q_heads must be divisible by this
    pub head_dim: u32,
    pub vocab: u32,
    pub moe_intermediate: u32,  // per-expert FFN inner dim
    pub n_experts: u32,
    pub top_k: u32,             // experts selected per token
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub context_length: u32,
    pub eos_token_id: u32,
    pub bos_token_id: Option<u32>,
}

impl ModelConfig {
    /// Build from GGUF metadata. Auto-detects "qwen3moe" vs hypothetical "qwen3"
    /// prefix based on `general.architecture`.
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g.metadata_string("general.architecture")?;
        let p = &arch; // metadata key prefix
        let key = |suffix: &str| format!("{p}.{suffix}");

        let n_layer = g.metadata_u32(&key("block_count"))?;
        let d_model = g.metadata_u32(&key("embedding_length"))?;
        let n_q_heads = g.metadata_u32(&key("attention.head_count"))?;
        let n_kv_heads = g.metadata_u32(&key("attention.head_count_kv"))?;
        // key_length / value_length are head_dim. Prefer key_length; fall back to d/h.
        let head_dim = g.metadata_u32(&key("attention.key_length"))
            .unwrap_or(d_model / n_q_heads);
        let rms_eps = g.metadata_f32(&key("attention.layer_norm_rms_epsilon"))?;
        let rope_theta = g.metadata_f32(&key("rope.freq_base"))?;
        let context_length = g.metadata_u32(&key("context_length"))?;

        // MoE-specific (only present on qwen3moe arch).
        let n_experts = g.metadata_u32(&key("expert_count"))?;
        let top_k = g.metadata_u32(&key("expert_used_count"))?;
        let moe_intermediate = g.metadata_u32(&key("expert_feed_forward_length"))?;

        // Vocab from tokenizer array length.
        let vocab = g.metadata_value("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok().map(|a| a.len() as u32))
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens"))?;

        let eos_token_id = g.metadata_u32("tokenizer.ggml.eos_token_id")?;
        let bos_token_id = g.metadata_u32("tokenizer.ggml.bos_token_id").ok();

        Ok(Self {
            arch, n_layer, d_model, n_q_heads, n_kv_heads, head_dim, vocab,
            moe_intermediate, n_experts, top_k, rope_theta, rms_eps,
            context_length, eos_token_id, bos_token_id,
        })
    }

    /// Bytes per fp32 hidden state for a given token count.
    pub fn hidden_bytes(&self, n_tok: u32) -> u64 {
        n_tok as u64 * self.d_model as u64 * 4
    }

    /// Sanity check: GQA group size must be a positive integer.
    pub fn gqa_group(&self) -> u32 {
        debug_assert!(self.n_q_heads % self.n_kv_heads == 0,
            "n_q_heads {} not divisible by n_kv_heads {}",
            self.n_q_heads, self.n_kv_heads);
        self.n_q_heads / self.n_kv_heads
    }

    pub fn from_multi_gguf(mg: &MultiGgufFile) -> Result<Self> {
        let arch = mg.metadata_string("general.architecture")?;
        let p = &arch;
        let key = |suffix: &str| format!("{p}.{suffix}");

        let n_layer = mg.metadata_u32(&key("block_count"))?;
        let d_model = mg.metadata_u32(&key("embedding_length"))?;
        let n_q_heads = mg.metadata_u32(&key("attention.head_count"))?;
        let n_kv_heads = mg.metadata_u32(&key("attention.head_count_kv"))?;
        let head_dim = mg.metadata_u32(&key("attention.key_length"))
            .unwrap_or(d_model / n_q_heads);
        let rms_eps = mg.metadata_f32(&key("attention.layer_norm_rms_epsilon"))?;
        let rope_theta = mg.metadata_f32(&key("rope.freq_base"))?;
        let context_length = mg.metadata_u32(&key("context_length"))?;
        let n_experts = mg.metadata_u32(&key("expert_count"))?;
        let top_k = mg.metadata_u32(&key("expert_used_count"))?;
        let moe_intermediate = mg.metadata_u32(&key("expert_feed_forward_length"))?;

        let vocab = mg.metadata_value("tokenizer.ggml.tokens")
            .and_then(|v| v.to_vec().ok().map(|a| a.len() as u32))
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens"))?;

        let eos_token_id = mg.metadata_u32("tokenizer.ggml.eos_token_id")?;
        let bos_token_id = mg.metadata_u32("tokenizer.ggml.bos_token_id").ok();

        Ok(Self {
            arch, n_layer, d_model, n_q_heads, n_kv_heads, head_dim, vocab,
            moe_intermediate, n_experts, top_k, rope_theta, rms_eps,
            context_length, eos_token_id, bos_token_id,
        })
    }
}

/// Static tensor-name builder. Names match Qwen3-MoE GGUF emitted by llama.cpp's
/// converter (verified against Qwen3-30B-A3B Q4_K_M).
pub struct TensorNames;

impl TensorNames {
    pub const EMBED: &'static str       = "token_embd.weight";
    pub const OUT_NORM: &'static str    = "output_norm.weight";
    pub const LM_HEAD: &'static str     = "output.weight";

    pub fn attn_norm(l: u32)   -> String { format!("blk.{l}.attn_norm.weight") }
    pub fn attn_q(l: u32)      -> String { format!("blk.{l}.attn_q.weight") }
    pub fn attn_k(l: u32)      -> String { format!("blk.{l}.attn_k.weight") }
    pub fn attn_v(l: u32)      -> String { format!("blk.{l}.attn_v.weight") }
    pub fn attn_o(l: u32)      -> String { format!("blk.{l}.attn_output.weight") }
    pub fn attn_q_norm(l: u32) -> String { format!("blk.{l}.attn_q_norm.weight") }
    pub fn attn_k_norm(l: u32) -> String { format!("blk.{l}.attn_k_norm.weight") }
    pub fn ffn_norm(l: u32)    -> String { format!("blk.{l}.ffn_norm.weight") }
    pub fn router(l: u32)      -> String { format!("blk.{l}.ffn_gate_inp.weight") }
    pub fn gate_exps(l: u32)   -> String { format!("blk.{l}.ffn_gate_exps.weight") }
    pub fn up_exps(l: u32)     -> String { format!("blk.{l}.ffn_up_exps.weight") }
    pub fn down_exps(l: u32)   -> String { format!("blk.{l}.ffn_down_exps.weight") }
}
