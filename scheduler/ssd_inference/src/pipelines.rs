//! Pre-built compute pipelines for every shader the forward pass uses.
//! Built once at engine startup; per-dispatch cost is reduced to a
//! `DescriptorBinder::new` (descriptor pool + write descriptor set).

use anyhow::Result;
use crate::compute::{
    ComputePipeline,
    EMBEDDING_SPV, RMSNORM_SPV, MATMUL_XWT_SPV, MATVEC_WT_SPV,
    FUSED_Q4K_MATVEC_SPV, FUSED_Q5K_MATVEC_SPV, FUSED_Q6K_MATVEC_SPV,
    ROPE_SPV, SCALED_DOT_SPV, SOFTMAX_SPV, ATTN_V_SPV, RESIDUAL_ADD_SPV,
    SWIGLU_SPV, WEIGHTED_ADD_SPV, TOP_K_SPV, WEIGHTED_SUM_8_SPV,
    DEQUANT_Q4_K_SPV, DEQUANT_Q6_K_SPV, DEQUANT_Q2_K_SPV, DEQUANT_Q3_K_SPV,
};
use crate::device::VulkanContext;

pub struct Pipelines {
    pub embedding:   ComputePipeline,
    pub rmsnorm:     ComputePipeline,
    pub matmul:      ComputePipeline,  // Y = X @ W^T (GGUF [out, in] layout)
    pub matvec:      ComputePipeline,  // Optimized M=1 matvec, 128-thread reduce
    pub fused_q4k:   ComputePipeline,
    pub fused_q5k:   ComputePipeline,
    pub fused_q6k:   ComputePipeline,  // GPU fused Q4K dequant+matvec (zero intermediate buffer)
    pub rope:        ComputePipeline,
    pub scaled_dot:  ComputePipeline,
    pub softmax:     ComputePipeline,
    pub attn_v:      ComputePipeline,
    pub residual:    ComputePipeline,
    pub swiglu:      ComputePipeline,
    pub weighted:    ComputePipeline,
    pub topk:        ComputePipeline,
    pub dq_q4k:      ComputePipeline,
    pub dq_q6k:      ComputePipeline,
    pub dq_q2k:      ComputePipeline,
    pub dq_q3k:      ComputePipeline,
    pub wsum8:        ComputePipeline,
}

impl Pipelines {
    pub fn build(ctx: &VulkanContext) -> Result<Self> {
        // (n_storage_buffers, push_constant_bytes) per shader — must match GLSL layouts
        Ok(Self {
            embedding:  ComputePipeline::new(ctx, EMBEDDING_SPV,    2, 8)?,   // (table, y) + (token, hidden)
            rmsnorm:    ComputePipeline::new(ctx, RMSNORM_SPV,      3, 8)?,   // (x, w, y) + (hidden, eps)
            matmul:     ComputePipeline::new(ctx, MATMUL_XWT_SPV,   3, 12)?,  // (X, W, Y) + (M, N, K)
            matvec:     ComputePipeline::new(ctx, MATVEC_WT_SPV,   3, 8)?,   // (x, W, y) + (N, K)
            fused_q4k:  ComputePipeline::new(ctx, FUSED_Q4K_MATVEC_SPV, 3, 12)?,
            fused_q5k:  ComputePipeline::new(ctx, FUSED_Q5K_MATVEC_SPV, 3, 12)?,
            fused_q6k:  ComputePipeline::new(ctx, FUSED_Q6K_MATVEC_SPV, 3, 12)?,
            rope:       ComputePipeline::new(ctx, ROPE_SPV,         1, 16)?,  // (x) + (n_h, d, base, theta)
            scaled_dot: ComputePipeline::new(ctx, SCALED_DOT_SPV,   3, 28)?,  // (Q, K, scores) + 7×u32/f32
            softmax:    ComputePipeline::new(ctx, SOFTMAX_SPV,      2, 4)?,   // (x, y) + (dim)
            attn_v:     ComputePipeline::new(ctx, ATTN_V_SPV,       3, 20)?,  // (s, V, out) + 5×u32
            residual:   ComputePipeline::new(ctx, RESIDUAL_ADD_SPV, 3, 4)?,
            swiglu:     ComputePipeline::new(ctx, SWIGLU_SPV,       3, 4)?,
            weighted:   ComputePipeline::new(ctx, WEIGHTED_ADD_SPV, 2, 8)?,
            topk:       ComputePipeline::new(ctx, TOP_K_SPV,        3, 8)?,
            dq_q4k:     ComputePipeline::new(ctx, DEQUANT_Q4_K_SPV, 2, 4)?,
            dq_q6k:     ComputePipeline::new(ctx, DEQUANT_Q6_K_SPV, 2, 4)?,
            dq_q2k:     ComputePipeline::new(ctx, DEQUANT_Q2_K_SPV, 2, 4)?,
            dq_q3k:     ComputePipeline::new(ctx, DEQUANT_Q3_K_SPV, 2, 8)?,
            // wsum8: (h, d_all) + push(D, n_experts, weights[8]) = 2 buffers, 4+4+32=40 bytes push
            wsum8:      ComputePipeline::new(ctx, WEIGHTED_SUM_8_SPV, 2, 40)?,
        })
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        for p in [&self.embedding, &self.rmsnorm, &self.matmul, &self.matvec,
                  &self.fused_q4k, &self.fused_q5k, &self.fused_q6k,
                  &self.rope,
                  &self.scaled_dot, &self.softmax, &self.attn_v, &self.residual,
                  &self.swiglu, &self.weighted, &self.topk, &self.wsum8,
                  &self.dq_q4k, &self.dq_q6k, &self.dq_q2k, &self.dq_q3k] {
            p.destroy(ctx);
        }
    }
}
