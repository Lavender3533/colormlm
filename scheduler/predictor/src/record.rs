//! 激活记录 — 从 llama.cpp router hook 写入的原始数据
//!
//! 这是 FFI 边界的固定布局,llama.cpp 的 C 代码会直接构造这个 struct
//! 写到环形缓冲区,Rust 端按 `repr(C)` 读取。
//!
//! **不要随便改字段顺序或大小**,改动需要同步更新 llama.cpp 侧的 hook 代码。

use bytemuck::{Pod, Zeroable};

/// 单个 token 在某一层的 router 激活快照。
///
/// 大小固定 112 字节(8 + 4 + 2 + 1 + 1 + 32 + 64),适合放入 ring buffer。
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ActivationRecord {
    /// 单调时间戳,纳秒(用于性能分析与排序)
    pub timestamp_ns: u64,
    /// 推理流内的 token 序号
    pub token_idx: u32,
    /// 层号(0..n_layers)
    pub layer: u16,
    /// 实际激活的专家个数,通常等于 top-k(如 8)
    pub n_experts_used: u8,
    pub _padding: u8,
    /// 激活的专家 ID,有效部分 `[..n_experts_used]`
    pub expert_ids: [u16; 16],
    /// 各专家的 router softmax 权重,与 `expert_ids` 一一对应
    pub expert_weights: [f32; 16],
}

const _: () = assert!(std::mem::size_of::<ActivationRecord>() == 112);

impl ActivationRecord {
    pub fn experts(&self) -> &[u16] {
        &self.expert_ids[..self.n_experts_used as usize]
    }

    pub fn weights(&self) -> &[f32] {
        &self.expert_weights[..self.n_experts_used as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_112() {
        assert_eq!(std::mem::size_of::<ActivationRecord>(), 112);
    }

    #[test]
    fn experts_slice_matches_n_used() {
        let mut r = ActivationRecord::zeroed();
        r.n_experts_used = 3;
        r.expert_ids[0] = 17;
        r.expert_ids[1] = 42;
        r.expert_ids[2] = 89;
        assert_eq!(r.experts(), &[17, 42, 89]);
    }
}
