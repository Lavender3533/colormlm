//! 共现矩阵的只读快照,推理热路径上调用。
//!
//! 数据布局(全部一维 `Vec<u32>` + 手算 offset):
//! - `counts[layer * N * N + prev * N + next]` — 跃迁计数
//! - `row_totals[layer * N + prev]` — 行总计(归一化用)
//! - `global_freq[expert]` — 跨层全局命中频率(冷启动 fallback)
//!
//! 内存占用(以 1T MiMo 为例):
//! 70 层 × 384² × 4B = 41 MB

use smallvec::SmallVec;

/// 预测候选返回值,栈上 32,极少需要堆分配
pub type Candidates = SmallVec<[u16; 32]>;

#[derive(Clone, Debug)]
pub struct CooccurMatrix {
    n_layers: u16,
    n_experts: u16,
    version: u64,
    counts: Box<[u32]>,      // 长度 = n_layers × n_experts × n_experts
    row_totals: Box<[u32]>,  // 长度 = n_layers × n_experts
    global_freq: Box<[u32]>, // 长度 = n_experts
}

impl CooccurMatrix {
    pub fn new(n_layers: u16, n_experts: u16) -> Self {
        let n = n_experts as usize;
        let l = n_layers as usize;
        Self {
            n_layers,
            n_experts,
            version: 0,
            counts: vec![0u32; l * n * n].into_boxed_slice(),
            row_totals: vec![0u32; l * n].into_boxed_slice(),
            global_freq: vec![0u32; n].into_boxed_slice(),
        }
    }

    pub fn from_parts(
        n_layers: u16,
        n_experts: u16,
        version: u64,
        counts: Box<[u32]>,
        row_totals: Box<[u32]>,
        global_freq: Box<[u32]>,
    ) -> Self {
        let n = n_experts as usize;
        let l = n_layers as usize;
        assert_eq!(counts.len(), l * n * n);
        assert_eq!(row_totals.len(), l * n);
        assert_eq!(global_freq.len(), n);
        Self { n_layers, n_experts, version, counts, row_totals, global_freq }
    }

    #[inline]
    pub fn n_layers(&self) -> u16 { self.n_layers }
    #[inline]
    pub fn n_experts(&self) -> u16 { self.n_experts }
    #[inline]
    pub fn version(&self) -> u64 { self.version }

    #[inline]
    pub fn total_observations(&self) -> u64 {
        self.global_freq.iter().map(|&v| v as u64).sum()
    }

    pub fn counts(&self) -> &[u32] { &self.counts }
    pub fn row_totals(&self) -> &[u32] { &self.row_totals }
    pub fn global_freq(&self) -> &[u32] { &self.global_freq }

    /// 预测下一层激活的候选专家(按分数排序)。
    ///
    /// `current_layer` 是当前层号,矩阵里 `counts[current_layer]` 表示
    /// "从 current_layer 到 current_layer+1 的跃迁"。
    ///
    /// 冷启动策略:
    /// - 总观测 < 100:返回前 `top_k` 个专家(全装)
    /// - 总观测 < 1000:按 global_freq 取 top_k
    /// - 充足:用矩阵求和排序
    pub fn predict(&self, current_layer: u16, activated: &[u16], top_k: usize) -> Candidates {
        let n = self.n_experts as usize;
        let cap = top_k.min(n);

        let total = self.total_observations();
        if total < 100 {
            return (0..cap as u16).collect();
        }

        if total < 1000 {
            return self.top_k_by_global_freq(cap);
        }

        if current_layer as usize + 1 >= self.n_layers as usize {
            // 最后一层无下一层,退化到 global_freq
            return self.top_k_by_global_freq(cap);
        }

        let layer_off = current_layer as usize * n * n;
        let mut scores = vec![0u32; n];

        for &prev in activated {
            if (prev as usize) >= n { continue; }
            let row_off = layer_off + prev as usize * n;
            let row = &self.counts[row_off..row_off + n];
            for (s, &c) in scores.iter_mut().zip(row.iter()) {
                *s = s.saturating_add(c);
            }
        }

        top_k_by_score(&scores, cap)
    }

    fn top_k_by_global_freq(&self, top_k: usize) -> Candidates {
        top_k_by_score(&self.global_freq, top_k)
    }
}

/// 取分数 top_k,按分数降序、相同分数按 id 升序
fn top_k_by_score(scores: &[u32], top_k: usize) -> Candidates {
    if top_k == 0 { return Candidates::new(); }

    // 部分排序:用 BinaryHeap 维护 top_k 即可,但 N 一般 ≤ 384,直接全排序也很快
    let mut indexed: Vec<(u16, u32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i as u16, s))
        .collect();
    // 降序按 score,score 相同按 id 升序(稳定 fallback)
    indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_matrix() -> CooccurMatrix {
        CooccurMatrix::new(16, 64)
    }

    #[test]
    fn cold_start_returns_first_k() {
        let m = empty_matrix();
        let pred = m.predict(0, &[0, 1, 2], 8);
        assert_eq!(pred.len(), 8);
        assert_eq!(&pred[..], &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn predicts_from_counts_when_warm() {
        let mut counts = vec![0u32; 16 * 64 * 64];
        let mut row_totals = vec![0u32; 16 * 64];
        let mut global_freq = vec![0u32; 64];

        // 假装 layer 5 → layer 6:prev=0 强相关 next=42
        let layer_off = 5 * 64 * 64;
        let row_off = layer_off + 0 * 64;
        counts[row_off + 42] = 1000;
        counts[row_off + 17] = 500;
        counts[row_off + 9] = 100;
        row_totals[5 * 64 + 0] = 1600;

        // 让 global_freq 总和 ≥ 1000
        for i in 0..64 {
            global_freq[i] = 100;
        }

        let m = CooccurMatrix::from_parts(
            16, 64, 1,
            counts.into_boxed_slice(),
            row_totals.into_boxed_slice(),
            global_freq.into_boxed_slice(),
        );

        let pred = m.predict(5, &[0], 4);
        assert_eq!(pred[0], 42);
        assert_eq!(pred[1], 17);
        assert_eq!(pred[2], 9);
    }

    #[test]
    fn last_layer_falls_back_to_global() {
        let mut global_freq = vec![0u32; 64];
        global_freq[55] = 5000;
        global_freq[3] = 3000;
        for i in 0..64 {
            if i != 55 && i != 3 { global_freq[i] = 50; }
        }
        let m = CooccurMatrix::from_parts(
            16, 64, 1,
            vec![0u32; 16 * 64 * 64].into_boxed_slice(),
            vec![0u32; 16 * 64].into_boxed_slice(),
            global_freq.into_boxed_slice(),
        );
        let pred = m.predict(15, &[0], 2);  // 最后一层
        assert_eq!(pred[0], 55);
        assert_eq!(pred[1], 3);
    }
}
