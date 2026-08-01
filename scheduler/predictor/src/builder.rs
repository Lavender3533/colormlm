//! 矩阵累加器 — 后台从 hook 流接收激活,原子累加。
//!
//! 行为:
//! - `observe(record)` 接收一条激活,如果与同 token 上一层激活构成相邻跃迁,累加 counts
//! - 每 `decay_interval` 次 observe,所有计数 × `decay_factor`(默认 0.99)
//! - `build_snapshot()` 把当前累加状态做成只读 [`CooccurMatrix`]
//!
//! 并发模型:多生产者(收激活的线程)无锁累加 counts,单 mutex 保护 per-token 跟踪表。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use parking_lot::Mutex;
use smallvec::SmallVec;

use crate::matrix::CooccurMatrix;
use crate::record::ActivationRecord;

pub struct MatrixBuilder {
    n_layers: u16,
    n_experts: u16,
    counts: Box<[AtomicU32]>,
    row_totals: Box<[AtomicU32]>,
    global_freq: Box<[AtomicU32]>,

    pub decay_factor: f32,
    pub decay_interval: u64,
    observations_since_decay: AtomicU64,
    total_observations: AtomicU64,
    snapshot_version: AtomicU64,

    /// Per-token state: 每个 token 独立追踪上一层激活,允许 batch 乱序到达。
    /// 当 token 走到最后一层后,从表中淘汰,避免内存无界增长。
    per_token: Mutex<HashMap<u32, PerTokenState>>,
}

#[derive(Clone)]
struct PerTokenState {
    last_layer: u16,
    last_experts: SmallVec<[u16; 16]>,
}

impl MatrixBuilder {
    pub fn new(n_layers: u16, n_experts: u16) -> Self {
        let n = n_experts as usize;
        let l = n_layers as usize;
        let counts = (0..l * n * n).map(|_| AtomicU32::new(0)).collect::<Vec<_>>().into_boxed_slice();
        let row_totals = (0..l * n).map(|_| AtomicU32::new(0)).collect::<Vec<_>>().into_boxed_slice();
        let global_freq = (0..n).map(|_| AtomicU32::new(0)).collect::<Vec<_>>().into_boxed_slice();

        Self {
            n_layers,
            n_experts,
            counts,
            row_totals,
            global_freq,
            decay_factor: 0.99,
            decay_interval: 1000,
            observations_since_decay: AtomicU64::new(0),
            total_observations: AtomicU64::new(0),
            snapshot_version: AtomicU64::new(0),
            per_token: Mutex::new(HashMap::new()),
        }
    }

    pub fn n_layers(&self) -> u16 { self.n_layers }
    pub fn n_experts(&self) -> u16 { self.n_experts }
    pub fn total_observations(&self) -> u64 { self.total_observations.load(Ordering::Relaxed) }

    /// 接收一条激活记录。若同 token 已有上一层记录且层号连续,累加跃迁。
    /// 记录在层 = n_layers - 1 时自动清理 per_token 条目。
    pub fn observe(&self, record: &ActivationRecord) {
        let n = self.n_experts as usize;

        // 1) global_freq 总是更新
        for &eid in record.experts() {
            if (eid as usize) < n {
                self.global_freq[eid as usize].fetch_add(1, Ordering::Relaxed);
            }
        }

        // 2) per-token 跟踪与跃迁累加
        {
            let mut table = self.per_token.lock();

            // 取出旧状态(避免在 mutex 内长时间持有)
            let prev = table.remove(&record.token_idx);

            if let Some(prev) = prev.as_ref() {
                if prev.last_layer + 1 == record.layer
                    && (prev.last_layer as usize) < self.n_layers as usize - 1
                {
                    let prev_layer = prev.last_layer;
                    let layer_off = prev_layer as usize * n * n;
                    let next_n = record.experts().len() as u32;

                    for &prev_e in &prev.last_experts {
                        if (prev_e as usize) >= n { continue; }
                        let row_off = layer_off + prev_e as usize * n;
                        self.row_totals[prev_layer as usize * n + prev_e as usize]
                            .fetch_add(next_n, Ordering::Relaxed);
                        for &next_e in record.experts() {
                            if (next_e as usize) >= n { continue; }
                            self.counts[row_off + next_e as usize].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }

            // 3) 更新或淘汰 per-token
            if (record.layer as usize) >= self.n_layers as usize - 1 {
                // 最后一层,token 完成,不再保留状态
            } else {
                let mut next_experts = SmallVec::new();
                next_experts.extend_from_slice(record.experts());
                table.insert(record.token_idx, PerTokenState {
                    last_layer: record.layer,
                    last_experts: next_experts,
                });
            }
        }

        // 4) bookkeeping & 衰减
        let prev = self.observations_since_decay.fetch_add(1, Ordering::Relaxed);
        self.total_observations.fetch_add(1, Ordering::Relaxed);

        if prev + 1 >= self.decay_interval {
            if self.observations_since_decay
                .compare_exchange(prev + 1, 0, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                self.decay();
            }
        }
    }

    /// 当前 per-token 表中跟踪的 token 数(用于诊断)
    pub fn live_tokens(&self) -> usize {
        self.per_token.lock().len()
    }

    /// 手动清理 per-token 表(比如一次推理结束)
    pub fn clear_per_token(&self) {
        self.per_token.lock().clear();
    }

    fn decay(&self) {
        let f = self.decay_factor;
        let scale = |v: u32| -> u32 { (v as f32 * f) as u32 };

        for c in self.counts.iter() {
            let v = c.load(Ordering::Relaxed);
            c.store(scale(v), Ordering::Relaxed);
        }
        for c in self.row_totals.iter() {
            let v = c.load(Ordering::Relaxed);
            c.store(scale(v), Ordering::Relaxed);
        }
        for c in self.global_freq.iter() {
            let v = c.load(Ordering::Relaxed);
            c.store(scale(v), Ordering::Relaxed);
        }
    }

    /// 把当前累加状态拷贝出来,做成只读快照
    pub fn build_snapshot(&self) -> CooccurMatrix {
        let counts: Box<[u32]> = self.counts.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        let row_totals: Box<[u32]> = self.row_totals.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        let global_freq: Box<[u32]> = self.global_freq.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        let version = self.snapshot_version.fetch_add(1, Ordering::Relaxed) + 1;
        CooccurMatrix::from_parts(self.n_layers, self.n_experts, version, counts, row_totals, global_freq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    fn make_record(token_idx: u32, layer: u16, experts: &[u16]) -> ActivationRecord {
        let mut r = ActivationRecord::zeroed();
        r.token_idx = token_idx;
        r.layer = layer;
        r.n_experts_used = experts.len() as u8;
        for (i, &e) in experts.iter().enumerate() {
            r.expert_ids[i] = e;
            r.expert_weights[i] = 1.0 / experts.len() as f32;
        }
        r
    }

    #[test]
    fn observe_records_transition() {
        let b = MatrixBuilder::new(8, 16);
        b.observe(&make_record(0, 0, &[1, 2]));
        b.observe(&make_record(0, 1, &[3, 4]));
        let snap = b.build_snapshot();

        // counts[layer=0][prev=1][next=3] should be 1
        let n = 16;
        assert_eq!(snap.counts()[0 * n * n + 1 * n + 3], 1);
        assert_eq!(snap.counts()[0 * n * n + 1 * n + 4], 1);
        assert_eq!(snap.counts()[0 * n * n + 2 * n + 3], 1);
        assert_eq!(snap.counts()[0 * n * n + 2 * n + 4], 1);
    }

    #[test]
    fn no_transition_across_tokens() {
        let b = MatrixBuilder::new(8, 16);
        b.observe(&make_record(0, 5, &[1]));
        b.observe(&make_record(1, 0, &[2]));  // 不同 token,不应该累加跃迁
        let snap = b.build_snapshot();
        assert_eq!(snap.total_observations(), 2);
        // counts 应该全零(没有跃迁)
        assert!(snap.counts().iter().all(|&c| c == 0));
    }

    #[test]
    fn global_freq_always_updated() {
        let b = MatrixBuilder::new(8, 16);
        b.observe(&make_record(0, 0, &[5, 7, 9]));
        let snap = b.build_snapshot();
        assert_eq!(snap.global_freq()[5], 1);
        assert_eq!(snap.global_freq()[7], 1);
        assert_eq!(snap.global_freq()[9], 1);
    }

    #[test]
    fn handles_out_of_order_batched_prefill() {
        // Simulate collector's prefill order: (token0..N, layer0), (token0..N, layer1), ...
        let b = MatrixBuilder::new(4, 8);
        let n_tokens = 3;
        for layer in 0..4u16 {
            for tok in 0..n_tokens {
                let experts: Vec<u16> = vec![tok as u16, (tok + 4) as u16];
                b.observe(&make_record(tok, layer, &experts));
            }
        }
        let snap = b.build_snapshot();

        // For each token, layer 0 → layer 1 transition should be recorded.
        // Token 0: experts [0,4] → [0,4]:  counts[0][0][0]++, counts[0][0][4]++, counts[0][4][0]++, counts[0][4][4]++
        let n = 8;
        for tok in 0..n_tokens {
            let prev_e = tok as usize;
            let next_e = tok as usize;
            assert!(snap.counts()[0 * n * n + prev_e * n + next_e] >= 1,
                "missing transition for token {} layer 0→1", tok);
        }
    }

    #[test]
    fn live_tokens_evicted_at_last_layer() {
        let b = MatrixBuilder::new(4, 8);
        b.observe(&make_record(0, 0, &[1]));
        b.observe(&make_record(0, 1, &[2]));
        b.observe(&make_record(0, 2, &[3]));
        assert_eq!(b.live_tokens(), 1);
        b.observe(&make_record(0, 3, &[4]));  // last layer
        assert_eq!(b.live_tokens(), 0, "token should be evicted at last layer");
    }

    #[test]
    fn decay_reduces_counts() {
        let mut b = MatrixBuilder::new(4, 8);
        b.decay_factor = 0.5;
        b.decay_interval = 10;

        // 6 tokens × 2 records each = 12 observations, triggers one decay
        for token in 0..6 {
            b.observe(&make_record(token, 0, &[1]));
            b.observe(&make_record(token, 1, &[2]));
        }

        let snap = b.build_snapshot();
        // 6 transitions accumulated to (layer=0, prev=1, next=2),
        // decayed by 0.5, so should be ~3
        let n = 8;
        let v = snap.counts()[0 * n * n + 1 * n + 2];
        assert!(v <= 3 && v >= 2, "expected ~3 after decay, got {}", v);
    }
}
