//! MoE 专家预取调度器(scheduler-core)。
//!
//! 串联三块:
//! - [`predictor::CooccurMatrix`] 预测下一层激活专家
//! - [`expert_cache::ExpertCache`] 维护专家在哪一层存储
//! - 实际加载/卸载由后端实现(此 crate 不直接搬数据)
//!
//! 单线程主调用入口在 [`Scheduler::on_layer_complete`],由 llama.cpp callback 触发:
//! "我刚算完 layer N,激活了 \[A1..A8\] 这 8 个专家"
//! → scheduler 查矩阵预测 layer N+1 候选 → 用 cache 决定哪些要预取
//! → 返回 [`SchedulerCommand`] 列表给后端

use std::sync::Arc;

use arc_swap::ArcSwap;
use predictor::CooccurMatrix;
use expert_cache::{ExpertCache, ExpertId, CacheEvent, Tier};
use smallvec::SmallVec;

/// 调度器要后端做的动作
#[derive(Clone, Debug)]
pub enum SchedulerCommand {
    /// 异步把专家加载到 VRAM(后端实际去搬)
    PrefetchToVram { expert: ExpertId, currently_at: Tier },
    /// 异步淘汰(从 VRAM 降到 RAM)
    EvictFromVram { expert: ExpertId },
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerConfig {
    /// 过预取系数:实际激活 K 个,我们提前装 K' 个候选
    pub prefetch_k_prime: usize,
    /// 是否启用
    pub enabled: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { prefetch_k_prime: 16, enabled: true }
    }
}

pub struct Scheduler {
    matrix: ArcSwap<CooccurMatrix>,
    cache: Arc<ExpertCache>,
    config: SchedulerConfig,
}

impl Scheduler {
    pub fn new(matrix: CooccurMatrix, cache: Arc<ExpertCache>, config: SchedulerConfig) -> Self {
        assert_eq!(matrix.n_layers(), cache.n_layers(), "layer count mismatch");
        assert_eq!(matrix.n_experts(), cache.n_experts_per_layer(), "expert count mismatch");
        Self {
            matrix: ArcSwap::from_pointee(matrix),
            cache,
            config,
        }
    }

    /// 替换底层矩阵(后台 builder 训完后切换)
    pub fn swap_matrix(&self, new_matrix: CooccurMatrix) {
        self.matrix.store(Arc::new(new_matrix));
    }

    pub fn cache(&self) -> &ExpertCache { &self.cache }

    /// 当前层算完后被 llama.cpp callback 调用。
    ///
    /// 输入:
    /// - `current_layer`: 刚算完的层号
    /// - `activated_experts`: 这一层实际激活的 K 个专家 ID
    ///
    /// 行为:
    /// 1. touch 已激活专家(更新 LRU)
    /// 2. 查 matrix 预测下一层候选
    /// 3. 在 cache 里 promote 候选,产生加载/淘汰命令
    pub fn on_layer_complete(
        &self,
        current_layer: u16,
        activated_experts: &[u16],
    ) -> SmallVec<[SchedulerCommand; 32]> {
        if !self.config.enabled {
            return SmallVec::new();
        }

        // 1) 更新 LRU
        for &eid in activated_experts {
            self.cache.touch(ExpertId::new(current_layer, eid));
        }

        // 2) 预测下一层
        let next_layer = current_layer + 1;
        if next_layer >= self.cache.n_layers() {
            return SmallVec::new();  // 最后一层无下一层
        }

        let matrix = self.matrix.load();
        let candidates = matrix.predict(current_layer, activated_experts, self.config.prefetch_k_prime);

        // 3) 转成 ExpertId 并请求 promote
        let next_layer_experts: SmallVec<[ExpertId; 32]> =
            candidates.iter().map(|&e| ExpertId::new(next_layer, e)).collect();

        let events = self.cache.request_to_vram(&next_layer_experts);

        // 4) 把 cache 事件翻译成 scheduler 命令
        events.into_iter().map(|ev| match ev {
            CacheEvent::Promote { expert, from, .. } => {
                SchedulerCommand::PrefetchToVram { expert, currently_at: from }
            }
            CacheEvent::Demote { expert, .. } => {
                SchedulerCommand::EvictFromVram { expert }
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use predictor::{ActivationRecord, MatrixBuilder};
    use bytemuck::Zeroable;

    fn make_record(token: u32, layer: u16, experts: &[u16]) -> ActivationRecord {
        let mut r = ActivationRecord::zeroed();
        r.token_idx = token;
        r.layer = layer;
        r.n_experts_used = experts.len() as u8;
        for (i, &e) in experts.iter().enumerate() {
            r.expert_ids[i] = e;
            r.expert_weights[i] = 1.0 / experts.len() as f32;
        }
        r
    }

    fn warm_matrix() -> CooccurMatrix {
        let b = MatrixBuilder::new(4, 8);
        // 训 200 次让 total_observations > 100,跳出冷启动
        for token in 0..100u32 {
            b.observe(&make_record(token, 0, &[1, 2]));
            b.observe(&make_record(token, 1, &[3, 4]));
        }
        b.build_snapshot()
    }

    #[test]
    fn scheduler_prefetches_predicted_experts() {
        let matrix = warm_matrix();
        let cache = Arc::new(ExpertCache::new(4, 8, /*vram*/ 16, /*ram*/ 32));
        let mut config = SchedulerConfig::default();
        config.prefetch_k_prime = 4;
        let sched = Scheduler::new(matrix, cache.clone(), config);

        // 模拟刚算完 layer 0 with experts [1, 2]
        let cmds = sched.on_layer_complete(0, &[1, 2]);

        // 应该至少触发把 layer1 的 [3, 4] 装进 VRAM
        let prefetched: Vec<ExpertId> = cmds.iter().filter_map(|c| match c {
            SchedulerCommand::PrefetchToVram { expert, .. } => Some(*expert),
            _ => None,
        }).collect();

        assert!(prefetched.iter().any(|e| e.layer == 1 && (e.expert == 3 || e.expert == 4)),
            "expected to prefetch layer 1 expert 3 or 4, got: {:?}", prefetched);
    }

    #[test]
    fn last_layer_emits_no_commands() {
        let matrix = warm_matrix();
        let cache = Arc::new(ExpertCache::new(4, 8, 16, 32));
        let sched = Scheduler::new(matrix, cache, SchedulerConfig::default());
        let cmds = sched.on_layer_complete(3, &[1, 2]);
        assert!(cmds.is_empty(), "no next layer to prefetch");
    }
}
