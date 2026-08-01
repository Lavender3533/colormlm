//! 分层专家缓存:维护 (ExpertId → Tier) 状态,支持容量约束 + LRU 淘汰。
//!
//! 这个模块只管"账本",不真正搬运权重数据。具体加载/卸载由调度器和后端实现。
//!
//! 设计要点:
//! - 每层级独立配额(VRAM 8GB / RAM 24GB 等)
//! - 用稠密索引 `[layer * n_experts + expert]` 存状态,O(1) 查询
//! - LRU 用全局时钟(每次 access += 1),淘汰时找最小 last_access

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use smallvec::SmallVec;

use crate::tier::{ExpertId, Tier};

/// 一次缓存事件,scheduler 可以基于事件触发后端动作
#[derive(Clone, Debug)]
pub enum CacheEvent {
    /// 请求把专家 promote 到目标 tier(后端要执行实际加载)
    Promote { expert: ExpertId, from: Tier, to: Tier },
    /// 请求把专家 demote 到目标 tier(后端要执行实际淘汰)
    Demote { expert: ExpertId, from: Tier, to: Tier },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub n_in_vram: u32,
    pub n_in_ram: u32,
    pub n_in_ssd: u32,
    pub n_in_hdd: u32,
    pub n_not_loaded: u32,
    pub vram_capacity: u32,
    pub ram_capacity: u32,
    pub access_clock: u64,
    pub total_accesses: u64,
    pub vram_hits: u64,
    pub ram_hits: u64,
    pub misses: u64,
}

#[derive(Clone, Copy)]
struct Slot {
    tier: Tier,
    last_access: u64,
}

impl Default for Slot {
    fn default() -> Self {
        Self { tier: Tier::NotLoaded, last_access: 0 }
    }
}

pub struct ExpertCache {
    n_layers: u16,
    n_experts_per_layer: u16,
    slots: Mutex<Box<[Slot]>>,
    clock: AtomicU64,
    vram_capacity: u32,
    ram_capacity: u32,
    stats: Mutex<CacheStats>,
}

impl ExpertCache {
    /// 创建。`vram_capacity` / `ram_capacity` 单位是"专家数"。
    pub fn new(n_layers: u16, n_experts_per_layer: u16, vram_capacity: u32, ram_capacity: u32) -> Self {
        let total = n_layers as usize * n_experts_per_layer as usize;
        Self {
            n_layers,
            n_experts_per_layer,
            slots: Mutex::new(vec![Slot::default(); total].into_boxed_slice()),
            clock: AtomicU64::new(0),
            vram_capacity,
            ram_capacity,
            stats: Mutex::new(CacheStats {
                vram_capacity,
                ram_capacity,
                ..Default::default()
            }),
        }
    }

    pub fn n_layers(&self) -> u16 { self.n_layers }
    pub fn n_experts_per_layer(&self) -> u16 { self.n_experts_per_layer }

    fn idx(&self, e: ExpertId) -> usize {
        e.dense_index(self.n_experts_per_layer)
    }

    /// 查询专家当前所在层级
    pub fn locate(&self, expert: ExpertId) -> Tier {
        self.slots.lock()[self.idx(expert)].tier
    }

    /// 标记访问(更新 LRU 时钟)。返回该专家此次访问前所在的 tier。
    /// 调用方应在每次实际使用专家时调用,以驱动 LRU。
    pub fn touch(&self, expert: ExpertId) -> Tier {
        let now = self.clock.fetch_add(1, Ordering::Relaxed) + 1;
        let mut slots = self.slots.lock();
        let slot = &mut slots[self.idx(expert)];
        let prev_tier = slot.tier;
        slot.last_access = now;

        let mut stats = self.stats.lock();
        stats.access_clock = now;
        stats.total_accesses += 1;
        match prev_tier {
            Tier::Vram => stats.vram_hits += 1,
            Tier::Ram  => stats.ram_hits += 1,
            _          => stats.misses += 1,
        }
        prev_tier
    }

    /// 把一组专家请求 promote 到 VRAM。
    /// 容量不够时,从 VRAM 中按 LRU 淘汰最冷的(降到 RAM)。
    /// 返回触发的 CacheEvent 列表(scheduler 用它派遣后端动作)。
    pub fn request_to_vram(&self, experts: &[ExpertId]) -> SmallVec<[CacheEvent; 32]> {
        let mut events = SmallVec::new();
        let mut slots = self.slots.lock();

        for &e in experts {
            let idx = self.idx(e);
            if slots[idx].tier == Tier::Vram { continue; }

            // 检查 VRAM 容量
            let n_in_vram = slots.iter().filter(|s| s.tier == Tier::Vram).count() as u32;
            if n_in_vram >= self.vram_capacity {
                // 找 VRAM 中 last_access 最小的,降到 RAM
                let evict_idx = slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.tier == Tier::Vram)
                    .min_by_key(|(_, s)| s.last_access)
                    .map(|(i, _)| i);
                if let Some(ei) = evict_idx {
                    let from = slots[ei].tier;
                    slots[ei].tier = Tier::Ram;
                    let evicted = self.id_from_idx(ei);
                    events.push(CacheEvent::Demote { expert: evicted, from, to: Tier::Ram });
                }
            }

            // 然后做 promote
            let from = slots[idx].tier;
            slots[idx].tier = Tier::Vram;
            // 时钟 +1,标记为最新
            let now = self.clock.fetch_add(1, Ordering::Relaxed) + 1;
            slots[idx].last_access = now;
            events.push(CacheEvent::Promote { expert: e, from, to: Tier::Vram });
        }

        // 同步 stats
        self.refresh_stats(&slots);
        events
    }

    /// 强制把专家直接置位到指定 tier(初始化 / 调试用,不触发淘汰)
    pub fn set_tier(&self, expert: ExpertId, tier: Tier) {
        let mut slots = self.slots.lock();
        slots[self.idx(expert)].tier = tier;
        self.refresh_stats(&slots);
    }

    /// 当前快照统计
    pub fn stats(&self) -> CacheStats {
        let slots = self.slots.lock();
        self.refresh_stats(&slots);
        *self.stats.lock()
    }

    fn refresh_stats(&self, slots: &[Slot]) {
        let mut stats = self.stats.lock();
        stats.n_in_vram = 0;
        stats.n_in_ram = 0;
        stats.n_in_ssd = 0;
        stats.n_in_hdd = 0;
        stats.n_not_loaded = 0;
        for s in slots {
            match s.tier {
                Tier::Vram => stats.n_in_vram += 1,
                Tier::Ram => stats.n_in_ram += 1,
                Tier::Ssd => stats.n_in_ssd += 1,
                Tier::Hdd => stats.n_in_hdd += 1,
                Tier::NotLoaded => stats.n_not_loaded += 1,
            }
        }
    }

    fn id_from_idx(&self, idx: usize) -> ExpertId {
        let n = self.n_experts_per_layer as usize;
        ExpertId {
            layer: (idx / n) as u16,
            expert: (idx % n) as u16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> ExpertCache {
        ExpertCache::new(4, 8, /*vram*/ 4, /*ram*/ 16)
    }

    #[test]
    fn initial_state_all_not_loaded() {
        let c = cache();
        for layer in 0..4u16 {
            for expert in 0..8u16 {
                assert_eq!(c.locate(ExpertId::new(layer, expert)), Tier::NotLoaded);
            }
        }
    }

    #[test]
    fn promote_to_vram() {
        let c = cache();
        let e = ExpertId::new(0, 0);
        c.request_to_vram(&[e]);
        assert_eq!(c.locate(e), Tier::Vram);
        assert_eq!(c.stats().n_in_vram, 1);
    }

    #[test]
    fn lru_eviction_when_vram_full() {
        let c = cache();
        let experts: Vec<_> = (0..5u16).map(|i| ExpertId::new(0, i)).collect();
        // VRAM 容量 = 4
        c.request_to_vram(&experts[0..4]);
        assert_eq!(c.stats().n_in_vram, 4);

        // 第 5 个会触发 LRU 淘汰(experts[0] 被淘到 RAM)
        let events = c.request_to_vram(&experts[4..5]);
        assert_eq!(c.stats().n_in_vram, 4);
        assert_eq!(c.locate(experts[0]), Tier::Ram);  // 最先放的最先被淘
        assert_eq!(c.locate(experts[4]), Tier::Vram);
        // 应该有 1 个 demote + 1 个 promote
        let n_demote = events.iter().filter(|e| matches!(e, CacheEvent::Demote { .. })).count();
        let n_promote = events.iter().filter(|e| matches!(e, CacheEvent::Promote { .. })).count();
        assert_eq!(n_demote, 1);
        assert_eq!(n_promote, 1);
    }

    #[test]
    fn touch_updates_lru() {
        let c = cache();
        let experts: Vec<_> = (0..4u16).map(|i| ExpertId::new(0, i)).collect();
        c.request_to_vram(&experts);
        // touch experts[0] → 它现在是最新的
        c.touch(experts[0]);

        // 加第 5 个,应该淘汰 experts[1] (现在最旧)
        let new_e = ExpertId::new(0, 4);
        c.request_to_vram(&[new_e]);
        assert_eq!(c.locate(experts[0]), Tier::Vram, "touched expert should stay");
        assert_eq!(c.locate(experts[1]), Tier::Ram, "oldest non-touched should be evicted");
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let c = cache();
        let e1 = ExpertId::new(0, 0);
        let e2 = ExpertId::new(0, 1);

        c.touch(e1);  // miss (not loaded)
        c.request_to_vram(&[e1]);
        c.touch(e1);  // VRAM hit
        c.set_tier(e2, Tier::Ram);
        c.touch(e2);  // RAM hit

        let s = c.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.vram_hits, 1);
        assert_eq!(s.ram_hits, 1);
    }
}
