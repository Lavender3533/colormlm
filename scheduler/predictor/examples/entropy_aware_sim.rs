//! Entropy-aware 层分配策略模拟
//!
//! 基于 layer_entropy_analysis 的发现:layer 0-3 几乎全广(uniq=112-127),
//! 且 top32 只覆盖 56-67%。如果把这几层的高频 expert 直接 pin 进 VRAM,
//! 应该能显著降低 sync miss。
//!
//! 对比 5 个策略:
//!   A. 均匀(等同 async_prefetch_sim 的 1.58b chain LA=4)
//!   B. Pin layer 0-3 的 top-K(高熵层吃更多 VRAM 配额)
//!   C. Pin entropy-反比例(全 48 层按 entropy 排序拿 VRAM)
//!   D. Pin 全局 top hot(对照 Session 6 hot/cold 实验)
//!   E. Hybrid:高熵层 pin top-K + 低熵层靠 predictor
//!
//! 用法:
//!   cargo run --release --example entropy_aware_sim -- ../data/activations_qwen_wiki.bin ../data/qwen_matrix.bin

use predictor::{ActivationRecord, CooccurMatrix, load_matrix};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read as _;

const NVME_BPS: f64 = 1.51 * 1024.0 * 1024.0 * 1024.0;
const PCIE_BPS: f64 = 14.0 * 1024.0 * 1024.0 * 1024.0;
const GPU_TPS: f64 = 14.5;
const N_LAYERS: u16 = 48;
const N_EXPERTS: u16 = 128;
const TOTAL: u32 = N_LAYERS as u32 * N_EXPERTS as u32;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier { Vram, Ram, Cold }

struct Cache {
    tiers: Vec<Tier>,
    access: Vec<u64>,
    pinned: Vec<bool>,
    clock: u64,
    vram_cap: u32,
    n_vram: u32,
}

impl Cache {
    fn new(vram_cap: u32) -> Self {
        Self {
            tiers: vec![Tier::Cold; TOTAL as usize],
            access: vec![0u64; TOTAL as usize],
            pinned: vec![false; TOTAL as usize],
            clock: 0, vram_cap, n_vram: 0,
        }
    }

    fn idx(l: u16, e: u16) -> usize { l as usize * N_EXPERTS as usize + e as usize }

    fn tier(&self, l: u16, e: u16) -> Tier { self.tiers[Self::idx(l, e)] }

    fn pin(&mut self, l: u16, e: u16) {
        let i = Self::idx(l, e);
        if self.tiers[i] != Tier::Vram {
            self.tiers[i] = Tier::Vram;
            self.n_vram += 1;
        }
        self.pinned[i] = true;
        self.clock += 1;
        self.access[i] = self.clock;
    }

    fn touch(&mut self, l: u16, e: u16) {
        self.clock += 1;
        let i = Self::idx(l, e);
        self.access[i] = self.clock;
    }

    fn promote(&mut self, l: u16, e: u16) {
        let i = Self::idx(l, e);
        if self.tiers[i] == Tier::Vram { return; }
        if self.n_vram >= self.vram_cap {
            // evict LRU non-pinned
            let mut min_a = u64::MAX;
            let mut min_i = usize::MAX;
            for (j, ((&t, &a), &p)) in self.tiers.iter()
                .zip(self.access.iter())
                .zip(self.pinned.iter())
                .enumerate()
            {
                if t == Tier::Vram && !p && a < min_a { min_a = a; min_i = j; }
            }
            if min_i == usize::MAX { return; } // 全部 pinned,放弃 promote
            self.tiers[min_i] = Tier::Ram;
            self.n_vram -= 1;
        }
        self.tiers[i] = Tier::Vram;
        self.n_vram += 1;
        self.clock += 1;
        self.access[i] = self.clock;
    }
}

struct IoQueue {
    inflight: HashMap<(u16, u16), f64>,
    ready: HashMap<(u16, u16), Tier>,
    bus_free_at: f64,
}

impl IoQueue {
    fn new() -> Self { Self { inflight: HashMap::new(), ready: HashMap::new(), bus_free_at: 0.0 } }

    fn start_async(&mut self, l: u16, e: u16, bytes: usize, bw: f64, now: f64) {
        if self.inflight.contains_key(&(l, e)) || self.ready.contains_key(&(l, e)) { return; }
        let start = now.max(self.bus_free_at);
        let dur = bytes as f64 / bw;
        let finish = start + dur;
        self.bus_free_at = finish;
        self.inflight.insert((l, e), finish);
    }

    fn advance_time(&mut self, now: f64) {
        let done: Vec<_> = self.inflight.iter().filter(|(_, &f)| f <= now).map(|(&k, _)| k).collect();
        for k in done {
            self.inflight.remove(&k);
            self.ready.insert(k, Tier::Ram);
        }
    }

    fn check(&mut self, l: u16, e: u16, now: f64) -> PrefetchResult {
        if self.ready.remove(&(l, e)).is_some() { return PrefetchResult::Ready; }
        if let Some(&finish) = self.inflight.get(&(l, e)) {
            let wait = (finish - now).max(0.0);
            self.inflight.remove(&(l, e));
            return PrefetchResult::InFlight(wait);
        }
        PrefetchResult::NotFound
    }
}

enum PrefetchResult { Ready, InFlight(f64), NotFound }

#[derive(Clone)]
struct PinPlan {
    /// (layer, expert) 对的 pin 列表
    pins: Vec<(u16, u16)>,
}

impl PinPlan {
    fn empty() -> Self { Self { pins: vec![] } }
}

struct Cfg {
    label: &'static str,
    vram_cap: u32,
    expert_bytes: usize,
    k_prime: usize,
    lookahead: usize,
    chain: bool,
    pin_plan: PinPlan,
}

#[derive(Default)]
struct SimResult {
    n_tokens: u64,
    total_time: f64,
    stall_time: f64,
    vram_hits: u64,
    prefetch_hits: u64,
    sync_loads: u64,
    total_uses: u64,
    pinned_count: u32,
}

fn simulate(cfg: &Cfg, records_by_token: &[Vec<&ActivationRecord>], matrix: &CooccurMatrix) -> SimResult {
    let mut cache = Cache::new(cfg.vram_cap);
    for &(l, e) in &cfg.pin_plan.pins {
        if cache.n_vram < cfg.vram_cap {
            cache.pin(l, e);
        }
    }
    let pinned_count = cache.n_vram;

    let mut io = IoQueue::new();
    let mut res = SimResult::default();
    let layer_compute_s = (1.0 / GPU_TPS) / N_LAYERS as f64;
    let mut t: f64 = 0.0;

    for token_records in records_by_token {
        res.n_tokens += 1;
        for r in token_records {
            io.advance_time(t);
            for &eid in r.experts() {
                res.total_uses += 1;
                cache.touch(r.layer, eid);
                let tier = cache.tier(r.layer, eid);
                if tier == Tier::Vram {
                    res.vram_hits += 1;
                } else {
                    match io.check(r.layer, eid, t) {
                        PrefetchResult::Ready => {
                            res.prefetch_hits += 1;
                            cache.promote(r.layer, eid);
                        }
                        PrefetchResult::InFlight(wait) => {
                            res.stall_time += wait;
                            t += wait;
                            res.prefetch_hits += 1;
                            cache.promote(r.layer, eid);
                        }
                        PrefetchResult::NotFound => {
                            let bw = if tier == Tier::Ram { PCIE_BPS } else { NVME_BPS };
                            let lt = cfg.expert_bytes as f64 / bw;
                            res.stall_time += lt;
                            t += lt;
                            res.sync_loads += 1;
                            cache.promote(r.layer, eid);
                        }
                    }
                }
            }
            t += layer_compute_s;

            if cfg.k_prime > 0 {
                let mut hop: Vec<u16> = r.experts().to_vec();
                for ahead in 1..=cfg.lookahead {
                    let target_l = r.layer as usize + ahead;
                    if target_l >= N_LAYERS as usize { break; }
                    let target_l = target_l as u16;
                    let pred_layer = if cfg.chain { r.layer + ahead as u16 - 1 } else { r.layer };
                    if pred_layer as usize >= N_LAYERS as usize { break; }
                    let preds = matrix.predict(pred_layer, &hop, cfg.k_prime);
                    for &e in preds.iter() {
                        if cache.tier(target_l, e) != Tier::Vram {
                            let bw = if cache.tier(target_l, e) == Tier::Ram { PCIE_BPS } else { NVME_BPS };
                            io.start_async(target_l, e, cfg.expert_bytes, bw, t);
                        }
                    }
                    if cfg.chain { hop = preds.iter().copied().collect(); }
                }
            }
        }
    }
    res.total_time = t;
    res.pinned_count = pinned_count;
    res
}

// ─── Pin plan builders ──────────────────────────────────

/// 按层统计 (layer, expert) → count
fn per_layer_freq(records: &[ActivationRecord]) -> Vec<Vec<u64>> {
    let mut freq = vec![vec![0u64; N_EXPERTS as usize]; N_LAYERS as usize];
    for r in records {
        for &eid in r.experts() {
            if r.layer < N_LAYERS && eid < N_EXPERTS {
                freq[r.layer as usize][eid as usize] += 1;
            }
        }
    }
    freq
}

fn per_layer_entropy(freq: &[Vec<u64>]) -> Vec<f64> {
    freq.iter().map(|row| {
        let total: u64 = row.iter().sum();
        if total == 0 { return 0.0; }
        let mut h = 0.0f64;
        for &c in row {
            if c > 0 {
                let p = c as f64 / total as f64;
                h -= p * p.log2();
            }
        }
        h
    }).collect()
}

/// Plan B: 把指定 layer 集合的 top-K expert 全 pin 进 VRAM
fn pin_top_k_in_layers(freq: &[Vec<u64>], layers: &[u16], k: usize) -> PinPlan {
    let mut pins = Vec::new();
    for &l in layers {
        let mut row: Vec<(u16, u64)> = freq[l as usize].iter().enumerate()
            .map(|(e, &c)| (e as u16, c)).collect();
        row.sort_by(|a, b| b.1.cmp(&a.1));
        for (e, c) in row.into_iter().take(k) {
            if c > 0 { pins.push((l, e)); }
        }
    }
    PinPlan { pins }
}

/// Plan C: 全 48 层按 entropy 反比例分配 VRAM 配额
/// 高 entropy → 多名额(因为 miss 概率高);低 entropy → 少名额(predictor 够用)
fn pin_by_entropy_proportional(freq: &[Vec<u64>], entropy: &[f64], total_budget: u32) -> PinPlan {
    let sum_e: f64 = entropy.iter().sum();
    if sum_e <= 0.0 { return PinPlan::empty(); }
    let mut pins = Vec::new();
    let mut consumed = 0u32;
    for (l, &h) in entropy.iter().enumerate() {
        let quota = ((h / sum_e) * total_budget as f64).round() as u32;
        let quota = quota.min(N_EXPERTS as u32).min(total_budget - consumed);
        if quota == 0 { continue; }
        let mut row: Vec<(u16, u64)> = freq[l].iter().enumerate()
            .map(|(e, &c)| (e as u16, c)).collect();
        row.sort_by(|a, b| b.1.cmp(&a.1));
        for (e, c) in row.into_iter().take(quota as usize) {
            if c > 0 {
                pins.push((l as u16, e));
                consumed += 1;
            }
        }
        if consumed >= total_budget { break; }
    }
    PinPlan { pins }
}

/// Plan D: 全局热门 expert(对照 hot/cold)
fn pin_global_top(records: &[ActivationRecord], k: usize) -> PinPlan {
    let mut freq: HashMap<(u16, u16), u64> = HashMap::new();
    for r in records {
        for &eid in r.experts() {
            *freq.entry((r.layer, eid)).or_insert(0) += 1;
        }
    }
    let mut v: Vec<((u16, u16), u64)> = freq.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    let pins: Vec<(u16, u16)> = v.into_iter().take(k).map(|(p, _)| p).collect();
    PinPlan { pins }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let act_path = args.get(1).map(|s| s.as_str()).unwrap_or("data/activations_qwen_wiki.bin");
    let mat_path = args.get(2).map(|s| s.as_str()).unwrap_or("data/qwen_matrix.bin");

    let mut buf = Vec::new();
    File::open(act_path)?.read_to_end(&mut buf)?;
    let rec_size = std::mem::size_of::<ActivationRecord>();
    let n_total = buf.len() / rec_size;
    let records: &[ActivationRecord] = bytemuck::cast_slice(&buf[..n_total * rec_size]);

    let mut by_token: HashMap<u32, Vec<&ActivationRecord>> = HashMap::new();
    for r in records {
        by_token.entry(r.token_idx).or_default().push(r);
    }
    let mut tokens: Vec<u32> = by_token.keys().copied().collect();
    tokens.sort();
    for v in by_token.values_mut() { v.sort_by_key(|r| r.layer); }
    let token_records: Vec<Vec<&ActivationRecord>> = tokens.iter().map(|t| by_token[t].clone()).collect();

    let matrix = load_matrix(mat_path).expect("failed to load matrix");

    let freq = per_layer_freq(records);
    let entropy = per_layer_entropy(&freq);

    println!("=== Entropy-Aware 层分配策略 ===");
    println!("Records: {} | Tokens: {} | VRAM cap: 512 expert slots",
        records.len(), tokens.len());
    println!("固定参数:1.58-bit (0.5 MB/expert), K'=32, lookahead=4, chain");
    println!();

    let v_cap = 512u32;
    let bytes = 524_288usize;
    let kp = 32;
    let la = 4;
    let chain = true;

    // 找出高熵 layer(top 4 + top 8 + top 12)
    let mut by_h: Vec<(usize, f64)> = entropy.iter().enumerate().map(|(i, &h)| (i, h)).collect();
    by_h.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top4_layers: Vec<u16> = by_h.iter().take(4).map(|(l, _)| *l as u16).collect();
    let top8_layers: Vec<u16> = by_h.iter().take(8).map(|(l, _)| *l as u16).collect();

    println!("高熵 4 层:{:?}  (entropy {:?})",
        top4_layers,
        top4_layers.iter().map(|&l| format!("{:.2}", entropy[l as usize])).collect::<Vec<_>>());
    println!("高熵 8 层:{:?}", top8_layers);
    println!();

    let plans: Vec<(String, Cfg)> = vec![
        ("A. 均匀 (Session6 best)".into(), Cfg {
            label: "A", vram_cap: v_cap, expert_bytes: bytes,
            k_prime: kp, lookahead: la, chain, pin_plan: PinPlan::empty()
        }),
        ("B1. Pin top4 layers × top32 (128 pins)".into(), Cfg {
            label: "B1", vram_cap: v_cap, expert_bytes: bytes,
            k_prime: kp, lookahead: la, chain,
            pin_plan: pin_top_k_in_layers(&freq, &top4_layers, 32)
        }),
        ("B2. Pin top4 layers × top64 (256 pins)".into(), Cfg {
            label: "B2", vram_cap: v_cap, expert_bytes: bytes,
            k_prime: kp, lookahead: la, chain,
            pin_plan: pin_top_k_in_layers(&freq, &top4_layers, 64)
        }),
        ("B3. Pin top4 layers × all128 (512 pins)".into(), Cfg {
            label: "B3", vram_cap: v_cap, expert_bytes: bytes,
            k_prime: kp, lookahead: la, chain,
            pin_plan: pin_top_k_in_layers(&freq, &top4_layers, 128)
        }),
        ("B4. Pin top8 layers × top32 (256 pins)".into(), Cfg {
            label: "B4", vram_cap: v_cap, expert_bytes: bytes,
            k_prime: kp, lookahead: la, chain,
            pin_plan: pin_top_k_in_layers(&freq, &top8_layers, 32)
        }),
        ("C1. Entropy-比例 256 budget".into(), Cfg {
            label: "C1", vram_cap: v_cap, expert_bytes: bytes,
            k_prime: kp, lookahead: la, chain,
            pin_plan: pin_by_entropy_proportional(&freq, &entropy, 256)
        }),
        ("C2. Entropy-比例 384 budget".into(), Cfg {
            label: "C2", vram_cap: v_cap, expert_bytes: bytes,
            k_prime: kp, lookahead: la, chain,
            pin_plan: pin_by_entropy_proportional(&freq, &entropy, 384)
        }),
        ("D. 全局 top 200 (hot/cold)".into(), Cfg {
            label: "D", vram_cap: v_cap, expert_bytes: bytes,
            k_prime: kp, lookahead: la, chain,
            pin_plan: pin_global_top(records, 200)
        }),
    ];

    println!("{:<40} {:>6} {:>7} {:>7} {:>7} {:>8} {:>7} {:>8}",
        "策略", "pin数", "VRAM%", "pfHit%", "sync%", "stall%", "t/s", "vs base");
    println!("{}", "-".repeat(100));

    let baseline_tps = GPU_TPS;
    let mut a_tps = 0.0;

    for (name, cfg) in &plans {
        let r = simulate(cfg, &token_records, &matrix);
        let tps = r.n_tokens as f64 / r.total_time;
        let vram_pct = r.vram_hits as f64 / r.total_uses as f64 * 100.0;
        let pf_pct = r.prefetch_hits as f64 / r.total_uses as f64 * 100.0;
        let sync_pct = r.sync_loads as f64 / r.total_uses as f64 * 100.0;
        let stall_pct = r.stall_time / r.total_time * 100.0;
        let vs_base = tps / baseline_tps * 100.0;
        if cfg.label == "A" { a_tps = tps; }
        let vs_a_marker = if cfg.label == "A" {
            "  <-- 基准"
        } else if tps > a_tps + 0.05 { "  ✅" }
        else if tps < a_tps - 0.05 { "  ❌" }
        else { "  ≈" };

        println!("{:<40} {:>6} {:>6.1}% {:>6.1}% {:>6.1}% {:>7.2}% {:>6.2} {:>6.1}%{}",
            name, r.pinned_count, vram_pct, pf_pct, sync_pct, stall_pct, tps, vs_base, vs_a_marker);
    }

    println!();
    println!("基准 A = Session 6 最佳(均匀,无 pin)");
    println!("✅ 表示比 A 快 ≥ 0.05 t/s");
    println!("vs base = 相对 14.5 t/s 全 GPU baseline");

    Ok(())
}
