//! Time-driven async prefetch simulation
//!
//! 精确模拟 GPU compute 和 IO prefetch 的时间重叠:
//! - GPU 按层计算,每层 T_layer 时间
//! - Predictor 在每层结束后异步发起下一层 IO
//! - IO 按 FIFO 队列串行(单 NVMe 通道),与 GPU compute 并行
//! - 当 GPU 需要的 expert 还没搬完时,GPU 等待(stall)
//!
//! 输出:真实等效 t/s,GPU stall 占比,prefetch 命中率

use predictor::{ActivationRecord, CooccurMatrix, load_matrix};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read as _;

const NVME_BPS: f64 = 1.51 * 1024.0 * 1024.0 * 1024.0;
const PCIE_BPS: f64 = 14.0 * 1024.0 * 1024.0 * 1024.0;
const GPU_TPS: f64  = 14.5;
const N_LAYERS: u16 = 48;
const N_EXPERTS: u16 = 128;
const TOTAL: u32 = N_LAYERS as u32 * N_EXPERTS as u32;

// ── Minimal cache ───────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier { Vram, Ram, Cold }

struct Cache {
    tiers: Vec<Tier>,
    access: Vec<u64>,
    clock: u64,
    vram_cap: u32,
    n_vram: u32,
    ne: u16,
}

impl Cache {
    fn new(vram_cap: u32) -> Self {
        Self {
            tiers: vec![Tier::Cold; TOTAL as usize],
            access: vec![0u64; TOTAL as usize],
            clock: 0, vram_cap, n_vram: 0,
            ne: N_EXPERTS,
        }
    }

    fn idx(&self, l: u16, e: u16) -> usize { l as usize * self.ne as usize + e as usize }

    fn tier(&self, l: u16, e: u16) -> Tier { self.tiers[self.idx(l, e)] }

    fn touch(&mut self, l: u16, e: u16) {
        self.clock += 1;
        let i = self.idx(l, e);
        self.access[i] = self.clock;
    }

    fn promote(&mut self, l: u16, e: u16) {
        let i = self.idx(l, e);
        if self.tiers[i] == Tier::Vram { return; }
        if self.n_vram >= self.vram_cap {
            let mut min_a = u64::MAX;
            let mut min_i = 0;
            for (j, (&t, &a)) in self.tiers.iter().zip(self.access.iter()).enumerate() {
                if t == Tier::Vram && a < min_a { min_a = a; min_i = j; }
            }
            self.tiers[min_i] = Tier::Ram;
            self.n_vram -= 1;
        }
        self.tiers[i] = Tier::Vram;
        self.n_vram += 1;
        self.clock += 1;
        self.access[i] = self.clock;
    }
}

// ── IO queue ────────────────────────────────────────────

struct IoQueue {
    inflight: HashMap<(u16, u16), f64>,
    ready: HashMap<(u16, u16), Tier>,
    bus_free_at: f64,
}

impl IoQueue {
    fn new() -> Self {
        Self { inflight: HashMap::new(), ready: HashMap::new(), bus_free_at: 0.0 }
    }

    fn start_async(&mut self, l: u16, e: u16, bytes: usize, bw: f64, now: f64) {
        if self.inflight.contains_key(&(l, e)) || self.ready.contains_key(&(l, e)) { return; }
        let start = now.max(self.bus_free_at);
        let dur = bytes as f64 / bw;
        let finish = start + dur;
        self.bus_free_at = finish;
        self.inflight.insert((l, e), finish);
    }

    fn advance_time(&mut self, now: f64) {
        let done: Vec<_> = self.inflight.iter()
            .filter(|(_, &f)| f <= now)
            .map(|(&k, _)| k)
            .collect();
        for key in done {
            self.inflight.remove(&key);
            self.ready.insert(key, Tier::Ram);
        }
    }

    fn check(&mut self, l: u16, e: u16, now: f64) -> PrefetchResult {
        if self.ready.remove(&(l, e)).is_some() {
            return PrefetchResult::Ready;
        }
        if let Some(&finish) = self.inflight.get(&(l, e)) {
            let wait = (finish - now).max(0.0);
            self.inflight.remove(&(l, e));
            return PrefetchResult::InFlight(wait);
        }
        PrefetchResult::NotFound
    }
}

enum PrefetchResult {
    Ready,
    InFlight(f64),
    NotFound,
}

// ── Config ──────────────────────────────────────────────

struct Cfg {
    label: &'static str,
    vram_cap: u32,
    expert_bytes: usize,
    k_prime: usize,
    lookahead: usize,
    chain: bool,
}

#[derive(Default)]
struct Result {
    n_tokens: u64,
    total_time: f64,
    stall_time: f64,
    vram_hits: u64,
    prefetch_hits: u64,
    sync_loads: u64,
    total_uses: u64,
}

// ── Simulation ──────────────────────────────────────────

fn simulate(
    cfg: &Cfg,
    records_by_token: &[Vec<&ActivationRecord>],
    matrix: &CooccurMatrix,
) -> Result {
    let mut cache = Cache::new(cfg.vram_cap);
    let mut io = IoQueue::new();
    let mut res = Result::default();

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
                            let load_time = cfg.expert_bytes as f64 / bw;
                            res.stall_time += load_time;
                            t += load_time;
                            res.sync_loads += 1;
                            cache.promote(r.layer, eid);
                        }
                    }
                }
            }

            // compute this layer
            t += layer_compute_s;

            // prefetch with lookahead (flat or chain)
            if cfg.k_prime > 0 {
                let mut hop_experts: Vec<u16> = r.experts().to_vec();
                for ahead in 1..=cfg.lookahead {
                    let target_l = r.layer as usize + ahead;
                    if target_l >= N_LAYERS as usize { break; }
                    let target_l = target_l as u16;

                    let pred_layer = if cfg.chain {
                        // chain: predict from (r.layer + ahead - 1) → target_l
                        r.layer + ahead as u16 - 1
                    } else {
                        // flat: always predict from r.layer
                        r.layer
                    };
                    if pred_layer as usize >= N_LAYERS as usize { break; }

                    let preds = matrix.predict(pred_layer, &hop_experts, cfg.k_prime);
                    for &e in preds.iter() {
                        if cache.tier(target_l, e) != Tier::Vram {
                            let bw = if cache.tier(target_l, e) == Tier::Ram {
                                PCIE_BPS
                            } else {
                                NVME_BPS
                            };
                            io.start_async(target_l, e, cfg.expert_bytes, bw, t);
                        }
                    }

                    if cfg.chain {
                        hop_experts = preds.iter().copied().collect();
                    }
                }
            }
        }
    }

    res.total_time = t;
    res
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let act_path = args.get(1).map(|s| s.as_str())
        .unwrap_or("data/activations_qwen_wiki.bin");
    let mat_path = args.get(2).map(|s| s.as_str())
        .unwrap_or("data/qwen_matrix.bin");

    let mut buf = Vec::new();
    File::open(act_path)?.read_to_end(&mut buf)?;
    let rec_size = std::mem::size_of::<ActivationRecord>();
    let n_total = buf.len() / rec_size;
    let records: &[ActivationRecord] = bytemuck::cast_slice(&buf[..n_total * rec_size]);

    // group by token, sorted by layer within each token
    let mut by_token: HashMap<u32, Vec<&ActivationRecord>> = HashMap::new();
    for r in records {
        by_token.entry(r.token_idx).or_default().push(r);
    }
    let mut tokens: Vec<u32> = by_token.keys().copied().collect();
    tokens.sort();
    for v in by_token.values_mut() {
        v.sort_by_key(|r| r.layer);
    }
    let token_records: Vec<Vec<&ActivationRecord>> =
        tokens.iter().map(|t| by_token[t].clone()).collect();

    let matrix = load_matrix(mat_path).expect("failed to load matrix");

    println!("=== Time-Driven Async Prefetch Simulation ===");
    println!("Records: {} | Tokens: {}", records.len(), tokens.len());
    println!("Per-layer compute: {:.3} ms | Per-token: {:.1} ms",
        1000.0 / GPU_TPS / N_LAYERS as f64, 1000.0 / GPU_TPS);
    println!();

    let cfgs = [
        Cfg { label: "Baseline (全GPU)", vram_cap: TOTAL, expert_bytes: 2_621_440, k_prime: 0, lookahead: 0, chain: false },
        Cfg { label: "假1T 无prefetch", vram_cap: 512, expert_bytes: 2_621_440, k_prime: 0, lookahead: 0, chain: false },
        // Q4: flat vs chain
        Cfg { label: "Q4 flat LA=1", vram_cap: 512, expert_bytes: 2_621_440, k_prime: 32, lookahead: 1, chain: false },
        Cfg { label: "Q4 flat LA=4", vram_cap: 512, expert_bytes: 2_621_440, k_prime: 32, lookahead: 4, chain: false },
        Cfg { label: "Q4 chain LA=4", vram_cap: 512, expert_bytes: 2_621_440, k_prime: 32, lookahead: 4, chain: true },
        Cfg { label: "Q4 flat LA=8", vram_cap: 512, expert_bytes: 2_621_440, k_prime: 32, lookahead: 8, chain: false },
        Cfg { label: "Q4 chain LA=8", vram_cap: 512, expert_bytes: 2_621_440, k_prime: 32, lookahead: 8, chain: true },
        // 1.58b: flat vs chain
        Cfg { label: "1.58b flat LA=1", vram_cap: 512, expert_bytes: 524_288, k_prime: 32, lookahead: 1, chain: false },
        Cfg { label: "1.58b flat LA=4", vram_cap: 512, expert_bytes: 524_288, k_prime: 32, lookahead: 4, chain: false },
        Cfg { label: "1.58b chain LA=4", vram_cap: 512, expert_bytes: 524_288, k_prime: 32, lookahead: 4, chain: true },
        Cfg { label: "1.58b flat LA=8", vram_cap: 512, expert_bytes: 524_288, k_prime: 32, lookahead: 8, chain: false },
        Cfg { label: "1.58b chain LA=8", vram_cap: 512, expert_bytes: 524_288, k_prime: 32, lookahead: 8, chain: true },
    ];

    println!("{:<25} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8}",
        "", "VRAM%", "pfHit%", "sync%", "stall%", "t/s", "vs base");
    println!("{}", "-".repeat(80));

    let baseline_tps = GPU_TPS;

    for cfg in &cfgs {
        let r = simulate(cfg, &token_records, &matrix);
        let tps = r.n_tokens as f64 / r.total_time;
        let vram_pct = r.vram_hits as f64 / r.total_uses as f64 * 100.0;
        let pf_pct = r.prefetch_hits as f64 / r.total_uses as f64 * 100.0;
        let sync_pct = r.sync_loads as f64 / r.total_uses as f64 * 100.0;
        let compute_time = r.n_tokens as f64 / GPU_TPS;
        let stall_pct = r.stall_time / r.total_time * 100.0;
        let vs_base = tps / baseline_tps * 100.0;

        println!("{:<25} {:>6.1}% {:>6.1}% {:>6.1}% {:>7.2}% {:>7.1} {:>7.1}%",
            cfg.label, vram_pct, pf_pct, sync_pct, stall_pct, tps, vs_base);
    }

    println!();
    println!("VRAM%    = 直接命中(已在显存)");
    println!("pfHit%   = 预测器提前搬好了(异步完成,0等待)");
    println!("sync%    = 没预测到,GPU 必须等(同步加载)");
    println!("stall%   = GPU 等 IO 的时间占比(越低越好)");
    println!("vs base  = 相对全 GPU baseline 的速度百分比");

    Ok(())
}
