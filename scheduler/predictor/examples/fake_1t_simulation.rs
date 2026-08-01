//! 假 1T 场景模拟 + Hot/Cold 分割实验
//!
//! 用 30B 真实激活数据,模拟 1T 比例的 swap 场景,
//! 验证预测器 + hot/cold 分割的叠加效果。
//!
//! 用法:
//!   fake_1t_simulation <activations.bin> <matrix.bin>

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

// ── Fast LRU cache with pinning ─────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier { Vram, Ram, Cold }

struct FastCache {
    tiers: Vec<Tier>,
    last_access: Vec<u64>,
    pinned: Vec<bool>,
    clock: u64,
    vram_cap: u32,
    n_in_vram: u32,
    n_experts: u16,
}

impl FastCache {
    fn new(n_layers: u16, n_experts: u16, vram_cap: u32) -> Self {
        let total = n_layers as usize * n_experts as usize;
        Self {
            tiers: vec![Tier::Cold; total],
            last_access: vec![0u64; total],
            pinned: vec![false; total],
            clock: 0, vram_cap, n_in_vram: 0,
            n_experts,
        }
    }

    fn idx(&self, layer: u16, expert: u16) -> usize {
        layer as usize * self.n_experts as usize + expert as usize
    }

    fn pin_expert(&mut self, layer: u16, expert: u16) {
        let i = self.idx(layer, expert);
        if self.tiers[i] != Tier::Vram {
            self.tiers[i] = Tier::Vram;
            self.n_in_vram += 1;
        }
        self.pinned[i] = true;
        self.clock += 1;
        self.last_access[i] = self.clock;
    }

    fn touch(&mut self, layer: u16, expert: u16) -> Tier {
        self.clock += 1;
        let i = self.idx(layer, expert);
        self.last_access[i] = self.clock;
        self.tiers[i]
    }

    fn promote_to_vram(&mut self, layer: u16, expert: u16) {
        let i = self.idx(layer, expert);
        if self.tiers[i] == Tier::Vram { return; }

        if self.n_in_vram >= self.vram_cap {
            let mut min_access = u64::MAX;
            let mut min_idx = usize::MAX;
            for (j, ((&t, &a), &p)) in self.tiers.iter()
                .zip(self.last_access.iter())
                .zip(self.pinned.iter())
                .enumerate()
            {
                if t == Tier::Vram && !p && a < min_access {
                    min_access = a;
                    min_idx = j;
                }
            }
            if min_idx == usize::MAX { return; }
            self.tiers[min_idx] = Tier::Ram;
            self.n_in_vram -= 1;
        }

        if self.tiers[i] == Tier::Ram {}
        self.tiers[i] = Tier::Vram;
        self.n_in_vram += 1;
        self.clock += 1;
        self.last_access[i] = self.clock;
    }
}

// ── Simulation config ───────────────────────────────────

struct SimConfig {
    vram_cap: u32,
    expert_bytes: usize,
    k_prime: usize,
    hot_count: usize,
}

#[derive(Default)]
struct Stats {
    vram_hits: u64,
    ram_hits: u64,
    ssd_misses: u64,
    n_tokens: u64,
}

impl Stats {
    fn total(&self) -> u64 { self.vram_hits + self.ram_hits + self.ssd_misses }
    fn vram_pct(&self) -> f64 { self.vram_hits as f64 / self.total().max(1) as f64 * 100.0 }

    fn serial_tps(&self, cfg: &SimConfig) -> f64 {
        let ram_io = self.ram_hits as f64 * cfg.expert_bytes as f64 / PCIE_BPS;
        let ssd_io = self.ssd_misses as f64 * cfg.expert_bytes as f64 / NVME_BPS;
        let compute = self.n_tokens as f64 / GPU_TPS;
        self.n_tokens as f64 / (compute + ram_io + ssd_io)
    }

    fn overlap_tps(&self, cfg: &SimConfig) -> f64 {
        let ram_io = self.ram_hits as f64 * cfg.expert_bytes as f64 / PCIE_BPS;
        let ssd_io = self.ssd_misses as f64 * cfg.expert_bytes as f64 / NVME_BPS;
        let compute = self.n_tokens as f64 / GPU_TPS;
        self.n_tokens as f64 / compute.max(ram_io + ssd_io)
    }
}

// ── Expert frequency analysis ───────────────────────────

fn compute_hot_experts(records: &[ActivationRecord]) -> Vec<(u16, u16, u64)> {
    let mut freq: HashMap<(u16, u16), u64> = HashMap::new();
    for r in records {
        for &eid in r.experts() {
            *freq.entry((r.layer, eid)).or_default() += 1;
        }
    }
    let mut sorted: Vec<_> = freq.into_iter().map(|((l, e), c)| (l, e, c)).collect();
    sorted.sort_by(|a, b| b.2.cmp(&a.2));
    sorted
}

fn run_simulation(
    cfg: &SimConfig,
    sorted_records: &[&ActivationRecord],
    matrix: &CooccurMatrix,
    hot_experts: &[(u16, u16, u64)],
) -> Stats {
    let mut cache = FastCache::new(N_LAYERS, N_EXPERTS, cfg.vram_cap);

    for &(layer, expert, _) in hot_experts.iter().take(cfg.hot_count) {
        cache.pin_expert(layer, expert);
    }

    let mut stats = Stats::default();
    let mut prev_token = u32::MAX;

    for r in sorted_records {
        if r.token_idx != prev_token {
            prev_token = r.token_idx;
            stats.n_tokens += 1;
        }

        for &eid in r.experts() {
            let tier = cache.touch(r.layer, eid);
            match tier {
                Tier::Vram => stats.vram_hits += 1,
                Tier::Ram => {
                    stats.ram_hits += 1;
                    cache.promote_to_vram(r.layer, eid);
                }
                _ => {
                    stats.ssd_misses += 1;
                    cache.promote_to_vram(r.layer, eid);
                }
            }
        }

        if cfg.k_prime > 0 && (r.layer as usize + 1) < N_LAYERS as usize {
            let candidates = matrix.predict(r.layer, r.experts(), cfg.k_prime);
            for &e in candidates.iter() {
                cache.promote_to_vram(r.layer + 1, e);
            }
        }
    }

    stats
}

// ── Main ────────────────────────────────────────────────

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

    let mut sorted: Vec<&ActivationRecord> = records.iter().collect();
    sorted.sort_by_key(|r| (r.token_idx, r.layer));

    let matrix = load_matrix(mat_path).expect("failed to load matrix");
    let hot_experts = compute_hot_experts(records);

    println!("=== Fake-1T + Hot/Cold Simulation ===");
    println!("Records: {} | Tokens: ~{}", records.len(),
        sorted.iter().map(|r| r.token_idx).collect::<std::collections::HashSet<_>>().len());
    println!("Hardware: NVMe {:.2} GB/s | PCIe {:.1} GB/s | GPU {:.1} t/s",
        NVME_BPS / 1e9, PCIE_BPS / 1e9, GPU_TPS);
    println!();

    // ── Hot expert 频率分布 ─────────────────────────────
    println!("=== Expert 使用频率 Top-20 ===");
    let total_uses: u64 = hot_experts.iter().map(|x| x.2).sum();
    let mut cumulative = 0u64;
    for (i, &(layer, expert, count)) in hot_experts.iter().take(20).enumerate() {
        cumulative += count;
        println!("  #{:>2}: L{:<2} E{:<3}  count={:<6}  累计占比={:.1}%",
            i + 1, layer, expert, count, cumulative as f64 / total_uses as f64 * 100.0);
    }

    // 累计分布关键点
    println!();
    for &pct_target in &[10, 20, 50, 80, 90, 95, 99] {
        let mut cum = 0u64;
        let target = total_uses * pct_target / 100;
        let n = hot_experts.iter().take_while(|x| { cum += x.2; cum <= target }).count() + 1;
        println!("  Top {:>4} experts ({:.1}% of {}) 覆盖 {}% 使用量",
            n, n as f64 / TOTAL as f64 * 100.0, TOTAL, pct_target);
    }
    println!();

    // ── Hot/Cold + Predictor 联合扫描 ───────────────────
    let hot_counts = [0, 50, 100, 200, 384, 512];
    let k_primes = [0, 16, 32, 64];

    for &(label, expert_bytes) in &[
        ("Q4_K_M (2.6 MB)", 2_621_440usize),
        ("1.58-bit (0.5 MB)", 524_288usize),
    ] {
        println!("=== Sweep: {} | vram_cap=512 ===", label);
        print!("{:>10}", "hot_pins");
        for &kp in &k_primes {
            print!(" {:>15}", format!("K'={}", kp));
        }
        println!();

        for &hc in &hot_counts {
            print!("{:>10}", hc);
            for &kp in &k_primes {
                let cfg = SimConfig {
                    vram_cap: 512,
                    expert_bytes,
                    k_prime: kp,
                    hot_count: hc,
                };
                let stats = run_simulation(&cfg, &sorted, &matrix, &hot_experts);
                print!(" {:>7.1} ({:4.1}%)", stats.serial_tps(&cfg), stats.vram_pct());
            }
            println!();
        }
        println!();
    }

    // ── 最终对比:最佳 hot/cold + predictor vs baseline ──
    println!("=== 最佳配置 vs Baseline ===");
    let configs = [
        ("Baseline (全 GPU, 无 swap)", SimConfig { vram_cap: TOTAL, expert_bytes: 2_621_440, k_prime: 0, hot_count: 0 }),
        ("假1T 无优化", SimConfig { vram_cap: 512, expert_bytes: 2_621_440, k_prime: 0, hot_count: 0 }),
        ("+ Predictor K'=32", SimConfig { vram_cap: 512, expert_bytes: 2_621_440, k_prime: 32, hot_count: 0 }),
        ("+ Hot 200 pins", SimConfig { vram_cap: 512, expert_bytes: 2_621_440, k_prime: 32, hot_count: 200 }),
        ("+ 1.58-bit 量化", SimConfig { vram_cap: 512, expert_bytes: 524_288, k_prime: 32, hot_count: 200 }),
        ("全部叠加 K'=64", SimConfig { vram_cap: 512, expert_bytes: 524_288, k_prime: 64, hot_count: 384 }),
    ];

    println!("{:<30} {:>8} {:>8} {:>10} {:>10}", "", "VRAM%", "miss%", "ser t/s", "ovlp t/s");
    for (name, cfg) in &configs {
        let stats = run_simulation(cfg, &sorted, &matrix, &hot_experts);
        let miss_pct = stats.ssd_misses as f64 / stats.total().max(1) as f64 * 100.0;
        println!("{:<30} {:>7.1}% {:>7.2}% {:>9.1} {:>9.1}",
            name, stats.vram_pct(), miss_pct,
            stats.serial_tps(cfg), stats.overlap_tps(cfg));
    }
    println!();
    println!("Baseline = {:.1} t/s. 越接近越好。", GPU_TPS);

    Ok(())
}
