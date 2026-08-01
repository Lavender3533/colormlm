//! Scheduler 端到端模拟:不改 llama.cpp,直接在 collector 数据上跑调度器,
//! 看不同 VRAM 配额、不同 K' 下的命中率。
//!
//! 这是 Step 4 在写真正集成代码前的"电脑模拟",回答关键问题:
//!   "如果调度器接进来了,我们到底能省多少 cache miss?"
//!
//! 用法:
//!   simulate_scheduler <activations.bin> <n_experts> <n_layers> [vram_cap] [k_prime]

use predictor::{ActivationRecord, MatrixBuilder};
use expert_cache::{ExpertCache, ExpertId, Tier};
use scheduler_core::{Scheduler, SchedulerConfig};
use std::fs::File;
use std::io::Read;
use std::sync::Arc;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "data/activations.bin".to_string());
    let n_experts: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_layers: u16  = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    // VRAM 容量(以专家数计)。默认值 = 总专家数的 25%
    let vram_cap: u32 = args.get(4).and_then(|s| s.parse().ok())
        .unwrap_or((n_layers as u32 * n_experts as u32) / 4);
    let k_prime: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(16);

    println!("=== Scheduler simulation ===");
    println!("Data:        {}", path);
    println!("Model:       {} layers × {} experts/layer", n_layers, n_experts);
    println!("VRAM cap:    {} experts ({:.1}% of total {})",
        vram_cap, vram_cap as f64 / (n_layers as f64 * n_experts as f64) * 100.0,
        n_layers as u32 * n_experts as u32);
    println!("K' (prefetch): {}", k_prime);
    println!();

    // 1. 读 records
    let mut buf = Vec::new();
    File::open(&path)?.read_to_end(&mut buf)?;
    let rec_size = std::mem::size_of::<ActivationRecord>();
    let n_total = buf.len() / rec_size;
    let records: &[ActivationRecord] = bytemuck::cast_slice(&buf[..n_total * rec_size]);
    println!("Loaded {} records", records.len());

    // 2. 训矩阵(用 70% 数据,留 30% 测试)
    let split = (records.len() as f64 * 0.7) as usize;
    let train = &records[..split];
    let test = &records[split..];

    let builder = MatrixBuilder::new(n_layers, n_experts);
    for r in train { builder.observe(r); }
    let matrix = builder.build_snapshot();
    println!("Matrix trained on {} records, coverage = {:.2}%",
        train.len(),
        matrix.counts().iter().filter(|&&c| c > 0).count() as f64
            / matrix.counts().len() as f64 * 100.0);

    // 4. 模拟两次:scheduler OFF (baseline) 和 scheduler ON
    let mut sorted_test: Vec<&ActivationRecord> = test.iter().collect();
    sorted_test.sort_by_key(|r| (r.token_idx, r.layer));

    println!();
    println!("=== Run A: Scheduler OFF (baseline, only sync-load on miss) ===");
    let stats_off = simulate(&sorted_test, n_layers, n_experts, vram_cap, &matrix, k_prime, false);
    print_stats(&stats_off);

    println!();
    println!("=== Run B: Scheduler ON (proactive prefetch enabled) ===");
    let stats_on = simulate(&sorted_test, n_layers, n_experts, vram_cap, &matrix, k_prime, true);
    print_stats(&stats_on);

    println!();
    println!("=== Comparison ===");
    let baseline_misses = stats_off.misses as f64;
    let opt_misses = stats_on.misses as f64;
    if baseline_misses > 0.0 {
        let saved_pct = (baseline_misses - opt_misses) / baseline_misses * 100.0;
        println!("Misses reduced: {:.0} → {:.0} ({:.1}% reduction)",
            baseline_misses, opt_misses, saved_pct);
    }

    Ok(())
}

#[derive(Default, Debug)]
struct SimStats {
    hits_vram: u64,
    hits_ram: u64,
    misses: u64,
    prefetches: u64,
    evictions: u64,
    total_uses: u64,
}

fn simulate(
    sorted_test: &[&ActivationRecord],
    n_layers: u16,
    n_experts: u16,
    vram_cap: u32,
    matrix: &predictor::CooccurMatrix,
    k_prime: usize,
    scheduler_enabled: bool,
) -> SimStats {
    let cache = Arc::new(ExpertCache::new(n_layers, n_experts, vram_cap, vram_cap * 4));
    let mut config = SchedulerConfig::default();
    config.prefetch_k_prime = k_prime;
    config.enabled = scheduler_enabled;
    let sched = Scheduler::new(matrix.clone(), cache.clone(), config);

    let mut stats = SimStats::default();

    for r in sorted_test {
        let layer = r.layer;
        for &eid in r.experts() {
            let id = ExpertId::new(layer, eid);
            let prev = cache.touch(id);
            stats.total_uses += 1;
            match prev {
                Tier::Vram => stats.hits_vram += 1,
                Tier::Ram  => stats.hits_ram += 1,
                _          => {
                    stats.misses += 1;
                    let _ = cache.request_to_vram(&[id]);
                }
            }
        }

        let cmds = sched.on_layer_complete(layer, r.experts());
        for cmd in cmds {
            match cmd {
                scheduler_core::SchedulerCommand::PrefetchToVram { .. } => stats.prefetches += 1,
                scheduler_core::SchedulerCommand::EvictFromVram { .. } => stats.evictions += 1,
            }
        }
    }
    stats
}

fn print_stats(s: &SimStats) {
    let total = s.total_uses as f64;
    let hit_total = s.hits_vram + s.hits_ram;
    println!("  VRAM hits:  {:>10}  ({:5.1}%)", s.hits_vram, s.hits_vram as f64 / total * 100.0);
    println!("  RAM hits:   {:>10}  ({:5.1}%)", s.hits_ram,  s.hits_ram  as f64 / total * 100.0);
    println!("  Misses:     {:>10}  ({:5.1}%)  ← need sync-load (slow)", s.misses, s.misses as f64 / total * 100.0);
    println!("  Total hit:  {:>10}  ({:5.1}%)", hit_total, hit_total as f64 / total * 100.0);
    println!("  Prefetches: {}", s.prefetches);
    println!("  Evictions:  {}", s.evictions);
}
