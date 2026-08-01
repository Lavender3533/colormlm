//! train_matrix — 端到端 MVP 第一个数据点。
//!
//! 1. 读 collector 写出的 activations.bin
//! 2. 按顺序喂给 MatrixBuilder
//! 3. 出快照
//! 4. 自检:用每对连续层 (N → N+1) 验证 predict() 命中率
//!
//! 用法:cargo run --release --example train_matrix -- <activations.bin> [n_experts] [n_layers]

use predictor::{ActivationRecord, MatrixBuilder};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "data/activations.bin".to_string());
    let n_experts: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_layers: u16 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);

    println!("Loading {} (n_layers={}, n_experts={})", path, n_layers, n_experts);

    let mut buf = Vec::new();
    File::open(&path)?.read_to_end(&mut buf)?;
    let rec_size = std::mem::size_of::<ActivationRecord>();
    let n_total = buf.len() / rec_size;
    let records: &[ActivationRecord] = bytemuck::cast_slice(&buf[..n_total * rec_size]);
    println!("Loaded {} records", records.len());

    // Builder now handles streaming/out-of-order via per-token tracking,
    // no need to sort.

    // ──────────────────────────────────────────────────
    // Train: feed all records into MatrixBuilder
    // ──────────────────────────────────────────────────
    let builder = MatrixBuilder::new(n_layers, n_experts);
    let t0 = std::time::Instant::now();
    for r in records {
        builder.observe(r);
    }
    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Trained in {:.1} ms ({:.0} obs/sec)",
        train_ms, records.len() as f64 / train_ms * 1000.0);

    let snap = builder.build_snapshot();
    println!("Snapshot version={}, total observations={}", snap.version(), snap.total_observations());

    // Coverage stats
    let n_layer_cells = n_layers as usize * n_experts as usize * n_experts as usize;
    let nonzero_cells = snap.counts().iter().filter(|&&c| c > 0).count();
    println!("Matrix coverage: {} / {} cells = {:.2}%",
        nonzero_cells, n_layer_cells,
        nonzero_cells as f64 / n_layer_cells as f64 * 100.0);

    // ──────────────────────────────────────────────────
    // Self-validation: for each (token_t, layer_l → layer_l+1) pair, predict.
    // Note: collector writes prefill records as (token0..N, layer0), (token0..N, layer1), ...
    // so we can't just use windows(2) — must build an index by (token, layer).
    // ──────────────────────────────────────────────────
    println!();
    println!("=== Validation (self-test on training data) ===");

    // Build (token, layer) → record index
    let mut by_key: HashMap<(u32, u16), &ActivationRecord> = HashMap::with_capacity(records.len());
    for r in records {
        by_key.insert((r.token_idx, r.layer), r);
    }

    let top_ks = [8usize, 16, 32];
    for &top_k in &top_ks {
        let mut total = 0;
        let mut sum_recall = 0.0f64;
        let mut perfect = 0;

        for r in records {
            // For each record, look up (same token, next layer)
            let next_layer = r.layer + 1;
            if next_layer >= n_layers { continue; }
            let Some(next) = by_key.get(&(r.token_idx, next_layer)) else { continue; };

            let prev_experts: Vec<u16> = r.experts().to_vec();
            let actual: HashSet<u16> = next.experts().iter().copied().collect();
            let predicted: HashSet<u16> =
                snap.predict(r.layer, &prev_experts, top_k).into_iter().collect();
            let hits = actual.intersection(&predicted).count();

            sum_recall += hits as f64 / actual.len() as f64;
            if hits == actual.len() { perfect += 1; }
            total += 1;
        }

        if total == 0 {
            println!("top_k={:3}: no validation pairs", top_k);
            continue;
        }
        let avg_recall = sum_recall / total as f64;
        println!("top_k={:3}: {} pairs, avg recall = {:.1}% (perfect: {} / {} = {:.1}%)",
            top_k, total, avg_recall * 100.0,
            perfect, total, perfect as f64 / total as f64 * 100.0);
    }

    println!();
    println!("Note: this is in-sample evaluation (train = test). True test would split.");

    Ok(())
}
