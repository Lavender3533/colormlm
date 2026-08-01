//! 验证"石子入水"假设:浅层激活分布广、深层激活分布窄?
//!
//! 对每层计算:
//!   - Shannon entropy(bits)— 越高分布越均匀(广);越低越集中(窄)
//!   - Top-K 累积频率(K=8/16/32)— 多少 token 命中前 K 个 expert
//!   - Unique experts seen — 该层一共激活过多少个不同 expert
//!   - Gini-like 集中度
//!
//! 用法:
//!   cargo run --release --example layer_entropy_analysis -- ../data/activations_qwen_wiki.bin

use predictor::ActivationRecord;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

const N_EXPERTS: usize = 128; // Qwen3-30B-A3B

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/activations_qwen_wiki.bin".to_string());

    let mut f = File::open(&path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let n_records = buf.len() / std::mem::size_of::<ActivationRecord>();
    let records = bytemuck::cast_slice::<u8, ActivationRecord>(
        &buf[..n_records * std::mem::size_of::<ActivationRecord>()],
    );

    println!("=== Layer Entropy Analysis ===");
    println!("File: {}", path);
    println!("Records: {}", records.len());

    // 找出层范围
    let mut layer_max = 0u16;
    for r in records {
        layer_max = layer_max.max(r.layer);
    }
    let n_layers = (layer_max as usize) + 1;
    println!("Layers: 0..{} ({} total)", layer_max, n_layers);
    println!();

    // 每层一个频次表
    // counts[layer][expert_id] = 激活次数
    let mut counts: Vec<Vec<u64>> = vec![vec![0u64; N_EXPERTS]; n_layers];
    let mut totals: Vec<u64> = vec![0u64; n_layers];
    let mut tokens_per_layer: Vec<u64> = vec![0u64; n_layers];

    for r in records {
        let l = r.layer as usize;
        if l >= n_layers {
            continue;
        }
        tokens_per_layer[l] += 1;
        for &eid in r.experts() {
            let e = eid as usize;
            if e < N_EXPERTS {
                counts[l][e] += 1;
                totals[l] += 1;
            }
        }
    }

    // 表头
    println!(
        "{:>5}  {:>8}  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}",
        "layer", "tokens", "activ", "entropy", "ent_norm", "uniq", "top8%", "top32%"
    );
    println!("{:->5}  {:->8}  {:->10}  {:->8}  {:->8}  {:->10}  {:->10}  {:->10}",
             "", "", "", "", "", "", "", "");

    let max_entropy = (N_EXPERTS as f64).log2();

    let mut summary: Vec<(usize, f64, f64, usize, f64, f64)> = Vec::new();

    for l in 0..n_layers {
        let total = totals[l];
        if total == 0 {
            continue;
        }
        // 概率分布
        let mut probs: Vec<f64> = counts[l].iter().map(|&c| c as f64 / total as f64).collect();
        // entropy
        let mut h = 0.0f64;
        for &p in &probs {
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        // unique experts
        let uniq = probs.iter().filter(|&&p| p > 0.0).count();
        // top-K 累积
        probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top8: f64 = probs.iter().take(8).sum::<f64>() * 100.0;
        let top32: f64 = probs.iter().take(32).sum::<f64>() * 100.0;

        let ent_norm = h / max_entropy;

        println!(
            "{:>5}  {:>8}  {:>10}  {:>8.3}  {:>8.3}  {:>10}  {:>9.1}%  {:>9.1}%",
            l, tokens_per_layer[l], total, h, ent_norm, uniq, top8, top32
        );

        summary.push((l, h, ent_norm, uniq, top8, top32));
    }

    println!();
    println!("=== 假设验证 ===");

    // 假设: 浅层 entropy 高,深层 entropy 低
    // 用三段平均比较:0..n/3, n/3..2n/3, 2n/3..n
    if summary.len() >= 3 {
        let n = summary.len();
        let third = n / 3;
        let avg_h = |start: usize, end: usize| -> (f64, f64, f64, f64) {
            let slice = &summary[start..end];
            let len = slice.len() as f64;
            let h = slice.iter().map(|(_, h, _, _, _, _)| *h).sum::<f64>() / len;
            let uniq = slice.iter().map(|(_, _, _, u, _, _)| *u as f64).sum::<f64>() / len;
            let top8 = slice.iter().map(|(_, _, _, _, t8, _)| *t8).sum::<f64>() / len;
            let top32 = slice.iter().map(|(_, _, _, _, _, t32)| *t32).sum::<f64>() / len;
            (h, uniq, top8, top32)
        };

        let (h0, u0, t8_0, t32_0) = avg_h(0, third);
        let (h1, u1, t8_1, t32_1) = avg_h(third, 2 * third);
        let (h2, u2, t8_2, t32_2) = avg_h(2 * third, n);

        println!(
            "{:<14} entropy={:>6.3}  uniq_experts={:>6.1}  top8%={:>5.1}  top32%={:>5.1}",
            "浅层 (0..{})", h0, u0, t8_0, t32_0
        );
        println!(
            "{:<14} entropy={:>6.3}  uniq_experts={:>6.1}  top8%={:>5.1}  top32%={:>5.1}",
            "中间层", h1, u1, t8_1, t32_1
        );
        println!(
            "{:<14} entropy={:>6.3}  uniq_experts={:>6.1}  top8%={:>5.1}  top32%={:>5.1}",
            "深层", h2, u2, t8_2, t32_2
        );
        println!();

        let delta = h0 - h2;
        let ratio = if h2 > 0.0 { h0 / h2 } else { 0.0 };
        println!(
            "Δ(浅-深) entropy = {:+.3} bits  (比率 {:.2}x)",
            delta, ratio
        );
        println!(
            "Δ(深-浅) top32% = {:+.1} pp",
            t32_2 - t32_0
        );

        if delta > 0.3 {
            println!("✅ 假设基本成立:浅层熵显著高于深层(差 > 0.3 bits)");
        } else if delta > 0.1 {
            println!("⚠️  假设部分成立:浅层熵略高于深层(差 0.1~0.3 bits)");
        } else if delta.abs() <= 0.1 {
            println!("❌ 假设不成立:浅层与深层熵几乎一致");
        } else {
            println!("🔄 反假设:深层熵反而更高({:+.3} bits)", -delta);
        }
    }

    // 极端层
    println!();
    println!("=== 极端层 ===");
    let mut by_h = summary.clone();
    by_h.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("熵最低 5 层(最集中,适合 SSD + 预测器):");
    for (l, h, hn, uniq, t8, t32) in by_h.iter().take(5) {
        println!("  layer {:>2}  H={:.3} ({:.1}%)  uniq={}  top8%={:.1}  top32%={:.1}",
                 l, h, hn * 100.0, uniq, t8, t32);
    }
    println!("熵最高 5 层(最广,适合 VRAM 常驻):");
    for (l, h, hn, uniq, t8, t32) in by_h.iter().rev().take(5) {
        println!("  layer {:>2}  H={:.3} ({:.1}%)  uniq={}  top8%={:.1}  top32%={:.1}",
                 l, h, hn * 100.0, uniq, t8, t32);
    }

    // 全局 expert 频次,看是否有"全局热门 expert"
    println!();
    println!("=== 全局 expert 热度(跨所有层) ===");
    let mut global: HashMap<u16, u64> = HashMap::new();
    let mut grand = 0u64;
    for r in records {
        for &eid in r.experts() {
            *global.entry(eid).or_insert(0) += 1;
            grand += 1;
        }
    }
    let mut g_vec: Vec<(u16, u64)> = global.into_iter().collect();
    g_vec.sort_by(|a, b| b.1.cmp(&a.1));
    let total = grand as f64;
    let cumul: Vec<(usize, f64)> = [10, 32, 64, 128]
        .iter()
        .map(|&k| {
            let s: u64 = g_vec.iter().take(k).map(|(_, c)| *c).sum();
            (k, s as f64 / total * 100.0)
        })
        .collect();
    for (k, p) in cumul {
        println!("  Top {:<3} 全局 expert 覆盖 {:.1}% 激活", k, p);
    }

    Ok(())
}
