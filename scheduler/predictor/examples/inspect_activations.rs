use predictor::ActivationRecord;
use std::fs::File;
use std::io::Read;

fn main() -> std::io::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "data/activations.bin".to_string());

    let mut f = File::open(&path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let n_records = buf.len() / std::mem::size_of::<ActivationRecord>();
    let records = bytemuck::cast_slice::<u8, ActivationRecord>(&buf[..n_records * std::mem::size_of::<ActivationRecord>()]);

    println!("File: {}", path);
    println!("Total records: {}", records.len());
    println!();

    // Stats
    let mut layer_min = u16::MAX;
    let mut layer_max = 0u16;
    let mut expert_min = u16::MAX;
    let mut expert_max = 0u16;
    let mut weight_sum = 0.0f64;
    let mut weight_count = 0usize;
    let mut n_used_min = u8::MAX;
    let mut n_used_max = 0u8;
    let mut token_min = u32::MAX;
    let mut token_max = 0u32;

    for r in records {
        layer_min = layer_min.min(r.layer);
        layer_max = layer_max.max(r.layer);
        n_used_min = n_used_min.min(r.n_experts_used);
        n_used_max = n_used_max.max(r.n_experts_used);
        token_min = token_min.min(r.token_idx);
        token_max = token_max.max(r.token_idx);

        for i in 0..r.n_experts_used as usize {
            expert_min = expert_min.min(r.expert_ids[i]);
            expert_max = expert_max.max(r.expert_ids[i]);
            weight_sum += r.expert_weights[i] as f64;
            weight_count += 1;
        }
    }

    println!("Layer range:    {} .. {}", layer_min, layer_max);
    println!("Token range:    {} .. {}", token_min, token_max);
    println!("Expert ID range: {} .. {}", expert_min, expert_max);
    println!("Experts/record: {} .. {}", n_used_min, n_used_max);
    println!("Avg expert weight: {:.4} (sum/{} samples)", weight_sum / weight_count as f64, weight_count);
    println!();

    // Show first 3 records in detail
    println!("First 3 records:");
    for (i, r) in records.iter().take(3).enumerate() {
        println!("  [{}] token={} layer={} n={} ts={}", i, r.token_idx, r.layer, r.n_experts_used, r.timestamp_ns);
        let experts: Vec<_> = (0..r.n_experts_used as usize).map(|j| (r.expert_ids[j], r.expert_weights[j])).collect();
        println!("       experts={:?}", experts);
    }

    Ok(())
}
