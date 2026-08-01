//! Inspect a GGUF file: list metadata, count expert tensors, time per-expert reads.
//!
//! Usage:
//!   cargo run --release --example inspect_gguf -- path/to/model.gguf

use anyhow::Result;
use gguf_reader::{ExpertKind, GgufFile};
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());

    println!("Opening {}", path);
    let t0 = Instant::now();
    let g = GgufFile::open(&path)?;
    println!("  open + parse header: {:.2} ms  | file size {:.1} MB | data_start {} bytes",
             t0.elapsed().as_secs_f64() * 1000.0,
             g.file_size() as f64 / 1024.0 / 1024.0,
             g.data_start());

    println!("\n=== Metadata keys ({}) ===", g.metadata_keys().len());
    let mut keys: Vec<&String> = g.metadata_keys();
    keys.sort();
    for k in keys.iter().take(15) {
        println!("  {}", k);
    }
    if keys.len() > 15 { println!("  ... ({} more)", keys.len() - 15); }

    println!("\n=== Tensor count ===");
    println!("  total tensors: {}", g.n_tensors());

    let exps = g.list_expert_tensors();
    println!("  expert tensors (packed): {}", exps.len());
    let n_layers_with_exps = exps.iter().map(|e| e.layer).collect::<std::collections::HashSet<_>>().len();
    println!("  MoE layers detected:     {}", n_layers_with_exps);

    if let Some(first) = exps.first() {
        println!("\nFirst expert tensor:");
        println!("  name:      {}", first.name);
        println!("  layer:     {}", first.layer);
        println!("  kind:      {:?}", first.kind);
        println!("  shape:     {:?}", first.shape);
        println!("  byte_off:  {} ({:.2} MB into file)", first.byte_offset, first.byte_offset as f64 / 1e6);
        println!("  byte_size: {} ({:.2} MB)", first.byte_size, first.byte_size as f64 / 1024.0 / 1024.0);
    }

    // Try to slice individual experts (assume 128 for Qwen 30B-A3B)
    let n_experts = 128u32;
    println!("\n=== Per-expert slot read (n_experts={n_experts}) ===");

    // Pick a representative middle layer
    let target_layer: Option<u32> = exps.iter().map(|e| e.layer).max().map(|m| m / 2);
    if let Some(l) = target_layer {
        for kind in [ExpertKind::GateExps, ExpertKind::UpExps, ExpertKind::DownExps] {
            match g.expert_slot_bytes(l, kind, 0, n_experts) {
                Ok(s) => {
                    println!("  layer {} {:?} slot 0:  {:>8} bytes  ({:.2} KB)",
                             l, kind, s.len(), s.len() as f64 / 1024.0);
                }
                Err(e) => println!("  layer {} {:?}: ERROR {}", l, kind, e),
            }
        }

        // Time 1000 random expert reads (simulates per-expert on-demand load)
        println!("\n=== Timing: read 1000 random expert slots ===");
        let mut layers: Vec<u32> = exps.iter().map(|e| e.layer).collect();
        layers.sort(); layers.dedup();
        let kinds = [ExpertKind::GateExps, ExpertKind::UpExps, ExpertKind::DownExps];
        let n_iters = 1000;
        let mut total_bytes: usize = 0;
        // Use a deterministic LCG so we don't pull rand
        let mut state: u64 = 0xDEADBEEF;
        let next = |s: &mut u64| -> u64 { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1); *s };

        let t = Instant::now();
        for _ in 0..n_iters {
            let l = layers[next(&mut state) as usize % layers.len()];
            let k = kinds[next(&mut state) as usize % kinds.len()];
            let slot = (next(&mut state) as u32) % n_experts;
            let bytes = g.expert_slot_bytes(l, k, slot, n_experts)?;
            // touch first byte to force page-in (mmap is lazy)
            std::hint::black_box(bytes[0]);
            total_bytes += bytes.len();
        }
        let dt = t.elapsed().as_secs_f64();
        println!("  {} reads in {:.2} ms  ({:.3} ms/read avg)",
                 n_iters, dt * 1000.0, dt * 1000.0 / n_iters as f64);
        println!("  total {:.2} MB read at {:.2} MB/s",
                 total_bytes as f64 / 1024.0 / 1024.0,
                 total_bytes as f64 / dt / 1024.0 / 1024.0);
        println!("  (note: with mmap'd file, this is page cache or NVMe depending on OS state)");

        // Force-fault every byte of one whole expert tensor and time it (simulates first-touch from cold)
        println!("\n=== Cold full-tensor read (sum bytes to defeat optimizer) ===");
        let name = format!("blk.{}.ffn_gate_exps.weight", l);
        let bytes = g.tensor_bytes(&name)?;
        let t = Instant::now();
        let mut sum: u64 = 0;
        for chunk in bytes.chunks(4096) {
            sum = sum.wrapping_add(chunk[0] as u64);
        }
        let dt = t.elapsed().as_secs_f64();
        println!("  read {:.2} MB in {:.2} ms = {:.2} MB/s  (page-fault driven)",
                 bytes.len() as f64 / 1024.0 / 1024.0,
                 dt * 1000.0,
                 bytes.len() as f64 / dt / 1024.0 / 1024.0);
        std::hint::black_box(sum);
    }

    Ok(())
}
