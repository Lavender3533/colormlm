//! Engine demo: load model, prefetch a batch of experts, report stats.
//!
//! Mimics the new engine's "scheduler issues a list of expert loads, loader
//! pipelines them through transfer queue" loop. This is the c-stage's
//! hello-world.
//!
//! Usage:
//!   cargo run --release -p ssd_inference --example engine_demo \
//!     -- ../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf

use anyhow::Result;
use ssd_inference::{Engine, EngineConfig, ExpertKind};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read as _;
use std::time::Instant;

const ACTIVE_PER_LAYER: u32 = 8; // top-8 routing
const N_TOKENS_TO_SIMULATE: u32 = 8;

// Mirror of predictor::ActivationRecord (Pod, 112 bytes). Re-declared here to
// avoid pulling predictor as a dep — same layout, asserted by size_of check.
#[repr(C)]
#[derive(Clone, Copy)]
struct ActRecord {
    timestamp_ns: u64,
    token_idx: u32,
    layer: u16,
    n_experts_used: u8,
    _padding: u8,
    expert_ids: [u16; 16],
    expert_weights: [f32; 16],
}
const _: () = assert!(std::mem::size_of::<ActRecord>() == 112);

fn load_activations(path: &str) -> Result<Vec<ActRecord>> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    let n = buf.len() / std::mem::size_of::<ActRecord>();
    let recs: Vec<ActRecord> = (0..n).map(|i| {
        let off = i * std::mem::size_of::<ActRecord>();
        unsafe {
            std::ptr::read_unaligned(buf.as_ptr().add(off) as *const ActRecord)
        }
    }).collect();
    Ok(recs)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next()
        .unwrap_or_else(|| "../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());
    // Optional 2nd arg: path to real activations (.bin). If present, use replay
    // mode; else fall back to deterministic LCG.
    let act_path = args.next();

    // Pick preset by file name heuristic
    let config = if path.contains("235B") {
        println!("Detected 235B model — using qwen3_235b_q2 preset");
        EngineConfig::qwen3_235b_q2()
    } else {
        EngineConfig::qwen3_30b()
    };

    println!("=== ssd_inference engine demo ===\n");

    let t0 = Instant::now();
    let mut engine = Engine::new(&path, config)?;
    println!("Engine init: {:.2} ms", t0.elapsed().as_secs_f64() * 1000.0);

    let s = engine.stats();
    println!("  GPU:                  {}", s.gpu_name);
    println!("  Dedicated transfer:   {}", s.has_dedicated_transfer);
    println!("  VRAM pool capacity:   {} slots × {} MB = {} MB",
             s.vram_pool_capacity,
             engine.config.vram_slot_bytes / 1024 / 1024,
             s.vram_pool_total_bytes / 1024 / 1024);
    println!("  GGUF tensors:         {}", engine.gguf.n_tensors());
    println!("  Reader handles:       1 (single-thread seek_read)");

    // Detect actual MoE layer count from this GGUF (may be partial for split files)
    let exps = engine.gguf.list_expert_tensors();
    let n_layers_in_shard: u32 = exps.iter().map(|e| e.layer).max().map(|m| m + 1).unwrap_or(0);
    println!("  MoE layers in shard:  {}", n_layers_in_shard);
    let avg_expert_size = exps.iter().map(|e| (e.byte_size as usize) / 128).sum::<usize>()
        / exps.len().max(1);
    println!("  Avg expert size:      {} KB", avg_expert_size / 1024);
    println!();

    // ── Pick workload mode ─────────────────────────────────
    let kinds = [ExpertKind::GateExps, ExpertKind::UpExps, ExpertKind::DownExps];
    let n_experts_per_layer = engine.config.n_experts_total;

    enum Workload {
        Random {
            n_tokens: u32,
            n_layers: u32,
            active: u32,
        },
        Replay {
            // grouped by token: for each token, list of (layer, expert_ids[..n_used])
            token_records: Vec<Vec<(u16, Vec<u16>)>>,
            n_layers_in_shard: u32,
        },
    }

    let workload = if let Some(ap) = act_path {
        println!("Replay mode: loading {}", ap);
        let recs = load_activations(&ap)?;
        let n_total_layers = recs.iter().map(|r| r.layer).max().unwrap_or(0) + 1;
        println!("  records: {} | model layers (full): {}", recs.len(), n_total_layers);
        if (n_layers_in_shard as u16) < n_total_layers {
            println!("  ⚠️  GGUF shard has {} layers, model has {} — replaying ONLY layer < {}",
                n_layers_in_shard, n_total_layers, n_layers_in_shard);
            println!("     Reported t/s is for partial model — real t/s ≈ result × shard/full ratio");
        }

        let mut by_token: BTreeMap<u32, Vec<(u16, Vec<u16>)>> = BTreeMap::new();
        let mut kept = 0;
        let mut skipped = 0;
        for r in &recs {
            if (r.layer as u32) >= n_layers_in_shard {
                skipped += 1;
                continue;
            }
            let n = r.n_experts_used as usize;
            let exp = r.expert_ids[..n].to_vec();
            by_token.entry(r.token_idx).or_default().push((r.layer, exp));
            kept += 1;
        }
        println!("  records kept: {} (skipped {} for out-of-shard layers)", kept, skipped);

        // sort each token's records by layer
        for v in by_token.values_mut() {
            v.sort_by_key(|(l, _)| *l);
        }
        let token_records: Vec<_> = by_token.into_values().collect();
        println!("  unique tokens: {}", token_records.len());
        Workload::Replay { token_records, n_layers_in_shard }
    } else {
        println!("Random mode (LCG): {} tokens × {} layers × {} active × 3 kinds = {} loads",
            N_TOKENS_TO_SIMULATE, n_layers_in_shard, ACTIVE_PER_LAYER,
            N_TOKENS_TO_SIMULATE * n_layers_in_shard * ACTIVE_PER_LAYER * 3);
        Workload::Random {
            n_tokens: N_TOKENS_TO_SIMULATE,
            n_layers: n_layers_in_shard,
            active: ACTIVE_PER_LAYER,
        }
    };

    let mut rng_state: u64 = 0x1234_5678_DEAD_BEEF;
    let mut next = |s: &mut u64| -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*s >> 32) as u32
    };

    let t_sim = Instant::now();
    let actual_tokens: u32;
    match workload {
        Workload::Random { n_tokens, n_layers, active } => {
            actual_tokens = n_tokens;
            for token in 0..n_tokens {
                let t_token = Instant::now();
                for layer in 0..n_layers {
                    for _a in 0..active {
                        let slot = next(&mut rng_state) % n_experts_per_layer;
                        for &kind in &kinds {
                            engine.ensure_expert_in_vram(layer, kind, slot)?;
                        }
                    }
                }
                engine.flush()?;
                print_token_line(token, t_token, t_sim, &engine);
            }
        }
        Workload::Replay { token_records, n_layers_in_shard: _ } => {
            // Cap to fit ~3-5 min run; 128 tokens × ~1272 unique loads each
            // touches all ~20k experts in shard ~5×, exceeding 24 GB page cache.
            let max_tokens = 128u32.min(token_records.len() as u32);
            actual_tokens = max_tokens;
            for (ti, recs) in token_records.iter().take(max_tokens as usize).enumerate() {
                let t_token = Instant::now();
                for (layer, experts) in recs {
                    for &eid in experts {
                        for &kind in &kinds {
                            engine.ensure_expert_in_vram(*layer as u32, kind, eid as u32)?;
                        }
                    }
                }
                engine.flush()?;
                if ti < 4 || ti % 16 == 0 || ti == max_tokens as usize - 1 {
                    print_token_line(ti as u32, t_token, t_sim, &engine);
                }
            }
        }
    }
    let total_s = t_sim.elapsed().as_secs_f64();

    let s = engine.stats();
    let throughput_eps = s.uploads_total as f64 / total_s;
    let throughput_gbs = s.bytes_uploaded_total as f64 / total_s / 1e9;
    let tokens_per_s = actual_tokens as f64 / total_s;

    println!("\n=== Results ===");
    println!("  Total time:    {:.2} s   (over {} tokens)", total_s, actual_tokens);
    println!("  Uploads:       {} ({:.1} GB)", s.uploads_total, s.bytes_uploaded_total as f64 / 1e9);
    println!("  Throughput:    {:.0} expert-loads/s | {:.2} GB/s", throughput_eps, throughput_gbs);
    println!("  Effective t/s (partial-model): {:.3}", tokens_per_s);
    println!();
    println!("  Reference points:");
    println!("    NVMe sustained read:    1.51 GB/s");
    println!("    llama.cpp ngl=0:        0.28 t/s (full model)");
    println!("    llama.cpp ngl=10+ot:    0.78 t/s (full model, Session 5 best)");

    Ok(())
}

fn print_token_line(token: u32, t_token: Instant, t_sim: Instant, engine: &Engine) {
    let token_dt = t_token.elapsed().as_secs_f64();
    let s = engine.stats();
    let elapsed = t_sim.elapsed().as_secs_f64();
    println!("  token {:>3}: {:.2}s ({:.2}s elapsed)  vram={:>3}/{}  uploads={:<7}  total={:>5} MB  cum_BW={:.2} GB/s",
        token, token_dt, elapsed, s.vram_pool_loaded, s.vram_pool_capacity,
        s.uploads_total, s.bytes_uploaded_total / 1024 / 1024,
        s.bytes_uploaded_total as f64 / elapsed / 1e9);
}
