use anyhow::Result;
use gguf_reader::MultiGgufFile;
use ssd_inference::model::ModelConfig;

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../../models/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf".into());

    println!("Opening multi-shard: {}", path);
    let mg = MultiGgufFile::open(&path)?;
    println!("  Shards: {}", mg.n_shards());
    println!("  Total tensors: {}", mg.tensor_names().len());

    let cfg = ModelConfig::from_multi_gguf(&mg)?;
    println!("\nModel: {} | layers={} d={} q_heads={} kv_heads={} experts={} top_k={}",
        cfg.arch, cfg.n_layer, cfg.d_model, cfg.n_q_heads, cfg.n_kv_heads, cfg.n_experts, cfg.top_k);
    println!("  moe_intermediate={} head_dim={} vocab={}", cfg.moe_intermediate, cfg.head_dim, cfg.vocab);

    // Verify tensors from both shards are accessible
    for l in [0, cfg.n_layer / 2, cfg.n_layer - 1] {
        let name = format!("blk.{}.attn_q.weight", l);
        match mg.tensor_info(&name) {
            Some(info) => println!("  {} -> {:?} {:?}", name, info.ggml_dtype, info.shape.dims()),
            None => println!("  {} -> MISSING", name),
        }
    }

    let experts = mg.list_expert_tensors();
    let min_l = experts.iter().map(|e| e.layer).min().unwrap_or(0);
    let max_l = experts.iter().map(|e| e.layer).max().unwrap_or(0);
    println!("\n  Expert tensors: {} (layers {}..{})", experts.len(), min_l, max_l);

    // Check expert quant types at layer 0
    for name in ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"] {
        if let Some(info) = mg.tensor_info(name) {
            let size = mg.tensor_byte_size(name).unwrap_or(0);
            println!("  {} -> {:?} {:?} ({:.1} MB)", name, info.ggml_dtype, info.shape.dims(), size as f64 / 1e6);
        }
    }

    // Scan all layers for non-Q4K attention weights
    println!("\n  Non-Q4K attention weights:");
    for l in 0..cfg.n_layer {
        for suffix in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            let name = format!("blk.{}.{}.weight", l, suffix);
            if let Some(info) = mg.tensor_info(&name) {
                let dt = format!("{:?}", info.ggml_dtype);
                if dt != "Q4K" && dt != "F32" {
                    println!("    L{:02} {} -> {}", l, suffix, dt);
                }
            }
        }
    }

    println!("\nMultiGgufFile OK!");
    Ok(())
}
