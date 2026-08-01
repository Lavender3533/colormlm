//! Day 2 verification: CPU-dequant the embedding table from GGUF and inspect
//! one row. Sanity check: nonzero, finite, sensible norm.
//!
//! Run: cargo run --release -p ssd_inference --example embed_lookup -- model.gguf [token_id]

use anyhow::Result;
use candle_core::quantized::GgmlDType;
use gguf_reader::GgufFile;
use ssd_inference::{
    weights::{cpu_dequant_q4_k, cpu_dequant_q6_k},
    ModelConfig, TensorNames, Tok,
};
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());
    let token_id: u32 = std::env::args().nth(2)
        .map(|s| s.parse().unwrap()).unwrap_or(9707); // "Hello"

    println!("Opening {}", path);
    let g = GgufFile::open(&path)?;
    let cfg = ModelConfig::from_gguf(&g)?;
    let info = g.tensor_info(TensorNames::EMBED).unwrap();
    println!("embed dtype={:?} shape={:?} ({:.1} MB Q → {:.1} MB fp32)",
        info.ggml_dtype, info.shape.dims(),
        g.tensor_bytes(TensorNames::EMBED)?.len() as f64 / 1024.0 / 1024.0,
        cfg.vocab as f64 * cfg.d_model as f64 * 4.0 / 1024.0 / 1024.0);

    let bytes = g.tensor_bytes(TensorNames::EMBED)?;
    let t = Instant::now();
    let embed = match info.ggml_dtype {
        GgmlDType::Q4K => cpu_dequant_q4_k(bytes),
        GgmlDType::Q6K => cpu_dequant_q6_k(bytes),
        other => anyhow::bail!("unsupported embed dtype {:?}", other),
    };
    println!("CPU dequant: {:.2} s", t.elapsed().as_secs_f64());

    let d = cfg.d_model as usize;
    let row = &embed[token_id as usize * d..(token_id as usize + 1) * d];

    let tok = Tok::from_gguf(&g)?;
    let token_str = tok.id_to_str(token_id).unwrap_or_else(|| "?".into());

    let n_nz = row.iter().filter(|&&x| x != 0.0).count();
    let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    let max_abs = row.iter().map(|x| x.abs()).fold(0f32, f32::max);
    let any_nan = row.iter().any(|x| x.is_nan() || !x.is_finite());

    println!("\n=== Row for token {} ({:?}) ===", token_id, token_str);
    println!("  shape:    [{}] fp32", row.len());
    println!("  nonzero:  {} / {}", n_nz, row.len());
    println!("  L2 norm:  {:.4}", norm);
    println!("  max abs:  {:.4}", max_abs);
    println!("  any NaN:  {}", any_nan);
    println!("  first 16: {:?}", &row[..16]);

    if any_nan { anyhow::bail!("embedding row has NaN/Inf"); }
    if n_nz < row.len() / 4 { anyhow::bail!("row mostly zero — dequant suspect"); }
    if !(0.5..50.0).contains(&norm) { anyhow::bail!("norm {:.3} suspicious", norm); }
    println!("\nDay 2 verification: OK");
    Ok(())
}
