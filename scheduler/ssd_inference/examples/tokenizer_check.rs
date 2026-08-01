//! Day 1 verification: ModelConfig::from_gguf parses Qwen3-30B metadata,
//! and our BPE tokenizer round-trips a few prompts.
//!
//! Usage:
//!   cargo run --release -p ssd_inference --example tokenizer_check -- path/to/model.gguf

use anyhow::Result;
use gguf_reader::GgufFile;
use ssd_inference::{ModelConfig, Tok};

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());

    println!("Opening {}", path);
    let g = GgufFile::open(&path)?;

    let cfg = ModelConfig::from_gguf(&g)?;
    println!("\n=== ModelConfig ===");
    println!("  arch:             {}", cfg.arch);
    println!("  n_layer:          {}", cfg.n_layer);
    println!("  d_model:          {}", cfg.d_model);
    println!("  n_q_heads:        {}", cfg.n_q_heads);
    println!("  n_kv_heads:       {}  (GQA group {})", cfg.n_kv_heads, cfg.gqa_group());
    println!("  head_dim:         {}", cfg.head_dim);
    println!("  vocab:            {}", cfg.vocab);
    println!("  moe_intermediate: {}", cfg.moe_intermediate);
    println!("  n_experts:        {}", cfg.n_experts);
    println!("  top_k:            {}", cfg.top_k);
    println!("  rope_theta:       {}", cfg.rope_theta);
    println!("  rms_eps:          {}", cfg.rms_eps);
    println!("  context_length:   {}", cfg.context_length);
    println!("  eos_token_id:     {}", cfg.eos_token_id);
    println!("  bos_token_id:     {:?}", cfg.bos_token_id);

    println!("\n=== Tokenizer round-trip ===");
    let tok = Tok::from_gguf(&g)?;

    // Reference token ids cross-checked with the HF Qwen3 tokenizer (Qwen2-style BPE):
    //   "Hello, my name is"           => [9707, 11, 847, 829, 374]
    //   "The capital of France is"    => [785, 6722, 315, 9625, 374]
    //   "1 + 1 ="                     => [16, 488, 220, 16, 284]
    let prompts = [
        ("Hello, my name is",        &[9707u32, 11, 847, 829, 374][..]),
        ("The capital of France is", &[785u32, 6722, 315, 9625, 374][..]),
        ("1 + 1 =",                  &[16u32, 488, 220, 16, 284][..]),
    ];

    let mut all_ok = true;
    for (text, expected) in prompts {
        let ids = tok.encode(text)?;
        let round = tok.decode(&ids)?;
        let ok = ids == expected;
        all_ok &= ok;
        let mark = if ok { "OK".to_string() } else { format!("MISMATCH expected {:?}", expected) };
        println!("  \"{}\" -> {:?}  {}", text, ids, mark);
        println!("      decoded: {:?}", round);
    }

    // Special-token sanity
    println!("\n=== Special tokens ===");
    println!("  EOS id={}  str={:?}", cfg.eos_token_id, tok.id_to_str(cfg.eos_token_id));
    if let Some(b) = cfg.bos_token_id {
        println!("  BOS id={}  str={:?}", b, tok.id_to_str(b));
    } else {
        println!("  BOS: none (Qwen2/3 sets add_bos_token=false)");
    }

    if !all_ok {
        anyhow::bail!("tokenizer round-trip mismatch — abort Day 1");
    }
    println!("\nDay 1 verification: OK");
    Ok(())
}
