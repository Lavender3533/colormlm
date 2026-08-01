//! End-to-end generation: prompt → prefill → autoregressive decode loop.
//!
//! Run:
//!   cargo run --release -p ssd_inference --example first_token -- \
//!       path/to/model.gguf "The capital of France is"

use anyhow::Result;
use gguf_reader::GgufFile;
use ssd_inference::{
    engine::{Engine, EngineConfig},
    forward::Forward,
    kv_cache::KvCache,
    pipelines::Pipelines,
    weights::LoadedWeights,
    ModelConfig, Tok,
};
use std::io::Write;
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../../models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".to_string());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello, my name is".to_string());
    let max_tokens: usize = std::env::args().nth(3)
        .map(|s| s.parse().unwrap()).unwrap_or(32);

    println!("Opening {}\nPrompt: {:?}\nmax_tokens: {}\n", path, prompt, max_tokens);

    let g = GgufFile::open(&path)?;
    let cfg = ModelConfig::from_gguf(&g)?;
    let tok = Tok::from_gguf(&g)?;
    let token_ids = tok.encode(&prompt)?;
    println!("Model: {} | layers={} d={} experts={} top_k={}",
        cfg.arch, cfg.n_layer, cfg.d_model, cfg.n_experts, cfg.top_k);
    println!("Prompt tokens ({}): {:?}\n", token_ids.len(), token_ids);

    let mut ec = EngineConfig::qwen3_30b();
    if let Ok(s) = std::env::var("POOL_SLOTS") {
        ec.vram_pool_slots = s.parse().expect("POOL_SLOTS must be u32");
    }
    let mut engine = Engine::new(&path, ec)?;
    println!("GPU: {} ({:.1} GB)", engine.ctx.gpu_name, engine.ctx.vram_size() as f64 / 1e9);

    let t0 = Instant::now();
    let weights = LoadedWeights::load(&engine.ctx, &g, &cfg, false)?;
    let kv = KvCache::new(&engine.ctx, &cfg, 512)?;
    let pipes = Pipelines::build(&engine.ctx)?;
    println!("Init: {:.2} s\n", t0.elapsed().as_secs_f64());

    let ctx_ptr: *const ssd_inference::device::VulkanContext = &engine.ctx;
    let ctx_ref = unsafe { &*ctx_ptr };
    let mut fwd = Forward::new(ctx_ref, &mut engine, &weights, &kv, &cfg, &pipes)?;

    // ── Prefill ──
    let t_prefill = Instant::now();
    for &tid in &token_ids {
        fwd.step(tid)?;
    }
    let first = fwd.get_next_token()?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    let first_str = tok.id_to_str(first).unwrap_or_else(|| "?".into());
    println!("Prefill: {:.0} ms ({} tokens, {:.1} ms/tok)",
        prefill_ms, token_ids.len(), prefill_ms / token_ids.len() as f64);

    // ── Decode loop ──
    print!("{}", prompt);
    print!("{}", first_str.replace('Ġ', " "));
    std::io::stdout().flush()?;

    let mut generated = vec![first];
    let t_decode = Instant::now();
    for _ in 1..max_tokens {
        let prev = *generated.last().unwrap();
        if prev == cfg.eos_token_id { break; }
        fwd.step(prev)?;
        let next = fwd.get_next_token()?;
        let s = tok.id_to_str(next).unwrap_or_else(|| "?".into());
        print!("{}", s.replace('Ġ', " "));
        std::io::stdout().flush()?;
        generated.push(next);
    }
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let n_dec = generated.len().saturating_sub(1);
    println!("\n\nDecode: {} tokens in {:.1} s ({:.2} ms/tok, {:.3} t/s)",
        n_dec, decode_ms / 1000.0,
        if n_dec > 0 { decode_ms / n_dec as f64 } else { 0.0 },
        if decode_ms > 0.0 { n_dec as f64 / decode_ms * 1000.0 } else { 0.0 });

    drop(fwd);
    pipes.destroy(&engine.ctx);
    kv.destroy(&engine.ctx);
    weights.destroy(&engine.ctx);
    Ok(())
}
