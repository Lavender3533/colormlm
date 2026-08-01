use anyhow::Result;
use gguf_reader::MultiGgufFile;
use ssd_inference::device::VulkanContext;
use ssd_inference::model::ModelConfig;
use ssd_inference::streaming_weights::StreamingWeights;
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "../../models/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf".into());

    println!("Opening multi-shard: {}", path);
    let mg = MultiGgufFile::open(&path)?;
    let cfg = ModelConfig::from_multi_gguf(&mg)?;
    println!("Model: {} | layers={} d={} experts={}", cfg.arch, cfg.n_layer, cfg.d_model, cfg.n_experts);

    let ctx = VulkanContext::init()?;
    println!("GPU: {} ({:.1} GB)", ctx.gpu_name, ctx.vram_size() as f64 / 1e9);

    let t0 = Instant::now();
    let mut sw = StreamingWeights::new(&ctx, &mg, &cfg)?;
    println!("StreamingWeights init: {:.2}s\n", t0.elapsed().as_secs_f64());

    // Test uploading a few layers
    for l in [0, 47, 93] {
        let t = Instant::now();
        sw.upload_layer(&ctx, &mg, l)?;
        println!("  upload_layer({}) -> {:.1}ms", l, t.elapsed().as_secs_f64() * 1000.0);
    }

    // Upload all layers to measure average time
    let t_all = Instant::now();
    for l in 0..cfg.n_layer {
        sw.upload_layer(&ctx, &mg, l)?;
    }
    let total_ms = t_all.elapsed().as_secs_f64() * 1000.0;
    println!("\n  All {} layers uploaded in {:.0}ms ({:.1}ms/layer)",
        cfg.n_layer, total_ms, total_ms / cfg.n_layer as f64);

    sw.destroy(&ctx);
    println!("\nOK!");
    Ok(())
}
