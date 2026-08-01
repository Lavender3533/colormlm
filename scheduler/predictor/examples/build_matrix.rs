//! Build a CooccurMatrix from a collector activations.bin file and save it.
//!
//! Usage:
//!   build_matrix <activations.bin> <out_matrix.bin> <n_experts> <n_layers>

use predictor::{ActivationRecord, MatrixBuilder, save_matrix};
use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let in_path = args.get(1).expect("usage: build_matrix <in.bin> <out.bin> <n_experts> <n_layers>");
    let out_path = args.get(2).expect("usage: build_matrix <in.bin> <out.bin> <n_experts> <n_layers>");
    let n_experts: u16 = args.get(3).expect("n_experts required").parse()?;
    let n_layers: u16 = args.get(4).expect("n_layers required").parse()?;

    println!("Loading activations from {}", in_path);
    let mut buf = Vec::new();
    File::open(in_path)?.read_to_end(&mut buf)?;
    let rec_size = std::mem::size_of::<ActivationRecord>();
    let n = buf.len() / rec_size;
    let records: &[ActivationRecord] = bytemuck::cast_slice(&buf[..n * rec_size]);
    println!("  {} records loaded", records.len());

    println!("Building matrix ({} layers × {} experts) ...", n_layers, n_experts);
    let builder = MatrixBuilder::new(n_layers, n_experts);
    let t0 = std::time::Instant::now();
    for r in records {
        builder.observe(r);
    }
    let snap = builder.build_snapshot();
    println!("  trained in {:.1} ms ({} obs)",
        t0.elapsed().as_secs_f64() * 1000.0, snap.total_observations());

    let nonzero = snap.counts().iter().filter(|&&c| c > 0).count();
    println!("  matrix coverage: {} / {} = {:.2}%",
        nonzero, snap.counts().len(),
        nonzero as f64 / snap.counts().len() as f64 * 100.0);

    println!("Saving to {}", out_path);
    save_matrix(&snap, out_path)?;
    let sz = std::fs::metadata(out_path)?.len();
    println!("  done. matrix file size: {} bytes ({:.2} MB)", sz, sz as f64 / 1_048_576.0);

    Ok(())
}
