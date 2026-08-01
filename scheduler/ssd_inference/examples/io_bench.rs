//! Cold-read benchmark: which IO path comes closest to NVMe sustained 1.51 GB/s?
//!
//! Compares 3 paths reading expert byte ranges from a real GGUF file:
//!
//!   1. **mmap** (current `gguf_reader::expert_slot_bytes`) — reference baseline,
//!      4 KB page-fault per page, ~0.5 GB/s in 235B test.
//!   2. **File::seek_read** (single thread) — direct big-block syscall.
//!   3. **multi-threaded seek_read** — N threads, M reads each. Tests if
//!      parallel issue helps pre-saturate the NVMe queue depth.
//!
//! IMPORTANT: cache invalidation between runs is hard on Windows. We use
//! a 2 GB working set out of a 47 GB file, picked at random ranges far apart,
//! to keep page-cache hits low. Run multiple times if numbers look hot.
//!
//! Usage:
//!   cargo run --release -p ssd_inference --example io_bench -- \
//!     "../models/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf"

use anyhow::Result;
use gguf_reader::GgufFile;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const N_READS: usize = 16000;   // 16k × ~2 MB = 32 GB workload — exceeds 32 GB RAM
const N_EXPERTS: u32 = 128;

/// Build a list of (file_offset, byte_size) for expert reads, randomized
/// across all expert slots to defeat sequential prefetch.
fn build_workload(g: &GgufFile) -> Vec<(u64, usize)> {
    let exps = g.list_expert_tensors();
    let mut work = Vec::with_capacity(exps.len() * N_EXPERTS as usize);
    for ts in &exps {
        let per_slot = (ts.byte_size as usize) / N_EXPERTS as usize;
        for slot in 0..N_EXPERTS {
            let off = ts.byte_offset + (slot as u64) * per_slot as u64;
            work.push((off, per_slot));
        }
    }
    // shuffle deterministically (Fisher-Yates with LCG)
    let mut state: u64 = 0xABCD_DEFA;
    for i in (1..work.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state as usize) % (i + 1);
        work.swap(i, j);
    }
    work.into_iter().take(N_READS).collect()
}

fn bench_mmap(g: &GgufFile, work: &[(u64, usize)]) -> (f64, usize) {
    use std::hint::black_box;
    let t0 = Instant::now();
    let mut total = 0usize;
    for &(off, sz) in work {
        // Use the mmap directly via tensor_offset/byte_size? GgufFile only exposes
        // tensor_bytes(name). For raw offset reads we need a slice of the mmap;
        // expose via a tiny helper: we'll just read 4 KB strided to force page-in.
        // Instead: use the slot-bytes API via brute reconstruction would need names.
        // Cleanest: open a separate Mmap of the same file ourselves.
        // (skipped here, see bench_mmap_raw below)
        let _ = (off, sz);
        total += sz;
    }
    let dt = t0.elapsed().as_secs_f64();
    black_box(total);
    let _ = g; // silence
    (dt, total)
}

/// Mmap a file ourselves (bypassing GgufFile) to test pure mmap cold-read perf.
fn bench_mmap_raw(path: &PathBuf, work: &[(u64, usize)]) -> Result<(f64, usize)> {
    use memmap2::Mmap;
    let f = File::open(path)?;
    let mm = unsafe { Mmap::map(&f)? };
    let t0 = Instant::now();
    let mut total = 0usize;
    let mut sink: u64 = 0;
    for &(off, sz) in work {
        let s = &mm[off as usize..off as usize + sz];
        // Touch one byte per 4 KB page to force fault
        for i in (0..s.len()).step_by(4096) {
            sink = sink.wrapping_add(s[i] as u64);
        }
        total += sz;
    }
    std::hint::black_box(sink);
    Ok((t0.elapsed().as_secs_f64(), total))
}

fn bench_seek_read(path: &PathBuf, work: &[(u64, usize)]) -> Result<(f64, usize)> {
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let t0 = Instant::now();
    let mut total = 0usize;
    let mut sink: u64 = 0;
    for &(off, sz) in work {
        f.seek(SeekFrom::Start(off))?;
        f.read_exact(&mut buf[..sz])?;
        sink = sink.wrapping_add(buf[0] as u64).wrapping_add(buf[sz - 1] as u64);
        total += sz;
    }
    std::hint::black_box(sink);
    Ok((t0.elapsed().as_secs_f64(), total))
}

fn bench_seek_read_mt(path: &PathBuf, work: &[(u64, usize)], n_threads: usize) -> Result<(f64, usize)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    let path_arc: Arc<PathBuf> = Arc::new(path.clone());
    let work_arc: Arc<Vec<(u64, usize)>> = Arc::new(work.to_vec());
    let cursor = Arc::new(AtomicU64::new(0));
    let total_bytes = Arc::new(AtomicU64::new(0));

    let t0 = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..n_threads {
        let path = path_arc.clone();
        let work = work_arc.clone();
        let cursor = cursor.clone();
        let total_bytes = total_bytes.clone();
        handles.push(std::thread::spawn(move || -> Result<()> {
            let mut f = File::open(&*path)?;
            let mut buf = vec![0u8; 8 * 1024 * 1024];
            let mut sink: u64 = 0;
            loop {
                let i = cursor.fetch_add(1, Ordering::Relaxed) as usize;
                if i >= work.len() { break; }
                let (off, sz) = work[i];
                f.seek(SeekFrom::Start(off))?;
                f.read_exact(&mut buf[..sz])?;
                sink = sink.wrapping_add(buf[0] as u64);
                total_bytes.fetch_add(sz as u64, Ordering::Relaxed);
            }
            std::hint::black_box(sink);
            Ok(())
        }));
    }
    for h in handles { h.join().unwrap()?; }
    let dt = t0.elapsed().as_secs_f64();
    Ok((dt, total_bytes.load(Ordering::Relaxed) as usize))
}

fn report(label: &str, dt: f64, bytes: usize) {
    let mb = bytes as f64 / 1024.0 / 1024.0;
    let bw = bytes as f64 / dt / 1e9;
    println!("  {:<28}  {:>7.2} s  {:>7.0} MB  {:>5.2} GB/s", label, dt, mb, bw);
}

fn main() -> Result<()> {
    let path: PathBuf = std::env::args().nth(1)
        .unwrap_or_else(|| "../models/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf".to_string())
        .into();

    println!("=== Cold-read IO benchmark ===");
    println!("File: {}", path.display());

    let g = GgufFile::open(&path)?;
    let work = build_workload(&g);
    let total_mb: usize = work.iter().map(|w| w.1).sum::<usize>() / 1024 / 1024;
    println!("Workload: {} reads, total {} MB ({:.1} GB)\n",
        work.len(), total_mb, total_mb as f64 / 1024.0);

    println!("Reference: NVMe sustained read = 1.51 GB/s");
    println!("           Mmap 4K page-fault (235B test) = 0.47 GB/s\n");

    // Run benches; first run after init = cold (most likely), but Windows's
    // page cache is sticky — subsequent runs may show wildly different numbers.
    println!("--- mmap (raw, page-faulted) ---");
    let (dt, b) = bench_mmap_raw(&path, &work)?;
    report("mmap_raw", dt, b);

    println!("--- File::seek + read_exact (single thread) ---");
    let (dt, b) = bench_seek_read(&path, &work)?;
    report("seek_read[1]", dt, b);

    for nt in [2usize, 4, 8] {
        println!("--- File::seek + read_exact (multi-thread x{}) ---", nt);
        let (dt, b) = bench_seek_read_mt(&path, &work, nt)?;
        report(&format!("seek_read[{}]", nt), dt, b);
    }

    // suppress unused mmap-via-GgufFile bench
    let _ = bench_mmap;

    println!("\nNote: re-run several times to gauge cache stickiness.");
    println!("First run is closest to truly cold; later runs hit Windows page cache.");
    Ok(())
}
