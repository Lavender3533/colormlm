use anyhow::{bail, Context, Result};
use gguf_reader::GgufFile;
use memmap2::Mmap;
use std::time::Instant;
use std::{env, fs::File};

#[repr(C)]
struct MemoryRangeEntry {
    virtual_address: *const u8,
    number_of_bytes: usize,
}

extern "system" {
    fn PrefetchVirtualMemory(
        h_process: windows_sys::Win32::Foundation::HANDLE,
        number_of_entries: usize,
        virtual_addresses: *const MemoryRangeEntry,
        flags: u32,
    ) -> windows_sys::Win32::Foundation::BOOL;
}

fn main() -> Result<()> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf".into());

    eprintln!("[ssd-helper] opening {path}");
    let file = File::open(&path).context("open gguf")?;
    let mmap = unsafe { Mmap::map(&file)? };
    eprintln!("[ssd-helper] mmap'd {} MB", mmap.len() / (1024 * 1024));

    let gguf = GgufFile::open(&path).context("parse gguf")?;
    let experts = gguf.list_expert_tensors();
    if experts.is_empty() {
        bail!("no expert tensors found — is this a MoE model?");
    }

    let mut ranges: Vec<MemoryRangeEntry> = Vec::new();
    let mut total_bytes: u64 = 0;

    for ex in &experts {
        let offset = ex.byte_offset as usize;
        let size = ex.byte_size as usize;
        if offset + size > mmap.len() {
            eprintln!(
                "[ssd-helper] WARN: tensor {} offset+size exceeds file, skipping",
                ex.name
            );
            continue;
        }
        ranges.push(MemoryRangeEntry {
            virtual_address: unsafe { mmap.as_ptr().add(offset) },
            number_of_bytes: size,
        });
        total_bytes += ex.byte_size;
    }

    eprintln!(
        "[ssd-helper] found {} expert tensors, total {} MB",
        ranges.len(),
        total_bytes / (1024 * 1024)
    );

    eprintln!("[ssd-helper] prefetching...");
    let t0 = Instant::now();

    let batch_size = 64;
    let mut ok_count = 0usize;
    let mut fail_count = 0usize;

    for chunk in ranges.chunks(batch_size) {
        let ret = unsafe {
            PrefetchVirtualMemory(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                chunk.len(),
                chunk.as_ptr() as *const _,
                0,
            )
        };
        if ret != 0 {
            ok_count += chunk.len();
        } else {
            fail_count += chunk.len();
        }
    }

    let elapsed = t0.elapsed();
    let mb = total_bytes as f64 / (1024.0 * 1024.0);
    let secs = elapsed.as_secs_f64();

    eprintln!(
        "[ssd-helper] done in {:.2}s ({:.1} MB/s), ok={ok_count} fail={fail_count}",
        secs,
        mb / secs
    );
    eprintln!("[ssd-helper] page cache should now be warm — launch llama-cli");

    Ok(())
}
