//! File-backed expert reader: bypasses mmap to avoid 4 KB page-fault penalty.
//!
//! Holds metadata (tensor name → byte offset + size) and reads via
//! `File::seek + read_exact`. Supports multi-shard GGUF files.

use anyhow::{anyhow, bail, Result};
use gguf_reader::{ExpertKind, GgufFile, MultiGgufFile};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ExpertSlotKey {
    layer: u32,
    kind: u8,
    slot: u32,
}

#[derive(Clone, Copy, Debug)]
struct ExpertLoc {
    off: u64,
    size: usize,
    shard: usize,
}

pub struct ExpertReader {
    paths: Vec<PathBuf>,
    files: Vec<Mutex<File>>,
    locs: HashMap<ExpertSlotKey, ExpertLoc>,
    n_experts_total: u32,
}

impl ExpertReader {
    pub fn from_gguf(gguf: &GgufFile, path: impl Into<PathBuf>, n_experts_total: u32) -> Result<Self> {
        let path = path.into();
        let file = File::open(&path)?;
        let mut locs = HashMap::new();
        for ts in gguf.list_expert_tensors() {
            let total = ts.byte_size as usize;
            if total % n_experts_total as usize != 0 {
                bail!("tensor {} byte_size {} not divisible by n_experts {}",
                      ts.name, total, n_experts_total);
            }
            let per_slot = total / n_experts_total as usize;
            for slot in 0..n_experts_total {
                let key = ExpertSlotKey { layer: ts.layer, kind: ts.kind as u8, slot };
                let off = ts.byte_offset + (slot as u64) * (per_slot as u64);
                locs.insert(key, ExpertLoc { off, size: per_slot, shard: 0 });
            }
        }
        Ok(Self { paths: vec![path], files: vec![Mutex::new(file)], locs, n_experts_total })
    }

    pub fn from_multi_gguf(mg: &MultiGgufFile, n_experts_total: u32) -> Result<Self> {
        let mut paths = Vec::new();
        let mut files = Vec::new();
        let mut locs = HashMap::new();
        for si in 0..mg.n_shards() {
            let g = mg.shard(si);
            paths.push(g.path.clone());
            files.push(Mutex::new(File::open(&g.path)?));
            for ts in g.list_expert_tensors() {
                let total = ts.byte_size as usize;
                if total % n_experts_total as usize != 0 {
                    bail!("tensor {} byte_size {} not divisible by n_experts {}",
                          ts.name, total, n_experts_total);
                }
                let per_slot = total / n_experts_total as usize;
                for slot in 0..n_experts_total {
                    let key = ExpertSlotKey { layer: ts.layer, kind: ts.kind as u8, slot };
                    let off = ts.byte_offset + (slot as u64) * (per_slot as u64);
                    locs.insert(key, ExpertLoc { off, size: per_slot, shard: si });
                }
            }
        }
        Ok(Self { paths, files, locs, n_experts_total })
    }

    pub fn n_experts_total(&self) -> u32 { self.n_experts_total }

    pub fn read_into(&self, layer: u32, kind: ExpertKind, slot: u32, dest: &mut [u8]) -> Result<()> {
        let key = ExpertSlotKey { layer, kind: kind as u8, slot };
        let loc = self.locs.get(&key)
            .ok_or_else(|| anyhow!("no expert slot ({}, {:?}, {})", layer, kind, slot))?;
        if dest.len() != loc.size {
            bail!("dest buffer {} != expert size {}", dest.len(), loc.size);
        }
        let mut f = self.files[loc.shard].lock();
        f.seek(SeekFrom::Start(loc.off))?;
        f.read_exact(dest)?;
        Ok(())
    }

    pub fn expert_size(&self, layer: u32, kind: ExpertKind, slot: u32) -> Option<usize> {
        self.locs.get(&ExpertSlotKey { layer, kind: kind as u8, slot }).map(|l| l.size)
    }

    pub fn path(&self) -> &PathBuf { &self.paths[0] }
    pub fn n_experts_indexed(&self) -> usize { self.locs.len() }

    pub fn expert_loc(&self, layer: u32, kind: ExpertKind, slot: u32) -> Option<(u64, usize)> {
        self.locs.get(&ExpertSlotKey { layer, kind: kind as u8, slot })
            .map(|l| (l.off, l.size))
    }

    pub fn expert_shard(&self, layer: u32, kind: ExpertKind, slot: u32) -> Option<usize> {
        self.locs.get(&ExpertSlotKey { layer, kind: kind as u8, slot })
            .map(|l| l.shard)
    }

    pub fn shard_path(&self, shard: usize) -> &PathBuf { &self.paths[shard] }

    pub fn parallel_read(&self, requests: &[(u32, ExpertKind, u32)]) -> Result<Vec<(u32, ExpertKind, u32, Vec<u8>)>> {
        requests.par_iter()
            .map(|&(layer, kind, slot)| {
                let key = ExpertSlotKey { layer, kind: kind as u8, slot };
                let loc = self.locs.get(&key)
                    .ok_or_else(|| anyhow!("no expert slot ({}, {:?}, {})", layer, kind, slot))?;
                let mut buf = vec![0u8; loc.size];
                let mut f = File::open(&self.paths[loc.shard])?;
                f.seek(SeekFrom::Start(loc.off))?;
                f.read_exact(&mut buf)?;
                Ok((layer, kind, slot, buf))
            })
            .collect()
    }
}

pub struct MultiFileExpertReader {
    paths: Vec<PathBuf>,
    files: Vec<Mutex<File>>,
    locs: HashMap<ExpertSlotKey, ExpertLoc>,
    n_experts_total: u32,
    next_handle: std::sync::atomic::AtomicUsize,
}

impl MultiFileExpertReader {
    pub fn from_gguf(gguf: &GgufFile, path: impl Into<PathBuf>, n_experts_total: u32, n_handles: usize) -> Result<Self> {
        let path = path.into();
        let mut files = Vec::with_capacity(n_handles);
        for _ in 0..n_handles {
            files.push(Mutex::new(File::open(&path)?));
        }
        let mut locs = HashMap::new();
        for ts in gguf.list_expert_tensors() {
            let total = ts.byte_size as usize;
            if total % n_experts_total as usize != 0 {
                bail!("tensor {} byte_size {} not divisible by n_experts {}",
                      ts.name, total, n_experts_total);
            }
            let per_slot = total / n_experts_total as usize;
            for slot in 0..n_experts_total {
                let key = ExpertSlotKey { layer: ts.layer, kind: ts.kind as u8, slot };
                let off = ts.byte_offset + (slot as u64) * (per_slot as u64);
                locs.insert(key, ExpertLoc { off, size: per_slot, shard: 0 });
            }
        }
        Ok(Self {
            paths: vec![path], files, locs, n_experts_total,
            next_handle: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn read_into(&self, layer: u32, kind: ExpertKind, slot: u32, dest: &mut [u8]) -> Result<()> {
        use std::sync::atomic::Ordering;
        let key = ExpertSlotKey { layer, kind: kind as u8, slot };
        let loc = self.locs.get(&key)
            .ok_or_else(|| anyhow!("no expert slot ({}, {:?}, {})", layer, kind, slot))?;
        if dest.len() != loc.size {
            bail!("dest buffer {} != expert size {}", dest.len(), loc.size);
        }
        let h = self.next_handle.fetch_add(1, Ordering::Relaxed) % self.files.len();
        let mut f = self.files[h].lock();
        f.seek(SeekFrom::Start(loc.off))?;
        f.read_exact(dest)?;
        Ok(())
    }

    pub fn expert_size(&self, layer: u32, kind: ExpertKind, slot: u32) -> Option<usize> {
        self.locs.get(&ExpertSlotKey { layer, kind: kind as u8, slot }).map(|l| l.size)
    }

    pub fn n_handles(&self) -> usize { self.files.len() }
    pub fn n_experts_total(&self) -> u32 { self.n_experts_total }
}
