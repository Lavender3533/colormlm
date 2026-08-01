//! Per-expert GGUF reader: mmap full file, expose tensor offsets,
//! provide byte-range views for on-demand expert loading.
//!
//! Built on top of `candle_core::quantized::gguf_file::Content` for header
//! parsing (handles all GGUF v3 quantization formats) but strips out
//! candle's tensor materialization — we want raw bytes only, GPU dequants.

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::gguf_file::{Content, TensorInfo};
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

// Re-export so downstream crates don't need to depend on candle directly.
pub use candle_core::quantized::gguf_file::Value as MetaValue;

pub struct GgufFile {
    pub path: PathBuf,
    mmap: Mmap,
    content: Content,
    /// Absolute file offset where tensor data section begins (after metadata,
    /// padded to alignment). Per-tensor `info.offset` is relative to this.
    data_start: u64,
}

#[derive(Debug, Clone)]
pub struct ExpertSlice {
    pub layer: u32,
    pub expert: u32,
    pub kind: ExpertKind,
    pub name: String,
    pub byte_offset: u64,
    pub byte_size: u64,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertKind {
    /// `blk.{L}.ffn_gate_exps.weight` — packed 3D `[hidden, intermediate, n_experts]`
    GateExps,
    /// `blk.{L}.ffn_up_exps.weight`
    UpExps,
    /// `blk.{L}.ffn_down_exps.weight`
    DownExps,
}

impl ExpertKind {
    fn from_name(name: &str) -> Option<Self> {
        if name.ends_with("ffn_gate_exps.weight") { Some(Self::GateExps) }
        else if name.ends_with("ffn_up_exps.weight") { Some(Self::UpExps) }
        else if name.ends_with("ffn_down_exps.weight") { Some(Self::DownExps) }
        else { None }
    }
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| format!("open {:?}", path))?;
        let mmap = unsafe { Mmap::map(&file)? };

        // candle reads via &mut File (it seeks). We use a separate File handle
        // here so the mmap stays a stable read-only view.
        let mut header_file = File::open(&path)?;
        let content = Content::read(&mut header_file)
            .map_err(|e| anyhow!("gguf header parse failed: {e}"))?;

        // tensor_data_offset is the absolute file position of the data section.
        // Available since candle 0.4+; see candle-core/src/quantized/gguf_file.rs.
        let data_start = content.tensor_data_offset;

        Ok(Self { path, mmap, content, data_start })
    }

    pub fn metadata_keys(&self) -> Vec<&String> {
        self.content.metadata.keys().collect()
    }

    /// Typed metadata access. Returns a `MetaValue` (re-exported candle enum).
    pub fn metadata_value(&self, key: &str) -> Option<&MetaValue> {
        self.content.metadata.get(key)
    }

    /// Get a u32 metadata value (auto-upcasts smaller uint types via candle's `to_u64`).
    pub fn metadata_u32(&self, key: &str) -> Result<u32> {
        let v = self.content.metadata.get(key)
            .ok_or_else(|| anyhow!("metadata key not found: {key}"))?;
        // Use to_u64 first (handles auto-upcast from u8/u16/u32/bool), then narrow.
        let n = v.to_u64().map_err(|e| anyhow!("metadata {key} not u32-compatible: {e}"))?;
        if n > u32::MAX as u64 { bail!("metadata {key} = {n} overflows u32"); }
        Ok(n as u32)
    }

    /// Get an f32 metadata value.
    pub fn metadata_f32(&self, key: &str) -> Result<f32> {
        let v = self.content.metadata.get(key)
            .ok_or_else(|| anyhow!("metadata key not found: {key}"))?;
        v.to_f32().map_err(|e| anyhow!("metadata {key} not f32: {e}"))
    }

    /// Get a string metadata value (cloned).
    pub fn metadata_string(&self, key: &str) -> Result<String> {
        let v = self.content.metadata.get(key)
            .ok_or_else(|| anyhow!("metadata key not found: {key}"))?;
        v.to_string()
            .map(|s| s.clone())
            .map_err(|e| anyhow!("metadata {key} not string: {e}"))
    }

    /// Get a string-array metadata value (e.g. `tokenizer.ggml.tokens`).
    pub fn metadata_string_array(&self, key: &str) -> Result<Vec<String>> {
        let v = self.content.metadata.get(key)
            .ok_or_else(|| anyhow!("metadata key not found: {key}"))?;
        let arr = v.to_vec().map_err(|e| anyhow!("metadata {key} not array: {e}"))?;
        let mut out = Vec::with_capacity(arr.len());
        for (i, item) in arr.iter().enumerate() {
            let s = item.to_string()
                .map_err(|e| anyhow!("metadata {key}[{i}] not string: {e}"))?;
            out.push(s.clone());
        }
        Ok(out)
    }

    pub fn n_tensors(&self) -> usize { self.content.tensor_infos.len() }

    pub fn tensor_names(&self) -> Vec<&String> {
        self.content.tensor_infos.keys().collect()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.content.tensor_infos.get(name)
    }

    /// Absolute file offset for a tensor's data.
    pub fn tensor_offset(&self, name: &str) -> Option<u64> {
        self.content.tensor_infos.get(name).map(|t| self.data_start + t.offset)
    }

    /// Total size (bytes) for a tensor in its on-disk quantized representation.
    pub fn tensor_byte_size(&self, name: &str) -> Option<u64> {
        let info = self.content.tensor_infos.get(name)?;
        let n = info.shape.elem_count();
        let dt = info.ggml_dtype;
        let block = dt.block_size();
        let block_bytes = dt.type_size();
        let n_blocks = (n + block - 1) / block; // ceil
        Some((n_blocks * block_bytes) as u64)
    }

    /// Get a raw byte slice for any tensor by name. Zero-copy view into mmap.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let off = self.tensor_offset(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let sz = self.tensor_byte_size(name).unwrap();
        let start = off as usize;
        let end = start + sz as usize;
        if end > self.mmap.len() {
            bail!("tensor {name} extends past file end ({end} > {})", self.mmap.len());
        }
        Ok(&self.mmap[start..end])
    }

    /// Enumerate all packed expert tensors. For each MoE layer, returns one
    /// entry per (layer, expert_kind) — the expert dimension is *inside* the
    /// tensor (slot 0..n_experts along axis 2). Use `expert_slot_bytes` to
    /// further slice into individual experts.
    pub fn list_expert_tensors(&self) -> Vec<ExpertSlice> {
        let mut out = Vec::new();
        for (name, info) in &self.content.tensor_infos {
            let Some(kind) = ExpertKind::from_name(name) else { continue };
            // name format: "blk.{L}.ffn_*_exps.weight"
            let layer: u32 = name
                .strip_prefix("blk.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(u32::MAX);
            let off = self.data_start + info.offset;
            let sz = self.tensor_byte_size(name).unwrap();
            out.push(ExpertSlice {
                layer,
                expert: u32::MAX, // packed, real expert dim is inside
                kind,
                name: name.clone(),
                byte_offset: off,
                byte_size: sz,
                shape: info.shape.dims().to_vec(),
            });
        }
        out.sort_by_key(|e| (e.layer, e.kind as u8));
        out
    }

    /// Get the byte slice for a single expert slot inside a packed expert tensor.
    /// Assumes the expert dimension is the *last* axis (GGUF convention for MoE
    /// tensors: `[..hidden.., n_experts]`), so slot bytes are contiguous.
    pub fn expert_slot_bytes(&self, layer: u32, kind: ExpertKind, slot: u32, n_experts: u32) -> Result<&[u8]> {
        let name = match kind {
            ExpertKind::GateExps => format!("blk.{layer}.ffn_gate_exps.weight"),
            ExpertKind::UpExps   => format!("blk.{layer}.ffn_up_exps.weight"),
            ExpertKind::DownExps => format!("blk.{layer}.ffn_down_exps.weight"),
        };
        let total = self.tensor_bytes(&name)?;
        let total_len = total.len();
        if total_len % n_experts as usize != 0 {
            bail!("expert tensor {name} byte size {total_len} not divisible by n_experts {n_experts}");
        }
        let per_slot = total_len / n_experts as usize;
        let start = slot as usize * per_slot;
        let end = start + per_slot;
        Ok(&total[start..end])
    }

    pub fn file_size(&self) -> u64 { self.mmap.len() as u64 }
    pub fn data_start(&self) -> u64 { self.data_start }
}

/// Multi-shard GGUF: merges tensor lookups across N split files.
/// Metadata comes from shard 0 only. Each tensor is mapped to its shard.
pub struct MultiGgufFile {
    shards: Vec<GgufFile>,
    /// tensor_name → shard index
    tensor_shard: std::collections::HashMap<String, usize>,
}

impl MultiGgufFile {
    /// Open a split GGUF. Pass the path to shard 1 (the `-00001-of-00002.gguf`).
    /// Automatically discovers other shards by replacing the shard number.
    pub fn open(shard0_path: impl AsRef<Path>) -> Result<Self> {
        let p = shard0_path.as_ref().to_string_lossy().to_string();

        // Detect split pattern: -00001-of-NNNNN.gguf
        let re_pat = "-00001-of-";
        if let Some(pos) = p.find(re_pat) {
            let prefix = &p[..pos];
            let after_of = &p[pos + re_pat.len()..];
            let n_shards: usize = after_of.split('.').next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);

            let mut shards = Vec::with_capacity(n_shards);
            let mut tensor_shard = std::collections::HashMap::new();

            for i in 0..n_shards {
                let shard_path = format!("{}-{:05}-of-{:05}.gguf", prefix, i + 1, n_shards);
                let g = GgufFile::open(&shard_path)
                    .with_context(|| format!("opening shard {}", shard_path))?;
                for name in g.tensor_names() {
                    tensor_shard.insert(name.clone(), i);
                }
                shards.push(g);
            }

            Ok(Self { shards, tensor_shard })
        } else {
            // Single file, no split
            let g = GgufFile::open(&p)?;
            let mut tensor_shard = std::collections::HashMap::new();
            for name in g.tensor_names() {
                tensor_shard.insert(name.clone(), 0);
            }
            Ok(Self { shards: vec![g], tensor_shard })
        }
    }

    pub fn shard(&self, idx: usize) -> &GgufFile { &self.shards[idx] }
    pub fn n_shards(&self) -> usize { self.shards.len() }

    /// Metadata access (from shard 0).
    pub fn metadata_value(&self, key: &str) -> Option<&MetaValue> {
        self.shards[0].metadata_value(key)
    }
    pub fn metadata_u32(&self, key: &str) -> Result<u32> { self.shards[0].metadata_u32(key) }
    pub fn metadata_f32(&self, key: &str) -> Result<f32> { self.shards[0].metadata_f32(key) }
    pub fn metadata_string(&self, key: &str) -> Result<String> { self.shards[0].metadata_string(key) }
    pub fn metadata_keys(&self) -> Vec<&String> { self.shards[0].metadata_keys() }

    /// Find which shard has this tensor + return its info.
    pub fn tensor_info(&self, name: &str) -> Option<&candle_core::quantized::gguf_file::TensorInfo> {
        let &si = self.tensor_shard.get(name)?;
        self.shards[si].tensor_info(name)
    }

    /// Get raw bytes for a tensor (routes to correct shard's mmap).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let &si = self.tensor_shard.get(name)
            .ok_or_else(|| anyhow!("tensor not found in any shard: {name}"))?;
        self.shards[si].tensor_bytes(name)
    }

    /// Absolute file offset of a tensor within its shard file.
    pub fn tensor_offset(&self, name: &str) -> Option<(usize, u64)> {
        let &si = self.tensor_shard.get(name)?;
        self.shards[si].tensor_offset(name).map(|off| (si, off))
    }

    pub fn tensor_byte_size(&self, name: &str) -> Option<u64> {
        let &si = self.tensor_shard.get(name)?;
        self.shards[si].tensor_byte_size(name)
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensor_shard.keys().cloned().collect()
    }

    /// List all expert tensors across all shards.
    pub fn list_expert_tensors(&self) -> Vec<ExpertSlice> {
        let mut out = Vec::new();
        for g in &self.shards {
            out.extend(g.list_expert_tensors());
        }
        out.sort_by_key(|e| (e.layer, e.kind as u8));
        out
    }

    /// Get the shard index and GgufFile path for a given tensor.
    pub fn tensor_shard_path(&self, name: &str) -> Option<(&Path, usize)> {
        let &si = self.tensor_shard.get(name)?;
        Some((self.shards[si].path.as_path(), si))
    }
}
