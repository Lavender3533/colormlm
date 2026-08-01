//! GGUF-ish 简化二进制矩阵格式。
//!
//! 文件结构:
//!
//! ```text
//! 0x00  magic       8B   "MOEMTRX\0"
//! 0x08  version     u32  当前 = 1
//! 0x0C  n_layers    u16
//! 0x0E  n_experts   u16
//! 0x10  total_obs   u64  累计观测数(诊断用)
//! 0x18  reserved    8B   零
//! 0x20  counts[]         n_layers × n_experts × n_experts × u32
//!       row_totals[]     n_layers × n_experts × u32
//!       global_freq[]    n_experts × u32
//! ```
//!
//! 所有整数小端序。

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::matrix::CooccurMatrix;

const MAGIC: &[u8; 8] = b"MOEMTRX\0";
const VERSION: u32 = 1;

#[derive(Debug)]
pub enum FormatError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    SizeMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatError::Io(e) => write!(f, "io error: {}", e),
            FormatError::BadMagic => write!(f, "bad magic bytes (not a MOE matrix file)"),
            FormatError::UnsupportedVersion(v) => write!(f, "unsupported version: {}", v),
            FormatError::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for FormatError {}

impl From<std::io::Error> for FormatError {
    fn from(e: std::io::Error) -> Self { FormatError::Io(e) }
}

pub fn save<P: AsRef<Path>>(matrix: &CooccurMatrix, path: P) -> Result<(), FormatError> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&matrix.n_layers().to_le_bytes())?;
    w.write_all(&matrix.n_experts().to_le_bytes())?;
    w.write_all(&matrix.total_observations().to_le_bytes())?;
    w.write_all(&[0u8; 8])?;  // reserved

    write_u32_slice(&mut w, matrix.counts())?;
    write_u32_slice(&mut w, matrix.row_totals())?;
    write_u32_slice(&mut w, matrix.global_freq())?;
    w.flush()?;
    Ok(())
}

pub fn load<P: AsRef<Path>>(path: P) -> Result<CooccurMatrix, FormatError> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC { return Err(FormatError::BadMagic); }

    let version = read_u32(&mut r)?;
    if version != VERSION { return Err(FormatError::UnsupportedVersion(version)); }

    let n_layers = read_u16(&mut r)?;
    let n_experts = read_u16(&mut r)?;
    let _total_obs = read_u64(&mut r)?;  // diagnostic only
    let mut reserved = [0u8; 8];
    r.read_exact(&mut reserved)?;

    let l = n_layers as usize;
    let n = n_experts as usize;
    let counts = read_u32_box(&mut r, l * n * n)?;
    let row_totals = read_u32_box(&mut r, l * n)?;
    let global_freq = read_u32_box(&mut r, n)?;

    Ok(CooccurMatrix::from_parts(n_layers, n_experts, version as u64, counts, row_totals, global_freq))
}

fn write_u32_slice<W: Write>(w: &mut W, data: &[u32]) -> std::io::Result<()> {
    // Use bytemuck for zero-copy view of u32 slice as bytes (little-endian on x86)
    let bytes: &[u8] = bytemuck::cast_slice(data);
    w.write_all(bytes)
}

fn read_u32_box<R: Read>(r: &mut R, n: usize) -> Result<Box<[u32]>, FormatError> {
    let mut v = vec![0u32; n];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut v[..]);
    r.read_exact(bytes)?;
    Ok(v.into_boxed_slice())
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, FormatError> {
    let mut buf = [0u8; 2]; r.read_exact(&mut buf)?; Ok(u16::from_le_bytes(buf))
}
fn read_u32<R: Read>(r: &mut R) -> Result<u32, FormatError> {
    let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(u32::from_le_bytes(buf))
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64, FormatError> {
    let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatrixBuilder;
    use crate::record::ActivationRecord;
    use bytemuck::Zeroable;

    fn make_record(token: u32, layer: u16, experts: &[u16]) -> ActivationRecord {
        let mut r = ActivationRecord::zeroed();
        r.token_idx = token;
        r.layer = layer;
        r.n_experts_used = experts.len() as u8;
        for (i, &e) in experts.iter().enumerate() { r.expert_ids[i] = e; }
        r
    }

    #[test]
    fn roundtrip_in_temp_file() {
        let b = MatrixBuilder::new(4, 8);
        for tok in 0..50 {
            for layer in 0..4u16 {
                b.observe(&make_record(tok, layer, &[(layer + tok as u16) % 8, (layer + 1) % 8]));
            }
        }
        let m1 = b.build_snapshot();
        let path = std::env::temp_dir().join("moe_matrix_roundtrip_test.bin");
        save(&m1, &path).unwrap();
        let m2 = load(&path).unwrap();

        assert_eq!(m1.n_layers(), m2.n_layers());
        assert_eq!(m1.n_experts(), m2.n_experts());
        assert_eq!(m1.counts(), m2.counts());
        assert_eq!(m1.row_totals(), m2.row_totals());
        assert_eq!(m1.global_freq(), m2.global_freq());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_bad_magic() {
        let path = std::env::temp_dir().join("moe_matrix_bad_magic.bin");
        std::fs::write(&path, b"NOTAMTRX01234567").unwrap();
        let err = load(&path).unwrap_err();
        assert!(matches!(err, FormatError::BadMagic));
        let _ = std::fs::remove_file(path);
    }
}
