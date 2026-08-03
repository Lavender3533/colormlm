//! Polaris S14 本地 embedding safetensors shard 的权威只读解析器。
//!
//! Production chat 在启动期一次性验证完整 shard 身份，之后只从同一只读 mmap
//! 借用 token 对应的 8 KiB BF16 行。这里不创建 Range cache、不启动下载器，也不
//! 接受其他 safetensors 布局冒充当前冻结的 `embed.weight`。

use anyhow::{bail, Context, Result};
use memmap2::{Mmap, MmapOptions};
use polaris_s14_runner::VOCAB_SIZE;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    ffi::OsStr,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};

pub const S14_EMBEDDING_SHARD_FILE_NAME: &str = "model-00001-of-00048.safetensors";
pub const S14_EMBEDDING_SHARD_BYTES: u64 = 1_059_061_856;
pub const S14_EMBEDDING_SHARD_SHA256: &str =
    "f3668ba4cccf1ca6a7eb84e888fb92c1cdc7204d472ba9db771e6fd3abf6b874";
pub const S14_EMBEDDING_HEADER_BYTES: usize = 88;
pub const S14_EMBEDDING_PAYLOAD_START: usize = 96;
pub const S14_EMBEDDING_PAYLOAD_END: usize = 1_059_061_856;
pub const S14_EMBEDDING_ROW_BYTES: usize = 8_192;
pub const S14_EMBEDDING_ROWS: u32 = 129_280;

/// 启动期验签后常驻 factory 的只读 owner。`File` 与 `Mmap` 一起保留，确保每个
/// request/block 都借用同一个已验证文件映射，而不是按 token 重新打开路径。
pub struct S14LocalEmbeddingShard {
    path: PathBuf,
    _file: File,
    mmap: Mmap,
}

impl std::fmt::Debug for S14LocalEmbeddingShard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("S14LocalEmbeddingShard")
            .field("path", &self.path)
            .field("bytes", &self.mmap.len())
            .field("sha256", &S14_EMBEDDING_SHARD_SHA256)
            .finish()
    }
}

impl S14LocalEmbeddingShard {
    /// 打开并验签唯一允许的本地 shard。文件缺失、文件名/长度/header/payload/SHA
    /// 任一漂移都会在 production root ready 之前 fail-closed。
    pub fn open_verified(path: impl AsRef<Path>) -> Result<Self> {
        let requested = path.as_ref();
        if requested.file_name() != Some(OsStr::new(S14_EMBEDDING_SHARD_FILE_NAME)) {
            bail!(
                "S14 embedding shard 文件名必须为 {S14_EMBEDDING_SHARD_FILE_NAME}: {}",
                requested.display()
            );
        }
        let path = requested.canonicalize().with_context(|| {
            format!(
                "S14 本地 embedding shard 缺失或路径不可访问: {}",
                requested.display()
            )
        })?;
        let file = open_immutable_read(&path)
            .with_context(|| format!("只读打开 S14 embedding shard: {}", path.display()))?;
        let metadata_before = file
            .metadata()
            .with_context(|| format!("读取 S14 embedding shard metadata: {}", path.display()))?;
        if !metadata_before.is_file() || metadata_before.len() != S14_EMBEDDING_SHARD_BYTES {
            bail!(
                "S14 embedding shard size/type 漂移: path={} expected={} actual={}",
                path.display(),
                S14_EMBEDDING_SHARD_BYTES,
                metadata_before.len()
            );
        }

        // SAFETY: file 以只读方式打开，长度已冻结；owner 同时保留 File 与只读 Mmap，
        // 不向调用方暴露可写映射或裸指针。
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("只读 mmap S14 embedding shard: {}", path.display()))?;
        if mmap.len() as u64 != S14_EMBEDDING_SHARD_BYTES {
            bail!("S14 embedding shard mmap 长度漂移");
        }
        validate_safetensors_header(&mmap)?;

        let actual_sha256 = format!("{:x}", Sha256::digest(&mmap[..]));
        if actual_sha256 != S14_EMBEDDING_SHARD_SHA256 {
            bail!(
                "S14 embedding shard SHA-256 失败: expected={S14_EMBEDDING_SHARD_SHA256} actual={actual_sha256}"
            );
        }
        let metadata_after = file
            .metadata()
            .context("SHA-256 后重读 S14 embedding shard metadata")?;
        if metadata_after.len() != metadata_before.len()
            || metadata_after.modified().ok() != metadata_before.modified().ok()
        {
            bail!("S14 embedding shard 在完整 SHA-256 期间发生变化");
        }

        Ok(Self {
            path,
            _file: file,
            mmap,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn row(&self, token_id: u32) -> Result<&[u8]> {
        if token_id >= S14_EMBEDDING_ROWS || token_id >= VOCAB_SIZE {
            bail!("S14 embedding token {token_id} 越出冻结 vocab");
        }
        let start = usize::try_from(token_id)
            .ok()
            .and_then(|token| token.checked_mul(S14_EMBEDDING_ROW_BYTES))
            .and_then(|offset| S14_EMBEDDING_PAYLOAD_START.checked_add(offset))
            .context("S14 embedding row start overflow")?;
        let end = start
            .checked_add(S14_EMBEDDING_ROW_BYTES)
            .context("S14 embedding row end overflow")?;
        if end > S14_EMBEDDING_PAYLOAD_END {
            bail!("S14 embedding token {token_id} 越出冻结 payload");
        }
        self.mmap
            .get(start..end)
            .context("S14 embedding mmap row 越界")
    }
}

#[cfg(windows)]
fn open_immutable_read(path: &Path) -> Result<File> {
    use std::os::windows::fs::OpenOptionsExt;

    // 与 VerifiedMappedAssetStore 相同：只允许其他句柄并发读取。owner 存活期间
    // 拒绝写入和删除，避免整文件 SHA 通过后 mmap 内容又被原地替换。
    OpenOptions::new()
        .read(true)
        .share_mode(0x0000_0001)
        .open(path)
        .with_context(|| format!("open immutable S14 embedding shard {}", path.display()))
}

#[cfg(not(windows))]
fn open_immutable_read(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .open(path)
        .with_context(|| format!("open S14 embedding shard {}", path.display()))
}

fn validate_safetensors_header(mmap: &[u8]) -> Result<()> {
    if mmap.len() != S14_EMBEDDING_PAYLOAD_END
        || S14_EMBEDDING_PAYLOAD_END as u64 != S14_EMBEDDING_SHARD_BYTES
        || S14_EMBEDDING_PAYLOAD_START != 8 + S14_EMBEDDING_HEADER_BYTES
        || S14_EMBEDDING_ROWS != VOCAB_SIZE
        || u64::from(S14_EMBEDDING_ROWS) * S14_EMBEDDING_ROW_BYTES as u64
            != (S14_EMBEDDING_PAYLOAD_END - S14_EMBEDDING_PAYLOAD_START) as u64
    {
        bail!("S14 embedding shard 冻结常量合同漂移");
    }
    let header_len = u64::from_le_bytes(
        mmap.get(..8)
            .context("S14 embedding shard 缺少 header length")?
            .try_into()
            .context("S14 embedding header length ABI 漂移")?,
    );
    if header_len != S14_EMBEDDING_HEADER_BYTES as u64 {
        bail!(
            "S14 embedding safetensors header 长度漂移: expected={} actual={header_len}",
            S14_EMBEDDING_HEADER_BYTES
        );
    }
    let header: Value = serde_json::from_slice(
        mmap.get(8..S14_EMBEDDING_PAYLOAD_START)
            .context("S14 embedding safetensors header 越界")?,
    )
    .context("解析 S14 embedding safetensors header")?;
    let root = header
        .as_object()
        .context("S14 embedding safetensors header 不是对象")?;
    if root.len() != 1 {
        bail!("S14 embedding safetensors 只允许唯一 embed.weight tensor");
    }
    let tensor = root
        .get("embed.weight")
        .and_then(Value::as_object)
        .context("S14 embedding safetensors 缺少 embed.weight")?;
    let dtype_ok = tensor.get("dtype").and_then(Value::as_str) == Some("BF16");
    let shape_ok = tensor
        .get("shape")
        .and_then(Value::as_array)
        .is_some_and(|shape| {
            shape.len() == 2
                && shape[0].as_u64() == Some(u64::from(S14_EMBEDDING_ROWS))
                && shape[1].as_u64() == Some(4_096)
        });
    let offsets_ok = tensor
        .get("data_offsets")
        .and_then(Value::as_array)
        .is_some_and(|offsets| {
            offsets.len() == 2
                && offsets[0].as_u64() == Some(0)
                && offsets[1].as_u64()
                    == Some((S14_EMBEDDING_PAYLOAD_END - S14_EMBEDDING_PAYLOAD_START) as u64)
        });
    if tensor.len() != 3 || !dtype_ok || !shape_ok || !offsets_ok {
        bail!("S14 embed.weight dtype/shape/data_offsets 合同漂移");
    }
    Ok(())
}
