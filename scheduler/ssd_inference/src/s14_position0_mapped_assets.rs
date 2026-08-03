//! FullDepth43 position0 payload 的进程内只读映射与一次性 SHA 门。
//!
//! 11.24 GB 资产不能每 token 重新 `read + SHA + Vec`。本模块第一次看到
//! path/bytes/SHA 身份时打开只读文件、建立 mmap 并完整校验；成功后保存文件句柄与
//! 映射，后续 token 只复用同一不可变 lease。Windows 上文件句柄仅共享 READ，阻止
//! 当前进程存活期间的写入和删除，避免“校验后替换”污染热路径。

use anyhow::{bail, Context, Result};
use memmap2::{Mmap, MmapOptions};
use polaris_s14_runner::Position0Asset;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MappedAssetKey {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Debug)]
pub struct VerifiedMappedAsset {
    key: MappedAssetKey,
    tensor: String,
    #[allow(dead_code)]
    file: File,
    mmap: Mmap,
}

impl VerifiedMappedAsset {
    pub fn tensor(&self) -> &str {
        &self.tensor
    }

    pub fn path(&self) -> &Path {
        &self.key.path
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }

    pub fn expected_sha256(&self) -> &str {
        &self.key.sha256
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerifiedMappedAssetStats {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    /// 唯一映射的逻辑字节。它是虚拟映射规模，不等于物理 RAM 常驻量。
    pub mapped_logical_bytes: u64,
    pub sha256_bytes: u64,
}

#[derive(Debug)]
pub struct VerifiedMappedAssetStore {
    allowed_root: PathBuf,
    entries: HashMap<MappedAssetKey, Arc<VerifiedMappedAsset>>,
    stats: VerifiedMappedAssetStats,
}

impl VerifiedMappedAssetStore {
    pub fn new(allowed_root: &Path) -> Result<Self> {
        let allowed_root = allowed_root
            .canonicalize()
            .with_context(|| format!("resolve payload root {}", allowed_root.display()))?;
        if !allowed_root.is_dir() {
            bail!(
                "position0 payload root 不是目录: {}",
                allowed_root.display()
            );
        }
        Ok(Self {
            allowed_root,
            entries: HashMap::new(),
            stats: VerifiedMappedAssetStats::default(),
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> VerifiedMappedAssetStats {
        self.stats
    }

    /// 按请求顺序返回 lease。所有 miss 均验证成功后才原子发布到 store；任一资产失败，
    /// 本批新映射全部丢弃，旧的可信 entries 保持不变。
    pub fn map_verified_batch(
        &mut self,
        assets: &[Position0Asset],
    ) -> Result<Vec<Arc<VerifiedMappedAsset>>> {
        if assets.is_empty() {
            bail!("position0 mapped asset batch 不能为空");
        }
        let mut keys = Vec::with_capacity(assets.len());
        let mut seen = HashSet::with_capacity(assets.len());
        for asset in assets {
            let canonical = asset
                .path
                .canonicalize()
                .with_context(|| format!("resolve position0 payload {}", asset.path.display()))?;
            if !canonical.starts_with(&self.allowed_root) {
                bail!(
                    "position0 payload 越出允许根目录: root={} path={}",
                    self.allowed_root.display(),
                    canonical.display()
                );
            }
            validate_sha256(&asset.sha256)?;
            let key = MappedAssetKey {
                path: canonical,
                bytes: asset.bytes,
                sha256: asset.sha256.clone(),
            };
            if !seen.insert(key.clone()) {
                bail!("position0 mapped batch 含重复 path/bytes/SHA 身份");
            }
            keys.push(key);
        }

        self.stats.requests = self
            .stats
            .requests
            .checked_add(assets.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("mapped asset request counter overflow"))?;
        let mut output = vec![None; assets.len()];
        let mut misses = Vec::new();
        for (index, (asset, key)) in assets.iter().zip(&keys).enumerate() {
            if let Some(existing) = self.entries.get(key) {
                if existing.tensor != asset.tensor {
                    bail!("同一 payload 身份被不同 tensor 名复用");
                }
                output[index] = Some(Arc::clone(existing));
                self.stats.hits += 1;
            } else {
                misses.push((index, asset.tensor.clone(), key.clone()));
                self.stats.misses += 1;
            }
        }

        let opened: Vec<Result<(usize, Arc<VerifiedMappedAsset>)>> = misses
            .into_par_iter()
            .map(|(index, tensor, key)| {
                let file = open_immutable_read(&key.path)?;
                let metadata = file
                    .metadata()
                    .with_context(|| format!("stat mapped payload {}", key.path.display()))?;
                if !metadata.is_file() || metadata.len() != key.bytes {
                    bail!(
                        "mapped payload 字节漂移: {} expected={} actual={}",
                        key.path.display(),
                        key.bytes,
                        metadata.len()
                    );
                }
                let mmap = unsafe { MmapOptions::new().map(&file) }
                    .with_context(|| format!("mmap payload {}", key.path.display()))?;
                let observed = sha256_bytes(&mmap);
                if observed != key.sha256 {
                    bail!(
                        "mapped payload SHA-256 漂移: {} expected={} actual={}",
                        key.path.display(),
                        key.sha256,
                        observed
                    );
                }
                Ok((
                    index,
                    Arc::new(VerifiedMappedAsset {
                        key,
                        tensor,
                        file,
                        mmap,
                    }),
                ))
            })
            .collect();

        let mut verified = Vec::with_capacity(opened.len());
        for result in opened {
            verified.push(result?);
        }
        for (index, entry) in verified {
            self.stats.mapped_logical_bytes = self
                .stats
                .mapped_logical_bytes
                .checked_add(entry.key.bytes)
                .ok_or_else(|| anyhow::anyhow!("mapped logical bytes overflow"))?;
            self.stats.sha256_bytes = self
                .stats
                .sha256_bytes
                .checked_add(entry.key.bytes)
                .ok_or_else(|| anyhow::anyhow!("mapped SHA bytes overflow"))?;
            self.entries.insert(entry.key.clone(), Arc::clone(&entry));
            output[index] = Some(entry);
        }
        output
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                entry.ok_or_else(|| anyhow::anyhow!("mapped asset 输出槽 {index} 未发布"))
            })
            .collect()
    }
}

#[cfg(windows)]
fn open_immutable_read(path: &Path) -> Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_SHARE_READ only：保留并发读，拒绝写入和删除直到 lease 被释放。
    OpenOptions::new()
        .read(true)
        .share_mode(0x0000_0001)
        .open(path)
        .with_context(|| format!("open immutable mapped payload {}", path.display()))
}

#[cfg(not(windows))]
fn open_immutable_read(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .open(path)
        .with_context(|| format!("open mapped payload {}", path.display()))
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("mapped payload SHA-256 必须是 64 位小写十六进制");
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct FixtureDir(PathBuf);

    impl FixtureDir {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "polaris-position0-mmap-{}-{stamp}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn asset(&self, name: &str, tensor: &str, bytes: &[u8]) -> Position0Asset {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            Position0Asset {
                tensor: tensor.into(),
                kind: "fixture".into(),
                expert_id: None,
                dtype: "I8".into(),
                shape: vec![bytes.len() as u64],
                bytes: bytes.len() as u64,
                range_key: format!("fixture:{name}"),
                cache_key: sha256_bytes(bytes),
                path,
                sha256: sha256_bytes(bytes),
                proof_path: self.0.join(format!("{name}.json")),
                proof_sha256: "a".repeat(64),
                hash_authority: "tofu".into(),
                payload_rehashed_by_builder: true,
                source: Value::Null,
            }
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn second_batch_reuses_same_mapping_without_rehash() {
        let fixture = FixtureDir::new();
        let assets = [
            fixture.asset("a.bin", "a", b"alpha"),
            fixture.asset("b.bin", "b", b"beta"),
        ];
        let mut store = VerifiedMappedAssetStore::new(&fixture.0).unwrap();
        let first = store.map_verified_batch(&assets).unwrap();
        let first_stats = store.stats();
        assert_eq!(first_stats.misses, 2);
        assert_eq!(first_stats.sha256_bytes, 9);
        let second = store.map_verified_batch(&assets).unwrap();
        assert!(Arc::ptr_eq(&first[0], &second[0]));
        assert!(Arc::ptr_eq(&first[1], &second[1]));
        let stats = store.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.sha256_bytes, 9);
        assert_eq!(first[0].bytes(), b"alpha");
    }

    #[test]
    fn batch_failure_publishes_no_new_mapping_and_root_escape_is_rejected() {
        let fixture = FixtureDir::new();
        let good = fixture.asset("good.bin", "good", b"good");
        let mut bad = fixture.asset("bad.bin", "bad", b"bad!");
        bad.sha256 = "0".repeat(64);
        let mut store = VerifiedMappedAssetStore::new(&fixture.0).unwrap();
        assert!(store.map_verified_batch(&[good, bad]).is_err());
        assert!(store.is_empty());

        let outside = FixtureDir::new();
        let escaped = outside.asset("outside.bin", "outside", b"outside");
        assert!(store.map_verified_batch(&[escaped]).is_err());
        assert!(store.is_empty());
    }

    #[test]
    fn duplicate_identity_and_tensor_alias_are_rejected() {
        let fixture = FixtureDir::new();
        let asset = fixture.asset("same.bin", "same", b"same");
        let mut store = VerifiedMappedAssetStore::new(&fixture.0).unwrap();
        assert!(store
            .map_verified_batch(&[asset.clone(), asset.clone()])
            .is_err());
        assert!(store.is_empty());
        store
            .map_verified_batch(std::slice::from_ref(&asset))
            .unwrap();
        let mut alias = asset;
        alias.tensor = "different".into();
        assert!(store.map_verified_batch(&[alias]).is_err());
    }
}
