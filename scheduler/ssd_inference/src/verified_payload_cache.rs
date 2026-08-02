//! 已经按 SHA-256 验证的 Range payload 的有界 host-RAM LRU。
//!
//! FullDepth Vulkan worker 过去在每层请求中重复读取并哈希约 105 MiB
//! top-6+shared 权重。该缓存以 canonical path、字节数和 SHA-256 为身份，
//! 首次读入时完整校验，之后只返回不可变 `Arc<[u8]>`。文件在首次验证后
//! 即使被外部改写，也不会污染当前 worker 已持有的可信副本。

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const VERIFIED_PAYLOAD_BATCH_MAX_TASKS: usize = 8;

#[derive(Debug, Clone)]
pub struct VerifiedPayloadRequest {
    pub path: PathBuf,
    pub expected_bytes: usize,
    pub expected_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PayloadKey {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}

#[derive(Debug)]
struct CacheEntry {
    payload: Arc<[u8]>,
    last_touch: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VerifiedPayloadCacheStats {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub oversized_bypasses: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub disk_bytes_read: u64,
    pub bytes_served: u64,
}

impl VerifiedPayloadCacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.requests as f64
        }
    }
}

/// 单进程有界缓存。它只缓存已经完整读入并通过 SHA-256 的 payload。
#[derive(Debug)]
pub struct VerifiedPayloadCache {
    capacity_bytes: usize,
    current_bytes: usize,
    clock: u64,
    entries: HashMap<PayloadKey, CacheEntry>,
    stats: VerifiedPayloadCacheStats,
}

impl VerifiedPayloadCache {
    pub fn new(capacity_bytes: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            bail!("verified payload cache 容量必须大于 0");
        }
        Ok(Self {
            capacity_bytes,
            current_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
            stats: VerifiedPayloadCacheStats::default(),
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> VerifiedPayloadCacheStats {
        self.stats
    }

    /// 加载并验证 payload；相同 path/bytes/SHA 的后续调用不再访问磁盘。
    pub fn load_verified(
        &mut self,
        path: &Path,
        expected_bytes: usize,
        expected_sha256: &str,
    ) -> Result<Arc<[u8]>> {
        if expected_bytes == 0 {
            bail!("verified payload 的期望字节数必须大于 0");
        }
        validate_sha256(expected_sha256)?;
        let canonical = path
            .canonicalize()
            .with_context(|| format!("resolve verified payload {}", path.display()))?;
        if !canonical.is_file() {
            bail!("verified payload 不是普通文件: {}", canonical.display());
        }
        let key = PayloadKey {
            path: canonical.clone(),
            bytes: expected_bytes,
            sha256: expected_sha256.to_owned(),
        };
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("verified payload cache clock overflow"))?;
        self.stats.requests += 1;
        self.stats.bytes_served = self
            .stats
            .bytes_served
            .checked_add(expected_bytes as u64)
            .ok_or_else(|| anyhow::anyhow!("verified payload bytes_served overflow"))?;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_touch = self.clock;
            self.stats.hits += 1;
            return Ok(Arc::clone(&entry.payload));
        }

        self.stats.misses += 1;
        let bytes = std::fs::read(&canonical)
            .with_context(|| format!("read verified payload {}", canonical.display()))?;
        self.stats.disk_bytes_read = self
            .stats
            .disk_bytes_read
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("verified payload disk byte counter overflow"))?;
        if bytes.len() != expected_bytes {
            bail!(
                "verified payload 字节漂移: {} expected={} actual={}",
                canonical.display(),
                expected_bytes,
                bytes.len()
            );
        }
        let observed = sha256_bytes(&bytes);
        if observed != expected_sha256 {
            bail!(
                "verified payload SHA-256 漂移: {} expected={} actual={}",
                canonical.display(),
                expected_sha256,
                observed
            );
        }
        let payload: Arc<[u8]> = bytes.into();
        if expected_bytes > self.capacity_bytes {
            self.stats.oversized_bypasses += 1;
            return Ok(payload);
        }

        while self.current_bytes + expected_bytes > self.capacity_bytes {
            self.evict_oldest()?;
        }
        self.current_bytes += expected_bytes;
        self.stats.current_bytes = self.current_bytes as u64;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.current_bytes);
        self.entries.insert(
            key,
            CacheEntry {
                payload: Arc::clone(&payload),
                last_touch: self.clock,
            },
        );
        Ok(payload)
    }

    /// 有界并行读取并验证一批互不重复的 payload，全部成功后才发布到 LRU。
    ///
    /// 返回顺序严格等于请求顺序。cache hit 仍直接复用不可变 `Arc`；miss 最多拆成
    /// `VERIFIED_PAYLOAD_BATCH_MAX_TASKS` 个并行任务。任一 miss 校验失败时，本批次
    /// 新读取的 payload 一个也不会进入缓存。
    pub fn load_verified_batch(
        &mut self,
        requests: &[VerifiedPayloadRequest],
    ) -> Result<Vec<Arc<[u8]>>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        #[derive(Debug)]
        struct BatchMiss {
            index: usize,
            key: PayloadKey,
            touch: u64,
        }

        let mut outputs: Vec<Option<Arc<[u8]>>> = vec![None; requests.len()];
        let mut misses = Vec::new();
        let mut batch_keys = HashSet::with_capacity(requests.len());

        for (index, request) in requests.iter().enumerate() {
            if request.expected_bytes == 0 {
                bail!("verified payload batch 的期望字节数必须大于 0");
            }
            validate_sha256(&request.expected_sha256)?;
            let canonical = request
                .path
                .canonicalize()
                .with_context(|| format!("resolve verified payload {}", request.path.display()))?;
            if !canonical.is_file() {
                bail!("verified payload 不是普通文件: {}", canonical.display());
            }
            let key = PayloadKey {
                path: canonical,
                bytes: request.expected_bytes,
                sha256: request.expected_sha256.clone(),
            };
            if !batch_keys.insert(key.clone()) {
                bail!("verified payload batch 含重复 path/bytes/SHA 身份");
            }

            self.clock = self
                .clock
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("verified payload cache clock overflow"))?;
            let touch = self.clock;
            self.stats.requests += 1;
            self.stats.bytes_served = self
                .stats
                .bytes_served
                .checked_add(request.expected_bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("verified payload bytes_served overflow"))?;
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.last_touch = touch;
                self.stats.hits += 1;
                outputs[index] = Some(Arc::clone(&entry.payload));
            } else {
                self.stats.misses += 1;
                misses.push(BatchMiss { index, key, touch });
            }
        }

        if !misses.is_empty() {
            let chunk_size = misses.len().div_ceil(VERIFIED_PAYLOAD_BATCH_MAX_TASKS);
            let grouped: Vec<Vec<Result<(usize, PayloadKey, u64, Arc<[u8]>)>>> = misses
                .par_chunks(chunk_size)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|miss| {
                            let bytes = std::fs::read(&miss.key.path).with_context(|| {
                                format!("read verified payload {}", miss.key.path.display())
                            })?;
                            if bytes.len() != miss.key.bytes {
                                bail!(
                                    "verified payload 字节漂移: {} expected={} actual={}",
                                    miss.key.path.display(),
                                    miss.key.bytes,
                                    bytes.len()
                                );
                            }
                            let observed = sha256_bytes(&bytes);
                            if observed != miss.key.sha256 {
                                bail!(
                                    "verified payload SHA-256 漂移: {} expected={} actual={}",
                                    miss.key.path.display(),
                                    miss.key.sha256,
                                    observed
                                );
                            }
                            Ok((
                                miss.index,
                                miss.key.clone(),
                                miss.touch,
                                Arc::<[u8]>::from(bytes),
                            ))
                        })
                        .collect()
                })
                .collect();
            let mut verified = Vec::with_capacity(misses.len());
            for result in grouped.into_iter().flatten() {
                verified.push(result?);
            }

            self.stats.disk_bytes_read = self
                .stats
                .disk_bytes_read
                .checked_add(
                    verified
                        .iter()
                        .map(|(_, _, _, payload)| payload.len() as u64)
                        .sum::<u64>(),
                )
                .ok_or_else(|| anyhow::anyhow!("verified payload disk byte counter overflow"))?;

            // 只有整批SHA全部成功后才按原请求顺序原子发布已验证Arc。
            for (index, key, touch, payload) in verified {
                if key.bytes > self.capacity_bytes {
                    self.stats.oversized_bypasses += 1;
                    outputs[index] = Some(payload);
                    continue;
                }
                while self.current_bytes + key.bytes > self.capacity_bytes {
                    self.evict_oldest()?;
                }
                self.current_bytes += key.bytes;
                self.stats.current_bytes = self.current_bytes as u64;
                self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.current_bytes);
                self.entries.insert(
                    key,
                    CacheEntry {
                        payload: Arc::clone(&payload),
                        last_touch: touch,
                    },
                );
                outputs[index] = Some(payload);
            }
        }

        outputs
            .into_iter()
            .enumerate()
            .map(|(index, payload)| {
                payload
                    .ok_or_else(|| anyhow::anyhow!("verified payload batch 输出槽 {index} 未发布"))
            })
            .collect()
    }

    fn evict_oldest(&mut self) -> Result<()> {
        let key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_touch)
            .map(|(key, _)| key.clone())
            .ok_or_else(|| anyhow::anyhow!("verified payload cache 无条目可淘汰"))?;
        let removed = self
            .entries
            .remove(&key)
            .ok_or_else(|| anyhow::anyhow!("verified payload cache LRU 索引漂移"))?;
        self.current_bytes = self
            .current_bytes
            .checked_sub(removed.payload.len())
            .ok_or_else(|| anyhow::anyhow!("verified payload cache 字节账本下溢"))?;
        self.stats.evictions += 1;
        self.stats.current_bytes = self.current_bytes as u64;
        Ok(())
    }
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("verified payload SHA-256 必须是 64 位小写十六进制");
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
    use std::time::{SystemTime, UNIX_EPOCH};

    struct FixtureDir(PathBuf);

    impl FixtureDir {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "polaris-verified-payload-cache-{}-{stamp}",
                std::process::id()
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn write(&self, name: &str, bytes: &[u8]) -> (PathBuf, String) {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            (path, sha256_bytes(bytes))
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn second_request_is_memory_hit_even_if_source_changes() {
        let fixture = FixtureDir::new();
        let (path, hash) = fixture.write("payload.bin", b"frozen");
        let mut cache = VerifiedPayloadCache::new(64).unwrap();
        let first = cache.load_verified(&path, 6, &hash).unwrap();
        std::fs::write(&path, b"damage").unwrap();
        let second = cache.load_verified(&path, 6, &hash).unwrap();
        assert_eq!(&*first, b"frozen");
        assert!(Arc::ptr_eq(&first, &second));
        let stats = cache.stats();
        assert_eq!(stats.requests, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.disk_bytes_read, 6);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn lru_evicts_oldest_verified_payload() {
        let fixture = FixtureDir::new();
        let (a, ah) = fixture.write("a.bin", b"aaaa");
        let (b, bh) = fixture.write("b.bin", b"bbbb");
        let (c, ch) = fixture.write("c.bin", b"cccc");
        let mut cache = VerifiedPayloadCache::new(8).unwrap();
        cache.load_verified(&a, 4, &ah).unwrap();
        cache.load_verified(&b, 4, &bh).unwrap();
        cache.load_verified(&a, 4, &ah).unwrap();
        cache.load_verified(&c, 4, &ch).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);
        cache.load_verified(&b, 4, &bh).unwrap();
        assert_eq!(cache.stats().misses, 4);
    }

    #[test]
    fn rejects_corrupt_or_oversized_identity_without_cache_pollution() {
        let fixture = FixtureDir::new();
        let (path, hash) = fixture.write("payload.bin", b"0123456789");
        let mut cache = VerifiedPayloadCache::new(4).unwrap();
        let payload = cache.load_verified(&path, 10, &hash).unwrap();
        assert_eq!(payload.len(), 10);
        assert!(cache.is_empty());
        assert_eq!(cache.stats().oversized_bypasses, 1);
        assert!(cache.load_verified(&path, 9, &hash).is_err());
        assert!(cache.load_verified(&path, 10, &"0".repeat(64)).is_err());
        assert!(cache.is_empty());
    }

    #[test]
    fn batch_preserves_request_order_and_turns_second_batch_into_hits() {
        let fixture = FixtureDir::new();
        let rows: Vec<_> = (0..12)
            .map(|index| {
                let payload = vec![index as u8; index + 1];
                let (path, hash) = fixture.write(&format!("{index:02}.bin"), &payload);
                (
                    VerifiedPayloadRequest {
                        path,
                        expected_bytes: payload.len(),
                        expected_sha256: hash,
                    },
                    payload,
                )
            })
            .collect();
        let requests: Vec<_> = rows.iter().map(|(request, _)| request.clone()).collect();
        let mut cache = VerifiedPayloadCache::new(1024).unwrap();

        let first = cache.load_verified_batch(&requests).unwrap();
        for (payload, (_, expected)) in first.iter().zip(&rows) {
            assert_eq!(&**payload, expected.as_slice());
        }
        let first_stats = cache.stats();
        assert_eq!(first_stats.requests, 12);
        assert_eq!(first_stats.misses, 12);
        assert_eq!(first_stats.hits, 0);
        assert_eq!(first_stats.disk_bytes_read, 78);

        let second = cache.load_verified_batch(&requests).unwrap();
        for (left, right) in first.iter().zip(&second) {
            assert!(Arc::ptr_eq(left, right));
        }
        let second_stats = cache.stats();
        assert_eq!(second_stats.requests, 24);
        assert_eq!(second_stats.misses, 12);
        assert_eq!(second_stats.hits, 12);
        assert_eq!(second_stats.disk_bytes_read, 78);
    }

    #[test]
    fn failed_batch_does_not_publish_any_new_payload() {
        let fixture = FixtureDir::new();
        let (good_path, good_hash) = fixture.write("good.bin", b"good");
        let (bad_path, _) = fixture.write("bad.bin", b"bad!");
        let requests = vec![
            VerifiedPayloadRequest {
                path: good_path,
                expected_bytes: 4,
                expected_sha256: good_hash,
            },
            VerifiedPayloadRequest {
                path: bad_path,
                expected_bytes: 4,
                expected_sha256: "0".repeat(64),
            },
        ];
        let mut cache = VerifiedPayloadCache::new(64).unwrap();
        assert!(cache.load_verified_batch(&requests).is_err());
        assert!(cache.is_empty());
        assert_eq!(cache.stats().current_bytes, 0);
    }

    #[test]
    fn batch_rejects_duplicate_identity_before_parallel_publish() {
        let fixture = FixtureDir::new();
        let (path, hash) = fixture.write("same.bin", b"same");
        let request = VerifiedPayloadRequest {
            path,
            expected_bytes: 4,
            expected_sha256: hash,
        };
        let mut cache = VerifiedPayloadCache::new(64).unwrap();
        assert!(cache
            .load_verified_batch(&[request.clone(), request])
            .is_err());
        assert!(cache.is_empty());
    }
}
