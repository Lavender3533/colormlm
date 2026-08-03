//! StarFold packed MXFP4 的系统内存 L2。
//!
//! verified mmap lease 解决了重复 proof/SHA，但生产热路径仍会把同一 weight/scale
//! 重新拼成 `[weight rows][scale rows]`。本缓存保存已经 proof-bound 的 packed payload，
//! 并按“层 × 投影”分片，避免一次 43 层顺序扫描把尾层热项全部冲掉。

use crate::{
    s14_starfold_cache::{StarfoldPageKey, StarfoldTensorSegment},
    s14_starfold_runtime::{S14StarfoldMicrotileSource, S14StarfoldVerifiedMicrotile},
};
use anyhow::{bail, Context, Result};
use polaris_s14_runner::FULL_DEPTH_LAYERS;
use std::{collections::BTreeMap, env, sync::Arc};

pub const S14_STARFOLD_PACKED_L2_ENV: &str = "POLARIS_S14_PACKED_L2_MIB";
pub const S14_STARFOLD_PACKED_L2_CONTRACT_VERSION: u32 = 1;
pub const S14_STARFOLD_PACKED_L2_DEFAULT_MIB: u64 = 4_096;
pub const S14_STARFOLD_PACKED_L2_MAX_MIB: u64 = 16_384;
pub const S14_STARFOLD_PACKED_L2_MIB_BYTES: u64 = 1024 * 1024;
pub const S14_STARFOLD_PACKED_L2_MAX_BYTES: u64 =
    S14_STARFOLD_PACKED_L2_MAX_MIB * S14_STARFOLD_PACKED_L2_MIB_BYTES;
const PROJECTION_SHARDS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldPackedL2Config {
    pub capacity_bytes: u64,
}

impl S14StarfoldPackedL2Config {
    pub fn from_env() -> Result<Self> {
        let mib = match env::var(S14_STARFOLD_PACKED_L2_ENV) {
            Ok(raw) => raw
                .parse::<u64>()
                .with_context(|| format!("解析 {S14_STARFOLD_PACKED_L2_ENV}={raw:?} 为 MiB"))?,
            Err(env::VarError::NotPresent) => S14_STARFOLD_PACKED_L2_DEFAULT_MIB,
            Err(env::VarError::NotUnicode(_)) => {
                bail!("{S14_STARFOLD_PACKED_L2_ENV} 不是 UTF-8")
            }
        };
        if mib > S14_STARFOLD_PACKED_L2_MAX_MIB {
            bail!(
                "{S14_STARFOLD_PACKED_L2_ENV} 必须位于 0..={S14_STARFOLD_PACKED_L2_MAX_MIB}: actual={mib}"
            );
        }
        Ok(Self {
            capacity_bytes: mib
                .checked_mul(S14_STARFOLD_PACKED_L2_MIB_BYTES)
                .context("StarFold packed L2 capacity bytes overflow")?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct S14StarfoldPackedL2Key {
    weight_page: StarfoldPageKey,
    weight_cache_key: String,
    weight_range_key: String,
    weight_offset: u64,
    weight_bytes: u32,
    scale_page: StarfoldPageKey,
    scale_cache_key: String,
    scale_range_key: String,
    scale_offset: u64,
    scale_bytes: u32,
    window_bytes: u32,
}

impl S14StarfoldPackedL2Key {
    pub fn from_sources(
        weight: &S14StarfoldMicrotileSource,
        scale: &S14StarfoldMicrotileSource,
        window_bytes: u32,
    ) -> Result<Self> {
        let expected_scale = match weight.span.key.segment {
            StarfoldTensorSegment::W1Weight => StarfoldTensorSegment::W1Scale,
            StarfoldTensorSegment::W2Weight => StarfoldTensorSegment::W2Scale,
            StarfoldTensorSegment::W3Weight => StarfoldTensorSegment::W3Scale,
            segment => bail!("StarFold packed L2 key 的 weight segment 非法: {segment:?}"),
        };
        if scale.span.key.segment != expected_scale
            || scale.span.key.layer != weight.span.key.layer
            || scale.span.key.expert_id != weight.span.key.expert_id
            || scale.span.key.tile_index != weight.span.key.tile_index
        {
            bail!("StarFold packed L2 weight/scale identity 不属于同一 tile");
        }
        if window_bytes == 0
            || weight.span.byte_len == 0
            || scale.span.byte_len == 0
            || u64::from(weight.span.byte_len) + u64::from(scale.span.byte_len)
                > u64::from(window_bytes)
        {
            bail!("StarFold packed L2 key bytes 越出物理窗口");
        }
        Ok(Self {
            weight_page: weight.span.key,
            weight_cache_key: weight.planned.cache_key.clone(),
            weight_range_key: weight.planned.range_key.clone(),
            weight_offset: weight.span.source_segment_offset,
            weight_bytes: weight.span.byte_len,
            scale_page: scale.span.key,
            scale_cache_key: scale.planned.cache_key.clone(),
            scale_range_key: scale.planned.range_key.clone(),
            scale_offset: scale.span.source_segment_offset,
            scale_bytes: scale.span.byte_len,
            window_bytes,
        })
    }

    fn shard(&self) -> Result<usize> {
        let projection = match self.weight_page.segment {
            StarfoldTensorSegment::W1Weight => 0,
            StarfoldTensorSegment::W3Weight => 1,
            StarfoldTensorSegment::W2Weight => 2,
            segment => bail!("StarFold packed L2 shard segment 非法: {segment:?}"),
        };
        let layer = usize::from(self.weight_page.layer);
        if layer >= FULL_DEPTH_LAYERS.len() {
            bail!("StarFold packed L2 layer 越出 FullDepth43: {layer}");
        }
        Ok(layer * PROJECTION_SHARDS + projection)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarfoldPackedL2Stats {
    pub capacity_bytes: u64,
    pub resident_bytes: u64,
    pub entries: usize,
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub rejections: u64,
    pub evictions: u64,
    pub avoided_pack_bytes: u64,
}

#[derive(Debug)]
struct PackedEntry {
    proof: Arc<S14StarfoldVerifiedMicrotile>,
    bytes: u64,
    frequency: u64,
    last_use: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Observation {
    frequency: u64,
    last_use: u64,
}

/// 单 runtime 持有的懒分配 packed L2。production session 已串行独占 runtime，
/// 因此这里不引入 mutex；proof 内的 mmap lease 仍由进程级 cache 负责并发生命周期。
#[derive(Debug)]
pub struct S14StarfoldPackedL2Cache {
    capacity_bytes: u64,
    shard_capacity_bytes: u64,
    clock: u64,
    resident_bytes: u64,
    entries: BTreeMap<S14StarfoldPackedL2Key, PackedEntry>,
    observations: BTreeMap<S14StarfoldPackedL2Key, Observation>,
    stats: S14StarfoldPackedL2Stats,
}

impl S14StarfoldPackedL2Cache {
    pub fn new(config: S14StarfoldPackedL2Config) -> Result<Self> {
        let shard_count = u64::try_from(FULL_DEPTH_LAYERS.len() * PROJECTION_SHARDS)
            .context("StarFold packed L2 shard count overflow")?;
        let shard_capacity_bytes = if config.capacity_bytes == 0 {
            0
        } else {
            config.capacity_bytes / shard_count
        };
        Ok(Self {
            capacity_bytes: config.capacity_bytes,
            shard_capacity_bytes,
            clock: 0,
            resident_bytes: 0,
            entries: BTreeMap::new(),
            observations: BTreeMap::new(),
            stats: S14StarfoldPackedL2Stats {
                capacity_bytes: config.capacity_bytes,
                ..S14StarfoldPackedL2Stats::default()
            },
        })
    }

    pub fn lookup(
        &mut self,
        key: &S14StarfoldPackedL2Key,
    ) -> Option<Arc<S14StarfoldVerifiedMicrotile>> {
        self.clock = self.clock.saturating_add(1);
        self.stats.lookups = self.stats.lookups.saturating_add(1);
        let observation = self.observations.entry(key.clone()).or_default();
        observation.frequency = observation.frequency.saturating_add(1);
        observation.last_use = self.clock;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.frequency = entry.frequency.saturating_add(1);
            entry.last_use = self.clock;
            self.stats.hits = self.stats.hits.saturating_add(1);
            self.stats.avoided_pack_bytes =
                self.stats.avoided_pack_bytes.saturating_add(entry.bytes);
            return Some(Arc::clone(&entry.proof));
        }
        self.stats.misses = self.stats.misses.saturating_add(1);
        None
    }

    pub fn admit(
        &mut self,
        key: S14StarfoldPackedL2Key,
        proof: Arc<S14StarfoldVerifiedMicrotile>,
    ) -> Result<Arc<S14StarfoldVerifiedMicrotile>> {
        if let Some(existing) = self.entries.get(&key) {
            return Ok(Arc::clone(&existing.proof));
        }
        let bytes = proof.byte_len();
        let shard = key.shard()?;
        if self.capacity_bytes == 0
            || bytes == 0
            || bytes > self.shard_capacity_bytes
            || bytes > self.capacity_bytes
        {
            self.stats.rejections = self.stats.rejections.saturating_add(1);
            return Ok(proof);
        }
        let candidate_frequency = self
            .observations
            .get(&key)
            .map_or(1, |observation| observation.frequency.max(1));
        let mut shard_bytes = self
            .entries
            .iter()
            .filter(|(entry_key, _)| entry_key.shard().ok() == Some(shard))
            .map(|(_, entry)| entry.bytes)
            .sum::<u64>();

        while shard_bytes.saturating_add(bytes) > self.shard_capacity_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(entry_key, _)| entry_key.shard().ok() == Some(shard))
                .min_by_key(|(_, entry)| (entry.frequency, entry.last_use))
                .map(|(entry_key, entry)| (entry_key.clone(), entry.frequency));
            let Some((victim_key, victim_frequency)) = victim else {
                self.stats.rejections = self.stats.rejections.saturating_add(1);
                return Ok(proof);
            };
            // 单次顺序扫描不会驱逐已有热项。新项至少再次出现，且频率严格高于
            // 当前最低频项，才允许进入该层/投影的有限 RAM 配额。
            if candidate_frequency <= victim_frequency {
                self.stats.rejections = self.stats.rejections.saturating_add(1);
                return Ok(proof);
            }
            let removed = self
                .entries
                .remove(&victim_key)
                .context("StarFold packed L2 victim 消失")?;
            shard_bytes = shard_bytes.saturating_sub(removed.bytes);
            self.resident_bytes = self.resident_bytes.saturating_sub(removed.bytes);
            self.stats.evictions = self.stats.evictions.saturating_add(1);
        }

        if self.resident_bytes.saturating_add(bytes) > self.capacity_bytes {
            // 分片预算总和理论上不会越出总预算；这里保持 fail-closed，不做跨层驱逐。
            self.stats.rejections = self.stats.rejections.saturating_add(1);
            return Ok(proof);
        }
        self.clock = self.clock.saturating_add(1);
        self.entries.insert(
            key,
            PackedEntry {
                proof: Arc::clone(&proof),
                bytes,
                frequency: candidate_frequency,
                last_use: self.clock,
            },
        );
        self.resident_bytes = self.resident_bytes.saturating_add(bytes);
        self.stats.admissions = self.stats.admissions.saturating_add(1);
        self.refresh_snapshot();
        Ok(proof)
    }

    pub fn stats(&self) -> S14StarfoldPackedL2Stats {
        let mut stats = self.stats;
        stats.resident_bytes = self.resident_bytes;
        stats.entries = self.entries.len();
        stats
    }

    fn refresh_snapshot(&mut self) {
        self.stats.resident_bytes = self.resident_bytes;
        self.stats.entries = self.entries.len();
    }
}
