//! S14 任意 input token / position 的原生资产计划。
//!
//! `Position0WholeTokenManifest` 冻结的是 BOS position0 的历史数值证据，不能充当
//! production 输入目录。本模块直接从固定 revision 的 FullDepth43 catalog 派生：
//!
//! - `embed.weight[token:token+1]` 的精确 8 KiB Range 与兼容现有 Range cache 的 key；
//! - 每个 compressor/indexer APE 的静态资产身份和 `position % ratio` 行绑定；
//! - RoPE position、window slot、overlap remainder row 与压缩块边界。
//!
//! 计划本身不访问网络或读取 payload；cache miss 与 GPU 接线交给后续原生 Rust runtime。

use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{
    Position0Asset, COMPRESS_RATIOS, FULL_DEPTH_LAYERS, HIDDEN_SIZE, MODEL_REPO, MODEL_REVISION,
    VOCAB_SIZE,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

pub const S14_INPUT_ASSET_PLAN_FORMAT: &str = "polaris-s14-input-asset-plan-v1";
pub const S14_FULL_DEPTH_CATALOG_FORMAT: &str = "polaris-fulldepth43-native-top6-catalog-v1";
pub const S14_FULL_DEPTH_CATALOG_PROFILE: &str = "fulldepth43_native_top6";
pub const S14_FULL_DEPTH_CATALOG_SHA256: &str =
    "ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049";
pub const S14_ATTENTION_WINDOW: u32 = 128;
pub const S14_RANGE_CACHE_META_FORMAT: &str = "polaris-s14-range-cache-entry-v1";
pub const S14_RANGE_CACHE_VERIFIED_TRANSPORT: &str = "HTTPS/206/exact-Content-Range";

const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const EMBEDDING_ROW_BYTES: u64 = HIDDEN_SIZE as u64 * BF16_BYTES;
const EXPECTED_POSITION_ASSET_COUNT: usize = 62;

#[derive(Debug, Clone)]
pub struct S14InputAssetPlanner {
    catalog_sha256: String,
    cache_root: PathBuf,
    embedding: CatalogAsset,
    position_assets: Vec<CatalogPositionAsset>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14InputAssetPlan {
    pub format: &'static str,
    pub catalog_sha256: String,
    pub position: u32,
    pub input_token_id: u32,
    pub embedding: S14PlannedRangeAsset,
    pub position_execution: S14PositionExecutionPlan,
}

/// 已把计划中的本地 Range proof 解析为现有 runtime 可消费资产的 token 输入。
///
/// 这里仍不映射 payload；`VerifiedMappedAssetStore` 必须在 GPU 发布前完成完整
/// payload SHA-256。把 embedding 与 position 资产放进同一对象，避免 position1
/// 误拿 position0 的 APE/remainder 参数。
#[derive(Debug, Clone)]
pub struct S14PreparedTokenInput {
    pub plan: S14InputAssetPlan,
    pub embedding: Position0Asset,
    pub position_assets: Vec<Position0Asset>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14PositionExecutionPlan {
    pub rope_position: u32,
    pub window_slot: u32,
    pub active_window_tokens: u32,
    pub ape_rows: Vec<S14PositionAssetBinding>,
}

impl S14PositionExecutionPlan {
    /// 独立复核 position/RoPE/window/APE/remainder/compressed-boundary 合同。
    /// GPU recorder 可调用本入口，无需信任上游序列化对象。
    pub fn validate_for_position(&self, position: u32) -> Result<()> {
        let next_position = position
            .checked_add(1)
            .ok_or_else(|| anyhow!("S14 position overflow"))?;
        if self.rope_position != position
            || self.window_slot != position % S14_ATTENTION_WINDOW
            || self.active_window_tokens != next_position.min(S14_ATTENTION_WINDOW)
            || self.ape_rows.len() != EXPECTED_POSITION_ASSET_COUNT
        {
            bail!("S14 position execution RoPE/window/APE count 漂移");
        }

        for (&layer, &ratio) in FULL_DEPTH_LAYERS.iter().zip(COMPRESS_RATIOS.iter()) {
            let expected_kinds: &[S14PositionAssetKind] = match ratio {
                0 => &[],
                4 => &[
                    S14PositionAssetKind::CompressorApe,
                    S14PositionAssetKind::IndexerCompressorApe,
                ],
                128 => &[S14PositionAssetKind::CompressorApe],
                _ => bail!("S14 position execution 未注册 ratio {ratio}"),
            };
            for &kind in expected_kinds {
                let matches = self
                    .ape_rows
                    .iter()
                    .filter(|binding| binding.layer == layer && binding.kind == kind)
                    .collect::<Vec<_>>();
                if matches.len() != 1 {
                    bail!("S14 L{layer} position APE kind 数量漂移");
                }
                validate_position_binding(matches[0], position, ratio)?;
            }
            if self
                .ape_rows
                .iter()
                .any(|binding| binding.layer == layer && !expected_kinds.contains(&binding.kind))
            {
                bail!("S14 L{layer} position APE kind/ratio 漂移");
            }
        }
        Ok(())
    }
}

fn validate_position_binding(
    binding: &S14PositionAssetBinding,
    position: u32,
    ratio: u16,
) -> Result<()> {
    let ratio_u32 = u32::from(ratio);
    let next_position = position
        .checked_add(1)
        .ok_or_else(|| anyhow!("S14 position overflow"))?;
    let ape_row = (position % ratio_u32) as u16;
    let row_elements = binding
        .asset
        .shape
        .get(1)
        .copied()
        .ok_or_else(|| anyhow!("S14 APE 缺少 row width"))?;
    let row_bytes = row_elements
        .checked_mul(F32_BYTES)
        .ok_or_else(|| anyhow!("S14 APE row bytes overflow"))?;
    let row_offset = u64::from(ape_row)
        .checked_mul(row_bytes)
        .ok_or_else(|| anyhow!("S14 APE row offset overflow"))?;
    let block_ready = next_position % ratio_u32 == 0;
    let expected_remainder = if ratio == 4 { ratio + ape_row } else { ape_row };
    let expected_rope = block_ready.then(|| next_position - ratio_u32);
    if binding.ratio != ratio
        || binding.asset.shape.first().copied() != Some(u64::from(ratio))
        || binding.ape_row != ape_row
        || binding.ape_row_byte_offset != row_offset
        || binding.ape_row_bytes != row_bytes
        || binding.remainder_row != expected_remainder
        || binding.compressed_block_ready != block_ready
        || binding.completed_blocks_after != next_position / ratio_u32
        || binding.compressed_rope_position != expected_rope
    {
        bail!(
            "S14 L{} position APE/remainder/boundary 漂移",
            binding.layer
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14PositionAssetKind {
    CompressorApe,
    IndexerCompressorApe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14PositionAssetBinding {
    pub layer: u8,
    pub kind: S14PositionAssetKind,
    pub ratio: u16,
    pub ape_row: u16,
    pub ape_row_byte_offset: u64,
    pub ape_row_bytes: u64,
    pub remainder_row: u16,
    pub compressed_block_ready: bool,
    pub completed_blocks_after: u32,
    pub compressed_rope_position: Option<u32>,
    pub asset: S14PlannedRangeAsset,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14PlannedRangeAsset {
    pub tensor: String,
    pub parent_tensor: Option<String>,
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub range_key: String,
    pub cache_key: String,
    pub payload_path: PathBuf,
    pub proof_path: PathBuf,
    pub identity: S14RangeIdentity,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct S14RangeIdentity {
    pub repo: String,
    pub revision: String,
    pub source_file: String,
    pub source_file_bytes: u64,
    pub start: u64,
    pub end: u64,
    pub header_tensor_table_sha256: String,
}

#[derive(Debug, Deserialize)]
struct S14RangeCacheProof {
    format: String,
    cache_key: String,
    identity: S14RangeIdentity,
    bytes: u64,
    observed_sha256: String,
    expected_sha256: Option<String>,
    hash_authority: String,
    authoritative: bool,
    verified_transport: String,
}

impl S14PlannedRangeAsset {
    #[allow(clippy::too_many_arguments)]
    pub fn from_identity(
        cache_root: &Path,
        tensor: String,
        parent_tensor: Option<String>,
        kind: String,
        dtype: String,
        shape: Vec<u64>,
        bytes: u64,
        range_key: String,
        identity: S14RangeIdentity,
    ) -> Result<Self> {
        if cache_root.as_os_str().is_empty()
            || tensor.is_empty()
            || kind.is_empty()
            || dtype.is_empty()
            || shape.is_empty()
        {
            bail!("S14 planned Range 逻辑 identity 为空");
        }
        validate_range_identity(&identity, bytes)?;
        let expected_range_key = format!(
            "{}:{}-{}",
            identity.source_file, identity.start, identity.end
        );
        if range_key != expected_range_key {
            bail!("S14 planned Range range_key 漂移");
        }
        let cache_key = range_cache_key(&identity)?;
        Ok(Self {
            tensor,
            parent_tensor,
            kind,
            dtype,
            shape,
            bytes,
            range_key,
            payload_path: cache_root.join(format!("{cache_key}.bin")),
            proof_path: cache_root.join(format!("{cache_key}.json")),
            cache_key,
            identity,
        })
    }

    /// 只解析并验证本地 Python Range cache proof；payload 内容 SHA 由
    /// `VerifiedMappedAssetStore` 在映射发布前完成。
    pub fn resolve_cached_position0_asset(
        &self,
        cache_root: &Path,
        expert_id: Option<u16>,
    ) -> Result<Position0Asset> {
        validate_range_identity(&self.identity, self.bytes)?;
        let rebuilt = Self::from_identity(
            cache_root,
            self.tensor.clone(),
            self.parent_tensor.clone(),
            self.kind.clone(),
            self.dtype.clone(),
            self.shape.clone(),
            self.bytes,
            self.range_key.clone(),
            self.identity.clone(),
        )?;
        if rebuilt.cache_key != self.cache_key {
            bail!("S14 Range cache key 不是 canonical identity 派生");
        }

        let canonical_root = cache_root
            .canonicalize()
            .with_context(|| format!("resolve S14 Range cache root {}", cache_root.display()))?;
        if !canonical_root.is_dir() {
            bail!("S14 Range cache root 不是目录");
        }
        let expected_payload = canonical_root.join(format!("{}.bin", self.cache_key));
        let expected_proof = canonical_root.join(format!("{}.json", self.cache_key));
        let partial = canonical_root.join(format!("{}.bin.part", self.cache_key));
        if partial.exists() {
            bail!("S14 Range cache 存在未提交 partial: {}", partial.display());
        }
        let payload_path = self
            .payload_path
            .canonicalize()
            .with_context(|| format!("resolve cached payload {}", self.payload_path.display()))?;
        let proof_path = self
            .proof_path
            .canonicalize()
            .with_context(|| format!("resolve cached proof {}", self.proof_path.display()))?;
        if payload_path != expected_payload || proof_path != expected_proof {
            bail!("S14 Range cache payload/proof 路径不是 canonical key 派生");
        }
        let payload_meta = fs::metadata(&payload_path)
            .with_context(|| format!("stat cached payload {}", payload_path.display()))?;
        if !payload_meta.is_file() || payload_meta.len() != self.bytes {
            bail!(
                "S14 cached payload 字节漂移: expected={} actual={}",
                self.bytes,
                payload_meta.len()
            );
        }

        let proof_bytes = fs::read(&proof_path)
            .with_context(|| format!("读取 cached proof {}", proof_path.display()))?;
        let proof: S14RangeCacheProof =
            serde_json::from_slice(&proof_bytes).context("解析 S14 Range cache proof")?;
        validate_cache_proof(&proof, self)?;
        let proof_sha256 = sha256_bytes(&proof_bytes);
        let source =
            serde_json::to_value(&self.identity).context("编码 S14 Range source identity")?;
        let resolved = Position0Asset {
            tensor: self.tensor.clone(),
            kind: self.kind.clone(),
            expert_id,
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            bytes: self.bytes,
            range_key: self.range_key.clone(),
            cache_key: self.cache_key.clone(),
            path: payload_path,
            sha256: proof.observed_sha256,
            proof_path,
            proof_sha256,
            hash_authority: proof.hash_authority,
            payload_rehashed_by_builder: false,
            source,
        };
        self.validate_resolved_position0_asset(&resolved, expert_id)?;
        Ok(resolved)
    }

    /// 逐字段绑定 planned Range 与 runtime 资产。SHA 的内容真实性仍由映射层复核；
    /// 此处负责阻止 token/Range/proof 路径或 source identity 被同形资产替换。
    pub fn validate_resolved_position0_asset(
        &self,
        asset: &Position0Asset,
        expert_id: Option<u16>,
    ) -> Result<()> {
        let source = serde_json::to_value(&self.identity)
            .context("编码 S14 planned Range source identity")?;
        let expected_payload = self
            .payload_path
            .canonicalize()
            .with_context(|| format!("resolve planned payload {}", self.payload_path.display()))?;
        let expected_proof = self
            .proof_path
            .canonicalize()
            .with_context(|| format!("resolve planned proof {}", self.proof_path.display()))?;
        if asset.tensor != self.tensor
            || asset.kind != self.kind
            || asset.expert_id != expert_id
            || asset.dtype != self.dtype
            || asset.shape != self.shape
            || asset.bytes != self.bytes
            || asset.range_key != self.range_key
            || asset.cache_key != self.cache_key
            || asset.path != expected_payload
            || asset.proof_path != expected_proof
            || asset.source != source
            || asset.sha256.len() != 64
            || asset.proof_sha256.len() != 64
        {
            bail!(
                "S14 resolved Range 与 planned identity 漂移: {}",
                self.tensor
            );
        }
        validate_sha256(&asset.sha256)?;
        validate_sha256(&asset.proof_sha256)?;
        Ok(())
    }
}

fn validate_cache_proof(proof: &S14RangeCacheProof, planned: &S14PlannedRangeAsset) -> Result<()> {
    validate_sha256(&proof.observed_sha256)?;
    if proof.format != S14_RANGE_CACHE_META_FORMAT
        || proof.cache_key != planned.cache_key
        || proof.identity != planned.identity
        || proof.bytes != planned.bytes
        || proof.verified_transport != S14_RANGE_CACHE_VERIFIED_TRANSPORT
    {
        bail!("S14 Range cache proof format/key/identity/bytes/transport 漂移");
    }
    if proof.authoritative {
        if proof.hash_authority != "official_lock"
            || proof.expected_sha256.as_deref() != Some(proof.observed_sha256.as_str())
        {
            bail!("S14 authoritative Range cache proof 自相矛盾");
        }
    } else if proof.hash_authority != "tofu" || proof.expected_sha256.is_some() {
        bail!("S14 TOFU Range cache proof 试图冒充权威");
    }
    Ok(())
}

impl S14InputAssetPlanner {
    /// 加载并完整哈希固定 FullDepth43 catalog。该入口不接受调用者自选 catalog SHA。
    pub fn load_pinned(catalog_path: &Path, cache_root: &Path) -> Result<Self> {
        let bytes = fs::read(catalog_path)
            .with_context(|| format!("读取 S14 catalog {}", catalog_path.display()))?;
        let observed = sha256_bytes(&bytes);
        if observed != S14_FULL_DEPTH_CATALOG_SHA256 {
            bail!(
                "S14 catalog SHA-256 漂移: expected={} actual={observed}",
                S14_FULL_DEPTH_CATALOG_SHA256
            );
        }
        Self::from_verified_catalog_bytes(&bytes, observed, cache_root)
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn cache_root(&self) -> &Path {
        &self.cache_root
    }

    /// 解析一个已验证计划需要的本地 embedding 与全部 position 资产 proof。
    /// payload 仍须在使用点由 `VerifiedMappedAssetStore` 完整哈希。
    pub fn prepare_cached_input(&self, plan: &S14InputAssetPlan) -> Result<S14PreparedTokenInput> {
        self.validate_plan(plan)?;
        let embedding = plan
            .embedding
            .resolve_cached_position0_asset(&self.cache_root, None)?;
        let position_assets = plan
            .position_execution
            .ape_rows
            .iter()
            .map(|binding| {
                binding
                    .asset
                    .resolve_cached_position0_asset(&self.cache_root, None)
            })
            .collect::<Result<Vec<_>>>()?;
        if position_assets.len() != EXPECTED_POSITION_ASSET_COUNT {
            bail!("S14 prepared position asset 数量漂移");
        }
        Ok(S14PreparedTokenInput {
            plan: plan.clone(),
            embedding,
            position_assets,
        })
    }

    /// prologue 只需要 embedding 时的窄入口。它仍重新派生整个 token/position
    /// 计划，禁止调用者把 token5 的行挂到 token0/position0。
    pub fn resolve_cached_embedding(&self, plan: &S14InputAssetPlan) -> Result<Position0Asset> {
        self.validate_plan(plan)?;
        plan.embedding
            .resolve_cached_position0_asset(&self.cache_root, None)
    }

    /// 为一个尚未提交的 token candidate 生成确定性输入计划。
    pub fn plan(
        &self,
        position: u32,
        input_token_id: u32,
        max_seq_len: u32,
    ) -> Result<S14InputAssetPlan> {
        if input_token_id >= VOCAB_SIZE {
            bail!("S14 input token {input_token_id} 越出 vocab {VOCAB_SIZE}");
        }
        if max_seq_len == 0 || position >= max_seq_len {
            bail!("S14 position 越出 sequence: position={position} max_seq_len={max_seq_len}");
        }
        let embedding = self.derive_embedding_row(input_token_id)?;
        let next_position = position
            .checked_add(1)
            .ok_or_else(|| anyhow!("S14 position overflow"))?;
        let mut ape_rows = Vec::with_capacity(self.position_assets.len());
        for source in &self.position_assets {
            let ratio_u32 = u32::from(source.ratio);
            let ape_row = (position % ratio_u32) as u16;
            let row_bytes = source
                .asset
                .shape
                .get(1)
                .copied()
                .ok_or_else(|| anyhow!("S14 APE 缺少 row width"))?
                .checked_mul(F32_BYTES)
                .ok_or_else(|| anyhow!("S14 APE row bytes overflow"))?;
            let row_byte_offset = u64::from(ape_row)
                .checked_mul(row_bytes)
                .ok_or_else(|| anyhow!("S14 APE row offset overflow"))?;
            let row_end = row_byte_offset
                .checked_add(row_bytes)
                .ok_or_else(|| anyhow!("S14 APE row end overflow"))?;
            if row_end > source.asset.bytes {
                bail!("S14 APE row 越界: {}", source.asset.tensor);
            }
            let overlap = source.ratio == 4;
            let remainder_row = if overlap {
                source
                    .ratio
                    .checked_add(ape_row)
                    .ok_or_else(|| anyhow!("S14 overlap remainder row overflow"))?
            } else {
                ape_row
            };
            let block_ready = next_position % ratio_u32 == 0;
            ape_rows.push(S14PositionAssetBinding {
                layer: source.layer,
                kind: source.kind,
                ratio: source.ratio,
                ape_row,
                ape_row_byte_offset: row_byte_offset,
                ape_row_bytes: row_bytes,
                remainder_row,
                compressed_block_ready: block_ready,
                completed_blocks_after: next_position / ratio_u32,
                compressed_rope_position: block_ready.then(|| next_position - ratio_u32),
                asset: self.plan_catalog_asset(&source.asset)?,
            });
        }
        if ape_rows.len() != EXPECTED_POSITION_ASSET_COUNT {
            bail!("S14 position APE 计划数量漂移");
        }
        let plan = S14InputAssetPlan {
            format: S14_INPUT_ASSET_PLAN_FORMAT,
            catalog_sha256: self.catalog_sha256.clone(),
            position,
            input_token_id,
            embedding,
            position_execution: S14PositionExecutionPlan {
                rope_position: position,
                window_slot: position % S14_ATTENTION_WINDOW,
                active_window_tokens: next_position.min(S14_ATTENTION_WINDOW),
                ape_rows,
            },
        };
        plan.position_execution.validate_for_position(position)?;
        Ok(plan)
    }

    fn from_verified_catalog_bytes(
        bytes: &[u8],
        catalog_sha256: String,
        cache_root: &Path,
    ) -> Result<Self> {
        validate_sha256(&catalog_sha256)?;
        if cache_root.as_os_str().is_empty() {
            bail!("S14 Range cache root 不能为空");
        }
        let catalog: CatalogDocument =
            serde_json::from_slice(bytes).context("解析 S14 FullDepth43 catalog")?;
        if catalog.format != S14_FULL_DEPTH_CATALOG_FORMAT
            || catalog.repo != MODEL_REPO
            || catalog.revision != MODEL_REVISION
            || catalog.profile.id != S14_FULL_DEPTH_CATALOG_PROFILE
            || catalog.profile.repo != MODEL_REPO
            || catalog.profile.revision != MODEL_REVISION
            || catalog.profile.layers != FULL_DEPTH_LAYERS
            || catalog.profile.compress_ratios != COMPRESS_RATIOS
        {
            bail!("S14 catalog identity/profile 漂移");
        }
        if catalog.boundary.embedding.len() != 1 {
            bail!("S14 catalog embedding 边界必须唯一");
        }
        let embedding = catalog.boundary.embedding.into_iter().next().unwrap();
        validate_embedding_table(&embedding)?;

        let mut position_assets = Vec::with_capacity(EXPECTED_POSITION_ASSET_COUNT);
        for (&layer, &ratio) in FULL_DEPTH_LAYERS.iter().zip(COMPRESS_RATIOS.iter()) {
            let row = catalog
                .layers
                .get(&layer.to_string())
                .ok_or_else(|| anyhow!("S14 catalog 缺少 L{layer}"))?;
            let prefix = format!("layers.{layer}.");
            match ratio {
                0 => {
                    if row
                        .non_expert
                        .iter()
                        .any(|asset| asset.tensor.ends_with(".compressor.ape"))
                    {
                        bail!("S14 ratio0 L{layer} 不得包含 compressor APE");
                    }
                }
                4 | 128 => {
                    let main_name = format!("{prefix}attn.compressor.ape");
                    let main = find_unique_asset(&row.non_expert, &main_name)?;
                    validate_ape(main, layer, ratio, false)?;
                    position_assets.push(CatalogPositionAsset {
                        layer,
                        kind: S14PositionAssetKind::CompressorApe,
                        ratio,
                        asset: main.clone(),
                    });
                    let indexer_name = format!("{prefix}attn.indexer.compressor.ape");
                    if ratio == 4 {
                        let indexer = find_unique_asset(&row.non_expert, &indexer_name)?;
                        validate_ape(indexer, layer, ratio, true)?;
                        position_assets.push(CatalogPositionAsset {
                            layer,
                            kind: S14PositionAssetKind::IndexerCompressorApe,
                            ratio,
                            asset: indexer.clone(),
                        });
                    } else if row
                        .non_expert
                        .iter()
                        .any(|asset| asset.tensor == indexer_name)
                    {
                        bail!("S14 ratio128 L{layer} 不得包含 indexer APE");
                    }
                }
                _ => bail!("S14 catalog 出现未注册 compress ratio {ratio}"),
            }
        }
        if position_assets.len() != EXPECTED_POSITION_ASSET_COUNT {
            bail!(
                "S14 position asset count 漂移: expected={} actual={}",
                EXPECTED_POSITION_ASSET_COUNT,
                position_assets.len()
            );
        }
        let unique = position_assets
            .iter()
            .map(|asset| asset.asset.tensor.as_str())
            .collect::<HashSet<_>>();
        if unique.len() != position_assets.len() {
            bail!("S14 position asset tensor 重复");
        }
        Ok(Self {
            catalog_sha256,
            cache_root: cache_root.to_path_buf(),
            embedding,
            position_assets,
        })
    }

    fn derive_embedding_row(&self, token_id: u32) -> Result<S14PlannedRangeAsset> {
        let row_start = self
            .embedding
            .start
            .checked_add(
                u64::from(token_id)
                    .checked_mul(EMBEDDING_ROW_BYTES)
                    .ok_or_else(|| anyhow!("embedding token byte offset overflow"))?,
            )
            .ok_or_else(|| anyhow!("embedding row start overflow"))?;
        let row_end = row_start
            .checked_add(EMBEDDING_ROW_BYTES - 1)
            .ok_or_else(|| anyhow!("embedding row end overflow"))?;
        if row_end > self.embedding.end {
            bail!("embedding token {token_id} 越出固定表");
        }
        self.plan_range(
            format!("embed.weight[{token_id}:{}]", token_id + 1),
            Some(self.embedding.tensor.clone()),
            "embedding_row".to_owned(),
            "BF16".to_owned(),
            vec![1, HIDDEN_SIZE as u64],
            row_start,
            row_end,
            &self.embedding,
        )
    }

    fn plan_catalog_asset(&self, asset: &CatalogAsset) -> Result<S14PlannedRangeAsset> {
        self.plan_range(
            asset.tensor.clone(),
            None,
            asset.kind.clone(),
            asset.dtype.clone(),
            asset.shape.clone(),
            asset.start,
            asset.end,
            asset,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_range(
        &self,
        tensor: String,
        parent_tensor: Option<String>,
        kind: String,
        dtype: String,
        shape: Vec<u64>,
        start: u64,
        end: u64,
        source: &CatalogAsset,
    ) -> Result<S14PlannedRangeAsset> {
        let bytes = end
            .checked_sub(start)
            .and_then(|delta| delta.checked_add(1))
            .ok_or_else(|| anyhow!("S14 planned Range bytes overflow"))?;
        let identity = S14RangeIdentity {
            repo: MODEL_REPO.to_owned(),
            revision: MODEL_REVISION.to_owned(),
            source_file: source.file.clone(),
            source_file_bytes: source.file_bytes,
            start,
            end,
            header_tensor_table_sha256: source.header_tensor_table_sha256.clone(),
        };
        S14PlannedRangeAsset::from_identity(
            &self.cache_root,
            tensor,
            parent_tensor,
            kind,
            dtype,
            shape,
            bytes,
            format!("{}:{start}-{end}", source.file),
            identity,
        )
    }

    /// 重新派生并逐字段比较，禁止调用者篡改 token、position、Range 或行索引。
    pub fn validate_plan(&self, plan: &S14InputAssetPlan) -> Result<()> {
        if plan.format != S14_INPUT_ASSET_PLAN_FORMAT || plan.catalog_sha256 != self.catalog_sha256
        {
            bail!("S14 input asset plan catalog/format 漂移");
        }
        plan.position_execution
            .validate_for_position(plan.position)?;
        let rebuilt = self.plan(
            plan.position,
            plan.input_token_id,
            plan.position
                .checked_add(1)
                .ok_or_else(|| anyhow!("S14 plan position overflow"))?,
        )?;
        if &rebuilt != plan {
            bail!("S14 input asset plan 不是本 catalog 的确定性派生");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct CatalogPositionAsset {
    layer: u8,
    kind: S14PositionAssetKind,
    ratio: u16,
    asset: CatalogAsset,
}

#[derive(Debug, Deserialize)]
struct CatalogDocument {
    format: String,
    repo: String,
    revision: String,
    profile: CatalogProfile,
    boundary: CatalogBoundary,
    layers: BTreeMap<String, CatalogLayer>,
}

#[derive(Debug, Deserialize)]
struct CatalogProfile {
    id: String,
    repo: String,
    revision: String,
    layers: Vec<u8>,
    compress_ratios: Vec<u16>,
}

#[derive(Debug, Deserialize)]
struct CatalogBoundary {
    embedding: Vec<CatalogAsset>,
}

#[derive(Debug, Deserialize)]
struct CatalogLayer {
    non_expert: Vec<CatalogAsset>,
}

#[derive(Debug, Clone, Deserialize)]
struct CatalogAsset {
    tensor: String,
    kind: String,
    layer: Option<u8>,
    file: String,
    file_bytes: u64,
    header_tensor_table_sha256: String,
    start: u64,
    end: u64,
    bytes: u64,
    dtype: String,
    shape: Vec<u64>,
    range_key: String,
}

fn validate_embedding_table(asset: &CatalogAsset) -> Result<()> {
    validate_catalog_asset(asset)?;
    let expected_bytes = u64::from(VOCAB_SIZE)
        .checked_mul(u64::from(HIDDEN_SIZE))
        .and_then(|value| value.checked_mul(BF16_BYTES))
        .ok_or_else(|| anyhow!("embedding table bytes overflow"))?;
    if asset.tensor != "embed.weight"
        || asset.kind != "boundary"
        || asset.layer.is_some()
        || asset.dtype != "BF16"
        || asset.shape != [u64::from(VOCAB_SIZE), u64::from(HIDDEN_SIZE)]
        || asset.bytes != expected_bytes
    {
        bail!("S14 catalog embedding table ABI 漂移");
    }
    Ok(())
}

fn validate_ape(asset: &CatalogAsset, layer: u8, ratio: u16, indexer: bool) -> Result<()> {
    validate_catalog_asset(asset)?;
    let width = match (ratio, indexer) {
        (4, false) => 1024,
        (4, true) => 256,
        (128, false) => 512,
        _ => bail!("S14 APE ratio/kind 未注册"),
    };
    if asset.kind != "non_expert"
        || asset.layer != Some(layer)
        || asset.dtype != "F32"
        || asset.shape != [u64::from(ratio), width]
    {
        bail!("S14 L{layer} APE shape/dtype/layer 漂移");
    }
    Ok(())
}

fn validate_catalog_asset(asset: &CatalogAsset) -> Result<()> {
    validate_source_file(&asset.file)?;
    validate_sha256(&asset.header_tensor_table_sha256)?;
    let bytes = asset
        .end
        .checked_sub(asset.start)
        .and_then(|delta| delta.checked_add(1))
        .ok_or_else(|| anyhow!("catalog asset Range 非法"))?;
    let element_bytes = match asset.dtype.as_str() {
        "BF16" => BF16_BYTES,
        "F32" => F32_BYTES,
        _ => bail!("input/position planner 不接受 dtype {}", asset.dtype),
    };
    let shape_bytes = asset.shape.iter().try_fold(element_bytes, |bytes, &dim| {
        if dim == 0 {
            bail!("catalog asset shape 含 0");
        }
        bytes
            .checked_mul(dim)
            .ok_or_else(|| anyhow!("catalog asset shape bytes overflow"))
    })?;
    if asset.end >= asset.file_bytes
        || asset.bytes != bytes
        || asset.bytes != shape_bytes
        || asset.range_key != format!("{}:{}-{}", asset.file, asset.start, asset.end)
    {
        bail!("catalog asset Range/shape/bytes 漂移: {}", asset.tensor);
    }
    Ok(())
}

fn find_unique_asset<'a>(assets: &'a [CatalogAsset], tensor: &str) -> Result<&'a CatalogAsset> {
    let mut matches = assets.iter().filter(|asset| asset.tensor == tensor);
    let result = matches
        .next()
        .ok_or_else(|| anyhow!("S14 catalog 缺少 position asset {tensor}"))?;
    if matches.next().is_some() {
        bail!("S14 catalog position asset 重复: {tensor}");
    }
    Ok(result)
}

fn range_cache_key(identity: &S14RangeIdentity) -> Result<String> {
    let mut canonical = BTreeMap::<&str, Value>::new();
    canonical.insert("end", Value::from(identity.end));
    canonical.insert(
        "header_tensor_table_sha256",
        Value::from(identity.header_tensor_table_sha256.clone()),
    );
    canonical.insert("repo", Value::from(identity.repo.clone()));
    canonical.insert("revision", Value::from(identity.revision.clone()));
    canonical.insert("source_file", Value::from(identity.source_file.clone()));
    canonical.insert("source_file_bytes", Value::from(identity.source_file_bytes));
    canonical.insert("start", Value::from(identity.start));
    let bytes = serde_json::to_vec(&canonical).context("编码 canonical S14 Range identity")?;
    Ok(sha256_bytes(&bytes))
}

fn validate_range_identity(identity: &S14RangeIdentity, bytes: u64) -> Result<()> {
    validate_source_file(&identity.source_file)?;
    validate_sha256(&identity.header_tensor_table_sha256)?;
    let expected = identity
        .end
        .checked_sub(identity.start)
        .and_then(|delta| delta.checked_add(1))
        .ok_or_else(|| anyhow!("S14 Range identity 非法"))?;
    if identity.repo != MODEL_REPO
        || identity.revision != MODEL_REVISION
        || identity.end >= identity.source_file_bytes
        || expected != bytes
    {
        bail!("S14 Range identity repo/revision/bounds 漂移");
    }
    Ok(())
}

fn validate_source_file(file: &str) -> Result<()> {
    if file.is_empty()
        || file == "."
        || file == ".."
        || file.contains('/')
        || file.contains('\\')
        || !file.ends_with(".safetensors")
    {
        bail!("S14 source shard 文件名非法");
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("SHA-256 必须是 64 位小写十六进制");
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
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct FixtureDir(PathBuf);

    impl FixtureDir {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "polaris-s14-input-plan-{}-{stamp}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn catalog_asset(
        tensor: String,
        kind: &str,
        layer: Option<u8>,
        file: String,
        start: u64,
        dtype: &str,
        shape: Vec<u64>,
    ) -> Value {
        let element_bytes = if dtype == "BF16" { 2 } else { 4 };
        let bytes = shape.iter().product::<u64>() * element_bytes;
        json!({
            "tensor": tensor,
            "kind": kind,
            "layer": layer,
            "file": file,
            "file_bytes": start + bytes + 4096,
            "header_tensor_table_sha256": "a".repeat(64),
            "start": start,
            "end": start + bytes - 1,
            "bytes": bytes,
            "dtype": dtype,
            "shape": shape,
            "range_key": format!("{}:{}-{}", file, start, start + bytes - 1),
        })
    }

    fn fixture_catalog() -> Vec<u8> {
        let embedding_bytes = u64::from(VOCAB_SIZE) * u64::from(HIDDEN_SIZE) * 2;
        let embedding = json!({
            "tensor": "embed.weight",
            "kind": "boundary",
            "layer": null,
            "file": "model-00001-of-00048.safetensors",
            "file_bytes": 1_059_061_856u64,
            "header_tensor_table_sha256": "a2d42126e37e08d24d127a2a536569bca972b13511650600ae8ec093dee03797",
            "start": 96,
            "end": 96 + embedding_bytes - 1,
            "bytes": embedding_bytes,
            "dtype": "BF16",
            "shape": [VOCAB_SIZE, HIDDEN_SIZE],
            "range_key": format!("model-00001-of-00048.safetensors:96-{}", 96 + embedding_bytes - 1),
        });
        let mut layers = serde_json::Map::new();
        for (&layer, &ratio) in FULL_DEPTH_LAYERS.iter().zip(COMPRESS_RATIOS.iter()) {
            let file = format!("model-layer-{layer:02}.safetensors");
            let mut non_expert = Vec::new();
            if ratio == 4 {
                non_expert.push(catalog_asset(
                    format!("layers.{layer}.attn.compressor.ape"),
                    "non_expert",
                    Some(layer),
                    file.clone(),
                    128,
                    "F32",
                    vec![4, 1024],
                ));
                non_expert.push(catalog_asset(
                    format!("layers.{layer}.attn.indexer.compressor.ape"),
                    "non_expert",
                    Some(layer),
                    file,
                    32_768,
                    "F32",
                    vec![4, 256],
                ));
            } else if ratio == 128 {
                non_expert.push(catalog_asset(
                    format!("layers.{layer}.attn.compressor.ape"),
                    "non_expert",
                    Some(layer),
                    file,
                    128,
                    "F32",
                    vec![128, 512],
                ));
            }
            layers.insert(layer.to_string(), json!({"non_expert": non_expert}));
        }
        serde_json::to_vec(&json!({
            "format": S14_FULL_DEPTH_CATALOG_FORMAT,
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "profile": {
                "id": S14_FULL_DEPTH_CATALOG_PROFILE,
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "layers": FULL_DEPTH_LAYERS.to_vec(),
                "compress_ratios": COMPRESS_RATIOS.to_vec(),
            },
            "boundary": {"embedding": [embedding]},
            "layers": layers,
        }))
        .unwrap()
    }

    fn planner(root: &Path) -> S14InputAssetPlanner {
        let bytes = fixture_catalog();
        S14InputAssetPlanner::from_verified_catalog_bytes(&bytes, sha256_bytes(&bytes), root)
            .unwrap()
    }

    #[test]
    fn arbitrary_token_and_position_derive_exact_embedding_and_ape_rows() {
        let fixture = FixtureDir::new();
        let planner = planner(&fixture.0);
        let plan = planner.plan(129, 7, 256).unwrap();
        assert_eq!(plan.position, 129);
        assert_eq!(plan.input_token_id, 7);
        assert_eq!(plan.embedding.tensor, "embed.weight[7:8]");
        assert_eq!(plan.embedding.bytes, 8192);
        assert_eq!(plan.embedding.identity.start, 96 + 7 * 8192);
        assert_eq!(plan.embedding.identity.end, 96 + 8 * 8192 - 1);
        assert_eq!(plan.position_execution.rope_position, 129);
        assert_eq!(plan.position_execution.window_slot, 1);
        assert_eq!(plan.position_execution.active_window_tokens, 128);
        assert_eq!(plan.position_execution.ape_rows.len(), 62);

        let l2_main = plan
            .position_execution
            .ape_rows
            .iter()
            .find(|row| row.layer == 2 && row.kind == S14PositionAssetKind::CompressorApe)
            .unwrap();
        assert_eq!(l2_main.ape_row, 1);
        assert_eq!(l2_main.ape_row_byte_offset, 4096);
        assert_eq!(l2_main.ape_row_bytes, 4096);
        assert_eq!(l2_main.remainder_row, 5);
        assert!(!l2_main.compressed_block_ready);
        assert_eq!(l2_main.completed_blocks_after, 32);

        let l3_main = plan
            .position_execution
            .ape_rows
            .iter()
            .find(|row| row.layer == 3)
            .unwrap();
        assert_eq!(l3_main.ape_row, 1);
        assert_eq!(l3_main.ape_row_byte_offset, 2048);
        assert_eq!(l3_main.remainder_row, 1);
        assert_eq!(l3_main.completed_blocks_after, 1);
    }

    #[test]
    fn position1_binds_token5_rope_window_and_remainder_rows() {
        let fixture = FixtureDir::new();
        let planner = planner(&fixture.0);
        let plan = planner.plan(1, 5, 4096).unwrap();
        assert_eq!(plan.embedding.tensor, "embed.weight[5:6]");
        assert_eq!(plan.position_execution.rope_position, 1);
        assert_eq!(plan.position_execution.window_slot, 1);
        assert_eq!(plan.position_execution.active_window_tokens, 2);
        for row in &plan.position_execution.ape_rows {
            assert_eq!(row.ape_row, 1);
            assert!(!row.compressed_block_ready);
            assert_eq!(row.remainder_row, if row.ratio == 4 { 5 } else { 1 });
        }
        planner.validate_plan(&plan).unwrap();
        let mut tampered = plan.clone();
        tampered.position_execution.ape_rows[0].remainder_row = 0;
        assert!(planner.validate_plan(&tampered).is_err());
    }

    #[test]
    fn compressor_boundaries_bind_official_rows_and_rope_origin() {
        let fixture = FixtureDir::new();
        let planner = planner(&fixture.0);
        let ratio4 = planner.plan(3, 11, 256).unwrap();
        for row in ratio4
            .position_execution
            .ape_rows
            .iter()
            .filter(|row| row.ratio == 4)
        {
            assert_eq!(row.ape_row, 3);
            assert_eq!(row.remainder_row, 7);
            assert!(row.compressed_block_ready);
            assert_eq!(row.compressed_rope_position, Some(0));
            assert_eq!(row.completed_blocks_after, 1);
        }
        let ratio128 = planner.plan(127, 11, 256).unwrap();
        for row in ratio128
            .position_execution
            .ape_rows
            .iter()
            .filter(|row| row.ratio == 128)
        {
            assert_eq!(row.ape_row, 127);
            assert_eq!(row.remainder_row, 127);
            assert!(row.compressed_block_ready);
            assert_eq!(row.compressed_rope_position, Some(0));
        }
    }

    #[test]
    fn frozen_bos_range_cache_key_matches_existing_python_contract() {
        let fixture = FixtureDir::new();
        let planner = planner(&fixture.0);
        let plan = planner.plan(0, 0, 1).unwrap();
        assert_eq!(
            plan.embedding.cache_key,
            "e82a65bc07bd137f74f809c07c8ff4d3a97b6e91dac94583f8e998accc7805e1"
        );
    }

    #[test]
    fn token_position_and_sequence_bounds_fail_closed() {
        let fixture = FixtureDir::new();
        let planner = planner(&fixture.0);
        assert!(planner.plan(0, VOCAB_SIZE, 1).is_err());
        assert!(planner.plan(0, 0, 0).is_err());
        assert!(planner.plan(1, 0, 1).is_err());
    }
}
