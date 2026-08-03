//! K=4/8 causal-block 每层 top-6 union Range 的 production 身份与物化门。
//!
//! 本模块不录制 attention/MoE kernel；它把 K 份在线 route 合并成按专家排序的唯一
//! 物理 Range，逐字段绑定 catalog/Range proof，并通过一个跨层、跨 token 常驻的
//! `VerifiedMappedAssetStore` 批量 SHA。物化结果可直接写入持久 staging，随后整层只需
//! 一条 `vkCmdCopyBuffer` 上传到 union bank。

use crate::{
    s14_causal_block_layer::{S14CausalBlockLayerRangePlan, S14CausalBlockPhysicalRange},
    s14_dynamic_page_cache_readiness::{
        materialize_dynamic_page_plans_batched, DynamicPageFetchMode,
    },
    s14_dynamic_routed_page_plan::{
        DynamicRoutedPagePlan, ExpertRangeIdentity, FullDepthExpertCatalog, OnlineTop6,
        RoutedProjection, RoutedRangePart,
    },
    s14_input_asset_plan::{S14PlannedRangeAsset, S14RangeIdentity},
    s14_position0_mapped_assets::{
        VerifiedMappedAsset, VerifiedMappedAssetStats, VerifiedMappedAssetStore,
    },
    GpuBuffer,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    Position0Asset, RouteDecision, EXPERTS_PER_TOKEN, EXPERT_PAGE_BYTES, MODEL_REPO, MODEL_REVISION,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
};

const PHYSICAL_RANGES_PER_EXPERT: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PhysicalKey {
    expert_id: u16,
    projection: u8,
    part: u8,
}

impl PhysicalKey {
    fn new(expert_id: u16, projection: RoutedProjection, part: RoutedRangePart) -> Self {
        Self {
            expert_id,
            projection: projection_ordinal(projection),
            part: part_ordinal(part),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14CausalBlockUnionPlannedRange {
    pub expert_id: u16,
    pub projection: RoutedProjection,
    pub part: RoutedRangePart,
    pub planned: S14PlannedRangeAsset,
}

/// 已把 K lanes 的 catalog 身份合并为唯一物理 Range，但尚未打开 payload。
#[derive(Clone, Debug)]
pub struct S14CausalBlockUnionIdentityPlan {
    pub layer: u8,
    pub base_position: u32,
    pub block_size: usize,
    pub unique_experts: usize,
    pub physical_ranges: usize,
    pub union_expert_bytes: u64,
    pub ranges: Vec<S14CausalBlockUnionPlannedRange>,
    route_plans: Vec<DynamicRoutedPagePlan>,
}

impl S14CausalBlockUnionIdentityPlan {
    pub fn route_plans(&self) -> &[DynamicRoutedPagePlan] {
        &self.route_plans
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14CausalBlockUnionMaterializeTelemetry {
    pub unique_experts: usize,
    pub physical_ranges: usize,
    pub proof_assets: usize,
    pub explicit_fetch_lane_plans: usize,
    pub mmap_requests_this_call: u64,
    pub mmap_hits_this_call: u64,
    pub mmap_misses_this_call: u64,
    pub sha256_bytes_this_call: u64,
    /// 物理 Range 在 host 上拼成连续 union；GPU 上传必须是一条 copy region。
    pub staging_range_copies: usize,
    pub gpu_upload_copy_regions: u32,
}

/// 一层 union 的 proof/SHA lease。`mapped_assets` 顺序与 layer range plan 完全一致。
#[derive(Debug)]
pub struct S14CausalBlockMaterializedUnion {
    pub layer: u8,
    pub unique_experts: usize,
    pub union_expert_bytes: u64,
    pub assets: Vec<Position0Asset>,
    pub mapped_assets: Vec<Arc<VerifiedMappedAsset>>,
    pub telemetry: S14CausalBlockUnionMaterializeTelemetry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockUnionStageReceipt {
    pub staged_bytes: u64,
    pub host_range_copies: usize,
    pub gpu_upload_copy_regions: u32,
}

impl S14CausalBlockMaterializedUnion {
    /// 直接写入 runtime 常驻的 HOST_COHERENT staging，不构造整页临时 `Vec`。
    pub fn stage_into_gpu(&self, staging: &GpuBuffer) -> Result<S14CausalBlockUnionStageReceipt> {
        if staging.mapped().is_null() {
            bail!("causal-block union staging 不是持久映射 buffer");
        }
        if staging.size() < self.union_expert_bytes {
            bail!(
                "causal-block union staging 容量不足: required={} allocated={}",
                self.union_expert_bytes,
                staging.size()
            );
        }
        if self.mapped_assets.len() != self.telemetry.physical_ranges {
            bail!("causal-block union mapped Range 数量漂移");
        }
        let mut offset = 0usize;
        for mapped in &self.mapped_assets {
            let bytes = mapped.bytes();
            let end = offset
                .checked_add(bytes.len())
                .context("causal-block union staging offset overflow")?;
            if end > self.union_expert_bytes as usize {
                bail!("causal-block union mapped payload 超出 used bytes");
            }
            unsafe { staging.write_at(offset, bytes) };
            offset = end;
        }
        if offset as u64 != self.union_expert_bytes {
            bail!("causal-block union staging bytes 与 Range plan 漂移");
        }
        Ok(S14CausalBlockUnionStageReceipt {
            staged_bytes: self.union_expert_bytes,
            host_range_copies: self.mapped_assets.len(),
            gpu_upload_copy_regions: 1,
        })
    }

    /// Vulkan backend 应把本 region 作为整层唯一一次 staging→union-bank copy。
    pub fn upload_copy_region(&self) -> vk::BufferCopy {
        vk::BufferCopy::default().size(self.union_expert_bytes)
    }
}

/// 跨 bundle/block 共享的 verified mmap/SHA owner。只共享不可变的文件
/// lease，不共享 route、union placement、GPU upload 或任何模型状态。
#[derive(Clone, Debug)]
pub struct S14CausalBlockSharedMappedAssetStore {
    cache_root: PathBuf,
    inner: Arc<Mutex<VerifiedMappedAssetStore>>,
}

impl S14CausalBlockSharedMappedAssetStore {
    pub fn new(cache_root: &Path) -> Result<Self> {
        let cache_root = cache_root.canonicalize().with_context(|| {
            format!(
                "resolve causal-block shared cache root {}",
                cache_root.display()
            )
        })?;
        let store = VerifiedMappedAssetStore::new(&cache_root)?;
        Ok(Self {
            cache_root,
            inner: Arc::new(Mutex::new(store)),
        })
    }

    pub fn stats(&self) -> Result<VerifiedMappedAssetStats> {
        Ok(self.lock()?.stats())
    }

    fn validate_cache_root(&self, cache_root: &Path) -> Result<()> {
        let observed = cache_root.canonicalize().with_context(|| {
            format!(
                "resolve causal-block materializer cache root {}",
                cache_root.display()
            )
        })?;
        if observed != self.cache_root {
            bail!(
                "causal-block shared mapped store cache root漂移: shared={} observed={}",
                self.cache_root.display(),
                observed.display()
            );
        }
        Ok(())
    }

    fn lock(&self) -> Result<MutexGuard<'_, VerifiedMappedAssetStore>> {
        self.inner
            .lock()
            .map_err(|_| anyhow::anyhow!("causal-block shared mapped store mutex poisoned"))
    }
}

/// 跨层、跨 token 常驻。重复 Range 只命中 mmap lease，不重复做 payload SHA。
#[derive(Debug)]
pub struct S14CausalBlockUnionMaterializer {
    cache_root: PathBuf,
    fetch_mode: DynamicPageFetchMode,
    store: S14CausalBlockSharedMappedAssetStore,
}

impl S14CausalBlockUnionMaterializer {
    pub fn new(cache_root: &Path, fetch_mode: DynamicPageFetchMode) -> Result<Self> {
        let store = S14CausalBlockSharedMappedAssetStore::new(cache_root)?;
        Self::with_shared_store(cache_root, fetch_mode, store)
    }

    pub fn with_shared_store(
        cache_root: &Path,
        fetch_mode: DynamicPageFetchMode,
        store: S14CausalBlockSharedMappedAssetStore,
    ) -> Result<Self> {
        store.validate_cache_root(cache_root)?;
        Ok(Self {
            cache_root: cache_root.to_path_buf(),
            fetch_mode,
            store,
        })
    }

    pub fn store_stats(&self) -> Result<VerifiedMappedAssetStats> {
        self.store.stats()
    }

    pub fn materialize(
        &mut self,
        identity_plan: &S14CausalBlockUnionIdentityPlan,
    ) -> Result<S14CausalBlockMaterializedUnion> {
        let mut explicit_fetch_lane_plans = 0usize;
        let mut assets = match resolve_union_assets(identity_plan, &self.cache_root) {
            Ok(assets) => assets,
            Err(local_error) if self.fetch_mode == DynamicPageFetchMode::LocalOnly => {
                return Err(local_error.context("causal-block union Range local-only 物化失败"));
            }
            Err(_) => {
                // K lanes 已在本层同时可知；把完全相同的物理 Range 去重后只启动一次
                // transport，随后仍逐 lane 复核 proof/SHA，再按 union 计划唯一映射。
                materialize_dynamic_page_plans_batched(
                    &identity_plan.route_plans,
                    &self.cache_root,
                    DynamicPageFetchMode::ExplicitFetch,
                )
                .map_err(|error| anyhow::anyhow!(error))
                .context("causal-block batched lane Range 显式 fetch/proof/SHA 失败")?;
                explicit_fetch_lane_plans = identity_plan.route_plans.len();
                resolve_union_assets(identity_plan, &self.cache_root)
                    .context("causal-block union fetch 后 proof identity 复核失败")?
            }
        };

        let (before, mapped_assets, after) = {
            let mut store = self.store.lock()?;
            let before = store.stats();
            let mapped_assets = store
                .map_verified_batch(&assets)
                .context("causal-block union payload 批量 SHA/mmap 失败")?;
            let after = store.stats();
            (before, mapped_assets, after)
        };
        if mapped_assets.len() != identity_plan.physical_ranges {
            bail!("causal-block union mapped Range 数量漂移");
        }
        for (planned, (asset, mapped)) in identity_plan
            .ranges
            .iter()
            .zip(assets.iter_mut().zip(&mapped_assets))
        {
            planned
                .planned
                .validate_resolved_position0_asset(asset, Some(planned.expert_id))?;
            if mapped.tensor() != planned.planned.tensor
                || mapped.expected_sha256() != asset.sha256
                || mapped.bytes().len() as u64 != planned.planned.bytes
            {
                bail!("causal-block union mapped lease 与 planned Range 漂移");
            }
            asset.payload_rehashed_by_builder = true;
        }
        let telemetry = S14CausalBlockUnionMaterializeTelemetry {
            unique_experts: identity_plan.unique_experts,
            physical_ranges: identity_plan.physical_ranges,
            proof_assets: assets.len(),
            explicit_fetch_lane_plans,
            mmap_requests_this_call: checked_delta(after.requests, before.requests, "requests")?,
            mmap_hits_this_call: checked_delta(after.hits, before.hits, "hits")?,
            mmap_misses_this_call: checked_delta(after.misses, before.misses, "misses")?,
            sha256_bytes_this_call: checked_delta(
                after.sha256_bytes,
                before.sha256_bytes,
                "sha256_bytes",
            )?,
            staging_range_copies: mapped_assets.len(),
            gpu_upload_copy_regions: 1,
        };
        Ok(S14CausalBlockMaterializedUnion {
            layer: identity_plan.layer,
            unique_experts: identity_plan.unique_experts,
            union_expert_bytes: identity_plan.union_expert_bytes,
            assets,
            mapped_assets,
            telemetry,
        })
    }
}

/// 从 K lanes 的真实 route 重建完整 catalog identity，并逐项绑定 orchestrator 的瘦
/// `S14CausalBlockLayerRangePlan`。任何 tensor/range_key/bytes/order 漂移均在 I/O 前拒绝。
pub fn build_causal_block_union_identity_plan(
    catalog: &FullDepthExpertCatalog,
    cache_root: &Path,
    base_position: u32,
    routes: &[RouteDecision],
    range_plan: &S14CausalBlockLayerRangePlan,
) -> Result<S14CausalBlockUnionIdentityPlan> {
    if !matches!(range_plan.block_size, 4 | 8) || routes.len() != range_plan.block_size {
        bail!("causal-block union identity 要求精确 K=4/8 routes");
    }
    let expected_physical_ranges = range_plan
        .unique_experts
        .checked_mul(PHYSICAL_RANGES_PER_EXPERT)
        .context("causal-block union physical range count overflow")?;
    if range_plan.physical_ranges != expected_physical_ranges
        || range_plan.ranges.len() != expected_physical_ranges
        || range_plan.union_expert_bytes
            != (range_plan.unique_experts as u64)
                .checked_mul(EXPERT_PAGE_BYTES)
                .context("causal-block union expert bytes overflow")?
    {
        bail!("causal-block union layer range plan count/bytes 漂移");
    }

    let route_plans = build_causal_block_route_plans(catalog, base_position, routes)?;
    let mut identities = BTreeMap::<PhysicalKey, ExpertRangeIdentity>::new();
    let mut union_experts = BTreeSet::new();
    for route_plan in &route_plans {
        if route_plan.layer != range_plan.layer {
            bail!("causal-block union route layer 漂移");
        }
        for physical in route_plan.physical_ranges()? {
            union_experts.insert(physical.expert_id);
            let key = PhysicalKey::new(physical.expert_id, physical.projection, physical.part);
            if let Some(existing) = identities.get(&key) {
                if existing != physical.range {
                    bail!("causal-block 同一 union Range 完整 identity 漂移");
                }
            } else {
                identities.insert(key, physical.range.clone());
            }
        }
    }
    if union_experts.len() != range_plan.unique_experts
        || identities.len() != expected_physical_ranges
    {
        bail!("causal-block union 专家/Range 去重数量漂移");
    }

    let expected_keys = union_experts
        .iter()
        .flat_map(|&expert_id| {
            [
                RoutedProjection::W1,
                RoutedProjection::W2,
                RoutedProjection::W3,
            ]
            .into_iter()
            .flat_map(move |projection| {
                [RoutedRangePart::Weight, RoutedRangePart::Scale]
                    .into_iter()
                    .map(move |part| PhysicalKey::new(expert_id, projection, part))
            })
        })
        .collect::<Vec<_>>();
    let mut ranges = Vec::with_capacity(expected_physical_ranges);
    let mut observed_bytes = 0u64;
    let mut expert_bytes = BTreeMap::<u16, u64>::new();
    for (expected_key, thin) in expected_keys.iter().zip(&range_plan.ranges) {
        let observed_key = PhysicalKey::new(thin.expert_id, thin.projection, thin.part);
        if observed_key != *expected_key {
            bail!("causal-block union physical Range canonical order 漂移");
        }
        let identity = identities
            .get(expected_key)
            .context("causal-block union 缺少完整 Range identity")?;
        validate_thin_range(thin, identity)?;
        observed_bytes = observed_bytes
            .checked_add(identity.bytes)
            .context("causal-block union bytes overflow")?;
        let bytes = expert_bytes.entry(thin.expert_id).or_default();
        *bytes = bytes
            .checked_add(identity.bytes)
            .context("causal-block union expert page bytes overflow")?;
        ranges.push(S14CausalBlockUnionPlannedRange {
            expert_id: thin.expert_id,
            projection: thin.projection,
            part: thin.part,
            planned: planned_asset(cache_root, identity)?,
        });
    }
    if observed_bytes != range_plan.union_expert_bytes
        || expert_bytes
            .values()
            .any(|&bytes| bytes != EXPERT_PAGE_BYTES)
    {
        bail!("causal-block union 每专家页/总字节不等于 production ABI");
    }
    Ok(S14CausalBlockUnionIdentityPlan {
        layer: range_plan.layer,
        base_position,
        block_size: range_plan.block_size,
        unique_experts: range_plan.unique_experts,
        physical_ranges: range_plan.physical_ranges,
        union_expert_bytes: range_plan.union_expert_bytes,
        ranges,
        route_plans,
    })
}

/// Build the exact per-lane catalog plans as soon as the authoritative GPU
/// router output is available.  The resulting plans are suitable for a
/// cache-only prefetch ticket; the later union identity builder reconstructs
/// them independently and requires exact equality before publication.
pub fn build_causal_block_route_plans(
    catalog: &FullDepthExpertCatalog,
    base_position: u32,
    routes: &[RouteDecision],
) -> Result<Vec<DynamicRoutedPagePlan>> {
    if !matches!(routes.len(), 4 | 8) {
        bail!("causal-block route plans 要求精确K=4/8 routes");
    }
    let layer = routes[0].layer;
    routes
        .iter()
        .enumerate()
        .map(|(lane, route)| {
            if route.layer != layer {
                bail!("causal-block route plans layer 漂移");
            }
            let expert_ids: [u16; EXPERTS_PER_TOKEN] = route
                .expert_ids
                .clone()
                .try_into()
                .map_err(|_| anyhow::anyhow!("causal-block route 不是精确 top-6 IDs"))?;
            let route_weights: [f32; EXPERTS_PER_TOKEN] = route
                .weights
                .clone()
                .try_into()
                .map_err(|_| anyhow::anyhow!("causal-block route 不是精确 top-6 weights"))?;
            let position = u64::from(base_position)
                .checked_add(lane as u64)
                .context("causal-block route plan position overflow")?;
            catalog
                .plan(OnlineTop6 {
                    layer,
                    position,
                    expert_ids,
                    route_weights,
                })
                .context("causal-block catalog route plan 失败")
        })
        .collect()
}

fn resolve_union_assets(
    plan: &S14CausalBlockUnionIdentityPlan,
    cache_root: &Path,
) -> Result<Vec<Position0Asset>> {
    plan.ranges
        .iter()
        .map(|range| {
            range
                .planned
                .resolve_cached_position0_asset(cache_root, Some(range.expert_id))
                .with_context(|| format!("解析 union Range {}", range.planned.tensor))
        })
        .collect()
}

fn planned_asset(cache_root: &Path, range: &ExpertRangeIdentity) -> Result<S14PlannedRangeAsset> {
    S14PlannedRangeAsset::from_identity(
        cache_root,
        range.tensor.clone(),
        None,
        range.kind.clone(),
        range.dtype.clone(),
        range.shape.clone(),
        range.bytes,
        range.range_key.clone(),
        S14RangeIdentity {
            repo: MODEL_REPO.to_owned(),
            revision: MODEL_REVISION.to_owned(),
            source_file: range.file.clone(),
            source_file_bytes: range.file_bytes,
            start: range.start,
            end: range.end,
            header_tensor_table_sha256: range.header_tensor_table_sha256.clone(),
        },
    )
}

fn validate_thin_range(
    thin: &S14CausalBlockPhysicalRange,
    identity: &ExpertRangeIdentity,
) -> Result<()> {
    if thin.expert_id != identity.expert_id
        || thin.tensor != identity.tensor
        || thin.range_key != identity.range_key
        || thin.bytes != identity.bytes
    {
        bail!("causal-block union tensor/range_key/bytes identity 漂移");
    }
    Ok(())
}

fn checked_delta(after: u64, before: u64, field: &str) -> Result<u64> {
    after
        .checked_sub(before)
        .with_context(|| format!("causal-block mapped telemetry {field} underflow"))
}

const fn projection_ordinal(projection: RoutedProjection) -> u8 {
    match projection {
        RoutedProjection::W1 => 0,
        RoutedProjection::W2 => 1,
        RoutedProjection::W3 => 2,
    }
}

const fn part_ordinal(part: RoutedRangePart) -> u8 {
    match part {
        RoutedRangePart::Weight => 0,
        RoutedRangePart::Scale => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s14_dynamic_routed_packing::{
        S14_DYNAMIC_ROUTED_SCALE_BYTES, S14_DYNAMIC_ROUTED_WEIGHT_BYTES,
    };
    use polaris_s14_runner::{router_kind_for_layer, FULL_DEPTH_LAYERS};
    use serde_json::json;

    const TEST_LAYER: u8 = 7;
    const TEST_FILE: &str = "model-test.safetensors";
    const TEST_HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn k4_k8_build_one_canonical_union_with_six_ranges_per_expert() {
        for block_size in [4, 8] {
            let routes = routes(block_size);
            let catalog = catalog(&routes);
            let range_plan = layer_range_plan(&catalog, &routes, block_size);
            let plan = build_causal_block_union_identity_plan(
                &catalog,
                Path::new("fixture-cache"),
                17,
                &routes,
                &range_plan,
            )
            .unwrap();
            assert_eq!(plan.block_size, block_size);
            assert_eq!(plan.physical_ranges, plan.unique_experts * 6);
            assert_eq!(
                plan.union_expert_bytes,
                plan.unique_experts as u64 * EXPERT_PAGE_BYTES
            );
            assert_eq!(plan.route_plans.len(), block_size);
            for expert in plan.ranges.chunks_exact(6) {
                assert!(expert
                    .iter()
                    .all(|range| range.expert_id == expert[0].expert_id));
                assert_eq!(
                    expert.iter().map(|range| range.planned.bytes).sum::<u64>(),
                    EXPERT_PAGE_BYTES
                );
            }
        }
    }

    #[test]
    fn tensor_range_key_bytes_and_order_drift_are_rejected_before_io() {
        let routes = routes(4);
        let catalog = catalog(&routes);
        let base = layer_range_plan(&catalog, &routes, 4);
        for mutate in 0..4 {
            let mut drift = base.clone();
            match mutate {
                0 => drift.ranges[0].tensor.push_str(".drift"),
                1 => drift.ranges[0].range_key.push_str(".drift"),
                2 => drift.ranges[0].bytes -= 1,
                3 => drift.ranges.swap(0, 1),
                _ => unreachable!(),
            }
            assert!(build_causal_block_union_identity_plan(
                &catalog,
                Path::new("fixture-cache"),
                0,
                &routes,
                &drift,
            )
            .is_err());
        }
    }

    fn routes(block_size: usize) -> Vec<RouteDecision> {
        (0..block_size)
            .map(|lane| RouteDecision {
                layer: TEST_LAYER,
                kind: router_kind_for_layer(TEST_LAYER).unwrap(),
                expert_ids: (lane as u16..lane as u16 + EXPERTS_PER_TOKEN as u16).collect(),
                weights: vec![0.25; EXPERTS_PER_TOKEN],
            })
            .collect()
    }

    fn layer_range_plan(
        catalog: &FullDepthExpertCatalog,
        routes: &[RouteDecision],
        block_size: usize,
    ) -> S14CausalBlockLayerRangePlan {
        let mut identities =
            BTreeMap::<PhysicalKey, (RoutedProjection, RoutedRangePart, ExpertRangeIdentity)>::new(
            );
        for (lane, route) in routes.iter().enumerate() {
            let plan = catalog
                .plan(OnlineTop6 {
                    layer: route.layer,
                    position: lane as u64,
                    expert_ids: route.expert_ids.clone().try_into().unwrap(),
                    route_weights: route.weights.clone().try_into().unwrap(),
                })
                .unwrap();
            for physical in plan.physical_ranges().unwrap() {
                identities
                    .entry(PhysicalKey::new(
                        physical.expert_id,
                        physical.projection,
                        physical.part,
                    ))
                    .or_insert((physical.projection, physical.part, physical.range.clone()));
            }
        }
        let ranges = identities
            .values()
            .map(|(projection, part, range)| S14CausalBlockPhysicalRange {
                expert_id: range.expert_id,
                projection: *projection,
                part: *part,
                tensor: range.tensor.clone(),
                range_key: range.range_key.clone(),
                bytes: range.bytes,
            })
            .collect::<Vec<_>>();
        let unique_experts = routes
            .iter()
            .flat_map(|route| route.expert_ids.iter().copied())
            .collect::<BTreeSet<_>>()
            .len();
        S14CausalBlockLayerRangePlan {
            layer: TEST_LAYER,
            block_size,
            unique_experts,
            physical_ranges: ranges.len(),
            union_expert_bytes: ranges.iter().map(|range| range.bytes).sum(),
            ranges,
        }
    }

    fn catalog(routes: &[RouteDecision]) -> FullDepthExpertCatalog {
        let expert_ids = routes
            .iter()
            .flat_map(|route| route.expert_ids.iter().copied())
            .collect::<BTreeSet<_>>();
        let mut ordinal = 0u64;
        let file_bytes = 100_000_000_000u64;
        let mut layers = serde_json::Map::new();
        for &layer in &FULL_DEPTH_LAYERS {
            let mut experts = serde_json::Map::new();
            for &expert_id in &expert_ids {
                let mut ranges = Vec::with_capacity(6);
                for projection in ["w1", "w2", "w3"] {
                    for (part, dtype, shape, bytes) in [
                        (
                            "weight",
                            "I8",
                            vec![2048u64, 2048],
                            S14_DYNAMIC_ROUTED_WEIGHT_BYTES,
                        ),
                        (
                            "scale",
                            "F8_E8M0",
                            vec![2048u64, 128],
                            S14_DYNAMIC_ROUTED_SCALE_BYTES,
                        ),
                    ] {
                        let start = ordinal * S14_DYNAMIC_ROUTED_WEIGHT_BYTES;
                        let end = start + bytes - 1;
                        ranges.push(json!({
                            "tensor": format!(
                                "layers.{layer}.ffn.experts.{expert_id}.{projection}.{part}"
                            ),
                            "kind": "routed_expert",
                            "layer": layer,
                            "file": TEST_FILE,
                            "file_bytes": file_bytes,
                            "header_tensor_table_sha256": TEST_HASH,
                            "start": start,
                            "end": end,
                            "bytes": bytes,
                            "dtype": dtype,
                            "shape": shape,
                            "range_key": format!("{TEST_FILE}:{start}-{end}"),
                            "expert_id": expert_id,
                        }));
                        ordinal += 1;
                    }
                }
                experts.insert(expert_id.to_string(), json!(ranges));
            }
            layers.insert(layer.to_string(), json!({ "experts": experts }));
        }
        FullDepthExpertCatalog::from_json_str(
            &json!({
                "format": crate::s14_dynamic_routed_page_plan::FULL_DEPTH_EXPERT_CATALOG_FORMAT,
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "profile": {
                    "id": "fulldepth43_native_top6",
                    "repo": MODEL_REPO,
                    "revision": MODEL_REVISION,
                    "layers": FULL_DEPTH_LAYERS.to_vec(),
                    "top_k": EXPERTS_PER_TOKEN,
                },
                "headers": {
                    "files": {
                        TEST_FILE: {
                            "file_bytes": file_bytes,
                            "tensor_table_sha256": TEST_HASH,
                        }
                    }
                },
                "layers": layers,
            })
            .to_string(),
        )
        .unwrap()
    }
}
