//! Online Polaris S14 top-6 routes mapped to verified expert payload ranges.
//!
//! This module deliberately stops at a production-ready paging plan. Vulkan
//! router readback and command recording consume this plan but live elsewhere.

use crate::{
    s14_input_asset_plan::{S14PlannedRangeAsset, S14RangeIdentity},
    s14_position0_mapped_assets::{VerifiedMappedAsset, VerifiedMappedAssetStore},
};
use anyhow::{bail, Context, Result};
use polaris_s14_runner::{
    Position0Asset, EXPERTS_PER_TOKEN, FULL_DEPTH_LAYERS, MODEL_REPO, MODEL_REVISION,
    N_ROUTED_EXPERTS,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    fs::File,
    io::BufReader,
    path::Path,
    sync::Arc,
};

pub const FULL_DEPTH_EXPERT_CATALOG_FORMAT: &str = "polaris-fulldepth43-native-top6-catalog-v1";
pub const DYNAMIC_ROUTED_PAGE_COUNT: usize = EXPERTS_PER_TOKEN * 3;
pub const DYNAMIC_ROUTED_RANGE_COUNT: usize = DYNAMIC_ROUTED_PAGE_COUNT * 2;

/// Online router output for one layer and token position.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct OnlineTop6 {
    pub layer: u8,
    pub position: u64,
    pub expert_ids: [u16; EXPERTS_PER_TOKEN],
    pub route_weights: [f32; EXPERTS_PER_TOKEN],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedProjection {
    W1,
    W2,
    W3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedRangePart {
    Weight,
    Scale,
}

impl RoutedProjection {
    const ALL: [Self; 3] = [Self::W1, Self::W2, Self::W3];

    pub const fn tensor_stem(self) -> &'static str {
        match self {
            Self::W1 => "w1",
            Self::W2 => "w2",
            Self::W3 => "w3",
        }
    }
}

/// Exact byte identity of one safetensors range.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ExpertRangeIdentity {
    pub tensor: String,
    pub kind: String,
    pub layer: u8,
    pub file: String,
    pub file_bytes: u64,
    pub header_tensor_table_sha256: String,
    pub start: u64,
    pub end: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub range_key: String,
    pub expert_id: u16,
}

/// One logical expert projection payload. Consumers upload `weight` and
/// `scale` together; their order here is intentionally fixed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicRoutedPage {
    pub route_slot: usize,
    pub expert_id: u16,
    pub projection: RoutedProjection,
    pub weight: ExpertRangeIdentity,
    pub scale: ExpertRangeIdentity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicRoutedPagePlan {
    pub layer: u8,
    pub position: u64,
    pub expert_ids: [u16; EXPERTS_PER_TOKEN],
    pub route_weights: [f32; EXPERTS_PER_TOKEN],
    /// Always slot-major: slot0 W1/W2/W3, then slot1 W1/W2/W3, and so on.
    pub pages: Vec<DynamicRoutedPage>,
}

#[derive(Clone, Copy, Debug)]
pub struct DynamicRoutedPhysicalRange<'a> {
    pub physical_index: usize,
    pub route_slot: usize,
    pub expert_id: u16,
    pub projection: RoutedProjection,
    pub part: RoutedRangePart,
    pub range: &'a ExpertRangeIdentity,
}

impl DynamicRoutedPhysicalRange<'_> {
    pub fn planned_asset(&self, cache_root: &Path) -> Result<S14PlannedRangeAsset> {
        S14PlannedRangeAsset::from_identity(
            cache_root,
            self.range.tensor.clone(),
            None,
            self.range.kind.clone(),
            self.range.dtype.clone(),
            self.range.shape.clone(),
            self.range.bytes,
            self.range.range_key.clone(),
            S14RangeIdentity {
                repo: MODEL_REPO.to_owned(),
                revision: MODEL_REVISION.to_owned(),
                source_file: self.range.file.clone(),
                source_file_bytes: self.range.file_bytes,
                start: self.range.start,
                end: self.range.end,
                header_tensor_table_sha256: self.range.header_tensor_table_sha256.clone(),
            },
        )
    }
}

/// A fully local, proof-checked and payload-rehashed lease set. Both vectors
/// use the same fixed physical order: slot-major, W1/W2/W3, weight then scale.
#[derive(Debug)]
pub struct MaterializedDynamicRoutedPagePlan {
    pub assets: Vec<Position0Asset>,
    pub mapped_assets: Vec<Arc<VerifiedMappedAsset>>,
}

impl DynamicRoutedPagePlan {
    /// Return the canonical 36-range source order used by materialization:
    /// slot-major, W1/W2/W3, weight then scale.
    pub fn physical_ranges(&self) -> Result<Vec<DynamicRoutedPhysicalRange<'_>>> {
        validate_dynamic_plan_structure(self)?;
        let mut output = Vec::with_capacity(DYNAMIC_ROUTED_RANGE_COUNT);
        let mut seen_ranges = HashSet::with_capacity(DYNAMIC_ROUTED_RANGE_COUNT);
        for page in &self.pages {
            for (part, range) in [
                (RoutedRangePart::Weight, &page.weight),
                (RoutedRangePart::Scale, &page.scale),
            ] {
                if !seen_ranges.insert(range.range_key.as_str()) {
                    bail!("dynamic routed physical Range 重复: {}", range.range_key);
                }
                output.push(DynamicRoutedPhysicalRange {
                    physical_index: output.len(),
                    route_slot: page.route_slot,
                    expert_id: page.expert_id,
                    projection: page.projection,
                    part,
                    range,
                });
            }
        }
        if output.len() != DYNAMIC_ROUTED_RANGE_COUNT {
            bail!("dynamic routed physical Range 数量不为 {DYNAMIC_ROUTED_RANGE_COUNT}");
        }
        Ok(output)
    }

    /// Resolve all 36 physical ranges from an existing Python Range cache.
    /// This entry point never downloads and publishes no mappings unless every
    /// proof, byte count and observed payload SHA-256 succeeds.
    pub fn materialize_cached(
        &self,
        cache_root: &Path,
    ) -> Result<MaterializedDynamicRoutedPagePlan> {
        let mut assets = Vec::with_capacity(DYNAMIC_ROUTED_RANGE_COUNT);
        for physical in self.physical_ranges()? {
            assets.push(
                physical
                    .planned_asset(cache_root)?
                    .resolve_cached_position0_asset(cache_root, Some(physical.expert_id))?,
            );
        }
        if assets.len() != DYNAMIC_ROUTED_RANGE_COUNT {
            bail!("dynamic routed physical Range 数量不为 {DYNAMIC_ROUTED_RANGE_COUNT}");
        }

        let mut store = VerifiedMappedAssetStore::new(cache_root)?;
        let mapped_assets = store.map_verified_batch(&assets)?;
        if mapped_assets.len() != DYNAMIC_ROUTED_RANGE_COUNT {
            bail!("dynamic routed verified mmap 数量不为 {DYNAMIC_ROUTED_RANGE_COUNT}");
        }
        for asset in &mut assets {
            asset.payload_rehashed_by_builder = true;
        }
        Ok(MaterializedDynamicRoutedPagePlan {
            assets,
            mapped_assets,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct FullDepthExpertCatalog {
    format: String,
    repo: String,
    revision: String,
    profile: CatalogProfile,
    headers: CatalogHeaders,
    layers: BTreeMap<String, CatalogLayer>,
}

#[derive(Debug, Deserialize)]
struct CatalogProfile {
    id: String,
    repo: String,
    revision: String,
    layers: Vec<u8>,
    top_k: usize,
}

#[derive(Debug, Deserialize)]
struct CatalogHeaders {
    files: BTreeMap<String, CatalogHeader>,
}

#[derive(Debug, Deserialize)]
struct CatalogHeader {
    file_bytes: u64,
    tensor_table_sha256: String,
}

#[derive(Debug, Deserialize)]
struct CatalogLayer {
    experts: BTreeMap<String, Vec<ExpertRangeIdentity>>,
}

impl FullDepthExpertCatalog {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("打开 expert catalog 失败: {}", path.display()))?;
        let catalog: Self = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("解析 expert catalog 失败: {}", path.display()))?;
        catalog.validate_contract()?;
        Ok(catalog)
    }

    pub fn from_json_str(json: &str) -> Result<Self> {
        let catalog: Self = serde_json::from_str(json).context("解析 expert catalog JSON 失败")?;
        catalog.validate_contract()?;
        Ok(catalog)
    }

    pub fn plan(&self, route: OnlineTop6) -> Result<DynamicRoutedPagePlan> {
        build_dynamic_routed_page_plan(self, route)
    }

    fn validate_contract(&self) -> Result<()> {
        if self.format != FULL_DEPTH_EXPERT_CATALOG_FORMAT {
            bail!("expert catalog format 不匹配: {}", self.format);
        }
        if self.repo != MODEL_REPO || self.profile.repo != MODEL_REPO {
            bail!("expert catalog repo identity 不匹配");
        }
        if self.revision != MODEL_REVISION || self.profile.revision != MODEL_REVISION {
            bail!("expert catalog revision identity 不匹配");
        }
        if self.profile.id != "fulldepth43_native_top6" {
            bail!(
                "expert catalog profile identity 不匹配: {}",
                self.profile.id
            );
        }
        if self.profile.top_k != EXPERTS_PER_TOKEN {
            bail!("expert catalog top_k 必须为 {EXPERTS_PER_TOKEN}");
        }
        if self.profile.layers.as_slice() != FULL_DEPTH_LAYERS {
            bail!("expert catalog full-depth layer 顺序不匹配");
        }
        Ok(())
    }
}

/// Minimal production entry point from online router IDs to exact paging work.
pub fn build_dynamic_routed_page_plan(
    catalog: &FullDepthExpertCatalog,
    route: OnlineTop6,
) -> Result<DynamicRoutedPagePlan> {
    catalog.validate_contract()?;
    if !FULL_DEPTH_LAYERS.contains(&route.layer) {
        bail!("router layer {} 不在 FullDepth43 中", route.layer);
    }

    let mut seen = [false; N_ROUTED_EXPERTS as usize];
    for (slot, (&expert_id, &weight)) in route
        .expert_ids
        .iter()
        .zip(route.route_weights.iter())
        .enumerate()
    {
        if expert_id >= N_ROUTED_EXPERTS {
            bail!("route slot {slot} expert_id {expert_id} 越界");
        }
        if seen[expert_id as usize] {
            bail!("route slot {slot} expert_id {expert_id} 重复");
        }
        if !weight.is_finite() {
            bail!("route slot {slot} route_weight 非有限值");
        }
        seen[expert_id as usize] = true;
    }

    let layer_key = route.layer.to_string();
    let layer = catalog
        .layers
        .get(&layer_key)
        .with_context(|| format!("catalog 缺少 layer {}", route.layer))?;
    let mut pages = Vec::with_capacity(DYNAMIC_ROUTED_PAGE_COUNT);

    for (route_slot, &expert_id) in route.expert_ids.iter().enumerate() {
        let expert_key = expert_id.to_string();
        let ranges = layer
            .experts
            .get(&expert_key)
            .with_context(|| format!("catalog layer {} 缺少 expert {}", route.layer, expert_id))?;
        if ranges.len() != RoutedProjection::ALL.len() * 2 {
            bail!(
                "catalog layer {} expert {} 必须恰有 6 个 ranges，实际 {}",
                route.layer,
                expert_id,
                ranges.len()
            );
        }

        for projection in RoutedProjection::ALL {
            let stem = projection.tensor_stem();
            let prefix = format!("layers.{}.ffn.experts.{}.{stem}", route.layer, expert_id);
            let weight_name = format!("{prefix}.weight");
            let scale_name = format!("{prefix}.scale");
            let weight = unique_range(catalog, ranges, &weight_name, route.layer, expert_id)?;
            let scale = unique_range(catalog, ranges, &scale_name, route.layer, expert_id)?;
            pages.push(DynamicRoutedPage {
                route_slot,
                expert_id,
                projection,
                weight: weight.clone(),
                scale: scale.clone(),
            });
        }
    }

    if pages.len() != DYNAMIC_ROUTED_PAGE_COUNT {
        bail!("dynamic routed page 数量不为 {DYNAMIC_ROUTED_PAGE_COUNT}");
    }
    Ok(DynamicRoutedPagePlan {
        layer: route.layer,
        position: route.position,
        expert_ids: route.expert_ids,
        route_weights: route.route_weights,
        pages,
    })
}

fn unique_range<'a>(
    catalog: &FullDepthExpertCatalog,
    ranges: &'a [ExpertRangeIdentity],
    tensor: &str,
    layer: u8,
    expert_id: u16,
) -> Result<&'a ExpertRangeIdentity> {
    let mut matches = ranges.iter().filter(|range| range.tensor == tensor);
    let range = matches
        .next()
        .with_context(|| format!("catalog 缺少 tensor {tensor}"))?;
    if matches.next().is_some() {
        bail!("catalog tensor {tensor} 重复");
    }
    validate_range_identity(catalog, range, layer, expert_id, tensor)?;
    Ok(range)
}

fn validate_dynamic_plan_structure(plan: &DynamicRoutedPagePlan) -> Result<()> {
    if !FULL_DEPTH_LAYERS.contains(&plan.layer) {
        bail!("dynamic routed plan layer 不在 FullDepth43 中");
    }
    let mut seen_experts = [false; N_ROUTED_EXPERTS as usize];
    for (slot, (&expert_id, &weight)) in plan
        .expert_ids
        .iter()
        .zip(plan.route_weights.iter())
        .enumerate()
    {
        if expert_id >= N_ROUTED_EXPERTS {
            bail!("dynamic routed plan slot {slot} expert_id 越界");
        }
        if seen_experts[expert_id as usize] {
            bail!("dynamic routed plan slot {slot} expert_id 重复");
        }
        if !weight.is_finite() {
            bail!("dynamic routed plan slot {slot} route_weight 非有限值");
        }
        seen_experts[expert_id as usize] = true;
    }
    if plan.pages.len() != DYNAMIC_ROUTED_PAGE_COUNT {
        bail!("dynamic routed page 数量不为 {DYNAMIC_ROUTED_PAGE_COUNT}");
    }
    for (slot, &expert_id) in plan.expert_ids.iter().enumerate() {
        for (projection_index, projection) in RoutedProjection::ALL.into_iter().enumerate() {
            let page = &plan.pages[slot * RoutedProjection::ALL.len() + projection_index];
            if page.route_slot != slot
                || page.expert_id != expert_id
                || page.projection != projection
            {
                bail!("dynamic routed page slot/expert/projection 顺序漂移");
            }
            let prefix = format!(
                "layers.{}.ffn.experts.{}.{}",
                plan.layer,
                expert_id,
                projection.tensor_stem()
            );
            validate_expert_range_fields(
                &page.weight,
                plan.layer,
                expert_id,
                &format!("{prefix}.weight"),
            )?;
            validate_expert_range_fields(
                &page.scale,
                plan.layer,
                expert_id,
                &format!("{prefix}.scale"),
            )?;
        }
    }
    Ok(())
}

fn validate_range_identity(
    catalog: &FullDepthExpertCatalog,
    range: &ExpertRangeIdentity,
    layer: u8,
    expert_id: u16,
    expected_tensor: &str,
) -> Result<()> {
    validate_expert_range_fields(range, layer, expert_id, expected_tensor)?;
    let header = catalog
        .headers
        .files
        .get(&range.file)
        .with_context(|| format!("catalog headers 缺少 shard {}", range.file))?;
    if header.file_bytes != range.file_bytes
        || header.tensor_table_sha256 != range.header_tensor_table_sha256
    {
        bail!("tensor {expected_tensor} shard/header identity 不匹配");
    }
    Ok(())
}

fn validate_expert_range_fields(
    range: &ExpertRangeIdentity,
    layer: u8,
    expert_id: u16,
    expected_tensor: &str,
) -> Result<()> {
    if range.tensor != expected_tensor || range.layer != layer || range.expert_id != expert_id {
        bail!("tensor {expected_tensor} 的 layer/expert identity 不匹配");
    }
    if range.kind != "routed_expert" {
        bail!("tensor {expected_tensor} kind 必须为 routed_expert");
    }
    if range.file.is_empty() || range.header_tensor_table_sha256.is_empty() {
        bail!("tensor {expected_tensor} 缺少文件 identity");
    }
    let expected_bytes = range
        .end
        .checked_sub(range.start)
        .and_then(|span| span.checked_add(1))
        .with_context(|| format!("tensor {expected_tensor} range 非法"))?;
    if range.bytes != expected_bytes {
        bail!("tensor {expected_tensor} bytes 与闭区间不一致");
    }
    let expected_range_key = format!("{}:{}-{}", range.file, range.start, range.end);
    if range.range_key != expected_range_key {
        bail!("tensor {expected_tensor} range_key 不一致");
    }
    if range.end >= range.file_bytes {
        bail!("tensor {expected_tensor} range 超出 shard 文件");
    }
    if range.dtype.is_empty() || range.shape.is_empty() {
        bail!("tensor {expected_tensor} dtype/shape identity 为空");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    const TEST_LAYER: u8 = 7;
    const TEST_FILE: &str = "model-test.safetensors";
    const TEST_HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    struct CacheDir(PathBuf);

    impl CacheDir {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "polaris-dynamic-range-cache-{}-{stamp}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for CacheDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn range(layer: u8, expert_id: u16, suffix: &str, ordinal: u64) -> ExpertRangeIdentity {
        let start = ordinal * 32;
        let end = start + 31;
        ExpertRangeIdentity {
            tensor: format!("layers.{layer}.ffn.experts.{expert_id}.{suffix}"),
            kind: "routed_expert".into(),
            layer,
            file: TEST_FILE.into(),
            file_bytes: 1_000_000,
            header_tensor_table_sha256: TEST_HASH.into(),
            start,
            end,
            bytes: 32,
            dtype: if suffix.ends_with("weight") {
                "I8"
            } else {
                "F8_E8M0"
            }
            .into(),
            shape: vec![4, 8],
            range_key: format!("{TEST_FILE}:{start}-{end}"),
            expert_id,
        }
    }

    fn fixture_catalog(ids: &[u16]) -> FullDepthExpertCatalog {
        let mut experts = BTreeMap::new();
        for &id in ids {
            let mut ranges = Vec::new();
            for (ordinal, suffix) in [
                "w1.scale",
                "w1.weight",
                "w2.scale",
                "w2.weight",
                "w3.scale",
                "w3.weight",
            ]
            .into_iter()
            .enumerate()
            {
                ranges.push(range(
                    TEST_LAYER,
                    id,
                    suffix,
                    u64::from(id) * 6 + ordinal as u64,
                ));
            }
            experts.insert(id.to_string(), ranges);
        }
        FullDepthExpertCatalog {
            format: FULL_DEPTH_EXPERT_CATALOG_FORMAT.into(),
            repo: MODEL_REPO.into(),
            revision: MODEL_REVISION.into(),
            profile: CatalogProfile {
                id: "fulldepth43_native_top6".into(),
                repo: MODEL_REPO.into(),
                revision: MODEL_REVISION.into(),
                layers: FULL_DEPTH_LAYERS.to_vec(),
                top_k: EXPERTS_PER_TOKEN,
            },
            headers: CatalogHeaders {
                files: BTreeMap::from([(
                    TEST_FILE.into(),
                    CatalogHeader {
                        file_bytes: 1_000_000,
                        tensor_table_sha256: TEST_HASH.into(),
                    },
                )]),
            },
            layers: BTreeMap::from([(TEST_LAYER.to_string(), CatalogLayer { experts })]),
        }
    }

    fn route(ids: [u16; EXPERTS_PER_TOKEN]) -> OnlineTop6 {
        OnlineTop6 {
            layer: TEST_LAYER,
            position: 19,
            expert_ids: ids,
            route_weights: [0.30, 0.25, 0.20, 0.12, 0.08, 0.05],
        }
    }

    fn planned_range(root: &Path, range: &ExpertRangeIdentity) -> S14PlannedRangeAsset {
        S14PlannedRangeAsset::from_identity(
            root,
            range.tensor.clone(),
            None,
            range.kind.clone(),
            range.dtype.clone(),
            range.shape.clone(),
            range.bytes,
            range.range_key.clone(),
            S14RangeIdentity {
                repo: MODEL_REPO.into(),
                revision: MODEL_REVISION.into(),
                source_file: range.file.clone(),
                source_file_bytes: range.file_bytes,
                start: range.start,
                end: range.end,
                header_tensor_table_sha256: range.header_tensor_table_sha256.clone(),
            },
        )
        .unwrap()
    }

    fn payload_sha256(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn write_cache(plan: &DynamicRoutedPagePlan, root: &Path) {
        for (physical_index, range) in plan
            .pages
            .iter()
            .flat_map(|page| [&page.weight, &page.scale])
            .enumerate()
        {
            let planned = planned_range(root, range);
            let payload = vec![(physical_index + 1) as u8; range.bytes as usize];
            let observed = payload_sha256(&payload);
            fs::write(&planned.payload_path, &payload).unwrap();
            let proof = json!({
                "format": crate::s14_input_asset_plan::S14_RANGE_CACHE_META_FORMAT,
                "cache_key": planned.cache_key,
                "identity": planned.identity,
                "bytes": planned.bytes,
                "observed_sha256": observed,
                "expected_sha256": null,
                "hash_authority": "tofu",
                "authoritative": false,
                "verified_transport":
                    crate::s14_input_asset_plan::S14_RANGE_CACHE_VERIFIED_TRANSPORT,
            });
            fs::write(&planned.proof_path, serde_json::to_vec(&proof).unwrap()).unwrap();
        }
    }

    #[test]
    fn maps_online_ids_to_eighteen_slot_major_real_expert_pages() {
        let ids = [126, 3, 250, 17, 99, 42];
        let plan = fixture_catalog(&ids).plan(route(ids)).unwrap();

        assert_eq!(plan.pages.len(), DYNAMIC_ROUTED_PAGE_COUNT);
        for (slot, expert_id) in ids.into_iter().enumerate() {
            let pages = &plan.pages[slot * 3..slot * 3 + 3];
            assert_eq!(
                pages.iter().map(|page| page.route_slot).collect::<Vec<_>>(),
                vec![slot; 3]
            );
            assert_eq!(
                pages.iter().map(|page| page.expert_id).collect::<Vec<_>>(),
                vec![expert_id; 3]
            );
            assert_eq!(
                pages.iter().map(|page| page.projection).collect::<Vec<_>>(),
                RoutedProjection::ALL
            );
            assert!(pages[0].weight.tensor.ends_with("w1.weight"));
            assert!(pages[0].scale.tensor.ends_with("w1.scale"));
        }
        assert_eq!(plan.expert_ids, ids);
    }

    #[test]
    fn rejects_duplicate_online_expert_identity() {
        let ids = [1, 2, 3, 4, 5, 1];
        assert!(fixture_catalog(&ids)
            .plan(route(ids))
            .unwrap_err()
            .to_string()
            .contains("重复"));
    }

    #[test]
    fn rejects_out_of_range_online_expert_identity() {
        let ids = [1, 2, 3, 4, 5, N_ROUTED_EXPERTS];
        assert!(fixture_catalog(&ids)
            .plan(route(ids))
            .unwrap_err()
            .to_string()
            .contains("越界"));
    }

    #[test]
    fn rejects_missing_or_duplicate_projection_range() {
        let ids = [1, 2, 3, 4, 5, 6];
        let mut catalog = fixture_catalog(&ids);
        let ranges = catalog
            .layers
            .get_mut("7")
            .unwrap()
            .experts
            .get_mut("1")
            .unwrap();
        ranges[1] = ranges[0].clone();
        assert!(catalog
            .plan(route(ids))
            .unwrap_err()
            .to_string()
            .contains("缺少 tensor"));
    }

    #[test]
    fn rejects_layer_or_expert_identity_drift() {
        let ids = [1, 2, 3, 4, 5, 6];
        let mut catalog = fixture_catalog(&ids);
        catalog
            .layers
            .get_mut("7")
            .unwrap()
            .experts
            .get_mut("1")
            .unwrap()[0]
            .expert_id = 9;
        assert!(catalog
            .plan(route(ids))
            .unwrap_err()
            .to_string()
            .contains("identity"));
    }

    #[test]
    fn rejects_range_and_header_identity_drift() {
        let ids = [1, 2, 3, 4, 5, 6];
        let mut catalog = fixture_catalog(&ids);
        catalog
            .layers
            .get_mut("7")
            .unwrap()
            .experts
            .get_mut("1")
            .unwrap()[0]
            .range_key = "wrong:0-31".into();
        assert!(catalog
            .plan(route(ids))
            .unwrap_err()
            .to_string()
            .contains("range_key"));

        let mut catalog = fixture_catalog(&ids);
        catalog
            .layers
            .get_mut("7")
            .unwrap()
            .experts
            .get_mut("1")
            .unwrap()[0]
            .header_tensor_table_sha256 = "wrong".into();
        assert!(catalog
            .plan(route(ids))
            .unwrap_err()
            .to_string()
            .contains("header identity"));
    }

    #[test]
    fn expert_range_cache_key_and_paths_match_python_contract() {
        let fixture = CacheDir::new();
        let ids = [1, 2, 3, 4, 5, 6];
        let plan = fixture_catalog(&ids).plan(route(ids)).unwrap();
        let planned = planned_range(&fixture.0, &plan.pages[0].weight);
        assert_eq!(
            planned.cache_key,
            "d8b72a6047463f4b11303e21a65bbf4da4a1d4923c78ec05a7dc43eab1b1d900"
        );
        assert_eq!(
            planned.payload_path.file_name().unwrap(),
            "d8b72a6047463f4b11303e21a65bbf4da4a1d4923c78ec05a7dc43eab1b1d900.bin"
        );
        assert_eq!(
            planned.proof_path.file_name().unwrap(),
            "d8b72a6047463f4b11303e21a65bbf4da4a1d4923c78ec05a7dc43eab1b1d900.json"
        );
    }

    #[test]
    fn materializes_thirty_six_ranges_in_fixed_order() {
        let fixture = CacheDir::new();
        let ids = [1, 2, 3, 4, 5, 6];
        let plan = fixture_catalog(&ids).plan(route(ids)).unwrap();
        write_cache(&plan, &fixture.0);

        let materialized = plan.materialize_cached(&fixture.0).unwrap();
        assert_eq!(materialized.assets.len(), DYNAMIC_ROUTED_RANGE_COUNT);
        assert_eq!(materialized.mapped_assets.len(), DYNAMIC_ROUTED_RANGE_COUNT);
        let expected_first_slot = [
            "w1.weight",
            "w1.scale",
            "w2.weight",
            "w2.scale",
            "w3.weight",
            "w3.scale",
        ];
        for (asset, suffix) in materialized.assets[..6].iter().zip(expected_first_slot) {
            assert!(asset.tensor.ends_with(suffix));
            assert_eq!(asset.expert_id, Some(ids[0]));
            assert!(asset.payload_rehashed_by_builder);
        }
        for (asset, mapped) in materialized.assets.iter().zip(&materialized.mapped_assets) {
            assert_eq!(asset.tensor, mapped.tensor());
            assert_eq!(asset.bytes as usize, mapped.bytes().len());
            assert_eq!(asset.sha256, mapped.expected_sha256());
        }
    }

    #[test]
    fn missing_payload_or_proof_fails_closed() {
        let fixture = CacheDir::new();
        let ids = [1, 2, 3, 4, 5, 6];
        let plan = fixture_catalog(&ids).plan(route(ids)).unwrap();
        write_cache(&plan, &fixture.0);
        let first = planned_range(&fixture.0, &plan.pages[0].weight);
        fs::remove_file(&first.proof_path).unwrap();
        assert!(plan.materialize_cached(&fixture.0).is_err());

        write_cache(&plan, &fixture.0);
        fs::remove_file(&first.payload_path).unwrap();
        assert!(plan.materialize_cached(&fixture.0).is_err());
    }

    #[test]
    fn proof_identity_or_observed_payload_hash_drift_fails_closed() {
        let fixture = CacheDir::new();
        let ids = [1, 2, 3, 4, 5, 6];
        let plan = fixture_catalog(&ids).plan(route(ids)).unwrap();
        write_cache(&plan, &fixture.0);
        let first = planned_range(&fixture.0, &plan.pages[0].weight);
        let mut proof: serde_json::Value =
            serde_json::from_slice(&fs::read(&first.proof_path).unwrap()).unwrap();
        proof["identity"]["start"] = json!(0);
        fs::write(&first.proof_path, serde_json::to_vec(&proof).unwrap()).unwrap();
        assert!(plan.materialize_cached(&fixture.0).is_err());

        write_cache(&plan, &fixture.0);
        fs::write(&first.payload_path, vec![0xff; first.bytes as usize]).unwrap();
        assert!(plan.materialize_cached(&fixture.0).is_err());
    }
}
