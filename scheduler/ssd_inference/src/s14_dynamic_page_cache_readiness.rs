//! Readiness and explicitly authorized fetch for an online top-6 routed page plan.
//!
//! Inspection never downloads. It derives the Python Range cache identity and
//! paths for all 36 physical ranges, validates any committed proof/payload
//! through the production read-only mmap gate, and reports every unready page.
//! Materialization defaults to local-only; explicit fetch is followed by Rust
//! proof/identity/payload-SHA verification and canonical routed arena packing.

use crate::{
    s14_dynamic_routed_packing::{S14DynamicRoutedUploadPlan, S14_DYNAMIC_ROUTED_ARENA_BYTES},
    s14_dynamic_routed_page_plan::{
        DynamicRoutedPagePlan, MaterializedDynamicRoutedPagePlan, OnlineTop6, RoutedRangePart,
        DYNAMIC_ROUTED_RANGE_COUNT,
    },
    s14_input_asset_plan::S14PlannedRangeAsset,
    s14_position0_mapped_assets::VerifiedMappedAssetStore,
};
use anyhow::{bail, Context, Result};
use polaris_s14_runner::Position0Asset;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{mpsc, Mutex, OnceLock},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const S14_DYNAMIC_PAGE_CACHE_READINESS_FORMAT: &str =
    "polaris-s14-dynamic-page-cache-readiness-v1";
pub const S14_DYNAMIC_PAGE_FETCH_MANIFEST_FORMAT: &str =
    "polaris-s14-dynamic-page-fetch-manifest-v1";
pub const S14_DYNAMIC_PAGE_FETCH_PYTHON_ENV: &str = "S14_DYNAMIC_PAGE_FETCH_PYTHON";
pub const S14_DYNAMIC_PAGE_FETCH_DRIVER_ENV: &str = "S14_DYNAMIC_PAGE_FETCH_DRIVER";
pub const S14_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS_ENV: &str =
    "S14_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS";
pub const S14_DYNAMIC_PAGE_FETCH_LOG_DIR_ENV: &str = "S14_DYNAMIC_PAGE_FETCH_LOG_DIR";

const DEFAULT_DYNAMIC_PAGE_FETCH_TIMEOUT: Duration = Duration::from_secs(7_200);
const EXACT_ROUTE_PREFETCH_TIMEOUT: Duration = Duration::from_secs(240);
const MAX_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS: u64 = 86_400;
#[cfg(test)]
const DYNAMIC_PAGE_FETCH_POLL_INTERVAL: Duration = Duration::from_millis(200);
const DYNAMIC_PAGE_FETCH_LOG_TAIL_BYTES: u64 = 16 * 1024;
const DYNAMIC_PAGE_FETCH_PROTOCOL_PREFIX_BYTES: usize = 64;
const DYNAMIC_PAGE_FETCH_PROGRESS_SNAPSHOT_CHARS: usize = 2 * 1024;
const DYNAMIC_PAGE_FETCH_PROGRESS_FORMAT: &str = "polaris-s14-dynamic-page-fetch-progress-v1";
const DYNAMIC_PAGE_FETCH_HEARTBEAT_IDLE_TIMEOUT: Duration = Duration::from_secs(15);
const DYNAMIC_PAGE_FETCH_LOCAL_STAGE_TIMEOUT: Duration = Duration::from_secs(90);
const DYNAMIC_PAGE_FETCH_FAST_STAGE_TIMEOUT: Duration = Duration::from_secs(30);
const DYNAMIC_PAGE_FETCH_WORKER_CLEANUP_GUARD: Duration = Duration::from_secs(20);

/// Network access is denied by default. Callers must construct
/// `ExplicitFetch` from an explicit user/runtime authorization boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DynamicPageFetchMode {
    #[default]
    LocalOnly,
    ExplicitFetch,
}

impl DynamicPageFetchMode {
    const fn is_authorized(self) -> bool {
        matches!(self, Self::ExplicitFetch)
    }
}

/// Existing Range transport process used only after `ExplicitFetch`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicPageRangeTransport {
    python: PathBuf,
    driver: PathBuf,
}

impl DynamicPageRangeTransport {
    pub fn new(python: impl Into<PathBuf>, driver: impl Into<PathBuf>) -> Self {
        Self {
            python: python.into(),
            driver: driver.into(),
        }
    }

    pub fn python(&self) -> &Path {
        &self.python
    }

    pub fn driver(&self) -> &Path {
        &self.driver
    }
}

impl Default for DynamicPageRangeTransport {
    fn default() -> Self {
        let python = std::env::var_os(S14_DYNAMIC_PAGE_FETCH_PYTHON_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("python"));
        let driver = std::env::var_os(S14_DYNAMIC_PAGE_FETCH_DRIVER_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(default_fetch_driver);
        Self { python, driver }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicPageCacheStatus {
    Ready,
    MissingPayloadAndProof,
    MissingPayload,
    MissingProof,
    PartialPresent,
    Invalid,
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicPageCacheRangeReadiness {
    pub physical_index: usize,
    pub route_slot: usize,
    pub expert_id: u16,
    pub projection: String,
    pub part: String,
    pub tensor: String,
    pub bytes: u64,
    pub range_key: String,
    pub cache_key: String,
    pub payload_path: PathBuf,
    pub proof_path: PathBuf,
    pub partial_path: PathBuf,
    pub payload_exists: bool,
    pub proof_exists: bool,
    pub partial_exists: bool,
    pub status: DynamicPageCacheStatus,
    pub issue: Option<String>,
}

impl DynamicPageCacheRangeReadiness {
    pub fn is_ready(&self) -> bool {
        self.status == DynamicPageCacheStatus::Ready
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicPageCacheReadinessReport {
    pub format: &'static str,
    pub layer: u8,
    pub position: u64,
    pub expert_ids: [u16; 6],
    pub cache_root: PathBuf,
    pub range_count: usize,
    pub total_bytes: u64,
    pub ready_count: usize,
    pub unready_count: usize,
    pub missing_payload_bytes: u64,
    pub unready_bytes: u64,
    pub ranges: Vec<DynamicPageCacheRangeReadiness>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicPageFetchManifest {
    pub format: &'static str,
    pub layer: u8,
    pub position: u64,
    pub entries: Vec<DynamicPageFetchEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct DynamicPageFetchEntry {
    pub tensor: String,
    pub kind: String,
    pub layer: u8,
    pub expert_id: u16,
    pub file: String,
    pub file_bytes: u64,
    pub header_tensor_table_sha256: String,
    pub start: u64,
    pub end: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub range_key: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicPageFetchRequired {
    pub route_json: String,
    pub missing_cache_keys: Vec<String>,
    pub missing_payload_paths: Vec<PathBuf>,
    pub missing_proof_paths: Vec<PathBuf>,
    pub unready_bytes: u64,
}

/// Exact-route cache warming receipt.  This receipt is never sufficient for
/// model execution: the authoritative union materializer must still resolve
/// every proof and rehash/mmap every payload before publishing a GPU upload.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DynamicPageFetchOnlyReceipt {
    pub layer: u8,
    pub position: u64,
    pub physical_ranges: usize,
    pub requested_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub downloaded_bytes: u64,
    pub request_wall_ms: f64,
    pub transport_invocations: u32,
}

/// Packet-scoped Range fetch input.  It deliberately carries the already
/// derived [`S14PlannedRangeAsset`] rather than rebuilding a whole routed-page
/// plan: constellation execution can therefore fetch only the weight/scale
/// pairs needed by the next immutable packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicPagePlannedFetchAsset {
    pub expert_id: u16,
    pub planned: S14PlannedRangeAsset,
}

impl DynamicPagePlannedFetchAsset {
    pub fn new(expert_id: u16, planned: S14PlannedRangeAsset) -> Self {
        Self { expert_id, planned }
    }
}

/// Cheap local admission check used by storage-aware routing decisions.  It
/// never upgrades an asset to verified: the authoritative mmap/SHA gate still
/// runs later.  A partial file or a payload with the wrong byte count is cold.
pub fn planned_range_asset_committed_locally(planned: &S14PlannedRangeAsset) -> bool {
    let partial = planned
        .payload_path
        .with_file_name(format!("{}.bin.part", planned.cache_key));
    !partial.exists()
        && planned.proof_path.is_file()
        && fs::metadata(&planned.payload_path)
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == planned.bytes)
}

#[derive(Debug)]
pub enum DynamicPageMaterializeError {
    FetchRequired(DynamicPageFetchRequired),
    Failed(anyhow::Error),
}

/// End-to-end production lease: exact online route pages, Rust-rehashed mmap
/// leases and the canonical ragged arena packing all share one identity chain.
#[derive(Debug)]
pub struct MaterializedDynamicRoutedArena {
    pub pages: MaterializedDynamicRoutedPagePlan,
    pub upload: S14DynamicRoutedUploadPlan,
}

/// Resolve one planned non-expert Range (currently token embedding) through
/// the same explicit Range transport and Rust SHA gate used by routed pages.
/// Local-only remains the default; `ExplicitFetch` is the caller's network
/// authorization boundary.
pub fn materialize_planned_range_asset(
    planned: &S14PlannedRangeAsset,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
) -> Result<Position0Asset> {
    match planned.resolve_cached_position0_asset(cache_root, None) {
        Ok(asset) => return Ok(asset),
        Err(error) if !fetch_mode.is_authorized() => return Err(error),
        Err(_) => {}
    }

    let partial = cache_root.join(format!("{}.bin.part", planned.cache_key));
    if partial.exists() {
        bail!(
            "planned Range cache 存在未提交 partial: {}",
            partial.display()
        );
    }
    let manifest = DynamicPageFetchManifest {
        format: S14_DYNAMIC_PAGE_FETCH_MANIFEST_FORMAT,
        layer: 0,
        position: 0,
        entries: vec![DynamicPageFetchEntry {
            tensor: planned.tensor.clone(),
            kind: planned.kind.clone(),
            layer: 0,
            expert_id: 0,
            file: planned.identity.source_file.clone(),
            file_bytes: planned.identity.source_file_bytes,
            header_tensor_table_sha256: planned.identity.header_tensor_table_sha256.clone(),
            start: planned.identity.start,
            end: planned.identity.end,
            bytes: planned.bytes,
            dtype: planned.dtype.clone(),
            shape: planned.shape.clone(),
            range_key: planned.range_key.clone(),
        }],
    };
    let transport = DynamicPageRangeTransport::default();
    invoke_existing_range_transport(&transport, &manifest, cache_root, planned.bytes)?;
    let asset = planned.resolve_cached_position0_asset(cache_root, None)?;
    let mut store = VerifiedMappedAssetStore::new(cache_root)?;
    let mapped = store.map_verified_batch(std::slice::from_ref(&asset))?;
    if mapped.len() != 1 {
        bail!("planned Range Rust SHA mmap 回执数量漂移");
    }
    Ok(asset)
}

impl MaterializedDynamicRoutedArena {
    pub fn arena_logical_bytes(&self) -> u64 {
        self.upload.layout.arena_logical_bytes
    }

    pub fn stage_into(&self, target: &mut [u8]) -> Result<()> {
        self.upload.stage_into(target)
    }
}

impl fmt::Display for DynamicPageMaterializeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FetchRequired(required) => write!(
                formatter,
                "dynamic routed cache 缺少 {} 页、{} 字节；fetch 未授权",
                required.missing_cache_keys.len(),
                required.unready_bytes
            ),
            Self::Failed(error) => write!(formatter, "{error:#}"),
        }
    }
}

impl Error for DynamicPageMaterializeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::FetchRequired(_) => None,
            Self::Failed(error) => Some(error.as_ref()),
        }
    }
}

impl From<anyhow::Error> for DynamicPageMaterializeError {
    fn from(error: anyhow::Error) -> Self {
        Self::Failed(error)
    }
}

/// Inspect all local pages without creating directories, proofs or payloads.
pub fn inspect_dynamic_page_cache(
    plan: &DynamicRoutedPagePlan,
    cache_root: &Path,
) -> Result<DynamicPageCacheReadinessReport> {
    if cache_root.exists() && !cache_root.is_dir() {
        bail!("dynamic page cache root 不是目录: {}", cache_root.display());
    }
    let report_root = if cache_root.exists() {
        cache_root
            .canonicalize()
            .with_context(|| format!("resolve dynamic page cache {}", cache_root.display()))?
    } else {
        cache_root.to_path_buf()
    };
    let physical_ranges = plan.physical_ranges()?;
    let mut store = if cache_root.is_dir() {
        Some(VerifiedMappedAssetStore::new(cache_root)?)
    } else {
        None
    };
    let mut rows = Vec::with_capacity(DYNAMIC_ROUTED_RANGE_COUNT);
    let mut total_bytes = 0u64;
    let mut ready_count = 0usize;
    let mut missing_payload_bytes = 0u64;
    let mut unready_bytes = 0u64;

    for physical in physical_ranges {
        let planned = physical.planned_asset(&report_root)?;
        let partial_path = report_root.join(format!("{}.bin.part", planned.cache_key));
        let payload_exists = planned.payload_path.is_file();
        let proof_exists = planned.proof_path.is_file();
        let partial_exists = partial_path.exists();
        let (status, issue) = if partial_exists {
            (
                DynamicPageCacheStatus::PartialPresent,
                Some("存在未提交 .bin.part".to_owned()),
            )
        } else if !payload_exists && !proof_exists {
            (DynamicPageCacheStatus::MissingPayloadAndProof, None)
        } else if !payload_exists {
            (DynamicPageCacheStatus::MissingPayload, None)
        } else if !proof_exists {
            (DynamicPageCacheStatus::MissingProof, None)
        } else {
            match planned.resolve_cached_position0_asset(&report_root, Some(physical.expert_id)) {
                Ok(asset) => {
                    let mapping = store
                        .as_mut()
                        .context("cache root 存在但 verified mapped store 未初始化")?
                        .map_verified_batch(std::slice::from_ref(&asset));
                    match mapping {
                        Ok(_) => (DynamicPageCacheStatus::Ready, None),
                        Err(error) => (DynamicPageCacheStatus::Invalid, Some(format!("{error:#}"))),
                    }
                }
                Err(error) => (DynamicPageCacheStatus::Invalid, Some(format!("{error:#}"))),
            }
        };
        total_bytes = total_bytes
            .checked_add(physical.range.bytes)
            .context("dynamic page total bytes overflow")?;
        if status == DynamicPageCacheStatus::Ready {
            ready_count += 1;
        } else {
            unready_bytes = unready_bytes
                .checked_add(physical.range.bytes)
                .context("dynamic page unready bytes overflow")?;
        }
        if !payload_exists {
            missing_payload_bytes = missing_payload_bytes
                .checked_add(physical.range.bytes)
                .context("dynamic page missing payload bytes overflow")?;
        }
        rows.push(DynamicPageCacheRangeReadiness {
            physical_index: physical.physical_index,
            route_slot: physical.route_slot,
            expert_id: physical.expert_id,
            projection: physical.projection.tensor_stem().to_owned(),
            part: match physical.part {
                RoutedRangePart::Weight => "weight",
                RoutedRangePart::Scale => "scale",
            }
            .to_owned(),
            tensor: physical.range.tensor.clone(),
            bytes: physical.range.bytes,
            range_key: physical.range.range_key.clone(),
            cache_key: planned.cache_key,
            payload_path: planned.payload_path,
            proof_path: planned.proof_path,
            partial_path,
            payload_exists,
            proof_exists,
            partial_exists,
            status,
            issue,
        });
    }
    if rows.len() != DYNAMIC_ROUTED_RANGE_COUNT {
        bail!("dynamic page readiness Range count 漂移");
    }
    Ok(DynamicPageCacheReadinessReport {
        format: S14_DYNAMIC_PAGE_CACHE_READINESS_FORMAT,
        layer: plan.layer,
        position: plan.position,
        expert_ids: plan.expert_ids,
        cache_root: report_root,
        range_count: rows.len(),
        total_bytes,
        ready_count,
        unready_count: rows.len() - ready_count,
        missing_payload_bytes,
        unready_bytes,
        ranges: rows,
    })
}

/// Build the exact catalog entries accepted by Python RangeCache.fetch.
pub fn build_unready_fetch_manifest(
    plan: &DynamicRoutedPagePlan,
    report: &DynamicPageCacheReadinessReport,
) -> Result<DynamicPageFetchManifest> {
    if report.format != S14_DYNAMIC_PAGE_CACHE_READINESS_FORMAT
        || report.layer != plan.layer
        || report.position != plan.position
        || report.expert_ids != plan.expert_ids
        || report.ranges.len() != DYNAMIC_ROUTED_RANGE_COUNT
    {
        bail!("dynamic page readiness report 与 plan identity 不一致");
    }
    let physical_ranges = plan.physical_ranges()?;
    let mut entries = Vec::with_capacity(report.unready_count);
    for (physical, readiness) in physical_ranges.into_iter().zip(&report.ranges) {
        let planned = physical.planned_asset(&report.cache_root)?;
        let expected_part = match physical.part {
            RoutedRangePart::Weight => "weight",
            RoutedRangePart::Scale => "scale",
        };
        if physical.physical_index != readiness.physical_index
            || physical.route_slot != readiness.route_slot
            || physical.expert_id != readiness.expert_id
            || physical.projection.tensor_stem() != readiness.projection
            || expected_part != readiness.part
            || physical.range.tensor != readiness.tensor
            || physical.range.bytes != readiness.bytes
            || physical.range.range_key != readiness.range_key
            || planned.cache_key != readiness.cache_key
            || planned.payload_path != readiness.payload_path
            || planned.proof_path != readiness.proof_path
            || report
                .cache_root
                .join(format!("{}.bin.part", planned.cache_key))
                != readiness.partial_path
        {
            bail!("dynamic page readiness physical identity/order 漂移");
        }
        if readiness.is_ready() {
            continue;
        }
        let range = physical.range;
        entries.push(DynamicPageFetchEntry {
            tensor: range.tensor.clone(),
            kind: range.kind.clone(),
            layer: range.layer,
            expert_id: range.expert_id,
            file: range.file.clone(),
            file_bytes: range.file_bytes,
            header_tensor_table_sha256: range.header_tensor_table_sha256.clone(),
            start: range.start,
            end: range.end,
            bytes: range.bytes,
            dtype: range.dtype.clone(),
            shape: range.shape.clone(),
            range_key: range.range_key.clone(),
        });
    }
    if entries.len() != report.unready_count {
        bail!("dynamic page fetch manifest unready count 漂移");
    }
    Ok(DynamicPageFetchManifest {
        format: S14_DYNAMIC_PAGE_FETCH_MANIFEST_FORMAT,
        layer: plan.layer,
        position: plan.position,
        entries,
    })
}

/// Production cache gate for one online route. Network access is impossible
/// unless fetch_authorized is explicitly true.
pub fn materialize_dynamic_page_plan(
    plan: &DynamicRoutedPagePlan,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
) -> std::result::Result<MaterializedDynamicRoutedPagePlan, DynamicPageMaterializeError> {
    let transport = DynamicPageRangeTransport::default();
    materialize_dynamic_page_plan_with_transport(plan, cache_root, fetch_mode, &transport)
}

/// 将同一 causal-block 层的 K=4/8 lane route 合并成一次 Range transport。
///
/// 每个 lane 仍先独立生成并验证完整 readiness/manifest identity；这里只按
/// `range_key` 去重完全相同的物理 Range，然后一次启动现有 Python transport。
/// transport 返回后，每个原始 plan 都重新执行 Rust proof、payload SHA 与 mmap 门，
/// 因而批处理只减少进程/连接往返，不放宽任何资产验证。
pub fn materialize_dynamic_page_plans_batched(
    plans: &[DynamicRoutedPagePlan],
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
) -> std::result::Result<Vec<MaterializedDynamicRoutedPagePlan>, DynamicPageMaterializeError> {
    if plans.is_empty() || plans.len() > 8 {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "batched dynamic page plans 数量必须在1..=8，actual={}",
            plans.len()
        )));
    }
    let first_layer = plans[0].layer;
    if plans.iter().any(|plan| plan.layer != first_layer) {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "batched dynamic page plans 必须属于同一层"
        )));
    }

    let mut merged = BTreeMap::<String, DynamicPageFetchEntry>::new();
    let mut first_required = None;
    for plan in plans {
        let report = inspect_dynamic_page_cache(plan, cache_root)?;
        if report.unready_count == 0 {
            continue;
        }
        if first_required.is_none() {
            first_required = Some(fetch_required_error(plan, &report)?);
        }
        let manifest = build_unready_fetch_manifest(plan, &report)?;
        for entry in manifest.entries {
            match merged.entry(entry.range_key.clone()) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(entry);
                }
                std::collections::btree_map::Entry::Occupied(slot) => {
                    if slot.get() != &entry {
                        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
                            "batched dynamic page 重复 range_key identity 漂移: {}",
                            slot.key()
                        )));
                    }
                }
            }
        }
    }

    if !merged.is_empty() {
        if !fetch_mode.is_authorized() {
            return Err(DynamicPageMaterializeError::FetchRequired(
                first_required.expect("unready manifest 必有 structured error"),
            ));
        }
        let entries = merged.into_values().collect::<Vec<_>>();
        let budget = entries.iter().try_fold(0u64, |sum, entry| {
            sum.checked_add(entry.bytes)
                .context("batched dynamic page download budget overflow")
        })?;
        let manifest = DynamicPageFetchManifest {
            format: S14_DYNAMIC_PAGE_FETCH_MANIFEST_FORMAT,
            layer: first_layer,
            position: plans.iter().map(|plan| plan.position).min().unwrap_or(0),
            entries,
        };
        let transport = DynamicPageRangeTransport::default();
        invoke_existing_range_transport(&transport, &manifest, cache_root, budget)
            .context("batched dynamic page Range transport 失败")?;
    }

    plans
        .iter()
        .map(|plan| {
            plan.materialize_cached(cache_root)
                .context("batched dynamic page fetch 后 Rust proof/SHA 复核失败")
        })
        .collect::<Result<Vec<_>>>()
        .map_err(DynamicPageMaterializeError::Failed)
}

/// Warm one authoritative K-lane route manifest without creating a model
/// lease.  All physical identities are merged by `range_key` and sent to the
/// resident transport once.  Cache hits remain local; misses retain the
/// transport's HTTPS 206/Content-Range/length/SHA contract.
///
/// The caller must subsequently run the normal union materializer.  This
/// helper deliberately does not call `materialize_cached`, avoiding a second
/// eager Rust SHA/mmap pass before the authoritative one.
pub fn fetch_dynamic_page_plans_batched_only(
    plans: &[DynamicRoutedPagePlan],
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
) -> std::result::Result<DynamicPageFetchOnlyReceipt, DynamicPageMaterializeError> {
    if plans.is_empty() || plans.len() > 8 {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "fetch-only dynamic page plans 数量必须在1..=8，actual={}",
            plans.len()
        )));
    }
    let layer = plans[0].layer;
    if plans.iter().any(|plan| plan.layer != layer) {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "fetch-only dynamic page plans 必须属于同一层"
        )));
    }

    let position = plans.iter().map(|plan| plan.position).min().unwrap_or(0);
    let mut merged = BTreeMap::<String, DynamicPageFetchEntry>::new();
    let mut committed_paths = BTreeMap::<String, (PathBuf, PathBuf, PathBuf)>::new();
    for plan in plans {
        for physical in plan.physical_ranges()? {
            let range = physical.range;
            let planned = physical.planned_asset(cache_root)?;
            let canonical_paths = (
                planned.payload_path,
                planned.proof_path,
                cache_root.join(format!("{}.bin.part", planned.cache_key)),
            );
            let entry = DynamicPageFetchEntry {
                tensor: range.tensor.clone(),
                kind: range.kind.clone(),
                layer: range.layer,
                expert_id: range.expert_id,
                file: range.file.clone(),
                file_bytes: range.file_bytes,
                header_tensor_table_sha256: range.header_tensor_table_sha256.clone(),
                start: range.start,
                end: range.end,
                bytes: range.bytes,
                dtype: range.dtype.clone(),
                shape: range.shape.clone(),
                range_key: range.range_key.clone(),
            };
            match merged.entry(entry.range_key.clone()) {
                std::collections::btree_map::Entry::Vacant(slot) => {
                    committed_paths.insert(entry.range_key.clone(), canonical_paths);
                    slot.insert(entry);
                }
                std::collections::btree_map::Entry::Occupied(slot) => {
                    if slot.get() != &entry {
                        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
                            "fetch-only dynamic page 重复 range_key identity 漂移: {}",
                            slot.key()
                        )));
                    }
                    if committed_paths.get(slot.key()) != Some(&canonical_paths) {
                        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
                            "fetch-only dynamic page 重复 range_key canonical cache identity 漂移: {}",
                            slot.key()
                        )));
                    }
                }
            }
        }
    }
    let entries = merged.into_values().collect::<Vec<_>>();
    let requested_bytes = entries.iter().try_fold(0u64, |sum, entry| {
        sum.checked_add(entry.bytes)
            .context("fetch-only dynamic page budget overflow")
    })?;
    let mut receipt = DynamicPageFetchOnlyReceipt {
        layer,
        position,
        physical_ranges: entries.len(),
        requested_bytes,
        ..DynamicPageFetchOnlyReceipt::default()
    };
    if !fetch_mode.is_authorized() {
        return Ok(receipt);
    }
    if entries.is_empty() || requested_bytes == 0 {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "fetch-only dynamic page manifest/budget 为空"
        )));
    }
    // ExplicitFetch 只授权“缺页时可访问网络”，不意味着每个热层都必须跨进程调用
    // Python transport。真正的 proof/identity/payload-SHA 门仍由随后每个 microtile 的
    // `verify_microtile` 执行；这里只做不会把损坏文件冒充 verified asset 的尺寸快检。
    // 这样热缓存 block 不再为 43 层各支付一次 JSONL/进程往返。
    let all_committed_locally = entries.iter().all(|entry| {
        let Some((payload, proof, partial)) = committed_paths.get(&entry.range_key) else {
            return false;
        };
        !partial.exists()
            && proof.is_file()
            && fs::metadata(payload)
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() == entry.bytes)
    });
    if all_committed_locally {
        receipt.cache_hits = u64::try_from(entries.len()).map_err(|_| {
            DynamicPageMaterializeError::Failed(anyhow::anyhow!("本地热缓存 range 数量超过 u64"))
        })?;
        return Ok(receipt);
    }
    let manifest = DynamicPageFetchManifest {
        format: S14_DYNAMIC_PAGE_FETCH_MANIFEST_FORMAT,
        layer,
        position,
        entries,
    };
    let transport = DynamicPageRangeTransport::default();
    if !transport.driver.is_file() {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "dynamic Range transport driver 不存在: {}",
            transport.driver.display()
        )));
    }
    let timeout = dynamic_page_fetch_timeout()
        .map(|configured| configured.min(EXACT_ROUTE_PREFETCH_TIMEOUT))?;
    let transport_receipt = invoke_persistent_range_transport(
        &transport,
        &manifest,
        cache_root,
        requested_bytes,
        timeout,
    )?;
    receipt.cache_hits = transport_receipt.cache_hits;
    receipt.cache_misses = transport_receipt.cache_misses;
    receipt.downloaded_bytes = transport_receipt.downloaded_bytes;
    receipt.request_wall_ms = transport_receipt.request_wall_ms;
    receipt.transport_invocations = 1;
    Ok(receipt)
}

/// Fetch only the canonical Range assets required by one constellation
/// packet.  Hot assets return without entering the Python worker; cold assets
/// preserve the same persistent HTTPS 206/Content-Range/length/SHA transport
/// contract as the full-layer route warmup.  This function does not mmap or
/// publish a model lease: `verify_microtile` remains the authoritative gate.
pub fn fetch_planned_range_assets_batched_only(
    layer: u8,
    position: u64,
    assets: &[DynamicPagePlannedFetchAsset],
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
) -> std::result::Result<DynamicPageFetchOnlyReceipt, DynamicPageMaterializeError> {
    const MAX_PACKET_RANGE_ASSETS: usize = 48;

    if assets.is_empty() || assets.len() > MAX_PACKET_RANGE_ASSETS {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "packet fetch planned assets 数量必须在1..={MAX_PACKET_RANGE_ASSETS}，actual={}",
            assets.len()
        )));
    }

    // Canonical cache identity, not the display range key, owns deduplication.
    // This keeps different repo/revision/header identities from aliasing merely
    // because they happen to use the same source-file byte interval.
    let mut merged = BTreeMap::<String, DynamicPageFetchEntry>::new();
    let mut committed_paths = BTreeMap::<String, (PathBuf, PathBuf, PathBuf)>::new();
    for asset in assets {
        let planned = &asset.planned;
        let identity = &planned.identity;
        let expected_bytes = identity
            .end
            .checked_sub(identity.start)
            .and_then(|delta| delta.checked_add(1))
            .context("packet fetch planned Range 长度 overflow")?;
        if planned.bytes == 0
            || planned.bytes != expected_bytes
            || planned.range_key
                != format!(
                    "{}:{}-{}",
                    identity.source_file, identity.start, identity.end
                )
        {
            return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
                "packet fetch planned Range identity/bytes 漂移: tensor={}, cache_key={}",
                planned.tensor,
                planned.cache_key
            )));
        }
        let entry = DynamicPageFetchEntry {
            tensor: planned.tensor.clone(),
            kind: planned.kind.clone(),
            layer,
            expert_id: asset.expert_id,
            file: identity.source_file.clone(),
            file_bytes: identity.source_file_bytes,
            header_tensor_table_sha256: identity.header_tensor_table_sha256.clone(),
            start: identity.start,
            end: identity.end,
            bytes: planned.bytes,
            dtype: planned.dtype.clone(),
            shape: planned.shape.clone(),
            range_key: planned.range_key.clone(),
        };
        let canonical_paths = (
            planned.payload_path.clone(),
            planned.proof_path.clone(),
            cache_root.join(format!("{}.bin.part", planned.cache_key)),
        );
        match merged.entry(planned.cache_key.clone()) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                committed_paths.insert(planned.cache_key.clone(), canonical_paths);
                slot.insert(entry);
            }
            std::collections::btree_map::Entry::Occupied(slot) => {
                if slot.get() != &entry || committed_paths.get(slot.key()) != Some(&canonical_paths)
                {
                    return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
                        "packet fetch canonical cache identity 漂移: {}",
                        slot.key()
                    )));
                }
            }
        }
    }

    let requested_bytes = merged.values().try_fold(0u64, |sum, entry| {
        sum.checked_add(entry.bytes)
            .context("packet fetch planned Range budget overflow")
    })?;
    let mut receipt = DynamicPageFetchOnlyReceipt {
        layer,
        position,
        physical_ranges: merged.len(),
        requested_bytes,
        ..DynamicPageFetchOnlyReceipt::default()
    };
    if !fetch_mode.is_authorized() {
        return Ok(receipt);
    }
    if requested_bytes == 0 {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "packet fetch planned Range manifest/budget 为空"
        )));
    }

    let all_committed_locally = merged.iter().all(|(cache_key, entry)| {
        let Some((payload, proof, partial)) = committed_paths.get(cache_key) else {
            return false;
        };
        !partial.exists()
            && proof.is_file()
            && fs::metadata(payload)
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() == entry.bytes)
    });
    if all_committed_locally {
        receipt.cache_hits = u64::try_from(merged.len()).map_err(|_| {
            DynamicPageMaterializeError::Failed(anyhow::anyhow!(
                "packet fetch 本地热缓存 range 数量超过 u64"
            ))
        })?;
        return Ok(receipt);
    }

    let manifest = DynamicPageFetchManifest {
        format: S14_DYNAMIC_PAGE_FETCH_MANIFEST_FORMAT,
        layer,
        position,
        entries: merged.into_values().collect(),
    };
    let transport = DynamicPageRangeTransport::default();
    if !transport.driver.is_file() {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "dynamic Range transport driver 不存在: {}",
            transport.driver.display()
        )));
    }
    let timeout = dynamic_page_fetch_timeout()
        .map(|configured| configured.min(EXACT_ROUTE_PREFETCH_TIMEOUT))?;
    let transport_receipt = invoke_persistent_range_transport(
        &transport,
        &manifest,
        cache_root,
        requested_bytes,
        timeout,
    )?;
    receipt.cache_hits = transport_receipt.cache_hits;
    receipt.cache_misses = transport_receipt.cache_misses;
    receipt.downloaded_bytes = transport_receipt.downloaded_bytes;
    receipt.request_wall_ms = transport_receipt.request_wall_ms;
    receipt.transport_invocations = 1;
    Ok(receipt)
}

pub fn materialize_dynamic_page_plan_with_transport(
    plan: &DynamicRoutedPagePlan,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
    transport: &DynamicPageRangeTransport,
) -> std::result::Result<MaterializedDynamicRoutedPagePlan, DynamicPageMaterializeError> {
    materialize_dynamic_page_plan_with_fetcher(
        plan,
        cache_root,
        fetch_mode,
        |manifest, root, budget| invoke_existing_range_transport(transport, manifest, root, budget),
    )
}

/// Production entry point from an online top-6 plan to the canonical routed
/// arena. In order: local materialize, optional explicit Range fetch, Rust
/// proof/identity/payload-SHA verification, then fixed ragged ABI packing.
pub fn materialize_dynamic_routed_arena(
    plan: &DynamicRoutedPagePlan,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
) -> std::result::Result<MaterializedDynamicRoutedArena, DynamicPageMaterializeError> {
    let transport = DynamicPageRangeTransport::default();
    materialize_dynamic_routed_arena_with_transport(plan, cache_root, fetch_mode, &transport)
}

pub fn materialize_dynamic_routed_arena_with_transport(
    plan: &DynamicRoutedPagePlan,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
    transport: &DynamicPageRangeTransport,
) -> std::result::Result<MaterializedDynamicRoutedArena, DynamicPageMaterializeError> {
    materialize_dynamic_routed_arena_with_fetcher(
        plan,
        cache_root,
        fetch_mode,
        |manifest, root, budget| invoke_existing_range_transport(transport, manifest, root, budget),
    )
}

fn materialize_dynamic_page_plan_with_fetcher<F>(
    plan: &DynamicRoutedPagePlan,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
    fetcher: F,
) -> std::result::Result<MaterializedDynamicRoutedPagePlan, DynamicPageMaterializeError>
where
    F: FnOnce(&DynamicPageFetchManifest, &Path, u64) -> Result<()>,
{
    let initial_error = match plan.materialize_cached(cache_root) {
        Ok(materialized) => return Ok(materialized),
        Err(error) => error,
    };
    let report = inspect_dynamic_page_cache(plan, cache_root)?;
    if report.unready_count == 0 {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "dynamic page 首次 materialize 失败但 readiness 未发现缺页: {initial_error:#}"
        )));
    }
    if !fetch_mode.is_authorized() {
        return Err(DynamicPageMaterializeError::FetchRequired(
            fetch_required_error(plan, &report)?,
        ));
    }

    let manifest = build_unready_fetch_manifest(plan, &report)?;
    if manifest.entries.is_empty() || report.unready_bytes == 0 {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "dynamic page fetch 已授权但 manifest/budget 为空"
        )));
    }
    fetcher(&manifest, cache_root, report.unready_bytes)
        .context("dynamic page Range transport 失败")?;
    plan.materialize_cached(cache_root)
        .context("dynamic page fetch 后 Rust proof/SHA 复核失败")
        .map_err(DynamicPageMaterializeError::Failed)
}

fn materialize_dynamic_routed_arena_with_fetcher<F>(
    plan: &DynamicRoutedPagePlan,
    cache_root: &Path,
    fetch_mode: DynamicPageFetchMode,
    fetcher: F,
) -> std::result::Result<MaterializedDynamicRoutedArena, DynamicPageMaterializeError>
where
    F: FnOnce(&DynamicPageFetchManifest, &Path, u64) -> Result<()>,
{
    let pages = materialize_dynamic_page_plan_with_fetcher(plan, cache_root, fetch_mode, fetcher)?;
    let upload = S14DynamicRoutedUploadPlan::build(plan, &pages)
        .context("dynamic page canonical routed arena packing 失败")?;
    if upload.layout.layer != plan.layer
        || upload.layout.position != plan.position
        || upload.layout.arena_logical_bytes != S14_DYNAMIC_ROUTED_ARENA_BYTES
        || upload.layout.placements.len() != DYNAMIC_ROUTED_RANGE_COUNT
    {
        return Err(DynamicPageMaterializeError::Failed(anyhow::anyhow!(
            "dynamic page canonical routed arena identity/bytes/count 漂移"
        )));
    }
    Ok(MaterializedDynamicRoutedArena { pages, upload })
}

fn fetch_required_error(
    plan: &DynamicRoutedPagePlan,
    report: &DynamicPageCacheReadinessReport,
) -> Result<DynamicPageFetchRequired> {
    let route = OnlineTop6 {
        layer: plan.layer,
        position: plan.position,
        expert_ids: plan.expert_ids,
        route_weights: plan.route_weights,
    };
    let unready = report
        .ranges
        .iter()
        .filter(|range| !range.is_ready())
        .collect::<Vec<_>>();
    if unready.len() != report.unready_count {
        bail!("dynamic page structured fetch error count 漂移");
    }
    Ok(DynamicPageFetchRequired {
        route_json: serde_json::to_string(&route).context("encode OnlineTop6 route JSON")?,
        missing_cache_keys: unready
            .iter()
            .map(|range| range.cache_key.clone())
            .collect(),
        missing_payload_paths: unready
            .iter()
            .map(|range| range.payload_path.clone())
            .collect(),
        missing_proof_paths: unready
            .iter()
            .map(|range| range.proof_path.clone())
            .collect(),
        unready_bytes: report.unready_bytes,
    })
}

#[cfg(test)]
struct DynamicFetchArtifacts {
    manifest: PathBuf,
    stdout: PathBuf,
    stderr: PathBuf,
}

#[cfg(test)]
impl DynamicFetchArtifacts {
    fn write(
        manifest: &DynamicPageFetchManifest,
        cache_root: &Path,
    ) -> Result<(Self, fs::File, fs::File)> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_nanos();
        let log_dir = std::env::var_os(S14_DYNAMIC_PAGE_FETCH_LOG_DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| cache_root.join("_dynamic_fetch_logs"));
        fs::create_dir_all(&log_dir)
            .with_context(|| format!("create dynamic fetch log dir {}", log_dir.display()))?;
        let stem = format!(
            "fetch-{}-{stamp}-l{}-p{}",
            std::process::id(),
            manifest.layer,
            manifest.position
        );
        let artifacts = Self {
            manifest: log_dir.join(format!("{stem}.manifest.json")),
            stdout: log_dir.join(format!("{stem}.stdout.log")),
            stderr: log_dir.join(format!("{stem}.stderr.log")),
        };
        let mut manifest_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&artifacts.manifest)
            .with_context(|| {
                format!(
                    "create dynamic fetch manifest {}",
                    artifacts.manifest.display()
                )
            })?;
        serde_json::to_writer(&mut manifest_file, manifest)
            .context("encode dynamic fetch manifest")?;
        manifest_file
            .flush()
            .context("flush dynamic fetch manifest")?;

        let stdout = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&artifacts.stdout)
            .with_context(|| {
                format!("create dynamic fetch stdout {}", artifacts.stdout.display())
            })?;
        let stderr = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&artifacts.stderr)
            .with_context(|| {
                format!("create dynamic fetch stderr {}", artifacts.stderr.display())
            })?;
        Ok((artifacts, stdout, stderr))
    }

    fn cleanup_success(&self) {
        for path in [&self.manifest, &self.stdout, &self.stderr] {
            if let Err(error) = fs::remove_file(path) {
                eprintln!(
                    "warning: dynamic Range transport 成功后清理日志失败 path={} error={error}",
                    path.display()
                );
            }
        }
    }
}

fn default_fetch_driver() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../fast16/research/polaris_meridian_v1/s14_range_pack/fetch_dynamic_range_pages.py",
    )
}

struct PersistentDynamicFetchWorker {
    transport: DynamicPageRangeTransport,
    child: Child,
    stdin: ChildStdin,
    stdout_receiver: mpsc::Receiver<(std::io::Result<usize>, Vec<u8>)>,
    stdout_thread: Option<thread::JoinHandle<()>>,
    stderr_log: PathBuf,
    next_request_id: u64,
}

#[derive(Debug)]
struct DynamicPageTransportReceipt {
    cache_hits: u64,
    cache_misses: u64,
    downloaded_bytes: u64,
    request_wall_ms: f64,
}

impl PersistentDynamicFetchWorker {
    fn spawn(transport: &DynamicPageRangeTransport, cache_root: &Path) -> Result<Self> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before Unix epoch")?
            .as_nanos();
        let log_dir = std::env::var_os(S14_DYNAMIC_PAGE_FETCH_LOG_DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| cache_root.join("_dynamic_fetch_logs"));
        fs::create_dir_all(&log_dir)
            .with_context(|| format!("create dynamic fetch log dir {}", log_dir.display()))?;
        let stderr_log = log_dir.join(format!(
            "fetch-worker-{}-{stamp}.stderr.log",
            std::process::id()
        ));
        let stderr = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stderr_log)
            .with_context(|| format!("create persistent fetch stderr {}", stderr_log.display()))?;
        let mut child = Command::new(&transport.python)
            .arg("-u")
            .arg(&transport.driver)
            .arg("--serve")
            // The JSONL pipe is a UTF-8 protocol, not a Windows console.  Pin
            // Python before it opens stdio so a localized error cannot be
            // encoded with the active ANSI code page and hide the real error.
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8:strict")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr))
            .spawn()
            .with_context(|| {
                format!(
                    "启动 persistent dynamic Range worker python={} driver={} stderr={}",
                    transport.python.display(),
                    transport.driver.display(),
                    stderr_log.display()
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .context("persistent fetch worker 缺少stdin")?;
        let mut stdout = child
            .stdout
            .take()
            .map(BufReader::new)
            .context("persistent fetch worker 缺少stdout")?;
        let (stdout_sender, stdout_receiver) = mpsc::channel();
        let stdout_thread = thread::spawn(move || loop {
            let mut line = Vec::new();
            let read = stdout.read_until(b'\n', &mut line);
            let terminal = match &read {
                Ok(bytes) => *bytes == 0,
                Err(_) => true,
            };
            if stdout_sender.send((read, line)).is_err() || terminal {
                return;
            }
        });
        eprintln!(
            "persistent dynamic Range worker started pid={} stderr={}",
            child.id(),
            stderr_log.display()
        );
        Ok(Self {
            transport: transport.clone(),
            child,
            stdin,
            stdout_receiver,
            stdout_thread: Some(stdout_thread),
            stderr_log,
            next_request_id: 0,
        })
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(stdout_thread) = self.stdout_thread.take() {
            let _ = stdout_thread.join();
        }
    }

    fn request(
        &mut self,
        manifest: &DynamicPageFetchManifest,
        cache_root: &Path,
        download_budget_bytes: u64,
        timeout: Duration,
    ) -> Result<DynamicPageTransportReceipt> {
        match self.child.try_wait() {
            Ok(None) => {}
            Ok(Some(status)) => bail!(
                "persistent dynamic Range worker 已退出: status={} stderr_log={} stderr_tail={}",
                status,
                self.stderr_log.display(),
                read_log_tail(&self.stderr_log)
            ),
            Err(error) => bail!(
                "poll persistent dynamic Range worker 失败: error={} stderr_log={} stderr_tail={}",
                error,
                self.stderr_log.display(),
                read_log_tail(&self.stderr_log)
            ),
        }
        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .context("persistent fetch request_id overflow")?;
        let cache_root_utf8 = cache_root
            .to_str()
            .context("persistent fetch cache root 必须为UTF-8")?;
        let worker_deadline = timeout
            .saturating_sub(DYNAMIC_PAGE_FETCH_WORKER_CLEANUP_GUARD)
            .max(Duration::from_secs(1));
        serde_json::to_writer(
            &mut self.stdin,
            &serde_json::json!({
                "op": "fetch_manifest",
                "request_id": request_id,
                // Inline the already validated/merged manifest.  The resident
                // worker no longer needs a per-layer manifest tempfile and the
                // Rust side no longer re-opens that file to count entries.
                "manifest": manifest,
                "cache_root": cache_root_utf8,
                "download_budget_bytes": download_budget_bytes,
                "progress_format": DYNAMIC_PAGE_FETCH_PROGRESS_FORMAT,
                "worker_deadline_seconds": worker_deadline.as_secs_f64(),
            }),
        )
        .context("encode persistent dynamic fetch request")?;
        self.stdin
            .write_all(b"\n")
            .context("write persistent dynamic fetch newline")?;
        self.stdin
            .flush()
            .context("flush persistent dynamic fetch request")?;

        let request_started = Instant::now();
        let mut last_heartbeat = request_started;
        let mut last_heartbeat_sequence = 0u64;
        let mut last_progress_revision = 0u64;
        let mut last_progress_snapshot = String::from("none");
        let response = loop {
            let hard_remaining = timeout.saturating_sub(request_started.elapsed());
            let heartbeat_remaining =
                DYNAMIC_PAGE_FETCH_HEARTBEAT_IDLE_TIMEOUT.saturating_sub(last_heartbeat.elapsed());
            let read_timeout = hard_remaining.min(heartbeat_remaining);
            if read_timeout.is_zero() {
                self.stop();
                let hard_expired = hard_remaining.is_zero();
                bail!(
                    "persistent dynamic Range worker {}并已回收: request_id={} hard_timeout_seconds={} heartbeat_idle_seconds={} last_progress={} stderr_log={} stderr_tail={}",
                    if hard_expired {
                        "请求硬截止耗尽"
                    } else {
                        "progress heartbeat 停滞"
                    },
                    request_id,
                    timeout.as_secs(),
                    DYNAMIC_PAGE_FETCH_HEARTBEAT_IDLE_TIMEOUT.as_secs(),
                    last_progress_snapshot,
                    self.stderr_log.display(),
                    read_log_tail(&self.stderr_log)
                );
            }

            let (read, line) = match self.stdout_receiver.recv_timeout(read_timeout) {
                Ok(result) => result,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    self.stop();
                    let hard_expired = request_started.elapsed() >= timeout;
                    bail!(
                        "persistent dynamic Range worker {}并已回收: request_id={} hard_timeout_seconds={} heartbeat_idle_seconds={} last_progress={} stderr_log={} stderr_tail={}",
                        if hard_expired {
                            "请求硬截止耗尽"
                        } else {
                            "progress heartbeat 停滞"
                        },
                        request_id,
                        timeout.as_secs(),
                        DYNAMIC_PAGE_FETCH_HEARTBEAT_IDLE_TIMEOUT.as_secs(),
                        last_progress_snapshot,
                        self.stderr_log.display(),
                        read_log_tail(&self.stderr_log)
                    );
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.stop();
                    bail!(
                        "persistent dynamic Range worker stdout reader 断开并已回收: request_id={} last_progress={} stderr_log={} stderr_tail={}",
                        request_id,
                        last_progress_snapshot,
                        self.stderr_log.display(),
                        read_log_tail(&self.stderr_log)
                    );
                }
            };
            let read = read.context("read persistent dynamic fetch response")?;
            if read == 0 {
                let status = self.child.wait().ok();
                bail!(
                    "persistent dynamic Range worker stdout EOF: request_id={} status={:?} last_progress={} stderr_log={} stderr_tail={}",
                    request_id,
                    status,
                    last_progress_snapshot,
                    self.stderr_log.display(),
                    read_log_tail(&self.stderr_log)
                );
            }
            let line = match std::str::from_utf8(&line) {
                Ok(line) => line,
                Err(error) => {
                    let prefix = protocol_hex_prefix(&line);
                    self.stop();
                    bail!(
                        "persistent dynamic Range worker 返回非 UTF-8 JSONL 并已回收: request_id={} error={} bytes={} prefix_hex={} last_progress={} stderr_log={} stderr_tail={}",
                        request_id,
                        error,
                        line.len(),
                        prefix,
                        last_progress_snapshot,
                        self.stderr_log.display(),
                        read_log_tail(&self.stderr_log)
                    );
                }
            };
            let frame: serde_json::Value = serde_json::from_str(line)
                .context("decode persistent dynamic fetch response JSON")?;
            if frame.get("request_id").and_then(serde_json::Value::as_u64) != Some(request_id) {
                bail!("persistent dynamic fetch response request_id 漂移");
            }
            match frame.get("event").and_then(serde_json::Value::as_str) {
                Some("progress") => {
                    let snapshot = validate_persistent_progress(
                        &frame,
                        manifest.entries.len(),
                        download_budget_bytes,
                        timeout,
                        last_heartbeat_sequence,
                        last_progress_revision,
                    )?;
                    last_heartbeat = Instant::now();
                    last_heartbeat_sequence = snapshot.heartbeat_sequence;
                    last_progress_revision = snapshot.progress_revision;
                    last_progress_snapshot = snapshot.compact;
                }
                Some(event) => bail!(
                    "persistent dynamic fetch response event 未知: event={event} last_progress={last_progress_snapshot}"
                ),
                None => break frame,
            }
        };
        if response.get("ok").and_then(serde_json::Value::as_bool) != Some(true) {
            bail!(
                "persistent dynamic fetch 请求失败: type={} error={} progress={} transport={} last_heartbeat={} stderr_log={} stderr_tail={}",
                response
                    .get("error_type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
                response
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown"),
                response
                    .get("progress")
                    .map(compact_protocol_value)
                    .unwrap_or_else(|| String::from("none")),
                response
                    .get("transport")
                    .map(compact_protocol_value)
                    .unwrap_or_else(|| String::from("none")),
                last_progress_snapshot,
                self.stderr_log.display(),
                read_log_tail(&self.stderr_log)
            );
        }
        let result = response
            .get("result")
            .context("persistent fetch response 缺少result")?;
        if result.get("format").and_then(serde_json::Value::as_str)
            != Some("polaris-s14-dynamic-page-fetch-result-v1")
            || result
                .get("requested_bytes")
                .and_then(serde_json::Value::as_u64)
                != Some(download_budget_bytes)
        {
            bail!("persistent dynamic fetch result format/budget 漂移");
        }
        let expected_ranges = u64::try_from(manifest.entries.len())
            .context("persistent fetch manifest range count overflow")?;
        let range_count = result
            .get("range_count")
            .and_then(serde_json::Value::as_u64)
            .context("persistent dynamic fetch result 缺少range_count")?;
        let cache_hits = result
            .get("cache_hits")
            .and_then(serde_json::Value::as_u64)
            .context("persistent dynamic fetch result 缺少cache_hits")?;
        let cache_misses = result
            .get("cache_misses")
            .and_then(serde_json::Value::as_u64)
            .context("persistent dynamic fetch result 缺少cache_misses")?;
        if range_count != expected_ranges
            || cache_hits.checked_add(cache_misses) != Some(expected_ranges)
        {
            bail!("persistent dynamic fetch result range/hit/miss count 漂移");
        }
        let downloaded_bytes = result
            .get("downloaded_bytes")
            .and_then(serde_json::Value::as_u64)
            .context("persistent dynamic fetch result 缺少downloaded_bytes")?;
        let request_wall_ms = result
            .get("request_wall_ms")
            .and_then(serde_json::Value::as_f64)
            .context("persistent dynamic fetch result 缺少request_wall_ms")?;
        if !request_wall_ms.is_finite()
            || request_wall_ms < 0.0
            || downloaded_bytes > download_budget_bytes
        {
            bail!("persistent dynamic fetch result timing/download bytes 非法");
        }
        Ok(DynamicPageTransportReceipt {
            cache_hits,
            cache_misses,
            downloaded_bytes,
            request_wall_ms,
        })
    }
}

#[derive(Debug)]
struct PersistentProgressSnapshot {
    heartbeat_sequence: u64,
    progress_revision: u64,
    compact: String,
}

fn validate_persistent_progress(
    frame: &serde_json::Value,
    expected_ranges: usize,
    expected_bytes: u64,
    hard_timeout: Duration,
    previous_heartbeat_sequence: u64,
    previous_progress_revision: u64,
) -> Result<PersistentProgressSnapshot> {
    let progress = frame
        .get("progress")
        .context("persistent dynamic fetch progress frame 缺少 progress")?;
    if progress.get("format").and_then(serde_json::Value::as_str)
        != Some(DYNAMIC_PAGE_FETCH_PROGRESS_FORMAT)
    {
        bail!("persistent dynamic fetch progress format 漂移");
    }
    let heartbeat_sequence = progress
        .get("heartbeat_sequence")
        .and_then(serde_json::Value::as_u64)
        .context("persistent dynamic fetch progress 缺少 heartbeat_sequence")?;
    let progress_revision = progress
        .get("progress_revision")
        .and_then(serde_json::Value::as_u64)
        .context("persistent dynamic fetch progress 缺少 progress_revision")?;
    if heartbeat_sequence <= previous_heartbeat_sequence
        || progress_revision < previous_progress_revision
    {
        bail!("persistent dynamic fetch progress sequence/revision 非单调");
    }

    let ranges_total = progress
        .get("ranges_total")
        .and_then(serde_json::Value::as_u64)
        .context("persistent dynamic fetch progress 缺少 ranges_total")?;
    let ranges_started = progress
        .get("ranges_started")
        .and_then(serde_json::Value::as_u64)
        .context("persistent dynamic fetch progress 缺少 ranges_started")?;
    let ranges_completed = progress
        .get("ranges_completed")
        .and_then(serde_json::Value::as_u64)
        .context("persistent dynamic fetch progress 缺少 ranges_completed")?;
    let ranges_failed = progress
        .get("ranges_failed")
        .and_then(serde_json::Value::as_u64)
        .context("persistent dynamic fetch progress 缺少 ranges_failed")?;
    let active_ranges = progress
        .get("active_ranges")
        .and_then(serde_json::Value::as_u64)
        .context("persistent dynamic fetch progress 缺少 active_ranges")?;
    let expected_ranges = u64::try_from(expected_ranges)
        .context("persistent dynamic fetch expected ranges overflow")?;
    let settled_ranges = ranges_completed
        .checked_add(ranges_failed)
        .context("persistent dynamic fetch settled ranges overflow")?;
    if ranges_total != expected_ranges
        || progress
            .get("requested_bytes")
            .and_then(serde_json::Value::as_u64)
            != Some(expected_bytes)
        || ranges_started > ranges_total
        || settled_ranges > ranges_started
        || settled_ranges.checked_add(active_ranges) != Some(ranges_started)
    {
        bail!("persistent dynamic fetch progress range/budget 状态漂移");
    }

    let stage = progress
        .get("stage")
        .and_then(serde_json::Value::as_str)
        .context("persistent dynamic fetch progress 缺少 stage")?;
    let stage_elapsed_ms = progress
        .get("stage_elapsed_ms")
        .and_then(serde_json::Value::as_f64)
        .context("persistent dynamic fetch progress 缺少 stage_elapsed_ms")?;
    let request_elapsed_ms = progress
        .get("request_elapsed_ms")
        .and_then(serde_json::Value::as_f64)
        .context("persistent dynamic fetch progress 缺少 request_elapsed_ms")?;
    let origin_deadline_seconds = progress
        .get("origin_deadline_seconds")
        .and_then(serde_json::Value::as_f64)
        .context("persistent dynamic fetch progress 缺少 origin_deadline_seconds")?;
    let worker_deadline_seconds = progress
        .get("worker_deadline_seconds")
        .and_then(serde_json::Value::as_f64)
        .context("persistent dynamic fetch progress 缺少 worker_deadline_seconds")?;
    if !stage_elapsed_ms.is_finite()
        || !request_elapsed_ms.is_finite()
        || !origin_deadline_seconds.is_finite()
        || !worker_deadline_seconds.is_finite()
        || stage_elapsed_ms < 0.0
        || request_elapsed_ms < stage_elapsed_ms
        || !(5.0..=240.0).contains(&origin_deadline_seconds)
        || worker_deadline_seconds < 1.0
        || worker_deadline_seconds > hard_timeout.as_secs_f64()
    {
        bail!("persistent dynamic fetch progress timing 非法");
    }
    let stage_timeout = match stage {
        "manifest_validated" | "predownload_check" | "cache_finalize" | "complete" => {
            DYNAMIC_PAGE_FETCH_FAST_STAGE_TIMEOUT
        }
        "cache_prepare" | "root_budget_lock" | "startup_trim" | "predownload_trim" => {
            DYNAMIC_PAGE_FETCH_LOCAL_STAGE_TIMEOUT
        }
        "fetch_ranges" => hard_timeout,
        _ => bail!("persistent dynamic fetch progress stage 未知: {stage}"),
    };
    if stage_elapsed_ms > stage_timeout.as_secs_f64() * 1000.0 {
        bail!(
            "persistent dynamic fetch stage 截止耗尽: stage={} stage_elapsed_ms={} stage_timeout_seconds={} snapshot={}",
            stage,
            stage_elapsed_ms,
            stage_timeout.as_secs(),
            compact_protocol_value(progress)
        );
    }
    Ok(PersistentProgressSnapshot {
        heartbeat_sequence,
        progress_revision,
        compact: compact_protocol_value(progress),
    })
}

fn compact_protocol_value(value: &serde_json::Value) -> String {
    value
        .to_string()
        .chars()
        .take(DYNAMIC_PAGE_FETCH_PROGRESS_SNAPSHOT_CHARS)
        .collect()
}

fn protocol_hex_prefix(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(DYNAMIC_PAGE_FETCH_PROTOCOL_PREFIX_BYTES)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl Drop for PersistentDynamicFetchWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

static PERSISTENT_DYNAMIC_FETCH_WORKER: OnceLock<Mutex<Option<PersistentDynamicFetchWorker>>> =
    OnceLock::new();

fn invoke_persistent_range_transport(
    transport: &DynamicPageRangeTransport,
    manifest: &DynamicPageFetchManifest,
    cache_root: &Path,
    download_budget_bytes: u64,
    timeout: Duration,
) -> Result<DynamicPageTransportReceipt> {
    let worker_lock = PERSISTENT_DYNAMIC_FETCH_WORKER.get_or_init(|| Mutex::new(None));
    let mut slot = worker_lock
        .lock()
        .map_err(|_| anyhow::anyhow!("persistent dynamic fetch worker mutex poisoned"))?;
    let replace = slot
        .as_ref()
        .map(|worker| worker.transport != *transport)
        .unwrap_or(false);
    if replace {
        if let Some(mut worker) = slot.take() {
            worker.stop();
        }
    }
    if slot.is_none() {
        *slot = Some(PersistentDynamicFetchWorker::spawn(transport, cache_root)?);
    }
    let result = slot.as_mut().expect("worker just initialized").request(
        manifest,
        cache_root,
        download_budget_bytes,
        timeout,
    );
    match result {
        Ok(response) => Ok(response),
        Err(error) => {
            if let Some(mut worker) = slot.take() {
                worker.stop();
            }
            Err(error)
        }
    }
}

fn invoke_existing_range_transport(
    transport: &DynamicPageRangeTransport,
    manifest: &DynamicPageFetchManifest,
    cache_root: &Path,
    download_budget_bytes: u64,
) -> Result<()> {
    if !transport.driver.is_file() {
        bail!(
            "dynamic Range transport driver 不存在: {}",
            transport.driver.display()
        );
    }
    let timeout = dynamic_page_fetch_timeout()?;
    invoke_persistent_range_transport(
        transport,
        manifest,
        cache_root,
        download_budget_bytes,
        timeout,
    )
    .map(|_| ())
}

#[cfg(test)]
fn invoke_existing_range_transport_with_timeout(
    transport: &DynamicPageRangeTransport,
    manifest: &DynamicPageFetchManifest,
    cache_root: &Path,
    download_budget_bytes: u64,
    timeout: Duration,
) -> Result<()> {
    let (artifacts, stdout, mut stderr) = DynamicFetchArtifacts::write(manifest, cache_root)?;
    writeln!(
        stderr,
        "rust_transport_start manifest={} budget_bytes={} timeout_seconds={}",
        artifacts.manifest.display(),
        download_budget_bytes,
        timeout.as_secs()
    )
    .context("write dynamic fetch launch metadata")?;
    stderr
        .flush()
        .context("flush dynamic fetch launch metadata")?;

    let mut child = Command::new(&transport.python)
        .arg(&transport.driver)
        .arg("--manifest")
        .arg(&artifacts.manifest)
        .arg("--cache-root")
        .arg(cache_root)
        .arg("--download-budget-bytes")
        .arg(download_budget_bytes.to_string())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .with_context(|| {
            format!(
                "启动 dynamic Range transport {} manifest={} stdout={} stderr={}",
                transport.python.display(),
                artifacts.manifest.display(),
                artifacts.stdout.display(),
                artifacts.stderr.display()
            )
        })?;

    eprintln!(
        "dynamic Range transport started pid={} manifest={} stdout={} stderr={} timeout_seconds={}",
        child.id(),
        artifacts.manifest.display(),
        artifacts.stdout.display(),
        artifacts.stderr.display(),
        timeout.as_secs()
    );
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Err(error) => {
                let kill_error = child.kill().err();
                let status = child.wait().ok();
                bail!(
                    "poll dynamic Range transport child 失败并已回收: error={} status={:?} kill_error={:?} stdout_log={} stderr_log={} stdout_tail={} stderr_tail={}",
                    error,
                    status,
                    kill_error,
                    artifacts.stdout.display(),
                    artifacts.stderr.display(),
                    read_log_tail(&artifacts.stdout),
                    read_log_tail(&artifacts.stderr)
                );
            }
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() >= timeout => {
                let kill_error = child.kill().err();
                let status = child.wait().ok();
                bail!(
                    "dynamic Range transport 超时并已回收: timeout_seconds={} status={:?} kill_error={:?} stdout_log={} stderr_log={} stdout_tail={} stderr_tail={}",
                    timeout.as_secs(),
                    status,
                    kill_error,
                    artifacts.stdout.display(),
                    artifacts.stderr.display(),
                    read_log_tail(&artifacts.stdout),
                    read_log_tail(&artifacts.stderr)
                );
            }
            Ok(None) => thread::sleep(DYNAMIC_PAGE_FETCH_POLL_INTERVAL),
        }
    };
    if !status.success() {
        bail!(
            "dynamic Range transport 退出失败: status={} stdout_log={} stderr_log={} stdout_tail={} stderr_tail={}",
            status,
            artifacts.stdout.display(),
            artifacts.stderr.display(),
            read_log_tail(&artifacts.stdout),
            read_log_tail(&artifacts.stderr)
        );
    }
    eprintln!(
        "dynamic Range transport finished pid={} status={} elapsed_ms={:.3}",
        child.id(),
        status,
        started.elapsed().as_secs_f64() * 1_000.0
    );
    artifacts.cleanup_success();
    Ok(())
}

fn dynamic_page_fetch_timeout() -> Result<Duration> {
    let Some(raw) = std::env::var_os(S14_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS_ENV) else {
        return Ok(DEFAULT_DYNAMIC_PAGE_FETCH_TIMEOUT);
    };
    let raw = raw.into_string().map_err(|_| {
        anyhow::anyhow!("{S14_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS_ENV} 必须为 UTF-8")
    })?;
    let seconds = raw.parse::<u64>().with_context(|| {
        format!("{S14_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS_ENV} 必须为正整数，actual={raw:?}")
    })?;
    if !(1..=MAX_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS).contains(&seconds) {
        bail!(
            "{S14_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS_ENV} 必须在1..={MAX_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS}，actual={seconds}"
        );
    }
    Ok(Duration::from_secs(seconds))
}

fn read_log_tail(path: &Path) -> String {
    let result = (|| -> Result<Vec<u8>> {
        let mut file = fs::File::open(path)
            .with_context(|| format!("open dynamic fetch log {}", path.display()))?;
        let bytes = file
            .metadata()
            .with_context(|| format!("stat dynamic fetch log {}", path.display()))?
            .len();
        file.seek(SeekFrom::Start(
            bytes.saturating_sub(DYNAMIC_PAGE_FETCH_LOG_TAIL_BYTES),
        ))
        .with_context(|| format!("seek dynamic fetch log {}", path.display()))?;
        let mut tail = Vec::new();
        file.read_to_end(&mut tail)
            .with_context(|| format!("read dynamic fetch log {}", path.display()))?;
        Ok(tail)
    })();
    match result {
        Ok(bytes) => String::from_utf8_lossy(&bytes).trim().to_owned(),
        Err(error) => format!("<log unavailable: {error:#}>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s14_dynamic_routed_page_plan::{
        DynamicRoutedPage, ExpertRangeIdentity, RoutedProjection,
    };
    use crate::s14_input_asset_plan::{
        S14_RANGE_CACHE_META_FORMAT, S14_RANGE_CACHE_VERIFIED_TRANSPORT,
    };
    use polaris_s14_runner::{MODEL_REPO, MODEL_REVISION};
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        cell::Cell,
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    struct CacheDir(PathBuf);

    impl CacheDir {
        fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "polaris-dynamic-readiness-{}-{stamp}",
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

    fn range(expert_id: u16, projection: &str, part: &str, ordinal: u64) -> ExpertRangeIdentity {
        let start = ordinal * 16;
        let end = start + 15;
        ExpertRangeIdentity {
            tensor: format!("layers.7.ffn.experts.{expert_id}.{projection}.{part}"),
            kind: "routed_expert".into(),
            layer: 7,
            file: "model-test.safetensors".into(),
            file_bytes: 1_000_000,
            header_tensor_table_sha256: "a".repeat(64),
            start,
            end,
            bytes: 16,
            dtype: if part == "weight" { "I8" } else { "F8_E8M0" }.into(),
            shape: vec![4, 4],
            range_key: format!("model-test.safetensors:{start}-{end}"),
            expert_id,
        }
    }

    fn plan() -> DynamicRoutedPagePlan {
        let expert_ids = [1, 2, 3, 4, 5, 6];
        let mut pages = Vec::new();
        for (slot, expert_id) in expert_ids.into_iter().enumerate() {
            for (projection_index, projection) in [
                RoutedProjection::W1,
                RoutedProjection::W2,
                RoutedProjection::W3,
            ]
            .into_iter()
            .enumerate()
            {
                let stem = projection.tensor_stem();
                let ordinal = (slot * 6 + projection_index * 2) as u64;
                pages.push(DynamicRoutedPage {
                    route_slot: slot,
                    expert_id,
                    projection,
                    weight: range(expert_id, stem, "weight", ordinal),
                    scale: range(expert_id, stem, "scale", ordinal + 1),
                });
            }
        }
        DynamicRoutedPagePlan {
            layer: 7,
            position: 9,
            expert_ids,
            route_weights: [0.30, 0.25, 0.20, 0.12, 0.08, 0.05],
            pages,
        }
    }

    fn arena_range(
        expert_id: u16,
        projection: RoutedProjection,
        part: &str,
        ordinal: u64,
    ) -> ExpertRangeIdentity {
        use crate::s14_dynamic_routed_packing::{
            S14_DYNAMIC_ROUTED_SCALE_BYTES, S14_DYNAMIC_ROUTED_WEIGHT_BYTES,
        };

        let weight = part == "weight";
        let bytes = if weight {
            S14_DYNAMIC_ROUTED_WEIGHT_BYTES
        } else {
            S14_DYNAMIC_ROUTED_SCALE_BYTES
        };
        let (dtype, shape) = match (projection, weight) {
            (RoutedProjection::W1 | RoutedProjection::W3, true) => ("I8", vec![2048, 2048]),
            (RoutedProjection::W2, true) => ("I8", vec![4096, 1024]),
            (RoutedProjection::W1 | RoutedProjection::W3, false) => ("F8_E8M0", vec![2048, 128]),
            (RoutedProjection::W2, false) => ("F8_E8M0", vec![4096, 64]),
        };
        let start = ordinal * 5_000_000;
        let end = start + bytes - 1;
        ExpertRangeIdentity {
            tensor: format!(
                "layers.7.ffn.experts.{expert_id}.{}.{part}",
                projection.tensor_stem()
            ),
            kind: "routed_expert".into(),
            layer: 7,
            file: "model-test.safetensors".into(),
            file_bytes: 200_000_000,
            header_tensor_table_sha256: "a".repeat(64),
            start,
            end,
            bytes,
            dtype: dtype.into(),
            shape,
            range_key: format!("model-test.safetensors:{start}-{end}"),
            expert_id,
        }
    }

    fn arena_plan() -> DynamicRoutedPagePlan {
        let expert_ids = [1, 2, 3, 4, 5, 6];
        let mut pages = Vec::new();
        for (slot, &expert_id) in expert_ids.iter().enumerate() {
            for (projection_index, projection) in [
                RoutedProjection::W1,
                RoutedProjection::W2,
                RoutedProjection::W3,
            ]
            .into_iter()
            .enumerate()
            {
                let ordinal = (slot * 6 + projection_index * 2) as u64;
                pages.push(DynamicRoutedPage {
                    route_slot: slot,
                    expert_id,
                    projection,
                    weight: arena_range(expert_id, projection, "weight", ordinal),
                    scale: arena_range(expert_id, projection, "scale", ordinal + 1),
                });
            }
        }
        DynamicRoutedPagePlan {
            layer: 7,
            position: 9,
            expert_ids,
            route_weights: [0.30, 0.25, 0.20, 0.12, 0.08, 0.05],
            pages,
        }
    }

    fn sha256(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn populate(plan: &DynamicRoutedPagePlan, root: &Path) {
        for physical in plan.physical_ranges().unwrap() {
            let planned = physical.planned_asset(root).unwrap();
            let bytes = vec![(physical.physical_index + 1) as u8; physical.range.bytes as usize];
            let observed = sha256(&bytes);
            fs::write(&planned.payload_path, bytes).unwrap();
            fs::write(
                &planned.proof_path,
                serde_json::to_vec(&json!({
                    "format": S14_RANGE_CACHE_META_FORMAT,
                    "cache_key": planned.cache_key,
                    "identity": {
                        "repo": MODEL_REPO,
                        "revision": MODEL_REVISION,
                        "source_file": physical.range.file,
                        "source_file_bytes": physical.range.file_bytes,
                        "start": physical.range.start,
                        "end": physical.range.end,
                        "header_tensor_table_sha256":
                            physical.range.header_tensor_table_sha256,
                    },
                    "bytes": physical.range.bytes,
                    "observed_sha256": observed,
                    "expected_sha256": null,
                    "hash_authority": "tofu",
                    "authoritative": false,
                    "verified_transport": S14_RANGE_CACHE_VERIFIED_TRANSPORT,
                }))
                .unwrap(),
            )
            .unwrap();
        }
    }

    fn fake_fetch_manifest() -> DynamicPageFetchManifest {
        DynamicPageFetchManifest {
            format: S14_DYNAMIC_PAGE_FETCH_MANIFEST_FORMAT,
            layer: 41,
            position: 1,
            entries: Vec::new(),
        }
    }

    fn fake_python_transport(cache: &CacheDir, source: &str) -> DynamicPageRangeTransport {
        let driver = cache.0.join("fake_dynamic_fetch.py");
        fs::write(&driver, source.as_bytes()).unwrap();
        let python = std::env::var_os(S14_DYNAMIC_PAGE_FETCH_PYTHON_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("python"));
        DynamicPageRangeTransport::new(python, driver)
    }

    #[test]
    fn empty_cache_reports_all_thirty_six_ranges_and_exact_bytes() {
        let cache = CacheDir::new();
        let plan = plan();
        let report = inspect_dynamic_page_cache(&plan, &cache.0).unwrap();
        assert_eq!(report.range_count, DYNAMIC_ROUTED_RANGE_COUNT);
        assert_eq!(report.ready_count, 0);
        assert_eq!(report.unready_count, DYNAMIC_ROUTED_RANGE_COUNT);
        assert_eq!(report.total_bytes, 36 * 16);
        assert_eq!(report.missing_payload_bytes, 36 * 16);
        assert!(report.ranges.iter().all(|row| !row.is_ready()));
        assert_eq!(report.ranges[0].part, "weight");
        assert_eq!(report.ranges[1].part, "scale");

        let fetch = build_unready_fetch_manifest(&plan, &report).unwrap();
        assert_eq!(fetch.entries.len(), DYNAMIC_ROUTED_RANGE_COUNT);
        assert_eq!(fetch.entries[0].range_key, report.ranges[0].range_key);
    }

    #[test]
    fn committed_cache_is_proof_and_payload_sha_verified() {
        let cache = CacheDir::new();
        let plan = plan();
        populate(&plan, &cache.0);
        let report = inspect_dynamic_page_cache(&plan, &cache.0).unwrap();
        assert_eq!(report.ready_count, DYNAMIC_ROUTED_RANGE_COUNT);
        assert_eq!(report.unready_bytes, 0);
        assert!(build_unready_fetch_manifest(&plan, &report)
            .unwrap()
            .entries
            .is_empty());
    }

    #[test]
    fn missing_proof_and_corrupt_payload_are_unready_fail_closed() {
        let cache = CacheDir::new();
        let plan = plan();
        populate(&plan, &cache.0);
        let physical = plan.physical_ranges().unwrap();
        let first = physical[0].planned_asset(&cache.0).unwrap();
        let second = physical[1].planned_asset(&cache.0).unwrap();
        fs::remove_file(first.proof_path).unwrap();
        fs::write(second.payload_path, vec![0xff; 16]).unwrap();

        let report = inspect_dynamic_page_cache(&plan, &cache.0).unwrap();
        assert_eq!(report.ready_count, DYNAMIC_ROUTED_RANGE_COUNT - 2);
        assert_eq!(
            report.ranges[0].status,
            DynamicPageCacheStatus::MissingProof
        );
        assert_eq!(report.ranges[1].status, DynamicPageCacheStatus::Invalid);
        assert_eq!(report.unready_bytes, 32);
    }

    #[test]
    fn unauthorized_materialize_returns_route_json_and_missing_keys() {
        let cache = CacheDir::new();
        let plan = plan();
        let error = materialize_dynamic_page_plan_with_fetcher(
            &plan,
            &cache.0,
            DynamicPageFetchMode::LocalOnly,
            |_, _, _| panic!("unauthorized path must not invoke fetcher"),
        )
        .unwrap_err();
        let DynamicPageMaterializeError::FetchRequired(required) = error else {
            panic!("expected structured fetch-required error");
        };
        let route: OnlineTop6 = serde_json::from_str(&required.route_json).unwrap();
        assert_eq!(route.layer, plan.layer);
        assert_eq!(route.position, plan.position);
        assert_eq!(route.expert_ids, plan.expert_ids);
        assert_eq!(
            required.missing_cache_keys.len(),
            DYNAMIC_ROUTED_RANGE_COUNT
        );
        assert_eq!(
            required.missing_payload_paths.len(),
            DYNAMIC_ROUTED_RANGE_COUNT
        );
        assert_eq!(required.unready_bytes, 36 * 16);
    }

    #[test]
    fn full_hit_materializes_without_invoking_fetcher() {
        let cache = CacheDir::new();
        let plan = plan();
        populate(&plan, &cache.0);
        let materialized = materialize_dynamic_page_plan_with_fetcher(
            &plan,
            &cache.0,
            DynamicPageFetchMode::LocalOnly,
            |_, _, _| panic!("full-hit path must not invoke fetcher"),
        )
        .unwrap();
        assert_eq!(materialized.assets.len(), DYNAMIC_ROUTED_RANGE_COUNT);
        assert_eq!(materialized.mapped_assets.len(), DYNAMIC_ROUTED_RANGE_COUNT);
    }

    #[test]
    fn authorized_miss_invokes_fetcher_once_then_rust_retries_materialize() {
        let cache = CacheDir::new();
        let plan = plan();
        let calls = Cell::new(0usize);
        let materialized = materialize_dynamic_page_plan_with_fetcher(
            &plan,
            &cache.0,
            DynamicPageFetchMode::ExplicitFetch,
            |manifest, root, budget| {
                calls.set(calls.get() + 1);
                assert_eq!(manifest.entries.len(), DYNAMIC_ROUTED_RANGE_COUNT);
                assert_eq!(budget, 36 * 16);
                populate(&plan, root);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(materialized.assets.len(), DYNAMIC_ROUTED_RANGE_COUNT);
        assert!(materialized
            .assets
            .iter()
            .all(|asset| asset.payload_rehashed_by_builder));
    }

    #[test]
    fn explicit_fetch_rehashes_and_builds_exact_canonical_routed_arena() {
        let cache = CacheDir::new();
        let plan = arena_plan();
        let calls = Cell::new(0usize);
        let arena = materialize_dynamic_routed_arena_with_fetcher(
            &plan,
            &cache.0,
            DynamicPageFetchMode::ExplicitFetch,
            |manifest, root, budget| {
                calls.set(calls.get() + 1);
                assert_eq!(manifest.entries.len(), DYNAMIC_ROUTED_RANGE_COUNT);
                assert_eq!(budget, S14_DYNAMIC_ROUTED_ARENA_BYTES);
                populate(&plan, root);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(arena.pages.assets.len(), DYNAMIC_ROUTED_RANGE_COUNT);
        assert!(arena
            .pages
            .assets
            .iter()
            .all(|asset| asset.payload_rehashed_by_builder));
        assert_eq!(arena.arena_logical_bytes(), S14_DYNAMIC_ROUTED_ARENA_BYTES);
        assert_eq!(
            arena.upload.layout.placements.len(),
            DYNAMIC_ROUTED_RANGE_COUNT
        );
    }

    #[test]
    fn transport_failure_propagates_child_stdout_stderr_and_keeps_logs() {
        let cache = CacheDir::new();
        let transport = fake_python_transport(
            &cache,
            "import sys\nprint('fake-stdout', flush=True)\nprint('fake-stderr', file=sys.stderr, flush=True)\nraise SystemExit(7)\n",
        );
        let error = invoke_existing_range_transport_with_timeout(
            &transport,
            &fake_fetch_manifest(),
            &cache.0,
            1,
            Duration::from_secs(5),
        )
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("status=exit code: 7"), "{message}");
        assert!(message.contains("fake-stdout"), "{message}");
        assert!(message.contains("fake-stderr"), "{message}");
        assert!(message.contains("stdout_log="), "{message}");
        assert!(message.contains("stderr_log="), "{message}");
        let artifacts = fs::read_dir(cache.0.join("_dynamic_fetch_logs"))
            .unwrap()
            .count();
        assert_eq!(artifacts, 3);
    }

    #[test]
    fn transport_timeout_kills_reaps_and_reports_log_tail() {
        let cache = CacheDir::new();
        let transport = fake_python_transport(
            &cache,
            "import sys, time\nprint('before-timeout', file=sys.stderr, flush=True)\ntime.sleep(60)\n",
        );
        let error = invoke_existing_range_transport_with_timeout(
            &transport,
            &fake_fetch_manifest(),
            &cache.0,
            1,
            Duration::from_millis(100),
        )
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("超时并已回收"), "{message}");
        assert!(message.contains("before-timeout"), "{message}");
        assert!(message.contains("status=Some"), "{message}");
    }
}
