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
    error::Error,
    fmt,
    fs::{self, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
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
const MAX_DYNAMIC_PAGE_FETCH_TIMEOUT_SECONDS: u64 = 86_400;
const DYNAMIC_PAGE_FETCH_POLL_INTERVAL: Duration = Duration::from_millis(200);
const DYNAMIC_PAGE_FETCH_LOG_TAIL_BYTES: u64 = 16 * 1024;

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

#[derive(Clone, Debug, Serialize)]
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

struct DynamicFetchArtifacts {
    manifest: PathBuf,
    stdout: PathBuf,
    stderr: PathBuf,
}

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
    invoke_existing_range_transport_with_timeout(
        transport,
        manifest,
        cache_root,
        download_budget_bytes,
        timeout,
    )
}

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
