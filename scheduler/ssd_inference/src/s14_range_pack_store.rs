//! Polaris S14 Range cache 的进程级不可变 SSD pack 读取器。
//!
//! 急行路径把近期 loose Range 合并为少量 append-only pack。索引一次解析，pack
//! 一次 mmap；每个 payload 在首次取得 StarFold lease 时仍重算 SHA-256，随后由既有
//! single-flight verified lease cache 复用。未被 pack 覆盖的 Range 继续走 loose/远端链。

use crate::s14_input_asset_plan::{S14PlannedRangeAsset, S14RangeIdentity};
use anyhow::{bail, Context, Result};
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    env,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

pub const S14_RANGE_PACK_INDEX_ENV: &str = "POLARIS_S14_RANGE_PACK_INDEX";
pub const S14_RANGE_PACK_INDEX_FORMAT: &str = "polaris-s14-range-pack-index-v1";
pub const S14_RANGE_PACK_HEADER_BYTES: u64 = 4096;
const S14_RANGE_PACK_MAGIC: &[u8; 8] = b"PS14PACK";
const S14_RANGE_PACK_VERSION: u32 = 1;
const S14_RANGE_PACK_MAX_INDEX_BYTES: u64 = 128 * 1024 * 1024;
const S14_RANGE_PACK_MAX_PACKS: usize = 256;
const S14_RANGE_PACK_MAX_ENTRIES: usize = 250_000;
const S14_RANGE_PACK_MAX_PROOF_BYTES: u64 = 16 * 1024 * 1024;
const S14_RANGE_CACHE_PROOF_FORMAT: &str = "polaris-s14-range-cache-entry-v1";
const S14_RANGE_VERIFIED_TRANSPORT: &str = "HTTPS/206/exact-Content-Range";

#[derive(Debug, Deserialize)]
struct RangePackIndexDocument {
    format: String,
    generation: u64,
    cache_root: PathBuf,
    alignment: u64,
    packs: BTreeMap<String, RangePackFileRecord>,
    entries: BTreeMap<String, RangePackEntryRecord>,
}

#[derive(Debug, Deserialize)]
struct RangePackFileRecord {
    bytes: u64,
    sha256: String,
    entries: u64,
}

#[derive(Clone, Debug, Deserialize)]
struct RangePackEntryRecord {
    pack: String,
    offset: u64,
    bytes: u64,
    observed_sha256: String,
    proof_sha256: String,
    hash_authority: String,
    authoritative: bool,
    identity: S14RangeIdentity,
}

#[derive(Debug, Deserialize)]
struct RangeCacheProofDocument {
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

#[derive(Debug)]
struct MappedRangePack {
    path: PathBuf,
    #[allow(dead_code)]
    file: File,
    mmap: Mmap,
}

#[derive(Debug)]
pub struct S14RangePackStore {
    canonical_cache_root: PathBuf,
    index_path: PathBuf,
    index_sha256: String,
    #[allow(dead_code)]
    index_file: File,
    alignment: u64,
    packs: HashMap<String, Arc<MappedRangePack>>,
    entries: HashMap<String, Arc<RangePackEntryRecord>>,
}

/// 尚未散列的 pack slice。只有 [`verify_payload`](Self::verify_payload) 成功后，
/// StarFold 才能把它发布为 verified lease。
#[derive(Clone, Debug)]
pub struct S14PackedRangeSource {
    store: Arc<S14RangePackStore>,
    entry: Arc<RangePackEntryRecord>,
    pack: Arc<MappedRangePack>,
}

impl S14PackedRangeSource {
    pub fn pack_path(&self) -> &Path {
        &self.pack.path
    }

    pub fn index_path(&self) -> &Path {
        &self.store.index_path
    }

    pub fn index_sha256(&self) -> &str {
        &self.store.index_sha256
    }

    pub fn payload_sha256(&self) -> &str {
        &self.entry.observed_sha256
    }

    pub fn proof_sha256(&self) -> &str {
        &self.entry.proof_sha256
    }

    pub fn hash_authority(&self) -> &str {
        &self.entry.hash_authority
    }

    pub fn payload_bytes(&self) -> Result<&[u8]> {
        let start =
            usize::try_from(self.entry.offset).context("S14 Range pack entry offset 超出 usize")?;
        let end_u64 = self
            .entry
            .offset
            .checked_add(self.entry.bytes)
            .context("S14 Range pack entry end overflow")?;
        let end = usize::try_from(end_u64).context("S14 Range pack entry end 超出 usize")?;
        self.pack
            .mmap
            .get(start..end)
            .context("S14 Range pack entry slice 越界")
    }

    pub fn verify_payload(&self) -> Result<()> {
        let observed = sha256_hex(self.payload_bytes()?);
        if observed != self.entry.observed_sha256 {
            bail!(
                "S14 Range pack payload SHA-256 漂移: pack={} offset={} bytes={} expected={} actual={observed}",
                self.pack.path.display(),
                self.entry.offset,
                self.entry.bytes,
                self.entry.observed_sha256
            );
        }
        Ok(())
    }
}

impl S14RangePackStore {
    fn load(index_path: &Path, expected_cache_root: &Path) -> Result<Self> {
        let canonical_cache_root = expected_cache_root.canonicalize().with_context(|| {
            format!(
                "resolve S14 Range cache root {}",
                expected_cache_root.display()
            )
        })?;
        if !canonical_cache_root.is_dir() {
            bail!("S14 Range cache root 不是目录");
        }
        let index_path = index_path
            .canonicalize()
            .with_context(|| format!("resolve S14 Range pack index {}", index_path.display()))?;
        let (mut index_file, index_bytes) = open_immutable_bytes(&index_path, "index")?;
        if index_bytes == 0 || index_bytes > S14_RANGE_PACK_MAX_INDEX_BYTES {
            bail!(
                "S14 Range pack index bytes 超出1..={S14_RANGE_PACK_MAX_INDEX_BYTES}: actual={index_bytes}"
            );
        }
        let mut raw = Vec::with_capacity(
            usize::try_from(index_bytes).context("S14 Range pack index bytes 超出 usize")?,
        );
        use std::io::Read;
        index_file
            .read_to_end(&mut raw)
            .with_context(|| format!("读取 S14 Range pack index {}", index_path.display()))?;
        if raw.len() as u64 != index_bytes {
            bail!("S14 Range pack index 读取长度漂移");
        }
        let index_sha256 = sha256_hex(&raw);
        let document: RangePackIndexDocument =
            serde_json::from_slice(&raw).context("解析 S14 Range pack index JSON")?;
        validate_document_shape(&document, &canonical_cache_root)?;

        let pack_root = index_path
            .parent()
            .context("S14 Range pack index 缺少父目录")?
            .canonicalize()
            .context("resolve S14 Range pack root")?;
        let mut packs = HashMap::with_capacity(document.packs.len());
        for (name, record) in &document.packs {
            validate_pack_name(name)?;
            validate_sha256(&record.sha256, "pack")?;
            if record.bytes < S14_RANGE_PACK_HEADER_BYTES || record.entries == 0 {
                bail!("S14 Range pack file record bytes/entries 非法: {name}");
            }
            let path = pack_root.join(name).canonicalize().with_context(|| {
                format!(
                    "resolve S14 Range pack payload {}",
                    pack_root.join(name).display()
                )
            })?;
            if !path.starts_with(&pack_root) {
                bail!("S14 Range pack payload 越出 index 目录: {}", path.display());
            }
            let file = open_immutable(&path)
                .with_context(|| format!("打开 S14 Range pack payload {}", path.display()))?;
            let metadata = file
                .metadata()
                .with_context(|| format!("stat S14 Range pack payload {}", path.display()))?;
            if !metadata.is_file() || metadata.len() != record.bytes {
                bail!(
                    "S14 Range pack bytes 漂移: pack={name} expected={} actual={}",
                    record.bytes,
                    metadata.len()
                );
            }
            let mmap = unsafe { MmapOptions::new().map(&file) }
                .with_context(|| format!("mmap S14 Range pack payload {}", path.display()))?;
            validate_pack_header(&mmap, name)?;
            packs.insert(name.clone(), Arc::new(MappedRangePack { path, file, mmap }));
        }

        let mut entries = HashMap::with_capacity(document.entries.len());
        let mut ranges = HashMap::<String, Vec<(u64, u64, String)>>::new();
        let mut per_pack_counts = HashMap::<String, u64>::new();
        for (cache_key, entry) in document.entries {
            validate_sha256(&cache_key, "cache_key")?;
            validate_sha256(&entry.observed_sha256, "payload")?;
            validate_sha256(&entry.proof_sha256, "proof")?;
            if entry.bytes == 0
                || entry.offset < S14_RANGE_PACK_HEADER_BYTES
                || entry.offset % document.alignment != 0
                || !matches!(entry.hash_authority.as_str(), "tofu" | "official_lock")
                || (entry.authoritative != (entry.hash_authority == "official_lock"))
            {
                bail!("S14 Range pack entry 合同非法: cache_key={cache_key}");
            }
            verify_loose_proof_anchor(&canonical_cache_root, &cache_key, &entry)?;
            let pack = packs
                .get(&entry.pack)
                .with_context(|| format!("S14 Range pack entry 引用未知 pack: {}", entry.pack))?;
            let end = entry
                .offset
                .checked_add(entry.bytes)
                .context("S14 Range pack entry end overflow")?;
            if end > pack.mmap.len() as u64 {
                bail!("S14 Range pack entry 越出 pack: cache_key={cache_key}");
            }
            ranges.entry(entry.pack.clone()).or_default().push((
                entry.offset,
                end,
                cache_key.clone(),
            ));
            *per_pack_counts.entry(entry.pack.clone()).or_default() += 1;
            entries.insert(cache_key, Arc::new(entry));
        }
        for (pack_name, mut rows) in ranges {
            rows.sort_by_key(|row| row.0);
            for pair in rows.windows(2) {
                if pair[0].1 > pair[1].0 {
                    bail!(
                        "S14 Range pack entry 重叠: pack={pack_name} left={} right={}",
                        pair[0].2,
                        pair[1].2
                    );
                }
            }
        }
        for (name, record) in &document.packs {
            if per_pack_counts.get(name).copied().unwrap_or(0) != record.entries {
                bail!("S14 Range pack entry count 漂移: pack={name}");
            }
        }
        Ok(Self {
            canonical_cache_root,
            index_path,
            index_sha256,
            index_file,
            alignment: document.alignment,
            packs,
            entries,
        })
    }

    pub fn lookup(
        self: &Arc<Self>,
        planned: &S14PlannedRangeAsset,
    ) -> Result<Option<S14PackedRangeSource>> {
        let Some(entry) = self.entries.get(&planned.cache_key).cloned() else {
            return Ok(None);
        };
        if entry.bytes != planned.bytes || entry.identity != planned.identity {
            bail!(
                "S14 Range pack entry 与 planned identity 漂移: cache_key={}",
                planned.cache_key
            );
        }
        let expected_payload = self
            .canonical_cache_root
            .join(format!("{}.bin", planned.cache_key));
        let expected_proof = self
            .canonical_cache_root
            .join(format!("{}.json", planned.cache_key));
        if planned.payload_path != expected_payload || planned.proof_path != expected_proof {
            bail!("S14 Range pack planned loose 路径不是 canonical cache key 派生");
        }
        if entry.offset % self.alignment != 0 {
            bail!("S14 Range pack entry alignment 漂移");
        }
        let pack = self
            .packs
            .get(&entry.pack)
            .cloned()
            .context("S14 Range pack mapped payload 消失")?;
        Ok(Some(S14PackedRangeSource {
            store: Arc::clone(self),
            entry,
            pack,
        }))
    }
}

enum ProcessRangePackStore {
    Disabled,
    Ready(Arc<S14RangePackStore>),
    Failed(String),
}

static PROCESS_RANGE_PACK_STORE: OnceLock<ProcessRangePackStore> = OnceLock::new();

/// 环境变量未配置时完全保持原 loose 路径；显式配置但索引无效时 fail-closed。
pub fn process_s14_range_pack_store(cache_root: &Path) -> Result<Option<Arc<S14RangePackStore>>> {
    let state = PROCESS_RANGE_PACK_STORE.get_or_init(|| {
        let Some(path) = env::var_os(S14_RANGE_PACK_INDEX_ENV) else {
            return ProcessRangePackStore::Disabled;
        };
        match S14RangePackStore::load(&PathBuf::from(path), cache_root) {
            Ok(store) => ProcessRangePackStore::Ready(Arc::new(store)),
            Err(error) => ProcessRangePackStore::Failed(format!("{error:#}")),
        }
    });
    match state {
        ProcessRangePackStore::Disabled => Ok(None),
        ProcessRangePackStore::Ready(store) => {
            if cache_root != store.canonical_cache_root {
                let canonical = cache_root.canonicalize().with_context(|| {
                    format!("resolve S14 Range cache root {}", cache_root.display())
                })?;
                if canonical != store.canonical_cache_root {
                    bail!("进程级 S14 Range pack store 被不同 cache root 复用");
                }
            }
            Ok(Some(Arc::clone(store)))
        }
        ProcessRangePackStore::Failed(error) => {
            bail!("加载 S14 Range pack store 失败: {error}")
        }
    }
}

fn validate_document_shape(document: &RangePackIndexDocument, cache_root: &Path) -> Result<()> {
    if document.format != S14_RANGE_PACK_INDEX_FORMAT
        || document.generation == 0
        || document.alignment < 64
        || !document.alignment.is_power_of_two()
        || document.packs.is_empty()
        || document.packs.len() > S14_RANGE_PACK_MAX_PACKS
        || document.entries.is_empty()
        || document.entries.len() > S14_RANGE_PACK_MAX_ENTRIES
    {
        bail!("S14 Range pack index header/count/alignment 非法");
    }
    let declared_root = document.cache_root.canonicalize().with_context(|| {
        format!(
            "resolve S14 Range pack declared cache root {}",
            document.cache_root.display()
        )
    })?;
    if declared_root != cache_root {
        bail!("S14 Range pack index cache_root 与 runtime 不一致");
    }
    Ok(())
}

fn validate_pack_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('/')
        || name.contains('\\')
        || !name.starts_with("range-pack-")
        || !name.ends_with(".bin")
    {
        bail!("S14 Range pack 文件名非法: {name:?}");
    }
    Ok(())
}

fn validate_pack_header(bytes: &[u8], name: &str) -> Result<()> {
    if bytes.len() < S14_RANGE_PACK_HEADER_BYTES as usize
        || bytes.get(0..8) != Some(S14_RANGE_PACK_MAGIC.as_slice())
    {
        bail!("S14 Range pack magic/header 漂移: {name}");
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let header_bytes = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if version != S14_RANGE_PACK_VERSION || u64::from(header_bytes) != S14_RANGE_PACK_HEADER_BYTES {
        bail!("S14 Range pack version/header bytes 漂移: {name}");
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("S14 Range pack {label} SHA-256 必须是64位小写十六进制");
    }
    Ok(())
}

/// pack index 不是新的真实性根。进程启动时把每个 entry 重新绑定到原 loose proof；
/// payload 首次使用时再由 packed source 对 pack slice 重算 SHA。
fn verify_loose_proof_anchor(
    cache_root: &Path,
    cache_key: &str,
    entry: &RangePackEntryRecord,
) -> Result<()> {
    let declared_path = cache_root.join(format!("{cache_key}.json"));
    let proof_path = declared_path
        .canonicalize()
        .with_context(|| format!("resolve S14 Range loose proof {}", declared_path.display()))?;
    if !proof_path.starts_with(cache_root) {
        bail!(
            "S14 Range loose proof 越出 cache root: {}",
            proof_path.display()
        );
    }
    let (mut proof_file, proof_bytes) = open_immutable_bytes(&proof_path, "loose proof")?;
    if proof_bytes == 0 || proof_bytes > S14_RANGE_PACK_MAX_PROOF_BYTES {
        bail!(
            "S14 Range loose proof bytes 超出范围: {}",
            proof_path.display()
        );
    }
    let mut raw = Vec::with_capacity(
        usize::try_from(proof_bytes).context("S14 Range loose proof bytes 超出 usize")?,
    );
    use std::io::Read;
    proof_file
        .read_to_end(&mut raw)
        .with_context(|| format!("读取 S14 Range loose proof {}", proof_path.display()))?;
    if raw.len() as u64 != proof_bytes || sha256_hex(&raw) != entry.proof_sha256 {
        bail!("S14 Range pack entry loose proof SHA/bytes 漂移: {cache_key}");
    }
    let proof: RangeCacheProofDocument =
        serde_json::from_slice(&raw).context("解析 S14 Range loose proof JSON")?;
    let expected_authoritative = entry.hash_authority == "official_lock";
    let expected_sha_ok = if expected_authoritative {
        proof.expected_sha256.as_deref() == Some(entry.observed_sha256.as_str())
    } else {
        proof.expected_sha256.is_none()
    };
    let range_bytes = entry
        .identity
        .end
        .checked_sub(entry.identity.start)
        .and_then(|value| value.checked_add(1))
        .context("S14 Range pack identity bounds overflow")?;
    if proof.format != S14_RANGE_CACHE_PROOF_FORMAT
        || proof.cache_key != cache_key
        || canonical_range_cache_key(&proof.identity)? != cache_key
        || proof.identity != entry.identity
        || proof.bytes != entry.bytes
        || range_bytes != entry.bytes
        || proof.observed_sha256 != entry.observed_sha256
        || proof.hash_authority != entry.hash_authority
        || proof.authoritative != entry.authoritative
        || proof.authoritative != expected_authoritative
        || !expected_sha_ok
        || proof.verified_transport != S14_RANGE_VERIFIED_TRANSPORT
    {
        bail!("S14 Range pack entry 未绑定原 loose proof: {cache_key}");
    }
    Ok(())
}

fn canonical_range_cache_key(identity: &S14RangeIdentity) -> Result<String> {
    let mut canonical = BTreeMap::<&str, serde_json::Value>::new();
    canonical.insert("end", serde_json::Value::from(identity.end));
    canonical.insert(
        "header_tensor_table_sha256",
        serde_json::Value::from(identity.header_tensor_table_sha256.clone()),
    );
    canonical.insert("repo", serde_json::Value::from(identity.repo.clone()));
    canonical.insert(
        "revision",
        serde_json::Value::from(identity.revision.clone()),
    );
    canonical.insert(
        "source_file",
        serde_json::Value::from(identity.source_file.clone()),
    );
    canonical.insert(
        "source_file_bytes",
        serde_json::Value::from(identity.source_file_bytes),
    );
    canonical.insert("start", serde_json::Value::from(identity.start));
    let bytes = serde_json::to_vec(&canonical).context("编码 canonical S14 Range identity")?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn open_immutable_bytes(path: &Path, label: &str) -> Result<(File, u64)> {
    let file = open_immutable(path)
        .with_context(|| format!("打开 S14 Range pack {label} {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("stat S14 Range pack {label} {}", path.display()))?;
    if !metadata.is_file() {
        bail!("S14 Range pack {label} 不是文件: {}", path.display());
    }
    Ok((file, metadata.len()))
}

#[cfg(windows)]
fn open_immutable(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .share_mode(0x0000_0001)
        .open(path)
}

#[cfg(not(windows))]
fn open_immutable(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().read(true).open(path)
}
