//! Polaris S14 committed K-block 的可恢复持久检查点。
//!
//! 文件只在 host/device 已提交字节完全一致时发布；同一份二进制 arena 因此同时是
//! `DecoderStateV1::native_arena` 与 active device checkpoint 的可恢复镜像。头部、载荷和
//! 整体摘要均经过 SHA-256 闭合，身份不同或任意字节损坏均 fail-closed。

use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{
    DecoderStateV1, NativeState, NativeStateArena, Position0WholeTokenManifest, TokenRecord,
    MODEL_REPO, MODEL_REVISION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

const MAGIC: &[u8; 8] = b"S14K4CP1";
const SCHEMA_VERSION: u32 = 1;
const FIXED_HEADER_BYTES: usize = 8 + 4 + 8 + 8;
const TRAILER_BYTES: usize = 32;
const MAX_JSON_HEADER_BYTES: u64 = 4 * 1024 * 1024;
const MAX_ARENA_BYTES: u64 = 512 * 1024 * 1024;
const FORMAT: &str = "polaris-s14-durable-committed-k-block-v1";
const DEVICE_MIRROR: &str = "native_arena_byte_exact_active_device_checkpoint";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct S14DurableCheckpointIdentity {
    pub model_repo: String,
    pub model_revision: String,
    pub graph_profile: String,
    pub manifest_sha256: String,
    pub catalog_sha256: String,
    pub proof_identity_sha256: String,
}

impl S14DurableCheckpointIdentity {
    /// 从 runtime 真正加载的 manifest/catalog 原始字节生成固定身份。
    pub fn from_runtime_assets(
        manifest: &Position0WholeTokenManifest,
        manifest_path: &Path,
        catalog_path: &Path,
    ) -> Result<Self> {
        if manifest.repo != MODEL_REPO || manifest.revision != MODEL_REVISION {
            bail!("durable checkpoint 拒绝非固定 S14 repo/revision");
        }
        let manifest_sha256 = sha256_file(manifest_path)
            .with_context(|| format!("hash S14 manifest: {}", manifest_path.display()))?;
        let catalog_sha256 = sha256_file(catalog_path)
            .with_context(|| format!("hash S14 catalog: {}", catalog_path.display()))?;
        if catalog_sha256 != manifest.catalog.sha256 {
            bail!(
                "durable checkpoint catalog SHA 与 manifest 漂移: observed={catalog_sha256} expected={}",
                manifest.catalog.sha256
            );
        }
        let mut proof = Sha256::new();
        proof.update(b"polaris-s14-durable-proof-root-v1\0");
        for value in [
            manifest.repo.as_str(),
            manifest.revision.as_str(),
            manifest.profile.as_str(),
            manifest_sha256.as_str(),
            catalog_sha256.as_str(),
            manifest.source_report.sha256.as_str(),
        ] {
            proof.update((value.len() as u64).to_le_bytes());
            proof.update(value.as_bytes());
        }
        Ok(Self {
            model_repo: manifest.repo.clone(),
            model_revision: manifest.revision.clone(),
            graph_profile: manifest.profile.clone(),
            manifest_sha256,
            catalog_sha256,
            proof_identity_sha256: hex_digest(proof.finalize().as_slice()),
        })
    }

    fn validate(&self) -> Result<()> {
        if self.model_repo != MODEL_REPO
            || self.model_revision != MODEL_REVISION
            || self.graph_profile.is_empty()
        {
            bail!("durable checkpoint model/revision/profile identity 非法");
        }
        for (name, value) in [
            ("manifest", self.manifest_sha256.as_str()),
            ("catalog", self.catalog_sha256.as_str()),
            ("proof", self.proof_identity_sha256.as_str()),
        ] {
            validate_sha256(name, value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14DurableCheckpointReceipt {
    pub path: PathBuf,
    pub position: u32,
    pub input_token_id: u32,
    pub commit_epoch: u64,
    pub active_bank: usize,
    pub arena_bytes: u64,
    pub arena_sha256: String,
    pub file_sha256: String,
}

#[derive(Debug)]
pub struct S14RestoredCommittedBlock {
    pub authoritative: DecoderStateV1,
    pub identity: S14DurableCheckpointIdentity,
    pub file_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DecoderStateMetadataV1 {
    abi_version: u32,
    commit_epoch: u64,
    position: u32,
    input_token_id: u32,
    active_fixed_bank: u8,
    committed_tokens: Vec<TokenRecord>,
    native: NativeState,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointHeaderV1 {
    format: String,
    schema_version: u32,
    identity: S14DurableCheckpointIdentity,
    decoder: DecoderStateMetadataV1,
    native_arena_bytes: u64,
    native_arena_sha256: String,
    device_checkpoint_bytes: u64,
    device_checkpoint_sha256: String,
    device_checkpoint_mirror: String,
}

/// 原子发布一个已提交 block。`device_checkpoint_bytes` 必须来自当前 active bank
/// 的完整 readback，不允许传 host fixture 冒充。
pub fn persist_committed_block(
    path: &Path,
    identity: &S14DurableCheckpointIdentity,
    authoritative: &DecoderStateV1,
    device_checkpoint_bytes: &[u8],
) -> Result<S14DurableCheckpointReceipt> {
    identity.validate()?;
    validate_committed_state(authoritative)?;
    let arena = authoritative.native_arena.bytes();
    if device_checkpoint_bytes != arena {
        bail!(
            "durable checkpoint 拒绝 host/device 非同源已提交字节: host={} device={}",
            arena.len(),
            device_checkpoint_bytes.len()
        );
    }
    let arena_bytes = u64::try_from(arena.len()).context("native arena 长度无法表示")?;
    if arena_bytes == 0 || arena_bytes > MAX_ARENA_BYTES {
        bail!("durable checkpoint native arena 长度越界: {arena_bytes}");
    }
    let arena_sha256 = sha256_bytes(arena);
    let header = CheckpointHeaderV1 {
        format: FORMAT.to_owned(),
        schema_version: SCHEMA_VERSION,
        identity: identity.clone(),
        decoder: DecoderStateMetadataV1 {
            abi_version: authoritative.abi_version,
            commit_epoch: authoritative.commit_epoch,
            position: authoritative.position,
            input_token_id: authoritative.input_token_id,
            active_fixed_bank: authoritative.active_fixed_bank,
            committed_tokens: authoritative.committed_tokens.clone(),
            native: authoritative.native.clone(),
        },
        native_arena_bytes: arena_bytes,
        native_arena_sha256: arena_sha256.clone(),
        device_checkpoint_bytes: arena_bytes,
        device_checkpoint_sha256: arena_sha256.clone(),
        device_checkpoint_mirror: DEVICE_MIRROR.to_owned(),
    };
    let json = serde_json::to_vec(&header).context("encode durable checkpoint header")?;
    let json_bytes = u64::try_from(json.len()).context("checkpoint header 长度无法表示")?;
    if json_bytes == 0 || json_bytes > MAX_JSON_HEADER_BYTES {
        bail!("durable checkpoint JSON header 长度越界: {json_bytes}");
    }

    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("创建 checkpoint 目录: {}", parent.display()))?;
    let temp = unique_temp_path(path, authoritative.commit_epoch)?;
    let write_result = write_checkpoint_file(&temp, json_bytes, arena_bytes, &json, arena);
    let file_sha256 = match write_result {
        Ok(value) => value,
        Err(error) => {
            let _ = fs::remove_file(&temp);
            return Err(error);
        }
    };
    if let Err(error) = atomic_publish(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    sync_parent_directory(parent)?;

    Ok(S14DurableCheckpointReceipt {
        path: path.to_path_buf(),
        position: authoritative.position,
        input_token_id: authoritative.input_token_id,
        commit_epoch: authoritative.commit_epoch,
        active_bank: usize::from(authoritative.active_fixed_bank),
        arena_bytes,
        arena_sha256,
        file_sha256,
    })
}

/// 加载并验证持久 block。返回前已完成整体 SHA、载荷 SHA、身份、长度与
/// `DecoderStateV1::validate` 的全部检查。
pub fn restore_committed_block(
    path: &Path,
    expected_identity: &S14DurableCheckpointIdentity,
) -> Result<S14RestoredCommittedBlock> {
    expected_identity.validate()?;
    let file =
        File::open(path).with_context(|| format!("打开 durable checkpoint: {}", path.display()))?;
    let file_bytes = file.metadata()?.len();
    let minimum = (FIXED_HEADER_BYTES + TRAILER_BYTES) as u64;
    if file_bytes <= minimum || file_bytes > MAX_JSON_HEADER_BYTES + MAX_ARENA_BYTES + minimum {
        bail!("durable checkpoint 文件长度越界: {file_bytes}");
    }
    let mut reader = BufReader::new(file);
    let mut fixed = [0u8; FIXED_HEADER_BYTES];
    reader
        .read_exact(&mut fixed)
        .context("读取 durable checkpoint fixed header")?;
    if &fixed[..8] != MAGIC {
        bail!("durable checkpoint magic 漂移");
    }
    let schema = u32::from_le_bytes(fixed[8..12].try_into().unwrap());
    let json_bytes = u64::from_le_bytes(fixed[12..20].try_into().unwrap());
    let arena_bytes = u64::from_le_bytes(fixed[20..28].try_into().unwrap());
    if schema != SCHEMA_VERSION
        || json_bytes == 0
        || json_bytes > MAX_JSON_HEADER_BYTES
        || arena_bytes == 0
        || arena_bytes > MAX_ARENA_BYTES
    {
        bail!("durable checkpoint schema/header/arena 长度非法");
    }
    let expected_file_bytes = (FIXED_HEADER_BYTES as u64)
        .checked_add(json_bytes)
        .and_then(|value| value.checked_add(arena_bytes))
        .and_then(|value| value.checked_add(TRAILER_BYTES as u64))
        .context("durable checkpoint 文件长度溢出")?;
    if file_bytes != expected_file_bytes {
        bail!(
            "durable checkpoint 文件长度不闭合: actual={file_bytes} expected={expected_file_bytes}"
        );
    }

    let mut hasher = Sha256::new();
    hasher.update(fixed);
    let mut json = vec![0u8; usize::try_from(json_bytes)?];
    reader
        .read_exact(&mut json)
        .context("读取 durable checkpoint JSON header")?;
    hasher.update(&json);
    let mut arena = vec![0u8; usize::try_from(arena_bytes)?];
    reader
        .read_exact(&mut arena)
        .context("读取 durable checkpoint arena")?;
    hasher.update(&arena);
    let mut stored_digest = [0u8; TRAILER_BYTES];
    reader
        .read_exact(&mut stored_digest)
        .context("读取 durable checkpoint trailer")?;
    let observed_digest = hasher.finalize();
    if observed_digest.as_slice() != stored_digest {
        bail!("durable checkpoint 整体 SHA-256 漂移");
    }
    let file_sha256 = hex_digest(&stored_digest);
    let header: CheckpointHeaderV1 =
        serde_json::from_slice(&json).context("解析 durable checkpoint JSON header")?;
    if header.format != FORMAT
        || header.schema_version != SCHEMA_VERSION
        || &header.identity != expected_identity
    {
        bail!("durable checkpoint format/schema/runtime identity 漂移");
    }
    header.identity.validate()?;
    if header.native_arena_bytes != arena_bytes
        || header.device_checkpoint_bytes != arena_bytes
        || header.native_arena_sha256 != header.device_checkpoint_sha256
        || header.device_checkpoint_mirror != DEVICE_MIRROR
        || sha256_bytes(&arena) != header.native_arena_sha256
    {
        bail!("durable checkpoint native/device 镜像长度或 SHA 漂移");
    }

    let native_arena =
        NativeStateArena::from_verified_checkpoint_bytes(&header.decoder.native, arena)
            .context("从 durable checkpoint 恢复 NativeStateArena")?;
    let authoritative = DecoderStateV1 {
        abi_version: header.decoder.abi_version,
        commit_epoch: header.decoder.commit_epoch,
        position: header.decoder.position,
        input_token_id: header.decoder.input_token_id,
        active_fixed_bank: header.decoder.active_fixed_bank,
        committed_tokens: header.decoder.committed_tokens,
        native: header.decoder.native,
        native_arena,
    };
    validate_committed_state(&authoritative)?;
    Ok(S14RestoredCommittedBlock {
        authoritative,
        identity: header.identity,
        file_sha256,
    })
}

fn validate_committed_state(state: &DecoderStateV1) -> Result<()> {
    state
        .validate()
        .context("validate durable DecoderStateV1")?;
    if state.position == 0
        || state.commit_epoch != u64::from(state.position)
        || usize::from(state.active_fixed_bank) != (state.commit_epoch as usize & 1)
        || state.native_arena.is_empty()
    {
        bail!(
            "durable checkpoint 拒绝非已提交 block 身份: position={} epoch={} bank={}",
            state.position,
            state.commit_epoch,
            state.active_fixed_bank
        );
    }
    Ok(())
}

fn write_checkpoint_file(
    temp: &Path,
    json_bytes: u64,
    arena_bytes: u64,
    json: &[u8],
    arena: &[u8],
) -> Result<String> {
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(temp)
        .with_context(|| format!("创建 checkpoint temp: {}", temp.display()))?;
    let mut writer = BufWriter::new(file);
    let mut fixed = Vec::with_capacity(FIXED_HEADER_BYTES);
    fixed.extend_from_slice(MAGIC);
    fixed.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
    fixed.extend_from_slice(&json_bytes.to_le_bytes());
    fixed.extend_from_slice(&arena_bytes.to_le_bytes());
    debug_assert_eq!(fixed.len(), FIXED_HEADER_BYTES);
    let mut hasher = Sha256::new();
    for bytes in [fixed.as_slice(), json, arena] {
        writer.write_all(bytes)?;
        hasher.update(bytes);
    }
    let digest = hasher.finalize();
    writer.write_all(digest.as_slice())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(hex_digest(digest.as_slice()))
}

fn unique_temp_path(path: &Path, epoch: u64) -> Result<PathBuf> {
    let name = path
        .file_name()
        .context("durable checkpoint path 缺少文件名")?
        .to_string_lossy();
    for sequence in 0..1024u32 {
        let candidate = path.with_file_name(format!(
            ".{name}.tmp.{}.{}.{}",
            std::process::id(),
            epoch,
            sequence
        ));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(anyhow!("durable checkpoint temp 名额尽"))
}

#[cfg(windows)]
fn atomic_publish(temp: &Path, destination: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, new_name: *const u16, flags: u32) -> i32;
    }

    let existing: Vec<u16> = temp
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let new_name: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let moved = unsafe {
        MoveFileExW(
            existing.as_ptr(),
            new_name.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "原子发布 checkpoint {} -> {}",
                temp.display(),
                destination.display()
            )
        });
    }
    Ok(())
}

#[cfg(not(windows))]
fn atomic_publish(temp: &Path, destination: &Path) -> Result<()> {
    fs::rename(temp, destination).with_context(|| {
        format!(
            "原子发布 checkpoint {} -> {}",
            temp.display(),
            destination.display()
        )
    })
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> Result<()> {
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> Result<()> {
    // Windows 发布路径已使用 MOVEFILE_WRITE_THROUGH。
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    // Windows 主线程默认栈通常只有 1 MiB；这是 runtime load 公共路径，
    // 不能要求每个调用者额外扩栈。
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(hasher.finalize().as_slice()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes).as_slice())
}

fn validate_sha256(name: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        bail!("durable checkpoint {name} SHA-256 格式非法");
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn committed_position1() -> DecoderStateV1 {
        let mut state = DecoderStateV1::new(8, 0).unwrap();
        state.commit_epoch = 1;
        state.position = 1;
        state.input_token_id = 5;
        state.active_fixed_bank = 1;
        state.native.position = 1;
        state.committed_tokens.push(TokenRecord {
            position: 0,
            input_token_id: 0,
            predicted_token_id: 5,
        });
        state.validate().unwrap();
        state
    }

    fn identity() -> S14DurableCheckpointIdentity {
        S14DurableCheckpointIdentity {
            model_repo: MODEL_REPO.to_owned(),
            model_revision: MODEL_REVISION.to_owned(),
            graph_profile: "fulldepth43_native_top6".to_owned(),
            manifest_sha256: "1".repeat(64),
            catalog_sha256: "2".repeat(64),
            proof_identity_sha256: "3".repeat(64),
        }
    }

    #[test]
    fn committed_block_roundtrip_and_corruption_fail_closed() {
        let state = committed_position1();
        let root = std::env::temp_dir().join(format!(
            "polaris-s14-durable-checkpoint-{}-{}",
            std::process::id(),
            state.native_arena.len()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("latest.s14ckpt");
        let receipt =
            persist_committed_block(&path, &identity(), &state, state.native_arena.bytes())
                .unwrap();
        // 同一 latest 路径必须能用 temp->atomic replace 覆盖，不留半文件窗口。
        persist_committed_block(&path, &identity(), &state, state.native_arena.bytes()).unwrap();
        let restored = restore_committed_block(&path, &identity()).unwrap();
        assert_eq!(restored.authoritative, state);
        assert_eq!(receipt.file_sha256, restored.file_sha256);
        let mut wrong_identity = identity();
        wrong_identity.catalog_sha256 = "4".repeat(64);
        assert!(restore_committed_block(&path, &wrong_identity)
            .unwrap_err()
            .to_string()
            .contains("runtime identity 漂移"));

        let mut bytes = fs::read(&path).unwrap();
        let payload_index = FIXED_HEADER_BYTES + 16;
        bytes[payload_index] ^= 0x80;
        fs::write(&path, bytes).unwrap();
        assert!(restore_committed_block(&path, &identity())
            .unwrap_err()
            .to_string()
            .contains("整体 SHA-256 漂移"));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&root);
    }
}
