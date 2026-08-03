//! FullDepth43 position0 whole-token 资产清单的运行时合同。
//!
//! Python 构建器负责把一次真实运行的 embedding、43 层 route/Range 与最终头冻结到
//! 同一份 JSON。这里是 Rust worker 的 fail-closed 入口：在任何 GPU 计算开始前验证
//! 模型身份、层序、压缩率、top-6、资产身份与汇总账本。完整 payload SHA 仍由实际
//! 读取它的 worker 在计算前校验，不能用本模块的结构验证替代。

use crate::{
    COMPRESS_RATIOS, EXPERTS_PER_TOKEN, FULL_DEPTH_LAYERS, MODEL_REPO, MODEL_REVISION,
    N_ROUTED_EXPERTS, VOCAB_SIZE,
};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

pub const POSITION0_MANIFEST_FORMAT: &str = "polaris-fulldepth43-position0-whole-token-manifest-v1";
pub const POSITION0_PROFILE: &str = "fulldepth43_native_top6";
pub const POSITION0_SOURCE_RUN: &str = "causal-block-k4-forced-fetch2-20260802";
pub const POSITION0_SOURCE_REPORT_SHA256: &str =
    "800908258abe4d0f045ca11750e3289924034119b084c2474bb8b9d4df6c8764";
pub const POSITION0_CATALOG_SHA256: &str =
    "ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049";
pub const POSITION0_CAPTURE_CHAIN_SHA256: &str =
    "7265e997d2d9a04c2100604feeea2ca24123142020d3a2c9e88b7b7a2712124e";

const BASE_NON_EXPERT_SUFFIXES: [&str; 21] = [
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.scale",
    "attn.wkv.weight",
    "attn.wo_a.scale",
    "attn.wo_a.weight",
    "attn.wo_b.scale",
    "attn.wo_b.weight",
    "attn.wq_a.scale",
    "attn.wq_a.weight",
    "attn.wq_b.scale",
    "attn.wq_b.weight",
    "attn_norm.weight",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
];
const COMPRESSOR_SUFFIXES: [&str; 4] = [
    "attn.compressor.ape",
    "attn.compressor.norm.weight",
    "attn.compressor.wgate.weight",
    "attn.compressor.wkv.weight",
];
const RATIO4_INDEXER_SUFFIXES: [&str; 7] = [
    "attn.indexer.compressor.ape",
    "attn.indexer.compressor.norm.weight",
    "attn.indexer.compressor.wgate.weight",
    "attn.indexer.compressor.wkv.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.wq_b.scale",
    "attn.indexer.wq_b.weight",
];
const SHARED_SUFFIXES: [&str; 6] = [
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.shared_experts.w3.weight",
];
const EXPERT_SUFFIXES: [&str; 6] = [
    "w1.scale",
    "w1.weight",
    "w2.scale",
    "w2.weight",
    "w3.scale",
    "w3.weight",
];
const FINAL_TENSORS: [&str; 5] = [
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "head.weight",
    "norm.weight",
];

#[derive(Debug, Clone, Deserialize)]
pub struct Position0WholeTokenManifest {
    pub format: String,
    pub repo: String,
    pub revision: String,
    pub profile: String,
    pub position: u32,
    pub input_token_id: u32,
    pub expected_output_token_id: u32,
    pub catalog: Position0Catalog,
    pub source_report: Position0SourceReport,
    pub verification_policy: Position0VerificationPolicy,
    pub embedding_row: Position0Asset,
    pub layers: Vec<Position0Layer>,
    #[serde(rename = "final")]
    pub final_section: Position0Final,
    pub summary: Position0Summary,
    pub runtime_contract: String,
    pub claim_limit: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0Catalog {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0SourceReport {
    pub path: PathBuf,
    pub sha256: String,
    pub format: String,
    pub status: String,
    pub position0_state_committed: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0VerificationPolicy {
    pub builder_payload_mode: String,
    pub range_proof_metadata_checked: bool,
    pub capture_and_report_files_rehashed: bool,
    pub runtime_must_verify_payload_sha256_before_compute: bool,
    pub source_run_rust_receipts_are_historical_evidence_only: bool,
    pub fail_closed: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0Asset {
    pub tensor: String,
    pub kind: String,
    pub expert_id: Option<u16>,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    pub range_key: String,
    pub cache_key: String,
    pub path: PathBuf,
    pub sha256: String,
    pub proof_path: PathBuf,
    pub proof_sha256: String,
    pub hash_authority: String,
    pub payload_rehashed_by_builder: bool,
    pub source: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0LayerAssets {
    pub non_expert: Vec<Position0Asset>,
    pub router: Vec<Position0Asset>,
    pub shared: Vec<Position0Asset>,
    pub routed: Vec<Position0Asset>,
}

impl Position0LayerAssets {
    pub fn iter(&self) -> impl Iterator<Item = &Position0Asset> {
        self.non_expert
            .iter()
            .chain(&self.router)
            .chain(&self.shared)
            .chain(&self.routed)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0Capture {
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub input_path: PathBuf,
    pub input_bytes: u64,
    pub input_sha256: String,
    pub k1_moe_output_path: PathBuf,
    pub k1_moe_output_bytes: u64,
    pub k1_moe_output_sha256: String,
    pub payload_count: u64,
    pub payload_bytes: u64,
    pub payload_identity_sha256: String,
    pub verified_before_compute_in_source_run: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0Reference {
    pub moe_branch_f32_le_sha256: String,
    pub layer_output_f32_le_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0Layer {
    pub layer: u8,
    pub compress_ratio: u16,
    pub route_source: String,
    pub expert_ids: Vec<u16>,
    pub route_weights: Vec<f32>,
    pub reference: Position0Reference,
    pub capture: Position0Capture,
    pub assets: Position0LayerAssets,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0Final {
    pub assets: Vec<Position0Asset>,
    pub expected_output_token_id: u32,
    pub normalized_f32_le_sha256: String,
    pub backend: String,
    pub gpu_head_sha256: String,
    pub max_logit: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Position0Summary {
    pub layer_count: u64,
    pub asset_reference_count: u64,
    pub asset_unique_count: u64,
    pub asset_bytes: u64,
    pub moe_payload_references: u64,
    pub moe_payload_bytes: u64,
    pub capture_manifest_count: u64,
    pub capture_manifest_chain_sha256: String,
    pub capture_input_bytes: u64,
    pub capture_k1_output_bytes: u64,
    pub payloads_rehashed_by_builder: u64,
    pub payload_bytes_rehashed_by_builder: u64,
    pub missing_assets: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Position0ManifestError {
    Io(String),
    Json(String),
    Contract(String),
}

impl fmt::Display for Position0ManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Position0ManifestError {}

impl Position0WholeTokenManifest {
    pub fn load(path: &Path) -> Result<Self, Position0ManifestError> {
        let payload = fs::read(path).map_err(|error| {
            Position0ManifestError::Io(format!("读取 {} 失败: {error}", path.display()))
        })?;
        let manifest: Self = serde_json::from_slice(&payload).map_err(|error| {
            Position0ManifestError::Json(format!("解析 {} 失败: {error}", path.display()))
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn all_assets(&self) -> impl Iterator<Item = &Position0Asset> {
        std::iter::once(&self.embedding_row)
            .chain(self.layers.iter().flat_map(|layer| layer.assets.iter()))
            .chain(self.final_section.assets.iter())
    }

    pub fn validate(&self) -> Result<(), Position0ManifestError> {
        require(
            self.format == POSITION0_MANIFEST_FORMAT,
            "manifest format 漂移",
        )?;
        require(self.repo == MODEL_REPO, "模型仓库漂移")?;
        require(self.revision == MODEL_REVISION, "模型 revision 漂移")?;
        require(
            self.profile == POSITION0_PROFILE,
            "profile 不是 FullDepth43/native-top6",
        )?;
        require(
            self.position == 0,
            "position0 manifest 的 position 必须为 0",
        )?;
        require(self.input_token_id == 0, "position0 输入必须为 BOS=0")?;
        require(
            self.expected_output_token_id < VOCAB_SIZE,
            "manifest 最终 token 越界",
        )?;
        require(
            self.final_section.expected_output_token_id == self.expected_output_token_id,
            "manifest 与 final 的 token 不一致",
        )?;
        require(
            self.final_section.max_logit.is_finite(),
            "final max_logit 非有限",
        )?;
        require(
            self.layers.len() == FULL_DEPTH_LAYERS.len(),
            "manifest 不是 43 层",
        )?;
        require(
            self.summary.layer_count == FULL_DEPTH_LAYERS.len() as u64
                && self.summary.capture_manifest_count == FULL_DEPTH_LAYERS.len() as u64,
            "summary 层数/capture 数不闭合",
        )?;
        require(
            self.summary.missing_assets == 0,
            "manifest 声明存在缺失资产",
        )?;
        require(
            normalized_path(&self.source_report.path).contains(POSITION0_SOURCE_RUN)
                && self
                    .source_report
                    .path
                    .file_name()
                    .is_some_and(|name| name == "model_report.json"),
            "source report 不是冻结的同次 K4 运行",
        )?;
        require(
            self.source_report.sha256 == POSITION0_SOURCE_REPORT_SHA256
                && self.source_report.format == "polaris-fulldepth43-native-top6-reference-v1"
                && self.source_report.status == "complete"
                && self.source_report.position0_state_committed,
            "source report 身份/完成状态漂移",
        )?;
        require(
            self.catalog.sha256 == POSITION0_CATALOG_SHA256,
            "FullDepth43 catalog SHA 漂移",
        )?;
        require(
            self.verification_policy.builder_payload_mode == "range_proof_metadata_only"
                && self.verification_policy.range_proof_metadata_checked
                && self.verification_policy.capture_and_report_files_rehashed
                && self
                    .verification_policy
                    .runtime_must_verify_payload_sha256_before_compute
                && self
                    .verification_policy
                    .source_run_rust_receipts_are_historical_evidence_only
                && self.verification_policy.fail_closed,
            "manifest verification policy 被放宽",
        )?;

        validate_asset(&self.embedding_row)?;
        require(
            self.embedding_row.tensor == "embed.weight[0:1]"
                && self.embedding_row.dtype == "BF16"
                && self.embedding_row.shape == [1, 4096]
                && self.embedding_row.bytes == 8192,
            "BOS embedding row 合同漂移",
        )?;

        for (entry, &expected_layer) in self.layers.iter().zip(FULL_DEPTH_LAYERS.iter()) {
            require(entry.layer == expected_layer, "层序不是严格 L0→L42")?;
            require(
                entry.compress_ratio == COMPRESS_RATIOS[entry.layer as usize],
                "层 compressor ratio 漂移",
            )?;
            require(
                entry.expert_ids.len() == EXPERTS_PER_TOKEN
                    && entry.route_weights.len() == EXPERTS_PER_TOKEN,
                "层 route 不是严格 top-6",
            )?;
            let mut ids = HashSet::new();
            for &expert in &entry.expert_ids {
                require(expert < N_ROUTED_EXPERTS, "层 route expert 越界")?;
                require(ids.insert(expert), "层 route expert 重复")?;
            }
            let mut route_sum = 0.0f32;
            for &weight in &entry.route_weights {
                require(weight.is_finite() && weight >= 0.0, "层 route weight 非法")?;
                route_sum += weight;
            }
            require(
                (route_sum - 1.5).abs() <= 2.0e-6,
                "层 route weight 和不为 1.5",
            )?;
            let expected_source = if entry.layer < 3 {
                "current_token_tid2eid_physical_i64"
            } else {
                "sqrtsoftplus_plus_bias_top6"
            };
            require(
                entry.route_source == expected_source,
                "层 route source 漂移",
            )?;
            require(
                entry.capture.input_bytes == 4096 * 4
                    && entry.capture.k1_moe_output_bytes == 4096 * 2
                    && entry.capture.verified_before_compute_in_source_run,
                "层 capture shape/来源验证漂移",
            )?;
            let expected_layer_component = format!("layer-{:02}", entry.layer);
            let capture_path = normalized_path(&entry.capture.manifest_path);
            require(
                capture_path.contains(POSITION0_SOURCE_RUN)
                    && capture_path.contains("position-000000")
                    && capture_path.contains(&expected_layer_component),
                "层 capture 不是冻结的同次 position0 运行",
            )?;
            validate_hash(&entry.reference.moe_branch_f32_le_sha256)?;
            validate_hash(&entry.reference.layer_output_f32_le_sha256)?;
            validate_hash(&entry.capture.manifest_sha256)?;
            validate_hash(&entry.capture.input_sha256)?;
            validate_hash(&entry.capture.k1_moe_output_sha256)?;
            validate_hash(&entry.capture.payload_identity_sha256)?;
            for asset in entry.assets.iter() {
                validate_asset(asset)?;
            }
            validate_layer_asset_sets(entry)?;
        }
        for asset in &self.final_section.assets {
            validate_asset(asset)?;
        }
        validate_final_asset_set(&self.final_section.assets)?;
        validate_hash(&self.final_section.normalized_f32_le_sha256)?;
        validate_hash(&self.final_section.gpu_head_sha256)?;
        validate_hash(&self.summary.capture_manifest_chain_sha256)?;
        require(
            self.summary.capture_manifest_chain_sha256 == POSITION0_CAPTURE_CHAIN_SHA256,
            "capture manifest chain SHA 漂移",
        )?;

        let assets: Vec<&Position0Asset> = self.all_assets().collect();
        let mut identities = HashSet::with_capacity(assets.len());
        let mut total_bytes = 0u64;
        for asset in &assets {
            let identity = (
                asset.range_key.as_str(),
                asset.path.as_path(),
                asset.bytes,
                asset.sha256.as_str(),
            );
            require(identities.insert(identity), "manifest 含重复资产身份")?;
            total_bytes = total_bytes
                .checked_add(asset.bytes)
                .ok_or_else(|| contract("资产字节账本溢出"))?;
        }
        require(
            self.summary.asset_reference_count == assets.len() as u64
                && self.summary.asset_unique_count == identities.len() as u64
                && self.summary.asset_bytes == total_bytes,
            "资产 count/bytes summary 不闭合",
        )?;

        let moe_assets: Vec<&Position0Asset> = self
            .layers
            .iter()
            .flat_map(|layer| layer.assets.shared.iter().chain(&layer.assets.routed))
            .collect();
        let moe_bytes = moe_assets.iter().try_fold(0u64, |sum, asset| {
            sum.checked_add(asset.bytes)
                .ok_or_else(|| contract("MoE 字节账本溢出"))
        })?;
        require(
            self.summary.moe_payload_references == moe_assets.len() as u64
                && self.summary.moe_payload_bytes == moe_bytes,
            "MoE count/bytes summary 不闭合",
        )?;
        require(
            self.summary.capture_input_bytes == self.layers.len() as u64 * 4096 * 4
                && self.summary.capture_k1_output_bytes == self.layers.len() as u64 * 4096 * 2,
            "capture 字节账本不闭合",
        )?;
        require(!self.runtime_contract.is_empty(), "runtime contract 为空")?;
        require(!self.claim_limit.is_empty(), "claim limit 为空")?;
        Ok(())
    }
}

fn validate_layer_asset_sets(entry: &Position0Layer) -> Result<(), Position0ManifestError> {
    let prefix = format!("layers.{}.", entry.layer);
    let mut expected_non_expert = BASE_NON_EXPERT_SUFFIXES
        .iter()
        .map(|suffix| format!("{prefix}{suffix}"))
        .collect::<Vec<_>>();
    match entry.compress_ratio {
        0 => require(entry.layer < 2, "ratio0 只允许 L0/L1")?,
        4 => {
            expected_non_expert.extend(
                COMPRESSOR_SUFFIXES
                    .iter()
                    .chain(RATIO4_INDEXER_SUFFIXES.iter())
                    .map(|suffix| format!("{prefix}{suffix}")),
            );
        }
        128 => {
            expected_non_expert.extend(
                COMPRESSOR_SUFFIXES
                    .iter()
                    .map(|suffix| format!("{prefix}{suffix}")),
            );
        }
        _ => return Err(contract("层 compressor ratio 未注册")),
    }
    validate_exact_assets(
        &entry.assets.non_expert,
        &expected_non_expert,
        "non_expert",
        None,
        "层 non_expert 精确张量集合漂移",
    )?;

    let router_suffixes = if entry.layer < 3 {
        ["ffn.gate.tid2eid", "ffn.gate.weight"]
    } else {
        ["ffn.gate.bias", "ffn.gate.weight"]
    };
    let expected_router = router_suffixes
        .iter()
        .map(|suffix| format!("{prefix}{suffix}"))
        .collect::<Vec<_>>();
    validate_exact_assets(
        &entry.assets.router,
        &expected_router,
        "router",
        None,
        "层 router 精确张量集合漂移",
    )?;

    let expected_shared = SHARED_SUFFIXES
        .iter()
        .map(|suffix| format!("{prefix}{suffix}"))
        .collect::<Vec<_>>();
    validate_exact_assets(
        &entry.assets.shared,
        &expected_shared,
        "shared",
        None,
        "层 shared 精确张量集合漂移",
    )?;

    let selected: HashSet<u16> = entry.expert_ids.iter().copied().collect();
    let mut expected_routed = Vec::with_capacity(entry.expert_ids.len() * EXPERT_SUFFIXES.len());
    for &expert_id in &entry.expert_ids {
        expected_routed.extend(
            EXPERT_SUFFIXES
                .iter()
                .map(|suffix| format!("{prefix}ffn.experts.{expert_id}.{suffix}")),
        );
    }
    validate_exact_tensor_names(
        &entry.assets.routed,
        &expected_routed,
        "层 routed expert 精确张量集合漂移",
    )?;
    for asset in &entry.assets.routed {
        require(asset.kind == "routed_expert", "routed asset kind 漂移")?;
        let expert_id = asset
            .expert_id
            .ok_or_else(|| contract("routed asset 缺少 expert_id"))?;
        require(
            selected.contains(&expert_id),
            "routed asset expert_id 未被 route 选中",
        )?;
        let expected_for_id = EXPERT_SUFFIXES
            .iter()
            .map(|suffix| format!("{prefix}ffn.experts.{expert_id}.{suffix}"))
            .collect::<HashSet<_>>();
        require(
            expected_for_id.contains(&asset.tensor),
            "routed asset tensor/expert_id 不对齐",
        )?;
    }
    for &expert_id in &entry.expert_ids {
        require(
            entry
                .assets
                .routed
                .iter()
                .filter(|asset| asset.expert_id == Some(expert_id))
                .count()
                == EXPERT_SUFFIXES.len(),
            "每个 routed expert 必须正好六个 tensor",
        )?;
    }
    Ok(())
}

fn validate_final_asset_set(assets: &[Position0Asset]) -> Result<(), Position0ManifestError> {
    let expected = FINAL_TENSORS
        .iter()
        .map(|tensor| (*tensor).to_owned())
        .collect::<Vec<_>>();
    validate_exact_assets(
        assets,
        &expected,
        "boundary",
        None,
        "final 精确张量集合漂移",
    )
}

fn validate_exact_assets(
    assets: &[Position0Asset],
    expected_tensors: &[String],
    expected_kind: &str,
    expected_expert_id: Option<u16>,
    drift_message: &'static str,
) -> Result<(), Position0ManifestError> {
    validate_exact_tensor_names(assets, expected_tensors, drift_message)?;
    for asset in assets {
        require(asset.kind == expected_kind, drift_message)?;
        require(asset.expert_id == expected_expert_id, drift_message)?;
    }
    Ok(())
}

fn validate_exact_tensor_names(
    assets: &[Position0Asset],
    expected_tensors: &[String],
    drift_message: &'static str,
) -> Result<(), Position0ManifestError> {
    let actual = assets
        .iter()
        .map(|asset| asset.tensor.as_str())
        .collect::<HashSet<_>>();
    let expected = expected_tensors
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    require(actual.len() == assets.len(), drift_message)?;
    require(expected.len() == expected_tensors.len(), drift_message)?;
    require(actual == expected, drift_message)
}

fn validate_asset(asset: &Position0Asset) -> Result<(), Position0ManifestError> {
    require(!asset.tensor.is_empty(), "资产 tensor 为空")?;
    require(!asset.kind.is_empty(), "资产 kind 为空")?;
    require(asset.bytes > 0, "资产 bytes 为 0")?;
    require(!asset.shape.is_empty(), "资产 shape 为空")?;
    require(
        asset.shape.iter().all(|&dimension| dimension > 0),
        "资产 shape 含 0",
    )?;
    let dtype_bytes = match asset.dtype.as_str() {
        "BF16" => 2u64,
        "F32" => 4u64,
        "F8_E4M3" | "F8_E8M0" | "I8" => 1u64,
        "I64" => 8u64,
        _ => return Err(contract("资产 dtype 未注册")),
    };
    if dtype_bytes != 0 {
        let expected = asset
            .shape
            .iter()
            .try_fold(dtype_bytes, |bytes, &dimension| {
                bytes
                    .checked_mul(dimension)
                    .ok_or_else(|| contract("资产 shape 字节溢出"))
            })?;
        require(expected == asset.bytes, "资产 shape/dtype/bytes 不闭合")?;
    }
    if let Some(expert) = asset.expert_id {
        require(expert < N_ROUTED_EXPERTS, "资产 expert ID 越界")?;
    }
    require(!asset.range_key.is_empty(), "资产 range_key 为空")?;
    validate_hash(&asset.cache_key)?;
    validate_hash(&asset.sha256)?;
    validate_hash(&asset.proof_sha256)?;
    require(!asset.path.as_os_str().is_empty(), "资产 path 为空")?;
    require(
        !asset.proof_path.as_os_str().is_empty(),
        "资产 proof path 为空",
    )?;
    require(asset.hash_authority == "tofu", "资产 hash authority 漂移")?;
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), Position0ManifestError> {
    require(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "SHA-256 不是 64 位十六进制",
    )
}

fn normalized_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn require(condition: bool, message: &'static str) -> Result<(), Position0ManifestError> {
    if condition {
        Ok(())
    } else {
        Err(contract(message))
    }
}

fn contract(message: &'static str) -> Position0ManifestError {
    Position0ManifestError::Contract(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_manifest() -> Position0WholeTokenManifest {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        Position0WholeTokenManifest::load(&path).unwrap()
    }

    fn assert_contract_contains(result: Result<(), Position0ManifestError>, expected: &str) {
        match result.unwrap_err() {
            Position0ManifestError::Contract(message) => assert!(
                message.contains(expected),
                "expected contract containing {expected:?}, got {message:?}"
            ),
            other => panic!("expected Contract, got {other:?}"),
        }
    }

    fn asset(tensor: &str, bytes: u64, shape: &[u64], dtype: &str) -> Position0Asset {
        Position0Asset {
            tensor: tensor.to_owned(),
            kind: "fixture".to_owned(),
            expert_id: None,
            dtype: dtype.to_owned(),
            shape: shape.to_vec(),
            bytes,
            range_key: "shard:0-1".to_owned(),
            cache_key: "a".repeat(64),
            path: PathBuf::from("fixture.bin"),
            sha256: "b".repeat(64),
            proof_path: PathBuf::from("fixture.json"),
            proof_sha256: "c".repeat(64),
            hash_authority: "tofu".to_owned(),
            payload_rehashed_by_builder: false,
            source: Value::Null,
        }
    }

    #[test]
    fn asset_gate_rejects_shape_and_hash_drift() {
        validate_asset(&asset("x", 8, &[1, 4], "BF16")).unwrap();
        let mut bad = asset("x", 7, &[1, 4], "BF16");
        assert!(matches!(
            validate_asset(&bad),
            Err(Position0ManifestError::Contract(_))
        ));
        bad.bytes = 8;
        bad.sha256 = "not-a-sha".to_owned();
        assert!(matches!(
            validate_asset(&bad),
            Err(Position0ManifestError::Contract(_))
        ));
    }

    #[test]
    fn real_position0_manifest_passes_exact_per_layer_tensor_sets() {
        let manifest = real_manifest();
        assert_eq!(manifest.layers.len(), 43);
        assert_eq!(manifest.layers[0].assets.non_expert.len(), 21);
        assert_eq!(manifest.layers[1].assets.non_expert.len(), 21);
        assert_eq!(manifest.layers[2].assets.non_expert.len(), 32);
        assert_eq!(manifest.layers[3].assets.non_expert.len(), 25);
        assert!(manifest.layers.iter().all(|layer| {
            layer.assets.router.len() == 2
                && layer.assets.shared.len() == 6
                && layer.assets.routed.len() == 36
        }));
        assert_eq!(manifest.final_section.assets.len(), 5);
        manifest.validate().unwrap();
    }

    #[test]
    fn exact_tensor_gate_rejects_ratio_router_shared_expert_and_final_tampering() {
        let baseline = real_manifest();

        let mut ratio0 = baseline.clone();
        ratio0.layers[0].assets.non_expert[0].tensor = "layers.0.attn.compressor.ape".to_owned();
        assert_contract_contains(ratio0.validate(), "non_expert 精确张量集合");

        let mut ratio4 = baseline.clone();
        let indexer = ratio4.layers[2]
            .assets
            .non_expert
            .iter_mut()
            .find(|asset| asset.tensor == "layers.2.attn.indexer.compressor.ape")
            .unwrap();
        indexer.tensor = "layers.2.attn.compressor.ape".to_owned();
        assert_contract_contains(ratio4.validate(), "non_expert 精确张量集合");

        let mut ratio128 = baseline.clone();
        let compressor = ratio128.layers[3]
            .assets
            .non_expert
            .iter_mut()
            .find(|asset| asset.tensor == "layers.3.attn.compressor.ape")
            .unwrap();
        compressor.tensor = "layers.3.attn.indexer.compressor.ape".to_owned();
        assert_contract_contains(ratio128.validate(), "non_expert 精确张量集合");

        let mut router = baseline.clone();
        router.layers[2].assets.router[0].tensor = "layers.2.ffn.gate.bias".to_owned();
        assert_contract_contains(router.validate(), "router 精确张量集合");

        let mut shared = baseline.clone();
        shared.layers[4].assets.shared[0].tensor =
            "layers.4.ffn.shared_experts.w1.unregistered".to_owned();
        assert_contract_contains(shared.validate(), "shared 精确张量集合");

        let mut routed = baseline.clone();
        let original = routed.layers[42].assets.routed[0].expert_id.unwrap();
        let replacement = routed.layers[42]
            .expert_ids
            .iter()
            .copied()
            .find(|expert_id| *expert_id != original)
            .unwrap();
        routed.layers[42].assets.routed[0].expert_id = Some(replacement);
        assert_contract_contains(routed.validate(), "tensor/expert_id 不对齐");

        let mut final_section = baseline;
        let head_fn = final_section
            .final_section
            .assets
            .iter_mut()
            .find(|asset| asset.tensor == "hc_head_fn")
            .unwrap();
        head_fn.tensor = "hc_head_fn_missing".to_owned();
        assert_contract_contains(final_section.validate(), "final 精确张量集合");
    }
}
