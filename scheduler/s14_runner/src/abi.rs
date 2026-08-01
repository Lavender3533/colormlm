use crate::{MODEL_REPO, MODEL_REVISION, N_LAYERS, N_ROUTED_EXPERTS};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

pub const ABI_SAMPLE_BYTES: u64 = 13_377_540;
const EXPERT_PAYLOAD_BYTES: u64 = 13_369_344;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertAbiManifest {
    pub format: String,
    pub repo: String,
    pub revision: String,
    pub layer: u8,
    pub expert_id: u16,
    pub purpose: String,
    pub integrity: String,
    pub header_sha256: String,
    pub payload_bytes: u64,
    pub entries: Vec<AbiEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiEntry {
    pub name: String,
    pub kind: String,
    pub file: String,
    pub start: u64,
    pub end: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub sha256_tofu: String,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidatedExpertAbi {
    pub layer: u8,
    pub expert_id: u16,
    pub expert_payload_bytes: u64,
    pub router_sidecar_bytes: u64,
    /// ABI samples prove byte layout only. They never authorize a route.
    pub routing_authority: bool,
    pub integrity_authority: String,
}

pub fn validate_expert_abi_manifest_json(encoded: &str) -> Result<ValidatedExpertAbi, AbiError> {
    let manifest: ExpertAbiManifest = serde_json::from_str(encoded)
        .map_err(|error| AbiError(format!("manifest JSON: {error}")))?;
    manifest.validate()
}

impl ExpertAbiManifest {
    pub fn validate(&self) -> Result<ValidatedExpertAbi, AbiError> {
        if self.format != "polaris-deepseek-abi-sample-v1"
            || self.repo != MODEL_REPO
            || self.revision != MODEL_REVISION
        {
            return Err(AbiError("ABI sample 身份或格式不匹配".into()));
        }
        if self.layer >= N_LAYERS || self.expert_id >= N_ROUTED_EXPERTS {
            return Err(AbiError("ABI sample layer/expert 超出预注册图范围".into()));
        }
        if self.purpose != "format_and_kernel_abi_only_not_capability" {
            return Err(AbiError("ABI sample purpose 不允许声明路由能力".into()));
        }
        if self.payload_bytes != ABI_SAMPLE_BYTES {
            return Err(AbiError(format!(
                "ABI sample 总字节应为 {ABI_SAMPLE_BYTES}，实际 {}",
                self.payload_bytes
            )));
        }
        let entry_sum: u64 = self.entries.iter().map(|entry| entry.bytes).sum();
        if entry_sum != self.payload_bytes {
            return Err(AbiError("ABI entry 字节和与 payload_bytes 不一致".into()));
        }
        let mut names = BTreeSet::new();
        let mut expert_bytes = 0u64;
        let mut router_bytes = 0u64;
        let mut expert_entries = 0usize;
        for entry in &self.entries {
            if !names.insert(entry.name.as_str()) {
                return Err(AbiError(format!("重复 ABI entry: {}", entry.name)));
            }
            if entry
                .end
                .checked_sub(entry.start)
                .and_then(|span| span.checked_add(1))
                != Some(entry.bytes)
            {
                return Err(AbiError(format!("{} Range/bytes 不闭合", entry.name)));
            }
            if entry.sha256_tofu.len() != 64 {
                return Err(AbiError(format!("{} 缺少 SHA-256", entry.name)));
            }
            match entry.kind.as_str() {
                "expert_tensor" => {
                    expert_entries += 1;
                    expert_bytes += entry.bytes;
                }
                "router_row" | "router_bias" => router_bytes += entry.bytes,
                other => return Err(AbiError(format!("未知 ABI kind: {other}"))),
            }
        }
        if expert_entries != 6 || expert_bytes != EXPERT_PAYLOAD_BYTES {
            return Err(AbiError(format!(
                "专家 ABI 必须是 6 tensors/{EXPERT_PAYLOAD_BYTES} B"
            )));
        }
        if router_bytes != 8_196 {
            return Err(AbiError("router row/bias sidecar 应为 8,196 B".into()));
        }
        Ok(ValidatedExpertAbi {
            layer: self.layer,
            expert_id: self.expert_id,
            expert_payload_bytes: expert_bytes,
            router_sidecar_bytes: router_bytes,
            routing_authority: false,
            integrity_authority: "tofu_fixed_revision_not_authoritative".into(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiError(pub String);

impl fmt::Display for AbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AbiError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn external_l42_e0_manifest_is_compatible_when_present() {
        let path = Path::new("D:/models/Polaris-S14/abi_samples/l42_e0/manifest.json");
        if !path.exists() {
            eprintln!("skip: external ABI sample is absent");
            return;
        }
        let encoded = std::fs::read_to_string(path).unwrap();
        let abi = validate_expert_abi_manifest_json(&encoded).unwrap();
        assert_eq!(abi.layer, 42);
        assert_eq!(abi.expert_id, 0);
        assert_eq!(abi.expert_payload_bytes, EXPERT_PAYLOAD_BYTES);
        assert!(
            !abi.routing_authority,
            "E0 ABI sample must never become a route"
        );
    }
}
