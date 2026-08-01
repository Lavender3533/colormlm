use crate::{GraphProfile, MODEL_REPO, MODEL_REVISION};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const REQUIRED_CAPABILITIES: [&str; 17] = [
    "fixed_revision_official_graph",
    "loading_ready_fence_publication",
    "verified_route_first_range_provider",
    "native_tokenizer",
    "native_bf16_embedding_row_lookup",
    "mhc_four_stream_sinkhorn",
    "fp8_sparse_attention",
    "compressor_ratio4_overlap_state",
    "compressor_ratio128_state",
    "indexer_fp4_hadamard_top512",
    "hash_router_layers_0_to_2",
    "score_router_sqrtsoftplus_bias_scale_1_5",
    "mxfp4_ue8m0_routed_expert",
    "shared_expert_swiglu_limit_10",
    "native_hc_head_rmsnorm_bf16_lm_head",
    "greedy_full_vocab_argmax",
    "vulkan_official_numerical_parity",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityStatus {
    Missing,
    Passed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    StaticAudit,
    SyntheticTest,
    MeasuredRuntime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEntry {
    pub status: CapabilityStatus,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityManifest {
    pub format: String,
    pub repo: String,
    pub revision: String,
    pub profile: GraphProfile,
    pub selected_layers: Vec<u8>,
    pub experts_per_token: usize,
    pub backend: String,
    pub evidence_kind: EvidenceKind,
    pub native_forward_ready: bool,
    pub capabilities: BTreeMap<String, CapabilityEntry>,
}

impl CapabilityManifest {
    pub fn validate_identity(&self) -> Result<(), GateError> {
        if self.format != "polaris-local-s14-capabilities-v1" {
            return Err(GateError::Identity("capability format 非 v1".into()));
        }
        if self.repo != MODEL_REPO || self.revision != MODEL_REVISION {
            return Err(GateError::Identity("拒绝非冻结 repo/revision".into()));
        }
        if self.selected_layers.as_slice() != self.profile.layers()
            || self.experts_per_token != self.profile.experts_per_token()
        {
            return Err(GateError::Identity(
                "拒绝非预注册 profile 的层集合/top-k".into(),
            ));
        }
        if self.backend != "vulkan" {
            return Err(GateError::Identity(
                "local S14 runner 只接受 vulkan backend".into(),
            ));
        }
        Ok(())
    }

    pub fn missing_capabilities(&self) -> Vec<String> {
        let mut missing: Vec<String> = REQUIRED_CAPABILITIES
            .iter()
            .filter_map(|name| match self.capabilities.get(*name) {
                Some(entry)
                    if entry.status == CapabilityStatus::Passed
                        && !entry.evidence.trim().is_empty() =>
                {
                    None
                }
                Some(entry) if entry.status == CapabilityStatus::Passed => {
                    Some(format!("{name}:passed_without_evidence"))
                }
                _ => Some((*name).to_string()),
            })
            .collect();
        for &profile_capability in self.profile.profile_capabilities() {
            match self.capabilities.get(profile_capability) {
                Some(entry)
                    if entry.status == CapabilityStatus::Passed
                        && !entry.evidence.trim().is_empty() => {}
                Some(entry) if entry.status == CapabilityStatus::Passed => {
                    missing.push(format!("{profile_capability}:passed_without_evidence"));
                }
                _ => missing.push(profile_capability.into()),
            }
        }
        missing
    }

    pub fn gate_production(&self) -> Result<(), GateError> {
        self.validate_identity()?;
        let mut missing = self.missing_capabilities();
        if self.evidence_kind != EvidenceKind::MeasuredRuntime {
            missing.push("evidence_kind:measured_runtime".into());
        }
        if !self.native_forward_ready {
            missing.push("native_forward_ready".into());
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(GateError::Unavailable(missing))
        }
    }

    #[cfg(test)]
    pub(crate) fn synthetic_test_pass() -> Self {
        Self::synthetic_test_pass_for(GraphProfile::S14Top6)
    }

    #[cfg(test)]
    pub(crate) fn synthetic_test_pass_for(profile: GraphProfile) -> Self {
        let mut capabilities: BTreeMap<String, CapabilityEntry> = REQUIRED_CAPABILITIES
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    CapabilityEntry {
                        status: CapabilityStatus::Passed,
                        evidence: "synthetic_lifecycle_test_only".into(),
                    },
                )
            })
            .collect();
        for &profile_capability in profile.profile_capabilities() {
            capabilities.insert(
                profile_capability.into(),
                CapabilityEntry {
                    status: CapabilityStatus::Passed,
                    evidence: "synthetic_identity_graph_test_only".into(),
                },
            );
        }
        Self {
            format: "polaris-local-s14-capabilities-v1".into(),
            repo: MODEL_REPO.into(),
            revision: MODEL_REVISION.into(),
            profile,
            selected_layers: profile.layers().to_vec(),
            experts_per_token: profile.experts_per_token(),
            backend: "vulkan".into(),
            evidence_kind: EvidenceKind::SyntheticTest,
            native_forward_ready: false,
            capabilities,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateError {
    Identity(String),
    Unavailable(Vec<String>),
    Parse(String),
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity(message) => write!(f, "S14 identity gate: {message}"),
            Self::Unavailable(missing) => {
                write!(f, "S14 native forward unavailable: {}", missing.join(", "))
            }
            Self::Parse(message) => write!(f, "capability manifest parse error: {message}"),
        }
    }
}

impl std::error::Error for GateError {}

impl From<serde_json::Error> for GateError {
    fn from(value: serde_json::Error) -> Self {
        Self::Parse(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_vulkan_matrix_hard_refuses() {
        let manifest: CapabilityManifest =
            serde_json::from_str(crate::CURRENT_VULKAN_CAPABILITIES_JSON).unwrap();
        manifest.validate_identity().unwrap();
        let GateError::Unavailable(missing) = manifest.gate_production().unwrap_err() else {
            panic!("expected unavailable gate")
        };
        assert!(missing.iter().any(|x| x == "mhc_four_stream_sinkhorn"));
        assert!(missing.iter().any(|x| x == "native_forward_ready"));
    }

    #[test]
    fn synthetic_evidence_never_passes_production_gate() {
        let manifest = CapabilityManifest::synthetic_test_pass();
        assert!(manifest.gate_production().is_err());
    }

    #[test]
    fn fulldepth_native_top6_profile_is_exact_and_current_matrix_refuses() {
        let manifest: CapabilityManifest =
            serde_json::from_str(crate::CURRENT_FULL_DEPTH_CAPABILITIES_JSON).unwrap();
        manifest.validate_identity().unwrap();
        assert_eq!(manifest.profile, GraphProfile::FullDepth43NativeTop6);
        assert!(manifest
            .missing_capabilities()
            .iter()
            .any(|name| name == "full_depth43_native_top6_operator_weight_parity"));
        assert!(manifest
            .missing_capabilities()
            .iter()
            .any(|name| name == "atomic_recursive_state_checkpoint_commit"));
        assert!(manifest.gate_production().is_err());
    }

    #[test]
    fn deprecated_top1_manifest_cannot_enter_production_gate() {
        assert!(serde_json::from_str::<CapabilityManifest>(
            crate::DEPRECATED_FULL_DEPTH_TOP1_NEGATIVE_CONTRACT_JSON
        )
        .is_err());
    }
}
