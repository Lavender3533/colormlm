use serde::{Deserialize, Serialize};
use std::fmt;

pub const MODEL_REPO: &str = "deepseek-ai/DeepSeek-V4-Flash-0731";
pub const MODEL_REVISION: &str = "7872f01b1d1fe23eabc4c98b48bffcef5a386062";
pub const N_LAYERS: u8 = 43;
pub const HIDDEN_SIZE: u32 = 4096;
pub const HC_STREAMS: u32 = 4;
pub const VOCAB_SIZE: u32 = 129_280;
pub const N_ROUTED_EXPERTS: u16 = 256;
pub const EXPERTS_PER_TOKEN: usize = 6;
pub const SELECTED_LAYERS: [u8; 14] = [0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42];
pub const FULL_DEPTH_LAYERS: [u8; 43] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
];

/// Official `inference/config.json` main-layer compression ratios. MTP entries
/// 43..45 are intentionally outside this array and outside S14.
pub const COMPRESS_RATIOS: [u16; N_LAYERS as usize] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterKind {
    Hash,
    Score,
}

/// The runner intentionally exposes exactly two production graphs. There is
/// no arbitrary layer/top-k constructor and therefore no post-result sweep or
/// quality-unsafe top-1 reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphProfile {
    S14Top6,
    FullDepth43NativeTop6,
}

impl GraphProfile {
    pub fn layers(self) -> &'static [u8] {
        match self {
            Self::S14Top6 => &SELECTED_LAYERS,
            Self::FullDepth43NativeTop6 => &FULL_DEPTH_LAYERS,
        }
    }

    pub fn experts_per_token(self) -> usize {
        EXPERTS_PER_TOKEN
    }

    pub fn profile_capabilities(self) -> &'static [&'static str] {
        match self {
            Self::S14Top6 => &["s14_identity_skip_state_parity"],
            Self::FullDepth43NativeTop6 => &[
                "full_depth43_native_top6_operator_weight_parity",
                "causal_block_k1_k4_k8",
                "atomic_recursive_state_checkpoint_commit",
            ],
        }
    }

    pub fn priority(self) -> &'static str {
        match self {
            Self::S14Top6 => "primary_first_test",
            Self::FullDepth43NativeTop6 => "flagship_quality_verifier",
        }
    }

    pub fn permits_identity_layers(self) -> bool {
        matches!(self, Self::S14Top6)
    }
}

pub fn is_selected_layer(layer: u8) -> bool {
    SELECTED_LAYERS.binary_search(&layer).is_ok()
}

pub fn router_kind_for_layer(layer: u8) -> Result<RouterKind, ContractError> {
    if layer >= N_LAYERS {
        return Err(ContractError(format!("layer {layer} 超出官方 0..42")));
    }
    Ok(if layer < 3 {
        RouterKind::Hash
    } else {
        RouterKind::Score
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct S14Contract {
    pub format: String,
    pub repo: String,
    pub revision: String,
    pub selected_layers: Vec<u8>,
    pub hidden_size: u32,
    pub hc_streams: u32,
    pub vocab_size: u32,
    pub routed_experts: u16,
    pub experts_per_token: usize,
    pub route_scale: String,
    pub skipped_block_semantics: String,
}

impl S14Contract {
    pub fn frozen() -> Self {
        Self {
            format: "polaris-local-s14-interop-v1".into(),
            repo: MODEL_REPO.into(),
            revision: MODEL_REVISION.into(),
            selected_layers: SELECTED_LAYERS.to_vec(),
            hidden_size: HIDDEN_SIZE,
            hc_streams: HC_STREAMS,
            vocab_size: VOCAB_SIZE,
            routed_experts: N_ROUTED_EXPERTS,
            experts_per_token: EXPERTS_PER_TOKEN,
            route_scale: "1.5_exact".into(),
            skipped_block_semantics: "identity_no_state_mutation".into(),
        }
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        let expected = Self::frozen();
        if self != &expected {
            return Err(ContractError("互操作契约与冻结 S14 身份不一致".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError(pub String);

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_contract_matches_shared_json() {
        let root: serde_json::Value = serde_json::from_str(crate::INTEROP_CONTRACT_JSON).unwrap();
        let wire: S14Contract = serde_json::from_value(root["contract"].clone()).unwrap();
        wire.validate().unwrap();
        assert_eq!(COMPRESS_RATIOS.len(), 43);
        assert_eq!(COMPRESS_RATIOS[42], 4);
    }

    #[test]
    fn production_profiles_cannot_express_top1_or_full_depth_skips() {
        let profile = GraphProfile::FullDepth43NativeTop6;
        assert_eq!(profile.layers(), FULL_DEPTH_LAYERS);
        assert_eq!(profile.experts_per_token(), 6);
        assert!(!profile.permits_identity_layers());
        assert!(serde_json::from_str::<GraphProfile>("\"full_depth_top1\"").is_err());
    }
}
