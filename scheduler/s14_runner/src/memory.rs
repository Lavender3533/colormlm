use serde::{Deserialize, Serialize};

pub const EXPERT_PAGE_BYTES: u64 = 13_369_344;
pub const NON_ROUTED_ALL_LAYERS_BYTES: u64 = 2_194_713_552;
pub const NON_ROUTED_MAX_LAYER_BYTES: u64 = 173_503_576;
pub const FULL_DEPTH_NON_ROUTED_BYTES: u64 = 6_727_565_512;
pub const BF16_HEAD_BYTES: u64 = 1_059_061_760;
pub const BF16_EMBEDDING_ROW_BYTES: u64 = 8_192;
pub const FINAL_NORM_BYTES: u64 = 8_192;
pub const NATIVE_STATE_4096_BYTES: u64 = 14_401_536;

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const VRAM_LIMIT: u64 = 8 * GIB;
const RAM_LIMIT: u64 = 32 * GIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetKind {
    CorrectnessColdStreamBf16Head,
    SteadyStateBf16Head,
    SteadyStateFp8HeadCandidate,
    FullDepthTop1Capacity,
    HostRam,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryLine {
    pub name: String,
    pub bytes: u64,
    pub residency: String,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryLedger {
    pub format: String,
    pub kind: BudgetKind,
    pub limit_bytes: u64,
    pub lines: Vec<MemoryLine>,
    pub assigned_bytes: u64,
    pub slack_bytes: u64,
    pub expert_page_slots: Option<u32>,
    pub expert_slots_per_layer_floor: Option<u32>,
    pub measurement_status: String,
    pub runnable_status: String,
}

impl MemoryLedger {
    pub fn correctness_cold_stream() -> Self {
        let mut lines = common_vram_lines(BF16_HEAD_BYTES, "official_header_bf16");
        lines.push(line(
            "current_non_routed_layer_slot",
            NON_ROUTED_MAX_LAYER_BYTES,
            "vram_loading_ready_single_layer",
            "D:/models/Polaris-S14 header sum max=L2",
        ));
        finish_vram(
            BudgetKind::CorrectnessColdStreamBf16Head,
            lines,
            "correctness_only_non_routed_2_194_713_552B_reuploaded_per_token",
        )
    }

    pub fn steady_state_bf16_head() -> Self {
        let mut lines = common_vram_lines(BF16_HEAD_BYTES, "official_header_bf16");
        lines.push(line(
            "all_14_layers_non_routed",
            NON_ROUTED_ALL_LAYERS_BYTES,
            "vram_ready_pinned_layer_addressable",
            "D:/models/Polaris-S14 exact 14-layer header sum",
        ));
        finish_vram(
            BudgetKind::SteadyStateBf16Head,
            lines,
            "capacity_plan_only_target_20_tps_not_measured",
        )
    }

    /// Research-only alternative: FP8 head weights plus one UE8M0 byte per
    /// 64-weight group. It is not official BF16 head semantics until a separate
    /// full-vocab numerical parity artifact exists.
    pub fn steady_state_fp8_head_candidate() -> Self {
        let fp8_weights = 129_280u64 * 4096;
        let ue8m0_scales = 129_280u64 * (4096 / 64);
        let mut lines = common_vram_lines(
            fp8_weights + ue8m0_scales,
            "candidate_fp8_plus_ue8m0_group64_not_implemented",
        );
        lines.push(line(
            "all_14_layers_non_routed",
            NON_ROUTED_ALL_LAYERS_BYTES,
            "vram_ready_pinned_layer_addressable",
            "D:/models/Polaris-S14 exact 14-layer header sum",
        ));
        finish_vram(
            BudgetKind::SteadyStateFp8HeadCandidate,
            lines,
            "not_runnable_requires_quantized_head_full_vocab_parity",
        )
    }

    pub fn host_ram() -> Self {
        let lines = vec![
            line("os_and_driver_reserve", 8 * GIB, "host_reserved", "policy"),
            line(
                "verified_expert_ram_cache_1024_pages",
                1024 * EXPERT_PAGE_BYTES,
                "host_ready_cache",
                "exact pure expert ABI page bytes",
            ),
            line(
                "four_page_upload_staging_ring",
                4 * EXPERT_PAGE_BYTES,
                "host_visible_loading",
                "four independent Vulkan transfer slots",
            ),
            line(
                "boundary_and_non_routed_upload_window",
                256 * MIB,
                "host_streaming",
                "configured capacity",
            ),
            line(
                "verified_metadata_arena",
                64 * MIB,
                "host_ready",
                "raw local metadata is 16,502,290 B; reserve includes parse overhead",
            ),
            line("runtime_heap", GIB, "host", "configured capacity"),
            line(
                "vulkan_host_allocation_reserve",
                2 * GIB,
                "host_reserved",
                "policy",
            ),
            line(
                "filesystem_and_failure_safety",
                4 * GIB,
                "host_reserved",
                "policy",
            ),
        ];
        let assigned_bytes = lines.iter().map(|item| item.bytes).sum();
        Self {
            format: "polaris-s14-memory-ledger-v1".into(),
            kind: BudgetKind::HostRam,
            limit_bytes: RAM_LIMIT,
            lines,
            assigned_bytes,
            slack_bytes: RAM_LIMIT - assigned_bytes,
            expert_page_slots: Some(1024),
            expert_slots_per_layer_floor: Some(1024 / 14),
            measurement_status: "configured_capacity_not_runtime_measurement".into(),
            runnable_status: "provider_capacity_plan".into(),
        }
    }

    /// Pre-registered quality-failure fallback. All 43 non-routed layers and
    /// BF16 head are pinned; only six expert pages fit while retaining 512 MiB
    /// failure reserve. This is a capacity proof, not a 20 tok/s claim: the
    /// real-header static scan is 8,361,509,064 B/token at top-1.
    pub fn full_depth_top1_capacity() -> Self {
        let mut lines = vec![
            line(
                "lm_head",
                BF16_HEAD_BYTES,
                "vram_ready_pinned",
                "official_header_bf16",
            ),
            line(
                "embedding_current_row",
                BF16_EMBEDDING_ROW_BYTES,
                "vram_per_token",
                "row lookup",
            ),
            line(
                "final_norm",
                FINAL_NORM_BYTES,
                "vram_ready_pinned",
                "official BF16",
            ),
            line(
                "all_43_layers_non_routed",
                FULL_DEPTH_NON_ROUTED_BYTES,
                "vram_ready_pinned_layer_addressable",
                "D:/models/Polaris-S14/fulldepth_kadaptive_budget.json",
            ),
            line(
                "native_full_depth_state_4096",
                46_055_424,
                "vram_mutable",
                "43 KV + 41 compressor + 21 indexer containers",
            ),
            line(
                "compute_route_logits_and_transfer_scratch",
                128 * MIB,
                "vram_mutable",
                "configured capacity",
            ),
            line(
                "vram_failure_safety_tight",
                512 * MIB,
                "vram_reserved",
                "tight fallback reserve",
            ),
            line(
                "current_layer_expert_pages",
                6 * EXPERT_PAGE_BYTES,
                "vram_loading_then_fence_published_ready",
                "top1 uses one; capacity also covers one official top6 set",
            ),
        ];
        let assigned_bytes = lines.iter().map(|item| item.bytes).sum();
        Self {
            format: "polaris-s14-memory-ledger-v1".into(),
            kind: BudgetKind::FullDepthTop1Capacity,
            limit_bytes: VRAM_LIMIT,
            lines: std::mem::take(&mut lines),
            assigned_bytes,
            slack_bytes: VRAM_LIMIT - assigned_bytes,
            expert_page_slots: Some(6),
            expert_slots_per_layer_floor: Some(0),
            measurement_status: "real_header_capacity_not_runtime_speed_measurement".into(),
            runnable_status: "hard_reject_until_fulldepth_top1_route_and_operator_parity".into(),
        }
    }

    pub fn validate(&self) -> bool {
        self.lines.iter().map(|line| line.bytes).sum::<u64>() == self.assigned_bytes
            && self.assigned_bytes + self.slack_bytes == self.limit_bytes
            && self.assigned_bytes <= self.limit_bytes
    }
}

fn common_vram_lines(head_bytes: u64, head_evidence: &str) -> Vec<MemoryLine> {
    vec![
        line("lm_head", head_bytes, "vram_ready_pinned", head_evidence),
        line(
            "embedding_current_row",
            BF16_EMBEDDING_ROW_BYTES,
            "vram_per_token",
            "129280x4096 BF16 table is source-only; one 4096 row is uploaded",
        ),
        line(
            "final_norm",
            FINAL_NORM_BYTES,
            "vram_ready_pinned",
            "official BF16",
        ),
        line(
            "native_hc_kv_compressor_indexer_state_4096",
            NATIVE_STATE_4096_BYTES,
            "vram_mutable",
            "Rust state layout with 256-byte slice alignment",
        ),
        line(
            "compute_route_logits_and_transfer_scratch",
            128 * MIB,
            "vram_mutable",
            "configured capacity; host staging is accounted in RAM",
        ),
        line("vram_failure_safety", GIB, "vram_reserved", "policy"),
    ]
}

fn finish_vram(kind: BudgetKind, mut lines: Vec<MemoryLine>, runnable: &str) -> MemoryLedger {
    let fixed: u64 = lines.iter().map(|item| item.bytes).sum();
    let expert_page_slots = ((VRAM_LIMIT - fixed) / EXPERT_PAGE_BYTES) as u32;
    lines.push(line(
        "layer_expert_ready_cache",
        expert_page_slots as u64 * EXPERT_PAGE_BYTES,
        "vram_loading_then_fence_published_ready",
        "key=(layer,expert); pure expert payload excludes router row/bias sidecar",
    ));
    let assigned_bytes = lines.iter().map(|item| item.bytes).sum();
    MemoryLedger {
        format: "polaris-s14-memory-ledger-v1".into(),
        kind,
        limit_bytes: VRAM_LIMIT,
        lines,
        assigned_bytes,
        slack_bytes: VRAM_LIMIT - assigned_bytes,
        expert_page_slots: Some(expert_page_slots),
        expert_slots_per_layer_floor: Some(expert_page_slots / 14),
        measurement_status: "configured_capacity_not_runtime_measurement".into(),
        runnable_status: runnable.into(),
    }
}

fn line(name: &str, bytes: u64, residency: &str, evidence: &str) -> MemoryLine {
    MemoryLine {
        name: name.into(),
        bytes,
        residency: residency.into(),
        evidence: evidence.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ledgers_close_without_theoretical_measurement_claims() {
        let cold = MemoryLedger::correctness_cold_stream();
        let steady = MemoryLedger::steady_state_bf16_head();
        let quant = MemoryLedger::steady_state_fp8_head_candidate();
        let ram = MemoryLedger::host_ram();
        let full = MemoryLedger::full_depth_top1_capacity();
        for ledger in [&cold, &steady, &quant, &ram, &full] {
            assert!(ledger.validate());
            assert!(ledger.measurement_status.contains("not_runtime"));
        }
        assert_eq!(cold.expert_page_slots, Some(458));
        assert_eq!(steady.expert_page_slots, Some(307));
        assert_eq!(steady.expert_slots_per_layer_floor, Some(21));
        assert_eq!(quant.expert_page_slots, Some(346));
        assert_eq!(ram.assigned_bytes, 30_185_357_312);
        assert_eq!(ram.slack_bytes, 4_174_381_056);
        assert_eq!(full.expert_page_slots, Some(6));
        assert_eq!(full.assigned_bytes, 8_584_003_784);
        assert_eq!(full.slack_bytes, 5_930_808);
    }

    #[test]
    fn state_ledger_matches_state_container() {
        let state = crate::NativeState::decode_layout(4096).unwrap();
        assert_eq!(state.arena_bytes, NATIVE_STATE_4096_BYTES);
    }
}
