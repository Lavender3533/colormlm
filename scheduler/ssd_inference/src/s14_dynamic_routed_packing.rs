//! Dynamic FullDepth43 top-6 expert payloads packed for the production ragged ABI.
//!
//! The Range materializer owns proof checking and payload hashing. This module
//! only accepts that verified result, reorders its slot-major W1/W2/W3 source
//! order into the shader ABI order W1/S1/W3/S3/W2/S2, and produces a bounded
//! single-arena layout plus immutable metadata bytes.

use crate::{
    s14_dynamic_routed_page_plan::{
        DynamicRoutedPagePlan, MaterializedDynamicRoutedPagePlan, RoutedProjection,
        DYNAMIC_ROUTED_PAGE_COUNT, DYNAMIC_ROUTED_RANGE_COUNT,
    },
    s14_position0_mapped_assets::VerifiedMappedAsset,
    s14_vulkan::{S14RaggedBranchOffsets, S14RaggedMatvecShape, S14RaggedProjection},
};
use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{Position0Asset, EXPERTS_PER_TOKEN};
use std::{ops::Range, sync::Arc};

pub const S14_DYNAMIC_ROUTED_ALIGNMENT: u64 = 256;
pub const S14_DYNAMIC_ROUTED_HIDDEN: u32 = 4096;
pub const S14_DYNAMIC_ROUTED_INTERMEDIATE: u32 = 2048;
pub const S14_DYNAMIC_ROUTED_WEIGHT_BYTES: u64 = 4_194_304;
pub const S14_DYNAMIC_ROUTED_SCALE_BYTES: u64 = 262_144;
pub const S14_DYNAMIC_ROUTED_SLOT_BYTES: u64 = 13_369_344;
pub const S14_DYNAMIC_ROUTED_ARENA_BYTES: u64 = 80_216_064;
pub const S14_DYNAMIC_ROUTED_METADATA_BYTES: usize = EXPERTS_PER_TOKEN * 6 * 4;
pub const S14_DYNAMIC_ROUTED_ROUTE_IDS_BYTES: usize = EXPERTS_PER_TOKEN * 4;
pub const S14_DYNAMIC_ROUTED_ROUTE_WEIGHTS_BYTES: usize = EXPERTS_PER_TOKEN * 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S14DynamicRoutedRangeRole {
    W1Weight,
    W1Scale,
    W3Weight,
    W3Scale,
    W2Weight,
    W2Scale,
}

impl S14DynamicRoutedRangeRole {
    const ABI_ORDER: [Self; 6] = [
        Self::W1Weight,
        Self::W1Scale,
        Self::W3Weight,
        Self::W3Scale,
        Self::W2Weight,
        Self::W2Scale,
    ];

    const fn projection(self) -> RoutedProjection {
        match self {
            Self::W1Weight | Self::W1Scale => RoutedProjection::W1,
            Self::W3Weight | Self::W3Scale => RoutedProjection::W3,
            Self::W2Weight | Self::W2Scale => RoutedProjection::W2,
        }
    }

    const fn is_weight(self) -> bool {
        matches!(self, Self::W1Weight | Self::W3Weight | Self::W2Weight)
    }

    const fn suffix(self) -> &'static str {
        match self {
            Self::W1Weight => "w1.weight",
            Self::W1Scale => "w1.scale",
            Self::W3Weight => "w3.weight",
            Self::W3Scale => "w3.scale",
            Self::W2Weight => "w2.weight",
            Self::W2Scale => "w2.scale",
        }
    }

    const fn source_projection_index(self) -> usize {
        // MaterializedDynamicRoutedPagePlan is W1, W2, W3; each page is weight, scale.
        match self.projection() {
            RoutedProjection::W1 => 0,
            RoutedProjection::W2 => 1,
            RoutedProjection::W3 => 2,
        }
    }

    const fn source_range_index(self) -> usize {
        self.source_projection_index() * 2 + if self.is_weight() { 0 } else { 1 }
    }

    const fn expected_bytes(self) -> u64 {
        if self.is_weight() {
            S14_DYNAMIC_ROUTED_WEIGHT_BYTES
        } else {
            S14_DYNAMIC_ROUTED_SCALE_BYTES
        }
    }

    fn validate_shape(self, dtype: &str, shape: &[u64]) -> bool {
        match self {
            Self::W1Weight | Self::W3Weight => dtype == "I8" && shape == [2048, 2048],
            Self::W2Weight => dtype == "I8" && shape == [4096, 1024],
            Self::W1Scale | Self::W3Scale => dtype == "F8_E8M0" && shape == [2048, 128],
            Self::W2Scale => dtype == "F8_E8M0" && shape == [4096, 64],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14DynamicRoutedPlacement {
    pub route_slot: usize,
    pub expert_id: u16,
    pub role: S14DynamicRoutedRangeRole,
    /// Index in `MaterializedDynamicRoutedPagePlan::{assets,mapped_assets}`.
    pub source_index: usize,
    pub tensor: String,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14DynamicRoutedSlotMetadata {
    pub route_slot: usize,
    pub expert_id: u16,
    pub route_weight_bits: u32,
    pub offsets: S14RaggedBranchOffsets,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14DynamicRoutedArenaLayout {
    pub layer: u8,
    pub position: u64,
    pub logical_payload_bytes: u64,
    pub arena_logical_bytes: u64,
    /// Strict slot-major W1/S1/W3/S3/W2/S2 order.
    pub placements: Vec<S14DynamicRoutedPlacement>,
    pub slots: [S14DynamicRoutedSlotMetadata; EXPERTS_PER_TOKEN],
}

impl S14DynamicRoutedArenaLayout {
    pub fn build(plan: &DynamicRoutedPagePlan, assets: &[Position0Asset]) -> Result<Self> {
        if plan.pages.len() != DYNAMIC_ROUTED_PAGE_COUNT
            || assets.len() != DYNAMIC_ROUTED_RANGE_COUNT
        {
            bail!("dynamic routed plan/materialized Range count drift");
        }
        let mut placements = Vec::with_capacity(DYNAMIC_ROUTED_RANGE_COUNT);
        let mut cursor = 0u64;
        let mut logical_payload_bytes = 0u64;

        for (slot, (&expert_id, &route_weight)) in plan
            .expert_ids
            .iter()
            .zip(plan.route_weights.iter())
            .enumerate()
        {
            if !route_weight.is_finite() || route_weight < 0.0 {
                bail!("dynamic routed slot {slot} route weight must be finite and non-negative");
            }
            for role in S14DynamicRoutedRangeRole::ABI_ORDER {
                let source_index = slot * 6 + role.source_range_index();
                let asset = assets
                    .get(source_index)
                    .ok_or_else(|| anyhow!("dynamic routed source Range index missing"))?;
                let page = plan
                    .pages
                    .get(slot * 3 + role.source_projection_index())
                    .ok_or_else(|| anyhow!("dynamic routed source page index missing"))?;
                let range = if role.is_weight() {
                    &page.weight
                } else {
                    &page.scale
                };
                let expected_tensor = format!(
                    "layers.{}.ffn.experts.{}.{}",
                    plan.layer,
                    expert_id,
                    role.suffix()
                );
                if page.route_slot != slot
                    || page.expert_id != expert_id
                    || page.projection != role.projection()
                    || range.tensor != expected_tensor
                    || asset.tensor != expected_tensor
                    || asset.kind != "routed_expert"
                    || asset.expert_id != Some(expert_id)
                    || asset.dtype != range.dtype
                    || asset.shape != range.shape
                    || asset.bytes != range.bytes
                    || asset.bytes != role.expected_bytes()
                    || asset.range_key != range.range_key
                    || !role.validate_shape(&asset.dtype, &asset.shape)
                    || !asset.payload_rehashed_by_builder
                {
                    bail!(
                        "dynamic routed slot {slot} {} verified asset ABI drift",
                        role.suffix()
                    );
                }
                cursor = align_up(cursor, S14_DYNAMIC_ROUTED_ALIGNMENT)?;
                let end = cursor
                    .checked_add(asset.bytes)
                    .ok_or_else(|| anyhow!("dynamic routed arena placement overflow"))?;
                if cursor > u32::MAX as u64 || asset.bytes == 0 || asset.bytes % 4 != 0 {
                    bail!("dynamic routed placement is outside ragged u32/four-byte ABI");
                }
                logical_payload_bytes = logical_payload_bytes
                    .checked_add(asset.bytes)
                    .ok_or_else(|| anyhow!("dynamic routed logical byte sum overflow"))?;
                placements.push(S14DynamicRoutedPlacement {
                    route_slot: slot,
                    expert_id,
                    role,
                    source_index,
                    tensor: asset.tensor.clone(),
                    offset: cursor,
                    bytes: asset.bytes,
                });
                cursor = end;
            }
        }

        let arena_logical_bytes = align_up(cursor, S14_DYNAMIC_ROUTED_ALIGNMENT)?;
        if placements.len() != DYNAMIC_ROUTED_RANGE_COUNT
            || arena_logical_bytes == 0
            || arena_logical_bytes > u32::MAX as u64 + 1
        {
            bail!("dynamic routed arena size/count exceeds production bounds");
        }
        let slots: [S14DynamicRoutedSlotMetadata; EXPERTS_PER_TOKEN] = (0..EXPERTS_PER_TOKEN)
            .map(|slot| {
                let rows = &placements[slot * 6..slot * 6 + 6];
                let offset = |index: usize| -> Result<u32> {
                    u32::try_from(rows[index].offset)
                        .context("dynamic routed offset does not fit ragged u32 ABI")
                };
                Ok(S14DynamicRoutedSlotMetadata {
                    route_slot: slot,
                    expert_id: plan.expert_ids[slot],
                    route_weight_bits: plan.route_weights[slot].to_bits(),
                    offsets: S14RaggedBranchOffsets {
                        w1: offset(0)?,
                        s1: offset(1)?,
                        w3: offset(2)?,
                        s3: offset(3)?,
                        w2: offset(4)?,
                        s2: offset(5)?,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow!("dynamic routed slot metadata is not strict top-6"))?;
        let layout = Self {
            layer: plan.layer,
            position: plan.position,
            logical_payload_bytes,
            arena_logical_bytes,
            placements,
            slots,
        };
        layout.validate_ragged_abi()?;
        Ok(layout)
    }

    pub fn ragged_metadata(&self) -> [S14RaggedBranchOffsets; EXPERTS_PER_TOKEN] {
        self.slots.map(|slot| slot.offsets)
    }

    pub fn immutable_bytes(&self) -> S14DynamicRoutedImmutableBytes {
        let mut ragged_metadata_le = Vec::with_capacity(S14_DYNAMIC_ROUTED_METADATA_BYTES);
        let mut route_ids_le = Vec::with_capacity(S14_DYNAMIC_ROUTED_ROUTE_IDS_BYTES);
        let mut route_weights_le = Vec::with_capacity(S14_DYNAMIC_ROUTED_ROUTE_WEIGHTS_BYTES);
        for slot in self.slots {
            for word in slot.offsets.words() {
                ragged_metadata_le.extend_from_slice(&word.to_le_bytes());
            }
            route_ids_le.extend_from_slice(&u32::from(slot.expert_id).to_le_bytes());
            route_weights_le.extend_from_slice(&slot.route_weight_bits.to_le_bytes());
        }
        debug_assert_eq!(ragged_metadata_le.len(), S14_DYNAMIC_ROUTED_METADATA_BYTES);
        debug_assert_eq!(route_ids_le.len(), S14_DYNAMIC_ROUTED_ROUTE_IDS_BYTES);
        debug_assert_eq!(
            route_weights_le.len(),
            S14_DYNAMIC_ROUTED_ROUTE_WEIGHTS_BYTES
        );
        S14DynamicRoutedImmutableBytes {
            ragged_metadata_le,
            route_ids_le,
            route_weights_le,
        }
    }

    fn validate_ragged_abi(&self) -> Result<()> {
        let metadata = self.ragged_metadata();
        for (n, k, projection) in [
            (
                S14_DYNAMIC_ROUTED_INTERMEDIATE,
                S14_DYNAMIC_ROUTED_HIDDEN,
                S14RaggedProjection::W1,
            ),
            (
                S14_DYNAMIC_ROUTED_INTERMEDIATE,
                S14_DYNAMIC_ROUTED_HIDDEN,
                S14RaggedProjection::W3,
            ),
            (
                S14_DYNAMIC_ROUTED_HIDDEN,
                S14_DYNAMIC_ROUTED_INTERMEDIATE,
                S14RaggedProjection::W2,
            ),
        ] {
            S14RaggedMatvecShape::new(
                EXPERTS_PER_TOKEN as u32,
                EXPERTS_PER_TOKEN as u32,
                n,
                k,
                projection,
            )?
            .validate_mxfp4(self.arena_logical_bytes, &metadata)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14DynamicRoutedImmutableBytes {
    pub ragged_metadata_le: Vec<u8>,
    pub route_ids_le: Vec<u8>,
    pub route_weights_le: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14DynamicRoutedImmutableOffsets {
    pub ragged_metadata: u64,
    pub route_ids: u64,
    pub route_weights: u64,
}

impl S14DynamicRoutedImmutableBytes {
    /// Atomically validates all three immutable destinations before writing any bytes.
    pub fn write_into(
        &self,
        target: &mut [u8],
        offsets: S14DynamicRoutedImmutableOffsets,
    ) -> Result<()> {
        if self.ragged_metadata_le.len() != S14_DYNAMIC_ROUTED_METADATA_BYTES
            || self.route_ids_le.len() != S14_DYNAMIC_ROUTED_ROUTE_IDS_BYTES
            || self.route_weights_le.len() != S14_DYNAMIC_ROUTED_ROUTE_WEIGHTS_BYTES
        {
            bail!("dynamic routed immutable byte lengths drift");
        }
        let sections = [
            section_range(
                offsets.ragged_metadata,
                self.ragged_metadata_le.len(),
                target.len(),
            )?,
            section_range(offsets.route_ids, self.route_ids_le.len(), target.len())?,
            section_range(
                offsets.route_weights,
                self.route_weights_le.len(),
                target.len(),
            )?,
        ];
        let mut ordered = sections.clone();
        ordered.sort_unstable_by_key(|range| range.start);
        if ordered.windows(2).any(|pair| pair[0].end > pair[1].start) {
            bail!("dynamic routed immutable sections overlap");
        }
        target[sections[0].clone()].copy_from_slice(&self.ragged_metadata_le);
        target[sections[1].clone()].copy_from_slice(&self.route_ids_le);
        target[sections[2].clone()].copy_from_slice(&self.route_weights_le);
        Ok(())
    }
}

#[derive(Debug)]
pub struct S14DynamicRoutedUploadPlan {
    pub layout: S14DynamicRoutedArenaLayout,
    /// Same ABI order as `layout.placements`.
    mapped_assets: Vec<Arc<VerifiedMappedAsset>>,
}

impl S14DynamicRoutedUploadPlan {
    pub fn build(
        plan: &DynamicRoutedPagePlan,
        materialized: &MaterializedDynamicRoutedPagePlan,
    ) -> Result<Self> {
        if materialized.assets.len() != materialized.mapped_assets.len() {
            bail!("dynamic routed materialized asset/mmap ledger drift");
        }
        let layout = S14DynamicRoutedArenaLayout::build(plan, &materialized.assets)?;
        let mut mapped_assets = Vec::with_capacity(layout.placements.len());
        for placement in &layout.placements {
            let asset = &materialized.assets[placement.source_index];
            let mapped = &materialized.mapped_assets[placement.source_index];
            if mapped.tensor() != asset.tensor
                || mapped.path() != asset.path
                || mapped.bytes().len() as u64 != asset.bytes
                || mapped.expected_sha256() != asset.sha256
            {
                bail!(
                    "dynamic routed verified mmap identity drift: {}",
                    asset.tensor
                );
            }
            mapped_assets.push(Arc::clone(mapped));
        }
        Ok(Self {
            layout,
            mapped_assets,
        })
    }

    pub fn mapped_assets(&self) -> &[Arc<VerifiedMappedAsset>] {
        &self.mapped_assets
    }

    /// Packs verified mmap bytes into one caller-owned host staging arena.
    pub fn stage_into(&self, target: &mut [u8]) -> Result<()> {
        let arena_bytes = usize::try_from(self.layout.arena_logical_bytes)
            .context("dynamic routed arena does not fit host usize")?;
        if target.len() < arena_bytes || self.mapped_assets.len() != self.layout.placements.len() {
            bail!("dynamic routed staging arena capacity/lease count drift");
        }
        for (placement, mapped) in self.layout.placements.iter().zip(&self.mapped_assets) {
            if mapped.tensor() != placement.tensor || mapped.bytes().len() as u64 != placement.bytes
            {
                bail!("dynamic routed mapped lease changed after packing plan build");
            }
        }
        target[..arena_bytes].fill(0);
        for (placement, mapped) in self.layout.placements.iter().zip(&self.mapped_assets) {
            let start = usize::try_from(placement.offset)?;
            let end = start
                .checked_add(mapped.bytes().len())
                .ok_or_else(|| anyhow!("dynamic routed staging range overflow"))?;
            target[start..end].copy_from_slice(mapped.bytes());
        }
        Ok(())
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("dynamic routed alignment must be a non-zero power of two");
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| anyhow!("dynamic routed alignment overflow"))
}

fn section_range(offset: u64, bytes: usize, capacity: usize) -> Result<Range<usize>> {
    if offset % 4 != 0 || bytes == 0 || bytes % 4 != 0 {
        bail!("dynamic routed immutable section is not four-byte aligned");
    }
    let start = usize::try_from(offset)?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| anyhow!("dynamic routed immutable section overflow"))?;
    if end > capacity {
        bail!("dynamic routed immutable section exceeds destination");
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s14_dynamic_routed_page_plan::{DynamicRoutedPage, ExpertRangeIdentity, OnlineTop6};
    use serde_json::Value;
    use std::path::PathBuf;

    const LAYER: u8 = 7;

    fn range(
        slot: usize,
        expert_id: u16,
        projection: RoutedProjection,
        weight: bool,
    ) -> ExpertRangeIdentity {
        let stem = projection.tensor_stem();
        let suffix = if weight { "weight" } else { "scale" };
        let ordinal = slot * 6
            + match projection {
                RoutedProjection::W1 => 0,
                RoutedProjection::W2 => 2,
                RoutedProjection::W3 => 4,
            }
            + usize::from(!weight);
        let (dtype, shape, bytes) = match (projection, weight) {
            (RoutedProjection::W1 | RoutedProjection::W3, true) => {
                ("I8", vec![2048, 2048], S14_DYNAMIC_ROUTED_WEIGHT_BYTES)
            }
            (RoutedProjection::W2, true) => {
                ("I8", vec![4096, 1024], S14_DYNAMIC_ROUTED_WEIGHT_BYTES)
            }
            (RoutedProjection::W1 | RoutedProjection::W3, false) => {
                ("F8_E8M0", vec![2048, 128], S14_DYNAMIC_ROUTED_SCALE_BYTES)
            }
            (RoutedProjection::W2, false) => {
                ("F8_E8M0", vec![4096, 64], S14_DYNAMIC_ROUTED_SCALE_BYTES)
            }
        };
        let start = ordinal as u64 * 16_000_000;
        ExpertRangeIdentity {
            tensor: format!("layers.{LAYER}.ffn.experts.{expert_id}.{stem}.{suffix}"),
            kind: "routed_expert".into(),
            layer: LAYER,
            file: "model-test.safetensors".into(),
            file_bytes: 1_000_000_000,
            header_tensor_table_sha256: "a".repeat(64),
            start,
            end: start + bytes - 1,
            bytes,
            dtype: dtype.into(),
            shape,
            range_key: format!("model-test.safetensors:{start}-{}", start + bytes - 1),
            expert_id,
        }
    }

    fn fixture() -> (DynamicRoutedPagePlan, Vec<Position0Asset>) {
        let route = OnlineTop6 {
            layer: LAYER,
            position: 1,
            expert_ids: [126, 3, 250, 17, 99, 42],
            route_weights: [0.30, 0.25, 0.20, 0.12, 0.08, 0.05],
        };
        let mut pages = Vec::new();
        let mut assets = Vec::new();
        for (slot, &expert_id) in route.expert_ids.iter().enumerate() {
            for projection in [
                RoutedProjection::W1,
                RoutedProjection::W2,
                RoutedProjection::W3,
            ] {
                let weight = range(slot, expert_id, projection, true);
                let scale = range(slot, expert_id, projection, false);
                pages.push(DynamicRoutedPage {
                    route_slot: slot,
                    expert_id,
                    projection,
                    weight: weight.clone(),
                    scale: scale.clone(),
                });
                for source in [&weight, &scale] {
                    assets.push(Position0Asset {
                        tensor: source.tensor.clone(),
                        kind: source.kind.clone(),
                        expert_id: Some(expert_id),
                        dtype: source.dtype.clone(),
                        shape: source.shape.clone(),
                        bytes: source.bytes,
                        range_key: source.range_key.clone(),
                        cache_key: format!("{slot:064x}"),
                        path: PathBuf::from(format!("{}.bin", source.tensor)),
                        sha256: "b".repeat(64),
                        proof_path: PathBuf::from(format!("{}.json", source.tensor)),
                        proof_sha256: "c".repeat(64),
                        hash_authority: "tofu".into(),
                        payload_rehashed_by_builder: true,
                        source: Value::Null,
                    });
                }
            }
        }
        (
            DynamicRoutedPagePlan {
                layer: route.layer,
                position: route.position,
                expert_ids: route.expert_ids,
                route_weights: route.route_weights,
                pages,
            },
            assets,
        )
    }

    #[test]
    fn packs_verified_ranges_in_existing_ragged_abi_order() {
        let (plan, assets) = fixture();
        let layout = S14DynamicRoutedArenaLayout::build(&plan, &assets).unwrap();
        assert_eq!(layout.placements.len(), 36);
        assert_eq!(layout.logical_payload_bytes, S14_DYNAMIC_ROUTED_ARENA_BYTES);
        assert_eq!(layout.arena_logical_bytes, S14_DYNAMIC_ROUTED_ARENA_BYTES);
        assert_eq!(
            layout.placements[..6]
                .iter()
                .map(|placement| placement.role)
                .collect::<Vec<_>>(),
            S14DynamicRoutedRangeRole::ABI_ORDER
        );
        assert_eq!(
            layout.placements[..6]
                .iter()
                .map(|placement| placement.source_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 4, 5, 2, 3]
        );
        assert_eq!(
            layout.slots[0].offsets.words(),
            [0, 4_194_304, 4_456_448, 8_650_752, 8_912_896, 13_107_200]
        );
        assert_eq!(
            layout.slots[1].offsets.w1 as u64,
            S14_DYNAMIC_ROUTED_SLOT_BYTES
        );
    }

    #[test]
    fn emits_little_endian_metadata_ids_and_weights_for_immutable_storage() {
        let (plan, assets) = fixture();
        let layout = S14DynamicRoutedArenaLayout::build(&plan, &assets).unwrap();
        let immutable = layout.immutable_bytes();
        assert_eq!(&immutable.ragged_metadata_le[..4], &0u32.to_le_bytes());
        assert_eq!(&immutable.route_ids_le[..4], &126u32.to_le_bytes());
        assert_eq!(
            &immutable.route_weights_le[..4],
            &plan.route_weights[0].to_bits().to_le_bytes()
        );
        let mut target = vec![0u8; 536];
        immutable
            .write_into(
                &mut target,
                S14DynamicRoutedImmutableOffsets {
                    ragged_metadata: 0,
                    route_ids: 256,
                    route_weights: 512,
                },
            )
            .unwrap();
        assert_eq!(&target[256..260], &126u32.to_le_bytes());
        assert!(immutable
            .write_into(
                &mut target,
                S14DynamicRoutedImmutableOffsets {
                    ragged_metadata: 0,
                    route_ids: 128,
                    route_weights: 512,
                },
            )
            .is_err());
    }

    #[test]
    fn rejects_negative_route_weight_and_asset_shape_drift() {
        let (mut plan, mut assets) = fixture();
        plan.route_weights[2] = -0.1;
        assert!(S14DynamicRoutedArenaLayout::build(&plan, &assets).is_err());
        plan.route_weights[2] = 0.2;
        assets[0].shape = vec![1, 1];
        assert!(S14DynamicRoutedArenaLayout::build(&plan, &assets).is_err());
    }
}
