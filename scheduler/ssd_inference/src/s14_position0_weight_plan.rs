//! FullDepth43 position0 的双 bank 权重布局计划。
//!
//! 每层约 247--254 MiB，不能把 43 层同时常驻 8 GiB 显存。本模块把清单中的
//! 当前层真实资产按 256 字节边界装入两个交替复用的 rolling bank；embedding 与
//! final head 放入独立 resident arena。这里只建立严格、可审计的布局，不读取资产、
//! 不上传 GPU，也不产生 token。

use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{Position0Asset, Position0WholeTokenManifest, FULL_DEPTH_LAYERS};
use std::path::PathBuf;

pub const S14_POSITION0_WEIGHT_ALIGNMENT: u64 = 256;
pub const S14_POSITION0_ROLLING_BANKS: usize = 2;
pub const S14_POSITION0_HEAD_CHUNK_ROWS: u64 = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0AssetPlacement {
    pub tensor: String,
    pub kind: String,
    pub expert_id: Option<u16>,
    pub path: PathBuf,
    pub sha256: String,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0LayerWeightPlan {
    pub layer: u8,
    pub bank: usize,
    pub logical_bytes: u64,
    pub used_bytes: u64,
    pub assets: Vec<S14Position0AssetPlacement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0ResidentWeightPlan {
    pub logical_bytes: u64,
    pub used_bytes: u64,
    pub assets: Vec<S14Position0AssetPlacement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0WeightPlan {
    pub layers: Vec<S14Position0LayerWeightPlan>,
    pub rolling_bank_bytes: u64,
    pub rolling_device_bytes: u64,
    pub resident: S14Position0ResidentWeightPlan,
    pub device_weight_bytes: u64,
    pub logical_payload_bytes: u64,
}

/// 速度主线布局：所有 token 都会复用的 attention/router/shared 与小型 final
/// 参数常驻；每层只滚动当前 top-6 routed experts；1.06 GB BF16 vocab head 以两个
/// 4096-row chunk 交替扫描。它避免每 token 重新搬运约 6.7 GB 静态权重。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0HybridWeightPlan {
    pub resident: S14Position0ResidentWeightPlan,
    pub routed_layers: Vec<S14Position0LayerWeightPlan>,
    pub routed_bank_bytes: u64,
    pub routed_device_bytes: u64,
    pub head_weight: S14Position0AssetPlacement,
    pub head_row_bytes: u64,
    pub head_chunk_rows: u64,
    pub head_chunk_bytes: u64,
    pub head_chunk_count: u64,
    pub head_device_bytes: u64,
    pub device_weight_bytes: u64,
    pub logical_payload_bytes: u64,
}

impl S14Position0HybridWeightPlan {
    pub fn build(manifest: &Position0WholeTokenManifest) -> Result<Self> {
        manifest
            .validate()
            .map_err(|error| anyhow!("position0 manifest invalid: {error}"))?;

        let head = manifest
            .final_section
            .assets
            .iter()
            .find(|asset| asset.tensor == "head.weight")
            .ok_or_else(|| anyhow!("position0 final head.weight missing"))?;
        if head.dtype != "BF16" || head.shape.len() != 2 || head.shape[0] == 0 || head.shape[1] == 0
        {
            bail!("position0 final head.weight shape/dtype drift");
        }
        let head_row_bytes = head.shape[1]
            .checked_mul(2)
            .ok_or_else(|| anyhow!("position0 head row bytes overflow"))?;
        if head.bytes
            != head.shape[0]
                .checked_mul(head_row_bytes)
                .ok_or_else(|| anyhow!("position0 head bytes overflow"))?
        {
            bail!("position0 head byte ledger drift");
        }
        let head_chunk_bytes = align_up(
            S14_POSITION0_HEAD_CHUNK_ROWS
                .checked_mul(head_row_bytes)
                .ok_or_else(|| anyhow!("position0 head chunk bytes overflow"))?,
            S14_POSITION0_WEIGHT_ALIGNMENT,
        )?;
        let head_chunk_count = head.shape[0].div_ceil(S14_POSITION0_HEAD_CHUNK_ROWS);
        let head_device_bytes = head_chunk_bytes
            .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
            .ok_or_else(|| anyhow!("position0 head device bytes overflow"))?;
        let head_weight = S14Position0AssetPlacement {
            tensor: head.tensor.clone(),
            kind: head.kind.clone(),
            expert_id: head.expert_id,
            path: head.path.clone(),
            sha256: head.sha256.clone(),
            offset: 0,
            bytes: head.bytes,
        };

        let mut resident_refs = Vec::<&Position0Asset>::new();
        resident_refs.push(&manifest.embedding_row);
        for layer in &manifest.layers {
            resident_refs.extend(&layer.assets.non_expert);
            resident_refs.extend(&layer.assets.router);
            resident_refs.extend(&layer.assets.shared);
        }
        resident_refs.extend(
            manifest
                .final_section
                .assets
                .iter()
                .filter(|asset| asset.tensor != "head.weight"),
        );
        let (resident_assets, resident_logical_bytes, resident_used_bytes) =
            pack_assets(resident_refs.iter().copied())?;

        let mut routed_layers = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
        let mut routed_bank_bytes = 0u64;
        let mut routed_logical_bytes = 0u64;
        for (&expected_layer, layer) in FULL_DEPTH_LAYERS.iter().zip(&manifest.layers) {
            let (assets, logical_bytes, used_bytes) = pack_assets(layer.assets.routed.iter())?;
            routed_bank_bytes = routed_bank_bytes.max(used_bytes);
            routed_logical_bytes = routed_logical_bytes
                .checked_add(logical_bytes)
                .ok_or_else(|| anyhow!("position0 routed logical bytes overflow"))?;
            routed_layers.push(S14Position0LayerWeightPlan {
                layer: expected_layer,
                bank: expected_layer as usize % S14_POSITION0_ROLLING_BANKS,
                logical_bytes,
                used_bytes,
                assets,
            });
        }
        routed_bank_bytes = align_up(routed_bank_bytes, S14_POSITION0_WEIGHT_ALIGNMENT)?;
        let routed_device_bytes = routed_bank_bytes
            .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
            .ok_or_else(|| anyhow!("position0 routed device bytes overflow"))?;
        let logical_payload_bytes = resident_logical_bytes
            .checked_add(routed_logical_bytes)
            .and_then(|bytes| bytes.checked_add(head.bytes))
            .ok_or_else(|| anyhow!("position0 hybrid logical bytes overflow"))?;
        if logical_payload_bytes != manifest.summary.asset_bytes {
            bail!("position0 hybrid logical payload ledger drift");
        }
        let device_weight_bytes = resident_used_bytes
            .checked_add(routed_device_bytes)
            .and_then(|bytes| bytes.checked_add(head_device_bytes))
            .ok_or_else(|| anyhow!("position0 hybrid device bytes overflow"))?;
        let plan = Self {
            resident: S14Position0ResidentWeightPlan {
                logical_bytes: resident_logical_bytes,
                used_bytes: resident_used_bytes,
                assets: resident_assets,
            },
            routed_layers,
            routed_bank_bytes,
            routed_device_bytes,
            head_weight,
            head_row_bytes,
            head_chunk_rows: S14_POSITION0_HEAD_CHUNK_ROWS,
            head_chunk_bytes,
            head_chunk_count,
            head_device_bytes,
            device_weight_bytes,
            logical_payload_bytes,
        };
        plan.validate(manifest)?;
        Ok(plan)
    }

    pub fn validate(&self, manifest: &Position0WholeTokenManifest) -> Result<()> {
        if self.routed_layers.len() != FULL_DEPTH_LAYERS.len()
            || self.routed_bank_bytes == 0
            || self.routed_bank_bytes % S14_POSITION0_WEIGHT_ALIGNMENT != 0
            || self.routed_device_bytes
                != self.routed_bank_bytes * S14_POSITION0_ROLLING_BANKS as u64
            || self.head_chunk_rows != S14_POSITION0_HEAD_CHUNK_ROWS
            || self.head_chunk_bytes % S14_POSITION0_WEIGHT_ALIGNMENT != 0
            || self.head_device_bytes != self.head_chunk_bytes * S14_POSITION0_ROLLING_BANKS as u64
        {
            bail!("position0 hybrid bank/chunk ledger drift");
        }
        for ((&expected_layer, expected), actual) in FULL_DEPTH_LAYERS
            .iter()
            .zip(&manifest.layers)
            .zip(&self.routed_layers)
        {
            if actual.layer != expected_layer
                || actual.bank != expected_layer as usize % S14_POSITION0_ROLLING_BANKS
                || actual.used_bytes > self.routed_bank_bytes
            {
                bail!("position0 hybrid routed layer drift at L{expected_layer}");
            }
            validate_asset_placements(&actual.assets, expected.assets.routed.iter())?;
        }
        let mut resident_expected = Vec::<&Position0Asset>::new();
        resident_expected.push(&manifest.embedding_row);
        for layer in &manifest.layers {
            resident_expected.extend(&layer.assets.non_expert);
            resident_expected.extend(&layer.assets.router);
            resident_expected.extend(&layer.assets.shared);
        }
        resident_expected.extend(
            manifest
                .final_section
                .assets
                .iter()
                .filter(|asset| asset.tensor != "head.weight"),
        );
        validate_asset_placements(&self.resident.assets, resident_expected.iter().copied())?;
        let head = manifest
            .final_section
            .assets
            .iter()
            .find(|asset| asset.tensor == "head.weight")
            .ok_or_else(|| anyhow!("position0 final head.weight missing"))?;
        if self.head_weight.tensor != head.tensor
            || self.head_weight.path != head.path
            || self.head_weight.sha256 != head.sha256
            || self.head_weight.bytes != head.bytes
            || self.logical_payload_bytes != manifest.summary.asset_bytes
            || self.device_weight_bytes
                != self.resident.used_bytes + self.routed_device_bytes + self.head_device_bytes
        {
            bail!("position0 hybrid final/device ledger drift");
        }
        Ok(())
    }
}

impl S14Position0WeightPlan {
    pub fn build(manifest: &Position0WholeTokenManifest) -> Result<Self> {
        manifest
            .validate()
            .map_err(|error| anyhow!("position0 manifest invalid: {error}"))?;

        let mut layers = Vec::with_capacity(FULL_DEPTH_LAYERS.len());
        let mut rolling_bank_bytes = 0u64;
        let mut logical_payload_bytes = 0u64;
        for (&expected_layer, layer) in FULL_DEPTH_LAYERS.iter().zip(&manifest.layers) {
            if layer.layer != expected_layer {
                bail!(
                    "position0 layer order drift: expected L{} actual L{}",
                    expected_layer,
                    layer.layer
                );
            }
            let (assets, logical_bytes, used_bytes) = pack_assets(layer.assets.iter())
                .with_context(|| format!("plan rolling assets for L{expected_layer}"))?;
            rolling_bank_bytes = rolling_bank_bytes.max(used_bytes);
            logical_payload_bytes = logical_payload_bytes
                .checked_add(logical_bytes)
                .ok_or_else(|| anyhow!("position0 logical payload bytes overflow"))?;
            layers.push(S14Position0LayerWeightPlan {
                layer: expected_layer,
                bank: expected_layer as usize % S14_POSITION0_ROLLING_BANKS,
                logical_bytes,
                used_bytes,
                assets,
            });
        }
        rolling_bank_bytes = align_up(rolling_bank_bytes, S14_POSITION0_WEIGHT_ALIGNMENT)?;
        let rolling_device_bytes = rolling_bank_bytes
            .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
            .ok_or_else(|| anyhow!("position0 rolling device bytes overflow"))?;

        let resident_iter =
            std::iter::once(&manifest.embedding_row).chain(manifest.final_section.assets.iter());
        let (resident_assets, resident_logical_bytes, resident_used_bytes) =
            pack_assets(resident_iter).context("plan resident embedding/final assets")?;
        logical_payload_bytes = logical_payload_bytes
            .checked_add(resident_logical_bytes)
            .ok_or_else(|| anyhow!("position0 logical payload bytes overflow"))?;

        if logical_payload_bytes != manifest.summary.asset_bytes {
            bail!(
                "position0 logical payload ledger drift: manifest={} plan={}",
                manifest.summary.asset_bytes,
                logical_payload_bytes
            );
        }
        let device_weight_bytes = rolling_device_bytes
            .checked_add(resident_used_bytes)
            .ok_or_else(|| anyhow!("position0 device weight bytes overflow"))?;
        let plan = Self {
            layers,
            rolling_bank_bytes,
            rolling_device_bytes,
            resident: S14Position0ResidentWeightPlan {
                logical_bytes: resident_logical_bytes,
                used_bytes: resident_used_bytes,
                assets: resident_assets,
            },
            device_weight_bytes,
            logical_payload_bytes,
        };
        plan.validate(manifest)?;
        Ok(plan)
    }

    pub fn validate(&self, manifest: &Position0WholeTokenManifest) -> Result<()> {
        if self.layers.len() != FULL_DEPTH_LAYERS.len() {
            bail!("position0 rolling plan must contain exactly 43 layers");
        }
        if self.rolling_bank_bytes == 0
            || self.rolling_bank_bytes % S14_POSITION0_WEIGHT_ALIGNMENT != 0
        {
            bail!("position0 rolling bank size is not 256-byte aligned");
        }
        if self.rolling_device_bytes
            != self
                .rolling_bank_bytes
                .checked_mul(S14_POSITION0_ROLLING_BANKS as u64)
                .ok_or_else(|| anyhow!("position0 rolling bytes overflow"))?
        {
            bail!("position0 rolling device ledger drift");
        }

        for ((&expected_layer, expected), actual) in FULL_DEPTH_LAYERS
            .iter()
            .zip(&manifest.layers)
            .zip(&self.layers)
        {
            if actual.layer != expected_layer
                || actual.bank != expected_layer as usize % S14_POSITION0_ROLLING_BANKS
                || actual.used_bytes > self.rolling_bank_bytes
            {
                bail!("position0 rolling layer/bank/capacity drift at L{expected_layer}");
            }
            validate_asset_placements(actual.assets.as_slice(), expected.assets.iter())
                .with_context(|| format!("validate rolling placements for L{expected_layer}"))?;
        }
        let resident_expected =
            std::iter::once(&manifest.embedding_row).chain(manifest.final_section.assets.iter());
        validate_asset_placements(&self.resident.assets, resident_expected)
            .context("validate resident placements")?;
        if self.resident.used_bytes % S14_POSITION0_WEIGHT_ALIGNMENT != 0
            || self.device_weight_bytes
                != self
                    .rolling_device_bytes
                    .checked_add(self.resident.used_bytes)
                    .ok_or_else(|| anyhow!("position0 device bytes overflow"))?
            || self.logical_payload_bytes != manifest.summary.asset_bytes
        {
            bail!("position0 resident/device/logical ledger drift");
        }
        Ok(())
    }
}

fn pack_assets<'a>(
    assets: impl IntoIterator<Item = &'a Position0Asset>,
) -> Result<(Vec<S14Position0AssetPlacement>, u64, u64)> {
    let mut placements = Vec::new();
    let mut logical_bytes = 0u64;
    let mut cursor = 0u64;
    for asset in assets {
        if asset.bytes == 0 {
            bail!("zero-byte position0 asset: {}", asset.tensor);
        }
        cursor = align_up(cursor, S14_POSITION0_WEIGHT_ALIGNMENT)?;
        let end = cursor
            .checked_add(asset.bytes)
            .ok_or_else(|| anyhow!("asset placement overflow: {}", asset.tensor))?;
        logical_bytes = logical_bytes
            .checked_add(asset.bytes)
            .ok_or_else(|| anyhow!("asset logical bytes overflow: {}", asset.tensor))?;
        placements.push(S14Position0AssetPlacement {
            tensor: asset.tensor.clone(),
            kind: asset.kind.clone(),
            expert_id: asset.expert_id,
            path: asset.path.clone(),
            sha256: asset.sha256.clone(),
            offset: cursor,
            bytes: asset.bytes,
        });
        cursor = end;
    }
    Ok((
        placements,
        logical_bytes,
        align_up(cursor, S14_POSITION0_WEIGHT_ALIGNMENT)?,
    ))
}

fn validate_asset_placements<'a>(
    actual: &[S14Position0AssetPlacement],
    expected: impl IntoIterator<Item = &'a Position0Asset>,
) -> Result<()> {
    let expected = expected.into_iter().collect::<Vec<_>>();
    if actual.len() != expected.len() {
        bail!(
            "asset placement count drift: expected={} actual={}",
            expected.len(),
            actual.len()
        );
    }
    let mut previous_end = 0u64;
    for (placement, asset) in actual.iter().zip(expected) {
        if placement.offset % S14_POSITION0_WEIGHT_ALIGNMENT != 0
            || placement.offset < previous_end
            || placement.bytes == 0
            || placement.tensor != asset.tensor
            || placement.kind != asset.kind
            || placement.expert_id != asset.expert_id
            || placement.path != asset.path
            || placement.sha256 != asset.sha256
            || placement.bytes != asset.bytes
        {
            bail!("asset placement drift: {}", asset.tensor);
        }
        previous_end = placement
            .offset
            .checked_add(placement.bytes)
            .ok_or_else(|| anyhow!("asset placement end overflow: {}", asset.tensor))?;
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("alignment must be a non-zero power of two");
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| anyhow!("alignment overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn manifest() -> Position0WholeTokenManifest {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../fast16/research/polaris_meridian_v1/whole_token_runtime/position0_whole_token_manifest.json",
        );
        Position0WholeTokenManifest::load(&path).unwrap()
    }

    #[test]
    fn real_manifest_fits_two_rolling_banks_and_one_resident_arena() {
        let manifest = manifest();
        let plan = S14Position0WeightPlan::build(&manifest).unwrap();
        assert_eq!(plan.layers.len(), 43);
        assert_eq!(plan.layers[0].bank, 0);
        assert_eq!(plan.layers[1].bank, 1);
        assert_eq!(plan.layers[42].bank, 0);
        assert!(plan.rolling_bank_bytes < 256 * 1024 * 1024);
        assert!(plan.rolling_device_bytes < 512 * 1024 * 1024);
        assert!(plan.resident.used_bytes < 1024 * 1024 * 1024 + 64 * 1024);
        assert!(plan.device_weight_bytes < 1536 * 1024 * 1024);
        assert_eq!(plan.logical_payload_bytes, 11_236_196_572);
        assert_eq!(plan.resident.assets.len(), 6);
        plan.validate(&manifest).unwrap();
    }

    #[test]
    fn hybrid_plan_residents_static_weights_and_streams_only_routed_and_head_chunks() {
        let manifest = manifest();
        let plan = S14Position0HybridWeightPlan::build(&manifest).unwrap();
        assert_eq!(plan.routed_layers.len(), 43);
        assert_eq!(plan.routed_layers[0].logical_bytes, 80_216_064);
        assert_eq!(plan.routed_bank_bytes, 80_216_064);
        assert_eq!(plan.routed_device_bytes, 160_432_128);
        assert_eq!(plan.head_row_bytes, 8192);
        assert_eq!(plan.head_chunk_rows, 4096);
        assert_eq!(plan.head_chunk_bytes, 33_554_432);
        assert_eq!(plan.head_chunk_count, 32);
        assert_eq!(plan.head_device_bytes, 67_108_864);
        assert!(plan.resident.used_bytes < 6_800_000_000);
        assert!(plan.device_weight_bytes < 7_000_000_000);
        assert_eq!(plan.logical_payload_bytes, 11_236_196_572);
        plan.validate(&manifest).unwrap();
    }

    #[test]
    fn layout_is_aligned_non_overlapping_and_capacity_bounded() {
        let manifest = manifest();
        let plan = S14Position0WeightPlan::build(&manifest).unwrap();
        for layer in &plan.layers {
            let mut end = 0u64;
            for asset in &layer.assets {
                assert_eq!(asset.offset % S14_POSITION0_WEIGHT_ALIGNMENT, 0);
                assert!(asset.offset >= end);
                end = asset.offset + asset.bytes;
            }
            assert!(end <= layer.used_bytes);
            assert!(layer.used_bytes <= plan.rolling_bank_bytes);
        }
    }

    #[test]
    fn tampered_plan_is_rejected_without_gpu_work() {
        let manifest = manifest();
        let mut plan = S14Position0WeightPlan::build(&manifest).unwrap();
        plan.layers[12].assets[0].offset += 1;
        assert!(plan.validate(&manifest).is_err());

        let mut plan = S14Position0WeightPlan::build(&manifest).unwrap();
        plan.layers[28].bank = 1 - plan.layers[28].bank;
        assert!(plan.validate(&manifest).is_err());

        let mut plan = S14Position0WeightPlan::build(&manifest).unwrap();
        plan.logical_payload_bytes -= 1;
        assert!(plan.validate(&manifest).is_err());
    }

    #[test]
    fn alignment_overflow_is_fail_closed() {
        assert_eq!(align_up(1, 256).unwrap(), 256);
        assert!(align_up(u64::MAX, 256).is_err());
        assert!(align_up(1, 3).is_err());
    }
}
