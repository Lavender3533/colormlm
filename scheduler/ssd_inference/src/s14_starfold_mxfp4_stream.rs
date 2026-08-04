//! S14 StarFold packed-row MXFP4 的 proof-bound 流式组包。
//!
//! 每个 tile 分别从 weight/scale 的权威 Range 取完整行，再固定拼成
//! `[weight rows][scale rows]`。这里只借用一个受启动合同约束的 packed supertile，
//! 不恢复整专家或旧 GiB union bank。

use crate::{
    s14_starfold_cache::{StarfoldMicrotileSpan, StarfoldPageKey},
    s14_starfold_expert_schedule::{S14StarfoldExpertProjection, S14StarfoldExpertTileWork},
    s14_starfold_mxfp4_tile::{S14StarfoldMxfp4ScaleAudit, S14StarfoldMxfp4TileShape},
    s14_starfold_runtime::{
        S14StarfoldB4LayerPlan, S14StarfoldMicrotileSource, S14StarfoldRuntime,
        S14StarfoldVerifiedMicrotile,
    },
};
use anyhow::{bail, Context, Result};
use std::sync::Arc;

#[derive(Debug)]
pub struct S14StarfoldPackedMxfp4Tile {
    pub expert_id: u16,
    pub projection: S14StarfoldExpertProjection,
    pub shape: S14StarfoldMxfp4TileShape,
    pub tile_index: u32,
    proof: Arc<S14StarfoldVerifiedMicrotile>,
    scale_audit: S14StarfoldMxfp4ScaleAudit,
}

impl S14StarfoldPackedMxfp4Tile {
    pub fn proof(&self) -> &Arc<S14StarfoldVerifiedMicrotile> {
        &self.proof
    }

    pub fn into_proof(self) -> Arc<S14StarfoldVerifiedMicrotile> {
        self.proof
    }

    pub const fn scale_audit(&self) -> S14StarfoldMxfp4ScaleAudit {
        self.scale_audit
    }
}

/// 从完整 Range proof 构造一个整行 tile。weight/scale 各自校验后才允许组包；
/// 组包后立即扫描 UE8M0 `0xff`，其 audit 与 proof 一起交给 Vulkan compute owner。
pub fn materialize_packed_mxfp4_tile(
    runtime: &mut S14StarfoldRuntime,
    layer_plan: &S14StarfoldB4LayerPlan,
    expert_id: u16,
    work: &S14StarfoldExpertTileWork,
) -> Result<S14StarfoldPackedMxfp4Tile> {
    materialize_packed_mxfp4_tile_from_projection(
        runtime,
        layer_plan,
        expert_id,
        work,
        work.projection,
    )
}

/// Materialize a logical projection from an explicitly bound source
/// projection.  Production normally passes the same projection.  The only
/// supported storage-resilience twin is W3 <- W1: both share the exact (n,k)
/// MXFP4 layout, while packet identity records the W1 proof source.
pub fn materialize_packed_mxfp4_tile_from_projection(
    runtime: &mut S14StarfoldRuntime,
    layer_plan: &S14StarfoldB4LayerPlan,
    expert_id: u16,
    work: &S14StarfoldExpertTileWork,
    source_projection: S14StarfoldExpertProjection,
) -> Result<S14StarfoldPackedMxfp4Tile> {
    let window_bytes = runtime.contract().microtile_bytes;
    let shape = work.projection.shape(window_bytes)?;
    let source_shape = source_projection.shape(window_bytes)?;
    if source_shape != shape
        || (source_projection != work.projection
            && !(work.projection == S14StarfoldExpertProjection::W3
                && source_projection == S14StarfoldExpertProjection::W1))
    {
        bail!("S14 StarFold source projection 不属于受支持的同形 twin 合同");
    }
    let expected = shape.tile(work.tile.tile_index)?;
    if expected != work.tile {
        bail!("S14 StarFold packed tile work 与 shape 计算结果漂移");
    }
    let weight_template =
        source_template(layer_plan, expert_id, source_projection.weight_segment())?;
    let scale_template = source_template(layer_plan, expert_id, source_projection.scale_segment())?;

    let weight_offset = u64::from(expected.row_base)
        .checked_mul(shape.packed_weight_row_bytes())
        .context("S14 StarFold weight tile offset overflow")?;
    let scale_offset = u64::from(expected.row_base)
        .checked_mul(shape.scale_row_bytes())
        .context("S14 StarFold scale tile offset overflow")?;
    let weight = dynamic_source(
        weight_template,
        expected.tile_index,
        weight_offset,
        expected.weight_bytes,
    )?;
    let scale = dynamic_source(
        scale_template,
        expected.tile_index,
        scale_offset,
        expected.scale_bytes,
    )?;

    let weight_proof = runtime.verify_microtile(&weight)?;
    let scale_proof = runtime.verify_microtile(&scale)?;
    let packed = runtime.pack_verified_mxfp4(weight_proof, scale_proof)?;
    let packed_layout = packed
        .packed_mxfp4()
        .context("S14 StarFold packer 未返回 PackedMxfp4 proof")?
        .layout();
    if packed_layout.weight_bytes != expected.weight_bytes
        || packed_layout.scale_bytes != expected.scale_bytes
        || packed_layout.total_bytes != expected.payload_bytes
        || packed_layout.scale_offset != expected.weight_bytes
    {
        bail!("S14 StarFold packed proof layout 与 tile ABI 漂移");
    }
    let scale_audit =
        S14StarfoldMxfp4ScaleAudit::scan_host_payload(shape, expected.tile_index, packed.bytes())?;
    Ok(S14StarfoldPackedMxfp4Tile {
        expert_id,
        projection: work.projection,
        shape,
        tile_index: expected.tile_index,
        proof: packed,
        scale_audit,
    })
}

fn source_template(
    plan: &S14StarfoldB4LayerPlan,
    expert_id: u16,
    segment: crate::s14_starfold_cache::StarfoldTensorSegment,
) -> Result<&S14StarfoldMicrotileSource> {
    plan.microtile_sources()
        .iter()
        .find(|source| source.span.key.expert_id == expert_id && source.span.key.segment == segment)
        .with_context(|| {
            format!(
                "S14 StarFold 缺少 expert={expert_id} segment={segment:?} 的 Range proof template"
            )
        })
}

fn dynamic_source(
    template: &S14StarfoldMicrotileSource,
    tile_index: u32,
    source_segment_offset: u64,
    byte_len: u64,
) -> Result<S14StarfoldMicrotileSource> {
    let byte_len = u32::try_from(byte_len).context("S14 StarFold packed source bytes 超出 u32")?;
    let end = source_segment_offset
        .checked_add(u64::from(byte_len))
        .context("S14 StarFold packed source end overflow")?;
    if byte_len == 0 || end > template.planned.bytes {
        bail!("S14 StarFold packed source 越出权威 Range");
    }
    Ok(S14StarfoldMicrotileSource {
        span: StarfoldMicrotileSpan {
            key: StarfoldPageKey {
                layer: template.span.key.layer,
                expert_id: template.span.key.expert_id,
                segment: template.span.key.segment,
                tile_index,
            },
            source_segment_offset,
            // 动态整行 tile 不依赖旧 union arena；这里保留源内单调 offset 作为稳定诊断值。
            union_stream_offset: source_segment_offset,
            byte_len,
        },
        planned: template.planned.clone(),
    })
}
