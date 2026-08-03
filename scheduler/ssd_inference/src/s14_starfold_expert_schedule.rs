//! S14 StarFold B4 的专家中心执行计划。
//!
//! 计划把 lane-major top-6 route 改写为 expert-major 物理程序，但不裁剪、不近似、
//! 不改 route 顺序或权重。同一专家的 packed MXFP4 tile 只上传一次，随后连续服务
//! 所有命中该专家的 lane；W1/W3、SwiGLU、W2 完成后立即按原 route weight 累加，
//! 因而无需恢复 1.2 GiB K8 union bank。

use crate::{
    s14_starfold_cache::{StarfoldTensorSegment, STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_mxfp4_tile::{S14StarfoldMxfp4TileShape, S14StarfoldMxfp4TileSpec},
    s14_starfold_runtime::S14StarfoldB4LayerPlan,
};
use anyhow::{bail, Context, Result};
use std::collections::BTreeSet;

pub const S14_STARFOLD_HIDDEN: u32 = 4096;
pub const S14_STARFOLD_INTERMEDIATE: u32 = 2048;
pub const S14_STARFOLD_B4_ROUTE_BINDINGS: usize = STARFOLD_B4_LANES * STARFOLD_TOP_K;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldExpertProjection {
    W1,
    W3,
    W2,
}

impl S14StarfoldExpertProjection {
    pub const EXECUTION_ORDER: [Self; 3] = [Self::W1, Self::W3, Self::W2];

    pub const fn weight_segment(self) -> StarfoldTensorSegment {
        match self {
            Self::W1 => StarfoldTensorSegment::W1Weight,
            Self::W3 => StarfoldTensorSegment::W3Weight,
            Self::W2 => StarfoldTensorSegment::W2Weight,
        }
    }

    pub const fn scale_segment(self) -> StarfoldTensorSegment {
        match self {
            Self::W1 => StarfoldTensorSegment::W1Scale,
            Self::W3 => StarfoldTensorSegment::W3Scale,
            Self::W2 => StarfoldTensorSegment::W2Scale,
        }
    }

    pub fn shape(self, window_capacity_bytes: u32) -> Result<S14StarfoldMxfp4TileShape> {
        let (n, k) = match self {
            Self::W1 | Self::W3 => (S14_STARFOLD_INTERMEDIATE, S14_STARFOLD_HIDDEN),
            Self::W2 => (S14_STARFOLD_HIDDEN, S14_STARFOLD_INTERMEDIATE),
        };
        S14StarfoldMxfp4TileShape::new(n, k, window_capacity_bytes)
    }
}

/// 一个原始 lane/rank 对去重专家的精确引用。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct S14StarfoldExpertLaneUse {
    pub lane: u8,
    pub route_rank: u8,
    pub route_weight: f32,
}

/// 一次上传后必须连续服务的所有 lane。`tile` 的全局 row 区间直接写入该投影输出。
#[derive(Clone, Debug, PartialEq)]
pub struct S14StarfoldExpertTileWork {
    pub projection: S14StarfoldExpertProjection,
    pub tile: S14StarfoldMxfp4TileSpec,
}

/// 单一物理专家的完整 W1→W3→SwiGLU→W2 程序。
#[derive(Clone, Debug, PartialEq)]
pub struct S14StarfoldExpertProgram {
    pub expert_id: u16,
    pub lane_mask: u8,
    pub lane_uses: Vec<S14StarfoldExpertLaneUse>,
    /// 保持 W1、W3、W2 投影顺序；每个 tile 在所有 `lane_uses` 上复用后才退休。
    pub tiles: Vec<S14StarfoldExpertTileWork>,
}

impl S14StarfoldExpertProgram {
    pub fn tiles_for(
        &self,
        projection: S14StarfoldExpertProjection,
    ) -> impl Iterator<Item = &S14StarfoldExpertTileWork> {
        self.tiles
            .iter()
            .filter(move |work| work.projection == projection)
    }

    pub fn dispatches_without_reupload(&self) -> usize {
        self.tiles.len() * self.lane_uses.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct S14StarfoldB4ExpertSchedule {
    pub layer: u16,
    pub base_position: u64,
    pub window_capacity_bytes: u32,
    pub experts: Vec<S14StarfoldExpertProgram>,
    pub exact_route_bindings: usize,
    /// packed tile 的物理上传次数；不会乘以命中 lane 数。
    pub packed_uploads: usize,
    /// 同一 resident tile 对不同 lane 的真实 matvec dispatch 次数。
    pub matvec_dispatches: usize,
}

impl S14StarfoldB4ExpertSchedule {
    pub fn build(plan: &S14StarfoldB4LayerPlan, window_capacity_bytes: u32) -> Result<Self> {
        let routes = &plan.authoritative_routes;
        if routes.lanes().len() != STARFOLD_B4_LANES {
            bail!("S14 StarFold 专家计划要求精确 B=4");
        }
        if plan.micro_union.lane_bindings.len() != STARFOLD_B4_LANES {
            bail!("S14 StarFold micro-union lane 数漂移");
        }

        let mut seen_experts = BTreeSet::new();
        let mut experts = Vec::with_capacity(plan.micro_union.experts.len());
        let mut exact_route_bindings = 0usize;
        let mut packed_uploads = 0usize;
        let mut matvec_dispatches = 0usize;

        for (union_index, union_expert) in plan.micro_union.experts.iter().enumerate() {
            if !seen_experts.insert(union_expert.expert_id) {
                bail!("S14 StarFold expert-major 计划出现重复专家");
            }
            let union_index_u8 =
                u8::try_from(union_index).context("S14 StarFold union expert index 超出 u8")?;
            let mut lane_uses = Vec::with_capacity(STARFOLD_B4_LANES);
            let mut observed_lane_mask = 0u8;
            for (lane, bindings) in plan.micro_union.lane_bindings.iter().enumerate() {
                for binding in bindings {
                    if binding.union_expert_index != union_index_u8 {
                        continue;
                    }
                    if binding.expert_id != union_expert.expert_id {
                        bail!("S14 StarFold lane binding 与 union expert identity 漂移");
                    }
                    let route_rank = usize::from(binding.route_rank);
                    let authoritative = routes
                        .lanes()
                        .get(lane)
                        .and_then(|row| row.get(route_rank))
                        .context("S14 StarFold lane/rank 越出权威 route")?;
                    if authoritative.expert_id != binding.expert_id
                        || authoritative.weight.to_bits() != binding.weight.to_bits()
                    {
                        bail!("S14 StarFold 去重计划改写了权威 route ID/weight");
                    }
                    observed_lane_mask |= 1u8 << lane;
                    lane_uses.push(S14StarfoldExpertLaneUse {
                        lane: u8::try_from(lane).context("S14 StarFold lane 超出 u8")?,
                        route_rank: binding.route_rank,
                        route_weight: binding.weight,
                    });
                    exact_route_bindings += 1;
                }
            }
            if lane_uses.is_empty() || observed_lane_mask != union_expert.lane_mask {
                bail!("S14 StarFold expert lane mask 与真实 bindings 不一致");
            }
            lane_uses.sort_by_key(|use_| (use_.lane, use_.route_rank));

            let mut tiles = Vec::new();
            for projection in S14StarfoldExpertProjection::EXECUTION_ORDER {
                let shape = projection.shape(window_capacity_bytes)?;
                for tile_index in 0..shape.tile_count() {
                    tiles.push(S14StarfoldExpertTileWork {
                        projection,
                        tile: shape.tile(tile_index)?,
                    });
                }
            }
            packed_uploads = packed_uploads
                .checked_add(tiles.len())
                .context("S14 StarFold packed upload count overflow")?;
            matvec_dispatches = matvec_dispatches
                .checked_add(
                    tiles
                        .len()
                        .checked_mul(lane_uses.len())
                        .context("S14 StarFold tile×lane dispatch overflow")?,
                )
                .context("S14 StarFold matvec dispatch count overflow")?;
            experts.push(S14StarfoldExpertProgram {
                expert_id: union_expert.expert_id,
                lane_mask: union_expert.lane_mask,
                lane_uses,
                tiles,
            });
        }

        if exact_route_bindings != S14_STARFOLD_B4_ROUTE_BINDINGS {
            bail!(
                "S14 StarFold 必须无损覆盖 {S14_STARFOLD_B4_ROUTE_BINDINGS} 个 route binding，实际 {exact_route_bindings}"
            );
        }
        if experts.is_empty() || experts.len() > S14_STARFOLD_B4_ROUTE_BINDINGS {
            bail!("S14 StarFold unique expert 数非法");
        }
        Ok(Self {
            layer: routes.layer(),
            base_position: routes.base_position(),
            window_capacity_bytes,
            experts,
            exact_route_bindings,
            packed_uploads,
            matvec_dispatches,
        })
    }
}
