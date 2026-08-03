//! FullDepth43 position0/K=1 的单 arena 工作区布局。
//!
//! 生产后端不得为每个中间张量单独创建 Vulkan allocation。这里把完整层前向所需的
//! 激活、路由与最终 head 临时量固定到一个 256-byte 对齐的 workspace 中；descriptor
//! 只绑定子范围。各槽互不重叠，因此同一 command buffer 内的读写依赖可以由明确的
//! barrier 管理，不能依靠偶然的 allocation 隔离。

use anyhow::{anyhow, bail, Result};
use std::{collections::BTreeMap, ops::Range};

pub const S14_POSITION0_WORKSPACE_ALIGNMENT: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum S14Position0WorkspaceSlot {
    HiddenStreamsA,
    HiddenStreamsB,
    HcNormalizedInput,
    HcBranchBf16,
    HcBranchF32,
    HcAux,
    HcInverseRms,
    QueryLowF32,
    QueryLowBf16,
    QueryF32,
    QueryBf16,
    KeyValueF32,
    KeyValueBf16,
    AttentionBf16,
    AttentionF32,
    GroupedWoAF32,
    AttentionBranchF32,
    AttentionBranchBf16,
    CompressorProjectionF32,
    CompressorScoreF32,
    RouterLogitsF32,
    RouterIdsU32,
    RouterWeightsF32,
    RouterSelectedScoresF32,
    RouterRankingScoresF32,
    ExpertGateF32,
    ExpertUpF32,
    ExpertHiddenF32,
    ExpertDownF32,
    SharedGateF32,
    SharedUpF32,
    SharedHiddenF32,
    SharedDownF32,
    MoeAccumulatorF32,
    FinalHiddenBf16,
    FinalNormalizedBf16,
    FinalNormalizedF32,
    HeadChunkLogitsF32,
    HeadArgmax,
}

impl S14Position0WorkspaceSlot {
    pub const ALL: [Self; 39] = [
        Self::HiddenStreamsA,
        Self::HiddenStreamsB,
        Self::HcNormalizedInput,
        Self::HcBranchBf16,
        Self::HcBranchF32,
        Self::HcAux,
        Self::HcInverseRms,
        Self::QueryLowF32,
        Self::QueryLowBf16,
        Self::QueryF32,
        Self::QueryBf16,
        Self::KeyValueF32,
        Self::KeyValueBf16,
        Self::AttentionBf16,
        Self::AttentionF32,
        Self::GroupedWoAF32,
        Self::AttentionBranchF32,
        Self::AttentionBranchBf16,
        Self::CompressorProjectionF32,
        Self::CompressorScoreF32,
        Self::RouterLogitsF32,
        Self::RouterIdsU32,
        Self::RouterWeightsF32,
        Self::RouterSelectedScoresF32,
        Self::RouterRankingScoresF32,
        Self::ExpertGateF32,
        Self::ExpertUpF32,
        Self::ExpertHiddenF32,
        Self::ExpertDownF32,
        Self::SharedGateF32,
        Self::SharedUpF32,
        Self::SharedHiddenF32,
        Self::SharedDownF32,
        Self::MoeAccumulatorF32,
        Self::FinalHiddenBf16,
        Self::FinalNormalizedBf16,
        Self::FinalNormalizedF32,
        Self::HeadChunkLogitsF32,
        Self::HeadArgmax,
    ];

    fn logical_bytes(self) -> u64 {
        match self {
            // BF16 [4,4096]
            Self::HiddenStreamsA | Self::HiddenStreamsB => 4 * 4096 * 2,
            // HC-pre expands the four BF16 streams to F32 before the split/reduce pass.
            Self::HcNormalizedInput => 4 * 4096 * 4,
            Self::HcBranchBf16 => 4096 * 2,
            Self::HcBranchF32 => 4096 * 4,
            Self::HcAux => 20 * 4,
            Self::HcInverseRms => 4,
            Self::QueryLowF32 => 1024 * 4,
            Self::QueryLowBf16 => 1024 * 2,
            Self::QueryF32 => 64 * 512 * 4,
            Self::QueryBf16 => 64 * 512 * 2,
            Self::KeyValueF32 => 512 * 4,
            Self::KeyValueBf16 => 512 * 2,
            Self::AttentionBf16 => 64 * 512 * 2,
            Self::AttentionF32 => 64 * 512 * 4,
            // 八组各输出 1024，随后展平为官方 [8192] low output。
            Self::GroupedWoAF32 => 8192 * 4,
            Self::AttentionBranchF32 => 4096 * 4,
            Self::AttentionBranchBf16 => 4096 * 2,
            // position0 的 overlap compressor 最大投影为 2*512。
            Self::CompressorProjectionF32 | Self::CompressorScoreF32 => 1024 * 4,
            Self::RouterLogitsF32 | Self::RouterRankingScoresF32 => 256 * 4,
            Self::RouterIdsU32 => 6 * 4,
            Self::RouterWeightsF32 | Self::RouterSelectedScoresF32 => 6 * 4,
            // 六个 routed expert 的 w1/w3 与 w2 输出；一次 grouped dispatch 后按原顺序归约。
            Self::ExpertGateF32 | Self::ExpertUpF32 | Self::ExpertHiddenF32 => 6 * 2048 * 4,
            Self::ExpertDownF32 => 6 * 4096 * 4,
            // shared expert 的中间宽度在当前 checkpoint 为 2048。
            Self::SharedGateF32 | Self::SharedUpF32 | Self::SharedHiddenF32 => 2048 * 4,
            Self::SharedDownF32 | Self::MoeAccumulatorF32 => 4096 * 4,
            Self::FinalHiddenBf16 | Self::FinalNormalizedBf16 => 4096 * 2,
            Self::FinalNormalizedF32 | Self::HeadChunkLogitsF32 => 4096 * 4,
            // token id、logit 与有效标志，预留一个完整 descriptor 对齐块。
            Self::HeadArgmax => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14Position0WorkspaceRegion {
    pub offset: u64,
    pub logical_bytes: u64,
    pub descriptor_bytes: u64,
}

impl S14Position0WorkspaceRegion {
    pub fn range(self) -> Range<u64> {
        self.offset..self.offset + self.logical_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14Position0WorkspaceLayout {
    regions: BTreeMap<S14Position0WorkspaceSlot, S14Position0WorkspaceRegion>,
    used_bytes: u64,
    capacity_bytes: u64,
}

impl S14Position0WorkspaceLayout {
    pub fn build(capacity_bytes: u64) -> Result<Self> {
        if capacity_bytes == 0 {
            bail!("position0 workspace capacity 不能为空");
        }
        let mut regions = BTreeMap::new();
        let mut cursor = 0u64;
        for slot in S14Position0WorkspaceSlot::ALL {
            cursor = align_up(cursor, S14_POSITION0_WORKSPACE_ALIGNMENT)?;
            let logical_bytes = slot.logical_bytes();
            let descriptor_bytes = align_up(logical_bytes, S14_POSITION0_WORKSPACE_ALIGNMENT)?;
            let end = cursor
                .checked_add(descriptor_bytes)
                .ok_or_else(|| anyhow!("position0 workspace layout overflow"))?;
            if end > capacity_bytes {
                bail!(
                    "position0 workspace 不足: slot={slot:?} end={end} capacity={capacity_bytes}"
                );
            }
            let replaced = regions.insert(
                slot,
                S14Position0WorkspaceRegion {
                    offset: cursor,
                    logical_bytes,
                    descriptor_bytes,
                },
            );
            debug_assert!(replaced.is_none());
            cursor = end;
        }
        let layout = Self {
            regions,
            used_bytes: cursor,
            capacity_bytes,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn region(&self, slot: S14Position0WorkspaceSlot) -> S14Position0WorkspaceRegion {
        self.regions[&slot]
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub fn validate(&self) -> Result<()> {
        if self.regions.len() != S14Position0WorkspaceSlot::ALL.len() {
            bail!("position0 workspace slot ledger 不完整");
        }
        let mut ordered = self.regions.values().copied().collect::<Vec<_>>();
        ordered.sort_by_key(|region| region.offset);
        let mut previous_end = 0u64;
        for region in ordered {
            if region.offset % S14_POSITION0_WORKSPACE_ALIGNMENT != 0
                || region.logical_bytes == 0
                || region.descriptor_bytes < region.logical_bytes
                || region.descriptor_bytes % S14_POSITION0_WORKSPACE_ALIGNMENT != 0
                || region.offset < previous_end
            {
                bail!("position0 workspace region 对齐/范围漂移");
            }
            previous_end = region
                .offset
                .checked_add(region.descriptor_bytes)
                .ok_or_else(|| anyhow!("position0 workspace region overflow"))?;
        }
        if previous_end != self.used_bytes || self.used_bytes > self.capacity_bytes {
            bail!("position0 workspace byte ledger 漂移");
        }
        Ok(())
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("workspace alignment 必须是非零二次幂");
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| anyhow!("workspace alignment overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_position0_layout_is_small_aligned_and_complete() {
        let layout = S14Position0WorkspaceLayout::build(512 * 1024 * 1024).unwrap();
        assert!(layout.used_bytes() < 2 * 1024 * 1024);
        assert_eq!(
            layout
                .region(S14Position0WorkspaceSlot::HiddenStreamsA)
                .logical_bytes,
            32_768
        );
        assert_eq!(
            layout
                .region(S14Position0WorkspaceSlot::GroupedWoAF32)
                .logical_bytes,
            32_768
        );
        assert_eq!(
            layout
                .region(S14Position0WorkspaceSlot::ExpertDownF32)
                .logical_bytes,
            98_304
        );
        layout.validate().unwrap();
    }

    #[test]
    fn every_region_is_non_overlapping_and_descriptor_aligned() {
        let layout = S14Position0WorkspaceLayout::build(2 * 1024 * 1024).unwrap();
        let mut ranges = S14Position0WorkspaceSlot::ALL
            .iter()
            .map(|&slot| layout.region(slot))
            .collect::<Vec<_>>();
        ranges.sort_by_key(|region| region.offset);
        for pair in ranges.windows(2) {
            assert!(pair[0].offset + pair[0].descriptor_bytes <= pair[1].offset);
            assert_eq!(pair[1].offset % S14_POSITION0_WORKSPACE_ALIGNMENT, 0);
        }
    }

    #[test]
    fn insufficient_capacity_and_alignment_overflow_fail_closed() {
        assert!(S14Position0WorkspaceLayout::build(4096).is_err());
        assert!(align_up(u64::MAX, S14_POSITION0_WORKSPACE_ALIGNMENT).is_err());
    }
}
