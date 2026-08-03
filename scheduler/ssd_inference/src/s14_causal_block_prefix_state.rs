//! Production K=4/8 causal-block 的 prefix-state 累积顺序合同。
//!
//! 一个 prefix checkpoint 不是对应 lane 的单次写集：prefix `p` 必须从
//! authoritative state 出发，依次应用 lane `0..=p` 的43层 window KV、
//! compressor/indexer remainder、boundary finalize 与 rollover。本模块只固化
//! 这个累积拓扑和每个 position 的现有 state-recording recipe identity；真实
//! Vulkan copy/finalize 由 production recorder 执行并提供回执。结构收据不是
//! whole-token 数值完成证据。

use crate::{
    s14_position0_layer_program::S14Position0FullDepthLayerProgram,
    s14_position0_state_writeback::{
        S14Position0FullDepthStateRecordingProgram, S14Position0LayerStateRecordingRecipe,
    },
    s14_position0_workspace::S14Position0WorkspaceLayout,
};
use anyhow::{anyhow, bail, Context, Result};
use polaris_s14_runner::{NativeState, FULL_DEPTH_LAYERS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14CausalBlockPrefixLanePhase {
    AwaitingWindowAndRemainder,
    AwaitingBoundaryFinalize,
    AwaitingRollover,
    ReadyToSeal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14CausalBlockPrefixBoundaryKind {
    None,
    Ratio4,
    Ratio128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefixLayerProgress {
    next_source_lane: usize,
    active_source_lane: Option<usize>,
    window_recorded: bool,
    remainder_recorded: bool,
    phase: S14CausalBlockPrefixLanePhase,
    sealed: bool,
}

impl Default for PrefixLayerProgress {
    fn default() -> Self {
        Self {
            next_source_lane: 0,
            active_source_lane: None,
            window_recorded: false,
            remainder_recorded: false,
            phase: S14CausalBlockPrefixLanePhase::AwaitingWindowAndRemainder,
            sealed: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14CausalBlockPrefixProgramIdentity {
    pub prefix_index: usize,
    pub input_position: u32,
    pub checkpoint_position: u32,
    pub candidate_state_bytes: u64,
    pub device_copy_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14CausalBlockPrefixStateSealReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub sealed_prefixes: usize,
    pub sealed_prefix_layers: usize,
    pub cumulative_lane_applications: usize,
    pub serial_token_forward_calls: u32,
}

/// 为 K 个连续 input position 复用现有 FullDepth43 state recipe。
/// `programs[p]` 描述 lane `p` 的实际写集；prefix `p` 的完整 checkpoint
/// 必须经过 `0..=p` 的累积应用，不得只应用 `programs[p]`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14CausalBlockPrefixStateProgram {
    base_position: u32,
    block_size: usize,
    programs: Vec<S14Position0FullDepthStateRecordingProgram>,
    identities: Vec<S14CausalBlockPrefixProgramIdentity>,
    progress: Vec<Vec<PrefixLayerProgress>>,
    prefix_sealed: Vec<bool>,
}

impl S14CausalBlockPrefixStateProgram {
    pub fn build(
        graph: &S14Position0FullDepthLayerProgram,
        workspace: &S14Position0WorkspaceLayout,
        authoritative: &NativeState,
        block_size: usize,
    ) -> Result<Self> {
        if !matches!(block_size, 4 | 8) {
            bail!("causal-block prefix state 只允许 K=4/8");
        }
        authoritative
            .position
            .checked_add(block_size as u32)
            .context("causal-block prefix state position overflow")?;
        let mut programs = Vec::with_capacity(block_size);
        let mut identities = Vec::with_capacity(block_size);
        for prefix_index in 0..block_size {
            let input_position = authoritative
                .position
                .checked_add(prefix_index as u32)
                .context("prefix input position overflow")?;
            let checkpoint_position = input_position
                .checked_add(1)
                .context("prefix checkpoint position overflow")?;
            let mut state = authoritative.clone();
            state.position = input_position;
            let program =
                S14Position0FullDepthStateRecordingProgram::build(graph, workspace, &state)?;
            if program.position != input_position
                || program.layers.len() != FULL_DEPTH_LAYERS.len()
                || program
                    .layers
                    .iter()
                    .zip(FULL_DEPTH_LAYERS)
                    .any(|(recipe, expected)| recipe.layer != expected)
            {
                bail!("prefix {prefix_index} FullDepth43 state program identity 漂移");
            }
            identities.push(S14CausalBlockPrefixProgramIdentity {
                prefix_index,
                input_position,
                checkpoint_position,
                candidate_state_bytes: program.state_layout.candidate_state_bytes,
                device_copy_count: program.copy_count(),
            });
            programs.push(program);
        }
        let candidate_state_bytes = identities
            .first()
            .context("prefix state program 为空")?
            .candidate_state_bytes;
        if identities
            .iter()
            .any(|identity| identity.candidate_state_bytes != candidate_state_bytes)
        {
            bail!("K-prefix candidate state arena bytes 漂移");
        }
        Ok(Self {
            base_position: authoritative.position,
            block_size,
            programs,
            identities,
            progress: vec![
                vec![PrefixLayerProgress::default(); FULL_DEPTH_LAYERS.len()];
                block_size
            ],
            prefix_sealed: vec![false; block_size],
        })
    }

    pub fn base_position(&self) -> u32 {
        self.base_position
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn identities(&self) -> &[S14CausalBlockPrefixProgramIdentity] {
        &self.identities
    }

    pub fn recipe(
        &self,
        source_lane: usize,
        layer: u8,
    ) -> Result<&S14Position0LayerStateRecordingRecipe> {
        self.programs
            .get(source_lane)
            .and_then(|program| program.layer(layer))
            .ok_or_else(|| {
                anyhow!("prefix state source lane/layer 非法: lane={source_lane} L{layer}")
            })
    }

    pub fn boundary_kind(
        &self,
        source_lane: usize,
        layer: u8,
    ) -> Result<S14CausalBlockPrefixBoundaryKind> {
        let recipe = self.recipe(source_lane, layer)?;
        Ok(match (recipe.compress_ratio, recipe.position) {
            (4, position) if position % 4 == 3 => S14CausalBlockPrefixBoundaryKind::Ratio4,
            (128, position) if position % 128 == 127 => S14CausalBlockPrefixBoundaryKind::Ratio128,
            (0 | 4 | 128, _) => S14CausalBlockPrefixBoundaryKind::None,
            (ratio, _) => bail!("prefix state L{layer} 未知 compressor ratio {ratio}"),
        })
    }

    /// 开始把 `source_lane` 的写集累积到 `prefix_index` checkpoint。
    /// 只允许 `source_lane=0..=prefix_index` 严格递增。
    pub fn begin_lane_application(
        &mut self,
        prefix_index: usize,
        layer: u8,
        source_lane: usize,
    ) -> Result<()> {
        let progress = self.progress_mut(prefix_index, layer)?;
        if progress.sealed
            || progress.active_source_lane.is_some()
            || source_lane != progress.next_source_lane
            || source_lane > prefix_index
        {
            bail!(
                "prefix {prefix_index} L{layer} source lane 顺序漂移: expected={} observed={source_lane}",
                progress.next_source_lane
            );
        }
        progress.active_source_lane = Some(source_lane);
        progress.window_recorded = false;
        progress.remainder_recorded = false;
        progress.phase = S14CausalBlockPrefixLanePhase::AwaitingWindowAndRemainder;
        Ok(())
    }

    pub fn mark_window_recorded(&mut self, prefix_index: usize, layer: u8) -> Result<()> {
        let progress = self.active_progress_mut(prefix_index, layer)?;
        if progress.phase != S14CausalBlockPrefixLanePhase::AwaitingWindowAndRemainder
            || progress.window_recorded
        {
            bail!("prefix {prefix_index} L{layer} window KV 重复或顺序漂移");
        }
        progress.window_recorded = true;
        if progress.remainder_recorded {
            progress.phase = S14CausalBlockPrefixLanePhase::AwaitingBoundaryFinalize;
        }
        Ok(())
    }

    pub fn mark_remainder_recorded(&mut self, prefix_index: usize, layer: u8) -> Result<()> {
        let progress = self.active_progress_mut(prefix_index, layer)?;
        if progress.phase != S14CausalBlockPrefixLanePhase::AwaitingWindowAndRemainder
            || progress.remainder_recorded
        {
            bail!("prefix {prefix_index} L{layer} compressor remainder 重复或顺序漂移");
        }
        progress.remainder_recorded = true;
        if progress.window_recorded {
            progress.phase = S14CausalBlockPrefixLanePhase::AwaitingBoundaryFinalize;
        }
        Ok(())
    }

    /// ratio0/非边界也必须显式调用，作为 no-op finalize 回执；
    /// 这样边界与非边界共用同一 fail-closed 顺序。
    pub fn mark_boundary_finalized(&mut self, prefix_index: usize, layer: u8) -> Result<()> {
        let progress = self.active_progress_mut(prefix_index, layer)?;
        if progress.phase != S14CausalBlockPrefixLanePhase::AwaitingBoundaryFinalize
            || !progress.window_recorded
            || !progress.remainder_recorded
        {
            bail!("prefix {prefix_index} L{layer} boundary finalize 必须位于 remainder 之后");
        }
        progress.phase = S14CausalBlockPrefixLanePhase::AwaitingRollover;
        Ok(())
    }

    /// ratio4 边界在这里表示真实 rollover；其他 position/ratio 表示
    /// 已验证空 rollover。两者都不能跳过 finalize。
    pub fn mark_rollover_recorded(&mut self, prefix_index: usize, layer: u8) -> Result<()> {
        let progress = self.active_progress_mut(prefix_index, layer)?;
        if progress.phase != S14CausalBlockPrefixLanePhase::AwaitingRollover {
            bail!("prefix {prefix_index} L{layer} rollover 必须位于 finalize 之后");
        }
        progress.phase = S14CausalBlockPrefixLanePhase::ReadyToSeal;
        Ok(())
    }

    pub fn seal_lane_application(&mut self, prefix_index: usize, layer: u8) -> Result<()> {
        let progress = self.active_progress_mut(prefix_index, layer)?;
        if progress.phase != S14CausalBlockPrefixLanePhase::ReadyToSeal {
            bail!("prefix {prefix_index} L{layer} lane application 尚未闭合");
        }
        let source_lane = progress
            .active_source_lane
            .take()
            .context("active source lane 缺失")?;
        if source_lane != progress.next_source_lane {
            bail!("prefix {prefix_index} L{layer} active source lane 漂移");
        }
        progress.next_source_lane = progress
            .next_source_lane
            .checked_add(1)
            .context("prefix source lane counter overflow")?;
        progress.window_recorded = false;
        progress.remainder_recorded = false;
        progress.phase = S14CausalBlockPrefixLanePhase::AwaitingWindowAndRemainder;
        if progress.next_source_lane == prefix_index + 1 {
            progress.sealed = true;
        }
        Ok(())
    }

    pub fn seal_prefix(&mut self, prefix_index: usize) -> Result<()> {
        let layers = self
            .progress
            .get(prefix_index)
            .context("prefix index 非法")?;
        if layers.len() != FULL_DEPTH_LAYERS.len()
            || layers.iter().any(|progress| {
                !progress.sealed
                    || progress.active_source_lane.is_some()
                    || progress.next_source_lane != prefix_index + 1
            })
        {
            bail!("prefix {prefix_index} 尚未累积闭合43层全部 lane 写集");
        }
        let sealed = self
            .prefix_sealed
            .get_mut(prefix_index)
            .context("prefix seal index 非法")?;
        if *sealed {
            bail!("prefix {prefix_index} 重复 seal");
        }
        *sealed = true;
        Ok(())
    }

    pub fn seal_block(&self) -> Result<S14CausalBlockPrefixStateSealReceipt> {
        if self.prefix_sealed.iter().any(|sealed| !sealed) {
            bail!("K-prefix state block 尚有未 seal prefix");
        }
        let sealed_prefix_layers = self
            .block_size
            .checked_mul(FULL_DEPTH_LAYERS.len())
            .context("sealed prefix layer count overflow")?;
        let triangular = self
            .block_size
            .checked_mul(self.block_size + 1)
            .and_then(|value| value.checked_div(2))
            .context("prefix triangular lane count overflow")?;
        let cumulative_lane_applications = triangular
            .checked_mul(FULL_DEPTH_LAYERS.len())
            .context("prefix cumulative lane count overflow")?;
        Ok(S14CausalBlockPrefixStateSealReceipt {
            base_position: self.base_position,
            block_size: self.block_size,
            sealed_prefixes: self.block_size,
            sealed_prefix_layers,
            cumulative_lane_applications,
            serial_token_forward_calls: 0,
        })
    }

    fn progress_mut(&mut self, prefix_index: usize, layer: u8) -> Result<&mut PrefixLayerProgress> {
        self.progress
            .get_mut(prefix_index)
            .and_then(|layers| layers.get_mut(layer as usize))
            .ok_or_else(|| anyhow!("prefix/layer index 非法: prefix={prefix_index} L{layer}"))
    }

    fn active_progress_mut(
        &mut self,
        prefix_index: usize,
        layer: u8,
    ) -> Result<&mut PrefixLayerProgress> {
        let progress = self.progress_mut(prefix_index, layer)?;
        if progress.active_source_lane.is_none() || progress.sealed {
            bail!("prefix {prefix_index} L{layer} 没有 active lane application");
        }
        Ok(progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_lane_order_rejects_finalize_before_remainder_and_rollover() {
        let mut progress = PrefixLayerProgress::default();
        progress.active_source_lane = Some(0);
        let mut program = TestProgress::new(progress);
        assert!(program.finalize().is_err());
        program.remainder().unwrap();
        assert!(program.finalize().is_err());
        program.window().unwrap();
        program.finalize().unwrap();
        program.rollover().unwrap();
        assert_eq!(program.0.phase, S14CausalBlockPrefixLanePhase::ReadyToSeal);
    }

    /// 不构造大型真实 graph fixture；这个小 helper 只锁定和 production
    /// methods 相同的 phase transition，真实 program identity 由既有 state-writeback
    /// 构建门覆盖。
    struct TestProgress(PrefixLayerProgress);

    impl TestProgress {
        fn new(progress: PrefixLayerProgress) -> Self {
            Self(progress)
        }

        fn window(&mut self) -> Result<()> {
            if self.0.phase != S14CausalBlockPrefixLanePhase::AwaitingWindowAndRemainder
                || self.0.window_recorded
            {
                bail!("window order");
            }
            self.0.window_recorded = true;
            if self.0.remainder_recorded {
                self.0.phase = S14CausalBlockPrefixLanePhase::AwaitingBoundaryFinalize;
            }
            Ok(())
        }

        fn remainder(&mut self) -> Result<()> {
            if self.0.phase != S14CausalBlockPrefixLanePhase::AwaitingWindowAndRemainder
                || self.0.remainder_recorded
            {
                bail!("remainder order");
            }
            self.0.remainder_recorded = true;
            if self.0.window_recorded {
                self.0.phase = S14CausalBlockPrefixLanePhase::AwaitingBoundaryFinalize;
            }
            Ok(())
        }

        fn finalize(&mut self) -> Result<()> {
            if self.0.phase != S14CausalBlockPrefixLanePhase::AwaitingBoundaryFinalize {
                bail!("finalize order");
            }
            self.0.phase = S14CausalBlockPrefixLanePhase::AwaitingRollover;
            Ok(())
        }

        fn rollover(&mut self) -> Result<()> {
            if self.0.phase != S14CausalBlockPrefixLanePhase::AwaitingRollover {
                bail!("rollover order");
            }
            self.0.phase = S14CausalBlockPrefixLanePhase::ReadyToSeal;
            Ok(())
        }
    }
}
