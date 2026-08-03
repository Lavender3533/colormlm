//! S14 StarFold 一个 B4 层的 routed expert 生产 owner。

use crate::{
    s14_starfold_cache::{STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_expert_schedule::{
        S14StarfoldB4ExpertSchedule, S14StarfoldExpertProjection, S14_STARFOLD_HIDDEN,
    },
    s14_starfold_mxfp4_tile::S14StarfoldMxfp4ExternalSlice,
    s14_starfold_prepare_owner::{S14StarfoldPrepareOwner, S14StarfoldPrepareReceipt},
    s14_starfold_routed_executor::{
        S14StarfoldProjectionExecutionReceipt, S14StarfoldRoutedExecutor,
    },
    s14_starfold_runtime::{S14StarfoldB4LayerPlan, S14StarfoldRuntime},
    VulkanContext,
};
use anyhow::{bail, Context, Result};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldRoutedDownBinding {
    pub branches: S14StarfoldMxfp4ExternalSlice,
    pub positions: u32,
    pub branches_per_position: u32,
}

#[derive(Clone, Debug)]
pub struct S14StarfoldB4RoutedLayerReceipt {
    pub layer: u16,
    pub base_position: u64,
    pub unique_experts: u32,
    pub w1: S14StarfoldProjectionExecutionReceipt,
    pub w3: S14StarfoldProjectionExecutionReceipt,
    pub prepare: S14StarfoldPrepareReceipt,
    pub w2: S14StarfoldProjectionExecutionReceipt,
    pub packed_uploads: u32,
    pub packed_upload_bytes: u64,
    pub lane_dispatches: u32,
    pub serial_token_forward_calls: u32,
}

pub struct S14StarfoldB4RoutedLayerOwner {
    routed: S14StarfoldRoutedExecutor,
    prepare: S14StarfoldPrepareOwner,
}

impl S14StarfoldB4RoutedLayerOwner {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        Ok(Self {
            routed: S14StarfoldRoutedExecutor::new(Arc::clone(&context))?,
            prepare: S14StarfoldPrepareOwner::new(context)?,
        })
    }

    pub fn execute(
        &mut self,
        runtime: &mut S14StarfoldRuntime,
        layer_plan: &S14StarfoldB4LayerPlan,
        input_f32: S14StarfoldMxfp4ExternalSlice,
    ) -> Result<(
        S14StarfoldB4RoutedLayerReceipt,
        S14StarfoldRoutedDownBinding,
    )> {
        runtime
            .fetch_b4_layer_ranges(layer_plan)
            .context("S14 StarFold B4 exact-route cache warmup")?;
        let schedule =
            S14StarfoldB4ExpertSchedule::build(layer_plan, runtime.contract().microtile_bytes)?;
        let buffers = self.prepare.routed_buffers(input_f32)?;
        let layout = self.prepare.layout();
        let w1 = self.routed.execute_projection(
            runtime,
            layer_plan,
            &schedule,
            S14StarfoldExpertProjection::W1,
            buffers,
            layout,
        )?;
        let w3 = self.routed.execute_projection(
            runtime,
            layer_plan,
            &schedule,
            S14StarfoldExpertProjection::W3,
            buffers,
            layout,
        )?;
        let prepare = self.prepare.submit_prepare(&schedule)?;
        let w2 = self.routed.execute_projection(
            runtime,
            layer_plan,
            &schedule,
            S14StarfoldExpertProjection::W2,
            buffers,
            layout,
        )?;
        let packed_uploads = w1
            .packed_uploads
            .checked_add(w3.packed_uploads)
            .and_then(|value| value.checked_add(w2.packed_uploads))
            .context("S14 StarFold B4 packed upload count overflow")?;
        let packed_upload_bytes = w1
            .packed_upload_bytes
            .checked_add(w3.packed_upload_bytes)
            .and_then(|value| value.checked_add(w2.packed_upload_bytes))
            .context("S14 StarFold B4 packed upload bytes overflow")?;
        let lane_dispatches = w1
            .lane_dispatches
            .checked_add(w3.lane_dispatches)
            .and_then(|value| value.checked_add(w2.lane_dispatches))
            .context("S14 StarFold B4 lane dispatch count overflow")?;
        if prepare.serial_token_forward_calls != 0
            || w1.serial_token_forward_calls != 0
            || w3.serial_token_forward_calls != 0
            || w2.serial_token_forward_calls != 0
            || packed_uploads == 0
        {
            bail!("S14 StarFold B4 routed layer 退化为串行 token 或空执行");
        }

        let down_bytes =
            (STARFOLD_B4_LANES * STARFOLD_TOP_K) as u64 * u64::from(S14_STARFOLD_HIDDEN) * 4;
        let workspace = buffers.workspace;
        let down_offset = workspace
            .offset
            .checked_add(layout.down)
            .context("S14 StarFold routed down offset overflow")?;
        let down_end = down_offset
            .checked_add(down_bytes)
            .context("S14 StarFold routed down end overflow")?;
        if down_end > workspace.capacity_bytes {
            bail!("S14 StarFold routed down binding 越出 workspace");
        }
        let receipt = S14StarfoldB4RoutedLayerReceipt {
            layer: schedule.layer,
            base_position: schedule.base_position,
            unique_experts: schedule.experts.len() as u32,
            w1,
            w3,
            prepare,
            w2,
            packed_uploads,
            packed_upload_bytes,
            lane_dispatches,
            serial_token_forward_calls: 0,
        };
        Ok((
            receipt,
            S14StarfoldRoutedDownBinding {
                branches: S14StarfoldMxfp4ExternalSlice {
                    buffer: workspace.buffer,
                    capacity_bytes: workspace.capacity_bytes,
                    offset: down_offset,
                    logical_bytes: down_bytes,
                },
                positions: STARFOLD_B4_LANES as u32,
                branches_per_position: STARFOLD_TOP_K as u32,
            },
        ))
    }

    pub fn try_destroy(&mut self) -> Result<()> {
        self.routed.try_destroy()?;
        self.prepare.try_destroy()
    }
}
