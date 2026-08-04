//! S14 StarFold 一个 B4 层的 routed expert 生产 owner。

use crate::{
    s14_starfold_cache::{STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_expert_schedule::{
        S14StarfoldB4ExpertSchedule, S14StarfoldExpertProjection, S14_STARFOLD_HIDDEN,
    },
    s14_starfold_mxfp4_tile::S14StarfoldMxfp4ExternalSlice,
    s14_starfold_prefetch_pipeline::{
        S14StarfoldActiveComputeIdentity, S14StarfoldPrefetchLease, S14StarfoldPrefetchPlanner,
        S14StarfoldPrefetchWindowRequest, S14StarfoldStaticSsdIntent,
    },
    s14_starfold_prepare_owner::{S14StarfoldPrepareOwner, S14StarfoldPrepareReceipt},
    s14_starfold_routed_executor::{
        S14StarfoldProjectionExecutionReceipt, S14StarfoldRoutedBuffers, S14StarfoldRoutedExecutor,
        S14StarfoldRoutedWorkspaceLayout,
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
    prefetch: S14StarfoldPrefetchPlanner,
}

impl S14StarfoldB4RoutedLayerOwner {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        Ok(Self {
            routed: S14StarfoldRoutedExecutor::new(Arc::clone(&context))?,
            prepare: S14StarfoldPrepareOwner::new(context)?,
            prefetch: S14StarfoldPrefetchPlanner::production_defaults()?,
        })
    }

    pub fn execute(
        &mut self,
        runtime: &mut S14StarfoldRuntime,
        layer_plan: &S14StarfoldB4LayerPlan,
        input_f32: S14StarfoldMxfp4ExternalSlice,
        current_compute: S14StarfoldActiveComputeIdentity,
    ) -> Result<(
        S14StarfoldB4RoutedLayerReceipt,
        S14StarfoldRoutedDownBinding,
    )> {
        let schedule =
            S14StarfoldB4ExpertSchedule::build(layer_plan, runtime.contract().microtile_bytes)?;
        let buffers = self.prepare.routed_buffers(input_f32)?;
        let layout = self.prepare.layout();
        let window_bytes = runtime.contract().microtile_bytes;
        let mib = crate::s14_starfold_cache::STARFOLD_ONE_MIB;
        let (w1, w3, prepare, w2) = if [mib, 2 * mib, 4 * mib, 8 * mib].contains(&window_bytes) {
            // Legacy microtile execution still consumes the complete layer
            // synchronously.  Constellation windows below intentionally skip
            // this barrier and fetch only the next packet's Range pairs.
            runtime
                .fetch_b4_layer_ranges(layer_plan)
                .context("S14 StarFold B4 microtile exact-route cache warmup")?;
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
            (w1, w3, prepare, w2)
        } else if [16 * mib, 32 * mib, 64 * mib].contains(&window_bytes) {
            self.execute_constellation_layer(
                runtime,
                layer_plan,
                &schedule,
                buffers,
                layout,
                current_compute,
            )?
        } else {
            bail!(
                "S14 StarFold production window {window_bytes} bytes 不属于 1/2/4/8 microtile 或 16/32/64 constellation 合同"
            )
        };
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

    fn execute_constellation_layer(
        &mut self,
        runtime: &mut S14StarfoldRuntime,
        layer_plan: &S14StarfoldB4LayerPlan,
        schedule: &S14StarfoldB4ExpertSchedule,
        buffers: S14StarfoldRoutedBuffers,
        layout: S14StarfoldRoutedWorkspaceLayout,
        current_compute: S14StarfoldActiveComputeIdentity,
    ) -> Result<(
        S14StarfoldProjectionExecutionReceipt,
        S14StarfoldProjectionExecutionReceipt,
        S14StarfoldPrepareReceipt,
        S14StarfoldProjectionExecutionReceipt,
    )> {
        let mut epoch = self.routed.begin_constellation_layer_epoch(
            runtime,
            schedule.layer,
            schedule.base_position,
        )?;
        let execution = (|| -> Result<_> {
            let mut w1_packets = self.routed.constellation_packet_producer(
                self.prefetch.clone(),
                current_compute.clone(),
                layer_plan,
                schedule,
                S14StarfoldExpertProjection::W1,
                buffers,
                layout,
            )?;
            let w1 = self
                .routed
                .execute_constellation_projection_stream_in_layer_epoch(
                    runtime,
                    &mut epoch,
                    &mut w1_packets,
                )?;
            let mut w3_packets = self.routed.constellation_packet_producer(
                self.prefetch.clone(),
                current_compute.clone(),
                layer_plan,
                schedule,
                S14StarfoldExpertProjection::W3,
                buffers,
                layout,
            )?;
            let w3 = self
                .routed
                .execute_constellation_projection_stream_in_layer_epoch(
                    runtime,
                    &mut epoch,
                    &mut w3_packets,
                )?;
            let prepare = self.prepare.submit_prepare(schedule)?;
            self.routed
                .mark_constellation_layer_prepare_submitted(&mut epoch)?;
            let mut w2_packets = self.routed.constellation_packet_producer(
                self.prefetch.clone(),
                current_compute,
                layer_plan,
                schedule,
                S14StarfoldExpertProjection::W2,
                buffers,
                layout,
            )?;
            let w2 = self
                .routed
                .execute_constellation_projection_stream_in_layer_epoch(
                    runtime,
                    &mut epoch,
                    &mut w2_packets,
                )?;
            Ok((w1, w3, prepare, w2))
        })();
        let drain = if execution.is_ok() {
            self.routed
                .finish_constellation_layer_epoch(runtime, &mut epoch)
        } else {
            self.routed
                .abort_constellation_layer_epoch(runtime, &mut epoch)
        };
        match (execution, drain) {
            (Ok(receipts), Ok(())) => Ok(receipts),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(drain)) => Err(anyhow::anyhow!(
                "{error:#}; constellation layer epoch drain={drain:#}"
            )),
        }
    }

    pub fn issue_static_l2_prefetch(
        &self,
        current_compute: S14StarfoldActiveComputeIdentity,
        intent: S14StarfoldStaticSsdIntent,
    ) -> Result<S14StarfoldPrefetchLease> {
        let mut leases = self
            .prefetch
            .issue_window(S14StarfoldPrefetchWindowRequest::new(
                current_compute,
                None,
                Some(intent),
            )?)?;
        if leases.routed_l1.is_some() {
            bail!("S14 StarFold static L+2 window 意外签发 routed lease");
        }
        leases
            .static_l2
            .take()
            .context("S14 StarFold static L+2 window 缺少 lease")
    }

    pub fn try_destroy(&mut self) -> Result<()> {
        self.routed.try_destroy()?;
        self.prepare.try_destroy()
    }
}
