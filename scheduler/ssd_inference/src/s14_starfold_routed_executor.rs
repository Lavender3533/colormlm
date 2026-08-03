//! S14 StarFold routed expert 的生产执行器。
//!
//! 执行顺序由外层固定为 W1 → W3 → batched SwiGLU/route-weight prepare → W2。
//! 本 owner 负责三个 MXFP4 投影阶段：每个 packed tile 只上传一次，并在同一 Vulkan
//! command 中处理命中该专家的全部 B4 lane。

use crate::{
    s14_starfold_cache::{STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_expert_schedule::{
        S14StarfoldB4ExpertSchedule, S14StarfoldExpertProjection, S14_STARFOLD_HIDDEN,
        S14_STARFOLD_INTERMEDIATE,
    },
    s14_starfold_mxfp4_compute::{S14StarfoldMxfp4ComputeOwner, S14StarfoldMxfp4LaneIo},
    s14_starfold_mxfp4_stream::materialize_packed_mxfp4_tile,
    s14_starfold_mxfp4_tile::S14StarfoldMxfp4ExternalSlice,
    s14_starfold_runtime::{S14StarfoldB4LayerPlan, S14StarfoldRuntime},
    s14_starfold_vulkan_windows::S14StarfoldTimelinePoint,
    VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use std::sync::Arc;

const F32_BYTES: u64 = 4;
const WORKSPACE_ALIGNMENT: u64 = 256;
const ROUTE_BRANCHES: u64 = (STARFOLD_B4_LANES * STARFOLD_TOP_K) as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldRoutedWorkspaceLayout {
    pub route_weights: u64,
    pub gate: u64,
    pub up: u64,
    pub hidden: u64,
    pub down: u64,
    pub bytes: u64,
}

impl S14StarfoldRoutedWorkspaceLayout {
    pub fn build() -> Result<Self> {
        let mut cursor = 0u64;
        let route_weights = take(&mut cursor, ROUTE_BRANCHES * F32_BYTES)?;
        let intermediate_bytes = ROUTE_BRANCHES
            .checked_mul(u64::from(S14_STARFOLD_INTERMEDIATE))
            .and_then(|elements| elements.checked_mul(F32_BYTES))
            .context("S14 StarFold routed intermediate bytes overflow")?;
        let hidden_bytes = ROUTE_BRANCHES
            .checked_mul(u64::from(S14_STARFOLD_HIDDEN))
            .and_then(|elements| elements.checked_mul(F32_BYTES))
            .context("S14 StarFold routed down bytes overflow")?;
        let gate = take(&mut cursor, intermediate_bytes)?;
        let up = take(&mut cursor, intermediate_bytes)?;
        let hidden = take(&mut cursor, intermediate_bytes)?;
        let down = take(&mut cursor, hidden_bytes)?;
        Ok(Self {
            route_weights,
            gate,
            up,
            hidden,
            down,
            bytes: align_up(cursor)?,
        })
    }

    pub fn projection_output_base(self, projection: S14StarfoldExpertProjection) -> u64 {
        match projection {
            S14StarfoldExpertProjection::W1 => self.gate,
            S14StarfoldExpertProjection::W3 => self.up,
            S14StarfoldExpertProjection::W2 => self.down,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldRoutedBuffers {
    /// 四个 lane 连续排列的 F32 HC/FFN 输入，精确 `4 × 4096`。
    pub input_f32: S14StarfoldMxfp4ExternalSlice,
    /// route weights、gate、up、prepared hidden、down 共用的 arena。
    pub workspace: S14StarfoldMxfp4ExternalSlice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldProjectionExecutionReceipt {
    pub layer: u16,
    pub base_position: u64,
    pub projection: S14StarfoldExpertProjection,
    pub unique_experts: u32,
    pub packed_uploads: u32,
    pub packed_upload_bytes: u64,
    pub lane_dispatches: u32,
    pub queue_submit_calls: u32,
    pub completion: S14StarfoldTimelinePoint,
    pub serial_token_forward_calls: u32,
}

pub struct S14StarfoldRoutedExecutor {
    compute: S14StarfoldMxfp4ComputeOwner,
    next_consumer_id: u64,
    next_transfer_epoch: u64,
}

impl S14StarfoldRoutedExecutor {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        Ok(Self {
            compute: S14StarfoldMxfp4ComputeOwner::new(context)?,
            next_consumer_id: 1,
            next_transfer_epoch: 1,
        })
    }

    pub fn execute_projection(
        &mut self,
        runtime: &mut S14StarfoldRuntime,
        layer_plan: &S14StarfoldB4LayerPlan,
        schedule: &S14StarfoldB4ExpertSchedule,
        projection: S14StarfoldExpertProjection,
        buffers: S14StarfoldRoutedBuffers,
        layout: S14StarfoldRoutedWorkspaceLayout,
    ) -> Result<S14StarfoldProjectionExecutionReceipt> {
        validate_buffers(buffers, layout)?;
        if schedule.layer != layer_plan.authoritative_routes.layer()
            || schedule.base_position != layer_plan.authoritative_routes.base_position()
            || schedule.window_capacity_bytes != runtime.contract().microtile_bytes
        {
            bail!("S14 StarFold routed executor 的 plan/schedule/runtime identity 漂移");
        }

        let transfer_epoch = self.next_transfer_epoch;
        self.next_transfer_epoch = self
            .next_transfer_epoch
            .checked_add(1)
            .context("S14 StarFold transfer epoch overflow")?;
        runtime.begin_transfer_block_epoch(transfer_epoch)?;

        let execution = (|| -> Result<S14StarfoldProjectionExecutionReceipt> {
            let mut packed_uploads = 0u32;
            let mut packed_upload_bytes = 0u64;
            let mut lane_dispatches = 0u32;
            let mut queue_submit_calls = 0u32;
            let mut completion = None;
            for expert in &schedule.experts {
                for work in expert.tiles_for(projection) {
                    let packed =
                        materialize_packed_mxfp4_tile(runtime, layer_plan, expert.expert_id, work)?;
                    let proof = Arc::clone(packed.proof());
                    let payload_bytes = proof.byte_len();
                    let ticket = runtime.reserve_verified_upload(proof, 1)?;
                    let ready =
                        runtime.upload_verified_microtile_in_epoch(transfer_epoch, ticket)?;

                    let mut lane_ios = Vec::with_capacity(expert.lane_uses.len());
                    for lane_use in &expert.lane_uses {
                        let branch = branch_index(lane_use.lane, lane_use.route_rank)?;
                        let (input, output) =
                            projection_io(projection, branch, lane_use.lane, buffers, layout)?;
                        lane_ios.push(S14StarfoldMxfp4LaneIo {
                            lane: lane_use.lane,
                            input_f32: input,
                            output_f32: output,
                        });
                    }
                    let consumer_id = self.next_consumer_id;
                    self.next_consumer_id = self
                        .next_consumer_id
                        .checked_add(1)
                        .context("S14 StarFold compute consumer id overflow")?;
                    let ready_binding = ready.binding();
                    let ready_proof = Arc::clone(ready.proof());
                    let receipt = self.compute.submit_ready_tile_batch(
                        runtime.vulkan_windows_mut()?,
                        ready_binding,
                        ready_proof,
                        consumer_id,
                        packed.shape,
                        packed.tile_index,
                        &lane_ios,
                        packed.scale_audit(),
                    )?;
                    packed_uploads = packed_uploads
                        .checked_add(1)
                        .context("S14 StarFold packed upload count overflow")?;
                    packed_upload_bytes = packed_upload_bytes
                        .checked_add(payload_bytes)
                        .context("S14 StarFold packed upload bytes overflow")?;
                    lane_dispatches = lane_dispatches
                        .checked_add(receipt.lane_dispatches)
                        .context("S14 StarFold lane dispatch count overflow")?;
                    queue_submit_calls = queue_submit_calls
                        .checked_add(receipt.queue_submit_calls)
                        .context("S14 StarFold queue submit count overflow")?;
                    completion = Some(receipt.signal_compute);
                }
            }
            if packed_uploads == 0 || lane_dispatches == 0 || packed_uploads != queue_submit_calls {
                bail!("S14 StarFold routed projection 没有形成 upload→batch-dispatch 一一对应");
            }
            Ok(S14StarfoldProjectionExecutionReceipt {
                layer: schedule.layer,
                base_position: schedule.base_position,
                projection,
                unique_experts: schedule.experts.len() as u32,
                packed_uploads,
                packed_upload_bytes,
                lane_dispatches,
                queue_submit_calls,
                completion: completion
                    .context("S14 StarFold routed projection 缺少 compute completion")?,
                serial_token_forward_calls: 0,
            })
        })();
        let drain = runtime.drain_transfer_block_epoch(transfer_epoch);
        match (execution, drain) {
            (Ok(receipt), Ok(())) => Ok(receipt),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(drain)) => {
                Err(anyhow::anyhow!("{error:#}; transfer epoch drain={drain:#}"))
            }
        }
    }

    pub fn try_destroy(&mut self) -> Result<()> {
        self.compute.try_destroy()
    }
}

fn projection_io(
    projection: S14StarfoldExpertProjection,
    branch: u64,
    lane: u8,
    buffers: S14StarfoldRoutedBuffers,
    layout: S14StarfoldRoutedWorkspaceLayout,
) -> Result<(S14StarfoldMxfp4ExternalSlice, S14StarfoldMxfp4ExternalSlice)> {
    let (input_base, input_index, input_width, output_base, output_width) = match projection {
        S14StarfoldExpertProjection::W1 | S14StarfoldExpertProjection::W3 => (
            buffers.input_f32,
            u64::from(lane),
            u64::from(S14_STARFOLD_HIDDEN),
            layout.projection_output_base(projection),
            u64::from(S14_STARFOLD_INTERMEDIATE),
        ),
        S14StarfoldExpertProjection::W2 => (
            buffers.workspace,
            branch,
            u64::from(S14_STARFOLD_INTERMEDIATE),
            layout.down,
            u64::from(S14_STARFOLD_HIDDEN),
        ),
    };
    let input_relative = if projection == S14StarfoldExpertProjection::W2 {
        layout
            .hidden
            .checked_add(elements_bytes(input_index, input_width)?)
            .context("S14 StarFold W2 input offset overflow")?
    } else {
        elements_bytes(input_index, input_width)?
    };
    let output_relative = output_base
        .checked_add(elements_bytes(branch, output_width)?)
        .context("S14 StarFold projection output offset overflow")?;
    Ok((
        sub_slice(input_base, input_relative, input_width * F32_BYTES)?,
        sub_slice(buffers.workspace, output_relative, output_width * F32_BYTES)?,
    ))
}

fn branch_index(lane: u8, route_rank: u8) -> Result<u64> {
    if usize::from(lane) >= STARFOLD_B4_LANES || usize::from(route_rank) >= STARFOLD_TOP_K {
        bail!("S14 StarFold lane/rank 越出 B4 top-6");
    }
    Ok(u64::from(lane) * STARFOLD_TOP_K as u64 + u64::from(route_rank))
}

fn validate_buffers(
    buffers: S14StarfoldRoutedBuffers,
    layout: S14StarfoldRoutedWorkspaceLayout,
) -> Result<()> {
    let input_bytes = STARFOLD_B4_LANES as u64 * u64::from(S14_STARFOLD_HIDDEN) * F32_BYTES;
    validate_slice(buffers.input_f32, input_bytes, "B4 input")?;
    validate_slice(buffers.workspace, layout.bytes, "routed workspace")?;
    if buffers.input_f32.buffer == buffers.workspace.buffer {
        let input_end = buffers
            .input_f32
            .offset
            .checked_add(input_bytes)
            .context("S14 StarFold input end overflow")?;
        let workspace_end = buffers
            .workspace
            .offset
            .checked_add(layout.bytes)
            .context("S14 StarFold workspace end overflow")?;
        if buffers.input_f32.offset < workspace_end && buffers.workspace.offset < input_end {
            bail!("S14 StarFold input 与 routed workspace 重叠");
        }
    }
    Ok(())
}

fn validate_slice(slice: S14StarfoldMxfp4ExternalSlice, required: u64, label: &str) -> Result<()> {
    let end = slice
        .offset
        .checked_add(required)
        .context("S14 StarFold external slice end overflow")?;
    if slice.buffer == vk::Buffer::null()
        || slice.logical_bytes < required
        || end > slice.capacity_bytes
    {
        bail!("S14 StarFold {label} external slice capacity 非法");
    }
    Ok(())
}

fn sub_slice(
    base: S14StarfoldMxfp4ExternalSlice,
    relative: u64,
    bytes: u64,
) -> Result<S14StarfoldMxfp4ExternalSlice> {
    let relative_end = relative
        .checked_add(bytes)
        .context("S14 StarFold sub-slice end overflow")?;
    if bytes == 0 || relative_end > base.logical_bytes {
        bail!("S14 StarFold sub-slice 越出 logical arena");
    }
    Ok(S14StarfoldMxfp4ExternalSlice {
        buffer: base.buffer,
        capacity_bytes: base.capacity_bytes,
        offset: base
            .offset
            .checked_add(relative)
            .context("S14 StarFold sub-slice offset overflow")?,
        logical_bytes: bytes,
    })
}

fn elements_bytes(index: u64, width: u64) -> Result<u64> {
    index
        .checked_mul(width)
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .context("S14 StarFold lane/branch byte offset overflow")
}

fn take(cursor: &mut u64, bytes: u64) -> Result<u64> {
    let offset = align_up(*cursor)?;
    *cursor = offset
        .checked_add(bytes)
        .context("S14 StarFold routed workspace overflow")?;
    Ok(offset)
}

fn align_up(value: u64) -> Result<u64> {
    value
        .checked_add(WORKSPACE_ALIGNMENT - 1)
        .map(|sum| sum & !(WORKSPACE_ALIGNMENT - 1))
        .context("S14 StarFold workspace alignment overflow")
}
