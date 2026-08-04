//! S14 StarFold routed expert 的生产执行器。
//!
//! 执行顺序由外层固定为 W1 → W3 → batched SwiGLU/route-weight prepare → W2。
//! 本 owner 负责三个 MXFP4 投影阶段：每个 packed tile 只上传一次，并在同一 Vulkan
//! command 中处理命中该专家的全部 B4 lane。

#[path = "s14_starfold_constellation_packet.rs"]
pub mod constellation_packet;

use crate::{
    s14_starfold_cache::{STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_expert_schedule::{
        S14StarfoldB4ExpertSchedule, S14StarfoldExpertProjection, S14_STARFOLD_HIDDEN,
        S14_STARFOLD_INTERMEDIATE,
    },
    s14_starfold_mxfp4_compute::{
        S14StarfoldConstellationComputeSubmissionReceipt, S14StarfoldMxfp4ComputeOwner,
        S14StarfoldMxfp4LaneIo,
    },
    s14_starfold_mxfp4_stream::materialize_packed_mxfp4_tile,
    s14_starfold_mxfp4_tile::S14StarfoldMxfp4ExternalSlice,
    s14_starfold_prefetch_pipeline::{
        S14StarfoldActiveComputeIdentity, S14StarfoldPacketProjection,
        S14StarfoldPrefetchFailurePhase, S14StarfoldPrefetchLease, S14StarfoldPrefetchPlanner,
        S14StarfoldPrefetchTarget, S14StarfoldRoutedExpertSet, S14StarfoldSameLayerPacketIntent,
    },
    s14_starfold_runtime::{S14StarfoldB4LayerPlan, S14StarfoldRuntime},
    s14_starfold_vulkan_windows::S14StarfoldTimelinePoint,
    VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use std::{collections::BTreeSet, ops::Range, sync::Arc};

use constellation_packet::{
    build_starfold_constellation_packet_group, build_starfold_constellation_packets,
    S14StarfoldConstellationCandidate, S14StarfoldConstellationLane,
    S14StarfoldConstellationPacket, S14StarfoldConstellationRuntimeHook,
    S14StarfoldResidentWindowKey,
};

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
    active_constellation_layer: Option<S14StarfoldActiveConstellationLayer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum S14StarfoldConstellationLayerPhase {
    W1,
    W3,
    Prepare,
    W2,
    Complete,
    Poisoned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct S14StarfoldActiveConstellationLayer {
    epoch: u64,
    layer: u16,
    base_position: u64,
    phase: S14StarfoldConstellationLayerPhase,
}

/// 一个 16/32/64 MiB constellation 层的唯一 transfer epoch lease。
///
/// lease 本身不在 Drop 中 drain：runtime hook 的可变所有权只存在于显式
/// finish/abort 边界，避免错误路径双 drain 或伪恢复。
pub struct S14StarfoldConstellationLayerEpoch {
    epoch: u64,
    layer: u16,
    base_position: u64,
    closed: bool,
}

#[derive(Debug)]
pub struct S14StarfoldPrefetchedConstellationPacket {
    packet: Arc<S14StarfoldConstellationPacket>,
    lease: S14StarfoldPrefetchLease,
}

impl S14StarfoldPrefetchedConstellationPacket {
    fn into_parts(
        self,
    ) -> (
        Arc<S14StarfoldConstellationPacket>,
        S14StarfoldPrefetchLease,
    ) {
        (self.packet, self.lease)
    }
}

/// 只持有 immutable plan/schedule；每次 `materialize_next` 最多构造一个 window packet。
/// 上一个 packet 已提交 GPU 后，host 在这里准备下一个，从而由固定 A/B window 提供背压。
pub struct S14StarfoldConstellationPacketProducer<'a> {
    planner: S14StarfoldPrefetchPlanner,
    current_compute: S14StarfoldActiveComputeIdentity,
    layer_plan: &'a S14StarfoldB4LayerPlan,
    schedule: &'a S14StarfoldB4ExpertSchedule,
    projection: S14StarfoldExpertProjection,
    buffers: S14StarfoldRoutedBuffers,
    layout: S14StarfoldRoutedWorkspaceLayout,
    descriptor_alignment: u64,
    groups: Vec<Range<usize>>,
    group_bytes: Vec<u64>,
    next_group: usize,
}

impl<'a> S14StarfoldConstellationPacketProducer<'a> {
    pub fn new(
        planner: S14StarfoldPrefetchPlanner,
        current_compute: S14StarfoldActiveComputeIdentity,
        layer_plan: &'a S14StarfoldB4LayerPlan,
        schedule: &'a S14StarfoldB4ExpertSchedule,
        projection: S14StarfoldExpertProjection,
        buffers: S14StarfoldRoutedBuffers,
        layout: S14StarfoldRoutedWorkspaceLayout,
        descriptor_alignment: u64,
    ) -> Result<Self> {
        validate_buffers(buffers, layout)?;
        if current_compute.layer().layer() != schedule.layer
            || schedule.layer != layer_plan.authoritative_routes.layer()
            || schedule.base_position != layer_plan.authoritative_routes.base_position()
            || schedule.window_capacity_bytes < 16 * 1024 * 1024
            || descriptor_alignment == 0
            || !descriptor_alignment.is_power_of_two()
        {
            bail!("S14 StarFold constellation producer plan/schedule/identity 漂移");
        }
        let scheduled = S14StarfoldRoutedExpertSet::from_route_experts(
            schedule.experts.iter().map(|expert| expert.expert_id),
        )?;
        if &scheduled != current_compute.routed_experts() {
            bail!("S14 StarFold constellation producer expert set 与权威 route 漂移");
        }
        let (groups, group_bytes) =
            plan_constellation_packet_groups(schedule, projection, descriptor_alignment)?;
        Ok(Self {
            planner,
            current_compute,
            layer_plan,
            schedule,
            projection,
            buffers,
            layout,
            descriptor_alignment,
            groups,
            group_bytes,
            next_group: 0,
        })
    }

    pub const fn projection(&self) -> S14StarfoldExpertProjection {
        self.projection
    }

    pub fn materialize_next(
        &mut self,
        runtime: &mut S14StarfoldRuntime,
    ) -> Result<Option<S14StarfoldPrefetchedConstellationPacket>> {
        let Some(group) = self.groups.get(self.next_group).cloned() else {
            return Ok(None);
        };
        let packet_ordinal = u16::try_from(self.next_group)
            .context("S14 StarFold constellation producer ordinal 超出 u16")?;
        let expert_ids = self.schedule.experts[group.clone()]
            .iter()
            .map(|expert| expert.expert_id)
            .collect::<Vec<_>>();
        let expert_set =
            S14StarfoldRoutedExpertSet::from_route_experts(expert_ids.iter().copied())?;
        let intent = S14StarfoldSameLayerPacketIntent::new(
            self.current_compute.layer(),
            packet_projection(self.projection),
            packet_ordinal,
            expert_set,
            self.group_bytes[self.next_group],
        )?;
        let mut lease = self.planner.issue_same_layer_packet(intent)?;
        if let Err(error) =
            runtime.fetch_b4_packet_ranges(self.layer_plan, self.projection, &expert_ids)
        {
            let cleanup = lease.fail(S14StarfoldPrefetchFailurePhase::SsdFetch);
            return match cleanup {
                Ok(_) => Err(error),
                Err(cleanup) => Err(anyhow::anyhow!(
                    "{error:#}; constellation packet Range lease cleanup={cleanup:#}"
                )),
            };
        }
        let materialized = self.materialize_group(runtime, group, packet_ordinal);
        let packet = match materialized {
            Ok(packet) => packet,
            Err(error) => {
                let cleanup = lease.fail(S14StarfoldPrefetchFailurePhase::RamMaterialize);
                return match cleanup {
                    Ok(_) => Err(error),
                    Err(cleanup) => Err(anyhow::anyhow!(
                        "{error:#}; constellation prefetch lease cleanup={cleanup:#}"
                    )),
                };
            }
        };
        if let Err(error) = lease.mark_ready(packet.payload_bytes) {
            let cleanup = lease.fail(S14StarfoldPrefetchFailurePhase::RamMaterialize);
            return match cleanup {
                Ok(_) => Err(error),
                Err(cleanup) => Err(anyhow::anyhow!(
                    "{error:#}; constellation ready lease cleanup={cleanup:#}"
                )),
            };
        }
        self.next_group += 1;
        Ok(Some(S14StarfoldPrefetchedConstellationPacket {
            packet,
            lease,
        }))
    }

    fn materialize_group(
        &self,
        runtime: &mut S14StarfoldRuntime,
        group: Range<usize>,
        packet_ordinal: u16,
    ) -> Result<Arc<S14StarfoldConstellationPacket>> {
        let mut candidates = Vec::with_capacity(group.len());
        for expert in &self.schedule.experts[group] {
            let mut works = expert.tiles_for(self.projection);
            let work = works
                .next()
                .context("S14 StarFold constellation producer 缺少 projection tile")?;
            if works.next().is_some() {
                bail!("S14 StarFold constellation producer 要求每专家单 tile");
            }
            let packed =
                materialize_packed_mxfp4_tile(runtime, self.layer_plan, expert.expert_id, work)?;
            let mut lanes = Vec::with_capacity(expert.lane_uses.len());
            for lane_use in &expert.lane_uses {
                let branch = branch_index(lane_use.lane, lane_use.route_rank)?;
                let (input_f32, output_f32) = projection_io(
                    self.projection,
                    branch,
                    lane_use.lane,
                    self.buffers,
                    self.layout,
                )?;
                lanes.push(S14StarfoldConstellationLane {
                    lane: lane_use.lane,
                    route_rank: lane_use.route_rank,
                    route_weight_bits: lane_use.route_weight.to_bits(),
                    input_f32,
                    output_f32,
                });
            }
            candidates.push(S14StarfoldConstellationCandidate {
                expert_id: expert.expert_id,
                projection: self.projection,
                shape: packed.shape,
                tile_index: packed.tile_index,
                scale_audit: packed.scale_audit(),
                proof: Arc::clone(packed.proof()),
                lanes,
            });
        }
        build_starfold_constellation_packet_group(
            self.schedule.layer,
            self.schedule.base_position,
            self.projection,
            packet_ordinal,
            self.schedule.window_capacity_bytes,
            self.descriptor_alignment,
            candidates,
        )
    }
}

impl S14StarfoldRoutedExecutor {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        Ok(Self {
            compute: S14StarfoldMxfp4ComputeOwner::new(context)?,
            next_consumer_id: 1,
            next_transfer_epoch: 1,
            active_constellation_layer: None,
        })
    }

    pub fn constellation_packet_producer<'a>(
        &self,
        planner: S14StarfoldPrefetchPlanner,
        current_compute: S14StarfoldActiveComputeIdentity,
        layer_plan: &'a S14StarfoldB4LayerPlan,
        schedule: &'a S14StarfoldB4ExpertSchedule,
        projection: S14StarfoldExpertProjection,
        buffers: S14StarfoldRoutedBuffers,
        layout: S14StarfoldRoutedWorkspaceLayout,
    ) -> Result<S14StarfoldConstellationPacketProducer<'a>> {
        S14StarfoldConstellationPacketProducer::new(
            planner,
            current_compute,
            layer_plan,
            schedule,
            projection,
            buffers,
            layout,
            self.compute.descriptor_offset_alignment(),
        )
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
        if self.active_constellation_layer.is_some() {
            bail!("S14 StarFold 活动 constellation layer 内禁止进入 microtile projection");
        }
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
                    if ready_binding.key()
                        != S14StarfoldResidentWindowKey::Microtile(ready_proof.key())
                    {
                        bail!("S14 StarFold microtile ready binding 未使用统一 resident key");
                    }
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

    /// 只完成 proof-bound host materialize 与确定性星座分包，不触碰 Vulkan window。
    /// 调用方必须随后使用统一 resident window owner 上的显式
    /// [`S14StarfoldConstellationRuntimeHook`]，不能静默回退旧逐专家上传。
    pub fn materialize_projection_constellations(
        &mut self,
        runtime: &mut S14StarfoldRuntime,
        layer_plan: &S14StarfoldB4LayerPlan,
        schedule: &S14StarfoldB4ExpertSchedule,
        projection: S14StarfoldExpertProjection,
        buffers: S14StarfoldRoutedBuffers,
        layout: S14StarfoldRoutedWorkspaceLayout,
    ) -> Result<Vec<Arc<S14StarfoldConstellationPacket>>> {
        validate_buffers(buffers, layout)?;
        let window_capacity_bytes = runtime.contract().microtile_bytes;
        if schedule.layer != layer_plan.authoritative_routes.layer()
            || schedule.base_position != layer_plan.authoritative_routes.base_position()
            || schedule.window_capacity_bytes != window_capacity_bytes
        {
            bail!("S14 StarFold 星座 materialize 的 plan/schedule/runtime identity 漂移");
        }
        let mut candidates = Vec::with_capacity(schedule.experts.len());
        for expert in &schedule.experts {
            let mut works = expert.tiles_for(projection);
            let work = works
                .next()
                .context("S14 StarFold 星座候选缺少 projection tile")?;
            if works.next().is_some() {
                bail!("S14 StarFold 星座模式要求 16/32/64 MiB 下每专家投影单 tile");
            }
            let packed =
                materialize_packed_mxfp4_tile(runtime, layer_plan, expert.expert_id, work)?;
            let mut lanes = Vec::with_capacity(expert.lane_uses.len());
            for lane_use in &expert.lane_uses {
                let branch = branch_index(lane_use.lane, lane_use.route_rank)?;
                let (input_f32, output_f32) =
                    projection_io(projection, branch, lane_use.lane, buffers, layout)?;
                lanes.push(S14StarfoldConstellationLane {
                    lane: lane_use.lane,
                    route_rank: lane_use.route_rank,
                    route_weight_bits: lane_use.route_weight.to_bits(),
                    input_f32,
                    output_f32,
                });
            }
            candidates.push(S14StarfoldConstellationCandidate {
                expert_id: expert.expert_id,
                projection,
                shape: packed.shape,
                tile_index: packed.tile_index,
                scale_audit: packed.scale_audit(),
                proof: Arc::clone(packed.proof()),
                lanes,
            });
        }
        build_starfold_constellation_packets(
            schedule.layer,
            schedule.base_position,
            projection,
            window_capacity_bytes,
            self.compute.descriptor_offset_alignment(),
            candidates,
        )
    }

    /// 通过显式 runtime hook 执行已经物化的星座包。每包一次 transfer submit、一次
    /// compute submit；任一阶段失败仍 drain 同一 epoch，保持 A/B 回滚边界。
    pub fn execute_materialized_constellations_with_hook<H>(
        &mut self,
        hook: &mut H,
        packets: Vec<Arc<S14StarfoldConstellationPacket>>,
    ) -> Result<S14StarfoldProjectionExecutionReceipt>
    where
        H: S14StarfoldConstellationRuntimeHook,
    {
        if self.active_constellation_layer.is_some() {
            bail!("S14 StarFold 活动 layer epoch 内禁止进入单 projection 兼容包装");
        }
        let transfer_epoch = self.allocate_transfer_epoch()?;
        hook.begin_constellation_epoch(transfer_epoch)?;
        let execution = self.execute_constellation_packets_in_epoch(hook, transfer_epoch, packets);
        let drain = hook.drain_constellation_epoch(transfer_epoch);
        match (execution, drain) {
            (Ok(receipt), Ok(())) => Ok(receipt),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(drain)) => Err(anyhow::anyhow!(
                "{error:#}; constellation epoch drain={drain:#}"
            )),
        }
    }

    pub fn begin_constellation_layer_epoch<H>(
        &mut self,
        hook: &mut H,
        layer: u16,
        base_position: u64,
    ) -> Result<S14StarfoldConstellationLayerEpoch>
    where
        H: S14StarfoldConstellationRuntimeHook,
    {
        if self.active_constellation_layer.is_some() {
            bail!("S14 StarFold constellation layer epoch 已活动");
        }
        let epoch = self.allocate_transfer_epoch()?;
        hook.begin_constellation_epoch(epoch)?;
        self.active_constellation_layer = Some(S14StarfoldActiveConstellationLayer {
            epoch,
            layer,
            base_position,
            phase: S14StarfoldConstellationLayerPhase::W1,
        });
        Ok(S14StarfoldConstellationLayerEpoch {
            epoch,
            layer,
            base_position,
            closed: false,
        })
    }

    pub fn execute_constellation_projection_in_layer_epoch<H>(
        &mut self,
        hook: &mut H,
        owner: &mut S14StarfoldConstellationLayerEpoch,
        packets: Vec<Arc<S14StarfoldConstellationPacket>>,
    ) -> Result<S14StarfoldProjectionExecutionReceipt>
    where
        H: S14StarfoldConstellationRuntimeHook,
    {
        let first = packets
            .first()
            .context("S14 StarFold layer epoch 缺少 constellation packet")?;
        first.validate()?;
        let key = first.key();
        let projection = projection_from_code(key.projection)?;
        let active = self.validate_layer_epoch(owner)?;
        if key.layer != active.layer || key.base_position != active.base_position {
            bail!("S14 StarFold layer epoch packet layer/base identity 漂移");
        }
        let expected_phase = match projection {
            S14StarfoldExpertProjection::W1 => S14StarfoldConstellationLayerPhase::W1,
            S14StarfoldExpertProjection::W3 => S14StarfoldConstellationLayerPhase::W3,
            S14StarfoldExpertProjection::W2 => S14StarfoldConstellationLayerPhase::W2,
        };
        if active.phase != expected_phase {
            bail!("S14 StarFold layer epoch projection/phase 顺序漂移");
        }
        let execution = self.execute_constellation_packets_in_epoch(hook, owner.epoch, packets);
        match execution {
            Ok(receipt) => {
                let next = match projection {
                    S14StarfoldExpertProjection::W1 => S14StarfoldConstellationLayerPhase::W3,
                    S14StarfoldExpertProjection::W3 => S14StarfoldConstellationLayerPhase::Prepare,
                    S14StarfoldExpertProjection::W2 => S14StarfoldConstellationLayerPhase::Complete,
                };
                self.active_constellation_layer
                    .as_mut()
                    .context("S14 StarFold layer epoch active state 丢失")?
                    .phase = next;
                Ok(receipt)
            }
            Err(error) => {
                if let Some(active) = self.active_constellation_layer.as_mut() {
                    active.phase = S14StarfoldConstellationLayerPhase::Poisoned;
                }
                Err(error)
            }
        }
    }

    /// production packet producer/consumer 流水：提交 packet N 后立即在 host 物化 N+1；
    /// compute fence 只在固定 A/B window 真正复用时产生背压，不先物化整投影。
    pub fn execute_constellation_projection_stream_in_layer_epoch(
        &mut self,
        runtime: &mut S14StarfoldRuntime,
        owner: &mut S14StarfoldConstellationLayerEpoch,
        producer: &mut S14StarfoldConstellationPacketProducer<'_>,
    ) -> Result<S14StarfoldProjectionExecutionReceipt> {
        let projection = producer.projection();
        let active = self.validate_layer_epoch(owner)?;
        let expected_phase = match projection {
            S14StarfoldExpertProjection::W1 => S14StarfoldConstellationLayerPhase::W1,
            S14StarfoldExpertProjection::W3 => S14StarfoldConstellationLayerPhase::W3,
            S14StarfoldExpertProjection::W2 => S14StarfoldConstellationLayerPhase::W2,
        };
        if active.phase != expected_phase
            || producer.schedule.layer != active.layer
            || producer.schedule.base_position != active.base_position
        {
            bail!("S14 StarFold constellation stream layer/projection phase 漂移");
        }
        let execution =
            self.execute_constellation_packet_stream_in_epoch(runtime, owner.epoch, producer);
        match execution {
            Ok(receipt) => {
                let next = match projection {
                    S14StarfoldExpertProjection::W1 => S14StarfoldConstellationLayerPhase::W3,
                    S14StarfoldExpertProjection::W3 => S14StarfoldConstellationLayerPhase::Prepare,
                    S14StarfoldExpertProjection::W2 => S14StarfoldConstellationLayerPhase::Complete,
                };
                self.active_constellation_layer
                    .as_mut()
                    .context("S14 StarFold constellation stream active state 丢失")?
                    .phase = next;
                Ok(receipt)
            }
            Err(error) => {
                if let Some(active) = self.active_constellation_layer.as_mut() {
                    active.phase = S14StarfoldConstellationLayerPhase::Poisoned;
                }
                Err(error)
            }
        }
    }

    pub fn mark_constellation_layer_prepare_submitted(
        &mut self,
        owner: &mut S14StarfoldConstellationLayerEpoch,
    ) -> Result<()> {
        let active = self.validate_layer_epoch(owner)?;
        if active.phase != S14StarfoldConstellationLayerPhase::Prepare {
            bail!("S14 StarFold layer prepare 未处于 Prepare phase");
        }
        self.active_constellation_layer
            .as_mut()
            .context("S14 StarFold layer epoch active state 丢失")?
            .phase = S14StarfoldConstellationLayerPhase::W2;
        Ok(())
    }

    pub fn finish_constellation_layer_epoch<H>(
        &mut self,
        hook: &mut H,
        owner: &mut S14StarfoldConstellationLayerEpoch,
    ) -> Result<()>
    where
        H: S14StarfoldConstellationRuntimeHook,
    {
        let active = self.validate_layer_epoch(owner)?;
        if active.phase != S14StarfoldConstellationLayerPhase::Complete {
            bail!("S14 StarFold layer epoch 未完成 W2，禁止 finish");
        }
        match hook.drain_constellation_epoch(owner.epoch) {
            Ok(()) => {
                self.active_constellation_layer = None;
                owner.closed = true;
                Ok(())
            }
            Err(error) => {
                self.active_constellation_layer
                    .as_mut()
                    .context("S14 StarFold layer epoch active state 丢失")?
                    .phase = S14StarfoldConstellationLayerPhase::Poisoned;
                Err(error)
            }
        }
    }

    pub fn abort_constellation_layer_epoch<H>(
        &mut self,
        hook: &mut H,
        owner: &mut S14StarfoldConstellationLayerEpoch,
    ) -> Result<()>
    where
        H: S14StarfoldConstellationRuntimeHook,
    {
        self.validate_layer_epoch(owner)?;
        match hook.drain_constellation_epoch(owner.epoch) {
            Ok(()) => {
                self.active_constellation_layer = None;
                owner.closed = true;
                Ok(())
            }
            Err(error) => {
                self.active_constellation_layer
                    .as_mut()
                    .context("S14 StarFold layer epoch active state 丢失")?
                    .phase = S14StarfoldConstellationLayerPhase::Poisoned;
                Err(error)
            }
        }
    }

    fn execute_constellation_packets_in_epoch<H>(
        &mut self,
        hook: &mut H,
        transfer_epoch: u64,
        packets: Vec<Arc<S14StarfoldConstellationPacket>>,
    ) -> Result<S14StarfoldProjectionExecutionReceipt>
    where
        H: S14StarfoldConstellationRuntimeHook,
    {
        let first = packets
            .first()
            .context("S14 StarFold 星座执行缺少 packet")?;
        first.validate()?;
        let key = first.key();
        let projection = projection_from_code(key.projection)?;
        let mut experts = BTreeSet::new();
        let mut routes = BTreeSet::new();
        let mut packed_uploads = 0u32;
        let mut packed_upload_bytes = 0u64;
        let mut lane_dispatches = 0u32;
        let mut queue_submit_calls = 0u32;
        let mut completion = None;
        for (ordinal, packet) in packets.into_iter().enumerate() {
            packet.validate()?;
            let packet_key = packet.key();
            if packet_key.layer != key.layer
                || packet_key.base_position != key.base_position
                || packet_key.projection != key.projection
                || usize::from(packet_key.packet_ordinal) != ordinal
            {
                bail!("S14 StarFold 星座 packet 序列 identity 漂移");
            }
            for member in packet.members() {
                experts.insert(member.expert_id);
                for lane in member.lanes() {
                    if !routes.insert((lane.lane, lane.route_rank)) {
                        bail!("S14 StarFold 星座执行检测到重复 lane/rank");
                    }
                }
            }
            let packet_bytes = packet.payload_bytes;
            let ready = hook.upload_constellation_packet_in_epoch(transfer_epoch, packet)?;
            let consumer_id = self.next_consumer_id;
            self.next_consumer_id = self
                .next_consumer_id
                .checked_add(1)
                .context("S14 StarFold 星座 compute consumer id overflow")?;
            let receipt: S14StarfoldConstellationComputeSubmissionReceipt =
                self.compute.submit_ready_constellation_batch(
                    hook.constellation_windows_mut()?,
                    ready,
                    consumer_id,
                )?;
            packed_uploads = packed_uploads
                .checked_add(receipt.packet.transfer_submit_calls)
                .context("S14 StarFold 星座 packet upload count overflow")?;
            packed_upload_bytes = packed_upload_bytes
                .checked_add(packet_bytes)
                .context("S14 StarFold 星座 packet upload bytes overflow")?;
            lane_dispatches = lane_dispatches
                .checked_add(receipt.lane_dispatches)
                .context("S14 StarFold 星座 lane dispatch count overflow")?;
            queue_submit_calls = queue_submit_calls
                .checked_add(receipt.queue_submit_calls)
                .context("S14 StarFold 星座 queue submit count overflow")?;
            completion = Some(receipt.signal_compute);
        }
        if packed_uploads == 0
            || lane_dispatches != (STARFOLD_B4_LANES * STARFOLD_TOP_K) as u32
            || routes.len() != STARFOLD_B4_LANES * STARFOLD_TOP_K
            || packed_uploads != queue_submit_calls
        {
            bail!("S14 StarFold 星座执行未形成精确24 route与一包一提交");
        }
        Ok(S14StarfoldProjectionExecutionReceipt {
            layer: key.layer,
            base_position: key.base_position,
            projection,
            unique_experts: u32::try_from(experts.len())
                .context("S14 StarFold 星座 unique experts 超出 u32")?,
            packed_uploads,
            packed_upload_bytes,
            lane_dispatches,
            queue_submit_calls,
            completion: completion.context("S14 StarFold 星座执行缺少 completion")?,
            serial_token_forward_calls: 0,
        })
    }

    fn execute_constellation_packet_stream_in_epoch(
        &mut self,
        runtime: &mut S14StarfoldRuntime,
        transfer_epoch: u64,
        producer: &mut S14StarfoldConstellationPacketProducer<'_>,
    ) -> Result<S14StarfoldProjectionExecutionReceipt> {
        let projection = producer.projection();
        let mut experts = BTreeSet::new();
        let mut routes = BTreeSet::new();
        let mut packed_uploads = 0u32;
        let mut packed_upload_bytes = 0u64;
        let mut lane_dispatches = 0u32;
        let mut queue_submit_calls = 0u32;
        let mut completion = None;
        let mut packet_ordinal = 0u16;

        while let Some(prefetched) = producer.materialize_next(runtime)? {
            let (packet, lease) = prefetched.into_parts();
            packet.validate()?;
            let key = packet.key();
            let packet_experts = S14StarfoldRoutedExpertSet::from_route_experts(
                packet.members().iter().map(|member| member.expert_id),
            )?;
            let target_matches = matches!(
                lease.target(),
                S14StarfoldPrefetchTarget::ConstellationPacket {
                    projection: planned_projection,
                    packet_ordinal: planned_ordinal,
                    routed_experts,
                } if *planned_projection == packet_projection(projection)
                    && *planned_ordinal == packet_ordinal
                    && routed_experts == &packet_experts
            );
            if key.layer != producer.schedule.layer
                || key.base_position != producer.schedule.base_position
                || key.projection != projection_code(projection)
                || key.packet_ordinal != packet_ordinal
                || lease.identity() != producer.current_compute.layer()
                || !target_matches
            {
                bail!("S14 StarFold constellation stream packet/prefetch identity 漂移");
            }
            for member in packet.members() {
                if !experts.insert(member.expert_id) {
                    bail!("S14 StarFold constellation stream 跨 packet 专家重复");
                }
                for lane in member.lanes() {
                    if !routes.insert((lane.lane, lane.route_rank)) {
                        bail!("S14 StarFold constellation stream 跨 packet route 重复");
                    }
                }
            }

            let packet_bytes = packet.payload_bytes;
            let ready = runtime.upload_constellation_packet_in_epoch(transfer_epoch, packet)?;
            let consumer_id = self.next_consumer_id;
            self.next_consumer_id = self
                .next_consumer_id
                .checked_add(1)
                .context("S14 StarFold constellation stream consumer id overflow")?;
            let receipt = self.compute.submit_ready_constellation_batch(
                runtime.vulkan_windows_mut()?,
                ready,
                consumer_id,
            )?;
            lease
                .consume()
                .context("S14 StarFold constellation packet 提交后释放 producer lease")?;
            packed_uploads = packed_uploads
                .checked_add(receipt.packet.transfer_submit_calls)
                .context("S14 StarFold constellation stream upload count overflow")?;
            packed_upload_bytes = packed_upload_bytes
                .checked_add(packet_bytes)
                .context("S14 StarFold constellation stream upload bytes overflow")?;
            lane_dispatches = lane_dispatches
                .checked_add(receipt.lane_dispatches)
                .context("S14 StarFold constellation stream lane dispatch overflow")?;
            queue_submit_calls = queue_submit_calls
                .checked_add(receipt.queue_submit_calls)
                .context("S14 StarFold constellation stream queue submit overflow")?;
            completion = Some(receipt.signal_compute);
            packet_ordinal = packet_ordinal
                .checked_add(1)
                .context("S14 StarFold constellation stream packet ordinal overflow")?;
        }
        if packet_ordinal == 0
            || packed_uploads != u32::from(packet_ordinal)
            || lane_dispatches != (STARFOLD_B4_LANES * STARFOLD_TOP_K) as u32
            || routes.len() != STARFOLD_B4_LANES * STARFOLD_TOP_K
            || packed_uploads != queue_submit_calls
        {
            bail!("S14 StarFold constellation stream 未形成精确24 route与一包一提交");
        }
        Ok(S14StarfoldProjectionExecutionReceipt {
            layer: producer.schedule.layer,
            base_position: producer.schedule.base_position,
            projection,
            unique_experts: u32::try_from(experts.len())
                .context("S14 StarFold constellation stream unique experts 超出 u32")?,
            packed_uploads,
            packed_upload_bytes,
            lane_dispatches,
            queue_submit_calls,
            completion: completion.context("S14 StarFold constellation stream 缺少 completion")?,
            serial_token_forward_calls: 0,
        })
    }

    fn allocate_transfer_epoch(&mut self) -> Result<u64> {
        let epoch = self.next_transfer_epoch;
        self.next_transfer_epoch = self
            .next_transfer_epoch
            .checked_add(1)
            .context("S14 StarFold constellation transfer epoch overflow")?;
        Ok(epoch)
    }

    fn validate_layer_epoch(
        &self,
        owner: &S14StarfoldConstellationLayerEpoch,
    ) -> Result<S14StarfoldActiveConstellationLayer> {
        if owner.closed {
            bail!("S14 StarFold constellation layer epoch 已关闭");
        }
        let active = self
            .active_constellation_layer
            .context("S14 StarFold constellation layer epoch 未活动")?;
        if active.epoch != owner.epoch
            || active.layer != owner.layer
            || active.base_position != owner.base_position
        {
            bail!("S14 StarFold constellation layer epoch lease identity 漂移");
        }
        Ok(active)
    }

    pub fn try_destroy(&mut self) -> Result<()> {
        if self.active_constellation_layer.is_some() {
            bail!("S14 StarFold 活动 constellation layer epoch 下禁止销毁 routed executor");
        }
        self.compute.try_destroy()
    }
}

fn projection_from_code(code: u8) -> Result<S14StarfoldExpertProjection> {
    match code {
        0 => Ok(S14StarfoldExpertProjection::W1),
        1 => Ok(S14StarfoldExpertProjection::W3),
        2 => Ok(S14StarfoldExpertProjection::W2),
        _ => bail!("S14 StarFold 星座 packet projection code 非法"),
    }
}

const fn projection_code(projection: S14StarfoldExpertProjection) -> u8 {
    match projection {
        S14StarfoldExpertProjection::W1 => 0,
        S14StarfoldExpertProjection::W3 => 1,
        S14StarfoldExpertProjection::W2 => 2,
    }
}

const fn packet_projection(projection: S14StarfoldExpertProjection) -> S14StarfoldPacketProjection {
    match projection {
        S14StarfoldExpertProjection::W1 => S14StarfoldPacketProjection::W1,
        S14StarfoldExpertProjection::W3 => S14StarfoldPacketProjection::W3,
        S14StarfoldExpertProjection::W2 => S14StarfoldPacketProjection::W2,
    }
}

fn plan_constellation_packet_groups(
    schedule: &S14StarfoldB4ExpertSchedule,
    projection: S14StarfoldExpertProjection,
    descriptor_alignment: u64,
) -> Result<(Vec<Range<usize>>, Vec<u64>)> {
    let mut groups = Vec::new();
    let mut group_bytes = Vec::new();
    let mut start = 0usize;
    let mut cursor = 0u64;
    for (index, expert) in schedule.experts.iter().enumerate() {
        let mut works = expert.tiles_for(projection);
        let work = works
            .next()
            .context("S14 StarFold constellation group plan 缺少 projection tile")?;
        if works.next().is_some() {
            bail!("S14 StarFold constellation group plan 要求每专家单 tile");
        }
        let offset = align_up_to(cursor, descriptor_alignment)?;
        let mut end = offset
            .checked_add(work.tile.payload_bytes)
            .context("S14 StarFold constellation group end overflow")?;
        if end > u64::from(schedule.window_capacity_bytes) && index > start {
            groups.push(start..index);
            group_bytes.push(cursor);
            start = index;
            end = work.tile.payload_bytes;
        }
        if end > u64::from(schedule.window_capacity_bytes) {
            bail!("S14 StarFold constellation 单专家 payload 越出 window");
        }
        cursor = end;
    }
    if start < schedule.experts.len() {
        groups.push(start..schedule.experts.len());
        group_bytes.push(cursor);
    }
    if groups.is_empty()
        || groups.len() != group_bytes.len()
        || groups.iter().any(|group| group.is_empty())
    {
        bail!("S14 StarFold constellation group plan 为空或不完整");
    }
    Ok((groups, group_bytes))
}

fn align_up_to(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("S14 StarFold constellation descriptor alignment 非法");
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("S14 StarFold constellation alignment overflow")
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
