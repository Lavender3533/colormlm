//! 同层、同投影 MXFP4 专家的 StarFold 星座包合同。
//!
//! 本模块只构造 proof-bound host packet 与未来 production runtime 所需的窄接口；
//! 它不会把普通 `StarfoldPageKey` 冒充成多专家身份，也不会自行提交 Vulkan 工作。

use crate::{
    s14_starfold_cache::{StarfoldPageKey, STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_expert_schedule::S14StarfoldExpertProjection,
    s14_starfold_mxfp4_tile::{
        S14StarfoldMxfp4ExternalSlice, S14StarfoldMxfp4ScaleAudit, S14StarfoldMxfp4TileShape,
    },
    s14_starfold_runtime::S14StarfoldVerifiedMicrotile,
    s14_starfold_vulkan_windows::{
        S14StarfoldReadyBinding, S14StarfoldVulkanWindows, S14StarfoldWindowId,
    },
};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc};

pub const S14_STARFOLD_CONSTELLATION_CONTRACT_VERSION: u32 = 1;
pub const S14_STARFOLD_CONSTELLATION_MIN_WINDOW_BYTES: u32 = 16 * 1024 * 1024;
pub const S14_STARFOLD_CONSTELLATION_MAX_WINDOW_BYTES: u32 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct S14StarfoldConstellationPacketKey {
    pub layer: u16,
    pub base_position: u64,
    pub projection: u8,
    pub packet_ordinal: u16,
    pub identity_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldConstellationLane {
    pub lane: u8,
    pub route_rank: u8,
    /// 必须是权威 route weight 的原始 IEEE-754 bits，禁止重新归一化或舍入。
    pub route_weight_bits: u32,
    pub input_f32: S14StarfoldMxfp4ExternalSlice,
    pub output_f32: S14StarfoldMxfp4ExternalSlice,
}

#[derive(Debug)]
pub struct S14StarfoldConstellationCandidate {
    pub expert_id: u16,
    pub projection: S14StarfoldExpertProjection,
    pub shape: S14StarfoldMxfp4TileShape,
    pub tile_index: u32,
    pub scale_audit: S14StarfoldMxfp4ScaleAudit,
    pub proof: Arc<S14StarfoldVerifiedMicrotile>,
    pub lanes: Vec<S14StarfoldConstellationLane>,
}

#[derive(Debug)]
pub struct S14StarfoldConstellationMember {
    pub expert_id: u16,
    pub source_key: StarfoldPageKey,
    pub shape: S14StarfoldMxfp4TileShape,
    pub tile_index: u32,
    pub window_offset: u64,
    pub payload_bytes: u64,
    pub scale_audit: S14StarfoldMxfp4ScaleAudit,
    proof: Arc<S14StarfoldVerifiedMicrotile>,
    lanes: Vec<S14StarfoldConstellationLane>,
}

impl S14StarfoldConstellationMember {
    pub fn proof(&self) -> &Arc<S14StarfoldVerifiedMicrotile> {
        &self.proof
    }

    pub fn lanes(&self) -> &[S14StarfoldConstellationLane] {
        &self.lanes
    }
}

#[derive(Debug)]
pub struct S14StarfoldConstellationPacket {
    key: S14StarfoldConstellationPacketKey,
    pub window_capacity_bytes: u32,
    pub descriptor_alignment: u64,
    pub payload_bytes: u64,
    pub logical_payload_bytes: u64,
    payload: Arc<[u8]>,
    members: Vec<S14StarfoldConstellationMember>,
}

impl S14StarfoldConstellationPacket {
    pub const fn key(&self) -> S14StarfoldConstellationPacketKey {
        self.key
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn members(&self) -> &[S14StarfoldConstellationMember] {
        &self.members
    }

    /// O(member + route) 的热路径检查。packet 字段私有且只能由本模块 builder 构造，
    /// 因此这里不重复扫描几十 MiB payload 或重算 SHA。
    pub fn validate(&self) -> Result<()> {
        validate_packet_contract(
            self.window_capacity_bytes,
            self.descriptor_alignment,
            self.payload_bytes,
        )?;
        if self.members.is_empty()
            || self.payload.len() as u64 != self.payload_bytes
            || self.logical_payload_bytes == 0
            || self.logical_payload_bytes > self.payload_bytes
        {
            bail!("S14 StarFold 星座包 payload/member 合同漂移");
        }
        let mut logical = 0u64;
        let mut previous_end = 0u64;
        let projection = projection_from_code(self.key.projection)?;
        for member in &self.members {
            validate_member_structure(
                member,
                self.window_capacity_bytes,
                self.descriptor_alignment,
            )?;
            if member.window_offset < previous_end {
                bail!("S14 StarFold 星座包成员发生重叠或顺序漂移");
            }
            if member.source_key.layer != self.key.layer
                || member.source_key.segment != projection.weight_segment()
            {
                bail!("S14 StarFold 星座包成员 layer/projection identity 漂移");
            }
            let end = member
                .window_offset
                .checked_add(member.payload_bytes)
                .context("S14 StarFold 星座包成员 end overflow")?;
            if end > self.payload_bytes {
                bail!("S14 StarFold 星座包成员越出 packet payload");
            }
            logical = logical
                .checked_add(member.payload_bytes)
                .context("S14 StarFold 星座包 logical bytes overflow")?;
            previous_end = end;
        }
        if logical != self.logical_payload_bytes || previous_end != self.payload_bytes {
            bail!("S14 StarFold 星座包逻辑/物理 payload 统计漂移");
        }
        if self.key.identity_sha256 == [0; 32] {
            bail!("S14 StarFold 星座包 identity SHA-256 为空");
        }
        Ok(())
    }

    /// 只在 immutable packet 构造完成时调用一次的重校验。
    fn validate_fully(&self) -> Result<()> {
        self.validate()?;
        for member in &self.members {
            let end = member
                .window_offset
                .checked_add(member.payload_bytes)
                .context("S14 StarFold 星座包 full validation end overflow")?;
            let start = usize::try_from(member.window_offset)
                .context("S14 StarFold 星座包成员 offset 超出 usize")?;
            let end = usize::try_from(end).context("S14 StarFold 星座包成员 end 超出 usize")?;
            if &self.payload[start..end] != member.proof.bytes() {
                bail!("S14 StarFold 星座包成员 bytes 与 proof lease 漂移");
            }
            let expected_audit = S14StarfoldMxfp4ScaleAudit::scan_host_payload(
                member.shape,
                member.tile_index,
                member.proof.bytes(),
            )?;
            if member.scale_audit != expected_audit {
                bail!("S14 StarFold 星座包成员 scale audit 漂移");
            }
        }
        let projection = projection_from_code(self.key.projection)?;
        let expected = packet_identity(
            self.key.layer,
            self.key.base_position,
            projection,
            self.key.packet_ordinal,
            self.window_capacity_bytes,
            self.descriptor_alignment,
            &self.members,
            &self.payload,
        )?;
        if expected != self.key.identity_sha256 {
            bail!("S14 StarFold 星座包 identity SHA-256 漂移");
        }
        Ok(())
    }
}

/// 构造确定性、保持 expert schedule 顺序的星座包。当前 MXFP4 形状在 16 MiB 及以上
/// 都是每专家单 tile，因此这里严格要求每个 B4 lane/rank 恰好出现一次，共 24 项。
pub fn build_starfold_constellation_packets(
    layer: u16,
    base_position: u64,
    projection: S14StarfoldExpertProjection,
    window_capacity_bytes: u32,
    descriptor_alignment: u64,
    candidates: Vec<S14StarfoldConstellationCandidate>,
) -> Result<Vec<Arc<S14StarfoldConstellationPacket>>> {
    validate_packet_contract(
        window_capacity_bytes,
        descriptor_alignment,
        u64::from(window_capacity_bytes),
    )?;
    if candidates.is_empty() {
        bail!("S14 StarFold 星座包候选不能为空");
    }
    let mut experts = BTreeSet::new();
    let mut routes = BTreeSet::new();
    for candidate in &candidates {
        validate_candidate(candidate, layer, projection, window_capacity_bytes)?;
        if !experts.insert(candidate.expert_id) {
            bail!("S14 StarFold 星座包同投影专家重复");
        }
        for lane in &candidate.lanes {
            if !routes.insert((lane.lane, lane.route_rank)) {
                bail!("S14 StarFold 星座包 lane/rank 重复");
            }
        }
    }
    let expected_routes = STARFOLD_B4_LANES * STARFOLD_TOP_K;
    if routes.len() != expected_routes
        || (0..STARFOLD_B4_LANES)
            .any(|lane| (0..STARFOLD_TOP_K).any(|rank| !routes.contains(&(lane as u8, rank as u8))))
    {
        bail!("S14 StarFold 星座包未精确覆盖 B4 top-6 的24条权威 route");
    }

    let mut groups = Vec::<Vec<S14StarfoldConstellationCandidate>>::new();
    let mut current = Vec::new();
    let mut cursor = 0u64;
    for candidate in candidates {
        let offset = align_up(cursor, descriptor_alignment)?;
        let end = offset
            .checked_add(candidate.proof.byte_len())
            .context("S14 StarFold 星座包候选 end overflow")?;
        if end > u64::from(window_capacity_bytes) && !current.is_empty() {
            groups.push(std::mem::take(&mut current));
            cursor = 0;
        }
        let offset = align_up(cursor, descriptor_alignment)?;
        let end = offset
            .checked_add(candidate.proof.byte_len())
            .context("S14 StarFold 星座包候选 end overflow")?;
        if end > u64::from(window_capacity_bytes) {
            bail!("S14 StarFold 单个星座成员放不进 window");
        }
        cursor = end;
        current.push(candidate);
    }
    if !current.is_empty() {
        groups.push(current);
    }

    let mut packets = Vec::with_capacity(groups.len());
    for (ordinal, group) in groups.into_iter().enumerate() {
        let ordinal = u16::try_from(ordinal).context("S14 StarFold 星座包 ordinal 超出 u16")?;
        packets.push(Arc::new(build_packet(
            layer,
            base_position,
            projection,
            ordinal,
            window_capacity_bytes,
            descriptor_alignment,
            group,
        )?));
    }
    Ok(packets)
}

#[derive(Debug)]
pub struct S14StarfoldConstellationReadyPacket {
    binding: S14StarfoldReadyBinding<S14StarfoldConstellationPacketKey>,
    packet: Arc<S14StarfoldConstellationPacket>,
}

impl S14StarfoldConstellationReadyPacket {
    pub fn new(
        binding: S14StarfoldReadyBinding<S14StarfoldConstellationPacketKey>,
        packet: Arc<S14StarfoldConstellationPacket>,
    ) -> Result<Self> {
        packet.validate()?;
        if binding.key() != packet.key() || binding.byte_len() != packet.payload_bytes {
            bail!("S14 StarFold 星座包 ready binding identity/bytes 漂移");
        }
        Ok(Self { binding, packet })
    }

    pub fn into_parts(
        self,
    ) -> (
        S14StarfoldReadyBinding<S14StarfoldConstellationPacketKey>,
        Arc<S14StarfoldConstellationPacket>,
    ) {
        (self.binding, self.packet)
    }
}

/// 主线接入星座包所需的最小 runtime hook。现有 runtime 的窗口 key 仍是单页
/// `StarfoldPageKey`，因此本 trait 暂不为它提供伪实现；主线应把 resident key 提升为
/// `Microtile | Constellation` 后，在同一 A/B owner 上实现这些方法。
pub trait S14StarfoldConstellationRuntimeHook {
    fn begin_constellation_epoch(&mut self, epoch: u64) -> Result<()>;

    fn upload_constellation_packet_in_epoch(
        &mut self,
        epoch: u64,
        packet: Arc<S14StarfoldConstellationPacket>,
    ) -> Result<S14StarfoldConstellationReadyPacket>;

    fn constellation_windows_mut(
        &mut self,
    ) -> Result<&mut S14StarfoldVulkanWindows<S14StarfoldConstellationPacketKey>>;

    fn drain_constellation_epoch(&mut self, epoch: u64) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldConstellationMemberReceipt {
    pub expert_id: u16,
    pub source_key: StarfoldPageKey,
    pub window_offset: u64,
    pub payload_bytes: u64,
    pub lane_dispatches: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldConstellationPacketReceipt {
    pub contract_version: u32,
    pub key: S14StarfoldConstellationPacketKey,
    pub window: S14StarfoldWindowId,
    pub window_generation: u64,
    pub packet_bytes: u64,
    pub logical_payload_bytes: u64,
    pub members: Vec<S14StarfoldConstellationMemberReceipt>,
    pub transfer_submit_calls: u32,
    pub compute_submit_calls: u32,
    pub serial_token_forward_calls: u32,
}

impl S14StarfoldConstellationPacketReceipt {
    pub fn validate(&self) -> Result<()> {
        let member_bytes = self.members.iter().try_fold(0u64, |sum, member| {
            sum.checked_add(member.payload_bytes)
                .context("S14 StarFold 星座回执 member bytes overflow")
        })?;
        let lane_dispatches = self.members.iter().try_fold(0u32, |sum, member| {
            sum.checked_add(member.lane_dispatches)
                .context("S14 StarFold 星座回执 lane dispatch overflow")
        })?;
        if self.contract_version != S14_STARFOLD_CONSTELLATION_CONTRACT_VERSION
            || self.key.identity_sha256 == [0; 32]
            || self.window_generation == 0
            || self.packet_bytes == 0
            || member_bytes != self.logical_payload_bytes
            || self.logical_payload_bytes > self.packet_bytes
            || self.members.is_empty()
            || lane_dispatches == 0
            || self.transfer_submit_calls != 1
            || self.compute_submit_calls != 1
            || self.serial_token_forward_calls != 0
        {
            bail!("S14 StarFold 星座包回执不能证明一次 upload→一次 compute");
        }
        Ok(())
    }
}

fn build_packet(
    layer: u16,
    base_position: u64,
    projection: S14StarfoldExpertProjection,
    packet_ordinal: u16,
    window_capacity_bytes: u32,
    descriptor_alignment: u64,
    candidates: Vec<S14StarfoldConstellationCandidate>,
) -> Result<S14StarfoldConstellationPacket> {
    let mut cursor = 0u64;
    let mut logical_payload_bytes = 0u64;
    let mut members = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let window_offset = align_up(cursor, descriptor_alignment)?;
        let payload_bytes = candidate.proof.byte_len();
        cursor = window_offset
            .checked_add(payload_bytes)
            .context("S14 StarFold 星座包 cursor overflow")?;
        logical_payload_bytes = logical_payload_bytes
            .checked_add(payload_bytes)
            .context("S14 StarFold 星座包 logical bytes overflow")?;
        members.push(S14StarfoldConstellationMember {
            expert_id: candidate.expert_id,
            source_key: candidate.proof.key(),
            shape: candidate.shape,
            tile_index: candidate.tile_index,
            window_offset,
            payload_bytes,
            scale_audit: candidate.scale_audit,
            proof: candidate.proof,
            lanes: candidate.lanes,
        });
    }
    let payload_capacity =
        usize::try_from(cursor).context("S14 StarFold 星座包 payload bytes 超出 usize")?;
    let mut payload = vec![0u8; payload_capacity];
    for member in &members {
        let start = usize::try_from(member.window_offset)
            .context("S14 StarFold 星座包 copy offset 超出 usize")?;
        let end = start
            .checked_add(member.proof.bytes().len())
            .context("S14 StarFold 星座包 copy end overflow")?;
        payload[start..end].copy_from_slice(member.proof.bytes());
    }
    let identity_sha256 = packet_identity(
        layer,
        base_position,
        projection,
        packet_ordinal,
        window_capacity_bytes,
        descriptor_alignment,
        &members,
        &payload,
    )?;
    let packet = S14StarfoldConstellationPacket {
        key: S14StarfoldConstellationPacketKey {
            layer,
            base_position,
            projection: projection_code(projection),
            packet_ordinal,
            identity_sha256,
        },
        window_capacity_bytes,
        descriptor_alignment,
        payload_bytes: cursor,
        logical_payload_bytes,
        payload: Arc::from(payload.into_boxed_slice()),
        members,
    };
    packet.validate_fully()?;
    Ok(packet)
}

fn validate_candidate(
    candidate: &S14StarfoldConstellationCandidate,
    layer: u16,
    projection: S14StarfoldExpertProjection,
    window_capacity_bytes: u32,
) -> Result<()> {
    let spec = candidate.shape.tile(candidate.tile_index)?;
    let key = candidate.proof.key();
    let expected_audit = S14StarfoldMxfp4ScaleAudit::scan_host_payload(
        candidate.shape,
        candidate.tile_index,
        candidate.proof.bytes(),
    )?;
    if candidate.projection != projection
        || candidate.shape.window_capacity_bytes() != window_capacity_bytes
        || candidate.shape.tile_count() != 1
        || candidate.tile_index != 0
        || candidate.expert_id != key.expert_id
        || key.layer != layer
        || key.segment != projection.weight_segment()
        || key.tile_index != candidate.tile_index
        || candidate.proof.packed_mxfp4().is_none()
        || candidate.proof.byte_len() != spec.payload_bytes
        || candidate.scale_audit != expected_audit
        || candidate.lanes.is_empty()
        || candidate.lanes.len() > STARFOLD_B4_LANES
    {
        bail!("S14 StarFold 星座候选 proof/shape/expert identity 漂移");
    }
    let mut lanes = BTreeSet::new();
    for lane in &candidate.lanes {
        if usize::from(lane.lane) >= STARFOLD_B4_LANES
            || usize::from(lane.route_rank) >= STARFOLD_TOP_K
            || !f32::from_bits(lane.route_weight_bits).is_finite()
            || !lanes.insert(lane.lane)
        {
            bail!("S14 StarFold 星座候选 lane/rank/weight 非法");
        }
    }
    Ok(())
}

fn validate_member_structure(
    member: &S14StarfoldConstellationMember,
    window_capacity_bytes: u32,
    descriptor_alignment: u64,
) -> Result<()> {
    let spec = member.shape.tile(member.tile_index)?;
    if member.expert_id != member.source_key.expert_id
        || member.source_key != member.proof.key()
        || member.payload_bytes != member.proof.byte_len()
        || member.payload_bytes != spec.payload_bytes
        || member.window_offset % descriptor_alignment != 0
        || member.shape.window_capacity_bytes() != window_capacity_bytes
        || member.lanes.is_empty()
        || member.lanes.len() > STARFOLD_B4_LANES
    {
        bail!("S14 StarFold 星座包成员合同漂移");
    }
    let mut lanes = BTreeSet::new();
    for lane in &member.lanes {
        if usize::from(lane.lane) >= STARFOLD_B4_LANES
            || usize::from(lane.route_rank) >= STARFOLD_TOP_K
            || !f32::from_bits(lane.route_weight_bits).is_finite()
            || !lanes.insert(lane.lane)
        {
            bail!("S14 StarFold 星座包成员 lane/rank/weight 非法");
        }
    }
    Ok(())
}

fn validate_packet_contract(
    window_capacity_bytes: u32,
    descriptor_alignment: u64,
    payload_bytes: u64,
) -> Result<()> {
    if !(S14_STARFOLD_CONSTELLATION_MIN_WINDOW_BYTES..=S14_STARFOLD_CONSTELLATION_MAX_WINDOW_BYTES)
        .contains(&window_capacity_bytes)
        || !window_capacity_bytes.is_power_of_two()
        || descriptor_alignment == 0
        || !descriptor_alignment.is_power_of_two()
        || descriptor_alignment > u64::from(window_capacity_bytes)
        || payload_bytes == 0
        || payload_bytes > u64::from(window_capacity_bytes)
    {
        bail!("S14 StarFold 星座包只允许 16/32/64 MiB 动态窗口与合法 descriptor alignment");
    }
    Ok(())
}

fn packet_identity(
    layer: u16,
    base_position: u64,
    projection: S14StarfoldExpertProjection,
    packet_ordinal: u16,
    window_capacity_bytes: u32,
    descriptor_alignment: u64,
    members: &[S14StarfoldConstellationMember],
    payload: &[u8],
) -> Result<[u8; 32]> {
    let mut sha = Sha256::new();
    sha.update(b"polaris-s14-starfold-constellation-packet-v1\0");
    sha.update(S14_STARFOLD_CONSTELLATION_CONTRACT_VERSION.to_le_bytes());
    sha.update(layer.to_le_bytes());
    sha.update(base_position.to_le_bytes());
    sha.update([projection_code(projection)]);
    sha.update(packet_ordinal.to_le_bytes());
    sha.update(window_capacity_bytes.to_le_bytes());
    sha.update(descriptor_alignment.to_le_bytes());
    sha.update(
        u32::try_from(members.len())
            .context("S14 StarFold 星座包 member count 超出 u32")?
            .to_le_bytes(),
    );
    for member in members {
        sha.update(member.expert_id.to_le_bytes());
        sha.update(member.source_key.layer.to_le_bytes());
        sha.update(member.source_key.expert_id.to_le_bytes());
        sha.update([member.source_key.segment as u8]);
        sha.update(member.source_key.tile_index.to_le_bytes());
        sha.update(member.tile_index.to_le_bytes());
        sha.update(member.window_offset.to_le_bytes());
        sha.update(member.payload_bytes.to_le_bytes());
        let packed = member
            .proof
            .packed_mxfp4()
            .context("S14 StarFold 星座 identity 缺少 packed MXFP4 proof")?;
        for proof in [packed.weight_proof(), packed.scale_proof()] {
            update_identity_bytes(&mut sha, proof.asset().sha256.as_bytes())?;
            update_identity_bytes(&mut sha, proof.asset().proof_sha256.as_bytes())?;
            update_identity_bytes(&mut sha, proof.asset().range_key.as_bytes())?;
            update_identity_bytes(&mut sha, proof.source().planned.tensor.as_bytes())?;
            sha.update(proof.source().span.source_segment_offset.to_le_bytes());
            sha.update(proof.source().span.byte_len.to_le_bytes());
        }
        sha.update(
            u32::try_from(member.lanes.len())
                .context("S14 StarFold 星座包 lane count 超出 u32")?
                .to_le_bytes(),
        );
        for lane in &member.lanes {
            sha.update([lane.lane, lane.route_rank]);
            sha.update(lane.route_weight_bits.to_le_bytes());
        }
    }
    sha.update(payload);
    Ok(sha.finalize().into())
}

fn update_identity_bytes(sha: &mut Sha256, bytes: &[u8]) -> Result<()> {
    sha.update(
        u64::try_from(bytes.len())
            .context("S14 StarFold 星座 identity field bytes 超出 u64")?
            .to_le_bytes(),
    );
    sha.update(bytes);
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("S14 StarFold 星座包 alignment overflow")
}

const fn projection_code(projection: S14StarfoldExpertProjection) -> u8 {
    match projection {
        S14StarfoldExpertProjection::W1 => 0,
        S14StarfoldExpertProjection::W3 => 1,
        S14StarfoldExpertProjection::W2 => 2,
    }
}

fn projection_from_code(code: u8) -> Result<S14StarfoldExpertProjection> {
    match code {
        0 => Ok(S14StarfoldExpertProjection::W1),
        1 => Ok(S14StarfoldExpertProjection::W3),
        2 => Ok(S14StarfoldExpertProjection::W2),
        _ => bail!("S14 StarFold 星座包 projection code 非法"),
    }
}
