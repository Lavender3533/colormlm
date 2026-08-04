//! S14 StarFold 的有界跨层预取计划与租约。
//!
//! production FullDepth43 的动态 router 使下一层专家集合依赖当前层输出。因此这里不
//! 预测专家，也不启动 I/O：调用方只有在持有真实 route receipt 后，才能提交 L+1
//! routed RAM materialize 意图；L+2 只接受不依赖专家路由的 static page 意图。
//!
//! 每个租约都绑定 block sequence、plan generation、真实网络层和精确 expert set（static
//! 租约的 expert set 按合同为空）。RAM 与 SSD in-flight 字节由同一个 ledger 原子预留；
//! cancel/fail/consume 或未显式收口的 Drop 都会回收预算。

use anyhow::{bail, Context, Result};
use polaris_s14_runner::{FULL_DEPTH_LAYERS, N_ROUTED_EXPERTS};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, MutexGuard},
};

pub const S14_STARFOLD_PREFETCH_CONTRACT_VERSION: u32 = 1;
pub const S14_STARFOLD_PREFETCH_PACKET_RAM_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
pub const S14_STARFOLD_PREFETCH_STATIC_SSD_BUDGET_BYTES: u64 = 192 * 1024 * 1024;
pub const S14_STARFOLD_PREFETCH_MAX_ACTIVE_LEASES: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct S14StarfoldPrefetchLayerIdentity {
    block_sequence: u64,
    generation: u64,
    layer_ordinal: u16,
    layer: u16,
}

impl S14StarfoldPrefetchLayerIdentity {
    pub fn new(block_sequence: u64, generation: u64, layer_ordinal: usize) -> Result<Self> {
        if generation == 0 {
            bail!("S14 StarFold prefetch generation 不能为0");
        }
        let &layer = FULL_DEPTH_LAYERS
            .get(layer_ordinal)
            .context("S14 StarFold prefetch layer ordinal 越出 FullDepth43")?;
        Ok(Self {
            block_sequence,
            generation,
            layer_ordinal: u16::try_from(layer_ordinal)
                .context("S14 StarFold prefetch layer ordinal 超出 u16")?,
            layer: u16::from(layer),
        })
    }

    pub const fn block_sequence(self) -> u64 {
        self.block_sequence
    }

    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn layer_ordinal(self) -> u16 {
        self.layer_ordinal
    }

    pub const fn layer(self) -> u16 {
        self.layer
    }

    fn same_block_generation(self, other: Self) -> bool {
        self.block_sequence == other.block_sequence && self.generation == other.generation
    }

    fn is_ahead_of(self, current: Self, distance: u16) -> bool {
        self.same_block_generation(current)
            && current.layer_ordinal.checked_add(distance) == Some(self.layer_ordinal)
    }
}

/// 真实 route receipt 的 canonical expert union。输入允许同一专家被多个 lane 重复选择，
/// 但保存时只保留排序后的唯一 expert ID。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldRoutedExpertSet {
    expert_ids: Vec<u16>,
}

impl S14StarfoldRoutedExpertSet {
    pub fn from_route_experts(expert_ids: impl IntoIterator<Item = u16>) -> Result<Self> {
        let mut canonical = expert_ids.into_iter().collect::<Vec<_>>();
        if canonical.is_empty() {
            bail!("S14 StarFold routed prefetch expert set 不能为空");
        }
        if canonical
            .iter()
            .any(|&expert_id| expert_id >= N_ROUTED_EXPERTS)
        {
            bail!("S14 StarFold routed prefetch expert ID 越界");
        }
        canonical.sort_unstable();
        canonical.dedup();
        Ok(Self {
            expert_ids: canonical,
        })
    }

    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldActiveComputeIdentity {
    layer: S14StarfoldPrefetchLayerIdentity,
    routed_experts: S14StarfoldRoutedExpertSet,
}

impl S14StarfoldActiveComputeIdentity {
    pub fn new(
        layer: S14StarfoldPrefetchLayerIdentity,
        routed_experts: S14StarfoldRoutedExpertSet,
    ) -> Self {
        Self {
            layer,
            routed_experts,
        }
    }

    pub const fn layer(&self) -> S14StarfoldPrefetchLayerIdentity {
        self.layer
    }

    pub fn routed_experts(&self) -> &S14StarfoldRoutedExpertSet {
        &self.routed_experts
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldRoutedRamIntent {
    layer: S14StarfoldPrefetchLayerIdentity,
    routed_experts: S14StarfoldRoutedExpertSet,
    materialize_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum S14StarfoldPacketProjection {
    W1,
    W3,
    W2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldSameLayerPacketIntent {
    layer: S14StarfoldPrefetchLayerIdentity,
    projection: S14StarfoldPacketProjection,
    packet_ordinal: u16,
    routed_experts: S14StarfoldRoutedExpertSet,
    materialize_bytes: u64,
}

impl S14StarfoldSameLayerPacketIntent {
    pub fn new(
        layer: S14StarfoldPrefetchLayerIdentity,
        projection: S14StarfoldPacketProjection,
        packet_ordinal: u16,
        routed_experts: S14StarfoldRoutedExpertSet,
        materialize_bytes: u64,
    ) -> Result<Self> {
        if materialize_bytes == 0
            || materialize_bytes > S14_STARFOLD_PREFETCH_PACKET_RAM_BUDGET_BYTES
        {
            bail!("S14 StarFold same-layer packet prefetch bytes 越出单窗口预算");
        }
        Ok(Self {
            layer,
            projection,
            packet_ordinal,
            routed_experts,
            materialize_bytes,
        })
    }

    pub const fn layer(&self) -> S14StarfoldPrefetchLayerIdentity {
        self.layer
    }

    pub const fn projection(&self) -> S14StarfoldPacketProjection {
        self.projection
    }

    pub const fn packet_ordinal(&self) -> u16 {
        self.packet_ordinal
    }

    pub fn routed_experts(&self) -> &S14StarfoldRoutedExpertSet {
        &self.routed_experts
    }

    pub const fn materialize_bytes(&self) -> u64 {
        self.materialize_bytes
    }
}

impl S14StarfoldRoutedRamIntent {
    pub fn new(
        layer: S14StarfoldPrefetchLayerIdentity,
        routed_experts: S14StarfoldRoutedExpertSet,
        materialize_bytes: u64,
    ) -> Result<Self> {
        if materialize_bytes == 0 {
            bail!("S14 StarFold routed RAM prefetch bytes 不能为0");
        }
        Ok(Self {
            layer,
            routed_experts,
            materialize_bytes,
        })
    }

    pub const fn layer(&self) -> S14StarfoldPrefetchLayerIdentity {
        self.layer
    }

    pub fn routed_experts(&self) -> &S14StarfoldRoutedExpertSet {
        &self.routed_experts
    }

    pub const fn materialize_bytes(&self) -> u64 {
        self.materialize_bytes
    }
}

/// 只能描述不依赖动态 expert route 的层静态资产；这里刻意没有 routed-expert 变体。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum S14StarfoldStaticPageClass {
    Attention,
    Router,
    SharedExpert,
    Normalization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldStaticPageIntent {
    class: S14StarfoldStaticPageClass,
    asset_key: String,
    bytes: u64,
}

impl S14StarfoldStaticPageIntent {
    pub fn new(
        class: S14StarfoldStaticPageClass,
        asset_key: impl Into<String>,
        bytes: u64,
    ) -> Result<Self> {
        let asset_key = asset_key.into();
        if asset_key.trim().is_empty() || bytes == 0 {
            bail!("S14 StarFold static page prefetch 要求非空 asset key/bytes");
        }
        Ok(Self {
            class,
            asset_key,
            bytes,
        })
    }

    pub const fn class(&self) -> S14StarfoldStaticPageClass {
        self.class
    }

    pub fn asset_key(&self) -> &str {
        &self.asset_key
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldStaticSsdIntent {
    layer: S14StarfoldPrefetchLayerIdentity,
    pages: Vec<S14StarfoldStaticPageIntent>,
    fetch_bytes: u64,
}

impl S14StarfoldStaticSsdIntent {
    pub fn new(
        layer: S14StarfoldPrefetchLayerIdentity,
        pages: Vec<S14StarfoldStaticPageIntent>,
    ) -> Result<Self> {
        if pages.is_empty() {
            bail!("S14 StarFold static SSD prefetch pages 不能为空");
        }
        let mut keys = BTreeSet::new();
        let mut fetch_bytes = 0u64;
        for page in &pages {
            if page.asset_key.trim().is_empty() || page.bytes == 0 {
                bail!("S14 StarFold static SSD prefetch page identity 非法");
            }
            if !keys.insert(page.asset_key.as_str()) {
                bail!("S14 StarFold static SSD prefetch asset key 重复");
            }
            fetch_bytes = fetch_bytes
                .checked_add(page.bytes)
                .context("S14 StarFold static SSD prefetch bytes overflow")?;
        }
        Ok(Self {
            layer,
            pages,
            fetch_bytes,
        })
    }

    pub const fn fetch_bytes(&self) -> u64 {
        self.fetch_bytes
    }

    pub const fn layer(&self) -> S14StarfoldPrefetchLayerIdentity {
        self.layer
    }

    pub fn pages(&self) -> &[S14StarfoldStaticPageIntent] {
        &self.pages
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldStaticMaterializeReceipt {
    pub layer: S14StarfoldPrefetchLayerIdentity,
    pub assets: usize,
    pub bytes: u64,
}

impl S14StarfoldStaticMaterializeReceipt {
    pub fn validate_for(&self, intent: &S14StarfoldStaticSsdIntent) -> Result<()> {
        if self.layer != intent.layer
            || self.assets != intent.pages.len()
            || self.bytes != intent.fetch_bytes
        {
            bail!("S14 StarFold static prefetch materialize receipt 与 intent 漂移");
        }
        Ok(())
    }
}

/// 当前计算 L 时允许发出的唯一两类意图：已知 route 的 L+1 RAM materialize，以及
/// 与 route 无关的 L+2 static SSD fetch。两个字段都为空没有调度价值，直接拒绝。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S14StarfoldPrefetchWindowRequest {
    current_compute: S14StarfoldActiveComputeIdentity,
    routed_l1: Option<S14StarfoldRoutedRamIntent>,
    static_l2: Option<S14StarfoldStaticSsdIntent>,
}

impl S14StarfoldPrefetchWindowRequest {
    pub fn new(
        current_compute: S14StarfoldActiveComputeIdentity,
        routed_l1: Option<S14StarfoldRoutedRamIntent>,
        static_l2: Option<S14StarfoldStaticSsdIntent>,
    ) -> Result<Self> {
        if routed_l1.is_none() && static_l2.is_none() {
            bail!("S14 StarFold prefetch window 没有可证明的预取意图");
        }
        if let Some(next) = &routed_l1 {
            if !next.layer.is_ahead_of(current_compute.layer, 1) {
                bail!("S14 StarFold routed RAM prefetch 必须绑定同 block/generation 的 L+1");
            }
        }
        if let Some(two_ahead) = &static_l2 {
            if !two_ahead.layer.is_ahead_of(current_compute.layer, 2) {
                bail!("S14 StarFold static SSD prefetch 必须绑定同 block/generation 的 L+2");
            }
        }
        Ok(Self {
            current_compute,
            routed_l1,
            static_l2,
        })
    }

    pub fn current_compute(&self) -> &S14StarfoldActiveComputeIdentity {
        &self.current_compute
    }

    pub fn routed_l1(&self) -> Option<&S14StarfoldRoutedRamIntent> {
        self.routed_l1.as_ref()
    }

    pub fn static_l2(&self) -> Option<&S14StarfoldStaticSsdIntent> {
        self.static_l2.as_ref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldPrefetchBudget {
    max_ram_materialize_bytes: u64,
    max_ssd_fetch_bytes: u64,
    max_active_leases: usize,
}

impl S14StarfoldPrefetchBudget {
    pub fn new(
        max_ram_materialize_bytes: u64,
        max_ssd_fetch_bytes: u64,
        max_active_leases: usize,
    ) -> Result<Self> {
        if max_active_leases == 0 || (max_ram_materialize_bytes == 0 && max_ssd_fetch_bytes == 0) {
            bail!("S14 StarFold prefetch budget 必须启用至少一个 tier 和一个 lease");
        }
        Ok(Self {
            max_ram_materialize_bytes,
            max_ssd_fetch_bytes,
            max_active_leases,
        })
    }

    pub const fn max_ram_materialize_bytes(self) -> u64 {
        self.max_ram_materialize_bytes
    }

    pub const fn max_ssd_fetch_bytes(self) -> u64 {
        self.max_ssd_fetch_bytes
    }

    pub const fn max_active_leases(self) -> usize {
        self.max_active_leases
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldPrefetchTier {
    RoutedRamL1,
    SameLayerPacketRam,
    StaticSsdL2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S14StarfoldPrefetchTarget {
    RoutedExperts(S14StarfoldRoutedExpertSet),
    ConstellationPacket {
        projection: S14StarfoldPacketProjection,
        packet_ordinal: u16,
        routed_experts: S14StarfoldRoutedExpertSet,
    },
    StaticPages(Vec<S14StarfoldStaticPageIntent>),
}

impl S14StarfoldPrefetchTarget {
    pub fn expert_ids(&self) -> &[u16] {
        match self {
            Self::RoutedExperts(experts) => experts.expert_ids(),
            Self::ConstellationPacket { routed_experts, .. } => routed_experts.expert_ids(),
            Self::StaticPages(_) => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActiveLeasePhase {
    Issued,
    Ready,
    CancelRequested,
}

#[derive(Clone, Copy, Debug)]
struct ActiveLeaseRecord {
    identity: S14StarfoldPrefetchLayerIdentity,
    tier: S14StarfoldPrefetchTier,
    reserved_bytes: u64,
    phase: ActiveLeasePhase,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S14StarfoldPrefetchSnapshot {
    pub reserved_ram_materialize_bytes: u64,
    pub reserved_ssd_fetch_bytes: u64,
    pub active_leases: usize,
    pub ready_leases: usize,
    pub issued_leases: u64,
    pub consumed_leases: u64,
    pub cancelled_leases: u64,
    pub failed_leases: u64,
    pub dropped_leases: u64,
    pub cancellation_requests: u64,
}

#[derive(Debug)]
struct PrefetchLedger {
    next_lease_id: u64,
    active: BTreeMap<u64, ActiveLeaseRecord>,
    snapshot: S14StarfoldPrefetchSnapshot,
}

impl Default for PrefetchLedger {
    fn default() -> Self {
        Self {
            next_lease_id: 1,
            active: BTreeMap::new(),
            snapshot: S14StarfoldPrefetchSnapshot::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct S14StarfoldPrefetchPlanner {
    budget: S14StarfoldPrefetchBudget,
    ledger: Arc<Mutex<PrefetchLedger>>,
}

impl S14StarfoldPrefetchPlanner {
    pub fn new(budget: S14StarfoldPrefetchBudget) -> Self {
        Self {
            budget,
            ledger: Arc::new(Mutex::new(PrefetchLedger::default())),
        }
    }

    pub const fn budget(&self) -> S14StarfoldPrefetchBudget {
        self.budget
    }

    pub fn snapshot(&self) -> Result<S14StarfoldPrefetchSnapshot> {
        let ledger = self.lock()?;
        let mut snapshot = ledger.snapshot;
        snapshot.active_leases = ledger.active.len();
        snapshot.ready_leases = ledger
            .active
            .values()
            .filter(|record| record.phase == ActiveLeasePhase::Ready)
            .count();
        Ok(snapshot)
    }

    pub fn production_defaults() -> Result<Self> {
        Ok(Self::new(S14StarfoldPrefetchBudget::new(
            S14_STARFOLD_PREFETCH_PACKET_RAM_BUDGET_BYTES,
            S14_STARFOLD_PREFETCH_STATIC_SSD_BUDGET_BYTES,
            S14_STARFOLD_PREFETCH_MAX_ACTIVE_LEASES,
        )?))
    }

    /// 同层 route 已经权威确定后，给一个 constellation packet 预留有界 producer RAM。
    /// 该租约只覆盖“已物化、尚未提交”的 host 队列；提交进入固定 A/B window 后 consume。
    pub fn issue_same_layer_packet(
        &self,
        intent: S14StarfoldSameLayerPacketIntent,
    ) -> Result<S14StarfoldPrefetchLease> {
        self.issue_one(
            intent.layer,
            S14StarfoldPrefetchTier::SameLayerPacketRam,
            S14StarfoldPrefetchTarget::ConstellationPacket {
                projection: intent.projection,
                packet_ordinal: intent.packet_ordinal,
                routed_experts: intent.routed_experts,
            },
            intent.materialize_bytes,
        )
    }

    /// 原子签发一个 lookahead window。预算不足时不会留下半个租约。
    pub fn issue_window(
        &self,
        request: S14StarfoldPrefetchWindowRequest,
    ) -> Result<S14StarfoldPrefetchWindowLeases> {
        let mut pending = Vec::with_capacity(2);
        if let Some(intent) = request.routed_l1 {
            pending.push((
                intent.layer,
                S14StarfoldPrefetchTier::RoutedRamL1,
                S14StarfoldPrefetchTarget::RoutedExperts(intent.routed_experts),
                intent.materialize_bytes,
            ));
        }
        if let Some(intent) = request.static_l2 {
            pending.push((
                intent.layer,
                S14StarfoldPrefetchTier::StaticSsdL2,
                S14StarfoldPrefetchTarget::StaticPages(intent.pages),
                intent.fetch_bytes,
            ));
        }

        let requested_ram = pending
            .iter()
            .filter(|(_, tier, _, _)| *tier == S14StarfoldPrefetchTier::RoutedRamL1)
            .try_fold(0u64, |sum, (_, _, _, bytes)| sum.checked_add(*bytes))
            .context("S14 StarFold prefetch requested RAM bytes overflow")?;
        let requested_ssd = pending
            .iter()
            .filter(|(_, tier, _, _)| *tier == S14StarfoldPrefetchTier::StaticSsdL2)
            .try_fold(0u64, |sum, (_, _, _, bytes)| sum.checked_add(*bytes))
            .context("S14 StarFold prefetch requested SSD bytes overflow")?;

        let mut ledger = self.lock()?;
        let pending_count = u64::try_from(pending.len())
            .context("S14 StarFold prefetch pending lease count 超出 u64")?;
        ledger
            .next_lease_id
            .checked_add(pending_count)
            .context("S14 StarFold prefetch lease ID overflow")?;
        let next_ram = ledger
            .snapshot
            .reserved_ram_materialize_bytes
            .checked_add(requested_ram)
            .context("S14 StarFold prefetch RAM reservation overflow")?;
        let next_ssd = ledger
            .snapshot
            .reserved_ssd_fetch_bytes
            .checked_add(requested_ssd)
            .context("S14 StarFold prefetch SSD reservation overflow")?;
        if next_ram > self.budget.max_ram_materialize_bytes
            || next_ssd > self.budget.max_ssd_fetch_bytes
            || ledger.active.len().saturating_add(pending.len()) > self.budget.max_active_leases
        {
            bail!("S14 StarFold prefetch window 超出 RAM/SSD/lease 有界预算");
        }

        let mut leases = Vec::with_capacity(pending.len());
        for (identity, tier, target, reserved_bytes) in pending {
            let lease_id = ledger.next_lease_id;
            ledger.next_lease_id = ledger
                .next_lease_id
                .checked_add(1)
                .context("S14 StarFold prefetch lease ID overflow")?;
            ledger.active.insert(
                lease_id,
                ActiveLeaseRecord {
                    identity,
                    tier,
                    reserved_bytes,
                    phase: ActiveLeasePhase::Issued,
                },
            );
            ledger.snapshot.issued_leases = ledger.snapshot.issued_leases.saturating_add(1);
            leases.push(S14StarfoldPrefetchLease {
                lease_id,
                identity,
                tier,
                target,
                ledger: Arc::clone(&self.ledger),
                terminal: false,
            });
        }
        ledger.snapshot.reserved_ram_materialize_bytes = next_ram;
        ledger.snapshot.reserved_ssd_fetch_bytes = next_ssd;
        drop(ledger);

        let mut routed_l1 = None;
        let mut static_l2 = None;
        for lease in leases {
            match lease.tier {
                S14StarfoldPrefetchTier::RoutedRamL1 => routed_l1 = Some(lease),
                S14StarfoldPrefetchTier::SameLayerPacketRam => {
                    bail!("S14 StarFold cross-layer window 混入 same-layer packet lease")
                }
                S14StarfoldPrefetchTier::StaticSsdL2 => static_l2 = Some(lease),
            }
        }
        Ok(S14StarfoldPrefetchWindowLeases {
            current_compute: request.current_compute,
            routed_l1,
            static_l2,
        })
    }

    fn issue_one(
        &self,
        identity: S14StarfoldPrefetchLayerIdentity,
        tier: S14StarfoldPrefetchTier,
        target: S14StarfoldPrefetchTarget,
        reserved_bytes: u64,
    ) -> Result<S14StarfoldPrefetchLease> {
        if reserved_bytes == 0 {
            bail!("S14 StarFold prefetch single lease bytes 不能为0");
        }
        let mut ledger = self.lock()?;
        let next_ram = ledger.snapshot.reserved_ram_materialize_bytes.checked_add(
            if matches!(
                tier,
                S14StarfoldPrefetchTier::RoutedRamL1 | S14StarfoldPrefetchTier::SameLayerPacketRam
            ) {
                reserved_bytes
            } else {
                0
            },
        );
        let next_ssd = ledger.snapshot.reserved_ssd_fetch_bytes.checked_add(
            if tier == S14StarfoldPrefetchTier::StaticSsdL2 {
                reserved_bytes
            } else {
                0
            },
        );
        let (Some(next_ram), Some(next_ssd)) = (next_ram, next_ssd) else {
            bail!("S14 StarFold prefetch single lease reservation overflow");
        };
        if next_ram > self.budget.max_ram_materialize_bytes
            || next_ssd > self.budget.max_ssd_fetch_bytes
            || ledger.active.len() >= self.budget.max_active_leases
        {
            bail!("S14 StarFold prefetch single lease 超出有界预算");
        }
        let lease_id = ledger.next_lease_id;
        ledger.next_lease_id = ledger
            .next_lease_id
            .checked_add(1)
            .context("S14 StarFold prefetch lease ID overflow")?;
        ledger.active.insert(
            lease_id,
            ActiveLeaseRecord {
                identity,
                tier,
                reserved_bytes,
                phase: ActiveLeasePhase::Issued,
            },
        );
        ledger.snapshot.reserved_ram_materialize_bytes = next_ram;
        ledger.snapshot.reserved_ssd_fetch_bytes = next_ssd;
        ledger.snapshot.issued_leases = ledger.snapshot.issued_leases.saturating_add(1);
        drop(ledger);
        Ok(S14StarfoldPrefetchLease {
            lease_id,
            identity,
            tier,
            target,
            ledger: Arc::clone(&self.ledger),
            terminal: false,
        })
    }

    /// 请求取消或 block generation 被替换时，批量撤销该 generation 的所有预取。
    /// 这里只置取消位，不提前归还仍可能被 worker 使用的预算；worker 观察取消后必须
    /// `cancel` 或 Drop lease，届时才完成物理回收。
    pub fn cancel_generation(&self, block_sequence: u64, generation: u64) -> Result<usize> {
        let mut ledger = self.lock()?;
        let ids = ledger
            .active
            .iter()
            .filter_map(|(&lease_id, record)| {
                (record.identity.block_sequence == block_sequence
                    && record.identity.generation == generation)
                    .then_some((lease_id, record.phase))
            })
            .filter_map(|(lease_id, phase)| {
                (phase != ActiveLeasePhase::CancelRequested).then_some(lease_id)
            })
            .collect::<Vec<_>>();
        for lease_id in &ids {
            let record = ledger
                .active
                .get_mut(lease_id)
                .context("S14 StarFold generation cancel lease 消失")?;
            record.phase = ActiveLeasePhase::CancelRequested;
            ledger.snapshot.cancellation_requests =
                ledger.snapshot.cancellation_requests.saturating_add(1);
        }
        Ok(ids.len())
    }

    fn lock(&self) -> Result<MutexGuard<'_, PrefetchLedger>> {
        self.ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("S14 StarFold prefetch ledger poisoned"))
    }
}

#[derive(Debug)]
pub struct S14StarfoldPrefetchWindowLeases {
    pub current_compute: S14StarfoldActiveComputeIdentity,
    pub routed_l1: Option<S14StarfoldPrefetchLease>,
    pub static_l2: Option<S14StarfoldPrefetchLease>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldPrefetchFailurePhase {
    SsdFetch,
    ProofValidation,
    RamMaterialize,
    Consumer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S14StarfoldPrefetchTerminalOutcome {
    Consumed,
    Cancelled,
    Failed(S14StarfoldPrefetchFailurePhase),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldPrefetchReadyReceipt {
    pub lease_id: u64,
    pub tier: S14StarfoldPrefetchTier,
    pub retained_bytes: u64,
    pub reclaimed_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldPrefetchReleaseReceipt {
    pub lease_id: u64,
    pub identity: S14StarfoldPrefetchLayerIdentity,
    pub tier: S14StarfoldPrefetchTier,
    pub reclaimed_bytes: u64,
    pub outcome: S14StarfoldPrefetchTerminalOutcome,
}

#[derive(Debug)]
pub struct S14StarfoldPrefetchLease {
    lease_id: u64,
    identity: S14StarfoldPrefetchLayerIdentity,
    tier: S14StarfoldPrefetchTier,
    target: S14StarfoldPrefetchTarget,
    ledger: Arc<Mutex<PrefetchLedger>>,
    terminal: bool,
}

impl S14StarfoldPrefetchLease {
    pub const fn lease_id(&self) -> u64 {
        self.lease_id
    }

    pub const fn identity(&self) -> S14StarfoldPrefetchLayerIdentity {
        self.identity
    }

    pub const fn tier(&self) -> S14StarfoldPrefetchTier {
        self.tier
    }

    pub fn target(&self) -> &S14StarfoldPrefetchTarget {
        &self.target
    }

    /// 后台 I/O/materialize worker 可在阶段边界轮询；generation 被 planner 撤销后立即
    /// 返回 true。该方法不等待 worker，也不隐式启动新的任务。
    pub fn cancellation_requested(&self) -> Result<bool> {
        let ledger = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("S14 StarFold prefetch ledger poisoned"))?;
        Ok(ledger.active.get(&self.lease_id).map_or(true, |record| {
            record.phase == ActiveLeasePhase::CancelRequested
        }))
    }

    /// producer 完成 I/O/materialize 后保持租约。RAM ready payload 继续占预算直到消费；
    /// SSD fetch 完成后不再占 in-flight 预算，但 lease identity 仍留给 consumer 验收。
    pub fn mark_ready(&mut self, actual_bytes: u64) -> Result<S14StarfoldPrefetchReadyReceipt> {
        let mut ledger = self
            .ledger
            .lock()
            .map_err(|_| anyhow::anyhow!("S14 StarFold prefetch ledger poisoned"))?;
        let record = ledger
            .active
            .get_mut(&self.lease_id)
            .context("S14 StarFold prefetch ready lease 已失活")?;
        if record.phase != ActiveLeasePhase::Issued
            || record.tier != self.tier
            || record.identity != self.identity
        {
            bail!("S14 StarFold prefetch ready phase/tier/identity 漂移");
        }
        if actual_bytes == 0 || actual_bytes > record.reserved_bytes {
            bail!("S14 StarFold prefetch ready bytes 越出 reservation");
        }
        let retained_bytes = match self.tier {
            S14StarfoldPrefetchTier::RoutedRamL1 | S14StarfoldPrefetchTier::SameLayerPacketRam => {
                actual_bytes
            }
            S14StarfoldPrefetchTier::StaticSsdL2 => 0,
        };
        let reclaimed_bytes = record.reserved_bytes - retained_bytes;
        record.reserved_bytes = retained_bytes;
        record.phase = ActiveLeasePhase::Ready;
        release_tier_bytes(&mut ledger.snapshot, self.tier, reclaimed_bytes);
        Ok(S14StarfoldPrefetchReadyReceipt {
            lease_id: self.lease_id,
            tier: self.tier,
            retained_bytes,
            reclaimed_bytes,
        })
    }

    pub fn consume(self) -> Result<S14StarfoldPrefetchReleaseReceipt> {
        self.finish(S14StarfoldPrefetchTerminalOutcome::Consumed)
    }

    pub fn cancel(self) -> Result<S14StarfoldPrefetchReleaseReceipt> {
        self.finish(S14StarfoldPrefetchTerminalOutcome::Cancelled)
    }

    pub fn fail(
        self,
        phase: S14StarfoldPrefetchFailurePhase,
    ) -> Result<S14StarfoldPrefetchReleaseReceipt> {
        self.finish(S14StarfoldPrefetchTerminalOutcome::Failed(phase))
    }

    fn finish(
        mut self,
        outcome: S14StarfoldPrefetchTerminalOutcome,
    ) -> Result<S14StarfoldPrefetchReleaseReceipt> {
        let reclaimed_bytes = {
            let mut ledger = self
                .ledger
                .lock()
                .map_err(|_| anyhow::anyhow!("S14 StarFold prefetch ledger poisoned"))?;
            let record = *ledger
                .active
                .get(&self.lease_id)
                .context("S14 StarFold prefetch terminal lease 已失活")?;
            if record.tier != self.tier || record.identity != self.identity {
                bail!("S14 StarFold prefetch terminal tier/identity 漂移");
            }
            if outcome == S14StarfoldPrefetchTerminalOutcome::Consumed
                && record.phase != ActiveLeasePhase::Ready
            {
                bail!("S14 StarFold prefetch lease 尚未 ready，禁止 consume");
            }
            ledger.active.remove(&self.lease_id);
            release_tier_bytes(&mut ledger.snapshot, self.tier, record.reserved_bytes);
            match outcome {
                S14StarfoldPrefetchTerminalOutcome::Consumed => {
                    ledger.snapshot.consumed_leases =
                        ledger.snapshot.consumed_leases.saturating_add(1)
                }
                S14StarfoldPrefetchTerminalOutcome::Cancelled => {
                    ledger.snapshot.cancelled_leases =
                        ledger.snapshot.cancelled_leases.saturating_add(1)
                }
                S14StarfoldPrefetchTerminalOutcome::Failed(_) => {
                    ledger.snapshot.failed_leases = ledger.snapshot.failed_leases.saturating_add(1)
                }
            }
            record.reserved_bytes
        };
        self.terminal = true;
        Ok(S14StarfoldPrefetchReleaseReceipt {
            lease_id: self.lease_id,
            identity: self.identity,
            tier: self.tier,
            reclaimed_bytes,
            outcome,
        })
    }
}

impl Drop for S14StarfoldPrefetchLease {
    fn drop(&mut self) {
        if self.terminal {
            return;
        }
        if let Ok(mut ledger) = self.ledger.lock() {
            if let Some(record) = ledger.active.remove(&self.lease_id) {
                release_tier_bytes(&mut ledger.snapshot, record.tier, record.reserved_bytes);
                ledger.snapshot.cancelled_leases =
                    ledger.snapshot.cancelled_leases.saturating_add(1);
                ledger.snapshot.dropped_leases = ledger.snapshot.dropped_leases.saturating_add(1);
            }
        }
        self.terminal = true;
    }
}

fn release_tier_bytes(
    snapshot: &mut S14StarfoldPrefetchSnapshot,
    tier: S14StarfoldPrefetchTier,
    bytes: u64,
) {
    match tier {
        S14StarfoldPrefetchTier::RoutedRamL1 | S14StarfoldPrefetchTier::SameLayerPacketRam => {
            snapshot.reserved_ram_materialize_bytes = snapshot
                .reserved_ram_materialize_bytes
                .saturating_sub(bytes)
        }
        S14StarfoldPrefetchTier::StaticSsdL2 => {
            snapshot.reserved_ssd_fetch_bytes =
                snapshot.reserved_ssd_fetch_bytes.saturating_sub(bytes)
        }
    }
}
