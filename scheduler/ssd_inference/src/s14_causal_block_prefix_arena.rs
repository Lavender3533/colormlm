//! Production K=4/8 prefix checkpoint arena 的强资源与初始化生命周期。
//!
//! 该模块不计算 KV/HC/compressor/indexer 数值。它把外部 producer 已分配的单一
//! device arena 绑定为 K 份互不重叠的完整 NativeState checkpoint，并在一个 recording
//! command 中把 authoritative device state 复制到每个 prefix 的基线。只有
//! `S14CausalBlockPrefixStateProgram` 已证明所有 prefix 按 `0..=p` 累积并 seal 后，
//! terminal owner 才能取得 checkpoint slices。结构回执不能替代真实 Vulkan 数值门。

use crate::{
    s14_causal_block_prefix_state::S14CausalBlockPrefixStateSealReceipt,
    s14_causal_block_production_evidence::{
        S14CausalBlockProductionEvidenceLedger, S14CausalBlockProductionEvidenceSnapshot,
    },
    s14_causal_block_terminal_owner::S14CausalBlockOwnedBufferSlice,
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use std::{
    fmt,
    sync::{Arc, Mutex},
};

const PREFIX_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockPrefixCheckpointLayout {
    pub block_size: usize,
    pub checkpoint_state_bytes: u64,
    pub checkpoint_stride_bytes: u64,
    pub used_bytes: u64,
}

impl S14CausalBlockPrefixCheckpointLayout {
    pub fn build(block_size: usize, checkpoint_state_bytes: u64) -> Result<Self> {
        if !matches!(block_size, 4 | 8) || checkpoint_state_bytes == 0 {
            bail!("prefix checkpoint layout 只接受 K=4/8 与非零 state bytes");
        }
        let checkpoint_stride_bytes = align_up(checkpoint_state_bytes, PREFIX_ALIGNMENT)?;
        let used_bytes = checkpoint_stride_bytes
            .checked_mul(block_size.saturating_sub(1) as u64)
            .and_then(|prefix| prefix.checked_add(checkpoint_state_bytes))
            .context("prefix checkpoint arena bytes overflow")?;
        Ok(Self {
            block_size,
            checkpoint_state_bytes,
            checkpoint_stride_bytes,
            used_bytes,
        })
    }

    pub fn checkpoint_offset(self, arena_offset: u64, prefix_index: usize) -> Result<u64> {
        if prefix_index >= self.block_size {
            bail!("prefix checkpoint index 越界");
        }
        arena_offset
            .checked_add(self.checkpoint_stride_bytes * prefix_index as u64)
            .context("prefix checkpoint offset overflow")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixArenaPhase {
    Ready,
    InitializationRecorded,
    /// 43层 producer 已完成 K-prefix 累积写集，但 terminal owner 尚未
    /// 在 post-seal 窗口验收。回执必须留在强 owner 内，禁止预先导出 slices。
    PrefixProgramReceiptPublished(S14CausalBlockPrefixStateSealReceipt),
    PrefixesSealed,
    Aborted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockPrefixInitializationReceipt {
    pub base_position: u32,
    pub block_size: usize,
    pub checkpoint_state_bytes: u64,
    pub copy_regions: usize,
    pub copied_bytes: u64,
    pub serial_token_forward_calls: u32,
}

/// 外部 production allocator 的强绑定。`arena` 必须由同一 context 创建并由调用方的
/// 顶层 runtime owner 保活；本对象不从裸 Vulkan handle 重建或销毁外部 allocation。
pub struct S14CausalBlockPrefixCheckpointArena {
    context: Arc<VulkanContext>,
    arena: Arc<GpuBuffer>,
    arena_offset: u64,
    base_position: u32,
    layout: S14CausalBlockPrefixCheckpointLayout,
    phase: Mutex<PrefixArenaPhase>,
    evidence: S14CausalBlockProductionEvidenceLedger,
}

impl fmt::Debug for S14CausalBlockPrefixCheckpointArena {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockPrefixCheckpointArena")
            .field("context", &Arc::as_ptr(&self.context))
            .field("arena", &self.arena.handle())
            .field("arena_offset", &self.arena_offset)
            .field("base_position", &self.base_position)
            .field("layout", &self.layout)
            .field("phase", &self.phase.lock().ok().map(|phase| *phase))
            .finish()
    }
}

impl S14CausalBlockPrefixCheckpointArena {
    pub fn bind(
        context: Arc<VulkanContext>,
        arena: Arc<GpuBuffer>,
        arena_offset: u64,
        base_position: u32,
        block_size: usize,
        checkpoint_state_bytes: u64,
    ) -> Result<Arc<Self>> {
        let layout =
            S14CausalBlockPrefixCheckpointLayout::build(block_size, checkpoint_state_bytes)?;
        base_position
            .checked_add(block_size as u32)
            .context("prefix checkpoint position overflow")?;
        if arena.handle() == vk::Buffer::null()
            || arena_offset % PREFIX_ALIGNMENT != 0
            || arena_offset
                .checked_add(layout.used_bytes)
                .is_none_or(|end| end > arena.size())
        {
            bail!("prefix checkpoint arena handle/alignment/capacity 非法");
        }
        let evidence = S14CausalBlockProductionEvidenceLedger::new(base_position, block_size)?;
        Ok(Arc::new(Self {
            context,
            arena,
            arena_offset,
            base_position,
            layout,
            phase: Mutex::new(PrefixArenaPhase::Ready),
            evidence,
        }))
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn layout(&self) -> S14CausalBlockPrefixCheckpointLayout {
        self.layout
    }

    pub fn base_position(&self) -> u32 {
        self.base_position
    }

    pub fn buffer(&self) -> &Arc<GpuBuffer> {
        &self.arena
    }

    /// 纯 host、只读 production 证据；不泄露 Vulkan owner 或写入口。
    pub fn production_evidence_snapshot(&self) -> Result<S14CausalBlockProductionEvidenceSnapshot> {
        self.evidence.snapshot()
    }

    pub(crate) fn record_completed_ratio4_layer(
        &self,
        layer: u8,
        receipt: crate::s14_causal_block_ratio4_boundary::S14CausalBlockRatio4BoundaryRecordingReceipt,
    ) -> Result<()> {
        self.evidence.record_completed_ratio4_layer(layer, receipt)
    }

    pub fn prefix_offset(&self, prefix_index: usize) -> Result<u64> {
        self.layout
            .checkpoint_offset(self.arena_offset, prefix_index)
    }

    /// Host checkpoint finalizer 的只读门。terminal owner 必须已经在 post-seal
    /// 窗口验收 producer receipt；在此之前禁止把 device arena 当成完整 checkpoint 回读。
    pub(crate) fn validate_host_readback_ready(&self) -> Result<()> {
        if *self.lock_phase()? != PrefixArenaPhase::PrefixesSealed {
            bail!("prefix checkpoint host readback 只能发生在 terminal post-seal 验收之后");
        }
        Ok(())
    }

    /// Generation terminal owner 取得 arena 强引用前的只读门。43层 producer 此时
    /// 只能发布 seal receipt，不能越权提前把 arena 置为 `PrefixesSealed`；真正的
    /// receipt 验收、evidence 提交与 checkpoint slice 导出仍由随后唯一一次
    /// `seal_and_export_terminal_checkpoints` 原子完成。
    pub(crate) fn validate_terminal_receipt_published(&self) -> Result<()> {
        if !matches!(
            *self.lock_phase()?,
            PrefixArenaPhase::PrefixProgramReceiptPublished(_)
        ) {
            bail!("prefix checkpoint terminal owner 获取前缺少43层 producer seal receipt");
        }
        Ok(())
    }

    /// 在 caller 已开始的 command 中把同一份 authoritative state 复制为 K 个 prefix
    /// checkpoint 基线。本函数不 submit/wait；后续每个 prefix 必须在同一 queue/timeline
    /// 上按 `0..=p` 应用真实 lane 写集。
    ///
    /// # Safety
    /// `command` 必须处于 recording 状态；`authoritative` 必须在本次 command 完成前有效。
    pub unsafe fn record_authoritative_initialization(
        &self,
        context: &Arc<VulkanContext>,
        command: vk::CommandBuffer,
        authoritative: &GpuBuffer,
        authoritative_offset: u64,
    ) -> Result<S14CausalBlockPrefixInitializationReceipt> {
        if !Arc::ptr_eq(context, &self.context) {
            bail!("prefix checkpoint initializer 与 arena VulkanContext 漂移");
        }
        if command == vk::CommandBuffer::null()
            || authoritative.handle() == vk::Buffer::null()
            || authoritative_offset % 4 != 0
            || authoritative_offset
                .checked_add(self.layout.checkpoint_state_bytes)
                .is_none_or(|end| end > authoritative.size())
            || authoritative.handle() == self.arena.handle()
        {
            bail!("prefix checkpoint authoritative source/command 非法或与目标 arena alias");
        }
        {
            let mut phase = self.lock_phase()?;
            if *phase != PrefixArenaPhase::Ready {
                bail!("prefix checkpoint authoritative 初始化只能录制一次");
            }
            *phase = PrefixArenaPhase::InitializationRecorded;
        }

        let source = vk::BufferMemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::HOST_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .buffer(authoritative.handle())
            .offset(authoritative_offset)
            .size(self.layout.checkpoint_state_bytes);
        let mut destinations = Vec::with_capacity(self.layout.block_size);
        let mut copies = Vec::with_capacity(self.layout.block_size);
        for prefix in 0..self.layout.block_size {
            let offset = self.prefix_offset(prefix)?;
            destinations.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_READ
                            | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .buffer(self.arena.handle())
                    .offset(offset)
                    .size(self.layout.checkpoint_state_bytes),
            );
            copies.push(
                vk::BufferCopy::default()
                    .src_offset(authoritative_offset)
                    .dst_offset(offset)
                    .size(self.layout.checkpoint_state_bytes),
            );
        }
        let mut acquire = Vec::with_capacity(1 + destinations.len());
        acquire.push(source);
        acquire.extend(destinations);
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &acquire,
            &[],
        );
        context.device.cmd_copy_buffer(
            command,
            authoritative.handle(),
            self.arena.handle(),
            &copies,
        );
        let publish = copies
            .iter()
            .map(|copy| {
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_READ
                            | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .buffer(self.arena.handle())
                    .offset(copy.dst_offset)
                    .size(copy.size)
            })
            .collect::<Vec<_>>();
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &publish,
            &[],
        );
        Ok(S14CausalBlockPrefixInitializationReceipt {
            base_position: self.base_position,
            block_size: self.layout.block_size,
            checkpoint_state_bytes: self.layout.checkpoint_state_bytes,
            copy_regions: copies.len(),
            copied_bytes: self
                .layout
                .checkpoint_state_bytes
                .checked_mul(self.layout.block_size as u64)
                .context("prefix initialization copied bytes overflow")?,
            serial_token_forward_calls: 0,
        })
    }

    /// 只由同一 block-major producer 在43层完成后发布一次性回执。
    /// 本步不验收、不 seal arena，也不导出 checkpoint slices；真正的
    /// 验收只能在 terminal owner 的 post-seal `record_and_publish` 内发生。
    pub fn publish_prefix_program_seal_receipt(
        &self,
        receipt: S14CausalBlockPrefixStateSealReceipt,
    ) -> Result<()> {
        let mut phase = self.lock_phase()?;
        publish_receipt_for_phase(&mut phase, receipt)
    }

    /// terminal owner 的唯一延迟导出入口。先在持锁状态下对同源 producer
    /// receipt 执行完整 K×43层三角覆盖验证，成功后才原子进入
    /// `PrefixesSealed`，然后导出 slices。验证失败不会误发布 arena。
    pub(crate) fn seal_and_export_terminal_checkpoints(
        &self,
    ) -> Result<Vec<S14CausalBlockOwnedBufferSlice>> {
        {
            let mut phase = self.lock_phase()?;
            validate_and_seal_phase(&mut phase, self.base_position, self.layout, &self.evidence)?;
        }
        (0..self.layout.block_size)
            .map(|prefix| {
                Ok(S14CausalBlockOwnedBufferSlice::new(
                    Arc::clone(&self.arena),
                    self.prefix_offset(prefix)?,
                ))
            })
            .collect()
    }

    /// Teacher-forced prompt prefill 的非 terminal seal 门。它只验收同源 K4×43 层
    /// prefix program/evidence 并开放指定 prefix 的 host readback；不执行 final hidden、
    /// generation head，也不导出或发布任何预测 token。
    pub(crate) fn seal_for_starfold_teacher_forced_prefill(&self) -> Result<()> {
        let mut phase = self.lock_phase()?;
        validate_and_seal_phase(&mut phase, self.base_position, self.layout, &self.evidence)?;
        Ok(())
    }

    /// 任一已录制 command 被丢弃或 producer 失败时永久关闭本 one-shot arena binding。
    pub fn abort(&self) {
        if let Ok(mut phase) = self.phase.lock() {
            *phase = PrefixArenaPhase::Aborted;
        }
    }

    fn lock_phase(&self) -> Result<std::sync::MutexGuard<'_, PrefixArenaPhase>> {
        self.phase
            .lock()
            .map_err(|_| anyhow::anyhow!("prefix checkpoint arena lifecycle poisoned"))
    }
}

fn publish_receipt_for_phase(
    phase: &mut PrefixArenaPhase,
    receipt: S14CausalBlockPrefixStateSealReceipt,
) -> Result<()> {
    if *phase != PrefixArenaPhase::InitializationRecorded {
        bail!("prefix checkpoint seal receipt 只能由完成 authoritative 初始化的 producer 发布一次");
    }
    *phase = PrefixArenaPhase::PrefixProgramReceiptPublished(receipt);
    Ok(())
}

fn validate_and_seal_phase(
    phase: &mut PrefixArenaPhase,
    base_position: u32,
    layout: S14CausalBlockPrefixCheckpointLayout,
    evidence: &S14CausalBlockProductionEvidenceLedger,
) -> Result<S14CausalBlockPrefixStateSealReceipt> {
    let PrefixArenaPhase::PrefixProgramReceiptPublished(receipt) = *phase else {
        bail!("prefix checkpoint terminal 导出前缺少同源43层 producer seal receipt");
    };
    validate_prefix_seal_receipt(base_position, layout, receipt)?;
    evidence.record_prefix_seal(receipt)?;
    *phase = PrefixArenaPhase::PrefixesSealed;
    Ok(receipt)
}

fn validate_prefix_seal_receipt(
    base_position: u32,
    layout: S14CausalBlockPrefixCheckpointLayout,
    receipt: S14CausalBlockPrefixStateSealReceipt,
) -> Result<()> {
    let triangular = layout
        .block_size
        .checked_mul(layout.block_size + 1)
        .and_then(|value| value.checked_div(2))
        .context("prefix triangular count overflow")?;
    let expected_layers = layout
        .block_size
        .checked_mul(polaris_s14_runner::FULL_DEPTH_LAYERS.len())
        .context("prefix layer count overflow")?;
    let expected_applications = triangular
        .checked_mul(polaris_s14_runner::FULL_DEPTH_LAYERS.len())
        .context("prefix lane application count overflow")?;
    if receipt.base_position != base_position
        || receipt.block_size != layout.block_size
        || receipt.sealed_prefixes != layout.block_size
        || receipt.sealed_prefix_layers != expected_layers
        || receipt.cumulative_lane_applications != expected_applications
        || receipt.serial_token_forward_calls != 0
    {
        bail!("prefix checkpoint seal receipt 与 K-prefix×43层累积合同漂移");
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("prefix checkpoint alignment overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_checkpoint_layout_is_k_aligned_and_exact() {
        let k4 = S14CausalBlockPrefixCheckpointLayout::build(4, 46_000_003).unwrap();
        assert_eq!(k4.checkpoint_stride_bytes % PREFIX_ALIGNMENT, 0);
        assert_eq!(k4.checkpoint_offset(512, 0).unwrap(), 512);
        assert_eq!(
            k4.checkpoint_offset(512, 3).unwrap() + k4.checkpoint_state_bytes,
            512 + k4.used_bytes
        );
        let k8 = S14CausalBlockPrefixCheckpointLayout::build(8, 46_000_003).unwrap();
        assert!(k8.used_bytes > k4.used_bytes);
        assert!(S14CausalBlockPrefixCheckpointLayout::build(1, 1).is_err());
        assert!(S14CausalBlockPrefixCheckpointLayout::build(4, 0).is_err());
    }

    #[test]
    fn seal_receipt_requires_triangular_k_prefix_layer_coverage() {
        let layout = S14CausalBlockPrefixCheckpointLayout::build(4, 1024).unwrap();
        let valid = S14CausalBlockPrefixStateSealReceipt {
            base_position: 1,
            block_size: 4,
            sealed_prefixes: 4,
            sealed_prefix_layers: 172,
            cumulative_lane_applications: 430,
            serial_token_forward_calls: 0,
        };
        validate_prefix_seal_receipt(1, layout, valid).unwrap();
        let mut invalid = valid;
        invalid.cumulative_lane_applications -= 1;
        assert!(validate_prefix_seal_receipt(1, layout, invalid).is_err());
    }

    #[test]
    fn deferred_prefix_receipt_cannot_preseal_or_republish() {
        let layout = S14CausalBlockPrefixCheckpointLayout::build(4, 1024).unwrap();
        let valid = S14CausalBlockPrefixStateSealReceipt {
            base_position: 1,
            block_size: 4,
            sealed_prefixes: 4,
            sealed_prefix_layers: 172,
            cumulative_lane_applications: 430,
            serial_token_forward_calls: 0,
        };
        let mut phase = PrefixArenaPhase::Ready;
        assert!(publish_receipt_for_phase(&mut phase, valid).is_err());
        phase = PrefixArenaPhase::InitializationRecorded;
        publish_receipt_for_phase(&mut phase, valid).unwrap();
        assert!(publish_receipt_for_phase(&mut phase, valid).is_err());
        let evidence = S14CausalBlockProductionEvidenceLedger::new(1, 4).unwrap();
        let receipt = validate_and_seal_phase(&mut phase, 1, layout, &evidence).unwrap();
        assert_eq!(receipt, valid);
        assert_eq!(phase, PrefixArenaPhase::PrefixesSealed);
        assert!(validate_and_seal_phase(&mut phase, 1, layout, &evidence).is_err());
    }
}
