//! GPU head 回读后完成 K=4/8 host checkpoint ledger 的 production finalizer。
//!
//! K-lane 数值图在 head token 未知时已经生成每个前缀的完整 native state snapshot；本模块只在
//! terminal GPU argmax 返回后补齐 token ledger/epoch/bank/teacher-force identity。它不生成或修改
//! native state 字节，也不接受缺少完整 arena owner 的占位 checkpoint。

use crate::{
    s14_causal_block_prefix_arena::S14CausalBlockPrefixCheckpointArena,
    s14_causal_block_terminal_adapter::S14CausalBlockHostCandidateFinalizer,
    s14_head_chunk_argmax::S14HeadArgmaxResult, GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use polaris_s14_runner::{
    BatchedWholeTokenOutput, BatchedWholeTokenPosition, DecoderStateV1, GraphProfile, NativeState,
    NativeStateArena, RouteDecision, TokenRecord, BATCHED_CAUSAL_WHOLE_TOKEN_MODE, VOCAB_SIZE,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct S14CausalBlockPreparedHostCheckpoint {
    native: NativeState,
    native_arena: NativeStateArena,
}

impl S14CausalBlockPreparedHostCheckpoint {
    pub fn new(native: NativeState, native_arena: NativeStateArena) -> Result<Self> {
        native.validate_for(GraphProfile::FullDepth43NativeTop6)?;
        native_arena.validate(&native)?;
        Ok(Self {
            native,
            native_arena,
        })
    }

    pub fn position(&self) -> u32 {
        self.native.position
    }

    pub fn arena_bytes(&self) -> usize {
        self.native_arena.len()
    }
}

/// K份 device checkpoint 的同源 host readback。`draft_token_ids[lane]` 是该 checkpoint 发布后
/// 的 teacher-force next input，和 `WholeTokenFutureBlock` 的最长前缀合同一致。
#[derive(Debug)]
pub struct S14CausalBlockPreparedHostCandidateBatch {
    authoritative: DecoderStateV1,
    draft_token_ids: Vec<u32>,
    checkpoints: Vec<S14CausalBlockPreparedHostCheckpoint>,
}

impl S14CausalBlockPreparedHostCandidateBatch {
    pub fn new(
        authoritative: DecoderStateV1,
        draft_token_ids: Vec<u32>,
        checkpoints: Vec<S14CausalBlockPreparedHostCheckpoint>,
    ) -> Result<Self> {
        authoritative.validate()?;
        let block_size = draft_token_ids.len();
        if !matches!(block_size, 4 | 8)
            || checkpoints.len() != block_size
            || draft_token_ids.iter().any(|&token| token >= VOCAB_SIZE)
        {
            bail!("prepared host candidate 只接受等长 K=4/8 有效 draft/checkpoint");
        }
        let end_position = authoritative
            .position
            .checked_add(block_size as u32)
            .context("prepared host candidate position overflow")?;
        if end_position > authoritative.native.max_seq_len {
            bail!("prepared host candidate 越出 max_seq_len");
        }
        for (lane, checkpoint) in checkpoints.iter().enumerate() {
            let expected_position = authoritative
                .position
                .checked_add(lane as u32 + 1)
                .context("prepared checkpoint position overflow")?;
            if checkpoint.native.position != expected_position
                || checkpoint.native.max_seq_len != authoritative.native.max_seq_len
                || checkpoint.native.profile != GraphProfile::FullDepth43NativeTop6
                || checkpoint.native.poisoned
                || checkpoint.native_arena.arena_id() != authoritative.native_arena.arena_id()
                || checkpoint.native_arena.len() != authoritative.native_arena.len()
            {
                bail!("prepared checkpoint {lane} position/layout/arena identity 漂移");
            }
            checkpoint.native_arena.validate(&checkpoint.native)?;
        }
        Ok(Self {
            authoritative,
            draft_token_ids,
            checkpoints,
        })
    }
}

impl S14CausalBlockHostCandidateFinalizer for S14CausalBlockPreparedHostCandidateBatch {
    fn block_size(&self) -> usize {
        self.checkpoints.len()
    }

    fn base_position(&self) -> u32 {
        self.authoritative.position
    }

    fn complete_after_gpu_head(
        self: Box<Self>,
        head_results: &[S14HeadArgmaxResult],
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<BatchedWholeTokenOutput> {
        let Self {
            authoritative,
            draft_token_ids,
            checkpoints,
        } = *self;
        let block_size = checkpoints.len();
        if head_results.len() != block_size || routes_by_position.len() != block_size {
            bail!("prepared host candidate 的 GPU head/routes K 漂移");
        }

        let mut ledger = authoritative.committed_tokens.clone();
        let mut positions = Vec::with_capacity(block_size);
        for (lane, ((prepared, head), routes)) in checkpoints
            .into_iter()
            .zip(head_results)
            .zip(routes_by_position)
            .enumerate()
        {
            if head.token_id >= VOCAB_SIZE {
                bail!("prepared host candidate GPU head token 越界");
            }
            let token_position = authoritative
                .position
                .checked_add(lane as u32)
                .context("prepared host token position overflow")?;
            let input_token_id = if lane == 0 {
                authoritative.input_token_id
            } else {
                draft_token_ids[lane - 1]
            };
            ledger.push(TokenRecord {
                position: token_position,
                input_token_id,
                predicted_token_id: head.token_id,
            });
            let checkpoint = DecoderStateV1 {
                abi_version: authoritative.abi_version,
                commit_epoch: authoritative
                    .commit_epoch
                    .checked_add(lane as u64 + 1)
                    .context("prepared host checkpoint epoch overflow")?,
                position: token_position
                    .checked_add(1)
                    .context("prepared host checkpoint position overflow")?,
                input_token_id: draft_token_ids[lane],
                active_fixed_bank: authoritative.active_fixed_bank ^ ((lane as u8 + 1) & 1),
                committed_tokens: ledger.clone(),
                native: prepared.native,
                native_arena: prepared.native_arena,
            };
            checkpoint.validate()?;
            positions.push(BatchedWholeTokenPosition {
                predicted_token_id: head.token_id,
                routes: routes.clone(),
                checkpoint,
            });
        }

        Ok(BatchedWholeTokenOutput {
            mode: BATCHED_CAUSAL_WHOLE_TOKEN_MODE.to_owned(),
            forward_calls: 1,
            positions,
        })
    }
}

/// 43层 producer 完成前不伪造 host snapshot。该 finalizer 只强持有 authoritative
/// metadata 与同源 prefix arena；terminal 已等待 producer timeline、完成 GPU head 且
/// post-seal 验收通过后，才一次批量回读 K 份完整 NativeState checkpoint。
#[derive(Debug)]
pub struct S14CausalBlockDeferredHostCandidateBatch {
    authoritative: DecoderStateV1,
    draft_token_ids: Vec<u32>,
    prefix_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
}

impl S14CausalBlockDeferredHostCandidateBatch {
    pub fn new(
        authoritative: DecoderStateV1,
        draft_token_ids: Vec<u32>,
        prefix_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
    ) -> Result<Self> {
        authoritative.validate()?;
        let layout = prefix_arena.layout();
        if authoritative.position != prefix_arena.base_position()
            || draft_token_ids.len() != layout.block_size
            || !matches!(layout.block_size, 4 | 8)
            || draft_token_ids.iter().any(|&token| token >= VOCAB_SIZE)
            || layout.checkpoint_state_bytes != authoritative.native.arena_bytes
            || authoritative.native_arena.len() as u64 != layout.checkpoint_state_bytes
        {
            bail!("deferred host candidate 的 authoritative/K/state arena identity 漂移");
        }
        authoritative
            .position
            .checked_add(layout.block_size as u32)
            .filter(|&position| position <= authoritative.native.max_seq_len)
            .context("deferred host candidate 越出 max_seq_len")?;
        Ok(Self {
            authoritative,
            draft_token_ids,
            prefix_arena,
        })
    }
}

impl S14CausalBlockHostCandidateFinalizer for S14CausalBlockDeferredHostCandidateBatch {
    fn block_size(&self) -> usize {
        self.draft_token_ids.len()
    }

    fn base_position(&self) -> u32 {
        self.authoritative.position
    }

    fn complete_after_gpu_head(
        self: Box<Self>,
        head_results: &[S14HeadArgmaxResult],
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<BatchedWholeTokenOutput> {
        let Self {
            authoritative,
            draft_token_ids,
            prefix_arena,
        } = *self;
        let block_size = draft_token_ids.len();
        if head_results.len() != block_size || routes_by_position.len() != block_size {
            bail!("deferred host candidate 的 GPU head/routes K 漂移");
        }
        let checkpoint_bytes = readback_prefix_checkpoints(&prefix_arena)?;
        let checkpoints = checkpoint_bytes
            .into_iter()
            .enumerate()
            .map(|(lane, bytes)| {
                let mut native = authoritative.native.clone();
                native.position = authoritative
                    .position
                    .checked_add(lane as u32 + 1)
                    .context("deferred checkpoint position overflow")?;
                let native_arena = NativeStateArena::from_verified_checkpoint_bytes(&native, bytes)
                    .context("从 production device prefix 恢复 host NativeStateArena")?;
                S14CausalBlockPreparedHostCheckpoint::new(native, native_arena)
            })
            .collect::<Result<Vec<_>>>()?;
        Box::new(S14CausalBlockPreparedHostCandidateBatch::new(
            authoritative,
            draft_token_ids,
            checkpoints,
        )?)
        .complete_after_gpu_head(head_results, routes_by_position)
    }
}

fn readback_prefix_checkpoints(
    prefix_arena: &S14CausalBlockPrefixCheckpointArena,
) -> Result<Vec<Vec<u8>>> {
    prefix_arena.validate_host_readback_ready()?;
    let context: &Arc<VulkanContext> = prefix_arena.context();
    let layout = prefix_arena.layout();
    let total_bytes = layout
        .checkpoint_state_bytes
        .checked_mul(layout.block_size as u64)
        .context("prefix host readback bytes overflow")?;
    let total_host_bytes =
        usize::try_from(total_bytes).context("prefix host readback bytes 超出 host usize")?;

    let pool = unsafe {
        context.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(context.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )?
    };
    let command = match unsafe {
        context.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
    } {
        Ok(commands) => match commands.first().copied() {
            Some(command) => command,
            None => {
                unsafe { context.device.destroy_command_pool(pool, None) };
                bail!("prefix host readback command allocation 返回空集合");
            }
        },
        Err(error) => {
            unsafe { context.device.destroy_command_pool(pool, None) };
            return Err(error.into());
        }
    };
    let fence = match unsafe {
        context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    } {
        Ok(fence) => fence,
        Err(error) => {
            unsafe { context.device.destroy_command_pool(pool, None) };
            return Err(error.into());
        }
    };
    let readback = match GpuBuffer::new(
        context,
        total_bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        true,
    ) {
        Ok(buffer) => buffer,
        Err(error) => {
            unsafe {
                context.device.destroy_fence(fence, None);
                context.device.destroy_command_pool(pool, None);
            }
            return Err(error);
        }
    };

    let mut submitted = false;
    let mut completed = false;
    let mut result = (|| -> Result<Vec<Vec<u8>>> {
        let mut source_barriers = Vec::with_capacity(layout.block_size);
        let mut copies = Vec::with_capacity(layout.block_size);
        for lane in 0..layout.block_size {
            let source_offset = prefix_arena.prefix_offset(lane)?;
            source_barriers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .buffer(prefix_arena.buffer().handle())
                    .offset(source_offset)
                    .size(layout.checkpoint_state_bytes),
            );
            copies.push(
                vk::BufferCopy::default()
                    .src_offset(source_offset)
                    .dst_offset(layout.checkpoint_state_bytes * lane as u64)
                    .size(layout.checkpoint_state_bytes),
            );
        }
        unsafe {
            context.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            context.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &source_barriers,
                &[],
            );
            context.device.cmd_copy_buffer(
                command,
                prefix_arena.buffer().handle(),
                readback.handle(),
                &copies,
            );
            let host_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .buffer(readback.handle())
                .offset(0)
                .size(total_bytes);
            context.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[host_barrier],
                &[],
            );
            context.device.end_command_buffer(command)?;
            let commands = [command];
            context.device.queue_submit(
                context.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                fence,
            )?;
            submitted = true;
            context.device.wait_for_fences(&[fence], true, u64::MAX)?;
            completed = true;
        }
        let state_bytes = usize::try_from(layout.checkpoint_state_bytes)
            .context("prefix checkpoint bytes 超出 host usize")?;
        let all =
            unsafe { std::slice::from_raw_parts(readback.mapped() as *const u8, total_host_bytes) };
        if all.len() != total_host_bytes {
            bail!("prefix host readback mapped bytes 长度漂移");
        }
        let mut chunks = all.chunks_exact(state_bytes);
        let checkpoints = chunks.by_ref().map(<[u8]>::to_vec).collect::<Vec<_>>();
        if !chunks.remainder().is_empty() || checkpoints.len() != layout.block_size {
            bail!("prefix host readback checkpoint 切分数量/余数漂移");
        }
        Ok(checkpoints)
    })();

    let mut safe_to_destroy = !submitted || completed;
    if submitted && !completed {
        match unsafe { context.device.queue_wait_idle(context.q_graphics) } {
            Ok(()) => safe_to_destroy = true,
            Err(drain_error) => {
                let original = match &result {
                    Ok(_) => "未知 prefix host readback wait 失败".to_owned(),
                    Err(error) => format!("{error:#}"),
                };
                result = Err(anyhow::anyhow!(
                    "{original}; graphics queue drain 也失败: {drain_error:?}; pending readback 资源保留到 Vulkan context teardown"
                ));
            }
        }
    }
    if safe_to_destroy {
        readback.destroy(context);
        unsafe {
            context.device.destroy_fence(fence, None);
            context.device.destroy_command_pool(pool, None);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use polaris_s14_runner::{RouterKind, WholeTokenFutureBlock, FULL_DEPTH_LAYERS};

    fn routes(seed: u16) -> Vec<RouteDecision> {
        FULL_DEPTH_LAYERS
            .iter()
            .map(|&layer| RouteDecision {
                layer,
                kind: if layer <= 2 {
                    RouterKind::Hash
                } else {
                    RouterKind::Score
                },
                expert_ids: vec![seed, seed + 1, seed + 2, seed + 3, seed + 4, seed + 5],
                weights: vec![0.25; 6],
            })
            .collect()
    }

    #[test]
    fn gpu_head_completes_valid_teacher_forced_k4_checkpoint_chain() {
        let authoritative = DecoderStateV1::new(32, 0).unwrap();
        let draft = vec![5, 223, 939, 21];
        let checkpoints = (1..=4)
            .map(|position| {
                let mut native = authoritative.native.clone();
                native.position = position;
                S14CausalBlockPreparedHostCheckpoint::new(
                    native,
                    authoritative.native_arena.clone(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let routes_by_position = (0..4).map(routes).collect::<Vec<_>>();
        let heads = [5, 223, 17, 19]
            .into_iter()
            .map(|token_id| S14HeadArgmaxResult {
                token_id,
                logit: 1.0,
            })
            .collect::<Vec<_>>();
        let batch = S14CausalBlockPreparedHostCandidateBatch::new(
            authoritative.clone(),
            draft.clone(),
            checkpoints,
        )
        .unwrap();
        let output = Box::new(batch)
            .complete_after_gpu_head(&heads, &routes_by_position)
            .unwrap();
        let future =
            WholeTokenFutureBlock::from_batched_output(&authoritative, draft, output).unwrap();
        let decision = future.decision();
        assert_eq!(decision.accepted_prefix, vec![5, 223]);
        assert_eq!(decision.fallback_token_id, Some(17));
        assert_eq!(future.selected_checkpoint().unwrap().0, 2);
    }
}
