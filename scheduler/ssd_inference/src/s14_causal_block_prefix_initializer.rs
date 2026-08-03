//! position0 committed device bank 到 K=4 prefix checkpoint arena 的 production 初始化 owner。
//!
//! 该 owner 是 detached single-token runtime 与 block-major producer 之间的唯一桥：它只接受
//! 已真实提交的 position0 `0 -> 5` host/device identity，分配一块 K=4 device checkpoint arena，
//! 在同一 graphics queue 录制并完成 authoritative baseline copy，随后才允许构造 prefix producer。
//! 初始化失败会 abort arena；若等待失败且 queue 也无法 drain，则保留 pending Vulkan 资源直到
//! context teardown，禁止销毁仍可能在执行的 command 或 buffer。

use crate::{
    s14_causal_block_prefix_arena::{
        S14CausalBlockPrefixCheckpointArena, S14CausalBlockPrefixCheckpointLayout,
        S14CausalBlockPrefixInitializationReceipt,
    },
    s14_causal_block_prefix_producer::{
        S14CausalBlockPrefixStateProducer, S14CausalBlockSharedPrefixStateProgram,
    },
    s14_causal_block_terminal_owner::S14CausalBlockOwnedBufferSlice,
    s14_whole_token_device::WholeTokenDetachedCommittedState,
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::DecoderStateV1;
use std::{fmt, sync::Arc};

pub const S14_CAUSAL_BLOCK_BOOTSTRAP_BASE_POSITION: u32 = 1;
pub const S14_CAUSAL_BLOCK_BOOTSTRAP_BLOCK_SIZE: usize = 4;

const POSITION0_INPUT_TOKEN_ID: u32 = 0;
const POSITION0_PREDICTED_TOKEN_ID: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockPrefixInitializationCompletion {
    pub authoritative_epoch: u64,
    pub authoritative_active_bank: usize,
    pub authoritative_state_bytes: u64,
    pub base_position: u32,
    pub block_size: usize,
    pub copy_regions: usize,
    pub copied_bytes: u64,
    pub serial_token_forward_calls: u32,
    pub queue_submit_calls: u32,
    pub fence_wait_calls: u32,
    pub completed_before_producer: bool,
}

/// 成功构造即代表 authoritative copy 已 submit 且 fence 完成。调用方只能通过
/// `build_prefix_state_producer` 进入 K=4 producer；owner 必须晚于 provider/producer/terminal 销毁。
#[must_use = "K=4 bundle/provider/producer 销毁后必须显式 destroy prefix initialization owner"]
pub struct S14CausalBlockPrefixInitializationOwner {
    context: Arc<VulkanContext>,
    authoritative: DecoderStateV1,
    authoritative_device: Option<Arc<GpuBuffer>>,
    prefix_storage: Option<Arc<GpuBuffer>>,
    prefix_arena: Option<Arc<S14CausalBlockPrefixCheckpointArena>>,
    completion: S14CausalBlockPrefixInitializationCompletion,
}

impl fmt::Debug for S14CausalBlockPrefixInitializationOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockPrefixInitializationOwner")
            .field("context", &Arc::as_ptr(&self.context))
            .field("position", &self.authoritative.position)
            .field(
                "authoritative_device",
                &self
                    .authoritative_device
                    .as_ref()
                    .map(|buffer| buffer.handle()),
            )
            .field(
                "prefix_storage",
                &self.prefix_storage.as_ref().map(|buffer| buffer.handle()),
            )
            .field("prefix_arena", &self.prefix_arena.as_ref().map(Arc::as_ptr))
            .field("completion", &self.completion)
            .finish()
    }
}

impl S14CausalBlockPrefixInitializationOwner {
    /// 消费 position0 runtime 导出的 detached committed bank。成功返回前已经完成 K=4
    /// authoritative baseline copy；不存在“只录 command、尚未提交”或“提交后未等待”的 owner。
    pub fn initialize(
        context: Arc<VulkanContext>,
        authoritative: DecoderStateV1,
        committed: WholeTokenDetachedCommittedState,
    ) -> Result<Self> {
        let WholeTokenDetachedCommittedState {
            buffer: committed_buffer,
            state_bytes,
            epoch,
            active_bank,
            source_device,
            source_graphics_queue,
            source_graphics_queue_family,
        } = committed;
        if let Err(error) = validate_position0_committed_identity(
            &context,
            &authoritative,
            &committed_buffer,
            state_bytes,
            epoch,
            active_bank,
            source_device,
            source_graphics_queue,
            source_graphics_queue_family,
        ) {
            committed_buffer.destroy(&context);
            return Err(error);
        }

        let layout = S14CausalBlockPrefixCheckpointLayout::build(
            S14_CAUSAL_BLOCK_BOOTSTRAP_BLOCK_SIZE,
            state_bytes,
        )?;
        let prefix_buffer = match GpuBuffer::new_vram(
            &context,
            layout.used_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                committed_buffer.destroy(&context);
                return Err(error.context("allocate K=4 prefix checkpoint storage"));
            }
        };
        let authoritative_device = Arc::new(committed_buffer);
        let prefix_storage = Arc::new(prefix_buffer);
        let prefix_arena = match S14CausalBlockPrefixCheckpointArena::bind(
            Arc::clone(&context),
            Arc::clone(&prefix_storage),
            0,
            S14_CAUSAL_BLOCK_BOOTSTRAP_BASE_POSITION,
            S14_CAUSAL_BLOCK_BOOTSTRAP_BLOCK_SIZE,
            state_bytes,
        ) {
            Ok(arena) => arena,
            Err(error) => {
                destroy_unshared_buffer(&context, prefix_storage);
                destroy_unshared_buffer(&context, authoritative_device);
                return Err(error.context("bind K=4 prefix checkpoint arena"));
            }
        };

        let (command_pool, command, fence) = match allocate_initialization_command(&context) {
            Ok(resources) => resources,
            Err(error) => {
                prefix_arena.abort();
                drop(prefix_arena);
                destroy_unshared_buffer(&context, prefix_storage);
                destroy_unshared_buffer(&context, authoritative_device);
                return Err(error.context("allocate prefix initialization command"));
            }
        };

        let mut submitted = false;
        let mut completed = false;
        let initialization = (|| -> Result<S14CausalBlockPrefixInitializationReceipt> {
            unsafe {
                context.device.begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let receipt = prefix_arena.record_authoritative_initialization(
                    &context,
                    command,
                    authoritative_device.as_ref(),
                    0,
                )?;
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
                Ok(receipt)
            }
        })();

        let initialization = match initialization {
            Ok(receipt) => receipt,
            Err(error) => {
                prefix_arena.abort();
                return Err(cleanup_failed_initialization(
                    &context,
                    prefix_arena,
                    prefix_storage,
                    authoritative_device,
                    command_pool,
                    fence,
                    submitted,
                    completed,
                    error,
                ));
            }
        };
        let completion = match validate_initialization_receipt(
            initialization,
            epoch,
            active_bank,
            state_bytes,
        ) {
            Ok(completion) => completion,
            Err(error) => {
                prefix_arena.abort();
                return Err(cleanup_failed_initialization(
                    &context,
                    prefix_arena,
                    prefix_storage,
                    authoritative_device,
                    command_pool,
                    fence,
                    submitted,
                    completed,
                    error,
                ));
            }
        };

        unsafe {
            context.device.destroy_fence(fence, None);
            context.device.destroy_command_pool(command_pool, None);
        }
        Ok(Self {
            context,
            authoritative,
            authoritative_device: Some(authoritative_device),
            prefix_storage: Some(prefix_storage),
            prefix_arena: Some(prefix_arena),
            completion,
        })
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn authoritative(&self) -> &DecoderStateV1 {
        &self.authoritative
    }

    pub fn completion(&self) -> S14CausalBlockPrefixInitializationCompletion {
        self.completion
    }

    pub fn authoritative_device_state(&self) -> Result<S14CausalBlockOwnedBufferSlice> {
        let buffer = self
            .authoritative_device
            .as_ref()
            .context("prefix initialization authoritative device 已销毁")?;
        Ok(S14CausalBlockOwnedBufferSlice {
            buffer: Arc::clone(buffer),
            offset: 0,
        })
    }

    pub fn prefix_arena(&self) -> Result<&Arc<S14CausalBlockPrefixCheckpointArena>> {
        self.prefix_arena
            .as_ref()
            .context("prefix initialization arena 已销毁")
    }

    /// 只有成功初始化后的 owner 才暴露 producer constructor，保证 authoritative K=4 copy
    /// 已在同一 graphics queue 完成，不允许首层与 baseline copy 并发竞态。
    pub fn build_prefix_state_producer(
        &self,
        program: S14CausalBlockSharedPrefixStateProgram,
    ) -> Result<S14CausalBlockPrefixStateProducer> {
        if !self.completion.completed_before_producer
            || self.completion.queue_submit_calls != 1
            || self.completion.fence_wait_calls != 1
        {
            bail!("prefix initialization 尚未完成 submit/wait，禁止进入 K=4 producer");
        }
        S14CausalBlockPrefixStateProducer::new(
            Arc::clone(&self.context),
            Arc::clone(self.prefix_arena()?),
            program,
        )
    }

    /// 必须在所有 provider/producer/ratio4/terminal owner 与 sealed future 释放后调用。
    /// 若还有 Arc consumer，先 abort arena 并拒绝释放，避免 use-after-free。
    pub fn destroy(&mut self) -> Result<()> {
        if let Some(arena) = self.prefix_arena.as_ref() {
            arena.abort();
        }
        if self
            .prefix_arena
            .as_ref()
            .is_some_and(|arena| Arc::strong_count(arena) != 1)
            || self
                .prefix_storage
                .as_ref()
                .is_some_and(|buffer| Arc::strong_count(buffer) != 2)
            || self
                .authoritative_device
                .as_ref()
                .is_some_and(|buffer| Arc::strong_count(buffer) != 1)
        {
            bail!("prefix initialization 仍被 provider/producer/terminal 持有，拒绝销毁");
        }

        if let Some(arena) = self.prefix_arena.take() {
            drop(Arc::try_unwrap(arena).map_err(|arena| {
                self.prefix_arena = Some(arena);
                anyhow!("prefix initialization arena Arc 在销毁时漂移")
            })?);
        }
        if let Some(storage) = self.prefix_storage.take() {
            match Arc::try_unwrap(storage) {
                Ok(buffer) => buffer.destroy(&self.context),
                Err(storage) => {
                    self.prefix_storage = Some(storage);
                    bail!("prefix checkpoint storage Arc 在销毁时漂移");
                }
            }
        }
        if let Some(authoritative) = self.authoritative_device.take() {
            match Arc::try_unwrap(authoritative) {
                Ok(buffer) => buffer.destroy(&self.context),
                Err(authoritative) => {
                    self.authoritative_device = Some(authoritative);
                    bail!("authoritative committed device Arc 在销毁时漂移");
                }
            }
        }
        Ok(())
    }
}

impl Drop for S14CausalBlockPrefixInitializationOwner {
    fn drop(&mut self) {
        if let Some(arena) = self.prefix_arena.as_ref() {
            arena.abort();
        }
    }
}

fn validate_position0_committed_identity(
    context: &VulkanContext,
    authoritative: &DecoderStateV1,
    committed_buffer: &GpuBuffer,
    state_bytes: u64,
    epoch: u64,
    active_bank: usize,
    source_device: vk::Device,
    source_graphics_queue: vk::Queue,
    source_graphics_queue_family: u32,
) -> Result<()> {
    authoritative
        .validate()
        .context("validate position0 committed host state")?;
    let records = authoritative.committed_tokens.as_slice();
    if authoritative.position != S14_CAUSAL_BLOCK_BOOTSTRAP_BASE_POSITION
        || authoritative.native.position != S14_CAUSAL_BLOCK_BOOTSTRAP_BASE_POSITION
        || authoritative.commit_epoch != epoch
        || usize::from(authoritative.active_fixed_bank) != active_bank
        || authoritative.input_token_id != POSITION0_PREDICTED_TOKEN_ID
        || state_bytes != authoritative.native.arena_bytes
        || state_bytes != authoritative.native_arena.len() as u64
        || state_bytes == 0
        || state_bytes > committed_buffer.size()
        || state_bytes % 4 != 0
        || active_bank > 1
        || committed_buffer.handle() == vk::Buffer::null()
        || source_device != context.device.handle()
        || source_graphics_queue != context.q_graphics
        || source_graphics_queue_family != context.qf_graphics
        || records.len() != 1
        || records[0].position != 0
        || records[0].input_token_id != POSITION0_INPUT_TOKEN_ID
        || records[0].predicted_token_id != POSITION0_PREDICTED_TOKEN_ID
    {
        bail!("prefix initialization 只接受同源 position0 0→5 committed host/device bank");
    }
    Ok(())
}

fn validate_initialization_receipt(
    receipt: S14CausalBlockPrefixInitializationReceipt,
    epoch: u64,
    active_bank: usize,
    state_bytes: u64,
) -> Result<S14CausalBlockPrefixInitializationCompletion> {
    let copied_bytes = state_bytes
        .checked_mul(S14_CAUSAL_BLOCK_BOOTSTRAP_BLOCK_SIZE as u64)
        .context("prefix initialization copied bytes overflow")?;
    if receipt.base_position != S14_CAUSAL_BLOCK_BOOTSTRAP_BASE_POSITION
        || receipt.block_size != S14_CAUSAL_BLOCK_BOOTSTRAP_BLOCK_SIZE
        || receipt.checkpoint_state_bytes != state_bytes
        || receipt.copy_regions != S14_CAUSAL_BLOCK_BOOTSTRAP_BLOCK_SIZE
        || receipt.copied_bytes != copied_bytes
        || receipt.serial_token_forward_calls != 0
    {
        bail!("prefix initialization receipt 与 authoritative K=4 baseline copy 漂移");
    }
    Ok(S14CausalBlockPrefixInitializationCompletion {
        authoritative_epoch: epoch,
        authoritative_active_bank: active_bank,
        authoritative_state_bytes: state_bytes,
        base_position: receipt.base_position,
        block_size: receipt.block_size,
        copy_regions: receipt.copy_regions,
        copied_bytes: receipt.copied_bytes,
        serial_token_forward_calls: receipt.serial_token_forward_calls,
        queue_submit_calls: 1,
        fence_wait_calls: 1,
        completed_before_producer: true,
    })
}

fn allocate_initialization_command(
    context: &VulkanContext,
) -> Result<(vk::CommandPool, vk::CommandBuffer, vk::Fence)> {
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
                bail!("prefix initialization command allocation 返回空集合");
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
    Ok((pool, command, fence))
}

#[allow(clippy::too_many_arguments)]
fn cleanup_failed_initialization(
    context: &Arc<VulkanContext>,
    prefix_arena: Arc<S14CausalBlockPrefixCheckpointArena>,
    prefix_storage: Arc<GpuBuffer>,
    authoritative_device: Arc<GpuBuffer>,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    submitted: bool,
    completed: bool,
    error: anyhow::Error,
) -> anyhow::Error {
    let mut safe_to_destroy = !submitted || completed;
    let mut result = error;
    if submitted && !completed {
        match unsafe { context.device.queue_wait_idle(context.q_graphics) } {
            Ok(()) => safe_to_destroy = true,
            Err(drain_error) => {
                result = anyhow!(
                    "{result:#}; prefix initialization graphics queue drain 也失败: {drain_error:?}; pending source/arena/command 资源保留到 Vulkan context teardown"
                );
            }
        }
    }
    if safe_to_destroy {
        unsafe {
            context.device.destroy_fence(fence, None);
            context.device.destroy_command_pool(command_pool, None);
        }
        drop(prefix_arena);
        destroy_unshared_buffer(context, prefix_storage);
        destroy_unshared_buffer(context, authoritative_device);
    }
    result
}

fn destroy_unshared_buffer(context: &VulkanContext, buffer: Arc<GpuBuffer>) {
    if let Ok(buffer) = Arc::try_unwrap(buffer) {
        buffer.destroy(context);
    }
}
