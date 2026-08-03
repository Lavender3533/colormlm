//! S14 production K=4 任意 token embedding 到 causal-block hidden 的强 owner。
//!
//! 每个 token 的真实 Range 只有 BF16 `[4096]`；本模块必须先经
//! `S14InputAssetPlanner` 解析 proof，再由 `VerifiedMappedAssetStore` 完整 SHA+mmap，
//! 最后在同一 Vulkan context 上把 K=4 行分别 broadcast 成 `[4,4,4096]`
//! HC hidden。这里不接受固定 BOS、manifest replay、零填充或 host capture。

use crate::{
    compute::StorageBufferSlice,
    s14_causal_block_hc_qkv_recorder::S14CausalBlockHiddenBank,
    s14_causal_block_layer::{
        S14CausalBlockHiddenBinding, S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
        S14_CAUSAL_BLOCK_HC_STREAMS, S14_CAUSAL_BLOCK_STREAM_WIDTH,
    },
    s14_embedding_broadcast::{
        validate_embedding_broadcast_status, S14EmbeddingBroadcastDispatch,
        S14EmbeddingBroadcastPipeline, S14EmbeddingBroadcastShape,
    },
    s14_input_asset_plan::S14InputAssetPlanner,
    s14_position0_mapped_assets::{VerifiedMappedAsset, VerifiedMappedAssetStore},
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use polaris_s14_runner::Position0Asset;
use std::{collections::BTreeMap, fmt, sync::Arc};

pub const S14_CAUSAL_BLOCK_K4_INPUT_TOKENS: usize = 4;
pub const S14_CAUSAL_BLOCK_K4_EMBEDDING_ROW_BYTES: u64 = S14_CAUSAL_BLOCK_STREAM_WIDTH as u64 * 2;
pub const S14_CAUSAL_BLOCK_K4_SOURCE_BYTES: u64 =
    S14_CAUSAL_BLOCK_K4_INPUT_TOKENS as u64 * S14_CAUSAL_BLOCK_K4_EMBEDDING_ROW_BYTES;
pub const S14_CAUSAL_BLOCK_K4_HIDDEN_BYTES: u64 =
    S14_CAUSAL_BLOCK_K4_INPUT_TOKENS as u64 * S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64 * 2;
/// Production recorder 同时预注册 K=4/8，因此底层 bank 必须满足 K=8
/// capacity；本 owner 仍只发布前 K=4 的真实 embedding binding。
pub const S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES: u64 =
    8 * S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64 * 2;

/// 强持有 production K=4 初始 hidden。`GpuBuffer` 没有 RAII Drop，因此
/// 使用者必须在所有 recorder/bank clone 释放后显式调用 `destroy()`。
#[must_use = "K=4 input hidden 的所有消费者释放后必须显式 destroy"]
pub struct S14CausalBlockK4InputHiddenOwner {
    context: Arc<VulkanContext>,
    hidden: Option<Arc<GpuBuffer>>,
    alternate: Option<Arc<GpuBuffer>>,
    base_position: u32,
    token_ids: [u32; S14_CAUSAL_BLOCK_K4_INPUT_TOKENS],
    generation: u64,
}

impl fmt::Debug for S14CausalBlockK4InputHiddenOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockK4InputHiddenOwner")
            .field("context", &Arc::as_ptr(&self.context))
            .field("hidden_present", &self.hidden.is_some())
            .field("alternate_present", &self.alternate.is_some())
            .field("base_position", &self.base_position)
            .field("token_ids", &self.token_ids)
            .field("generation", &self.generation)
            .finish()
    }
}

impl S14CausalBlockK4InputHiddenOwner {
    /// 为 positions `0..4` 的任意四个 token 构建真实 production hidden。
    ///
    /// 重复 token 合法；映射层会只对唯一 Range 做一次 SHA，然后把同一
    /// verified lease 绑回多个 lane。任一 proof/SHA/shape/context/Vulkan 门失败都不发布 binding。
    pub fn build(
        context: Arc<VulkanContext>,
        planner: &S14InputAssetPlanner,
        store: &mut VerifiedMappedAssetStore,
        token_ids: [u32; S14_CAUSAL_BLOCK_K4_INPUT_TOKENS],
        generation: u64,
    ) -> Result<Self> {
        Self::build_at(
            context,
            planner,
            store,
            0,
            S14_CAUSAL_BLOCK_K4_INPUT_TOKENS as u32,
            token_ids,
            generation,
        )
    }

    /// 为任意连续 positions `[base_position, base_position+4)` 构建 K=4 hidden。
    /// embedding 数值虽不含 position，资产计划仍必须绑定实际 token/position，
    /// 防止 position1+ production builder 把 base1 请求伪装成 base0 proof 身份。
    #[allow(clippy::too_many_arguments)]
    pub fn build_at(
        context: Arc<VulkanContext>,
        planner: &S14InputAssetPlanner,
        store: &mut VerifiedMappedAssetStore,
        base_position: u32,
        max_seq_len: u32,
        token_ids: [u32; S14_CAUSAL_BLOCK_K4_INPUT_TOKENS],
        generation: u64,
    ) -> Result<Self> {
        let block_end = base_position
            .checked_add(S14_CAUSAL_BLOCK_K4_INPUT_TOKENS as u32)
            .context("S14 K=4 input position end overflow")?;
        if max_seq_len == 0 || block_end > max_seq_len {
            bail!(
                "S14 K=4 input positions 越出 sequence: base={base_position} end={block_end} max_seq_len={max_seq_len}"
            );
        }
        let mut assets = Vec::with_capacity(S14_CAUSAL_BLOCK_K4_INPUT_TOKENS);
        for (lane, &token_id) in token_ids.iter().enumerate() {
            let position = base_position
                .checked_add(u32::try_from(lane).context("K=4 embedding lane position overflow")?)
                .context("K=4 embedding position overflow")?;
            let plan = planner
                .plan(position, token_id, max_seq_len)
                .with_context(|| format!("plan S14 K=4 lane {lane} token {token_id}"))?;
            let asset = planner
                .resolve_cached_embedding(&plan)
                .with_context(|| format!("resolve S14 K=4 lane {lane} token {token_id}"))?;
            validate_embedding_asset(&asset, token_id)?;
            assets.push(asset);
        }

        let (unique_assets, lane_to_unique) = deduplicate_embedding_assets(&assets)?;
        let unique_leases = store
            .map_verified_batch(&unique_assets)
            .context("mmap+SHA S14 K=4 embedding Range")?;
        if unique_leases.len() != unique_assets.len() {
            bail!("S14 K=4 verified embedding lease 数量漂移");
        }

        let mut lane_leases = Vec::with_capacity(S14_CAUSAL_BLOCK_K4_INPUT_TOKENS);
        for (lane, (&unique_index, asset)) in lane_to_unique.iter().zip(&assets).enumerate() {
            let lease = unique_leases
                .get(unique_index)
                .cloned()
                .with_context(|| format!("S14 K=4 lane {lane} verified lease 索引越界"))?;
            validate_verified_lease(asset, &lease, lane)?;
            lane_leases.push(lease);
        }

        let hidden = upload_and_broadcast_k4(&context, &lane_leases)
            .context("upload+broadcast S14 K=4 input hidden")?;
        let alternate = match GpuBuffer::new_vram(
            &context,
            S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                hidden.destroy(&context);
                return Err(error.context("allocate S14 K=4 alternate hidden bank"));
            }
        };
        let mut owner = Self {
            context,
            hidden: Some(Arc::new(hidden)),
            alternate: Some(Arc::new(alternate)),
            base_position,
            token_ids,
            generation,
        };
        if let Err(error) = owner.binding() {
            owner
                .destroy()
                .context("destroy invalid S14 K=4 input hidden")?;
            return Err(error.context("validate S14 K=4 input hidden binding"));
        }
        Ok(owner)
    }

    pub fn context(&self) -> &Arc<VulkanContext> {
        &self.context
    }

    pub fn token_ids(&self) -> &[u32; S14_CAUSAL_BLOCK_K4_INPUT_TOKENS] {
        &self.token_ids
    }

    pub fn base_position(&self) -> u32 {
        self.base_position
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// 交给会强持有 hidden bank 的 recorder/bundle。`destroy()` 会在 clone 尚未
    /// 释放时 fail-closed，防止提前销毁 Vulkan memory。
    pub fn hidden_bank(&self) -> Result<S14CausalBlockHiddenBank> {
        let buffer = Arc::clone(
            self.hidden
                .as_ref()
                .context("S14 K=4 input hidden 已销毁")?,
        );
        if buffer.handle() == vk::Buffer::null()
            || buffer.size() < S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES
        {
            bail!("S14 K=4 input hidden buffer handle/capacity 漂移");
        }
        Ok(S14CausalBlockHiddenBank {
            buffer,
            offset: 0,
            capacity_bytes: S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES,
        })
    }

    /// 返回 production recorder 交替写入的 A/B 双 bank。A 已由四个真实 embedding
    /// 初始化；B 在首层被完整覆盖后才读取，因此不需要伪造 host 零值输入。
    pub fn hidden_banks(&self) -> Result<[S14CausalBlockHiddenBank; 2]> {
        let primary = self.hidden_bank()?;
        let alternate = Arc::clone(
            self.alternate
                .as_ref()
                .context("S14 K=4 alternate hidden 已销毁")?,
        );
        if alternate.handle() == vk::Buffer::null()
            || alternate.size() < S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES
            || alternate.handle() == primary.buffer.handle()
        {
            bail!("S14 K=4 alternate hidden handle/capacity/alias 漂移");
        }
        Ok([
            primary,
            S14CausalBlockHiddenBank {
                buffer: alternate,
                offset: 0,
                capacity_bytes: S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES,
            },
        ])
    }

    /// 导出精确 `[4,4,4096]` BF16 binding，不导出 host/mmap 指针。
    pub fn binding(&self) -> Result<S14CausalBlockHiddenBinding> {
        self.hidden_bank()?
            .binding(S14_CAUSAL_BLOCK_K4_INPUT_TOKENS, self.generation)
    }

    /// 幂等显式销毁。若 hidden bank 仍被 recorder/bundle 强持有，则保留
    /// owner 并拒绝销毁。
    pub fn destroy(&mut self) -> Result<()> {
        let Some(hidden) = self.hidden.as_ref() else {
            if self.alternate.is_some() {
                bail!("S14 K=4 hidden owner primary/alternate 生命周期漂移");
            }
            return Ok(());
        };
        let refs = Arc::strong_count(hidden);
        let alternate_refs = self.alternate.as_ref().map_or(0, Arc::strong_count);
        if refs != 1 || alternate_refs != 1 {
            bail!(
                "S14 K=4 input hidden 仍被 recorder/bundle 持有: primary_refs={refs} alternate_refs={alternate_refs}"
            );
        }
        let alternate = self
            .alternate
            .take()
            .context("S14 K=4 alternate hidden owner 漂移")?;
        let hidden = self
            .hidden
            .take()
            .context("S14 K=4 input hidden owner 漂移")?;
        match Arc::try_unwrap(hidden) {
            Ok(buffer) => {
                buffer.destroy(&self.context);
                match Arc::try_unwrap(alternate) {
                    Ok(buffer) => {
                        buffer.destroy(&self.context);
                        Ok(())
                    }
                    Err(alternate) => {
                        self.alternate = Some(alternate);
                        bail!("S14 K=4 alternate hidden Arc ownership 漂移")
                    }
                }
            }
            Err(hidden) => {
                self.hidden = Some(hidden);
                self.alternate = Some(alternate);
                bail!("S14 K=4 input hidden Arc ownership 漂移")
            }
        }
    }
}

fn validate_embedding_asset(asset: &Position0Asset, token_id: u32) -> Result<()> {
    let expected_tensor = format!("embed.weight[{token_id}:{}]", token_id + 1);
    if asset.tensor != expected_tensor
        || asset.kind != "embedding_row"
        || asset.expert_id.is_some()
        || asset.dtype != "BF16"
        || asset.shape != [1, S14_CAUSAL_BLOCK_STREAM_WIDTH as u64]
        || asset.bytes != S14_CAUSAL_BLOCK_K4_EMBEDDING_ROW_BYTES
        || asset.sha256.len() != 64
        || asset.proof_sha256.len() != 64
    {
        bail!("S14 K=4 token {token_id} embedding asset ABI 漂移");
    }
    Ok(())
}

fn deduplicate_embedding_assets(
    assets: &[Position0Asset],
) -> Result<(
    Vec<Position0Asset>,
    [usize; S14_CAUSAL_BLOCK_K4_INPUT_TOKENS],
)> {
    if assets.len() != S14_CAUSAL_BLOCK_K4_INPUT_TOKENS {
        bail!("S14 K=4 embedding asset 数量必须为 4");
    }
    let mut by_cache_key = BTreeMap::<String, usize>::new();
    let mut unique = Vec::<Position0Asset>::new();
    let mut lane_to_unique = [0usize; S14_CAUSAL_BLOCK_K4_INPUT_TOKENS];
    for (lane, asset) in assets.iter().enumerate() {
        if let Some(&index) = by_cache_key.get(&asset.cache_key) {
            let existing = unique
                .get(index)
                .context("S14 K=4 duplicate embedding unique index 越界")?;
            if existing.tensor != asset.tensor
                || existing.path != asset.path
                || existing.bytes != asset.bytes
                || existing.sha256 != asset.sha256
                || existing.proof_path != asset.proof_path
                || existing.proof_sha256 != asset.proof_sha256
                || existing.source != asset.source
            {
                bail!("S14 K=4 同 cache key embedding identity 不一致");
            }
            lane_to_unique[lane] = index;
        } else {
            let index = unique.len();
            by_cache_key.insert(asset.cache_key.clone(), index);
            unique.push(asset.clone());
            lane_to_unique[lane] = index;
        }
    }
    if unique.is_empty() || unique.len() > S14_CAUSAL_BLOCK_K4_INPUT_TOKENS {
        bail!("S14 K=4 unique embedding 数量非法");
    }
    Ok((unique, lane_to_unique))
}

fn validate_verified_lease(
    asset: &Position0Asset,
    lease: &VerifiedMappedAsset,
    lane: usize,
) -> Result<()> {
    if lease.tensor() != asset.tensor
        || lease.path() != asset.path
        || lease.expected_sha256() != asset.sha256
        || lease.bytes().len() as u64 != asset.bytes
        || lease.bytes().len() as u64 != S14_CAUSAL_BLOCK_K4_EMBEDDING_ROW_BYTES
    {
        bail!("S14 K=4 lane {lane} verified mmap identity/bytes 漂移");
    }
    Ok(())
}

fn upload_and_broadcast_k4(
    context: &VulkanContext,
    lane_leases: &[Arc<VerifiedMappedAsset>],
) -> Result<GpuBuffer> {
    if lane_leases.len() != S14_CAUSAL_BLOCK_K4_INPUT_TOKENS {
        bail!("S14 K=4 upload lease 数量必须为 4");
    }
    if S14_CAUSAL_BLOCK_HC_STREAMS != 4
        || S14_CAUSAL_BLOCK_STREAM_WIDTH != 4096
        || S14_CAUSAL_BLOCK_K4_SOURCE_BYTES != 32_768
        || S14_CAUSAL_BLOCK_K4_HIDDEN_BYTES != 131_072
        || S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES != 262_144
    {
        bail!("S14 K=4 embedding/HC 常量合同漂移");
    }

    let limits = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical)
            .limits
    };
    let storage_alignment = u64::from(limits.min_storage_buffer_offset_alignment.max(1));
    let token_hidden_bytes = S14_CAUSAL_BLOCK_K4_TOKEN_HIDDEN_BYTES;
    if S14_CAUSAL_BLOCK_K4_EMBEDDING_ROW_BYTES % storage_alignment != 0
        || token_hidden_bytes % storage_alignment != 0
    {
        bail!("S14 K=4 embedding/hidden stride 不满足 storage alignment {storage_alignment}");
    }
    let status_stride = align_up(4, storage_alignment)?;
    let status_bytes = status_stride
        .checked_mul(S14_CAUSAL_BLOCK_K4_INPUT_TOKENS as u64)
        .context("S14 K=4 status bytes overflow")?;

    let mut scratch = K4UploadScratch::new();
    let result = (|| -> Result<GpuBuffer> {
        scratch.source = Some(GpuBuffer::new(
            context,
            S14_CAUSAL_BLOCK_K4_SOURCE_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?);
        scratch.status = Some(GpuBuffer::new(
            context,
            status_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            true,
        )?);
        scratch.output = Some(GpuBuffer::new_vram(
            context,
            S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?);

        for (lane, lease) in lane_leases.iter().enumerate() {
            if lease.bytes().len() as u64 != S14_CAUSAL_BLOCK_K4_EMBEDDING_ROW_BYTES {
                bail!("S14 K=4 lane {lane} mmap row bytes 漂移");
            }
            let source_offset = lane
                .checked_mul(S14_CAUSAL_BLOCK_K4_EMBEDDING_ROW_BYTES as usize)
                .context("S14 K=4 source offset overflow")?;
            unsafe {
                scratch
                    .source
                    .as_ref()
                    .context("S14 K=4 source buffer 缺失")?
                    .write_at(source_offset, lease.bytes());
            }
            let status_offset = usize::try_from(
                status_stride
                    .checked_mul(lane as u64)
                    .context("S14 K=4 status offset overflow")?,
            )?;
            unsafe {
                scratch
                    .status
                    .as_ref()
                    .context("S14 K=4 status buffer 缺失")?
                    .write_at(status_offset, &0u32.to_le_bytes());
            }
        }

        scratch.pipeline = Some(S14EmbeddingBroadcastPipeline::new(context)?);
        scratch.pool = unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?
        };
        let commands = unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(scratch.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        };
        scratch.command = commands
            .first()
            .copied()
            .context("S14 K=4 command allocation 返回空集合")?;
        scratch.fence = unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        let shape = S14EmbeddingBroadcastShape::new(S14_CAUSAL_BLOCK_STREAM_WIDTH as u32)?;
        for lane in 0..S14_CAUSAL_BLOCK_K4_INPUT_TOKENS {
            let source_offset = S14_CAUSAL_BLOCK_K4_EMBEDDING_ROW_BYTES * lane as u64;
            let output_offset = token_hidden_bytes * lane as u64;
            let status_offset = status_stride * lane as u64;
            let dispatch = scratch
                .pipeline
                .as_ref()
                .context("S14 K=4 embedding pipeline 缺失")?
                .bind_slices(
                    context,
                    shape,
                    StorageBufferSlice {
                        buffer: scratch.source.as_ref().context("S14 K=4 source 缺失")?,
                        offset: source_offset,
                    },
                    StorageBufferSlice {
                        buffer: scratch.output.as_ref().context("S14 K=4 output 缺失")?,
                        offset: output_offset,
                    },
                    StorageBufferSlice {
                        buffer: scratch.status.as_ref().context("S14 K=4 status 缺失")?,
                        offset: status_offset,
                    },
                )?;
            scratch.dispatches.push(dispatch);
        }

        unsafe {
            context.device.begin_command_buffer(
                scratch.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            // K=8 capacity 的上半部分不是本 K=4 请求的模型输入；仍先全量清零，
            // 避免 bundle 持有期间保留未初始化 VRAM。前 K=4 随后被真实 embedding 覆盖。
            context.device.cmd_fill_buffer(
                scratch.command,
                scratch
                    .output
                    .as_ref()
                    .context("S14 K=4 output 缺失")?
                    .handle(),
                0,
                S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES,
                0,
            );
            let pipeline = scratch
                .pipeline
                .as_ref()
                .context("S14 K=4 embedding pipeline 缺失")?;
            for dispatch in &scratch.dispatches {
                pipeline.cmd(context, scratch.command, dispatch);
            }
            let output_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(
                    scratch
                        .output
                        .as_ref()
                        .context("S14 K=4 output 缺失")?
                        .handle(),
                )
                .offset(0)
                .size(S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES);
            context.device.cmd_pipeline_barrier(
                scratch.command,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[output_barrier],
                &[],
            );
            let status_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .buffer(
                    scratch
                        .status
                        .as_ref()
                        .context("S14 K=4 status 缺失")?
                        .handle(),
                )
                .offset(0)
                .size(status_bytes);
            context.device.cmd_pipeline_barrier(
                scratch.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[status_barrier],
                &[],
            );
            context.device.end_command_buffer(scratch.command)?;
            let command_buffers = [scratch.command];
            context.device.queue_submit(
                context.q_graphics,
                &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
                scratch.fence,
            )?;
            scratch.submitted = true;
            context
                .device
                .wait_for_fences(&[scratch.fence], true, u64::MAX)?;
            scratch.completed = true;
        }

        let status = scratch.status.as_ref().context("S14 K=4 status 缺失")?;
        if status.mapped().is_null() || status.size() < status_bytes {
            bail!("S14 K=4 status mapping/capacity 漂移");
        }
        let status_slice = unsafe {
            std::slice::from_raw_parts(status.mapped() as *const u8, usize::try_from(status_bytes)?)
        };
        for lane in 0..S14_CAUSAL_BLOCK_K4_INPUT_TOKENS {
            let offset = usize::try_from(status_stride * lane as u64)?;
            let bytes: [u8; 4] = status_slice
                .get(offset..offset + 4)
                .context("S14 K=4 status readback 越界")?
                .try_into()
                .map_err(|_| anyhow!("S14 K=4 status readback 长度漂移"))?;
            validate_embedding_broadcast_status(u32::from_le_bytes(bytes))
                .with_context(|| format!("S14 K=4 lane {lane} embedding broadcast"))?;
        }

        let output = scratch
            .output
            .as_ref()
            .context("S14 K=4 output owner 缺失")?;
        if output.handle() == vk::Buffer::null()
            || output.size() < S14_CAUSAL_BLOCK_PRODUCTION_HIDDEN_BANK_BYTES
        {
            bail!("S14 K=4 output handle/capacity 漂移");
        }
        scratch.output.take().context("S14 K=4 output owner 消失")
    })();

    match result {
        Ok(output) => {
            scratch.destroy_safe(context);
            Ok(output)
        }
        Err(error) => {
            let mut safe_to_destroy = !scratch.submitted || scratch.completed;
            let mut drain_error = None;
            if scratch.submitted && !scratch.completed {
                match unsafe { context.device.queue_wait_idle(context.q_graphics) } {
                    Ok(()) => safe_to_destroy = true,
                    Err(observed) => drain_error = Some(observed),
                }
            }
            if safe_to_destroy {
                scratch.destroy_safe(context);
                Err(error)
            } else {
                // In-flight Vulkan 资源不能冒险销毁。GpuBuffer/pipeline/binder 均无
                // Rust Drop，因此丢弃 wrapper 会把 handle 安全保留到 context teardown。
                Err(anyhow!(
                    "{error:#}; graphics queue drain 也失败: {:?}; in-flight K=4 upload resources 保留到 Vulkan context teardown",
                    drain_error
                ))
            }
        }
    }
}

const S14_CAUSAL_BLOCK_K4_TOKEN_HIDDEN_BYTES: u64 =
    S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE as u64 * 2;

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        bail!("S14 K=4 alignment 不能为 0");
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .context("S14 K=4 aligned bytes overflow")
}

struct K4UploadScratch {
    source: Option<GpuBuffer>,
    status: Option<GpuBuffer>,
    output: Option<GpuBuffer>,
    pipeline: Option<S14EmbeddingBroadcastPipeline>,
    dispatches: Vec<S14EmbeddingBroadcastDispatch>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    submitted: bool,
    completed: bool,
}

impl K4UploadScratch {
    fn new() -> Self {
        Self {
            source: None,
            status: None,
            output: None,
            pipeline: None,
            dispatches: Vec::with_capacity(S14_CAUSAL_BLOCK_K4_INPUT_TOKENS),
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            submitted: false,
            completed: false,
        }
    }

    fn destroy_safe(&mut self, context: &VulkanContext) {
        for dispatch in self.dispatches.drain(..) {
            dispatch.binder.destroy(context);
        }
        if let Some(pipeline) = self.pipeline.take() {
            pipeline.destroy(context);
        }
        unsafe {
            if self.fence != vk::Fence::null() {
                context.device.destroy_fence(self.fence, None);
                self.fence = vk::Fence::null();
            }
            if self.pool != vk::CommandPool::null() {
                context.device.destroy_command_pool(self.pool, None);
                self.pool = vk::CommandPool::null();
                self.command = vk::CommandBuffer::null();
            }
        }
        if let Some(status) = self.status.take() {
            status.destroy(context);
        }
        if let Some(source) = self.source.take() {
            source.destroy(context);
        }
        if let Some(output) = self.output.take() {
            output.destroy(context);
        }
    }
}
