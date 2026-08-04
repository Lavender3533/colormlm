//! S14 StarFold K4/K8 的 HC-pre/HC-post 物理桥。
//!
//! 该 owner 把 production HC/QKV 的 `[K,4,4096]` BF16 输出转换为 routed/shared
//! experts 使用的 `[K,4096]` F32；exact reduce 完成后再把 `[K,4096]` F32 写回
//! production A/B HC bank。它不加载 routed expert，也不拥有 token commit。

use crate::{
    compute::{DescriptorBinder, StorageBufferSlice},
    s14_causal_block_hc_qkv_recorder::S14CausalBlockHiddenBank,
    s14_causal_block_layer::S14CausalBlockHiddenBinding,
    s14_e4m3_qdq::{validate_e4m3_qdq_status, S14E4m3QdqPipeline, S14E4m3QdqShape},
    s14_f32_to_bf16::{validate_f32_to_bf16_status, S14F32ToBf16Pipeline, S14F32ToBf16Shape},
    s14_hc_post::{validate_hc_post_status, S14HcPostPipeline, S14HcPostShape},
    s14_position0_hybrid_weight_arena::S14Position0StaticLayerLayout,
    s14_position0_paged_weight_arena::{
        S14Position0PagedWeightArena, S14Position0StaticLayerBinding,
    },
    s14_starfold_mxfp4_tile::S14StarfoldMxfp4ExternalSlice,
    s14_vulkan::{S14F32MatvecShape, S14HcPreShape, S14NumericPipelines},
    GpuBuffer, VulkanContext,
};
use anyhow::{anyhow, bail, Context, Result};
use ash::vk;
use std::sync::Arc;

const K4: usize = 4;
const K8: usize = 8;
const HIDDEN: u32 = 4096;
const HC_STREAMS: u64 = 4;
const HC_FLAT: u32 = HIDDEN * HC_STREAMS as u32;
const NORM_EPS: f32 = 1.0e-6;
const ALIGNMENT_FLOOR: u64 = 256;

#[derive(Clone, Copy, Debug)]
struct StridedRegion {
    offset: u64,
    stride: u64,
}

impl StridedRegion {
    fn lane(self, lane: usize) -> Result<u64> {
        self.offset
            .checked_add(
                self.stride
                    .checked_mul(lane as u64)
                    .context("S14 StarFold HC bridge lane stride overflow")?,
            )
            .context("S14 StarFold HC bridge lane offset overflow")
    }
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceLayout {
    residual: StridedRegion,
    hc_norm: StridedRegion,
    mixes: StridedRegion,
    branch_bf16: StridedRegion,
    branch_f32: StridedRegion,
    aux: StridedRegion,
    inverse: StridedRegion,
    qdq_scales: StridedRegion,
    reduced_f32: u64,
    reduced_bf16: u64,
    output_hc_bf16: u64,
    bytes: u64,
}

impl WorkspaceLayout {
    fn build(block_size: usize, alignment: u64) -> Result<Self> {
        validate_block_size(block_size)?;
        let mut cursor = 0u64;
        let residual = take_strided(
            &mut cursor,
            HC_STREAMS * HIDDEN as u64 * 2,
            block_size,
            alignment,
        )?;
        let hc_norm = take_strided(&mut cursor, HC_FLAT as u64 * 4, block_size, alignment)?;
        let mixes = take_strided(&mut cursor, 24 * 4, block_size, alignment)?;
        let branch_bf16 = take_strided(&mut cursor, HIDDEN as u64 * 2, block_size, alignment)?;
        let branch_f32 = take_strided(&mut cursor, HIDDEN as u64 * 4, block_size, alignment)?;
        let aux = take_strided(&mut cursor, 20 * 4, block_size, alignment)?;
        let inverse = take_strided(&mut cursor, 4, block_size, alignment)?;
        let qdq_scales = take_strided(
            &mut cursor,
            (HIDDEN / 128) as u64 * 4,
            block_size,
            alignment,
        )?;
        let reduced_f32 = take(
            &mut cursor,
            block_size as u64 * HIDDEN as u64 * 4,
            alignment,
        )?;
        let reduced_bf16 = take(
            &mut cursor,
            block_size as u64 * HIDDEN as u64 * 2,
            alignment,
        )?;
        let output_hc_bf16 = take(
            &mut cursor,
            block_size as u64 * HC_STREAMS * HIDDEN as u64 * 2,
            alignment,
        )?;
        let bytes = align_up(cursor, alignment)?;
        Ok(Self {
            residual,
            hc_norm,
            mixes,
            branch_bf16,
            branch_f32,
            aux,
            inverse,
            qdq_scales,
            reduced_f32,
            reduced_bf16,
            output_hc_bf16,
            bytes,
        })
    }
}

#[derive(Clone, Copy)]
struct StaticHcWeights<'a> {
    buffer: &'a GpuBuffer,
    logical_bytes: u64,
    stream_bank: Option<usize>,
    hc_fn: u64,
    hc_scale: u64,
    hc_base: u64,
    ffn_norm: u64,
}

#[derive(Clone, Copy, Debug)]
struct PreparedLayer {
    layer: u8,
    base_position: u32,
    block_size: usize,
    post_attention_hidden: S14CausalBlockHiddenBinding,
    next_hidden: S14CausalBlockHiddenBinding,
    static_stream_bank: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct S14StarfoldHcPrepareReceipt {
    pub layer: u8,
    pub base_position: u32,
    pub block_size: usize,
    pub moe_input_f32: S14StarfoldMxfp4ExternalSlice,
    pub reduced_output_f32: S14StarfoldMxfp4ExternalSlice,
    pub next_hidden: S14CausalBlockHiddenBinding,
    pub hc_pre_dispatch_calls: u32,
    pub qdq_dispatch_calls: u32,
    pub queue_submit_calls: u32,
    pub serial_token_forward_calls: u32,
    pub static_stream_bank: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldHcFinalizeReceipt {
    pub layer: u8,
    pub base_position: u32,
    pub block_size: usize,
    pub next_hidden: S14CausalBlockHiddenBinding,
    pub f32_to_bf16_dispatch_calls: u32,
    pub hc_post_dispatch_calls: u32,
    pub queue_submit_calls: u32,
    pub serial_token_forward_calls: u32,
}

pub struct S14StarfoldHcBridgeOwner {
    context: Arc<VulkanContext>,
    static_arena: Arc<S14Position0PagedWeightArena>,
    hidden_banks: [S14CausalBlockHiddenBank; 2],
    /// A/B bank 与 workspace 的最大物理容量。它不是当前 block 的 K。
    capacity_block_size: usize,
    /// 当前事务显式声明的物理 K；只能由 stage begin/finish 边界改变。
    active_block_size: Option<usize>,
    workspace: Option<GpuBuffer>,
    control: Option<GpuBuffer>,
    layout: WorkspaceLayout,
    numeric: Option<S14NumericPipelines>,
    qdq: Option<S14E4m3QdqPipeline>,
    f32_to_bf16: Option<S14F32ToBf16Pipeline>,
    hc_post: Option<S14HcPostPipeline>,
    command_pool: vk::CommandPool,
    prepare_command: vk::CommandBuffer,
    finalize_command: vk::CommandBuffer,
    prepare_fence: vk::Fence,
    finalize_fence: vk::Fence,
    prepare_binders: Vec<DescriptorBinder>,
    finalize_binders: Vec<DescriptorBinder>,
    prepared: Option<PreparedLayer>,
    destroyed: bool,
}

impl S14StarfoldHcBridgeOwner {
    pub fn new(
        context: Arc<VulkanContext>,
        static_arena: Arc<S14Position0PagedWeightArena>,
        hidden_banks: [S14CausalBlockHiddenBank; 2],
    ) -> Result<Self> {
        let capacity_block_size = validate_hidden_bank_capacity(&hidden_banks)?;
        let layout = WorkspaceLayout::build(capacity_block_size, storage_alignment(&context))?;
        let workspace = new_device_buffer(&context, layout.bytes)?;
        let control = match new_control_buffer(&context, 4) {
            Ok(buffer) => buffer,
            Err(error) => {
                workspace.destroy(&context);
                return Err(error);
            }
        };
        let numeric = S14NumericPipelines::new(&context)?;
        let qdq = S14E4m3QdqPipeline::new(&context)?;
        let f32_to_bf16 = S14F32ToBf16Pipeline::new(&context)?;
        let hc_post = S14HcPostPipeline::new(&context)?;
        let (command_pool, commands, fences) = create_commands(&context)?;
        Ok(Self {
            context,
            static_arena,
            hidden_banks,
            capacity_block_size,
            active_block_size: None,
            workspace: Some(workspace),
            control: Some(control),
            layout,
            numeric: Some(numeric),
            qdq: Some(qdq),
            f32_to_bf16: Some(f32_to_bf16),
            hc_post: Some(hc_post),
            command_pool,
            prepare_command: commands[0],
            finalize_command: commands[1],
            prepare_fence: fences[0],
            finalize_fence: fences[1],
            prepare_binders: Vec::with_capacity(capacity_block_size * 4),
            finalize_binders: Vec::with_capacity(capacity_block_size + 1),
            prepared: None,
            destroyed: false,
        })
    }

    pub(crate) fn validate_hidden_bank_rebind(
        &self,
        hidden_banks: &[S14CausalBlockHiddenBank; 2],
    ) -> Result<()> {
        self.ensure_live()?;
        if self.prepared.is_some() || self.active_block_size.is_some() {
            bail!("S14 StarFold HC bridge 仍有 active/prepared block，禁止 rebind hidden banks");
        }
        let capacity_block_size = validate_hidden_bank_capacity(hidden_banks)?;
        if capacity_block_size != self.capacity_block_size {
            bail!(
                "S14 StarFold HC bridge rebind capacity 漂移: owner={} banks={capacity_block_size}",
                self.capacity_block_size
            );
        }
        Ok(())
    }

    /// command/fence/pipeline/workspace owner 原地保留；只在上一块完全 drain 后替换 A/B
    /// hidden owner 引用。
    pub(crate) fn rebind_hidden_banks(
        &mut self,
        hidden_banks: [S14CausalBlockHiddenBank; 2],
    ) -> Result<()> {
        self.validate_hidden_bank_rebind(&hidden_banks)?;
        self.release_finalize_binders();
        self.release_prepare_binders();
        self.hidden_banks = hidden_banks;
        Ok(())
    }

    pub(crate) fn begin_block(&mut self, block_size: usize) -> Result<()> {
        self.ensure_live()?;
        validate_block_size(block_size)?;
        if self.active_block_size.is_some() || self.prepared.is_some() {
            bail!("S14 StarFold HC bridge 上一 block 尚未 finish/drain");
        }
        if block_size > self.capacity_block_size {
            bail!(
                "S14 StarFold HC bridge active K 超过常驻容量: active={block_size} capacity={}",
                self.capacity_block_size
            );
        }
        for bank in &self.hidden_banks {
            bank.binding(block_size, 0)
                .context("S14 StarFold HC bridge active K 无法绑定 hidden bank")?;
        }
        self.active_block_size = Some(block_size);
        Ok(())
    }

    pub(crate) fn finish_block(&mut self) -> Result<()> {
        self.ensure_live()?;
        if self.active_block_size.is_none() || self.prepared.is_some() {
            bail!("S14 StarFold HC bridge block 尚未开始或仍有 prepared layer");
        }
        self.active_block_size = None;
        Ok(())
    }

    pub(crate) fn abort_block(&mut self) {
        self.prepared = None;
        self.active_block_size = None;
    }

    fn active_block_size(&self) -> Result<usize> {
        self.active_block_size
            .context("S14 StarFold HC bridge 缺少显式 active K")
    }

    pub fn prepare_layer(
        &mut self,
        layer: u8,
        base_position: u32,
        post_attention_hidden: S14CausalBlockHiddenBinding,
    ) -> Result<S14StarfoldHcPrepareReceipt> {
        self.ensure_live()?;
        if self.prepared.is_some() {
            bail!("S14 StarFold HC bridge 上一层尚未 finalize");
        }
        let block_size = self.active_block_size()?;
        validate_hidden(post_attention_hidden, block_size)?;
        let next_hidden = opposite_hidden_bank(
            &self.hidden_banks,
            post_attention_hidden,
            block_size,
            post_attention_hidden
                .generation
                .checked_add(1)
                .context("S14 StarFold HC bridge hidden generation overflow")?,
        )?;
        self.release_prepare_binders();
        let static_arena = Arc::clone(&self.static_arena);
        let weights = resolve_static_hc_weights(&static_arena, layer)?;
        unsafe {
            self.control()?.write_at(0, &0u32.to_le_bytes());
            self.context
                .device
                .reset_command_buffer(self.prepare_command, vk::CommandBufferResetFlags::empty())?;
            self.context.device.begin_command_buffer(
                self.prepare_command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.context.device.cmd_copy_buffer(
                self.prepare_command,
                post_attention_hidden.buffer,
                self.workspace()?.handle(),
                &[vk::BufferCopy::default()
                    .src_offset(post_attention_hidden.offset)
                    .dst_offset(self.layout.residual.offset)
                    .size(post_attention_hidden.bytes)],
            );
            transfer_to_compute_barrier(&self.context, self.prepare_command);
        }

        let numeric = self.numeric.as_ref().context("HC bridge numeric 已销毁")?;
        let qdq = self.qdq.as_ref().context("HC bridge QDQ 已销毁")?;
        let workspace = self.workspace()?;
        let control = self.control()?;
        let hc_shape = S14HcPreShape::new(HIDDEN)?;
        let mut binders = Vec::with_capacity(block_size * 4);
        for lane in 0..block_size {
            let residual = self.layout.residual.lane(lane)?;
            let hc_norm = self.layout.hc_norm.lane(lane)?;
            let mixes = self.layout.mixes.lane(lane)?;
            let branch_bf16 = self.layout.branch_bf16.lane(lane)?;
            let branch_f32 = self.layout.branch_f32.lane(lane)?;
            let aux = self.layout.aux.lane(lane)?;
            let inverse = self.layout.inverse.lane(lane)?;
            let dispatch = numeric.bind_hc_normalize_input_arena(
                &self.context,
                hc_shape,
                NORM_EPS,
                workspace,
                self.layout.bytes,
                residual,
                hc_norm,
                inverse,
            )?;
            unsafe {
                numeric.cmd_hc_normalize_input(&self.context, self.prepare_command, &dispatch);
                compute_barrier(&self.context, self.prepare_command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric.bind_f32_matvec_arenas(
                &self.context,
                S14F32MatvecShape::new(24, HC_FLAT, 1)?,
                weights.buffer,
                weights.logical_bytes,
                weights.hc_fn,
                workspace,
                self.layout.bytes,
                hc_norm,
                workspace,
                self.layout.bytes,
                mixes,
            )?;
            unsafe {
                numeric.cmd_f32_matvec(&self.context, self.prepare_command, &dispatch);
                compute_barrier(&self.context, self.prepare_command);
            }
            binders.push(dispatch.binder);
            let dispatch = numeric.bind_hc_split_reduce_norm_arenas(
                &self.context,
                hc_shape,
                NORM_EPS,
                weights.buffer,
                weights.logical_bytes,
                weights.hc_scale,
                weights.hc_base,
                weights.ffn_norm,
                workspace,
                self.layout.bytes,
                residual,
                mixes,
                branch_bf16,
                branch_f32,
                aux,
                inverse,
            )?;
            unsafe {
                numeric.cmd_hc_split_reduce_norm(&self.context, self.prepare_command, &dispatch);
                compute_barrier(&self.context, self.prepare_command);
            }
            binders.push(dispatch.binder);
            let dispatch = qdq.bind_slices(
                &self.context,
                S14E4m3QdqShape::new(1, HIDDEN, 128)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: branch_bf16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.layout.qdq_scales.lane(lane)?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: branch_f32,
                },
                StorageBufferSlice::whole(control),
            )?;
            unsafe {
                qdq.cmd(&self.context, self.prepare_command, &dispatch);
                compute_barrier(&self.context, self.prepare_command);
            }
            binders.push(dispatch.binder);
        }
        unsafe {
            publish_and_host_barrier(&self.context, self.prepare_command);
            self.context
                .device
                .end_command_buffer(self.prepare_command)?;
            self.context.device.reset_fences(&[self.prepare_fence])?;
            self.context.device.queue_submit(
                self.context.q_graphics,
                &[vk::SubmitInfo::default()
                    .command_buffers(std::slice::from_ref(&self.prepare_command))],
                self.prepare_fence,
            )?;
            self.context
                .device
                .wait_for_fences(&[self.prepare_fence], true, u64::MAX)?;
        }
        validate_e4m3_qdq_status(self.read_status()?)?;
        let input = external_slice(
            workspace,
            self.layout.branch_f32.offset,
            block_size as u64 * HIDDEN as u64 * 4,
        );
        let reduced = external_slice(
            workspace,
            self.layout.reduced_f32,
            block_size as u64 * HIDDEN as u64 * 4,
        );
        self.prepare_binders = binders;
        self.prepared = Some(PreparedLayer {
            layer,
            base_position,
            block_size,
            post_attention_hidden,
            next_hidden,
            static_stream_bank: weights.stream_bank,
        });
        Ok(S14StarfoldHcPrepareReceipt {
            layer,
            base_position,
            block_size,
            moe_input_f32: input,
            reduced_output_f32: reduced,
            next_hidden,
            hc_pre_dispatch_calls: block_size as u32 * 3,
            qdq_dispatch_calls: block_size as u32,
            queue_submit_calls: 1,
            serial_token_forward_calls: 0,
            static_stream_bank: weights.stream_bank,
        })
    }

    pub fn finalize_layer(
        &mut self,
        layer: u8,
        base_position: u32,
        reduced_output_f32: S14StarfoldMxfp4ExternalSlice,
    ) -> Result<S14StarfoldHcFinalizeReceipt> {
        self.ensure_live()?;
        let prepared = self
            .prepared
            .context("S14 StarFold HC bridge 缺少 prepared layer")?;
        if prepared.layer != layer || prepared.base_position != base_position {
            bail!("S14 StarFold HC bridge finalize layer/base identity 漂移");
        }
        let block_size = self.active_block_size()?;
        if prepared.block_size != block_size {
            bail!("S14 StarFold HC bridge finalize block_size identity 漂移");
        }
        let expected = external_slice(
            self.workspace()?,
            self.layout.reduced_f32,
            block_size as u64 * HIDDEN as u64 * 4,
        );
        if reduced_output_f32.buffer != expected.buffer
            || reduced_output_f32.capacity_bytes != expected.capacity_bytes
            || reduced_output_f32.offset != expected.offset
            || reduced_output_f32.logical_bytes != expected.logical_bytes
        {
            bail!("S14 StarFold HC bridge exact-reduce output binding 漂移");
        }
        self.release_finalize_binders();
        unsafe {
            self.control()?.write_at(0, &0u32.to_le_bytes());
            self.context.device.reset_command_buffer(
                self.finalize_command,
                vk::CommandBufferResetFlags::empty(),
            )?;
            self.context.device.begin_command_buffer(
                self.finalize_command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            compute_barrier(&self.context, self.finalize_command);
        }
        let workspace = self.workspace()?;
        let control = self.control()?;
        let f32_to_bf16 = self
            .f32_to_bf16
            .as_ref()
            .context("HC bridge F32->BF16 已销毁")?;
        let hc_post = self.hc_post.as_ref().context("HC bridge HC-post 已销毁")?;
        let dispatch = f32_to_bf16.bind_slices(
            &self.context,
            S14F32ToBf16Shape::new(block_size as u32 * HIDDEN)?,
            StorageBufferSlice {
                buffer: workspace,
                offset: self.layout.reduced_f32,
            },
            StorageBufferSlice {
                buffer: workspace,
                offset: self.layout.reduced_bf16,
            },
            StorageBufferSlice::whole(control),
        )?;
        unsafe {
            f32_to_bf16.cmd(&self.context, self.finalize_command, &dispatch);
            compute_barrier(&self.context, self.finalize_command);
        }
        let mut binders = vec![dispatch.binder];
        for lane in 0..block_size {
            let dispatch = hc_post.bind_slices(
                &self.context,
                S14HcPostShape::new(HIDDEN)?,
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.layout.reduced_bf16 + lane as u64 * HIDDEN as u64 * 2,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.layout.residual.lane(lane)?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.layout.aux.lane(lane)?,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.layout.aux.lane(lane)? + 16,
                },
                StorageBufferSlice {
                    buffer: workspace,
                    offset: self.layout.output_hc_bf16
                        + lane as u64 * HC_STREAMS * HIDDEN as u64 * 2,
                },
                StorageBufferSlice::whole(control),
            )?;
            unsafe {
                hc_post.cmd(&self.context, self.finalize_command, &dispatch);
                compute_barrier(&self.context, self.finalize_command);
            }
            binders.push(dispatch.binder);
        }
        unsafe {
            compute_to_transfer_barrier(&self.context, self.finalize_command);
            self.context.device.cmd_copy_buffer(
                self.finalize_command,
                workspace.handle(),
                prepared.next_hidden.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(self.layout.output_hc_bf16)
                    .dst_offset(prepared.next_hidden.offset)
                    .size(prepared.next_hidden.bytes)],
            );
            transfer_and_shader_publish_barrier(&self.context, self.finalize_command);
            self.context
                .device
                .end_command_buffer(self.finalize_command)?;
            self.context.device.reset_fences(&[self.finalize_fence])?;
            self.context.device.queue_submit(
                self.context.q_graphics,
                &[vk::SubmitInfo::default()
                    .command_buffers(std::slice::from_ref(&self.finalize_command))],
                self.finalize_fence,
            )?;
            self.context
                .device
                .wait_for_fences(&[self.finalize_fence], true, u64::MAX)?;
        }
        self.finalize_binders = binders;
        let status = self.read_status()?;
        validate_f32_to_bf16_status(status)?;
        validate_hc_post_status(status)?;
        self.prepared = None;
        Ok(S14StarfoldHcFinalizeReceipt {
            layer,
            base_position,
            block_size,
            next_hidden: prepared.next_hidden,
            f32_to_bf16_dispatch_calls: 1,
            hc_post_dispatch_calls: block_size as u32,
            queue_submit_calls: 1,
            serial_token_forward_calls: 0,
        })
    }

    pub fn abort_prepared_layer(&mut self) {
        self.prepared = None;
    }

    pub fn destroy(&mut self) -> Result<()> {
        if self.destroyed {
            return Ok(());
        }
        unsafe { self.context.device.device_wait_idle()? };
        self.release_finalize_binders();
        self.release_prepare_binders();
        unsafe {
            self.context.device.destroy_fence(self.finalize_fence, None);
            self.context.device.destroy_fence(self.prepare_fence, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        if let Some(pipeline) = self.hc_post.take() {
            pipeline.destroy(&self.context);
        }
        if let Some(pipeline) = self.f32_to_bf16.take() {
            pipeline.destroy(&self.context);
        }
        if let Some(pipeline) = self.qdq.take() {
            pipeline.destroy(&self.context);
        }
        if let Some(pipeline) = self.numeric.take() {
            pipeline.destroy(&self.context);
        }
        if let Some(buffer) = self.control.take() {
            buffer.destroy(&self.context);
        }
        if let Some(buffer) = self.workspace.take() {
            buffer.destroy(&self.context);
        }
        self.prepared = None;
        self.active_block_size = None;
        self.destroyed = true;
        Ok(())
    }

    fn ensure_live(&self) -> Result<()> {
        if self.destroyed {
            bail!("S14 StarFold HC bridge 已销毁");
        }
        Ok(())
    }

    fn workspace(&self) -> Result<&GpuBuffer> {
        self.workspace
            .as_ref()
            .context("HC bridge workspace 已销毁")
    }

    fn control(&self) -> Result<&GpuBuffer> {
        self.control.as_ref().context("HC bridge control 已销毁")
    }

    fn read_status(&self) -> Result<u32> {
        let pointer = self.control()?.mapped();
        if pointer.is_null() {
            bail!("S14 StarFold HC bridge status 未映射");
        }
        Ok(unsafe { u32::from_le(std::ptr::read_unaligned(pointer.cast::<u32>())) })
    }

    fn release_prepare_binders(&mut self) {
        for binder in self.prepare_binders.drain(..).rev() {
            binder.destroy(&self.context);
        }
    }

    fn release_finalize_binders(&mut self) {
        for binder in self.finalize_binders.drain(..).rev() {
            binder.destroy(&self.context);
        }
    }
}

impl Drop for S14StarfoldHcBridgeOwner {
    fn drop(&mut self) {
        let _ = self.destroy();
    }
}

fn resolve_static_hc_weights<'a>(
    arena: &'a S14Position0PagedWeightArena,
    layer: u8,
) -> Result<StaticHcWeights<'a>> {
    let ready = arena.ready_static_layer(layer)?;
    let (buffer, layout, stream_bank) = match ready {
        S14Position0StaticLayerBinding::Resident { buffer, layout } => (buffer, layout, None),
        S14Position0StaticLayerBinding::Streamed {
            bank,
            buffer,
            layout,
        } => (buffer, layout, Some(bank)),
    };
    if layout.layer != layer || layout.requested_bytes > buffer.size() {
        bail!("S14 StarFold HC bridge static layer identity/capacity 漂移");
    }
    Ok(StaticHcWeights {
        buffer,
        logical_bytes: layout.requested_bytes,
        stream_bank,
        hc_fn: static_asset(layout, layer, "hc_ffn_fn", 24 * HC_FLAT as u64 * 4)?,
        hc_scale: static_asset(layout, layer, "hc_ffn_scale", 3 * 4)?,
        hc_base: static_asset(layout, layer, "hc_ffn_base", 24 * 4)?,
        ffn_norm: static_asset(layout, layer, "ffn_norm.weight", HIDDEN as u64 * 2)?,
    })
}

fn static_asset(
    layout: &S14Position0StaticLayerLayout,
    layer: u8,
    suffix: &str,
    expected_bytes: u64,
) -> Result<u64> {
    let tensor = format!("layers.{layer}.{suffix}");
    let mut matches = layout.assets.iter().filter(|asset| asset.tensor == tensor);
    let asset = matches
        .next()
        .with_context(|| format!("S14 StarFold HC bridge 缺少 {tensor}"))?;
    if matches.next().is_some()
        || asset.bytes != expected_bytes
        || asset
            .local_offset
            .checked_add(asset.bytes)
            .is_none_or(|end| end > layout.requested_bytes)
    {
        bail!("S14 StarFold HC bridge static asset 漂移: {tensor}");
    }
    Ok(asset.local_offset)
}

fn opposite_hidden_bank(
    banks: &[S14CausalBlockHiddenBank; 2],
    current: S14CausalBlockHiddenBinding,
    block_size: usize,
    generation: u64,
) -> Result<S14CausalBlockHiddenBinding> {
    validate_block_size(block_size)?;
    let mut output = None;
    let mut current_seen = false;
    for bank in banks {
        let binding = bank.binding(block_size, generation)?;
        if binding.buffer == current.buffer && binding.offset == current.offset {
            current_seen = true;
        } else {
            if output.is_some() {
                bail!("S14 StarFold HC bridge A/B 输出 identity 非唯一");
            }
            output = Some(binding);
        }
    }
    if !current_seen {
        bail!("S14 StarFold HC bridge post-attention hidden 不属于 A/B bank");
    }
    output.context("S14 StarFold HC bridge 缺少 opposite hidden bank")
}

fn validate_hidden(binding: S14CausalBlockHiddenBinding, block_size: usize) -> Result<()> {
    validate_block_size(block_size)?;
    let expected = hidden_bytes(block_size)?;
    if binding.buffer == vk::Buffer::null()
        || binding.offset % 4 != 0
        || binding.bytes != expected
        || binding.block_size != block_size
    {
        bail!(
            "S14 StarFold HC bridge hidden 不是精确 [K,4,4096] BF16: active_K={block_size} expected_bytes={expected} actual=(buffer={:?}, offset={}, bytes={}, K={}, generation={})",
            binding.buffer,
            binding.offset,
            binding.bytes,
            binding.block_size,
            binding.generation,
        );
    }
    Ok(())
}

fn validate_hidden_bank_capacity(banks: &[S14CausalBlockHiddenBank; 2]) -> Result<usize> {
    let block_size = block_size_from_hidden_bank_capacity(banks[0].capacity_bytes)?;
    let right_block_size = block_size_from_hidden_bank_capacity(banks[1].capacity_bytes)?;
    if block_size != right_block_size {
        bail!(
            "S14 StarFold HC bridge A/B hidden bank capacity 漂移: left_K={block_size} right_K={right_block_size}"
        );
    }
    let left = banks[0].binding(block_size, 0)?;
    let right = banks[1].binding(block_size, 0)?;
    if left.buffer == right.buffer
        && left.offset < right.offset + right.bytes
        && right.offset < left.offset + left.bytes
    {
        bail!("S14 StarFold HC bridge A/B hidden banks 重叠");
    }
    Ok(block_size)
}

fn block_size_from_hidden_bank_capacity(capacity_bytes: u64) -> Result<usize> {
    for block_size in [K4, K8] {
        if capacity_bytes == hidden_bytes(block_size)? {
            return Ok(block_size);
        }
    }
    bail!("S14 StarFold HC bridge hidden bank capacity 不能唯一推导 K4/K8: bytes={capacity_bytes}")
}

fn validate_block_size(block_size: usize) -> Result<()> {
    if !matches!(block_size, K4 | K8) {
        bail!("S14 StarFold HC bridge 仅支持 K4/K8: block_size={block_size}");
    }
    Ok(())
}

fn hidden_bytes(block_size: usize) -> Result<u64> {
    validate_block_size(block_size)?;
    (block_size as u64)
        .checked_mul(HC_STREAMS)
        .and_then(|value| value.checked_mul(HIDDEN as u64))
        .and_then(|value| value.checked_mul(2))
        .context("S14 StarFold HC bridge hidden bytes overflow")
}

fn create_commands(
    context: &VulkanContext,
) -> Result<(vk::CommandPool, [vk::CommandBuffer; 2], [vk::Fence; 2])> {
    let pool = unsafe {
        context.device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(context.qf_graphics)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?
    };
    let commands = unsafe {
        context.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(2),
        )?
    };
    let commands: [vk::CommandBuffer; 2] = commands
        .try_into()
        .map_err(|_| anyhow!("S14 StarFold HC bridge command count 漂移"))?;
    let left = unsafe {
        context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?
    };
    let right = match unsafe {
        context
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
    } {
        Ok(fence) => fence,
        Err(error) => {
            unsafe {
                context.device.destroy_fence(left, None);
                context.device.destroy_command_pool(pool, None);
            }
            return Err(error.into());
        }
    };
    Ok((pool, commands, [left, right]))
}

fn external_slice(
    buffer: &GpuBuffer,
    offset: u64,
    logical_bytes: u64,
) -> S14StarfoldMxfp4ExternalSlice {
    S14StarfoldMxfp4ExternalSlice {
        buffer: buffer.handle(),
        capacity_bytes: buffer.size(),
        offset,
        logical_bytes,
    }
}

fn new_device_buffer(context: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new_vram(
        context,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
    )
}

fn new_control_buffer(context: &VulkanContext, bytes: u64) -> Result<GpuBuffer> {
    GpuBuffer::new(
        context,
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::empty(),
        true,
    )
}

fn storage_alignment(context: &VulkanContext) -> u64 {
    unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical)
            .limits
            .min_storage_buffer_offset_alignment
    }
    .max(ALIGNMENT_FLOOR)
}

fn take_strided(
    cursor: &mut u64,
    bytes: u64,
    block_size: usize,
    alignment: u64,
) -> Result<StridedRegion> {
    validate_block_size(block_size)?;
    let stride = align_up(bytes, alignment)?;
    let region_bytes = stride
        .checked_mul(block_size as u64)
        .context("S14 StarFold HC bridge strided workspace overflow")?;
    let offset = take(cursor, region_bytes, alignment)?;
    Ok(StridedRegion { offset, stride })
}

fn take(cursor: &mut u64, bytes: u64, alignment: u64) -> Result<u64> {
    let offset = align_up(*cursor, alignment)?;
    *cursor = offset
        .checked_add(bytes)
        .context("S14 StarFold HC bridge workspace overflow")?;
    Ok(offset)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        bail!("S14 StarFold HC bridge alignment 非法");
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .context("S14 StarFold HC bridge alignment overflow")
}

unsafe fn transfer_to_compute_barrier(context: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    context.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

unsafe fn compute_barrier(context: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    context.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

unsafe fn compute_to_transfer_barrier(context: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    context.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

unsafe fn publish_and_host_barrier(context: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ);
    context.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

unsafe fn transfer_and_shader_publish_barrier(context: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::HOST_READ);
    context.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::HOST,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}
