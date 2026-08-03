//! K=4/8 causal-block attention/router 的最小真实 Vulkan recorder。
//!
//! 本模块故意从已经由正式 HC/Q/KV 投影产生的 device slices 开始：
//! `[K,64,512]` query、K 行 current KV 与 `[K,4096]` router input。它不会把
//! `[K,4,4096]` HC streams 直接冒充 query/router input，也不实现完整
//! `S14CausalBlockLayerBackend`。一次 `record` 在同一个 command buffer 内录制：
//!
//! 1. 一个 `dispatch(64,K,1)` 的 block-causal attention；
//! 2. 一个 BF16 `[256,4096] x F32 [K,4096]` 共享权重扫描；
//! 3. 一个 `dispatch(K,1,1)` 的在线 top-6 后处理。
//!
//! 输出仍需正式 HC/post-attention 写回后才能成为 `[K,4,4096]` hidden binding。

use crate::compute::{ComputePipeline, DescriptorBinder, StorageBufferSlice};
use crate::s14_route_postprocess_gpu::S14RoutePostprocessGpuMode;
use crate::s14_vulkan::{S14Bf16MatvecDispatch, S14Bf16MatvecShape, S14NumericPipelines};
use crate::{GpuBuffer, VulkanContext};
use anyhow::{bail, Context, Result};
use ash::vk;

pub const S14_CAUSAL_BLOCK_ATTENTION_ROUTER_K4: u32 = 4;
pub const S14_CAUSAL_BLOCK_ATTENTION_ROUTER_K8: u32 = 8;
pub const S14_CAUSAL_BLOCK_ATTENTION_HEADS: u32 = 64;
pub const S14_CAUSAL_BLOCK_ATTENTION_HEAD_DIM: u32 = 512;
pub const S14_CAUSAL_BLOCK_ROUTER_EXPERTS: u32 = 256;
pub const S14_CAUSAL_BLOCK_ROUTER_TOP_K: u32 = 6;
pub const S14_CAUSAL_BLOCK_ROUTER_HIDDEN: u32 = 4096;

pub const S14_CAUSAL_BLOCK_STATUS_ATTENTION_INVALID_SHAPE: u32 = 1;
pub const S14_CAUSAL_BLOCK_STATUS_ATTENTION_NON_FINITE: u32 = 2;
pub const S14_CAUSAL_BLOCK_STATUS_ROUTE_NON_FINITE_LOGIT_OR_SCORE: u32 = 1 << 8;
pub const S14_CAUSAL_BLOCK_STATUS_ROUTE_NON_FINITE_BIAS: u32 = 1 << 9;
pub const S14_CAUSAL_BLOCK_STATUS_ROUTE_INVALID_PHYSICAL_ID: u32 = 1 << 10;
pub const S14_CAUSAL_BLOCK_STATUS_ROUTE_INVALID_NORMALIZATION: u32 = 1 << 11;
pub const S14_CAUSAL_BLOCK_STATUS_ROUTE_INVALID_MODE: u32 = 1 << 12;
pub const S14_CAUSAL_BLOCK_STATUS_KNOWN_MASK: u32 = S14_CAUSAL_BLOCK_STATUS_ATTENTION_INVALID_SHAPE
    | S14_CAUSAL_BLOCK_STATUS_ATTENTION_NON_FINITE
    | S14_CAUSAL_BLOCK_STATUS_ROUTE_NON_FINITE_LOGIT_OR_SCORE
    | S14_CAUSAL_BLOCK_STATUS_ROUTE_NON_FINITE_BIAS
    | S14_CAUSAL_BLOCK_STATUS_ROUTE_INVALID_PHYSICAL_ID
    | S14_CAUSAL_BLOCK_STATUS_ROUTE_INVALID_NORMALIZATION
    | S14_CAUSAL_BLOCK_STATUS_ROUTE_INVALID_MODE;

const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const STATUS_BYTES: u64 = 4;
const ROPE_SCALARS_PER_POSITION: u64 = 64;

pub const S14_CAUSAL_BLOCK_ATTENTION_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_causal_block_attention.spv"));
pub const S14_CAUSAL_BLOCK_ROUTE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/s14_causal_block_route.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14CausalBlockAttentionRouterShape {
    pub block_size: u32,
    pub base_position: u32,
    pub committed_window_rows: u32,
}

impl S14CausalBlockAttentionRouterShape {
    /// 闭合 position0..126 的 pre-compression contiguous window；base0 的 committed rows
    /// 数学上就是0，仅由上层 ForcedPrefill bootstrap 门开放。
    pub fn new(block_size: u32, base_position: u32, committed_window_rows: u32) -> Result<Self> {
        if !matches!(
            block_size,
            S14_CAUSAL_BLOCK_ATTENTION_ROUTER_K4 | S14_CAUSAL_BLOCK_ATTENTION_ROUTER_K8
        ) {
            bail!("S14 causal-block attention/router K 只允许4或8");
        }
        if committed_window_rows != base_position {
            bail!("S14 causal-block attention 要求 committed_window_rows=base_position");
        }
        let end = base_position
            .checked_add(block_size)
            .context("S14 causal-block attention position overflow")?;
        if end > 127 {
            bail!("S14 causal-block attention 首版只允许 block end<=127");
        }
        let shape = Self {
            block_size,
            base_position,
            committed_window_rows,
        };
        for bytes in [
            shape.query_bf16_bytes()?,
            shape.committed_window_bf16_bytes()?,
            shape.current_block_kv_bf16_bytes()?,
            shape.rope_f32_bytes()?,
            shape.attention_output_bf16_bytes()?,
            shape.router_input_f32_bytes()?,
            shape.router_logits_f32_bytes()?,
            shape.route_output_bytes()?,
        ] {
            if bytes == 0 {
                bail!("S14 causal-block attention/router buffer bytes 不能为空");
            }
        }
        Ok(shape)
    }

    pub fn query_bf16_bytes(self) -> Result<u64> {
        checked_bytes(
            u64::from(self.block_size)
                * u64::from(S14_CAUSAL_BLOCK_ATTENTION_HEADS)
                * u64::from(S14_CAUSAL_BLOCK_ATTENTION_HEAD_DIM),
            BF16_BYTES,
            "causal-block query",
        )
    }

    pub fn committed_window_bf16_bytes(self) -> Result<u64> {
        checked_bytes(
            // Vulkan descriptor range 不能为0。base0 的 shader committed count 仍为0，
            // 因而不会读取这一哨兵行；这里只保持 router/ratio4 共用 owner 可绑定。
            u64::from(self.committed_window_rows.max(1))
                * u64::from(S14_CAUSAL_BLOCK_ATTENTION_HEAD_DIM),
            BF16_BYTES,
            "causal-block committed window",
        )
    }

    pub fn current_block_kv_bf16_bytes(self) -> Result<u64> {
        checked_bytes(
            u64::from(self.block_size) * u64::from(S14_CAUSAL_BLOCK_ATTENTION_HEAD_DIM),
            BF16_BYTES,
            "causal-block current KV",
        )
    }

    pub const fn sink_f32_bytes(self) -> u64 {
        S14_CAUSAL_BLOCK_ATTENTION_HEADS as u64 * F32_BYTES
    }

    pub fn rope_f32_bytes(self) -> Result<u64> {
        checked_bytes(
            u64::from(self.block_size) * ROPE_SCALARS_PER_POSITION,
            F32_BYTES,
            "causal-block RoPE",
        )
    }

    pub fn attention_output_bf16_bytes(self) -> Result<u64> {
        self.query_bf16_bytes()
    }

    pub const fn router_weight_bf16_bytes(self) -> u64 {
        S14_CAUSAL_BLOCK_ROUTER_EXPERTS as u64 * S14_CAUSAL_BLOCK_ROUTER_HIDDEN as u64 * BF16_BYTES
    }

    pub fn router_input_f32_bytes(self) -> Result<u64> {
        checked_bytes(
            u64::from(self.block_size) * u64::from(S14_CAUSAL_BLOCK_ROUTER_HIDDEN),
            F32_BYTES,
            "causal-block router input",
        )
    }

    pub fn router_logits_f32_bytes(self) -> Result<u64> {
        checked_bytes(
            u64::from(self.block_size) * u64::from(S14_CAUSAL_BLOCK_ROUTER_EXPERTS),
            F32_BYTES,
            "causal-block router logits",
        )
    }

    pub fn route_aux_bytes(self, mode: S14RoutePostprocessGpuMode) -> Result<u64> {
        match mode {
            S14RoutePostprocessGpuMode::BiasTop6 => {
                Ok(S14_CAUSAL_BLOCK_ROUTER_EXPERTS as u64 * F32_BYTES)
            }
            S14RoutePostprocessGpuMode::PhysicalIds => checked_bytes(
                u64::from(self.block_size) * u64::from(S14_CAUSAL_BLOCK_ROUTER_TOP_K),
                F32_BYTES,
                "causal-block physical IDs",
            ),
        }
    }

    pub fn route_output_bytes(self) -> Result<u64> {
        checked_bytes(
            u64::from(self.block_size) * u64::from(S14_CAUSAL_BLOCK_ROUTER_TOP_K),
            F32_BYTES,
            "causal-block route output",
        )
    }
}

#[derive(Clone, Copy)]
pub struct S14CausalBlockAttentionRouterBindings<'a> {
    pub query_bf16: StorageBufferSlice<'a>,
    pub committed_window_kv_bf16: StorageBufferSlice<'a>,
    pub current_block_kv_bf16: StorageBufferSlice<'a>,
    pub sink_f32: StorageBufferSlice<'a>,
    pub rope_f32: StorageBufferSlice<'a>,
    pub attention_output_bf16: StorageBufferSlice<'a>,
    pub router_weight_bf16: StorageBufferSlice<'a>,
    pub router_input_f32: StorageBufferSlice<'a>,
    pub router_logits_f32: StorageBufferSlice<'a>,
    pub route_aux: StorageBufferSlice<'a>,
    pub expert_ids_u32: StorageBufferSlice<'a>,
    pub route_weights_f32: StorageBufferSlice<'a>,
    /// Candidate 开始前必须由 owner 清零；本 recorder 只做 atomic sticky OR。
    pub sticky_status_u32: StorageBufferSlice<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14CausalBlockAttentionRouterRecordingReceipt {
    pub shape: S14CausalBlockAttentionRouterShape,
    pub recorder_calls: u32,
    pub device_rows: u32,
    pub attention_dispatch_calls: u32,
    pub router_weight_scan_calls: u32,
    pub route_postprocess_dispatch_calls: u32,
    pub serial_token_forward_calls: u32,
    /// 本 recorder 输出 attention 与 route device rows，但不伪造 HC/post-attention 写回。
    pub hc_hidden_integration_complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14CausalBlockAttentionRecordingReceipt {
    pub shape: S14CausalBlockAttentionRouterShape,
    pub recorder_calls: u32,
    pub device_rows: u32,
    pub attention_dispatch_calls: u32,
    pub serial_token_forward_calls: u32,
}

impl S14CausalBlockAttentionRecordingReceipt {
    pub fn validate(self) -> Result<()> {
        if self.recorder_calls != 1
            || self.device_rows != self.shape.block_size
            || self.attention_dispatch_calls != 1
            || self.serial_token_forward_calls != 0
        {
            bail!("S14 causal-block attention 回执不能证明一次 K-row device dispatch");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14CausalBlockRouterRecordingReceipt {
    pub shape: S14CausalBlockAttentionRouterShape,
    pub recorder_calls: u32,
    pub device_rows: u32,
    pub router_weight_scan_calls: u32,
    pub route_postprocess_dispatch_calls: u32,
    pub serial_token_forward_calls: u32,
}

impl S14CausalBlockRouterRecordingReceipt {
    pub fn validate(self) -> Result<()> {
        if self.recorder_calls != 1
            || self.device_rows != self.shape.block_size
            || self.router_weight_scan_calls != 1
            || self.route_postprocess_dispatch_calls != 1
            || self.serial_token_forward_calls != 0
        {
            bail!("S14 causal-block router 回执不能证明一次 K-row shared-weight scan");
        }
        Ok(())
    }
}

impl S14CausalBlockAttentionRouterRecordingReceipt {
    pub fn validate(self) -> Result<()> {
        if self.recorder_calls != 1
            || self.device_rows != self.shape.block_size
            || self.attention_dispatch_calls != 1
            || self.router_weight_scan_calls != 1
            || self.route_postprocess_dispatch_calls != 1
            || self.serial_token_forward_calls != 0
            || self.hc_hidden_integration_complete
        {
            bail!("S14 causal-block attention/router 回执不能证明一次 K-row device recorder");
        }
        Ok(())
    }
}

/// 持有三个 pipeline 与全部 descriptor pool。绑定的 buffer 由上层 runtime 拥有，且必须
/// 活到录制命令完成。调用方必须在 device work drain 后调用 `destroy`。
pub struct S14CausalBlockAttentionRouterRecorder {
    shape: S14CausalBlockAttentionRouterShape,
    mode: S14RoutePostprocessGpuMode,
    attention_pipeline: ComputePipeline,
    route_pipeline: ComputePipeline,
    numeric_pipelines: S14NumericPipelines,
    attention_binder: DescriptorBinder,
    router_dispatch: S14Bf16MatvecDispatch,
    route_binder: DescriptorBinder,
}

impl S14CausalBlockAttentionRouterRecorder {
    pub fn bind(
        ctx: &VulkanContext,
        shape: S14CausalBlockAttentionRouterShape,
        mode: S14RoutePostprocessGpuMode,
        bindings: S14CausalBlockAttentionRouterBindings<'_>,
    ) -> Result<Self> {
        let shape = S14CausalBlockAttentionRouterShape::new(
            shape.block_size,
            shape.base_position,
            shape.committed_window_rows,
        )?;
        validate_writable_ranges(shape, mode, bindings)?;

        let attention_pipeline = ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_ATTENTION_SPV, 7, 20)?;
        let route_pipeline = match ComputePipeline::new(ctx, S14_CAUSAL_BLOCK_ROUTE_SPV, 5, 8) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                attention_pipeline.destroy(ctx);
                return Err(error.context("创建 S14 causal-block route pipeline"));
            }
        };
        let numeric_pipelines = match S14NumericPipelines::new(ctx) {
            Ok(pipelines) => pipelines,
            Err(error) => {
                route_pipeline.destroy(ctx);
                attention_pipeline.destroy(ctx);
                return Err(error.context("创建 S14 causal-block BF16 router pipeline"));
            }
        };

        let attention_binder = match DescriptorBinder::new_with_offsets(
            ctx,
            &attention_pipeline,
            &[
                slice_range(bindings.query_bf16, shape.query_bf16_bytes()?),
                slice_range(
                    bindings.committed_window_kv_bf16,
                    shape.committed_window_bf16_bytes()?,
                ),
                slice_range(
                    bindings.current_block_kv_bf16,
                    shape.current_block_kv_bf16_bytes()?,
                ),
                slice_range(bindings.sink_f32, shape.sink_f32_bytes()),
                slice_range(bindings.rope_f32, shape.rope_f32_bytes()?),
                slice_range(
                    bindings.attention_output_bf16,
                    shape.attention_output_bf16_bytes()?,
                ),
                slice_range(bindings.sticky_status_u32, STATUS_BYTES),
            ],
        ) {
            Ok(binder) => binder,
            Err(error) => {
                numeric_pipelines.destroy(ctx);
                route_pipeline.destroy(ctx);
                attention_pipeline.destroy(ctx);
                return Err(error.context("绑定 S14 causal-block attention descriptors"));
            }
        };

        let router_shape = S14Bf16MatvecShape::new(
            S14_CAUSAL_BLOCK_ROUTER_EXPERTS,
            S14_CAUSAL_BLOCK_ROUTER_HIDDEN,
            shape.block_size,
        )?;
        let router_dispatch = match numeric_pipelines.bind_bf16_matvec_arenas(
            ctx,
            router_shape,
            bindings.router_weight_bf16.buffer,
            bindings.router_weight_bf16.buffer.size(),
            bindings.router_weight_bf16.offset,
            bindings.router_input_f32.buffer,
            bindings.router_input_f32.buffer.size(),
            bindings.router_input_f32.offset,
            bindings.router_logits_f32.buffer,
            bindings.router_logits_f32.buffer.size(),
            bindings.router_logits_f32.offset,
        ) {
            Ok(dispatch) => dispatch,
            Err(error) => {
                attention_binder.destroy(ctx);
                numeric_pipelines.destroy(ctx);
                route_pipeline.destroy(ctx);
                attention_pipeline.destroy(ctx);
                return Err(error.context("绑定 S14 causal-block batched router matvec"));
            }
        };

        let route_binder = match DescriptorBinder::new_with_offsets(
            ctx,
            &route_pipeline,
            &[
                slice_range(bindings.router_logits_f32, shape.router_logits_f32_bytes()?),
                slice_range(bindings.route_aux, shape.route_aux_bytes(mode)?),
                slice_range(bindings.expert_ids_u32, shape.route_output_bytes()?),
                slice_range(bindings.route_weights_f32, shape.route_output_bytes()?),
                slice_range(bindings.sticky_status_u32, STATUS_BYTES),
            ],
        ) {
            Ok(binder) => binder,
            Err(error) => {
                router_dispatch.binder.destroy(ctx);
                attention_binder.destroy(ctx);
                numeric_pipelines.destroy(ctx);
                route_pipeline.destroy(ctx);
                attention_pipeline.destroy(ctx);
                return Err(error.context("绑定 S14 causal-block route descriptors"));
            }
        };

        Ok(Self {
            shape,
            mode,
            attention_pipeline,
            route_pipeline,
            numeric_pipelines,
            attention_binder,
            router_dispatch,
            route_binder,
        })
    }

    pub fn shape(&self) -> S14CausalBlockAttentionRouterShape {
        self.shape
    }

    /// # Safety
    ///
    /// `command` 必须处于 recording 状态；全部绑定资源必须活到 command 完成，sticky status
    /// 必须已清零。该窄入口只录制 attention；正式 HC/QKV adapter 必须在它之后录制
    /// wo_a/wo_b、attention HC-post 与 FFN HC-pre，再调用 `record_router`。
    pub unsafe fn record_attention(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<S14CausalBlockAttentionRecordingReceipt> {
        if command == vk::CommandBuffer::null() {
            bail!("S14 causal-block attention command 不能为空");
        }
        ctx.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.attention_pipeline.pipeline,
        );
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.attention_pipeline.layout,
            0,
            &[self.attention_binder.set],
            &[],
        );
        let mut attention_push = [0u8; 20];
        attention_push[..4].copy_from_slice(&S14_CAUSAL_BLOCK_ATTENTION_HEADS.to_le_bytes());
        attention_push[4..8].copy_from_slice(&S14_CAUSAL_BLOCK_ATTENTION_HEAD_DIM.to_le_bytes());
        attention_push[8..12].copy_from_slice(&self.shape.committed_window_rows.to_le_bytes());
        attention_push[12..16].copy_from_slice(&self.shape.base_position.to_le_bytes());
        attention_push[16..20].copy_from_slice(&self.shape.block_size.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.attention_pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &attention_push,
        );
        ctx.device.cmd_dispatch(
            command,
            S14_CAUSAL_BLOCK_ATTENTION_HEADS,
            self.shape.block_size,
            1,
        );

        compute_write_to_read_barrier(ctx, command);
        let receipt = S14CausalBlockAttentionRecordingReceipt {
            shape: self.shape,
            recorder_calls: 1,
            device_rows: self.shape.block_size,
            attention_dispatch_calls: 1,
            serial_token_forward_calls: 0,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    /// # Safety
    ///
    /// `command` 必须与产生 `router_input_f32` 的 HC/QKV command graph 相同且仍在 recording；
    /// 调用方必须先建立 compute write→read barrier。本方法在 router matvec 与 route shader
    /// 之间、以及 route shader 之后建立 barrier。
    pub unsafe fn record_router(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<S14CausalBlockRouterRecordingReceipt> {
        if command == vk::CommandBuffer::null() {
            bail!("S14 causal-block router command 不能为空");
        }
        self.numeric_pipelines
            .cmd_bf16_matvec(ctx, command, &self.router_dispatch);
        compute_write_to_read_barrier(ctx, command);

        ctx.device.cmd_bind_pipeline(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.route_pipeline.pipeline,
        );
        ctx.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            self.route_pipeline.layout,
            0,
            &[self.route_binder.set],
            &[],
        );
        let mut route_push = [0u8; 8];
        route_push[..4].copy_from_slice(&self.mode.as_raw().to_le_bytes());
        route_push[4..].copy_from_slice(&self.shape.block_size.to_le_bytes());
        ctx.device.cmd_push_constants(
            command,
            self.route_pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &route_push,
        );
        ctx.device
            .cmd_dispatch(command, self.shape.block_size, 1, 1);
        compute_write_to_read_barrier(ctx, command);

        let receipt = S14CausalBlockRouterRecordingReceipt {
            shape: self.shape,
            recorder_calls: 1,
            device_rows: self.shape.block_size,
            router_weight_scan_calls: 1,
            route_postprocess_dispatch_calls: 1,
            serial_token_forward_calls: 0,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    /// 零输入 numeric 门兼容入口。production hidden 路径必须使用拆分入口，在 attention 与
    /// router 之间插入真实 wo/post-attention/FFN-HC recorder。
    ///
    /// # Safety
    ///
    /// 与 `record_attention`、`record_router` 相同。
    pub unsafe fn record(
        &self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<S14CausalBlockAttentionRouterRecordingReceipt> {
        let attention = self.record_attention(ctx, command)?;
        let router = self.record_router(ctx, command)?;

        let receipt = S14CausalBlockAttentionRouterRecordingReceipt {
            shape: self.shape,
            recorder_calls: 1,
            device_rows: self.shape.block_size,
            attention_dispatch_calls: attention.attention_dispatch_calls,
            router_weight_scan_calls: router.router_weight_scan_calls,
            route_postprocess_dispatch_calls: router.route_postprocess_dispatch_calls,
            serial_token_forward_calls: 0,
            hc_hidden_integration_complete: false,
        };
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        self.route_binder.destroy(ctx);
        self.router_dispatch.binder.destroy(ctx);
        self.attention_binder.destroy(ctx);
        self.numeric_pipelines.destroy(ctx);
        self.route_pipeline.destroy(ctx);
        self.attention_pipeline.destroy(ctx);
    }
}

fn validate_writable_ranges(
    shape: S14CausalBlockAttentionRouterShape,
    mode: S14RoutePostprocessGpuMode,
    bindings: S14CausalBlockAttentionRouterBindings<'_>,
) -> Result<()> {
    let reads = [
        (bindings.query_bf16, shape.query_bf16_bytes()?, "query"),
        (
            bindings.committed_window_kv_bf16,
            shape.committed_window_bf16_bytes()?,
            "committed window",
        ),
        (
            bindings.current_block_kv_bf16,
            shape.current_block_kv_bf16_bytes()?,
            "current block KV",
        ),
        (bindings.sink_f32, shape.sink_f32_bytes(), "sink"),
        (bindings.rope_f32, shape.rope_f32_bytes()?, "RoPE"),
        (
            bindings.router_weight_bf16,
            shape.router_weight_bf16_bytes(),
            "router weight",
        ),
        (
            bindings.router_input_f32,
            shape.router_input_f32_bytes()?,
            "router input",
        ),
        (
            bindings.route_aux,
            shape.route_aux_bytes(mode)?,
            "route aux",
        ),
    ];
    let writes = [
        (
            bindings.attention_output_bf16,
            shape.attention_output_bf16_bytes()?,
            "attention output",
        ),
        (
            bindings.router_logits_f32,
            shape.router_logits_f32_bytes()?,
            "router logits",
        ),
        (
            bindings.expert_ids_u32,
            shape.route_output_bytes()?,
            "expert IDs",
        ),
        (
            bindings.route_weights_f32,
            shape.route_output_bytes()?,
            "route weights",
        ),
        (bindings.sticky_status_u32, STATUS_BYTES, "sticky status"),
    ];
    for (write_index, (write, write_bytes, write_name)) in writes.iter().copied().enumerate() {
        validate_slice(write, write_bytes, write_name)?;
        for (read, read_bytes, read_name) in reads.iter().copied() {
            validate_slice(read, read_bytes, read_name)?;
            if slices_overlap(write, write_bytes, read, read_bytes)? {
                bail!("S14 causal-block writable {write_name} 与 readonly {read_name} 重叠");
            }
        }
        for (other, other_bytes, other_name) in writes.iter().copied().skip(write_index + 1) {
            if slices_overlap(write, write_bytes, other, other_bytes)? {
                bail!("S14 causal-block writable {write_name} 与 {other_name} 重叠");
            }
        }
    }
    Ok(())
}

fn validate_slice(slice: StorageBufferSlice<'_>, bytes: u64, label: &str) -> Result<()> {
    let end = slice
        .offset
        .checked_add(bytes)
        .with_context(|| format!("S14 causal-block {label} range overflow"))?;
    if bytes == 0 || end > slice.buffer.size() {
        bail!(
            "S14 causal-block {label} slice 越界: offset={} bytes={} capacity={}",
            slice.offset,
            bytes,
            slice.buffer.size()
        );
    }
    Ok(())
}

fn slices_overlap(
    left: StorageBufferSlice<'_>,
    left_bytes: u64,
    right: StorageBufferSlice<'_>,
    right_bytes: u64,
) -> Result<bool> {
    let left_end = left
        .offset
        .checked_add(left_bytes)
        .context("left range overflow")?;
    let right_end = right
        .offset
        .checked_add(right_bytes)
        .context("right range overflow")?;
    Ok(left.buffer.handle() == right.buffer.handle()
        && left.offset < right_end
        && right.offset < left_end)
}

fn slice_range(slice: StorageBufferSlice<'_>, bytes: u64) -> (&GpuBuffer, u64, u64) {
    (slice.buffer, slice.offset, bytes)
}

fn checked_bytes(elements: u64, element_bytes: u64, label: &str) -> Result<u64> {
    elements
        .checked_mul(element_bytes)
        .with_context(|| format!("S14 {label} byte size overflow"))
}

unsafe fn compute_write_to_read_barrier(ctx: &VulkanContext, command: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    ctx.device.cmd_pipeline_barrier(
        command,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
    );
}

pub fn validate_causal_block_attention_router_status(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let unknown = code & !S14_CAUSAL_BLOCK_STATUS_KNOWN_MASK;
    if unknown != 0 {
        bail!("S14 causal-block attention/router 未知 status bits 0x{unknown:08x}");
    }
    bail!("S14 causal-block attention/router fail-closed, status=0x{code:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k4_k8_shapes_freeze_one_device_batch_byte_contract() {
        for block_size in [4, 8] {
            let shape = S14CausalBlockAttentionRouterShape::new(block_size, 11, 11).unwrap();
            assert_eq!(
                shape.query_bf16_bytes().unwrap(),
                u64::from(block_size) * 64 * 512 * 2
            );
            assert_eq!(shape.committed_window_bf16_bytes().unwrap(), 11 * 512 * 2);
            assert_eq!(
                shape.current_block_kv_bf16_bytes().unwrap(),
                u64::from(block_size) * 512 * 2
            );
            assert_eq!(
                shape.rope_f32_bytes().unwrap(),
                u64::from(block_size) * 64 * 4
            );
            assert_eq!(
                shape.router_input_f32_bytes().unwrap(),
                u64::from(block_size) * 4096 * 4
            );
            assert_eq!(
                shape.router_logits_f32_bytes().unwrap(),
                u64::from(block_size) * 256 * 4
            );
            assert_eq!(
                shape.route_output_bytes().unwrap(),
                u64::from(block_size) * 6 * 4
            );
            assert_eq!(
                shape
                    .route_aux_bytes(S14RoutePostprocessGpuMode::BiasTop6)
                    .unwrap(),
                256 * 4
            );
            assert_eq!(
                shape
                    .route_aux_bytes(S14RoutePostprocessGpuMode::PhysicalIds)
                    .unwrap(),
                u64::from(block_size) * 6 * 4
            );
            S14CausalBlockAttentionRouterRecordingReceipt {
                shape,
                recorder_calls: 1,
                device_rows: block_size,
                attention_dispatch_calls: 1,
                router_weight_scan_calls: 1,
                route_postprocess_dispatch_calls: 1,
                serial_token_forward_calls: 0,
                hc_hidden_integration_complete: false,
            }
            .validate()
            .unwrap();
        }
    }

    #[test]
    fn unsupported_k_position_or_fake_completion_fail_closed() {
        for block_size in [0, 1, 2, 3, 5, 6, 7, 9] {
            assert!(S14CausalBlockAttentionRouterShape::new(block_size, 1, 1).is_err());
        }
        assert!(S14CausalBlockAttentionRouterShape::new(4, 0, 0).is_err());
        assert!(S14CausalBlockAttentionRouterShape::new(4, 9, 8).is_err());
        assert!(S14CausalBlockAttentionRouterShape::new(8, 120, 120).is_err());

        let shape = S14CausalBlockAttentionRouterShape::new(4, 1, 1).unwrap();
        for invalid in [
            S14CausalBlockAttentionRouterRecordingReceipt {
                shape,
                recorder_calls: 4,
                device_rows: 4,
                attention_dispatch_calls: 1,
                router_weight_scan_calls: 1,
                route_postprocess_dispatch_calls: 1,
                serial_token_forward_calls: 0,
                hc_hidden_integration_complete: false,
            },
            S14CausalBlockAttentionRouterRecordingReceipt {
                shape,
                recorder_calls: 1,
                device_rows: 4,
                attention_dispatch_calls: 1,
                router_weight_scan_calls: 1,
                route_postprocess_dispatch_calls: 1,
                serial_token_forward_calls: 4,
                hc_hidden_integration_complete: false,
            },
            S14CausalBlockAttentionRouterRecordingReceipt {
                shape,
                recorder_calls: 1,
                device_rows: 4,
                attention_dispatch_calls: 1,
                router_weight_scan_calls: 1,
                route_postprocess_dispatch_calls: 1,
                serial_token_forward_calls: 0,
                hc_hidden_integration_complete: true,
            },
        ] {
            assert!(invalid.validate().is_err());
        }

        validate_causal_block_attention_router_status(0).unwrap();
        for code in [
            1,
            2,
            1 << 8,
            1 << 12,
            S14_CAUSAL_BLOCK_STATUS_KNOWN_MASK,
            1 << 31,
        ] {
            assert!(validate_causal_block_attention_router_status(code).is_err());
        }
    }
}
