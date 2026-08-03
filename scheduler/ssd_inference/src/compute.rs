//! Compute pipeline scaffolding: SPIR-V loading, descriptor set layouts,
//! pipeline creation. Wraps the verbose Vulkan ceremony into a small API
//! the engine can use to invoke shaders.

use crate::buffer::GpuBuffer;
use crate::device::VulkanContext;
use anyhow::{bail, Result};
use ash::vk;

/// Bytecode for the bundled vector_add shader (compiled at build time by glslc).
pub const VECTOR_ADD_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vector_add.spv"));

/// Naive fp32 matmul: C[M,N] = A[M,K] * B[K,N]. Workgroup 16x16, push (M,N,K).
pub const MATMUL_FP32_NAIVE_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/matmul_fp32_naive.spv"));

/// X @ W^T: Y[M,N] = X[M,K] * W^T where W is [N,K] (GGUF [out, in] layout).
pub const MATMUL_XWT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul_xwt.spv"));

/// Optimized matvec for M=1: y[n] = dot(x, W[n,:]). 128-thread parallel reduce.
pub const MATVEC_WT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec_wt.spv"));

/// Fused Q4K dequant + matvec on GPU. Reads Q4K bytes directly, no fp32 intermediate.
pub const FUSED_Q4K_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_q4k_matvec.spv"));

pub const FUSED_Q5K_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_q5k_matvec.spv"));

pub const FUSED_Q6K_MATVEC_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fused_q6k_matvec.spv"));

/// Q4_K_M block → fp32 dequant. Workgroup 256, one block per workgroup,
/// push (n_blocks). Output stride is 256 floats per block.
pub const DEQUANT_Q4_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q4_k.spv"));

/// Q6_K block → fp32 dequant. 210 bytes/256 weights, byte-aligned reads.
pub const DEQUANT_Q6_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q6_k.spv"));

pub const DEQUANT_Q2_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q2_k.spv"));

pub const DEQUANT_Q3_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dequant_q3_k.spv"));

/// RMSNorm: y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i].
/// Workgroup 256, one normalized vector per workgroup. Push (hidden, eps).
pub const RMSNORM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));

/// Numerically stable softmax. Workgroup 256, one vector per workgroup. Push (dim).
pub const SOFTMAX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax.spv"));

/// SwiGLU: y[i] = silu(gate[i]) * up[i]. Workgroup 256, one thread per element. Push (n).
pub const SWIGLU_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv"));

/// Token embedding lookup: y[i] = table[token_id * hidden + i].
pub const EMBEDDING_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embedding.spv"));

/// In-place RoPE rotation. Dispatch (n_heads, n_tokens, 1), wg size 64 (= head_dim/2).
pub const ROPE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));

/// Attention scaled dot product: scores = (Q @ K^T) / sqrt(d), with causal mask.
/// Dispatch (n_q_heads, n_tok, 1), wg size 256.
pub const SCALED_DOT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/scaled_dot.spv"));

/// Attention output mix: out = scores @ V. Dispatch (n_q_heads, n_tok, 1), wg size 128.
pub const ATTN_V_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_v.spv"));

/// Residual add: y[i] = a[i] + b[i]. Element-wise.
pub const RESIDUAL_ADD_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/residual_add.spv"));

/// Weighted accumulate: y[i] += w * b[i]. Used for MoE expert mixing.
pub const WEIGHTED_ADD_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/weighted_add.spv"));

/// Top-K selection: idx, val of top-K from logits[N]. K=8 N≤256 → < 0.1 ms.
pub const TOP_K_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/top_k.spv"));

/// Weighted sum of 8 expert outputs: h[i] += Σ(w[e] * d[e*D+i]).
pub const WEIGHTED_SUM_8_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/weighted_sum_8.spv"));

/// Generic compute pipeline for shaders with N storage buffer bindings + push constants.
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
    pub shader_module: vk::ShaderModule,
    n_bindings: u32,
    push_bytes: u32,
}

impl ComputePipeline {
    /// Create a compute pipeline from raw SPIR-V bytecode (4-byte aligned).
    /// `n_storage_buffers` = number of `buffer` bindings in the shader (0..N).
    /// `push_constant_bytes` = sizeof(push struct), 0 if no push constants.
    pub fn new(
        ctx: &VulkanContext,
        spv_bytes: &[u8],
        n_storage_buffers: u32,
        push_constant_bytes: u32,
    ) -> Result<Self> {
        unsafe {
            // SPIR-V is 4-byte words. ash takes &[u32], not &[u8].
            assert!(spv_bytes.len() % 4 == 0, "SPIR-V bytes not 4-aligned");
            let spv_words: Vec<u32> = spv_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let module = ctx.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&spv_words),
                None,
            )?;

            // Descriptor set layout: N storage buffer bindings, all visible to compute
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_storage_buffers)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;

            // Pipeline layout: 1 set + optional push constant range
            let mut layout_ci = vk::PipelineLayoutCreateInfo::default();
            let set_layouts = [set_layout];
            layout_ci = layout_ci.set_layouts(&set_layouts);
            let push_ranges;
            if push_constant_bytes > 0 {
                push_ranges = [vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .offset(0)
                    .size(push_constant_bytes)];
                layout_ci = layout_ci.push_constant_ranges(&push_ranges);
            }
            let pipeline_layout = ctx.device.create_pipeline_layout(&layout_ci, None)?;

            // Compute pipeline
            let entry_name = std::ffi::CString::new("main").unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry_name);
            let create_info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)];
            let pipeline = ctx
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &create_info, None)
                .map_err(|(_, e)| anyhow::anyhow!("create_compute_pipelines: {e}"))?[0];

            Ok(Self {
                pipeline,
                layout: pipeline_layout,
                set_layout,
                shader_module: module,
                n_bindings: n_storage_buffers,
                push_bytes: push_constant_bytes,
            })
        }
    }

    pub fn n_bindings(&self) -> u32 {
        self.n_bindings
    }
    pub fn push_bytes(&self) -> u32 {
        self.push_bytes
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device.destroy_pipeline_layout(self.layout, None);
            ctx.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            ctx.device.destroy_shader_module(self.shader_module, None);
        }
    }
}

/// Reusable descriptor pool — allocate many sets from one pool, reset all at once.
pub struct DescriptorArena {
    pool: vk::DescriptorPool,
    allocated: u32,
    capacity: u32,
}

impl DescriptorArena {
    pub fn new(ctx: &VulkanContext, capacity: u32, max_bindings_per_set: u32) -> Result<Self> {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(capacity * max_bindings_per_set)];
        let pool = unsafe {
            ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(capacity)
                    .pool_sizes(&pool_sizes)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                None,
            )?
        };
        Ok(Self {
            pool,
            allocated: 0,
            capacity,
        })
    }

    pub fn reset(&mut self, ctx: &VulkanContext) -> Result<()> {
        unsafe {
            ctx.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())?;
        }
        self.allocated = 0;
        Ok(())
    }

    pub fn alloc_set(
        &mut self,
        ctx: &VulkanContext,
        pipe: &ComputePipeline,
        buffers: &[(&GpuBuffer, u64, u64)],
    ) -> Result<vk::DescriptorSet> {
        if self.allocated >= self.capacity {
            bail!(
                "descriptor arena full ({}/{})",
                self.allocated,
                self.capacity
            );
        }
        validate_storage_buffer_ranges(ctx, pipe, buffers)?;
        unsafe {
            let set_layouts = [pipe.set_layout];
            let set = ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&set_layouts),
            )?[0];

            let infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|(b, off, sz)| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(b.handle())
                        .offset(*off)
                        .range(*sz)
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = (0..pipe.n_bindings as usize)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]))
                })
                .collect();
            ctx.device.update_descriptor_sets(&writes, &[]);
            self.allocated += 1;
            Ok(set)
        }
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

/// One-shot descriptor pool sized for a single dispatch's binding set.
pub struct DescriptorBinder {
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
}

/// 由其他 production owner 保活的 storage buffer 视图。它只供 descriptor 写入时使用，
/// 不拥有 Vulkan memory，也绝不能调用 `GpuBuffer::destroy`。调用者必须保证 handle 在
/// descriptor set/command 完成前有效。
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExternalStorageBuffer {
    pub buffer: vk::Buffer,
    pub capacity: u64,
}

/// 一个 storage buffer 内的逻辑子范围起点。具体字节长度由算子 shape 决定；这样
/// production graph 可以把全部临时张量放进单一 workspace，而不制造几十个 Vulkan
/// allocation。offset 仍会由 `DescriptorBinder::new_with_offsets` 按设备限制校验。
#[derive(Clone, Copy)]
pub struct StorageBufferSlice<'a> {
    pub buffer: &'a GpuBuffer,
    pub offset: u64,
}

impl<'a> StorageBufferSlice<'a> {
    pub fn whole(buffer: &'a GpuBuffer) -> Self {
        Self { buffer, offset: 0 }
    }
}

/// 同一底层 buffer 的两个可写/只读子范围是否相交。溢出视为合同错误，不能退化为
/// “不重叠”。算子用它替代仅比较 buffer handle 的旧 alias 检查。
pub fn storage_buffer_slices_overlap(
    left: StorageBufferSlice<'_>,
    left_bytes: u64,
    right: StorageBufferSlice<'_>,
    right_bytes: u64,
) -> Result<bool> {
    let left_end = left
        .offset
        .checked_add(left_bytes)
        .ok_or_else(|| anyhow::anyhow!("storage buffer left slice overflow"))?;
    let right_end = right
        .offset
        .checked_add(right_bytes)
        .ok_or_else(|| anyhow::anyhow!("storage buffer right slice overflow"))?;
    Ok(left.buffer.handle() == right.buffer.handle()
        && left.offset < right_end
        && right.offset < left_end)
}

impl DescriptorBinder {
    pub fn new(
        ctx: &VulkanContext,
        pipe: &ComputePipeline,
        buffers: &[(&GpuBuffer, u64)],
    ) -> Result<Self> {
        let triples: Vec<(&GpuBuffer, u64, u64)> =
            buffers.iter().map(|(b, sz)| (*b, 0u64, *sz)).collect();
        Self::new_with_offsets(ctx, pipe, &triples)
    }

    /// Like `new`, but each binding can target a sub-range `[offset, offset+size)`
    /// of the underlying buffer. Used for KV-cache layer slabs and packed expert
    /// tensors. `offset` must be a multiple of `minStorageBufferOffsetAlignment`.
    pub fn new_with_offsets(
        ctx: &VulkanContext,
        pipe: &ComputePipeline,
        buffers: &[(&GpuBuffer, u64, u64)],
    ) -> Result<Self> {
        validate_storage_buffer_ranges(ctx, pipe, buffers)?;
        unsafe {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(pipe.n_bindings)];
            let pool = ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?;

            let set_layouts = [pipe.set_layout];
            let set = ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            )?[0];

            let infos: Vec<vk::DescriptorBufferInfo> = buffers
                .iter()
                .map(|(b, off, sz)| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(b.handle())
                        .offset(*off)
                        .range(*sz)
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = (0..pipe.n_bindings as usize)
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[i]))
                })
                .collect();
            ctx.device.update_descriptor_sets(&writes, &[]);
            Ok(Self { pool, set })
        }
    }

    /// `new_with_offsets` 的 non-owning production 边界。只在资源 owner 已通过独立
    /// capacity/lifetime gate、但不能把 `GpuBuffer` 所有权移入 recorder 时使用。
    pub(crate) fn new_with_external_offsets(
        ctx: &VulkanContext,
        pipe: &ComputePipeline,
        buffers: &[(ExternalStorageBuffer, u64, u64)],
    ) -> Result<Self> {
        validate_external_storage_buffer_ranges(ctx, pipe, buffers)?;
        unsafe {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(pipe.n_bindings)];
            let pool = ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            let set_layouts = [pipe.set_layout];
            let set = match ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&set_layouts),
            ) {
                Ok(sets) => sets[0],
                Err(error) => {
                    ctx.device.destroy_descriptor_pool(pool, None);
                    return Err(error.into());
                }
            };
            let infos = buffers
                .iter()
                .map(|(binding, offset, size)| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(binding.buffer)
                        .offset(*offset)
                        .range(*size)
                })
                .collect::<Vec<_>>();
            let writes = (0..pipe.n_bindings as usize)
                .map(|index| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(index as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&infos[index]))
                })
                .collect::<Vec<_>>();
            ctx.device.update_descriptor_sets(&writes, &[]);
            Ok(Self { pool, set })
        }
    }

    pub fn destroy(&self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

fn validate_external_storage_buffer_ranges(
    ctx: &VulkanContext,
    pipe: &ComputePipeline,
    buffers: &[(ExternalStorageBuffer, u64, u64)],
) -> Result<()> {
    if buffers.len() != pipe.n_bindings as usize {
        bail!(
            "external descriptor binding count drift: expected={} actual={}",
            pipe.n_bindings,
            buffers.len()
        );
    }
    let limits = unsafe {
        ctx.instance
            .get_physical_device_properties(ctx.physical)
            .limits
    };
    let alignment = limits.min_storage_buffer_offset_alignment.max(1);
    let max_range = u64::from(limits.max_storage_buffer_range);
    for (index, (binding, offset, size)) in buffers.iter().enumerate() {
        if binding.buffer == vk::Buffer::null() || binding.capacity == 0 || *size == 0 {
            bail!("external descriptor binding {index} handle/capacity/range 不能为空");
        }
        if *offset % alignment != 0 {
            bail!(
                "external descriptor binding {index} offset 未对齐: offset={offset} alignment={alignment}"
            );
        }
        if *size > max_range {
            bail!(
                "external descriptor binding {index} range 超过设备上限: size={size} max={max_range}"
            );
        }
        let end = offset
            .checked_add(*size)
            .ok_or_else(|| anyhow::anyhow!("external descriptor binding {index} range overflow"))?;
        if end > binding.capacity {
            bail!(
                "external descriptor binding {index} 越界: end={end} capacity={}",
                binding.capacity
            );
        }
    }
    Ok(())
}

fn validate_storage_buffer_ranges(
    ctx: &VulkanContext,
    pipe: &ComputePipeline,
    buffers: &[(&GpuBuffer, u64, u64)],
) -> Result<()> {
    if buffers.len() != pipe.n_bindings as usize {
        bail!(
            "descriptor binding count drift: expected={} actual={}",
            pipe.n_bindings,
            buffers.len()
        );
    }
    let limits = unsafe {
        ctx.instance
            .get_physical_device_properties(ctx.physical)
            .limits
    };
    let alignment = limits.min_storage_buffer_offset_alignment.max(1);
    let max_range = u64::from(limits.max_storage_buffer_range);
    for (binding, (buffer, offset, size)) in buffers.iter().enumerate() {
        if *size == 0 {
            bail!("descriptor binding {binding} range 不能为空");
        }
        if *offset % alignment != 0 {
            bail!(
                "descriptor binding {binding} offset 未按设备要求对齐: offset={} alignment={alignment}",
                offset
            );
        }
        if *size > max_range {
            bail!(
                "descriptor binding {binding} range 超过设备上限: size={} max={max_range}",
                size
            );
        }
        let end = offset
            .checked_add(*size)
            .ok_or_else(|| anyhow::anyhow!("descriptor binding {binding} range overflow"))?;
        if end > buffer.size() {
            bail!(
                "descriptor binding {binding} 越出 buffer: offset={} size={} capacity={}",
                offset,
                size,
                buffer.size()
            );
        }
    }
    Ok(())
}
