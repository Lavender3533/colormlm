//! S14 StarFold W1/W3 与 W2 之间的 batched SwiGLU/route-weight command owner。

use crate::{
    compute::DescriptorBinder,
    s14_starfold_cache::{STARFOLD_B4_LANES, STARFOLD_TOP_K},
    s14_starfold_expert_schedule::{S14StarfoldB4ExpertSchedule, S14_STARFOLD_INTERMEDIATE},
    s14_starfold_mxfp4_tile::S14StarfoldMxfp4ExternalSlice,
    s14_starfold_routed_executor::{S14StarfoldRoutedBuffers, S14StarfoldRoutedWorkspaceLayout},
    s14_vulkan::S14NumericPipelines,
    GpuBuffer, VulkanContext,
};
use anyhow::{bail, Context, Result};
use ash::vk;
use std::sync::Arc;

const ROUTE_BRANCHES: u32 = (STARFOLD_B4_LANES * STARFOLD_TOP_K) as u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14StarfoldPrepareReceipt {
    pub layer: u16,
    pub base_position: u64,
    pub exact_route_weights: u32,
    pub update_buffer_calls: u32,
    pub prepare_dispatch_calls: u32,
    pub queue_submit_calls: u32,
    pub serial_token_forward_calls: u32,
}

pub struct S14StarfoldPrepareOwner {
    context: Arc<VulkanContext>,
    numeric: Option<S14NumericPipelines>,
    workspace: Option<GpuBuffer>,
    layout: S14StarfoldRoutedWorkspaceLayout,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    binder: Option<DescriptorBinder>,
    in_flight: bool,
    destroyed: bool,
}

impl S14StarfoldPrepareOwner {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        let layout = S14StarfoldRoutedWorkspaceLayout::build()?;
        let workspace = GpuBuffer::new_vram(
            &context,
            layout.bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
        )
        .context("分配 S14 StarFold routed workspace")?;
        let numeric = match S14NumericPipelines::new(&context) {
            Ok(numeric) => numeric,
            Err(error) => {
                workspace.destroy(&context);
                return Err(error).context("创建 S14 StarFold prepare numeric pipelines");
            }
        };
        let command_pool = match unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.qf_graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .context("创建 S14 StarFold prepare command pool")
        {
            Ok(pool) => pool,
            Err(error) => {
                numeric.destroy(&context);
                workspace.destroy(&context);
                return Err(error);
            }
        };
        let command_buffer = match unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .context("分配 S14 StarFold prepare command buffer")
        {
            Ok(buffers) if buffers.len() == 1 => buffers[0],
            Ok(_) => {
                unsafe { context.device.destroy_command_pool(command_pool, None) };
                numeric.destroy(&context);
                workspace.destroy(&context);
                bail!("S14 StarFold prepare command buffer 数量漂移");
            }
            Err(error) => {
                unsafe { context.device.destroy_command_pool(command_pool, None) };
                numeric.destroy(&context);
                workspace.destroy(&context);
                return Err(error);
            }
        };
        let fence = match unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .context("创建 S14 StarFold prepare fence")
        {
            Ok(fence) => fence,
            Err(error) => {
                unsafe { context.device.destroy_command_pool(command_pool, None) };
                numeric.destroy(&context);
                workspace.destroy(&context);
                return Err(error);
            }
        };
        Ok(Self {
            context,
            numeric: Some(numeric),
            workspace: Some(workspace),
            layout,
            command_pool,
            command_buffer,
            fence,
            binder: None,
            in_flight: false,
            destroyed: false,
        })
    }

    pub const fn layout(&self) -> S14StarfoldRoutedWorkspaceLayout {
        self.layout
    }

    pub fn routed_buffers(
        &self,
        input_f32: S14StarfoldMxfp4ExternalSlice,
    ) -> Result<S14StarfoldRoutedBuffers> {
        let workspace = self
            .workspace
            .as_ref()
            .context("S14 StarFold routed workspace 已销毁")?;
        Ok(S14StarfoldRoutedBuffers {
            input_f32,
            workspace: S14StarfoldMxfp4ExternalSlice {
                buffer: workspace.handle(),
                capacity_bytes: workspace.size(),
                offset: 0,
                logical_bytes: self.layout.bytes,
            },
        })
    }

    /// W1/W3 tile commands 已先提交到同一 graphics queue；本 command 在队列序上等待它们，
    /// 一次写入 24 个权威 route weight 并完成整批 SwiGLU。随后提交的 W2 tile 自动位于其后。
    pub fn submit_prepare(
        &mut self,
        schedule: &S14StarfoldB4ExpertSchedule,
    ) -> Result<S14StarfoldPrepareReceipt> {
        if self.destroyed {
            bail!("S14 StarFold prepare owner 已销毁");
        }
        self.wait_previous()?;
        let weights = exact_route_weights(schedule)?;
        let weight_bytes = weights
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let workspace = self
            .workspace
            .as_ref()
            .context("S14 StarFold routed workspace 已销毁")?;
        let dispatch = self
            .numeric
            .as_ref()
            .context("S14 StarFold numeric pipelines 已销毁")?
            .bind_batched_official_expert_prepare_arena(
                &self.context,
                ROUTE_BRANCHES,
                S14_STARFOLD_INTERMEDIATE,
                workspace,
                self.layout.bytes,
                self.layout.gate,
                self.layout.up,
                self.layout.route_weights,
                self.layout.hidden,
            )?;

        unsafe {
            self.context
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .context("reset S14 StarFold prepare command buffer")?;
        unsafe {
            self.context.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .context("begin S14 StarFold prepare command buffer")?;
        unsafe {
            self.context.device.cmd_update_buffer(
                self.command_buffer,
                workspace.handle(),
                self.layout.route_weights,
                &weight_bytes,
            );
            transfer_to_compute_barrier(
                &self.context,
                self.command_buffer,
                workspace.handle(),
                self.layout.route_weights,
                weight_bytes.len() as u64,
            );
            self.numeric
                .as_ref()
                .expect("numeric checked")
                .cmd_batched_official_expert_prepare(&self.context, self.command_buffer, &dispatch);
            compute_write_to_read_barrier(
                &self.context,
                self.command_buffer,
                workspace.handle(),
                self.layout.hidden,
                ROUTE_BRANCHES as u64 * u64::from(S14_STARFOLD_INTERMEDIATE) * 4,
            );
        }
        if let Err(error) = unsafe { self.context.device.end_command_buffer(self.command_buffer) }
            .context("end S14 StarFold prepare command buffer")
        {
            dispatch.binder.destroy(&self.context);
            return Err(error);
        }
        if let Err(error) = unsafe {
            self.context.device.queue_submit(
                self.context.q_graphics,
                &[vk::SubmitInfo::default()
                    .command_buffers(std::slice::from_ref(&self.command_buffer))],
                self.fence,
            )
        }
        .context("提交 S14 StarFold prepare command")
        {
            dispatch.binder.destroy(&self.context);
            return Err(error);
        }
        self.binder = Some(dispatch.binder);
        self.in_flight = true;
        Ok(S14StarfoldPrepareReceipt {
            layer: schedule.layer,
            base_position: schedule.base_position,
            exact_route_weights: ROUTE_BRANCHES,
            update_buffer_calls: 1,
            prepare_dispatch_calls: 1,
            queue_submit_calls: 1,
            serial_token_forward_calls: 0,
        })
    }

    pub fn try_destroy(&mut self) -> Result<()> {
        if self.destroyed {
            return Ok(());
        }
        if self.in_flight {
            let complete = unsafe { self.context.device.get_fence_status(self.fence) }
                .context("查询 S14 StarFold prepare fence")?;
            if !complete {
                bail!("S14 StarFold prepare command 仍在执行");
            }
            self.finish_previous()?;
        }
        self.destroy_resources();
        Ok(())
    }

    fn wait_previous(&mut self) -> Result<()> {
        if !self.in_flight {
            return Ok(());
        }
        unsafe {
            self.context
                .device
                .wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)
        }
        .context("等待 S14 StarFold prepare fence")?;
        self.finish_previous()
    }

    fn finish_previous(&mut self) -> Result<()> {
        unsafe {
            self.context
                .device
                .reset_fences(std::slice::from_ref(&self.fence))
        }
        .context("reset S14 StarFold prepare fence")?;
        self.binder
            .take()
            .context("S14 StarFold prepare 缺 descriptor lifetime owner")?
            .destroy(&self.context);
        self.in_flight = false;
        Ok(())
    }

    fn destroy_resources(&mut self) {
        if self.destroyed {
            return;
        }
        if let Some(binder) = self.binder.take() {
            binder.destroy(&self.context);
        }
        unsafe {
            self.context.device.destroy_fence(self.fence, None);
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
        if let Some(numeric) = self.numeric.take() {
            numeric.destroy(&self.context);
        }
        if let Some(workspace) = self.workspace.take() {
            workspace.destroy(&self.context);
        }
        self.fence = vk::Fence::null();
        self.command_pool = vk::CommandPool::null();
        self.destroyed = true;
    }
}

impl Drop for S14StarfoldPrepareOwner {
    fn drop(&mut self) {
        if self.destroyed {
            return;
        }
        let _ = unsafe { self.context.device.device_wait_idle() };
        self.destroy_resources();
    }
}

fn exact_route_weights(schedule: &S14StarfoldB4ExpertSchedule) -> Result<[u32; 24]> {
    let mut weights = [0u32; 24];
    let mut filled = [false; 24];
    for expert in &schedule.experts {
        for lane_use in &expert.lane_uses {
            let index =
                usize::from(lane_use.lane) * STARFOLD_TOP_K + usize::from(lane_use.route_rank);
            if index >= weights.len() || filled[index] {
                bail!("S14 StarFold route weight lane/rank 重复或越界");
            }
            weights[index] = lane_use.route_weight.to_bits();
            filled[index] = true;
        }
    }
    if filled.iter().any(|filled| !filled) {
        bail!("S14 StarFold route weights 未无损覆盖 B4×top-6");
    }
    Ok(weights)
}

unsafe fn transfer_to_compute_barrier(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    bytes: u64,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(bytes);
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&barrier),
            &[],
        );
    }
}

unsafe fn compute_write_to_read_barrier(
    context: &VulkanContext,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    bytes: u64,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(bytes);
    unsafe {
        context.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            std::slice::from_ref(&barrier),
            &[],
        );
    }
}
