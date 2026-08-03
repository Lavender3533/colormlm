//! FullDepth43 layer streamer 的双时间线同步原语。
//!
//! 权重传输和层计算使用两条独立 timeline semaphore。这样 L(n+1) 的权重上传
//! 可以与 L(n) 的计算重叠；只有复用同一个 ping-pong 页时，transfer 才等待该页
//! 上一次 compute 完成。热路径不做逐层 host fence，整 token 最后只等待一次。

use crate::VulkanContext;
use anyhow::{bail, Result};
use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14LayerTicket {
    pub transfer_value: u64,
    pub compute_value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct S14TimelineDrainReceipt {
    pub transfer_value: u64,
    pub compute_value: u64,
    /// 一次 `vkWaitSemaphores` 同时等待所有已提交的 timeline。
    pub host_wait_calls: u32,
}

pub struct S14DualQueueTimeline {
    transfer: vk::Semaphore,
    compute: vk::Semaphore,
    last_transfer: u64,
    last_compute: u64,
}

impl S14DualQueueTimeline {
    pub fn new(ctx: &VulkanContext) -> Result<Self> {
        if !ctx.timeline_semaphore {
            bail!("FullDepth43 双队列流水要求 Vulkan timeline semaphore");
        }
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let transfer = unsafe { ctx.device.create_semaphore(&create_info, None)? };
        let compute = match unsafe { ctx.device.create_semaphore(&create_info, None) } {
            Ok(semaphore) => semaphore,
            Err(error) => {
                unsafe { ctx.device.destroy_semaphore(transfer, None) };
                return Err(error.into());
            }
        };
        Ok(Self {
            transfer,
            compute,
            last_transfer: 0,
            last_compute: 0,
        })
    }

    pub fn last_transfer_value(&self) -> u64 {
        self.last_transfer
    }

    pub fn last_compute_value(&self) -> u64 {
        self.last_compute
    }

    /// 提交一层权重上传。`reuse_after_compute` 仅在复用同一个设备页时传入；
    /// 新页或另一个 ping-pong 页传 `None`，从而允许真正的 transfer/compute 重叠。
    ///
    /// # Safety
    ///
    /// `command` 必须已经结束录制，并且其资源至少存活到对应 timeline 值完成。
    pub unsafe fn submit_transfer(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        reuse_after_compute: Option<u64>,
    ) -> Result<u64> {
        if let Some(value) = reuse_after_compute {
            if value == 0 || value > self.last_compute {
                bail!(
                    "transfer 页复用等待值非法: value={value}, submitted_compute={}",
                    self.last_compute
                );
            }
        }
        let signal_value = self
            .last_transfer
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("transfer timeline 溢出"))?;
        let command_buffers = [command];
        let signal_semaphores = [self.transfer];
        let signal_values = [signal_value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let mut submit = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let wait_semaphores;
        let wait_stages;
        let wait_values;
        if let Some(value) = reuse_after_compute {
            wait_semaphores = [self.compute];
            wait_stages = [vk::PipelineStageFlags::TRANSFER];
            wait_values = [value];
            timeline = timeline.wait_semaphore_values(&wait_values);
            submit = submit
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages);
        }
        submit = submit.push_next(&mut timeline);
        ctx.device
            .queue_submit(ctx.q_transfer, &[submit], vk::Fence::null())?;
        self.last_transfer = signal_value;
        Ok(signal_value)
    }

    /// 提交一层计算，严格等待该层已经上传完成。函数只入队，不等待主机。
    ///
    /// # Safety
    ///
    /// `command` 必须已经结束录制，并且其资源至少存活到返回值完成。
    pub unsafe fn submit_compute(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
        wait_transfer: u64,
    ) -> Result<u64> {
        if wait_transfer == 0 || wait_transfer > self.last_transfer {
            bail!(
                "compute 上传等待值非法: value={wait_transfer}, submitted_transfer={}",
                self.last_transfer
            );
        }
        let signal_value = self
            .last_compute
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("compute timeline 溢出"))?;
        let command_buffers = [command];
        let wait_semaphores = [self.transfer];
        let wait_stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let signal_semaphores = [self.compute];
        let wait_values = [wait_transfer];
        let signal_values = [signal_value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        ctx.device
            .queue_submit(ctx.q_graphics, &[submit], vk::Fence::null())?;
        self.last_compute = signal_value;
        Ok(signal_value)
    }

    /// 提交不依赖新权重 transfer 的 compute 段，例如 embedding、final HC 或只消费
    /// 常驻权重的尾部归约。graphics queue 自身保证它排在此前 compute submit 之后；
    /// 本函数仍递增同一 compute timeline，使 final head 可以成为整 token 的最终票据。
    ///
    /// # Safety
    ///
    /// `command` 必须已经结束录制，并且其资源至少存活到返回的 timeline 值完成。
    pub unsafe fn submit_compute_only(
        &mut self,
        ctx: &VulkanContext,
        command: vk::CommandBuffer,
    ) -> Result<u64> {
        let signal_value = self
            .last_compute
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("compute timeline 溢出"))?;
        let command_buffers = [command];
        let signal_semaphores = [self.compute];
        let signal_values = [signal_value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        ctx.device
            .queue_submit(ctx.q_graphics, &[submit], vk::Fence::null())?;
        self.last_compute = signal_value;
        Ok(signal_value)
    }

    /// 整 token 最终唯一主机等待点。
    pub fn wait_compute(&self, ctx: &VulkanContext, value: u64, timeout_ns: u64) -> Result<()> {
        if value == 0 || value > self.last_compute {
            bail!(
                "host compute 等待值非法: value={value}, submitted_compute={}",
                self.last_compute
            );
        }
        let semaphores = [self.compute];
        let values = [value];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { ctx.device.wait_semaphores(&wait, timeout_ns)? };
        Ok(())
    }

    /// staging 页由主机重写前，只需等待对应 transfer 已经把旧字节复制进 VRAM；
    /// 不需要等待使用该 VRAM bank 的 compute。生产实现应把这些短等待留在权重生产线程，
    /// token 主线程仍只在整图末尾调用一次 `wait_compute`。
    pub fn wait_transfer(&self, ctx: &VulkanContext, value: u64, timeout_ns: u64) -> Result<()> {
        if value == 0 || value > self.last_transfer {
            bail!(
                "host transfer 等待值非法: value={value}, submitted_transfer={}",
                self.last_transfer
            );
        }
        let semaphores = [self.transfer];
        let values = [value];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { ctx.device.wait_semaphores(&wait, timeout_ns)? };
        Ok(())
    }

    /// 错误路径的一次性联合 drain。它不会签发 token 完成回执，只保证所有已经
    /// 入队的 transfer/compute 都不再引用 candidate、staging 或滚动权重页。
    /// 即使最后一次失败发生在 transfer submit 之后、compute submit 之前，也只需
    /// 一次 `vkWaitSemaphores`，不会漏掉孤立的 transfer。
    pub fn drain_all(
        &self,
        ctx: &VulkanContext,
        timeout_ns: u64,
    ) -> Result<S14TimelineDrainReceipt> {
        let mut semaphores = Vec::with_capacity(2);
        let mut values = Vec::with_capacity(2);
        if self.last_transfer != 0 {
            semaphores.push(self.transfer);
            values.push(self.last_transfer);
        }
        if self.last_compute != 0 {
            semaphores.push(self.compute);
            values.push(self.last_compute);
        }
        if semaphores.is_empty() {
            return Ok(S14TimelineDrainReceipt {
                transfer_value: 0,
                compute_value: 0,
                host_wait_calls: 0,
            });
        }
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { ctx.device.wait_semaphores(&wait, timeout_ns)? };
        Ok(S14TimelineDrainReceipt {
            transfer_value: self.last_transfer,
            compute_value: self.last_compute,
            host_wait_calls: 1,
        })
    }

    pub fn completed_values(&self, ctx: &VulkanContext) -> Result<(u64, u64)> {
        let transfer = unsafe { ctx.device.get_semaphore_counter_value(self.transfer)? };
        let compute = unsafe { ctx.device.get_semaphore_counter_value(self.compute)? };
        Ok((transfer, compute))
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        unsafe {
            ctx.device.destroy_semaphore(self.compute, None);
            ctx.device.destroy_semaphore(self.transfer, None);
        }
    }
}
