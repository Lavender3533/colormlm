//! Production K=4/8 causal-block Vulkan backend 的 fail-closed 骨架。
//!
//! 这里固定 block 生命周期、MoE production adapter 与 device future lease 所有权。
//! K-lane attention/router、真实 grouped shader、batched terminal/head 或 checkpoint recorder
//! 任一未注入时，相关数值入口明确报错，绝不退化成 K 次 K=1 forward 或伪造回执。

use crate::s14_causal_block_hc_qkv_adapter::S14CausalBlockVulkanHcQkvAdapter;
use crate::s14_causal_block_layer::{
    S14CausalBlockAbortReceipt, S14CausalBlockAttentionRouterOutput, S14CausalBlockBeginReceipt,
    S14CausalBlockCheckpointBackend, S14CausalBlockDeviceFutureOwner,
    S14CausalBlockDeviceFutureReceipt, S14CausalBlockFinalOutput, S14CausalBlockFullDepthBackend,
    S14CausalBlockGroupedMoeOutput, S14CausalBlockHiddenBinding, S14CausalBlockLayerBackend,
    S14CausalBlockLayerInput, S14CausalBlockLayerRangePlan, S14CausalBlockSealReceipt,
    S14CausalBlockUnionBankBinding, S14CausalBlockUnionMaterializeReceipt,
};
use crate::s14_causal_block_moe_adapter::S14CausalBlockVulkanMoeAdapter;
use ash::vk;
use polaris_s14_runner::{LayerCausalBatchPlan, RouteDecision, FULL_DEPTH_LAYERS};
use std::{fmt, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum S14CausalBlockVulkanPhase {
    Idle,
    Recording {
        base_position: u32,
        block_size: usize,
        bank_index: usize,
    },
    LayersSealed {
        base_position: u32,
        block_size: usize,
        bank_index: usize,
    },
}

/// 真实 Vulkan kernels/recorders 的 production backend 边界。已注入的 MoE adapter 可执行
/// proof/upload/grouped submit；attention、grouped shader 或 terminal 任一缺失仍 fail-closed。
#[derive(Debug)]
pub struct S14CausalBlockVulkanBackend {
    phase: S14CausalBlockVulkanPhase,
    hc_qkv: Option<Box<dyn S14CausalBlockVulkanHcQkvAdapter>>,
    moe: Option<Box<dyn S14CausalBlockVulkanMoeAdapter>>,
    terminal: Option<Box<dyn S14CausalBlockVulkanTerminalRecorder>>,
    last_exported_unvalidated: bool,
    begin_calls: u32,
    seal_calls: u32,
    abort_calls: u32,
}

impl Default for S14CausalBlockVulkanBackend {
    fn default() -> Self {
        Self {
            phase: S14CausalBlockVulkanPhase::Idle,
            hc_qkv: None,
            moe: None,
            terminal: None,
            last_exported_unvalidated: false,
            begin_calls: 0,
            seal_calls: 0,
            abort_calls: 0,
        }
    }
}

impl S14CausalBlockVulkanBackend {
    pub fn new_fail_closed() -> Self {
        Self::default()
    }

    pub fn with_terminal_recorder(
        terminal: impl S14CausalBlockVulkanTerminalRecorder + 'static,
    ) -> Self {
        Self {
            terminal: Some(Box::new(terminal)),
            ..Self::default()
        }
    }

    pub fn with_moe_adapter(adapter: impl S14CausalBlockVulkanMoeAdapter + 'static) -> Self {
        Self {
            moe: Some(Box::new(adapter)),
            ..Self::default()
        }
    }

    pub fn with_hc_qkv_adapter(adapter: impl S14CausalBlockVulkanHcQkvAdapter + 'static) -> Self {
        Self {
            hc_qkv: Some(Box::new(adapter)),
            ..Self::default()
        }
    }

    /// 允许先构造已带 MoE/terminal 的 backend，再独立安装 HC/QKV adapter；只在完全 idle
    /// 且尚未安装时接受，避免替换仍拥有 command/timeline 的 production owner。
    pub fn install_hc_qkv_adapter(
        &mut self,
        adapter: impl S14CausalBlockVulkanHcQkvAdapter + 'static,
    ) -> Result<(), String> {
        if self.phase != S14CausalBlockVulkanPhase::Idle
            || self.last_exported_unvalidated
            || self.hc_qkv.is_some()
        {
            return Err("K-lane Vulkan HC/QKV adapter 只能安装到完全 idle 的空槽".into());
        }
        self.hc_qkv = Some(Box::new(adapter));
        Ok(())
    }

    /// 允许在 MoE/HC-QKV owner 均已构造后安装独立 production terminal adapter。
    /// 只在完全 idle 且 terminal 空槽时接受，避免替换持有 command/arena lease 的 owner。
    pub fn install_terminal_recorder(
        &mut self,
        terminal: impl S14CausalBlockVulkanTerminalRecorder + 'static,
    ) -> Result<(), String> {
        if self.phase != S14CausalBlockVulkanPhase::Idle
            || self.last_exported_unvalidated
            || self.terminal.is_some()
        {
            return Err("K-lane Vulkan terminal recorder 只能安装到完全 idle 的空槽".into());
        }
        self.terminal = Some(Box::new(terminal));
        Ok(())
    }

    /// 供独立 attention/HC adapter 在取得真实 K-row route 后使用。只保存强身份，
    /// 不读取或伪造 device hidden；缺少本调用时 union materialize 必须 fail-closed。
    pub fn capture_attention_router_output_for_moe(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
        output: &S14CausalBlockAttentionRouterOutput,
    ) -> Result<(), String> {
        let (base_position, block_size, _) = self.recording_identity()?;
        if input.base_position != base_position
            || input.input_token_ids.len() != block_size
            || output.routes.len() != block_size
        {
            return Err("K-lane Vulkan MoE route capture 与 active block 漂移".into());
        }
        self.moe
            .as_mut()
            .ok_or_else(|| "production K-lane Vulkan MoE adapter 尚未注入".to_owned())?
            .capture_attention_router_output(input, output)
    }

    /// 只能在 backend 完全 idle 且没有待验收 future 时显式销毁 MoE graph owner。
    pub fn destroy_moe_adapter(&mut self) -> Result<(), String> {
        if self.phase != S14CausalBlockVulkanPhase::Idle || self.last_exported_unvalidated {
            return Err("K-lane Vulkan MoE adapter 只能在完全 idle 时销毁".into());
        }
        if let Some(adapter) = self.moe.as_mut() {
            adapter.destroy()?;
        }
        self.moe = None;
        Ok(())
    }

    /// 只能在 backend 完全 idle 且没有待验收 future 时显式销毁 HC/QKV command owner。
    pub fn destroy_hc_qkv_adapter(&mut self) -> Result<(), String> {
        if self.phase != S14CausalBlockVulkanPhase::Idle || self.last_exported_unvalidated {
            return Err("K-lane Vulkan HC/QKV adapter 只能在完全 idle 时销毁".into());
        }
        if let Some(adapter) = self.hc_qkv.as_mut() {
            adapter.destroy()?;
        }
        self.hc_qkv = None;
        Ok(())
    }

    pub fn is_idle(&self) -> bool {
        self.phase == S14CausalBlockVulkanPhase::Idle
    }

    /// Orchestrator 已把 terminal 导出的 owned future 成功封入 sealed future。
    /// 只有收到这份显式回执，backend 才允许开始下一块；失败路径仍走 abort。
    pub fn acknowledge_export_validated(&mut self) -> Result<(), String> {
        if self.phase != S14CausalBlockVulkanPhase::Idle || !self.last_exported_unvalidated {
            return Err("没有等待验收的 K-lane Vulkan future export".into());
        }
        if let Some(adapter) = self.moe.as_mut() {
            adapter.finish_validated_block()?;
        }
        if let Some(adapter) = self.hc_qkv.as_mut() {
            adapter.finish_validated_block()?;
        }
        self.last_exported_unvalidated = false;
        Ok(())
    }

    fn recording_identity(&self) -> Result<(u32, usize, usize), String> {
        match self.phase {
            S14CausalBlockVulkanPhase::Recording {
                base_position,
                block_size,
                bank_index,
            } => Ok((base_position, block_size, bank_index)),
            _ => Err("K-lane Vulkan backend 当前不在 recording 阶段".into()),
        }
    }
}

/// 真实 backend 的 terminal 注入点。实现必须在一个 K-row command graph 中录制
/// final HC、32-chunk batched lm-head、K-prefix checkpoint 导出和 timeline signal。
pub trait S14CausalBlockVulkanTerminalRecorder: fmt::Debug {
    fn record_batched_terminal_head_and_checkpoints(
        &mut self,
        completed_layers: usize,
        base_position: u32,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockFinalOutput, String>;

    /// block 在 terminal 前失败时清掉未消费 candidate source。已经提交的 terminal GPU
    /// 工作必须由具体 recorder 在返回错误前 drain；默认实现兼容无 pending source 的 owner。
    fn drain_and_abort_batched_terminal(&mut self, _completed_layers: usize) -> Result<(), String> {
        Ok(())
    }
}

impl S14CausalBlockLayerBackend for S14CausalBlockVulkanBackend {
    fn run_k_lane_attention_router(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockAttentionRouterOutput, String> {
        let (base_position, block_size, _) = self.recording_identity()?;
        if input.base_position != base_position
            || input.input_token_ids.len() != block_size
            || input.input_hidden.block_size != block_size
        {
            return Err("K-lane Vulkan attention 输入与 active block 漂移".into());
        }
        let recorded = self
            .hc_qkv
            .as_mut()
            .ok_or_else(|| "production K-lane Vulkan HC/QKV adapter 尚未注入".to_owned())?
            .run_k_lane_hc_qkv_attention_router(input)?;
        recorded.receipt.validate(input, &recorded.output)?;
        self.moe
            .as_mut()
            .ok_or_else(|| "production K-lane Vulkan MoE adapter 尚未注入".to_owned())?
            .capture_attention_router_output(input, &recorded.output)?;
        Ok(recorded.output)
    }

    fn materialize_union_ranges(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockUnionMaterializeReceipt, String> {
        let (_, block_size, bank_index) = self.recording_identity()?;
        if bank.bank_index != bank_index || range_plan.block_size != block_size {
            return Err("K-lane Vulkan union materialize 与 active block/bank 漂移".into());
        }
        self.moe
            .as_mut()
            .ok_or_else(|| "production K-lane Vulkan MoE adapter 尚未注入".to_owned())?
            .materialize_union_ranges(bank, range_plan)
    }

    fn run_grouped_moe(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        routes: &[RouteDecision],
        batch_plan: &LayerCausalBatchPlan,
        range_plan: &S14CausalBlockLayerRangePlan,
    ) -> Result<S14CausalBlockGroupedMoeOutput, String> {
        let (_, block_size, bank_index) = self.recording_identity()?;
        if bank.bank_index != bank_index
            || post_attention_hidden.block_size != block_size
            || routes.len() != block_size
            || batch_plan.block_size != block_size
            || range_plan.block_size != block_size
        {
            return Err("K-lane Vulkan grouped MoE 与 active block/bank 漂移".into());
        }
        let output = self
            .moe
            .as_mut()
            .ok_or_else(|| "production K-lane Vulkan MoE adapter 尚未注入".to_owned())?
            .run_grouped_moe(bank, post_attention_hidden, routes, batch_plan, range_plan)?;
        self.hc_qkv
            .as_mut()
            .ok_or_else(|| "production K-lane Vulkan HC/QKV adapter 尚未注入".to_owned())?
            .capture_grouped_moe_output(post_attention_hidden, &output)?;
        Ok(output)
    }
}

impl S14CausalBlockFullDepthBackend for S14CausalBlockVulkanBackend {
    fn begin_full_depth_block(
        &mut self,
        bank: S14CausalBlockUnionBankBinding,
        base_position: u32,
        block_size: usize,
    ) -> Result<S14CausalBlockBeginReceipt, String> {
        if self.phase != S14CausalBlockVulkanPhase::Idle {
            return Err("已有 K-lane Vulkan block 未释放".into());
        }
        if self.last_exported_unvalidated {
            return Err("上一 K-lane Vulkan future 尚未通过 orchestrator 验收/释放".into());
        }
        if !matches!(block_size, 4 | 8) || bank.buffer == vk::Buffer::null() || bank.bank_index >= 2
        {
            return Err("K-lane Vulkan begin binding/K 非法".into());
        }
        base_position
            .checked_add(block_size as u32)
            .ok_or_else(|| "K-lane Vulkan block position overflow".to_owned())?;
        self.begin_calls = self
            .begin_calls
            .checked_add(1)
            .ok_or_else(|| "K-lane Vulkan begin counter overflow".to_owned())?;
        if let Some(adapter) = self.hc_qkv.as_mut() {
            adapter.begin_block(base_position, block_size)?;
        }
        if let Some(adapter) = self.moe.as_mut() {
            if let Err(error) = adapter.begin_block(bank, base_position, block_size) {
                if let Some(hc_qkv) = self.hc_qkv.as_mut() {
                    let _ = hc_qkv.drain_and_abort(0);
                }
                return Err(error);
            }
        }
        self.phase = S14CausalBlockVulkanPhase::Recording {
            base_position,
            block_size,
            bank_index: bank.bank_index,
        };
        Ok(S14CausalBlockBeginReceipt {
            begin_calls: 1,
            base_position,
            block_size,
            bank_index: bank.bank_index,
            active: true,
            serial_token_forward_calls: 0,
        })
    }

    fn seal_full_depth_layers(
        &mut self,
        completed_layers: usize,
    ) -> Result<S14CausalBlockSealReceipt, String> {
        let (base_position, block_size, bank_index) = self.recording_identity()?;
        if completed_layers != FULL_DEPTH_LAYERS.len() {
            return Err("K-lane Vulkan backend 禁止 seal 不完整43层图".into());
        }
        self.hc_qkv
            .as_mut()
            .ok_or_else(|| "production K-lane Vulkan HC/QKV adapter 尚未注入".to_owned())?
            .seal_and_drain(completed_layers)?;
        self.moe
            .as_mut()
            .ok_or_else(|| "production K-lane Vulkan MoE adapter 尚未注入".to_owned())?
            .seal_and_drain(completed_layers)?;
        self.seal_calls = self
            .seal_calls
            .checked_add(1)
            .ok_or_else(|| "K-lane Vulkan seal counter overflow".to_owned())?;
        self.phase = S14CausalBlockVulkanPhase::LayersSealed {
            base_position,
            block_size,
            bank_index,
        };
        Ok(S14CausalBlockSealReceipt {
            seal_calls: 1,
            completed_layers,
            drained: true,
            active: false,
            head_submit_calls: 0,
            checkpoint_commit_calls: 0,
            serial_token_forward_calls: 0,
        })
    }

    fn drain_and_abort_full_depth_block(
        &mut self,
        completed_layers: usize,
    ) -> Result<S14CausalBlockAbortReceipt, String> {
        if self.phase == S14CausalBlockVulkanPhase::Idle && !self.last_exported_unvalidated {
            return Err("没有 K-lane Vulkan block 可 abort".into());
        }
        let mut adapter_errors = Vec::new();
        if let Some(adapter) = self.hc_qkv.as_mut() {
            if let Err(error) = adapter.drain_and_abort(completed_layers) {
                adapter_errors.push(format!("HC/QKV: {error}"));
            }
        }
        if let Some(adapter) = self.moe.as_mut() {
            if let Err(error) = adapter.drain_and_abort(completed_layers) {
                adapter_errors.push(format!("MoE: {error}"));
            }
        }
        if let Some(terminal) = self.terminal.as_mut() {
            if let Err(error) = terminal.drain_and_abort_batched_terminal(completed_layers) {
                adapter_errors.push(format!("terminal: {error}"));
            }
        }
        self.abort_calls = self
            .abort_calls
            .checked_add(1)
            .ok_or_else(|| "K-lane Vulkan abort counter overflow".to_owned())?;
        self.phase = S14CausalBlockVulkanPhase::Idle;
        self.last_exported_unvalidated = false;
        if !adapter_errors.is_empty() {
            return Err(format!(
                "K-lane Vulkan adapter drain/abort 失败: {}",
                adapter_errors.join("; ")
            ));
        }
        Ok(S14CausalBlockAbortReceipt {
            abort_calls: 1,
            completed_layers,
            drained: true,
            active: false,
            head_submit_calls: 0,
            checkpoint_commit_calls: 0,
        })
    }
}

impl S14CausalBlockCheckpointBackend for S14CausalBlockVulkanBackend {
    fn run_batched_final_head_and_export_checkpoints(
        &mut self,
        completed_layers: usize,
        final_hidden: S14CausalBlockHiddenBinding,
        routes_by_position: &[Vec<RouteDecision>],
    ) -> Result<S14CausalBlockFinalOutput, String> {
        match self.phase {
            S14CausalBlockVulkanPhase::LayersSealed {
                base_position,
                block_size,
                ..
            } => {
                if completed_layers != FULL_DEPTH_LAYERS.len()
                    || final_hidden.block_size != block_size
                    || routes_by_position.len() != block_size
                    || routes_by_position
                        .iter()
                        .any(|routes| routes.len() != FULL_DEPTH_LAYERS.len())
                {
                    return Err("K-lane Vulkan terminal 输入与 sealed 43层图漂移".into());
                }
                let terminal = self.terminal.as_mut().ok_or_else(|| {
                    "production K-lane Vulkan batched terminal/head/checkpoint recorder 尚未接线"
                        .to_owned()
                })?;
                let output = terminal.record_batched_terminal_head_and_checkpoints(
                    completed_layers,
                    base_position,
                    final_hidden,
                    routes_by_position,
                )?;
                self.phase = S14CausalBlockVulkanPhase::Idle;
                self.last_exported_unvalidated = true;
                Ok(output)
            }
            _ => Err("K-lane Vulkan final head 只能在43层 sealed 后执行".into()),
        }
    }

    fn acknowledge_export_validated(&mut self) -> Result<(), String> {
        S14CausalBlockVulkanBackend::acknowledge_export_validated(self)
    }
}

/// Runtime checkpoint arena 的 lease pool。实现者持有真正的 Vulkan buffer/allocator；
/// `release_future_lease` 必须是幂等、无 panic 的本地资源归还操作。
pub trait S14CausalBlockVulkanFutureLeasePool: fmt::Debug {
    fn release_future_lease(&self, lease_id: u64, receipt: S14CausalBlockDeviceFutureReceipt);
}

/// 可从真实 backend 移入 `S14CausalBlockOwnedDeviceFuture` 的不可克隆 Vulkan lease。
/// sealed future 被拒绝、rollback 或 drop 时会把 arena slot 归还给 pool。
#[derive(Debug)]
pub struct S14CausalBlockVulkanFutureLease {
    receipt: S14CausalBlockDeviceFutureReceipt,
    lease_id: u64,
    pool: Arc<dyn S14CausalBlockVulkanFutureLeasePool>,
    armed: bool,
}

impl S14CausalBlockVulkanFutureLease {
    pub fn new(
        receipt: S14CausalBlockDeviceFutureReceipt,
        lease_id: u64,
        pool: Arc<dyn S14CausalBlockVulkanFutureLeasePool>,
    ) -> Result<Self, String> {
        if lease_id == 0 {
            return Err("K-lane Vulkan future lease_id 不能为0".into());
        }
        Ok(Self {
            receipt,
            lease_id,
            pool,
            armed: true,
        })
    }
}

impl S14CausalBlockDeviceFutureOwner for S14CausalBlockVulkanFutureLease {
    fn receipt(&self) -> S14CausalBlockDeviceFutureReceipt {
        self.receipt
    }
}

impl Drop for S14CausalBlockVulkanFutureLease {
    fn drop(&mut self) {
        if self.armed {
            self.pool.release_future_lease(self.lease_id, self.receipt);
            self.armed = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    #[test]
    fn unvalidated_export_blocks_next_block_until_orchestrator_ack() {
        let mut backend = S14CausalBlockVulkanBackend::default();
        backend.last_exported_unvalidated = true;
        let bank = S14CausalBlockUnionBankBinding {
            bank_index: 0,
            buffer: vk::Buffer::from_raw(1),
            allocated_bank_bytes: 1,
        };
        assert!(backend.begin_full_depth_block(bank, 0, 4).is_err());
        backend.acknowledge_export_validated().unwrap();
        backend.begin_full_depth_block(bank, 0, 4).unwrap();
        backend.drain_and_abort_full_depth_block(0).unwrap();
    }
}
