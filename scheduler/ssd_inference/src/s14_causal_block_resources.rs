//! S14 production K=4/8 causal-block union routed bank 的容量合同。
//!
//! 当前单 token routed bank 只容纳 top-6（80,216,064 B）。真正的 block-major
//! forward 必须在同一层同时保留 K 个 position 的 `(layer, expert)` union；因此
//! 生产分配按最坏 `K*6` 个互异专家，而不是按一条观测 trace 欠配。
//!
//! 这里只建立资源/启动边界，不执行模型，也绝不循环调用 `S14Runtime::step`。

use crate::{
    s14_runtime::{S14Runtime, S14Session},
    GpuBuffer, VulkanContext,
};
use ash::vk;
use polaris_s14_runner::{LayerCausalBatchPlan, EXPERT_PAGE_BYTES, VOCAB_SIZE};
use std::fmt;

pub const S14_CAUSAL_BLOCK_SIZES: [usize; 2] = [4, 8];
pub const S14_CAUSAL_BLOCK_TOP_K: usize = 6;
pub const S14_CAUSAL_BLOCK_UNION_BANKS: usize = 2;
pub const S14_CAUSAL_BLOCK_PHYSICAL_RANGES_PER_EXPERT: usize = 6;
pub const S14_CAUSAL_BLOCK_BANK_ALIGNMENT: u64 = 256;

/// 当前 frozen production manifest 的单 token routed bank。
pub const S14_LEGACY_ROUTED_BANK_BYTES: u64 = 80_216_064;
pub const S14_LEGACY_ROUTED_DEVICE_BYTES: u64 =
    S14_LEGACY_ROUTED_BANK_BYTES * S14_CAUSAL_BLOCK_UNION_BANKS as u64;

/// 2026-08-03 N=8 真门观测值，仅作 telemetry，不能替代最坏容量门。
pub const S14_N8_OBSERVED_MAX_UNIQUE_PAGES_K4: usize = 24;
pub const S14_N8_OBSERVED_MAX_UNIQUE_PAGES_K8: usize = 46;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14CausalBlockUnionBankPlan {
    pub block_size: usize,
    pub top_k: usize,
    pub bank_count: usize,
    pub alignment: u64,
    pub expert_page_bytes: u64,
    pub max_unique_pages_per_layer: usize,
    pub max_physical_ranges_per_layer: usize,
    pub logical_bank_bytes: u64,
    pub allocated_bank_bytes: u64,
    pub allocated_device_bytes: u64,
    pub legacy_bank_bytes: u64,
    pub additional_bank_bytes: u64,
    pub additional_device_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14CausalBlockLayerUnionPlacement {
    pub unique_pages: usize,
    pub physical_ranges: usize,
    pub used_bytes: u64,
    pub slack_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14CausalBlockCapacityReceipt {
    pub required_bank_bytes: u64,
    pub allocated_bank_bytes: u64,
    pub required_bank_count: usize,
    pub allocated_bank_count: usize,
    pub required_device_bytes: u64,
    pub allocated_device_bytes: u64,
}

/// Runtime 常驻的一对 block-major routed union bank。
///
/// 物理 bank 按 K=8 的最坏 48 个唯一专家页分配；K=4 复用同一对 bank 的前缀，
/// 因此不会为 K=4 再重复占用一对设备缓冲。两种 K 的容量门都必须在构造时通过。
pub struct S14CausalBlockUnionBanks {
    k4_plan: S14CausalBlockUnionBankPlan,
    k8_plan: S14CausalBlockUnionBankPlan,
    k4_capacity: S14CausalBlockCapacityReceipt,
    k8_capacity: S14CausalBlockCapacityReceipt,
    banks: [GpuBuffer; S14_CAUSAL_BLOCK_UNION_BANKS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14CausalBlockLaunch {
    pub base_position: u32,
    pub block_size: usize,
    pub input_token_id: u32,
    pub draft_token_ids: Vec<u32>,
    pub union_banks: S14CausalBlockUnionBankPlan,
}

impl S14CausalBlockUnionBankPlan {
    pub fn build(block_size: usize) -> Result<Self, S14CausalBlockResourceError> {
        Self::build_aligned(block_size, S14_CAUSAL_BLOCK_BANK_ALIGNMENT)
    }

    pub fn build_aligned(
        block_size: usize,
        alignment: u64,
    ) -> Result<Self, S14CausalBlockResourceError> {
        if !S14_CAUSAL_BLOCK_SIZES.contains(&block_size) {
            return Err(resource_error("causal-block union bank 只允许 K=4/8"));
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(resource_error("union bank alignment 必须是非零二次幂"));
        }
        let max_unique_pages_per_layer = block_size
            .checked_mul(S14_CAUSAL_BLOCK_TOP_K)
            .ok_or_else(|| resource_error("union page count overflow"))?;
        let max_physical_ranges_per_layer = max_unique_pages_per_layer
            .checked_mul(S14_CAUSAL_BLOCK_PHYSICAL_RANGES_PER_EXPERT)
            .ok_or_else(|| resource_error("union physical Range count overflow"))?;
        let logical_bank_bytes = u64::try_from(max_unique_pages_per_layer)
            .ok()
            .and_then(|pages| pages.checked_mul(EXPERT_PAGE_BYTES))
            .ok_or_else(|| resource_error("union bank logical bytes overflow"))?;
        let allocated_bank_bytes = align_up(logical_bank_bytes, alignment)?;
        let allocated_device_bytes = allocated_bank_bytes
            .checked_mul(S14_CAUSAL_BLOCK_UNION_BANKS as u64)
            .ok_or_else(|| resource_error("union bank device bytes overflow"))?;
        let additional_bank_bytes = allocated_bank_bytes
            .checked_sub(S14_LEGACY_ROUTED_BANK_BYTES)
            .ok_or_else(|| resource_error("union bank unexpectedly below legacy bank"))?;
        let additional_device_bytes = allocated_device_bytes
            .checked_sub(S14_LEGACY_ROUTED_DEVICE_BYTES)
            .ok_or_else(|| resource_error("union device bytes unexpectedly below legacy banks"))?;
        Ok(Self {
            block_size,
            top_k: S14_CAUSAL_BLOCK_TOP_K,
            bank_count: S14_CAUSAL_BLOCK_UNION_BANKS,
            alignment,
            expert_page_bytes: EXPERT_PAGE_BYTES,
            max_unique_pages_per_layer,
            max_physical_ranges_per_layer,
            logical_bank_bytes,
            allocated_bank_bytes,
            allocated_device_bytes,
            legacy_bank_bytes: S14_LEGACY_ROUTED_BANK_BYTES,
            additional_bank_bytes,
            additional_device_bytes,
        })
    }

    /// 真实分配后必须通过本门，才能进入 block command recording。
    pub fn gate_allocated_capacity(
        &self,
        allocated_bank_bytes: u64,
        allocated_bank_count: usize,
    ) -> Result<S14CausalBlockCapacityReceipt, S14CausalBlockResourceError> {
        let allocated_device_bytes = allocated_bank_bytes
            .checked_mul(allocated_bank_count as u64)
            .ok_or_else(|| resource_error("allocated union device bytes overflow"))?;
        if allocated_bank_count < self.bank_count
            || allocated_bank_bytes < self.allocated_bank_bytes
            || allocated_device_bytes < self.allocated_device_bytes
        {
            return Err(resource_error(format!(
                "K={} union bank capacity 不足: per_bank={allocated_bank_bytes}/{} banks={allocated_bank_count}/{} device={allocated_device_bytes}/{}",
                self.block_size,
                self.allocated_bank_bytes,
                self.bank_count,
                self.allocated_device_bytes,
            )));
        }
        Ok(S14CausalBlockCapacityReceipt {
            required_bank_bytes: self.allocated_bank_bytes,
            allocated_bank_bytes,
            required_bank_count: self.bank_count,
            allocated_bank_count,
            required_device_bytes: self.allocated_device_bytes,
            allocated_device_bytes,
        })
    }

    /// 路由 readback 后的逐层实际用量门。任何超过 K×top-6 的 trace 都拒绝。
    pub fn place_layer_union(
        &self,
        unique_pages: usize,
    ) -> Result<S14CausalBlockLayerUnionPlacement, S14CausalBlockResourceError> {
        if unique_pages == 0 || unique_pages > self.max_unique_pages_per_layer {
            return Err(resource_error(format!(
                "K={} layer union unique pages 非法: {unique_pages}, max={}",
                self.block_size, self.max_unique_pages_per_layer
            )));
        }
        let used_bytes = u64::try_from(unique_pages)
            .ok()
            .and_then(|pages| pages.checked_mul(EXPERT_PAGE_BYTES))
            .ok_or_else(|| resource_error("layer union bytes overflow"))?;
        if used_bytes > self.allocated_bank_bytes {
            return Err(resource_error("layer union bytes 超出已规划 bank"));
        }
        Ok(S14CausalBlockLayerUnionPlacement {
            unique_pages,
            physical_ranges: unique_pages * S14_CAUSAL_BLOCK_PHYSICAL_RANGES_PER_EXPERT,
            used_bytes,
            slack_bytes: self.allocated_bank_bytes - used_bytes,
        })
    }

    /// 把 runner 的无损 `(layer, expert)` union 计划直接绑定到 production bank。
    pub fn place_layer_plan(
        &self,
        layer: &LayerCausalBatchPlan,
    ) -> Result<S14CausalBlockLayerUnionPlacement, S14CausalBlockResourceError> {
        if layer.block_size != self.block_size
            || layer.assignments != self.block_size * self.top_k
            || layer.unique_experts != layer.experts.len()
        {
            return Err(resource_error(
                "runner layer union plan 与 production bank K/top-k 维度漂移",
            ));
        }
        let placement = self.place_layer_union(layer.unique_experts)?;
        if placement.used_bytes != layer.union_expert_bytes {
            return Err(resource_error(
                "runner layer union bytes 与 production bank placement 漂移",
            ));
        }
        Ok(placement)
    }
}

impl S14CausalBlockUnionBanks {
    /// 在任何可选静态缓存之前分配 production union banks。
    /// 任一 bank 分配或 K4/K8 容量门失败时，已分配的设备缓冲会在返回前销毁。
    pub fn new(ctx: &VulkanContext) -> Result<Self, S14CausalBlockResourceError> {
        let k4_plan = S14CausalBlockUnionBankPlan::build(4)?;
        let k8_plan = S14CausalBlockUnionBankPlan::build(8)?;
        if k8_plan.allocated_device_bytes > ctx.vram_size() {
            return Err(resource_error(format!(
                "K8 union banks 超出 Vulkan heap: required={} heap={}",
                k8_plan.allocated_device_bytes,
                ctx.vram_size()
            )));
        }

        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let mut pending = PendingUnionBanks::new(ctx);
        for bank in 0..S14_CAUSAL_BLOCK_UNION_BANKS {
            pending
                .allocate(k8_plan.allocated_bank_bytes, usage)
                .map_err(|error| {
                    resource_error(format!(
                        "allocate S14 causal-block union bank {bank} 失败: {error}"
                    ))
                })?;
        }
        let allocated_bank_bytes = pending
            .minimum_bank_bytes()
            .ok_or_else(|| resource_error("union bank allocation ledger 为空"))?;
        let allocated_bank_count = pending.len();
        let (k4_capacity, k8_capacity) = gate_shared_union_bank_capacity(
            &k4_plan,
            &k8_plan,
            allocated_bank_bytes,
            allocated_bank_count,
        )?;
        let banks: [GpuBuffer; S14_CAUSAL_BLOCK_UNION_BANKS] = pending
            .finish()
            .try_into()
            .map_err(|buffers: Vec<GpuBuffer>| {
                for buffer in &buffers {
                    buffer.destroy(ctx);
                }
                resource_error(format!(
                    "union bank allocation count 漂移: actual={} expected={S14_CAUSAL_BLOCK_UNION_BANKS}",
                    buffers.len()
                ))
            })?;
        Ok(Self {
            k4_plan,
            k8_plan,
            k4_capacity,
            k8_capacity,
            banks,
        })
    }

    pub fn plan(
        &self,
        block_size: usize,
    ) -> Result<&S14CausalBlockUnionBankPlan, S14CausalBlockResourceError> {
        match block_size {
            4 => Ok(&self.k4_plan),
            8 => Ok(&self.k8_plan),
            _ => Err(resource_error("causal-block union bank 只允许 K=4/8")),
        }
    }

    pub fn capacity(
        &self,
        block_size: usize,
    ) -> Result<&S14CausalBlockCapacityReceipt, S14CausalBlockResourceError> {
        match block_size {
            4 => Ok(&self.k4_capacity),
            8 => Ok(&self.k8_capacity),
            _ => Err(resource_error("causal-block union bank 只允许 K=4/8")),
        }
    }

    pub fn bank(&self, bank: usize) -> Result<&GpuBuffer, S14CausalBlockResourceError> {
        self.banks
            .get(bank)
            .ok_or_else(|| resource_error(format!("invalid causal-block union bank {bank}")))
    }

    pub fn allocated_bank_bytes(&self) -> u64 {
        self.banks.iter().map(GpuBuffer::size).min().unwrap_or(0)
    }

    pub fn allocated_device_bytes(&self) -> Result<u64, S14CausalBlockResourceError> {
        self.banks.iter().try_fold(0u64, |sum, buffer| {
            sum.checked_add(buffer.size())
                .ok_or_else(|| resource_error("union bank allocated bytes overflow"))
        })
    }

    pub fn destroy(self, ctx: &VulkanContext) {
        for bank in &self.banks {
            bank.destroy(ctx);
        }
    }
}

struct PendingUnionBanks<'a> {
    ctx: &'a VulkanContext,
    buffers: Vec<GpuBuffer>,
}

impl<'a> PendingUnionBanks<'a> {
    fn new(ctx: &'a VulkanContext) -> Self {
        Self {
            ctx,
            buffers: Vec::with_capacity(S14_CAUSAL_BLOCK_UNION_BANKS),
        }
    }

    fn allocate(&mut self, bytes: u64, usage: vk::BufferUsageFlags) -> Result<(), String> {
        let buffer =
            GpuBuffer::new_vram(self.ctx, bytes, usage).map_err(|error| format!("{error:#}"))?;
        self.buffers.push(buffer);
        Ok(())
    }

    fn len(&self) -> usize {
        self.buffers.len()
    }

    fn minimum_bank_bytes(&self) -> Option<u64> {
        self.buffers.iter().map(GpuBuffer::size).min()
    }

    fn finish(mut self) -> Vec<GpuBuffer> {
        std::mem::take(&mut self.buffers)
    }
}

impl Drop for PendingUnionBanks<'_> {
    fn drop(&mut self) {
        for buffer in &self.buffers {
            buffer.destroy(self.ctx);
        }
    }
}

fn gate_shared_union_bank_capacity(
    k4_plan: &S14CausalBlockUnionBankPlan,
    k8_plan: &S14CausalBlockUnionBankPlan,
    allocated_bank_bytes: u64,
    allocated_bank_count: usize,
) -> Result<
    (S14CausalBlockCapacityReceipt, S14CausalBlockCapacityReceipt),
    S14CausalBlockResourceError,
> {
    if k4_plan.block_size != 4 || k8_plan.block_size != 8 {
        return Err(resource_error("shared union bank K4/K8 plan identity 漂移"));
    }
    let k4 = k4_plan.gate_allocated_capacity(allocated_bank_bytes, allocated_bank_count)?;
    let k8 = k8_plan.gate_allocated_capacity(allocated_bank_bytes, allocated_bank_count)?;
    Ok((k4, k8))
}

/// 纯合同构建器，供无需 GPU 的秒级测试与未来 production loader 共用。
pub fn prepare_causal_block_launch(
    base_position: u32,
    input_token_id: u32,
    draft_token_ids: &[u32],
) -> Result<S14CausalBlockLaunch, S14CausalBlockResourceError> {
    if input_token_id >= VOCAB_SIZE
        || draft_token_ids
            .iter()
            .any(|&token_id| token_id >= VOCAB_SIZE)
    {
        return Err(resource_error("causal-block token ID 越出冻结 vocab"));
    }
    let union_banks = S14CausalBlockUnionBankPlan::build(draft_token_ids.len())?;
    let end_position = base_position
        .checked_add(draft_token_ids.len() as u32)
        .ok_or_else(|| resource_error("causal-block position overflow"))?;
    if end_position > 127 {
        return Err(resource_error(
            "causal-block 越出当前 whole-token position<127 production 边界",
        ));
    }
    Ok(S14CausalBlockLaunch {
        base_position,
        block_size: draft_token_ids.len(),
        input_token_id,
        draft_token_ids: draft_token_ids.to_vec(),
        union_banks,
    })
}

impl S14Runtime {
    /// Production K=4/8 入口骨架。runtime 已常驻按 K8 最坏容量分配的双 union bank，
    /// 但 block-major graph 尚未接线，因此仍必须在 command recording 前 fail-closed。
    ///
    /// 后续接线只能用 block-major graph 填充此入口；禁止在这里循环调用 `step()`。
    pub fn begin_causal_block(
        &mut self,
        session: &mut S14Session,
        draft_token_ids: &[u32],
    ) -> Result<S14CausalBlockLaunch, S14CausalBlockResourceError> {
        let launch = prepare_causal_block_launch(
            session.position(),
            session.input_token_id(),
            draft_token_ids,
        )?;
        let runtime_banks = self
            .causal_block_union_banks()
            .ok_or_else(|| resource_error("S14 runtime union banks 已销毁"))?;
        let runtime_plan = runtime_banks.plan(launch.block_size)?;
        if runtime_plan != &launch.union_banks {
            return Err(resource_error(
                "runtime union bank plan 与 causal-block launch 漂移",
            ));
        }
        let _capacity = runtime_banks.capacity(launch.block_size)?;
        Err(resource_error(
            "K4/K8 union banks 已常驻并通过容量门，但 K-lane block-major graph 尚未接线",
        ))
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, S14CausalBlockResourceError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or_else(|| resource_error("union bank alignment overflow"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S14CausalBlockResourceError(String);

impl fmt::Display for S14CausalBlockResourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for S14CausalBlockResourceError {}

fn resource_error(message: impl Into<String>) -> S14CausalBlockResourceError {
    S14CausalBlockResourceError(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use polaris_s14_runner::{build_layer_causal_batch_plan, router_kind_for_layer, RouteDecision};

    #[test]
    fn k4_and_k8_allocate_worst_case_union_pages_not_observed_average() {
        let k4 = S14CausalBlockUnionBankPlan::build(4).unwrap();
        assert_eq!(k4.max_unique_pages_per_layer, 24);
        assert_eq!(k4.max_physical_ranges_per_layer, 144);
        assert_eq!(k4.allocated_bank_bytes, 320_864_256);
        assert_eq!(k4.allocated_device_bytes, 641_728_512);
        assert_eq!(k4.additional_device_bytes, 481_296_384);

        let k8 = S14CausalBlockUnionBankPlan::build(8).unwrap();
        assert_eq!(k8.max_unique_pages_per_layer, 48);
        assert_eq!(k8.max_physical_ranges_per_layer, 288);
        assert_eq!(k8.allocated_bank_bytes, 641_728_512);
        assert_eq!(k8.allocated_device_bytes, 1_283_457_024);
        assert_eq!(k8.additional_device_bytes, 1_123_024_896);
    }

    #[test]
    fn legacy_top6_banks_fail_closed_for_k4_and_k8() {
        for block_size in S14_CAUSAL_BLOCK_SIZES {
            let plan = S14CausalBlockUnionBankPlan::build(block_size).unwrap();
            let error = plan
                .gate_allocated_capacity(S14_LEGACY_ROUTED_BANK_BYTES, S14_CAUSAL_BLOCK_UNION_BANKS)
                .unwrap_err();
            assert!(error.to_string().contains("capacity 不足"));
        }
    }

    #[test]
    fn n8_observed_46_pages_fit_but_do_not_shrink_the_k8_allocation() {
        let plan = S14CausalBlockUnionBankPlan::build(8).unwrap();
        let observed = plan
            .place_layer_union(S14_N8_OBSERVED_MAX_UNIQUE_PAGES_K8)
            .unwrap();
        assert_eq!(observed.used_bytes, 614_989_824);
        assert_eq!(observed.physical_ranges, 276);
        assert_eq!(observed.slack_bytes, 26_738_688);
        assert_eq!(observed.slack_bytes, 2 * EXPERT_PAGE_BYTES);
        assert!(plan.place_layer_union(49).is_err());
    }

    #[test]
    fn full_capacity_gate_and_launch_contract_are_compile_ready() {
        let launch = prepare_causal_block_launch(8, 5, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let receipt = launch
            .union_banks
            .gate_allocated_capacity(641_728_512, 2)
            .unwrap();
        assert_eq!(receipt.required_device_bytes, 1_283_457_024);
        assert_eq!(launch.base_position, 8);
        assert_eq!(launch.block_size, 8);
        assert!(prepare_causal_block_launch(124, 5, &[1, 2, 3, 4]).is_err());
        assert!(prepare_causal_block_launch(0, 5, &[1, 2, 3]).is_err());
    }

    #[test]
    fn one_shared_k8_sized_pair_gates_k4_and_k8_and_rejects_partial_allocation() {
        let k4 = S14CausalBlockUnionBankPlan::build(4).unwrap();
        let k8 = S14CausalBlockUnionBankPlan::build(8).unwrap();
        let (k4_capacity, k8_capacity) = gate_shared_union_bank_capacity(
            &k4,
            &k8,
            k8.allocated_bank_bytes,
            S14_CAUSAL_BLOCK_UNION_BANKS,
        )
        .unwrap();
        assert_eq!(k4_capacity.required_bank_bytes, 320_864_256);
        assert_eq!(k8_capacity.required_bank_bytes, 641_728_512);
        assert_eq!(k4_capacity.allocated_bank_bytes, k8.allocated_bank_bytes);
        assert_eq!(k8_capacity.allocated_device_bytes, 1_283_457_024);

        assert!(gate_shared_union_bank_capacity(&k4, &k8, k8.allocated_bank_bytes, 1,).is_err());
        assert!(gate_shared_union_bank_capacity(
            &k4,
            &k8,
            k8.allocated_bank_bytes - 256,
            S14_CAUSAL_BLOCK_UNION_BANKS,
        )
        .is_err());
    }

    #[test]
    fn runner_union_plan_binds_directly_to_k8_production_bank() {
        let routes = (0..8)
            .map(|token| RouteDecision {
                layer: 1,
                kind: router_kind_for_layer(1).unwrap(),
                expert_ids: (token * 6..token * 6 + 6)
                    .map(|expert| expert as u16)
                    .collect(),
                weights: vec![0.25; 6],
            })
            .collect::<Vec<_>>();
        let layer = build_layer_causal_batch_plan(&routes).unwrap();
        let bank = S14CausalBlockUnionBankPlan::build(8).unwrap();
        let placement = bank.place_layer_plan(&layer).unwrap();
        assert_eq!(placement.unique_pages, 48);
        assert_eq!(placement.used_bytes, bank.allocated_bank_bytes);
        assert_eq!(placement.physical_ranges, 288);
    }
}
