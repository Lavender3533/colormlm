//! 把 FullDepth43 的真实 layer program 与 paged Vulkan arena 收敛成同步首 token 换页计划。
//!
//! 这里不执行上传或计算；它只消除控制面与真实资产布局之间的最后一层手工拼装。

use crate::{
    s14_position0_layer_program::{S14Position0FullDepthLayerProgram, S14Position0WeightBinding},
    s14_position0_paged_weight_arena::{
        S14Position0PagedWeightArena, S14Position0StaticLayerBinding,
    },
    s14_position0_synchronous_layer_pager::{
        S14Position0DeviceHiddenBinding, S14Position0DeviceHiddenSlot, S14Position0StaticPage,
        S14Position0SynchronousLayerPlan, S14Position0WeightPageTarget, S14Position0WeightUpload,
        S14_POSITION0_SYNCHRONOUS_LAYER_COUNT,
    },
    s14_position0_workspace::S14Position0WorkspaceSlot,
};
use anyhow::{bail, Result};

pub fn build_synchronous_layer_plans(
    program: &S14Position0FullDepthLayerProgram,
    arena: &S14Position0PagedWeightArena,
) -> Result<Vec<S14Position0SynchronousLayerPlan>> {
    if program.layers.len() != usize::from(S14_POSITION0_SYNCHRONOUS_LAYER_COUNT) {
        bail!("position0 synchronous program 不是43层");
    }
    let mut plans = Vec::with_capacity(program.layers.len());
    for layer in &program.layers {
        let (static_page, static_page_bytes) = match arena.static_layer(layer.layer)? {
            S14Position0StaticLayerBinding::Resident { layout, .. } => (
                S14Position0StaticPage::Resident { layer: layer.layer },
                layout.requested_bytes,
            ),
            S14Position0StaticLayerBinding::Streamed { bank, layout, .. } => (
                S14Position0StaticPage::Streamed { bank },
                layout.requested_bytes,
            ),
        };
        let static_target = S14Position0WeightPageTarget::Static(static_page);
        let routed_target = S14Position0WeightPageTarget::Routed {
            bank: layer.routed_bank,
        };
        let static_weights = layer
            .static_weights
            .iter()
            .chain(&layer.router_weights)
            .chain(&layer.shared_weights)
            .map(|binding| to_upload(binding, static_target))
            .collect::<Vec<_>>();
        let routed_weights = layer
            .routed_weights
            .iter()
            .map(|binding| to_upload(binding, routed_target))
            .collect::<Vec<_>>();
        let hidden = S14Position0DeviceHiddenBinding {
            input: hidden_slot(layer.hidden.input_slot)?,
            output: hidden_slot(layer.hidden.output_slot)?,
        };
        let plan = S14Position0SynchronousLayerPlan {
            layer: layer.layer,
            static_page,
            static_page_bytes,
            routed_bank: layer.routed_bank,
            routed_page_bytes: arena.plan().physical.routed_bank_bytes,
            static_weights,
            routed_weights,
            hidden,
        };
        plan.validate()?;
        plans.push(plan);
    }
    for pair in plans.windows(2) {
        if pair[0].hidden.output != pair[1].hidden.input {
            bail!(
                "position0 synchronous hidden 在 L{}→L{} 断链",
                pair[0].layer,
                pair[1].layer
            );
        }
    }
    Ok(plans)
}

fn to_upload(
    binding: &S14Position0WeightBinding,
    target: S14Position0WeightPageTarget,
) -> S14Position0WeightUpload {
    S14Position0WeightUpload {
        tensor: binding.tensor.clone(),
        kind: binding.kind.clone(),
        expert_id: binding.expert_id,
        source_path: binding.path.clone(),
        sha256: binding.sha256.clone(),
        destination_offset: binding.offset,
        bytes: binding.bytes,
        target,
    }
}

fn hidden_slot(slot: S14Position0WorkspaceSlot) -> Result<S14Position0DeviceHiddenSlot> {
    match slot {
        S14Position0WorkspaceSlot::HiddenStreamsA => Ok(S14Position0DeviceHiddenSlot::A),
        S14Position0WorkspaceSlot::HiddenStreamsB => Ok(S14Position0DeviceHiddenSlot::B),
        _ => bail!("position0 synchronous layer hidden 绑定到非 hidden workspace 槽"),
    }
}
