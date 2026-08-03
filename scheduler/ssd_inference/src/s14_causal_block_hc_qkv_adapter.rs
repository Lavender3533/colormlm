//! S14 K=4/8 causal-block HC/QKV/attention/router production adapter。
//!
//! 本模块只建立严格 owner 与机器可校验的接线边界，不提供伪数值 fallback。底层 recorder
//! 必须在一次 K-row command graph 中按以下顺序消费正式 `[K,4,4096]` BF16 hidden：
//!
//! 1. attention HC-pre 与 Q/current-KV 投影；
//! 2. K-lane causal attention；
//! 3. wo_a/wo_b 与 attention HC-post，写出新的 `[K,4,4096]` BF16 hidden；
//! 4. FFN HC-pre 生成 `[K,4096]` F32 router input；
//! 5. batched router 权重扫描与 K-row top-6。
//!
//! 若未注入实现该数值链的 recorder，Vulkan backend 保持 fail-closed；本模块不会用 K 次
//! whole-token、capture 或 BOS replay 伪装 K-lane。

use crate::s14_causal_block_layer::{
    S14CausalBlockAttentionRouterOutput, S14CausalBlockGroupedMoeOutput,
    S14CausalBlockHiddenBinding, S14CausalBlockLayerInput, S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE,
};
use ash::vk;
use polaris_s14_runner::{GraphProfile, FULL_DEPTH_LAYERS};
use std::fmt;

const BF16_BYTES: u64 = 2;

/// 单层真实数值 recorder 的强回执。各 record_calls 是一个有序 recorder scope，而不是声称
/// 该 scope 只含一个 shader dispatch；HC/QKV 的实际多个 dispatch 仍由同一个 command graph
/// owner 持有。attention/router 两段必须调用拆分后的 K-lane recorder 各一次。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct S14CausalBlockHcQkvLayerRecordingReceipt {
    pub base_position: u32,
    pub layer: u8,
    pub block_size: usize,
    pub input_hidden: S14CausalBlockHiddenBinding,
    pub post_attention_hidden: S14CausalBlockHiddenBinding,
    pub layer_record_calls: u32,
    pub command_graph_submit_calls: u32,
    pub hc_qkv_projection_record_calls: u32,
    pub attention_recording_calls: u32,
    pub attention_output_post_record_calls: u32,
    pub ffn_hc_router_input_record_calls: u32,
    pub router_recording_calls: u32,
    pub serial_token_forward_calls: u32,
    pub hc_hidden_integration_complete: bool,
}

impl S14CausalBlockHcQkvLayerRecordingReceipt {
    pub fn validate(
        self,
        input: &S14CausalBlockLayerInput<'_>,
        output: &S14CausalBlockAttentionRouterOutput,
    ) -> Result<(), String> {
        validate_hidden(input.input_hidden, input.input_token_ids.len(), "input")?;
        validate_hidden(
            output.post_attention_hidden,
            input.input_token_ids.len(),
            "post-attention output",
        )?;
        let expected_generation = input
            .input_hidden
            .generation
            .checked_add(1)
            .ok_or_else(|| "causal-block HC/QKV hidden generation overflow".to_owned())?;
        if self.base_position != input.base_position
            || self.layer != input.layer
            || self.block_size != input.input_token_ids.len()
            || self.input_hidden != input.input_hidden
            || self.post_attention_hidden != output.post_attention_hidden
            || self.layer_record_calls != 1
            || self.command_graph_submit_calls != 1
            || self.hc_qkv_projection_record_calls != 1
            || self.attention_recording_calls != 1
            || self.attention_output_post_record_calls != 1
            || self.ffn_hc_router_input_record_calls != 1
            || self.router_recording_calls != 1
            || self.serial_token_forward_calls != 0
            || !self.hc_hidden_integration_complete
            || output.forward_calls != 1
            || output.routes.len() != input.input_token_ids.len()
            || output.post_attention_hidden.generation != expected_generation
            || ranges_overlap(input.input_hidden, output.post_attention_hidden)?
        {
            return Err(
                "causal-block HC/QKV 回执未闭合一次 K-row hidden→Q/KV→attention→post→router 图"
                    .into(),
            );
        }
        for route in &output.routes {
            route
                .validate_for(GraphProfile::FullDepth43NativeTop6)
                .map_err(|error| format!("causal-block HC/QKV online route 非法: {error}"))?;
            if route.layer != input.layer {
                return Err("causal-block HC/QKV route layer 漂移".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct S14CausalBlockHcQkvRecordedLayer {
    pub output: S14CausalBlockAttentionRouterOutput,
    pub receipt: S14CausalBlockHcQkvLayerRecordingReceipt,
}

/// backend 持有的对象安全 HC/QKV adapter 边界。返回值保留完整强回执，backend 必须再次
/// 校验后才能把 output 交给 orchestrator 与 MoE adapter。
pub trait S14CausalBlockVulkanHcQkvAdapter: fmt::Debug {
    fn begin_block(&mut self, base_position: u32, block_size: usize) -> Result<(), String>;

    fn run_k_lane_hc_qkv_attention_router(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockHcQkvRecordedLayer, String>;

    /// grouped-MoE 已真实返回后，由 backend 回接其 output binding。这一步使下一层 input
    /// identity 可被逐字节验证，而不是只检查 generation。
    fn capture_grouped_moe_output(
        &mut self,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        output: &S14CausalBlockGroupedMoeOutput,
    ) -> Result<(), String>;

    fn seal_and_drain(&mut self, completed_layers: usize) -> Result<(), String>;

    fn drain_and_abort(&mut self, completed_layers: usize) -> Result<(), String>;

    fn finish_validated_block(&mut self) -> Result<(), String>;

    fn destroy(&mut self) -> Result<(), String>;
}

/// 真实 Vulkan command/timeline owner 的最窄注入点。production 实现必须直接录制 K 行；
/// 本 trait 不提供任何逐 token 方法。
pub trait S14CausalBlockHcQkvLayerRecorder: fmt::Debug {
    fn begin_block(&mut self, base_position: u32, block_size: usize) -> Result<(), String>;

    fn record_k_lane_hc_qkv_attention_router(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockHcQkvRecordedLayer, String>;

    fn seal_and_drain(&mut self, completed_layers: usize) -> Result<(), String>;

    fn drain_and_abort(&mut self, completed_layers: usize) -> Result<(), String>;

    fn finish_validated_block(&mut self) -> Result<(), String>;

    fn destroy(&mut self) -> Result<(), String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterLayerPhase {
    AwaitingAttention {
        expected_input: Option<S14CausalBlockHiddenBinding>,
    },
    AwaitingGrouped {
        post_attention_hidden: S14CausalBlockHiddenBinding,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdapterActiveBlock {
    base_position: u32,
    block_size: usize,
    next_layer: usize,
    layer_phase: AdapterLayerPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdapterPhase {
    Idle,
    Active(AdapterActiveBlock),
    LayersSealed,
    Poisoned {
        completed_layers: usize,
        drained: bool,
    },
    Destroyed,
}

pub struct S14CausalBlockProductionHcQkvAdapter<R: S14CausalBlockHcQkvLayerRecorder> {
    recorder: R,
    phase: AdapterPhase,
}

impl<R: S14CausalBlockHcQkvLayerRecorder> fmt::Debug for S14CausalBlockProductionHcQkvAdapter<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("S14CausalBlockProductionHcQkvAdapter")
            .field("phase", &self.phase)
            .field("recorder", &self.recorder)
            .finish()
    }
}

impl<R: S14CausalBlockHcQkvLayerRecorder> S14CausalBlockProductionHcQkvAdapter<R> {
    pub fn new(recorder: R) -> Self {
        Self {
            recorder,
            phase: AdapterPhase::Idle,
        }
    }

    fn begin_block_inner(&mut self, base_position: u32, block_size: usize) -> Result<(), String> {
        if self.phase != AdapterPhase::Idle {
            return Err("causal-block HC/QKV adapter 已有未释放 block".into());
        }
        if !matches!(block_size, 4 | 8) {
            return Err("causal-block HC/QKV adapter K 只允许4或8".into());
        }
        let end = base_position
            .checked_add(block_size as u32)
            .ok_or_else(|| "causal-block HC/QKV block position overflow".to_owned())?;
        if base_position == 0 || end > 127 {
            return Err("causal-block HC/QKV 首版只允许 position 1..126 contiguous window".into());
        }
        self.recorder.begin_block(base_position, block_size)?;
        self.phase = AdapterPhase::Active(AdapterActiveBlock {
            base_position,
            block_size,
            next_layer: 0,
            layer_phase: AdapterLayerPhase::AwaitingAttention {
                expected_input: None,
            },
        });
        Ok(())
    }

    fn run_layer_inner(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockHcQkvRecordedLayer, String> {
        let active = self.active()?;
        let expected_layer = expected_layer(active.next_layer)?;
        let expected_input = match active.layer_phase {
            AdapterLayerPhase::AwaitingAttention { expected_input } => expected_input,
            AdapterLayerPhase::AwaitingGrouped { .. } => {
                return Err("causal-block HC/QKV 上一层 grouped output 尚未回接".into())
            }
        };
        if input.base_position != active.base_position
            || input.layer != expected_layer
            || input.input_token_ids.len() != active.block_size
            || input.input_hidden.block_size != active.block_size
            || expected_input.is_some_and(|expected| expected != input.input_hidden)
        {
            return Err(
                "causal-block HC/QKV input 与 active block/layer/上一层 output 漂移".into(),
            );
        }
        validate_hidden(input.input_hidden, active.block_size, "input")?;

        let recorded = self.recorder.record_k_lane_hc_qkv_attention_router(input)?;
        recorded.receipt.validate(input, &recorded.output)?;
        let active = self.active_mut()?;
        active.layer_phase = AdapterLayerPhase::AwaitingGrouped {
            post_attention_hidden: recorded.output.post_attention_hidden,
        };
        Ok(recorded)
    }

    fn capture_grouped_inner(
        &mut self,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        output: &S14CausalBlockGroupedMoeOutput,
    ) -> Result<(), String> {
        let active = self.active()?;
        let captured_post = match active.layer_phase {
            AdapterLayerPhase::AwaitingGrouped {
                post_attention_hidden,
            } => post_attention_hidden,
            AdapterLayerPhase::AwaitingAttention { .. } => {
                return Err("causal-block HC/QKV grouped output 前缺少同层 attention".into())
            }
        };
        validate_hidden(output.output_hidden, active.block_size, "grouped output")?;
        let expected_generation = captured_post
            .generation
            .checked_add(1)
            .ok_or_else(|| "causal-block HC/QKV grouped generation overflow".to_owned())?;
        if post_attention_hidden != captured_post
            || output.grouped_submit_calls != 1
            || output.serial_token_forward_calls != 0
            || output.output_hidden.generation != expected_generation
            || ranges_overlap(captured_post, output.output_hidden)?
        {
            return Err("causal-block HC/QKV grouped output binding/dispatch 回执漂移".into());
        }
        let active = self.active_mut()?;
        active.next_layer = active
            .next_layer
            .checked_add(1)
            .ok_or_else(|| "causal-block HC/QKV layer counter overflow".to_owned())?;
        active.layer_phase = AdapterLayerPhase::AwaitingAttention {
            expected_input: Some(output.output_hidden),
        };
        Ok(())
    }

    fn seal_inner(&mut self, completed_layers: usize) -> Result<(), String> {
        let active = self.active()?;
        if completed_layers != FULL_DEPTH_LAYERS.len()
            || active.next_layer != completed_layers
            || !matches!(
                active.layer_phase,
                AdapterLayerPhase::AwaitingAttention {
                    expected_input: Some(_)
                }
            )
        {
            return Err("causal-block HC/QKV 禁止 seal 不完整或悬空 layer".into());
        }
        self.recorder.seal_and_drain(completed_layers)?;
        self.phase = AdapterPhase::LayersSealed;
        Ok(())
    }

    fn abort_inner(&mut self, completed_layers: usize) -> Result<(), String> {
        let (expected_completed, needs_drain) = match self.phase {
            AdapterPhase::Active(active) => (active.next_layer, true),
            AdapterPhase::LayersSealed => (FULL_DEPTH_LAYERS.len(), false),
            AdapterPhase::Poisoned {
                completed_layers,
                drained,
            } => (completed_layers, !drained),
            AdapterPhase::Idle => return Ok(()),
            AdapterPhase::Destroyed => return Err("causal-block HC/QKV adapter 已销毁".into()),
        };
        if needs_drain {
            self.recorder.drain_and_abort(expected_completed)?;
        }
        self.phase = AdapterPhase::Idle;
        if completed_layers != expected_completed {
            return Err(format!(
                "causal-block HC/QKV abort completed_layers 漂移: reported={completed_layers} expected={expected_completed}"
            ));
        }
        Ok(())
    }

    fn finish_inner(&mut self) -> Result<(), String> {
        if self.phase != AdapterPhase::LayersSealed {
            return Err("causal-block HC/QKV 没有已验收的 sealed block".into());
        }
        self.recorder.finish_validated_block()?;
        self.phase = AdapterPhase::Idle;
        Ok(())
    }

    fn destroy_inner(&mut self) -> Result<(), String> {
        if self.phase == AdapterPhase::Destroyed {
            return Ok(());
        }
        let completed_layers = match self.phase {
            AdapterPhase::Active(active) => Some(active.next_layer),
            AdapterPhase::LayersSealed => Some(FULL_DEPTH_LAYERS.len()),
            AdapterPhase::Poisoned {
                completed_layers, ..
            } => Some(completed_layers),
            AdapterPhase::Idle | AdapterPhase::Destroyed => None,
        };
        if let Some(completed_layers) = completed_layers {
            self.abort_inner(completed_layers)?;
        }
        self.recorder.destroy()?;
        self.phase = AdapterPhase::Destroyed;
        Ok(())
    }

    fn active(&self) -> Result<&AdapterActiveBlock, String> {
        match &self.phase {
            AdapterPhase::Active(active) => Ok(active),
            AdapterPhase::Poisoned { .. } => Err("causal-block HC/QKV adapter 已 poisoned".into()),
            _ => Err("causal-block HC/QKV adapter 当前没有 active block".into()),
        }
    }

    fn active_mut(&mut self) -> Result<&mut AdapterActiveBlock, String> {
        match &mut self.phase {
            AdapterPhase::Active(active) => Ok(active),
            AdapterPhase::Poisoned { .. } => Err("causal-block HC/QKV adapter 已 poisoned".into()),
            _ => Err("causal-block HC/QKV adapter 当前没有 active block".into()),
        }
    }

    fn poison(&mut self, error: String) -> String {
        let completed_layers = match self.phase {
            AdapterPhase::Active(active) => active.next_layer,
            AdapterPhase::Poisoned {
                completed_layers, ..
            } => {
                return format!(
                    "{error}; causal-block HC/QKV 已 poisoned, completed_layers={completed_layers}"
                )
            }
            _ => return error,
        };
        match self.recorder.drain_and_abort(completed_layers) {
            Ok(()) => {
                self.phase = AdapterPhase::Poisoned {
                    completed_layers,
                    drained: true,
                };
                format!(
                    "{error}; causal-block HC/QKV command owner 已立即 drain/abort, completed_layers={completed_layers}"
                )
            }
            Err(drain_error) => {
                self.phase = AdapterPhase::Poisoned {
                    completed_layers,
                    drained: false,
                };
                format!(
                    "{error}; causal-block HC/QKV drain/abort 失败并保持 poisoned: {drain_error}"
                )
            }
        }
    }
}

impl<R: S14CausalBlockHcQkvLayerRecorder> S14CausalBlockVulkanHcQkvAdapter
    for S14CausalBlockProductionHcQkvAdapter<R>
{
    fn begin_block(&mut self, base_position: u32, block_size: usize) -> Result<(), String> {
        self.begin_block_inner(base_position, block_size)
    }

    fn run_k_lane_hc_qkv_attention_router(
        &mut self,
        input: &S14CausalBlockLayerInput<'_>,
    ) -> Result<S14CausalBlockHcQkvRecordedLayer, String> {
        match self.run_layer_inner(input) {
            Ok(recorded) => Ok(recorded),
            Err(error) => Err(self.poison(error)),
        }
    }

    fn capture_grouped_moe_output(
        &mut self,
        post_attention_hidden: S14CausalBlockHiddenBinding,
        output: &S14CausalBlockGroupedMoeOutput,
    ) -> Result<(), String> {
        match self.capture_grouped_inner(post_attention_hidden, output) {
            Ok(()) => Ok(()),
            Err(error) => Err(self.poison(error)),
        }
    }

    fn seal_and_drain(&mut self, completed_layers: usize) -> Result<(), String> {
        match self.seal_inner(completed_layers) {
            Ok(()) => Ok(()),
            Err(error) => Err(self.poison(error)),
        }
    }

    fn drain_and_abort(&mut self, completed_layers: usize) -> Result<(), String> {
        self.abort_inner(completed_layers)
    }

    fn finish_validated_block(&mut self) -> Result<(), String> {
        self.finish_inner()
    }

    fn destroy(&mut self) -> Result<(), String> {
        self.destroy_inner()
    }
}

fn validate_hidden(
    binding: S14CausalBlockHiddenBinding,
    block_size: usize,
    label: &str,
) -> Result<(), String> {
    let expected_bytes = u64::try_from(
        block_size
            .checked_mul(S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE)
            .ok_or_else(|| format!("causal-block HC/QKV {label} elements overflow"))?,
    )
    .map_err(|_| format!("causal-block HC/QKV {label} bytes overflow"))?
    .checked_mul(BF16_BYTES)
    .ok_or_else(|| format!("causal-block HC/QKV {label} BF16 bytes overflow"))?;
    binding
        .offset
        .checked_add(binding.bytes)
        .ok_or_else(|| format!("causal-block HC/QKV {label} range overflow"))?;
    if binding.buffer == vk::Buffer::null()
        || binding.offset % 4 != 0
        || binding.bytes != expected_bytes
        || binding.block_size != block_size
    {
        return Err(format!(
            "causal-block HC/QKV {label} 不是精确 device [K,4,4096] BF16"
        ));
    }
    Ok(())
}

fn ranges_overlap(
    left: S14CausalBlockHiddenBinding,
    right: S14CausalBlockHiddenBinding,
) -> Result<bool, String> {
    let left_end = left
        .offset
        .checked_add(left.bytes)
        .ok_or_else(|| "causal-block HC/QKV left range overflow".to_owned())?;
    let right_end = right
        .offset
        .checked_add(right.bytes)
        .ok_or_else(|| "causal-block HC/QKV right range overflow".to_owned())?;
    Ok(left.buffer == right.buffer && left.offset < right_end && right.offset < left_end)
}

fn expected_layer(next_layer: usize) -> Result<u8, String> {
    FULL_DEPTH_LAYERS
        .get(next_layer)
        .copied()
        .ok_or_else(|| "causal-block HC/QKV layer index 越出43层".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use polaris_s14_runner::{router_kind_for_layer, RouteDecision, EXPERTS_PER_TOKEN};

    fn hidden(buffer: u64, block_size: usize, generation: u64) -> S14CausalBlockHiddenBinding {
        S14CausalBlockHiddenBinding {
            buffer: vk::Buffer::from_raw(buffer),
            offset: 0,
            bytes: (block_size * S14_CAUSAL_BLOCK_HC_ELEMENTS_PER_LANE * 2) as u64,
            block_size,
            generation,
        }
    }

    fn routes(layer: u8, block_size: usize) -> Vec<polaris_s14_runner::RouteDecision> {
        (0..block_size)
            .map(|lane| RouteDecision {
                layer,
                kind: router_kind_for_layer(layer).unwrap(),
                expert_ids: (0..EXPERTS_PER_TOKEN)
                    .map(|index| ((lane * EXPERTS_PER_TOKEN + index) % 256) as u16)
                    .collect(),
                weights: vec![0.25; EXPERTS_PER_TOKEN],
            })
            .collect()
    }

    #[derive(Debug)]
    struct FakeRecorder {
        output: S14CausalBlockHiddenBinding,
        integration_complete: bool,
        abort_calls: u32,
    }

    impl S14CausalBlockHcQkvLayerRecorder for FakeRecorder {
        fn begin_block(&mut self, _base_position: u32, _block_size: usize) -> Result<(), String> {
            Ok(())
        }

        fn record_k_lane_hc_qkv_attention_router(
            &mut self,
            input: &S14CausalBlockLayerInput<'_>,
        ) -> Result<S14CausalBlockHcQkvRecordedLayer, String> {
            let output = S14CausalBlockAttentionRouterOutput {
                post_attention_hidden: self.output,
                routes: routes(input.layer, input.input_token_ids.len()),
                forward_calls: 1,
            };
            Ok(S14CausalBlockHcQkvRecordedLayer {
                receipt: S14CausalBlockHcQkvLayerRecordingReceipt {
                    base_position: input.base_position,
                    layer: input.layer,
                    block_size: input.input_token_ids.len(),
                    input_hidden: input.input_hidden,
                    post_attention_hidden: output.post_attention_hidden,
                    layer_record_calls: 1,
                    command_graph_submit_calls: 1,
                    hc_qkv_projection_record_calls: 1,
                    attention_recording_calls: 1,
                    attention_output_post_record_calls: 1,
                    ffn_hc_router_input_record_calls: 1,
                    router_recording_calls: 1,
                    serial_token_forward_calls: 0,
                    hc_hidden_integration_complete: self.integration_complete,
                },
                output,
            })
        }

        fn seal_and_drain(&mut self, _completed_layers: usize) -> Result<(), String> {
            Ok(())
        }

        fn drain_and_abort(&mut self, _completed_layers: usize) -> Result<(), String> {
            self.abort_calls += 1;
            Ok(())
        }

        fn finish_validated_block(&mut self) -> Result<(), String> {
            Ok(())
        }

        fn destroy(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn k4_owner_closes_hidden_generation_and_rejects_false_hc_completion() {
        let block_size = 4;
        let layer = FULL_DEPTH_LAYERS[0];
        let tokens = [1, 2, 3, 4];
        let input_hidden = hidden(11, block_size, 0);
        let input = S14CausalBlockLayerInput {
            base_position: 1,
            layer,
            input_token_ids: &tokens,
            input_hidden,
            source: polaris_s14_runner::MaterializedTokenSource::SpeculativeDraft,
        };
        let mut adapter = S14CausalBlockProductionHcQkvAdapter::new(FakeRecorder {
            output: hidden(12, block_size, 1),
            integration_complete: true,
            abort_calls: 0,
        });
        adapter.begin_block(1, block_size).unwrap();
        let recorded = adapter.run_k_lane_hc_qkv_attention_router(&input).unwrap();
        recorded.receipt.validate(&input, &recorded.output).unwrap();
        adapter
            .capture_grouped_moe_output(
                recorded.output.post_attention_hidden,
                &S14CausalBlockGroupedMoeOutput {
                    output_hidden: hidden(11, block_size, 2),
                    grouped_submit_calls: 1,
                    serial_token_forward_calls: 0,
                    unique_experts: 24,
                },
            )
            .unwrap();
        adapter.drain_and_abort(1).unwrap();

        let mut invalid = S14CausalBlockProductionHcQkvAdapter::new(FakeRecorder {
            output: hidden(12, block_size, 1),
            integration_complete: false,
            abort_calls: 0,
        });
        invalid.begin_block(1, block_size).unwrap();
        let error = invalid
            .run_k_lane_hc_qkv_attention_router(&input)
            .unwrap_err();
        assert!(error.contains("未闭合一次 K-row"));
        assert!(error.contains("drain/abort"));
        invalid.drain_and_abort(0).unwrap();
    }
}
