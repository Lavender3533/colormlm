# 北极星 FullDepth43 whole-token runtime

状态：正在组装 K=1 单进程 token；本目录已经冻结状态事务，并把 43 层真实 router 收进一个
Vulkan command graph。尚未完成 embedding→43层→head 的完整闭环。

## 已完成

1. `DecoderStateV1`
   - FullDepth43/native-top6 专用；
   - `commit_epoch + position + input_token + A/B fixed bank`；
   - 43 层必须按 L0→L42 全部完成，final token 完成后才允许唯一一次 commit；
   - L0、L42、final 后丢弃 candidate 都保持提交态逐字段不变。
2. 通用 BF16 Vulkan 投影
   - `BF16[N,K] × F32[B,K] → F32[B,N]`；
   - B 只允许 1/4/8 范围内的 1..=8；一次权重扫描同时计算所有 B；
   - 支持 N>65535 的二维 dispatch；
   - 支持 weight/input/output 三个持久 arena 的严格 offset 绑定。
3. 43 层真实 router 回放
   - 43 份固定 revision 的 `[256,4096]` BF16 权重，共 90,177,536 B；
   - 43 份真实 F32 capture，共 704,512 B；
   - 一个 VRAM weight arena、一个 input arena、一个 output arena；
   - 1 次 submit、43 次 dispatch；
   - GPU/CPU 43 层全部逐元素一致，global max abs error 为 0；
   - RX 5700 XT router kernel 合计 0.72492 ms，含首次 86 MiB 上传的 submit wall 为 5.5913 ms。

## 证据边界

当前 capture 是 MoE activation-quant 后的 F32 输入，不是 router 原始 RMSNorm/BF16 输入。
因此报告中的官方 expert ID 只允许作为观测：L3–L42 原始 logits top-6 与已有正式路由集合
27/40 相同，但缺少 bias、sqrt-softplus 且输入边界不同，不能据此宣称正式路由已经迁移完成。

权威文件：

- `router_replay_position3.json`：43 层路径、长度和 SHA-256 冻结清单；
- `router_replay_position3_report.json`：真实 GPU/CPU 数值与计时；
- `scheduler/ssd_inference/examples/s14_router_replay.rs`：真实回放 worker；
- `scheduler/s14_runner/src/whole_token.rs`：原子 DecoderState 合同。

## 复现

```powershell
python -X utf8 fast16/research/polaris_meridian_v1/whole_token_runtime/build_router_replay_manifest.py `
  --capture-root .tmp-polaris-runs/causal-block-k4-forced-fetch2-20260802/captures/position-000003 `
  --position 3 `
  --output fast16/research/polaris_meridian_v1/whole_token_runtime/router_replay_position3.json

cd scheduler
cargo run --offline -p ssd_inference --example s14_bf16_matvec_numeric
cargo run --offline -p ssd_inference --example s14_router_replay -- `
  ../fast16/research/polaris_meridian_v1/whole_token_runtime/router_replay_position3.json `
  ../fast16/research/polaris_meridian_v1/whole_token_runtime/router_replay_position3_report.json
```

## 下一唯一主线

把 embedding、HC/RMSNorm、attention/KV/compressor、router 后处理、已有 grouped MoE 与 final
head 绑定到同一个 `DecoderStateV1` candidate。先闭合 position 0 的 BOS→首 token；成功前不得
推进 head worker position 或 checkpoint，成功后一次原子 commit。
