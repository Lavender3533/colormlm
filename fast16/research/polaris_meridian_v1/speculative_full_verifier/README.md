# Polaris 全深度原生验证器（离线原型）

本目录实现一个可运行、可拒绝错误输入的调度与成本原型。默认
命令不加载权重；`runtime_controller.py` 可由真实 runtime bridge 注入模型步进：

- `S14/top6` 草稿后端必须提交 DeepSeek 原生 token ID，以及每 token 的冻结 14 层 top-6 route；
- `FullDepth43/native-top6` 后端必须通过**一次** causal-block 调用返回与 N 个草稿位置对齐的 greedy token，以及完整 `N x 43 x 6` 原生 route；
- 只提交与 target 一致的最长草稿前缀。首个不一致处提交 FullDepth43 预测作为 fallback，后续草稿和预测全部丢弃；
- 草稿全一致时只提交 N 个已验证 token，不偷加未纳入对齐合同的 bonus token。
- `K=4/8` 草稿/target 双 runtime 使用两阶段原子提交，任一失败双边恢复轮次前 snapshot；
- 串行 target 桥仅供正确性对接；只有一次 `batched_causal` target forward 才有加速资格。
- `s14_runtime_bridge.py` 已把现有 `DecoderRuntime` snapshot/token report 转成真实 S14 草稿边界，不做 token 映射。
- `fulldepth_runtime_bridge.py` 直接复用 FullDepth43 的 `DecoderState`/
  `LayerRuntimeState`，保留真实 window KV 与 compressor remainder；
- `cpu_causal_block.py` 提供一次 `begin_causal_block` 正确性 API，
  保存 `K×43×6` route 和 K 个完整 checkpoint，mismatch 只提交
  到 fallback 位置。它内部仍是 K 次 CPU token forward，会报告
  `mode=cpu_causal_block_reference, forward_calls=K`，不可通过速度门。

完整资产审计、原子状态语义和下一内核边界见
[`SPECULATIVE_RUNTIME_AUDIT.md`](SPECULATIVE_RUNTIME_AUDIT.md)。

## 运行

在仓库根目录执行：

```powershell
python -m fast16.research.polaris_meridian_v1.speculative_full_verifier analyze `
  --asset-root D:/models/Polaris-S14

python -m fast16.research.polaris_meridian_v1.speculative_full_verifier tokenize-draft `
  --asset-root D:/models/Polaris-S14 `
  --text "S14 后端生成的候选 continuation" `
  --block-size 8

python -m fast16.research.polaris_meridian_v1.speculative_full_verifier replay-cache `
  full_depth_routes.jsonl `
  --asset-root D:/models/Polaris-S14

python -m unittest discover `
  -s fast16/research/polaris_meridian_v1/speculative_full_verifier/tests `
  -t . -v

python -m fast16.research.polaris_meridian_v1.speculative_full_verifier `
  static-speed-gate --baseline-seconds-per-token 219.76 --block-size 8
```

`tokenize-draft` 只把调用者提供的 S14 候选文本变成恰好 N 个本地 DeepSeek token，不伪装成已运行 S14 权重。实际 S14 后端通过 `S14Top6Backend` 直接提交 token ID 和 route。

## 真实字节审计

`assets.py` 只读以下本地资产，并交叉拒绝任何漂移：

- `fulldepth_kadaptive_budget.json`；
- base shard `00001..00045` 的 45 个 safetensors header；
- `route_first_catalog.json` 内的 22,013 个 range（含最终 HC/norm/head 三段）；
- `tokenizer.json` 指纹和 `config.json` 的 129,280 vocab 合同。

已生成的 [analysis_report.json](analysis_report.json) 来自当前 `D:/models/Polaris-S14` 真实元数据：

- 45 个 shard，67,612 个张量，实际 shard 文件字节 `156,023,192,948 B = 145.307922 GiB`；
- 43 层非路由 `6,727,565,512 B`，BF16 head `1,059,061,760 B`；
- `43 x 256 = 11,008` 个专家页，每页 `13,369,344 B`；
- 118 GiB SSD 短缺 `27.307922 GiB`，不能无外部后备地自包含 FullDepth43 base；
- 在 SSD 先放非专家 payload 和 shard 容器开销后，最多放 8,814 个专家页（容量覆盖 80.0690%，**不等于 route 命中率**）；
- 8 GiB 放非路由+head 后理论剩余 803,307,320 B，只容纳 60 个专家页，还没扣 KV、activation、workspace 和 runtime。

## 20/50 tok/s 条件（不假设接受率）

成本模型使用 22.03 GB/s（十进制）PCIe，只给乐观的 I/O 吞吐上界。命中率指“本轮前已驻留设备，不产生 PCIe 流量”的**去重后专家页**比例。RAM/SSD 命中仍需经 PCIe 传入 GPU。

同一个 block 的专家页去重介于两个可证伪边界：所有 token 的 top-6 完全重合（每块 258 页），以及 token 间完全不重合（每块 `258 x block` 页）。下表的命中率范围就是这两个边界：

| 固定权重策略 | 目标 | 可证伪最低条件 |
|---|---:|---|
| `stream_each_block` | 20 tok/s | block 1/2/4 不可达；block 8 至少接受前缀 7 + 1 native fallback，命中 70.273%--96.284%；block 16 在同样最小接受长度下需 70.273%--98.142%，全接受时需 0%--82.175% |
| `stream_each_block` | 50 tok/s | block 1/2/4/8/16 全部不可达；即使 block 16 全接受且专家页 100% 驻留，固定扫描仍只给出 45.267 tok/s 上界 |
| `resident_after_warmup` | 20 tok/s | 在零设备命中+完美块内页重用时，block 4/8/16 至少接受前缀 3 + 1 fallback；若 token 间无页重用，全接受仍需 68.066% 设备命中 |
| `resident_after_warmup` | 50 tok/s | 在零设备命中+完美块内页重用时，block 8/16 至少接受前缀 7 + 1 fallback；若 token 间无页重用，全接受仍需 87.226% 设备命中 |

`resident_after_warmup` 只是 budget 字节层面的理论分支，不证明实际 8 GiB runtime 能固定这些权重。完整的每个接受长度/命中率前沿都在 `analysis_report.json`，没有选一个假定接受率当结果。

## 专家页回放

`route_trace.schema.json` 定义一行一个 causal block 的 JSONL 输入。回放器强制要求每个 token 都有 43 层、每层恰好 6 个不同的 `0..255` 专家 ID，并分开记录：

1. 原始 `token x layer x 6` 页引用；
2. 块内 `(layer, expert)` 去重；
3. 跨块 LRU 设备缓存命中/未命中；
4. 按真实每页 13,369,344 B 计算的专家 PCIe 字节。

## 真实 K=4 同层回放

2026-08-02 已使用四个连续真实位置、43 层 FullDepth43/native-top6 capture 完成首个
`execute_causal_block_layer_replay` 门：172/172 行 BF16 输出与原单层路径精确一致，
1032 次 routed 引用中有 251 次块内复用，GPU 上传字节下降 36.43%。

但当前通用 GPU identity cache 为每个唯一专家分别分配/上传，worker wall 为原四次单层的
`1.83×`（speedup `0.546×`），因此不晋级。完整证据见
[`CAUSAL_BLOCK_K4_AB.md`](CAUSAL_BLOCK_K4_AB.md) 与
[`CAUSAL_BLOCK_K4_AB.json`](CAUSAL_BLOCK_K4_AB.json)。下一实现必须改用有界 union arena
的一次批量上传，不能把 K 次串行或当前负收益 replay 包装成 batch 加速。

## 证据边界

当前已有真实 FullDepth43 route、权重 payload、四位置 capture 与同层 K=4 GPU 回放证据；
仍没有真实草稿接受 trace，也没有包含 attention/router/KV/HC/final head 的 causal-block
forward。因此不证明 20/50 tok/s 已达到，也不宣称质量达到 DeepSeek、Claude 或 GPT。

CPU causal-block 的离线 fixture 已覆盖 K=1 对一次串行、K=4
对串行 K 的整状态等价，以及 K=8 的完整 route/checkpoint 覆盖；
但这是状态机与真实 `DecoderState` 张量桥的正确性证据，不是大模型
数值或速度证据。下一步仍需把 `execute()` 内的单 token worker 提升为
可复用 callback，再用真实 K=1/串行 K 金标对齐。
