# FullDepth43 speculative runtime 审计与实现边界

## 结论

本轮已落地 `K=4/8` 的双 runtime 原子接受/回退控制器，但当前
FullDepth43 CPU/PyTorch 参考路径仍然是逐 token 循环。因此：

- 串行桥可以验证语义正确性和回退，**不能晋级为加速版**；
- 真正的加速入口已固定为 `BatchedVerifierRuntime.begin_causal_block()`；
- 只有后端在一次 causal-block forward 里生成 K 个对齐 target prediction，
  并能在首个 mismatch 处截断 KV/compressor state，才可进入真实速度 A/B。

## 现有资产审计

### v38/v47 草稿头

`parallel_draft_v47/frozen_contract.json` 把基座冻结为
`ColorLM-v38-Qwen36-Shared-Sequence-Policy`，vocab 是 `248320`。现在的 S14/FullDepth43
是 DeepSeek 原生 vocab `129280`。两者的 token ID 空间不同，不能把 v47 输出
直接喂给 FullDepth43，否则即使 ID 未越界也不表示同一字符串。

而且 `parallel_draft_v47/offline_gate_report.json` 当前明确是 `stop_no_cpp`：

- 没有真实 v38 完整轨迹数据；
- 没有自由滚动 acceptance 证据；
- 没有通过 1.08x 解析速度门。

所以本轮没有把 v47 草稿头接到新主线。真实可用的同 tokenizer
草稿源是 S14 自身。

### S14 连续状态

`s14_first_real_token/executor.py` 已有：

- `DecoderSnapshot`；
- `DecoderRuntime.run_token()` 私有 state clone；
- 所有层成功后单点替换 snapshot；
- 异常时不提交 token/KV/compressor remainder；
- 最大 1,048,576 position 的连续状态合同。

这些语义足以实现 `SnapshotTokenRuntime`。S14 生成 K 个 token 时保留
每一步 snapshot：

- 全接受时保留第 K 步；
- 在第 `j` 位 mismatch 时，保留第 `j+1` 步已处理的 KV，
  并把 pending token 从被拒草稿替换为 FullDepth fallback；
- 任何失败恢复轮次前 snapshot。

### FullDepth43

`fulldepth43_native_top6/executor.py` 仍在
`for _ in range(config.token_count)` 内逐 token 执行 43 层和 final head。它没有：

- `[K, ...]` causal block activation；
- 一次扫描非路由权重/head 并摊销到 K 个位置的内核；
- 块内 K 个 native top-6 route 的去重加载；
- 可按 mismatch position 截断的 KV/compressor block transaction。

所以 `SerialSnapshotVerifierBackend` 每块会报告 `forward_calls=K`，
`SpeculativeRoundResult.speed_eligible_verifier=false`。

## 已落地的控制契约

`runtime_controller.py` 实现：

1. 仅允许 K=4/8；
2. 草稿必须带齐 14 层每层 6 个原生 route；
3. target 必须带齐 `K x 43 x 6` 原生 route；
4. 只提交 target 相等的最长草稿前缀；
5. 首个 mismatch 提交 target 原生 fallback，丢弃后续草稿；
6. 草稿与 target 两边先 `prepare_commit`，再双提交；
7. 任一边的生成、验证、prepare、commit 或 context 对齐失败，双边回退；
8. 生产边界拒绝伪装成 batch 的串行响应：必须是
   `mode=batched_causal, forward_calls=1`。

## 速度门

`speed_gate.py` 用实际提交 token 数计算：

```text
native_baseline_time = committed_tokens * native_seconds_per_token
candidate_time       = draft_seconds + verifier_seconds
speedup              = native_baseline_time / candidate_time
```

通过必须同时满足：

- 每轮一次 `batched_causal` target forward；
- 真实模型时延，不是控制面 microbench 或静态投影；
- 同进程相邻 native A/B；
- 有效 speedup 至少 1.08x。

当前串行路径即使乐观地设定“草稿零开销 + K 个全接受”，仍是
K 次 target forward，静态 speedup 只有 1.0x，因此正确结果是
`stop_serial_verifier`。

## 运行轻量验证

```powershell
python -m unittest discover `
  -s fast16/research/polaris_meridian_v1/speculative_full_verifier/tests `
  -t . -v

python -m fast16.research.polaris_meridian_v1.speculative_full_verifier `
  static-speed-gate `
  --baseline-seconds-per-token 219.76 `
  --block-size 8
```

第二条应返回码 2 和 `stop_serial_verifier`，这是硬门拒绝串行假加速，
不是脚本故障。`219.76s` 只是已有单 token 暖回放的一次墙钟观测，
不是稳态均值；上述命令仅用它演示串行下界。

## 下一个唯一加速实现点

在 FullDepth Vulkan 路径实现 `BatchedVerifierRuntime.begin_causal_block()`：

1. 输入 base snapshot 和 4/8 个 draft token；
2. 一次 causal block 产生 K 个 target logits/argmax；
3. 同块中记录 Kx43 native routes，对 `(layer, expert)` 页去重；
4. 保留可裁切的 K 位 KV/compressor append delta；
5. `prepare_commit(decision)` 只选到 mismatch+fallback 的有效前缀；
6. `rollback()` 丢弃整个 append delta；
7. 先过现有原子性测试，再跑同进程相邻 A/B。

这个边界才会把 FullDepth 非路由权重与 head 扫描摊销给 K 个位置，
是当前能真正提高有效 token/s 的 speculative 主线。
