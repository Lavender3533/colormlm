# FullDepth43 共享输入双投影 batch

日期：2026-08-02

## 结论

FullDepth43 的第一阶段投影边界合并已通过真实 RX 5700 XT 相邻 A/B。每层仍执行原生
packed-FP8 投影，但以下共享输入组合在一次 Python→arena→JSONL 请求内完成：

- Batch A：`wq_a + wkv`，覆盖全部43层；
- Batch B：`wq_b + indexer.wq_b`，只覆盖 `compress_ratio=4` 的21层。

两-token attention 请求由每 token `236` 降到 `172`，减少`27.12%`；投影本身仍为
`236/token`，没有删除模型计算或用近似结果冒充。

## 真实 L42 数值与常驻门

验证器：

```text
verify_dynamic_attention_shared_batch.py
```

同一持久 Rust/Vulkan worker 连续执行 Batch A、Batch B，再各重放一次：

- 8个输出全部命中冻结 L42 SHA；
- arena epoch严格为`0,1,2,3`，一个双投影 batch只推进一次；
- 重放的4个projection全部`gpu_slot_cache_hit=true`；
- 重放静态payload上传为`0 B`；
- 热Batch A约`4.04ms`，热Batch B约`8.44ms`。

该门只证明双投影协议、数值输出和slot复用，不单独证明完整token提速。

## 同路径相邻 A/B

两次运行使用相同Python入口、同一worker二进制、相同本地Range资产、相同两-token输入；
唯一变量是`--vulkan-attention-shared-batch`。

| 路径 | 两-token墙钟 | token 0 | token 1 | 输出 |
|---|---:|---:|---:|---|
| A：逐投影请求 | `65.712905s` | `41.540399s` | `22.292625s` | `[5,223]` |
| B：共享双投影 | `64.479194s` | `40.573774s` | `22.017747s` | `[5,223]` |

结果：

- 完整墙钟减少`1.233711s`，下降`1.8774%`；
- 有效吞吐提高`1.9133%`；
- token 0下降`2.3270%`，token 1下降`1.2330%`；
- attention计时总和下降`0.589039s`，下降`3.9083%`；
- 86/86层、472个attention projection、86次Vulkan MoE writeback全部完成；
- position 1仍为236/236 slot hit、静态payload上传`0 B`；
- CPU fallback为0，`error=null`。

完整机器可读结果见`FULLDEPTH43_SHARED_ATTENTION_BATCH_AB.json`。

## 实现边界

Batch不是完整attention融合。两个矩阵仍在Rust worker内依次dispatch；收益来自共享activation
写入、单次协议往返、统一epoch和减少Python验证边界。`wo_a→wo_b`之间不存在SwiGLU；官方路径是
`wo_a→BF16-carrying F32→group-128 E4M3FN activation quantize/dequantize→wo_b`，不能用相同的
共享输入batch伪装，下一阶段必须在worker内部建立包含官方中间重定量的链式图。

默认配置保持batch关闭；正式性能入口显式传入`--vulkan-attention-shared-batch`。这样同一路径
仍可做关闭对照和故障回滚，不把尚未稳定复测的实验开关静默变成全局行为。

## 下一速度硬门

实现worker内`wo_a→官方group-128 E4M3FN activation quantize/dequantize→wo_b`链式执行，目标是
继续减少每层一次跨进程边界和一次中间tensor往返。仍要求：

1. `[5,223]`短轨不变；
2. 43/43层×2、472个attention projection语义完整；
3. 86次MoE、零fallback、`error=null`；
4. 同路径开关式相邻A/B为正；
5. 不用局部kernel计时冒充完整token速度。
