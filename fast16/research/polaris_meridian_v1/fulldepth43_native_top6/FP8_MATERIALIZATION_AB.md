# FullDepth43 FP8 物化优化 A/B

日期：2026-08-02

## 结论

DeepSeek-V4 FullDepth43/native-top6 的 CPU FP8 E4M3 + UE8M0 物化路径完成两项同语义优化：

1. 用 `128 x 128` 分块 view 与原位广播替代两次 `np.repeat`，不再额外生成与 F32 权重同尺寸的 scale 矩阵。
2. 以绝对路径、文件大小和 `mtime_ns` 为进程内身份，缓存 FP8/UE8M0 NaN 编码扫描结果；底层文件变化时必须重新扫描。

两项合并后的真实连续两-token运行保持 `[5, 223]`、`43/43 x 2`、CPU fallback 为 0、`error=null`，相对可复用 Vulkan 槽基线耗时下降 `20.52%`。

## 固定口径

- 模型：`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- 图：`FullDepth43/native-top6`
- token 数：2
- 主机 FP8 cache：6 GiB
- GPU payload resident cache：关闭
- MoE：可复用有界 Vulkan 上传槽
- 最终词表头：持久 Vulkan BF16 head + device argmax

## A/B 结果

| 阶段 | 两-token耗时 | materialize_fp8 | 输出 | 正确性 |
|---|---:|---:|---|---|
| 固定 Vulkan 槽基线 | `117.9932255s` | `64.3112263s` | `[5, 223]` | 通过 |
| 仅分块原位广播 | `108.6967960s` | `47.9432s` | `[5, 223]` | 通过 |
| 分块广播 + 验证缓存 | `93.7805636s` | `41.5183913s` | `[5, 223]` | 通过 |

最后一档相对上一档：

- 总耗时：`-13.72%`
- `materialize_fp8`：`-13.40%`

相对固定槽基线累计：

- 总耗时：`-20.52%`
- 有效 TPS：约 `+25.82%`，达到 `0.0213264 token/s`
- `materialize_fp8`：`-35.44%`

相对最早连续两-token `146.5560s` 基线，累计缩短约 `36.01%`。这仍远低于可交互速度，不能外推为 Claude/GPT 质量或 `20--50 token/s`。

## 数值与失败门

- 真实 L42 `wq_b [32768,1024]` 微基准中，新旧广播结果逐位一致，`max_abs=0`、`rmse=0`。
- FullDepth43 测试：`45 passed, 2 subtests passed`。
- 测试覆盖整块广播逐位一致、非整块 fixture 旧语义，以及文件改变后缓存必须重新发现 FP8 NaN code。
- 正式两-token运行完成 86/86 次 Vulkan 层写回，两个 position 均为 43/43 层，两个 GPU worker 都报告 `cpu_fallback=false`。

本机原始证据目录：

```text
D:/project/大模型ssd化/.tmp-polaris-runs/
fp8-block-broadcast-ab-20260802-0500/
fp8-validation-cache-ab-20260802-044300/
```

## 下一步

CPU FP8 物化仍占 `41.52s / 两 token`，已经是明确的剩余大块。下一主线不再继续微调 NumPy，而是让 attention 投影直接消费 packed FP8 + UE8M0：先完成 L42 `wq_a` 的 exact Vulkan worker/arena 闭环，再扩展 `wkv/wq_b/indexer/wo_b`，最后实现 `wo_a` grouped BF16-weight 专用内核并推广至 43 层。
