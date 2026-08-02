# FullDepth43 深度 1 静态页预取

日期：2026-08-02

## 结论

下一层 `non_expert + router` 静态页现可在当前层 routed expert 已成功到齐后异步准备，
并与当前层 TensorStore、MoE、HC 和 Vulkan 工作重叠。正式进入下一层时仍验证
`phase/layer/token`并消费 Future；后台任务不能推进 route-first 状态机。

显式入口是 `--range-static-prefetch`，默认关闭。开启后读取池与单线程协调池严格分离，
避免协调任务在同一个耗尽的读取池中等待自身；`experts/shared/final`永远不进入预取白名单。

## 两-token相邻 A/B

两条运行保留共享 attention batch、attention output-chain、43层 Vulkan MoE 和 Vulkan final
head，唯一变量是 `range_static_prefetch`。

| 路径 | 完整 execution | token 0 | token 1 | 输出 |
|---|---:|---:|---:|---|
| 基线 | `65.7962s` | `41.0222s` | `22.7511s` | `[5,223]` |
| 候选 | `61.8648s` | `37.4733s` | `22.2641s` | `[5,223]` |

完整墙钟节省 `3.9313s`，下降 `5.97%`，有效吞吐提高 `6.35%`。`range_prepare_layer`
从 `9.0043s` 降至 `0.1435s`，下降 `98.41%`。

候选产生 84 个预取事件，全部为 `consumed`：后台 fetch 累计 `9.6354s`，正式
`prepare_layer()`等待仅 `0.0005365s`，按事件计算约 `9.6349s`工作在正式等待前完成。
这不是 9.63 秒墙钟收益；同时读取会与当前专家消费和其他阶段争用主机/SSD，因此只以完整
execution 的 `3.93s`作为真实收益。

## 不变量

- 输出仍为 `[5,223]`。
- 86/86层完成，472个attention projection未裁剪。
- 86次Vulkan MoE writeback，CPU fallback为0。
- position 1为236/236 GPU slot hit，静态payload上传为0 B。
- 两侧 proof telemetry 完全一致：4487次完整SHA、1737次proof hit、
  `14,297,784,540 B`哈希、`8,174,608,604 B`避免重复哈希。
- Future失败不会进入`LAYER_BASE_READY`或提交token；清理异常不会覆盖真正的模型/Vulkan主错误。
- 协调池和Range池传成同一个对象会在调度前拒绝，并发token session也会拒绝。

## 验证

- FullDepth43 Python：`79 passed, 9 subtests passed`
- 静态预取合同：9项通过
- `py_compile`、UTF-8无BOM、`git diff --check`通过

机器可读证据：`FULLDEPTH43_STATIC_PREFETCH_AB.json`。

## 下一主线

预取只隐藏静态页等待，没有消除首次payload的“完整SHA读取一次、数值路径再次消费”双遍，
也没有减少每个新token约3.06GB的新expert页。下一步进入有界生命周期的 sealed payload：
在一次读取中同时完成SHA与当前层数值消费，摘要验证和文件身份检查完成前禁止发布buffer。

本阶段仍约30秒/token，只是速度里程碑，不构成可交互、20--50 token/s、质量提升、长上下文、
Kimi前端能力或Claude/GPT追赶完成的证据。
