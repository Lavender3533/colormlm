# FullDepth43 可复用 Vulkan 上传槽 A/B

日期：2026-08-02

## 结论

固定、可复用的 Vulkan staging/device 上传槽通过真实 RX 5700 XT 门并晋级。默认路径仍然对
每层的 6 个 routed expert、1 个 shared expert 与输入激活做完整覆盖和上传，不保留上一层
payload 身份，也不启用已经被否决的 GPU resident LRU。

相同两-token门均输出 `[5, 223]`，每个 token 完整执行 43/43 层，CPU fallback 为 0，
`error=null`：

| 指标 | 原无复用槽基线 | 可复用槽候选 | 变化 |
|---|---:|---:|---:|
| 完整两-token时间 | 139.9328745 s | 117.9932255 s | -15.68% |
| 有效速度 | 0.0142926 token/s | 0.0169501 token/s | +18.59% |
| token 0 wall | 80.5465 s | 67.7718 s | -15.86% |
| token 1 wall | 58.5755 s | 49.5700 s | -15.38% |
| Python→Rust 86次调用 | 20.3746 s | 12.2578 s | -39.84% |
| worker non-kernel | 19.8029 s | 11.7180 s | -40.83% |
| GPU kernel合计 | 0.264412 s | 0.264932 s | 机器噪声范围 |

基线：
`D:/project/大模型ssd化/.tmp-polaris-runs/continuous-two-token-local-range-par3-20260802-035609/`

候选：
`D:/project/大模型ssd化/.tmp-polaris-runs/reusable-slot-ab-20260802-0415/`

## 数值与安全门

- 固化 L42 单层真实 capture 在 RX 5700 XT 上逐 BF16 位一致：4096/4096，
  `max_abs=0`、`rmse=0`。
- 两-token正式门 86/86 次响应都报告 `reusable_gpu_slot.enabled=true`。
- 每次响应都报告 resident GPU payload cache 为关闭，二者未混用。
- 固定逻辑 Vulkan 分配为 105,473,536 B，mapped staging 为 105,399,808 B，均受
  128 MiB 硬边界约束。
- route mix weight 仍逐请求绑定；槽只复用字节容器，不冻结路由或上一层内容。
- Rust定向测试 10/10、release离线编译、Python桥与profiler测试 12/12 通过。

## 声明边界

这一步消除了每 token 约 3600 次权重 buffer/memory 生命周期抖动，但仍保留每层约
100.5 MiB 上传，以及 descriptor、command buffer、query/fence 的逐请求创建。它把当前真实
速度推进到约 59 秒/token，仍远未达到可交互速度；后续应继续持久化 Vulkan 调度对象，并把
attention/router/compressor 与 FP8 materialization 移入长寿命 GPU 执行链。

## 后续负对照与下一主瓶颈

在固定 buffer 槽之上继续把35组 descriptor/binder改为启动时一次绑定，真实两-token结果仍为
`[5,223]`、43/43×2、零fallback，但仅从`117.9932s`降到`117.4504s`（约`0.46%`），
worker non-kernel仅从`11.7180s`降到`11.6146s`。该幅度不足以承担额外资源生命周期复杂度，
实现已撤回，不进入生产路径。

同次 profiler 显示当前最大独占耗时是387次CPU FP8 materialization，共`64.3112s`；其次才是
Range fetch `16.0991s`、Python→Rust Vulkan `12.2578s`和Range prepare `9.5231s`。下一主线
应让attention等投影直接消费packed FP8+UE8M0，避免在CPU展开整张F32矩阵。
