# FullDepth43 GPU payload cache 真实 A/B

日期：2026-08-02

## 固定条件

- 模型：DeepSeek-V4 FullDepth43/native-top6
- 硬件：RX 5700 XT 8GiB
- token：连续两 token，期望输出 `[5, 223]`
- 43/43 层、原生 top-6、fast production Vulkan
- 持久 GPU 最终词表头，CPU fallback 禁止
- 本地 Range cache 热态，不允许下载

## 结果

| 配置 | 状态 | 输出 | 总耗时 | GPU cache 命中/未命中 | 淘汰 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 无 GPU payload cache 基线 | complete | `[5, 223]` | 146.5560s | 不适用 | 不适用 | 保留 |
| 6GiB 朴素 LRU | blocked | `[5]` | 97.2220s（失败前） | 21/364 | 0 | OOM，否决 |
| 4GiB 朴素 LRU | complete | `[5, 223]` | 160.2900s | 0/602 | 317 | 慢约9.37%，否决 |

6GiB 在 position 1、layer 12 报 `A device memory allocation has failed`。失败前 GPU payload
resident 为 5,373,755,904 字节，另有 1,059,061,760 字节持久词表头以及 Vulkan 工作区和
桌面显存占用。

4GiB 虽然稳定，但一个 token 的顺序工作集已经超过容量。全局 LRU 在第二轮扫描时不断淘汰尚未
复用的后续层，最终形成 0 命中和 317 次淘汰，端到端反而回归。

## 决策

当前实现不得成为生产默认值。只有以下任一方案通过同口径真实 A/B 后才能重开：

1. 抗顺序扫描 resident set，容量满后不让一次性页面污染已证明可复用页面；
2. 按层生命周期保留 shared/高复用专家，当前请求的冷专家使用临时缓冲；
3. 删除逐层上传与文件边界，将 attention/router/compressor/MoE 合并进持久 GPU 执行图。

任何候选仍须满足：输出 token 不漂移、43/43 层、CPU fallback=0、无 OOM，并在相邻两 token
A/B 上获得明确端到端正收益。
