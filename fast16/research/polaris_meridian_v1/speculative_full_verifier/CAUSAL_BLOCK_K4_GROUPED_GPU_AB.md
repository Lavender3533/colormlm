# K=4 grouped GPU command graph：43 层精确与速度门

日期：2026-08-02

设备：AMD Radeon RX 5700 XT 8 GiB

状态：通过 MoE 组件门；不是端到端 token/s 晋级

## 单一变量

在上一版持久 union arena 基础上，把 K=4 的四次 compute submit、140 次 compute dispatch
收为一次 submit、9 次 grouped dispatch。输入、route slot、route weight、共享专家身份与
`0→5→shared` BF16-RNE 累加顺序保持不变。

本版 ragged shader 已把四行分支放进同一 command graph，但各 branch workgroup 仍分别解码
自己引用的权重；它还不是“相同 identity 只解码一次、同步服务四行”的最终实现。

## 结果

同一 release worker 内，每层按 `warm→measure` 执行一次，共 43 层、四个连续位置：

| 路径 | 43 层 wall | 43 层 GPU kernel | 相对上一版 |
|---|---:|---:|---:|
| 最初四次单层 | 8573.0042 ms | 538.16132 ms | — |
| 持久 union arena K=4 | 5262.9780 ms | 463.62472 ms | — |
| grouped GPU K=4 | 4391.0789 ms | 390.67568 ms | wall `1.1986×`，kernel `1.1867×` |

从最初四次单层到当前 grouped GPU，累计 wall 加速 `1.9524×`，GPU kernel 加速
`1.3775×`。逐层 wall 加速中位数为 `1.2601×`。

## 完整性门

- 43 层 × 4 行 = `172/172` BF16 字节精确一致。
- measure 阶段 staging allocations = `0`，device allocations = `0`。
- measure 阶段 host disk bytes = `0`。
- worker hello 明确声明 `causal_block_grouped_gpu_batch4=true`、dispatch 数为 `9`。
- `speed_eligible_verifier=false` 保持不变。

L39 的 measure wall 出现一次 `304.7563 ms` 主机抖动，使该层 wall 比上一版慢；但该层
GPU kernel 仍从 `9.78156 ms` 降到 `6.44012 ms`。本轮没有重跑或排除该层，以上总数包含
这次不利抖动。

## 决策边界

K=4 grouped GPU 路径可以进入后续 whole-token runtime；该结果只覆盖同层 MoE replay，
不包含 attention、router、KV、HC、final head，也不代表完整模型已达到可聊天速度或出现
新的质量、Kimi 前端、Claude/GPT 追赶证据。

机器可读原始数据见 `CAUSAL_BLOCK_K4_GROUPED_GPU_AB.json`。
