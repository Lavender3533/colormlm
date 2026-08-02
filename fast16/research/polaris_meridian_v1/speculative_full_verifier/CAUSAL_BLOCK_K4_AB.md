# FullDepth43 causal-block K=4 同层回放门

日期：2026-08-02

## 结论

首个真实 K=4 同层 MoE 回放已完成。数值合同通过、页复用机制成立，但当前 GPU 执行方式显著回归，不能晋级为速度候选。

## 真实输入

- donor：`deepseek-ai/DeepSeek-V4-Flash-0731`
- profile：`FullDepth43/native-top6`
- GPU：AMD Radeon RX 5700 XT 8 GiB
- 连续位置：`0..3`
- forced-prefill 输入 token：`[0, 128803, 30594, 128804]`
- 原生 head 输出 token：`[5, 271, 303, 30594]`
- capture：四位置 × 43 层 = 172 个真实 manifest
- CPU fallback：0

四 token 真实执行耗时 `153.4774s`。首次零下载运行在新版 Vulkan 轨迹选择到尚未缓存的 routed expert 时按合同拒绝；随后只在显式 2 GB 上限和 HTTPS Range 下补页，正式四 token 轨迹完整闭合。

## 协议

Rust worker 新增实验 op `execute_causal_block_layer_replay`：

- 只接受 K=4/8；
- manifest 必须同层、position 连续、capture root 互异；
- 同名 payload 的 tensor/kind/expert/dtype/shape/bytes/path/SHA 必须完全一致；
- 所有唯一 payload 必须在首次 GPU compute 前完成批量 SHA 验证；
- routed expert 与 shared 按 GPU 身份在块内只上传一次；
- 四行 BF16 结果原子写入一个文件，逐行绑定 offset 与 SHA；
- `speed_eligible_verifier=false`，不得冒充完整 causal verifier。

Python 客户端按真实 Rust hello、请求字段、嵌套输出、合并文件 offset、manifest SHA、input token、expert IDs 和 BF16 finite 合同逐项复验；任一漂移会 poison 并终止 worker。

## 43 层 A/B

| 指标 | 四次原单层 | 一次 K=4 block | 结果 |
|---|---:|---:|---:|
| BF16 输出 | 172 行 | 172 行 | `172/172` 精确一致 |
| worker wall | `11,679.3328 ms` | `21,390.2778 ms` | `0.5460×`，回归 `83.15%` |
| GPU kernel | `521.85044 ms` | `591.2532 ms` | 回归 `13.30%` |
| GPU 上传 | `18,128,766,976 B` | `11,523,654,144 B` | 减少 `36.4344%` |
| routed 引用 | `1032` | `1032` | 其中 `251` 次块内复用 |
| 每层 shared 上传 | 4 次 | 1 次 | 总计 43 次 |

逐层明细见 `CAUSAL_BLOCK_K4_AB.json`。

## 归因与决策

K=4 的字节复用是真实的，但当前 block 使用通用 `GpuPayloadCache`，对每个唯一 expert/shared 分别建立 GPU buffer、上传与同步。旧单层路径使用固定有界复用槽，一次请求批量上传 42 个张量。当前实现因此出现“传输字节更少，但小上传、分配和同步更多”的反转。

本版立即停止晋级，不接入端到端解码。下一唯一速度候选是：

1. 保留同层 K=4 union 与 36.43% 字节削减；
2. 改为一个有界 union staging/device arena，一次批量上传；
3. 预先计算每个 expert/shared 的固定 offset，四个位置只切换 offset 与 route weight；
4. 不在每层重建/销毁 19--23 个 GPU buffer；
5. 先过 L42 精确门，再跑同一 43 层相邻 A/B。

只有新实现同时保持 `172/172` BF16 精确一致且 worker wall 快于四次单层，才允许继续迁移 attention/router/KV/HC。当前结果不改变正式最佳 FullDepth43 约 `0.03767 token/s`，也不构成质量、长上下文、Kimi 前端或 Claude/GPT 追赶证据。
