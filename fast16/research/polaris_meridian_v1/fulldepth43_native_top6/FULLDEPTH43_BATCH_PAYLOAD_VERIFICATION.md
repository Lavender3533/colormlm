# FullDepth43 同层 payload 批验里程碑

## 结论

MoE route 确定后，Rust worker 现在可将当前层的 top-6 + shared 共 42 个 payload
最多分成 8 个并行任务读取并校验 SHA-256。只有全部成功后才原子发布到
`VerifiedPayloadCache`；任一页失败时，本批新页一个也不发布。之后保留原有顺序加载和
Vulkan MoE 数值路径，并要求 42/42 都是内存 hit、0 miss、0 新增读盘，才允许
GPU compute。

显式入口为：

```text
--vulkan-writeback-batch-verify-payloads
```

默认保持关闭，便于单变量 A/B 和回滚。

## 真实 RX 5700 XT 相邻 A/B

| 指标 | 基线 | 候选 | 变化 |
| --- | ---: | ---: | ---: |
| 完整两 token execution | 64.3844s | 53.0887s | -11.2957s / -17.54% |
| 有效吞吐 | 0.03106 token/s | 0.03767 token/s | +21.28% |
| position 0 | 40.0941s | 34.3976s | -5.6965s |
| position 1 | 22.0047s | 16.6947s | -5.3101s |
| Python→Rust Vulkan 边界 | 19.1428s | 9.4380s | -9.7048s |

为排除候选排在后面带来的 OS cache 偏置，候选后立即再跑一次关闭开关的反向基线。
反向基线为 58.6042s，相对同一候选的 53.0887s，候选仍节省 5.5155s / 9.41%，
吞吐仍提高 10.39%。因此批验不是单纯的运行顺序假象；正式结论保留正反两个
相邻观测，不用单个最优数字扩大宣称。

## 完整性门

- 三次运行输出均为 `[5, 223]`。
- 86/86 层、472/472 个 Attention 投影、86/86 次 Vulkan MoE、0 CPU fallback。
- 候选的 86 份 batch 回执全部是 42 entries、42 follow-up hits 和
  `all_verified_before_compute=true`。
- 总计 3612 个批验请求，其中 432 hit、3180 miss，Rust 并行完整读取并校验
  `7,593,067,008 B`。
- 84/84 个静态预取 Future 全部 consumed；position 1 仍为 236/236 Attention slot hit
  和 0 B 静态权重重传。
- Python RangeCache 与 Attention/MoE 回执现按 owner 独立闭合 count、bytes 及
  `(tensor, bytes, expected_sha256)` 的 mod 2^256 身份多重集和；闭合发生在
  token/checkpoint commit 之前。

机器可读证据见 `FULLDEPTH43_BATCH_PAYLOAD_VERIFICATION_AB.json`。当前最佳完整观测仍只有
`0.03767 token/s`，这是速度里程碑，不是 20–50 token/s、可交互聊天或 Claude/GPT 质量证据。
