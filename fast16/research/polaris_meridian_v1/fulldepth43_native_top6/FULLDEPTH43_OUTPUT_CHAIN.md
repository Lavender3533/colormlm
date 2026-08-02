# FullDepth43 attention 输出链

日期：2026-08-02

## 结论

`wo_a → wo_b`现已作为一个严格 worker 请求执行。真实路径不是 SwiGLU，而是：

```text
grouped attention input
→ grouped wo_a
→ BF16-carrying F32
→ group-128 E4M3FN activation quantize/dequantize
→ wo_b
→ BF16 output
```

默认仍关闭；候选入口显式传入`--vulkan-attention-output-chain`。与已经晋级的共享输入 batch
同时开启后，每 token attention 请求从`172`降到`129`，减少`25%`；实际投影仍为`236/token`，
没有删层、裁剪矩阵或跳过重定量。

## 数值硬门

真实 L42 在 RX 5700 XT 上连续命中四个冻结边界：

| 边界 | SHA-256 |
|---|---|
| grouped `wo_a` input | `eee925360c8709263a0cdfa3986c2d3ee91a38c4e4589a7220064b489ad40060` |
| `wo_a` BF16 output | `2be0aa3b4b67aae58f62a77d2a255d6240b5baf3d71f37c9084fd890741d2eb9` |
| group-128 E4M3FN 重定量 | `94b3f7fd24ee36b8553ed513d1986ef49162c053bd6dbf62f98b9579e20ea3f0` |
| `wo_b` BF16 output | `84ce63ca9233b07bea99741f9982accac17bc65025b0098b7017acd7dab6db10` |

热重放两个 slot 均命中，静态 payload 上传为`0 B`。八轮局部 A/B 中，独立两请求中位数为
`12.9140 ms`，链式请求为`12.4473 ms`，下降`3.61%`。

初版逐元素遍历127个E4M3FN候选值，局部反而回归`51.7%`，因此没有直接晋级。现在改为对有序
E4M3FN有限值做二分查找，每元素约7次比较；Torch ties-to-even向量、Rust 34项测试和四个真实
SHA均继续通过。

## 两-token完整门

基线保留共享输入 batch、关闭 output-chain；候选只额外开启 output-chain。优化后候选结果：

| 路径 | 完整 execution | token 0 | token 1 | 输出 |
|---|---:|---:|---:|---|
| 基线 | `72.1430s` | `44.5625s` | `25.2211s` | `[5,223]` |
| 候选 | `63.9234s` | `40.1248s` | `21.9310s` | `[5,223]` |

观测完整墙钟下降`8.2196s / 11.39%`，有效吞吐提高`12.86%`。更贴近改动路径的 attention
exclusive 从`16.6325s`降到`14.7892s`，下降`1.8433s / 11.08%`。与此同时 Range 两阶段
也有约`2.63s`机器/缓存波动，因此不能把完整`11.39%`全部归因于 output-chain。

候选仍满足：86/86层、472个attention projection、86次Vulkan MoE writeback、零fallback、
`error=null`；position 1为236/236 slot hit且静态上传`0 B`。

机器可读证据：

- `FULLDEPTH43_OUTPUT_CHAIN_L42_GATE.json`
- `FULLDEPTH43_OUTPUT_CHAIN_AB.json`

## 下一速度主线

output-chain只减少边界，不能解决主要瓶颈。当前下一阶段转向深度1的“下一层 static-only
验证预取”：只预取下一层 non-expert/router，禁止预取 experts/shared/final；正式
`prepare_layer()`仍等待 Future 并保持 fail-closed。现有剖析中 Range读取、SHA与准备合计仍约
24秒/两 token，预取的现实目标是隐藏5--8秒，而不是继续堆零碎矩阵 batch。

本阶段不证明质量提升、可交互聊天或20--50 token/s。当前模型仍约20--30秒/token。
