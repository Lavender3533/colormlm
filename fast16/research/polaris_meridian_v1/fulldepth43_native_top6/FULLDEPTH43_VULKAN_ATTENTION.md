# FullDepth43 通用 Vulkan attention 与全 slot cache 真实证据

日期：2026-08-02

## 结论

FullDepth43/native-top6 已把所有获批的 packed-FP8 attention 投影接入同一个持久
Rust/Vulkan worker，并在 RX 5700 XT 上完成两枚连续真实 token。完整运行覆盖 43 层、每枚 token
236 次 attention 投影，共 472 次；MoE 分支仍经 Vulkan 回写，最终 BF16 head 与 argmax 也由持久
Vulkan worker 执行。

与此前 attention 仍在 CPU/OpenBLAS 的两-token真实基线相比，总墙钟从
`93.780564 s` 降到 `63.885079 s`，减少 `31.9%`，即该次运行约 `1.468×`。输出 token
序列保持为 `[5, 223]`。第二枚 token 的 43 层 `elapsed_seconds` 之和为 `20.6784 s`；其
236 次 attention 请求全部命中 GPU slot cache，且每次 `payload_uploaded_bytes=0`。

这些结果证明通用 worker、动态 layer/projection、全 attention slot 常驻与连续 token 回放已经进入
真实执行链；它们不证明完整 GPU 模型、稳定 token/s、质量提升，尤其不证明 Vulkan 与
CPU/OpenBLAS 逐位等价。

## 证据文件

主要证据：

- `.tmp-polaris-runs/dynamic-attention-worker-all-slot-cache-gate.json`
  - SHA-256：`379cac1749be2070be19dabe6d08f2c06261ff70df08781ae54d2704c13bdebd`
- `.tmp-polaris-runs/fulldepth43-combined-vulkan-all-slot-cache-two-token.json`
  - SHA-256：`13ac69e0e131a03a82156362d8f14457b4654be445ecc57fd68d37f4da584d7a`

用于对照与负证据的真实报告：

- `.tmp-polaris-runs/fp8-validation-cache-ab-20260802-044300/model_report.json`
  - SHA-256：`17b22c04ba52a7f95f08ff926d5bc79cb6f82648fe7cd2cd36802c01eb0761cd`
  - attention CPU/OpenBLAS 两-token基线：`93.78056360000483 s`
  - 输出：`[5, 223]`
- `.tmp-polaris-runs/fulldepth43-attention-cpu-verify.json`
  - SHA-256：`53112c7898084ca86198f67d1bddd437482a1fd067bea7c52260437398bfa9f6`
  - Vulkan/OpenBLAS 在线数值负对照，停在 L0 `wq_b`

共同模型身份：

- repo：`deepseek-ai/DeepSeek-V4-Flash-0731`
- revision：`7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- profile：`fulldepth43_native_top6`
- FullDepth43 catalog SHA-256：
  `ca619984d4a46ad1a3701d2b4035766ea40c3a3dbedd3a474ce1df7aad4d0049`
- 下载授权：`false`
- 下载预算：`0 B`

## 通用 worker 独立门

`dynamic-attention-worker-all-slot-cache-gate.json` 先在 L42 对同一个 worker 连续发送 7 次请求。
前 6 次覆盖所有获批 projection，arena epoch 为 `0..5`：

| projection | 输出 SHA-256 |
|---|---|
| `layers.42.attn.wq_a` | `76469fd163f5db49de956eff9b29087afa4caa97d566be80bab9d9119facb0b8` |
| `layers.42.attn.wq_b` | `284391a5a45d6a5367060ecd444a21770e69fa7949455bea6823317f4fb43c04` |
| `layers.42.attn.wkv` | `3cc7f8f4264c6448dd32f9044c0d001107f06d57209a91a80fa56bdda59dd541` |
| `layers.42.attn.indexer.wq_b` | `d9adda7639665267be4fac36e2a74755bb5d730a4a2a8734695198fc4f331501` |
| `layers.42.attn.wo_a` | `2be0aa3b4b67aae58f62a77d2a255d6240b5baf3d71f37c9084fd890741d2eb9` |
| `layers.42.attn.wo_b` | `84ce63ca9233b07bea99741f9982accac17bc65025b0098b7017acd7dab6db10` |

第 7 次在 epoch `6` 重放 `wq_a`：

- `gpu_slot_cache_hit=true`
- `payload_uploaded_bytes=0`
- `gpu_slot_cache_entries=6`
- `gpu_slot_resident_bytes=115,350,400`
- 输出 SHA 仍为 `76469fd1...facb0b8`

因此这个独立门证明的是“同一 worker 能动态切换六类 L42 projection，并复用已经上传的 GPU
slot”；报告自身的 claim limit 也只允许外推到这一点，不能单凭它宣称 43 层或完整 token。

## 两-token完整执行

`fulldepth43-combined-vulkan-all-slot-cache-two-token.json` 的顶层状态为 `complete`：

- `native_token_executed=true`
- `fake_token_emitted=false`
- 43 层均提交状态，`runtime.next_position=2`
- `vulkan_writeback_layers` 共 86 项，`vulkan_writeback_fallbacks` 为 0
- `vulkan_attention_layers` 共 86 项
- `vulkan_attention_projection_count=472`
- attention worker：`persistent_process_dynamic_layer_projection`
- attention CPU fallback：`false`
- attention CPU verification：`false`
- activation quantization 仍为 CPU `e4m3fn quant/dequant`

提交序列为：

```text
position 0: input 0   -> output 5
position 1: input 5   -> output 223
```

即最终 `[5, 223]` 与 CPU-attention基线不变。这个“不变”只证明两枚 argmax token 在该输入轨迹上
没有翻转，不等价于所有 hidden、所有 logits 或所有 projection 逐位相同。

## 性能与 slot 常驻

两次真实运行的直接比较：

| 路径 | 两-token `execution_seconds` | 输出 |
|---|---:|---|
| CPU/OpenBLAS attention 基线 | `93.780564 s` | `[5, 223]` |
| 通用 Vulkan attention + 全 slot cache | `63.885079 s` | `[5, 223]` |

按报告原始值计算：

```text
(93.78056360000483 - 63.885079000006954) / 93.78056360000483
= 0.3187812426
= 31.8781% ≈ 31.9%
```

这是包含冷启动第一枚 token 的两-token墙钟，不是单 kernel 计时，也不是稳态吞吐承诺。两次运行
并非同一进程内的交错 A/B，因此 `31.9%` 应视为真实工程运行的方向性结果，而非已完成方差控制的
性能基准。

逐 token 汇总显示：

| token | 43层 `layer_seconds` | attention请求 | slot hit | payload上传 |
|---|---:|---:|---:|---:|
| position 0 | `36.1338 s` | 236 | 0 | `4,775,506,560 B` |
| position 1 | `20.6784 s` | 236 | **236/236** | **`0 B`** |

第一枚 token 填满 236 个唯一 attention slot；第二枚 token 对相同 projection 身份全部复用，因此
236 次均满足 `gpu_slot_cache_hit=true` 且 `payload_uploaded_bytes=0`。最终常驻统计为：

```text
gpu_slot_cache_entries = 236
gpu_slot_resident_bytes = 4,775,506,560 B
                        = 4.447537 GiB
```

独立复核后，缓存身份已固定为
`kernel + weight tensor/SHA + scale tensor/SHA`；仅scale变化也不能命中旧slot。worker同时在GPU
分配前执行`258`槽和`5 GiB`逻辑weight/scale常驻硬门，防止合法协议流无限增长。修复后的Rust
example测试为`25/25`，L42六投影加一次重放的真实短门再次通过。

`4.447537 GiB` 只表示 attention slot 的 packed weight/scale 常驻量，不是进程总显存；Vulkan
buffer、arena、MoE slot、最终头及驱动开销仍需另计。它也意味着这条“全常驻”路径依赖当前 8 GiB
GPU 的剩余显存包络，不能无条件复制到更小显存设备。

第二枚 token 的 attention 请求耗时之和为约 `3.5300 s`，而 43 层总和为 `20.6784 s`。剩余时间
仍分布在 CPU attention 非投影部分、HC、router、MoE准备/回写、张量转换及其他层级工作，不能把
slot hit 当成已经消除全部层开销。

## 数值边界：不能宣称 Vulkan/OpenBLAS 逐位等价

在开启在线 CPU verification 的负对照报告中，执行于首枚 token、L0
`layers.0.attn.wq_b` 时立即得到：

```text
max_abs = 6.103515625e-05
```

报告因此以 `status=blocked` 停止，未提交 token，错误为：

```text
L0 layers.0.attn.wq_b Vulkan/CPU BF16 不等价，max_abs=6.103515625e-05
```

这说明 frozen L42 fixture 的 SHA 闭合不能外推为所有层、所有在线 activation 下与 OpenBLAS 的
逐位等价。Vulkan shader 与 OpenBLAS 可能采用不同的浮点归约顺序；即使最终都做 BF16 RNE，某些
接近中点的元素仍可能落到相邻 BF16 值。

完整两-token报告明确记录 `cpu_verification=false`。所以 `[5, 223]` 不变是最终离散输出证据，
不是隐藏态逐位等价证据；文档与后续晋级不得写成“Vulkan attention 与 CPU reference exact”。

## 当前可声明与不可声明

可以声明：

- 单个持久 worker 已接受动态 layer、projection、position 与 activation。
- 标准 packed-FP8 与 grouped `wo_a` 均进入 43 层真实路径。
- 完整两-token运行执行了 472 次 Vulkan attention 投影且没有 attention CPU fallback。
- 第二枚 token 的 236 个 attention slot 全命中、payload 零上传。
- attention slot 常驻量为 `4.447537 GiB`。
- slot身份覆盖kernel、weight和scale，并有258槽/`5 GiB`常驻硬上限。
- 相对已记录 CPU-attention基线，两-token墙钟下降 `31.9%`，最终 `[5, 223]` 不变。

不可声明：

- Vulkan 与 OpenBLAS 所有层、所有元素逐位等价。
- 当前是 full-GPU token；HC、router、activation quantization 等仍在 CPU。
- `20.6784 s` 是完整第二 token 墙钟或稳定 tokens/s；它只是 43 个 layer 计时之和。
- 两-token结果证明模型质量、长上下文稳定性或 20/50 token/s。
- `4.447537 GiB` 是完整进程显存占用。

## 下一硬门

下一步应保留当前 `[5,223]` 与 slot hit 证据，同时把“性能路径”和“数值参考路径”分开：

1. 生产路径继续使用 Vulkan attention，但不得标注 exact/OpenBLAS-equivalent。
2. 参考验证需预先定义容差或逐 projection 的 BF16差异统计，不能在看到结果后放宽。
3. 对连续更多 token 记录 argmax、hidden误差传播、slot命中与显存峰值。
4. 继续定位第二 token `20.6784 s` 中 attention 之外的主要层级开销。
5. 只有完成相邻重复 A/B 与显存峰值采样后，才晋级为稳定性能结论。
