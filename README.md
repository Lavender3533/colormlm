# Polaris / 北极星

北极星（原 ColorLM）是一个面向消费级硬件的本地大模型架构研究项目。

目标不是把普通小模型包装成旗舰模型，而是研究：能否从数百 B 参数的开源 MoE 模型中，保留
原生 tokenizer、隐藏坐标、注意力、路由、专家、递归状态与输出头，通过稀深执行、专家分页、
GPU 热缓存和推测验证，在 **RX 5700 XT 8 GiB + 32 GiB RAM** 上得到兼顾质量与速度的本地模型。

> 项目仍处于研究阶段。尚未证明达到 Claude/GPT 的综合质量，也尚未达到 20--50 token/s。

## 当前主线：Polaris Native Sparse-Depth S14

S14 固定使用 `deepseek-ai/DeepSeek-V4-Flash-0731` 的原生模型坐标，只执行预注册的 14 个
residual block，其余层作为 identity：

```text
DeepSeek tokenizer + native embedding
  → L0 → L1 → L2
  → L6 → L7
  → L14 → L15
  → L22 → L23
  → L30 → L31
  → L40 → L41 → L42
  → native HC head + norm + BF16 lm head
```

每个保留层执行原生 attention、router、top-6 routed experts、shared expert 和完整 HC 状态。
L0--L2 使用 checkpoint 的 `tid2eid[token_id]` hash route；后续层使用原始
`(score + bias).topk(6)` 选择专家，并用未加 bias 的 score 计算权重。

静态预算：

| 项目 | 当前数值 |
|---|---:|
| DeepSeek donor | 304.18B 参数 |
| S14 本地完整切片 | 48.64 GiB |
| 每 token 粗估活跃参数 | 4.59B |
| 保留层 | 14 / 43 |
| routed experts | 14 × top-6 |

## 已验证的里程碑

### 1. 首个真实 S14 token

固定 BOS token 已通过全部 14 层、84 个真实 routed expert、14 个 shared expert 和真实全词表
输出头：

| 证据 | 结果 |
|---|---|
| 完成层 | 14 / 14 |
| argmax token | `108967`，解码为 ` Compression` |
| 首次精确 routed 下载 | `962,592,768 B` |
| 首次总耗时 | `714.769 s` |
| 热缓存复跑 | `57.979 s`，0 B 下载 |
| 确定性 | token、最终 state SHA、logits SHA 完全一致 |

仓库报告：
[`fast16/research/polaris_meridian_v1/s14_first_real_token/FIRST_TOKEN_REAL_REPORT.json`](fast16/research/polaris_meridian_v1/s14_first_real_token/FIRST_TOKEN_REAL_REPORT.json)

这证明真实图闭环，不证明语言质量。单个 `Compression` token 不能作为能力结果。

### 2. RX 5700 XT packed GPU parity

真实 L42/E126 已在 RX 5700 XT 上完成：

```text
w1 / w3 → limit-10 SwiGLU → w2 → route-weight mix
```

| 路径 | GPU dispatch | 对 CPU max abs | RMSE |
|---|---:|---:|---:|
| E126 最小专家整链 | 约 0.157 ms | `1.07e-6` | `1.48e-7` |
| 真实 FP8 `wq_a` | 约 0.084 ms | `5.96e-8` | `7.21e-9` |

证据：
[`scheduler/ssd_inference/evidence/s14_vulkan_numeric_rx5700xt.json`](scheduler/ssd_inference/evidence/s14_vulkan_numeric_rx5700xt.json)

孤立 kernel 时间不能换算为整模型 token/s。top-6、shared、attention、HC、缓存租约和状态提交
仍需进入同一 GPU 执行图。

### 3. Fail-closed 本地执行合同

- Rust Runner 只允许预注册 S14/top-6 与 FullDepth43/top-1 图。
- Range 状态机必须先得到真实路由，再允许读取恰好命中的专家页。
- Python executor JSONL 只传控制信息，hidden/state/logits 使用二进制 arena。
- SHA、dtype、shape、position、epoch、descriptor 或超时漂移会终止并 poison 会话。
- 合成 fixture 永远不能通过生产 capability gate，也不能冒充模型 token。

## 现在还缺什么

1. **连续 token：** position RoPE、window KV、HC、ratio4/128 compressor/indexer remainder
   必须跨 token 正确保存。
2. **完整 GPU 层：** top-6 routed + shared + attention + HC + norm/head 仍需合成一条常驻图。
3. **质量门：** 先跑冻结4题早停门，再跑八维16题；未过门不增加第二 donor。
4. **真实速度门：** 热启动相邻128 token，端到端至少20 token/s；不得用理论带宽或孤立 kernel
   冒充结果。
5. **旗舰验证：** 即使 S14 通过短门，也只能说明它比现有本地主干更值得继续，不能直接声称
   达到 Claude/GPT。

## v38 / v47 与 S14 的关系

- `v38` 是目前仍可使用的正式本地主干。
- `v47` 的 Parallel Genome Head、结构编译器和多 token 草稿研究可以作为专项器官或加速器。
- `S14` 是新的质量主脑候选，使用 DeepSeek 原生 4096 维坐标；v38 hidden 不直接注入 S14。
- 旧 Colab、FSQ 和 Block-8bit 文件保留为历史实验，不代表当前架构。

## 仓库导航

| 路径 | 内容 |
|---|---|
| [`PROJECT_STATE.md`](PROJECT_STATE.md) | 当前事实台账、失败对照和下一步顺序 |
| [`fast16/research/polaris_meridian_v1/`](fast16/research/polaris_meridian_v1/) | S14 架构、Range、数值参考与验证器 |
| [`scheduler/s14_runner/`](scheduler/s14_runner/) | capability-gated 原生执行状态机 |
| [`scheduler/ssd_inference/`](scheduler/ssd_inference/) | Vulkan kernel、显存页和专家执行实验 |
| [`fast16/research/v47_dual_tempo_bus/`](fast16/research/v47_dual_tempo_bus/) | v47 双节奏、草稿与能力头研究 |

## 最小离线自检

这些命令不下载模型权重：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.s14_first_real_token.selftest
cargo test --offline --manifest-path scheduler/s14_runner/Cargo.toml
cargo test --offline --manifest-path scheduler/ssd_inference/Cargo.toml --lib
```

真实权重不存放在 Git 仓库中。真实执行还需要固定 revision 的合法模型资产和本地
`D:/models/Polaris-S14` Range cache。

## 研究边界

- 不把“能加载、输出变化、单 token、合成 hidden、孤立 kernel”写成能力突破。
- 不用远端 API 输出冒充北极星本地能力。
- 不在看到 blind 结果后扫描层数、残差倍率或题目来挽救结论。
- 质量与速度必须分别由冻结任务和端到端实测证明。

完整且持续更新的证据以 [`PROJECT_STATE.md`](PROJECT_STATE.md) 为准。
