# Polaris / 北极星

北极星（原 ColorLM）是一个面向消费级硬件的本地大模型架构研究项目。

目标不是把普通小模型包装成旗舰模型，而是研究：能否从数百 B 参数的开源 MoE 模型中，保留
原生 tokenizer、隐藏坐标、注意力、路由、专家、递归状态与输出头，通过稀深执行、专家分页、
GPU 热缓存和推测验证，在 **RX 5700 XT 8 GiB + 32 GiB RAM** 上得到兼顾质量与速度的本地模型。

> 项目仍处于研究阶段。尚未证明达到 Claude/GPT 的综合质量，也尚未达到 20--50 token/s。

## 当前主线：Polaris Exact Cascade

北极星现在采用“完整主脑裁决 + 多级草稿”的主线：

```text
S14 / v38 / v47 产生 K-token 草稿或结构候选
  → DeepSeek 官方 tokenizer 与消息协议
  → FullDepth43/native-top6 causal-block 验证
  → 标准接受/回退与完整状态提交
```

只有完整 43 层量化 DeepSeek-V4 验证过的 token 才能提交。S14 固定跳过 29/43 层，未经
跳层训练，不能因为保留了原生张量就被当作完整 V4 的质量等价物；它的正式定位改为原生坐标
草稿器、路由预取器和速度实验台。v38/v47 则分别保留为快速备用草稿与专项器官。

这条路线只能以当前量化 V4 为质量上限，尚不能直接证明达到 Claude/GPT；其价值是避免为了
速度让最终隐藏状态持续偏离完整主脑。

## 已运行的草稿骨架：Polaris Native Sparse-Depth S14

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

### 2. 连续两个真实 S14 token

同一个 `DecoderRuntime` 已保留 14 层 window KV、HC、ratio4/128 compressor/indexer remainder，
并完成第二个位置：

| 证据 | 结果 |
|---|---|
| position 0 | `0 → 108967`，` Compression` |
| position 1 | `108967 → 53`，`S` |
| 原子提交 | `committed_tokens=2`，14 层状态全部提交 |
| position 1 logits SHA | `46b95489427932a0d5acfacd5ee6bc9ceac495df3daed5a6a58681a0d95a141d` |
| 本次新增下载 | `1,016,078,336 B` |
| 相邻 top-6 直接复用 | `8 / 84`，约 `9.52%` |

仓库报告：
[`fast16/research/polaris_meridian_v1/s14_first_real_token/TWO_TOKEN_REAL_REPORT.json`](fast16/research/polaris_meridian_v1/s14_first_real_token/TWO_TOKEN_REAL_REPORT.json)

裸 BOS 产生的 `CompressionS` 只证明连续状态正确，不是聊天质量样本。9.52% 也只是一对 token
的观测，不能外推为稳态命中率；但它已经否决“只保留上一 token 六个专家就足够”的乐观假设。

### 3. RX 5700 XT packed GPU parity

真实 L42/E126 已在 RX 5700 XT 上完成：

```text
w1 / w3 → limit-10 SwiGLU → w2 → route-weight mix
```

| 路径 | GPU dispatch | 对 CPU max abs | RMSE |
|---|---:|---:|---:|
| E126 最小专家整链 | 约 0.157 ms | `1.07e-6` | `1.48e-7` |
| 真实 FP8 `wq_a` | 约 0.084 ms | `5.96e-8` | `7.21e-9` |
| L42 top-6 routed + shared | 约 1.370 ms | `1.10e-5` | `1.43e-6` |

证据：
[`scheduler/ssd_inference/evidence/s14_vulkan_numeric_rx5700xt.json`](scheduler/ssd_inference/evidence/s14_vulkan_numeric_rx5700xt.json)

top-6/shared 已带 generation-safe 租约、整批发布/回滚和 compute fence；但它仍是单层、
GPU-resident、F32 中间语义。孤立层时间不能换算为整模型 token/s，attention、HC、官方 BF16
边界、activation requantization、norm/head 和真实缺页仍需进入同一 GPU 执行图。

### 4. Fail-closed 本地执行合同

- Rust Runner 当前可运行入口只允许预注册 S14/top-6；历史 FullDepth43/top-1 reduction 保持
  hard reject，不能用于 Exact Cascade。新的 FullDepth43/native-top6 causal-block 合同正在迁移。
- Range 状态机必须先得到真实路由，再允许读取恰好命中的专家页。
- Python executor JSONL 只传控制信息，hidden/state/logits 使用二进制 arena。
- SHA、dtype、shape、position、epoch、descriptor 或超时漂移会终止并 poison 会话。
- 合成 fixture 永远不能通过生产 capability gate，也不能冒充模型 token。

## 现在还缺什么

1. **官方 prompt prefill：** 把官方消息编码得到的任意 token 序列送进同一状态运行时；当前裸
   BOS 连续两 token 不能用于质量门。
2. **FullDepth43 精确裁决：** 先证明 K=1 对固定参考逐 token 对齐，再证明 K=4/8 causal block
   与 K=1 输出及失败回滚等价。
3. **完整 GPU 图：** attention、HC、router、top-6、shared、BF16/requant、norm/head 仍需合成
   常驻图，并测量 SSD→RAM→VRAM 实际字节。
4. **质量门：** FullDepth 输出先跑冻结4题早停门，再跑八维16题；S14/v38/v47 只按草稿
   接受率和专项净收益晋级。
5. **真实速度门：** 热启动相邻128 token，端到端至少20 token/s；不得用理论带宽或孤立 kernel
   冒充结果。
6. **存储与缓存：** FullDepth43 完整量化资产约145.31 GiB；当前不能假设所有专家驻留，
   必须通过合法资产、分层缓存和可验证预取解决容量与命中率。

## v38 / v47 与 S14 的关系

- `v38` 是目前仍可使用的正式本地主干，也是 Exact Cascade 的低成本草稿候选。
- `v47` 的 Parallel Genome Head、结构编译器和多 token 草稿研究保留为网页/设计专项器官与加速器。
- `S14` 使用 DeepSeek 原生 4096 维坐标，作为最接近主脑坐标的草稿器和路由预取器。
- `FullDepth43` 是唯一允许最终提交 token 的质量裁决器；其真实运行与速度仍未完成。
- v38/v47 的 hidden 不直接注入 DeepSeek；它们输出文本或结构候选，重编码后再由主脑验证。
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
