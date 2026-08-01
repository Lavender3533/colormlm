# 北极星质量架构 v1：原生稀深旗舰

**日期：** 2026-08-01
**状态：** 唯一首选已预注册，物理预算通过，质量完全未验证
**边界：** 本轮未下载权重、未启停模型、未训练，也没有把工具/IR 编译器写成模型智力。

## 先说结论

“把巨型模型的末层、专家或路由器接给 v38”不是首选了。在不训练的前提下，最有可能同时保留旗舰坐标、进入 50--70GB，并把每 token 活跃量降到 3--5B 的路线是：

> **让 304B DeepSeek-V4-Flash-0731 自己成为北极星的原生坐标系，把未保留的 residual block 变为 identity，只执行预注册的 14 个深度锚点。**

这个候选称为 **Polaris Native Sparse-Depth S14（原生稀深旗舰）**。它不是让一个小模型模仿大模型，也不是用搜索、单测或 HTML 编译器假装智力；它使用 DeepSeek 自己的 tokenizer、embedding、mHC/attention、router、MoE 专家和 lm head，只是把深度执行变稀疏。

它仍然可能失败。S14 只保留 32.56% 的原层，最长连续跳过 8 层，完全可能损失大部分推理。本文的“唯一首选”意思是 **唯一先做的可证伪实验**，不是已经证明它有 Claude/GPT 质量。

## 1. 不训练能不能“搬运能力”的硬条件

把 donor 写成算子复合：

```text
F_D = Head ∘ f_42 ∘ ... ∘ f_1 ∘ f_0 ∘ Embed
```

一个中间切片 `f_b ∘ ... ∘ f_a` 不只需要一个同宽 hidden，而是需要 donor 在第 `a` 层的 **Markov 充分状态**：

```text
(hidden, KV/CSA/HCA/Delta state, mHC channels, position, tokenizer prefix)
```

仅有 hidden cosine 高不代表状态充分。若这个 portal 仍需运行 donor 的前缀才能产生，它就没有节省前缀计算；若用 v38 hidden 合成 portal，则又回到了未验证的跨模型桥。

权重空间的精确运输要求算子共轭：

```text
f'_l = S_(l+1) ∘ f_l ∘ S_l^(-1)
```

由于 RMSNorm、SiLU/门乘、attention、MoE top-k 和递归状态都是非线性的，`S` 不能是任意 Procrustes 旋转。真正保持函数的对称主要是：

- 全图一致的 hidden/head/neuron/expert 置换；
- up/down 或 Q/K 之间成对的可逆缩放；
- 确实产生近相同函数的专家簇商。

不带 donor-native activation 的二阶几何对齐不满足这些条件。

原生稀深之所以成为首选，是因为 residual block 天然已经写成：

```text
h_(l+1) = h_l + Δ_l(h_l, state_l)
```

对被跳过的层令 `Δ_l = 0`，就是完整的 identity；不需要新 hidden 宽度、词表映射或假 portal。这只解决“坐标不打架”，不保证跳层后质量还在。

对一段被跳过的 residual maps，粗略误差上界可写为：

```text
||F_full(h) - F_skip(h)||
  ≤ Σ_j ||Δ_j(h_j)|| · Π_(k>j) (1 + L_k)
```

`L_k` 是后续 block 的 Lipschitz 上界。我们没有这些值，因而不能用“residual 一般很小”来宣称能力保留；必须直接测完整任务。

## 2. 七条路线比较

| 路线 | 不训练 | 原生坐标 | 旗舰通用能力通路 | 全本地 20--50 tok/s | 结论 |
|---|---|---|---|---|---|
| 连续末端皮层 | 是 | 只有 portal 正确时 | 弱；末层主要解码上游特征 | 字节可行 | 降为探针，不当主脑 |
| 原生状态门户 | 精确 portal 时是 | 可精确 | 门户产生本身等价于跑 donor 前缀 | 否 | 只作研究 oracle |
| 结构 IR 候选/验证 | 是 | 无跨 hidden 问题 | 上限受候选覆盖率限制 | 是 | 保留为编程/UI 协议，不计通用智力 |
| 权重空间算子对齐/专家商 | 闭式时是 | 仅精确对称/函数冗余成立 | 有理论可能，尚无巨型 donor 证据 | 成功压缩后可 | 第二研究线，不先下整模 |
| 同宽巨型 MoE 联邦 | 是 | 只有同祖先/同 gauge 可信 | 比跨宽强 | 原生 active 过大 | 必须再稀深/专家商 |
| 按需远端/本地混合 | 是 | 远端整模正确 | 可接近远端旗舰 | 不是全本地 | 只作 trace 显微镜 |
| **原生稀深旗舰 S14** | **是** | **是** | **未知，但仍是 donor 自己的图** | **物理可能，待实测** | **唯一首选** |

### 2.1 连续末端皮层

DeepSeek L40--L42 比“一颗 expert”完整，但它们接收的是前 40 层已构造的高级表示。用 embedding 或 v38 hidden 直接跳到 L40，不会凭空重建前 40 层的规划和知识取回过程。它适合证伪 portal，不适合预先承诺旗舰能力。

### 2.2 原生状态门户

精确 portal 能让末端皮层继续原模型，但产生这个 portal 需要原模型前缀。用云端生成 portal 可用于研究，却不是最终全本地模型。闭式 CCA/Procrustes 只能作为候选桥，不是函数等价证明。

### 2.3 结构 IR 候选/验证

若正确候选没有被模型提出，验证器无法创造它：

```text
P(success) ≤ P(correct candidate proposed) × P(correctly selected | proposed)
```

它对代码、工具协议和 UI 非常有用，但不能用编译器吐出的 2,000 个 HTML token 宣称模型本身达到了 Claude/GPT 的通用推理与知识。

### 2.4 权重空间算子对齐/专家商

这是唯一有严格无训练压缩形式的备选：若一簇专家在 donor-native hidden 上满足 `E_e(h) ≈ E_c(h)`，则：

```text
Σ_e p_e E_e(h) ≈ (Σ_e p_e) E_c(h)
```

但必须先消除 neuron permutation 和 up/down inverse scaling 等精确对称，然后用 donor-native state 证明函数近似。不能根据 expert ID 或连续神经元编号分簇。

Qwen3.5-397B-A17B 是此路线最值得的负对照：它是 397B、60 层、4096 hidden、512 experts/top-10、expert width 1024，并与 Qwen3.6 共用 248,320 词表/算子族。但其专家库约 386.55B 参数，非专家骨架下界约 10.45B；即使删掉所有专家，也已经超出 3--5B active。它要进入 70GB，需要同时证明 3--4:1 的专家函数等价折叠和 4096→2048 原生可分子网；成功概率比保留原宽度的稀深更低。

### 2.5 同宽巨型 MoE

“shape 一样”不等于“gauge 一样”。当前没有 300B+ donor 与 v38 的 2048 hidden、DeltaNet/KV/mHC state ABI 同时相同。DeepSeek 和 Qwen3.5-397B 是 4096，K3 是 7168，GLM-5.2 是 6144。因此正确做法不是把它们塞进 v38，而是选一个巨型 donor 的原生宽度，再在它内部降低深度/专家活动量。

### 2.6 远端/本地混合

它可以很快做出接近远端旗舰的体验，但能力在远端整模，不在本地切片。正确用途是一次性采集 native hidden、route、NLL 和层影响，作为科研显微镜；不允许把 API 结果算为北极星本地能力。

## 3. S14 的图

预注册层集合：

```text
S14 = [0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42]
```

图结构：

```text
DeepSeek tokenizer + native embedding
  → L0 → L1 → L2
  → identity(L3..L5)
  → L6 → L7
  → identity(L8..L13)
  → L14 → L15
  → identity(L16..L21)
  → L22 → L23
  → identity(L24..L29)
  → L30 → L31
  → identity(L32..L39)
  → L40 → L41 → L42
  → native norm + native lm_head
```

选择规则在看到任何 S14 质量结果前已冻结：

- L0--L2 保留三层 hash routing 入口；
- 中部四对锚点同时保留 `compress_ratio=4/128` 的相邻结构；
- L40--L42 保留官方 DSpark 所读取的末端三层；
- 保留原 layer index、RoPE/compression、mHC、router、shared path 和该层专家 ID；
- 跳过层是纯 identity，不重复使用某层，不把 residual 乘以 gap，不搜 alpha；
- v38 hidden 不进入 S14。v38 以后只能当草稿加速器，不是质量权威。

## 4. 物理与数学预算

数字由 `analyze_quality_architecture.py` 根据官方固定 revision 的真实 shard 字节生成。

| 项目 | S14 |
|---|---:|
| donor 源 checkpoint | 304,180,418,494 参数 |
| 保留层 | 14 / 43（32.56%） |
| 完整本地文件 | **52,231,273,716 B（52.231GB / 48.644GiB）** |
| 活跃参数粗估 | **4.5897B** |
| 每 token 活跃权重字节上界 | **4.3795GB** |
| 每 token routed expert | 14×6 = 84 页 / 1.1230GB |
| 20 tok/s 权重扫描带宽 | **87.59GB/s** |
| 50 tok/s 权重扫描带宽 | **218.98GB/s** |
| RX 5700 XT 30% 实效带宽上限 | 30.69 tok/s |
| RX 5700 XT 50% 实效带宽上限 | 51.15 tok/s |

`4.3795GB/token` 将官方 shard 里的非专家字节全部保留，将 256 专家银行替换为官方 top-6 页，再计入原生 BF16 output head。因为还包含 safetensors header，它是轻微保守的存储字节上界，但不是实测 tok/s。

速度可能性分两层：

1. **GPU 带宽下界通过：** 20 tok/s 只要约 19.6% 理论显存带宽，50 tok/s 需要 48.9%。因此 20 有物理空间，50 是激进目标。
2. **分页命中仍然很难：** 按本机 207.39MiB/s 随机多 span 实测，20/50 tok/s 分别需要 **99.032% / 99.613%** expert-page 命中率；即使按 3,500MiB/s 理想顺序盘，也需要 83.66% / 93.46%。必须证明 14 层路由的热集合足够集中。

当前 runtime 尚未完成 DeepSeek V4 的 FP4/UE8M0、FP8 attention、mHC 和 CSA/HCA 原生图，所以“物理预算通过”不等于“本地已能跑”。

## 5. 最短可证伪实验

不进行层数、alpha、residual scale 或题目的事后搜索。只测上面冻结的 S14。

### Gate 0：零权重预算（已完成）

- 52.231GB 本地文件进入 50--70GB 预算；
- 活跃参数估计 4.59B；
- 30% 理论显存带宽的权重扫描上限为 30.69 tok/s；
- 结论只是 `physical_budget_pass=true, quality_pass=null`。

### Gate 1：4 题早停冒烟

在一次可用的 32GB NPU/更大设备上按层流式执行 S14，固定 greedy、每题最多 32 token，题型预先固定为：

1. 简短数学/因果推理；
2. 小函数修复；
3. 严格 JSON 工具参数；
4. 中文需求理解与简洁回答。

任一以下条件触发立即停止，不下第二个 donor：

- 4 题低于 3/4；
- 出现稳定重复环、大量乱码或基本指令失败；
- 同 prompt 重放不确定（greedy 路径应一致）；
- 原生 state/cache 无法在跳层图中保持。

### Gate 2：冻结 16 题八维质量门

只有 Gate 1 通过才跑一次现有冻结八维 16 题，不为 S14 改题或规则。继续条件：

- 至少 13/16；
- 相对正式 v38 至少净增 3 题；
- 没有任一能力维度归零；
- 代码、工具和结构任务的改善必须由单测/协议判分，不只是文本变了。

通过 Gate 2 只能说“稀深旗舰比 v38 更值得继续”，不能说已追上 Claude/GPT。

### Gate 3：本机真速度

- hot start，相邻 128 token，端到端不低于 20 tok/s；
- 记录真实 GPU/CPU/SSD 字节、expert hit、p95 miss stall 和 KV/state 内存；
- 8GB VRAM + 32GB RAM 不超限，不使用远端推理；
- 若低于 20 tok/s，不用 isolated kernel 或理论带宽冒充结果。

三门同时通过后，才能研究 50 tok/s。后续加速的正确语义是“草稿 + S14 原生验证”：草稿只减少 S14 的逐 token 次数，所有提交 token 仍必须通过 S14 target；不能用 v38 未验证文本替换 S14 质量。两者 tokenizer 不同，需要 byte-lattice speculative 契约，这是 Gate 3 之后的独立工程问题。

## 6. 失败后不许做什么

若 S14 在 Gate 1/2 失败：

- 不对 blind 结果扫描 13/15/17 层；
- 不扫 residual multiplier；
- 不把末三层再改名为“思考皮层”后重新宣称成功；
- 不用编译器输出或远端 API 掩盖模型质量失败；
- 不转去训一个 8B LoRA 并称为同一目标。

那时只剩两个诚实选项：

1. 进入 Qwen3.5-397B 单层的“算子可分性/专家商”低成本负对照，先证明 2048 宽原生子网存在；
2. 承认“无训练 + 全本地 + 3--5B active + 完整旗舰能力”目前没有物理载体，需要放宽“完全不训练”或“完全本地”其中一条。

## 7. 可复核资料

- [机器规格与候选冻结文件](architecture_spec.json)
- [离线物理预算分析器](analyze_quality_architecture.py)
- [生成的预算报告](budget_report.json)
- `fast16/research/polaris_meridian_v0/k3_audit/REPORT.md`
- `fast16/research/polaris_meridian_v0/runtime_paging/AUDIT.md`
- `fast16/research/polaris_meridian_v0/giant_organs/GIANT_ORGANS_AUDIT.md`
- [DeepSeek-V4-Flash-0731 官方仓](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [ShortGPT：无训练 block influence 剪层的相关工作](https://arxiv.org/abs/2403.03853)

ShortGPT 只证明 residual block 剪层是一类有先例的无训练研究问题，不证明 DeepSeek V4 S14 会保留旗舰能力。
