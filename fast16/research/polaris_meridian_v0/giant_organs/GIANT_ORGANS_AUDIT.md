# 北极星 300B+ 巨型器官审计

**审计日期：** 2026-08-01
**范围：** DeepSeek-V4-Flash-0731 的专家路径与 DSpark；GLM-5.2 的专家路径、IndexShare 与 MTP。Kimi K3 只作交叉参照，详细字节审计见同级 `k3_audit/REPORT.md`。
**边界：** 未下载任何权重 payload，未训练，未启停模型/GPU，未运行 CMake，未修改 `PROJECT_STATE.md`。本报告只读取本地既有审计、官方配置/模型卡/推理源码、权重索引和 Hub 文件元数据。

## 结论

在这两个旗舰 checkpoint 中，**没有一块权重同时满足“独立可用、能把旗舰能力带入 v38、无需训练、还能在本机维持 20--50 tok/s”**。

这不是说它们不能切。两者的专家都能做 tensor Range 切片；真正的阻塞是能力所依赖的计算链：

- DeepSeek 的 L42 专家需要 DeepSeek 原生 4096 维隐藏态、score router、shared expert、mHC 残差语义和上下游层；DSpark 还显式读取 L40--L42 三处主干隐藏态，并复用 DeepSeek embedding/head。
- GLM 的 L77 专家需要 GLM 原生 6144 维隐藏态和路由；IndexShare 复用的是原生 DSA 的 top-k token 索引，不是可搬运的知识；MTP L78 需要 GLM embedding、末端隐藏态、输出头和几乎完整的一层 MoE。

因此三类结论必须分开：

| 类别 | 本轮结果 | 含义 |
|---|---|---|
| 独立可用 | **空集** | 没有可直接接到 v38、无需桥/训练而又保留旗舰能力的权重器官 |
| 必须保留原生路径 | DeepSeek L42 expert/MoE、DeepSeek DSpark、GLM L77 expert/MoE、GLM MTP | 可以物理切片，但必须在供体原生坐标、状态和输出契约中运行 |
| 仅架构借鉴 | DSpark 多深度草稿、IndexShare、GLM MTP 控制 | 可在北极星中重写同类机制；原权重不能直接迁移，且若要有效通常需要北极星原生轨迹校准/训练 |

### 对首个里程碑的直接建议

1. **不下载 DSpark 10.863 GB，也不下载 GLM MTP 的 26.807 GB 文件并集。** 它们不是独立“聪明头”。
2. 若主线需要验证巨型 donor 的真实 Range/分页 ABI，首个最小物理样本应是 **DeepSeek L42 单专家 + 对应 router 行**：逻辑净载荷 **13,377,540 B**。这只验证载荷、解码和槽位正确性，不进入能力晋级。
3. 第一个质量里程碑不能是“某专家输出非零”，而必须先取得供体原生 route trace 与隐藏态接口。用户当前不接受训练，因此 **4096/6144→2048 的跨模型能力接入保持阻塞**；不得用随机桥、同宽错觉或小 alpha 掩盖。
4. 20--50 tok/s 的现实路径仍是：v38 快主干保持当前约 26--29 tok/s，巨型器官只在低占空比、可验证的困难片段上调用，或先把巨型架构变成北极星原生的稀疏/推测机制。让 13B/40B active 的供体每 token 全程参与，不符合 RX 5700 XT 8 GiB + 32 GiB RAM 的带宽上限。

## 1. 审计口径

### 1.1 证据等级

- **[A]** 官方配置、模型卡、官方推理源码、官方权重索引或 Hub 文件大小。
- **[B]** 由 shape/dtype/官方配置可逐项复算的字节、MAC、cache 或文件并集。
- **[C]** 北极星本地既有实测，只能约束本机，不外推模型能力。
- **[U]** 尚无原生路由/隐藏态/端到端速度证据。

### 1.2 “可拆”不等于“可用”

一个候选只有同时满足下面四项，才可列入“独立可用”：

1. 输入接口能由 v38 当前状态直接提供，不依赖供体未运行的前置层；
2. 输出能在 v38 坐标或词表中解释，不需要未验证的矩形桥/token 映射；
3. 切片保留其原始功能所需的 router、norm、shared path、cache 与位置语义；
4. 在本机速度预算内，并有冻结任务上的净质量证据。

本轮所有权重候选最多满足“物理上能切”。

## 2. DeepSeek-V4-Flash-0731

### 2.1 官方结构快照

固定审计快照为 `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`（Hub API 于 2026-08-01 返回）。

| 项目 | 数值 | 证据 |
|---|---:|---|
| checkpoint 实计参数 | 304,180,418,494 | [A/B] Hub safetensors 元数据 |
| 核心模型口径 | 284B 总 / 13B active | [A] 官方模型卡 |
| 净权重 | 166,878,536,440 B | [A] `model.safetensors.index.json` |
| 主干 | 43 层，hidden 4096 | [A] config |
| MoE | 256 routed + 1 shared，top-6，intermediate 2048 | [A] config |
| 前 3 层 | hash routing | [A] `n_hash_layers=3` 与官方 `Gate` |
| 其余层 | `sqrtsoftplus` score routing | [A] config/源码 |
| 残差 | `hc_mult=4` 的 mHC | [A] config/源码 |
| 上下文 | 1,048,576 | [A] config |
| DSpark | 3 个 stage，目标 L40/L41/L42，block size 5，Markov rank 256 | [A] `inference/config.json` 与源码 |

顶层 `config.json` 仍写 `num_nextn_predict_layers=1`，但官方 `inference/config.json` 明确为 `n_mtp_layers=3`，权重索引也实际存在 `mtp.0..2`。切片必须以官方推理实现和实际索引为准。

### 2.2 L42 专家路径

官方 L42 分片 `model-00044-of-00048.safetensors` 为 **3,590,026,352 B**，整层 1,576 个 tensor 都在该片。

按官方 FP4 packed weight + UE8M0 scale 复算：

| 组成 | 字节 |
|---|---:|
| 单 routed expert 的 `w1/w2/w3 + scales` | 13,369,344 |
| 单 expert 对应 BF16 router 行 + FP32 bias | 8,196 |
| **最小 expert+route 身份页** | **13,377,540** |
| top-6 routed experts | 80,216,064 |
| shared expert（BF16） | 50,331,648 |
| 完整 router（BF16 weight + FP32 bias） | 2,098,176 |
| **router + 当前 top-6 + shared** | **132,645,888** |
| router + 256 experts + shared | 3,474,981,888 |

最后一个 132.646 MB 数字只是“已知 top-6 后的活动 FFN 权重”。它不是可预先固定的下载清单：L42 router 对每个 token 可能选择不同专家；没有供体原生路由轨迹时，任意挑六个专家等于猜测。

#### 为什么不能独立装到 v38

- v38 hidden 为 2048，DeepSeek 为 4096；本轮又明确不训练，因而没有可接受的双向矩形桥。
- L42 router 的含义建立在 DeepSeek 前 42 层形成的隐藏分布上。给它 v38 hidden（即使补零/重复/随机投影）不会保留专家身份。
- DeepSeek block 使用四路 mHC，专家残差的幅度和组合语义不是普通 `h + alpha * expert(h)`。
- FP4 expert 需要官方 per-32 scale 与相应 kernel；现有 Qwen Q4 路径不可把格式名当 ABI。
- 单个末层专家不携带 43 层训练形成的推理、agent 或长上下文能力。

**分类：必须保留原生路径。** 单专家页值得作为分页/量化 ABI 的 13.38 MB 工程样本，但不能叫能力器官。

### 2.3 DSpark 不是独立 draft model

三个 DSpark 文件为：

| stage | 文件 | 文件字节 | 索引 tensor 数 |
|---:|---|---:|---:|
| 0 | `model-00046-of-00048.safetensors` | 3,610,455,184 | 1,568 |
| 1 | `model-00047-of-00048.safetensors` | 3,560,111,960 | 1,565 |
| 2 | `model-00048-of-00048.safetensors` | 3,692,775,244 | 1,572 |
|  | **并集** | **10,863,342,388** | **4,705** |

官方源码给出的依赖链是：

```text
DeepSeek 原生 L40/L41/L42 hidden
    -> stage0 main_proj(concat 3 x 4096 -> 4096) + main_norm
    -> DSpark stage0 -> stage1 -> stage2
    -> 复用主模型 embedding/head
    -> Markov head + confidence head
    -> target model 验证接受
```

每个 stage 本身又包含 mHC、attention、MoE router、shared expert 和 256 专家银行；因此 10.863 GB 并不是一个“小预测头”。官方 vLLM/SGLang 也明确把 DSpark 作为**同 checkpoint 的 speculative decoding 模块**，不是单独的 draft 模型路径。

DSpark 的关键不可省依赖：

1. L40--L42 三处 4096 维原生主干隐藏态；
2. DeepSeek tokenizer、129,280 词表、embedding 和不共享的输出 head；
3. 三个 stage 的 KV/cache、mHC state、noise token 和 block 生命周期；
4. 主模型逐 token 验证，否则 speculative token 不能提交。

**分类：权重为“必须保留原生路径”；多深度草稿/置信提交机制为“仅架构借鉴”。** 对北极星可借鉴的不是这些权重，而是“三处深度状态→小草稿→原主干无损验证”的拓扑。若不做任何校准/训练，则只能先研究确定性 n-gram/原生 head 草稿，不能宣称移植 DSpark。

### 2.4 DeepSeek 与 20--50 tok/s

- 官方完整部署示例是单节点 4×GB300；官方未声称该 checkpoint 可在 8 GiB GPU 上运行。[A]
- 13B active 即便理想压到 4-bit也约 6.5 GB/token 的活动权重扫描量；还未计 shared/attention/mHC/cache。若从 32 GB RAM/PCIe 每 token 搬运，20 tok/s 需要约 130 GB/s 有效传输，已经越过本机外存链路。[B]
- 仅一个常驻 L42 top-6+shared FFN（132.646 MB）在孤立 kernel 层面可能很快，但 v38 端到端质量没有原生输入，速度可行不等于功能可行。[U]
- DSpark 三 stage 额外 10.863 GB，且主模型仍必须运行；在本机不能作为把 v38 从约 28 tok/s 推到 20--50 tok/s 的即插即用加速器。[B/C]

## 3. GLM-5.2

### 3.1 官方结构快照

固定审计快照为 `zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9`。

| 项目 | 数值 | 证据 |
|---|---:|---|
| 总/active 参数 | 744B / 40B | [A] 官方模型卡 |
| BF16 checkpoint 净权重 | 1,506,659,919,872 B | [A] index |
| 主干 | 78 层，hidden 6144 | [A] config |
| MoE | 前 3 层 dense；其后 256 routed + 1 shared，top-8，intermediate 2048 | [A] config |
| DSA | 32 index heads × 128 dim，top-2048 | [A] config |
| IndexShare | 21 个 `full` indexer 层，57 个 `shared` 层 | [A] `indexer_types` |
| MTP | 1 层（权重命名为 L78） | [A] config/index |
| 上下文 | 1,048,576 | [A] config |

### 3.2 L77 专家路径

L77 分布在四个官方 shard：

```text
model-00267  5,366,406,960 B
model-00268  5,360,347,320 B
model-00269  5,360,347,264 B
model-00270  5,366,430,968 B
文件并集       21,453,532,512 B
```

按 BF16 几何体积：

| 组成 | 字节 |
|---|---:|
| 单 routed expert：3 × 6144 × 2048 × 2 | 75,497,472 |
| 单 expert 对应 router 行 + FP32 bias | 12,292 |
| **最小 expert+route 身份页** | **75,509,764** |
| top-8 routed experts | 603,979,776 |
| shared expert | 75,497,472 |
| 完整 router + correction bias | 3,146,752 |
| **router + top-8 + shared** | **682,624,000** |
| 256 routed experts 银行 | 19,327,352,832 |

单专家在物理上比 DeepSeek 大约 5.65 倍；即使之后合法量化到约 4-bit，top-8+shared 也仍是约 170 MB 量级，并且 6144→2048 桥和原生路由问题完全没有消失。

**分类：必须保留原生路径。** 不建议把 GLM 单专家作为第一颗物理测试页：字节更大、坐标更远，且没有比 DeepSeek L42 更低风险的 runtime 信息增益。

### 3.3 IndexShare：值得借机制，不值得拆权重

Transformers 官方实现证明，GLM-5.2 的 IndexShare 做的是：

1. `full` 层用独立 indexer 根据该层 hidden、`q_resid` 与 indexer key cache 选 top-2048 token；
2. 随后的 `shared` 层不创建 indexer，直接复用前一个 full 层的 `topk_indices`；
3. 这四层仍各自运行自己的 MLA 投影、KV cache、attention、MoE 和残差。

一个 full indexer 的权重几何体积约为：

| tensor | BF16 字节 |
|---|---:|
| `wq_b`: 2048→4096 | 16,777,216 |
| `wk`: 6144→128 | 1,572,864 |
| `weights_proj`: 6144→32 | 393,216 |
| `k_norm` weight+bias | 512 |
| **合计** | **18,743,808** |

官方实现要求 `indexer.weights_proj` 运行时保留 FP32，因此运行时约 **19,137,024 B**。此外，每个 full indexer 的 BF16 key cache 在 1M context 下为 `1,048,576 × 128 × 2 = 268,435,456 B`；21 个 full indexer 理论合计 **5,637,144,576 B/sequence**，还不包括各层 MLA KV/cache。

官方模型卡称 IndexShare 在 1M context 将 per-token FLOPs 降低 2.9×。这说明它是长上下文 attention 的结构优化，不是短上下文权重带宽或通用智力模块。v38 当前是 Qwen3.6 混合 DeltaNet/attention 图，没有 GLM DSA index cache；直接搬 18.7 MB indexer 权重既不能选择 v38 的关键 token，也不能给 v38 增加 GLM 能力。

**分类：仅架构借鉴。** 可借的是“一个稳定索引决策跨连续 4 层复用”的协议，并在北极星自己的 attention/检索总线上验证；GLM indexer 权重本身不下载。

### 3.4 GLM MTP：几乎完整的一层，不是小头

GLM 的 MTP 权重命名为 `model.layers.78.*`，包含：

- `eh_proj`、`enorm`、`hnorm`、`input_layernorm`；
- 一套 256 routed experts、shared expert 与 router；
- 该预测层所需的其余 block 权重。

它横跨 `model-00270..00274-of-00282.safetensors`，标准文件级下载并集为：

| 文件集合 | 字节 |
|---|---:|
| 00270--00274 | **26,807,470,488** |
| 其中仅 256 experts 几何体积 | **19,327,352,832** |

00270 同时含 L77 与 L78 的起始张量，所以“只下 MTP 文件”仍会夹带别层数据；只有固定 revision 后读取 safetensors header 并按 tensor Range 才能避免。

官方只声称改进后的 MTP **acceptance length 最多提升 20%**，没有声称端到端吞吐提升 20%。在 CPU/GPU 混合、本就权重带宽受限的 8 GiB 机器上，多跑一层 6144-hidden top-8 MoE 很可能先增加延迟；只有真实接受长度覆盖额外计算后才会加速。

**分类：权重为“必须保留原生路径”；MTP 迭代复用 IndexShare 和 target 验证为“仅架构借鉴”。** 不下载 L78。

### 3.5 GLM 与 20--50 tok/s

40B active 的原生路径即使理想 4-bit也约 20 GB/token 权重扫描量；本机无法让它每 token 全程参与并保持 20 tok/s。[B]

IndexShare 可能在很长上下文降低 attention FLOPs，但不会消除 78 层 top-8 MoE 的权重扫描；在短/中上下文，v38 当前约 26--29 tok/s 更主要受混合 CPU/GPU 权重路径影响。[A/C]

GLM MTP 只有在同一 GLM 主干中、有足够接受率且 kernel 优化到位时才可能净加速；跨到 v38 的词表/hidden/状态均不兼容。[U]

## 4. 巨型模型横向判断

### 4.1 模块 Range 边界与 2048 坐标兼容矩阵

下表回答“能不能从仓库里单独切出来”。`可 Range` 只指 tensor/连续字节边界，不代表能力可移植。单个 attention head 通常融合在大投影矩阵的某一轴上；可以先 Range 取整张 tensor 再离线切行，但它不是自包含 attention 器官。

| 供体 / 模块 | 物理切片 | 模块输入/输出宽 | 对本地 2048 坐标 | 能力可移植性 |
|---|---|---:|---|---|
| DeepSeek embedding | 独立 tensor；1,059,061,760 B | token→4096 | **不兼容**；vocab 129,280 也不同 | 只提供 DeepSeek token 坐标，不带推理能力 |
| DeepSeek lm head | 独立 tensor；1,059,061,760 B | 4096→129,280 | **不兼容** | 不共享 embedding；不能按行接到 248,320 词表 |
| DeepSeek 单 routed expert | 可 Range；13,369,344 B | 4096→2048→4096 | **不兼容** | 必须保留原生 Lx hidden/router/mHC；单页仅适合 loader |
| DeepSeek shared expert | 可 Range；50,331,648 B/层 | 4096→2048→4096 | **不兼容** | 是每层固定 FFN 支路，不是独立通用知识模块 |
| DeepSeek router | 可 Range；2,098,176 B/层 | 4096→256 | **不兼容** | expert ID 只在同层同 checkpoint 有意义 |
| DeepSeek attention/层 | attention tensor 可分别 Range；L42 整层文件 3,590,026,352 B | 4096；CSA/HCA + mHC | **不兼容** | 必须保留 compressor/indexer/KV/mHC 与原生前置层；单 head 不自包含 |
| Kimi K3 MoonViT-V2 | **独立文件/命名空间**；802,428,928 B 净载荷 | 图像→1024 patch hidden | 语言 2048 不直接兼容，但视觉特征边界清楚 | **唯一接近独立器官的模块**：可独立做视觉编码；不包含网页代码/动作策略 |
| Kimi K3 mm projector | 独立文件；92,289,024 B 净载荷 | 4096→4096→7168 | **不兼容** | 原 projector 只通往 K3 7168 文本空间；北极星需新的 1024→2048 接口才可消费视觉塔 |
| Kimi K3 embedding/head | 独立 tensor；各 2,348,810,240 B | 163,840↔7168 | **不兼容** | 词表与 hidden 双重不兼容 |
| Kimi K3 单 routed expert | 连续 Range 页；17,547,264 B | latent 3584→3072→3584 | **不兼容** | 还依赖 7168↔3584 投影、同层 router/shared/attention；不能单独给前端能力 |
| Kimi K3 shared experts | 可 Range；2 个合计 264,241,152 B/层 | 7168 FFN path | **不兼容** | 必须留在 K3 层路径 |
| Kimi K3 router | 可 Range；12,848,640 B/层 | 7168→896 | **不兼容** | 无原生 hidden 时 route 无意义 |
| Kimi K3 KDA/MLA attention | 张量可 Range，单层模块可列 manifest | 7168 | **不兼容** | KDA state/MLA KV/AttnRes 使单 head/单层非独立；完整非专家骨架已 113.51 GB |
| GLM embedding | 独立 tensor；约 1,903,165,440 B BF16 | token→6144 | **不兼容**；vocab 154,880 | 只定义 GLM 输入坐标 |
| GLM lm head | 独立 tensor；约 1,903,165,440 B BF16 | 6144→154,880 | **不兼容** | 不能直接合入本地输出空间 |
| GLM 单 routed/shared expert | 可 Range；各 75,497,472 B BF16 | 6144→2048→6144 | **不兼容** | 必须保留原生层 hidden/router；shared 也不是独立能力 |
| GLM router | 可 Range；3,146,752 B/层 | 6144→256 | **不兼容** | expert ID 为层局部语义 |
| GLM IndexShare indexer | tensor 可 Range；约 18,743,808 B BF16/full 层 | hidden 6144 + q_resid 2048→top-k token IDs | **不兼容** | top-k 只适用于对应 GLM DSA/MLA cache；机制可借，权重不可移植 |
| GLM attention/层 | attention tensor 可 Range；L77 文件并集 21,453,532,512 B | 6144；MLA/DSA | **不兼容** | shared 层仍需各自 MLA/KV/MoE；单 head 非自包含 |
| MiniMax-M3 视觉/专家 | 索引可定位；expert 约 57--113 MB，视觉张量可 Range | 文本 6144，视觉 1280→6144 | **不兼容** | 自定义许可先审；视觉只给感知，MSA/动作策略仍在文本主干 |
| MiMo-V2.5-Pro expert/MTP | 单 expert/层约 38 MB Range；MTP 独立文件 2,463,641,280 B | 文本 6144 | **不兼容** | MTP 只加速原生 MiMo；SWA/GA 机制可借，权重不进首批 |

这张表中只有 **K3 视觉塔本体**在功能上可作为独立编码器运行；但它输出 1024 维视觉 patch，K3 原 projector 又指向 7168，仍不能“零训练、零接口工作”直接喂给 v38。其余 embedding/head、attention、shared/routed expert、router 均是**模块可切、能力不可单独搬**。

| 巨型供体 | 真正可物理拆的东西 | 能否直接给 v38 加智力 | 对 20--50 tok/s 的价值 |
|---|---|---|---|
| DeepSeek-V4-Flash-0731 | 13.37 MB 单 expert、3.59 GB 单层、10.86 GB DSpark | 否；4096/mHC/router/词表/原生 hidden 阻塞 | 单页适合验证分页 ABI；DSpark 只对原生 DeepSeek 有加速意义 |
| GLM-5.2 | 75.50 MB 单 expert、18.74 MB indexer、L78 MTP Range | 否；6144/DSA/MLA/router/词表阻塞 | IndexShare 拓扑值得复现；原权重和 MTP 不适合本机在线主线 |
| Kimi K3 | 17.55 MB MXFP4 expert page | 否；113.51 GB 非专家骨架与 25.83 GB/token routed 路径 | 适合研究真实 route trace；原生全深度 20--50 tok/s 已被字节下界否决 |
| MiniMax-M3 / MiMo-V2.5-Pro | 可 Range expert/MTP | 暂无独立能力证据 | 只保留 MSA、SWA/GA、MTP 设计参考，不新增权重 |

共同规律是：**几百 B 模型的知识不在某个 router、indexer 或 MTP 头里；它分布在原生全深度表示与后训练策略中。** 北极星要走新架构，必须把“外部巨型总容量”与“每 token 原生活动计算”分开，而不是把 13B/40B/104B active 原样搬到 8 GiB GPU。

## 5. 首批切片决策

### P0：零权重、立即可做

- 固定官方 revision、tensor 名、dtype/shape、shard、逻辑字节和 header Range 契约；
- loader 支持 FP4 packed + UE8M0 scale 的页级校验；
- 记录 `Loading/Ready`、fence 完成后发布、route miss 和字节账本；
- 不触碰模型能力声明。

下载：**0 B**。

### P1：只在主线批准 runtime ABI 样本时

`deepseek-v4-0731-l42-expert-page`：

- payload：一个完整 routed expert（含三矩阵 packed weight/scales）+ 同 ID router row/bias；
- 逻辑净载荷：**13,377,540 B**；
- 来源文件：`model-00044-of-00048.safetensors`；
- 获取方式：先取 header，再按实际 data offsets 做 Range；禁止整片下载；
- 用途：SHA-256、FP4 解码、槽位发布、重复读一致性和孤立 kernel；
- 禁止用途：宣称获得 DeepSeek 推理/agent 能力，或直接接 v38 晋级。

### HOLD

- DeepSeek L42 整片 3.590 GB；
- DeepSeek DSpark 10.863 GB；
- GLM L77 整层文件并集 21.454 GB；
- GLM L78/MTP 文件并集 26.807 GB；
- GLM indexer 权重和任意 GLM expert；
- 任意“没有原生 route trace 就预选 top-k 专家”的下载。

## 6. 对北极星新架构的可用脑暴

这些是从旗舰结构得到的、但不冒充权重融合的研究方向：

1. **跨层路由保持（来自 IndexShare）：** 不每层重新决定器官；每 3--4 层形成一个路由段，只在隐藏态漂移超过阈值时重选。目标是降低 router、页 miss 与同步次数。
2. **多深度难度观测（来自 DSpark）：** 快主干保留浅/中/深三个小摘要；困难 token 由三者共同决定是否进入递归/器官路径，而不是用最后一层单点熵。
3. **候选与验证分离（来自 DSpark/MTP）：** 巨型器官只提出短结构/关键决策，v38 在原生词表中提交；但没有共享词表时应交换 UTF-8 span/结构 IR，而非强行合 logits。
4. **器官段而非单专家：** 能力候选必须包含连续层站位、norm、router、shared path 与状态契约；单专家仅是存储页，不再叫“能力胶囊”。
5. **低占空比质量预算：** 若 v38 基线约 28 tok/s，想守住 20 tok/s，额外慢路径的平均成本最多约 `1/20 - 1/28 = 14.3 ms/token`。巨型器官必须按片段触发并摊销，不能逐 token 冷启动。

第 5 条是端到端预算，不是已经达到的速度。50 tok/s 高于当前 v38 基线，必须另靠北极星原生推测解码/结构化短输出实现，巨型权重切片本身不会自动带来。

## 7. 可证伪晋级门

任何 300B+ 器官进入能力主线前，至少同时满足：

1. 在供体原生坐标中取得冻结任务的 layer/token route trace；
2. 切片包含显式 norm/router/shared/state 依赖清单，不用 no-op 填掉缺专家；
3. 输入桥不训练时必须有数学上可验证的同坐标证据；否则停止，不用 alpha 搜索救；
4. 强制 donor/no-op 的关键 token NLL 或真实任务净改善可归因；
5. 相邻端到端速度不低于 20 tok/s，且记录实际 SSD/RAM/PCIe 字节、cache hit 与 stall；
6. 不能把 speculative acceptance、索引 FLOPs 降低或 isolated kernel 吞吐直接写成最终 token/s。

## 8. 官方来源

- [DeepSeek-V4-Flash-0731 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [DeepSeek 当前 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
- [DeepSeek 官方推理 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/config.json)
- [DeepSeek 官方推理实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py)
- [DeepSeek 权重索引](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json)
- [GLM-5.2 模型卡](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5.2 config](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/config.json)
- [GLM-5.2 权重索引](https://huggingface.co/zai-org/GLM-5.2/blob/b4734de4facf877f85769a911abafc5283eab3d9/model.safetensors.index.json)
- [Transformers GLM-MoE-DSA 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)
- [IndexShare 论文](https://arxiv.org/abs/2603.12201)

本地交叉证据：`fast16/research/model_scout/`、`fast16/research/capsule_lab/`、`fast16/research/polaris_meridian_v0/k3_audit/` 与 `fast16/research/polaris_meridian_v0/runtime_paging/`。
