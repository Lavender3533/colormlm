# ColorLM 模型侦察：GLM 与最新 Qwen 官方证据

抓取日期：2026-07-31（Asia/Shanghai）  
范围：GLM-5/5.2、GLM-4.7-Flash、Qwen3.6/3.5、Qwen-AgentWorld  
证据优先级：官方 GitHub/博客 > 官方 Hugging Face 模型卡与配置/索引 > 官方组织 API 元数据 > 第三方量化仓库的存在性。  
执行边界：**本次只读取网页、配置、README、`safetensors.index.json` 和 Hub API 元数据；没有下载任何模型权重，没有启动模型/GPU，没有运行 CMake。**

## 一、先纠正型号与发布时间

1. **GLM-5 不是传闻，已正式开源。** 官方模型仓库创建于 2026-02-11，MIT；官方博客资产记录发布日期 2026-02-11。随后已有 GLM-5.1（2026-04）和 **GLM-5.2（2026-06-16）**。截至抓取日，GLM-5.2 才是 GLM 的最新开源旗舰。
2. **“最新 Qwen”不是 Qwen3.5。** Qwen 官方 GitHub 明确写道 “Qwen3.6 is the latest addition”，新闻列出 Qwen3.6-35B-A3B 于 2026-04-16、Qwen3.6-27B 于 2026-04-22 发布。官方组织没有 Qwen3.7 或 Qwen4 开源 checkpoint；榜单中的 `Qwen3.7-Max` 是对照模型名，不能当成已开放 donor。
3. **Qwen-AgentWorld-35B-A3B 于 2026-06-24 正式发布**，时间晚于 Qwen3.6，但它是以 Qwen3.5-35B-A3B-Base 为底座、专门预测环境下一状态的 language world model，不是新的通用基座。

版本证据：

- GLM-5.2：[官方 GitHub](https://github.com/zai-org/GLM-5)、[官方博客](https://z.ai/blog/glm-5.2)、[HF](https://huggingface.co/zai-org/GLM-5.2)
- GLM-5：[官方博客](https://z.ai/blog/glm-5)、[技术报告](https://arxiv.org/abs/2602.15763)、[HF](https://huggingface.co/zai-org/GLM-5)
- Qwen3.6：[官方 GitHub 新闻](https://github.com/QwenLM/Qwen3.5#news)、[35B-A3B HF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)、[27B HF](https://huggingface.co/Qwen/Qwen3.6-27B)
- Qwen-AgentWorld：[官方 GitHub](https://github.com/QwenLM/Qwen-AgentWorld)、[HF](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)、[论文](http://arxiv.org/abs/2606.24597)

## 二、侦察结论与排名

| 排名 | 候选 | 档位 | 对 ColorLM 的主要价值 | 核心阻碍 |
|---:|---|---|---|---|
| 1 | **Qwen3.6-35B-A3B** | **值得提取** | hidden=2048、vocab=248320，与 ColorLM 本地记录同形；最新 agentic coding、preserved thinking、通用/工具/规划；Apache-2.0 | 新的 Gated DeltaNet 状态；专家按整层融合为大张量；同形不等于坐标已对齐 |
| 2 | **GLM-4.7-Flash** | **值得提取** | 30B-A3B、hidden=2048、MIT；末层/输出头/64 个独立专家集中在单一 2.539 GB 分片；SWE/τ²/Browse 能力可补 Coder-Next | vocab/tokenizer 不同；GLM MLA 状态与 Qwen 图不同；能力较 GLM-5.2 老一代 |
| 3 | **Qwen-AgentWorld-35B-A3B** | **值得提取（只做隔离实验）** | hidden=2048、vocab=248320、同 Qwen tokenizer；末层+输出头完整落在单一 3.890 GB 分片；覆盖 MCP/Search/Terminal/SWE/Android/Web/OS | 目标是“模拟环境输出”而不是“选择动作”；直接替换主输出头有行为反转风险；checkpoint 无视觉权重 |
| 4 | **GLM-5.2** | **只值得借鉴架构** | 当前最强 GLM；MIT；1M 上下文、DSA+IndexShare、长程 coding/agent/tool 指标强，和 Coder-Next 互补明显 | 744B/40B、hidden=6144、GLM tokenizer；末层需 21.45 GB 文件，单专家所在分片也需 5.37 GB；桥和 DSA 状态成本高 |
| 5 | **Qwen3.6-27B** | **只值得借鉴架构** | 最新密集版；代码、终端、AndroidWorld、preserved thinking 强；tokenizer/vocab 与 ColorLM 对齐 | hidden=5120，末层文件集 8.43 GB，输出头宽度不兼容，且没有可选专家 |
| 6 | Qwen3.5-397B-A17B / 35B-A3B | **不值得新投入** | 可用于补齐 Qwen3.6 未披露的通用 agent/长上下文对照；AgentWorld 的底座 | 通用线已被 Qwen3.6 替代；与现有 Qwen3-Coder-Next 的重叠高 |
| 7 | GLM-5 原版 | **不值得新投入** | 证实了 744B/40B、DSA 与 agentic engineering 路线 | 已被 GLM-5.2 正式替代，结构/体积几乎相同但能力和上下文更弱 |

推荐次序不是“直接装配次序”。任何权重提取都应先通过主线批准；首个批准对象应只包含 **Qwen3.6-35B-A3B 输出头所在分片**，其余按冻结能力门逐级放行。

## 三、ColorLM 本地兼容基线

只读本地证据 `fast16/research/v19_dual_head/token_map_report.json` 显示：

- ColorLM hidden=2048，vocab=248320；已知 `<|endoftext|>/<|im_start|>/<|im_end|>` 等控制 token 位于 Qwen3.5/3.6 的 248k 词表区间。
- 现有 Qwen3-Coder-Next donor hidden=2048，但 vocab=151936；本地曾做过 151936→248320 的 token 映射和独立输出头实验。
- Qwen3.6、Qwen3.5、Qwen-AgentWorld 的官方 `tokenizer.json` 是同一文件：12,807,982 B，SHA-256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`。这使 Qwen3.6-35B-A3B 和 AgentWorld 成为目前形状/词表最接近 ColorLM 的新 donor。

注意：词表和张量形状一致仍不能证明 hidden coordinate system 一致。必须做全词表原始字节比对、同 prompt 深层激活对齐、alpha=0/no-op 和 next-token NLL 门控，不能直接热插输出头或末层。

## 四、结构、tokenizer 与权重布局总表

下表中的参数/结构来自官方模型卡和 `config.json`；权重字节/分片来自官方 Hub `?blobs=true`；“激活参数”只有模型名或模型卡明确披露时才填写。

| 候选 | 总参/激活 | 层/hidden | 专家/top-k | tokenizer / vocab / head tied | 上下文 | BF16/原始权重与分片 | 官方量化 |
|---|---|---|---|---|---|---|---|
| GLM-5.2 | 744B/40B | 78 / 6144 | 256 routed + 1 shared / top-8 | TokenizersBackend，154880；`tie_word_embeddings=false` | 1,048,576 | 1,506.667 GB，282 片，0.294–5.368 GB | 官方 FP8 755.632 GB / 141 片 |
| GLM-5 | 744B/40B | 78 / 6144 | 256+1 / top-8 | 同 GLM，154880；不 tied | 202,752 | 1,507.736 GB，282 片 | 官方 FP8 756.178 GB / 142 片 |
| GLM-4.7-Flash | 30B/3B | 47 主层 + 1 MTP / 2048 | 64+1 / top-4 | PreTrainedTokenizer，154880；不 tied | config 202,752；tokenizer 128,000 | 62.444 GB，48 片，约 1.27–2.54 GB | 无官方 FP8/GPTQ；有社区 GGUF |
| Qwen3.6-35B-A3B | 35B/3B | 40 / 2048 | 256 + 1 shared / 8 routed | Qwen2Tokenizer，248320；不 tied | 262,144 原生，可扩至 1,010,000 | 71.904 GB，26 片，1.096–3.996 GB | 官方 FP8 37.464 GB / 42 片；无官方 GPTQ/GGUF |
| Qwen3.6-27B | 27B dense | 64 / 5120 | 无 | Qwen2Tokenizer，248320；不 tied | 同上 | 55.563 GB，15 片，0.509–3.995 GB | 官方 FP8 30.867 GB / 66 片 |
| Qwen3.5-397B-A17B | 397B/17B | 60 / 4096 | 512 + 1 shared / 10 routed | Qwen2Tokenizer，248320；不 tied | 262,144 原生，可扩 1,010,000 | 806.796 GB，94 片 | 官方 FP8 406.152 GB；GPTQ-Int4 235.708 GB |
| Qwen3.5-35B-A3B | 35B/3B | 40 / 2048 | 256+1 / 8 routed | Qwen2Tokenizer，248320；不 tied | 同上 | 71.904 GB，14 片 | 官方 FP8 37.464 GB；GPTQ-Int4 24.420 GB |
| Qwen-AgentWorld-35B-A3B | 35B/3B | 40 / 2048 | 256+1 / 8 routed | Qwen2Tokenizer，248320；不 tied | config/卡 262,144；tokenizer_config 131,072 | 69.321 GB，21 片，3.221–3.890 GB | 无官方量化；有社区 GGUF/FP8 |

架构要点：

- GLM-5/5.2：`GlmMoeDsaForCausalLM`，前三层 dense，后续 MoE；MLA/DSA，sigmoid `noaux_tc` router。GLM-5.2 的 IndexShare 每四个稀疏注意力层复用 indexer，官方称在 1M 上把 per-token FLOPs 降 2.9 倍；MTP acceptance length 最多提升 20%。
- GLM-4.7-Flash：`Glm4MoeLiteForCausalLM`，首层 dense，之后 64-expert MoE；每专家中间维 1536，hidden=2048。
- Qwen3.6-35B-A3B：沿用 `qwen3_5_moe` 图，10 组 `3×Gated DeltaNet + 1×Gated Attention`；每层 MoE，MTP；原生视觉编码器。
- Qwen3.6-27B：同样的 3:1 DeltaNet/全注意力混合，但 FFN 为 dense（intermediate=17408）。
- AgentWorld：与 Qwen3.5-35B-A3B 文本图同形，但 `language_model_only=true`，官方 vLLM 要求 `--language-model-only`，仓库不含视觉模块权重。

社区 GGUF 仅作“已有生态”证据，不能替代官方权重审计：

- GLM-5.2：[unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF)
- GLM-4.7-Flash：[ggml-org/GLM-4.7-Flash-GGUF](https://huggingface.co/ggml-org/GLM-4.7-Flash-GGUF)、[unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)
- Qwen3.6-35B-A3B：[unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)、[ggml-org/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF)
- Qwen3.6-27B：[unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)
- AgentWorld：[unsloth/Qwen-AgentWorld-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF)

## 五、官方基准证据

均为官方自报，harness、上下文、采样和 judge 不完全相同，**只可在同一模型卡表内横向比较，不能把不同卡的数字当严格排行榜**。

| 模型 | 推理 | 代码 | 工具/Agent/规划 | 长上下文/电脑操作 |
|---|---|---|---|---|
| GLM-5.2 | HLE 40.5；HLE+Tools 54.7；GPQA 91.2；AIME 2026 99.2 | SWE-bench Pro 62.1；NL2Repo 48.9；DeepSWE 46.2；Terminal-Bench 2.1 81.0/最佳 harness 82.7 | MCP-Atlas 76.8；Tool-Decathlon 48.2 | 官方用 400K 跑多项 coding，FrontierSWE/PostTrainBench/SWE-Marathon 用 1M；未披露 OSWorld |
| GLM-5 | HLE 30.5；HLE+Tools 50.4；GPQA 86.0 | SWE Verified 77.8；SWE Multilingual 73.3；Terminal 2.0 56.2/校正版 60.7 | τ² 89.7；MCP-Atlas 67.8；Tool-Decathlon 38.0；BrowseComp 62.0/带 context manage 75.9 | 200K SWE、202,752 HLE tools；无 OSWorld |
| GLM-4.7-Flash | AIME25 91.6；GPQA 75.2；HLE 14.4 | LiveCodeBench v6 64.0；SWE Verified 59.2 | τ² 79.5；BrowseComp 42.8 | 未披露专门长上下文/GUI 基准 |
| Qwen3.6-35B-A3B | GPQA 86.0；AIME26 92.7；SuperGPQA 64.7 | SWE Verified 73.4；SWE Pro 49.5；Terminal 2.0 51.5；LiveCodeBench v6 80.4 | DeepPlanning 25.9；MCPMark 37.0；MCP-Atlas 62.8；Tool-Decathlon 26.9；Claw-Eval Avg 68.7 | 卡中未报告 LongBench/OSWorld；仅声明 262K→1.01M |
| Qwen3.6-27B | GPQA 87.8；AIME26 94.1 | SWE Verified 77.2；SWE Pro 53.5；Terminal 2.0 59.3；LiveCodeBench 83.9；SkillsBench 48.2 | QwenClawBench 53.4；Claw-Eval Avg 72.4 | AndroidWorld 70.3；卡中未报告 LongBench |
| Qwen3.5-397B-A17B | GPQA 88.4；AIME26 91.3 | SWE Verified 76.4；Terminal 2 52.5；LiveCodeBench 83.6 | BFCL-V4 72.9；TAU2 86.7；Tool-Decathlon 38.3；MCP-Mark 46.1；BrowseComp 69.0/78.6 | LongBench v2 63.2；MMLongBench-Doc 61.5；OSWorld-Verified 62.2 |
| AgentWorld-35B-A3B | 不作为通用推理榜单发布 | AgentWorldBench SWE 65.63 | AgentWorldBench：MCP 64.79、Search 36.69、Terminal 53.96、总分 56.39 | Android 58.17、Web 49.55、OS 65.92；建议至少 128K |

Qwen3.6 的互补点不是单纯更高代码分：官方强调 **agentic coding、repository-level reasoning、frontend workflow 和 thinking preservation**。Qwen3-Coder-Next 已覆盖专门编程和工具；Qwen3.6 更适合作为同形的通用/规划/多模态后训练 donor。AgentWorld 则是环境预测辅助支路候选，而不是替代 Coder-Next 的 action policy。

## 六、官方最低部署硬件

各官方模型卡都**没有给出可验证的 GPU 型号、单卡显存或严格“最低硬件”**，因此不能把社区经验写成官方最低要求。可记录的官方部署示例只有：

- GLM-5：vLLM/SGLang 示例 `TP=8`、显存利用率 0.85；GLM-5.2 卡仅列 vLLM/SGLang/KTransformers/Transformers/Ascend 支持，未写 GPU 数。
- GLM-4.7-Flash：vLLM 示例 `TP=4`；未写 GPU 型号/显存。
- Qwen3.6-35B-A3B、Qwen3.6-27B：卡内 262K 示例均写 8 GPU；未写型号/显存，这只是官方例子，不是最低值。
- Qwen3.5-397B/35B 卡同样示例 8 GPU。
- AgentWorld-35B-A3B：SGLang/vLLM 示例 `TP=4`，262K；官方提示 OOM 时缩短上下文，并建议保持至少 128K。

由权重文件体积可得存储/聚合内存下限，但这是推导而非官方最低硬件：BF16 权重本身约 GLM-5.2 1.507 TB、Qwen3.6-35B 71.9 GB、Qwen3.6-27B 55.6 GB、GLM-4.7-Flash 62.4 GB；运行还需 KV/DeltaNet 状态、视觉模块、框架缓冲和通信空间。

## 七、选择性提取可行性（依据官方 safetensors 索引）

“能只下载某张量”必须分两种情况：

- **标准 Hub 文件下载**：最小单位是完整 safetensors 分片，下表给出真实网络体积。
- **自定义 HTTP Range 活检**：理论上可读 safetensors header 后按 byte range 提取独立 tensor；Qwen 的 routed experts 是融合张量，单专家还需按 expert 轴切片，官方索引本身不提供单专家文件。主线批准前不得执行任何 Range 权重读取。

| 候选 | 输出头所在分片 | 完整末层文件集 | 全部 router 的文件代价 | 少量专家 |
|---|---:|---:|---:|---|
| GLM-5.2 | shard 1，5.343 GB；head 与 embedding 同片 | L77：4 片，21.454 GB | 76 片，407.560 GB | 专家为独立 tensor；L77/E0 三张量约 75.5 MB，但所在 shard 267 为 5.366 GB |
| GLM-5 | shard 1，5.343 GB | L77：5 片，26.801 GB | 76 片，403.571 GB | 独立专家；单专家所在分片约 5.36 GB |
| GLM-4.7-Flash | shard 47，2.539 GB | L46 全部都在 shard 47，2.539 GB | 47 片，61.006 GB | 64 个专家各自独立 tensor；单专家 BF16 约 18.9 MB，但标准下载仍为 2.539 GB |
| Qwen3.6-35B-A3B | shard 26，2.231 GB；head 实体约 1.017 GB | L39：shard 25+26，共 6.064 GB | 41 个 router（含 MTP）散在 17 片，37.918 GB；实体约 43 MB | `experts.gate_up_proj/down_proj` 为融合 tensor；单专家实体约 6.29 MB，但完整 L39 需两片 |
| Qwen3.6-27B | shard 8，3.879 GB；head 实体约 2.543 GB | L63：shard 12+13+15，共 8.425 GB | 无 router | dense，无少量专家路径 |
| Qwen3.5-397B-A17B | shard 91，9.639 GB | L59：5 片，41.218 GB | 所有 router 共置 shard 94，4.741 GB | 融合专家；L59 两个 expert tensor 落两片，共 17.180 GB |
| Qwen3.5-35B-A3B | shard 9，5.255 GB | L39：4 片，18.330 GB | 所有 router 共置 shard 14，2.225 GB | 融合专家；L39 expert tensor 两片 10.737 GB |
| AgentWorld-35B-A3B | shard 21，3.890 GB | L39 **完整同在 shard 21**，3.890 GB | 40 个 router 在 4 片，14.119 GB | 融合专家；L39 的两大专家 tensor 同在 shard 21 |

### 接入难点与 Coder-Next 互补性

**Qwen3.6-35B-A3B**

- 最强兼容信号：hidden 2048、vocab 248320、Qwen tokenizer 文件与 ColorLM 所属 248k 词表体系一致，head 形状 `[248320, 2048]`。
- 不能直接复用 Coder-Next 的独立专家提取器：Qwen3.6 把每层 256 个 routed experts 融在 `gate_up_proj`/`down_proj` 两张大 tensor 中。
- Gated DeltaNet 有卷积/递归状态；连续层岛必须明确 state 生命周期，不能只接 FFN 当成普通 Qwen3 MoE。
- 相对 Coder-Next 的互补性：通用对话、视觉、planning、thinking preservation、frontend/repository workflow；代码能力有重叠，但不是纯重复。

**GLM-4.7-Flash**

- hidden=2048 可省去维度桥；独立专家 tensor 与现有 Coder-Next Range 活检思路更接近。
- GLM tokenizer/vocab 154880 与 ColorLM 不同，输出头不能直接替换；只适合残差胶囊、专家或带 transport 的连续块。
- GLM 的 MLA 投影、router 规则和 preserved-thinking 模板需要新运行时适配。
- 相对 Coder-Next 的互补性：更偏通用推理、Browse/τ² 和轻量 agent，且 MIT。

**AgentWorld-35B-A3B**

- hidden/vocab/tokenizer 与 Qwen3.6-35B 同形，且 shard 21 单文件包含完整末层、router、融合专家、norm 和 `lm_head`，是最干净的 agent 专项活检文件。
- 训练目标是给定 action/history 预测 environment observation；它适合 gated simulator/critic/auxiliary head，不适合作为默认 action generator。
- 相对 Coder-Next 的互补性：MCP、Search、Terminal、SWE、Android、Web、OS 的环境动力学；不等于工具调用策略。

**GLM-5.2**

- 价值主要在 DSA/IndexShare、长程 MTP、1M context 和长时 agent 后训练；不是廉价张量 donor。
- hidden 6144→2048、vocab 154880→248320、DSA index/KV 状态和 78 层深度都要求新桥；单专家虽可 Range 切出约 75.5 MB，文件级最小下载仍约 5.37 GB。
- 相对 Coder-Next 的能力互补最强，但单位接入成本也最高；当前应读架构/训练方法，不应先下权重。

**Qwen3.6-27B**

- tokenizer/vocab 对齐但 hidden 5120，dense FFN 没有“少量专家”捷径。
- 其 Terminal/SWE/AndroidWorld 分数高于 35B-A3B，适合当能力 teacher 或架构对照；不适合第一批物理 graft。

## 八、待主线批准的精确下载清单

以下仅是计划，**本次未下载**。全部固定到抓取时的官方 revision，避免后续同名仓库漂移。

### A. 首批唯一建议：Qwen3.6-35B-A3B 输出头审计

仓库：`Qwen/Qwen3.6-35B-A3B`  
revision：`995ad96eacd98c81ed38be0c5b274b04031597b0`

先保存元数据：`README.md`、`LICENSE`、`config.json`、`generation_config.json`、`chat_template.jinja`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`、`model.safetensors.index.json`。

批准后第一阶段只取：

- `model-00026-of-00026.safetensors` — 2,231,416,848 B；SHA-256 `1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613`。包含 `lm_head.weight`、L39 router、L39 attention/norm/shared expert、融合 `down_proj`；不含 L39 `gate_up_proj`。

只有输出头/坐标门通过后，第二阶段再取：

- `model-00025-of-00026.safetensors` — 3,832,888,256 B；SHA-256 `778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154`。包含 L39 融合 `gate_up_proj`。与 shard 26 合计 6,064,305,104 B，构成完整 L39。
- 可选 `model-00001-of-00026.safetensors` — 3,996,199,712 B；SHA-256 `adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478`。包含 token embedding；只有确认需要 embedding-side transport 时才取。

### B. 第二候选：GLM-4.7-Flash 单文件末层/专家审计

仓库：`zai-org/GLM-4.7-Flash`  
revision：`7dd20894a642a0aa287e9827cb1a1f7f91386b67`

- `model-00047-of-00048.safetensors` — 2,539,429,936 B；SHA-256 `1bcc5d06065d2a564894657945ccfe9411762421c2c60acf91de31050cd4d84d`。包含 `lm_head` 和完整 L46（64 独立 experts、router、shared expert、attention、norm）。
- 可选 embedding：`model-00001-of-00048.safetensors` — 1,438,134,344 B；SHA-256 `90abe0d075755853145c96906a1300f57c167fcc9aa67221239b448abf54933c`。

### C. 第三候选：AgentWorld 隔离辅助头

仓库：`Qwen/Qwen-AgentWorld-35B-A3B`  
revision：`60d2b0434a53d2e62a7c00a489586815d94ebffb`

- `model-00021-of-00021.safetensors` — 3,889,712,984 B；SHA-256 `e6379e7108900493e234856276c32250c113e4fc461511f72d6b1015441e6057`。包含完整 L39、router、融合 experts、final norm 和 `lm_head`。
- 可选 embedding：`model-00011-of-00021.safetensors` — 3,784,847,976 B；SHA-256 `e3eb8dc24da411913a950a4191d513f9ae8a70a26b1c0488b887cc8a59b2f603`。

### 明确不建议批准的权重

- GLM-5.2：不要下载 shard 1 或 L77 的 4 个分片；先完成 DSA/IndexShare 状态接口和 6144→2048 桥设计审查。
- Qwen3.6-27B：不要下载输出头/末层；先证明 5120→2048 transport 的收益高于 Qwen3.6-35B 同形 donor。
- GLM-5、Qwen3.5 通用 checkpoint：已被新版本替代，不进入新下载队列。
- 任意第三方 GGUF：只用于生态可用性参考，不作为首轮供体来源；官方 revision 和张量索引应是审计基准。

## 九、来源索引

### GLM 官方

- [GLM-5/5.2 GitHub README](https://github.com/zai-org/GLM-5)
- [GLM-5.2 HF 模型卡](https://huggingface.co/zai-org/GLM-5.2) / [config](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) / [index](https://huggingface.co/zai-org/GLM-5.2/blob/main/model.safetensors.index.json)
- [GLM-5 HF 模型卡](https://huggingface.co/zai-org/GLM-5) / [config](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)
- [GLM-4.7-Flash HF 模型卡](https://huggingface.co/zai-org/GLM-4.7-Flash) / [config](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json) / [index](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/model.safetensors.index.json)
- [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)、[IndexShare](https://arxiv.org/abs/2603.12201)

### Qwen 官方

- [Qwen3.6/3.5 GitHub](https://github.com/QwenLM/Qwen3.5)
- [Qwen3.6-35B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) / [config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json) / [index](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/model.safetensors.index.json)
- [Qwen3.6-27B 模型卡](https://huggingface.co/Qwen/Qwen3.6-27B) / [config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json) / [index](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/model.safetensors.index.json)
- [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)、[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen-AgentWorld GitHub](https://github.com/QwenLM/Qwen-AgentWorld)、[模型卡](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)、[index](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B/blob/main/model.safetensors.index.json)

### 元数据接口

- `https://huggingface.co/api/models/{repo}?blobs=true`：revision、创建/修改时间、文件字节、LFS SHA-256。
- `https://huggingface.co/api/models?author=zai-org...` 与 `?author=Qwen...`：截至 2026-07-31 的官方组织模型枚举，用于排除未发布的传闻型号。

## 十、证据边界

- 所有 benchmark 是模型发布方自报；没有本地复跑。
- “实体张量体积”由公开 config 的形状与 BF16 两字节推导；“下载体积”来自官方 Hub 文件元数据。
- `safetensors.index.json` 只说明 tensor→shard 映射，不证明服务器长期支持稳定的任意 Range；也不证明切出的张量能在 ColorLM 坐标中工作。
- 官方没有严格最低硬件披露，本文没有用社区配置冒充官方最低要求。
- 本报告没有将 Qwen3.7-Max、DeepSeek-V4 等榜单对照名当作可下载开源 checkpoint。
