# DeepSeek / Moonshot-Kimi donor 侦察

**截止时间：** 2026-07-31（Asia/Shanghai）  
**执行边界：** 本轮只读取官网、官方模型仓、配置、权重索引、许可证、推理代码、仓库文件清单和 HTTP `HEAD`。没有请求任何 safetensors/GGUF 权重内容，没有启动模型或占用 GPU。

证据等级沿用 [METHODOLOGY.md](./METHODOLOGY.md)：A=官方明确，B=可复算推导，C=社区实现，U=官方未披露。

## 结论先行

用户点名的两个名称都是真实官方发布，不需要回退到旧型号：

| 排名 | 候选 | 名称核查 | 本组结论 | 最有价值的互补能力 | 建议的首轮权重动作 |
|---:|---|---|---|---|---|
| 1 | `deepseek-ai/DeepSeek-V4-Flash-0731` | 官方 HF 仓于 2026-07-31 创建；模型卡称其为取代 preview 的正式版 | **值得提取** | MIT、1M 上下文、通用推理/工具/长程 agent；13B 激活；单层仅 3.590 GB | 主线批准后只取末层 L42 分片；先不取输出头和 DSpark |
| 2 | `moonshotai/Kimi-K3` | 官方博客 2026-07-27 发布；官方 HF、GitHub、技术报告齐全 | **值得提取（仅视觉塔优先）；文本主干只值得借鉴架构** | 当前两者中唯一原生视觉与电脑操作 donor；OSWorld-Verified 84.8；KDA/AttnRes/1M | 主线批准后优先取独立视觉塔+投影器 894.738 MB；文本 MoE 等桥接原型通过后再议 |

不建议把 `DeepSeek-V4-Flash` preview 当第三个 donor。它已被 0731 正式版明确 supersede；现有 `ggml-org/DeepSeek-V4-Flash-GGUF` 也对应 preview，不是 0731 权重。

---

## 1. DeepSeek-V4-Flash-0731

### 1.1 官方身份、日期、许可

- **A 官方入口：** [Hugging Face 模型仓](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)、[官方 README](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/README.md)、[技术报告 arXiv:2606.19348](https://arxiv.org/abs/2606.19348)、[官方 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json)、[官方 LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE)。
- **A/B 发布日期：** HF 官方组织仓 `createdAt=2026-07-31T07:30:24Z`，模型名和 README 均为 `0731`；按本轮口径记 **2026-07-31**。
- **A 许可：** 标准 **MIT License**，README 明确代码和模型权重均为 MIT。允许使用、修改、分发、再许可和商业使用，只需保留版权/许可声明。
- **A 版本关系：** 0731 README 称其为 DeepSeek-V4-Flash 正式版，取代 preview，显著增强 agent 能力；结构与 `DeepSeek-V4-Flash-DSpark` 相同，附带 speculative decoding 模块。

### 1.2 参数与架构

| 字段 | 结论 | 证据 |
|---|---|---|
| 总参数 / 激活参数 | **核心模型 284B / 13B**；0731 仓库 safetensors 实计 **304,180,418,494** 参数 | A/B，[V4 preview 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 与 [0731 HF API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731)；差额来自 DSpark/MTP 等附加张量，`13B active` 是核心模型口径 |
| 主干层数 | **43** | A，`num_hidden_layers=43` |
| 隐藏维 | **4096** | A，`hidden_size=4096` |
| MoE | **256 routed + 1 shared，top-6** | A，`n_routed_experts=256`、`n_shared_experts=1`、`num_experts_per_tok=6` |
| 专家中间维 | **2048** | A，`moe_intermediate_size=2048` |
| 特殊路由 | 前 **3** 层为 hash routing，其余为 score routing | A，`n_hash_layers=3`；官方 `inference/model.py` 的 `Gate` 实现 |
| 注意力 | CSA/HCA 混合；64 query heads、1 KV head；`head_dim=512` | A，官方报告/配置 |
| 长上下文 | **1,048,576 tokens**；YaRN 从原生 65,536 外推 16 倍 | A，`max_position_embeddings` 和 `rope_scaling` |
| 残差结构 | manifold-constrained Hyper-Connections（mHC） | A，官方模型卡/报告 |
| speculative 模块 | DSpark，目标主干 L40-L42；索引中为 `mtp.0..2` 三个大块 | A，`dspark_target_layer_ids=[40,41,42]` 与权重索引 |
| 权重精度 | routed experts FP4；大多数其余线性层 FP8；embedding/head 实际为 BF16 体积 | A/B，模型卡、配置、官方推理代码与分片大小 |

注意：配置同时出现 `num_nextn_predict_layers=1`，但 0731 索引实际有 `mtp.0`、`mtp.1`、`mtp.2`，且 DSpark 目标层是 3 个。做提取清单时应以 **权重索引实际张量** 为准，不能只按这个旧兼容字段判断。

### 1.3 tokenizer 与输出头

- **A tokenizer：** `PreTrainedTokenizerFast`，`tokenizer.json` 是 Hugging Face Tokenizers **byte-level BPE**（ByteLevel decoder），主配置词表 **129,280**；模型最大长度 1,048,576。官方还提供 `encoding/encoding_dsv4.py`，负责 OpenAI-compatible messages、reasoning effort 和工具调用文本的编码/解析，**没有 Jinja chat template**。
- **A 特殊 token：** BOS id 0，EOS id 1，pad 复用 EOS；0731 支持 `low/high/max` 三档 `reasoning_effort`。
- **A 输出头不共享：** `tie_word_embeddings=false`；索引中的 `embed.weight` 和 `head.weight` 是两个独立张量。
- **B 头部体积：** 单个 embedding/head 均为 `129280 * 4096 * 2 = 1,059,061,760` 字节（0.986 GiB，BF16）。`head.weight` 位于 `model-00045-of-00048.safetensors`，该分片总长 1,059,332,516 字节。
- **兼容结论：** ColorLM/Qwen3-Coder-Next 词表为 151,936，token id 语义不对齐；即使训练 `2048 -> 4096` 隐藏桥，也不能把 DeepSeek 输出头按行直接替换。输出头只适合 logit 蒸馏或基于共享文本重拟合词表映射，不是首轮 donor。

### 1.4 权重体积、分片、量化现状

- **A/B 官方 safetensors：** `model.safetensors.index.json.metadata.total_size = 166,878,536,440` 字节（166.879 GB / 155.420 GiB，张量净载荷）。48 个 safetensors 文件合计 **166,886,535,336** 字节（166.887 GB / 155.427 GiB，含 header）；仓库已知文件总计 166,898,660,330 字节。
- **A/B 参数口径：** HF safetensors 元数据对 0731 仓库的实际计数为 **304,180,418,494**，高于模型卡的 284B 核心 Flash 口径。存储与下载预算必须按前者/实际分片计，核心能力对比仍按官方 284B/13B 计。
- **A 分片结构：** 48 片。`00001` 是 embedding；`00002..00044` 基本是一层主干一片；`00045` 是 norm/output head；`00046..00048` 是三个 DSpark/MTP 块。
- **A 关键分片：** L42 全部 1,576 个张量都在 `model-00044-of-00048.safetensors`，文件大小 **3,590,026,352** 字节（3.590 GB / 3.343 GiB）。DSpark 三片总计 **10,863,342,388** 字节（10.863 GB / 10.117 GiB）。
- **A 原生量化：** 官方 checkpoint 是 FP4 experts + FP8 mixed。0731 自身不是 GGUF。
- **C GGUF/社区量化：** 截止本次查询，`unsloth/DeepSeek-V4-Flash-0731-GGUF` 只有 `.gitattributes` 和 README，共 6,876 字节，是占位仓，**尚无 GGUF 文件**。`ggml-org/DeepSeek-V4-Flash-GGUF` 有 preview 的单文件 `DeepSeek-V4-Flash-MXFP4.gguf`，154,991,536,896 字节；它不能替代 0731。另有 preview 的 NVIDIA NVFP4、SGLang FP8 等转换。

动态文件清单来源：[HF 官方 API](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731?blobs=true)、[0731 Unsloth GGUF 占位仓](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)、[preview ggml-org GGUF](https://huggingface.co/ggml-org/DeepSeek-V4-Flash-GGUF)。社区仓只作为 C 级“已有/尚无转换”证据。

### 1.5 官方最低部署硬件

- **A 作者给出的具体可运行例：** 0731 模型卡用 vLLM 在**单个 4 x GB300 节点**运行，启用 data-parallel 4、expert parallel、FP8 KV 与 DSpark。这是作者公布的最小具体例子，不应误写成理论最低要求。
- **U 作者没有声明通用“最低 VRAM”。** vLLM 官方 recipe 对 preview checkpoint 估算权重最低约 **170 GB VRAM**，并给出 TP8/TEP/DEP 配置；但 0731 多出约 7.27 GB DSpark 权重，且长达 1M 的 KV/工作区会显著增加显存。该 170 GB 不能当作 0731 全 1M 上下文端到端最低值。
- 参考：[0731 vLLM 命令](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#how-to-run-with-vllm)、[vLLM V4-Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)。

### 1.6 能力与基准

0731 正式版只公布了新 agent/coding 表；通用推理和 1M 长上下文沿用同结构 preview 官方表。不同 harness 不横比。

| 能力 | 官方结果 | 条件/说明 |
|---|---:|---|
| 通用知识/推理 | MMLU-Pro 86.2；GPQA Diamond 88.1 | A，V4-Flash **Max** preview 官方表 |
| 编程推理 | LiveCodeBench 91.6；Codeforces rating 3052 | A，V4-Flash Max preview 官方表 |
| 1M 长上下文 | MRCR 1M 78.7；CorpusQA 1M 60.5 | A，V4-Flash Max preview 官方表 |
| 代码 Agent（0731） | Terminal-Bench 2.1 **82.7**；NL2Repo **54.2**；DeepSWE **54.4**；Cybergym **76.7** | A，0731；DeepSeek Harness minimal mode、max effort、`temperature=1.0/top_p=0.95` |
| 工具/Agent（0731） | Toolathlon-Verified **70.3**；Agents' Last Exam **25.2**；AutomationBench Public **25.1** | A，0731 |
| 规划/长程执行 | DSBench-FullStack 68.7；DSBench-Hard 59.6 | A，但两者是 DeepSeek 内部集，只作旁证 |
| 电脑操作 | **U，无视觉/OSWorld 官方结果** | 文本模型，不能作为视觉 computer-use donor |

来源：[0731 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#introduction)、[V4-Flash preview 统一评测表](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash#evaluation-results)。

### 1.7 局部提取可行性与体积

官方 CDN 对 L42 分片的 `HEAD` 返回 `Accept-Ranges: bytes`，`Content-Length=3590026352`。权重索引可先定位分片，再读取 safetensors header 元数据得到精确 byte offsets；因此技术上可以只取 router、少量 experts、末层或 head，不需要整模。

**B 末层 L42 逻辑体积复算：**

- 单 routed expert 有 `w1/w2/w3` 三组 FP4 权重和每 32 元素一个 UE8M0 scale。
- 每矩阵逻辑参数 `4096 * 2048`；packed FP4 为 `4,194,304` 字节，scale 为 `262,144` 字节。
- 单专家：`3 * (4,194,304 + 262,144) = 13,369,344` 字节（12.75 MiB）。
- top-6 活跃专家：`6 * 13,369,344 = 80,216,064` 字节（76.50 MiB）。
- score router：`256 * 4096 * 2 + 256 * 4 = 2,098,176` 字节（BF16 weight + FP32 bias）。
- shared expert：`3 * 4096 * 2048 * 2 = 50,331,648` 字节（48 MiB，BF16）。
- “router + 6 experts + shared expert”理论净载荷：**132,645,888 字节**（126.50 MiB），未含 attention/mHC/norm。
- 全 256 routed experts：`256 * 13,369,344 = 3,422,552,064` 字节；加 router/shared 后 3,474,981,888 字节。和 3.590 GB 全层分片相差的是 attention、mHC、norm、scale/header 等。

**能否直接接 ColorLM：** 不能。当前 donor/ColorLM 隐藏宽均为 2048；DeepSeek 是 4096，需要训练矩形 `2048 <-> 4096` 入/出桥。CSA/HCA 有压缩状态、稀疏索引和独立 cache；mHC 不是普通残差。最稳妥的首轮实验是只用 L42 FFN/MoE 残差岛，先绕开注意力与 DSpark。

### 1.8 与 Qwen3-Coder-Next 的互补性

- **强互补：** Qwen donor 是 2048-hidden、512 experts/top-10、262K、编程专向；DeepSeek 是 4096-hidden、256/top-6、1M，0731 强化通用工具、终端、网络仓库级 agent，并带 DSpark。它能补“通用推理、长上下文、规划/工具”，而不是重复复制 coder 专长。
- **工程优势：** 284B/13B 明显小于 Kimi K3 的 2.8T/104B，末层仅 3.590 GB；MIT 许可也比 Kimi 自定义许可更干净。
- **主要风险：** 非 2048 坐标、定制 FP4 dtype/UE8M0 scale、mHC、CSA/HCA 状态、无 Jinja chat template。现有 Qwen BF16 专家加载路径不能原样复用。

### 1.9 决策

**类别：值得提取。** 但首轮仅批准 L42，不批准整模、输出头或 DSpark。成功门槛应是：矩形桥可训练、FP4 expert 可在 ColorLM runtime 解码、强制残差试验优于 no-op，随后再做路由专家选择。

---

## 2. Kimi K3

### 2.1 官方身份、日期、许可

- **A 官方入口：** [Kimi K3 Tech Blog](https://www.kimi.com/blog/kimi-k3)、[Hugging Face 模型仓](https://huggingface.co/moonshotai/Kimi-K3)、[GitHub](https://github.com/MoonshotAI/Kimi-K3)、[完整技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)、[官方 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)、[LICENSE](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)。
- **A/B 发布日期：** 官方博客页面标注 **July 27, 2026**；HF 仓实际 `createdAt=2026-06-13T06:42:57Z`，说明仓库曾提前建立。对外发布日期记 **2026-07-27**，仓库创建日另保留。
- **A 许可：** 自定义 **Kimi K3 License**，不是 MIT/Apache/OSI 许可证。它广泛允许使用、修改、分发、微调和商业使用，但有两条重要商业限制：
  - Licensee 及关联方若经营 Model-as-a-Service，连续任意 12 个月合计收入超过 2,000 万美元，商业使用前须与 Moonshot 另签协议；
  - 商业产品/服务若超过 1 亿 MAU 或月收入超过 2,000 万美元，UI 必须显著展示 `Kimi K3`；内部使用豁免这两条。

### 2.2 参数与架构

| 字段 | 结论 | 证据 |
|---|---|---|
| 总参数 / 激活参数 | **2.8T / 104B** | A，官方模型卡/技术报告 |
| 主干层数 | **93**，第 0 层 dense，后 92 层 MoE | A，`num_hidden_layers=93`、`first_k_dense_replace=1` |
| 隐藏维 | **7168** | A，`hidden_size=7168` |
| MoE | **896 routed + 2 shared，top-16** | A，官方模型卡/config |
| LatentMoE | routed expert 输入/输出在 **3584** latent 宽上；单专家中间维 **3072** | A，`routed_expert_hidden_size=3584`、`moe_intermediate_size=3072` |
| 注意力 | **69 KDA + 24 Gated MLA** | A，官方模型卡；末层 93 也计入 full-attention 列表 |
| AttnRes | 每 12 层一个 attention-residual block | A，`attn_res_block_size=12` |
| 长上下文 | **1,048,576 tokens** | A |
| 激活 | SiTU-GLU | A |
| 原生多模态 | MoonViT-V2，27 层、hidden 1024、约 **401M** 参数；文本/图像/视频输入能力 | A，模型卡/config |
| 量化 | SFT 起做 QAT，MXFP4 weights / MXFP8 activations | A |

### 2.3 tokenizer 与输出头

- **A tokenizer：** 自定义 `TikTokenTokenizer`，核心文件 `tiktoken.model`（2,795,286 字节）和 `tokenization_kimi.py`；主词表 **163,840**。BOS/EOS/PAD 分别为 `[BOS]`/`[EOS]`/`[PAD]`；多模态还有 image placeholder token。
- **A 输出头不共享：** 顶层和 text config 都是 `tie_word_embeddings=false`。
- **B 头部体积：** 单个 embedding/head 为 `163840 * 7168 * 2 = 2,348,810,240` 字节（2.188 GiB，BF16）；两者及 norm 同在 `model-00094-of-000096.safetensors`，文件总长 4,697,664,072 字节。
- **兼容结论：** 词表 163,840 与 Qwen3-Coder-Next 的 151,936 不同，且隐藏宽 7168 vs 2048。输出头既不能按 token id 直接换，也不能用现有方阵桥接；首轮不提取。
- **Agent 协议风险：** 官方博客明确 K3 依赖 preserved thinking history；若 harness 不回传完整历史 reasoning，或会话中途从别的模型切换到 K3，质量可能剧烈不稳定。ColorLM 工具协议若借鉴 K3，需要把该历史语义纳入状态契约。

### 2.4 权重体积、分片、GGUF

- **A/B 官方 safetensors：** index 净载荷 `1,560,860,324,864` 字节（1,560.860 GB / 1,453.664 GiB）。96 个 safetensors 文件合计 **1,560,936,091,448** 字节；仓库已知文件总计 1,560,998,984,390 字节。
- **A 分片结构：** 96 片。`00001` 含首个 dense 层；`00002..00093` 基本每个 MoE 层一片；`00094` 是 embedding/norm/lm_head；`00095` 只含 3 个 `mm_projector` 张量；`00096` 只含 165 个 `vision_tower` 张量。
- **A 末层：** L92 的全部 5,401 个张量都在 `model-00093-of-000096.safetensors`，文件大小 **16,567,507,176** 字节（16.568 GB / 15.430 GiB）。
- **A 视觉独立分片：** `model-00095-of-000096.safetensors` = **92,289,328** 字节；`model-00096-of-000096.safetensors` = **802,448,352** 字节；合计 **894,737,680** 字节（894.738 MB / 853.29 MiB）。
- **C GGUF：** [unsloth/Kimi-K3-GGUF](https://huggingface.co/unsloth/Kimi-K3-GGUF) 已有完整多档分片：
  - `UD-IQ1_S`：14 片，593,997,933,024 字节（594.00 GB）；
  - `UD-IQ1_M`：15 片，648,872,012,448 字节；
  - `UD-IQ2_XXS`：16 片，711,067,773,664 字节；
  - `UD-Q2_K_XL`：19 片，861,277,858,912 字节；
  - `UD-Q4_K_XL`：32 片，1,508,668,683,104 字节；
  - `UD-Q8_K_XL`：34 片，1,561,157,884,384 字节；
  - 另有 BF16/F16/F32 mmproj，约 0.90/0.90/1.79 GB。

社区 GGUF 证明 llama.cpp 路径正在形成，但最小完整文本权重仍约 594 GB，不是 ColorLM 首轮可控 donor。

### 2.5 官方最低部署硬件

- **U Moonshot 模型卡未声明最低 GPU/VRAM。** 它只推荐 vLLM、SGLang、TokenSpeed。
- **A（推理引擎官方，不是模型作者最低承诺）：** vLLM recipe 对原生 MXFP4 checkpoint 估算 **1,680 GB 最低 VRAM**，默认硬件 B300；单/多节点 TP/TEP profile 从 8 GPU 起，H100 profile 标为 32 GPU。完整 1M context 还需 KV 和工作区。
- 因而报告应写“作者未披露最低；vLLM 当前 recipe 估算 1.68 TB/至少 8 卡，H100 32 卡”，不能写成“8 卡通吃”。
- 来源：[官方部署段](https://huggingface.co/moonshotai/Kimi-K3#5-deployment)、[vLLM Kimi K3 recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3)。

### 2.6 能力与基准

以下均来自 K3 官方模型卡/技术博客，K3 使用 `max` effort、`temperature=1.0`。单步任务通常 `top_p=0.95`，agentic 任务 `top_p=1.0`；不同 harness 不作严格横比。

| 能力 | Kimi K3 | 备注 |
|---|---:|---|
| 通用推理 | GPQA Diamond **93.5**；HLE-Full **43.5 / 56.0**（无/有工具） | 官方表 |
| 编程 | DeepSWE **67.5**；ProgramBench **77.8**；Terminal-Bench 2.1 **88.3**；FrontierSWE **81.2** | 多数用 Kimi Code harness |
| 长程编程/规划 | SWE-Marathon **42.0**；Kimi Code Bench 2.0 **72.9** | KCB 是内部集；SWE-Marathon 有公开任务但使用 H20 校准分支 |
| 搜索/研究 Agent | BrowseComp **91.2**；DeepSearchQA F1 **95.0**；ResearchRubrics **76.2** | BrowseComp 使用 300K 触发的 context compaction；无 compaction、完整 1M 时为 90.4 |
| 工具协议 | Toolathlon-Verified **76.5**；MCPMark-Verified **94.5**；MCP-Atlas **84.2** | MCP-Atlas 公开 500 题、100 turn |
| 通用 Agent | AutomationBench **30.8**；Agents' Last Exam **28.3**；APEX-Agents **41.0** | 官方表 |
| 办公/电脑操作 | OfficeQA Pro **63.3**；SpreadsheetBench 2 **34.8**；OSWorld-Verified **84.8**；OSWorld 2.0 **58.3** | K3 最关键的 ColorLM 互补项 |
| 视觉 | MMMU-Pro **81.6 / 83.4**（无/有 Python）；OmniDocBench **91.1**；Video-MME **90.0** | 原生视觉塔 |

来源：[HF Evaluation Results](https://huggingface.co/moonshotai/Kimi-K3#3-evaluation-results)、[Kimi K3 Tech Blog](https://www.kimi.com/blog/kimi-k3)。官方脚注中部分分数来自第三方榜单；本报告保留官方表口径，不把所有数字当同 harness 横比。

### 2.7 局部提取可行性与体积

官方 CDN 对 L92 分片 `HEAD` 返回 `Accept-Ranges: bytes`，`Content-Length=16567507176`。索引逐张量列出 497,220 个键；L92 的每个 expert 都有 `w1/w2/w3.weight_packed` 与 `.weight_scale`，所以可做精确 Range 活检。

**B 单个 K3 routed expert：**

- LatentMoE 宽 3584，中间维 3072；三矩阵参数总数 `3 * 3584 * 3072 = 33,030,144`。
- MXFP4 packed：`33,030,144 / 2 = 16,515,072` 字节。
- group size 32、uint8 scale：`33,030,144 / 32 = 1,032,192` 字节。
- 单专家合计 **17,547,264** 字节（16.734 MiB）。top-16 的专家净载荷为 280,756,224 字节。

**B 单专家最小可运行周边：**

- router：`896 * 7168 * 2 + 896 * 4 = 12,848,640` 字节；
- latent down/up projections：`2 * 7168 * 3584 * 2 = 102,760,448` 字节；
- routed latent norm：7,168 字节；
- 以上加 1 个 routed expert：**133,163,520** 字节（127.0 MiB）；
- 再加 2 个 shared experts（BF16，`3 * 7168 * 6144 * 2 = 264,241,152`）：**397,404,672** 字节（379.0 MiB）。

这只是张量净载荷。真正的 K3 层还依赖 7168 隐藏坐标、KDA/MLA、AttnRes 和 SiTU；“一个 expert 只有 17.5 MB”不等于它可脱离 latent projection 和训练分布独立表达能力。

**视觉路径更干净：** 视觉塔和 projector 已按命名空间独占 95/96 两片。视觉塔输出 1024 维，原 projector 映射到 K3 的 7168 维。ColorLM 可以保留 401M MoonViT-V2，另训 `1024 -> 2048` projector；这样无需引入 K3 文本主干、tokenizer 或 7168 隐藏桥。

### 2.8 与 Qwen3-Coder-Next 的互补性

- **能力互补极强：** Qwen donor 强在代码，K3 补原生视觉、OSWorld、办公自动化、研究/浏览、1M context 和长程 agent。
- **结构互补但不兼容：** Qwen 是 hidden 2048、expert width 512、top-10；K3 是 hidden 7168、latent 3584、expert 3072、top-16，还引入 KDA、Gated MLA、AttnRes、SiTU。现有 `2048x2048` 正交桥与 Qwen BF16 expert ABI 均不能复用。
- **许可/运维风险：** Kimi 自定义商业条款比 Qwen/DeepSeek 的 Apache/MIT 更复杂；完整模型 1.56 TB，推理 recipe 约 1.68 TB VRAM。它不适合直接成为第二个完整文本岛。
- **最佳互补切片：** 视觉塔 802.45 MB + projector 92.29 MB；其次是借鉴 preserved-thinking 状态契约、KDA/AttnRes/Stable LatentMoE 架构。文本 expert 仅在 2048<->3584/7168 桥有真实激活数据后再做。

### 2.9 决策

**整体类别：值得提取（仅视觉塔优先）；文本主干当前只值得借鉴架构。**

- 视觉部分满足“小于 1 GB、独立分片、电脑操作强互补”三个条件，值得进入审批清单。
- 文本末层整片 16.57 GB，单专家虽小但坐标和状态依赖太重；在没有 K3 激活桥、SiTU/MXFP4 runtime 和 K3 License 产品审查前，不应下载文本权重。

---

## 3. 审批后的精确下载清单

以下均为**提案，不是已执行下载**。文件大小来自 2026-07-31 HF `?blobs=true` API；真正请求前固定当前 commit SHA，避免 `main` 漂移。

### P0：现在只保存元数据，不取权重

这部分用于生成 byte-range 计划，体积很小，不含模型权重：

1. DeepSeek 0731：`config.json`、`tokenizer_config.json`、`tokenizer.json`、`model.safetensors.index.json`、`encoding/`、`inference/model.py`、`inference/kernel.py`、README、LICENSE。固定 revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`。
2. Kimi K3：`config.json`、`configuration_kimi_k3.py`、`modeling_kimi_k3.py`、`modeling_kimi_linear.py`、`tokenization_kimi.py`、`tiktoken.model`、`model.safetensors.index.json`、README、LICENSE。固定 revision：`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`。

### P1：推荐批准的权重

| 优先级 | repo@revision | 精确文件 | 字节 | 目的 | 是否必须整片 |
|---|---|---|---:|---|---|
| 1 | `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c...` | `model-00044-of-00048.safetensors` | **3,590,026,352** | L42 完整 MoE/attention/mHC 末层 donor；离线再选择 experts | 首次建议整片，避免无路由统计时任意挑 expert |
| 2 | `moonshotai/Kimi-K3@9f62e4...` | `model-00096-of-000096.safetensors` | **802,448,352** | 401M MoonViT-V2 视觉塔 | 是；该片仅含 `vision_tower.*` |
| 3 | 同上 | `model-00095-of-000096.safetensors` | **92,289,328** | 参考原 `mm_projector`；便于分析 1024->7168 对齐 | 是；仅 3 个 projector 张量 |

P1 合计：**4,484,764,032 字节**（4.485 GB / 4.177 GiB）。如果主线只批准一个实验，先 DeepSeek L42；如果目标优先补电脑操作，先 Kimi 95+96 两片（894,737,680 字节）。

### P2：条件式 Range 提取，不应直接整片

1. **DeepSeek L42 小胶囊：** 在 L42 header 中定位：
   - `layers.42.ffn.gate.weight`、`layers.42.ffn.gate.bias`；
   - 经路由统计选出的 6 个 `layers.42.ffn.experts.{eid}.{w1,w2,w3}.{weight,scale}`；
   - `layers.42.ffn.shared_experts.{w1,w2,w3}.{weight,scale}`。
   - 理论净载荷约 **132,645,888** 字节。**禁止在没有真实路由统计时把 0..5 当“高价值专家”。**
2. **Kimi L92 文本胶囊：** 只有在 2048<->3584 latent bridge 原型通过后，Range 取：
   - `language_model.model.layers.92.block_sparse_moe.gate.*`；
   - `routed_expert_{down,up}_proj.weight` 和 `routed_expert_norm.weight`；
   - 经真实路由统计选出的一个 expert 的 6 个 packed/scale 张量。
   - 理论净载荷 **133,163,520** 字节；若保留 shared experts 则 **397,404,672** 字节。

### P3：明确暂不下载

- DeepSeek `model-00045` output head（1,059,332,516 字节）：词表/隐藏宽双重不兼容。
- DeepSeek `model-00046..00048` DSpark（10,863,342,388 字节）：需要先实现三目标层 speculative 状态契约。
- Kimi `model-00093` 完整 L92（16,567,507,176 字节）：首轮过大，且文本 ABI 未就绪。
- Kimi `model-00094` embedding/head（4,697,664,072 字节）：163,840 词表和 7168 hidden 均不兼容。
- 任何完整官方权重、完整 GGUF 或社区 0731 占位 GGUF。

---

## 4. 最终分级摘要

| 候选/切片 | 分级 | 原因 |
|---|---|---|
| DeepSeek-V4-Flash-0731 L42 | **值得提取** | MIT；3.590 GB 可控；1M/agent/tool 与现有 coder donor 互补；需新矩形桥和 FP4 runtime |
| Kimi K3 vision tower + projector | **值得提取** | 小于 0.9 GB；独立分片；唯一直接补视觉电脑操作能力的强 donor |
| Kimi K3 文本 MoE/KDA/AttnRes | **只值得借鉴架构（当前）** | 1.56 TB、7168/3584 坐标、KDA 状态、SiTU、定制许可；孤立 expert 语义不可靠 |
| DeepSeek-V4-Flash preview / preview GGUF | **不值得投入** | 0731 已正式取代，权重能力较弱且容易误混版本 |

**推荐顺序：** DeepSeek L42 文本残差岛 -> Kimi MoonViT-V2 视觉前端 -> 真实激活驱动的 DeepSeek top-6 Range 活检 -> Kimi LatentMoE 原型。任何权重请求均应等待主线批准。

## 5. 关键动态 API 与可复核入口

- [DeepSeek 0731 官方仓 API（含 blob 大小）](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731?blobs=true)
- [Kimi K3 官方仓 API（含 blob 大小）](https://huggingface.co/api/models/moonshotai/Kimi-K3?blobs=true)
- [DeepSeek 0731 权重索引](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/model.safetensors.index.json)
- [Kimi K3 权重索引](https://huggingface.co/moonshotai/Kimi-K3/blob/main/model.safetensors.index.json)
- [DeepSeek 官方推理实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/inference)
- [Kimi K3 官方实现与技术报告](https://github.com/MoonshotAI/Kimi-K3)
