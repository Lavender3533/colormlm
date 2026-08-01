# ColorLM 供体侦察：MiniMax、Step、MiMo 与电脑操作模型

**截止时间：** 2026-07-31（Asia/Shanghai）  
**调查范围：** MiniMax 最新开放权重、StepFun、Xiaomi MiMo，以及 DeepSeek/Qwen/GLM/Kimi 之外与 ColorLM 最互补的 MIT 模型。  
**证据约定：** `[官方]` 为模型卡/官方公告/官方代码；`[配置推导]` 为 `config.json` 或 safetensors index 可复核的结论；`[社区]` 只证明 GGUF/量化已存在，不背书质量或可运行性；`[未披露]` 表示官方材料没有给出。

> **本轮没有下载任何权重，没有向权重 URL 发起 Range 请求。主线批准前，本文末尾的清单仅是下载提案，不是执行指令。**

## 候选排名与结论

| 排名 | 候选 | 开放许可 | 主要互补性 | 可控提取点 | 结论 |
|---:|---|---|---|---|---|
| 1 | `stepfun-ai/Step-3.7-Flash` | Apache-2.0 | 工具轨迹完整性、Agent、视觉、256K，同时保持强编码 | 独立末层 shard 9.245 GB；官方 GGUF；视觉塔独立 3.962 GB | **值得提取** |
| 2 | `microsoft/Fara1.5-27B` | MIT | 视觉定位、电脑/浏览器操作、关键点安全停顿 | 完整视觉栈集中在 1.309 GB 末 shard；词表大小与 ColorLM 同为 248,320 | **值得提取**（先视觉栈，不先提头） |
| 3 | `MiniMaxAI/MiniMax-M3` | MiniMax Community License，非 OSI | 原生图像/视频、1M MSA、Cowork/长程 Agent | 单专家 tensor 可 Range，但文件级末层 12.25–16.10 GB，且商业派生有许可风险 | **只值得借鉴架构**（除非先获书面授权） |
| 4 | `XiaomiMiMo/MiMo-V2.5-Pro` | MIT | 1M 上下文、千次工具调用级轨迹、强编码/规划 | MTP 独立 2.464 GB；单专家可 Range，但专家并行分片不利于文件级末层提取 | **只值得借鉴架构**（能力与 Coder-Next 重叠偏高） |

三档显式汇总：

- **值得提取：** Step-3.7-Flash、Fara1.5-27B。
- **只值得借鉴架构：** MiniMax-M3、MiMo-V2.5-Pro。
- **不值得投入：** 本子组的四个完整候选中暂无；MiniMax M2.x、MiMo-V2-Flash 和 Fara1.5-4B/9B 是被新代/大规格替代的去重型号，未冒充成额外完整候选卡。

排名不表示已经通过 donor 准入。ColorLM 本地主干/Coder-Next 隐藏宽度为 2,048；四个候选的文本宽度分别为 4,096/5,120/6,144/6,144，均必须先做真实深层激活桥和冻结反事实门，不能凭榜单直接装配。

---

## 1. MiniMax-M3

### 1.1 链接、日期与许可

- `[官方]` 权重/模型卡：<https://huggingface.co/MiniMaxAI/MiniMax-M3> 。HF 仓库 `createdAt=2026-06-02T07:49:31Z`，本文以 **2026-06-02** 作为权重发布日。官方 GitHub：<https://github.com/MiniMax-AI/MiniMax-M3>。
- `[官方 API]` 本文的 BF16 文件/索引快照固定于 revision **`f0e1c1e04d40177e4673a22097036854f536e9c0`**；MXFP8 快照固定于 **`c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`**。
- `[官方]` 许可全文：<https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE>。这是 **MiniMax Community License**，不是 MIT/Apache，也不是 OSI 标准开源许可。
- `[官方]` 商业限制：商用产品需显著展示 `Built with MiniMax M3`；年收入不超过 2,000 万美元需一次性邮件通知，超过则需先取得书面授权；许可还禁止军事用途等。提取层/头/专家并形成 ColorLM 派生物不会规避这些条款。

### 1.2 架构参数

- `[官方]` 约 **428B 总参数 / 23B 激活参数**，原生文本+图像+视频，1M context。
- `[配置推导]` 文本干：60 层，隐藏维 6,144，前 3 层 dense、后 57 层 MoE；128 个 routed experts，top-4，再加 1 个 shared expert；MoE/shared intermediate 3,072，dense intermediate 12,288；64 Q heads / 4 KV heads，head dim 128。
- `[配置推导]` MSA：每块 128 token，选 top-16 blocks，4 个 index heads，index dim 128；前 3 层不用 sparse attention，后 57 层使用。视觉塔 32 层、hidden 1,280、16 heads，project 到 6,144。
- `[官方]` 模型卡声称 MSA 在 1M 上下文相对 M2 达到 9× prefill、15× decode，每 token 计算量降到 1/20。

### 1.3 tokenizer 与输出头

- `[配置推导]` `PreTrainedTokenizerFast`，vocab 200,064，仓库同时提供 `vocab.json` + `merges.txt` + `tokenizer.json`，因而是 BPE 资产形式；定义 FIM、代码仓库、function call、image/video 等专用 token。
- `[配置推导]` `tie_word_embeddings=false`；index 中 `language_model.lm_head.weight` 和 `language_model.model.embed_tokens.weight` 是两个独立 tensor。
- `[配置推导]` tokenizer 配置的 `model_max_length=40,960,000` 与模型配置的 1,048,576 不一致；部署应以模型配置/模型卡的 **1M** 为能力上限，不应将 tokenizer 上限解读为 40.96M 模型上下文。

### 1.4 权重大小、分片与量化

- `[官方 API/索引]` BF16：59 个 safetensors shard，HF 文件列表合计 **854,176,398,808 B**；index `metadata.total_size=869,157,697,024 B`。两者存在约 15.0 GB 差异，官方未解释；真正下载预算应用文件列表总和，逻辑 tensor 总量另行保留。
- `[官方]` `MiniMax-M3-MXFP8`：<https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8>，31 shards，文件列表合计 **443,749,077,256 B**，index 逻辑总量 451,543,283,200 B。
- `[官方合作方]` NVIDIA NVFP4：<https://huggingface.co/nvidia/MiniMax-M3-NVFP4>，88 shards，250,103,762,320 B；AMD MXFP4：<https://huggingface.co/amd/MiniMax-M3-MXFP4>，59 shards，242,666,026,728 B。
- `[社区]` GGUF 存在：<https://huggingface.co/bartowski/MiniMax-M3-GGUF>。例：IQ1_S 3 片/90,527,490,400 B，IQ2_XXS 3 片/116,611,867,008 B，IQ4_XS 6 片/229,685,754,592 B，Q8_0 12 片/453,611,164,768 B，社区 mmproj 约 1.73 GB。这只证明有转换物，不证明 MSA/视频在 ColorLM 当前运行时可用。

### 1.5 官方最低部署硬件

- `[未披露]` MiniMax 模型卡没有给出一个通用的“最低内存/VRAM”数字。
- `[官方模型卡链接的部署方案]` KTransformers/KT-Kernel 教程：<https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/MiniMax-M3-Tutorial.md>。其最小示例是 **1× H20 96 GB + CPU 专家 offload**，但 CPU RAM 下限未披露；推荐吞吐方案为 8× H20/H100。AMD ATOM 官方方案用 4× MI355。因此“单96GB GPU可启动”不等于整个 444 GB MXFP8 权重只需 96 GB 总内存。

### 1.6 基准：代码、工具/Agent、推理、长上下文

- `[官方 eval metadata]` SWE-bench Verified **80.5**，SWE-bench Pro **59.0**（Claude Code scaffold，4 次平均）。
- `[官方 eval metadata]` Claw-Eval **74.5**，Apex-Agents **27.7**，YC-Bench 2.1M（资金类指标，不是百分比）。
- `[官方 eval metadata]` MMMU-Pro **78.1**，Video-MME (w/ sub) **85.4**。
- `[官方仓库收录的外部榜]` Long-Horizon-Terminal-Bench 2026-07-16：mean reward×100 = **38.5**，solved@0.95 = 3/46。
- `[未披露]` 模型卡没有给出可与其他候选直接对齐的 AIME/GPQA/HLE 纯推理表，也没有公布 1M 长上下文准确率曲线；官方强证据主要是 MSA 效率与长程 Agent 成绩。

官方 eval 文件：<https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/.eval_results/minimax-m3.yaml> 与 <https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/.eval_results/lhtb.yaml>。

### 1.7 能否只下载头/末层/路由器/少量专家

- `[索引实证]` BF16 头+输入 embedding+最终 norm 都在 `model-00001-of-00059.safetensors`，文件 **5,583,706,344 B**。头 tensor 理论 BF16 体积为 `200064×6144×2 = 2,458,386,432 B`；标准 `hf download` 不能只取该 tensor，批准后可读 safetensors header 再作精确 Range。
- `[索引实证]` 末层 L59 在 BF16 `model-00059-of-00059.safetensors`，**16,099,232,552 B**；MXFP8 L59 在 `model-00029-of-00031.safetensors`，**12,246,524,552 B**。
- `[配置推导]` 单个 routed expert 有 3 个 `6144×3072` 矩阵，BF16 理论约 **113,246,208 B**，MXFP8 权重下限约 56.6 MB 再加 scale；M3 将每个 expert 存为独立 tensor，因而批准后可精确 Range 一个专家。
- `[索引实证]` 57 层 router 分散在几乎所有主干 shard；文件级获取等于约 837.7 GB BF16/435.4 GB MXFP8，只有 tensor Range 才合理。不带对应 expert 时，单独 router 也没有可迁移语义。

### 1.8 体积、ColorLM 难点、互补性与结论

- **预计提取体积：** 单 expert 0.057–0.113 GB；单 router 约 1.57 MB；但要形成一层连续块，最小文件级代价是 12.25 GB MXFP8 或 16.10 GB BF16。
- **兼容难点：** 6,144→2,048 深层桥；200,064→248,320 tokenizer/头映射；MSA index branch/top-block 状态；Gemma norm、QK norm、部分 RoPE、视觉/视频输入管线；且 MiniMax 商业派生许可是硬门。
- **对 Coder-Next 的互补：** 原生图像/视频、Cowork、1M 稀疏注意力和长程终端能力很强；编码本身与 Coder-Next 有重叠。
- **明确结论：只值得借鉴架构。** 优先借鉴 MSA 的 index branch/block selection 和原生多模态训练思路；在得到商业派生的书面许可意见前，不建议下载任何 M3 权重切片。

---

## 2. Step-3.7-Flash

### 2.1 链接、日期与许可

- `[官方]` 模型卡：<https://huggingface.co/stepfun-ai/Step-3.7-Flash>；公告：<https://static.stepfun.com/blog/step-3.7-flash/>。官方公告页标注 **2026-05-29**（HF 仓库先于 2026-05-23 创建）。
- `[官方 API]` BF16 快照 revision **`5f6244077ac62e04eec3f320501ff8c2b293373a`**；FP8 **`b3d7916fccac844cca050d7520f2aaa513f9a84f`**；NVFP4 **`4275532ffd9a9496ff36b7a2dc4a9db1048da438`**；官方 GGUF **`0b69336d2fd2adfdef9c66e425f7778196c31482`**。
- `[官方]` **Apache License 2.0**；模型卡 front matter 与底部声明一致。对 ColorLM donor 的商业可用性明显优于 M3。

### 2.2 架构参数

- `[官方]` **198B 总参数 / 约 11B 激活**；其中语言主干约 196B，视觉 encoder 约 1.8B；context 256K。
- `[配置推导]` 主干 45 层，hidden 4,096；L0–L2 为 dense，L3–L44 为 42 个 MoE 层；288 routed experts，top-8，每层再有 shared expert；routed/shared intermediate 均为 1,280，dense intermediate 11,264。
- `[配置推导]` 64 个全注意头/8 groups/head dim 128；局部注意配置使用 96 heads/8 groups，sliding window 512；每 4 层 1 层 full attention；head-wise attention gate、QK norm、分层 RoPE。
- `[配置推导]` 主干后还有 3 层 MTP（index 中 L45–L47）。视觉 encoder 47 层，hidden 1,536，16 heads，patch 14，输入尺寸 728。

### 2.3 tokenizer 与输出头

- `[配置推导]` `LlamaTokenizerFast`，vocab 128,896，含 thinking、FIM、tool-call/tool-output 专用 token。模型配置为 262,144 context，tokenizer 文件仍写 131,072；官方模型卡确认可用上限为 256K。
- `[索引实证]` `lm_head.weight` 和 `model.embed_tokens.weight` 为两个独立实体 tensor，因此该 checkpoint **不共享输出头与 embedding**；顶层 config 未显式写 `tie_word_embeddings`，本结论依据权重索引而不是默认值。

### 2.4 权重大小、分片与量化

- `[官方]` BF16 仓库：24 个语言 safetensors + 2 个 ViT safetensors，合计 **402,730,833,632 B**；index 逻辑总量 402,730,656,512 B。
- `[官方]` FP8：<https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8>，26 个权重文件，**212,523,666,872 B**；NVFP4：<https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4>，13 主干 shard + MTP，合计 **129,241,604,232 B**。
- `[官方]` 官方 GGUF：<https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF>。BF16 9 片/394,017,423,776 B；IQ3_XXS 2 片/75,758,680,960 B；Q3_K_M 3 片/93,801,444,352 B；IQ4_XS 3 片/104,993,562,624 B；Q4_K_S 3 片/111,499,087,872 B；Q8_0 5 片/209,417,871,264 B。另有 3.973 GB mmproj、6.974 GB BF16 MTP 和 3.707 GB Q8 MTP。

### 2.5 官方最低部署硬件

- `[官方]` llama.cpp 章节明确：Q4_K_S 语言模型 111.5 GB + FP16 mmproj 3.97 GB + 约 7 GB runtime overhead，**最低统一内存/VRAM 120 GB，推荐 128 GB**。官方举例为 Mac Studio、NVIDIA DGX Station、AMD Ryzen AI Max+ 395 系统。
- `[官方]` 数据中心示例：BF16/FP8 用 TP8，NVFP4 用 TP4；模型卡未将特定 GPU SKU 声明为硬性最低值。

### 2.6 基准：代码、工具/Agent、推理、长上下文

- `[官方]` 工具/Agent：ClawEval-1.1 **67.1**，Toolathlon **49.5**，HLE with Tool **48.1**。
- `[官方]` 编码/终端：SWE-Bench Pro **56.3**，Terminal-Bench 2.1 **59.5**。
- `[官方]` 专业与视觉：GDPVal-AA **45.8**，SimpleVQA (Search) **79.2**，V* (Python) **95.3**。
- `[官方]` 支持 low/medium/high 三档推理，但模型卡没有单独给出无工具 HLE/GPQA/AIME 数字。`[未披露]` 也没有 128K/256K needle/GraphWalks 准确率曲线；256K 是架构/产品上限，不是长文准确率证明。

### 2.7 能否只下载头/末层/路由器/少量专家

- `[索引实证]` `model-00024.safetensors` 为 **6,968,188,464 B**，内含输出头、embedding、final norm 和 3 层 MTP。单头 BF16 理论体积 `128896×4096×2 = 1,055,916,032 B`。标准下载只能取整 shard；批准后 Range 可只取头 tensor。
- `[索引实证]` 最后主干层 L44 单独放在 `model-00023.safetensors`，**9,245,052,456 B**；L42–L43 在 `model-00022.safetensors`，18,624,846,976 B。因而连续 L42–L44 的文件级下载量是 **27,869,899,432 B**，分片边界很干净。
- `[配置推导]` 一个 BF16 routed expert 约 `3×4096×1280×2 = 31,457,280 B`。但 288 个 expert 在每层被打包为 3 个大 tensor，需要知道 tensor shape/offset 后再按 expert 所在 axis-0 切片作 Range；普通 HF 文件下载无法只取一个 expert。
- `[索引实证]` 全部 router 跟各 MoE 层同 shard 分布，文件级下载接近全主干；单层 router 理论约 2.36 MB，但必须等批准后用 safetensors header 获取精确偏移。
- `[索引实证]` 视觉塔与 projector 独立为 `model-vit-00001.safetensors` 1,613,990,904 B + `model-vit-00002.safetensors` 2,348,122,376 B，合计 **3,962,113,280 B**，可不下载语言主干。

### 2.8 体积、ColorLM 难点、互补性与结论

- **预计提取体积：** 最低可独立测的末层文件 9.245 GB；三层连续岛 27.870 GB；单 expert Range 约 31.5 MB；视觉栈 3.962 GB；头 tensor 约 1.056 GB，但头所在整 shard 6.968 GB。
- **兼容难点：** 4,096→2,048 深层桥；128,896→248,320 token map；混合 64/96-head attention、head gate、分层 RoPE、SWA state、MTP；视觉 projector 输出 4,096，仍需视觉→ColorLM 桥。有利点是 Apache-2.0、官方 GGUF 和 llama.cpp 支持已经存在。
- **对 Coder-Next 的互补：** 编码有重叠，但 ClawEval/Toolathlon 的轨迹稳定性、视觉搜索/验证和 256K 明显补齐通用 Agent 面。
- **明确结论：值得提取。** 第一阶段只提 L44 或视觉栈，必须在冻结任务集上证明相对 Coder-Next 的 tools/planning/vision 净增益；未通过前不扩大到 27.87 GB 三层岛。

---

## 3. MiMo-V2.5-Pro

### 3.1 链接、日期与许可

- `[官方]` 模型卡：<https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro>；公告：<https://mimo.xiaomi.com/mimo-v2-5-pro>。官方公告日期 **2026-04-27**。
- `[官方 API]` 本文的文件/索引快照固定于 revision **`21d1ecfecd7bd70f31be25ca49d7edd21f003659`**。
- `[官方]` **MIT License**。

### 3.2 架构参数

- `[官方]` **1.02T 总参数 / 42B 激活**，1M context，训练语料 27T tokens，原生 32K 训练后扩到 1M。
- `[官方/配置]` 70 层（1 dense + 69 MoE），hidden 6,144，dense intermediate 16,384，MoE intermediate 2,048；384 routed experts，top-8，无 shared expert。
- `[官方/配置]` 128 Q heads / 8 KV heads，QK dim 192、V dim 128；10 层 global attention + 60 层 sliding-window attention，约 1:6 交错，window 128，带 attention sink bias；3 层 MTP。

### 3.3 tokenizer 与输出头

- `[配置推导]` `Qwen2Tokenizer`，BPE，vocab 152,576；含 tool call/response、thinking、FIM/repository token。tokenizer 文件写 131,272，模型配置与官方卡确认 1,048,576 context。
- `[配置推导]` `tie_word_embeddings=false`，输出头与 embedding 不共享。

### 3.4 权重大小、分片与量化

- `[官方 API/索引]` checkpoint 是 FP8 E4M3 mixed，32 个 expert-parallel 主 shard，EP0 再分成 2 个 shard，加 `model_mtp.safetensors`，共 34 个 safetensors；文件列表合计 **1,033,389,872,152 B**，index 逻辑总量 1,033,369,538,304 B。
- `[官方]` 当前 Xiaomi org 没有 MiMo-V2.5-Pro 官方 GGUF/更低比特量化仓库。
- `[社区]` GGUF 存在：<https://huggingface.co/unsloth/MiMo-V2.5-Pro-GGUF>。例：UD-IQ1_M 8 片/304,091,696,608 B，UD-IQ2_M 8 片/317,269,937,600 B，UD-IQ4_XS 11 片/490,697,584,416 B，MXFP4_MOE 14 片/610,512,613,600 B，Q8_0 24 片/1,087,618,889,024 B。

### 3.5 官方最低部署硬件

- `[官方]` Xiaomi 模型卡给出的参考拓扑是 `dp=2, ep=16, tp=16`（多节点），但没有把具体 GPU SKU/显存写成“最低要求”。
- `[模型卡链接的官方 vLLM recipe]` <https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5-Pro> 的单节点前置硬件是 **8× H200 (TP8)**。这是已验证的小型拓扑，不代表官方证明了更小设备不可行。

### 3.6 基准：代码、工具/Agent、推理、长上下文

- `[官方公告]` 通用 Agent：Claw-Eval pass^3 **63.8**，GDPVal-AA **1581 Elo**，τ³-bench **72.9**。
- `[官方公告]` 推理/工具：Humanity's Last Exam **34.0** 无工具 / **48.0** 有工具。
- `[官方公告]` 代码/Agent：SWE-Bench Pro **57.2**，SWE-bench Verified **78.9**，MiMo Coding Bench **73.7**，Terminal-Bench 2.0 **68.4**，FrontierSWE (Impl.) rank **3.4**（越低越好）。
- `[官方]` 基座表还报告 GPQA-Diamond 66.7、LiveCodeBench v6 39.6、SWE-Bench AgentLess 35.7、C-Eval 91.5 等；这些是 **Base** 设定，不与上面 post-training Agent 成绩混用。
- `[官方]` GraphWalks：512K 为 BFS **0.56** / Parents **0.92**；1M 仍有 **0.37 / 0.62**。这是四个候选中最清楚的 1M 准确率证据。

### 3.7 能否只下载头/末层/路由器/少量专家

- `[索引实证]` `lm_head.weight` 在 `model_pp0_ep0_shard1.safetensors`，整文件 **27,180,576,088 B**；头本身若为 BF16，几何下限为 `152576×6144×2 = 1,874,853,888 B`。embedding 在 `model_pp0_ep0_shard0.safetensors` 34,554,911,640 B。必须用批准后的 tensor Range 才值得单取头。
- `[索引实证]` MTP 是干净的独立文件 `model_mtp.safetensors`，**2,463,641,280 B**；它主要带来解码加速，不是独立能力 donor。
- `[索引实证]` 一个完整末层的 384 专家分布在 32 个 EP shard 中，文件级获取 L69 需要主权重约 **1,030,926,230,872 B**，不可接受。
- `[配置/索引推导]` 每个 EP shard 放约 12 个 expert 在全部 69 个 MoE 层的权重。单 expert/单层的 FP8 主矩阵下限为 `3×6144×2048 = 37,748,736 B` 再加 scale；单 expert 跨 69 层约 2.60 GB。因 tensor 按 expert 分开，批准后的精确 Range 可行；标准下载则需 31.26–34.55 GB 整个 EP shard。
- `[索引实证]` 69 个 router 集中在 EP0 的两个 shard，文件级为 61,735,487,728 B；若 router 为 BF16，权重 tensor 总量约 325.6 MB。不配同一套 expert ID 时不应单独迁移。

### 3.8 体积、ColorLM 难点、互补性与结论

- **预计提取体积：** 单 expert/层 Range 约 38 MB，单 expert 跨全部 MoE 层约 2.6 GB，MTP 整文件 2.464 GB；文件级的完整末层约 1.031 TB，完全不适合。
- **兼容难点：** 6,144→2,048 桥；152,576→248,320 tokenizer map；FP8 block scale、expert-parallel checkpoint 重排、fused QKV、非对称 QK/V head dim、SWA/GA 状态与 attention sink；ColorLM 当前虽有 MIMO2 整模图路径，但不等于 Pro 版深层岛已可接。
- **对 Coder-Next 的互补：** 1M 轨迹稳定性、规划、长程工具调用有互补；但代码/SWE/Terminal 能力与 Coder-Next 高度重叠，且参数和桥宽代价大。
- **明确结论：只值得借鉴架构。** 借鉴点为 6:1 SWA/GA、128-window + sink bias、三层 MTP 和长程 MOPD 后训练。如后续冻结反事实证明其 planning/tools 专家有独立增益，再提升为少量 expert Range 提取候选。

---

## 4. Microsoft Fara1.5-27B

### 4.1 链接、日期与许可

- `[官方]` 模型卡：<https://huggingface.co/microsoft/Fara1.5-27B>；官方 harness：<https://github.com/microsoft/fara>；论文：<https://huggingface.co/papers/2606.20785>。
- `[官方 API]` 本文的 27B 文件/索引快照固定于 revision **`299c8406a6c6256d45ec200d1ac12b34c5599d9b`**。
- `[官方元数据冲突]` 模型卡表格写“Release date: **2026-05-21**”，但 27B HF 仓库 `createdAt=2026-07-17T20:09:59Z`。本文同时保留两个日期：5 月 21 日是官方卡所称 Fara1.5 发布日，7 月 17 日是 **27B 权重仓可验证的公开日**。
- `[官方]` **MIT License**。

### 4.2 架构参数

- `[官方]` 27B dense 多模态 decoder-only CUA，基于 `Qwen/Qwen3.5-27B` SFT；context 262,144，训练期 2026-01 至 2026-04，训练计算 64× B200 / 6 天。
- `[索引实证]` 精确总参数 **27,356,728,560**，dense 模型因而激活参数约等于总参数；无 MoE experts/top-k。
- `[配置推导]` 文本干 64 层，hidden 5,120，intermediate 17,408，24 Q heads / 4 KV heads，head dim 256；每 4 层 1 层 full attention，其余 3 层是 linear/Gated-Delta 路径；1 层 MTP 配置。
- `[配置推导]` 视觉塔 27 层，hidden 1,152，intermediate 4,304，16 heads，patch 16，project 到 5,120。模型只看截图，不读 DOM/accessibility tree；操作通过 XML-tagged `computer_use` 工具调用输出。

### 4.3 tokenizer 与输出头

- `[配置推导]` `Qwen2Tokenizer`，vocab **248,320**，`Qwen3VLProcessor`；输出头与 embedding 都独立，`tie_word_embeddings=false`。
- `[本地可复核]` ColorLM v19 token-map 中 base vocab 同为 **248,320**；这是四个候选中唯一个词表大小直接对齐的模型。仍需逐 token 验证 ID/字节串一致，不能仅凭大小相同宣称 tokenizer 完全相同。
- `[官方]` 点击/输入/滚动等不使用独立 action head，而是普通 LM 头生成 XML + JSON 参数。因此只提 lm_head 不会自动获得电脑操作能力。

### 4.4 权重大小、分片与量化

- `[官方 API/索引]` BF16：10 个 safetensors shard，文件列表合计 **54,713,606,240 B**，index `total_size=54,713,457,120 B`。
- `[官方]` Microsoft org 截至访问日没有 Fara1.5-27B 官方量化/GGUF。
- `[社区]` GGUF：<https://huggingface.co/bartowski/Fara1.5-27B-GGUF>。例：IQ2_M 10,634,372,928 B，Q4_K_S 16,474,163,008 B，Q4_K_M 17,533,552,448 B，Q8_0 28,665,067,328 B，BF16 GGUF 2 片合计 53,808,281,344 B，mmproj 约 0.928 GB。

### 4.5 官方最低部署硬件

- `[官方]` 需要能容纳 27B BF16 的 GPU 组，已测 A6000/A100/H100/B200；**建议至少 2 张 GPU 分片**。官方没有给出单卡最小 VRAM 或量化部署下限。
- `[官方]` 建议在 MagenticLite/Docker 沙箱中运行，带域名 allow-list、watch mode 和即时 pause；训练常用分辨率 1440×900。

### 4.6 基准：代码、工具/Agent、推理、长上下文

- `[官方]` Fara1.5-27B：WebVoyager **89.3**，Online-Mind2Web **72.3**，WebTailBench outcome success **40.2**。9B 对应 86.6/63.4/32.3，4B 对应 80.8/57.3/27.4。
- `[官方]` 官方卡没有报告 SWE-Bench、Terminal-Bench、AIME/GPQA/HLE 或专门长上下文基准。它的能力证据非常专用：视觉网页操作、坐标 grounding、长轨迹动作预测与关键点安全停顿。
- `[官方]` 局限：只主打英文和浏览器；多步误差会累积，会受页面 prompt injection/欺骗性视觉内容影响。

### 4.7 能否只下载头/末层/路由器/少量专家

- `[索引实证]` 头和 embedding 都在 `model-00001-of-00010.safetensors`，整文件 **5,958,075,952 B**；每个 tensor 的 BF16 几何体积是 `248320×5120×2 = 2,542,796,800 B`。可在批准后用 Range 单取，但不建议将头作为第一个 CUA donor 试验。
- `[索引实证]` 完整视觉塔+视觉 merger 全在 `model-00010-of-00010.safetensors`，整文件只有 **1,309,493,264 B**，同时夹带 final norm 和 L63 的部分 tensor。社区 mmproj 约 0.928 GB，可用作视觉部分的体积交叉检查，但不是官方 tensor 精确总和。
- `[索引实证]` L63 跨 `model-00009-of-00010.safetensors` 和 shard 10，最小文件级下载量 **7,146,136,056 B**；L62 在 shard 9。没有 router/expert，因为该模型是 dense。

### 4.8 体积、ColorLM 难点、互补性与结论

- **预计提取体积：** 视觉 shard 1.309 GB，经 safetensors header 精确筛选后视觉 tensor 约 0.93 GB 级；末层文件级 7.146 GB；单头 Range 2.543 GB。
- **兼容难点：** 5,120→2,048 桥；视觉 projector 原本输出 5,120；3:1 linear/full attention 的 recurrent/conv state；ColorLM 需新增截图、像素座标和 CUA harness；工具 XML schema/系统 prompt/关键点停顿规则是能力契约的一部分。
- **有利兼容点：** MIT；Qwen 系列；vocab size 与 ColorLM 同为 248,320；视觉栈恰好独立落在 1.309 GB shard，是本轮最干净的电脑操作类提取边界。
- **对 Coder-Next 的互补：** 几乎正交：Coder-Next 强在文本编码/终端，Fara 强在截图感知、像素定位、浏览器动作与人机安全停顿。
- **明确结论：值得提取。** 第一个候选是视觉 shard/merger，不是 lm_head。但必须说清：单纯视觉塔只提供截图表示，Fara 的顺序动作策略分布在全 LM；只提视觉栈不能宣称 ColorLM 已经获得 Fara 的 CUA 成功率。

---

## 5. 精确下载提案（待主线批准）

### 5.1 P0：先批准 Step 单末层，不下头

| 仓库（固定 revision） | 文件 | 精确大小 | 目的 | 是否本轮下载 |
|---|---|---:|---|---|
| `stepfun-ai/Step-3.7-Flash@5f6244077ac62e04eec3f320501ff8c2b293373a` | `model-00023.safetensors` | 9,245,052,456 B | 完整 L44：attention + norms + router + shared expert + 288 routed experts | **否，待批准** |
| 同上 | `config.json` | 6,300 B | 架构契约 | 只是元数据，已在线核验，未落地 |
| 同上 | `model.safetensors.index.json` | 119,419 B | tensor→shard 契约 | 只是元数据，已在线核验，未落地 |
| 同上 | `configuration_step3p7.py` + `modeling_step3p7.py` | 8,375 + 56,815 B | 审计图和张量布局 | 只是代码，未落地 |
| 同上 | `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` + `chat_template.jinja` | 9,976,972 + 163,405 + 468 + 5,723 B | tokenizer/工具协议对齐 | 非权重，待与 P0 一起批准落地 |

**P0 不包含** `model-00024.safetensors`：Step 头的 vocab/hid 均不匹配 ColorLM，且该 shard 夹带 embedding+MTP；尚无证据表明它比末层更互补。

### 5.2 P1：批准后的 Step 扩展，二选一

- **连续岛方向：** 增加 `model-00022.safetensors` 18,624,846,976 B，形成 L42–L44，总下载 27,869,899,432 B。
- **多模态方向：** 改为下载 `model-vit-00001.safetensors` 1,613,990,904 B + `model-vit-00002.safetensors` 2,348,122,376 B，总计 3,962,113,280 B。

批准后的固定 URL（**本轮未请求**）：

```text
https://huggingface.co/stepfun-ai/Step-3.7-Flash/resolve/5f6244077ac62e04eec3f320501ff8c2b293373a/model-00023.safetensors
https://huggingface.co/stepfun-ai/Step-3.7-Flash/resolve/5f6244077ac62e04eec3f320501ff8c2b293373a/model-00022.safetensors
https://huggingface.co/stepfun-ai/Step-3.7-Flash/resolve/5f6244077ac62e04eec3f320501ff8c2b293373a/model-vit-00001.safetensors
https://huggingface.co/stepfun-ai/Step-3.7-Flash/resolve/5f6244077ac62e04eec3f320501ff8c2b293373a/model-vit-00002.safetensors
```

两个方向不应同时启动，否则无法归因新能力来自连续语言层还是视觉输入。

### 5.3 P0 并行的低体积 CUA 候选：Fara 视觉 shard

| 仓库（固定 revision） | 文件 | 精确大小 | 目的 | 是否本轮下载 |
|---|---|---:|---|---|
| `microsoft/Fara1.5-27B@299c8406a6c6256d45ec200d1ac12b34c5599d9b` | `model-00010-of-00010.safetensors` | 1,309,493,264 B | 完整视觉塔+merger，以及可丢弃的少量末层附带 tensor | **否，待批准** |
| 同上 | `config.json` | 3,646 B | Qwen3.5/vision 契约 | 元数据，未落地 |
| 同上 | `model.safetensors.index.json` | 111,110 B | 视觉 tensor 完整性 | 元数据，已在线核验，未落地 |
| 同上 | `processor_config.json` + `preprocessor_config.json` | 1,190 + 390 B | 图像尺寸/patch 契约 | 非权重，待批准落地 |
| 同上 | `tokenizer.json` + `tokenizer_config.json` + `chat_template.jinja` + `vocab.json` + `merges.txt` | 19,989,343 + 1,183 + 7,756 + 6,722,759 + 3,353,259 B | 验证 248,320 vocab 的逐 token 同构性和 CUA 协议 | 非权重，待批准落地 |

批准后的固定 URL（**本轮未请求**）：

```text
https://huggingface.co/microsoft/Fara1.5-27B/resolve/299c8406a6c6256d45ec200d1ac12b34c5599d9b/model-00010-of-00010.safetensors
```

**P0 不包含** `model-00001-of-00010.safetensors`：CUA 操作策略不是一个独立 action head，对 5,120 维 lm_head 做 2,048 维映射在没有视觉轨迹激活桥时缺少因果依据。

### 5.4 当前明确不下载

- MiniMax-M3 任何权重/量化/GGUF：先解决派生商用许可与 MSA 运行时门。
- MiMo-V2.5-Pro 任何权重：先用元数据与冻结任务证明它相对 Coder-Next 不是重复购买代码能力。
- Step 整模 GGUF/FP8/NVFP4：当前 donor 目标是可归因的层/视觉树切片，不是再增一个 75–129 GB 以上的整模。

### 5.5 Range 精确性边界

本轮严格禁止对权重 URL 做 Range，因而本文已给出 **精确文件名、文件字节数和目标 tensor 名**，但没有伪造 tensor 起止字节偏移。主线如批准“只读取权重 shard 的 safetensors header”，下一步才能生成带 commit SHA、URL、`Range: bytes=start-end`、期望 SHA256 和理论 tensor shape 的真正字节级清单。

---

## 6. 来源索引（全部访问于 2026-07-31）

### MiniMax-M3

- 模型卡：<https://huggingface.co/MiniMaxAI/MiniMax-M3>
- 配置：<https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json>
- 文件列表：<https://huggingface.co/api/models/MiniMaxAI/MiniMax-M3?blobs=true>
- 索引：<https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/model.safetensors.index.json>
- 许可：<https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE>
- 评测：<https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/.eval_results/minimax-m3.yaml>
- LHTB：<https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/.eval_results/lhtb.yaml>
- MSA 报告：<https://arxiv.org/abs/2606.13392>
- MXFP8：<https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8>
- KTransformers 部署：<https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/MiniMax-M3-Tutorial.md>

### Step-3.7-Flash

- 公告：<https://static.stepfun.com/blog/step-3.7-flash/>
- 模型卡：<https://huggingface.co/stepfun-ai/Step-3.7-Flash>
- 配置：<https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/main/config.json>
- 文件列表：<https://huggingface.co/api/models/stepfun-ai/Step-3.7-Flash?blobs=true>
- 索引：<https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/main/model.safetensors.index.json>
- 官方 GGUF：<https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF>
- FP8：<https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8>
- NVFP4：<https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4>

### MiMo-V2.5-Pro

- 公告：<https://mimo.xiaomi.com/mimo-v2-5-pro>
- 模型卡：<https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro>
- 配置：<https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro/blob/main/config.json>
- 文件列表：<https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.5-Pro?blobs=true>
- 索引：<https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro/blob/main/model.safetensors.index.json>
- vLLM recipe：<https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.5-Pro>
- 社区 GGUF：<https://huggingface.co/unsloth/MiMo-V2.5-Pro-GGUF>

### Fara1.5-27B

- 模型卡：<https://huggingface.co/microsoft/Fara1.5-27B>
- 配置：<https://huggingface.co/microsoft/Fara1.5-27B/blob/main/config.json>
- 文件列表：<https://huggingface.co/api/models/microsoft/Fara1.5-27B?blobs=true>
- 索引：<https://huggingface.co/microsoft/Fara1.5-27B/blob/main/model.safetensors.index.json>
- 官方 harness：<https://github.com/microsoft/fara>
- 论文：<https://huggingface.co/papers/2606.20785>
- 社区 GGUF：<https://huggingface.co/bartowski/Fara1.5-27B-GGUF>
