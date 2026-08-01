# ColorLM donor 候选总排名

**截止日期：** 2026-07-31（Asia/Shanghai）  
**排名口径：** ColorLM 可控提取的投资回报，不是整模聊天榜。能力、许可、最小可归因切片、隐藏维/词表兼容性和现有 Qwen3-Coder-Next donor 的增量同时计分。

> 所有权重动作均为 `HOLD - NO WEIGHTS DOWNLOADED`。不同模型的官方基准 harness、推理预算和上下文设置不同，分数只用于确认能力覆盖，不做跨表硬排序。

## 总排名

| 排名 | 候选与目标部件 | 许可 | 关键结构 | 最小建议文件 / 真正目标净载荷 | 对 Coder-Next 的主要互补 | 决策 |
|---:|---|---|---|---|---|---|
| 1 | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 输出头/末层半片 | Apache-2.0 | 35B/3B，40 层，hidden 2048，256+1 experts/top-8 | 2.231 GB shard / 1.017 GB head | 通用对话、规划、preserved thinking、agentic/frontend coding；与 ColorLM 同为 248,320 词表体系 | **值得提取** |
| 2 | [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) L42 | MIT | 核心 284B/13B，43 层，hidden 4096，256+1/top-6，1M | 3.590 GB 完整层 / 132.646 MB router+top-6+shared 理论值 | 通用推理、工具、长上下文、长程 agent | **值得提取** |
| 3 | [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) L46 | MIT | 30B/3B，47+1 层，hidden 2048，64+1/top-4 | 2.539 GB 单 shard / 单 expert 约 18.9 MB | Browse、tau2、轻量通用 agent；hidden 无需维度桥 | **值得提取** |
| 4 | [Fara1.5-27B](https://huggingface.co/microsoft/Fara1.5-27B) 视觉塔+merger | MIT | 27.36B dense，64 层，hidden 5120；视觉 hidden 1152 | 1.309 GB shard / 约 0.93 GB 视觉 tensor | 截图理解、像素定位、浏览器动作、关键点安全停顿 | **值得提取** |
| 5 | [Step-3.7-Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash) L44 或视觉栈 | Apache-2.0 | 198B/约11B，45+3 层，hidden 4096，288+1/top-8，256K | 9.245 GB L44；或 3.962 GB 视觉栈；单 expert 约 31.5 MB | 工具轨迹完整性、Agent、视觉搜索/验证 | **值得提取**，但排在 4096 桥评审之后 |
| 6 | [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) MoonViT-V2+projector | Kimi K3 License，非 MIT/Apache | 2.8T/104B，93 层，hidden 7168，896+2/top-16；视觉 hidden 1024 | 0.895 GB 两片 | OSWorld、办公、浏览、原生图像/视频、1M agent | **值得提取视觉前端**；文本干只借鉴架构，且先审许可 |
| 7 | [Qwen-AgentWorld-35B-A3B](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B) simulator/critic 支路 | Apache-2.0 | 35B/3B，40 层，hidden 2048，256+1/top-8 | 3.890 GB 完整 L39+head | MCP/Search/Terminal/Web/Android/OS 环境动力学 | **值得提取（隔离实验）**；禁止替代 action policy |
| 8 | [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) DSA/IndexShare | MIT | 744B/40B，78 层，hidden 6144，256+1/top-8，1M | L77 文件集 21.454 GB；单 expert 约 75.5 MB | 长程 coding、agent/tool、1M sparse attention | **只值得借鉴架构** |
| 9 | [MiMo-V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) SWA/GA+MTP | MIT | 1.02T/42B，70 层，hidden 6144，384/top-8，1M | 单 expert/层 Range 约 38 MB；文件级末层约 1.031 TB | 1M 规划、千次工具调用轨迹 | **只值得借鉴架构** |
| 10 | [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) MSA/多模态设计 | MiniMax Community License，非 OSI | 428B/23B，60 层，hidden 6144，128+1/top-4，1M | 单 expert 57-113 MB；末层文件 12.25-16.10 GB | 图像/视频、Cowork、长程终端 | **只值得借鉴架构**；书面许可前不取权重 |
| 11 | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Apache-2.0 | 27B dense，64 层，hidden 5120 | 末层文件集 8.425 GB；head 2.543 GB | 新版通用/agentic 后训练 | **只值得借鉴架构**；被同代 35B-A3B 的兼容性压倒 |

## 能力覆盖结论

| ColorLM 目标 | 首选 donor | 备选 | 原因 |
|---|---|---|---|
| 通用对话 / 推理 / 规划 | Qwen3.6-35B-A3B | DeepSeek-V4-Flash-0731、GLM-4.7-Flash | Qwen 首片风险最低；DeepSeek 的 1M/agent 增量最大；GLM 是 hidden 2048 的 MIT 备选 |
| 编程 | 继续保留 Qwen3-Coder-Next | Qwen3.6、Step-3.7 | 新 donor 应补通用性和轨迹稳定性，不能因同类 SWE 分数重复购买代码能力 |
| 工具 / Agent | DeepSeek-V4-Flash-0731 | Step-3.7、AgentWorld 辅助支路 | DeepSeek 强互补；Step 有工具轨迹证据；AgentWorld 只预测 observation |
| 长上下文 | DeepSeek-V4-Flash-0731 | GLM-5.2、MiMo-V2.5-Pro | DeepSeek 兼顾 1M 与可控末层；后二者先借鉴稀疏注意力和状态接口 |
| 电脑操作 | Fara1.5-27B 视觉栈 | Kimi K3 视觉前端 | Fara 是 MIT 且专注截图 CUA；Kimi 指标更广但自定义许可，且只取视觉塔不能复制完整动作策略 |

## 不值得投入

- `DeepSeek-V4-Flash` preview 及其 GGUF：已被 0731 正式版取代，且 preview GGUF 不能冒充 0731。
- `GLM-5` 原版：GLM-5.2 已正式取代，参数/接入成本相同而能力更旧。
- Qwen3.5 通用系列：同代 Qwen3.6 已提供更好的后训练能力；35B-A3B 还有更小的首片。
- MiniMax M2.x、MiMo-V2-Flash、Fara1.5-4B/9B：被同系列当前强版本取代，不再重复建 donor 路线。
- 完整 checkpoint、完整社区 GGUF、未固定 revision 的权重 URL：不满足可归因和可复现要求。

## 十项字段覆盖索引

每个主候选的官方链接/日期/许可、参数结构、tokenizer/head tying、权重/分片/GGUF、官方部署硬件、能力基准、局部提取可行性、预计体积/兼容难点、与 Coder-Next 互补性和三档结论，分别记录在：

- DeepSeek、Kimi：[deepseek_kimi.md](deepseek_kimi.md)
- GLM、Qwen、AgentWorld：[glm_qwen_evidence.md](glm_qwen_evidence.md)
- MiniMax、Step、MiMo、Fara：[OTHERS_EVIDENCE.md](OTHERS_EVIDENCE.md)

方法和证据等级见 [METHODOLOGY.md](METHODOLOGY.md)，审批文件见 [DOWNLOAD_PLAN.md](DOWNLOAD_PLAN.md)。
