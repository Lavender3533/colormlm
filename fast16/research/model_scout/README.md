# ColorLM 模型侦察包

**状态：`HOLD - NO WEIGHTS DOWNLOADED`**  
**截止日期：** 2026-07-31（Asia/Shanghai）

本目录是 ColorLM 通用对话、推理、编程、工具、长上下文、规划和电脑操作 donor 的开放权重侦察结果。排名按“可控切片给 ColorLM 带来的增量/接入成本”计算，不等同于整模能力排名。

## 先看这里

1. [CANDIDATE_RANKING.md](CANDIDATE_RANKING.md)：跨系列总排名、三档结论和能力覆盖建议。
2. [DOWNLOAD_PLAN.md](DOWNLOAD_PLAN.md)：固定 revision、文件名、字节数和 SHA-256 的待批准清单。
3. [METHODOLOGY.md](METHODOLOGY.md)：证据等级、ColorLM 兼容基线和体积估算口径。

## 证据报告

- [deepseek_kimi.md](deepseek_kimi.md)：DeepSeek-V4-Flash-0731、Kimi K3。
- [glm_qwen_evidence.md](glm_qwen_evidence.md)：GLM-5/5.2、GLM-4.7-Flash、Qwen3.6、Qwen3.5、Qwen-AgentWorld。
- [OTHERS_EVIDENCE.md](OTHERS_EVIDENCE.md)：MiniMax-M3、Step-3.7-Flash、MiMo-V2.5-Pro、Fara1.5-27B。

## 当前结论

- 首批唯一建议是 Qwen3.6-35B-A3B 的 shard 26，共 2,231,416,848 B；先验证 tokenizer 字节同构、坐标对齐、NLL 和精确回退。
- 文本第二阶段首选 DeepSeek-V4-Flash-0731 L42；hidden 4096 桥和 FP4 解码通过评审后才放行。
- 电脑操作首选 Fara1.5-27B 视觉 shard；Kimi K3 视觉前端是能力更广但许可更复杂的备选。
- GLM-5.2、MiniMax-M3、MiMo-V2.5-Pro 当前只吸收架构思路，不申请权重。

本轮没有下载权重，没有启停本地模型或占用 GPU，没有运行 CMake；未修改 `PROJECT_STATE.md` 或 `fast16/research/v19_dual_head/`。
