# ColorLM 第二 donor 候选筛选

日期：2026-07-31

## 结论

当前没有候选可直接进入正式装配。四个候选都至少缺少“相对 v17 的独立互补增益”和“真实深层
入口/出口桥”之一，不能用官方榜单、总参数量或整模 `llama.cpp` 支持代替这两项因果证据。

研究顺序固定为：

1. **近期可测第二 donor：Kimi K3。** 只复用已有 L12 固定双胶囊，在 v17 上跑冻结的八维短门。
   它是唯一已有本地反事实 NLL 正信号和可消费运行资产的候选，但现有资产不是连续层块。
2. **新连续岛结构审计：GLM-5。** 它在工具、中文、通用推理和长程 Agent 上最可能补充
   Coder-Next，且当前 `llama.cpp` 已有 GLM-DSA 整模路径。审计不等于准入，不下载整模。
3. **工程备选：MiMo-V2-Flash。** 如果 GLM-5 的 DSA 状态或单块预算不可控，MiMo 的 4096 宽、
   15B active 和现成 MIMO2/SWA 图更容易做出连续块，但能力与 Coder-Next 重叠更高。
4. **暂缓：DeepSeek-V3.2。** MLA 路径成熟，但 7168 宽和能力重叠使其当前单位迁移价值最低。

机器可读评分见 `donor_candidate_matrix.json`。加权分只排研究顺序；任何硬门失败都直接否决。

## 证据边界

### Kimi K3

已有证据：v13 的 72 个反事实 teacher token 上，L12 K3 固定图相对 no-op 的平均 NLL 从
`2.487283` 降到 `2.438846`，12 个任务中 8 个改善，工具组 5/5 改善；运行权重约
`190,871,552` 字节。7168→2048 词嵌入运输和真实 MXFP4 latent 专家已落地。

边界：该 teacher 参与过 K3 站点选择，基线也不是当前 v17；两个宏胶囊是孤立专家，不保留
KDA、AttnRes 或连续层状态。K3 的原生视觉能力还依赖视觉塔和输入路径，纯文本切片不能继承。

### GLM-5

已有证据：官方开放 MIT 权重，744B/40B active、6144 宽、78 层、256 专家 top-8，目标包括
复杂系统工程、推理、工具和长程 Agent。当前源码已有 `LLM_ARCH_GLM_DSA`、MLA、DSA indexer、
MoE 和 744B.A40B 类型路径。

边界：整模能转换/执行不等于能把 3–4 层作为有界状态岛接入 ColorLM。当前没有本地 GLM-5
张量索引、深层激活、桥、切片成本或相对 v17 的任务翻转。

### MiMo-V2-Flash 与 DeepSeek-V3.2

MiMo 的 309B/15B active、4096 宽和 SWA/全注意力交替结构最利于控制工程成本，且源码已有
MIMO2 图；但官方能力重心仍偏代码/Agent，需先证明不是重复购买 Coder-Next 能力。

DeepSeek-V3.2 有 MIT 权重和现成 DeepSeek2/MLA 路径，但 7168 宽、稀疏注意力与约 37B active
使桥和状态成本都较高；在没有独立推理/工具翻转前暂缓。

## 五个硬门

| 硬门 | 通过定义 | 当前结果 |
|---|---|---|
| 连续性 | 3–4 个相邻层，注意力/线性状态、Norm、MoE 与残差顺序完整 | 四者均未通过；K3 现有资产明确不连续 |
| 深层可运输性 | 用真实主干/供体同语义深层状态拟合，整 prompt LOTO 和多切分稳定 | 四者均未通过 |
| 可执行性 | 状态归属 context/sequence；关键算子在 Vulkan；alpha=0 不建分支 | 只有整模或孤立胶囊的部分证据 |
| 互补性 | 冻结短门中修正至少一个 v17 稳定失败，且没有任一维度回归 | 四者相对 v17 均未知 |
| 单位成本 | 能力增益、额外 GiB、token/s 和冷页流量同时记录 | 仅 K3 孤立胶囊成本已知 |

所以筛选结果不是“选中即构建”，而是把 K3 送入能力门、把 GLM-5 送入只读结构审计。

## K3 的最小准入实验

固定两条路径：`A=v17`，`B=v17 + K3 L12 固定双胶囊`。禁止任务关键词门控，禁止按结果调 alpha，
禁止同时改变 Coder 岛、输出头、chat template、seed 或上下文。

1. 在 `multicap_short_gate_v1.json` 的 16 个任务上做强制 no-op/K3 精确 next-token NLL。
2. 每题只统计预先冻结的答案/动作关键 token，不使用连续开头 token 冒充决策点。
3. NLL 通过后才做贪心生成；用 `validate_multicap_gate.py` 离线精确判分。
4. K3 目标维度暂定 `tools + planning`：两维都不得低于 v17，且至少一维净增 1 题。
5. 任一基线正确题被 K3 改错、所有 LOTO 方向不为改善、工具名/参数/停止行为回归，立即停止。

通过仅意味着“现有 K3 固定支路有资格作为第二 donor 候选”，不意味着连续 K3 岛已经成立。

## GLM-5 的只读结构审计清单

在 K3 能力门没有结论前不做权重提取。后续若获准，只核对：连续层张量完备性、DSA indexer
状态依赖、MLA KV 生命周期、每层共享/路由专家字节、3–4 层 Q4/Q6 估算、6144↔2048 桥成本、
以及现有 Vulkan 算子的岛内复用点。任何一项要求 CPU 全量回退或突破 8GB 显存路径即转 MiMo。

## 来源

- 本地：`PROJECT_STATE.md`、`fast16/V17_DONOR_SCREEN.md`、
  `fast16/research/v13_causal_sparse_report.json`、`fast16/research/KIMI_K3_EXPERT_CAPSULE.md`。
- 官方：[GLM-5](https://huggingface.co/zai-org/GLM-5)、
  [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)、
  [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)、
  [MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)。

