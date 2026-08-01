# ColorLM 并行 B 组交接

日期：2026-07-31

## 本组边界

本组只完成第二 donor 候选筛选、多能力短门设计和输出头降本算法推演。未启动或停止模型，未运行
CMake，未调用 GPU，未修改 `PROJECT_STATE.md`、`fast16/research/v19_dual_head/` 或现有运行时。
全部新产物都在 `fast16/research/parallel_b/`。

## 三项结论

### 1. 第二 donor：当前无人直接准入

- **近期可测候选：Kimi K3。** 它是唯一同时拥有本地真实权重切片、运行接入和反事实 NLL 正信号
  的候选。首轮只允许把已有 L12 固定双胶囊叠到 v17 做冻结 A/B，不使用学习路由。
- **新连续岛审计首选：GLM-5。** 其工具、中文、通用推理和长程 Agent 目标最可能补充
  Coder-Next，MIT 权重且当前 llama.cpp 有 GLM-DSA 整模路径；但没有本地深层桥或因果能力证据。
- **工程备选：MiMo-V2-Flash。** 4096 宽、15B active、现成 MIMO2/SWA 图，连续块成本最可能
  受控；能力与 Coder-Next 重叠更高。
- **暂缓：DeepSeek-V3.2。** 7168 宽、MLA/稀疏注意力成本较高，当前互补证据不足。

任何候选都没有同时通过连续性、深层可运输性、运行可执行性、相对 v17 互补性和单位成本五门。
完整证据与硬门见 `SECOND_DONOR_SCREEN.md`，机器可读评分见 `donor_candidate_matrix.json`。

### 2. 多能力短门：8 维 16 题，严格无回归

`multicap_short_gate_v1.json` 已冻结推理、知识、长上下文、代码、工具、规划、电脑操作、自然交流
八维，每维两题。每题都带精确 validator 和预先选择的关键决策点。

执行顺序固定：

1. G0：alpha=0 物理旁路和运行参数等价。
2. G1：关键决策 token 的强制 no-op/donor NLL；不使用连续开头 token。
3. G2：G1 通过后才贪心生成，并做精确离线判分。

K3 目标维度是 `tools + planning`。生成准入要求至少修正一题、目标维度净增、且没有任何基线
正确题变错。至少三个维度净增才可描述为“多维改善”。阈值和停止条件见
`MULTICAP_SHORT_GATE.md`。

判分命令：

```powershell
python -X utf8 fast16/research/parallel_b/validate_multicap_gate.py compare `
  --baseline <v17.responses.jsonl> `
  --candidate <k3.responses.jsonl> `
  --target-dimension tools `
  --target-dimension planning
```

判分器只覆盖 G2；G0/G1 必须由运行报告单独给出。

### 3. 输出头降本：只保留带证书的级联方案

当前 v19.1 独立 smoke60 已失败，因此本方案不得先行实施。稠密输出头以后通过冻结能力门后，
唯一推荐是：

```text
中心化 Q6_K 输出矩阵
  -> r64 低秩 Q8/F16 侦察头
  -> 逐行残差范数上界
  -> 最多 512 个候选的原始 Q6_K 精确行分页
  -> top-1 上界证书；不能认证则回退稠密头
```

推荐静态点 `r64-c512-cache8192`：估算 `9,604,864 MAC/token`，为当前稠密头的 `3.56%`；
估算 GPU 常驻 `22.98 MiB`，全 miss 行上传 `0.820 MiB/token`。这些是算术上限，不是 token/s
实测。温度采样、top-p 和 teacher NLL 首版全部回退稠密头。

算法、误差界和验证门见 `OUTPUT_HEAD_COST_REDUCTION.md`。复算命令：

```powershell
python -X utf8 fast16/research/parallel_b/output_head_cost_model.py
```

## 产物索引

| 文件 | 用途 |
|---|---|
| `donor_candidate_matrix.json` | 四个候选的官方快照、本地证据、加权分和硬门状态 |
| `SECOND_DONOR_SCREEN.md` | 筛选结论、证据边界、K3 最小实验与 GLM-5 审计清单 |
| `multicap_short_gate_v1.json` | 16 题冻结题库、validator 和关键决策点 |
| `validate_multicap_gate.py` | 结果 JSONL 的纯离线精确评分与配对比较 |
| `MULTICAP_SHORT_GATE.md` | G0/G1/G2 契约、阈值、记录字段和停止条件 |
| `output_head_cost_model.py` | 当前稠密头与三个级联点的静态成本复算 |
| `OUTPUT_HEAD_COST_REDUCTION.md` | r64 侦察、误差上界、精确行分页与回退算法 |
| `HANDOFF.md` | 本交接摘要 |

## 下一执行顺序

1. 主线先决定是否允许在 v17 上复用现有 K3 L12 双胶囊做一次冻结 A/B；不改变 alpha 和题库。
2. 允许后，从 16 题参考输出生成 tokenizer 绑定的关键 token teacher，保留完整前缀和题库哈希。
3. 先跑 G0/G1。任一 LOTO 非改善、工具/规划组不通过或单题回归超过阈值，立即停止，不做生成。
4. G1 通过后跑一次 G2，用判分器出报告；任何 pass→fail 都拒绝 K3。
5. K3 得出结论后，才做 GLM-5 的只读连续块结构审计；GLM-5 不过预算则转 MiMo。
6. 只有稠密 v19 输出头先通过多能力门，才离线拟合 r64 因子并测证书回退率。

## 已验证

- 两个 JSON 均可严格解析：候选 4 项，短门 16 项。
- 两个 Python 文件均通过 AST 语法解析，未生成或加载模型资产。
- 16/16 合成正确响应通过；Markdown 围栏 JSON 和工具额外文本被正确拒绝。
- A/B 比较器的“目标维度净增且零回归”正路径与“一题回归即拒绝”负路径均通过。
- 候选加权分按声明权重复算一致。
- 输出头脚本复现现有包 `222,169,248` 字节和稠密 `269,541,376 MAC/token`。
- 目录内所有文件均通过严格 UTF-8 解码。

## 剩余风险

- K3 的历史 72-token 集合参与过站点选择，且基线不是 v17；只能作为候选依据，不能复用为晋级证据。
- 16 题短门只测原子、可判定能力；通过后仍不能宣称仓库级编码、真实电脑操作或前沿通用能力。
- 官方参数和许可证为 2026-07-31 的模型页快照，实际提取前需再次锁定 revision 与文件哈希。
- 低秩残差范数界可能过松；若 r96 仍无法把稠密 fallback 压到 1%，应停止方案，不取消证书。

