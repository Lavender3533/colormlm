# ColorLM 多能力短门 v1

## 目标

这套短门回答一个窄问题：强制打开第二 donor 后，是否在其目标能力上修正 v17 的稳定失败，且
没有损伤其他能力。它不用于宣称前沿水平，也不把模型自述、开头连续 token 或输出变化当作能力。

题库 `multicap_short_gate_v1.json` 覆盖 8 维、每维 2 题，共 16 题：

| 维度 | 两类探针 | 判分方式 |
|---|---|---|
| reasoning | 状态转换、唯一约束排列 | 精确 JSON |
| knowledge | 冻结事实定位、冲突时不猜 | 精确 JSON 与引用 |
| long_context | 跨段双标记、字段最新修订 | 精确 JSON |
| coding | Python 嵌套边界、Rust 闭区间 | 精确修复行 |
| tools | 文件读取、带目录和 glob 的搜索 | 工具名、全部参数、停止原因 |
| planning | 部署依赖、预算优化 | 精确动作序列/选择 |
| computer_use | 保存对话框、筛选菜单 | 离线 UI 状态到动作 JSON |
| communication | 严格字段、术语与简洁偏好 | 精确 JSON/文本 |

题目故意短、可判定、低成本。它只覆盖原子决策，不替代仓库级编码、真实桌面或长程 Agent 任务。

## 冻结协议

1. 比较路径固定为 `A=v17`、`B=v17 + 强制 donor`；K3 首轮不使用学习路由。
2. 固定模型二进制、Coder 岛、chat template、上下文、seed、贪心采样、输出上限和批形状。
3. `decision_points` 在任何候选 NLL 产生前冻结；不能按 B 的结果重新挑 token。
4. 每题独立 context。knowledge 题只能用题内冻结事实；computer_use 题不控制真实桌面。
5. 第一轮仅跑一次 A/B。失败先归因，不换随机种子刷分。

## 三阶段门

### G0：链路与旁路

- donor 关闭时不加载权重、不建图节点；固定输出逐 token 等同 v17。
- 16 个任务的输入哈希、运行参数和结果记录完整。
- 任一项不满足，后续结果无效。

### G1：关键决策 token 的反事实 NLL

对每题参考答案中的 `decision_points` 建 teacher prefix，分别采集强制 no-op 与强制 donor 的
pre-sampling NLL。`delta = NLL_donor - NLL_v17`，负值表示改善。

K3 的首轮门固定为：

- 16 题中至少 10 题 task mean `delta < 0`；
- `tools` 与 `planning` 两组均值都 `< 0`，且每组至少 3/4 题改善（按关键点计）；
- 其余任一维度组均值不得 `> +0.01 nat`；
- 最差单题平均回归不得 `> +0.05 nat`；
- 所有 leave-one-task-out 的剩余任务总均值都 `< 0`。

样本很小，不使用显著性话术。这个门只拒绝明显依赖单题/单 token 的候选。

### G2：实际生成

G1 通过后才生成。结果按 `response_record_schema` 写成 JSONL，再用离线判分器比较：

```powershell
python -X utf8 fast16/research/parallel_b/validate_multicap_gate.py compare `
  --baseline <v17.responses.jsonl> `
  --candidate <donor.responses.jsonl> `
  --target-dimension tools `
  --target-dimension planning
```

第二 donor 准入要求：所有 16 题记录齐全；至少一个基线失败被修正；没有任何基线正确题改错；
目标维度不回归且合计至少净增 1 题。至少 3 个维度净增才允许写“多维改善”，否则只能描述目标
能力改善。判分器的 `generation_gate_pass` 也不包含 G0/G1，三门必须同时通过。

## 运行后记录

每个任务至少记录：路径 A/B、任务 ID、最终文本或结构化工具调用、finish reason、生成 token 数、
墙钟、pre-sampling NLL、donor 强制状态、alpha 和所有运行参数。性能只在能力通过后做一次相邻
128-token 检查；速度不能挽救能力失败。

## 立即停止条件

- 工具调用出现空必填参数、错误工具或未以 `tool_calls` 停止；
- communication/knowledge 的校准题从正确变为猜测；
- 任一基线正确题被 donor 改错；
- LOTO 方向存在非改善；
- 结果依赖修改题目、alpha、参考答案或关键 token；
- 关闭路径不再逐 token 等同 v17。

