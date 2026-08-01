# K3 / 强教师前端轨迹蒸馏方案

## 能力声明边界

本目录没有启动或测试模型，没有使用 GPU，也没有读取/修改权重。**现有切片没有被证明具备 K3 前端能力，本方案也不构成这种证明。**“K3/强教师”在本文仅指未来可通过冻结门筛选的教师候选层级，不代表仓库中已有 K3 切片已达到该层级。

## 目标

让学生学会从需求约束出发，产生任务特异、可响应、可交互、可访问且低外部依赖的单文件前端，并能根据静态/浏览器反馈做局部修复。训练目标不是模仿教师措辞或堆叠 CSS，而是提升冻结 validation/blind 上相对普通三卡片基线的任务成功率。

## 教师准入

教师候选必须先在与学生相同的提示、输出上限和运行环境下评估：

1. train 仅用于采样策略和格式调试；不得用于宣称能力。
2. validation 至少 6/8 通过，8/8 关键条件通过，中位模板惩罚不高于 6。
3. blind 开封前冻结教师版本、系统提示、采样参数、运行时与评分器哈希；blind 只跑一次。
4. 与 `fixtures/ordinary_three_cards.html` 比较：中位静态分增益至少 12，至少 7/8 任务增益不低于 10，且无负增益。
5. 教师还需通过未来的浏览器行为检查；仅静态高分不能成为强教师。

未满足这些条件的输出可作为负例或修复例，不能标记 `teacher_accepted=true`。

## 可观测轨迹格式

不蒸馏私有思维链，只保存可审计的工作产品与动作：

```json
{
  "schema_version": "frontend-trajectory-v1",
  "task_id": "pf47-train-01",
  "teacher_id": "future-teacher-hash",
  "frozen_generation": {"temperature": 0, "seed": 47, "template_id": "..."},
  "steps": [
    {"kind": "requirements", "artifact": {"states": [], "breakpoints": [], "a11y": []}},
    {"kind": "design_contract", "artifact": {"tokens": {}, "layout": {}, "interactions": []}},
    {"kind": "html_snapshot", "sha256": "...", "content": "..."},
    {"kind": "static_feedback", "report": {}},
    {"kind": "repair_patch", "target_findings": [], "patch": "..."},
    {"kind": "final_evidence", "static": {}, "browser": null}
  ],
  "accepted": false
}
```

`requirements` 只能复述可见需求；`repair_patch` 必须指出它修复的 finding id；`browser` 为 null 时明确表示没有动态证据。训练输入不得包含 validation/blind 的参考实现、正则或判分阈值。

## 数据组成

建议第一阶段 8,000–20,000 条短轨迹，按下列比例而非按单一“漂亮页面”堆量：

- 35% 从零实现：覆盖表格、图、表单、工作流、编辑器、监控台等不同信息架构。
- 25% 定向修复：alt/label/focus/reduced-motion、横向溢出、死按钮、空链接、第三方依赖。
- 20% 反模板偏好对：同一需求下“普通三卡片”负例对“任务特异实现”正例。
- 10% 响应式变换：桌面结构转移动结构，要求信息等价而非单纯 `display:none`。
- 10% 交互状态机：空/加载/错误/成功/撤销/恢复路径。

每条轨迹保存源提示哈希、教师/运行时哈希、最终 HTML 哈希、静态报告和（未来）浏览器回放摘要。去重同时看 prompt MinHash、DOM 标签序列、CSS 选择器集合和截图感知哈希，防止大量同模板改文案。

## 训练阶段

1. **结构 SFT**：只用通过 UTF-8、骨架、任务信号和无禁用模式的轨迹，学习需求清单、设计契约与首版 HTML。
2. **修复 SFT**：输入首版 HTML + 机器 findings，目标为最小修复补丁和修后 HTML；禁止把评分阈值直接写入提示。
3. **偏好优化**：以任务成功、模板惩罚、可访问性和依赖安全构造多目标偏好对。不得仅按最终总分排序，以免学生学会堆 CSS。
4. **轨迹压缩**：学生先预测结构化需求/状态清单，再生成 HTML；部署时可隐藏中间结构，但必须保留可选审计输出。
5. **回归蒸馏**：加入故意带空链接、无标签表单、emoji 图标、无断点三卡片等 hard negatives，要求识别并修复。

## 防污染与模板隔离

- train/validation/blind 的 24 个 `template_family` 物理不重叠，哈希由 `MANIFEST.json` 冻结。
- 训练仅可读取 `data/train.jsonl`；validation 只用于版本选择，不能回流训练；blind 在候选冻结后由独立操作者开封。
- 任何读取 validation/blind 正则、阈值或人工答案后的再训练都会使该 split 作废。
- 不能按 blind 失败题补丁后重跑同一版本；必须新建题族与新版本。
- 教师与学生使用同一前后处理；关键词路由、检索参考 HTML 或人工修补不能记为模型能力。

## 分阶段晋级

| 阶段 | 所需证据 | 允许结论 |
|---|---|---|
| A 静态开发 | train 结果与自测 | 接线正常，不代表泛化 |
| B 静态验证 | validation ≥6/8、关键项全过、相对基线规则全过 | 有超越普通三卡片的验证证据 |
| C 静态盲测 | 冻结后 blind ≥6/8，规则全过 | 在本 24 门契约上有盲测证据 |
| D 动态盲测 | 视口、键盘、交互、axe、视觉成对盲评 | 可讨论受限前端能力 |
| E 外部复核 | 独立题集/环境复现 | 才能讨论可迁移能力 |

即使完成 E，也不能从“前端短门”推导为 K3 全能力、长期 Agent 或完整软件工程能力。

