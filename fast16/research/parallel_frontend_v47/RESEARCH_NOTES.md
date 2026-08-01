# ColorLM v47 并行前端评测研究笔记

## 结论先行

前端生成不能只测“HTML 能否打开”或“是否有三张卡片”。可靠的短测至少要把六类证据分开：结构正确性、响应式、真实行为线索、视觉编排、依赖风险、可访问性；再单独惩罚低信息量的通用模板。静态评分适合快速筛选和复现，但不能替代浏览器渲染、交互回放、视觉盲评或真实用户任务成功率。

本轮网络检索被用户要求停止，未完成对用户点名的 **DesignBench、FronTalk、ArtifactsBench** 原始论文页面核验。因此本文只把这三个名字作为“待核验的评测线索”，不写作者、发布日期、样本量或榜单数字，也不声称本评分器复现了它们。下面采用的是能够明确指向公开标准/论文的共通方法思想。

## 可核验的公开方法线索

1. **Design2Code: How Far Are We From Automating Front-End Engineering?**（2024，公开预印本）把截图到实现的前端生成视为结构与视觉共同问题。对本项目的启发是：DOM/CSS 静态正确性只能覆盖一部分，最终仍需截图相似度与人工视觉偏好。
2. **WebSight: A Vision-Language Dataset for Learning Vision-Based Web Coding**（2024，公开数据/论文）强调网页截图与代码的成对监督。对蒸馏的启发是：保留“需求—实现—渲染证据”的配对，而非只存最终 HTML。
3. **WebArena: A Realistic Web Environment for Building Autonomous Agents** 与 **VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks** 把成功定义在可执行任务轨迹，而不是页面外观。对本项目的启发是：按钮存在不等于能用，必须另设浏览器行为门；本轮评分器只检查事件接线线索，不伪称任务成功。
4. **W3C WCAG 2.2**、**WAI-ARIA Authoring Practices Guide** 与 **WHATWG HTML Living Standard** 提供可访问名称、焦点、语义元素、减少运动、表单标签等可核验规则。这些规则被静态契约吸收，但颜色对比度与级联后状态仍留给浏览器审计。

公开入口（需在可联网环境复核版本号）：

- Design2Code：<https://arxiv.org/abs/2403.03163>
- WebSight：<https://arxiv.org/abs/2403.09029>
- WebArena：<https://arxiv.org/abs/2307.13854>
- VisualWebArena：<https://arxiv.org/abs/2401.13649>
- WCAG 2.2：<https://www.w3.org/TR/WCAG22/>
- ARIA APG：<https://www.w3.org/WAI/ARIA/apg/>
- HTML Living Standard：<https://html.spec.whatwg.org/>

## 本地六样本实测

评分器版本为 `parallel-frontend-static-v1.0.0`。结果来自 `sample_audit_report.json`，同一字节输入可复现。

| 样本 | 最终分 | 毛分 | 模板惩罚 | 结构 | 响应式 | 交互 | 视觉 | 依赖 | 可访问性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| index.html | 76.85 | 79.85 | 3 | 17 | 14 | 11.25 | 15.60 | 10 | 12 |
| index.v17版本产出.html | 75.55 | 83.55 | 8 | 18 | 14 | 11.25 | 15.80 | 10 | 14.50 |
| index.v19测试版产出.html | 67.55 | 78.55 | 11 | 18 | 14 | 7.50 | 14.55 | 10 | 14.50 |
| index.v7版本产出.html | 75.45 | 81.45 | 6 | 15 | 13.25 | 14 | 15.70 | 10 | 13.50 |
| index38.html | 69.49 | 77.49 | 8 | 18 | 10.50 | 13.75 | 13.74 | 10 | 11.50 |
| index46.html | 70.61 | 78.61 | 8 | 18 | 13.25 | 10 | 14.86 | 10 | 12.50 |

### index.html 与 index38/index46

- `index.html` 是六样本静态最终分第一。它的优势主要来自更深的 CSS、更多视觉技法、移动断点和脚本接线；它仍只有一个主要内容 section、四个 `href="#"`、标题从 h1 跳到 h3、无明确焦点样式且未处理减少运动，所以不能称为完整高质量前端。
- `index38.html` 命中典型“首屏 + 三优势 + 联系表单”。它确有平滑锚点和表单，交互静态分高于 `index46.html`，但未发现媒体查询，且火箭/调色板/锁等 emoji 充当 UI 图标；视觉编排与可访问性均弱于 `index.html`。
- `index46.html` 也使用三卡片特征区，存在单一 768px 断点和平滑锚点；但交互层更薄，缺少联系表单等任务闭环。其响应式和视觉分略高于 index38，交互分明显更低。
- 这组三者说明“代码更长/结构更完整”不等于“更强前端”。普通三卡片可能拿到不错的结构分，因此必须把模板惩罚、行为证据和可访问性分开报告。

## 从评测思想到本契约

| 评测问题 | 本轮静态证据 | 后续浏览器/人工证据 |
|---|---|---|
| HTML 是否基本正确 | doctype、骨架、标签栈、唯一 id、标题层级 | DOM 修复后结构、控制台错误 |
| 是否响应式 | viewport、媒体查询、流式单位、明显超宽风险 | 375/768/1024/1440 截图与溢出检测 |
| 是否真的交互 | 原生控件、事件监听、状态选择器、表单路径 | 点击/键盘轨迹、状态变化、刷新恢复 |
| 是否超越模板 | CSS/内容层级/视觉资产 + 三卡片惩罚 | 双盲成对偏好、任务特异性评分 |
| 是否离线可靠 | 外链、HTTP、第三方脚本、版本锁定 | 断网加载、CSP/SRI、供应链扫描 |
| 是否可访问 | lang、alt、label、焦点、地标、减少运动 | axe、键盘遍历、读屏、计算后对比度 |

## 不能越过的结论边界

- 六个样本的分数只代表本静态契约下的相对证据，不是任何外部榜单成绩。
- 未运行浏览器截图、axe、Lighthouse 或真实交互，所以不能宣称 WCAG 合规、无溢出或功能完成。
- 没有运行任何模型；本轮没有证明 ColorLM v47、现有切片或任何 K3 路由具备前端生成能力。
- DesignBench/FronTalk/ArtifactsBench 的具体元数据和评分公式必须在网络恢复后对照原文补证，不能从本文反推。

