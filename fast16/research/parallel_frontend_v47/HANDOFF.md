# ColorLM v47 并行前端能力研究交接

## 交付结论

隔离目录内已完成可复现的纯静态前端评分器、机器可读六样本审计、24 条冻结短门、K3/强教师轨迹蒸馏方案和自检。没有修改本目录之外的任何文件，没有修改 `PROJECT_STATE.md`、模型或运行时；没有启停模型、使用 GPU、下载权重或做模型长测。

核心结论：六个样本中 `index.html` 静态最终分最高（76.85），`index38.html` 为 69.49，`index46.html` 为 70.61。`index.html` 的 CSS/视觉与事件接线更丰富，但仍有空链接、标题跳级、焦点与减少运动缺失，不能写成“完整高质量前端”。index38/index46 均明显带有普通三卡片模板结构；index38 的表单/交互闭环更强，index46 的响应式/视觉静态分略高。

## 文件清单

### 评分器与契约

- `score_html.py`：Python 标准库静态评分器；支持单文件或目录，输出确定性 JSON。
- `scoring_contract.json`：六维权重、模板惩罚、公式、等级和结论边界。
- `audit_report.schema.json`：审计报告 Draft 2020-12 JSON Schema。
- `sample_audit_report.json`：六个桌面 HTML 的机器可读实测报告；SHA-256 为 `bf7f9b2d1728c6e5ec0c9befab450aa5107d70cd6fe8268fc8b99563824c3f3a`。

### 冻结短门

- `data/train.jsonl`：8 条开发题，8 个独立模板族。
- `data/validation.jsonl`：8 条验证题，模板族与 train/blind 不重叠。
- `data/blind.jsonl`：8 条盲测题，模板族与 train/validation 不重叠。
- `gate.schema.json`：单条短门契约。
- `validate_gates.py`：短门检查、离线 HTML 评估、相对普通三卡片基线的比较与晋级判断。
- `fixtures/ordinary_three_cards.html`：固定普通三卡片基线，静态分 42.85、模板惩罚 17。
- `fixtures/advanced_reference.html`：仅用于评分器自测的合成高级夹具，静态分 89.60、模板惩罚 0；不是模型产出。

### 研究、蒸馏与冻结

- `RESEARCH_NOTES.md`：公开评测思想、本地六样本比较及研究边界。由于网络检索按指令停止，对 DesignBench/FronTalk/ArtifactsBench 未核验元数据，不伪造作者/数字。
- `DISTILLATION_PLAN.md`：教师准入、可观测轨迹格式、数据配比、训练阶段、防污染与晋级门。
- `freeze_manifest.py`、`MANIFEST.json`：14 个核心文件的无 BOM UTF-8 与 SHA-256 冻结清单。
- `selftest.py`、`SELFTEST_REPORT.json`：13 项纯 CPU/静态自检与实测结果。
- `HANDOFF.md`：本交接。

## 评分契约

正向毛分共 100：结构 18、响应式 16、交互 16、视觉复杂度 20、依赖安全 10、可访问性 20。模板惩罚最高 20，最终公式为：

```text
final_score = clamp(sum(dimension_scores) - template_penalty, 0, 100)
```

模板惩罚覆盖恰好三张通用卡片、低内容层级、无独特视觉资产、通用落地页套话、浅 CSS 系统和三卡片无行为层。为避免误伤，三卡片只有在视觉技法不足或通用文案密度高时才触发最高签名惩罚。

该评分器不执行 JavaScript、不联网、不栅格化，因此不能证明真实点击、视口无溢出、计算后对比度、WCAG 合规或视觉审美。它是快速筛选器，不是浏览器测评替代品。

## 六样本实测排名

| 排名 | 文件 | 最终分 | 毛分 | 模板惩罚 |
|---:|---|---:|---:|---:|
| 1 | index.html | 76.85 | 79.85 | 3 |
| 2 | index.v17版本产出.html | 75.55 | 83.55 | 8 |
| 3 | index.v7版本产出.html | 75.45 | 81.45 | 6 |
| 4 | index46.html | 70.61 | 78.61 | 8 |
| 5 | index38.html | 69.49 | 77.49 | 8 |
| 6 | index.v19测试版产出.html | 67.55 | 78.55 | 11 |

完整维度、每项证据、外部依赖、finding 与源文件 SHA-256 均在 `sample_audit_report.json`。

## 24 条短门与隔离

- train/validation/blind 各 8 条，共 24 条；24 个 `template_family` 唯一，跨 split 交集为空。
- 每题同时冻结任务特异正则、候选禁用模式、六维最低分、最大模板惩罚和五项关键条件。
- validation/blind 只有在候选 8/8 完整、关键条件全过、至少 6/8 题通过、中位增益至少 12、至少 7/8 相对基线增益不低于 10、无回归且中位模板惩罚不高于 6 时，才可写“在本短门上比普通三卡片更好”。
- `blind.jsonl` 是治理层盲测，不是加密密封；训练操作者能读取文件就会污染 blind。正式运行应由独立操作者保管 blind，并在候选冻结后一次性开封。

## 已运行命令与实测输出

所有命令均为纯 CPU/静态：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -X utf8 fast16/research/parallel_frontend_v47/score_html.py `
  'C:\Users\Kangnaixi\Desktop\新建文件夹' `
  --output fast16/research/parallel_frontend_v47/sample_audit_report.json `
  --source-root 'C:\Users\Kangnaixi\Desktop\新建文件夹' --compact

python -X utf8 fast16/research/parallel_frontend_v47/freeze_manifest.py
python -X utf8 fast16/research/parallel_frontend_v47/validate_gates.py check
python -X utf8 fast16/research/parallel_frontend_v47/selftest.py
```

实测摘要：

```text
sample_count=6
gate_check.ok=true
gate_count=24
split_counts={train:8, validation:8, blind:8}
cross_split_family_overlaps={train:validation:[], train:blind:[], validation:blind:[]}
selftest.ok=true
selftest.tests=13
ordinary_three_cards.final_score=42.85
ordinary_three_cards.template_penalty=17
advanced_reference.final_score=89.60
advanced_reference.template_penalty=0
manifest.files=14
```

## 后续使用

对任意候选 HTML 目录做验证集离线比较，文件名必须是 `pf47-validation-01.html` 到 `pf47-validation-08.html`：

```powershell
python -X utf8 fast16/research/parallel_frontend_v47/validate_gates.py compare `
  --split validation `
  --candidate-dir <候选HTML目录> `
  --output <比较报告.json>
```

候选冻结并通过 validation 后，再由独立操作者按相同命令开封 blind。任何基于 validation/blind 题面、正则、阈值或失败分析的再训练都会污染该 split，必须换新题族和版本。

## K3 声明

`DISTILLATION_PLAN.md` 只定义未来教师轨迹蒸馏流程。当前没有 K3/强教师模型运行结果，也没有证据表明现有切片具备 K3 前端能力。允许的最强表述是：“已建立用于筛选未来 K3/强教师候选的静态短门与蒸馏协议。”
