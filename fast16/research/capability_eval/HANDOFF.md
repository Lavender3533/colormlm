# ColorLM 八维能力短门交接

## 结论边界

这里已建立一个纯 CPU、确定性、共 48 题的短门。八个维度各 6 题，物理分为开发集 3 题、任务级留出集 2 题、最终留出集 1 题。每题都包含完整输入、参考答案、自动 validator、关键决策 token 和明确失败条件。

该短门只判断冻结候选是否在这些短任务上严格优于 v17。`computer_use` 是离线 UI 状态到动作选择探针，不是实际桌面执行；`tools` 是结构化调用决策，不执行工具。即使最终通过，也不得声称长榜、真实桌面、长期 Agent 或全面能力提升。本次没有运行任何模型，没有启动服务，没有使用 GPU，也没有运行 CMake。

## 文件

- `data/dev.jsonl`：24 题，可用于接线、格式和早期调试；成绩不参与晋级。
- `data/task_holdout.jsonl`：16 题，候选配置冻结后只运行一次配对比较。
- `data/final_holdout.jsonl`：8 题，只有任务级留出通过后才能开封和运行。
- `schemas/task.schema.json`：题目 JSONL 每行的 schema。
- `schemas/response.schema.json`：模型响应 JSONL 每行的统一 schema。
- `schemas/report.schema.json`：`score`、`compare`、`promote` 统一报告 schema。
- `validate.py`：schema 检查、自动判分、配对比较、LOTO 与晋级判定。
- `selftest.py`、`SELFTEST_REPORT.json`：不调用模型的 validator 自测及结果。
- `MANIFEST.json`：冻结题数和三份题库 SHA-256。

模型请求只能由 task 的 `input.messages`、`input.tools`、`input.temperature` 和 `input.max_output_tokens` 构造。严禁把 `reference_answer`、`validator`、`critical_decision_tokens` 或 `failure_conditions` 放进模型上下文。

响应文件每行必须符合 `capability-response-v1`。普通回答把原始最终文本写入 `output`，`tool_calls` 为空；工具题的 `output` 必须为 `""` 或 `null`，并且只能有一个 `{name, arguments}`。同一文件的 `run_id`、`model_id` 和完整 `generation` 配置必须一致。`critical_token_observations` 是可选诊断字段，不参与生成正确率晋级，不能用它替代真实生成结果。

## 防污染协议

1. 开发集允许迭代，但任何基于开发集的调参必须在触碰任务级留出集前结束。
2. 开封任务级留出集前，冻结候选二进制/权重哈希、运行时、chat template、temperature=0、seed、上下文上限和输出上限；v17 与候选使用同一套设置，除候选本身外不得有差异。
3. 任务级留出集只允许一次 A/B 运行。失败后若修改候选、模板、路由、alpha、提示或判分器，本版本任务级留出作废，不能重跑刷分。
4. 任务级留出未通过时，禁止读取、运行或分析最终留出。通过后才允许一次性生成 v17 与候选最终响应。
5. 最终留出一旦开封，任何后续调参都会使本版本最终结论作废；需要新建不同任务族的新版本。
6. 任一题的答案、validator 或关键 token 进入训练、提示、RAG、关键词路由或人工修补流程，该 split 立即作废。不得按输出挑题、删题或改答案。
7. 只接受模型直接生成和原生工具调用。关键词路由、检索答案、外部流程包装或手工改写响应不能记作模型能力。

## 晋级规则

所有规则均由 `validate.py` 固化，速度或资源结果不能挽救能力失败。

任务级留出集必须同时满足：A/B 各 16 条响应完整且 schema 合法；基线正确题零回归；候选至少修正 4 题；至少 4 个维度净增；八维均不回归；逐一留出 16 个任务后，剩余任务的候选总正确数仍严格高于 v17。

最终留出集必须同时满足：A/B 各 8 条响应完整且 schema 合法；基线正确题零回归；候选至少修正 2 题；至少 2 个维度净增；八维均不回归。只有任务级留出和最终留出都通过，`promotable` 才为 `true`。

任何缺题、重复 task id、未知 task id、运行参数不一致、`length/error` 停止、工具名/参数/停止原因错误、工具调用夹带文本，都会按失败或不完整处理。

## 主机下一步命令

先做纯 CPU 复核，不运行模型：

```powershell
Set-Location 'D:\project\大模型ssd化'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -X utf8 fast16/research/capability_eval/validate.py check
python -X utf8 fast16/research/capability_eval/selftest.py
New-Item -ItemType Directory -Force fast16/research/capability_eval/runs, fast16/research/capability_eval/reports | Out-Null
```

主机用自己的冻结推理入口生成以下 UTF-8 JSONL；本目录不负责启动模型：

```text
fast16/research/capability_eval/runs/v17.dev.responses.jsonl
fast16/research/capability_eval/runs/candidate.dev.responses.jsonl
fast16/research/capability_eval/runs/v17.task_holdout.responses.jsonl
fast16/research/capability_eval/runs/candidate.task_holdout.responses.jsonl
fast16/research/capability_eval/runs/v17.final_holdout.responses.jsonl
fast16/research/capability_eval/runs/candidate.final_holdout.responses.jsonl
```

先验证开发集接线：

```powershell
python -X utf8 fast16/research/capability_eval/validate.py compare `
  --tasks fast16/research/capability_eval/data/dev.jsonl `
  --baseline fast16/research/capability_eval/runs/v17.dev.responses.jsonl `
  --candidate fast16/research/capability_eval/runs/candidate.dev.responses.jsonl `
  --out fast16/research/capability_eval/reports/dev.compare.json
```

冻结候选后运行任务级留出，并先看 `decision.status`；只有值为 `task_holdout_pass` 才继续：

```powershell
python -X utf8 fast16/research/capability_eval/validate.py compare `
  --tasks fast16/research/capability_eval/data/task_holdout.jsonl `
  --baseline fast16/research/capability_eval/runs/v17.task_holdout.responses.jsonl `
  --candidate fast16/research/capability_eval/runs/candidate.task_holdout.responses.jsonl `
  --out fast16/research/capability_eval/reports/task_holdout.compare.json
Get-Content -Raw -Encoding UTF8 fast16/research/capability_eval/reports/task_holdout.compare.json
```

通过后才生成最终留出响应，再做最终复核：

```powershell
python -X utf8 fast16/research/capability_eval/validate.py promote `
  --task-holdout-baseline fast16/research/capability_eval/runs/v17.task_holdout.responses.jsonl `
  --task-holdout-candidate fast16/research/capability_eval/runs/candidate.task_holdout.responses.jsonl `
  --final-baseline fast16/research/capability_eval/runs/v17.final_holdout.responses.jsonl `
  --final-candidate fast16/research/capability_eval/runs/candidate.final_holdout.responses.jsonl `
  --out fast16/research/capability_eval/reports/promotion.json
Get-Content -Raw -Encoding UTF8 fast16/research/capability_eval/reports/promotion.json
```

最终只以 `reports/promotion.json` 中的 `decision.promotable` 为准。开发集或单个 split 的领先均不能写成候选优于 v17。

## 已完成自测

`python -X utf8 fast16/research/capability_eval/selftest.py` 已返回退出码 0：`ok=true`、8 组测试、48 题。测试覆盖通过路径、LOTO、最终晋级、单题回归阻断、缺题、重复记录和工具夹带文本。当前 Python 环境未安装可选的 `jsonschema` 包，因此实例检查由 `validate.py` 的内置严格检查完成；三个 JSON Schema 自身均已成功解析并确认 Draft 2020-12 标识。没有 validator 自测失败项。
