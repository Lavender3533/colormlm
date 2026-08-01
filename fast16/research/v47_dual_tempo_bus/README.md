# ColorLM v47 Dual-Tempo Neural Bus

## 当前结论

v47 是一条新的研究主线，不是已经晋级的体验模型。正式最佳仍是
`ColorLM-v38-Qwen36-Shared-Sequence-Policy`。v47 只接受两种能被单独归因的增益：

1. **快通道**：小草稿头一次提出多个 token，v38 逐 token 无损验证；拒绝时输出必须与 v38 一致。
2. **慢通道**：只在困难决策处运行 `K=0..4` 次局部潜在循环，再调用一个有完整序列监督的能力岛。

它不再把单个供体专家切片称为完整能力，也不再使用固定 alpha、无监督硬 top-1 或单 token 稀疏头
冒充“更聪明”。

## 为什么先做序列岛

v43/v44 已经证明：teacher-forced NLL 可以显著改善，但单 token 小头不一定改变真实生成。
因此 v47 的最小能力单元必须对**完整短序列**负责：

- 工具岛：直接生成完整工具名、参数和结束标记；
- 前端岛：先生成结构化 Design IR，再由 v38 根据 IR 生成/编辑/修复页面；
- 规划岛：生成短步骤图或动作序列，不直接替换通用语言主干。

## 文件

- `DESIGN.md`：双节奏架构、成本边界和晋级顺序。
- `dual_tempo_contract.json`：冻结的总线与停止条件。
- `sequence_island.schema.json`：每任务一次 hidden + 完整 target 序列的数据契约。
- `frontend_design_ir.schema.json`：前端能力岛的结构化目标。
- `prepare_sequence_capture.py`：借用现有 v13 tokenizer/chat-template 生成全序列 teacher，压缩为每任务一次采集。
- `evaluate_semantic_compression.py`：统一验收短 Design Genome 的质量增益、确定性编译、HTML/IR
  展开比与真实端到端墙钟；`--selftest` 可做纯 CPU 自检。
- `fit_parallel_genome_head.py`：把 terminal hidden 一次并行映射到全部 Design Genome 角色字段，避免
  自回归 IR 的重复、截断与曝光偏差；真实前端训练集不足128条时硬拒绝。
- `genome_head_ontology.json`：并行字段、角色槽位和允许值的冻结本体。
- `fit_sequence_island.py`：CPU 可训练的 terminal-hidden → GRU 完整短序列原型。
- `measure_shortlist_coverage.py`：流式读取既有 CNOB，筛掉过大的草稿词表投影。
- `make_synthetic_fixture.py`：只验证训练器和切分纪律的合成夹具，不作为模型能力证据。
- `verify_v47.py`：UTF-8、JSON、schema、数据切分和训练产物自检。

## 最短执行顺序

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:TEMP=(Resolve-Path .\fast16\research\v47_dual_tempo_bus).Path
$env:TMP=$env:TEMP

# 纯 CPU 基础设施自检；不启动模型
python .\fast16\research\v47_dual_tempo_bus\make_synthetic_fixture.py
python .\fast16\research\v47_dual_tempo_bus\fit_sequence_island.py `
  --dataset .\fast16\research\v47_dual_tempo_bus\selfcheck\sequence_dataset.jsonl `
  --capture .\fast16\research\v47_dual_tempo_bus\selfcheck\initial_states.cnob `
  --output .\fast16\research\v47_dual_tempo_bus\selfcheck\sequence_island.npz `
  --report .\fast16\research\v47_dual_tempo_bus\selfcheck\fit_report.json `
  --epochs 80 --wall-seconds 20
python .\fast16\research\v47_dual_tempo_bus\verify_v47.py
```

真实采集必须使用全新冻结任务和按采集变量启动的 v38 服务；具体命令见 `prepare_sequence_capture.py -h`
和 `DESIGN.md`。没有真实 capture、跨模板 validation、一次性 blind 之前，不建立 v47 用户启动入口。

## 已完成的离线速度筛选

现有 v44 开发采集包含 408 个工具名、参数名/值和结束字段关键 token。动态 shortlist 扫描结果：

| 结构 | 总覆盖 | validation 覆盖 | 平均候选行 | 结论 |
|---|---:|---:|---:|---|
| native top-16 / 最多128行 | 406/408 | 98.04% | 86.43 | 拒绝，有2个关键漏词 |
| native top-32 / 最多192行 | 408/408 | 100% | 112.61 | 允许训练低秩草稿头 |

因此默认冻结为 rank-64、native top-32、最近上下文96、train-only高频64、最多192行。该结果来自
已消费的 teacher-forced 开发数据，只证明“没有必要做四个完整词表投影”；独立完整轨迹上的覆盖率、
低秩头接受长度和真实 A/B 仍未完成。

## 单条前端开发题结果

`pf47-train-01` 上，紧凑 Design IR + 确定性结构收尾得到完整运维台页面，静态短门从固定三卡片
基线的 42.85 分升到 80.55 分，浏览器中状态筛选、详情抽屉、告警弹窗和 375px 卡片切换均实际
工作。移动端文字仍较拥挤，而且候选包含确定性编译修复；因此只批准继续训练 IR 序列岛，不批准
模型能力晋级。完整失败过程、浏览器证据和下一步见 `HANDOFF.md`。
