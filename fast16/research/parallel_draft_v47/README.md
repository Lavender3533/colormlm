# ColorLM v47 多 token 草稿离线原型

本目录是独立、纯 CPU 的可证伪原型。它不加载模型、不启动或停止服务、不调用 GPU、不下载权重、
不运行 CMake，也不修改 v47 主线或 C++。

## 冻结算法

- 第 1 个 token 直接复用 v38 原生 logits 的 top-1；没有第 1 位训练参数，也没有额外词表投影。
- 只训练未来第 2–4 位：`2048→rank-64` 后经过两层 rank-64 cascade，一次形成三个 latent，逐层 CE。
- 每个 anchor 只构造一次动态 shortlist：原生 top-32、最近上下文 96、train-only v38 高频目标 64，
  去重后最多 192。
- shortlist token key 由 token ID 即时生成固定 Rademacher 向量；运行时只算最多 192 行，绝不做
  `2048×248320` 或 `64×248320` 的未来完整词表投影。这是最小可证伪 key 原型，不宣称是最终最优表示。
- 目标验证必须来自同一 anchor 的 v38 温度 0 自由滚动。oracle 只做必要覆盖，不能喂给自由滚动模拟。

按主线新增的文献约束，本目录把 NanoSpec（arXiv:2605.26444）作为上下文动态极小词表的设计依据；
把 Draft-OPD（arXiv:2605.29343）指出的 offline-to-inference mismatch 落为自由滚动和拒绝位置回放硬门；
把 FastEagle（arXiv:2509.20416）的一次整块草稿与逐层监督约束落为 cascaded block head。论文数字不外推到本项目。

## 文件

- `frozen_contract.json`：rank、shortlist、覆盖、接受与 8% 停止门。
- `data_contract.schema.json`、`row.schema.json`：manifest、JSONL 行与 NPZ 数组契约。
- `error_replay.schema.json`：首拒绝位置回放；训练器拒绝 validation/test 回灌。
- `draft_core.py`：数据校验、候选集构造、hash key 与 cascaded block head。
- `validate_dataset.py`：完整 oracle/v38 轨迹 shortlist 必要覆盖门。
- `train_future_head.py`：未来第 2–4 位逐层训练；可选 train-only error replay。
- `acceptance_simulator.py`：自由滚动、拒绝位置、v38 接受长度与上下文分桶。
- `cost_model.py`：精确 FLOPs/权重/带宽/临时内存，并比较串行头和单次 block 头。
- `offline_gate.py`：统一停止门。
- `audit_existing_assets.py`：只读审计 v44 现有资产为何不够。
- `selftest.py`：纯合成、纯 CPU 轻量自检。

## 当前真实结论

`evidence_gap_report.json` 证明现有 408 条数据来自 v36、仅有 train/validation、是 96 个任务的稀疏
teacher-forced 关键位置，而且没有任何四连位置窗口；它没有完整 oracle token 序列，也没有 v38 自由滚动
validator。原 `408/408` 报告自身 `gate=false`。所以本轮没有在真实资产上训练 head、没有预测接受长度，
`offline_gate_report.json` 正确给出 `stop_no_cpp`。

## 当前复现

在工作区根目录执行：

```powershell
New-Item -ItemType Directory -Force fast16/research/parallel_draft_v47/repro | Out-Null
python -B fast16/research/parallel_draft_v47/selftest.py --output fast16/research/parallel_draft_v47/repro/selftest_report.json
python -B fast16/research/parallel_draft_v47/audit_existing_assets.py --output fast16/research/parallel_draft_v47/repro/evidence_gap_report.json
python -B fast16/research/parallel_draft_v47/cost_model.py --output fast16/research/parallel_draft_v47/repro/cost_report.json
python -B fast16/research/parallel_draft_v47/offline_gate.py --asset-audit fast16/research/parallel_draft_v47/repro/evidence_gap_report.json --cost fast16/research/parallel_draft_v47/repro/cost_report.json --output fast16/research/parallel_draft_v47/repro/offline_gate_report.json
```

自检应返回0。其余三条在当前证据下按合同返回2，同时完整写报告；这表示候选被停止，不是脚本故障。
若`repro/`已有同名证据，请换一个新目录；研究脚本不会覆盖旧报告。

## 新数据到位后的顺序

```powershell
python -B fast16/research/parallel_draft_v47/validate_dataset.py --dataset <dataset-manifest.json> --output <dataset-audit.json>
python -B fast16/research/parallel_draft_v47/train_future_head.py --dataset <dataset-manifest.json> --output <head.npz>
python -B fast16/research/parallel_draft_v47/acceptance_simulator.py --dataset <dataset-manifest.json> --model <head.npz> --output <acceptance.json> --error-replay-output <train-error-replay.jsonl>
python -B fast16/research/parallel_draft_v47/cost_model.py --acceptance <acceptance.json> --output <cost-with-acceptance.json>
python -B fast16/research/parallel_draft_v47/offline_gate.py --asset-audit <dataset-audit.json> --acceptance <acceptance.json> --cost <cost-with-acceptance.json> --output <offline-gate.json>
```

必须逐步执行；任一步返回 2 就停止。最后解析加速下界不足 `1.08x` 时，明确停止且不改 C++。
