# ColorLM v47 多 token 草稿算法交接

## 一句话结论

rank-64、top-32/recent-96/train-frequent-64/limit-192 的多 token 草稿已经形成可训练、可自由滚动、
可做拒绝位置 error replay 和 8% 停止判断的纯 CPU 原型；但现有真实资产不满足数据契约，因此本轮
**没有训练真实 head、没有预测真实接受长度，最终决定为 `stop_no_cpp`**。

## 真实完成项

### 数据与防自欺契约

- `frozen_contract.json` 固定第 1 token 只复用 v38 原生 logits；训练器只含第 2–4 位参数。
- `data_contract.schema.json`、`row.schema.json` 要求每个 anchor 一次 terminal hidden、原生 top-32、
  最近96 token、完整 oracle 1–4 token，以及从同一 anchor 由 v38 温度0自由滚动得到的 validator 1–4 token。
- split 必须按 `group_id` 和 `template_cluster_id` 双重隔离；test 只开一次。
- oracle/teacher-forced 覆盖只是一道必要门。`acceptance_simulator.py` 不使用 teacher token 覆盖 proposal，
  单独报告自由滚动候选命中、首拒绝位置和 v38 接受长度。
- `error_replay.schema.json` 与训练器共同强制只有 train 拒绝记录可回灌；validation/test 会被拒绝。

### 算法原型

- `draft_core.py` 实现冻结候选优先级、train-only 高频表、无完整词表表格的 rank-64 hash key，以及
  一次 cascaded block head。
- head 结构为 `2048→64→64→64`，三个未来层按 `1.0/0.7/0.5` 做逐层 CE；第2位 proposal 不会
  作为第3/4位核心输入，降低串行误差传播。
- `train_future_head.py` 实现全批量 Adam、候选内 CE、train-only error replay 加权、模型与数据 SHA 绑定。
- `validate_dataset.py` 在训练前检查完整 oracle/v38 轨迹的 anchor shortlist 覆盖。
- `acceptance_simulator.py` 覆盖通过后才执行自由滚动；拒绝按 shortlist 漏词/候选内排序分型，并按
  train/validation/test、短/中/长上下文分桶。

### 精确成本与停止门

冻结最坏 `C=192,D=2048,R=64,P=3` 的 `cost_report.json`：

- cascaded block：139,456 参数，F16 278,912字节（0.26599MiB）；370,111 FLOPs/周期；
  另有1次sqrt、192次tanh、196,672次整数hash原语；持久/外部参考带宽286,080字节；
  Python参考峰值scratch 61,184字节，流式key可为12,288字节。
- 串行未来头：135,360参数，F16 270,720字节；370,239 FLOPs/周期。它少4,096参数，但有3个
  proposal-dependent评分阶段；block只有1个，可一次批量形成三个位置分数。
- 两者都没有未来完整词表投影；第1 token 的额外投影 FLOPs 严格为0。
- 接受样本到位后，对 `A∈[0,4]` 使用95%单侧 Hoeffding 下界；硬门只计已接受草稿 token，
  再除以 `1.35 + draft/base`。评估少于64 anchor或下界小于 `1.08x`，立即停止、不改 C++。

## 现有资产的真实审计

`evidence_gap_report.json` 对 v44 文件做了只读扫描：

- 408个样本来自96个任务，仅有train 72任务、validation 24任务，无test。
- capture 来自 `ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf`，不是v38。
- 408份hidden/logits形状真实为`[2048]`与`[248320]`，但采样点只是每个语义字段首个判别token。
- 96个任务中四连稀疏位置窗口为0；没有完整oracle token IDs，也没有v38自由滚动validator token IDs。
- 原 `408/408` shortlist报告自身是`development_feasibility_only`且`gate=false`。

因此它只能证明“v36的408个teacher-forced稀疏位置上，target在shortlist中”，不能证明一次anchor的完整
4-token覆盖、自由滚动命中、接受长度或加速。`offline_gate_report.json` 四项硬门全部失败并给出
`stop_no_cpp`；这是正确科学结论，不是工具失败。

## 轻量自检

`selftest_report.json`：21项通过，纯CPU、未用GPU。合成144 anchor上训练损失
`2.09037→1.02777`；用合成真值head验证平均接受长度4.0，同时验证错误head能生成拒绝位置回放。
这些数字只证明管线可运行，不是ColorLM结果。

精确复现命令：

```powershell
New-Item -ItemType Directory -Force fast16/research/parallel_draft_v47/repro | Out-Null
python -B fast16/research/parallel_draft_v47/selftest.py --output fast16/research/parallel_draft_v47/repro/selftest_report.json
python -B fast16/research/parallel_draft_v47/audit_existing_assets.py --output fast16/research/parallel_draft_v47/repro/evidence_gap_report.json
python -B fast16/research/parallel_draft_v47/cost_model.py --output fast16/research/parallel_draft_v47/repro/cost_report.json
python -B fast16/research/parallel_draft_v47/offline_gate.py --asset-audit fast16/research/parallel_draft_v47/repro/evidence_gap_report.json --cost fast16/research/parallel_draft_v47/repro/cost_report.json --output fast16/research/parallel_draft_v47/repro/offline_gate_report.json
```

当前自检退出0；后三条按停止合同退出2并写完整报告。`repro/`已有同名证据时应换新目录，不要删除现有证据。

## 未证明项

- v38在完整轨迹上的0.995 shortlist覆盖；
- rank-64 hash key对真实v38未来token的自由滚动泛化；
- validation/test平均接受长度与拒绝位置分布；
- `1.35`批验证成本因子在当前运行图上的真实性；
- KV/sequence临时分支、拒绝回滚、原子提交与贪心逐token等价；
- 真实端到端吞吐、p95、RAM/VRAM和冻结任务零回归。

## 主线下一步所需最小采集

只需一次新的 v38 CPU/GPU采集会话，不要继续消费 v44 稀疏点：

1. 每个 anchor 保存一次 v38 terminal hidden `[2048] F32`。
2. 同一步只保存 v38 原生 top-32 token ID/logit；第1 token固定为top-1，不保存未来完整词表投影。
3. 保存最近96个已提交token。
4. 从同一anchor、同一贪心配置让v38自由滚动最多4 token，保存`validator_token_ids`和是否提前EOS。
5. 独立保存完整`oracle_token_ids`和是否提前EOS；oracle只进覆盖门，不进自由滚动head输入。
6. 为每条记录保存`group_id/template_cluster_id/split/context_bucket`；三split双重隔离，
   validation+test合计至少64 anchor，并覆盖`<2K / 2K–8K / >8K`三个上下文桶。
7. 先执行 `validate_dataset.py`。完整oracle和validator轨迹覆盖任一低于0.995就停止；通过后才训练，
   再跑自由滚动、成本与统一停止门。解析下界不足8%仍停止，不写C++。

新数据到位后的完整命令见 `README.md`。本目录之外没有任何改动。
