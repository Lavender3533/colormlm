# v40：范数匹配的 v36 原生策略头（停止）

v39 原生头的权重 L2 为 `81.1192`，旧 v29 头为 `21.1139`。v40 不扫描 alpha，而是预先固定
唯一尺度：

```text
scale = ||W_v29||₂ / ||W_v39||₂ = 0.2602826278
W_v40 = scale × W_v39
```

旧 20 题已降级为开发数据。开发诊断中，范数匹配候选为 17/20 任务 NLL 改善，最坏任务回归
从 v39 的 `+0.160573` 降到 `+0.037672`。随后在看到结果前冻结 12 道新题（6 continue、
6 finish），SHA-256 为 `e708fe20acad8992e893950eb3cc8024141391eee37f1d48229e060869883cc5`。

严格生成盲测：

```text
v36 control: 5/12
v40:         5/12
逐题通过/失败集合完全相同
```

结论：范数匹配降低了离线风险，但修正不足以改变实际解码。按合同停止 v40，不扫描其他尺度，
不替换 v38。运行包仅为复现实验候选，不是可用版本。

关键产物：

- `v40_contract.json`
- `blind_tasks_v1.jsonl`
- `dev-analysis.json`
- `v36.blind12.generation.json`
- `v40.blind12.generation.json`
- `runtime-v1/`

