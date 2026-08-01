# FullDepth → K3 counterfactual gate

这是一个可由 FullDepth 当前 token 的连续 `float/BF16 [1,1,4096]`
hidden 调用的最小研究接口。它不把 K3 router prior、词关键字或
HTML 评分当作能力路由。

## 当前结论

- L28/E780 真实 hybrid runtime 已逐文件 SHA-256 验证：
  `112,974,848` bytes。
- L92/E291 真实 F16 runtime 已逐文件 SHA-256 验证：
  `95,427,584` bytes。
- L28/E780 的 `b_in/b_out` 实际复用 L28/E41 bridge，不是 E780 独立证明的
  portal。
- K3 coordinate transport 的 held-out embedding cosine 均值是 `0.47558`，
  但 router cosine 相对打乱对照只多 `0.00235`；它不是能力标签。
- v20 现有证据状态是 `rejected_for_capability`，且不满足 LOTO。
- `parallel_frontend_v47` 是 24 题冻结 HTML 评测器，不是 K3 权重或
  hidden gate。
- 当前没有获批的 FullDepth `4096 ↔ 2048` portal，所以真实默认路径
  严格为 **no-op**。

机器可读结果见 `ASSET_AUDIT_REPORT.json`。这些只证明权重真实和接口
可运行，不证明已获得 K3 前端能力。

## 执行契约

```text
FullDepth current-token hidden [1,1,4096]
  └─ counterfactual linear gate（只由冻结 no-op/donor NLL 拟合）
       ├─ no-op: 返回原 tensor 对象，不加载 portal/胶囊
       └─ selected + approved portal
            └─ 4096→2048 → 真实 K3 capsule → 2048→4096 → alpha residual
```

`alpha=0` 在形状检查和任何懒加载之前返回；单测验证返回值与
输入是同一个 tensor 对象。门未批准、分数低于阈值、portal 未批准或
loader 缺失时也全部 fail closed。

FullDepth 可用调用点是每层 `hc_pre + RMSNorm` 后的当前 token
`ffn_input [1,1,4096]`。站点层必须在冻结评测合同中预先固定，不能看结果
后换层。

## 纯离线验证

```powershell
python -X utf8 -m pytest `
  fast16/research/polaris_meridian_v1/k3_counterfactual_gate/tests/test_gate.py -q

python -X utf8 -m fast16.research.polaris_meridian_v1.k3_counterfactual_gate.audit_assets `
  --research-root D:/project/大模型ssd化/fast16/research `
  --output fast16/research/polaris_meridian_v1/k3_counterfactual_gate/ASSET_AUDIT_REPORT.json
```

审计会读取并哈希现有胶囊，不下载权重、不启动模型。

## Counterfactual 标定输入

NPZ 必须在看 donor 结果前冻结，包含：

- `hidden`: `[N,4096]` FullDepth 当前 token 连续 hidden；
- `no_op_logits` / `donor_logits`: 同前缀强制两路 `[N,V]`；
- `target_ids`: `[N]` teacher-forced 目标 token；
- `task_ids`: `[N]` 无 pickle Unicode 数组。

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.k3_counterfactual_gate.counterfactual `
  --input frozen_counterfactual.npz `
  --frozen-contract frozen_tasks.json `
  --output-dir calibrated_gate
```

标定器用对偶 ridge 拟合 predicted NLL advantage，并对完整任务做
leave-one-task-out。只有强制 donor 在多数任务改善、每个 LOTO 门策略均
改善且冻结合同存在时，`gate.json` 才可以写入 `approved=true`。

## 不可跨过的边界

- 不允许用 `router_colorlm_f32.npy` 单独开门。
- 不允许截断/重复 4096 维 hidden 伪造 2048 维 portal。
- 不允许用 HTML 评分、文本关键字或主机任务类型路由。
- 不允许将权重加载成功、hidden 变化或合成单测写成 K3 能力。
