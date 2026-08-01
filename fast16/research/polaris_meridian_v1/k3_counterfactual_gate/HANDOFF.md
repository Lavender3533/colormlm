# 交接

## 已完成

1. `runtime.py`
   - FullDepth `[1,1,4096]` 连续 hidden 门接口；
   - `alpha=0` 物理旁路；
   - gate/no portal 时不加载胶囊；
   - 可懒加载真实 F16 K3 latent macro runtime；
   - 只接受有 held-out hidden + NLL 证据的显式 portal manifest。
2. `counterfactual.py`
   - 同前缀 no-op/donor target NLL；
   - 任务级强制 donor 收益；
   - leave-one-complete-task-out 拟合与策略收益；
   - 冻结合同缺失时禁止 `approved=true`。
3. `audit_assets.py` / `ASSET_AUDIT_REPORT.json`
   - 已哈希验证 L28/E780 与 L92/E291 真实 runtime；
   - 已标记 coordinate transport、v20 和 `parallel_frontend_v47` 的证据上限；
   - 当前机器状态为 `assets_verified_gate_disabled`。

## 下一步唯一正式路径

1. 选定一个 FullDepth 站点，在看结果前冻结任务、关键 token 和站点。
2. 导出该站点的连续 `[N,4096]` hidden，不得只导出 norm/hash 摘要。
3. 单独标定 `4096↔2048` portal；必须有 held-out hidden 和 next-token NLL，
   否则 runtime 保持 no-op。
4. 在同前缀强制 `{no-op, L28/E780, L92/E291}`，生成冻结 NPZ。
5. 运行 `counterfactual.py`。只有 gate 和 portal 都获批，才把
   `FullDepthK3Bus.apply(..., alpha>0)` 接入研究图。
6. 最后用 `parallel_frontend_v47` 的 validation/blind 题测生成结果；评分器
   不进入在线门。

## 立即可运行

```powershell
python -X utf8 -m pytest `
  fast16/research/polaris_meridian_v1/k3_counterfactual_gate/tests/test_gate.py -q

python -X utf8 -m fast16.research.polaris_meridian_v1.k3_counterfactual_gate.audit_assets `
  --research-root D:/project/大模型ssd化/fast16/research `
  --output fast16/research/polaris_meridian_v1/k3_counterfactual_gate/ASSET_AUDIT_REPORT.json
```

本交接不要求启动模型、下载新权重或运行长榜。
