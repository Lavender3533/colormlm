# ColorLM v18 激活桥离线稳定性审计

日期：2026-07-31

## 结论

当前无先验激活桥仍是研究候选，不能装入正式 v18 神经岛。

真实深层激活的几何信号很强：12 折整提示 LOTO 中，12 条 prompt 的余弦提升均为
正，合并中位余弦提升为 `+0.3657`。但稳定性晋级要求同时满足绝对误差、相对误差和
多 seed 通过率；现有尺度和仅用训练集拟合的最小二乘尺度都没有通过全部门槛。

本次只读取现有 CLM9 dump、采集收据和嵌入桥，没有启动模型、下载权重或修改正式模型。

## 严格结果

| 候选尺度 | LOTO | LOTO NRMSE | LOTO NRMSE 比率 | 5 个唯一 seed | 最差 seed NRMSE 比率 | 总结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 现有中位范数比 | 失败 | 0.9381 | 0.9559 | 0/5 | 0.9848 | reject |
| 训练集最小二乘 | 通过 | 0.9158 | 0.9332 | 2/5 | 0.9641 | reject |

门槛在执行前固定：

- 整提示余弦中位提升 `>= 0.03`；
- 相对各自训练集校准后的嵌入桥，NRMSE 比率 `<= 0.95`；
- 候选绝对 NRMSE `<= 0.95`；
- 正收益 prompt 比例 `>= 0.67`；
- 5 个不同整提示划分至少 `80%` 全门通过；
- 任一 seed 不得出现余弦或 NRMSE 回归；
- LOTO 相对全量拟合的预测余弦中位数 `>= 0.95`、最差值 `>= 0.90`。

训练集最小二乘版本的 LOTO 门全部通过，但 5 个 seed 只有 `2/5` 全门通过，低于
`4/5` 的晋级要求。三个失败切分的主要失败项是 NRMSE 比率，不是余弦方向、绝对
NRMSE 或数值稳定性。不能通过放宽 `0.95` 门槛把它改写成通过。

## 审计发现

1. `activation_bridge.py` 当前会把候选桥的训练尺度同时用于候选和嵌入基线。该比较
   对几何方向并不公平：两个正交方向应分别只用训练 prompt 校准尺度，再共同在留出
   prompt 上验收。本审计采用独立训练尺度，未读取留出尺度。
2. 无先验全量拟合的有效秩为 `772/2048`，只覆盖隐藏宽度的 `37.70%`。这是正式部署
   前仍需谨慎的样本覆盖风险。
3. 已观测分布上的方向稳定性并不差：LOTO 相对全量拟合的最差预测余弦为 `0.9960`，
   5-seed 最差为 `0.9896`。当前主要瓶颈是未见 prompt 上的幅度误差，而不是桥方向
   随划分完全翻转。
4. 单次 3-prompt 留出报告可以碰巧通过，因此不能再用一个 seed 的 `candidate` 字段
   作为重打包依据。稳定性报告总决策必须是 `stable_candidate` 才能进入隔离包。

## 可复现命令

现有中位范数尺度：

```powershell
python fast16\research\v18_activation_bridge\stability_check.py `
  --allow-rejected `
  --output fast16\research\v18_activation_bridge\stability_report.json
```

仅用每折训练 prompt 校准最小二乘尺度：

```powershell
python fast16\research\v18_activation_bridge\stability_check.py `
  --scale-strategy train_least_squares `
  --allow-rejected `
  --output fast16\research\v18_activation_bridge\stability_report_train_ls.json
```

移除 `--allow-rejected` 后，拒绝结果会返回退出码 `2`，适合接入晋级流水线。

## 下一步约束

在增加覆盖更多整提示的配对激活，或用训练内层验证选择有理论依据的正则化/尺度后，
重新执行同一套门。多 seed 通过率达到 `>= 0.8` 前，不生成 v18 隔离运行包，也不进入
 代码/工具能力短测。现有 v17 正式模型保持不变。

## v18.1 后续结果

在不增加采集样本、不放宽门槛的条件下，新增 `nullspace_anchored` 正交完成：保留非零奇异值
对应的真实激活Procrustes最优方向，只在零奇异值自由度中选择距离旧嵌入桥最近的正交完成。
训练尺度继续严格只使用训练prompt的最小二乘值。

- LOTO中位余弦提升：`+0.3760`；
- LOTO绝对NRMSE：`0.9021`，相对公平嵌入桥比率：`0.9192`；
- 12条prompt：`12/12`正提升；
- 5个互不重复整prompt切分：`5/5`通过，最差NRMSE比率`0.9475`；
- 最差切分相对全量拟合预测余弦：`0.9896`；
- 运行矩阵正交RMSE与往返RMSE：均约`5.95e-8`。

总决策为`stable_candidate`。完整报告与运行矩阵位于
`candidate-nullspace-anchor-v1/`，新隔离包为`runtime-v2/`。该结果晋级的是坐标桥稳定性，
不是55B或闭源模型能力声明；独立短能力题通过前，v17仍是正式版。
