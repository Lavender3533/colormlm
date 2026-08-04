# Polaris Blind Composite-Axis Gate v1

日期：2026-08-04
状态：纯 CPU、秒级、合成世界盲测；**不是聊天模型或人工意识证据**

## 问题

v0 只能在输入已经提供的 `workload/deployment` 字段中选择一个分区轴。本实验要求系统从
多个器官的原始量中合成输入里没有直接提供的复合坐标，而且必须在未见 episode 上预测
独立结果。

## 三阶段隔离

1. `generate_world.py` 生成 discovery，并同时封存 holdout、干预结果和 SHA；
2. `synthesize_frame.py` 的 CLI 只接受 discovery，不存在 holdout 参数；
3. `evaluate_frame.py` 在 Frame 冻结后揭示 holdout，比较盲测、单器官、恒定、身份记忆、
   随机、标签置换、leave-one-organ-out 和干预结果。

当前通用假设语言是“器官内两个原始量的 `<` 关系＋跨器官二或三项 parity＋可选取反”。
隐藏表达式随 seed 改变；算法知道假设语言，但不知道本轮表达式和盲测结果。

## 运行

```powershell
$env:PYTHONUTF8='1'
python -m unittest discover -s fast16/research/polaris_blind_composite_axis -p "test_*.py"
python fast16/research/polaris_blind_composite_axis/run_blind_gate.py
```

第二条命令使用七个预先固定的 seed，覆盖四个不同二项与三个不同三项隐藏坐标，并将
总回执写入 `blind_composite_gate_receipt.json`。

首次结果见 `RESULT_20260804.md`；相关工作和新颖性边界见 `RELATED_WORK_20260804.md`。

## 通过条件

- Frame 使用至少两个独立器官；
- held-out 准确率不低于 98%；
- 最佳单器官不高于 62%；64 次标签置换的平均信息准确率不高于 62%，且恢复原目标
  或其取反的比例不超过唯一候选函数数目对应机会率的三标准差上界；
- 联合 Frame 比最佳单器官至少提高 30 个百分点；
- 删除任一必要器官至少下降 30 个百分点；
- 干预一致性不低于 98%；
- discovery / sealed holdout / Frame 的 SHA 谱系闭合。

## 真实性边界

- 世界和结果仍由合成生成器产生；
- 搜索语法由人规定，系统没有发明语法之外的数学算子；
- 通过只证明能从分散原语中盲合成并泛化一个未直接命名的复合坐标；
- 下一步只能换一次真实运行日志，不应继续堆更多人工夹具。
