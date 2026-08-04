# Polaris 证据耦合星座动力学 v0

日期：2026-08-04
状态：最小可证伪结构实验；**不是聊天模型，不是人工意识证据**

## 要验证的数学语义

本实验不再把多个器官的 logits 平均成一个答案，也不用一个中央路由器提前选岛。
它只实现四个动作：

1. `propose`：器官向当前思维场提出局部候选；
2. `merge_or_branch`：不同 slot 的变化合并，同 slot 异值保留为并行世界；
3. `attach_evidence`：证据只更新依赖它的候选，同一来源的新回执取代旧回执；
4. `partial_commit`：只提交所有存活分支的共有交集，并要求证据与依赖已闭合。

探索优先级与证据认证被物理分开。模型提议可以得到很高的探索优先级，但没有外部证据时仍不允许
进入已提交子图。

## 三个冻结场景

- Java/PHP：后端分支未定时先提交共有布局/API；Composer 回执反转后只回滚 PHP；
- 截图网页：全表格、全卡片和响应式混合是三个分支，浏览器证据只保留混合分支；
- 事实冲突：两个大模型给出相反答案，重复或高置信的神经输出不得提交，真实 probe 回执后才能提交。

## 秒级入口

```powershell
$env:PYTHONUTF8='1'
python -m unittest discover -s fast16/research/polaris_evidence_constellation -p "test_*.py"
python fast16/research/polaris_evidence_constellation/run_minimal_gate.py
```

第二条命令会写出 UTF-8 `minimal_gate_receipt.json`。

## 真实性边界

- 本轮没有启动 v38、v17、S14 或任何 GPU 模型；
- 场景与证据是冻结夹具，只证明结构语义；
- 结构通过不等于它会自主生成好候选，也不证明能力超过 Transformer；
- 下一步只能用一个实时神经提议器取代一个夹具提议，而不是同时接入所有权重岛。
