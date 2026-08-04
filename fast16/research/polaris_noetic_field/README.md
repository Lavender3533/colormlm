# Polaris Noetic Field v0

日期：2026-08-04

状态：最小可证伪结构实验；**不是聊天模型，不是实时权重执行成果**

## 验证的不是新名字

这个实验把 Transformer 的固定层流水线改成事件驱动的共享状态：

```text
真实 base 激活 ─┐
                   ├→ 共享候选场 → 多轮表达/抑制 → 稳定后 commit
真实 donor 激活─┘                       ↘ 冲突未解则 rollback
```

场在运行期不读取 prompt 文本、task id 或正确 label，因而不是关键词路由。
label 只在所有提交完成后评分。

## 真实性边界

- 输入是 `inference_arch/capture_dev60b.npz` 中保存的60个真实 base/donor
  logits 和 donor hidden，不是随机假权重。
- 脚本首先核对 capture SHA-256。
- 它是离线激活回放，没有当场执行 donor 权重；所以只能验证意识场的
  竞争、稳定、commit 和 rollback 骨架。
- 结果好坏都必须保留；不得把“运行成功”写成“能力提升”。

## 短命令

```powershell
$env:PYTHONUTF8='1'
python fast16/research/polaris_noetic_field/noetic_field.py
```

输出 `replay_receipt.json`，包含每个位置每轮的候选 token、场变化、冲突、
base/donor 表达强度和最终 commit/rollback 回执。

## 下一个真门

只有当这个结构不是退化的一次性门控时，才把 `IslandOpinion` 的来源替换为：

1. v17/v18 连续原始权重岛的实时激活；
2. S14 Range/Vulkan 执行器的实时权重岛回执；
3. Kimi 视觉岛的带类型视觉对象。

实时门只认3件事：原始权重确实执行、岛意见确实进入场、失败时精确回滚。
