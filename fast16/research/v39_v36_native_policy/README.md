# v39：v36 原生序列策略头（停止）

## 目的

验证把 v29 的冻结算法直接迁移到 v36 原生 terminal hidden 后，是否能得到比 v38 更可靠的
显式工具策略头。v39 复用 v29 的 20 个任务、192 个 teacher token、候选行规则、ridge 参数、
裁剪值和留出门；没有根据 v39 留出结果扫描参数。

## 产物

- `policy_head_contract.json`：预先冻结的拟合与晋级合同。
- `policy-states.cnob`：v36 原生采集的 192 组 base logits + terminal hidden，192,307,200 字节。
- `policy-base-nll.jsonl`：192 条强制 next-token NLL。
- `policy-base-nll.jsonl.manifest.json`：模型别名、精确覆盖和输入哈希。
- `policy-head-report.json`：离线拟合与留出结果。
- `policy-head-weights.npz`：失败候选权重，只保留复现，不得作为运行包。

## 结果

采集使用：

```text
base = ColorLM-v36-Qwen36-Global-Shared-Backbone.gguf
alias = ColorLM-v39-V36-Native-Policy-Capture
records = 192/192
exact NLL coverage = 100%
CNOB SHA-256 = 102459ba47a0a9cae311f2e839b42c367353a469b2bb4b2bc1f2d7c65d23201d
```

冻结拟合结果：

| split | mean target NLL delta | candidate win rate |
|---|---:|---:|
| train | -4.154131 | 100.00% |
| validation | -0.571971 | 68.57% |
| test | -0.081604 | 57.14% |

最坏留出任务平均 NLL 回归为 `+0.160573`。合同要求 test 候选胜率至少 60%，且任一留出任务
回归不超过 `+0.03`；两项均失败。因此 `gate_passed=false`。

## 决定

- 停止 v39，不构建 runtime package，不跑生成门，不命名可体验模型。
- v38 继续作为当前最佳研究/体验候选；v29 继续作为 8105 稳定回滚入口。
- 该失败表明 v29 的无条件多 token ridge 修正迁到 v36 后发生明显训练过拟合。下一候选必须先有
  显式 no-op/置信收缩，并用新的未见任务做最终留出，不能查看本次留出后扫描 lambda、margin、
  clipping 或阈值来挽救 v39。

v39 是有效负结果，不是能力升级。
