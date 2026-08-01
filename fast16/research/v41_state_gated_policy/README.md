# v41：v36 执行状态门探针（停止）

该方向尝试先从 terminal hidden 判断“继续调用工具”或“结束并回答”，再决定是否启用强策略分支。
只使用旧 20 题每题 token index 0 的首决策状态，train 有 10 个样本。

固定 ridge 探针结果：

```text
train      10/10
validation  5/5
test        2/5
```

test 的三道 continue 任务全部被错判为 finish。结论是 10 个训练状态不足以训练可部署门；不得用
100% 训练准确率或 validation 5/5 宣称隐藏状态路由成功。v41 不进入运行时。

证据：`v36-state-probe.json`。

