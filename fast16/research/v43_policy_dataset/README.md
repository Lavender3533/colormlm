# ColorLM v43 Policy Dataset

## 结论

v43已完成数据、采集、F32拟合、运行包、C++/Vulkan图和真实生成门闭环，但没有获得能力晋级。

```text
v36: 12/24
v43: 12/24
wins: 0
regressions: 0
net fixes: 0
decision: stop v43 runtime candidate
```

当前正式最好版本仍是 `ColorLM-v38-Qwen36-Shared-Sequence-Policy`。v43不建立启动入口，不在已消费的
validation/test上扫描PCA rank、ridge、strength或候选token。

## 数据与采集

- 120条完整状态轨迹，60个成对反事实组。
- train/validation/test：72/24/24。
- 六类能力：工具参数纠错、继续/结束、缺参澄清、多步规划、电脑操作、代码调试。
- 720/720 teacher token、720/720精确base NLL。
- 720个2048维terminal hidden和720个248320维base logits。
- `v36-states.cnob`：721,152,000字节。

限制：validation/test与train仍使用相同语义生成骨架，仅替换实体和状态。它们可以验证运行时原型，
不能当作跨模板盲测。

## 冻结模型

```text
normalized v36 terminal hidden
→ train-only PCA rank 8
→ 9-class ridge classifier
→ class 0 = exact no-op
→ classes 1..8 = eight sparse token corrections
```

参数在查看留出结果前冻结：

```text
PCA rank: 8
ridge lambda: 0.1
correction strength: 12.0
```

最终F32回放：

| split | 平均NLL delta | 任务胜/负 | exact no-op |
|---|---:|---:|---:|
| train | -1.02578 | 72/0 | 25.46% |
| validation | -1.03262 | 24/0 | 24.31% |
| test | -1.02190 | 24/0 | 25.69% |

这些数字只批准运行时实现，不等于生成能力改善。

## 运行时实现

`runtime-v1/` 使用 `colorlm-sequence-policy-runtime-v3`，共6张张量、74,180字节：

- `policy.base_ids[8]`
- `policy.hidden_mean[2048]`
- `policy.pca_components[2048,8]`
- `policy.classifier[8,9]`
- `policy.classifier_bias[9]`
- `policy.correction_strength[1]`

CPU独立回放在720个样本上与最终报告一致：预测类别差异`0/720`，no-op差异`0/720`。Vulkan图完成
归一化、PCA、分类、softmax、精确no-op门和稀疏logit回写。实现过程还修正了Vulkan `GGML_STEP`
在零点与CPU/CUDA不一致的错误：现在统一为 `x > 0`。

## 真实生成门

测试只使用冻结的24条test轨迹，严格比较工具名、参数、结束原因和裸JSON。通过要求为净修复至少3、
回归最多1。

| 能力 | v36 | v43 |
|---|---:|---:|
| 工具参数纠错 | 2/4 | 2/4 |
| 继续/结束 | 4/4 | 4/4 |
| 缺参澄清 | 2/4 | 2/4 |
| 多步规划 | 2/4 | 2/4 |
| 电脑操作 | 2/4 | 2/4 |
| 代码调试 | 0/4 | 0/4 |
| 合计 | 12/24 | 12/24 |

剔除每次请求随机生成的tool-call ID后，18/24条输出与v36相同，6/24条输出改变；改变的6条仍全部
保持原有通过/失败状态。

## 失败归因与下一步

v43证明了一个重要边界：“留出teacher token NLL大幅改善”不能推出“真实生成更聪明”。当候选行主要是
工具包装、换行和JSON结构token时，策略头可以显著降低NLL，却不能直接改正工具、参数和动作决策。

v44词表span审计把这一点定量化了：

- 120/120条任务存在前6 token之外的关键动作token。
- 2,654个关键token occurrence中2,274个不在v43监督窗口内。
- 工具名span从index 4开始，但390个子token中210个超出窗口。
- 参数span从index 9–24才开始，1,324/1,324全部超出窗口。
- 结束JSON字段有740/940个子token超出窗口。

审计报告位于 `../v44_critical_action_bus/critical-span-audit.json`。

下一代必须：

1. 使用全新模板簇和未见blind，不复用v43 test调参。
2. 候选从公共结构token改为动作前缀、工具名、关键参数名/值组。
3. 用完整轨迹成功和反事实动作翻转作为目标，NLL只作资格门。
4. 仅在train/validation上选结构，冻结后一次跑blind。

## 主要产物

- `policy_contract.json`：预注册合同。
- `trajectory_tasks_v1.jsonl` / `trajectory_oracle_v1.jsonl`：冻结轨迹与oracle。
- `teacher.jsonl` / `base-nll.jsonl` / `v36-states.cnob`：采雁证据。
- `policy-report-f32.json` / `policy-weights-f32.npz`：最终F32拟合与报告。
- `runtime-v1/`：冻结运行包。
- `runtime-package-cpu-selfcheck.json`：720样本独立CPU回放。
- `v43-generation-gate.json`：真实生成否决报告。
