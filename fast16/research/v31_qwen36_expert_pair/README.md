# Qwen3.6 路由器/专家配对研究（v31–v38）

## 结论

v6 的40层 `ffn_gate_inp` 已由3-bit全秩差分逼近Qwen3.6路由器，但对应 routed/shared expert
仍来自旧ColorLM/GLM主干。字节审计确认该错配真实存在；token ID身份字段完全一致，只有chat
template和BOS操作元数据不同。

局部恢复配对没有搬来可观察能力；40层全局MoE配对产生了首次方向性增益。进一步消融证明，
可复现净收益主要来自精确router/shared expert，而不是整套高精度routed expert bank。当前最佳
组合是`v38 = v36 shared-backbone + v29显式工具序列策略头`；v29继续保留为稳定回滚入口。

| 版本 | 结构 | 独立短门 | 速度变化 | 决策 |
|---|---|---:|---:|---|
| v31 | 仅L39闭合MoE配对 | 5/8→5/8 | -25.24% | 否决，GGUF已回收 |
| v32 | L36–L39+final norm+output head | 4/8→4/8 | -23.44% | 否决，GGUF已回收 |
| v33 | 40层router+routed/shared expert全局配对 | 5/8→6/8，1胜0回归 | -6.99% | 保留研究候选 |
| v34 | v33全部40层MoE压到IQ3_S/F16 | 6/8→5/8，1回归 | -3.96% | 否决，GGUF已回收 |
| v35 | 仅前29层压缩，后11层高精度 | 4/8→4/8 | -6.43% | 否决，GGUF已回收 |
| v36 | 40层router/shared expert，恢复v6 routed bank | 独立4/8→5/8，1胜0回归 | +2.98% | 晋级为新核心 |
| v37 | v36 + v17连续Coder岛 | 5/8→5/8，0胜0回归 | -20.60% | 否决组合 |
| v38 | v36 + v29显式工具策略头 | 工具状态7/20→11/20，4胜0回归 | +0.34% | 当前最佳候选 |

v33的变化不是只有格式题：长上下文从`region/quota`均错变为`region`正确，规划从超预算组合变为
合法但少选一项，严格术语/JSON题从失败变为通过。证据规模仍小，不能宣称通用能力已提升。

v36只有`12.679GiB`，替换40层共240个Qwen3.6张量：精确router、shared expert、shared gate和
MoE入口norm；256路routed expert bank全部恢复为v6低比特权重。它在未参与构建的冻结16题上与
正式v17逐题完全一致（均`10/16`），生成速度`21.161→26.832 token/s`（`+26.80%`）。因此
v17连续Coder岛在这套原子门上没有提供额外通过项，不能抵消其`20.60%`组合速度成本。

v38不加载Coder岛，只在显式非空tools请求中启用约0.125MiB的v29策略头。冻结20题从v36的
`7/20`提升到`11/20`，修复4题、0回归；无tools固定请求与v36逐字段一致，显式tools固定请求
速度`28.689→28.787 token/s`。这是目前最强的本地可运行组合，但仍不是Claude/GPT级模型：
测试留出仍为`2/5`，通用冻结题的电脑操作为`0/2`，规划为`1/2`。

## 保留产物

- `pair_audit.json`：tokenizer与L39字节/路由数值审计。
- `build_qwen36_global_moe.py`：v33可复现构建器。
- `v31_gate.json`至`v35_gate.json`：全部在各候选生成前冻结的独立短门。
- `gate_report.json`、`v32_gate_report.json`至`v35_gate_report.json`：逐题输出与速度数据。
- `make_v34_tensor_types.py`、`v34_tensor_types.txt`、`v35_tensor_types.txt`：逐张量量化契约。
- `ColorLM-v33-Qwen36-Global-MoE-Pair.gguf.json`：360张量来源清单。
- `build_qwen36_shared_backbone.py`：v36可复现构建器与240张量来源报告。
- `v36_gate_report.json`、`v36_vs_v17_full16_report.json`：独立门与正式v17逐题比较。
- `v37_gate_report.json`：Coder岛组合零增益、明显降速的负对照。
- `v38_policy_gate_report.json`、`v38_runtime_check.json`：4项工具净胜、旁路与速度证据。
- `analyze_qwen36_routes.py`、`compare_layer_hidden.py`：全层隐藏状态/路由归因工具。

## 下一步

不再扫描统一量化精度，也不再把v17 Coder岛叠到v36。下一条应围绕v38仍稳定失败的规划、电脑
操作和工具澄清任务，冻结新的唯一可判定任务族，在v36隐藏坐标上重新采集/拟合原生策略头；禁止
复用v17训练hidden后继续调分。只有v36原生小头出现留出净胜且零回归，才进入v39运行候选。
