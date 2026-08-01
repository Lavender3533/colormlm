# v20 K3 双专家、共享执行与精度实验报告

日期：2026-08-01

## 结论

v20 没有建立可复现的能力增益，不能替换 v17。最初一次 F16 双专家 dev60 得到
`mean NLL delta=-0.1048089`，但在当前同一 teacher、计划、alpha、上下文与运行时上重跑为
`+0.00358585`；撤销批量出口后仍为 `+0.00315469`。旧正信号因此降级为不可复现结果。

共享运输工程仍有真实价值：E41/E780 的 `b_in/norm/b_out` 完全相同；共享一次入口运输、
两路 latent 专家、批量出口的 64-token A/B 从 `13.29996` 提升到 `14.43169 token/s`
（`+8.51%`），该短提示输出 SHA-256 一致。但这个速度事实不等于能力提升。

## 冻结结果

| 路径 | 双胶囊专家/桥权重 | mean NLL delta | LOTO | 判定 |
|---|---:|---:|---|---|
| F16 首次记录 | 182.03 MiB | -0.1048089 | 全正 | 不可复现 |
| F16 当前复现 | 182.03 MiB | +0.00358585 | 失败 | 否决 |
| F16 分离出口复现 | 182.03 MiB | +0.00315469 | 失败 | 否决 |
| Q4_0 专家 | 91.46 MiB | +0.08766037 | 失败 | 否决 |
| Q8_0 专家 | 122.96 MiB | -0.01410018 | 失败 | 不晋级 |
| 原生 MXFP4 专家 | 89.50 MiB | -0.01008845 | 失败 | 不晋级 |
| F16 prefill/MXFP4 decode v4 | 215.50 MiB | +0.00358585 | 失败 | 研究负对照 |

Q4 的相邻 64-token 测试为 F16 `21.3666`、Q4 `21.5230 token/s`（`+0.73%`），但质量反转，
因此跳过 Q8/MXFP4 的正式速度晋级。原生 MXFP4 是供体 packed/scale 到 ggml block 的无损重排；
抽样 61,440 个权重值相对现有 F16 解码逐值一致、最大绝对误差 0，但不同 Vulkan kernel 的
prefill 微小数值漂移仍会被后续层放大。

## 新增工程能力

- K3 v3 混合精度胶囊 ABI：桥 F16，专家支持 Q4_0/Q5_0/Q8_0/MXFP4，router F32。
- K3 v4 阶段自适应精度 ABI：同时装载 F16 与原生 MXFP4 专家；当前只作负对照，不进入默认路径。
- `compile_k3_q4_capsule.py` 可从已校验 v2 F16 胶囊构建上述运行包，并写入逐张量尺寸与 SHA-256。
- C++ 加载器验证格式、dtype、shape、bytes、SHA-256 与运行策略；alpha=0 物理旁路未改变。
- 两颗专家共享输入运输与批量输出图仍保留为速度路径；不再用旧 dev60 结果宣称能力增益。

## 产物

- `fast16/research/compile_k3_q4_capsule.py`
- `fast16/models/ColorLM-v20-K3-Q4-Shared-Trunk.k3plan.json`
- `fast16/models/ColorLM-v20-K3-Q8-Shared-Trunk.k3plan.json`
- `fast16/models/ColorLM-v20-K3-MXFP4-Shared-Trunk.k3plan.json`
- `fast16/models/ColorLM-v20-K3-Hybrid-Shared-Trunk.k3plan.json`
- `fast16/research/v20_k3_*_dev60.jsonl` 及对应 comparison JSON

## 下一步约束

停止在 E41/E780 上继续扫 alpha、精度或门控。下一能力候选必须换成连续供体块、末端输出证据
或更强基座，并先通过可复现的独立任务门；当前 v20 只保留工程 ABI 与负对照资产。
