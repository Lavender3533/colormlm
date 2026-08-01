# ColorLM 推理架构 HANDOFF

日期：2026-07-31  
结论：唯一推荐 **R256-CNOB**；当前仅研究候选，不声称能力提升

2026-07-31更新：60-token一次采集已完成；当前Coder donor的固定alpha与单donor no-op LOTO
均失败（2/10任务改善，平均NLL变化+0.34862），所以R256-CNOB没有运行时晋级资格。8105的
v17已恢复。后续K3或其他donor必须生成自己的capture并从本文件的同一门重新开始。

## 主机下一次单 GPU 实验：最短顺序

1. 只加 capture tap，不先实现新融合。`qwen35moe.cpp` 三处：
   - base LM head 后、donor 分支前：raw `base_logits`（当前约 2349 行）。
   - `donor_terminal` 完成 `inp_out_ids` gather 后、output norm 前（约 2353--2357 行）。
   - donor `ggml_mul_mat` 后、`ggml_mean` 中心化前：raw `donor_mapped_logits`（约
     2362--2367 行）。保留新指针，不能采已经中心化/乘 alpha 的张量。
2. capture 开关只在研究运行启用。一次 graph evaluation 覆盖 teacher-forced shard 的所有输出
   位置；同一启动写出 base logits、所有 donor raw mapped logits、terminal hidden 和 manifest。
   格式严格按 `capture_schema.json`，不要每个 alpha 重启模型。
3. 关闭模型后在 CPU 回放：

```powershell
python fast16/research/inference_arch/offline_bus.py replay capture.npz `
  --alphas 0,0.001,0.003,0.01,0.03 --output alpha_report.json
```

4. 先验收 `alpha=0` 精确等于 base；再用同一 capture 做温度、rank 64/128/256/384 和门的
   离线拟合。不要从 smoke60 选择新的全局 alpha。
5. rank-256 头逼近通过后，才实现 sparsemax `(no-op, donor1, donor2...)` 门。no-op 支持集必须
   精确返回 base，并跳过 `U_i q_i` 大投影。
6. donor2 必须用 LOTO 比较 `base+donor1+donor2` 对 `base+donor1`；task-bootstrap 95% CI
   上界 `<0`、冲突集不回归、最坏任务过预算，三者缺一不可。
7. 第一次实验不要同时接 DSpark 或新专家缓存。质量门通过后，再分别做独立短 A/B。

## 为什么不是 v19 固定 alpha

- v19 的 211.88 MiB 输出头每 token 约 269.54 M MAC，独立短门明显变慢。
- `alpha=0.03` 在独立 smoke60 平均 NLL 回归 `+0.0087696`，35/60 token 变差。
- R256-CNOB 不含运行时全局 alpha：`g=sparsemax(A phi+b)`，
  `z=z_base+sum_i g_i*d_i`；base-only 槽给出精确 no-op。
- 当前 rank-256 估算为 34.22 M MAC、28.37 MiB，分别约缩小 7.88x 和 7.47x；是否保真必须
  用 capture 证伪，不能据估算宣称提升。

## 候选和判定

| 候选 | 主要代价 | 用途 | 快速否决条件 |
|---|---:|---|---|
| Dense-CNOB | 269.54 M MAC / 211.88 MiB | 隔离验证 no-op 门 | LOTO 质量不过或激活仍与 v19 同速 |
| **R256-CNOB** | 34.22 M MAC / 28.37 MiB | **唯一推荐** | rank-256 留出保真或条件互补不过 |
| C2F-Exact | 约 2.76 M MAC，host 保留全头 | greedy/草稿候选 | top-1 recall <99.9% 或尾部概率无界 |
| R256-DSpark | draft + batched target | 分布不变的加速 | 接受率 <75% 或速度提升 <15% |
| 动态专家缓存 | 容量/上传依真实 expert 大小 | MoE I/O 加速 | trace 模拟收益 <15% 或显存越界 |

DeepSeek-V4-Flash-0731 的 43 层、4096 hidden、256/top-6、DSpark rank-256 是任务给定设计输入；
公开站点受用户策略阻止，未独立核验。借鉴仅限 rank-256 有界旁路、top-6/256 的小活跃集和
43 层跨层预取，不移植未经验证的细节。

## 多 donor 必交报告

- 主比较：`NLL(B+D1+D2)-NLL(B+D1)`，不是 `D2` 对 base。
- task 级 LOTO 和 task-cluster bootstrap 95% CI。
- code/tool 分组、决策 token、最坏任务、leave-one-donor-out。
- donor top-1 冲突集的 NLL delta、仲裁正确率和各 donor 支持率。
- exact no-op 率；task id/文本/关键词不得进入门特征。

## 已交付与自测

- `ARCHITECTURES.md`：五个候选的数学、计算/内存估算和可证伪实验。
- `capture_schema.json`：一次采集文件契约。
- `offline_bus.py`：纯 CPU alpha 回放、sparse no-op、LOTO 和冲突仲裁原型。
- `test_offline_bus.py`：4 项自测。

已执行：

```text
python -m unittest -v test_offline_bus.py
Ran 4 tests in 13.558s
OK
```

自测只证明数值路径、alpha=0、no-op 和条件互补统计代码成立；没有运行模型或长榜，也不构成
真实能力结论。
