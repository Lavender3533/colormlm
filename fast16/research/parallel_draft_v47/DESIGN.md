# v47 rank-64 动态 shortlist 草稿算法设计

## 1. 证据层级

三类指标不可互相替代：

1. **oracle/teacher-forced 覆盖**：同一 anchor 的 shortlist 是否含完整 oracle 与 v38 validator 轨迹；
   只是必要条件。
2. **自由滚动候选命中**：head 一次产生未来 block 后，v38 validator token 是否仍在冻结 shortlist，
   并记录首个错误是“候选遗漏”还是“候选内排序错误”。teacher token 不得覆盖 proposal。
3. **v38 验证接受长度**：从第 1 位原生 token 起，与同一 anchor 的 v38 自由贪心 token 逐位比较，
   首个不一致立即拒绝。只有这一层能进入速度下界。

这直接阻断 `408/408 → 可加速` 的错误外推。

## 2. 一次 cascaded block head

设 anchor terminal hidden 为 `h∈R^2048`、rank 为 64、未来位置数为 3：

```text
t1 = argmax(v38_native_logits)             # 固定，完全不训练
z2 = tanh(normalize(h) · W_in + b2)
z3 = tanh(z2 · W_23 + b3)
z4 = tanh(z3 · W_34 + b4)
score_j(c) = z_j · hash_key_64(c), c∈shortlist(h, context)
```

三个 `z` 在看到任何 proposal token 之前形成，损失固定为 `1.0*CE2 + 0.7*CE3 + 0.5*CE4`。
这避免串行未来头把第 2 位错误作为第 3/4 位的输入，也允许三个 shortlist score 在实现时批量计算。

hash key 是原型的明确可证伪选择：它把表示能力限制暴露出来，不需要完整词表行表，也不会偷偷引入
未来全词表投影。如果真实接受率不足，先判定该表示失败；不得在已消费 test 上换 key 或扫 rank。

## 3. 候选集

每个 anchor 仅使用当时可得信息，严格按以下顺序去重：

1. v38 原生 logits top-32；
2. 最近 96 个已提交上下文 token，按从近到远；
3. 仅 train 的 v38 validator 未来目标频次 top-64；
4. 达到 192 行立即截断。

oracle、validation、test 和 proposal 后验都不能扩候选。block 模式的候选在 anchor 固定，拒绝回放因此能
明确区分 shortlist 漏词与 head 排序错误。

## 4. Error replay

自由滚动保存首拒绝位置、proposal/validator token、proposal 前缀，以及 validator 是否仍在 shortlist。
只有 `split=train` 的记录可按固定权重回灌。validation/test 的拒绝位置只进入报告，训练器遇到它们会报错，
防止 Draft-OPD 所警告的离线覆盖与推理分布错位被留出集调参掩盖。

## 5. 精确成本口径

冻结上界 `C=192,D=2048,R=64,P=3` 下，单次 block head 为：

- 参数：`D·R + (P-1)·R² + P·R = 139,456`；F16 `278,912` 字节。
- FLOPs：hidden 归一化、key 浮点缩放、三层矩阵与三次 `C×R` 点积合计 `370,111`；另列 1 次
  sqrt、192 次 tanh。每周期生成12,288个hash元素的196,672次整数原语单列，不混入FLOPs。
- 参考带宽：权重、hidden、candidate ID、score 写出合计 `286,080` 字节/周期，不含缓存复用。
- Python 向量化参考峰值 scratch `61,184` 字节；流式 key 可降到 `12,288` 字节。

串行未来头共享一个 `R×R` 递归矩阵，所以少 `4,096` 参数，但需要 3 个 proposal-dependent score 阶段，
且比 block 多 128 次 token-key 加法。完整数字以 `cost_report.json` 为准。

## 6. 8% 停止门

覆盖门通过后才计算接受长度。对 validation+test 的平均接受草稿 token 数 `A∈[0,4]`，成本脚本使用
95% 单侧 Hoeffding 下界，再以最坏 `1.35` 个基座步验证成本和 192 行草稿成本计算：

```text
speedup_lower = lower95(A) / (1.35 + draft_flops / 6e9)
```

硬门故意只计已接受且可提交的草稿 token；`A+1` 常见解析式只作诊断，避免在尚无临时 sequence branch
证据时把额外 validator token 算成已兑现收益。样本少于64或下界小于 `1.08x`，均停止、不改 C++。
