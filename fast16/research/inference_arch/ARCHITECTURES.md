# ColorLM 下一代推理架构候选

日期：2026-07-31  
状态：CPU/只读研究；没有运行模型，没有能力提升结论

## 0. 结论

唯一推荐：**R256-CNOB（Rank-256 Confidence-gated Neural Output Bus）**。

它保留 base logits 为不可破坏的安全路径，用含显式 no-op 槽的 sparsemax 门逐 token 选择 donor；
donor 输出头以 rank-256 因子替代 v19 的 131,612 行全量 Q6_K 投影。推理公式中没有全局常量
`alpha`。动态专家缓存和推测解码是后续正交加速，不应与第一次质量实验捆绑。

## 1. 已知证据和边界

### 1.1 v19 必须解释的反证

本地 `fast16/research/v19_dual_head/` 给出：

- v19 输出头映射 131,612 个 donor token，覆盖 donor 词表 86.62%、base 词表 53.00%。
- 输出头包为 222,169,248 B，即 211.88 MiB；投影为 `131612 x 2048`，约
  269.54 M MAC/token。
- 12-token 小样本上 `alpha=0.03` 看似改善，但独立 smoke60 平均 NLL 变化为
  `+0.0087696`，20 win / 35 loss / 5 equal，故回归。
- 最差任务 `tool-run-command` 平均 NLL 变化 `+0.0474791`；代码组和工具组平均方向都变差。
- 相邻短门没有质量差异，但解码速度分别从 `19.51 -> 15.86 token/s` 和
  `17.41 -> 13.19 token/s`。
- Vulkan compute buffer 从 563.3 MiB 增到 1504.44 MiB。

因此问题不只是 alpha 选错：固定全局增益无法表达“此 token 不应干预”，而全词表输出头在每个
token 都支付高成本。

### 1.2 DeepSeek-V4-Flash-0731 的使用口径

任务给定字段为 43 层、hidden 4096、256 专家、top-6、DSpark rank-256。仓库中没有该模型的
官方 config；本次对 Google、Hugging Face、GitHub 的直接核对均被用户级站点策略阻止。因此：

- 本文把这些字段当作**任务输入假设**，不声称已独立核验。
- 不猜测 DSpark 未给出的训练损失或内核；只借鉴“rank-256 有界旁路/草稿通道”这一原则。
- 43 层和 256/top-6 只用于层流水与缓存估算，不主张把层数或专家数直接移植给 ColorLM。

可借鉴点：top-6/256 每 token 只激活 2.34375% 专家，适合“先路由、后异步预取”；43 层提供
跨层流水窗口；4096 hidden 使全量输出投影更贵；rank-256 适合作为显式计算预算，而非另一个
无约束大头。

## 2. 统一记号

- `z_b in R^V`：base logits。
- `h_i in R^{d_i}`：donor i 的 terminal hidden。
- `M_i`：donor i 精确 token-bytes 映射到 base 的 token 集。
- `l_i`：donor 在 `M_i` 上的原始 logits。
- `P_i`：把 `M_i` 上的向量 scatter 到 base 词表，未映射位置写 0。
- `T_i`：仅用训练折拟合的 donor 温度，不是全局融合强度。

先构造 shift-invariant 的 donor 相对意见：

```text
r_i(v) = l_i(v)/T_i - z_b(v)/T_b,                      v in M_i
c_i    = E_{v ~ p_b(. | M_i)}[r_i(v)]
d_i    = P_i(r_i - c_i)
```

`d_i` 在 base 条件概率下零均值，避免 v19 将“映射组整体抬高”误当成知识增益。

门只读数值状态，不读 prompt 文本、关键词、task id 或标签：

```text
phi_t = stats(z_b, h_1..h_D, coarse_1..coarse_D)
g_t   = sparsemax(A phi_t + b),   g_t in simplex(no-op, donor_1..donor_D)
z_t   = z_b + sum_i g_{t,i} d_i
```

当 sparsemax 支持集只有 no-op 时，`g=(1,0,...,0)`，于是 `z_t=z_b` 精确成立。这里是连续训练的
神经路由/MoE 门，不是关键词路由，也不是用手写任务 `if` 冒充融合。

## 3. 候选架构

### A. Dense-CNOB：全量头 + 置信 no-op 门

直接在 v19 的完整 mapped Q6_K 头前加入上述门；门激活 donor 时才执行原投影。

```text
l_i = W_i RMSNorm(h_i)
z   = z_b + sum_i g_i d_i
```

当前 donor 的单次激活代价仍为 269.54 M MAC、211.88 MiB 权重。优点是与 v19 全量头最接近，
可隔离验证门；缺点是只要门激活就仍很慢。

可证伪实验：在同一 capture 上 LOTO 训练门；若独立任务折不能同时做到 `mean NLL delta < 0`、
最坏任务不回归且 no-op 覆盖实际出现，则否决。若激活 token 的 wall time 与 v19 相当，也不作为
速度方案。

### B. R256-CNOB：rank-256 低秩头 + sparse no-op 门（唯一推荐）

对每个 donor 拟合：

```text
q_i       = R_i RMSNorm(h_i),       R_i in R^{256 x d_i}
coarse_i  = C_i q_i,                C_i in R^{512 x 256}
g         = sparsemax(A phi(z_b, h_i, q_i, coarse_i) + b)
l_i       = U_i q_i                 仅对 g_i > 0 的 donor 执行
```

`W_i ~= U_i R_i` 可用激活加权 KL/回归拟合；门训练加 `lambda * (1-g_noop)` 风险正则，让没有可靠
增益证据的 token 回到 base。`lambda`、温度和门权重都必须只在训练折拟合，不能查看留出标签。

当前 2048 hidden、131,612 mapped token、rank 256 的估算：

- 投影：`2048*256 + 131612*256 = 34.22 M MAC/token`，是全头的 1/7.88。
- Q6_K `U` 为 27,638,520 B；FP16 `R` 为 1,048,576 B；加 I64 map 和 norm 共约
  **28.37 MiB**，是 211.88 MiB 的 1/7.47。
- base-only 时只付 `R h + 512` 个 coarse 分数，约 0.66 M MAC/donor，不执行 `U q`。
- gate/温度参数远小于 1 MiB。

可证伪实验分两关：

1. 头逼近关：rank 64/128/256/384 同 capture 比较。rank-256 若留出 top-1 一致率低于 99.5%，
   或相对完整 donor 头使目标 NLL 平均恶化超过 0.002，则否决 rank-256。
2. 总线关：预注册独立 code/tool 任务做 LOTO；`base+donor1+donor2` 必须相对
   `base+donor1` 的 task-bootstrap 95% CI 上界小于 0，冲突集平均 delta 不大于 0，且最坏任务
   不超过预注册回归预算。速度只做相邻短 A/B；不得用质量失败换速度。

风险：低秩重建误差与门误判会叠加。必须先用已采集 logits 离线证明，再接运行时。

### C. C2F-Exact：粗到细候选 + 精确行投影

保留完整 Q6_K 头在 host/mmap，只缓存候选行：

```text
q          = R h
clusters   = TopB(C q)
S          = base TopK union selected clusters union control tokens
l_i[S]     = W_i[S] h               精确 Q6_K 行
```

取 512 clusters、B=4、均匀簇约 1,028 行，估算为 `R h` 0.52 M + centroid 0.13 M + 精确行
2.11 M，合计约 **2.76 M MAC/token**。4,096 行 GPU cache 约 6.56 MiB，但 host 仍保留
211.88 MiB 完整头。

可证伪实验：完整头 capture 上测候选 top-1 recall、target-token recall 和 log-sum-exp 尾部误差。
任一独立任务 top-1 recall <99.9%，或无法给出精确/有界归一化概率，则不能用于 NLL 门和采样；
最多作为 greedy 草稿器。

### D. R256-DSpark：rank-256 块推测/验证

rank-256 旁路一次提出 K 个 token，R256-CNOB 完整目标在一个 batch 中验证。接受规则必须是标准
speculative decoding 的概率修正，而不是“两个模型同意就输出”：

```text
a_j = min(1, p_target(x_j | prefix_j) / q_draft(x_j | prefix_j))
reject 后从 normalized(max(p_target - q_draft, 0)) 采样
```

它只降低达到**同一 target 分布**的成本，不提高能力。理想整块接受时速度上限近似：

```text
speedup <= K*C_target(1) / (K*C_draft + C_target_batch(K))
```

可证伪实验：K=2/4/8，要求分布一致性测试通过、接受率至少 75%、相邻 wall throughput 至少提升
15%；否则关闭。第一次单 GPU 实验不做此项。

### E. Router-Aware Expert Cache：43 层 top-6 动态专家缓存

对层 `l`：

```text
E_{l,t}       = Top6(router_l(h_{l,t}))
prefetch      = TopP(predict(E_{l+1,t} | router_l, history))
evict score   = recency + frequency + predicted_reuse - transfer_cost
stall_bytes   = sum_{e in E - cache} bytes(e)
```

容量应以 2--4 个 active set 的小窗口起步，并让下一层上传与当前层计算重叠。只作量纲示例：若
hidden=4096、expert intermediate=2048、Q4，每专家约 `3*4096*2048*0.5 = 12 MiB`；top-6
每层 72 MiB，256 个全驻留为 3 GiB/层。实际 intermediate 未由任务给定，必须用真实 config
重算。

本地已有正反证：v17.2 四层各 32 槽、约 216 MiB 的短测从 4.45 到 10.41 token/s；但 64 槽
把显存推到约 7363 MiB 并降至 0.89 token/s。另一条 29 层 x 8 槽只有约 25%--27% 命中且未超
基线。因此“更多槽”不是答案。

可证伪实验：先只做 trace replay，预注册命中率、每 token 上传字节、重叠后 stall 和显存上限；
模拟收益不足 15% 不接 GPU。即使加速成立，它也不构成质量提升。

## 4. 一次采集，多 alpha/多门离线回放

### 4.1 最短运行时 tap

现有 `llama.cpp/src/models/qwen35moe.cpp` 的最短接线是：

1. **base logits**：`cur = build_lora_mm(model.output, hidden)` 之后、进入
   `if (neural_output_head)` 之前（当前约 2349 行）。
2. **donor terminal hidden**：`inp_out_ids` gather 完成之后、donor output norm 之前（约
   2353--2357 行）。
3. **raw mapped donor logits**：`ggml_mul_mat(output, donor_norm)` 之后、`ggml_mean` 中心化之前
   （约 2362--2367 行）。必须保留 `mapped_logits_raw`，不要采集后续已中心化或乘 alpha 的张量。

只在显式 capture 开关下为三个张量命名并通过 eval callback 复制到 pinned host；普通运行路径不得
复制。`inp_out_ids` 应覆盖 teacher-forced 序列中所有待测位置，使一次 graph evaluation 产生一整个
shard，而不是每个 alpha 重跑。

### 4.2 文件和启动策略

数组契约见 `capture_schema.json`。每个 shard 存：

```text
base_logits[N,V_base]
labels[N], task_ids[N], sample_ids[N]
donor_i_logits[N,M_i], donor_i_base_ids[M_i]
donor_i_hidden[N,D_i]
```

统一用 FP16/BF16 落盘并在离线脚本中升到 FP64。60-token 的当前 base+donor mapped logits 约
`60*(248320+131612)*2 = 43.48 MiB`，hidden 仅约 0.23 MiB/donor。manifest 必须记录模型、
token map、teacher、build 的 SHA-256 和 tap 的 norm 前后位置。

下一次 GPU 只启动一次 capture 运行点；同一组前缀同时写 base、所有已接 donor 的 raw logits 和
terminal hidden。之后关闭模型，所有 alpha、温度、门和 rank sweep 均在 CPU 离线做。

旧 v19 的精确重建：

```text
branch_i = scatter(l_i - mean_{M_i}(l_i))
z_alpha  = z_b + alpha * branch_i
```

命令：

```powershell
python fast16/research/inference_arch/offline_bus.py replay capture.npz `
  --alphas 0,0.001,0.003,0.01,0.03 --output alpha_report.json
```

必须先验证 alpha=0 的 logits 字节相等或 NLL 差为 0；随后才能信任 sweep。新 R256-CNOB 不选择
“最佳全局 alpha”，而是在 train folds 拟合温度、低秩头和 sparse gate，再在完全留出的 task 上
评估。

## 5. 多 donor 的互补性和冲突仲裁

新增 donor2 不能只与 base 比；主比较固定为：

```text
Delta_2 = NLL(base + donor1 + donor2) - NLL(base + donor1)
```

必要条件：

- task 级 leave-one-task-out，门和温度只在其余 task 拟合。
- `mean(Delta_2)<0` 且 task-cluster bootstrap 95% CI 上界 `<0`。
- 预注册 code/tool 各组、决策 token、最坏任务；不能靠单一任务拉动。
- 冲突集 `argmax(donor1) != argmax(donor2)` 单独报告，冲突集平均 `Delta_2<=0`。
- 报告 exact no-op 率、各 donor 稀疏支持率、pairwise disagreement 和去掉任一 donor 的消融。
- donor3 以后逐个做条件边际，不能把多个新 donor 打包后隐藏负贡献。

仲裁由同一个 sparsemax 门完成，输入是 logits/hidden/coarse 数值统计和 donor 间公共 token 子集的
分布距离；task id 仅用于 split/report。CPU 原型的 `validate` 命令已经实现上述基线比较、LOTO、
task bootstrap 和冲突集报告。

## 6. 当前原型的适用范围

`offline_bus.py` 已实现：旧 alpha 回放、mapped 残差、sparsemax no-op、数值门、LOTO、多 donor
条件互补和冲突集。它使用完整 donor logits 构造部分门特征，目的是验证统计契约；运行时版本应
把这些特征替换成 rank-256 coarse head 统计，才兑现 no-op 时跳过大投影的成本模型。

合成自测只能证明实现内部一致，不能证明真实模型质量。真实结论必须等一次 capture 后用预注册
留出实验证伪。
