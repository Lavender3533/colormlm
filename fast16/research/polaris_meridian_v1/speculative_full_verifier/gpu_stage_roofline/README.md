# Polaris 草稿 + FullDepth43 GPU 阶段级 roofline

这是一个不加载模型、不下载权重的离线调度与成本原型。它把已提供的 GPU 计时锚点、真实字节和两 token 冷缓存反例拼成可运行、可证伪的阶段时间线。S14/v38/v47 只做草稿；最终提交 token 必须由 FullDepth43/native-top6 exact causal verifier 完整执行 43 层后给出，不允许跳层。输出是乐观 roofline，不是完整 GPU token 实测。

## 运行

在仓库根目录执行：

```powershell
python -m fast16.research.polaris_meridian_v1.speculative_full_verifier.gpu_stage_roofline report `
  --asset-root D:/models/Polaris-S14 `
  --output fast16/research/polaris_meridian_v1/speculative_full_verifier/gpu_stage_roofline/roofline_report.json
```

也可注入待实测的阶段时间，观察结论是否被推翻：

```powershell
python -m fast16.research.polaris_meridian_v1.speculative_full_verifier.gpu_stage_roofline simulate `
  --target-tps 20 --expert-hit-rate 0.5 `
  --hc-ms-per-layer 0.1 --router-ms-per-layer 0.02 `
  --norm-head-ms-per-token 0.5 `
  --command-buffer-mode resident_per_layer --submit-overhead-us 8
```

测试：

```powershell
python -m unittest discover `
  -s fast16/research/polaris_meridian_v1/speculative_full_verifier/gpu_stage_roofline/tests `
  -t . -v
```

## 阶段和证据等级

每层的最短依赖顺序是：

1. HC-attention pre → attention（其中 `wq_a` 是已测必要子操作）→ HC-attention post。
2. HC-FFN pre → native top-6 router。
3. routed 专家页查找/PCIe copy 与 shared expert 并行；除此之外不声明 copy/compute 重叠。
4. 六个 routed experts → routed/shared 累加 → HC-FFN post。
5. 进入下一保留层；L42 后执行最终 HC → norm → BF16 head/top-1。

冻结锚点如下：

- CPU 热缓存完整 S14 token：57.979147 s/token。只报告真实 CPU 基线和目标倍数，绝不外推 GPU。
- RX 5700 XT L42 单 expert 最小链：0.157038 ms，5 dispatch，100 次 timestamp 迭代。
- L42 `wq_a` 必要子操作：0.0838004 ms，1 dispatch，100 次 timestamp 迭代。
- L42 top-6 routed+shared 最小 GPU-resident batch：1.3696 ms，35 dispatch，当前只有 1 次 timestamp；不含完整官方 BF16/requant 边界。
- S14 固定层为 14 层：0、1、2、6、7、14、15、22、23、30、31、40、41、42。

因此当前 GPU 已知必要子操作/包络为 `14 × (1.3696 + 0.0838004) = 20.3476056 ms/token`。attention 余项、HC、router、norm/head 和 host 提交尚未实测；报告把它们设为 0 只是为了给出最乐观硬上界。

## 可证伪边界

- 在上述零未知阶段假设下，20 tok/s 需要至少约 30.095% 的 routed 专家页在验证前已驻留设备；任何新增阶段时间都会提高该门槛。
- 50 tok/s 的周期只有 20 ms，已经小于 20.3476056 ms 已知锚点。当前单 token 内核即使 100% 专家命中也不可达；至少需要 1.017381 倍的 measured-kernel 吞吐提升，而且这仍不给未测阶段留下空间。
- batch 行只给出“若达到理想线性批收益”所需效率，不把 batch 外推当成实测。command-buffer 行同样只给 host 提交开销的硬预算。
- 真实 token0→token1 在 84 个 `(layer, expert)` 页中只交集 8 页（9.5238%），对应 76 个 expert miss 和 1,016,070,144 B。该点低于 20 tok/s 乐观门槛；离散地至少需要命中 26/84 页，即除了上一 token 的 8 页，还需历史工作集或预测预取成功覆盖 18 个冷页。
- 8/84 仅是两个 token、仅保留上一 token 专家的真实反例，不能泛化成任何长序列稳态接受率或命中率。

报告同时列出 8 GiB VRAM、32 GiB RAM、118 GiB SSD 和 22.03 GB/s PCIe 的容量/带宽上界。容量没有扣除 KV cache、activation、workspace 和 runtime；“装得下多少页”不等于“路由会命中多少页”。

## FullDepth43 K=1/4/8

同一份 `roofline_report.json` 的 `full_depth43_exact_verifier` 同时包含 K=1、K=4、K=8 在 20/50 tok/s 下的完整接受前沿。每个前沿逐项枚举接受前缀 `0..K`；一次 causal block 总会先让全部 K 个位置走完 43 层，之后才接受最长一致前缀。遇到首个不一致时提交前缀和一个 FullDepth 原生 fallback；全一致时保守地只提交 K 个草稿 token。

45 个真实 header 证明 43 层 `wq_a`、43×256 routed experts 和 shared experts 的 packed shape 完整且同构，并严格确认 native router 的真实结构（43 个 gate weight，L3..L42 共 40 个 gate bias）。因此报告可以把 L42 的 GPU 锚点投影为每个 FullDepth verified position 62.4962172 ms 的“当前已知子操作工作”，但它仍不是 FullDepth token 实测，并遗漏完整 attention、HC、router、norm/head、BF16/requant 和 runtime 开销。全接受时，仅让这部分投影达到 20/50 tok/s 就分别需要至少 1.2499243×/3.1248109× 吞吐提升；K>1 的行会给出相对于理想 K-way batch 的最低效率，绝不假设已经获得线性收益。

真实 `fulldepth_kadaptive_budget.json` 和 45 headers 给出非路由+BF16 head 为 7,786,627,272 B，在 22.03 GB/s 下每块冷扫描至少 353.455618 ms。因此 fixed 每块流式扫描时，K=1/K=4 的 20 tok/s 以及 K≤8 的 50 tok/s 都被固定字节直接否决；K=8、20 tok/s 仍需视块内专家页复用程度达到 70.2729% 至 96.2841% 的设备页命中。若固定权重常驻，8 GiB 理论上只剩 60 个专家页，而每个位置至少涉及 258 个逐层专家页；报告同时列出 resident-fixed 和任意最优 fixed/expert 驻留两种容量前沿。

45 个 base shard 合计 156,023,192,948 B，超过 118 GiB SSD；因此该 SSD 不能无外部后备地完整自包含 FullDepth43。所有 FullDepth 结论都是必要条件或不可能性证明，不代表 20/50 tok/s 已达成，也不宣称质量已达到 DeepSeek。
