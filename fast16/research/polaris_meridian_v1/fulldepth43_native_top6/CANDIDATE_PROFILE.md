# FullDepth43 Vulkan candidate 端到端剖析

## 结论

现有单 token candidate 的真实端到端墙钟是 `96.1953254s`，不是 GPU kernel 的
`0.73413264s`。43 层合计 `94.3488587s`，已经解释总墙钟的 98.08%；Vulkan worker
自报合计 `9.209151s`，其中 GPU kernel 合计仅 `0.73413264s`。因此旧运行仍有
`86.9861744s` 位于 worker 自报墙钟之外，当前最高优先级是 Python CPU 热路径、
权重物化/capture/验证边界，而不是继续只抠 shader kernel。

这些数字来自既有 candidate 报告，并未重新占用 GPU。旧报告没有 attention、HC、
router、compressor/indexer、文件 I/O 的子阶段计时，所以这 `86.986s` 必须保持为
`unattributed_residual`，禁止硬分摊。

## 已落地的测量基础件

`candidate_profile.py` 提供可撤销的上下文剖析器，不修改 `executor.py` 或 shader：

- 记录 attention、compressor/indexer、router、HC、Range、tensor load、FP8/FP4
  materialize、linear、capture、final head 的 inclusive/exclusive 墙钟。
- 逐层记录时间，并把 `PersistentVulkanWriteback.execute` 墙钟继续拆成
  Python→Rust execute、worker 自报、GPU kernel、worker 非 kernel 和 Python IPC/校验。
- `memmap + materialize` 明确标注为“页缺失/读取与解量化混合”，不伪称纯磁盘 I/O。
- 离开上下文后恢复全部原方法；单元测试覆盖嵌套 exclusive 计时、worker 拆分和恢复。

`run_candidate_profile.py` 是独立单-token入口，默认不开 FP8 cache；它强制全 43 层
Vulkan、禁止 CPU fallback，并同时写出模型报告、runtime profile 和端到端摘要。

## 本轮最高收益改动与下一候选

主线已删除 `_linear_fp8/_linear_fp4` 每次调用后的显式 `gc.collect()`。旧 candidate
运行仍含这项开销；按约 296 次/token、单次约 59ms 的先前探针，只能给出
`约 17.5s/token` 的待实测节省估计，不能从 96.195s 直接减掉后当成新成绩。
剖析报告会同时记录“被测运行”和“当前源码”的 `gc_removed`，防止混淆。

本模块还提供默认 8GiB 的 `MaterializedFp8Cache`：跨 token 有界复用只读 FP8
解量化权重，LRU 淘汰并暴露 hit/miss/resident bytes。默认不缓存 FP4 routed experts，
避免与 Vulkan worker 的 10GiB payload LRU 重复占内存。它只有包住 token2+ 才可能
命中，目前仅通过合成命中/只读/容量测试，没有端到端加速声明。

## 可证实的旧运行分布

| 边界 | 秒/token | 占总墙钟 |
|---|---:|---:|
| 端到端 | 96.1953 | 100.00% |
| 43 层墙钟合计 | 94.3489 | 98.08% |
| 层循环之外 | 1.8465 | 1.92% |
| Vulkan worker 自报 | 9.2092 | 9.57% |
| GPU kernel | 0.7341 | 0.76% |
| worker 内非 kernel | 8.4750 | 8.81% |
| worker 自报之外、尚未归因 | 86.9862 | 90.43% |

逐层原始分布见 `CANDIDATE_PROFILE_REPORT.json`。

## 下一次短测合同

1. 合入 `gc_removed` 后运行一次相同单 token candidate，墙钟上限 5 分钟。
2. 用 `CandidateProfiler` 采完整阶段，不再只依赖旧层摘要。
3. 报告当前端到端 TPS，并单列各阶段；任何缓存加速必须同时报告 hit rate、RAM 和输出 token。
4. 若显式 GC 删除没有复现预期收益，按剖析最大 exclusive 阶段选择下一项；不得反向用
   `0.734s` kernel 推导整 token TPS。
