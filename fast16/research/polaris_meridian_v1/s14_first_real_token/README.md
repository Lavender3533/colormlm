# Polaris S14 第一个真实 token

本目录把冻结的 L42 CPU 参考泛化到固定 revision 的预注册 S14 层：

```text
[0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42]
```

执行入口只接受 BOS token 0，并严格遵守 route-first 生命周期：先校验当前层的
non-expert/router Range，执行 attention 与原生 router，再把恰好 6 个 expert ID
提交给 `RouteFirstSession`；只有此后才能读取或下载对应 36 个 routed payload。
它不会请求完整 embedding 或完整 shard，embedding 只读取 token 0 的 8,192 字节行。

前三层的 expert ID 来自 checkpoint 中物理 `I64 [129280,6]` 的 `tid2eid`，但 routed
权重仍由真实 hidden 的 `sqrt(softplus(gate))` 产生；其余层由
`(score + bias).topk(6)` 选 ID，并用未加 bias 的 score 归一化到 1.5。每层都执行
attention HC、FFN HC、6 个 routed expert 和 shared expert。ratio4/ratio128 层在 token0
没有压缩 KV 输出，但仍写入各自独立的 remainder state。

先跑零网络合同自检：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.s14_first_real_token.selftest
```

显式授权精确 Range 下载，并先停在 L0：

```powershell
$env:POLARIS_HF_ENDPOINT = "https://hf-mirror.com"
python -X utf8 -m fast16.research.polaris_meridian_v1.s14_first_real_token.executor `
  --download-missing `
  --download-budget-bytes 80216064 `
  --range-attempts 4 `
  --range-workers 3 `
  --stop-after-layer 0 `
  --report D:/models/Polaris-S14/s14_l0_real_report.json
```

完整 14 层运行需要为尚未命中的 expert 页提供足够预算，并移除
`--stop-after-layer`。任何 cache miss、SHA/shape/dtype 漂移、route 漂移、预算超限或
算子错误都会早停，并把 `status=blocked`、精确 stage/layer 和 traceback 写入 UTF-8
JSON 报告。

`--range-workers` 最大为 3，且只并发已经提交的 top-6 expert 页。执行器先按
`range_key` 去重并汇总整层 routed 字节；整层不能一次通过剩余下载预算时自动回到
顺序路径。并发不会改变每页独立的 206、Content-Range、SHA 与原子发布合同。

结果只能称为固定 S14 选择层、真实权重、原生 top-6 与真实 final head 的 CPU
correctness 证据。它不是完整 43 层 DeepSeek-V4，不宣称质量或速度。

## 已冻结的首次真实运行

`FIRST_TOKEN_REAL_REPORT.json` 固化了 2026-08-01 的完整 14 层运行：输入 BOS 0，
最终 argmax 为 token `108967`（`" Compression"`）。该次端到端进程新下载
962,592,768 字节；L0/L1 的 160,432,128 字节 routed payload 已由前序 checkpoint
缓存。观察耗时 714.769 秒只用于复现记录，不是性能基准。
