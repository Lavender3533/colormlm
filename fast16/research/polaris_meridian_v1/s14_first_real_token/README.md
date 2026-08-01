# Polaris S14 连续两个真实 token

本目录把冻结的 L42 CPU 参考泛化到固定 revision 的预注册 S14 层：

```text
[0, 1, 2, 6, 7, 14, 15, 22, 23, 30, 31, 40, 41, 42]
```

默认执行入口从 BOS token 0 运行一个 token；`--token-count 2` 会在同一
`DecoderRuntime` 中继续 position1。每个 token 都严格遵守 route-first 生命周期：先校验当前层的
non-expert/router Range，执行 attention 与原生 router，再把恰好 6 个 expert ID
提交给 `RouteFirstSession`；只有此后才能读取或下载对应 36 个 routed payload。
它不会请求完整 embedding 或完整 shard，每个 token 只读取自己的 8,192 字节 embedding 行。
当前 correctness 范围明确只到 position1；position2 之后以及 ratio4/ratio128 的
首次压缩输出边界尚未实现，不能据此宣称已经具备任意长度解码。

前三层的 expert ID 来自 checkpoint 中物理 `I64 [129280,6]` 的
`tid2eid[current_token_id]`，但 routed
权重仍由真实 hidden 的 `sqrt(softplus(gate))` 产生；其余层由
`(score + bias).topk(6)` 选 ID，并用未加 bias 的 score 归一化到 1.5。每层都执行
attention HC、FFN HC、6 个 routed expert 和 shared expert。ratio4/ratio128 层在 token0
没有压缩 KV 输出，但仍写入各自独立的 remainder state。

跨 token runtime 长期复用一个 `RangeCache`，但每个 token 新建一个
`RouteFirstSession`。position1 从 token `108967` 自己的 embedding row 复制出四流输入，
不复用 position0 final 的单路 hidden；每层先对 q/kv 做 position RoPE，把当前 KV 写入
窗口后读取 p0+p1，再对 attention 输出做 inverse RoPE。ratio4/ratio128 的 main remainder
以及 ratio4 独立 indexer remainder 都从已提交 state clone 后追加；p1 不产生 compressed
block。只有全部 14 层、final HC/norm/BF16 head 与 argmax 都成功，14 层
`next_layer_states` 和下一个 token ID 才通过单个 snapshot 替换原子提交；失败会保留旧 state。

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

同一进程连续运行 token0→token1：

```powershell
$env:POLARIS_HF_ENDPOINT = "https://hf-mirror.com"
python -X utf8 -m fast16.research.polaris_meridian_v1.s14_first_real_token.executor `
  --download-missing `
  --download-budget-bytes 1395864371 `
  --range-attempts 4 `
  --range-workers 3 `
  --token-count 2 `
  --report D:/models/Polaris-S14/s14_two_real_tokens_report.json
```

`--range-workers` 最大为 3，且只并发已经提交的 top-6 expert 页。执行器先按
`range_key` 去重并汇总整层 routed 字节；整层不能一次通过剩余下载预算时自动回到
顺序路径。并发不会改变每页独立的 206、Content-Range、SHA 与原子发布合同。

结果只能称为固定 S14 选择层、真实权重、原生 top-6 与真实 final head 的 CPU
correctness 证据。它不是完整 43 层 DeepSeek-V4，不宣称质量或速度。

## 已冻结的真实运行

`FIRST_TOKEN_REAL_REPORT.json` 固化了 2026-08-01 的完整 14 层运行：输入 BOS 0，
最终 argmax 为 token `108967`（`" Compression"`）。该次端到端进程新下载
962,592,768 字节；L0/L1 的 160,432,128 字节 routed payload 已由前序 checkpoint
缓存。观察耗时 714.769 秒只用于复现记录，不是性能基准。

`TWO_TOKEN_REAL_REPORT.json` 固化了同一 `DecoderRuntime` 的连续运行：position0 复现
`0 → 108967`，position1 从 token `108967` 的独立 embedding row 开始并生成 token
`53`（`"S"`）。position1 logits SHA-256 为
`46b95489427932a0d5acfacd5ee6bc9ceac495df3daed5a6a58681a0d95a141d`；进程新增下载
1,016,078,336 字节。两 token 的 14/14 层相邻 SHA 均闭合，position1 的 ratio4
main/indexer 写 row 5、ratio128 写 row 1，active window 为 p0+p1 两行。观察总耗时
1,005.144 秒只作为复现记录，不是性能声明。
