# Polaris S14 Range → Runner 桥

`SubprocessRangeProvider` 用一个持久 UTF-8 JSONL 子进程把 Rust
`LocalS14Runner` 接到 Python `RouteFirstSession` / `RangeCache`。

固定顺序是：

```text
hello(固定 revision + s14_top6 + download_authorized)
  → prepare_base(L0 额外含当前 token 的 embedding row；其余为 non-expert + router)
  → executor 消费已校验 embedding row
  → Rust native attention / official current-token router
  → prepare_routed(exact top-6 + shared pages ready)
  → Rust native routed/shared MoE
  → release_layer
  → 最后一层释放时返回 HC head / norm / BF16 lm head 页面
  → executor 计算完整 logits，Runner 自己做 argmax
```

Python 只返回已由 `RangeCache` 校验的本地绝对路径、字节数、SHA-256
证明和 TOFU/authoritative 标记。Rust 再校验路径、文件长度、页面类型和
expert ID，然后把句柄放入 `ReadyBaseLease` / `ReadyRoutedLease`。
路由仍由 Rust native executor 计算；Python 不能从 sidecar 猜测 expert。
首层只返回当前 `token_id` 对应的 8,192 字节 BF16 embedding 行，不会为了
一个 token 读取完整 1.06GB embedding 矩阵。

默认 worker 是严格离线的：

```powershell
python -X utf8 scheduler/s14_runner/python/range_worker.py `
  --catalog D:/models/Polaris-S14/route_first_catalog.json `
  --cache-dir D:/models/Polaris-S14/range_cache
```

只有启动 worker 时同时显式传入 `--download-authorized` 和正数
`--download-budget-bytes`，并且 Rust `RangeBridgeConfig.download_authorized=true`，
hello 安全门才会通过。catalog 中的 `download_authorized` 必须始终为
`false`，不能自行授权。

任何协议错误、非 JSON stdout、超时或子进程退出都会 kill worker
并 poison provider，不会返回 ready lease/token。若原生 router 在 top-6
提交前失败，清理路径使用 `abort_layer`，不把半完成层标记为 ready。

这个桥证明了调度和安全闭环，不证明已执行模型 token、质量提升或速度。
