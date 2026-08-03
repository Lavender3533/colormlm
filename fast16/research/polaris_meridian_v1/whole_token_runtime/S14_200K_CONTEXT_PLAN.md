# Polaris S14 200K 上下文状态计划（2026-08-03）

## 已落地的结构门

- `TokenStateTxn` 新增 ratio128 boundary，在 position `127,255,...` 必须同时发布
  当前 remainder 与已完成的 compressed KV。
- `DecoderStateV1` 不再以 position 127 作 host 上限；真实上限回到
  `max_seq_len`。
- `WholeTokenFutureBlock` 的 K=4/8 checkpoint 链可跨 ratio128 边界，仍要求
  一次 `batched_causal_whole_token` forward，禁止 K 次串行伪装。
- 定向门：position127 原子提交 `1/1`，K=4 跨边界 `1/1`。

## 200K 精确 ABI 容量

`LongContextMemoryPlan::target_200k()` 从 `NativeState` 的同一 BufferSlice ABI 计算：

| 部分 | 字节 |
|---|---:|
| 完整 flat arena | 1,393,885,184 |
| 禁止的整体 A/B | 2,787,770,368 |
| HC + 43层 window + compressor remainder | 17,874,944 |
| ratio4 indexer search resident | 268,800,000 |
| ratio128 coarse history resident | 32,010,240 |
| ratio4 main history pageable | 1,075,200,000 |
| 普通 token dirty write-set | 373,760 |
| ratio4 + ratio128 同时边界 dirty upper bound | 636,160 |

生产策略固定为：

1. HC/window/remainder 常驻。
2. ratio4 indexer 和 ratio128 coarse history 保持 GPU search-resident。
3. ratio4 main compressed history 放 host/SSD 页，只按 indexer 命中页进入 attention。
4. candidate 只保存 dirty pages，通过 logical length/epoch 原子发布；禁止为
   checkpoint/rollback 复制整块 1.394GB arena。

## 尚未完成

- production Vulkan ratio128 pool/norm/RoPE/finalize 与 device dirty writeback。
- 200K indexer 的真实长度性能和 selected-main-KV 页命中率。
- 200K prompt prefill 的 chunk/block 调度、中段/尾段取回和真实对话门。

因此当前只能写为“200K host 状态与容量边界已成立”，不能写为
“200K 可聊天”。
