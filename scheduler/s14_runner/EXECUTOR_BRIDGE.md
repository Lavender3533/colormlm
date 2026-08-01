# NativeS14Executor Python 子进程桥

`SubprocessNativeExecutor` 是持久、fail-closed 的 UTF-8 JSONL 客户端。它覆盖：

1. `embed_row`
2. `attention_then_route`
3. `routed_and_shared_moe_then_hc_post`
4. `hc_head_norm_full_logits`

JSONL 仅传递协议身份、request ID、position、layer、state epoch、Range artifact
句柄和二进制 tensor view。响应行硬限 1 MiB；`hidden_values`、`state_values`、
`logits` 和任何 token 字段都会 poison 并终止子进程。

## 二进制 arena

公开构造器固定使用：

- hidden：`BF16 little-endian [4, 4096]`；
- logits：`F32 little-endian [129280]`；
- recursive state：`NativeState.arena_bytes`，其 `BufferSlice` offset/bytes/dtype/shape
  通过 `state.layout` 一并发送；
- logits scratch：在 recursive arena 后按 4096 字节对齐。

每个 tensor view 都含绝对路径、offset、bytes、dtype、shape；每次变更响应必须同时
回显整个 opaque state arena view 与实际写入的 hidden view。Rust 随后验证文件长度、
descriptor 身份、position/layer/epoch、hidden 实际变更、完整覆盖 sentinel 以及
BF16/F32 有限值。bridge 拥有临时 arena，退出时删除；默认总大小硬限 512 MiB。

## 错误和超时

以下任一条件都会 kill worker 并永久 poison bridge；带 `&mut NativeState` 的调用也会
同步设置 `state.poisoned=true`：

- worker error、EOF、非 UTF-8/非 JSON、request/op 顺序漂移；
- 响应超时；
- shape/path/offset/bytes/dtype/position/epoch 漂移；
- hidden 未更新、logits 未完整写入或出现 NaN/Inf；
- JSONL 携带 tensor 数组或 token 字段。

小尺寸 fixture shape 只存在于 `cfg(test)` 私有构造路径。端到端 runner 测试让
fixture head 返回 32 个 logits，随后由 runner 的生产 `129280` 硬门拒绝并 poison
state，因此测试不会得到或冒充模型 token。

本模块不提供真实 Python kernel worker，不读取或下载权重。生产 peer 仍需实现原生
embedding、attention/router、routed/shared MoE、HC/KV 状态和完整 head logits。
