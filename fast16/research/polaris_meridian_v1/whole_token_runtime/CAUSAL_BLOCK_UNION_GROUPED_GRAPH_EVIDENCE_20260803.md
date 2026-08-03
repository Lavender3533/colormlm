# S14 K=4/8 union Range 与 grouped MoE graph 结构证据（2026-08-03）

## 本刀真实完成

- `s14_causal_block_union_materializer.rs`
  - 从同层 K=4/8 份真实 `RouteDecision` 重建完整 `FullDepthExpertCatalog` identity。
  - 按 `(expert_id, W1/W2/W3, weight/scale)` 去重并强制专家升序、每专家 6 个物理 Range。
  - 对 orchestrator 的瘦计划逐项复核 `tensor/range_key/bytes/order`，每专家必须精确等于 `EXPERT_PAGE_BYTES`。
  - 解析本地 Range proof 后，用跨层/跨 token 常驻的 `VerifiedMappedAssetStore` 做批量 SHA/mmap；热命中不重复 SHA。
  - 直接把各 Range lease 写入持久 staging，不构造整 union 临时 Vec；Vulkan 上传合同固定为一个连续 `vk::BufferCopy`。
  - local-only 默认不触网；显式 fetch 仍复用既有逐 lane top-6 Range transport，fetch 后重新执行 union proof/SHA 复核。
- `s14_causal_block_grouped_graph.rs`
  - graph 生命周期常驻 2 个 command pool、43 个 transfer command、43 个 compute command、2 个 staging 和双队列 timeline。
  - 每 block 只 reset 2 个 pool，telemetry 固定 `resource_allocations_this_block=0`。
  - 每层只允许 1 次 union transfer submit 与 1 次 grouped compute submit；recorder 回执必须覆盖全部唯一专家和 K×6 assignments，`serial_token_forward_calls=0`。
  - staging 复用只等待对应旧 transfer；union bank 覆盖通过 compute timeline 排序；成功 seal 与失败 abort 都执行联合 drain。

## 当前 production 接线边界

两个组件已经由 `ssd_inference/src/lib.rs` 编入 crate，但主线程同时拥有的
`s14_causal_block_vulkan_backend.rs` 仍是 fail-closed 骨架；其三个数值入口仍明确返回
“recorder/uploader 尚未接线”。因此当前没有模型可运行路径，也没有端到端 token/s 数据。

精确接线顺序：

1. backend/runtime 生命周期持有一个 `S14CausalBlockUnionMaterializer`、一个
   `S14CausalBlockGroupedGraph` 和真实 grouped shader recorder。
2. `begin_full_depth_block` 调用 `grouped_graph.begin_block`。
3. K-lane attention/router 完成后缓存同层 K routes。
4. `materialize_union_ranges` 调用 `build_causal_block_union_identity_plan` →
   `materializer.materialize` → `grouped_graph.upload_union_layer`。
5. `run_grouped_moe` 调用 `grouped_graph.record_and_submit_grouped_moe`，把强回执映射为
   `S14CausalBlockGroupedMoeOutput`。
6. `seal_full_depth_layers` 调用 `seal_and_drain`；任何错误调用 `drain_and_abort`，随后沿用
   现有 lane checkpoint/最长一致前缀 rollback。

## 秒级门

- `cargo check --offline -p ssd_inference --lib`：通过（dev，仓库既有 warnings）。
- union 定向门：2 passed，0 failed，0.35s。
- grouped graph 定向门：1 passed，0 failed，0.00s（编译缓存外）。
- 两个新增 Rust 文件各自已通过 rustfmt；全 package `cargo fmt --check` 被主线程同时新增且
  尚未格式化的 `examples/s14_ratio4_global_topk_numeric.rs` 阻断，未改该无关文件。
- 未启动模型，未运行旧榜，未做 release build，未启动 3000/11435。

## 尚缺的真实 Vulkan 部分

- K-lane causal attention/router shader、descriptor 与无逐 lane host wait 的 route 输出。
- 实现 `S14CausalBlockGroupedMoeRecorder` 的真实 grouped expert shader/pipeline/descriptor；
  当前只有持久 command graph 和强回执门，绝不声称已经执行 MoE 数值。
- batched final head、K 份 device checkpoint arena 导出与 backend factory/runtime 字段接线。

## 性能表述边界

这是结构/编译证据，不是局部 kernel benchmark，更不是端到端 token/s。K=8 最坏每层
48 个专家页，即 641,728,512 B；43 层约 27.59 GB/block（约 3.45 GB/生成 lane-token，
尚未计 attention/head/状态）。若这些页每层都必须跨 PCIe 上传，仅权重带宽就会把
RX5700XT 的端到端上限压到远低于 20–50 token/s；达到目标必须依赖显著页复用、分页预取/
压缩或减少跨 PCIe 字节，不能由本结构门推导。
