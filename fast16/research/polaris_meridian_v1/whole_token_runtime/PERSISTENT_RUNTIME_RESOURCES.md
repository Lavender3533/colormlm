# S14 whole-token 跨 token 常驻资源证据

日期：2026-08-03

## 结论

本刀把以下资源从 production token 循环提升到 `S14Runtime`/example 外层生命周期：

- `S14Position0HybridUploader` 的 6 个 staging buffer、transfer command pool/buffer 和 fence；
- 单一 `VerifiedMappedAssetStore` 的只读 mmap 与 SHA 缓存，包含 backend 构造期 embedding/tid2eid 映射；
- 43 层 `S14Position0L0GraphPlan`；
- backend 同步 fallback 的 graphics command pool/buffer；
- whole-token transfer/compute 两个 command pool 及 75/120 个 command buffer。

每 token 只 reset 两个 whole-token command pool，并重置 uploader 的 static/routed/head 游标；不再
create/allocate 上述资源。动态 GPU top-6 仍逐层生成，36 个物理 Range 的身份、proof/SHA 与布局校验
仍走原合同，没有固定路由、跳过 SHA 或扩大 I/O 授权。

## 第二 step 结构 telemetry

`S14StepOutput` 和 production example 现在暴露：

```text
persistent_step_index
persistent_resource_allocations_this_step
persistent_command_pool_resets_this_step
persistent_resources_reused_from_previous_step
candidate_scoped_backend_rebuilt_this_step
```

结构合同强制第二个成功 step 为：

```text
persistent_step_index=1
persistent_resource_allocations_this_step=0
persistent_command_pool_resets_this_step=2
persistent_resources_reused_from_previous_step=true
candidate_scoped_backend_rebuilt_this_step=true
```

其中 `allocations=0` 只覆盖本刀列出的 persistent 资源类，不表示整个 token 路径没有任何 Vulkan/host
对象分配。定向单测同时拒绝 staging/store/graph/backend-command/whole-token-command 任一隐藏重分配。

## 仍为 candidate-scoped 的精确阻塞

`S14Position0L0GpuOwner<'ctx>` 仍持有 paged arena 中 static/routed/workspace `GpuBuffer` 的借用；adapter
同时持有当前 `S14Session` 的 candidate/sticky buffer，录制过程中生成绑定这些 buffer 的
`DescriptorBinder`。把该 owner 原样存回拥有 arena 的 `S14Runtime` 会形成 Rust 自引用；跨 session
复用 binder 又会把 descriptor 指向上一 session/candidate bank。

因此本刀没有伪装 descriptor 已常驻，显式固定
`candidate_scoped_backend_rebuilt_this_step=true`。随 owner 仍逐 step 重建的还有 pipeline/descriptor、
immutable 快照、router readback 和内部 timeline；terminal chain、state readback 与外部 paged timeline
也仍是 token-scoped。下一刀若继续提升，必须先把 pipeline owner 与 buffer binding 拆开：runtime 只持有
无借用 pipeline/layout，step 只创建或更新绑定当前 arena offset/candidate bank 的 descriptor 状态。

## 事务与失败恢复

- session 独占仍由 `S14Runtime::step(&mut self, &mut S14Session)` 保证；
- 成功路径：final wait → terminal/readback 销毁 → candidate backend 收敛 → timeline 销毁 → persistent
  command step 完成 → host/device 原子发布；
- 失败路径：timeline drain；若 drain 失败则 backend 退化到 device-wide idle；随后重置 persistent
  uploader/command 游标，最后 rollback candidate；
- cleanup 错误只附加到原模型错误，不覆盖原始 failure；
- abort 会回退 persistent step counter，已 drain 的半 token 可再次 begin，不会造成 host/command
  telemetry 序号永久漂移。

## 已跑秒级门

- `cargo fmt -p ssd_inference -- --check`：通过；
- `cargo check --offline -p ssd_inference --lib`：通过；
- `cargo check --offline --release -p ssd_inference --example s14_position0_paged_43_layers_real`：通过；
- `cargo check --offline -p ssd_inference --example s14_position0_synchronous_43_layers_real`：通过；
- `s14_runtime::tests`：`4/4`；
- `drained_partial_token_can_begin_again_without_reallocation`：`1/1`。

本轮没有启动模型、没有重跑 N=8、没有跑旧榜。权威基线仍是
`.tmp-polaris-tests/n8-production-20260803-proxy.stdout.log`：热态 position 0--6 均值
`14.638320s/token = 0.068314 token/s`。本刀只证明资源生命周期与结构 allocation 门，不提供新的
局部 kernel 数字，也不声称端到端 token/s 已提高。RX 5700 XT 8 GiB 上的物理带宽/显存上限、局部
kernel 吞吐和完整 token/s 仍须分开测量；删除这些重建不足以单独达到 `20--50 token/s`。
