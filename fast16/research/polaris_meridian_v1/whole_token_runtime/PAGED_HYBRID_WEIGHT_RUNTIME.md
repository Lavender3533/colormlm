# FullDepth43 position0 自适应分页权重运行时

日期：2026-08-02

设备：AMD Radeon RX 5700 XT 8 GiB（Vulkan heap `8,573,157,376 B`）

状态：显存分配与四类真实上传哨兵通过；尚未闭合完整 token。

## 被实机否决的全常驻方案

原 Hybrid 计划请求：

```text
43 层 static       6,727,605,760 B
resident-small           279,040 B
routed 双 bank       160,432,128 B
head 双 chunk         67,108,864 B
总权重              6,955,425,792 B
workspace 门          536,870,912 B
```

即使拆成 48 个、最大不足 192 MiB 的 Vulkan allocation，当前 WDDM 桌面占用下仍在
L36/L39 出现 `VK_ERROR_OUT_OF_DEVICE_MEMORY`。因此 heap 总量不能代替进程瞬时预算，
也不能把“理论小于 8 GiB”写成可运行。

## 自适应分页方案

分配顺序固定为：

1. 512 MiB whole-token workspace；
2. resident-small；
3. routed 双 bank；
4. head 双 chunk；
5. static 双 stream bank；
6. 按 L0→L42 贪心常驻静态层，第一次可选分配失败后其余层流式。

真实独立 arena 运行：

```text
resident_static_layers       32
streamed_static_layers       11
essential_bytes       1,111,699,968
resident_static_bytes 4,993,579,264
allocated_device_bytes 6,105,279,232
recurring_static_upload_bytes 1,734,026,496
allocation_ms              543.4714
```

可选静态缓存分配失败不会改变数值路径；43 层仍全部执行，只改变 11 层权重从何处取得。

## 真实上传链

所有 payload 只能经 `VerifiedMappedAssetStore` 完整 SHA-256、只读 mmap 后进入 staging。
实机同时验证了四种不同目标：

- 常驻 static tensor；
- 首个 streamed static layer；
- L0 routed top-6 bank；
- BF16 head 第 0 个 4096-row chunk。

四段各回读 32 B，共 128 B，与 mmap 源逐字节一致。

逐资产上传的首版观测：

```text
startup_static_ms      92,198.7870
transfer_submits             1,088
startup_static_bytes 4,680,484,452
```

按物理层分组、批量 SHA 与一次 submit 后：

```text
resident_static_layers        32
streamed_static_layers        11
startup_assets_uploaded    1,158
startup_assets_deferred       405
startup_static_bytes 4,993,827,860
startup_static_ms      26,590.2743
transfer_submits                36
first_streamed_layer           L32
first_streamed_bytes   167,299,160
routed_L0_bytes         80,216,064
head_chunk0_bytes       33,554,432
sha256_bytes         6,300,404,844
sentinel_bytes                 128
```

新路径在上传更多静态字节的情况下，观测墙钟仍为旧路径的约 28.84%，submit 减少
96.69%。两次 WDDM 可用显存不同，因而这不是严格同层数单变量 A/B，但方向和收益量级明确。

## 结论边界

- 已证明：当前桌面状态下有一种真实可分配、可上传、保留完整 43 层数值语义的权重布局。
- 未证明：完整 attention/router/MoE/HC/head 已在同一个 whole-token command graph 中运行。
- 未证明：当前达到可聊天速度、20–50 token/s、Kimi 前端或 Claude/GPT 质量。
- 下一硬门：让 concrete `Position0LayerBackend` 直接消费这些真实 buffer，完成
  `BOS → L0..L42 → chunked head → token 5 → 原子 commit`。
