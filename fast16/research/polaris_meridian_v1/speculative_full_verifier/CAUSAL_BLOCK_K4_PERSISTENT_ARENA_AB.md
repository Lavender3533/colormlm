# K=4 持久 Union Arena 相邻 A/B

日期：2026-08-02
设备：AMD Radeon RX 5700 XT 8 GiB

## 结论

持久 union arena 已通过完整43层的数值与局部速度门：172/172个BF16输出行逐字节一致，
同一worker内四次单层总wall为8573.0042 ms，K=4 block总wall为5262.9780 ms，观测
加速1.6289倍。逐层加速中位数为1.5870倍；排除L42一次异常的488 ms单层抖动后，总加速仍为
1.5724倍，因此对外只表述为约1.5--1.6倍。

这不是完整模型token/s。测量只覆盖43层同层MoE replay，不含attention、router、KV、HC和
final head；`speed_eligible_verifier=false`继续保留。

## 测量合同

- 一个持久Rust/Vulkan worker，禁用general/shared resident GPU cache。
- 每层先在独立capture root运行一次K=4 block，只用于把该层权重装入host verified cache。
- 紧接着在新鲜measure root连续运行四次单层，再运行一次K=4 block。
- A/B两侧host disk read均为0；所有分配、打包、上传、kernel、readback及输出发布均计入
  `wall_ms`。
- measure block必须报告持久buffer、零request allocation、一次copy和一次upload submit。
- 每行同时比较原始8192字节BF16 payload与SHA-256，任何一行不一致即整轮失败。

## 结果

| 指标 | 四次单层 | K=4 block | 变化 |
|---|---:|---:|---:|
| 43层worker wall | 8573.0042 ms | 5262.9780 ms | 1.6289x |
| GPU上传字节 | 18,128,766,976 | 11,523,654,144 | -36.43% |
| BF16精确行 | 172/172 | 172/172 | 无漂移 |
| measure request arena分配 | 不适用 | 0/43 | 全部复用 |

逐层最小加速为1.2468倍，中位为1.5870倍；L42的4.0302倍含明显单层侧系统抖动，不作为
可复现上界。完整逐层wall、kernel、上传量、arena telemetry和每行SHA位于
`CAUSAL_BLOCK_K4_PERSISTENT_ARENA_AB.json`。

## 实现边界

- 固定backing逻辑容量为1 GiB，但descriptor边界和copy字节始终使用本次
  `plan.arena_bytes`，不会访问未更新的旧尾部。
- arena按完整`GpuMoeIdentity`排序、去重和查offset，不依赖route slot。
- 四位置执行顺序、每行六个routed expert顺序、shared expert位置及BF16累加顺序保持不变。
- `alpha`、能力路由和供体内容均未改变；本轮只优化S14/FullDepth43速度数据面。

## 下一步

把attention、router、KV、HC、MoE和head收进单一Rust/Vulkan whole-token runtime，复用持久
command/descriptor资源，并在同一GPU context内执行真实连续K=4状态。只有完整token结果通过
数值、状态和相邻速度门后，才能更新端到端token/s。
