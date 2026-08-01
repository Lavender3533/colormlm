# 现有运行时审计与首个速度里程碑

审计日期：2026-08-01。范围仅限本地代码与已有报告；未启动模型、未下载权重。

## 可直接复用

### llama.cpp 真实图内槽池

- `llama.cpp/src/llama-slot-pool.{h,cpp}` 已有固定 VRAM 槽、每层 logical-to-slot LUT、
  `ggml_backend_tensor_set` 上传和 `MUL_MAT_ID` 所需张量。
- `llama.cpp/src/models/qwen35moe.cpp` 已接入 exact miss-and-stage：命中更新 LRU，miss
  上传真实专家后再更新 LUT，不再把 miss 静默映射到 slot 0。
- 现有实现只从完整 host-visible packed tensor 取源字节，仍不是 300B+ donor 的独立 SSD
  冷页；全局 mutex、同步上传及每 token 的图回调也是当前热路径成本。

已有 `fast16_exact_slot_pool_benchmark_report.json` 证明 16-token 固定贪心输出一致，但速度
证据否定默认启用：初扫 `8x28` 为 24.23 tok/s、看似 +7.59%，相邻 128-token A/B 却是
16.98 对 22.52 tok/s，即 **-24.60%**。因此它是正确性底座，不是已完成的速度方案。

### scheduler 独立组件

- `scheduler/gguf_reader`：单/多分片 GGUF 元数据、tensor offset、mmap byte view。
- `scheduler/ssd_inference/src/expert_reader.rs`：用 `seek + read_exact` 绕开 4 KiB mmap
  fault，已有多分片位置表与并行读取。
- `expert_loader.rs`：Vulkan transfer queue、多个 staging/fence、批量磁盘读和上传。
- `vram_pool.rs`：VRAM 固定槽与 LRU。
- `scheduler-core` + `predictor`：跨层共现预测、异步预取模拟。
- `expert_cache`：VRAM/RAM/SSD/HDD 层级账本，但只按“专家个数”计容量，未实际搬运。

### 必须先修的工程风险

1. `expert_loader.rs` 的 staging 固定为 1.37 MiB，大小检查只有 `debug_assert!`。Kimi K3
   单专家约 16.734 MiB；release 构建会绕过检查，随后对 staging 创建过长 slice，存在
   越界风险。必须改为按最大页动态 staging 或硬错误，不能直接复用。
2. `batch_enqueue` 在 miss 数超过 pipeline depth 时会复用同一个 staging/fence；当前先读
   完全部 miss 再提交，重复 pipeline 槽可能被后续读取覆盖。批量长度必须分波次。
3. `VramPool::reserve` 在传输完成前就发布索引。异步调用者若立即 lookup，可能看见尚未
   ready 的槽。需要 `Loading/Ready` 状态和 fence 完成后的原子发布。
4. 当前 GGUF 单专家切片假定专家轴物理连续。这对现有 qwen GGUF 已由实现与字节 spot
   check 支持，但新量化/新架构仍须用 tensor layout 契约验证。
5. 现有预测器数字大多来自模拟，不是 RX 5700 XT 上集成推理证据；不能据此声明 20--50
   tok/s。

## 本目录补上的缺口

- 统一的 Safetensors/GGUF 专家页目录，面向 300B+ 分片，不物化整模型。
- Safetensors 默认执行组件完整性硬门；本地 Coder shard 40 实测发现 398 个完整三组件专家、
  114 个仅有 down 组件，已拒绝把该单分片作为可运行目录。
- 以真实字节数而不是专家个数约束 RAM 热集。
- 合并 Kimi/DeepSeek 单专家的相邻 packed/scale span。
- single-flight、异步预取、线程独立文件句柄和零拷贝缓存命中 API。
- 不依赖大权重的快速并发与吞吐自检。

## 本机短测结果

- 合成 Safetensors：32 页 × 2 MiB，10 项断言全部通过；进程未缓存读取
  1,662.67 MiB/s，四 worker 预取墙钟 1,733.65 MiB/s，RAM 命中约 397.69 ns/get。
  这些数据仍受 Windows OS 文件缓存影响。
- 真实 `ColorLM-v6-Q3Router-Fused-A1.gguf`（13,731,822,496 B）：一次头部扫描建立
  40 × 256 = 10,240 个专家页，专家银行净载荷 11,492,392,960 B；每页 1,122,304 B、
  三个不连续量化 span。目录建立约 13.94 秒，只发生一次，不在 token 热路径。
- 从真实 GGUF 均匀跨层/专家选择 64 页，共 68.50 MiB：四 worker 首轮墙钟
  **207.39 MiB/s**（OS cache uncontrolled）；8 个均匀页 direct/cache SHA-256 全一致；RAM
  命中约 **390.17 ns/get**，不复制 payload。
- Rust 复用基础：`cargo test -p expert_cache -p gguf_reader` 为 5/5 通过；
  `cargo check -p ssd_inference` 通过，但有 21 个 warning，不能代替运行时集成验证。

速度硬边界也已量化：K3 原生 92 个 MoE 层 × top-16 = 每 token 1,472 个专家页，单页
16.734 MiB。按本次 207.39 MiB/s 随机多 span 读取，20 tok/s 只容许 0.620 个 miss/token，
所需命中率 **99.958%**；50 tok/s 只容许 0.248 个，所需 **99.983%**。即便乐观按
3,500 MiB/s 顺序盘，也分别要求 99.290% 与 99.716%。更根本的是 K3 报告的 104B 激活
参数，即使 Q4 每 token 仅扫描一次也需约 52 GB；RX 5700 XT 448 GB/s 理论上限仅
8.62 tok/s，未计计算与同步。因此目标不能是“原样运行 K3 经络”，必须是 300B+ 总容量、
每 token 约 3--5B 活跃的新执行架构。

## 还没有完成

1. 尚未把目录和页读取器移植到 C++ 并接入 `qwen35moe`/未来 Kimi、DeepSeek 原生图。
2. 尚未实现 Windows page-aligned pinned staging、Vulkan 双缓冲和 fence-ready LUT。
3. 尚无 Kimi K3 93 层真实路由轨迹，无法证明 50--70 GiB 缓存覆盖率或前端专家是否集中。
4. 尚无 300B+ donor 的原生 attention、latent projection、shared expert 和 KV/state 常驻
   契约；只分页 routed FFN 不能单独形成“能力经络”。
5. 尚无真实端到端 tok/s。磁盘吞吐只回答页搬运上限，不代表模型生成速度。

## 建议的第一个可证伪里程碑

选择一个**单层**真实巨型 donor（优先现有 Kimi K3 source-plan 的 16.734 MiB MXFP4 专家）
完成下列闭环，再扩到全深度：

1. 从真实 Safetensors 分片建立目录，随机抽 32 个专家逐 span SHA-256 对照 Range 计划；
2. C++ 动态 staging + 4 波次预取，fence 完成后才发布 LUT；
3. 记录 route hit、SSD bytes、read/upload/stall 微秒和 evictions；
4. 用合成路由先证明页内容与 slot 映射永不串专家；
5. 再采一条真实 93 层路由轨迹，测 8/16/32/64 GiB 热集命中与 stall；
6. 只有 `p95 miss stall` 与实际计算重叠后，才把“20 tok/s”设为端到端晋级门。

这条路径不训练，也不以小模型替代目标；它只解决北极星巨型 donor 经络能否在本机按需
存在并保持低热路径开销。
