# Qwen3.6 解码同步开销审计 (2026-07-26 深夜场)

## 背景
- Qwen3.6-35B-A3B-UD-Q4_K_M (21GB, qwen35moe arch), RX 5700 XT Vulkan + `--n-cpu-moe 34`
- 实测解码 85.36 ms/token (11.71 t/s), bs=1 时 70 graph splits/token
- 前一场工作流产出"CPU 专家 ∥ GPU shared expert 共执行手术方案A"(5-8 人日, 预期 5-12ms/token)
- 对抗审查 agent 死于瞬时 403, verdict=null → 本场人工补审

## 人工对抗审查结论: 方案A 的收益前提被推翻

### 翻案证据 (全部有 file/log 实锤)
1. **shared expert 极小**: `expert_shared_feed_forward_length = 512` (server 日志 kv 30),
   每层 3×2048×512 ≈ 3.15M 参数 ≈ 1.8MB @Q4_K → GPU ~6µs/层, 34 层 **~0.2-1ms/token**。
   方案A 假设的"每层 0.1-0.3ms 共 3-8ms"**高估 20-50 倍**(设计时没拿到真实 n_ff_shexp)。
   → **重叠共执行的矿几乎是空的, 方案A 原设计不值 5-8 人日。**
2. **routed 专家也是 512**: `expert_feed_forward_length = 512` (kv 29)。
   CPU 每 token: 34 层 × 8 专家 × 3 矩阵 × 2048×512 @Q4_K ≈ **482 MB** ÷ RAM 有效带宽
   (~35-45 GB/s) ≈ **11-16 ms** 计算下限。
3. **字节账对不上**: CPU ~12-16ms + GPU ~5-8ms (含 attention/lm_head ~2GB 读) + 拷贝 <1ms
   = **~20-24ms**, 实测 85.36ms → **残差 ~60ms = 纯同步/提交/唤醒固定开销 (~70%+)**。
   与 v4 (IQ3_S, 29 层) 时代审计的 roofline 18% 结论同构。
   ÷ ~68 次排空/token ≈ **~0.9ms/次固定成本**。

### 固定开销的机制拆解 (代码实锤)
每次 sched 排空 (ggml_backend_synchronize on Vulkan) 的路径:
`ggml_vk_synchronize` (ggml-vulkan.cpp:14126) →
- **空 submission 挂 fence** (14171: `queue.submit({}, ctx->fence)`) — 每次一个 vkQueueSubmit
- `ggml_vk_wait_for_fence` (2101): 若 almost_ready_fence_pending → **waitForFences 内核等待**
  (Windows WDDM 唤醒延迟 0.1-1ms) + 自旋等主 fence
- **almost_ready 在解码必然 pending**: 14890 行 `almost_ready = 剩余节点 < n_nodes/5`,
  每层 GPU split 只有 ~15-30 节点 → 最后节点必触发 → **每次排空都付内核等待**
- `ggml_vk_graph_cleanup` (13568): command pool 清理/semaphore 销毁/event 重置, 几十 µs
另: 每个 CPU split 的 graph_compute 有线程池唤醒/join 成本 (×34/token)。

### 方案A 的一个正确遗产 (若后续仍要做共执行)
阻塞式预取可以**完全替代** event 机制: 在 shared split 提交前对下一 CPU split 的输入做
同步 D2H (只等 attn+router 真依赖), 然后异步提交 shared, CPU 与 shared 自然并行。
- 不需要 Vulkan event/semaphore 改动
- galloc 数据竞争消失 (读完才提交 shared), 不需要 FLAG_OUTPUT 缓解
- 佐证: out_memcpys 只在 fence 等待后执行 (14181-14184), 原 async+event 设计对非 pinned
  目标本来就不成立; 而 CPU 侧 input_cpy 在 Vulkan_Host pinned buffer (日志: Vulkan_Host
  compute buffer 72.29 MiB), 阻塞 D2H 走直连 DMA
但矿只有 ~0.2-1ms, **优先级降到最后**。

## 已落的两个补丁 (待编译验证)
1. **`GGML_SCHED_PROFILE=1`** (ggml-backend.cpp): compute_splits 9 个位点 ns 级分解,
   每 128 token dump 到 stderr。位点: dst_drain_input / dst_drain / expert_copy /
   src_drain / dst_drain_fb / blocking_copy / compute_cpu / compute_gpu / final_sync。
   判读: compute_cpu = CPU 专家真实耗时 (含线程池唤醒); src_drain+dst_drain+final_sync
   = host 等待; compute_gpu = 提交侧 host 成本; blocking_copy = 拷贝。
2. **`GGML_VK_SPIN_FENCE=1`** (ggml-vulkan.cpp wait_for_fence): 跳过 almost_ready 的
   waitForFences 内核等待, 纯自旋主 fence (主 fence 信号 ⇒ 同队列先提交的 almost_ready
   必已信号, 直接 reset 安全)。烧一个核换掉 WDDM 唤醒延迟。

## 实验队列 (顺序执行, 每步一次 A/B)
1. 编译 (需先杀 llama-server 进程解锁 exe): `cmake --build build --config Release --target llama-server`
2. **基线分解**: GGML_SCHED_PROFILE=1 启动, 跑 ~256 token 解码, 读 [sched-prof] 行
   → 确认残差 ~60ms 落在哪些位点
3. **自旋 A/B**: GGML_VK_SPIN_FENCE=1 (+PROFILE), 同 prompt 对比 t/s
   预期: 若内核等待是主犯, 收益可达 10-30ms/token 量级; 若无效果, 主犯在提交侧/线程池
4. 视 2/3 结果选下一刀:
   - src_drain 大 + 自旋有效 → 继续削 fence 路径 (eager fence: graph_compute 末批直接挂
     ctx->fence, synchronize 省掉空提交)
   - compute_cpu 远超 11-16ms → 线程池唤醒问题, 试 --threads 调优 / threadpool 常驻
   - compute_gpu 大 → 命令缓冲录制/提交侧成本, 考虑减少 split 数 (方案C 的 router 迁 CPU
     反而增加跨设备点, 需重估!)
5. 方案C (router/求和迁 CPU) **重估后再动**: 它省的是拷贝字节 (KB 级) 和触点, 在 ~0.9ms/次
   固定成本模型下, 减少排空次数才有意义, 减字节几乎无意义

## 结论一句话
真正的敌人不是"串行没重叠"(重叠矿 ~0.2-1ms), 而是 **~68 次/token × ~0.9ms 的
fence/提交/唤醒固定开销 (~60ms, 占 70%)**。先测量分解, 再选刀。

---

## 实验结果 (2026-07-26/27 夜, 全部已跑)

### 基线分解 (GGML_SCHED_PROFILE=1, 稳态窗, ms/token)
src_drain 26.7 / compute_cpu 25.4 / compute_gpu(提交) 10.8 / blocking_copy 8.2 /
final_sync 8.1 / 其余 1.3 → tracked 80.1。**~44ms 是等待/同步机制费, 翻案确认。**

### 已判定的刀
1. **GGML_VK_SPIN_FENCE=1: 赢, +14-16ms (82.65→68.47 ms/tok 干净首轮, 12.1→14.6 t/s)**。
   已固化为 start_qwen36_bare.py 默认 env。分解: src_drain −5.6, copy −2.2, cpu −4.8
   (核不睡频率不掉), gpu提交 −2.7。它在基线之后跑还更快, 热漂移只会低估收益, 结论稳。
2. **--poll 100: 无效** (67.90 vs 68.47 持平), 已从脚本撤销。compute_cpu 超理论值的
   主因不是线程唤醒, 更可能是 6 线程 gather 下 RAM 有效带宽 ~23GB/s 的现实。
3. **GGML_VK_EAGER_FENCE=1 (graph 末批直接挂 fence 省空提交): 负结果, +6-9ms 反而更差**
   (同期对照 92.09 vs 83.03 稳态)。机制未完全定位 (cmd_buf 回收两路径等价, 已排除)。
   补丁保留 env 门控休眠, 默认关。别再顺序 A/B 追它。

### 测量方法论教训
- **热漂移**: 连续压测 ~40 分钟后同配置稳态从 64 → 83 ms/tok。今后 A/B 必须同热窗
  交错 (ABAB), 或每臂前后各插一次对照。
- **ngram 投机污染轮次**: 服务器生命周期内 ngram 表跨请求累积, round 2+ 被起草加速
  (接受率见过 0.897)。干净对比只用各自 server 首轮 (round 1)。

### 下一刀 (未做, 排队)
**合并 fence 周期** (compute_splits 手术): 目前每 CPU 层 = 1 次排空 fence 周期 + 2 次
各带独立 submit+fence 的同步读 (hidden+ids)。改为: graph 尾不挂 fence → 两个 D2H 用
get_tensor_async 录进新 ctx → 一次带 fence 的提交 + 单次等待。src_drain+blocking_copy
合计 ~33ms, 地板 ≈ GPU 真依赖 + 每层 1 个机制周期 ≈ 15-20ms, **潜在 −10~18ms**。
~30-40 行改动, 全部原语已验证存在。做完这刀再看 final_sync (6 次/token) 与
compute_gpu (0.31ms×35 提交录制)。

### 当前状态
生产配置 = start_qwen36_bare.py (SPIN_FENCE 默认开): 干净 ~14.6 t/s, 投机热态 20-24 t/s。
对比昨晚初测 11.0-11.7 t/s。
