# FullDepth43 / native-top6 reference

这是 DeepSeek-V4-Flash-0731 固定 revision 的完整 43 层 CPU/PyTorch
correctness 入口，不是 S14 跳层模式，也不是 top-1 近似。

硬合同：

- 层集合必须精确为 `0..42`，不允许 identity skip。
- 每层必须先执行 attention 与官方 router，再读取当前 token 的
  真实 top-6 expert 页。
- 每层都执行 top-6 routed experts + shared expert + mHC post。
- 连续 token 保留全 43 层 window KV 和 ratio-4/128 compressor remainder。
- forced-prefill 队列未耗尽时，下一个输入必须取官方 `token_ids[position]`；
  队列耗尽后才改用原生 argmax。
- 只有 43 层全部完成后，原生 HC/norm/BF16 head argmax 才能提交 token。
- 任何缺页、顺序、状态或预算错误都 fail closed，token/cursor/43 层 state
  一起原子回滚，不会伪造 token。

## 当前真实状态

本机已有完整官方 index 和 45 个固定 revision header，因此已在
`D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json` 生成了不含
payload 的全 43 层 catalog。

```text
catalog ranges                 67,612
static prerequisite ranges     1,564
static candidate-ready         1,564 / 7,786,905,820 bytes
static missing                     0 /             0 bytes
one-token native top-6 cold           3,449,290,752 bytes
current cold upper bound               3,449,290,752 bytes
```

当前 preflight 为 `ready`，静态缺页已归零。2026-08-02 的首次真实执行
已通过全部 `0..42` 层，每层记录 6 个唯一原生路由专家，最终
HC/norm/BF16 head 提交了真实 `token_id=5`。机器可读证据在
`first_real_token_report.json`；该结果只证明 correctness，不是速度或质量声明。

同日首个官方聊天 forced-prefill 也已完成。严格输入为
`[0,128803,30594,128804,128821]`，解码为
`<｜begin▁of▁sentence｜><｜User｜>你好<｜Assistant｜><think>`；五个位置均完成
43/43 层，forced cursor 最终为 5，最后原生 head 输出 `token_id=3648`，解码为
`好的`。完整报告 `first_preview_real_report.json` 为 315,496 bytes，SHA-256
`beaebe27d5a295d68bbf7b475841be428eca60504c792ac8d54b189e17f17908`，总 correctness
墙钟 4,218.604 秒，其中包含大量首次 Range 下载。一个合理首 token 仍然不是质量晋级，
但它证明了官方聊天编码、五位置 KV/compressor 状态和完整 43 层原生输出链闭环。

需要严格区分“运行时曾拥有状态”与“状态已持久化”。这次已完成的
`first_preview_real_report.json` 只包含 shape、摘要、route 和 ledger，不包含完整
43 层 window KV/compressor tensor，因此 **不可恢复，也不能事后伪造成
checkpoint**。要获得该前缀的首个可恢复状态，仍需在已有 Range cache 上执行
一次带 `--checkpoint` 的五 token 前缀。

该轨迹还提供了第一批真实速度结构证据。排除 BOS 特殊对后，相邻位置每层平均复用
`2.1938/6` 个专家；正常位置 1--4 的 K=4 联合专家数平均为 `15.4186/24`，即仅看
动态 routed expert 字节可少读约 35.76%。这证明 causal block 有实质空间，但不足以单独
推出 token/s；完整数据见
`../speculative_full_verifier/first_preview_route_speed_report.json`。

重建 catalog 并跑离线 preflight（不下载权重）：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.preflight `
  --rebuild-catalog `
  --asset-root D:/models/Polaris-S14 `
  --catalog D:/models/Polaris-S14/fulldepth43_native_top6_catalog.json
```

显式运行入口（已缓存路由页命中时可直接运行）：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run
```

首个官方聊天预览使用仓内 `first_preview_forced_prefill.json` 的 5 个 token：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
  --forced-prefill `
  --token-count 5
```

`--forced-prefill` 不带路径时固定使用上述仓内产物，也可显式指定其他通过
S14 format/revision/vocab/BOS/token-hash/decoder-consumption 验收的产物。中间四次
argmax 只记录为反事实输出，不会覆盖 forced 队列的下一输入。

## 连续生成 checkpoint/resume

`checkpoint.py` 使用严格 UTF-8 JSON manifest + 原始 binary tensor payload，
不使用 pickle。manifest 绑定固定 repo/revision/profile/tokenizer、position、
committed ledger、forced cursor、43 层 KV/compressor 形状和每个 tensor/整体
payload SHA-256。payload 先以内容寻址文件发布，manifest 最后原子替换。
任一缺层、截断、位翻转、ledger 断链或模型/tokenizer 不匹配都会在前向前拒绝。

首次生成可恢复的五 token 状态：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
  --forced-prefill `
  --token-count 5 `
  --checkpoint D:/models/Polaris-S14/checkpoints/first-chat-prefix.json
```

从 position 5 继续一个 token，并将新状态原子写回同一 checkpoint：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
  --resume-checkpoint D:/models/Polaris-S14/checkpoints/first-chat-prefix.json `
  --checkpoint D:/models/Polaris-S14/checkpoints/first-chat-prefix.json `
  --token-count 1
```

checkpoint 只消除已提交前缀的冷重放；新 token 本身仍需完整 FullDepth43
计算，因此它是连续性/可用性基础设施，不是 token/s 证据。

新 token 若命中未缓存专家页，要允许补页，必须同时显式传入
`--download-missing` 和不小于
preflight cold upper bound 的 `--download-budget-bytes`。catalog 本身永远保持
`download_authorized=false`。

`preflight_report.json` 只记录本机当前缺口；真实执行证据以
`first_real_token_report.json` 为准。

## Vulkan 单层桥

executor 可在真实 FullDepth 前向经过指定层时，固化已完成层前缀、FFN 激活、原生
top-6 route 和 42 个带 SHA proof 的 routed/shared payload：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
  --vulkan-bridge-capture <fresh-dir> `
  --vulkan-bridge-layer 42
```

当前只允许单 token，capture 目录必须不存在。只指定 capture 时仍是旧的
只读桥，不改变 CPU correctness 计算。

2026-08-02 新增了可选的持久 worker 回写：

```powershell
python -X utf8 -m fast16.research.polaris_meridian_v1.fulldepth43_native_top6.executor run `
  --vulkan-bridge-capture <fresh-dir> `
  --vulkan-bridge-layer 42 `
  --vulkan-writeback-worker scheduler/target/release/examples/s14_vulkan_numeric.exe
```

worker 在同一进程内复用 Vulkan device/pipeline，实现 `w1/w3 -> BF16`、
`route-weight-before-w2`、每 128 元素 E4M3FN 重量化和 `w2 -> BF16`。executor
仍计算一次 CPU 参考；只有 4096 个 BF16 逐位相等才使用 **GPU 返回的
tensor** 重建 `hc_post` 并继续前向。任何协议、SHA、shape、非有限值或数值不等会
poison worker 并在 token commit 前 fail closed。

真实 RX 5700 XT 证据和声明边界见
`scheduler/ssd_inference/FULLDEPTH43_VULKAN_BRIDGE.md`。

## Vulkan 全层 A/B 门

`run_vulkan_all_layer_ab.py` 把单层写回扩展为三个严格相邻阶段：43 层逐层
CPU/GPU BF16 对齐、CPU warm baseline、关闭 CPU 专家双算的 43 层 Vulkan 候选。
任一层不相等、发生 fallback、少跑一层或最终 token 漂移都会拒绝结果：

```powershell
cargo build --release --example s14_vulkan_numeric `
  --manifest-path scheduler/ssd_inference/Cargo.toml

python -X utf8 -m `
  fast16.research.polaris_meridian_v1.fulldepth43_native_top6.run_vulkan_all_layer_ab `
  --worker scheduler/target/release/examples/s14_vulkan_numeric.exe `
  --output-root <fresh-output-dir>
```

即使通过，该门也只证明 43 个 MoE 分支来自 Vulkan；attention、HC、router 和 head
仍在 CPU，不能写成完整 GPU token、20/50 token/s 或能力提升。

## 连续 token 剖析入口

`run_candidate_profile.py` 支持一次严格执行 `1..16` 个连续 token，并记录每个 token 的
精确 wall time、43层覆盖、committed ledger、FP8 materialization cache、Range proof cache
与持久 Vulkan 边界。下面是当前两 token 技术门；它不下载缺失权重：

```powershell
python -X utf8 -m `
  fast16.research.polaris_meridian_v1.fulldepth43_native_top6.run_candidate_profile `
  --worker scheduler/target/release/examples/s14_vulkan_numeric.exe `
  --vulkan-final-head-worker scheduler/target/release/examples/s14_bf16_head.exe `
  --vulkan-final-head-scratch .tmp-polaris-runs/fd43-proof/head `
  --output-root .tmp-polaris-runs/fd43-proof `
  --token-count 2 `
  --fp8-cache-gib 6
```

输出目录必须是新目录。生产 worker 默认禁用已被真实 A/B 否决的 GPU payload LRU；只有显式
设置 `POLARIS_GPU_PAYLOAD_CACHE_GIB=1..7` 才会启用实验缓存。6GiB实验会在8GiB显卡上OOM，
4GiB实验为0命中且回归约9.37%，不得用于正式入口。默认3个 Range worker会并行验证本地已缓存页，
且保持输入顺序；MoE生产路径还会创建约100.5MiB固定 Vulkan 上传槽并逐层复用。

2026-08-02 的最新正式门已进一步把43层 packed-FP8 attention接入同一 token worker。236个
attention权重槽跨token驻留于RX 5700 XT（`4,775,506,560 B`，约`4.448 GiB`），第二token
`236/236`命中且零权重重传。完整两-token墙钟从`93.7806s`降至`63.8851s`，输出仍为`[5,223]`；
第二token层主体为`20.6784s`。slot身份绑定kernel、weight与scale的完整tensor/SHA，并有258槽、
`5 GiB`逻辑常驻硬上限，超预算在分配前拒绝。运行入口新增：

```powershell
python -X utf8 -m `
  fast16.research.polaris_meridian_v1.fulldepth43_native_top6.run_candidate_profile `
  --worker scheduler/target/release/examples/s14_vulkan_numeric.exe `
  --vulkan-attention-worker scheduler/target/release/examples/s14_vulkan_numeric.exe `
  --vulkan-attention-shared-batch `
  --vulkan-attention-output-chain `
  --vulkan-final-head-worker scheduler/target/release/examples/s14_bf16_head.exe `
  --vulkan-final-head-scratch <scratch-dir> `
  --output-root <fresh-output-dir> `
  --token-count 2 `
  --range-static-prefetch `
  --range-gpu-verifier-ownership
```

Vulkan与NumPy/OpenBLAS的归约顺序存在最高`6.103515625e-05`的已知投影差异，短轨最终输出不变，
但不能声称所有层逐位等价。实现、数值边界和A/B证据见`FULLDEPTH43_VULKAN_ATTENTION.md`与
`FULLDEPTH43_VULKAN_ATTENTION_AB.json`。这个入口仍是新架构连续性与剖析工具，不是可交互聊天服务。

`--range-gpu-verifier-ownership`只允许在零下载、全层Vulkan Attention与MoE、关闭CPU verify和
fallback的正式候选上使用。它让GPU专属页由Rust worker在计算前做唯一内容SHA；Python仍完整验证
所有CPU页，cache miss也仍走完整SHA。真实两-token门为`60.4563s→58.4415s`（-3.33%），
输出与所有权闭环保持；详见`FULLDEPTH43_GPU_VERIFIER_OWNERSHIP.md`。

当前层route确定后的42个MoE payload还可显式启用
`--vulkan-writeback-batch-verify-payloads`，由Rust做最多8路并行读取/SHA并在全部成功后原子发布。
正反相邻门中完整墙钟收益分别为17.54%和9.41%，输出仍为`[5,223]`；详见
`FULLDEPTH43_BATCH_PAYLOAD_VERIFICATION.md`。

逐层 manifest 也可用 `--vulkan-writeback-inline-manifest` 改为规范 JSON + SHA-256 的
内存直传。该路径真实消除了86/86个 `bridge_manifest.json`，并把Python IPC/响应校验
从`0.32039s`降至`0.25399s`；但完整两-token正向门回归4.45%，反向门又改善3.69%，
方向不一致。故它保留为默认关闭的研究开关，不晋级；完整负门见
`FULLDEPTH43_INLINE_MANIFEST_AB.md`及同目录JSON。

只常驻43层shared experts的实验入口为
`POLARIS_SHARED_GPU_PAYLOAD_CACHE_GIB=2`。第二token真实达到43/43 GPU hit和0 B shared重传，
但正反完整墙钟方向仍不一致，因此默认保持0；详见
`FULLDEPTH43_SHARED_GPU_CACHE_AB.md`。该负门说明下一阶段必须迁移到单一持久整token/多token
Rust/Vulkan执行块，不能继续靠小LRU逼近交互速度。
