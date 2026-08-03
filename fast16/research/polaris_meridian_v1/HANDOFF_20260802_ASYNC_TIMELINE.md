# 北极星 FullDepth43 异步流水交接（2026-08-02 20:45）

## 用户目标

- 最终质量追赶 Claude / GPT，不把普通小模型蒸馏当终点。
- 本机可用、长上下文、Kimi 前端及其他模型能力岛。
- 最终速度目标 20–50 token/s；当前任何局部门不得冒充该速度。
- 用户要求研发优先：不跑旧长榜，不重复 20 分钟测试；只保留秒级编译/结构门和一次必要真实门。

## 已有权威里程碑

固定 BOS 已真实闭合：

```text
BOS embedding
→ DeepSeek-V4 FullDepth43 L0..L42
→ final HC/RMSNorm
→ 32块真实 BF16 output head
→ GPU argmax token 5
→ 43层状态回读
→ host/device 两阶段原子提交
```

权威证据：

`scheduler/ssd_inference/evidence/fulldepth43_position0_whole_token_rx5700xt_20260802.json`

当前同步门为 `15.565704 s/token ≈ 0.0642 token/s`，只使用
`manifest_reference_replay`；不代表动态在线路由、多 token、聊天速度或 Claude/GPT 质量。

## 本次已完成

### 1. 43 层统一 paged timeline 桥

- 新增 `scheduler/ssd_inference/src/s14_position0_paged_layer_bridge.rs`。
- 43 层 stage→transfer→descriptor reconfigure→compute ticket 全部进入同一 timeline。
- hidden 全程留在 GPU A/B workspace；L42 后同一 timeline 继续交给 terminal/head。
- 专属门 3/3，旧 pager 回归 5/5。

### 2. 双 bank head 异步 stager

- `s14_position0_hybrid_upload.rs` 新增 head record→stage API。
- 不再由 uploader 自己 submit/wait；pending/bank/chunk 全部 fail-closed。
- 相关门 12/12。

### 3. 精确 device dirty write-set

- `s14_position0_state_writeback.rs` 新增：
  - `merged_layer_state_ranges()`
  - `merged_device_dirty_write_set(state)`
- 43 层 167 次状态写回 + terminal HC 合计 `373,760 B`，仅约 46 MiB arena 的 `0.81%`。
- 同步真实 example 已改为逐 range `mark_candidate_dirty`，不再登记整个 arena。
- 定向门 7/7；目标 example `cargo check` 通过。

### 4. 层权重异步 record/stage 基础件（主代理新增）

`s14_position0_hybrid_upload.rs` 已新增：

- `S14Position0LayerCopyReceipt`
- `record_next_layer_copies(...)`
- `stage_recorded_layer(...)`
- static staging 改为双 bank；总 staging 仍小于 1 GiB、不到逻辑权重十分之一。

`s14_position0_layer_backend.rs` 已新增：

- 43 份每层约 43 KiB immutable 元数据快照，避免异步录制后主机覆盖早期 descriptor 元数据；
- `record_embedding_prologue(...)`
- `record_next_layer_transfer(...)`
- `stage_recorded_layer(...)`
- `record_paged_layer_compute(...)`
- `validate_recorded_layer_binding(...)`
- `record_next_head_transfer(...)`
- `stage_recorded_head(...)`
- `record_layer_command_on(...)`，支持外部 command buffer 且不重置内部 pool。

最新验证：

```powershell
cd D:\project\大模型ssd化\scheduler
cargo check --offline -p ssd_inference --lib
```

结果通过；只有项目既有 warning。

`cargo test --offline -p ssd_inference s14_position0_hybrid_upload --lib` 为 8/8。

## 当前精确断点

基础 API 已编译，但还没有新的 production example 把它们全部接在一起。下一步必须直接修改或新增
`s14_position0_paged_43_layers_real.rs`：

1. 由 `WholeTokenDeviceState::begin_candidate` 得到 prologue command。
2. `backend.record_embedding_prologue` 后交给 `S14Position0PagedLayerTimeline::submit_prologue_compute_only`。
3. 为 43 层各分配一个 transfer/compute command。
4. 每层先：
   - `backend.record_next_layer_transfer`
   - `backend.record_paged_layer_compute`
5. 再交给 `S14Position0PagedLayerBridge::submit_next_layer`：
   - stage closure 调 `backend.stage_recorded_layer`
   - reconfigure closure只核验 plan/bank；descriptor 已按固定 A/B bank 录制。
6. `seal_layers()` 后把同一个 timeline 交给 `S14Position0TerminalChain`。
7. 32 个 head chunk 使用：
   - `backend.record_next_head_transfer`
   - `terminal.record_head_chunk`
   - `terminal.submit_recorded_head`
   - stage closure 调 `backend.stage_recorded_head`
8. terminal readback 后唯一 final candidate wait；然后按精确 dirty write-set 发布 device state。

先只跑编译门。编译通过后跑一次真实 BOS 门，核对：

- token 仍为 5；
- L42 hidden SHA 仍为 `6ad6ec7...b60`；
- 不再有 43 次 layer compute host wait 和 32 次 uploader fence wait；
- 如 timeline 仍报告 producer bank reuse wait，要如实计数，不能写成“全 token 仅一次 host API wait”。

## 后续主线（不要跑偏）

1. 在线 router 后取得实际 top-6，并按实际专家身份查页/上传/继续 MoE；移除固定 BOS manifest route replay。
2. 闭合真实连续 K=4 whole-token，利用一次权重扫描批量验证多 token。
3. v47 作为 token 草稿岛：只提升速度、保持 FullDepth verifier 分布。
4. v47/Kimi/工具等作为能力岛时，必须单独过能力门；不能把草稿接受误称为质量增强。
5. 最终 20–50 token/s 只能由端到端实测证明，局部 kernel 或 MoE replay 加速不可代替。

## FastCtx

已按官方方式安装 `fastctx v0.2.3`：

- binary：`C:/Users/Kangnaixi/.fastctx/bin/fastctx.exe`
- Co​dex MCP 配置与 AGENTS guidance 已写入。
- `fastctx status` 全部 PASS，MCP 握手返回 4 个工具。
- 新 Co​dex 任务会出现 `mcp__fastctx__read/grep/glob/replace`。
- fastshell 保持安全默认关闭。

新任务开场建议：

```text
继续北极星 FullDepth43，先用 fastctx 读取 PROJECT_STATE.md 和
fast16/research/polaris_meridian_v1/HANDOFF_20260802_ASYNC_TIMELINE.md，
直接接线 production paged whole-token，不跑旧长榜。
```

## 工作树安全

- 工作树非常脏，且大量 S14 文件仍未跟踪。
- 禁止 `git add .`、`git clean`、`git reset --hard`。
- 所有文本保持 UTF-8 无 BOM。

## 2026-08-02 续接结果：production paged whole-token 已闭合

- 已新增 `scheduler/ssd_inference/examples/s14_position0_paged_43_layers_real.rs`，把 prologue、
  43 层 layer transfer/compute、terminal prelude、32 个 head chunk、最终 readback 与 device/host
  两阶段提交接到同一个 paged timeline。
- terminal 新增 presealed timeline 提交入口和 terminal/state/HС 合并 readback 入口，避免 bridge
  已 seal 后重复 seal，也避免发布未更新的 candidate HC。
- 精确 example 编译门通过；按约定只跑一次真实 Release BOS 门，未跑旧榜与全量回归。
- 真实结果：token `5`、L42 hidden SHA-256
  `6ad6ec7bab0ac9a13e5065f2bb116e1df66ec9a850f05edccfcfc9e3a3279b60`、43/43 层、
  `commit_epoch=1`、`active_bank=1`，数值与同步权威门一致。
- runtime 为 `11.520325s`，相对同步 `15.565704s`缩短约`25.99%`。逐层 compute host wait为0、
  分层/head uploader fence wait为0；但static cold-start wait尚未单独计数，backend销毁仍含一次
  device-wide idle。双bank staging还有`71`次producer reuse wait，加最终wait后timeline host API
  wait为`72`，禁止写成“whole token只wait一次”。
- 权威证据：
  `scheduler/ssd_inference/evidence/fulldepth43_position0_paged_whole_token_rx5700xt_20260802.json`。
- 下一速度断点：消除/后台化71次producer staging reuse wait；质量/可用性断点仍是在线top-6、
  动态专家页和连续多token。当前仍是`manifest_reference_replay`，不得宣称已经可聊天。
- 独立example的成功路径已闭合；早期错误仍依赖进程退出回收资源。改造成常驻网页服务前必须加入
  统一`drain_all→rollback_external_candidate→destroy`清理守卫。真实门后已新增明确的external
  candidate armed语义，并为已收敛external timeline加入无device-wide idle的销毁快路；二者只过
  目标编译门，按“不重复真实门”要求尚未二次实跑。

## 2026-08-02 产品边界纠正：S14 模型本体优先

- 最终产品是 **Polaris S14 新架构模型本体**。FullDepth43 是 S14 的全深度主干/验证器，
  不是另一个独立产品；v47 以后只作为 S14 草稿岛，Kimi 等只作为能力岛。
- 旧 Python `fulldepth43_native_top6` executor 与固定 `0→5→223` 包装器已降级为历史诊断，
  停止继续投入。它们不能替代新的 Rust/Vulkan S14 任意输入、连续状态或产品验收。
- host 状态事务已泛化为 `TokenStateTxn`，position0/1/2可提交；position3首次ratio4压缩边界仍
  fail-closed。任意input token/position资产与production prologue embedding绑定已经完成。
- 在线top-6现可生成、proof校验并mmap 36个真实Range，再打包为既有ragged shader消费的动态
  routed arena；production position0 example已默认串起每层
  `static→probe→48B route→materialize→upload→continuation`，manifest replay降为显式诊断。
- position1两行window-KV+RoPE Vulkan核在ratio0/4两门上各32,768元素逐位等于CPU参考；device
  recipe也已参数化为读window row0、写row1、APE row1、ratio4 row5、ratio128 row1，position3
  压缩边界继续fail-closed。当前剩余顺序固定为：把这些position1部件接入production N=2入口 →
  新Rust/Vulkan连续双token唯一真实门。接口壳、输出变化和旧executor均不得写成S14完成。
- 只有上述模型门成立后，才接 Ollama `/api/chat` 与 OpenAI `/v1/chat/completions`；前端复用
  Open WebUI，不自研网页。继续禁止旧长榜、全量回归和重复真实门，只保留秒级结构门与一次必要真门。

## 2026-08-02 23:42：production 在线 N=2 已正式闭合

- 唯一必要的新 Rust/Vulkan 真门已经通过，不再是待办。position0输入0输出5，position1输入5
  输出223；两者均为43/43层、在线top-6、真实动态Range、零fallback，最终
  `output_tokens=[5,223]`、`commit_epoch=2`、`active_bank=0`。
- 热缓存local-only独立复核为position0 `12.528s`、position1 `10.808s`、总`25.233s`；stderr
  为空且无残留进程。证据日志为
  `D:/project/大模型ssd化/.tmp-polaris-tests/n2-local-pass-20260802-2342.stdout.log`。
- 显式fetch复核总`140.480s`，其中position1在线补齐L41/E115/E169的12页、25.5MiB；12/12
  完整落盘。旧的“父Rust消失、Python仍下载”由外层约720秒超时强杀造成，fetch桥已补独立日志、
  7200秒内部超时、非零日志尾传播与kill+wait回收。
- 当前产品断点改为position2/3：先让position2消费已提交row0/1，再闭合position3首次ratio4压缩
  边界，随后至少跑8--16 token短生成。协议/API可与该实现并行，但在连续生成长度成立前不得把
  Open WebUI页面称为模型已可用。继续禁止旧长榜、旧executor和重复N=2真门。

## 2026-08-03 续接结果：production 在线 N=4 已正式闭合

- 首次 N=4 Rust/Vulkan production 主进程明确 exit code `0`，墙钟约 `1897.1s`。该轮包含首次
  精确缺页获取；成功后临时 fetch manifest/log 已按实现清理。为补齐被工具截断的中段而准备的
  重复复核已在编译阶段停止，没有启动第二个模型实例，也没有重复跑模型真门。
- position0--3 各 `43/43` 层；直接恢复到的输出为 position0=`5`、position1=`223`、
  position2=`939`。position3 单项 token 行被工具截断，未伪造；生产硬门要求并已通过的前缀为
  `[5,223]`。
- exit `0` 前的必经硬判定已闭合：`172` 次在线 top-6、`6192` 个实际 Range、
  `cpu_compute_fallbacks=0`、`commit_epoch=4`、host/device `position=4`、epoch/active bank一致、
  position3 ratio4 boundary committed、position4 fail-closed。
- **当前精确断点已转为 position4+ 通用连续运行时**。position3 专用 attention 只利用 compressed
  cache 基数为1的恒等 indexer；position4+ 必须运行真实 compressed indexer top-k，不能复用该
  特例。`s14_runner::whole_token`、`TokenStateTxn`和state writeback仍显式拒绝 position4+。
- 下一顺序固定为：通用 compressed-indexer attention与状态事务 → 秒级编译/结构门 → 唯一一次
  8--16 token Rust/Vulkan短生成 → 持久 S14 `ChatEngine` → 既有 `/api/chat` 与
  `/v1/chat/completions` → Open WebUI。继续不跑旧长榜、旧N=2、旧Python executor或自研网页。

## 2026-08-03 续接结果：position4+ 结构与 runtime facade 已闭合

- position4--126 的 host事务、device candidate、paged timeline、layer backend outer gate 与state
  writeback已泛化；position7第二次ratio4 rollover有结构断言。position127 ratio128边界继续拒绝。
- 新增持久 `ssd_inference::s14_runtime::{S14Runtime,S14Session,S14StepOutput}`，生产昂贵资源跨token
  常驻，session独占连续状态；错误保持 drain→owner destroy→rollback。默认Range为local-only。
- 新增正式 `WholeTokenCandidate::commit_with_next_input` 支持官方prompt forced-prefill；预测仍写
  ledger，runtime不再在commit后直接改state。runner定向测试5/5、position3 softmax定向测试2/2、
  `cargo check --offline -p ssd_inference --lib`全部通过；未启动模型、未跑旧榜。
- 当前唯一真实数值阻塞仍是position4 compressed indexer：必须完整执行
  `wq_b→RoPE→Hadamard/FP4→score/top-k→compressed attention`。backend仍诚实fail-closed；
  不得把结构闸门或runtime facade写成8--16 token、聊天接口或S14模型完成。

## 2026-08-03 纠正后的当前断点：S14 K=4 block-major 模型本体

- 单token production 已推进到position2051分页边界，N=8权威输出为
  `[5,223,939,21,695,553,1266,16179]`；344次在线top-6、12,384个实际physical Range、
  `cpu_compute_fallbacks=0`、`commit_epoch=8`均成立。热态约`14.638320s/token`，仍不是网页聊天速度。
- 当前产品路径不再继续包装旧Python executor、固定BOS replay或独立网页。FullDepth43只作为
  Polaris S14的全深度主干/验证器；v47后续只作草稿岛，Kimi/MiMo等只作能力岛。网页/API继续关闭，
  只有新Rust/Vulkan S14模型门通过后才复用Open WebUI。
- 分支`codex/polaris-s14-k4-runtime`的基线提交为`88b6621`。K=4/8已有block-major 43层调度、
  图内在线top-6、实际Range union/grouped MoE、最长一致前缀与device checkpoint合同；禁止退化成
  K次whole-token forward，强回执固定`serial_token_forward_calls=0`。
- production terminal已收口为唯一post-seal路径：idle时安装真实terminal owner与host candidate
  finalizer，43层seal后精确发布一次同源source；owner强持有K-row final HC/norm、32块真实head、
  K份完整checkpoint与producer timeline。GPU batched head完成前不能预造prediction。
- **当前唯一必要数值缺口不是下载页，而是K=4跨ratio4边界**。base position 1的四个lane对应
  position `[1,2,3,4]`；position3必须执行remainder→main/indexer finalize/writeback→first
  compressed-block attention→rollover，position4必须消费正式compressed indexer/sparse attention。
  现有contiguous-window causal shader没有这条路径，不能用“输出了四行”冒充真实S14 K=4。
- 与该边界同时收口的43层provider必须在同一block-major图内拥有真实static权重、K行RoPE/current
  KV/committed KV，并生成每个lane的完整window KV、HC、ratio4/128 compressor与indexer prefix
  checkpoint；terminal final hidden、checkpoint与producer timeline必须同源。
- 下一顺序固定为：补K=4 ratio4 boundary-aware attention/state producer → 只跑library离线编译和每个
  新模块一个秒级定向门 → 唯一一次Rust/Vulkan K=4 whole-token真门。真门必须同时证明43/43层、
  在线top-6、实际Range proof/SHA/mmap、grouped MoE、ratio4数值路径、batched terminal/head、K份
  checkpoint、最长一致前缀、零CPU fallback与零串行token forward；通过前不得宣称S14模型已完成。
- 存储清理只处理高置信、无引用、可重建副本。最新约`1.188GiB`旧v17 build、已否决v18 `.npy`
  和临时v38快照已移入Windows回收站；S14、v47/v38、v17 runtime-v3、v18 runtime-v2、供体、
  Range/head/checkpoint及Rust/shader缓存全部保留。回收站未清空前这些空间仍可恢复且不会物理释放。

## 2026-08-03 19:28：K=4 连续两块成立，hot 性能门未过

- 同输入 production 真门严格串行 A/B；两轮都是43层两块、block1 position 1→5、
  block2 base 5、`committed=true`，真实 selected token 均为 `17351`。这是新
  Rust/Vulkan S14 continuation，不是旧executor、fixture或固定checkpoint replay。
- A：`139.703s`，66 miss，下载`140.25MiB`，fetch `67.302s`，union/SHA/mmap
  `8.542s`，其他计算/编排/启动/checkpoint残差`63.859s`。cold `<5min` 成立。
- B：`100.334s`，仍有24 miss与`51.00MiB`下载，fetch `44.490s`，union
  `6.518s`，残差`49.326s`。B不能称hot，hot `<90s` 失败`10.334s`；禁止通过
  重复第三次刷热掩盖。两轮union都是0 mmap hit，分别对`17.457GiB`/
  `17.481GiB` payload重做SHA，这是当前最硬的hot路径浪费。
- durable checkpoint新增的`sha256_file()`曾因1MiB栈数组导致Windows主线程
  `0xC00000FD`；已改为堆缓冲，根修后 B 轮已在普通主线程通过。
- 下一唯一顺序：跨 block 常驻 verified mapped store/SHA receipt → 用已完成的
  pack-index 规划降低53,914小文件探测 → 用已知task/K-block route做有界预取 →
  再做一次真正`cache_misses=0` A/B。不跑质量榜、旧长榜或第三次刷热。D盘约余
  `70.45GiB`，必须始终保留至少`20GiB`。
