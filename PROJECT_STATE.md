# 北极星（Polaris）项目现状

更新日期：2026-08-01

跨对话/跨工程接手续读：`fast16/HANDOFF_2026-07-31.md`。

## 正式命名与兼容层

- 2026-08-01 起，模型与项目的正式对外名称为**北极星（Polaris）**。
- 历史版本号继续沿用；下一正式运行别名使用`Polaris-v*`。现有`ColorLM-v*`名称只表示历史实验
  证据，不能因为品牌更名而改写旧结果。
- 为避免一次性破坏运行时，`COLORLM_*`环境变量、`fast16/`研究路径和已有脚本文件名暂时作为
  底层兼容层保留。兼容命名不代表仍以ColorLM作为对外品牌。

## 2026-08-01 目标与算力约束再确认

- 终点保持为追赶 Claude/GPT 的通用本地单模型，不把前端、代码或工具能力岛误当成最终模型。
- 用户允许小型蒸馏训练，并可使用免费云 GPU / NVIDIA 免费额度；任何密钥只通过本机环境变量或
  平台 Secret 注入，禁止写入提示、源码、日志、报告和 Git。
- 本地硬件仍按 RX 5700 XT 8GiB 显存、32GiB 内存设计。正式 Claude Code 入口先保证 32K/64K
  稳定可用；200K+ 作为隔离实验档，只有通过首段/中段/尾段检索、工具协议与长程速度门后才晋级。
- v47 的 Parallel Genome Head 与多 token 草稿头属于可复用的 Neural Bus 能力/加速部件，不改变
  正式最佳仍为 v38 的事实，也不作为 Claude/GPT 综合能力已经接近的证据。

## 长期目标

构建可在8GiB显存与32GiB内存上运行的本地单模型。长期目标不是代码/工具专项模型，而是追赶
Claude/GPT的通用智能体能力，覆盖推理、知识、长上下文、编程、工具、规划、电脑操作与自然交流；
编程和工具只是当前最便宜、最可自动验收的突破口。当前首个多供体交付保持在15GiB以内；后续若
能力增益明确，磁盘权重预算可扩到50–70GiB。Claude Opus/Codex级真实任务完成率是长期验收目标，
不用模型自述、单维NLL或单个演示代替综合能力证据。多维验收矩阵见
`fast16/research/v19_dual_head/colormlm_frontier_capability_matrix.json`。

## 当前可用主模型

- 当前没有常驻模型服务；所有v36/v38验证服务均已释放。需要体验时只启动一个入口，避免研究端口
  与`8105`并存造成32GiB内存不足。
- `ColorLM-v38-Qwen36-Shared-Sequence-Policy`是当前最佳研究/体验候选：12.679GiB v36核心加
  约0.125MiB显式工具策略头；启动入口为`fast16/run-colormlm-v38-qwen36-sequence-policy.bat`，
  端口`8138`。冻结工具状态题`7/20→11/20`、4净胜0回归，无tools物理旁路逐字段等于v36。
- v39已用v36原生terminal hidden重采192组logits/hidden并复用冻结v29算法；validation/test平均
  NLL虽为`-0.572/-0.0816`，但test候选胜率仅`57.14%`且最坏留出任务回归`+0.1606`，未过
  `60%/+0.03`硬门。v39已停止，不构建运行包；这不是可体验版本，v38地位不变。
- v40范数匹配头在全新12题上与v36逐题相同`5/12`；v41执行状态门test仅`2/5`；v42把旧策略
  头接到20.20GiB v33后虽将旧工具题`6/20→9/20`，但八维门仅`8/16`且512 batch显存溢出。
  三条线均停止，当前最佳仍是v38。
- v43用120条成对工具状态轨迹采集720个v36 terminal hidden/base-logit状态，拟合了
  74,180字节的PCA rank-8、显式no-op九类策略头。F32离线NLL门通过，但真实生成门中
  v36与v43均为`12/24`，`0`修复、`0`回归；只有6/24条规范化输出发生变化，没有一条变成正确。
  按预注册合同v43已停止，不扫描rank/强度，v38仍是当前最好可用版。
- `ColorLM-v36-Qwen36-Global-Shared-Backbone`是当前最快新核心：40层精确Qwen3.6 router/shared
  expert配合v6 routed bank，12.679GiB；入口为`fast16/run-colormlm-v36-qwen36-shared-backbone.bat`，
  端口`8136`。冻结16题与v17逐题同为`10/16`，速度快`26.80%`。
- `ColorLM-v29-Sequence-Policy`已晋级为显式工具模态增量版：无tools请求物理走v17图；显式非空
  `tools`请求启用16行、约0.125MiB的隐藏状态条件序列策略头。正式入口为
  `fast16/run-colormlm-v29-sequence-policy.bat`；Claude Code隔离入口为
  `fast16/claude-v29-sequence-policy.bat`，端口均为`8105`；它现在是稳定回滚版，不删除。
- `ColorLM-v17-Coder-Neural-Island`：稳定旧版。它在12.789GiB v6主干的第35层接入
  3.514GiB连续Coder神经岛，有效权重合计约16.303GiB。
- 神经岛在供体坐标内连续执行Qwen3-Coder-Next L44–L47：三层Gated DeltaNet后接一层
  Full Attention，再经出口坐标桥以`alpha=0.02`残差写回主干。
- v17.2已把三层循环状态和L47 KV迁入`llama_context/seq`原生内存生命周期，并默认启用
  四层各32槽的精确GPU热专家缓存；完整冷专家仍保留在`Vulkan_Host`。
- v17正式入口为`fast16/run-colormlm-v17-coder-island.bat`，服务端口为`8105`、上下文16384；
  运行包为`fast16/research/v17_coder_island/runtime-v3/island.json`。
- `ColorLM-v18-Deep-Activation-Island`降为隔离研究候选，不得替换v17。其真实激活桥在12折
  整提示LOTO上获得`12/12`正余弦提升、合并提升`+0.3657`，但中位范数尺度的NRMSE比率
  为`0.9559`且5个唯一切分`0/5`通过；训练集最小二乘尺度虽通过LOTO，也只有`2/5`切分
  通过，低于`4/5`稳定门。现有`runtime-v1`仅保留复现，不作为正式运行包。
- `ColorLM-v18.1-Nullspace-Anchored-Island`已把零奇异值自由度替换为距离旧嵌入桥最近的
  正交完成。同一772状态严格门从`0/5`或`2/5`提升为`5/5`，LOTO NRMSE比率`0.9192`、
  最差切分`0.9475`，12/12 prompt方向仍为正；运行矩阵正交与往返RMSE均约`5.95e-8`。
  新隔离包为`runtime-v2`，静态启动契约已通过，但在独立能力短测前仍不替换v17正式版。
- v18.1当前证明真实深层坐标可稳定运输；尚未证明整体编程能力超过v6、普通55B或前沿闭源
  模型。下一步只做独立短能力题，能力不回归后才考虑扩为L40-L47八层岛。
- v18.1运行级短门中，`read_file`首次正确给出必填`path`且以`tool_calls`停止；工具结果回合没有
  再次误调工具，但96-token预算内因冗长被截断。独立Rust指令遵循A/B中，v18.1与v17都生成
  Markdown围栏和多余`VecDeque`并在128 token截断，属于同类失败。结论是未见质量回归、也未见
  能力增益，v18.1保持研究候选；该轮结束时曾恢复v17，当前没有常驻服务。
- `ColorLM-v6-Q3Router-Fused-A1.gguf`仍是12.79GiB安全核心与无侧链回滚基线。

## Coder-Next研究线

- `v7`：Qwen3-Coder-Next单专家直接嫁接。工程闭环成功，但坐标审计表明路由匹配不优于
  随机打乱维度，因此仅保留为失败对照。
- 共享token正交运输：未见token平均余弦从0.0013升至0.4692，token身份恢复率98.54%。
- `v8`：运输后的Coder-Next专家471嫁接到ColorLM第39层槽位201，12.94GiB，Vulkan
  生成18.5 token/s。
- 最小决策表：v6代码8/8、工具4/4；v8代码7/8、工具4/4。v8出现真实代码回归，未晋级。

## Kimi K3研究线

- 官方仓库：`moonshotai/Kimi-K3`；公开权重总量约1.454TiB，未下载整模。
- 已用HTTP Range提取真实路由、桥接张量和指定MXFP4专家，并拟合7168→2048半正交运输。
- v9-K3-rc1首先在L12/L28接入两颗完整K3 latent宏胶囊；v10扩为四颗：
  L12使用L28/E41、E780，L28使用L65/E539、E752。
- v10运行时支持每站2–8颗候选，以当前隐藏状态和`router.f32`做cosine/temperature softmax，
  并保留连续no-op通道；不读取文本关键词。
- 四颗K3运行权重共381,743,104字节，均由Vulkan0执行。当前所有候选先计算再软混合，
  因而完成的是能力融合机制，不是lazy-expert加速。
- 这些本地K3资产是少量路由、桥和FFN宏胶囊，不是K3完整连续网络或独立“前端编程模块”。
  网页生成能力分布在完整训练与多层计算中；在没有K3末端隐藏态/输出头反事实证据前，任何
  本地前端题改善都不能归因为“吃到K3前端能力”。

## Demand-Routed Synaptic Paging

- ColorLM v6专家权重共10.70GiB，每个专家每层平均1.070MiB。
- 每层16个热专家的40层GPU槽池只需0.669GiB；32个热专家需1.338GiB。
- 95%缓存命中时理论SSD读取为17.125MiB/token，按3.5GiB/s约4.78ms；99%命中约0.96ms。
- 旧`qwen3moe`槽池的miss曾错误回落到slot 0，不能用于主模型；当前已另行实现并验证
  `qwen35moe`精确miss路径，结果见下方“qwen35moe精确专家槽池”。
- 设计文档：`fast16/research/DEMAND_ROUTED_SYNAPTIC_PAGING.md`。

## 当前判定

工程层面已从v12的并联小胶囊推进到v18的真实深层激活桥连续神经岛：构建期GLM/Qwen3.6
合金与运行时Coder岛在同一个llama-server、同一个token图内执行，供体连续层拥有上下文级
原生状态和精确GPU热专家缓存。v18首次用完整80B.A3B供体L43真实输出标定入口，不再用词嵌入
几何代替深层状态。能力层面仍未建立超过v6或前沿闭源模型的充分证据；启动、状态正确、激活
对齐和token/s只证明执行契约与候选价值，不把“权重更多、能加载、输出变化”写成“更聪明”。

### 2026-08-01 基座谱系短门

- 新增8维、16题冻结短门，覆盖推理、知识、长上下文、编程、工具、规划、电脑操作和自然交流；
  判分与响应分离，禁止看完输出后修改答案契约。
- 同一16题中，原生Qwen3.5为`9/16`，v17为`10/16`。v17只修复
  `long-context-two-markers`，相对原生基座严格零回归；这支持继续保留v17，但样本规模不足以
  外推为普通55B或Claude/GPT级能力。
- 8题冒烟中v17、Qwen3.6-35B-A3B、GLM-4.7-Flash均为`3/8`。Qwen3.6改善代码但回归长上下文，
  首token约慢3倍；GLM改善标准工具调用但回归知识格式。因此两者都不能整模替换v17。
- 工具题的24-token上限会使Qwen系正确参数JSON偶尔少最后一个`}`；它同时反映协议效率差异和
  tokenizer不公平性，后续工具门应按完整决策跨度预算，而不能把截断误记成模型完全不会调用工具。
- 产物位于`fast16/research/parallel_b/runs/`；采集器为
  `fast16/research/parallel_b/run_multicap_base_gate.py`。

### 2026-08-01 v21 GLM Agent Island深层桥否决

- GLM-4.7 GGUF为`deepseek2`架构、隐藏宽2048。L43–L46是四层完整MLA+64专家MoE连续岛，
  原始量化约`1.74GiB`，无需下载新权重；它仍是有互补工具能力证据的有效供体候选。
- 复用ColorLM L35真实激活并采集GLM L42输出，12条提示得到766个配对状态，平均token覆盖
  `94.71%`。共享token正交桥把留出余弦中位数从约`-0.0006`提高到`0.2623`。
- 第一版nullspace-anchored正交激活桥把留出余弦中位数从`0.0126`提高到`0.5594`，3/3留出
  提示均改善；但NRMSE比率`0.9827`且原始尺度`4.40`超出契约，因此拒绝。
- 第二版锚定低秩岭桥由内层留出选择正则`1.0`，留出余弦中位数进一步达到`0.7329`且3/3提示
  改善；NRMSE比率仍为`0.9727`，未达到预先固定的`<=0.95`门，因此仍拒绝。
- 正式决策：不实现GLM岛C++运行图，不宣称v21成功，也不在同一766状态上继续扫描。若重开，
  必须新增独立提示和更多配对状态，再检验带偏置或协方差运输；现有结果作为“跨模型深层坐标可
  部分运输，但幅度尚未稳定”的研究证据封存。
- 产物位于`fast16/research/v21_glm_agent_island/`。当前GLM与v17服务均已主动停止。

### 2026-08-01 v22 GLM独立外部门关闭

- 在查看结果前冻结16条全新Agent/工具提示；ColorLM L35与GLM L42得到1227个配对状态，
  16/16提示成功，平均覆盖率`99.25%`。
- 冻结v21岭桥不使用新数据拟合或缩放，外部余弦中位数`0.0074 -> 0.7405`，16/16提示均提升；
  NRMSE比率仍为`0.9671`，未达到`<=0.95`门。方向运输是真实泛化，幅度契约仍失败。
- 唯一预声明的仿射候选只用旧766状态拟合/选参；外部余弦中位数为`0.4216`，NRMSE比率
  `0.9809`，比冻结岭桥更差。按契约不增加第三候选、不扫描外部集、不实现GLM C++岛。
- 完整报告与冻结哈希位于`fast16/research/v22_glm_agent_bridge/`。主干与GLM采集服务均已释放。

### 2026-08-01 v23 Fara电脑操作供体准备完成

- 下一位完整可运行供体固定为Fara1.5-27B Q5_K_M及F16视觉投影。27B整模只作为离线教师，
  不进入正式常驻运行图；目标是蒸馏截图理解、浏览器首动作与关键点停顿能力。
- 当前D盘余`69.43 GiB`，没有`llama-server`进程。两个权重文件尚未到位；固定revision、大小、
  SHA-256、镜像链接与落盘路径见`fast16/research/v23_fara_cua_donor/DOWNLOAD.md`。
- `llama.cpp/build-v17-perf`已静态确认支持dense Qwen3.5、Qwen3-VL `mmproj`、OpenAI
  `image_url`、Q8 KV、自动显存放置及禁用reasoning；新增8125隔离启动器、大小/SHA验收器和
  按GGUF原始token bytes逐ID审计器。所有Python均通过语法检查，文本为UTF-8无BOM。
- 已冻结4题电脑操作首动作门：两个视觉点击、缺电话号码停顿、缺航班目的地停顿。必须至少
  `3/4`且两个`ask_user_question`题全过；契约SHA-256为
  `bab9195e483b63af391d916d796c847e04873f894ee2d739a206da176a7fd924`，运行器会在请求前硬校验
  契约与四张截图哈希，禁止事后换题或改图刷分。
- 若教师短门通过，最终高速结构固定为“Fara视觉塔+merger（仅图像prefill）→5120→2048视觉桥
  →v17主干→低秩CUA动作残差头”；无图像时不建立新节点，纯文本目标零额外token延迟。
  若短门失败则立即关闭该供体，不调提示刷分；若通过，同一次教师加载只采8–16条正确首动作
  轨迹、top-32 logits和末层hidden，再实现视觉桥与小于32MiB的动作头。

### 2026-08-01 v24 n-gram推测解码否决与默认纠偏

- 审计发现正式启动器曾无条件启用`ngram-mod(match=16,min=4,max=16)`，但项目没有为连续神经岛
  建立过对应的状态checkpoint/rollback等价证据。现已把推测类型变成显式参数，正式默认改为
  `none`，n-gram只保留隔离研究入口。
- 同一v17、同一seed、四题相邻短门中，none为`14.2444 token/s`，ngram-mod为
  `14.1415 token/s`（`-0.72%`）；仅1/4消息精确一致，Rust、中文和工具题均发生输出或停止差异，
  工具题速度下降`19.52%`。因此该路径同时未过质量等价门和速度门，禁止默认启用。
- 本次纠偏不增加模型能力，但移除了一个会让同seed行为漂移、且没有总体加速的运行时变量。
  完整冻结契约、逐题哈希和报告位于`fast16/research/v24_speed_quality_bus/`。
- 下一条无损速度线为真实专家路由序列采集与离线缓存策略模拟；当前v17四层32槽在32 token观测
  中仅`24.22%`命中，并上传约`1.60 GiB`，优先降低该上传量而不是继续扫描推测参数。

### 2026-08-01 v24真实路由与LFU缓存负对照

- 新增默认关闭的四层top-10专家路由二进制dump和离线缓存模拟器。512条真实记录中，32槽LRU
  各层命中率为`32.19/23.20/20.70/25.62%`；衰减LFU提高到
  `33.36/25.08/25.94/32.81%`，四层均正；Belady离线上界为`44.38–52.66%`。
- 已实现可回退`lfu97`精确淘汰策略。真实192步A/B中，miss从`5,761`降至`5,510`，少上传
  `423.56 MiB`；文本与工具函数/参数保持等价，但总速度从`14.7205`变为`14.6881 token/s`
  （`-0.22%`），未形成速度收益，因此正式默认继续LRU。
- 该结果否决继续扫描缓存策略：上传量只是瓶颈之一，Vulkan提交/同步和四层MoE计算仍占主导。
  下一主线转为把3.514GiB连续岛蒸馏成GPU常驻低秩微岛，直接消除分页和四层专家计算。

### 2026-08-01 v25无状态低秩微岛未准入

- 新增默认关闭的教师成对dump，在同一运行图导出`ffn_residual -> raw island_delta`；现有文件包含
  `128`组成对图记录，合计`314` token，约`5`段序列。该采集不改变默认运行路径。
- 无状态低秩线性探针的留出结果：rank 32/64/128/256中位余弦分别为
  `0.2292/0.2363/0.1810/0.1002`，relative RMSE为
  `0.9826/0.9987/1.0225/1.1472`。rank 32仅微弱优于no-op，高rank已经过拟合，所有NPZ
  仅为研究产物，禁止当作正式模型或能力提升。
- 复核发现旧探针只是按图记录的前80%/后20%切分，并非严格按完整请求留出；该结果只能否决
  “立即接入无状态线性头”，不能证明有状态蒸馏不可行。
- v25下一候选固定为低于4MiB的有状态微岛：
  `q_t=U(h_t-mean); z_t=rho*z_(t-1)+q_t; delta=[q_t,z_t]V+y_mean`。必须一次性采集至少8条
  独立短请求，按请求划分训练/验证/测试；只有留出`relative RMSE<=0.90`、中位余弦`>=0.40`、
  相对同rank无状态头改善`>=0.02`且每条测试请求均优于no-op，才允许实现C++运行图。
- 已按上述契约补采8条独立请求、1034个实际请求token；另有一段2-token服务内部序列被显式
  丢弃。简单状态模型测试为中位余弦`0.2482`、relative RMSE`0.9576`，弱于匹配无状态头的
  `0.2666/0.9426`；0.84MiB门控非线性模型也仅为`0.2718/0.9405`，整岛微层未准入。
- 分站教师dump显示L44/L45/L46/L47留出结果分别为：
  `0.3605/0.9252`、`0.3951/0.8624`、`0.4442/0.8568`、`0.8434/0.6902`
  （中位余弦/relative RMSE）。只有L47通过分站门，且四层均选择`rho=0`。
- 已实现默认关闭的F16线性微层运行包和C++图路径。完整L47替换rank 32包为270,336字节；
  prefill保留完整L47、仅decode替换的rank 8包为73,728字节。两者冻结4题均保持实际内容、
  工具名和参数，方向性吞吐约`+7%`，但冻结16题都从v17的`10/16`降为`9/16`，共同回归
  `reasoning-constraint-order`的停止边界，因此均未晋级，正式模型仍是v17。
- 下一条唯一压缩主线改为保留L47真实Attention/KV，只蒸馏L47的`MoE + shared expert`支路；
  完整数字和运行契约见`fast16/research/v25_stateful_micro_island/V25_MICRO_ISLAND_RESULT.md`。

### 2026-08-01 v26 L47微MoE否决

- 新增默认关闭的L47 `post-attention norm -> MoE + shared expert residual`教师dump；同一8条独立
  请求得到1034个有效token，严格按请求4/2/2训练、验证、测试。
- rank-64微支路通过预设离线门：测试集中位余弦`0.6618`、relative RMSE`0.7876`，两条测试
  请求均优于no-op。运行包只有532,480字节，真实Vulkan加载、SHA校验与生成均通过。
- 冻结4题中三项普通输出逐字等价，工具名`read_file`和参数`{"path":"src/main.rs"}`等价；但完整
  冻结16题从v17的`10/16`降为`9/16`，没有新增通过项，`reasoning-constraint-order`因正确JSON后
  继续生成而回归。
- 按预先固定的零净回归硬门，v26立即否决，不建立正式启动器，不追加速度A/B。v25/v26三种
  L47近似都在同一停止边界回归，因此停止整条L47压缩线，不再扫描rank、rho、精度或数据切分。
- 正式模型仍为v17。默认关闭的教师dump、微层加载器、权重校验和运行图作为通用基础设施保留。
  下一阶段回到能力增益：先证明独立教师在当前缺失维度上的互补性，再蒸馏能力头或接第二供体。
  完整报告见`fast16/research/v26_l47_moe_micro/V26_L47_MOE_RESULT.md`。

### 2026-08-01 v27工具协议总线启动

- GLM-4.7补跑完整冻结16题为`9/16`。相对v17，它把两道工具题从`0/2`全部修为`2/2`，但同时
  回归知识、代码和规划各一题；因此GLM不能整体替换v17，也不能全局注入，只允许作为工具协议供体。
- v27结构固定为显式工具模态下的末端稀疏协议头：从v17 terminal hidden产生最多1024个候选token
  的低秩logit修正；无`tools`请求时要求物理旁路，不使用文本关键词路由。
- 新增12题冻结工具协议门，按完整任务6/3/3训练、验证、测试；契约SHA-256为
  `a7b4d77e1c2550fa7958586f3050d7bb4ca48dbcfce56b942b7f9997b24587ac`。GLM教师须至少`9/12`
  且领先v17至少3题才允许拟合能力头。
- 已新增默认关闭的末层采集通道。未启用v19输出头时，CNOB kind 4保存F32 base hidden、kind 1
  保存F32 base logits；v19启用输出头时原三张量ABI不变。Vulkan Release编译和8119真实启动通过，
  两条记录的`[2048,1]` hidden与`[248320,1]` logits均通过严格字节检查，服务已停止。
- 设计、冻结门与采集检查器位于`fast16/research/v27_tool_protocol_bus/`。下一断点是教师准入，
  未过即停止，不构建运行头；通过后只采正确工具轨迹并做留出拟合。
- 教师准入最终结果为GLM `12/12`、v17 `11/12`。GLM只修复留出题
  `tool-ask-clarification`，没有达到预先冻结的“领先v17至少3题”，因此v27已停止，不拟合协议头。
  该结果同时纠正了旧24-token工具题的截断偏差：v17在足够预算下已能稳定完成11种未见schema。
  下一能力瓶颈不是一次工具JSON，而是工具结果回合后的停止、继续和跨步骤状态规划。

### 2026-08-01 v28执行状态头启动

- v28目标从一次工具JSON转向工具结果回合后的模型内执行控制：从terminal hidden和上下文中的真实
  tool-result表示，预测`{no-op, continue-tool, ask-user, finish}`状态，再只修正控制token与候选工具
  token；无工具schema时要求物理旁路，不使用主机关键词分类。
- 已在候选产生前冻结8条状态门，覆盖成功后停止、文件缺失恢复、测试失败定位、验证/激活/健康检查
  顺序、缺信息询问及禁止重复写入；按完整场景4/2/2切分，SHA-256为
  `17e62d828a110a24c1dd6c1fb38b88e7e013fe21b50e57082268f9dce041a35a`。
- 新增OpenAI多轮工具历史采集器和纯离线判分器，均通过Python语法及JSON结构检查。下一断点只跑
  一次v17基线；若已`8/8`则停止v28，否则才允许用训练4题拟合小头，验证/测试保持完全留出。
- v17严格基线为`4/8`（训练`1/4`、验证`2/2`、测试`1/2`）；GLM-4.7同为`4/8`且错误分布
  不同，没有形成可复制的整体教师。v17会在正确JSON外包Markdown、在恢复工具调用前附加文本，
  并把缺信息询问输出为普通文本；验证→激活→健康检查顺序本身正确。
- v28不使用GLM蒸馏。下一步只采首个决策位置的terminal hidden，用固定二元ridge探针检验
  `continue-tool / finish`可分性；离线不可分即停止，不写运行时状态头。GLM与v17服务均已释放。
- 首决策CNOB采集已完成：8题各一份`[2048,1]` terminal hidden与`[248320,1]` base logits，
  capture SHA-256为`8a9ac1ab21dbe6e4222a2f6bb811f423b7e766de61590214f45a6ab3a26de5bd`。
  固定centered dual ridge在train/validation/test均为`100%`，说明末层状态确实可分。
- 但预注册双控制token修正没有通过运行前离线门：固定beta `2.3998654010`时train/validation为
  `100%`，test仅`50%`；`state-missing-target-ask`仍被原生普通文本首token压过。按契约停止
  双token头，不扫描强度、sharpness、token或阈值，不实现C++路径，正式模型保持v17。
- v28留下的正证据是“状态信号存在”，负证据是“常量token bias不能可靠兑现”。若重开执行控制，
  必须换为新的动作空间结构并建立全新留出场景，旧测试集只能作开发诊断，不能再次充当晋级门。

### 2026-08-01 v30动态词汇策略头：机制通过、能力未晋级

- v30把v29固定16行输出改成动态低秩双线性残差：从当前tools请求及生成前缀选最多256个唯一token，
  对terminal hidden与基座token embedding分别做L2归一化和rank-32投影，再逐候选点积并原位修正logit。
  运行权重仅`524416`字节；无tools仍物理旁路，API token历史、动态I32/I64索引与Vulkan图均已接通。
- 旧v29数据回顾探针因一个不可由输入唯一推出的目标JSON产生单任务回归，按旧契约记为失败。随后在
  看到结果前冻结12个目标可客观推出的新任务，用完全冻结的同一权重做178-token独立门：平均NLL
  `-0.67284`，12/12任务改善、0任务回归、动态目标覆盖`93.26%`，因此只获得运行时原型资格。
- 真实生成A/B中v17与v30均为`12/12`；去除随机tool-call ID后12题结果完全相同。平均decode速度
  `22.629 -> 22.214 token/s`（`-1.83%`）。因此动态机制真实可运行且能降低NLL，但没有产生可观察
  能力增益，不替换v29、不宣称工具智能提升。完整证据见
  `fast16/research/v30_dynamic_lexical_policy/runtime_gate_report.json`。
- 下一轮必须新建v17不能全过、但目标仍能由输入唯一推出的更难任务族；本12题只作为机制回归门，
  不得复用来调参或晋级。正式用户入口继续保持v29。

### 2026-08-01 v31–v35 Qwen3.6全深度MoE配对

- 字节审计确认ColorLM与Qwen3.6的248320 token、merges、token type、EOS/PAD及预分词器身份完全
  一致；chat template和供体额外BOS声明不同，但不影响token ID坐标。v6在L39闭合MoE内只改变
  router，其余norm/routed/shared expert仍与v5逐字节相同；v6 router与Qwen3.6余弦`0.999122`，
  但不是精确供体路由。因此“Qwen3.6 router驱动旧专家”的全深度错配假设成立。
- v31只恢复L39闭合MoE，独立短门`5/8→5/8`且速度`-25.24%`；v32恢复L36–L39连续终端段及
  final norm/output head，`4/8→4/8`且速度`-23.44%`。两者多数生成逐字一致，证明能力差异不
  集中在单层或末四层；失败GGUF已回收，脚本和报告保留。
- v33把40层精确router、256路routed expert、shared expert及MoE入口norm全部恢复为Qwen3.6，
  同时保留ColorLM token embedding、Attention/Gated DeltaNet状态主干、final norm和输出头。
  单文件`20.200GiB`、360个真实供体张量；全新八维短门从`5/8`升到`6/8`，1项净胜、0回归。
  长上下文由两个字段都错变为一个字段正确，规划由超预算变为合法但非最优，严格术语/JSON题通过；
  96-token速度`27.616→25.686 token/s`（`-6.99%`），略过`-5%`速度门，因此只保留研究候选。
- v34把全部40层routed expert量化为IQ3_S、shared expert量化为F16，体积`14.976GiB`且速度仅
  `-3.96%`，但独立短门`6/8→5/8`并出现推理回归；v35只压缩CPU前29层、保留后11层高精度，
  体积`16.485GiB`，仍为`4/8→4/8`且速度`-6.43%`。两者均否决并回收GGUF，说明统一或按
  CPU/GPU分段量化不能保留v33的有限正信号。
- 当前正式入口仍是v29；v33研究入口为`fast16/run-colormlm-v33-qwen36-global-moe.bat`，端口
  `8133`。下一步不再扫精度，而是定位v33正向变化对应的层/专家，构建稀疏全深度Qwen专家银行；
  只有新独立门继续零回归且速度通过，才接入v17连续Coder岛。完整证据见
  `fast16/research/v31_qwen36_expert_pair/README.md`。

### 2026-08-01 v36–v38 shared-backbone与工具策略组合

- v36只保留40层Qwen3.6精确router、shared expert、shared gate和MoE入口norm，共240个供体张量；
  256路routed expert bank恢复为v6。单文件`12.679GiB`，相对v6的新独立八维门`4/8→5/8`、
  1净胜0回归，速度`+2.98%`。开发归因门同样1净胜0回归且速度`+1.81%`。
- 在项目既有、未为v36调过的冻结16题中，v36与正式v17逐题胜负完全一致，均为`10/16`；
  v36为`26.832 token/s`，v17为`21.161 token/s`，相对快`26.80%`。因此v36可以承担这套门
  覆盖的通用核心能力，但不能据此宣称超过更大的开源模型或Claude/GPT。
- v37把v17连续Coder岛接入v36后，八维门仍为`5/8`、0修复0回归，速度却下降`20.60%`；组合
  已否决，不生成独立GGUF、不设启动入口。该结果说明已验证模块也不能无条件叠加。
- v38只把v29的16行显式工具策略头接入v36，不加载Coder岛。冻结20题从`7/20`升到`11/20`，
  修复`read-success-finish`、`read-missing-list`、`config-missing-list`、`read-version-finish`四题，
  0回归；train/validation/test分别为`6/10`、`3/5`、`2/5`。
- v38无tools固定请求与v36的content/tool_calls/finish_reason/usage逐字段一致；显式tools固定请求
  `28.689→28.787 token/s`（`+0.34%`）。v38因此晋级为当前最佳研究/体验候选，v29保留为
  Claude Code和`8105`稳定回滚版。证据见`fast16/research/v31_qwen36_expert_pair/README.md`。
- v39在完全相同的20任务/192 token和冻结拟合合同下，改为采集v36自身terminal hidden。
  192/192精确NLL与CNOB均完整，但train/validation/test平均NLL变化为
  `-4.1541/-0.5720/-0.0816`；test候选胜率`57.14%`低于`60%`，最坏留出任务回归
  `+0.1606`高于`+0.03`。按预注册合同停止，不构建runtime、不跑生成门、不扫描留出参数。
  证据与失败边界见`fast16/research/v39_v36_native_policy/README.md`。

### 2026-08-01 v40–v42 保守门控与大核心组合

- v40把v39权重按旧v29/新v39权重范数比固定缩放到`0.2602826`，没有扫描强度。开发集离线最坏
  任务回归降到`+0.0377`，但全新12题严格生成与v36逐题完全相同，均为`5/12`。候选无法改变
  解码，按合同停止。
- v41在v36首决策hidden上训练continue/finish固定ridge探针；train/validation为`10/10、5/5`，
  test只有`2/5`，三道continue均错判finish。小样本隐藏态分类不能作为生产路由。
- v42把v29策略头接到20.20GiB v33。512 batch因Vulkan显存不足启动失败；统一启动器新增可配置
  `--batch-size/--ubatch-size`后，256可稳定运行。旧工具题从`6/20`升到`9/20`且0回归，但既有
  八维16题只有`8/16`，低于v36/v38的`10/16`并回归coding、planning，故不建立用户入口。
- 完整证据见`fast16/research/v40_conservative_policy/README.md`、
  `fast16/research/v41_state_gated_policy/README.md`和`fast16/research/v42_v33_policy/README.md`。

### 2026-08-01 v29序列策略头晋级为工具模态增量版

- v29把v28的首token常量bias改为terminal hidden条件化、逐token、16行稀疏logit残差；候选token
  只由训练任务中跨至少2个独立任务重复出现的token决定，validation/test不参与选行或拟合。
- 冻结离线NLL门在train/validation/test分别为`-1.1428/-0.8812/-1.1522`，20/20任务净改善；
  CNOB重算与API NLL最大差`0.000714`，排除了明显token错位。
- 真实20题生成A/B中，v17为`7/20`、v29为`9/20`；v29修复2题且零回归，但测试留出仍同为
  `2/5`。原八维冻结16题双方均为`10/16`且逐题集合相同，因此这是窄工具状态增益，不是通用智能提升。
- 初版全词表零分支速度为`16.28 token/s`，相对v17的`20.21`回归`19.46%`。改为原位读取/回写
  16个候选logit后为`20.49 token/s`，固定输出等价，计算缓冲约`1980.39 -> 1036.30 MiB`。
- 已接通API请求级物理旁路：非空`tools`且`tool_choice!=none`才建立策略节点；无tools固定请求逐字
  等于v17，显式tools固定请求逐字段等于v29。当前正式契约固定`parallel=1`，不支持混合模态多槽batch。
- 完整运行报告与哈希：`fast16/research/v29_sequence_policy_head/runtime-gate-report.json`。v29可以作为
  v17的安全工具模态增量入口使用，但严格工具状态题只有`9/20`，不得宣称达到Claude/GPT水平。

### 2026-08-01 v20 K3双专家精度线关闭

- v20把v17连续Coder L44–L47岛与K3 L28/E41、E780双专家放入同一图。E41/E780共享的
  `b_in/norm/b_out`完成共享入口和批量出口；固定64-token工程A/B从`13.29996`提升到
  `14.43169 token/s`（`+8.51%`），短提示输出SHA-256一致。这是执行优化，不是能力证据。
- 最初一次F16 dev60记录为平均NLL变化`-0.1048089`，但当前同契约复现为`+0.00358585`；
  撤销批量出口再跑仍为`+0.00315469`。旧正信号不可复现，v20不得据此晋级。
- 专家Q4_0为`+0.08766037`，质量明确反转；Q8_0为`-0.01410018`、原生MXFP4为
  `-0.01008845`，均未通过逐任务LOTO；F16 prefill/MXFP4 decode v4为`+0.00358585`。
  不再对E41/E780扫描alpha、精度或门控。
- 新增K3 v3/v4混合精度ABI、原生MXFP4无损重排编译器与严格manifest校验，作为后续供体
  工程基础保留；所有Q4/Q8/MXFP4/hybrid计划均为研究负对照，不替换v17正式版。
- 完整数字与产物见`fast16/research/V20_K3_PRECISION_REPORT.md`和
  `fast16/research/v20_k3_precision_summary.json`。当前没有维持8107试用服务。

### 2026-07-31 v19末端双输出头实现断点

- 已按两个GGUF的`tokenizer.ggml.tokens`原始bytes建立donor-to-base精确映射：
  `131,612/151,936` donor token可映射（`86.6233%`），base覆盖`53.0010%`，无重复、
  歧义或目标碰撞；EOS、FIM、tool、think等控制token均完成元数据驱动审计。
- 已确认donor `output.weight`不是embedding共享权重，并实现donor L47最终隐藏态经独立
  `output_norm`、Q6_K输出投影、精确token scatter后与base logits连续融合。mapped logits按
  输出位置零均值化，未映射base token获得严格零更新。
- 运行包升级为`fast16/research/v19_dual_head/runtime-head-v2/`：只保存131,612个精确映射
  Q6_K原始行，逐行SHA-256校验与donor原张量比特一致；包大小`222,169,248`字节，比全词表
  v1少`34,670,784`字节，并减少`13.38%` donor输出投影行数。
- 独立构建`llama.cpp/build-v19-dual-head/bin/Release/llama-server.exe`已由MSVC/Vulkan成功编译
  链接。启动器会先清除所有继承的输出头环境变量；输出头`alpha=0`时不验证/加载运行包、
  不设置环境变量且C++不建立分支。v19固定以v17 `runtime-v3`为基线，不使用v18.1桥。
- 真实运行短门已完成。`alpha=0`以v19新二进制运行时，固定seed的64/64生成token与正式v17
  完全一致，12/12精确next-token NLL逐浮点值一致，最大绝对差`0.0`；同时验证启动器会清除
  污染父环境中的输出头变量，因此物理旁路成立。
- 非零输出头已在Vulkan0完整执行。12个teacher token的alpha sweep中，`0.03`为当前最佳测试点：
  平均NLL `1.768049 -> 1.692279`，平均变化`-0.075769`，7项改善、5项回归；删除单个最大改善点
  后平均变化仍为`-0.034278`。但收益主要来自`code-close-elements`，另一个`code-truncate`几乎持平，
  样本覆盖不足以宣称通用能力提升。
- 独立短门比较正式v17与v19 `alpha=0.03`：两者binary-search代码输出逐字一致，均先给出正确函数
  主体、随后无必要续写直至128-token上限；两者`read_file`均正确返回`{"path":"src/main.rs"}`并以
  `tool_calls`停止。未观察到能力增益，v19保持研究候选，不替换v17。
- 输出头成本显著：Vulkan compute buffer约从`563.30 MiB`增至`1504.44 MiB`；独立早期短解码约
  `2.24 token/s`。相邻短门虽因请求形状和推测解码达到代码`15.86`、工具`13.19 token/s`，仍低于
  对照的`19.51`、`17.41 token/s`。正式报告见
  `fast16/research/v19_dual_head/v19_dual_head_short_report.json`。
- 启动器日志可观察性已修复：非零输出头进入mode摘要，日志suffix包含alpha；`alpha=0`条件和物理
  旁路不变。v19.1以`alpha=0.03`作为固定评估点而非最终冻结参数，先保持同一teacher、chat template、
  context和运行参数完成独立对照。10任务每题连续前6 token的60-token集合只算链路与初步泛化冒烟，
  不能替代真正能力决策；必须进一步按任务统计均值、胜负、最差回归、代码/工具分组和逐任务LOTO，
  并补充预先选择的关键决策token：代码运算符、边界条件、返回值、关键API，以及工具名、参数名、
  参数值、结束标记。只有多数任务改善、无单类明显崩坏、所有leave-one-task-out方向为正、关键token
  通过且实际生成相对v17出现可验证净改善后，才进入置信门、概率校准和输出头降本。严格契约见
  `fast16/research/v19_dual_head/v19_1_evaluation_contract.json`。

### 2026-07-31 v19 CNOB一次采集与否决

- 新增默认关闭的一次采集路径，同一teacher-forced请求同时保存base全词表logits、Coder donor
  末端hidden和未中心化mapped logits；arm文件阻止启动warmup污染第0条记录。真实60/60 token
  精确采集完成，打包形状分别为`[60,248320]`、`[60,2048]`和`[60,131612]`，`alpha=0`
  离线精确等价base。
- 固定全局分支从`alpha=0.0001`到`0.03`全部平均回归；最小点仍为22胜、37负、1平，
  `mean_delta=+0.00002053`。因此不再扫描或上线固定全局alpha。
- 新增纯CPU反事实NLL闭式小门与单donor leave-one-task-out验证。显式no-op率为`58.33%`，但
  仅`2/10`任务改善、`8/10`回归，总平均NLL变化`+0.34862`，Java留出任务最差
  `+3.68891`。当前Qwen Coder输出donor无法可靠泛化，R256-CNOB不得据此进入运行时。
- 本轮v19候选已否决，8106停止；正式v17已恢复到`http://127.0.0.1:8105/v1`并核验alias。
  产物见`fast16/research/inference_arch/capture_dev60b.*`。下一 donor（包括K3）必须重新采集
  自身末端hidden/logits并单独通过同一no-op LOTO门，不能沿用本轮失败门或只替换模型名称。

### 2026-07-29 Fast16加速短验证

- 同一台RX 5700 XT、同一v6 GGUF、`--n-cpu-moe 29`、64 token解码的A/B结果：
  - baseline（不设置`GGML_VK_SPIN_FENCE`）：`16.709 token/s`
  - spin fence：`15.363 token/s`
  - spin fence比基线慢`8.05%`，因此不作为ColorLM v6默认配置。
- 这不否定该开关对Qwen3.6裸模型的历史收益；同步优化必须按模型图和CPU/GPU切分分别验证。
- Fast16 Runtime v1现在默认走baseline；复现实验时显式传`--spin-fence`。

### 2026-07-29 CPU split同步合并

- 在`llama.cpp/ggml/src/ggml-backend.cpp`加入环境门控的`GGML_SCHED_MERGE_CPU_SYNC`：
  同一CPU split内的多个输入先统一等待，再进行复制，避免重复提交和fence等待。
- 固定`--n-cpu-moe 29`、Vulkan、64 token短解码：baseline `14.229 token/s`，合并同步
  `16.433 token/s`，相对提升`15.49%`。
- 线程点位短测：8线程约`20.34 token/s`，10线程约`19.10 token/s`；Fast16默认改为8线程。
- `n_cpu_moe=28`在同一设置下约`17.56 token/s`，保留29层CPU专家切分。
- Fast16 Runtime v1默认启用合并同步和8线程；可用`--no-merge-cpu-sync`回退。

### 2026-07-29 CPU尾部同步消除

- 合并同步后的128 token剖析：CPU专家`26.70 ms/token`、GPU提交`8.05 ms/token`、
  阻塞复制`4.81 ms/token`、`final_sync`约`4.95 ms/token`。
- CPU backend图计算是阻塞完成的，因此在scheduler尾部再次同步CPU是重复等待；加入环境门控
  `GGML_SCHED_SKIP_CPU_FINAL_SYNC`，GPU尾部同步仍保留。
- 64 token短A/B：baseline `18.228 token/s`，跳过CPU尾同步`20.209 token/s`，相对提升
  `10.87%`。
- Fast16 Runtime v1默认启用该优化；可用`--no-skip-cpu-final-sync`回退。

### 2026-07-29 Vulkan到pinned host批量读取

- 在`llama.cpp/ggml/src/ggml-backend.cpp`加入环境门控的`GGML_SCHED_BATCH_CPU_READ`。
  同一CPU split的隐藏状态和专家ID不再分别走阻塞读取，而是先把两个Vulkan读取排入同一
  command buffer，再统一同步一次source backend。
- 快路为all-or-nothing，只接受同一Vulkan backend、默认GPU buffer、同设备pinned host、
  连续同布局、非用户输入、非权重输入和无destination event；其余情况完整回退原路径。
- 固定ASCII提示、贪心24-token验证中，基线与批量读取输出逐token一致。
- 相邻128-token、各2重复：基线`20.44 ± 0.73 token/s`，批量读取
  `22.88 ± 1.05 token/s`，相对提升`11.94%`。
- 分项剖析确认快路每token覆盖29个CPU MoE split；旧的58次`blocking_copy`消失。
  机器状态会使CPU专家计算在约20.6–26.8ms/token间漂移，因此不拿跨时段绝对速度作结论。
- Fast16 Runtime v1默认启用；可用`--no-batch-cpu-read`回退。详细记录见
  `fast16/research/fast16_batch_cpu_read_report.json`。

### 2026-07-29 Alder Lake IQ3_S gather内核

- ColorLM v6的CPU专家下投影为IQ3_S。原AVX2内核用16次标量查表构造两个向量；目标
  i5-12400F实测改为两条`_mm256_i32gather_epi32`更快，数学值和后续dot顺序不变。
- 固定提示贪心24-token与标量内核输出逐token一致。
- 两组相邻反向确认：标量`23.93 ± 0.41`→gather`25.49 ± 0.54 token/s`；随后恢复
  标量为`20.95 ± 0.96`，重新启用gather为`26.26 ± 0.22 token/s`。绝对速度仍受机器
  状态影响，但两组方向一致，因此在本机Release AVX2构建中保留gather。
- 该结论是CPU微架构相关优化，不宣称适用于源码注释提到的Ryzen 7950X。详细记录见
  `fast16/research/fast16_iq3s_gather_report.json`。
- gather后只复核相邻运行点：`n_cpu_moe=30`为`24.46 ± 1.91`，10线程为
  `24.03 ± 1.80 token/s`，均未超过生产参数的29层、8线程，因此默认参数不变。

### 2026-07-29 单token MMID静态调度失败对照

- 曾实现单token `MUL_MAT_ID`快路，保留相同量化和dot内核，只删除256专家分组、扫描
  和原子chunk调度；专项形状与模型输出均一致。
- 反向A/B中快路先跑`22.22 ± 1.55`，旧路随后为`23.93 ± 0.41 token/s`，快路慢
  `7.15%`。说明当前动态chunk调度对该CPU更有效，实验代码已完整撤回。
- gate/up只共享一次很小的输入量化、权重仍需读两遍，暂不为其增加通用fusion复杂度。

## 当前架构方向

- 权威主文档：`fast16/COLORLM_NEURAL_BUS.md`。
- 最终结构：Fast Spine + v6安全路径 + 残差专家胶囊 + Neural Bus路由站。
- 运行时：虚拟专家差分、专家/神经元两级路由、GPU/RAM/SSD三级分页。
- 当前不继续围绕Claude Code开发，先完成独立Fast16推理加速。

### 2026-07-29 qwen35moe精确专家槽池

- 已把槽池正确接入ColorLM实际使用的`qwen35moe`，只在单token解码、独立gate/up/down、
  无专家缩放且槽位足够时启用；预填充和不支持布局自动走原始MoE。
- miss不再错误回落到slot 0：当前token选中的专家先同步上传真实IQ2_S/IQ3_S权重，更新LUT后
  才执行GPU `MUL_MAT_ID`。
- 固定提示、贪心采样16 token的原始MoE与槽池输出逐token一致；首个专家的CPU源数据和
  VRAM数据抽查一致。
- `29层×8槽`命中率只有约25%–27%，稳定速度`22.19 ± 0.21 token/s`，未超过基线，
  因此没有把“能跑”当成果。
- 早期同约240MiB显存预算的sweep中，`8层×28槽`曾得到`24.23 ± 0.18 token/s`，
  相对当时基线`22.52 ± 1.30 token/s`高`7.59%`；该结果只作为历史候选，不再视为稳定收益。
- 随后机器状态出现明显漂移。2026-07-29同一进程条件下的相邻128-token复核为：槽池
  `16.98 token/s`、基线`22.52 token/s`，槽池慢`24.60%`。
- 因此Fast16 Runtime v1默认关闭槽池；仅研究时显式传`--slot-pool`启用`8×28`，
  `--no-slot-pool`仍作为兼容参数保留。详细记录见
  `fast16/research/fast16_exact_slot_pool_benchmark_report.json`。

## 下一步顺序

1. 冻结v17为正式版，封存已否决的v19输出头、v20 K3单专家和v21 GLM当前桥；不再从相同样本
   榨取正信号。
2. Kimi K3与DeepSeek-V4当前无法在32GiB内存完整运行，因而不能本地产生真实深层状态；禁止
   再用单专家代替连续供体。下一位优先使用可完整运行的Fara1.5-27B Q5_K_M及F16 mmproj，
   先验证其截图/电脑操作互补性，再采真实深层状态；下载清单见
   `fast16/research/v23_fara_cua_donor/DOWNLOAD.md`。
3. 新供体先过四道离线门：真实供体深层激活、共享坐标桥、整提示独立留出稳定性、供体在目标
   能力短门上的互补性。四门未过，不写C++图、不编译、不启动长测。
4. 通过后才构建第二连续岛和no-op冲突仲裁；继续保留32槽精确缓存、原生状态生命周期、
   `alpha=0`物理旁路与一次相邻速度检查作为不可破坏约束。

### 2026-07-29 Neural Bus v1 实现断点

- 契约已固定在`fast16/research/NEURAL_BUS_V1_SPEC.md`：不替换任何v6专家，在第12、28层
  并联Coder-Next胶囊，以隐藏状态能量门控进行受控残差写回。
- 已从v8中抽出完成坐标运输的Coder-Next第47层专家471，不复制13GiB模型。
  胶囊在`fast16/research/neural_bus_capsules/coder_next_l47_e471_q4_0`，三个Q4_0张量
  共1.69MiB，形状与SHA-256记在`capsule.json`。
- 已新增通用侧载权重容器`llama.cpp/src/llama-neural-bus.{h,cpp}`，并在
  `llama.cpp/src/models/qwen35moe.cpp`中接入同token GGML图。计算为
  `native_moe + alpha * hidden_gate * coder_delta`，不读文本或关键词。
- `COLORLM_NEURAL_BUS_ALPHA`默认为0。为0时不加载胶囊、不建立任何新图节点，
  因而代码路径精确回到v6。
- 已在`fast16/start_fast16_runtime.py`增加`--neural-bus`参数，并新增
  `fast16/run-neural-bus-v1.bat`，默认用独立的8097端口，不覆盖8096生产服务。
- Python语法检查、胶囊提取和Release `llama-server`编译均已通过。
- Neural Bus v1的最小验收已全部完成：
  - `alpha=0`的固定seed输出与旧v6逐字一致，共19个生成token。
  - trace记录到第12、28层残差RMS均非零，证明胶囊实际进入同token计算图。
  - 128-token相邻A/B：v6 `20.42 token/s`，Neural Bus `21.61 token/s`，未出现速度回归。
  - 三项短Code8：v6为2/3，Neural Bus为3/3；胶囊修正了`close_elements`逻辑错误。
- 本阶段判定为“原型成功并晋级”。完整数据见
  `fast16/research/neural_bus_v1_report.json`。当前试用服务为`127.0.0.1:8097/v1`。
- 下一阶段先用1–30秒校准把能量门换成小型神经路由器，再加第二颗Coder胶囊，
  验证隐藏状态top-1竞价是否比单胶囊常开更聪明。

### 2026-07-29 Claude Code隔离实验入口

- 已建立全局命令`colorlab-claude`，固定连接Neural Bus试用服务
  `http://127.0.0.1:8097`，模型名为`ColorLM-Neural-Bus-v1`。
- 配置和会话独立存放在`fast16/runtime/claude-neural-bus-lab/config`，整个运行目录已加入
  `.gitignore`；启动时使用`--bare --disable-slash-commands --setting-sources ""`，不读取
  日常Claude Code的hooks、plugins、skills、memory以及用户/项目settings。
- 已关闭telemetry、错误上报和非必要网络流量，并使用本地占位API key，避免读取Claude
  订阅凭据；Bedrock、Vertex、Foundry开关会被显式清除，启动器原生使用PowerShell并
  强制UTF-8。这里的本地隔离指模型API只连接8097，不是操作系统级断网沙箱。
- 真实Claude Code回环已通过：模型成功调用`Read`，两轮内读取
  `fast16/research/NEURAL_BUS_V1_SPEC.md`，中文结果精确为
  `# ColorLM Neural Bus v1 运行时契约`。
- 该入口隔离Claude状态，但仍会读写启动命令所在的代码目录。需要隔离代码改动时，应在
  测试副本或Git worktree中启动`colorlab-claude`。

### 2026-07-29 Neural Bus v2双胶囊失败对照

- 已直接从本地Coder-Next BF16活检切片构建专家471与专家0两颗运输胶囊，不生成新的
  13GiB中间GGUF；总权重约3.39MiB。
- 运行时已实现由运输后的供体router行驱动的`primary/secondary/no-op`三路隐藏状态硬竞争，
  并保留v1能量安全门和`alpha=0`整段旁路。Release核心编译和真实Vulkan启动均通过。
- embedding路由拟合bias使留出准确率从73.23%降至72.55%，因此自动拒绝bias并使用0；
  该实验只算坐标运输sanity check，不算能力校准。
- 三项短Code8结果为2/3，`close_elements`重新失败；对照为v6 2/3、v1 3/3。
  因能力门失败，速度测试按规则跳过，v2不晋级。
- `8097`已恢复Neural Bus v1与32K上下文。v2源码、胶囊和
  `fast16/research/neural_bus_v2_report.json`作为可复现失败对照保留。
- 下一次路由升级必须先实现`no-op/e0/e471`强制路径的反事实next-token NLL标签，不能再用
  token embedding或不可获得的推理时“误差信号”代替能力监督。

### 2026-07-29 ColorLM v9-alpha交付

- 已把当前胜出的Neural Bus v1冻结为`ColorLM-v9-alpha`，模型包契约位于
  `fast16/models/ColorLM-v9-alpha.clmpkg.json`，锁定v6基座、胶囊三张量SHA-256、
  第12/28层站点、alpha与32K运行参数。
- 新入口为`fast16/run-colormlm-v9-alpha.bat`；隔离Claude Code命令`colorlab-claude`
  也已切换到该版本名。8097实际暴露模型为`ColorLM-v9-alpha`。
- alpha阶段采用GGUF加1.69MiB侧载胶囊的零复制包，不额外复制13GiB基座；单文件封装是
  后续发布工程，不阻塞使用，也不把简单重命名伪装成新能力。
- v9路由研究与alpha交付解耦。任何新路由必须先超过v9-alpha的人工体验，才允许覆盖8097。

## 磁盘规则

- D盘当前约105.5GiB空闲（2026-07-31记录）。
- v6是生产基线，不删除。
- v7是未旋转失败对照，v8是运输对照；需要新建13GiB checkpoint前先决定删除哪一个。
- K3/Coder-Next远程活检缓存只有小切片和索引，不下载1.454TiB或148GiB整模。

### 2026-07-29 v9宏胶囊预算修正

- 用户人工试用判定`ColorLM-v9-alpha`的自然对话和Claude Code工具调用仍不足以投入使用；
  它保留为Neural Bus v1研究对照，不再把Code8单题翻转当成产品级v9。
- Grok审稿对v2失败原因的诊断成立：供体embedding路由饿死e471，能量门只是幅度保险，
  反事实next-token NLL是后续通用路由标尺。但其v1.5方案只优化1.69MiB单胶囊，
  不作为最终模型容量上限。
- 最终15--30GiB预算允许把v6的12.79GiB安全主干之外的数GiB真正分配给多层宏胶囊库。
  路由校准和宏胶囊提取并行，不等待小胶囊路由完成后才开始K3/DeepSeek/小米供体研究。
- K3官方结构已核实为latent MoE：7168维隐藏态先投影到3584维，完整专家为
  `3584 -> 3072 -> 3584`，再经RMSNorm投回7168维。第92层两个共享投影共约98MiB，
  每颗原始MXFP4专家约16.73MiB；一层16专家冷库约366MiB，适合构成数GiB多层能力库。
- `fast16/research/kimi_k3_expert_capsule.py`可纯离线规划并以HTTP Range提取任意K3专家的
  连续MXFP4张量。该阶段之后已完成真实隐藏态候选清单、latent bridge、完整K3宏胶囊和
  `v9-K3-rc1`启动包，后续结果见下节。

### 2026-07-29 v9-K3-rc1到v12-Neural-Alloy交付

- 上一段“下一可见产物”已经完成，不再是当前断点。`v9-K3-rc1`把两颗真实K3宏胶囊接入
  L12/L28；`v10-K3-Multi`进一步接入四颗K3专家与隐藏态连续软路由。
- `v11-Coder-K3`在同一GGML图中同时执行Coder E471和K3支路。两条支路读取同一份原始
  `attn_post_norm`并并行写回残差，尚不是供体之间互相通信的级联推理。
- `v12-Neural-Alloy`将v6已有GLM/Qwen3.6核心血统与v11的Coder/K3运行支路统一为
  `colorlm-neural-slice-abi-v1`计划。v12相对v11没有新增专家权重，但正式入口现在真实解析并
  验证该计划，因此它是统一交付契约，不再只是换alias。
- v12有效权重`14,115,335,072`字节（13.146GiB）。alloy plan SHA-256为
  `588145eaf57de0b6a5fbae43470497816c2d97b5ef6cd30fe8f79b6cf8c7e67f`，正式`llama.dll`
  SHA-256为`dd5d61924b49fd62ff1fec3e98e473b1209e059cba34da9380bbcf5fbb704640`。
- Coder Q4加载器现在强制校验`capsule.json`、结构、张量集合、文件名、尺寸和SHA-256；K3
  继续执行manifest与全部运行张量SHA校验。普通启动校验核心尺寸与所有小清单，正式验收可加
  `--verify-alloy-core-sha256`完整读取核心GGUF。
- 官方入口为`fast16/run-colormlm-v12-neural-alloy.bat`。完整关闭两条运行时侧链的验证命令：
  `run-colormlm-v12-neural-alloy.bat --neural-bus-alpha 0 --k3-alpha 0`。`--validate-only`可在
  不启动模型的情况下检查最终契约。
- release验收脚本为`fast16/research/verify_v12_neural_alloy.py`。全核心SHA、当前DLL/服务、
  alias、32768上下文、Vulkan Coder/K3非零残差与2-token最短烟测均通过；长榜按约定未运行。

### 2026-07-30 ColorLM v13因果稀疏版

- 完成逐站强制路径和反事实next-token NLL研究。12个任务、72个teacher token上，L12单独K3
  的平均NLL为`2.438846`，相对no-op的`2.487283`改善`1.9474%`；12项中8项改善，
  其中5个工具任务全部改善。
- L28单独K3也有正收益，但L12与L28同时开启时平均NLL退化到`2.492037`，首次确认跨层
  胶囊会产生破坏性干涉。正式图因此只保留L12的K3专家41和780，不再按“胶囊越多越强”堆叠。
- 全局、L12和L28线性Intelligence Router均未通过稳定性门。L28在单个seed曾通过，
  但12个随机拆分仅4次通过，严格LOTO准确率`32.73%`，低于固定控制`40.0%`，已拒绝晋级。
- 正式人工候选为`ColorLM-v13-Causal-Sparse-L12`：v6核心加L12两颗K3宏胶囊，不加载
  Coder与L28支路。有效权重`13,922,694,048`字节（`12.967 GiB`），比v12少约`192.6 MiB`侧链。
- 相邻热态64-token速度为v12 `13.8802 token/s`、v13 `13.8412 token/s`，判定持平，
  不宣称提速。TypeScript短代码和`read_file`工具调用烟测通过，未运行长榜。
- 2026-07-30从零重启时曾遇到一次Vulkan `ErrorOutOfHostMemory`；确认8102无残留监听且系统
  内存恢复后，干净重启成功。运行态包契约、alias、32K上下文和L12计划均再次验收通过。
- 当前正式试用地址为`http://127.0.0.1:8102/v1`；隔离Claude Code入口`colorlab-claude`
  已从旧v9切换到v13。完整报告见`fast16/research/v13_causal_sparse_report.json`。
- v13恢复后已通过真实Anthropic `/v1/messages`短工具回合：模型返回`read_file`、参数
  `{"path":"src/main.rs"}`且`stop_reason=tool_use`，证明`colorlab-claude`连接的v13协议链可用。

### 2026-07-30 v14联合站点路由负对照

- 新增`fast16/research/v14_joint_site_router.py`，把站点选择改为联合四状态
  `{no-op, 仅L12, 仅L28, 双站}`，避免两个独立路由器忽略跨层干涉。目标直接使用实际
  next-token NLL，不再把分类准确率当最终目标。
- 现有72 token的逐token oracle平均NLL为`2.342122`，比固定L12的`2.438846`仍有
  `3.966%`理论余量；四路最佳次数分别为`14/14/16/28`，说明双站不是完全无效，
  而是少数巨大损失拖垮平均值。平均非加性交互项为`+0.084682 NLL`。
- 对拼接的L12/L28 no-op隐藏态做嵌套LOTO核回归：外层逐任务留出，内层再逐任务选择
  linear/RBF、正则和安全margin。路由NLL为`2.466255`，比固定L12差`1.124%`；
  12个任务零净改善，90%任务bootstrap增益区间为`[-0.058819, -0.000114]`。
- 该联合路由严格拒绝，不生成运行计划，不占用v14产品版本。失败揭示新的监督错位：
  no-op轨迹隐藏态不能可靠预测L12写回后L28分支的收益。下一研究输入必须来自L12已生效的
  当前轨迹，并加入站内实际胶囊delta与原生残差的幅度/方向特征，而不是继续换分类器。
- 可复现报告为`fast16/research/v14_joint_site_router_report.json`；脚本离线约7秒，未运行长榜。

### 2026-07-30 v15残差感知路由研究

- 为L28候选K3支路新增仅校准时启用的6维GGML探针：`hidden_rms`、`native_rms`、
  `k3_delta_rms`、`hidden_delta_cos`、`native_delta_cos`、`energy_gate`。默认不开环境变量时
  不建立探针节点，不增加生产图计算。启动器新增`--k3-feature-dump`及站点/记录上限参数。
- 采集器新增CL3F二进制格式、严格追加字节边界、文件身份、前缀SHA、有限值、shape、
  sample顺序和精确记录数校验。一次`max_records=16`导致只得到12/72条时被正确拒绝，
  未生成可训练sidecar。
- L12生效后L28隐藏态相对no-op发生`6.61% RMS`位移，但隐藏态四路联合门仍比固定L12差
  `1.124%`；只允许`L12/L12+L28`的二元门差`1.370%`，两者均拒绝。
- 6维实际残差特征的嵌套LOTO二元门显著缩小差距，但仍比固定L12差`0.128%`；
  12个任务中5个改善，90%任务bootstrap区间`[-0.009405, 0.001161]`跨零，因此不晋级，
  不生成运行计划。报告为`fast16/research/v15_residual_aware_router_report.json`。
- 重新编译后双站研究路径出现数值轨迹漂移，因此旧双站NLL未与新探针数据混用；新DLL下
  重新采集了L12对照。正式L12的72个NLL与探针前二进制逐值完全一致，最大差`0.0`。
- 当前`llama.dll` SHA-256为
  `5869647f579b9f3c395cfc7fb349787d637c1170084e068d6038b0ef452d8659`。v13包与研究报告
  哈希已同步，8102运行态验收和Anthropic `read_file`工具回合再次通过。
- 决策：L28路线封存为研究负对照，正式图继续使用v13固定L12。下一能力增量不再从当前
  72-token路由集榨取，需增加独立任务覆盖或选择新的互补供体能力切片。

### 2026-07-30 v16 Coder完整神经块可行性与提取阶段

- 冻结v13正式服务，不再继续尝试L28隐藏态路由。v16选择Qwen3-Coder-Next L47完整全注意力块，
  而不是再运输单颗FFN专家。供体与主干均为2048 hidden、16个Q头、2个KV头、256 head dim；
  供体独立RoPE theta为5,000,000，不能复用主干的10,000,000。
- 已生成`colorlm-neural-block-abi-v1`机器契约和精确Range计划。L47共1549张量，BF16为
  `3,284,153,344`字节（3.059GiB），Q4_0估算0.860GiB；512专家每token精确激活10个。
- 1549张量在ModelScope两个分片中连续成两个Range（516MiB与2616MiB），无需下载完整80B供体。
  新增断点提取器，默认只预检，`--download`才访问远端；首次长连接中断后已升级为16MiB子Range
  自动续传，已有partial必须保留。
- 独立KV的实现路径已核实：首站从L39修正为L35，因为最后一层会提前裁成输出token，而L35仍有
  完整prompt轨迹。ColorLM L35原生使用recurrent/ColorKernel状态；`llama_memory_hybrid`已支持同一
  层号同时注册recurrent和Attention cache。v16启用时将L35加入attention filter即可，
  不需要伪造第41层或重写整个缓存。`alpha=0`时不注册、不加载、不建图，保持v13物理旁路。
- 下一断点：完成Range提取和Q4块包编译；随后接入L35双缓存、复用Qwen3Next full-attention图、
  完成shared expert与512专家exact top-10分页。详细设计见`fast16/V16_NEURAL_BLOCK.md`，机器计划见
  `fast16/research/v16_coder_neural_block_plan.json`。
- 2026-07-30继续：L35完整Neural Block图已接入源码，路径为坐标桥、独立RoPE全注意力、512专家
  top-10与共享专家、去identity残差、能量门回注；`alpha=0`物理旁路不变。提取器新增完整shard
  Range导入与已有`.part`逐字节校验。当前仍缺shard40后2.066 GiB；C++首轮编译已进入
  `qwen35moe.cpp`并修正类声明，第二轮编译复核被工具审批通道中断，尚不记为编译通过。
- 2026-07-30 v16构建完成：完整shard40为`3,365,572,528`字节，已有500MiB断点经逐字节校验后
  成功导入；两个L47 Range共3.059GiB。编译器将1549个BF16源张量编成18个运行张量，正式
  `weights.bin`为0.876GiB（含两张F16坐标桥）。CPU与独立Vulkan构建均通过，独立
  `llama-server.exe`位于`llama.cpp/build-v16-vulkan/bin/Release/`。首次8104加载在基础模型阶段
  因8102的v13进程已占14.68GiB私有内存，无法再分配1GiB Vulkan_Host缓冲而退出；神经块尚未
  加载，因此这不是v16权重/图失败。已新增v16一键启动与隔离Claude Code入口，待获得许可后
  临时停止8102并进行单实例启动级验证。
- 2026-07-30 v16首次运行通过：经用户许可停止8102后，独立Vulkan服务在8104加载成功。
  日志确认完整块`L47 -> L35`以896.89MiB加载到Vulkan0，512专家top-10；v13的两颗K3胶囊也
  同时加载。首次建图发现ColorLM MRoPE四轨位置不能直接传给供体NeoX RoPE，修正为同图取时间
  位置轨后重新编译通过。最短真实生成返回`OK`（16 prompt + 2 completion tokens，3.55s）。这证明
  同图可加载、可建图、可生成；尚不宣称能力提升。正式记录为
  `fast16/research/v16_startup_report.json`，当前服务为`http://127.0.0.1:8104/v1`。
- 2026-07-30 Claude Code兼容入口修复：Anthropic顶层`system`与`messages`中的后置system现在会在
  转换层合并为唯一首条system，严格Qwen3.5模板不再抛`System message must be at the beginning`。
  重复system直连请求返回HTTP 200/`OK`（2.30s），隔离Claude Code真实`-p`请求返回`OK`
  （15.5s）。同时修复PowerShell空`--setting-sources`吞掉`--settings`的问题。
- 工具能力尚未晋级：64-token直连能生成并解析`tool_use`，但该次必填参数成为空对象；真实Claude
  Code在工具结果回填后未及时停止，180s测试被取消。当前结论是“Claude Code对话入口可用”，
  不是“工具闭环已稳定”；下一步应修复工具参数遵循与结果回合停止，不重复测试已解决的500。

### 2026-07-30 v16.1缓存隔离修复

- 源码审计推翻了“v16已有独立KV”的旧记录：L47首版通过缓存型`build_attn`复用L35层位，
  供体K/V可能覆盖主干后续token所需的注意力缓存。这是结构正确性问题，不是单纯窗口过大。
- v16.1改为当前ubatch内局部因果注意力，彻底停止供体读写主干KV；保留供体Norm、RoPE、
  query gate、shared expert与512专家top-10。真正独立的有界KV将在v17四层神经岛实现。
- Anthropic兼容接口新增可配置输出硬上限，默认1024 token，避免Claude客户端的32K/64K预算
  触发失控长生成；OpenAI兼容接口保持原行为。
- 独立Vulkan Release编译和8104启动通过。最短请求返回`OK`；128-token Rust生成实际解码
  `14.36 token/s`；4,428-token提示正确取回标记，预填充`200.56 token/s`，随后短解码
  `19.53 token/s`。这证明缓存隔离和长上下文稳定性，不证明能力已超过v6。
- 当前主线：先构建Qwen3-Coder-Next连续四层神经岛，再按`fast16/V17_DONOR_SCREEN.md`筛选
  GLM-5、Kimi K3连续块、DeepSeek与MiMo；不再把未经归因的小切片直接叠进正式模型。

### 2026-07-31 v17连续Coder神经岛

- 已完成Qwen3-Coder-Next L44–L47四层连续岛：L44/L45/L46为Gated DeltaNet，L47为
  Full Attention；四层都保留原生Norm、共享专家、512专家top-10 MoE，并只在入口和出口各做
  一次2048维坐标运输。岛权重为`3,773,072,128`字节（3.514GiB），与v6主干合计约
  `17,504,894,624`字节（16.303GiB）。
- 正式运行包为`fast16/research/v17_coder_island/runtime-v3/island.json`，注入主干L35，
  `alpha=0.02`，16K上下文；正式入口为`fast16/run-colormlm-v17-coder-island.bat`，当前服务
  为`http://127.0.0.1:8105/v1`。
- 源码审计发现旧私有状态的生命周期不可靠：L47私有KV在图复用时会把首个ubatch位置固化，
  L44–L46私有循环状态也不属于`llama_context/seq`。v17.2已删除全部私有状态实现，改用
  llama.cpp原生hybrid memory：L44/L45/L46分别占recurrent槽`3/7/11`，L47占独立KV槽`34`。
  状态现在随context和sequence创建、清理；Neural Block与Neural Island强制互斥，Island启用时
  也禁止`COLORLM_V4_KERNEL_LAYERS`占用`3/7/11`。
- 4235-token标记取回短测通过；随后独立新请求返回`NONE`，没有读到前一请求标记。该结果证明
  当前单sequence服务中的长提示状态和请求间清理路径正确，不外推为多sequence并发已验收。
- Release `llama.vcxproj`增量编译和链接通过；当前`llama.dll` SHA-256为
  `a03d8701fe054b71aaa87d6ebaa85eea6b67a0f0ed085709fdadf20538dfe72e`。旧私有状态符号扫描为零。

### 2026-07-31 v17.2精确专家缓存

- 四层完整专家银行继续驻留`Vulkan_Host`，每层GPU只放32个热槽，四层共增加约216MiB显存。
  缓存只用于单token decode；prefill始终使用完整专家银行，避免一个批次的不同token选出超过
  槽容量的专家集合。
- decode miss会先上传真实top-10专家和逻辑专家到物理槽的LUT，再继续计算；同一步命中的槽被
  钉住，多个miss批量提交后只同步一次。miss不能回落到错误专家，也不能淘汰当前步骤仍要用的槽。
- 同一160-token Rust短请求中，关闭缓存为`4.45 token/s`、服务计时37.130秒；32槽缓存为
  `10.41 token/s`、服务计时16.418秒，解码吞吐约`2.34x`。该输出命中160-token上限而被截断，
  因此只算连续生成和缓存性能证据，不算代码任务完成。
- 相同4235-token标记请求中，关闭缓存为prompt `228.23 token/s`、generation
  `5.80 token/s`；32槽为prompt `241.35 token/s`、generation `13.39 token/s`，解码约
  `2.31x`，两次均取回正确标记。prefill理论路径未改变，prompt差异只记观测值，不归因为缓存收益。
- 以上是两次单实例短比较，不是重复多轮稳定基准；它足以支持“32槽作为v17默认值”，不证明
  普遍任务都能保持相同倍数。完整机器记录见
  `fast16/research/v17_2_state_cache_report.json`。

### 2026-07-31 v17.3专家路由与缓存可观测性

- 专家缓存现在按供体层输出`steps/hits/misses/hit_rate/upload_bytes/upload_batches/resident/capacity`，
  首个decode步和每32步记录一次。一次39-token Rust短生成在第32步的四层合计命中率为
  `24.22%`，上传`1,716,387,840`字节；这说明32槽仍受PCIe上传限制，但可在8GiB显存边界内运行。
- 否决了把缓存回调移出专家ID主链、试图恢复Vulkan top-k路由融合的方案。旁支版同题只有
  `3.88 token/s`，并且缺少从上传完成到专家remap的显式跨后端依赖；恢复串行精确依赖后达到
  `13.36 token/s`，输出为正确的Rust偶数求和函数。该单次短测只证明这次消融方向，不外推为
  稳定通用吞吐。
- 64槽把合计命中率提高到`35.31%`，但整卡专用显存观测达到约`7362.9 MiB`，同题降到
  `0.89 token/s`，因此明确拒绝；正式配置保持每层32槽。
- LRU同一token共用一个epoch，避免top-10内部排名制造伪新旧顺序。启动器在Neural Island启用时
  固定`--fit off`，防止自动放置忽略侧载岛和缓存显存；正式入口已切换到独立
  `build-v17-perf`二进制，8105在线。完整报告见
  `fast16/research/v17_3_cache_route_report.json`。

### 2026-07-31 v18真实深层激活桥

- 完整供体为`Qwen3-Coder-Next-UD-IQ3_S.gguf`，29,690,687,488字节，GGUF元数据确认是
  79.67B参数、48层、2048隐藏宽度、512专家top-10的Qwen3Next。串行采集时完整供体可在本机
  以约1.76GiB Vulkan权重和28.07GiB CPU mmap加载；空输入warmup会异常退出，采集入口已固定
  `--no-warmup`，真实prompt路径稳定完成。
- 通用scheduler callback精确采集ColorLM L35 `attn_residual`与供体L43 `l_out`，默认关闭时
  不挂callback。两个词表不要求切词相同，而是按原prompt累计UTF-8字节结束边界配对。
- 12条代码/调试/工具prompt得到主干815 token、供体778 token，其中772个深层状态成功配对，
  平均匹配覆盖率`95.06%`；训练/留出按整条prompt拆分。
- 两个不同留出划分均通过：seed18的余弦中位数`0.1951 -> 0.5499`、同原生幅度RMSE比
  `0.7437`；seed42为`0.1899 -> 0.5312`、RMSE比`0.7640`，两次均为3/3留出prompt正提升。
  RMSE门已修正为新旧桥使用同一个训练集原生幅度；各自在留出集重新找最优scale只保留诊断，
  不参与晋级。
- 隔离包为`fast16/research/v18_activation_bridge/runtime-v1/island.json`，权重总量不变，
  只替换四层包的F16入口/出口桥并重算全部SHA-256。`alpha=0`短门走物理旁路，同seed输出与
  v6路径逐字一致。
- 一条未参与拟合输出判分的Rust所有权短任务中，v17和v18都给出同一正确拥有所有权实现；
  v18单次26.09秒、v17单次35.56秒，只记无明显速度回归，不把单次时延当稳定提速或能力胜利。
- 当前最重要残余风险是零先验Procrustes只有约600个训练状态，未观测子空间仍欠约束。v18作为
  正式研究候选可人工使用；扩L40-L47前，优先补到至少2048个配对状态或实现只在观测子空间
  学习、其正交补保持旧桥的nullspace-anchored完成方式。

### 2026-07-31 v18多切分稳定性纠偏

- 上述两个单独留出切分不足以支持晋级。新增12折整提示LOTO和5个唯一seed审计，并让候选桥与
  embedding基线各自只使用训练prompt校准幅度，禁止共用候选尺度造成偏乐观比较。
- 现有中位范数尺度的LOTO余弦提升为`+0.3657`，12条prompt全部正提升，但NRMSE比率为
  `0.9559`，超过`0.95`门槛；5个seed为`0/5`通过。
- 训练集最小二乘尺度的LOTO通过，NRMSE为`0.9158`、比率为`0.9332`，但5个seed只有`2/5`
  通过，低于预先固定的`4/5`要求。失败集中在未见prompt的幅度泛化，不是方向翻转；LOTO最差
  预测方向余弦仍为`0.9960`。
- 无先验桥有效秩仅`772/2048`（`37.70%`）。因此v18隔离包只保留为研究产物，不覆盖v17正式
  入口。完整结论与复现命令见`fast16/research/v18_activation_bridge/STABILITY_AUDIT.md`和
  `stability_report*.json`。

### 2026-07-31 v18 nullspace-anchored稳定候选

- 在不改变观测子空间Procrustes最优解的前提下，新桥只用旧embedding桥来确定零奇异值子空间的
  正交补；这样保留772个真实配对状态学到的方向，同时不让未观测的`62.30%`维度随机旋转。
- `nullspace_anchored + train_least_squares`通过同一套预先固定的严格门：12折LOTO余弦提升
  `+0.3760`、NRMSE `0.9021`、相对基线比率`0.9192`、12/12 prompt正提升；5个唯一seed为
  `5/5`通过，最差NRMSE比率`0.9475`，最差预测方向稳定性`0.9896`。
- 稳定候选运行尺度为`2.26531358`，F32正交与往返RMSE约`5.95e-8`。三张运行矩阵和完整稳定
  报告位于`fast16/research/v18_activation_bridge/candidate-nullspace-anchor-v1/`。
- 隔离神经岛已封装为`fast16/research/v18_activation_bridge/runtime-v2/island.json`，权重仍为
  `3,773,072,128`字节，只替换入口/出口桥及相关哈希。它已具备进入最短运行时短测的资格；
  在运行短门通过前不覆盖8105的v17正式入口。

### 2026-07-31 v18.1最短运行门

- `runtime-v2`静态校验与真实Vulkan启动通过；必填工具参数测试正确返回`read_file`及
  `{"path":"src/main.rs"}`。工具结果回合没有重复调用工具，但96-token预算内未及时收束，
  因此只记partial。
- 独立Unicode Rust任务中v18.1与v17均出现同类Markdown/冗余结构并在128 token截断，未观察到
  能力增益。附加的简单Rust偶数求和烟测完整正确，39 token冷启动解码为`6.62 token/s`；该单次
  结果不证明稳定回归，但足以阻止当前候选覆盖正式速度路径。
- 决策为`hold_research_candidate`：nullspace桥的几何突破保留，产品版本仍为v17。正式服务已恢复
  `http://127.0.0.1:8105/v1`并核验模型名。报告见
  `fast16/research/v18_activation_bridge/v18_1_runtime_short_report.json`和
  `v18_1_additional_smoke_report.json`。

### 2026-08-01 v43反事实no-op策略头否决

- 数据集为120条完整工具状态轨迹、60个成对反事实组，train/validation/test为
  `72/24/24`；覆盖工具参数纠错、继续/结束、缺参澄清、多步规划、电脑操作和代码调试。
  三个分割仍共享同一语义生成骨架，因此只是运行时原型数据，不是跨模板盲测。
- v36釆集完成`720/720` teacher token和`720/720`精确NLL，CNOB中有720个2048维terminal
  hidden和720个248320维base logits，共721,152,000字节。数据、teacher和CNOB SHA-256分别为
  `a4ee853502a1326800b89d7cacd27fdf7b48266ad664d5b23cb14f8f6c2b2fcb`、
  `1d2be3d8a67ac9e75a5d01c58a4dea656bb373001cb9417db56270d23b9271f9`、
  `dfa3cea982a307eac6cc075fb5bf1a3d8d17f9918c5433d0f08b07dde4762f3c`。
- 冻结结构为单样本L2归一化→仅train PCA rank 8→九类ridge分类；class 0为精确no-op，
  其余8类只修正8个稀疏候选token。最终F32回放train/validation/test平均NLL delta为
  `-1.02578/-1.03262/-1.02190`，任务胜负为`72/0、24/0、24/0`，精确no-op率约25%。
- 运行包为`colorlm-sequence-policy-runtime-v3`，6张F32/I64张量、74,180字节。独立CPU自检
  通过720/720样本，类别预测和no-op判定均为720/720一致。C++已在Vulkan图内执行
  hidden归一化、PCA、分类、softmax、no-op门和稀疏logit回写；同时修正Vulkan
  `GGML_STEP`的`x >= 0`错误，使其与CPU/CUDA的`x > 0`合同一致。
- 预注册的24条test真实生成门结果：v36和v43均为`12/24`，wins=`[]`、regressions=`[]`、
  net=`0`，未达到“净修复至少3、回归最多1”。剔除随机tool-call ID后，6/24条输出发生
  语义或JSON形式变化，但没有任何一条从失败翻转为通过。因候选没资格晋级，未跑旁路/速度门。
- 否决原因已收敛：离线目标只奖励6个teacher-forced token上的NLL，8个候选行又多为工具
  包装和JSON结构token；它能改变格式/短语概率，但不足以选择正确工具、参数或高层动作。
  validation/test共用语义骨架还高估了离线泛化。v43运行候选按合同停止，不对已消费test扫参。
- v44的词表关键span审计已给出直接证据：120/120条任务都有关键动作token落在v43的前6 token
  窗口之外；2,654个关键token occurrence中2,274个未被监督。工具名span从index 4开始，但
  390个子token中210个在窗口外；参数span从index 9–24才开始，1,324/1,324全部在窗口外；
  结束JSON字段也有740/940在窗口外。报告位于`fast16/research/v44_critical_action_bus/critical-span-audit.json`。
- 下一代不再追求“更好的公共标点token NLL”，而是要建立新模板簇/新盲测，并让状态类别直接选择
  动作前缀、工具名和关键参数token组，再以完整轨迹成功作为晋级门。完整产物位于
  `fast16/research/v43_policy_dataset/`。

### 2026-08-01 v44关键动作稀疏头否决

- 408/408条关键span teacher完成精确NLL与hidden/full logits采集，CNOB哈希为
  `e41b9da7f9c7042418f1fcca866524d2b079b18740f43d757c4e6fc09e7a53b9`，test泄漏为0。
- 预注册rank16/ridge 0.1/strength 12结果为train和validation均为0 rescue、0 regression、
  NLL delta 0、exact no-op 100%，开发门失败，不允许扫参数挽救。
- validation在28个候选token内原生已答对94/98，最多只有4个可救样本，但合同要求
  至少8个rescue。这证明“候选集内top-1”是错位代理目标，无法代替全词表上的完整动作
  生成。v44稀疏单token头正式停止。
- 下一步为v45主脑筛选：只使用新跨模板validation 60题，顺序比较当前v38、完整
  Qwen3.6-35B-A3B和完整GLM-4.7-Flash；blind 60题保持未触碰。若无底座净胜至少4题，
  则保留v38并转完整span序列解码器/能力岛。

### 2026-08-01 v45完整主脑筛选与v46连续中层皮层

- v45在新跨模板validation上：v38=`30/60`，完整Qwen3.6=`40/60`且`10胜0回归`，
  GLM-4.7=`17/60`。但一次性blind上Qwen仅`23/60`对v38的`22/60`，`4胜3回归`，
  未达净胜4的门，所以完整换脑未晋级。
- v46把Qwen3.6 L16–L31的292个完整层张量运输到v36，保留v36前16层、后8层、
  embedding、final norm和output head。物理GGUF为`15.767GiB`，Vulkan真实启动和24–60题运行稳定。
- 已消费validation的开发门中v46=`35/60`对v38=`30/60`，`5胜0回归`，规划`+4`、
  调试`+1`，墙钟几乎相同。该结果只允许创建新blind，不允许直接晋级。
- 全新、六类各4题的24题blind已冻结，哈希为
  `4f4114c4f9bb24ac975f1ded85db1a5dda06181a7a63d603ce181c36112aa721`。一次运行结果：
  v38=`15/24`，v46=`13/24`，`1胜3回归`，规划`-3`，墙钟`+12.87%`。
- v46已被blind否决；不扫其他层范围，不把validation大增益写成能力突破。正式最佳仍是
  `ColorLM-v38-Qwen36-Shared-Sequence-Policy`。下一步转跨模板受控序列能力岛/真实任务蒸馏，停止无路由直接层拼接。

### 2026-08-01 v47 Dual-Tempo Neural Bus 研究基础设施与前端单题证据

- 正式最佳仍是`ColorLM-v38-Qwen36-Shared-Sequence-Policy`；v47没有用户启动入口，也没有声明
  模型能力晋级。新主线固定为双节奏：简单token走多token草稿并由v38无损验证，困难决策才走
  `K=0..4`局部潜在循环与完整序列能力岛。
- `fast16/research/v47_dual_tempo_bus/`已实现每任务一次terminal hidden绑定完整target序列的
  数据契约、CPU GRU序列岛训练器、Design IR schema、动态shortlist覆盖扫描和严格自检。合成夹具
  154,889参数、F32约0.59MiB，80 epoch约0.58秒，跨模板validation完整序列率100%；这只证明
  训练管线可用，不是ColorLM能力证据。
- 四个未来位完整词表投影约2.50B FLOPs/步、解析比约41.6%，已从默认设计移除。v44的408个
  工具名/参数/结束字段关键token上，top-16/128行只有406/408；top-32、最近上下文96、train-only
  高频64、最多192行达到408/408和validation 100%覆盖。下一草稿候选冻结为rank-64/192行；
  teacher-forced开发覆盖不等于真实接受长度或端到端加速。
- DALI式动态专家放置离线回放把GPU上传从1961次降到519次，但产生1347次CPU执行；冷miss只降
  2.56%，加预取后SSD字节无明显优势，未过预声明5%门，暂不进入C++。
- 并行前端组已建立24题冻结短门，train/validation/blind各8题且24个模板族无交叉。六个桌面
  历史页面静态排名中`index.html`第一；v38/v46仍偏通用模板。
- 主线只消费`pf47-train-01`一条train开发题。v38直接HTML在1800/2200 token均未闭合；紧凑
  Design IR可在1065字节内正确表达筛选、排序表格、详情抽屉、告警与375px规则，但条件HTML在
  3000 token仍截断，一次1400 token续写又重复函数并截断。确定性尾部编译器补齐状态、过滤、
  键盘、ESC、焦点与闭合后，冻结静态门从普通三卡片42.85分/惩罚17提升到80.55分/惩罚6，
  所有关键项通过。
- 浏览器真实检查：桌面筛选后10行全部为警告，详情抽屉和告警弹窗可用；375px下表格隐藏、50张
  卡片显示、无横向溢出、控制台无错误。移动卡片文字仍拥挤。候选含确定性编译，只批准继续训练
  IR序列岛和通用结构编译器，不允许宣称纯v38/v47已获得80.55分。
- 下一步顺序：8条train紧凑IR教师与真实initial-hidden采集→序列岛→8条validation门；并行训练
  rank-64/192行草稿头并先看完整轨迹接受长度。validation通过后才一次blind；DALI分页保持暂停。
  完整交接见`fast16/research/v47_dual_tempo_bus/HANDOFF.md`。

### 2026-08-01 北极星 v47 Parallel Genome Head 真实主干训练

- 新编译的`build-v19-dual-head`服务已完成terminal-only capture冒烟：1条`kind=4`记录，hidden
  宽度2048且全部有限。随后对144条唯一prompt正式重放，NLL精确覆盖`144/144`，耗时
  `257.73s`；CNOB含连续record `0..143`，共144条2048维terminal hidden，SHA-256为
  `5b664eb4bbd9f52e856b4feb956fcef4b46027436bdf4206145d45c4745ae400`。
- 27.9万参数、latent 128、20字段并行Genome Head在本机CPU训练约`3.84s`。train完整Genome
  `128/128`；train-only internal validation完整Genome率`93.75%`、字段准确率`99.6875%`、
  最弱字段`93.75%`，通过预注册的`75% / 90% / 50%`内部开发门。
- 六个历史网页已转成九类哈希化负约束：默认三卡片、emoji图标、假交互、远程资源、焦点、
  减少运动、响应式、语义HTML和表单标签。数据生成器现在严格投影全部九类合同，不复制历史
  HTML正文、远程URL或绝对路径。
- 当前决策仅为`allow_compiler_ab`：Genome字段准确不等于网页质量。下一步必须将头输出交给
  确定性编译器，在冻结validation上跑静态评分与375/768/1024/1440四档浏览器action trace；
  通过后才允许一次blind，未通过前不得宣称北极星v47能力晋级。
- 免费Ascend 910B4已用4096×4096 FP16矩阵乘真实验证。`fast16/cloud/ascend910b/`提供强制
  `npu:0`的doctor与训练包装器；若未真正落到NPU会失败，禁止静默CPU冒充。该云算力主要用于
  后续更大的多能力头、LoRA和蒸馏，而不是为了加速本机仅需数秒的小头。

### 2026-08-01 北极星经络：300B+器官路线冻结

- 用户再次确认主线是新架构与几百B/T级donor切片，不用小模型替代目标，也不做大规模蒸馏、
  LoRA或epoch训练。新里程碑名为`Polaris Meridian / 北极星经络`：巨型器官库允许300B+总容量，
  但本机每token实际活跃量必须约3--5B，并守住20 token/s。
- K3官方索引审计复算2.8T checkpoint净载荷`1,560,860,324,864`字节。92层×top-16每token
  触及1,472个17,547,264字节专家页，专家载荷25.83GB；非routed文本骨架本身113.51GB。
  原生K3在RX 5700 XT上20--50 token/s被容量和带宽下界否决，但82,432个专家页均可精确Range。
- 新增统一Safetensors/GGUF页目录、按字节LRU、4-worker预取与single-flight。真实v6 GGUF建立
  10,240页，64页/68.50MiB首轮读取207.39MiB/s，热命中约390ns/get，抽样SHA全部一致。
  K3原生路径若要20 token/s需约99.958%页命中；分页只能解决容量，不能解决104B active计算。
- 对四颗真实K3专家完成五组`48×64`连续神经元微块扫描。top-4激活神谕输出cosine仅
  `0.400--0.493`，top-8为`0.539--0.617`，top-16为`0.709--0.764`；真实输出贡献神谕没有
  明显改善，低秩解析路由更差。结论：连续编号不是能力经络，停止机械微块进入C++。
- DeepSeek-V4-Flash-0731（实计304.18B/13B active）和GLM-5.2（744B/40B active）均可做
  精确tensor Range，但单专家依赖原生hidden/router/shared/state/上下层，物理可切不等于能力可搬。
  单专家今后只称“页”；能力器官必须是连续层段+原生状态ABI+成对激活门+归因/缓存合同。
- 首个300B+可证伪对象定为DeepSeek V4 L40--L42连续末端皮层。先在临时云端做供体原生逐层
  流式前向，只回传配对hidden、mHC、route trace和NLL，再用nullspace-anchored矩形CCA/
  Procrustes闭式构造入口/出口门；原生trace工具未就绪前不下载整层或DSpark。
- 正式术语统一为“多能力总线 + 候选/验证调度”：规划、知识、工具、代码、校验、视觉/UI、
  停止/记忆、对话是可合并/删除的能力槽，不是固定玄学结构。低占空比调度为难度检测→能力
  路由→候选生成→原生验证→提交与缓存。所有晋级仍只看留出激活、反事实NLL、完整任务和
  实际字节。完整契约见
  `fast16/research/polaris_meridian_v0/MERIDIAN_ARCHITECTURE.md`。

### 2026-08-01 北极星原生稀深旗舰 S14

- 在任何真实质量结果产生前完成七路线物理/函数边界审计。L40--L42跨模型portal降为状态采集与
  连续皮层探针，不再预设为质量主脑；当前唯一先证伪的质量主脑改为
  `Polaris Native Sparse-Depth S14`。该变更来自坐标充分性审计，不是看完blind后的调参。
- S14直接使用固定revision的304.18B DeepSeek-V4-Flash-0731原生tokenizer、embedding、4096
  hidden、mHC、attention、router、MoE、norm与lm head；只执行预注册的
  `[0,1,2,6,7,14,15,22,23,30,31,40,41,42]`，其余residual block为identity。它不是4.59B
  小模型替换主线，而是304B donor的稀深执行图。
- 官方索引静态复算：完整本地切片`52,231,273,716 B`（48.644GiB），每token活跃参数粗估
  `4.5897B`，活跃权重上界`4.3795GB/token`。RX 5700 XT在30%理论带宽效率下权重扫描上限
  `30.69 token/s`；这只证明20 token/s存在物理空间，不是端到端测速。随机多span SSD下要守住
  20 token/s仍需约99.03%专家页命中，质量与分页局部性均未验证。
- S14质量状态固定为`physical_budget_pass=true, quality_pass=null`。最短门是四题原生早停；
  低于3/4、重复/乱码、原生state不闭合或指令失败立即停止，不扫描层数和residual scale挽救。
  通过后才跑冻结八维16题和本机真实20 token/s门。
- DeepSeek原生状态采集已实现固定revision元数据、26块/token CNOB、原子提交、NPU/CPU doctor、
  adapter证明与合成自检；合成6/6通过，但真实Ascend/native forward尚未发生。完整base forward
  文件并集约156.02GB，官方CUDA/FP4语义不能因`torch_npu`矩阵乘可用而假定已兼容。
- S14精确Range打包器已实现header-only抓取、14层route-trace强制覆盖、非专家/命中专家tensor
  选择、1GiB分段、Range哈希锁、原子恢复日志与显式execute门；离线自检6/6通过且未访问网络。
  50GiB GitCode overlay对52.231GB shard上界只余约1.356GiB，低于2GiB安全余量，故上界包拒绝
  直接落盘；必须先取得原生trace缩小精确包，或采用可校验外部盘/逐层RAM消费。
- Ascend S14已新增严格单层`load→execute→free`适配器骨架、设备doctor和10项原生语义支持矩阵；
  合成生命周期自检6/6通过，14层最大同时在生层数为1，异常加载路径也会释放。但MXFP4/UE8M0、
  FP8 attention、mHC、CSA/HCA和官方图移植仍未完成，因此`native_forward_ready=false`，不得把
  `torch_npu`可见或矩阵乘成功写成DeepSeek已经能跑。
- Kimi K3若指网页截图理解，可独立研究MoonViT-V2视觉塔；若指HTML/CSS/JS与工具策略，能力分布
  在多层文本电路和后训练策略中。下一步必须在预冻结前端题上采原生router/NLL，定位跨任务重复
  胜出的多层“专家社区”，禁止猜一颗“前端专家”。八卦/五行仅启发生成、抑制和循环拓扑，正式
  晋级仍只认hidden、NLL、完整任务、字节和速度。
- K3前端社区聚合器已冻结24题来源与revision，要求每个关键token记录原生top-16路由和逐专家
  leave-one-out NLL；只有跨任务反事实正收益节点才能组成跨层社区。合成自检覆盖严格输入门、
  社区聚合和精确Range dry-run，未运行K3、未下载权重，输出默认`download_authorized=false`。
- 巨型页运行时已修复三项正确性风险：staging按最大页动态分配且尺寸硬校验；batch miss按
  pipeline depth分波；VRAM槽使用`Loading/Ready`生命周期，fence完成前lookup不可见且Loading
  不参与淘汰。Rust库check通过、first_token_235b示例check通过、纯逻辑测试2/2通过；尚缺真实
  Vulkan/GPU集成smoke，旧refactored示例和test_streaming仍有本轮之前的独立编译缺口。
- 设计与证据入口：`fast16/research/polaris_meridian_v1/MERIDIAN_BUS_ARCHITECTURE.md`、
  `fast16/research/polaris_meridian_v1/quality_architecture/ARCHITECTURE.md`、
  `fast16/research/polaris_meridian_v1/deepseek_stream/README.md`、
  `fast16/research/polaris_meridian_v1/ascend_s14/README.md`、
  `fast16/research/polaris_meridian_v1/k3_frontend_community/README.md`。

### 2026-08-01 北极星 S14 全本地执行链第一波落地

- 主线已合入三项实现提交：`38df410` 本地 S14 数值参考内核、`4422329` route-first 动态
  Range 状态机、`936ad78` capability-gated Rust Runner 与 Vulkan 分页接线。它们把
  `header/catalog → 当前层 router → top-k → 精确专家 Range → Loading/Ready → layer commit`
  的安全边界落成了代码，但当前 capability gate 仍会拒绝真实 native forward，不会吐假 token。
- CPU/PyTorch 参考语义覆盖 packed I8(E2M1×2)、UE8M0、FP4 分块 linear、FP8 E4M3+E8M0、
  mHC/Sinkhorn 与 sparse attention。针对性测试 `15/15` 通过，并实际读取本机 L42/E0 与
  L42 FP8/HC ABI 样本复核；这些结果证明真实字节格式和参考数值路径，不证明完整 S14 质量。
- 动态 Range 状态机的离线 fake HTTPS 测试 `6/6` 通过，旧静态 Range 回归仍为 `6/6`。
  已从固定 revision 的全部45个 base shard header（67,612 tensors）生成真实只读 catalog：
  `D:/models/Polaris-S14/route_first_catalog.json`，14,547,734 B，22,010条 Range，SHA-256
  `61a0ddd24a1f83f698049feff976f43ba4fc43b31b9513f937602132ad44bbb0`；其中506条为 S14
  prerequisite Range。`download_authorized=false`，本轮没有下载大权重。
- Rust Runner 冻结且只允许两套预注册图：首测 `S14/top-6`，质量失败备选
  `FullDepth43/top-1`。Rust状态机测试 `14/14`、Python互操作自检通过，`ssd_inference --lib`
  编译通过；Runner明确记录4路HC、KV/compressor/indexer状态、29个identity层、贪心head契约
  与真实测速计数器，缺少原生算子或实机数值证据时硬拒绝。
- 第二波已并行启动：RX 5700 XT packed FP4/FP8 Vulkan 数值核、Python Range→Rust Runner
  运行时桥、hash/score router + compressor/indexer/head 原生参考算子。下一真实里程碑仍是
  “单层真实 weight forward”，不是继续写架构文档，也不是先下载完整4.0GiB S14骨架。

### 2026-08-01 北极星 S14 真实单层、最终头与 Range→Runner 闭环

- 真实 L42 单层单 token 已完成并固化为可重复脚本：读取固定 revision 的76个 Range payload、
  共247,515,224字节，执行HC、FP8稀疏注意力、原生top-6 router、六个FP4 routed experts、
  FP8 shared expert与第二次HC；五个F32指纹逐项复现，负向拒绝4/4通过。该结果是完整真实单层，
  仍不是S14首token或质量证明。
- 旧 catalog 漏掉最终 `hc_head_base/fn/scale`，现已修正为22,013条Range、509条prerequisite；
  外部 TOFU skeleton 为4,313,107,428字节，正式 catalog 已通过固定header、metadata与skeleton
  三方一致性校验。旧 catalog/skeleton 均保留了 pre-HC-head 备份。
- 已用严格HTTPS 206与精确Content-Range取得真实最终边界：HC head+norm共270,356字节，原生
  BF16 `head.weight` 1,059,061,760字节；全词表129,280维真实权重投影在CPU参考路径耗时
  0.305秒。此次输入为合成四路hidden，只证明最终权重/公式闭环，不是模型输出质量。
- Range状态机新增按token派生8,192字节原生embedding行与按词表行派生head块，避免单token
  读取完整1.06GB embedding。真实L0离线worker smoke返回24个已校验artifact、118,437,720
  字节，全部cache hit，耗时0.798秒，且abort清理成功、未下载任何新页。
- Rust Runner已接入持久UTF-8 JSONL Range桥；真实input token ID会传到Python，executor只在
  已验证embedding row ready后运行，最后HC/norm/head artifacts也在Runner做argmax前交给
  executor。Rust原测试14/14、桥测试7/7、Python worker 3/3、Range测试7/7及Clippy均通过。
- 当前唯一主阻塞缩小为：把已验证的单层算子泛化到预注册14层、逐层用真实router取84个命中
  expert页并串到已完成的最终头，生成第一个真实S14 token。完成前不得宣称S14可用、20 token/s
  或质量接近Claude/GPT。

### 2026-08-01 北极星 S14 首个真实 token 与 RX 5700 XT 专家整链

- 固定 S14 的首个真实 token 已完整生成：BOS embedding 依次通过
  `[0,1,2,6,7,14,15,22,23,30,31,40,41,42]`，每层执行原生 attention、router、
  top-6 routed experts、shared expert 与两段 HC，最后进入真实 HC head、norm 和 BF16 全词表
  head。14/14 层完成，argmax token ID 为 `108967`，解码为 ` Compression`。
- 该次 correctness 运行总耗时 `714.7690811s`，精确 routed 下载 `962,592,768 B`，发生7次
  可恢复镜像断连；没有 SHA、shape、dtype、路由、状态或算子错误。最终 logits 形状
  `[1,129280]`，F32 little-endian SHA-256 为
  `f414aef5894fe66d609d2650bf9b64510b3fa30ad76514180eff007c5853c3c4`。仓库内固化报告位于
  `fast16/research/polaris_meridian_v1/s14_first_real_token/FIRST_TOKEN_REAL_REPORT.json`。
- 首 token 只证明真实图闭环，不证明语言质量或速度；`Compression` 单 token 不能作为能力结果。
  当前下一主阻塞已变为 token1+：必须保留每层 window KV、HC、ratio4/128 compressor/indexer
  remainder，正确执行 position RoPE，并按当前 token 读取 L0--L2 `tid2eid`，才能形成连续输出。
- `NativeS14Executor` 持久子进程桥已合入：JSONL 只传控制信息，BF16 hidden/state 与 F32 logits
  走桥拥有的二进制 arena；超时、descriptor/epoch/shape 漂移与非有限值全部 poison。Rust库测试
  20/20、Range桥7/7及Clippy `-D warnings`通过。
- RX 5700 XT 已真实执行 L42/E126 的 GPU-resident
  `w1/w3 -> limit-10 SwiGLU -> w2 -> route-weight mix`：平均 `0.157038 ms`，对CPU参考
  max abs `1.072883606e-6`、RMSE `1.483723944e-7`；真实 FP8 `wq_a` 平均 `0.0838004 ms`。
  这证明 packed 核与最小专家整链 parity，不代表完整层或整模型 token/s。下一步扩展为 top-6
  加 shared 的批量租约、fence publication、generation-safe eviction 与 GPU accumulator。

### 2026-08-01 北极星连续 token、GPU top-6/shared 与 Exact Cascade 决策

- 同一 `DecoderRuntime` 已真实完成连续两个 S14 token。position0 为
  `0 -> 108967 (" Compression")`；position1 消费 token `108967`，输出
  `53 ("S")`。position1 通过14/14层和真实 final head 后，14层 window KV、HC、
  ratio4/128 compressor/indexer remainder 才一次性提交；`committed_tokens=2`、`error=null`。
- position1 normalized SHA-256 为
  `243451bf535ee60e67e0ff89031abb3008266f0f116b084f6f9ed88322f71465`，logits SHA-256 为
  `46b95489427932a0d5acfacd5ee6bc9ceac495df3daed5a6a58681a0d95a141d`。完整外部报告为
  `D:/models/Polaris-S14/s14_two_real_tokens_report.json`，251,656字节，SHA-256
  `5ce3d5bcf1f1ad788659487cd070005078787736d104cba39fcb71108a25abe8`；仓内冻结摘要为
  `fast16/research/polaris_meridian_v1/s14_first_real_token/TWO_TOKEN_REAL_REPORT.json`。
- 本次新增下载 `1,016,078,336 B`，总 correctness 耗时 `1005.144s`，其中 position1
  `951.225s`。该耗时包含慢速精确 Range 与 CPU reference，只作为正确性记录。
- 两个相邻 token 的同层 top-6 交集总计仅 `8/84=9.5238%`：L2/L15/L22/L30各复用1个，
  L41/L42各复用2个，其余层0。单样本不能外推长序列稳态，但足以否决“仅缓存上一 token
  专家即可获得高命中”的假设；速度路径必须研究更大的跨 token 工作集、预测预取和批量验证。
- RX 5700 XT 已真实完成同一 L42 输入的六路 routed expert 顺序计算、accumulator 清零、
  route-weight 累加以及 shared FP8 累加，共35个 dispatch，GPU时间 `1.3696000 ms`；对CPU参考
  max abs `1.096725464e-5`、RMSE `1.426471034e-6`。`VramPool` 已加入 generation/pin，
  Loading与pinned页不可淘汰，双池批量在全部preflight成功前不可发布，失败全回滚，compute
  fence释放前slot不可复用。该结果仍不是完整层、完整token或端到端token/s。
- 质量路线正式修正为 **Polaris Exact Cascade**：完整43层量化 DeepSeek-V4-Flash 的
  `FullDepth43/native-top6` 是唯一允许提交 token 的裁决器；S14、v38、v47只作为多token草稿、
  专项候选和专家预取来源。最终验证器不得自适应跳层，也不得把 v38/v47 hidden 直接注入
  DeepSeek坐标；候选必须重编码后执行标准最长一致前缀接受/回退与完整状态提交。
- 现有 Rust Runner 中的 `full_depth_top1` 只是历史容量备选且能力清单仍为 hard reject；它会丢弃
  原生 top-6 中五路专家，不能作为 Exact Cascade 的质量路径。生产合同必须迁移为
  `FullDepth43/native-top6`，在真实算子、K=1对齐和K=4/8状态等价完成前继续拒绝执行。
- 纠正理由：S14固定跳过29/43层且没有跳层训练，attention、mHC、compressor、router和head的
  输入分布均会漂移；`4.59B active`不能类比训练完善的4.59B模型。继续向S14堆器官不能可靠
  把质量上限推到完整V4，更不能据此宣称追平Claude/GPT。
- 下一硬门顺序：官方消息编码到任意 token 序列 forced-prefill；FullDepth43 K=1 对固定参考
  逐token对齐；K=4/8 causal-block与K=1输出/回滚等价；S14 K=8相对FullDepth平均连续接受
  至少4 token且aggregate agreement不低于85%；最终端到端p50不低于20 token/s、p95不低于
  12 token/s、VRAM低于7.7GiB、RAM低于30GiB。所有门均为待验证，不能提前写成已完成。

### 2026-08-01 官方 forced-prefill、FullDepth43 合同与速度硬边界

- 官方 DeepSeek-V4 消息、reasoning effort 与 DSML 工具协议已经编译为确定 token 队列。正式
  tokenizer 固定 SHA-256
  `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`、词表129,280、BOS 0
  和8个关键协议token ID；冻结工具输入产生370个token。forced-prefill测试7/7、11个子测试、
  官方编码fixture 4/4和Windows CRLF语义自检通过。当前只完成编码，没有执行这370个模型位置。
- Rust Runner 已把历史 FullDepth/top1 从生产图枚举移除，新增
  `FullDepth43/native-top6` Exact Cascade 合同：K只允许1/4/8，每个位置必须给出43层、每层6个
  不同 routed expert 及attention/shared/mHC/KV/compressor/indexer状态证明；最长一致前缀、
  原生fallback、checkpoint原子提交和失败回滚均有测试。当前 capability 仍因真实后端缺失而
  在产生token前硬拒绝，不能把合同测试写成FullDepth已运行。
- 合并后 Rust 单元测试30/30、Range桥集成7/7、Clippy `-D warnings`、Python共享合同自检以及
  FullDepth/roofline Python测试39/39均通过。
- RX 5700 XT 已知S14必要GPU包络为20.3476056ms/token（仍遗漏多个阶段）。20 token/s在此
  乐观下界下至少需要30.0948%专家页命中；50 token/s即使100%命中也已被当前单token内核
  否决。FullDepth43按同一L42锚点投影，每个verified position已有62.4962172ms已知工作；全接受
  达20/50 token/s仍分别至少需要1.2499243x/3.1248109x吞吐提升，并且不能依靠K值本身自动获得。
- 下一实现断点：position2+与ratio4/128首次压缩边界、forced token队列原子消费、真实
  FullDepth43 K=1算子/权重入口和固定参考逐token对齐。K=1未运行前没有追上Claude/GPT的质量证据。
