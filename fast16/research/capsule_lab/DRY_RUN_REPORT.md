# ColorLM 通用胶囊实验室 Dry-run 报告

日期：2026-07-31  
状态：`DRY_RUN_ONLY / WAITING_FOR_MAINLINE_APPROVAL`

## 1. 边界与结论

本轮只读取了目录元数据、JSON 配置/索引、Safetensors 头部和 GGUF 头部。没有启动或停止模型，
没有访问 GPU，没有下载权重，没有执行张量提取，也没有计算 10--22 GB 模型文件的全文件哈希。

结论：五类胶囊都能定义为统一的、可流式验证的包，但当前只有 Qwen3-Coder-Next 的末端块、
共享/路由专家和 Gated DeltaNet（本报告工作名为 DSpark）具备现成的精确头部证据与坐标桥。
GLM 的张量边界清楚，但直接共享专家替换已有失败证据；在形成 GLM 深层坐标投影前，不应批准
GLM 胶囊接入。输出头还受到不同 tokenizer 的严格约束，不允许直接按 token id 混合 logits。

## 2. 资产审计

### 2.1 原始供体

| 供体 | 本地状态 | 架构元数据 | 证据与完整性状态 |
|---|---:|---|---|
| Qwen/Qwen3-Coder-Next | 非完整模型；有 6,313,251 B 索引、config、tokenizer、5 个 shard header、完整 shard 40 | 48 层，hidden 2048，512 专家，top-10，专家宽 512；索引声明 159,348,782,592 B、74,391 张量、40 shards | index SHA-256 `4a23cf5b586da565eee0032cbad4aeb088dbbb7593b62ffeb1b8e0399f1d0d7e`；config SHA-256 `a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8`；未重哈希 3.37 GB shard |
| `GLM-4.7-Flash-Q5_K_M.gguf` | 完整本地文件，21,408,850,272 B | GGUF `deepseek2`，47 层，hidden 2048，64 专家，top-4，专家宽 1536，844 张量 | 只核对 GGUF 头与文件长度；原始文件 SHA-256 在已审计 sidecar 中缺失，本轮未做重 I/O 补算 |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 完整本地文件，22,134,528,992 B | GGUF `qwen35moe`，40 层，hidden 2048，256 专家，top-8，专家宽 512，733 张量 | 头部可见全部张量；NAL sidecar 有逐张量/manifest 哈希，但原始 GGUF 全文件 SHA-256 未在本轮重算 |

### 2.2 已有 donor 衍生权重

| 资产 | 字节 | 已有 SHA-256 / 结论 |
|---|---:|---|
| Coder L44 Q4_0 block | 944,203,648 | `fc98e4b21767de4db6b840adf156f0e3dbf11da3af461f0d09e1d7d95117c7fd`（manifest 记录，本轮未重算 payload） |
| Coder L45 Q4_0 block | 944,203,648 | `19d10f0efd195c3636d8ec8dae2365bbadfdb0b4ac21984e1121cd639d56df5d`（同上） |
| Coder L46 Q4_0 block | 944,203,648 | `253551de42d376c277b1f859012c55db9bfcae01db1a213d1ba2fa0c335d979a`（同上） |
| Coder L47 Q4_0 block | 940,461,184 | `eda3d32940538fbc135f7fa44ba2d39b1ee475d6b998d0aea81600515c2a188e`（同上） |
| Coder -> ColorLM 正交桥 F32 NPY | 16,777,344 | `7209781fa72220c337314e8456497c83cf1b51798d2b4321414200a98de04d12`（已有 manifest 记录） |
| Coder L47/E0 v2 Q4_0 胶囊 | 3 x 589,824 B + 8,192 B router | 三矩阵及 router 均有逐文件 SHA-256；已运输并把桥折叠进权重 |
| Coder L47/E471 v2 Q4_0 胶囊 | 3 x 589,824 B + 8,192 B router | 三矩阵及 router 均有逐文件 SHA-256；已运输并把桥折叠进权重 |
| `ColorLM-v5-GLM-SynapticGraft.gguf` | 13,731,822,240 | 40 层各保留 480 个 ColorLM 神经元并嫁接 32 个 GLM 神经元；sidecar 没有成品全文件哈希 |
| `ColorLM-v6-Q3Router-Fused-A1.gguf` | 13,731,822,496 | 已验证全文件 SHA-256 `bc94febd230212cd157000777240dd2562b9a2561b5d28e861f543c2e74cb2e5`；内含 GLM graft 与 40 层 Qwen3.6 router delta |
| `ColorLM-v7-CoderNext-Biopsy-E1.gguf` | 13,897,497,792 | sidecar 记录 SHA-256 `27ad28319057ce9a9edb32b8a2ed460028c8bec8b74c09883aab8e8fc8b09d43` |
| `ColorLM-v8-CoderNext-Transport-E471.gguf` | 13,897,498,016 | sidecar 记录 SHA-256 `a4d7dc7cf59071d8b7ea373babe31a5d14045557ed2fddd6a6030b25f524b0da` |

“manifest 记录”不等于本轮重新读取 payload 校验。批准正式提取时必须重算目标 capsule 的每个
payload 哈希；不得把历史记录直接提升为本次 `verified`。

## 3. 五类胶囊设计

精确张量名、形状、来源层和字节数见 `TENSOR_CATALOG.md`。统一边界如下。

| 类型 | 最小自包含边界 | 首选候选 | 当前判定 |
|---|---|---|---|
| 输出头 `output_head` | final norm、输出矩阵、token map、logit 合并规则 | Coder `lm_head.weight`；GLM `output.weight` | 可提取，暂不可直接接入；先批准 token 映射策略 |
| 末端层 `terminal_layer` | input norm、Attention/DeltaNet、post norm、shared expert、router、routed expert bank、状态 ABI | Coder L47；GLM L46；Qwen3.6 L39 | Coder L47 已有可复用 Q4 包；GLM/Qwen3.6 可由 GGUF 头精确切片 |
| 共享专家 `shared_expert` | gate/up/down；供体有额外 shared gate 时一并包含 | Coder L47、GLM L46、Qwen3.6 L39 | Coder 可用；GLM 必须先运输与能量校准 |
| 路由专家 `routed_expert` | router（或单行 router）、校正 bias、专家 gate/up/down、top-k 语义 | Coder L47/E0、E471；GLM L46；Qwen3.6 L39 | 单专家按页流式提取可行；不得把 miss 映射到错误专家 |
| DSpark `dspark` | input norm + 一层 Gated DeltaNet 核 + recurrent/conv state ABI | Coder L44、L45、L46 | 可行；`DSpark` 是本实验室工作名，上游与仓库正式名均为 `gated_deltanet` |

仓库中没有 `DSpark` 字面定义。因此契约内 `capsule_type` 使用 `dspark`，但 `operator` 必须写
`qwen3_next.gated_deltanet`，不能让工作名掩盖真实算子。

## 4. 流式提取方案

### 4.1 审批前阶段（本轮停在这里）

1. 只解析 config、索引和容器头，建立 `source_name -> shard/offset/length/shape/dtype` 表。
2. 对每个计划张量检查：名称唯一、shape/dtype 精确匹配、区间不重叠、区间不越过文件长度。
3. 输出 `capsule.json` 的 `status=dry_run`、`approval.approved=false`；所有 payload SHA-256 为
   `null` 且 verification 为 `pending`。
4. 不创建 `.bin/.npy/.gguf`，不预分配大文件，不访问网络。

### 4.2 批准后的执行算法

1. 锁定不可变来源：远端必须用 commit revision + ETag/Last-Modified；本地必须记录绝对路径、
   文件长度和已批准的 source digest 策略。
2. Safetensors 使用 `payload_base = 8 + header_len`；GGUF 使用头部对齐后的 `data_offset`。
   禁止 `from_pretrained`、`state_dict`、整 shard `np.fromfile` 或整模型 mmap 张量视图。
3. 按源文件和连续区间排序。每次只打开一个 shard，`seek` 到绝对偏移，以 8 MiB 块读入；
   同时更新 source-range SHA-256。需要量化/转置时使用量化块对齐 tile，建议 source 8 MiB +
   destination 8 MiB + codec scratch，峰值工作集硬上限 32 MiB。
4. 每个输出先写 `*.part`，边写边计算 payload SHA-256；长度和哈希通过后再原子重命名。
   中断只保留 `.part` 和带 source identity 的 resume receipt，禁止将半成品列为 capsule payload。
5. 逐 payload 复读校验后生成最终 `capsule.json`。其原始 UTF-8/LF 字节哈希写在
   `capsule.json.sha256`，避免在 manifest 内产生自引用哈希。
6. `content_root_sha256` 对按 UTF-8 字节序排序的 `path NUL bytes NUL sha256 LF` 清单求哈希；
   manifest 自身不进入 content root。任何 payload 变化都必须改变 content root。

### 4.3 已知连续区间

- Coder L47 BF16：shard 39 的 `3,458,776,392..3,999,854,920`（541,078,528 B）和
  shard 40 的 `622,493,616..3,365,568,432`（2,743,074,816 B），合计 3,284,153,344 B。
- Coder 输出矩阵：shard 40 `163,760..622,493,616`（622,329,856 B）；final norm 位于
  `3,365,568,432..3,365,572,528`，必须作为第二个 range，不能把中间 2.74 GB 一起读取。
- GLM L46：GGUF 绝对区间 `20,941,466,720..21,408,850,272`，467,383,552 B。
- Qwen3.6 L39：GGUF 绝对区间 `21,577,905,120..22,134,528,992`，556,623,872 B。

## 5. `capsule.json` 与 SHA-256 契约

机器契约见 `capsule.schema.json`，未提取实例见 `capsule.example.json`。关键规则：

- `capsule_id + capsule_type + source revision + tensor source_name` 共同确定语义身份；文件名不是身份。
- `source_shape` 表示供体逻辑形状；`storage_shape` 表示 GGML/GGUF 存储维序，二者不得混用。
- 每个 payload 必须有精确 bytes 和 SHA-256；每个 tensor 必须能追溯到 source shard/range。
- `recorded` 只表示继承历史 manifest；只有本次逐字节复读成功才能写 `verified`。
- transport、token map、量化 codec 和 runtime ABI 都是显式依赖，必须以 SHA-256 固定版本。
- `status=verified` 前，schema 要求 payload 哈希存在；`approval.approved=false` 时提取器必须拒绝写载荷。

## 6. ColorLM 接入成本

### 6.1 坐标投影

三个模型 hidden width 都是 2048，但同宽不代表同坐标。

- 全秩单向 F16 `2048 x 2048`：8,388,608 B、4,194,304 MAC，约 8.39 MFLOP/token。
- 独立入口+出口：16 MiB、约 16.78 MFLOP/token；若严格正交且运行时支持转置复用，可只存
  一张 8 MiB 矩阵。
- rank-256 双因子单向：2 MiB、约 2.10 MFLOP/token；独立双向约 4 MiB、4.19 MFLOP/token，
  但不得在未过深层激活留出门前替代全秩桥。
- Coder 已有全秩桥：未见 token 检索 top-1 98.54%，但 test embedding cosine mean 仅 0.469；
  它适合作初始化/基线，不证明所有深层都已对齐。
- GLM 没有已批准深层桥。历史探针显示整块 shared expert 输出能量约为 ColorLM 的 5--28 倍、
  方向余弦约 0；直接接入应判定为阻塞项。

### 6.2 Token 映射

本轮只比较 tokenizer 中的精确 token 字符串，不读取 embedding 权重：

| 映射 | base vocab | donor id 空间 | 精确共享字符串 | 相同 id | donor 覆盖 | base 覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| ColorLM -> Coder-Next | 248,320 | 151,936（其中 267 个 id 在 tokenizer JSON 中未填充） | 131,612 | 281 | 86.62%（按完整 id 空间） | 53.00% |
| ColorLM -> GLM | 248,320 | 154,880 | 143,045 | 281 | 92.36% | 57.61% |

建议使用两个 int32 表：`base_to_donor` 约 993,280 B；Coder `donor_to_base` 607,744 B，
GLM `donor_to_base` 619,520 B。未匹配项写 `-1`，禁止映射到 `<unk>`。输出头 v1 只能在精确
一对一 token 上合并 logits，并显式 mask 其余 token；跨多个 token 的 byte 等价不是同一步
概率，必须另做 trie/DFA 或蒸馏投影，不能用字符串启发式冒充等价。

### 6.3 运行开销量级

| 胶囊 | 权重/活跃读取 | 主要每 token 计算（不含通用调度） |
|---|---:|---:|
| Coder 输出头 BF16 | 593.5 MiB；Q6_K 估算 243.43 MiB | 311,164,928 MAC，约 622.33 MFLOP；另加 token scatter/mask |
| GLM 输出头现有 Q6_K | 248.14 MiB | 317,194,240 MAC，约 634.39 MFLOP |
| Coder shared expert BF16 / 现有 Q4_0 | 6.004 MiB / 1.689 MiB | 3,145,728 MAC，约 6.29 MFLOP |
| Coder routed expert top-10（Q4_0） | 16.875 MiB + 0.563 MiB router | 约 32.51M MAC，约 65.01 MFLOP |
| GLM shared expert（现有 Q6_K） | 7.383 MiB | 9,437,184 MAC，约 18.87 MFLOP |
| GLM routed expert top-4（现有 Q5/Q6_K） | 26.344 MiB + 0.500 MiB router/bias | 约 37.88M MAC，约 75.76 MFLOP |
| Coder DSpark core（现有 Q4_0/F32） | 18.204 MiB | 约 34.24M MAC，约 68.5 MFLOP；另有 recurrent update |

DSpark 的估算 F16 状态约为每层每 sequence 1.05 MiB recurrent matrix + 48 KiB conv history；
三层约 3.14 MiB。实际布局必须由 runtime ABI 核对，不能把该估算写成 payload shape。Coder L47
完整注意力另有约 2 KiB/token 的 F16 KV（2 KV heads、head dim 256、K+V）；32K context 约
64 MiB/sequence，且 attention 计算随上下文长度线性增加。

## 7. 建议批准顺序

1. 先批准“复用并重新校验”现有 Coder L47/E471 单专家胶囊，验证统一 manifest 与 loader，
   不新增大权重。
2. 再批准 Coder L44 单层 DSpark core（不含该层 512 专家银行），验证 state ABI 与硬旁路。
3. 只有 token-map 策略获批后才提取输出头；它的运行带宽远高于单专家，不应作为第一颗胶囊。
4. GLM 先做小桥/深层激活 dry-run 计划；全共享专家、末层或路由银行保持拒绝提取状态。

当前没有任何已批准提取项。下一步只接受主线给出明确 capsule id、来源 revision、payload dtype
和 SHA 策略后，再单独生成执行计划。
