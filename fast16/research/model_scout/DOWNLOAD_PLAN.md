# ColorLM donor 精确下载清单（待批准）

**状态：`HOLD - NO WEIGHTS DOWNLOADED`**  
**截止日期：** 2026-07-31（Asia/Shanghai）

本文是审批清单，不是下载脚本。本轮只读取了 README、LICENSE、config、tokenizer、safetensors index 和 Hub 文件元数据，没有请求任何 safetensors/GGUF 权重内容。

## 放行原则

1. 权重请求必须固定到下表 commit，不使用漂移的 `main`/`master`。
2. 先对元数据、tokenizer 原始字节和目标张量形状做门禁，再放行权重。
3. 不在没有真实路由统计时指定“能力专家”；`expert 0` 只能用于 ABI smoke，不得宣称为高价值 donor。
4. 第三方 GGUF 只证明转换生态，首轮审计以官方 safetensors 和官方 index 为准。

## P0：元数据包（不含权重）

对进入 P1/P2 的仓库，只保存实际存在的下列小文件：`README.md`、`LICENSE*`、`config.json`、`generation_config.json`、`tokenizer*`、`vocab.json`、`merges.txt`、`chat_template*`、`model.safetensors.index.json` 以及官方自定义配置/建模代码。不对并不存在的文件名做猜测请求。

## P1：首个唯一建议

### Qwen3.6-35B-A3B 输出头/末层半片审计

- repo：`Qwen/Qwen3.6-35B-A3B`
- revision：`995ad96eacd98c81ed38be0c5b274b04031597b0`
- 文件：`model-00026-of-00026.safetensors`
- 字节：**2,231,416,848**
- SHA-256：`1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613`
- 内容：`lm_head.weight`、L39 router、attention/norm/shared expert、融合 `down_proj`；不含 L39 融合 `gate_up_proj`。
- 预计真正提取：`lm_head.weight` 形状 `[248320, 2048]`，BF16 净载荷 **1,017,118,720** 字节。
- 理由：ColorLM 本地基线同为 hidden `2048`、vocab `248320`，官方 tokenizer 文件与 ColorLM 所用 Qwen 248k 词表体系一致。
- 放行后仍禁止直接热替换：必须先通过全词表字节比对、坐标对齐、next-token NLL 与 `alpha=0` 精确回退门。

P1 总量：**2,231,416,848 字节**（2.231 GB / 2.078 GiB）。

## P2：P1 通过后的能力分支

### A. DeepSeek-V4-Flash-0731 L42

- repo：`deepseek-ai/DeepSeek-V4-Flash-0731`
- revision：`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 文件：`model-00044-of-00048.safetensors`
- 字节：**3,590,026,352**
- SHA-256：`422d3889fa20c238b7f97464c14df0bcf3328f189c294f41a3a334421dc560c7`
- 内容：L42 完整 1,576 张量，含 MoE、attention、mHC、norm；不含 output head 和 DSpark 三片。
- 预计小胶囊：router + 经真实路由统计选出的 top-6 + shared expert，理论净载荷 **132,645,888** 字节。
- 前置：`2048 <-> 4096` 矩形激活桥、FP4+UE8M0 解码、FFN-only 隔离岛。

### B. Kimi K3 视觉前端

- repo：`moonshotai/Kimi-K3`
- revision：`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
- `model-00096-of-000096.safetensors`：**802,448,352** 字节，SHA-256 `9d10c74fc10161bef9463a8541a634a97f521f43c99368ea7243ce0c79cdbf7c`，仅 `vision_tower.*`。
- `model-00095-of-000096.safetensors`：**92,289,328** 字节，SHA-256 `01d41139abb8cf3b5288a97318cc4ab92676671b4eb141031cd80d7c3ced6122`，仅 3 个 `mm_projector` 张量。
- 合计：**894,737,680** 字节（0.895 GB / 0.833 GiB）。
- 目标：保留 401M MoonViT-V2，重训 `1024 -> 2048` projector；不引入 K3 的 7168 文本主干。
- 前置：Kimi K3 自定义许可审查、图像 token 协议设计、computer-use 隔离验收集。

### C. Fara1.5-27B 视觉塔与 merger

- repo：`microsoft/Fara1.5-27B`
- revision：`299c8406a6c6256d45ec200d1ac12b34c5599d9b`
- 文件：`model-00010-of-00010.safetensors`
- 字节：**1,309,493,264**
- SHA-256：`207ec8c6a11973996b5270043a6c0650d495d5e59287cd5636e2fea695b197e3`
- 内容：完整视觉塔和 vision merger，另夹带 final norm 与 L63 的少量 tensor；离线提取时丢弃非视觉张量。
- 预计真正提取：约 **0.93 GB** 视觉 tensor；精确净载荷需在批准读取 safetensors header 后计算。
- 目标：MIT 许可下补截图理解、像素定位和浏览器 CUA；重训 `1152 -> 2048` 视觉桥，并单独实现截图/坐标/tool harness。
- 限定：Fara 没有独立 action head，单取视觉栈不能复制其 WebVoyager/Online-Mind2Web 动作策略。

如主线当期目标是文本 agent/tool，选 A。GUI/电脑操作优先选 C；只有 Kimi 自定义许可通过并且需要更广的办公/视频能力时才选 B。A 与一个视觉分支可以分阶段推进；B/C 不建议同时放行，以免无法归因。

## P3：仅在 P1/P2 能力门通过后

### GLM-4.7-Flash L46

- repo：`zai-org/GLM-4.7-Flash`
- revision：`7dd20894a642a0aa287e9827cb1a1f7f91386b67`
- 文件：`model-00047-of-00048.safetensors`
- 字节：**2,539,429,936**
- SHA-256：`1bcc5d06065d2a564894657945ccfe9411762421c2c60acf91de31050cd4d84d`
- 内容：`lm_head`、完整 L46、64 个独立 experts、router、shared expert、attention、norm。
- 实体单专家 BF16 约 **18.9 MB**；但 GLM tokenizer/vocab 不同，首次只做 residual/expert 隔离岛。

### Qwen-AgentWorld-35B-A3B 环境模拟辅助支路

- repo：`Qwen/Qwen-AgentWorld-35B-A3B`
- revision：`60d2b0434a53d2e62a7c00a489586815d94ebffb`
- 文件：`model-00021-of-00021.safetensors`
- 字节：**3,889,712,984**
- SHA-256：`e6379e7108900493e234856276c32250c113e4fc461511f72d6b1015441e6057`
- 内容：完整 L39、router、融合 experts、final norm 与 `lm_head`。
- 限定：它的训练目标是根据 action/history 预测 environment observation，只能作 simulator/critic/auxiliary head，禁止直接替代 action policy。

P1 + P2(A) + P2(B) + P2(C) + P3 全部文件合计 **14,454,817,064 字节**（14.455 GB / 13.462 GiB）。这个合计只用于上限预算，**不是建议一次性下载**。

## Range-only 候选：禁止整片

### Step-3.7-Flash

- repo：`stepfun-ai/Step-3.7-Flash`
- revision：`5f6244077ac62e04eec3f320501ff8c2b293373a`
- L44 整片 `model-00023.safetensors`：**9,245,052,456** 字节，SHA-256 `05c2c2a08df421f617794e137429246a6ea60dd908fc691263242a12325dae7f`；不建议整片。
- BF16 单专家净载荷 **31,457,280** 字节（`4096/1280`三矩阵）；须另取 FP32 router 和 shared expert，并训练 `2048 <-> 4096` 桥。
- 放行条件：先完成 4096 维矩形桥的无权重设计评审，再只读 header 生成 router/shared/selected-expert 的 Range。

## 明确不下载

- DeepSeek `model-00045` output head 和 `00046..00048` DSpark：词表/隐藏宽不兼容，DSpark 还需三目标层状态契约。
- Kimi K3 文本 L92/embedding/head：7168/3584 坐标、KDA/MLA/AttnRes/SiTU 与自定义许可尚未通过。
- GLM-5.2 任何权重：先完成 DSA/IndexShare 状态接口和 `6144 -> 2048` 桥评审。
- MiniMax-M3 权重：MiniMax Community License 不是 MIT/Apache/OSI 宽松许可，商业使用带展示/通知/收入阈值授权和禁止用途条款。
- Qwen3.6-27B、Qwen3.5 通用系列、GLM-5 原版、DeepSeek V4 preview 及 preview GGUF：已有更合适的新版或同形供体。
- 任何完整官方 checkpoint、完整社区 GGUF，以及所有未固定 revision 的权重 URL。
