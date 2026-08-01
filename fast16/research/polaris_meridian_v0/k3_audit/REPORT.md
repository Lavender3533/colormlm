# Kimi K3 全深度稀疏经络元数据审计

**审计日期：** 2026-08-01
**固定版本：** `moonshotai/Kimi-K3@9f62e4e9fffbd0a83ddd60e1c209d828994b3569`
**执行边界：** 只读取官方 config、自定义代码、59,764,096 字节的 safetensors index、Hub 文件元数据，以及代表分片的 safetensors header byte-range。没有下载任何权重 payload，没有启停模型、GPU 或 CMake。

## 结论

K3 的每个 routed expert 确实是可单独 Range 提取的物理页，一页精确为 **17,547,264 字节（16.734375 MiB）**。但“70GB 内存放 K3 原生骨架，再按 top-16 加载专家”不成立：

- 不含 routed experts、不含视觉栈的原生文本骨架已有 **113,509,540,864 字节（113.510 GB / 105.714 GiB）**；
- 92 个 MoE 层在单 token 上各取 top-16，共需引用 **1,472 个 layer-expert 页**，专家权重是 **25,829,572,608 字节（25.830 GB / 24.056 GiB）**；
- 70GB 若全部专用于专家缓存，只能放 **3,989 页**，平均每个 MoE 层 **43.36 个专家**，即全专家库的 **4.84%**；
- 如果相邻 token 的路由集合完全不重合，70GB 只够 **2.71 条 token 路径**，第 3 个 token 会超过缓存。

因此，**Range 切片工程可行，但在当前 70GB 容量与 20–50 tok/s 目标下，K3 原生全深度前向不可作为北极星在线主干**。它可以作为“路由轨迹/能力经络的离线供体”，但必须先证明前端任务的路由命中在每层高度集中；不能在没有原生路由 trace 时猜“哪些专家懂前端”。

## 1. 官方结构与完整性

| 项目 | 官方元数据 |
|---|---:|
| 总参数 / 激活参数 | 2.8T / 104B |
| 文本层数 | 93 |
| 第 0 层 | dense FFN + KDA |
| MoE 层 | 92 |
| KDA / Gated MLA | 69 / 24（MoE 层中分别为 68 / 24） |
| hidden / LatentMoE hidden | 7168 / 3584 |
| 每层 routed experts / top-k | 896 / 16 |
| 每层 shared experts | 2 |
| 专家中间维 | 3072 |
| 上下文 | 1,048,576 |
| routed expert 量化 | MXFP4 packed weight + group-size 32 U8 scale |

官方 index 逐键审计结果：92 个 MoE 层每层都有 `896 × 6 = 5,376` 个 expert tensor，总计 **494,592** 个；每层的专家完整落在单独分片中。六种 tensor 为 `w1/w2/w3 × weight_packed/weight_scale`。

下面的体积复算与官方 index `metadata.total_size = 1,560,860,324,864` **逐字节相等**：

```text
1,446,456,066,048  92 层 routed expert bank
  113,509,540,864  文本非 routed-expert 骨架
      894,717,952  vision tower + mm projector 净载荷
-----------------
1,560,860,324,864  官方 index 总净载荷
```

## 2. 非 routed-expert 原生骨架

这里“骨架”包括 embedding/head、dense L0、attention/KDA/MLA、AttnRes、norm、router、7168↔3584→7168 LatentMoE 投影和 shared experts；只排除 92 层的 896 个 routed experts，且不计视觉栈。

| 骨架部分 | 数量 | 单位字节 | 合计字节 |
|---|---:|---:|---:|
| Dense L0（KDA + 33792 FFN + AttnRes/norm） | 1 | 2,341,213,184 | 2,341,213,184 |
| KDA MoE 层非 routed 部分 | 68 | 1,267,744,256 | 86,206,609,408 |
| Gated MLA MoE 层非 routed 部分 | 24 | 844,335,616 | 20,264,054,784 |
| embedding + lm_head + final/output AttnRes norms | 1 | 4,697,663,488 | 4,697,663,488 |
| **合计** |  |  | **113,509,540,864** |

每个 MoE 层中固定存在：

- router weight + correction bias：12,848,640 字节；
- LatentMoE down/up projection + norm：102,767,616 字节；
- 2 个 BF16 shared experts：264,241,152 字节；
- attention + AttnRes + layer norm 等：KDA 层 887,886,848 字节，MLA 层 464,478,208 字节。

这是当前 checkpoint 的真实存储，不是参数量估算。config 明确将 `self_attn` / `shared_experts` / dense MLP / lm_head 等排除在 MXFP4 之外，它们在 header 中为 BF16/F32。

## 3. 单专家页和 HTTP Range

对 L92 的 `model-00093-of-000096.safetensors` 进行了真实 header Range 审计：

- 文件大小：16,567,507,176 字节；
- CDN `Accept-Ranges: bytes`；
- header 长度：823,008 字节，data 起点是文件偏移 823,016；
- 896 个专家尺寸全部相同；一个专家内的 6 个 tensor **无缝连续**；专家与专家之间也无 gap。

| tensor | dtype / shape | 字节 |
|---|---|---:|
| `w1.weight_packed` | U8 `[3072,1792]` | 5,505,024 |
| `w1.weight_scale` | U8 `[3072,112]` | 344,064 |
| `w2.weight_packed` | U8 `[3584,1536]` | 5,505,024 |
| `w2.weight_scale` | U8 `[3584,96]` | 344,064 |
| `w3.weight_packed` | U8 `[3072,1792]` | 5,505,024 |
| `w3.weight_scale` | U8 `[3072,112]` | 344,064 |
| **单 expert page** |  | **17,547,264** |

L92 expert 0 是一段 HTTP Range：文件字节 **845,158,632..862,705,895**（包含结尾）。不过有一个必须写入加载器的坑：

> 页在文件里按 tensor 名字的字典序排列，开头是 `0,1,10,100,...`，不是数值上的 `0,1,2,3,...`。所以不能用 `base + expert_id * page_size` 计算 Range；必须先从当层 header 生成 `expert_id -> [start,end]` manifest。

因而准确结论是：**单专家可以用一个连续 Range 获取，但“专家轴按数值 ID 等距排列”为假。**

## 4. 93 层 top-16 与 70GB 缓存

K3 是 93 层，但第 0 层为 dense，所以 top-16 发生在其余 **92 层**。

| 量 | 数值 |
|---|---:|
| 每层 top-16 专家页 | 16 |
| 每层 top-16 字节 | 280,756,224（267.75 MiB） |
| 每 token 穿过 92 层的页引用 | 1,472 |
| 每 token routed expert 权重 | 25,829,572,608 字节（24.056 GiB） |
| 整个 routed expert bank | 82,432 页 / 1,446,456,066,048 字节 |
| 70,000,000,000 字节缓存 | 3,989 页 / 43.36 页每层 / 4.84% expert bank |
| 70 GiB 缓存 | 4,283 页 / 46.55 页每层 / 5.20% expert bank |

最坏情况下，3 个连续 token 的路由集合互不重合，需要 4,416 页 / 77.489GB，超过 70GB。是否可用取决于真实前端任务中的：

1. 每层专家集中度；
2. 相邻 token 的 Jaccard 重叠；
3. 跨 prompt 的热集合覆盖率；
4. 预取提前量和 SSD/RAM/GPU 三级命中率。

没有这四类原生 route trace，“70GB 可覆盖 K3 前端经络”仍是待验证假说，不是已证实成果。

另一条硬件约束：仅 routed expert 权重就是 25.83GB/token。若每 token 都必须重读这些页，20 tok/s 需要至少 **516.6 GB/s**，50 tok/s 需要 **1.291 TB/s** 的专家权重有效带宽，还没算 113.51GB 骨架、KV 和 kernel 开销。这使 K3 原生全深度路径不符合当前本机 20–50 tok/s 目标。

即使假设将整个文本骨架“理想地”从 BF16 压到 4-bit（完全忽略 scale、内核、KV、质量损失的乐观下界），骨架仍约 28.38GB；70GB 总预算只剩 41.62GB，约 2,372 个专家页。官方并没有这个 4-bit 骨架 checkpoint，该数字不能当作可用版本。

## 5. “K3 前端能力”不等于视觉塔

官方 checkpoint 中可独立分片的视觉部分是：

- MoonViT-V2 `vision_tower`：802,428,928 字节净载荷（27 层、hidden 1024）；
- `mm_projector`：92,289,024 字节净载荷（`4096→4096→7168` + norm）；
- 合计 894,717,952 字节净载荷（带 safetensors header 的两个文件合计 894,737,680 字节）。

这两片做的是图像/视频 patch 编码和投影，可以提供“看懂截图”的感知特征，但它们不包含：

- HTML/CSS/JavaScript 生成策略；
- UI 布局、审美和需求分解；
- 工具调用、浏览器动作和错误恢复策略；
- Kimi Code harness 与 max reasoning 下的长程 agent 能力。

这些能力主要分布在 93 层文本主干、MoE 路由/专家、原生 tokenizer/状态契约和后训练策略中。官方模型卡报告 DeepSWE、FrontierSWE、Kimi Code Bench、OSWorld 等整模成绩，**没有报告“前端编程专属 expert ID”，也没有证据说视觉塔单独复制网页编程能力**。

所以：K3 视觉塔可作为北极星的“眼睛”候选，不能被宣称为“K3 前端编程因子”。

## 6. 对北极星里程碑的约束

这次审计支持继续做两件事：

1. **虚拟专家页 manifest：** 固定 revision，按每层 header 生成 `layer/expert -> shard/range/sha256/dtype/shape`，这是真正可实现的基础设施。
2. **原生 route trace 门：** 只有在 K3 原生坐标中对冻结前端题记录 92 层 top-16，并证明 70GB 热集合的覆盖率和相邻 token 重用率后，才能下载专家 payload。

这次审计否决两个过度宣称：

- “只下 K3 骨架 + top-16 就能在 70GB 里跑整个 K3”：**否**；
- “只拆 0.895GB 视觉塔就能吃到 K3 的网页编程能力”：**否**。

它确认的真成果是：**K3 的 82,432 个 layer-expert 页在官方 safetensors 中可精确定位，每页可一次 Range 获取；但路由集中性和快速执行性仍未被证明。**

## 可复核来源

- [Kimi K3 官方模型卡](https://huggingface.co/moonshotai/Kimi-K3)
- [固定 revision config](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json)
- [固定 revision modeling_kimi_linear.py](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py)
- [固定 revision configuration_kimi_k3.py](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/configuration_kimi_k3.py)
- [固定 revision safetensors index](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/model.safetensors.index.json)
- [Hub API（文件大小/LFS SHA-256）](https://huggingface.co/api/models/moonshotai/Kimi-K3?blobs=true)
