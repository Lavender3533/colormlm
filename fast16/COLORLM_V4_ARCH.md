# ColorLM v4 架构规范

## 目标与边界

ColorLM v4 是一个独立模型，不是现成聊天模型的提示词外壳。最终运行时只加载
ColorLM 自己的张量、持续状态和词表，不加载供体模型，也不使用关键词 `if`、
BM25 文本拼接或候选答案投票来产生核心能力。

硬约束：

- 模型文件总大小不超过 15 GiB；
- RX 5700 XT 8 GiB 上使用 Vulkan GPU；
- 32 GiB 内存和 SSD mmap 承担冷权重；
- 新知识更新只修改模型内的快速状态，单次耗时 1--30 秒；
- 首个版本处理文本，后续再加入视觉通道；
- 供体权重只用于一次性编译，不成为最终运行依赖。

## 计算图

模型使用 40 个状态层，按 10 个宏单元排列。每个宏单元包含三个
ColorDelta 层和一个 ColorKernel 层：

```text
token embedding
    -> 10 x [ColorDelta -> ColorDelta -> ColorDelta -> ColorKernel]
    -> recurrent depth mixer
    -> RMSNorm
    -> token head
```

它不是标准 Transformer 堆叠。40 层全部使用常量大小的递归状态，不执行
softmax(QK) 的完整 token-token 注意力，也不保存完整 token KV。

### ColorDelta

对第 `t` 个 token，隐藏状态为 `x_t`，矩阵状态为 `S_t`：

```text
q, k, v, decay, beta = Project(RMSNorm(x_t))
prediction = S_(t-1) @ k
error      = v - prediction
S_t        = exp(-softplus(decay)) * S_(t-1) + beta * outer(error, k)
y_t        = S_t @ q
x_t        = x_t + Output(y_t)
```

这是可学习的连续状态更新。时间复杂度随序列长度线性增长，推理时不需要完整
KV Cache。`S_t` 是模型状态，不是检索结果。

### ColorKernel

ColorKernel 用正核特征近似供体的完整注意力。它维护固定大小的值状态 `L` 和
归一化状态 `z`，特征数 `m` 不随上下文增长：

```text
q, k, v = Project(RMSNorm(x))
phi(u)  = elu(W_feature u) + 1
L       = decay * L + outer(phi(k), v)
z       = decay * z + phi(k)
y       = (phi(q) @ L) / (phi(q) @ z + epsilon)
x       = x + W_output y
```

`W_feature` 由供体文件哈希确定的正交矩阵生成，不需要训练。供体 Q/K/V/O 权重
迁移到投影层，但 softmax 注意力本身不会进入最终模型。这里没有关键词匹配、
文档召回或离散任务判断。

### 稀疏神经专家场

每层有 256 个参数专家，每个 token 激活 8 个路由专家和 1 个共享专家：

```text
routing = softmax(W_router * RMSNorm(x))
expert  = sum(top8(routing)[e] * E_e(x)) + E_shared(x)
x       = x + expert
```

路由来自模型隐藏状态和可学习权重。专家使用 ColorQ3 分组矢量量化存储；运行时
把高频专家留在显存池，冷专家由 mmap 和异步预取进入显存。SSD 调度只改变张量
所在位置，不改变神经计算结果。

### 固定递归深度

最后两个宏单元共享一次附加前向。两次结果通过连续门控融合：

```text
h1 = Macro(h0, state)
h2 = Macro(h1, state)
alpha = sigmoid(W_depth * h1)
h = (1 - alpha) * h1 + alpha * h2
```

两条路径总是计算，不通过规则决定“要不要思考”。后续可把第二次计算改为 GPU
批处理，以减少递归带来的延迟。

## 快速学习

每个宏单元另外维护低秩快速状态 `F = A @ B`。它用局部 Delta 规则更新，不对
40 层主权重做反向传播：

```text
key   = normalize(P_key * h)
value = P_value * target
error = value - F * key
F     = lambda * F + eta * outer(error, key)
h     = h + gate * P_out * (F * key)
```

更新是连续矩阵运算，可在 GPU 上完成。默认状态为零，因此未写入知识时不会污染
基础能力。持久化文件保存 `F`，不会把原文重新拼到用户问题前面。

## 知识供体编译

v4 不从随机权重开始。供体编译器一次性读取强模型权重并完成：

1. 映射词嵌入、输出头、归一化和 MoE 专家到 ColorLM 张量名；
2. 把序列层投影为 ColorDelta/ColorKernel 的初始化参数；
3. 对专家做 ColorQ3 分组矢量量化和逐层误差校正；
4. 用不超过 30 秒的隐藏状态校准只优化残差门控和尺度；
5. 写出独立 `.clm`，随后供体文件可以删除。

首个供体优先选择以门控 DeltaNet 和稀疏 MoE 为主的模型，从而最大限度保留
函数结构。供体不是产品架构，也不会在运行时作为第二个模型存在。

## 15 GiB 预算

| 区域 | 上限 |
|---|---:|
| ColorQ3 专家权重 | 11.5 GiB |
| 状态层、路由器、共享专家 | 1.6 GiB |
| 词嵌入和输出头 | 0.9 GiB |
| 快速权重和持久状态 | 0.5 GiB |
| manifest、量化表和校验信息 | 0.2 GiB |
| 预留 | 0.3 GiB |
| **总计** | **15.0 GiB** |

## 验收门槛

每个阶段必须通过行为验证，不能用“程序成功退出”代替能力验证：

1. 供体编译后，固定提示集的 next-token KL 和任务结果达到设定阈值；
2. 禁用快速状态时，同一问题不受项目 README 或无关文档影响；
3. 写入一条新事实后，不改提示词即可在新会话回忆，且无关问题不被污染；
4. Java、Rust、Go 代码必须实际编译并通过测试；
5. 推理、中文、多轮约束、长上下文和代码任务分别与目标模型盲测；
6. 最终文件不超过 15 GiB，GPU 日志证明 Vulkan 执行，输出速度单独记录。
