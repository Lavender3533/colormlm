# Neural Slice ABI

## 目标

把任意供体模型中的自包含神经模块拆成切片，接入一个公共残差流，并最终编译成单一模型文件。
切片不是文本检索、提示词路由或多进程模型调用；它必须直接参与同一个 token 的神经计算图。

## v1：同宽直接切片

ColorLM 与 GLM-4.7 的架构不同，但残差宽度都是 2048。GLM 每层共享专家是一个完整的
`2048 -> 1536 -> 2048` 模块，因此可直接替换 ColorLM 的共享专家：

```text
h_main  = ColorLM_MoE(h)
h_slice = GLM_SharedExpert(RMSNorm(h))
h_next  = residual + h_main + sigmoid(ColorLM_gate(h)) * h_slice
```

第一版按归一化深度把 GLM 第 1-46 层映射到 ColorLM 第 0-39 层，120个 GLM 张量被编译
进一个 GGUF。ColorLM 的注意力、SSM、256专家主干和逐 token 共享专家门控保持不变。

## 通用接口

一个切片包至少包含：

- `input_width` 与 `output_width`
- 入口归一化规则
- 神经算子和冻结权重
- 残差缩放范围
- 状态布局（无状态、KV、SSM或外部记忆）
- 计算量、常驻内存和每 token 权重带宽
- 来源模型、来源层和张量哈希

当供体宽度不同，使用两个小桥：

```text
z = P_in * RMSNorm(h)
u = DonorSlice(z)
h' = h + gate(h) * P_out * u
```

`P_in/P_out` 优先通过共同词表锚点的正交 Procrustes、解析伪逆或短时激活校准获得；
它们是数值坐标桥，不承担知识训练。

## 映射模式

- `depth`：供体层按归一化深度对齐，优先保证稳定。
- `bands`：前10层取低层切片，中10层取中层切片，后20层交替注入高层切片。
- `alternating`：供体最低层与最高层逐层交替，用于测试强交叠的稳定边界。

## 下一阶段

- 给每层切片增加可学习或解析得到的连续门控强度。
- 同一层并联 GLM、Qwen 和代码供体切片，只激活 top-k。
- 把注意力、MoE专家、SSM状态核都纳入切片类型。
- 实现切片感知的融合算子，避免额外内存读取拖慢生成。
- 用能力增益/额外毫秒和能力增益/额外GiB选择保留哪些切片。

## Synaptic Graft

整块共享专家替换的首次短生成能够加载并达到 13.4 token/s，但语义输出崩坏。随机 RMSNorm
输入探针显示 GLM 切片输出能量是 ColorLM 原共享专家的 5-28 倍，输出方向余弦约为 0。

第二种切片方式保持每层共享专家宽度为512：保留重要度最高的480个 ColorLM 神经元，从
对应 GLM 层的1536个神经元中选择重要度最高的32个，并缩放 GLM 下投影，使嫁接支路目标
能量为保留 ColorLM 支路的3%。重要度定义为：

```text
importance_i = RMS(gate_row_i) * RMS(up_row_i) * RMS(down_column_i)
```

这种组合不增加每 token 矩阵乘法尺寸，可推广到任何具有相同输入/输出宽度的 SwiGLU 模块。
