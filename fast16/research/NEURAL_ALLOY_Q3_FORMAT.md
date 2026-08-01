# Neural Alloy q3_g64 v1

每个差分张量按 GGUF 的逻辑展开顺序切成64值分组。每组使用一个F16绝对尺度和64个三位
有符号整数，量化范围为 `[-3, 3]`。三位整数占24字节，尺度占2字节，因此每组固定26字节。

文件包含固定头、UTF-8 JSON张量清单、4096字节对齐的数据区。张量数据区偏移按64字节对齐，
每条清单记录名称、GGUF逻辑形状、逻辑值数、组数、相对偏移、字节数和基座类型。数据偏移
相对于数据区起点，避免JSON长度变化影响记录。

运行时语义：

```text
W_effective(layer, module, h) = W_base + alpha(layer, module, h) * dequant_q3(delta)
```

v1容器先支持固定alpha和CPU参考解码。后续 Vulkan 算子直接读取26字节块，在矩阵乘法累加
阶段应用F16尺度和alpha，不生成完整F32差分矩阵。

## 构建期融合路径

`compile_q3_fusion.py`可以把选中的q3差分直接物化为标准GGUF张量：

```text
W_fused = W_base + alpha * dequant_q3(delta)
```

这条路径在构建时解码，运行时使用llama.cpp现有Vulkan矩阵内核，不依赖LoRA、
CPU custom op或额外模型进程。`ColorLM-v6-Q3Router-Fused-A1.gguf`已物化40层MoE
路由差分，单文件大小12.79GiB；RX 5700 XT的最短生成检查为18.0 token/s。

构建期融合是固定alpha的零运行时开销路径。动态每层、每模块和每token alpha仍需要
原生Vulkan q3_g64解码与累加算子。
