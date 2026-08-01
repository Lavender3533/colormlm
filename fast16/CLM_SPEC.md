# CLM ZeroTrain v0

CLM 是面向零训练模型的单文件容器。v0 的目标是先建立可运行的新模型本体：冻结神经核心、连续关联记忆和共享高层递归都由同一前向运行时执行。

## 二进制布局

```text
0                         64
+-------------------------+
| 固定头                  |
+-------------------------+
| UTF-8 JSON manifest     |
+-------------------------+  4 KiB 对齐
| tensor payload          |
| 每个 tensor 64B 对齐    |
+-------------------------+
```

固定头使用小端编码 `<8sIIQQ32s>`：

| 字段 | 类型 | 含义 |
|---|---|---|
| magic | 8 bytes | `CLMZERO1` |
| version | u32 | 当前为 1 |
| flags | u32 | v0 保留 |
| manifest_len | u64 | JSON 字节数 |
| payload_offset | u64 | tensor 区绝对偏移 |
| manifest_sha256 | 32 bytes | manifest 完整性校验 |

manifest 保存架构、分词表、递归参数、记忆参数和 tensor 的 dtype、shape、相对偏移及 SHA-256。

## v0 前向传播

```text
x = Embedding(token) + Position(token)
x = LowerLayers(x)
x = x + memory_hidden_scale * AssociativeMemory(x)
h1 = UpperLayers(x)
h2 = UpperLayers(h1)
h  = lerp(h1, h2, recurrence_alpha)
logits = TokenHead(RMSNorm(h)) + memory_logit_scale * MemoryPrior
```

关联记忆使用冻结 token embedding 生成归一化 Key。推理时通过余弦相似度和 Softmax 连续融合 Top-K Value，不修改神经权重。

## 零训练约束

- 打包过程只读取现有权重并转换存储 dtype。
- 记忆构建只进行前向编码、量化和索引。
- 运行时没有优化器、损失函数和反向传播。
- 新知识写入 memory tensor，不写入神经权重。

## 后续格式扩展

CLM tensor 表不绑定 PyTorch。后续版本会加入 GGUF 核心映射、分块量化、SSD 分页记忆和 Vulkan 直接读取，同时保持 manifest 和 mmap 读取方式。

## v1 UTF-8 Byte 移植

v1 将旧字符词表解析为 256 个字节 token，并加入 PAD/MASK。已有 ASCII 行保持原权重；其余字节由原 embedding 和输出头的谱基按字节位编码确定性合成。该过程只使用权重矩阵，不读取语料。

DirectML 图由 CLM 权重导出，包含 embedding、双向 attention、FFN、RMSNorm、共享高层递归和输出头。关联记忆检索在 CPU 上完成，memory context 作为连续向量输入 GPU 图。
