# Neural Alloy Router v1

## 这是什么

这是 Neural Alloy 的首个可推理模型级原型。它在一个 ColorLM 推理图中计算：

```text
W_router(alpha) = W_ColorLM + alpha * (W_Qwen3.6 - W_ColorLM)
```

当前覆盖 40 层 MoE 路由矩阵。每个差分都使用满秩分解 `A=delta, B=I`，
因此它不是低秩近似。adapter 使用 F16 存储，大小为 47,192,128 bytes。

## 已验证

- 40/40 路由张量成功装入同一个推理图。
- F16 差分存储相对 L2 误差为 `0.0000414053`。
- `alpha=0` 与完全不加载 adapter 的固定采样输出逐字一致。
- `alpha=1` 在同一提示、种子和采样参数下改变了输出，证明差分进入 logits。
- RX 5700 XT Vulkan 推理成功：约 8.3 token/s。

## 当前边界

这一版证明了动态权重合金机制可运行，但只替换路由器，不能据此声称模型能力已经提升。
下一阶段应把相同语义扩展为 q3_g64 全秩差分算子，覆盖注意力、共享专家、路由专家、
归一化与输出层，再比较完整 `alpha=0/0.5/1` 的真实能力。

## 试用

在项目根目录运行：

```bat
fast16\run-neural-alloy-router.bat 1
```

参数 `0` 是纯 ColorLM 基座，`0.5` 是中点，`1` 是供体路由权重。
