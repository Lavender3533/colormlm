# ColorLM Neural Bus v1 运行时契约

更新日期：2026-07-29

## 目标与边界

v1验证“冻结的外来神经模块能否在同一token计算图中，以可关闭残差参与ColorLM推理”。
它不替换v6专家，不启动第二个模型，不读取文本关键词，不做检索或外部任务分类。

## 胶囊张量

Coder-Next胶囊使用已完成坐标运输的第47层专家47：

| 张量 | GGML形状 | 量化 | 含义 |
|---|---:|---|---|
| `gate.q4_0` | `[2048, 512]` | Q4_0 | SwiGLU gate |
| `up.q4_0` | `[2048, 512]` | Q4_0 | SwiGLU up |
| `down.q4_0` | `[512, 2048]` | Q4_0 | 返回ColorLM残差空间 |

两个2048维坐标桥已物化到三个权重中，运行时不再执行稠密`2048×2048`投影。胶囊权重
共约1.69 MiB，原型期间作为独立侧载资产；通过验收后再作为额外张量写入单一GGUF。

## 图内数学

默认路由站为ColorLM的第12、28层（从0开始）。胶囊与原生MoE读取同一个
`attn_post_norm`隐藏状态：

```text
delta = W_down(SiLU(W_gate h) * (W_up h))
r     = sqrt(mean(delta^2) / mean(h_residual^2))
g     = sigmoid(-sharpness * log(clamp(r / target_ratio)))
h_out = h_residual + h_native_moe + alpha * g * delta
```

`g`由当前token的隐藏状态和胶囊修正量连续计算，不是主机端`if`或关键词路由。
v1先用能量门控防止未校准胶囊压倒主干，后续再将它替换为1–30秒校准的小路由器。

## 关闭等价条件

- `COLORLM_NEURAL_BUS_ALPHA` 缺失或为`0`时，不加载胶囊。
- `alpha=0`时，不向GGML图加入任何Neural Bus节点，代码路径精确回到v6。
- 胶囊路径、张量大小或后端分配无效时，启动直接失败，不用零权重或其他专家代替。

## 原型开关

| 环境变量 | 默认 | 范围 |
|---|---:|---:|
| `COLORLM_NEURAL_BUS_CAPSULE` | 空 | 胶囊目录 |
| `COLORLM_NEURAL_BUS_ALPHA` | `0` | `0..1` |
| `COLORLM_NEURAL_BUS_SITES` | `12,28` | `0..n_layer-1` |
| `COLORLM_NEURAL_BUS_TARGET_RATIO` | `0.08` | `0.001..1` |
| `COLORLM_NEURAL_BUS_SHARPNESS` | `4` | `0.25..16` |

## 最小验收

1. 关闭总线后，固定seed贪心生成逐token等价v6。
2. 开启总线后，中间`neural_bus_delta`非零且输出实际受到影响。
3. 只跑3个短编程任务，不跑长榜。
4. 128-token相邻A/B的生成速度损失不超过15%。
5. 未同时达到能力和速度门槛时，v6仍是生产默认。
