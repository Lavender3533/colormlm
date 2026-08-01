# v42：v33 全局 MoE + v29 策略头（否决）

目的：验证 20.20 GiB 的 v33 较高精度全局 Qwen3.6 MoE 核心能否作为质量档，再叠加已验证的
v29 显式工具策略头。没有复制 GGUF，也没有下载新权重。

运行工程结果：

- batch/ubatch 512：模型权重约 `6820.85 MiB`、策略计算缓冲约 `1356.30 MiB`，8 GiB Vulkan
  warmup OOM。
- batch/ubatch 256：稳定启动并完成全部短门。
- 统一启动器现支持 `--batch-size` 与 `--ubatch-size`；默认仍为 512，不改变已有入口。

能力结果：

```text
旧工具状态20题：v33 6/20 → v33+策略头 9/20（3修复、0回归）
既有八维16题：v33+策略头 8/16
v36/v38同一八维16题：10/16
```

八维门相对 v36/v38 回归一题 coding 和一题 planning；tools、computer use 仍为 0/2。
因此 v42 更大、更慢且综合能力更低，否决，不建立用户启动入口，不替换 v38。

证据：

- `v33.policy20.control.json`
- `v33.policy20.v29head.json`
- `v33-policy.full16.responses.jsonl`
- `v33-policy.full16.score.json`

