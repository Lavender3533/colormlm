# Polaris Meridian v0：K3 微块扫描

- 胶囊：`D:\project\大模型ssd化\fast16\research\neural_bus_capsules\kimi_k3_l28_e41_real\runtime_v2`
- 隐藏态：`D:\project\大模型ssd化\fast16\research\v14_live\l12_feature.hidden.npz`
- 样本：72 条真实隐藏态
- 切分：48 × 64 神经元
- 解析路由秩：4

| 路由 | top 块 | 激活能量覆盖 | cosine 均值 | cosine P10 | NRMSE 均值 | 动态 F16/Token |
|---|---:|---:|---:|---:|---:|---:|
| activation_oracle | 1 | 0.0598 | 0.2544 | 0.1957 | 1.2285 | 1.19 MiB |
| activation_oracle | 2 | 0.1089 | 0.3489 | 0.2901 | 1.1505 | 2.38 MiB |
| activation_oracle | 4 | 0.1928 | 0.4642 | 0.4121 | 1.0442 | 4.75 MiB |
| activation_oracle | 8 | 0.3293 | 0.5964 | 0.5472 | 0.9036 | 9.50 MiB |
| activation_oracle | 16 | 0.5388 | 0.7520 | 0.7191 | 0.7062 | 19.00 MiB |
| activation_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| output_contribution_oracle | 1 | 0.0577 | 0.2717 | 0.2288 | 1.2305 | 1.19 MiB |
| output_contribution_oracle | 2 | 0.1059 | 0.3593 | 0.3084 | 1.1522 | 2.38 MiB |
| output_contribution_oracle | 4 | 0.1888 | 0.4714 | 0.4254 | 1.0437 | 4.75 MiB |
| output_contribution_oracle | 8 | 0.3246 | 0.6014 | 0.5535 | 0.9018 | 9.50 MiB |
| output_contribution_oracle | 16 | 0.5333 | 0.7564 | 0.7257 | 0.7012 | 19.00 MiB |
| output_contribution_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| analytic_proxy | 1 | 0.0278 | 0.1760 | 0.0995 | 1.2857 | 1.19 MiB |
| analytic_proxy | 2 | 0.0562 | 0.2466 | 0.1766 | 1.2340 | 2.38 MiB |
| analytic_proxy | 4 | 0.1088 | 0.3477 | 0.2745 | 1.1471 | 4.75 MiB |
| analytic_proxy | 8 | 0.1966 | 0.4639 | 0.4099 | 1.0396 | 9.50 MiB |
| analytic_proxy | 16 | 0.3764 | 0.6296 | 0.5738 | 0.8614 | 19.00 MiB |
| analytic_proxy | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |

## 判定

**停止 top-4，检查 top-8/16**：即使激活神谕的 top-4 也无法稳定重构完整专家方向。

该结果只回答‘微块能否重构一颗真实 K3 专家的输出方向’，不等于整模能力提升。
