# Polaris Meridian v0：K3 微块扫描

- 胶囊：`D:\project\大模型ssd化\fast16\research\neural_bus_capsules\kimi_k3_l28_e41_real\runtime_v2`
- 隐藏态：`D:\project\大模型ssd化\fast16\research\v14_live\l12_feature.hidden.npz`
- 样本：72 条真实隐藏态
- 切分：48 × 64 神经元
- 解析路由秩：4

| 路由 | top 块 | 激活能量覆盖 | cosine 均值 | cosine P10 | NRMSE 均值 | 动态 F16/Token |
|---|---:|---:|---:|---:|---:|---:|
| activation_oracle | 1 | 0.0656 | 0.2785 | 0.2166 | 1.2157 | 1.19 MiB |
| activation_oracle | 2 | 0.1173 | 0.3650 | 0.3012 | 1.1396 | 2.38 MiB |
| activation_oracle | 4 | 0.2027 | 0.4743 | 0.4171 | 1.0325 | 4.75 MiB |
| activation_oracle | 8 | 0.3411 | 0.6062 | 0.5594 | 0.8929 | 9.50 MiB |
| activation_oracle | 16 | 0.5539 | 0.7642 | 0.7377 | 0.6886 | 19.00 MiB |
| activation_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| output_contribution_oracle | 1 | 0.0645 | 0.2785 | 0.2161 | 1.2231 | 1.19 MiB |
| output_contribution_oracle | 2 | 0.1149 | 0.3675 | 0.2959 | 1.1439 | 2.38 MiB |
| output_contribution_oracle | 4 | 0.1987 | 0.4753 | 0.4178 | 1.0370 | 4.75 MiB |
| output_contribution_oracle | 8 | 0.3356 | 0.6095 | 0.5679 | 0.8928 | 9.50 MiB |
| output_contribution_oracle | 16 | 0.5483 | 0.7688 | 0.7477 | 0.6843 | 19.00 MiB |
| output_contribution_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| analytic_proxy | 1 | 0.0278 | 0.1703 | 0.1013 | 1.2772 | 1.19 MiB |
| analytic_proxy | 2 | 0.0506 | 0.2265 | 0.1559 | 1.2346 | 2.38 MiB |
| analytic_proxy | 4 | 0.1077 | 0.3384 | 0.2668 | 1.1484 | 4.75 MiB |
| analytic_proxy | 8 | 0.2025 | 0.4562 | 0.3760 | 1.0403 | 9.50 MiB |
| analytic_proxy | 16 | 0.3768 | 0.6192 | 0.5596 | 0.8693 | 19.00 MiB |
| analytic_proxy | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |

## 判定

**停止 top-4，检查 top-8/16**：即使激活神谕的 top-4 也无法稳定重构完整专家方向。

该结果只回答‘微块能否重构一颗真实 K3 专家的输出方向’，不等于整模能力提升。
