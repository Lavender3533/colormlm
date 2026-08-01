# Polaris Meridian v0：K3 微块扫描

- 胶囊：`D:\project\大模型ssd化\fast16\research\neural_bus_capsules\kimi_k3_l65_e539_real\runtime_v2`
- 隐藏态：`D:\project\大模型ssd化\fast16\research\v14_live\l12_feature.hidden.npz`
- 样本：72 条真实隐藏态
- 切分：48 × 64 神经元
- 解析路由秩：4

| 路由 | top 块 | 激活能量覆盖 | cosine 均值 | cosine P10 | NRMSE 均值 | 动态 F16/Token |
|---|---:|---:|---:|---:|---:|---:|
| activation_oracle | 1 | 0.0477 | 0.2148 | 0.1751 | 1.2524 | 1.19 MiB |
| activation_oracle | 2 | 0.0892 | 0.2977 | 0.2612 | 1.1846 | 2.38 MiB |
| activation_oracle | 4 | 0.1623 | 0.4003 | 0.3695 | 1.0951 | 4.75 MiB |
| activation_oracle | 8 | 0.2900 | 0.5391 | 0.5099 | 0.9595 | 9.50 MiB |
| activation_oracle | 16 | 0.5006 | 0.7092 | 0.6820 | 0.7622 | 19.00 MiB |
| activation_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| output_contribution_oracle | 1 | 0.0476 | 0.2168 | 0.1849 | 1.2518 | 1.19 MiB |
| output_contribution_oracle | 2 | 0.0890 | 0.2980 | 0.2599 | 1.1852 | 2.38 MiB |
| output_contribution_oracle | 4 | 0.1618 | 0.4035 | 0.3732 | 1.0932 | 4.75 MiB |
| output_contribution_oracle | 8 | 0.2895 | 0.5395 | 0.5107 | 0.9595 | 9.50 MiB |
| output_contribution_oracle | 16 | 0.5001 | 0.7098 | 0.6834 | 0.7617 | 19.00 MiB |
| output_contribution_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| analytic_proxy | 1 | 0.0254 | 0.1589 | 0.1106 | 1.2944 | 1.19 MiB |
| analytic_proxy | 2 | 0.0459 | 0.2163 | 0.1681 | 1.2499 | 2.38 MiB |
| analytic_proxy | 4 | 0.0917 | 0.3045 | 0.2492 | 1.1770 | 4.75 MiB |
| analytic_proxy | 8 | 0.1768 | 0.4222 | 0.3780 | 1.0732 | 9.50 MiB |
| analytic_proxy | 16 | 0.3486 | 0.5909 | 0.5608 | 0.9037 | 19.00 MiB |
| analytic_proxy | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |

## 判定

**停止 top-4，检查 top-8/16**：即使激活神谕的 top-4 也无法稳定重构完整专家方向。

该结果只回答‘微块能否重构一颗真实 K3 专家的输出方向’，不等于整模能力提升。
