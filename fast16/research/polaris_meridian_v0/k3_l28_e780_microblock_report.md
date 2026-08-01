# Polaris Meridian v0：K3 微块扫描

- 胶囊：`D:\project\大模型ssd化\fast16\research\neural_bus_capsules\kimi_k3_l28_e780_real\runtime_v2`
- 隐藏态：`D:\project\大模型ssd化\fast16\research\v14_live\l12_feature.hidden.npz`
- 样本：72 条真实隐藏态
- 切分：48 × 64 神经元
- 解析路由秩：4

| 路由 | top 块 | 激活能量覆盖 | cosine 均值 | cosine P10 | NRMSE 均值 | 动态 F16/Token |
|---|---:|---:|---:|---:|---:|---:|
| activation_oracle | 1 | 0.0673 | 0.2874 | 0.2301 | 1.2002 | 1.19 MiB |
| activation_oracle | 2 | 0.1215 | 0.3832 | 0.3228 | 1.1181 | 2.38 MiB |
| activation_oracle | 4 | 0.2074 | 0.4928 | 0.4371 | 1.0113 | 4.75 MiB |
| activation_oracle | 8 | 0.3396 | 0.6170 | 0.5768 | 0.8745 | 9.50 MiB |
| activation_oracle | 16 | 0.5448 | 0.7645 | 0.7334 | 0.6845 | 19.00 MiB |
| activation_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| output_contribution_oracle | 1 | 0.0652 | 0.3017 | 0.2520 | 1.2038 | 1.19 MiB |
| output_contribution_oracle | 2 | 0.1190 | 0.3909 | 0.3374 | 1.1179 | 2.38 MiB |
| output_contribution_oracle | 4 | 0.2040 | 0.4979 | 0.4444 | 1.0093 | 4.75 MiB |
| output_contribution_oracle | 8 | 0.3358 | 0.6229 | 0.5885 | 0.8712 | 9.50 MiB |
| output_contribution_oracle | 16 | 0.5414 | 0.7682 | 0.7429 | 0.6806 | 19.00 MiB |
| output_contribution_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| analytic_proxy | 1 | 0.0301 | 0.1731 | 0.0951 | 1.2769 | 1.19 MiB |
| analytic_proxy | 2 | 0.0545 | 0.2401 | 0.1630 | 1.2263 | 2.38 MiB |
| analytic_proxy | 4 | 0.1029 | 0.3235 | 0.2394 | 1.1531 | 4.75 MiB |
| analytic_proxy | 8 | 0.1939 | 0.4512 | 0.3774 | 1.0396 | 9.50 MiB |
| analytic_proxy | 16 | 0.3724 | 0.6286 | 0.5683 | 0.8577 | 19.00 MiB |
| analytic_proxy | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |

## 判定

**停止 top-4，检查 top-8/16**：即使激活神谕的 top-4 也无法稳定重构完整专家方向。

该结果只回答‘微块能否重构一颗真实 K3 专家的输出方向’，不等于整模能力提升。
