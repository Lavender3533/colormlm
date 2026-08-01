# Polaris Meridian v0：K3 微块扫描

- 胶囊：`D:\project\大模型ssd化\fast16\research\neural_bus_capsules\kimi_k3_l65_e752_real\runtime_v2`
- 隐藏态：`D:\project\大模型ssd化\fast16\research\v14_live\l12_feature.hidden.npz`
- 样本：72 条真实隐藏态
- 切分：48 × 64 神经元
- 解析路由秩：4

| 路由 | top 块 | 激活能量覆盖 | cosine 均值 | cosine P10 | NRMSE 均值 | 动态 F16/Token |
|---|---:|---:|---:|---:|---:|---:|
| activation_oracle | 1 | 0.0507 | 0.2250 | 0.1788 | 1.2471 | 1.19 MiB |
| activation_oracle | 2 | 0.0938 | 0.3050 | 0.2624 | 1.1803 | 2.38 MiB |
| activation_oracle | 4 | 0.1687 | 0.4094 | 0.3684 | 1.0893 | 4.75 MiB |
| activation_oracle | 8 | 0.2949 | 0.5408 | 0.5108 | 0.9582 | 9.50 MiB |
| activation_oracle | 16 | 0.5041 | 0.7090 | 0.6836 | 0.7624 | 19.00 MiB |
| activation_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| output_contribution_oracle | 1 | 0.0506 | 0.2251 | 0.1788 | 1.2475 | 1.19 MiB |
| output_contribution_oracle | 2 | 0.0936 | 0.3049 | 0.2591 | 1.1809 | 2.38 MiB |
| output_contribution_oracle | 4 | 0.1685 | 0.4085 | 0.3697 | 1.0910 | 4.75 MiB |
| output_contribution_oracle | 8 | 0.2944 | 0.5407 | 0.5081 | 0.9590 | 9.50 MiB |
| output_contribution_oracle | 16 | 0.5035 | 0.7098 | 0.6868 | 0.7615 | 19.00 MiB |
| output_contribution_oracle | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |
| analytic_proxy | 1 | 0.0238 | 0.1545 | 0.0981 | 1.3005 | 1.19 MiB |
| analytic_proxy | 2 | 0.0464 | 0.2106 | 0.1514 | 1.2568 | 2.38 MiB |
| analytic_proxy | 4 | 0.0923 | 0.2989 | 0.2413 | 1.1840 | 4.75 MiB |
| analytic_proxy | 8 | 0.1765 | 0.4131 | 0.3610 | 1.0834 | 9.50 MiB |
| analytic_proxy | 16 | 0.3479 | 0.5857 | 0.5424 | 0.9097 | 19.00 MiB |
| analytic_proxy | 48 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 57.00 MiB |

## 判定

**停止 top-4，检查 top-8/16**：即使激活神谕的 top-4 也无法稳定重构完整专家方向。

该结果只回答‘微块能否重构一颗真实 K3 专家的输出方向’，不等于整模能力提升。
