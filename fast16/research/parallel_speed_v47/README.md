# parallel_speed_v47

ColorLM v47并行速度架构的纯CPU离线研究包。它不加载模型、不启动服务、不调用GPU，只读取已冻结的v38/v36、v24和v17记录。

## 文件

- `DESIGN.md`：A/B/C最小结构、状态语义、成本和晋级合同。
- `manifest.json`：结构参数、证据路径与SHA-256、声明边界。
- `schemas/`：manifest、短门合同和报告的JSON Schema。
- `short_gate_contract.json`：预声明离线门与后续运行门。
- `cost_model.py`：标准库解析成本计算器。
- `offline_gate.py`：证据哈希、UTF-8、成本门和v24路由回放。
- `selftest.py`：不读取权重的纯CPU单元自检。
- `cost_report.json`、`offline_gate_report.json`：本轮实测产物。
- `HANDOFF.md`：结果、限制与下一步。

## 复现

在工作区根目录使用PowerShell：

```powershell
python fast16/research/parallel_speed_v47/selftest.py
python fast16/research/parallel_speed_v47/cost_model.py --output fast16/research/parallel_speed_v47/cost_report.json
python fast16/research/parallel_speed_v47/offline_gate.py --output fast16/research/parallel_speed_v47/offline_gate_report.json
```

`offline_gate.py`以退出码0表示所有离线硬门通过，以退出码2表示至少一个硬门失败。失败仍会完整写报告。当前C门预期可能失败；这不是脚本故障。

## 禁止外推

报告不证明论文速度、真实端到端加速或运行时可直接集成。A/B没有训练权重；C只回放v17四层岛的一段短路由。任何运行候选都要重新做质量等价、真实吞吐、p95延迟、RAM/VRAM/SSD/PCIe观测。
