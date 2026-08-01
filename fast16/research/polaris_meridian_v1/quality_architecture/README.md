# Polaris Quality Architecture v1

本目录回答一个严格问题：在不训练、不用工具外壳冒充模型智力的条件下，300B+ donor 是否可以变成本机 20--50 tok/s 的单一本地模型。

唯一首选为 `Polaris Native Sparse-Depth S14`：使用 DeepSeek-V4-Flash-0731 原生 tokenizer/embedding/head 和 14 个预注册 residual block，跳过层为 identity。

- [架构决策与证伪门](ARCHITECTURE.md)
- [冻结规格](architecture_spec.json)
- [离线分析器](analyze_quality_architecture.py)
- [生成预算](budget_report.json)

当前只有物理可行性：52.231GB 本地文件、约 4.59B active、权重扫描上界 4.38GB/token。`quality_pass` 仍为 `null`，不得宣称已达到 Claude/GPT 质量。
