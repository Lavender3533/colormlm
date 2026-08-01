# Kimi K3 前端专家社区定位

本目录只建立 `moonshotai/Kimi-K3` 的 HTML/CSS/JavaScript 能力定位基础设施。它不下载权重、不启动模型，也不预设任何 expert ID。

## 能力边界

- MoonViT-V2 是图像/视频 patch 编码器，适合网页截图理解。
- HTML/CSS/JavaScript 生成、交互设计、无障碍和工具策略属于 K3 文本主干的分布式能力。
- 一个 routed expert 页只有 16.734375 MiB，但没有供体原生 hidden、router、shared path 和相邻层状态时，不是独立的“前端能力”。
- 因此定位目标是由真实 K3 原生 trace 支持的**跨层专家社区**，而不是从旧实验编号里挑一颗专家。

## 固定输入

- K3：`moonshotai/Kimi-K3@9f62e4e9fffbd0a83ddd60e1c209d828994b3569`。
- 架构：93 个文本层，其中 L1--L92 为 896 experts、top-16 MoE。
- 前端任务：`parallel_frontend_v47/data/{train,validation,blind}.jsonl`，共 24 题；路径和 SHA-256 固定在 `source_contract.json`。
- 关键 token 必须在看到 router/NLL trace 前冻结。`task.schema.json` 规定了完整前缀、token ID、文本、类别和选择理由。

## 原生 trace

`trace.schema.json` 要求每个 task/token/layer 记录：

1. 完整前缀 SHA-256、目标 token ID/text、目标 NLL；
2. 该层原生 top-16 expert ID 与权重；
3. 对关键 token 的逐 expert leave-one-out NLL；
4. `ablated_nll - native_nll`，正值才表示该 expert 在该点有帮助；
5. 固定 repo/revision 和 `synthetic=false`。

仅有 route 频率不够。高频 expert 可能只是通用语法/格式 expert；没有反事实正收益时不会进入社区。

## 聚合算法

`community.py` 是纯标准库、确定性的 dry-run：

- 节点：`(layer, expert)`；
- 节点门：跨任务关键 token 覆盖 + leave-one-out 正收益；
- 共现边：同一关键 token 同层共同被路由且双方均有正收益；
- 连续层边：同一关键 token 在相邻或近邻层连续出现且双方均有正收益；
- 社区：通过证据门的图连通分量；
- 输出：只生成 Range 候选 manifest，`download_authorized=false`。

若没有固定 revision 的 safetensors header catalog，候选只给出六个 tensor 名模板，Range 状态为 `blocked_missing_pinned_header`；绝不套用旧 `master` header 的 offset。提供 `range_catalog.schema.json` 对应的固定 header catalog 后，才会填入精确 byte range，但仍不会下载。

## 使用

```powershell
python -X utf8 -m py_compile community.py selftest.py
python -X utf8 selftest.py --output SELFTEST_REPORT.json --force

# 真实 trace dry-run；不加 --allow-synthetic 时拒绝合成记录
python -X utf8 community.py `
  --tasks frozen_k3_frontend_tokens.jsonl `
  --trace native_k3_frontend_trace.jsonl `
  --output range_candidates.json
```

有固定 revision header catalog 时：

```powershell
python -X utf8 community.py `
  --tasks frozen_k3_frontend_tokens.jsonl `
  --trace native_k3_frontend_trace.jsonl `
  --header-catalog pinned_header_ranges.json `
  --output range_candidates.json
```

无论哪种命令都只写 dry-run 清单，不包含 HTTP 请求或下载动作。

## 晋级边界

社区定位通过只允许下一步做 Range 审批和供体原生消融。它不证明这些页能跨 7168→北极星坐标工作，也不证明北极星已经得到 K3 前端能力。
