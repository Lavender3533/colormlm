# 主线交接

## 已完成

- 固定 `moonshotai/Kimi-K3@9f62e4e9fffbd0a83ddd60e1c209d828994b3569`；
- 固定现有 24 道前端题和评分合同 SHA-256；
- 定义任务/关键 token、逐层 top-16 router、leave-one-out NLL 和固定 header Range schema；
- 实现“节点收益门 + 同 token 共现 + 连续层依赖 + 连通社区”的确定性聚合；
- 输出只读 Range 候选 manifest，代码中没有网络和下载路径；
- 合成自检 10/10 通过，能拒绝 revision 漂移、重复 expert、NLL 差值伪造和非固定 header catalog。

## 主线下一步

1. 在看到任何 K3 router/NLL 前，为现有 24 题用固定 tokenizer 解析关键 token，生成符合 `task.schema.json` 的 `frozen_k3_frontend_tokens.jsonl`。
2. 在固定 K3 原生 runtime 上只采这些 token 的 L1--L92 top-16；反事实只对关键 token 的选中 expert 做 leave-one-out。
3. 生成符合 `trace.schema.json` 的真实 JSONL，`synthetic=false`、`native_forward_completed=true`。
4. 运行 `community.py`。第一次可以不提供 header catalog：先看是否存在跨任务、跨层、NLL 正收益社区；若没有候选，停止，不下载。
5. 只有出现社区后，才读取固定 revision 的 safetensors header，生成 `range_catalog.schema.json` 对应 catalog，再跑一次精确 Range dry-run供主线审批。

## 当前结论

真实 K3 trace 尚未采集，因此当前 expert 候选集合仍为空。`SELFTEST_REPORT.json` 中的 L20/E101 等编号全部是合成 fixture，不得用于下载或能力宣称。MoonViT 只提供截图视觉编码，不是网页代码能力。
