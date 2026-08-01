# ColorLM Capsule Lab Handoff

状态：`WAITING_FOR_MAINLINE_APPROVAL`  
日期：2026-07-31

## 已完成

- 仅通过配置、索引和容器头审计 Qwen3-Coder-Next、GLM-4.7-Flash、Qwen3.6 及已有 donor 衍生权重。
- 定义输出头、末端层、共享专家、路由专家、DSpark 五类胶囊。
- 固化精确 tensor 名称/shape/来源层/字节数，见 `TENSOR_CATALOG.md`。
- 定义 8 MiB chunk、32 MiB 峰值工作集、`.part` 原子提交的流式提取方案。
- 定义 `capsule.schema.json`、逐 payload SHA-256、外置 `capsule.json.sha256` 和 content root。
- 完成坐标投影、token 映射、权重带宽、FLOP 与状态/KV 开销估算。

## 审批前仍然禁止

- 不运行任何提取、量化、转置或全文件 SHA-256 作业。
- 不下载 shard、权重或 tokenizer。
- 不启动/停止模型，不运行推理，不访问 GPU。
- 不把 `recorded` 哈希改写为 `verified`。
- 不允许 `master` 作为正式提取 revision。

## 推荐的第一项批准

优先复用已有 `Qwen3-Coder-Next L47/E471 v2 Q4_0`，只迁移到统一 manifest 并逐 payload
复核哈希。该步骤不需要新增大权重，最适合验证 loader/契约。若主线希望验证新提取路径，则第二选择是
`qwen3-coder-next-l44-dspark-core-bf16`，但必须先固定 commit revision 和 recurrent state ABI。

输出头和 GLM 胶囊当前不建议批准：输出头尚需确定 token-logit 合并策略；GLM 尚无通过留出门的深层
坐标桥，且整块 shared expert 直接替换已有能量与方向失配证据。

## 主线批准必须明确给出

1. `capsule_id` 与唯一 capsule 类型。
2. 不可变 source commit/revision；本地 GGUF 则给出是否批准一次全文件顺序 SHA-256。
3. tensor 范围（单专家、top-k bank、单层 core 或整末层）。
4. payload dtype/量化格式及是否允许转置。
5. 坐标桥与 token map 的固定 SHA-256。
6. 允许的最大 source bytes、输出 bytes、峰值 RAM 与 I/O 预算。
7. approval id，建议格式：

```text
APPROVE_CAPSULE_EXTRACTION <approval-id> <capsule-id> <pinned-revision> <payload-dtype>
```

在上述字段齐全前，`capsule.example.json` 保持 `status=dry_run`、`approved=false`，实验室等待主线批准。
