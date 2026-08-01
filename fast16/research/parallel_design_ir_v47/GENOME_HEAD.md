# terminal-hidden → Design Genome 并行分类头

## 结论

当前主算法不再把 270–315 字节 Genome 当作自回归文本让小 GRU 逐 token 生成。GBNF 和 llama.cpp schema 只服务于教师制作、冷启动与故障诊断；最终路径从每个任务唯一一次、尚未看到答案前缀的 2048 维 terminal hidden，同时预测所有固定角色槽。

## 输入记录

机器契约见 `sequence_record.schema.json`。每条任务必须包含：

- `group_id/template_cluster_id/split`，拆分单位是完整任务与模板族；
- 原始 UTF-8 prompt 及其 SHA-256；
- prompt 内 2–64 个 copy 候选，每个候选带 UTF-8 字节起止位置和原文；
- 一份 `[2048]` initial terminal hidden，`initial_only=true`、`answer_prefix_tokens=0`；
- 结构化 `slot_targets`，以及仅供审计/GBNF 回放的规范单行 `target_text`。

不能把 teacher-forced 答案各 token 的 hidden 当独立样本。那会把答案前缀泄漏给头，也会人为把 8 个任务膨胀成数百条伪样本。

## 固定输出头

所有头读同一共享表示 `u = GELU(W·LN(h_terminal))`，默认 `d=128`：

- copy pointer：`title`、`lede` 两头，对请求内候选 span 做 masked softmax；
- visual：`mode/palette/density/shape`；
- layout：`grammar/mobile/breakpoint_profile`；
- component：`primary/controls/content/detail/support` 五头，各自只有角色合法词表；
- action：`data/view/commit/state` 四头；
- responsive：`main/overlay` 两头；
- `a=255` 与 `z=inline` 第一版固定，不训练无信息量常量头。

按当前最大 64 个 copy 候选计，类别 logit 总数约 251。`2048→128` 共享投影、LayerNorm 和所有分类头合计约 0.30M F32 参数，约 1.2 MiB；即便增至 `d=256` 也约 0.59M 参数。一次前向即可得到 Genome，不存在 88-token 自回归延迟、重复尾巴或中途截断。

copy pointer 不把候选文字加入闭集词表。候选可由主机在 prompt 中抽取标题短语、专名、数字和要求片段，并用 prompt token hidden 池化得到 `k_j`；指针分数为 `uᵀQk_j`。现有 8 条教师只有固定的 title/lede 正例，尚不足以证明自动候选抽取或 pointer 排序，正式训练前必须加入同 prompt 的干扰候选。

## 训练目标

主损失是逐槽交叉熵，不使用 token-level teacher forcing：

```text
L_slot = Σ_s w_s CE(p_s, y_s)
L_copy = CE(p_title, j_title) + CE(p_lede, j_lede)
L_valid = -log Σ_{g∈Valid(layout,role)} p_role(g)
L_pair  = Σ compatibility_penalty(component, action, responsive)
L = L_slot + 1.5 L_copy + 0.25 L_valid + 0.10 L_pair
```

训练时用教师 layout 给角色 mask，同时对预测 layout 计算 `L_valid`，避免只在 teacher 条件下合法。推理时每头取 top-k，在很小的笛卡尔积上按总负对数概率寻找通过 `ir_core.validate_ir` 的最低成本组合；找不到时返回显式 no-op/拒绝，不由编译器暗改语义。

8 条 train 只能用于实现与过拟合冒烟，不能选 hidden width、正则或阈值后再把同一数据称作泛化。准入前至少需要同 schema 的新模板族，并按 `group_id + template_cluster_id` 整组留出。

## 推理与编译边界

1. 主机从 prompt 建 copy 候选，不生成新文案。
2. Genome Head 一次并行预测固定槽。
3. 合法组合投影只删除非法组合，不按题目 ID 或关键词补组件。
4. `compile_design_genome.py` 展开通用组件目录、CSS/JS、焦点/ESC/live-region/断点保障。
5. 任务专有 title/lede 只由 copy slot 注入。

因此模型贡献是选择基因与 copy 引用；编译器贡献是确定性结构和行为；组件目录内的示例数据/标签属于显式硬编码资产。三者必须分别评估。
