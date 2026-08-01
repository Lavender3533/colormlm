# 北极星 M1：原生连续皮层净增益

状态：在任何真实 donor 质量结果产生前，被原生稀深 S14 路线取代。本文保留为跨模型 portal
诊断合同，不再是当前质量主脑的晋级合同。活跃冻结规格见
`quality_architecture/architecture_spec.json`，完整理由见
`quality_architecture/ARCHITECTURE.md`。

## M1 要证明什么

在不把小模型换成新主干、不做大规模蒸馏/LoRA/epoch 训练的条件下，把一个 300B+ 开源 donor
的连续原生层段接入北极星；候选在冻结盲测上比当前正式 v38 有可归因净增益，并在用户机器上
保持至少 20 token/s。

M1 不是“已经追上 Claude/GPT”。它必须第一次证明新架构能从旗舰开源模型取得真实、可复现、
非单题的能力增量，同时不牺牲到不可用速度。后续多个这样的增量才构成追赶曲线。

## 固定候选

- 快主干：`ColorLM-v38-Qwen36-Shared-Sequence-Policy`；
- 首供体：`deepseek-ai/DeepSeek-V4-Flash-0731` 固定 revision；
- 首器官：L40--L42 连续末端皮层，包含实际所需 attention、mHC、norm、router、shared path 和
  route trace 命中的 routed expert 页；
- 门户：北极星 2048 ↔ DeepSeek 4096 成对原生激活的闭式 residual portal；
- 调度：默认 no-op，只允许隐藏态难度触发；关键词、任务名和主机分类禁止进入在线路由。

若官方权重/源码证明 L40--L42 无法形成可执行连续边界，可以在看到质量结果前改站位一次；改动、
理由和新哈希必须先写入报告，不能看完 blind 再挑层。

## 必须同时通过的证据

### 1. 供体与采集真实性

1. donor 实计或官方总参数不少于 300B；
2. 在供体原生坐标中完成真实前向，不用 embedding 映射伪造内部 hidden；
3. 至少 12 个互不重复任务族、每族多个关键 token；按完整任务分割；
4. 采集 L39--L42 hidden、mHC/state 摘要、router top-k、token ID、目标 token NLL；
5. revision、tensor、Range、SHA-256、dtype/shape 和缺失依赖全部可复核。

### 2. 状态门户门

1. 只用闭式线性代数；backprop=false、epochs=0；
2. leave-one-complete-task-out；
3. 输入/输出 portal 相对预注册 anchor 的平均 cosine 都至少 `+0.10`；
4. 输入/输出 NRMSE 比率都不高于 `0.90`；
5. 至少 80% 留出任务的输入和输出方向分别改善；
6. donor router top-k recall 至少 `0.70`，且相对 anchor 提升至少 `0.10`；
7. 通过只允许进入器官 A/B，不算能力提升。

### 3. 可归因能力门

1. 同前缀、同 seed 的强制 `{no-op, donor}` next-token NLL；
2. 多数独立任务改善；逐任务 LOTO 总方向仍为正；
3. 预先冻结的关键决策 token（推理结论、运算符、边界、工具名、参数、结束状态）整体改善；
4. 实际生成与 v38 不完全相同，差异必须由单测、工具协议或固定评分规则证明为净改善；
5. 任何能力类别不得出现两个以上净回归；不得用平均值掩盖一个类别崩坏。

### 4. 冻结 blind 质量门

使用至少 24 题、覆盖推理/知识/长上下文/代码/工具/规划/电脑操作/对话的全新 blind：

- 相对 v38 至少净胜 4 题；
- 严重回归为 0；
- 任一类别净回归不得低于 `-1`；
- 判分器、单测与人工规则在看到候选输出前冻结；
- 同时保留 Claude/GPT 参考列，但不能因暂时拿不到参考 API 而把 v38 净胜门删掉。

### 5. 本机速度与正确性门

1. `no-op` 物理旁路：不加载器官、不建节点，固定 seed 逐 token 等于 v38；
2. RX 5700 XT 8GB + 32GB RAM 上相邻 A/B，解码不低于 `20 token/s`；
3. 记录首 token、prefill、decode、p50/p95、SSD/RAM/PCIe/VRAM 字节、cache hit、stall；
4. 不允许串专家、未完成 fence 的槽可见、pipeline staging 覆盖或残页静默进入；
5. 50 token/s 是优化上沿，不是 M1 最低门；达到 20 后优先继续提高质量。

## 立刻停止条件

- 用单颗专家非零输出宣称取得 DeepSeek 能力；
- 没有原生 hidden/route trace 就下载或猜 top-k expert；
- portal 几何门失败后扫描 alpha 挽救；
- blind 失败后换题、换判分或反复消费 blind；
- 速度低于 20 token/s 仍把版本设为正式；
- 用工具外壳、网页编译器或 API 调度成绩冒充模型本体追上 Claude/GPT。

## 当前完成度

- 300B+ donor 审计：完成；
- K3 机械微块路线：已证伪并停止；
- 专家页目录/RAM 缓存：原型完成；
- 闭式原生状态 portal 编译器：实现并通过合成自检；
- DeepSeek 原生流式 hidden/route 采集：进行中；
- 真实 portal、连续器官、NLL、blind、20 token/s：均未完成。
