# Polaris Cognitive Substrate 最小门

这不是能力评测，也没有使用模型。它只验证“思维图”是否比一次性路由多出
可执行的状态语义：

1. Java/PHP 候选同时保留，证据不足时不选。
2. 现实证据只更新依赖它的假设。
3. PHP 获得完整支持后才 commit。
4. Composer 真实失败后回滚 PHP 决策，但保留用户目标和视觉设计。
5. JVM 环境后续就绪时，原有 Java 候选可直接获得支持并 commit。

这能证明局部重写、晚绑定和局部回滚合同可实现；不能证明它会思考、
拥有意识、比 Transformer 更好，或已经是新模型。

```powershell
$env:PYTHONUTF8='1'
python fast16/research/polaris_cognitive_substrate/minimal_thought_graph.py
```

## 真实权重岛出生—修剪门

`live_birth_prune_gate.py` 只在外部验证器产生失败回执后启动 v17 Qwen
L44–L47 连续权重岛，使用岛产生修复，再用受限 AST 和7个外部用例验证。
无论候选是否通过，临时岛进程都必须终止。

```powershell
$env:PYTHONUTF8='1'
python fast16/research/polaris_cognitive_substrate/live_birth_prune_gate.py
```

它是进程级出生/修剪原型，不是单运行时内的原生动态计算图。结果与边界见
`LIVE_GATE_RESULT_20260804.md`。

## Embryo v0：优先复用 v38 + v47

`polaris_embryo_v0.py` 是第一条整体纵切面，但刻意不重新生成模型产物：

1. 校验现有 v47 Parallel Genome Head、字段本体和负约束的发布 SHA，并把它注册为条件器官；
2. 把 v47 方法下由 v38 生成的历史 Design IR 作为兼容规划 packet，不冒充 Genome Head 输出；
3. 消费现有 `ColorLM-v38-Qwen36-Shared-Sequence-Policy` 加确定性尾部修复的前端候选；
4. 重新执行确定性静态 Helix 门，并消费已有浏览器动作回执；
5. 验证通过直接版本提交，不出生能力岛；失败时只发出出生请求并停止短门。

```powershell
$env:PYTHONUTF8='1'
python fast16/research/polaris_cognitive_substrate/polaris_embryo_v0.py
```

本路径不下载、不训练、不编译、不启动 S14，也不启动模型进程。它证明 v38、
v47 兼容合同、验证和版本状态可以被同一个权威合同组装；因为使用的是冻结开发题
混合产物，commit 只提交该 HTML 夹具，不能据此声称已完成任意任务在线生成或
v47 Genome Head 能力晋级。

## 在线 v38 → v47 Genome validation 短门

`live_v38_v47_validation_gate.py` 只让现成 v38 在线生成约 88 token 的闭集
Design Genome，再由 v47 通用组件编译器展开 HTML，避免让本地模型生成上千 token
源码。它只消费冻结 `validation` 题，禁止自动修复或重试。

```powershell
python fast16/research/polaris_cognitive_substrate/live_v38_v47_validation_gate.py
```

2026-08-04 的第一次 `pf47-validation-01` 在线门在合法组合投影处失败：v38
输出了完整 Genome，但选择 `split` 布局的同时选择了 `table/drawer/dialog`，三项
都不属于该布局的允许集合。运行时随后停止，没有编译 HTML、没有改规则重跑。
真实失败回执位于 `live_v38_v47_validation_01/live_report.json`。这把当前短板定位为
布局与组件的联合结构决策，而不是 S14 式权重加载或长文本生成。
