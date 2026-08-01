# ColorLM v47 Design Genome 质量算法组交接

## 一句话状态

已把单题“短 Design IR + 尾部修复”升级为不按题目 ID/文字分支的闭集 Design Genome 原型：5 个角色组件槽、4 个角色动作槽、2 个角色响应式槽与视觉/布局枚举由模型选择，用户文字走 copy slots，通用编译器确定性展开完整 HTML/CSS/JS。8 条 train 的静态契约为 `8/8`；没有训练 Genome Head，没有启动模型/GPU，没有读取或运行 validation/blind，因此不能宣称模型能力或泛化晋级。

## 真实完成项

### 1. 从自由文本 IR 收缩为固定角色 Design Genome

教师目标是规范单行 JSON，字段顺序稳定：

```text
q: title/lede copy 引用
y: mode/palette/density/shape
l: layout/mobile/breakpoint profile
c: primary/controls/content/detail/support
x: data/view/commit/state
r: main/overlay
a: 固定无障碍位图 255
z: 固定 inline 资产
```

8 条 Genome 为 `270–315` UTF-8 字节，均在新增优选区间 `150–600` 内，平均 `289.62` 字节；1200 仅保留为恢复器硬上限。每条 sidecar 只有 title/lede 两段 copy 文本，自检会校验 prompt SHA-256，并要求文本是对应 train prompt 的连续原文片段。用户专有词不进入闭集分类词表。

教师文件位于 `teachers/*.genome.json` 与 `teachers/*.slots.json`。严格离线结构是 `design_ir.schema.json`，copy 数据是 `copy_slots.schema.json`，完整 hidden 记录格式是 `sequence_record.schema.json`。

### 2. 有界 GBNF、llama.cpp 兼容 schema 与规范解析器

主线提供的真实 v38 证据揭示了三个边界：

- 原严格 schema 的多个 `items:false` 会让 llama.cpp 返回 HTTP 400：`Unrecognized schema: false`；
- 即使删掉布尔 schema，JSON Schema 路径会漂亮打印并在 256 token 截断；
- 无界 GBNF 的 `component*` 会重复到截断，而固定槽单行 GBNF 曾以约 205 字节、88 completion tokens、8.58 秒正常 stop。

因此本目录同时保留：

- `design_ir.schema.json`：离线严格版，可使用 2020-12 tuple 与 `items:false`，不直接发给 llama.cpp；
- `design_ir.llamacpp.schema.json`：请求兼容版，采用 draft-07 `items:[...]` tuple，并对每个 tuple 设置相同的 `minItems/maxItems`；没有 `prefixItems`，也没有作为 `items` 的布尔子 schema；
- `design_genome.gbnf`：固定 5/4/2 角色槽，组件每槽独立枚举，整份 grammar 不含 `*` 或 `+`；输出被强制为单行；
- `decode_genome.py`：提取首个完整合法对象、忽略包装/重复对象、补最后缺失括号；若在语义槽中途截断，只返回 `needs_resume + resume_prefix`，不猜缺失组件。

本地 llama.cpp 源码的 converter 对 tuple `prefixItems` 有单测，但项目 `grammars/README.md` 仍明确标为 broken；而 C++ 实现同时出现 `items` 与 `prefixItems` 时优先取 `items`。因此生产兼容版不使用 `prefixItems`。自检已用本地 `json_schema_to_grammar.py` 成功转换兼容版，输出 grammar 为 4378 字节。

GBNF 仍只是冷启动/教师制作接口。主线后续应训练一次前向的并行 Genome Head，避免把 88 token 再变成运行时解码成本。

### 3. 角色、组合与行为的编译前合法性

`ir_core.py` 不只检查 JSON 类型，还执行：

- 组件是否属于当前位置的 `primary/controls/content/detail/support` 词表；
- layout 是否接受该位置的组件族；
- `data/view/commit/state` 动作是否有真实组件承接；
- `main/overlay` 响应式变换是否有对应组件；
- 组件对是否重复、copy 引用是否存在、扩展是否偷渡自由字符串。

这直接针对真实样本里“杂志题选 sidebar.docs/filters.status、动作 sort/copy”的语义漂移。`CATALOG_PROMPT.md` 提供紧凑语义先验，但提示本身不是安全边界；最终拒绝由结构 validator 完成，编译器不会静默补成另一种页面。

### 4. 通用组件编译器

`compile_design_genome.py` 只分派 component family/variant，不读取 task ID、prompt 或题目标题。组件目录在 `component_catalog.json`。编译器统一保障：

- 唯一 HTML/head/body、语义地标、闭合标签、UTF-8；
- 自包含 CSS/JS，无外部依赖；
- focus-visible、skip link、live region、表单 label/error 关联；
- ESC 关闭、焦点恢复、overlay Tab 环、减少运动；
- 筛选、排序、预览/抽屉、表单验证、复制、标签页、开关、收藏、年份图与状态播报；
- 断点 profile 与 `table>cards/grid>stack/drawer>full/aside>drawer/schedule>accordion/chart>scroll/code>scroll` 的确定性实现。

8 条 train 的 HTML 为 `18,428–21,235` 字节；HTML/Genome 展开比分别为：

```text
train-01 75.301x   train-02 67.880x   train-03 62.468x   train-04 66.490x
train-05 67.149x   train-06 70.439x   train-07 61.235x   train-08 77.074x
均值 68.505x
```

现有冻结静态评分器下，8 条 train 分数为 `91.0/91.5/91.5/92.0/91.5/91.5/91.5/92.5`，模板惩罚均为 0，全部 critical 与逐维阈值通过。报告是 `SELFTEST_REPORT.json`，逐页编译报告在 `compiled/`，失败边界在 `FAILURE_REPORT.json`。

这些高分主要证明组件编译器覆盖了静态契约，不能归为模型质量；静态评分也不执行 JS、不栅格化、不测对比度。

### 5. terminal-hidden 并行 Genome Head 契约

`GENOME_HEAD.md` 给出了完整数据与训练目标。每任务只允许一份、答案前缀为 0 的 `[2048]` initial terminal hidden。默认共享投影 `2048→128`，再并行预测 copy、视觉、布局、5 个组件角色、4 个动作角色和 2 个响应式角色；最大 64 copy 候选时约 0.30M F32 参数、约 1.2 MiB。

训练使用逐槽 CE、copy pointer CE、非法质量损失与组合一致性损失，不使用 token teacher forcing。推理对各头 top-k 做小规模合法组合投影；没有合法组合就拒绝/no-op，不由编译器猜答案。

## 模型贡献、编译器贡献与硬编码边界

| 类别 | 本原型中的责任 | 不允许的归因 |
|---|---|---|
| 模型/Genome Head | 选择视觉、布局、5+4+2 角色槽和 copy 引用 | 当前尚未训练，不能把教师 Genome 或编译页得分写成模型得分 |
| 编译器 | 组件展开、页面闭合、通用 JS、响应式、键盘/焦点/ESC/live region、无外链 | 这是确定性程序贡献，不是模型学会了 HTML/JS |
| 显式硬编码 | `component_catalog.json` 的通用组件标签、示例数据和行为实现；8 条教师基因选择与 copy 标注 | 目录没有 task ID/标题分支，但组件库仍是硬编码资产，必须如实报告 |

编译器源码自检会扫描 8 个 task ID/标题，当前无命中。组件目录含“文档、商店、运维、日程”等可复用域组件，这是有意的闭集库，不伪装为神经能力。

## llama.cpp 真实请求格式

推荐 GBNF 模式。先准备无 BOM UTF-8 的 `prompt.txt`，再构造请求：

```powershell
python fast16/research/parallel_design_ir_v47/build_request.py --mode gbnf --model ColorLM-v38-Qwen36-Shared-Sequence-Policy --prompt-file prompt.txt --slots fast16/research/parallel_design_ir_v47/teachers/pf47-train-01.slots.json --output fast16/research/parallel_design_ir_v47/request.gbnf.json
curl.exe -sS http://127.0.0.1:8138/v1/chat/completions -H "Content-Type: application/json" --data-binary "@fast16/research/parallel_design_ir_v47/request.gbnf.json"
```

生成器写出的关键请求字段精确为：

```json
{"model":"ColorLM-v38-Qwen36-Shared-Sequence-Policy","messages":[{"role":"system","content":"<CATALOG_PROMPT + copy slots>"},{"role":"user","content":"<prompt>"}],"temperature":0,"max_tokens":160,"stream":false,"chat_template_kwargs":{"enable_thinking":false},"grammar":"<design_genome.gbnf 原文>"}
```

兼容 schema 只用于链路回归，不是推荐质量路径：

```powershell
python fast16/research/parallel_design_ir_v47/build_request.py --mode schema --model ColorLM-v38-Qwen36-Shared-Sequence-Policy --prompt-file prompt.txt --slots fast16/research/parallel_design_ir_v47/teachers/pf47-train-01.slots.json --output fast16/research/parallel_design_ir_v47/request.schema.json
```

其关键字段采用 llama.cpp 当前文档格式：`response_format.type=json_schema`，`response_format.schema` 是解析后的 `design_ir.llamacpp.schema.json` 完整对象，不是路径字符串。`build_request.py` 会原样内联整个对象，避免手工拼接遗漏 tuple 约束。

本组没有发送上述请求、没有启动 v38；这里只把主线已发现的真实兼容性边界编码进原型和自检。

## 精确轻量复现命令

在项目根 `D:\project\大模型ssd化`：

```powershell
python -m py_compile fast16/research/parallel_design_ir_v47/ir_core.py fast16/research/parallel_design_ir_v47/decode_genome.py fast16/research/parallel_design_ir_v47/compile_design_genome.py fast16/research/parallel_design_ir_v47/build_request.py fast16/research/parallel_design_ir_v47/selftest.py
python fast16/research/parallel_design_ir_v47/selftest.py
python llama.cpp/examples/json_schema_to_grammar.py fast16/research/parallel_design_ir_v47/design_ir.llamacpp.schema.json
python fast16/research/parallel_design_ir_v47/compile_design_genome.py fast16/research/parallel_design_ir_v47/teachers/pf47-train-01.genome.json --slots fast16/research/parallel_design_ir_v47/teachers/pf47-train-01.slots.json --output fast16/research/parallel_design_ir_v47/compiled/pf47-train-01.html --report fast16/research/parallel_design_ir_v47/compiled/pf47-train-01.compile.json
python fast16/research/parallel_design_ir_v47/decode_genome.py fast16/research/parallel_design_ir_v47/teachers/pf47-train-01.genome.json --output fast16/research/parallel_design_ir_v47/compiled/pf47-train-01.canonical.json --report fast16/research/parallel_design_ir_v47/compiled/pf47-train-01.decode.json
```

`selftest.py` 明确只调用 `load_split("train")`；不要改成 `check_all()`，后者会读取 validation/blind。

## 未证明项

- 没有真实 initial hidden，Genome Head 尚未实现/训练/留出评估；0.30M 只是结构预算。
- 8 条教师对固定槽分类仍极小，训练集精确率没有泛化意义。
- copy 候选当前人工标注；自动 span 提取、干扰候选、未见专名 pointer 尚未证明。
- 8 页只过静态契约，没有浏览器功能、375/768/1440 截图、控制台、焦点顺序、对比度或视觉盲评。
- 组件目录的域覆盖只有当前 8 类；未知页面需要版本化扩充 registry，不能通过自由字符串绕开闭集。
- llama.cpp 兼容 schema 只通过本地 converter；真实 v38 已知有漂亮打印/截断风险，GBNF 才是当前冷启动推荐路径。
- 没有读取或运行 validation/blind，不能给出任何留出结论。

## 下一步准入门

1. 只用 8 条 train 采一次 v38 initial terminal hidden，先做实现冒烟；禁止把 8/8 训练拟合写成泛化。
2. 为每条 prompt 增加真实 copy 干扰候选，验证 span pointer，而不是把 q 永久固定成 `[0,1]`。
3. 冻结 `d=128`、slot vocab、合法组合投影、copy extractor、编译器与全部哈希后，才允许碰 validation。
4. validation 必须按主线既定门：至少 6/8，通过相对模板中位增益至少 12、零关键回归；同时增加真实浏览器响应式/交互/无障碍检查。未过立即停止，不看 blind。
5. validation 通过后才允许一次 blind；blind 前禁止根据留出失败补 catalog 词、调角色或加编译规则。
6. 只有并行 Genome Head 在完整任务上形成净增益，才考虑接入 v47 慢通道；GBNF 只保留冷启动与诊断。
