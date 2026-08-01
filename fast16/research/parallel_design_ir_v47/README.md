# Parallel Design Genome v47

这是一个只消费 8 条 train 的前端算法原型：固定角色槽的闭集 Design Genome 负责选择布局、组件、动作、响应式与视觉系统；copy slots 负责搬运用户专有文字；通用编译器确定性生成完整 HTML/CSS/JS。

核心入口：

- `design_ir.schema.json`：严格离线 schema；
- `design_ir.llamacpp.schema.json`：llama.cpp grammar 转换兼容 schema；
- `design_genome.gbnf`：无 `*`/`+`、固定槽位、单行输出的冷启动 grammar；
- `decode_genome.py`：输出解析、规范化、重复抑制和截断续写状态；
- `compile_design_genome.py`：通用组件编译器；
- `teachers/`：8 条 train Genome 与 prompt copy slots；
- `GENOME_HEAD.md`：terminal-hidden 并行分类头的数据和训练目标；
- `selftest.py`：不启动模型、不联网、不使用 GPU 的静态自检。

快速自检：

```powershell
python fast16/research/parallel_design_ir_v47/selftest.py
```

单页编译：

```powershell
python fast16/research/parallel_design_ir_v47/compile_design_genome.py fast16/research/parallel_design_ir_v47/teachers/pf47-train-01.genome.json --slots fast16/research/parallel_design_ir_v47/teachers/pf47-train-01.slots.json --output fast16/research/parallel_design_ir_v47/compiled/pf47-train-01.html --report fast16/research/parallel_design_ir_v47/compiled/pf47-train-01.compile.json
```

范围与未证明项以 `HANDOFF.md` 为准。不要用本目录工具读取或运行 validation/blind。
