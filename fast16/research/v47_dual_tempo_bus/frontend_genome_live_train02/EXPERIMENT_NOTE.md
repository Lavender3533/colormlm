# 真实 v38 冷启动 Genome 说明

- v38 在有界角色 GBNF 下用 89 completion token、7.51 秒生成完整 Genome。
- 五个组件、四个动作、两个响应式字段均由模型一次选对。
- 模型把 `layout` 选为 `split`；语义校验发现它与其余五个组件唯一兼容的布局为 `editorial`，本文件
  在编译前显式修正这一字段。
- 因此该页属于“模型 Genome + 一字段确定性语义修复 + 通用编译器”的 train 开发证据，不能写成
  Parallel Genome Head 或纯 v38 的能力结果。
