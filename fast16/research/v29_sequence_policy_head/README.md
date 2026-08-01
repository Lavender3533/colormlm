# v29 Sequence Policy Head

v29承接v28的正负证据：v17末层hidden能分辨继续工具与结束，但固定全局beta的双token修正无法在
全新场景稳定压过原生logits。v29因此改成逐token、隐藏状态条件化的稀疏序列残差头。

固定结构：

```text
terminal hidden（逐样本L2归一化）
→ centered multi-output dual ridge
→ 只修正训练任务中跨至少2个独立任务重复出现的policy token行
→ 原生全词表logits + 稀疏残差
```

候选token只由train任务统计，validation/test不参与选行、拟合或参数选择。算法、lambda、目标margin、
修正裁剪和通过门均冻结在`policy_head_contract.json`；禁止在留出结果后扫描。

运行目标只限显式`tools`模态，无tools请求物理旁路；不使用关键词或主机任务分类。

## 最终结果

- 离线NLL门通过：train/validation/test平均变化分别为`-1.1428/-0.8812/-1.1522`，20/20任务
  净改善。该结果只允许进入运行门，不能单独宣称能力提升。
- 真实20题严格生成门：v17为`7/20`，v29为`9/20`。v29修复
  `policy-config-missing-list`与`policy-read-version-finish`，没有回归v17的7道正确题；测试留出仍为
  `2/5`，说明增益真实但很小。
- 原冻结八维16题：v17与v29均为`10/16`，逐题通过/失败集合完全相同，0修复、0回归。v29没有
  提升通用推理、知识、代码或电脑操作能力。
- 初版全词表零分支只有`16.28 token/s`，相对v17的`20.21 token/s`回归`19.46%`。运行图改为
  只读取/回写16个候选logit后达到`20.49 token/s`，固定工具请求输出逐字段不变，计算缓冲约从
  `1980.39 MiB`降到`1036.30 MiB`。
- 已接通请求级门控：OpenAI/Anthropic请求显式携带非空`tools`且`tool_choice!=none`时建立策略图；
  无tools请求不建立策略节点。固定无tools请求逐字复现v17，固定tools请求逐字段复现v29。
- 运行时固定`parallel=1`；当前门控不支持在同一decode batch中混合tools/no-tools多槽请求。

因此v29晋级为安全的“显式工具模态增量版”，不是通用聪明版，更不是Claude/GPT级模型。完整数字、
哈希和声明边界见`runtime-gate-report.json`。

## 使用

普通OpenAI兼容服务：

```powershell
Set-Location 'D:\project\大模型ssd化'
.\fast16\run-colormlm-v29-sequence-policy.bat
```

隔离Claude Code：

```powershell
Set-Location 'D:\project\大模型ssd化'
.\fast16\claude-v29-sequence-policy.bat
```

两者正式端口均为`8105`；研究验收服务可使用其他端口，不能与正式入口同时占用内存。
