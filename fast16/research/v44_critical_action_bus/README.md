# ColorLM v44 Critical Action Bus

## 当前进度

v44的第一项是先证伪v43为什么离线很好、真实生成却为零收益。已在真实v36词表上完成120条目标的
UTF-8字节级token span审计：

```text
critical token occurrences:             2654
outside the first-six-token window:      2274
tasks affected:                          120/120
argument tokens outside the window:      1324/1324
tool-name subtokens outside the window:  210/390
finish-field subtokens outside window:   740/940
```

这不是参数强度问题，而是监督目标错位：v43每题只釆集目标的前6个token，而参数字段最早要到index 9才出现。

## 下一个可证伪假设

只采集并学习完整轨迹中的判别性动作token：

1. 动作前缀：调工具、询问或结束。
2. 工具名的语义子token，排除引号、冒号和逗号。
3. 关键参数名与可从状态推出的值。
4. 结束JSON中真正决定任务是否完成的字段。

如果这些关键span的离线改善仍不能在全新模板簇上带来真实净修复，就停止稀疏logit策略头路线，改做可训练的
小型序列解码器/能力岛，不再用NLL局部好看结果延长实验。

## 科学约束

- v43 test已消费，只用于失败归因，不用于v44调参或晋级。
- v44必须新建跨模板簇的blind。
- 只有完整轨迹净修复可以晋级，NLL只是进入运行时门的资格。
- 不用文本关键词、主机任务分类或生成后自评作为在线路由。

## 产物

- `audit_critical_spans.py`：启动单个v36服务，用 `/tokenize?with_pieces`做字节级span映射。
- `critical-span-audit.json`：120条任务的完整token位置和汇总证据。

## 2026-08-01 v44 冻结开发门结果

- 408/408条关键span teacher完成精确NLL与terminal hidden/full logits采集，CNOB为
  `408,652,800`字节，test泄漏为0。
- 冻结的rank16/ridge 0.1/strength 12拟合结果为validation的rescue/regression均为0，
  平均NLL delta为0，exact no-op率为100%，开发门失败。
- 更重要的先验检查：validation在冻结28个候选token内原生已答对94/98，最多只有
  4个可能rescue，而合同要求至少8个。因此这个门同时暴露了“候选集内top-1”目标与真实全词表
  生成错位，不得通过扫strength挑选validation来补救。
- 按冻结合同停止v44稀疏单token头。保留关键span数据和捕获，下一步转向完整序列解码器/
  能力岛；在动手前先用全新跨模板validation一次性筛选已有完整Qwen3.6与GLM-4.7底座。
