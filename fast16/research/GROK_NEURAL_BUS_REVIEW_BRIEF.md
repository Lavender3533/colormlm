# Grok只读审稿任务：ColorLM Neural Bus

## 角色

你是独立研究审稿人，不是实现代理。本轮不要修改文件、启动模型、下载权重或重复跑基准。
目标是找出当前方法为什么只能局部修正、为什么v2双胶囊路由回归，并给出一个可证伪的下一步。

## 必须先读

1. `PROJECT_STATE.md`
2. `fast16/COLORLM_NEURAL_BUS.md`
3. `fast16/research/NEURAL_BUS_V1_SPEC.md`
4. `fast16/research/neural_bus_v1_report.json`
5. `fast16/research/neural_bus_v2_capsule_build_report.json`
6. `fast16/research/neural_bus_v2_report.json`
7. `llama.cpp/src/llama-neural-bus.{h,cpp}`
8. `llama.cpp/src/models/qwen35moe.cpp`中Neural Bus相关代码

## 已知事实

- v1：单颗Coder-Next专家471胶囊，Code8为3/3，已晋级。
- v2：增加专家0、供体路由行和`primary/secondary/no-op`硬竞争，真实运行成功，但Code8退回2/3。
- v2的embedding标签只验证坐标运输，不是第12/28层能力标签；拟合bias在留出集上更差，已拒绝。
- 专家0目前只是候选对照，没有证据证明它与专家471能力互补。
- 不能用推理时不可获得的“真实错误率”作为路由输入。

## 需要回答的问题

1. 用严格数学解释v1偶然修正`close_elements`而v2又回归的最可能原因，并列出可区分这些原因的最小观测。
2. 设计一个30秒墙钟预算内的反事实标注方法：对同一参考token分别强制`no-op/e0/e471`，用下一token NLL产生路由标签。
3. 给出在llama.cpp/GGML中采集第12、28层少量连续特征的最低开销方案，禁止文本关键词和主机端任务分类。
4. 判断下一版应使用硬top-1、三路softmax还是带no-op的稀疏门，并说明关闭等价如何保持。
5. 搜索并引用最接近的公开论文或成品：模型嫁接、冻结专家路由、adapter fusion、MoE路由蒸馏、activation steering、test-time routing。区分真正相同与仅表面相似。
6. 给出一个明确的晋级门：哪些数值通过才继续，哪些结果出现就立刻停止。

## 输出格式

- 先写不超过10条的审稿结论，按严重程度排序。
- 再写一个唯一推荐方案，不要罗列十种可能。
- 给出数学、所需张量、参数量、预计显存和每token额外计算量。
- 给出最多5项的最小实验，不允许长榜、重复benchmark或下载新模型。
- 每个重要判断标记为“已有证据”“合理推断”或“待验证”。
- 不要把“能加载”“输出改变”或单题偶然改善称为能力突破。
