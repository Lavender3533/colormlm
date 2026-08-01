# Demand-Routed Synaptic Paging

## 目标

让路由器常驻GPU，只让当前任务真正需要的专家权重进入GPU或RAM。冷专家保持在SSD，
从而让8GiB显存和32GiB内存承载更大的总专家库。该机制负责容量和速度，不把缓存命中
伪装成模型智力提升。

## 当前事实

- ColorLM v6有40层，每层256个路由专家，每个token选择8个专家。
- GGUF中的每个专家由gate、up和down三块连续量化权重组成。
- `llama-slot-pool.*`已有GPU槽位、logical-to-slot LUT、同步stage、频率观察和LRU雏形。
- 该槽池目前只接入`qwen3moe.cpp`，没有接入ColorLM使用的`qwen35moe.cpp`。
- 当前miss会错误回落到slot 0，因此不保持输出正确性，不能用于主模型。
- 当前源专家仍来自完整CPU张量，不是真正的SSD冷页。

## 三层驻留

1. GPU hot：每层K个高频专家，直接运行Vulkan矩阵内核。
2. RAM warm：最近使用专家的压缩页，用pinned buffer异步上传。
3. SSD cold：GGUF专家页和索引，按需读取，不保留完整供体模型进程。

专家内部的512个中间神经元是第二级页。先完成专家级分页；只有真实路由统计表明专家
内部也有稳定稀疏性时，再把gate/up行和down列按64或128神经元组成子页。

## 正确性约束

- miss不能映射到其他专家。
- 缓存开关关闭时，输出必须与原始GGUF逐token一致。
- miss必须走CPU原专家或等待正确专家上传，不能静默近似。
- 压缩、淘汰和驻留策略分离；先可逆迁移，再决定是否降低冷页位宽。

## 第一版实现顺序

1. 把`slot_pool`从`qwen3moe`抽成架构无关组件并接入`qwen35moe`。
2. 增加exact miss path：GPU命中专家走槽池，miss走原CPU权重并合并结果。
3. 为每层记录命中率、miss字节、上传等待和专家频率，不先做预测。
4. 使用GGUF mmap偏移作为SSD页源，移除`--no-mmap`下的完整CPU专家常驻要求。
5. 加入下一层专家转移矩阵和异步预取；预测失败仍走exact miss path。
6. 只有命中率和延迟达标后，才尝试神经元子页和冷页更低位宽。

## 验收门

- 8GiB显存下不降低Code8和工具调用正确率。
- 固定seed贪心输出与无分页基线一致。
- RAM峰值、VRAM峰值和SSD读取字节均可观测。
- 95%命中率下的生成速度不低于当前v6的90%。
