# ColorLM v16 Coder Neural Block

## 结论

v16采用Qwen3-Coder-Next第47层作为第一颗完整Coder神经块。该层与ColorLM主干同为2048维、
16个Q头、2个KV头和256维head，因此张量接口、KV宽度和Vulkan算子均已有实现基础。它不是把
不同模型按层号直接交替拼接，而是一条带输入/输出坐标桥和可关闭残差的供体支路。2026-07-30
审计发现首版没有实现真正独立KV，v16.1已改为微批次内局部因果注意力；独立有界KV属于v17。

机器可读契约：`fast16/research/v16_coder_neural_block_plan.json`。

## 精确结构

供体Qwen3-Coder-Next共有48层，每四层为三层Gated DeltaNet和一层全注意力。选择L47是因为它是
最后一颗全注意力层，结构为：

```text
ColorLM L35 hidden (2048)
  -> input transport (2048 x 2048)
  -> donor RMSNorm
  -> donor full attention (NeoX RoPE layout)
       Q: 16 x 256, KV: 2 x 256
       donor RoPE theta: 5,000,000
       v16.1: current-ubatch causal attention, no host KV writes
  -> attention residual
  -> donor post-attention RMSNorm
  -> donor shared expert + routed MoE (512 experts, exact top-10)
  -> output transport (2048 x 2048)
  -> energy gate * alpha
  -> add to native ColorLM hidden
```

主干的RoPE theta是10,000,000；供体块必须保留自己的5,000,000，不能复用主干角度。输入token位置
相同，但旋转频率属于块内部契约。

## 可行性证据

- `llama.cpp`已经实现Qwen3Next完整全注意力、Q/K Norm、query gate、RoPE和MoE图。
- ColorLM主干和供体的hidden、Q/KV head数量及head dim一致。
- ColorLM L35原生走recurrent/ColorKernel状态。首版实际调用通用`build_attn`并以L35作为KV层号，
  这会在主干注意力缓存中访问未独立注册的供体状态；v16.1不再调用缓存型注意力，也不写主干KV。
- 不选L39是因为主干最后一层会在FFN前裁成输出token；L35仍保留完整prompt token，可建立正确供体KV。
- `alpha=0`时不加载供体权重、不把L35注册进Attention KV、不创建供体图节点，保持v13物理旁路。
- 远端1549个L47张量在两个分片中各自连续，可合并成两个HTTP Range，不需要下载80B完整模型。

## 资源预算

| 项目 | 精确或估算值 |
|---|---:|
| L47张量 | 1549 |
| BF16完整块 | 3,284,153,344 bytes / 3.059 GiB |
| BF16非路由专家部分 | 60.0 MiB |
| BF16路由专家部分 | 3.000 GiB |
| Q4_0完整块估算 | 0.860 GiB |
| 每token激活的10个BF16专家 | 60.0 MiB |
| 远端合并Range | 2 |

第一版最终权重约在v13的12.967 GiB基础上增加0.86 GiB以及两张运输矩阵，仍远低于用户允许的
15--30 GiB；后续可增加多个完整块，而不是继续堆未经证明的小专家。

## 不能省略的部分

1. 输入/输出运输必须分开。旧的共享token正交矩阵只能作为初始化，不能当作层坐标已经对齐的证据。
2. Attention必须有独立KV，不能与原生状态复用。v16.1先采用无跨批缓存的局部因果注意力止血；
   v17四层岛必须实现独立、有界、可回收的供体KV，不能把局部模式冒充完整长期注意力。
3. 路由必须执行供体原生top-10。v16首版先让约0.86GiB Q4_0块常驻设备；SSD专家分页是后续
   性能阶段，未命中时必须等待正确页或整块no-op，不能替换成槽位0。
4. 第一版先强制整块残差，不训练语义路由。只有整块本身产生稳定正收益后才学习no-op门。
5. 不用“输出发生变化”作为晋级证据。最小门是短反事实NLL、代码/工具任务和人工项目体验。

## 实现顺序

1. 完成两个Range的断点提取，并从BF16段直接编译Q4_0块包。
2. 首版曾复用主干KV层位；v16.1已删除该共享写入，改为当前微批次内因果注意力。
3. 复用`qwen3next.cpp`的full-attention算子，读取Neural Block权重和独立RoPE参数。
4. 接入Norm、shared expert和512专家精确top-10计算；首版不冒充已实现SSD分页。
5. 在L35输出处通过坐标桥和能量门加入残差；残差严格为
   `T^T(block(T*h)-T*h)`，`alpha=0`继续逐token等价v13。
6. 只做一次最短因果NLL门和人工Claude Code任务。整块无收益时先修坐标桥，不盲加第二块。

## 后续架构头脑风暴

单块证明有效后，v17优先形成一个完整的四层Coder岛：三层Gated DeltaNet负责持续状态，一层完整
Attention负责跨文件检索，四层共享同一输入/输出坐标边界。这样供体内部坐标连续，只有岛的两端
需要运输，比四个孤立层分别桥接更可靠。再下一步才加入1--4轮共享权重规划循环和隐藏态验证头。

## 2026-07-30 运行与Claude Code状态

- 独立Vulkan服务运行于`http://127.0.0.1:8104/v1`，完整L47块占896.89MiB Vulkan显存。
- Anthropic转换层会把顶层与消息列表中的system内容归并到唯一首条system；重复system短请求和
  隔离Claude Code真实短对话均已通过。
- 工具协议可以返回`tool_use`，但首次短验证出现必填参数为空，工具结果回合随后没有及时结束。
  这属于当前模型/工具回路的能力缺口，不是神经块加载或Anthropic 500；修复前只验收对话入口。

## 2026-07-30 v16.1缓存隔离与输出止血

- 源码审计确认首版L47通过通用缓存型`build_attn`写入L35层位，并没有形成文档声称的独立KV。
  该写入会污染后续token的主干注意力坐标，是长生成变慢和行为漂移的高优先级根因。
- v16.1将供体注意力改为当前ubatch内的因果注意力，保留Q/K Norm、NeoX RoPE、query gate、
  shared expert和512专家top-10，但完全不读写主干KV。真正跨token的独立滑窗KV移入v17。
- Anthropic兼容层新增`COLORLM_ANTHROPIC_MAX_TOKENS`，启动器默认1024，客户端请求32K/64K时
  由本地硬上限接管；普通OpenAI接口的输出预算不受影响。
- 8104单实例验证：2-token `OK`为3.32秒；128-token Rust生成实际解码`14.36 token/s`；
  4,428-token提示预填充`200.56 token/s`，标记取回正确，后续7-token解码`19.53 token/s`。
- 结论仅为缓存隔离和长上下文稳定性通过。v16.1尚未证明比v6更聪明，且128-token速度仍需优化。
