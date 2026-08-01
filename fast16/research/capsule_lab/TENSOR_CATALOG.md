# 五类胶囊精确张量目录

本目录只记录已由本地索引/头部证明的名称、形状和字节数。`BF16` 表中的字节数是源张量大小；
GGUF 表中的 shape 是 GGML 存储维序，字节数是当前量化类型的实际 payload 大小。

## 1. 输出头胶囊

| 供体 | 精确张量名 | 来源 | shape | dtype | bytes |
|---|---|---|---:|---|---:|
| Qwen3-Coder-Next | `lm_head.weight` | post-L47 / shard 40 | `[151936, 2048]` | BF16 | 622,329,856 |
| Qwen3-Coder-Next | `model.norm.weight` | post-L47 / shard 40 | `[2048]` | BF16 | 4,096 |
| GLM-4.7-Flash | `output.weight` | post-L46 | `[2048, 154880]` | Q6_K | 260,198,400 |
| GLM-4.7-Flash | `output_norm.weight` | post-L46 | `[2048]` | F32 | 8,192 |
| Qwen3.6-35B-A3B | `output.weight` | post-L39 | `[2048, 248320]` | Q6_K | 417,177,600 |
| Qwen3.6-35B-A3B | `output_norm.weight` | post-L39 | `[2048]` | F32 | 8,192 |

胶囊总量：Coder BF16 622,333,952 B；GLM 当前 GGUF 260,206,592 B；Qwen3.6 当前 GGUF
417,185,792 B。Coder Q6_K 仅按 GGML block size 估算为 255,252,480 B，不是已生成文件。

## 2. 末端层胶囊

### 2.1 Qwen3-Coder-Next L47（Safetensors 逻辑 shape）

| 精确张量名 | shape | BF16 bytes |
|---|---:|---:|
| `model.layers.47.input_layernorm.weight` | `[2048]` | 4,096 |
| `model.layers.47.self_attn.q_proj.weight` | `[8192, 2048]` | 33,554,432 |
| `model.layers.47.self_attn.k_proj.weight` | `[512, 2048]` | 2,097,152 |
| `model.layers.47.self_attn.v_proj.weight` | `[512, 2048]` | 2,097,152 |
| `model.layers.47.self_attn.o_proj.weight` | `[2048, 4096]` | 16,777,216 |
| `model.layers.47.self_attn.q_norm.weight` | `[256]` | 512 |
| `model.layers.47.self_attn.k_norm.weight` | `[256]` | 512 |
| `model.layers.47.post_attention_layernorm.weight` | `[2048]` | 4,096 |
| `model.layers.47.mlp.gate.weight` | `[512, 2048]` | 2,097,152 |
| `model.layers.47.mlp.shared_expert.gate_proj.weight` | `[512, 2048]` | 2,097,152 |
| `model.layers.47.mlp.shared_expert.up_proj.weight` | `[512, 2048]` | 2,097,152 |
| `model.layers.47.mlp.shared_expert.down_proj.weight` | `[2048, 512]` | 2,097,152 |
| `model.layers.47.mlp.shared_expert_gate.weight` | `[1, 2048]` | 4,096 |
| `model.layers.47.mlp.experts.{i}.gate_proj.weight`，`i=0..511` | `[512, 2048]` | 每个 2,097,152 |
| `model.layers.47.mlp.experts.{i}.up_proj.weight`，`i=0..511` | `[512, 2048]` | 每个 2,097,152 |
| `model.layers.47.mlp.experts.{i}.down_proj.weight`，`i=0..511` | `[2048, 512]` | 每个 2,097,152 |

闭区间 `i=0..511` 与三个固定后缀定义了 1,536 个精确名称；加 13 个非路由张量共 1,549 个。
非路由部分 62,927,872 B，路由银行 3,221,225,472 B，总计 3,284,153,344 B。现有 Q4_0
运行包的层核心（不含两张 8 MiB transport）为 923,683,968 B。

### 2.2 GLM-4.7-Flash L46（GGUF storage shape）

| 精确张量名 | shape | dtype | bytes |
|---|---:|---|---:|
| `blk.46.attn_k_b.weight` | `[192, 512, 20]` | Q8_0 | 2,088,960 |
| `blk.46.attn_kv_a_mqa.weight` | `[2048, 576]` | Q8_0 | 1,253,376 |
| `blk.46.attn_kv_a_norm.weight` | `[512]` | F32 | 2,048 |
| `blk.46.attn_norm.weight` | `[2048]` | F32 | 8,192 |
| `blk.46.attn_output.weight` | `[5120, 2048]` | Q5_K | 7,208,960 |
| `blk.46.attn_q_a.weight` | `[2048, 768]` | Q5_K | 1,081,344 |
| `blk.46.attn_q_a_norm.weight` | `[768]` | F32 | 3,072 |
| `blk.46.attn_q_b.weight` | `[768, 5120]` | Q5_K | 2,703,360 |
| `blk.46.attn_v_b.weight` | `[512, 256, 20]` | Q8_0 | 2,785,280 |
| `blk.46.exp_probs_b.bias` | `[64]` | F32 | 256 |
| `blk.46.ffn_down_exps.weight` | `[1536, 2048, 64]` | Q6_K | 165,150,720 |
| `blk.46.ffn_down_shexp.weight` | `[1536, 2048]` | Q6_K | 2,580,480 |
| `blk.46.ffn_gate_exps.weight` | `[2048, 1536, 64]` | Q5_K | 138,412,032 |
| `blk.46.ffn_gate_inp.weight` | `[2048, 64]` | F32 | 524,288 |
| `blk.46.ffn_gate_shexp.weight` | `[2048, 1536]` | Q6_K | 2,580,480 |
| `blk.46.ffn_norm.weight` | `[2048]` | F32 | 8,192 |
| `blk.46.ffn_up_exps.weight` | `[2048, 1536, 64]` | Q5_K | 138,412,032 |
| `blk.46.ffn_up_shexp.weight` | `[2048, 1536]` | Q6_K | 2,580,480 |

合计 467,383,552 B；GGUF 中是连续末端区间。

### 2.3 Qwen3.6-35B-A3B L39（GGUF storage shape）

| 精确张量名 | shape | dtype | bytes |
|---|---:|---|---:|
| `blk.39.attn_k.weight` | `[2048, 512]` | Q8_0 | 1,114,112 |
| `blk.39.attn_k_norm.weight` | `[256]` | F32 | 1,024 |
| `blk.39.attn_norm.weight` | `[2048]` | F32 | 8,192 |
| `blk.39.attn_output.weight` | `[4096, 2048]` | Q8_0 | 8,912,896 |
| `blk.39.attn_q.weight` | `[2048, 8192]` | Q8_0 | 17,825,792 |
| `blk.39.attn_q_norm.weight` | `[256]` | F32 | 1,024 |
| `blk.39.attn_v.weight` | `[2048, 512]` | Q8_0 | 1,114,112 |
| `blk.39.ffn_down_exps.weight` | `[512, 2048, 256]` | Q6_K | 220,200,960 |
| `blk.39.ffn_down_shexp.weight` | `[512, 2048]` | Q8_0 | 1,114,112 |
| `blk.39.ffn_gate_exps.weight` | `[2048, 512, 256]` | Q4_K | 150,994,944 |
| `blk.39.ffn_gate_inp.weight` | `[2048, 256]` | F32 | 2,097,152 |
| `blk.39.ffn_gate_inp_shexp.weight` | `[2048]` | F32 | 8,192 |
| `blk.39.ffn_gate_shexp.weight` | `[2048, 512]` | Q8_0 | 1,114,112 |
| `blk.39.ffn_up_exps.weight` | `[2048, 512, 256]` | Q4_K | 150,994,944 |
| `blk.39.ffn_up_shexp.weight` | `[2048, 512]` | Q8_0 | 1,114,112 |
| `blk.39.post_attention_norm.weight` | `[2048]` | F32 | 8,192 |

合计 556,623,872 B；GGUF 中是连续末端区间。

## 3. 共享专家胶囊

| 供体/层 | 精确张量名 | shape | dtype/bytes |
|---|---|---:|---:|
| Coder L47 | `model.layers.47.mlp.shared_expert.gate_proj.weight` | `[512,2048]` | BF16 / 2,097,152 |
| Coder L47 | `model.layers.47.mlp.shared_expert.up_proj.weight` | `[512,2048]` | BF16 / 2,097,152 |
| Coder L47 | `model.layers.47.mlp.shared_expert.down_proj.weight` | `[2048,512]` | BF16 / 2,097,152 |
| Coder L47 | `model.layers.47.mlp.shared_expert_gate.weight` | `[1,2048]` | BF16 / 4,096 |
| GLM L46 | `blk.46.ffn_gate_shexp.weight` | `[2048,1536]` | Q6_K / 2,580,480 |
| GLM L46 | `blk.46.ffn_up_shexp.weight` | `[2048,1536]` | Q6_K / 2,580,480 |
| GLM L46 | `blk.46.ffn_down_shexp.weight` | `[1536,2048]` | Q6_K / 2,580,480 |
| Qwen3.6 L39 | `blk.39.ffn_gate_shexp.weight` | `[2048,512]` | Q8_0 / 1,114,112 |
| Qwen3.6 L39 | `blk.39.ffn_up_shexp.weight` | `[2048,512]` | Q8_0 / 1,114,112 |
| Qwen3.6 L39 | `blk.39.ffn_down_shexp.weight` | `[512,2048]` | Q8_0 / 1,114,112 |
| Qwen3.6 L39 | `blk.39.ffn_gate_inp_shexp.weight` | `[2048]` | F32 / 8,192 |

总量：Coder BF16 6,295,552 B（现有 Q4_0 为 1,770,624 B）；GLM 7,741,440 B；
Qwen3.6 3,350,528 B。入口 `ffn_norm/post_attention_layernorm` 是显式依赖，不默认复制进此胶囊。

## 4. 路由专家胶囊

### 4.1 Coder L47 单专家 `i`

- `model.layers.47.mlp.gate.weight`：`[512,2048]` BF16，2,097,152 B；单专家胶囊可只取第
  `i` 行 `[1,2048]`，4,096 B BF16，或运输后 F32 8,192 B。
- `model.layers.47.mlp.experts.{i}.gate_proj.weight`：`[512,2048]` BF16，2,097,152 B。
- `model.layers.47.mlp.experts.{i}.up_proj.weight`：`[512,2048]` BF16，2,097,152 B。
- `model.layers.47.mlp.experts.{i}.down_proj.weight`：`[2048,512]` BF16，2,097,152 B。

每专家 BF16 6,291,456 B；top-10 为 62,914,560 B。现有 Q4_0 每专家 1,769,472 B，top-10
16,875 MiB；现有 E0/E471 v2 另带 8,192 B F32 router 行。

### 4.2 GLM L46 packed bank

- router：`blk.46.ffn_gate_inp.weight` `[2048,64]` F32 524,288 B；校正 bias：
  `blk.46.exp_probs_b.bias` `[64]` F32 256 B。
- gate/up：`blk.46.ffn_gate_exps.weight`、`blk.46.ffn_up_exps.weight`，均为
  `[2048,1536,64]` Q5_K 138,412,032 B。
- down：`blk.46.ffn_down_exps.weight` `[1536,2048,64]` Q6_K 165,150,720 B。

按 bank 的专家轴精确切片后，每专家 6,905,856 B；top-4 为 27,623,424 B；全 bank
441,974,784 B。切片必须保持 Q5_K/Q6_K block 对齐。

### 4.3 Qwen3.6 L39 packed bank

- router：`blk.39.ffn_gate_inp.weight` `[2048,256]` F32 2,097,152 B。
- gate/up：`blk.39.ffn_gate_exps.weight`、`blk.39.ffn_up_exps.weight`，均为
  `[2048,512,256]` Q4_K 150,994,944 B。
- down：`blk.39.ffn_down_exps.weight` `[512,2048,256]` Q6_K 220,200,960 B。

每专家 2,039,808 B；top-8 为 16,318,464 B；全 bank 522,190,848 B。

## 5. DSpark 胶囊

`DSpark` 在本报告中严格定义为 Qwen3-Next 单层 Gated DeltaNet core，不含该层 MoE。L44、L45、
L46 的 shape/dtype 完全相同，精确名称只把 `{L}` 替换为 `44`、`45` 或 `46`：

| 精确张量名 | shape | BF16 bytes |
|---|---:|---:|
| `model.layers.{L}.input_layernorm.weight` | `[2048]` | 4,096 |
| `model.layers.{L}.linear_attn.A_log` | `[32]` | 64 |
| `model.layers.{L}.linear_attn.conv1d.weight` | `[8192,1,4]` | 65,536 |
| `model.layers.{L}.linear_attn.dt_bias` | `[32]` | 64 |
| `model.layers.{L}.linear_attn.in_proj_ba.weight` | `[64,2048]` | 262,144 |
| `model.layers.{L}.linear_attn.in_proj_qkvz.weight` | `[12288,2048]` | 50,331,648 |
| `model.layers.{L}.linear_attn.norm.weight` | `[128]` | 256 |
| `model.layers.{L}.linear_attn.out_proj.weight` | `[2048,4096]` | 16,777,216 |

每层 BF16 总量 67,441,024 B；去掉 input norm 的算子 core 为 67,436,928 B。来源 shard：L44 为
`model-00037-of-00040.safetensors`，L45 为 shard 38，L46 为 shard 39。现有 L44--L46 Q4_0/F32
运行包中，上述 8 个张量各占 19,088,128 B。

DSpark 运行依赖还包括独立 recurrent matrix state、conv history、sequence 生命周期和 reset 规则；
这些是 runtime state，不是权重张量，必须写入 `capsule.json.interface.state`。
