# ColorLM v18 深层激活坐标桥

## 结论先行

v17 当前使用的不是深层激活桥，而是共享 token 词嵌入上的正交 Procrustes 桥。
它有真实价值：889 个留出 token 的嵌入余弦中位数从约 `0.0014` 提升到
`0.4579`，并保持严格正交。但这只能证明词表几何可运输，不能证明
ColorLM L35 的上下文状态与 Coder-Next L44 的真实输入状态对齐。

当前运行图的准确接口是：

```text
ColorLM L35 attn_residual (= ffn_residual)
  -> transport_in
  -> Coder-Next L44 input
  -> L44 -> L45 -> L46 -> L47
  -> donor_delta
  -> transport_out
  -> ColorLM L35 residual update
```

所以校准必须配对以下两个张量，而不是配对词嵌入：

- 主干：`qwen35moe.cpp` 中 L35 的 `attn_residual`，即传给
  `build_neural_island_delta(..., ffn_residual, ...)` 的原始张量。
- 供体：完整 Qwen3-Coder-Next 中 L43 的 `l_out`；它正是 L44 的
  `layer_input`。

不要把 v17 图中的 `neural_island_transport_in` 当作供体标签。它本身就是旧桥的
输出，用它回归会形成循环证明。

## 当前资源边界

- ColorLM 与 Coder-Next 隐藏宽度均为 `2048`，可以使用方阵桥。
- 本地已有 Coder-Next L44-L47 的运行权重和部分原始 Range。
- 本地没有完整 Coder-Next L0-L43 模型，因此现在不能生成“真实 L44 输入”。
- 现有 `COLORLM_HIDDEN_DUMP` 只在 ColorLM 路径采集 `attn_post_norm`，阶段也不对。

因此本目录已完成解析、配对、拟合、留出验收和运行包重打包；真正采集只差一个
通用 eval-callback dump 接口，以及之后准备完整供体推理权重。这里没有下载或启动模型。

## 已实现工具

### `activation_bridge.py`

提供四个子命令：

```powershell
python fast16\research\v18_activation_bridge\activation_bridge.py inspect <dump.bin>
python fast16\research\v18_activation_bridge\activation_bridge.py collect ...
python fast16\research\v18_activation_bridge\activation_bridge.py fit ...
python fast16\research\v18_activation_bridge\activation_bridge.py self-test
```

核心约束：

- 原生解析项目现有 `CLM9 v1`，拒绝坏 header、截断 payload、非 F32 与 NaN/Inf。
- `collect` 在每个请求前后记录字节边界，不会把 warmup 或前一请求错配进来。
- 同时调用 `/tokenize` 保存 token piece 和原始 prompt。每个 tokenizer 独立沿原文
  UTF-8 字节推进游标，只在两侧“累计结束字节”相同的位置配对隐藏态。例如
  `hello | 世 | 界` 与 `he | llo | 世界` 会在字节 5 和 11 配对，不要求词表或切词相同。
- BOS、控制 token、空 piece、非法字节以及不能贴合当前原文字节游标的 piece 不参与
  配对，也不会推进游标；其后的正常 token 仍可恢复映射。
- 收据格式为 `colorlm-activation-capture-v2`。报告逐 prompt 记录原文字节数、两侧
  token 数、可映射/排除 token 数、共同结束边界数和匹配覆盖率。匹配覆盖率定义为
  `matched / max(base_mappable, donor_mappable)`，对切得更细的一侧保持保守。
- 训练/留出按整条 prompt 划分，不把同一 prompt 的相邻 token 随机拆到两边。
- 使用“激活交叉协方差 + 可调嵌入桥先验”的正交 Procrustes。`prior_samples>0`会让未覆盖
  方向趋向旧桥；当前通过的v18候选用`prior_samples=0`，因此仍要把未观测子空间欠约束列为
  明确风险，下一步用更多激活或nullspace-only先验解决。
- 独立拟合稳健标量尺度，并输出真正的运行矩阵：
  `donor_column = W_in @ colorlm_column` 与
  `colorlm_delta_column = W_out @ donor_delta_column`。
- 同时输出旧编译器兼容的纯正交 `donor_to_colorlm` 矩阵。

### `repack_island_bridge.py`

只有拟合报告 `promotion.decision == "candidate"` 才默认工作。它会：

1. 保留 v17 正式包不动；
2. 把四层包复制到新目录；
3. 只替换每层最后两个 F16 transport 张量；
4. 更新张量、整块权重、block manifest 与 island manifest 的 SHA-256；
5. 在复制前验证 `W_out @ W_in` 的可逆误差。

## 最小通用采集补丁建议

不要再给 `qwen35moe.cpp` 和 `qwen3next.cpp` 分别复制一套 dump 代码。最小且可复用的
做法是给 llama.cpp 的 scheduler eval callback 增加一个二进制精确过滤器：

1. 在 `llama.cpp/common/debug.{h,cpp}` 新增
   `common_activation_dump_cb_user_data`，沿用 `common_debug_cb_eval` 已有的
   `ask=true/false` 两阶段与 `ggml_backend_tensor_get` GPU 回读方式。
2. 仅当环境变量 `COLORLM_ACTIVATION_DUMP` 与
   `COLORLM_ACTIVATION_TENSOR` 同时存在时，把该 callback 挂到 server 的
   `common_params.cb_eval`。默认路径零开销。
3. tensor 名必须完整正则匹配，不做关键词任务路由：
   - 主干：`^attn_residual-35$`
   - 供体：`^l_out-43$`
4. 输出继续使用 `<IIiI4qQ + F32 payload` 的 `CLM9 v1`，layer 从 tensor 名尾部解析。
   两个 stage 使用两个独立文件，所以 header 不需要破坏性升级。
5. 增加 `COLORLM_ACTIVATION_DUMP_MAX_RECORDS`，超过上限后只停止写文件，不影响推理。

为什么选 scheduler callback：图构建回调已经把节点命名为
`attn_residual-35` / `l_out-43`；scheduler callback 对 Qwen3Next、Qwen3.5 MoE 和后续
供体都通用，也能正确从 Vulkan tensor 回读。它避免在模型图里插入额外 custom op，
更符合“通用神经总线”的研究基础设施方向。

## 采集与拟合命令

通用 dump 补丁完成后，分别启动两个单实例服务。启动前设置：

```powershell
# ColorLM 原生目标状态
$env:COLORLM_ACTIVATION_DUMP = 'D:\project\大模型ssd化\fast16\research\v18_activation_bridge\captures\base.bin'
$env:COLORLM_ACTIVATION_TENSOR = '^attn_residual-35$'
$env:COLORLM_ACTIVATION_DUMP_MAX_RECORDS = '32'

# Coder-Next 服务则改为 donor.bin 与 ^l_out-43$
```

服务启动并完成权重加载后，两个端口依次采同一份短 prompt：

```powershell
python fast16\research\v18_activation_bridge\activation_bridge.py collect `
  --endpoint http://127.0.0.1:8120 `
  --expect-model ColorLM-v17-Coder-Neural-Island `
  --dump fast16\research\v18_activation_bridge\captures\base.bin `
  --layer 35 --stage attn_residual `
  --prompts fast16\research\v18_activation_bridge\calibration_prompts.jsonl `
  --output fast16\research\v18_activation_bridge\captures\base.receipt.json

python fast16\research\v18_activation_bridge\activation_bridge.py collect `
  --endpoint http://127.0.0.1:8121 `
  --expect-model Qwen3-Coder-Next `
  --dump fast16\research\v18_activation_bridge\captures\donor.bin `
  --layer 43 --stage l_out `
  --prompts fast16\research\v18_activation_bridge\calibration_prompts.jsonl `
  --output fast16\research\v18_activation_bridge\captures\donor.receipt.json

python fast16\research\v18_activation_bridge\activation_bridge.py fit `
  --base-receipt fast16\research\v18_activation_bridge\captures\base.receipt.json `
  --donor-receipt fast16\research\v18_activation_bridge\captures\donor.receipt.json `
  --output-dir fast16\research\v18_activation_bridge\candidate
```

显存和内存不足以同时驻留两套模型时，用 `capture_model.py` 串行采集。它会检查端口、
启动一个服务、采集、写运行收据，然后在成功或失败时都关闭该服务。主干采完后再执行供体
命令即可；不会要求两套模型同时加载：

```powershell
python fast16\research\v18_activation_bridge\capture_model.py `
  --model fast16\models\ColorLM-v6-Q3Router-Fused-A1.gguf `
  --alias ColorLM-v18-base-capture --port 8120 `
  --cpu-moe-layers 29 --no-mmap `
  --tensor '^attn_residual-35$' --layer 35 --stage attn_residual `
  --dump fast16\research\v18_activation_bridge\captures\base.bin `
  --prompts fast16\research\v18_activation_bridge\calibration_prompts.jsonl `
  --receipt fast16\research\v18_activation_bridge\captures\base.receipt.json

python fast16\research\v18_activation_bridge\capture_model.py `
  --model fast16\models\donor\qwen3-coder-next-iq3s\Qwen3-Coder-Next-UD-IQ3_S.gguf `
  --alias Qwen3-Coder-Next-capture --port 8121 `
  --cpu-moe-layers 48 `
  --tensor '^l_out-43$' --layer 43 --stage l_out `
  --dump fast16\research\v18_activation_bridge\captures\donor.bin `
  --prompts fast16\research\v18_activation_bridge\calibration_prompts.jsonl `
  --receipt fast16\research\v18_activation_bridge\captures\donor.receipt.json
```

完整79.67B供体已在本机按上述路径通过串行采集。采集器固定`--no-warmup`，因为该供体在
空输入warmup路径会异常退出，而真实prompt prefill已稳定完成12/12。

通过后再生成隔离的 v18 包：

```powershell
python fast16\research\v18_activation_bridge\repack_island_bridge.py `
  --source-island fast16\research\v17_coder_island\runtime-v3\island.json `
  --bridge-report fast16\research\v18_activation_bridge\candidate\activation_bridge_report.json `
  --input-weight fast16\research\v18_activation_bridge\candidate\coder_activation_input_weight_f32.npy `
  --output-weight fast16\research\v18_activation_bridge\candidate\coder_activation_output_weight_f32.npy `
  --output fast16\research\v18_activation_bridge\runtime-v1
```

## 时间和样本路线

模型加载时间不计入校准，因为两个模型只需各加载一次。

### 30 秒到 90 秒快门

- 6-8 条原文完全一致、能产生共同 UTF-8 结束边界的 prompt。
- 至少 512 个配对 token，建议 768-1024。
- 25% prompt 留出；先用小规模正先验做快门，零先验只能作为待复核候选。
- 采集只生成 1 token；热服务下主要成本是两次短 prefill。
- 2048x2048 SVD 通常是数秒到数十秒，取决于 BLAS 与 CPU。

这个门只决定“是否值得做 v18 候选”，不宣称能力提升。

### 数分钟正式候选

- 12-24 条 prompt，覆盖代码生成、调试、工具协议、多语言指令。
- 至少 2048 个配对 token；推荐 4096。
- 以`prior_samples=0/64/128`做短消融，并至少独立重复一次不同prompt划分；零先验胜出时必须
  明确记录低秩空白子空间风险。
- 仍只做一次短能力 A/B，不跑长榜。

## 默认验收门

必须同时满足：

- 成功产生共同 UTF-8 结束边界的 prompt 数 `>= 6`；总匹配 token 数 `>= 512`。
- 整 prompt 留出集相对嵌入桥的余弦中位数提升 `>= 0.03`。
- 新旧桥都使用训练集得到的同一个供体原生幅度，留出相对 RMSE 不超过嵌入桥的 `0.95x`；
  各自在留出集重新拟合最优scale只作诊断，禁止参与晋级比较。
- 至少 `67%` 的留出 prompt 单独获得正余弦提升。
- F32 正交误差与入口/出口 cycle RMSE 均 `<= 5e-5`。
- 稳健尺度在 `[0.25, 4.0]` 内。

任何门失败都输出完整报告和矩阵，但报告标记 `reject`，重打包器默认拒绝安装。
这能把“输出变了”与“桥真的更接近供体深层坐标”区分开。

## 下一步最小能力验证

桥通过几何门并装入隔离包后，只做三轨短测：

1. v17 嵌入桥；
2. v18 激活桥；
3. v18 `alpha=0` 精确退回主干。

当前一条Rust所有权任务中v17/v18均给出同一正确实现，`alpha=0`输出逐字回退。下一步仍需
使用2-4个未参与校准的真实编程/工具任务；v18至少不能低于v17，且
`alpha=0` 必须逐 token 等价。几何门通过但短能力回归，就保留研究报告、拒绝晋级，
不继续扩到 L40-L47 八层岛。
