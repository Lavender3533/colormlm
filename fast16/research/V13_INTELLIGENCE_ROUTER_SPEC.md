# ColorLM v13 Intelligence Router 最小科研契约

## 目标

用同一个 teacher token 前缀分别测量三条原子路径的 next-token NLL：

- `no_op`：Coder 与 K3 都关闭。
- `coder`：只打开 Qwen3-Coder-Next E471 残差。
- `k3`：只打开现有四颗 Kimi K3 胶囊及其站内连续路由。

标签只能是 `argmin_route(-log p(target_token | prefix))`。生成后的文本是否正确、关键词、任务类别和供体 embedding 均不能作为 v13 路由标签。

## 已复用的现有接口

1. `llama-server /completion`
   - `logit_bias=[[target_token,100]]`：强制每条路径生成同一个 teacher token。
   - `temperature=-1` 与 `post_sampling_probs=false`：外层 `logprob` 来自采样器前的原始 logits softmax，不含强制 bias，因此无需依赖 target 是否进入 top-k。
   - `cache_prompt=false`：不同前缀的 batch 方式保持一致。
2. `llama-server /tokenize` 与 `/apply-template`
   - teacher 使用真实模型 tokenizer 和真实 chat/tool 模板。
3. `qwen35moe.cpp` 已有隐藏状态探针，启动器已提供正式参数
   - `--hidden-dump <path>`
   - `--hidden-dump-sites 12,28`
   - `--hidden-dump-max-records <N>`
   - 二进制格式为已有 `CLM9 v1`。采集器在首个 teacher 请求前记录文件字节偏移，只解析该偏移之后、本轮请求实际追加的完整记录。已有 warmup 即使与首个 teacher 长度相同，也不会被扫描或误配。

## 三条路径的进程级强制配置

当前源码提供 `--force-path no_op|coder|k3`。它在同一套已加载权重上强制全部路由站走一条原子路径。由于本机内存预算，仍应一次只保留一个服务，依次启动三个短校准进程。

以下命令均在仓库根目录运行。建议先停止正式 `8101` 服务，并把短校准上下文限制为 `4096`。

### 1. no_op + 隐藏状态采集

```powershell
.\fast16\run-colormlm-v12-neural-alloy.bat --port 8110 --ctx-size 4096 `
  --runtime-alias ColorLM-v13-probe-no-op --force-path no_op `
  --hidden-dump fast16\research\v13_no_op_hidden.bin `
  --hidden-dump-sites 12,28 --hidden-dump-max-records 160
```

### 2. coder only

```powershell
.\fast16\run-colormlm-v12-neural-alloy.bat --port 8110 --ctx-size 4096 `
  --runtime-alias ColorLM-v13-probe-coder --force-path coder
```

### 3. K3 only

```powershell
.\fast16\run-colormlm-v12-neural-alloy.bat --port 8110 --ctx-size 4096 `
  --runtime-alias ColorLM-v13-probe-k3 --force-path k3
```

每次切换配置都必须结束旧的 `8110` 监听进程。采集器会检查 alias，防止把旧进程误记成新路径；启动器的 `--validate-only` 输出也会记录 `force_path`。

## 最短数据流程

只在第一次 `no_op` 服务上生成 teacher：

```powershell
python fast16\research\v13_counterfactual_router.py prepare `
  --endpoint http://127.0.0.1:8110 `
  --expect-model ColorLM-v13-probe-no-op `
  --tasks fast16\research\v13_counterfactual_tasks.example.jsonl `
  --output fast16\research\v13_teacher.jsonl `
  --max-target-tokens 12 --max-samples 36
```

仍在 `no_op` 服务上采集 NLL 与隐藏状态：

```powershell
python fast16\research\v13_counterfactual_router.py collect `
  --endpoint http://127.0.0.1:8110 --expect-model ColorLM-v13-probe-no-op `
  --route no_op --teacher fast16\research\v13_teacher.jsonl `
  --output fast16\research\v13_no_op.jsonl `
  --hidden-dump fast16\research\v13_no_op_hidden.bin
```

切换到对应服务后，分别运行：

```powershell
python fast16\research\v13_counterfactual_router.py collect `
  --endpoint http://127.0.0.1:8110 --expect-model ColorLM-v13-probe-coder `
  --route coder --teacher fast16\research\v13_teacher.jsonl `
  --output fast16\research\v13_coder.jsonl

python fast16\research\v13_counterfactual_router.py collect `
  --endpoint http://127.0.0.1:8110 --expect-model ColorLM-v13-probe-k3 `
  --route k3 --teacher fast16\research\v13_teacher.jsonl `
  --output fast16\research\v13_k3.jsonl
```

最后校准两个站点各自的 3 路线性 softmax 路由：

```powershell
python fast16\research\v13_counterfactual_router.py calibrate `
  --shard no_op=fast16\research\v13_no_op.jsonl `
  --shard coder=fast16\research\v13_coder.jsonl `
  --shard k3=fast16\research\v13_k3.jsonl `
  --feature-route no_op `
  --output-dir fast16\research\v13_router_candidate
```

如三份 shard 来自只强制一个站点的反事实采集，应只让监督作用于该站点：

```powershell
python fast16\research\v13_counterfactual_router.py calibrate `
  --shard no_op=fast16\research\v13_no_op_l12.jsonl `
  --shard coder=fast16\research\v13_coder_l12.jsonl `
  --shard k3=fast16\research\v13_k3_l12.jsonl `
  --feature-route no_op --sites 12 `
  --output-dir fast16\research\v13_router_l12_candidate
```

`--sites 12` 与 `--sites 28` 均受支持；未传 `--sites` 时，默认拟合隐藏 sidecar 声明的全部站点。报告中的 `supervision.applies_to_sites` 是监督实际作用范围，不能把逐站反事实标签外推到另一个站点。

## 数据完整性规则

- 强制生成的 token ID 必须等于 teacher token，否则整条采集立即失败。
- 外层原始 logprob 必须有限；只有三个路线都得到精确 logprob 的 token 才能进入 `argmin NLL` 标签集。
- 第一名与第二名 NLL 差距小于 `0.05` 的 token 默认丢弃，避免把数值噪声训练成路由规则。
- 隐藏状态必须来自 `attn_post_norm`，每 token 做 L2 单位化后进入线性路由。
- hidden dump 在请求开始前必须位于合法记录边界；采集结束后新增记录数必须精确等于 `teacher样本数 × sites数`。dump 被截断、重写、夹入额外记录或包含 NaN/Inf 时立即失败。
- 按 task 分组留出，不能把同一答案相邻 token 同时泄漏到训练集和留出集。
- 留出准确率必须至少超过训练集 majority 路线在留出集上的准确率 10 个百分点，才标记为 `candidate`。
- 产物权重只用 train tasks 拟合，heldout tasks 不参与落盘权重拟合。完成独立 final test 前，不允许拿 heldout 全量重拟合后仍沿用旧留出指标。

## 产物 ABI

每个站点输出：

- `layer_12/weight.f32` / `layer_28/weight.f32`
  - little-endian F32
  - shape `[3, 2048]`
- `layer_12/bias.f32` / `layer_28/bias.f32`
  - little-endian F32
  - shape `[3]`
- `layer_12/router.json` / `layer_28/router.json`
  - 固定 class 顺序、哈希、训练/留出指标、`status`、`deployable` 与 train-only 拟合证明。
- `calibration_report.json`
  - 始终生成，记录 manifest 验真、监督站点、对照门结果与 `runtime_plan.generated/reason`。
- `ColorLM-v13-Intelligence-Router.intplan.json`
  - 仅当所选站点全部通过晋级门时生成，必须包含 `status: candidate` 与 `calibration_report_sha256`。
  - `rejected_control` 不生成该文件；若目录中存在同名陈旧计划，校准器会删除它，防止误加载。

运行时公式：

```text
x = attn_post_norm / max(||attn_post_norm||_2, 1e-8)
route_prob = softmax(W @ x + b)
```

## 当前接入状态

C++ 源码已经具备三路路由加载器、L12/L28 图内 softmax、`no_op/coder/k3` 连续残差缩放与 `--force-path` 反事实接口。当前剩余工作是：

1. 用三条强制路径采集首批真实 NLL 与 no-op 隐藏状态。
2. 生成并校验每站点 train-only 的 `weight.f32`、`bias.f32`、`router.json`；只有通过对照门才生成总 `intplan.json`。
3. 把候选路由计划写入 v13 alloy plan/package，并做一次短编程与工具回合验收。
4. 第一版仍是 dense residual：先计算 Coder/K3 再混合；路由证明有效后再做 lazy expert。
5. `--neural-bus-alpha 0 --k3-alpha 0` 的精确回退路径不得改变。

这份最小流程证明的是“隐藏状态是否能预测哪条现有路径对正确下一 token 更有利”。在它通过之前，继续增加供体没有可归因价值。
