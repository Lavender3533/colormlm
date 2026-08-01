# 北极星 v47 Parallel Genome Head（研究预览）

北极星（Polaris）是原 ColorLM 研究项目的正式新名称。本目录只发布 v47 的小型前端 Design
Genome 能力头、训练数据合同和可复现 terminal hidden；不包含 34.66B 主模型 GGUF。

## 当前证据

- 主干：北极星 v36 / Qwen3.6 shared-backbone，hidden width 2048。
- 训练样本：128；train-only internal validation：16。
- 参数量：278,911；F32 权重约 1.06 MiB。
- internal validation 完整 20 字段 Genome：93.75%。
- internal validation 字段准确率：99.6875%；最弱字段：93.75%。
- 历史网页负约束：默认三卡片、emoji 图标、假交互、远程资源、可见焦点、减少运动、响应式、
  语义 HTML、表单标签，共九类。

这些结果只允许进入确定性编译器 A/B，不证明最终网页质量、通用能力提升或追平 GPT/Claude。
在冻结 validation 的静态门、四档视口浏览器 action trace 和一次 blind 通过前，本包状态保持
`research_preview`。

## 文件

- `genome_head.npz`：Parallel Genome Head 权重。
- `dataset.jsonl`：144 条训练与内部开发记录。
- `initial_states_hidden_only.cnob`：144 条连续 terminal hidden，仅约 1.13 MiB。
- `genome_head_ontology.json`：20 字段封闭本体。
- `negative_contract.json`：九类前端负约束。
- `fit_report.json`：训练指标与权重哈希。
- `manifest.json`：发布边界和所有文件 SHA-256。

## 源码与复现

源码仓库：<https://github.com/Lavender3533/colormlm>

Ascend 910B4 使用源码中的：

```bash
python fast16/cloud/ascend910b/doctor.py --device npu --matrix-size 4096
python fast16/cloud/ascend910b/run_genome_head_npu.py \
  --device npu --require-device \
  --dataset /path/to/dataset.jsonl \
  --capture /path/to/initial_states_hidden_only.cnob \
  --output /tmp/polaris_genome_head_npu.npz \
  --report /tmp/polaris_genome_head_npu_report.json \
  --runtime-report /tmp/polaris_npu_runtime.json
```

运行报告必须明确记录实际设备为 `npu:0`；禁止用 CPU fallback 冒充 NPU 结果。

## 910B4 正式云任务：训练 8B LoRA donor

本地小头只需数秒，不占用云配额。云端应直接训练可移植的Qwen3-8B Design Genome donor：

```bash
python -m pip install -U atomgit modelscope transformers peft accelerate safetensors sentencepiece
atomgit download gcw_qCzqdxKl/ColorLM -d /tmp/polaris-v47-assets
modelscope download --model Qwen/Qwen3-8B --local_dir /root/models/Qwen3-8B
mkdir -p /root/polaris-output
python -X utf8 /tmp/polaris-v47-assets/train_polaris_design_donor.py \
  --dataset /tmp/polaris-v47-assets/dataset.jsonl \
  --model-id /root/models/Qwen3-8B \
  --output-dir /root/polaris-output/design-genome-lora \
  --report /root/polaris-output/design-genome-lora.report.json \
  --epochs 3 --max-length 768 --batch-size 1 --gradient-accumulation 8 \
  --lora-rank 16 --lora-alpha 32 --wall-seconds 3300
```

`wall-seconds=3300`为当前剩余约一小时的实例预留下载和保存时间。训练脚本强制`npu:0`，不会
回退CPU。

## 命名兼容

对外模型名已经是北极星（Polaris）。历史脚本、`fast16/`路径和`COLORLM_*`环境变量暂时保留，
仅用于兼容已有运行时和可复现实验。
