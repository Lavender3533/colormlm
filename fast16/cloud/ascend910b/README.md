# 北极星（Polaris）Ascend 910B4 云训练入口

适配已验证的 Ubuntu 22、CANN 8.5、Python 3.11、`torch 2.9.0+cpu + torch_npu`、单卡 Ascend 910B4 32 GiB。所有脚本均不下载权重，也不启动或停止模型服务。

## 1. 真机诊断

在仓库根目录执行；这条命令必须选择 `npu:0`，并真实运行一个 FP16 矩阵乘：

```bash
python -X utf8 fast16/cloud/ascend910b/doctor.py --device npu --matrix-size 4096
```

结果应为 UTF-8 JSON，且同时满足 `ok: true`、`selection.selected: "npu:0"`、`operation.finite: true`。CANN 文件 owner 警告在平台镜像中常见，只要上述三项成立就不影响本入口。

## 2. 设备与反向传播自检

```bash
python -X utf8 fast16/cloud/ascend910b/training_device.py --device npu --require-device
```

本脚本真实执行小型前向、反向传播和 AdamW 更新。容器的 `/dev/shm` 只有 64 MiB，因此适配层固定 `num_workers=0`、`pin_memory=false`、`persistent_workers=false`，不建立 DataLoader worker 共享内存队列。

## 3. 运行原版 v47 Parallel Genome Head

包装器复用原训练器，不复制训练算法。请把示例路径换成云端真实数据路径：

```bash
mkdir -p fast16/cloud/ascend910b/output
python -X utf8 fast16/cloud/ascend910b/run_genome_head_npu.py \
  --device npu --require-device \
  --dataset /path/to/genome_dataset.jsonl \
  --capture /path/to/terminal_hidden.cnob \
  --ontology fast16/research/v47_dual_tempo_bus/genome_head_ontology.json \
  --output fast16/cloud/ascend910b/output/genome_head.npz \
  --report fast16/cloud/ascend910b/output/genome_head.report.json \
  --latent-width 128 --batch-size 32 --epochs 120 --wall-seconds 300
```

包装器另写 `genome_head.report.json.runtime.json`，其中 `selection.selected` 是实际训练设备。若希望 NPU 不可用时自动回退 CPU，改用 `--device auto` 并去掉 `--require-device`。

根盘仅 50 GiB；训练数据和输出应控制在配额内，不要在此入口缓存基座权重。纯 CPU 兼容检查可执行：

```bash
python -X utf8 fast16/cloud/ascend910b/doctor.py --device cpu --matrix-size 128
python -X utf8 fast16/cloud/ascend910b/training_device.py --device cpu --steps 4
```

## 4. 真正占用 NPU 的 Design Genome LoRA donor

Parallel Genome Head 在本地只需数秒，不值得占用云配额。当前云端主任务是加载可放入32GiB HBM
的Qwen3-8B，并训练可移植的北极星Design Genome LoRA donor：

```bash
python -m pip install -U atomgit modelscope transformers peft accelerate safetensors sentencepiece
atomgit download gcw_qCzqdxKl/ColorLM -d /tmp/polaris-v47-assets
modelscope download --model Qwen/Qwen3-8B --local_dir /root/models/Qwen3-8B

mkdir -p /root/polaris-output
python -X utf8 fast16/cloud/ascend910b/train_polaris_design_donor.py \
  --dataset /tmp/polaris-v47-assets/dataset.jsonl \
  --model-id /root/models/Qwen3-8B \
  --output-dir /root/polaris-output/design-genome-lora \
  --report /root/polaris-output/design-genome-lora.report.json \
  --epochs 3 --max-length 768 --batch-size 1 --gradient-accumulation 8 \
  --lora-rank 16 --lora-alpha 32 --wall-seconds 4200
```

该入口硬编码选择`npu:0`，NPU不可用时直接失败，不会CPU回退。`wall-seconds=4200`会在两小时
实例结束前留出保存和下载adapter的时间。当前128/16划分仍只属于八个train家族的内部开发门；
adapter必须再经过冻结跨家族validation、确定性编译网页A/B和一次blind，才能作为北极星正式
能力部件。
