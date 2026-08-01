# ColorLM v47 前端负样本挖掘

本目录把用户实际试用得到的六个 HTML 转换为可复现的失败指纹与负约束。它不保存 HTML 原文、正文片段、绝对源路径或远程 URL；远程资源仅保留 SHA-256、类别和计数。

## 产物

- `mine_frontend_failures.py`：复用 `parallel_frontend_v47` 冻结静态评分口径，并补充 emoji UI 图标、惰性交互、远程占位资源、四档视口、语义 HTML 与表单标签检测。
- `report.json`：六个样本的源码哈希、静态分数快照、九类失败计数与失败特征哈希。
- `negative_contract.json`：可直接合并到 `build_parallel_genome_dataset.py` 每行 `anti_pattern_contract` 的 `dataset_projection.value`，以及教师筛选条件。
- `selftest.py`：验证两次内存重建完全一致、九类约束齐全、现有 JSON 与输入一致、无 BOM、无 HTML/远程 URL 泄漏。

九类合同覆盖：默认三卡片、emoji 图标、空链接/假交互、远程占位图、可见焦点、减少运动、响应式、语义 HTML、表单标签。

## 重建

在项目根目录运行：

```powershell
python -X utf8 fast16/research/v47_dual_tempo_bus/frontend_failure_mining/mine_frontend_failures.py `
  --input-dir "C:\Users\Kangnaixi\Desktop\新建文件夹"
```

确定性检查与完整自检：

```powershell
python -X utf8 fast16/research/v47_dual_tempo_bus/frontend_failure_mining/mine_frontend_failures.py `
  --input-dir "C:\Users\Kangnaixi\Desktop\新建文件夹" --check

python -X utf8 fast16/research/v47_dual_tempo_bus/frontend_failure_mining/selftest.py `
  --input-dir "C:\Users\Kangnaixi\Desktop\新建文件夹"
```

## 接入合同

数据构建器读取 `negative_contract.json` 后，对每条记录执行严格深合并：

```text
row["anti_pattern_contract"] <- negative_contract["dataset_projection"]["value"]
```

若已有键与合同值冲突，应拒绝生成，不能静默覆盖。教师候选必须满足 `teacher_screening.required_failure_state` 全部为 `false`，然后再进行 375/768/1024/1440px 浏览器、键盘和交互回放。静态通过不代表真实页面已经通过。
