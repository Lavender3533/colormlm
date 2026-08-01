"""只从冻结的 8 个 train 家族构造 Parallel Genome Head 训练/内部开发数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
FRONTEND = PROJECT / "fast16/research/parallel_frontend_v47"
DESIGN = PROJECT / "fast16/research/parallel_design_ir_v47"
TRAIN_GATES = FRONTEND / "data/train.jsonl"
TEACHERS = DESIGN / "teachers"
ONTOLOGY = HERE / "genome_head_ontology.json"
NEGATIVE_CONTRACT = HERE / "frontend_failure_mining/negative_contract.json"
FORMAT = "colorlm-v47-parallel-genome-record-v1"
SYSTEM = (
    "你是专业的前端产品设计架构器。先理解业务、信息层级、交互和响应式约束，"
    "再给出结构决策；不要退化为通用三卡片落地页。"
)


TITLE_BANKS: dict[str, list[str]] = {
    "pf47-train-01": [
        "北岸边缘节点控制台", "星港节点健康中心", "远洋边缘集群台", "霜桥节点告警室",
        "云岬设备运行台", "赤湾边缘运维中心", "极光节点监控台", "林海网关控制室",
        "环城边缘资产台", "南岭节点值守台", "深空节点状态台", "海峡算力运维台",
        "群岛网关作业台", "雪原边缘控制台", "城域节点指挥台", "河谷设备监控台",
        "沙洲边缘运营台", "轨道节点健康台",
    ],
    "pf47-train-02": [
        "纸上回声杂志铺", "边角独立刊物店", "慢印出版商店", "城市切片杂志社",
        "墨线季刊商店", "野生字体刊物铺", "未完稿独立书店", "页间风景杂志店",
        "离线阅读刊物社", "纸面实验商店", "折页文化杂志铺", "街区观察刊物店",
        "蓝墨独立出版店", "留白季刊商店", "纸岸设计杂志铺", "回形针刊物店",
        "小批量杂志商店", "新页独立刊物铺",
    ],
    "pf47-train-03": [
        "折线工业设计档案", "器物研究案例集", "工艺验证作品志", "握持实验项目集",
        "材料行为设计录", "日常器具案例库", "工业原型观察站", "形态研究作品集",
        "产品验证时间线", "制造细节案例志", "触感实验项目册", "场景设计档案馆",
        "结构迭代案例集", "人因研究作品录", "产品决策过程志", "实体交互案例库",
        "工业设计证据集", "器具原型演进录",
    ],
    "pf47-train-04": [
        "木屑社区工坊预约", "邻里陶艺工作室", "河岸修理工坊", "周末版画预约台",
        "社区木工时段中心", "街角手作工坊", "共享缝纫工作室", "玻璃工艺预约站",
        "青少年创客工坊", "社区摄影暗房", "旧物修复工作坊", "开放式金工教室",
        "纸艺体验预约页", "邻里模型制作室", "植物染工坊预约", "社区雕刻工作室",
        "公共创作实验室", "小型印刷工坊预约",
    ],
    "pf47-train-05": [
        "星河支付 API 工作台", "轻舟消息接口文档", "原子库存 API 手册", "云梯身份接口台",
        "深蓝搜索 API 文档", "北斗任务接口中心", "流光数据 API 工作台", "织网通知接口文档",
        "林间地图 API 手册", "方舟账单接口台", "潮汐文件 API 文档", "远山分析接口中心",
        "纸飞机邮件 API 台", "星尘事件接口文档", "港湾物流 API 手册", "回声语音接口台",
        "轨迹定位 API 文档", "墨盒内容接口中心",
    ],
    "pf47-train-06": [
        "账户隐私与会话中心", "个人安全设置台", "身份与设备控制中心", "数据使用偏好中心",
        "登录会话管理页", "访问权限设置中心", "个人资料安全台", "隐私边界控制室",
        "设备与授权中心", "账号保护设置页", "会话审计工作台", "个人数据控制中心",
        "登录安全管理台", "授权设备设置页", "隐私选择中心", "账户风险控制台",
        "身份安全与记录", "个人访问设置台",
    ],
    "pf47-train-07": [
        "潮岸多舞台音乐节", "旧仓声音周末", "河谷现场日程", "岛屿电子音乐节",
        "林间声场时间表", "城市屋顶音乐周", "港口多舞台演出", "深夜电台现场节",
        "山谷独立音乐节", "厂房声音实验日", "海边舞台日程", "街区现场周末",
        "公园多舞台演出", "极昼音乐节时间表", "冬季室内声场", "湖畔现场日程",
        "青年声音艺术节", "城市回声音乐周",
    ],
    "pf47-train-08": [
        "城市树冠变化故事", "街区绿荫数据志", "城市林冠观察录", "社区树木覆盖叙事",
        "热岛与树荫数据故事", "城市绿网年度报告", "街道树冠比较志", "公园覆盖率观察",
        "邻里绿量数据叙事", "城市树冠方法手册", "树荫公平数据故事", "街区林冠演变录",
        "城市绿地证据志", "树木覆盖年度故事", "步行道绿荫观察", "城市生态数据叙事",
        "社区林冠比较报告", "树冠覆盖趋势故事",
    ],
}


STYLE_LANGUAGE = {
    "dark": ["深色高对比但不刺眼", "暗色控制室氛围", "深色界面并保持正文清晰"],
    "light": ["明亮克制且层级清楚", "浅色界面与清晰边界", "高可读性的明亮工作区"],
    "paper": ["纸张感与编辑排版", "温和纸色和印刷感", "编辑式纸面视觉"],
}


SKELETONS = [
    "为“{title}”设计单文件 HTML。核心业务：{lede}。{style}；{layout_hint}。{quality}",
    "请规划并实现“{title}”的独立网页，优先保证{lede}。页面需要{layout_hint}，视觉采用{style}。{quality}",
    "制作一个可直接打开的“{title}”前端原型。必须覆盖{lede}，并用{layout_hint}组织信息。整体{style}。{quality}",
    "交付“{title}”的单页界面：业务重点为{lede}；信息架构应{layout_hint}；设计语言为{style}。{quality}",
    "不要写产品介绍稿，请真正实现“{title}”。需要{lede}，结构上{layout_hint}，视觉保持{style}。{quality}",
    "请把“{title}”做成可操作的完整页面，而非展示图。关键流程是{lede}；{layout_hint}；风格{style}。{quality}",
]


LAYOUT_HINTS = {
    "dashboard": "形成指标、控制区、主数据区和详情层的明确层级",
    "editorial": "使用非对称编辑网格和商品内容节奏，不堆卖点卡片",
    "timeline": "用连续时间线讲清阶段证据，并保留对比操作",
    "split": "在宽屏使用主流程与费用摘要双栏，窄屏自然单列",
    "docs": "建立可定位的侧栏、代码区与响应内容层级",
    "settings": "让设置、设备记录和危险操作各自拥有清楚边界",
    "schedule": "按日期、舞台和时间组织密集日程并标出冲突",
    "story": "用叙事段落、图表、表格替代和来源说明递进表达",
}


QUALITY_VARIANTS = [
    "禁止 emoji 充当图标，禁止默认三卡片，使用统一内联 SVG；可见焦点、减少运动，并检查 375/768/1024/1440px。",
    "不要使用空链接、远程图片或占位文案；交互必须有反馈，键盘可达，移动端不得产生整页横向滚动。",
    "内容至少有三层信息深度；按钮和可点击卡片需有 150–300ms 反馈，正文对比度达到 4.5:1。",
    "避免“极速/现代/安全”式通用文案和三项卖点；用真实业务状态、数据、错误和确认流程表达页面。",
]


DISTRACTORS = [
    "品牌故事", "联系我们", "功能亮点", "立即开始", "企业愿景", "常见问题", "团队介绍", "客户评价"
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def labels_from_genome(genome: dict[str, Any]) -> dict[str, str]:
    components = [".".join(item) for item in genome["c"]]
    return {
        "mode": genome["y"][0],
        "palette": genome["y"][1],
        "density": genome["y"][2],
        "shape": genome["y"][3],
        "layout": genome["l"][0],
        "mobile": genome["l"][1],
        "breakpoint": str(genome["l"][2]),
        "primary": components[0],
        "controls": components[1],
        "content": components[2],
        "detail": components[3],
        "support": components[4],
        "data_action": genome["x"][0],
        "view_action": genome["x"][1],
        "commit_action": genome["x"][2],
        "state_action": genome["x"][3],
        "main_responsive": genome["r"][0],
        "overlay_responsive": genome["r"][1],
        "accessibility": str(genome["a"]),
        "asset_policy": genome["z"],
    }


def byte_span(prompt: str, text: str) -> tuple[int, int]:
    start_char = prompt.index(text)
    end_char = start_char + len(text)
    return len(prompt[:start_char].encode("utf-8")), len(prompt[:end_char].encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "genome_dataset_v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    ontology_payload = json.loads(ONTOLOGY.read_text(encoding="utf-8"))
    ontology = {field["name"]: set(field["values"]) for field in ontology_payload["fields"]}
    negative_payload = json.loads(NEGATIVE_CONTRACT.read_text(encoding="utf-8"))
    if negative_payload.get("format") != "colorlm-v47-frontend-negative-contract-v1":
        raise ValueError("前端负约束格式不符合 v47 契约")
    projection = negative_payload.get("dataset_projection")
    if not isinstance(projection, dict) or projection.get("target_field") != "anti_pattern_contract":
        raise ValueError("前端负约束缺少 anti_pattern_contract 投影")
    if projection.get("merge_strategy") != "strict_deep_merge_reject_conflicts":
        raise ValueError("前端负约束必须使用严格无冲突合并")
    anti_pattern_contract = projection.get("value")
    if not isinstance(anti_pattern_contract, dict) or not anti_pattern_contract:
        raise ValueError("前端负约束投影为空")
    gates = read_jsonl(TRAIN_GATES)
    if len(gates) != 8 or any(row.get("split") != "train" for row in gates):
        raise ValueError("只允许读取冻结 train.jsonl，且必须正好 8 条")
    if {row["id"] for row in gates} != set(TITLE_BANKS):
        raise ValueError("train 家族与标题库不一致")

    rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    record = 0
    source_files = [TRAIN_GATES, ONTOLOGY, NEGATIVE_CONTRACT]
    for gate in gates:
        family = gate["id"]
        genome_path = TEACHERS / f"{family}.genome.json"
        slots_path = TEACHERS / f"{family}.slots.json"
        source_files.extend([genome_path, slots_path])
        genome = json.loads(genome_path.read_text(encoding="utf-8"))
        slots = json.loads(slots_path.read_text(encoding="utf-8"))["slots"]
        labels = labels_from_genome(genome)
        if set(labels) != set(ontology):
            raise ValueError(f"{family} 标签字段与 ontology 不一致")
        invalid = {name: value for name, value in labels.items() if value not in ontology[name]}
        if invalid:
            raise ValueError(f"{family} 标签不在 ontology: {invalid}")
        base_lede = str(slots[1]["text"])
        layout = genome["l"][0]
        mode = genome["y"][0]

        for variant, title in enumerate(TITLE_BANKS[family]):
            split = "train" if variant < 16 else "validation"
            local_index = variant if split == "train" else variant - 16
            style = STYLE_LANGUAGE[mode][variant % len(STYLE_LANGUAGE[mode])]
            quality = QUALITY_VARIANTS[(variant + int(family[-2:])) % len(QUALITY_VARIANTS)]
            lede = base_lede + [
                "，并用真实状态和反馈呈现完整流程",
                "，保证关键动作在首屏后仍能被发现",
                "，让移动端和键盘操作保持等价",
            ][variant % 3]
            prompt = SKELETONS[variant % len(SKELETONS)].format(
                title=title,
                lede=lede,
                style=style,
                layout_hint=LAYOUT_HINTS[layout],
                quality=quality,
            )
            prompt += " 禁止使用默认三卡片模板与 emoji 图标。"
            start_title, end_title = byte_span(prompt, title)
            start_lede, end_lede = byte_span(prompt, lede)
            distractors = [DISTRACTORS[(variant + offset) % len(DISTRACTORS)] for offset in range(4)]
            copy_candidates = [
                {"id": 0, "kind": "title", "text": title, "start_utf8": start_title, "end_utf8": end_title},
                {"id": 1, "kind": "lede", "text": lede, "start_utf8": start_lede, "end_utf8": end_lede},
                *[
                    {"id": index + 2, "kind": "distractor", "text": text}
                    for index, text in enumerate(distractors)
                ],
            ]
            task_id = f"pf47-aug-{family[-2:]}-{split}-{local_index:02d}"
            group_id = f"pf47-aug-group-{family[-2:]}-{split}-{local_index:02d}"
            cluster_id = f"pf47-aug-cluster-{family[-2:]}-{split}-{variant % len(SKELETONS):02d}"
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ]
            target = canonical_json(genome)
            tasks.append(
                {
                    "id": task_id,
                    "group_id": group_id,
                    "template_cluster_id": cluster_id,
                    "split": split,
                    "capability": "frontend_parallel_genome",
                    "target_mode": "complete_design_genome",
                    "messages": messages,
                    "target": target,
                }
            )
            rows.append(
                {
                    "format": FORMAT,
                    "task_id": task_id,
                    "group_id": group_id,
                    "template_cluster_id": cluster_id,
                    "split": split,
                    "capability": "frontend_parallel_genome",
                    "capture_record": record,
                    "messages_sha256": sha256_json(messages),
                    "source_family": family,
                    "prompt": prompt,
                    "copy_candidates": copy_candidates,
                    "copy_targets": {"title": 0, "lede": 1},
                    "target_genome": target,
                    "labels": labels,
                    "anti_pattern_contract": dict(anti_pattern_contract),
                    "metadata": {
                        "source_scope": "train_only",
                        "internal_validation": split == "validation",
                        "variant": variant,
                    },
                }
            )
            record += 1

    if len(rows) != 144 or Counter(row["split"] for row in rows) != Counter({"train": 128, "validation": 16}):
        raise AssertionError("必须生成 128 train + 16 train-only internal validation")
    for key in ("group_id", "template_cluster_id"):
        seen: dict[str, str] = {}
        for row in rows:
            previous = seen.setdefault(row[key], row["split"])
            if previous != row["split"]:
                raise AssertionError(f"{key} 跨 split 泄漏")
    if len({row["task_id"] for row in rows}) != len(rows) or len({row["prompt"] for row in rows}) != len(rows):
        raise AssertionError("task_id 或 prompt 重复")
    if any("三卡片" not in row["prompt"] and "三项卖点" not in row["prompt"] for row in rows):
        raise AssertionError("每条 prompt 必须显式携带模板退化约束")

    report = {
        "format": "colorlm-v47-parallel-genome-dataset-manifest-v1",
        "status": "train_only_prepared",
        "claim_limit": "internal validation 仍来自 8 个 train 家族，只用于实现开发；不等于冻结 validation 泛化。",
        "row_count": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "source_family_count": len(gates),
        "unique_prompt_count": len({row["prompt"] for row in rows}),
        "label_field_count": len(ontology),
        "source_files": [
            {"path": str(path.relative_to(PROJECT)).replace("\\", "/"), "sha256": sha256_file(path)}
            for path in source_files
        ],
        "official_validation_or_blind_read": False,
        "copy_candidates_per_row": 6,
        "anti_pattern_contract_sha256": sha256_json(anti_pattern_contract),
        "anti_patterns": sorted(str(item["id"]) for item in negative_payload["constraints"]),
        "research_basis": [
            {"name": "Design2Code", "url": "https://arxiv.org/abs/2403.03163", "implication": "监督视觉元素召回与布局，而非只检查语法"},
            {"name": "WebDesignIter", "url": "https://arxiv.org/abs/2607.10621", "implication": "持久设计知识与结构合同优先于自由长代码"},
        ],
    }
    if args.selftest:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / "tasks.jsonl", args.output_dir / "dataset.jsonl", args.output_dir / "manifest.json"]
    if not args.force and any(path.exists() for path in paths):
        raise FileExistsError("输出已存在；拒绝覆盖，确需重建请使用 --force")
    paths[0].write_text("".join(canonical_json(row) + "\n" for row in tasks), encoding="utf-8", newline="\n")
    paths[1].write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    report.update(
        {
            "tasks": str(paths[0].resolve()),
            "tasks_sha256": sha256_file(paths[0]),
            "dataset": str(paths[1].resolve()),
            "dataset_sha256": sha256_file(paths[1]),
        }
    )
    paths[2].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
