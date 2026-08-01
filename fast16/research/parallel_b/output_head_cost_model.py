"""v19 mapped output head 与低秩侦察/精确行分页方案的静态成本模型。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


MIB = 1024 * 1024


@dataclass(frozen=True)
class Geometry:
    mapped_rows: int = 131_612
    hidden_size: int = 2_048
    q6_row_bytes: int = 1_680
    mapping_id_bytes: int = 8

    @property
    def dense_projection_bytes(self) -> int:
        return self.mapped_rows * self.q6_row_bytes

    @property
    def dense_package_bytes(self) -> int:
        output_norm = self.hidden_size * 4
        mapping = self.mapped_rows * self.mapping_id_bytes
        return output_norm + self.dense_projection_bytes + mapping

    @property
    def dense_macs(self) -> int:
        return self.mapped_rows * self.hidden_size


@dataclass(frozen=True)
class CascadePreset:
    name: str
    rank: int
    exact_candidate_rows: int
    gpu_cache_rows: int


def model_preset(geometry: Geometry, preset: CascadePreset) -> dict[str, object]:
    # A 使用逐行 Q8 + F16 scale；B 使用 F16。rho 是含量化误差的逐行 F16 上界。
    scout_a = geometry.mapped_rows * preset.rank
    scout_a_scales = geometry.mapped_rows * 2
    scout_b = preset.rank * geometry.hidden_size * 2
    residual_norms = geometry.mapped_rows * 2
    mean_row = geometry.hidden_size * 4
    mapping = geometry.mapped_rows * geometry.mapping_id_bytes
    exact_cache = preset.gpu_cache_rows * geometry.q6_row_bytes
    cache_ids = preset.gpu_cache_rows * geometry.mapping_id_bytes
    resident_bytes = sum(
        [
            scout_a,
            scout_a_scales,
            scout_b,
            residual_norms,
            mean_row,
            mapping,
            exact_cache,
            cache_ids,
        ]
    )

    macs = (
        preset.rank * geometry.hidden_size
        + geometry.mapped_rows * preset.rank
        + preset.exact_candidate_rows * geometry.hidden_size
        + geometry.hidden_size
    )
    worst_upload = preset.exact_candidate_rows * geometry.q6_row_bytes
    return {
        **asdict(preset),
        "macs_per_token": macs,
        "dense_mac_fraction": macs / geometry.dense_macs,
        "ideal_mac_reduction_x": geometry.dense_macs / macs,
        "gpu_resident_bytes": resident_bytes,
        "gpu_resident_mib": resident_bytes / MIB,
        "resident_fraction_vs_current_package": resident_bytes
        / geometry.dense_package_bytes,
        "cold_exact_projection_bytes": geometry.dense_projection_bytes,
        "worst_case_row_upload_bytes_per_token": worst_upload,
        "worst_case_row_upload_mib_per_token": worst_upload / MIB,
        "components": {
            "scout_a_q8": scout_a,
            "scout_a_f16_scales": scout_a_scales,
            "scout_b_f16": scout_b,
            "residual_norms_f16": residual_norms,
            "center_mean_row_f32": mean_row,
            "mapping_i64": mapping,
            "exact_q6_gpu_cache": exact_cache,
            "cache_ids_i64": cache_ids,
        },
    }


def build_report(geometry: Geometry) -> dict[str, object]:
    presets = [
        CascadePreset("r32-c256-cache4096", 32, 256, 4_096),
        CascadePreset("r64-c512-cache8192", 64, 512, 8_192),
        CascadePreset("r96-c1024-cache16384", 96, 1_024, 16_384),
    ]
    return {
        "format": "colorlm-output-head-cascade-static-cost-v1",
        "geometry": asdict(geometry),
        "current_dense": {
            "projection_bytes": geometry.dense_projection_bytes,
            "package_bytes": geometry.dense_package_bytes,
            "package_mib": geometry.dense_package_bytes / MIB,
            "macs_per_token": geometry.dense_macs,
        },
        "presets": [model_preset(geometry, preset) for preset in presets],
        "limitations": [
            "MAC 数不是实测 token/s。",
            "GPU resident 估算不含 GGML 对齐、图工作区、临时 logits 和后端元数据。",
            "最坏上传量假设全部候选行 cache miss；真实值必须由行缓存 trace 给出。",
            "若误差上界不能在候选上限内认证，必须回退稠密头。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="输出头级联方案静态成本模型")
    parser.add_argument("--mapped-rows", type=int, default=131_612)
    parser.add_argument("--hidden-size", type=int, default=2_048)
    parser.add_argument("--q6-row-bytes", type=int, default=1_680)
    args = parser.parse_args()
    geometry = Geometry(args.mapped_rows, args.hidden_size, args.q6_row_bytes)
    print(json.dumps(build_report(geometry), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
