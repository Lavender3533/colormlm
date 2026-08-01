"""冻结的 FullDepth43/native-top6 profile。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
N_LAYERS = 43
TOP_K = 6
ROUTE_SCALE = 1.5
LAYERS = tuple(range(N_LAYERS))
COMPRESS_RATIOS = (0, 0) + tuple(4 if layer % 2 == 0 else 128 for layer in range(2, N_LAYERS))


class ProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionProfile:
    profile_id: str
    repo: str
    revision: str
    layers: tuple[int, ...]
    top_k: int
    route_scale: float
    compress_ratios: tuple[int, ...]
    skipped_layer_semantics: str

    def validate(self) -> None:
        if self.repo != REPO or self.revision != REVISION:
            raise ProfileError("拒绝非冻结 donor revision")
        if self.profile_id != "fulldepth43_native_top6":
            raise ProfileError("拒绝非 FullDepth43/native-top6 profile")
        if self.layers != LAYERS or self.top_k != TOP_K:
            raise ProfileError("必须覆盖原生 0..42 全部层且使用 top-6")
        if self.route_scale != ROUTE_SCALE:
            raise ProfileError("原生 route scale 必须精确为 1.5")
        if self.compress_ratios != COMPRESS_RATIOS:
            raise ProfileError("43 层 compressor ratio 与固定 config 不一致")
        if self.skipped_layer_semantics != "forbidden":
            raise ProfileError("FullDepth profile 禁止 identity skip")

    def ratio_for(self, layer: int) -> int:
        if layer not in self.layers:
            raise ProfileError(f"layer {layer} 不属于冻结 profile")
        return self.compress_ratios[layer]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "id": self.profile_id,
            "repo": self.repo,
            "revision": self.revision,
            "layers": list(self.layers),
            "top_k": self.top_k,
            "route_scale": self.route_scale,
            "compress_ratios": list(self.compress_ratios),
            "skipped_layer_semantics": self.skipped_layer_semantics,
        }


FULLDEPTH43_NATIVE_TOP6 = ExecutionProfile(
    profile_id="fulldepth43_native_top6",
    repo=REPO,
    revision=REVISION,
    layers=LAYERS,
    top_k=TOP_K,
    route_scale=ROUTE_SCALE,
    compress_ratios=COMPRESS_RATIOS,
    skipped_layer_semantics="forbidden",
)
FULLDEPTH43_NATIVE_TOP6.validate()
