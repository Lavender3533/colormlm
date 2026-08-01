"""DeepSeek-V4 S14 的纯 PyTorch/CPU 数值参考原语。

这里的实现用于语义对齐和小规模确定性测试，不是性能内核，也不表示 S14 已经跑通。
"""

from .attention import sparse_attention
from .fp8 import (
    FP8_GROUP_SIZE,
    decode_fp8_e4m3fn,
    decode_scaled_fp8_e4m3fn,
    fp8_linear,
    fp8_weight_linear,
)
from .hc import hc_post, hc_pre, hc_split_sinkhorn
from .mxfp4 import (
    FP4_E2M1_VALUES,
    FP4_GROUP_SIZE,
    FP8_ACTIVATION_GROUP_SIZE,
    decode_mxfp4,
    decode_ue8m0,
    fp4_linear,
    unpack_mxfp4_e2m1,
)

__all__ = [
    "FP4_E2M1_VALUES",
    "FP4_GROUP_SIZE",
    "FP8_ACTIVATION_GROUP_SIZE",
    "FP8_GROUP_SIZE",
    "decode_fp8_e4m3fn",
    "decode_mxfp4",
    "decode_scaled_fp8_e4m3fn",
    "decode_ue8m0",
    "fp4_linear",
    "fp8_linear",
    "fp8_weight_linear",
    "hc_post",
    "hc_pre",
    "hc_split_sinkhorn",
    "sparse_attention",
    "unpack_mxfp4_e2m1",
]
