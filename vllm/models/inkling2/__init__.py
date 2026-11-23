# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2: platform-specific implementations live under `nvidia/` and `amd/`;
configs, processing and the DSpark speculator are shared."""

from importlib import import_module
from typing import TYPE_CHECKING

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from .nvidia.dspark import Inkling2DSparkModel, Inkling2MarkovHead
    from .nvidia.loop import Inkling2LoopForConditionalGeneration
    from .nvidia.model import Inkling2ForConditionalGeneration
    from .nvidia.ops.attention import (
        block_decode_attention,
        gate_sink,
        rel_decode_attention,
    )
    from .nvidia.ops.attn_res import attn_res
    from .nvidia.ops.dspark import add_softcap, markov_gate, mix_rmsnorm
    from .nvidia.ops.norm import rms_norm
    from .nvidia.ops.sconv import attention_prologue, shift_rel_proj, short_conv
    from .nvidia.ops.vision import gather_patches
    from .nvidia.vision import Inkling2VisionEncoder
    from .quant_config import Inkling2QuantConfig

# Each export's module under the platform's package (`amd/` on ROCm,
# `nvidia/` elsewhere), or at the package root for the shared ones.
_MODULES = {
    "Inkling2DSparkModel": "dspark",
    "Inkling2ForConditionalGeneration": "model",
    "Inkling2LoopForConditionalGeneration": "loop",
    "Inkling2MarkovHead": "dspark",
    "Inkling2QuantConfig": "quant_config",
    "Inkling2VisionEncoder": "vision",
    "add_softcap": "ops.dspark",
    "attention_prologue": "ops.sconv",
    "attn_res": "ops.attn_res",
    "block_decode_attention": "ops.attention",
    "gate_sink": "ops.attention",
    "gather_patches": "ops.vision",
    "markov_gate": "ops.dspark",
    "mix_rmsnorm": "ops.dspark",
    "rel_decode_attention": "ops.attention",
    "rms_norm": "ops.norm",
    "shift_rel_proj": "ops.sconv",
    "short_conv": "ops.sconv",
}
_SHARED = {"quant_config"}

__all__ = [
    "Inkling2DSparkModel",
    "Inkling2ForConditionalGeneration",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2MarkovHead",
    "Inkling2QuantConfig",
    "Inkling2VisionEncoder",
    "add_softcap",
    "attention_prologue",
    "attn_res",
    "block_decode_attention",
    "gate_sink",
    "gather_patches",
    "markov_gate",
    "mix_rmsnorm",
    "rel_decode_attention",
    "rms_norm",
    "shift_rel_proj",
    "short_conv",
]


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _MODULES[name]
    if module in _SHARED:
        package = __name__
    elif current_platform.is_rocm():
        package = f"{__name__}.amd"
    else:
        package = f"{__name__}.nvidia"
    return getattr(import_module(f"{package}.{module}"), name)
