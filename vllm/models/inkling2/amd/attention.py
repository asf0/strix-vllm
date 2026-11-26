# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 attention: relative-bias GQA with short-conv K/V, L2-clipped
q/k/r, gated output and per-token attention sinks."""

from typing import ClassVar, cast

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.inkling.amd.sconv_swa_attn import InklingConvState
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayerConfig,
    get_kv_quant_mode,
)

from ..configs import Inkling2ModelConfig
from .ops import attention as port_ops
from .ops import sconv
from .ops.attention import (
    gated_rel_attention,
    plan_decode_attention,
    rel_decode_attention,
)
from .ops.sconv import attention_prologue, shift_rel_proj
from .sconv import K_STREAM, V_STREAM, Inkling2ShortConv, conv_metadata


class Inkling2AttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Orders decode/verify requests first."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)


class Inkling2AttentionBackend(FlashAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "int8_per_token_head",
    ]

    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        # int8_per_token_head packs one fp32 scale after each head's data;
        # publish the same inline padding the Triton backend's kernels read.
        return TritonAttentionBackend.customize_spec(spec)

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return Inkling2AttentionMetadataBuilder


def inkling2_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`x @ weight.T` for the attention projections (bf16 `weight`, or a row
    slice of one). A kernel plugin may replace it, e.g. with int8 decode copies.
    """
    return nn.functional.linear(x, weight)


# A kernel plugin may replace `gated_rel_attention` and `rel_decode_attention`
# in this module. An int8 cache needs it to: those and `rel_attention_prefill`
# then also get the fp32 `k_scale` / `v_scale` views `(num_blocks, N, Hkv)`.

# Prefill relative attention from the clipped r instead of the sheared logits
# (a kernel plugin may set it): called as `gated_rel_attention` with `r`
# `(T, H, d_rel)` and the `(d_rel, extent)` `rel_proj` in place of
# `sheared_bias`; its `supports(num_heads, num_kv_heads, head_dim,
# block_size, extent, window)` attribute says which layers it takes.
rel_attention_prefill = None


def _check_cache_config(cache_config: CacheConfig) -> None:
    """Refuse a KV cache the kernels at hand cannot write and read."""
    cache_dtype = cache_config.cache_dtype
    if cache_dtype in ("auto", "bfloat16"):
        return
    if cache_dtype != "int8_per_token_head":
        raise ValueError(
            "Inkling 2 KV cache dtype must be auto, bfloat16 or "
            f"int8_per_token_head (got {cache_dtype!r})"
        )
    if cache_config.block_size % 32:
        raise ValueError(
            "Inkling 2 int8 KV cache needs a KV block size that is a "
            f"multiple of 32 (got {cache_config.block_size})"
        )
    # The port's own kernels would run bf16 code on the int8 cache.
    missing = [
        name
        for name, absent in (
            ("inkling2_attention", rel_attention_prefill is None),
            (
                "inkling2_rel_decode",
                rel_decode_attention is port_ops.rel_decode_attention,
            ),
            ("inkling2_kv_conv", sconv.kv_conv_prefill is None),
        )
        if absent
    ]
    if missing:
        raise ValueError(
            "Inkling 2 int8 KV cache needs the libstrix kernels "
            f"({', '.join(missing)}) that are missing or disabled; "
            "run with --kv-cache-dtype bfloat16 or enable them"
        )


def _kv_cache_layer_config(
    vllm_config: VllmConfig,
    num_kv_heads: int,
    head_dim: int,
    sliding_window: int | None,
) -> KVCacheLayerConfig:
    cache_dtype = vllm_config.cache_config.cache_dtype
    spec = AttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=kv_cache_dtype_str_to_dtype(cache_dtype, vllm_config.model_config),
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
    )
    if sliding_window is not None:
        return SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)
    return FullAttentionManager.layer_config(spec)


class Inkling2CacheLayer(AttentionLayerBase):
    """A layer with a paged KV cache and the kernels' views of it."""

    # Of the bound cache: the `(num_blocks, N, Hkv, D)` key and value caches,
    # and the `k_scale` / `v_scale` views of an int8 cache (empty for bf16).
    kv_views: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | None = None

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        super().bind_kv_cache(kv_cache)
        # `[K | V]` per (head, slot); int8_per_token_head appends one fp32
        # scale to each half (see TritonAttentionBackend.customize_spec).
        half = kv_cache.shape[-1] // 2
        k, v = kv_cache.transpose(1, 2).split(half, dim=-1)
        scales = {}
        if kv_cache.dtype == torch.int8:
            scales = {
                "k_scale": k.view(torch.float32)[..., -1],
                "v_scale": v.view(torch.float32)[..., -1],
            }
            k, v = k[..., :-4], v[..., :-4]
        self.kv_views = (
            canonicalize_singleton_dim_strides(k),
            canonicalize_singleton_dim_strides(v),
            scales,
        )


class Inkling2AttentionPassCache(nn.Module, Inkling2CacheLayer):
    """The KV cache of one extra loop pass of a looped `Inkling2Attention`
    (each pass of the loop core attends over its own keys and values)."""

    def __init__(
        self,
        prefix: str,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        # Read by the metadata builders of the layer's attention group.
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.kv_cache = torch.tensor([])
        forward_context = get_current_vllm_config().compilation_config
        if prefix in forward_context.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context.static_forward_context[prefix] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config, self.num_kv_heads, self.head_dim, self.sliding_window
        )


class Inkling2Attention(nn.Module, Inkling2CacheLayer):
    """Inkling2 attention for one decoder layer.

    Local layers project all of q/k/v from one input; full layers project the
    query side (q, r, gate, sink) from `x_q`, k from `x_k` and v from
    `x_v`, and use dynamic K/V convolutions and log-scaled q/r. Local layers
    given separate `x_k` / `x_v` (loop passes after the first) project them
    as full layers do. A looped layer has per-pass conv-state caches:
    `conv_states[p]` and pass `p`'s KV cache (this layer's for pass 0,
    `pass_caches[p - 1]` after); with `loop_share_cache` all passes share
    pass 0's KV cache.
    """

    def __init__(
        self,
        config: Inkling2ModelConfig,
        is_local: bool,
        conv_state: InklingConvState,
        prefix: str,
        pass_conv_states: tuple[InklingConvState, ...] = (),
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.is_local = is_local
        num_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        self.d_rel = config.d_rel
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window_size
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.window = config.sliding_window_size - 1 if is_local else -1
        self.qk_clip = config.qk_l2_clip_threshold
        self.rel_clip = config.rel_l2_clip_threshold
        self.eps = config.rms_norm_eps
        assert config.use_value_norm_no_gamma

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Query-side outputs first so full layers can project them in one GEMM.
        self.split_sizes = [
            q_size,
            self.num_heads * self.d_rel,
            q_size,
            self.num_heads,
            kv_size,
            kv_size,
        ]
        # Full layers project the query side alone; rounding its N up to 8
        # keeps the output rows 16-byte aligned for hipBLASLt (the extra
        # columns, from the K rows, are discarded).
        q_side = sum(self.split_sizes[:4])
        self.q_side_splits = [*self.split_sizes[:4], -q_side % 8]
        pad_size = -sum(self.split_sizes) % 8
        if pad_size:
            self.split_sizes.append(pad_size)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * tp_size for size in self.split_sizes],
            bias=False,
            prefix=f"{prefix}.qkvr",
        )
        self.wo_ud = RowParallelLinear(
            q_size * tp_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            prefix=f"{prefix}.wo_ud",
        )
        self.rel_logits_proj = nn.Parameter(
            torch.empty(self.d_rel, self.rel_extent), requires_grad=False
        )
        self.q_scale = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim), requires_grad=False
        )
        self.r_scale = nn.Parameter(
            torch.empty(self.num_heads, self.d_rel), requires_grad=False
        )
        for param in (self.q_scale, self.r_scale):
            set_weight_attrs(
                param,
                {
                    "weight_loader": lambda p, w: p.data.copy_(
                        w.narrow(0, tp_rank * p.shape[0], p.shape[0])
                    )
                },
            )

        self.conv_state = conv_state
        self.conv_states = (conv_state, *pass_conv_states)
        self.pass_caches = nn.ModuleList(
            Inkling2AttentionPassCache(
                f"{prefix}.pass{p}",
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.sliding_window if is_local else None,
            )
            for p in range(1, 1 if config.loop_share_cache else len(self.conv_states))
        )
        self.k_sconv, self.v_sconv = (
            Inkling2ShortConv(
                kv_size,
                config.sconv_kernel_size,
                conv_state,
                stream,
                dynamic_input_size=None if is_local else config.hidden_size,
                num_heads=self.num_kv_heads,
            )
            for stream in (K_STREAM, V_STREAM)
        )

        vllm_config = get_current_vllm_config()
        _check_cache_config(vllm_config.cache_config)
        self.kv_cache = torch.tensor([])
        forward_context = vllm_config.compilation_config.static_forward_context
        if prefix in forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        forward_context[prefix] = self

    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2AttentionBackend

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        return _kv_cache_layer_config(
            vllm_config,
            self.num_kv_heads,
            self.head_dim,
            self.sliding_window if self.is_local else None,
        )

    def _pass_cache(self, pass_idx: int) -> Inkling2CacheLayer:
        """The layer owning the KV cache of loop pass `pass_idx`."""
        if pass_idx == 0 or not self.pass_caches:
            return self
        return self.pass_caches[pass_idx - 1]

    def _project(
        self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor
    ) -> list[torch.Tensor]:
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = inkling2_linear(x_q, weight)
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = inkling2_linear(x_q, weight[: sum(self.q_side_splits)])
        k = inkling2_linear(x_k, weight[q_side:k_end])
        v = inkling2_linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        """Return this rank's bf16 `(T, hidden)` partial output (to be
        all-reduced over TP), written to `out` if given.

        Local layers read q/k/v from `x_q` (or `x_k` / `x_v` if given); full
        layers take `x_k`, `x_v` and the fp32 log-scaling `tau` for q/r.
        `pass_idx` picks the loop pass's caches.
        """
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
        )
        projected = inkling2_linear(output, self.wo_ud.weight)
        if out is None:
            return projected
        return out.copy_(projected)

    @eager_break_during_capture
    def _attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_dynamic: torch.Tensor | None,
        v_dynamic: torch.Tensor | None,
        tau: torch.Tensor | None,
        output: torch.Tensor,
        pass_idx: int = 0,
    ) -> None:
        """Write the gated attention output to `output`.

        The prologue's layout and the attention plan follow the batch's
        requests, so this runs eagerly in piecewise CUDA graphs, which are
        captured on a dummy batch and replayed for any batch of the same
        token count.
        """
        attn_metadata = get_forward_context().attn_metadata
        cache = self._pass_cache(pass_idx)
        conv_state = self.conv_states[pass_idx]
        if not isinstance(attn_metadata, dict) or cache.kv_views is None:
            output.zero_()
            return
        md = cast(FlashAttentionMetadata, attn_metadata[cache.prefix])
        conv_md = conv_metadata(conv_state)
        assert conv_md is not None
        num_tokens = q.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        key_cache, value_cache, scales = cache.kv_views
        clipped_q = q.new_empty(q.shape)
        # Decode and verify batches split the keys; the split count follows
        # the batch alone, so it holds for every replay of a CUDA graph.
        num_splits = plan_decode_attention(
            self.is_local,
            md.seq_lens.shape[0],
            md.max_query_len,
            self.num_heads,
            self.num_kv_heads,
            self.rel_extent,
        )
        fused_rel = (
            not num_splits
            and rel_attention_prefill is not None
            and rel_attention_prefill.supports(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                key_cache.shape[1],
                self.rel_extent,
                self.window,
            )
        )
        if fused_rel:
            rel_logits = q.new_empty((num_tokens, self.num_heads, self.d_rel))
        else:
            # Sheared as `rel_attention` reads it (see attention_prologue).
            rel_logits = q.new_empty(
                (num_tokens + 128, self.num_heads, self.rel_extent + 256)
            )
        attention_prologue(
            q,
            r.view(num_tokens, self.num_heads, self.d_rel),
            k,
            v,
            self.q_scale,
            self.r_scale,
            tau,
            self.rel_proj_shifted,
            self.k_sconv.conv_weight,
            self.v_sconv.conv_weight,
            k_dynamic,
            v_dynamic,
            conv_state.kv_cache,
            positions,
            conv_md.block_table,
            conv_md.seq_idx,
            conv_md.slot_mapping,
            conv_md.query_start,
            key_cache,
            value_cache,
            md.slot_mapping,
            clipped_q,
            rel_logits,
            md.query_start_loc,
            md.seq_lens,
            clip_threshold=self.qk_clip,
            rel_threshold=self.rel_clip,
            norm_eps=self.eps,
            r_only=fused_rel,
        )
        if num_splits:
            rel_decode_attention(
                clipped_q,
                key_cache,
                value_cache,
                rel_logits,
                gate,
                sinks,
                output,
                block_table=md.block_table,
                cache_seqlens=md.seq_lens,
                cu_seqlens_q=md.query_start_loc,
                max_query_len=md.max_query_len,
                num_actual_tokens=md.num_actual_tokens,
                softmax_scale=self.scaling,
                is_local=self.is_local,
                num_splits=num_splits,
                **scales,
            )
            return
        # Fused: r and the projection in place of the sheared logits.
        rel_args = (rel_logits, self.rel_logits_proj) if fused_rel else (rel_logits,)
        attend = rel_attention_prefill if fused_rel else gated_rel_attention
        attend(
            clipped_q,
            key_cache,
            value_cache,
            *rel_args,
            gate,
            sinks,
            output,
            block_table=md.block_table,
            cache_seqlens=md.seq_lens,
            cu_seqlens_q=md.query_start_loc,
            max_seqlen_q=md.max_query_len,
            num_actual_tokens=md.num_actual_tokens,
            softmax_scale=self.scaling,
            window=self.window,
            **scales,
        )
