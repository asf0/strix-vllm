# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV cache of the ROCm Inkling 2 attention (bf16 and
int8_per_token_head): the views of a bound cache, the kernels a cache dtype
needs, and the arguments the kernels get (none of them runs)."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.models.inkling2.amd import attention
from vllm.models.inkling2.amd.ops import attention as port_ops
from vllm.models.inkling2.amd.ops import sconv
from vllm.v1.worker.utils import clear_layer_kv_caches

BLOCKS, NKV, PAGE, D = 2, 2, 4, 128
HALF = D + 4  # int8 values, then their fp32 scale
# A block's page is slot-major ([slot][head][content]) and padded.
PAGE_STRIDE = PAGE * NKV * 2 * HALF + 64
LAYER = BLOCKS * PAGE_STRIDE


class CacheLayer(attention.Inkling2CacheLayer):
    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_config(self, vllm_config):
        raise NotImplementedError


def int8_cache(storage: torch.Tensor, layer: int) -> torch.Tensor:
    """Layer `layer` of the packed storage, `(blocks, Hkv, page, content)`."""
    strides = (PAGE_STRIDE, 2 * HALF, NKV * 2 * HALF, 1)
    return storage.as_strided((BLOCKS, NKV, PAGE, 2 * HALF), strides, layer * LAYER)


def test_int8_views_address_the_layers_own_bytes():
    storage = torch.zeros(2 * LAYER, dtype=torch.int8)
    for i in range(2):
        layer = CacheLayer()
        layer.bind_kv_cache(int8_cache(storage, i))
        key, value, scales = layer.kv_views
        k_scale, v_scale = scales["k_scale"], scales["v_scale"]
        assert key.shape == value.shape == (BLOCKS, PAGE, NKV, D)
        assert k_scale.shape == v_scale.shape == (BLOCKS, PAGE, NKV)
        block, slot, head = BLOCKS - 1, PAGE - 1, NKV - 1
        codes = (torch.arange(1, D + 1) % 127).to(torch.int8)
        key[block, slot, head] = codes
        value[block, slot, head] = -codes
        k_scale[block, slot, head] = 0.5 + i
        v_scale[block, slot, head] = 0.25 + i
        off = i * LAYER + block * PAGE_STRIDE + (slot * NKV + head) * 2 * HALF
        content = storage[off : off + 2 * HALF].clone()
        assert torch.equal(content[:D], codes)
        assert torch.equal(content[HALF : HALF + D], -codes)
        assert content[D:HALF].view(torch.float32).item() == 0.5 + i
        assert content[HALF + D :].view(torch.float32).item() == 0.25 + i


def test_bf16_views_and_rebinding():
    layer = CacheLayer()
    first, second = (torch.zeros(BLOCKS, NKV, PAGE, 2 * D).bfloat16() for _ in "ab")
    for cache in (first, second):
        layer.bind_kv_cache(cache)
        key, value, scales = layer.kv_views
        assert scales == {}
        assert key.shape == value.shape == (BLOCKS, PAGE, NKV, D)
        assert key.data_ptr() == cache.data_ptr()
        assert value.data_ptr() == cache[..., D:].data_ptr()
    # The views must not keep a cleared cache's memory.
    clear_layer_kv_caches([layer])
    assert layer.kv_views is None


def test_int8_cache_needs_the_plugin_kernels(monkeypatch):
    def check(dtype, block_size=64):
        attention._check_cache_config(
            SimpleNamespace(cache_dtype=dtype, block_size=block_size)
        )

    monkeypatch.setattr(attention, "rel_attention_prefill", None)
    monkeypatch.setattr(
        attention, "rel_decode_attention", port_ops.rel_decode_attention
    )
    monkeypatch.setattr(sconv, "kv_conv_prefill", None)
    check("auto")
    check("bfloat16")
    with pytest.raises(ValueError, match="fp8"):
        check("fp8")
    kernels = "inkling2_attention, inkling2_rel_decode, inkling2_kv_conv"
    with pytest.raises(ValueError, match=kernels):
        check("int8_per_token_head")
    monkeypatch.setattr(attention, "rel_attention_prefill", lambda: None)
    monkeypatch.setattr(sconv, "kv_conv_prefill", lambda: None)
    with pytest.raises(ValueError, match=r"\(inkling2_rel_decode\)"):
        check("int8_per_token_head")
    monkeypatch.setattr(attention, "rel_decode_attention", lambda: None)
    check("int8_per_token_head")
    with pytest.raises(ValueError, match="multiple of 32"):
        check("int8_per_token_head", block_size=16)


def _run_attention(monkeypatch, kernels, kv_cache, is_local, num_splits):
    """`Inkling2Attention._attention` over stub metadata with `kernels` (name
    -> function) in place of the module's."""
    num_heads, d_rel, tokens = 3 * NKV, 16, 5
    layer = attention.Inkling2Attention.__new__(attention.Inkling2Attention)
    torch.nn.Module.__init__(layer)
    conv_state = SimpleNamespace(kv_cache=None)
    sconv_weights = SimpleNamespace(conv_weight=None)
    vars(layer).update(
        prefix="attn",
        is_local=is_local,
        num_heads=num_heads,
        num_kv_heads=NKV,
        head_dim=D,
        d_rel=d_rel,
        scaling=D**-0.5,
        rel_extent=1024,
        window=1023 if is_local else -1,
        qk_clip=1.0,
        rel_clip=1.0,
        eps=1e-6,
        q_scale=None,
        r_scale=None,
        rel_proj_shifted=None,
        rel_logits_proj=None,
        pass_caches=(),
        conv_states=(conv_state,),
        k_sconv=sconv_weights,
        v_sconv=sconv_weights,
    )
    layer.bind_kv_cache(kv_cache)
    md = SimpleNamespace(
        seq_lens=torch.zeros(1),
        max_query_len=tokens,
        slot_mapping=None,
        query_start_loc=None,
        block_table=None,
        num_actual_tokens=tokens,
    )
    conv_md = SimpleNamespace(
        block_table=None, seq_idx=None, slot_mapping=None, query_start=None
    )
    context = SimpleNamespace(attn_metadata={"attn": md})
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)
    monkeypatch.setattr(attention, "conv_metadata", lambda state: conv_md)
    monkeypatch.setattr(attention, "plan_decode_attention", lambda *a: num_splits)
    monkeypatch.setattr(attention, "rel_attention_prefill", None)
    for name, kernel in kernels.items():
        monkeypatch.setattr(attention, name, kernel)
    q = torch.zeros(tokens, num_heads * D).bfloat16()
    layer._attention(
        None,
        q,
        torch.zeros(tokens, num_heads * d_rel).bfloat16(),
        gate=q,
        sinks=None,
        k=None,
        v=None,
        k_dynamic=None,
        v_dynamic=None,
        tau=None,
        output=q,
    )


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("num_splits", [0, 4])
def test_kernels_get_the_arguments_they_take(monkeypatch, is_local, num_splits, int8):
    """The port's kernels run a bf16 cache without a plugin, so they must
    take every argument; an int8 cache adds only the scale views."""
    calls = {}

    def recorder(name, port):
        def kernel(*args, **kwargs):
            scales = {n: kwargs.pop(n) for n in ("k_scale", "v_scale") if n in kwargs}
            inspect.signature(port).bind(*args, **kwargs)
            calls[name] = scales

        return kernel

    if int8:
        cache = int8_cache(torch.zeros(LAYER, dtype=torch.int8), 0)
    else:
        cache = torch.zeros(BLOCKS, NKV, PAGE, 2 * D).bfloat16()
    kernels = {
        "attention_prologue": sconv.attention_prologue,
        "gated_rel_attention": port_ops.gated_rel_attention,
        "rel_decode_attention": port_ops.rel_decode_attention,
    }
    _run_attention(
        monkeypatch,
        {name: recorder(name, port) for name, port in kernels.items()},
        cache,
        is_local,
        num_splits,
    )
    attend = "rel_decode_attention" if num_splits else "gated_rel_attention"
    assert set(calls) == {"attention_prologue", attend}
    assert calls["attention_prologue"] == {}
    assert set(calls[attend]) == ({"k_scale", "v_scale"} if int8 else set())
