# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN RecoverSSM speculative decode against the per-draft-slot spec path.

RecoverSSM must reproduce the existing spec kernel exactly: the same outputs
during verify, and after the commit the same recurrent state and conv window
the baseline would read from slot ``num_accepted - 1`` on the next step.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip(
        reason="GDN RecoverSSM kernels require a CUDA or ROCm GPU.",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops import gdn_recoverssm  # noqa: E402
from vllm.model_executor.layers.mamba.ops.gdn_recoverssm import (  # noqa: E402
    GDNRecoverSSMCommitContext,
    gdn_recoverssm_verify,
)
from vllm.model_executor.models.qwen3_5 import (  # noqa: E402
    Qwen3_5ForConditionalGeneration,
)
from vllm.third_party.flash_linear_attention.ops import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
    GDNRecoverSSMAttentionMetadata,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

DEVICE = torch.device("cuda")
NUM_SPEC = 2
SPEC_LEN = NUM_SPEC + 1
H = 2  # key heads
HV = 6  # value heads; GQA ratio 3 as in Qwen3.5
K = 128
V = 128
CONV_KERNEL = 4
HISTORY = CONV_KERNEL - 1
PREFIX = "model.layers.0.linear_attn"


def _records(num_blocks: int) -> tuple[torch.Tensor, ...]:
    shapes = MambaStateShapeCalculator.append_gdn_recoverssm_records(
        (), 1, H, HV, K, V, spec_query_len=SPEC_LEN
    )
    return tuple(
        torch.full((num_blocks, *shape), float("nan"), device=DEVICE)
        for shape in shapes
    )


def test_recoverssm_config_state_layout():
    """The class-level page layout must match what the layer allocates."""
    vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=types.SimpleNamespace(
                linear_num_key_heads=16,
                linear_num_value_heads=48,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        cache_config=types.SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype="float32",
            use_gdn_recoverssm=True,
        ),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=2),
    )
    model_cls = Qwen3_5ForConditionalGeneration

    assert model_cls.get_mamba_state_dtype_from_config(vllm_config) == (
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.float32,
    )
    assert model_cls.get_mamba_state_shape_from_config(vllm_config)[1:] == (
        (48, 128, 128),
        (48, 3, 128),
        (16, 3, 128),
        (48, 3),
    )


@pytest.mark.parametrize(
    ("align_mode", "use_request_indices", "conv_state_dim_first"),
    [
        pytest.param(False, False, True, id="none"),
        pytest.param(False, True, False, id="request-indexed"),
        pytest.param(True, False, True, id="aligned"),
        pytest.param(True, True, False, id="aligned-request-indexed"),
    ],
)
@torch.inference_mode()
def test_verify_and_commit_match_spec_kernel(
    monkeypatch: pytest.MonkeyPatch,
    align_mode: bool,
    use_request_indices: bool,
    conv_state_dim_first: bool,
):
    monkeypatch.setattr(
        gdn_recoverssm, "is_conv_state_dim_first", lambda: conv_state_dim_first
    )
    torch.manual_seed(20261004)
    num_layers, num_blocks, conv_dim = 2, 24, 16
    query_lens = [3, 2, 3, 3, 3]
    accepted = [2, 1, 1, 3, 3]
    num_seqs = len(query_lens)
    total_tokens = sum(query_lens)
    query_start_loc = torch.tensor(
        [0, 3, 5, 8, 11, 14], dtype=torch.int32, device=DEVICE
    )
    # Baseline keeps the state after draft i in slot i; RecoverSSM only uses
    # slot 0, the checkpoint.
    slots = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [17, 18, 19]]
    source = torch.tensor([s[0] for s in slots], dtype=torch.int32, device=DEVICE)

    rows = [0, 2, 3, 5, 6] if use_request_indices else list(range(num_seqs))
    batch_size = rows[-1] + 1
    request_indices = (
        torch.tensor(rows, dtype=torch.int32, device=DEVICE)
        if use_request_indices
        else None
    )
    num_sampled = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
    num_sampled[rows] = torch.tensor(accepted, dtype=torch.int32, device=DEVICE)

    # Align mode, 4-token blocks: seq 0 crosses a boundary after 1 accepted
    # token; seq 2 drafted past one but accepts none of it, so the state
    # moves back to the earlier block; seq 4 lands exactly on a boundary
    # while the next block is still unallocated (null), so the state must
    # stay in the block holding its last token.
    final_blocks = [s[0] for s in slots]
    boundary: dict[int, tuple[int, int]] = {}
    block_table = num_computed = None
    if align_mode:
        block_table = torch.full((batch_size, 2), -1, dtype=torch.int32, device=DEVICE)
        block_table[rows] = torch.tensor(
            [[13, 1], [4, 14], [15, 7], [16, 10], [17, 0]],
            dtype=torch.int32,
            device=DEVICE,
        )
        num_computed = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
        num_computed[rows] = torch.tensor(
            [3, 1, 2, 4, 1], dtype=torch.int32, device=DEVICE
        )
        final_blocks = [1, 4, 15, 10, 17]
        # seq -> (block, tokens up to the boundary)
        boundary = {0: (13, 1), 4: (17, 3)}

    q, k = (
        torch.randn(1, total_tokens, H, K, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(2)
    )
    v = torch.randn(1, total_tokens, HV, V, dtype=torch.bfloat16, device=DEVICE)
    b, a = torch.randn(total_tokens, 2 * HV, dtype=torch.bfloat16, device=DEVICE).chunk(
        2, dim=-1
    )

    layers = []
    expected = []
    for _ in range(num_layers):
        A_log = 0.2 * torch.randn(HV, dtype=torch.float32, device=DEVICE)
        dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=DEVICE)
        state = 0.05 * torch.randn(num_blocks, HV, V, K, device=DEVICE)
        conv_shape = (
            (num_blocks, conv_dim, HISTORY + NUM_SPEC)
            if conv_state_dim_first
            else (num_blocks, HISTORY + NUM_SPEC, conv_dim)
        )
        conv = torch.randn(conv_shape, dtype=torch.bfloat16, device=DEVICE)

        baseline_state = state.clone()
        baseline_out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            initial_state=baseline_state,
            inplace_final_state=True,
            cu_seqlens=query_start_loc,
            ssm_state_indices=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
            num_accepted_tokens=torch.ones(num_seqs, dtype=torch.int32, device=DEVICE),
            use_qk_l2norm_in_kernel=True,
        )

        records = _records(num_blocks)
        checkpoint = state.clone()
        out = gdn_recoverssm_verify(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=checkpoint,
            correction_cache=records[0],
            key_cache=records[1],
            decay_cache=records[2],
            query_start_loc=query_start_loc,
            state_indices=source,
            spec_query_len=SPEC_LEN,
        )
        torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)
        torch.testing.assert_close(checkpoint, state, atol=0, rtol=0)

        layers.append(types.SimpleNamespace(kv_cache=(conv, checkpoint, *records)))
        expected.append((baseline_state, conv.clone()))

    context = GDNRecoverSSMCommitContext.create(
        layers, spec_query_len=SPEC_LEN, max_num_reqs=batch_size
    )
    context.commit(
        num_sampled,
        source,
        query_start_loc,
        request_indices=request_indices,
        block_table=block_table,
        num_computed_tokens=num_computed,
        mamba_block_size=4 if align_mode else None,
    )

    # (seq, block, tokens folded in): the new position plus a crossed boundary.
    commits = [(seq, final_blocks[seq], accepted[seq]) for seq in range(num_seqs)]
    commits += [(seq, block, n) for seq, (block, n) in boundary.items()]
    for layer, (baseline_state, conv_before) in zip(layers, expected):
        conv_after = layer.kv_cache[0]
        if not conv_state_dim_first:
            conv_before = conv_before.transpose(-1, -2)
            conv_after = conv_after.transpose(-1, -2)
        for seq, block, num_tokens in commits:
            torch.testing.assert_close(
                layer.kv_cache[1][block],
                baseline_state[slots[seq][num_tokens - 1]],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                conv_after[block, :, :HISTORY],
                conv_before[
                    slots[seq][0], :, num_tokens - 1 : num_tokens - 1 + HISTORY
                ],
                atol=0,
                rtol=0,
            )


def _make_vllm_config():
    config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=16,
        hf_config_override={"linear_key_head_dim": K},
    )
    config.cache_config.mamba_cache_mode = "none"
    config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=NUM_SPEC
    )
    return config


def _build_layer(kv_cache, A_log, dt_bias, conv_weight, use_recoverssm):
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=H * K,
        value_dim=HV * V,
        activation="silu",
        A_log=A_log,
        dt_bias=dt_bias,
        conv1d=types.SimpleNamespace(weight=conv_weight, bias=None),
        kv_cache=kv_cache,
        num_spec=NUM_SPEC,
        use_recoverssm=use_recoverssm,
    )
    for name in ("rearrange_mixed_qkv", "_forward_core"):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


@torch.inference_mode()
def test_forward_core_two_steps_match_spec_slots():
    """Two MTP steps through the real builder and ``_forward_core``.

    The second step only matches if the commit left the checkpoint and conv
    window where the baseline reads them via ``num_accepted_tokens``.
    """
    torch.manual_seed(7)
    vllm_config = _make_vllm_config()
    conv_dim = 2 * H * K + HV * V
    steps = [
        # (query_lens, draft tokens, sampled tokens)
        ([3, 2], [2, 1], [2, 1]),
        ([3, 3], [2, 2], [3, 2]),
    ]
    slots = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device=DEVICE)
    num_blocks = 7

    conv_shape, state_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, NUM_SPEC
    )
    conv_seed = 0.1 * torch.randn(
        num_blocks, *conv_shape, dtype=torch.bfloat16, device=DEVICE
    )
    state_seed = 0.05 * torch.randn(num_blocks, *state_shape, device=DEVICE)
    A_log = 0.2 * torch.randn(HV, dtype=torch.float32, device=DEVICE)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=DEVICE)
    conv_weight = 0.2 * torch.randn(
        conv_dim, 1, CONV_KERNEL, dtype=torch.bfloat16, device=DEVICE
    )

    layers, builders = {}, {}
    for use_recoverssm in (False, True):
        vllm_config.cache_config.use_gdn_recoverssm = use_recoverssm
        kv_cache = (conv_seed.clone(), state_seed.clone())
        if use_recoverssm:
            kv_cache += _records(num_blocks)
        layer = _build_layer(kv_cache, A_log, dt_bias, conv_weight, use_recoverssm)
        vllm_config.compilation_config.static_forward_context[PREFIX] = layer
        builders[use_recoverssm] = GDNAttentionMetadataBuilder(
            kv_cache_spec=MambaSpec(
                block_size=16,
                shapes=((16, 64),),
                dtypes=(torch.float16,),
                num_speculative_blocks=0 if use_recoverssm else NUM_SPEC,
            ),
            layer_names=[PREFIX],
            vllm_config=vllm_config,
            device=DEVICE,
        )
        layers[use_recoverssm] = layer

    num_accepted = torch.ones(2, dtype=torch.int32, device=DEVICE)
    for query_lens, draft_tokens, sampled in steps:
        num_tokens = sum(query_lens)
        mixed_qkv = 0.3 * torch.randn(
            num_tokens, conv_dim, dtype=torch.bfloat16, device=DEVICE
        )
        b, a = torch.randn(
            num_tokens, 2 * HV, dtype=torch.bfloat16, device=DEVICE
        ).chunk(2, dim=-1)
        outputs = {}
        for use_recoverssm, layer in layers.items():
            common = create_common_attn_metadata(
                BatchSpec(seq_lens=[64, 64], query_lens=query_lens), 16, DEVICE
            )
            common.block_table_tensor[:, :3] = slots
            with set_current_vllm_config(vllm_config):
                meta = builders[use_recoverssm].build(
                    common_prefix_len=0,
                    common_attn_metadata=common,
                    num_accepted_tokens=num_accepted,
                    num_decode_draft_tokens_cpu=torch.tensor(
                        draft_tokens, dtype=torch.int32
                    ),
                )
            assert isinstance(meta, GDNRecoverSSMAttentionMetadata) == use_recoverssm
            out = torch.zeros(num_tokens, HV, V, dtype=torch.bfloat16, device=DEVICE)
            context = types.SimpleNamespace(attn_metadata={PREFIX: meta})
            with patch.object(
                qwen_gdn_linear_attn, "get_forward_context", return_value=context
            ):
                layer._forward_core(
                    mixed_qkv=mixed_qkv.clone(), b=b, a=a, core_attn_out=out
                )
            if use_recoverssm:
                meta.commit_recoverssm_state(
                    torch.tensor(sampled, dtype=torch.int32, device=DEVICE)
                )
            outputs[use_recoverssm] = out
        torch.testing.assert_close(outputs[True], outputs[False], atol=0, rtol=0)
        num_accepted = torch.tensor(sampled, dtype=torch.int32, device=DEVICE)

    baseline, recover = layers[False].kv_cache, layers[True].kv_cache
    conv_baseline, conv_recover = baseline[0], recover[0]
    if not is_conv_state_dim_first():
        conv_baseline = conv_baseline.transpose(-1, -2)
        conv_recover = conv_recover.transpose(-1, -2)
    for seq, num_tokens in enumerate(steps[-1][2]):
        checkpoint = int(slots[seq, 0])
        torch.testing.assert_close(
            recover[1][checkpoint],
            baseline[1][int(slots[seq, num_tokens - 1])],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            conv_recover[checkpoint, :, :HISTORY],
            conv_baseline[checkpoint, :, num_tokens - 1 : num_tokens - 1 + HISTORY],
            atol=0,
            rtol=0,
        )
