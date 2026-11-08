# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.models.qwen4_exp.amd.mtp import (
    _reduce_lm_head,
    _scatter_draft_logits,
)


@pytest.mark.cpu_test
def test_reduced_mtp_head_preserves_selected_logits_and_argmax(default_vllm_config):
    torch.manual_seed(0)
    head = ParallelLMHead(12, 3, params_dtype=torch.float32, disable_tp=True)
    with torch.no_grad():
        head.weight[:12].copy_(torch.arange(36, dtype=torch.float32).view(12, 3) / 100)

    token_id_list = [0, 1, 2, 8, 11]
    reduced = _reduce_lm_head(head, token_id_list)
    token_ids = torch.tensor(token_id_list)
    torch.testing.assert_close(reduced.weight[:5], head.weight[token_ids])

    hidden_states = torch.randn(4, 3)
    processor = LogitsProcessor(12)
    reduced_logits = processor(reduced, hidden_states)
    logits = _scatter_draft_logits(reduced_logits, token_ids, 12)

    assert logits.shape == (4, 12)
    omitted = [i for i in range(12) if i not in token_id_list]
    assert torch.isneginf(logits[:, omitted]).all()
    torch.testing.assert_close(logits[:, token_ids], reduced_logits[:, :5])

    reduced_top = processor.get_top_tokens(reduced, hidden_states)
    torch.testing.assert_close(token_ids[reduced_top], logits.argmax(dim=-1))


@pytest.mark.cpu_test
def test_reduced_mtp_head_rejects_tensor_parallel():
    head = ParallelLMHead(12, 3, disable_tp=True)
    head.tp_size = 2
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        _reduce_lm_head(head, [0, 1, 2])


@pytest.mark.cpu_test
def test_install_draft_vocab_borrows_target_head(default_vllm_config):
    """The MTP runner attaches the target head; install swaps in a reduced
    copy and drops the pending request (no double install)."""
    from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP

    model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
    torch.nn.Module.__init__(model)
    model.vocab_size = 12
    model._draft_token_id_list = [0, 2, 5]
    model.lm_head = None

    target = ParallelLMHead(12, 3, params_dtype=torch.float32, disable_tp=True)
    with torch.no_grad():
        target.weight[:12].copy_(
            torch.arange(36, dtype=torch.float32).view(12, 3) / 100
        )
    target_before = target.weight.clone()
    model.logits_processor = LogitsProcessor(12)

    model.install_draft_vocab(target)
    token_ids = torch.tensor([0, 2, 5])
    torch.testing.assert_close(model.draft_token_ids, token_ids)
    torch.testing.assert_close(model.lm_head.weight[:3], target.weight[token_ids])
    # The target's full head is untouched (verification keeps all rows).
    torch.testing.assert_close(target.weight, target_before)

    # A second call is a no-op.
    model.install_draft_vocab(target)
    torch.testing.assert_close(model.draft_token_ids, token_ids)
