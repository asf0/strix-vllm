# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM speculative verify and accepted-state commit for Qwen GDN.

Verify runs the 1 + k token window off the request's checkpoint without
writing the recurrent state and records each token's decay, normalized key and
beta-scaled correction in the checkpoint block; after sampling, the commit
replays the accepted prefix onto the checkpoint in the same operation order.

The kernels live in a kernel plugin (libstrix's strix_vllm, which registers
them on load); this module only forwards to it.
"""

from collections.abc import Sequence
from typing import Any, Protocol

import torch


class GDNRecoverSSMProvider(Protocol):
    def verify(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """`gdn_recoverssm_verify` below."""
        ...

    def create_commit_context(
        self, layers: Sequence[Any], *, spec_query_len: int, max_num_reqs: int
    ) -> Any:
        """A commit plan over GDN layers sharing a block table, with
        `commit(num_accepted_tokens, state_indices, query_start_loc,
        request_indices=None, block_table=None, num_computed_tokens=None,
        mamba_block_size=None)`."""
        ...


_provider: GDNRecoverSSMProvider | None = None


def register_gdn_recoverssm(provider: GDNRecoverSSMProvider) -> None:
    global _provider
    _provider = provider


def get_gdn_recoverssm_provider() -> GDNRecoverSSMProvider:
    if _provider is None:
        raise RuntimeError(
            "GDN RecoverSSM (--use-replayssm with speculative decoding on Qwen "
            "GDN models) needs its kernels from the libstrix plugin "
            "(strix_vllm); install it or drop --use-replayssm."
        )
    return _provider


def gdn_recoverssm_verify(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Outputs for the speculative window; fills the correction, key and
    decay records. Same arguments as the plugin's kernel."""
    return get_gdn_recoverssm_provider().verify(*args, **kwargs)


class GDNRecoverSSMCommitContext:
    """Per-group commit plan over every GDN layer that shares a block table."""

    @classmethod
    def create(
        cls, layers: Sequence[Any], *, spec_query_len: int, max_num_reqs: int
    ) -> Any:
        return get_gdn_recoverssm_provider().create_commit_context(
            layers, spec_query_len=spec_query_len, max_num_reqs=max_num_reqs
        )


__all__ = [
    "GDNRecoverSSMCommitContext",
    "GDNRecoverSSMProvider",
    "gdn_recoverssm_verify",
    "get_gdn_recoverssm_provider",
    "register_gdn_recoverssm",
]
