# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEDiskEmbedding
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        self._disk_ple: list[tuple[nn.Module, Qwen4ExpPLEDiskEmbedding]] = []
        self._ple_hash_params: list[tuple[np.ndarray, ...]] | None = None
        self._host_prefill_tokens: dict[str, np.ndarray] = {}
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        # Disk-offloaded n-gram table: hash pure-prefill steps on the host and
        # start their row gather here, before the GPU reaches the PLE layer.
        self._disk_ple = [
            (module, module.ngram_embedding)
            for module in model.modules()
            if isinstance(
                getattr(module, "ngram_embedding", None), Qwen4ExpPLEDiskEmbedding
            )
        ]
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        context = self.ngram_context[:num_reqs_padded]
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self._disk_ple and new_req_data.prefill_token_ids is not None:
            self._host_prefill_tokens[new_req_data.req_id] = np.asarray(
                new_req_data.prefill_token_ids, dtype=np.int64
            )

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._host_prefill_tokens.pop(req_id, None)

    def _host_ngram_ids(
        self,
        tokens: np.ndarray,
        multipliers: np.ndarray,
        sizes: np.ndarray,
        offsets: np.ndarray,
        heads_per_ngram: int,
    ) -> np.ndarray:
        """Host copy of ple_ngram_ids for one request: ``tokens`` is the
        n-gram context (EOS-padded) followed by the scheduled tokens."""
        ctx = self.ngram_context_len
        eos = self.ngram_eos_token_id
        n = tokens.size - ctx
        order = np.arange(ctx * heads_per_ngram) // heads_per_ngram + 2
        # int64 products wrap like the GPU kernel's.
        mixed = np.repeat((tokens[ctx:] * multipliers[0])[:, None], order.size, axis=1)
        crossed = np.zeros(n, dtype=bool)
        for shift in range(1, ctx + 1):
            candidate = np.where(crossed, eos, tokens[ctx - shift : ctx - shift + n])
            crossed |= candidate == eos
            term = candidate * multipliers[shift]
            mixed ^= np.where(order[None, :] > shift, term[:, None], 0)
        # numpy's % on int64 already follows torch.remainder for positive sizes.
        return mixed % sizes[None, :] + offsets[None, :]

    def _prefetch_disk_ple(self, input_batch: InputBatch) -> None:
        num_reqs = input_batch.num_reqs
        if not num_reqs or input_batch.num_draft_tokens:
            return
        ctx = self.ngram_context_len
        starts = input_batch.num_computed_prefill_tokens_np[:num_reqs]
        lens = input_batch.num_scheduled_tokens[:num_reqs]
        spans = []
        for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
            tokens = self._host_prefill_tokens.get(req_id)
            start, length = int(starts[i]), int(lens[i])
            # Only steps whose every token is a known prefill token.
            if tokens is None or start + length > tokens.size:
                return
            context = tokens[max(0, start - ctx) : start]
            pad = np.full(ctx - context.size, self.ngram_eos_token_id, dtype=np.int64)
            spans.append(np.concatenate([pad, context, tokens[start : start + length]]))
        if self._ple_hash_params is None:
            self._ple_hash_params = [
                (
                    owner.layer_multipliers.cpu().numpy(),
                    owner.ngram_heads_vocab_sizes.cpu().numpy(),
                    owner.ngram_heads_offsets.cpu().numpy(),
                )
                for owner, _ in self._disk_ple
            ]
        for (owner, embedding), params in zip(self._disk_ple, self._ple_hash_params):
            ids = np.concatenate(
                [
                    self._host_ngram_ids(span, *params, owner.heads_per_ngram)
                    for span in spans
                ]
            )
            embedding.prefetch_host_ids(ids)

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs
        if self._disk_ple:
            self._prefetch_disk_ple(input_batch)

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc[: num_reqs + 1]
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])

        ngram_context = self.ngram_context[:num_reqs]
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
