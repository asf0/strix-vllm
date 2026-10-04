# GatedDeltaNet RecoverSSM

RecoverSSM lets speculative decoding on Qwen GatedDeltaNet (GDN) models keep
one recurrent state per request instead of one per draft token. It is the GDN
counterpart of the Kimi-K3 KDA RecoverSSM path and is enabled with
`--use-replayssm` together with a speculative config.

## Why

A GDN layer carries a fixed-size recurrent state per request. For
Qwen3.5/3.8 that is `num_v_heads x head_v_dim x head_k_dim` float32 values
plus the convolution window, about 3.2 MiB per layer. With speculative
decoding, the spec kernel writes the state after every token of the
`1 + k` verification window into its own block, so the next step can read
the state at whatever position acceptance stopped. The scheduler therefore
reserves `num_speculative_blocks = k` extra state blocks per GDN group and
request.

On Qwen3.8-27B (48 GDN layers in 3 KV cache groups) with MTP `k = 2`, that is
9 blocks per request instead of 3, about 470 MB instead of 157 MB. At 32
running sequences these extra blocks take roughly a fifth of the KV cache pool
regardless of context length, and each verification step writes three full
states per layer instead of one.

## How it works

RecoverSSM splits the recurrent update of a verification step in two.

**Verify** (`gdn_recoverssm_verify`, during the forward pass) loads the
request's checkpoint state, runs all `1 + k` tokens in registers to produce
the attention output, and never writes the state back. For each token it
stores three small records in the checkpoint block:

| Record | Shape per block | Contents |
| --- | --- | --- |
| correction | `(num_v_heads, 1 + k, head_v_dim)` float32 | `beta * (v - (decay * S) k)` |
| key | `(num_k_heads, 1 + k, head_k_dim)` float32 | L2-normalized key |
| decay | `(num_v_heads, 1 + k)` float32 | `exp(g)` |

These records are appended to the Mamba page as extra state tensors (about
99 KB per layer on Qwen3.8, 3% of the page). They are not copied by the
prefix-caching state copy functions, which only copy the conv and recurrent
states.

**Commit** (`GDNRecoverSSMCommitContext.commit`, after sampling) runs once per
GDN KV cache group, for all layers of the group:

1. A plan kernel turns each request's sampled-token count into a commit length
   and the destination blocks.
2. The conv window is compacted so the accepted history starts at offset 0.
   The next step's conv update therefore always reads with
   `num_accepted_tokens = 1`.
3. The recurrent state is replayed: `S = fma(c_t, k_t, S * decay_t)` for each
   accepted token, then stored once.

The commit is driven by `RecoverSSMState` in the V2 model runner's
`MambaHybridModelState.postprocess_state`, the same hook Kimi-K3 uses.

### Numerics

Verify uses the same operation order as `fused_sigmoid_gating_delta_rule_update`,
and the commit reproduces its rounding: the decayed state is rounded before
the rank-1 update (an explicit `tl.fma` keeps the compiler from fusing the
decay into it). Verify outputs and committed states are bitwise identical to
the per-slot spec kernel, and greedy outputs of the full model are bitwise
identical (tokens and logprobs) when both runs use the Triton GDN path.

### Memory traffic

Per verification step and layer, the per-slot kernel reads the state once and
writes it `1 + k` times. RecoverSSM reads it once in verify, then reads and
writes it once in the commit. At `k = 2` that is three full-state passes
instead of four.

## Prefix caching (`--mamba-cache-mode align`)

In align mode the commit also places the state on the Mamba block grid:

- The running state goes to the block holding the last computed token,
  column `(num_computed + accepted - 1) // block_size`. A count that lands
  exactly on a block boundary keeps the state in the block that just filled,
  since the next block is only allocated once the next window needs it (no
  speculative blocks are reserved).
- When acceptance crosses a block boundary, the state at the boundary is also
  written to that block, so it can serve as a prefix-cache checkpoint.

`RecoverSSMState` then sets the running column to the same block and resets
`num_accepted_tokens` to 1, so the align pre-copy before the next forward moves
the state without a speculative offset.

!!! note
    The shared plan kernel (from the Kimi-K3 implementation) and the align
    postprocess kernel originally used `(num_computed + accepted) // block_size`.
    With no speculative blocks reserved, a count that landed exactly on a
    boundary pointed at an unallocated block: the commit was skipped and the
    align postprocess copied that block over the real state. Both now use the
    last computed token's block.

## Usage

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
    --use-replayssm
```

Requirements:

- Architecture `Qwen3NextForCausalLM`, `Qwen3_5ForCausalLM`,
  `Qwen3_5MoeForCausalLM`, `Qwen3_5ForConditionalGeneration` or
  `Qwen3_5MoeForConditionalGeneration`, with speculative decoding.
- CUDA or ROCm, the V2 model runner, `--mamba-cache-mode` `none` or `align`,
  pipeline parallel size 1, no KV connector and no
  `--enable-mamba-cache-stochastic-rounding`.

The records make the Mamba page about 3% larger, which can change the hybrid
attention block size (it is rounded up to the kernel block alignment). On
gfx1151 with `int8_per_token_head` KV cache, pass `--block-size 32`: Qwen3.8
then gets 1632-token blocks instead of 1616, which keeps the Triton prefill
KV-piece path (it needs the block size to be a multiple of its 32-token tile).

On ROCm, batches of up to 8 speculative requests normally use the fused HIP
MTP decode kernel. RecoverSSM always uses the Triton verify kernel, so these
small batches lose that kernel.

## Results

Qwen3.8-27B BF16, `int8_per_token_head` KV cache, MTP `k = 2`, 32 sequences of
4K-token real text generating 1024 tokens each, gfx1151 (Strix Halo), block
size 1632 for both runs:

| | Per-draft-slot states | RecoverSSM |
| --- | --- | --- |
| GDN state blocks per request | 9 | 3 |
| KV cache usage while decoding | 46-50% | 22-26% |
| Decode throughput | 177.2 tok/s | 183.7 tok/s |
| Mean acceptance length | 2.557 | 2.564 |

## Code and tests

- Kernels: `vllm/model_executor/layers/mamba/ops/gdn_recoverssm.py`.
- Metadata: `GDNRecoverSSMAttentionMetadata` in
  `vllm/v1/attention/backends/gdn_attn.py`.
- Layer routing: `QwenGatedDeltaNetAttention._forward_core`.
- Configuration: `CacheConfig.use_gdn_recoverssm`, set by
  `VllmConfig.validate_mamba_cached_kernel`.
- Tests: `tests/kernels/mamba/test_gdn_recoverssm.py` (verify and commit
  against the per-slot kernel in none and align mode, including a boundary
  landing; two MTP steps through the real builder and `_forward_core`) and
  `tests/v1/attention/test_gdn_metadata_builder.py`.
