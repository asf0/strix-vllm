# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton MXFP8 batched expert GEMMs (`tl.dot_scaled`: gfx950's scaled MFMA)
for the Inkling2 MoE.

The GEMMs read expert-sorted activation rows whose expert segments are padded
to the token tile, with block scales in the R8c4 (tile < 128) or R128c4 layout
(`mxfp8`), and the expert weights and their block scales as stored in the
checkpoint (row-major). Outputs are bf16.
"""

import functools
from dataclasses import dataclass

import torch

from vllm.platforms.rocm import on_gfx1x
from vllm.triton_utils import tl, triton

from .mxfp8 import _scale_offsets

# Token tile by batch size (tokens up to the first entry); the smallest is one
# 16-row MFMA tile.
_TILES = ((128, 16), (384, 32), (768, 64), (4096, 128), (1 << 31, 256))


@dataclass(frozen=True)
class ExpertPlan:
    """Token tile and launched tiles (an upper bound) for a batch."""

    tile: int
    max_ctas: int

    @property
    def rows(self) -> int:
        return self.tile * self.max_ctas


@functools.cache
def expert_plan(
    tokens: int, topk: int, num_experts: int, tile: int | None = None
) -> ExpertPlan:
    """Plan for `tokens` routed to `topk` of all experts, `num_experts`
    of them local."""
    if tile is None:
        tile = next(t for max_tokens, t in _TILES if tokens <= max_tokens)
    # Tiles if the local rows fill as many experts as possible, then pile up.
    rows = tokens * topk
    max_ctas = min(num_experts, rows) + (rows - min(num_experts, rows)) // tile
    return ExpertPlan(tile, max_ctas)


@triton.jit
def _mxfp8_bmm_kernel(
    A,
    AScale,
    W,
    WScale,
    Out,
    Tiles,
    max_ctas,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Program (n, m) computes BM rows of token tile m // SUB: a part of it, or
    # (for tiles below BM rows) the whole tile. Only routed rows are read and
    # written: past them are the next tile's.
    SUB: tl.constexpr = TILE // BM if TILE >= BM else 1
    pid_m = tl.program_id(1)
    tile = pid_m // SUB
    if tile >= tl.load(Tiles + 2 * max_ctas):
        return
    row_end = tl.load(Tiles + max_ctas + tile)
    row0 = tile * TILE + (pid_m % SUB) * BM
    if row0 >= row_end:
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    rows = row0 + tl.arange(0, BM)
    row_mask = (rows < row_end)[:, None]
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    groups = tl.arange(0, BK // 32)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + ks[None, :]
    w_ptrs = W + (expert * N + cols[:, None]) * K + ks[None, :]
    ws_ptrs = WScale + (expert * N + cols[:, None]) * (K // 32) + groups[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        scale_cols = k // 32 + groups[None, :]
        a_scale_ptrs = AScale + _scale_offsets(rows[:, None], scale_cols, K // 32, TILE)
        a = tl.load(a_ptrs + k, row_mask, other=0.0)
        a_scale = tl.load(a_scale_ptrs, row_mask, other=0)
        w = tl.load(w_ptrs + k)
        w_scale = tl.load(ws_ptrs + k // 32)
        acc = tl.dot_scaled(a, a_scale, "e4m3", tl.trans(w), w_scale, "e4m3", acc)
    out_ptrs = Out + rows[:, None].to(tl.int64) * N + cols[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), row_mask)


@triton.jit
def _e2m1_to_f32(code):
    """fp32 value of each 4-bit e2m1 `code` (sign in bit 3)."""
    mag = code & 7
    exponent = (mag >> 1).to(tl.float32)
    normal = (1.0 + 0.5 * (mag & 1).to(tl.float32)) * tl.exp2(exponent - 1.0)
    value = tl.where(mag < 2, 0.5 * mag.to(tl.float32), normal)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _nvfp4_bmm_kernel(
    A,
    AScale,
    W,
    WScale,
    WAmax,
    WDenominator,
    Out,
    Tiles,
    max_ctas,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # As `_mxfp8_bmm_kernel`, with NVFP4 weights decoded to bf16 in the
    # kernel: two e2m1 values per byte (the first in the low nibble), an e4m3
    # scale per 16 values and an fp32 outer scale `amax / denominator` per
    # expert.
    SUB: tl.constexpr = TILE // BM if TILE >= BM else 1
    pid_m = tl.program_id(1)
    tile = pid_m // SUB
    if tile >= tl.load(Tiles + 2 * max_ctas):
        return
    row_end = tl.load(Tiles + max_ctas + tile)
    row0 = tile * TILE + (pid_m % SUB) * BM
    if row0 >= row_end:
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    outer = tl.load(WAmax + expert) / tl.load(WDenominator)
    rows = row0 + tl.arange(0, BM)
    row_mask = (rows < row_end)[:, None]
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    groups = tl.arange(0, BK // 32)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + ks[None, :]
    weight_rows = (expert * N + cols[:, None]).to(tl.int64)
    w_ptrs = W + weight_rows * (K // 2) + tl.arange(0, BK // 2)[None, :]
    ws_ptrs = WScale + weight_rows * (K // 16) + tl.arange(0, BK // 16)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        scale_cols = k // 32 + groups[None, :]
        a_scale_ptrs = AScale + _scale_offsets(rows[:, None], scale_cols, K // 32, TILE)
        a = tl.load(a_ptrs + k, row_mask, other=0.0)
        a_scale = tl.load(a_scale_ptrs, row_mask, other=0)
        packed = tl.load(w_ptrs + k // 2)
        low = _e2m1_to_f32(packed & 15)
        high = _e2m1_to_f32(packed >> 4)
        values = tl.join(low, high).reshape(BN, BK // 16, 16)
        block_scale = tl.load(ws_ptrs + k // 16).to(tl.float32) * outer
        w = (values * block_scale[:, :, None]).reshape(BN, BK).to(tl.bfloat16)
        acc = tl.dot_scaled(a, a_scale, "e4m3", tl.trans(w), None, "bf16", acc)
    out_ptrs = Out + rows[:, None].to(tl.int64) * N + cols[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), row_mask)


def bmm_nvfp4(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_amax: torch.Tensor,
    weight_denominator: torch.Tensor,
    tiles: torch.Tensor,
    plan: ExpertPlan,
) -> torch.Tensor:
    """`bmm` with NVFP4 expert weights in the checkpoint layout: `weight` is
    `[E, N, K // 2]` uint8, `weight_scale` `[E, N, K // 16]` e4m3, and the
    outer scale of expert `e` is `weight_amax[e] / weight_denominator`."""
    num_experts, n, k_half = weight.shape
    k = 2 * k_half
    assert act.shape == (plan.rows, k) and weight.dtype == torch.uint8
    assert weight_scale.shape == (num_experts, n, k // 16)
    assert weight_scale.dtype == torch.float8_e4m3fn
    assert weight_amax.shape == (num_experts,) and weight_denominator.numel() == 1
    assert weight.is_contiguous() and weight_scale.is_contiguous()
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    bm = min(max(tile, 16), 128)
    bn = 128 if n % 128 == 0 else 64
    bk = 128
    assert n % bn == 0 and k % bk == 0
    sub = tile // bm if tile >= bm else 1
    _nvfp4_bmm_kernel[(n // bn, plan.max_ctas * sub)](
        act,
        act_scale,
        weight,
        weight_scale,
        weight_amax,
        weight_denominator,
        out,
        tiles,
        plan.max_ctas,
        N=n,
        K=k,
        TILE=tile,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=4,
        num_stages=2,
    )
    return out


def bmm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    tiles: torch.Tensor,
    plan: ExpertPlan,
) -> torch.Tensor:
    """bf16 `[plan.rows, N]` expert GEMM of the tile-padded MXFP8 `act` rows
    with their swizzled block scales `act_scale`, by each tile's expert
    (`tiles`: the `moe.route` metadata). Rows past each tile's routed rows
    are left uninitialized."""
    num_experts, n, k = weight.shape
    assert act.shape == (plan.rows, k) and weight.dtype == torch.float8_e4m3fn
    assert weight_scale.shape == (num_experts, n, k // 32)
    assert weight.is_contiguous() and weight_scale.is_contiguous()
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    bm = min(max(tile, 16), 128)
    bn = 128 if n % 128 == 0 else 64
    # RDNA's 64 KB of LDS: 128-wide K tiles, one stage for the largest tiles.
    small_lds = on_gfx1x()
    bk = 256 if k % 256 == 0 and bm >= 64 and not small_lds else 128
    num_stages = 1 if small_lds and bm == 128 else 2
    assert n % bn == 0 and k % bk == 0
    sub = tile // bm if tile >= bm else 1
    _mxfp8_bmm_kernel[(n // bn, plan.max_ctas * sub)](
        act,
        act_scale,
        weight,
        weight_scale,
        out,
        tiles,
        plan.max_ctas,
        N=n,
        K=k,
        TILE=tile,
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=8 if bm == 128 and bk == 256 else 4,
        num_stages=num_stages,
    )
    return out
