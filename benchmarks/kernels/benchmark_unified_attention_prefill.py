# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep unified_attention prefill launch configs at production shapes.

Defaults mirror the Qwen3.8-27B teacher on gfx1151: one sequence, an
8192-token prefill chunk after KV depth D, 24 query / 4 KV heads, head_dim
256, int8_per_token_head KV (--kv-cache-dtype) in the LBNHC layout with
1552-token blocks.

The cache is built with the Triton backend's own packed-view helpers and
written by the production quantizing writer. Launch configs are swapped by
wrapping ``kernel_unified_attention``, so everything else is the production
wrapper.

    # Plan only; touches no GPU.
    python benchmarks/kernels/benchmark_unified_attention_prefill.py --dry-run
    # Every config against an fp32 reference on a small odd shape.
    python benchmarks/kernels/benchmark_unified_attention_prefill.py --check
    # Sweep all configs at the first depth, the best --top-k deeper.
    python benchmarks/kernels/benchmark_unified_attention_prefill.py \\
        --output attn_sweep.json
"""

import argparse
import itertools
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LaunchConfig:
    block_m: int
    tile: int
    warps: int
    stages: int

    @property
    def name(self) -> str:
        return f"M{self.block_m}_T{self.tile}_w{self.warps}_s{self.stages}"


DEFAULT = None  # Production launch parameters, no override.


def causal_flops(args, depth: int, chunk: int) -> float:
    keys_seen = chunk * depth + chunk * (chunk + 1) / 2
    return 4 * args.num_q_heads * args.head_size * keys_seen


def parse_configs(args) -> list[LaunchConfig]:
    if args.configs:
        configs = []
        for spec in args.configs:
            block_m, tile, warps, stages = (int(x) for x in spec.split(":"))
            configs.append(LaunchConfig(block_m, tile, warps, stages))
        return configs
    return [
        LaunchConfig(*values)
        for values in itertools.product(
            args.block_m, args.tile, args.warps, args.stages
        )
    ]


def cache_bytes(args, num_tokens: int) -> int:
    per_half = {"auto": 2 * args.head_size, "fp8": args.head_size}.get(
        args.kv_cache_dtype, args.head_size + 4
    )
    return num_tokens * args.num_kv_heads * 2 * per_half


def print_plan(args, configs: list[LaunchConfig]) -> None:
    group = args.num_q_heads // args.num_kv_heads
    tokens = max(args.depths) + args.chunk
    print(f"GQA group {group}; configs ({len(configs)} + production default):")
    for config in configs:
        block_q = config.block_m // group
        rows = f"{block_q * group}/{config.block_m} rows"
        print(f"  {config.name:18s} BLOCK_Q={block_q:<3d} {rows}")
    print(f"KV cache for {tokens} tokens: {cache_bytes(args, tokens) / 2**20:.0f} MiB")
    for depth in args.depths:
        tflop = causal_flops(args, depth, args.chunk) / 1e12
        print(f"  depth {depth:>6d}: {tflop:6.2f} TFLOP per call")


class LaunchOverride:
    """Stands in for ``kernel_unified_attention`` with one launch config."""

    def __init__(self, kernel, config: LaunchConfig):
        self.kernel = kernel
        self.config = config

    def __getitem__(self, grid):
        def launch(**kwargs):
            assert not kwargs["IS_3D"], "prefill must take the 2D path"
            block_q = self.config.block_m // kwargs["num_queries_per_kv"]
            assert block_q > 0, f"BLOCK_M {self.config.block_m} < GQA group"
            num_q_blocks = kwargs["query_ptr"].shape[0] // block_q
            kwargs.update(
                BLOCK_M=self.config.block_m,
                BLOCK_Q=block_q,
                TILE_SIZE=self.config.tile,
                num_warps=self.config.warps,
                num_stages=self.config.stages,
            )
            new_grid = (num_q_blocks + kwargs["num_seqs"], grid[1])
            return self.kernel[new_grid](**kwargs)

        return launch


class Harness:
    def __init__(self, args):
        import torch

        from vllm.platforms import current_platform
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        from vllm.v1.attention.ops import triton_unified_attention as tua
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
            triton_reshape_and_cache_flash_per_token_head_quant,
        )
        from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

        self.torch = torch
        self.tua = tua
        self.kernel = tua.kernel_unified_attention
        self.quant_mode = get_kv_quant_mode(args.kv_cache_dtype)
        per_token_head = self.quant_mode.is_per_token_head
        fp8_dtype = current_platform.fp8_dtype()
        self.args = args
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        bs, nkv, hs = args.block_size, args.num_kv_heads, args.head_size
        max_tokens = max(args.depths) + args.chunk

        # Block 0 is the null block; the rest are handed out shuffled, as
        # after the free list has churned.
        num_blocks = math.ceil(max_tokens / bs) + 1
        order = torch.randperm(num_blocks - 1, generator=generator, device=device)
        self.block_table = (order + 1).to(torch.int32).unsqueeze(0)
        if args.contiguous_blocks:
            self.block_table = torch.arange(
                1, num_blocks, dtype=torch.int32, device=device
            ).unsqueeze(0)

        # LBNHC: one layer's region is (block, token, head, content). The
        # backend receives the logical (block, head, token, content) view.
        # Per-token-head modes pad each K/V half with one inline fp32 scale.
        storage_dtype, content = {
            "int8_per_token_head": (torch.int8, 2 * (hs + 4)),
            "fp8_per_token_head": (torch.uint8, 2 * (hs + 4)),
            "auto": (torch.bfloat16, 2 * hs),
            "fp8": (torch.uint8, 2 * hs),
        }[args.kv_cache_dtype]
        physical = torch.zeros(
            num_blocks, bs, nkv, content, dtype=storage_dtype, device=device
        )
        logical = physical.permute(0, 2, 1, 3)
        # Per-tensor FP8 scale sized so N(0, 1) data spans the FP8 range.
        self.kv_scale = torch.tensor(
            [0.02 if self.quant_mode == KVQuantMode.FP8_PER_TENSOR else 1.0],
            dtype=torch.float32,
            device=device,
        )
        if per_token_head:
            impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
            impl._kv_quant_mode = self.quant_mode
            impl.fp8_dtype = fp8_dtype
            self.key_cache, self.value_cache = impl._pth_key_value_caches(logical)
            self.k_scale_cache = impl._k_scale_cache
            self.v_scale_cache = impl._v_scale_cache
        else:
            self.key_cache, self.value_cache = logical.transpose(1, 2).split(hs, dim=-1)
            if self.quant_mode == KVQuantMode.FP8_PER_TENSOR:
                self.key_cache = self.key_cache.view(fp8_dtype)
                self.value_cache = self.value_cache.view(fp8_dtype)
            self.k_scale_cache = self.v_scale_cache = None

        positions = torch.arange(max_tokens, device=device)
        slots = (
            self.block_table[0, positions // bs].long() * bs + positions % bs
        ).contiguous()
        for start in range(0, max_tokens, 16384):
            end = min(start + 16384, max_tokens)
            shape = (end - start, nkv, hs)
            key = torch.randn(shape, generator=generator, device=device)
            value = torch.randn(shape, generator=generator, device=device)
            key, value = key.to(torch.bfloat16), value.to(torch.bfloat16)
            if per_token_head:
                triton_reshape_and_cache_flash_per_token_head_quant(
                    key,
                    value,
                    self.key_cache,
                    self.value_cache,
                    self.k_scale_cache,
                    self.v_scale_cache,
                    slots[start:end],
                    kv_quant_mode=self.quant_mode,
                )
            else:
                triton_reshape_and_cache_flash(
                    key,
                    value,
                    self.key_cache,
                    self.value_cache,
                    slots[start:end],
                    args.kv_cache_dtype,
                    self.kv_scale,
                    self.kv_scale,
                )
        self.query = torch.randn(
            args.chunk,
            args.num_q_heads,
            hs,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        torch.accelerator.synchronize()

    def run(self, config: LaunchConfig | None, depth: int, chunk: int):
        torch = self.torch
        device = self.query.device
        query = self.query[:chunk]
        out = torch.empty_like(query)
        seq_len = depth + chunk
        # Non-per-token-head modes get the layer scale per (seq, kv head).
        descale = None
        if self.k_scale_cache is None:
            descale = self.kv_scale.expand((1, self.args.num_kv_heads))
        self.tua.kernel_unified_attention = (
            self.kernel if config is None else LaunchOverride(self.kernel, config)
        )
        try:
            self.tua.unified_attention(
                q=query,
                k=self.key_cache,
                v=self.value_cache,
                out=out,
                cu_seqlens_q=torch.tensor([0, chunk], dtype=torch.int32, device=device),
                max_seqlen_q=chunk,
                seqused_k=torch.tensor([seq_len], dtype=torch.int32, device=device),
                max_seqlen_k=seq_len,
                softmax_scale=self.args.head_size**-0.5,
                causal=True,
                window_size=(-1, -1),
                block_table=self.block_table,
                softcap=0,
                q_descale=None,
                k_descale=descale,
                v_descale=descale,
                kv_quant_mode=self.quant_mode,
                k_scale_cache=self.k_scale_cache,
                v_scale_cache=self.v_scale_cache,
            )
        finally:
            self.tua.kernel_unified_attention = self.kernel
        return out

    def time_ms(self, config: LaunchConfig | None, depth: int) -> tuple:
        torch = self.torch
        out = self.run(config, depth, self.args.chunk)  # compile + warm up
        torch.accelerator.synchronize()
        times = []
        for _ in range(self.args.reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.run(config, depth, self.args.chunk)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
            if times[-1] > self.args.slow_call_ms:
                break
        return sorted(times)[len(times) // 2], out

    def reference(self, depth: int, chunk: int):
        """fp32 causal GQA attention over the dequantized cache."""
        torch = self.torch
        args = self.args
        bs, nkv, hs = args.block_size, args.num_kv_heads, args.head_size
        group = args.num_q_heads // nkv
        seq_len = depth + chunk
        positions = torch.arange(seq_len, device=self.query.device)
        blocks = self.block_table[0, positions // bs].long()
        slots = positions % bs

        def gather(cache):
            # Index FP8 through its bytes; FP8 gather kernels may be missing.
            if cache.dtype.is_floating_point and cache.element_size() == 1:
                return cache.view(torch.uint8)[blocks, slots, :, :hs].view(cache.dtype)
            return cache[blocks, slots, :, :hs]

        key = gather(self.key_cache).float() * self.kv_scale
        value = gather(self.value_cache).float() * self.kv_scale
        if self.k_scale_cache is not None:
            key *= self.k_scale_cache[blocks, slots, :, None]
            value *= self.v_scale_cache[blocks, slots, :, None]
        query = self.query[:chunk].float().view(chunk, nkv, group, hs)
        scores = torch.einsum("tkgd,skd->kgts", query, key) * hs**-0.5
        query_pos = depth + torch.arange(chunk, device=positions.device)
        future = positions[None, :] > query_pos[:, None]
        scores.masked_fill_(future, float("-inf"))
        out = torch.einsum("kgts,skd->tkgd", scores.softmax(-1), value)
        return out.reshape(chunk, args.num_q_heads, hs)


def compare(torch, out, ref) -> dict:
    out, ref = out.float().flatten(), ref.float().flatten()
    return {
        "max_abs": (out - ref).abs().max().item(),
        "rel_max": ((out - ref).abs().max() / ref.abs().max()).item(),
        "cosine": torch.nn.functional.cosine_similarity(out, ref, dim=0).item(),
    }


def check(harness: Harness, configs: list[LaunchConfig], args) -> bool:
    torch = harness.torch
    ref = harness.reference(args.check_depth, args.check_chunk)
    ok = True
    for config in [DEFAULT, *configs]:
        name = "default" if config is None else config.name
        try:
            out = harness.run(config, args.check_depth, args.check_chunk)
            torch.accelerator.synchronize()
        except Exception as e:  # compile or resource failures per config
            print(f"{name:18s} SKIP {type(e).__name__}: {str(e)[:120]}")
            continue
        stats = compare(torch, out, ref)
        passed = stats["cosine"] > 0.9999 and stats["rel_max"] < 0.02
        ok &= passed
        print(
            f"{name:18s} {'ok  ' if passed else 'FAIL'} cos={stats['cosine']:.6f} "
            f"rel_max={stats['rel_max']:.4f} max_abs={stats['max_abs']:.4g}"
        )
    return ok


def sweep(harness: Harness, configs: list[LaunchConfig], args) -> list[dict]:
    torch = harness.torch
    results: list[dict] = []
    candidates = [DEFAULT, *configs]
    for depth_idx, depth in enumerate(args.depths):
        if depth_idx == 1 and not args.full:
            timed = sorted(
                (r for r in results if r["ms"] is not None and r["config"]),
                key=lambda r: r["ms"],
            )
            keep = {r["config"]["name"] for r in timed[: args.top_k]}
            candidates = [DEFAULT] + [c for c in configs if c.name in keep]
        flops = causal_flops(args, depth, args.chunk)
        baseline_ms = baseline_out = None
        for config in candidates:
            name = "default" if config is None else config.name
            row = {
                "depth": depth,
                "chunk": args.chunk,
                "config": None if config is None else {"name": name, **asdict(config)},
                "ms": None,
            }
            try:
                started = time.monotonic()
                ms, out = harness.time_ms(config, depth)
                row.update(ms=ms, tflops=flops / ms / 1e9)
                row["wall_s"] = round(time.monotonic() - started, 1)
                if config is None:
                    baseline_ms, baseline_out = ms, out
                else:
                    row["speedup"] = baseline_ms / ms
                    row.update(compare(torch, out, baseline_out))
                del out
            except Exception as e:  # compile or resource failures per config
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            results.append(row)
            print(format_row(row), flush=True)
            if args.output:
                write_results(args, results)
    return results


def format_row(row: dict) -> str:
    name = row["config"]["name"] if row["config"] else "default"
    head = f"depth {row['depth']:>6d} {name:18s}"
    if row["ms"] is None:
        return f"{head} {row['error']}"
    text = f"{head} {row['ms']:10.1f} ms {row['tflops']:6.2f} TFLOPS"
    if "speedup" in row:
        text += f" x{row['speedup']:5.2f} vs default, cos={row['cosine']:.6f}"
    return text


def write_results(args, results: list[dict]) -> None:
    import torch
    import triton

    root = Path(__file__).resolve().parents[2]
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "shape": {
            k: getattr(args, k)
            for k in (
                "chunk",
                "num_q_heads",
                "num_kv_heads",
                "head_size",
                "block_size",
                "kv_cache_dtype",
            )
        },
        "env": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "vllm_sha": sha,
        },
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[8192, 32768, 65536, 131072]
    )
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--num-q-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=1552)
    parser.add_argument(
        "--kv-cache-dtype",
        default="int8_per_token_head",
        choices=["int8_per_token_head", "fp8_per_token_head", "auto", "fp8"],
    )
    parser.add_argument("--block-m", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--tile", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--configs",
        nargs="+",
        help="explicit BLOCK_M:TILE:WARPS:STAGES list instead of the grid",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="configs carried past the first depth (with the default)",
    )
    parser.add_argument(
        "--full", action="store_true", help="every config at every depth"
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--slow-call-ms",
        type=float,
        default=15000,
        help="stop repeating a config once one call exceeds this",
    )
    parser.add_argument("--contiguous-blocks", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-depth", type=int, default=3001)
    parser.add_argument("--check-chunk", type=int, default=701)
    parser.add_argument("--output", help="JSON results path, rewritten per row")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    assert args.num_q_heads % args.num_kv_heads == 0

    configs = parse_configs(args)
    if args.check:
        args.depths, args.chunk = [args.check_depth], args.check_chunk
    print_plan(args, configs)
    if args.dry_run:
        return
    if args.check:
        raise SystemExit(0 if check(Harness(args), configs, args) else 1)
    sweep(Harness(args), configs, args)


if __name__ == "__main__":
    main()
