# benchmark/benchmark_newton_schulz.py
"""
Newton-Schulz kernel benchmarks.

Compares seven orthogonalization implementations:
  1. zeropower_via_newtonschulz5  (pure PyTorch reference)
  2. newton_schulz_triton         (Triton kernels)
  3. polar_express                 (pure PyTorch)
  4. polar_express_triton          (Triton kernels)
  5. GramNewtonSchulz(ns_use_kernels=False, use_gram_newton_schulz=False)
  6. GramNewtonSchulz(ns_use_kernels=True,  use_gram_newton_schulz=False)
  7. GramNewtonSchulz(ns_use_kernels=True,  use_gram_newton_schulz=True, reset=[2])

Examples
--------
# One-off timing (1024 x 1024, batch=1 & 4)
python benchmark/benchmark_newton_schulz.py single --m 1024
python benchmark/benchmark_newton_schulz.py single --m 1024 --n 1024 --batch_size 4

# Grid sweep (default starts at 512, doubling each step)
python benchmark/benchmark_newton_schulz.py grid --batch_size 4 --expansion 1

# Grid sweep starting at a different dimension
python benchmark/benchmark_newton_schulz.py grid --start_dim 256 --batch_size 4

# TFLOPS plot (writes PNG & PDF in ./plots)
python benchmark/benchmark_newton_schulz.py plot --batch_size 1
"""
import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple
import torch
from tqdm import tqdm

try:
    import triton.testing as tt
except ImportError:
    raise ImportError(
        "Benchmarks require the 'triton' package. "
        "Install it with: pip install dion[triton]  (or: pip install triton)"
    )

from newton_schulz_triton import (
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)
from polar_express import polar_express, polar_express_triton

from gram_newton_schulz import GramNewtonSchulz

# -----------------------------------------------------------------------------
# Provider registry
# -----------------------------------------------------------------------------

# Shared compile_kwargs for GramNewtonSchulz instances
_GNS_COMPILE_KWARGS = dict(fullgraph=True, mode="default")


PROVIDERS: OrderedDict[str, Callable] = OrderedDict({
    "zeropower_via_newtonschulz5": zeropower_via_newtonschulz5,
    "polar_express": polar_express,
    "GNS(kernels=F,gram=F)": GramNewtonSchulz(
        ns_use_kernels=False,
        use_gram_newton_schulz=False,
        compile_kwargs=_GNS_COMPILE_KWARGS,
    ),
    "GNS(kernels=F,gram=T)": GramNewtonSchulz(
        ns_use_kernels=False,
        use_gram_newton_schulz=True,
        compile_kwargs=_GNS_COMPILE_KWARGS,
    ),
    "newton_schulz_triton": newton_schulz_triton,
    "polar_express_triton": polar_express_triton,
    "GNS(kernels=T,gram=F)": GramNewtonSchulz(
        ns_use_kernels=True,
        use_gram_newton_schulz=False,
        compile_kwargs=_GNS_COMPILE_KWARGS,
    ),
    "GNS(kernels=T,gram=T)": GramNewtonSchulz(
        ns_use_kernels=True,
        use_gram_newton_schulz=True,
        gram_newton_schulz_reset_iterations=[2],
        compile_kwargs=_GNS_COMPILE_KWARGS,
    ),
})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def gemm_cost(m: int, n: int) -> int:
    """
    Return the FLOP count of the three GEMMs done per Newton-Schulz iteration.
    Derivation: see paper / original comment.
    """
    return 4 * m * m * n + 2 * m * m * m  # == 4 m²n + 2 m³


def tflops(ms: float, flops: int, steps: int, batch: int) -> float:
    return batch * steps * flops * 1e-12 / (ms * 1e-3)


def pretty_time(ms: float) -> str:
    return f"{ms:7.3f} ms"


def bench_once(
    m: int,
    n: int,
    providers: OrderedDict[str, Callable],
    *,
    batch_size: int = 1,
    steps: int = 5,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Time all providers and return a dict of name -> runtime (ms)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this benchmark")

    G = torch.randn(batch_size, m, n, dtype=dtype, device="cuda")
    flops = gemm_cost(m, n)
    results: Dict[str, float] = {}

    for name, fn in tqdm(providers.items(), desc="Benchmarking", leave=False):
        fn(G)  # warmup
        ms = tt.do_bench(lambda fn=fn: fn(G))
        tf = tflops(ms, flops, steps, batch_size)
        results[name] = ms

    fastest_ms = min(results.values())
    print(f"{'=' * 60}")
    print(f"m = {m}, n = {n},\tbatch = {batch_size}")
    print(f"{'=' * 60}")
    for name, ms in results.items():
        tf = tflops(ms, flops, steps, batch_size)
        print(f"  {name:40s}  {ms / fastest_ms:5.2f}x  {pretty_time(ms)}  {tf:5.2f} TFLOPS")
    print()
    return results


def bench_grid(
    dims: Iterable[int],
    providers: OrderedDict[str, Callable],
    *,
    expansion: int = 1,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
):
    """Sweep over square/rectangular sizes."""
    for d in dims:
        bench_once(
            d,
            d * expansion,
            providers,
            batch_size=batch_size,
            dtype=dtype,
        )


def bench_plot(
    providers: OrderedDict[str, Callable],
    batch_size: int,
    *,
    out_dir: Path = Path("plots"),
):
    """Generate TFLOPS vs. size curves using Triton's perf_report helper."""
    provider_names = list(providers.keys())

    @tt.perf_report(
        tt.Benchmark(
            x_names=["dim"],
            x_vals=[128 * i for i in range(1, 8)],
            line_arg="provider",
            line_vals=provider_names,
            line_names=provider_names,
            ylabel="TFLOPS",
            plot_name=f"newton_schulz_batch{batch_size}",
            args={"batch_size": batch_size},
        )
    )
    def bench(dim: int, provider: str, batch_size: int):
        G = torch.randn(batch_size, dim, dim, dtype=torch.bfloat16, device="cuda")
        fn = providers[provider]
        fn(G)  # warmup
        ms = tt.do_bench(lambda: fn(G))
        return tflops(ms, gemm_cost(dim, dim), steps=5, batch=batch_size)

    bench.run(print_data=True, save_path=str(out_dir))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmarks for Newton-Schulz orthogonalization variants"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # -- single --
    sp_single = sub.add_parser("single", help="run a single benchmark")
    sp_single.add_argument("--m", type=int, required=True, help="rows")
    sp_single.add_argument("--n", type=int, help="cols (defaults to m)")
    sp_single.add_argument("--batch_size", type=int, default=1)
    sp_single.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16"],
        help="input dtype",
    )

    # -- grid --
    sp_grid = sub.add_parser("grid", help="sweep a list of sizes")
    sp_grid.add_argument("--batch_size", type=int, default=1)
    sp_grid.add_argument(
        "--start_dim", type=int, default=512, help="smallest dimension (doubled each step)",
    )
    sp_grid.add_argument(
        "--expansion", type=int, default=1, help="n = m * expansion",
    )
    sp_grid.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16"],
        help="input dtype",
    )

    # -- plot --
    sp_plot = sub.add_parser("plot", help="generate TFLOPS curves and write plots")
    sp_plot.add_argument("--batch_size", type=int, default=1)

    return p.parse_args()


def main():
    args = parse()
    torch._dynamo.config.cache_size_limit = 100  # noqa: SLF001

    if args.mode == "grid":
        dtype = getattr(torch, args.dtype)
        dims = [args.start_dim * (2 ** i) for i in range(5)]
        bench_grid(
            dims,
            PROVIDERS,
            expansion=args.expansion,
            batch_size=args.batch_size,
            dtype=dtype,
        )
    elif args.mode == "plot":
        bench_plot(PROVIDERS, args.batch_size)
    else:  # single
        dtype = getattr(torch, args.dtype)
        m = args.m
        n = args.n or m
        bench_once(m, n, PROVIDERS, batch_size=args.batch_size, dtype=dtype)


if __name__ == "__main__":
    main()
