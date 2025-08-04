# benchmarks/bench_newton_schulz.py
"""
Newton-Schulz kernel benchmarks.

Examples
--------
# One-off timing (1024 x 1024, batch=1 & 4)
python -m benchmarks.bench_newton_schulz --m 1024 --n 1024
python -m benchmarks.bench_newton_schulz --m 1024 --n 1024 --batch_size 4

# Grid sweep like the original 'benchmark_many_sizes'
python -m benchmarks.bench_newton_schulz --grid --batch_size 4 --expansion 1

# TFLOPS plot (writes PNG & PDF in ./plots)
python -m benchmarks.bench_newton_schulz --plot --batch_size 1
"""
import argparse
from pathlib import Path
from typing import Iterable, Tuple
import torch
import triton.testing as tt

from dion.newton_schulz_triton import (
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)

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
    *,
    batch_size: int = 1,
    steps: int = 5,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """Time reference vs. Triton kernels once and return the two runtimes (ms)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this benchmark")

    G = torch.randn(batch_size, m, n, dtype=dtype, device="cuda")
    # reference
    t_ref = tt.do_bench(lambda: zeropower_via_newtonschulz5(G))
    # triton
    # start with a warmup run
    newton_schulz_triton(G)
    # then measure the actual time
    t_tri = tt.do_bench(lambda: newton_schulz_triton(G))

    flops = gemm_cost(m, n)
    ref_tflops = tflops(t_ref, flops, steps, batch_size)
    tri_tflops = tflops(t_tri, flops, steps, batch_size)

    print(
        f"[{batch_size=}  {m=}, {n=}]  "
        f"torch {pretty_time(t_ref)}  {ref_tflops:5.2f} TFLOPS  |  "
        f"triton {pretty_time(t_tri)}  {tri_tflops:5.2f} TFLOPS  "
        f"(speed-up x{t_ref/t_tri:4.2f})"
    )
    return t_ref, t_tri


def bench_grid(
    dims: Iterable[int],
    *,
    expansion: int = 1,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
):
    """Sweep over square/rectangular sizes (equiv. to original benchmark_many_sizes)."""
    speedups = []
    for d in dims:
        tr, tt_ = bench_once(
            d,
            d * expansion,
            batch_size=batch_size,
            dtype=dtype,
        )
        speedups.append(tr / tt_)
    print("Speed-ups:", ", ".join(f"{s:4.2f}x" for s in speedups))
    print("Theoretical max:", f"{(4*expansion+2)/(3*expansion+1):4.2f}x")


def bench_plot(batch_size: int, *, out_dir: Path = Path("plots")):
    """Generate TFLOPS vs. size curves using Triton's perf_report helper."""
    if tt is None:
        raise RuntimeError("Triton not available - cannot build plots")

    @tt.perf_report(
        tt.Benchmark(
            x_names=["dim"],
            x_vals=[128 * i for i in range(1, 8)],
            line_arg="provider",
            line_vals=["torch", "triton"],
            line_names=["torch", "triton"],
            ylabel="TFLOPS",
            plot_name=f"newton_schulz_batch{batch_size}",
            args={"batch_size": batch_size},
        )
    )
    def bench(dim: int, provider: str, batch_size: int):
        G = torch.randn(batch_size, dim, dim, dtype=torch.bfloat16, device="cuda")
        if provider == "torch":
            ms = tt.do_bench(lambda: zeropower_via_newtonschulz5(G))
        else:  # "triton"
            ms = tt.do_bench(lambda: newton_schulz_triton(G))
        return tflops(ms, gemm_cost(dim, dim), steps=5, batch=batch_size)

    bench.run(print_data=True, save_path=str(out_dir))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmarks for Newton-Schulz Triton kernels"
    )
    # mutually exclusive groups
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true", help="run a single benchmark")
    mode.add_argument("--grid", action="store_true", help="sweep a list of sizes")
    mode.add_argument(
        "--plot", action="store_true", help="generate TFLOPS curves and write plots"
    )
    # single run parameters
    p.add_argument("--m", type=int, help="rows")
    p.add_argument("--n", type=int, help="cols (defaults to m)")
    # common options
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--expansion", type=int, default=1, help="n = m * expansion (grid mode)"
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="input dtype",
    )
    return p.parse_args()


def main():
    args = parse()

    # -----------------------------------------------------------------------------#
    # General settings
    # -----------------------------------------------------------------------------#

    # Allow a lot of recompiles in Torch-Triton
    torch._dynamo.config.cache_size_limit = 100  # noqa: SLF001

    dtype = getattr(torch, args.dtype)

    if args.grid:
        dims = [512, 1024, 2048, 4096, 8192]
        bench_grid(
            dims,
            expansion=args.expansion,
            batch_size=args.batch_size,
            dtype=dtype,
        )
    elif args.plot:
        bench_plot(args.batch_size)
    else:  # single run
        m = args.m
        n = args.n or m
        bench_once(m, n, batch_size=args.batch_size, dtype=dtype)


if __name__ == "__main__":
    main()
