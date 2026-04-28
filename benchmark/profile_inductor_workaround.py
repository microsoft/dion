"""Profile the loop_ordering_after_fusion workaround on dion2_pre_orthogonalize.

Compares full Dion2 optimizer step throughput with and without the workaround.
Uses same-shape params only (different shapes crash without the workaround).

Usage:
    python benchmark/profile_inductor_workaround.py [--iters 200] [--warmup 20] [--n_params 8] [--shape 768 768]
"""

import argparse
import time
import torch


def bench_optimizer(label, dion2_module, shape, n_params, warmup, iters):
    Dion2 = dion2_module.Dion2
    params = [torch.nn.Parameter(torch.randn(shape, device="cuda")) for _ in range(n_params)]
    opt = Dion2(params, lr=0.01)

    for _ in range(warmup):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    print(f"{label}: {us:.0f} us/step")
    return us


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--n_params", type=int, default=8)
    parser.add_argument("--shape", type=int, nargs="+", default=[768, 768])
    args = parser.parse_args()
    shape = tuple(args.shape)

    print(f"Shape: {shape}, params: {args.n_params}, warmup: {args.warmup}, iters: {args.iters}")
    print()

    import dion.dion2

    t_with = bench_optimizer("With workaround   ", dion.dion2, shape, args.n_params, args.warmup, args.iters)

    # Reset compiled caches and recompile without the config patch
    torch._dynamo.reset()
    orig = dion.dion2.dion2_pre_orthogonalize.__wrapped__
    dion.dion2.dion2_pre_orthogonalize = torch.compile(orig, fullgraph=True)

    t_without = bench_optimizer("Without workaround", dion.dion2, shape, args.n_params, args.warmup, args.iters)

    print(f"\nDifference: {(t_with / t_without - 1) * 100:+.1f}%")


if __name__ == "__main__":
    main()
