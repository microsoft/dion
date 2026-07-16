"""CUDA-graph capture of the optimizer step (``dion.cuda_graph.CudaGraphOptimizer``).

The megabatched Muon/NorMuon/Dion2/NorDion2 step is a fixed-shape computation (top-k
selection changes which rows are chosen, not the shapes), so capturing it once and
replaying collapses the per-matrix host dispatch to a single graph launch. These tests
check that the wrapper is numerically correct:

- capture+replay matches running the bare optimizer eagerly, step for step, for the
  matrix (Muon/NorMuon/Dion2/NorDion2) path and the AdamW scalar path together; and
- a per-step LR schedule takes effect under replay -- the LR is carried as a device
  tensor the optimizer reads and the wrapper refreshes before each replay -- rather than
  freezing at the capture-time value.
"""
import pytest
import torch

from dion import Dion2, Muon, NorDion2, NorMuon
from dion.cuda_graph import CudaGraphOptimizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(DEVICE == "cpu", reason="requires CUDA")

STEPS, WARMUP, SEED = 12, 3, 0
OPTIMIZERS = [Muon, NorMuon, Dion2, NorDion2]


def _build(optimizer_cls):
    torch.manual_seed(SEED)
    weights = [torch.nn.Parameter(torch.randn(64, 128, device=DEVICE)),
               torch.nn.Parameter(torch.randn(128, 64, device=DEVICE))]
    biases = [torch.nn.Parameter(torch.randn(64, device=DEVICE)),
              torch.nn.Parameter(torch.randn(128, device=DEVICE))]
    opt = optimizer_cls([
        {"params": weights},
        {"params": biases, "algorithm": "adamw"},
    ], distributed_mesh=None, lr=0.02)
    return weights + biases, opt


def _grad_seq(params):
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    return [[torch.randn(p.shape, device=DEVICE, generator=gen) for p in params]
            for _ in range(STEPS)]


def _run(params, opt, grad_seq, step_fn, lrs=None):
    # Stable .grad buffers (the graph pins them); refill in place, never reallocate.
    for p in params:
        p.grad = torch.zeros_like(p)
    for t, gs in enumerate(grad_seq):
        if lrs is not None:
            for g in opt.param_groups:
                g["lr"] = lrs[t]
        for p, g in zip(params, gs):
            p.grad.copy_(g)
        step_fn()
    return [p.detach().clone() for p in params]


@pytest.mark.parametrize("optimizer_cls", OPTIMIZERS)
def test_cudagraph_matches_eager(optimizer_cls):
    p0, _ = _build(optimizer_cls)
    grad_seq = _grad_seq(p0)

    pe, oe = _build(optimizer_cls)
    final_eager = _run(pe, oe, grad_seq, oe.step)

    pg, og = _build(optimizer_cls)
    wrap = CudaGraphOptimizer(og, warmup_steps=WARMUP)
    final_graph = _run(pg, og, grad_seq, wrap.step)

    diff = max((a - b).abs().max().item() for a, b in zip(final_eager, final_graph))
    assert diff <= 1e-5, f"{optimizer_cls.__name__}: capture-vs-eager diff {diff:.3e}"


@pytest.mark.parametrize("optimizer_cls", OPTIMIZERS)
def test_cudagraph_tracks_scheduled_lr(optimizer_cls):
    p0, o0 = _build(optimizer_cls)
    base = o0.param_groups[0]["lr"]
    grad_seq = _grad_seq(p0)
    # A schedule that varies ~8x across the run (warmup then decay).
    sched = [base * (0.25 + 1.5 * (t + 1) / STEPS) for t in range(STEPS)]
    const = [base] * STEPS

    pe, oe = _build(optimizer_cls)
    final_eager = _run(pe, oe, grad_seq, oe.step, lrs=sched)

    pg, og = _build(optimizer_cls)
    wrap = CudaGraphOptimizer(og, warmup_steps=WARMUP)
    final_graph = _run(pg, og, grad_seq, wrap.step, lrs=sched)

    pc, oc = _build(optimizer_cls)
    wrapc = CudaGraphOptimizer(oc, warmup_steps=WARMUP)
    final_const = _run(pc, oc, grad_seq, wrapc.step, lrs=const)

    tracks = max((a - b).abs().max().item() for a, b in zip(final_eager, final_graph))
    nontrivial = max((a - b).abs().max().item() for a, b in zip(final_graph, final_const))
    assert tracks <= 1e-5, (
        f"{optimizer_cls.__name__}: graph did not track the schedule (diff {tracks:.3e})"
    )
    assert nontrivial > 1e-3, (
        f"{optimizer_cls.__name__}: schedule looks frozen -- scheduled and constant runs "
        f"agree to {nontrivial:.3e}, so the test would pass even if the LR were baked"
    )
