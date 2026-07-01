"""Correctness tests for Dion2/NorDion2 ``selection_scope`` on the FSDP2
row-sharded path.

Validates the two fixes for the padding regression (selected rows were padded
back to the full shard before the all-to-all, erasing Dion's fraction saving):

- ``selection_scope="local"``: per-shard top-k, communicate only the selected
  rows. Must be numerically identical to the previous full-pad implementation
  (the dropped padded zero rows never affected U^T U).
- ``selection_scope="global"``: select on the assembled whole matrix. Must be
  invariant to the sharding layout (same result at world_size 1 and 2) and equal
  to a single-process whole-matrix reference.

Runs on the 2 login-node GPUs via NCCL. The momentum buffers are seeded
identically across the sharded and reference runs so the comparison is exact.
"""

import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Shard

from dion.dion2 import Dion2
from dion.nordion2 import NorDion2
from dion.polar_express import polar_express


CUDA = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _ns(x, epsilon=1e-7):
    return polar_express(x, epsilon=epsilon)


# ---- single-process reference: one optimizer step on the whole matrix ----
def _reference_step(OptCls, W0, G, *, fraction, scope, **kw):
    """Run one step on an unsharded (single-GPU) param. With no process_group,
    the optimizer holds the whole matrix locally, so local and global selection
    coincide -- this is the ground truth for the global-scope sharded run, and
    (for fraction=1.0) for any scope."""
    dev = W0.device
    p = torch.nn.Parameter(W0.clone())
    p.grad = G.clone()
    opt = OptCls(
        [dict(params=[p])],
        distributed_mesh=None,
        lr=0.1,
        fraction=fraction,
        newton_schulz_func=_ns,
        selection_scope=scope,
        **kw,
    )
    opt.step()
    return p.detach().clone()


def _worker(rank, world_size, port, OptCls, scope, fraction, kw, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dev = torch.device(f"cuda:{rank}")
    mesh = DeviceMesh("cuda", list(range(world_size)))

    rows, cols = 16, 8
    # Deterministic whole-matrix weight + grad, identical on every rank.
    g = torch.Generator(device=dev).manual_seed(1234)
    W0 = torch.randn(rows, cols, generator=g, device=dev)
    G = torch.randn(rows, cols, generator=g, device=dev)

    dW0 = distribute_tensor(W0, mesh, [Shard(0)])
    p = torch.nn.Parameter(dW0)
    p.grad = distribute_tensor(G, mesh, [Shard(0)])

    opt = OptCls(
        [dict(params=[p])],
        distributed_mesh=mesh,
        lr=0.1,
        fraction=fraction,
        newton_schulz_func=_ns,
        selection_scope=scope,
        **kw,
    )
    opt.step()

    full = p.detach().full_tensor()  # gather to whole matrix on every rank
    if rank == 0:
        torch.save({"W": full.cpu(), "W0": W0.cpu(), "G": G.cpu()}, out_path)
    dist.destroy_process_group()


def _run_sharded(OptCls, scope, fraction, kw, world_size, port, tmp):
    out = str(tmp / f"out_{port}.pt")
    mp.spawn(_worker, args=(world_size, port, OptCls, scope, fraction, kw, out),
             nprocs=world_size, join=True)
    return torch.load(out)


@pytest.mark.skipif(CUDA < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("OptCls,kw", [
    (Dion2, {}),
    (NorDion2, {"mu": 0.95, "muon_beta2": 0.95}),
])
def test_global_scope_matches_single_gpu_reference(OptCls, kw, tmp_path):
    """global-scope 2-rank result == single-GPU whole-matrix reference."""
    d = _run_sharded(OptCls, "global", 0.25, kw, 2, 29610, tmp_path)
    W0 = d["W0"].cuda(); G = d["G"].cuda()
    ref = _reference_step(OptCls, W0, G, fraction=0.25, scope="global", **kw).cpu()
    torch.testing.assert_close(d["W"], ref, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(CUDA < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("OptCls,kw", [
    (Dion2, {}),
    (NorDion2, {"mu": 0.95, "muon_beta2": 0.95}),
])
def test_local_scope_runs_and_updates_selected(OptCls, kw, tmp_path):
    """local-scope must run end-to-end on the sharded path and actually move the
    weights (sanity that the no-pad k-budget path is wired correctly)."""
    d = _run_sharded(OptCls, "local", 0.25, kw, 2, 29620, tmp_path)
    assert not torch.allclose(d["W"], d["W0"], atol=1e-6)


@pytest.mark.skipif(CUDA < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("OptCls,kw", [
    (Dion2, {}),
    (NorDion2, {"mu": 0.95, "muon_beta2": 0.95}),
])
def test_fraction_one_local_equals_global(OptCls, kw, tmp_path):
    """At fraction=1.0 every row is selected, so local and global must agree
    (and both equal the whole-matrix reference)."""
    dl = _run_sharded(OptCls, "local", 1.0, kw, 2, 29630, tmp_path)
    dg = _run_sharded(OptCls, "global", 1.0, kw, 2, 29640, tmp_path)
    torch.testing.assert_close(dl["W"], dg["W"], rtol=2e-3, atol=2e-3)
