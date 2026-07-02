"""Correctness tests for Dion2/NorDion2 ``selection_scope`` on the FSDP2
row-sharded path.

Validates the two fixes for the padding regression (selected rows were padded
back to the full shard before the all-to-all, erasing Dion's fraction saving):

- ``selection_scope="local"``: per-shard top-k, communicate only the selected
  rows. Must be numerically identical to the previous full-pad implementation
  (the dropped padded zero rows never affected U^T U). ``test_local_scope_nopad
  _equals_pad`` checks this directly by running the *same* optimizer once through
  the no-pad fast path and once through the forced full-shard-pad path.
- ``selection_scope="global"``: select on the assembled whole matrix. Must be
  invariant to the sharding layout (same result at world_size 1 and 2) and equal
  to a single-process whole-matrix reference.

Runs on 2 GPUs via NCCL. The momentum buffers are seeded identically across the
sharded and reference runs.

The test matrix is wide (rows <= cols) on purpose: the row-sharded path always
selects along the sharded (row) dimension, but an *unsharded* Dion2 selects the
shorter dimension, so the single-GPU reference only selects rows -- and is thus
comparable to the sharded run -- when rows <= cols. (NorDion2 always selects
rows.) Tolerances are bf16-scale: the communicated shards and Newton-Schulz run
in bf16, whose unit-in-last-place near magnitude 1 is ~8e-3, so exact-arithmetic
equalities only hold to ~1e-2.
"""

import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DeviceMesh, distribute_tensor, Shard

from dion.dion2 import Dion2, dion2_capped_threshold_async
from dion.nordion2 import NorDion2
from dion.polar_express import polar_express


CUDA = torch.cuda.device_count() if torch.cuda.is_available() else 0

# bf16 comm + Newton-Schulz: ~1 ULP near magnitude 1 is ~8e-3, so
# exact-in-real-arithmetic identities only hold to bf16 scale.
BF16_RTOL = 1.5e-2
BF16_ATOL = 1.5e-2


def _ns(x, epsilon=1e-7):
    return polar_express(x, epsilon=epsilon)


# ---- single-process reference: one optimizer step on the whole matrix ----
def _reference_step(OptCls, W0, G, *, fraction, scope, **kw):
    """Run one step on an unsharded (single-GPU) param. With no process_group,
    the optimizer holds the whole matrix locally, so local and global selection
    coincide -- this is the ground truth for the global-scope sharded run, and
    (for fraction=1.0) for any scope."""
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


def _worker(rank, world_size, port, OptCls, scope, fraction, kw, out_path, force_pad, shape):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dev = torch.device(f"cuda:{rank}")
    mesh = DeviceMesh("cuda", list(range(world_size)))

    if force_pad:
        # Force the pre-fix full-shard-pad path: wrap megabatch_orthogonalize_async
        # so local_comm_size is dropped (None). The local scope still selects its
        # k rows, but the megabatch then pads each shard back to the full
        # ceil(global/world_size) size before the all-to-all -- the old behavior
        # whose padded zero rows this PR eliminates. Patch both modules' bound
        # name (each imports megabatch_orthogonalize_async into its namespace).
        import dion.dion2 as _d2
        import dion.nordion2 as _nd2
        _real_mb = _d2.megabatch_orthogonalize_async
        def _forced_mb(*a, **k):
            k["local_comm_size"] = None
            return _real_mb(*a, **k)
        _d2.megabatch_orthogonalize_async = _forced_mb
        _nd2.megabatch_orthogonalize_async = _forced_mb

    # Wide (rows <= cols) so the unsharded reference also selects rows; see the
    # module docstring. Deterministic whole-matrix weight + grad on every rank.
    rows, cols = shape
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


def _run_sharded(
    OptCls, scope, fraction, kw, world_size, port, tmp, force_pad=False, shape=(16, 32)
):
    out = str(tmp / f"out_{port}.pt")
    mp.spawn(
        _worker,
        args=(world_size, port, OptCls, scope, fraction, kw, out, force_pad, shape),
        nprocs=world_size,
        join=True,
    )
    return torch.load(out)


def test_global_scope_k_uses_global_size_not_padded():
    """Global-scope selection must derive k from the TRUE unsharded size, not the
    zero-padded assembled size. On the row-sharded path the matrix handed to the
    select-and-orthogonalize wrapper is padded to ceil(global/world)*world rows;
    deriving k from that padded size selects ceil(fraction*padded) slices -- more
    than the whole-matrix top-k, and a count that varies with world_size -- when
    global is not divisible by world_size, silently breaking the exact/reproducible
    guarantee that motivates the global default. The existing sharded tests use a
    divisible matrix (16 rows, 2 ranks) so they never exercise this. Pure-CPU unit
    test of the wrapper (no GPUs, no distribution)."""
    import math
    from dion.dion2 import _make_select_and_orthogonalize

    fraction, global_rows, cols = 0.35, 17, 6
    # Simulate world_size=4: padded_local=ceil(17/4)=5 -> assembled = 20 rows
    # (17 real + 3 zero-pad), so ceil(0.35*20)=7 but the true top-k is
    # ceil(0.35*17)=6.
    padded_rows = 20
    torch.manual_seed(0)
    X = torch.cat(
        [torch.randn(global_rows, cols), torch.zeros(padded_rows - global_rows, cols)],
        dim=0,
    )
    # Identity NS isolates the selection logic from the orthogonalization.
    sel_ns = _make_select_and_orthogonalize(
        lambda t, epsilon=None: t, fraction, -2, global_select_size=global_rows
    )
    out = sel_ns(X)
    n_selected = int((out.abs().sum(-1) > 0).sum())
    assert n_selected == math.ceil(fraction * global_rows)  # 6, not ceil(.35*20)=7
    # Padded zero rows must never be selected.
    assert out[global_rows:].abs().sum() == 0


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
    torch.testing.assert_close(d["W"], ref, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.skipif(CUDA < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("OptCls,kw", [
    (Dion2, {}),
    (NorDion2, {"mu": 0.95, "muon_beta2": 0.95}),
])
def test_global_scope_uneven_shards_matches_reference(OptCls, kw, tmp_path):
    """global-scope over shards that do NOT divide evenly must still equal the
    single-GPU whole-matrix reference. 17 rows over 4 ranks chunk to [5,5,5,2];
    each shard zero-pads to ceil(17/4)=5, so the assembled matrix has 20 rows
    (17 real + 3 pad). With fraction=0.35 the true top-k is ceil(0.35*17)=6, but a
    k derived from the padded size would be ceil(0.35*20)=7 -- selecting an extra
    row and diverging from the reference (and varying with world_size). This is
    the uneven-division counterpart to the divisible test above, and exercises the
    fix end-to-end on the row-sharded path."""
    d = _run_sharded(OptCls, "global", 0.35, kw, 4, 29670, tmp_path, shape=(17, 32))
    W0 = d["W0"].cuda(); G = d["G"].cuda()
    ref = _reference_step(OptCls, W0, G, fraction=0.35, scope="global", **kw).cpu()
    torch.testing.assert_close(d["W"], ref, rtol=BF16_RTOL, atol=BF16_ATOL)


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
def test_local_scope_nopad_equals_pad(OptCls, kw, tmp_path):
    """The headline fix: the local no-pad fast path must be numerically identical
    to the old full-shard-pad path. Both select the same per-shard top-k rows;
    the only difference is whether the dropped rows are zero-padded back before
    the all-to-all (they never affect U^T U). Compare the real optimizer against
    itself with the pad path forced on."""
    nopad = _run_sharded(OptCls, "local", 0.25, kw, 2, 29650, tmp_path)
    pad = _run_sharded(OptCls, "local", 0.25, kw, 2, 29660, tmp_path, force_pad=True)
    torch.testing.assert_close(nopad["W"], pad["W"], rtol=BF16_RTOL, atol=BF16_ATOL)


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
    torch.testing.assert_close(dl["W"], dg["W"], rtol=BF16_RTOL, atol=BF16_ATOL)


# ---- selection_scope="global_capped" ----
#
# The capacity rule: all-gather slice norms only, every rank derives the same
# global top-(k*world_size) threshold, and only slices at/above it ("winners")
# are applied -- spare slots travel as zeros and non-winners skip error-feedback
# decay so their momentum accumulates until they win. NOTE the budget is
# top-(k*world_size), NOT top-(ceil(fraction*global)); they coincide when
# world_size divides the sharded dim (test below) and the capped budget is
# slightly larger otherwise (empty-shard test below exploits this: with
# k*world >= real rows, everything wins and capped(f) == global(1.0)).


def _capped_worker(rank, world_size, port, OptCls, scope, kw, out_path, shape,
                   fraction, boost, steps):
    """Multi-step worker with a crafted gradient: small noise everywhere, plus
    rows in ``boost`` set to a constant so their momentum L1 norm is exactly the
    boost magnitude. Steps after the first use a zero gradient (isolates the
    error-feedback dynamics). Saves the full W after every step."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dev = torch.device(f"cuda:{rank}")
    mesh = DeviceMesh("cuda", list(range(world_size)))

    rows, cols = shape
    g = torch.Generator(device=dev).manual_seed(1234)
    W0 = torch.randn(rows, cols, generator=g, device=dev)
    G = torch.randn(rows, cols, generator=g, device=dev) * 0.01
    for r, mag in boost.items():
        G[r] = mag / cols  # exact row L1 norm == mag

    p = torch.nn.Parameter(distribute_tensor(W0, mesh, [Shard(0)]))
    opt = OptCls(
        [dict(params=[p])],
        distributed_mesh=mesh,
        lr=0.1,
        fraction=fraction,
        newton_schulz_func=_ns,
        selection_scope=scope,
        **kw,
    )
    Ws = []
    for s in range(steps):
        grad = G if s == 0 else torch.zeros_like(G)
        p.grad = distribute_tensor(grad, mesh, [Shard(0)])
        opt.step()
        Ws.append(p.detach().full_tensor().cpu())
    if rank == 0:
        torch.save({"W0": W0.cpu(), "G": G.cpu(), "Ws": Ws}, out_path)
    dist.destroy_process_group()


def _run_capped(OptCls, scope, kw, port, tmp, *, shape=(16, 32), fraction=0.25,
                boost=None, steps=1):
    out = str(tmp / f"out_{port}.pt")
    mp.spawn(
        _capped_worker,
        args=(2, port, OptCls, scope, kw, out, shape, fraction, boost or {}, steps),
        nprocs=2,
        join=True,
    )
    return torch.load(out)


def _changed_rows(W_after, W_before):
    """Row indices where anything changed at all (bitwise)."""
    return set((W_after != W_before).any(dim=-1).nonzero().flatten().tolist())


@pytest.mark.skipif(CUDA < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("OptCls,kw", [
    (Dion2, {}),
    (NorDion2, {"mu": 0.95, "muon_beta2": 0.95}),
])
def test_capped_scope_matches_global_when_winners_fit(OptCls, kw, tmp_path):
    """When every rank owns at most k winners and world_size divides the rows
    (so the top-(k*world) budget equals global's top-ceil(fraction*global)),
    capped must equal global-scope exactly. 16 rows / 2 ranks / fraction 0.25:
    k=2, budget 4; the 4 boosted rows are split 2 per rank."""
    boost = {0: 3.0, 1: 2.9, 8: 2.8, 9: 2.7}
    dc = _run_capped(OptCls, "global_capped", kw, 29680, tmp_path, boost=boost)
    dg = _run_capped(OptCls, "global", kw, 29681, tmp_path, boost=boost)
    torch.testing.assert_close(dc["Ws"][0], dg["Ws"][0],
                               rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.skipif(CUDA < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("OptCls,kw", [
    (Dion2, {"ef_decay": 0.5, "weight_decay": 0.0}),
    (NorDion2, {"mu": 0.5, "muon_beta2": 0.95, "weight_decay": 0.0}),
])
def test_capped_scope_defers_overflow_winners(OptCls, kw, tmp_path):
    """All 4 global winners live on rank 0 but it only has k=2 slots: step 1
    applies exactly its top-2 {0,1} (rows 2,3 deferred WITHOUT error-feedback
    decay; rank 1's spare slots carry zeros, so NO row of rank 1 changes --
    this is what distinguishes capped from local, which would fill and apply
    rank 1's top-2). Step 2 (zero grad): rows 0,1 decayed to 1.5/1.45 while the
    deferred 2,3 kept 2.8/2.7, so exactly {2,3} are applied. weight_decay=0
    makes non-applied rows bitwise unchanged."""
    boost = {0: 3.0, 1: 2.9, 2: 2.8, 3: 2.7}
    d = _run_capped(OptCls, "global_capped", kw, 29682, tmp_path,
                    boost=boost, steps=2)
    assert _changed_rows(d["Ws"][0], d["W0"]) == {0, 1}
    assert _changed_rows(d["Ws"][1], d["Ws"][0]) == {2, 3}


@pytest.mark.skipif(CUDA < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("OptCls,kw", [
    (Dion2, {}),
    (NorDion2, {"mu": 0.95, "muon_beta2": 0.95}),
])
def test_capped_scope_uneven_shards_fraction_one_equals_global(OptCls, kw, tmp_path):
    """Uneven division (17 rows over 4 ranks -> [5,5,5,2] + one short shard):
    at fraction=1.0 the threshold is the -1 pad value, everything wins, and
    capped must equal global (both apply all 17 rows; the megabatch zero-pads
    the short shard's spare slots)."""
    dc = _run_sharded(OptCls, "global_capped", 1.0, kw, 4, 29683, tmp_path,
                      shape=(17, 32))
    dg = _run_sharded(OptCls, "global", 1.0, kw, 4, 29684, tmp_path,
                      shape=(17, 32))
    torch.testing.assert_close(dc["W"], dg["W"], rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.skipif(CUDA < 4, reason="needs 4 GPUs")
def test_capped_scope_uneven_shards_runs(tmp_path):
    """Partial-fraction sanity on uneven shards: finite and actually updates."""
    d = _run_sharded(Dion2, "global_capped", 0.35, {}, 4, 29685, tmp_path,
                     shape=(17, 32))
    assert torch.isfinite(d["W"]).all()
    assert not torch.allclose(d["W"], d["W0"], atol=1e-6)


@pytest.mark.skipif(CUDA < 4, reason="needs 4 GPUs")
def test_capped_scope_empty_shard(tmp_path):
    """3 rows over 4 ranks chunk to [1,1,1,0]: rank 3's shard is empty. The
    capped budget k*world = 4 exceeds the 3 real rows, so the threshold is the
    -1 pad value and every real row wins -- which also makes capped(0.5) equal
    global(1.0), a live demonstration of the top-(k*world) vs
    top-(ceil(fraction*global)) budget distinction."""
    dc = _run_sharded(Dion2, "global_capped", 0.5, {}, 4, 29686, tmp_path,
                      shape=(3, 32))
    assert torch.isfinite(dc["W"]).all()
    dg = _run_sharded(Dion2, "global", 1.0, {}, 4, 29687, tmp_path,
                      shape=(3, 32))
    torch.testing.assert_close(dc["W"], dg["W"], rtol=BF16_RTOL, atol=BF16_ATOL)


def _thresh_consistency_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dev = torch.device(f"cuda:{rank}")

    # Distinct norms everywhere (no ties), different on each rank; one short
    # shard (rank 1 has 3 of padded_local=4 slices) to exercise the -1 pad.
    N, padded_local, k = 2, 4, 2
    local = padded_local - rank  # rank0: 4, rank1: 3
    norms = (torch.arange(N * local, dtype=torch.float32, device=dev)
             .reshape(N, local) * 1.7 + 3.0 + rank * 0.37)

    gen = dion2_capped_threshold_async(norms, padded_local, k, world_size, None)
    next(gen)
    try:
        next(gen)
        raise AssertionError("generator should be exhausted after the collective")
    except StopIteration as e:
        thresh = e.value

    assert thresh.shape == (N, 1)
    # (f) rank consistency: every rank must hold the bit-identical threshold.
    both = [torch.empty_like(thresh) for _ in range(world_size)]
    dist.all_gather(both, thresh.contiguous())  # thresh is a topk slice view
    assert torch.equal(both[0], both[1])
    # With distinct norms and k_total <= real slices, exactly k*world_size
    # slices per matrix sit at/above the threshold globally.
    winners = (norms >= thresh).sum()
    dist.all_reduce(winners)
    assert winners.item() == N * k * world_size
    dist.destroy_process_group()


@pytest.mark.skipif(CUDA < 2, reason="needs 2 GPUs")
def test_capped_threshold_rank_consistent():
    mp.spawn(_thresh_consistency_worker, args=(2, 29688), nprocs=2, join=True)


def test_capped_threshold_negative_pad_raises():
    """local_size > padded_local would make F.pad silently TRIM the norms; the
    guard must raise instead. (Raise happens before the collective, so no
    process group is needed.) Pure-CPU unit test."""
    gen = dion2_capped_threshold_async(torch.zeros(1, 5), 3, 1, 2, None)
    with pytest.raises(ValueError, match="exceeds padded size"):
        next(gen)
