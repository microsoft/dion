"""Tests for the Triton masked post-orthogonalize kernels.

``dion2_post_orthogonalize_masked_triton`` and
``nordion2_post_orthogonalize_masked_triton`` fuse the masked posts used by
the "global" and "global_capped" selection scopes. Parity is checked against
the compiled masked posts on identical inputs:

- Unselected rows must match bitwise for X (both compute a*x with one
  rounding) and be untouched for M/V.
- Selected rows differ at FP-rounding level: the compiled Dion2 path rounds
  ``neg_lr * u`` to U's dtype before the add (two/three roundings), the
  Triton kernel computes ``a*x + neg_b*u`` in fp32 with a single rounding.

The end-to-end test runs NorDion2 with selection_scope="global_capped" on
2 GPUs, triton_post_ortho True vs False, and compares final params.
"""

import multiprocessing as mp
import os
import pytest
import torch

from dion.dion2 import dion2_post_orthogonalize_masked
from dion.nordion2 import nordion2_post_orthogonalize_masked
from dion.dion2_triton import (
    TRITON_AVAILABLE,
    dion2_post_orthogonalize_masked_triton,
    nordion2_post_orthogonalize_masked_triton,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AND_CUDA = CUDA_AVAILABLE and TRITON_AVAILABLE

torch._dynamo.config.cache_size_limit = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_masked_data(
    n_mat, rows, cols, sel_rows_per_mat, seed=42, x_dtype=torch.float32,
    with_v=False,
):
    """Lists of X, M, (V,) U with U exactly zero outside selected rows.

    ``sel_rows_per_mat`` is a list (len n_mat) of per-matrix selected-row
    counts; 0 means no rows selected on that matrix.
    """
    torch.manual_seed(seed)
    X, M, V, U, sels = [], [], [], [], []
    for i in range(n_mat):
        X.append(torch.randn(rows, cols, device=DEVICE, dtype=x_dtype))
        M.append(torch.randn(rows, cols, device=DEVICE, dtype=x_dtype))
        if with_v:
            V.append(
                torch.rand(rows, 1, device=DEVICE, dtype=x_dtype) + 0.01
            )
        u = torch.zeros(rows, cols, device=DEVICE, dtype=torch.bfloat16)
        k = sel_rows_per_mat[i]
        sel = torch.randperm(rows, device=DEVICE)[:k]
        if k:
            u[sel] = torch.randn(k, cols, device=DEVICE, dtype=torch.bfloat16)
        U.append(u)
        mask = torch.zeros(rows, dtype=torch.bool, device=DEVICE)
        mask[sel] = True
        sels.append(mask)
    scalars = dict(
        base_lr=torch.tensor(0.01),
        adjusted_lr=torch.tensor(0.02),
        weight_decay=torch.tensor(0.1),
        ef_decay=torch.tensor(0.95),
    )
    if with_v:
        scalars["muon_beta2"] = torch.tensor(0.95)
        return X, M, V, U, sels, scalars
    return X, M, U, sels, scalars


def _tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    return dict(atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Function-level parity: Dion2 masked
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TRITON_AND_CUDA, reason="requires CUDA + triton")
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,cols,sel_counts",
    [
        (64, 128, [16, 0, 64]),  # partial / none / all selected
        (1, 32, [1, 0, 1]),      # single row
        (128, 4099, [32, 7, 0]), # cols > BLOCK_N, non-pow2
    ],
)
def test_dion2_masked_parity(x_dtype, rows, cols, sel_counts):
    X, M, U, sels, sc = _make_masked_data(
        len(sel_counts), rows, cols, sel_counts, x_dtype=x_dtype
    )
    X_ref = [x.clone() for x in X]
    M_ref = [m.clone() for m in M]

    dion2_post_orthogonalize_masked(
        X=X_ref, M=M_ref, U=U, select_dim=-2,
        base_lr=sc["base_lr"], adjusted_lr=sc["adjusted_lr"],
        weight_decay=sc["weight_decay"], ef_decay=sc["ef_decay"],
    )
    dion2_post_orthogonalize_masked_triton(
        X=X, M=M, U=U, select_dim=-2,
        base_lr=sc["base_lr"], adjusted_lr=sc["adjusted_lr"],
        weight_decay=sc["weight_decay"], ef_decay=sc["ef_decay"],
    )

    for x, x_ref, m, m_ref, sel in zip(X, X_ref, M, M_ref, sels):
        # Unselected rows: both paths compute a*x with one rounding -> bitwise.
        torch.testing.assert_close(x[~sel], x_ref[~sel], rtol=0, atol=0)
        # M untouched on unselected rows (compiled multiplies by exactly 1).
        torch.testing.assert_close(m[~sel], m_ref[~sel], rtol=0, atol=0)
        torch.testing.assert_close(x, x_ref, **_tols(x_dtype))
        torch.testing.assert_close(m, m_ref, **_tols(x_dtype))


@pytest.mark.skipif(not TRITON_AND_CUDA, reason="requires CUDA + triton")
def test_dion2_masked_batched_3d():
    """Leading batch dim: (B, rows, cols) with per-(b,row) selection."""
    torch.manual_seed(0)
    B, rows, cols = 3, 32, 64
    x = torch.randn(B, rows, cols, device=DEVICE)
    m = torch.randn(B, rows, cols, device=DEVICE)
    u = torch.zeros(B, rows, cols, device=DEVICE, dtype=torch.bfloat16)
    for b in range(B):
        sel = torch.randperm(rows)[: 4 * (b + 1)]
        u[b, sel] = torch.randn(len(sel), cols, device=DEVICE, dtype=torch.bfloat16)
    args = dict(
        base_lr=torch.tensor(0.01), adjusted_lr=torch.tensor(0.02),
        weight_decay=torch.tensor(0.1), ef_decay=torch.tensor(0.95),
        select_dim=-2,
    )
    x_ref, m_ref = x.clone(), m.clone()
    dion2_post_orthogonalize_masked(X=[x_ref], M=[m_ref], U=[u], **args)
    dion2_post_orthogonalize_masked_triton(X=[x], M=[m], U=[u], **args)
    torch.testing.assert_close(x, x_ref, **_tols(torch.float32))
    torch.testing.assert_close(m, m_ref, **_tols(torch.float32))


@pytest.mark.skipif(not TRITON_AND_CUDA, reason="requires CUDA + triton")
def test_dion2_masked_empty_shard_skipped():
    args = dict(
        base_lr=torch.tensor(0.01), adjusted_lr=torch.tensor(0.02),
        weight_decay=torch.tensor(0.1), ef_decay=torch.tensor(0.95),
        select_dim=-2,
    )
    x = torch.empty(0, 64, device=DEVICE)
    m = torch.empty(0, 64, device=DEVICE)
    u = torch.empty(0, 64, device=DEVICE, dtype=torch.bfloat16)
    # Must not raise.
    dion2_post_orthogonalize_masked_triton(X=[x], M=[m], U=[u], **args)


# ---------------------------------------------------------------------------
# Function-level parity: NorDion2 masked
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TRITON_AND_CUDA, reason="requires CUDA + triton")
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,cols,sel_counts",
    [
        (64, 128, [16, 0, 64]),
        (128, 4099, [32, 7, 0]),
        (16, 20480, [4, 16, 0]),  # production-wide rows
    ],
)
def test_nordion2_masked_parity(x_dtype, rows, cols, sel_counts):
    X, M, V, U, sels, sc = _make_masked_data(
        len(sel_counts), rows, cols, sel_counts, x_dtype=x_dtype, with_v=True
    )
    X_ref = [x.clone() for x in X]
    M_ref = [m.clone() for m in M]
    V_ref = [v.clone() for v in V]

    kw = dict(
        base_lr=sc["base_lr"], adjusted_lr=sc["adjusted_lr"],
        weight_decay=sc["weight_decay"], ef_decay=sc["ef_decay"],
        muon_beta2=sc["muon_beta2"], select_dim=-2,
    )
    nordion2_post_orthogonalize_masked(X=X_ref, M=M_ref, V=V_ref, U=U, **kw)
    nordion2_post_orthogonalize_masked_triton(X=X, M=M, V=V, U=U, **kw)

    for x, x_ref, m, m_ref, v, v_ref, sel in zip(
        X, X_ref, M, M_ref, V, V_ref, sels
    ):
        torch.testing.assert_close(x[~sel], x_ref[~sel], rtol=0, atol=0)
        torch.testing.assert_close(m[~sel], m_ref[~sel], rtol=0, atol=0)
        torch.testing.assert_close(v[~sel], v_ref[~sel], rtol=0, atol=0)
        torch.testing.assert_close(x, x_ref, **_tols(x_dtype))
        torch.testing.assert_close(m, m_ref, **_tols(x_dtype))
        torch.testing.assert_close(v, v_ref, **_tols(x_dtype))


@pytest.mark.skipif(not TRITON_AND_CUDA, reason="requires CUDA + triton")
def test_nordion2_masked_no_rows_selected_anywhere():
    """All-zero U: pure weight decay, V and M untouched, no NaN from the
    Frobenius rescale (norm 0 / clamp)."""
    X, M, V, U, sels, sc = _make_masked_data(
        2, 32, 64, [0, 0], with_v=True
    )
    kw = dict(
        base_lr=sc["base_lr"], adjusted_lr=sc["adjusted_lr"],
        weight_decay=sc["weight_decay"], ef_decay=sc["ef_decay"],
        muon_beta2=sc["muon_beta2"], select_dim=-2,
    )
    X_ref = [x.clone() for x in X]
    M_ref = [m.clone() for m in M]
    V_ref = [v.clone() for v in V]
    nordion2_post_orthogonalize_masked(X=X_ref, M=M_ref, V=V_ref, U=U, **kw)
    nordion2_post_orthogonalize_masked_triton(X=X, M=M, V=V, U=U, **kw)
    for x, x_ref in zip(X, X_ref):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x, x_ref, rtol=0, atol=0)
    for m, m_ref in zip(M, M_ref):
        torch.testing.assert_close(m, m_ref, rtol=0, atol=0)
    for v, v_ref in zip(V, V_ref):
        torch.testing.assert_close(v, v_ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# End-to-end: NorDion2 global_capped, triton vs compiled post (2 GPUs)
# ---------------------------------------------------------------------------

def _e2e_worker(rank, world_size, port, triton_post, out_path):
    import torch.distributed as dist
    from dion import NorDion2

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    torch.manual_seed(1234)  # same init on all ranks (replicated DDP-style)
    full = [torch.randn(64, 96) for _ in range(3)]
    params = [
        torch.nn.Parameter(
            torch.tensor_split(w, world_size, dim=0)[rank].clone().to(device)
        )
        for w in full
    ]
    opt = NorDion2(
        [dict(params=params)],
        distributed_mesh=dist.group.WORLD,
        lr=0.01,
        fraction=0.25,
        selection_scope="global_capped",
        capacity_factor=1.5,
        use_triton=False,
        triton_post_ortho=triton_post,
    )
    torch.manual_seed(99)
    for _ in range(3):
        grads = [torch.randn(64, 96) for _ in range(3)]
        for p, g in zip(params, grads):
            p.grad = torch.tensor_split(g, world_size, dim=0)[rank].clone().to(device)
        opt.step()
        opt.zero_grad()

    if rank == 0:
        torch.save([p.detach().cpu() for p in params], out_path)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not TRITON_AND_CUDA or torch.cuda.device_count() < 2,
    reason="requires >=2 GPUs + triton",
)
def test_nordion2_capped_e2e_triton_vs_compiled(tmp_path):
    ctx = mp.get_context("spawn")
    outs = {}
    for triton_post, port in ((False, 29711), (True, 29713)):
        out = tmp_path / f"params_{triton_post}.pt"
        procs = [
            ctx.Process(
                target=_e2e_worker, args=(r, 2, port, triton_post, str(out))
            )
            for r in range(2)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
            assert p.exitcode == 0
        outs[triton_post] = torch.load(out)

    for p_ref, p_tri in zip(outs[False], outs[True]):
        torch.testing.assert_close(p_tri, p_ref, atol=1e-5, rtol=1e-4)
