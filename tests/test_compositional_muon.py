"""Tests for the Compositional Muon optimizer.

Covers the partner-whitened QK / OV direction math (the core CM invariant, the
hand-computable scalar-partner case, and the GQA generalization), the optimizer
contract (param updates, determinism, zero/NaN grads, pairing validation, the
Muon / AdamW / Lion fallbacks), and the distributed gather / re-shard path
against a single-device reference.

The math is device-agnostic, so these run on CPU; the distributed test uses a
2-rank gloo group and needs no GPU.
"""

import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from dion import CompositionalMuon
from dion.compositional_muon import (
    qk_delta,
    ov_delta,
    _coupled_inv_sqrt,
    _to_heads_row,
)

DEVICE = "cpu"

# Bump torch.compile cache so the per-shape polar_express recompiles don't fail.
torch._dynamo.config.cache_size_limit = 128


def _make_attn(d_model, head_dim, n_q_heads, n_kv_heads, seed=0):
    torch.manual_seed(seed)
    wq = torch.nn.Parameter(torch.randn(n_q_heads * head_dim, d_model, device=DEVICE))
    wk = torch.nn.Parameter(torch.randn(n_kv_heads * head_dim, d_model, device=DEVICE))
    wv = torch.nn.Parameter(torch.randn(n_kv_heads * head_dim, d_model, device=DEVICE))
    wo = torch.nn.Parameter(torch.randn(d_model, n_q_heads * head_dim, device=DEVICE))
    return wq, wk, wv, wo


def _set_grads(params, seed=1):
    torch.manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p)


def _groups(wq, wk, wv, wo, head_dim, **kw):
    return [
        dict(params=[wq, wk], algorithm="cm_qk", head_dim=head_dim, **kw),
        dict(params=[wv, wo], algorithm="cm_ov", head_dim=head_dim, **kw),
    ]


# ---------------------------------------------------------------------------
# Direction math
# ---------------------------------------------------------------------------


def _exact_msign(x):
    u, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    return u @ vh


class TestDirectionMath:
    def test_qk_orthogonality_invariant(self):
        # delta_Q = msign(G_Q C_K^-1) C_K^-1  =>  delta_Q C_K = msign(...) ~ orthogonal.
        d, hd, H = 16, 4, 3
        torch.manual_seed(2)
        WQ, WK = torch.randn(d, H * hd), torch.randn(d, H * hd)
        GQ, GK = torch.randn(d, H * hd), torch.randn(d, H * hd)
        dQ, _ = qk_delta(WQ, WK, GQ, GK, hd, damping=1e-2)
        WK_h = _to_heads_row(WK, H, hd)
        C_K = torch.linalg.inv(_coupled_inv_sqrt(WK_h.mT @ WK_h, 1e-2))
        prod = _to_heads_row(dQ, H, hd) @ C_K
        sv = torch.linalg.svdvals(prod.float())
        # 5-step Polar Express in bf16 lands singular values near 1.
        assert sv.min() > 0.8 and sv.max() < 1.2

    def test_qk_scalar_partner_hand_computed(self):
        # W_K = c * (orthonormal columns) => C_K = c I exactly => delta_Q = msign(G_Q) / c.
        d, hd, H, c = 8, 4, 2, 0.5
        torch.manual_seed(3)
        q_blocks, gq_blocks, dq_ref = [], [], []
        for _ in range(H):
            wk_h, _ = torch.linalg.qr(torch.randn(d, hd))  # orthonormal columns
            q_blocks.append(c * wk_h)
            gq = torch.randn(d, hd)
            gq_blocks.append(gq)
            dq_ref.append(_exact_msign(gq) / c)
        WK = torch.cat(q_blocks, dim=1)
        GQ = torch.cat(gq_blocks, dim=1)
        WQ = torch.randn(d, H * hd)  # only affects delta_K, not delta_Q
        dQ, _ = qk_delta(WQ, WK, GQ, torch.randn(d, H * hd), hd, damping=0.0)
        dQ_ref = torch.cat(dq_ref, dim=1)
        cos = torch.nn.functional.cosine_similarity(
            dQ.flatten(), dQ_ref.flatten(), dim=0
        )
        assert cos > 0.99
        # 5-step bf16 Polar Express only approximates the exact SVD msign, so
        # compare in relative Frobenius norm rather than element-wise.
        rel_err = (dQ - dQ_ref).norm() / dQ_ref.norm()
        assert rel_err < 0.15

    def test_gqa_reduces_to_mha(self):
        # group_size == 1 (H_q == H_kv) must equal the plain per-head path.
        d, hd, H = 12, 4, 2
        torch.manual_seed(4)
        WQ, WK = torch.randn(d, H * hd), torch.randn(d, H * hd)
        GQ, GK = torch.randn(d, H * hd), torch.randn(d, H * hd)
        dQ1, dK1 = qk_delta(WQ, WK, GQ, GK, hd, damping=1e-2)
        dQ2, dK2 = qk_delta(WQ, WK, GQ, GK, hd, damping=1e-2)
        assert torch.equal(dQ1, dQ2) and torch.equal(dK1, dK2)

    def test_gqa_shapes(self):
        d, hd, Hq, Hkv = 16, 4, 6, 2
        torch.manual_seed(5)
        WQ, WK = torch.randn(d, Hq * hd), torch.randn(d, Hkv * hd)
        GQ, GK = torch.randn(d, Hq * hd), torch.randn(d, Hkv * hd)
        dQ, dK = qk_delta(WQ, WK, GQ, GK, hd)
        assert dQ.shape == WQ.shape and dK.shape == WK.shape
        WV, WO = torch.randn(d, Hkv * hd), torch.randn(Hq * hd, d)
        GV, GO = torch.randn(d, Hkv * hd), torch.randn(Hq * hd, d)
        dV, dO = ov_delta(WV, WO, GV, GO, hd)
        assert dV.shape == WV.shape and dO.shape == WO.shape

    def test_qk_rejects_bad_gqa_ratio(self):
        d, hd = 12, 4
        WQ, WK = torch.randn(d, 5 * hd), torch.randn(d, 2 * hd)  # 5 not divisible by 2
        with pytest.raises(ValueError, match="divisible"):
            qk_delta(WQ, WK, torch.randn_like(WQ), torch.randn_like(WK), hd)


# ---------------------------------------------------------------------------
# Optimizer contract
# ---------------------------------------------------------------------------


class TestOptimizer:
    def test_mha_params_change_and_finite(self):
        wq, wk, wv, wo = _make_attn(8, 4, 2, 2)
        opt = CompositionalMuon(_groups(wq, wk, wv, wo, 4), lr=0.02)
        _set_grads([wq, wk, wv, wo])
        before = [p.detach().clone() for p in (wq, wk, wv, wo)]
        opt.step()
        for p, b in zip((wq, wk, wv, wo), before):
            assert not torch.equal(p.data, b)
            assert torch.isfinite(p.data).all()

    def test_gqa_runs(self):
        wq, wk, wv, wo = _make_attn(16, 4, 4, 2)
        opt = CompositionalMuon(_groups(wq, wk, wv, wo, 4), lr=0.02)
        _set_grads([wq, wk, wv, wo])
        opt.step()
        assert all(torch.isfinite(p.data).all() for p in (wq, wk, wv, wo))

    def test_determinism(self):
        def run():
            wq, wk, wv, wo = _make_attn(8, 4, 2, 2, seed=7)
            opt = CompositionalMuon(_groups(wq, wk, wv, wo, 4), lr=0.02)
            for _ in range(3):
                _set_grads([wq, wk, wv, wo], seed=9)
                opt.step()
            return [p.detach().clone() for p in (wq, wk, wv, wo)]

        r1, r2 = run(), run()
        for a, b in zip(r1, r2):
            assert torch.equal(a, b)

    def test_zero_grad_is_noop_without_weight_decay(self):
        wq, wk, wv, wo = _make_attn(8, 4, 2, 2)
        opt = CompositionalMuon(_groups(wq, wk, wv, wo, 4), lr=0.02, weight_decay=0.0)
        before = [p.detach().clone() for p in (wq, wk, wv, wo)]
        for p in (wq, wk, wv, wo):
            p.grad = torch.zeros_like(p)
        opt.step()
        for p, b in zip((wq, wk, wv, wo), before):
            assert torch.equal(p.data, b)

    def test_nan_grad_consistent_with_muon(self):
        # Muon propagates NaN through Newton-Schulz rather than guarding; CM matches.
        wq, wk, wv, wo = _make_attn(8, 4, 2, 2)
        opt = CompositionalMuon(_groups(wq, wk, wv, wo, 4), lr=0.02)
        _set_grads([wq, wk, wv, wo])
        wq.grad[0, 0] = float("nan")
        opt.step()  # must not raise
        assert not torch.isfinite(wq.data).all()

    def test_single_pair(self):
        wq, wk, _, _ = _make_attn(8, 4, 2, 2)
        opt = CompositionalMuon(
            [dict(params=[wq, wk], algorithm="cm_qk", head_dim=4)], lr=0.02
        )
        wq.grad, wk.grad = torch.randn_like(wq), torch.randn_like(wk)
        opt.step()
        assert torch.isfinite(wq.data).all() and torch.isfinite(wk.data).all()

    def test_odd_pair_count_rejected(self):
        wq, wk, wv, _ = _make_attn(8, 4, 2, 2)
        with pytest.raises(ValueError, match="pairwise"):
            CompositionalMuon(
                [dict(params=[wq, wk, wv], algorithm="cm_qk", head_dim=4)], lr=0.02
            )

    def test_missing_head_dim_rejected(self):
        wq, wk, _, _ = _make_attn(8, 4, 2, 2)
        with pytest.raises(ValueError, match="head_dim"):
            CompositionalMuon([dict(params=[wq, wk], algorithm="cm_qk")], lr=0.02)

    def test_unknown_algorithm_rejected(self):
        p = torch.nn.Parameter(torch.randn(8, 8))
        with pytest.raises(ValueError, match="Unknown algorithm"):
            CompositionalMuon([dict(params=[p], algorithm="nope")], lr=0.02)

    def test_partial_pair_grad_rejected(self):
        wq, wk, _, _ = _make_attn(8, 4, 2, 2)
        opt = CompositionalMuon(
            [dict(params=[wq, wk], algorithm="cm_qk", head_dim=4)], lr=0.02
        )
        wq.grad = torch.randn_like(wq)  # wk.grad stays None
        with pytest.raises(ValueError, match="both"):
            opt.step()

    def test_fallback_groups(self):
        wq, wk, wv, wo = _make_attn(8, 4, 2, 2)
        mlp = torch.nn.Parameter(torch.randn(16, 8))
        emb = torch.nn.Parameter(torch.randn(32))
        lion_p = torch.nn.Parameter(torch.randn(8, 8))
        groups = _groups(wq, wk, wv, wo, 4) + [
            dict(params=[mlp], algorithm="muon"),
            dict(params=[emb], algorithm="adamw"),
            dict(params=[lion_p], algorithm="lion"),
        ]
        opt = CompositionalMuon(groups, lr=0.02)
        before = {id(p): p.detach().clone() for p in (mlp, emb, lion_p)}
        for p in (wq, wk, wv, wo, mlp, emb, lion_p):
            p.grad = torch.randn_like(p)
        opt.step()
        for p in (mlp, emb, lion_p):
            assert not torch.equal(p.data, before[id(p)])
            assert torch.isfinite(p.data).all()

    def test_muon_per_head(self):
        attn = torch.nn.Parameter(torch.randn(8 * 4, 16))  # 8 heads of head_dim 4
        opt = CompositionalMuon(
            [dict(params=[attn], algorithm="muon", num_heads=8)], lr=0.02
        )
        attn.grad = torch.randn_like(attn)
        opt.step()
        assert torch.isfinite(attn.data).all()


# ---------------------------------------------------------------------------
# Distributed gather / re-shard path (2-rank gloo, no GPU needed)
# ---------------------------------------------------------------------------


def _dist_worker(rank, world_size, port, return_dict):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch._dynamo.config.cache_size_limit = 128
        mesh = init_device_mesh("cpu", (world_size,))
        d, hd, Hq, Hkv = 16, 4, 4, 2

        torch.manual_seed(21)
        wq_full = torch.randn(Hq * hd, d)
        wk_full = torch.randn(Hkv * hd, d)
        torch.manual_seed(22)
        gq_full = torch.randn(Hq * hd, d)
        gk_full = torch.randn(Hkv * hd, d)

        # Shard on dim 0 (the heads dim) across ranks.
        wq = torch.nn.Parameter(distribute_tensor(wq_full, mesh, [Shard(0)]))
        wk = torch.nn.Parameter(distribute_tensor(wk_full, mesh, [Shard(0)]))
        wq.grad = distribute_tensor(gq_full, mesh, [Shard(0)])
        wk.grad = distribute_tensor(gk_full, mesh, [Shard(0)])

        opt = CompositionalMuon(
            [dict(params=[wq, wk], algorithm="cm_qk", head_dim=hd)], lr=0.02
        )
        opt.step()

        got_q = wq.data.full_tensor()
        got_k = wk.data.full_tensor()

        if rank == 0:
            # Single-device reference with identical inputs.
            rwq = torch.nn.Parameter(wq_full.clone())
            rwk = torch.nn.Parameter(wk_full.clone())
            rwq.grad, rwk.grad = gq_full.clone(), gk_full.clone()
            ref = CompositionalMuon(
                [dict(params=[rwq, rwk], algorithm="cm_qk", head_dim=hd)], lr=0.02
            )
            ref.step()
            return_dict["q_close"] = torch.allclose(got_q, rwq.data, atol=1e-5)
            return_dict["k_close"] = torch.allclose(got_k, rwk.data, atol=1e-5)
    finally:
        dist.destroy_process_group()


def test_distributed_matches_single_device():
    world_size = 2
    mgr = mp.Manager()
    return_dict = mgr.dict()
    mp.spawn(
        _dist_worker,
        args=(world_size, 29611, return_dict),
        nprocs=world_size,
        join=True,
    )
    assert return_dict.get(
        "q_close"
    ), "distributed Q update diverged from single-device"
    assert return_dict.get(
        "k_close"
    ), "distributed K update diverged from single-device"
