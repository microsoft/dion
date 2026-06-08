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
    _unwrap_subclass,
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
        # group_size == 1 (H_q == H_kv): the batched MHA path must equal looping
        # each head through an independent single-head qk_delta (no cross-head
        # leakage via the grouping/aggregation/expansion machinery).
        d, hd, H = 12, 4, 3
        torch.manual_seed(4)
        WQ, WK = torch.randn(d, H * hd), torch.randn(d, H * hd)
        GQ, GK = torch.randn(d, H * hd), torch.randn(d, H * hd)
        dQ, dK = qk_delta(WQ, WK, GQ, GK, hd, damping=1e-2)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            dQ_h, dK_h = qk_delta(
                WQ[:, sl], WK[:, sl], GQ[:, sl], GK[:, sl], hd, damping=1e-2
            )
            assert torch.allclose(dQ[:, sl], dQ_h, atol=1e-3)
            assert torch.allclose(dK[:, sl], dK_h, atol=1e-3)

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

    def test_unwrap_subclass(self):
        # Plain tensors pass through; a wrapper subclass unwraps to its master.
        plain = torch.randn(4, 4)
        assert _unwrap_subclass(plain) is plain

        class _FakeWeightWrapper:
            def __init__(self, data):
                self._data = data

            def __tensor_flatten__(self):
                return ["_data"], None

        master = torch.randn(4, 4)
        assert _unwrap_subclass(_FakeWeightWrapper(master)) is master


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

    def test_ov_budget_uses_query_head_count(self):
        # d_model deliberately != H_q * head_dim so the O head count must be read
        # from o_proj's in_features (dim 1), not its out_features (= d_model).
        # The shared V factor takes 1/group_size of the budget; group_size = H_q/H_kv.
        d_model, hd, Hq, Hkv, lr = 24, 4, 4, 2, 0.02
        group_size = Hq // Hkv  # == 2
        _, _, wv, wo = _make_attn(d_model, hd, Hq, Hkv)
        opt = CompositionalMuon(
            [dict(params=[wv, wo], algorithm="cm_ov", head_dim=hd)], lr=lr
        )
        _set_grads([wv, wo])
        wv_before, gv = wv.detach().clone(), wv.grad.clone()
        # Reference V direction via the same dtype path the optimizer uses
        # (bf16 momentum from zero -> math convention -> ov_delta).
        Wv = wv_before.mT.float()
        Wo = wo.detach().mT.float()
        Gv = gv.to(torch.bfloat16).mT.float()
        Go = wo.grad.to(torch.bfloat16).mT.float()
        delta_V, _ = ov_delta(Wv, Wo, Gv, Go, hd, damping=1e-2)
        update_V = delta_V.mT.to(torch.bfloat16)
        opt.step()
        applied = (wv_before - wv.data).float()
        expected_correct = (update_V * (lr * 0.5 / group_size)).float()
        expected_buggy = (update_V * (lr * 0.5 / (group_size + 1))).float()
        assert torch.allclose(applied, expected_correct, atol=1e-6)
        assert not torch.allclose(applied, expected_buggy, atol=1e-6)

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

    def test_normuon_fallback(self):
        mlp = torch.nn.Parameter(torch.randn(32, 16))
        opt = CompositionalMuon(
            [dict(params=[mlp], algorithm="normuon", num_heads=None)],
            lr=0.02,
            muon_beta2=0.9,
        )
        before = mlp.detach().clone()
        mlp.grad = torch.randn_like(mlp)
        opt.step()
        assert not torch.equal(mlp.data, before)
        assert torch.isfinite(mlp.data).all()
        # NorMuon keeps a per-neuron variance buffer; vanilla Muon does not.
        assert "variance_neuron" in opt.state[mlp]

    def test_normuon_fallback_per_head(self):
        attn = torch.nn.Parameter(torch.randn(8 * 4, 16))
        opt = CompositionalMuon(
            [dict(params=[attn], algorithm="normuon", num_heads=8)],
            lr=0.02,
            muon_beta2=0.9,
        )
        attn.grad = torch.randn_like(attn)
        opt.step()
        assert torch.isfinite(attn.data).all()


# ---------------------------------------------------------------------------
# Distributed gather / re-shard path (2-rank gloo, no GPU needed)
# ---------------------------------------------------------------------------


def _dist_worker(rank, world_size, port, return_dict):
    import dion.compositional_muon as cm_mod

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        torch._dynamo.config.cache_size_limit = 128
        mesh = init_device_mesh("cpu", (world_size,))
        d, hd, Hq, Hkv = 16, 4, 4, 2

        torch.manual_seed(21)
        wq_full, wk_full = torch.randn(Hq * hd, d), torch.randn(Hkv * hd, d)
        wv_full, wo_full = torch.randn(Hkv * hd, d), torch.randn(d, Hq * hd)
        torch.manual_seed(22)
        gq_full, gk_full = torch.randn(Hq * hd, d), torch.randn(Hkv * hd, d)
        gv_full, go_full = torch.randn(Hkv * hd, d), torch.randn(d, Hq * hd)

        def shard(t):
            p = torch.nn.Parameter(distribute_tensor(t, mesh, [Shard(0)]))
            return p

        wq, wk = shard(wq_full), shard(wk_full)
        wv, wo = shard(wv_full), shard(
            wo_full
        )  # q/k/v shard on heads; o shards on hidden
        for p, g in ((wq, gq_full), (wk, gk_full), (wv, gv_full), (wo, go_full)):
            p.grad = distribute_tensor(g, mesh, [Shard(0)])

        opt = CompositionalMuon(
            [
                dict(params=[wq, wk], algorithm="cm_qk", head_dim=hd),
                dict(params=[wv, wo], algorithm="cm_ov", head_dim=hd),
            ],
            lr=0.02,
        )

        # Count gathers: the head-sharded QK pair must take the no-comm local path
        # (zero _full calls for q/k); only the hidden-sharded O factor gathers.
        orig_full = cm_mod._full
        calls = {"n": 0}

        def counting_full(x):
            calls["n"] += 1
            return orig_full(x)

        cm_mod._full = counting_full
        try:
            opt.step()
        finally:
            cm_mod._full = orig_full

        got = {
            n: p.data.full_tensor()
            for n, p in (("q", wq), ("k", wk), ("v", wv), ("o", wo))
        }

        if rank == 0:
            ref_p = {
                n: torch.nn.Parameter(t.clone())
                for n, t in (
                    ("q", wq_full),
                    ("k", wk_full),
                    ("v", wv_full),
                    ("o", wo_full),
                )
            }
            for n, g in (
                ("q", gq_full),
                ("k", gk_full),
                ("v", gv_full),
                ("o", go_full),
            ):
                ref_p[n].grad = g.clone()
            ref = CompositionalMuon(
                [
                    dict(
                        params=[ref_p["q"], ref_p["k"]], algorithm="cm_qk", head_dim=hd
                    ),
                    dict(
                        params=[ref_p["v"], ref_p["o"]], algorithm="cm_ov", head_dim=hd
                    ),
                ],
                lr=0.02,
            )
            ref.step()
            for n in ("q", "k", "v", "o"):
                return_dict[f"{n}_close"] = torch.allclose(
                    got[n], ref_p[n].data, atol=1e-5
                )
            # QK (4 factors over 2 pairs) gathered nothing; OV gathered V and O.
            return_dict["qk_full_calls"] = calls["n"]
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
    for n in ("q", "k", "v", "o"):
        assert return_dict.get(
            f"{n}_close"
        ), f"distributed {n} update diverged from single-device"
    # The head-sharded QK pair takes the no-comm local path (0 gathers); only the
    # OV pair gathers (its 2 weights + 2 momentum buffers = 4 _full calls).
    assert return_dict.get("qk_full_calls") == 4, (
        f"expected only OV to gather (4 _full calls), got {return_dict.get('qk_full_calls')} "
        "- the QK no-comm local path may not have been taken"
    )
