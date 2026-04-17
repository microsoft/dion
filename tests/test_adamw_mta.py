"""Tests for the Triton multi-tensor-apply AdamW kernel (``adamw_update_foreach``).

Covers:
- Math correctness vs ``torch.optim.AdamW(fused=True)`` (ground truth).
- CWD branch vs a reference foreach implementation.
- All supported dtypes (fp32, fp16, bf16) and non-zero ``step`` (bias correction).
- Heterogeneous shapes (1D / 2D / 3D / different numels in the same call).
- Empty list is a no-op.
- MTA metadata cache: hits on repeat calls with the same tensor objects,
  rebuilds on changed tensor objects.
- Integration: Dion ``Muon`` with ``algorithm="adamw"`` scalar group runs
  through the new kernel end-to-end.
"""

import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")


def _reference_adamw(X, G, M, V, lr, beta1, beta2, weight_decay, step, eps, cautious_wd):
    """Reference implementation used to validate the CWD branch.

    Non-CWD branch is validated against ``torch.optim.AdamW(fused=True)`` in
    ``test_vs_torch_fused``; this reference exists solely to give CWD a
    spec-equivalent baseline.
    """
    N = len(X)
    for i in range(N):
        M[i].mul_(beta1).add_(G[i], alpha=1 - beta1)
        V[i].mul_(beta2).addcmul_(G[i], G[i], value=1 - beta2)

    bc1 = 1 - beta1 ** step
    bc2_sqrt = (1 - beta2 ** step) ** 0.5
    step_size = lr / bc1
    for i in range(N):
        denom = V[i].to(torch.float32).sqrt().div_(bc2_sqrt).add_(eps)
        update = step_size * M[i].to(torch.float32) / denom
        if cautious_wd:
            mask = (X[i].to(torch.float32) * update >= 0).to(torch.float32)
            X[i].copy_((X[i].to(torch.float32)
                        - lr * weight_decay * X[i].to(torch.float32) * mask
                        - update).to(X[i].dtype))
        else:
            X[i].copy_((X[i].to(torch.float32) * (1 - lr * weight_decay)
                        - update).to(X[i].dtype))


def _make(shapes, dtype, seed=0):
    torch.manual_seed(seed)
    X = [torch.randn(s, device=DEVICE, dtype=dtype) for s in shapes]
    G = [torch.randn(s, device=DEVICE, dtype=dtype) for s in shapes]
    M = [torch.zeros_like(x) for x in X]
    V = [torch.zeros_like(x) for x in X]
    return X, G, M, V


def _clone(X, G, M, V):
    return ([t.clone() for t in X], [t.clone() for t in G],
            [t.clone() for t in M], [t.clone() for t in V])


def _tol(dtype):
    if dtype == torch.float32:
        return dict(atol=1e-5, rtol=1e-5)
    # bf16 / fp16 — the kernel promotes to fp32 internally, but bf16 stores
    # only 7 mantissa bits, so the final cast loses precision.
    return dict(atol=5e-3, rtol=5e-3)


class TestMathVsForeachReference:
    """Validate the Triton kernel against a pure-torch foreach reference.

    This covers both the non-CWD and CWD branches and the bias-correction
    math for ``step > 1``.
    """

    # fp16 is intentionally omitted: this reference does moment updates in the
    # storage dtype, so ``grad * grad`` overflows fp16's ~65k range for
    # outlier values. The Triton kernel promotes to fp32 internally and is
    # *more* numerically stable than the reference in fp16. fp16 correctness
    # is covered instead by ``test_fp16_no_overflow`` below.
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("cautious_wd", [False, True])
    @pytest.mark.parametrize("step", [1, 5])
    def test_matches_reference(self, dtype, cautious_wd, step):
        from dion.scalar_opts import adamw_update_foreach
        shapes = [(16, 32), (8, 8), (13,), (4, 4, 4), (1024,)]
        X, G, M, V = _make(shapes, dtype)

        # Warm M/V up to ``step - 1`` via the reference so step `step` exercises
        # non-trivial bias correction and non-zero moments.
        for s in range(1, step):
            _reference_adamw(X, G, M, V, lr=1e-3, beta1=0.9, beta2=0.999,
                             weight_decay=0.01, step=s, eps=1e-8,
                             cautious_wd=cautious_wd)
            # Refresh grads so the next step sees a different G (mirrors training).
            for g in G:
                g.copy_(torch.randn_like(g))

        X_tri, G_tri, M_tri, V_tri = _clone(X, G, M, V)
        X_ref, G_ref, M_ref, V_ref = _clone(X, G, M, V)

        adamw_update_foreach(
            X_tri, G_tri, M_tri, V_tri,
            lr=torch.tensor(1e-3), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.01),
            step=step, epsilon=1e-8, cautious_wd=cautious_wd,
        )
        _reference_adamw(X_ref, G_ref, M_ref, V_ref, lr=1e-3, beta1=0.9,
                         beta2=0.999, weight_decay=0.01, step=step,
                         eps=1e-8, cautious_wd=cautious_wd)

        tol = _tol(dtype)
        for a, b in zip(X_tri, X_ref):
            assert torch.allclose(a, b, **tol), f"X mismatch: max {(a-b).abs().max()}"
        for a, b in zip(M_tri, M_ref):
            assert torch.allclose(a, b, **tol)
        for a, b in zip(V_tri, V_ref):
            assert torch.allclose(a, b, **tol)


class TestVsTorchFused:
    """Non-CWD path should be numerically equivalent to
    ``torch.optim.AdamW(fused=True)`` (which dispatches ``_fused_adamw_``)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bit_exact_step1(self, dtype):
        from dion.scalar_opts import adamw_update_foreach
        shapes = [(16, 32), (8, 8), (13,), (4, 4, 4), (1024,)]
        X, G, M, V = _make(shapes, dtype)
        X_tri, G_tri, M_tri, V_tri = _clone(X, G, M, V)

        adamw_update_foreach(
            X_tri, G_tri, M_tri, V_tri,
            lr=torch.tensor(1e-3), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.01),
            step=1, epsilon=1e-8, cautious_wd=False,
        )

        X_torch = [x.clone().requires_grad_(True) for x in X]
        for x, g in zip(X_torch, G):
            x.grad = g.clone()
        opt = torch.optim.AdamW(
            X_torch, lr=1e-3, betas=(0.9, 0.999),
            weight_decay=0.01, eps=1e-8, fused=True,
        )
        opt.step()

        # bf16: bit-exact. fp32: within ~1 ulp (the two kernels choose a
        # slightly different fp32 op order in the bias-correction step).
        tol = dict(atol=1e-6, rtol=0) if dtype == torch.float32 else dict(atol=0, rtol=0)
        for a, b in zip(X_tri, X_torch):
            assert torch.allclose(a, b.detach(), **tol), (
                f"maxdiff={(a-b.detach()).abs().max()}"
            )


class TestFp16:
    """fp16 needs its own test because the naive foreach reference would
    overflow on ``grad * grad``. We check two things instead: (a) no
    NaN/Inf in the output, and (b) the result tracks a fp32-promoted
    reference (which is what the Triton kernel computes internally)."""

    def _promoted_reference(self, X, G, M, V, lr, beta1, beta2, wd, step, eps):
        N = len(X)
        for i in range(N):
            g32 = G[i].to(torch.float32)
            m32 = M[i].to(torch.float32) * beta1 + g32 * (1 - beta1)
            v32 = V[i].to(torch.float32) * beta2 + g32 * g32 * (1 - beta2)
            bc1 = 1 - beta1 ** step
            bc2_sqrt = (1 - beta2 ** step) ** 0.5
            denom = v32.sqrt() / bc2_sqrt + eps
            update = (lr / bc1) * m32 / denom
            x32 = X[i].to(torch.float32) * (1 - lr * wd) - update
            X[i].copy_(x32.to(X[i].dtype))
            M[i].copy_(m32.to(M[i].dtype))
            V[i].copy_(v32.to(V[i].dtype))

    def test_fp16_no_overflow(self):
        from dion.scalar_opts import adamw_update_foreach
        shapes = [(16, 32), (13,), (1024,)]
        X, G, M, V = _make(shapes, torch.float16)
        X_tri, G_tri, M_tri, V_tri = _clone(X, G, M, V)
        X_ref, G_ref, M_ref, V_ref = _clone(X, G, M, V)

        adamw_update_foreach(
            X_tri, G_tri, M_tri, V_tri,
            lr=torch.tensor(1e-3), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.01),
            step=1, epsilon=1e-8, cautious_wd=False,
        )
        self._promoted_reference(X_ref, G_ref, M_ref, V_ref,
                                 lr=1e-3, beta1=0.9, beta2=0.999,
                                 wd=0.01, step=1, eps=1e-8)

        for t in X_tri + M_tri + V_tri:
            assert torch.isfinite(t).all(), "fp16 path produced NaN/Inf"
        for a, b in zip(X_tri, X_ref):
            assert torch.allclose(a, b, atol=5e-3, rtol=5e-3)


class TestShapes:
    def test_empty_list_noop(self):
        from dion.scalar_opts import adamw_update_foreach
        # Should not crash or launch anything.
        adamw_update_foreach(
            [], [], [], [],
            lr=torch.tensor(1e-3), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.01),
            step=1, epsilon=1e-8,
        )

    def test_heterogeneous_shapes(self):
        """Mix 1D, 2D, 3D, and an odd-sized tensor that straddles block boundaries."""
        from dion.scalar_opts import adamw_update_foreach
        shapes = [(1,), (5,), (1025,), (7, 3), (2, 3, 5), (1024,)]
        X, G, M, V = _make(shapes, torch.bfloat16)
        X_tri, G_tri, M_tri, V_tri = _clone(X, G, M, V)
        X_ref, G_ref, M_ref, V_ref = _clone(X, G, M, V)

        adamw_update_foreach(
            X_tri, G_tri, M_tri, V_tri,
            lr=torch.tensor(1e-3), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.01),
            step=1, epsilon=1e-8, cautious_wd=False,
        )
        _reference_adamw(X_ref, G_ref, M_ref, V_ref, lr=1e-3, beta1=0.9,
                         beta2=0.999, weight_decay=0.01, step=1,
                         eps=1e-8, cautious_wd=False)
        for a, b in zip(X_tri, X_ref):
            assert torch.allclose(a, b, **_tol(torch.bfloat16))

    def test_single_tensor(self):
        from dion.scalar_opts import adamw_update_foreach
        X, G, M, V = _make([(64, 128)], torch.bfloat16)
        before = X[0].clone()
        adamw_update_foreach(
            X, G, M, V,
            lr=torch.tensor(1e-3), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.01),
            step=1, epsilon=1e-8,
        )
        assert not torch.equal(X[0], before)


class TestCaching:
    """The MTA metadata cache is what eliminates the per-step
    ``cudaStreamSynchronize``. These tests exercise the cache directly."""

    def _call(self, X, G, M, V, cautious_wd=False):
        from dion.scalar_opts import adamw_update_foreach
        adamw_update_foreach(
            X, G, M, V,
            lr=torch.tensor(1e-3), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.01),
            step=1, epsilon=1e-8, cautious_wd=cautious_wd,
        )

    def test_cache_hit_across_steps(self):
        """Repeat calls with the SAME tensor objects must reuse cached
        metadata and pointer tensors (same object identity)."""
        from dion.scalar_opts import _mta_cache
        X, G, M, V = _make([(128,), (256,), (4, 8)], torch.bfloat16)

        self._call(X, G, M, V)
        n_before = len(_mta_cache)
        key = next(iter(_mta_cache))
        buf_before = _mta_cache[key]
        ptrs_x_id = id(buf_before["ptrs_x"])
        numels_id = id(buf_before["numels"])

        for _ in range(3):
            self._call(X, G, M, V)

        assert len(_mta_cache) == n_before, "cache grew on repeated calls"
        buf_after = _mta_cache[key]
        assert id(buf_after["ptrs_x"]) == ptrs_x_id, "ptrs_x tensor rebuilt on hit"
        assert id(buf_after["numels"]) == numels_id, "numels tensor rebuilt on hit"

    def test_cache_miss_on_new_params(self):
        """Calling with a disjoint set of params must add a new cache entry,
        not overwrite the previous one."""
        from dion.scalar_opts import _mta_cache
        X1, G1, M1, V1 = _make([(128,)], torch.bfloat16, seed=1)
        X2, G2, M2, V2 = _make([(256,)], torch.bfloat16, seed=2)

        self._call(X1, G1, M1, V1)
        n1 = len(_mta_cache)
        self._call(X2, G2, M2, V2)
        n2 = len(_mta_cache)
        assert n2 == n1 + 1

        # Original entry must still be usable (tensors still alive).
        self._call(X1, G1, M1, V1)

    def test_fresh_grads_handled(self):
        """Simulate ``zero_grad(set_to_none=True)`` by replacing the grad
        tensors between steps. The cache entry for X/M/V must survive while
        ``ptrs_g`` gets rebuilt."""
        from dion.scalar_opts import _mta_cache
        X, G, M, V = _make([(128,), (256,)], torch.bfloat16)

        self._call(X, G, M, V)
        key = next(
            k for k in _mta_cache
            if k[0] == tuple(t.data_ptr() for t in X)
            and k[1] == tuple(t.data_ptr() for t in M)
        )
        ptrs_x_id = id(_mta_cache[key]["ptrs_x"])

        # Fresh grads → different Python objects with different data_ptrs.
        G_new = [torch.randn_like(g) for g in G]
        self._call(X, G_new, M, V)

        assert id(_mta_cache[key]["ptrs_x"]) == ptrs_x_id, (
            "stable X/M/V cache evicted when only grads changed"
        )

    def test_cache_hit_across_fresh_wrappers(self):
        """``param.data`` and ``DTensor.to_local()`` return a fresh Python
        wrapper each call (different ``id()`` but same underlying storage).
        Cache must still hit -- otherwise it grows unboundedly under FSDP /
        repeated ``.data`` access."""
        from dion.scalar_opts import _mta_cache
        _mta_cache.clear()
        params = [torch.nn.Parameter(torch.randn(128, device=DEVICE, dtype=torch.bfloat16)),
                  torch.nn.Parameter(torch.randn(256, device=DEVICE, dtype=torch.bfloat16))]
        M = [torch.zeros_like(p.data) for p in params]
        V = [torch.zeros_like(p.data) for p in params]
        G = [torch.randn_like(p.data) for p in params]

        # Each call re-accesses ``.data`` -> fresh Python wrapper, stable storage.
        for _ in range(5):
            X_fresh = [p.data for p in params]
            self._call(X_fresh, G, M, V)
        assert len(_mta_cache) == 1, (
            f"cache grew to {len(_mta_cache)} entries across fresh ``.data`` wrappers; "
            "key must be stable across Python wrapper re-creation"
        )


class TestIntegration:
    """End-to-end: Dion's NorMuon with an AdamW scalar group dispatches
    through our kernel and makes progress on a tiny training step."""

    def test_normuon_with_adamw_scalars(self):
        from dion import NorMuon
        torch.manual_seed(42)
        weights = [torch.nn.Parameter(torch.randn(64, 128, device=DEVICE)),
                   torch.nn.Parameter(torch.randn(128, 64, device=DEVICE))]
        biases = [torch.nn.Parameter(torch.randn(64, device=DEVICE)),
                  torch.nn.Parameter(torch.randn(128, device=DEVICE))]
        opt = NorMuon([
            {"params": weights},
            {"params": biases, "algorithm": "adamw"},
        ], lr=0.01)
        before = [p.data.clone() for p in biases]
        for _ in range(3):
            for p in weights + biases:
                p.grad = torch.randn_like(p)
            opt.step()
        for b_now, b_before in zip(biases, before):
            assert not torch.equal(b_now.data, b_before), "scalar params didn't update"

    def test_muon_with_adamw_cautious_wd(self):
        from dion import Muon
        torch.manual_seed(42)
        weights = [torch.nn.Parameter(torch.randn(32, 64, device=DEVICE))]
        biases = [torch.nn.Parameter(torch.randn(32, device=DEVICE))]
        opt = Muon([
            {"params": weights},
            {"params": biases, "algorithm": "adamw", "cautious_wd": True},
        ], lr=0.01)
        for _ in range(3):
            for p in weights + biases:
                p.grad = torch.randn_like(p)
            opt.step()
