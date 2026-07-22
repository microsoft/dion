"""Tests for the RMNP optimizer (arXiv:2603.20527).

The tests in this file split into two groups:

* CPU-runnable checks of the ``row_normalize`` preconditioner primitive: the
  per-row (input-dimension) ℓ₂ normalization, its scale invariance, and the
  paper's central claim that orthogonalization and row-wise ℓ₂ normalization
  coincide when the update's Gram matrix is (block-)diagonally dominant.
* CUDA-gated end-to-end checks that RMNP takes steps and decreases a toy loss.

The broader end-to-end / megabatch / num_heads / split_sizes coverage lives
alongside the other optimizers in ``test_optimizers.py``.
"""

import pytest
import torch

from dion.rmnp import row_normalize

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


def _orthonormal_rows(m, n, seed=0, dtype=torch.float64):
    """Return an ``(m, n)`` matrix (m <= n) whose rows are orthonormal (Q Qᵀ = I)."""
    assert m <= n
    torch.manual_seed(seed)
    a = torch.randn(n, m, dtype=dtype)
    q = torch.linalg.qr(a)[0]  # (n, m), orthonormal columns
    return q.mT.contiguous()  # (m, n), orthonormal rows


# ---------------------------------------------------------------------------
# row_normalize primitive
# ---------------------------------------------------------------------------

class TestRowNormalize:
    def test_preserves_shape_2d_and_3d(self):
        for shape in [(8, 16), (4, 8, 16)]:
            x = torch.randn(*shape)
            assert row_normalize(x).shape == x.shape

    def test_rows_become_unit_norm(self):
        """Each row (over d_in = last axis) has unit ℓ₂ norm after normalization."""
        x = torch.randn(16, 32)
        y = row_normalize(x, epsilon=0.0)
        torch.testing.assert_close(
            y.norm(dim=-1), torch.ones(16), atol=1e-5, rtol=1e-5
        )

    def test_matches_manual_formula(self):
        """row_normalize(X)[i] == X[i] / (‖X[i]‖ + eps), i.e. (diag(X Xᵀ))^{-1/2} X."""
        x = torch.randn(12, 20)
        eps = 1e-8
        expected = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
        torch.testing.assert_close(row_normalize(x, epsilon=eps), expected)

    def test_normalizes_input_dimension_not_output(self):
        """The 'row' axis is d_in (last), so per-column ℓ₂ norms are NOT unit."""
        x = torch.randn(8, 32)
        y = row_normalize(x, epsilon=0.0)
        # Rows (d_in) are unit-norm; columns (d_out) are generally not.
        assert torch.allclose(y.norm(dim=-1), torch.ones(8), atol=1e-5)
        assert not torch.allclose(y.norm(dim=-2), torch.ones(32), atol=1e-2)

    def test_scale_invariant(self):
        """row_normalize(c·X) == row_normalize(X) for c > 0.

        This is why RMNP can reuse Muon's momentum update ``M ← μM + G`` in place
        of the paper's EMA ``V ← βV + (1−β)G``: the two differ only by the
        constant factor ``1−β``, which cancels under row normalization.
        """
        x = torch.randn(10, 24)
        base = row_normalize(x, epsilon=0.0)
        for c in (0.1, 2.0, 100.0):
            torch.testing.assert_close(row_normalize(c * x, epsilon=0.0), base)

    def test_zero_row_is_finite(self):
        """A zero row must not produce NaN/Inf (epsilon guards the division)."""
        x = torch.randn(6, 16)
        x[2] = 0.0
        y = row_normalize(x, epsilon=1e-8)
        assert torch.isfinite(y).all()
        torch.testing.assert_close(y[2], torch.zeros(16))


# ---------------------------------------------------------------------------
# Asymptotic equivalence to orthogonalization (the paper's core claim)
# ---------------------------------------------------------------------------

class TestOrthogonalizationEquivalence:
    """When the update's Gram matrix ``V Vᵀ`` is diagonal (orthogonal rows),
    row-wise ℓ₂ normalization equals full orthogonalization exactly, because
    Muon's ``(V Vᵀ)^{-1/2}`` reduces to RMNP's ``(diag(V Vᵀ))^{-1/2}``. The
    RMNP paper argues this holds asymptotically for Transformers, where the
    layerwise Hessian (hence ``V Vᵀ``) is empirically block-diagonally dominant.
    """

    def test_equals_polar_factor_on_diagonal_gram(self):
        # V = diag(d) @ Q with Q Qᵀ = I, so V Vᵀ = diag(d²) is diagonal.
        m, n = 6, 24
        q = _orthonormal_rows(m, n, seed=1)
        d = torch.rand(m, dtype=torch.float64) + 0.5
        v = d[:, None] * q

        rn = row_normalize(v, epsilon=0.0)

        # Reference orthogonalization = polar factor U @ Vh (all singular
        # values set to 1) that Newton-Schulz approximates.
        u, _, vh = torch.linalg.svd(v, full_matrices=False)
        polar = u @ vh

        torch.testing.assert_close(rn, polar, atol=1e-10, rtol=1e-10)

    def test_differs_when_gram_not_diagonal(self):
        """Contrast: with correlated (non-orthogonal) rows the two diverge, so
        the equivalence is specific to the diagonal-Gram regime, not trivial."""
        torch.manual_seed(2)
        m, n = 6, 24
        x = torch.randn(m, n, dtype=torch.float64)
        x[1] = x[0] + 0.01 * x[1]  # make rows 0 and 1 nearly collinear

        rn = row_normalize(x, epsilon=0.0)
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        polar = u @ vh

        assert (rn - polar).abs().max().item() > 1e-2

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
    def test_matches_newton_schulz_kernel_on_diagonal_gram(self):
        """row_normalize agrees with the shipped Newton-Schulz routine on a
        diagonal-Gram matrix, within the tolerance of NS's bf16 iteration."""
        from dion.newton_schulz_triton import zeropower_via_newtonschulz5

        m, n = 8, 64
        q = _orthonormal_rows(m, n, seed=3).float().cuda()
        d = (torch.rand(m, device="cuda") + 0.5)
        v = d[:, None] * q

        rn = row_normalize(v, epsilon=0.0).float()
        ns = zeropower_via_newtonschulz5(v).float()
        # NS runs internally in bf16 and only approximately reaches the polar
        # factor; empirically the gap is ~0.02, matching test_newton_shulz.py.
        assert (rn - ns).abs().max().item() < 0.05


# ---------------------------------------------------------------------------
# End-to-end optimization (CUDA-gated, like the other optimizers)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestRMNPEndToEnd:
    def test_applied_update_has_unit_norm_rows(self):
        """On the first step (zero momentum, no weight decay, adjust_lr=None) the
        applied update is exactly ``lr · row_normalize(G)``, so ``(W₀ − W₁)/lr``
        has unit-norm rows -- the defining property of the RMNP update."""
        from dion import RMNP

        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(16, 48, device="cuda"))
        before = w.data.clone()
        lr = 0.05
        opt = RMNP([w], lr=lr, weight_decay=0.0, adjust_lr=None)
        torch.manual_seed(1)
        w.grad = torch.randn_like(w)
        opt.step()

        update = (before - w.data) / lr
        torch.testing.assert_close(
            update.norm(dim=-1),
            torch.ones(16, device="cuda"),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_loss_decreases_on_toy_problem(self):
        from dion import RMNP

        torch.manual_seed(3)
        a = torch.randn(64, 32, device="cuda")
        target = torch.randn(64, 16, device="cuda")
        w = torch.nn.Parameter(torch.randn(32, 16, device="cuda"))
        opt = RMNP([w], lr=0.02, weight_decay=0.0)

        first, last = None, None
        for i in range(60):
            loss = ((a @ w - target) ** 2).mean()
            if i == 0:
                first = loss.item()
            last = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
