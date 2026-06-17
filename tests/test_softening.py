"""
CPU tests for Soft-Muon softening: the ``softening`` blend that interpolates
between the orthogonalized (Schatten-inf) update and the spectrally-normalized
momentum, producing a heavier-tailed finite-Schatten-p update.
"""
import pytest
import torch

from dion import Muon, NorMuon
from dion.megabatch_base import _soften_newton_schulz


def _polar(X, epsilon=None):
    # Exact orthogonalization (all singular values -> 1) via SVD, fp32.
    U, _, Vh = torch.linalg.svd(X.to(torch.float64), full_matrices=False)
    return (U @ Vh).to(X.dtype)


@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (5, 5)])
def test_softening_zero_recovers_base(shape):
    torch.manual_seed(0)
    X = torch.randn(*shape)
    softened = _soften_newton_schulz(_polar, 0.0)
    assert torch.equal(softened(X), _polar(X))


@pytest.mark.parametrize("shape", [(6, 4), (4, 6)])
@pytest.mark.parametrize("s", [0.25, 0.5, 0.75, 1.0])
def test_softened_singular_values_match_closed_form(shape, s):
    torch.manual_seed(1)
    X = torch.randn(*shape, dtype=torch.float64)
    out = _soften_newton_schulz(_polar, s)(X)

    sigma = torch.linalg.svdvals(X)
    fro = torch.linalg.norm(X)
    predicted = torch.sort((1 - s) + s * sigma / fro, descending=True).values
    actual = torch.sort(torch.linalg.svdvals(out), descending=True).values

    assert torch.allclose(actual, predicted, atol=1e-6)


def test_softening_increases_spectral_spread():
    # Heavier tail: the update's max/min singular-value ratio grows with s.
    torch.manual_seed(2)
    X = torch.randn(8, 5, dtype=torch.float64)
    spreads = []
    for s in [0.0, 0.3, 0.6, 1.0]:
        sv = torch.linalg.svdvals(_soften_newton_schulz(_polar, s)(X))
        spreads.append((sv.max() / sv.min()).item())
    assert spreads == sorted(spreads)
    assert spreads[0] == pytest.approx(1.0, abs=1e-9)


def test_softening_preserves_singular_vectors():
    torch.manual_seed(3)
    X = torch.randn(7, 4, dtype=torch.float64)
    out = _soften_newton_schulz(_polar, 0.5)(X)
    Ux, _, Vhx = torch.linalg.svd(X, full_matrices=False)
    Uo, _, Vho = torch.linalg.svd(out, full_matrices=False)
    # Columns of U and rows of Vh align up to sign.
    assert torch.allclose(torch.abs(Ux.T @ Uo), torch.eye(4, dtype=X.dtype), atol=1e-6)
    assert torch.allclose(torch.abs(Vhx @ Vho.T), torch.eye(4, dtype=X.dtype), atol=1e-6)


def test_softening_with_real_newton_schulz_backend():
    # The other tests use exact SVD orthogonalization (_polar). This one drives
    # the wrapper through a real finite-step Newton-Schulz backend (polar_express,
    # which also returns bf16) to catch dtype/shape/finiteness regressions the
    # exact-polar tests cannot, and to confirm the blend and the monotone tail
    # reweighting survive a backend whose singular values are only approximately 1.
    from dion.polar_express import polar_express

    torch.manual_seed(5)
    X = torch.randn(64, 32)
    eps = torch.tensor(1e-7)

    # The wrapper is exactly the documented blend on the real backend output.
    s = 0.5
    ortho = polar_express(X, epsilon=eps)
    expected = torch.lerp(ortho, (X / X.norm()).to(ortho.dtype), s)
    out = _soften_newton_schulz(polar_express, s)(X, epsilon=eps)
    assert out.dtype == ortho.dtype
    assert out.shape == X.shape
    assert torch.equal(out, expected)

    # Output stays finite across the full range, and the update is heavier-tailed
    # at s=1 (full spectral decay) than at s=0 (orthogonalized). Strict per-step
    # monotonicity is only guaranteed for an exact polar factor (tested above with
    # _polar); a finite-step NS map adds its own ~1.3x spread at s=0, so we assert
    # the robust endpoint relation rather than monotonicity over intermediate s.
    spreads = {}
    for s in [0.0, 0.3, 0.6, 1.0]:
        sv = torch.linalg.svdvals(_soften_newton_schulz(polar_express, s)(X, epsilon=eps).float())
        assert torch.isfinite(sv).all()
        spreads[s] = (sv.max() / sv.min()).item()
    assert spreads[1.0] > spreads[0.0]


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_invalid_softening_raises(bad):
    p = [torch.zeros(4, 4, requires_grad=True)]
    with pytest.raises(ValueError, match="softening"):
        NorMuon(p, softening=bad)


def test_optimizer_wires_softening():
    p = [torch.zeros(4, 4, requires_grad=True)]
    assert NorMuon(p, softening=0.0)._softening == 0.0
    opt = NorMuon([torch.zeros(4, 4, requires_grad=True)], softening=0.4)
    assert opt._softening == 0.4


def test_optimizer_step_with_softening_updates_params():
    torch.manual_seed(4)
    w = torch.randn(8, 6, requires_grad=True)
    before = w.detach().clone()
    opt = Muon([w], lr=0.1, newton_schulz_func=_polar, softening=0.5)
    w.grad = torch.randn_like(w)
    opt.step()
    assert torch.isfinite(w).all()
    assert not torch.equal(w.detach(), before)
