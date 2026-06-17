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


@pytest.mark.parametrize("bad", [-0.1, 1.5])
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
