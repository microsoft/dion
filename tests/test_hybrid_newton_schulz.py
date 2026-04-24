"""Tests for hybrid (8 Muon + 2 classical) Newton-Schulz orthogonalization."""

import pytest
import torch

from dion.hybrid_newton_schulz import hybrid_newton_schulz
from dion.polar_express import polar_express
from dion.newton_schulz_triton import zeropower_via_newtonschulz5


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHAPES = [
    (64, 128),    # wide
    (128, 64),    # tall
    (128, 128),   # square
]
BATCH_SHAPES = [
    (4, 64, 128),   # batched wide
    (4, 128, 64),   # batched tall
]
# Hybrid NS runs 10 bf16 iterations; the last 2 steps use the canonical
# (2, -1.5, 0.5) schedule which pins singular values at 1, so orthogonality
# residuals are comparable to NS5 rather than PE (looser 0.15 used for PE).
ORTHO_ATOL = 0.05


@pytest.mark.parametrize("shape", SHAPES + BATCH_SHAPES)
def test_hybrid_newton_schulz_produces_orthogonal_output(shape):
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)
    X = hybrid_newton_schulz(G)

    if shape[-2] <= shape[-1]:
        product = X.float() @ X.float().mT
    else:
        product = X.float().mT @ X.float()

    eye = torch.eye(product.shape[-1], device=DEVICE).expand_as(product)
    err = (product - eye).abs().max().item()
    assert err < ORTHO_ATOL, (
        f"Not orthogonal for shape {shape}: max error {err:.4f}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_hybrid_newton_schulz_vs_references(shape):
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)

    X_hybrid = hybrid_newton_schulz(G)
    X_pe = polar_express(G)
    X_ns = zeropower_via_newtonschulz5(G)

    assert X_hybrid.shape == X_pe.shape == X_ns.shape

    for label, X, atol in [
        ("hybrid", X_hybrid, ORTHO_ATOL),
        ("PE", X_pe, 0.15),
        ("NS", X_ns, 0.15),
    ]:
        if shape[-2] <= shape[-1]:
            product = X.float() @ X.float().mT
        else:
            product = X.float().mT @ X.float()
        eye = torch.eye(product.shape[-1], device=DEVICE)
        err = (product - eye).abs().max().item()
        assert err < atol, f"{label} not orthogonal for shape {shape}: {err:.4f}"
