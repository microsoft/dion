"""Tests for polar express orthogonalization functions."""

import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from polar_express import polar_express, polar_express_triton
from newton_schulz_triton import zeropower_via_newtonschulz5


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
# PE uses 5 bf16 iterations; orthogonality error is ~0.13 for non-square,
# ~0.08 for square. NS gets ~0.02-0.04 with the same iteration count.
# PE is optimized for training loss convergence, not per-step orthogonality.
ORTHO_ATOL = 0.15


@pytest.mark.parametrize("shape", SHAPES + BATCH_SHAPES)
def test_polar_express_produces_orthogonal_output(shape):
    """Check that polar_express output is approximately orthogonal."""
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)
    X = polar_express(G)

    if shape[-2] <= shape[-1]:
        product = X.float() @ X.float().mT
    else:
        product = X.float().mT @ X.float()

    eye = torch.eye(product.shape[-1], device=DEVICE).expand_as(product)
    err = (product - eye).abs().max().item()
    assert err < ORTHO_ATOL, (
        f"Not orthogonal for shape {shape}: max error {err:.4f}"
    )


@pytest.mark.parametrize("shape", SHAPES + BATCH_SHAPES)
def test_polar_express_triton_produces_orthogonal_output(shape):
    """Check that polar_express_triton output is approximately orthogonal."""
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)
    X = polar_express_triton(G)

    if shape[-2] <= shape[-1]:
        product = X.float() @ X.float().mT
    else:
        product = X.float().mT @ X.float()

    eye = torch.eye(product.shape[-1], device=DEVICE).expand_as(product)
    err = (product - eye).abs().max().item()
    assert err < ORTHO_ATOL, (
        f"Not orthogonal for shape {shape}: max error {err:.4f}"
    )


@pytest.mark.parametrize("shape", SHAPES + BATCH_SHAPES)
def test_pure_and_triton_match(shape):
    """Check that pure and triton implementations produce the same output."""
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)

    X_pure = polar_express(G)
    X_triton = polar_express_triton(G)

    assert X_pure.shape == X_triton.shape
    # bf16 accumulation differences between torch.compile and triton kernels
    max_diff = (X_pure.float() - X_triton.float()).abs().max().item()
    assert max_diff < 0.03, (
        f"Pure vs triton mismatch for shape {shape}: max_diff = {max_diff:.4f}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_polar_express_vs_newton_schulz(shape):
    """Both PE and NS should produce approximately orthogonal outputs."""
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)

    X_pe = polar_express(G)
    X_ns = zeropower_via_newtonschulz5(G)

    assert X_pe.shape == X_ns.shape

    for label, X, atol in [("PE", X_pe, ORTHO_ATOL), ("NS", X_ns, ORTHO_ATOL)]:
        if shape[-2] <= shape[-1]:
            product = X.float() @ X.float().mT
        else:
            product = X.float().mT @ X.float()
        eye = torch.eye(product.shape[-1], device=DEVICE)
        err = (product - eye).abs().max().item()
        assert err < atol, f"{label} not orthogonal for shape {shape}: {err:.4f}"
