"""Tests for GramNewtonSchulz orthogonalization."""

import pytest
import torch

from gram_newton_schulz import GramNewtonSchulz


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
ORTHO_ATOL = 0.15

# Match the kwargs used in production (megabatch_base.py)
GNS_KWARGS = dict(
    gram_newton_schulz_reset_iterations=[2],
    compile_kwargs=dict(fullgraph=True, mode="default"),
)


def orthogonality_error(X, shape):
    """Compute max |X X^T - I| or |X^T X - I| depending on aspect ratio."""
    if shape[-2] <= shape[-1]:
        product = X.float() @ X.float().mT
    else:
        product = X.float().mT @ X.float()
    eye = torch.eye(product.shape[-1], device=DEVICE).expand_as(product)
    return (product - eye).abs().max().item()


@pytest.fixture(params=[False, True], ids=["no_gram", "gram"])
def use_gram(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["pure", "kernels"])
def ns_use_kernels(request):
    return request.param


@pytest.fixture
def gns(use_gram, ns_use_kernels):
    return GramNewtonSchulz(
        ns_use_kernels=ns_use_kernels,
        use_gram_newton_schulz=use_gram,
        **GNS_KWARGS,
    )


@pytest.mark.parametrize("shape", SHAPES + BATCH_SHAPES)
def test_produces_orthogonal_output(gns, shape):
    """Check that GNS output is approximately orthogonal."""
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)
    X = gns(G)
    err = orthogonality_error(X, shape)
    assert err < ORTHO_ATOL, (
        f"Not orthogonal for shape {shape}: max error {err:.4f}"
    )


@pytest.mark.parametrize("shape", SHAPES + BATCH_SHAPES)
@pytest.mark.parametrize("use_gram", [False, True], ids=["no_gram", "gram"])
def test_pure_and_kernels_match(shape, use_gram):
    """Check that pure and kernel implementations produce the same output."""
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)

    gns_pure = GramNewtonSchulz(ns_use_kernels=False, use_gram_newton_schulz=use_gram, **GNS_KWARGS)
    gns_kern = GramNewtonSchulz(ns_use_kernels=True, use_gram_newton_schulz=use_gram, **GNS_KWARGS)

    X_pure = gns_pure(G)
    X_kern = gns_kern(G)

    assert X_pure.shape == X_kern.shape
    torch.testing.assert_close(X_pure.float(), X_kern.float(), atol=0.01, rtol=0)


@pytest.mark.parametrize("shape", SHAPES)
def test_output_shape_matches_input(gns, shape):
    """Output should have the same shape as input."""
    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)
    X = gns(G)
    assert X.shape == G.shape


def _get_reference_fns():
    """Return named reference orthogonalization functions for comparison."""
    from dion.muon import zeropower_via_newtonschulz5
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmark"))
    from polar_express import polar_express

    return {
        "newton_schulz": zeropower_via_newtonschulz5,
        "polar_express": polar_express,
    }


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("ref_name", ["newton_schulz", "polar_express"])
def test_gns_vs_reference(gns, shape, ref_name):
    """GNS output should roughly match legacy reference implementations."""
    ref_fn = _get_reference_fns()[ref_name]

    torch.manual_seed(42)
    G = torch.randn(shape, device=DEVICE)

    X_gns = gns(G)
    X_ref = ref_fn(G)

    assert X_gns.shape == X_ref.shape
    torch.testing.assert_close(X_gns.float(), X_ref.float(), atol=0.15, rtol=0)
