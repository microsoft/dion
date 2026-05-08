"""Tests that Dion2 zero_mode produces results identical to the default
gather/scatter path.

Zero-mode replaces the compact submatrix (gather → ortho → scatter) with a
full-sized zero-masked matrix. Because zero rows/columns contribute nothing to
Newton-Schulz (they stay zero through the iteration and don't affect the Gram
matrix of the non-zero entries), the two paths should produce identical
parameter updates and momentum buffers.

Momentum buffers are bitwise identical. Parameters may differ at the level of
floating-point rounding (~1e-6) because Newton-Schulz matrix multiplications
operate on different shapes (compact k×n vs full m×n with zero rows), which
changes the summation order in matmul.
"""

import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()

torch._dynamo.config.cache_size_limit = 64

# Tolerance for parameter comparison (Newton-Schulz floating-point rounding)
ATOL = 1e-4
RTOL = 1e-4


def _make_params(shapes, device=DEVICE):
    """Create parameters with deterministic values."""
    torch.manual_seed(42)
    return [torch.nn.Parameter(torch.randn(s, device=device)) for s in shapes]


def _run_steps(optimizer_cls, params, opt_kwargs, n_steps=3):
    """Run n optimizer steps with deterministic gradients."""
    opt = optimizer_cls(params, **opt_kwargs)
    for step in range(n_steps):
        torch.manual_seed(100 + step)
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    return opt


def _get_momentum(opt):
    """Extract momentum buffers from optimizer state."""
    return [opt.state[p]["momentum"].clone() for p in opt.param_groups[0]["params"]]


def _assert_equivalent(opt_default, opt_zero, p_default, p_zero):
    """Assert parameters are close and momentum is bitwise identical."""
    for pd, pz in zip(p_default, p_zero):
        assert torch.allclose(pd.data, pz.data, atol=ATOL, rtol=RTOL), (
            f"Parameters differ beyond tolerance: "
            f"max diff = {(pd.data - pz.data).abs().max().item():.2e}"
        )
    for md, mz in zip(_get_momentum(opt_default), _get_momentum(opt_zero)):
        assert torch.equal(md, mz), "Momentum buffers differ"


# ---------------------------------------------------------------------------
# Full optimizer comparison: default vs zero_mode
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestZeroModeEquivalence:
    """Verify that zero_mode=True produces identical results to the default
    gather/scatter path across a range of configurations."""

    @pytest.mark.parametrize("shapes, fraction, ef_decay, n_steps", [
        ([(32, 128)], 0.25, 0.95, 3),           # wide, low fraction
        ([(32, 128)], 0.5, 0.95, 3),             # wide, high fraction
        ([(128, 32)], 0.25, 0.95, 3),            # tall, low fraction
        ([(128, 32)], 0.5, 0.95, 3),             # tall, high fraction
        ([(64, 64)], 0.5, 0.95, 3),              # square
        ([(4, 32, 128)], 0.5, 0.95, 3),          # 3D (batch of matrices)
        ([(64, 128)] * 4, 0.25, 0.9, 3),         # megabatch (multiple same-shape)
        ([(64, 128)], 0.5, 0.0, 3),              # ef_decay=0 (full decay)
        ([(64, 128)], 0.25, 0.95, 10),           # many steps
    ])
    def test_equivalence(self, shapes, fraction, ef_decay, n_steps):
        from dion import Dion2
        kwargs = dict(lr=0.01, fraction=fraction, ef_decay=ef_decay)

        p_default = _make_params(shapes)
        opt_default = _run_steps(Dion2, p_default, {**kwargs, "zero_mode": False}, n_steps)

        p_zero = _make_params(shapes)
        opt_zero = _run_steps(Dion2, p_zero, {**kwargs, "zero_mode": True}, n_steps)

        _assert_equivalent(opt_default, opt_zero, p_default, p_zero)
