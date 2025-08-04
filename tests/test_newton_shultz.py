# tests/test_newton_schulz.py
import pytest
import torch

from dion.newton_schulz_triton import (
    ns_line_1,
    ns_line_2,
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)

# -----------------------------------------------------------------------------#
# General settings
# -----------------------------------------------------------------------------#

# Allow a lot of recompiles in Torch-Triton
torch._dynamo.config.cache_size_limit = 100  # noqa: SLF001

CUDA_AVAILABLE = torch.cuda.is_available()

# -----------------------------------------------------------------------------#
# Helper
# -----------------------------------------------------------------------------#


def _assert_close(result: torch.Tensor, correct: torch.Tensor, *, tol: float = 5e-2):
    """Assert two tensors are close enough for the test to pass."""
    assert (
        result.dtype == correct.dtype
    ), f"dtype mismatch — got {result.dtype}, expected {correct.dtype}"
    assert (
        result.shape == correct.shape
    ), f"shape mismatch — got {result.shape}, expected {correct.shape}"
    assert torch.allclose(
        result, correct, atol=tol, rtol=tol
    ), f"max-abs-diff {torch.abs(result - correct).max().item():.3e} > {tol}"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize("m,n", [(256, 256), (256, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ns_line_1(m: int, n: int, dtype: torch.dtype):
    """ns_line_1 should compute A @ A^T (batched and unbatched)."""
    A = torch.randn(m, n, dtype=dtype, device="cuda")
    _assert_close(ns_line_1(A), A @ A.mT)

    A_batched = torch.randn(4, m, n, dtype=dtype, device="cuda")
    _assert_close(ns_line_1(A_batched), A_batched @ A_batched.mT)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ns_line_2(m: int, dtype: torch.dtype):
    """ns_line_2 should compute alpha(A@A^T) + beta*A for symmetric A."""
    alpha, beta = torch.randn(1).item(), torch.randn(1).item()

    A = torch.randn(m, m, dtype=dtype, device="cuda")
    A = (A + A.mT) / 2  # ensure symmetry
    correct = alpha * (A @ A.mT) + beta * A
    _assert_close(ns_line_2(A, alpha=alpha, beta=beta), correct)

    A_batched = torch.randn(4, m, m, dtype=dtype, device="cuda")
    A_batched = (A_batched + A_batched.mT) / 2
    correct_batched = alpha * (A_batched @ A_batched.mT) + beta * A_batched
    _assert_close(ns_line_2(A_batched, alpha=alpha, beta=beta), correct_batched)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize("m,n", [(256, 256), (256, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_newton_schulz_triton(m: int, n: int, dtype: torch.dtype):
    """Fast Triton implementation should match the reference Newton-Schulz."""
    G = torch.randn(m, n, dtype=dtype, device="cuda")
    _assert_close(newton_schulz_triton(G), zeropower_via_newtonschulz5(G))

    G_batched = torch.randn(4, m, n, dtype=dtype, device="cuda")
    _assert_close(
        newton_schulz_triton(G_batched), zeropower_via_newtonschulz5(G_batched)
    )
