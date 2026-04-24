import torch
from torch import Tensor


# Hybrid Newton-Schulz coefficients from DeepSeek-V4 (Algorithm 1, Eq. 28).
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
# First 8 iterations drive singular values rapidly toward 1; last 2 iterations
# stabilize at 1 via the standard (2, -1.5, 0.5) fixed-point schedule.
_HYBRID_NS_COEFFS = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2


@torch.compile(dynamic=False, fullgraph=True)
def hybrid_newton_schulz(G: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Hybrid Newton-Schulz orthogonalization from DeepSeek-V4.

    Performs 10 iterations in two stages: 8 steps with coefficients
    (3.4445, -4.7750, 2.0315) for rapid convergence, then 2 steps with
    (2.0, -1.5, 0.5) to stabilize singular values at 1. Input is normalized
    by its Frobenius norm (paper verbatim; no safety cushion).

    Paper: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
    Section 2.4, Algorithm 1 and Equation 28.

    Signature matches zeropower_via_newtonschulz5: func(input, epsilon) -> Tensor.
    """
    assert G.ndim >= 2
    X = G.bfloat16()

    is_tall = G.size(-2) > G.size(-1)

    # Frobenius normalization: ensures ||X||_2 <= 1 since ||X||_2 <= ||X||_F.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    if is_tall:
        # Tall: use X.mT @ X (small cols x cols) + right-multiply.
        for a, b, c in _HYBRID_NS_COEFFS:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        # Wide: use X @ X.mT (small rows x rows) + left-multiply.
        for a, b, c in _HYBRID_NS_COEFFS:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X
