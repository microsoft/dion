import torch
from torch import Tensor


# Two-stage Newton-Schulz schedule.
# Stage 1 (8 iterations): Keller Jordan's Muon quintic, tuned to inflate
#   small singular values aggressively (https://kellerjordan.github.io/posts/muon/).
#   p(x) = 3.4445 x - 4.7750 x^3 + 2.0315 x^5 does not fix at 1 — iterates
#   oscillate around 1 with slowly decaying amplitude.
# Stage 2 (2 iterations): classical quintic p(x) = 2x - 1.5 x^3 + 0.5 x^5,
#   which has p(1) = 1 and p'(1) = 0 (cubic convergence), pinning singular
#   values at 1 once they are in the basin of attraction.
_HYBRID_NS_COEFFS = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2


@torch.compile(dynamic=False, fullgraph=True)
def hybrid_newton_schulz(G: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Two-stage Newton-Schulz orthogonalization.

    Performs 10 iterations: 8 with Keller Jordan's Muon coefficients
    (3.4445, -4.7750, 2.0315) to inflate small singular values quickly,
    then 2 with the classical quintic (2, -1.5, 0.5) to pin them at 1.
    Input is scaled by its Frobenius norm (||X||_2 <= ||X||_F), so the
    starting spectral norm is at most 1 with no safety cushion.

    Refs:
      - Muon optimizer: https://kellerjordan.github.io/posts/muon/
      - Classical Newton-Schulz iteration for the matrix sign function.

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
