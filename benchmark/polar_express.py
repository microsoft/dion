import torch
from torch import Tensor

from newton_schulz_triton import ns_line_1, ns_line_2

# Polar Express coefficients (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
# Matches the battle-tested KellerJordan/karpathy implementations.
# Safety factor of 1.02 is baked into all but the last polynomial.
_POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Polar Express orthogonalization (pure PyTorch, torch.compile'd).

    Polar Express: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

    Signature matches zeropower_via_newtonschulz5: func(input, epsilon) -> Tensor.
    """
    assert G.ndim >= 2
    X = G.bfloat16()

    is_tall = G.size(-2) > G.size(-1)

    # Ensure spectral norm is at most 1, with safety factor matching coefficients
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + epsilon)

    if is_tall:
        # Tall: use X.mT @ X (small cols x cols) + right-multiply
        for a, b, c in _POLAR_EXPRESS_COEFFS:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        # Wide: use X @ X.mT (small rows x rows) + left-multiply
        for a, b, c in _POLAR_EXPRESS_COEFFS:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X


@torch.compile(dynamic=False, fullgraph=True)
def polar_express_triton(G: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Polar Express orthogonalization using Triton kernels that exploit
    the symmetry of A = X @ X.mT and B = c*(A@A) + b*A, computing only
    half the blocks and mirroring across the diagonal.

    Signature matches zeropower_via_newtonschulz5: func(input, epsilon) -> Tensor.
    """
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1, with safety factor matching coefficients
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + epsilon)

    # Pre-allocate buffers for the symmetric intermediates and output
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    for a, b, c in _POLAR_EXPRESS_COEFFS:
        ns_line_1(X, out=A)                     # A = X @ X.mT  (symmetric, half-compute)
        ns_line_2(A, alpha=c, beta=b, out=B)    # B = c*(A@A) + b*A  (symmetric, half-compute)
        line_3(X, B, X, beta=a, out=C)          # C = a*X + B@X
        X, C = C, X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
