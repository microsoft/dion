"""Tests for the Triton kernel ``dion2_post_orthogonalize_triton``.

Two levels of testing:
1. **Function-level**: compare the Triton kernel output directly against the
   compiled ``dion2_post_orthogonalize`` on identical inputs.
   - Unselected entries should be bitwise identical (both compute a*x).
   - Selected entries differ at the FP-rounding level because the Triton kernel
     fuses ``a*x - b*u`` in one expression (one rounding) while the compiled
     version does ``x *= a`` then ``scatter_add_`` (two roundings).
2. **End-to-end**: run Dion2 optimizer with ``use_triton=True`` vs default
   and verify parameters are close.
"""

import math
import pytest
import torch

from dion.dion2 import dion2_post_orthogonalize
from dion.dion2_triton import TRITON_AVAILABLE, dion2_post_orthogonalize_triton

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AND_CUDA = CUDA_AVAILABLE and TRITON_AVAILABLE

torch._dynamo.config.cache_size_limit = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_data(shape, k, select_dim, seed=42, x_dtype=torch.float32):
    """Create X, U, indices for a single tensor.

    Returns:
        X: (*leading, M, N) in x_dtype
        U: (*leading, k, N) or (*leading, M, k) bfloat16
        indices: (*leading, k) int64
    """
    torch.manual_seed(seed)
    M, N = shape[-2], shape[-1]
    leading = shape[:-2]

    X = torch.randn(shape, device=DEVICE, dtype=x_dtype)

    if select_dim == -2:
        u_shape = (*leading, k, N)
    else:
        u_shape = (*leading, M, k)
    U = torch.randn(u_shape, device=DEVICE, dtype=torch.bfloat16)

    full_dim = M if select_dim == -2 else N
    # Generate unique random indices per leading element
    idx_shape = (*leading, full_dim)
    # For each batch element, pick k random indices
    flat_leading = max(1, int(torch.tensor(leading).prod().item())) if leading else 1
    indices_flat = torch.stack(
        [torch.randperm(full_dim, device=DEVICE)[:k] for _ in range(flat_leading)]
    )
    indices = indices_flat.reshape(*leading, k)

    return X, U, indices


def _build_selected_mask(indices, select_dim, shape):
    """Build a boolean mask over X marking selected entries.

    Returns a mask with the same shape as X where True = selected.
    """
    leading = shape[:-2]
    M, N = shape[-2], shape[-1]

    flat_leading = max(1, int(torch.tensor(leading).prod().item())) if leading else 1
    idx_flat = indices.reshape(flat_leading, -1)

    if select_dim == -2:
        # Build row mask: (flat_leading, M)
        row_mask = torch.zeros(flat_leading, M, device=indices.device, dtype=torch.bool)
        row_mask.scatter_(1, idx_flat, True)
        # Expand to (flat_leading, M, N)
        full_mask = row_mask.unsqueeze(-1).expand(flat_leading, M, N)
    else:
        col_mask = torch.zeros(flat_leading, N, device=indices.device, dtype=torch.bool)
        col_mask.scatter_(1, idx_flat, True)
        full_mask = col_mask.unsqueeze(-2).expand(flat_leading, M, N)

    return full_mask.reshape(shape)


# ---------------------------------------------------------------------------
# Function-level tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TRITON_AND_CUDA, reason="CUDA and Triton required")
class TestPostOrthoTritonKernel:
    """Compare Triton kernel output against compiled reference."""

    @pytest.mark.parametrize("shape, select_dim, fraction, weight_decay", [
        # 2D shapes
        ((64, 128), -2, 0.25, 0.01),     # wide, row selection
        ((64, 128), -2, 0.5, 0.01),      # wide, row selection, higher fraction
        ((128, 64), -2, 0.25, 0.01),     # tall, row selection
        ((64, 128), -1, 0.25, 0.01),     # wide, col selection
        ((128, 64), -1, 0.5, 0.01),      # tall, col selection
        ((64, 64), -2, 0.5, 0.01),       # square
        # 3D shapes (batched)
        ((4, 32, 128), -2, 0.25, 0.01),  # 3D row selection
        ((4, 128, 32), -1, 0.5, 0.01),   # 3D col selection
        ((8, 64, 64), -2, 0.5, 0.01),    # 3D square
        # Edge cases
        ((64, 128), -2, 1.0, 0.01),      # k = full_dim (all selected)
        ((64, 128), -2, 1/64, 0.01),     # k = 1 (single row selected)
        ((64, 128), -2, 0.25, 0.0),      # zero weight decay
    ])
    @pytest.mark.parametrize("x_dtype", [torch.float64, torch.float32, torch.bfloat16])
    def test_single_tensor(self, shape, select_dim, fraction, weight_decay, x_dtype):
        full_dim = shape[-2] if select_dim == -2 else shape[-1]
        k = max(1, int(math.ceil(fraction * full_dim)))

        X_ref, U_ref, indices = _make_test_data(shape, k, select_dim, x_dtype=x_dtype)
        X_tri = X_ref.clone()
        U_tri = U_ref.clone()

        base_lr = torch.tensor(0.01, device=DEVICE)
        adjusted_lr = torch.tensor(0.005, device=DEVICE)
        weight_decay = torch.tensor(weight_decay, device=DEVICE)

        kwargs = dict(indices=[indices], base_lr=base_lr, adjusted_lr=adjusted_lr,
                      weight_decay=weight_decay, select_dim=select_dim)
        dion2_post_orthogonalize(X=[X_ref], U=[U_ref], **kwargs)
        dion2_post_orthogonalize_triton(X=[X_tri], U=[U_tri], **kwargs)

        # Build mask for selected vs unselected
        sel_mask = _build_selected_mask(indices, select_dim, shape)

        # Dtype-aware tolerances: bf16 has coarser rounding (~1 ULP = 2^-7 * value)
        # so the one-vs-two-rounding difference is larger.
        if x_dtype == torch.bfloat16:
            atol, rtol = 1e-2, 1e-2
        elif x_dtype == torch.float64:
            atol, rtol = 1e-8, 1e-8
        else:
            atol, rtol = 1e-6, 1e-5

        # Unselected entries: both compute a*x with the same rounding path
        # (promote to f32, multiply, store back) — should be bitwise identical
        assert torch.equal(X_ref[~sel_mask], X_tri[~sel_mask]), (
            f"Unselected entries differ! "
            f"max diff = {(X_ref[~sel_mask] - X_tri[~sel_mask]).abs().max().item():.2e}"
        )

        # Selected entries: fused vs two-step rounding
        assert torch.allclose(X_ref[sel_mask], X_tri[sel_mask], atol=atol, rtol=rtol), (
            f"Selected entries differ beyond tolerance: "
            f"max diff = {(X_ref[sel_mask] - X_tri[sel_mask]).abs().max().item():.2e}"
        )

        # Overall comparison
        assert torch.allclose(X_ref, X_tri, atol=atol, rtol=rtol), (
            f"Overall max diff = {(X_ref - X_tri).abs().max().item():.2e}"
        )

    def test_megabatch(self):
        """Multiple same-shape tensors in a single call."""
        shape = (64, 128)
        k = 16
        select_dim = -2
        n_tensors = 4

        X_refs, X_tris, Us_ref, Us_tri, idxs = [], [], [], [], []
        for i in range(n_tensors):
            x, u, idx = _make_test_data(shape, k, select_dim, seed=42 + i)
            X_refs.append(x)
            X_tris.append(x.clone())
            Us_ref.append(u.clone())
            Us_tri.append(u.clone())
            idxs.append(idx)

        base_lr = torch.tensor(0.01, device=DEVICE)
        adjusted_lr = torch.tensor(0.005, device=DEVICE)
        weight_decay = torch.tensor(0.01, device=DEVICE)

        kwargs = dict(indices=idxs, base_lr=base_lr, adjusted_lr=adjusted_lr,
                      weight_decay=weight_decay, select_dim=select_dim)
        dion2_post_orthogonalize(X=X_refs, U=Us_ref, **kwargs)
        dion2_post_orthogonalize_triton(X=X_tris, U=Us_tri, **kwargs)

        for i in range(n_tensors):
            sel_mask = _build_selected_mask(idxs[i], select_dim, shape)
            assert torch.equal(X_refs[i][~sel_mask], X_tris[i][~sel_mask]), f"Tensor {i}: unselected differ"
            assert torch.allclose(X_refs[i][sel_mask], X_tris[i][sel_mask], atol=1e-6, rtol=1e-5), (
                f"Tensor {i}: selected differ, max diff = "
                f"{(X_refs[i][sel_mask] - X_tris[i][sel_mask]).abs().max().item():.2e}"
            )
            assert torch.allclose(X_refs[i], X_tris[i], atol=1e-6, rtol=1e-5), (
                f"Tensor {i}: max diff = {(X_refs[i] - X_tris[i]).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# End-to-end optimizer tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TRITON_AND_CUDA, reason="CUDA and Triton required")
class TestDion2TritonEndToEnd:
    """Run Dion2 optimizer with use_triton=True vs default and compare."""

    def _make_params(self, shapes, dtype=torch.float32):
        torch.manual_seed(42)
        return [torch.nn.Parameter(torch.randn(s, device=DEVICE, dtype=dtype)) for s in shapes]

    def _run_steps(self, params, opt_kwargs, n_steps=3):
        from dion import Dion2
        opt = Dion2(params, **opt_kwargs)
        for step in range(n_steps):
            torch.manual_seed(100 + step)
            for p in params:
                p.grad = torch.randn_like(p)
            opt.step()
        return opt

    @pytest.mark.parametrize("shapes, fraction, ef_decay", [
        ([(32, 128)], 0.25, 0.95),
        ([(128, 32)], 0.5, 0.95),
        ([(4, 32, 128)], 0.5, 0.95),
        ([(64, 128)] * 4, 0.25, 0.9),
    ])
    @pytest.mark.parametrize("use_triton", [False, True])
    @pytest.mark.parametrize("use_gram_newton_schulz", [False, True])
    @pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16])
    def test_triton_vs_default(self, shapes, fraction, ef_decay, use_triton, use_gram_newton_schulz, param_dtype):
        """Triton post-ortho should match default up to fused-rounding tolerance."""
        kwargs = dict(
            lr=0.01, fraction=fraction, ef_decay=ef_decay,
            use_triton=use_triton, use_gram_newton_schulz=use_gram_newton_schulz,
        )

        p_default = self._make_params(shapes, dtype=param_dtype)
        opt_default = self._run_steps(p_default, {**kwargs, "triton_post_ortho": False}, n_steps=3)

        p_triton = self._make_params(shapes, dtype=param_dtype)
        opt_triton = self._run_steps(p_triton, {**kwargs, "triton_post_ortho": True}, n_steps=3)

        # Momentum should be bitwise identical (same pre-ortho path)
        for pd, pt in zip(p_default, p_triton):
            md = opt_default.state[pd]["momentum"]
            mt = opt_triton.state[pt]["momentum"]
            assert torch.equal(md, mt), "Momentum buffers differ"

        # Dtype-aware tolerance: bf16 rounding compounds across steps
        if param_dtype == torch.bfloat16:
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 1e-6, 1e-5

        # Parameters differ only at fused-rounding level (triton fuses a*x - b*u
        # in one expression vs two separate ops in the compiled version)
        for pd, pt in zip(p_default, p_triton):
            assert torch.allclose(pd.data, pt.data, atol=atol, rtol=rtol), (
                f"Parameters differ beyond tolerance: "
                f"max diff = {(pd.data - pt.data).abs().max().item():.2e}"
            )
