"""Tests for ``adamw_update_foreach``.

The non-CWD path delegates to ``torch._fused_adamw_``; the CWD path adds a
post-step correction. Tests cover:

- non-CWD matches ``torch.optim.AdamW(fused=True)`` (trivial: same kernel)
- CWD matches the pure-torch foreach reference math
- end-to-end ``NorMuon`` / ``Muon`` with ``algorithm="adamw"``
"""
import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(DEVICE == "cpu", reason="requires CUDA")


def _foreach_cwd_reference(X, G, M, V, lr, beta1, beta2, wd, step, eps):
    """Old pure-foreach CWD math, used as ground truth for the post-correction."""
    n = len(X)
    torch._foreach_lerp_(M, G, [1 - beta1] * n)
    G2 = torch._foreach_mul(G, G)
    torch._foreach_lerp_(V, G2, [1 - beta2] * n)
    bc1 = 1 - beta1 ** step
    bc2_sqrt = (1 - beta2 ** step) ** 0.5
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bc2_sqrt)
    torch._foreach_add_(denom, [eps] * n)
    adj_lr = lr / bc1
    M_div = torch._foreach_div(M, denom)
    # CWD: decay only where sign(X * update_dir) >= 0
    masks = torch._foreach_mul(X, M_div)
    masks = torch._foreach_sign(masks)
    masks = torch._foreach_add(masks, 1)
    masks = torch._foreach_minimum(masks, 1)
    decay = torch._foreach_mul(X, masks)
    torch._foreach_mul_(decay, lr * wd)
    torch._foreach_sub_(X, decay)
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_sub_(X, M_div)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("step", [1, 5])
def test_cwd_matches_foreach_reference(dtype, step):
    from dion.scalar_opts import adamw_update_foreach
    torch.manual_seed(42)
    shapes = [(128,), (64, 32), (16,), (1,)]
    X_ref = [torch.randn(*s, device=DEVICE, dtype=dtype) for s in shapes]
    G     = [torch.randn(*s, device=DEVICE, dtype=dtype) * 0.1 for s in shapes]
    M_ref = [torch.randn_like(x) * 0.01 for x in X_ref]
    V_ref = [torch.rand_like(x) * 0.01 for x in X_ref]

    X = [x.clone() for x in X_ref]
    M = [m.clone() for m in M_ref]
    V = [v.clone() for v in V_ref]

    lr, beta1, beta2, wd, eps = 1e-2, 0.9, 0.999, 0.1, 1e-8

    _foreach_cwd_reference(X_ref, G, M_ref, V_ref, lr, beta1, beta2, wd, step, eps)
    adamw_update_foreach(
        X, G, M, V,
        lr=torch.tensor(lr), beta1=torch.tensor(beta1),
        beta2=torch.tensor(beta2), weight_decay=torch.tensor(wd),
        step=step, epsilon=eps, cautious_wd=True,
    )

    tol = 1e-5 if dtype == torch.float32 else 5e-3
    for a, b in zip(X_ref, X):
        assert torch.allclose(a, b, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_non_cwd_matches_torch_optim_adamw_fused(dtype):
    """Non-CWD path matches ``torch.optim.AdamW(fused=True)`` to within a ulp;
    both dispatch to the same ``torch._fused_adamw_`` kernel, minor drift comes
    from how the step-count tensor is staged."""
    from dion.scalar_opts import adamw_update_foreach
    torch.manual_seed(0)
    shapes = [(128,), (64, 32), (1,)]
    params_ours = [torch.nn.Parameter(torch.randn(*s, device=DEVICE, dtype=dtype))
                   for s in shapes]
    params_ref  = [torch.nn.Parameter(p.detach().clone()) for p in params_ours]
    grads       = [torch.randn_like(p.data) * 0.1 for p in params_ours]

    M = [torch.zeros_like(p.data) for p in params_ours]
    V = [torch.zeros_like(p.data) for p in params_ours]

    opt = torch.optim.AdamW(params_ref, lr=1e-2, betas=(0.9, 0.999),
                            weight_decay=0.1, eps=1e-8, fused=True)

    for step in range(1, 6):
        for p, g in zip(params_ours, grads):
            p.grad = g
        for p, g in zip(params_ref, grads):
            p.grad = g
        adamw_update_foreach(
            [p.data for p in params_ours], grads, M, V,
            lr=torch.tensor(1e-2), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.1),
            step=step, epsilon=1e-8, cautious_wd=False,
        )
        opt.step()

        tol = 1e-6 if dtype == torch.float32 else 1e-2
        for po, pr in zip(params_ours, params_ref):
            assert torch.allclose(po.data, pr.data, atol=tol, rtol=tol), (
                f"drift at step {step}"
            )


def test_empty_list():
    from dion.scalar_opts import adamw_update_foreach
    adamw_update_foreach(
        [], [], [], [],
        lr=torch.tensor(1e-2), beta1=torch.tensor(0.9),
        beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.0),
        step=1, epsilon=1e-8,
    )


def test_cwd_zero_wd_skips_correction():
    """With wd=0, CWD must be a no-op vs non-CWD (nothing to undo)."""
    from dion.scalar_opts import adamw_update_foreach
    torch.manual_seed(0)
    X1 = [torch.randn(64, device=DEVICE)]
    X2 = [X1[0].clone()]
    G  = [torch.randn(64, device=DEVICE)]
    M1, V1 = [torch.zeros_like(X1[0])], [torch.zeros_like(X1[0])]
    M2, V2 = [torch.zeros_like(X2[0])], [torch.zeros_like(X2[0])]

    kw = dict(lr=torch.tensor(1e-2), beta1=torch.tensor(0.9),
              beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.0),
              step=1, epsilon=1e-8)
    adamw_update_foreach(X1, G, M1, V1, cautious_wd=False, **kw)
    adamw_update_foreach(X2, G, M2, V2, cautious_wd=True, **kw)
    assert torch.equal(X1[0], X2[0])


class TestIntegration:
    def test_normuon_with_adamw_scalars(self):
        from dion import NorMuon
        torch.manual_seed(42)
        weights = [torch.nn.Parameter(torch.randn(64, 128, device=DEVICE)),
                   torch.nn.Parameter(torch.randn(128, 64, device=DEVICE))]
        biases = [torch.nn.Parameter(torch.randn(64, device=DEVICE)),
                  torch.nn.Parameter(torch.randn(128, device=DEVICE))]
        opt = NorMuon([
            {"params": weights},
            {"params": biases, "algorithm": "adamw"},
        ], lr=0.01)
        before = [p.data.clone() for p in biases]
        for _ in range(3):
            for p in weights + biases:
                p.grad = torch.randn_like(p)
            opt.step()
        for b_now, b_before in zip(biases, before):
            assert not torch.equal(b_now.data, b_before), "scalar params didn't update"

    def test_muon_with_adamw_cautious_wd(self):
        from dion import Muon
        torch.manual_seed(42)
        weights = [torch.nn.Parameter(torch.randn(64, 128, device=DEVICE))]
        biases = [torch.nn.Parameter(torch.randn(64, device=DEVICE))]
        opt = Muon([
            {"params": weights},
            {"params": biases, "algorithm": "adamw",
             "weight_decay": 0.1, "cautious_wd": True},
        ], lr=0.01)
        before = biases[0].data.clone()
        for _ in range(3):
            for p in weights + biases:
                p.grad = torch.randn_like(p)
            opt.step()
        assert not torch.equal(biases[0].data, before)
        assert torch.isfinite(biases[0].data).all()
