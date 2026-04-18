"""Tests for ``adamw_update_foreach``.

The non-CWD path delegates to ``torch._fused_adamw_``; the CWD path adds a
post-step correction. Tests cover:

- non-CWD matches ``torch.optim.AdamW(fused=True)`` (trivial: same kernel)
- CWD matches an fp32-promoted reference that mirrors the kernel's
  internal math (an all-storage-dtype reference is numerically less
  accurate than the kernel at bf16 and would hide real drift)
- hand-computed CWD spot-check (exact, catches shared bugs)
- shared-step tensor correctness across many params (regression guard
  for a future PyTorch that starts mutating ``state_steps`` inside
  ``_fused_adamw_`` -- our shared tensor would race)
- optimizer actually optimizes on a synthetic regression
- end-to-end ``NorMuon`` / ``Muon`` with ``algorithm="adamw"``
"""
import math

import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(DEVICE == "cpu", reason="requires CUDA")


def _fp32_cwd_reference(X, G, M, V, lr, beta1, beta2, wd, step, eps):
    """fp32-promoted reference matching ``_fused_adamw_``'s internal math.
    Moment updates, bias correction and the sqrt/div all happen in fp32;
    only the final stores round back to storage dtype. An all-bf16 foreach
    reference would actually be *less* accurate than the kernel and hide
    real drift behind a loose tolerance."""
    for i in range(len(X)):
        g32 = G[i].float()
        x32_orig = X[i].float()
        m32 = M[i].float() * beta1 + g32 * (1 - beta1)
        v32 = V[i].float() * beta2 + g32 * g32 * (1 - beta2)
        bc1 = 1 - beta1 ** step
        bc2s = (1 - beta2 ** step) ** 0.5
        denom = v32.sqrt() / bc2s + eps
        update = (lr / bc1) * m32 / denom
        mask = (x32_orig * update >= 0).float()
        x32 = x32_orig - lr * wd * x32_orig * mask - update
        X[i].copy_(x32.to(X[i].dtype))
        M[i].copy_(m32.to(M[i].dtype))
        V[i].copy_(v32.to(V[i].dtype))


@pytest.mark.parametrize("dtype,atol", [
    (torch.float32, 1e-6),
    (torch.bfloat16, 4e-3),
])
@pytest.mark.parametrize("step", [1, 5])
def test_cwd_matches_fp32_reference(dtype, atol, step):
    from dion.scalar_opts import adamw_update_foreach
    torch.manual_seed(42)
    shapes = [(128,), (64, 32), (16,), (1,)]
    X_ref = [torch.randn(*s, device=DEVICE, dtype=dtype) for s in shapes]
    G     = [torch.randn(*s, device=DEVICE, dtype=dtype) * 0.1 for s in shapes]
    M_ref = [torch.randn_like(x) * 0.01 for x in X_ref]
    V_ref = [torch.rand_like(x).abs() * 0.01 for x in X_ref]

    X = [x.clone() for x in X_ref]
    M = [m.clone() for m in M_ref]
    V = [v.clone() for v in V_ref]

    lr, beta1, beta2, wd, eps = 1e-2, 0.9, 0.999, 0.1, 1e-8
    _fp32_cwd_reference(X_ref, G, M_ref, V_ref, lr, beta1, beta2, wd, step, eps)
    adamw_update_foreach(
        X, G, M, V,
        lr=torch.tensor(lr), beta1=torch.tensor(beta1),
        beta2=torch.tensor(beta2), weight_decay=torch.tensor(wd),
        step=step, epsilon=eps, cautious_wd=True,
    )

    for a, b in zip(X_ref, X):
        diff = (a - b).abs().max().item()
        assert diff <= atol, f"CWD drift {diff:.3e} > {atol:.1e} ({dtype}, step={step})"


def test_cwd_hand_computed_spot_check():
    """Exact-match spot check against a hand-computed expected tensor.
    Inputs chosen so the CWD mask is [1, 0, 0, 1] -- two positions get
    decay, two get the correction that undoes it."""
    from dion.scalar_opts import adamw_update_foreach
    X = [torch.tensor([ 1.0,  1.0, -1.0, -1.0], device=DEVICE)]
    G = [torch.tensor([ 2.0, -2.0,  2.0, -2.0], device=DEVICE)]
    M = [torch.zeros(4, device=DEVICE)]
    V = [torch.zeros(4, device=DEVICE)]
    lr, beta1, beta2, wd, eps = 0.1, 0.9, 0.999, 0.5, 1e-8

    adamw_update_foreach(
        X, G, M, V,
        lr=torch.tensor(lr), beta1=torch.tensor(beta1),
        beta2=torch.tensor(beta2), weight_decay=torch.tensor(wd),
        step=1, epsilon=eps, cautious_wd=True,
    )

    # Hand computation in fp32.
    g = torch.tensor([2.0, -2.0, 2.0, -2.0])
    x_orig = torch.tensor([1.0, 1.0, -1.0, -1.0])
    m_new = (1 - beta1) * g
    v_new = (1 - beta2) * g * g
    bc1 = 1 - beta1
    bc2s = math.sqrt(1 - beta2)
    denom = v_new.sqrt() / bc2s + eps
    update = (lr / bc1) * m_new / denom
    # sign(x_orig * update) = sign(x_orig * m_new) = [+,-,-,+] -> mask=[1,0,0,1]
    mask = torch.tensor([1.0, 0.0, 0.0, 1.0])
    expected = x_orig - lr * wd * x_orig * mask - update

    got = X[0].cpu()
    assert torch.allclose(got, expected, atol=1e-6, rtol=0), f"got {got}, expected {expected}"


def test_shared_step_tensor_safe_across_many_params():
    """The implementation passes the same 0-d step tensor N times to
    ``_fused_adamw_``. If a future PyTorch starts mutating ``state_steps``
    inside the op, that shared tensor would race and params would get
    inconsistent bias correction. Assert: N-param shared-step call equals
    N single-param calls with independent step tensors."""
    from dion.scalar_opts import adamw_update_foreach
    n = 64
    torch.manual_seed(0)
    X = [torch.randn(8, device=DEVICE) for _ in range(n)]
    G = [torch.randn(8, device=DEVICE) for _ in range(n)]
    M = [torch.zeros_like(x) for x in X]
    V = [torch.zeros_like(x) for x in X]

    X_single = [x.clone() for x in X]
    G_single = [g.clone() for g in G]
    M_single = [m.clone() for m in M]
    V_single = [v.clone() for v in V]

    kw = dict(lr=torch.tensor(1e-2), beta1=torch.tensor(0.9),
              beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.01),
              step=5, epsilon=1e-8, cautious_wd=False)

    adamw_update_foreach(X, G, M, V, **kw)
    for i in range(n):
        adamw_update_foreach(
            [X_single[i]], [G_single[i]], [M_single[i]], [V_single[i]], **kw
        )

    for i, (a, b) in enumerate(zip(X, X_single)):
        assert torch.equal(a, b), f"param {i} diverged: shared-step differs from single-param"


def test_optimizer_makes_progress_on_regression():
    """End-to-end: the optimizer actually optimizes. 20 steps of AdamW on
    a linear regression with a closed-form optimum must reduce loss by
    >=10x. Catches any regression where the update is zero, sign-flipped,
    or otherwise broken in a way the numerical-parity tests would miss."""
    from dion.scalar_opts import adamw_update_foreach
    torch.manual_seed(0)
    W_true = torch.randn(8, 16, device=DEVICE)
    x = torch.randn(256, 16, device=DEVICE)
    y = x @ W_true.T

    W = torch.nn.Parameter(torch.randn(8, 16, device=DEVICE) * 0.1)
    M = [torch.zeros_like(W.data)]
    V = [torch.zeros_like(W.data)]

    losses = []
    for step in range(1, 21):
        W.grad = None
        loss = (x @ W.T - y).pow(2).mean()
        loss.backward()
        losses.append(loss.item())
        adamw_update_foreach(
            [W.data], [W.grad], M, V,
            lr=torch.tensor(1e-1), beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999), weight_decay=torch.tensor(0.0),
            step=step, epsilon=1e-8, cautious_wd=False,
        )

    assert losses[-1] < losses[0] / 10, (
        f"insufficient progress: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_non_cwd_matches_torch_optim_adamw_fused(dtype):
    """Non-CWD path matches ``torch.optim.AdamW(fused=True)`` (same
    underlying kernel). fp32 agrees to sub-ULP; bf16 to ~1 ULP."""
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

        atol = 1e-6 if dtype == torch.float32 else 2e-2  # ~1 bf16 ULP
        for po, pr in zip(params_ours, params_ref):
            diff = (po.data - pr.data).abs().max().item()
            assert diff <= atol, f"drift at step {step}: {diff:.3e} > {atol:.1e}"


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
