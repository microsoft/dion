"""Tests for NorMuon's defensive ``nan_guard_fallback`` option.

Background. microsoft/dion#76 reports that NorMuon + gram-newton-schulz +
quack-kernels 0.4.1 occasionally produces NaN parameters on certain
hardware/shapes; gradients are clean entering ``optimizer.step()`` but the
post-step parameter is all-NaN. ``nan_guard_fallback`` adds a single
``all_reduce(MAX)`` of one byte per shape group per step so all ranks
agree on whether to skip the update; on detection the entire post-ortho
path is bypassed so the parameter, the variance buffer, and the run state
remain strictly unchanged for these params.

These tests run single-rank (no NCCL/MPI) and use a custom Newton-Schulz
function that injects NaN, isolating the guard logic from any specific
NS backend.
"""

import warnings

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


def _nan_ns(X, epsilon=None):
    return torch.full_like(X, float("nan"))


def _identity_ns(X, epsilon=None):
    # Bypass torch.compile and shape gymnastics; a plain identity is enough
    # to validate the guard and the no-fallback baseline.
    return X.clone()


def _make_param(shape, seed=42):
    torch.manual_seed(seed)
    return torch.nn.Parameter(torch.randn(shape, device=DEVICE))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="NorMuon's compiled kernels target CUDA")
def test_fallback_off_lets_nan_propagate_to_params():
    from dion import NorMuon

    p = _make_param((8, 16))
    p.grad = torch.randn_like(p)

    opt = NorMuon(
        [p],
        lr=0.01,
        newton_schulz_func=_nan_ns,
        nan_guard_fallback=False,
    )
    opt.step()

    # Without the guard, the NaN NS output flows through normalization and
    # the post-ortho update, poisoning the parameter. This is the bug from
    # issue #76 in unit-test form.
    assert torch.isnan(p.data).any(), (
        "expected NaN to propagate to params with the guard off; if this "
        "asserts, NorMuon's NaN behavior changed and the fallback test "
        "below may no longer be exercising the right path."
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="NorMuon's compiled kernels target CUDA")
def test_fallback_on_skips_step_when_ns_returns_nan():
    from dion import NorMuon

    p = _make_param((8, 16))
    before = p.data.clone()
    p.grad = torch.randn_like(p)

    opt = NorMuon(
        [p],
        lr=0.01,
        newton_schulz_func=_nan_ns,
        nan_guard_fallback=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opt.step()

    assert torch.isfinite(p.data).all()
    # ``return`` before post-ortho means the parameter is bit-exact unchanged.
    assert torch.equal(p.data, before)
    # Variance buffer must also be left at its initial zero state - skipping
    # the step means "this batch never happened" for these params.
    assert torch.all(opt.state[p]["variance_neuron"] == 0)

    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("nan_guard_fallback" in m for m in msgs), msgs


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="NorMuon's compiled kernels target CUDA")
def test_fallback_on_does_not_block_normal_step():
    from dion import NorMuon

    p = _make_param((8, 16))
    before = p.data.clone()
    p.grad = torch.randn_like(p)

    # Identity NS gives finite output -> guard sees nothing wrong, step
    # proceeds normally and the parameter changes.
    opt = NorMuon(
        [p],
        lr=0.01,
        newton_schulz_func=_identity_ns,
        nan_guard_fallback=True,
    )
    opt.step()

    assert torch.isfinite(p.data).all()
    assert not torch.equal(p.data, before)
