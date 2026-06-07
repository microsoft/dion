"""Tests for eager optimizer-state pre-population in DistributedOrthoBase.

The DistributedOrthoBase family (Muon, NorMuon, Dion2, NorDion2) materializes
optimizer state for every parameter at construction time, including parameters
that may never receive a gradient. This keeps state_dict() complete and
rank-symmetric (a requirement for distributed checkpointing) and is numerically
inert: a grad-less param's buffers stay zero and its weights are untouched.

Ported behavior from InternLM/xtuner v1/optim/muon.py.
"""

import pytest
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_AVAILABLE = torch.cuda.is_available()


def _optimizer_cases():
    from dion import Dion2, Muon, NorDion2, NorMuon

    # (class, kwargs, extra state keys beyond "momentum" for the ortho algo)
    return [
        (Muon, dict(lr=0.01), set()),
        (NorMuon, dict(lr=0.01), {"variance_neuron"}),
        (Dion2, dict(lr=0.01), set()),
        (NorDion2, dict(lr=0.01), {"variance_neuron"}),
    ]


def _make_params():
    torch.manual_seed(42)
    active = torch.nn.Parameter(torch.randn(64, 128, device=DEVICE))
    frozen = torch.nn.Parameter(torch.randn(64, 128, device=DEVICE))
    return active, frozen


@pytest.mark.parametrize("opt_cls,opt_kwargs,extra_keys", _optimizer_cases())
def test_state_prepopulated_at_init(opt_cls, opt_kwargs, extra_keys):
    active, frozen = _make_params()
    opt = opt_cls([active, frozen], **opt_kwargs)

    # Both params have state immediately, before any step or gradient.
    assert active in opt.state
    assert frozen in opt.state

    expected_keys = {"momentum"} | extra_keys
    for p in (active, frozen):
        assert set(opt.state[p].keys()) == expected_keys
        assert torch.count_nonzero(opt.state[p]["momentum"]) == 0
        assert opt.state[p]["momentum"].shape == p.shape


@pytest.mark.parametrize("opt_cls,opt_kwargs,extra_keys", _optimizer_cases())
def test_state_dict_complete_and_resumable(opt_cls, opt_kwargs, extra_keys):
    active, frozen = _make_params()
    opt = opt_cls([active, frozen], **opt_kwargs)

    # state_dict() carries an entry for every param (indices 0 and 1), even
    # though neither has seen a gradient yet.
    sd = opt.state_dict()
    assert set(sd["state"].keys()) == {0, 1}

    # Resuming into a fresh optimizer must not raise and must restore both keys.
    active2, frozen2 = _make_params()
    opt2 = opt_cls([active2, frozen2], **opt_kwargs)
    opt2.load_state_dict(sd)
    assert set(opt2.state.keys()) == {active2, frozen2}


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for optimizer step")
@pytest.mark.parametrize("opt_cls,opt_kwargs,extra_keys", _optimizer_cases())
def test_gradless_param_is_inert(opt_cls, opt_kwargs, extra_keys):
    active, frozen = _make_params()
    frozen_before = frozen.data.clone()
    opt = opt_cls([active, frozen], **opt_kwargs)

    for step in range(3):
        torch.manual_seed(100 + step)
        active.grad = torch.randn_like(active)
        frozen.grad = None  # never receives a gradient
        opt.step()

    # The grad-less param is untouched and its buffers remain zero, proving the
    # pre-populated state does not perturb the update.
    assert torch.equal(frozen.data, frozen_before)
    for key in {"momentum"} | extra_keys:
        assert torch.count_nonzero(opt.state[frozen][key]) == 0

    # The active param did train.
    assert torch.count_nonzero(opt.state[active]["momentum"]) > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for optimizer step")
def test_prepopulation_is_numerically_inert_for_active_param():
    from dion import Muon

    # First step on an active param: with momentum pre-initialized to zero,
    # M = mu * 0 + G == G, so the buffer equals the gradient. This is exactly
    # what the lazy path produced, so pre-population changes nothing numerically.
    active, _ = _make_params()
    opt = Muon([active], lr=0.01)
    torch.manual_seed(7)
    g = torch.randn_like(active)
    active.grad = g.clone()
    opt.step()
    assert torch.equal(
        opt.state[active]["momentum"], g.to(opt.state[active]["momentum"].dtype)
    )
