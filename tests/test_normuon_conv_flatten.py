"""Flattened convolution updates must match the same matrix representation.

Adapted from JohnLangford's proposed NorMuon fix (61e34048253c0dce8c26792b64ae94ae6e7de953).
Distributed comparisons use the SAME sharding: NorMuon intentionally rescales
per local shard, so unsharded parity is not an appropriate FSDP oracle.
"""

import copy

import pytest
import torch

from dion import Dion2, Dion3, NorDion2, NorMuon


CONV_SHAPES = [(6, 3, 5), (8, 4, 3, 3), (6, 2, 3, 3, 3),
               (12, 2, 1, 1), (8, 1, 3, 3), (4, 1, 1, 1)]


@pytest.fixture(scope="module", autouse=True)
def isolated_compiler_state():
    # Shape sweeps must neither inherit nor leak compiled specializations.
    # Keep compilation enabled while making the budget independent of order.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def polynomial_ns(x, epsilon):
    x = x.double()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + epsilon)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.mT
    for _ in range(5):
        x = 1.5 * x - 0.5 * (x @ x.mT) @ x
    return x.mT if tall else x


@pytest.fixture(autouse=True)
def compilation_budget(monkeypatch):
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 128)


def make_optimizer(params, **options):
    defaults = dict(lr=7e-4, weight_decay=1e-2, nesterov=True,
                    flatten=True, adjust_lr="rms_norm", newton_schulz_func=polynomial_ns)
    return NorMuon(params, **(defaults | options))


@pytest.mark.parametrize("shape", CONV_SHAPES)
@pytest.mark.parametrize("count", [1, 2, 3, 5])
@pytest.mark.parametrize("adjust_lr", ["rms_norm", "spectral_norm", None])
def test_conv_updates_match_flattened_matrices(shape, count, adjust_lr):
    torch.manual_seed(7)
    conv = [torch.nn.Parameter(torch.randn(shape, dtype=torch.float64)) for _ in range(count)]
    matrix = [torch.nn.Parameter(p.detach().flatten(1).clone()) for p in conv]
    a = make_optimizer(conv, adjust_lr=adjust_lr)
    b = make_optimizer(matrix, adjust_lr=adjust_lr)
    for step in range(3):
        for i, (p, q) in enumerate(zip(conv, matrix)):
            p.grad = torch.randn_like(p) * (i + step + 1)
            q.grad = p.grad.flatten(1).clone()
        a.step()
        b.step()
        for p, q in zip(conv, matrix):
            torch.testing.assert_close(p.flatten(1), q, rtol=1e-10, atol=1e-12)
            for key in ("momentum", "variance_neuron"):
                torch.testing.assert_close(a.state[p][key].flatten(1), b.state[q][key],
                                           rtol=1e-10, atol=1e-12)
            assert a.state[p]["variance_neuron_layout_version"] == 1


@pytest.mark.parametrize("nesterov", [False, True])
def test_zero_rank_deficient_and_noncontiguous_gradients(nesterov):
    torch.manual_seed(12)
    params = [torch.nn.Parameter(torch.randn(8, 3, 3, 3, dtype=torch.float64))
              for _ in range(3)]
    ref = [torch.nn.Parameter(p.detach().flatten(1).clone()) for p in params]
    a, b = make_optimizer(params, nesterov=nesterov), make_optimizer(ref, nesterov=nesterov)
    gradients = [torch.zeros_like(params[0]), torch.ones_like(params[0]),
                 torch.randn(8, 3, 3, 3, 2, dtype=torch.float64)[..., 0]]
    assert not gradients[-1].is_contiguous()
    for _ in range(3):
        for p, q, g in zip(params, ref, gradients):
            p.grad, q.grad = g, g.flatten(1)
        a.step()
        b.step()
        for p, q in zip(params, ref):
            torch.testing.assert_close(p.flatten(1), q, rtol=1e-10, atol=1e-12)
            assert torch.isfinite(p).all()


@pytest.mark.parametrize("shape", CONV_SHAPES)
def test_flatten_false_retains_trailing_matrix_geometry(shape):
    torch.manual_seed(13)
    p = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().reshape(-1, *shape[-2:]).clone())
    a, b = make_optimizer([p], flatten=False), make_optimizer([q], flatten=False)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.reshape_as(q).clone()
        a.step()
        b.step()
        torch.testing.assert_close(p.reshape_as(q), q, rtol=1e-10, atol=1e-12)
    assert "variance_neuron_layout_version" not in a.state[p]


@pytest.mark.parametrize("shape", [(5, 7), (7, 5)])
def test_linear_flatten_is_noop(shape):
    torch.manual_seed(14)
    p = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    a, b = make_optimizer([p], flatten=True), make_optimizer([q], flatten=False)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        a.step()
        b.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ("momentum", "variance_neuron"):
            torch.testing.assert_close(a.state[p][key], b.state[q][key], rtol=0, atol=0)
    assert "variance_neuron_layout_version" not in a.state[p]


def test_group_overrides_added_groups_and_gradless_state():
    flat = torch.nn.Parameter(torch.randn(8, 4, 3, 3, dtype=torch.float64))
    spatial = torch.nn.Parameter(torch.randn_like(flat))
    scalar = torch.nn.Parameter(torch.randn_like(flat))
    opt = make_optimizer([dict(params=[flat], flatten=True),
                          dict(params=[spatial], flatten=False),
                          dict(params=[scalar], algorithm="adamw")], flatten=False)
    late = torch.nn.Parameter(torch.randn_like(flat))
    opt.add_param_group(dict(params=[late], flatten=True))
    for p in (flat, late):
        assert opt.state[p]["variance_neuron"].shape == (8, 1, 1, 1)
        assert opt.state[p]["variance_neuron_layout_version"] == 1
    assert opt.state[spatial]["variance_neuron"].shape == (8, 4, 3, 1)
    assert "variance_neuron" not in opt.state[scalar]
    before = late.detach().clone()
    flat.grad = torch.randn_like(flat)
    opt.step()
    torch.testing.assert_close(late, before, rtol=0, atol=0)
    assert torch.count_nonzero(opt.state[late]["momentum"]) == 0
    assert torch.count_nonzero(opt.state[late]["variance_neuron"]) == 0


def assert_nested_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("shape,flatten", [((8, 4, 3, 3), True), ((4, 1, 1, 1), True),
                                          ((5, 7), True), ((8, 4, 3, 3), False)])
def test_checkpoint_roundtrip(shape, flatten):
    torch.manual_seed(15)
    p = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    a = make_optimizer([p], flatten=flatten)
    p.grad = torch.randn_like(p)
    a.step()
    q = torch.nn.Parameter(p.detach().clone())
    b = make_optimizer([q], flatten=flatten)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    for _ in range(3):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        a.step()
        b.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        assert_nested_equal(a.state_dict(), b.state_dict())


@pytest.mark.parametrize("shape", [(8, 4, 3, 3), (4, 1, 1, 1)])
@pytest.mark.parametrize("corruption", ["legacy", "shape", "version", "missing_variance",
                                       "tensor_version", "missing_algorithm", "missing_flatten"])
def test_incompatible_checkpoint_rejected_without_mutation(shape, corruption):
    p = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    opt = make_optimizer([p])
    p.grad = torch.randn_like(p)
    opt.step()
    before = copy.deepcopy(opt.state_dict())
    bad = copy.deepcopy(before)
    state = bad["state"][0]
    if corruption == "legacy":
        del state["variance_neuron_layout_version"]
        state["variance_neuron"] = torch.zeros_like(p[..., :1])
    elif corruption == "shape":
        state["variance_neuron"] = torch.zeros(shape[0], 2, dtype=p.dtype)
    elif corruption == "version":
        state["variance_neuron_layout_version"] = 99
    elif corruption == "missing_variance":
        del state["variance_neuron"]
    elif corruption == "tensor_version":
        state["variance_neuron_layout_version"] = torch.tensor(1)
    else:
        del bad["param_groups"][0][corruption.removeprefix("missing_")]
    bad["param_groups"][0]["lr"] = 9.0
    error = KeyError if corruption in ("missing_algorithm", "missing_flatten") else ValueError
    message = corruption.removeprefix("missing_") if error is KeyError else "Incompatible NorMuon"
    with pytest.raises(error, match=message):
        opt.load_state_dict(bad)
    assert_nested_equal(opt.state_dict(), before)


@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize("has_grad", [False, True])
def test_changing_live_flatten_is_rejected(flatten, has_grad):
    matrix = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float64))
    p = torch.nn.Parameter(torch.randn(4, 1, 1, 1, dtype=torch.float64))
    opt = make_optimizer([dict(params=[matrix]), dict(params=[p])], flatten=flatten)
    matrix.grad = torch.randn_like(matrix)
    if has_grad:
        p.grad = torch.randn_like(p)
    opt.param_groups[1]["flatten"] = not flatten
    weights = [param.detach().clone() for param in (matrix, p)]
    states = [copy.deepcopy(opt.state[param]) for param in (matrix, p)]
    with pytest.raises(ValueError, match="Incompatible NorMuon"):
        opt.step()
    for param, weight, state in zip((matrix, p), weights, states):
        torch.testing.assert_close(param, weight, rtol=0, atol=0)
        assert_nested_equal(opt.state[param], state)


@pytest.mark.parametrize("cls", [Dion2, NorDion2, Dion3])
@pytest.mark.parametrize("shape", [(6, 3, 5), (8, 4, 3, 3), (6, 2, 3, 3, 3)])
@pytest.mark.parametrize("has_grad", [False, True])
def test_low_rank_rejects_at_construction(cls, shape, has_grad):
    """The guard fires before a forward pass, not at the first step()."""
    matrix = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float64))
    conv = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    matrix.grad = torch.randn_like(matrix)
    if has_grad:
        conv.grad = torch.randn_like(conv)
    with pytest.raises(NotImplementedError, match="flatten=True"):
        cls([dict(params=[matrix]), dict(params=[conv])], lr=7e-4,
            flatten=True, newton_schulz_func=polynomial_ns)


@pytest.mark.parametrize("cls", [Dion2, NorDion2, Dion3])
def test_low_rank_rejects_late_added_group(cls):
    """A rejected group must not remain installed or acquire state/caches."""
    matrix = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float64))
    conv = torch.nn.Parameter(torch.randn(8, 4, 3, 3, dtype=torch.float64))
    opt = cls([matrix], lr=7e-4, weight_decay=torch.tensor(1e-2),
              flatten=True, newton_schulz_func=polynomial_ns)
    before = copy.deepcopy(opt.state_dict())
    cache_before = copy.deepcopy(opt._hyperparam_tensors)
    cache_ids = {key: id(value) for key, value in opt._hyperparam_tensors.items()}
    live_before = copy.deepcopy(opt._live_hyperparams_by_group)
    group = opt.param_groups[0]
    with pytest.raises(NotImplementedError, match="flatten=True"):
        # Exercise PyTorch's iterable normalization as well as default flatten.
        opt.add_param_group(dict(params=(p for p in [conv])))
    assert len(opt.param_groups) == 1 and opt.param_groups[0] is group
    assert conv not in opt.state
    assert_nested_equal(opt.state_dict(), before)
    assert_nested_equal(opt._hyperparam_tensors, cache_before)
    assert {key: id(value) for key, value in opt._hyperparam_tensors.items()} == cache_ids
    assert_nested_equal(opt._live_hyperparams_by_group, live_before)

    # Subsequent valid additions and updates must work with the correct indices.
    late = torch.nn.Parameter(torch.randn_like(matrix))
    opt.add_param_group(dict(params=[late]))
    weights = [p.detach().clone() for p in (matrix, late, conv)]
    for p in (matrix, late, conv):
        p.grad = torch.randn_like(p)
    opt.step()
    for p, original in zip((matrix, late), weights):
        assert not torch.equal(p, original)
        assert torch.count_nonzero(opt.state[p]["momentum"]) > 0
    torch.testing.assert_close(conv, weights[-1], rtol=0, atol=0)
    assert conv not in opt.state


@pytest.mark.parametrize("cls", [Dion2, NorDion2, Dion3])
@pytest.mark.parametrize("shape", [(6, 3, 5), (8, 4, 3, 3), (6, 2, 3, 3, 3)])
@pytest.mark.parametrize("has_grad", [False, True])
@pytest.mark.parametrize("source", ["live_group", "checkpoint"])
def test_low_rank_runtime_guard_prevents_partial_updates(cls, shape, has_grad, source):
    matrix = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float64))
    conv = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
    opt = cls([dict(params=[matrix]), dict(params=[conv], flatten=False)],
              lr=7e-4, flatten=True, newton_schulz_func=polynomial_ns)
    matrix.grad = torch.randn_like(matrix)
    opt.step()  # Give the earlier valid group nonzero state before rejection.
    if has_grad:
        conv.grad = torch.randn_like(conv)
    if source == "live_group":
        opt.param_groups[1]["flatten"] = True
    else:
        legacy = copy.deepcopy(opt.state_dict())
        legacy["param_groups"][1]["flatten"] = True
        opt.load_state_dict(legacy)
    weights = [p.detach().clone() for p in (matrix, conv)]
    states = [copy.deepcopy(opt.state[p]) for p in (matrix, conv)]
    with pytest.raises(NotImplementedError, match="flatten=True"):
        opt.step()
    for p, weight, state in zip((matrix, conv), weights, states):
        torch.testing.assert_close(p, weight, rtol=0, atol=0)
        # Group step counters may advance, but weights/momentum/variance may not.
        assert_nested_equal(opt.state[p], state)


@pytest.mark.parametrize("cls", [Dion2, NorDion2])
@pytest.mark.parametrize("scope", ["local", "global"])
def test_low_rank_flatten_would_have_used_the_wrong_axes(cls, scope, monkeypatch):
    """Why the guard exists, not just that it fires.

    Neutralizing the guard, a flattened convolution does NOT receive the update
    its equivalent [out, prod(rest)] matrix receives. select_dim is derived from
    the raw trailing two dimensions before any flattening, so for a 3D+
    parameter the top-k ranks kernel slices instead of output channels, and the
    error-feedback mask lands on those same wrong axes. The megabatch
    Newton-Schulz fix does not reach this: selection happens first, in
    dion2_pre_orthogonalize, on the unflattened tensor.
    """
    monkeypatch.setattr(cls, "_supports_flattened_3d", True)
    shape = (8, 4, 5, 3)
    torch.manual_seed(0)
    init = torch.randn(shape, dtype=torch.float64)
    conv = torch.nn.Parameter(init.clone())
    matrix = torch.nn.Parameter(init.flatten(1).clone())
    options = dict(lr=1e-2, fraction=0.5, selection_scope=scope,
                   newton_schulz_func=polynomial_ns)
    flattened = cls([conv], flatten=True, **options)
    reference = cls([matrix], flatten=False, **options)
    for _ in range(2):
        g = torch.randn(shape, dtype=torch.float64)
        conv.grad, matrix.grad = g, g.flatten(1)
        flattened.step()
        reference.step()
    assert not torch.allclose(conv.detach().flatten(1), matrix.detach(),
                              rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("cls", [Dion2, NorDion2, Dion3])
@pytest.mark.parametrize("algorithm", ["adamw", "lion"])
def test_low_rank_guard_exempts_fallback_groups(cls, algorithm):
    matrix = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float64))
    fallback = torch.nn.Parameter(torch.randn(8, 4, 3, 3, dtype=torch.float64))
    opt = cls([dict(params=[matrix]), dict(params=[fallback], algorithm=algorithm)],
              lr=7e-4, flatten=True, newton_schulz_func=polynomial_ns)
    before = fallback.detach().clone()
    matrix.grad = torch.randn_like(matrix)
    fallback.grad = torch.randn_like(fallback)
    opt.step()
    assert not torch.equal(fallback, before)
    assert torch.count_nonzero(opt.state[fallback]["momentum"]) > 0


@pytest.mark.parametrize("cls", [Dion2, NorDion2])
def test_low_rank_flatten_false_still_steps(cls):
    p = torch.nn.Parameter(torch.randn(8, 4, 3, 3, dtype=torch.float64))
    p.grad = torch.randn_like(p)
    opt = cls([p], lr=7e-4, flatten=False, newton_schulz_func=polynomial_ns)
    opt.step()
    assert torch.isfinite(p).all()
