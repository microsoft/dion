"""Regression coverage for flattening stacked convolution parameters."""

import math
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.megabatch_base import (
    adjust_lr_rms_norm,
    adjust_lr_spectral_norm,
    megabatch_orthogonalize_async,
    muon_update_newton_schulz,
)


CUDA_DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0


@pytest.fixture(scope="module", autouse=True)
def isolated_compiler_state():
    # Shape sweeps must neither inherit nor leak compiled specializations.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _drain(generator):
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value


@pytest.mark.parametrize(
    "param_shape, expected_matrix_shape",
    [
        ((5, 7), (5, 7)),
        ((5, 3, 4), (5, 12)),
        ((5, 3, 2, 2), (5, 12)),
        ((5, 3, 2, 2, 3), (5, 36)),  # dense Conv3d: OIDHW
        ((5, 2, 2, 3, 3), (5, 36)),  # Alternative output-first sparse layout
    ],
)
@pytest.mark.parametrize("n_params", [1, 2, 3, 5])
def test_flatten_preserves_megabatch_axis_and_parameter_rows(
    param_shape, expected_matrix_shape, n_params
):
    inputs = [
        torch.arange(math.prod(param_shape), dtype=torch.float32)
        .reshape(param_shape)
        .add_(1000 * index)
        for index in range(n_params)
    ]
    seen_shapes = []

    def shape_spy(x, epsilon):
        seen_shapes.append(tuple(x.shape))
        return x

    result = _drain(
        megabatch_orthogonalize_async(
            inputs,
            comm_dim=None,
            device_rank=0,
            world_size=1,
            process_group=None,
            newton_schulz_func=shape_spy,
            flatten=True,
            epsilon=torch.tensor(1e-7),
            global_comm_dim_size=None,
        )
    )

    expected_shape = (
        expected_matrix_shape
        if n_params == 1
        else (n_params, *expected_matrix_shape)
    )
    assert seen_shapes == [expected_shape]
    assert len(result) == n_params
    for actual, expected in zip(result, inputs):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # LR adjustment still reads the unstacked parameter shape. Its matrix
    # geometry must agree with the last two dimensions seen by NS.
    rows, cols = expected_matrix_shape
    assert adjust_lr_rms_norm(1.0, param_shape, flatten=True) == pytest.approx(
        0.2 * math.sqrt(max(rows, cols))
    )
    assert adjust_lr_spectral_norm(
        1.0, param_shape, flatten=True
    ) == pytest.approx(math.sqrt(rows / cols))


def _distributed_conv_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device(f"cuda:{rank}")
        n_params = 4
        global_shape = (8, 3, 3, 3)
        local_rows = global_shape[0] // world_size
        inputs = []
        for index in range(n_params):
            full = (
                torch.arange(math.prod(global_shape), device=device)
                .reshape(global_shape)
                .float()
                .add_(1000 * index)
            )
            inputs.append(full.narrow(0, rank * local_rows, local_rows).contiguous())

        seen_shapes = []

        def shape_spy(x, epsilon):
            seen_shapes.append(tuple(x.shape))
            return x

        result = _drain(
            megabatch_orthogonalize_async(
                inputs,
                comm_dim=-4,
                device_rank=rank,
                world_size=world_size,
                process_group=dist.group.WORLD,
                newton_schulz_func=shape_spy,
                flatten=True,
                epsilon=torch.tensor(1e-7, device=device),
                global_comm_dim_size=global_shape[0],
            )
        )

        assert seen_shapes == [(2, global_shape[0], math.prod(global_shape[1:]))]
        assert len(result) == n_params
        for actual, expected in zip(result, inputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_two_gpu_fsdp_like_conv_megabatch_flatten():
    if CUDA_DEVICE_COUNT < 2:
        pytest.skip("needs >= 2 CUDA devices for NCCL all-to-all")
    mp.spawn(
        _distributed_conv_worker,
        args=(2, _find_free_port()),
        nprocs=2,
        join=True,
    )


def polynomial_reference(x, epsilon):
    """FP64 NS oracle with independently normalized trailing matrices."""
    x = x.double()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + epsilon)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.mT
    for _ in range(5):
        gram = x @ x.mT
        x = 1.5 * x - 0.5 * gram @ x
    return x.mT if tall else x


def _orthogonalize(inputs, func=polynomial_reference, flatten=True,
                   return_stacked=False):
    return _drain(megabatch_orthogonalize_async(
        inputs, comm_dim=None, device_rank=0, world_size=1,
        process_group=None, newton_schulz_func=func, flatten=flatten,
        epsilon=torch.tensor(1e-7, dtype=inputs[0].dtype, device=inputs[0].device),
        global_comm_dim_size=None, return_stacked=return_stacked,
    ))


CONV_SHAPES = [
    (8, 4, 3), (16, 2, 1), (8, 4, 2), (8, 1, 7),
    (8, 4, 3, 3), (16, 2, 1, 1), (8, 1, 3, 3),
    (8, 4, 3, 3, 3), (16, 2, 1, 1, 1), (8, 1, 3, 3, 3),
    (8, 3, 3, 3, 4), (16, 1, 1, 1, 2),
]


@pytest.mark.parametrize("shape", CONV_SHAPES)
@pytest.mark.parametrize("n_params", [1, 2, 3, 5])
@pytest.mark.parametrize("return_stacked", [False, True])
def test_numeric_flatten_against_independent_parameters(shape, n_params, return_stacked):
    generator = torch.Generator().manual_seed(91)
    inputs = [torch.randn(shape, generator=generator, dtype=torch.float64) * (i + 1)
              for i in range(n_params)]
    actual = _orthogonalize(inputs, return_stacked=return_stacked)
    expected = [polynomial_reference(x.reshape(shape[0], -1), 1e-7).reshape(shape)
                for x in inputs]
    if return_stacked:
        assert actual.shape == (n_params, *shape)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("shape", CONV_SHAPES)
def test_zero_rank_deficient_noncontiguous_and_layer_independence(shape):
    torch.manual_seed(13)
    x = torch.randn((*shape, 2), dtype=torch.float64)[..., 0]
    assert not x.is_contiguous()
    rank_one = torch.ones_like(x)
    inputs = [x, rank_one, torch.zeros_like(x)]
    result = _orthogonalize(inputs)
    modified = _orthogonalize([x, rank_one * 0.01, torch.zeros_like(x)])
    torch.testing.assert_close(result[0], modified[0], rtol=0, atol=0)
    assert torch.count_nonzero(result[2]) == 0
    for a, b in zip(_orthogonalize(list(reversed(inputs))), reversed(result)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("shape", CONV_SHAPES)
@pytest.mark.parametrize("n_params", [1, 3, 5])
def test_flatten_false_preserves_previous_geometry(shape, n_params):
    inputs = [torch.randn(shape, dtype=torch.float64) for _ in range(n_params)]
    stacked = torch.stack(inputs)
    original_input = inputs[0] if n_params == 1 else stacked
    if original_input.ndim >= 4:
        original_input = original_input.flatten(end_dim=-3)
    expected = polynomial_reference(original_input, 1e-7).reshape(stacked.shape)
    torch.testing.assert_close(_orthogonalize(inputs, flatten=False, return_stacked=True),
                               expected, rtol=0, atol=0)


@pytest.mark.skipif(CUDA_DEVICE_COUNT < 1, reason="CUDA required")
@pytest.mark.parametrize("shape", CONV_SHAPES)
def test_real_polar_express_matches_independent_convolutions(shape, monkeypatch):
    from dion.polar_express import polar_express
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 100)
    torch.manual_seed(42)
    inputs = [torch.randn(shape, device="cuda") for _ in range(3)]
    actual = _orthogonalize(inputs, func=polar_express)
    epsilon = torch.tensor(1e-7, device="cuda")
    explicitly_flattened = torch.stack([x.reshape(shape[0], -1) for x in inputs])
    expected = polar_express(explicitly_flattened, epsilon=epsilon).reshape(3, *shape)
    torch.testing.assert_close(torch.stack(actual), expected, rtol=0, atol=0)
    # Compiled batched/unbatched BF16 graphs have different fusion/rounding.
    # Eager GEMM and BMM can also select different accumulation orders. Only
    # the production batched-vs-batched comparison above is bitwise exact.
    eager = polar_express._torchdynamo_orig_callable
    eager_actual = _orthogonalize(inputs, func=eager)
    for x, a in zip(inputs, eager_actual):
        reference = eager(x.reshape(shape[0], -1), epsilon=epsilon).reshape(shape)
        torch.testing.assert_close(a, reference, rtol=2e-2, atol=2e-3)
        assert torch.isfinite(a).all()


def _replicated_worker(rank, world_size, port, out_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    results = []
    try:
        for shape in ((8, 4), (8, 4, 3), (8, 4, 3, 3), (8, 4, 3, 3, 3)):
            for count in (1, 2, 3, 5):
                torch.manual_seed(93)
                inputs = [torch.randn(shape, dtype=torch.float64) for _ in range(count)]
                result = _drain(megabatch_orthogonalize_async(
                    inputs, comm_dim=None, device_rank=rank, world_size=world_size,
                    process_group=dist.group.WORLD, newton_schulz_func=polynomial_reference,
                    flatten=True, epsilon=torch.tensor(1e-7, dtype=torch.float64),
                    global_comm_dim_size=None, return_stacked=True))
                expected = torch.stack([polynomial_reference(x.reshape(shape[0], -1), 1e-7)
                                        .reshape(shape) for x in inputs])
                results.append((dict(shape=shape, count=count, rank=rank), result, expected))
        torch.save(results, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _check_worker_results(out_dir, world_size):
    for rank in range(world_size):
        for case, actual, expected in torch.load(
            out_dir / f"rank{rank}.pt", weights_only=True
        ):
            torch.testing.assert_close(
                actual, expected, rtol=1e-10, atol=1e-12,
                msg=lambda msg: f"{case}: {msg}",
            )


def test_replicated_distributed_convolution_geometry(tmp_path):
    mp.spawn(_replicated_worker, args=(2, _find_free_port(), str(tmp_path)),
             nprocs=2, join=True)
    _check_worker_results(tmp_path, 2)


@pytest.mark.parametrize("shape", [(5, 7), (7, 5), (5, 5)])
@pytest.mark.parametrize("n_params", [1, 2, 5])
@pytest.mark.parametrize("flatten", [False, True])
def test_linear_parameters_remain_independent_matrices(shape, n_params, flatten):
    generator = torch.Generator().manual_seed(12)
    inputs = [torch.randn(shape, generator=generator, dtype=torch.float64)
              for _ in range(n_params)]
    expected = torch.stack([polynomial_reference(x, 1e-7) for x in inputs])
    actual = _orthogonalize(inputs, flatten=flatten, return_stacked=True)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    if not flatten:
        # Match the pre-fix single/batched dispatch for bitwise comparison.
        historical = (polynomial_reference(inputs[0], 1e-7).unsqueeze(0)
                      if n_params == 1 else
                      polynomial_reference(torch.stack(inputs), 1e-7))
        torch.testing.assert_close(actual, historical, rtol=0, atol=0)


@pytest.mark.parametrize("has_megabatch_dim", [False, True])
def test_row_splitting_is_unchanged(has_megabatch_dim):
    shape = (3, 8, 5) if has_megabatch_dim else (8, 5)
    x = torch.randn(shape, dtype=torch.float64)
    splits, scales = (4, 2, 2), (1.0, 0.5, 0.25)
    expected = torch.cat([
        polynomial_reference(block, 1e-7) * scale
        for block, scale in zip(x.split(splits, dim=-2), scales)
    ], dim=-2)
    actual = muon_update_newton_schulz(
        x, polynomial_reference, flatten=False, epsilon=torch.tensor(1e-7),
        split_sizes=splits, split_scales=scales,
        has_megabatch_dim=has_megabatch_dim,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-12)
    with pytest.raises(AssertionError, match="incompatible"):
        muon_update_newton_schulz(
            x, polynomial_reference, flatten=True, epsilon=torch.tensor(1e-7),
            split_sizes=splits, has_megabatch_dim=has_megabatch_dim,
        )


def _gloo_all_to_all(outputs, inputs, group, async_op):
    """Adapt equal-size list all-to-all to Gloo's supported single-buffer API.

    Data still travels between real processes. Only the collective interface
    is adapted, so the production sharded packing/NS/unpacking runs on CPU.
    """
    send = torch.stack(inputs)
    recv = torch.empty_like(send)
    work = dist.all_to_all_single(recv, send, group=group, async_op=async_op)

    class CopyWork:
        def __init__(self):
            # Keep the async send buffer alive until communication completes.
            self.send = send

        def wait(self):
            work.wait()
            for output, received in zip(outputs, recv):
                output.copy_(received)

    return CopyWork()


def _sharded_cpu_worker(rank, world_size, port, out_dir):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=120))
    original_all_to_all = dist.all_to_all
    dist.all_to_all = _gloo_all_to_all
    results = []
    try:
        # Linear + Conv1d/2d/3d, singleton/non-divisible megabatches, empty
        # row shards, and both output- and input-channel sharding.
        for ndim in (2, 3, 4, 5):
            for rows in (1, 5, 8):
                for axis in (0, 1):
                    # Put the varied size on the actual sharded axis so both
                    # axes cover divisible, uneven and empty shards.
                    shape = ([rows, 4] if axis == 0 else [4, rows])
                    shape = tuple(shape) + (3,) * (ndim - 2)
                    count = 3
                    torch.manual_seed(15)
                    full = [torch.randn(shape, dtype=torch.float64)
                            for _ in range(count)]
                    expected = [polynomial_reference(x.reshape(shape[0], -1), 1e-7)
                                .reshape(shape) for x in full]
                    chunk_size = math.ceil(shape[axis] / world_size)
                    start = min(rank * chunk_size, shape[axis])
                    size = min(chunk_size, shape[axis] - start)
                    local = [x.narrow(axis, start, size).contiguous() for x in full]
                    actual = _drain(megabatch_orthogonalize_async(
                        local, comm_dim=axis - ndim, device_rank=rank,
                        world_size=world_size, process_group=dist.group.WORLD,
                        newton_schulz_func=polynomial_reference, flatten=True,
                        epsilon=torch.tensor(1e-7, dtype=torch.float64),
                        global_comm_dim_size=shape[axis], return_stacked=True,
                    ))
                    reference = torch.stack([x.narrow(axis, start, size) for x in expected])
                    results.append((dict(shape=shape, axis=axis, rank=rank), actual, reference))
        torch.save(results, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.all_to_all = original_all_to_all
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_sharded_distributed_flatten_against_independent_parameters(world_size, tmp_path):
    mp.spawn(_sharded_cpu_worker, args=(world_size, _find_free_port(), str(tmp_path)),
             nprocs=world_size, join=True)
    _check_worker_results(tmp_path, world_size)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("nesterov", [False, True])
def test_muon_conv_modules_match_separate_optimizers(ndim, nesterov, monkeypatch):
    """Exercise optimizer grouping, momentum and weight updates, not just NS."""
    from dion import Muon

    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 128)
    torch.manual_seed(101)
    cls = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)[ndim - 1]
    layers = [cls(3, 5, 3, bias=False).double() for _ in range(2)]
    reference = [torch.nn.Parameter(layer.weight.detach().clone()) for layer in layers]
    options = dict(lr=7e-4, weight_decay=1e-2, nesterov=nesterov,
                   flatten=True, adjust_lr="rms_norm", newton_schulz_func=polynomial_reference)
    grouped = Muon([layer.weight for layer in layers], **options)
    separate = [Muon([p], **options) for p in reference]
    for step in range(3):
        for index, (layer, p) in enumerate(zip(layers, reference)):
            gradient = torch.randn_like(p) * (index + 1 + step)
            layer.weight.grad = gradient.clone()
            p.grad = gradient.clone()
        grouped.step()
        for opt in separate:
            opt.step()
        for layer, p, opt in zip(layers, reference, separate):
            torch.testing.assert_close(layer.weight, p, rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(grouped.state[layer.weight]["momentum"],
                                       opt.state[p]["momentum"], rtol=0, atol=0)


def test_megabatch_axis_flag_is_required():
    with pytest.raises(TypeError, match="has_megabatch_dim"):
        muon_update_newton_schulz(torch.ones(2, 3), polynomial_reference,
                                 flatten=True, epsilon=1e-7)
