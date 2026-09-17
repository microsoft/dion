"""Optimizer-level conv flatten coverage with real collectives and FSDP2.

NorMuon compares against an equivalent 2D matrix under the SAME row sharding:
its norm-preserving rescale intentionally remains local to each shard.
"""

import copy
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DeviceMesh, DTensor, Shard, distribute_tensor

from dion import Muon, NorMuon
from test_megabatch_flatten import _gloo_all_to_all
from test_normuon_conv_flatten import polynomial_ns


CUDA_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
# Each dimensionality gets divisible and uneven/empty shards. Count=1 pins
# singleton dispatch, count=5 requires megabatch padding on both world sizes.
CASES = [(1, 8, 1), (1, 5, 5), (2, 8, 3),
         (2, 1, 5), (3, 8, 3), (3, 5, 5)]


@pytest.fixture(scope="module", autouse=True)
def isolated_compiler_state():
    # Isolate parent-process CUDA-graph cases; spawned workers are already fresh.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup(rank, world, port, backend):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    torch._dynamo.config.cache_size_limit = 128  # Worker-local; process exits after test.
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world,
                            timeout=timedelta(seconds=180))


def _options():
    return dict(lr=7e-4, weight_decay=1e-2, nesterov=True,
                flatten=True, adjust_lr="rms_norm", newton_schulz_func=polynomial_ns)


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def _record(results, label, params, refs, opt, ref_opt):
    for index, (p, q) in enumerate(zip(params, refs)):
        pairs = [("weight", p, q)]
        for key in ("momentum", "variance_neuron"):
            if key in opt.state[p]:
                pairs.append((key, opt.state[p][key], ref_opt.state[q][key]))
        for key, actual, expected in pairs:
            actual, expected = _local(actual).detach(), _local(expected).detach()
            actual = actual.reshape(expected.shape)
            # Save local tensors on EVERY rank, including empty shards. This
            # avoids a rank-0-only full_tensor() collective and keeps errors in
            # the parent where the complete case label can be shown.
            results.append((f"{label}, param={index}, {key}",
                            actual.cpu().clone(), expected.cpu().clone()))


def _assert_results(directory, world, rtol=1e-10, atol=1e-12):
    for rank in range(world):
        results = torch.load(directory / f"rank{rank}.pt", weights_only=True)
        assert results
        for label, actual, expected in results:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol,
                                       msg=lambda msg: f"rank={rank}, {label}: {msg}")
            assert torch.isfinite(actual).all(), (rank, label)


def _cpu_worker(rank, world, port, out_dir, sharded):
    _setup(rank, world, port, "gloo")
    old_all_to_all = dist.all_to_all
    if sharded:
        # Gloo's single-buffer API transports real data; native list-alltoall
        # remains covered authoritatively by the NCCL/FSDP2 tests below.
        dist.all_to_all = _gloo_all_to_all
    results = []
    try:
        mesh = DeviceMesh("cpu", list(range(world)))
        for cls in (Muon, NorMuon):
            for ndim, rows, count in CASES:
                torch.manual_seed(81)
                shape = (rows, 3) + (3,) * ndim
                weights = [torch.randn(shape, dtype=torch.float64) for _ in range(count)]
                params, refs = [], []
                for w in weights:
                    p, q = w.clone(), w.flatten(1).clone()
                    if sharded:
                        p = distribute_tensor(p, mesh, [Shard(0)])
                        q = distribute_tensor(q, mesh, [Shard(0)])
                    params.append(nn.Parameter(p))
                    refs.append(nn.Parameter(q))
                opt = cls(params, distributed_mesh=mesh, **_options())
                ref_opt = cls(refs, distributed_mesh=mesh if sharded else None, **_options())
                for step in range(3):
                    for i, (p, q) in enumerate(zip(params, refs)):
                        g = torch.randn(shape, dtype=torch.float64) * (i + step + 1)
                        if sharded:
                            p.grad = distribute_tensor(g, mesh, [Shard(0)])
                            q.grad = distribute_tensor(g.flatten(1), mesh, [Shard(0)])
                        else:
                            p.grad, q.grad = g, g.flatten(1)
                    opt.step()
                    ref_opt.step()
                    _record(results, f"{cls.__name__}, {ndim=}, {rows=}, {count=}, {step=}",
                            params, refs, opt, ref_opt)
        torch.save(results, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.all_to_all = old_all_to_all
        dist.destroy_process_group()


@pytest.mark.parametrize("sharded,world", [(False, 2), (True, 2), (True, 4)])
def test_distributed_optimizer_conv_matrix_equivalence(sharded, world, tmp_path):
    mp.spawn(_cpu_worker, args=(world, _port(), str(tmp_path), sharded),
             nprocs=world, join=True)
    _assert_results(tmp_path, world)


def _column_guard_worker(rank, world, port, out_dir):
    _setup(rank, world, port, "gloo")
    try:
        mesh = DeviceMesh("cpu", list(range(world)))
        errors = []
        for shape in [(8, 4, 3), (8, 4, 3, 3), (8, 4, 3, 3, 3)]:
            for axis in range(1, len(shape)):
                p = nn.Parameter(distribute_tensor(torch.zeros(shape), mesh, [Shard(axis)]))
                try:
                    NorMuon([p], distributed_mesh=mesh, **_options())
                except NotImplementedError as exc:
                    errors.append((shape, axis, str(exc)))
                else:
                    errors.append((shape, axis, "NOT REJECTED"))
        torch.save(errors, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_normuon_rejects_all_convolution_column_shards(tmp_path):
    mp.spawn(_column_guard_worker, args=(2, _port(), str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        errors = torch.load(tmp_path / f"rank{rank}.pt", weights_only=True)
        assert len(errors) == 9
        for shape, axis, error in errors:
            assert "flatten=True" in error and f"dim {axis}" in error, (shape, axis, error)


class ConvolutionGroup(nn.Module):
    def __init__(self, ndim, rows, count):
        super().__init__()
        cls = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[ndim - 1]
        self.layers = nn.ModuleList([cls(3, rows, 3, bias=False) for _ in range(count)])

    def forward(self, x):
        return sum((i + 1) * layer(x).float().square().mean()
                   for i, layer in enumerate(self.layers))


def _fsdp_worker(rank, world, port, out_dir, optimizer_name):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    _setup(rank, world, port, "nccl")
    results = []
    try:
        mesh = DeviceMesh("cuda", list(range(world)))
        cls = {"Muon": Muon, "NorMuon": NorMuon}[optimizer_name]
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for ndim, rows, count in CASES:
            torch.manual_seed(91)
            model = ConvolutionGroup(ndim, rows, count).cuda()
            refs = [nn.Parameter(distribute_tensor(layer.weight.detach().flatten(1).clone(),
                                                  mesh, [Shard(0)])) for layer in model.layers]
            for layer in model.layers:
                fully_shard(layer, mesh=mesh, mp_policy=policy)
            fully_shard(model, mesh=mesh, mp_policy=policy)
            params = [layer.weight for layer in model.layers]
            opt = cls(params, distributed_mesh=mesh, **_options())
            ref_opt = cls(refs, distributed_mesh=mesh, **_options())
            for step in range(3):
                opt.zero_grad(set_to_none=True)
                ref_opt.zero_grad(set_to_none=True)
                # Different data per rank exercises real FSDP gradient reduction.
                torch.manual_seed(100 + rank + step * world)
                model(torch.randn((2, 3) + (5,) * ndim, device="cuda")).backward()
                for p, q in zip(params, refs):
                    # Every rank participates in this gather. Feed the SAME
                    # reduced gradient to the equivalently sharded matrix oracle.
                    full_grad = p.grad.full_tensor()
                    q.grad = distribute_tensor(full_grad.flatten(1), mesh, [Shard(0)])
                opt.step()
                ref_opt.step()
                _record(results, f"{ndim=}, {rows=}, {count=}, {step=}",
                        params, refs, opt, ref_opt)
                # Exercise optimizer-state reload while it is still sharded.
                if step == 1:
                    opt.load_state_dict(copy.deepcopy(opt.state_dict()))
                    ref_opt.load_state_dict(copy.deepcopy(ref_opt.state_dict()))
        torch.save(results, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("optimizer_name", ["Muon", "NorMuon"])
def test_real_fsdp2_conv_matrix_equivalence(world, optimizer_name, tmp_path):
    if CUDA_COUNT < world:
        pytest.skip(f"requires {world} CUDA devices")
    mp.spawn(_fsdp_worker, args=(world, _port(), str(tmp_path), optimizer_name),
             nprocs=world, join=True)
    _assert_results(tmp_path, world, rtol=2e-5, atol=2e-7)


@pytest.mark.skipif(CUDA_COUNT < 1, reason="CUDA required")
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_normuon_production_polar_express_and_cuda_graph(ndim, monkeypatch):
    from dion.cuda_graph import CudaGraphOptimizer

    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 128)
    torch.manual_seed(52)
    shape = (8, 3) + (3,) * ndim
    eager_params = [nn.Parameter(torch.randn(shape, device="cuda")) for _ in range(3)]
    graph_params = [nn.Parameter(p.detach().clone()) for p in eager_params]
    matrix_params = [nn.Parameter(p.detach().flatten(1).clone()) for p in eager_params]
    options = _options()
    del options["newton_schulz_func"]  # Real compiled Polar Express.
    eager = NorMuon(eager_params, **options)
    graph_opt = NorMuon(graph_params, **options)
    matrix = NorMuon(matrix_params, **options)
    wrapper = CudaGraphOptimizer(graph_opt, warmup_steps=3)
    for p in eager_params + graph_params + matrix_params:
        p.grad = torch.zeros_like(p)
    try:
        for step in range(8):
            for a, b, c in zip(eager_params, graph_params, matrix_params):
                gradient = torch.randn_like(a)
                a.grad.copy_(gradient)
                b.grad.copy_(gradient)
                c.grad.copy_(gradient.flatten(1))
            eager.step()
            wrapper.step()
            matrix.step()
            for a, b, c in zip(eager_params, graph_params, matrix_params):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
                torch.testing.assert_close(a.flatten(1), c, rtol=2e-5, atol=2e-7)
                for key in ("momentum", "variance_neuron"):
                    torch.testing.assert_close(eager.state[a][key], graph_opt.state[b][key],
                                               rtol=2e-5, atol=2e-7)
                    torch.testing.assert_close(eager.state[a][key].flatten(1), matrix.state[c][key],
                                               rtol=2e-5, atol=2e-7)
    finally:
        wrapper.release()
