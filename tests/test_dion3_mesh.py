"""Dion3 multidimensional layouts against its existing 1D update.

CPU tests use real Gloo ranks with the list-alltoall adapter already used by
the megabatch tests. They exercise redistribution, state and update numerics;
the CUDA tests run real FSDP and compiled Triton on one GPU, and native NCCL
collectives on 4/8 GPUs when available.
"""

import copy
import math
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from dion import Dion3
from test_megabatch_flatten import _gloo_all_to_all


def _cases(ndim):
    if ndim == 2:
        return [
            ((7, 11), (Shard(0), Shard(1)), 0),
            ((7, 11), (Shard(1), Shard(0)), 1),
            ((1, 3), (Shard(0), Shard(1)), 0),
            ((7, 11), (Shard(0), Shard(0)), 0),
            ((8, 11), (_StridedShard(0, split_factor=2), Shard(0)), 0),
            ((7, 1), (Replicate(), Shard(1)), None),
            ((7, 11), (Replicate(), Shard(0)), 1),
        ]
    return [
        ((7, 11), (Replicate(), Shard(0), Shard(1)), 1),
        ((7, 11), (Shard(0), Shard(0), Shard(1)), 0),
        ((3, 7, 11), (Shard(0), Shard(1), Shard(2)), 1),
        ((3, 1, 3), (Shard(0), Shard(1), Shard(2)), 1),
        ((1, 7, 11), (Shard(0), Shard(1), Shard(2)), 1),
    ]


def _mesh_worker(rank, ndim, rendezvous, backend="gloo"):
    torch.set_num_threads(1)
    torch.compiler.set_stance("force_eager")
    device_type = "cuda" if backend == "nccl" else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        init_method=rendezvous,
        rank=rank,
        world_size=2**ndim,
        timeout=timedelta(seconds=180),
    )
    try:
        mesh = init_device_mesh(
            device_type,
            (2,) * ndim,
            mesh_dim_names=tuple(f"axis{i}" for i in range(ndim)),
        )
        collective_context = (
            patch.object(dist, "all_to_all", _gloo_all_to_all)
            if backend == "gloo"
            else nullcontext()
        )
        with collective_context:
            for shape, placements, row_axis in _cases(ndim):
                for scope in ("local", "global"):
                    for fraction in (0.5, 1.0):
                        _compare_layout(
                            mesh, shape, placements, row_axis, scope, fraction
                        )
            _fsdp_tp_training(dp_size=2 ** (ndim - 2), device_type=device_type)
            if ndim == 3:
                _reject_incomplete_mesh(mesh)
    finally:
        dist.destroy_process_group()


def _compare_layout(mesh, shape, placements, row_axis, scope, fraction):
    torch.manual_seed(104)
    initial = [torch.randn(shape).to(mesh.device_type) for _ in range(3)]
    params = [
        torch.nn.Parameter(distribute_tensor(t, mesh, placements)) for t in initial
    ]
    # Full rows/columns with only batch shards retained: the independent 1D
    # reference receives exactly this subset of matrices on each batch rank.
    batch_layout = tuple(
        p if p.is_shard() and p.dim < len(shape) - 2 else Replicate()
        for p in placements
    )
    ref_mesh = mesh[f"axis{row_axis}"] if row_axis is not None else None

    def reference_tensor(t):
        batch = distribute_tensor(t, mesh, batch_layout).to_local()
        if ref_mesh is not None:
            return distribute_tensor(batch, ref_mesh, [Shard(len(shape) - 2)])
        return batch.clone()

    refs = [torch.nn.Parameter(reference_tensor(t)) for t in initial]
    kwargs = dict(lr=0.02, fraction=fraction, selection_scope=scope)
    opt = Dion3(params, distributed_mesh=mesh, **kwargs)
    ref_opt = Dion3(refs, distributed_mesh=ref_mesh, **kwargs)
    variance_layout = tuple(
        Replicate() if p.is_shard() and p.dim == len(shape) - 1 else p
        for p in placements
    )
    for step in range(3):
        torch.manual_seed(200 + step)
        for p, r in zip(params, refs):
            grad = torch.randn(shape).to(mesh.device_type)
            p.grad = distribute_tensor(grad, mesh, placements)
            r.grad = reference_tensor(grad)
        opt.step()
        if (
            math.prod(
                refs[0].to_local().shape[:-2]
                if isinstance(refs[0], DTensor)
                else refs[0].shape[:-2]
            )
            != 0
        ):
            ref_opt.step()
        for p, r in zip(params, refs):
            assert p.placements == placements
            assert opt.state[p]["momentum"].placements == placements
            assert opt.state[p]["variance_neuron"].placements == variance_layout
            for actual, expected in [(p, r)] + [
                (opt.state[p][key], ref_opt.state[r][key])
                for key in ("momentum", "variance_neuron")
            ]:
                actual = actual.redistribute(placements=batch_layout).to_local()
                expected = (
                    expected.full_tensor()
                    if isinstance(expected, DTensor)
                    else expected
                )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        if step == 1:
            # Preserve state through optimizer reconstruction and resume.
            saved = copy.deepcopy(opt.state_dict())
            opt = Dion3(params, distributed_mesh=mesh, **kwargs)
            opt.load_state_dict(saved)


def _reject_incomplete_mesh(mesh):
    p = torch.nn.Parameter(
        distribute_tensor(
            torch.ones(4, 6, device=mesh.device_type),
            mesh,
            [Replicate(), Shard(0), Shard(1)],
        )
    )
    p.grad = torch.ones_like(p)
    opt = Dion3([p], distributed_mesh=mesh["axis0", "axis1"])
    before = p.to_local().clone()
    with pytest.raises(ValueError, match="every sharded"):
        opt.step()
    torch.testing.assert_close(p.to_local(), before, rtol=0, atol=0)


def _fsdp_tp_training(dp_size, device_type="cpu"):
    """Exercise real FSDP2 + TP forward/backward and train.py dispatch."""
    import argparse
    import train
    from models.gpt_model import GPT, GPTConfig, parallelize_gpt_model

    # Give the existing two axes the names used by the training entry point.
    root = init_device_mesh(
        device_type, (dp_size, 2, 2), mesh_dim_names=("dp", "fs", "tp")
    )
    torch.manual_seed(42)
    model = GPT(
        GPTConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_embd=16)
    )
    model.init_weights()
    model.to(device_type)
    parallelize_gpt_model(model, root)
    hp = train.Hyperparameters(optimizer="dion3", scalar_opt="adamw", model_dim=16)
    flags = argparse.Namespace(
        no_triton=True, use_polar_express=True, use_gram_newton_schulz=False
    )
    opt = train.init_optimizer(model, root, None, hp, flags)
    before = [p.full_tensor().clone() for p in model.parameters()]
    for _ in range(3):
        x = (
            torch.arange(16, device=device_type).view(2, 8)
            + root["dp"].get_local_rank()
        ) % 32
        loss = model(x, (x + 1) % 32)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert any(
        not torch.equal(b, p.full_tensor()) for b, p in zip(before, model.parameters())
    )
    for p in model.parameters():
        local = p.to_local().contiguous()
        replicas = [torch.empty_like(local) for _ in range(dp_size)]
        dist.all_gather(replicas, local, group=root["dp"].get_group())
        for replica in replicas:
            torch.testing.assert_close(replica, local, rtol=0, atol=0)


@pytest.mark.parametrize("ndim", [2, 3])
def test_multidimensional_mesh_matches_1d_reference(ndim, tmp_path):
    mp.spawn(
        _mesh_worker,
        args=(ndim, (tmp_path / "rdzv").as_uri()),
        nprocs=2**ndim,
        join=True,
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_multidimensional_mesh_nccl(ndim, tmp_path):
    if torch.cuda.device_count() < 2**ndim:
        pytest.skip(f"needs {2**ndim} GPUs for native NCCL")
    mp.spawn(
        _mesh_worker,
        args=(ndim, (tmp_path / "rdzv").as_uri(), "nccl"),
        nprocs=2**ndim,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ndim", [2, 3])
def test_multidimensional_mesh_fsdp_cuda(ndim, tmp_path):
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=(tmp_path / "rdzv").as_uri(), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh(
            "cuda", (1,) * ndim, mesh_dim_names=tuple(f"axis{i}" for i in range(ndim))
        )
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32, bias=False),
        ).cuda()
        fully_shard(model, mesh=mesh["axis0"])
        opt = Dion3(
            model.parameters(),
            distributed_mesh=mesh,
            use_triton=True,
            triton_post_ortho=True,
        )
        before = [p.full_tensor().clone() for p in model.parameters()]
        for _ in range(3):
            loss = model(torch.randn(4, 32, device="cuda")).square().mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            assert torch.isfinite(loss)
        assert all(
            not torch.equal(b, p.full_tensor())
            for b, p in zip(before, model.parameters())
        )
    finally:
        dist.destroy_process_group()
