"""Regression test: ``megabatch_orthogonalize_async`` must not hang when some
ranks hold empty (numel=0) FSDP2 padding-only shards.

Background. FSDP2's ``_chunk_with_empty(t, world_size, dim=shard_dim)`` produces
fewer than ``world_size`` non-empty chunks when the sharded global dim is
smaller than ``world_size`` or doesn't divide evenly to fill all ranks (e.g.
shape ``(18, D)`` over ``world_size=8`` gives 6 real chunks of ``(3, D)`` plus
2 empty ``(0, D)`` chunks). The optimizer step then sees per-rank local shards
of differing sizes; ``torch.stack(...)`` and the subsequent
``dist.all_to_all`` produce mismatched per-pair sizes and hang at NCCL.
"""

import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dion.megabatch_base import megabatch_orthogonalize_async
from dion.opt_utils import AsyncRuntime, AsyncTask
from dion.polar_express import polar_express


CUDA_DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _ns_func(x, epsilon=1e-7):
    return polar_express(x, epsilon=epsilon)


def _worker(rank: int, world_size: int, global_dim_0: int, dim_1: int, n_params: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    pg = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")

    # Mimic FSDP2 contiguous chunking on a (global_dim_0, dim_1) param.
    # FSDP2 narrows the per-rank shard back to 2D shape (k, dim_1) where k>=0
    # and k=0 for padding-only ranks (see _init_sharded_param in
    # torch/distributed/fsdp/_fully_shard/_fsdp_param.py).
    full = torch.arange(global_dim_0 * dim_1, device=device).view(global_dim_0, dim_1).float()
    real_chunks = list(torch.chunk(full, world_size, dim=0))
    if rank < len(real_chunks):
        local_shape = real_chunks[rank].shape
    else:
        local_shape = torch.Size([0, dim_1])

    U = []
    for i in range(n_params):
        if local_shape[0] == 0:
            U.append(torch.zeros(local_shape, dtype=torch.float32, device=device))
        else:
            g = torch.Generator(device=device).manual_seed(rank * 17 + i)
            U.append(torch.randn(local_shape, dtype=torch.float32, device=device, generator=g))

    state = {}

    def _task_gen():
        result = yield from megabatch_orthogonalize_async(
            U,
            comm_dim=-2,
            device_rank=rank,
            world_size=world_size,
            process_group=pg,
            newton_schulz_func=_ns_func,
            flatten=False,
            epsilon=torch.tensor(1e-7, device=device),
            global_comm_dim_size=global_dim_0,
        )
        state["result"] = result

    runtime = AsyncRuntime((t for t in [AsyncTask(_task_gen())]), max_concurrent_tasks=1)
    runtime.run()

    assert len(state["result"]) == n_params
    for t in state["result"]:
        assert t.shape == local_shape, (
            f"rank {rank}: expected {local_shape}, got {tuple(t.shape)}"
        )

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "global_dim_0, world_size, n_params",
    [
        # world_size=2 cases (run on 2-GPU dev boxes too):
        # 1 row over 2 ranks: rank 0 has (1, D), rank 1 has (0, D) — empty shard.
        (1, 2, 8),
        # 3 rows over 2 ranks: (2, D) + (1, D) — non-divisible, no empty.
        (3, 2, 8),
        # 5 rows over 4 ranks: ceil(5/4)=2 chunks of size 2 fill ranks 0..2, rank 3 empty.
        # Mirrors the sparse-3b (18, D) over 8 ranks pattern at smaller scale.
        (5, 4, 8),
        # Smaller-than-world-size: 2 rows, 4 ranks: ranks 2 and 3 empty.
        (2, 4, 8),
        # Non-divisible, no empty ranks (sanity: still works after fix).
        (15, 4, 8),
    ],
)
def test_megabatch_orthogonalize_async_handles_empty_shards(global_dim_0, world_size, n_params):
    if CUDA_DEVICE_COUNT < world_size:
        pytest.skip(f"needs >= {world_size} CUDA devices for NCCL alltoall")
    # Unique port per parametrization to avoid bind collisions.
    port = 29500 + global_dim_0 * 100 + world_size
    mp.spawn(
        _worker,
        args=(world_size, global_dim_0, 16, n_params, port),
        nprocs=world_size,
        join=True,
    )


def test_megabatch_orthogonalize_async_requires_global_comm_dim_size():
    # Pins the explicit-exception contract: the sharded branch must raise
    # ValueError when global_comm_dim_size is omitted, not assert (which
    # vanishes under ``python -O``). Uses a single-rank gloo PG so this
    # runs CPU-only and stays out of the GPU test budget.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29401")
    already_init = dist.is_initialized()
    if not already_init:
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        U = [torch.zeros(2, 4) for _ in range(4)]
        gen = megabatch_orthogonalize_async(
            U,
            comm_dim=-2,
            device_rank=0,
            world_size=1,
            process_group=dist.group.WORLD,
            newton_schulz_func=_ns_func,
            flatten=False,
            epsilon=torch.tensor(1e-7),
            global_comm_dim_size=None,
        )
        with pytest.raises(ValueError, match="global_comm_dim_size"):
            next(gen)
    finally:
        if not already_init and dist.is_initialized():
            dist.destroy_process_group()
