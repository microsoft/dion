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


@pytest.mark.skipif(CUDA_DEVICE_COUNT < 4, reason="needs >= 4 CUDA devices for NCCL alltoall")
@pytest.mark.parametrize(
    "global_dim_0, world_size, n_params",
    [
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
    # Unique port per parametrization to avoid bind collisions.
    port = 29500 + global_dim_0 * 100 + world_size
    mp.spawn(
        _worker,
        args=(world_size, global_dim_0, 16, n_params, port),
        nprocs=world_size,
        join=True,
    )
