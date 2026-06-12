import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from dion import Muon2F

CUDA_DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _identity_ortho(x, epsilon):
    return x


def _row_sharded_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        device = torch.device(f"cuda:{rank}")
        mesh = DeviceMesh("cuda", list(range(world_size)))

        full_param = torch.arange(32, device=device, dtype=torch.float32).view(4, 8)
        full_grad = (
            torch.arange(32, device=device, dtype=torch.float32).view(4, 8) + 1
        ) / 10
        local_param = torch.chunk(full_param, world_size, dim=0)[rank].contiguous()
        local_grad = torch.chunk(full_grad, world_size, dim=0)[rank].contiguous()

        dt_param = DTensor.from_local(
            local_param,
            device_mesh=mesh,
            placements=[Shard(0)],
            shape=full_param.shape,
            stride=full_param.stride(),
        )
        dt_grad = DTensor.from_local(
            local_grad,
            device_mesh=mesh,
            placements=[Shard(0)],
            shape=full_grad.shape,
            stride=full_grad.stride(),
        )

        p = torch.nn.Parameter(dt_param)
        p.grad = dt_grad

        beta2 = 0.9
        opt = Muon2F(
            [p],
            distributed_mesh=mesh,
            lr=0.0,
            weight_decay=0.0,
            adjust_lr=None,
            mu=0.0,
            muon_beta2=beta2,
            newton_schulz_func=_identity_ortho,
        )
        opt.step()

        state = opt.state[p]
        expected_row = (1 - beta2) * (local_grad * local_grad).sum(dim=-1, keepdim=True)
        expected_col = (1 - beta2) * (full_grad * full_grad).sum(dim=-2, keepdim=True)
        torch.testing.assert_close(state["variance_row"], expected_row)
        torch.testing.assert_close(state["variance_col"], expected_col)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason="needs >= 2 CUDA devices")
def test_muon2f_row_sharded_dtensor_syncs_column_stats():
    mp.spawn(_row_sharded_worker, args=(2, 29623), nprocs=2, join=True)
