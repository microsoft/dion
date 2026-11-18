"""NCCL receive buffers must be contiguous even when orthogonalization is not."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from dion.megabatch_base import megabatch_orthogonalize_async


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("return_stacked", [False, True])
def test_all_gather_with_noncontiguous_orthogonalization(tmp_path, return_stacked):
    # One NCCL rank is sufficient to enforce the same layout requirements as
    # multi-GPU DDP. Two parameters select the megabatched all-gather path.
    dist.init_process_group(
        "nccl",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        inputs = [torch.randn(16, 32, device="cuda") for _ in range(2)]

        def noncontiguous_result(x, epsilon):
            # Preserve values and shape while returning a transposed layout,
            # as orthogonalization can do when it works on the shorter axis.
            result = x.transpose(-2, -1).contiguous().transpose(-2, -1)
            assert not result.is_contiguous()
            return result

        task = megabatch_orthogonalize_async(
            inputs,
            comm_dim=None,
            device_rank=0,
            world_size=1,
            process_group=dist.group.WORLD,
            newton_schulz_func=noncontiguous_result,
            flatten=False,
            epsilon=torch.tensor(1e-7, device="cuda"),
            global_comm_dim_size=None,
            return_stacked=return_stacked,
        )
        while True:
            try:
                next(task)
            except StopIteration as stop:
                result = stop.value
                break

        if return_stacked:
            torch.testing.assert_close(result, torch.stack(inputs))
        else:
            assert len(result) == len(inputs)
            for actual, expected in zip(result, inputs):
                torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()
