import torch
import torch.distributed as dist
from collections import defaultdict
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Union

from .newton_schulz_triton import (
    TRITON_AVAILABLE,
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)
from .opt_utils import AsyncRuntime, AsyncTask, to_local
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async
from .muon import muon_update_newton_schulz, adjust_lr_spectral_norm, adjust_lr_rms_norm


class DistributedOrthoBase(Optimizer):
    """
    Shared base class for distributed orthogonalization optimizers (NorMuon, Dion2).
    Handles distributed setup, Newton-Schulz config, step orchestration,
    shard detection, and scalar optimizer tasks (Lion, AdamW).

    Subclasses must implement ``_create_ortho_tasks()``.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]],
        algo_name: str,
        defaults: dict,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
    ):
        super().__init__(params, defaults)
        self._algo_name = algo_name

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh supported, but got {distributed_mesh.ndim}D. "
                    f"For HSDP, provide the 1D sharded sub-mesh."
                )
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError(
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. "
                f"Expected DeviceMesh or ProcessGroup."
            )
        self._distributed_mesh = distributed_mesh

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            if not TRITON_AVAILABLE:
                raise ImportError(
                    "use_triton=True requires the 'triton' package, which is not installed. "
                    "Install it with: pip install dion[triton]  (or: pip install triton)"
                )
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ortho_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            group["step"] += 1
            algo = group["algorithm"]
            if algo == self._algo_name:
                ortho_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        ortho_tasks = self._create_ortho_tasks(ortho_groups)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(ortho_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """Get optimizer state, or lazy-initialize if it doesn't exist."""
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _get_shard_info(self, param: Tensor, group: dict):
        """Determine sharding info. Returns (is_batch_sharded, is_matrix_sharded, sharded_tensor_dim)."""
        is_batch_sharded = False
        is_matrix_sharded = False
        sharded_tensor_dim = None

        if not isinstance(param, DTensor):
            return is_batch_sharded, is_matrix_sharded, sharded_tensor_dim

        if not isinstance(self._distributed_mesh, DeviceMesh):
            raise RuntimeError(
                "Must create optimizer with DeviceMesh if using DTensor parameters."
            )

        shard_placements = [
            (i, p)
            for i, p in enumerate(param.placements)
            if p.is_shard() and param.device_mesh.size(i) > 1
        ]

        if not group["flatten"]:
            matrix_dims = {param.ndim - 1, param.ndim - 2}
            is_batch_sharded = any(
                p.dim not in matrix_dims for _, p in shard_placements
            )
            shard_placements = [
                (i, p) for i, p in shard_placements if p.dim in matrix_dims
            ]

        if len(shard_placements) == 1:
            is_matrix_sharded = True
            sharded_mesh_dim = shard_placements[0][0]
            sharded_tensor_dim = shard_placements[0][1].dim

            if (
                param.device_mesh.get_group(sharded_mesh_dim)
                != self._process_group
            ):
                raise RuntimeError(
                    f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} "
                    f"different from the optimizer's device mesh. "
                    f"DTensor has mesh: {param.device_mesh}, placements: {param.placements}, "
                    f"but optimizer was created with mesh: {self._distributed_mesh}."
                )
        elif len(shard_placements) > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support parameters with multiple sharded dimensions."
            )

        return is_batch_sharded, is_matrix_sharded, sharded_tensor_dim

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """Subclasses implement this to create orthogonalization tasks."""
        raise NotImplementedError

    def _create_lion_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == "lion"
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "lion") for p in params]
            momentums = [s["momentum"] for s in states]

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                    cautious_wd=group.get("cautious_wd", False),
                )
            )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == "adamw"
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                    step=torch.tensor(group["step"]),
                    epsilon=torch.tensor(group["epsilon"]),
                    cautious_wd=group.get("cautious_wd", False),
                )
            )


def megabatch_orthogonalize_async(
    U: List[Tensor],
    comm_dim: Optional[int],
    device_rank: int,
    world_size: int,
    process_group: Optional[ProcessGroup],
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Generator[None, None, List[Tensor]]:
    """
    Shared megabatch communication + Newton-Schulz orthogonalization.

    This is a generator that yields at async communication points and uses
    ``return value`` to pass the result back to the caller. In Python, ``return``
    inside a generator raises ``StopIteration(value)``, and the caller recovers
    the value via ``result = yield from megabatch_orthogonalize_async(...)``.
    The ``yield from`` transparently forwards intermediate yields to AsyncRuntime.

    Args:
        U: List of tensors to orthogonalize (all same shape).
        comm_dim: Dimension for cat/split in all-to-all (negative index).
            None for non-sharded parameters.
        device_rank: This device's rank.
        world_size: Total number of devices.
        process_group: Distributed process group. None for single-GPU.
        newton_schulz_func: Newton-Schulz orthogonalization function.
        flatten: Whether to flatten 3D+ tensors to 2D.
        epsilon: Small value for numerical stability.
    """
    N = len(U)

    # Pad to divisible by world_size (needed by both distributed paths)
    if process_group is not None and N > 1:
        pad_n = (world_size - N % world_size) % world_size
        U_work = U + [torch.zeros_like(U[0])] * pad_n if pad_n > 0 else U
        N_total = len(U_work)
        per_rank = N_total // world_size
    else:
        U_work = U

    if comm_dim is not None and process_group is not None:
        # --- Mega-batched sharded FSDP2 path ---
        input_chunks = [
            torch.stack(U_work[r * per_rank : (r + 1) * per_rank])
            for r in range(world_size)
        ]

        output_chunks = [torch.empty_like(c) for c in input_chunks]
        work = dist.all_to_all(
            output_chunks, input_chunks, group=process_group, async_op=True
        )
        yield
        work.wait()

        # comm_dim is negative, so it correctly indexes the stacked tensor
        full_matrices = torch.cat(output_chunks, dim=comm_dim)
        full_matrices = muon_update_newton_schulz(
            full_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        split_chunks = [
            s.contiguous()
            for s in torch.tensor_split(full_matrices, world_size, dim=comm_dim)
        ]

        recv_chunks = [torch.empty_like(c) for c in split_chunks]
        work = dist.all_to_all(
            recv_chunks, split_chunks, group=process_group, async_op=True
        )
        yield
        work.wait()

        result = [recv_chunks[r][i] for r in range(world_size) for i in range(per_rank)]
        return result[:N]

    elif N > 1 and process_group is not None:
        # --- Mega-batched non-sharded path ---
        start = device_rank * per_rank
        my_matrices = torch.stack(U_work[start : start + per_rank])
        my_matrices = muon_update_newton_schulz(
            my_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        all_chunks = [torch.empty_like(my_matrices) for _ in range(world_size)]
        work = dist.all_gather(
            all_chunks, my_matrices.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

        result = [all_chunks[r][i] for r in range(world_size) for i in range(per_rank)]
        return result[:N]

    elif N == 1:
        return [
            muon_update_newton_schulz(
                U[0],
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
            )
        ]

    else:
        # N > 1, no process_group (single GPU or batch-sharded 3D)
        stacked = torch.stack(U)
        stacked = muon_update_newton_schulz(
            stacked,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        return [stacked[i] for i in range(N)]
