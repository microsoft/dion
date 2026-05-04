import math
import torch
import torch.distributed as dist
from collections import defaultdict
from itertools import chain
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .newton_schulz_triton import (
    TRITON_AVAILABLE,
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)
from .polar_express import polar_express, polar_express_triton
from .opt_utils import AsyncRuntime, AsyncTask, to_local
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async


def canonical_normalization(
    normalization: Optional[str],
    allow_none: bool = False,
) -> Optional[str]:
    """Validate and canonicalize an update-normalization mode string.

    Accepts ``'neuron'``/``'short_axis'`` (case-insensitive). When
    ``allow_none=True``, also accepts ``None`` and the string ``'None'``,
    returning Python ``None`` for both. Raises ``ValueError`` otherwise.
    """
    if normalization is None or (
        isinstance(normalization, str) and normalization.lower() == "none"
    ):
        if allow_none:
            return None
        raise ValueError(
            f"Invalid normalization: {normalization}. Must be 'neuron' or 'short_axis'."
        )
    if isinstance(normalization, str):
        value = normalization.lower()
        if value in ("neuron", "short_axis"):
            return value
    valid = (
        "None, 'None', 'neuron', or 'short_axis'"
        if allow_none
        else "'neuron' or 'short_axis'"
    )
    raise ValueError(f"Invalid normalization: {normalization}. Must be {valid}.")


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
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
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

        # Orthogonalization function configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(
                    f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}"
                )
            self._newton_schulz_func = newton_schulz_func
        elif use_gram_newton_schulz:
            try:
                from gram_newton_schulz import GramNewtonSchulz
            except ImportError:
                raise ImportError(
                    "use_gram_newton_schulz=True requires the 'gram-newton-schulz' package, "
                    "which is not installed. "
                    "Install it with: pip install gram-newton-schulz"
                )
            use_polar_express = True
            _gns = GramNewtonSchulz(
                ns_use_kernels=use_triton,
                use_gram_newton_schulz=True,
                gram_newton_schulz_reset_iterations=[2],
                # Some compiler crashes were observed with mode="reduce-overhead" when we also compile the entire optimizer step.
                compile_kwargs=dict(fullgraph=True, mode="default"),
            )
            self._newton_schulz_func = lambda X, epsilon=None: _gns(X)
        elif use_polar_express and use_triton:
            self._newton_schulz_func = polar_express_triton
        elif use_polar_express:
            self._newton_schulz_func = polar_express
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

    def _resolve_num_heads(self, group: dict) -> Optional[int]:
        """Validate the ``num_heads`` option on a param group.

        Returns the group's ``num_heads`` when it is set and > 1 (the only case
        that actually triggers the per-head code path). Returns ``None`` when
        ``num_heads`` is unset or equals 1 (both are no-ops). Raises
        ``ValueError`` for invalid values or incompatible combinations.
        """
        num_heads = group.get("num_heads")
        if num_heads is None:
            return None
        # bool is a subclass of int in Python; reject it explicitly.
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError(
                f"num_heads must be a positive integer if set, got {num_heads!r}."
            )
        if num_heads == 1:
            return None
        if group.get("flatten"):
            raise ValueError(
                "num_heads > 1 is incompatible with flatten=True: flattening "
                "the per-head 3D view collapses heads back into a single 2D "
                "matrix, defeating per-head Newton-Schulz."
            )
        return num_heads

    def _prepare_head_split(
        self,
        num_heads: int,
        params: List[Tensor],
        *extras: List[Tensor],
    ) -> Tuple[List[Tensor], ...]:
        """Reshape 2D params (and same-dim-0 companion tensors) into 3D per-head views.

        A 2D weight of shape ``(num_heads * head_dim, ...)`` is returned as
        a 3D local tensor of shape ``(num_heads_local, head_dim, ...)``. The same
        split is applied to any ``extras`` lists (grads, momentums, and NorMuon's
        per-neuron variance buffer of shape ``(out, 1)``).

        In-place updates on the returned views propagate to the underlying storage.
        Callers must also mark the resulting tensors as batch-sharded (skip NS
        all-to-all) since each rank's shard now holds whole heads.
        """
        first = params[0]
        full_shape = first.shape
        if first.ndim != 2:
            raise ValueError(
                f"num_heads is only supported for 2D parameters, got shape {tuple(full_shape)}."
            )
        if full_shape[0] % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must divide dim 0 of the parameter "
                f"(got shape {tuple(full_shape)})."
            )
        head_dim = full_shape[0] // num_heads

        if isinstance(first, DTensor):
            shard_placements = [
                (i, p)
                for i, p in enumerate(first.placements)
                if p.is_shard() and first.device_mesh.size(i) > 1
            ]
            if any(p.dim != 0 for _, p in shard_placements):
                raise NotImplementedError(
                    f"num_heads requires sharding on dim 0 (the heads dim) or no sharding; "
                    f"got placements {first.placements}."
                )
            if shard_placements:
                sharded_mesh_dim = shard_placements[0][0]
                world = first.device_mesh.size(sharded_mesh_dim)
                if num_heads % world != 0:
                    raise ValueError(
                        f"num_heads ({num_heads}) must be divisible by the sharding "
                        f"world_size ({world}) so each rank holds whole heads."
                    )

        def _as_3d(t: Tensor) -> Tensor:
            local = t.to_local() if isinstance(t, DTensor) else t
            local_dim0 = local.shape[0]
            if local_dim0 % head_dim != 0:
                raise RuntimeError(
                    f"Local shard dim 0 ({local_dim0}) is not a multiple of head_dim "
                    f"({head_dim}); shard boundaries must align with heads."
                )
            return local.view(local_dim0 // head_dim, head_dim, *local.shape[1:])

        return tuple([_as_3d(t) for t in lst] for lst in (params,) + extras)

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
    if process_group is not None and (N > 1 or comm_dim is not None):
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


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
) -> Tensor:
    """
    Flatten the input tensor if needed and call the Newton-Schulz function.
    """
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon).reshape(original_shape)


def adjust_lr_rms_norm(lr, param_shape, flatten):
    """Adjust learning rate for constant element-wise RMS norm."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(fan_out, fan_in))
    return lr * adjusted_ratio


def adjust_lr_spectral_norm(lr, param_shape, flatten):
    """Adjust from spectral norm 1 to RMS operator norm 1."""
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]
    return lr * math.sqrt(fan_out / fan_in)
