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

from .newton_schulz_triton import newton_schulz_triton, zeropower_via_newtonschulz5
from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)
from .scalar_opts import adamw_update_foreach_async, lion_update_foreach_async

# Reuse Muon's helper functions
from .muon import (
    muon_update_newton_schulz,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)


class Dion2(Optimizer):
    """
    Distributed Dion2 optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        fraction: Fraction of submatrix to orthogonalize per update (0 < fraction <= 1).
        ef_decay: Error-feedback decay factor applied to selected submatrix.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        epsilon: Small value to avoid division by zero.
        adjust_lr: How to adjust the learning rate for Muon updates ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate.
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float) -> Tensor`.
        verbose: Whether to print debug information during updates. If True, it prints whether rows or columns are selected for the submatrix selection process.

    Dion2 optimizer by Ahn et al.: TBD
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        fraction: float = 0.25,
        ef_decay: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        verbose: bool = False,
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if ef_decay < 0.0:
            raise ValueError(f"Invalid ef_decay: {ef_decay}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            ef_decay=ef_decay,
            fraction=fraction,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="dion2",
            step=0,
        )
        super().__init__(params, defaults)

        # Distributed configuration
        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError(
                    f"Only 1D DeviceMesh supported, but got {distributed_mesh.ndim}D. For HSDP, provide the 1D sharded sub-mesh."
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
                f"Invalid distributed_mesh type: {type(distributed_mesh)}. Expected DeviceMesh or ProcessGroup."
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
            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5
        self.verbose = verbose

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dion2_groups = []
        lion_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "dion2":
                dion2_groups.append(group)
            elif algo == "lion":
                lion_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        dion2_tasks = self._create_dion2_tasks(dion2_groups, verbose=self.verbose)
        lion_tasks = self._create_lion_tasks(lion_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(dion2_tasks, lion_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """
        Get optimizer state for the given parameter tensor,
        or lazy-initialize it if it doesn't exist.
        """
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _get_shard_info(self, param: Tensor, group: dict):
        """Determine sharding info for a parameter. Returns (is_batch_sharded, is_matrix_sharded, sharded_tensor_dim)."""
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
                    f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                    f"DTensor has mesh: {param.device_mesh}, placements: {param.placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                )
        elif len(shard_placements) > 1:
            raise NotImplementedError(
                "Dion2 does not support parameters with multiple sharded dimensions."
            )

        return is_batch_sharded, is_matrix_sharded, sharded_tensor_dim

    def _create_dion2_tasks(
        self,
        param_groups: List[dict],
        verbose: bool = False,
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched Dion2 task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == "dion2"
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Dion2 only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Most hyperparameters as tensors for torch.compile
            # Here "fraction" only determines the dimension of the submatrix
            # to be orthonormalized. Hence, it doesn't need to be a tensor
            dion2_args = dict(
                lr=torch.tensor(group["lr"]),
                ef_decay=torch.tensor(group["ef_decay"]),
                fraction=group["fraction"],
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
                verbose=verbose,
            )

            # Group parameters by shape, sharding, and dtype for mega-batching
            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "dion2") for p in params]
                momentums = [s["momentum"] for s in states]

                is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                    self._get_shard_info(params[0], group)
                )

                megabatch_args = dion2_args
                if is_batch_sharded and not is_matrix_sharded:
                    # Batch-sharded 3D tensors already contain whole local matrices,
                    # so we can stack them and run the local megabatch path without
                    # cross-rank communication.
                    megabatch_args = {**dion2_args, "process_group": None}

                yield AsyncTask(
                    dion2_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=sharded_tensor_dim,
                        **megabatch_args,
                    )
                )

    def _create_lion_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "lion",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for Lion updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            yield AsyncTask(
                lion_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                )
            )

    def _create_adamw_tasks(
        self,
        param_groups: List[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to generate AsyncTask objects for AdamW updates.
        """
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            step = torch.tensor(group["step"])

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def dion2_update_batch_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    ef_decay: Tensor,  # Error-feedback factor (scalar tensor)
    fraction: float,  # Fraction of submatrix to orthogonalize (0 < fraction <= 1)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    verbose: bool = False,
) -> Generator[None, None, None]:
    """
    Batched version of Dion2 update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """
    assert len(X) == len(G)
    assert len(X) == len(M)

    # Determine selection dimension based on sharding and tensor shape:
    # For sharded matrices, we align select_dim with shard_dim
    # For unsharded matrices (DDP or single-GPU), we select the shorter dimension
    ndim = X[0].ndim
    select_dim = None

    if shard_dim is not None:
        # Normalize shard_dim to negative indexing for unified treatment
        shard_dim = shard_dim if shard_dim < 0 else shard_dim - ndim
        if shard_dim == -2:
            select_dim = -2  # Row-sharded
        elif shard_dim == -1:
            select_dim = -1  # Column-sharded

    # Fall-back to shorter dimension when DDP, Single-GPU, or batch-sharded
    if select_dim is None:
        num_rows, num_cols = X[0].shape[-2:]
        select_dim = -2 if num_rows <= num_cols else -1

    # Print how the selection choice based on shard_dim and tensor shape
    if verbose:
        _print_selection_choice(X[0].shape, shard_dim, select_dim, ndim)

    # Update momentum and select top-α fraction along select_dim
    U_selected, indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=ef_decay,
        select_dim=select_dim,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        # Use all-to-all to transform from a batch of shards to a single whole matrix
        # https://www.essential.ai/blog/infra
        assert len(X) == world_size, "Batch size must equal world size"
        assert (
            process_group is not None
        ), "process_group must be provided for sharded DTensors"
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert (
            X[0].size(shard_dim) % world_size == 0
        ), f"Shard dimension {shard_dim} size {X[0].size(shard_dim)} is not divisible by world size {world_size}."

        # Allocate buffers to receive shards of one whole submatrix from other devices
        recv_shards = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(
            recv_shards, U_selected, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Concatentate shards to form a whole matrix to orthogonalize
        # Only submatrix is orthogonalized!
        full_submatrix = torch.cat(recv_shards, dim=select_dim)
        full_submatrix = muon_update_newton_schulz(
            full_submatrix, newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

        # Split result back into shards
        # Contiguous is needed for all-to-all to work correctly
        send_shards = [
            t.contiguous()
            for t in torch.tensor_split(full_submatrix, world_size, dim=select_dim)
        ]

        # Redistribute the orthogonalized tensor back to original layout
        U_ortho = [torch.empty_like(u) for u in U_selected]
        work = dist.all_to_all(U_ortho, send_shards, group=process_group, async_op=True)
        yield
        work.wait()

    # Matrices are not sharded, so we can distribute the batch across different devices
    # Get a single matrix of the batch corresponding to this device
    elif len(U_selected) > 1:
        assert len(U_selected) == world_size, "Batch size must equal world size"
        assert process_group is not None

        single_matrix = U_selected[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_ortho = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Allocate empty tensors to receive updates from other devices
        U_ortho = [torch.empty_like(u) for u in U_selected]
        # All gather orthogonalized results from other devices into buffer
        work = dist.all_gather(
            U_ortho, single_ortho.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

    # Single tensor with no sharded dimension. This happens in 2 cases:
    # - Running on a single GPU
    # - 3D+ tensors sharded along a batch dimension (different whole matrices per device)
    else:
        assert len(U_selected) == 1
        U_ortho = [
            muon_update_newton_schulz(
                U_selected[0], newton_schulz_func, flatten=flatten, epsilon=epsilon
            )
        ]

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr: {adjust_lr}")

    # Update model parameters with orthogonalized output
    # Weight update is applied to selected slices only
    dion2_post_orthogonalize(
        X=to_local(X),
        U=U_ortho,
        indices=indices_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        select_dim=select_dim,
    )


def dion2_update_megabatch_async(
    X: List[Tensor],  # All same-shape params (may be more than world_size)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    ef_decay: Tensor,  # Error-feedback factor (scalar tensor)
    fraction: float,  # Fraction of submatrix to orthogonalize (0 < fraction <= 1)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    shard_dim: Optional[int] = None,  # Shard dimension for DTensor (if applicable)
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    verbose: bool = False,
) -> Generator[None, None, None]:
    """
    Mega-batched Dion2 update: processes ALL same-shape parameters in one
    communication round instead of world_size-sized batches. This reduces
    the number of all-to-all/all-gather rounds from O(N/world_size) to O(1)
    per shape group, and enables batched Newton-Schulz on stacked 3D tensors.
    """
    N = len(X)
    assert N == len(G) == len(M)

    # Determine selection dimension based on sharding and tensor shape
    ndim = X[0].ndim
    select_dim = None

    if shard_dim is not None:
        shard_dim = shard_dim if shard_dim < 0 else shard_dim - ndim
        if shard_dim == -2:
            select_dim = -2  # Row-sharded
        elif shard_dim == -1:
            select_dim = -1  # Column-sharded

    if select_dim is None:
        num_rows, num_cols = X[0].shape[-2:]
        select_dim = -2 if num_rows <= num_cols else -1

    if verbose:
        _print_selection_choice(X[0].shape, shard_dim, select_dim, ndim)

    # Update momentum and select top-α fraction along select_dim
    U_selected, indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=ef_decay,
        select_dim=select_dim,
    )

    if shard_dim is not None and process_group is not None:
        # --- Mega-batched sharded FSDP2 path ---
        # Pad N to be divisible by world_size
        pad_n = (world_size - N % world_size) % world_size
        U_work = U_selected + [torch.zeros_like(U_selected[0])] * pad_n if pad_n > 0 else U_selected
        N_total = len(U_work)
        per_rank = N_total // world_size

        # Stack shards for each target rank into a single tensor
        input_chunks = [
            torch.stack(U_work[r * per_rank : (r + 1) * per_rank])
            for r in range(world_size)
        ]  # each: [per_rank, *shard_shape]

        output_chunks = [torch.empty_like(c) for c in input_chunks]
        work = dist.all_to_all(
            output_chunks, input_chunks, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Cat shards from all ranks along select_dim to form full submatrices
        # select_dim is negative, so it correctly indexes the stacked tensor
        full_submatrices = torch.cat(output_chunks, dim=select_dim)

        # Batched Newton-Schulz on stacked tensor
        full_submatrices = muon_update_newton_schulz(
            full_submatrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # Split back and all-to-all back
        split_chunks = [
            s.contiguous()
            for s in torch.tensor_split(full_submatrices, world_size, dim=select_dim)
        ]

        recv_chunks = [torch.empty_like(c) for c in split_chunks]
        work = dist.all_to_all(
            recv_chunks, split_chunks, group=process_group, async_op=True
        )
        yield
        work.wait()

        # Unstack back to list in original order
        U_ortho = [recv_chunks[r][i] for r in range(world_size) for i in range(per_rank)]
        U_ortho = U_ortho[:N]  # Remove padding

    elif N > 1 and process_group is not None:
        # --- Mega-batched non-sharded path ---
        # Each GPU orthogonalizes N/world_size submatrices instead of 1
        pad_n = (world_size - N % world_size) % world_size
        U_work = U_selected + [torch.zeros_like(U_selected[0])] * pad_n if pad_n > 0 else U_selected
        N_total = len(U_work)
        per_rank = N_total // world_size

        # This GPU processes its assigned chunk
        start = device_rank * per_rank
        my_matrices = torch.stack(U_work[start : start + per_rank])

        my_matrices = muon_update_newton_schulz(
            my_matrices,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )

        # All-gather: each rank broadcasts its processed chunk
        all_chunks = [torch.empty_like(my_matrices) for _ in range(world_size)]
        work = dist.all_gather(
            all_chunks, my_matrices.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()

        # Unstack back to list in original order
        U_ortho = [all_chunks[r][i] for r in range(world_size) for i in range(per_rank)]
        U_ortho = U_ortho[:N]

    elif N == 1:
        U_ortho = [
            muon_update_newton_schulz(
                U_selected[0],
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
            )
        ]

    else:
        # N > 1 but no process_group (single GPU or batch-sharded 3D):
        # batch Newton-Schulz on stacked tensor
        stacked = torch.stack(U_selected)
        stacked = muon_update_newton_schulz(
            stacked,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
        )
        U_ortho = [stacked[i] for i in range(N)]

    # Compute scaled learning rate
    # Do this before to_local(X) because we use the full tensor shape, not the shard shape
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr: {adjust_lr}")

    # Update model parameters with orthogonalized output
    # Weight update is applied to selected slices only
    dion2_post_orthogonalize(
        X=to_local(X),
        U=U_ortho,
        indices=indices_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        select_dim=select_dim,
    )


@torch.compile(fullgraph=True)
def dion2_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    fraction: Tensor,
    ef_decay: Tensor,
    select_dim: int,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Update momentum with gradient and compute the input to orthogonalization.
    More specifically, it does the following steps:
        - updates the momentum with gradient
        - computes the top-k indices (according to L1 norm) to determine submatrices
        - (other norms can be used such as L2 norm)
        - does in-place error-feedback decay on the selected submatrices
        - output submatrices and indices
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype

    # norm_dim is the dimension we compute norm over
    # select_dim is the dimension we select submatrix from
    num_select = M[0].size(select_dim)
    norm_dim = -1 if select_dim == -2 else -2
    k = max(1, int(math.ceil(fraction * num_select)))

    # Update momentum: M = M + G
    G = [g.to(dtype=dtype) for g in G]
    torch._foreach_add_(M, G)

    M_stacked = torch.stack(M, dim=0)

    # Compute L1 norm along norm_dim (sum of absolute values)
    slice_norms = M_stacked.norm(p=1, dim=norm_dim)

    # Batched topk: indices shape (batch_size, k)
    _, indices = torch.topk(slice_norms, k, dim=-1, sorted=False)

    # Batched gather for slice extraction
    if select_dim == -2:
        # Selecting rows
        num_cols = M[0].size(-1)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_cols)
        selected_stacked = torch.gather(M_stacked, dim=-2, index=indices_expanded)
    else:
        # Selecting cols
        num_rows = M[0].size(-2)
        indices_expanded = indices.unsqueeze(-2).expand(-1, num_rows, -1)
        selected_stacked = torch.gather(M_stacked, dim=-1, index=indices_expanded)

    # Apply error feedback decay to selected slices in original M tensors
    indices_list = list(indices.unbind(dim=0))
    for m, idx in zip(M, indices_list):
        selected_slice = m.index_select(dim=select_dim, index=idx)
        m.index_copy_(dim=select_dim, index=idx, source=selected_slice * ef_decay)

    # Convert to bf16 and unstack for communication
    U_selected = list(selected_stacked.to(dtype=torch.bfloat16).unbind(dim=0))

    return U_selected, indices_list


@torch.compile(fullgraph=True)
def dion2_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    indices: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    select_dim: int,
):
    """
    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Convert U to match parameter dtype
    dtype = X[0].dtype
    U = [u.to(dtype=dtype) for u in U]
    # Apply weight update
    neg_lr = -adjusted_lr
    U_scaled = [neg_lr * u for u in U]
    for x, u_scaled, idx in zip(X, U_scaled, indices):
        x.index_add_(dim=select_dim, index=idx, source=u_scaled)


# A helper function to print selection chocie for each matrix
# It only prints once `verbose` is set True
_printed_configs: set = set()


def _print_selection_choice(
    shape: torch.Size,
    shard_dim: Optional[int],
    select_dim: int,
    ndim: int,
):
    config_key = (tuple(shape), shard_dim, select_dim)
    if config_key not in _printed_configs:
        _printed_configs.add(config_key)

        num_rows, num_cols = shape[-2:]
        select_info = "rows" if select_dim == -2 else "columns"
        norm_info = "row norms" if select_dim == -2 else "col norms"

        if shard_dim is None:
            mode = "DDP/Single-GPU"
            shorter = "rows" if num_rows <= num_cols else "cols"
            reason = f"shorter dim = {shorter} ({min(num_rows, num_cols)})"
        else:
            # Normalize shard_dim for display
            normalized = shard_dim if shard_dim < 0 else shard_dim - ndim
            if normalized == -2:
                mode = "FSDP"
                reason = f"row-sharded (shard_dim={shard_dim}→-2)"
            elif normalized == -1:
                mode = "FSDP"
                reason = f"col-sharded (shard_dim={shard_dim}→-1)"
            else:
                mode = "FSDP batch-sharded"
                shorter = "rows" if num_rows <= num_cols else "cols"
                reason = f"shard_dim={shard_dim} (batch), shorter = {shorter}"

        print(
            f"[Dion2] Shape {tuple(shape)}: {mode}, {reason} → "
            f"select top-α {select_info} by {norm_info}"
        )
