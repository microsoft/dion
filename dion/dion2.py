# dion2.py
import math
import torch
import torch.distributed as dist
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
        shard_independent: If True, use shard-independent row selection for Dion2.
            This incurs higher communication cost due to all-to-all of full matrices,

    Dion2 optimizer by Ahn et al.: https://arxiv.org/abs/2512.16928
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
        shard_independent: bool = False,
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
        self.shard_independent = shard_independent

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
        dion2_tasks = self._create_dion2_tasks(dion2_groups)
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

    def _create_dion2_tasks(
        self,
        param_groups: List[dict],
    ) -> Generator["AsyncTask", None, None]:
        """
        Helper function to create batches of Dion2 matrices and generate
        AsyncTask objects so we can process multiple batches concurrently.
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
                shard_independent=self.shard_independent,
            )

            # Create batches of parameters of size self._world_size
            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "dion2") for p in params]
                momentums = [s["momentum"] for s in states]

                # Get sharding state for DTensor
                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    if not isinstance(self._distributed_mesh, DeviceMesh):
                        raise RuntimeError(
                            "Must create optimizer with DeviceMesh if using DTensor parameters."
                        )

                    # Find the sharded placement and get its mesh and tensor dimensions
                    # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    # If we don't flatten 3D matrices, we can ignore shard placements along batch dimensions
                    # Only keep placements that shard one of the two matrix dimensions
                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    # We currently do not support tensors sharded along the last dimension because Dion2
                    # normalization later assumes a full trailing axis when computing means.
                    if any(p.dim == params[0].ndim - 1 for _, p in shard_placements):
                        raise NotImplementedError(
                            "Dion2 currently does not support parameters sharded along the last dimension. "
                            "Please avoid shards at dim -1."
                            "(Note: Default behavior of FSDP2 is to shard along dim-0.)"
                        )

                    # Check that we have no more than 1 sharded matrix dimension
                    # Note that non-flattened 3D tensors can have additional sharded batch dimensions
                    # Flattened 3D tensors are limited to one sharded dimension out of all dimensions
                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim
                    elif len(shard_placements) > 1:
                        raise NotImplementedError(
                            "Dion2 does not support parameters with multiple sharded dimensions."
                        )

                    # Check that the sharded mesh dimension matches optimizer's device mesh
                    if (
                        sharded_mesh_dim is not None
                        and params[0].device_mesh.get_group(sharded_mesh_dim)
                        != self._process_group
                    ):
                        raise RuntimeError(
                            f"Got DTensor sharded over mesh dimension {sharded_mesh_dim} different from the optimizer's device mesh. "
                            f"DTensor has mesh: {params[0].device_mesh}, placements: {params[0].placements}, but optimizer was created with mesh: {self._distributed_mesh}."
                        )

                # Special case for 3D tensors sharded along batch dimension
                # As long as matrix dimensions are not sharded, each device will have whole matrices
                # Each device already has different matrices of the batch, so we can't parallelize further
                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m in zip(params, gradients, momentums):
                        yield AsyncTask(
                            dion2_update_batch_async(
                                X=[x],
                                G=[g],
                                M=[m],
                                shard_dim=None,  # No sharded matrix dim
                                **dion2_args,
                            )
                        )
                # Otherwise, we parallelize the Muon update across devices
                else:
                    yield AsyncTask(
                        dion2_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **dion2_args,
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
    shard_independent: bool = False,
) -> Generator[None, None, None]:
    """
    Batched version of Dion2 update. Batch size should be equal to number of GPUs.
    All tensors in a batch should have identical shape, sharding, and dtype.
    Identical hyperparameters are used for all tensors in the batch.
    """
    assert len(X) == len(G)
    assert len(X) == len(M)

    # ALways select submtrix based on row-wise selection
    # This way it chooses the submatrix neuron-wise
    select_dim = -2

    # Shard-independent path:
    # The row-selection process is independent of the sharding configuration.
    # However, this comes with the cost of all-to-all communication of full matrices.
    if shard_dim is not None and shard_independent:
        assert len(X) == world_size
        assert process_group is not None

        M_local = to_local(M)
        G_local = to_local(G)
        dtype = M_local[0].dtype

        # Update momentum locally: M = M + G
        G_casted = [g.to(dtype=dtype) for g in G_local]
        torch._foreach_add_(M_local, G_casted)

        # All-to-all to get one full matrix per device
        M_recv = [torch.empty_like(M_local[0]) for _ in range(world_size)]
        work = dist.all_to_all(M_recv, M_local, group=process_group, async_op=True)
        yield
        work.wait()

        # Concatenate to form full matrix
        M_full = torch.cat(M_recv, dim=shard_dim)
        full_rows, full_cols = M_full.shape[-2:]

        # Select top-k on full matrix
        num_select = M_full.size(select_dim)
        k = max(1, int(math.ceil(fraction * num_select)))

        slice_norms = M_full.norm(p=1, dim=-1)
        _, indices = torch.topk(slice_norms, k, dim=-1, sorted=False)

        # Extract and orthonormalize
        U_selected = M_full.index_select(dim=select_dim, index=indices).to(
            dtype=torch.bfloat16
        )
        U_ortho = muon_update_newton_schulz(
            U_selected, newton_schulz_func, flatten=flatten, epsilon=epsilon
        )

        # Construct dense update matrix (zeros on the unchosen rows/cols)
        U_dense_full = torch.zeros(
            M_full.shape, dtype=U_ortho.dtype, device=U_ortho.device
        )
        idx_exp = indices.unsqueeze(-1).expand(-1, full_cols)
        U_dense_full.scatter_(-2, idx_exp, U_ortho)

        # All-to-all scatter dense U back to shards
        U_send = [
            t.contiguous()
            for t in torch.tensor_split(U_dense_full, world_size, dim=shard_dim)
        ]
        U_recv = [torch.empty_like(U_send[0]) for _ in range(world_size)]


        work = dist.all_to_all(U_recv, U_send, group=process_group, async_op=True)
        yield
        work.wait()

        # Infer selected indices from non-zero rows/cols and apply error-feedback decay locally
        dion2_apply_ef_decay_from_dense(
            M=M_local,
            U_dense=U_recv,
            ef_decay=ef_decay,
            select_dim=select_dim,
        )

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

        # Apply weight update using dense U
        dion2_post_orthogonalize_dense(
            X=to_local(X),
            U_dense=U_recv,
            base_lr=lr,
            adjusted_lr=adjusted_lr,
            weight_decay=weight_decay,
        )
    
    # This path corresponds to the case when `shard-independent` is False
    # When shard_dim is not None, the row selection process depends on the sharding configuration.
    # This is because each shard chooses its own top-k rows based on local momentum.
    # This path only communicates the selected submatrices for orthogonalization, which leads to less communication.
    else:
        # Update momentum and select top-fraction along select_dim
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
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Convert U to match parameter dtype
    dtype = X[0].dtype
    U = [u.to(dtype=dtype) for u in U]
    # Apply weight update
    neg_lr = -adjusted_lr
    U_scaled = [neg_lr * u for u in U]
    for x, u_scaled, idx in zip(X, U_scaled, indices):
        x.index_add_(dim=select_dim, index=idx, source=u_scaled)


def dion2_apply_ef_decay_from_dense(
    M: List[Tensor],
    U_dense: List[Tensor],
    ef_decay: Tensor,
    select_dim: int,
):
    """
    Infer selected indices from non-zero rows/cols in U_dense,
    then apply error-feedback decay to those rows/cols in M.
    This function is used for `shard-independent' path of the Dion2 update.
    """
    norm_dim = -1 if select_dim == -2 else -2

    for m, u in zip(M, U_dense):
        # Check if ANY element in the row/col is non-zero
        any_nonzero = (u != 0).any(dim=norm_dim)
        indices = torch.where(any_nonzero)[0]

        # Apply error-feedback decay on the non-zero rows/cols
        if indices.numel() > 0:
            selected_slice = m.index_select(dim=select_dim, index=indices)
            m.index_copy_(
                dim=select_dim, index=indices, source=selected_slice * ef_decay
            )


def dion2_post_orthogonalize_dense(
    X: List[Tensor],
    U_dense: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Apply weight decay and weight update.
    This function assumes that the update is dense (zeros on unselected rows/cols).
    This is used for `shard-independent' path of the Dion2 update.
    """
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Dense weight update
    dtype = X[0].dtype
    U_casted = [u.to(dtype=dtype) for u in U_dense]
    torch._foreach_add_(X, U_casted, alpha=-adjusted_lr)
