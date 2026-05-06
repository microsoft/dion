import math
import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import (
    DistributedOrthoBase,
    megabatch_orthogonalize_async,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)
from .opt_utils import AsyncTask, to_local


class Dion2(DistributedOrthoBase):
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
        use_gram_newton_schulz: Whether to use Gram Newton-Schulz for orthogonalization.
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is ``func(input: Tensor, epsilon: float) -> Tensor``.
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
        use_polar_express: bool = True,
        use_gram_newton_schulz: bool = False,
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
            fraction=float(fraction),
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="dion2",
            step=0,
        )
        super().__init__(
            params, distributed_mesh, "dion2", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )
        self.verbose = verbose

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched Dion2 task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Dion2 only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            update_args = dict(
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
                verbose=self.verbose,
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            num_heads = self._resolve_num_heads(group)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, self._algo_name) for p in params]
                momentums = [s["momentum"] for s in states]

                if num_heads is not None:
                    params, gradients, momentums = self._prepare_head_split(
                        num_heads, params, gradients, momentums
                    )
                    megabatch_args = {**update_args, "process_group": None}
                    shard_dim = None
                else:
                    is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                        self._get_shard_info(params[0], group)
                    )
                    megabatch_args = update_args
                    if is_batch_sharded and not is_matrix_sharded:
                        megabatch_args = {**update_args, "process_group": None}
                    shard_dim = sharded_tensor_dim

                yield AsyncTask(
                    dion2_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
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
    communication round instead of world_size-sized batches.
    """
    N = len(X)
    assert N == len(G) == len(M)

    # Determine selection dimension based on sharding and tensor shape:
    # For sharded matrices, we align select_dim with shard_dim
    # For unsharded matrices (DDP or single-GPU), we select the shorter dimension
    ndim = X[0].ndim
    select_dim = None
    is_sharded = shard_dim is not None

    if is_sharded:
        shard_dim_neg = shard_dim if shard_dim < 0 else shard_dim - ndim
        if shard_dim_neg == -2:
            select_dim = -2  # Row-sharded
        elif shard_dim_neg == -1:
            select_dim = -1  # Column-sharded

    if select_dim is None:
        num_rows, num_cols = X[0].shape[-2:]
        select_dim = -2 if num_rows <= num_cols else -1

    if verbose:
        _print_selection_choice(X[0].shape, shard_dim, select_dim, ndim)

    # Pre-orthogonalize: momentum update + submatrix selection
    U_selected, indices_list = dion2_pre_orthogonalize(
        G=to_local(G),
        M=to_local(M),
        fraction=fraction,
        ef_decay=ef_decay,
        select_dim=select_dim,
    )

    # comm_dim for sharded communication: use select_dim (which equals normalized shard_dim)
    comm_dim = select_dim if is_sharded else None

    # On the sharded path X[0] must still be a DTensor, so .shape[comm_dim]
    # is the unsharded global size. The megabatch fn uses this to compute
    # the rank-consistent pad size for its alltoall. Catch the case where a
    # future refactor moves to_local(X) above this point and silently
    # collapses .shape to the local size.
    if comm_dim is not None:
        if not isinstance(X[0], DTensor):
            raise TypeError(
                "Sharded path requires X[0] to be a DTensor so .shape gives "
                f"the global size; got {type(X[0]).__name__}."
            )
        global_comm_dim_size = X[0].shape[comm_dim]
    else:
        global_comm_dim_size = None

    # Orthogonalize via shared megabatch communication
    U_ortho = yield from megabatch_orthogonalize_async(
        U_selected,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=flatten,
        epsilon=epsilon,
        global_comm_dim_size=global_comm_dim_size,
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

    # Post-orthogonalize: apply update to selected indices only
    dion2_post_orthogonalize(
        X=to_local(X),
        U=U_ortho,
        indices=indices_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        select_dim=select_dim,
    )


# Workaround for a torch.compile bug in PyTorch ≤2.11's inductor backend:
# the post-fusion loop reordering pass crashes when ForeachKernelSchedulerNode
# appears inside a FusedSchedulerNode.  Only triggered by recompilation across
# different tensor dimensionalities (e.g. 2D then 3D).
# https://github.com/pytorch/pytorch/issues/176591
# TODO: remove this decorator when pytorch/pytorch#176591 is fixed.
_inductor_workaround = (
    torch._inductor.config.patch(loop_ordering_after_fusion=False)
    if torch.__version__ < "2.13"
    else lambda fn: fn
)


@_inductor_workaround
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

    # Extract the selected rows/columns from each momentum tensor.
    # `indices` has shape (..., k) where k is the number of selected slices.
    # `gather` requires the index tensor to have the same number of dimensions
    # as the source, so we expand the indices to cover the non-selected dimension.
    if select_dim == -2:
        # Selecting rows: expand indices from (..., k) to (..., k, num_cols)
        num_cols = M[0].size(-1)
        indices_expanded = indices.unsqueeze(-1).expand(*indices.shape, num_cols)
        selected_stacked = torch.gather(M_stacked, dim=-2, index=indices_expanded)
    else:
        # Selecting cols: expand indices from (..., k) to (..., num_rows, k)
        num_rows = M[0].size(-2)
        indices_expanded = indices.unsqueeze(-2).expand(
            *indices.shape[:-1], num_rows, indices.shape[-1]
        )
        selected_stacked = torch.gather(M_stacked, dim=-1, index=indices_expanded)

    # Apply error feedback decay to selected slices in the original M tensors.
    # We reuse the already-gathered slices and write them back (scaled) using
    # scatter_, which places values into positions specified by the index tensor.
    indices_list = list(indices.unbind(dim=0))
    selected_list = list(selected_stacked.unbind(dim=0))
    for m, idx, selected in zip(M, indices_list, selected_list):
        if select_dim == -2:
            idx_exp = idx.unsqueeze(-1).expand(*idx.shape, m.size(-1))
        else:
            idx_exp = idx.unsqueeze(-2).expand(*idx.shape[:-1], m.size(-2), idx.shape[-1])
        m.scatter_(dim=select_dim, index=idx_exp, src=selected * ef_decay)

    # Convert to bf16 and unstack for communication
    U_selected = list(selected_stacked.to(dtype=torch.bfloat16).unbind(dim=0))

    return U_selected, indices_list


# NOTE: if this function starts failing with an InductorError on recompilation
# across tensor ranks, apply the same _inductor_workaround used on
# dion2_pre_orthogonalize above.  See pytorch/pytorch#176591.
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
    # Apply the orthogonalized update to only the selected rows/columns.
    # scatter_add_ accumulates values into positions specified by the index tensor:
    #   x[..., idx_exp[..., i, j], j] += u_scaled[..., i, j]  (for select_dim == -2)
    # where i ranges over the k selected rows and j over all columns.
    for x, u_scaled, idx in zip(X, U_scaled, indices):
        if select_dim == -2:
            idx_exp = idx.unsqueeze(-1).expand_as(u_scaled)
        else:
            idx_exp = idx.unsqueeze(-2).expand_as(u_scaled)
        x.scatter_add_(dim=select_dim, index=idx_exp, src=u_scaled)


# A helper function to print selection choice for each matrix
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
