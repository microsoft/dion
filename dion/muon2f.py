import math
import torch
import torch.distributed as dist
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
from .muon import muon_update_post_orthogonalize
from .opt_utils import AsyncTask, to_local


class Muon2F(DistributedOrthoBase):
    """
    Distributed MUON2-F optimizer for PyTorch FSDP2. Also compatible with DDP.

    MUON2-F applies a factorized adaptive second-moment preconditioner to the
    momentum before Newton-Schulz orthogonalization. It follows MUON2-F's
    Adafactor-style row/column second-moment approximation while reusing Muon's
    distributed orthogonalization and post-orthogonalization update.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon2F, this will be scaled based on the
            matrix dimensions. For element-wise update rules, this is the actual
            learning rate and no additional scaling is done.
        mu: Momentum factor.
        muon_beta2: Second-moment decay for the factorized MUON2-F preconditioner.
        betas: Tuple of (beta1, beta2) for AdamW and Lion scalar algorithms.
        weight_decay: Weight decay factor.
        cautious_wd: Whether to apply weight decay only where update and parameter
            signs align.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate ("spectral_norm" or
            "rms_norm" or None).
        flatten: Whether to flatten 3D+ tensors to 2D for Muon updates.
        use_gram_newton_schulz: Whether to use Gram Newton-Schulz.
        use_triton: Whether to use Triton kernels for Newton-Schulz. Ignored if
            custom function is provided.
        use_polar_express: Whether to use Polar Express for orthogonalization.
        newton_schulz_func: Custom orthogonalization function with signature
            ``func(input: Tensor, epsilon: float) -> Tensor``.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        use_gram_newton_schulz: bool = False,
        use_triton: bool = False,
        use_polar_express: bool = True,
        newton_schulz_func: Optional[Callable] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if muon_beta2 < 0.0:
            raise ValueError(f"Invalid muon_beta2: {muon_beta2}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. "
                "Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            mu=mu,
            muon_beta2=muon_beta2,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="muon2f",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(
            params,
            distributed_mesh,
            "muon2f",
            defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_muon2f_state(
        self, param: Tensor, flatten: bool, num_heads: Optional[int]
    ) -> dict:
        state = self.state[param]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(param)

        local_param = param.to_local() if isinstance(param, DTensor) else param
        row_shape, col_shape = _factorized_state_shapes(
            local_shape=local_param.shape,
            flatten=flatten,
            num_heads=num_heads,
            global_shape=param.shape,
        )
        state_needs_init = (
            "variance_row" not in state
            or "variance_col" not in state
            or state["variance_row"].shape != row_shape
            or state["variance_col"].shape != col_shape
        )
        if state_needs_init:
            state["variance_row"] = torch.zeros(
                row_shape, dtype=local_param.dtype, device=local_param.device
            )
            state["variance_col"] = torch.zeros(
                col_shape, dtype=local_param.dtype, device=local_param.device
            )
        return state

    def _get_shard_info(self, param: Tensor, group: dict):
        result = super()._get_shard_info(param, group)
        _, is_matrix_sharded, sharded_tensor_dim = result
        if is_matrix_sharded and sharded_tensor_dim == param.ndim - 1:
            raise NotImplementedError(
                "Muon2F currently does not support parameters sharded along "
                "the last dimension. Please avoid shards at dim -1 so "
                "row-wise second-moment statistics stay local."
            )
        return result

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched MUON2-F task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon2F optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
                cautious_wd=group["cautious_wd"],
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            num_heads = self._resolve_num_heads(group)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [
                    self._get_or_initialize_muon2f_state(p, group["flatten"], num_heads)
                    for p in params
                ]
                momentums = [s["momentum"] for s in states]
                variances_row = [s["variance_row"] for s in states]
                variances_col = [s["variance_col"] for s in states]

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
                    muon2f_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V_row=variances_row,
                        V_col=variances_col,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


def _factorized_state_shapes(
    local_shape: torch.Size,
    flatten: bool,
    num_heads: Optional[int],
    global_shape: Optional[torch.Size] = None,
) -> Tuple[torch.Size, torch.Size]:
    if num_heads is not None:
        shape_for_heads = global_shape if global_shape is not None else local_shape
        if len(shape_for_heads) != 2:
            raise ValueError(
                "num_heads is only supported for 2D parameters, got shape "
                f"{tuple(shape_for_heads)}."
            )
        if shape_for_heads[0] % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must divide dim 0 of the parameter "
                f"(got shape {tuple(shape_for_heads)})."
            )
        head_dim = shape_for_heads[0] // num_heads
        if local_shape[0] % head_dim != 0:
            raise RuntimeError(
                f"Local shard dim 0 ({local_shape[0]}) is not a multiple of "
                f"head_dim ({head_dim}); shard boundaries must align with heads."
            )
        local_heads = local_shape[0] // head_dim
        return (
            torch.Size((local_heads, head_dim, 1)),
            torch.Size((local_heads, 1, local_shape[1])),
        )

    if flatten and len(local_shape) >= 3:
        return torch.Size((local_shape[0], 1)), torch.Size(
            (1, math.prod(local_shape[1:]))
        )

    return torch.Size((*local_shape[:-1], 1)), torch.Size(
        (*local_shape[:-2], 1, local_shape[-1])
    )


def muon2f_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V_row: List[Tensor],
    V_col: List[Tensor],
    lr: Tensor,
    momentum: Tensor,
    muon_beta2: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    """
    Mega-batched MUON2-F update: updates momentum and factorized second moments,
    preconditions the momentum, then uses the shared Muon orthogonalization path.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V_row) == len(V_col)

    G_local = to_local(G)
    M_local = to_local(M)
    V_row_local = to_local(V_row)
    V_col_local = to_local(V_col)

    # Convert shard_dim to negative for comm_dim
    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None

    sync_col_stats = process_group is not None and comm_dim == -2
    G_stacked = torch.stack(G_local)
    M_stacked = torch.stack(M_local)
    V_row_stacked = torch.stack(V_row_local)
    V_col_stacked = torch.stack(V_col_local)

    M_stacked, V_row_stacked, V_col_stacked, col_updates = (
        muon2f_update_moments_stacked(
            G=G_stacked,
            M=M_stacked,
            V_row=V_row_stacked,
            V_col=V_col_stacked,
            momentum=momentum,
            beta2=muon_beta2,
            flatten=flatten,
            update_col=not sync_col_stats,
        )
    )

    _copy_stacked_to_list(M_stacked, M_local)
    _copy_stacked_to_list(V_row_stacked, V_row_local)

    if not sync_col_stats:
        _copy_stacked_to_list(V_col_stacked, V_col_local)

    # With last-dimension sharding rejected, row-wise statistics are local.
    # Row-sharded parameters still need global column statistics.
    if sync_col_stats:
        work = dist.all_reduce(col_updates, group=process_group, async_op=True)
        yield
        work.wait()
        V_col_stacked = muon2f_apply_col_moment_updates_stacked(
            V_col=V_col_stacked,
            col_updates=col_updates,
            beta2=muon_beta2,
        )
        _copy_stacked_to_list(V_col_stacked, V_col_local)

    U_stacked = muon2f_precondition_momentum_stacked(
        G=G_stacked,
        M=M_stacked,
        V_row=V_row_stacked,
        V_col=V_col_stacked,
        momentum=momentum,
        epsilon=epsilon,
        nesterov=nesterov,
        flatten=flatten,
    )
    U = list(U_stacked.unbind(dim=0))

    # On the sharded path X[0] must still be a DTensor, so .shape[comm_dim]
    # is the unsharded global size.
    if comm_dim is not None:
        if not isinstance(X[0], DTensor):
            raise TypeError(
                "Sharded path requires X[0] to be a DTensor so .shape gives "
                f"the global size; got {type(X[0]).__name__}."
            )
        global_comm_dim_size = X[0].shape[comm_dim]
    else:
        global_comm_dim_size = None

    U = yield from megabatch_orthogonalize_async(
        U,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=flatten,
        epsilon=epsilon,
        global_comm_dim_size=global_comm_dim_size,
    )

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        cautious_wd=cautious_wd,
    )


def _copy_stacked_to_list(stacked: Tensor, tensors: List[Tensor]) -> None:
    for dst, src in zip(tensors, stacked.unbind(dim=0)):
        dst.copy_(src)


@torch.compile(fullgraph=True)
def muon2f_update_moments_stacked(
    G: Tensor,
    M: Tensor,
    V_row: Tensor,
    V_col: Tensor,
    momentum: Tensor,
    beta2: Tensor,
    flatten: bool,
    update_col: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype = M.dtype
    G = G.to(dtype=dtype)

    M = M * momentum + G
    G_matrix = _flatten_stacked_for_muon2f(G, flatten)
    grad_sq = G_matrix * G_matrix
    row_updates = grad_sq.sum(dim=-1, keepdim=True)
    col_updates = grad_sq.sum(dim=-2, keepdim=True)
    V_row = V_row * beta2 + row_updates * (1 - beta2)
    if update_col:
        V_col = V_col * beta2 + col_updates * (1 - beta2)
    return M, V_row, V_col, col_updates


@torch.compile(fullgraph=True)
def muon2f_apply_col_moment_updates_stacked(
    V_col: Tensor,
    col_updates: Tensor,
    beta2: Tensor,
) -> Tensor:
    return V_col * beta2 + col_updates * (1 - beta2)


@torch.compile(fullgraph=True)
def muon2f_precondition_momentum_stacked(
    G: Tensor,
    M: Tensor,
    V_row: Tensor,
    V_col: Tensor,
    momentum: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
) -> Tensor:
    dtype = M.dtype
    if nesterov:
        U = M * momentum + G.to(dtype=dtype)
    else:
        U = M

    original_shape = U.shape
    U_matrix = _flatten_stacked_for_muon2f(U, flatten)
    moment_sum = V_col.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    variance = V_row * V_col / moment_sum
    U_matrix = U_matrix / (variance.sqrt() + epsilon)

    if flatten and len(original_shape) >= 4:
        U_matrix = U_matrix.reshape(original_shape)
    return U_matrix.to(dtype=torch.bfloat16)


def muon2f_update_moments(
    G: List[Tensor],
    M: List[Tensor],
    V_row: List[Tensor],
    V_col: List[Tensor],
    momentum: Tensor,
    beta2: Tensor,
    flatten: bool,
    update_col: bool = True,
) -> Optional[List[Tensor]]:
    """
    Update momentum and factorized gradient second moments in place.
    Inputs should be regular Tensor lists, not DTensor lists.
    """
    momentum_f = float(momentum)
    beta2_f = float(beta2)

    dtype = M[0].dtype
    G_cast = [g.to(dtype=dtype) for g in G]

    torch._foreach_mul_(M, momentum_f)
    torch._foreach_add_(M, G_cast)

    col_updates = []
    for g, v_row, v_col in zip(G_cast, V_row, V_col):
        g_matrix = _flatten_for_muon2f(g, flatten)
        grad_sq = g_matrix * g_matrix
        v_row.mul_(beta2_f).add_(grad_sq.sum(dim=-1, keepdim=True), alpha=1 - beta2_f)
        col_update = grad_sq.sum(dim=-2, keepdim=True)
        if update_col:
            v_col.mul_(beta2_f).add_(col_update, alpha=1 - beta2_f)
        else:
            col_updates.append(col_update)

    return None if update_col else col_updates


def muon2f_apply_col_moment_updates(
    V_col: List[Tensor],
    col_updates: List[Tensor],
    beta2: Tensor,
) -> None:
    beta2_f = float(beta2)
    for v_col, col_update in zip(V_col, col_updates):
        v_col.mul_(beta2_f).add_(col_update, alpha=1 - beta2_f)


def muon2f_precondition_momentum(
    G: List[Tensor],
    M: List[Tensor],
    V_row: List[Tensor],
    V_col: List[Tensor],
    momentum: Tensor,
    epsilon: Tensor,
    nesterov: bool,
    flatten: bool,
) -> List[Tensor]:
    """
    Build the preconditioned momentum used as input to orthogonalization.
    Inputs should be regular Tensor lists, not DTensor lists.
    """
    momentum_f = float(momentum)
    eps_f = float(epsilon)
    dtype = M[0].dtype
    U = []

    for g, m, v_row, v_col in zip(G, M, V_row, V_col):
        if nesterov:
            u = m * momentum_f + g.to(dtype=dtype)
        else:
            u = m

        original_shape = u.shape
        u_matrix = _flatten_for_muon2f(u, flatten)
        moment_sum = v_col.sum(dim=-1, keepdim=True).clamp_min(eps_f)
        variance = v_row * v_col / moment_sum
        u_matrix = u_matrix / (variance.sqrt() + eps_f)

        if flatten and len(original_shape) >= 3:
            u_matrix = u_matrix.reshape(original_shape)
        U.append(u_matrix.to(dtype=torch.bfloat16))

    return U


def _flatten_for_muon2f(X: Tensor, flatten: bool) -> Tensor:
    if flatten and X.ndim >= 3:
        return X.flatten(start_dim=1)
    return X


def _flatten_stacked_for_muon2f(X: Tensor, flatten: bool) -> Tensor:
    if flatten and X.ndim >= 4:
        return X.flatten(start_dim=2)
    return X
