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
from .muon import muon_update_post_orthogonalize
from .opt_utils import AsyncTask, to_local


class Muon2(DistributedOrthoBase):
    """
    Distributed MUON2 optimizer for PyTorch FSDP2. Also compatible with DDP.

    MUON2 applies an adaptive second-moment preconditioner to the
    momentum before Newton-Schulz orthogonalization. It uses an Adam-style
    second-moment term before a Muon-style distributed orthogonalization and
    post-orthogonalization update.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For Muon2, this will be scaled based on the
            matrix dimensions. For element-wise update rules, this is the actual
            learning rate and no additional scaling is done.
        mu: Momentum factor.
        muon_beta2: Second-moment decay for the MUON2 preconditioner.
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
            algorithm="muon2",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(
            params,
            distributed_mesh,
            "muon2",
            defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_muon2_state(self, param: Tensor) -> dict:
        state = self.state[param]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(param)
        if "variance" not in state:
            state["variance"] = torch.zeros_like(param)
        return state

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched MUON2 task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon2 optimizer only supports matrix parameters."

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
                states = [self._get_or_initialize_muon2_state(p) for p in params]
                momentums = [s["momentum"] for s in states]
                variances = [s["variance"] for s in states]

                if num_heads is not None:
                    params, gradients, momentums, variances = self._prepare_head_split(
                        num_heads, params, gradients, momentums, variances
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
                    muon2_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


def muon2_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
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
    Mega-batched MUON2 update: updates momentum and second moments (variances),
    preconditions the momentum, then uses the shared Muon orthogonalization path.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V)

    G_local = to_local(G)
    M_local = to_local(M)
    V_local = to_local(V)

    # Convert shard_dim to negative for comm_dim
    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None

    G_stacked = torch.stack(G_local)
    M_stacked = torch.stack(M_local)
    V_stacked = torch.stack(V_local)

    M_stacked, V_stacked = muon2_update_moments_stacked(
        G=G_stacked,
        M=M_stacked,
        V=V_stacked,
        momentum=momentum,
        beta2=muon_beta2,
    )

    _copy_stacked_to_list(M_stacked, M_local)
    _copy_stacked_to_list(V_stacked, V_local)

    U_stacked = muon2_precondition_momentum_stacked(
        G=G_stacked,
        M=M_stacked,
        V=V_stacked,
        momentum=momentum,
        epsilon=epsilon,
        nesterov=nesterov,
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
def muon2_update_moments_stacked(
    G: Tensor,
    M: Tensor,
    V: Tensor,
    momentum: Tensor,
    beta2: Tensor,
) -> Tuple[Tensor, Tensor]:
    dtype = M.dtype
    G = G.to(dtype=dtype)

    M = M * momentum + G
    grad_sq = G * G
    V = V * beta2 + grad_sq * (1 - beta2)
    return M, V


@torch.compile(fullgraph=True)
def muon2_precondition_momentum_stacked(
    G: Tensor,
    M: Tensor,
    V: Tensor,
    momentum: Tensor,
    epsilon: Tensor,
    nesterov: bool,
) -> Tensor:
    dtype = M.dtype
    if nesterov:
        U = M * momentum + G.to(dtype=dtype)
    else:
        U = M

    U = U / (V.sqrt() + epsilon)

    return U.to(dtype=torch.bfloat16)


def muon2_update_moments(
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    momentum: Tensor,
    beta2: Tensor,
) -> None:
    """
    Update momentum and gradient second moments in place.
    Inputs should be regular Tensor lists, not DTensor lists.
    """
    momentum_f = float(momentum)
    beta2_f = float(beta2)

    dtype = M[0].dtype
    G_cast = [g.to(dtype=dtype) for g in G]

    torch._foreach_mul_(M, momentum_f)
    torch._foreach_add_(M, G_cast)

    for g, v in zip(G_cast, V):
        grad_sq = g * g
        v.mul_(beta2_f).add_(grad_sq, alpha=1 - beta2_f)

    return None


def muon2_precondition_momentum(
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    momentum: Tensor,
    epsilon: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """
    Build the preconditioned momentum used as input to orthogonalization.
    Inputs should be regular Tensor lists, not DTensor lists.
    """
    momentum_f = float(momentum)
    eps_f = float(epsilon)
    dtype = M[0].dtype
    U = []

    for g, m, v in zip(G, M, V):
        if nesterov:
            u = m * momentum_f + g.to(dtype=dtype)
        else:
            u = m

        u = u / (v.sqrt() + eps_f)
        U.append(u.to(dtype=torch.bfloat16))

    return U
