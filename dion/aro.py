import math
import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import DistributedOrthoBase, megabatch_orthogonalize_async
from .opt_utils import AsyncTask, to_local
from .muon import adjust_lr_spectral_norm, adjust_lr_rms_norm


class ARO(DistributedOrthoBase):
    """
    Adaptively Rotated Optimization (ARO) optimizer.

    ARO performs normed steepest descent in a rotated coordinate system,
    where the rotation is determined by a norm-informed policy that couples
    the rotation to the base optimizer's transformation.

    Each parameter of shape [m, n] maintains an m×m rotation matrix R in
    float32, adding O(m²) memory per parameter (e.g., 64 MB for m=4096).

    FSDP is not supported — use DDP or single-GPU.

    Reference: https://arxiv.org/abs/2602.09006

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
        lr: Base learning rate.
        mu: Momentum factor for EMA gradient accumulation.
        betas: Tuple of (beta1, beta2) for AdamW and Lion scalar parameter groups.
        weight_decay: Weight decay factor.
        epsilon: Small value for numerical stability.
        base_opt: Base optimizer function applied in the rotated frame.
            "row_norm": f(X) = sqrt(n) * X / ||x_i||  (row normalization)
            "sign": f(X) = sign(X)
        adjust_lr: How to adjust the learning rate for ARO updates.
            "spectral_norm", "rms_norm", or None.
        flatten: Whether to flatten 3D+ tensors to 2D.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union["DeviceMesh", ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        base_opt: str = "row_norm",
        adjust_lr: Optional[str] = "rms_norm",
        flatten: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor: {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if base_opt not in ("row_norm", "sign"):
            raise ValueError(
                f"Invalid base_opt: {base_opt}. Must be 'row_norm' or 'sign'."
            )
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            epsilon=epsilon,
            base_opt=base_opt,
            flatten=flatten,
            adjust_lr=adjust_lr,
            algorithm="aro",
            step=0,
        )
        super().__init__(params, distributed_mesh, "aro", defaults)

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == "aro" and "rotation" not in state:
            m = param.shape[-2]
            state["rotation"] = torch.eye(m, device=param.device, dtype=torch.float32)
        return state

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == "aro"
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "ARO only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                epsilon=torch.tensor(group["epsilon"]),
                base_opt=group["base_opt"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "aro") for p in params]
                momentums = [s["momentum"] for s in states]
                rotations = [s["rotation"] for s in states]

                is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                    self._get_shard_info(params[0], group)
                )

                if is_matrix_sharded:
                    raise NotImplementedError(
                        "ARO does not support FSDP-sharded parameters. "
                        "Use DDP or single-GPU instead."
                    )

                megabatch_args = update_args
                if is_batch_sharded:
                    megabatch_args = {**update_args, "process_group": None}

                yield AsyncTask(
                    aro_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        R=rotations,
                        **megabatch_args,
                    )
                )


def aro_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    R: List[Tensor],  # float32 rotation matrices
    lr: Tensor,
    momentum: Tensor,
    weight_decay: Tensor,
    epsilon: Tensor,
    base_opt: str,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    process_group: Optional[ProcessGroup] = None,
) -> Generator[None, None, None]:
    """
    Megabatched ARO update. Pre-computes cross-alignment matrices locally,
    then distributes QR orthogonalization across ranks via the shared
    megabatch_orthogonalize_async infrastructure.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(R)

    M_local = to_local(M)
    G_local = to_local(G)

    # Update momentum: M = mu * M + (1-mu) * G
    G_cast = [g.to(dtype=m.dtype) for g, m in zip(G_local, M_local)]
    torch._foreach_lerp_(M_local, G_cast, 1 - momentum)

    base_opt_fn = _get_base_opt_fn(base_opt)

    # Pre-compute cross-alignment matrices (local, all params)
    cross_list = []
    for i in range(N):
        M_f32 = M_local[i].float()
        rotated = R[i].mT @ M_f32
        f_rotated = base_opt_fn(rotated)
        cross_list.append(M_f32 @ f_rotated.mT)

    # Distribute QR across ranks via shared megabatch infrastructure
    def qr_orthogonalize(X_in, epsilon=None):
        Q, _ = torch.linalg.qr(X_in)
        return Q

    Q_list = yield from megabatch_orthogonalize_async(
        cross_list,
        comm_dim=None,  # non-sharded
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=qr_orthogonalize,
        flatten=False,
        epsilon=epsilon,
    )

    # Post-compute: use new rotations to produce update directions (local, all params)
    U_list = []
    for i in range(N):
        Q = Q_list[i]
        R[i].copy_(Q)
        M_f32 = M_local[i].float()
        rotated_new = Q.mT @ M_f32
        f_new = base_opt_fn(rotated_new)
        U_list.append(Q @ f_new)

    # Compute adjusted learning rate
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr: {adjust_lr}")

    # Apply weight decay and parameter update
    aro_post_update(
        X=to_local(X),
        U=U_list,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def _get_base_opt_fn(base_opt: str):
    if base_opt == "row_norm":
        return _base_opt_row_norm
    elif base_opt == "sign":
        return _base_opt_sign
    raise ValueError(f"Unknown base_opt: {base_opt}")


def _base_opt_row_norm(X: Tensor) -> Tensor:
    """f(X) = sqrt(n) * X / ||x_i||_row"""
    n = X.shape[-1]
    row_norms = X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return math.sqrt(n) * X / row_norms


def _base_opt_sign(X: Tensor) -> Tensor:
    """f(X) = sign(X)"""
    return torch.sign(X)


@torch.compile(fullgraph=True)
def aro_post_update(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """Apply weight decay and parameter update."""
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)
    dtype = X[0].dtype
    U = [u.to(dtype=dtype) for u in U]
    torch._foreach_mul_(U, -adjusted_lr)
    torch._foreach_add_(X, U)
