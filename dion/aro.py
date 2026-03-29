import math
import torch
import torch.distributed as dist
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import DistributedOrthoBase
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
    Megabatched ARO update. Distributes the per-parameter ARO computation
    (QR + matmuls) across ranks via all-gather, matching the DDP pattern
    used by Muon/NorMuon for Newton-Schulz.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(R)

    M_local = to_local(M)
    G_local = to_local(G)

    # Update momentum: M = mu * M + (1-mu) * G
    G_cast = [g.to(dtype=m.dtype) for g, m in zip(G_local, M_local)]
    torch._foreach_lerp_(M_local, G_cast, 1 - momentum)

    base_opt_fn = _get_base_opt_fn(base_opt)

    if N > 1 and process_group is not None:
        # --- Distributed DDP megabatch ---
        pad_n = (world_size - N % world_size) % world_size
        if pad_n > 0:
            M_work = M_local + [torch.zeros_like(M_local[0])] * pad_n
            R_work = R + [torch.eye(
                R[0].shape[-1], device=R[0].device, dtype=R[0].dtype
            ).expand_as(R[0]).clone() for _ in range(pad_n)]
        else:
            M_work = M_local
            R_work = R

        N_total = len(M_work)
        per_rank = N_total // world_size

        start = device_rank * per_rank
        my_M = torch.stack(M_work[start : start + per_rank]).float()
        my_R = torch.stack(R_work[start : start + per_rank])

        my_U, my_R_new = _aro_step_batched(my_M, my_R, base_opt_fn)

        # All-gather update directions and new rotations concurrently
        all_U = [torch.empty_like(my_U) for _ in range(world_size)]
        all_R = [torch.empty_like(my_R_new) for _ in range(world_size)]
        work_u = dist.all_gather(
            all_U, my_U.contiguous(), group=process_group, async_op=True
        )
        work_r = dist.all_gather(
            all_R, my_R_new.contiguous(), group=process_group, async_op=True
        )
        yield
        work_u.wait()
        work_r.wait()

        U_list = [all_U[r][i] for r in range(world_size) for i in range(per_rank)][:N]
        R_new_list = [all_R[r][i] for r in range(world_size) for i in range(per_rank)][:N]

    elif N == 1:
        U, R_new = _aro_step_single(M_local[0].float(), R[0], base_opt_fn)
        U_list = [U]
        R_new_list = [R_new]

    else:
        # N > 1, no process_group (single GPU or batch-sharded)
        M_stack = torch.stack(M_local).float()
        R_stack = torch.stack(R)
        U_stack, R_new_stack = _aro_step_batched(M_stack, R_stack, base_opt_fn)
        U_list = [U_stack[i] for i in range(N)]
        R_new_list = [R_new_stack[i] for i in range(N)]

    # Update rotation state in-place
    for r, r_new in zip(R, R_new_list):
        r.copy_(r_new)

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


def _aro_step_single(M: Tensor, R: Tensor, base_opt_fn) -> Tuple[Tensor, Tensor]:
    """ARO step for a single parameter. M: [m, n] float32, R: [m, m] float32."""
    # Rotate gradient into R's frame
    rotated = R.mT @ M
    f_rotated = base_opt_fn(rotated)

    # Cross-alignment → QR for new rotation
    cross = M @ f_rotated.mT
    Q, _ = torch.linalg.qr(cross)

    # Re-rotate with new R, apply base opt, rotate back
    rotated_new = Q.mT @ M
    f_new = base_opt_fn(rotated_new)
    U = Q @ f_new

    return U, Q


def _aro_step_batched(M: Tensor, R: Tensor, base_opt_fn) -> Tuple[Tensor, Tensor]:
    """Batched ARO step. M: [N, m, n] float32, R: [N, m, m] float32."""
    rotated = R.mT @ M
    f_rotated = base_opt_fn(rotated)

    cross = M @ f_rotated.mT
    Q, _ = torch.linalg.qr(cross)

    rotated_new = Q.mT @ M
    f_new = base_opt_fn(rotated_new)
    U = Q @ f_new

    return U, Q


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
