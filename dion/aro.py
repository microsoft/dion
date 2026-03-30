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
            "sinkhorn": Alternating row/column L2 normalization (recommended).
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
        base_opt: str = "sinkhorn",
        adjust_lr: Optional[str] = "rms_norm",
        flatten: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor: {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if base_opt not in ("row_norm", "sign", "sinkhorn"):
            raise ValueError(
                f"Invalid base_opt: {base_opt}. Must be 'row_norm', 'sign', or 'sinkhorn'."
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

                megabatch_args = update_args
                if is_batch_sharded and not is_matrix_sharded:
                    megabatch_args = {**update_args, "process_group": None}

                yield AsyncTask(
                    aro_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        R=rotations,
                        shard_dim=sharded_tensor_dim,
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
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
) -> Generator[None, None, None]:
    """
    Megabatched ARO update. Distributes the full ARO computation (rotation,
    cross-alignment, QR, update direction) across ranks via the shared
    megabatch_orthogonalize_async infrastructure.

    For FSDP: the all-to-all reassembles full matrices before the ARO step;
    for DDP: each rank processes its assigned params then all-gathers.
    In both cases, the ARO computation runs inside a closure that captures
    the rotation matrices R for the assigned params.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(R)

    M_local = to_local(M)
    G_local = to_local(G)

    # Update momentum: M = mu * M + (1-mu) * G
    G_cast = [g.to(dtype=m.dtype) for g, m in zip(G_local, M_local)]
    torch._foreach_lerp_(M_local, G_cast, 1 - momentum)

    base_opt_fn = _get_base_opt_fn(base_opt)
    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None

    # Determine which params are assigned to this rank so the closure
    # can access the right rotation matrices.
    pad_n = (world_size - N % world_size) % world_size if process_group is not None and N > 1 else 0
    N_total = N + pad_n
    per_rank = N_total // world_size if process_group is not None and N > 1 else N
    start = device_rank * per_rank if process_group is not None and N > 1 else 0

    # Pad R to match padded M list, stack this rank's assigned rotations
    R_padded = R + [torch.eye(
        R[0].shape[-1], device=R[0].device, dtype=R[0].dtype
    )] * pad_n if pad_n > 0 else R
    R_my = torch.stack(R_padded[start : start + per_rank])
    R_new_holder = [None]

    def aro_ortho_fn(M_batch, epsilon=None):
        """Full ARO step: rotation → base_opt → cross-alignment → QR → update.

        M_batch is [per_rank, m, n] — full (unsharded) matrices for the
        params assigned to this rank, after all-to-all reassembly (FSDP)
        or direct stacking (DDP).

        """
        M_f32 = M_batch.float()

        # Phase 1: compute cross-alignment matrix
        rotated = R_my.mT @ M_f32
        f_rotated = base_opt_fn(rotated)
        del rotated
        cross = M_f32 @ f_rotated.mT
        del f_rotated, M_f32

        # Phase 2: Shifted Cholesky QR — uses only matmul + Cholesky +
        # triangular solve, avoiding Householder QR's large workspace.
        # Release cached-but-free memory so cusolver can allocate handles
        # and workspace (it allocates outside PyTorch's caching allocator).
        torch.cuda.empty_cache()
        Q = _shifted_cholesky_qr(cross)
        del cross
        R_new_holder[0] = Q

        # Phase 3: compute update direction with new rotation
        M_f32 = M_batch.float()
        rotated_new = Q.mT @ M_f32
        f_new = base_opt_fn(rotated_new)
        del rotated_new, M_f32
        return (Q @ f_new).to(M_batch.dtype)

    # Distribute ARO computation via shared megabatch infrastructure.
    # For FSDP: all-to-all reassembles full M, aro_ortho_fn runs on full
    #   matrices, result is scattered back.
    # For DDP: each rank runs aro_ortho_fn on its assigned chunk, all-gather.
    U_list = yield from megabatch_orthogonalize_async(
        M_local,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=aro_ortho_fn,
        flatten=False,
        epsilon=epsilon,
    )

    # Update rotation state for this rank's assigned params
    if R_new_holder[0] is not None:
        Q_new = R_new_holder[0]
        for i in range(per_rank):
            idx = start + i
            if idx < N:
                R[idx].copy_(Q_new[i])

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


def _shifted_cholesky_qr(A: Tensor) -> Tensor:
    """Orthogonalize A via Shifted Cholesky QR.

    Uses matmul + Cholesky + triangular solve, which need far less
    GPU workspace than Householder QR (torch.linalg.qr). Adds a
    small shift to the Gram matrix diagonal for numerical stability.

    If Cholesky fails (input too ill-conditioned), falls back to
    Householder QR.

    Same approach as dion.py's orthogonalize() and the ARO paper's
    recommended implementation.
    """
    G = A.mT @ A  # Gram matrix [*, m, m], via cuBLAS
    # Shift proportional to the Frobenius norm of A
    shift = A.norm() ** 2 * 1e-7
    G.diagonal(dim1=-2, dim2=-1).add_(shift)
    # Synchronize + release cached memory before cusolver call.
    # Without synchronize, empty_cache cannot release blocks with pending
    # ops on other CUDA streams (e.g. NCCL all-to-all, torch.compile).
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # Upper Cholesky: G = R^T R, then Q = A @ R^{-1}
    R, info = torch.linalg.cholesky_ex(G, upper=True)
    if (info != 0).any():
        # Fallback: Householder QR for ill-conditioned inputs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        Q, _ = torch.linalg.qr(A)
        return Q
    return torch.linalg.solve_triangular(R, A, upper=True, left=False)


def _get_base_opt_fn(base_opt: str):
    if base_opt == "row_norm":
        return _base_opt_row_norm
    elif base_opt == "sign":
        return _base_opt_sign
    elif base_opt == "sinkhorn":
        return _base_opt_sinkhorn
    raise ValueError(f"Unknown base_opt: {base_opt}")


def _base_opt_row_norm(X: Tensor) -> Tensor:
    """f(X) = sqrt(n) * X / ||x_i||_row"""
    n = X.shape[-1]
    row_norms = X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return math.sqrt(n) * X / row_norms


def _base_opt_sign(X: Tensor) -> Tensor:
    """f(X) = sign(X)"""
    return torch.sign(X)


def _base_opt_sinkhorn(X: Tensor, num_iters: int = 5, eps: float = 1e-8) -> Tensor:
    """SR-Sinkhorn normalization: alternating L2 row/column normalization.

    Each iteration normalizes rows to have L2 norm sqrt(cols), then
    columns to have L2 norm sqrt(rows). This corresponds to the
    square-root iterates of the classical Sinkhorn algorithm applied
    to the matrix of squared entries.

    Reference: https://arxiv.org/abs/2502.06742
    """
    m, n = X.shape[-2], X.shape[-1]
    for _ in range(num_iters):
        # Row normalization: each row gets L2 norm sqrt(n)
        row_norms = X.norm(dim=-1, keepdim=True).clamp(min=eps)
        X = X * (math.sqrt(n) / row_norms)
        # Column normalization: each column gets L2 norm sqrt(m)
        col_norms = X.norm(dim=-2, keepdim=True).clamp(min=eps)
        X = X * (math.sqrt(m) / col_norms)
    return X


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
