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
    canonical_normalization,
    megabatch_orthogonalize_async,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)
from .opt_utils import AsyncTask, to_local
from .muon import muon_update_pre_orthogonalize, muon_update_post_orthogonalize


def _canonical_normalization(normalization: str) -> str:
    return canonical_normalization(normalization, allow_none=False)


class NorMuon(DistributedOrthoBase):
    """
    Distributed NorMuon optimizer for PyTorch FSDP2. Also compatible with DDP.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. For NorMuon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu: Momentum factor for NorMuon algorithm.
        muon_beta2: Second beta parameter for NorMuon algorithm's adaptive updates.
        normalization: Update normalization mode. 'neuron' (default) normalizes
            along the last dimension. 'short_axis' normalizes along the shorter
            matrix axis (equivalent to 'neuron' for tall matrices, but for wide
            matrices reduces along rows instead of columns).
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        cautious_wd: Whether to apply weight decay only where update and parameter signs align.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
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

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    NorMuon optimizer: https://arxiv.org/abs/2510.05491
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        normalization: str = "neuron",
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
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        normalization = _canonical_normalization(normalization)
        defaults = dict(
            lr=lr,
            mu=mu,
            muon_beta2=muon_beta2,
            normalization=normalization,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="normuon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(
            params, distributed_mesh, "normuon", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        # Variance buffer is initialized lazily in _get_or_initialize_variance
        # since its shape depends on the group's normalization mode.
        return super()._get_or_initialize_state(param, algo)

    def _get_or_initialize_variance(
        self, param: Tensor, state: dict, normalization: str, flatten: bool
    ) -> Tensor:
        """Lazily allocate the per-param variance EMA buffer.

        Note on resume / mode switch: the buffer is keyed by shape, so loading
        a checkpoint and then changing the ``normalization`` mode (e.g.
        ``neuron`` -> ``short_axis`` for a wide matrix) silently re-initializes
        the EMA to zero. This is intentional: the saved variance is no longer
        meaningful under the new mode, and a hard error would force the user
        to manually clear state. The trade-off is a brief warm-up period where
        the EMA is rebuilding from the new reduction axis.
        """
        local = to_local(param)
        if flatten and local.ndim >= 3:
            shape = [local.shape[0], math.prod(local.shape[1:])]
        else:
            shape = list(local.shape)
        if normalization == "neuron":
            shape[-1] = 1
        elif normalization == "short_axis":
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            shape[red_dim] = 1
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        variance = state.get("variance_neuron")
        if variance is None or tuple(variance.shape) != tuple(shape):
            variance = torch.zeros(shape, device=local.device, dtype=local.dtype)
            state["variance_neuron"] = variance
        return variance

    def _get_shard_info(self, param: Tensor, group: dict):
        result = super()._get_shard_info(param, group)
        _, is_matrix_sharded, sharded_tensor_dim = result
        if is_matrix_sharded and sharded_tensor_dim == param.ndim - 1:
            raise NotImplementedError(
                "NorMuon currently does not support parameters sharded along the last dimension. "
                "Please avoid shards at dim -1."
            )
        return result

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched NorMuon task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "NorMuon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            normalization = _canonical_normalization(group["normalization"])
            update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                weight_decay=torch.tensor(group["weight_decay"]),
                normalization=normalization,
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

                # Variance buffer shape depends on normalization + (post-split)
                # param shape, so initialize after head split.
                variances_neuron = [
                    self._get_or_initialize_variance(
                        p, s, normalization, flatten=group["flatten"]
                    )
                    for p, s in zip(params, states)
                ]

                yield AsyncTask(
                    normuon_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances_neuron,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


def normuon_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    momentum: Tensor,
    muon_beta2: Tensor,
    weight_decay: Tensor,
    normalization: str,
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
    Mega-batched NorMuon update: processes ALL same-shape parameters in one
    communication round instead of world_size-sized batches.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V)

    # Pre-orthogonalize: update momentum
    U = muon_update_pre_orthogonalize(
        G=to_local(G), M=to_local(M), momentum=momentum, nesterov=nesterov,
    )

    # Convert shard_dim to negative for comm_dim
    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None

    # Orthogonalize via shared megabatch communication
    U = yield from megabatch_orthogonalize_async(
        U,
        comm_dim=comm_dim,
        device_rank=device_rank,
        world_size=world_size,
        process_group=process_group,
        newton_schulz_func=newton_schulz_func,
        flatten=flatten,
        epsilon=epsilon,
    )

    # NorMuon normalization using stacked tensors for fewer kernel launches
    V_local = to_local(V)
    U_stacked = torch.stack(U)
    V_stacked = torch.stack(V_local)
    if normalization == "neuron":
        if flatten and X[0].ndim >= 3:
            U_stacked, V_stacked = normuon_normalization_stacked_flattened(
                U_stacked, V_stacked, muon_beta2,
            )
        else:
            U_stacked, V_stacked = normuon_normalization_stacked(
                U_stacked, V_stacked, muon_beta2,
            )
    elif normalization == "short_axis":
        U_stacked, V_stacked = yield from normuon_short_axis_normalization_async(
            U_stacked,
            V_stacked,
            muon_beta2=muon_beta2,
            param_shape=X[0].shape,
            flatten=flatten,
            shard_dim=shard_dim,
            process_group=process_group,
        )
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    for i in range(N):
        V_local[i].copy_(V_stacked[i])
    U = [U_stacked[i] for i in range(N)]

    # Compute scaled learning rate
    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    # Post-orthogonalize: apply update
    muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        cautious_wd=cautious_wd,
    )


@torch.compile(fullgraph=True)
def _normuon_short_axis_sum_sq_compiled(U: Tensor, red_dim: int) -> Tensor:
    return U.float().square().sum(dim=red_dim, keepdim=True)


@torch.compile(fullgraph=True)
def _normuon_short_axis_finish_compiled(
    U: Tensor,
    V: Tensor,
    sum_sq: Tensor,
    red_dim_size: int,
    muon_beta2: Tensor,
) -> Tuple[Tensor, Tensor]:
    V_dtype = V.dtype
    U_v = U.to(dtype=V_dtype)
    norm_U = U_v.norm(p=2, dim=(-2, -1), keepdim=True)
    variance_new = (sum_sq / red_dim_size).to(dtype=V_dtype)
    V = torch.lerp(V, variance_new, 1 - muon_beta2)
    denom = V.sqrt() + 1e-8
    normalized_U = U_v / denom
    norm_U_new = normalized_U.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    normalized_U = normalized_U * (norm_U / norm_U_new)
    return normalized_U, V


def normuon_short_axis_normalization_async(
    U: Tensor,  # [N, rows, cols]
    V: Tensor,  # [N, ..., 1]   (variance buffer, with 1 along the long axis)
    muon_beta2: Tensor,
    param_shape: torch.Size,
    flatten: bool,
    shard_dim: Optional[int],
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, Tuple[Tensor, Tensor]]:
    """
    Short-axis NorMuon normalization: variance is tracked per index along the
    shorter matrix axis (reduction is over the longer axis). Equivalent to
    'neuron' for tall matrices; for wide matrices it reduces along rows
    instead of columns. An async all-reduce is issued when the reduced axis
    is sharded.

    Pure-tensor pre-/post-comm sections are compiled; the async dist.all_reduce
    sits in between and stays in eager Python.
    """
    original_shape = U.shape
    local_shape = U.shape[1:]
    if flatten and len(param_shape) >= 3:
        U = U.flatten(start_dim=2)
        flat_shape = torch.Size((local_shape[0], math.prod(local_shape[1:])))
        shard_dim_neg = None
        if shard_dim is not None:
            shard_dim_pos = shard_dim if shard_dim >= 0 else shard_dim + len(param_shape)
            shard_dim_neg = -2 if shard_dim_pos == 0 else -1
    else:
        flat_shape = local_shape
        shard_dim_neg = None
        if shard_dim is not None:
            shard_dim_neg = shard_dim if shard_dim < 0 else shard_dim - len(local_shape)

    red_dim = -1
    if flat_shape[-2] < flat_shape[-1]:
        red_dim = -2

    sum_sq = _normuon_short_axis_sum_sq_compiled(U, red_dim)
    if process_group is not None and shard_dim_neg == red_dim:
        work = dist.all_reduce(sum_sq, group=process_group, async_op=True)
        yield
        work.wait()

    normalized_U, V = _normuon_short_axis_finish_compiled(
        U, V, sum_sq, flat_shape[red_dim], muon_beta2,
    )
    if flatten and len(param_shape) >= 3:
        normalized_U = normalized_U.reshape(original_shape)
    return normalized_U, V


@torch.compile(fullgraph=True)
def normuon_normalization_stacked(
    U: Tensor,  # [N, rows, cols]
    V: Tensor,  # [N, rows, 1]  (variance neuron buffer)
    muon_beta2: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    NorMuon normalization on stacked 3D tensors for minimal kernel launches.
    Equivalent to normuon_normalization but operates on a single stacked tensor
    instead of a list, reducing per-element kernel overhead.
    Returns (normalized_U, updated_V).
    """
    V_dtype = V.dtype
    U = U.to(dtype=V_dtype)

    # Frobenius norm per matrix: [N, 1, 1]
    norm_U = U.norm(p=2, dim=(-2, -1), keepdim=True)

    # Neuron-wise variance: mean of squares along last dim -> [N, rows, 1]
    neuron_norms = (U * U).mean(dim=-1, keepdim=True)

    # Update variance buffer (EMA)
    V = torch.lerp(V, neuron_norms, 1 - muon_beta2)

    # Normalize
    denom = V.sqrt() + 1e-8
    normalized_U = U / denom

    # Rescale to preserve Frobenius norm
    norm_U_new = normalized_U.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    normalized_U = normalized_U * (norm_U / norm_U_new)

    return normalized_U, V


@torch.compile(fullgraph=True)
def normuon_normalization_stacked_flattened(
    U: Tensor,
    V: Tensor,
    muon_beta2: Tensor,
) -> Tuple[Tensor, Tensor]:
    original_shape = U.shape
    U = U.flatten(start_dim=2)
    normalized_U, V = normuon_normalization_stacked(U, V, muon_beta2)
    return normalized_U.reshape(original_shape), V
