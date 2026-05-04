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
from .muon import muon_update_pre_orthogonalize


def _canonical_normalization(normalization: Optional[str]) -> Optional[str]:
    return canonical_normalization(normalization, allow_none=True)


class MuonH(DistributedOrthoBase):
    """
    Distributed MuonH optimizer for PyTorch FSDP2. Also compatible with DDP.

    MuonH uses Muon's momentum + orthogonalization direction, then applies a
    hyperball update: the step is scaled relative to each parameter's Frobenius
    norm, and the parameter is projected back to its initial Frobenius norm.

    .. note::
        Matrix parameters MUST be non-zero at optimizer construction. The
        hyperball radius is initialized from the parameter's Frobenius norm on
        the first step, so a zero-initialized parameter (common for output
        projections under standard Muon practice) will raise ``ValueError``.
        Use a non-zero init (e.g. Kaiming) for any layer assigned to MuonH.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base hyperball learning rate. This controls the relative move size
            before projection back to the Frobenius sphere.
        mu: Momentum factor for Muon.
        muon_beta2: EMA factor for optional normalization buffers.
        normalization: Optional update normalization. None/'None' disables it,
            'neuron' normalizes along the last dimension, and 'short_axis'
            normalizes along the shorter matrix axis.
        betas: Tuple of (beta1, beta2) for AdamW and Lion param groups.
        weight_decay: Weight decay factor for AdamW/Lion param groups.
            MuonH matrix params do not use decoupled weight decay.
        epsilon: Small value for orthogonalization.
        hyperball_eps: Small value for hyperball norm divisions.
        nesterov: Whether to use Nesterov momentum.
        adjust_lr: How to adjust the learning rate ("spectral_norm" or "rms_norm" or None).
        flatten: Whether to flatten 3D+ tensors to 2D for orthogonalization.
        use_gram_newton_schulz: Whether to use Gram Newton-Schulz for orthogonalization.
        use_triton: Whether to use Triton kernel for Newton-Schulz.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    Hyperball optimization: https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        muon_beta2: float = 0.95,
        normalization: Optional[str] = None,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        hyperball_eps: float = 1e-10,
        nesterov: bool = False,
        adjust_lr: Optional[str] = None,
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
        if hyperball_eps < 0.0:
            raise ValueError(f"Invalid hyperball_eps: {hyperball_eps}")
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
            algorithm="muonh",
            step=0,
            epsilon=epsilon,
            hyperball_eps=hyperball_eps,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        super().__init__(
            params, distributed_mesh, "muonh", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == self._algo_name and "hyperball_radius" not in state:
            local = to_local(param)
            state["hyperball_radius"] = torch.zeros(
                (), device=local.device, dtype=torch.float32
            )
            # Stored as a CPU tensor (not a Python bool) so it survives
            # state-dict round-trips through tensor-only checkpointing
            # libraries. CPU avoids a device sync on the steady-state check.
            state["hyperball_radius_initialized"] = torch.zeros(
                (), dtype=torch.bool
            )
        return state

    def _get_or_initialize_hyperball_radius(
        self, param: Tensor, state: dict, flatten: bool
    ) -> Tensor:
        local = to_local(param)
        shape = () if flatten or local.ndim == 2 else tuple(local.shape[:-2])
        radius = state.get("hyperball_radius")
        if radius is None or tuple(radius.shape) != shape:
            radius = torch.zeros(shape, device=local.device, dtype=torch.float32)
            state["hyperball_radius"] = radius
            state["hyperball_radius_initialized"] = torch.zeros((), dtype=torch.bool)
        return radius

    def _get_or_initialize_variance(
        self, param: Tensor, state: dict, normalization: Optional[str], flatten: bool
    ) -> Optional[Tensor]:
        if normalization is None:
            return None

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

        variance = state.get("variance_normalization")
        if variance is None or tuple(variance.shape) != tuple(shape):
            variance = torch.zeros(shape, device=local.device, dtype=local.dtype)
            state["variance_normalization"] = variance
        return variance

    def _initialize_hyperball_radii(
        self,
        params: List[Tensor],
        states: List[dict],
        process_group: Optional[ProcessGroup],
        flatten: bool,
        hyperball_eps: Tensor,
    ):
        if all(s["hyperball_radius_initialized"] for s in states):
            return

        local_params = to_local(params)
        sq_norms = _local_square_sums(local_params, flatten=flatten)
        if process_group is not None:
            dist.all_reduce(sq_norms, group=process_group)
        radii = sq_norms.sqrt()

        eps = float(hyperball_eps)
        if bool((radii <= eps).any().item()):
            raise ValueError(
                "MuonH requires non-zero matrix parameters because the "
                "hyperball radius is initialized from the parameter norm."
            )

        for state, radius in zip(states, radii):
            if not state["hyperball_radius_initialized"]:
                state["hyperball_radius"].copy_(radius)
                state["hyperball_radius_initialized"].fill_(True)

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched MuonH task creation: groups ALL same-shape parameters
        into a single task to minimize communication rounds and kernel launches.
        """
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "MuonH optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            normalization = _canonical_normalization(group["normalization"])
            hyperball_eps = torch.tensor(group["hyperball_eps"])
            update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
                muon_beta2=torch.tensor(group["muon_beta2"]),
                normalization=normalization,
                epsilon=torch.tensor(group["epsilon"]),
                hyperball_eps=hyperball_eps,
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                adjust_lr=group["adjust_lr"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
                newton_schulz_func=self._newton_schulz_func,
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
                    raise ValueError(
                        "MuonH does not support num_heads > 1. Split heads into "
                        "explicit parameters or use a native 3D parameter instead."
                    )

                is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                    self._get_shard_info(params[0], group)
                )
                megabatch_args = update_args
                if is_batch_sharded and not is_matrix_sharded:
                    megabatch_args = {**update_args, "process_group": None}
                shard_dim = sharded_tensor_dim

                variances = [
                    self._get_or_initialize_variance(
                        p, s, normalization, flatten=group["flatten"]
                    )
                    for p, s in zip(params, states)
                ]
                radii = [
                    self._get_or_initialize_hyperball_radius(
                        p, s, flatten=group["flatten"]
                    )
                    for p, s in zip(params, states)
                ]
                self._initialize_hyperball_radii(
                    params=params,
                    states=states,
                    process_group=megabatch_args["process_group"] if shard_dim is not None else None,
                    flatten=group["flatten"],
                    hyperball_eps=hyperball_eps,
                )

                yield AsyncTask(
                    muonh_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances,
                        R=radii,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


def muonh_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Optional[Tensor]],
    R: List[Tensor],
    lr: Tensor,
    momentum: Tensor,
    muon_beta2: Tensor,
    normalization: Optional[str],
    epsilon: Tensor,
    hyperball_eps: Tensor,
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
    newton_schulz_func: Optional[Callable] = None,
) -> Generator[None, None, None]:
    """
    Mega-batched MuonH update: Muon orthogonalization followed by optional
    normalization and a Frobenius-sphere hyperball step.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V) == len(R)

    U = muon_update_pre_orthogonalize(
        G=to_local(G), M=to_local(M), momentum=momentum, nesterov=nesterov,
    )

    comm_dim = (shard_dim - X[0].ndim) if shard_dim is not None else None
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

    if normalization is not None:
        U = yield from muonh_normalization_async(
            U=U,
            V=V,
            muon_beta2=muon_beta2,
            normalization=normalization,
            param_shape=X[0].shape,
            flatten=flatten,
            shard_dim=shard_dim,
            process_group=process_group,
            epsilon=hyperball_eps,
        )

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)
    else:
        raise ValueError(f"Unknown adjust_lr value: {adjust_lr}")

    yield from muonh_update_post_orthogonalize_async(
        X=to_local(X),
        U=U,
        R=R,
        adjusted_lr=adjusted_lr,
        epsilon=hyperball_eps,
        flatten=flatten,
        process_group=process_group if shard_dim is not None else None,
    )


@torch.compile(fullgraph=True)
def _muonh_sum_sq_compiled(U_stacked: Tensor, red_dim: int) -> Tensor:
    return U_stacked.float().square().sum(dim=red_dim, keepdim=True)


@torch.compile(fullgraph=True)
def _muonh_normalize_finish_compiled(
    U_stacked: Tensor,
    V_stacked: Tensor,
    sum_sq: Tensor,
    red_dim_size: int,
    muon_beta2: Tensor,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    V_dtype = V_stacked.dtype
    variance_new = (sum_sq / red_dim_size).to(dtype=V_dtype)
    norm_U = U_stacked.float().norm(p=2, dim=(-2, -1), keepdim=True)
    V_stacked = torch.lerp(V_stacked, variance_new, 1 - muon_beta2)
    normalized_U = U_stacked.float() / (V_stacked.float().sqrt() + epsilon)
    norm_U_new = normalized_U.norm(p=2, dim=(-2, -1), keepdim=True).clamp(min=epsilon)
    normalized_U = (normalized_U * (norm_U / norm_U_new)).to(dtype=V_dtype)
    return normalized_U, V_stacked


def muonh_normalization_async(
    U: List[Tensor],
    V: List[Tensor],
    muon_beta2: Tensor,
    normalization: str,
    param_shape: torch.Size,
    flatten: bool,
    shard_dim: Optional[int],
    process_group: Optional[ProcessGroup],
    epsilon: Tensor,
) -> Generator[None, None, List[Tensor]]:
    """Optional MuonH update normalization, with reductions when the normalized axis is sharded.

    Pure-tensor pre-/post-comm sections are compiled; the async dist.all_reduce
    sits in between and stays in eager Python.
    """
    U_stacked = torch.stack(U)
    V_stacked = torch.stack(V)
    original_shape = U_stacked.shape

    local_shape = U[0].shape

    if flatten and len(param_shape) >= 3:
        U_stacked = U_stacked.flatten(start_dim=2)
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
    if normalization == "short_axis" and flat_shape[-2] < flat_shape[-1]:
        red_dim = -2

    sum_sq = _muonh_sum_sq_compiled(U_stacked, red_dim)
    if process_group is not None and shard_dim_neg == red_dim:
        work = dist.all_reduce(sum_sq, group=process_group, async_op=True)
        yield
        work.wait()

    normalized_U, V_stacked = _muonh_normalize_finish_compiled(
        U_stacked, V_stacked, sum_sq, flat_shape[red_dim], muon_beta2, float(epsilon),
    )

    if flatten and len(param_shape) >= 3:
        normalized_U = normalized_U.reshape(original_shape)

    torch._foreach_copy_(list(V), list(V_stacked.unbind(0)))
    return list(normalized_U.unbind(0))


@torch.compile(fullgraph=True)
def _muonh_local_sums_compiled(
    X_stacked: Tensor, U_stacked: Tensor, flatten: bool
) -> Tensor:
    """Per-matrix local |U|^2, |X|^2, <X, U> packed into one fp32 tensor.

    Packing into one tensor lets the caller fuse the three reductions into a
    single async all-reduce instead of issuing one round per quantity.
    """
    X_f = X_stacked.float()
    U_f = U_stacked.float()
    red_dims = tuple(range(1, X_stacked.ndim)) if flatten else (-2, -1)
    u_sq = U_f.square().sum(dim=red_dims)
    x_sq = X_f.square().sum(dim=red_dims)
    xu = (X_f * U_f).sum(dim=red_dims)
    return torch.stack([u_sq, x_sq, xu], dim=0)


@torch.compile(fullgraph=True)
def _muonh_apply_step_compiled(
    X_stacked: Tensor,
    U_stacked: Tensor,
    sums: Tensor,
    radii: Tensor,
    adjusted_lr: Tensor,
    eps: float,
    flatten: bool,
) -> Tensor:
    """Apply hyperball step + Frobenius-sphere projection in one fused multiply-subtract.

    Uses the closed form |X - s*U|^2 = |X|^2 - 2*s*<X,U> + s^2*|U|^2 to skip
    materializing the intermediate candidate tensor and a second reduction.
    """
    x_dtype = X_stacked.dtype
    u_sq = sums[0]
    x_sq = sums[1]
    xu = sums[2]

    u_norm = u_sq.sqrt().clamp_min(eps)
    s_uf = adjusted_lr * radii / u_norm  # = lr * r / |U|
    candidate_sq = (x_sq - 2.0 * s_uf * xu + s_uf * s_uf * u_sq).clamp_min(eps * eps)
    candidate_norm = candidate_sq.sqrt()
    scale_X_f = radii / candidate_norm   # = r / |candidate|
    scale_U_f = s_uf * scale_X_f         # = lr*r / (|U| * |candidate|) * r

    extra = X_stacked.ndim - 1 if flatten else 2
    scale_X = scale_X_f.to(dtype=x_dtype)
    scale_U = scale_U_f.to(dtype=x_dtype)
    for _ in range(extra):
        scale_X = scale_X.unsqueeze(-1)
        scale_U = scale_U.unsqueeze(-1)
    return X_stacked * scale_X - U_stacked * scale_U


def muonh_update_post_orthogonalize_async(
    X: List[Tensor],
    U: List[Tensor],
    R: List[Tensor],
    adjusted_lr: Tensor,
    epsilon: Tensor,
    flatten: bool,
    process_group: Optional[ProcessGroup],
) -> Generator[None, None, None]:
    """Apply a scale-invariant hyperball step and project back to stored radii.

    Folds the two FSDP2 reductions (update Frobenius norm and post-step
    Frobenius norm) into a single all-reduce of [|U|^2, |X|^2, <X, U>], using
    the closed form for |candidate|^2.
    """
    eps = float(epsilon)
    radii = torch.stack(list(R))
    x_dtype = X[0].dtype

    U_stacked = torch.stack(U).to(dtype=x_dtype)
    X_stacked = torch.stack(list(X))

    sums = _muonh_local_sums_compiled(X_stacked, U_stacked, flatten)
    if process_group is not None:
        work = dist.all_reduce(sums, group=process_group, async_op=True)
        yield
        work.wait()

    final_stacked = _muonh_apply_step_compiled(
        X_stacked, U_stacked, sums, radii, adjusted_lr, eps, flatten,
    )
    torch._foreach_copy_(list(X), list(final_stacked.unbind(0)))


def _local_square_sums(tensors: List[Tensor], flatten: bool) -> Tensor:
    """Per-tensor sum of squares, fused via a single stack + flatten + sum."""
    stacked = torch.stack(tensors).float()
    if flatten:
        return stacked.square().flatten(1).sum(dim=-1)
    return stacked.square().sum(dim=(-2, -1))
