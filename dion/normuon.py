import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import DistributedOrthoBase, megabatch_orthogonalize_async
from .opt_utils import AsyncTask, to_local
from .muon import (
    muon_update_pre_orthogonalize,
    muon_update_post_orthogonalize,
    adjust_lr_spectral_norm,
    adjust_lr_rms_norm,
)


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
        use_triton: Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func: Use a custom Newton-Schulz function for orthogonalization.
            Signature is ``func(input: Tensor, epsilon: float) -> Tensor``.
        skip_update_prob: SkipUpdate survival probability p ∈ (0, 1].
            Each parameter matrix is independently skipped with probability (1-p) at each step.
            Surviving updates are rescaled by 1/p to keep updates unbiased in expectation.
            Moment buffers always update densely (regardless of skip).
            None (default) disables SkipUpdate.
            See: "On Surprising Effectiveness of Masking Updates in Adaptive Optimizers".
        magma_tau: Magma temperature τ > 0. When set, enables Magma mode which replaces the
            fixed 1/p rescaling with an adaptive EMA scale based on momentum-gradient alignment:
              ẽ_t = sigmoid(cossim(momentum_before, grad) / τ)
              s_t = 0.9 * s_{t-1} + 0.1 * ẽ_t
            Bernoulli(0.5) masking is still applied; s_t modulates the surviving update.
            Requires skip_update_prob to also be set. None (default) uses fixed SkipUpdate scaling.
            See: "On Surprising Effectiveness of Masking Updates in Adaptive Optimizers".

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
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = "rms_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Optional[Callable] = None,
        skip_update_prob: Optional[float] = None,
        magma_tau: Optional[float] = None,
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
        # SkipUpdate / Magma: validate parameters
        if skip_update_prob is not None and not (0.0 < skip_update_prob <= 1.0):
            raise ValueError(f"skip_update_prob must be in (0, 1], got {skip_update_prob}")
        if magma_tau is not None and magma_tau <= 0.0:
            raise ValueError(f"magma_tau must be > 0, got {magma_tau}")
        if magma_tau is not None and skip_update_prob is None:
            raise ValueError("magma_tau requires skip_update_prob to be set")

        defaults = dict(
            lr=lr,
            mu=mu,
            muon_beta2=muon_beta2,
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
            skip_update_prob=skip_update_prob,  # SkipUpdate: survival prob (None = disabled)
            magma_tau=magma_tau,  # Magma: temperature for adaptive scaling (None = disabled)
        )
        super().__init__(
            params, distributed_mesh, "normuon", defaults,
            use_triton=use_triton, newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == self._algo_name and "variance_neuron" not in state:
            state["variance_neuron"] = torch.zeros_like(param[..., 0:1])
            # Magma: per-param EMA scale, init=0.5 (neutral alignment)
            state["magma_scale"] = torch.tensor(0.5, device=param.device, dtype=param.dtype)
        return state

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
                skip_update_prob=group["skip_update_prob"],  # SkipUpdate: survival probability
                magma_tau=group["magma_tau"],  # Magma: temperature (None = plain SkipUpdate)
            )

            shape_groups: dict[tuple, list] = defaultdict(list)
            for p in group_params:
                sharding = p.placements if isinstance(p, DTensor) else None
                shape_groups[(p.shape, sharding, p.dtype)].append(p)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, self._algo_name) for p in params]
                momentums = [s["momentum"] for s in states]
                variances_neuron = [s["variance_neuron"] for s in states]
                magma_scales = [s["magma_scale"] for s in states]  # Magma EMA scale per param

                is_batch_sharded, is_matrix_sharded, sharded_tensor_dim = (
                    self._get_shard_info(params[0], group)
                )

                megabatch_args = update_args
                if is_batch_sharded and not is_matrix_sharded:
                    megabatch_args = {**update_args, "process_group": None}

                yield AsyncTask(
                    normuon_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        V=variances_neuron,
                        S=magma_scales,
                        shard_dim=sharded_tensor_dim,
                        **megabatch_args,
                    )
                )


def normuon_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    S: List[Tensor],  # Magma EMA scale buffer, scalar per param (modified in place)
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
    skip_update_prob: Optional[float] = None,  # SkipUpdate: survival probability (None = disabled)
    magma_tau: Optional[float] = None,  # Magma: temperature for adaptive scaling (None = disabled)
) -> Generator[None, None, None]:
    """
    Mega-batched NorMuon update: processes ALL same-shape parameters in one
    communication round instead of world_size-sized batches.
    """
    N = len(X)
    assert N == len(G) == len(M) == len(V)

    # Magma: snapshot momentum before it's updated, for cosine similarity with current grad.
    # muon_update_pre_orthogonalize updates M in-place, so we must clone beforehand.
    G_local = to_local(G)
    M_local = to_local(M)
    if magma_tau is not None:
        M_before = [m.clone() for m in M_local]

    # Pre-orthogonalize: update momentum
    U = muon_update_pre_orthogonalize(
        G=G_local, M=M_local, momentum=momentum, nesterov=nesterov,
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
    U_stacked, V_stacked = normuon_normalization_stacked(U_stacked, V_stacked, muon_beta2)
    for i in range(N):
        V_local[i].copy_(V_stacked[i])
    U = [U_stacked[i] for i in range(N)]

    # SkipUpdate / Magma: stochastic block masking per parameter matrix.
    # Moments always update densely (above); only the final update direction is masked.
    # Reference: "On Surprising Effectiveness of Masking Updates in Adaptive Optimizers".
    if skip_update_prob is not None and skip_update_prob < 1.0:
        U = list(U)
        S_local = to_local(S)

        for i in range(len(U)):
            # Sample one Bernoulli scalar per parameter block (not per element)
            keep = torch.bernoulli(torch.tensor(skip_update_prob, device=U[i].device))

            if magma_tau is not None:
                # Magma: adaptive scale via momentum-gradient cosine similarity.
                # ẽ_t = sigmoid(cossim(μ_t_before, g_t) / τ)
                # s_t = 0.9 * s_{t-1} + 0.1 * ẽ_t   (EMA, updated in-place)
                mu = M_before[i].flatten().float()
                g = G_local[i].flatten().float()
                cos = torch.dot(mu, g) / (mu.norm() * g.norm() + 1e-8)
                e_tilde = torch.sigmoid(cos / magma_tau)
                S_local[i].mul_(0.9).add_(e_tilde * 0.1)  # EMA update in-place
                scale = S_local[i]
            else:
                # Plain SkipUpdate: fixed unbiasing rescale of 1/p
                scale = 1.0 / skip_update_prob

            U[i] = U[i] * (keep * scale)  # zero-out or scale entire matrix

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
