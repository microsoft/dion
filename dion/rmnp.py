import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import ParamsT
from typing import Generator, List, Optional, Tuple, Union

from .megabatch_base import DistributedOrthoBase, compute_split_lr_scales
from .muon import muon_update_megabatch_async
from .opt_utils import AsyncTask


@torch.compile(fullgraph=True)
def row_normalize(X: Tensor, epsilon: float = 1e-8) -> Tensor:
    """
    RMNP preconditioner: row-wise (input-dimension, ``d_in``) ℓ₂ normalization.

    Drop-in replacement for Muon's Newton-Schulz orthogonalization. For an
    update matrix ``X`` of shape ``(d_out, d_in)`` (rows indexed by output
    neuron, columns by input feature), every row is divided by its own ℓ₂
    norm over the input dimension::

        row_normalize(X)[i, :] = X[i, :] / (‖X[i, :]‖₂ + epsilon)

    which is exactly ``(diag(X Xᵀ))^{-1/2} X``. RMNP thus replaces Muon's full
    orthogonalization factor ``(X Xᵀ)^{-1/2}`` with its diagonal
    ``(diag(X Xᵀ))^{-1/2}``, and the two coincide when ``X Xᵀ`` is
    (block-)diagonally dominant -- the empirically observed structure of the
    Transformer layerwise Hessian that motivates RMNP.

    Batched inputs of shape ``(..., d_out, d_in)`` are normalized along the
    last axis, matching the ``func(input, epsilon) -> Tensor`` contract of the
    Newton-Schulz functions that this replaces. The cost is ``O(d_out · d_in)``
    versus ``O(d_out · d_in · min(d_out, d_in))`` for Newton-Schulz.

    RMNP: Row-Momentum Normalized Preconditioning, arXiv:2603.20527.
    """
    return X / (X.norm(p=2, dim=-1, keepdim=True) + epsilon)


class RMNP(DistributedOrthoBase):
    """
    Distributed RMNP optimizer for PyTorch FSDP2. Also compatible with DDP.

    RMNP (Row-Momentum Normalized Preconditioning) replaces Muon's Newton-Schulz
    orthogonalization with a simple row-wise (input-dimension, ``d_in``) ℓ₂
    normalization of the momentum update (see :func:`row_normalize`). This drops
    the per-iteration cost from ``O(mn·min(m, n))`` to ``O(mn)`` for an ``m×n``
    weight while, in the paper's experiments, matching Muon-level optimization
    quality: orthogonalization and row-wise ℓ₂ normalization are shown to be
    asymptotically equivalent for the Transformer. Everything else -- the
    momentum buffer, weight decay, and weight update -- follows Muon, so RMNP
    reuses Muon's distributed assembly and inherits the same sharding support.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
            Use DeviceMesh for FSDP2 and ProcessGroup for DistributedDataParallel.
        lr: Base learning rate. With ``adjust_lr`` set, this is scaled based on
            the matrix dimensions; with ``adjust_lr=None`` (the RMNP default) it
            is used as-is, matching the paper's ``W ← W − η·row_normalize(V)``.
            For element-wise update rules, this is the actual learning rate and
            no additional scaling is done.
        mu: Momentum factor for RMNP algorithm.
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor.
        cautious_wd: Whether to apply weight decay only where update and parameter signs align.
        epsilon: Small value added to each row norm to avoid division by zero.
        nesterov: Whether to use Nesterov momentum. Off by default, matching the paper.
        adjust_lr: How to adjust the learning rate ("spectral_norm" or "rms_norm" or None).
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            None: Do not adjust the learning rate. This is the RMNP default and
            matches the paper, which tunes the matrix learning rate directly and
            applies no post-normalization rescaling; the other options are
            offered for consistency with the rest of the family but were not
            studied in the RMNP paper.
        flatten: Whether to flatten 3D+ tensors to 2D for RMNP updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.

    Because row normalization is scale-invariant, the paper's momentum EMA
    ``V ← β·V + (1−β)·G`` and Muon's ``M ← μ·M + G`` produce the same normalized
    update (they differ only by the constant factor ``1−β``, which cancels), so
    RMNP reuses Muon's momentum machinery unchanged.

    Param groups may also set the ``num_heads`` or ``split_sizes`` options,
    which behave exactly as in Muon (they operate on the assembled matrices and
    each row/block is normalized independently). See the README for details.

    RMNP optimizer: https://arxiv.org/abs/2603.20527
    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Optional[str] = None,
        flatten: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            cautious_wd=cautious_wd,
            algorithm="rmnp",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
        )
        # Fix the preconditioner to row normalization: passing newton_schulz_func
        # short-circuits the base class's Newton-Schulz/Polar-Express selection,
        # so RMNP has no use_triton / use_polar_express / use_gram_newton_schulz
        # options -- there is no Newton-Schulz iteration to accelerate.
        super().__init__(
            params, distributed_mesh, "rmnp", defaults,
            newton_schulz_func=row_normalize,
        )

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        """
        Mega-batched RMNP task creation: groups ALL same-shape parameters into a
        single task to minimize communication rounds and kernel launches. This
        mirrors Muon and reuses ``muon_update_megabatch_async``; the only
        difference is the orthogonalization function, which is
        :func:`row_normalize` (set on ``self._newton_schulz_func``).
        """
        for group in param_groups:
            assert group["algorithm"] == "rmnp"
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "RMNP optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            update_args = dict(
                lr=torch.tensor(group["lr"]),
                momentum=torch.tensor(group["mu"]),
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
            split_sizes = self._resolve_split_sizes(group)

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "rmnp") for p in params]
                momentums = [s["momentum"] for s in states]

                split_args = {}
                if split_sizes is not None:
                    self._validate_split_shape(split_sizes, params)
                    split_args = dict(
                        split_sizes=split_sizes,
                        split_scales=compute_split_lr_scales(
                            split_sizes, params[0].shape, group["adjust_lr"]
                        ),
                    )

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
                    muon_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=shard_dim,
                        **split_args,
                        **megabatch_args,
                    )
                )
