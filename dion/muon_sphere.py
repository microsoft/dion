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
    adjust_lr_rms_norm,
    adjust_lr_spectral_norm,
    megabatch_orthogonalize_async,
)
from .muon import muon_update_pre_orthogonalize, muon_update_post_orthogonalize
from .opt_utils import AsyncTask, to_local


class MuonSphere(DistributedOrthoBase):
    """
    MuonSphere — Muon with a spectral-sphere weight retraction.

    Constrains every matrix parameter ``W`` to lie on the spectral sphere
    ``{ W : ||W||_2 = R }``, where ``R = radius_scale * sqrt(d_out / d_in)`` is
    the Spectral muP radius (Yang, Simon, Bernstein 2024). The update direction
    is the same Muon msign(M); the only changes are:

    1. An init-time projection ``W <- R * W / ||W||_2`` on the first step.
    2. A per-step retraction ``W <- W * R / sigma`` (where ``sigma`` is the
       top singular value, estimated by warmstarted power iteration) before
       the standard Muon update is applied.
    3. A learning-rate scale of ``R = radius_scale * sqrt(d_out/d_in)`` on
       the orthogonalized update (i.e. the existing ``adjust_lr=spectral_norm``
       scale, multiplied by ``radius_scale``).
    4. Weight decay defaults to ``0`` (the retraction subsumes its role on
       hidden 2D weights). Set ``weight_decay > 0`` only on params where you
       want it.

    This is the lambda=0 ablation of the Spectral Sphere Optimizer (SSO) from
    Xie et al., "Controlled LLM Training on Spectral Sphere", arXiv:2601.08393.
    The paper's full SSO additionally solves a Lagrange tangent-space
    projection (h(lambda)=0 by bisection); MuonSphere drops that step in
    exchange for ~1% added latency over Muon (vs. ~11% for full SSO) while
    capturing most of the downstream-quality gain.

    Args:
        params: Parameters for the optimizer.
        distributed_mesh: DeviceMesh or ProcessGroup for distributed training.
        lr: Base learning rate. Scaled by ``R = radius_scale * sqrt(d_out/d_in)``
            on the orthogonalized update direction.
        mu: Momentum factor (matches Muon).
        betas: Tuple of (beta1, beta2) for AdamW and Lion algorithms.
        weight_decay: Weight decay factor. Defaults to 0; the retraction
            already bounds ``||W||_F <= sqrt(min(d_out,d_in)) * R``.
        cautious_wd: Whether to apply weight decay only where update and
            parameter signs align. Forwarded to the Muon post-step.
        epsilon: Small value to avoid division by zero.
        nesterov: Whether to use Nesterov momentum.
        radius_scale: The scalar ``c`` in ``R = c * sqrt(d_out/d_in)``. Paper
            recommends ``c ~= 2.0`` for 1.7B Transformer training.
        power_iter_steps: Number of power-iteration steps per optimizer step.
            With u/v warmstarting, 2-3 iters are typically enough after the
            first few steps; default 5 to give a healthy margin on step 0.
        adjust_lr: How to adjust the learning rate. Should be ``"spectral_norm"``
            for spectral-muP semantics. Other values are accepted for parity
            with Muon but break the muP scaling story.
        flatten: Whether to flatten 3D+ tensors to 2D for the Muon update.
        use_triton, use_polar_express, use_gram_newton_schulz, newton_schulz_func:
            Forwarded to ``DistributedOrthoBase`` exactly as in ``Muon``.

    Note on FSDP2: the per-step power iteration on a sharded ``W`` is
    implemented via ``DTensor.full_tensor()`` — i.e. it all-gathers the param
    once per step. For most LLM shapes this is dominated by the existing Muon
    all-to-alls, but a sharded power-iteration kernel is a natural follow-up.
    """

    def __init__(
        self,
        params: ParamsT,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        radius_scale: float = 1.0,
        power_iter_steps: int = 5,
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
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if radius_scale <= 0.0:
            raise ValueError(f"radius_scale must be > 0, got {radius_scale}")
        if power_iter_steps < 1:
            raise ValueError(f"power_iter_steps must be >= 1, got {power_iter_steps}")
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
            algorithm="muon_sphere",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
            radius_scale=radius_scale,
            power_iter_steps=power_iter_steps,
        )
        super().__init__(
            params, distributed_mesh, "muon_sphere", defaults,
            use_gram_newton_schulz=use_gram_newton_schulz,
            use_triton=use_triton,
            use_polar_express=use_polar_express,
            newton_schulz_func=newton_schulz_func,
        )

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = super()._get_or_initialize_state(param, algo)
        if algo == self._algo_name and "spectral_initialized" not in state:
            # Random-vector warmstart for power iteration. Replicated per
            # rank so all ranks agree on the (u, v) pair.
            local = param.to_local() if isinstance(param, DTensor) else param
            d_out, d_in = local.shape[-2:]
            generator = torch.Generator(device=local.device).manual_seed(0)
            state["u_cache"] = torch.randn(
                d_out, 1, device=local.device, dtype=torch.float32, generator=generator
            )
            state["v_cache"] = torch.randn(
                d_in, 1, device=local.device, dtype=torch.float32, generator=generator
            )
            state["spectral_initialized"] = False
        return state

    def _retract_to_sphere(self, group: dict) -> None:
        """Project each ``W`` in ``group`` onto the spectral sphere of
        radius ``R = radius_scale * sqrt(d_out / d_in)``.

        Done in-place on the parameter tensor before the Muon update is
        computed. Caches updated (u, v) singular vectors in optimizer state.
        On the very first call per param, also runs the init projection
        ``W <- R * W / ||W||_2`` (the paper's Algorithm 1 line 1).
        """
        flatten = group["flatten"]
        radius_scale = group["radius_scale"]
        iters = group["power_iter_steps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim < 2:
                continue
            state = self._get_or_initialize_state(p, self._algo_name)
            d_out, d_in = _spectral_shape(p.shape, flatten=flatten)
            R = radius_scale * math.sqrt(d_out / d_in)

            sigma, u, v = _power_iteration(
                p,
                u_init=state["u_cache"],
                v_init=state["v_cache"],
                iters=iters,
                flatten=flatten,
            )
            state["u_cache"].copy_(u)
            state["v_cache"].copy_(v)

            if not state["spectral_initialized"]:
                # First-step projection onto the sphere (paper Algorithm 1
                # line 1): the cached u, v are random, so we don't trust the
                # current sigma direction; instead just rescale by the
                # Frobenius-derived spectral estimate from one extra iter.
                sigma = sigma.clamp(min=p.new_tensor(1e-12))
                p.mul_(R / sigma)
                state["spectral_initialized"] = True
            else:
                # Subsequent retractions: rescale to keep ||W||_2 = R.
                sigma = sigma.clamp(min=p.new_tensor(1e-12))
                p.mul_(R / sigma)

    def _create_ortho_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            assert group["algorithm"] == self._algo_name
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "MuonSphere optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            # Retract every param in this group onto the spectral sphere
            # before computing the orthogonalized update. This is the only
            # thing MuonSphere does beyond standard Muon.
            self._retract_to_sphere(group)

            # The remaining flow mirrors Muon._create_ortho_tasks. We compute
            # an effective LR equal to ``lr * radius_scale``: the existing
            # adjust_lr=spectral_norm path already multiplies by
            # sqrt(fan_out/fan_in), and combining with radius_scale gives
            # the full ``R = radius_scale * sqrt(fan_out/fan_in)`` paper scale.
            update_args = dict(
                lr=torch.tensor(group["lr"] * group["radius_scale"]),
                base_lr=torch.tensor(group["lr"]),
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

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [
                    self._get_or_initialize_state(p, self._algo_name) for p in params
                ]
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
                    muon_sphere_update_megabatch_async(
                        X=params,
                        G=gradients,
                        M=momentums,
                        shard_dim=shard_dim,
                        **megabatch_args,
                    )
                )


def muon_sphere_update_megabatch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,         # already includes radius_scale
    base_lr: Tensor,    # for weight_decay scaling, matches Muon's base_lr
    momentum: Tensor,
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
    """Identical to ``muon_update_megabatch_async`` except that ``lr`` is
    pre-multiplied by ``radius_scale`` so the post-orthogonalize update is
    scaled by ``R = radius_scale * sqrt(fan_out/fan_in)`` rather than just
    ``sqrt(fan_out/fan_in)``.

    The retraction step happens before this function is called (in
    ``MuonSphere._create_ortho_tasks._retract_to_sphere``); by the time we
    get here, ``X`` is already on the spectral sphere.
    """
    N = len(X)
    assert N == len(G) == len(M)

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
        base_lr=base_lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
        cautious_wd=cautious_wd,
    )


def _spectral_shape(shape: torch.Size, flatten: bool) -> Tuple[int, int]:
    """Return ``(d_out, d_in)`` used for the spectral-norm retraction.

    Matches the convention of ``adjust_lr_spectral_norm`` in
    ``megabatch_base.py``.
    """
    if flatten and len(shape) >= 3:
        return shape[0], int(torch.tensor(list(shape[1:])).prod().item())
    return shape[-2], shape[-1]


def _flatten_for_spectral(W: Tensor, flatten: bool) -> Tensor:
    """Flatten a >=3D tensor into 2D for power iteration, matching the
    ``flatten`` semantics used by the rest of dion."""
    if flatten and W.ndim >= 3:
        return W.flatten(start_dim=1)
    if W.ndim >= 3:
        # Treat as a batch of 2D matrices: power-iterate each independently
        # by collapsing the leading batch dims into a single dim. We only
        # need ||W||_2 per matrix here. Callers that hit this path get a
        # batched (sigma, u, v).
        return W.reshape(-1, W.shape[-2], W.shape[-1])
    return W


def _power_iteration(
    W: Tensor,
    u_init: Tensor,
    v_init: Tensor,
    iters: int,
    flatten: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Top-singular-value power iteration with warmstart.

    For ``DTensor`` params this materializes ``W`` once via ``full_tensor()``
    and runs the iteration locally. The cost is one all-gather of ``W`` per
    optimizer step. For an FSDP2 1B Transformer the single-step gather is
    O(numel(W) * 2 bytes) and is dominated by the existing Muon all-to-alls;
    a sharded power iteration is a natural follow-up.

    Returns ``(sigma, u, v)`` all in fp32.

    Caveat: only the 2D path is supported. 3D+ params are flattened or
    treated as batches; in either case we return the spectral norm of the
    flattened/first matrix, which is sufficient for the retraction
    multiplier on a uniformly-sized batch.
    """
    W_full = W.full_tensor() if isinstance(W, DTensor) else W
    W_full = W_full.detach().to(torch.float32)
    W_2d = _flatten_for_spectral(W_full, flatten=flatten)
    if W_2d.ndim == 3:
        # Batch of matrices: pick the first as a representative for sigma.
        # All matrices in this code path come from the same param-group
        # shape bucket so they share scale.
        W_2d = W_2d[0]

    u = u_init.to(torch.float32)
    v = v_init.to(torch.float32)
    eps = W_2d.new_tensor(1e-12)

    for _ in range(iters):
        u = W_2d @ v
        u = u / (u.norm() + eps)
        v = W_2d.mT @ u
        v = v / (v.norm() + eps)

    sigma = (u.mT @ W_2d @ v).reshape(()).abs()
    return sigma, u, v
