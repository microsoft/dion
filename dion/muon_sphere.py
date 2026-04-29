import math
import torch
from collections import defaultdict
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import ParamsT
from typing import Callable, Generator, List, Optional, Tuple, Union

from .megabatch_base import DistributedOrthoBase
from .muon import muon_update_megabatch_async
from .opt_utils import AsyncTask


class MuonSphere(DistributedOrthoBase):
    """
    MuonSphere — Muon with a spectral-sphere weight retraction.

    Constrains every matrix parameter ``W`` to lie on the spectral sphere
    ``{ W : ||W||_2 = R }``, where ``R = radius_scale * sqrt(d_out / d_in)`` is
    the Spectral muP radius (Yang, Simon, Bernstein 2024). The update direction
    is the same Muon msign(M); the only changes are:

    1. A per-step retraction ``W <- W * R / sigma`` (where ``sigma`` is the
       top singular value, estimated by warmstarted power iteration) before
       the standard Muon update is applied. The first call doubles as the
       init-time projection onto the sphere.
    2. A learning-rate scale of ``R = radius_scale * sqrt(d_out/d_in)`` on
       the orthogonalized update (i.e. the existing ``adjust_lr=spectral_norm``
       scale, multiplied by ``radius_scale``).
    3. Weight decay defaults to ``0`` (the retraction subsumes its role on
       hidden 2D weights). Set ``weight_decay > 0`` only on params where you
       want it. Weight decay is scaled by the unscaled ``lr``, not by
       ``lr * radius_scale``.

    This is the lambda=0 ablation of the Spectral Sphere Optimizer (SSO) from
    Xie et al., "Controlled LLM Training on Spectral Sphere", arXiv:2601.08393.
    The paper's full SSO additionally solves a Lagrange tangent-space
    projection (h(lambda)=0 by bisection); MuonSphere drops that step in
    exchange for ~1% added latency over Muon (vs. ~11% for full SSO) while
    capturing most of the downstream-quality gain.

    Supported parameter shapes:
    - 2D matrices (the standard LLM linear-layer case).
    - 3D+ tensors with ``flatten=True`` (e.g. conv weights), retracted as
      a single 2D matrix of shape ``(d_out, prod(d_in...))``.

    Not yet supported (raise ``NotImplementedError``):
    - 3D+ tensors with ``flatten=False`` (per-matrix retraction in a batch
      requires batched power iteration with replicated DTensor multipliers).
    - ``num_heads > 1`` (per-head retraction needs sharding-aware power
      iteration on the per-head view).
    Both are natural follow-ups; the paper itself targets 2D LLM weights.

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

    def _retract_to_sphere(self, group: dict) -> None:
        """Project each ``W`` in ``group`` onto the spectral sphere of
        radius ``R = radius_scale * sqrt(d_out / d_in)``.

        Done in-place on the parameter tensor before the Muon update is
        computed. Caches updated (u, v) singular vectors in optimizer state.
        On the very first call per param, the same rescale serves as the
        init-time projection (paper Algorithm 1 line 1).
        """
        flatten = group["flatten"]
        radius_scale = group["radius_scale"]
        iters = group["power_iter_steps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim < 2:
                continue
            if not flatten and p.ndim > 2:
                raise NotImplementedError(
                    f"MuonSphere does not yet support {p.ndim}D parameters with "
                    f"flatten=False (got shape {tuple(p.shape)}). Per-matrix "
                    f"retraction in a batch requires batched power iteration "
                    f"with replicated DTensor multipliers; pass flatten=True "
                    f"to retract as a single 2D view, or split the batch into "
                    f"separate 2D parameters."
                )
            state = self._get_or_initialize_state(p, self._algo_name)
            d_out, d_in = _spectral_shape(p.shape, flatten=flatten)
            if "u_cache" not in state:
                # Random-vector warmstart for power iteration. The cache holds
                # the GLOBAL (un-sharded) u, v vectors: each rank materializes
                # the full ``W`` via ``DTensor.full_tensor()`` inside
                # ``_power_iteration`` and runs the iteration locally, so the
                # cache dimensions must match the global, post-flatten matrix
                # shape.
                local = p.to_local() if isinstance(p, DTensor) else p
                generator = torch.Generator(device=local.device).manual_seed(0)
                state["u_cache"] = torch.randn(
                    d_out, 1, device=local.device, dtype=torch.float32, generator=generator
                )
                state["v_cache"] = torch.randn(
                    d_in, 1, device=local.device, dtype=torch.float32, generator=generator
                )
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
            p.mul_(R / sigma.clamp(min=1e-12))

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

            num_heads = self._resolve_num_heads(group)
            if num_heads is not None:
                raise NotImplementedError(
                    "MuonSphere does not yet support num_heads > 1: per-head "
                    "retraction needs sharding-aware power iteration on the "
                    "per-head view. Use Muon for num_heads splits or drop the "
                    "num_heads option."
                )

            # Retract every param in this group onto the spectral sphere
            # before computing the orthogonalized update. This is the only
            # thing MuonSphere does beyond standard Muon.
            self._retract_to_sphere(group)

            # We thread two LRs into the shared muon helper: ``lr`` carries
            # the radius_scale (so the post-orthogonalize update is scaled by
            # ``R = radius_scale * sqrt(fan_out/fan_in)`` after adjust_lr),
            # and ``base_lr`` is the unscaled lr used for weight decay.
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

            for (_shape, _sharding, _dtype), params in shape_groups.items():
                gradients = [p.grad for p in params]
                states = [
                    self._get_or_initialize_state(p, self._algo_name) for p in params
                ]
                momentums = [s["momentum"] for s in states]

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
                        **megabatch_args,
                    )
                )


def _spectral_shape(shape: torch.Size, flatten: bool) -> Tuple[int, int]:
    """Return ``(d_out, d_in)`` used for the spectral-norm retraction.

    Matches the convention of ``adjust_lr_spectral_norm`` in
    ``megabatch_base.py``.
    """
    if flatten and len(shape) >= 3:
        return shape[0], math.prod(shape[1:])
    return shape[-2], shape[-1]


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

    Returns ``(sigma, u, v)`` all in fp32. ``sigma`` is a 0-d tensor;
    ``u`` is ``(d_out, 1)`` and ``v`` is ``(d_in, 1)``, matching
    ``_spectral_shape(W.shape, flatten)``.

    Only the 2D path (post-flatten) is supported; ``_retract_to_sphere``
    rejects 3D+ params with ``flatten=False`` upstream.
    """
    W_full = W.full_tensor() if isinstance(W, DTensor) else W
    W_full = W_full.detach().to(torch.float32)
    if flatten and W_full.ndim >= 3:
        W_2d = W_full.flatten(start_dim=1)
    else:
        W_2d = W_full
    assert W_2d.ndim == 2, (
        f"_power_iteration expects a 2D matrix after flatten, got shape "
        f"{tuple(W_2d.shape)}."
    )

    u = u_init.to(torch.float32)
    v = v_init.to(torch.float32)
    eps = 1e-12

    for _ in range(iters):
        u = W_2d @ v
        u = u / (u.norm() + eps)
        v = W_2d.mT @ u
        v = v / (v.norm() + eps)

    sigma = (u.mT @ W_2d @ v).reshape(()).abs()
    return sigma, u, v
