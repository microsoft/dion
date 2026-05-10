"""Reference Aurora implementation.

Single-file readable port of https://github.com/tilde-research/aurora-release.
Mirrors ``src/aurora.py`` (preconditioned polar) and ``src/polar.py``
(simple-quintic Newton-Schulz) byte-for-byte in their math, wrapped in a
PyTorch ``Optimizer`` that follows the same param-group conventions as
``muon_reference.Muon``.

This module is for clarity and reproducibility. For training, prefer
``dion.Aurora``, which integrates with FSDP2 / DDP and uses
``polar_express`` (faster than the simple-quintic polar in this file).

Aurora: https://blog.tilderesearch.com/blog/aurora
Reference: https://github.com/tilde-research/aurora-release
"""

import math
import torch
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Optional, Tuple


@torch.no_grad()
def polar(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Polar factor via 12-step simple-quintic Newton-Schulz.

    p(sigma) = 2*sigma - 1.5*sigma^3 + 0.5*sigma^5; sigma=1 is super-attracting.
    Matches ``src/polar.py`` from the Aurora reference repo.
    """
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.no_grad()
def aurora_polar(
    G: torch.Tensor,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Aurora's leverage-uniform polar via diagonal preconditioning.

    For square ``G`` reduces to ``polar(G)``. For non-square ``G`` transposes
    to tall, runs ``pp_iterations`` rounds of polar with row-norm
    preconditioning, then applies the ``max(1, m/n)^0.5`` aspect-ratio
    scaling. Matches ``src/aurora.py`` from the Aurora reference repo.
    """
    m, n = G.size(-2), G.size(-1)
    if m == n:
        update = polar(G, eps=eps)
    else:
        transposed = m < n
        if transposed:
            G = G.mT
            m, n = n, m
        G32 = G.to(torch.float32)
        target_row_sq = n / m
        row_norm = G32.norm(dim=-1, keepdim=True).clamp_(min=eps)
        D = 1.0 / row_norm
        for k in range(pp_iterations):
            U = polar(D * G32, eps=eps)
            if k < pp_iterations - 1:
                row_sq = (
                    U.to(torch.float32)
                    .pow(2)
                    .sum(dim=-1, keepdim=True)
                    .clamp_(min=eps * eps)
                )
                D = D * (target_row_sq / row_sq).pow(pp_beta)
        update = U.mT if transposed else U
    update = update * (max(1.0, m / n) ** 0.5)
    return update


class Aurora(Optimizer):
    """Reference Aurora optimizer (single-file, no FSDP/DDP integration).

    Mirrors the param-group style of ``dion.muon_reference.Muon``: the
    ``algorithm`` key on a param group selects ``aurora`` (default for
    matrix params), ``adamw``, or ``lion``.

    For distributed training, use ``dion.Aurora`` instead.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.05,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.95, 0.95),
        weight_decay: float = 0.025,
        epsilon: float = 1e-7,
        nesterov: bool = True,
        pp_iterations: int = 2,
        pp_beta: float = 0.5,
    ):
        defaults = dict(
            lr=lr,
            momentum=mu,
            betas=betas,
            weight_decay=weight_decay,
            epsilon=epsilon,
            nesterov=nesterov,
            pp_iterations=pp_iterations,
            pp_beta=pp_beta,
        )
        super().__init__(params, defaults)

        if isinstance(params, dict):
            params = [params]

        for param_or_param_group in params:
            if isinstance(param_or_param_group, dict):
                algo = param_or_param_group.get("algorithm", "aurora")
                if algo not in ("aurora", "adamw", "lion"):
                    raise ValueError(f"Unknown algorithm: {algo}")
                for p in param_or_param_group["params"]:
                    self.state[p]["algorithm"] = algo
                    if algo == "aurora" and p.ndim != 2:
                        raise ValueError(
                            f"Aurora requires 2D parameters, but got {p.ndim}D"
                        )
            else:
                p = (
                    param_or_param_group[1]
                    if isinstance(param_or_param_group, tuple)
                    and len(param_or_param_group) == 2
                    else param_or_param_group
                )
                if not isinstance(p, torch.Tensor):
                    raise ValueError(
                        f"Invalid parameter type: {type(param_or_param_group)}"
                    )
                self.state[p]["algorithm"] = "aurora"
                if p.ndim != 2:
                    raise ValueError(
                        f"Aurora requires 2D parameters, but got {p.ndim}D"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["epsilon"]
            pp_iterations = group["pp_iterations"]
            pp_beta = group["pp_beta"]

            aurora_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "aurora"
            ]
            for p in aurora_params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                # SGD momentum (Nesterov by default), matching the reference
                # ``aurora()`` function's lerp-style EMA momentum.
                buf.lerp_(g, 1 - momentum)
                u = g.lerp(buf, momentum) if group["nesterov"] else buf.clone()

                if isinstance(u, DTensor):
                    u_local = u.full_tensor()
                    u_local = aurora_polar(
                        u_local, pp_iterations=pp_iterations, pp_beta=pp_beta, eps=eps
                    )
                    u = DTensor.from_local(
                        u_local,
                        device_mesh=u.device_mesh,
                        placements=None,
                        run_check=False,
                    ).redistribute(placements=p.placements)
                else:
                    u = aurora_polar(
                        u, pp_iterations=pp_iterations, pp_beta=pp_beta, eps=eps
                    )

                p.mul_(1 - lr * weight_decay)
                p.add_(u, alpha=-lr)

            adamw_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "adamw"
            ]
            beta1, beta2 = group["betas"]
            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.mul_(1 - lr * weight_decay)
                p.add_(g, alpha=-lr / scale)

            lion_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "lion"
            ]
            for p in lion_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                update = buf.lerp(g, 1 - beta1).sign_()
                buf.lerp_(g, 1 - beta2)
                p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss
