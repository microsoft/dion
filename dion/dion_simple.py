import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from torch.distributed.tensor import DTensor
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass

from .scalar_opts import adamw_update, lion_update


@dataclass
class DionMixedPrecisionConfig:
    momentum_dtype: Optional[torch.dtype] = None  # None => use param.dtype
    Q_dtype: Optional[torch.dtype] = None  # None => use param.dtype


@torch.compile()
def dion_update(
    X: Tensor,
    G: Tensor,
    M: Tensor,
    Q: Tensor,
    lr: float,
    mu: float,
    weight_decay: float,
    epsilon: float = 1e-8,
):
    """
    Dion optimizer algorithm.
    """
    # Add new gradient to momentum
    M.add_(G.to(M.dtype))

    # Compute low-rank approximation of M = P @ Q^T
    P = M @ Q.to(M.dtype)
    # orthonormalize in fp32 then cast back to M.dtype
    P32, _ = torch.linalg.qr(P.to(torch.float32))
    P = P32.to(M.dtype)

    # 2) R = M^T @ P  (state dtype)
    R = M.T @ P

    # Error feedback
    # M = M - (1 - mu) * (P @ R.T)
    M.addmm_(P, R.T, alpha=-(1 - mu))

    # Column normalize R to get new Q
    R32 = R.to(torch.float32)
    den = R32.norm(dim=0, keepdim=True) + float(epsilon)
    Q_new = (R32 / den).to(Q.dtype)
    Q.copy_(Q_new)

    # Compute update scale factor based on matrix shape
    fan_out, fan_in = X.size(0), X.size(1)
    scaled_lr = lr * ((fan_out / fan_in) ** 0.5)

    # Apply weight decay
    X.mul_(1 - weight_decay * lr)

    # Apply the weight update
    # X = X - scaled_lr * (P @ Q.T)
    Pd = P.to(X.dtype)
    QdT = Q.T.to(X.dtype)
    X.addmm_(Pd, QdT, alpha=-scaled_lr)


class Dion(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float,
        mu: float = 0.95,  # For Dion
        betas: Tuple[float, float] = (0.9, 0.95),  # For AdamW and Lion
        weight_decay: float = 0.01,
        rank: int = 768,
        epsilon: float = 1e-8,
        mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rank < 1:
            raise ValueError(f"Invalid rank value: {rank}")

        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="dion",
            step=0,
        )
        super().__init__(params, defaults)

        self.rank = rank
        self.epsilon = torch.tensor(epsilon)
        self._mixed_precision_config = (
            mixed_precision_config or DionMixedPrecisionConfig()
        )

        # Check that all Dion parameters are 2D tensors
        for group in self.param_groups:
            if group["algorithm"] == "dion":
                for p in group["params"]:
                    if p.dim() != 2:
                        raise ValueError(
                            f"Expected Dion parameters to be 2D tensor, but got {p.dim()}D."
                        )
                    if isinstance(p, DTensor):
                        raise NotImplementedError(
                            "This version of Dion optimizer does not support distributed tensors."
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            algo = group["algorithm"]
            group["step"] += 1
            step = group["step"]

            # Wrap hyperparameters in tensors for torch.compile
            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])

            if algo == "dion":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    # Get optimizer state for this parameter
                    state = self.state[param]
                    if not state:
                        self._init_opt_state_dion(param, state)

                    # Apply update
                    dion_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        Q=state["Q"],
                        lr=lr,
                        mu=mu,
                        weight_decay=weight_decay,
                        epsilon=self.epsilon,
                    )

            elif algo == "adamw":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    # Get optimizer state for this parameter
                    state = self.state[param]
                    if not state:
                        self._init_opt_state_adam(param, state)

                    # Apply update
                    adamw_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        V=state["variance"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,
                        epsilon=self.epsilon,
                    )

            elif algo == "lion":
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    # Get optimizer state for this parameter
                    state = self.state[param]
                    if not state:
                        self._init_opt_state_lion(param, state)

                    # Apply update
                    lion_update(
                        X=param,
                        G=param.grad,
                        M=state["momentum"],
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                    )

            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        return loss

    def _init_opt_state_dion(self, param: Tensor, state: Dict[str, Any]):
        # momentum in configured dtype
        mom_dtype = self._mixed_precision_config.momentum_dtype or param.dtype
        state["momentum"] = torch.zeros_like(param, dtype=mom_dtype)

        # Q in configured dtype
        q_dtype = self._mixed_precision_config.Q_dtype or param.dtype
        r = min(self.rank, min(param.shape))
        state["Q"] = torch.randn((param.size(1), r), device=param.device, dtype=q_dtype)

    def _init_opt_state_adam(self, param: Tensor, state: Dict[str, Any]):
        state["momentum"] = torch.zeros_like(param)
        state["variance"] = torch.zeros_like(param)

    def _init_opt_state_lion(self, param: Tensor, state: Dict[str, Any]):
        state["momentum"] = torch.zeros_like(param)
