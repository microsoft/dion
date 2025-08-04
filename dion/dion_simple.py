import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from torch.distributed.tensor import DTensor
from typing import Any, Dict, Tuple

from .scalar_opts import adamw_update, lion_update


@torch.compile()
def dion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    Q: Tensor,  # Q matrix for power iteration (modified in place)
    lr: float,  # Learning rate
    mu: float,  # Momentum factor
    weight_decay: float,
    epsilon: float = 1e-8,
):
    """
    Dion optimizer algorithm.
    """
    # Add new gradient to momentum
    M.add_(G)

    # Compute low-rank approximation of M = P @ Q^T
    P = M @ Q
    P, _ = torch.linalg.qr(P)
    R = M.T @ P

    # Error feedback
    # M = M - (1 - mu) * (P @ R.T)
    M.addmm_(P, R.T, alpha=-(1 - mu))

    # Column normalize R to get new Q
    Q_new = R / (R.norm(dim=0, keepdim=True) + epsilon)
    Q.copy_(Q_new)

    # Compute update scale factor based on matrix shape
    fan_out = X.size(0)
    fan_in = X.size(1)
    scaled_lr = lr * ((fan_out / fan_in) ** 0.5)

    # Apply weight decay
    X.mul_(1 - weight_decay * lr)

    # Apply the weight update
    # X = X - scaled_lr * (P @ Q.T)
    X.addmm_(P, Q.T, alpha=-scaled_lr)


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
        # Initialize momentum and Q for a 2D matrix parameter
        state["momentum"] = torch.zeros_like(param)
        r = min(self.rank, min(param.shape))
        state["Q"] = torch.randn(
            (param.size(1), r), device=param.device, dtype=param.dtype
        )

    def _init_opt_state_adam(self, param: Tensor, state: Dict[str, Any]):
        state["momentum"] = torch.zeros_like(param)
        state["variance"] = torch.zeros_like(param)

    def _init_opt_state_lion(self, param: Tensor, state: Dict[str, Any]):
        state["momentum"] = torch.zeros_like(param)
