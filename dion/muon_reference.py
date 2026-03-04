import math
import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer, ParamsT
from typing import Optional, Tuple


#@torch.compile()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to approximate the orthogonalization of G.
    """
    assert len(G.shape) == 2, "Expected 2D tensor"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


# Muon version based on code from Moonlight
# https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Modified from original MoonshotAI implementation to take similar param groups as distributed Muon.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.95, 0.95),
        weight_decay: float = 0.1,
        epsilon: float = 1e-8,
        nesterov: bool = True,
        adjust_lr: Optional[str] = "spectral_norm",
        ns_steps: int = 5,
    ):
        if adjust_lr not in ("spectral_norm", "rms_norm", None):
            raise ValueError(
                f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or None."
            )

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=mu,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr=adjust_lr,
            betas=betas,
            epsilon=epsilon,
        )
        super().__init__(params, defaults)

        if isinstance(params, dict):
            params = [params]

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for param_or_param_group in params:
            # Input is a list of param_group dicts
            if isinstance(param_or_param_group, dict):
                algo = param_or_param_group.get("algorithm", "muon")
                if algo not in ("muon", "adamw", "lion"):
                    raise ValueError(f"Unknown algorithm: {algo}")

                for p in param_or_param_group["params"]:
                    self.state[p]["algorithm"] = algo
                    if algo == "muon" and p.ndim != 2:
                        raise ValueError(
                            f"Muon requires 2D parameters, but got {p.ndim}D"
                        )

            # Input is a list of parameter tensors
            # Assume Muon by default, because we require scalar params be specified explicitly
            else:
                if isinstance(param_or_param_group, torch.Tensor):
                    p = param_or_param_group
                elif (
                    isinstance(param_or_param_group, tuple)
                    and len(param_or_param_group) == 2
                ):
                    # (name, param) tuple
                    p = param_or_param_group[1]
                else:
                    raise ValueError(
                        f"Invalid parameter type: {type(param_or_param_group)}"
                    )
                self.state[p]["algorithm"] = "muon"
                if p.ndim != 2:
                    raise ValueError(f"Muon requires 2D parameters, but got {p.ndim}D")

    def adjust_lr_to_match_adam(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def adjust_lr_spectral_norm(self, lr, param_shape):
        # Adjust from spectral norm 1 to RMS operator norm 1
        # https://arxiv.org/abs/2310.17813
        fan_out, fan_in = param_shape[:2]
        adjusted_lr = lr * math.sqrt(fan_out / fan_in)
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            muon_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "muon"
            ]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["epsilon"]

            # generate weight updates in distributed fashion
            for p in muon_params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                if isinstance(g, DTensor):
                    # all-gather into full unsharded matrix
                    g_local = g.full_tensor()

                    # calculate muon update
                    u_local = zeropower_via_newtonschulz5(
                        g_local, steps=group["ns_steps"], eps=eps
                    )

                    # convert back to DTensor and re-shard to original placements
                    u = DTensor.from_local(
                        u_local,
                        device_mesh=g.device_mesh,
                        placements=None,  # fully replicated
                        run_check=False,
                    ).redistribute(placements=g.placements)

                else:
                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"], eps=eps)

                # scale update
                if group["adjust_lr"] is None:
                    adjusted_lr = lr
                elif group["adjust_lr"] == "spectral_norm":
                    adjusted_lr = self.adjust_lr_spectral_norm(lr, p.shape)
                elif group["adjust_lr"] == "rms_norm":
                    adjusted_lr = self.adjust_lr_to_match_adam(lr, p.shape)
                else:
                    raise ValueError(f"Unknown adjust_lr value: {group['adjust_lr']}")

                # apply weight decay
                p.mul_(1 - lr * weight_decay)

                # apply update
                p.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            adamw_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "adamw"
            ]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["epsilon"]
            weight_decay = group["weight_decay"]

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

            ############################
            #       Lion backup        #
            ############################

            lion_params = [
                p for p in group["params"] if self.state[p]["algorithm"] == "lion"
            ]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

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


# Muon version based on Keller Jordan repo
# https://github.com/KellerJordan/modded-nanogpt
class MuonKellerJordan(Optimizer):
    """
    Muon optimizer - runs standard SGD with momentum and then orthogonalizes each 2D update.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

        # Make sure no parameters are DTensor
        for group in self.param_groups:
            for p in group["params"]:
                if isinstance(p, DTensor):
                    raise NotImplementedError("DTensor parameters not supported.")

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            total_params = sum(p.numel() for p in group["params"])
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(group["params"]):
                if i % int(os.environ["WORLD_SIZE"]) == int(os.environ["RANK"]):
                    g = p.grad
                    assert g is not None, "Gradient is None"
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g *= (g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr_idx = 0
            for p in group["params"]:
                g = updates_flat[curr_idx : curr_idx + p.numel()].view_as(p).type_as(p)
                p.add_(g, alpha=-lr)
                curr_idx += p.numel()
