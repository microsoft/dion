import torch
from torch import Tensor
from typing import Generator, List


@torch.compile(fullgraph=True)
def adamw_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    V: Tensor,  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
):
    """
    AdamW optimizer algorithm.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    M.lerp_(G.to(M.dtype), 1 - beta1)
    # V = beta2 * V + (1 - beta2) * G * G
    V.mul_(beta2).addcmul_(G, G, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    if cautious_wd:
        # Compute update direction (pre-LR) for CWD mask
        update_dir = M / denom

        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (update_dir * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)


@torch.compile(fullgraph=True)
def lion_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm. Sign update should guarantee RMS norm equal to 1.
    """
    assert X.shape == G.shape
    assert X.shape == M.shape

    G = G.to(M.dtype)

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = M.lerp(G, 1 - beta1).sign_()

    # Update momentum with new gradient
    # M = beta2 * M + (1 - beta2) * G
    M.lerp_(G, 1 - beta2)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay
        decay_mask = (U * X >= 0).to(dtype=X.dtype)
        decay = (X * decay_mask) * coeff
        X.sub_(decay)
    else:
        # Apply weight decay
        X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    X.add_(U, alpha=-lr)


_step_tensor_cache: dict = {}


def _get_step_tensor(device: torch.device) -> Tensor:
    t = _step_tensor_cache.get(device)
    if t is None:
        t = torch.zeros((), dtype=torch.float32, device=device)
        _step_tensor_cache[device] = t
    return t


def adamw_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor or float)
    beta1: Tensor,  # Beta 1 (scalar tensor or float)
    beta2: Tensor,  # Beta 2 (scalar tensor or float)
    weight_decay: Tensor,  # Weight decay (scalar tensor or float)
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
):
    """AdamW update for a list of tensors.

    Dispatches through ``torch._fused_adamw_``, which is already a
    multi-tensor-apply kernel; avoids the ~6-call ``torch._foreach_*`` chain
    that otherwise dispatches one aten op per tensor on the CPU side.

    Cautious weight decay (https://arxiv.org/pdf/2510.12402) is applied as a
    post-step correction rather than inside the kernel:

        X_std   = X - lr*wd*X - update                    (standard AdamW)
        X_cwd   = X - lr*wd*X*mask - update               (CWD)
                = X_std + lr*wd*X*(1 - mask)

    where ``mask = (sign(M_new · X_orig) >= 0)``. We add back the decay that
    was over-applied on elements where momentum and param disagree in sign.
    """
    if not X:
        return
    n = len(X)
    assert n == len(G) == len(M) == len(V)

    lr_f = float(lr)
    beta1_f = float(beta1)
    beta2_f = float(beta2)
    wd_f = float(weight_decay)
    eps_f = float(epsilon)

    do_cwd_correction = cautious_wd and wd_f > 0.0
    if do_cwd_correction:
        X_orig = [x.clone() for x in X]

    # Cache the step scalar per device. ``torch.tensor(x, device="cuda")``
    # from a Python float stages through pageable CPU memory and issues a
    # blocking ``cudaMemcpy``, which defeats the point of going fused.
    # ``fill_`` on a cached 0-d CUDA tensor is a kernel launch — async.
    step_t = _get_step_tensor(X[0].device)
    step_t.fill_(float(step))
    torch._fused_adamw_(
        X, G, M, V, [],
        [step_t] * n,
        amsgrad=False,
        beta1=beta1_f, beta2=beta2_f,
        lr=lr_f, weight_decay=wd_f, eps=eps_f,
        maximize=False,
    )

    if do_cwd_correction:
        # mask == 0  <=>  sign(M_new) * sign(X_orig) < 0  (over-decayed).
        signs = torch._foreach_mul(M, X_orig)
        undo_masks = [(s < 0).to(x.dtype) for s, x in zip(signs, X_orig)]
        correction = torch._foreach_mul(X_orig, undo_masks)
        torch._foreach_mul_(correction, lr_f * wd_f)
        torch._foreach_add_(X, correction)


@torch.compile(fullgraph=True)
def lion_update_foreach(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    cautious_wd: bool = False,
):
    """
    Lion optimizer algorithm (foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)

    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Compute sign update
    # U = sign(beta1 * M + (1 - beta1) * G)
    U = torch._foreach_lerp(M, G, [1 - beta1] * batch_size)
    torch._foreach_sign_(U)

    # Update momentum in place with new gradient
    # M = beta2 * M + (1 - beta2) * G
    torch._foreach_lerp_(M, G, [1 - beta2] * batch_size)

    if cautious_wd:
        # Apply cautious weight decay: only where update and parameter signs align
        # Reference: https://arxiv.org/pdf/2510.12402
        coeff = lr * weight_decay

        decay_masks = torch._foreach_mul(X, U)
        decay_masks = torch._foreach_sign(decay_masks)  # {-1, 0, 1}
        decay_masks = torch._foreach_add(decay_masks, 1)  # {0, 1, 2}
        decay_masks = torch._foreach_minimum(decay_masks, 1)  # {0, 1, 1}

        decay_terms = torch._foreach_mul(X, decay_masks)
        torch._foreach_mul_(decay_terms, coeff)
        torch._foreach_sub_(X, decay_terms)
    else:
        # Apply weight decay
        torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - lr * U
    torch._foreach_mul_(U, lr)
    torch._foreach_sub_(X, U)


def adamw_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    step: int,
    epsilon: float,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    adamw_update_foreach(
        X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon, cautious_wd
    )
    yield


def lion_update_foreach_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    weight_decay: Tensor,
    cautious_wd: bool = False,
) -> Generator[None, None, None]:
    lion_update_foreach(X, G, M, lr, beta1, beta2, weight_decay, cautious_wd)
    yield
