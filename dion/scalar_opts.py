import torch
from torch import Tensor
from typing import Generator, List

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


_TORCH_TO_TRITON_DTYPE = (
    {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    if _TRITON_AVAILABLE
    else {}
)


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


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_adamw_mta_kernel(
        ptrs_x, ptrs_g, ptrs_m, ptrs_v,      # int64 [N]: per-tensor data_ptr()s
        numels,                               # int32 [N]: per-tensor numel
        block_to_tensor, block_to_chunk_start,  # int32 [total_blocks]: host-built partition
        lr, beta1, beta2, weight_decay, eps,
        bias_correction1, bias_correction2_sqrt,
        BLOCK_SIZE: tl.constexpr,
        CAUTIOUS_WD: tl.constexpr,
        DTYPE: tl.constexpr,
    ):
        """Multi-tensor-apply AdamW. Mirrors ``at::native::adam_math``
        (``ADAM_MODE::ADAMW``), with an optional CWD branch gating decoupled
        weight decay on ``sign(param * update) >= 0``
        (https://arxiv.org/pdf/2510.12402)."""
        pid = tl.program_id(0)
        t_idx = tl.load(block_to_tensor + pid)
        chunk_start = tl.load(block_to_chunk_start + pid)
        n = tl.load(numels + t_idx)

        x_ptr = tl.cast(tl.load(ptrs_x + t_idx), tl.pointer_type(DTYPE))
        g_ptr = tl.cast(tl.load(ptrs_g + t_idx), tl.pointer_type(DTYPE))
        m_ptr = tl.cast(tl.load(ptrs_m + t_idx), tl.pointer_type(DTYPE))
        v_ptr = tl.cast(tl.load(ptrs_v + t_idx), tl.pointer_type(DTYPE))

        offs = chunk_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n

        # Promote to fp32 for the math (matches C++ ``opmath_t``).
        param = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        exp_avg = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        exp_avg_sq = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
        step_size = lr / bias_correction1
        denom = tl.sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
        update = step_size * exp_avg / denom

        if CAUTIOUS_WD:
            # CWD needs the mask to reference the update direction from the
            # updated moments, so WD is applied after ``update`` is computed.
            cwd_mask = (param * update >= 0.0).to(tl.float32)
            param = param - lr * weight_decay * param * cwd_mask - update
        else:
            param = param * (1.0 - lr * weight_decay) - update

        tl.store(x_ptr + offs, param.to(DTYPE), mask=mask)
        tl.store(m_ptr + offs, exp_avg.to(DTYPE), mask=mask)
        tl.store(v_ptr + offs, exp_avg_sq.to(DTYPE), mask=mask)


def _pinned_int_tensor_to_device(values, dtype: torch.dtype, device: torch.device):
    """Build a CUDA int tensor via a pinned-memory staging buffer so the H2D
    copy is genuinely asynchronous. ``torch.tensor(...).to(non_blocking=True)``
    from pageable memory would fall back to a synchronous copy."""
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(
        device, non_blocking=True
    )


def _build_mta_metadata(numels_list: List[int], block_size: int, device: torch.device):
    """Build per-block partition metadata for ``_fused_adamw_mta_kernel``.
    Returns ``(numels, block_to_tensor, block_to_chunk_start, total_blocks)``
    as int32 CUDA tensors plus a python int."""
    N = len(numels_list)
    numels_cpu = torch.tensor(numels_list, dtype=torch.int32)
    chunks = (numels_cpu + block_size - 1) // block_size
    total_blocks = int(chunks.sum().item())

    block_to_tensor_cpu = torch.repeat_interleave(
        torch.arange(N, dtype=torch.int32), chunks
    )
    prefix = torch.zeros(N, dtype=torch.int32)
    prefix[1:] = torch.cumsum(chunks[:-1], dim=0)
    block_to_chunk_start_cpu = (
        torch.arange(total_blocks, dtype=torch.int32)
        - torch.repeat_interleave(prefix, chunks)
    ) * block_size

    numels = numels_cpu.pin_memory().to(device, non_blocking=True)
    block_to_tensor = block_to_tensor_cpu.pin_memory().to(device, non_blocking=True)
    block_to_chunk_start = block_to_chunk_start_cpu.pin_memory().to(
        device, non_blocking=True
    )
    return numels, block_to_tensor, block_to_chunk_start, total_blocks


# Cache for MTA metadata + pointer tensors. Optimizer state (X/M/V) is
# allocated once and updated in place, so the per-step H2D copies that
# otherwise cause ``cudaStreamSynchronize`` can be skipped entirely. Keyed on
# per-tensor ``data_ptr()`` (stable across the tensor's lifetime, unlike
# ``id()`` -- ``param.data`` / ``DTensor.to_local()`` return a fresh Python
# wrapper each call). Grads are validated separately per step since
# ``zero_grad(set_to_none=True)`` allocates a new grad buffer.
_mta_cache: dict = {}


def _get_mta_buffers(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    block_size: int,
    device: torch.device,
):
    x_ptrs = tuple(t.data_ptr() for t in X)
    m_ptrs = tuple(t.data_ptr() for t in M)
    v_ptrs = tuple(t.data_ptr() for t in V)
    key = (x_ptrs, m_ptrs, v_ptrs, block_size, device)

    entry = _mta_cache.get(key)
    if entry is None:
        numels, block_to_tensor, block_to_chunk_start, total_blocks = (
            _build_mta_metadata([t.numel() for t in X], block_size, device)
        )
        entry = {
            "ptrs_x": _pinned_int_tensor_to_device(
                list(x_ptrs), torch.int64, device),
            "ptrs_m": _pinned_int_tensor_to_device(
                list(m_ptrs), torch.int64, device),
            "ptrs_v": _pinned_int_tensor_to_device(
                list(v_ptrs), torch.int64, device),
            "numels": numels,
            "block_to_tensor": block_to_tensor,
            "block_to_chunk_start": block_to_chunk_start,
            "total_blocks": total_blocks,
            "g_ptrs": None,
            "ptrs_g": None,
        }
        _mta_cache[key] = entry

    g_ptrs = tuple(t.data_ptr() for t in G)
    if entry["g_ptrs"] != g_ptrs:
        entry["ptrs_g"] = _pinned_int_tensor_to_device(
            list(g_ptrs), torch.int64, device
        )
        entry["g_ptrs"] = g_ptrs

    return entry


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
    """AdamW optimizer algorithm, multi-tensor-apply implementation.

    Processes all parameters in the list with a single Triton kernel launch,
    replicating the math of ``torch._fused_adamw_`` (see
    ``ATen/native/cuda/fused_adam_utils.cuh``) with an optional CWD branch.

    Requirements: all tensors in X/G/M/V must share device and dtype
    (``float32`` / ``float16`` / ``bfloat16``) and be contiguous. Matches
    ``torch._fused_adamw_``'s constraints and Dion's ``zeros_like(param)``
    state initialization.
    """
    if not X:
        return
    assert len(G) == len(X) and len(M) == len(X) and len(V) == len(X)

    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "adamw_update_foreach requires Triton. Install triton or use "
            "adamw_update (single-tensor) instead."
        )

    device = X[0].device
    dtype = X[0].dtype

    # Bias corrections are shared across all params in a Dion group (single
    # ``group['step']``), so precompute on CPU and pass as kernel scalars,
    # avoiding per-param scalar tensor staging.
    lr_f = float(lr)
    beta1_f = float(beta1)
    beta2_f = float(beta2)
    weight_decay_f = float(weight_decay)
    # float() unwraps 0-d tensors too; needed because Dion passes step/epsilon
    # as tensors and ``float ** tensor`` would return a tensor (which Triton
    # then treats as a pointer in the kernel signature).
    step_f = float(step)
    bc1 = 1.0 - beta1_f ** step_f
    bc2_sqrt = (1.0 - beta2_f ** step_f) ** 0.5

    BLOCK_SIZE = 1024
    buf = _get_mta_buffers(X, G, M, V, BLOCK_SIZE, device)

    _fused_adamw_mta_kernel[(buf["total_blocks"],)](
        buf["ptrs_x"], buf["ptrs_g"], buf["ptrs_m"], buf["ptrs_v"],
        buf["numels"], buf["block_to_tensor"], buf["block_to_chunk_start"],
        lr_f, beta1_f, beta2_f, weight_decay_f, float(epsilon),
        bc1, bc2_sqrt,
        BLOCK_SIZE=BLOCK_SIZE,
        CAUTIOUS_WD=cautious_wd,
        DTYPE=_TORCH_TO_TRITON_DTYPE[dtype],
    )


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
