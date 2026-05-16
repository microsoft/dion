import torch
from torch import Tensor
from typing import List

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    import types

    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.autotune = lambda **kw: lambda fn: fn
    triton.Config = dict
    triton.heuristics = lambda _: lambda fn: fn
    triton.cdiv = lambda a, b: (a + b - 1) // b
    tl = types.ModuleType("triton.language")
    tl.constexpr = int


# We cannot use @triton.autotune here because the kernel is in-place on X:
# autotune benchmark runs would corrupt X by applying the update multiple times.
# Instead we use heuristics to select block sizes based on the selection mode.


def _build_index_map(indices: Tensor, full_dim: int) -> Tensor:
    """Build a dense index map from sparse indices.

    Args:
        indices: (*leading, k) int64 tensor of selected row/col indices.
        full_dim: size of the full dimension (M for rows, N for cols).

    Returns:
        (*leading, full_dim) int32 tensor where map[..., i] = j if index i
        is the j-th selected entry, else -1.
    """
    shape = indices.shape[:-1] + (full_dim,)
    index_map = torch.full(shape, -1, dtype=torch.int32, device=indices.device)
    k = indices.shape[-1]
    values = torch.arange(k, dtype=torch.int32, device=indices.device)
    values = values.expand_as(indices)
    index_map.scatter_(-1, indices, values)
    return index_map


@triton.heuristics(
    {
        "BLOCK_M": lambda args: 1 if args["SELECT_ROWS"] else 64,
        "BLOCK_N": lambda args: 256 if args["SELECT_ROWS"] else 64,
    }
)
@triton.jit
def _dion2_post_ortho_kernel(
    X_ptr,
    U_ptr,
    map_ptr,
    a,
    b,
    M,
    N,
    x_stride_b,
    x_stride_m,
    x_stride_n,
    u_stride_b,
    u_stride_m,
    u_stride_n,
    map_stride_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SELECT_ROWS: tl.constexpr,
):
    """Fused weight-decay + selective update kernel.

    For each element x[b, m, n]:
      - If row m (SELECT_ROWS) or col n (!SELECT_ROWS) is selected with
        position j in U: x = a*x - b*u[b, j, n] (or u[b, m, j])
      - Otherwise: x = a*x

    The masked load returns 0.0 for unselected entries, so the single
    expression ``a*x - b*u`` handles both cases with one FP rounding.
    """
    pid = tl.program_id(0)

    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    blocks_per_matrix = num_blocks_m * num_blocks_n

    batch_idx = pid // blocks_per_matrix
    local_pid = pid % blocks_per_matrix
    block_m = local_pid // num_blocks_n
    block_n = local_pid % num_blocks_n

    X_ptr += batch_idx * x_stride_b
    U_ptr += batch_idx * u_stride_b
    map_ptr += batch_idx * map_stride_b

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = X_ptr + offs_m[:, None] * x_stride_m + offs_n[None, :] * x_stride_n
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    if SELECT_ROWS:
        # index_map: (B, M) — position in U's k dimension, or -1
        map_vals = tl.load(map_ptr + offs_m, mask=mask_m, other=-1)
        safe_map = tl.where(map_vals >= 0, map_vals, 0)

        # U: (B, k, N)
        u_ptrs = (
            U_ptr + safe_map[:, None] * u_stride_m + offs_n[None, :] * u_stride_n
        )
        u_mask = (map_vals[:, None] >= 0) & mask_n[None, :]
        u = tl.load(u_ptrs, mask=u_mask, other=0.0).to(tl.float32)
    else:
        # index_map: (B, N) — position in U's k dimension, or -1
        map_vals = tl.load(map_ptr + offs_n, mask=mask_n, other=-1)
        safe_map = tl.where(map_vals >= 0, map_vals, 0)

        # U: (B, M, k)
        u_ptrs = (
            U_ptr + offs_m[:, None] * u_stride_m + safe_map[None, :] * u_stride_n
        )
        u_mask = mask_m[:, None] & (map_vals[None, :] >= 0)
        u = tl.load(u_ptrs, mask=u_mask, other=0.0).to(tl.float32)

    # Fused: a*x - b*u. Unselected entries have u=0, so result = a*x.
    result = a * x - b * u

    tl.store(x_ptrs, result, mask=mask)


def dion2_post_orthogonalize_triton(
    X: List[Tensor],
    U: List[Tensor],
    indices: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    select_dim: int,
):
    """Triton-fused version of dion2_post_orthogonalize.

    Applies weight decay and selective update in a single pass:
      selected:   x = (1 - base_lr*wd)*x - adjusted_lr*u
      unselected: x = (1 - base_lr*wd)*x

    Args:
        X: list of parameter tensors (*leading, M, N)
        U: list of update tensors (any dtype; upcast to float32 in-kernel)
        indices: list of index tensors (*leading, k), dtype=int64
        base_lr, adjusted_lr, weight_decay: scalar tensors
        select_dim: -2 (row selection) or -1 (column selection)
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for dion2_post_orthogonalize_triton")

    if select_dim not in (-2, -1):
        raise ValueError(f"select_dim must be -2 or -1, got {select_dim}")

    a = (1 - base_lr * weight_decay).item()
    b = adjusted_lr.item()
    SELECT_ROWS = select_dim == -2

    for x, u, idx in zip(X, U, indices):
        if not x.is_contiguous():
            raise ValueError("dion2_post_orthogonalize_triton requires contiguous X tensors")
        orig_shape = x.shape
        M, N = orig_shape[-2], orig_shape[-1]
        B = x.numel() // (M * N)

        x_flat = x.reshape(B, M, N)
        u_flat = u.reshape(B, *u.shape[-2:])
        idx_flat = idx.reshape(B, idx.shape[-1])

        full_dim = M if SELECT_ROWS else N
        index_map = _build_index_map(idx_flat, full_dim)

        grid = lambda meta: (
            B * triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )

        _dion2_post_ortho_kernel[grid](
            x_flat,
            u_flat,
            index_map,
            a,
            b,
            M,
            N,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            u_flat.stride(0),
            u_flat.stride(1),
            u_flat.stride(2),
            index_map.stride(0),
            SELECT_ROWS=SELECT_ROWS,
        )
