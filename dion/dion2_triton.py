import torch
from torch import Tensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
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
    if select_dim not in (-2, -1):
        raise ValueError(f"select_dim must be -2 or -1, got {select_dim}")

    # The kernel writes X in-place through a raw data pointer, which is only valid
    # for tensors backed by a dense buffer in their logical dtype/layout. Traceable
    # wrapper subclasses (e.g. the quantized-weight wrappers used by MXFP8 training,
    # or DTensor) hold their data in wrapped inner tensors and do not expose such a
    # buffer at data_ptr(), so a raw write corrupts memory and triggers an illegal
    # memory access. Fall back to dion2_post_orthogonalize_fused, which routes
    # through __torch_dispatch__ and updates the wrapped weight correctly while
    # preserving the kernel's single-rounding numerics (computes a*x - b*u in
    # float32 for the selected slices and writes once); the plain eager
    # dion2_post_orthogonalize would round the selected slices twice. Plain dense
    # subclasses such as nn.Parameter are not wrapper subclasses and stay on the
    # kernel, so the single-GPU/DDP path (where params are not converted by
    # to_local) keeps the fast path.
    if any(is_traceable_wrapper_subclass(x) for x in X):
        from .dion2 import dion2_post_orthogonalize_fused

        dion2_post_orthogonalize_fused(
            X, U, indices, base_lr, adjusted_lr, weight_decay, select_dim
        )
        return

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for dion2_post_orthogonalize_triton")

    a = (1 - base_lr * weight_decay).item()
    b = adjusted_lr.item()
    SELECT_ROWS = select_dim == -2

    for x, u, idx in zip(X, U, indices):
        # Empty FSDP2 local shard (sharded dim < world_size or uneven chunking):
        # nothing to write back on this rank. The non-triton path no-ops naturally
        # via scatter_add_ over an empty index; here we must skip explicitly because
        # B = x.numel() // (M * N) divides by zero when M or N is 0.
        if x.numel() == 0:
            continue
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


# ---------------------------------------------------------------------------
# Masked posts (selection_scope="global"/"global_capped")
#
# The masked post-orthogonalize has no index list: U arrives full-shard-shaped
# and exactly zero outside the selected rows, so selection membership is
# derived from the nonzero rows of U itself. These kernels fuse the entire
# masked post into one (Dion2) or two (NorDion2) row-programmed passes,
# replacing the compiled path's separate full-matrix sweeps (weight-decay
# foreach, mask reduction, EF decay, NorMuon normalization, weight update).
# Like the indexed kernel above, updates compute in fp32 with a single
# rounding on store, and in-place writes rule out @triton.autotune.
#
# Scalars (lr, weight decay, ...) arrive as 0-dim CPU tensors, so .item() is
# free (no GPU sync) and values are passed to the kernels by value.
# ---------------------------------------------------------------------------


@triton.heuristics(
    {
        "BLOCK_N": lambda args: max(128, min(1024, triton.next_power_of_2(args["N"]))),
        "num_warps": lambda args: 8 if args["N"] >= 2048 else 4,
    }
)
@triton.jit
def _dion2_post_ortho_masked_kernel(
    X_ptr,
    M_ptr,
    U_ptr,
    a,
    neg_b,
    ef,
    Mrows,
    N,
    x_stride_b,
    x_stride_m,
    x_stride_n,
    m_stride_b,
    m_stride_m,
    m_stride_n,
    u_stride_b,
    u_stride_m,
    u_stride_n,
    BLOCK_N: tl.constexpr,
):
    """Fused masked post for Dion2: one program per (batch, row).

    Pass 1 reduces |u| over the row to derive the selected flag (U is exactly
    zero on non-selected rows). Pass 2 applies, in fp32 with one rounding:
      selected:   x = a*x + neg_b*u   and   m = ef*m
      unselected: x = a*x             and   m untouched (ef would be *1)
    Skipping M entirely on unselected rows is bitwise identical to the
    compiled path's multiply-by-one and saves the M read+write there.
    """
    pid = tl.program_id(0)
    b = pid // Mrows
    row = pid % Mrows
    X_ptr += b * x_stride_b + row * x_stride_m
    M_ptr += b * m_stride_b + row * m_stride_m
    U_ptr += b * u_stride_b + row * u_stride_m

    absacc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        idx = off + tl.arange(0, BLOCK_N)
        msk = idx < N
        u = tl.load(U_ptr + idx * u_stride_n, mask=msk, other=0.0).to(tl.float32)
        absacc += tl.abs(u)
    sel = tl.sum(absacc) > 0.0

    if sel:
        for off in range(0, N, BLOCK_N):
            idx = off + tl.arange(0, BLOCK_N)
            msk = idx < N
            x = tl.load(X_ptr + idx * x_stride_n, mask=msk, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + idx * u_stride_n, mask=msk, other=0.0).to(tl.float32)
            m = tl.load(M_ptr + idx * m_stride_n, mask=msk, other=0.0).to(tl.float32)
            tl.store(X_ptr + idx * x_stride_n, a * x + neg_b * u, mask=msk)
            tl.store(M_ptr + idx * m_stride_n, ef * m, mask=msk)
    else:
        for off in range(0, N, BLOCK_N):
            idx = off + tl.arange(0, BLOCK_N)
            msk = idx < N
            x = tl.load(X_ptr + idx * x_stride_n, mask=msk, other=0.0).to(tl.float32)
            tl.store(X_ptr + idx * x_stride_n, a * x, mask=msk)


def dion2_post_orthogonalize_masked_triton(
    X: List[Tensor],
    M: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    ef_decay: Tensor,
    select_dim: int,
):
    """Triton-fused version of ``dion2_post_orthogonalize_masked``.

    Falls back to the compiled masked post when the fast path does not apply
    (no triton/CUDA, column selection, or tensor-subclass params). Only row
    selection (``select_dim == -2``) is fused: that is the only mode the
    global/global_capped scopes produce today.
    """
    use_fallback = (
        not TRITON_AVAILABLE
        or select_dim != -2
        or not X
        or not X[0].is_cuda
        or any(is_traceable_wrapper_subclass(x) for x in X)
    )
    if use_fallback:
        from .dion2 import dion2_post_orthogonalize_masked

        dion2_post_orthogonalize_masked(
            X=X, M=M, U=U, base_lr=base_lr, adjusted_lr=adjusted_lr,
            weight_decay=weight_decay, ef_decay=ef_decay, select_dim=select_dim,
        )
        return

    a = (1 - base_lr * weight_decay).item()
    neg_b = (-adjusted_lr).item()
    # Match the compiled path exactly: it multiplies M by ef_decay rounded to
    # M's dtype (``ef_decay.to(M[0].dtype)``), so round before upcasting.
    ef = ef_decay.to(M[0].dtype).item()

    for x, m, u in zip(X, M, U):
        if x.numel() == 0:
            continue
        if not (x.is_contiguous() and m.is_contiguous() and u.is_contiguous()):
            raise ValueError(
                "dion2_post_orthogonalize_masked_triton requires contiguous tensors"
            )
        orig_shape = x.shape
        Mrows, N = orig_shape[-2], orig_shape[-1]
        B = x.numel() // (Mrows * N)
        x_flat = x.reshape(B, Mrows, N)
        m_flat = m.reshape(B, Mrows, N)
        u_flat = u.reshape(B, Mrows, N)

        _dion2_post_ortho_masked_kernel[(B * Mrows,)](
            x_flat, m_flat, u_flat,
            a, neg_b, ef,
            Mrows, N,
            x_flat.stride(0), x_flat.stride(1), x_flat.stride(2),
            m_flat.stride(0), m_flat.stride(1), m_flat.stride(2),
            u_flat.stride(0), u_flat.stride(1), u_flat.stride(2),
        )


@triton.heuristics(
    {
        "BLOCK_N": lambda args: max(128, min(1024, triton.next_power_of_2(args["N"]))),
        "num_warps": lambda args: 8 if args["N"] >= 2048 else 4,
    }
)
@triton.jit
def _nordion2_post_ortho_masked_stats_kernel(
    U_ptr,
    V_ptr,
    M_ptr,
    P_ptr,
    ef,
    lerp_w,
    N,
    u_stride_m,
    u_stride_n,
    v_stride_m,
    m_stride_m,
    m_stride_n,
    p_stride_r,
    BLOCK_N: tl.constexpr,
):
    """NorDion2 masked post, phase A: one program per row.

    Computes the row's sum of squares of U (fp32), derives the selected flag,
    updates the per-neuron variance V (lerp toward mean-square, selected rows
    only -- unselected rows are stored back unchanged), applies error-feedback
    decay to selected rows of momentum M, and emits per-row partials into P:
      P[0, row] = sum_sq(u_row)                (-> Frobenius norm of U)
      P[1, row] = sum_sq(u_row) / denom^2      (-> Frobenius norm of U/denom)
      P[2, row] = denom = sqrt(v_new) + 1e-8   (fp32, pre-rounding V store)
    The matrix-level Frobenius rescale needs all rows, hence the two-phase
    split; the epilogue reduces P and phase B applies the weight update.
    """
    row = tl.program_id(0)
    U_ptr += row * u_stride_m
    M_ptr += row * m_stride_m
    V_ptr += row * v_stride_m

    ssq = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        idx = off + tl.arange(0, BLOCK_N)
        msk = idx < N
        u = tl.load(U_ptr + idx * u_stride_n, mask=msk, other=0.0).to(tl.float32)
        ssq += u * u
    sumsq = tl.sum(ssq)
    sel = sumsq > 0.0

    v_old = tl.load(V_ptr).to(tl.float32)
    neuron = sumsq / N.to(tl.float32)
    v_lerped = v_old + lerp_w * (neuron - v_old)
    v_new = tl.where(sel, v_lerped, v_old)
    denom = tl.sqrt(v_new) + 1e-8
    tl.store(V_ptr, v_new)

    tl.store(P_ptr + 0 * p_stride_r + row, sumsq)
    tl.store(P_ptr + 1 * p_stride_r + row, sumsq / (denom * denom))
    tl.store(P_ptr + 2 * p_stride_r + row, denom)

    if sel:
        for off in range(0, N, BLOCK_N):
            idx = off + tl.arange(0, BLOCK_N)
            msk = idx < N
            m = tl.load(M_ptr + idx * m_stride_n, mask=msk, other=0.0).to(tl.float32)
            tl.store(M_ptr + idx * m_stride_n, ef * m, mask=msk)


@triton.heuristics(
    {
        "BLOCK_N": lambda args: max(128, min(1024, triton.next_power_of_2(args["N"]))),
        "num_warps": lambda args: 8 if args["N"] >= 2048 else 4,
    }
)
@triton.jit
def _nordion2_post_ortho_masked_apply_kernel(
    X_ptr,
    U_ptr,
    P_ptr,
    scale_ptr,
    mat_idx,
    a,
    neg_b,
    N,
    x_stride_m,
    x_stride_n,
    u_stride_m,
    u_stride_n,
    p_stride_r,
    BLOCK_N: tl.constexpr,
):
    """NorDion2 masked post, phase B: one program per row.

    Applies, in fp32 with a single rounding on store:
      selected:   x = a*x + neg_b * scale * u / denom
      unselected: x = a*x   (weight decay only; u is exactly zero anyway)
    where scale = ||U||_F / max(||U/denom||_F, 1e-8) comes from the epilogue
    reduction over phase A's partials.
    """
    row = tl.program_id(0)
    X_ptr += row * x_stride_m
    U_ptr += row * u_stride_m

    sumsq = tl.load(P_ptr + 0 * p_stride_r + row)
    sel = sumsq > 0.0

    if sel:
        denom = tl.load(P_ptr + 2 * p_stride_r + row)
        scale = tl.load(scale_ptr + mat_idx).to(tl.float32)
        coef = neg_b * scale / denom
        for off in range(0, N, BLOCK_N):
            idx = off + tl.arange(0, BLOCK_N)
            msk = idx < N
            x = tl.load(X_ptr + idx * x_stride_n, mask=msk, other=0.0).to(tl.float32)
            u = tl.load(U_ptr + idx * u_stride_n, mask=msk, other=0.0).to(tl.float32)
            tl.store(X_ptr + idx * x_stride_n, a * x + coef * u, mask=msk)
    else:
        for off in range(0, N, BLOCK_N):
            idx = off + tl.arange(0, BLOCK_N)
            msk = idx < N
            x = tl.load(X_ptr + idx * x_stride_n, mask=msk, other=0.0).to(tl.float32)
            tl.store(X_ptr + idx * x_stride_n, a * x, mask=msk)


def nordion2_post_orthogonalize_masked_triton(
    X: List[Tensor],
    M: List[Tensor],
    V: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
    ef_decay: Tensor,
    muon_beta2: Tensor,
    select_dim: int,
):
    """Triton-fused version of ``nordion2_post_orthogonalize_masked``.

    Two row-programmed kernels per matrix plus one batched epilogue reduction
    per megabatch group (the NorMuon Frobenius rescale is a matrix-level
    reduction, so a grid-wide dependency forces the two-phase split; the
    epilogue is a deterministic torch reduction rather than fp32 atomics).
    Falls back to the compiled masked post when the fast path does not apply.
    All matrices in a megabatch shape group share one shape; mixed shapes fall
    back too.
    """
    shapes = {tuple(x.shape) for x in X}
    use_fallback = (
        not TRITON_AVAILABLE
        or select_dim != -2
        or not X
        or not X[0].is_cuda
        or X[0].ndim != 2
        or len(shapes) != 1
        or any(is_traceable_wrapper_subclass(x) for x in X)
    )
    if use_fallback:
        from .nordion2 import nordion2_post_orthogonalize_masked

        nordion2_post_orthogonalize_masked(
            X=X, M=M, V=V, U=U, base_lr=base_lr, adjusted_lr=adjusted_lr,
            weight_decay=weight_decay, ef_decay=ef_decay,
            muon_beta2=muon_beta2, select_dim=select_dim,
        )
        return

    Mrows, N = X[0].shape
    if Mrows == 0 or N == 0:
        return

    a = (1 - base_lr * weight_decay).item()
    neg_b = (-adjusted_lr).item()
    ef = ef_decay.to(M[0].dtype).item()
    lerp_w = (1 - muon_beta2).item()

    for x, m, v, u in zip(X, M, V, U):
        if not (
            x.is_contiguous() and m.is_contiguous() and u.is_contiguous()
            and v.is_contiguous()
        ):
            raise ValueError(
                "nordion2_post_orthogonalize_masked_triton requires contiguous tensors"
            )

    n_mat = len(X)
    device = X[0].device
    # Per-row partials: [n_mat, 3, rows] fp32 (sumsq, sumsq/denom^2, denom).
    P = torch.empty((n_mat, 3, Mrows), dtype=torch.float32, device=device)

    for i, (m, v, u) in enumerate(zip(M, V, U)):
        _nordion2_post_ortho_masked_stats_kernel[(Mrows,)](
            u, v, m, P[i],
            ef, lerp_w,
            N,
            u.stride(0), u.stride(1),
            v.stride(0),
            m.stride(0), m.stride(1),
            P.stride(1),
        )

    # Epilogue: matrix-level Frobenius rescale factors, one deterministic
    # reduction for the whole group. Matches the compiled path's
    # norm_U / clamp(norm_U_new, 1e-8).
    sums = P[:, :2, :].sum(dim=-1)  # [n_mat, 2]
    scale = sums[:, 0].sqrt() / sums[:, 1].sqrt().clamp_min(1e-8)
    scale = scale.contiguous()

    for i, (x, u) in enumerate(zip(X, U)):
        _nordion2_post_ortho_masked_apply_kernel[(Mrows,)](
            x, u, P[i], scale, i,
            a, neg_b,
            N,
            x.stride(0), x.stride(1),
            u.stride(0), u.stride(1),
            P.stride(1),
        )
