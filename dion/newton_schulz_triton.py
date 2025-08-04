import torch
import triton
import triton.language as tl
from torch import Tensor


def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Helper function to map Triton program ID to (batch, row, col) of the output matrix.
    """
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N

    return batch_idx, m_idx, n_idx


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_1_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """
    Input A has shape (M, K)
    Output C has shape (M, M)
    Compute C = A @ A.T
    """

    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ns_line_1(A: Tensor, *, out: Tensor = None):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    if A.ndim > 3 or A.ndim < 2:
        raise ValueError(f"Input tensor must be 2D or 3D, but got {A.ndim}D tensor.")

    M, K = A.shape[-2:]

    if out is None:
        out = torch.empty((*A.shape[:-1], M), device=A.device, dtype=A.dtype)
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_1_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )

    return out


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_2_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """
    Input A is square matrix with shape (M, M)
    Output C has shape (M, M)
    Compute C = alpha * A @ A.T + beta * A
    """

    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ns_line_2(A: Tensor, alpha: float, beta: float, *, out: Tensor = None):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    if A.ndim > 3 or A.ndim < 2:
        raise ValueError(f"Input tensor must be 2D or 3D, but got {A.ndim}D tensor.")

    M, K = A.shape[-2:]
    if M != K:
        raise ValueError(
            f"Input must be symmetric square matrix, but got shape {A.shape}"
        )

    if out is None:
        out = torch.empty((*A.shape[:-1], M), device=A.device, dtype=A.dtype)
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_2_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )

    return out


@torch.compile(dynamic=False, fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Reference implementation of Newton-Schulz without Triton.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def newton_schulz_triton(G: Tensor, epsilon: float = 1e-7):
    """
    Triton implementation of Newton-Schulz iteration
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the NS iterations
    for a, b, c in ns_consts:
        ns_line_1(X, out=A)  # A = X @ X.mT
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def check_result(result, correct, atol=1e-2):
    assert (
        result.dtype == correct.dtype
    ), f"Result dtype {result.dtype} does not match correct dtype {correct.dtype}"
    assert (
        result.shape == correct.shape
    ), f"Shape mismatch: {result.shape} != {correct.shape}"

    if torch.allclose(result, correct, atol=atol):
        print("Test passed")
    else:
        print("Test failed")
        if torch.allclose(result.triu(), correct.triu(), atol=atol):
            print("- Upper triangular part matches")
        if torch.allclose(result.tril(), correct.tril(), atol=atol):
            print("- Lower triangular part matches")
        abs_diff = torch.abs(result - correct)
        print("- Max absolute difference:", abs_diff.max().item())
        print(abs_diff)


def test_ns_line_1(m, n, dtype=torch.bfloat16):
    print(f"Testing ns_line_1 with shape ({m}, {n}) and dtype {dtype}")

    A = torch.randn(m, n, dtype=dtype, device="cuda")
    result = ns_line_1(A)
    correct = A @ A.mT
    check_result(result, correct)

    # Test with batch dimension
    A = torch.randn(4, m, n, dtype=dtype, device="cuda")
    result = ns_line_1(A)
    correct = A @ A.mT
    check_result(result, correct)


def test_ns_line_2(m, dtype=torch.bfloat16):
    print(f"Testing ns_line_2 with shape ({m}, {m}) and dtype {dtype}")

    A = torch.randn(m, m, dtype=dtype, device="cuda")
    A = (A + A.mT) / 2  # Make symmetric
    alpha = torch.randn(1).item()
    beta = torch.randn(1).item()
    result = ns_line_2(A, alpha=alpha, beta=beta)

    A = A.to(torch.float32)
    correct = alpha * (A @ A.mT) + beta * A
    check_result(result, correct.to(dtype))

    # Test with batch dimension
    A = torch.randn(4, m, m, dtype=dtype, device="cuda")
    A = (A + A.mT) / 2  # Make symmetric
    result = ns_line_2(A, alpha=alpha, beta=beta)

    A = A.to(torch.float32)
    correct = alpha * (A @ A.mT) + beta * A
    check_result(result, correct.to(dtype))


def test_newton_schulz_triton(m, n, dtype=torch.bfloat16):
    print(f"Testing newton_schulz_triton with shape ({m}, {n}) and dtype {dtype}")

    G = torch.randn(m, n, dtype=dtype, device="cuda")
    result = newton_schulz_triton(G)
    correct = zeropower_via_newtonschulz5(G)
    check_result(result, correct)

    # Test with batch dimension
    G = torch.randn(4, m, n, dtype=dtype, device="cuda")
    result = newton_schulz_triton(G)
    correct = zeropower_via_newtonschulz5(G)
    check_result(result, correct)


def benchmark_newton_schulz_triton(m, n, batch_size=1, dtype=torch.bfloat16):
    print(
        f"Benchmarking newton_schulz_triton with shape ({m}, {n}) and batch size {batch_size}"
    )
    G = torch.randn(batch_size, m, n, dtype=dtype, device="cuda")
    if batch_size == 1:
        G = G.squeeze(0)

    def estimate_tflops(ms):
        steps = 5  # Number of Newton-Schulz iterations
        mm_cost = (2 * m * n * m) + (2 * m * m * m) + (2 * m * m * n)
        return batch_size * steps * mm_cost * 1e-12 / (ms * 1e-3)

    time_torch = triton.testing.do_bench(lambda: zeropower_via_newtonschulz5(G))
    print(f"Torch NS: {time_torch:.4f} ms, {estimate_tflops(time_torch):.2f} TFLOP/s")

    time_triton = triton.testing.do_bench(lambda: newton_schulz_triton(G))
    print(
        f"Triton NS: {time_triton:.4f} ms, {estimate_tflops(time_triton):.2f} TFLOP/s"
    )

    print(f"Speedup: {time_torch / time_triton:.2f}x")
    return time_torch, time_triton


def benchmark_many_sizes(batch_size=1, expansion=1, dtype=torch.bfloat16):
    dim = [512, 1024, 2048, 4096, 8192]
    speedups = []

    for d in dim:
        time_torch, time_triton = benchmark_newton_schulz_triton(
            d, d * expansion, batch_size=batch_size, dtype=dtype
        )
        time_ratio = time_torch / time_triton
        speedups.append(time_ratio)

    print(f"Speedups: {speedups}")
    max_speedup = (4 * expansion + 2) / (3 * expansion + 1)
    print(f"Maximum theoretical speedup: {max_speedup:.2f}x")


def benchmark_plot(batch_size=1):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["d"],
            x_vals=[128 * i for i in range(1, 33)],
            line_arg="provider",
            line_names=["torch", "triton"],
            line_vals=["torch", "triton"],
            ylabel="TFLOPS",
            plot_name=f"newton_schulz_{batch_size=}",
            args={"batch_size": batch_size},
        )
    )
    def benchmark(d: int, provider: str, batch_size: int):
        G = torch.randn(batch_size, d, d, dtype=torch.bfloat16, device="cuda")

        if provider == "torch":
            ms = triton.testing.do_bench(lambda: zeropower_via_newtonschulz5(G))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: newton_schulz_triton(G))

        def estimate_tflops(ms):
            steps = 5
            mm_cost = (2 * d * d * d) + (2 * d * d * d) + (2 * d * d * d)
            return batch_size * steps * mm_cost * 1e-12 / (ms * 1e-3)

        return estimate_tflops(ms)

    benchmark.run(print_data=True, save_path="plots")


if __name__ == "__main__":
    # Allow a lot of recompiles
    torch._dynamo.config.cache_size_limit = 100

    # Run tests
    # test_ns_line_1(1024, 1024)
    # test_ns_line_1(1024, 4096)
    # test_ns_line_2(1024)
    # test_newton_schulz_triton(1024, 1024)
    # test_newton_schulz_triton(1024, 4096)

    # d = 1024
    # benchmark_newton_schulz_triton(d, d)
    # benchmark_newton_schulz_triton(d, d, batch_size=4)

    # benchmark_many_sizes(batch_size=1, expansion=1)
    # benchmark_many_sizes(batch_size=4, expansion=1)
    # benchmark_many_sizes(batch_size=1, expansion=4)
    # benchmark_many_sizes(batch_size=4, expansion=4)

    benchmark_plot(batch_size=1)
    benchmark_plot(batch_size=4)
