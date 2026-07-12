# tests/test_newton_shulz.py
"""
Accuracy tests for Triton Newton-Schulz kernels against a numpy float64
CPU reference. Each test asserts that the Triton kernel has similar or
better error (both mean and max) compared to PyTorch's cuBLAS for the
same operation.
"""
import numpy as np
import pytest
import torch

from dion.newton_schulz_triton import (
    TRITON_AVAILABLE,
    _batch_offset,
    ns_line_1,
    ns_line_2,
    newton_schulz_triton,
    zeropower_via_newtonschulz5,
)

torch._dynamo.config.cache_size_limit = 100  # noqa: SLF001

CUDA_AVAILABLE = torch.cuda.is_available()

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _fixed_offset_probe(out_ptr, batch_idx, batch_stride):
        # Exercises the shipped helper: the multiply must happen in int64.
        tl.store(out_ptr, _batch_offset(batch_idx, batch_stride))

    @triton.jit
    def _naive_offset_probe(out_ptr, batch_idx, batch_stride):
        # Pre-fix behaviour: an int32 * int32 multiply that overflows.
        tl.store(out_ptr, (batch_idx * batch_stride).to(tl.int64))


# For bf16/f16, Triton should be at least as accurate as cuBLAS (multiplier=1).
# For f32, Triton's tl.dot uses a less favorable internal reduction tree than
# cuBLAS even with input_precision="ieee", so we allow some slack.
# Empirically (unbatched, 20 runs each):
#   mean ratio: up to ~3.6x  (shape 256x1024)
#   max  ratio: up to ~14x   (shape 256x1024, outlier-sensitive)
# Batched cases show ratio 1.0 because torch.bmm uses the same reduction
# order as Triton (i.e. both produce bitwise-identical results), unlike
# torch.mm which uses a different cuBLAS algorithm.
# This is a Triton limitation — improving it would require raw CUDA.
_F32_MEAN_ERR_MULTIPLIER = 5
_F32_MAX_ERR_MULTIPLIER = 15


def _abs_errs(result: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """Return (mean, max) absolute error between a GPU result and a CPU reference."""
    diff = (result.cpu().float() - reference.float()).abs()
    return diff.mean().item(), diff.max().item()


def _numpy_ref_aat(A: torch.Tensor) -> torch.Tensor:
    """Compute A @ A^T in numpy float64, return as float32."""
    a = A.cpu().float().numpy().astype(np.float64)
    out = a @ a.T if a.ndim == 2 else a @ np.swapaxes(a, -2, -1)
    return torch.from_numpy(out.astype(np.float32))


def _numpy_ref_ns_line_2(A: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Compute alpha * A @ A^T + beta * A in numpy float64."""
    a = A.cpu().float().numpy().astype(np.float64)
    aT = a.T if a.ndim == 2 else np.swapaxes(a, -2, -1)
    out = alpha * (a @ aT) + beta * a
    return torch.from_numpy(out.astype(np.float32))


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="triton not installed")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize("m,n", [(256, 256), (256, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_ns_line_1_accuracy(m: int, n: int, dtype: torch.dtype):
    """Triton ns_line_1 should have similar or better error than cuBLAS for A @ A^T."""
    mean_mul = _F32_MEAN_ERR_MULTIPLIER if dtype == torch.float32 else 1
    max_mul = _F32_MAX_ERR_MULTIPLIER if dtype == torch.float32 else 1
    for A in [
        torch.randn(m, n, dtype=dtype, device="cuda"),
        torch.randn(4, m, n, dtype=dtype, device="cuda"),
    ]:
        ref = _numpy_ref_aat(A)
        triton_mean, triton_max = _abs_errs(ns_line_1(A), ref)
        cublas_mean, cublas_max = _abs_errs(A @ A.mT, ref)
        assert triton_mean <= cublas_mean * mean_mul, (
            f"Triton mean err {triton_mean:.3e} > cuBLAS mean err {cublas_mean:.3e} * {mean_mul} "
            f"(shape={tuple(A.shape)}, dtype={A.dtype})"
        )
        assert triton_max <= cublas_max * max_mul, (
            f"Triton max err {triton_max:.3e} > cuBLAS max err {cublas_max:.3e} * {max_mul} "
            f"(shape={tuple(A.shape)}, dtype={A.dtype})"
        )


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="triton not installed")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_ns_line_2_accuracy(m: int, dtype: torch.dtype):
    """Triton ns_line_2 should have similar or better error than cuBLAS."""
    mean_mul = _F32_MEAN_ERR_MULTIPLIER if dtype == torch.float32 else 1
    max_mul = _F32_MAX_ERR_MULTIPLIER if dtype == torch.float32 else 1
    alpha, beta = torch.randn(1).item(), torch.randn(1).item()

    for A in [
        torch.randn(m, m, dtype=dtype, device="cuda"),
        torch.randn(4, m, m, dtype=dtype, device="cuda"),
    ]:
        A = (A + A.mT) / 2
        ref = _numpy_ref_ns_line_2(A, alpha, beta)
        triton_mean, triton_max = _abs_errs(ns_line_2(A, alpha=alpha, beta=beta), ref)
        cublas_mean, cublas_max = _abs_errs(alpha * (A @ A.mT) + beta * A, ref)
        assert triton_mean <= cublas_mean * mean_mul, (
            f"Triton mean err {triton_mean:.3e} > cuBLAS mean err {cublas_mean:.3e} * {mean_mul} "
            f"(shape={tuple(A.shape)}, dtype={A.dtype})"
        )
        assert triton_max <= cublas_max * max_mul, (
            f"Triton max err {triton_max:.3e} > cuBLAS max err {cublas_max:.3e} * {max_mul} "
            f"(shape={tuple(A.shape)}, dtype={A.dtype})"
        )


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="triton not installed")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
@pytest.mark.parametrize("m,n", [(256, 256), (256, 1024)])
def test_newton_schulz_triton_vs_reference(m: int, n: int):
    """Triton and reference Newton-Schulz should agree within tolerance.

    Both implementations use the same algorithm (same constants, same
    iteration count) and always operate in bf16 internally. Small
    differences arise from kernel-level reduction order.
    """
    for G in [
        torch.randn(m, n, dtype=torch.float32, device="cuda"),
        torch.randn(4, m, n, dtype=torch.float32, device="cuda"),
    ]:
        triton_out = newton_schulz_triton(G)
        ref_out = zeropower_via_newtonschulz5(G)
        diff = (triton_out - ref_out).abs().max().item()
        # Empirically max diff is ~7.8e-3 across 50 runs; 0.02 gives ~2.5x headroom.
        assert diff < 0.02, (
            f"Newton-Schulz implementations diverged: max diff {diff:.3e} "
            f"(shape={tuple(G.shape)}, dtype={G.dtype})"
        )


# Regression tests for int64 batched pointer offsets (PR #102).
#
# Real Muon megabatches (e.g. the w13 tensor with shape [320, 2560, 3072]) push
# the last batch's element offset past INT32_MAX. The per-matrix stride still
# fits in int32, so the overflow comes purely from batch_idx * stride: the helper
# must widen batch_idx to int64 before the multiply. These probes verify that in
# ~one element of memory instead of allocating the multi-GB tensor it would take
# to trigger the overflow through the public kernels.
_NS_ROWS, _NS_COLS, _NS_BATCHES = 2560, 3072, 320
_INT32_MAX = 2**31 - 1


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="triton not installed")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
def test_batch_offset_no_int32_overflow():
    """_batch_offset computes batch_idx * stride in int64 for large batches."""
    stride = _NS_ROWS * _NS_COLS  # 7,864,320 (fits in int32)
    last_batch = _NS_BATCHES - 1  # 319
    expected = last_batch * stride  # 2,508,718,080 (exceeds int32)
    assert stride < _INT32_MAX
    assert expected > _INT32_MAX

    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    _fixed_offset_probe[(1,)](out, last_batch, stride)
    assert out.item() == expected, f"overflow: got {out.item()}, want {expected}"

    # The guard has teeth: the pre-fix int32 multiply wraps to a wrong (negative)
    # offset for the same inputs, so a regression would fail the assert above.
    naive = torch.zeros(1, dtype=torch.int64, device="cuda")
    _naive_offset_probe[(1,)](naive, last_batch, stride)
    assert naive.item() != expected
    assert naive.item() < 0


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="triton not installed")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA device required")
def test_batch_offset_ttir_multiplies_in_i64():
    """The generated TTIR performs the batch-offset multiply in i64."""
    stride = _NS_ROWS * _NS_COLS
    last_batch = _NS_BATCHES - 1
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    compiled = _fixed_offset_probe.warmup(out, last_batch, stride, grid=(1,))
    ttir = compiled.asm["ttir"]

    muli_lines = [ln for ln in ttir.splitlines() if "arith.muli" in ln]
    assert muli_lines, f"no integer multiply found in TTIR:\n{ttir}"
    non_i64 = [ln for ln in muli_lines if "i64" not in ln]
    assert not non_i64, f"batch-offset multiply not in i64: {non_i64}"
