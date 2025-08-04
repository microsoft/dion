"""Utilities for testing, including checking for optional dependencies."""

import pytest
import importlib


def has_module(module_name: str) -> bool:
    """Check if a module is available."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def has_triton() -> bool:
    """Check if triton is available."""
    return has_module('triton')


def has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_distributed() -> bool:
    """Check if distributed training is available."""
    try:
        import torch.distributed as dist
        return dist.is_available()
    except ImportError:
        return False


# Pytest markers for optional dependencies
requires_triton = pytest.mark.skipif(not has_triton(), reason="requires triton")
requires_cuda = pytest.mark.skipif(not has_cuda(), reason="requires CUDA")
requires_distributed = pytest.mark.skipif(not has_distributed(), reason="requires distributed")


def skip_if_import_fails(import_func):
    """Decorator to skip test if import fails."""
    def decorator(test_func):
        try:
            import_func()
            return test_func
        except ImportError as e:
            return pytest.mark.skip(reason=f"Import failed: {e}")(test_func)
    return decorator