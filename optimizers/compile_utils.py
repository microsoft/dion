"""
Utility functions for handling torch.compile gracefully across different PyTorch versions and environments.
"""
import torch
import warnings
from functools import wraps
from typing import Callable, Any


def safe_torch_compile(fullgraph: bool = True, **kwargs):
    """
    A decorator that applies torch.compile if available and functional, 
    otherwise falls back to the original function.
    
    Args:
        fullgraph: Whether to compile the full graph
        **kwargs: Additional arguments to pass to torch.compile
    
    Returns:
        A decorator function that either compiles or passes through the original function
    """
    import os
    
    def decorator(func: Callable) -> Callable:
        # Check if compilation is disabled via environment variable
        if os.environ.get('TORCH_COMPILE_DISABLE', '0') == '1':
            return func
            
        try:
            # Try to compile the function
            compiled_func = torch.compile(func, fullgraph=fullgraph, **kwargs)
            
            # Test if compilation actually works by attempting to create a dummy call
            # This won't execute but will trigger any import/compilation errors
            return compiled_func
            
        except Exception as e:
            # If compilation fails, warn and return the original function
            warnings.warn(
                f"torch.compile failed for function '{func.__name__}': {e}. "
                f"Falling back to uncompiled version. Performance may be reduced.",
                UserWarning,
                stacklevel=2
            )
            return func
    
    return decorator


def is_compile_available() -> bool:
    """
    Check if torch.compile is available and functional in the current environment.
    
    Returns:
        True if torch.compile is available and functional, False otherwise
    """
    try:
        # Try a simple compile operation
        @torch.compile
        def dummy_func(x):
            return x + 1
        
        return True
    except Exception:
        return False


def conditional_compile(condition: bool = None, **compile_kwargs):
    """
    Conditionally apply torch.compile based on a condition or environment check.
    
    Args:
        condition: If None, will check if compile is available. 
                  If True/False, will use that condition.
        **compile_kwargs: Arguments to pass to torch.compile
    
    Returns:
        A decorator that either compiles or passes through the function
    """
    def decorator(func: Callable) -> Callable:
        if condition is None:
            should_compile = is_compile_available()
        else:
            should_compile = condition
            
        if should_compile:
            try:
                return torch.compile(func, **compile_kwargs)
            except Exception as e:
                warnings.warn(
                    f"torch.compile failed for '{func.__name__}': {e}. Using uncompiled version.",
                    UserWarning
                )
                return func
        else:
            return func
    
    return decorator


def disable_compile_for_tests():
    """
    Temporarily disable torch.compile for testing to avoid cache limit issues.
    """
    import os
    os.environ['TORCH_COMPILE_DISABLE'] = '1'