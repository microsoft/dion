"""
JAX/Optax implementation of the DION optimizer.
Based on the PyTorch reference implementation in dion_reference.py

https://arxiv.org/abs/2504.05295
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.tree_util import tree_map


@dataclass
class DionMixedPrecisionConfig:
    """Configuration for mixed precision in Dion optimizer."""
    momentum_dtype: Optional[jnp.dtype] = None
    Q_dtype: Optional[jnp.dtype] = None
    variance_dtype: Optional[jnp.dtype] = None


class DionState(NamedTuple):
    """State for the DION optimizer."""
    momentum: Any  # Momentum buffer
    Q: Any  # Q matrix for power iteration
    variance: Optional[Any] = None  # For AdamW variant
    count: Any = None  # Step counter
    mu: Any = None  # For schedule
    rng_key: Optional[Any] = None  # Random key for RCQR


def dion(
    learning_rate: Union[float, optax.Schedule],
    rank_fraction: float = 1.0,
    rank_multiple_of: int = 1,
    mu: float = 0.95,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    power_iters: int = 1,
    qr_method: str = "rcqr",
    cqr_warmup_steps: int = 150,
    rcqr_oversample: float = 1.25,
    mixed_precision_config: Optional[DionMixedPrecisionConfig] = None,
    algorithm: str = "dion",
    seed: int = 0,
) -> optax.GradientTransformation:
    """
    Create a DION optimizer.
    
    Args:
        learning_rate: Learning rate or schedule
        rank_fraction: r/d fraction for low-rank approximation
        rank_multiple_of: Round up the low-rank dimension to a multiple of this
        mu: Momentum factor for DION
        betas: Beta parameters for AdamW variant
        weight_decay: Weight decay coefficient
        eps: Small constant for numerical stability
        power_iters: Number of power iterations
        qr_method: Method for QR decomposition ('qr', 'cqr', 'rcqr')
        cqr_warmup_steps: Number of warmup steps before enabling CQR
        rcqr_oversample: Oversampling factor for RCQR
        mixed_precision_config: Configuration for mixed precision
        algorithm: Algorithm variant ('dion', 'adamw', 'lion')
        seed: Random seed for initialization
    
    Returns:
        An optax gradient transformation
    """
    if mixed_precision_config is None:
        mixed_precision_config = DionMixedPrecisionConfig()
    
    def init_fn(params):
        """Initialize optimizer state."""
        rng_key = jax.random.PRNGKey(seed)
        
        def init_param(key, param):
            if algorithm == "dion" and param.ndim == 2:
                # Initialize DION state for matrix parameters
                m, n = param.shape
                r = int(rank_fraction * min(m, n))
                r = rank_multiple_of * math.ceil(r / rank_multiple_of)
                r = min(r, m, n)
                
                # Determine Q shape based on transposition
                is_transposed = m < n
                Q_shape = (m, r) if is_transposed else (n, r)
                
                # Initialize Q matrix
                Q_dtype = mixed_precision_config.Q_dtype or param.dtype
                Q = jax.random.normal(key, Q_shape, dtype=Q_dtype)
                
                # Initialize momentum
                momentum_dtype = mixed_precision_config.momentum_dtype or param.dtype
                momentum = jnp.zeros_like(param, dtype=momentum_dtype)
                
                return DionState(
                    momentum=momentum,
                    Q=Q,
                    count=jnp.zeros([], jnp.int32),
                    mu=jnp.array(mu, dtype=jnp.float32),
                    rng_key=key
                )
            elif algorithm == "adamw":
                # Initialize AdamW state
                momentum_dtype = mixed_precision_config.momentum_dtype or param.dtype
                variance_dtype = mixed_precision_config.variance_dtype or param.dtype
                
                return DionState(
                    momentum=jnp.zeros_like(param, dtype=momentum_dtype),
                    Q=None,
                    variance=jnp.zeros_like(param, dtype=variance_dtype),
                    count=jnp.zeros([], jnp.int32),
                    mu=None,
                    rng_key=None
                )
            else:  # lion or scalar parameters
                momentum_dtype = mixed_precision_config.momentum_dtype or param.dtype
                
                return DionState(
                    momentum=jnp.zeros_like(param, dtype=momentum_dtype),
                    Q=None,
                    variance=None,
                    count=jnp.zeros([], jnp.int32),
                    mu=None,
                    rng_key=None
                )
        
        # Split keys for each parameter
        param_keys = jax.random.split(rng_key, len(jax.tree_util.tree_leaves(params)))
        key_iter = iter(param_keys)
        
        return tree_map(lambda p: init_param(next(key_iter), p), params)
    
    def update_fn(updates, state, params):
        """Apply DION updates."""
        if callable(learning_rate):
            lr = learning_rate(state.count)
        else:
            lr = learning_rate
        
        def update_param(grad, state, param):
            if algorithm == "dion" and param.ndim == 2:
                # DION update for matrix parameters
                new_state, new_param = dion_update(
                    param, grad, state,
                    lr=lr, weight_decay=weight_decay, eps=eps,
                    power_iters=power_iters, qr_method=qr_method,
                    cqr_warmup_steps=cqr_warmup_steps,
                    rcqr_oversample=rcqr_oversample
                )
                return -new_param + param, new_state
            
            elif algorithm == "adamw":
                # AdamW update
                new_state, new_param = adamw_update(
                    param, grad, state,
                    lr=lr, beta1=betas[0], beta2=betas[1],
                    weight_decay=weight_decay, eps=eps
                )
                return -new_param + param, new_state
            
            else:  # lion or scalar parameters
                # Lion update
                new_state, new_param = lion_update(
                    param, grad, state,
                    lr=lr, beta1=betas[0], beta2=betas[1],
                    weight_decay=weight_decay
                )
                return -new_param + param, new_state
        
        updates, new_state = tree_map(update_param, updates, state, params)
        
        # Increment step counter
        new_state = tree_map(
            lambda s: s._replace(count=s.count + 1) if s.count is not None else s,
            new_state
        )
        
        return updates, new_state
    
    return optax.GradientTransformation(init_fn, update_fn)


@partial(jax.jit, static_argnames=('power_iters', 'qr_method', 'cqr_warmup_steps'))
def dion_update(
    X: jnp.ndarray,  # Model weights
    G: jnp.ndarray,  # Gradient
    state: DionState,  # Optimizer state
    lr: float,
    weight_decay: float,
    eps: float,
    power_iters: int,
    qr_method: str,
    cqr_warmup_steps: int,
    rcqr_oversample: float,
) -> Tuple[DionState, jnp.ndarray]:
    """DION optimizer update step."""
    M = state.momentum
    Q = state.Q
    mu = state.mu
    step = state.count
    rng_key = state.rng_key
    
    # Match dtype of Q and M
    Q = Q.astype(M.dtype)
    
    # Add gradient to momentum
    M = M + G
    
    # Determine if we should transpose
    m, n = X.shape
    is_transposed = m < n
    
    # Compute low-rank approximation M ≈ PQ^T
    if rng_key is not None:
        rng_key, subkey = jax.random.split(rng_key)
    else:
        subkey = None
    
    P, R = power_iteration(
        M.T if is_transposed else M,
        Q,
        power_iters=power_iters,
        qr_method=qr_method if step > cqr_warmup_steps else "rcqr",
        oversample=rcqr_oversample,
        rng_key=subkey
    )
    
    # Handle all-zero case
    P, R = fix_all_zero_or_nan(P, R, Q, M)
    
    # Error feedback: M = M - (1 - mu) * (P @ R.T)
    if not is_transposed:
        M = M - (1 - mu) * (P @ R.T)
    else:
        M = M - (1 - mu) * (R @ P.T)
    
    # Column normalize R to get new Q
    R = R.astype(jnp.float32)
    R_norm = jnp.linalg.norm(R, axis=0, keepdims=True) + eps
    Q = (R / R_norm).astype(P.dtype)
    
    # Apply weight decay
    X = X * (1 - lr * weight_decay)
    
    # Compute update scale factor
    fan_out, fan_in = X.shape
    scaled_lr = ((fan_out / fan_in) ** 0.5) * lr
    
    # Apply weight update
    if not is_transposed:
        X = X - scaled_lr * (P @ Q.T)
    else:
        X = X - scaled_lr * (Q @ P.T)
    
    # Update state
    new_state = state._replace(
        momentum=M,
        Q=Q,
        count=step + 1,
        rng_key=rng_key
    )
    
    return new_state, X


def power_iteration(
    B: jnp.ndarray,
    Q_init: jnp.ndarray,
    power_iters: int,
    qr_method: str,
    oversample: float,
    rng_key: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute low-rank approximation B ≈ PQ^T using power iteration."""
    Q = Q_init
    
    for _ in range(power_iters):
        P = B @ Q
        P = orthogonalize(P, qr_method=qr_method, oversample=oversample, rng_key=rng_key)
        Q = B.T @ P
    
    return P, Q


def orthogonalize(
    P: jnp.ndarray,
    qr_method: str = "rcqr",
    oversample: float = 1.25,
    rng_key: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Orthogonalize matrix using specified method."""
    m, n = P.shape
    original_dtype = P.dtype
    
    if qr_method == "cqr":
        # Cholesky QR
        P_32 = P.astype(jnp.float32)
        try:
            R = jnp.linalg.cholesky(P_32.T @ P_32)
            Q = jax.scipy.linalg.solve_triangular(R, P_32.T, lower=False).T
        except:
            # Fallback to RCQR if Cholesky fails
            qr_method = "rcqr"
    
    if qr_method == "qr" or (qr_method == "rcqr" and m <= n):
        # Standard QR
        Q, _ = jnp.linalg.qr(P.astype(jnp.float32))
    
    if qr_method == "rcqr" and m > n:
        # Randomized Cholesky QR
        k = math.ceil(oversample * n / 128.0) * 128
        std = math.sqrt(1.0 / k)
        
        # Generate random sketch matrix
        if rng_key is not None:
            S = jax.random.normal(rng_key, (k, m), dtype=P.dtype) * std
        else:
            # Fallback to deterministic initialization
            S = jnp.ones((k, m), dtype=P.dtype) * std
        
        SP = S @ P
        
        # QR decomposition
        _, R = jnp.linalg.qr(SP.astype(jnp.float32))
        Q = jax.scipy.linalg.solve_triangular(R, P.astype(jnp.float32).T, lower=False).T
        
        # Second iteration for better orthogonalization
        QQ = Q.T @ Q
        R = jnp.linalg.cholesky(QQ)
        Q = jax.scipy.linalg.solve_triangular(R, Q.T, lower=False).T
    
    return Q.astype(original_dtype)


def fix_all_zero_or_nan(
    P: jnp.ndarray,
    R: jnp.ndarray,
    Q_init: jnp.ndarray,
    B: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Handle all-zero or NaN cases."""
    is_all_zero = jnp.all(B == 0)
    not_all_zero = ~is_all_zero
    
    P = jnp.nan_to_num(P) * not_all_zero
    R = jnp.nan_to_num(R) * not_all_zero + Q_init * is_all_zero
    
    return P, R


@jax.jit
def adamw_update(
    X: jnp.ndarray,
    G: jnp.ndarray,
    state: DionState,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
) -> Tuple[DionState, jnp.ndarray]:
    """AdamW optimizer update."""
    M = state.momentum
    V = state.variance
    step = state.count + 1
    
    # Update momentum and variance
    M = beta1 * M + (1 - beta1) * G
    V = beta2 * V + (1 - beta2) * (G * G)
    
    # Bias correction
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    # Compute update
    M_hat = M / bias_correction1
    V_hat = V / bias_correction2
    
    # Apply weight decay
    X = X * (1 - lr * weight_decay)
    
    # Apply update
    X = X - lr * M_hat / (jnp.sqrt(V_hat) + eps)
    
    new_state = state._replace(
        momentum=M,
        variance=V,
        count=step
    )
    
    return new_state, X


@jax.jit
def lion_update(
    X: jnp.ndarray,
    G: jnp.ndarray,
    state: DionState,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
) -> Tuple[DionState, jnp.ndarray]:
    """Lion optimizer update."""
    M = state.momentum
    
    # Compute update direction
    update = beta1 * M + (1 - beta1) * G
    
    # Apply weight decay
    X = X * (1 - lr * weight_decay)
    
    # Apply update with sign
    X = X - lr * jnp.sign(update)
    
    # Update momentum
    M = beta2 * M + (1 - beta2) * G
    
    new_state = state._replace(momentum=M, count=state.count + 1)
    
    return new_state, X


# Utility functions for creating parameter groups
def create_param_groups(params, is_embedding_fn=None, is_lm_head_fn=None):
    """
    Create parameter groups for different algorithms.
    
    Args:
        params: Model parameters
        is_embedding_fn: Function to identify embedding parameters
        is_lm_head_fn: Function to identify language model head parameters
    
    Returns:
        List of parameter groups with algorithm assignments
    """
    matrix_params = []
    vector_params = []
    embed_params = []
    lm_head_params = []
    
    def categorize_param(path, param):
        if param.ndim == 2:
            if is_embedding_fn and is_embedding_fn(path):
                embed_params.append((path, param))
            elif is_lm_head_fn and is_lm_head_fn(path):
                lm_head_params.append((path, param))
            else:
                matrix_params.append((path, param))
        else:
            vector_params.append((path, param))
    
    # Traverse parameter tree
    jax.tree_util.tree_map_with_path(categorize_param, params)
    
    return {
        'matrix': matrix_params,
        'vector': vector_params,
        'embedding': embed_params,
        'lm_head': lm_head_params
    }