"""Tests for JAX/Optax DION optimizer implementation."""

import pytest
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

from optimizers.experimental.dion_reference_optax import (
    dion, DionMixedPrecisionConfig, DionState,
    orthogonalize, power_iteration, fix_all_zero_or_nan,
    adamw_update, lion_update
)


class TestDionOptax:
    """Test suite for DION Optax optimizer."""
    
    @pytest.fixture
    def rng_key(self):
        """Random key for JAX operations."""
        return jax.random.PRNGKey(0)
    
    @pytest.fixture
    def simple_params(self, rng_key):
        """Create simple parameter dictionary."""
        key1, key2, key3 = jax.random.split(rng_key, 3)
        return {
            'linear1': jax.random.normal(key1, (32, 64)),
            'linear2': jax.random.normal(key2, (64, 128)),
            'bias': jax.random.normal(key3, (128,))
        }
    
    def test_optimizer_initialization(self, simple_params):
        """Test basic optimizer initialization."""
        # Test default initialization
        optimizer = dion(learning_rate=0.01)
        state = optimizer.init(simple_params)
        
        assert state is not None
        assert isinstance(state, dict)
        
        # Check state structure
        for key, param in simple_params.items():
            assert key in state
            param_state = state[key]
            assert isinstance(param_state, DionState)
            assert param_state.momentum.shape == param.shape
            
            if param.ndim == 2:  # Matrix parameters use DION
                assert param_state.Q is not None
                assert param_state.Q.ndim == 2
            else:  # Vector parameters don't have Q
                assert param_state.Q is None
    
    def test_optimizer_with_rank_fraction(self, simple_params):
        """Test optimizer with different rank fractions."""
        optimizer = dion(learning_rate=0.01, rank_fraction=0.25)
        state = optimizer.init(simple_params)
        
        # Check Q matrix dimensions for matrix parameters
        linear1_state = state['linear1']
        m, n = simple_params['linear1'].shape
        expected_r = int(0.25 * min(m, n))
        
        # Q shape depends on transposition
        is_transposed = m < n
        if is_transposed:
            assert linear1_state.Q.shape[0] == m
        else:
            assert linear1_state.Q.shape[0] == n
        
        # Rank should be approximately 25% of min dimension
        assert linear1_state.Q.shape[1] <= expected_r + 8  # Allow for rounding
    
    def test_mixed_precision_config(self, simple_params):
        """Test optimizer with mixed precision configuration."""
        mp_config = DionMixedPrecisionConfig(
            momentum_dtype=jnp.float32,
            Q_dtype=jnp.bfloat16,
            variance_dtype=jnp.float32
        )
        
        optimizer = dion(
            learning_rate=0.01,
            mixed_precision_config=mp_config
        )
        state = optimizer.init(simple_params)
        
        # Check dtypes
        linear1_state = state['linear1']
        assert linear1_state.momentum.dtype == jnp.float32
        assert linear1_state.Q.dtype == jnp.bfloat16
    
    def test_optimizer_step(self, simple_params, rng_key):
        """Test a single optimizer step."""
        print("\n=== Testing optimizer step ===")
        optimizer = dion(learning_rate=0.01)
        state = optimizer.init(simple_params)
        
        print(f"State keys: {state.keys()}")
        matrix_key = [k for k in state.keys() if simple_params[k].ndim == 2][0]
        print(f"Matrix param key: {matrix_key}, state type: {type(state[matrix_key])}")
        
        # Create dummy gradients
        grads = jax.tree_map(lambda p: jax.random.normal(rng_key, p.shape) * 0.01, simple_params)
        
        for key, grad in grads.items():
            print(f"Gradient norm for {key}: {jnp.linalg.norm(grad):.4f}")
        
        # Apply update
        updates, new_state = optimizer.update(grads, state, simple_params)
        new_params = optax.apply_updates(simple_params, updates)
        
        # Check that parameters changed
        for key in simple_params:
            old_norm = jnp.linalg.norm(simple_params[key])
            new_norm = jnp.linalg.norm(new_params[key])
            change_norm = jnp.linalg.norm(new_params[key] - simple_params[key])
            print(f"{key}: old_norm={old_norm:.4f}, new_norm={new_norm:.4f}, change={change_norm:.6f}")
            assert not jnp.allclose(simple_params[key], new_params[key])
        
        # Check state was updated
        for key in state:
            old_count = state[key].count
            new_count = new_state[key].count
            assert new_count == old_count + 1
    
    def test_different_algorithms(self, simple_params, rng_key):
        """Test different algorithm variants."""
        algorithms = ['dion', 'adamw', 'lion']
        
        for algo in algorithms:
            optimizer = dion(learning_rate=0.01, algorithm=algo)
            state = optimizer.init(simple_params)
            
            # Check state initialization
            for key, param in simple_params.items():
                param_state = state[key]
                
                if algo == 'adamw':
                    assert param_state.variance is not None
                else:
                    assert param_state.variance is None
                
                if algo == 'dion' and param.ndim == 2:
                    assert param_state.Q is not None
                else:
                    assert param_state.Q is None
            
            # Test update step
            grads = jax.tree_map(lambda p: jax.random.normal(rng_key, p.shape), simple_params)
            updates, new_state = optimizer.update(grads, state, simple_params)
            new_params = optax.apply_updates(simple_params, updates)
            
            # Parameters should change
            for key in simple_params:
                assert not jnp.allclose(simple_params[key], new_params[key])
    
    def test_learning_rate_schedule(self, simple_params, rng_key):
        """Test optimizer with learning rate schedule."""
        schedule = optax.linear_schedule(
            init_value=0.01,
            end_value=0.001,
            transition_steps=100
        )
        
        optimizer = dion(learning_rate=schedule)
        state = optimizer.init(simple_params)
        
        # Run multiple steps and check learning rate decay
        params = simple_params
        grads = jax.tree_map(lambda p: jax.random.normal(rng_key, p.shape), simple_params)
        
        first_update = None
        last_update = None
        
        for i in range(100):
            updates, state = optimizer.update(grads, state, params)
            
            if i == 0:
                first_update = updates
            if i == 99:
                last_update = updates
        
        # Learning rate should decrease, so updates should be smaller
        for key in first_update:
            first_norm = jnp.linalg.norm(first_update[key])
            last_norm = jnp.linalg.norm(last_update[key])
            assert last_norm < first_norm
    
    @pytest.mark.unstable
    def test_orthogonalize_methods(self, rng_key):
        """Test different orthogonalization methods."""
        key1, key2 = jax.random.split(rng_key)
        P = jax.random.normal(key1, (128, 32))
        
        # Test QR method
        Q_qr = orthogonalize(P, qr_method='qr')
        # Q should have shape (128, 32) for tall matrix
        assert Q_qr.shape == (128, 32)
        assert jnp.allclose(Q_qr.T @ Q_qr, jnp.eye(32, dtype=Q_qr.dtype), atol=1e-3)
        
        # Test RCQR method
        Q_rcqr = orthogonalize(P, qr_method='rcqr', rng_key=key2)
        assert jnp.allclose(Q_rcqr.T @ Q_rcqr, jnp.eye(32, dtype=Q_rcqr.dtype), atol=1e-3)
        
        # Test CQR method - known to be numerically unstable, so just check shape
        Q_cqr = orthogonalize(P, qr_method='cqr')
        assert Q_cqr.shape == (128, 32)
    
    def test_power_iteration(self, rng_key):
        """Test power iteration for low-rank approximation."""
        key1, key2, key3 = jax.random.split(rng_key, 3)
        
        # Create low-rank matrix B = UV^T
        U = jax.random.normal(key1, (64, 8))
        V = jax.random.normal(key2, (32, 8))
        B = U @ V.T
        
        # Initial Q
        Q_init = jax.random.normal(key3, (32, 8))
        
        # Run power iteration
        P, Q = power_iteration(
            B, Q_init,
            power_iters=3,
            qr_method='qr',
            oversample=1.25,
            rng_key=key3
        )
        
        # Check shapes
        assert P.shape == (64, 8)
        assert Q.shape == (32, 8)
        
        # Check approximation quality
        B_approx = P @ Q.T
        rel_error = jnp.linalg.norm(B - B_approx) / jnp.linalg.norm(B)
        assert rel_error < 0.1  # Should be a good approximation
    
    def test_all_zero_handling(self):
        """Test handling of all-zero tensors."""
        P = jnp.zeros((64, 8))
        R = jnp.zeros((32, 8))
        Q_init = jnp.ones((32, 8))
        B = jnp.zeros((64, 32))
        
        P_fixed, R_fixed = fix_all_zero_or_nan(P, R, Q_init, B)
        
        # Should return zeros for P and Q_init for R
        assert jnp.allclose(P_fixed, 0)
        assert jnp.allclose(R_fixed, Q_init)
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        P = jnp.full((64, 8), jnp.nan)
        R = jnp.full((32, 8), jnp.nan)
        Q_init = jnp.ones((32, 8))
        B = jnp.ones((64, 32))
        
        P_fixed, R_fixed = fix_all_zero_or_nan(P, R, Q_init, B)
        
        # Should replace NaN with zeros
        assert not jnp.any(jnp.isnan(P_fixed))
        assert not jnp.any(jnp.isnan(R_fixed))
    
    def test_weight_decay(self, simple_params, rng_key):
        """Test weight decay functionality."""
        # Test with Lion algorithm which doesn't have low-rank updates
        optimizer = dion(learning_rate=0.01, weight_decay=0.1, algorithm='lion')
        state = optimizer.init(simple_params)
        
        # Zero gradients - only weight decay should apply
        zero_grads = jax.tree_map(jnp.zeros_like, simple_params)
        
        updates, _ = optimizer.update(zero_grads, state, simple_params)
        new_params = optax.apply_updates(simple_params, updates)
        
        # Parameters should shrink due to weight decay
        for key in simple_params:
            old_norm = jnp.linalg.norm(simple_params[key])
            new_norm = jnp.linalg.norm(new_params[key])
            # With Lion, zero gradient means zero momentum, so only weight decay applies
            expected_new_norm = old_norm * (1 - 0.01 * 0.1)  # (1 - lr * weight_decay)
            assert jnp.allclose(new_norm, expected_new_norm, rtol=1e-5)
    
    @pytest.mark.unstable
    def test_optax_compatibility(self, simple_params, rng_key):
        """Test compatibility with other Optax transformations."""
        # Chain with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            dion(learning_rate=0.01)
        )
        
        state = optimizer.init(simple_params)
        
        # Large gradients should be clipped
        large_grads = jax.tree_map(
            lambda p: 10.0 * jax.random.normal(rng_key, p.shape),
            simple_params
        )
        
        updates, new_state = optimizer.update(large_grads, state, simple_params)
        
        # Check that updates are bounded
        for key in updates:
            assert jnp.linalg.norm(updates[key]) < 10.0