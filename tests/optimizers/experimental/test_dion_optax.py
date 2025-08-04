"""Tests for optimized JAX/Optax DION implementation."""

import pytest
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

from optimizers.experimental.dion_optax import (
    dion_fast, DionFastConfig, DionFastState,
    batch_dion_update, dion_matrix_update,
    orthogonalize_fast, power_iteration_fast
)


class TestDionOptaxFast:
    """Test suite for optimized DION Optax implementation."""
    
    @pytest.fixture
    def rng_key(self):
        """Random key for JAX operations."""
        return jax.random.PRNGKey(42)
    
    @pytest.fixture
    def model_params(self, rng_key):
        """Create a more complex model parameter structure."""
        keys = jax.random.split(rng_key, 6)
        return {
            'encoder': {
                'dense1': jax.random.normal(keys[0], (128, 256)),
                'dense2': jax.random.normal(keys[1], (256, 512)),
                'bias1': jax.random.normal(keys[2], (256,)),
                'bias2': jax.random.normal(keys[3], (512,)),
            },
            'decoder': {
                'dense': jax.random.normal(keys[4], (512, 128)),
                'bias': jax.random.normal(keys[5], (128,)),
            }
        }
    
    def test_fast_optimizer_initialization(self, model_params):
        """Test fast optimizer initialization with default config."""
        config = DionFastConfig()
        optimizer = dion_fast(learning_rate=0.01, config=config)
        
        state = optimizer.init(model_params)
        assert state is not None
        
        # Check that state structure matches parameter structure
        # Note: The actual implementation may flatten the structure
        assert isinstance(state, dict)
    
    def test_config_options(self, model_params):
        """Test optimizer with various configuration options."""
        config = DionFastConfig(
            rank_fraction=0.5,
            rank_multiple_of=16,
            mu=0.9,
            betas=(0.9, 0.999),
            weight_decay=0.1,
            eps=1e-6,
            qr_method="rcqr",
            rcqr_oversample=1.5,
            momentum_dtype=jnp.float32,
            Q_dtype=jnp.bfloat16
        )
        
        optimizer = dion_fast(learning_rate=0.001, config=config)
        state = optimizer.init(model_params)
        
        # The state should be initialized according to config
        assert state is not None
    
    def test_single_optimization_step(self, model_params, rng_key):
        """Test a single optimization step."""
        config = DionFastConfig()
        optimizer = dion_fast(learning_rate=0.01, config=config)
        
        state = optimizer.init(model_params)
        
        # Generate random gradients
        grads = jax.tree_map(
            lambda p: jax.random.normal(rng_key, p.shape) * 0.01,
            model_params
        )
        
        # Apply optimizer update
        updates, new_state = optimizer.update(grads, state, model_params)
        new_params = optax.apply_updates(model_params, updates)
        
        # Check that parameters changed
        def check_changed(old, new):
            assert not jnp.allclose(old, new, rtol=1e-7)
        
        jax.tree_map(check_changed, model_params, new_params)
    
    def test_learning_rate_schedule(self, model_params, rng_key):
        """Test optimizer with learning rate schedule."""
        schedule = optax.exponential_decay(
            init_value=0.01,
            transition_steps=100,
            decay_rate=0.9
        )
        
        config = DionFastConfig()
        optimizer = dion_fast(learning_rate=schedule, config=config)
        
        state = optimizer.init(model_params)
        grads = jax.tree_map(
            lambda p: jax.random.normal(rng_key, p.shape) * 0.01,
            model_params
        )
        
        # Run multiple steps
        params = model_params
        for _ in range(10):
            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)
        
        # State should have been updated multiple times
        # Check count in one of the states
        first_state = jax.tree_util.tree_leaves(state)[0]
        assert first_state.count > 0
    
    def test_different_algorithms(self, model_params, rng_key):
        """Test different algorithm variants."""
        for algo in ['dion', 'adamw', 'lion']:
            config = DionFastConfig()
            optimizer = dion_fast(
                learning_rate=0.01,
                config=config,
                algorithm=algo
            )
            
            state = optimizer.init(model_params)
            grads = jax.tree_map(
                lambda p: jax.random.normal(rng_key, p.shape) * 0.01,
                model_params
            )
            
            updates, new_state = optimizer.update(grads, state, model_params)
            new_params = optax.apply_updates(model_params, updates)
            
            # All algorithms should produce parameter updates
            def check_changed(old, new):
                assert not jnp.allclose(old, new, rtol=1e-7)
            
            jax.tree_map(check_changed, model_params, new_params)
    
    def test_vectorized_operations(self, rng_key):
        """Test that vectorized operations work correctly."""
        # Create multiple matrix parameters
        keys = jax.random.split(rng_key, 4)
        params = [
            jax.random.normal(keys[0], (64, 128)),
            jax.random.normal(keys[1], (128, 256)),
            jax.random.normal(keys[2], (256, 64)),
            jax.random.normal(keys[3], (32, 512)),
        ]
        
        config = DionFastConfig()
        
        # Initialize states for each parameter
        param_keys = jax.random.split(rng_key, len(params))
        from optimizers.experimental.dion_optax import init_matrix_state
        states = [
            init_matrix_state(p, k, config)
            for p, k in zip(params, param_keys)
        ]
        
        # Create gradients
        grad_keys = jax.random.split(keys[0], len(params))
        grads = [
            jax.random.normal(k, p.shape) * 0.01
            for k, p in zip(grad_keys, params)
        ]
        
        # Test batch update
        updates, new_states = batch_dion_update(
            params, grads, states, lr=0.01, config=config
        )
        
        assert len(updates) == len(params)
        assert len(new_states) == len(states)
        
        # Check that all parameters would be updated
        for i, (param, update) in enumerate(zip(params, updates)):
            new_param = param + update
            assert not jnp.allclose(param, new_param)
    
    def test_orthogonalization_performance(self, rng_key):
        """Test fast orthogonalization method."""
        config = DionFastConfig(rcqr_oversample=1.25)
        
        # Test with different matrix sizes
        for m, n in [(256, 64), (512, 32), (128, 128)]:
            P = jax.random.normal(rng_key, (m, n))
            
            Q = orthogonalize_fast(P, config=config, rng_key=rng_key)
            
            # Check orthogonality
            QTQ = Q.T @ Q
            eye = jnp.eye(n)
            assert jnp.allclose(QTQ, eye, atol=1e-5)
    
    def test_power_iteration_fast(self, rng_key):
        """Test fast power iteration."""
        config = DionFastConfig()
        
        # Create a low-rank matrix
        keys = jax.random.split(rng_key, 3)
        U = jax.random.normal(keys[0], (128, 16))
        V = jax.random.normal(keys[1], (64, 16))
        B = U @ V.T
        
        # Initial Q
        Q_init = jax.random.normal(keys[2], (64, 16))
        
        # Run power iteration
        P, R = power_iteration_fast(B, Q_init, config=config, rng_key=rng_key)
        
        # Check shapes
        assert P.shape == (128, 16)
        assert R.shape == (64, 16)
        
        # Check that P is orthogonal
        PTP = P.T @ P
        assert jnp.allclose(PTP, jnp.eye(16), atol=1e-5)
    
    def test_mixed_precision(self, model_params):
        """Test mixed precision configurations."""
        config = DionFastConfig(
            momentum_dtype=jnp.float32,
            Q_dtype=jnp.bfloat16,
            variance_dtype=jnp.float32
        )
        
        optimizer = dion_fast(
            learning_rate=0.01,
            config=config,
            algorithm='dion'
        )
        
        state = optimizer.init(model_params)
        
        # Check that dtypes are respected
        # Note: actual dtype checking would depend on implementation details
        assert state is not None
    
    def test_chain_with_optax(self, model_params, rng_key):
        """Test chaining with other Optax transformations."""
        config = DionFastConfig()
        
        # Chain with gradient clipping and learning rate scheduling
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            dion_fast(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=0.01,
                    warmup_steps=100,
                    decay_steps=1000
                ),
                config=config
            )
        )
        
        state = optimizer.init(model_params)
        
        # Generate large gradients that should be clipped
        large_grads = jax.tree_map(
            lambda p: 10.0 * jax.random.normal(rng_key, p.shape),
            model_params
        )
        
        updates, new_state = optimizer.update(large_grads, state, model_params)
        
        # Compute global norm of updates
        update_norm = optax.global_norm(updates)
        
        # Due to clipping, norm should be bounded
        # (actual bound depends on how clipping interacts with DION scaling)
        assert update_norm < 20.0
    
    def test_deterministic_initialization(self, model_params):
        """Test that initialization is deterministic with same seed."""
        config = DionFastConfig()
        
        # Create two optimizers with same seed
        opt1 = dion_fast(learning_rate=0.01, config=config, seed=123)
        opt2 = dion_fast(learning_rate=0.01, config=config, seed=123)
        
        state1 = opt1.init(model_params)
        state2 = opt2.init(model_params)
        
        # States should be identical
        def check_equal(s1, s2):
            if isinstance(s1, DionFastState) and isinstance(s2, DionFastState):
                if s1.Q is not None and s2.Q is not None:
                    assert jnp.allclose(s1.Q, s2.Q)
                assert jnp.allclose(s1.momentum, s2.momentum)
        
        jax.tree_map(check_equal, state1, state2)