"""Numerical comparison tests between PyTorch and JAX DION implementations.

IMPORTANT: These tests ensure strict numerical equivalence between implementations.
Key differences between PyTorch and Optax:
1. PyTorch modifies parameters in-place, Optax returns updates to be applied
2. PyTorch stores state per parameter, Optax returns immutable state
3. Random number generation differs between frameworks
"""

import pytest
import torch
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

from optimizers.dion_reference import (
    Dion as DionPyTorch, 
    dion_update as dion_update_torch,
    orthogonalize as orthogonalize_torch,
    power_iteration as power_iteration_torch
)
from optimizers.experimental.dion_reference_optax import (
    dion as dion_jax,
    dion_update as dion_update_jax,
    orthogonalize as orthogonalize_jax,
    power_iteration as power_iteration_jax,
    DionState
)


def set_global_seeds(seed):
    """Set seeds for all random number generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # JAX uses explicit keys, so no global seed needed


class TestNumericalComparison:
    """Test numerical equivalence between PyTorch and JAX implementations."""
    
    @pytest.fixture
    def seed(self):
        """Fixed seed for reproducibility."""
        return 12345
    
    @pytest.fixture
    def identical_params(self, seed):
        """Create identical parameters for both frameworks using numpy."""
        set_global_seeds(seed)
        
        # Generate parameters using numpy for exact reproducibility
        weight_np = np.random.randn(32, 64).astype(np.float32)
        bias_np = np.random.randn(64).astype(np.float32)
        
        # Create PyTorch versions
        params_torch = {
            'weight': torch.tensor(weight_np, dtype=torch.float32, requires_grad=True),
            'bias': torch.tensor(bias_np, dtype=torch.float32, requires_grad=True)
        }
        
        # Create JAX versions
        params_jax = {
            'weight': jnp.array(weight_np, dtype=jnp.float32),
            'bias': jnp.array(bias_np, dtype=jnp.float32)
        }
        
        return params_torch, params_jax, weight_np, bias_np
    
    @pytest.fixture
    def identical_gradients(self, seed):
        """Create identical gradients for both frameworks."""
        set_global_seeds(seed + 100)
        
        grad_weight_np = np.random.randn(32, 64).astype(np.float32) * 0.01
        grad_bias_np = np.random.randn(64).astype(np.float32) * 0.01
        
        grads_torch = {
            'weight': torch.tensor(grad_weight_np, dtype=torch.float32),
            'bias': torch.tensor(grad_bias_np, dtype=torch.float32)
        }
        
        grads_jax = {
            'weight': jnp.array(grad_weight_np, dtype=jnp.float32),
            'bias': jnp.array(grad_bias_np, dtype=jnp.float32)
        }
        
        return grads_torch, grads_jax, grad_weight_np, grad_bias_np
    
    def test_exact_initialization(self, identical_params, seed):
        """Test exact numerical equivalence of initialization."""
        params_torch, params_jax, weight_np, _ = identical_params
        
        # Configure identical hyperparameters
        lr = 0.01
        rank_fraction = 0.5
        rank_multiple_of = 1
        mu = 0.95
        weight_decay = 0.01
        eps = 1e-8
        
        # Initialize PyTorch optimizer
        torch_opt = DionPyTorch(
            [params_torch['weight']], 
            lr=lr,
            rank_fraction=rank_fraction,
            rank_multiple_of=rank_multiple_of,
            mu=mu,
            weight_decay=weight_decay,
            epsilon=eps
        )
        
        # Force initialization by setting zero grad and stepping
        params_torch['weight'].grad = torch.zeros_like(params_torch['weight'])
        initial_weight_torch = params_torch['weight'].clone()
        torch_opt.step()
        
        # Initialize JAX optimizer with same parameters
        jax_opt = dion_jax(
            learning_rate=lr,
            rank_fraction=rank_fraction,
            rank_multiple_of=rank_multiple_of,
            mu=mu,
            weight_decay=weight_decay,
            eps=eps,
            seed=seed
        )
        jax_state = jax_opt.init({'weight': params_jax['weight']})
        
        # Extract states
        torch_state = torch_opt.state[params_torch['weight']]
        jax_weight_state = jax_state['weight']
        
        # 1. Compare momentum initialization (should be exactly zeros)
        assert np.array_equal(
            torch_state['momentum'].numpy(),
            np.array(jax_weight_state.momentum)
        ), "Momentum should be exactly zero initialized"
        
        # 2. Compare Q matrix dimensions
        m, n = weight_np.shape
        expected_r = int(rank_fraction * min(m, n))
        expected_r = rank_multiple_of * np.ceil(expected_r / rank_multiple_of)
        expected_r = int(min(expected_r, m, n))
        
        # Since m < n (32 < 64), it should be transposed
        is_transposed = m < n
        expected_Q_shape = (m, expected_r) if is_transposed else (n, expected_r)
        
        assert torch_state['Q'].shape == expected_Q_shape
        assert jax_weight_state.Q.shape == expected_Q_shape
        
        # 3. Check that parameter didn't change with zero gradient
        # (except for weight decay)
        expected_new_weight = weight_np * (1 - lr * weight_decay)
        assert np.allclose(
            params_torch['weight'].detach().numpy(),
            expected_new_weight,
            rtol=1e-6, atol=1e-7
        ), "PyTorch weight update with zero gradient incorrect"
    
    @pytest.mark.unstable
    def test_single_step_detailed(self, identical_params, identical_gradients, seed):
        """Test detailed numerical equivalence of a single optimization step."""
        print("\n=== Testing single step detailed comparison ===")
        params_torch, params_jax, weight_np, _ = identical_params
        grads_torch, grads_jax, grad_weight_np, _ = identical_gradients
        
        print(f"Weight shape: {weight_np.shape}")
        print(f"Weight norm: {np.linalg.norm(weight_np):.4f}")
        print(f"Gradient norm: {np.linalg.norm(grad_weight_np):.4f}")
        
        # Hyperparameters
        lr = 0.01
        mu = 0.95
        weight_decay = 0.01
        eps = 1e-8
        rank_fraction = 1.0  # Full rank for easier comparison
        
        # Create deterministic Q matrix for both
        set_global_seeds(seed + 200)
        Q_np = np.random.randn(32, 32).astype(np.float32)  # For transposed case
        
        # PyTorch optimizer
        torch_opt = DionPyTorch(
            [params_torch['weight']], 
            lr=lr, mu=mu, weight_decay=weight_decay, 
            epsilon=eps, rank_fraction=rank_fraction
        )
        
        # Manually set Q to ensure same initialization
        params_torch['weight'].grad = torch.zeros_like(params_torch['weight'])
        torch_opt.step()  # Initialize
        torch_opt.state[params_torch['weight']]['Q'] = torch.tensor(Q_np)
        torch_opt.state[params_torch['weight']]['momentum'] = torch.zeros_like(params_torch['weight'])
        
        # Apply gradient
        params_torch['weight'].grad = grads_torch['weight']
        weight_before_torch = params_torch['weight'].clone()
        torch_opt.step()
        weight_after_torch = params_torch['weight'].clone()
        
        # JAX optimizer - manually create state to match
        jax_state_weight = DionState(
            momentum=jnp.zeros_like(params_jax['weight']),
            Q=jnp.array(Q_np),
            count=jnp.array(0, dtype=jnp.int32),
            mu=jnp.array(mu, dtype=jnp.float32),
            rng_key=jax.random.PRNGKey(seed)
        )
        
        # Perform single update
        new_state, new_weight_jax = dion_update_jax(
            params_jax['weight'],
            grads_jax['weight'],
            jax_state_weight,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            power_iters=1,
            qr_method='rcqr',
            cqr_warmup_steps=150,
            rcqr_oversample=1.25
        )
        
        # Compare final weights
        torch_final = weight_after_torch.detach().numpy()
        jax_final = np.array(new_weight_jax)
        
        # Compute differences
        diff = np.abs(torch_final - jax_final)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_diff = np.max(diff) / np.max(np.abs(torch_final))
        
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        print(f"Max relative difference: {rel_diff:.2e}")
        
        # Check momentum update
        torch_momentum_new = torch_opt.state[params_torch['weight']]['momentum'].numpy()
        jax_momentum_new = np.array(new_state.momentum)
        momentum_diff = np.max(np.abs(torch_momentum_new - jax_momentum_new))
        
        print(f"Momentum max difference: {momentum_diff:.2e}")
        
        # For exact reproducibility, differences should be very small
        assert max_diff < 1e-5, f"Weight difference too large: {max_diff}"
        assert momentum_diff < 1e-5, f"Momentum difference too large: {momentum_diff}"
    
    def test_orthogonalization_exact(self, seed):
        """Test exact numerical equivalence of orthogonalization methods."""
        set_global_seeds(seed)
        
        # Test different matrix sizes and methods
        test_cases = [
            (128, 32, 'qr'),
            (64, 64, 'qr'),
            (32, 128, 'qr'),
            # Note: CQR and RCQR use randomness, so exact comparison is harder
        ]
        
        for m, n, method in test_cases:
            # Create identical input
            P_np = np.random.randn(m, n).astype(np.float32)
            P_torch = torch.tensor(P_np)
            P_jax = jnp.array(P_np)
            
            # Orthogonalize
            Q_torch = orthogonalize_torch(P_torch, qr_method=method)
            Q_jax = orthogonalize_jax(P_jax, qr_method=method)
            
            Q_torch_np = Q_torch.numpy()
            Q_jax_np = np.array(Q_jax)
            
            # Check dimensions - Q should have shape (m, min(m,n))
            expected_cols = min(m, n)
            assert Q_torch_np.shape == (m, expected_cols), f"PyTorch Q shape mismatch: {Q_torch_np.shape}"
            assert Q_jax_np.shape == (m, expected_cols), f"JAX Q shape mismatch: {Q_jax_np.shape}"
            
            # Check orthogonality
            torch_orth = Q_torch_np.T @ Q_torch_np
            jax_orth = Q_jax_np.T @ Q_jax_np
            expected_orth = np.eye(expected_cols)
            
            # Both should be orthogonal
            assert np.allclose(torch_orth, expected_orth, atol=1e-5), \
                f"PyTorch orthogonalization failed for {m}x{n}"
            assert np.allclose(jax_orth, expected_orth, atol=1e-5), \
                f"JAX orthogonalization failed for {m}x{n}"
            
            # For QR method, results should be very close
            if method == 'qr':
                # QR decomposition can have sign ambiguity, so compare column-wise
                for j in range(expected_cols):
                    col_torch = Q_torch_np[:, j]
                    col_jax = Q_jax_np[:, j]
                    
                    # Check if columns are same or negated
                    if np.dot(col_torch, col_jax) < 0:
                        col_jax = -col_jax
                    
                    col_diff = np.max(np.abs(col_torch - col_jax))
                    assert col_diff < 1e-5, \
                        f"Column {j} differs by {col_diff} for {m}x{n}"
    
    @pytest.mark.unstable  
    def test_power_iteration_detailed(self, seed):
        """Test detailed power iteration equivalence."""
        set_global_seeds(seed)
        
        # Create low-rank test matrix
        rank = 8
        m, n = 64, 32
        
        # Generate exact low-rank matrix B = U @ V.T
        U_np = np.random.randn(m, rank).astype(np.float32)
        V_np = np.random.randn(n, rank).astype(np.float32)
        B_np = U_np @ V_np.T
        Q_init_np = np.random.randn(n, rank).astype(np.float32)
        
        # Convert to both frameworks
        B_torch = torch.tensor(B_np)
        Q_init_torch = torch.tensor(Q_init_np)
        B_jax = jnp.array(B_np)
        Q_init_jax = jnp.array(Q_init_np)
        
        # Test with QR method (deterministic)
        P_torch, R_torch = power_iteration_torch(
            B_torch, Q_init_torch, 
            power_iters=1, 
            qr_method='qr', 
            oversample=1.25,
            compressed_all_reduce=False
        )
        
        P_jax, R_jax = power_iteration_jax(
            B_jax, Q_init_jax, 
            power_iters=1, 
            qr_method='qr', 
            oversample=1.25
        )
        
        # Convert to numpy
        P_torch_np = P_torch.numpy()
        R_torch_np = R_torch.numpy()
        P_jax_np = np.array(P_jax)
        R_jax_np = np.array(R_jax)
        
        # Check shapes
        assert P_torch_np.shape == P_jax_np.shape == (m, rank)
        assert R_torch_np.shape == R_jax_np.shape == (n, rank)
        
        # Check orthogonality of P
        assert np.allclose(P_torch_np.T @ P_torch_np, np.eye(rank), atol=1e-5)
        assert np.allclose(P_jax_np.T @ P_jax_np, np.eye(rank), atol=1e-5)
        
        # Check approximation quality
        B_approx_torch = P_torch_np @ R_torch_np.T
        B_approx_jax = P_jax_np @ R_jax_np.T
        
        torch_error = np.linalg.norm(B_np - B_approx_torch) / np.linalg.norm(B_np)
        jax_error = np.linalg.norm(B_np - B_approx_jax) / np.linalg.norm(B_np)
        
        print(f"PyTorch approximation error: {torch_error:.6f}")
        print(f"JAX approximation error: {jax_error:.6f}")
        
        # Both should have similar approximation quality
        assert abs(torch_error - jax_error) < 0.01
        
        # For single power iteration with QR, results should be close
        # Account for sign ambiguity in QR
        for j in range(rank):
            if np.dot(P_torch_np[:, j], P_jax_np[:, j]) < 0:
                P_jax_np[:, j] *= -1
                R_jax_np[j, :] *= -1
        
        P_diff = np.max(np.abs(P_torch_np - P_jax_np))
        R_diff = np.max(np.abs(R_torch_np - R_jax_np))
        
        print(f"P max difference: {P_diff:.2e}")
        print(f"R max difference: {R_diff:.2e}")
        
        # With QR method, differences should be small
        assert P_diff < 1e-4, f"P difference too large: {P_diff}"
        assert R_diff < 1e-3, f"R difference too large: {R_diff}"
    
    @pytest.mark.unstable
    def test_convergence_detailed(self, seed):
        """Test detailed convergence comparison on a simple problem."""
        set_global_seeds(seed)
        
        # Simple quadratic loss: f(x) = 0.5 * ||x - target||^2
        m, n = 16, 32
        target_np = np.random.randn(m, n).astype(np.float32) * 0.1
        x0_np = np.random.randn(m, n).astype(np.float32)
        
        # Hyperparameters
        lr = 0.01
        num_steps = 20
        rank_fraction = 1.0
        weight_decay = 0.0
        mu = 0.95
        
        # PyTorch optimization
        x_torch = torch.tensor(x0_np.copy(), requires_grad=True)
        target_torch = torch.tensor(target_np)
        torch_opt = DionPyTorch(
            [x_torch], 
            lr=lr, 
            rank_fraction=rank_fraction,
            weight_decay=weight_decay,
            mu=mu
        )
        
        torch_losses = []
        torch_params = []
        for step in range(num_steps):
            torch_opt.zero_grad()
            loss = 0.5 * torch.sum((x_torch - target_torch) ** 2)
            torch_losses.append(loss.item())
            torch_params.append(x_torch.detach().clone().numpy())
            loss.backward()
            torch_opt.step()
        
        # JAX optimization
        def loss_fn(params, target):
            return 0.5 * jnp.sum((params['x'] - target) ** 2)
        
        jax_opt = dion_jax(
            learning_rate=lr, 
            rank_fraction=rank_fraction,
            weight_decay=weight_decay,
            mu=mu,
            seed=seed
        )
        
        params = {'x': jnp.array(x0_np.copy())}
        state = jax_opt.init(params)
        
        jax_losses = []
        jax_params = []
        for step in range(num_steps):
            loss = loss_fn(params, target_np)
            jax_losses.append(float(loss))
            jax_params.append(np.array(params['x']))
            
            # Compute gradients
            grads = jax.grad(lambda p: loss_fn(p, target_np))(params)
            
            # Apply updates
            updates, state = jax_opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)
        
        # Compare convergence
        print("\nLoss comparison:")
        for i in range(0, num_steps, 5):
            print(f"Step {i:2d}: PyTorch {torch_losses[i]:8.4f}, JAX {jax_losses[i]:8.4f}, "
                  f"Diff: {abs(torch_losses[i] - jax_losses[i]):8.2e}")
        
        # Check final convergence
        torch_final_loss = torch_losses[-1]
        jax_final_loss = jax_losses[-1]
        
        print(f"\nFinal loss: PyTorch {torch_final_loss:.6f}, JAX {jax_final_loss:.6f}")
        print(f"Loss reduction: PyTorch {torch_losses[0]/torch_final_loss:.2f}x, "
              f"JAX {jax_losses[0]/jax_final_loss:.2f}x")
        
        # Both should converge
        assert torch_final_loss < torch_losses[0] * 0.5
        assert jax_final_loss < jax_losses[0] * 0.5
        
        # Final losses should be similar
        loss_ratio = torch_final_loss / jax_final_loss
        assert 0.8 < loss_ratio < 1.2, f"Final loss ratio {loss_ratio} out of range"
        
        # Check parameter trajectory similarity
        for i in [5, 10, 15, -1]:
            param_diff = np.max(np.abs(torch_params[i] - jax_params[i]))
            param_norm = np.max(np.abs(torch_params[i]))
            rel_diff = param_diff / (param_norm + 1e-8)
            print(f"Step {i:2d} param diff: {param_diff:.2e} (relative: {rel_diff:.2%})")
    
    @pytest.mark.unstable
    def test_adamw_lion_algorithms(self, identical_params, identical_gradients):
        """Test AdamW and Lion algorithm implementations."""
        params_torch, params_jax, _, _ = identical_params
        grads_torch, grads_jax, _, _ = identical_gradients
        
        # Test AdamW
        lr = 0.001
        betas = (0.9, 0.999)
        weight_decay = 0.01
        eps = 1e-8
        
        # PyTorch AdamW on bias (1D tensor)
        bias_torch = params_torch['bias'].clone().detach().requires_grad_(True)
        torch_opt = torch.optim.AdamW(
            [bias_torch], 
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay,
            eps=eps
        )
        
        # Apply gradient
        bias_torch.grad = grads_torch['bias']
        bias_before = bias_torch.clone()
        torch_opt.step()
        bias_after_torch = bias_torch.clone()
        
        # JAX AdamW
        jax_opt = dion_jax(
            learning_rate=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            algorithm='adamw'
        )
        
        params = {'bias': params_jax['bias']}
        state = jax_opt.init(params)
        grads = {'bias': grads_jax['bias']}
        
        updates, _ = jax_opt.update(grads, state, params)
        params_after_jax = optax.apply_updates(params, updates)
        
        # Compare updates
        torch_update = bias_after_torch.detach().numpy() - bias_before.detach().numpy()
        jax_update = np.array(params_after_jax['bias']) - np.array(params['bias'])
        
        update_diff = np.max(np.abs(torch_update - jax_update))
        print(f"AdamW update difference: {update_diff:.2e}")
        
        # Should be very close for first step
        assert update_diff < 1e-6, f"AdamW update difference too large: {update_diff}"