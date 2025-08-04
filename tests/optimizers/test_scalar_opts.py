import pytest
import torch
import numpy as np
from typing import List
import math

from optimizers.scalar_opts import (
    adamw_update, lion_update,
    adamw_update_foreach, lion_update_foreach
)


class TestScalarOptimizers:
    """Test scalar optimizer implementations (Lion and AdamW)"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_adamw_basic_update(self, device):
        """Test basic AdamW update functionality"""
        torch.manual_seed(42)
        
        # Create test tensors
        X = torch.randn(32, 16, device=device)
        G = torch.randn_like(X) * 0.01
        M = torch.zeros_like(X)
        V = torch.zeros_like(X)
        
        # Hyperparameters
        lr = torch.tensor(0.001)
        beta1 = torch.tensor(0.9)
        beta2 = torch.tensor(0.999)
        weight_decay = torch.tensor(0.01)
        epsilon = 1e-8
        step = 1
        
        # Save original
        X_orig = X.clone()
        
        # Run update
        adamw_update(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
        
        # Check that parameters changed
        assert not torch.allclose(X, X_orig)
        
        # Check momentum was updated
        assert not torch.allclose(M, torch.zeros_like(M))
        
        # Check variance was updated
        assert not torch.allclose(V, torch.zeros_like(V))
    
    def test_adamw_momentum_accumulation(self, device):
        """Test AdamW momentum accumulation over multiple steps"""
        torch.manual_seed(42)
        
        X = torch.randn(16, 8, device=device)
        G = torch.ones_like(X) * 0.1  # Constant gradient
        M = torch.zeros_like(X)
        V = torch.zeros_like(X)
        
        lr = torch.tensor(0.001)
        beta1 = torch.tensor(0.9)
        beta2 = torch.tensor(0.999)
        weight_decay = torch.tensor(0.0)
        epsilon = 1e-8
        
        # Run multiple steps
        for step in range(1, 11):
            M_before = M.clone()
            adamw_update(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
            
            # Check momentum is accumulating towards gradient
            assert torch.norm(M - G) < torch.norm(M_before - G)
    
    def test_adamw_bias_correction(self, device):
        """Test AdamW bias correction in early steps"""
        torch.manual_seed(42)
        
        X = torch.randn(8, 8, device=device)
        G = torch.randn_like(X)
        
        # Test with and without bias correction
        results = []
        
        for step in [1, 10, 100]:
            X_test = X.clone()
            M = torch.zeros_like(X)
            V = torch.zeros_like(X)
            
            adamw_update(
                X_test, G, M, V,
                lr=torch.tensor(0.01),
                beta1=torch.tensor(0.9),
                beta2=torch.tensor(0.999),
                weight_decay=torch.tensor(0.0),
                step=step,
                epsilon=1e-8
            )
            
            update_magnitude = torch.norm(X - X_test).item()
            results.append((step, update_magnitude))
        
        # Due to bias correction, the effective learning rate changes with step
        # Step 1 has the most aggressive bias correction
        # We just check that all updates are different and reasonable
        assert results[0][1] > 0
        assert results[1][1] > 0
        assert results[2][1] > 0
        # Updates should stabilize as bias correction diminishes
        assert abs(results[1][1] - results[2][1]) < abs(results[0][1] - results[1][1])
    
    def test_adamw_weight_decay(self, device):
        """Test AdamW weight decay implementation"""
        torch.manual_seed(42)
        
        X = torch.randn(16, 16, device=device) * 10  # Large weights
        G = torch.zeros_like(X)  # Zero gradient to isolate weight decay
        M = torch.zeros_like(X)
        V = torch.ones_like(X)  # Non-zero to avoid division issues
        
        lr = torch.tensor(0.1)
        weight_decay = torch.tensor(0.01)
        
        X_before = X.clone()
        
        adamw_update(
            X, G, M, V, lr,
            beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999),
            weight_decay=weight_decay,
            step=1,
            epsilon=1e-8
        )
        
        # With zero gradient and ones variance, the main change should be weight decay
        # X_new ≈ X_old * (1 - lr * weight_decay)
        expected_decay_factor = 1 - lr.item() * weight_decay.item()
        actual_ratio = (torch.norm(X) / torch.norm(X_before)).item()
        
        assert abs(actual_ratio - expected_decay_factor) < 0.01
    
    def test_lion_basic_update(self, device):
        """Test basic Lion update functionality"""
        torch.manual_seed(42)
        
        X = torch.randn(32, 16, device=device)
        G = torch.randn_like(X) * 0.01
        M = torch.zeros_like(X)
        
        lr = torch.tensor(0.001)
        beta1 = torch.tensor(0.9)
        beta2 = torch.tensor(0.99)
        weight_decay = torch.tensor(0.01)
        
        X_orig = X.clone()
        
        # Run update
        lion_update(X, G, M, lr, beta1, beta2, weight_decay)
        
        # Check that parameters changed
        assert not torch.allclose(X, X_orig)
        
        # Check momentum was updated
        assert not torch.allclose(M, torch.zeros_like(M))
    
    def test_lion_sign_update(self, device):
        """Test Lion's sign-based update mechanism"""
        torch.manual_seed(42)
        
        X = torch.zeros(10, 10, device=device)
        M = torch.zeros_like(X)
        
        # Create gradient with known signs
        G = torch.ones_like(X)
        G[:5, :] = -1  # First half negative
        
        lr = torch.tensor(0.1)
        beta1 = torch.tensor(0.0)  # No momentum interpolation
        beta2 = torch.tensor(0.0)  # No momentum update
        weight_decay = torch.tensor(0.0)
        
        lion_update(X, G, M, lr, beta1, beta2, weight_decay)
        
        # Update should be exactly -lr * sign(G)
        expected = -lr * torch.sign(G)
        assert torch.allclose(X, expected)
    
    def test_lion_momentum_interpolation(self, device):
        """Test Lion's momentum interpolation for update direction"""
        torch.manual_seed(42)
        
        X = torch.zeros(8, 8, device=device)
        
        # Set up momentum and gradient with different directions
        M = torch.ones_like(X)
        G = -torch.ones_like(X)  # Opposite direction
        
        lr = torch.tensor(0.1)
        beta1 = torch.tensor(0.5)  # Equal weight
        beta2 = torch.tensor(0.0)  # Don't update momentum
        weight_decay = torch.tensor(0.0)
        
        lion_update(X, G, M, lr, beta1, beta2, weight_decay)
        
        # With beta1=0.5, interpolation should give zero, so sign=0
        # But sign(0) = 0 in PyTorch
        assert torch.allclose(X, torch.zeros_like(X))
    
    def test_scalar_opts_dtype_handling(self, device):
        """Test dtype handling in scalar optimizers"""
        dtypes = [torch.float32, torch.float64]
        
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        
        for dtype in dtypes:
            # Test AdamW
            X = torch.randn(8, 8, device=device, dtype=dtype)
            G = torch.randn_like(X) * 0.01
            M = torch.zeros_like(X)
            V = torch.zeros_like(X)
            
            adamw_update(
                X, G, M, V,
                lr=torch.tensor(0.001, dtype=dtype),
                beta1=torch.tensor(0.9, dtype=dtype),
                beta2=torch.tensor(0.999, dtype=dtype),
                weight_decay=torch.tensor(0.01, dtype=dtype),
                step=1,
                epsilon=1e-8
            )
            
            assert X.dtype == dtype
            assert M.dtype == dtype
            assert V.dtype == dtype
            
            # Test Lion
            X = torch.randn(8, 8, device=device, dtype=dtype)
            G = torch.randn_like(X) * 0.01
            M = torch.zeros_like(X)
            
            lion_update(
                X, G, M,
                lr=torch.tensor(0.001, dtype=dtype),
                beta1=torch.tensor(0.9, dtype=dtype),
                beta2=torch.tensor(0.99, dtype=dtype),
                weight_decay=torch.tensor(0.01, dtype=dtype)
            )
            
            assert X.dtype == dtype
            assert M.dtype == dtype
    
    def test_foreach_implementations(self, device):
        """Test foreach implementations match single tensor versions"""
        torch.manual_seed(42)
        
        batch_size = 5
        
        # Create batches of tensors
        X_single = [torch.randn(16, 8, device=device) for _ in range(batch_size)]
        X_foreach = [x.clone() for x in X_single]
        
        G = [torch.randn_like(x) * 0.01 for x in X_single]
        
        # AdamW test
        M_single = [torch.zeros_like(x) for x in X_single]
        M_foreach = [m.clone() for m in M_single]
        V_single = [torch.zeros_like(x) for x in X_single]
        V_foreach = [v.clone() for v in V_single]
        
        lr = torch.tensor(0.001)
        beta1 = torch.tensor(0.9)
        beta2 = torch.tensor(0.999)
        weight_decay = torch.tensor(0.01)
        step = 1
        epsilon = 1e-8
        
        # Run single tensor updates
        for i in range(batch_size):
            adamw_update(
                X_single[i], G[i], M_single[i], V_single[i],
                lr, beta1, beta2, weight_decay, step, epsilon
            )
        
        # Run foreach update
        adamw_update_foreach(
            X_foreach, G, M_foreach, V_foreach,
            lr, beta1, beta2, weight_decay, step, epsilon
        )
        
        # Compare results
        for i in range(batch_size):
            assert torch.allclose(X_single[i], X_foreach[i], atol=1e-6)
            assert torch.allclose(M_single[i], M_foreach[i], atol=1e-6)
            assert torch.allclose(V_single[i], V_foreach[i], atol=1e-6)
        
        # Lion test
        X_single = [torch.randn(16, 8, device=device) for _ in range(batch_size)]
        X_foreach = [x.clone() for x in X_single]
        M_single = [torch.zeros_like(x) for x in X_single]
        M_foreach = [m.clone() for m in M_single]
        
        # Run single tensor updates
        for i in range(batch_size):
            lion_update(
                X_single[i], G[i], M_single[i],
                lr, beta1, beta2, weight_decay
            )
        
        # Run foreach update
        lion_update_foreach(
            X_foreach, G, M_foreach,
            lr, beta1, beta2, weight_decay
        )
        
        # Compare results
        for i in range(batch_size):
            assert torch.allclose(X_single[i], X_foreach[i], atol=1e-6)
            assert torch.allclose(M_single[i], M_foreach[i], atol=1e-6)
    
    def test_zero_gradient_behavior(self, device):
        """Test behavior with zero gradients"""
        X = torch.randn(8, 8, device=device) * 10
        G = torch.zeros_like(X)
        
        # Test AdamW
        M = torch.zeros_like(X)
        V = torch.zeros_like(X)
        X_adamw = X.clone()
        
        adamw_update(
            X_adamw, G, M, V,
            lr=torch.tensor(0.1),
            beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999),
            weight_decay=torch.tensor(0.01),
            step=1,
            epsilon=1e-8
        )
        
        # Should only apply weight decay
        expected = X * (1 - 0.1 * 0.01)
        assert torch.allclose(X_adamw, expected, atol=1e-6)
        
        # Test Lion
        M = torch.zeros_like(X)
        X_lion = X.clone()
        
        lion_update(
            X_lion, G, M,
            lr=torch.tensor(0.1),
            beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.99),
            weight_decay=torch.tensor(0.01)
        )
        
        # Should only apply weight decay (sign of interpolation is 0)
        expected = X * (1 - 0.1 * 0.01)
        assert torch.allclose(X_lion, expected, atol=1e-6)
    
    def test_extreme_values(self, device):
        """Test handling of extreme values"""
        # Test with very large values
        X = torch.tensor([[1e30, -1e30]], device=device, dtype=torch.float32)
        G = torch.tensor([[1e20, -1e20]], device=device, dtype=torch.float32)
        M = torch.zeros_like(X)
        V = torch.zeros_like(X)
        
        # AdamW should handle this gracefully
        X_test = X.clone()
        adamw_update(
            X_test, G, M, V,
            lr=torch.tensor(1e-10),  # Very small LR
            beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.999),
            weight_decay=torch.tensor(0.0),
            step=1,
            epsilon=1e-8
        )
        
        assert torch.isfinite(X_test).all()
        
        # Lion should also handle this (sign operation normalizes)
        X_test = X.clone()
        M = torch.zeros_like(X)
        lion_update(
            X_test, G, M,
            lr=torch.tensor(1e-10),
            beta1=torch.tensor(0.9),
            beta2=torch.tensor(0.99),
            weight_decay=torch.tensor(0.0)
        )
        
        assert torch.isfinite(X_test).all()
    
    def test_gradient_accumulation_pattern(self, device):
        """Test gradient accumulation patterns in both optimizers"""
        torch.manual_seed(42)
        
        # Create cyclic gradient pattern
        X = torch.zeros(4, 4, device=device)
        gradients = [
            torch.ones_like(X),
            -torch.ones_like(X),
            torch.ones_like(X),
            -torch.ones_like(X),
        ]
        
        # Test AdamW
        M_adamw = torch.zeros_like(X)
        V_adamw = torch.zeros_like(X)
        X_adamw = X.clone()
        
        for step, G in enumerate(gradients, 1):
            adamw_update(
                X_adamw, G, M_adamw, V_adamw,
                lr=torch.tensor(0.01),
                beta1=torch.tensor(0.9),
                beta2=torch.tensor(0.999),
                weight_decay=torch.tensor(0.0),
                step=step,
                epsilon=1e-8
            )
        
        # Momentum should be close to zero after cycling
        assert torch.norm(M_adamw) < 0.5
        
        # Test Lion
        M_lion = torch.zeros_like(X)
        X_lion = X.clone()
        
        for G in gradients:
            lion_update(
                X_lion, G, M_lion,
                lr=torch.tensor(0.01),
                beta1=torch.tensor(0.9),
                beta2=torch.tensor(0.99),
                weight_decay=torch.tensor(0.0)
            )
        
        # Lion momentum should also be small after cycling
        assert torch.norm(M_lion) < 0.5