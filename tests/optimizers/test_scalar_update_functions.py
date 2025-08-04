"""Direct tests for scalar optimizer update functions."""

import pytest
import torch
from optimizers.scalar_opts import adamw_update, lion_update


class TestScalarUpdateFunctions:
    """Test the individual update functions directly."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_adamw_update_function(self, device):
        """Test adamw_update function directly"""
        torch.manual_seed(42)
        
        # Create tensors
        shape = (32, 16)
        X = torch.randn(shape, device=device)
        G = torch.randn(shape, device=device) * 0.01
        M = torch.zeros(shape, device=device)
        V = torch.zeros(shape, device=device)
        
        # Parameters
        lr = torch.tensor(0.001)
        beta1 = torch.tensor(0.9)
        beta2 = torch.tensor(0.999)
        weight_decay = torch.tensor(0.01)
        epsilon = torch.tensor(1e-8)
        step = torch.tensor(1)
        
        # Store original for comparison
        X_orig = X.clone()
        
        # Call update function
        try:
            # The function might be compiled, which could fail in some environments
            adamw_update(X, G, M, V, lr, beta1, beta2, weight_decay, epsilon, step)
            
            # Check that parameters were updated
            assert not torch.allclose(X, X_orig), "Parameters were not updated"
            
            # Check momentum was updated
            assert not torch.allclose(M, torch.zeros_like(M)), "Momentum was not updated"
            
            # Check variance was updated
            assert not torch.allclose(V, torch.zeros_like(V)), "Variance was not updated"
            
        except Exception as e:
            # If torch.compile fails, that's okay for testing
            if "torch.compile" in str(e) or "dynamo" in str(e):
                pytest.skip("torch.compile not available in this environment")
            else:
                raise
    
    def test_lion_update_function(self, device):
        """Test lion_update function directly"""
        torch.manual_seed(42)
        
        # Create tensors
        shape = (32, 16)
        X = torch.randn(shape, device=device)
        G = torch.randn(shape, device=device) * 0.01
        M = torch.zeros(shape, device=device)
        
        # Parameters
        lr = torch.tensor(0.001)
        beta = torch.tensor(0.9)
        weight_decay = torch.tensor(0.01)
        
        # Store original for comparison
        X_orig = X.clone()
        
        # Call update function
        try:
            lion_update(X, G, M, lr, beta, weight_decay)
            
            # Check that parameters were updated
            assert not torch.allclose(X, X_orig), "Parameters were not updated"
            
            # Check momentum was updated
            assert not torch.allclose(M, torch.zeros_like(M)), "Momentum was not updated"
            
        except Exception as e:
            # If torch.compile fails, that's okay for testing
            if "torch.compile" in str(e) or "dynamo" in str(e):
                pytest.skip("torch.compile not available in this environment")
            else:
                raise
    
    def test_update_functions_with_weight_decay(self, device):
        """Test that weight decay is applied correctly"""
        torch.manual_seed(42)
        
        # Large weights to see weight decay effect
        X_adamw = torch.ones(10, 10, device=device) * 10.0
        X_lion = X_adamw.clone()
        
        # Zero gradient to isolate weight decay
        G = torch.zeros_like(X_adamw)
        
        # AdamW test
        M_adamw = torch.zeros_like(X_adamw)
        V_adamw = torch.zeros_like(X_adamw)
        
        try:
            adamw_update(
                X_adamw, G, M_adamw, V_adamw,
                lr=torch.tensor(0.1),
                beta1=torch.tensor(0.9),
                beta2=torch.tensor(0.999),
                weight_decay=torch.tensor(0.1),
                epsilon=torch.tensor(1e-8),
                step=torch.tensor(1)
            )
            
            # Weight should decrease due to decay
            assert X_adamw.mean() < 10.0, "AdamW weight decay not applied"
            
        except Exception as e:
            if "torch.compile" in str(e) or "dynamo" in str(e):
                pytest.skip("torch.compile not available")
            else:
                raise
        
        # Lion test
        M_lion = torch.zeros_like(X_lion)
        
        try:
            lion_update(
                X_lion, G, M_lion,
                lr=torch.tensor(0.1),
                beta=torch.tensor(0.9),
                weight_decay=torch.tensor(0.1)
            )
            
            # Weight should decrease due to decay
            assert X_lion.mean() < 10.0, "Lion weight decay not applied"
            
        except Exception as e:
            if "torch.compile" in str(e) or "dynamo" in str(e):
                pytest.skip("torch.compile not available")
            else:
                raise