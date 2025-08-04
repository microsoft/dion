"""Tests comparing different Muon optimizer implementations."""

import pytest
import torch
import torch.nn as nn
from .base_comparison import BaseOptimizerComparison

# Try to import Muon implementations
try:
    from optimizers.muon_reference import Muon as MuonReference
    HAS_MUON_REFERENCE = True
except ImportError:
    HAS_MUON_REFERENCE = False
    MuonReference = None

try:
    from optimizers.muon import Muon as MuonOptimized  
    HAS_MUON_OPTIMIZED = True
except ImportError:
    HAS_MUON_OPTIMIZED = False
    MuonOptimized = None


@pytest.mark.skipif(not HAS_MUON_REFERENCE or not HAS_MUON_OPTIMIZED, 
                    reason="Muon implementations require optional dependencies")
class TestMuonImplementations(BaseOptimizerComparison):
    """Compare different Muon optimizer implementations for consistency."""
    
    def test_muon_optimized_vs_reference(self, device):
        """Compare MuonOptimized with MuonReference"""
        torch.manual_seed(42)
        
        # Create two identical models
        model_ref = self.create_simple_model(device)
        model_opt = self.create_simple_model(device)
        model_opt.load_state_dict(model_ref.state_dict())
        
        # Create optimizers
        lr = 0.02
        params_ref = list(model_ref.parameters())
        params_opt = list(model_opt.parameters())
        
        # MuonReference uses slightly different defaults
        opt_ref = MuonReference(
            params_ref, lr=lr, momentum=0.95, 
            backend='newton', backend_steps=5
        )
        opt_opt = MuonOptimized(
            params_opt, lr=lr, momentum=0.95,
            newton_schulz_steps=5
        )
        
        # Run multiple steps
        for step in range(3):
            # Generate same gradients
            self.generate_gradients(model_ref, device, seed=step)
            self.generate_gradients(model_opt, device, seed=step)
            
            # Take optimizer steps
            opt_ref.step()
            opt_opt.step()
            
            # Compare model states
            state_ref = self.get_model_state(model_ref)
            state_opt = self.get_model_state(model_opt)
            
            # Muon implementations might have larger differences due to different backends
            assert self.compare_model_states(state_ref, state_opt, rtol=1e-3, atol=1e-4), \
                f"Models diverged at step {step}"
            
            # Zero gradients
            opt_ref.zero_grad()
            opt_opt.zero_grad()
    
    def test_muon_newton_schulz_iterations(self, device):
        """Test that different Newton-Schulz iteration counts work correctly"""
        torch.manual_seed(42)
        
        iteration_counts = [1, 3, 5, 10]
        
        for n_steps in iteration_counts:
            # Create models
            model_ref = nn.Linear(32, 16, bias=False).to(device)
            model_opt = nn.Linear(32, 16, bias=False).to(device)
            model_opt.load_state_dict(model_ref.state_dict())
            
            # Create optimizers
            opt_ref = MuonReference(
                list(model_ref.parameters()), 
                lr=0.01, 
                backend='newton',
                backend_steps=n_steps
            )
            opt_opt = MuonOptimized(
                list(model_opt.parameters()),
                lr=0.01,
                newton_schulz_steps=n_steps
            )
            
            # Generate gradient
            grad = torch.randn(16, 32, device=device) * 0.01
            model_ref.weight.grad = grad.clone()
            model_opt.weight.grad = grad.clone()
            
            # Step
            opt_ref.step()
            opt_opt.step()
            
            # Should produce similar results
            assert torch.allclose(model_ref.weight, model_opt.weight, rtol=1e-3, atol=1e-4), \
                f"Divergence with {n_steps} Newton-Schulz iterations"
    
    def test_muon_momentum_consistency(self, device):
        """Test momentum handling across Muon implementations"""
        torch.manual_seed(42)
        
        # Test different momentum values
        momentum_values = [0.0, 0.5, 0.9, 0.95, 0.99]
        
        for momentum in momentum_values:
            # Create parameters
            param_ref = torch.randn(32, 16, device=device, requires_grad=True)
            param_opt = param_ref.clone().detach().requires_grad_(True)
            
            # Create optimizers
            opt_ref = MuonReference([param_ref], lr=0.01, momentum=momentum)
            opt_opt = MuonOptimized([param_opt], lr=0.01, momentum=momentum)
            
            # Apply same gradient multiple times
            grad = torch.randn_like(param_ref) * 0.01
            
            for _ in range(5):
                param_ref.grad = grad.clone()
                param_opt.grad = grad.clone()
                
                opt_ref.step()
                opt_opt.step()
                
                # Parameters should match
                assert torch.allclose(param_ref, param_opt, rtol=1e-3, atol=1e-4), \
                    f"Momentum {momentum} produces different results"
    
    def test_muon_adaptive_vs_fixed_lr(self, device):
        """Test adaptive learning rate feature if supported"""
        torch.manual_seed(42)
        
        # Create models
        model_ref = nn.Linear(32, 16, bias=False).to(device)
        model_opt = nn.Linear(32, 16, bias=False).to(device)
        model_opt.load_state_dict(model_ref.state_dict())
        
        # Check if adaptive LR is supported
        try:
            opt_ref = MuonReference(
                list(model_ref.parameters()),
                lr=0.01,
                adaptive_lr=True
            )
            opt_opt = MuonOptimized(
                list(model_opt.parameters()),
                lr=0.01,
                adaptive=True
            )
        except (TypeError, ValueError):
            # Adaptive LR not supported
            pytest.skip("Adaptive learning rate not supported")
        
        # Run steps
        for step in range(5):
            grad = torch.randn(16, 32, device=device) * 0.01
            model_ref.weight.grad = grad.clone()
            model_opt.weight.grad = grad.clone()
            
            opt_ref.step()
            opt_opt.step()
            
            # Should produce similar results
            assert torch.allclose(model_ref.weight, model_opt.weight, rtol=1e-3, atol=1e-4)
    
    def test_muon_with_weight_decay(self, device):
        """Test weight decay handling in Muon optimizers"""
        torch.manual_seed(42)
        
        # Large weights to make weight decay visible
        param_ref = torch.randn(16, 16, device=device, requires_grad=True) * 10
        param_opt = param_ref.clone().detach().requires_grad_(True)
        
        weight_decay = 0.1
        
        # Check if weight decay is supported
        try:
            opt_ref = MuonReference([param_ref], lr=0.01, weight_decay=weight_decay)
            opt_opt = MuonOptimized([param_opt], lr=0.01, weight_decay=weight_decay)
        except (TypeError, ValueError):
            # Weight decay not supported
            pytest.skip("Weight decay not supported in Muon")
        
        # Small gradient
        grad = torch.randn_like(param_ref) * 0.001
        param_ref.grad = grad.clone()
        param_opt.grad = grad.clone()
        
        # Step
        opt_ref.step()
        opt_opt.step()
        
        # Parameters should match and show weight decay effect
        assert torch.allclose(param_ref, param_opt, rtol=1e-4, atol=1e-5)
        
        # Check that weight decay was applied
        original_norm = torch.randn(16, 16, device=device).mul_(10).norm().item()
        assert param_ref.norm().item() < original_norm * 0.99
    
    def test_muon_mixed_parameter_groups(self, device):
        """Test Muon with mixed parameter groups"""
        torch.manual_seed(42)
        
        # Create models
        model_ref = self.create_mixed_model(device)
        model_opt = self.create_mixed_model(device)
        model_opt.load_state_dict(model_ref.state_dict())
        
        # Build parameter groups - Muon might only support matrix params
        def build_muon_groups(model):
            matrix_params = []
            for name, param in model.named_parameters():
                if param.ndim == 2 and 'embedding' not in name:
                    matrix_params.append(param)
            return [{"params": matrix_params}]
        
        groups_ref = build_muon_groups(model_ref)
        groups_opt = build_muon_groups(model_opt)
        
        # Create optimizers
        opt_ref = MuonReference(groups_ref, lr=0.01)
        opt_opt = MuonOptimized(groups_opt, lr=0.01)
        
        # Run steps
        for step in range(3):
            self.generate_gradients(model_ref, device, seed=step)
            self.generate_gradients(model_opt, device, seed=step)
            
            opt_ref.step()
            opt_opt.step()
            
            # Compare only the parameters that were optimized
            for (name_ref, param_ref), (name_opt, param_opt) in zip(
                model_ref.named_parameters(), model_opt.named_parameters()
            ):
                if param_ref.ndim == 2 and 'embedding' not in name_ref:
                    assert torch.allclose(param_ref, param_opt, rtol=1e-3, atol=1e-4), \
                        f"Parameter {name_ref} diverged"
            
            opt_ref.zero_grad()
            opt_opt.zero_grad()