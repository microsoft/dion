"""Tests comparing different Dion optimizer implementations."""

import pytest
import torch
import torch.nn as nn
from .base_comparison import BaseOptimizerComparison

# Import optimizer variants
from optimizers.dion_reference import Dion as DionReference
from optimizers.dion_simple import Dion as DionSimple

# Try to import optimizers that require optional dependencies
try:
    from optimizers.dion import Dion as DionOptimized
    HAS_DION_OPTIMIZED = True
except ImportError:
    HAS_DION_OPTIMIZED = False
    DionOptimized = None


class TestDionImplementations(BaseOptimizerComparison):
    """Compare different Dion optimizer implementations for consistency."""
    
    def test_dion_simple_vs_reference(self, device):
        """Compare DionSimple with DionReference"""
        torch.manual_seed(42)
        
        # Create two identical models
        model_ref = self.create_simple_model(device)
        model_simple = self.create_simple_model(device)
        model_simple.load_state_dict(model_ref.state_dict())
        
        # Create optimizers with same settings
        lr = 0.01
        params_ref = list(model_ref.parameters())
        params_simple = list(model_simple.parameters())
        
        # DionSimple uses fixed rank, so we need to match it
        rank = 32  
        opt_ref = DionReference(params_ref, lr=lr, mu=0.95, weight_decay=0.01, 
                              rank_fraction=rank/64.0)
        opt_simple = DionSimple(params_simple, lr=lr, mu=0.95, weight_decay=0.01,
                              rank=rank)
        
        # Run multiple steps
        for step in range(3):
            # Generate same gradients
            self.generate_gradients(model_ref, device, seed=step)
            self.generate_gradients(model_simple, device, seed=step)
            
            # Take optimizer steps
            opt_ref.step()
            opt_simple.step()
            
            # Compare model states
            state_ref = self.get_model_state(model_ref)
            state_simple = self.get_model_state(model_simple)
            
            # DionSimple uses slightly different implementation
            assert self.compare_model_states(state_ref, state_simple, rtol=5e-2, atol=1e-3), \
                f"Models diverged at step {step}"
            
            # Zero gradients
            opt_ref.zero_grad()
            opt_simple.zero_grad()
    
    @pytest.mark.skipif(not HAS_DION_OPTIMIZED, reason="DionOptimized requires optional dependencies")
    def test_dion_optimized_vs_reference(self, device):
        """Compare DionOptimized with DionReference in single device mode"""
        torch.manual_seed(42)
        
        # Create two identical models
        model_ref = self.create_simple_model(device)
        model_opt = self.create_simple_model(device)
        model_opt.load_state_dict(model_ref.state_dict())
        
        # Create optimizers
        lr = 0.01
        params_ref = list(model_ref.parameters())
        params_opt = list(model_opt.parameters())
        
        opt_ref = DionReference(
            params_ref, lr=lr, mu=0.95, weight_decay=0.01,
            rank_fraction=0.25, power_iters=1
        )
        opt_opt = DionOptimized(
            params_opt, lr=lr, mu=0.95, weight_decay=0.01,
            rank_fraction=0.25, power_iters=1
        )
        
        # Run multiple steps
        for step in range(3):
            self.generate_gradients(model_ref, device)
            self.generate_gradients(model_opt, device)
            
            opt_ref.step()
            opt_opt.step()
            
            state_ref = self.get_model_state(model_ref)
            state_opt = self.get_model_state(model_opt)
            
            assert self.compare_model_states(state_ref, state_opt, rtol=1e-4, atol=1e-5), \
                f"Models diverged at step {step}"
            
            opt_ref.zero_grad()
            opt_opt.zero_grad()
    
    @pytest.mark.skipif(not HAS_DION_OPTIMIZED, reason="DionOptimized requires optional dependencies")
    def test_rank_fraction_consistency(self, device):
        """Test that different Dion implementations handle rank_fraction consistently"""
        torch.manual_seed(42)
        
        rank_fractions = [1.0, 0.5, 0.25, 0.125]
        
        for rf in rank_fractions:
            # Create model
            model = nn.Linear(64, 32, bias=False).to(device)
            param = list(model.parameters())[0]
            
            # Create optimizers
            opt_ref = DionReference([param], lr=0.01, rank_fraction=rf)
            opt_opt = DionOptimized([param], lr=0.01, rank_fraction=rf)
            
            # Generate gradient
            param.grad = torch.randn_like(param) * 0.01
            
            # Take step to initialize states
            opt_ref.step()
            opt_opt.step()
            
            # Check Q matrix dimensions
            Q_ref = opt_ref.state[param]["Q"]
            Q_opt = opt_opt.state[param]["Q"]
            
            expected_rank = int(rf * min(param.shape))
            assert Q_ref.shape[1] == expected_rank, f"Reference Q shape mismatch for rf={rf}"
            assert Q_opt.shape[1] == expected_rank, f"Optimized Q shape mismatch for rf={rf}"
    
    def test_different_qr_methods(self, device):
        """Test that different QR methods produce similar results"""
        torch.manual_seed(42)
        
        qr_methods = ["qr", "rcqr"]  # "cqr" might fail on some matrices
        
        models = []
        optimizers = []
        
        for method in qr_methods:
            model = nn.Linear(64, 32, bias=False).to(device)
            torch.manual_seed(42)
            nn.init.xavier_uniform_(model.weight)
            models.append(model)
            
            opt = DionReference(
                list(model.parameters()), 
                lr=0.01, 
                qr_method=method,
                cqr_warmup_steps=0
            )
            optimizers.append(opt)
        
        # Run steps
        for step in range(3):
            # Same gradient for all
            torch.manual_seed(step)
            grad = torch.randn(32, 64, device=device) * 0.01
            
            for model, opt in zip(models, optimizers):
                model.weight.grad = grad.clone()
                opt.step()
            
            # Compare parameters
            ref_param = models[0].weight
            for i, model in enumerate(models[1:], 1):
                # RCQR uses randomization so allow more tolerance
                assert torch.allclose(ref_param, model.weight, rtol=1e-2, atol=1e-3), \
                    f"QR method {qr_methods[i]} diverged from {qr_methods[0]}"
    
    @pytest.mark.skipif(not HAS_DION_OPTIMIZED, reason="DionOptimized requires optional dependencies")
    def test_mixed_parameter_types(self, device):
        """Test consistency with mixed parameter types"""
        torch.manual_seed(42)
        
        # Create models
        model_ref = self.create_mixed_model(device)
        model_opt = self.create_mixed_model(device)
        model_opt.load_state_dict(model_ref.state_dict())
        
        # Build parameter groups
        groups_ref = self.build_param_groups_for_mixed_model(model_ref)
        groups_opt = self.build_param_groups_for_mixed_model(model_opt)
        
        # Create optimizers
        opt_ref = DionReference(groups_ref, lr=0.01)
        opt_opt = DionOptimized(groups_opt, lr=0.01)
        
        # Run steps
        for step in range(3):
            self.generate_gradients(model_ref, device, seed=step)
            self.generate_gradients(model_opt, device, seed=step)
            
            opt_ref.step()
            opt_opt.step()
            
            state_ref = self.get_model_state(model_ref)
            state_opt = self.get_model_state(model_opt)
            
            assert self.compare_model_states(state_ref, state_opt, rtol=1e-4, atol=1e-5)
            
            opt_ref.zero_grad()
            opt_opt.zero_grad()