"""Tests comparing fundamental characteristics across all optimizer types."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

# Import all optimizers
from optimizers.dion_reference import Dion as DionReference
from optimizers.scalar_opts import Lion, AdamW

# Try to import optional optimizers
try:
    from optimizers.muon_reference import Muon as MuonReference
    HAS_MUON_REFERENCE = True
except ImportError:
    HAS_MUON_REFERENCE = False
    MuonReference = None

try:
    from optimizers.dion_simple import Dion as DionSimple
    HAS_DION_SIMPLE = True
except ImportError:
    HAS_DION_SIMPLE = False
    DionSimple = None


class TestOptimizerCharacteristics:
    """Test fundamental characteristics that differ between optimizers."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_parameter_norm_evolution(self, device):
        """Compare how different optimizers affect parameter norms over time"""
        torch.manual_seed(42)
        
        # Test configuration
        param_shape = (64, 32)
        num_steps = 20
        
        # Optimizers to test
        configs = [
            ("AdamW", AdamW, {"lr": 0.001, "weight_decay": 0.1}),
            ("Lion", Lion, {"lr": 0.001, "weight_decay": 0.1}),
            ("Dion", DionReference, {"lr": 0.01, "weight_decay": 0.1}),
        ]
        
        if HAS_MUON_REFERENCE:
            configs.append(("Muon", MuonReference, {"lr": 0.02}))
        
        results = {}
        
        for name, opt_class, kwargs in configs:
            torch.manual_seed(42)
            param = nn.Parameter(torch.randn(param_shape, device=device) * 5.0)
            opt = opt_class([param], **kwargs)
            
            norms = [param.norm().item()]
            
            for _ in range(num_steps):
                # Small random gradient
                param.grad = torch.randn_like(param) * 0.01
                opt.step()
                opt.zero_grad()
                norms.append(param.norm().item())
            
            results[name] = norms
        
        # Analyze patterns
        # AdamW and Lion should show consistent decay due to weight decay
        assert results["AdamW"][-1] < results["AdamW"][0] * 0.5, "AdamW should decay weights"
        assert results["Lion"][-1] < results["Lion"][0] * 0.5, "Lion should decay weights"
        
        # Dion might behave differently due to orthogonal updates
        print(f"Final norm ratios: {[(k, v[-1]/v[0]) for k, v in results.items()]}")
    
    def test_gradient_noise_robustness(self, device):
        """Test optimizer behavior with different gradient noise levels"""
        torch.manual_seed(42)
        
        base_shape = (32, 32)
        noise_levels = [0.01, 0.1, 1.0]
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01, "rank_fraction": 0.5}),
        ]
        
        for noise_std in noise_levels:
            print(f"\nTesting with noise level: {noise_std}")
            
            for name, opt_class, kwargs in configs:
                torch.manual_seed(42)
                
                # Start from same initial point
                param = nn.Parameter(torch.eye(base_shape[0], device=device))
                opt = opt_class([param], **kwargs)
                
                # True gradient is towards negative identity
                true_grad = -torch.eye(base_shape[0], device=device) * 0.1
                
                # Track deviation from ideal path
                deviations = []
                
                for step in range(10):
                    # Add noise to gradient
                    noise = torch.randn_like(true_grad) * noise_std
                    param.grad = true_grad + noise
                    
                    param_before = param.clone()
                    opt.step()
                    
                    # Measure how much update deviates from true gradient direction
                    actual_update = param - param_before
                    ideal_update = -kwargs.get("lr", 0.001) * true_grad
                    
                    deviation = (actual_update - ideal_update).norm() / ideal_update.norm()
                    deviations.append(deviation.item())
                
                avg_deviation = np.mean(deviations)
                print(f"  {name}: avg deviation = {avg_deviation:.4f}")
                
                # Low-rank methods (Dion) might filter noise better
                if name == "Dion" and noise_std > 0.1:
                    assert avg_deviation < 5.0, f"Dion too sensitive to noise"
    
    def test_sparse_gradient_handling(self, device):
        """Test how optimizers handle sparse gradients"""
        torch.manual_seed(42)
        
        param_size = (128, 64)
        sparsity = 0.95  # 95% zeros
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        for name, opt_class, kwargs in configs:
            param = nn.Parameter(torch.randn(param_size, device=device))
            param_init = param.clone()
            opt = opt_class([param], **kwargs)
            
            # Create sparse gradient
            grad = torch.randn_like(param) * 0.1
            mask = torch.rand_like(grad) > sparsity
            sparse_grad = grad * mask
            
            param.grad = sparse_grad
            opt.step()
            
            # Check update pattern
            update = param - param_init
            
            # For AdamW/Lion, update should be localized to non-zero gradient regions
            if name in ["AdamW", "Lion"]:
                # Check sparsity is somewhat preserved
                update_sparsity = (update.abs() < 1e-8).float().mean()
                assert update_sparsity > 0.5, f"{name} should preserve some sparsity"
            
            # Dion might spread updates due to low-rank approximation
            if name == "Dion":
                update_sparsity = (update.abs() < 1e-8).float().mean()
                print(f"Dion update sparsity: {update_sparsity:.3f}")
    
    def test_learning_rate_sensitivity(self, device):
        """Test optimizer stability across different learning rates"""
        torch.manual_seed(42)
        
        # Test learning rate multiples
        lr_scales = [0.1, 1.0, 10.0, 100.0]
        
        configs = [
            ("AdamW", AdamW, 0.001),    # Base LR
            ("Lion", Lion, 0.001),
            ("Dion", DionReference, 0.01),
        ]
        
        if HAS_MUON_REFERENCE:
            configs.append(("Muon", MuonReference, 0.02))
        
        for name, opt_class, base_lr in configs:
            print(f"\n{name} learning rate sensitivity:")
            
            for lr_scale in lr_scales:
                torch.manual_seed(42)
                param = nn.Parameter(torch.randn(32, 32, device=device))
                
                lr = base_lr * lr_scale
                opt = opt_class([param], lr=lr)
                
                # Apply same gradients
                stable = True
                for _ in range(5):
                    param.grad = torch.randn_like(param) * 0.1
                    opt.step()
                    
                    if not torch.isfinite(param).all():
                        stable = False
                        break
                
                status = "stable" if stable else "unstable"
                param_norm = param.norm().item() if stable else float('inf')
                print(f"  lr={lr:.4f} ({lr_scale}x): {status}, final_norm={param_norm:.2f}")
    
    def test_batch_size_invariance(self, device):
        """Test if optimizers behave consistently across batch sizes"""
        torch.manual_seed(42)
        
        # Simulate different batch sizes by gradient scaling
        batch_sizes = [1, 16, 128]
        param_shape = (64, 32)
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        for name, opt_class, kwargs in configs:
            updates = {}
            
            for batch_size in batch_sizes:
                torch.manual_seed(42)
                param = nn.Parameter(torch.randn(param_shape, device=device))
                param_init = param.clone()
                opt = opt_class([param], **kwargs)
                
                # Simulate gradient from batch
                # Larger batch = smaller gradient variance
                grad_scale = 1.0 / np.sqrt(batch_size)
                param.grad = torch.randn_like(param) * 0.1 * grad_scale
                
                opt.step()
                
                update = (param - param_init).norm().item()
                updates[batch_size] = update
            
            # Check invariance (updates should be similar)
            update_values = list(updates.values())
            max_ratio = max(update_values) / min(update_values)
            
            print(f"{name} batch size invariance: {updates}, ratio: {max_ratio:.2f}")
            
            # Most optimizers should show some batch size dependence
            # but it shouldn't be extreme
            assert max_ratio < 10.0, f"{name} too sensitive to batch size"
    
    @pytest.mark.skipif(not HAS_MUON_REFERENCE, reason="Muon not available")
    def test_orthogonal_invariance(self, device):
        """Test if matrix optimizers are invariant to orthogonal transformations"""
        torch.manual_seed(42)
        
        n = 32
        param_original = torch.randn(n, n, device=device)
        
        # Generate random orthogonal matrix
        Q, _ = torch.linalg.qr(torch.randn(n, n, device=device))
        
        # Test configurations
        configs = [
            ("Dion", DionReference, {"lr": 0.01}),
            ("Muon", MuonReference, {"lr": 0.02}),
        ]
        
        for name, opt_class, kwargs in configs:
            # Original parameter
            param1 = nn.Parameter(param_original.clone())
            opt1 = opt_class([param1], **kwargs)
            
            # Orthogonally transformed parameter
            param2 = nn.Parameter(Q @ param_original @ Q.T)
            opt2 = opt_class([param2], **kwargs)
            
            # Apply corresponding gradients
            grad = torch.randn_like(param_original) * 0.1
            param1.grad = grad
            param2.grad = Q @ grad @ Q.T
            
            # Take steps
            opt1.step()
            opt2.step()
            
            # Check if updates are equivalent up to transformation
            param1_transformed = Q @ param1 @ Q.T
            
            assert torch.allclose(param1_transformed, param2, rtol=1e-4, atol=1e-5), \
                f"{name} not invariant to orthogonal transformation"
    
    def test_memory_momentum_differences(self, device):
        """Compare memory/momentum patterns across optimizers"""
        torch.manual_seed(42)
        
        steps = 10
        param_shape = (32, 16)
        
        # Apply alternating gradients to test memory
        grad1 = torch.randn(param_shape, device=device) * 0.1
        grad2 = -grad1  # Opposite direction
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001, "betas": (0.9, 0.999)}),
            ("Lion", Lion, {"lr": 0.001, "beta": 0.9}),
            ("Dion", DionReference, {"lr": 0.01, "mu": 0.9}),
        ]
        
        for name, opt_class, kwargs in configs:
            param = nn.Parameter(torch.randn(param_shape, device=device))
            opt = opt_class([param], **kwargs)
            
            positions = [param.clone()]
            
            for i in range(steps):
                # Alternate between two gradients
                param.grad = grad1 if i % 2 == 0 else grad2
                opt.step()
                positions.append(param.clone())
            
            # Analyze oscillation pattern
            distances = []
            for i in range(1, len(positions)):
                dist = (positions[i] - positions[i-1]).norm().item()
                distances.append(dist)
            
            # Check if optimizer dampens oscillations
            first_half = np.mean(distances[:steps//2])
            second_half = np.mean(distances[steps//2:])
            
            damping_ratio = second_half / first_half
            print(f"{name} oscillation damping: {damping_ratio:.3f}")
            
            # Optimizers with momentum should dampen oscillations
            if name in ["AdamW", "Dion"]:
                assert damping_ratio < 1.0, f"{name} should dampen oscillations"