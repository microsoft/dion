"""Tests comparing how different optimizers update parameters."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from .base_comparison import BaseOptimizerComparison

# Import optimizer variants
from optimizers.dion_reference import Dion as DionReference
from optimizers.scalar_opts import Lion, AdamW

# Try to import optional optimizers
try:
    from optimizers.muon_reference import Muon as MuonReference
    HAS_MUON_REFERENCE = True
except ImportError:
    HAS_MUON_REFERENCE = False
    MuonReference = None


class TestParameterUpdatePatterns(BaseOptimizerComparison):
    """Compare parameter update patterns across optimizers."""
    
    def test_update_magnitude_vs_gradient_magnitude(self, device):
        """Test relationship between gradient magnitude and update magnitude"""
        torch.manual_seed(42)
        
        param_shape = (32, 32)
        gradient_scales = [0.001, 0.01, 0.1, 1.0]
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        for name, opt_class, kwargs in configs:
            update_ratios = []
            
            for grad_scale in gradient_scales:
                torch.manual_seed(42)
                param = nn.Parameter(torch.randn(param_shape, device=device))
                param_init = param.clone()
                opt = opt_class([param], **kwargs)
                
                # Apply scaled gradient
                grad = torch.randn_like(param).div_(grad.norm()).mul_(grad_scale)
                param.grad = grad
                
                opt.step()
                
                # Measure update magnitude
                update = param - param_init
                update_magnitude = update.norm().item()
                
                # Ratio of update to gradient magnitude
                ratio = update_magnitude / grad_scale if grad_scale > 0 else 0
                update_ratios.append(ratio)
            
            print(f"\n{name} update/gradient ratios:")
            for scale, ratio in zip(gradient_scales, update_ratios):
                print(f"  grad_scale={scale}: ratio={ratio:.4f}")
            
            # Check for adaptive behavior
            # AdamW should show decreasing ratios (adaptive)
            # Lion should show constant ratios (sign-based)
            if name == "Lion":
                assert np.std(update_ratios) < 0.1, "Lion should have constant update ratio"
    
    def test_update_direction_vs_gradient_direction(self, device):
        """Test how update direction relates to gradient direction"""
        torch.manual_seed(42)
        
        param_shape = (64, 32)
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        if HAS_MUON_REFERENCE:
            configs.append(("Muon", MuonReference, {"lr": 0.02}))
        
        for name, opt_class, kwargs in configs:
            torch.manual_seed(42)
            
            # Test with different gradient patterns
            test_cases = [
                ("random", torch.randn(param_shape, device=device)),
                ("structured", torch.ones(param_shape, device=device).tril()),
                ("sparse", torch.zeros(param_shape, device=device).scatter_(
                    0, torch.randint(0, param_shape[0], (10,)), 1.0)),
            ]
            
            for pattern_name, grad_pattern in test_cases:
                param = nn.Parameter(torch.randn(param_shape, device=device))
                param_init = param.clone()
                opt = opt_class([param], **kwargs)
                
                # Normalize gradient
                grad = grad_pattern / grad_pattern.norm() * 0.1
                param.grad = grad
                
                opt.step()
                
                # Compute update
                update = param - param_init
                
                # Compute cosine similarity
                cosine_sim = torch.nn.functional.cosine_similarity(
                    update.flatten(), grad.flatten(), dim=0
                ).item()
                
                print(f"{name} - {pattern_name}: cosine_sim = {cosine_sim:.4f}")
                
                # All optimizers should generally move against gradient
                assert cosine_sim < 0, f"{name} not moving against gradient"
    
    def test_parameter_wise_update_scaling(self, device):
        """Test if updates scale appropriately with parameter magnitude"""
        torch.manual_seed(42)
        
        # Create parameters with different scales
        scales = [0.01, 0.1, 1.0, 10.0]
        base_shape = (16, 16)
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001, "weight_decay": 0.0}),
            ("Lion", Lion, {"lr": 0.001, "weight_decay": 0.0}),
            ("Dion", DionReference, {"lr": 0.01, "weight_decay": 0.0}),
        ]
        
        for name, opt_class, kwargs in configs:
            relative_updates = []
            
            for scale in scales:
                torch.manual_seed(42)
                # Scale parameter initialization
                param = nn.Parameter(torch.randn(base_shape, device=device) * scale)
                param_init = param.clone()
                opt = opt_class([param], **kwargs)
                
                # Apply same gradient pattern
                param.grad = torch.randn_like(param) * 0.01
                
                opt.step()
                
                # Compute relative update
                update = param - param_init
                relative_update = (update.abs() / (param_init.abs() + 1e-8)).mean().item()
                relative_updates.append(relative_update)
            
            print(f"\n{name} relative updates by parameter scale:")
            for scale, rel_update in zip(scales, relative_updates):
                print(f"  scale={scale}: relative_update={rel_update:.6f}")
            
            # Most optimizers should show scale-invariant relative updates
            # (except for weight decay effects)
            cv = np.std(relative_updates) / np.mean(relative_updates)
            print(f"  Coefficient of variation: {cv:.4f}")
    
    def test_sign_based_vs_magnitude_based_updates(self, device):
        """Compare sign-based (Lion) vs magnitude-based (AdamW) update patterns"""
        torch.manual_seed(42)
        
        param_shape = (32, 32)
        
        # Create structured gradients with varying magnitudes
        grad_base = torch.randn(param_shape, device=device)
        
        # Scale different regions differently
        grad_scaled = grad_base.clone()
        grad_scaled[:16, :] *= 10.0  # Top half has 10x larger gradients
        grad_scaled[16:, :] *= 0.1   # Bottom half has 0.1x smaller gradients
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
        ]
        
        for name, opt_class, kwargs in configs:
            param = nn.Parameter(torch.zeros(param_shape, device=device))
            opt = opt_class([param], **kwargs)
            
            param.grad = grad_scaled
            opt.step()
            
            # Analyze update pattern
            update = param.data
            
            # Check if updates reflect gradient magnitudes
            top_update_mean = update[:16, :].abs().mean().item()
            bottom_update_mean = update[16:, :].abs().mean().item()
            
            ratio = top_update_mean / bottom_update_mean if bottom_update_mean > 0 else float('inf')
            
            print(f"{name}: top/bottom update ratio = {ratio:.2f}")
            
            # AdamW should show larger updates where gradients are larger
            # Lion should show similar magnitude updates (sign-based)
            if name == "Lion":
                assert ratio < 2.0, "Lion updates should be magnitude-independent"
            elif name == "AdamW":
                assert ratio > 5.0, "AdamW updates should reflect gradient magnitudes"
    
    def test_update_patterns_with_momentum(self, device):
        """Test how momentum affects update patterns over time"""
        torch.manual_seed(42)
        
        param_shape = (32, 16)
        num_steps = 10
        
        # Alternating gradient pattern to test momentum
        grad1 = torch.randn(param_shape, device=device) * 0.1
        grad2 = -grad1 * 0.5  # Opposite but smaller
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001, "betas": (0.9, 0.999)}),
            ("Lion", Lion, {"lr": 0.001, "beta": 0.9}),
            ("Dion", DionReference, {"lr": 0.01, "mu": 0.9}),
        ]
        
        for name, opt_class, kwargs in configs:
            param = nn.Parameter(torch.randn(param_shape, device=device))
            opt = opt_class([param], **kwargs)
            
            updates = []
            
            for i in range(num_steps):
                param_before = param.clone()
                
                # Alternate gradients
                param.grad = grad1 if i % 2 == 0 else grad2
                opt.step()
                
                update = param - param_before
                updates.append(update)
            
            # Analyze momentum effect
            # With momentum, later updates should be smoother
            early_variance = torch.stack(updates[:3]).var(dim=0).mean().item()
            late_variance = torch.stack(updates[-3:]).var(dim=0).mean().item()
            
            variance_ratio = late_variance / early_variance
            print(f"{name}: variance ratio (late/early) = {variance_ratio:.4f}")
            
            # Momentum should reduce variance over time
            assert variance_ratio < 1.0, f"{name} momentum not smoothing updates"
    
    @pytest.mark.skipif(not HAS_MUON_REFERENCE, reason="Muon not available")
    def test_matrix_optimizer_update_structure(self, device):
        """Test structural properties of updates from matrix optimizers"""
        torch.manual_seed(42)
        
        param_shape = (64, 32)
        
        configs = [
            ("Dion", DionReference, {"lr": 0.01, "rank_fraction": 0.25}),
            ("Muon", MuonReference, {"lr": 0.02}),
        ]
        
        for name, opt_class, kwargs in configs:
            param = nn.Parameter(torch.randn(param_shape, device=device))
            param_init = param.clone()
            opt = opt_class([param], **kwargs)
            
            # Apply full-rank gradient
            param.grad = torch.randn_like(param) * 0.01
            opt.step()
            
            # Analyze update structure
            update = param - param_init
            
            # Compute effective rank of update
            U, S, Vt = torch.linalg.svd(update)
            
            # Normalize singular values
            S_normalized = S / S[0] if S[0] > 0 else S
            
            # Count significant singular values
            effective_rank = (S_normalized > 0.01).sum().item()
            rank_ratio = effective_rank / min(param_shape)
            
            print(f"{name}: effective rank = {effective_rank}/{min(param_shape)} ({rank_ratio:.2f})")
            
            # Dion with rank_fraction=0.25 should produce low-rank updates
            if name == "Dion":
                assert rank_ratio < 0.5, "Dion update rank too high"