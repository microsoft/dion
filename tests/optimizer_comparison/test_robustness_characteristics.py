"""Tests comparing robustness characteristics across optimizers."""

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


class TestRobustnessCharacteristics(BaseOptimizerComparison):
    """Test robustness properties across different optimizers."""
    
    def test_gradient_explosion_handling(self, device):
        """Test how optimizers handle sudden gradient explosions"""
        torch.manual_seed(42)
        
        param_shape = (32, 32)
        normal_grad_scale = 0.01
        explosion_scale = 100.0
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        for name, opt_class, kwargs in configs:
            param = nn.Parameter(torch.randn(param_shape, device=device))
            opt = opt_class([param], **kwargs)
            
            param_trajectory = [param.clone()]
            
            for step in range(10):
                if step == 5:
                    # Gradient explosion at step 5
                    grad_scale = explosion_scale
                else:
                    grad_scale = normal_grad_scale
                
                param.grad = torch.randn_like(param) * grad_scale
                opt.step()
                opt.zero_grad()
                
                param_trajectory.append(param.clone())
            
            # Check recovery after explosion
            pre_explosion_norm = param_trajectory[4].norm()
            post_explosion_norm = param_trajectory[6].norm()
            final_norm = param_trajectory[-1].norm()
            
            print(f"\n{name} gradient explosion handling:")
            print(f"  Pre-explosion: {pre_explosion_norm:.4f}")
            print(f"  Post-explosion: {post_explosion_norm:.4f}")
            print(f"  Final: {final_norm:.4f}")
            
            # Should not diverge catastrophically
            assert torch.isfinite(param).all(), f"{name} produced non-finite values"
            assert final_norm < pre_explosion_norm * 10, f"{name} diverged after gradient explosion"
            
            # Lion should be most robust (sign-based updates)
            if name == "Lion":
                assert final_norm < pre_explosion_norm * 2, "Lion should be robust to gradient explosion"
    
    def test_gradient_vanishing_recovery(self, device):
        """Test optimizer behavior with vanishing gradients"""
        torch.manual_seed(42)
        
        param_shape = (32, 32)
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001, "eps": 1e-8}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        for name, opt_class, kwargs in configs:
            param = nn.Parameter(torch.randn(param_shape, device=device))
            param_init = param.clone()
            opt = opt_class([param], **kwargs)
            
            # Apply very small gradients
            num_vanishing_steps = 20
            for _ in range(num_vanishing_steps):
                param.grad = torch.randn_like(param) * 1e-8
                opt.step()
                opt.zero_grad()
            
            # Then apply normal gradient
            param.grad = torch.randn_like(param) * 0.1
            param_before_recovery = param.clone()
            opt.step()
            
            # Check if optimizer can still make progress
            recovery_update = (param - param_before_recovery).norm()
            total_movement = (param - param_init).norm()
            
            print(f"{name}: recovery_update={recovery_update:.6f}, total_movement={total_movement:.6f}")
            
            # Should still be able to update after vanishing gradients
            assert recovery_update > 1e-4, f"{name} cannot recover from vanishing gradients"
    
    def test_sparse_gradient_robustness(self, device):
        """Test how optimizers handle extremely sparse gradients"""
        torch.manual_seed(42)
        
        param_shape = (128, 64)
        sparsity_levels = [0.9, 0.99, 0.999]  # 90%, 99%, 99.9% zeros
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        for sparsity in sparsity_levels:
            print(f"\nTesting with {sparsity*100}% sparsity:")
            
            for name, opt_class, kwargs in configs:
                torch.manual_seed(42)
                param = nn.Parameter(torch.randn(param_shape, device=device))
                param_init = param.clone()
                opt = opt_class([param], **kwargs)
                
                # Create sparse gradient
                grad = torch.randn_like(param)
                mask = torch.rand_like(param) > sparsity
                sparse_grad = grad * mask
                
                # Take multiple steps with sparse gradients
                for _ in range(10):
                    param.grad = sparse_grad
                    opt.step()
                    opt.zero_grad()
                
                # Analyze update pattern
                update = param - param_init
                update_sparsity = (update.abs() < 1e-8).float().mean()
                
                print(f"  {name}: update_sparsity={update_sparsity:.3f}")
                
                # Should still make some progress
                assert update.norm() > 1e-4, f"{name} made no progress with sparse gradients"
    
    def test_ill_conditioned_gradient_handling(self, device):
        """Test optimizer behavior with ill-conditioned gradients"""
        torch.manual_seed(42)
        
        n = 32
        condition_numbers = [10, 100, 1000]
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        if HAS_MUON_REFERENCE:
            configs.append(("Muon", MuonReference, {"lr": 0.02}))
        
        for cond_num in condition_numbers:
            print(f"\nCondition number = {cond_num}:")
            
            for name, opt_class, kwargs in configs:
                torch.manual_seed(42)
                param = nn.Parameter(torch.eye(n, device=device))
                opt = opt_class([param], **kwargs)
                
                # Create ill-conditioned gradient
                U, _ = torch.linalg.qr(torch.randn(n, n, device=device))
                S = torch.logspace(0, np.log10(cond_num), n, device=device)
                grad = U @ torch.diag(S) @ U.T
                grad = grad / grad.norm() * 0.1
                
                param.grad = grad
                param_before = param.clone()
                opt.step()
                
                # Check update stability
                update = param - param_before
                update_norm = update.norm()
                
                # Check if update preserved any structure
                update_cond = torch.linalg.cond(update + 1e-8 * torch.eye(n, device=device))
                
                print(f"  {name}: update_norm={update_norm:.4f}, update_cond={update_cond:.1f}")
                
                # Should handle ill-conditioning gracefully
                assert torch.isfinite(param).all(), f"{name} produced non-finite with ill-conditioned gradient"
    
    def test_noise_filtering_capability(self, device):
        """Test if optimizers can filter out noise from gradients"""
        torch.manual_seed(42)
        
        param_shape = (64, 32)
        signal_rank = 4  # True gradient has low rank
        noise_level = 0.5
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01, "rank_fraction": 0.25}),
        ]
        
        for name, opt_class, kwargs in configs:
            torch.manual_seed(42)
            param = nn.Parameter(torch.randn(param_shape, device=device))
            opt = opt_class([param], **kwargs)
            
            # Create low-rank signal + high-rank noise
            U = torch.randn(param_shape[0], signal_rank, device=device)
            V = torch.randn(param_shape[1], signal_rank, device=device)
            signal = U @ V.T
            signal = signal / signal.norm() * 0.1
            
            noise = torch.randn_like(signal) * noise_level
            
            # Track alignment with true signal
            signal_alignments = []
            
            for _ in range(10):
                param_before = param.clone()
                
                # Gradient = signal + noise
                param.grad = signal + noise
                opt.step()
                opt.zero_grad()
                
                # Measure update alignment with signal
                update = param - param_before
                alignment = torch.nn.functional.cosine_similarity(
                    update.flatten(), signal.flatten(), dim=0
                ).item()
                signal_alignments.append(alignment)
            
            avg_alignment = np.mean(signal_alignments)
            print(f"{name}: avg signal alignment = {avg_alignment:.4f}")
            
            # Low-rank optimizers (Dion) should filter noise better
            if name == "Dion":
                assert avg_alignment < -0.5, "Dion should align well with signal"
    
    def test_catastrophic_forgetting_resistance(self, device):
        """Test if optimizers resist catastrophic parameter changes"""
        torch.manual_seed(42)
        
        param_shape = (32, 32)
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        for name, opt_class, kwargs in configs:
            param = nn.Parameter(torch.randn(param_shape, device=device))
            opt = opt_class([param], **kwargs)
            
            # Train on task 1 (gradient pointing in one direction)
            task1_direction = torch.randn_like(param)
            task1_direction = task1_direction / task1_direction.norm()
            
            param_after_task1 = None
            for _ in range(20):
                param.grad = -task1_direction * 0.01  # Consistent direction
                opt.step()
                opt.zero_grad()
            param_after_task1 = param.clone()
            
            # Switch to task 2 (orthogonal direction)
            task2_direction = torch.randn_like(param)
            task2_direction = task2_direction - (task2_direction * task1_direction).sum() * task1_direction
            task2_direction = task2_direction / task2_direction.norm()
            
            for _ in range(20):
                param.grad = -task2_direction * 0.01
                opt.step()
                opt.zero_grad()
            
            # Check how much of task 1 progress was retained
            task1_progress = (param_after_task1 * task1_direction).sum()
            final_task1_component = (param * task1_direction).sum()
            
            retention = final_task1_component / task1_progress if abs(task1_progress) > 1e-6 else 0
            
            print(f"{name}: task 1 retention = {retention:.4f}")
            
            # Optimizers with momentum should retain some task 1 knowledge
            assert retention > 0.5, f"{name} forgot task 1 completely"