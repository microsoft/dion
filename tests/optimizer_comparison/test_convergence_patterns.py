"""Tests comparing convergence patterns and loss reduction across optimizers."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
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


class TestConvergencePatterns(BaseOptimizerComparison):
    """Compare how different optimizers converge on various objectives."""
    
    def test_quadratic_convergence_speed(self, device):
        """Compare convergence speed on a simple quadratic objective"""
        torch.manual_seed(42)
        
        # Create quadratic problem: minimize ||Ax - b||^2
        n = 32
        A = torch.randn(n, n, device=device)
        A = A @ A.T + torch.eye(n, device=device)  # Ensure positive definite
        b = torch.randn(n, device=device)
        
        # Optimal solution for reference
        x_opt = torch.linalg.solve(A, b)
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.1}),
            ("Lion", Lion, {"lr": 0.01}),
            ("Dion", DionReference, {"lr": 0.1}),
        ]
        
        if HAS_MUON_REFERENCE:
            configs.append(("Muon", MuonReference, {"lr": 0.1}))
        
        convergence_history = {}
        
        for name, opt_class, kwargs in configs:
            torch.manual_seed(42)
            x = nn.Parameter(torch.randn(n, device=device))
            opt = opt_class([x], **kwargs)
            
            errors = []
            for _ in range(50):
                # Compute gradient of quadratic
                residual = A @ x - b
                loss = 0.5 * (residual ** 2).sum()
                
                loss.backward()
                opt.step()
                opt.zero_grad()
                
                # Track distance to optimum
                error = (x - x_opt).norm().item()
                errors.append(error)
            
            convergence_history[name] = errors
        
        # Analyze convergence rates
        for name, errors in convergence_history.items():
            final_error = errors[-1]
            convergence_rate = errors[-1] / errors[10] if errors[10] > 0 else 0
            print(f"{name}: final_error={final_error:.6f}, rate={convergence_rate:.6f}")
            
            # All should converge
            assert final_error < 0.1, f"{name} failed to converge on quadratic"
    
    def test_noisy_convergence_stability(self, device):
        """Test convergence stability with noisy gradients"""
        torch.manual_seed(42)
        
        # Simple 2D optimization for visualization
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        
        noise_level = 0.5
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.0001}),
            ("Dion", DionReference, {"lr": 0.001}),
        ]
        
        for name, opt_class, kwargs in configs:
            torch.manual_seed(42)
            x = nn.Parameter(torch.tensor([0.0, 0.0], device=device))
            opt = opt_class([x], **kwargs)
            
            trajectory = [x.clone().detach()]
            losses = []
            
            for _ in range(100):
                # Compute gradient with noise
                x_np = x.detach().cpu().numpy()
                loss = rosenbrock(x_np)
                losses.append(loss)
                
                # Approximate gradient
                eps = 1e-5
                grad = torch.zeros_like(x)
                for i in range(2):
                    x_plus = x_np.copy()
                    x_plus[i] += eps
                    x_minus = x_np.copy()
                    x_minus[i] -= eps
                    grad[i] = (rosenbrock(x_plus) - rosenbrock(x_minus)) / (2 * eps)
                
                # Add noise
                grad += torch.randn_like(grad) * noise_level
                
                x.grad = grad.to(device)
                opt.step()
                opt.zero_grad()
                
                trajectory.append(x.clone().detach())
            
            # Check if converged near optimum [1, 1]
            final_x = trajectory[-1]
            distance_to_opt = ((final_x - torch.tensor([1.0, 1.0], device=device))**2).sum().sqrt()
            
            print(f"{name}: final_loss={losses[-1]:.4f}, dist_to_opt={distance_to_opt:.4f}")
            
            # More lenient check due to noise
            assert losses[-1] < losses[0] * 0.5, f"{name} failed to reduce loss with noise"
    
    def test_loss_landscape_navigation(self, device):
        """Test how optimizers navigate different loss landscapes"""
        torch.manual_seed(42)
        
        # Create model with different loss characteristics
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                return self.fc2(F.relu(self.fc1(x)))
        
        # Test on different objectives
        objectives = [
            ("mse", lambda pred, target: F.mse_loss(pred, target)),
            ("cross_entropy", lambda pred, target: F.cross_entropy(pred, target.argmax(dim=1))),
            ("huber", lambda pred, target: F.huber_loss(pred, target, delta=0.5)),
        ]
        
        configs = [
            ("AdamW", AdamW, {"lr": 0.001}),
            ("Lion", Lion, {"lr": 0.0001}),
            ("Dion", DionReference, {"lr": 0.01}),
        ]
        
        results = {}
        
        for obj_name, loss_fn in objectives:
            print(f"\nTesting {obj_name} objective:")
            
            for opt_name, opt_class, kwargs in configs:
                torch.manual_seed(42)
                model = TestModel().to(device)
                
                # Only optimize matrix parameters for Dion
                if opt_name == "Dion":
                    params = [p for p in model.parameters() if p.ndim == 2]
                else:
                    params = model.parameters()
                
                opt = opt_class(params, **kwargs)
                
                # Generate fixed data
                X = torch.randn(100, input_dim, device=device)
                y = torch.randn(100, output_dim, device=device)
                
                losses = []
                for _ in range(20):
                    pred = model(X)
                    loss = loss_fn(pred, y)
                    
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
                    losses.append(loss.item())
                
                improvement = (losses[0] - losses[-1]) / losses[0]
                results[(obj_name, opt_name)] = improvement
                print(f"  {opt_name}: improvement = {improvement:.2%}")
    
    def test_convergence_with_momentum_comparison(self, device):
        """Compare momentum effects on convergence across optimizers"""
        torch.manual_seed(42)
        
        # Simple linear regression problem
        n_features = 20
        n_samples = 100
        
        X = torch.randn(n_samples, n_features, device=device)
        true_w = torch.randn(n_features, device=device)
        y = X @ true_w + torch.randn(n_samples, device=device) * 0.1
        
        # Test different momentum settings
        momentum_configs = [
            ("AdamW_low", AdamW, {"lr": 0.01, "betas": (0.5, 0.999)}),
            ("AdamW_high", AdamW, {"lr": 0.01, "betas": (0.95, 0.999)}),
            ("Lion_low", Lion, {"lr": 0.001, "beta": 0.5}),
            ("Lion_high", Lion, {"lr": 0.001, "beta": 0.95}),
            ("Dion_low", DionReference, {"lr": 0.1, "mu": 0.5}),
            ("Dion_high", DionReference, {"lr": 0.1, "mu": 0.95}),
        ]
        
        for name, opt_class, kwargs in momentum_configs:
            torch.manual_seed(42)
            w = nn.Parameter(torch.randn(n_features, device=device))
            opt = opt_class([w], **kwargs)
            
            losses = []
            for _ in range(50):
                pred = X @ w
                loss = F.mse_loss(pred, y)
                
                loss.backward()
                opt.step()
                opt.zero_grad()
                
                losses.append(loss.item())
            
            # Analyze convergence smoothness
            # Calculate variance of loss differences
            loss_diffs = [losses[i+1] - losses[i] for i in range(len(losses)-1)]
            smoothness = torch.std(torch.tensor(loss_diffs))
            
            print(f"{name}: final_loss={losses[-1]:.4f}, smoothness={smoothness:.4f}")
            
            # High momentum should lead to smoother convergence
            if "high" in name:
                assert smoothness < 0.1, f"{name} convergence too erratic"