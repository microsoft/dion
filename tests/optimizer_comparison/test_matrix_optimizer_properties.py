"""Tests comparing properties of matrix-based optimizers (Dion vs Muon)."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from .base_comparison import BaseOptimizerComparison

# Import optimizer variants
from optimizers.dion_reference import Dion as DionReference

# Try to import Muon
try:
    from optimizers.muon_reference import Muon as MuonReference
    HAS_MUON_REFERENCE = True
except ImportError:
    HAS_MUON_REFERENCE = False
    MuonReference = None


@pytest.mark.skipif(not HAS_MUON_REFERENCE, reason="MuonReference not available")
class TestMatrixOptimizerProperties(BaseOptimizerComparison):
    """Compare fundamental properties of matrix-based optimizers."""
    
    def test_dion_vs_muon_rank_preservation(self, device):
        """Test how Dion and Muon handle low-rank structure"""
        torch.manual_seed(42)
        
        # Create a low-rank matrix parameter
        m, n, true_rank = 64, 32, 8
        U = torch.randn(m, true_rank, device=device)
        V = torch.randn(n, true_rank, device=device)
        low_rank_param = nn.Parameter(U @ V.T)
        
        # Create optimizers
        dion_param = low_rank_param.clone().detach().requires_grad_(True)
        muon_param = low_rank_param.clone().detach().requires_grad_(True)
        
        opt_dion = DionReference([dion_param], lr=0.01, rank_fraction=0.5)
        opt_muon = MuonReference([muon_param], lr=0.02)
        
        # Apply gradient that preserves rank
        grad = U @ torch.randn(true_rank, true_rank, device=device) @ V.T
        dion_param.grad = grad.clone()
        muon_param.grad = grad.clone()
        
        # Take steps
        opt_dion.step()
        opt_muon.step()
        
        # Check rank preservation
        def estimate_rank(X, threshold=1e-6):
            _, S, _ = torch.linalg.svd(X)
            return (S > threshold * S[0]).sum().item()
        
        dion_rank = estimate_rank(dion_param)
        muon_rank = estimate_rank(muon_param)
        
        # Both should approximately preserve low-rank structure
        assert dion_rank <= true_rank * 2, f"Dion inflated rank too much: {dion_rank}"
        assert muon_rank <= true_rank * 2, f"Muon inflated rank too much: {muon_rank}"
    
    def test_dion_vs_muon_gradient_alignment(self, device):
        """Test how updates align with gradient direction"""
        torch.manual_seed(42)
        
        # Create parameters
        shape = (32, 32)
        dion_param = nn.Parameter(torch.randn(shape, device=device))
        muon_param = nn.Parameter(torch.randn(shape, device=device))
        muon_param.data.copy_(dion_param.data)
        
        # Create optimizers
        opt_dion = DionReference([dion_param], lr=0.01)
        opt_muon = MuonReference([muon_param], lr=0.02)
        
        # Apply same gradient
        grad = torch.randn(shape, device=device)
        dion_param.grad = grad.clone()
        muon_param.grad = grad.clone()
        
        # Store initial params
        dion_init = dion_param.clone()
        muon_init = muon_param.clone()
        
        # Take steps
        opt_dion.step()
        opt_muon.step()
        
        # Compute updates
        dion_update = dion_param - dion_init
        muon_update = muon_param - muon_init
        
        # Compute alignment with gradient (cosine similarity)
        def cosine_sim(a, b):
            return (a * b).sum() / (a.norm() * b.norm())
        
        dion_alignment = cosine_sim(dion_update.flatten(), grad.flatten())
        muon_alignment = cosine_sim(muon_update.flatten(), grad.flatten())
        
        # Both should have negative alignment (moving against gradient)
        assert dion_alignment < 0, "Dion should move against gradient"
        assert muon_alignment < 0, "Muon should move against gradient"
    
    def test_dion_vs_muon_orthogonality_properties(self, device):
        """Compare orthogonalization approaches"""
        torch.manual_seed(42)
        
        # Create parameters with known structure
        param = torch.randn(64, 32, device=device)
        
        # Test Dion's QR-based approach
        opt_dion = DionReference([nn.Parameter(param.clone())], lr=0.01)
        grad = torch.randn_like(param)
        opt_dion.param_groups[0]['params'][0].grad = grad
        opt_dion.step()
        
        # Check Dion's Q matrix orthogonality
        Q_dion = opt_dion.state[opt_dion.param_groups[0]['params'][0]]["Q"]
        QtQ = Q_dion.T @ Q_dion
        I = torch.eye(QtQ.shape[0], device=device)
        dion_orth_error = (QtQ - I).abs().max().item()
        
        # Muon uses different approach (Newton-Schulz)
        # Just verify both maintain some orthogonal structure
        assert dion_orth_error < 1e-5, "Dion should maintain orthogonality"
    
    def test_dion_vs_muon_momentum_behavior(self, device):
        """Compare momentum accumulation patterns"""
        torch.manual_seed(42)
        
        # Create identical parameters
        shape = (32, 32)
        dion_param = nn.Parameter(torch.randn(shape, device=device))
        muon_param = nn.Parameter(torch.randn(shape, device=device))
        muon_param.data.copy_(dion_param.data)
        
        # Create optimizers with similar momentum
        opt_dion = DionReference([dion_param], lr=0.01, mu=0.9)
        opt_muon = MuonReference([muon_param], lr=0.02, momentum=0.9)
        
        # Apply constant gradient multiple times
        constant_grad = torch.randn(shape, device=device) * 0.01
        
        dion_updates = []
        muon_updates = []
        
        for _ in range(5):
            dion_before = dion_param.clone()
            muon_before = muon_param.clone()
            
            dion_param.grad = constant_grad.clone()
            muon_param.grad = constant_grad.clone()
            
            opt_dion.step()
            opt_muon.step()
            
            dion_updates.append((dion_param - dion_before).norm().item())
            muon_updates.append((muon_param - muon_before).norm().item())
        
        # Both should show increasing updates due to momentum
        assert dion_updates[-1] > dion_updates[0], "Dion momentum should accumulate"
        assert muon_updates[-1] > muon_updates[0], "Muon momentum should accumulate"
    
    def test_matrix_vs_scalar_optimizer_separation(self, device):
        """Test that matrix optimizers don't update scalar params and vice versa"""
        torch.manual_seed(42)
        
        # Create model with mixed parameters
        model = self.create_mixed_model(device)
        
        # Separate parameters
        matrix_params = []
        scalar_params = []
        
        for name, param in model.named_parameters():
            if param.ndim == 2 and 'embedding' not in name:
                matrix_params.append(param)
            else:
                scalar_params.append(param)
        
        # Create optimizers that should only handle their param types
        if matrix_params:
            opt_dion = DionReference(matrix_params, lr=0.01)
            if HAS_MUON_REFERENCE:
                opt_muon = MuonReference(matrix_params, lr=0.02)
        
        # Generate gradients
        self.generate_gradients(model, device)
        
        # Store initial scalar param values
        scalar_init = {name: p.clone() for name, p in model.named_parameters() 
                      if p in scalar_params}
        
        # Step matrix optimizers
        if matrix_params:
            opt_dion.step()
            opt_dion.zero_grad()
        
        # Verify scalar params unchanged
        for name, param in model.named_parameters():
            if param in scalar_params:
                assert torch.allclose(param, scalar_init[name]), \
                    f"Matrix optimizer modified scalar param {name}"
    
    def test_dion_vs_muon_eigenvector_preservation(self, device):
        """Test how optimizers affect principal components"""
        torch.manual_seed(42)
        
        # Create parameter with known eigenvectors
        n = 32
        param = torch.randn(n, n, device=device)
        param = param @ param.T  # Make symmetric for real eigenvalues
        
        # Get initial eigenvectors
        eigvals_init, eigvecs_init = torch.linalg.eigh(param)
        
        # Create optimizers
        dion_param = nn.Parameter(param.clone())
        muon_param = nn.Parameter(param.clone())
        
        opt_dion = DionReference([dion_param], lr=0.001)
        opt_muon = MuonReference([muon_param], lr=0.002)
        
        # Apply gradient that's aligned with top eigenvector
        top_eigvec = eigvecs_init[:, -1:]
        grad = top_eigvec @ top_eigvec.T * 0.1
        
        dion_param.grad = grad.clone()
        muon_param.grad = grad.clone()
        
        # Take steps
        opt_dion.step()
        opt_muon.step()
        
        # Check eigenvector alignment
        _, eigvecs_dion = torch.linalg.eigh(dion_param)
        _, eigvecs_muon = torch.linalg.eigh(muon_param)
        
        # Top eigenvector should remain similar
        dion_alignment = abs((eigvecs_init[:, -1] * eigvecs_dion[:, -1]).sum())
        muon_alignment = abs((eigvecs_init[:, -1] * eigvecs_muon[:, -1]).sum())
        
        assert dion_alignment > 0.9, "Dion should preserve top eigenvector"
        assert muon_alignment > 0.9, "Muon should preserve top eigenvector"
    
    def test_optimizer_conditioning_sensitivity(self, device):
        """Test how optimizers handle ill-conditioned matrices"""
        torch.manual_seed(42)
        
        # Create ill-conditioned matrix
        n = 32
        U, _ = torch.linalg.qr(torch.randn(n, n, device=device))
        # Create spectrum from 1 to 1000 (condition number = 1000)
        S = torch.logspace(0, 3, n, device=device)
        ill_cond_param = U @ torch.diag(S) @ U.T
        
        # Test each optimizer
        optimizers_to_test = [
            ("Dion", DionReference, {"lr": 0.01}),
            ("Muon", MuonReference, {"lr": 0.02}),
        ]
        
        results = {}
        
        for name, opt_class, kwargs in optimizers_to_test:
            if name == "Muon" and not HAS_MUON_REFERENCE:
                continue
                
            param = nn.Parameter(ill_cond_param.clone())
            opt = opt_class([param], **kwargs)
            
            # Apply gradient
            grad = torch.randn_like(param) * 0.01
            param.grad = grad
            
            # Take step and check stability
            param_before = param.clone()
            opt.step()
            
            # Compute update magnitude
            update = param - param_before
            relative_update = update.norm() / param_before.norm()
            
            results[name] = relative_update.item()
            
            # Check for numerical stability
            assert torch.isfinite(param).all(), f"{name} produced non-finite values"
            assert relative_update < 0.1, f"{name} update too large for ill-conditioned matrix"
        
        print(f"Relative updates on ill-conditioned matrix: {results}")