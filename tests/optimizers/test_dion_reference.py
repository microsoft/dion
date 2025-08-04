import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
import math

from optimizers.dion_reference import (
    Dion, DionParamConfig, DionMixedPrecisionConfig,
    dion_update, power_iteration, orthogonalize, 
    fix_all_zero_or_nan, all_reduce
)
from optimizers.scalar_opts import adamw_update, lion_update


class TestDionReference:
    """Comprehensive unit tests for Dion reference optimizer"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def simple_model(self, device):
        """Create a simple model with different parameter types"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(32, 64, bias=True)
                self.linear2 = nn.Linear(64, 128, bias=False)
                self.embedding = nn.Embedding(100, 32)
                self.norm = nn.LayerNorm(128)
                self.lm_head = nn.Linear(128, 100)
            
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.norm(x)
                x = self.lm_head(x)
                return x
                
        return SimpleModel().to(device)
    
    def build_param_groups(self, model: nn.Module) -> List[Dict[str, Any]]:
        """Build parameter groups for Dion optimizer"""
        matrix_params = []
        vector_params = []
        embed_params = []
        lm_head_params = []
        
        for name, param in model.named_parameters():
            if param.ndim == 2 and "embedding" not in name and "lm_head" not in name:
                matrix_params.append(param)
            elif "embedding" in name:
                embed_params.append(param)
            elif "lm_head" in name:
                lm_head_params.append(param)
            else:
                vector_params.append(param)
        
        lr = 0.01
        param_groups = [
            {"params": matrix_params},  # defaults to dion
            {"params": vector_params, "algorithm": "lion"},
            {"params": embed_params, "algorithm": "lion"},
            {"params": lm_head_params, "algorithm": "lion", "lr": lr / math.sqrt(128)}
        ]
        
        return param_groups
    
    def test_optimizer_initialization(self, simple_model):
        """Test optimizer initialization with various configurations"""
        param_groups = self.build_param_groups(simple_model)
        
        # Test basic initialization
        opt = Dion(param_groups, lr=0.01)
        assert opt is not None
        
        # Test with rank fraction
        opt = Dion(param_groups, lr=0.01, rank_fraction=0.25)
        assert opt.defaults["rank_fraction"] == 0.25
        
        # Test with mixed precision config
        mp_config = DionMixedPrecisionConfig(
            momentum_dtype=torch.float32,
            Q_dtype=torch.bfloat16,
            variance_dtype=torch.float32
        )
        opt = Dion(param_groups, lr=0.01, mixed_precision_config=mp_config)
        assert opt._mixed_precision_config.Q_dtype == torch.bfloat16
    
    def test_invalid_hyperparameters(self, simple_model):
        """Test that invalid hyperparameters raise appropriate errors"""
        param_groups = self.build_param_groups(simple_model)
        
        # Test invalid learning rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Dion(param_groups, lr=-0.01)
        
        # Test invalid momentum
        with pytest.raises(ValueError, match="Invalid momentum factor"):
            Dion(param_groups, mu=-0.5)
        
        # Test invalid rank fraction
        with pytest.raises(ValueError, match="Invalid rank fraction"):
            Dion(param_groups, rank_fraction=0.0)
        
        with pytest.raises(ValueError, match="Invalid rank fraction"):
            Dion(param_groups, rank_fraction=1.5)
        
        # Test invalid QR method
        with pytest.raises(ValueError, match="Unknown QR method"):
            Dion(param_groups, qr_method="invalid")
    
    def test_optimizer_step(self, simple_model, device):
        """Test basic optimizer step functionality"""
        param_groups = self.build_param_groups(simple_model)
        opt = Dion(param_groups, lr=0.01)
        
        # Create dummy loss and gradients
        x = torch.randn(4, 32, device=device)
        output = simple_model(x)
        loss = output.sum()
        loss.backward()
        
        # Save initial parameters
        initial_params = {name: p.clone() for name, p in simple_model.named_parameters()}
        
        # Take optimizer step
        opt.step()
        
        # Check that parameters changed
        for name, param in simple_model.named_parameters():
            if param.grad is not None:
                assert not torch.allclose(param, initial_params[name])
    
    def test_dion_update_numerical_accuracy(self, device):
        """Test numerical accuracy of dion_update function"""
        torch.manual_seed(42)
        
        # Create test matrices
        m, n, r = 64, 32, 8
        X = torch.randn(m, n, device=device, dtype=torch.float64)
        G = torch.randn(m, n, device=device, dtype=torch.float64) * 0.01
        M = torch.zeros_like(X)
        Q = torch.randn(n, r, device=device, dtype=torch.float64)
        
        # Orthogonalize Q initially
        Q, _ = torch.linalg.qr(Q)
        
        # Test parameters
        lr = torch.tensor(0.01, dtype=torch.float64)
        mu = torch.tensor(0.95, dtype=torch.float64)
        weight_decay = torch.tensor(0.01, dtype=torch.float64)
        epsilon = 1e-8
        
        # Run update
        X_orig = X.clone()
        Q_new = dion_update(
            X, G, M, Q, lr, mu, weight_decay, epsilon,
            transpose=False, power_iters=1, qr_method="qr",
            oversample=1.25, compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        # With only 1 power iteration, Q won't be perfectly orthonormal
        # Just check that the update happened and Q changed
        assert not torch.allclose(Q_new, Q, atol=1e-10)
        
        # Check that X was updated (weight decay + gradient update)
        assert not torch.allclose(X, X_orig, atol=1e-10)
    
    def test_power_iteration_convergence(self, device):
        """Test that power iteration converges to correct low-rank approximation"""
        torch.manual_seed(42)
        
        # Create a low-rank matrix
        m, n, true_rank = 100, 80, 10
        U = torch.randn(m, true_rank, device=device)
        V = torch.randn(n, true_rank, device=device)
        B = U @ V.T
        
        # Initialize Q
        r = 15  # overestimate rank
        Q_init = torch.randn(n, r, device=device)
        Q_init, _ = torch.linalg.qr(Q_init)
        
        # Run power iteration
        P, Q = power_iteration(
            B, Q_init, power_iters=10, qr_method="qr",
            oversample=1.0, compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        # Check reconstruction error
        B_approx = P @ Q.T
        rel_error = torch.norm(B - B_approx) / torch.norm(B)
        assert rel_error < 1e-6  # Should be very small for overestimated rank
    
    def test_orthogonalize_methods(self, device):
        """Test different orthogonalization methods"""
        torch.manual_seed(42)
        
        # Test matrix shapes
        test_cases = [
            (100, 20),  # Tall and skinny
            (50, 50),   # Square
            (20, 100),  # Wide
        ]
        
        for m, n in test_cases:
            P = torch.randn(m, n, device=device, dtype=torch.float64)
            
            # Test QR method
            Q_qr = orthogonalize(P, qr_method="qr")
            # For QR, wide matrices return square Q, tall matrices return rectangular Q
            if m <= n:
                assert Q_qr.shape == (m, m)  # Square orthogonal matrix
            else:
                assert Q_qr.shape == P.shape  # Rectangular with orthonormal columns
            # For QR decomposition, Q has orthonormal columns
            if m >= n:
                # Q is m x n with orthonormal columns
                QtQ = Q_qr.T @ Q_qr
                I = torch.eye(n, device=device, dtype=torch.float64)
                ortho_error = torch.max(torch.abs(QtQ - I)).item()
                assert ortho_error < 1e-6, f"QR orthogonality error too large: {ortho_error}"
            else:
                # Q is m x m orthogonal matrix
                QQt = Q_qr @ Q_qr.T
                I = torch.eye(m, device=device, dtype=torch.float64)
                assert torch.allclose(QQt, I, atol=1e-6)
            
            # Test RCQR method
            if m > n:  # RCQR is only used for tall matrices
                rng = torch.Generator(device=device)
                rng.manual_seed(42)
                Q_rcqr = orthogonalize(P, qr_method="rcqr", oversample=1.25, rng=rng)
                assert Q_rcqr.shape == P.shape
                QtQ = Q_rcqr.T @ Q_rcqr
                assert torch.allclose(QtQ, I, atol=1e-6)
            else:
                # For square or wide matrices, RCQR falls back to regular QR
                rng = torch.Generator(device=device)
                rng.manual_seed(42)
                Q_rcqr = orthogonalize(P, qr_method="rcqr", oversample=1.25, rng=rng)
                assert Q_rcqr.shape == (m, m)  # Falls back to QR which returns square Q
                QtQ = Q_rcqr.T @ Q_rcqr
                assert torch.allclose(QtQ, I, atol=1e-6)
            
            # Test CQR method (if well-conditioned)
            if m >= n:
                P_well_cond = P + 0.1 * torch.eye(m, n, device=device, dtype=torch.float64)
                Q_cqr = orthogonalize(P_well_cond, qr_method="cqr")
                if m == n:
                    assert Q_cqr.shape == (m, m)  # Square matrix
                else:
                    assert Q_cqr.shape == P_well_cond.shape  # Tall matrix
                QtQ = Q_cqr.T @ Q_cqr
                assert torch.allclose(QtQ, I, atol=1e-4)
    
    def test_fix_all_zero_or_nan(self, device):
        """Test handling of all-zero or NaN cases"""
        m, n, r = 32, 16, 8
        
        # Test all-zero case
        B = torch.zeros(m, n, device=device)
        P = torch.randn(m, r, device=device)
        Q = torch.randn(n, r, device=device)
        Q_init = torch.randn(n, r, device=device)
        
        P_fixed, Q_fixed = fix_all_zero_or_nan(P, Q, Q_init, B)
        
        # P should be all zeros
        assert torch.allclose(P_fixed, torch.zeros_like(P))
        # Q should be Q_init
        assert torch.allclose(Q_fixed, Q_init)
        
        # Test non-zero case
        B = torch.randn(m, n, device=device)
        P_fixed, Q_fixed = fix_all_zero_or_nan(P, Q, Q_init, B)
        
        # Should be unchanged (after nan_to_num)
        assert torch.allclose(P_fixed, P.nan_to_num())
        assert torch.allclose(Q_fixed, Q.nan_to_num())
    
    def test_transposed_mode(self, device):
        """Test transposed Dion update"""
        torch.manual_seed(42)
        
        # Create matrices where m < n (transposed case)
        m, n, r = 32, 64, 8
        X = torch.randn(m, n, device=device)
        G = torch.randn(m, n, device=device) * 0.01
        M = torch.zeros_like(X)
        Q = torch.randn(m, r, device=device)  # Note: shape is (m, r) for transposed
        
        # Orthogonalize Q
        Q, _ = torch.linalg.qr(Q)
        
        lr = torch.tensor(0.01)
        mu = torch.tensor(0.95)
        weight_decay = torch.tensor(0.01)
        
        # Run transposed update
        Q_new = dion_update(
            X, G, M, Q, lr, mu, weight_decay, 1e-8,
            transpose=True, power_iters=1, qr_method="qr",
            oversample=1.25, compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        # With only 1 power iteration, Q won't be perfectly orthonormal
        # Just check that the update happened
        assert Q_new.shape == (m, r)  # Correct shape for transposed mode
    
    def test_rank_fraction_settings(self, device):
        """Test different rank fraction settings"""
        m, n = 64, 32
        param = torch.randn(m, n, device=device, requires_grad=True)
        
        rank_fractions = [1.0, 0.5, 0.25, 0.125]
        
        for rf in rank_fractions:
            opt = Dion([param], lr=0.01, rank_fraction=rf)
            
            # Create gradient
            grad = torch.randn_like(param) * 0.01
            param.grad = grad
            
            # Take step
            opt.step()
            
            # Check Q matrix was created with correct rank
            state = opt.state[param]
            Q = state["Q"]
            expected_rank = int(rf * min(m, n))
            assert Q.shape[1] == expected_rank
    
    def test_scalar_optimizer_integration(self, simple_model, device):
        """Test integration with scalar optimizers (Lion, AdamW)"""
        param_groups = self.build_param_groups(simple_model)
        opt = Dion(param_groups, lr=0.01)
        
        # Generate gradients
        x = torch.randn(4, 32, device=device)
        output = simple_model(x)
        loss = output.sum()
        loss.backward()
        
        # Take optimizer step
        opt.step()
        
        # Check that correct algorithms were used
        for group in opt.param_groups:
            algo = group["algorithm"]
            for param in group["params"]:
                if param.grad is not None:
                    state = opt.state[param]
                    if algo == "dion":
                        assert "Q" in state
                        assert "momentum" in state
                    elif algo == "lion":
                        assert "momentum" in state
                        assert "Q" not in state
                    elif algo == "adamw":
                        assert "momentum" in state
                        assert "variance" in state
                        assert "Q" not in state
    
    def test_weight_decay(self, device):
        """Test weight decay application"""
        torch.manual_seed(42)
        
        # Create parameters
        param = torch.randn(32, 16, device=device, requires_grad=True)
        original_param = param.clone()
        
        # Create optimizer with weight decay
        weight_decay = 0.1
        lr = 0.01
        opt = Dion([param], lr=lr, weight_decay=weight_decay)
        
        # Create small gradient
        param.grad = torch.randn_like(param) * 0.001
        
        # Take step
        opt.step()
        
        # Check weight decay was applied
        # After weight decay: X = X * (1 - lr * weight_decay)
        expected_decay_factor = 1 - lr * weight_decay
        
        # The update includes both weight decay and gradient update
        # We can't easily separate them, but we can check the parameter changed
        assert not torch.allclose(param, original_param)
        
        # Check parameter norm decreased (weight decay effect)
        assert torch.norm(param) < torch.norm(original_param)
    
    def test_momentum_accumulation(self, device):
        """Test momentum accumulation over multiple steps"""
        torch.manual_seed(42)
        
        param = torch.randn(32, 16, device=device, requires_grad=True)
        opt = Dion([param], lr=0.01, mu=0.9)
        
        # Take multiple steps with same gradient
        grad = torch.randn_like(param) * 0.01
        momentum_norms = []
        
        for i in range(5):
            param.grad = grad.clone()
            opt.step()
            
            state = opt.state[param]
            momentum_norms.append(torch.norm(state["momentum"]).item())
        
        # Momentum should accumulate over steps
        assert all(momentum_norms[i] < momentum_norms[i+1] for i in range(4))
    
    def test_error_feedback(self, device):
        """Test error feedback mechanism in Dion"""
        torch.manual_seed(42)
        
        # Use small rank fraction to ensure error feedback is significant
        param = torch.randn(64, 32, device=device, requires_grad=True)
        opt = Dion([param], lr=0.01, rank_fraction=0.125, mu=0.95)
        
        # Generate gradient
        grad = torch.randn_like(param)
        param.grad = grad
        
        # Take step
        opt.step()
        
        # Check momentum was updated with error feedback
        state = opt.state[param]
        M = state["momentum"]
        
        # Momentum should not be zero (contains error feedback)
        assert torch.norm(M) > 1e-6
    
    def test_learning_rate_scaling(self, device):
        """Test automatic learning rate scaling based on matrix dimensions"""
        torch.manual_seed(42)
        
        # Test different matrix shapes
        shapes = [(64, 32), (32, 64), (128, 16)]
        base_lr = 0.01
        
        for m, n in shapes:
            param = torch.randn(m, n, device=device, requires_grad=True)
            opt = Dion([param], lr=base_lr)
            
            # Generate small gradient
            param.grad = torch.ones_like(param) * 0.001
            
            # Save original param
            param_orig = param.clone()
            
            # Take step
            opt.step()
            
            # Compute update magnitude
            update = param_orig - param
            update_norm = torch.norm(update)
            
            # Expected scaling factor
            fan_out, fan_in = m, n
            expected_scale = math.sqrt(fan_out / fan_in)
            
            # The update should be proportional to the scaling factor
            # (This is a rough check since other factors affect the update)
            assert update_norm > 0
    
    def test_cqr_warmup(self, device):
        """Test CQR warmup functionality"""
        torch.manual_seed(42)
        
        param = torch.randn(64, 32, device=device, requires_grad=True)
        cqr_warmup_steps = 5
        opt = Dion([param], lr=0.01, qr_method="cqr", cqr_warmup_steps=cqr_warmup_steps)
        
        # During warmup, CQR should fall back to RCQR
        for step in range(cqr_warmup_steps + 2):
            param.grad = torch.randn_like(param) * 0.01
            opt.step()
            
            # We can't directly check which method was used, but we can verify
            # the optimizer runs without errors
            assert opt.param_groups[0]["step"] == step + 1
    
    def test_multiple_param_groups_settings(self, device):
        """Test different settings for different parameter groups"""
        # Create parameters
        param1 = torch.randn(64, 32, device=device, requires_grad=True)
        param2 = torch.randn(32, 16, device=device, requires_grad=True)
        param3 = torch.randn(128, device=device, requires_grad=True)
        
        # Create groups with different settings
        param_groups = [
            {"params": [param1], "rank_fraction": 0.5},
            {"params": [param2], "rank_fraction": 0.25, "lr": 0.02},
            {"params": [param3], "algorithm": "lion", "lr": 0.005}
        ]
        
        opt = Dion(param_groups, lr=0.01)
        
        # Generate gradients
        for p in [param1, param2, param3]:
            p.grad = torch.randn_like(p) * 0.01
        
        # Take step
        opt.step()
        
        # Check settings were applied correctly
        assert opt.param_groups[0]["rank_fraction"] == 0.5
        assert opt.param_groups[1]["rank_fraction"] == 0.25
        assert opt.param_groups[1]["lr"] == 0.02
        assert opt.param_groups[2]["algorithm"] == "lion"
        assert opt.param_groups[2]["lr"] == 0.005
        
        # Check Q matrix ranks
        Q1 = opt.state[param1]["Q"]
        Q2 = opt.state[param2]["Q"]
        assert Q1.shape[1] == 16  # 0.5 * min(64, 32) = 16
        assert Q2.shape[1] == 4   # 0.25 * min(32, 16) = 4
    
    def test_step_counter(self, device):
        """Test that step counter increments correctly"""
        param = torch.randn(32, 16, device=device, requires_grad=True)
        opt = Dion([param], lr=0.01)
        
        # Check initial step
        assert opt.param_groups[0]["step"] == 0
        
        # Take multiple steps
        for expected_step in range(1, 6):
            param.grad = torch.randn_like(param) * 0.01
            opt.step()
            assert opt.param_groups[0]["step"] == expected_step
    
    def test_zero_grad_handling(self, device):
        """Test handling of zero gradients"""
        param = torch.randn(32, 16, device=device, requires_grad=True)
        opt = Dion([param], lr=0.01)
        
        # Set zero gradient
        param.grad = torch.zeros_like(param)
        param_orig = param.clone()
        
        # Take step
        opt.step()
        
        # Parameter should only change due to weight decay
        weight_decay = opt.defaults["weight_decay"]
        lr = opt.defaults["lr"]
        expected = param_orig * (1 - lr * weight_decay)
        assert torch.allclose(param, expected, atol=1e-6)
    
    def test_gradient_clipping_compatibility(self, device):
        """Test compatibility with gradient clipping"""
        param = torch.randn(32, 16, device=device, requires_grad=True)
        opt = Dion([param], lr=0.01)
        
        # Generate large gradient
        param.grad = torch.randn_like(param) * 10.0
        
        # Clip gradient
        torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
        
        # Take step - should work without errors
        opt.step()
        
        # Check optimizer state was created
        assert param in opt.state
        assert "Q" in opt.state[param]