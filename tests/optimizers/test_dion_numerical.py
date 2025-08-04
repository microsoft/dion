import pytest
import torch
import numpy as np
from typing import Tuple
import math

from optimizers.dion_reference import (
    dion_update, power_iteration, orthogonalize,
    fix_all_zero_or_nan
)


class TestDionNumericalAccuracy:
    """Test numerical accuracy and stability of Dion optimizer components"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_orthogonalization_stability(self, device):
        """Test numerical stability of orthogonalization methods"""
        torch.manual_seed(42)
        
        # Test with ill-conditioned matrices
        n = 50
        # Create matrix with large condition number
        U, S, Vt = torch.linalg.svd(torch.randn(n, n, device=device))
        S_modified = torch.logspace(0, -10, n, device=device)  # Condition number ~1e10
        A = U @ torch.diag(S_modified) @ Vt
        
        # Test each method
        methods = ["qr", "rcqr"]
        for method in methods:
            if method == "rcqr":
                rng = torch.Generator(device=device).manual_seed(42)
                Q = orthogonalize(A, qr_method=method, rng=rng)
            else:
                Q = orthogonalize(A, qr_method=method)
            
            # Check orthogonality
            QtQ = Q.T @ Q
            I = torch.eye(n, device=device)
            ortho_error = torch.norm(QtQ - I, p='fro')
            
            # RCQR and QR should maintain reasonable orthogonality even for ill-conditioned inputs
            assert ortho_error < 1e-5, f"{method} failed orthogonality test with error {ortho_error}"
    
    def test_power_iteration_accuracy(self, device):
        """Test accuracy of power iteration for different matrix types"""
        torch.manual_seed(42)
        
        test_cases = [
            # (name, matrix_generator, expected_error)
            ("low_rank", self._create_low_rank_matrix, 1e-10),
            ("full_rank", self._create_full_rank_matrix, 1e-2),
            ("noisy_low_rank", self._create_noisy_low_rank_matrix, 1e-3),
        ]
        
        for name, matrix_gen, expected_error in test_cases:
            m, n, r = 100, 80, 10
            B = matrix_gen(m, n, r, device)
            
            # Initialize Q
            Q_init = torch.randn(n, r, device=device, dtype=torch.float64)
            Q_init, _ = torch.linalg.qr(Q_init)
            
            # Run power iteration
            P, Q = power_iteration(
                B, Q_init, power_iters=20, qr_method="qr",
                oversample=1.0, compressed_all_reduce=False,
                replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
            )
            
            # Check reconstruction error
            B_approx = P @ Q.T
            rel_error = torch.norm(B - B_approx, p='fro') / torch.norm(B, p='fro')
            
            assert rel_error < expected_error, f"{name}: relative error {rel_error} > {expected_error}"
    
    def _create_low_rank_matrix(self, m: int, n: int, r: int, device: torch.device) -> torch.Tensor:
        """Create exact low-rank matrix"""
        U = torch.randn(m, r, device=device, dtype=torch.float64)
        V = torch.randn(n, r, device=device, dtype=torch.float64)
        U, _ = torch.linalg.qr(U)
        V, _ = torch.linalg.qr(V)
        S = torch.diag(torch.linspace(10, 1, r, device=device, dtype=torch.float64))
        return U @ S @ V.T
    
    def _create_full_rank_matrix(self, m: int, n: int, r: int, device: torch.device) -> torch.Tensor:
        """Create full-rank matrix"""
        return torch.randn(m, n, device=device, dtype=torch.float64)
    
    def _create_noisy_low_rank_matrix(self, m: int, n: int, r: int, device: torch.device) -> torch.Tensor:
        """Create low-rank matrix with noise"""
        low_rank = self._create_low_rank_matrix(m, n, r, device)
        noise = torch.randn(m, n, device=device, dtype=torch.float64) * 0.01
        return low_rank + noise
    
    def test_gradient_accumulation_precision(self, device):
        """Test precision of gradient accumulation in momentum"""
        torch.manual_seed(42)
        
        # Use double precision for testing
        m, n, r = 32, 16, 4
        X = torch.randn(m, n, device=device, dtype=torch.float64)
        M = torch.zeros_like(X)
        Q = torch.randn(n, r, device=device, dtype=torch.float64)
        Q, _ = torch.linalg.qr(Q)
        
        # Accumulate many small gradients
        num_steps = 100
        grad_scale = 1e-6
        
        for i in range(num_steps):
            G = torch.randn_like(X) * grad_scale
            
            # Manual momentum update for comparison
            M_expected = M.clone()
            M_expected.add_(G)
            
            # Run dion update
            Q = dion_update(
                X.clone(), G, M, Q,
                lr=torch.tensor(0.0, dtype=torch.float64),  # No weight update
                mu=torch.tensor(1.0, dtype=torch.float64),  # No error feedback
                weight_decay=torch.tensor(0.0, dtype=torch.float64),
                epsilon=1e-8, transpose=False, power_iters=1,
                qr_method="qr", compressed_all_reduce=False,
                replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
            )
            
            # Check momentum accumulation is accurate
            assert torch.allclose(M, M_expected, atol=1e-14)
    
    def test_error_feedback_accuracy(self, device):
        """Test accuracy of error feedback mechanism"""
        torch.manual_seed(42)
        
        m, n, r = 64, 32, 4  # Very low rank
        X = torch.randn(m, n, device=device, dtype=torch.float64)
        G = torch.randn(m, n, device=device, dtype=torch.float64) * 0.1
        M = G.clone()  # Start with gradient as momentum
        Q = torch.randn(n, r, device=device, dtype=torch.float64)
        Q, _ = torch.linalg.qr(Q)
        
        mu = 0.9
        
        # Compute low-rank approximation manually
        P_manual = M @ Q
        M_approx = P_manual @ Q.T
        error = M - M_approx
        M_after_feedback = M - (1 - mu) * M_approx
        
        # Run dion update
        Q_new = dion_update(
            X.clone(), torch.zeros_like(G), M, Q,
            lr=torch.tensor(0.0, dtype=torch.float64),
            mu=torch.tensor(mu, dtype=torch.float64),
            weight_decay=torch.tensor(0.0, dtype=torch.float64),
            epsilon=1e-8, transpose=False, power_iters=1,
            qr_method="qr", compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        # Check error feedback was applied correctly
        assert torch.allclose(M, M_after_feedback, atol=1e-10)
    
    def test_learning_rate_scaling_precision(self, device):
        """Test precision of learning rate scaling"""
        test_shapes = [
            (128, 64),
            (64, 128),
            (256, 32),
            (32, 256),
        ]
        
        for m, n in test_shapes:
            X = torch.eye(m, n, device=device, dtype=torch.float64)  # Identity for easy tracking
            G = torch.zeros_like(X)
            M = torch.zeros_like(X)
            r = min(m, n) // 2
            Q = torch.randn(n, r, device=device, dtype=torch.float64)
            Q, _ = torch.linalg.qr(Q)
            
            # Create simple update pattern
            P = torch.ones(m, r, device=device, dtype=torch.float64)
            M.copy_(P @ Q.T)
            
            base_lr = 1.0  # Use 1.0 to clearly see scaling
            
            # Run update
            X_before = X.clone()
            Q_new = dion_update(
                X, G, M, Q,
                lr=torch.tensor(base_lr, dtype=torch.float64),
                mu=torch.tensor(0.0, dtype=torch.float64),
                weight_decay=torch.tensor(0.0, dtype=torch.float64),
                epsilon=1e-8, transpose=False, power_iters=0,  # Skip power iteration
                qr_method="qr", compressed_all_reduce=False,
                replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
            )
            
            # Check scaling factor
            update = X_before - X
            expected_scale = math.sqrt(m / n)
            
            # The update magnitude should match the scaling
            update_scale = torch.abs(update).max().item()
            assert abs(update_scale - expected_scale * base_lr) < 1e-10
    
    def test_weight_decay_precision(self, device):
        """Test precision of weight decay application"""
        torch.manual_seed(42)
        
        X = torch.randn(32, 16, device=device, dtype=torch.float64) * 10  # Large weights
        G = torch.zeros_like(X)
        M = torch.zeros_like(X)
        Q = torch.randn(16, 4, device=device, dtype=torch.float64)
        Q, _ = torch.linalg.qr(Q)
        
        lr = 0.1
        weight_decay = 0.01
        
        X_before = X.clone()
        
        # Run update with only weight decay
        Q_new = dion_update(
            X, G, M, Q,
            lr=torch.tensor(lr, dtype=torch.float64),
            mu=torch.tensor(1.0, dtype=torch.float64),
            weight_decay=torch.tensor(weight_decay, dtype=torch.float64),
            epsilon=1e-8, transpose=False, power_iters=1,
            qr_method="qr", compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        # Check weight decay was applied exactly
        expected = X_before * (1 - lr * weight_decay)
        assert torch.allclose(X, expected, atol=1e-14)
    
    def test_mixed_precision_consistency(self, device):
        """Test consistency across different precision settings"""
        torch.manual_seed(42)
        
        # Create test data
        m, n, r = 32, 16, 4
        X_f32 = torch.randn(m, n, device=device, dtype=torch.float32)
        X_f64 = X_f32.to(torch.float64)
        
        G_f32 = torch.randn_like(X_f32) * 0.01
        G_f64 = G_f32.to(torch.float64)
        
        M_f32 = torch.zeros_like(X_f32)
        M_f64 = torch.zeros_like(X_f64)
        
        Q_f32 = torch.randn(n, r, device=device, dtype=torch.float32)
        Q_f32, _ = torch.linalg.qr(Q_f32)
        Q_f64 = Q_f32.to(torch.float64)
        
        # Common parameters
        lr = torch.tensor(0.01)
        mu = torch.tensor(0.95)
        weight_decay = torch.tensor(0.01)
        
        # Run updates in both precisions
        Q_new_f32 = dion_update(
            X_f32, G_f32, M_f32, Q_f32,
            lr.to(torch.float32), mu.to(torch.float32), 
            weight_decay.to(torch.float32),
            epsilon=1e-8, transpose=False, power_iters=1,
            qr_method="qr", compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        Q_new_f64 = dion_update(
            X_f64, G_f64, M_f64, Q_f64,
            lr.to(torch.float64), mu.to(torch.float64),
            weight_decay.to(torch.float64),
            epsilon=1e-8, transpose=False, power_iters=1,
            qr_method="qr", compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        # Check results are consistent (within float32 precision)
        assert torch.allclose(X_f32, X_f64.to(torch.float32), atol=1e-5, rtol=1e-5)
        assert torch.allclose(Q_new_f32, Q_new_f64.to(torch.float32), atol=1e-5, rtol=1e-5)
    
    def test_zero_gradient_edge_case(self, device):
        """Test behavior with zero gradients"""
        m, n, r = 16, 8, 4
        X = torch.randn(m, n, device=device)
        G = torch.zeros_like(X)  # Zero gradient
        M = torch.randn_like(X) * 0.1  # Non-zero momentum
        Q = torch.randn(n, r, device=device)
        Q, _ = torch.linalg.qr(Q)
        
        X_before = X.clone()
        M_before = M.clone()
        
        # Run update
        Q_new = dion_update(
            X, G, M, Q,
            lr=torch.tensor(0.01), mu=torch.tensor(0.95),
            weight_decay=torch.tensor(0.0),  # No weight decay to isolate effect
            epsilon=1e-8, transpose=False, power_iters=1,
            qr_method="qr", compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        # Momentum should be unchanged (only adds zero gradient)
        assert torch.allclose(M, M_before)
        
        # Weight update should still happen based on existing momentum
        assert not torch.allclose(X, X_before)
    
    def test_extreme_learning_rates(self, device):
        """Test stability with extreme learning rates"""
        torch.manual_seed(42)
        
        X = torch.randn(32, 16, device=device)
        G = torch.randn_like(X) * 0.01
        M = torch.zeros_like(X)
        Q = torch.randn(16, 4, device=device)
        Q, _ = torch.linalg.qr(Q)
        
        # Test very small and very large learning rates
        test_lrs = [1e-10, 1e-5, 1e-1, 1.0, 10.0]
        
        for lr in test_lrs:
            X_test = X.clone()
            M_test = M.clone()
            Q_test = Q.clone()
            
            # Should not produce NaN or Inf
            Q_new = dion_update(
                X_test, G, M_test, Q_test,
                lr=torch.tensor(lr), mu=torch.tensor(0.95),
                weight_decay=torch.tensor(0.0),
                epsilon=1e-8, transpose=False, power_iters=1,
                qr_method="qr", compressed_all_reduce=False,
                replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
            )
            
            assert torch.isfinite(X_test).all(), f"NaN/Inf in X with lr={lr}"
            assert torch.isfinite(Q_new).all(), f"NaN/Inf in Q with lr={lr}"
            assert torch.isfinite(M_test).all(), f"NaN/Inf in M with lr={lr}"
    
    def test_rank_deficient_matrices(self, device):
        """Test handling of rank-deficient matrices"""
        torch.manual_seed(42)
        
        # Create rank-deficient matrix
        m, n, true_rank = 32, 16, 4
        U = torch.randn(m, true_rank, device=device)
        V = torch.randn(n, true_rank, device=device)
        M = U @ V.T  # Rank 4 matrix
        
        # Try to approximate with higher rank
        r = 8
        Q_init = torch.randn(n, r, device=device)
        Q_init, _ = torch.linalg.qr(Q_init)
        
        # Power iteration should still work
        P, Q = power_iteration(
            M, Q_init, power_iters=10, qr_method="qr",
            oversample=1.0, compressed_all_reduce=False,
            replicate_mesh=None, inner_shard_mesh_dim=None, rng=None
        )
        
        # Check that approximation captures the true rank
        M_approx = P @ Q.T
        assert torch.allclose(M, M_approx, atol=1e-6)
        
        # Check effective rank of result
        _, S, _ = torch.linalg.svd(P)
        effective_rank = (S > 1e-6).sum().item()
        assert effective_rank <= true_rank + 1  # Allow small numerical error