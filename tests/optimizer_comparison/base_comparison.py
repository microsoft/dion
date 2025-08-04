"""Base class for optimizer comparison tests with shared utilities."""

import torch
import torch.nn as nn
from typing import Dict
import pytest


class BaseOptimizerComparison:
    """Base class with common utilities for optimizer comparison tests."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def create_simple_model(self, device):
        """Create a simple model for testing"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 128, bias=False)
                self.linear2 = nn.Linear(128, 64, bias=False)
            
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x
                
        model = SimpleModel().to(device)
        # Initialize with same weights for reproducibility
        torch.manual_seed(42)
        for p in model.parameters():
            nn.init.xavier_uniform_(p)
        return model
    
    def create_mixed_model(self, device):
        """Create a model with different parameter types"""
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 16, bias=True)
                self.embedding = nn.Embedding(100, 32)
                self.norm = nn.LayerNorm(16)
                
            def forward(self, x_indices):
                x = self.embedding(x_indices)
                x = self.linear(x)
                x = self.norm(x)
                return x
                
        return MixedModel().to(device)
    
    def generate_gradients(self, model: nn.Module, device: torch.device, seed: int = 42):
        """Generate consistent gradients for testing"""
        torch.manual_seed(seed)
        
        if hasattr(model, 'embedding'):
            # For models with embeddings
            x = torch.randint(0, 100, (16,), device=device)
        else:
            # For linear models
            x = torch.randn(32, 64, device=device)
            
        out = model(x)
        loss = out.sum()
        loss.backward()
    
    def get_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get a copy of model parameters"""
        return {name: p.clone().detach() for name, p in model.named_parameters()}
    
    def compare_model_states(self, state1: Dict[str, torch.Tensor], 
                           state2: Dict[str, torch.Tensor], 
                           rtol: float = 1e-5, atol: float = 1e-6) -> bool:
        """Compare two model states"""
        for name in state1:
            if not torch.allclose(state1[name], state2[name], rtol=rtol, atol=atol):
                diff = torch.abs(state1[name] - state2[name]).max().item()
                rel_diff = (torch.abs(state1[name] - state2[name]) / 
                           (torch.abs(state1[name]) + 1e-8)).max().item()
                print(f"Mismatch in {name}: max_diff={diff}, max_rel_diff={rel_diff}")
                return False
        return True
    
    def build_param_groups_for_mixed_model(self, model):
        """Build parameter groups for mixed model"""
        matrix_params = []
        scalar_params = []
        
        for name, param in model.named_parameters():
            if param.ndim == 2 and 'embedding' not in name:
                matrix_params.append(param)
            else:
                scalar_params.append(param)
        
        groups = []
        if matrix_params:
            groups.append({"params": matrix_params})
        if scalar_params:
            groups.append({"params": scalar_params, "algorithm": "lion"})
            
        return groups