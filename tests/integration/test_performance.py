"""Performance tests for optimizer implementations."""

import pytest
import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple
import numpy as np

# Import optimizers
from optimizers.dion_reference import Dion as DionReference
from optimizers.scalar_opts import Lion, AdamW

# Try to import optional optimizers
try:
    from optimizers.dion import Dion as DionOptimized
    HAS_DION_OPTIMIZED = True
except ImportError:
    HAS_DION_OPTIMIZED = False
    DionOptimized = None


class PerformanceModel(nn.Module):
    """Model for performance testing with configurable size."""
    def __init__(self, layers: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1], bias=False))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.integration
@pytest.mark.performance
class TestPerformance:
    """Performance tests for optimizer implementations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def benchmark_optimizer_step(
        self, 
        optimizer_class, 
        model: nn.Module, 
        device: torch.device,
        num_steps: int = 100,
        warmup_steps: int = 10,
        **optimizer_kwargs
    ) -> Dict[str, float]:
        """Benchmark optimizer step time."""
        # Create optimizer
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        
        # Warmup
        for _ in range(warmup_steps):
            # Generate gradient
            x = torch.randn(32, model.layers[0].in_features, device=device)
            loss = model(x).sum()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Synchronize before timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Time the steps
        step_times = []
        for _ in range(num_steps):
            # Generate gradient
            x = torch.randn(32, model.layers[0].in_features, device=device)
            loss = model(x).sum()
            loss.backward()
            
            # Time the step
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            optimizer.step()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            step_times.append(end_time - start_time)
            optimizer.zero_grad()
        
        return {
            "mean_time": np.mean(step_times),
            "std_time": np.std(step_times),
            "min_time": np.min(step_times),
            "max_time": np.max(step_times),
            "median_time": np.median(step_times),
        }
    
    def test_dion_scaling_with_dimension(self, device):
        """Test how Dion performance scales with matrix dimensions."""
        if device.type != "cuda":
            pytest.skip("Performance test requires CUDA")
        
        dimensions = [
            [512, 512],
            [1024, 1024],
            [2048, 2048],
            [4096, 4096],
        ]
        
        results = {}
        
        for dims in dimensions:
            model = PerformanceModel(dims).to(device)
            
            # Test reference implementation
            ref_stats = self.benchmark_optimizer_step(
                DionReference, model, device, 
                lr=0.01, rank_fraction=0.25
            )
            
            dim_str = f"{dims[0]}x{dims[1]}"
            results[f"DionReference_{dim_str}"] = ref_stats["mean_time"]
            
            # Test optimized if available
            if HAS_DION_OPTIMIZED:
                opt_stats = self.benchmark_optimizer_step(
                    DionOptimized, model, device,
                    lr=0.01, rank_fraction=0.25
                )
                results[f"DionOptimized_{dim_str}"] = opt_stats["mean_time"]
        
        # Print results
        print("\nDion Scaling Results:")
        for key, time_ms in results.items():
            print(f"{key}: {time_ms*1000:.3f}ms")
        
        # Optimized should be faster for large dimensions
        if HAS_DION_OPTIMIZED:
            assert results["DionOptimized_4096x4096"] < results["DionReference_4096x4096"] * 1.5
    
    def test_rank_fraction_impact(self, device):
        """Test performance impact of different rank fractions."""
        if device.type != "cuda":
            pytest.skip("Performance test requires CUDA")
        
        model = PerformanceModel([2048, 2048]).to(device)
        rank_fractions = [0.125, 0.25, 0.5, 1.0]
        
        results = {}
        
        for rf in rank_fractions:
            stats = self.benchmark_optimizer_step(
                DionReference, model, device,
                lr=0.01, rank_fraction=rf, num_steps=50
            )
            results[rf] = stats["mean_time"]
        
        # Print results
        print("\nRank Fraction Impact:")
        for rf, time_ms in results.items():
            print(f"rank_fraction={rf}: {time_ms*1000:.3f}ms")
        
        # Lower rank should be faster
        assert results[0.125] < results[1.0]
    
    @pytest.mark.skipif(not HAS_DION_OPTIMIZED, reason="DionOptimized not available")
    def test_dion_optimized_speedup(self, device):
        """Test speedup of optimized Dion implementation."""
        if device.type != "cuda":
            pytest.skip("Performance test requires CUDA")
        
        # Test on various model sizes
        model_configs = [
            ([1024, 1024], "small"),
            ([2048, 2048, 2048], "medium"),
            ([4096, 2048, 4096], "large"),
        ]
        
        for layers, name in model_configs:
            model_ref = PerformanceModel(layers).to(device)
            model_opt = PerformanceModel(layers).to(device)
            model_opt.load_state_dict(model_ref.state_dict())
            
            # Benchmark reference
            ref_stats = self.benchmark_optimizer_step(
                DionReference, model_ref, device,
                lr=0.01, rank_fraction=0.25
            )
            
            # Benchmark optimized
            opt_stats = self.benchmark_optimizer_step(
                DionOptimized, model_opt, device,
                lr=0.01, rank_fraction=0.25
            )
            
            speedup = ref_stats["mean_time"] / opt_stats["mean_time"]
            
            print(f"\n{name} model speedup: {speedup:.2f}x")
            print(f"  Reference: {ref_stats['mean_time']*1000:.3f}ms")
            print(f"  Optimized: {opt_stats['mean_time']*1000:.3f}ms")
            
            # Should see some speedup
            assert speedup > 0.8, f"Optimized version slower for {name} model"
    
    def test_memory_efficiency(self, device):
        """Test memory usage of different optimizers."""
        if device.type != "cuda":
            pytest.skip("Memory profiling requires CUDA")
        
        # Large model to make memory usage significant
        model = PerformanceModel([4096, 4096, 4096]).to(device)
        
        optimizer_configs = [
            (DionReference, {"lr": 0.01, "rank_fraction": 0.25}, "Dion(rf=0.25)"),
            (DionReference, {"lr": 0.01, "rank_fraction": 1.0}, "Dion(rf=1.0)"),
            (AdamW, {"lr": 0.001}, "AdamW"),
            (Lion, {"lr": 0.001}, "Lion"),
        ]
        
        results = {}
        
        for opt_class, kwargs, name in optimizer_configs:
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Create optimizer
            optimizer = opt_class(model.parameters(), **kwargs)
            
            # Do some steps to allocate state
            for _ in range(5):
                x = torch.randn(32, 4096, device=device)
                loss = model(x).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Get memory usage
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            results[name] = peak_memory
            
            # Cleanup
            del optimizer
            torch.cuda.empty_cache()
        
        # Print results
        print("\nMemory Usage (GB):")
        for name, memory_gb in results.items():
            print(f"{name}: {memory_gb:.3f} GB")
        
        # Dion with low rank should use less memory than AdamW
        assert results["Dion(rf=0.25)"] < results["AdamW"]
        
        # Lion should be most memory efficient (only momentum)
        assert results["Lion"] < results["AdamW"]
    
    def test_batch_processing_efficiency(self, device):
        """Test efficiency of batch processing in optimizers."""
        if device.type != "cuda":
            pytest.skip("Performance test requires CUDA")
        
        # Create multiple small models
        num_models = 10
        models = [PerformanceModel([512, 512]).to(device) for _ in range(num_models)]
        
        # Test batched vs sequential processing
        # Sequential
        start_time = time.perf_counter()
        for model in models:
            # Separate matrix parameters (2D) from vector parameters (1D)
            matrix_params = [p for p in model.parameters() if p.ndim == 2]  
            vector_params = [p for p in model.parameters() if p.ndim != 2]
            
            param_groups = [
                dict(params=matrix_params),  # uses dion algorithm by default
                dict(params=vector_params, algorithm="lion")  # use lion for 1D params
            ]
            
            opt = DionReference(param_groups, lr=0.01)
            for _ in range(10):
                x = torch.randn(32, 512, device=device)
                loss = model(x).sum()
                loss.backward()
                opt.step()
                opt.zero_grad()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        sequential_time = time.perf_counter() - start_time
        
        print(f"\nSequential processing time: {sequential_time:.3f}s")
        
        # Note: True batched optimizer processing would require
        # specialized implementations not currently available