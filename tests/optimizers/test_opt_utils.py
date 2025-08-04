import pytest
import torch
from torch.distributed.tensor import DTensor, init_device_mesh, Shard, Replicate
from typing import List

from optimizers.opt_utils import (
    to_local, dtensor_from_local, create_param_batches,
    pad_batch, AsyncTask, AsyncRuntime
)


class TestOptUtils:
    """Test optimizer utility functions"""
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_to_local_single_tensor(self, device):
        """Test to_local with single tensor"""
        # Regular tensor - should return as-is
        tensor = torch.randn(4, 4, device=device)
        result = to_local(tensor)
        assert result is tensor
        
        # List of regular tensors
        tensors = [torch.randn(4, 4, device=device) for _ in range(3)]
        results = to_local(tensors)
        assert all(r is t for r, t in zip(results, tensors))
    
    def test_create_param_batches(self, device):
        """Test parameter batching by shape, sharding, and dtype"""
        # Create parameters with different properties
        params = [
            # Same shape and dtype
            torch.randn(32, 16, device=device, dtype=torch.float32),
            torch.randn(32, 16, device=device, dtype=torch.float32),
            torch.randn(32, 16, device=device, dtype=torch.float32),
            # Different shape
            torch.randn(64, 32, device=device, dtype=torch.float32),
            torch.randn(64, 32, device=device, dtype=torch.float32),
            # Different dtype
            torch.randn(32, 16, device=device, dtype=torch.float64),
            # Single parameter group
            torch.randn(128, 64, device=device, dtype=torch.float32),
        ]
        
        batch_size = 2
        batches = list(create_param_batches(params, batch_size))
        
        # Should create 4 batches:
        # - 2 batches for first 3 params (32,16,float32)
        # - 1 batch for next 2 params (64,32,float32)
        # - 1 batch for float64 param
        # - 1 batch for single param
        assert len(batches) == 5
        
        # Check batch sizes
        assert len(batches[0]) == 2  # First two (32,16,float32)
        assert len(batches[1]) == 1  # Last one (32,16,float32)
        assert len(batches[2]) == 2  # Both (64,32,float32)
        assert len(batches[3]) == 1  # The float64 one
        assert len(batches[4]) == 1  # The single (128,64)
        
        # Check all params in same batch have same properties
        for batch in batches:
            if len(batch) > 1:
                first = batch[0]
                for param in batch[1:]:
                    assert param.shape == first.shape
                    assert param.dtype == first.dtype
    
    def test_pad_batch(self, device):
        """Test batch padding functionality"""
        # Create initial batch
        batch = [torch.randn(16, 8, device=device) for _ in range(3)]
        target_size = 5
        
        # Pad batch
        padded = pad_batch(batch, target_size)
        
        assert len(padded) == target_size
        
        # First 3 should be original tensors
        for i in range(3):
            assert padded[i] is batch[i]
        
        # Last 2 should be dummy tensors with same shape
        for i in range(3, 5):
            assert padded[i].shape == batch[0].shape
            assert padded[i].device == batch[0].device
            assert padded[i].dtype == batch[0].dtype
    
    def test_async_task_basic(self):
        """Test basic AsyncTask functionality"""
        # Create a simple generator
        counter = 0
        
        def task_generator():
            nonlocal counter
            counter += 1
            yield
            counter += 1
            yield
            counter += 1
        
        task = AsyncTask(task_generator())
        
        # First step already ran in __init__
        assert counter == 1
        
        # Run next step
        still_running = task.run()
        assert still_running
        assert counter == 2
        
        # Run final step
        still_running = task.run()
        assert not still_running
        assert counter == 3
        
        # Further runs should return False
        still_running = task.run()
        assert not still_running
        assert counter == 3
    
    def test_async_runtime_sequential(self):
        """Test AsyncRuntime with sequential tasks"""
        results = []
        
        def create_task(task_id):
            def task_gen():
                results.append(f"task{task_id}_step1")
                yield
                results.append(f"task{task_id}_step2")
                yield
                results.append(f"task{task_id}_done")
            return AsyncTask(task_gen())
        
        # Generator that creates tasks
        def task_generator():
            for i in range(3):
                yield create_task(i)
        
        runtime = AsyncRuntime(task_generator(), max_concurrent_tasks=1)
        runtime.run()
        
        # With max_concurrent_tasks=1, tasks should run sequentially
        expected = [
            "task0_step1", "task0_step2", "task0_done",
            "task1_step1", "task1_step2", "task1_done",
            "task2_step1", "task2_step2", "task2_done",
        ]
        assert results == expected
    
    def test_async_runtime_concurrent(self):
        """Test AsyncRuntime with concurrent tasks"""
        results = []
        
        def create_task(task_id):
            def task_gen():
                results.append((task_id, "start"))
                yield
                results.append((task_id, "middle"))
                yield
                results.append((task_id, "end"))
            return AsyncTask(task_gen())
        
        def task_generator():
            for i in range(3):
                yield create_task(i)
        
        runtime = AsyncRuntime(task_generator(), max_concurrent_tasks=2)
        runtime.run()
        
        # With max_concurrent_tasks=2, first two tasks should interleave
        # Check that task 1 starts before task 0 ends
        task0_start = results.index((0, "start"))
        task0_end = results.index((0, "end"))
        task1_start = results.index((1, "start"))
        
        assert task1_start < task0_end
        
        # All tasks should complete
        for i in range(3):
            assert (i, "start") in results
            assert (i, "middle") in results
            assert (i, "end") in results
    
    def test_async_runtime_error_handling(self):
        """Test AsyncRuntime with invalid max_concurrent_tasks"""
        def dummy_generator():
            yield
        
        with pytest.raises(ValueError, match="cannot be <= 0"):
            AsyncRuntime(dummy_generator(), max_concurrent_tasks=0)
        
        with pytest.raises(ValueError, match="cannot be <= 0"):
            AsyncRuntime(dummy_generator(), max_concurrent_tasks=-1)
    
    def test_empty_batch_handling(self, device):
        """Test handling of empty parameter lists"""
        # Empty parameter list
        params = []
        batches = list(create_param_batches(params, batch_size=2))
        assert len(batches) == 0
        
        # Single parameter
        params = [torch.randn(10, 10, device=device)]
        batches = list(create_param_batches(params, batch_size=2))
        assert len(batches) == 1
        assert len(batches[0]) == 1
    
    def test_batch_grouping_complex(self, device):
        """Test complex parameter grouping scenarios"""
        # Create parameters with various combinations
        params = []
        
        # Group 1: (32, 16), float32 - 5 params
        for _ in range(5):
            params.append(torch.randn(32, 16, device=device, dtype=torch.float32))
        
        # Group 2: (32, 16), float64 - 3 params
        for _ in range(3):
            params.append(torch.randn(32, 16, device=device, dtype=torch.float64))
        
        # Group 3: (16, 32), float32 - 4 params
        for _ in range(4):
            params.append(torch.randn(16, 32, device=device, dtype=torch.float32))
        
        batch_size = 3
        batches = list(create_param_batches(params, batch_size))
        
        # Should create:
        # - 2 batches for group 1 (3 + 2)
        # - 1 batch for group 2 (3)
        # - 2 batches for group 3 (3 + 1)
        assert len(batches) == 5
        
        # Verify batch contents
        batch_idx = 0
        # Group 1 batches
        assert len(batches[batch_idx]) == 3
        assert all(p.shape == (32, 16) and p.dtype == torch.float32 for p in batches[batch_idx])
        batch_idx += 1
        
        assert len(batches[batch_idx]) == 2
        assert all(p.shape == (32, 16) and p.dtype == torch.float32 for p in batches[batch_idx])
        batch_idx += 1
        
        # Group 2 batch
        assert len(batches[batch_idx]) == 3
        assert all(p.shape == (32, 16) and p.dtype == torch.float64 for p in batches[batch_idx])
        batch_idx += 1
        
        # Group 3 batches
        assert len(batches[batch_idx]) == 3
        assert all(p.shape == (16, 32) and p.dtype == torch.float32 for p in batches[batch_idx])
        batch_idx += 1
        
        assert len(batches[batch_idx]) == 1
        assert all(p.shape == (16, 32) and p.dtype == torch.float32 for p in batches[batch_idx])