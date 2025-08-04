# Claude Context - DION Optimizer Project

## Current Status (Session from 2025-08-04)

### Completed Work
1. **JAX/Optax Implementation** (PR #9: https://github.com/microsoft/dion/pull/9)
   - Created `optimizers/experimental/dion_reference_optax.py` - functional reference implementation
   - Created `optimizers/experimental/dion_optax.py` - optimized version (has bugs, needs work)
   - Comprehensive test suite with numerical comparisons
   - Documented known issues in `tests/potential_issues.md`
   - Testing guide in `tests/JAX_TESTING_GUIDE.md`

### Key Technical Details
- **Environment**: Google Colab with NVIDIA L4/T4 GPU
- **JAX GPU Testing**: Always use `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- **Test Status**: 10 stable tests passing, unstable tests marked with `@pytest.mark.unstable`
- **Known Issues**:
  - Numerical precision differences (GPU ~1e-3 vs CPU ~1e-7)
  - CQR method numerically unstable on GPU
  - Static shape requirements for JIT compilation
  - Optimized implementation has state management bugs

### Important Commands
```bash
# Run stable tests only
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/ -m "not unstable"

# Run all tests
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/ -v

# Check GPU availability
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

## Next Steps

### 1. Final Polish on Optax Implementation
- [ ] Fix state management in `dion_optax.py` (optimized version)
- [ ] Resolve remaining numerical precision issues
- [ ] Add proper handling for RCQR random initialization
- [ ] Ensure all stable tests pass consistently

### 2. Additional Testing
- [ ] **Smoke Tests**: Basic functionality across different scenarios
  - Various tensor shapes and dtypes
  - Different learning rates and hyperparameters
  - Multi-device/distributed settings
  
- [ ] **Integration Tests**: Full training runs
  - Simple models (MLP on MNIST)
  - Compare convergence with PyTorch version
  - Benchmark performance
  
- [ ] **Model Integration**: `models/experimental/`
  - Create example GPT model using JAX/Flax
  - Demonstrate DION optimizer usage
  - Compare with AdamW baseline

### 3. Prepare for Optax Submission
- [ ] **Code Quality**:
  - Follow Optax coding standards
  - Add comprehensive docstrings
  - Type hints throughout
  - Remove experimental/debugging code
  
- [ ] **Documentation**:
  - Write tutorial notebook
  - Add to Optax docs
  - Include citations to DION paper
  
- [ ] **Testing for Optax**:
  - Match Optax test patterns
  - Add parameterized tests
  - Ensure compatibility with Optax chains
  
- [ ] **Benchmarking**:
  - Performance comparison with Adam/AdamW
  - Memory usage analysis
  - Scaling tests

### 4. Training Runs for Validation
- [ ] **Reproduction Studies**:
  - Recreate key results from DION paper
  - Document hyperparameter sensitivity
  - Compare PyTorch vs JAX implementations
  
- [ ] **New Experiments**:
  - Test on Flax model zoo
  - Vision models (ResNet, ViT)
  - Language models (GPT, BERT)

## Optax Contribution Process

### Prerequisites
1. Implementation follows Optax patterns (✓ mostly done)
2. Comprehensive test coverage
3. Documentation and examples
4. Performance benchmarks
5. Paper citations and acknowledgments

### Submission Steps
1. Fork google/optax repository
2. Create feature branch from main
3. Add DION to `optax/_src/` (not experimental)
4. Update `optax/__init__.py` exports
5. Add tests to `optax/_src/*_test.py`
6. Update documentation
7. Create pull request with:
   - Clear description
   - Link to paper
   - Benchmark results
   - Example usage

### Code Structure for Optax
```python
# optax/_src/dion.py
def dion(
    learning_rate: ScalarOrSchedule,
    rank_fraction: float = 1.0,
    ...
) -> base.GradientTransformation:
    """DION optimizer.
    
    References:
      [Atsentia et al., 2024](https://arxiv.org/abs/2504.05295)
    """
    ...

# optax/_src/dion_test.py
class DionTest(parameterized.TestCase):
    ...
```

## Key Contacts & Resources
- DION Paper: https://arxiv.org/abs/2504.05295
- Optax Repo: https://github.com/google-deepmind/optax
- Optax Contributing: https://github.com/google-deepmind/optax/blob/main/CONTRIBUTING.md

## Session Context Preservation
- Working directory: `/content/dion`
- Branch: `feature/optax-dion-optimizer`
- Author for commits: `Amund Tveit <amund@atsentia.ai>`
- No Claude attribution in commits