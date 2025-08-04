# JAX Testing Guide

This guide explains how to run JAX/Optax tests for the DION optimizer implementation.

## Environment Setup

### GPU Memory Pre-allocation

JAX by default pre-allocates the entire GPU memory, which can cause issues in shared environments like Colab. To disable this:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Or prefix your commands:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/
```

**What this does**: Tells JAX to allocate GPU memory on-demand rather than grabbing all available memory at startup.

### Other Useful Environment Variables

```bash
# Show JAX transformations and compilations
export JAX_LOG_COMPILES=1

# Disable JAX's internal frame filtering in tracebacks
export JAX_TRACEBACK_FILTERING=off

# Force CPU-only execution
export JAX_PLATFORM_NAME=cpu

# Control JAX's default dtype
export JAX_DEFAULT_DTYPE_BITS=32  # Use float32 instead of float64
```

## Running Tests

### Basic Test Execution

```bash
# Run all experimental optimizer tests
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/

# Run only stable tests (skip unstable ones)
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/ -m "not unstable"

# Run only unstable tests
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/ -m unstable

# Run specific test file
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/test_dion_reference_optax.py

# Run specific test method
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/test_dion_reference_optax.py::TestDionOptax::test_optimizer_step

# With verbose output
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/ -v

# With detailed print statements
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/optimizers/experimental/ -xvs
```

### Test Markers

Tests are marked with `@pytest.mark.unstable` for:
- Tests with known numerical precision issues
- Tests with GPU-specific failures
- Tests for incomplete implementations

To run tests by stability:
```bash
# Only stable tests
pytest -m "not unstable"

# Only unstable tests  
pytest -m unstable

# All tests (default)
pytest
```

### Test Options

- `-x`: Stop on first failure
- `-v`: Verbose output (show test names)
- `-s`: No capture (show print statements)
- `--tb=short`: Shorter traceback format
- `--tb=no`: No traceback
- `-q`: Quiet mode

### GPU vs CPU Testing

```bash
# Force CPU testing
JAX_PLATFORM_NAME=cpu python -m pytest tests/

# Check which device JAX is using
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

## Common Issues and Solutions

### 1. GPU Memory Errors
```
RuntimeError: Resource exhausted: Out of memory
```
**Solution**: Always use `XLA_PYTHON_CLIENT_PREALLOCATE=false`

### 2. Numerical Precision Differences
JAX on GPU often shows different numerical precision than CPU:
- GPU QR decomposition: ~1e-3 precision
- CPU QR decomposition: ~1e-7 precision

**Solution**: Use appropriate tolerances (`atol=1e-3` for GPU tests)

### 3. JIT Compilation Errors
```
TracerBoolConversionError: Attempted to convert a traced array to a boolean
```
**Solution**: Avoid dynamic control flow in JIT-compiled functions. Use `lax.cond` instead of `if`.

### 4. Static Shape Requirements
```
TypeError: Shapes must be 1D sequences of concrete values of integer type
```
**Solution**: Use static computations for array shapes in JIT context.

## Test Structure

### Reference Implementation Tests (`test_dion_reference_optax.py`)
- `test_optimizer_initialization`: Basic state initialization
- `test_optimizer_step`: Single optimization step
- `test_different_algorithms`: DION, AdamW, Lion variants
- `test_orthogonalize_methods`: QR, CQR, RCQR methods
- `test_weight_decay`: Weight decay functionality
- `test_learning_rate_schedule`: Dynamic learning rates

### Numerical Comparison Tests (`test_numerical_comparison.py`)
- Compares PyTorch and JAX implementations
- Tests exact initialization, single steps, convergence
- Expected to show small numerical differences

### Optimized Implementation Tests (`test_dion_optax.py`)
- Tests for the vectorized/optimized version
- Currently has implementation issues

## Debugging Tips

### 1. Enable Detailed Logging
```python
# In your test
print(f"State keys: {state.keys()}")
print(f"Update norm: {jnp.linalg.norm(updates['weight'])}")
```

### 2. Check Device Placement
```python
import jax
print(f"Default backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")
```

### 3. Disable JIT for Debugging
```python
# Temporarily disable JIT
with jax.disable_jit():
    result = optimizer.update(grads, state, params)
```

### 4. Trace Function Calls
```bash
JAX_LOG_COMPILES=1 python -m pytest tests/
```

## Expected Behavior

### Successful Test Run
```
============================= test session starts ==============================
platform linux -- Python 3.11.13, pytest-8.3.3, pluggy-1.5.0
collected 12 items

tests/optimizers/experimental/test_dion_reference_optax.py::TestDionOptax::test_optimizer_initialization PASSED [  8%]
tests/optimizers/experimental/test_dion_reference_optax.py::TestDionOptax::test_optimizer_step PASSED [ 16%]
...
========================= 10 passed, 2 failed in 16.68s =========================
```

### Known Failures
1. **CQR orthogonalization**: Numerically unstable on GPU
2. **RCQR with deterministic init**: Falls back to non-random initialization
3. **Numerical comparisons**: Small differences between PyTorch and JAX

## Performance Considerations

### GPU Execution
- First run includes JIT compilation time
- Subsequent runs are much faster
- Use batch operations with `vmap` for efficiency

### Memory Usage
- JAX creates copies rather than in-place updates
- Monitor memory with `nvidia-smi` on GPU
- Use mixed precision to reduce memory

## Integration with CI/CD

For GitHub Actions or other CI systems:

```yaml
- name: Run JAX Tests
  env:
    XLA_PYTHON_CLIENT_PREALLOCATE: false
    JAX_PLATFORM_NAME: cpu  # Use CPU in CI
  run: |
    python -m pytest tests/optimizers/experimental/ -v
```

## Troubleshooting Checklist

1. ✓ Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`
2. ✓ Check JAX version compatibility
3. ✓ Verify GPU/CPU device selection
4. ✓ Use appropriate numerical tolerances
5. ✓ Handle static shape requirements
6. ✓ Account for JIT compilation constraints
7. ✓ Consider numerical precision differences