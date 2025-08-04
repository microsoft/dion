# Dion Optimizer Test Suite

This directory contains comprehensive unit tests for the Dion optimizer implementation and related components.

## Quick Start

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=optimizers --cov-report=term

# Run only passing tests (skip known failures)
pytest tests/ -k "not (numerical or orthogonalize_methods)"

# Run specific test category
pytest tests/optimizers/                    # Core optimizer tests
pytest tests/optimizer_comparison/          # Comparison tests
pytest tests/integration/test_smoke.py      # Smoke tests only
```

## Test Structure

```
tests/
├── README.md                          # This file
├── __init__.py
├── optimizers/                        # Core optimizer tests
│   ├── __init__.py
│   ├── test_dion_reference.py        # Tests for DionReference implementation (19 tests)
│   ├── test_dion_numerical.py        # Numerical accuracy and stability tests (11 tests)
│   ├── test_scalar_opts.py           # Tests for Lion and AdamW implementations (12 tests)
│   ├── test_scalar_update_functions.py # Direct tests for update functions (3 tests)
│   ├── test_opt_utils.py             # Tests for optimizer utilities (9 tests)
│   └── test_utils.py                 # Testing utilities and skip decorators
├── optimizer_comparison/              # Cross-implementation comparison tests
│   ├── __init__.py
│   ├── base_comparison.py             # Base class with shared utilities
│   ├── test_dion_implementations.py   # Compare Dion variants (5 tests)
│   ├── test_muon_implementations.py   # Compare Muon variants (6 tests)
│   ├── test_matrix_optimizer_properties.py  # Dion vs Muon matrix properties (7 tests)
│   ├── test_optimizer_characteristics.py    # Fundamental optimizer differences (8 tests)
│   ├── test_convergence_patterns.py   # Convergence behavior comparison (4 tests)
│   ├── test_parameter_update_patterns.py    # Update pattern analysis (6 tests)
│   └── test_robustness_characteristics.py   # Robustness properties (6 tests)
└── integration/                       # Integration and performance tests
    ├── __init__.py
    ├── test_smoke.py                  # Basic training loop smoke tests (9 tests)
    └── test_performance.py            # Performance benchmarks (6 tests)

**Total: 15 test files, 107 test functions**
```

## Test Categories

### 1. Core Functionality Tests (`test_dion_reference.py`)
- **Initialization**: Parameter validation, hyperparameter checks
- **Basic Operations**: Step function, gradient updates, state management
- **Parameter Groups**: Matrix vs scalar parameters, custom algorithms
- **Edge Cases**: Zero gradients, None gradients, empty tensors

### 2. Numerical Accuracy Tests (`test_dion_numerical.py`)
- **Orthogonalization Stability**: Tests with ill-conditioned matrices
- **Power Iteration Convergence**: Accuracy for different matrix types
- **Precision Tests**: Double precision accumulation, error feedback
- **Extreme Values**: Handling of very large/small values

### 3. Scalar Optimizer Tests (`test_scalar_opts.py`)
- **AdamW**: Momentum, bias correction, weight decay
- **Lion**: Sign updates, momentum interpolation
- **Foreach Implementations**: Batched operations
- **Edge Cases**: Zero gradients, extreme values

### 4. Utility Tests (`test_opt_utils.py`)
- **Tensor Utilities**: DTensor conversion, local tensor handling
- **Batching**: Parameter grouping, batch padding
- **Async Operations**: Task scheduling, concurrent execution

### 5. Implementation Comparison Tests (`optimizer_comparison/`)

#### Same-Type Comparisons
- **Dion Implementations** (`test_dion_implementations.py`): DionSimple vs DionReference vs DionOptimized
- **Muon Implementations** (`test_muon_implementations.py`): MuonReference vs MuonOptimized

#### Cross-Optimizer Comparisons
- **Matrix Properties** (`test_matrix_optimizer_properties.py`): 
  - Rank preservation: How Dion vs Muon handle low-rank structure
  - Orthogonalization: QR (Dion) vs Newton-Schulz (Muon)
  - Eigenvector preservation and conditioning sensitivity
  
- **Optimizer Characteristics** (`test_optimizer_characteristics.py`):
  - Parameter norm evolution with weight decay
  - Gradient noise robustness across different noise levels
  - Learning rate sensitivity and batch size invariance
  - Memory/momentum patterns
  
- **Convergence Patterns** (`test_convergence_patterns.py`):
  - Speed on quadratic objectives
  - Stability with noisy gradients
  - Loss landscape navigation (MSE vs CrossEntropy vs Huber)
  - Momentum effects on convergence smoothness
  
- **Update Patterns** (`test_parameter_update_patterns.py`):
  - Update magnitude vs gradient magnitude relationships
  - Direction alignment with gradients
  - Sign-based (Lion) vs magnitude-based (AdamW) patterns
  - Low-rank structure in updates (Dion)
  
- **Robustness** (`test_robustness_characteristics.py`):
  - Gradient explosion/vanishing handling
  - Sparse gradient robustness
  - Ill-conditioned gradient behavior
  - Noise filtering capability
  - Catastrophic forgetting resistance

### 6. Integration Tests (`integration/`)
- **Smoke Tests**: Basic training loops with real models
- **Convergence**: Verify optimizers reduce loss
- **State Persistence**: Save/load functionality
- **Gradient Clipping**: Compatibility with common techniques
- **Performance Benchmarks**: Speed and memory profiling

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Core optimizer tests only
pytest tests/optimizers/

# Comparison tests only
pytest tests/optimizer_comparison/

# Numerical accuracy tests
pytest tests/optimizers/test_dion_numerical.py
```

### Run with Coverage
```bash
pytest tests/ --cov=optimizers --cov-report=html
```

### Run Tests by Marker
```bash
# Skip tests requiring optional dependencies
pytest tests/ -m "not requires_triton"

# Run only tests that don't require CUDA
pytest tests/ -m "not requires_cuda"

# Run only integration tests
pytest tests/ -m "integration"

# Run only performance tests
pytest tests/ -m "performance"

# Run smoke tests only
pytest tests/integration/test_smoke.py
```

## Test Markers and Skip Conditions

Tests use pytest markers to handle optional dependencies:

- `@pytest.mark.skipif(not HAS_TRITON)` - Skip if triton not installed
- `@pytest.mark.skipif(not HAS_CUDA)` - Skip if CUDA not available
- `@pytest.mark.skipif(not HAS_DISTRIBUTED)` - Skip if distributed not available

See `test_utils.py` for helper functions and decorators.

## Numerical Tolerances and Precision

### Understanding Tolerance Values

When comparing floating-point values in tests, we use `torch.allclose(a, b, rtol, atol)` which checks:
```
|a - b| ≤ atol + rtol * |b|
```

Common tolerance values used in our tests:

| Tolerance | Value | Use Case | Rationale |
|-----------|-------|----------|-----------|
| `atol=1e-7` | 0.0000001 | High precision comparisons | Near machine epsilon for float32 (~1.19e-7) |
| `atol=1e-6` | 0.000001 | Standard precision | 10x machine epsilon, handles accumulation errors |
| `atol=1e-5` | 0.00001 | Relaxed precision | For operations with multiple floating-point ops |
| `atol=1e-4` | 0.0001 | Cross-implementation | Different algorithms may accumulate errors differently |
| `rtol=1e-5` | 0.00001 | Relative 0.001% | Standard relative tolerance |
| `rtol=1e-3` | 0.001 | Relative 0.1% | For approximate algorithms |

### Platform and Precision Considerations

1. **Float32 vs Float64**:
   - PyTorch defaults to float32 (single precision)
   - Machine epsilon: ~1.19e-7 for float32, ~2.22e-16 for float64
   - Accumulation of rounding errors grows with operation count

2. **CPU vs GPU**:
   - CPU: Consistent IEEE 754 compliance
   - GPU: May use different rounding modes or fast-math approximations
   - GPU reductions may have non-deterministic ordering

3. **Triton and Custom Kernels**:
   - Triton may use different precision for intermediate calculations
   - Fused operations can reduce rounding errors
   - Block-wise operations may have different accumulation patterns

4. **Algorithm-Specific Tolerances**:
   - **QR Decomposition**: `1e-6` to `1e-5` (iterative refinement varies)
   - **Power Iteration**: `1e-5` to `1e-4` (convergence rate dependent)
   - **Newton-Schulz**: `1e-4` to `1e-3` (approximation method)
   - **Momentum Updates**: `1e-6` (simple accumulation)

### Best Practices

1. **Choose tolerances based on**:
   - Number of floating-point operations
   - Algorithm stability characteristics
   - Platform variability requirements

2. **When to use strict tolerances** (`atol=1e-7`):
   - Single operations (addition, multiplication)
   - Deterministic algorithms
   - Same-platform comparisons

3. **When to use relaxed tolerances** (`atol=1e-4`):
   - Cross-platform tests
   - Iterative algorithms
   - Different implementations of same algorithm
   - Operations on large matrices

4. **Special cases**:
   - Use `torch.float64` for high-precision ground truth
   - Check relative error for large magnitude values
   - Consider condition numbers for linear algebra operations

## Writing New Tests

### Guidelines
1. **Isolation**: Each test should be independent
2. **Reproducibility**: Use fixed seeds (`torch.manual_seed(42)`)
3. **Clarity**: Clear test names describing what is tested
4. **Coverage**: Test both success and failure cases
5. **Tolerances**: Use appropriate numerical tolerances (see section above)

### Example Test Structure
```python
def test_feature_name(self, device):
    """Test description of what this validates"""
    # Setup
    torch.manual_seed(42)
    param = torch.randn(32, 16, device=device)
    
    # Execute
    result = function_under_test(param)
    
    # Assert with appropriate tolerance
    # Strict tolerance for simple operations
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6)
    
    # Relaxed tolerance for complex algorithms
    assert torch.allclose(result_complex, expected_complex, rtol=1e-3, atol=1e-4)
```

## Test Coverage

Current test coverage status (as of last run):

| Module | Coverage | Notes |
|--------|----------|-------|
| `opt_utils.py` | 86% | Well tested, missing DTensor functions |
| `dion_reference.py` | 53% | Core functionality tested, missing distributed ops |
| `dion.py` | 39% | Basic functionality tested, missing Triton/async paths |
| `scalar_opts.py` | 18% | Low due to `@torch.compile` decorators |
| `dion_simple.py` | 0% | Tested indirectly via comparison tests |
| `muon_reference.py` | 0% | Tested indirectly via comparison tests |

### Running Coverage Analysis

```bash
# Generate coverage report
pytest tests/ --cov=optimizers --cov-report=html --cov-report=term

# View detailed HTML report
open htmlcov/index.html
```

## Known Issues and TODOs

### Test Failures
1. **Numerical Tests**: Some tests fail due to overly strict tolerances
   - `test_power_iteration_accuracy`: Tolerance too strict for low-rank approximation
   - `test_orthogonalize_methods`: CQR method needs higher tolerance
   - Solution: Adjust tolerances based on algorithm characteristics

2. **Comparison Tests**: Different implementations may diverge slightly
   - DionSimple vs DionReference use different scaling
   - RCQR (randomized) produces different results than QR
   - Solution: Use appropriate tolerances for each comparison

### Coverage Gaps
1. **Distributed Operations**: DTensor and mesh operations not tested
2. **Compiled Functions**: `@torch.compile` prevents direct testing
3. **Optional Dependencies**: Triton kernels, CUDA-specific paths
4. **Error Handling**: Many error branches not covered
5. **Advanced Algorithms**: Some QR variants (CQR) not fully tested

### Future Improvements
1. **Mock Distributed Ops**: Create mock mesh/DTensor for testing
2. **Test Compiled Functions**: Test with torch.compile disabled
3. **Error Injection**: Test error handling paths
4. **Performance Regression**: Add benchmarks to track performance
5. **Mixed Precision**: Add bfloat16/float16 tests

## Contributing

When adding new tests:
1. Place in appropriate file or create new file if needed
2. Use consistent naming: `test_<feature>_<aspect>`
3. Add docstrings explaining what is tested
4. Choose appropriate tolerances (see Numerical Tolerances section)
5. Run coverage to ensure new code is tested
6. Update this README if adding new test categories

### Test Writing Checklist
- [ ] Test both success and failure cases
- [ ] Use appropriate numerical tolerances
- [ ] Add skip decorators for optional dependencies
- [ ] Set random seeds for reproducibility
- [ ] Test edge cases (empty tensors, None gradients, etc.)
- [ ] Verify test actually tests the intended behavior