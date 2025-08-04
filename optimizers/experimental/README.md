# Experimental Optimizers

This directory contains experimental implementations of optimizers using alternative frameworks.

## JAX/Optax DION Implementations

### Overview

This module provides JAX/Optax implementations of the DION (Distributed Shampoo) optimizer:

- **`dion_reference_optax.py`**: Reference implementation based on `dion_reference.py`, following Optax's functional style
- **`dion_optax.py`**: Optimized implementation based on `dion.py` with advanced JAX features

### Installation

Ensure you have the required dependencies:

```bash
pip install jax>=0.4.0 optax>=0.1.7 flax>=0.7.0
```

### Usage

#### Basic Usage with Optax

```python
import jax
import jax.numpy as jnp
import optax
from optimizers.experimental.dion_reference_optax import dion

# Create optimizer
optimizer = dion(
    learning_rate=0.01,
    rank_fraction=0.25,
    qr_method='rcqr'
)

# Initialize parameters and optimizer state
params = {'w': jnp.ones((128, 64))}
opt_state = optimizer.init(params)

# Compute gradients
def loss_fn(params):
    return jnp.sum(params['w'] ** 2)

grads = jax.grad(loss_fn)(params)

# Update parameters
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

#### Usage with Flax

```python
import flax.linen as nn
from flax.training import train_state
from optimizers.experimental.dion_reference_optax import dion

class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# Create model and optimizer
model = Model()
optimizer = dion(learning_rate=0.01)

# Create training state
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=model.init(rng, dummy_input),
    tx=optimizer
)
```

### Key Features

1. **Low-rank approximation**: Efficient computation using rank-r approximations
2. **Multiple QR methods**: Support for QR, Cholesky QR (CQR), and Randomized CQR
3. **Mixed precision**: Configurable precision for different optimizer states
4. **Distributed training**: JAX-native support for multi-device training
5. **Functional API**: Clean integration with JAX's functional programming style

### Differences from PyTorch Implementation

1. **State Management**: Uses Optax's immutable state pattern instead of in-place updates
2. **Parallelism**: Leverages JAX's `vmap`, `pmap`, and `jit` for automatic optimization
3. **Random Number Generation**: Uses JAX's explicit RNG handling
4. **Gradients**: Works with JAX's functional gradient computation

### Performance Considerations

- The JAX implementation benefits from XLA compilation for improved performance
- Automatic vectorization with `vmap` for batch operations
- Efficient multi-device support with `pmap`
- Consider using `jax.jit` for production workloads

### Algorithm Details

The DION optimizer implements the distributed Shampoo algorithm with low-rank approximations:

1. Maintains momentum buffer M and low-rank factor Q
2. Computes low-rank approximation: M ≈ PQ^T
3. Updates parameters using orthogonalized factors
4. Supports various orthogonalization methods for numerical stability

For more details, see the [DION paper](https://arxiv.org/abs/2504.05295).

### Testing

Run tests with:
```bash
pytest tests/optimizers/experimental/
```

### Contributing

When adding new experimental optimizers:
1. Follow the existing naming conventions
2. Provide both reference and optimized implementations when applicable
3. Include comprehensive tests
4. Document key differences from standard implementations