# Keras Backend Support - Configuration Guide

This package now supports multiple Keras versions and backends for maximum flexibility:

## Supported Configurations

### Keras 2 (TensorFlow backend)
- **TensorFlow <= 2.15**: Uses bundled Keras or standalone Keras 2
- **TensorFlow >= 2.16**: Requires `tf_keras` package
- Installation: `pip install tensorflow` or `pip install tf-keras`

### Keras 3 (Backend-agnostic)
- **TensorFlow backend**: High performance, production-ready
- **PyTorch backend**: For PyTorch-based workflows
- **JAX backend**: For numerical computing research
- Installation: `pip install keras`

## Configuration Methods

### 1. Automatic Detection (Default)
The package automatically detects and uses the first available Keras installation:

```python
from mt.keras_src import *

# Automatically uses:
# 1. Keras 2 if available (tf_keras, standalone Keras, or tensorflow.keras)
# 2. Keras 3 if Keras 2 is not available
```

### 2. Environment Variables

#### Select Keras Version
```bash
# Force Keras 3
export KERAS_SOURCE=keras3

# Force Keras 2 (tf_keras)
export KERAS_SOURCE=tf_keras

# Force Keras 2 (standalone)
export KERAS_SOURCE=keras

# Force Keras 2 (TensorFlow bundled)
export KERAS_SOURCE=tensorflow.keras
```

#### Select Backend (Keras 3 only)
```bash
# TensorFlow backend (default)
export KERAS_BACKEND=tensorflow

# PyTorch backend
export KERAS_BACKEND=torch

# JAX backend
export KERAS_BACKEND=jax

# NumPy backend (CPU only, debugging)
export KERAS_BACKEND=numpy
```

#### Prefer Keras 3
```bash
# Prefer Keras 3 if available, fall back to Keras 2
export KERAS_PREFER=keras3
```

### 3. Python Code Configuration

For Keras 3, set the backend before importing:

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

from mt.keras_src import *
```

## Installation Examples

### Setup for Keras 3 with TensorFlow Backend
```bash
pip install keras tensorflow
```

### Setup for Keras 3 with PyTorch Backend
```bash
pip install keras torch

# Then configure the backend:
export KERAS_BACKEND=torch
```

### Setup for Keras 2 with TensorFlow <= 2.15
```bash
pip install tensorflow
```

### Setup for Keras 2 with TensorFlow >= 2.16
```bash
pip install tensorflow tf-keras
```

## Backend-Agnostic Operations

The package includes a compatibility layer (`ops_compat`) for backend-agnostic operations:

```python
from mt.keras_src.ops_compat import ops

# These work regardless of backend:
x = ops.reshape(x, shape)
x = ops.reduce_sum(x, axis=-1)
x = ops.expand_dims(x, axis=-1)
y = ops.stop_gradient(y)
```

Supported operations:
- `shape`, `reshape`, `squeeze`, `expand_dims`, `transpose`
- `reduce_sum`, `reduce_mean`, `reduce_prod`
- `concatenate`, `stack`, `tile`, `pad`
- `ones`, `zeros`, `constant`, `cast`
- `abs`, `pow`, `tanh`, `floor`, `sqrt`
- `matmul`, `stop_gradient`

## Troubleshooting

### ImportError: Keras 3 not installed
```bash
pip install keras
```

### ImportError: tf_keras not installed
```bash
# For TensorFlow >= 2.16
pip install tf-keras
```

### ImportError: PyTorch not installed
```bash
export KERAS_BACKEND=tensorflow  # Use TensorFlow instead
# Or install PyTorch:
pip install torch
```

### Check Active Configuration
```python
from mt.keras_src.base import keras_source, keras_version, keras_backend

print(f"Keras source: {keras_source}")
print(f"Keras version: {keras_version}")
print(f"Backend: {keras_backend}")
```

## API Compatibility

### Layer Support
All custom layers are compatible with both Keras 2 and Keras 3:

- `mt.keras_src.layers_src.Counter`
- `mt.keras_src.layers_src.Floor`
- `mt.keras_src.layers_src.SoftBend`
- `mt.keras_src.layers_src.NormedConv2D`
- `mt.keras_src.layers_src.SimpleMHA2D`
- `mt.keras_src.constraints_src.CenterAround`

### Model Saving/Loading
Supported formats depend on Keras version:

| Format | Keras 2 | Keras 3 | Notes |
|--------|---------|---------|-------|
| `.h5` | ✓ | ✓ | HDF5 format (legacy) |
| `.tf` | ✓ | ✓ | TensorFlow SavedModel |
| `.keras` | ✓ (2.15+) | ✓ | Native Keras format (recommended for 3.x) |

```python
from mt.keras_src import d_modelFileFormats

# Check available formats
print(d_modelFileFormats)
```

## Performance Notes

- **Keras 2**: Optimized for TensorFlow, best for TensorFlow users
- **Keras 3 + TensorFlow**: Good performance, cleaner API
- **Keras 3 + PyTorch**: Growing support, excellent for PyTorch workflows
- **Keras 3 + JAX**: Best for research with automatic differentiation

## Migration Guide: Keras 2 → Keras 3

Most code requires no changes. However:

1. **Custom TensorFlow ops**: Use `ops_compat` instead
2. **Gradients**: Use `grad_compat.custom_gradient` decorator
3. **Backend-specific code**: Abstract via `keras.backend` API

Example migration:

```python
# Old Keras 2 code
import tensorflow as tf
x = tf.reshape(x, shape)
y = tf.stop_gradient(y)

# New code (works with any backend)
from mt.keras_src.ops_compat import ops
x = ops.reshape(x, shape)
y = ops.stop_gradient(y)
```

## Additional Resources

- [Keras Official Documentation](https://keras.io/)
- [Keras 3 Migration Guide](https://keras.io/guides/migrating_to_keras_3/)
- [PyTorch Backend](https://keras.io/api/config/#keras_backend)
