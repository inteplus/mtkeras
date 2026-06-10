# Architecture & Design Decisions

## Overview

The mtkeras package has been extended to support multiple Keras versions and backends while maintaining a unified API and backward compatibility.

## Architecture Diagram

```
User Code
    ↓
mt.keras_src (main entry point)
    ↓
├─ base.py (Keras version & backend detection)
├─ ops_compat.py (backend-agnostic operations)
├─ grad_compat.py (custom gradient compatibility)
└─ Custom layers (Counter, Floor, SoftBend, etc.)
    ↓
├─ Keras 2 Path:
│  ├─ tf_keras
│  ├─ keras (standalone, <3.0)
│  └─ tensorflow.keras
│
└─ Keras 3 Path:
   ├─ TensorFlow backend
   ├─ PyTorch backend
   ├─ JAX backend
   └─ NumPy backend
```

## Key Design Decisions

### 1. Priority-Based Backend Detection

**Rationale**: Support both old and new code paths without breaking existing deployments.

```
Priority (if not explicitly set):
1. Keras 2 (if available) - for backward compatibility
2. Keras 3 (if available) - for new features
```

**Can be overridden via environment variables**:
- `KERAS_SOURCE=keras3` - Force Keras 3
- `KERAS_PREFER=keras3` - Prefer Keras 3 but fallback to Keras 2

### 2. Operations Compatibility Layer

**Rationale**: Keras 3 uses backend-agnostic `keras.ops` while Keras 2 requires TensorFlow ops.

```python
# Before (TensorFlow-specific):
import tensorflow as tf
y = tf.reshape(x, shape)

# After (backend-agnostic):
from mt.keras_src.ops_compat import ops
y = ops.reshape(x, shape)
```

**Implementation**:
- `ops_compat.py` provides unified interface
- For Keras 3: wraps `keras.ops`
- For Keras 2: wraps TensorFlow operations
- Gradual migration possible without rewriting everything

### 3. Custom Gradient Compatibility

**Rationale**: Keras 2 uses `@tf.custom_gradient`, Keras 3 uses `keras.ops.custom_gradient`.

```python
from mt.keras_src.grad_compat import custom_gradient

@custom_gradient
def my_op(x):
    def grad(upstream):
        return upstream
    return ops.floor(x), grad
```

### 4. Graceful Fallbacks

**For decorators and utilities not available in all versions**:

```python
try:
    from tensorflow.python.util.tf_export import keras_export
except ImportError:
    def keras_export(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
```

This ensures code works even if internal TensorFlow APIs change.

### 5. Environment-Based Configuration

**Why not programmatic?**

- Keras 3 requires backend to be set **before** importing `keras`
- Using env vars ensures correct ordering
- Automatic for most users, manual override available

**Supported variables**:
```bash
KERAS_BACKEND=tensorflow|torch|jax|numpy
KERAS_SOURCE=keras3|keras|tf_keras|tensorflow.keras
KERAS_PREFER=keras3  # Prefer but allow fallback
```

## Compatibility Matrix

| Scenario | Keras 2 | Keras 3-TF | Keras 3-Torch |
|----------|---------|-----------|---------------|
| Custom ops | ✓ | ✓ | ✓ |
| Custom gradients | ✓ | ✓ | ✓ (native) |
| Model save/load | ✓ | ✓ | ✓ |
| TensorFlow Ops | ✓ | ✓ (via bridge) | ✗ |
| PyTorch Ops | ✗ | ✗ | ✓ (native) |
| JAX Ops | ✗ | ✗ | ~partial |

## Migration Path for Users

### Phase 1: Continue with Keras 2 (No changes needed)
```python
# Works exactly as before
from mt.keras_src import layers, models, ...
```

### Phase 2: Upgrade to Keras 3 (Automatic)
```bash
pip install keras tensorflow
# Code works unchanged, automatically uses Keras 3
```

### Phase 3: Switch to PyTorch (Manual)
```bash
pip install keras torch
export KERAS_BACKEND=torch
# Code works with PyTorch backend
```

### Phase 4: Update Custom Code
```python
# Replace TensorFlow-specific code:
from mt.keras_src.ops_compat import ops
from mt.keras_src.grad_compat import custom_gradient

@custom_gradient
def my_op(x):
    return ops.floor(x), lambda u: u
```

## Files & Responsibilities

### Detection & Configuration
- **base.py**: Version/backend detection, exports config variables
- **ops_compat.py**: Operation compatibility wrapper
- **grad_compat.py**: Gradient decorator compatibility

### Layers & Custom Code
- **layers_src/**: All custom layers use `ops_compat` and `grad_compat`
- **constraints_src/**: All constraints use `ops_compat`
- **applications_src/**: Updated to use string-based activation functions

### Main Entry Point
- **__init__.py**: Handles imports based on detected version

## Performance Implications

### Keras 2
- Direct TensorFlow operations (low overhead)
- Best performance for TensorFlow workloads
- Mature and stable

### Keras 3 + TensorFlow
- Slight overhead from abstraction layer
- Similar performance to Keras 2
- Cleaner API, better forward compatibility

### Keras 3 + PyTorch
- Native PyTorch operations (good performance)
- Enables PyTorch ecosystem integration
- Growing community support

## Testing Strategy

1. **Unit tests** - Each compatibility layer function
2. **Integration tests** - Full model training with different backends
3. **Regression tests** - Existing Keras 2 code still works
4. **Performance tests** - Compare backends

### Test Coverage Areas
- Layer instantiation and forward pass
- Model saving and loading
- Training loops with different backends
- Custom operations correctness
- Gradient computation

## Future Enhancements

1. **Keras 4.0 support** - When released
2. **ONNX export** - Platform-agnostic model deployment
3. **Multi-backend training** - Using `keras.distributed`
4. **Performance optimizations** - Backend-specific kernels
5. **Complete JAX support** - For research workflows

## Known Limitations

1. **TensorFlow-specific features**: Some internal APIs only available in Keras 2
2. **Custom gradients with PyTorch**: Need careful testing for numerical correctness
3. **Model serialization**: Different formats for different backends
4. **Activation functions**: Limited to those available in `keras.activations`

## Troubleshooting Guide

### Issue: ImportError - Keras 3 not found
**Solution**: `pip install keras`

### Issue: ImportError - PyTorch not found
**Solution**: `pip install torch` OR `export KERAS_BACKEND=tensorflow`

### Issue: Performance degradation
**Solution**: Verify correct backend is being used via:
```python
from mt.keras_src.base import keras_source, keras_backend
print(f"Keras: {keras_source}, Backend: {keras_backend}")
```

### Issue: Model doesn't train with PyTorch backend
**Solution**: Ensure all custom operations use `ops_compat`, not raw PyTorch/TensorFlow ops

## References

- [Keras Official Docs](https://keras.io/)
- [Keras 3 Backend Support](https://keras.io/api/config/)
- [Keras 3 Migration Guide](https://keras.io/guides/migrating_to_keras_3/)
