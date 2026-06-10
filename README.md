# mtkeras
Keras namespace packages that build on mttf.

This repo provides `mt.keras` and `mt.keras_src` while depending on `mttf` for the shared `mt.tf` package.

## Features

- **Multi-version support**: Works with Keras 2 (TensorFlow backend) and Keras 3
- **Multi-backend support**: Use TensorFlow, PyTorch, JAX, or NumPy backends
- **Backend-agnostic**: Custom operations automatically adapt to the active backend
- **Backward compatible**: Existing Keras 2 code works without changes

## Quick Start

### Keras 2 (Default)
```bash
pip install tensorflow
```

### Keras 3 with TensorFlow
```bash
pip install keras tensorflow
```

### Keras 3 with PyTorch
```bash
pip install keras torch
export KERAS_BACKEND=torch
```

## Configuration

See [BACKEND_CONFIGURATION.md](BACKEND_CONFIGURATION.md) for detailed configuration options.

### Auto-detection
```python
from mt.keras_src import *  # Automatically detects and uses available Keras
```

### Manual backend selection
```bash
export KERAS_BACKEND=torch      # For Keras 3 with PyTorch
export KERAS_SOURCE=keras3      # Force Keras 3
export KERAS_SOURCE=tf_keras    # Force Keras 2 with tf_keras
```

## Supported Components

- Custom layers: `Counter`, `Floor`, `SoftBend`, `NormedConv2D`, `SimpleMHA2D`
- Applications: `MobileViT`, `MobileNetV3` models
- Constraints: `CenterAround`
- Operations: Backend-agnostic via `ops_compat` module

## API Compatibility

All APIs are designed to work seamlessly with both Keras 2 and Keras 3. Backend-specific implementations are abstracted through:

- `mt.keras_src.ops_compat` - Backend-agnostic operations
- `mt.keras_src.grad_compat` - Custom gradient functions
- Automatic backend detection in `mt.keras_src.base`
