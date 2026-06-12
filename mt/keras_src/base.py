"""Determines the working Keras version and backend to be used by mt.keras.

Supports:
- Keras 2 (with TensorFlow backend via tf_keras, keras, or tensorflow.keras)
- Keras 3 (with TensorFlow or PyTorch backend)

Configuration:
- Set KERAS_BACKEND env variable to "tensorflow" or "torch" for Keras 3
- Set KERAS_SOURCE env variable to "keras3", "keras", "tf_keras", or "tensorflow.keras"
"""

import os
from packaging.version import Version

# Get backend and source from environment or auto-detect
KERAS_BACKEND_ENV = os.environ.get("KERAS_BACKEND", "").lower()
KERAS_SOURCE_ENV = os.environ.get("KERAS_SOURCE", "").lower()


def _detect_keras_2():
    """Detect Keras 2 availability and version."""
    import tensorflow as tf

    tf_ver = Version(tf.__version__)
    if tf_ver >= Version("2.16"):
        try:
            import tf_keras

            return "tf_keras", tf_keras.__version__, tf_keras
        except ImportError:
            raise ImportError(
                f"TensorFlow {tf_ver} requires tf_keras. Please install it: pip install tf_keras"
            )
    else:
        try:
            import keras

            kr_ver = Version(keras.__version__)
        except ImportError:
            kr_ver = None

        if kr_ver is None or kr_ver >= Version("3.0"):
            # Keras 3 or not installed, use bundled Keras from TensorFlow
            return "tensorflow.keras", tf.__version__, tf.keras
        else:
            # Standalone Keras 2
            return "keras", keras.__version__, keras


def _detect_keras_3():
    """Detect Keras 3 availability and backend."""
    try:
        import keras

        kr_ver = Version(keras.__version__)
        if kr_ver >= Version("3.0"):
            # KERAS_BACKEND env var must be set before importing keras; read from
            # keras.backend.backend() to get whatever backend keras resolved to.
            backend = KERAS_BACKEND_ENV or keras.backend.backend()
            return "keras3", keras.__version__, keras, backend
    except ImportError:
        pass
    return None


# Priority detection logic
keras_package = None
keras_version = None
keras_source = None
keras_backend = None

# 1. If KERAS_SOURCE is explicitly set, use that
if KERAS_SOURCE_ENV == "keras3":
    result = _detect_keras_3()
    if result:
        keras_source, keras_version, keras_package, keras_backend = result
    else:
        raise ImportError(
            "Keras 3 requested but not installed. Install with: pip install keras"
        )
elif KERAS_SOURCE_ENV in ["keras", "tf_keras", "tensorflow.keras"]:
    keras_source, keras_version, keras_package = _detect_keras_2()
    if KERAS_SOURCE_ENV == "keras" and keras_source != "keras":
        raise ImportError("Standalone Keras 2 requested but not available")
    elif KERAS_SOURCE_ENV == "tf_keras" and keras_source != "tf_keras":
        raise ImportError("tf_keras requested but not available")
    elif KERAS_SOURCE_ENV == "tensorflow.keras" and keras_source != "tensorflow.keras":
        raise ImportError("tensorflow.keras requested but not available")

# 2. Try Keras 3 first if KERAS_BACKEND is set or auto-detection enabled
elif KERAS_BACKEND_ENV or os.environ.get("KERAS_PREFER", "") == "keras3":
    result = _detect_keras_3()
    if result:
        keras_source, keras_version, keras_package, keras_backend = result
    else:
        # Fall back to Keras 2
        keras_source, keras_version, keras_package = _detect_keras_2()

# 3. Default: Try Keras 2 first, then Keras 3
else:
    try:
        keras_source, keras_version, keras_package = _detect_keras_2()
    except ImportError:
        # If Keras 2 detection fails, try Keras 3
        result = _detect_keras_3()
        if result:
            keras_source, keras_version, keras_package, keras_backend = result
        else:
            raise ImportError(
                "No compatible Keras installation found. "
                "Install Keras 2 (pip install tf-keras) or Keras 3 (pip install keras)"
            )

# Set default backend for Keras 2 if not already set
if keras_backend is None:
    keras_backend = "tensorflow"
