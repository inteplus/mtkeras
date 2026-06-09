"""Determines working Keras 3 + torch backend for mt.tkeras."""

import os
from packaging.version import Version

try:
    import torch  # noqa: F401
except ImportError as e:
    raise ImportError("mt.tkeras requires PyTorch. Please install torch.") from e

os.environ.setdefault("KERAS_BACKEND", "torch")

try:
    import keras
except ImportError as e:
    raise ImportError("mt.tkeras requires Keras 3. Please install keras>=3.") from e

kr_ver = Version(keras.__version__)
if kr_ver < Version("3.0"):
    raise ImportError(
        f"mt.tkeras requires Keras 3. Detected keras=={keras.__version__}."
    )

keras_version = keras.__version__
keras_source = "keras"
