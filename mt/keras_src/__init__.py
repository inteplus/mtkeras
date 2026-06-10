from packaging.version import Version

from .base import keras_version, keras_source, keras_backend, keras_package

# Handle imports based on Keras version and source
if keras_source == "keras3":
    # Keras 3 has backend-agnostic API
    from keras import *  # noqa: F401, F403
    from keras import backend
elif keras_source == "tf_keras":
    from tf_keras import *  # noqa: F401, F403
elif keras_source == "keras":
    from keras import *  # noqa: F401, F403
elif keras_source == "tensorflow.keras":
    from tensorflow.keras import *  # noqa: F401, F403
else:
    raise ImportError(
        f"Unknown value '{keras_source}' for variable 'keras_source'. "
        f"Valid options: 'keras3', 'keras', 'tf_keras', 'tensorflow.keras'"
    )

# Model file formats supported by each Keras version
d_modelFileFormats = {"H5": ".h5", "TF": ".tf"}

# Keras 2.15+, tf_keras, and Keras 3 support .keras format
if keras_source == "keras3":
    d_modelFileFormats["Keras"] = ".keras"
elif keras_source == "tf_keras":
    d_modelFileFormats["Keras"] = ".keras"
elif keras_source in ["keras", "tensorflow.keras"]:
    if Version(keras_version) >= Version("2.15"):
        d_modelFileFormats["Keras"] = ".keras"
