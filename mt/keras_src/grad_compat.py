"""Gradient utilities compatibility layer for Keras 2 and Keras 3.

This module provides decorators for custom gradient functions that work with both
Keras 2 (using TensorFlow) and Keras 3 (using keras backend).
"""

from .base import keras_source

if keras_source == "keras3":
    import keras

    def custom_gradient(f):
        """Decorator for custom gradient functions in Keras 3."""
        # Keras 3 uses a different API for custom gradients
        return keras.ops.custom_gradient(f)

else:
    import tensorflow as tf

    def custom_gradient(f):
        """Decorator for custom gradient functions in Keras 2."""
        return tf.custom_gradient(f)
