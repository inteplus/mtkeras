"""Backend-agnostic operations compatibility layer for Keras 2 and Keras 3.

This module provides a unified interface for operations that works with both:
- Keras 2 (using TensorFlow operations)
- Keras 3 (using keras.ops backend-agnostic API)

Usage:
    from .ops_compat import ops

    result = ops.reshape(x, shape)  # Works with any backend
"""

from .base import keras_source

if keras_source == "keras3":
    # For Keras 3, use the backend-agnostic API
    import keras.ops as ops_impl
else:
    # For Keras 2, create a wrapper around TensorFlow operations
    import tensorflow as tf

    class _TFOpsWrapper:
        """Wrapper to expose TensorFlow operations for Keras 2."""

        def shape(self, x):
            return tf.shape(x)

        def reshape(self, x, shape):
            return tf.reshape(x, shape)

        def reduce_sum(self, x, axis=None, keepdims=False):
            return tf.reduce_sum(x, axis=axis, keepdims=keepdims)

        def reduce_mean(self, x, axis=None, keepdims=False):
            return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

        def reduce_prod(self, x, axis=None, keepdims=False):
            return tf.reduce_prod(x, axis=axis, keepdims=keepdims)

        def expand_dims(self, x, axis):
            return tf.expand_dims(x, axis=axis)

        def concatenate(self, xs, axis=0):
            return tf.concat(xs, axis=axis)

        def squeeze(self, x, axis=None):
            return tf.squeeze(x, axis=axis)

        def stop_gradient(self, x):
            return tf.stop_gradient(x)

        def ones(self, shape, dtype=None):
            return tf.ones(shape, dtype=dtype)

        def zeros(self, shape, dtype=None):
            return tf.zeros(shape, dtype=dtype)

        def constant(self, value, dtype=None):
            return tf.constant(value, dtype=dtype)

        def cast(self, x, dtype):
            return tf.cast(x, dtype)

        def abs(self, x):
            return tf.abs(x)

        def pow(self, x, y):
            return tf.pow(x, y)

        def tanh(self, x):
            return tf.tanh(x)

        def floor(self, x):
            return tf.math.floor(x)

        def sqrt(self, x):
            return tf.sqrt(x)

        def matmul(self, a, b):
            return tf.matmul(a, b)

        def stack(self, xs, axis=0):
            return tf.stack(xs, axis=axis)

        def transpose(self, x, perm=None):
            return tf.transpose(x, perm=perm)

        def tile(self, x, multiples):
            return tf.tile(x, multiples)

        def pad(self, x, paddings, mode="CONSTANT", constant_values=0):
            return tf.pad(x, paddings, mode=mode, constant_values=constant_values)

        def einsum(self, subscripts, *operands):
            return tf.einsum(subscripts, *operands)

    ops_impl = _TFOpsWrapper()

# Export the ops interface
ops = ops_impl
