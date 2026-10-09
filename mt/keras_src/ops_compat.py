"""Backend-agnostic operations compatibility layer for Keras 2 and Keras 3.

This module provides a unified interface for operations that works with both:
- Keras 2 (using TensorFlow operations)
- Keras 3 (using keras.ops backend-agnostic API)

Usage:
    from .ops_compat import ops

    result = ops.reshape(x, shape)  # Works with any backend

The exported object ``ops`` exposes a TensorFlow-flavoured subset of operations: ``shape``,
``reshape``, ``reduce_sum``/``sum``, ``reduce_mean``/``mean``, ``reduce_prod``, ``expand_dims``,
``concatenate``, ``squeeze``, ``stop_gradient``, ``ones``, ``zeros``, ``constant``, ``cast``,
``abs``, ``pow``, ``tanh``, ``floor``, ``sqrt``, ``matmul``, ``stack``, ``transpose`` (with a
``perm`` argument), ``tile``, ``pad`` and ``einsum``. With Keras 3, any other ``keras.ops``
function is also reachable as an attribute; with Keras 2 only the functions listed above exist.
"""

from .base import keras_source

if keras_source == "keras3":
    # For Keras 3, use the backend-agnostic API
    import keras.ops as _keras_ops

    class _Keras3OpsWrapper:
        """Exposes ``keras.ops`` plus TensorFlow-style aliases (``reduce_*``, ``constant``, ...).

        Unknown attributes are forwarded to ``keras.ops``. ``transpose`` takes ``perm`` (as in
        TensorFlow) and is forwarded as ``axes``.
        """

        def __getattr__(self, name):
            return getattr(_keras_ops, name)

        def reduce_sum(self, x, axis=None, keepdims=False):
            return _keras_ops.sum(x, axis=axis, keepdims=keepdims)

        def reduce_mean(self, x, axis=None, keepdims=False):
            return _keras_ops.mean(x, axis=axis, keepdims=keepdims)

        def reduce_prod(self, x, axis=None, keepdims=False):
            return _keras_ops.prod(x, axis=axis, keepdims=keepdims)

        def transpose(self, x, perm=None):
            return _keras_ops.transpose(x, axes=perm)

        def constant(self, value, dtype=None):
            # keras.ops has no `constant`; convert_to_tensor is the equivalent
            return _keras_ops.convert_to_tensor(value, dtype=dtype)

        def pow(self, x, y):
            # keras.ops uses `power`, not `pow`
            return _keras_ops.power(x, y)

    ops_impl = _Keras3OpsWrapper()
else:
    # For Keras 2, create a wrapper around TensorFlow operations
    import tensorflow as tf

    class _TFOpsWrapper:
        """Exposes a fixed subset of TensorFlow operations under ``keras.ops``-like names.

        ``sum``/``mean`` are aliases of ``reduce_sum``/``reduce_mean``, ``concatenate`` is
        ``tf.concat`` and ``floor`` is ``tf.math.floor``.
        """

        def shape(self, x):
            return tf.shape(x)

        def reshape(self, x, shape):
            return tf.reshape(x, shape)

        def reduce_sum(self, x, axis=None, keepdims=False):
            return tf.reduce_sum(x, axis=axis, keepdims=keepdims)

        def sum(self, x, axis=None, keepdims=False):
            return tf.reduce_sum(x, axis=axis, keepdims=keepdims)

        def reduce_mean(self, x, axis=None, keepdims=False):
            return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

        def mean(self, x, axis=None, keepdims=False):
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
