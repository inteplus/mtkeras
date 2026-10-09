"""Gradient utilities compatibility layer for Keras 2 and Keras 3.

This module provides a single decorator, :func:`custom_gradient`, that works with both
Keras 2 (using ``tf.custom_gradient``) and Keras 3 (using ``keras.ops.custom_gradient``).

Usage::

    from mt.keras_src.grad_compat import custom_gradient

    @custom_gradient
    def my_floor(x):
        def grad(upstream):
            return upstream  # straight-through gradient
        return ops.floor(x), grad

The decorated function must return ``(output, grad_fn)`` where ``grad_fn`` maps the upstream
gradient to the gradient with respect to the inputs.
"""

from .base import keras_source

if keras_source == "keras3":
    import keras

    def custom_gradient(f):
        """Decorator for custom gradient functions in Keras 3.

        Parameters
        ----------
        f : callable
            function returning ``(output, grad_fn)``

        Returns
        -------
        callable
            the function wrapped with :func:`keras.ops.custom_gradient`
        """
        # Keras 3 uses a different API for custom gradients
        return keras.ops.custom_gradient(f)

else:
    import tensorflow as tf

    def custom_gradient(f):
        """Decorator for custom gradient functions in Keras 2.

        Parameters
        ----------
        f : callable
            function returning ``(output, grad_fn)``

        Returns
        -------
        callable
            the function wrapped with ``tf.custom_gradient``
        """
        return tf.custom_gradient(f)
