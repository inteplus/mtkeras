from .. import layers
from ..grad_compat import custom_gradient
from ..ops_compat import ops


@custom_gradient
def floor(x):
    """Element-wise floor with a straight-through (identity) gradient.

    The forward pass returns ``floor(x)``. In the backward pass the upstream gradient is passed
    through unchanged, as if the function were the identity (straight-through estimator).

    Parameters
    ----------
    x : tensor-like
        input tensor of any shape with a floating-point dtype

    Returns
    -------
    tensor-like
        tensor of the same shape and dtype as `x` containing the floored values
    """
    def grad(upstream):  # identity
        return upstream

    return ops.floor(x), grad


class Floor(layers.Layer):
    """Backend-agnostic element-wise floor layer with an identity (straight-through) gradient.

    The layer has no weights and no constructor argument other than the standard
    :class:`keras.layers.Layer` keyword arguments. It rounds every element down to the nearest
    integer value (the dtype stays floating-point), but lets gradients flow as if it were the
    identity, which makes it usable for quantisation inside trainable models.

    Input shape
    -----------
    Any shape, floating-point dtype.

    Output shape
    ------------
    Same as the input shape.

    See Also
    --------
    :func:`floor` : the underlying function.
    """

    def call(self, x):
        """Returns :func:`floor` applied to `x`, with identity gradient."""
        return floor(x)

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        """Returns `input_shape` unchanged."""
        return input_shape

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__
