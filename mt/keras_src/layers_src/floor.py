from .. import layers
from ..grad_compat import custom_gradient
from ..ops_compat import ops


@custom_gradient
def floor(x):
    def grad(upstream):  # identity
        return upstream

    return ops.floor(x), grad


class Floor(layers.Layer):
    """Backend-agnostic floor with identity gradient."""

    def call(self, x):
        return floor(x)

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        return input_shape

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__
