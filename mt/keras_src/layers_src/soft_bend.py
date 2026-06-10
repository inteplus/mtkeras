from .. import layers
from ..ops_compat import ops


class SoftBend(layers.Layer):
    """Soft bend activation layer.

    Function: `|x|^alpha * tanh(x)`, bending the linear activation a bit.

    If alpha is less than 1, it acts as a soft squash.
    If alpha is greater than 1, it acts as a soft explode.
    If alpha is 1, it acts as the linear activation function.

    Parameters
    ----------
    alpha : float
        the bending coefficient
    **kwds : dict
        keyword arguments passed as-is to the super class
    """

    def __init__(self, alpha: float = 0.5, **kwds):
        super(SoftBend, self).__init__(**kwds)

        self.alpha = alpha

    def call(self, x):
        return ops.pow(ops.abs(x), self.alpha) * ops.tanh(x)

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        return input_shape

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__
