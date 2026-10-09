from .. import layers
from ..ops_compat import ops


class SoftBend(layers.Layer):
    """Soft bend activation layer.

    Function: ``|x|^alpha * tanh(x)``, bending the linear activation a bit. The function is odd
    (``f(-x) = -f(x)``) and passes through the origin.

    - If alpha is less than 1, it acts as a soft squash.
    - If alpha is greater than 1, it acts as a soft explode.
    - If alpha is 1, it is ``|x| * tanh(x)``, which is close to the linear activation function
      near zero.

    Parameters
    ----------
    alpha : float, optional
        the bending coefficient (the exponent applied to ``|x|``). Defaults to 0.5.
    **kwds : dict
        keyword arguments passed as-is to the super class (e.g. ``name``)

    Input shape
    -----------
    Any shape, floating-point dtype.

    Output shape
    ------------
    Same as the input shape.

    Notes
    -----
    The layer has no weights. ``alpha`` is serialised by :meth:`get_config`.
    """

    def __init__(self, alpha: float = 0.5, **kwds):
        super(SoftBend, self).__init__(**kwds)

        self.alpha = alpha

    def call(self, x):
        """Returns ``|x|^alpha * tanh(x)`` element-wise."""
        return ops.pow(ops.abs(x), self.alpha) * ops.tanh(x)

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        """Returns `input_shape` unchanged."""
        return input_shape

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__

    def get_config(self):
        """Returns the layer config: the base config plus ``alpha``."""
        config = {"alpha": self.alpha}
        base_config = super(SoftBend, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = layers.Layer.get_config.__doc__
