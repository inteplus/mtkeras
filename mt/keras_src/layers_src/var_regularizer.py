from mt import tp

from .. import layers
from ..ops_compat import ops


class VarianceRegularizer(layers.Layer):
    """A regularizer on the variance of the input tensor.

    The layer is a pass-through: it returns its input unchanged but adds an activity-style loss
    ``rate * sum(var)`` to the model via ``add_loss``, where ``var`` is the variance of the input
    over the axes `l_axes` (computed with ``keepdims=False``, the remaining entries then summed
    over all remaining axes including the batch axis).

    A positive rate penalises large variance (makes the variance smaller). A negative rate
    rewards large variance (makes the variance larger).

    Parameters
    ----------
    rate : float, optional
        the regularisation coefficient. Defaults to 0.01.
    l_axes : list of int, optional
        the axes over which the variance is computed. Defaults to ``[-1]`` (the last axis).
        Note that the default list is shared between instances; do not mutate it.
    **kwargs : dict
        keyword arguments passed as-is to :class:`keras.layers.Layer`

    Input shape
    -----------
    Any shape.

    Output shape
    ------------
    Same as the input shape (the input is returned as is).
    """

    def __init__(self, rate=1e-2, l_axes: list = [-1], **kwargs):
        super(VarianceRegularizer, self).__init__(**kwargs)
        self.rate = rate
        self.l_axes = l_axes

    def call(self, x):
        """Adds ``rate * sum(variance of x over l_axes)`` to the layer losses; returns `x`."""
        mean = ops.mean(x, axis=self.l_axes, keepdims=True)
        err = x - mean
        esq = err * err
        var = ops.mean(esq, axis=self.l_axes)
        sum_var = ops.sum(var)
        self.add_loss(tp.cast(tp.Any, self.rate * sum_var))
        return x

    call.__doc__ = layers.Layer.call.__doc__

    def get_config(self):
        """Returns the layer config: the base config plus ``rate`` and ``l_axes``."""
        config = {
            "rate": self.rate,
            "l_axes": self.l_axes,
        }
        base_config = super(VarianceRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    get_config.__doc__ = layers.Layer.get_config.__doc__
