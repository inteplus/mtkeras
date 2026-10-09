from .. import layers, initializers
from ..ops_compat import ops


class Counter(layers.Layer):
    """A layer that counts training steps, returning the current count as a tensor.

    The layer owns a weight ``counter`` of shape ``(1,)`` initialised to
    0. Every time the layer is called with a truthy ``training`` argument, the counter is
    incremented by 1. When ``training`` is false (the default) the counter is left unchanged. In
    both cases the call returns the current counter value. The input tensor is only used to tie
    the output to the graph (its value never affects the result, as it is multiplied by 0 with
    gradients stopped).

    Input shape
    -----------
    Any shape.

    Output shape
    ------------
    ``(1,)``, a float tensor holding the counter value, whatever the input shape is.

    Notes
    -----
    The layer takes no constructor argument other than the standard
    :class:`keras.layers.Layer` keyword arguments.

    See Also
    --------
    :class:`NormedConv2D` : uses a counter to anneal from a plain convolution to a normalised one.
    """

    def build(self, input_shape):
        """Creates the ``counter`` weight (shape ``(1,)``, initialised to 0)."""
        initializer = initializers.Constant(value=0.0)
        self.counter = self.add_weight(
            name="counter", shape=(1,), initializer=initializer
        )
        self.incrementor = ops.constant([1.0])

    def call(self, x, training: bool = False):
        """Increments the counter if ``training`` is truthy; returns the counter, shape ``(1,)``."""
        if training:
            self.counter.assign_add(self.incrementor)
        y = ops.reshape(x, [-1])[:1]
        y = ops.stop_gradient(y) * 0.0
        return self.counter + y

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        """Returns ``(1,)`` regardless of ``input_shape``."""
        return (1,)

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__
