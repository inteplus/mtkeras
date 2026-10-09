from .. import layers


class Identical(layers.Layer):
    """An identity layer that returns its input unchanged, mainly for renaming purposes.

    Wrapping a tensor with this layer creates a new graph node with a name of your choice, e.g.
    to give a model output a stable name. It has no weights and no constructor argument other
    than the standard :class:`keras.layers.Layer` keyword arguments (such as ``name``).

    Input shape
    -----------
    Any shape.

    Output shape
    ------------
    Same as the input shape.
    """

    def call(self, x):
        """Returns `x` as is."""
        return x

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        """Returns `input_shape` unchanged."""
        return input_shape

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__
