from keras import ops
from .. import layers


def floor(x):
    return x + ops.stop_gradient(ops.floor(x) - x)


class Floor(layers.Layer):
    """TensorFlow floor but gradient is identity."""

    def call(self, x):
        return floor(x)

    call.__doc__ = layers.Layer.call.__doc__

    def compute_output_shape(self, input_shape):
        return input_shape

    compute_output_shape.__doc__ = layers.Layer.compute_output_shape.__doc__
