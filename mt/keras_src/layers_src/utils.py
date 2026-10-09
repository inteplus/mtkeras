"""Functional-style helpers that build small Conv2D blocks on a tensor.

Both :func:`conv2d` and :func:`dense2d` create layers and apply them immediately, so they are
meant to be used while wiring a functional model (they do not return a layer).
"""

from mt import tp
from mt.base import model as base_model


def conv2d(name_scope: base_model.NameScope, x, filters, kernel_size, **kwargs):
    """Applies a LayerNormalization layer followed by a Conv2D layer (pre-norm convolution).

    Layers are created on the fly and applied to `x`; the name scope is advanced by one
    (``next(name_scope)``) and the layers are named ``<scope>prenorm`` and ``<scope>conv``.

    Parameters
    ----------
    name_scope : mt.base.model.NameScope
        the name scope. For every conv2d invocation, the name scope is iterated.
    x : tensor-like
        Keras tensor or TF tensor as input, shape ``(batch, height, width, channels)``
    filters : int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    **kwargs : dict
        all other keyword arguments to be passed as-is to Conv2D layer construction

    Returns
    -------
    tensor-like
        TF tensor as output, shape ``(batch, new_height, new_width, filters)``
    """

    from .. import layers

    next(name_scope)
    x = layers.LayerNormalization(name=name_scope("prenorm"))(x)
    x = layers.Conv2D(filters, kernel_size, name=name_scope("conv"), **kwargs)(x)

    return x


def dense2d(
    name_scope: base_model.NameScope,
    x,
    filters,
    kernel_size,
    activation="tanh",
    **kwargs
):
    """Applies a pre-norm "expand then project" convolutional block to `x`.

    The block is: LayerNormalization, a 1x1 Conv2D expanding the channels to twice the input
    channels (``x.shape[3] * 2``, ReLU activation), LayerNormalization, then a Conv2D with
    `filters` output channels, `kernel_size` and `activation`. The name scope is advanced by one.
    Note that `kwargs` are passed to both convolutions.

    Parameters
    ----------
    name_scope : mt.base.model.NameScope
        the name scope. For every dense2d invocation, the name scope is iterated.
    x : tensor-like
        Keras tensor or TF tensor as input, shape ``(batch, height, width, channels)`` with a
        statically known channel count
    filters : int
        The dimensionality of the output space (i.e. the number of output filters in the
        convolution).
    kernel_size : int or tuple or list
        An integer or tuple/list of 2 integers, specifying the height and width of the 2D
        convolution window. Can be a single integer to specify the same value for all spatial
        dimensions.
    activation : object, optional
        the activation of the last conv layer. Defaults to ``"tanh"``.
    **kwargs : dict
        all other keyword arguments to be passed as-is to both Conv2D layer constructions

    Returns
    -------
    tensor-like
        TF tensor as output, shape ``(batch, new_height, new_width, filters)``
    """

    from .. import layers

    next(name_scope)
    x = layers.LayerNormalization(name=name_scope("expand_prenorm"))(x)
    x = layers.Conv2D(
        x.shape[3] * 2, 1, name=name_scope("expand"), activation="relu", **kwargs
    )(x)
    x = layers.LayerNormalization(name=name_scope("project_prenorm"))(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        name=name_scope("project"),
        activation=activation,
        **kwargs
    )(x)

    return x
