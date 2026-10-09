# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
"""MobileViT model.

Most of the code here has been ripped and updated off from the following
`Keras tutorial <https://keras.io/examples/vision/mobilevit/>`_. Please refer
to the `MobileViT ICLR2022 paper <https://arxiv.org/abs/2110.02178>`_ for more details.

The paper authors' code is `here <https://github.com/apple/ml-cvnets>`_.
"""

from mt import tp
from mt.base import model as base_model

from .mobilenet_v3_split import (
    MobileNetV3Input,
    _inverted_res_block,
    backend,
    models,
    layers,
)
from .. import activations as keras_activations
from ..ops_compat import ops


def conv_block(x, filters=16, kernel_size=3, strides=2):
    """Applies a Conv2D layer (``"same"`` padding, swish activation) to `x`.

    Parameters
    ----------
    x : tensor-like
        input tensor of shape ``(B, H, W, C)``
    filters : int, optional
        number of output channels. Defaults to 16.
    kernel_size : int or tuple, optional
        convolution window size. Defaults to 3.
    strides : int or tuple, optional
        convolution strides. Defaults to 2.

    Returns
    -------
    tensor-like
        output tensor of shape ``(B, ceil(H/strides), ceil(W/strides), filters)``
    """
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation="swish", padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC


def inverted_residual_block(
    x, expanded_channels, output_channels, strides=1, block_id=0
):
    """Applies a MobileNetV2-style inverted residual block (swish, no squeeze-excite) to `x`.

    This is a thin wrapper of ``_inverted_res_block`` of :mod:`mobilenet_v3_split`.

    Parameters
    ----------
    x : tensor-like
        input tensor with a statically known channel count
    expanded_channels : int
        number of channels after the expansion; the expansion factor passed down is
        ``expanded_channels // input_channels`` (integer division)
    output_channels : int
        number of output channels
    strides : int, optional
        stride of the depthwise convolution. Defaults to 1.
    block_id : int, optional
        block index used for layer naming. Must be positive.

    Returns
    -------
    tensor-like
        the output tensor of the block

    Raises
    ------
    NotImplementedError
        if `block_id` is 0
    """
    if block_id == 0:
        raise NotImplementedError(
            "Zero block id for _inverted_res_block() is not implemented in MobileViT."
        )

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    infilters = backend.int_shape(x)[channel_axis]

    m = _inverted_res_block(
        x,
        expanded_channels // infilters,  # expansion
        output_channels,  # filters
        3,  # kernel_size
        strides,  # stride
        0,  # se_ratio
        "swish",  # activation
        block_id,
    )

    return m


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/


def mlp(x, hidden_units, dropout_rate):
    """Applies a stack of Dense (swish) + Dropout layers to `x`.

    Parameters
    ----------
    x : tensor-like
        input tensor whose last axis is the feature axis
    hidden_units : iterable of int
        output size of each Dense layer, in order
    dropout_rate : float
        rate of the Dropout layer following each Dense layer

    Returns
    -------
    tensor-like
        output tensor, last axis of size ``hidden_units[-1]``
    """
    for units in hidden_units:
        x = layers.Dense(units, activation="swish")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    """Applies `transformer_layers` pre-norm transformer encoder layers to `x`.

    Each layer is: LayerNorm, MultiHeadAttention (``key_dim=projection_dim``, dropout 0.1), skip
    connection, LayerNorm, an :func:`mlp` of sizes ``[2*C, C]`` (``C`` = last axis of `x`), skip
    connection.

    Parameters
    ----------
    x : tensor-like
        input tensor of shape ``(B, ..., C)``; attention runs over the second-to-last axis
    transformer_layers : int
        number of encoder layers to stack
    projection_dim : int
        the ``key_dim`` of the attention layers
    num_heads : int, optional
        number of attention heads. Defaults to 2.

    Returns
    -------
    tensor-like
        output tensor with the same shape as `x`
    """
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(
            x3,
            hidden_units=[x.shape[-1] * 2, x.shape[-1]],
            dropout_rate=0.1,
        )
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    """Applies a MobileViT block (local convs, transformer on 2x2 patches, fusion) to `x`.

    The input is projected by two convolutions to `projection_dim` channels, unfolded into
    ``2x2`` interleaved patches, processed by :func:`transformer_block`, folded back, projected
    to the input channel count, concatenated with `x` and fused by a final 3x3 convolution to
    `projection_dim` channels.

    Parameters
    ----------
    x : tensor-like
        input tensor of shape ``(B, H, W, C)`` with static, even `H` and `W`
    num_blocks : int
        number of transformer layers
    projection_dim : int
        number of channels of the transformer features and of the output
    strides : int, optional
        strides of all the convolutions. Defaults to 1; the block assumes 1 for the fold/unfold
        reshapes to be consistent.

    Returns
    -------
    tensor-like
        output tensor of shape ``(B, H, W, projection_dim)`` when ``strides=1``

    Raises
    ------
    mt.base.model.ModelSyntaxError
        if the height or the width of `x` is not divisible by 2
    """
    cell_size = 2  # 2x2 for the Transformer block

    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    if x.shape[1] % cell_size != 0:
        raise base_model.ModelSyntaxError(
            f"Input tensor must have height divisible by {cell_size}. Got {x.shape}."
        )

    if x.shape[2] % cell_size != 0:
        raise base_model.ModelSyntaxError(
            f"Input tensor must have width divisible by {cell_size}. Got {x.shape}."
        )

    # Unfold into patches and then pass through Transformers.
    z = local_features  # (B,H,W,C)
    z = layers.Reshape(
        (
            z.shape[1] // cell_size,
            cell_size,
            z.shape[2] // cell_size,
            cell_size,
            projection_dim,
        )
    )(
        z
    )  # (B,H/P,P,W/P,P,C)
    z = ops.transpose(z, perm=[0, 2, 4, 1, 3, 5])  # (B,P,P,H/P,W/P,C)
    non_overlapping_patches = layers.Reshape(
        (cell_size * cell_size, z.shape[3] * z.shape[4], projection_dim)
    )(
        z
    )  # (B,P*P,H*W/(P*P),C)
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    z = layers.Reshape(
        (
            cell_size,
            cell_size,
            x.shape[1] // cell_size,
            x.shape[2] // cell_size,
            projection_dim,
        )
    )(
        global_features
    )  # (B,P,P,H/P,W/P,C)
    z = ops.transpose(z, perm=[0, 3, 1, 4, 2, 5])  # (B,H/P,P,W/P,P,C)
    folded_feature_map = layers.Reshape((x.shape[1], x.shape[2], projection_dim))(z)

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convolution layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


def create_mobilevit(
    input_shape=None,
    model_type: str = "XXS",
    output_all: bool = False,
    name: tp.Optional[str] = None,
):
    """Creates a MobileViT model (convolutional stem, MobileNetV2 blocks and MobileViT blocks).

    The model takes an image of pixel values in ``[0, 255]`` (a ``Rescaling(1/255)`` layer is
    included) and returns a *list* of feature maps. Spatial sizes shrink by a factor of 2 at
    the stem and at each of the 4 MV2 down-sampling blocks (total stride 32). The last MobileViT
    block needs an even size, so the input height and width should be divisible by 64.

    Parameters
    ----------
    input_shape : tuple, optional
        Shape tuple ``(height, width, 3)`` of the input image, passed to
        :func:`MobileNetV3Input`. It should have exactly 3 input channels. E.g. ``(160, 160, 3)``
        is a valid value. When None, the default of :func:`MobileNetV3Input` is used.
    model_type : {'XXS', 'XS', 'S'}, optional
        one of the 3 variants introduced in the paper. Defaults to ``'XXS'``.
    output_all : bool, optional
        If True, the model returns 5 output tensors, one per stage: after the first MV2 block,
        after the 3 MV2 blocks of the first downsampling stage, and after each of the three
        MobileViT blocks (the last one followed by a 1x1 conv expansion). Otherwise, it returns
        a list holding only the output tensor of the last stage. Defaults to False.
    name : str, optional
        model name, if any. Defaults to ``'MobileViT<model_type>'``.

    Returns
    -------
    tensorflow.keras.Model
        the MobileViT model, uninitialised and not compiled. Its outputs are a list of 1 tensor
        (``output_all=False``) or 5 tensors (``output_all=True``).

    Raises
    ------
    ValueError
        if `model_type` is not one of ``'XXS'``, ``'XS'``, ``'S'``
    mt.base.model.ModelSyntaxError
        if some intermediate feature map has an odd height or width

    Examples
    --------
    Building the model needs a Keras installation and is not executed here:

    .. code-block:: python

       from mt.keras.applications import create_mobilevit
       model = create_mobilevit((256, 256, 3), model_type="XXS")
       feats = model(images)  # list with 1 feature map
    """

    model_type_id = ["XXS", "XS", "S"].index(model_type)

    expansion_factor = 2 if model_type_id == 0 else 4

    inputs = MobileNetV3Input(input_shape=input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Initial conv-stem -> MV2 block.
    x = conv_block(x, filters=16)
    x = inverted_residual_block(
        x,
        expanded_channels=16 * expansion_factor,
        output_channels=16 if model_type_id == 0 else 32,
        block_id=1,
    )
    outputs = [x]

    # Downsampling with MV2 block.
    output_channels = [24, 48, 64][model_type_id]
    x = inverted_residual_block(
        x,
        expanded_channels=16 * expansion_factor,
        output_channels=output_channels,
        strides=2,
        block_id=2,
    )
    x = inverted_residual_block(
        x,
        expanded_channels=24 * expansion_factor,
        output_channels=output_channels,
        block_id=3,
    )
    x = inverted_residual_block(
        x,
        expanded_channels=24 * expansion_factor,
        output_channels=output_channels,
        block_id=4,
    )
    if output_all:
        outputs.append(x)
    else:
        outputs = [x]

    # First MV2 -> MobileViT block.
    output_channels = [48, 64, 96][model_type_id]
    projection_dim = [64, 96, 144][model_type_id]
    x = inverted_residual_block(
        x,
        expanded_channels=48 * expansion_factor,
        output_channels=output_channels,
        strides=2,
        block_id=5,
    )
    x = mobilevit_block(x, num_blocks=2, projection_dim=projection_dim)
    if output_all:
        outputs.append(x)
    else:
        outputs = [x]

    # Second MV2 -> MobileViT block.
    output_channels = [64, 80, 128][model_type_id]
    projection_dim = [80, 120, 192][model_type_id]
    x = inverted_residual_block(
        x,
        expanded_channels=64 * expansion_factor,
        output_channels=output_channels,
        strides=2,
        block_id=6,
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=projection_dim)
    if output_all:
        outputs.append(x)
    else:
        outputs = [x]

    # Third MV2 -> MobileViT block.
    output_channels = [80, 96, 160][model_type_id]
    projection_dim = [96, 144, 240][model_type_id]
    x = inverted_residual_block(
        x,
        expanded_channels=80 * expansion_factor,
        output_channels=output_channels,
        strides=2,
        block_id=7,
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=projection_dim)
    filters = [320, 384, 640][model_type_id]
    x = conv_block(x, filters=filters, kernel_size=1, strides=1)
    if output_all:
        outputs.append(x)
    else:
        outputs = [x]

    if name is None:
        name = f"MobileViT{model_type}"
    return models.Model(inputs, outputs, name=name)
