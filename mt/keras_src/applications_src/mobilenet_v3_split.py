# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
"""MobileNet v3 models split into 5 submodels.

The MobileNetV3 model is split into 5 parts:

  - The input parser block that downsamples once (:func:`MobileNetV3Parser`).
  - Block 0 to 3 that downample once for each block (:func:`MobileNetV3LargeBlock`
    or :func:`MobileNetV3SmallBlock`). As of 2023/05/15, there's a possibility to have block 4
    for MobileNetV3Large.
  - The mixer block that turns the downsampled grid into a (1,1,feat_dim) batch
    (:func:`MobileNetV3Mixer`).
  - Optionally the output block that may or may not contain the clasification head
    (:func:`MobileNetV3Output`).

Input arguments follow those of MobileNetV3. One can also use :func:`MobileNetV3Split` to create
a model of submodels that is theoretically equivalent to the original MobileNetV3 model. However,
no pre-trained weights exist.
"""

import importlib

from mt import tp
from mt.base import model as base_model
from .. import keras_source

_name_sep = "_" if keras_source == "keras3" else "/"


def _import_mobilenet_v3_module(candidates):
    """Imports and returns the first importable module among the module names `candidates`.

    Raises ``ImportError`` (mentioning the last error) if none can be imported.
    """
    last_error = None
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except Exception as error:
            last_error = error
    raise ImportError(
        "Unable to import MobileNetV3 implementation from any known location. "
        f"Tried: {candidates}. Last error: {last_error}"
    )


if keras_source == "keras3":
    keras_pkg = importlib.import_module("keras")
    backend = keras_pkg.backend
    models = keras_pkg.models
    layers = keras_pkg.layers

    mobilenet_v3_mod = _import_mobilenet_v3_module(
        [
            "keras.src.applications.mobilenet_v3",
            "keras.applications.mobilenet_v3",
        ]
    )
elif keras_source == "keras":
    keras_pkg = importlib.import_module("keras")
    backend = keras_pkg.backend
    models = keras_pkg.models
    layers = keras_pkg.layers

    mobilenet_v3_mod = _import_mobilenet_v3_module(
        [
            "keras.applications.mobilenet_v3",
            "keras.src.applications.mobilenet_v3",
        ]
    )
elif keras_source == "tf_keras":
    keras_pkg = importlib.import_module("tf_keras")
    backend = keras_pkg.backend
    models = keras_pkg.models
    layers = keras_pkg.layers

    mobilenet_v3_mod = _import_mobilenet_v3_module(
        [
            "tf_keras.src.applications.mobilenet_v3",
            "tf_keras.applications.mobilenet_v3",
        ]
    )
elif keras_source == "tensorflow.keras":
    keras_pkg = importlib.import_module("tensorflow.keras")
    backend = keras_pkg.backend
    models = keras_pkg.models
    layers = keras_pkg.layers

    mobilenet_v3_mod = _import_mobilenet_v3_module(
        [
            "tensorflow.keras.src.applications.mobilenet_v3",
            "tensorflow.keras.applications.mobilenet_v3",
        ]
    )
else:
    raise ImportError(f"Unknown value '{keras_source}' for variable 'keras_source'.")

relu = mobilenet_v3_mod.relu
hard_swish = mobilenet_v3_mod.hard_swish
_depth = mobilenet_v3_mod._depth
_inverted_res_block = mobilenet_v3_mod._inverted_res_block


def MobileNetV3Input(
    input_shape=None,
):
    """Prepares a MobileNetV3 input layer.

    Parameters
    ----------
    input_shape : tuple, optional
        shape of one image without the batch axis, e.g. ``(224, 224, 3)`` for channels-last data.
        Height and width may be None. Defaults to ``(None, None, 3)``.

    Returns
    -------
    keras.KerasTensor
        the output of ``keras.layers.Input(shape=input_shape)``

    Raises
    ------
    mt.base.model.ModelSyntaxError
        if both the height and the width are known and either is smaller than 32
    """

    # If input_shape is None and input_tensor is None using standard shape
    if input_shape is None:
        input_shape = (None, None, 3)

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise base_model.ModelSyntaxError(
            f"Input size must be at least 32x32; got `input_shape={input_shape}`"
        )

    img_input = layers.Input(shape=input_shape)
    return img_input


def MobileNetV3Parser(
    img_input,
    model_type: str = "Large",  # only 'Small' or 'Large' are accepted
    minimalistic=False,
):
    """Prepares a MobileNetV3 parser block (the stem).

    The block is ``Rescaling`` (maps pixel values in ``[0, 255]`` to ``[-1, 1]``), a 3x3 stride-2
    Conv2D with 16 filters, BatchNormalization and an activation. It downsamples once.

    Parameters
    ----------
    img_input : keras.KerasTensor
        the image input tensor, typically the output of :func:`MobileNetV3Input`
    model_type : {'Small', 'Large'}, optional
        only used to name the model, ``MobileNetV3<model_type>Parser``. Defaults to ``'Large'``.
    minimalistic : bool, optional
        if True, ReLU is used as activation instead of hard-swish. Defaults to False.

    Returns
    -------
    keras.Model
        a model from `img_input` to a tensor of shape ``(B, H/2, W/2, 16)``
    """

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    if minimalistic:
        activation = relu
    else:
        activation = hard_swish

    x = img_input
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
    x = layers.Conv2D(
        16, kernel_size=3, strides=(2, 2), padding="same", use_bias=False, name="Conv"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name="Conv" + _name_sep + "BatchNorm",
    )(x)
    x = activation(x)

    # Create model.
    model = models.Model(img_input, x, name=f"MobileNetV3{model_type}Parser")

    return model


def MobileNetV3SmallBlock(
    block_id: int,  # only 0 to 3 are accepted here
    input_tensor,  # input tensor for the block
    alpha=1.0,
    minimalistic=False,
):
    """Prepares a MobileNetV3Small downsampling block.

    Each block downsamples once (stride 2) and consists of 1 (block 0), 2 (block 1), 5 (block 2)
    or 3 (block 3) inverted residual blocks, with output channels (before `alpha`) 16, 24, 48 and
    96 respectively.

    Parameters
    ----------
    block_id : int
        the block index in 0..3. Any value other than 0, 1 or 2 builds block 3.
    input_tensor : keras.KerasTensor
        input tensor of the block, e.g. the output of the previous block
    alpha : float, optional
        the width multiplier applied to the number of filters. Defaults to 1.0.
    minimalistic : bool, optional
        if True, uses 3x3 kernels, ReLU and no squeeze-and-excite. Defaults to False.

    Returns
    -------
    keras.Model
        a model from `input_tensor` to the block output, named ``MobileNetV3SmallBlock<id>``
    """

    def depth(d):
        return _depth(d * alpha)

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = input_tensor
    if block_id == 0:
        x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
    elif block_id == 1:
        x = _inverted_res_block(x, 72.0 / 16, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88.0 / 24, depth(24), 3, 1, None, relu, 2)
    elif block_id == 2:
        x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)
    else:
        x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
        x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)

    # Create model.
    model = models.Model(input_tensor, x, name=f"MobileNetV3SmallBlock{block_id}")

    return model


def MobileNetV3LargeBlock(
    block_id: int,  # only 0 to 4 are accepted here. 4 is only available as of 2023/05/15
    input_tensor,  # input tensor for the block
    alpha=1.0,
    minimalistic=False,
):
    """Prepares a MobileNetV3Large downsampling block.

    Each block downsamples once (stride 2) and consists of 3 inverted residual blocks (blocks 0,
    1, 3 and 4) or 6 (block 2). Output channels (before `alpha`) are 24, 40, 112, 160 and 320 for
    blocks 0 to 4 respectively. Block 4 is an MT addition not in the original MobileNetV3.

    Parameters
    ----------
    block_id : int
        the block index in 0..4. Any value other than 0, 1, 2 or 3 builds block 4.
    input_tensor : keras.KerasTensor
        input tensor of the block, e.g. the output of the previous block
    alpha : float, optional
        the width multiplier applied to the number of filters. Defaults to 1.0.
    minimalistic : bool, optional
        if True, uses 3x3 kernels, ReLU and no squeeze-and-excite. Defaults to False.

    Returns
    -------
    keras.Model
        a model from `input_tensor` to the block output, named ``MobileNetV3LargeBlock<id>``
    """

    def depth(d):
        return _depth(d * alpha)

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = input_tensor
    if block_id == 0:
        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
    elif block_id == 1:
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
    elif block_id == 2:
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11)
    elif block_id == 3:
        x = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13)
        x = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14)
    else:
        x = _inverted_res_block(x, 6, depth(320), kernel, 2, se_ratio, activation, 15)
        x = _inverted_res_block(x, 6, depth(320), kernel, 1, se_ratio, activation, 16)
        x = _inverted_res_block(x, 6, depth(320), kernel, 1, se_ratio, activation, 17)

    # Create model.
    model = models.Model(input_tensor, x, name=f"MobileNetV3LargeBlock{block_id}")

    return model


def MobileNetV3Mixer(
    input_tensor,
    params: base_model.MobileNetV3MixerParams,
    last_point_ch,
    alpha=1.0,
    model_type: str = "Large",  # only 'Small' or 'Large' are accepted
    minimalistic=False,
):
    """Prepares a MobileNetV3 mixer block, turning the feature grid into a global feature.

    The behaviour depends on ``params.variant``:

    - ``"mobilenet"``: the original head (1x1 conv, BatchNorm, activation, global average
      pooling, then a 1x1 conv to `last_point_ch` channels). Output shape ``(B, 1, 1, C)``.
    - ``"maxpool"``: global max pooling, output shape ``(B, C)``.
    - ``"mhapool"``: a cascade of :class:`~mt.keras.layers.MHAPool2D` layers (each preceded by a
      LayerNormalization) until the grid reaches 1x1, configured by
      ``params.mhapool_cascade_params``. After ``max_num_pooling_layers`` layers, global max
      pooling is used. Requires channels-last data.

    Parameters
    ----------
    input_tensor : keras.KerasTensor
        the feature grid, shape ``(B, H, W, C)``
    params : mt.base.model.MobileNetV3MixerParams
        parameters defining the mixer variant
    last_point_ch : int
        number of channels of the last 1x1 conv (``"mobilenet"`` variant only); multiplied by
        `alpha` when ``alpha > 1``
    alpha : float, optional
        the width multiplier. Defaults to 1.0.
    model_type : {'Small', 'Large'}, optional
        only used to name the model, ``MobileNetV3<model_type>Mixer``. Defaults to ``'Large'``.
    minimalistic : bool, optional
        if True, uses ReLU instead of hard-swish. Defaults to False.

    Returns
    -------
    keras.Model
        a model from `input_tensor` to the mixer output

    Raises
    ------
    mt.base.model.ModelSyntaxError
        if the channel count cannot be inferred, the variant is unknown, or ``"mhapool"`` is
        requested with channels-first data or with invalid ``mhapool_cascade_params``

    Notes
    -----
    As written, only the ``"mhapool"`` variant assigns the ``outputs`` list passed to the model;
    the ``"mobilenet"`` and ``"maxpool"`` variants end up with an unbound ``outputs`` variable
    (``UnboundLocalError``). In ``"mhapool"`` the per-block ``activation`` is computed but not
    passed to :class:`~mt.keras.layers.MHAPool2D`.
    """

    x = input_tensor
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    if params.variant == "mobilenet":

        if minimalistic:
            kernel = 3
            activation = relu
            se_ratio = None
        else:
            kernel = 5
            activation = hard_swish
            se_ratio = 0.25

        input_channels = x.shape[channel_axis]
        if input_channels is None:
            raise base_model.ModelSyntaxError(
                "Could not infer channel dimension for MobileNetV3 mixer input."
            )
        last_conv_ch = _depth(int(input_channels) * 6)

        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if alpha > 1.0:
            last_point_ch = _depth(last_point_ch * alpha)
        x = layers.Conv2D(
            last_conv_ch, kernel_size=1, padding="same", use_bias=False, name="Conv_1"
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name="Conv_1" + _name_sep + "BatchNorm",
        )(x)
        x = activation(x)
        x = layers.GlobalAveragePooling2D()(x)
        if channel_axis == 1:
            x = layers.Reshape((last_conv_ch, 1, 1))(x)
        else:
            x = layers.Reshape((1, 1, last_conv_ch))(x)
        x = layers.Conv2D(
            last_point_ch, kernel_size=1, padding="same", use_bias=True, name="Conv_2"
        )(x)
        x = activation(x)
    elif params.variant == "maxpool":
        x = layers.GlobalMaxPool2D()(x)
    elif params.variant == "mhapool":
        if backend.image_data_format() == "channels_first":
            raise base_model.ModelSyntaxError(
                "Mixer variant 'mhapool' requires channels_last image data format."
            )

        mhapool_params = params.mhapool_cascade_params
        if not isinstance(mhapool_params, base_model.MHAPool2DCascadeParams):
            raise base_model.ModelSyntaxError(
                "Parameter 'params.mhapool_cascade_params' is not of type "
                f"mt.base.model.MHAPool2DCascadeParams. Got: {type(mhapool_params)}."
            )

        from ..layers_src import MHAPool2D

        n_heads = mhapool_params.n_heads
        k = 0
        outputs = []
        while True:
            h = x.shape[1]
            w = x.shape[2]

            if h <= 1 and w <= 1:
                break

            c = x.shape[3]
            key_dim = (c + n_heads - 1) // n_heads
            value_dim = int(key_dim * mhapool_params.expansion_factor)
            k += 1
            block_name = f"MHAPool2DCascade_block{k}"
            if k > mhapool_params.max_num_pooling_layers:  # GlobalMaxPool2D
                x = layers.GlobalMaxPooling2D(
                    keepdims=True, name=block_name + _name_sep + "GlobalMaxPool"
                )(x)
            else:  # MHAPool2D
                x = layers.LayerNormalization()(x)
                if h <= 2 and w <= 2:
                    activation = mhapool_params.final_activation
                else:
                    activation = mhapool_params.activation
                x = MHAPool2D(
                    n_heads,
                    key_dim,
                    value_dim=value_dim,
                    pooling=mhapool_params.pooling,
                    dropout=mhapool_params.dropout,
                    name=block_name + _name_sep + "MHAPool",
                )(x)

            if mhapool_params.output_all:
                outputs.append(x)
            else:
                outputs = [x]
    else:
        raise base_model.ModelSyntaxError(
            f"Unknown mixer variant: '{params.variant}'."
        )

    # Create model.
    model = models.Model(
        input_tensor, outputs, name=f"MobileNetV3{model_type}Mixer"
    )

    return model


def MobileNetV3Output(
    input_tensor,
    model_type: str = "Large",  # only 'Small' or 'Large' are accepted
    include_top=True,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
):
    """Prepares a MobileNetV3 output block (the classification head or a global pooling).

    Parameters
    ----------
    input_tensor : keras.KerasTensor
        output of the mixer, of shape ``(B, 1, 1, C)`` if `include_top` is True
    model_type : {'Small', 'Large'}, optional
        only used to name the model, ``MobileNetV3<model_type>Output``. Defaults to ``'Large'``.
    include_top : bool, optional
        if True, the block is ``[Dropout] -> Conv2D(classes, 1) -> Flatten -> Activation``,
        giving shape ``(B, classes)``. Defaults to True.
    classes : int, optional
        number of classes, when `include_top` is True. Defaults to 1000.
    pooling : {None, 'avg', 'max'}, optional
        when `include_top` is False, the global pooling to apply. Defaults to None.
    dropout_rate : float, optional
        dropout rate before the logits conv when `include_top` is True; no dropout if not
        positive. Defaults to 0.2.
    classifier_activation : str or callable, optional
        activation of the top layer. Defaults to ``"softmax"``.

    Returns
    -------
    keras.Model or None
        the output block model, or None if `include_top` is False and `pooling` is neither
        ``'avg'`` nor ``'max'``
    """

    x = input_tensor
    if include_top:
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes, kernel_size=1, padding="same", name="Logits")(x)
        x = layers.Flatten()(x)
        x = layers.Activation(activation=classifier_activation, name="Predictions")(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)
        else:
            return None

    # Create model.
    model = models.Model(input_tensor, x, name=f"MobileNetV3{model_type}Output")

    return model


def MobileNetV3Split(
    input_shape=None,
    alpha: float = 1.0,
    model_type: str = "Large",
    max_n_blocks: int = 6,
    minimalistic: bool = False,
    mixer_params: tp.Optional[base_model.MobileNetV3MixerParams] = None,
    include_top: bool = True,
    pooling=None,
    classes: int = 1000,
    dropout_rate: float = 0.2,
    classifier_activation="softmax",
    output_all: bool = False,
    name: tp.Optional[str] = None,
):
    """Prepares a model of submodels which is equivalent to a MobileNetV3 model.

    Parameters
    ----------
    input_shape : tuple, optional
        Shape tuple ``(height, width, 3)`` of the input image, with exactly 3 input channels. E.g.
        ``(160, 160, 3)`` is a valid value. If None, the model accepts images of any size of at
        least 32x32 (``(None, None, 3)``). There is no ``input_tensor`` argument.
    alpha : float, optional
        controls the width of the network. This is known as the depth multiplier in the MobileNetV3
        paper, but the name is kept for consistency with MobileNetV1 in Keras.
        - If `alpha` < 1.0, proportionally decreases the number
            of filters in each layer.
        - If `alpha` > 1.0, proportionally increases the number
            of filters in each layer.
        - If `alpha` = 1, default number of filters from the paper
            are used at each layer.

        Defaults to 1.0.
    model_type : {'Small', 'Large'}, optional
        whether it is the small variant or the large variant. Any value other than ``'Large'``
        is treated as ``'Small'``. Defaults to ``'Large'``.
    max_n_blocks : int, optional
        the maximum number of blocks in the backbone. It is further constrained by the actual
        maximum number of blocks that the variant can implement (5 for Large, 4 for Small).
        Defaults to 6.
    minimalistic : bool, optional
        In addition to large and small models this module also contains so-called minimalistic
        models, these models have the same per-layer dimensions characteristic as MobilenetV3
        however, they do not utilize any of the advanced blocks (squeeze-and-excite units,
        hard-swish, and 5x5 convolutions). While these models are less efficient on CPU, they
        are much more performant on GPU/DSP.
    mixer_params : mt.base.model.MobileNetV3MixerParams, optional
        parameters for defining the mixer block
    include_top : bool, optional
        whether to include the fully-connected layer at the top of the network. Only valid if
        `mixer_params` is not null. Defaults to True.
    pooling : str, optional
        Optional pooling mode for feature extraction when `include_top` is False and
        `mixer_params` is not null.
        - `None` means that the output of the model will be the 4D tensor output of the last
          convolutional block.
        - `avg` means that global average pooling will be applied to the output of the last
          convolutional block, and thus the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    classes : int, optional
        Optional number of classes to classify images into, only to be specified if `mixer_params`
        is not null and `include_top` is True.
    dropout_rate : float, optional
        fraction of the input units to drop on the last layer. Only to be specified if
        `mixer_params` is not null and `include_top` is True. Defaults to 0.2.
    classifier_activation : object, optional
        A `str` or callable. The activation function to use on the "top" layer. Ignored unless
        `mixer_params` is not null and `include_top` is True. Set `classifier_activation=None` to
        return the logits of the "top" layer. When loading pretrained weights,
        `classifier_activation` can only be `None` or `"softmax"`.
    output_all : bool, optional
        If True, the model returns the output tensor of every submodel (the parser included, the
        input layer excluded). Otherwise, it returns the output tensor of the last submodel.
        Either way, the outputs are given to Keras as a list. Defaults to False.
    name : str, optional
        model name, if any. Defaults to ``'MobilenetV3LargeSplit'`` or ``'MobilenetV3SmallSplit'``
        (note the lower-case "n").

    Returns
    -------
    keras.Model
        the output MobileNetV3 model split into submodels: the parser, up to 5 (Large) or 4
        (Small) blocks, and, if `mixer_params` is given, the mixer and (if it returns a model)
        the output block. Without `mixer_params` the model stops after the last backbone block.

    Raises
    ------
    mt.base.model.ModelSyntaxError
        if `mixer_params` is not None and not a ``MobileNetV3MixerParams``, or other errors
        from :func:`MobileNetV3Input`

    Examples
    --------
    Building a backbone with 3 blocks needs a Keras installation and is not executed here:

    .. code-block:: python

       from mt.keras.applications import MobileNetV3Split
       model = MobileNetV3Split((224, 224, 3), model_type="Small", max_n_blocks=3)
       model.summary()
    """

    input_layer = MobileNetV3Input(input_shape=input_shape)
    input_block = MobileNetV3Parser(
        input_layer,
        model_type=model_type,
        minimalistic=minimalistic,
    )
    x = input_block(input_layer)
    outputs = [x]

    num_blocks = 5 if model_type == "Large" else 4
    if num_blocks > max_n_blocks:
        num_blocks = max_n_blocks
    for i in range(num_blocks):
        if model_type == "Large":
            block = MobileNetV3LargeBlock(i, x, alpha=alpha, minimalistic=minimalistic)
        else:
            block = MobileNetV3SmallBlock(i, x, alpha=alpha, minimalistic=minimalistic)
        x = block(x)
        if output_all:
            outputs.append(x)
        else:
            outputs = [x]

    if mixer_params is not None:
        if not isinstance(mixer_params, base_model.MobileNetV3MixerParams):
            raise base_model.ModelSyntaxError(
                "Argument 'mixer_params' is not an instance of "
                f"mt.base.model.MobileNetV3MixerParams. Got: {type(mixer_params)}."
            )

        if model_type == "Large":
            last_point_ch = 1280
        else:
            last_point_ch = 1024
        mixer_block = MobileNetV3Mixer(
            x,
            mixer_params,
            last_point_ch,
            alpha=alpha,
            model_type=model_type,
            minimalistic=minimalistic,
        )
        x = mixer_block(x)
        if output_all:
            if isinstance(x, (list, tuple)):
                outputs.extend(x)
            else:
                outputs.append(x)
        else:
            if isinstance(x, (list, tuple)):
                outputs = [x[-1]]
            else:
                outputs = [x]

        output_block = MobileNetV3Output(
            x,
            model_type=model_type,
            include_top=include_top,
            classes=classes,
            pooling=pooling,
            dropout_rate=dropout_rate,
            classifier_activation=classifier_activation,
        )
        if output_block is not None:
            x = output_block(x)
            if output_all:
                outputs.append(x)
            else:
                outputs = [x]

    # Create model.
    if name is None:
        name = f"MobilenetV3{model_type}Split"
    model = models.Model(input_layer, outputs, name=name)

    return model
