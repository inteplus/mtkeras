"""Keras layers plus MT layers.

Everything public from the selected Keras ``layers`` is re-exported together with the MT layers
listed in ``__api__`` (e.g. :class:`Identical`, :class:`Floor`, :class:`SoftBend`,
:class:`NormedConv2D`, :class:`MTTransformerEncoder`, the image resizing layers and the
:func:`conv2d` / :func:`dense2d` helpers).
"""

from .. import layers as _layers

for _x, _y in _layers.__dict__.items():
    if _x.startswith("_"):
        continue
    globals()[_x] = _y
__doc__ = _layers.__doc__

from .identical import *
from .floor import *
from .var_regularizer import *
from .simple_mha import *
from .image_sizing import *
from .counter import Counter
from .normed_conv2d import NormedConv2D
from .soft_bend import SoftBend
from .transformer_encoder import MTTransformerEncoder
from .utils import *


__api__ = [
    "Identical",
    "Floor",
    "VarianceRegularizer",
    "SimpleMHA2D",
    "MHAPool2D",
    "DUCLayer",
    "Downsize2D",
    "Upsize2D",
    "Downsize2D_V2",
    "Upsize2D_V2",
    "Downsize2D_V3",
    "Downsize2D_V4",
    "DownsizeX2D",
    "UpsizeX2D",
    "DownsizeY2D",
    "UpsizeY2D",
    "Counter",
    "conv2d",
    "dense2d",
    "SoftBend",
    "MTTransformerEncoder",
]
