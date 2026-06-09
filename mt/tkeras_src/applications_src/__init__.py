from .. import applications as _applications
from mt.base import logger

for _x, _y in _applications.__dict__.items():
    if _x.startswith("_"):
        continue
    globals()[_x] = _y
__doc__ = _applications.__doc__

from .classifier import create_classifier_block

try:
    from .mobilenet_v3_split import (
        MobileNetV3Input,
        MobileNetV3Parser,
        MobileNetV3SmallBlock,
        MobileNetV3LargeBlock,
        MobileNetV3Mixer,
        MobileNetV3Output,
        MobileNetV3Split,
    )
except Exception as e:
    logger.warn(
        f"Unable to import mt.tkeras_src.applications_src.mobilenet_v3_split: {e}"
    )

try:
    from .mobilevit import create_mobilevit
except Exception as e:
    logger.warn(f"Unable to import mt.tkeras_src.applications_src.mobilevit: {e}")


__api__ = [
    "MobileNetV3Input",
    "MobileNetV3Parser",
    "MobileNetV3SmallBlock",
    "MobileNetV3LargeBlock",
    "MobileNetV3Mixer",
    "MobileNetV3Output",
    "MobileNetV3Split",
    "create_mobilevit",
    "create_classifier_block",
]
